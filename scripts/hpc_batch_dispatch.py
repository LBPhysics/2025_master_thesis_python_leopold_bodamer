"""Generate and (optionally) submit SLURM jobs for combined t_coh × inh

this dispatcher creates 1D/2D workflows by creating the full
Cartesian product of coherence times and inhomogeneous samples. It validates the
simulation locally before launching any jobs to an hpc cluster.
"""

from __future__ import annotations

import argparse
import math
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence
import numpy as np

from qspectro2d.config.create_sim_obj import load_simulation
from qspectro2d.spectroscopy import check_the_solver, sample_from_gaussian
from qspectro2d.utils.data_io import save_info_file
from qspectro2d.utils.job_paths import allocate_job_dir, ensure_job_layout, job_label_token
from qspectro2d.core.simulation.time_axes import (
    compute_times_global,
    compute_t_coh,
    compute_t_det,
)

from calc_datas import (
    RUNS_ROOT,
    SCRIPTS_DIR,
    pick_config_yaml,
    build_combinations,
    write_json,
)


def _set_random_seed(seed: int | None) -> None:
    if seed is not None:
        np.random.seed(seed)


def estimate_slurm_resources(
    n_times: int,  # number of global times -> how many time steps
    n_inhom: int,
    n_t_coh: int,  # number of coherence times -> how many combinations
    n_batches: int,
    *,
    workers: int = 1,
    N_dim: int,
    solver: str = "ME",
    mem_safety: float = 100.0,
    base_mb: int = 500,
    time_safety: float = 10,
    base_time: float = 300.0,
    rwa_sl: bool = True,
) -> tuple[str, str]:
    """
    Estimate SLURM memory and runtime for QuTiP mesolve evolutions.
    """
    # ---------------------- MEMORY ----------------------
    bytes_per_solver = (
        n_times * (N_dim) * 16
    )  # doesnt scale quadratically because i dont store states
    total_bytes = mem_safety * workers * bytes_per_solver
    mem_mb = base_mb + total_bytes / (1024**2)
    requested_mem = f"{int(math.ceil(mem_mb))}M"

    # ---------------------- TIME ------------------------
    # Number of total independent simulations
    combos_total = n_inhom * n_t_coh
    combos_per_batch = max(1, combos_total // max(1, n_batches))

    # Empirical baseline: base_time s per combo for ME, 1 atom, n_times=1000, N=2
    t0 = 0.03  # basic case for pessimistic ME
    if solver == "Paper_eqs" or solver == "BR":
        t0 *= 5.0  # slower solver
    if not rwa_sl:
        t0 *= 5.0  # non-RWA is WAY slower
    base_t = t0

    # scaling ~ n_times * N^2  (sparse regime)
    time_per_combo = base_t * (n_t_coh / 1000) * ((N_dim) ** 2)

    # total time for one batch (divide by workers)
    total_seconds = time_per_combo * combos_per_batch * time_safety

    # Ensure minimum time of 1 minute to avoid SLURM rejection
    total_seconds = max(total_seconds, base_time)

    # convert to HH:MM:SS, clip to max 24h if needed
    h = int(total_seconds // 3600)
    m = int((total_seconds % 3600) // 60)
    s = int(total_seconds % 60)
    # Cap at 3 days (72 hours) to fit GPGPU partition limit
    if h >= 72:
        h, m, s = 72, 0, 0
    requested_time = f"{h:02d}:{m:02d}:{s:02d}"

    return requested_mem, requested_time


def _split_indices(n_items: int, n_batches: int) -> list[np.ndarray]:
    if n_batches <= 0:
        raise ValueError("n_batches must be positive")
    if n_items == 0:
        return [np.array([], dtype=int) for _ in range(n_batches)]
    return [chunk.astype(int) for chunk in np.array_split(np.arange(n_items), n_batches)]


def _render_slurm_script(
    *,
    job_name: str,
    batch_idx: int,
    n_batches: int,
    sim_type: str,
    combos_filename: str,
    samples_filename: str,
    time_cut: float,
    worker_path: Path,
    python_executable: Path,
    requested_mem: str,
    requested_time: str,
) -> str:
    python_cmd = shlex.quote(str(python_executable))
    worker_arg = shlex.quote(str(worker_path))
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --cpus-per-task=16
#SBATCH --mem={requested_mem}
#SBATCH --time={requested_time}
#SBATCH --partition=GPGPU,metis

set -euo pipefail

{python_cmd} {worker_arg} \
    --combos_file "{combos_filename}" \
    --samples_file "{samples_filename}" \
    --time_cut {time_cut:.12g} \
    --sim_type {sim_type} \
    --batch_id {batch_idx} \
    --n_batches {n_batches}
"""


def submit_sbatch(script_path: Path, cwd: Path | None = None) -> str:
    """Submit a SLURM script via sbatch."""
    sbatch = shutil.which("sbatch")
    if sbatch is None:
        raise RuntimeError("sbatch not found on PATH. Run this on your cluster login node.")

    if cwd is None:
        cwd = script_path.parent

    try:
        result = subprocess.run(
            [sbatch, str(script_path)],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        raise RuntimeError("sbatch command not found. Submit the script manually.")
    except subprocess.CalledProcessError as exc:
        msg = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise RuntimeError(f"sbatch failed with exit code {exc.returncode}: {msg}")

    return result.stdout.strip()


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dispatch generalized spectroscopy batches to an HPC cluster",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sim_type",
        choices=["0d", "1d", "2d"],
        default="2d",
        help="Simulation dimensionality",
    )
    parser.add_argument(
        "--n_batches",
        type=int,
        default=1,
        help="Total number of batches to split the combination space into",
    )
    parser.add_argument(
        "--rng_seed",
        type=int,
        default=None,
        help="Optional NumPy random seed for reproducible sampling",
    )
    parser.add_argument(
        "--no_submit",
        action="store_true",
        help="Only generate local artifacts; skip sbatch submission",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    config_path = pick_config_yaml().resolve()

    print("=" * 80)
    print("GENERALIZED HPC DISPATCHER")
    print(f"Config path: {config_path}")

    sim = load_simulation(config_path, validate=True)
    print("✅ Simulation object constructed.")

    time_cut = np.inf  # TODO ONLY CHECK LOCALLY check_the_solver(sim)
    print(f"✅ Solver validated. time_cut = {time_cut:.6g}")

    sim.simulation_config.sim_type = args.sim_type  # to ensure t_coh_axis has the right behavior

    n_inhom = sim.simulation_config.n_inhomogen
    if n_inhom <= 0:
        raise ValueError("n_inhom must be positive")

    _set_random_seed(args.rng_seed)
    base_freqs = np.asarray(sim.system.frequencies_cm, dtype=float)
    delta_cm = float(sim.system.delta_inhomogen_cm)
    samples = sample_from_gaussian(
        n_samples=n_inhom,
        fwhm=delta_cm,
        mu=base_freqs,
    )

    # Use SimulationModuleOQS.t_coh for coherence axis (handles 0d/1d/2d)
    t_coh_values = np.asarray(compute_t_coh(sim.simulation_config), dtype=float)
    times_global = np.asarray(compute_times_global(sim.simulation_config), dtype=float)

    combinations = build_combinations(t_coh_values, n_inhom)

    print(
        f"Prepared {len(combinations)} combination(s) → "
        f"|t_coh|={t_coh_values.size}, n_inhom={n_inhom}, n_batches={args.n_batches}"
    )

    # Estimate RAM and TIME based on batch size
    requested_mem, requested_time = estimate_slurm_resources(
        n_times=len(times_global),
        n_inhom=n_inhom,
        n_t_coh=len(t_coh_values),
        n_batches=args.n_batches,
        workers=16,  # BECAUSE I set every batch to use 16 CPUs
        N_dim=sim.system.dimension,
        solver=sim.simulation_config.ode_solver,
        rwa_sl=sim.simulation_config.rwa_sl,
    )

    label_token = job_label_token(sim.simulation_config, sim.system, sim_type=args.sim_type)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    job_label = f"hpc_{label_token}_{timestamp}"
    job_dir = allocate_job_dir(RUNS_ROOT, job_label)
    job_paths = ensure_job_layout(job_dir, base_name="raw")
    data_base_path = job_paths.data_base_path
    logs_dir = job_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Save a copy of the config file to the job directory
    config_copy_path = job_dir / config_path.name
    if not config_copy_path.exists():
        shutil.copy2(config_path, config_copy_path)
        print(f"✅ Config file copied to {config_copy_path}")

    # Use the copied config path for subsequent operations
    config_path = config_copy_path

    samples_file = job_dir / "samples.npy"
    np.save(samples_file, samples.astype(float))

    job_metadata = {
        "sim_type": args.sim_type,
        "signal_types": sim.simulation_config.signal_types,
        "t_det": compute_t_det(sim.simulation_config).tolist(),
        "t_coh": t_coh_values.tolist(),
        "n_inhom": n_inhom,
        "n_t_coh": int(t_coh_values.size),
        "n_combinations": len(combinations),
        "n_batches": int(args.n_batches),
        "time_cut": float(time_cut),
        "job_label": job_label,
        "job_token": label_token,
        "generated_at": timestamp,
        "job_dir": str(job_paths.job_dir),
        "data_dir": str(job_paths.data_dir),
        "figures_dir": str(job_paths.figures_dir),
        "data_base_name": job_paths.base_name,
        "data_base_path": str(data_base_path),
        "config_path": str(config_path),
        "rng_seed": args.rng_seed,
    }
    write_json(job_dir / "job_metadata.json", job_metadata)

    info_path = data_base_path.parent / f"{data_base_path.name}.pkl"
    if not info_path.exists():
        save_info_file(
            info_path,
            sim.system,
            sim.simulation_config,
            bath=getattr(sim, "bath", None),
            laser=getattr(sim, "laser", None),
            extra_payload=job_metadata,
        )

    batch_indices = _split_indices(len(combinations), args.n_batches)
    script_paths: list[Path] = []
    worker_path = (SCRIPTS_DIR / "run_batch.py").resolve()
    if not worker_path.exists():
        raise FileNotFoundError(f"Missing worker script: {worker_path}")

    if sys.executable:
        python_executable = Path(sys.executable).resolve()
    else:
        candidate = shutil.which("python") or shutil.which("python3")
        if candidate is None:
            raise RuntimeError("Unable to determine python executable for SLURM script")
        python_executable = Path(candidate).resolve()
    for batch_idx, indices in enumerate(batch_indices):
        combos_subset = [combinations[i].to_dict() for i in indices.tolist()]
        combos_file = job_dir / f"batch_{batch_idx:03d}.json"
        write_json(combos_file, {"combos": combos_subset})

        job_name = f"{args.sim_type}b{batch_idx:02d}of{args.n_batches:02d}"
        slurm_text = _render_slurm_script(
            job_name=job_name,
            batch_idx=batch_idx,
            n_batches=args.n_batches,
            sim_type=args.sim_type,
            combos_filename=combos_file.name,
            samples_filename=samples_file.name,
            time_cut=time_cut,
            worker_path=worker_path,
            python_executable=python_executable,
            requested_mem=requested_mem,
            requested_time=requested_time,
        )
        script_path = job_dir / f"{job_name}.slurm"
        script_path.write_text(slurm_text, encoding="utf-8")
        script_paths.append(script_path)
        print(f"  batch {batch_idx}: {len(combos_subset)} combo(s) → {script_path.name}")

    print(f"Artifacts written to {job_dir}")

    if args.no_submit:
        print("Skipping submission (--no_submit set).")
        return

    print("Submitting SLURM jobs...")
    for script_path in script_paths:
        submit_msg = submit_sbatch(script_path)
        print(f"  {script_path.name}: {submit_msg}")

    print("Done.")


if __name__ == "__main__":
    main()
