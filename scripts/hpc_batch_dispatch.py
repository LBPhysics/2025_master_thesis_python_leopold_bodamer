"""Generate and (optionally) submit SLURM jobs for combined t_coh × inh    

# Estimate time: scale based on solver, n_atoms, len_coh_times
    solver = sim.simulation_config.ode_solver
    n_atoms = sim.system.n_atoms
    if solver == "ME":
        base_time_per_combo_seconds = 1.5  # for len_t=1000, 1 combo, 1 atom, 1 t_coh
    else:  # BR
        base_time_per_combo_seconds = 2.5  # for len_t=1000, 1 combo, 1 atom, 1 t_coh
    base_time_per_combo_seconds *= n_atoms ** 2  # quadratic scaling with n_atoms (matrix diagonalization)
    base_time_per_combo_seconds *= len_coh_times  # scaling with coherence time points

This dispatcher creates 1D/2D workflows by creating the full
Cartesian product of coherence times and inhomogeneous samples. It validates the
simulation locally before launching any jobs to an hpc cluster, ensuring that solver
instabilities are detected once and early.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

from matplotlib.pylab import f
import numpy as np

from calc_datas import _pick_config_yaml
from qspectro2d.config.create_sim_obj import load_simulation
from qspectro2d.spectroscopy import check_the_solver, sample_from_gaussian
from qspectro2d.utils.data_io import save_info_file
from qspectro2d.utils.file_naming import generate_unique_data_base

SCRIPTS_DIR = Path(__file__).parent.resolve()
for _parent in SCRIPTS_DIR.parents:
    if (_parent / ".git").is_dir():
        PROJECT_ROOT = _parent
        break

JOB_ROOT = SCRIPTS_DIR / "batch_jobs"
JOB_ROOT.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Combination:
    index: int
    t_index: int
    t_coh: float
    inhom_index: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "index": int(self.index),
            "t_index": int(self.t_index),
            "t_coh_value": float(self.t_coh),
            "inhom_index": int(self.inhom_index),
        }
    

def _set_random_seed(seed: int | None) -> None:
    if seed is not None:
        np.random.seed(seed)


def estimate_slurm_resources(sim, n_inhom: int, n_times: int, n_batches: int) -> tuple[str, str]:
    """Estimate SLURM memory and time requirements based on workload."""
    combos = n_times * n_inhom
    num_combos_per_batch = combos // n_batches
    workers = sim.simulation_config.max_workers # should be 16
    # Estimate memory: base 1G + factor for data size (complex64 = 8 bytes)
    len_t = len(sim.times_local) # NOTE actually it saves only a portion of this len: t_det up to time_cut
    mem_mb = 2000 + 10 * (workers * len_t * 8) / (1024**2)  # 10 is a safety factor
    requested_mem_mb = int(math.ceil(mem_mb))
    requested_mem = f"{requested_mem_mb}M"
    
    # Estimate time: scale based on solver, n_atoms, len_coh_times
    solver = sim.simulation_config.ode_solver
    n_atoms = sim.system.n_atoms
    base_time_per_combo_seconds = 1.5  # normalized from example: 1 combo in 3s for ME with 1 atom, 1 t_coh value for len_t = 1000
    if solver == "BR":
        base_time_per_combo_seconds = 2.5  # for len_t=1000, 1 combo, 1 atom, 1 t_coh
    base_time_per_combo_seconds *= n_atoms ** 2  # quadratic scaling with n_atoms (matrix diagonalization)
    base_time_per_combo_seconds *= len_t / 1000  # scaling with detection time length
    time_seconds = num_combos_per_batch * base_time_per_combo_seconds
    time_hours = max(0.1, time_seconds / 3600)  # minimum ~36 seconds for safety
    if time_hours < 1:
        minutes = int(time_hours * 60)
        requested_time = f"00:{minutes:02d}:00"
    else:
        days = int(time_hours) // 24
        hours = int(time_hours) % 24
        if days > 0:
            requested_time = f"{days}-{hours:02d}:00:00"
        else:
            requested_time = f"{hours:02d}:00:00"
    
    return requested_mem, requested_time


def _coherence_axis(sim, sim_type: str) -> np.ndarray:
    if sim_type == "1d":
        return np.array([float(sim.simulation_config.t_coh)], dtype=float)
    return np.asarray(sim.t_det, dtype=float)


def _build_combinations(t_coh_values: Sequence[float], n_inhom: int) -> list[Combination]:
    combos: list[Combination] = []
    index = 0
    for t_idx, t_coh in enumerate(t_coh_values):
        for inhom_idx in range(n_inhom):
            combos.append(
                Combination(
                    index=index,
                    t_index=t_idx,
                    t_coh=float(t_coh),
                    inhom_index=inhom_idx,
                )
            )
            index += 1
    return combos


def _split_indices(n_items: int, n_batches: int) -> list[np.ndarray]:
    if n_batches <= 0:
        raise ValueError("n_batches must be positive")
    if n_items == 0:
        return [np.array([], dtype=int) for _ in range(n_batches)]
    return [chunk.astype(int) for chunk in np.array_split(np.arange(n_items), n_batches)]


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


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
#SBATCH --partition=GPGPU

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
        choices=["1d", "2d"],
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

    config_path = _pick_config_yaml().resolve()

    print("=" * 80)
    print("GENERALIZED HPC DISPATCHER")
    print(f"Config path: {config_path}")

    sim = load_simulation(config_path, validate=True)
    print("✅ Simulation object constructed.")

    _, time_cut = check_the_solver(sim)
    print(f"✅ Solver validated. time_cut = {time_cut:.6g}")

    data_base_path = generate_unique_data_base(
        sim.system, sim.simulation_config, data_root=PROJECT_ROOT / "data"
    )

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

    t_coh_values = _coherence_axis(sim, args.sim_type)
    combinations = _build_combinations(t_coh_values, n_inhom)

    print(
        f"Prepared {len(combinations)} combination(s) → "
        f"|t_coh|={t_coh_values.size}, n_inhom={n_inhom}, n_batches={args.n_batches}"
    )

    # Estimate RAM and TIME based on batch size
    len_coh_times = len(t_coh_values)
    # Set sim to max t_coh for accurate time estimation
    sim.update_delays(t_coh=max(t_coh_values))
    requested_mem, requested_time = estimate_slurm_resources(sim, n_inhom, len_coh_times, args.n_batches)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    job_label = f"{args.sim_type}_{n_inhom}inh_{t_coh_values.size}t_{timestamp}"
    job_dir = JOB_ROOT / job_label
    job_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = job_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    samples_file = job_dir / "samples.npy"
    np.save(samples_file, samples.astype(float))

    metadata = {
        "sim_type": args.sim_type,
        "n_inhom": n_inhom,
        "n_t_coh": int(t_coh_values.size),
        "n_combinations": len(combinations),
        "n_batches": int(args.n_batches),
        "time_cut": float(time_cut),
        "job_label": job_label,
        "generated_at": timestamp,
        "data_base_path": str(data_base_path),
        "rng_seed": args.rng_seed,
    }
    _write_json(job_dir / "metadata.json", metadata)

    save_info_file(
        data_base_path.parent / f"{data_base_path.name}.pkl",
        sim.system,
        sim.simulation_config,
        bath=getattr(sim, "bath", None),
        laser=getattr(sim, "laser", None),
        extra_payload=metadata,
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
        _write_json(combos_file, {"combos": combos_subset})

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
