"""Dispatch SLURM jobs for combined t_coh × inhomogeneity samples."""

from __future__ import annotations

import argparse
import math
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from qspectro2d.config import resolve_config, validate_config
from qspectro2d.config.factory import load_simulation
from qspectro2d.core.simulation.time_axes import (
    compute_t_coh,
    compute_t_det,
    compute_times_local,
)
from qspectro2d.spectroscopy import sample_from_gaussian
from qspectro2d.utils.data_io import (
    allocate_job_dir,
    ensure_job_layout,
    job_label_token,
    save_info_file,
)

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from local.calc_datas import RUNS_ROOT, build_combinations, pick_config_yaml, write_json

DEFAULT_CPUS_PER_TASK = 25
DEFAULT_PARTITION = "GPGPU,metis"


def _set_random_seed(seed: int | None) -> None:
    if seed is not None:
        np.random.seed(seed)


def _split_indices(n_items: int, n_batches: int) -> list[np.ndarray]:
    if n_batches <= 0:
        raise ValueError("n_batches must be positive")

    if n_items == 0:
        return [np.array([], dtype=int) for _ in range(n_batches)]

    return [chunk.astype(int) for chunk in np.array_split(np.arange(n_items), n_batches)]


def _phase_cycling_job_count(n_phases: int) -> int:
    if n_phases <= 0:
        raise ValueError("n_phases must be positive")
    return n_phases * n_phases + 2 * n_phases + 1


def _format_hms(total_seconds: float) -> str:
    total_seconds = int(max(0, math.ceil(total_seconds)))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    if hours >= 72:
        return "72:00:00"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def estimate_slurm_resources(
    *,
    n_times: int,
    n_inhom: int,
    n_t_coh: int,
    n_batches: int,
    workers: int,
    n_dim: int,
    solver: str,
    rwa_sl: bool,
    phase_cycling_jobs: int,
) -> tuple[str, str]:
    """
    Rough SLURM resource estimate.

    Model:
    - combinations are processed serially inside one batch job
    - each combination launches phase-cycling worker tasks in parallel
    - effective parallelism is limited by min(workers, phase_cycling_jobs)
    """
    combos_total = n_inhom * n_t_coh
    combos_per_batch = max(1, math.ceil(combos_total / max(1, n_batches)))

    # ---------------------- memory ----------------------
    # Conservative but simple.
    base_mem_mb = 1500.0
    bytes_per_worker = max(1, n_times) * max(1, n_dim) * 16.0 * 100.0
    workers_mem_mb = workers * bytes_per_worker / (1024.0**2)
    combos_mem_mb = min(1000.0, 2.0 * combos_per_batch)

    requested_mem = f"{int(math.ceil(base_mem_mb + workers_mem_mb + combos_mem_mb))}M"

    # ----------------------- time -----------------------
    # Base runtime for one single worker solve at modest size.
    base_seconds_per_solve = 0.45

    solver_factor = {
        "paper_eqs": 1.0,
        "lindblad": 1.5,
        "redfield": 4.0,
    }
    if solver not in solver_factor:
        raise ValueError(f"Unsupported solver '{solver}'.")

    rwa_factor = 1.0 if rwa_sl else 3.0
    effective_parallelism = max(1, min(workers, phase_cycling_jobs))
    phase_rounds = math.ceil(phase_cycling_jobs / effective_parallelism)

    seconds_per_combo = (
        base_seconds_per_solve
        * solver_factor[solver]
        * rwa_factor
        * (max(1, n_times) / 1000.0)
        * (max(1, n_dim) ** 2)
        * phase_rounds
    )

    safety_factor = 2.5
    minimum_batch_time_s = 600.0
    total_seconds = max(minimum_batch_time_s, seconds_per_combo * combos_per_batch * safety_factor)

    return requested_mem, _format_hms(total_seconds)


def _render_slurm_script(
    *,
    job_name: str,
    log_dir: Path,
    worker_path: Path,
    python_executable: Path,
    combos_path: Path,
    samples_path: Path,
    time_cut: float,
    sim_type: str,
    batch_idx: int,
    n_batches: int,
    cpus_per_task: int,
    requested_mem: str,
    requested_time: str,
    partition: str,
) -> str:
    python_cmd = shlex.quote(str(python_executable))
    worker_arg = shlex.quote(str(worker_path))

    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/%x.out
#SBATCH --error={log_dir}/%x.err
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={requested_mem}
#SBATCH --time={requested_time}
#SBATCH --partition={partition}

set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

{python_cmd} -u {worker_arg} \\
  --combos_file "{combos_path}" \\
  --samples_file "{samples_path}" \\
  --time_cut {time_cut:.12g} \\
  --sim_type {sim_type} \\
  --batch_id {batch_idx} \\
  --n_batches {n_batches}
"""


def submit_sbatch(script_path: Path) -> str:
    sbatch = shutil.which("sbatch")
    if sbatch is None:
        raise RuntimeError("sbatch not found on PATH. Run this on your cluster login node.")

    try:
        result = subprocess.run(
            [sbatch, str(script_path)],
            cwd=str(script_path.parent),
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise RuntimeError(f"sbatch failed with exit code {exc.returncode}: {message}") from exc

    return result.stdout.strip()


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dispatch spectroscopy batches to an HPC cluster",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path")
    parser.add_argument(
        "--sim_type",
        choices=["0d", "1d", "2d"],
        default=None,
        help="Simulation dimensionality override",
    )
    parser.add_argument(
        "--n_batches",
        type=int,
        default=1,
        help="Number of SLURM batch jobs",
    )
    parser.add_argument(
        "--rng_seed",
        type=int,
        default=None,
        help="Optional NumPy seed for reproducible inhomogeneous sampling",
    )
    parser.add_argument(
        "--time_cut",
        type=float,
        default=float("inf"),
        help="Safe evolution cutoff forwarded to run_batch.py",
    )
    parser.add_argument(
        "--cpus_per_task",
        type=int,
        default=DEFAULT_CPUS_PER_TASK,
        help="SLURM CPUs per batch job",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default=DEFAULT_PARTITION,
        help="SLURM partition(s)",
    )
    parser.add_argument(
        "--no_submit",
        action="store_true",
        help="Only generate job artifacts and SLURM scripts",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    if args.config:
        config_path = Path(args.config).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
    else:
        config_path = pick_config_yaml().resolve()

    print("=" * 80)
    print("HPC CALC DISPATCHER")
    print(f"Config path: {config_path}")

    merged_cfg = resolve_config(str(config_path))
    if args.sim_type is not None:
        merged_cfg["config"]["sim_type"] = args.sim_type

    # Keep worker count consistent with SLURM allocation.
    merged_cfg["config"]["max_workers"] = int(args.cpus_per_task)
    validate_config(merged_cfg)

    sim = load_simulation(merged_cfg)
    effective_sim_type = sim.simulation_config.sim_type
    print("✅ Merged config resolved and simulation object constructed.")

    n_inhom = int(sim.simulation_config.n_inhomogen)
    if n_inhom <= 0:
        raise ValueError("n_inhom must be positive")

    _set_random_seed(args.rng_seed)

    samples = sample_from_gaussian(
        n_samples=n_inhom,
        fwhm=float(sim.system.delta_inhomogen_cm),
        mu=np.asarray(sim.system.frequencies_cm, dtype=float),
    )

    t_coh_values = np.asarray(compute_t_coh(sim.simulation_config), dtype=float)
    t_det_values = np.asarray(compute_t_det(sim.simulation_config), dtype=float)
    times_local = np.asarray(compute_times_local(sim.simulation_config), dtype=float)
    combinations = build_combinations(t_coh_values, n_inhom)

    print(
        f"Prepared {len(combinations)} combination(s) -> "
        f"|t_coh|={t_coh_values.size}, n_inhom={n_inhom}, n_batches={args.n_batches}"
    )

    label_token = job_label_token(sim.simulation_config, sim.system, sim_type=effective_sim_type)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    job_dir = allocate_job_dir(RUNS_ROOT, f"hpc_{label_token}_{timestamp}")
    job_paths = ensure_job_layout(job_dir, base_name="raw")

    logs_dir = job_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    config_copy_path = job_dir / config_path.name
    with config_copy_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(merged_cfg, handle, sort_keys=False)

    samples_path = job_dir / "samples.npy"
    np.save(samples_path, samples.astype(float))

    n_phases = int(sim.simulation_config.n_phases)
    phase_cycling_jobs = _phase_cycling_job_count(n_phases)

    requested_mem, requested_time = estimate_slurm_resources(
        n_times=len(times_local),
        n_inhom=n_inhom,
        n_t_coh=len(t_coh_values),
        n_batches=args.n_batches,
        workers=int(args.cpus_per_task),
        n_dim=int(sim.system.dimension),
        solver=str(sim.simulation_config.ode_solver),
        rwa_sl=bool(sim.simulation_config.rwa_sl),
        phase_cycling_jobs=phase_cycling_jobs,
    )
    print(
        f"Requested resources: mem={requested_mem}, "
        f"time={requested_time}, cpus={args.cpus_per_task}"
    )

    job_metadata = {
        "sim_type": effective_sim_type,
        "signal_types": sim.simulation_config.signal_types,
        "t_det": t_det_values.tolist(),
        "t_coh": t_coh_values.tolist(),
        "n_inhom": n_inhom,
        "n_t_coh": int(t_coh_values.size),
        "job_dir": str(job_paths.job_dir),
        "data_dir": str(job_paths.data_dir),
        "figures_dir": str(job_paths.figures_dir),
        "data_base_name": job_paths.base_name,
        "data_base_path": str(job_paths.data_base_path),
        "config_path": str(config_copy_path),
        "merged_config": merged_cfg,
    }
    write_json(job_dir / "job_metadata.json", job_metadata)

    info_path = job_paths.data_base_path.parent / f"{job_paths.data_base_path.name}.pkl"
    if not info_path.exists():
        save_info_file(
            info_path,
            sim.system,
            sim.simulation_config,
            bath=getattr(sim, "bath", None),
            laser=getattr(sim, "laser", None),
            extra_payload=job_metadata,
        )

    worker_path = (SCRIPTS_DIR / "hpc" / "run_batch.py").resolve()
    if not worker_path.exists():
        raise FileNotFoundError(f"Missing worker script: {worker_path}")

    python_executable = Path(sys.executable).resolve()
    batch_indices = _split_indices(len(combinations), args.n_batches)
    script_paths: list[Path] = []

    for batch_idx, indices in enumerate(batch_indices):
        combos_subset = [combinations[i].to_dict() for i in indices.tolist()]
        combos_path = job_dir / f"batch_{batch_idx:03d}.json"
        write_json(combos_path, {"combos": combos_subset})

        job_name = f"{effective_sim_type}b{batch_idx:02d}of{args.n_batches:02d}"
        script_path = job_dir / f"{job_name}.slurm"
        script_path.write_text(
            _render_slurm_script(
                job_name=job_name,
                log_dir=logs_dir.resolve(),
                worker_path=worker_path,
                python_executable=python_executable,
                combos_path=combos_path.resolve(),
                samples_path=samples_path.resolve(),
                time_cut=float(args.time_cut),
                sim_type=effective_sim_type,
                batch_idx=batch_idx,
                n_batches=args.n_batches,
                cpus_per_task=int(args.cpus_per_task),
                requested_mem=requested_mem,
                requested_time=requested_time,
                partition=args.partition,
            ),
            encoding="utf-8",
        )
        script_paths.append(script_path)
        print(f"  batch {batch_idx}: {len(combos_subset)} combo(s) -> {script_path.name}")

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
