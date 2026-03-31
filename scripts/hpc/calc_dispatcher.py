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
from typing import Sequence

import numpy as np
import yaml

from qspectro2d.utils.data_io import (
    allocate_job_dir,
    ensure_job_layout,
    job_label_token,
    save_info_file,
)

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common.workflow import (
    PROJECT_ROOT,
    RUNS_ROOT,
    build_job_metadata,
    extract_job_unique_id,
    format_slurm_job_name,
    prepare_workflow,
    write_json,
)
from qspectro2d.core.simulation.time_axes import compute_times_local

DEFAULT_CPUS_PER_TASK = 25  # makes sense if each phase combo is as costly as the others
DEFAULT_PARTITION = "GPGPU,metis"


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
    combos_total = n_inhom * n_t_coh
    combos_per_batch = max(1, math.ceil(combos_total / max(1, n_batches)))

    base_mem_mb = 1500.0
    bytes_per_worker = max(1, n_times) * max(1, n_dim) * 16.0 * 200.0
    workers_mem_mb = workers * bytes_per_worker / (1024.0**2)
    combos_mem_mb = min(1000.0, 3.0 * combos_per_batch)
    requested_mem = f"{int(math.ceil(base_mem_mb + workers_mem_mb + combos_mem_mb))}M"

    ref_n_times = 1000.0
    ref_n_dim = 2.0

    ref_seconds_per_combo = {
        ("paper_eqs", True): 0.2,
        ("lindblad", True): 0.4,
        ("lindblad", False): 3.0,
        ("redfield", True): 15.0,
        ("redfield", False): 60.0,
    }

    key = (solver, bool(rwa_sl))

    effective_parallelism = max(1, min(workers, phase_cycling_jobs))
    phase_rounds = math.ceil(phase_cycling_jobs / effective_parallelism)

    time_scale = (max(1, n_times) / ref_n_times) ** 1.10
    atomic_dim_scale = (max(1, n_dim) / ref_n_dim) ** 2.0

    seconds_per_combo = ref_seconds_per_combo[key] * time_scale * atomic_dim_scale * phase_rounds

    batch_startup_s = 90.0
    io_per_combo_s = 0.05
    safety_factor = 2.5

    total_seconds = (
        batch_startup_s
        + combos_per_batch * io_per_combo_s
        + combos_per_batch * seconds_per_combo * safety_factor
    )

    minimum_batch_time_s = 300.0
    total_seconds = max(minimum_batch_time_s, total_seconds)
    requested_time = _format_hms(total_seconds)
    return requested_mem, requested_time


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


def submit_sbatch(script_path: Path, *, dependency: str | None = None) -> str:
    sbatch = shutil.which("sbatch")
    if sbatch is None:
        raise RuntimeError("sbatch not found on PATH. Run this on your cluster login node.")
    cmd = [sbatch]
    if dependency is not None:
        cmd.append(f"--dependency={dependency}")
    cmd.append(str(script_path))

    result = subprocess.run(
        cmd,
        cwd=str(script_path.parent),
        capture_output=True,
        text=True,
        check=True,
    )
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
    parser.add_argument("--n_batches", type=int, default=1, help="Number of SLURM batch jobs")
    parser.add_argument(
        "--rng_seed",
        type=int,
        default=None,
        help="Optional NumPy seed for reproducible inhomogeneous sampling",
    )
    parser.add_argument(
        "--time_cut",
        type=float,
        default=None,
        help="Optional override for the safe evolution cutoff",
    )
    parser.add_argument(
        "--cpus_per_task",
        type=int,
        default=DEFAULT_CPUS_PER_TASK,
        help="SLURM CPUs per primary batch job",
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

    print("=" * 80)
    print("HPC CALC DISPATCHER")

    prepared = prepare_workflow(
        config_path=args.config,
        sim_type=args.sim_type,
        rng_seed=args.rng_seed,
        max_workers=args.cpus_per_task,
        run_solver_check=False,
    )

    print(f"Config path: {prepared.config_path}")
    print("✅ Merged config resolved, validated, and simulation object constructed.")
    print(f"✅ Solver validated once. time_cut = {prepared.time_cut:.6g}")

    time_cut = float(prepared.time_cut if args.time_cut is None else args.time_cut)
    print(
        f"Prepared {len(prepared.combinations)} combination(s) -> "
        f"|t_coh|={prepared.t_coh_values.size}, "
        f"n_inhom={int(prepared.sim.simulation_config.n_inhomogen)}, "
        f"n_batches={args.n_batches}"
    )

    label_token = job_label_token(
        prepared.sim.simulation_config,
        prepared.sim.system,
        sim_type=prepared.sim_type,
    )
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    job_dir = allocate_job_dir(RUNS_ROOT, f"hpc_{label_token}_{timestamp}")
    job_paths = ensure_job_layout(job_dir, base_name="raw")
    job_unique_id = extract_job_unique_id(job_dir)

    logs_dir = job_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    config_copy_path = job_dir / prepared.config_path.name
    with config_copy_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(prepared.merged_cfg, handle, sort_keys=False)

    samples_path = job_dir / "samples.npy"
    np.save(samples_path, prepared.samples.astype(float))

    n_phases = int(prepared.sim.simulation_config.n_phases)
    phase_cycling_jobs = _phase_cycling_job_count(n_phases)

    coh_vals = prepared.t_coh_values
    n_t_coh = len(coh_vals)
    last_t_coh = coh_vals[-1]
    global_times = compute_times_local(prepared.sim.simulation_config, t_coh_override=last_t_coh)
    requested_mem, requested_time = estimate_slurm_resources(
        n_times=len(global_times),
        n_inhom=int(prepared.sim.simulation_config.n_inhomogen),
        n_t_coh=n_t_coh,
        n_batches=args.n_batches,
        workers=int(args.cpus_per_task),
        n_dim=int(prepared.sim.system.dimension),
        solver=str(prepared.sim.simulation_config.ode_solver),
        rwa_sl=bool(prepared.sim.simulation_config.rwa_sl),
        phase_cycling_jobs=phase_cycling_jobs,
    )
    print(
        f"Requested resources: mem={requested_mem}, "
        f"time={requested_time}, cpus={args.cpus_per_task}"
    )

    job_metadata = build_job_metadata(
        prepared,
        job_dir=job_paths.job_dir,
        data_dir=job_paths.data_dir,
        figures_dir=job_paths.figures_dir,
        data_base_name=job_paths.base_name,
        data_base_path=job_paths.data_base_path,
        config_path=config_copy_path,
        time_cut=time_cut,
    )
    write_json(job_dir / "job_metadata.json", job_metadata)

    info_path = job_paths.data_base_path.parent / f"{job_paths.data_base_path.name}.pkl"
    save_info_file(
        info_path,
        prepared.sim.system,
        prepared.sim.simulation_config,
        bath=getattr(prepared.sim, "bath", None),
        laser=getattr(prepared.sim, "laser", None),
        extra_payload=job_metadata,
    )

    worker_path = (SCRIPTS_DIR / "hpc" / "run_batch.py").resolve()
    if not worker_path.exists():
        raise FileNotFoundError(f"Missing worker script: {worker_path}")

    python_executable = Path(sys.executable).resolve()
    batch_indices = _split_indices(len(prepared.combinations), args.n_batches)
    script_paths: list[Path] = []

    for batch_idx, indices in enumerate(batch_indices):
        combos_subset = [prepared.combinations[i].to_dict() for i in indices.tolist()]
        combos_path = job_dir / f"batch_{batch_idx:03d}.json"
        write_json(combos_path, {"combos": combos_subset})

        job_name = format_slurm_job_name(
            "calc",
            prepared.sim_type,
            f"b{batch_idx:02d}of{args.n_batches:02d}",
            job_unique_id,
        )
        script_path = job_dir / f"{job_name}.slurm"
        script_path.write_text(
            _render_slurm_script(
                job_name=job_name,
                log_dir=logs_dir.resolve(),
                worker_path=worker_path,
                python_executable=python_executable,
                combos_path=combos_path.resolve(),
                samples_path=samples_path.resolve(),
                time_cut=time_cut,
                sim_type=prepared.sim_type,
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

    print("AFTERWORDS \n🎯 Plot with:")
    plot_script = (PROJECT_ROOT / "scripts" / "hpc" / "plot_dispatcher.py").resolve()
    print(f"python {plot_script} --job_dir {job_dir}")

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
