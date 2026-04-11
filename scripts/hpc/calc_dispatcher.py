"""Dispatch SLURM jobs for combined t_coh × inhomogeneity samples."""

from __future__ import annotations

import argparse
import math
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

from qspectro2d.utils.data_io import (
    allocate_job_dir,
    ensure_job_layout,
    save_info_file,
)

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common.workflow import (
    PROJECT_ROOT,
    RUNS_ROOT,
    build_job_metadata,
    build_job_dir_label,
    format_slurm_job_name,
    prepare_workflow,
    resolve_allocated_job_unique_id,
    write_json,
)
from qspectro2d.core.simulation.time_axes import compute_times_local

DEFAULT_CPUS_PER_TASK = 12  # makes sense if each phase combo is as costly as the others
DEFAULT_PARTITION = "GPGPU,metis"
REF_N_TIMES = 1000.0
REFERENCE_HILBERT_DIM = 2
TIME_SAFETY_FACTOR = 1.5
BATCH_STARTUP_S = 90.0
IO_PER_COMBO_S = 0.05
MINIMUM_BATCH_TIME_S = 300.0
MIB = 1024.0**2

@dataclass(frozen=True)
class BasisMetrics:
    hilbert_dim: int
    liouville_dim: int
    dense_superoperator_elements: int

    @property
    def dense_superoperator_mb(self) -> float:
        return self.dense_superoperator_elements * 16.0 / MIB


@dataclass(frozen=True)
class TimeProfile:
    ref_seconds_per_combo: float
    time_exponent: float
    liouville_exponent: float
    reference_hilbert_dim: int = REFERENCE_HILBERT_DIM


@dataclass(frozen=True)
class MemoryProfile:
    parent_base_mb: float
    worker_base_mb: float
    parent_dense_copies: float
    worker_dense_copies: float
    worker_history_copies: float
    accumulation_copies: float = 4.0


# These profiles are calibrated from completed strict-batch logs. The stored
# ``ref_seconds_per_combo`` values are normalized to a reference Hilbert space
# size and then scaled with the Liouville-space size, so the estimator can
# extrapolate beyond monomer/dimer-specific cases without solver-specific
# fallback branches.
PAPER_EQS_1D_PROFILE = TimeProfile(
    ref_seconds_per_combo=0.68,
    time_exponent=1.10,
    liouville_exponent=0.08,
)
PAPER_EQS_2D_PROFILE = TimeProfile(
    ref_seconds_per_combo=0.24,
    time_exponent=1.10,
    liouville_exponent=0.08,
)
LINDBLAD_RWA_1D_PROFILE = TimeProfile(
    ref_seconds_per_combo=2.03,
    time_exponent=1.10,
    liouville_exponent=0.55,
)
LINDBLAD_RWA_2D_PROFILE = TimeProfile(
    ref_seconds_per_combo=1.00,
    time_exponent=1.10,
    liouville_exponent=0.60,
)
LINDBLAD_NONRWA_PROFILE = TimeProfile(
    ref_seconds_per_combo=3.00,
    time_exponent=1.10,
    liouville_exponent=0.85,
)
REDFIELD_NONRWA_PROFILE = TimeProfile(
    ref_seconds_per_combo=25.93,
    time_exponent=1.10,
    liouville_exponent=1.25,
)
REDFIELD_RWA_PROFILE = TimeProfile(
    ref_seconds_per_combo=3.48,
    time_exponent=1.10,
    liouville_exponent=1.25,
)

TIME_PROFILES: dict[tuple[str, bool, str], TimeProfile] = {
    ("paper_eqs", True, "0d"): PAPER_EQS_1D_PROFILE,
    ("paper_eqs", True, "1d"): PAPER_EQS_1D_PROFILE,
    ("paper_eqs", True, "2d"): PAPER_EQS_2D_PROFILE,
    ("lindblad", True, "0d"): LINDBLAD_RWA_1D_PROFILE,
    ("lindblad", True, "1d"): LINDBLAD_RWA_1D_PROFILE,
    ("lindblad", True, "2d"): LINDBLAD_RWA_2D_PROFILE,
    ("lindblad", False, "0d"): LINDBLAD_NONRWA_PROFILE,
    ("lindblad", False, "1d"): LINDBLAD_NONRWA_PROFILE,
    ("lindblad", False, "2d"): LINDBLAD_NONRWA_PROFILE,
    ("redfield", False, "0d"): REDFIELD_NONRWA_PROFILE,
    ("redfield", False, "1d"): REDFIELD_NONRWA_PROFILE,
    ("redfield", False, "2d"): REDFIELD_NONRWA_PROFILE,
    ("redfield", True, "0d"): REDFIELD_RWA_PROFILE,
    ("redfield", True, "1d"): REDFIELD_RWA_PROFILE,
    ("redfield", True, "2d"): REDFIELD_RWA_PROFILE,
}

MEMORY_PROFILES: dict[str, MemoryProfile] = {
    "paper_eqs": MemoryProfile(
        parent_base_mb=192.0,
        worker_base_mb=96.0,
        parent_dense_copies=6.0,
        worker_dense_copies=2.0,
        worker_history_copies=24.0,
    ),
    "lindblad": MemoryProfile(
        parent_base_mb=192.0,
        worker_base_mb=40.0,
        parent_dense_copies=1.0,
        worker_dense_copies=0.5,
        worker_history_copies=6.0,
    ),
    "redfield": MemoryProfile(
        parent_base_mb=384.0,
        worker_base_mb=160.0,
        parent_dense_copies=6.0,
        worker_dense_copies=2.5,
        worker_history_copies=36.0,
    ),
}


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


def _format_mem_mb(requested_mb: float) -> str:
    rounded_mb = int(256 * math.ceil(max(1024.0, requested_mb) / 256.0))
    return f"{rounded_mb}M"


def _basis_metrics(hilbert_dim: int) -> BasisMetrics:
    hilbert_dim = max(1, int(hilbert_dim))
    liouville_dim = hilbert_dim * hilbert_dim
    return BasisMetrics(
        hilbert_dim=hilbert_dim,
        liouville_dim=liouville_dim,
        dense_superoperator_elements=liouville_dim * liouville_dim,
    )


def _time_profile(*, solver: str, rwa_sl: bool, sim_type: str) -> TimeProfile:
    specific_key = (str(solver), bool(rwa_sl), str(sim_type))
    try:
        return TIME_PROFILES[specific_key]
    except KeyError as exc:
        supported = ", ".join(str(key) for key in sorted(TIME_PROFILES))
        raise ValueError(
            "No calibrated time profile for "
            f"(solver={solver!r}, rwa_sl={bool(rwa_sl)!r}, sim_type={sim_type!r}). "
            f"Supported estimator profiles: {supported}"
        ) from exc


def _memory_profile(solver: str) -> MemoryProfile:
    try:
        return MEMORY_PROFILES[str(solver)]
    except KeyError as exc:
        supported = ", ".join(sorted(MEMORY_PROFILES))
        raise ValueError(
            f"No memory profile configured for solver={solver!r}. Supported: {supported}"
        ) from exc


def _basis_time_scale(metrics: BasisMetrics, profile: TimeProfile) -> float:
    reference_hilbert_dim = max(1, int(profile.reference_hilbert_dim))
    reference_liouville_dim = reference_hilbert_dim * reference_hilbert_dim
    return (metrics.liouville_dim / reference_liouville_dim) ** profile.liouville_exponent


def _estimate_requested_memory_mb(
    *,
    n_times: int,
    n_t_coh: int,
    workers: int,
    basis_dim: int,
    solver: str,
    sim_type: str,
    phase_cycling_jobs: int,
    signal_type_count: int,
) -> float:
    effective_workers = max(1, min(int(workers), int(phase_cycling_jobs)))
    signal_count = max(1, int(signal_type_count))
    n_t_det = max(1, int(n_times))
    n_t_coh = max(1, int(n_t_coh))
    metrics = _basis_metrics(basis_dim)
    profile = _memory_profile(solver)

    parent_python_mb = 384.0
    io_manifest_mb = 128.0

    output_grid_mb = signal_count * n_t_coh * n_t_det * 16.0 / MIB
    accumulation_mb = max(64.0, profile.accumulation_copies * output_grid_mb)
    worker_history_mb = n_t_det * signal_count * metrics.liouville_dim * 16.0 / MIB

    parent_solver_mb = (
        profile.parent_base_mb
        + profile.parent_dense_copies * metrics.dense_superoperator_mb
    )
    worker_solver_mb = (
        profile.worker_base_mb
        + profile.worker_dense_copies * metrics.dense_superoperator_mb
        + profile.worker_history_copies * worker_history_mb
    )

    estimated_mb = (
        parent_python_mb
        + io_manifest_mb
        + parent_solver_mb
        + accumulation_mb
        + effective_workers * worker_solver_mb
    )

    risky_dimer_floor_mb = 0.0
    if (
        str(sim_type) == "2d"
        and metrics.hilbert_dim >= 4
        and effective_workers >= 10
        and n_t_det >= 250
    ):
        if str(solver) == "paper_eqs":
            # Proteus validation on an intermediate dimer 2D paper_eqs run
            # showed the old 4096M blanket floor was much too high, but we
            # still keep a conservative floor for the heavier long-batch runs.
            risky_dimer_floor_mb = 2816.0
        elif str(solver) == "redfield":
            risky_dimer_floor_mb = 4096.0

    return max(estimated_mb, risky_dimer_floor_mb)


def estimate_slurm_resources(
    *,
    n_times: int,
    n_inhom: int,
    n_t_coh: int,
    n_batches: int,
    workers: int,
    basis_dim: int,
    solver: str,
    rwa_sl: bool,
    sim_type: str,
    phase_cycling_jobs: int,
    signal_type_count: int,
) -> tuple[str, str]:
    combos_total = n_inhom * n_t_coh
    combos_per_batch = max(1, math.ceil(combos_total / max(1, n_batches)))
    effective_workers = max(1, min(int(workers), int(phase_cycling_jobs)))
    metrics = _basis_metrics(basis_dim)

    requested_mem = _format_mem_mb(
        _estimate_requested_memory_mb(
            n_times=n_times,
            n_t_coh=n_t_coh,
            workers=effective_workers,
            basis_dim=basis_dim,
            solver=solver,
            sim_type=sim_type,
            phase_cycling_jobs=phase_cycling_jobs,
            signal_type_count=signal_type_count,
        )
    )

    effective_parallelism = effective_workers
    phase_rounds = math.ceil(phase_cycling_jobs / effective_parallelism)

    profile = _time_profile(
        solver=solver,
        rwa_sl=rwa_sl,
        sim_type=sim_type,
    )
    time_scale = (max(1, n_times) / REF_N_TIMES) ** profile.time_exponent
    basis_scale = _basis_time_scale(metrics, profile)
    seconds_per_combo = (
        profile.ref_seconds_per_combo * basis_scale * time_scale * phase_rounds
    )

    total_seconds = (
        BATCH_STARTUP_S
        + combos_per_batch * IO_PER_COMBO_S
        + combos_per_batch * seconds_per_combo * TIME_SAFETY_FACTOR
    )

    total_seconds = max(MINIMUM_BATCH_TIME_S, total_seconds)
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
    phase_pool_max_combos: int | None,
) -> str:
    python_cmd = shlex.quote(str(python_executable))
    worker_arg = shlex.quote(str(worker_path))
    phase_pool_exports = []
    if phase_pool_max_combos is not None:
        phase_pool_exports.append(
            f"export QSPECTRO_PHASE_POOL_MAX_COMBOS={int(phase_pool_max_combos)}"
        )
    phase_pool_exports_block = ""
    if phase_pool_exports:
        phase_pool_exports_block = "\n".join(phase_pool_exports) + "\n"

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
{phase_pool_exports_block}

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
        help="SLURM CPUs per batch job; this is also the phase-pool worker count",
    )
    parser.add_argument(
        "--phase_pool_max_combos",
        type=int,
        default=None,
        help="Optional pool recycling interval override to bound long-lived worker memory",
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
    if args.cpus_per_task <= 0:
        raise ValueError("--cpus_per_task must be positive")
    if args.phase_pool_max_combos is not None and args.phase_pool_max_combos <= 0:
        raise ValueError("--phase_pool_max_combos must be positive")

    cpus_per_task = int(args.cpus_per_task)
    phase_pool_max_combos = (
        int(args.phase_pool_max_combos) if args.phase_pool_max_combos is not None else None
    )

    print("=" * 80)
    print("HPC CALC DISPATCHER")

    prepared = prepare_workflow(
        config_path=args.config,
        sim_type=args.sim_type,
        rng_seed=args.rng_seed,
        max_workers=cpus_per_task,
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
        f"n_batches={args.n_batches}, "
        f"cpus_per_task={cpus_per_task}"
    )
    if phase_pool_max_combos is not None:
        print(
            f"Phase worker pools will be recycled every {phase_pool_max_combos} combination(s)",
        )

    timestamp = datetime.utcnow().strftime("%d_%H%M%S")
    base_label = build_job_dir_label(prepared.config_path, timestamp)
    job_dir = allocate_job_dir(RUNS_ROOT, base_label)
    job_paths = ensure_job_layout(job_dir, base_name="raw")
    job_unique_id = resolve_allocated_job_unique_id(
        job_dir,
        base_label=base_label,
        requested_unique_handle=timestamp,
    )

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
        workers=cpus_per_task,
        basis_dim=int(prepared.sim.system.dimension),
        solver=str(prepared.sim.simulation_config.ode_solver),
        rwa_sl=bool(prepared.sim.simulation_config.rwa_sl),
        sim_type=prepared.sim_type,
        phase_cycling_jobs=phase_cycling_jobs,
        signal_type_count=len(prepared.sim.simulation_config.signal_types),
    )
    print(
        f"Requested resources: mem={requested_mem}, "
        f"time={requested_time}, cpus={cpus_per_task}"
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
    job_metadata["n_batches"] = int(args.n_batches)
    job_metadata["job_unique_id"] = job_unique_id
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
                cpus_per_task=cpus_per_task,
                requested_mem=requested_mem,
                requested_time=requested_time,
                partition=args.partition,
                phase_pool_max_combos=phase_pool_max_combos,
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
