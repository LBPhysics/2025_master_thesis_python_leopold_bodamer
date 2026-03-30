"""Queue processing + plotting jobs on HPC for one completed spectroscopy run."""

from __future__ import annotations

import argparse
import json
import math
import sys
from functools import partial
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common.plot_settings import PAD_FACTOR
from common.workflow import final_processed_filename
from hpc.calc_dispatcher import submit_sbatch

PLOT_SCRIPT = (SCRIPTS_DIR / "local" / "plot_datas.py").resolve()
PROCESS_SCRIPT = (SCRIPTS_DIR / "local" / "process_datas.py").resolve()

JOB_NAME = "plot_data"
SLURM_PARTITION = "GPGPU"
SLURM_CPUS = 2
SLURM_MEM = "20G"
SLURM_TIME = "1:00:00"

print = partial(print, flush=True)


def _parse_time_to_seconds(value: str) -> int:
    parts = value.strip().split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + int(seconds)
    raise ValueError(f"Unsupported time format: {value}")


def _format_seconds(total_seconds: float) -> str:
    seconds = max(0, int(math.ceil(total_seconds)))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _parse_mem_to_mb(value: str) -> int:
    value = value.strip().upper()
    if value.endswith("G"):
        return int(math.ceil(float(value[:-1]) * 1024))
    if value.endswith("M"):
        return int(math.ceil(float(value[:-1])))
    raise ValueError(f"Unsupported memory format: {value}")


def _estimate_processing_resources(metadata: dict[str, object]) -> tuple[str, str]:
    n_inhom = int(metadata.get("n_inhom") or 1)
    n_t_coh = int(metadata.get("n_t_coh") or 1)
    artifacts = max(1, n_inhom * n_t_coh)

    base_seconds = 120.0
    per_artifact_seconds = 0.0185
    time_seconds = base_seconds + per_artifact_seconds * artifacts
    mem_mb = 4000.0
    return f"{int(math.ceil(mem_mb))}M", _format_seconds(time_seconds)


def _estimate_plot_resources(metadata: dict[str, object]) -> tuple[str, str]:
    signal_types = metadata.get("signal_types") or []
    n_signals = len(signal_types) if isinstance(signal_types, list) and signal_types else 1

    t_det = metadata.get("t_det") or []
    t_coh = metadata.get("t_coh") or []
    n_t_det = len(t_det) if isinstance(t_det, list) and t_det else 600
    n_t_coh = len(t_coh) if isinstance(t_coh, list) else 0

    is_2d = n_t_coh > 1 or str(metadata.get("sim_type", "")).lower() == "2d"
    points = max(1, n_t_det * max(n_t_coh, 1))

    if is_2d:
        base_seconds = 60.0
        per_point_seconds = 1.5e-5
        per_signal_overhead = 30.0
        time_seconds = max(
            300.0,
            base_seconds
            + (per_point_seconds * points * n_signals)
            + (per_signal_overhead * n_signals),
        )
        bytes_per_complex = 16.0
        work_factor = 2.0
        mem_mb = 2000.0 + (
            points * n_signals * bytes_per_complex * work_factor * (max(1.0, PAD_FACTOR) ** 2)
        ) / (1024**2)
    else:
        time_seconds = max(600.0, min(1200.0, 0.8 * n_t_det * n_signals))
        mem_mb = 2000.0 + 0.0005 * points * n_signals

    mem_mb = max(mem_mb, 2000.0)
    return f"{int(math.ceil(mem_mb))}M", _format_seconds(time_seconds)


def _next_logs_dir(job_dir: Path) -> Path:
    base = job_dir / "plotting_logs"
    if not base.exists():
        base.mkdir(parents=True)
        return base
    suffix = 1
    while True:
        candidate = job_dir / f"plotting_logs_{suffix}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
        suffix += 1


def _pick_anchor_artifact(data_dir: Path) -> Path:
    candidates = sorted(data_dir.glob("*_run_t*_s*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No raw run artifacts found in {data_dir}")
    return candidates[0]


def _render_slurm_script(
    *,
    job_dir: Path,
    logs_dir: Path,
    anchor_artifact: Path,
    final_artifact: Path,
    partition: str,
    cpus: int,
    mem: str,
    time_limit: str,
    skip_if_exists: bool = False,
    time_only: bool = False,
) -> str:
    process_cmd = f'python {PROCESS_SCRIPT.as_posix()} --abs_path "{anchor_artifact.as_posix()}"'
    if skip_if_exists:
        process_cmd += " --skip_if_exists"

    plot_cmd = f'python {PLOT_SCRIPT.as_posix()} --abs_path "{final_artifact.as_posix()}"'
    if time_only:
        plot_cmd += " --time_only"

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={JOB_NAME}",
        f"#SBATCH --output={logs_dir.resolve().as_posix()}/plotting.%j.out",
        f"#SBATCH --error={logs_dir.resolve().as_posix()}/plotting.%j.err",
    ]
    if partition:
        lines.append(f"#SBATCH --partition={partition}")
    if cpus:
        lines.append(f"#SBATCH --cpus-per-task={cpus}")
    if mem:
        lines.append(f"#SBATCH --mem={mem}")
    if time_limit:
        lines.append(f"#SBATCH --time={time_limit}")

    lines.extend(
        [
            "",
            "set -euo pipefail",
            f"mkdir -p {logs_dir.resolve().as_posix()}",
            f"cd {job_dir.as_posix()}",
            "echo \"[$(date '+%F %T')] Launching process_datas.py\"",
            process_cmd,
            f'if [ ! -f "{final_artifact.as_posix()}" ]; then',
            f'  echo "Expected processed artifact not found: {final_artifact.as_posix()}" >&2',
            "  exit 1",
            "fi",
            f'echo "Final processed artifact: {final_artifact.as_posix()}"',
            "echo \"[$(date '+%F %T')] Launching plot_datas.py\"",
            plot_cmd,
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and submit a SLURM script to process artifacts and plot on HPC."
    )
    parser.add_argument("--job_dir", type=str, required=True, help="Path to jobs/<job_label>")
    parser.add_argument(
        "--skip_if_exists",
        action="store_true",
        help="Skip processing if the final artifact already exists",
    )
    parser.add_argument(
        "--no_submit",
        action="store_true",
        help="Only generate the SLURM script; do not call sbatch",
    )
    parser.add_argument(
        "--time_only",
        action="store_true",
        help="Only plot time-domain signals (skip frequency-domain plots)",
    )
    args = parser.parse_args()

    job_dir = Path(args.job_dir).resolve()
    metadata_path = job_dir / "job_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    data_dir = Path(metadata["data_dir"]).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory missing: {data_dir}")

    anchor_artifact = _pick_anchor_artifact(data_dir)
    final_artifact = data_dir / final_processed_filename(str(metadata.get("sim_type", "1d")))
    print(f"Using anchor artifact: {anchor_artifact}")
    print(f"Expected processed artifact: {final_artifact}")

    logs_dir = _next_logs_dir(job_dir)
    selected_mem = SLURM_MEM
    selected_time = SLURM_TIME

    try:
        plot_mem, plot_time = _estimate_plot_resources(metadata)
        proc_mem, proc_time = _estimate_processing_resources(metadata)
        default_mem_mb = _parse_mem_to_mb(selected_mem)
        default_time_s = _parse_time_to_seconds(selected_time)
        selected_mem_mb = max(
            default_mem_mb, _parse_mem_to_mb(plot_mem), _parse_mem_to_mb(proc_mem)
        )
        total_time_s = (_parse_time_to_seconds(plot_time) + _parse_time_to_seconds(proc_time)) * 2.0
        selected_mem = f"{selected_mem_mb}M"
        selected_time = (
            _format_seconds(total_time_s) if total_time_s > default_time_s else selected_time
        )
        print(f"Estimated processing resources: mem={proc_mem}, time={proc_time}")
        print(f"Estimated plotting resources:   mem={plot_mem}, time={plot_time}")
        print(f"Using resources: mem={selected_mem}, time={selected_time}")
    except Exception as exc:
        print(
            f"Resource estimate failed ({exc}); using defaults: mem={selected_mem}, time={selected_time}"
        )

    script_content = _render_slurm_script(
        job_dir=job_dir,
        logs_dir=logs_dir,
        anchor_artifact=anchor_artifact,
        final_artifact=final_artifact,
        partition=SLURM_PARTITION,
        cpus=SLURM_CPUS,
        mem=selected_mem,
        time_limit=selected_time,
        skip_if_exists=args.skip_if_exists,
        time_only=args.time_only,
    )
    script_path = job_dir / f"{JOB_NAME}.slurm"
    script_path.write_text(script_content, encoding="utf-8")

    print(f"📝 Generated processing+plotting script: {script_path}")
    print(f"   Logs directory: {logs_dir}")

    if args.no_submit:
        print("Submission skipped (use without --no_submit to call sbatch).")
        return

    print("Submitting SLURM job...")
    submit_msg = submit_sbatch(script_path)
    print(f"  {script_path.name}: {submit_msg}")


if __name__ == "__main__":
    main()
