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
from common.workflow import extract_job_unique_id, final_processed_filename, format_slurm_job_name
from hpc.calc_dispatcher import submit_sbatch

PLOT_SCRIPT = (SCRIPTS_DIR / "local" / "plot_datas.py").resolve()
PROCESS_SCRIPT = (SCRIPTS_DIR / "local" / "process_datas.py").resolve()

PLOT_JOB_PREFIX = "plot"
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
    signal_types = metadata.get("signal_types") or []
    n_signals = len(signal_types) if isinstance(signal_types, list) and signal_types else 1

    n_inhom = int(metadata.get("n_inhom") or 1)
    n_t_coh = int(metadata.get("n_t_coh") or 1)

    t_det = metadata.get("t_det") or []
    n_t_det = len(t_det) if isinstance(t_det, list) and t_det else int(metadata.get("n_t_det") or 1)

    n_raw_artifacts = max(1, n_inhom * n_t_coh)

    bytes_per_complex = 16.0  # complex128
    gib = 1024.0**3

    # Conservative model for process_datas.py:
    # - all raw 1D complex signals may be loaded
    # - stacked arrays of similar size are created during aggregation
    # - final averaged arrays are kept as well
    raw_loaded_bytes = n_raw_artifacts * n_signals * n_t_det * bytes_per_complex
    stacked_bytes = raw_loaded_bytes
    averaged_bytes = n_signals * max(1, n_t_coh) * max(1, n_t_det) * bytes_per_complex

    baseline_bytes = 1.0 * gib
    safety_factor = 1.35

    peak_bytes = raw_loaded_bytes + stacked_bytes + averaged_bytes
    mem_mb = (baseline_bytes + safety_factor * peak_bytes) / (1024.0**2)
    mem_mb = max(mem_mb, 2500.0)

    # Keep time estimate informational only; the SLURM job time stays fixed later.
    time_seconds = 20.0 * 60.0
    return f"{int(math.ceil(mem_mb))}M", _format_seconds(time_seconds)


def _estimate_plot_resources(metadata: dict[str, object]) -> tuple[str, str]:
    signal_types = metadata.get("signal_types") or []
    n_signals = len(signal_types) if isinstance(signal_types, list) and signal_types else 1

    t_det = metadata.get("t_det") or []
    t_coh = metadata.get("t_coh") or []

    n_t_det = len(t_det) if isinstance(t_det, list) and t_det else int(metadata.get("n_t_det") or 1)
    n_t_coh_axis = len(t_coh) if isinstance(t_coh, list) and t_coh else 0
    n_t_coh = max(n_t_coh_axis, int(metadata.get("n_t_coh") or 0))

    sim_type = str(metadata.get("sim_type", "")).lower()
    is_2d = n_t_coh > 1 or sim_type == "2d"

    bytes_per_complex = 16.0  # complex128
    gib = 1024.0**3
    pad = max(1.0, float(PAD_FACTOR))

    if is_2d:
        n_t_coh = max(1, n_t_coh)
        n_t_det = max(1, n_t_det)

        n_pad_coh = int(math.ceil(pad * n_t_coh))
        n_pad_det = int(math.ceil(pad * n_t_det))

        # Conservative dense peak-memory model:
        # always estimate as if dense padded spectra are materialized,
        # even if later only a cropped/sparse ROI is kept.
        loaded_time_domain_bytes = n_signals * n_t_coh * n_t_det * bytes_per_complex
        first_fft_workspace_bytes = n_t_coh * n_pad_det * bytes_per_complex
        second_fft_workspace_bytes = n_pad_coh * n_pad_det * bytes_per_complex
        dense_padded_output_bytes = n_signals * n_pad_coh * n_pad_det * bytes_per_complex

        # Absorptive spectrum may create another dense padded array
        absorptive_dense_bytes = (
            n_pad_coh * n_pad_det * bytes_per_complex if n_signals >= 2 else 0.0
        )

        baseline_bytes = 2.0 * gib
        safety_factor = 1.30

        peak_bytes = (
            loaded_time_domain_bytes
            + first_fft_workspace_bytes
            + second_fft_workspace_bytes
            + dense_padded_output_bytes
            + absorptive_dense_bytes
        )

        mem_mb = (baseline_bytes + safety_factor * peak_bytes) / (1024.0**2)
        mem_mb = max(mem_mb, 3000.0)

        # Keep the SLURM time fixed later; this is just informative output.
        time_seconds = 3600.0
    else:
        n_t_det = max(1, n_t_det)
        n_pad_det = int(math.ceil(pad * n_t_det))

        loaded_time_domain_bytes = n_signals * n_t_det * bytes_per_complex
        dense_padded_output_bytes = n_signals * n_pad_det * bytes_per_complex
        workspace_bytes = n_pad_det * bytes_per_complex

        baseline_bytes = 1.5 * gib
        safety_factor = 1.30

        peak_bytes = loaded_time_domain_bytes + dense_padded_output_bytes + workspace_bytes

        mem_mb = (baseline_bytes + safety_factor * peak_bytes) / (1024.0**2)
        mem_mb = max(mem_mb, 2500.0)
        time_seconds = 3600.0

    return f"{int(math.ceil(mem_mb))}M", _format_seconds(time_seconds)


def _resolve_plot_logs_dir(job_dir: Path, metadata: dict[str, object]) -> Path:
    base_logs_dir = Path(metadata.get("logs_dir") or (job_dir / "logs")).resolve()
    plot_logs_dir = base_logs_dir / "plotting"
    plot_logs_dir.mkdir(parents=True, exist_ok=True)
    return plot_logs_dir


def _pick_anchor_artifact(data_dir: Path) -> Path:
    candidates = sorted(data_dir.glob("*_run_t*_s*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No raw run artifacts found in {data_dir}")
    return candidates[0]


def _render_slurm_script(
    *,
    job_name: str,
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
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={logs_dir.resolve().as_posix()}/%x.%j.out",
        f"#SBATCH --error={logs_dir.resolve().as_posix()}/%x.%j.err",
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

    job_unique_id = str(metadata.get("job_unique_id") or extract_job_unique_id(job_dir))
    job_name = format_slurm_job_name(
        PLOT_JOB_PREFIX, metadata.get("sim_type", "data"), job_unique_id
    )
    logs_dir = _resolve_plot_logs_dir(job_dir, metadata)

    selected_mem = SLURM_MEM
    selected_time = SLURM_TIME

    try:
        plot_mem, plot_time = _estimate_plot_resources(metadata)
        proc_mem, proc_time = _estimate_processing_resources(metadata)

        default_mem_mb = _parse_mem_to_mb(selected_mem)
        selected_mem_mb = max(
            default_mem_mb,
            _parse_mem_to_mb(plot_mem),
            _parse_mem_to_mb(proc_mem),
        )
        selected_mem = f"{selected_mem_mb}M"

        # Keep the job time fixed at the configured default.
        selected_time = SLURM_TIME

        print(f"Estimated processing resources: mem={proc_mem}, time={proc_time}")
        print(f"Estimated plotting resources:   mem={plot_mem}, time={plot_time}")
        print(f"Using resources: mem={selected_mem}, time={selected_time}")
    except Exception as exc:
        print(
            f"Resource estimate failed ({exc}); using defaults: mem={selected_mem}, time={selected_time}"
        )

    script_content = _render_slurm_script(
        job_name=job_name,
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
    script_path = job_dir / f"{job_name}.slurm"
    script_path.write_text(script_content, encoding="utf-8")

    print(f"📝 Generated processing+plotting script: {script_path}")
    print(f"   SLURM job name: {job_name}")
    print(f"   Logs directory: {logs_dir}")

    if args.no_submit:
        print("Submission skipped (use without --no_submit to call sbatch).")
        return

    print("Submitting SLURM job...")
    submit_msg = submit_sbatch(script_path)
    print(f"  {script_path.name}: {submit_msg}")


if __name__ == "__main__":
    main()
