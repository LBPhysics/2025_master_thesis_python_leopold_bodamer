"""Queue strict reduction and plotting as two dependent HPC jobs."""

from __future__ import annotations

import argparse
import json
import math
import re
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

REDUCE_JOB_PREFIX = "reduce"
PLOT_JOB_PREFIX = "plot"
SLURM_PARTITION = "GPGPU"
REDUCE_CPUS = 1
PLOT_CPUS = 2
DEFAULT_REDUCE_MEM = "4G"
DEFAULT_PLOT_MEM = "20G"
DEFAULT_REDUCE_TIME = "0:20:00"
DEFAULT_PLOT_TIME = "1:00:00"

print = partial(print, flush=True)


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


def _partial_count(data_dir: Path, prefix: str) -> int:
    return len(list(data_dir.glob(f"{prefix}_batch_*.partial.npz")))


def _estimate_processing_resources(
    metadata: dict[str, object], *, partial_count: int
) -> tuple[str, str]:
    signal_types = metadata.get("signal_types") or []
    n_signals = len(signal_types) if isinstance(signal_types, list) and signal_types else 1
    n_t_coh = int(metadata.get("n_t_coh") or 1)
    t_det = metadata.get("t_det") or []
    n_t_det = len(t_det) if isinstance(t_det, list) and t_det else 1

    bytes_per_complex = 16.0
    gib = 1024.0**3
    peak_bytes = n_signals * n_t_coh * n_t_det * bytes_per_complex * 3.0
    mem_mb = (0.75 * gib + peak_bytes) / (1024.0**2)
    mem_mb = max(mem_mb, 2000.0)

    elements_total = max(1, partial_count) * max(1, n_signals) * max(1, n_t_coh) * max(1, n_t_det)
    time_seconds = 90.0 + 12.0 * max(1, partial_count) + elements_total / 2_000_000.0
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

    bytes_per_complex = 16.0
    gib = 1024.0**3
    pad = max(1.0, float(PAD_FACTOR))
    n_components = 3

    if is_2d:
        n_t_coh = max(1, n_t_coh)
        n_t_det = max(1, n_t_det)
        n_pad_coh = int(math.ceil(pad * n_t_coh))
        n_pad_det = int(math.ceil(pad * n_t_det))
        peak_bytes = (
            n_signals * n_t_coh * n_t_det * bytes_per_complex
            + n_t_coh * n_pad_det * bytes_per_complex
            + n_pad_coh * n_pad_det * bytes_per_complex
            + n_signals * n_pad_coh * n_pad_det * bytes_per_complex
        )
        mem_mb = (2.0 * gib + 1.25 * peak_bytes) / (1024.0**2)
        mem_mb = max(mem_mb, 3000.0)

        n_time_figs = n_signals * n_components
        n_freq_figs = n_signals * n_components
        fft_elements = max(1, n_signals * n_pad_coh * n_pad_det)
        time_seconds = 120.0 + 8.0 * n_time_figs + 15.0 * n_freq_figs + fft_elements / 400_000.0
    else:
        n_t_det = max(1, n_t_det)
        n_pad_det = int(math.ceil(pad * n_t_det))
        peak_bytes = (n_signals * n_t_det + n_signals * n_pad_det + n_pad_det) * bytes_per_complex
        mem_mb = (1.5 * gib + 1.25 * peak_bytes) / (1024.0**2)
        mem_mb = max(mem_mb, 2500.0)

        n_time_figs = n_signals * n_components
        n_freq_figs = 0 if sim_type == "0d" else n_signals * n_components
        fft_elements = max(1, n_signals * n_pad_det)
        time_seconds = 90.0 + 5.0 * n_time_figs + 8.0 * n_freq_figs + fft_elements / 800_000.0

    return f"{int(math.ceil(mem_mb))}M", _format_seconds(time_seconds)


def _resolve_logs_dir(job_dir: Path, metadata: dict[str, object], stage: str) -> Path:
    base_logs_dir = Path(metadata.get("logs_dir") or (job_dir / "logs")).resolve()
    stage_logs_dir = base_logs_dir / stage
    stage_logs_dir.mkdir(parents=True, exist_ok=True)
    return stage_logs_dir


def _render_reduction_script(
    *,
    job_name: str,
    job_dir: Path,
    logs_dir: Path,
    partition: str,
    cpus: int,
    mem: str,
    time_limit: str,
    skip_if_exists: bool = False,
) -> str:
    process_cmd = f'python {PROCESS_SCRIPT.as_posix()} --job_dir "{job_dir.as_posix()}"'
    if skip_if_exists:
        process_cmd += " --skip_if_exists"

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
            "echo \"[$(date '+%F %T')] Launching strict reduction\"",
            process_cmd,
        ]
    )
    return "\n".join(lines) + "\n"


def _render_plot_script(
    *,
    job_name: str,
    job_dir: Path,
    logs_dir: Path,
    final_artifact: Path,
    partition: str,
    cpus: int,
    mem: str,
    time_limit: str,
    time_only: bool = False,
) -> str:
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


def _extract_job_id(sbatch_output: str) -> str:
    match = re.search(r"(\d+)\s*$", sbatch_output.strip())
    if not match:
        raise RuntimeError(f"Could not parse SLURM job id from sbatch output: {sbatch_output!r}")
    return match.group(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and submit strict reduction and dependent plotting jobs."
    )
    parser.add_argument(
        "--job_dir",
        type=str,
        required=True,
        help="Path to the job directory, e.g. jobs/01_123456_monomer",
    )
    parser.add_argument("--skip_if_exists", action="store_true")
    parser.add_argument("--no_submit", action="store_true")
    parser.add_argument("--time_only", action="store_true")
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

    final_artifact = data_dir / final_processed_filename(str(metadata.get("sim_type", "1d")))
    prefix = str(metadata["data_base_name"])
    partial_count = _partial_count(data_dir, prefix)
    print(f"Expected processed artifact: {final_artifact}")
    print(f"Detected partial artifact count: {partial_count}")

    job_unique_id = str(metadata.get("job_unique_id") or extract_job_unique_id(job_dir))
    reduce_job_name = format_slurm_job_name(
        REDUCE_JOB_PREFIX, metadata.get("sim_type", "data"), job_unique_id
    )
    plot_job_name = format_slurm_job_name(
        PLOT_JOB_PREFIX, metadata.get("sim_type", "data"), job_unique_id
    )
    reduce_logs_dir = _resolve_logs_dir(job_dir, metadata, "reduction")
    plot_logs_dir = _resolve_logs_dir(job_dir, metadata, "plotting")

    reduce_mem = DEFAULT_REDUCE_MEM
    reduce_time = DEFAULT_REDUCE_TIME
    plot_mem = DEFAULT_PLOT_MEM
    plot_time = DEFAULT_PLOT_TIME
    try:
        est_reduce_mem, est_reduce_time = _estimate_processing_resources(
            metadata, partial_count=partial_count
        )
        est_plot_mem, est_plot_time = _estimate_plot_resources(metadata)
        reduce_mem_mb = max(_parse_mem_to_mb(reduce_mem), _parse_mem_to_mb(est_reduce_mem))
        plot_mem_mb = max(_parse_mem_to_mb(plot_mem), _parse_mem_to_mb(est_plot_mem))
        reduce_mem = f"{reduce_mem_mb}M"
        plot_mem = f"{plot_mem_mb}M"
        reduce_time = est_reduce_time
        plot_time = est_plot_time
        print(f"Estimated reduction resources: mem={est_reduce_mem}, time={est_reduce_time}")
        print(f"Using reduction resources: mem={reduce_mem}, time={reduce_time}")
        print(f"Estimated plotting resources:  mem={est_plot_mem}, time={est_plot_time}")
        print(f"Using plotting resources:  mem={plot_mem}, time={plot_time}")
    except Exception as exc:
        print(
            f"Resource estimate failed ({exc}); using defaults: "
            f"reduction mem={reduce_mem}, time={reduce_time}; "
            f"plot mem={plot_mem}, time={plot_time}"
        )

    reduce_script_content = _render_reduction_script(
        job_name=reduce_job_name,
        job_dir=job_dir,
        logs_dir=reduce_logs_dir,
        partition=SLURM_PARTITION,
        cpus=REDUCE_CPUS,
        mem=reduce_mem,
        time_limit=reduce_time,
        skip_if_exists=args.skip_if_exists,
    )
    plot_script_content = _render_plot_script(
        job_name=plot_job_name,
        job_dir=job_dir,
        logs_dir=plot_logs_dir,
        final_artifact=final_artifact,
        partition=SLURM_PARTITION,
        cpus=PLOT_CPUS,
        mem=plot_mem,
        time_limit=plot_time,
        time_only=args.time_only,
    )

    reduce_script_path = job_dir / f"{reduce_job_name}.slurm"
    plot_script_path = job_dir / f"{plot_job_name}.slurm"
    reduce_script_path.write_text(reduce_script_content, encoding="utf-8")
    plot_script_path.write_text(plot_script_content, encoding="utf-8")

    print(f"📝 Generated reduction script: {reduce_script_path}")
    print(f"📝 Generated plotting script: {plot_script_path}")
    print(f"   Reduction logs: {reduce_logs_dir}")
    print(f"   Plot logs: {plot_logs_dir}")

    if args.no_submit:
        print("Submission skipped (use without --no_submit to call sbatch).")
        return

    print("Submitting reduction SLURM job...")
    reduce_submit_msg = submit_sbatch(reduce_script_path)
    print(f"  {reduce_script_path.name}: {reduce_submit_msg}")
    reduce_job_id = _extract_job_id(reduce_submit_msg)

    dependency = f"afterok:{reduce_job_id}"
    print(f"Submitting dependent plotting SLURM job with --dependency={dependency}...")
    plot_submit_msg = submit_sbatch(plot_script_path, dependency=dependency)
    print(f"  {plot_script_path.name}: {plot_submit_msg}")


if __name__ == "__main__":
    main()
