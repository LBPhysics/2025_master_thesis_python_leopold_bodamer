"""Post-process generalized spectroscopy runs and queue processing+plotting jobs on HPC.

This helper combines two manual steps executed after HPC computations finish:

1. Run :mod:`process_datas` to process run artifacts (stack and average).
2. Generate a ready-to-submit SLURM script that calls :mod:`process_datas` and :mod:`plot_datas` on the
   resulting artifact and optionally submit it via ``sbatch``.

The script targets the run-artifact workflow. It expects the same job
structure created by ``hpc_batch_dispatch.py`` and ``run_batch.py``.
Processing and plotting are done on the HPC cluster for speed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hpc_batch_dispatch import submit_sbatch
from calc_datas import SCRIPTS_DIR

PLOT_SCRIPT = (SCRIPTS_DIR / "plot_datas.py").resolve()

# Default SLURM settings used in the generated plotting script.
JOB_NAME = "plot_data"
SLURM_PARTITION: str = "GPGPU"
SLURM_CPUS: int = 2
SLURM_MEM: str = "200G"
SLURM_TIME: str = "1:00:00"


def _next_logs_dir(job_dir: Path) -> Path:
    """Create (and return) a fresh logs directory inside ``job_dir``."""

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


def _render_slurm_script(
    *,
    job_dir: Path,
    logs_dir: Path,
    artifact: Path,
    partition: str,
    cpus: int,
    mem: str,
    time_limit: str,
    skip_if_exists: bool = False,
) -> str:
    """Return the plotting SLURM script content."""

    job_dir_posix = job_dir.as_posix()
    # Use absolute log paths to avoid issues when --chdir is ignored by older SLURM versions
    logs_abs = logs_dir.resolve().as_posix()
    artifact_path = artifact.as_posix()
    plot_py = PLOT_SCRIPT.as_posix()
    process_py = (SCRIPTS_DIR / "process_datas.py").resolve().as_posix()

    # Important: SBATCH lines must come before any non-comment command
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={JOB_NAME}",
        # Use absolute paths for output/error and include job id for uniqueness
        f"#SBATCH --output={logs_abs}/plotting.%j.out",
        f"#SBATCH --error={logs_abs}/plotting.%j.err",
    ]

    # Optional SLURM resources
    if partition:
        lines.append(f"#SBATCH --partition={partition}")
    if cpus:
        lines.append(f"#SBATCH --cpus-per-task={cpus}")
    if mem:
        lines.append(f"#SBATCH --mem={mem}")
    if time_limit:
        lines.append(f"#SBATCH --time={time_limit}")

    # Build the process_datas command
    process_cmd = f'python {process_py} --abs_path "{artifact_path}"'
    if skip_if_exists:
        process_cmd += " --skip_if_exists"

    lines.extend(
        [
            "",
            "set -euo pipefail",
            # Ensure working dir and logs dir exist at runtime regardless of --chdir support
            f"mkdir -p {logs_abs}",
            f"cd {job_dir_posix}",
            "echo 'Launching process_datas.py'",
            f'final_path=$({process_cmd} 2>&1 | grep "Final processed artifact:" | sed "s/.*: //" | tr -d "\\n")',
            "echo 'Launching plot_datas.py'",
            f'python {plot_py} --abs_path "$final_path"',
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Generate and submit a SLURM script to process artifacts and plot on HPC.")
    )
    parser.add_argument(
        "--job_dir",
        type=str,
        required=True,
        help="Path to data/jobs/<job_label> (contains job_metadata.json)",
    )
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
    args = parser.parse_args()

    job_dir = Path(args.job_dir).resolve()
    metadata_path = job_dir / "job_metadata.json"
    if not metadata_path.exists():
        print(f"Metadata file not found: {metadata_path}")
        raise SystemExit("Metadata file missing.")

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    try:
        data_dir = Path(metadata["data_dir"]).resolve()
    except KeyError as exc:
        raise KeyError("job_metadata.json missing required key: data_dir") from exc
    if not data_dir.exists():
        print(f"Data directory missing: {data_dir}")
        raise SystemExit("Data directory missing.")

    # Find any *_s*.npz file (they are all equivalent for processing)
    candidates = list(data_dir.glob("*_s*.npz"))
    if not candidates:
        print(f"No artifacts found in {data_dir}")
        raise SystemExit("No artifacts found.")

    # Use the first one found
    artifact = candidates[0]
    print(f"Using artifact: {artifact}")

    logs_dir = _next_logs_dir(job_dir)
    script_content = _render_slurm_script(
        job_dir=job_dir,
        logs_dir=logs_dir,
        artifact=artifact,
        partition=SLURM_PARTITION,
        cpus=SLURM_CPUS,
        mem=SLURM_MEM,
        time_limit=SLURM_TIME,
        skip_if_exists=args.skip_if_exists,
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
