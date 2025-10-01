#!/usr/bin/env python3
"""Post-process generalized spectroscopy runs and queue plotting jobs.

This helper combines two manual steps executed after HPC computations finish:

1. Run :mod:`post_process_datas` to average inhomogeneous samples and stack
   across coherence points (when applicable).
2. Generate a ready-to-submit SLURM script that calls :mod:`plot_datas` on the
   resulting artifact and optionally submit it via ``sbatch``.

The script targets the new run-artifact workflow. It expects the same job
structure created by ``hpc_dispatch_generalized.py`` and ``run_generalized_batch.py``.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from post_process_datas import PostProcessResult, post_process_job

SCRIPTS_DIR = Path(__file__).parent.resolve()
PLOT_SCRIPT = (SCRIPTS_DIR / "plot_datas.py").resolve()
DEFAULT_SCRIPT_NAME = "plotting.slurm"

JOB_NAME = "plot_data"
# Default SLURM settings (can be overridden via CLI). Use None to defer to cluster defaults.
SLURM_PARTITION: str | None = None
SLURM_CPUS: int | None = 1
SLURM_MEM: str | None = None
SLURM_TIME: str | None = None


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
    final_artifact: Path,
    partition: str | None,
    cpus: int | None,
    mem: str | None,
    time_limit: str | None,
) -> str:
    """Return the plotting SLURM script content."""

    job_dir_posix = job_dir.as_posix()
    # Use absolute log paths to avoid issues when --chdir is ignored by older SLURM versions
    logs_abs = logs_dir.resolve().as_posix()
    plot_path = final_artifact.as_posix()
    plot_py = PLOT_SCRIPT.as_posix()

    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
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

    lines.extend(
        [
            "",
            # Ensure working dir and logs dir exist at runtime regardless of --chdir support
            f"mkdir -p {logs_abs}",
            f"cd {job_dir_posix}",
            "echo 'Launching plot_datas.py'",
            f'python {plot_py} --abs_path "{plot_path}"',
        ]
    )

    return "\n".join(lines) + "\n"


def _write_script(job_dir: Path, script_name: str, content: str) -> Path:
    target = job_dir / script_name
    target.write_text(content, encoding="utf-8", newline="\n")
    target.chmod(0o755)
    return target


def _submit_script(script_path: Path) -> bool:
    try:
        subprocess.run(["sbatch", str(script_path)], check=True)
        print(f"Submitted {script_path}")
        return True
    except FileNotFoundError:
        print("⚠️  sbatch command not found. Submit the script manually when available.")
    except subprocess.CalledProcessError as exc:
        print(
            f"⚠️  sbatch failed with exit code {exc.returncode}. Submit manually after inspection."
        )
    return False


def _record_target(job_dir: Path, final_artifact: Path) -> None:
    record = job_dir / "plotting_target.txt"
    record.write_text(final_artifact.as_posix() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Average/stack run artifacts and generate a plotting SLURM script (new pipeline)."
        )
    )
    parser.add_argument(
        "--job_dir",
        type=str,
        required=True,
        help="Path to batch_jobs_generalized/<job_label> (contains metadata.json)",
    )
    parser.add_argument(
        "--skip_inhom",
        action="store_true",
        help="Reuse existing averaged artifacts when present",
    )
    parser.add_argument(
        "--skip_stack",
        action="store_true",
        help="Reuse existing stacked artifact when present",
    )
    parser.add_argument(
        "--no_submit",
        action="store_true",
        help="Only generate the SLURM script; do not call sbatch",
    )
    parser.add_argument(
        "--script_name",
        default=DEFAULT_SCRIPT_NAME,
        help=f"Name of the generated SLURM script (default: {DEFAULT_SCRIPT_NAME})",
    )
    # SLURM resources are configured via module-level defaults above; no CLI needed.
    args = parser.parse_args()

    job_dir = Path(args.job_dir).resolve()
    result: PostProcessResult | None = post_process_job(
        job_dir,
        skip_inhom=args.skip_inhom,
        skip_stack=args.skip_stack,
    )

    if result is None:
        raise SystemExit("Post-processing failed; see messages above.")

    final_artifact = result.final_path
    print("📦 Final artifact for plotting:")
    print(f"  {final_artifact}")

    logs_dir = _next_logs_dir(job_dir)
    script_content = _render_slurm_script(
        job_dir=job_dir,
        logs_dir=logs_dir,
        final_artifact=final_artifact,
        partition=SLURM_PARTITION,
        cpus=SLURM_CPUS,
        mem=SLURM_MEM,
        time_limit=SLURM_TIME,
    )
    script_path = _write_script(job_dir, args.script_name, script_content)
    _record_target(job_dir, final_artifact)

    print(f"📝 Generated plotting script: {script_path}")
    print(f"   Logs directory: {logs_dir}")

    if args.no_submit:
        print("Submission skipped (use without --no_submit to call sbatch).")
        return

    _submit_script(script_path)


if __name__ == "__main__":
    main()
