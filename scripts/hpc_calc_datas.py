"""Minimal SLURM job generator and submitter for batched 1D/2D runs.

This script creates one SLURM script per batch index (0..n_batches-1)
and, by default, submits them with ``sbatch``.
Example usage:
    for a 2d time plot
    python hpc_calc_datas.py --n_batches 10 --sim_type 2d
    for a 1d plot with averaged inhomogeneous broadening
    python hpc_calc_datas.py --n_batches 5 --sim_type 1d
    Layout:
- Scripts are generated under ``SCRIPTS_DIR/batch_jobs/{n_batches}batches``
    where ``SCRIPTS_DIR`` is the directory containing this file.
- Logs are written to ``SCRIPTS_DIR/batch_jobs/{n_batches}batches[/_i]/logs/``
    via Slurm's ``--output``/``--error`` options.

Each job runs (from the batching directory):
    python ../../calc_datas.py --sim_type {sim_type} --n_batches {n_batches} --batch_idx {batch_idx}

Notes:
- Mail notifications are included ONLY for the first and last batch indices.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from thesis_paths import SCRIPTS_DIR


def _slurm_script_text(
    *,
    job_name: str,
    n_batches: int,
    batch_idx: int,
    sim_type: str,
    logs_subdir: str,
) -> str:
    """Render the SLURM script text for a single batch index.

    Mail directives are included only for first and last batches.
    """
    mail_lines = (
        (
            '#SBATCH --mail-type="END,FAIL"\n'
            "#SBATCH --mail-user=leopold.bodamer@student.uni-tuebingen.de\n"
        )
        if (batch_idx == 0 or batch_idx == n_batches - 1)
        else ""
    )

    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={logs_subdir}/%x.out
#SBATCH --error={logs_subdir}/%x.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=2G
#SBATCH --time=0-02:00:00
{mail_lines}

python ../../calc_datas.py --sim_type {sim_type} --n_batches {n_batches} --batch_idx {batch_idx}
"""


def _ensure_dirs(job_dir: Path, logs_subdir: str) -> None:
    """Create the batching directory and the given logs subdirectory if missing."""
    job_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = job_dir / logs_subdir
    logs_dir.mkdir(parents=True, exist_ok=True)


def _submit_job(script_path: Path) -> str:
    """Submit a job with sbatch and return the scheduler response.

    If ``sbatch`` is not available, a helpful error is raised.
    """
    sbatch = shutil.which("sbatch")
    if sbatch is None:
        raise RuntimeError("sbatch not found on PATH. Run this on your cluster login node.")

    # Submit from within the batching directory so relative log paths work.
    result = subprocess.run(
        [sbatch, script_path.name],
        cwd=str(script_path.parent),
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and submit SLURM jobs for batched 1D/2D spectroscopy runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n_batches",
        type=int,
        required=True,
        help="Total number of batches (creates one job per batch_idx 0..n_batches-1)",
    )
    parser.add_argument(
        "--generate_only",
        action="store_true",
        help="Only generate the .slurm scripts without submitting via sbatch.",
    )
    parser.add_argument(
        "--sim_type",
        type=str,
        default="2d",
        choices=["1d", "2d"],
        help="Execution mode (default: 2d)",
    )
    args = parser.parse_args()

    n_batches = int(args.n_batches)
    if n_batches <= 0:
        raise ValueError("--n_batches must be a positive integer")

    # Create a fixed job directory under SCRIPTS_DIR/batch_jobs (only logs dir will be unique)
    job_root = SCRIPTS_DIR / "batch_jobs"
    sim_type = str(args.sim_type).lower()
    base_name = f"{sim_type}_{n_batches}batches"
    job_dir = job_root / base_name

    # Determine a unique logs subdirectory name within job_dir
    logs_subdir = "logs"
    suffix = 0
    while (job_dir / logs_subdir).exists():
        suffix += 1
        logs_subdir = f"logs_{suffix}"

    _ensure_dirs(job_dir, logs_subdir)

    action_verb = "Generating" if args.generate_only else "Creating and submitting"
    print(f"{action_verb} {n_batches} SLURM jobs in {job_dir} (logs -> {logs_subdir}) ...")

    for batch_idx in range(n_batches):  # TODO also add the name of the config file to the job name
        job_name = f"{sim_type}b{batch_idx:02d}of{n_batches:02d}"
        script_name = f"{job_name}.slurm"
        script_path = job_dir / script_name

        content = _slurm_script_text(
            job_name=job_name,
            n_batches=n_batches,
            batch_idx=batch_idx,
            sim_type=sim_type,
            logs_subdir=logs_subdir,
        )
        script_path.write_text(content, encoding="utf-8")

        if args.generate_only:
            print(f"  generated {script_name}")
        else:
            try:
                submit_msg = _submit_job(script_path)
            except Exception as exc:  # Fail fast with a clear message
                raise RuntimeError(f"Failed to submit {script_name}: {exc}") from exc

            print(f"  submitted {script_name}: {submit_msg}")

    print("All jobs generated." if args.generate_only else "All jobs submitted.")


if __name__ == "__main__":  # pragma: no cover
    main()
