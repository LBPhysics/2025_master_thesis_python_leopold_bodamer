"""Collect failed phase-cycling points and dispatch 1D retry SLURM jobs."""

from __future__ import annotations

import argparse
import math
import re
import shlex
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common.retry_queue import dedupe_retry_candidates, ensure_retry_dir, load_retry_candidates
from common.workflow import write_json

DEFAULT_RETRY_BATCH_SIZE = 10
DEFAULT_RETRY_CPUS = 8
DEFAULT_RETRY_MEM = "2048M"
DEFAULT_RETRY_TIME = "01:00:00"
DEFAULT_PARTITION = "GPGPU,metis"


def _render_retry_slurm_script(
    *,
    job_name: str,
    log_dir: Path,
    worker_path: Path,
    python_executable: Path,
    retry_file: Path,
    batch_id: int,
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

{python_cmd} -u {worker_arg} \
  --retry_file "{retry_file}" \
  --batch_id {batch_id}
"""


def _parse_sbatch_job_id(message: str) -> str | None:
    match = re.search(r"Submitted batch job\s+(\d+)", message)
    return match.group(1) if match else None


def submit_sbatch(script_path: Path) -> str:
    sbatch = shutil.which("sbatch")
    if sbatch is None:
        raise RuntimeError("sbatch not found on PATH. Run this on your cluster login node.")

    result = subprocess.run(
        [sbatch, str(script_path)],
        cwd=str(script_path.parent),
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


import shutil


def main() -> None:
    parser = argparse.ArgumentParser(description="Dispatch 1D retries for failed phase-cycling points.")
    parser.add_argument("--job_dir", type=str, required=True)
    parser.add_argument("--retry_batch_size", type=int, default=DEFAULT_RETRY_BATCH_SIZE)
    parser.add_argument("--cpus_per_task", type=int, default=DEFAULT_RETRY_CPUS)
    parser.add_argument("--mem", type=str, default=DEFAULT_RETRY_MEM)
    parser.add_argument("--time", type=str, default=DEFAULT_RETRY_TIME)
    parser.add_argument("--partition", type=str, default=DEFAULT_PARTITION)
    parser.add_argument("--no_submit", action="store_true")
    args = parser.parse_args()

    job_dir = Path(args.job_dir).resolve()
    retry_dir = ensure_retry_dir(job_dir)
    logs_dir = job_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    candidate_files = sorted(retry_dir.glob("retry_candidates_*.jsonl"))
    if not candidate_files:
        print(f"No retry candidate files found in {retry_dir}; nothing to dispatch.")
        return

    candidates = dedupe_retry_candidates(load_retry_candidates(candidate_files))
    if not candidates:
        print(f"Retry candidate files were empty in {retry_dir}; nothing to dispatch.")
        return

    print(f"Collected {len(candidates)} unique retry candidate(s) from {len(candidate_files)} file(s).")

    batch_size = max(1, int(args.retry_batch_size))
    n_batches = math.ceil(len(candidates) / batch_size)

    worker_path = (SCRIPTS_DIR / "hpc" / "run_retry_batch.py").resolve()
    if not worker_path.exists():
        raise FileNotFoundError(f"Missing retry worker script: {worker_path}")

    python_executable = Path(sys.executable).resolve()
    script_paths: list[Path] = []

    for batch_id in range(n_batches):
        start = batch_id * batch_size
        stop = min(len(candidates), start + batch_size)
        batch_items = candidates[start:stop]

        retry_file = retry_dir / f"retry_batch_{batch_id:03d}.json"
        write_json(retry_file, {"retries": batch_items})

        job_name = f"retry1d{batch_id:02d}of{n_batches:02d}"
        script_path = retry_dir / f"{job_name}.slurm"
        script_path.write_text(
            _render_retry_slurm_script(
                job_name=job_name,
                log_dir=logs_dir.resolve(),
                worker_path=worker_path,
                python_executable=python_executable,
                retry_file=retry_file.resolve(),
                batch_id=batch_id,
                cpus_per_task=int(args.cpus_per_task),
                requested_mem=args.mem,
                requested_time=args.time,
                partition=args.partition,
            ),
            encoding="utf-8",
        )
        script_paths.append(script_path)
        print(f"  retry batch {batch_id}: {len(batch_items)} item(s) -> {script_path.name}")

    if args.no_submit:
        print("Retry SLURM scripts generated only (--no_submit set).")
        return

    print("Submitting retry SLURM jobs...")
    for script_path in script_paths:
        msg = submit_sbatch(script_path)
        print(f"  {script_path.name}: {msg}")


if __name__ == "__main__":
    main()
