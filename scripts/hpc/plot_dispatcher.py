"""Post-process generalized spectroscopy runs and queue processing+plotting jobs on HPC.

This helper combines two manual steps executed after HPC computations finish:

1. Run :mod:`process_datas` to process run artifacts (stack and average).
2. Generate a ready-to-submit SLURM script that calls :mod:`process_datas` and :mod:`plot_datas` on the
   resulting artifact and optionally submit it via ``sbatch``.

The script targets the run-artifact workflow. It expects the same job
structure created by ``scripts/hpc/calc_dispatcher.py`` and ``scripts/hpc/run_batch.py``.
Processing and plotting are done on the HPC cluster for speed.
"""

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

from hpc.calc_dispatcher import submit_sbatch

PLOT_SCRIPT = (SCRIPTS_DIR / "local" / "plot_datas.py").resolve()

# Default SLURM settings used in the generated plotting script.
JOB_NAME = "plot_data"
SLURM_PARTITION: str = "GPGPU"
SLURM_CPUS: int = 2
SLURM_MEM: str = "20G"
SLURM_TIME: str = "1:00:00"

# Plotting parameters (must mirror scripts/local/plot_datas.py)
PLOT_PAD_FACTOR: float = 20.0

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
	"""Return (mem, time) estimates for processing based on job metadata."""
	n_inhom = int(metadata.get("n_inhom") or 1)
	n_t_coh = int(metadata.get("n_t_coh") or 0)
	if n_t_coh <= 0:
		t_coh = metadata.get("t_coh") or []
		n_t_coh = len(t_coh) if isinstance(t_coh, list) else 1
	if n_inhom <= 0:
		n_inhom = 1
	if n_t_coh <= 0:
		n_t_coh = 1

	artifacts = n_inhom * n_t_coh
	# Calibrated from 150,050 artifacts -> 48m 28s processing time.
	base_seconds = 120.0
	per_artifact_seconds = 0.0185
	time_seconds = base_seconds + per_artifact_seconds * artifacts

	# Processing is mostly I/O bound; keep a modest memory floor.
	mem_mb = 4000.0

	requested_mem = f"{int(math.ceil(mem_mb))}M"
	requested_time = _format_seconds(time_seconds)
	return requested_mem, requested_time


def _estimate_plot_resources(metadata: dict[str, object]) -> tuple[str, str]:
	"""Return (mem, time) estimates for plotting based on job metadata."""
	signal_types = metadata.get("signal_types") or []
	n_signals = len(signal_types) if isinstance(signal_types, list) else 1
	if n_signals <= 0:
		n_signals = 1

	t_det = metadata.get("t_det") or []
	t_coh = metadata.get("t_coh") or []
	n_t_det = len(t_det) if isinstance(t_det, list) else 0
	n_t_coh = len(t_coh) if isinstance(t_coh, list) else 0

	if n_t_det <= 0:
		# Fallback to safe, small defaults
		n_t_det = 600
		n_t_coh = 0

	is_2d = n_t_coh > 0
	points = n_t_det * n_t_coh if is_2d else n_t_det
	points = max(points, 1)

	# Very simple scaling model (empirical, conservative).
	if is_2d:
		# 2D: FFT dominates, scale with total grid points.
		# Calibrated to ~6–7 minutes for ~3000x3000 grids (2 signals).
		base_seconds = 60.0
		per_point_seconds = 1.5e-5
		per_signal_overhead = 30.0
		time_seconds = max(
			300.0,
			base_seconds + (per_point_seconds * points * n_signals) + (per_signal_overhead * n_signals),
		)
		# Memory: estimate from complex arrays created during 2D FFTs.
		# Simple model = N * signals * bytes_per_complex * work_factor * pad_factor^2.
		bytes_per_complex = 16.0  # complex128
		work_factor = 2.0
		pad_factor = max(1.0, float(PLOT_PAD_FACTOR))
		mem_mb = (
			points
			* n_signals
			* bytes_per_complex
			* work_factor
			* (pad_factor**2)
		) / (1024**2)
		base_mem_mb = 2000.0
		mem_mb = base_mem_mb + mem_mb
	else:
		# 1D: expect 5–10 minutes even with averaging.
		# Baseline of 10 minutes for 1D as well.
		time_seconds = max(600.0, min(1200.0, 0.8 * n_t_det * n_signals))
		base_mem_mb = 2000.0
		per_point_mb = 0.0005
		mem_mb = base_mem_mb + per_point_mb * points * n_signals

	mem_mb = max(mem_mb, 2000.0)

	requested_mem = f"{int(math.ceil(mem_mb))}M"
	requested_time = _format_seconds(time_seconds)
	return requested_mem, requested_time


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
	time_only: bool = False,
) -> str:
	"""Return the plotting SLURM script content."""

	job_dir_posix = job_dir.as_posix()
	# Use absolute log paths to avoid issues when --chdir is ignored by older SLURM versions
	logs_abs = logs_dir.resolve().as_posix()
	artifact_path = artifact.as_posix()
	plot_py = PLOT_SCRIPT.as_posix()
	process_py = (SCRIPTS_DIR / "local" / "process_datas.py").resolve().as_posix()

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
	plot_cmd = f'python {plot_py} --abs_path "$final_path"'
	if time_only:
		plot_cmd += " --time_only"

	artifact_dir = artifact.parent.resolve().as_posix()

	lines.extend(
		[
			"",
			"set -euo pipefail",
			# Ensure working dir and logs dir exist at runtime regardless of --chdir support
			f"mkdir -p {logs_abs}",
			f"cd {job_dir_posix}",
			"echo \"[$(date '+%F %T')] Launching process_datas.py\"",
			f"{process_cmd}",
			"",
			"echo \"[$(date '+%F %T')] Locating processed artifact\"",
			"set +e",
			"set +o pipefail",
			f"final_path=$(python - <<'PY'\nfrom pathlib import Path\nimport sys\nartifact_dir = Path('{artifact_dir}')\ncandidates = []\nfor candidate in artifact_dir.glob('*_inhom_averaged.npz'):\n    try:\n        resolved = candidate.resolve()\n        candidates.append((resolved.stat().st_mtime, resolved))\n    except FileNotFoundError:\n        continue\nif not candidates:\n    sys.exit('No processed artifact found after process_datas run.')\nmtime, path = max(candidates, key=lambda item: item[0])\nprint(path.as_posix())\nPY\n)",
			"status=$?",
			"set -e",
			"set -o pipefail",
			'if [ $status -ne 0 ] || [ -z "$final_path" ]; then',
			"  echo 'Failed to locate processed artifact. Check process_datas output above.' >&2",
			"  exit 1",
			"fi",
			'echo "Final processed artifact: $final_path"',
			"echo \"[$(date '+%F %T')] Launching plot_datas.py\"",
			plot_cmd,
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
	parser.add_argument(
		"--time_only",
		action="store_true",
		help="Only plot time-domain signals (skip frequency-domain plots)",
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
	selected_mem = SLURM_MEM
	selected_time = SLURM_TIME
	try:
		plot_mem, plot_time = _estimate_plot_resources(metadata)
		proc_mem, proc_time = _estimate_processing_resources(metadata)
		default_mem_mb = _parse_mem_to_mb(selected_mem)
		default_time_s = _parse_time_to_seconds(selected_time)
		plot_mem_mb = _parse_mem_to_mb(plot_mem)
		proc_mem_mb = _parse_mem_to_mb(proc_mem)
		plot_time_s = _parse_time_to_seconds(plot_time)
		proc_time_s = _parse_time_to_seconds(proc_time)
		safety_factor = 2.0
		total_time_s = (plot_time_s + proc_time_s) * safety_factor
		total_time = _format_seconds(total_time_s)
		selected_mem_mb = max(default_mem_mb, plot_mem_mb, proc_mem_mb)
		selected_mem = f"{selected_mem_mb}M"
		selected_time = (
			total_time if total_time_s > default_time_s else selected_time
		)
		print(
			"Estimated resources (processing): "
			f"mem={proc_mem}, time={proc_time} (n_inhom={metadata.get('n_inhom')}, "
			f"n_t_coh={metadata.get('n_t_coh')})"
		)
		print(
			"Estimated resources (plotting): "
			f"mem={plot_mem}, time={plot_time} (signals={len(metadata.get('signal_types', []))}, "
			f"t_det={len(metadata.get('t_det', []))}, t_coh={len(metadata.get('t_coh', []))})"
		)
		print(f"Estimated total time: {total_time}")
		print(f"Using resources: mem={selected_mem}, time={selected_time}")
	except Exception as exc:
		print(
			f"Resource estimate failed ({exc}); using defaults: mem={selected_mem}, time={selected_time}"
		)

	script_content = _render_slurm_script(
		job_dir=job_dir,
		logs_dir=logs_dir,
		artifact=artifact,
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
