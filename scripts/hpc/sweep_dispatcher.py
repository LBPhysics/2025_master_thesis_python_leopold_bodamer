"""Generate and (optionally) submit SLURM jobs for a parameter sweep.

This script mirrors sweep_params.py but emits SLURM jobs so the sweep can be
performed on the cluster to measure realistic runtimes. Each case writes a
JSON result file that can be aggregated into a summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
	sys.path.insert(0, str(SCRIPTS_DIR))

from local.sweep_params import (
	BASELINE_OVERRIDES,
	DEFAULT_CONFIG,
	apply_overrides,
	build_ofat_cases,
	build_env,
)

PROJECT_ROOT = SCRIPTS_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "jobs" / "sweeps"


def _render_slurm_script(
	*,
	job_name: str,
	batch_path: Path,
	sim_type: str,
	results_dir: Path,
	python_executable: Path,
	partition: str,
	requested_mem: str,
	requested_time: str,
) -> str:
	python_cmd = str(python_executable)
	calc_datas = (SCRIPTS_DIR / "local" / "calc_datas.py").as_posix()
	batch_path_str = str(batch_path)
	results_dir_str = str(results_dir)
	return textwrap.dedent(
		f"""\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --cpus-per-task=16
#SBATCH --mem={requested_mem}
#SBATCH --time={requested_time}
#SBATCH --partition={partition}

set -euo pipefail

mkdir -p logs

export PYTHONPATH=\"{os.environ.get('PYTHONPATH', '')}\"

"{python_cmd}" - <<'PY'
import json
import os
import subprocess
import time
from pathlib import Path

batch_path = Path(r"{batch_path_str}")
results_dir = Path(r"{results_dir_str}")
results_dir.mkdir(parents=True, exist_ok=True)

with batch_path.open("r", encoding="utf-8") as handle:
    cases = json.load(handle)

python_cmd = r"{python_cmd}"
calc_datas = r"{calc_datas}"
sim_type = r"{sim_type}"

for case in cases:
    index = int(case["index"])
    label = case["label"]
    config_path = case["config_path"]
    start = time.perf_counter()
    proc = subprocess.run([python_cmd, calc_datas, "--sim_type", sim_type, "--config", config_path])
    elapsed = time.perf_counter() - start

	data = {{
		"index": index,
		"label": label,
		"config_path": config_path,
		"return_code": int(proc.returncode),
		"runtime_s": round(elapsed, 3),
	}}

	out_path = results_dir / f"case_{{index:03d}}.json"
	out_path.write_text(json.dumps(data, indent=2) + "\\n", encoding="utf-8")
PY
"""
	)


def _collect_results(results_dir: Path, output_dir: Path) -> None:
	results = []
	for path in sorted(results_dir.glob("case_*.json")):
		try:
			payload = json.loads(path.read_text(encoding="utf-8"))
		except Exception:
			continue
		if isinstance(payload, dict):
			results.append(payload)

	if not results:
		raise RuntimeError(f"No results found in {results_dir}")

	results.sort(key=lambda item: item.get("index", 0))

	json_path = output_dir / "summary.json"
	json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

	csv_path = output_dir / "summary.csv"
	with csv_path.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=results[0].keys())
		writer.writeheader()
		writer.writerows(results)

	print(f"Summary JSON: {json_path}")
	print(f"Summary CSV:  {csv_path}")


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Create SLURM sweep jobs to measure runtime scaling",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument(
		"--config",
		type=str,
		default=str(DEFAULT_CONFIG),
		help="Baseline YAML config to sweep around",
	)
	parser.add_argument(
		"--sim_type",
		choices=["0d", "1d", "2d"],
		default="1d",
		help="Simulation dimensionality",
	)
	parser.add_argument(
		"--label",
		type=str,
		default=None,
		help="Optional label for the sweep folder",
	)
	parser.add_argument(
		"--n_batches",
		type=int,
		default=5,
		help="Number of sweep batches (SLURM jobs)",
	)
	parser.add_argument(
		"--partition",
		type=str,
		default="GPGPU,metis",
		help="SLURM partition",
	)
	parser.add_argument(
		"--mem",
		type=str,
		default="8G",
		help="Memory per job",
	)
	parser.add_argument(
		"--time",
		type=str,
		default="00:30:00",
		help="Wall time per job",
	)
	parser.add_argument(
		"--no_submit",
		action="store_true",
		help="Only generate scripts; do not submit",
	)
	parser.add_argument(
		"--collect",
		action="store_true",
		help="Collect results from existing sweep directory",
	)
	parser.add_argument(
		"--sweep_dir",
		type=str,
		default=None,
		help="Existing sweep dir to collect (used with --collect)",
	)
	args = parser.parse_args()

	if args.collect:
		if not args.sweep_dir:
			raise ValueError("--sweep_dir is required when using --collect")
		sweep_dir = Path(args.sweep_dir).expanduser().resolve()
		results_dir = sweep_dir / "results"
		_collect_results(results_dir, sweep_dir)
		return

	base_path = Path(args.config).expanduser().resolve()
	if not base_path.exists():
		raise FileNotFoundError(f"Config file not found: {base_path}")

	with base_path.open("r", encoding="utf-8") as handle:
		base_cfg = yaml.safe_load(handle)

	base_cfg = apply_overrides(base_cfg, BASELINE_OVERRIDES)

	timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
	label = args.label or base_path.stem

	sweep_dir = DATA_DIR / f"{label}_{args.sim_type}_{timestamp}"
	sweep_dir.mkdir(parents=True, exist_ok=True)

	logs_dir = sweep_dir / "logs"
	logs_dir.mkdir(exist_ok=True)

	slurm_dir = sweep_dir / "slurm"
	slurm_dir.mkdir(exist_ok=True)

	results_dir = sweep_dir / "results"
	results_dir.mkdir(exist_ok=True)

	cases = build_ofat_cases(base_cfg)
	env = build_env()

	if sys.executable:
		python_executable = Path(sys.executable).resolve()
	else:
		candidate = shutil.which("python") or shutil.which("python3")
		if candidate is None:
			raise RuntimeError("Unable to determine python executable for SLURM script")
		python_executable = Path(candidate).resolve()

	print(f"Sweep directory: {sweep_dir}")
	print(f"Total cases: {len(cases)}")

	case_entries: list[dict[str, Any]] = []
	for idx, case in enumerate(cases, start=1):
		cfg = apply_overrides(base_cfg, case.overrides)
		case_path = sweep_dir / f"case_{idx:03d}.yaml"
		with case_path.open("w", encoding="utf-8") as handle:
			yaml.safe_dump(cfg, handle, sort_keys=False)
		case_entries.append(
			{
				"index": idx,
				"label": case.label,
				"config_path": str(case_path),
			}
		)

	n_batches = max(1, int(args.n_batches))
	batch_size = max(1, int(math.ceil(len(case_entries) / n_batches)))
	for batch_idx in range(n_batches):
		start = batch_idx * batch_size
		end = min(start + batch_size, len(case_entries))
		if start >= end:
			break
		batch_cases = case_entries[start:end]
		batch_path = sweep_dir / f"batch_{batch_idx:03d}.json"
		batch_path.write_text(json.dumps(batch_cases, indent=2) + "\n", encoding="utf-8")

		job_name = f"sweep_b{batch_idx:02d}"
		slurm_text = _render_slurm_script(
			job_name=job_name,
			batch_path=batch_path,
			sim_type=args.sim_type,
			results_dir=results_dir,
			python_executable=python_executable,
			partition=args.partition,
			requested_mem=args.mem,
			requested_time=args.time,
		)
		script_path = slurm_dir / f"{job_name}.slurm"
		script_path.write_text(slurm_text, encoding="utf-8")

	env_for_submit = os.environ.copy()
	env_for_submit.update(env)

	print(f"Artifacts written to {sweep_dir}")

	if args.no_submit:
		print("Skipping submission (--no_submit set).")
		return

	sbatch = shutil.which("sbatch")
	if sbatch is None:
		raise RuntimeError("sbatch not found on PATH. Run this on your cluster login node.")

	for script_path in sorted(slurm_dir.glob("*.slurm")):
		result = subprocess.run(
			[sbatch, str(script_path)],
			cwd=str(sweep_dir),
			env=env_for_submit,
			capture_output=True,
			text=True,
			check=True,
		)
		print(f"  {script_path.name}: {result.stdout.strip()}")

	print("Done. Use --collect once jobs finish to aggregate runtimes.")


if __name__ == "__main__":
	main()
