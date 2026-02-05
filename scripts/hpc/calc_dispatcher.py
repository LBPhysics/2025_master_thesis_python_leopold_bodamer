"""Generate and (optionally) submit SLURM jobs for combined t_coh × inh

this dispatcher creates 1D/2D workflows by creating the full
Cartesian product of coherence times and inhomogeneous samples. It validates the
simulation locally before launching any jobs to an hpc cluster.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import shutil
import subprocess
import sys
import statistics
from datetime import datetime
from pathlib import Path
from typing import Sequence
import numpy as np
import yaml

from qspectro2d.config.create_sim_obj import load_simulation, load_simulation_config
from qspectro2d.spectroscopy import sample_from_gaussian
from qspectro2d.utils.data_io import save_info_file
from qspectro2d.utils.job_paths import allocate_job_dir, ensure_job_layout, job_label_token
from qspectro2d.core.simulation.time_axes import (
	compute_times_local,
	compute_t_coh,
	compute_t_det,
)

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
	sys.path.insert(0, str(SCRIPTS_DIR))

from local.calc_datas import (
	RUNS_ROOT,
	pick_config_yaml,
	build_combinations,
	write_json,
)

SWEEPS_ROOT = RUNS_ROOT / "sweeps"


def _set_random_seed(seed: int | None) -> None:
	if seed is not None:
		np.random.seed(seed)


def _normalize_config_without_solver(config: dict) -> str:
	cleaned = json.loads(json.dumps(config))
	cleaned.get("config", {}).pop("solver", None)
	return json.dumps(cleaned, sort_keys=True)


def _normalize_config_without_rwa(config: dict) -> str:
	cleaned = json.loads(json.dumps(config))
	cleaned.get("laser", {}).pop("rwa_sl", None)
	return json.dumps(cleaned, sort_keys=True)


def _find_latest_sweep_summary() -> Path | None:
	if not SWEEPS_ROOT.exists():
		return None
	summaries = list(SWEEPS_ROOT.glob("**/summary.json"))
	if not summaries:
		return None
	return max(summaries, key=lambda path: path.stat().st_mtime)


def estimate_slurm_resources(
	n_times: int,  # number of time steps in the local grid
	n_inhom: int,
	n_t_coh: int,  # number of coherence times -> how many combinations
	n_batches: int,
	*,
	workers: int = 1,
	N_dim: int,
	solver: str = "lindblad",
	mem_safety: float = 100.0,
	base_mb: float = 500.0,
	mem_per_combo_mb: float = 1.0,
	time_safety: float = 2.5,
	base_time: float = 0.0,
	rwa_sl: bool = True,
	summary_path: Path | None = None,
	base_t_override: float | None = None,
 ) -> tuple[str, str]:
	"""
	Estimate SLURM memory and runtime for QuTiP evolutions.

	Scaling model (per batch):
		time ~ base_t * solver_factor * rwa_factor
			   * (n_times / 1000) * (N_dim**2)
			   * (combos_per_batch / workers)
			   * time_safety
	"""
	# Number of total independent simulations
	combos_total = n_inhom * n_t_coh
	batches = max(1, n_batches)
	combos_per_batch = max(1, int(math.ceil(combos_total / batches)))

	# ---------------------- MEMORY ----------------------
	bytes_per_solver = n_times * (N_dim) * 16  # only store the expectation values
	total_bytes = mem_safety * workers * bytes_per_solver
	combos_mem_mb = min(combos_per_batch * mem_per_combo_mb, 1000.0)
	mem_mb = base_mb + total_bytes / (1024**2) + combos_mem_mb
	requested_mem = f"{int(math.ceil(mem_mb))}M"

	# ---------------------- TIME ------------------------

	# Empirical baseline: base_t s per combo for paper_eqs+RWA, 1 atom, n_times=1000, N=2
	base_t = 0.45
	solver_factor = {
		"paper_eqs": 1.0,
		"lindblad": 1.0,
		"redfield": 1.0,
	}
	# Conservative no-RWA factor (max observed across solvers)
	rwa_factor = 3.0 if not rwa_sl else 1.0

	# Optional calibration using a direct timing override
	if base_t_override is not None:
		base_t = float(base_t_override)
	# Optional calibration using sweep summary
	elif summary_path and summary_path.exists():
		try:
			summary_data = json.loads(summary_path.read_text(encoding="utf-8"))

			grouped_by_solver: dict[str, dict[str, float]] = {}
			grouped_by_rwa: dict[str, dict[bool, float]] = {}
			base_t_samples: list[float] = []

			for entry in summary_data:
				config_path = entry.get("config_path")
				runtime_s = entry.get("runtime_s")
				if not config_path or runtime_s is None:
					continue

				cfg_path = Path(config_path)
				if not cfg_path.exists():
					continue

				cfg_obj = load_simulation_config(str(cfg_path))
				cfg_dict = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
				solver_name = cfg_dict.get("config", {}).get("solver", "paper_eqs")
				rwa_val = bool(cfg_dict.get("laser", {}).get("rwa_sl", True))

				n_times_cfg = len(np.asarray(compute_times_local(cfg_obj), dtype=float))
				if solver_name == "paper_eqs" and n_times_cfg > 0:
					denom = (n_times_cfg / 1000) * (N_dim**2)
					if denom > 0:
						base_t_samples.append(float(runtime_s) / denom)

				solver_key = _normalize_config_without_solver(cfg_dict)
				grouped_by_solver.setdefault(solver_key, {})[solver_name] = float(runtime_s)

				rwa_key = _normalize_config_without_rwa(cfg_dict)
				grouped_by_rwa.setdefault(rwa_key, {})[rwa_val] = float(runtime_s)

			if base_t_samples:
				base_t = statistics.median(base_t_samples)

			redfield_ratios: list[float] = []
			lindblad_ratios: list[float] = []
			for group in grouped_by_solver.values():
				if "paper_eqs" in group:
					paper_runtime = group["paper_eqs"]
					if paper_runtime > 0 and "redfield" in group:
						redfield_ratios.append(group["redfield"] / paper_runtime)
					if paper_runtime > 0 and "lindblad" in group:
						lindblad_ratios.append(group["lindblad"] / paper_runtime)

			if redfield_ratios:
				solver_factor["redfield"] = max(redfield_ratios)
			if lindblad_ratios:
				solver_factor["lindblad"] = max(lindblad_ratios)

			rwa_ratios: list[float] = []
			for group in grouped_by_rwa.values():
				if True in group and False in group:
					true_runtime = group[True]
					false_runtime = group[False]
					if true_runtime > 0:
						rwa_ratios.append(false_runtime / true_runtime)
			if rwa_ratios:
				rwa_factor = max(rwa_ratios)

			print(
				"Calibrated from sweep: "
				f"base_t={base_t:.4g} s, "
				f"solver_factor={{paper_eqs: 1.0, lindblad: {solver_factor['lindblad']:.3g}, redfield: {solver_factor['redfield']:.3g}}}, "
				f"rwa_factor={rwa_factor:.3g}"
			)
		except Exception:
			pass

	if solver not in solver_factor:
		raise ValueError(f"Unsupported solver '{solver}'.")
	if rwa_sl:
		rwa_factor = 1.0

	# scaling ~ n_times * N^2  (sparse regime)
	if base_t_override is not None:
		time_per_combo = base_t * (n_times / 1000) * (N_dim**2)
	else:
		time_per_combo = (
			base_t
			* solver_factor[solver]
			* rwa_factor
			* (n_times / 1000)
			* (N_dim**2)
		)

	# total time for one batch (divide by workers)
	total_seconds = time_per_combo * combos_per_batch * time_safety / max(1, workers)

	# Ensure minimum time of 1 minute to avoid SLURM rejection
	total_seconds = max(total_seconds, base_time)

	# convert to HH:MM:SS, clip to max 24h if needed
	h = int(total_seconds // 3600)
	m = int((total_seconds % 3600) // 60)
	s = int(total_seconds % 60)
	# Cap at 3 days (72 hours) to fit GPGPU partition limit
	if h >= 72:
		h, m, s = 72, 0, 0
	requested_time = f"{h:02d}:{m:02d}:{s:02d}"

	return requested_mem, requested_time


def _split_indices(n_items: int, n_batches: int) -> list[np.ndarray]:
	if n_batches <= 0:
		raise ValueError("n_batches must be positive")
	if n_items == 0:
		return [np.array([], dtype=int) for _ in range(n_batches)]
	return [chunk.astype(int) for chunk in np.array_split(np.arange(n_items), n_batches)]


def _render_slurm_script(
	*,
	job_name: str,
	batch_idx: int,
	n_batches: int,
	sim_type: str,
	combos_filename: str,
	samples_filename: str,
	time_cut: float,
	worker_path: Path,
	python_executable: Path,
	requested_mem: str,
	requested_time: str,
) -> str:
	python_cmd = shlex.quote(str(python_executable))
	worker_arg = shlex.quote(str(worker_path))
	return f"""#!/bin/bash
	#SBATCH --job-name={job_name}
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --cpus-per-task=16
#SBATCH --mem={requested_mem}
#SBATCH --time={requested_time}
#SBATCH --partition=GPGPU,metis

set -euo pipefail

{python_cmd} {worker_arg} \
	--combos_file "{combos_filename}" \
	--samples_file "{samples_filename}" \
	--time_cut {time_cut:.12g} \
	--sim_type {sim_type} \
	--batch_id {batch_idx} \
	--n_batches {n_batches}
"""


def submit_sbatch(script_path: Path, cwd: Path | None = None) -> str:
	"""Submit a SLURM script via sbatch."""
	sbatch = shutil.which("sbatch")
	if sbatch is None:
		raise RuntimeError("sbatch not found on PATH. Run this on your cluster login node.")

	if cwd is None:
		cwd = script_path.parent

	try:
		result = subprocess.run(
			[sbatch, str(script_path)],
			cwd=str(cwd),
			capture_output=True,
			text=True,
			check=True,
		)
	except FileNotFoundError:
		raise RuntimeError("sbatch command not found. Submit the script manually.")
	except subprocess.CalledProcessError as exc:
		msg = exc.stderr.strip() or exc.stdout.strip() or str(exc)
		raise RuntimeError(f"sbatch failed with exit code {exc.returncode}: {msg}")

	return result.stdout.strip()


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Dispatch generalized spectroscopy batches to an HPC cluster",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument(
		"--config",
		type=str,
		default=None,
		help=(
			"Optional path to a YAML simulation config. If omitted, the default "
			"selection logic is used (first '_' prefixed file in scripts/simulation_configs)."
		),
	)
	parser.add_argument(
		"--sim_type",
		choices=["0d", "1d", "2d"],
		default="2d",
		help="Simulation dimensionality",
	)
	parser.add_argument(
		"--n_batches",
		type=int,
		default=1,
		help="Total number of batches to split the combination space into",
	)
	parser.add_argument(
		"--mem-safety",
		type=float,
		default=100.0,
		help="Safety factor for memory estimates",
	)
	parser.add_argument(
		"--time-safety",
		type=float,
		default=2.5,
		help="Safety factor for runtime estimates",
	)
	parser.add_argument(
		"--base-mb",
		type=float,
		default=500.0,
		help="Base memory overhead in MB",
	)
	parser.add_argument(
		"--mem-per-combo-mb",
		type=float,
		default=1.0,
		help="Additional MB per combo (capped internally)",
	)
	parser.add_argument(
		"--base-time",
		type=float,
		default=60.0,
		help="Minimum wall time in seconds",
	)
	parser.add_argument(
		"--summary-path",
		type=str,
		default=None,
		help="Optional sweep summary JSON to calibrate runtime scaling",
	)
	parser.add_argument(
		"--rng_seed",
		type=int,
		default=None,
		help="Optional NumPy random seed for reproducible sampling",
	)
	parser.add_argument(
		"--no_submit",
		action="store_true",
		help="Only generate local artifacts; skip sbatch submission",
	)
	return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
	args = _parse_args(sys.argv[1:] if argv is None else argv)

	if args.config:
		config_path = Path(args.config).expanduser().resolve()
		if not config_path.exists():
			raise FileNotFoundError(f"Config file not found: {config_path}")
	else:
		config_path = pick_config_yaml().resolve()

	print("=" * 80)
	print("GENERALIZED HPC DISPATCHER")
	print(f"Config path: {config_path}")

	sim = load_simulation(config_path, run_validation=True)
	print("✅ Simulation object constructed.")

	time_cut = np.inf  # TODO ONLY CHECK LOCALLY check_the_solver(sim)
	print(f"✅ Solver NOT validated on hpc -> do it locally! time_cut = {time_cut:.6g}")

	sim.simulation_config.sim_type = args.sim_type  # to ensure t_coh_axis has the right behavior

	n_inhom = sim.simulation_config.n_inhomogen
	if n_inhom <= 0:
		raise ValueError("n_inhom must be positive")

	_set_random_seed(args.rng_seed)
	base_freqs = np.asarray(sim.system.frequencies_cm, dtype=float)
	delta_cm = float(sim.system.delta_inhomogen_cm)
	samples = sample_from_gaussian(
		n_samples=n_inhom,
		fwhm=delta_cm,
		mu=base_freqs,
	)

	# Use SimulationModuleOQS.t_coh for coherence axis (handles 0d/1d/2d)
	t_coh_values = np.asarray(compute_t_coh(sim.simulation_config), dtype=float)
	times_local = np.asarray(compute_times_local(sim.simulation_config), dtype=float)

	combinations = build_combinations(t_coh_values, n_inhom)

	print(
		f"Prepared {len(combinations)} combination(s) → "
		f"|t_coh|={t_coh_values.size}, n_inhom={n_inhom}, n_batches={args.n_batches}"
	)

	label_token = job_label_token(sim.simulation_config, sim.system, sim_type=args.sim_type)
	timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
	job_label = f"hpc_{label_token}_{timestamp}"
	job_dir = allocate_job_dir(RUNS_ROOT, job_label)
	job_paths = ensure_job_layout(job_dir, base_name="raw")
	data_base_path = job_paths.data_base_path
	logs_dir = job_dir / "logs"
	logs_dir.mkdir(exist_ok=True)

	samples_file = job_dir / "samples.npy"
	np.save(samples_file, samples.astype(float))

	calibrated_base_t: float | None = None

	# Estimate RAM and TIME based on batch size
	if args.summary_path:
		summary_path = Path(args.summary_path).resolve()
	else:
		summary_path = _find_latest_sweep_summary()
		if summary_path is not None:
			print(f"Using latest sweep summary: {summary_path}")
	requested_mem, requested_time = estimate_slurm_resources(
		n_times=len(times_local),
		n_inhom=n_inhom,
		n_t_coh=len(t_coh_values),
		n_batches=args.n_batches,
		workers=16,  # BECAUSE I set every batch to use 16 CPUs
		N_dim=sim.system.dimension,
		solver=sim.simulation_config.ode_solver,
		rwa_sl=sim.simulation_config.rwa_sl,
		mem_safety=args.mem_safety,
		base_mb=args.base_mb,
		mem_per_combo_mb=args.mem_per_combo_mb,
		time_safety=args.time_safety,
		base_time=args.base_time,
		summary_path=summary_path,
		base_t_override=calibrated_base_t,
	)
	print(
		f"Requested resources: mem={requested_mem}, time={requested_time}, cpus=16"
	)

	# Save a copy of the config file to the job directory
	config_copy_path = job_dir / config_path.name
	if not config_copy_path.exists():
		shutil.copy2(config_path, config_copy_path)
		print(f"✅ Config file copied to {config_copy_path}")

	# Use the copied config path for subsequent operations
	config_path = config_copy_path

	job_metadata = {
		"sim_type": args.sim_type,
		"signal_types": sim.simulation_config.signal_types,
		"t_det": compute_t_det(sim.simulation_config).tolist(),
		"t_coh": t_coh_values.tolist(),
		"n_inhom": n_inhom,
		"n_t_coh": int(t_coh_values.size),
		"n_combinations": len(combinations),
		"n_batches": int(args.n_batches),
		"time_cut": float(time_cut),
		"job_label": job_label,
		"job_token": label_token,
		"generated_at": timestamp,
		"job_dir": str(job_paths.job_dir),
		"data_dir": str(job_paths.data_dir),
		"figures_dir": str(job_paths.figures_dir),
		"data_base_name": job_paths.base_name,
		"data_base_path": str(data_base_path),
		"config_path": str(config_path),
		"rng_seed": args.rng_seed,
		"cpus_per_task": 16,
		"requested_mem": requested_mem,
		"requested_time": requested_time,
	}
	write_json(job_dir / "job_metadata.json", job_metadata)

	info_path = data_base_path.parent / f"{data_base_path.name}.pkl"
	if not info_path.exists():
		save_info_file(
			info_path,
			sim.system,
			sim.simulation_config,
			bath=getattr(sim, "bath"),
			laser=getattr(sim, "laser"),
			extra_payload=job_metadata,
		)

	batch_indices = _split_indices(len(combinations), args.n_batches)
	script_paths: list[Path] = []
	worker_path = (SCRIPTS_DIR / "hpc" / "run_batch.py").resolve()
	if not worker_path.exists():
		raise FileNotFoundError(f"Missing worker script: {worker_path}")

	if sys.executable:
		python_executable = Path(sys.executable).resolve()
	else:
		candidate = shutil.which("python") or shutil.which("python3")
		if candidate is None:
			raise RuntimeError("Unable to determine python executable for SLURM script")
		python_executable = Path(candidate).resolve()
	for batch_idx, indices in enumerate(batch_indices):
		combos_subset = [combinations[i].to_dict() for i in indices.tolist()]
		combos_file = job_dir / f"batch_{batch_idx:03d}.json"
		write_json(combos_file, {"combos": combos_subset})

		job_name = f"{args.sim_type}b{batch_idx:02d}of{args.n_batches:02d}"
		slurm_text = _render_slurm_script(
			job_name=job_name,
			batch_idx=batch_idx,
			n_batches=args.n_batches,
			sim_type=args.sim_type,
			combos_filename=combos_file.name,
			samples_filename=samples_file.name,
			time_cut=time_cut,
			worker_path=worker_path,
			python_executable=python_executable,
			requested_mem=requested_mem,
			requested_time=requested_time,
		)
		script_path = job_dir / f"{job_name}.slurm"
		script_path.write_text(slurm_text, encoding="utf-8")
		script_paths.append(script_path)
		print(f"  batch {batch_idx}: {len(combos_subset)} combo(s) → {script_path.name}")

	print(f"Artifacts written to {job_dir}")

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
