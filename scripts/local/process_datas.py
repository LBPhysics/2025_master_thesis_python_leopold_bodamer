"""Process spectroscopy data: stack per sample and average across samples.

This script automates the post-processing of simulation results from calc_datas.py:
- Groups 1D artifacts by inhomogeneity sample.
- Stacks each group into 2D if multiple coherence times exist.
- Averages the resulting 2D (or 1D) datasets across samples.
- Outputs the final inhomogeneity-averaged spectrum.

Examples:
	python scripts/local/process_datas.py --abs_path '/path/to/any/artifact.npz'
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
import time
from typing import Any
import numpy as np

from qspectro2d.utils.data_io import (
	load_run_artifact,
	save_run_artifact,
	save_info_file,
	split_prefix,
)
from qspectro2d.core.simulation.time_axes import compute_t_det

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
for _parent in SCRIPTS_DIR.parents:
	if (_parent / ".git").is_dir():
		PROJECT_ROOT = _parent
		break
else:
	raise RuntimeError("Could not locate project root (missing .git directory)")

DATA_DIR = (PROJECT_ROOT / "jobs").resolve()
DATA_DIR.mkdir(exist_ok=True)

print = partial(print, flush=True)

def _format_seconds(seconds: float) -> str:
	if seconds < 1:
		return f"{seconds * 1000:.1f} ms"
	if seconds < 60:
		return f"{seconds:.2f} s"
	mins, secs = divmod(seconds, 60)
	if mins < 60:
		return f"{int(mins)}m {secs:04.1f}s"
	hours, mins = divmod(mins, 60)
	return f"{int(hours)}h {int(mins)}m {secs:04.1f}s"


@dataclass(slots=True)
class RunEntry:
	path: Path
	metadata: dict[str, Any]
	signals: dict[str, np.ndarray]
	t_det: np.ndarray
	frequency_sample_cm: np.ndarray
	simulation_config: Any
	system: Any
	laser: Any | None
	bath: Any | None
	t_coh: np.ndarray | None = None
	job_metadata: dict[str, Any] | None = None


def _load_entry(path: Path) -> RunEntry:
	artifact = load_run_artifact(path)

	metadata = dict(artifact["metadata"])
	signals = {key: np.asarray(val) for key, val in artifact["signals"].items()}
	t_det = np.asarray(artifact["t_det"], dtype=float)
	freq_sample = np.asarray(artifact["frequency_sample_cm"], dtype=float)
	t_coh = (
		np.asarray(artifact.get("t_coh", []), dtype=float)
		if artifact.get("t_coh") is not None
		else None
	)

	sim_cfg = artifact["simulation_config"]
	system = artifact["system"]
	if sim_cfg is None or system is None:
		raise ValueError(f"Artifact {path} is missing simulation context")

	# Align detection axis with stored signal length if metadata drifted.
	if signals:
		det_len = next(iter(signals.values())).shape[-1]
		if t_det.size != det_len:
			cfg_for_det = deepcopy(sim_cfg)
			tc_val = metadata.get("t_coh_value")
			if tc_val is not None:
				cfg_for_det.t_coh_current = float(tc_val)
			new_t_det = compute_t_det(cfg_for_det)
			if new_t_det.size == det_len:
				t_det = new_t_det

	if sim_cfg.sim_type == "1d":
		t_coh = None

	return RunEntry(
		path=path,
		metadata=metadata,
		signals=signals,
		t_det=t_det,
		frequency_sample_cm=freq_sample,
		simulation_config=sim_cfg,
		system=system,
		laser=artifact.get("laser"),
		bath=artifact.get("bath"),
		t_coh=t_coh,
		job_metadata=artifact.get("job_metadata"),
	)


def _discover_entries(anchor: RunEntry) -> list[RunEntry]:
	"""Find all 1D artifacts in the same directory."""
	directory = anchor.path.parent

	entries: list[RunEntry] = []
	for candidate in sorted(directory.glob("*_run_t*_s*.npz")):
		entry = _load_entry(candidate)
		# Only process 1D artifacts, not already processed 2D
		if entry.metadata.get("sim_type") == "2d":
			continue
		entries.append(entry)

	return entries


def _group_by_sample(entries: list[RunEntry]) -> dict[int, list[RunEntry]]:
	"""Group entries by sample_index."""
	groups = defaultdict(list)
	for entry in entries:
		sample_idx = entry.metadata.get("sample_index", 0)
		groups[sample_idx].append(entry)
	return dict(groups)


def _stack_group_to_2d(group: list[RunEntry]) -> RunEntry:
	"""Stack a group of 1D entries into a single 2D entry."""
	if len(group) <= 1:
		return group[0]  # Already 1D

	# Sort by coherence time (more robust than t_index alone)
	group_sorted = sorted(group, key=lambda e: float(e.metadata.get("t_coh_value", 0.0)))

	# Check consistency
	reference = group_sorted[0]
	t_det = reference.t_det
	signal_types = list(reference.metadata.get("signal_types", reference.signals.keys()))
	freq_ref = reference.frequency_sample_cm
	ref_t_coh = reference.metadata.get("t_coh_value")
	if ref_t_coh is None:
		raise ValueError(f"Missing t_coh_value in metadata for {reference.path}")

	for entry in group_sorted[1:]:
		if not np.allclose(entry.frequency_sample_cm, freq_ref):
			raise ValueError(f"Inconsistent frequency for {entry.path}")
		current_signals = list(entry.metadata.get("signal_types", entry.signals.keys()))
		if current_signals != signal_types:
			raise ValueError(f"Inconsistent signals for {entry.path}")
		if entry.metadata.get("t_coh_value") is None:
			raise ValueError(f"Missing t_coh_value in metadata for {entry.path}")

	# Align to common detection-time window (no interpolation)
	start = max(float(entry.t_det[0]) for entry in group_sorted if entry.t_det.size)
	end = min(float(entry.t_det[-1]) for entry in group_sorted if entry.t_det.size)
	if start > end:
		raise ValueError("No overlapping t_det window across entries")

	# Build reference t_det grid from the first entry within the common window
	ref_mask = (reference.t_det >= start) & (reference.t_det <= end)
	if not np.any(ref_mask):
		raise ValueError(f"Reference entry {reference.path} has no samples in common t_det window")
	ref_t_det = reference.t_det[ref_mask]

	def _align_to_ref(entry: RunEntry) -> None:
		mask = (entry.t_det >= start) & (entry.t_det <= end)
		if not np.any(mask):
			raise ValueError(f"Entry {entry.path} has no samples in common t_det window")
		entry_t = entry.t_det[mask]
		# Map each ref_t_det to nearest index in entry_t
		idxs = np.searchsorted(entry_t, ref_t_det, side="left")
		idxs = np.clip(idxs, 0, len(entry_t) - 1)
		# Choose nearest of idx and idx-1
		best = np.zeros_like(idxs)
		for k, (i, t) in enumerate(zip(idxs, ref_t_det)):
			candidates = [i]
			if i > 0:
				candidates.append(i - 1)
			best_idx = min(candidates, key=lambda j: abs(entry_t[j] - t))
			best[k] = best_idx
		entry.t_det = ref_t_det
		for sig in signal_types:
			entry.signals[sig] = entry.signals[sig][mask][best]

	for entry in group_sorted:
		_align_to_ref(entry)

	# Update reference t_det after alignment
	t_det = ref_t_det

	# Collapse duplicate t_coh entries (can happen after re-runs)
	tol = 0.5 * float(reference.simulation_config.dt)
	collapsed: list[RunEntry] = []
	for entry in group_sorted:
		entry_t_coh = float(entry.metadata.get("t_coh_value", 0.0))
		if collapsed and np.isclose(entry_t_coh, float(collapsed[-1].metadata["t_coh_value"]), atol=tol):
			# Average signals for duplicate t_coh
			for sig in signal_types:
				collapsed[-1].signals[sig] = 0.5 * (
					collapsed[-1].signals[sig] + entry.signals[sig]
				)
			continue
		collapsed.append(entry)

	# Stack signals
	stacked_signals = {
		sig: np.stack([entry.signals[sig] for entry in collapsed], axis=0)
		for sig in signal_types
	}

	t_coh_values = [entry.metadata.get("t_coh_value", 0.0) for entry in collapsed]
	t_coh_axis = np.asarray(t_coh_values, dtype=float)

	# Sort by t_coh
	if not np.all(np.diff(t_coh_axis) >= 0):
		sort_idx = np.argsort(t_coh_axis)
		t_coh_axis = t_coh_axis[sort_idx]
		stacked_signals = {sig: arr[sort_idx] for sig, arr in stacked_signals.items()}

	# Create new metadata for 2D
	metadata_2d = dict(reference.metadata)
	metadata_2d.update(
		{
			"sim_type": "2d",
			"stacked_points": len(collapsed),
		}
	)
	metadata_2d.pop("t_coh_value", None)
	metadata_2d.pop("t_index", None)
	metadata_2d.pop("combination_index", None)

	sim_cfg = replace(
		reference.simulation_config,
		sim_type="2d",
		inhom_averaged=bool(reference.metadata.get("inhom_averaged")),
	)

	# Mock a 2D RunEntry
	return RunEntry(
		path=reference.path,  # Placeholder
		metadata=metadata_2d,
		signals=stacked_signals,
		t_det=t_det,
		frequency_sample_cm=freq_ref,
		simulation_config=sim_cfg,
		system=reference.system,
		laser=reference.laser,
		bath=reference.bath,
		t_coh=t_coh_axis,
		job_metadata=reference.job_metadata,
	)


def _average_entries(entries: list[RunEntry]) -> RunEntry:
	"""Average multiple entries (1D or 2D)."""
	if len(entries) == 1:
		single = entries[0]
		if single.simulation_config.inhom_averaged:
			return single
		# Set inhom_averaged=True for single entry
		sim_cfg = replace(single.simulation_config, inhom_averaged=True)
		metadata_out = dict(single.metadata)
		metadata_out.update({"inhom_averaged": True, "averaged_count": 1})
		metadata_out.pop("sample_index", None)
		return RunEntry(
			path=single.path,
			metadata=metadata_out,
			signals=single.signals,
			t_det=single.t_det,
			frequency_sample_cm=single.frequency_sample_cm,
			simulation_config=sim_cfg,
			system=single.system,
			laser=single.laser,
			bath=single.bath,
			t_coh=single.t_coh,
			job_metadata=single.job_metadata,
		)

	reference = entries[0]
	is_2d = reference.simulation_config.sim_type == "2d"

	if is_2d:
		# Check 2D consistency
		t_det = reference.t_det
		t_coh = reference.t_coh
		signal_types = list(reference.metadata.get("signal_types", reference.signals.keys()))
		if t_coh is None:
			raise ValueError("2D averaging requires a t_coh axis, but reference entry has None")

		for entry in entries[1:]:
			if entry.t_coh is None:
				raise ValueError(f"Missing t_coh axis in {entry.path}")
			if entry.t_coh.shape != t_coh.shape or not np.allclose(entry.t_coh, t_coh):
				raise ValueError(
					"Inconsistent t_coh axis across samples; "
					f"reference={reference.path}, offending={entry.path}"
				)

		# Average signals
		averaged_signals = {
			sig: np.mean(np.stack([entry.signals[sig] for entry in entries], axis=0), axis=0)
			for sig in signal_types
		}

		avg_freq = np.mean(
			np.stack([entry.frequency_sample_cm for entry in entries], axis=0), axis=0
		)

		metadata_out = dict(reference.metadata)
		metadata_out.update(
			{
				"inhom_averaged": True,
				"averaged_count": len(entries),
			}
		)
		metadata_out.pop("sample_index", None)

		sim_cfg = replace(
			reference.simulation_config,
			inhom_averaged=True,
		)

		return RunEntry(
			path=reference.path,
			metadata=metadata_out,
			signals=averaged_signals,
			t_det=t_det,
			frequency_sample_cm=avg_freq,
			simulation_config=sim_cfg,
			system=reference.system,
			laser=reference.laser,
			bath=reference.bath,
			t_coh=t_coh,
			job_metadata=reference.job_metadata,
		)
	else:
		# 1D averaging (reuse logic from avg_inhomogenity.py)
		t_det = reference.t_det
		signal_types = list(reference.metadata.get("signal_types", reference.signals.keys()))

		for entry in entries[1:]:
			current = list(entry.metadata.get("signal_types", entry.signals.keys()))
			if current != signal_types:
				raise ValueError(f"Inconsistent signals for {entry.path}")

		data_stack = {
			sig: np.stack([entry.signals[sig] for entry in entries], axis=0) for sig in signal_types
		}
		averaged_signals = {sig: np.mean(stack, axis=0) for sig, stack in data_stack.items()}

		freq_stack = np.stack([entry.frequency_sample_cm for entry in entries], axis=0)
		avg_freq = np.mean(freq_stack, axis=0)

		metadata_out = dict(reference.metadata)
		metadata_out.update(
			{
				"inhom_averaged": True,
				"averaged_count": len(entries),
			}
		)
		metadata_out.pop("sample_index", None)
		metadata_out.pop("combination_index", None)

		sim_cfg = replace(
			reference.simulation_config,
			inhom_averaged=True,
		)

		return RunEntry(
			path=reference.path,  # Placeholder
			metadata=metadata_out,
			signals=averaged_signals,
			t_det=t_det,
			frequency_sample_cm=avg_freq,
			simulation_config=sim_cfg,
			system=reference.system,
			laser=reference.laser,
			bath=reference.bath,
			job_metadata=reference.job_metadata,
		)


def process_datas(abs_path: Path, *, skip_if_exists: bool = False) -> Path:
	start_all = time.perf_counter()
	abs_path = abs_path.expanduser().resolve()
	print(f"Starting process_datas for: {abs_path}", flush=True)
	print(f"Skip if exists: {skip_if_exists}", flush=True)

	step_start = time.perf_counter()
	anchor = _load_entry(abs_path)
	print(
		"Loaded anchor artifact: "
		f"signals={list(anchor.signals.keys())}, "
		f"t_det={anchor.t_det.shape}, "
		f"t_coh={'None' if anchor.t_coh is None else anchor.t_coh.shape}"
	)
	print(f"Anchor load time: {_format_seconds(time.perf_counter() - step_start)}")

	step_start = time.perf_counter()
	entries = _discover_entries(anchor)
	print(f"Discovered {len(entries)} input artifacts")
	if not entries:
		raise FileNotFoundError("No 1D artifacts found for processing")
	print(f"Discovery time: {_format_seconds(time.perf_counter() - step_start)}")

	# Group by sample
	step_start = time.perf_counter()
	groups = _group_by_sample(entries)
	group_sizes = sorted((sample, len(group)) for sample, group in groups.items())
	print(f"Grouped into {len(groups)} samples: {group_sizes}")
	print(f"Grouping time: {_format_seconds(time.perf_counter() - step_start)}")

	# Stack each group to 2D if needed
	processed_per_sample: list[RunEntry] = []
	for sample_idx, group in groups.items():
		group_start = time.perf_counter()
		if len(group) > 1:
			print(f"Stacking sample {sample_idx}: {len(group)} artifacts")
			stacked = _stack_group_to_2d(group)
			processed_per_sample.append(stacked)
			print(
				f"Stacked sample {sample_idx}: t_coh={stacked.t_coh.shape if stacked.t_coh is not None else 'None'}, "
				f"signals={[sig.shape for sig in stacked.signals.values()]}"
			)
		else:
			processed_per_sample.append(group[0])
			print(f"Sample {sample_idx}: single artifact (no stacking)")
		print(
			f"Sample {sample_idx} processing time: {_format_seconds(time.perf_counter() - group_start)}"
		)

	# Average across samples
	step_start = time.perf_counter()
	final_entry = _average_entries(processed_per_sample)
	print(
		"Averaging complete: "
		f"sim_type={final_entry.simulation_config.sim_type}, "
		f"averaged_count={final_entry.metadata.get('averaged_count')}, "
		f"inhom_averaged={final_entry.metadata.get('inhom_averaged')}"
	)
	print(f"Averaging time: {_format_seconds(time.perf_counter() - step_start)}")

	# Save the final artifact
	directory, prefix = split_prefix(anchor.path)
	sim_type = final_entry.simulation_config.sim_type
	if sim_type == "2d":
		final_filename = f"2d_inhom_averaged.npz"
	else:
		final_filename = f"{sim_type}_inhom_averaged.npz"

	final_path = directory / final_filename

	if skip_if_exists and final_path.exists():
		print(f"⏭️  Final averaged artifact already exists: {final_path}")
		return final_path

	# Save info if needed
	info_path = directory / final_filename.replace(".npz", ".pkl")
	extra_payload: dict[str, Any] = {}
	if final_entry.job_metadata:
		extra_payload.update(final_entry.job_metadata)
	extra_payload.update({"t_det": final_entry.t_det, "t_coh": final_entry.t_coh})
	print(f"Writing info file: {info_path}")
	save_info_file(
		info_path,
		final_entry.system,
		final_entry.simulation_config,
		bath=final_entry.bath,
		laser=final_entry.laser,
		extra_payload=extra_payload,
	)

	out_path = save_run_artifact(
		signal_arrays=[final_entry.signals[sig] for sig in final_entry.signals],
		metadata=final_entry.metadata,
		frequency_sample_cm=final_entry.frequency_sample_cm,
		data_dir=directory,
		filename=final_filename,
	)

	# Get stacked points info
	stacked_points = (
		processed_per_sample[0].metadata.get("stacked_points", 1)
		if processed_per_sample and processed_per_sample[0].simulation_config.sim_type == "2d"
		else 1
	)

	print(f"✅ Processed and saved final averaged artifact: {out_path}")
	print(
		f"Processed {len(entries)} files, stacked {stacked_points} time points, "
		f"averaged {len(processed_per_sample)} samples"
	)
	print(f"Total time: {_format_seconds(time.perf_counter() - start_all)}")
	return out_path


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Process spectroscopy data: stack and average across inhomogeneity samples."
	)
	parser.add_argument("--abs_path", type=str, required=True, help="Path to any artifact (.npz)")
	parser.add_argument(
		"--skip_if_exists",
		action="store_true",
		help="Reuse existing final artifact if present",
	)
	args = parser.parse_args()

	processed_path = process_datas(Path(args.abs_path), skip_if_exists=args.skip_if_exists)
	print(f"Final processed artifact: {processed_path}")
	print("\n🎯 Plot with:")
	plot_script = (PROJECT_ROOT / "scripts" / "local" / "plot_datas.py").resolve()
	print(f"python \"{plot_script}\" --abs_path {processed_path}")


if __name__ == "__main__":
	main()
