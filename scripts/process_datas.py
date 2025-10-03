"""Process spectroscopy data: stack per sample and average across samples.

This script automates the post-processing of simulation results from calc_datas.py:
- Groups 1D artifacts by inhomogeneity sample.
- Stacks each group into 2D if multiple coherence times exist.
- Averages the resulting 2D (or 1D) datasets across samples.
- Outputs the final inhomogeneity-averaged spectrum.

Examples:
    python process_datas.py --abs_path '/path/to/any/artifact.npz'
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from qspectro2d.utils.data_io import load_run_artifact, save_run_artifact, save_info_file, split_prefix

SCRIPTS_DIR = Path(__file__).parent.resolve()
for _parent in SCRIPTS_DIR.parents:
    if (_parent / ".git").is_dir():
        PROJECT_ROOT = _parent
        break
DATA_DIR = (PROJECT_ROOT / "data").resolve()
DATA_DIR.mkdir(exist_ok=True)


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
    job_metadata: dict[str, Any] | None
    t_coh: np.ndarray | None = None


def _load_entry(path: Path) -> RunEntry:
    artifact = load_run_artifact(path)

    metadata = dict(artifact["metadata"])
    if "t_coh_value" in metadata:
        metadata["t_coh_value"] = float(np.asarray(metadata["t_coh_value"]))

    signals = {key: np.asarray(val) for key, val in artifact["signals"].items()}
    t_det = np.asarray(artifact["t_det"], dtype=float)
    freq_sample = np.asarray(artifact["frequency_sample_cm"], dtype=float)
    t_coh = np.asarray(artifact.get("t_coh", []), dtype=float) if artifact.get("t_coh") is not None else None

    sim_cfg = artifact["simulation_config"]
    system = artifact["system"]
    if sim_cfg is None or system is None:
        raise ValueError(f"Artifact {path} is missing simulation context")

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
        job_metadata=artifact.get("job_metadata"),
        t_coh=t_coh,
    )


def _discover_entries(anchor: RunEntry) -> list[RunEntry]:
    """Find all 1D artifacts in the same directory."""
    directory = anchor.path.parent

    entries: list[RunEntry] = []
    for candidate in sorted(directory.glob("*_run_t*_s*.npz")):
        entry = _load_entry(candidate)
        # Only process 1D artifacts, not already processed 2D
        if entry.simulation_config.sim_type == "2d":
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

    # Sort by t_index
    group_sorted = sorted(group, key=lambda e: e.metadata.get("t_index", 0))

    # Check consistency
    reference = group_sorted[0]
    t_det = reference.t_det
    signal_types = list(reference.metadata.get("signal_types", reference.signals.keys()))
    freq_ref = reference.frequency_sample_cm

    for entry in group_sorted[1:]:
        if entry.t_det.shape != t_det.shape or not np.allclose(entry.t_det, t_det):
            raise ValueError(f"Inconsistent t_det for {entry.path}")
        if not np.allclose(entry.frequency_sample_cm, freq_ref):
            raise ValueError(f"Inconsistent frequency for {entry.path}")
        current_signals = list(entry.metadata.get("signal_types", entry.signals.keys()))
        if current_signals != signal_types:
            raise ValueError(f"Inconsistent signals for {entry.path}")

    # Stack signals
    stacked_signals = {
        sig: np.stack([entry.signals[sig] for entry in group_sorted], axis=0) for sig in signal_types
    }

    t_coh_values = [entry.metadata.get("t_coh_value", 0.0) for entry in group_sorted]
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
            "stacked_points": len(group_sorted),
        }
    )
    metadata_2d.pop("t_coh_value", None)
    metadata_2d.pop("t_index", None)
    metadata_2d.pop("combination_index", None)

    sim_cfg = replace(
        reference.simulation_config,
        sim_type="2d",
        t_coh=None,
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
        job_metadata=reference.job_metadata,
        t_coh=t_coh_axis,
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
            job_metadata=single.job_metadata,
            t_coh=single.t_coh,
        )

    reference = entries[0]
    is_2d = reference.simulation_config.sim_type == "2d"

    if is_2d:
        # Check 2D consistency
        t_det = reference.t_det
        t_coh = reference.t_coh
        signal_types = list(reference.metadata.get("signal_types", reference.signals.keys()))

        for entry in entries[1:]:
            if entry.t_coh.shape != t_coh.shape or not np.allclose(entry.t_coh, t_coh):
                raise ValueError(f"Inconsistent t_coh for {entry.path}")

        # Average signals
        averaged_signals = {
            sig: np.mean(np.stack([entry.signals[sig] for entry in entries], axis=0), axis=0)
            for sig in signal_types
        }

        avg_freq = np.mean(np.stack([entry.frequency_sample_cm for entry in entries], axis=0), axis=0)

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
            t_coh=t_coh,
        )
    else:
        # 1D averaging (reuse logic from avg_inhomogenity.py)
        t_det = reference.t_det
        signal_types = list(reference.metadata.get("signal_types", reference.signals.keys()))

        for entry in entries[1:]:
            if entry.t_det.shape != t_det.shape or not np.allclose(entry.t_det, t_det):
                raise ValueError(f"Inconsistent t_det for {entry.path}")
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
    abs_path = abs_path.expanduser().resolve()
    anchor = _load_entry(abs_path)

    entries = _discover_entries(anchor)
    if not entries:
        raise FileNotFoundError("No 1D artifacts found for processing")

    # Group by sample
    groups = _group_by_sample(entries)

    # Stack each group to 2D if needed
    processed_per_sample: list[RunEntry] = []
    num_stacked = 0
    for sample_idx, group in groups.items():
        if len(group) > 1:
            stacked = _stack_group_to_2d(group)
            processed_per_sample.append(stacked)
            num_stacked += 1
        else:
            processed_per_sample.append(group[0])

    # Average across samples
    final_entry = _average_entries(processed_per_sample)

    # Save the final artifact
    directory, prefix = split_prefix(anchor.path)
    if final_entry.simulation_config.sim_type == "2d":
        final_filename = f"2d_inhom_averaged.npz"
        t_coh_for_save = final_entry.t_coh
    else:
        final_filename = f"1d_inhom_averaged.npz"
        t_coh_for_save = final_entry.metadata.get("t_coh_value")

    final_path = directory / final_filename

    if skip_if_exists and final_path.exists():
        print(f"⏭️  Final averaged artifact already exists: {final_path}")
        return final_path

    # Save info if needed
    info_path = directory / final_filename.replace('.npz', '.pkl')
    if not info_path.exists():
        save_info_file(
            info_path,
            final_entry.system,
            final_entry.simulation_config,
            bath=final_entry.bath,
            laser=final_entry.laser,
            extra_payload=final_entry.job_metadata,
        )

    out_path = save_run_artifact(
        signal_arrays=[final_entry.signals[sig] for sig in final_entry.signals],
        t_det=final_entry.t_det,
        metadata=final_entry.metadata,
        frequency_sample_cm=final_entry.frequency_sample_cm,
        data_dir=directory,
        filename=final_filename,
        t_coh=t_coh_for_save,
    )

    print(f"✅ Processed and saved final averaged artifact: {out_path}\n")
    print(f"   (processed {len(entries)} files, stacked {num_stacked} groups, averaged {len(processed_per_sample)} samples)\n")
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
    print(f"python plot_datas.py --abs_path {processed_path}")


if __name__ == "__main__":
    main()
