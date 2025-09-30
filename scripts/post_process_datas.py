#!/usr/bin/env python3
"""Post-process generalized batch outputs: average inhomogeneity and stack to 2D.

This script automates the averaging of inhomogeneous 1D results and stacking
into 2D datasets after all SLURM batches have completed.

Usage:
    python post_process_datas.py --job_dir <path_to_batch_jobs_generalized/job_label>

It reads the metadata.json from the job directory, discovers all generated
data files with matching inhom_group_id, averages over inhomogeneity per t_coh,
and stacks the results into a final 2D dataset.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from qspectro2d.utils.data_io import load_simulation_data, save_simulation_data


def _load_metadata(job_dir: Path) -> dict:
    """Load dispatcher metadata from job directory."""
    metadata_path = job_dir / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_data_files(output_root: Path, inhom_group_id: str) -> List[Path]:
    """Find all data files with matching inhom_group_id."""
    files = []
    for npz_file in output_root.rglob("*_data*.npz"):
        try:
            data = load_simulation_data(npz_file)
            if str(data.get("inhom_group_id", "")) == inhom_group_id:
                files.append(npz_file)
        except Exception:
            continue  # Skip corrupted files
    return sorted(files)


def _group_by_t_coh(files: List[Path]) -> Dict[float, List[Path]]:
    """Group files by t_coh_value."""
    groups: Dict[float, List[Path]] = defaultdict(list)
    for file_path in files:
        try:
            data = load_simulation_data(file_path)
            t_coh = float(data["t_coh_value"])
            groups[t_coh].append(file_path)
        except Exception:
            continue
    return dict(groups)


def _average_inhom(files: List[Path]) -> tuple[np.ndarray, dict]:
    """Average inhomogeneous files for a single t_coh."""
    if not files:
        raise ValueError("No files to average")

    # Load first file for metadata
    first_data = load_simulation_data(files[0])
    signal_types = list(first_data["signal_types"])
    t_det = np.asarray(first_data["t_det"], dtype=float)

    # Collect arrays
    stacks = {sig: [] for sig in signal_types}
    for file_path in files:
        data = load_simulation_data(file_path)
        if not np.allclose(data["t_det"], t_det):
            raise ValueError(f"Inconsistent t_det in {file_path}")
        for sig in signal_types:
            stacks[sig].append(np.asarray(data[sig]))

    # Average
    averaged = [np.mean(np.stack(stacks[sig]), axis=0) for sig in signal_types]

    metadata = {
        "signal_types": signal_types,
        "t_coh_value": first_data["t_coh_value"],
        "inhom_group_id": first_data.get("inhom_group_id"),
        "inhom_averaged": True,
    }

    return averaged, metadata


def _stack_to_2d(averaged_data: List[tuple[np.ndarray, dict]], t_det: np.ndarray) -> tuple[np.ndarray, np.ndarray, List[str]]:
    """Stack averaged 1D data into 2D."""
    if not averaged_data:
        raise ValueError("No data to stack")

    # Sort by t_coh
    averaged_data.sort(key=lambda x: x[1]["t_coh_value"])
    t_coh_vals = [meta["t_coh_value"] for _, meta in averaged_data]
    signal_types = averaged_data[0][1]["signal_types"]

    # Stack
    stacked = {}
    for sig in signal_types:
        arrays = [data[signal_types.index(sig)] for data, _ in averaged_data]
        stacked[sig] = np.stack(arrays)

    t_coh = np.array(t_coh_vals, dtype=float)

    return t_coh, stacked, signal_types


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-process generalized batch outputs into 2D dataset"
    )
    parser.add_argument(
        "--job_dir",
        type=str,
        required=True,
        help="Path to the batch job directory (containing metadata.json)",
    )
    args = parser.parse_args()

    job_dir = Path(args.job_dir).resolve()
    print("=" * 80)
    print("POST-PROCESS GENERALIZED BATCHES")
    print(f"Job directory: {job_dir}")

    # Load metadata
    metadata = _load_metadata(job_dir)
    output_root = Path(metadata["output_root"])
    inhom_group_id = str(metadata.get("inhom_group_id", ""))
    sim_type = metadata["sim_type"]

    print(f"Output root: {output_root}")
    print(f"Inhom group ID: {inhom_group_id}")
    print(f"Sim type: {sim_type}")

    # Find data files
    data_files = _find_data_files(output_root, inhom_group_id)
    print(f"Found {len(data_files)} data files")

    if not data_files:
        print("No data files found. Check if batches completed successfully.")
        return

    # Group by t_coh
    t_coh_groups = _group_by_t_coh(data_files)
    print(f"Grouped into {len(t_coh_groups)} t_coh values")

    # Average inhomogeneity per t_coh
    averaged_data = []
    for t_coh, files in t_coh_groups.items():
        print(f"Averaging {len(files)} files for t_coh={t_coh}")
        avg_data, avg_meta = _average_inhom(files)
        averaged_data.append((avg_data, avg_meta))

    if len(averaged_data) == 1 and sim_type == "1d":
        # For 1D, just save the averaged file
        first_data, first_meta = averaged_data[0]
        # Need to create a sim stub for saving
        # For simplicity, load from one of the original files
        original_data = load_simulation_data(data_files[0])
        class SimStub:
            system = original_data["system"]
            bath = original_data["bath"]
            laser = original_data["laser"]
            simulation_config = original_data["sim_config"]
            simulation_config.inhom_averaged = True
            simulation_config.inhom_index = None

        sim = SimStub()
        out_path = save_simulation_data(
            sim,
            first_meta,
            first_data,
            np.asarray(original_data["t_det"]),
            data_root=output_root,
        )
        print(f"Saved averaged 1D dataset: {out_path}")
        print("=" * 80)
        print("\n🎯 To plot the data, run:")
        print(f"python plot_datas.py --abs_path {out_path}")
        return

    # Stack to 2D
    print("Stacking to 2D...")
    original_data = load_simulation_data(data_files[0])  # For sim stub
    t_coh, stacked, signal_types = _stack_to_2d(averaged_data, np.asarray(original_data["t_det"]))

    # Create sim stub for 2D
    class SimStub2D:
        system = original_data["system"]
        bath = original_data["bath"]
        laser = original_data["laser"]
        simulation_config = original_data["sim_config"]
        simulation_config.sim_type = "2d"
        simulation_config.t_coh = None
        simulation_config.inhom_averaged = True
        simulation_config.inhom_index = None

    sim_2d = SimStub2D()
    metadata_2d = {"signal_types": signal_types}

    out_path = save_simulation_data(
        sim_2d,
        metadata_2d,
        [stacked[sig] for sig in signal_types],
        np.asarray(original_data["t_det"]),
        t_coh=t_coh,
        data_root=output_root,
    )

    print(f"Saved 2D dataset: {out_path}")
    print("=" * 80)
    print("\n🎯 To plot the data, run:")
    print(f"python plot_datas.py --abs_path {out_path}")


if __name__ == "__main__":
    main()