"""Stack/average inhomogeneous 1D configurations produced by calc_datas.py.

Usage:
    python stack_inhomogenity.py --abs_path "<path to one _data.npz>" [--skip_if_exists]

Given one file path from an inhomogeneous batch (same t_coh, same group id),
this script finds all sibling files in the same directory tree with matching
`inhom_group_id`, loads the individual 1D arrays per signal type, averages
them over the inhomogeneous configurations, and writes a new file containing
the averaged result.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List
import numpy as np

from qspectro2d import (
    load_simulation_data,
    save_simulation_data,
)
from qspectro2d.utils.data_io import collect_group_files
from qspectro2d.utils import generate_deterministic_data_base

from thesis_paths import DATA_DIR


def average_inhom_1d(abs_path: Path, *, skip_if_exists: bool = False) -> Path:
    """Average all 1D arrays across inhomogeneous configs for current group.

    Returns the path to the newly written averaged file.
    """
    files = collect_group_files(Path(abs_path))

    # Load first to get axes and metadata
    first = None
    first_file_idx = 0

    # Find the first loadable file for metadata
    while first is None and first_file_idx < len(files):
        try:
            first = load_simulation_data(files[first_file_idx])
        except Exception as e:
            print(f"⚠️  Skipping corrupted file during metadata loading: {files[first_file_idx]}")
            print(f"    Error: {e}")
            first_file_idx += 1

    if first is None:
        raise FileNotFoundError("No valid files found for averaging")

    t_det = np.asarray(first["t_det"], dtype=float)
    signal_types: List[str] = list(map(str, first["signal_types"]))

    # Collect arrays per type
    stacks: dict[str, List[np.ndarray]] = {k: [] for k in signal_types}
    valid_files = []

    for f in files:
        try:
            d = load_simulation_data(f)
            # sanity: axes must match shape
            if not np.allclose(d["t_det"], t_det):
                print(f"⚠️  Skipping file with mismatched t_det axis: {f}")
                continue
            for k in signal_types:
                arr = np.asarray(d[k])
                if arr.shape != (t_det.size,):
                    print(f"⚠️  Skipping file with unexpected array shape for {k}: {f}")
                    continue
                stacks[k].append(arr)
            valid_files.append(f)
        except Exception as e:
            print(f"⚠️  Skipping corrupted file during averaging: {f}")
            print(f"    Error: {e}")
            continue

    if not valid_files:
        raise FileNotFoundError("No valid files found for averaging")

    print(f"📊 Averaging {len(valid_files)} valid files out of {len(files)} total files")

    # Average
    averaged: List[np.ndarray] = []
    for k in signal_types:
        data = np.stack(stacks[k], axis=0)  # (n_files, t_det)
        averaged.append(np.mean(data, axis=0))

    # Compose metadata for output
    metadata = {
        "signal_types": signal_types,
        "t_coh_value": first.get("t_coh_value", None),
        "inhom_group_id": first.get("inhom_group_id", None),
    }

    # Use original module info to build save context
    first_bundle = load_simulation_data(valid_files[0])

    class _SimModuleStub:
        # Minimal stub to satisfy save_simulation_data interface
        def __init__(self, bundle: dict):
            self.system = bundle.get("system")
            self.bath = bundle.get("bath")
            self.laser = bundle.get("laser")
            self.simulation_config = bundle.get("sim_config")
            # Normalize config bookkeeping for averaged output:
            cfg = self.simulation_config
            # Use canonical index 0 for averaged file naming
            cfg.inhom_index = 0
            cfg.inhom_averaged = True
            cfg.inhom_enabled = True

    sim_inhom_stacked = _SimModuleStub(first_bundle)

    # If skip_if_exists, check whether an averaged file already exists for this group and t_coh.
    # We detect existence by attempting to build the same base name the saver would create;
    # since generate_unique_data_filename uses system+sim_config, we can't reconstruct it here.
    # Instead, we look alongside input files for a sibling averaged file with the same directory
    # Using new naming scheme with 'inhom_avg' prefix in filename.
    if skip_if_exists:
        # Build deterministic (non-enumerated) base for averaged output
        det_base = generate_deterministic_data_base(
            sim_inhom_stacked.system,
            sim_inhom_stacked.simulation_config,
            data_root=DATA_DIR,
            ensure=True,
        )
        folder = det_base.parent
        pattern = det_base.name + "*_data.npz"  # matches base_data.npz or base_1_data.npz etc.
        for p in folder.glob(pattern):
            try:
                d = load_simulation_data(p)
                if d.get("inhom_averaged", False) and d.get("inhom_group_id") == metadata.get(
                    "inhom_group_id"
                ):
                    print(f"⏭️  Averaged file already exists for this t_coh in folder: {p}")
                    return p
            except Exception:
                continue

    out_path = save_simulation_data(
        sim_inhom_stacked, metadata, averaged, t_det, data_root=DATA_DIR
    )
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Average inhomogeneous 1D configs into one file.")
    p.add_argument("--abs_path", type=str, required=True, help="Path to one *_data.npz file")
    p.add_argument(
        "--skip_if_exists",
        action="store_true",
        help="Do not overwrite existing averaged file",
    )
    args = p.parse_args()

    out = average_inhom_1d(Path(args.abs_path), skip_if_exists=args.skip_if_exists)
    print(f"Saved inhom-averaged dataset: {out}")
    print(f"\n🎯 To plot the 1D data, run:")
    print(f"python plot_datas.py --abs_path {out}")


if __name__ == "__main__":
    main()
