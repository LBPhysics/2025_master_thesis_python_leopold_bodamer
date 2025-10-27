"""Execute a batch of (t_coh, inhomogeneity) combinations for spectroscopy runs.

This worker script is intended to be called from generated SLURM jobs. It receives a
list of combinations produced by ``hpc_batch_dispatch.py`` and performs the
actual field computations, saving the resulting datasets to the standard ``data``
folder of the repository.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


from qspectro2d.spectroscopy.e_field_1d import parallel_compute_1d_e_comps
from qspectro2d.utils.data_io import (
    save_run_artifact,
)
from calc_datas import DATA_DIR

DATA_DIR.mkdir(exist_ok=True)


def _load_combinations(path: Path) -> list[dict[str, Any]]:
    """Load combination descriptors from JSON.

    The dispatcher writes either a bare list or an object with a ``combos`` key.
    """
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict) and "combos" in payload:
        combos = payload["combos"]
    else:
        combos = payload

    if not isinstance(combos, list):
        raise TypeError(f"Expected list of combinations, got {type(combos)!r}")

    return combos


def _iter_combos(subset: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for entry in subset:
        if not isinstance(entry, dict):
            raise TypeError("Each combination must be a dictionary")
        yield entry


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a batch of spectroscopy combinations (t_coh × inhom index)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--combos_file",
        type=str,
        required=True,
        help="JSON file with the combinations assigned to this batch",
    )
    parser.add_argument(
        "--samples_file",
        type=str,
        required=True,
        help="NumPy .npy file containing the sampled frequency grid (shape: n_inhom × n_atoms)",
    )
    parser.add_argument(
        "--time_cut",
        type=float,
        required=True,
        help="Maximum safe evolution time determined during solver diagnostics",
    )
    parser.add_argument(
        "--sim_type",
        choices=["0d", "1d", "2d"],
        required=True,
        help="Simulation dimensionality (affects job_metadata only)",
    )
    parser.add_argument(
        "--batch_id",
        type=int,
        default=0,
        help="Optional batch identifier for logging",
    )
    parser.add_argument(
        "--n_batches",
        type=int,
        default=1,
        help="Total number of batches (job_metadata only)",
    )
    args = parser.parse_args()

    combos_path = Path(args.combos_file).resolve()
    samples_path = Path(args.samples_file).resolve()
    job_dir = combos_path.parent
    job_metadata_path = job_dir / "job_metadata.json"
    job_metadata: dict[str, Any] | None = None
    if job_metadata_path.exists():
        with job_metadata_path.open("r", encoding="utf-8") as handle:
            job_metadata = json.load(handle)

    if job_metadata is None:
        raise ValueError("job_metadata.json not found")

    config_path = Path(job_metadata["config_path"])

    data_base_path = Path(job_metadata["data_base_path"])
    data_dir = data_base_path.parent
    prefix = data_base_path.name

    print("=" * 80)
    print("GENERALIZED BATCH RUNNER")
    print(f"Config: {config_path}")
    print(f"Combos file: {combos_path}")
    print(f"Samples file: {samples_path}")
    print(f"Output: {data_base_path}")

    combinations = _load_combinations(combos_path)
    if not combinations:
        print("No combinations provided; nothing to do.")
        return

    print(f"Loaded {len(combinations)} combination(s).")

    samples = np.load(samples_path)
    if samples.ndim != 2:
        raise ValueError(
            f"Expected samples array with shape (n_inhom, n_atoms); got {samples.shape}"
        )
    n_inhom, n_atoms = samples.shape

    samples_target = data_dir / f"{prefix}_samples.npy"
    if not samples_target.exists():
        shutil.copy2(samples_path, samples_target)

    batch_suffix = f"batch_{args.batch_id:03d}.json" if args.batch_id is not None else "batch.json"
    combos_target = data_dir / f"{prefix}_{batch_suffix}"
    if not combos_target.exists():
        shutil.copy2(combos_path, combos_target)

    signal_types = job_metadata["signal_types"]

    t_start = time.time()
    saved_paths: list[str] = []

    for combo in _iter_combos(combinations):
        t_idx = combo.get("t_index", 0)
        inhom_idx = combo.get("inhom_index")
        global_idx = combo.get("index", len(saved_paths))
        t_coh_val = combo.get("t_coh_value")

        if inhom_idx < 0 or inhom_idx >= n_inhom:
            raise IndexError(
                f"inhom_index {inhom_idx} out of range for {n_inhom} inhomogeneous samples"
            )

        # Update simulation configuration for this combination
        freq_vector = samples[inhom_idx, :].astype(float)

        print(
            f"\n--- combo {global_idx} / {len(combinations)}: t_idx={t_idx}, t_coh={t_coh_val:.4f} fs, "
            f"inhom_idx={inhom_idx} ---"
        )

        e_components = parallel_compute_1d_e_comps(
            config_path=str(config_path),
            t_coh=t_coh_val,
            freq_vector=freq_vector.tolist(),
            time_cut=args.time_cut,
        )

        metadata_combo = {
            "signal_types": signal_types,
            "t_coh_value": t_coh_val,
            "t_index": t_idx,
            "combination_index": global_idx,
            "sim_type": "1d" if args.sim_type == "2d" else args.sim_type,
            "batch_id": args.batch_id,
            "sample_index": inhom_idx,
        }

        path = save_run_artifact(
            signal_arrays=e_components,
            metadata=metadata_combo,
            frequency_sample_cm=freq_vector,
            data_dir=data_dir,
            filename=f"{prefix}_run_t{t_idx:03d}_s{inhom_idx:03d}.npz",
        )

        saved_paths.append(str(path))
        print(f"    ✅ saved {path}")

    elapsed = time.time() - t_start
    print("=" * 80)
    print(
        f"Completed {len(saved_paths)} combination(s) in {elapsed:.2f} s | "
        f"batch_id={args.batch_id}"
    )
    if saved_paths:
        print("Latest artifact:")
        print(f"  {saved_paths[-1]}")
    print("=" * 80)

    # If this is the last batch, suggest post-processing
    if (
        args.batch_id is not None
        and args.n_batches is not None
        and args.batch_id == args.n_batches - 1
    ):
        # Find the job directory by going up from combos_file
        combos_dir = Path(args.combos_file).parent
        job_dir = combos_dir.parent  # batch_jobs/<job_label>
        print("\n🎯 All batches completed! Finalize and queue plotting from SCRIPTS_DIR with:")
        print(f"python hpc_plot_datas.py --job_dir {job_dir.resolve()}")


if __name__ == "__main__":
    main()
