#!/usr/bin/env python3
"""Execute a batch of (t_coh, inhomogeneity) combinations for spectroscopy runs.

This worker script is intended to be called from generated SLURM jobs. It receives a
list of combinations produced by ``hpc_dispatch_generalized.py`` and performs the
actual field computations, saving the resulting datasets to the standard ``data``
folder of the repository.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from qspectro2d.config.create_sim_obj import load_simulation
from qspectro2d.spectroscopy.e_field_1d import parallel_compute_1d_e_comps
from qspectro2d.utils.data_io import save_data_only, save_simulation_data


SCRIPTS_DIR = Path(__file__).parent.resolve()
for _parent in SCRIPTS_DIR.parents:
    if (_parent / ".git").is_dir():
        PROJECT_ROOT = _parent
        break
else:  # pragma: no cover - defensive branch
    raise RuntimeError("Could not locate project root (missing .git directory)")

DATA_DIR = (PROJECT_ROOT / "data").resolve()
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

    if not isinstance(combos, list):  # pragma: no cover - defensive
        raise TypeError(f"Expected list of combinations, got {type(combos)!r}")

    return combos


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Cannot convert {value!r} to float") from exc


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Cannot convert {value!r} to int") from exc


def _format_combo(combo: dict[str, Any]) -> str:
    parts = [
        f"t_idx={combo.get('t_index', '?')}",
        f"t_coh={combo.get('t_coh', '?')}",
        f"inhom={combo.get('inhom_index', '?')}",
        f"idx={combo.get('index', '?')}",
    ]
    return ", ".join(parts)


def _iter_combos(subset: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for entry in subset:
        if not isinstance(entry, dict):  # pragma: no cover - defensive
            raise TypeError("Each combination must be a dictionary")
        yield entry


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a batch of spectroscopy combinations (t_coh × inhom index)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Absolute or relative path to the YAML configuration used for the simulation",
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
        help="NumPy .npy file containing the sampled frequency grid (shape: n_inhom × n_sites)",
    )
    parser.add_argument(
        "--time_cut",
        type=float,
        required=True,
        help="Maximum safe evolution time determined during solver diagnostics",
    )
    parser.add_argument(
        "--sim_type",
        choices=["1d", "2d"],
        required=True,
        help="Simulation dimensionality (affects metadata only)",
    )
    parser.add_argument(
        "--batch_id",
        type=int,
        default=None,
        help="Optional batch identifier for logging",
    )
    parser.add_argument(
        "--n_batches",
        type=int,
        default=None,
        help="Total number of batches (metadata only)",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(DATA_DIR),
        help="Destination root directory for saved data",
    )
    parser.add_argument(
        "--revalidate",
        action="store_true",
        help="If set, rerun full solver validation on the worker node",
    )
    args = parser.parse_args()

    combos_path = Path(args.combos_file).resolve()
    samples_path = Path(args.samples_file).resolve()
    config_path = Path(args.config_path).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GENERALIZED BATCH RUNNER")
    print(f"Config: {config_path}")
    print(f"Combos file: {combos_path}")
    print(f"Samples file: {samples_path}")
    print(f"Output root: {output_root}")

    combinations = _load_combinations(combos_path)
    if not combinations:
        print("No combinations provided; nothing to do.")
        return

    print(f"Loaded {len(combinations)} combination(s).")

    samples = np.load(samples_path)
    if samples.ndim != 2:
        raise ValueError(
            f"Expected samples array with shape (n_inhom, n_sites); got {samples.shape}"
        )
    n_inhom, n_sites = samples.shape

    sim = load_simulation(config_path, validate=args.revalidate)
    sim.simulation_config.sim_type = args.sim_type
    sim.simulation_config.n_inhomogen = n_inhom
    sim.simulation_config.inhom_enabled = n_inhom > 1

    base_freqs = np.asarray(sim.system.frequencies_cm, dtype=float)
    if base_freqs.size != n_sites:
        raise ValueError(
            "Mismatch between sampled frequencies and system frequencies: "
            f"sample columns = {n_sites}, system sites = {base_freqs.size}"
        )

    t_start = time.time()
    saved_paths: list[str] = []
    first_save = True

    for combo in _iter_combos(combinations):
        t_idx = _coerce_int(combo.get("t_index", combo.get("t_idx", 0)))
        inhom_idx = _coerce_int(combo.get("inhom_index"))
        global_idx = _coerce_int(combo.get("index", len(saved_paths)))
        t_coh_val = _coerce_float(combo.get("t_coh"))

        if inhom_idx < 0 or inhom_idx >= n_inhom:
            raise IndexError(
                f"inhom_index {inhom_idx} out of range for {n_inhom} inhomogeneous samples"
            )

        # Update simulation configuration for this combination
        sim.simulation_config.inhom_index = inhom_idx
        sim.simulation_config.t_coh = t_coh_val
        sim.update_delays(t_coh=t_coh_val)

        freq_vector = samples[inhom_idx, :].astype(float)
        sim.system.update_frequencies_cm(freq_vector.tolist())

        print(
            f"\n--- combo {global_idx}: t_idx={t_idx}, t_coh={t_coh_val:.4f} fs, "
            f"inhom_idx={inhom_idx} ---"
        )

        e_components = parallel_compute_1d_e_comps(sim_oqs=sim, time_cut=args.time_cut)

        metadata = {
            "signal_types": sim.simulation_config.signal_types,
            "t_coh_value": t_coh_val,
            "t_index": t_idx,
            "inhom_index": inhom_idx,
            "combination_index": global_idx,
            "inhom_group_id": sim.simulation_config.inhom_group_id,
            "sim_type": args.sim_type,
            "batch_id": args.batch_id,
            "n_batches": args.n_batches,
        }

        t_det_axis = sim.t_det

        if first_save:
            path = save_simulation_data(
                sim,
                metadata,
                e_components,
                t_det=t_det_axis,
                t_coh=None,
                data_root=output_root,
            )
            first_save = False
        else:
            path = save_data_only(
                sim,
                metadata,
                e_components,
                t_det=t_det_axis,
                t_coh=None,
                data_root=output_root,
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


if __name__ == "__main__":
    main()
