"""Average per-sample spectroscopy artifacts into a single 1D bundle."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from qspectro2d.utils.data_io import (
    ensure_info_file,
    load_run_artifact,
    resolve_run_prefix,
    save_run_artifact,
    save_info_file,
)
from qspectro2d.utils.file_naming import _generate_base_stem

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


@dataclass(slots=True)
class SimulationSnapshot:
    system: Any
    simulation_config: Any
    laser: Any | None = None
    bath: Any | None = None


def _load_entry(path: Path) -> RunEntry:
    artifact = load_run_artifact(path)

    metadata = dict(artifact["metadata"])
    if "t_coh_value" in metadata:
        metadata["t_coh_value"] = float(np.asarray(metadata["t_coh_value"]))

    signals = {key: np.asarray(val) for key, val in artifact["signals"].items()}
    t_det = np.asarray(artifact["t_det"], dtype=float)
    freq_sample = np.asarray(artifact["frequency_sample_cm"], dtype=float)

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
    )


def _discover_entries(anchor: RunEntry) -> list[RunEntry]:
    directory = anchor.path.parent
    prefix = anchor.path.stem.split("_run_", 1)[0]
    candidates = sorted(
        directory.glob(f"{prefix}_run_t{int(anchor.metadata['t_index']):03d}_c*.npz")
    )

    return [_load_entry(candidate) for candidate in candidates]


def _ensure_consistency(entries: list[RunEntry]) -> tuple[np.ndarray, list[str]]:
    if not entries:
        raise ValueError("No artifacts available for averaging")

    reference = entries[0]
    t_det = reference.t_det
    signal_types = list(reference.metadata.get("signal_types", reference.signals.keys()))

    for entry in entries[1:]:
        if entry.t_det.shape != t_det.shape or not np.allclose(entry.t_det, t_det):
            raise ValueError(f"Inconsistent t_det axis for artifact {entry.path}")
        current = list(entry.metadata.get("signal_types", entry.signals.keys()))
        if current != signal_types:
            raise ValueError(
                "Signal type mismatch across artifacts "
                f"(reference={signal_types}, current={current}, source={entry.path})"
            )

    return t_det, signal_types


def average_inhom_1d(abs_path: Path, *, skip_if_exists: bool = False) -> Path:
    abs_path = abs_path.expanduser().resolve()
    anchor = _load_entry(abs_path)

    if anchor.metadata.get("inhom_averaged"):
        return anchor.path

    entries = _discover_entries(anchor)
    raw_entries = [entry for entry in entries if not entry.metadata.get("inhom_averaged")]
    if not raw_entries:
        raise FileNotFoundError("No raw artifacts available for averaging")

    existing = [entry for entry in entries if entry.metadata.get("inhom_averaged")]
    if existing and skip_if_exists:
        return existing[0].path

    t_det, signal_types = _ensure_consistency(raw_entries)

    data_stack = {
        sig: np.stack([entry.signals[sig] for entry in raw_entries], axis=0) for sig in signal_types
    }
    averaged_signals = {sig: np.mean(stack, axis=0) for sig, stack in data_stack.items()}

    freq_stack = np.stack([entry.frequency_sample_cm for entry in raw_entries], axis=0)
    avg_freq = np.mean(freq_stack, axis=0)

    combination_indices = [int(entry.metadata.get("combination_index", 0)) for entry in raw_entries]
    new_combination_index = max(combination_indices) if combination_indices else 0

    metadata_out = dict(anchor.metadata)
    metadata_out.update(
        {
            "sample_index": None,
            "inhom_averaged": True,
            "averaged_count": len(raw_entries),
            "source_artifacts": [entry.path.name for entry in raw_entries],
            "combination_index": int(new_combination_index),
        }
    )

    sim_cfg = replace(
        anchor.simulation_config,
        inhom_averaged=True,
    )

    snapshot = SimulationSnapshot(
        system=anchor.system,
        simulation_config=sim_cfg,
        laser=anchor.laser,
        bath=anchor.bath,
    )

    extra_payload: dict[str, Any] | None = None
    if anchor.job_metadata or anchor.bath:
        extra_payload = {}
        if anchor.job_metadata:
            extra_payload["job_metadata"] = anchor.job_metadata
        if anchor.bath:
            extra_payload["bath"] = anchor.bath

    t_index = metadata_out["t_index"]
    expected_dir = anchor.path.parent
    prefix = _generate_base_stem(snapshot.simulation_config)
    expected_path = (
        expected_dir / f"{prefix}_run_t{int(t_index):03d}_c{int(new_combination_index):04d}.npz"
    )

    if skip_if_exists and expected_path.exists():
        print(f"⏭️  Averaged artifact already present: {expected_path}")
        return expected_path

    save_info_file(
        expected_dir / f"{prefix}.pkl",
        snapshot.system,
        snapshot.simulation_config,
        bath=snapshot.bath,
        laser=snapshot.laser,
        extra_payload=extra_payload,
    )

    out_path = save_run_artifact(
        snapshot,
        signal_arrays=[averaged_signals[sig] for sig in signal_types],
        t_det=t_det,
        metadata=metadata_out,
        frequency_sample_cm=avg_freq,
        data_dir=expected_dir,
        prefix=prefix,
        t_coh=metadata_out.get("t_coh_value"),
    )

    print(f"✅ Averaged {len(raw_entries)} artifacts for t_index={t_index} → {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Average all inhomogeneous samples for a single coherence index."
    )
    parser.add_argument("--abs_path", type=str, required=True, help="Path to a run artifact (.npz)")
    parser.add_argument(
        "--skip_if_exists",
        action="store_true",
        help="Reuse an existing averaged artifact if present",
    )
    args = parser.parse_args()

    averaged_path = average_inhom_1d(Path(args.abs_path), skip_if_exists=args.skip_if_exists)
    print(f"Saved inhom-averaged artifact: {averaged_path}")
    print("\n🎯 Next: plot with")
    print(f"python plot_datas.py --abs_path {averaged_path}")


if __name__ == "__main__":
    main()
