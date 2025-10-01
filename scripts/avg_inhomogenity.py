"""Average per-sample spectroscopy artifacts into a single 1D bundle.

This script works with the new run artifacts produced by ``save_run_artifact``.
Given one artifact path, it gathers all sibling files that share the same
``t_index`` and averages them across the inhomogeneous samples.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np

from qspectro2d.utils.data_io import (
    load_run_artifact,
    save_run_artifact,
    resolve_run_prefix,
    ensure_run_sidecar,
)

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
    simulation_config: dict[str, Any]
    system: dict[str, Any]
    laser: dict[str, Any] | None
    bath: dict[str, Any] | None
    job_metadata: dict[str, Any] | None


class _SystemStub:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = dict(payload)

    def to_dict(self) -> dict[str, Any]:
        return dict(self._payload)


class _SimulationConfigStub:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = dict(payload)
        for key, value in self._payload.items():
            setattr(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        return dict(self._payload)


class _LaserStub:
    def __init__(self, payload: dict[str, Any] | None) -> None:
        self._payload = dict(payload) if payload else {}

    def to_dict(self) -> dict[str, Any]:
        return dict(self._payload)


class _SimulationModuleStub:
    def __init__(
        self, system: dict[str, Any], sim_cfg: dict[str, Any], laser: dict[str, Any] | None
    ) -> None:
        self.system = _SystemStub(system)
        self.simulation_config = _SimulationConfigStub(sim_cfg)
        self.laser = _LaserStub(laser)


def _load_entry(path: Path) -> RunEntry:
    artifact = load_run_artifact(path)
    metadata = artifact.get("metadata", {})
    signals = artifact.get("signals", {})
    if not metadata or not signals:
        raise ValueError(
            f"Artifact {path} does not contain metadata/signals; ensure you're using new run artifacts."
        )

    metadata = dict(metadata)
    if "t_coh_value" not in metadata:
        raise KeyError(f"Missing 't_coh_value' in artifact metadata: {path}")
    try:
        metadata["t_coh_value"] = float(np.asarray(metadata["t_coh_value"]))
    except Exception as exc:
        raise ValueError(f"Invalid 't_coh_value' in artifact metadata: {path}") from exc

    t_det = np.asarray(artifact.get("t_det"), dtype=float)
    freq_sample = np.asarray(artifact.get("frequency_sample_cm"), dtype=float)
    sim_cfg = dict(artifact.get("simulation_config", {}))
    system = dict(artifact.get("system", {}))
    laser = artifact.get("laser")
    if laser is not None:
        laser = dict(laser)

    bath = artifact.get("bath")
    if bath is not None:
        bath = dict(bath)

    job_metadata = artifact.get("job_metadata")
    if job_metadata is not None:
        job_metadata = dict(job_metadata)

    return RunEntry(
        path=path,
        metadata=metadata,
        signals={key: np.asarray(val, dtype=float) for key, val in signals.items()},
        t_det=t_det,
        frequency_sample_cm=freq_sample,
        simulation_config=sim_cfg,
        system=system,
        laser=laser,
        bath=bath,
        job_metadata=job_metadata,
    )


def _artifact_prefix(path: Path) -> str:
    stem = path.stem
    if "_run_" not in stem:
        raise ValueError(f"Unexpected artifact filename (missing '_run_'): {path.name}")
    return stem.split("_run_", 1)[0]


def _discover_entries(anchor: RunEntry) -> list[RunEntry]:
    directory = anchor.path.parent
    prefix = _artifact_prefix(anchor.path)
    t_index = int(anchor.metadata.get("t_index", 0))
    candidates = sorted(directory.glob(f"{prefix}_run_t{t_index:03d}_c*_s*.npz"))

    entries: list[RunEntry] = []
    for candidate in candidates:
        try:
            entry = _load_entry(candidate)
        except Exception as exc:
            print(f"⚠️  Skipping unreadable artifact: {candidate}\n    {exc}")
            continue
        entries.append(entry)
    return entries


def _aggregate_sample_id(sample_ids: Iterable[str]) -> str:
    combined = "|".join(sorted(sample_ids))
    return hashlib.sha1(combined.encode("utf-8")).hexdigest()


def _ensure_consistent_axes(entries: list[RunEntry]) -> tuple[np.ndarray, list[str]]:
    if not entries:
        raise ValueError("No artifacts available for averaging")

    reference_t_det = entries[0].t_det
    signal_types = list(entries[0].metadata.get("signal_types", entries[0].signals.keys()))

    for entry in entries[1:]:
        if entry.t_det.shape != reference_t_det.shape or not np.allclose(
            entry.t_det, reference_t_det
        ):
            raise ValueError(f"Inconsistent t_det axis in artifact {entry.path}")
        current_signals = list(entry.metadata.get("signal_types", entry.signals.keys()))
        if current_signals != signal_types:
            raise ValueError(
                "Signal type mismatch across artifacts."
                f" Reference={signal_types}, current={current_signals}, source={entry.path}"
            )

    return reference_t_det, signal_types


def _select_raw_entries(entries: list[RunEntry]) -> tuple[list[RunEntry], list[RunEntry]]:
    averaged = [entry for entry in entries if entry.metadata.get("inhom_averaged")]
    raw = [entry for entry in entries if not entry.metadata.get("inhom_averaged")]
    return raw, averaged


def average_inhom_1d(abs_path: Path, *, skip_if_exists: bool = False) -> Path:
    """Average all inhomogeneous configurations for the coherence index of ``abs_path``."""

    abs_path = abs_path.expanduser().resolve()
    anchor = _load_entry(abs_path)

    metadata = anchor.metadata
    sample_size = int(metadata.get("sample_size", 1) or 1)

    if metadata.get("inhom_averaged") or sample_size <= 1:
        return anchor.path

    entries = _discover_entries(anchor)
    raw_entries, averaged_entries = _select_raw_entries(entries)

    if averaged_entries:
        existing = sorted(averaged_entries, key=lambda e: e.path)
        if skip_if_exists:
            print(f"⏭️  Using existing averaged artifact: {existing[0].path}")
            return existing[0].path
        else:
            print("ℹ️  Averaged artifact already exists; recreating from raw samples.")

    if not raw_entries:
        raise FileNotFoundError(
            "No raw artifacts available for averaging. Ensure you pointed to a per-sample run artifact."
        )

    t_det, signal_types = _ensure_consistent_axes(raw_entries)
    sample_ids = [str(entry.metadata.get("sample_id")) for entry in raw_entries]
    combination_indices = [int(entry.metadata.get("combination_index", 0)) for entry in raw_entries]

    data_stack = {
        sig: np.stack([entry.signals[sig] for entry in raw_entries], axis=0) for sig in signal_types
    }
    averaged_signals = {sig: np.mean(stack, axis=0) for sig, stack in data_stack.items()}

    freq_stack = np.stack([entry.frequency_sample_cm for entry in raw_entries], axis=0)
    avg_freq = np.mean(freq_stack, axis=0)

    aggregated_sample_id = _aggregate_sample_id(sample_ids)
    t_index = int(metadata.get("t_index", 0))
    new_combination_index = max(combination_indices) if combination_indices else t_index
    metadata_out = {**metadata}
    metadata_out.update(
        {
            "sample_id": aggregated_sample_id,
            "sample_index": None,
            "inhom_averaged": True,
            "averaged_count": len(raw_entries),
            "source_sample_ids": sample_ids,
            "source_artifacts": [entry.path.name for entry in raw_entries],
            "combination_index": int(new_combination_index),
        }
    )

    sim_cfg = dict(anchor.simulation_config)
    sim_cfg.update(
        {
            "inhom_averaged": True,
            "current_sample_id": aggregated_sample_id,
            "sample_size": int(metadata.get("sample_size", len(raw_entries))),
        }
    )

    sim_stub = _SimulationModuleStub(anchor.system, sim_cfg, anchor.laser)

    extra_payload: dict[str, Any] = {}
    if anchor.job_metadata:
        extra_payload["job"] = anchor.job_metadata
    if anchor.bath:
        extra_payload["bath"] = anchor.bath

    ensure_run_sidecar(
        sim_stub,
        data_root=DATA_DIR,
        extra_payload=extra_payload or None,
    )

    expected_dir, prefix = resolve_run_prefix(sim_stub.system, sim_stub.simulation_config, DATA_DIR)
    expected_path = (
        expected_dir
        / f"{prefix}_run_t{t_index:03d}_c{int(new_combination_index):04d}_s{aggregated_sample_id[:8]}.npz"
    )

    if skip_if_exists and expected_path.exists():
        print(f"⏭️  Averaged artifact already present: {expected_path}")
        return expected_path

    out_path = save_run_artifact(
        sim_stub,
        signal_arrays=[averaged_signals[sig] for sig in signal_types],
        t_det=t_det,
        metadata=metadata_out,
        frequency_sample_cm=avg_freq,
        sample_id=aggregated_sample_id,
        data_root=DATA_DIR,
        t_coh=None,
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
