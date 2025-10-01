"""Stack per-coherence run artifacts into a 2D dataset."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

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
else:  # pragma: no cover - defensive fallback
    raise RuntimeError("Could not locate project root (missing .git directory)")

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
            f"Artifact {path} is missing metadata/signals. Ensure you're using new run artifacts."
        )

    metadata = dict(metadata)
    metadata["t_coh_value"] = float(np.asarray(metadata["t_coh_value"]))

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
    target_sample_id = str(anchor.metadata.get("sample_id"))
    averaged_flag = bool(anchor.metadata.get("inhom_averaged"))

    candidates = sorted(directory.glob(f"{prefix}_run_t*_c*_s*.npz"))
    entries: list[RunEntry] = []
    for candidate in candidates:
        try:
            entry = _load_entry(candidate)
        except Exception as exc:
            print(f"⚠️  Skipping unreadable artifact: {candidate}\n    {exc}")
            continue

        if str(entry.metadata.get("sample_id")) != target_sample_id:
            continue
        if bool(entry.metadata.get("inhom_averaged")) != averaged_flag:
            continue
        entries.append(entry)

    return entries


def _ensure_consistency(entries: list[RunEntry]) -> tuple[np.ndarray, list[str]]:
    if not entries:
        raise ValueError("No artifacts available for stacking")

    reference = entries[0]
    t_det = reference.t_det
    signal_types = list(reference.metadata.get("signal_types", reference.signals.keys()))

    for entry in entries[1:]:
        if entry.t_det.shape != t_det.shape or not np.allclose(entry.t_det, t_det):
            raise ValueError(f"Inconsistent t_det axis for artifact {entry.path}")
        current_signals = list(entry.metadata.get("signal_types", entry.signals.keys()))
        if current_signals != signal_types:
            raise ValueError(
                "Signal type mismatch across artifacts."
                f" Reference={signal_types}, current={current_signals}, source={entry.path}"
            )

    return t_det, signal_types


def _sort_entries(entries: Iterable[RunEntry]) -> list[RunEntry]:
    def _sort_key(entry: RunEntry) -> tuple[int, float]:
        t_index = int(entry.metadata.get("t_index", 0))
        return t_index, float(entry.metadata["t_coh_value"])

    return sorted(entries, key=_sort_key)


def stack_artifacts(abs_path: Path, *, skip_if_exists: bool = False) -> Path:
    """Stack all artifacts that share the same sample_id as ``abs_path`` into a 2D dataset."""

    abs_path = abs_path.expanduser().resolve()
    anchor = _load_entry(abs_path)
    entries = _discover_entries(anchor)

    if len(entries) <= 1:
        raise ValueError(
            "Need at least two artifacts with distinct coherence indices to build a 2D dataset."
        )

    entries = _sort_entries(entries)
    t_det, signal_types = _ensure_consistency(entries)

    t_indices = [int(entry.metadata.get("t_index", idx)) for idx, entry in enumerate(entries)]
    t_coh_values = [float(entry.metadata["t_coh_value"]) for entry in entries]
    t_coh_axis = np.asarray(t_coh_values, dtype=float)

    stacked_signals = {
        sig: np.stack([entry.signals[sig] for entry in entries], axis=0) for sig in signal_types
    }

    sample_id = str(anchor.metadata.get("sample_id"))
    combination_indices = [int(entry.metadata.get("combination_index", 0)) for entry in entries]
    new_combination_index = max(combination_indices) + 1 if combination_indices else len(entries)
    new_t_index = max(t_indices) + 1 if t_indices else len(entries)

    metadata_out = {**anchor.metadata}
    metadata_out.update(
        {
            "sim_type": "2d",
            "signal_types": signal_types,
            "t_index": int(new_t_index),
            "combination_index": int(new_combination_index),
            "stacked_points": len(entries),
            "t_indices": t_indices,
            "t_coh_axis": t_coh_values,
            "source_artifacts": [entry.path.name for entry in entries],
        }
    )
    metadata_out.pop("t_coh_value", None)

    sim_cfg = dict(anchor.simulation_config)
    sim_cfg.update(
        {
            "sim_type": "2d",
            "t_coh": None,
            "current_sample_id": sample_id,
            "inhom_averaged": bool(anchor.metadata.get("inhom_averaged")),
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

    out_dir, prefix = resolve_run_prefix(sim_stub.system, sim_stub.simulation_config, DATA_DIR)
    expected_path = (
        out_dir
        / f"{prefix}_run_t{int(new_t_index):03d}_c{int(new_combination_index):04d}_s{sample_id[:8]}.npz"
    )

    if skip_if_exists and expected_path.exists():
        print(f"⏭️  Using existing stacked artifact: {expected_path}")
        return expected_path

    out_path = save_run_artifact(
        sim_stub,
        signal_arrays=[stacked_signals[sig] for sig in signal_types],
        t_det=t_det,
        metadata=metadata_out,
        frequency_sample_cm=anchor.frequency_sample_cm,
        sample_id=sample_id,
        data_root=DATA_DIR,
        t_coh=t_coh_axis,
    )

    print(f"✅ Stacked {len(entries)} artifacts into 2D dataset: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Stack per-t_coh artifacts into a 2D dataset.")
    parser.add_argument("--abs_path", type=str, required=True, help="Path to a run artifact (.npz)")
    parser.add_argument(
        "--skip_if_exists",
        action="store_true",
        help="Reuse an existing 2D artifact if it already exists",
    )
    args = parser.parse_args()

    stacked_path = stack_artifacts(Path(args.abs_path), skip_if_exists=args.skip_if_exists)
    print(f"Saved 2D dataset: {stacked_path}")
    print("\n🎯 Next: plot with")
    print(f"python plot_datas.py --abs_path {stacked_path}")


if __name__ == "__main__":
    main()
