"""Stack per-coherence run artifacts into a 2D dataset."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

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
else:
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
    simulation_config: Any
    system: Any
    laser: Any | None
    bath: Any | None
    job_metadata: Any | None


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
    laser = artifact.get("laser")
    bath = artifact.get("bath")
    job_metadata = artifact.get("job_metadata")

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
    averaged_flag = bool(anchor.metadata.get("inhom_averaged"))

    entries: list[RunEntry] = []
    for candidate in sorted(directory.glob(f"{prefix}_run_t*_c*.npz")):
        entry = _load_entry(candidate)
        # Only stack 1D artifacts, not already stacked 2D data
        if entry.simulation_config.sim_type == "2d":
            continue
        if bool(entry.simulation_config.inhom_averaged) != averaged_flag:
            continue
        entries.append(entry)

    return entries


def _ensure_consistency(entries: list[RunEntry]) -> tuple[np.ndarray, list[str]]:
    if not entries:
        raise ValueError("No artifacts available for stacking")

    reference = entries[0]
    t_det = reference.t_det
    signal_types = list(reference.metadata.get("signal_types", reference.signals.keys()))
    freq_ref = reference.frequency_sample_cm

    for entry in entries[1:]:
        if entry.t_det.shape != t_det.shape or not np.allclose(entry.t_det, t_det):
            raise ValueError(f"Inconsistent t_det axis for artifact {entry.path}")
        if not np.allclose(entry.frequency_sample_cm, freq_ref):
            raise ValueError(f"Inconsistent frequency configuration for artifact {entry.path}")
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
        t_coh = float(entry.metadata.get("t_coh_value", 0.0))
        return t_index, t_coh

    return sorted(entries, key=_sort_key)


def stack_artifacts(abs_path: Path, *, skip_if_exists: bool = False) -> Path:
    """Stack all artifacts that share the same frequency configuration into a 2D dataset."""

    abs_path = abs_path.expanduser().resolve()
    anchor = _load_entry(abs_path)
    
    # Ensure we're stacking 1D data, not 2D data
    if anchor.simulation_config.sim_type == "2d":
        raise ValueError("Cannot stack 2D artifacts. Provide a 1D artifact to stack into 2D.")
    
    entries = _discover_entries(anchor)

    if len(entries) <= 1:
        raise ValueError(
            "Need at least two artifacts with distinct coherence indices to build a 2D dataset."
        )

    entries = _sort_entries(entries)
    t_det, signal_types = _ensure_consistency(entries)

    t_indices = [int(entry.metadata.get("t_index", idx)) for idx, entry in enumerate(entries)]
    t_coh_values = [float(entry.metadata.get("t_coh_value", 0.0)) for entry in entries]
    t_coh_axis = np.asarray(t_coh_values, dtype=float)

    stacked_signals = {
        sig: np.stack([entry.signals[sig] for entry in entries], axis=0) for sig in signal_types
    }

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

    sim_cfg = replace(
        anchor.simulation_config,
        sim_type="2d",
        t_coh=None,
        inhom_averaged=bool(anchor.metadata.get("inhom_averaged")),
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

    out_dir = anchor.path.parent
    prefix = _generate_base_stem(snapshot.simulation_config)
    expected_path = (
        out_dir / f"{prefix}_run_t{int(new_t_index):03d}_c{int(new_combination_index):04d}.npz"
    )

    if skip_if_exists and expected_path.exists():
        return expected_path

    save_info_file(
        out_dir / f"{prefix}.pkl",
        snapshot.system,
        snapshot.simulation_config,
        bath=snapshot.bath,
        laser=snapshot.laser,
        extra_payload=extra_payload,
    )

    out_path = save_run_artifact(
        snapshot,
        signal_arrays=[stacked_signals[sig] for sig in signal_types],
        t_det=t_det,
        metadata=metadata_out,
        frequency_sample_cm=anchor.frequency_sample_cm,
        data_dir=out_dir,
        prefix=prefix,
        t_coh=t_coh_axis,
    )
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
