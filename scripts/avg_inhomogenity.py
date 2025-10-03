"""Average per-sample spectroscopy artifacts into a single 1D bundle."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from qspectro2d.utils.data_io import (
    load_run_artifact,
    save_run_artifact,
    save_info_file,
    split_prefix,
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
    """Find all related artifacts for the same t_index as the anchor.
       Includes the anchor itself."""
    directory, prefix = split_prefix(anchor.path)

    entries: list[RunEntry] = []
    for candidate in sorted(directory.glob(f"{prefix}_run_t*_c*.npz")):
        entry = _load_entry(candidate)
        # Only average raw (non-averaged) artifacts
        if bool(entry.simulation_config.inhom_averaged) != False:
            continue
        # Only include entries with the same t_index as the anchor
        if entry.metadata.get("t_index") != anchor.metadata.get("t_index"):
            continue
        
        entries.append(entry)

    return entries

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
    # Use simulation_config for averaged flag consistently
    raw_entries = [entry for entry in entries if not entry.simulation_config.inhom_averaged]
    if not raw_entries:
        raise FileNotFoundError("No raw artifacts available for averaging")

    existing = [entry for entry in entries if entry.simulation_config.inhom_averaged]
    if existing and skip_if_exists:
        return existing[0].path

    # Handle single entry case
    if len(raw_entries) == 1:
        single_entry = raw_entries[0]
        signal_types = list(single_entry.metadata.get("signal_types", single_entry.signals.keys()))
        t_index = single_entry.metadata["t_index"]

        metadata_out = dict(single_entry.metadata)
        metadata_out.update(
            {
                "t_index": t_index,
                "inhom_averaged": True,
                "averaged_count": 1,
                "source_artifacts": [single_entry.path.name],
            }
        )

        # Remove sample_index since it was averaged over
        metadata_out.pop("sample_index", None)
        # Remove combination_index as it's no longer relevant after averaging
        metadata_out.pop("combination_index", None)

        sim_cfg = replace(
            single_entry.simulation_config,
            inhom_averaged=True,
        )

        snapshot = SimulationSnapshot(
            system=single_entry.system,
            simulation_config=sim_cfg,
            laser=single_entry.laser,
            bath=single_entry.bath,
        )

        extra_payload: dict[str, Any] | None = None
        if single_entry.job_metadata or single_entry.bath:
            extra_payload = {}
            if single_entry.job_metadata:
                extra_payload["job_metadata"] = single_entry.job_metadata
            if single_entry.bath:
                extra_payload["bath"] = single_entry.bath

        expected_dir = single_entry.path.parent
        prefix = _generate_base_stem(snapshot.simulation_config)
        expected_path = expected_dir / f"{prefix}_run_t{int(t_index):03d}.npz"

        if skip_if_exists and expected_path.exists():
            print(f"⏭️  Averaged artifact already present: {expected_path}")
            return expected_path

        info_path = expected_dir / f"{prefix}.pkl"
        if not info_path.exists():
            save_info_file(
                info_path,
                snapshot.system,
                snapshot.simulation_config,
                bath=snapshot.bath,
                laser=snapshot.laser,
                extra_payload=extra_payload,
            )

        out_path = save_run_artifact(
            signal_arrays=[single_entry.signals[sig] for sig in signal_types],
            t_det=single_entry.t_det,
            metadata=metadata_out,
            frequency_sample_cm=single_entry.frequency_sample_cm,
            data_dir=expected_dir,
            filename=f"{prefix}_run_t{int(t_index):03d}.npz",
            t_coh=metadata_out.get("t_coh_value"),
        )

        print(f"⏭️ Single artifact treated as averaged for t_index={t_index} → {out_path}")
        return out_path

    # Multi-entry averaging
    t_det, signal_types = _ensure_consistency(raw_entries)

    data_stack = {
        sig: np.stack([entry.signals[sig] for entry in raw_entries], axis=0) for sig in signal_types
    }
    averaged_signals = {sig: np.mean(stack, axis=0) for sig, stack in data_stack.items()}

    freq_stack = np.stack([entry.frequency_sample_cm for entry in raw_entries], axis=0)
    avg_freq = np.mean(freq_stack, axis=0)

    metadata_out = dict(anchor.metadata)
    metadata_out.update(
        {
            "inhom_averaged": True,
            "averaged_count": len(raw_entries),
            "source_artifacts": [entry.path.name for entry in raw_entries],
        }
    )

    # Remove sample_index since it was averaged over
    # Remove combination_index as it's no longer relevant after averaging
    metadata_out.pop("combination_index", None)
    metadata_out.pop("sample_index", None)

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
    expected_path = expected_dir / f"{prefix}_run_t{int(t_index):03d}.npz"

    # Only skip if the specific expected averaged artifact exists
    if skip_if_exists and expected_path.exists():
        print(f"⏭️  Averaged artifact already present: {expected_path}")
        return expected_path

    info_path = expected_dir / f"{prefix}.pkl"
    if not info_path.exists():
        save_info_file(
            info_path,
            snapshot.system,
            snapshot.simulation_config,
            bath=snapshot.bath,
            laser=snapshot.laser,
            extra_payload=extra_payload,
        )

    out_path = save_run_artifact(
        signal_arrays=[averaged_signals[sig] for sig in signal_types],
        t_det=t_det,
        metadata=metadata_out,
        frequency_sample_cm=avg_freq,
        data_dir=expected_dir,
        filename=f"{prefix}_run_t{int(t_index):03d}.npz",
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
