"""
Data I/O operations for qspectro2d.
"""

from __future__ import annotations

# IMPORTS
import json
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence, TYPE_CHECKING

import numpy as np

# Type checking imports to avoid circular imports
if TYPE_CHECKING:
    from qspectro2d.core.laser_system.laser_class import LaserPulseSequence
    from qspectro2d.core.atomic_system.system_class import AtomicSystem
    from qspectro2d.core.simulation import SimulationConfig
    from qutip import BosonicEnvironment

_SIGNAL_PREFIX = "signal::"
_META_KEY = "metadata_json"
_T_DET_KEY = "t_det"
_T_COH_KEY = "t_coh"
_SAMPLE_KEY = "frequency_sample_cm"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Cannot serialize object of type {type(obj)!r}")


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, default=_json_default, sort_keys=True)


def split_prefix(path: Path) -> tuple[Path, str]:
    path = Path(path)
    stem = path.stem
    if "_run_" in stem:
        prefix = stem.rsplit("_run_", 1)[0]
    else:
        prefix = stem
    return path.parent, prefix


def _info_path(directory: Path, prefix: str) -> Path:
    return directory / f"{prefix}.pkl"


def save_run_artifact(
    *,
    signal_arrays: Sequence[np.ndarray],
    t_det: np.ndarray,
    metadata: Mapping[str, Any],
    frequency_sample_cm: Sequence[float],
    data_dir: Path | str,
    filename: str,
    t_coh: np.ndarray | None = None,
) -> Path:
    """Persist a single run (t_coh × sample) as a compressed ``.npz`` artifact."""

    directory = Path(data_dir)
    abs_path = directory / filename

    signal_types = list(metadata.get("signal_types", []))
    if len(signal_types) != len(signal_arrays):
        raise ValueError("signal_types metadata must match number of signal arrays")

    payload: dict[str, Any] = {
        _T_DET_KEY: np.asarray(t_det, dtype=float),
        _SAMPLE_KEY: np.asarray(frequency_sample_cm, dtype=float),
        _META_KEY: np.array(_json_dumps({**metadata}), dtype=np.str_),
    }

    if t_coh is not None:
        payload[_T_COH_KEY] = np.asarray(t_coh, dtype=float)

    for sig, data in zip(signal_types, signal_arrays):
        payload[f"{_SIGNAL_PREFIX}{sig}"] = np.asarray(data)

    np.savez_compressed(abs_path, **payload)

    return abs_path


def save_info_file(
    abs_info_path: Path,
    system: "AtomicSystem",
    sim_config: "SimulationConfig",
    bath: "BosonicEnvironment" | None = None,
    laser: "LaserPulseSequence" | None = None,
    *,
    extra_payload: Mapping[str, Any] | None = None,
) -> None:
    """
    Save system parameters and data configuration to pickle file.

    Args:
        abs_info_path: Absolute path for the info file (.pkl)
        system: System parameters object
        bath: QuTip Environment instance
        laser: Laser pulse sequence object
        sim_config: SimulationConfig instance used for the run (stored as object, not dict)
    """
    try:
        payload = {
            "system": system,
            # Store the SimulationConfig instance directly for full fidelity
            "sim_config": sim_config,
        }
        if bath is not None:
            payload["bath"] = bath
        if laser is not None:
            payload["laser"] = laser
        if extra_payload:
            for key, value in extra_payload.items():
                if value is None:
                    continue
                existing = payload.get(key)
                if isinstance(existing, Mapping) and isinstance(value, Mapping):
                    merged = dict(existing)
                    merged.update(value)
                    payload[key] = merged
                else:
                    payload[key] = value

        with open(abs_info_path, "wb") as info_file:
            pickle.dump(payload, info_file)
        print(f"Info saved: {abs_info_path}")
    except Exception as e:
        print(f"Failed to save info: {e}")
        raise


def load_info_file(abs_info_path: Path) -> dict:
    """
    Load pickle info file (.pkl) from absolute path.

    Args:
        abs_info_path: Absolute path to the info file (.pkl)

    Returns:
        dict: Dictionary containing system parameters and data configuration
    """
    try:
        # print(f"Loading info: {abs_info_path}")

        if not abs_info_path.exists():
            raise FileNotFoundError(f"Info file not found: {abs_info_path}")

        with open(abs_info_path, "rb") as info_file:
            info = pickle.load(info_file)

        # print(f"Loaded info: {abs_info_path}")
        return info
    except Exception as e:
        print(f"Failed to load info: {abs_info_path}; error: {e}")
        raise


def load_run_artifact(path: Path | str) -> dict[str, Any]:
    """Load a run artifact produced by :func:`save_run_artifact`."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as bundle:
        contents = {key: bundle[key] for key in bundle.files}

    metadata = json.loads(str(contents.pop(_META_KEY).item())) if _META_KEY in contents else {}

    # Load from info file
    directory, prefix = split_prefix(path)
    info_path = _info_path(directory, prefix)
    info = load_info_file(info_path)

    sim_cfg = info.get("sim_config", {})
    system = info.get("system", {})
    laser = info.get("laser")
    bath = info.get("bath")

    signals: dict[str, np.ndarray] = {}
    for key in list(contents.keys()):
        if key.startswith(_SIGNAL_PREFIX):
            sig = key[len(_SIGNAL_PREFIX) :]
            signals[sig] = contents.pop(key)

    return {
        "path": path,
        "signals": signals,
        "t_det": contents.get(_T_DET_KEY),
        "t_coh": contents.get(_T_COH_KEY),
        "frequency_sample_cm": contents.get(_SAMPLE_KEY),
        "metadata": metadata,
        "simulation_config": sim_cfg,
        "system": system,
        "laser": laser,
        "bath": bath,
    }


def load_simulation_data(abs_path: Path | str) -> dict:
    """Load and unpack a run artifact produced by :func:`save_run_artifact`."""

    artifact = load_run_artifact(abs_path)
    signals = artifact.get("signals", {})
    bundle = dict(artifact)
    bundle["signal_types"] = list(signals.keys())

    for name, array in signals.items():
        bundle[name] = array

    t_det = bundle.get("t_det")
    if t_det is None:
        raise KeyError("Run artifact is missing the 't_det' axis")

    if bundle.get("frequency_sample_cm") is None:
        raise KeyError("Run artifact is missing the frequency sample axis")

    return bundle
