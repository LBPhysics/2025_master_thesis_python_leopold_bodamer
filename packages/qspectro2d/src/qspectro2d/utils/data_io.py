"""
Data I/O operations for qspectro2d.

This module provides functionality for loading and saving simulation data,
including standardized file formats and directory management.
"""

from __future__ import annotations

# IMPORTS
import glob
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, List, TYPE_CHECKING

import numpy as np

# Type checking imports to avoid circular imports
if TYPE_CHECKING:
    from qspectro2d.core.laser_system.laser_class import LaserPulseSequence
    from qspectro2d.core.atomic_system.system_class import AtomicSystem
    from qspectro2d.core.simulation import SimulationConfig, SimulationModuleOQS
from qspectro2d.utils.file_naming import (
    generate_deterministic_data_base,
)

_SIGNAL_PREFIX = "signal::"
_META_KEY = "metadata_json"
_T_DET_KEY = "t_det"
_T_COH_KEY = "t_coh"
_SAMPLE_KEY = "frequency_sample_cm"
_SIM_CFG_KEY = "simulation_config_json"
_SYSTEM_KEY = "system_json"
_LASER_KEY = "laser_json"
_JOB_META_KEY = "job_metadata_json"
_SIDECAR_NAME = "job_metadata"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Cannot serialize object of type {type(obj)!r}")


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, default=_json_default, sort_keys=True)


def compute_sample_id(frequency_sample_cm: Sequence[float]) -> str:
    """Return a SHA1 hash representing the provided frequency vector."""

    arr = np.asarray(frequency_sample_cm, dtype=np.float64)
    return hashlib.sha1(arr.tobytes()).hexdigest()


def _split_prefix(path: Path) -> tuple[Path, str]:
    path = Path(path)
    stem = path.stem
    if "_run_" not in stem:
        raise ValueError(f"Artifact filename missing '_run_' segment: {path}")
    prefix = stem.split("_run_", 1)[0]
    return path.parent, prefix


def _sidecar_path(directory: Path, prefix: str) -> Path:
    return directory / f"{prefix}_{_SIDECAR_NAME}.json"


def _load_sidecar(directory: Path, prefix: str) -> dict[str, Any]:
    sidecar = _sidecar_path(directory, prefix)
    if not sidecar.exists():
        return {}
    with sidecar.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_run_sidecar(
    sim_module: "SimulationModuleOQS",
    *,
    data_root: Path | str,
    extra_payload: Mapping[str, Any] | None = None,
) -> Path:
    """Ensure a JSON sidecar with simulation-wide metadata exists for the run prefix."""

    directory, prefix = resolve_run_prefix(
        sim_module.system, sim_module.simulation_config, data_root
    )
    sidecar = _sidecar_path(directory, prefix)

    payload: dict[str, Any] = {}
    if sidecar.exists():
        try:
            with sidecar.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError:
            payload = {}

    def _setdefault(key: str, value: Any) -> None:
        if key not in payload and value is not None:
            payload[key] = value

    _setdefault("simulation_config", sim_module.simulation_config.to_dict())
    _setdefault("system", sim_module.system.to_dict())

    laser = getattr(sim_module, "laser", None)
    laser_to_dict = getattr(laser, "to_dict", None)
    if callable(laser_to_dict):
        _setdefault("laser", laser_to_dict())

    bath = getattr(sim_module, "bath", None) or getattr(sim_module, "bath_system", None)
    bath_to_dict = getattr(bath, "to_dict", None)
    if callable(bath_to_dict):
        _setdefault("bath", bath_to_dict())

    if extra_payload:
        for key, value in extra_payload.items():
            if value is not None:
                if isinstance(value, Mapping) and key in payload and isinstance(payload[key], dict):
                    merged = dict(payload[key])
                    merged.update(value)
                    payload[key] = merged
                else:
                    payload[key] = value

    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return sidecar


def resolve_run_prefix(
    system: "AtomicSystem", sim_config: "SimulationConfig", data_root: Path | str
) -> tuple[Path, str]:
    """Return ``(directory, prefix)`` for outputs of ``system`` and ``sim_config``."""

    base = Path(
        generate_deterministic_data_base(system, sim_config, data_root=data_root, ensure=True)
    )
    return base.parent, base.name


def save_run_artifact(
    sim_module: "SimulationModuleOQS",
    *,
    signal_arrays: Sequence[np.ndarray],
    t_det: np.ndarray,
    metadata: Mapping[str, Any],
    frequency_sample_cm: Sequence[float],
    sample_id: str,
    data_root: Path | str,
    t_coh: np.ndarray | None = None,
) -> Path:
    """Persist a single run (t_coh × sample) as a compressed ``.npz`` artifact."""

    directory, prefix = resolve_run_prefix(
        sim_module.system, sim_module.simulation_config, data_root
    )

    t_index = int(metadata.get("t_index", 0))
    combination_index = int(metadata.get("combination_index", 0))
    filename = f"{prefix}_run_t{t_index:03d}_c{combination_index:04d}_s{sample_id[:8]}.npz"
    abs_path = directory / filename

    signal_types = list(metadata.get("signal_types", []))
    if len(signal_types) != len(signal_arrays):
        raise ValueError("signal_types metadata must match number of signal arrays")

    payload: dict[str, Any] = {
        _T_DET_KEY: np.asarray(t_det, dtype=float),
        _SAMPLE_KEY: np.asarray(frequency_sample_cm, dtype=float),
        _META_KEY: np.array(_json_dumps({**metadata, "sample_id": sample_id}), dtype=np.str_),
    }

    if t_coh is not None:
        payload[_T_COH_KEY] = np.asarray(t_coh, dtype=float)

    for sig, data in zip(signal_types, signal_arrays):
        payload[f"{_SIGNAL_PREFIX}{sig}"] = np.asarray(data)

    np.savez_compressed(abs_path, **payload)
    return abs_path


def load_run_artifact(path: Path | str) -> dict[str, Any]:
    """Load a run artifact produced by :func:`save_run_artifact`."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as bundle:
        contents = {key: bundle[key] for key in bundle.files}

    metadata = json.loads(str(contents.pop(_META_KEY).item())) if _META_KEY in contents else {}
    sim_cfg = json.loads(str(contents.pop(_SIM_CFG_KEY).item())) if _SIM_CFG_KEY in contents else {}
    system = json.loads(str(contents.pop(_SYSTEM_KEY).item())) if _SYSTEM_KEY in contents else {}
    laser = json.loads(str(contents.pop(_LASER_KEY).item())) if _LASER_KEY in contents else None
    job_meta = (
        json.loads(str(contents.pop(_JOB_META_KEY).item())) if _JOB_META_KEY in contents else None
    )

    directory, prefix = _split_prefix(path)
    sidecar_payload = _load_sidecar(directory, prefix)

    sidecar_sim_cfg = sidecar_payload.get("simulation_config")
    if sidecar_sim_cfg:
        sim_cfg = sidecar_sim_cfg

    sidecar_system = sidecar_payload.get("system")
    if sidecar_system:
        system = sidecar_system

    sidecar_laser = sidecar_payload.get("laser")
    if sidecar_laser is not None:
        laser = sidecar_laser

    bath = sidecar_payload.get("bath")
    if bath is None:
        bath = sidecar_payload.get("bath_system")

    sidecar_job = sidecar_payload.get("job")
    if sidecar_job is not None:
        job_meta = sidecar_job
    elif "job_metadata" in sidecar_payload:
        job_meta = sidecar_payload.get("job_metadata")
    elif "metadata" in sidecar_payload and isinstance(sidecar_payload["metadata"], Mapping):
        job_meta = sidecar_payload["metadata"]

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
        "job_metadata": job_meta,
    }


def write_sidecar_json(
    system: "AtomicSystem",
    sim_config: "SimulationConfig",
    data_root: Path | str,
    *,
    name: str,
    payload: Mapping[str, Any],
) -> Path:
    """Write a JSON sidecar file next to the generated run artifacts."""

    directory, prefix = resolve_run_prefix(system, sim_config, data_root)
    path = directory / f"{prefix}_{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def save_simulation_data(
    sim_module: SimulationModuleOQS,
    metadata: dict,
    datas: List[np.ndarray],
    t_det: np.ndarray,
    t_coh: Optional[np.ndarray] = None,
    *,
    data_root: Path | str,
) -> Path:
    """Compatibility wrapper that forwards to :func:`save_run_artifact`."""

    freq_vector = np.asarray(sim_module.system.frequencies_cm, dtype=float)
    sample_id = metadata.get("sample_id") or compute_sample_id(freq_vector)

    extended_meta: dict[str, Any] = {
        "signal_types": sim_module.simulation_config.signal_types,
        "t_index": 0,
        "combination_index": 0,
        "sample_index": 0,
        "sample_size": sim_module.simulation_config.sample_size,
        **metadata,
        "sample_id": sample_id,
    }

    ensure_run_sidecar(sim_module, data_root=data_root, extra_payload=None)

    return save_run_artifact(
        sim_module,
        signal_arrays=datas,
        t_det=t_det,
        metadata=extended_meta,
        frequency_sample_cm=freq_vector,
        sample_id=sample_id,
        data_root=data_root,
        t_coh=t_coh,
    )


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


def list_available_files(abs_base_dir: Path) -> List[str]:
    """Return a sorted list of run artifacts stored beneath ``abs_base_dir``."""

    base = Path(abs_base_dir).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f"Base directory does not exist: {base}")
    if not base.is_dir():
        raise NotADirectoryError(f"Expected a directory path, got: {base}")

    pattern = str(base / "**" / "*_run_*.npz")
    artifacts = sorted(glob.glob(pattern, recursive=True))

    if not artifacts:
        print(f"No run artifacts found under {base}")
        return []

    print(f"Found {len(artifacts)} run artifact(s) under {base}")
    for file_path in artifacts:
        print(f"file: {file_path}")

    return artifacts
