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
    metadata: Mapping[str, Any],
    frequency_sample_cm: Sequence[float],
    data_dir: Path | str,
    filename: str,
) -> Path:
    """Persist a single run (t_coh × sample) as a compressed ``.npz`` artifact."""

    directory = Path(data_dir)
    abs_path = directory / filename

    signal_types = list(metadata.get("signal_types", []))
    if len(signal_types) != len(signal_arrays):
        raise ValueError("signal_types metadata must match number of signal arrays")

    payload: dict[str, Any] = {
        _SAMPLE_KEY: np.asarray(frequency_sample_cm, dtype=float),
        _META_KEY: np.array(_json_dumps({**metadata}), dtype=np.str_),
    }

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

    sim_cfg = info.get("sim_config")
    system = info.get("system")
    laser = info.get("laser")
    bath = info.get("bath")
    job_metadata = {
        key: info[key]
        for key in (
            "job_dir",
            "data_dir",
            "figures_dir",
            "data_base_path",
            "data_base_name",
        )
        if key in info
    }

    signals: dict[str, np.ndarray] = {}
    for key in list(contents.keys()):
        if key.startswith(_SIGNAL_PREFIX):
            sig = key[len(_SIGNAL_PREFIX) :]
            signals[sig] = contents.pop(key)

    return {
        "path": path,
        "signals": signals,
        "t_det": np.asarray(info.get("t_det", []), dtype=float),
        "t_coh": np.asarray(info.get("t_coh", []), dtype=float),
        "frequency_sample_cm": contents.get(_SAMPLE_KEY),
        "metadata": metadata,
        "simulation_config": sim_cfg,
        "system": system,
        "laser": laser,
        "bath": bath,
        "job_metadata": job_metadata,
    }


def load_simulation_data(abs_path: Path | str) -> dict:
    """Load and unpack a run artifact produced by :func:`save_run_artifact`."""

    artifact = load_run_artifact(abs_path)
    signals = artifact.get("signals")
    bundle = dict(artifact)
    bundle["signal_types"] = list(signals.keys())

    for name, array in signals.items():
        bundle[name] = array
    if bundle.get("frequency_sample_cm") is None:
        raise KeyError("Run artifact is missing the frequency sample axis")

    job_meta = artifact.get("job_metadata")
    if job_meta:
        bundle["job_metadata"] = job_meta

    return bundle


"""Helpers for allocating and managing per-run job directories.

The local and HPC workflows both create a dedicated "job directory" that
collects raw artifacts, processed outputs, figures, and metadata in one
place.  This module provides tiny utilities to derive those paths.
"""

from dataclasses import dataclass


def generate_base_sub_dir(sim_config: SimulationConfig, system: AtomicSystem) -> Path:
    """
    Generate standardized subdirectory path based on system and configuration.
    WILL BE subdir of DATA_DIR

    Args:
        info_config: Dictionary containing simulation parameters
        system: System parameters object

    Returns:
        Path: Relative path for data storage
    """
    # Base directory based on number of atoms and solver type
    parts: list[str] = []

    # Add simulation dimension (1d/2d)
    sim_f = sim_config.to_dict()
    sys_f = system.to_dict()

    # Add system details
    n_atoms = sys_f.get("n_atoms")
    n_chains = sys_f.get("n_chains")
    n_rings = sys_f.get("n_rings")
    if n_atoms > 2:
        if n_chains is not None and n_rings is not None:
            parts.append(f"{n_atoms}({n_chains}x{n_rings})_atoms")
    else:
        parts.append(f"{n_atoms}_atoms")

    if n_atoms > 1:
        # Add coupling strength if applicable. For inhom runs, avoid numeric per-run values.
        coupling_cm = sys_f.get("coupling_cm")
        if coupling_cm and coupling_cm > 0:
            parts.append(f"{round(coupling_cm, 0)}cm")

    # Add solver if available
    parts.append(sim_f.get("ode_solver"))

    # Add RWA if available
    parts.append("RWA" if sim_f.get("rwa_sl") else "noRWA")

    # Add time parameters
    parts.append(f"t_dm{sim_f.get('t_det_max')}_t_wait{sim_f.get('t_wait')}_dt_{sim_f.get('dt')}")

    # For inhomogeneous batches, avoid embedding per-run numeric parameters to keep a stable folder
    n_inhomogen = int(sim_f.get("n_inhomogen", 1))
    if n_inhomogen > 1:
        parts.append(f"inhom_{sys_f.get('delta_inhomogen_cm', '0')}_cm_{n_inhomogen}n")

    base_path = Path(*parts)
    return base_path


@dataclass(frozen=True)
class JobPaths:
    """Resolved paths for a spectroscopy job run."""

    job_dir: Path
    data_dir: Path
    figures_dir: Path
    base_name: str

    @property
    def data_base_path(self) -> Path:
        """Base path used for raw artifacts (stem only)."""

        return self.data_dir / self.base_name


def ensure_job_layout(job_dir: Path, base_name: str = "run") -> JobPaths:
    """Create the canonical job layout and return the resolved paths."""

    resolved_job = job_dir.resolve()
    data_dir = resolved_job / "data"
    figures_dir = resolved_job / "figures"

    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    return JobPaths(
        job_dir=resolved_job,
        data_dir=data_dir,
        figures_dir=figures_dir,
        base_name=base_name,
    )


def allocate_job_dir(root: Path, base_label: str) -> Path:
    """Allocate a unique job directory under ``root`` using ``base_label``."""

    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    candidate = root / base_label
    if not candidate.exists():
        try:
            candidate.mkdir(parents=True)
            return candidate
        except FileExistsError:
            pass

    counter = 1
    while True:
        candidate = root / f"{base_label}_{counter:02d}"
        if not candidate.exists():
            try:
                candidate.mkdir(parents=True)
                return candidate
            except FileExistsError:
                pass
        counter += 1


def job_label_token(sim_config, system, *, sim_type: str | None = None) -> str:
    """Return a flattened identifier derived from the config and system."""

    base = generate_base_sub_dir(sim_config, system)
    parts = [part for part in base.parts if part and str(part).lower() != "none"]
    prefix = (sim_type or getattr(sim_config, "sim_type", None) or "sim").strip()
    token_parts = [prefix, *parts]
    return "_".join(token_parts)
