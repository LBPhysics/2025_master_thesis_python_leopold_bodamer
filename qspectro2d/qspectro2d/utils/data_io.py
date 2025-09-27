"""
Data I/O operations for qspectro2d.

This module provides functionality for loading and saving simulation data,
including standardized file formats and directory management.
"""

from __future__ import annotations

# IMPORTS
import numpy as np
import pickle
import glob
from pathlib import Path
from typing import Optional, List, TYPE_CHECKING
from qutip import BosonicEnvironment

# Type checking imports to avoid circular imports
if TYPE_CHECKING:
    from qspectro2d.core.laser_system.laser_class import LaserPulseSequence
    from qspectro2d.core.atomic_system.system_class import AtomicSystem
    from qspectro2d.core.simulation import SimulationConfig, SimulationModuleOQS
from qspectro2d.utils.file_naming import generate_unique_data_filename, _generate_unique_filename


# data saving functions
def save_data_file(
    abs_data_path: Path,
    metadata: dict,
    datas: List[np.ndarray],
    t_det: np.ndarray,
    t_coh: Optional[np.ndarray] = None,
) -> None:
    """Save spectroscopy data(s) with a single np.savez_compressed call.

    Distinctions:
      - Dimensionality (1D vs 2D) inferred from t_coh is None or not.
      - Single vs multi-component data inferred from provided `datas`.

        Stored keys:
            - Axes: 't_det' and optionally 't_coh'
            - One array per signal type stored under its signal name (metadata['signal_types'])
            - all other metadata key-value pairs are stored at the top level
    """
    try:
        abs_data_path.parent.mkdir(parents=True, exist_ok=True)

        # Infer dimensionality
        is_2d = t_coh is not None

        # Base payload
        payload: dict = {
            "t_det": t_det,
        }
        if is_2d:
            payload["t_coh"] = t_coh

        # Optional metadata (e.g., inhom batching info)
        for k, v in metadata.items():
            payload[k] = v

        # Validate and populate component keys
        signal_types = metadata["signal_types"]
        if len(signal_types) != len(datas):
            raise ValueError(
                f"Length of signal_types ({len(signal_types)}) must match number of datas ({len(datas)})"
            )

        for i, data in enumerate(datas):
            sig_key = signal_types[i]
            if is_2d:
                if not isinstance(data, np.ndarray) or data.shape != (
                    len(t_coh),
                    len(t_det),
                ):
                    raise ValueError(
                        f"2D data must have shape (len(t_coh), len(t_det)) = ({len(t_coh)}, {len(t_det)})"
                    )
            else:
                if not isinstance(data, np.ndarray) or data.shape != (len(t_det),):
                    raise ValueError(f"1D data must have shape (len(t_det),) = ({len(t_det)},)")
            payload[sig_key] = data

        # Single write
        np.savez_compressed(abs_data_path, **payload)

    except Exception as e:
        print(f"Failed to save data: {e}")
        raise


def save_info_file(
    abs_info_path: Path,
    system: "AtomicSystem",
    bath: BosonicEnvironment,
    laser: "LaserPulseSequence",
    sim_config: "SimulationConfig",
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
        with open(abs_info_path, "wb") as info_file:
            pickle.dump(
                {
                    "system": system,
                    "bath": bath,
                    "laser": laser,
                    # Store the SimulationConfig instance directly for full fidelity
                    "sim_config": sim_config,
                },
                info_file,
            )
        print(f"Info saved: {abs_info_path}")
    except Exception as e:
        print(f"Failed to save info: {e}")
        raise


def save_simulation_data(
    sim_module: SimulationModuleOQS,
    metadata: dict,
    datas: List[np.ndarray],
    t_det: np.ndarray,
    t_coh: Optional[np.ndarray] = None,
    *,
    data_root: Path | str,
) -> Path:
    """
    Save spectroscopy simulation data (numpy arrays) along with known axes in one file,
    and system parameters and configuration in another file.

    Parameters:
        datas (List[np.ndarray]): Simulation results (1D/2D or absorptive tuple).
        t_det (np.ndarray): Detection time axis.
        t_coh (Optional[np.ndarray]): Coherence time axis for 2D data.

    Returns:
        Path]: absolute path to data dir for the saved numpy data file and info file.
    """

    system: "AtomicSystem" = sim_module.system
    sim_config: "SimulationConfig" = sim_module.simulation_config
    bath: "BosonicEnvironment" = sim_module.bath
    laser: "LaserPulseSequence" = sim_module.laser

    # Deterministic non-enumerated base (no suffix yet)
    abs_base_path = Path(generate_unique_data_filename(system, sim_config, data_root=data_root))
    base_dir = abs_base_path.parent
    base_stem = abs_base_path.name  # e.g. 1d_t_coh_33.3_inhom_000 (no _data suffix yet)

    # Enumerate on the full data stem so collisions create pattern base_data_1, base_data_2, ...
    data_stem = f"{base_stem}_data"
    unique_data_stem = Path(_generate_unique_filename(base_dir, data_stem)).name
    abs_data_path = base_dir / f"{unique_data_stem}.npz"
    # Derive matching info stem by swapping first occurrence of '_data' with '_info'
    info_stem = unique_data_stem.replace("_data", "_info", 1)
    abs_info_path = base_dir / f"{info_stem}.pkl"

    save_data_file(abs_data_path, metadata, datas, t_det, t_coh=t_coh)
    save_info_file(abs_info_path, system, bath, laser, sim_config)
    return abs_data_path


# data loading functions
def load_data_file(abs_data_path: Path) -> dict:
    """
    Load numpy data file (.npz) from absolute path.

    Args:
        abs_data_path: Absolute path to the numpy data file (.npz)

    Returns:
        dict: Dictionary containing loaded numpy data arrays
    """
    try:
        print(f"Loading data: {abs_data_path}")

        with np.load(abs_data_path, allow_pickle=True) as data_file:
            data_dict = {key: data_file[key] for key in data_file.files}
        # Enforce required key
        if "t_det" not in data_dict:
            raise KeyError("Missing 't_det' axis in data file (new format requirement)")
        print(f"Loaded data: {abs_data_path}")
        return data_dict
    except Exception as e:
        print(f"Failed to load data: {abs_data_path}; error: {e}")
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
        print(f"Loading info: {abs_info_path}")

        # Try to load the file directly if it exists
        if abs_info_path.exists():
            with open(abs_info_path, "rb") as info_file:
                info = pickle.load(info_file)
            print(f"Loaded info: {abs_info_path}")
            return info

        # Gracefully handle absent info files (e.g., post-processed averaged data)
        print(f"Info file not found; continuing without it: {abs_info_path}")
        return {}
    except Exception as e:
        print(f"Failed to load info: {abs_info_path}; error: {e}")
        raise


def load_simulation_data(abs_path: Path | str) -> dict:
    """Load simulation data bundle from either a data or info file path.

    Supports both base and enumerated variants introduced by the auto-enumeration
    logic in `save_simulation_data`.

    Accepted patterns:
      - ``..._data.npz``  <-> ``..._info.pkl``
      - ``..._data_<n>.npz``  <-> ``..._info_<n>.pkl`` (enumerated pair)
      - ``..._info.pkl`` or ``..._info_<n>.pkl`` (will infer the corresponding data file)

    Args:
        abs_path: Path to either the data (.npz) OR info (.pkl) file. Enumerated
                  suffix (``_<n>``) is optional.
    Returns:
        dict with merged data + metadata (info) contents.
    Raises:
        FileNotFoundError / ValueError on malformed inputs.
    """
    import re

    p = Path(abs_path)
    s = str(p)

    # Backward compatibility patterns:
    # Old style enumeration: base_data_1.npz / base_info_1.pkl
    old_data_enum = re.compile(r"(.*)_data_(\d+)\.npz$")
    old_info_enum = re.compile(r"(.*)_info_(\d+)\.pkl$")
    # New style enumeration (enumerate base before suffix): base_1_data.npz / base_1_info.pkl
    new_data_enum = re.compile(r"(.*)_data\.npz$")  # handled by endswith first
    # Generic enumerated base followed by _data: (base with optional _<n>)_data.npz
    base_enum_data = re.compile(r"(.*)_data\.npz$")

    if s.endswith("_data.npz"):
        # Non-enumerated base pair
        abs_data_path = p
        abs_info_path = Path(s[:-9] + "_info.pkl")
    elif s.endswith("_info.pkl"):
        abs_info_path = p
        abs_data_path = Path(s[:-9] + "_data.npz")
    else:
        # Try old style enumeration first
        m_data_old = old_data_enum.match(s)
        m_info_old = old_info_enum.match(s)
        if m_data_old:
            base, idx = m_data_old.groups()
            abs_data_path = p
            abs_info_path = Path(f"{base}_info_{idx}.pkl")
        elif m_info_old:
            base, idx = m_info_old.groups()
            abs_info_path = p
            abs_data_path = Path(f"{base}_data_{idx}.npz")
        else:
            # New style: unique base already enumerated -> <base>_data / <base>_info
            if s.endswith("_data.npz"):
                abs_data_path = p
                abs_info_path = Path(s[:-9] + "_info.pkl")
            elif s.endswith("_info.pkl"):
                abs_info_path = p
                abs_data_path = Path(s[:-9] + "_data.npz")
            else:
                raise ValueError(
                    "Unrecognized filename pattern. Expected '*_data.npz' or '*_info.pkl' (including enumerated base variants)."
                )

    print(f"Loading data bundle: {abs_data_path}")
    data_dict = load_data_file(abs_data_path)
    info_dict = load_info_file(abs_info_path)

    # Combine data and info into a single dictionary (info may be empty if missing)
    info_dict = info_dict or {}
    return {**data_dict, **info_dict}


def list_available_files(abs_base_dir: Path) -> List[str]:
    """
    List all available data files in a directory with their metadata without loading the full data.

    Args:
        abs_base_dir: Base directory path absolute to data dir

    Returns:
        List[str]: Sorted list of data/info file paths (strings)
    """
    if not abs_base_dir.is_dir():
        print(f"Not a directory, using parent: {abs_base_dir}")
        abs_base_dir = abs_base_dir.parent
    if not abs_base_dir.exists():
        raise FileNotFoundError(f"Base directory does not exist: {abs_base_dir}")

    print(f"Listing data files in: {abs_base_dir}")

    # Find all data files recursively
    print(f"Listing data files in: {abs_base_dir}")

    # Find all data and info files recursively
    data_pattern = str(abs_base_dir / "**" / "*_data.npz")
    info_pattern = str(abs_base_dir / "**" / "*_info.pkl")

    data_files = glob.glob(data_pattern, recursive=True)
    info_files = glob.glob(info_pattern, recursive=True)

    ### Combine and sort all file paths
    all_files = data_files + info_files
    all_files.sort()

    if not all_files:
        print(f"No data or info files found: {abs_base_dir}")
        return []

    # Print summary
    print(f"Found {len(all_files)} files in {abs_base_dir}")
    for file_path in all_files:
        print(f"file: {file_path}")

    return all_files


# ---------------------------------------------------------------------------
# Discovery and grouping helpers (shared by stacking scripts)
# ---------------------------------------------------------------------------


def discover_1d_files(folder: Path) -> List[Path]:
    """Return sorted list of all *_data.npz files in the folder.

    New naming scheme: averaged outputs carry an in-filename prefix segment
    'inhom_avg' (e.g. '1d_inhom_avg_t_coh_50_inhom_000_data.npz').

    If any averaged files are present we return only those, to avoid stacking
    raw per-config files with duplicate t_coh values.
    """
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")
    candidates = sorted(folder.glob("*_data.npz"))
    avgs = [p for p in candidates if "/inhom_avg_" in str(p).replace("\\", "/")]
    return avgs if avgs else candidates


def derive_2d_folder(from_1d_folder: Path) -> Path:
    """Map 1D folder .../data/1d_spectroscopy/... -> .../data/2d_spectroscopy/..."""
    parts = list(from_1d_folder.parts)
    try:
        idx = parts.index("1d_spectroscopy")
    except ValueError as exc:
        raise ValueError("The provided path must include '1d_spectroscopy'") from exc
    parts[idx] = "2d_spectroscopy"
    return Path(*parts)


def collect_group_files(anchor: Path) -> List[Path]:
    """Find all files with the same inhom_group_id as `anchor` in its directory.

    The anchor must be a path to a single `_data.npz` file from an inhomogeneous 1D run.
    """
    try:
        base = load_simulation_data(anchor)
    except Exception as e:
        print(f"❌ Anchor file is corrupted: {anchor}")
        print(f"    Error: {e}")
        print("🔍 Searching for valid files in the same directory...")

        # Try to find any valid file in the same directory to get the group_id
        dir_path = Path(anchor).parent
        all_npz = list(dir_path.glob("*_data.npz"))

        base = None
        for p in all_npz:
            if "/inhom_avg_" in str(p).replace("\\", "/"):
                continue
            try:
                temp_data = load_simulation_data(p)
                if temp_data.get("inhom_enabled", False):
                    print(f"✅ Found valid anchor file: {p}")
                    base = temp_data
                    break
            except Exception:
                continue
        if base is None:
            raise FileNotFoundError(
                "No valid inhomogeneous files found in directory to determine group_id"
            )

    group_id = base.get("inhom_group_id")
    if group_id is None:
        raise ValueError("Missing inhom_group_id in anchor file metadata.")
    # Keep t_coh constant if present (multiple coherence delays in same folder)
    anchor_tcoh = base.get("t_coh_value", None)

    dir_path = Path(anchor).parent
    all_npz = list(dir_path.glob("*_data.npz"))
    matches: List[Path] = []
    for p in all_npz:
        # Skip any already averaged outputs (identified by 'inhom_avg' prefix segment)
        if "/inhom_avg_" in str(p).replace("\\", "/"):
            continue
        try:
            d = load_simulation_data(p)
        except Exception as e:
            print(f"⚠️  Skipping corrupted file: {p}")
            print(f"    Error: {e}")
            continue
        if d.get("inhom_enabled", False) and d.get("inhom_group_id") == group_id:
            if anchor_tcoh is None or np.isclose(
                float(d.get("t_coh_value", 0.0)), float(anchor_tcoh)
            ):
                matches.append(p)
    if not matches:
        raise FileNotFoundError("No matching inhomogeneous files found for group.")
    return sorted(matches)
