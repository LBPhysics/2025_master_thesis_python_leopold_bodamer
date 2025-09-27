# TODO Get rid of this unnecessary stuff
"""
File naming utilities for qspectro2d.
This module provides functionality for generating standardized filenames
and directory paths for simulation data and plots.
"""

from __future__ import annotations

# IMPORTS
from pathlib import Path
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from qspectro2d.core.atomic_system.system_class import AtomicSystem
    from qspectro2d.core.simulation.sim_config import SimulationConfig


def _generate_base_filename(sim_config: "SimulationConfig") -> str:
    """Deterministic base filename including coherence time and inhom index.

        1d_inhom_avg_t_coh_<value>_inhom_<idx>
        2d_inhom_000
        2d_inhom_avg

        <value>: coherence time (fs), compact float (up to 6 significant digits)
        <idx>  : zero-padded integer ("000" for homogeneous / averaged runs)

    Uses the official ``SimulationConfig.inhom_index`` field introduced for
    bookkeeping."""

    parts: list[str] = []
    parts.append(sim_config.sim_type)
    # Collect inhom averaged markers
    if sim_config.inhom_averaged is True:
        parts.append("inhom_avg")

    t_coh_val = sim_config.t_coh
    if t_coh_val is not None:
        t_coh_part = f"t_coh_{float(t_coh_val):.6g}"  # removes trailing zeros if possible
        parts.append(t_coh_part)

    parts.append(f"inhom_{int(sim_config.inhom_index):03d}")
    base = f"{'_'.join(parts)}"
    return base


def _generate_unique_filename(path: Union[str, Path], base_name: str) -> str:
    """Return a unique base filename (no extension) inside ``path``.

    Strategy
    --------
    1. Create the directory if missing.
    2. Collect existing stems once (fast, avoids repeated directory scans).
    3. If ``base_name`` is free -> return it.
    4. Else append an incrementing ``_k`` suffix until free.

    The uniqueness check is extension-agnostic: any file whose ``stem`` matches
    the candidate blocks reuse, regardless of its extension(s). This means that
    saving multiple plot formats (``.png``, ``.pdf``, ``.svg``) for the *same* plot
    should be done by calling this function ONCE and then reusing the returned
    base for all extensions. Calling it separately for each extension would (by
    design) enumerate suffixed names.

    Concurrency note
    ----------------
    This is not atomic across processes. If you need true cross-process safety
    you would have to implement a lock file (e.g. ``candidate.lock``) with an
    ``os.open(..., O_CREAT|O_EXCL)`` pattern. Not needed for typical sequential
    simulation workflows and omitted for simplicity.
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)

    # Pre-collect existing stems (single pass over directory).
    try:
        existing_stems = {p.stem for p in dir_path.iterdir() if p.is_file()}
    except FileNotFoundError:
        existing_stems = set()

    # Fast path.
    if base_name not in existing_stems:
        return str(dir_path / base_name)

    # Collision: increment until a free stem is found.
    counter = 1
    while True:
        candidate = f"{base_name}_{counter}"
        if candidate not in existing_stems:
            return str(dir_path / candidate)
        counter += 1


def generate_base_sub_dir(sim_config: SimulationConfig, system: AtomicSystem) -> Path:
    """
    Generate standardized subdirectory path based on system and configuration.
    WILL BE subdir of DATA_DIR OR FIGURES_DIR

    Args:
        info_config: Dictionary containing simulation parameters
        system: System parameters object

    Returns:
        Path: Relative path for data storage
    """
    # Base directory based on number of atoms and solver type
    parts = []

    # Add simulation dimension (1d/2d)
    sim_f = sim_config.to_dict()
    sys_f = system.to_dict()

    parts.append(f"{sim_f['sim_type']}_spectroscopy")

    # Add system details
    n_atoms = sys_f.get("n_atoms")
    n_chains = sys_f.get("n_chains")
    n_rings = sys_f.get("n_rings")
    if n_atoms > 2:
        if n_chains is not None and n_rings is not None:
            parts.append(f"{n_atoms}({n_chains}x{n_rings})_atoms")
    else:
        parts.append(f"{n_atoms}_atoms")

    # Add solver if available
    parts.append(sim_f.get("ode_solver") or "solver?")

    # Add RWA if available
    parts.append("RWA" if sim_f.get("rwa_sl") else "noRWA")

    # For inhomogeneous batches, avoid embedding per-run numeric parameters to keep a stable folder
    n_inhomogen = int(sim_f.get("n_inhomogen", 1) or 1)
    if n_inhomogen > 1:
        # Generic markers so all inhom configs land in the same directory
        parts.append("inhom")

    if n_atoms > 1:
        # Add coupling strength if applicable. For inhom runs, avoid numeric per-run values.
        coupling_cm = sys_f.get("coupling_cm")
        if coupling_cm and coupling_cm > 0:
            parts.append(f"{round(coupling_cm, 0)}cm")

    # Add time parameters
    parts.append(
        f"t_dm{sim_f.get('t_det_max', 'na')}_t_wait_{sim_f.get('t_wait', 'na')}_dt_{sim_f.get('dt', 'na')}"
    )

    return Path(*parts)


def generate_deterministic_data_base(
    system: "AtomicSystem",
    sim_config: "SimulationConfig",
    *,
    data_root: Union[str, Path],
    ensure: bool = True,
) -> Path:
    """Return deterministic (non-enumerated) base path (no extension).

    This is the pure mapping from (system, sim_config) -> directory/filename stem
    without any collision resolution. Intended for tasks like discovery /
    skip-if-exists that need to glob for all variants belonging to the same
    logical parameter set.
    """
    relative_path = generate_base_sub_dir(sim_config, system)
    abs_path = Path(data_root) / relative_path
    if ensure:
        abs_path.mkdir(parents=True, exist_ok=True)
    base_name = _generate_base_filename(sim_config)
    return abs_path / base_name


def generate_unique_data_filename(
    system: "AtomicSystem",
    sim_config: "SimulationConfig",
    *,
    data_root: Union[str, Path],
    ensure: bool = True,
) -> str:
    """Return unique (possibly enumerated) base filename path (no extension).

    If the deterministic stem exists already (any file with that stem), an
    incrementing _<n> suffix is appended to the stem itself. The caller can
    then append domain-specific suffixes like ``_data`` / ``_info``.
    """
    deterministic = generate_deterministic_data_base(
        system, sim_config, data_root=data_root, ensure=ensure
    )
    unique = _generate_unique_filename(deterministic.parent, deterministic.name)
    return unique


def generate_unique_plot_filename(
    system: AtomicSystem,
    sim_config: SimulationConfig,
    domain: str,
    component: str | None = None,
    *,
    figures_root: Union[str, Path],
    ensure: bool = True,
) -> str:
    """
    Build a standardized filename for plots.

    Args:
        system: System parameters object
        info_config: Dictionary containing simulation parameters
        domain: Data domain ("time" or "freq")
        component: Optional component name ("real", "img", "abs", "phase")

    Returns:
        str: Standardized base filename for the plot (without extension)
    """
    # Validate domain
    if domain not in {"time", "freq"}:
        raise ValueError(f"Invalid domain '{domain}'. Expected 'time' or 'freq'.")

    # Validate component if provided
    if component and component not in {"real", "img", "abs", "phase"}:
        raise ValueError(
            f"Invalid component '{component}'. Expected one of 'real', 'img', 'abs', 'phase'."
        )

    # Start with basic structure
    relative_path = generate_base_sub_dir(sim_config, system)
    path = Path(figures_root) / relative_path
    if ensure:
        path.mkdir(parents=True, exist_ok=True)

    # Mirror flipped ordering for averaged markers so plots group nicely
    base_core = _generate_base_filename(sim_config)
    base_name = f"{base_core}_{domain}_domain"
    if component:
        base_name += f"_{component}"
    filename = _generate_unique_filename(path, base_name)
    return filename
