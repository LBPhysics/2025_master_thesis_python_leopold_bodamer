"""Shared workflow helpers for local and HPC spectroscopy scripts.

This module centralises the repeated preparation steps that used to live in both
``scripts/local/calc_datas.py`` and ``scripts/hpc/calc_dispatcher.py``:

- choose and resolve a config
- validate once
- construct the simulation once
- optionally run the solver diagnostic once
- generate t_coh / t_det axes
- sample inhomogeneous frequencies
- build the (t_coh, sample) combinations

The goal is not to add new features. It is to keep one clear source of truth for
workflow preparation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from qspectro2d.config import resolve_config, validate_config
from qspectro2d.config.factory import load_simulation
from qspectro2d.core.simulation.time_axes import (
    compute_t_coh,
    compute_t_det,
)
from qspectro2d.diagnostics import check_the_solver
from qspectro2d.spectroscopy import (
    sample_static_disorder,
)

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
for _parent in SCRIPTS_DIR.parents:
    if (_parent / ".git").is_dir():
        PROJECT_ROOT = _parent
        break
else:
    raise RuntimeError("Could not locate project root (missing .git directory)")

RUNS_ROOT = (PROJECT_ROOT / "jobs").resolve()
SIM_CONFIGS_DIR = SCRIPTS_DIR / "simulation_configs"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Combination:
    index: int
    t_index: int
    t_coh: float
    inhom_index: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "index": int(self.index),
            "t_index": int(self.t_index),
            "t_coh_value": float(self.t_coh),
            "inhom_index": int(self.inhom_index),
        }


@dataclass(frozen=True)
class PreparedWorkflow:
    config_path: Path
    merged_cfg: dict[str, Any]
    sim: Any
    sim_type: str
    samples: np.ndarray
    t_coh_values: np.ndarray
    t_det_axis: np.ndarray
    combinations: list[Combination]
    time_cut: float


__all__ = [
    "Combination",
    "PreparedWorkflow",
    "PROJECT_ROOT",
    "RUNS_ROOT",
    "SCRIPTS_DIR",
    "SIM_CONFIGS_DIR",
    "build_combinations",
    "build_job_dir_label",
    "build_job_metadata",
    "config_stem_token",
    "extract_job_unique_id",
    "format_slurm_job_name",
    "final_processed_filename",
    "pick_config_yaml",
    "prepare_workflow",
    "resolve_allocated_job_unique_id",
    "write_json",
]


def pick_config_yaml(config_dir: Path | None = None) -> Path:
    """Return the preferred YAML configuration from ``config_dir``.

    If any file starts with ``_``, the first such file is preferred. Otherwise,
    the lexicographically first YAML file is used.
    """
    directory = SIM_CONFIGS_DIR if config_dir is None else Path(config_dir)
    candidates = sorted(directory.glob("*.yaml"))
    if not candidates:
        raise FileNotFoundError(f"No .yaml config files found in {directory}.")
    marked = [entry for entry in candidates if entry.name.startswith("_")]
    return marked[0] if marked else candidates[0]


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _sanitize_token(value: object) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    token = re.sub(r"_+", "_", token).strip("._-")
    return token or "run"


def config_stem_token(config_path: Path | str) -> str:
    """Return a filesystem-safe token derived from the config filename stem."""
    return _sanitize_token(Path(config_path).stem)


def build_job_dir_label(config_path: Path | str, unique_handle: str) -> str:
    """Return the canonical job-directory label ``<timestamp>_<config_stem>``.

    ``unique_handle`` is expected to be the timestamp token such as
    ``DD_HHMMSS``.  If ``allocate_job_dir`` detects a collision it appends
    ``_NN`` to the directory name afterwards.
    """

    handle_token = _sanitize_token(unique_handle)
    config_token = config_stem_token(config_path)
    return "_".join(part for part in (handle_token, config_token) if part)


def extract_job_unique_id(job_dir: Path | str) -> str:
    """Extract the run-unique token from a job directory name.

    Current job directories are named ``DD_HHMMSS_config_stem``.
    This fallback parser also accepts legacy names where an older full-date
    timestamp token appeared at the end of the directory name.
    """

    name = Path(job_dir).name
    match = re.match(r"^(\d{2}_\d{6})(?:_|$)", name)
    if match:
        return match.group(1)
    match = re.search(r"(\d{8}_\d{6}(?:_\d{2})?)$", name)
    if match:
        return match.group(1)
    match = re.match(r"^(\d{8}_\d{6})(?:_|$)", name)
    if match:
        return match.group(1)
    match = re.search(r"(\d{2}_\d{6}(?:_\d{2})?)$", name)
    if match:
        return match.group(1)
    if "_" in name:
        return _sanitize_token(name.rsplit("_", 1)[-1])
    return _sanitize_token(name)


def resolve_allocated_job_unique_id(
    job_dir: Path | str,
    *,
    base_label: str,
    requested_unique_handle: str,
) -> str:
    """Resolve the effective unique id after ``allocate_job_dir`` suffixing.

    Use this immediately after allocating the directory so the stored
    ``job_unique_id`` preserves any collision suffix as
    ``DD_HHMMSS_NN`` without guessing from the config stem.
    """

    name = Path(job_dir).name
    base_token = _sanitize_token(base_label)
    unique_token = _sanitize_token(requested_unique_handle)

    if name == base_token:
        return unique_token

    prefix = f"{base_token}_"
    if name.startswith(prefix):
        suffix = name[len(prefix) :]
        if re.fullmatch(r"\d{2}", suffix):
            return f"{unique_token}_{suffix}"

    return extract_job_unique_id(name)


def format_slurm_job_name(*parts: object) -> str:
    """Join job-name tokens into a SLURM-safe label."""

    return "_".join(_sanitize_token(part) for part in parts if str(part).strip())


def build_combinations(t_coh_values: np.ndarray, n_inhom: int) -> list[Combination]:
    combos: list[Combination] = []
    index = 0
    for t_idx, t_coh in enumerate(np.asarray(t_coh_values, dtype=float)):
        for inhom_idx in range(int(n_inhom)):
            combos.append(
                Combination(
                    index=index,
                    t_index=t_idx,
                    t_coh=float(t_coh),
                    inhom_index=inhom_idx,
                )
            )
            index += 1
    return combos


def _set_random_seed(seed: int | None) -> None:
    if seed is not None:
        np.random.seed(int(seed))


def prepare_workflow(
    *,
    config_path: str | Path | None = None,
    sim_type: str | None = None,
    rng_seed: int | None = None,
    max_workers: int | None = None,
    run_solver_check: bool = True,
) -> PreparedWorkflow:
    """Resolve and prepare one full workflow definition.

    This function is intentionally shared by the local and HPC entry points so
    that configuration resolution, validation, simulation construction, axis
    generation, and inhomogeneous sampling happen once and the same way.
    """
    if config_path is None:
        resolved_config_path = pick_config_yaml().resolve()
    else:
        resolved_config_path = Path(config_path).expanduser().resolve()
        if not resolved_config_path.exists():
            raise FileNotFoundError(f"Config file not found: {resolved_config_path}")

    merged_cfg = resolve_config(str(resolved_config_path))
    if sim_type is not None:
        merged_cfg.setdefault("config", {})["sim_type"] = sim_type
    if max_workers is not None:
        merged_cfg.setdefault("config", {})["max_workers"] = int(max_workers)

    validate_config(merged_cfg, emit_runtime_warnings=False)
    sim = load_simulation(merged_cfg, emit_runtime_warnings=False)
    effective_sim_type = str(sim.simulation_config.sim_type)

    _set_random_seed(rng_seed)

    n_inhom = int(sim.simulation_config.n_inhomogen)
    if n_inhom <= 0:
        raise ValueError("n_inhomogen must be positive")

    mu = np.asarray(sim.system.frequencies_cm, dtype=float)
    fwhm = float(sim.system.delta_inhomogen_cm)
    corr = getattr(sim.system, "inhom_correlation", None)
    samples = sample_static_disorder(
        n_samples=n_inhom,
        fwhm=fwhm,
        mu=mu,
        corr=corr,
    )

    t_coh_values = np.asarray(compute_t_coh(sim.simulation_config), dtype=float)
    t_det_axis = np.asarray(compute_t_det(sim.simulation_config), dtype=float)

    # Ensure pulses do not overlap by enforcing minimum delays
    dt = float(sim.simulation_config.dt)
    if effective_sim_type in {"1d", "2d"}:
        t_coh_values = t_coh_values[t_coh_values >= dt]
        if t_coh_values.size == 0:
            raise ValueError(
                f"No valid t_coh values >= dt={dt:.3f} fs after filtering to prevent pulse overlap"
            )

    if t_det_axis.size == 0 and effective_sim_type != "0d":
        raise RuntimeError(
            "Invariant violation: empty global detection axis for non-0d run "
            f"(sim_type={effective_sim_type}, t_det={float(sim.simulation_config.t_det):.6g}, "
            f"dt={float(sim.simulation_config.dt):.6g})"
        )

    combinations = build_combinations(t_coh_values, n_inhom)
    time_cut = float(check_the_solver(sim)) if run_solver_check else float("inf")
    if time_cut <= 0:
        raise RuntimeError(f"Solver check returned non-positive time_cut={time_cut:.6g}")

    return PreparedWorkflow(
        config_path=resolved_config_path,
        merged_cfg=merged_cfg,
        sim=sim,
        sim_type=effective_sim_type,
        samples=np.asarray(samples, dtype=float),
        t_coh_values=t_coh_values,
        t_det_axis=t_det_axis,
        combinations=combinations,
        time_cut=time_cut,
    )


def build_job_metadata(
    prepared: PreparedWorkflow,
    *,
    job_dir: Path,
    data_dir: Path,
    figures_dir: Path,
    data_base_name: str,
    data_base_path: Path,
    config_path: Path,
    time_cut: float | None = None,
) -> dict[str, Any]:
    resolved_job_dir = Path(job_dir).resolve()
    resolved_config_path = Path(config_path).resolve()
    return {
        "sim_type": prepared.sim_type,
        "signal_types": list(prepared.sim.simulation_config.signal_types),
        "t_det": prepared.t_det_axis.tolist(),
        "t_coh": prepared.t_coh_values.tolist(),
        "n_inhom": int(prepared.sim.simulation_config.n_inhomogen),
        "n_t_coh": int(prepared.t_coh_values.size),
        "time_cut": float(prepared.time_cut if time_cut is None else time_cut),
        "job_dir": str(resolved_job_dir),
        "data_dir": str(Path(data_dir).resolve()),
        "figures_dir": str(Path(figures_dir).resolve()),
        "data_base_name": str(data_base_name),
        "data_base_path": str(Path(data_base_path).resolve()),
        "config_path": str(resolved_config_path),
        "config_stem": config_stem_token(resolved_config_path),
        "merged_config": prepared.merged_cfg,
    }


def final_processed_filename(sim_type: str) -> str:
    sim_type = str(sim_type).strip().lower()
    if sim_type == "2d":
        return "2d_inhom_averaged.npz"
    return f"{sim_type}_inhom_averaged.npz"
