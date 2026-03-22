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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from qspectro2d.config import resolve_config, validate_config
from qspectro2d.config.factory import load_simulation
from qspectro2d.core.simulation.time_axes import (
    compute_t_coh,
    compute_t_det,
    compute_times_local,
)
from qspectro2d.diagnostics import check_the_solver
from qspectro2d.spectroscopy import sample_from_gaussian

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
    times_local: np.ndarray
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
    "build_job_metadata",
    "final_processed_filename",
    "pick_config_yaml",
    "prepare_workflow",
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

    validate_config(merged_cfg)
    sim = load_simulation(merged_cfg)
    effective_sim_type = str(sim.simulation_config.sim_type)

    _set_random_seed(rng_seed)

    n_inhom = int(sim.simulation_config.n_inhomogen)
    if n_inhom <= 0:
        raise ValueError("n_inhomogen must be positive")

    samples = sample_from_gaussian(
        n_samples=n_inhom,
        fwhm=float(sim.system.delta_inhomogen_cm),
        mu=np.asarray(sim.system.frequencies_cm, dtype=float),
    )

    t_coh_values = np.asarray(compute_t_coh(sim.simulation_config), dtype=float)
    t_det_axis = np.asarray(compute_t_det(sim.simulation_config), dtype=float)
    times_local = np.asarray(compute_times_local(sim.simulation_config), dtype=float)

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

    return PreparedWorkflow(
        config_path=resolved_config_path,
        merged_cfg=merged_cfg,
        sim=sim,
        sim_type=effective_sim_type,
        samples=np.asarray(samples, dtype=float),
        t_coh_values=t_coh_values,
        t_det_axis=t_det_axis,
        times_local=times_local,
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
    return {
        "sim_type": prepared.sim_type,
        "signal_types": list(prepared.sim.simulation_config.signal_types),
        "t_det": prepared.t_det_axis.tolist(),
        "t_coh": prepared.t_coh_values.tolist(),
        "n_inhom": int(prepared.sim.simulation_config.n_inhomogen),
        "n_t_coh": int(prepared.t_coh_values.size),
        "time_cut": float(prepared.time_cut if time_cut is None else time_cut),
        "job_dir": str(Path(job_dir).resolve()),
        "data_dir": str(Path(data_dir).resolve()),
        "figures_dir": str(Path(figures_dir).resolve()),
        "data_base_name": str(data_base_name),
        "data_base_path": str(Path(data_base_path).resolve()),
        "config_path": str(Path(config_path).resolve()),
        "merged_config": prepared.merged_cfg,
    }


def final_processed_filename(sim_type: str) -> str:
    sim_type = str(sim_type).strip().lower()
    if sim_type == "2d":
        return "2d_inhom_averaged.npz"
    return f"{sim_type}_inhom_averaged.npz"
