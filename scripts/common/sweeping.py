"""Shared helpers for local and HPC parameter sweeps.

The main simplification here is that the sweep is now truly OFAT:
start from one baseline configuration and vary one parameter at a time.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from common.workflow import PROJECT_ROOT, SCRIPTS_DIR


DATA_DIR = PROJECT_ROOT / "jobs" / "sweeps"
DEFAULT_CONFIG = SCRIPTS_DIR / "simulation_configs" / "_monomer.yaml"
BASELINE_OVERRIDES = {
    "config.t_det": 10.0,
    "config.t_coh": 0.0,
    "config.t_wait": 0.0,
    "config.dt": 1.0,
    "config.n_inhomogen": 1,
    "config.n_atoms": 1,
    "config.solver": "paper_eqs",
    "bath.bath_type": "ohmic",
    "bath.coupling": 0.01,
    "bath.temperature": 0.1,
}


@dataclass(frozen=True)
class SweepCase:
    label: str
    overrides: dict[str, Any]


__all__ = [
    "BASELINE_OVERRIDES",
    "DATA_DIR",
    "DEFAULT_CONFIG",
    "SweepCase",
    "apply_overrides",
    "build_env",
    "build_ofat_cases",
    "load_yaml",
    "set_nested",
]


def load_yaml(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected YAML mapping at top level, got {type(payload)!r}")
    return payload


def set_nested(cfg: dict[str, Any], path: Iterable[str], value: Any) -> None:
    node = cfg
    keys = list(path)
    for key in keys[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    node[keys[-1]] = value


def _get_nested(cfg: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    node: Any = cfg
    for key in dotted_key.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def apply_overrides(base_cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    for dotted_key, value in overrides.items():
        set_nested(cfg, dotted_key.split("."), value)

    solver = cfg.get("config", {}).get("solver")
    if solver != "redfield":
        cfg.get("config", {}).pop("solver_options", None)
    return cfg


def build_ofat_cases(base_cfg: dict[str, Any]) -> list[SweepCase]:
    """Build a true one-factor-at-a-time sweep around ``base_cfg``.

    The baseline configuration itself is included as the first case. Every later
    case changes exactly one dotted-key path relative to that baseline.
    """
    grids: dict[str, list[Any]] = {
        "config.solver": ["paper_eqs", "redfield"],
        "bath.bath_type": ["ohmic"],
        "bath.temperature": [0.1],
        "bath.coupling": [0.01],
        "laser.rwa_sl": [True, False],
        "config.t_det": [10.0, 20.0, 50.0],
        "config.n_inhomogen": [1, 2],
        "config.n_atoms": [1, 2],
        "config.t_coh": [0.0, 10.0, 20.0],
        "config.t_wait": [0.0, 10.0, 20.0],
        "config.dt": [1.0, 0.5, 0.1],
    }

    cases = [SweepCase(label="baseline", overrides={})]
    for dotted_key, values in grids.items():
        baseline_value = _get_nested(base_cfg, dotted_key)
        for value in values:
            if value == baseline_value:
                continue
            label = f"{dotted_key.replace('.', '_')}={value}"
            cases.append(SweepCase(label=label, overrides={dotted_key: value}))
    return cases


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    extra_paths = [
        str(PROJECT_ROOT / "packages" / "qspectro2d" / "src"),
        str(PROJECT_ROOT / "packages" / "plotstyle" / "src"),
    ]
    existing = env.get("PYTHONPATH", "")
    parts = [p for p in extra_paths if p]
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env
