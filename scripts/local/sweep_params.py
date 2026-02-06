"""Sweep key simulation parameters and record runtime.

This script runs a one-factor-at-a-time (OFAT) sweep around a baseline YAML
configuration. It generates temporary YAML configs, runs calc_datas.py for each
case, and saves a CSV/JSON summary with timings.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SCRIPTS_DIR.parent
DATA_DIR = PROJECT_ROOT / "jobs" / "sweeps"
DEFAULT_CONFIG = SCRIPTS_DIR / "simulation_configs" / "_monomer.yaml"
BASELINE_OVERRIDES = {
    "config.t_det_max": 10.0,
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


@dataclass
class SweepCase:
    label: str
    overrides: dict[str, Any]


def set_nested(cfg: dict[str, Any], path: Iterable[str], value: Any) -> None:
    node = cfg
    keys = list(path)
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def apply_overrides(base_cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    cfg = json.loads(json.dumps(base_cfg))  # deep copy via JSON
    for dotted_key, value in overrides.items():
        path = dotted_key.split(".")
        set_nested(cfg, path, value)

    solver = cfg.get("config", {}).get("solver")
    if solver != "redfield":
        cfg.get("config", {}).pop("solver_options", None)
    return cfg


def build_ofat_cases(base_cfg: dict[str, Any]) -> list[SweepCase]:
    grids: dict[str, list[Any]] = {
        "config.solver": ["paper_eqs", "redfield"],
        "bath.bath_type": ["ohmic"],
        "bath.temperature": [0.1],
        "bath.coupling": [0.01],
        "laser.rwa_sl": [True, False],
        "config.t_det_max": [10.0, 20.0, 50.0],
        "config.n_inhomogen": [1, 2],
        "config.n_atoms": [1, 2],
        "config.t_coh": [0.0, 10.0, 20.0],
        "config.t_wait": [0.0, 10.0, 20.0],
        "config.dt": [1.0, 0.5, 0.1],
    }

    cases: list[SweepCase] = []
    keys = list(grids.keys())

    def _walk(idx: int, current: dict[str, Any]) -> None:
        if idx == len(keys):
            label_parts = [f"{key.replace('.', '_')}={current[key]}" for key in keys]
            cases.append(SweepCase("__".join(label_parts), dict(current)))
            return
        key = keys[idx]
        for value in grids[key]:
            current[key] = value
            _walk(idx + 1, current)

    _walk(0, {})
    return cases


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    extra_paths = [
        str(PROJECT_ROOT / "packages" / "qspectro2d" / "src"),
        str(PROJECT_ROOT / "packages" / "plotstyle" / "src"),
    ]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ";".join([p for p in extra_paths if p] + ([existing] if existing else []))
    return env


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a one-factor-at-a-time sweep and record runtime",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Baseline YAML config to sweep around",
    )
    parser.add_argument(
        "--sim_type",
        choices=["0d", "1d", "2d"],
        default="1d",
        help="Simulation dimensionality",
    )
    parser.add_argument(
        "--sim_types",
        nargs="+",
        default=None,
        help=(
            "Space- or comma-separated list of sim types to sweep (e.g., 1d 2d or 1d,2d). "
            "Overrides --sim_type."
        ),
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional label for the sweep folder",
    )
    args = parser.parse_args()

    base_path = Path(args.config).expanduser().resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"Config file not found: {base_path}")

    with base_path.open("r", encoding="utf-8") as handle:
        base_cfg = yaml.safe_load(handle)

    base_cfg = apply_overrides(base_cfg, BASELINE_OVERRIDES)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    label = args.label or base_path.stem

    if args.sim_types:
        raw_items: list[str] = []
        for item in args.sim_types:
            raw_items.extend(part for part in item.split(",") if part)
        sim_types = [item.strip() for item in raw_items if item.strip()]
    else:
        sim_types = [args.sim_type]

    valid_sim_types = {"0d", "1d", "2d"}
    invalid = [item for item in sim_types if item not in valid_sim_types]
    if invalid:
        raise ValueError(f"Invalid sim types: {invalid}. Allowed: {sorted(valid_sim_types)}")

    cases = build_ofat_cases(base_cfg)
    env = build_env()
    calc_datas = SCRIPTS_DIR / "local" / "calc_datas.py"

    for sim_type in sim_types:
        sweep_dir = DATA_DIR / f"{label}_{sim_type}_{timestamp}"
        sweep_dir.mkdir(parents=True, exist_ok=True)
        results: list[dict[str, Any]] = []

        print(f"Sweep directory: {sweep_dir}")
        print(f"Total cases: {len(cases)}")

        for idx, case in enumerate(cases, start=1):
            cfg = apply_overrides(base_cfg, case.overrides)
            case_path = sweep_dir / f"case_{idx:03d}.yaml"
            with case_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(cfg, handle, sort_keys=False)

            cmd = [
                sys.executable,
                str(calc_datas),
                "--sim_type",
                sim_type,
                "--config",
                str(case_path),
            ]

            print(f"[{idx}/{len(cases)}] {case.label}")
            start = time.perf_counter()
            proc = subprocess.run(cmd, env=env, cwd=str(SCRIPTS_DIR))
            elapsed = time.perf_counter() - start

            results.append(
                {
                    "index": idx,
                    "label": case.label,
                    "config_path": str(case_path),
                    "return_code": proc.returncode,
                    "runtime_s": round(elapsed, 3),
                }
            )

        json_path = sweep_dir / "summary.json"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
            handle.write("\n")

        csv_path = sweep_dir / "summary.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        print("Sweep complete.")
        print(f"Summary JSON: {json_path}")
        print(f"Summary CSV:  {csv_path}")


if __name__ == "__main__":
    main()
