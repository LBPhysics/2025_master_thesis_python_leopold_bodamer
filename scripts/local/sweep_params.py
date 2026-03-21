"""Sweep key simulation parameters and record runtime.

This version performs a true one-factor-at-a-time sweep around one baseline
configuration. Shared sweep helpers live in ``common.sweeping`` so the local and
HPC sweep scripts stay consistent.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from common.sweeping import (
    BASELINE_OVERRIDES,
    DATA_DIR,
    DEFAULT_CONFIG,
    apply_overrides,
    build_env,
    build_ofat_cases,
    load_yaml,
)
from common.workflow import SCRIPTS_DIR


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
            "Space- or comma-separated list of sim types to sweep (e.g. 1d 2d or 1d,2d). "
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

    base_cfg = apply_overrides(load_yaml(base_path), BASELINE_OVERRIDES)

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
        print(f"Total OFAT cases: {len(cases)}")

        for idx, case in enumerate(cases, start=1):
            cfg = apply_overrides(base_cfg, case.overrides)
            case_path = sweep_dir / f"case_{idx:03d}.yaml"
            with case_path.open("w", encoding="utf-8") as handle:
                import yaml

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
                    "return_code": int(proc.returncode),
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
