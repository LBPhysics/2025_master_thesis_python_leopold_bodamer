#!/usr/bin/env python3
"""Generate and (optionally) submit SLURM jobs for combined t_coh × inhom runs.

This dispatcher generalizes the original 1D/2D workflows by creating the full
Cartesian product of coherence times and inhomogeneous samples. It validates the
simulation locally before launching any remote jobs, ensuring that solver
instabilities are detected early.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np

from calc_datas import _pick_config_yaml
from qspectro2d.config.create_sim_obj import load_simulation
from qspectro2d.spectroscopy import check_the_solver, sample_from_gaussian


SCRIPTS_DIR = Path(__file__).parent.resolve()
for _parent in SCRIPTS_DIR.parents:
    if (_parent / ".git").is_dir():
        PROJECT_ROOT = _parent
        break
else:  # pragma: no cover - defensive
    raise RuntimeError("Could not locate project root (missing .git)")

JOB_ROOT = SCRIPTS_DIR / "batch_jobs_generalized"
JOB_ROOT.mkdir(parents=True, exist_ok=True)


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
            "t_coh": float(self.t_coh),
            "inhom_index": int(self.inhom_index),
        }


def _set_random_seed(seed: int | None) -> None:
    if seed is not None:
        np.random.seed(seed)


def _coherence_axis(sim, sim_type: str) -> np.ndarray:
    if sim_type == "1d":
        return np.array([float(sim.simulation_config.t_coh)], dtype=float)
    return np.asarray(sim.t_det, dtype=float)


def _build_combinations(t_values: Sequence[float], n_inhom: int) -> list[Combination]:
    combos: list[Combination] = []
    index = 0
    for t_idx, t_coh in enumerate(t_values):
        for inhom_idx in range(n_inhom):
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


def _split_indices(n_items: int, n_batches: int) -> list[np.ndarray]:
    if n_batches <= 0:
        raise ValueError("n_batches must be positive")
    if n_items == 0:
        return [np.array([], dtype=int) for _ in range(n_batches)]
    return [chunk.astype(int) for chunk in np.array_split(np.arange(n_items), n_batches)]


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _relative_worker_path(job_dir: Path) -> str:
    rel = Path("../../run_generalized_batch.py")
    depth = len(job_dir.relative_to(SCRIPTS_DIR).parts)
    if depth != 2:  # Expect batch_jobs_generalized/<job_name>
        rel = Path("../" * depth) / "run_generalized_batch.py"
    return str(rel)


def _render_slurm_script(
    *,
    job_name: str,
    job_dir: Path,
    batch_idx: int,
    n_batches: int,
    sim_type: str,
    config_path: Path,
    combos_filename: str,
    samples_filename: str,
    time_cut: float,
    output_root: Path,
) -> str:
    worker_rel_path = _relative_worker_path(job_dir)
    config_arg = shlex.quote(str(config_path))
    output_arg = shlex.quote(str(output_root))
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=2G
#SBATCH --time=0-04:00:00

set -euo pipefail

SCRIPT_DIR="$(cd \"$(dirname \"${{BASH_SOURCE[0]}}\")\" && pwd)"
PROJECT_SCRIPTS_DIR="$(cd \"${{SCRIPT_DIR}}/../..\" && pwd)"

python \"${{PROJECT_SCRIPTS_DIR}}/{worker_rel_path}\" \
    --config-path {config_arg} \
  --combos-file \"${{SCRIPT_DIR}}/{combos_filename}\" \
  --samples-file \"${{SCRIPT_DIR}}/{samples_filename}\" \
  --time-cut {time_cut:.12g} \
  --sim-type {sim_type} \
  --batch-id {batch_idx} \
    --n-batches {n_batches} \
    --output-root {output_arg}
"""


def _submit_local_job(script_path: Path) -> str:
    sbatch = shutil.which("sbatch")
    if sbatch is None:
        raise RuntimeError("sbatch not found on PATH. Run this on your cluster login node.")

    result = subprocess.run(
        [sbatch, script_path.name],
        cwd=str(script_path.parent),
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dispatch generalized spectroscopy batches to an HPC cluster",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sim-type",
        choices=["1d", "2d"],
        default="2d",
        help="Simulation dimensionality",
    )
    parser.add_argument(
        "--n-inhom",
        type=int,
        default=None,
        help="Number of inhomogeneous samples (defaults to config value)",
    )
    parser.add_argument(
        "--n-batches",
        type=int,
        default=1,
        help="Total number of batches to split the combination space into",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=None,
        help="Optional NumPy random seed for reproducible sampling",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Override configuration YAML (otherwise auto-selected)",
    )
    parser.add_argument(
        "--job-name-prefix",
        type=str,
        default="spec",
        help="Prefix for generated SLURM job names",
    )
    parser.add_argument(
        "--no-submit",
        action="store_true",
        help="Only generate local artifacts; skip sbatch submission",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(PROJECT_ROOT / "data"),
        help="Root directory for saved simulation data",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    config_path = (
        Path(args.config_path).resolve()
        if args.config_path is not None
        else _pick_config_yaml().resolve()
    )

    print("=" * 80)
    print("GENERALIZED HPC DISPATCHER")
    print(f"Config path: {config_path}")

    sim = load_simulation(config_path, validate=True)
    print("✅ Simulation object constructed.")

    _, time_cut = check_the_solver(sim)
    print(f"✅ Solver validated. time_cut = {time_cut:.6g}")

    n_inhom = args.n_inhom or int(sim.simulation_config.n_inhomogen or 1)
    if n_inhom <= 0:
        raise ValueError("n_inhom must be positive")

    _set_random_seed(args.rng_seed)
    base_freqs = np.asarray(sim.system.frequencies_cm, dtype=float)
    delta_cm = float(sim.system.delta_inhomogen_cm)
    samples = sample_from_gaussian(
        n_samples=n_inhom,
        fwhm=delta_cm,
        mu=base_freqs,
    )

    t_values = _coherence_axis(sim, args.sim_type)
    combinations = _build_combinations(t_values, n_inhom)

    print(
        f"Prepared {len(combinations)} combination(s) → "
        f"|t_coh|={t_values.size}, n_inhom={n_inhom}, n_batches={args.n_batches}"
    )

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    job_label = f"{args.sim_type}_{n_inhom}inh_{t_values.size}t_{timestamp}"
    job_dir = JOB_ROOT / job_label
    job_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = job_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    samples_file = job_dir / "samples.npy"
    np.save(samples_file, samples.astype(float))

    metadata = {
        "config_path": str(config_path),
        "sim_type": args.sim_type,
        "n_inhom": n_inhom,
        "n_t_coh": int(t_values.size),
        "n_combinations": len(combinations),
        "n_batches": int(args.n_batches),
        "time_cut": float(time_cut),
        "job_label": job_label,
        "generated_at": timestamp,
        "output_root": str(Path(args.output_root).resolve()),
        "rng_seed": args.rng_seed,
    }
    _write_json(job_dir / "metadata.json", metadata)

    batch_indices = _split_indices(len(combinations), args.n_batches)
    script_paths: list[Path] = []
    for batch_idx, indices in enumerate(batch_indices):
        combos_subset = [combinations[i].to_dict() for i in indices.tolist()]
        combos_file = job_dir / f"batch_{batch_idx:03d}.json"
        _write_json(combos_file, {"combos": combos_subset})

        job_name = f"{args.job_name_prefix}b{batch_idx:02d}of{args.n_batches:02d}"
        slurm_text = _render_slurm_script(
            job_name=job_name,
            job_dir=job_dir,
            batch_idx=batch_idx,
            n_batches=args.n_batches,
            sim_type=args.sim_type,
            config_path=config_path,
            combos_filename=combos_file.name,
            samples_filename=samples_file.name,
            time_cut=time_cut,
            output_root=Path(args.output_root).resolve(),
        )
        script_path = job_dir / f"{job_name}.slurm"
        script_path.write_text(slurm_text, encoding="utf-8")
        script_paths.append(script_path)
        print(
            f"  batch {batch_idx}: {len(combos_subset)} combo(s) → {script_path.name}"
        )

    print(f"Artifacts written to {job_dir}")

    if args.no_submit:
        print("Skipping submission (--no-submit set).")
        return

    print("Submitting SLURM jobs...")
    for script_path in script_paths:
        submit_msg = _submit_local_job(script_path)
        print(f"  {script_path.name}: {submit_msg}")

    print("Done.")


if __name__ == "__main__":
    main()
