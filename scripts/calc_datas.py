"""Run all combinations of (t_coh, inhomogeneity) locally without batching.

This script iterates over the Cartesian product of coherence times and
inhomogeneous frequency samples, saving each combination as an individual
artifact. The folder layout matches the HPC workflow so that downstream
processing remains uniform between local and cluster executions.

Examples
--------
    python calc_datas.py --sim_type 1d
    python calc_datas.py --sim_type 2d
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import warnings
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from qspectro2d.config.create_sim_obj import load_simulation
from qspectro2d.core.simulation.time_axes import compute_t_coh, compute_t_det
from qspectro2d.spectroscopy import check_the_solver, sample_from_gaussian
from qspectro2d.spectroscopy.e_field_1d import parallel_compute_1d_e_comps
from qspectro2d.utils.data_io import save_info_file, save_run_artifact
from qspectro2d.utils.job_paths import allocate_job_dir, ensure_job_layout, job_label_token

SCRIPTS_DIR = Path(__file__).parent.resolve()
for _parent in SCRIPTS_DIR.parents:
    if (_parent / ".git").is_dir():
        PROJECT_ROOT = _parent
        break
else:
    raise RuntimeError("Could not locate project root (missing .git directory)")

DATA_DIR = (PROJECT_ROOT / "data").resolve()
RUNS_ROOT = DATA_DIR / "jobs"
SIM_CONFIGS_DIR = SCRIPTS_DIR / "simulation_configs"
DATA_DIR.mkdir(exist_ok=True)
RUNS_ROOT.mkdir(exist_ok=True)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*c_ops, e_ops, args and options will be keyword only from qutip 5\.3.*",
    module=r"qutip\.solver\.brmesolve",
)


def pick_config_yaml(config_dir: Path | None = None) -> Path:
    """Return the preferred YAML configuration from ``config_dir``."""

    if config_dir is None:
        config_dir = SIM_CONFIGS_DIR
    candidates = sorted(config_dir.glob("*.yaml"))
    if not candidates:
        raise FileNotFoundError(f"No .yaml config files found in {config_dir}.")
    marked = [entry for entry in candidates if entry.name.startswith("_")]
    return marked[0] if marked else candidates[0]


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


def build_combinations(t_coh_values: np.ndarray, n_inhom: int) -> list[Combination]:
    combos: list[Combination] = []
    index = 0
    for t_idx, t_coh in enumerate(t_coh_values):
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


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all spectroscopy combinations (t_coh × inhom index) locally",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sim_type",
        choices=["0d", "1d", "2d"],
        default="2d",
        help="Simulation dimensionality",
    )
    parser.add_argument(
        "--rng_seed",
        type=int,
        default=None,
        help="Optional NumPy random seed for reproducible sampling",
    )
    args = parser.parse_args()

    config_path = pick_config_yaml().resolve()

    print("=" * 80)
    print("LOCAL ALL-COMBINATIONS RUNNER")
    print(f"Config path: {config_path}")

    sim = load_simulation(config_path, validate=True)
    print("✅ Simulation object constructed.")

    time_cut = check_the_solver(sim)
    print(f"✅ Solver validated. time_cut = {time_cut:.6g}")

    label_token = job_label_token(sim.simulation_config, sim.system, sim_type=args.sim_type)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    job_label = f"local_{label_token}_{timestamp}"
    job_dir = allocate_job_dir(RUNS_ROOT, job_label)
    job_paths = ensure_job_layout(job_dir, base_name="raw")
    data_base_path = job_paths.data_base_path

    print(f"Job workspace: {job_paths.job_dir}")

    config_copy_path = job_paths.job_dir / config_path.name
    if not config_copy_path.exists():
        shutil.copy2(config_path, config_copy_path)
        print(f"✅ Config file copied to {config_copy_path}")
    config_path = config_copy_path

    n_inhom = sim.simulation_config.n_inhomogen
    if n_inhom <= 0:
        raise ValueError("n_inhom must be positive")

    if args.rng_seed is not None:
        np.random.seed(args.rng_seed)

    base_freqs = np.asarray(sim.system.frequencies_cm, dtype=float)
    delta_cm = float(sim.system.delta_inhomogen_cm)
    samples = sample_from_gaussian(
        n_samples=n_inhom,
        fwhm=delta_cm,
        mu=base_freqs,
    )

    sim.simulation_config.sim_type = args.sim_type
    if args.sim_type == "1d" and getattr(sim.simulation_config, "t_coh_current", None) is None:
        sim.simulation_config.t_coh_current = float(sim.simulation_config.t_coh_max)
    t_coh_values = np.asarray(compute_t_coh(sim.simulation_config), dtype=float)

    # Derive a detection axis aligned with the active coherence grid to
    # avoid mismatches later when stacking/averaging.
    det_cfg = deepcopy(sim.simulation_config)
    if t_coh_values.size:
        det_cfg.t_coh_current = float(t_coh_values[0])
    t_det_axis = compute_t_det(det_cfg).tolist()
    combinations = build_combinations(t_coh_values, n_inhom)

    print(
        f"Prepared {len(combinations)} combination(s) → "
        f"|t_coh|={t_coh_values.size}, n_inhom={n_inhom}"
    )

    job_metadata = {
        "sim_type": args.sim_type,
        "signal_types": sim.simulation_config.signal_types,
        "t_det": t_det_axis,
        "t_coh": t_coh_values.tolist(),
        "n_inhom": n_inhom,
        "n_t_coh": int(t_coh_values.size),
        "n_combinations": len(combinations),
        "time_cut": float(time_cut),
        "t_coh_current": (
            float(sim.simulation_config.t_coh_current)
            if sim.simulation_config.t_coh_current is not None
            else None
        ),
        "job_label": job_label,
        "job_token": label_token,
        "data_base_path": str(data_base_path),
        "job_dir": str(job_paths.job_dir),
        "data_dir": str(job_paths.data_dir),
        "figures_dir": str(job_paths.figures_dir),
        "data_base_name": job_paths.base_name,
        "rng_seed": args.rng_seed,
    }

    info_path = data_base_path.parent / f"{data_base_path.name}.pkl"
    if not info_path.exists():
        save_info_file(
            info_path,
            sim.system,
            sim.simulation_config,
            bath=getattr(sim, "bath"),
            laser=getattr(sim, "laser"),
            extra_payload=job_metadata,
        )

    samples_target = data_base_path.parent / f"{data_base_path.name}_samples.npy"
    if not samples_target.exists():
        np.save(samples_target, samples.astype(float))

    combos_target = data_base_path.parent / f"{data_base_path.name}_combos.json"
    if not combos_target.exists():
        combos_dicts = [combo.to_dict() for combo in combinations]
        write_json(combos_target, {"combos": combos_dicts})

    print(f"Artifacts will be saved to {data_base_path.parent}")

    signal_types = sim.simulation_config.signal_types

    t_start = time.time()
    saved_paths: list[str] = []

    for combo in combinations:
        t_idx = combo.t_index
        inhom_idx = combo.inhom_index
        global_idx = combo.index
        t_coh_val = combo.t_coh

        freq_vector = samples[inhom_idx, :].astype(float)

        print(
            f"\n--- combo {global_idx + 1} / {len(combinations)}: "
            f"t_idx={t_idx}, t_coh={t_coh_val:.4f} fs, inhom_idx={inhom_idx} ---"
        )

        e_components = parallel_compute_1d_e_comps(
            config_path=str(config_path),
            t_coh=t_coh_val,
            freq_vector=freq_vector.tolist(),
            time_cut=time_cut,
        )

        metadata_combo = {
            "signal_types": signal_types,
            "t_coh_value": t_coh_val,
            "t_index": t_idx,
            "combination_index": global_idx,
            "sim_type": "1d" if args.sim_type == "2d" else args.sim_type,
            "sample_index": inhom_idx,
        }

        path = save_run_artifact(
            signal_arrays=e_components,
            metadata=metadata_combo,
            frequency_sample_cm=freq_vector,
            data_dir=data_base_path.parent,
            filename=f"{data_base_path.name}_run_t{t_idx:03d}_s{inhom_idx:03d}.npz",
        )

        saved_paths.append(str(path))
        print(f"    ✅ saved {path}")

    elapsed = time.time() - t_start
    print("=" * 80)
    print(f"Completed {len(saved_paths)} combination(s) in {elapsed:.2f} s")

    if saved_paths:
        print("Latest artifact:")
        print(f"  {saved_paths[-1]}")
        print("\n🎯 Next step:")
        print(f"     python process_datas.py --abs_path '{saved_paths[-1]}' --skip_if_exists")

    print("=" * 80)
    print("DONE")


if __name__ == "__main__":
    main()
