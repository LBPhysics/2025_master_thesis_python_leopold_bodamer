"""Run all combinations of (t_coh, inhomogeneity) locally without batching.

This script iterates over the full Cartesian product of coherence times and
inhomogeneous frequency samples, computing and saving each combination as an
individual file. It uses the same structure and naming conventions as the
HPC batch scripts (hpc_batch_dispatch.py and run_batch.py) for consistency.

The resulting files can be processed downstream:
- For 1D: average over inhomogeneity using avg_inhomogenity.py
- For 2D: stack per inhomogeneity sample using stack_times.py, then average

Examples:
    python calc_datas.py --sim_type 1d
    python calc_datas.py --sim_type 2d
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
import numpy as np
from dataclasses import dataclass

from qspectro2d.config.create_sim_obj import load_simulation
from qspectro2d.spectroscopy import check_the_solver, sample_from_gaussian
from qspectro2d.spectroscopy.e_field_1d import parallel_compute_1d_e_comps
from qspectro2d.utils.data_io import save_run_artifact, save_info_file
from qspectro2d.utils.file_naming import generate_unique_data_base
from qspectro2d.core.simulation.time_axes import compute_t_det, compute_t_coh

SCRIPTS_DIR = Path(__file__).parent.resolve()
for _parent in SCRIPTS_DIR.parents:
    if (_parent / ".git").is_dir():
        PROJECT_ROOT = _parent
        break
else:
    raise RuntimeError("Could not locate project root (missing .git directory)")

DATA_DIR = (PROJECT_ROOT / "data").resolve()
SIM_CONFIGS_DIR = SCRIPTS_DIR / "simulation_configs"
DATA_DIR.mkdir(exist_ok=True)


# Silence noisy but harmless warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in exp")
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*c_ops, e_ops, args and options will be keyword only from qutip 5\.3.*",
    module=r"qutip\.solver\.brmesolve",
)


# ---------------------------------------------------------------------------
# Helper function
# ---------------------------------------------------------------------------
def pick_config_yaml():
    """Pick a config YAML from scripts/simulation_configs.

    Preference order:
    1) Any file whose name starts with '_' (user-marked; Windows-safe)
    2) Otherwise, the first file in alphabetical order
    """
    cfg_candidates = sorted(SIM_CONFIGS_DIR.glob("*.yaml"))
    if not cfg_candidates:
        raise FileNotFoundError(
            f"No .yaml config files found in {SIM_CONFIGS_DIR}. Please add one."
        )
    # Prefer Windows-safe marker: leading underscore
    marked = [p for p in cfg_candidates if p.name.startswith("_")]
    return marked[0] if marked else cfg_candidates[0]


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

    _, time_cut = check_the_solver(sim)
    print(f"✅ Solver validated. time_cut = {time_cut:.6g}")

    data_base_path = generate_unique_data_base(
        sim.system, sim.simulation_config, data_root=DATA_DIR
    )

    n_inhom = sim.simulation_config.n_inhomogen
    if n_inhom <= 0:
        raise ValueError("n_inhom must be positive")

    # Set random seed if provided
    if args.rng_seed is not None:
        np.random.seed(args.rng_seed)

    base_freqs = np.asarray(sim.system.frequencies_cm, dtype=float)
    delta_cm = float(sim.system.delta_inhomogen_cm)
    samples = sample_from_gaussian(
        n_samples=n_inhom,
        fwhm=delta_cm,
        mu=base_freqs,
    )

    sim.simulation_config.sim_type = args.sim_type  # to ensure t_coh_axis has the right behavior
    t_coh_values = np.asarray(compute_t_coh(sim.simulation_config), dtype=float)
    combinations = build_combinations(t_coh_values, n_inhom)

    print(
        f"Prepared {len(combinations)} combination(s) → "
        f"|t_coh|={t_coh_values.size}, n_inhom={n_inhom}"
    )

    job_metadata = {
        "sim_type": args.sim_type,
        "signal_types": sim.simulation_config.signal_types,
        "t_det": compute_t_det(sim.simulation_config).tolist(),
        "t_coh": t_coh_values.tolist(),
        "n_inhom": n_inhom,
        "n_t_coh": int(t_coh_values.size),
        "n_combinations": len(combinations),
        "time_cut": float(time_cut),
        "data_base_path": str(data_base_path),
        "rng_seed": args.rng_seed,
    }

    info_path = data_base_path.parent / f"{data_base_path.name}.pkl"
    if not info_path.exists():
        save_info_file(
            info_path,
            sim.system,
            sim.simulation_config,
            bath=getattr(sim, "bath", None),
            laser=getattr(sim, "laser", None),
            extra_payload=job_metadata,
        )

    # Save samples and combinations for reference
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

        # Update simulation configuration for this combination
        freq_vector = samples[inhom_idx, :].astype(float)

        print(
            f"\n--- combo {global_idx + 1} / {len(combinations)}: t_idx={t_idx}, t_coh={t_coh_val:.4f} fs, "
            f"inhom_idx={inhom_idx} ---"
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
