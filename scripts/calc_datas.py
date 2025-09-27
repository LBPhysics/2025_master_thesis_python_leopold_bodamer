"""Flexible 1D / simple multi-``t_coh`` electronic spectroscopy runner (no inhomogeneity).

Two execution modes sharing the same underlying simulation object:
    1d mode:
        Processes a single coherence time (``t_coh`` from the _example.yaml <- marked with a leading underscore) and saves one file.

    2d mode:
        Treats every detection time value ``t_det`` as a coherence time point ``t_coh``
        and computes one 1D trace per point. Each processed ``t_coh`` produces an
        individual file which can later be stacked into a 2D dataset using ``stack_times.py``.

The resulting files are stored via ``save_simulation_data`` and contain
metadata keys required by downstream stacking & plotting scripts:
    - signal_types
    - t_coh_value

Examples:
    python calc_datas.py --sim_type 1d
    python calc_datas.py --sim_type 1d --n_batches 8 --batch_idx 2
    python calc_datas.py --sim_type 2d
    # 2D in batches (N batches, pick batch i)
    python calc_datas.py --sim_type 2d --n_batches 8 --batch_idx 0
"""

from __future__ import annotations

import argparse
import time
import warnings
import numpy as np
from qspectro2d.spectroscopy import sample_from_gaussian
from qspectro2d.spectroscopy.one_d_field import parallel_compute_1d_e_comps
from qspectro2d import save_simulation_data
from qspectro2d.config.create_sim_obj import create_base_sim_oqs
from qspectro2d.core.simulation import SimulationModuleOQS

from thesis_paths import DATA_DIR, SIM_CONFIGS_DIR


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
def _pick_config_yaml():
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


def _compute_e_components_for_tcoh(
    sim_oqs: SimulationModuleOQS, t_coh_val: float, *, time_cut: float
) -> list[np.ndarray]:
    """Compute polarization/field components for a single coherence time.

    This mutates the simulation object to set the current ``t_coh`` and
    corresponding pulse delay, then evaluates the 1D field components.
    """
    sim_oqs.update_delays(t_coh=t_coh_val)
    E_list = parallel_compute_1d_e_comps(sim_oqs=sim_oqs, time_cut=time_cut)
    return E_list


# ---------------------------------------------------------------------------
# Execution modes
# ---------------------------------------------------------------------------
def run_1d_mode(args) -> None:
    # Auto-pick YAML config (prefer '_' marked, else first sorted)
    config_path = _pick_config_yaml()
    print(f"🧩 Using config: {config_path.name}")
    sim_oqs, time_cut = create_base_sim_oqs(config_path=config_path)

    t_coh_val = float(sim_oqs.simulation_config.t_coh)
    sim_cfg = sim_oqs.simulation_config

    # Inhomogeneous 1D handling: only if (#samples > 1) AND (broadening > 0)
    n_inhom = sim_cfg.n_inhomogen
    delta_cm = sim_oqs.system.delta_inhomogen_cm

    if n_inhom > 1 and delta_cm > 0.0:
        # Sample site frequencies (cm^-1) for each configuration
        base_freqs_cm = np.asarray(sim_oqs.system.frequencies_cm, dtype=float)
        samples_cm = sample_from_gaussian(
            n_samples=n_inhom, fwhm=delta_cm, mu=base_freqs_cm
        )  # shape (n_inhom, n_sites)

        # Determine index subset for batching
        batch_idx: int = int(getattr(args, "batch_idx", 0))
        n_batches: int = int(getattr(args, "n_batches", 1))
        if n_batches == 1:
            indices = np.arange(n_inhom)
            batch_note = "all"
        else:
            chunks = np.array_split(np.arange(n_inhom), n_batches)
            if batch_idx < 0 or batch_idx >= len(chunks):
                raise IndexError(f"batch_idx {batch_idx} out of range for n_batches={n_batches}")
            indices = chunks[batch_idx]
            batch_note = f"batch {batch_idx+1}/{n_batches} (size={indices.size})"

        print(
            f"🎯 Running 1D inhomogeneous mode with t_coh = {t_coh_val:.2f} fs; "
            f"n_inhom={n_inhom}, Δ_inhom={delta_cm:g} cm⁻¹"
        )
        if indices.size:
            print(
                f"📦 Batching: {batch_note}; total configs={n_inhom}; "
                f"this job covers indices [{indices[0]}..{indices[-1]}]"
            )
        else:
            print(
                f"📦 Batching: {batch_note}; total configs={n_inhom}; this job covers no indices (empty chunk)"
            )

        saved_paths: list[str] = []
        start_time = time.time()
        for idx in indices.tolist():
            cfg_freqs = samples_cm[idx, :].astype(float).tolist()

            # Update system frequencies for this configuration
            sim_oqs.system.update_frequencies_cm(cfg_freqs)

            print(
                f"\n--- inhom_config={idx+1}/{n_inhom}  t_coh={t_coh_val:.2f} fs ---\n"
                f"    freqs_cm = {np.array2string(np.asarray(cfg_freqs), precision=2)}"
            )
            E_sigs = _compute_e_components_for_tcoh(sim_oqs, t_coh_val, time_cut=time_cut)

            # Persist dataset for this configuration
            # Update config bookkeeping for this specific inhom configuration
            sim_cfg.inhom_index = int(idx)

            metadata = {
                "signal_types": sim_cfg.signal_types,
                "t_coh_value": t_coh_val,
                "inhom_group_id": sim_cfg.inhom_group_id,
            }
            out_path = save_simulation_data(
                sim_oqs, metadata, E_sigs, t_det=sim_oqs.t_det, data_root=DATA_DIR
            )
            saved_paths.append(str(out_path))
            print(f"    ✅ Saved {out_path}")

        elapsed = time.time() - start_time
        print(
            f"\n✅ Finished computing {len(saved_paths)} inhomogeneous configs in {elapsed:.2f} s"
        )
        if saved_paths:
            example = saved_paths[-1]
            print("\n🎯 Next step (average inhomogeneous configs):")
            print(f"     python stack_inhomogenity.py --abs_path '{example}'")
        else:
            print("ℹ️  No files saved.")
        return

    # Homogeneous or single-config 1D
    print(f"🎯 Running 1D mode with t_coh = {t_coh_val:.2f} fs (from config)")

    E_sigs = _compute_e_components_for_tcoh(sim_oqs, t_coh_val, time_cut=time_cut)

    # Persist dataset
    # Homogeneous / single config bookkeeping
    sim_cfg.inhom_index = 0
    metadata = {
        "signal_types": sim_cfg.signal_types,
        "t_coh_value": t_coh_val,
        "inhom_group_id": sim_cfg.inhom_group_id,
    }
    abs_data_path = save_simulation_data(
        sim_oqs, metadata, E_sigs, t_det=sim_oqs.t_det, data_root=DATA_DIR
    )

    print(f"✅ Saved 1D result for t_coh={t_coh_val:.2f} fs.")
    print(f'\n🎯 To plot run: \npython plot_datas.py --abs_path "{abs_data_path}"')


def run_2d_mode(args) -> None:
    # Auto-pick YAML config (prefer '*' marked, else first sorted)
    config_path = _pick_config_yaml()
    print(f"🧩 Using config: {config_path.name}")
    sim_oqs, time_cut = create_base_sim_oqs(config_path=config_path)
    sim_cfg = sim_oqs.simulation_config
    print("🎯 Running 2D mode (iterate over t_det as t_coh)")

    # Reuse detection times as coherence-axis grid
    t_coh_vals = sim_oqs.t_det  # NOTE can only take every 10th value to check on local pc
    N_total = len(t_coh_vals)

    # Determine index subset for batching
    batch_idx: int = int(getattr(args, "batch_idx", 0))
    n_batches: int = int(getattr(args, "n_batches", 1))

    if n_batches == 1:
        indices = np.arange(N_total)
        batch_note = "all"
    else:
        # Split into contiguous chunks as evenly as possible
        # Use numpy to avoid off-by-one; ensures full coverage
        chunks = np.array_split(np.arange(N_total), n_batches)
        indices = chunks[batch_idx]
        batch_note = f"batch {batch_idx+1}/{n_batches} (size={indices.size})"

    if indices.size:
        span = (float(t_coh_vals[indices[0]]), float(t_coh_vals[indices[-1]]))
        print(
            f"📦 Batching: {batch_note}; total t_coh points={N_total}; "
            f"this job covers indices [{indices[0]}..{indices[-1]}] → t_coh in [{span[0]:.3g}, {span[1]:.3g}] fs"
        )
    else:
        print(
            f"📦 Batching: {batch_note}; total t_coh points={N_total}; this job covers no indices (empty chunk)"
        )

    saved_paths: list[str] = []
    start_time = time.time()
    for t_i in indices.tolist():
        t_coh_val = float(t_coh_vals[t_i])
        print(f"\n--- t_coh={t_coh_val:.2f} fs  [{t_i} / {N_total}]---")
        E_sigs = _compute_e_components_for_tcoh(sim_oqs, t_coh_val, time_cut=time_cut)

        sim_cfg.t_coh = t_coh_val
        sim_cfg.inhom_index = 0
        sim_cfg.inhom_enabled = (
            sim_cfg.inhom_enabled and sim_cfg.inhom_index > 0
        )  # stays False for 2D loop here
        metadata = {
            "signal_types": sim_cfg.signal_types,
            "t_coh_value": t_coh_val,
            "inhom_group_id": sim_cfg.inhom_group_id,
        }
        out_path = save_simulation_data(
            sim_oqs, metadata, E_sigs, t_det=sim_oqs.t_det, data_root=DATA_DIR
        )
        saved_paths.append(str(out_path))
        print(f"    ✅ Saved {out_path}")

    elapsed = time.time() - start_time
    print(f"\n✅ Finished computing {len(saved_paths)} t_coh points in {elapsed:.2f} s")
    if saved_paths:
        example = saved_paths[-1]
        if n_batches == 1:
            print("\n🎯 Next steps:")
            print("  1. Stack per-t_coh files into a 2D dataset:")
            print(f"     python stack_times.py --abs_path '{example}' --skip_if_exists")
            print("  2. Plot a single 1D file (example):")
            print(f"     python plot_datas.py --abs_path '{example}'")
        else:
            print("\n🎯 Next step:")
            print(f"     python hpc_plot_datas.py --abs_path '{example}'")
    else:
        print("ℹ️  No files saved.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 1D or multi-t_coh (2d-style) spectroscopy simulations (no inhomogeneity).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            """Examples:\n  python calc_datas.py --sim_type 1d\n  python calc_datas.py --sim_type 2d\n  python calc_datas.py --sim_type 2d --n_batches 8 --batch_idx 0"""
        ),
    )
    parser.add_argument(
        "--sim_type",
        type=str,
        default="1d",
        choices=["1d", "2d"],
        help="Execution mode (default: 1d)",
    )
    # Batching options (used for 2d mode and 1d-inhom mode)
    parser.add_argument(
        "--n_batches",
        type=int,
        default=1,
        help=("Split the 2d run or 1d-inhom run into N batches (default: 1, i.e., no batching)"),
    )
    parser.add_argument(
        "--batch_idx",
        type=int,
        default=0,
        help="Zero-based index selecting which batch to run (0..n_batches-1)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("SPECTROSCOPY SIMULATION")
    print(f"Mode: {args.sim_type}")

    if args.sim_type == "1d":
        run_1d_mode(args)
    else:  # 2d
        run_2d_mode(args)

    print("=" * 80)
    print("DONE")


if __name__ == "__main__":  # pragma: no cover
    main()
