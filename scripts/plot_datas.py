"""
Unified 1D/2D Electronic Spectroscopy Data Plotting Script

Clean, flexible CLI to load results and produce time/frequency-domain plots.

Examples (Windows PowerShell):
    # Base path (no suffix) or an .npz file
    python plot_datas.py --abs_path "C:/path/to/data/run_001"
    python plot_datas.py --abs_path "C:/path/to/data/run_042.npz"
    # Control zero-padding factor for FFT (1 disables, default 10)
    python plot_datas.py --abs_path "C:/path/to/data/run_001" --extend 5
    # Plot only time or frequency domain
    python plot_datas.py --abs_path "C:/path/to/data/run_001" --no-freq OR --no-time
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from typing import Any, List, Optional, Sequence, Dict, cast
from qutip import BosonicEnvironment
import sys
import argparse
import numpy as np
import warnings

### Project-specific imports
from qspectro2d.visualization.plotting import (
    plot_1d_el_field,
    plot_2d_el_field,
)
from qspectro2d.spectroscopy.post_processing import (
    extend_time_domain_data,
    compute_spectra,
)

from qspectro2d import generate_unique_plot_filename
from qspectro2d.core.bath_system.bath_fcts import extract_bath_parameters
from qspectro2d import load_simulation_data


from thesis_paths import FIGURES_PYTHON_DIR

from plotstyle import init_style, save_fig

init_style()


# Suppress noisy but harmless warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in exp")


# Helper to collect paths from save_fig (which may return a single path or a list of paths)
def _collect_saved_paths(accumulator: List[str], saved: Any) -> None:
    """Append saved file path(s) to ``accumulator`` (handles list / scalar)."""
    if isinstance(saved, (list, tuple, set)):
        accumulator.extend([str(s) for s in saved])
    else:  # single path-like
        accumulator.append(str(saved))


def _extend_and_fft(
    loaded_data_and_info: Dict[str, Any],
    pad_factor: float,
    dimension: str,
):
    """Zero-pad in time (positive direction) and compute spectra.

    Returns (freq_axes, data_freq, freq_labels).
    """
    # Extract inputs from loaded dict
    sim_config = loaded_data_and_info.get("sim_config")
    signal_types: Sequence[str] = sim_config.signal_types
    t_det = loaded_data_and_info.get("t_det")
    t_coh = loaded_data_and_info.get("t_coh") if dimension == "2d" else None

    datas: Sequence[Any] = [loaded_data_and_info.get(st) for st in signal_types]
    if any(d is None for d in datas):
        missing_signals = [st for st, d in zip(signal_types, datas) if d is None]
        raise KeyError(f"Missing signal arrays for: {missing_signals}")

    # Positive-direction multiplicative pad (1 disables)
    pad_factor = max(1.0, float(pad_factor))

    # Extend all signals at once
    ext_datas, ext_t_det, ext_t_coh = extend_time_domain_data(
        list(datas),
        np.asarray(t_det),
        np.asarray(t_coh) if t_coh is not None else None,
        pad=pad_factor,
    )

    # Compute spectra with requested sign conventions
    nu_cohs, nu_dets, datas_nu, out_types = compute_spectra(
        ext_datas, list(signal_types), ext_t_det, ext_t_coh
    )

    # Keep detection axis first in tuple
    freq_axes = (nu_dets, nu_cohs) if nu_cohs is not None else nu_dets
    return freq_axes, datas_nu, out_types


def _plot_components(
    *,
    datas: Sequence[Any],
    signal_types: Sequence[str],
    axis_det,
    axis_coh,
    domain: str,
    components: Sequence[str],
    plot_func,
    base_metadata: Dict[str, Any],
    system,
    sim_config,
    dimension: str,
    **kwargs,
) -> List[str]:
    """Loop over components (and signals for time domain) producing figures.

    For both time and frequency domain we make one figure per (signal/label, component).
    Frequency labels come from the FFT step (e.g., 'rephasing','nonrephasing','absorptive').
    """
    saved_paths: List[str] = []
    for comp in components:
        try:
            if domain == "time":
                for data, st in zip(datas, signal_types):
                    md = {**base_metadata, "signal_type": st}
                    fig = (
                        plot_func(
                            axis_det=axis_det,
                            data=data,
                            domain=domain,
                            component=comp,
                            function_symbol=r"$E_{k_s}$",
                            **md,
                        )
                        if dimension == "1d"
                        else plot_func(
                            axis_det=axis_det,
                            axis_coh=axis_coh,
                            data=data,
                            domain=domain,
                            use_custom_colormap=True,
                            component=comp,
                            **md,
                        )
                    )
                    base_name = generate_unique_plot_filename(
                        system=system,
                        sim_config=sim_config,
                        domain=domain,
                        component=comp,
                        figures_root=FIGURES_PYTHON_DIR,
                    )
                    # Append the freq-domain label to the filename for clear linkage
                    safe_label = str(st).replace(" ", "_")
                    filename = f"{base_name}_{safe_label}"
                    saved = save_fig(
                        fig,
                        filename=filename,
                    )
                    _collect_saved_paths(saved_paths, saved)
            else:  # frequency domain: iterate and save one per spectrum label
                for data, st in zip(datas, signal_types):
                    md = {**base_metadata, "signal_type": st}
                    fig = (
                        plot_func(
                            axis_det=axis_det,
                            data=data,
                            domain=domain,
                            use_custom_colormap=True,
                            component=comp,
                            **kwargs,
                            **md,
                        )
                        if dimension == "1d"
                        else plot_func(
                            axis_det=axis_det,
                            axis_coh=axis_coh,
                            data=data,
                            domain=domain,
                            use_custom_colormap=True,
                            component=comp,
                            **kwargs,
                            **md,
                        )
                    )
                    base_name = generate_unique_plot_filename(
                        system=system,
                        sim_config=sim_config,
                        domain=domain,
                        component=comp,
                        figures_root=FIGURES_PYTHON_DIR,
                    )
                    safe_label = str(st).replace(" ", "_")
                    filename = f"{base_name}_{safe_label}"
                    saved = save_fig(
                        fig,
                        filename=filename,
                    )
                    _collect_saved_paths(saved_paths, saved)
        except Exception as e:
            # Defensive: continue other components even if one fails
            print(f"❌ Error plotting {dimension.upper()} {domain} {comp}: {e}")
    return saved_paths


def main():
    parser = argparse.ArgumentParser(
        description="Plot 1D or 2D electronic spectroscopy data (time/frequency domains)."
    )

    # Required input
    parser.add_argument(
        "--abs_path",
        type=str,
        default=None,
        help=(
            "Absolute path to the saved results file (ends with '_data.npz' or '_info.pkl'). "
            "Alternatively, pass the base path without suffix to auto-resolve."
        ),
    )

    parser.add_argument(
        "--extend",
        type=float,
        default=10.0,
        help="Zero-padding factor for (>t_max) before FFT (1 disables)",
    )
    parser.add_argument(
        "--section",
        type=float,
        nargs="+",
        default=(1.5, 1.7),
        help="Frequency window: 1D -> two floats (min max), 2D -> four floats (min max min max)",
    )

    # Plot selection (both enabled by default)
    parser.add_argument(
        "--no_time",
        dest="plot_time",
        action="store_false",
        default=True,
        help="Disable time-domain plotting",
    )
    parser.add_argument(
        "--no_freq",
        dest="plot_freq",
        action="store_false",
        default=True,
        help="Disable frequency-domain plotting",
    )

    args = parser.parse_args()

    try:
        # Parse section from command line arguments
        section: Optional[Any] = None
        if len(args.section) == 2:
            section = (args.section[0], args.section[1])
        elif len(args.section) == 4:
            section = (args.section[0], args.section[1], args.section[2], args.section[3])
        else:
            raise ValueError("--section expects 2 (1D/2D) or 4 (2D) floats")

        # Plot selections (CLI flags)
        plot_time = bool(args.plot_time)
        plot_freq = bool(args.plot_freq)

        print(f"🔄 Loading: {args.abs_path}")
        loaded_data_and_info = load_simulation_data(abs_path=args.abs_path)

        # Quick probe to decide dimension and basic info
        sim_config = loaded_data_and_info["sim_config"]
        is_2d = sim_config.sim_type == "2d"
        dimension = "2d" if is_2d else "1d"

        # --- Plotting logic (formerly plot_data function) ---
        from qspectro2d.core.atomic_system.system_class import AtomicSystem
        from qspectro2d.core.laser_system.laser_class import LaserPulseSequence
        from qspectro2d.core.simulation import SimulationConfig

        # Extract required objects
        system = cast(AtomicSystem, loaded_data_and_info.get("system"))
        bath_env = cast(BosonicEnvironment, loaded_data_and_info.get("bath"))
        laser = cast(LaserPulseSequence, loaded_data_and_info.get("laser"))
        sim_config = cast(SimulationConfig, loaded_data_and_info.get("sim_config"))
        t_det = loaded_data_and_info.get("t_det")
        t_coh = loaded_data_and_info.get("t_coh") if dimension == "2d" else None

        # Print axes info
        n_t_det = len(t_det) if t_det is not None else 0
        det_rng = f"[{float(t_det[0]):.2f},{float(t_det[-1]):.2f}] fs" if n_t_det > 0 else "[—]"
        if is_2d:
            n_t_coh = len(t_coh) if t_coh is not None else 0
            coh_rng = f"[{float(t_coh[0]):.2f},{float(t_coh[-1]):.2f}] fs" if n_t_coh > 0 else "[—]"
        else:
            n_t_coh, coh_rng = 0, "—"
        print(
            f"   Axes: t_det n={n_t_det} {det_rng}; t_coh n={n_t_coh if is_2d else '—'} {coh_rng if is_2d else ''}"
        )

        signal_types: Sequence[str] = sim_config.signal_types

        # Collect raw data arrays
        datas = [loaded_data_and_info.get(st) for st in signal_types]
        if any(d is None for d in datas):
            missing_signals = [st for st, d in zip(signal_types, datas) if d is None]
            raise KeyError(f"Missing signal arrays for: {missing_signals}")

        # Detect if time-domain signals are all-zero (informative warning only)
        try:
            if datas and all(
                isinstance(a, np.ndarray) and a.size > 0 and np.allclose(a, 0) for a in datas
            ):
                print("⚠️  All-zero time-domain signals detected.")
        except Exception:
            pass

        # Prepare metadata for plotting
        ode_solver = sim_config.ode_solver
        if ode_solver == "BR" or ode_solver == "Paper_eqs":
            w0 = system.frequencies_fs[0]
            bath_dict = extract_bath_parameters(bath_env, w0)
        else:  # NOTE have to adjust if making ME also time-dep.
            bath_dict = {
                "deph_rate": 1 / 100,
                "down_rate": 1 / 300,
            }

        sim_dict = sim_config.to_dict()
        sim_dict.pop("signal_types", None)  # Remove to avoid duplication
        laser_dict = {k: v for k, v in laser.to_dict().items() if k != "pulses"}
        meta = {**system.to_dict(), **bath_dict, **laser_dict, **sim_dict}

        # Announce plotting context
        print(f"➡️  Plotting: dimension={dimension}, signals={list(signal_types)}")
        print(f"   t_det: n={len(t_det)} range=[{t_det[0]:.2f},{t_det[-1]:.2f}] fs")
        if dimension == "2d" and t_coh is not None:
            print(f"   t_coh: n={len(t_coh)} range=[{t_coh[0]:.2f},{t_coh[-1]:.2f}] fs")

        # Parse and normalize section formats
        pad_factor = float(args.extend)
        components = ["real", "abs", "img"]

        # Normalize section format based on dimension
        plot_section = None
        if section is not None and isinstance(section, (list, tuple)):
            try:
                if len(section) == 2:
                    if dimension == "1d":
                        plot_section = (float(section[0]), float(section[1]))
                    else:  # 2D: use same range for both axes
                        plot_section = (
                            (float(section[0]), float(section[1])),
                            (float(section[0]), float(section[1])),
                        )
                elif len(section) == 4:
                    if dimension == "1d":
                        plot_section = (float(section[0]), float(section[1]))
                    else:  # 2D: separate ranges for each axis
                        plot_section = (
                            (float(section[0]), float(section[1])),
                            (float(section[2]), float(section[3])),
                        )
            except Exception:
                plot_section = None

        # Choose plotting function
        plot_func = plot_1d_el_field if dimension == "1d" else plot_2d_el_field

        # ---- Time domain plotting ----
        all_saved: List[str] = []
        if plot_time:
            print("📊 Time domain ...")
            saved = _plot_components(
                datas=datas,
                signal_types=signal_types,
                axis_det=t_det,
                axis_coh=t_coh,
                domain="time",
                components=components,
                plot_func=plot_func,
                base_metadata=meta,
                system=system,
                sim_config=sim_config,
                dimension=dimension,
            )
            all_saved.extend(saved)
            for p in saved:
                print(f"   💾 {p}")

        # ---- Frequency domain plotting ----
        if plot_freq:
            print(f"📊 Frequency domain ... (extend={pad_factor})")
            try:
                freq_axes, freq_datas, kept_types = _extend_and_fft(
                    loaded_data_and_info=loaded_data_and_info,
                    pad_factor=pad_factor,
                    dimension=dimension,
                )
                axis_det_f, axis_coh_f = (freq_axes, None) if dimension == "1d" else freq_axes
                saved = _plot_components(
                    datas=freq_datas,
                    signal_types=kept_types,
                    axis_det=axis_det_f,
                    axis_coh=axis_coh_f,
                    domain="freq",
                    components=components,
                    plot_func=plot_func,
                    base_metadata=meta,
                    system=system,
                    sim_config=sim_config,
                    dimension=dimension,
                    section=plot_section,
                )
                all_saved.extend(saved)
                for p in saved:
                    print(f"   💾 {p}")
            except Exception as e:
                print(f"❌ Frequency domain skipped: {e}")

        plt.close("all")

        # Final summary
        print(f"✅ Done. Saved {len(all_saved)} file(s).")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
