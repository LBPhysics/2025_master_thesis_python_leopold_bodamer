"""
Plot processed spectroscopy data: plot time-domain and frequency-domain electric field for each signal type and component.

This script loads the final processed artifact from process_datas.py and generates
time-domain and frequency-domain plots for each signal type (e.g., 'rephasing', 'non-rephasing') and each
component ('real', 'img', 'abs', 'phase').

Examples:
    python plot_datas.py --abs_path '/path/to/final_averaged.npz'
    python plot_datas.py --abs_path '/path/to/final_averaged.npz' --extend 5
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import argparse

from qspectro2d import load_simulation_data, generate_unique_plot_filename
from qspectro2d.visualization.plotting import plot_el_field
from qspectro2d.spectroscopy.post_processing import compute_spectra
from qspectro2d.utils.file_naming import _generate_unique_filename
from plotstyle import save_fig

from calc_datas import PROJECT_ROOT

FIGURES_DIR = (PROJECT_ROOT / "figures").resolve()
FIGURES_DIR.mkdir(exist_ok=True)

# Section for cropping data
SECTION = (1.1, 2)  # or None #for no cropping

# Extend factor for zero-padding
EXTEND = 20.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot processed spectroscopy data in time and frequency domains."
    )
    parser.add_argument(
        "--abs_path", type=str, required=True, help="Path to the processed .npz file"
    )
    args = parser.parse_args()

    # Load the data
    data = load_simulation_data(args.abs_path)
    signals = data["signals"]
    t_det = data["t_det"]
    t_coh = data.get("t_coh")
    if t_coh is not None and t_coh.ndim == 0:
        t_coh = None
    system = data["system"]
    sim_config = data["simulation_config"]
    combined_dict = {**system.to_dict(), **sim_config.to_dict()}
    print(f"Loaded data from {args.abs_path}")
    print(f"Signal types: {list(signals.keys())}")
    print(f"Data dimension: {'2D' if t_coh is not None else '1D'}")

    components = ["real", "img", "abs"]
    saved_files = []

    for st, sig_data in signals.items():
        combined_dict["signal_type"] = st
        print(f"Plotting signal: {st}")
        for comp in components:
            fig = plot_el_field(
                axis_det=t_det,
                data=sig_data,
                axis_coh=t_coh,
                component=comp,
                domain="time",
                **combined_dict,
            )
            if fig is None:
                print(f"  Skipping {st} {comp}: unsupported data shape")
                continue

            # Generate filename
            base_name = Path(
                generate_unique_plot_filename(
                    system=system,
                    sim_config=sim_config,
                    domain="time",
                    component=comp,
                    figures_root=FIGURES_DIR,
                )
            )
            safe_label = str(st).replace(" ", "_")
            unique_path = _generate_unique_filename(
                base_name.parent, f"{base_name.name}_{safe_label}"
            )
            filename = unique_path

            # Save the figure
            saved = save_fig(fig, filename=filename)
            saved_files.append(str(saved))
            print(f"  Saved: {saved}")

    # Frequency domain
    print("Plotting frequency domain...")
    pad_factor = EXTEND
    nu_cohs, nu_dets, datas_nu, out_types = compute_spectra(
        list(signals.values()),
        list(signals.keys()),
        np.asarray(t_det),
        np.asarray(t_coh) if t_coh is not None else None,
        pad=pad_factor,
    )

    for idx, st in enumerate(out_types):
        combined_dict["signal_type"] = st
        sig_data_freq = datas_nu[idx]
        for comp in components:
            fig = plot_el_field(
                axis_det=nu_dets,
                data=sig_data_freq,
                axis_coh=nu_cohs,
                component=comp,
                domain="freq",
                section=SECTION,
                **combined_dict,
            )
            if fig is None:
                print(f"  Skipping {st} {comp}: unsupported data shape")
                continue

            # Generate filename
            base_name = Path(
                generate_unique_plot_filename(
                    system=system,
                    sim_config=sim_config,
                    domain="freq",
                    component=comp,
                    figures_root=FIGURES_DIR,
                )
            )
            safe_label = str(st).replace(" ", "_")
            unique_path = _generate_unique_filename(
                base_name.parent, f"{base_name.name}_{safe_label}"
            )
            filename = unique_path

            # Save the figure
            saved = save_fig(fig, filename=filename)
            saved_files.append(str(saved))
            print(f"  Saved: {saved}")

    print(f"✅ Plotted and saved {len(saved_files)} figures.")


if __name__ == "__main__":
    main()
