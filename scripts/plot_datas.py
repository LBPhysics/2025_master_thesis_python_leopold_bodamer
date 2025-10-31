"""
Plot processed spectroscopy data: plot time-domain and frequency-domain electric field for each signal type and component.

This script loads the final processed artifact from process_datas.py and generates
time-domain and frequency-domain plots for each signal type (e.g., 'rephasing', 'non-rephasing') and each
component ('real', 'img', 'abs', 'phase').

Examples:
    python plot_datas.py --abs_path '/path/to/final_averaged.npz'
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import argparse
from typing import Any

from qspectro2d import load_simulation_data
from qspectro2d.visualization.plotting import plot_el_field
from qspectro2d.spectroscopy.post_processing import compute_spectra
from plotstyle import save_fig, FIG_FORMAT

from calc_datas import PROJECT_ROOT

FIGURES_DIR = (PROJECT_ROOT / "figures").resolve()
FIGURES_DIR.mkdir(exist_ok=True)

# Section for cropping data
SECTION = (1.5, 1.7)  # or None #for no cropping

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
    print(f"Available signals: {list(signals.keys())}, with shapes {[s.shape for s in signals.values()]}")
    t_det = data["t_det"]
    t_coh = data.get("t_coh")
    print(f"Loaded t_det with shape: {t_det.shape}",
            f"and t_coh with shape: {t_coh.shape if t_coh is not None else 'None'}"
          )
    if t_coh is not None and (t_coh.ndim == 0 or t_coh.size == 0):
        t_coh = None
    system = data["system"]
    sim_config = data["simulation_config"]
    combined_dict = {**system.to_dict(), **sim_config.to_dict()}
    print(f"Loaded data from {args.abs_path}")
    print(f"Signal types: {list(signals.keys())}")
    print(f"Data dimension: {'2D' if t_coh is not None else '1D'}")

    components = ["real", "img", "abs"]
    saved_files = []

    job_metadata = data.get("job_metadata") or {}
    figures_root = _resolve_figures_dir(job_metadata)
    base_token = _figure_token(job_metadata, sim_config)

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

            stem = _figure_stem(base_token, st, "time", comp)
            filename = _unique_fig_path(figures_root, stem)

            saved = save_fig(fig, filename=filename)
            saved_files.append(str(saved))
            print(saved)
    print("Plotting frequency domain...")
    pad_factor = EXTEND
    nu_cohs, nu_dets, datas_nu, out_types = compute_spectra(
        list(signals.values()),
        list(signals.keys()),
        np.asarray(t_det),
        np.asarray(t_coh) if t_coh is not None else None,
        pad=pad_factor,
        nu_win_coh=SECTION[1] if SECTION is not None else 2.0,
        nu_win_det=SECTION[1] if SECTION is not None else 2.0,
    )

    for idx, st in enumerate(out_types):
        combined_dict["signal_type"] = st
        print(f"Plotting signal: {st}")

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

            stem = _figure_stem(base_token, st, "freq", comp)
            filename = _unique_fig_path(figures_root, stem)

            saved = save_fig(fig, filename=filename)
            saved_files.append(str(saved))
            print(saved)

    print(f"to see them go to:\n{Path(filename).parent}")


def _resolve_figures_dir(job_metadata: dict[str, Any]) -> Path:
    try:
        figures_dir = Path(job_metadata["figures_dir"]).expanduser()
    except KeyError as exc:
        raise KeyError("job_metadata.json missing required key: figures_dir") from exc

    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def _figure_token(job_metadata: dict[str, Any], sim_config: Any) -> str:
    if "job_label" in job_metadata:
        return str(job_metadata["job_label"])
    job_dir = job_metadata.get("job_dir")
    if job_dir:
        return Path(job_dir).name
    if job_metadata.get("data_base_name"):
        return str(job_metadata["data_base_name"])
    return getattr(sim_config, "sim_type", "run")


def _figure_stem(base_token: str, signal: str, domain: str, component: str) -> str:
    safe_signal = str(signal).replace(" ", "_").replace("/", "-")
    return f"{base_token}_{domain}_{component}_{safe_signal}".lower()


def _unique_fig_path(figures_dir: Path, stem: str) -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    default_ext = f".{FIG_FORMAT.lstrip('.')}"
    candidate = figures_dir / stem
    base = candidate
    counter = 1
    while candidate.with_suffix(default_ext).exists():
        candidate = base.with_name(f"{stem}_{counter:02d}")
        counter += 1
    return candidate


if __name__ == "__main__":
    main()
