"""
Plot processed spectroscopy data: plot time-domain and frequency-domain electric field for each signal type and component.

This script loads the final processed artifact from process_datas.py and generates
time-domain and frequency-domain plots for each signal type (e.g., 'rephasing', 'non-rephasing') and each
component ('real', 'img', 'abs', 'phase').

Examples:
        python "<repo>/scripts/local/plot_datas.py" --abs_path '/path/to/final_averaged.npz'
"""

from __future__ import annotations

from pathlib import Path
import argparse
from functools import partial
import time
from typing import Any

import numpy as np

from qspectro2d import load_simulation_data
from qspectro2d.visualization.plotting import plot_el_field
from qspectro2d.spectroscopy.post_processing import compute_spectra
from plotstyle import save_fig, FIG_FORMAT

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
for _parent in SCRIPTS_DIR.parents:
    if (_parent / ".git").is_dir():
        PROJECT_ROOT = _parent
        break
else:
    raise RuntimeError("Could not locate project root (missing .git directory)")

print = partial(print, flush=True)

# Section for cropping data
SECTION = (1.5, 1.7)  # or None #for no cropping

# PLOT_PAD_FACTOR factor for zero-padding
PLOT_PAD_FACTOR = 20.0


def _format_seconds(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    mins, secs = divmod(seconds, 60)
    if mins < 60:
        return f"{int(mins)}m {secs:04.1f}s"
    hours, mins = divmod(mins, 60)
    return f"{int(hours)}h {int(mins)}m {secs:04.1f}s"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot processed spectroscopy data in time and frequency domains."
    )
    parser.add_argument(
        "--abs_path", type=str, required=True, help="Path to the processed .npz file"
    )
    parser.add_argument(
        "--time_only",
        action="store_true",
        help="Only plot time-domain signals (skip frequency-domain plots).",
    )
    args = parser.parse_args()

    start_all = time.perf_counter()
    print(f"Starting plot_datas for: {args.abs_path}")

    # Load the data
    step_start = time.perf_counter()
    data = load_simulation_data(args.abs_path)
    signals = data["signals"]
    print(
        f"Available signals: {list(signals.keys())}, "
        f"shapes={[s.shape for s in signals.values()]}"
    )
    t_det = data["t_det"]
    t_coh = data.get("t_coh")
    print(
        f"Loaded t_det shape: {t_det.shape}; "
        f"t_coh shape: {t_coh.shape if t_coh is not None else 'None'}"
    )
    if t_coh is not None and (t_coh.ndim == 0 or t_coh.size == 0):
        t_coh = None
    sim_config = data["simulation_config"]
    print(f"Loaded data from {args.abs_path}")
    print(f"Signal types: {list(signals.keys())}")
    print(f"Data dimension: {'2D' if t_coh is not None else '1D'}")
    print(f"Load time: {_format_seconds(time.perf_counter() - step_start)}")

    components = ["real", "img", "abs"]
    saved_files = []

    job_metadata = data.get("job_metadata") or {}
    figures_root = _resolve_figures_dir(job_metadata)
    base_token = _figure_token(job_metadata, sim_config)

    for st, sig_data in signals.items():
        signal_start = time.perf_counter()
        print(f"Plotting time-domain signal: {st}")
        for comp in components:
            fig = plot_el_field(
                axis_det=t_det,
                data=sig_data,
                axis_coh=t_coh,
                component=comp,
                domain="time",
            )
            if fig is None:
                print(f"  Skipping {st} {comp}: unsupported data shape")
                continue

            stem = _figure_stem(base_token, st, "time", comp)
            filename = _unique_fig_path(figures_root, stem)

            saved = save_fig(fig, filename=filename)
            saved_files.append(str(saved))
            print(f"Saved: {saved}")
        print(f"Time-domain {st} done in {_format_seconds(time.perf_counter() - signal_start)}")
    if args.time_only:
        print("Skipping frequency-domain plots (--time_only set)")
        print(f"Figures folder: {Path(filename).parent}")
        print(f"Total time: {_format_seconds(time.perf_counter() - start_all)}")
        return
    print("Plotting frequency domain...")
    pad_factor = PLOT_PAD_FACTOR
    step_start = time.perf_counter()
    nu_cohs, nu_dets, datas_nu, out_types = compute_spectra(
        list(signals.values()),
        list(signals.keys()),
        np.asarray(t_det),
        np.asarray(t_coh) if t_coh is not None else None,
        pad=pad_factor,
        nu_win_coh=SECTION[1] if SECTION is not None else 2.0,
        nu_win_det=SECTION[1] if SECTION is not None else 2.0,
    )
    print(f"Frequency-domain transform time: {_format_seconds(time.perf_counter() - step_start)}")

    for idx, st in enumerate(out_types):
        signal_start = time.perf_counter()
        print(f"Plotting frequency-domain signal: {st}")

        sig_data_freq = datas_nu[idx]
        for comp in components:
            fig = plot_el_field(
                axis_det=nu_dets,
                data=sig_data_freq,
                axis_coh=nu_cohs,
                component=comp,
                domain="freq",
                section=SECTION,
            )
            if fig is None:
                print(f"  Skipping {st} {comp}: unsupported data shape")
                continue

            stem = _figure_stem(base_token, st, "freq", comp)
            filename = _unique_fig_path(figures_root, stem)

            saved = save_fig(fig, filename=filename)
            saved_files.append(str(saved))
            print(f"Saved: {saved}")
        print(
            f"Frequency-domain {st} done in {_format_seconds(time.perf_counter() - signal_start)}"
        )

    print(f"Figures folder: {Path(filename).parent}")
    print(f"Total time: {_format_seconds(time.perf_counter() - start_all)}")


def _resolve_figures_dir(job_metadata: dict[str, Any]) -> Path:
    try:
        figures_dir = Path(job_metadata["figures_dir"]).expanduser()
    except KeyError as exc:
        raise KeyError("job_metadata.json missing required key: figures_dir") from exc

    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def _figure_token(job_metadata: dict[str, Any], sim_config: Any) -> str:
    return ""


def _figure_stem(base_token: str, signal: str, domain: str, component: str) -> str:
    safe_signal = str(signal).replace(" ", "_").replace("/", "-")
    parts = [safe_signal, domain, component]
    stem = "_".join(filter(None, parts)).lower()
    return stem


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
