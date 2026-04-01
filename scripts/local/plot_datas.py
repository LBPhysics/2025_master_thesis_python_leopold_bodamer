"""Plot processed spectroscopy data in time and frequency domains."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any
import numpy as np
import argparse
import time
import sys

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


from plotstyle import FIG_FORMAT, save_fig
from qspectro2d import load_simulation_data
from qspectro2d.spectroscopy.post_processing import compute_spectra
from qspectro2d.visualization.plotting import convert_plot_axes, plot_el_field
from common.plot_settings import CUTOFF_PERCENT, CONTOUR_LINES, PAD_FACTOR, SECTION, TRANSPARENTCY, FIG_FORMATS

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
for _parent in SCRIPTS_DIR.parents:
    if (_parent / ".git").is_dir():
        PROJECT_ROOT = _parent
        break
else:
    raise RuntimeError("Could not locate project root (missing .git directory)")

SIGNAL_CODE = {
    "rephasing": "R",
    "nonrephasing": "NR",
    "absorptive": "A",
}
print = partial(print, flush=True)


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


def _resolve_figures_dir(job_metadata: dict[str, Any]) -> Path:
    figures_dir = Path(job_metadata["figures_dir"]).expanduser()
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def _extract_config_stem(job_metadata: dict[str, Any]) -> str:
    value = str(job_metadata.get("config_stem") or "").strip()
    if not value:
        raise KeyError("Missing required job_metadata['config_stem']")
    return value


def _figure_stem(domain: str, signal: str, component: str, *, config_stem: str) -> str:
    try:
        signal_code = SIGNAL_CODE[str(signal)]
    except KeyError as exc:
        raise KeyError(f"Unsupported signal type for filename mapping: {signal!r}") from exc

    return f"{domain}_{signal_code}_{component}_{config_stem}"


def _unique_fig_path(figures_dir: Path, stem: str) -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    default_ext = f".{FIG_FORMAT.lstrip('.')}"
    candidate = figures_dir / stem
    base = candidate
    counter = 1
    # Check for both SVG and PNG files to avoid conflicts (since we save both formats)
    while candidate.with_suffix(default_ext).exists() or candidate.with_suffix(".png").exists():
        candidate = base.with_name(f"{stem}_{counter:02d}")
        counter += 1
    return candidate


def _section_to_stored_freq_frame(
    section: tuple[tuple[float, float], tuple[float, float]] | None,
    *,
    rwa_sl: bool,
    carrier_freq_cm: float,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Convert a lab-frame plotting ROI to the stored FFT frame used by compute_spectra."""
    if section is None or not rwa_sl:
        return section

    shift = float(carrier_freq_cm) * 1e-4
    return (
        (section[0][0] - shift, section[0][1] - shift),
        (section[1][0] - shift, section[1][1] - shift),
    )


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

    step_start = time.perf_counter()
    data = load_simulation_data(args.abs_path)
    required_keys = ["signals", "t_det", "simulation_config", "job_metadata"]
    missing = [key for key in required_keys if key not in data]
    if missing:
        raise KeyError(f"Missing required data keys in artifact: {missing}")

    signals = data["signals"]
    t_det = np.asarray(data["t_det"], dtype=float)
    t_coh = np.asarray(data["t_coh"], dtype=float)
    if t_coh.ndim == 0 or t_coh.size == 0:
        t_coh = None

    simulation_config = data["simulation_config"]
    rwa_sl = simulation_config.rwa_sl
    carrier_freq_cm = simulation_config.carrier_freq_cm

    print(
        f"Available signals: {list(signals.keys())}, shapes={[s.shape for s in signals.values()]}"
    )
    print(
        f"Loaded t_det shape: {t_det.shape}; t_coh shape: {t_coh.shape if t_coh is not None else 'None'}"
    )
    print(f"Carrier frequency: {carrier_freq_cm:.2f} cm^-1")
    print(f"Load time: {_format_seconds(time.perf_counter() - step_start)}")

    components = ["real", "imag", "abs"]
    figures_root = _resolve_figures_dir(data["job_metadata"])
    config_stem = _extract_config_stem(data["job_metadata"])
    print(f"Config stem: {config_stem}")

    for signal_type, sig_data in signals.items():
        signal_start = time.perf_counter()
        print(f"Plotting time-domain signal: {signal_type}")
        for component in components:
            fig = plot_el_field(
                axis_det=t_det, data=sig_data, axis_coh=t_coh, component=component, domain="time"
            )
            saved = save_fig(
                fig,
                filename=_unique_fig_path(
                    figures_root,
                    _figure_stem("time", signal_type, component, config_stem=config_stem),
                ),
                formats=FIG_FORMATS,
            )
            print(f"Saved: {saved}")
        print(
            f"Time-domain {signal_type} done in {_format_seconds(time.perf_counter() - signal_start)}"
        )

    if args.time_only:
        print("Skipping frequency-domain plots (--time_only set)")
        print(f"Figures folder: {figures_root}")
        print(f"Total time: {_format_seconds(time.perf_counter() - start_all)}")
        return

    print("Plotting frequency domain...")
    step_start = time.perf_counter()

    stored_section = _section_to_stored_freq_frame(
        SECTION,
        rwa_sl=rwa_sl,
        carrier_freq_cm=carrier_freq_cm,
    )

    nu_cohs, nu_dets, datas_nu, out_types = compute_spectra(
        list(signals.values()),
        list(signals.keys()),
        t_det,
        t_coh,
        pad=PAD_FACTOR,
        section=stored_section,
    )

    if rwa_sl:
        nu_cohs_plot, nu_dets_plot = convert_plot_axes(
            nu_cohs,
            nu_dets,
            carrier_freq_cm=carrier_freq_cm,
        )
    else:
        nu_cohs_plot, nu_dets_plot = nu_cohs, nu_dets

    print(f"Frequency-domain transform time: {_format_seconds(time.perf_counter() - step_start)}")

    for idx, signal_type in enumerate(out_types):
        signal_start = time.perf_counter()
        print(f"Plotting frequency-domain signal: {signal_type}")
        for component in components:
            fig = plot_el_field(
                axis_det=nu_dets_plot,
                data=datas_nu[idx],
                axis_coh=nu_cohs_plot,
                component=component,
                domain="freq",
                cutoff_percent=CUTOFF_PERCENT,
                contour_lines=CONTOUR_LINES,
            )
            saved = save_fig(
                fig,
                filename=_unique_fig_path(
                    figures_root,
                    _figure_stem("freq", signal_type, component, config_stem=config_stem),
                ),
                transparent=TRANSPARENTCY,
                formats=FIG_FORMATS,
            )
            print(f"Saved: {saved}")
        print(
            f"Frequency-domain {signal_type} done in {_format_seconds(time.perf_counter() - signal_start)}"
        )

    print(f"Figures folder: {figures_root}")
    print(f"Total time: {_format_seconds(time.perf_counter() - start_all)}")


if __name__ == "__main__":
    main()
