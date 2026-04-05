"""Plot processed spectroscopy data in time and frequency domains."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any
import argparse
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from matplotlib.colors import Normalize, TwoSlopeNorm

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from plotstyle import FIG_FORMAT, save_fig
from qspectro2d import load_simulation_data
from qspectro2d.spectroscopy.post_processing import compute_spectra
from qspectro2d.visualization.plotting import convert_plot_axes, plot_el_field
from common.plot_settings import (
    CUTOFF_PERCENT,
    CONTOUR_LINES,
    PAD_FACTOR,
    APODIZATION_WINDOW,
    SECTION,
    TRANSPARENTCY,
    FIG_FORMATS,
    NORMALISE_TIME_DOMAIN,
    NORMALISE_FREQUENCY_DOMAIN,
    TIME_NORM_SCOPE,
    FREQ_NORM_SCOPE,
)

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


def _resolve_figures_dir(job_metadata: dict[str, Any], *, artifact_path: Path | str | None = None) -> Path:
    figures_dir = Path(job_metadata["figures_dir"]).expanduser()
    if artifact_path is not None and os.name == "nt" and str(job_metadata["figures_dir"]).startswith("/"):
        artifact_path = Path(artifact_path).resolve()
        fallback_dir = artifact_path.parent.parent / "figures"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        print(f"Using local figures dir for POSIX metadata path: {fallback_dir}")
        return fallback_dir

    try:
        figures_dir.mkdir(parents=True, exist_ok=True)
        return figures_dir
    except OSError:
        if artifact_path is None:
            raise

        artifact_path = Path(artifact_path).resolve()
        fallback_dir = artifact_path.parent.parent / "figures"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        print(f"Falling back to local figures dir: {fallback_dir}")
        return fallback_dir


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


def _max_abs_complex_data(data: np.ndarray | sp.spmatrix) -> float:
    """Return max(abs(data)) without densifying sparse matrices."""
    if sp.issparse(data):
        if data.nnz == 0:
            return 0.0
        values = np.asarray(data.data)
    else:
        values = np.asarray(data)

    values = np.abs(values)
    finite = np.isfinite(values)
    if not np.any(finite):
        return 0.0
    return float(np.max(values[finite]))


def _group_normalisation_factor(datas: list[np.ndarray | sp.spmatrix]) -> float | None:
    maxima = [_max_abs_complex_data(data) for data in datas]
    factor = max(maxima, default=0.0)
    if not np.isfinite(factor) or factor <= 0.0:
        return None
    return factor


def _max_abs_component_data(data: np.ndarray | sp.spmatrix, component: str) -> float:
    """Return max(abs(component(data))) without densifying sparse matrices."""
    if sp.issparse(data):
        if data.nnz == 0:
            return 0.0
        values = np.asarray(data.data)
    else:
        values = np.asarray(data)

    if component == "real":
        values = np.real(values)
    elif component == "imag":
        values = np.imag(values)
    elif component == "abs":
        values = np.abs(values)
    else:
        raise ValueError(f"Unsupported component: {component!r}")

    values = np.abs(values)
    finite = np.isfinite(values)
    if not np.any(finite):
        return 0.0
    return float(np.max(values[finite]))


def _normalised_peak(value: float, normalization_factor: float | None) -> float:
    if normalization_factor is None:
        return value

    factor = float(normalization_factor)
    if not np.isfinite(factor) or factor <= 0.0:
        return value
    return value / factor


def _plot_norm_for_component(
    data: np.ndarray | sp.spmatrix,
    component: str,
    normalization_factor: float | None,
) -> Normalize | None:
    if component in {"real", "imag"}:
        shared_signed_peak = max(
            _max_abs_component_data(data, "real"),
            _max_abs_component_data(data, "imag"),
        )
        vmax = _normalised_peak(shared_signed_peak, normalization_factor)
        if np.isfinite(vmax) and vmax > 0.0:
            return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        return None

    if component == "abs":
        if normalization_factor is not None:
            return Normalize(vmin=0.0, vmax=1.0)

        vmax = _max_abs_complex_data(data)
        if np.isfinite(vmax) and vmax > 0.0:
            return Normalize(vmin=0.0, vmax=vmax)
        return None

    return None


def _print_signal_scale_summary(
    *,
    domain: str,
    signal_type: str,
    data: np.ndarray | sp.spmatrix,
    normalization_factor: float | None,
) -> None:
    raw_abs_peak = _max_abs_complex_data(data)
    raw_real_peak = _max_abs_component_data(data, "real")
    raw_imag_peak = _max_abs_component_data(data, "imag")
    signed_peak = max(raw_real_peak, raw_imag_peak)

    print(f"  [{domain}] raw peak |S| = {raw_abs_peak:.6e}")

    if normalization_factor is None:
        print(
            "  "
            f"[{domain}] signed display range for {signal_type}: "
            f"[-{signed_peak:.3e}, {signed_peak:.3e}] "
            f"with |Re|max={raw_real_peak:.3e}, |Im|max={raw_imag_peak:.3e}"
        )
        return

    norm_abs_peak = _normalised_peak(raw_abs_peak, normalization_factor)
    norm_real_peak = _normalised_peak(raw_real_peak, normalization_factor)
    norm_imag_peak = _normalised_peak(raw_imag_peak, normalization_factor)
    norm_signed_peak = max(norm_real_peak, norm_imag_peak)

    print(
        "  "
        f"[{domain}] normalisation factor = {float(normalization_factor):.6e}; "
        f"normalised |S|max={norm_abs_peak:.3f}, "
        f"|Re|max={norm_real_peak:.3f}, "
        f"|Im|max={norm_imag_peak:.3f}, "
        f"signed display range=[-{norm_signed_peak:.3f}, {norm_signed_peak:.3f}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot processed spectroscopy data in time and frequency domains."
    )
    parser.add_argument(
        "--abs_path",
        type=str,
        required=True,
        help="Path to the processed .npz file",
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
    figures_root = _resolve_figures_dir(data["job_metadata"], artifact_path=args.abs_path)
    config_stem = _extract_config_stem(data["job_metadata"])
    print(f"Config stem: {config_stem}")

    # ---------------------------------------------------------------------
    # Time-domain normalisation factor
    # ---------------------------------------------------------------------
    if NORMALISE_TIME_DOMAIN and TIME_NORM_SCOPE == "all_signals":
        time_norm_factor_global = _group_normalisation_factor(list(signals.values()))
    else:
        time_norm_factor_global = None

    for signal_type, sig_data in signals.items():
        signal_start = time.perf_counter()
        print(f"Plotting time-domain signal: {signal_type}")

        if NORMALISE_TIME_DOMAIN:
            if TIME_NORM_SCOPE == "all_signals":
                time_norm_factor = time_norm_factor_global
            else:
                time_norm_factor = _group_normalisation_factor([sig_data])
        else:
            time_norm_factor = None

        _print_signal_scale_summary(
            domain="time",
            signal_type=signal_type,
            data=sig_data,
            normalization_factor=time_norm_factor,
        )

        for component in components:
            plot_norm = _plot_norm_for_component(
                sig_data,
                component,
                time_norm_factor,
            )

            fig = plot_el_field(
                axis_det=t_det,
                data=sig_data,
                axis_coh=t_coh,
                component=component,
                domain="time",
                normalization_factor=time_norm_factor,
                plot_norm=plot_norm,
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
        apodization=APODIZATION_WINDOW,
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

    # ---------------------------------------------------------------------
    # Frequency-domain normalisation factor
    # ---------------------------------------------------------------------
    if NORMALISE_FREQUENCY_DOMAIN and FREQ_NORM_SCOPE == "all_signals":
        freq_norm_factor_global = _group_normalisation_factor(list(datas_nu))
    else:
        freq_norm_factor_global = None

    for idx, signal_type in enumerate(out_types):
        signal_start = time.perf_counter()
        print(f"Plotting frequency-domain signal: {signal_type}")

        if NORMALISE_FREQUENCY_DOMAIN:
            if FREQ_NORM_SCOPE == "all_signals":
                freq_norm_factor = freq_norm_factor_global
            else:
                freq_norm_factor = _group_normalisation_factor([datas_nu[idx]])
        else:
            freq_norm_factor = None

        _print_signal_scale_summary(
            domain="freq",
            signal_type=signal_type,
            data=datas_nu[idx],
            normalization_factor=freq_norm_factor,
        )

        for component in components:
            plot_norm = _plot_norm_for_component(
                datas_nu[idx],
                component,
                freq_norm_factor,
            )

            fig = plot_el_field(
                axis_det=nu_dets_plot,
                data=datas_nu[idx],
                axis_coh=nu_cohs_plot,
                component=component,
                domain="freq",
                cutoff_percent=CUTOFF_PERCENT,
                contour_lines=CONTOUR_LINES,
                normalization_factor=freq_norm_factor,
                plot_norm=plot_norm,
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
