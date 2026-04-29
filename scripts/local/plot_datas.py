"""Plot processed spectroscopy data in time and frequency domains."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Sequence
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

from plotstyle import save_fig
from qspectro2d import load_simulation_data
from qspectro2d.spectroscopy.post_processing import compute_spectra
from qspectro2d.visualization.plotting import (
    convert_plot_axes,
    plot_el_field_signal_grid,
    validate_plot_components,
)
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
    COMPONENTS,
)

SignalData = np.ndarray | sp.spmatrix

print = partial(print, flush=True)


@dataclass
class DomainPlotPayload:
    signal_labels: list[str]
    signal_datas: list[SignalData]
    normalization_factors: list[float | None]
    plot_norms_by_signal: list[dict[str, Normalize | None]]

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


def _print_section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def _print_kv(label: str, value: object, *, indent: int = 2) -> None:
    print(f"{' ' * indent}{label:<22} {value}")


def _format_signal_shapes(signals: dict[str, np.ndarray | sp.spmatrix]) -> str:
    return ", ".join(f"{name}: {signal.shape}" for name, signal in signals.items())


def _print_path(label: str, path: Path | str, *, indent: int = 2) -> None:
    print(f"{' ' * indent}{label}:")
    print(f"{' ' * (indent + 2)}{Path(path)}")


def _print_saved_paths(saved: Path | str | Sequence[Path | str]) -> None:
    saved_paths = saved if isinstance(saved, (list, tuple)) else [saved]
    print("  Saved files:")
    for path in saved_paths:
        suffix = Path(path).suffix.lstrip(".").upper() or "FILE"
        print(f"    [{suffix}] {Path(path)}")


def _parse_apodization_window(value: str | None) -> str | None:
    if value is None:
        return APODIZATION_WINDOW

    cleaned = value.strip().lower()
    if cleaned == "none":
        return None
    if cleaned in {"hann", "hamming", "blackman"}:
        return cleaned
    raise ValueError(
        "apodization window must be one of: none, hann, hamming, blackman"
    )


def _resolve_figures_dir(job_metadata: dict[str, Any], *, artifact_path: Path | str | None = None) -> Path:
    figures_dir = Path(job_metadata["figures_dir"]).expanduser()
    if artifact_path is not None and os.name == "nt" and str(job_metadata["figures_dir"]).startswith("/"):
        artifact_path = Path(artifact_path).resolve()
        fallback_dir = artifact_path.parent.parent / "figures"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        print("\nFigures directory fallback:")
        print("  POSIX metadata path is not writable on Windows.")
        _print_path("Using local figures dir", fallback_dir, indent=2)
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
        print("\nFigures directory fallback:")
        print("  Could not create the metadata-defined figures directory.")
        _print_path("Using local figures dir", fallback_dir, indent=2)
        return fallback_dir


def _extract_config_stem(job_metadata: dict[str, Any]) -> str:
    value = str(job_metadata.get("config_stem") or "").strip()
    if not value:
        raise KeyError("Missing required job_metadata['config_stem']")
    return value

def _domain_figure_stem(
    domain: str,
    *,
    components: Sequence[str],
    config_stem: str,
    apodization_window: str | None = None,
) -> str:
    stem = f"{domain}_all_signals_{_component_tag(components)}_{config_stem}"
    if domain == "freq" and apodization_window is not None:
        stem = f"{stem}_apod_{apodization_window}"
    return stem


def _component_tag(components: Sequence[str]) -> str:
    return "_".join(components)


def _domain_suptitle(domain: str) -> str:
    return "Time Domain Signals" if domain == "time" else "Frequency Domain Spectra"


def _unique_fig_path(figures_dir: Path, stem: str, *, formats: Sequence[str]) -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    suffixes = {
        f".{str(fmt).strip().lstrip('.').lower()}"
        for fmt in formats
        if str(fmt).strip()
    }
    if not suffixes:
        suffixes = {".png"}
    candidate = figures_dir / stem
    base = candidate
    counter = 1
    while any(candidate.with_suffix(suffix).exists() for suffix in suffixes):
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
        vmax = _normalised_peak(_max_abs_complex_data(data), normalization_factor)
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

    print(f"  [{domain}] {signal_type}")
    _print_kv("raw peak |S|", f"{raw_abs_peak:.6e}", indent=4)

    if normalization_factor is None:
        _print_kv("|Re|max", f"{raw_real_peak:.3e}", indent=4)
        _print_kv("|Im|max", f"{raw_imag_peak:.3e}", indent=4)
        _print_kv("signed display", f"[-{signed_peak:.3e}, {signed_peak:.3e}]", indent=4)
        print("")
        return

    norm_abs_peak = _normalised_peak(raw_abs_peak, normalization_factor)
    norm_real_peak = _normalised_peak(raw_real_peak, normalization_factor)
    norm_imag_peak = _normalised_peak(raw_imag_peak, normalization_factor)
    norm_signed_peak = max(norm_real_peak, norm_imag_peak)

    _print_kv("normalisation", f"{float(normalization_factor):.6e}", indent=4)
    _print_kv("normalised |S|max", f"{norm_abs_peak:.3f}", indent=4)
    _print_kv("normalised |Re|max", f"{norm_real_peak:.3f}", indent=4)
    _print_kv("normalised |Im|max", f"{norm_imag_peak:.3f}", indent=4)
    _print_kv("signed display", f"[-{norm_signed_peak:.3f}, {norm_signed_peak:.3f}]", indent=4)
    print("")


def _normalisation_factor_for_signal(
    data: SignalData,
    *,
    normalise: bool,
    norm_scope: str,
    global_factor: float | None,
) -> float | None:
    if not normalise:
        return None
    if norm_scope == "all_signals":
        return global_factor
    return _group_normalisation_factor([data])


def _build_domain_plot_payload(
    *,
    domain: str,
    signal_items: Sequence[tuple[str, SignalData]],
    components: Sequence[str],
    normalise: bool,
    norm_scope: str,
) -> DomainPlotPayload:
    signal_items = list(signal_items)
    if normalise and norm_scope == "all_signals":
        global_factor = _group_normalisation_factor([data for _, data in signal_items])
    else:
        global_factor = None

    payload = DomainPlotPayload(
        signal_labels=[],
        signal_datas=[],
        normalization_factors=[],
        plot_norms_by_signal=[],
    )

    for signal_type, signal_data in signal_items:
        normalization_factor = _normalisation_factor_for_signal(
            signal_data,
            normalise=normalise,
            norm_scope=norm_scope,
            global_factor=global_factor,
        )
        _print_signal_scale_summary(
            domain=domain,
            signal_type=signal_type,
            data=signal_data,
            normalization_factor=normalization_factor,
        )
        payload.signal_labels.append(signal_type)
        payload.signal_datas.append(signal_data)
        payload.normalization_factors.append(normalization_factor)
        payload.plot_norms_by_signal.append(
            {
                component: _plot_norm_for_component(
                    signal_data,
                    component,
                    normalization_factor,
                )
                for component in components
            }
        )

    return payload


def _render_domain_figure(
    *,
    domain: str,
    axis_det: np.ndarray,
    signal_items: Sequence[tuple[str, SignalData]],
    components: Sequence[str],
    figures_root: Path,
    config_stem: str,
    normalise: bool,
    norm_scope: str,
    axis_coh: np.ndarray | None = None,
    cutoff_percent: float = 0.0,
    contour_lines: bool = False,
    apodization_window: str | None = None,
) -> None:
    payload = _build_domain_plot_payload(
        domain=domain,
        signal_items=signal_items,
        components=components,
        normalise=normalise,
        norm_scope=norm_scope,
    )

    print(f"  Building combined {domain}-domain figure...\n")
    fig = plot_el_field_signal_grid(
        axis_det=axis_det,
        signal_datas=payload.signal_datas,
        signal_labels=payload.signal_labels,
        axis_coh=axis_coh,
        components=components,
        domain=domain,
        cutoff_percent=cutoff_percent,
        contour_lines=contour_lines,
        normalization_factors=payload.normalization_factors,
        plot_norms_by_signal=payload.plot_norms_by_signal,
        normalization_scope=norm_scope,
        suptitle=_domain_suptitle(domain),
    )
    saved = save_fig(
        fig,
        filename=_unique_fig_path(
            figures_root,
            _domain_figure_stem(
                domain,
                components=components,
                config_stem=config_stem,
                apodization_window=apodization_window,
            ),
            formats=FIG_FORMATS,
        ),
        transparent=TRANSPARENTCY,
        formats=FIG_FORMATS,
    )
    _print_saved_paths(saved)


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
    parser.add_argument(
        "--freq_only",
        action="store_true",
        help="Only plot frequency-domain signals (skip time-domain plots).",
    )
    parser.add_argument(
        "--apodization_window",
        type=str,
        default=None,
        help="Override APODIZATION_WINDOW with one of: none, hann, hamming, blackman.",
    )
    args = parser.parse_args()
    apodization_window = _parse_apodization_window(args.apodization_window)
    if args.time_only and args.freq_only:
        raise ValueError("--time_only and --freq_only cannot be used together")

    start_all = time.perf_counter()
    _print_section("plot_datas")
    _print_path("Input artifact", args.abs_path)

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

    _print_section("Loaded Data")
    _print_kv("signals", list(signals.keys()))
    _print_kv("signal shapes", _format_signal_shapes(signals))
    _print_kv("t_det shape", t_det.shape)
    _print_kv("t_coh shape", t_coh.shape if t_coh is not None else "None")
    _print_kv("carrier frequency", f"{carrier_freq_cm:.2f} cm^-1")
    _print_kv("load time", _format_seconds(time.perf_counter() - step_start))

    components = validate_plot_components(COMPONENTS)
    figures_root = _resolve_figures_dir(data["job_metadata"], artifact_path=args.abs_path)
    config_stem = _extract_config_stem(data["job_metadata"])
    _print_section("Plot Settings")
    _print_kv("config stem", config_stem)
    _print_kv("components", list(components))
    _print_kv("pad factor", PAD_FACTOR)
    _print_kv("apodization", apodization_window)
    _print_kv("time transparent", TRANSPARENTCY)
    _print_kv("freq transparent", TRANSPARENTCY)
    _print_path("Figures folder", figures_root)

    if args.freq_only:
        _print_section("Time Domain")
        print("  Time-domain plotting skipped (--freq_only set).")
    else:
        _print_section("Time Domain")
        _render_domain_figure(
            domain="time",
            axis_det=t_det,
            signal_items=list(signals.items()),
            components=components,
            figures_root=figures_root,
            config_stem=config_stem,
            normalise=NORMALISE_TIME_DOMAIN,
            norm_scope=TIME_NORM_SCOPE,
            axis_coh=t_coh,
        )

    if args.time_only:
        _print_section("Done")
        print("  Frequency-domain plotting skipped (--time_only set).")
        _print_path("Figures folder", figures_root)
        _print_kv("total time", _format_seconds(time.perf_counter() - start_all))
        return

    _print_section("Frequency Domain")
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
        apodization=apodization_window,
    )

    if rwa_sl:
        nu_cohs_plot, nu_dets_plot = convert_plot_axes(
            nu_cohs,
            nu_dets,
            carrier_freq_cm=carrier_freq_cm,
        )
    else:
        nu_cohs_plot, nu_dets_plot = nu_cohs, nu_dets

    _print_kv("transform time", _format_seconds(time.perf_counter() - step_start))
    print("")

    _render_domain_figure(
        domain="freq",
        axis_det=nu_dets_plot,
        signal_items=list(zip(out_types, datas_nu)),
        components=components,
        figures_root=figures_root,
        config_stem=config_stem,
        normalise=NORMALISE_FREQUENCY_DOMAIN,
        norm_scope=FREQ_NORM_SCOPE,
        axis_coh=nu_cohs_plot,
        cutoff_percent=CUTOFF_PERCENT,
        contour_lines=CONTOUR_LINES,
        apodization_window=apodization_window,
    )

    _print_section("Done")
    _print_path("Figures folder", figures_root)
    _print_kv("total time", _format_seconds(time.perf_counter() - start_all))


if __name__ == "__main__":
    main()
