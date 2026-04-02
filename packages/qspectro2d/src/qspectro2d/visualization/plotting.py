from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from matplotlib.colors import TwoSlopeNorm

from plotstyle import COLORS, LINE_STYLES, init_style, simplify_figure_text

from ..core.laser_system import (
    LaserPulseSequence,
    e_pulses,
    epsilon_pulses,
    pulse_envelopes,
    single_pulse_envelope,
)
from ..core.laser_system.laser import DEFAULT_ACTIVE_WINDOW_NFWHM

init_style(
    rc_overrides={
        "lines.linewidth": 4.0,
    }
)


def _get_fig_ax(ax: plt.Axes | None = None, *, figsize: tuple[float, float] | None = None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax


def _add_legend(ax: plt.Axes, show_legend: bool) -> None:
    if show_legend:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))


def _pulse_color(index: int) -> str:
    return COLORS[index % len(COLORS)]


def _pulse_style(index: int) -> str:
    return LINE_STYLES[index % len(LINE_STYLES)]


def _plot_pulse_peak_lines(
    ax: plt.Axes,
    pulse_seq: LaserPulseSequence,
    *,
    show_window: bool = False,
) -> None:
    for idx, pulse in enumerate(pulse_seq.pulses):
        color = _pulse_color(idx + 1)
        t_peak = pulse.pulse_peak_time

        ax.axvline(
            t_peak,
            linestyle=LINE_STYLES[3],
            color=color,
            alpha=0.8,
            label=rf"$t_{{peak,{idx + 1}}}$",
        )

        if show_window:
            delta = DEFAULT_ACTIVE_WINDOW_NFWHM * pulse.pulse_fwhm_fs
            ax.axvline(
                t_peak - delta,
                linestyle=LINE_STYLES[2],
                color=color,
                alpha=0.35,
            )
            ax.axvline(
                t_peak + delta,
                linestyle=LINE_STYLES[2],
                color=color,
                alpha=0.35,
            )


def _complex_field_over_time(
    times: np.ndarray,
    pulse_seq: LaserPulseSequence,
    field_func,
) -> np.ndarray:
    return np.asarray([field_func(float(t), pulse_seq) for t in times], dtype=complex)


def _plot_complex_series(
    times: np.ndarray,
    values: np.ndarray,
    *,
    ax: plt.Axes,
    real_label: str,
    imag_label: str,
    abs_label: str | None = None,
    real_color: str = COLORS[0],
    imag_color: str = COLORS[1],
    abs_color: str = COLORS[2],
) -> None:
    ax.plot(
        times,
        np.real(values),
        label=real_label,
        linestyle=LINE_STYLES[0],
        color=real_color,
    )
    ax.plot(
        times,
        np.imag(values),
        label=imag_label,
        linestyle=LINE_STYLES[1],
        color=imag_color,
    )
    if abs_label is not None:
        ax.plot(
            times,
            np.abs(values),
            label=abs_label,
            linestyle=LINE_STYLES[2],
            color=abs_color,
        )


def plot_pulse_envelopes(
    times: np.ndarray,
    pulse_seq: LaserPulseSequence,
    ax: plt.Axes | None = None,
    show_legend: bool = True,
):
    """Plot combined and individual pulse envelopes."""
    fig, ax = _get_fig_ax(ax)

    ax.plot(
        times,
        pulse_envelopes(times, pulse_seq),
        label=r"$\text{Combined Envelope}$",
        linestyle=LINE_STYLES[0],
        color=COLORS[0],
        alpha=0.85,
    )

    for idx, pulse in enumerate(pulse_seq.pulses):
        color = _pulse_color(idx + 1)
        ax.plot(
            times,
            single_pulse_envelope(times, pulse),
            label=rf"$\text{{Pulse {idx + 1}}}$",
            linestyle=_pulse_style(idx + 1),
            color=color,
            alpha=0.65,
        )

    _plot_pulse_peak_lines(ax, pulse_seq, show_window=True)

    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"Envelope Amplitude")
    ax.set_title(r"Pulse Envelopes")
    _add_legend(ax, show_legend)
    simplify_figure_text(fig)
    return fig, ax


def plot_e_pulses(
    times: np.ndarray,
    pulse_seq: LaserPulseSequence,
    ax: plt.Axes | None = None,
    show_legend: bool = True,
):
    """Plot the complex RWA electric field."""
    fig, ax = _get_fig_ax(ax, figsize=(10, 6))
    e_field = _complex_field_over_time(times, pulse_seq, e_pulses)

    _plot_complex_series(
        times,
        e_field,
        ax=ax,
        real_label=r"$\mathrm{Re}[E(t)]$",
        imag_label=r"$\mathrm{Im}[E(t)]$",
    )
    _plot_pulse_peak_lines(ax, pulse_seq, show_window=False)

    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"Electric Field (RWA)")
    ax.set_title(r"RWA Electric Field Components")
    _add_legend(ax, show_legend)
    simplify_figure_text(fig)
    return fig, ax


def plot_epsilon_pulses(
    times: np.ndarray,
    pulse_seq: LaserPulseSequence,
    ax: plt.Axes | None = None,
    show_legend: bool = True,
):
    """Plot the full electric field including carrier oscillation."""
    fig, ax = _get_fig_ax(ax, figsize=(10, 6))
    epsilon_field = _complex_field_over_time(times, pulse_seq, epsilon_pulses)

    _plot_complex_series(
        times,
        epsilon_field,
        ax=ax,
        real_label=r"$\mathrm{Re}[\varepsilon(t)]$",
        imag_label=r"$\mathrm{Im}[\varepsilon(t)]$",
        abs_label=r"$|\varepsilon(t)|$",
        real_color=COLORS[3 % len(COLORS)],
        imag_color=COLORS[4 % len(COLORS)],
        abs_color=COLORS[5 % len(COLORS)],
    )
    _plot_pulse_peak_lines(ax, pulse_seq, show_window=False)

    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"Electric Field (Full)")
    ax.set_title(r"Full Electric Field with Carrier")
    _add_legend(ax, show_legend)
    simplify_figure_text(fig)
    return fig, ax


def plot_all_pulse_components(
    times: np.ndarray,
    pulse_seq: LaserPulseSequence,
    ax=None,
):
    """Plot envelopes, RWA field, and full field in one figure."""
    if ax is None:
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    else:
        axes = ax
        fig = axes[0].figure

    plot_pulse_envelopes(times, pulse_seq, ax=axes[0])
    plot_e_pulses(times, pulse_seq, ax=axes[1])
    plot_epsilon_pulses(times, pulse_seq, ax=axes[2])

    fig.suptitle(
        f"Pulse Analysis - {len(pulse_seq.pulses)} Pulse(s)",
        fontsize=16,
        y=0.98,
    )
    plt.tight_layout()
    simplify_figure_text(fig)
    return fig


def plot_example_evo(
    times_plot: np.ndarray,
    datas: list,
    pulse_seq: LaserPulseSequence,
    observable_strs: list[str],
    rwa_sl: bool = False,
    ax=None,
    **kwargs: dict,
):
    """Plot field and expectation values over time."""
    field_func = e_pulses if rwa_sl else epsilon_pulses

    e0 = pulse_seq.E0
    e_total = _complex_field_over_time(times_plot, pulse_seq, field_func) / e0

    if ax is None:
        fig, axes = plt.subplots(
            len(datas) + 1,
            1,
            figsize=(14, 2 + 2 * len(datas)),
            sharex=True,
        )
    else:
        axes = ax
        fig = axes[0].figure

    _plot_complex_series(
        times_plot,
        e_total,
        ax=axes[0],
        real_label=r"$\mathrm{Re}[E(t)]$",
        imag_label=r"$\mathrm{Im}[E(t)]$",
    )
    axes[0].set_ylabel(r"$E(t) / E_0$")
    _add_legend(axes[0], True)

    for idx, data in enumerate(datas):
        axis = axes[idx + 1]
        is_polarisation = idx >= len(observable_strs)
        observable_str = r"\text{Pol}" if is_polarisation else observable_strs[idx]

        axis.plot(
            times_plot,
            np.real(data),
            color=COLORS[0],
            linestyle=LINE_STYLES[0],
            label=r"$\mathrm{Re}\langle " + observable_str + r" \rangle$",
        )

        if is_polarisation:
            axis.plot(
                times_plot,
                np.imag(data),
                color=COLORS[1],
                linestyle=LINE_STYLES[1],
                label=r"$\mathrm{Im}\langle " + observable_str + r" \rangle$",
            )

        axis.set_ylabel(r"$\langle " + observable_str + r" \rangle$")
        _add_legend(axis, True)

    add_text_box(ax=axes[0], kwargs=kwargs)
    axes[-1].set_xlabel(r"$t\,/\,\mathrm{fs}$")

    plt.tight_layout()
    simplify_figure_text(fig)
    return fig


def plot_el_field(
    axis_det: np.ndarray,
    data: np.ndarray,
    axis_coh: np.ndarray | None = None,
    component: Literal["real", "imag", "abs", "phase"] = "real",
    domain: Literal["time", "freq"] = "time",
    ax: plt.Axes | None = None,
    cutoff_percent: float = 0.0,
    contour_lines: bool = False,
    **kwargs: dict,
) -> plt.Figure:
    """Plot emitted field."""
    if component not in {"real", "imag", "abs", "phase"}:
        raise ValueError("component must be one of: 'real', 'imag', 'abs', 'phase'")
    if domain not in {"time", "freq"}:
        raise ValueError("domain must be 'time' or 'freq'")

    axis_det = np.asarray(axis_det, dtype=float)
    axis_coh = None if axis_coh is None else np.asarray(axis_coh, dtype=float)

    if domain == "freq" and isinstance(data, sp.spmatrix):
        axis_coh, axis_det, data = _materialize_sparse_roi_for_plot(
            axis_det,
            data,
            axis_coh,
        )
        if np.size(data) == 0:
            raise ValueError(
                "Empty frequency ROI after sparse cropping. "
                "Check whether SECTION is specified in lab-frame units and "
                "was converted correctly before calling compute_spectra."
            )
    else:
        if isinstance(data, sp.spmatrix):
            data = data.toarray()
        data = np.asarray(data)

    if axis_coh is None and data.ndim == 2 and data.shape[1] == 1:
        data = data.squeeze(axis=1)

    plot_data, base_title = _component_data(np.asarray(data), component)

    if cutoff_percent > 0 and plot_data.size > 0:
        threshold = cutoff_percent / 100.0 * np.max(np.abs(plot_data))
        plot_data = np.ma.masked_where(np.abs(plot_data) < threshold, plot_data)

    fig, ax = _get_fig_ax(ax)

    if plot_data.ndim == 1:
        _plot_el_field_1d(
            ax=ax,
            axis_det=axis_det,
            plot_data=plot_data,
            component=component,
            domain=domain,
            base_title=base_title,
        )
    elif plot_data.ndim == 2:
        if axis_coh is None:
            raise ValueError("axis_coh must be provided for 2D data")
        _plot_el_field_2d(
            ax=ax,
            axis_coh=axis_coh,
            axis_det=axis_det,
            plot_data=plot_data,
            component=component,
            domain=domain,
            base_title=base_title,
            contour_lines=contour_lines,
        )
    else:
        raise ValueError("data must be 1D or 2D")

    add_text_box(ax, kwargs=kwargs)
    fig.tight_layout()
    simplify_figure_text(fig)
    return fig


def _plot_el_field_1d(
    *,
    ax: plt.Axes,
    axis_det: np.ndarray,
    plot_data: np.ndarray,
    component: str,
    domain: str,
    base_title: str,
) -> None:
    color, linestyle = _style_for_component(component, 1, domain=domain)

    values = np.asarray(plot_data, dtype=float)
    # if values.size > 0 and not np.all(np.isnan(values)):
    #    max_abs = np.max(np.abs(values))
    #    if max_abs > 0:
    #        values = values / max_abs

    x_label, y_label, title_suffix = _domain_labels(domain, 1)

    ax.plot(
        axis_det,
        values,
        color=color,
        linestyle=linestyle,
        label=base_title.strip(),
    )
    ax.set_title(base_title + " " + title_suffix)
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$" + y_label.strip("$") + r" / \max(|" + y_label.strip("$") + r"|)$")
    ax.legend()


def _plot_el_field_2d(
    *,
    ax: plt.Axes,
    axis_coh: np.ndarray,
    axis_det: np.ndarray,
    plot_data: np.ndarray,
    component: str,
    domain: str,
    base_title: str,
    contour_lines: bool = False,
) -> None:
    colormap, norm = _style_for_component(component, 2, data=plot_data, domain=domain)

    def _axis_extent(axis: np.ndarray) -> tuple[float, float]:
        axis = np.asarray(axis, dtype=float)
        if axis.size == 0:
            raise ValueError("Axis must contain at least one point for 2D plotting")
        if axis.size == 1:
            center = float(axis[0])
            pad = max(abs(center) * 1e-6, 1e-9)
            return center - pad, center + pad

        left = float(axis[0] - 0.5 * (axis[1] - axis[0]))
        right = float(axis[-1] + 0.5 * (axis[-1] - axis[-2]))
        if left == right:
            pad = max(abs(left) * 1e-6, 1e-9)
            return left - pad, right + pad
        return left, right

    coh_min, coh_max = _axis_extent(axis_coh)
    det_min, det_max = _axis_extent(axis_det)

    im = ax.imshow(
        plot_data.T,
        extent=[coh_min, coh_max, det_min, det_max],
        origin="lower",
        aspect="auto",
        cmap=colormap,
        norm=norm,
        interpolation="bilinear",
    )

    if contour_lines:
        add_custom_contour_lines(
            x=axis_coh,
            y=axis_det,
            data=plot_data.T,
            component=component,
            ax=ax,
        )

    x_label, y_label, cbar_label, title_suffix = _domain_labels(domain, 2)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(base_title + " " + title_suffix)
    plt.colorbar(im, ax=ax, label=cbar_label)


def crop_nd_data_along_axis(
    coord_array: np.ndarray,
    nd_data: np.ndarray,
    section: tuple[float, float],
    axis: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop N-dimensional dense data along one axis."""
    coord_array = np.asarray(coord_array, dtype=float)
    nd_data = np.asarray(nd_data)

    if coord_array.ndim != 1:
        raise ValueError("Coordinate array must be 1-dimensional")
    if nd_data.shape[axis] != coord_array.size:
        raise ValueError(
            f"Data shape along axis {axis} ({nd_data.shape[axis]}) "
            f"does not match coordinate array length ({coord_array.size})"
        )

    coord_min, coord_max = float(section[0]), float(section[1])
    if coord_min > coord_max:
        coord_min, coord_max = coord_max, coord_min

    mask = (coord_array >= coord_min) & (coord_array <= coord_max)
    indices = np.flatnonzero(mask)

    return coord_array[indices], np.take(nd_data, indices, axis=axis)


def _style_for_component(
    component: str,
    ndim: int,
    data: np.ndarray | None = None,
    domain: str = "time",
):
    """Return style parameters for a component."""
    if ndim == 1:
        color_index = {"abs": 0, "real": 1, "imag": 2, "phase": 3}.get(component, 0)
        style_index = {"abs": 0, "real": 1, "imag": 2, "phase": 3}.get(component, 0)
        return COLORS[color_index % len(COLORS)], LINE_STYLES[style_index % len(LINE_STYLES)]

    if ndim == 2:
        if component in {"real", "imag", "phase"}:
            if data is None or np.size(data) == 0:
                return plt.get_cmap("RdBu_r"), None

            vmax = float(np.max(np.abs(data)))
            if vmax > 0:
                return plt.get_cmap("RdBu_r"), TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
            return plt.get_cmap("RdBu_r"), None

        if domain == "freq":
            return "plasma", None
        return "viridis", None

    raise ValueError("ndim must be 1 or 2")


def _component_data(data: np.ndarray, component: str) -> tuple[np.ndarray, str]:
    """Return transformed data and a base title."""
    if component == "real":
        return np.real(data), r"$\text{Real}$"
    if component == "imag":
        return np.imag(data), r"$\text{Imag}$"
    if component == "abs":
        return np.abs(data), r"$\text{Abs}$"
    if component == "phase":
        return np.angle(data), r"$\text{Phase}$"
    raise ValueError("Invalid component.")


def _domain_labels(domain: str, ndim: int):
    """Return labels for lab-frame plotting."""
    if domain not in {"time", "freq"}:
        raise ValueError("Invalid domain. Use 'time' or 'freq'.")

    signal_label = r"$E_{k_S}$"
    title_suffix = "Time domain signal" if domain == "time" else "Spectrum"

    if ndim == 1:
        if domain == "time":
            return r"$t_{\mathrm{det}}$ [fs]", signal_label, title_suffix
        return r"$\omega_{\mathrm{det}}$ [$10^4$ cm$^{-1}$]", signal_label, title_suffix

    if ndim == 2:
        if domain == "time":
            return (
                r"$t_{\mathrm{coh}}$ [fs]",
                r"$t_{\mathrm{det}}$ [fs]",
                signal_label,
                title_suffix,
            )
        return (
            r"$\omega_{\mathrm{coh}}$ [$10^4$ cm$^{-1}$]",
            r"$\omega_{\mathrm{det}}$ [$10^4$ cm$^{-1}$]",
            signal_label,
            title_suffix,
        )

    raise ValueError("ndim must be 1 or 2")


def add_custom_contour_lines(
    x: np.ndarray,
    y: np.ndarray,
    data: np.ndarray,
    component: str,
    level_count: int = 10,
    ax: plt.Axes | None = None,
) -> None:
    """Add contour lines to a 2D plot.

    Parameters
    ----------
    x
        Horizontal axis values. Must match the second dimension of ``data``.
    y
        Vertical axis values. Must match the first dimension of ``data``.
    data
        2D array in plotting orientation, i.e. shape ``(len(y), len(x))``.
    component
        One of ``real``, ``imag``, ``abs``, ``phase``.
    """
    if ax is None:
        ax = plt.gca()

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    data = np.ma.asarray(data)

    if data.ndim != 2:
        raise ValueError("Contour data must be 2D")
    if data.shape != (y.size, x.size):
        raise ValueError(
            f"Contour data shape {data.shape} does not match "
            f"(len(y), len(x)) = {(y.size, x.size)}"
        )

    filled = np.asarray(data.filled(np.nan), dtype=float)
    finite = np.isfinite(filled)
    if not np.any(finite):
        return

    if component in {"real", "imag", "phase"}:
        vmax = float(np.nanmax(np.abs(filled)))
        if vmax <= 0:
            return

        if level_count == 10:
            percents = np.arange(0.05, 1.0, 0.10)
        else:
            percents = np.linspace(0.05, 0.95, level_count)

        positive_levels = percents * vmax
        negative_levels = -percents[::-1] * vmax

        has_pos = np.nanmax(filled) > 0
        has_neg = np.nanmin(filled) < 0

        if has_pos:
            ax.contour(
                x,
                y,
                filled,
                levels=positive_levels,
                colors=COLORS[0],
                linewidths=1.0,
                alpha=0.9,
            )
        if has_neg:
            ax.contour(
                x,
                y,
                filled,
                levels=negative_levels,
                colors=COLORS[1],
                linewidths=1.0,
                alpha=0.9,
                linestyles=LINE_STYLES[1],
            )
        return

    data_min = float(np.nanmin(filled))
    data_max = float(np.nanmax(filled))
    if np.isclose(data_min, data_max):
        return

    levels = np.linspace(data_min, data_max, level_count + 2)[1:-1]
    if levels.size == 0:
        return

    contour_plot = ax.contour(
        x,
        y,
        filled,
        levels=levels,
        colors=COLORS[1],
        linewidths=0.7,
        alpha=0.8,
    )
    ax.clabel(
        contour_plot,
        inline=True,
        fontsize=8,
        fmt="%.2f",
        levels=contour_plot.levels[::2],
    )


def _format_text_box_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.3g}"
    if isinstance(value, (int, str)):
        return str(value)
    if isinstance(value, np.ndarray):
        return f"array(shape={value.shape})"
    return (
        str(value).replace("_", r"\_").replace("^", r"\^").replace("{", r"\{").replace("}", r"\}")
    )


def add_text_box(
    ax: plt.Axes,
    kwargs: dict,
    position: tuple[float, float] = (0.98, 0.98),
    fontsize: int = 7,
    coords: Literal["axes", "figure"] = "axes",
):
    """Add a small info box that does not affect layout."""
    if not kwargs:
        return

    info_text = "\n".join(
        f"{key}: {_format_text_box_value(value)}" for key, value in kwargs.items()
    )

    if coords == "figure":
        fig = ax.figure
        artist = fig.text(
            position[0],
            position[1],
            info_text,
            transform=fig.transFigure,
            fontsize=fontsize,
            va="top",
            ha="right",
            bbox=dict(boxstyle="round,pad=0.3", alpha=0.05, edgecolor="black"),
        )
    else:
        artist = ax.text(
            position[0],
            position[1],
            info_text,
            transform=ax.transAxes,
            fontsize=fontsize,
            va="top",
            ha="right",
            bbox=dict(boxstyle="round,pad=0.3", alpha=0.05, edgecolor="black"),
            clip_on=False,
        )

    try:
        artist.set_in_layout(False)
    except Exception:
        pass


def convert_plot_axes(
    nu_coh: np.ndarray | None,
    nu_det: np.ndarray,
    *,
    carrier_freq_cm: float,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Convert rotating-frame detuning axes to lab-frame optical axes."""
    shift = float(carrier_freq_cm) * 1e-4
    nu_coh_plot = None if nu_coh is None else nu_coh + shift
    nu_det_plot = nu_det + shift
    return nu_coh_plot, nu_det_plot


def _materialize_sparse_roi_for_plot(
    axis_det: np.ndarray,
    data: sp.spmatrix,
    axis_coh: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    """Convert a sparse full-grid ROI into a small dense block for plotting only."""
    coo = data.tocoo()

    if axis_coh is None:
        if data.shape[1] != 1:
            raise ValueError("1D sparse spectra must have shape (n_det, 1)")
        if coo.nnz == 0:
            return None, np.asarray(axis_det[:0], dtype=float), np.zeros(0, dtype=data.dtype)

        order = np.argsort(coo.row)
        rows = coo.row[order]
        vals = np.asarray(coo.data)[order]
        return None, np.asarray(axis_det, dtype=float)[rows], vals

    if coo.nnz == 0:
        empty_coh = np.asarray(axis_coh[:0], dtype=float)
        empty_det = np.asarray(axis_det[:0], dtype=float)
        return empty_coh, empty_det, np.zeros((0, 0), dtype=data.dtype)

    rows = np.unique(coo.row)
    cols = np.unique(coo.col)

    row_pos = np.searchsorted(rows, coo.row)
    col_pos = np.searchsorted(cols, coo.col)

    dense_roi = np.zeros((rows.size, cols.size), dtype=data.dtype)
    dense_roi[row_pos, col_pos] = coo.data

    return (
        np.asarray(axis_coh, dtype=float)[rows],
        np.asarray(axis_det, dtype=float)[cols],
        dense_roi,
    )
