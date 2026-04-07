from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from matplotlib.colors import Normalize, TwoSlopeNorm
from plotstyle import (
    COLORS,
    LINE_STYLES,
    apply_decimal_axis_ticks,
    apply_decimal_colorbar_ticks,
    init_style,
    simplify_figure_text,
)

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


def _finite_values(data: np.ndarray) -> np.ndarray:
    data = np.ma.asarray(data)
    if np.ma.isMaskedArray(data):
        values = data.compressed()
    else:
        values = np.ravel(np.asarray(data))

    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _default_plot_norm(
    component: str,
    plot_data: np.ndarray,
) -> Normalize | None:
    """
    Default plotting norm if no explicit norm is passed.

    real/imag/phase:
        symmetric about zero using the actually shown data
    abs:
        from 0 to the actually shown maximum
    """
    values = _finite_values(plot_data)
    if values.size == 0:
        return None

    if component in {"real", "imag", "phase"}:
        vmax = float(np.max(np.abs(values)))
        if vmax <= 0.0:
            return None
        return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    if component == "abs":
        vmax = float(np.max(values))
        if vmax <= 0.0:
            return None
        return Normalize(vmin=0.0, vmax=vmax)

    raise ValueError(f"Unsupported component: {component!r}")
    
    
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
    #axes[-1].set_xlabel(r"$t\,/\,\mathrm{fs}$")

    plt.tight_layout()
    simplify_figure_text(fig)
    return fig


def _prepare_el_field_input(
    axis_det: np.ndarray,
    data: np.ndarray | sp.spmatrix,
    axis_coh: np.ndarray | None,
    domain: Literal["time", "freq"],
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    """Prepare plotting axes and dense data once before per-component rendering."""
    axis_det = np.asarray(axis_det, dtype=float)
    axis_coh = None if axis_coh is None else np.asarray(axis_coh, dtype=float)

    if domain == "freq" and isinstance(data, sp.spmatrix):
        axis_coh, axis_det, dense_data = _materialize_sparse_roi_for_plot(
            axis_det,
            data,
            axis_coh,
        )
        if np.size(dense_data) == 0:
            raise ValueError(
                "Empty frequency ROI after sparse cropping. "
                "Check whether SECTION is specified in lab-frame units and "
                "was converted correctly before calling compute_spectra."
            )
    else:
        dense_data = data.toarray() if isinstance(data, sp.spmatrix) else np.asarray(data)

    dense_data = np.asarray(dense_data)
    if axis_coh is None and dense_data.ndim == 2 and dense_data.shape[1] == 1:
        dense_data = dense_data.squeeze(axis=1)

    return axis_coh, axis_det, dense_data


def _prepare_component_plot_data(
    data: np.ndarray,
    component: str,
    *,
    cutoff_percent: float = 0.0,
    normalization_factor: float | None = None,
) -> tuple[np.ndarray, str, bool]:
    """Return transformed component data together with its title and norm state."""
    plot_data, base_title = _component_data(np.asarray(data), component)

    if cutoff_percent > 0 and plot_data.size > 0:
        threshold = cutoff_percent / 100.0 * np.max(np.abs(plot_data))
        plot_data = np.ma.masked_where(np.abs(plot_data) < threshold, plot_data)

    is_normalised = normalization_factor is not None and component != "phase"
    if is_normalised:
        factor = float(normalization_factor)
        if np.isfinite(factor) and factor > 0.0:
            plot_data = plot_data / factor

    return plot_data, base_title, is_normalised


def validate_plot_components(components: Sequence[str]) -> tuple[str, ...]:
    """Validate the component list used for composite job plots."""
    allowed = {"real", "imag", "abs"}
    validated: list[str] = []
    seen: set[str] = set()

    if not components:
        raise ValueError("components must be a non-empty sequence drawn from {'real', 'imag', 'abs'}")

    for component in components:
        if component not in allowed:
            raise ValueError(
                f"Unsupported composite component: {component!r}. "
                "Allowed values are 'real', 'imag', and 'abs'."
            )
        if component in seen:
            raise ValueError(f"Duplicate composite component: {component!r}")
        seen.add(component)
        validated.append(component)

    return tuple(validated)


def _composite_signed_colorbar_label(
    components: Sequence[str],
    *,
    is_normalised: bool,
) -> str:
    """Return an explicit shared colorbar label for signed components."""
    numerators: list[str] = []
    for component in components:
        if component == "real":
            numerators.append(r"\mathrm{Re}[E_{k_S}]")
        elif component == "imag":
            numerators.append(r"\mathrm{Im}[E_{k_S}]")

    if not numerators:
        raise ValueError("Signed colorbar label requires at least one signed component")

    numerator = numerators[0] if len(numerators) == 1 else f"({', '.join(numerators)})"
    if is_normalised:
        return rf"${numerator} / |E_{{k_S}}|_{{\max}}$"
    return rf"${numerator}$"


def _component_colorbar_specs(
    components: Sequence[str],
    panels: Sequence[dict[str, object]],
    mappables: dict[str, object],
) -> list[tuple[object, str]]:
    """Build shared colorbar specs for a component row."""
    colorbar_specs: list[tuple[object, str]] = []
    signed_components = [component for component in components if component in {"real", "imag"}]
    if signed_components:
        first_component = signed_components[0]
        signed_panel = next(panel for panel in panels if panel["component"] == first_component)
        colorbar_specs.append(
            (
                mappables[first_component],
                _composite_signed_colorbar_label(
                    signed_components,
                    is_normalised=bool(signed_panel["is_normalised"]),
                ),
            )
        )
    if "abs" in components:
        abs_panel = next(panel for panel in panels if panel["component"] == "abs")
        colorbar_specs.append(
            (
                mappables["abs"],
                _component_signal_label(
                    "abs",
                    is_normalised=bool(abs_panel["is_normalised"]),
                ),
            )
        )
    return colorbar_specs


def _colorbar_positions_for_anchor(
    anchor_box,
    *,
    colorbar_count: int,
) -> list[tuple[float, float, float, float]]:
    """Place one or two colorbars flush to the right of a subplot row."""
    if colorbar_count == 1:
        return [(anchor_box.x1 + 0.015, anchor_box.y0, 0.02, anchor_box.height)]
    if colorbar_count == 2:
        return [
            (anchor_box.x1 + 0.015, anchor_box.y0, 0.02, anchor_box.height),
            (anchor_box.x1 + 0.065, anchor_box.y0, 0.02, anchor_box.height),
        ]
    return []


def _add_component_colorbars(
    fig: plt.Figure,
    *,
    anchor_ax: plt.Axes,
    colorbar_specs: Sequence[tuple[object, str]],
) -> None:
    """Add one or two row-aligned colorbars next to the anchor axis."""
    anchor_box = anchor_ax.get_position()
    positions = _colorbar_positions_for_anchor(
        anchor_box,
        colorbar_count=len(colorbar_specs),
    )

    for idx, ((mappable, label), position) in enumerate(zip(colorbar_specs, positions)):
        cax = fig.add_axes(position)
        cax.set_in_layout(False)
        cbar = fig.colorbar(mappable, cax=cax)
        if label:
            cbar.set_label(label)
        if len(colorbar_specs) == 2 and idx == 0:
            cbar.set_label("")
        apply_decimal_colorbar_ticks(cbar, decimals=1)


def _signal_grid_figsize(n_rows: int, n_cols: int) -> tuple[float, float]:
    panel = 3.0
    w_gap = 0.0
    h_gap = 0.0
    left = 0.0
    right = 0.0
    bottom = 0.0
    top = 0.0

    fig_w = left + n_cols * panel + (n_cols - 1) * w_gap + right
    fig_h = bottom + n_rows * panel + (n_rows - 1) * h_gap + top
    return fig_w, fig_h


def _panel_display_vmax(panel: dict[str, object]) -> float:
    component = str(panel["component"])
    plot_data = np.asarray(panel["plot_data"], dtype=float)
    plot_norm = panel["plot_norm"]
    if plot_norm is None:
        plot_norm = _default_plot_norm(component, plot_data)

    if plot_norm is not None:
        vmax = getattr(plot_norm, "vmax", None)
        if vmax is not None and np.isfinite(vmax):
            return float(vmax)
        vmin = getattr(plot_norm, "vmin", None)
        if vmin is not None and np.isfinite(vmin):
            return abs(float(vmin))

    values = _finite_values(plot_data)
    if values.size == 0:
        return 0.0
    if component == "abs":
        return float(np.max(values))
    return float(np.max(np.abs(values)))


def _grouped_1d_axis_layout(
    components: Sequence[str],
) -> dict[str, dict[str, int | str]]:
    signed_indices = [idx for idx, component in enumerate(components) if component in {"real", "imag"}]
    abs_index = next((idx for idx, component in enumerate(components) if component == "abs"), None)

    layout: dict[str, dict[str, int | str]] = {}
    if signed_indices and abs_index is not None:
        signed_left = min(signed_indices)
        signed_right = max(signed_indices)
        if abs_index < signed_left:
            layout["abs"] = {"anchor_index": abs_index, "side": "left"}
            layout["signed"] = {"anchor_index": signed_right, "side": "right"}
        else:
            layout["signed"] = {"anchor_index": signed_left, "side": "left"}
            layout["abs"] = {"anchor_index": abs_index, "side": "right"}
        return layout

    if signed_indices:
        layout["signed"] = {"anchor_index": min(signed_indices), "side": "left"}
    if abs_index is not None:
        layout["abs"] = {"anchor_index": abs_index, "side": "left"}
    return layout


def _configure_visible_y_axis(ax: plt.Axes, *, side: str) -> None:
    if side == "right":
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.tick_params(axis="y", labelleft=False, left=False, labelright=True, right=True)
        return

    ax.yaxis.set_ticks_position("left")
    ax.yaxis.set_label_position("left")
    ax.tick_params(axis="y", labelleft=True, left=True, labelright=False, right=False)


def _expanded_1d_ylim(
    vmin: float,
    vmax: float,
    *,
    pad_fraction: float = 0.025,
) -> tuple[float, float]:
    scale = max(abs(float(vmin)), abs(float(vmax)))
    if not np.isfinite(scale) or scale <= 0.0:
        return float(vmin), float(vmax)

    pad = pad_fraction * scale
    if float(vmin) >= 0.0:
        return float(vmin), float(vmax) + pad
    return float(vmin) - pad, float(vmax) + pad


def _apply_grouped_1d_row_axes(
    row_axes: Sequence[plt.Axes],
    row_panels: Sequence[dict[str, object]],
    *,
    components: Sequence[str],
) -> None:
    layout = _grouped_1d_axis_layout(components)

    for axis in row_axes:
        axis.tick_params(
            axis="y",
            labelleft=False,
            left=False,
            labelright=False,
            right=False,
        )

    signed_spec = layout.get("signed")
    if signed_spec is not None:
        signed_indices = [idx for idx, component in enumerate(components) if component in {"real", "imag"}]
        signed_vmax = max(
            (
                _panel_display_vmax(panel)
                for panel in row_panels
                if str(panel["component"]) in {"real", "imag"}
            ),
            default=0.0,
        )
        if np.isfinite(signed_vmax) and signed_vmax > 0.0:
            ylim = _expanded_1d_ylim(-signed_vmax, signed_vmax)
            for idx in signed_indices:
                row_axes[idx].set_ylim(*ylim)

        signed_axis = row_axes[int(signed_spec["anchor_index"])]
        _configure_visible_y_axis(signed_axis, side=str(signed_spec["side"]))
        apply_decimal_axis_ticks(signed_axis, axis="y", decimals=1)

    abs_spec = layout.get("abs")
    if abs_spec is not None:
        abs_axis = row_axes[int(abs_spec["anchor_index"])]
        abs_vmax = max(
            (
                _panel_display_vmax(panel)
                for panel in row_panels
                if str(panel["component"]) == "abs"
            ),
            default=0.0,
        )
        if np.isfinite(abs_vmax) and abs_vmax > 0.0:
            abs_axis.set_ylim(*_expanded_1d_ylim(0.0, abs_vmax))

        _configure_visible_y_axis(abs_axis, side=str(abs_spec["side"]))
        apply_decimal_axis_ticks(abs_axis, axis="y", decimals=1)

def plot_el_field_signal_grid(
    axis_det: np.ndarray,
    signal_datas: Sequence[np.ndarray | sp.spmatrix],
    *,
    signal_labels: Sequence[str],
    components: Sequence[str],
    axis_coh: np.ndarray | None = None,
    domain: Literal["time", "freq"] = "time",
    cutoff_percent: float = 0.0,
    contour_lines: bool = False,
    normalization_factors: Sequence[float | None] | None = None,
    plot_norms_by_signal: Sequence[dict[str, Normalize | None]] | None = None,
    suptitle: str | None = None,
    **kwargs: dict,
) -> plt.Figure:
    """Plot all selected signal types in one domain-wide component grid."""
    components = validate_plot_components(components)
    signal_datas = list(signal_datas)
    signal_labels = list(signal_labels)

    if not signal_datas:
        raise ValueError("signal_datas must contain at least one signal")
    if len(signal_labels) != len(signal_datas):
        raise ValueError("signal_labels must have the same length as signal_datas")

    if normalization_factors is None:
        normalization_factors = [None] * len(signal_datas)
    else:
        normalization_factors = list(normalization_factors)
        if len(normalization_factors) != len(signal_datas):
            raise ValueError("normalization_factors must match signal_datas length")

    if plot_norms_by_signal is None:
        plot_norms_by_signal = [{} for _ in signal_datas]
    else:
        plot_norms_by_signal = [dict(plot_norms) for plot_norms in plot_norms_by_signal]
        if len(plot_norms_by_signal) != len(signal_datas):
            raise ValueError("plot_norms_by_signal must match signal_datas length")

    rows: list[dict[str, object]] = []
    ndim: int | None = None
    for signal_data, normalization_factor, plot_norms in zip(
        signal_datas,
        normalization_factors,
        plot_norms_by_signal,
    ):
        row_axis_coh, row_axis_det, dense_data = _prepare_el_field_input(
            axis_det,
            signal_data,
            axis_coh,
            domain,
        )
        row_panels: list[dict[str, object]] = []
        for component in components:
            plot_data, base_title, is_normalised = _prepare_component_plot_data(
                dense_data,
                component,
                cutoff_percent=cutoff_percent,
                normalization_factor=normalization_factor,
            )
            row_panels.append(
                {
                    "component": component,
                    "plot_data": plot_data,
                    "base_title": base_title,
                    "is_normalised": is_normalised,
                    "plot_norm": plot_norms.get(component),
                }
            )

        row_ndim = int(np.ndim(row_panels[0]["plot_data"]))
        if row_ndim not in {1, 2}:
            raise ValueError("Signal grid plots require 1D or 2D component data")
        if row_ndim == 2 and row_axis_coh is None:
            raise ValueError("axis_coh must be provided for 2D signal grid plots")
        if ndim is None:
            ndim = row_ndim
        elif row_ndim != ndim:
            raise ValueError("All signals in a grid must have the same dimensionality")

        rows.append(
            {
                "axis_coh": row_axis_coh,
                "axis_det": row_axis_det,
                "panels": row_panels,
            }
        )
    assert ndim is not None

    figsize = _signal_grid_figsize(len(rows), len(components))
    sharex = True
    sharey = (ndim == 2)   

    fig, axes = plt.subplots(
        len(rows),
        len(components),
        figsize=figsize,
        squeeze=False,
        sharex=sharex,
        sharey=sharey,
    )

    for row_idx, (row_label, row) in enumerate(zip(signal_labels, rows)):
        row_axes = axes[row_idx]
        row_panels = row["panels"]
        row_axis_det = row["axis_det"]
        row_axis_coh = row["axis_coh"]
        is_bottom_row = row_idx == len(rows) - 1
        row_name = str(row_label).replace("_", " ").title()
        for col_idx, (axis, panel) in enumerate(zip(row_axes, row_panels)):
            component = str(panel["component"])
            plot_data = panel["plot_data"]
            base_title = str(panel["base_title"])
            is_normalised = bool(panel["is_normalised"])
            plot_norm = panel["plot_norm"]
            title = base_title if row_idx == 0 else ""

            if np.ndim(plot_data) == 1:
                _plot_el_field_1d(
                    ax=axis,
                    axis_det=row_axis_det,
                    plot_data=plot_data,
                    component=component,
                    domain=domain,
                    base_title=base_title,
                    plot_norm=plot_norm,
                    is_normalised=is_normalised,
                    title=title,
                    show_xlabel=is_bottom_row,
                    show_ylabel=False,
                    show_legend=False,
                )
                axis.set_box_aspect(1)
            else:
                _plot_el_field_2d(
                    ax=axis,
                    axis_coh=row_axis_coh,
                    axis_det=row_axis_det,
                    plot_data=plot_data,
                    component=component,
                    domain=domain,
                    base_title=base_title,
                    contour_lines=contour_lines,
                    plot_norm=plot_norm,
                    is_normalised=is_normalised,
                    title=title,
                    show_xlabel=is_bottom_row,
                    show_ylabel=(col_idx == 0),
                    add_colorbar=False,
                    square_axes=True,
                )
                apply_decimal_axis_ticks(axis, axis="x", decimals=2)
                apply_decimal_axis_ticks(axis, axis="y", decimals=2)

            axis.tick_params(labelbottom=is_bottom_row)

        row_label_artist = row_axes[0].annotate(
            row_name,
            xy=(0.02, 0.94),
            xycoords="axes fraction",
            ha="left",
            va="top",
        )
        row_label_artist.set_in_layout(False)

    is_any_normalised = any(
        bool(panel["is_normalised"])
        for row in rows
        for panel in row["panels"]
    )
    if ndim == 1:
        for row_axes, row in zip(axes, rows):
            _apply_grouped_1d_row_axes(
                row_axes,
                row["panels"],
                components=components,
            )
    if kwargs:
        add_text_box(axes[0, 0], kwargs=kwargs)
    if suptitle:
        fig.suptitle(suptitle)

    if ndim == 1:
        top_margin = 0.96 if suptitle else 0.98
        _add_simple_1d_ylabel(
            axes,
            components=components,
            is_normalised=is_any_normalised,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, top_margin))
    else:
        fig.tight_layout()

    if ndim == 2:
        for row_idx, row in enumerate(rows):
            row_panels = row["panels"]
            row_mappables = {
                str(panel["component"]): axes[row_idx, col_idx].images[0]
                for col_idx, panel in enumerate(row_panels)
                if str(panel["component"]) in components
            }
            colorbar_specs = _component_colorbar_specs(components, row_panels, row_mappables)
            if colorbar_specs:
                _add_component_colorbars(
                    fig,
                    anchor_ax=axes[row_idx, -1],
                    colorbar_specs=colorbar_specs,
                )

    simplify_figure_text(fig)
    return fig


def _shared_1d_signal_axis_label(*, is_normalised: bool) -> str:
    if is_normalised:
        return r"$E_{k_S} / |E_{k_S}|_{\max}$"
    return r"$E_{k_S}$"


def _add_simple_1d_ylabel(
    axes: np.ndarray,
    *,
    components: Sequence[str],
    is_normalised: bool,
) -> None:
    layout = _grouped_1d_axis_layout(components)
    signed_spec = layout.get("signed")
    if signed_spec is None:
        return

    mid_row = axes.shape[0] // 2
    signed_axis = axes[mid_row, int(signed_spec["anchor_index"])]
    signed_axis.set_ylabel(
        _shared_1d_signal_axis_label(is_normalised=is_normalised),
        labelpad=6,
    )

def _plot_el_field_1d(
    *,
    ax: plt.Axes,
    axis_det: np.ndarray,
    plot_data: np.ndarray,
    component: str,
    domain: str,
    base_title: str,
    plot_norm: Normalize | None = None,
    is_normalised: bool = False,
    title: str | None = None,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    show_legend: bool = True,
) -> None:
    color, linestyle = _style_for_component(component, 1, domain=domain)

    values = np.asarray(plot_data, dtype=float)
    x_label, _, title_suffix = _domain_labels(domain, 1)
    y_label = _component_signal_label(component, is_normalised=is_normalised)

    ax.plot(
        axis_det,
        values,
        color=color,
        linestyle=linestyle,
        label=base_title.strip(),
    )
    ax.set_title(base_title + " " + title_suffix if title is None else title)
    ax.set_xlabel(x_label if show_xlabel else "")
    ax.set_ylabel(y_label if show_ylabel else "")
    if not show_xlabel:
        ax.tick_params(labelbottom=False)
    if not show_ylabel:
        ax.tick_params(labelleft=False)

    if plot_norm is not None:
        vmin = getattr(plot_norm, "vmin", None)
        vmax = getattr(plot_norm, "vmax", None)
        if (
            vmin is not None
            and vmax is not None
            and np.isfinite(vmin)
            and np.isfinite(vmax)
            and float(vmin) < float(vmax)
        ):
            ax.set_ylim(*_expanded_1d_ylim(float(vmin), float(vmax)))

    if show_legend:
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
    plot_norm: Normalize | None = None,
    is_normalised: bool = False,
    title: str | None = None,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    add_colorbar: bool = True,
    square_axes: bool = False,
):
    colormap = _style_for_component(component, 2, domain=domain)
    norm = plot_norm if plot_norm is not None else _default_plot_norm(component, plot_data)

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

    x_label, y_label, _, title_suffix = _domain_labels(domain, 2)
    cbar_label = _component_signal_label(component, is_normalised=is_normalised)
    ax.set_xlabel(x_label if show_xlabel else "")
    ax.set_ylabel(y_label if show_ylabel else "")
    ax.set_title(base_title + " " + title_suffix if title is None else title)
    if not show_xlabel:
        ax.tick_params(labelbottom=False)
    if not show_ylabel:
        ax.tick_params(labelleft=False)
    if square_axes:
        ax.set_box_aspect(1)

    if add_colorbar:
        cbar = ax.figure.colorbar(im, ax=ax, label=cbar_label)
        apply_decimal_colorbar_ticks(cbar, decimals=1)

    return im

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
    domain: str = "time",
):
    """Return style parameters for a component."""
    if ndim == 1:
        color_index = {"abs": 0, "real": 1, "imag": 2, "phase": 3}.get(component, 0)
        style_index = {"abs": 0, "real": 1, "imag": 2, "phase": 3}.get(component, 0)
        return COLORS[color_index % len(COLORS)], LINE_STYLES[style_index % len(LINE_STYLES)]

    if ndim == 2:
        if component in {"real", "imag", "phase"}:
            return plt.get_cmap("RdBu_r")
        if domain == "freq":
            return "plasma"
        return "viridis"

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


def _component_signal_label(component: str, *, is_normalised: bool = False) -> str:
    """Return an explicit axis/colorbar label for the plotted component."""
    if component == "real":
        numerator = r"\mathrm{Re}[E_{k_S}]"
    elif component == "imag":
        numerator = r"\mathrm{Im}[E_{k_S}]"
    elif component == "abs":
        numerator = r"|E_{k_S}|"
    elif component == "phase":
        return r"$\arg(E_{k_S})$"
    else:
        raise ValueError(f"Invalid component: {component!r}")

    if is_normalised:
        return rf"${numerator} / |E_{{k_S}}|_{{\max}}$"
    return rf"${numerator}$"


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
