from matplotlib.colors import TwoSlopeNorm
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, Tuple, Optional
import scipy.sparse as sp

from ..core.laser_system import (
    LaserPulseSequence,
    # functions
    pulse_envelopes,
    single_pulse_envelope,
    e_pulses,
    epsilon_pulses,
)


from plotstyle import init_style, COLORS, LINE_STYLES, simplify_figure_text

init_style()


def plot_pulse_envelopes(
    times: np.ndarray, pulse_seq: LaserPulseSequence, ax=None, show_legend=True
):
    """
    Plot the combined pulse envelope over time for N pulses using LaserPulseSequence.

    Parameters:
        times (np.ndarray): Array of time values.
        pulse_seq (LaserPulseSequence): LaserPulseSequence object containing pulses.
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. Defaults to None.

    Returns:
        tuple: (fig, ax) - Figure and axes objects with the plot.
    """
    # Calculate the combined envelope over time
    envelope = pulse_envelopes(times, pulse_seq)

    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots()  # Plot combined envelope
    else:
        fig = ax.figure
    ax.plot(
        times,
        envelope,
        label=r"$\text{Combined Envelope}$",
        linestyle=LINE_STYLES[0],
        alpha=0.8,
        color=COLORS[0],
    )  # Styles for individual pulses will cycle
    n_styles = len(LINE_STYLES)
    n_colors = len(COLORS)

    # Plot individual envelopes and annotations (support any number of pulses)
    for idx, pulse in enumerate(pulse_seq.pulses):

        # get the individual envelope from the pulse_envelopes function
        individual_envelope = single_pulse_envelope(times, pulse)
        t_peak = pulse.pulse_peak_time
        Delta_width = pulse.pulse_fwhm_fs
        ax.plot(
            times,
            individual_envelope,
            label=rf"$\text{{Pulse {idx + 1}}}$",
            linestyle=LINE_STYLES[(idx + 1) % n_styles],  # avoid reusing 0 used above
            alpha=0.6,
            color=COLORS[(idx + 1) % n_colors],
        )  # Annotate pulse key points
        ax.axvline(
            t_peak - Delta_width,
            linestyle=LINE_STYLES[3],
            label=rf"$t_{{peak, {idx + 1}}} - \Delta_{{{idx + 1}}}$",
            alpha=0.4,
            color=COLORS[(idx + 1) % n_colors],
        )
        ax.axvline(
            t_peak,
            linestyle=LINE_STYLES[0],
            label=rf"$t_{{peak, {idx + 1}}}$",
            alpha=0.8,
            color=COLORS[(idx + 1) % n_colors],
            linewidth=2,
        )
        ax.axvline(
            t_peak + Delta_width,
            linestyle=LINE_STYLES[3],
            label=rf"$t_{{peak, {idx + 1}}} + \Delta_{{{idx + 1}}}$",
            alpha=0.4,
            color=COLORS[(idx + 1) % n_colors],
        )

    # Final plot labeling
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"Envelope Amplitude")
    ax.set_title(r"Pulse Envelopes for Up to Three Pulses")
    if show_legend:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    simplify_figure_text(fig)
    return fig, ax


def plot_e_pulses(times: np.ndarray, pulse_seq: LaserPulseSequence, ax=None, show_legend=True):
    """
    Plot the RWA electric field (envelope only) over time for N pulses using LaserPulseSequence.

    Parameters:
        times (np.ndarray): Array of time values.
        pulse_seq (LaserPulseSequence): LaserPulseSequence object containing pulses.
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. Defaults to None.

    Returns:
        tuple: (fig, ax) - Figure and axes objects with the plot.
    """
    # Calculate the RWA electric field over time
    E_field = np.array([e_pulses(t, pulse_seq) for t in times])

    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))  # Plot real and imaginary parts
    else:
        fig = ax.figure
    ax.plot(
        times,
        np.real(E_field),
        label=r"$\mathrm{Re}[E(t)]$",
        linestyle=LINE_STYLES[0],
        color=COLORS[0],
    )
    ax.plot(
        times,
        np.imag(E_field),
        label=r"$\mathrm{Im}[E(t)]$",
        linestyle=LINE_STYLES[1],
        color=COLORS[1],
    )

    # Styles for any number of pulses (cycle through colors)
    n_colors = len(COLORS)

    # Plot pulse peak times for all pulses
    for idx, pulse in enumerate(pulse_seq.pulses):
        t_peak = pulse.pulse_peak_time
        ax.axvline(
            t_peak,
            linestyle=LINE_STYLES[3],  # "dotted"
            label=rf"$t_{{peak, {idx + 1}}}$",
            color=COLORS[(idx + 2) % n_colors],  # offset to avoid field line colors
        )

    # Final plot labeling
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"Electric Field (RWA)")
    ax.set_title(r"RWA Electric Field Components")
    if show_legend:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    simplify_figure_text(fig)
    return fig, ax


def plot_epsilon_pulses(
    times: np.ndarray, pulse_seq: LaserPulseSequence, ax=None, show_legend=True
):
    """
    Plot the full electric field (with carrier) over time for N pulses using LaserPulseSequence.

    Parameters:
        times (np.ndarray): Array of time values.
        pulse_seq (LaserPulseSequence): LaserPulseSequence object containing pulses.
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. Defaults to None.

    Returns:
        tuple: (fig, ax) - Figure and axes objects with the plot.
    """
    # Calculate the full electric field over time
    Epsilon_field = np.array([epsilon_pulses(t, pulse_seq) for t in times])
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    ax.plot(
        times,
        np.real(Epsilon_field),
        label=r"$\mathrm{Re}[\varepsilon(t)]$",
        linestyle=LINE_STYLES[0],
        color=COLORS[3],
    )
    ax.plot(
        times,
        np.imag(Epsilon_field),
        label=r"$\mathrm{Im}[\varepsilon(t)]$",
        linestyle=LINE_STYLES[1],
        color=COLORS[4],
    )
    ax.plot(
        times,
        np.abs(Epsilon_field),
        label=r"$|\varepsilon(t)|$",
        linestyle=LINE_STYLES[2],
        color=COLORS[5],
    )
    n_colors = len(COLORS)
    for idx, pulse in enumerate(pulse_seq.pulses):
        t_peak = pulse.pulse_peak_time
        ax.axvline(
            t_peak,
            linestyle=LINE_STYLES[3],
            label=rf"$t_{{peak, {idx + 1}}}$",
            color=COLORS[idx % n_colors],
        )
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"Electric Field (Full)")
    ax.set_title(r"Full Electric Field with Carrier")
    if show_legend:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    simplify_figure_text(fig)
    return fig, ax


def plot_all_pulse_components(times: np.ndarray, pulse_seq: LaserPulseSequence, ax=None):
    """
    Plot all pulse components: envelope, RWA field, and full field in a comprehensive figure.

    Parameters:
        times (np.ndarray): Array of time values.
        pulse_seq (LaserPulseSequence): LaserPulseSequence object containing pulses.
        ax (array-like of matplotlib.axes.Axes, optional): Array of axes objects to plot on. If None, creates new subplots.

    Returns:
        fig (matplotlib.figure.Figure): Figure object with all plots.
    """
    # Create figure with subplots if ax is None
    if ax is None:
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    else:
        axes = ax
        fig = axes[0].figure

    # Plot pulse envelope
    plot_pulse_envelopes(times, pulse_seq, ax=axes[0])

    # Plot RWA electric field
    plot_e_pulses(times, pulse_seq, ax=axes[1])

    # Plot full electric field
    plot_epsilon_pulses(times, pulse_seq, ax=axes[2])

    # Add overall title
    fig.suptitle(
        f"Comprehensive Pulse Analysis - {len(pulse_seq.pulses)} Pulse(s)",
        fontsize=16,
        y=0.98,
    )

    plt.tight_layout()

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
    """
    Plot the evolution of the electric field and expectation values for a given t_coh and t_wait.

    Parameters:
        times_plot (np.ndarray): Time axis for the plot.
        datas (list): List of arrays of expectation values to plot.
        pulse_seq (LaserPulseSequence): Laser pulse sequence object.
        t_coh (float): Coherence time.
        t_wait (float): Waiting time.
        system: System object containing all relevant parameters.
        rwa_sl (bool): Whether to use RWA or full field.
        ax (array-like of matplotlib.axes.Axes, optional): Array of axes objects to plot on. If None, creates new subplots.
        **kwargs: Additional keyword arguments for annotation.

    Returns:
        matplotlib.figure.Figure: The figure object.
    """
    if rwa_sl:
        field_func = e_pulses
    else:
        field_func = epsilon_pulses

    # Calculate total electric field
    E0 = pulse_seq.E0
    E_total = np.array([field_func(t, pulse_seq) / E0 for t in times_plot])

    # Create plot with appropriate size
    if ax is None:
        fig, axes = plt.subplots(len(datas) + 1, 1, figsize=(14, 2 + 2 * len(datas)), sharex=True)
    else:
        axes = ax
        fig = axes[0].figure

    # Plot electric field
    axes[0].plot(
        times_plot,
        np.real(E_total),
        color=COLORS[0],
        linestyle=LINE_STYLES[0],
        label=r"$\mathrm{Re}[E(t)]$",
    )
    axes[0].plot(
        times_plot,
        np.imag(E_total),
        color=COLORS[1],
        linestyle=LINE_STYLES[1],
        label=r"$\mathrm{Im}[E(t)]$",
    )
    axes[0].set_ylabel(r"$E(t) / E_0$")
    axes[0].legend(loc="center left", bbox_to_anchor=(1, 0.5))

    # Plot expectation values
    for idx, data in enumerate(datas):
        ax = axes[idx + 1]
        if idx == len(observable_strs):
            observable_str = r"\text{Pol}"
            # also Plot imaginary part, because the polarization is complex
            ax.plot(
                times_plot,
                np.imag(data),
                color=COLORS[1],
                linestyle=LINE_STYLES[1],
                label=r"$\mathrm{Im}" + observable_str + r" \rangle$",
            )
        else:
            observable_str = observable_strs[idx]

        # Plot real part
        ax.plot(
            times_plot,
            np.real(data),
            color=COLORS[0],
            linestyle=LINE_STYLES[0],
            label=r"$\mathrm{Re}\langle " + observable_str + r" \rangle$",
        )

        ax.set_ylabel(r"$\langle " + observable_str + r" \rangle$")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    add_text_box(ax=axes[0], kwargs=kwargs)

    # Set x-label only on the bottom subplot
    axes[-1].set_xlabel(r"$t\,/\,\mathrm{fs}$")

    plt.tight_layout()
    simplify_figure_text(fig)

    return fig


def plot_el_field(
    axis_det: np.ndarray,
    data: np.ndarray,
    axis_coh: Optional[np.ndarray] = None,
    component: Literal["real", "img", "abs", "phase"] = "real",
    domain: Literal["time", "freq"] = "time",
    section: Optional[tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
    cutoff_percent: float = 5.0,
    **kwargs: dict,
) -> plt.Figure:
    if component not in ("real", "img", "abs", "phase"):
        raise ValueError("Invalid component. Must be 'real', 'img', 'abs', or 'phase'.")

    # Convert sparse to dense if needed
    if isinstance(data, sp.spmatrix):
        data = data.toarray()

    # Handle 1D data stored as (n, 1)
    if axis_coh is None and data.ndim == 2 and data.shape[1] == 1:
        data = data.squeeze(axis=1)

    # Crop data if section provided
    if section is not None:
        if data.ndim == 1:
                axis_det, data = crop_nd_data_along_axis(axis_det, data, section, axis=0)
        elif data.ndim == 2:
            axis_coh, data = crop_nd_data_along_axis(axis_coh, data, section, axis=0)
            axis_det, data = crop_nd_data_along_axis(axis_det, data, section, axis=1)

    # Select the component to plot
    plot_data, base_title = _component_data(data, component)

    if cutoff_percent > 0:
        threshold = cutoff_percent / 100 * np.max(np.abs(plot_data))
        mask = np.abs(plot_data) < threshold
        plot_data = np.ma.masked_where(mask, plot_data)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    if data.ndim == 1:
        # 1D plot
        color, linestyle = _style_for_component(component, 1, domain=domain)
        ax.plot(axis_det, plot_data, color=color, linestyle=linestyle)
        x_label, y_label, title_suffix = _domain_labels(domain, 1)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(base_title + " " + title_suffix)
        fig.tight_layout()
        add_text_box(ax, kwargs=kwargs)
        return fig

    elif data.ndim == 2:
        # 2D plot
        colormap, norm = _style_for_component(component, 2, plot_data, domain)
        im = ax.imshow(
            plot_data.T,  # Transpose to have detection on y, coherence on x
            extent=[axis_coh.min(), axis_coh.max(), axis_det.min(), axis_det.max()],
            origin="lower",
            aspect="auto",
            cmap=colormap,
            norm=norm,
            interpolation="bilinear",
        )
        x_label, y_label, cbar_label, title_suffix = _domain_labels(domain, 2)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(base_title + " " + title_suffix)
        plt.colorbar(im, ax=ax, label=cbar_label)
        fig.tight_layout()
        add_text_box(ax, kwargs=kwargs)
        return fig


# HELPER FUNCTIONS
def crop_nd_data_along_axis(
    coord_array: np.ndarray,
    nd_data: np.ndarray,
    section: tuple[float, float],
    axis: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Crop n-dimensional data along a specified axis.

    Parameters:
        coord_array (np.ndarray): 1D coordinate array for the specified axis
        nd_data (np.ndarray): N-dimensional data array
        section (tuple[float, float]): Section boundaries as (min_val, max_val)
        axis (int): Axis along which to crop (default: 0)

    Returns:
        tuple: (cropped_coord_array, cropped_nd_data)

    Raises:
        ValueError: If coordinate array length doesn't match data shape along specified axis
    """
    ### Validate input dimensions
    if coord_array.ndim != 1:
        raise ValueError("Coordinate array must be 1-dimensional")

    if nd_data.shape[axis] != len(coord_array):
        raise ValueError(
            f"Data shape along axis {axis} ({nd_data.shape[axis]}) "
            f"does not match coordinate array length ({len(coord_array)})"
        )

    if isinstance(section, list):
        coord_min, coord_max = section[axis]
    else:
        coord_min, coord_max = section

    ### Validate coordinates are within data range
    coord_min = max(coord_min, np.min(coord_array))
    coord_max = min(coord_max, np.max(coord_array))

    ### Find indices within the specified section
    indices = np.where((coord_array >= coord_min) & (coord_array <= coord_max))[0]

    ### Ensure indices are within array bounds
    indices = indices[indices < len(coord_array)]

    ### Crop coordinate array
    cropped_coords = coord_array[indices]

    ### Crop data along specified axis using advanced indexing
    cropped_data = np.take(nd_data, indices, axis=axis)

    return cropped_coords, cropped_data


# INTERNAL HELPERS (1D/2D LABEL + COMPONENT LOGIC)
def _style_for_component(
    component: str, ndim: int, data: Optional[np.ndarray] = None, domain: str = "time"
) -> Tuple:
    """Return style parameters for a given component and dimension.

    For 1D: returns (color, linestyle)
    For 2D: returns (colormap, norm)
    """
    if ndim == 1:
        color_map = {"abs": 0, "real": 1, "img": 2, "phase": 3}
        idx = color_map.get(component, 0)
        color = COLORS[idx]
        linestyle = LINE_STYLES[0]
        return color, linestyle
    elif ndim == 2:
        norm = None
        if component in ("real", "img", "phase"):
            use_custom_colormap = True
        else:
            use_custom_colormap = False

        if use_custom_colormap:
            vmax = np.max(np.abs(data))
            vmin = -vmax
            vcenter = 0
            colormap = plt.get_cmap("RdBu_r")
            if vmin < vcenter < vmax:
                norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
            else:
                print(
                    f"Warning: Cannot use TwoSlopeNorm with vmin={vmin}, vcenter={vcenter}, vmax={vmax}. Using default normalization."
                )
        else:
            # standard colormap
            if domain == "time":
                colormap = "viridis"
            elif domain == "freq":
                colormap = "plasma"
            else:
                colormap = "viridis"  # default
        return colormap, norm
    else:
        raise ValueError("ndim must be 1 or 2")


def _component_data(data: np.ndarray, component: str) -> Tuple[np.ndarray, str]:
    """Return transformed data + base title according to component."""
    if component not in ("real", "img", "abs", "phase"):
        raise ValueError("Invalid component.")
    if component == "real":
        return np.real(data), r"$\text{Real }$"
    if component == "img":
        return np.imag(data), r"$\text{Imag }$"
    if component == "abs":
        return np.abs(data), r"$\text{Abs }$"
    return np.angle(data), r"$\text{Phase }$"  # phase


def _domain_labels(domain: str, ndim: int) -> Tuple[str, ...]:
    """Return labels for domain and dimension.

    For 1D: (x_label, y_label, title_suffix)
    For 2D: (x_label, y_label, colorbar_label, title_suffix)
    """
    if domain not in ("time", "freq"):
        raise ValueError("Invalid domain. Use 'time' or 'freq'.")
    signal_label = r"$E_{k_S}$"
    title_suffix = "Time domain signal" if domain == "time" else "Spectrum"
    if ndim == 1:
        if domain == "time":
            return r"$t_{\text{det}}$ [fs]", signal_label, title_suffix
        else:
            return r"$\omega_{\text{det}}$ [$10^4$ cm$^{-1}$]", signal_label, title_suffix
    elif ndim == 2:
        if domain == "time":
            return r"$t_{\text{coh}}$ [fs]", r"$t_{\text{det}}$ [fs]", signal_label, title_suffix
        else:
            return (
                r"$\omega_{\text{coh}}$ [$10^4$ cm$^{-1}$]",
                r"$\omega_{\text{det}}$ [$10^4$ cm$^{-1}$]",
                signal_label,
                title_suffix,
            )
    else:
        raise ValueError("ndim must be 1 or 2")


def add_custom_contour_lines(
    x: np.ndarray,
    y: np.ndarray,
    data: np.ndarray,
    component: str,
    level_count: int = 10,
) -> None:
    """
    Add custom contour lines to a 2D plot with different styles for positive/negative values.

    Parameters:
        x (np.ndarray): X-axis coordinate array
        y (np.ndarray): Y-axis coordinate array
        data (np.ndarray): 2D data array to contour
        component (str): Data component type ("real", "img", "phase", "abs")
        level_count (int): Number of contour levels in each region (positive/negative).
            If 10 (default), levels are placed at ±[5, 15, ..., 95]% of |max(data)|.
    """
    ### Add contour lines with different styles for positive and negative values
    if component in ("real", "img", "phase"):
        ### Determine contour levels based on the data range
        vmax = max(abs(np.min(data)), abs(np.max(data)))
        vmin = -vmax

        ### Create evenly spaced levels for both positive and negative regions
        if vmax > 0:
            # Use exact 10% steps by default: 5%, 15%, ..., 95%
            if level_count == 10:
                percents = np.arange(0.05, 1.0, 0.10)
            else:
                # Fallback: linearly spaced in [5%, 95%]
                percents = np.linspace(0.05, 0.95, level_count)

            # Matplotlib requires levels strictly increasing
            positive_levels = percents * vmax  # [0.05..0.95]*vmax ascending
            negative_levels = -percents[::-1] * vmax  # [-0.95..-0.05]*vmax ascending

            ### Plot positive contours (solid lines)
            pos_contour = plt.contour(
                x,
                y,
                data,
                levels=positive_levels,
                colors=COLORS[0],
                linewidths=0.7,
                alpha=0.8,
            )

            ### Plot negative contours (dashed lines)
            neg_contour = plt.contour(
                x,
                y,
                data,
                levels=negative_levels,
                colors=COLORS[1],
                linewidths=0.7,
                alpha=0.8,
                linestyles=LINE_STYLES[1],
            )

            ### NOTE optional: Add contour labels to every other contour line
            # plt.clabel(pos_contour, inline=True, fontsize=8, fmt='%.2f', levels=positive_levels[::2])
            # plt.clabel(neg_contour, inline=True, fontsize=8, fmt='%.2f', levels=negative_levels[::2])
    else:
        ### For abs and phase, use standard contours
        contour_plot = plt.contour(
            x,
            y,
            data,
            levels=level_count,
            colors=COLORS[1],
            linewidths=0.7,
            alpha=0.8,
        )
        ### Optional: Add contour labels
        plt.clabel(
            contour_plot,
            inline=True,
            fontsize=8,
            fmt="%.2f",
            levels=contour_plot.levels[::2],
        )


def add_text_box(
    ax,
    kwargs: dict,
    position: tuple = (0.98, 0.98),
    fontsize: int = 7,
    coords: Literal["axes", "figure"] = "axes",
):
    """
    Add a text box with additional parameters without affecting subplot layout.

    Parameters:
        ax (matplotlib.axes.Axes): Axes object (also used to access the parent Figure).
        kwargs (dict): Dictionary of parameters to display in the text box.
        position (tuple): Position of the text box in the chosen coordinate system.
                          For ``coords='axes'`` this is in axis coordinates (default top-right).
                          For ``coords='figure'`` this is in figure coordinates.
        fontsize (int): Font size for the text box (default: 11).
        coords (Literal["axes","figure"]): Coordinate system to place the text in.
            - "axes": anchored to the given Axes (default; backwards compatible)
            - "figure": anchored to the whole Figure (fully independent of Axes)

    Notes:
        - The created text artist is marked ``in_layout=False`` so that ``tight_layout``
          or ``constrained_layout`` ignore it and do not shrink other subplots.
        - ``clip_on=False`` is used so the text can extend beyond the Axes box if desired.
    """
    if kwargs:
        text_lines = []
        for key, value in kwargs.items():
            if isinstance(value, float):
                # Format floats to 3 significant digits
                text_lines.append(f"{key}: {value:.3g}")
            elif isinstance(value, (int, str)):
                # Add integers and strings directly
                text_lines.append(f"{key}: {value}")
            elif isinstance(value, np.ndarray):
                # Show shape for numpy arrays
                text_lines.append(f"{key}: array(shape={value.shape})")
            else:
                # TODO insert this into my plotstyle package
                # Convert other types to string and escape LaTeX special characters
                safe_str = (
                    str(value)
                    .replace("_", "\_")
                    .replace("^", "\^")
                    .replace("{", "\{")
                    .replace("}", "\}")
                )
                text_lines.append(f"{key}: {safe_str}")

        info_text = "\n".join(text_lines)

        # Create the text artist either in axes or figure coordinates
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
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.3", alpha=0.05, edgecolor="black"),
                clip_on=False,  # allow text to extend beyond axes region
            )

        # Ensure layout engines ignore this artist so it doesn't shrink subplots
        try:
            artist.set_in_layout(False)
        except Exception:
            # Older Matplotlib versions may not have in_layout on Text; ignore gracefully
            pass
