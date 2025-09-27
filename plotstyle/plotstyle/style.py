"""Matplotlib styling (standalone implementation).

Import from ``plotstyle``::

        from plotstyle import init_style, save_fig, set_size

Design goals:
        * Idempotent style initialization (safe in parallel workers)
        * Zero side effects until ``init_style()`` is called
        * Minimal, readable set of defaults suitable for thesis figures
"""

from __future__ import annotations
import os
import sys
import shutil
import math
from pathlib import Path
from typing import Optional, Iterable, Sequence, Union, List, Tuple
import matplotlib as mpl
from cycler import cycler
from .constants import (
    LATEX_DOC_WIDTH,
    FONT_SIZE,
    FIG_SIZE,
    DPI,
    FIG_FORMAT,
    TRANSPARENCY,
    COLORS,
    LINE_STYLES,
    MARKERS,
)

__all__ = [
    # Public constants / defaults
    "COLORS",
    "LINE_STYLES",
    "MARKERS",
    # public functions
    "init_style",
    "set_size",
    "save_fig",
    "format_sci_notation",
    "simplify_figure_text",
    "beautify_colorbar",
]

latex_available = False
_LATEX_PROBE_DONE = False


# TODO only init_style and save_fig and set_size should be public
def init_style(quiet: bool = True) -> None:
    """Initialize matplotlib rcParams once (idempotent) for thesis-quality plots."""
    _setup_backend()
    base_settings = {
        "font.family": "serif",
        "font.serif": ["cmu serif", "times new roman", "serif"],
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE + 2,
        "axes.labelsize": FONT_SIZE + 2,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "figure.figsize": FIG_SIZE,
        "figure.autolayout": True,
        "axes.grid": False,
        "axes.axisbelow": True,
        # Cartesian product for more unique combinations before repeating
        "axes.prop_cycle": cycler("color", COLORS) * cycler("linestyle", LINE_STYLES),
        "legend.frameon": True,
        "legend.fancybox": True,
        "legend.framealpha": 0.8,
        "savefig.transparent": TRANSPARENCY,
        "savefig.format": FIG_FORMAT,
        "savefig.dpi": DPI,
        "mathtext.default": "regular",
    }

    _ensure_latex_probe()
    if latex_available:
        latex_settings = {
            "text.usetex": True,
            # Include AMS packages for \mathbb, \mathfrak, symbols, and bold math; Palatino text
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{amsfonts}\usepackage{bm}",  # TODO update here your desired LaTeX preamble \usepackage{mathpazo}
        }
        base_settings.update(latex_settings)
    else:
        non_latex_settings = {
            "text.usetex": False,
            "mathtext.default": "regular",
        }
        base_settings.update(non_latex_settings)
    mpl.rcParams.update(base_settings)
    if not quiet:
        print("[plotstyle.init_style] matplotlib style initialized")
        print("latex used:", latex_available, "and backend used:", mpl.get_backend())


def save_fig(
    fig: mpl.figure.Figure,
    filename: Union[str, os.PathLike],
    formats: Optional[Sequence[str]] = None,
    dpi: int = DPI,
    transparent: bool = TRANSPARENCY,
    figsize: Optional[Tuple[float, float]] = None,
) -> List[Path]:
    """Save figure to one or multiple formats, creating directories as needed.

    Behavior:
    - If ``formats`` is None and ``filename`` has a suffix, save only that format.
    - If ``formats`` is None and no suffix, save using default ``FIG_FORMAT``.
    - If ``formats`` is provided, save one file per format, replacing any suffix.
    Returns a list of saved file paths.
    """
    path = Path(filename)
    saved: List[Path] = []

    # Ensure parent directory exists
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)

    if figsize is not None:
        fig.set_size_inches(figsize)

    # Sanitize figure text for LaTeX fallback (kept for backwards-compat)
    simplify_figure_text(fig)

    # Determine formats and targets
    # Only treat a suffix as a format if Matplotlib supports it
    try:
        supported_formats = {k.lower() for k in fig.canvas.get_supported_filetypes().keys()}
    except Exception:
        # Reasonable fallback set
        supported_formats = {
            "jpg",
            "pdf",
            "png",
            "svg",
        }

    def _append_ext(p: Path, ext: str) -> Path:
        return Path(str(p) + ("." + ext.lstrip(".")))

    ext = path.suffix.lower().lstrip(".")
    has_supported_ext = bool(ext) and (ext in supported_formats)

    if formats is None:
        if has_supported_ext:
            targets = [(path, ext)]
        else:
            # Keep the current name intact and append the default extension
            targets = [(_append_ext(path, FIG_FORMAT), FIG_FORMAT)]
    else:
        # Normalize and filter requested formats by supported set
        targets = []
        # Use a base path: if there's a supported ext, drop it; otherwise keep as-is
        base_path = path.with_suffix("") if has_supported_ext else path
        for fmt in formats:
            f = str(fmt).lower().lstrip(".")
            if f in supported_formats:
                targets.append((_append_ext(base_path, f), f))
        if not targets:
            # Fallback if all requested formats were invalid
            targets = [(_append_ext(base_path, FIG_FORMAT), FIG_FORMAT)]

    for out_path, fmt in targets:
        # Try saving with tight bounding box; if Matplotlib warns that tight layout
        # cannot be applied (common with colorbars/3D axes), retry without it.
        import warnings as _warnings

        with _warnings.catch_warnings(record=True) as _w:
            _warnings.simplefilter("always")
            fig.savefig(
                str(out_path),
                format=fmt,
                dpi=dpi,
                bbox_inches="tight",
                transparent=transparent,
            )
            _tight_failed = any("Tight layout not applied" in str(msg.message) for msg in _w)
        if _tight_failed:
            fig.savefig(
                str(out_path),
                format=fmt,
                dpi=dpi,
                bbox_inches=None,
                transparent=transparent,
            )
        saved.append(out_path)
    return saved


def set_size(
    width_pt: float = LATEX_DOC_WIDTH,
    fraction: float = 0.5,
    subplots: Tuple[int, int] = (1, 1),
    height_ratio: Optional[float] = None,
) -> Tuple[float, float]:
    """Compute (width, height) in inches for a figure scaled to LaTeX width.

    Parameters
    ----------
    width_pt : float
            Full width of the LaTeX text block in points.
    fraction : float
            Fraction of the width to occupy (0 < fraction <= 1).
    subplots : tuple
            (n_rows, n_cols) to scale height for multi-panel figures.
    height_ratio : float | None
            Optional manual golden-ratio-like modification; default uses golden ratio.
    """
    if not (0 < fraction <= 1):
        raise ValueError("fraction must be in the interval (0, 1]")
    if not (
        isinstance(subplots, tuple) and len(subplots) == 2 and subplots[0] >= 1 and subplots[1] >= 1
    ):
        raise ValueError("subplots must be a tuple of positive integers (n_rows, n_cols)")
    fig_width_pt = width_pt * fraction
    inches_per_pt = 1 / 72.27
    if height_ratio is None:
        height_ratio = (5**0.5 - 1) / 2  # golden ratio approximation
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * height_ratio * (subplots[0] / subplots[1])
    # Round a bit to avoid overly precise floats in metadata
    return (round(fig_width_in, 4), round(fig_height_in, 4))


def format_sci_notation(x: float, decimals: int = 1, include_dollar: bool = True) -> str:
    """Format a number into scientific notation suitable for axis labels.

    Uses LaTeX multiplication symbol when text.usetex is enabled, otherwise a dot.
    """
    if x == 0:
        return r"$0$" if include_dollar else "0"
    exp = int(math.floor(math.log10(abs(x))))
    coef = round(x / (10**exp), decimals)
    # MathText supports \times regardless of text.usetex
    mult_symbol = r" \times "
    if coef == 1:
        result = f"10^{{{exp}}}"
    else:
        result = f"{coef}{mult_symbol}10^{{{exp}}}"
    return f"${result}$" if include_dollar else result


def simplify_figure_text(
    fig: mpl.figure.Figure,
    force_sci: bool = False,
    sci_decimals: int = 1,
) -> mpl.figure.Figure:
    """Sanitize all text in a figure and beautify numeric ticks."""
    _ensure_latex_probe()

    def _sanitize_textobjs(objs: Iterable[mpl.text.Text]):
        for t in objs:
            try:
                s = t.get_text()
                if s:
                    if latex_available:
                        t.set_text(_escape_latex_text_outside_math(s))
                    else:
                        t.set_text(_strip_latex(s))
            except Exception:
                continue

    def _beautify_linear_axis(ax: mpl.axes.Axes, which: str) -> None:
        # Apply to linear scale only; skip log/symlog etc.
        scale = getattr(ax, f"get_{which}scale")()
        if scale != "linear":
            return
        # Get limits and target axis
        if which == "x":
            vmin, vmax = ax.get_xlim()
            axis = ax.xaxis
        elif which == "y":
            vmin, vmax = ax.get_ylim()
            axis = ax.yaxis
        else:  # 'z' for 3D
            if not hasattr(ax, "get_zlim"):
                return
            vmin, vmax = ax.get_zlim()
            axis = ax.zaxis

        # Skip axes whose ticks are intentionally disabled (e.g., colorbar short axis)
        if which in ("x", "y"):
            try:
                if axis.get_ticks_position() == "none":
                    return
            except Exception:
                pass

        import numpy as _np
        from matplotlib.ticker import MaxNLocator, FuncFormatter, ScalarFormatter

        if not _np.isfinite([vmin, vmax]).all() or vmin == vmax:
            return

        # Prefer integer ticks for small, integer-like ranges
        def _near_integers(a: float, b: float, tol: float = 1e-9) -> bool:
            return abs(a - round(a)) < tol and abs(b - round(b)) < tol

        rng = vmax - vmin
        prefer_integer = False
        if rng <= max(1.0, 10 ** max(-2, math.floor(math.log10(max(1e-12, abs(rng)))))):
            prefer_integer = _near_integers(vmin, vmax)

        axis.set_major_locator(
            MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10], integer=prefer_integer, min_n_ticks=3)
        )

        if force_sci:
            axis.set_major_formatter(
                FuncFormatter(
                    lambda x, _pos=None: format_sci_notation(x, decimals=max(0, sci_decimals))
                )
            )
        else:
            fmt = ScalarFormatter(useOffset=False, useMathText=True)
            fmt.set_powerlimits((-3, 3))
            axis.set_major_formatter(fmt)

    try:
        # Suptitle
        if getattr(fig, "_suptitle", None) is not None:
            st = fig._suptitle
            _sanitize_textobjs([st])

        # Axes content
        for ax in fig.get_axes():
            # Sanitize titles and labels (do NOT touch x-axis label)
            _sanitize_textobjs([ax.title, ax.yaxis.label])
            if hasattr(ax, "zaxis"):
                _sanitize_textobjs([ax.zaxis.label])

            # Sanitize ticklabels (do NOT touch x ticklabels)
            try:
                if ax.yaxis.get_ticks_position() != "none":
                    _sanitize_textobjs(ax.get_yticklabels())
            except Exception:
                pass
            if hasattr(ax, "zaxis"):
                _sanitize_textobjs(ax.get_zticklabels())

            # Beautify numeric ticks (do NOT touch x-axis)
            try:
                if ax.yaxis.get_ticks_position() != "none":
                    _beautify_linear_axis(ax, "y")
            except Exception:
                pass
            if hasattr(ax, "zaxis"):
                _beautify_linear_axis(ax, "z")

            # Legend
            leg = ax.get_legend()
            if leg is not None:
                _sanitize_textobjs(leg.get_texts())
    except Exception:
        pass
    return fig


def beautify_colorbar(
    cb: "mpl.colorbar.Colorbar",
    max_ticks: int = 5,
    sigfigs: int = 2,
    sci_limits: Tuple[int, int] = (-3, 3),
    integer_if_close: bool = True,
    force_sci: bool = False,
) -> "mpl.colorbar.Colorbar":
    """Format colorbar ticks with rounded numbers/scientific notation dynamically.

    Parameters
    ----------
    cb : matplotlib.colorbar.Colorbar
        Colorbar instance returned by ``plt.colorbar(...)``.
    max_ticks : int
        Max number of ticks to request.
    sigfigs : int
        Significant figures to show when using decimal or scientific formatting.
    sci_limits : tuple[int,int]
        Use scientific notation when the order of magnitude is outside this range.
        E.g., (-3, 3) means use decimal for 1e-3..1e3, scientific otherwise.
    integer_if_close : bool
        If the data range is near-integer values and small, prefer integer ticks.

    Returns
    -------
    cb : matplotlib.colorbar.Colorbar
        The same colorbar, for chaining.
    """
    import numpy as _np
    from matplotlib.ticker import MaxNLocator, FuncFormatter, ScalarFormatter

    vmin, vmax = cb.mappable.get_clim()
    if not _np.isfinite([vmin, vmax]).all():
        return cb

    # Prefer integer ticks when range and values look integer-like
    def _near_integers(a: float, b: float, tol: float = 1e-9) -> bool:
        return abs(a - round(a)) < tol and abs(b - round(b)) < tol

    prefer_integer = False
    if integer_if_close:
        rng = vmax - vmin
        if rng <= max(1.0, 10 ** max(-2, math.floor(math.log10(max(1e-12, abs(rng)))))):
            prefer_integer = _near_integers(vmin, vmax)

    # Locator & formatter
    cb.locator = MaxNLocator(
        nbins=max(2, max_ticks),
        steps=[1, 2, 2.5, 5, 10],
        integer=prefer_integer,
        min_n_ticks=3,
    )

    if force_sci:
        cb.formatter = FuncFormatter(
            lambda x, _pos=None: format_sci_notation(x, decimals=max(0, sigfigs - 1))
        )
    else:
        fmt = ScalarFormatter(useMathText=True, useOffset=False)
        fmt.set_powerlimits(sci_limits)
        cb.formatter = fmt
    cb.update_ticks()
    return cb


def _check_latex_available():
    """
    Check if LaTeX (pdflatex or latex) is installed and available in the system path.

    Returns
    -------
    bool
        True if LaTeX is available, False otherwise.
    """
    latex_commands = ["pdflatex", "latex", "latexmk"]
    return any(shutil.which(cmd) is not None for cmd in latex_commands)


def _ensure_latex_probe() -> None:
    """Probe for LaTeX availability once and cache the result."""
    global latex_available, _LATEX_PROBE_DONE
    if _LATEX_PROBE_DONE:
        return
    latex_available = _check_latex_available()
    _LATEX_PROBE_DONE = True


def _strip_latex(s: str) -> str:
    """Convert a LaTeX/mathtext label to a plain-text safe fallback.

    Notes
    -----
    This is a best-effort sanitizer aiming to avoid LaTeX-specific commands that
    would error in non-TeX environments. It preserves basic intent (subscripts
    via underscores, powers via caret) and removes formatting commands.
    """
    if not s:
        return s
    # Remove math delimiters
    out = s.replace("$", "")
    # Common spacing/formatting
    for tok in ("\,", "\ ", "\;", "\:", "\!", "\quad", "\qquad"):
        out = out.replace(tok, " ")
    for tok in ("\left", "\right"):
        out = out.replace(tok, "")
    # Text/roman wrappers
    import re as _re

    def _unbrace(cmd: str, text: str) -> str:
        # Match LaTeX-like wrappers such as \text{...}, \mathrm{...}, etc.,
        # without using f-strings to avoid brace-escaping issues.
        pattern = _re.compile(r"\\" + cmd + r"\{([^}]*)\}")
        return _re.sub(pattern, r"\1", text)

    for cmd in ("text", "mathrm", "mathbf", "mathit", "mathcal", "mathbb", "mathfrak"):
        out = _unbrace(cmd, out)
    # Greek symbols and a few common macros
    replacements = {
        r"\omega": "omega",
        r"\Omega": "Omega",
        r"\phi": "phi",
        r"\varphi": "phi",
        r"\theta": "theta",
        r"\Theta": "Theta",
        r"\mu": "mu",
        r"\varepsilon": "eps",
        r"\epsilon": "eps",
        r"\pi": "pi",
        r"\cdot": "*",
        r"\times": "x",
        r"\propto": "~",
        r"\infty": "inf",
        r"\langle": "<",
        r"\rangle": ">",
        r"\vec": "",  # drop arrow accent
        r"\hat": "",  # drop hat accent
        r"\bar": "",  # drop bar accent
    }
    for k, v in replacements.items():
        out = out.replace(k, v)
    # Remove braces but keep structure like E_{out} -> E_out; 10^{4} -> 10^4
    out = out.replace("{", "").replace("}", "")
    # Collapse multiple spaces
    out = _re.sub(r"\s+", " ", out).strip()
    return out


def _escape_latex_text_outside_math(s: str) -> str:
    """Escape LaTeX special chars in plain text segments (outside math $...$).

    Escapes: & % # _
    Leaves math segments (between $) unchanged to avoid breaking equations.
    """
    if not s or "$" not in s:
        # No math segments; escape globally
        return (
            s.replace("\\", "\\\\")
            .replace("&", r"\&")
            .replace("%", r"\%")
            .replace("#", r"\#")
            .replace("_", r"\_")
        )

    parts = s.split("$")
    out_parts: List[str] = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            # Text segment (outside math): escape
            escaped = (
                part.replace("\\", "\\\\")
                .replace("&", r"\&")
                .replace("%", r"\%")
                .replace("#", r"\#")
                .replace("_", r"\_")
            )
            out_parts.append(escaped)
        else:
            # Math segment: keep as-is
            out_parts.append(part)
    return "$".join(out_parts)


def _setup_backend() -> None:
    """Select a safe backend for HPC/headless without forcing GUI backends.

    - In Jupyter/IPython kernels, use the inline backend.
    - If a backend is already chosen via MPLBACKEND, respect it.
    - If likely headless on Unix (no DISPLAY), use Agg.
    - Otherwise, leave Matplotlib's default backend.
    """
    try:
        # Respect explicit environment setting
        if os.environ.get("MPLBACKEND"):
            return

        # Jupyter/IPython inline context
        if "ipykernel" in sys.modules:
            inline_name = "module://matplotlib_inline.backend_inline"

            if mpl.get_backend().lower() != inline_name:
                mpl.use(inline_name)
            return

        # Headless Unix: no DISPLAY -> Agg
        if os.name != "nt" and not os.environ.get("DISPLAY"):
            if mpl.get_backend().lower() != "agg":
                mpl.use("Agg")
            return
        # On Windows or when DISPLAY exists, keep default backend
    except Exception:
        # As a last resort, fall back to Agg
        try:
            mpl.use("Agg")
        except Exception:
            pass
