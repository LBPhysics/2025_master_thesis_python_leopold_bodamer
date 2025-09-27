"""Standalone plotting style package for good Matplotlib to LaTeX integration.

Usage:
    from plotstyle import init_style, save_fig, set_size
"""

from .style import (
    init_style,
    save_fig,
    set_size,
    format_sci_notation,
    simplify_figure_text,
    beautify_colorbar,
    latex_available,
)
from .constants import (
    COLORS,
    LINE_STYLES,
    MARKERS,
    LATEX_DOC_WIDTH,
    LATEX_FONT_SIZE,
    FONT_SIZE,
    FIG_SIZE,
    DPI,
    FIG_FORMAT,
    TRANSPARENCY,
)

__all__ = [
    "init_style",
    "save_fig",
    "set_size",
    "format_sci_notation",
    "simplify_figure_text",
    "beautify_colorbar",
    "COLORS",
    "LINE_STYLES",
    "MARKERS",
    "latex_available",
    "LATEX_DOC_WIDTH",
    "LATEX_FONT_SIZE",
    "FONT_SIZE",
    "FIG_SIZE",
    "DPI",
    "FIG_FORMAT",
    "TRANSPARENCY",
]
