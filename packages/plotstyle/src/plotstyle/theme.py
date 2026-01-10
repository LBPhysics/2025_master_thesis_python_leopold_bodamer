"""Reusable Matplotlib theme helpers for the thesis plot style package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

from cycler import cycler

from .constants import (
    COLORS,
    DPI,
    FIG_FORMAT,
    FIG_SIZE,
    FONT_SIZE,
    LATEX_DOC_WIDTH,
    LINE_WIDTH,
    LINE_STYLES,
    TRANSPARENCY,
)


@dataclass(frozen=True)
class PlotTheme:
    """Description of a Matplotlib theme used across the thesis project.

    Parameters
    ----------
    name:
        Human readable name used in debug prints/logging.
    serif_fonts:
        Ordered tuple of serif fonts to try when LaTeX rendering is available.
    sans_fonts:
        Optional fallback sans-serif fonts when LaTeX is not available.
    latex_preamble:
        Additional LaTeX packages inserted into the document preamble.
    legend_alpha:
        Default legend frame alpha channel (between 0 and 1).
    """

    name: str = "thesis-default"
    serif_fonts: Tuple[str, ...] = ("cmu serif", "times new roman", "serif")
    sans_fonts: Tuple[str, ...] = ("cmu bright", "dejavu sans", "sans-serif")
    latex_preamble: str = (
        r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{amsfonts}\usepackage{bm}"
    )
    legend_alpha: float = 0.8

    def prop_cycle(self) -> Iterable:
        """Return a combined color/linestyle cycler."""

        return cycler("color", COLORS) * cycler("linestyle", LINE_STYLES)

    def build_rcparams(self, *, latex_enabled: bool) -> Dict[str, object]:
        """Create the rcParams dictionary for Matplotlib."""

        base: Dict[str, object] = {
            "font.family": "serif" if latex_enabled else "sans-serif",
            "font.serif": list(self.serif_fonts),
            "font.sans-serif": list(self.sans_fonts),
            "font.size": FONT_SIZE,
            "lines.linewidth": LINE_WIDTH,
            "axes.titlesize": FONT_SIZE + 2,
            "axes.labelsize": FONT_SIZE + 2,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "figure.figsize": FIG_SIZE,
            "figure.autolayout": True,
            "axes.grid": False,
            "axes.axisbelow": True,
            "axes.prop_cycle": self.prop_cycle(),
            "legend.frameon": True,
            "legend.fancybox": True,
            "legend.framealpha": self.legend_alpha,
            "savefig.transparent": TRANSPARENCY,
            "savefig.format": FIG_FORMAT,
            "savefig.dpi": DPI,
            "mathtext.default": "regular",
        }

        if latex_enabled:
            base.update(
                {
                    "text.usetex": True,
                    "text.latex.preamble": self.latex_preamble,
                }
            )
        else:
            base.update(
                {
                    "text.usetex": False,
                    "mathtext.default": "regular",
                }
            )
        return base

    def figure_size(
        self, *, fraction: float = 0.5, height_ratio: float | None = None
    ) -> Tuple[float, float]:
        """Helper mirroring :func:`plotstyle.style.set_size` defaults."""

        from .style import set_size  # Local import to avoid cycles

        return set_size(
            width_pt=LATEX_DOC_WIDTH,
            fraction=fraction,
            height_ratio=height_ratio,
        )


DEFAULT_THEME = PlotTheme()

__all__ = ["PlotTheme", "DEFAULT_THEME"]
