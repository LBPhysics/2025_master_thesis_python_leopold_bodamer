"""Public constants and palettes for plotstyle.

Kept separate from matplotlib-dependent code to allow lightweight imports.
"""

from __future__ import annotations
from typing import Tuple, List

__all__ = [
    "LATEX_DOC_WIDTH",
    "LATEX_FONT_SIZE",
    "FONT_SIZE",
    "FIG_SIZE",
    "DPI",
    "FIG_FORMAT",
    "TRANSPARENCY",
    "COLORS",
    "LINE_STYLES",
    "MARKERS",
]
# Document layout related defaults
LATEX_DOC_WIDTH: float = 441.01775  # pt width of LaTeX document text block
LATEX_FONT_SIZE: int = 11
FONT_SIZE: int = 11
FIG_SIZE: Tuple[float, float] = (8, 6)
DPI: int = 300
FIG_FORMAT: str = "png"  # change to "svg" if desired
TRANSPARENCY: bool = True

# Palettes
COLORS: List[str] = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
LINE_STYLES = [
    "solid",
    "dashed",
    "dashdot",
    "dotted",
    (0, (3, 1, 1, 1)),
    (0, (5, 1)),
]
MARKERS = ["o", "s", "^", "v", "D", "p", "*", "X", "+", "x"]
