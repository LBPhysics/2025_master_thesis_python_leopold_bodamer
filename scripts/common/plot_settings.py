"""Shared plotting settings for local and HPC plotting scripts."""

from __future__ import annotations

SECTION: tuple[tuple[float, float], tuple[float, float]] | None = (
    (1.5, 1.7),  # coherence axis
    (1.5, 1.7),  # detection axis
)
PAD_FACTOR: float = 50.0
CUTOFF_PERCENT: float = 0.0
CONTOUR_LINES: bool = False
TRANSPARENTCY: bool = False
FIG_FORMATS: list[str] = ["svg", "png"]  # Save both high-quality SVG and low-cost PNG
__all__ = [
    "SECTION",
    "PAD_FACTOR",
    "CUTOFF_PERCENT",
    "CONTOUR_LINES",
    "TRANSPARENTCY",
    "FIG_FORMATS",
]
