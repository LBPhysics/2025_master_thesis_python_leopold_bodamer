"""Shared plotting settings for local and HPC plotting scripts."""

from __future__ import annotations

SECTION: tuple[tuple[float, float], tuple[float, float]] | None = (
    (1.5, 1.7),  # coherence axis
    (1.5, 1.7),  # detection axis
)
PAD_FACTOR: float = 50.0
APODIZATION_WINDOW: str | None = None
"""
Set to ``None`` to disable apodization, or choose one of:
    - "hann"
    - "hamming"
    - "blackman"

The same window is applied along every transformed axis:
    - 1D: detection axis
    - 2D: coherence and detection axes
"""
CUTOFF_PERCENT: float = 0.0
CONTOUR_LINES: bool = False
TRANSPARENTCY: bool = False
FIG_FORMATS: list[str] = ["svg", "png"]  # Save both high-quality SVG and low-cost PNG
# Order defines the left-to-right panel order in the composite figures.
# Allowed values for plot_datas.py are: "real", "imag", "abs".
COMPONENTS: list[str] = ["real", "imag"]#, "abs"]
__all__ = [
    "SECTION",
    "PAD_FACTOR",
    "APODIZATION_WINDOW",
    "CUTOFF_PERCENT",
    "CONTOUR_LINES",
    "TRANSPARENTCY",
    "FIG_FORMATS",
    "COMPONENTS",
    "NORMALISE_TIME_DOMAIN",
    "NORMALISE_FREQUENCY_DOMAIN",
    "TIME_NORM_SCOPE",
    "FREQ_NORM_SCOPE",
]


# -------------------------------------------------------------------------
# Normalisation settings
# -------------------------------------------------------------------------
# If True, divide every plotted component by one common factor:
#       max(abs(complex_data))
# so that |signal| peaks at 1.
NORMALISE_TIME_DOMAIN = True
NORMALISE_FREQUENCY_DOMAIN = True

# "all_signals"  -> one common factor across all signals in that domain
# "per_signal"   -> one common factor per signal_type
TIME_NORM_SCOPE = "all_signals"
FREQ_NORM_SCOPE = "all_signals" # OR per_signal         FREQ_NORM_SCOPE = "all_signals" -> renders rephasing be be much weaker than non-rephasing,
