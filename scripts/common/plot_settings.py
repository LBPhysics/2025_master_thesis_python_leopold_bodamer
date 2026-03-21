"""Shared plotting settings for local and HPC plotting scripts."""

from __future__ import annotations

SECTION: tuple[float, float] | None = (1.5, 1.7)
PLOT_PAD_FACTOR: float = 20.0

__all__ = ["SECTION", "PLOT_PAD_FACTOR"]
