# Plotstyle Package

A standalone Python package for professional, LaTeX-compatible matplotlib plotting styles, designed for academic and scientific publications.

## Table of contents
- [Overview](#overview)
- [Quick start](#quick-start)
- [Usage](#usage)
- [API reference](#api-reference)
- [Configuration & environment detection](#configuration--environment-detection)
- [Examples & notebooks](#examples--notebooks)

## Overview
`plotstyle` centralizes the figure aesthetics used in the Master’s thesis project. Calling `init_style()` once configures Matplotlib to:

1. Use LaTeX (when available) for crisp math and serifs.
2. Apply curated colors, line styles, and markers optimized for scientific plots.
3. Choose fonts and backends appropriate for Jupyter, scripts, or headless clusters.
4. Provide helpers for consistent figure sizing, scientific notation, and multi-format saving.


## Quick start
```bash
pip install -e .
```

```python
import numpy as np
import matplotlib.pyplot as plt

from plotstyle import init_style, save_fig, set_size

# Configure Matplotlib once per process / session
init_style()

fig, ax = plt.subplots(figsize=set_size(fraction=0.8))

x_vals = np.linspace(0.0, 10.0, 300)
ax.plot(x_vals, np.sin(x_vals), label=r"$\sin(x)$", linestyle="solid")

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_title(r"Sine wave with LaTeX-rendered labels")
ax.legend()

save_fig(fig, "./figures/sine_wave", formats=["pdf", "png"], dpi=300)
```

## Usage
```python
from plotstyle import init_style, save_fig

init_style(quiet=False)  # show detection summary (TeX, fonts, backend)
# ... generate plots ...
save_fig(fig, "./plots/run_001", formats=["pdf", "svg"])
```



## API reference

### Public functions
- `init_style(quiet: bool = True) -> None`: Configure Matplotlib rcParams, fonts, backend, and color cycles.
- `set_size(width_pt: float | None = None, fraction: float = 0.5, subplots: tuple[int, int] = (1, 1)) -> tuple[float, float]`: Compute figure size in inches based on LaTeX text width.
- `save_fig(fig, filename: str, formats: list[str] | None = None, dpi: int = 150, transparent: bool = True) -> None`: Persist a figure to disk, expanding directories and writing multiple formats.
- `format_sci_notation(value: float, decimals: int = 1, include_dollar: bool = True) -> str`: Produce a LaTeX-friendly scientific notation string.

### Constants
- `COLORS`: Thesis palette ordered for multipanel plots.
- `LINE_STYLES`: Cycle of unique line styles (`solid`, `dashed`, `dashdot`, `dotted`, ...).
- `MARKERS`: Marker cycle suited for low-ink plots.
- `FONT_SIZE`: Default font size (11 pt).
- `LATEX_DOC_WIDTH`: Default document width (points) used by `set_size` when `width_pt` is omitted.

## Configuration & environment detection
- LaTeX detection: If `latex` binary is found, `text.usetex=True` with AMS packages; otherwise the style falls back to Matplotlib’s MathText while preserving typography.
- Backend selection: Prefers interactive backends in notebooks, non-interactive (`Agg`) on headless or HPC nodes.
- Font fallback order: Palatino → CMU Serif → Times New Roman → Matplotlib defaults.
- Logging: Pass `quiet=False` to `init_style` to print a capability summary.

## Examples & notebooks
- `test_TeX_plots.ipynb`: Demonstrates LaTeX rendering, sizing heuristics, color/marker cycling, and saving to multiple formats. Update `FIGURES_TESTS_DIR` inside the notebook to your own output directory.
- Example scripts can be added under `examples/` (contributions welcome).
