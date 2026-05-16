# Plotstyle Package

`plotstyle` is the plotting companion package for the thesis workflow. It provides a small set of Matplotlib helpers for LaTeX-friendly figures, consistent sizing, and reproducible export settings.

## Installation

From the repository root:

```bash
pip install -e ./packages/plotstyle
```

The package is also installed automatically when you create the root conda environment with `environment.yml`.

## Public API

The package exports the following helpers from `plotstyle`:

- `init_style`
- `save_fig`
- `set_size`
- `format_sci_notation`
- `simplify_figure_text`
- `beautify_colorbar`
- `apply_decimal_axis_ticks`
- `apply_decimal_colorbar_ticks`
- `latex_available`

It also exports the common style constants and theme objects:

- `COLORS`
- `LINE_STYLES`
- `MARKERS`
- `LATEX_DOC_WIDTH`
- `LATEX_FONT_SIZE`
- `FONT_SIZE`
- `FIG_SIZE`
- `LINE_WIDTH`
- `DPI`
- `FIG_FORMAT`
- `TRANSPARENCY`
- `DEFAULT_THEME`
- `PlotTheme`

## Quick start

```python
import numpy as np
import matplotlib.pyplot as plt

from plotstyle import init_style, save_fig, set_size

init_style()

fig, ax = plt.subplots(figsize=set_size(fraction=0.8))

x_vals = np.linspace(0.0, 10.0, 300)
ax.plot(x_vals, np.sin(x_vals), label=r"$\sin(x)$", linestyle="solid")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.legend()

save_fig(fig, "./figures/sine_wave", formats=["pdf", "png"], dpi=300)
```

## Behavior

- If a `latex` binary is available, `init_style()` enables TeX rendering.
- If LaTeX is not available, the package falls back to Matplotlib math text.
- On headless machines, the backend selection falls back to a non-interactive backend.
- `save_fig()` writes all requested output formats and creates missing directories automatically.

## Notebook example

The committed notebook `test_TeX_plots.ipynb` demonstrates TeX rendering, figure sizing, color and marker cycling, and multi-format saving. Adjust the output paths inside the notebook before running it.
