# Thesis spectroscopy toolkit

This repository contains the Python stack that powers the spectroscopy part (and overall all the figures) of my master thesis.  The tree now follows a `packages/`-first layout so that the reusable libraries (`plotstyle` and `qspectro2d`) can be installed in editable mode while the CLI scripts remain thin orchestration layers. These scripts can be run locally for small test jobs or on HPC clusters for large parameter sweeps and generate the spectrocopic data and figures

My latex thesis can be found in the dedicated repository:
https://github.com/LBPhysics/2025_master_thesis_latex_leopold_bodamer.git

## Repository layout

```
thesis_python/
├─ packages/
│  ├─ plotstyle/        # LaTeX friendly Matplotlib helpers
│  └─ qspectro2d/       # Spectroscopy simulations (systems, baths, pulses, solvers)
├─ scripts/             # CLI entry points (simulate, stack, plot, HPC workflows)
├─ notebooks/           # Interactive exploration and regression notebooks
├─ environment.yml      # Conda environment used for development and HPC jobs
├─ pyproject.toml       # Shared tooling configuration (ruff/black/pytest)
├─ .vscode/             # Workspace settings, launch configs, tasks
└─ README.md            # You are here
```

Each package in `packages/` is a standalone `pyproject`.  When you create the conda environment the packages are installed in editable mode automatically.
## Environment setup

```bash
git clone https://github.com/LBPhysics/2025_master_thesis_python_leopold_bodamer.git
cd 2025_master_thesis_python_leopold_bodamer
conda env create -f environment.yml
conda activate m_env
```

The environment file already performs the editable installs for `plotstyle` and `qspectro2d`, so no additional pip commands are required after activation.

## Workflow overview

1. **Configure simulation** — duplicate a template in `scripts/simulation_configs/` and adjust physical parameters.  `_monomer.yaml` is still the default that `calc_datas.py` auto-selects.
2. **Simulate** — run `python scripts/calc_datas.py --sim_type {1d,2d}` locally.  For batched/HPC jobs keep using `hpc_calc_datas.py` / `hpc_plot_datas.py` (unchanged).
3. **Aggregate** — combine inhomogeneous traces with `stack_inhomogenity.py` or assemble 2D datasets with `stack_times.py`.
4. **Visualise** — use `plot_datas.py` to create publication-ready figures.  For LaTeX manuscripts the `plotstyle` package enables consistent fonts and line styles.

Simulation outputs remain under `data/` and plots under `figures/figures_from_python/` just like before.

## HPC reminder

```bash
git pull --ff-only
conda activate m_env
conda env update -f environment.yml --prune  # only when dependencies change
python scripts/hpc_calc_datas.py --n_batches N --sim_type 2d
```

Logs coming from batch submissions stay in the same directories referenced by the HPC helper scripts.

## Outstanding physics TODOs

- Reproducing Fig. 2 of https://pubs.aip.org/jcp/article/124/23/234504/930650/ — inhomogeneous broadening still wipes out the rephasing signal after $t_{\text{det}} \approx t_{\text{coh}}$.
- Reproducing Fig. 3 of https://pubs.aip.org/jcp/article/124/23/234505/930637/ — 2D spectral features remain rotated by $90^{\circ}$ compared to the reference.