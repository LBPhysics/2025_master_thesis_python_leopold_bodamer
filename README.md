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

There are two main workflows: **local execution** (for small tests) and **HPC batching** (for large parameter sweeps). Both produce the same plots but differ in scaling and automation.

### Local Workflow (Run on Your Machine)
For small-scale runs (e.g., quick tests or limited parameter sweeps). Supports two modes based on simulation type and broadening.

1. **Configure simulation** — Duplicate a template in `scripts/simulation_configs/` and adjust physical parameters. `_monomer.yaml` is the default that `calc_datas.py` auto-selects.

2. **Simulate** — Run `python scripts/calc_datas.py --sim_type {1d,2d}` locally.
   - **1D with inhomogeneous broadening**: Simulates with multiple realizations (samples from Gaussian). Outputs raw `.npz` files per realization.
   - **2D without inhomogeneous broadening**: Simulates homogeneous system across coherence times. Outputs raw `.npz` files per `t_coh` point.

3. **Aggregate** — Combine data for plotting.
   - For 1D inhom: `python scripts/avg_inhomogenity.py` (averages realizations).
   - For 2D hom: `python scripts/stack_times.py` (stacks across `t_coh`).

4. **Visualize** — Run `python scripts/plot_datas.py --abs_path /path/to/artifact.npz` to generate time/frequency-domain plots (e.g., signals, spectra). Options: `--extend` for zero-padding, `--section` for frequency windows, `--no_time`/`--no_freq` to skip domains.

Simulation outputs remain under `data/` and plots under `figures/figures_from_python/`.

### HPC Batching Workflow (Run on a Cluster)
For large-scale runs (e.g., full sweeps with many inhomogeneous samples and coherence times). Supports all combinations (hom/inhom, 1D/2D) with parallel batching. Processing and plotting are combined into one automated step.

1. **Dispatch batches** — Run `python scripts/hpc_batch_dispatch.py --sim_type {1d,2d} --n_batches N [--rng_seed S] [--no_submit]`. Generates SLURM jobs that split work across inhomogeneous samples and coherence times. Validates locally first.

2. **Run batches** — Batches auto-submit via `sbatch` (unless `--no_submit`). Each runs `run_batch.py` on the cluster, producing partial artifacts in `data/...`.

3. **Post-process and plot** — After batches finish, run `python scripts/hpc_plot_datas.py --job_dir scripts/batch_jobs/<label> [--skip_inhom] [--skip_stack] [--no_submit]`. Averages/stacks data and submits a single plotting SLURM job that runs `plot_datas.py`.

## HPC reminder

```bash
git pull --ff-only
conda activate m_env
conda env update -f environment.yml --prune  # only when dependencies change
python scripts/hpc_batch_dispatch.py --n_batches N --sim_type 2d
# Wait for batches to finish, then:
python scripts/hpc_plot_datas.py --job_dir scripts/batch_jobs/<label>
```

Logs from batch submissions stay in the same directories referenced by the HPC helper scripts.

## Persisting Problems:

- Reproducing Fig. 2 of https://pubs.aip.org/jcp/article/124/23/234504/930650/ — When including inhomogeneous broadening, the simulated signal still shows no rephasing after $t_{\text{det}} \approx t_{\text{coh}}$. Different realizations of the system have different transition energies, while the laser pulses drive at constant frequency of $\omega_L = 16000 cm^{-1}$. As a result, I would expect some realizations to dephase quicker the curve looks essentially identical to the case without inhomogeneous broadening. → This seems counterintuitive: I would expect a partial rephasing signal, since some realizations should come back in phase around $t_{\text{det}} \approx t_{\text{coh}}$, just like in the paper.

- Reproducing Fig. 3 of https://pubs.aip.org/jcp/article/124/23/234504/930650/ — The 2d spectra is at the same position as in the article, but I would say that the features are not exactly the same shape.
Also, I can't reproduce the broadening along the diagonal for a homogeneously broadened system.

- Reproducing Fig. 3 of https://pubs.aip.org/jcp/article/124/23/234505/930637/ —  Again the position of the spectral features look good, but in my simulation, both the real and imaginary parts show sign changes, while in the paper the real part only shows positive contributions. the features of every pulse appear to be rotated by $90^{\circ}$ compared to the reference. They are aligned along the anti-diagonal, while in the papers they are always aligned along the diagonal.
