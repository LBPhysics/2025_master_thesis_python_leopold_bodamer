# Thesis spectroscopy toolkit

This repository contains the reproducible Python code for the spectroscopy part of the thesis. It is organized as two installable packages under `packages/` and a thin CLI/workflow layer under `scripts/`.

## Repository layout

```
thesis_python/
├─ packages/
│  ├─ plotstyle/   # Matplotlib/LaTeX styling helpers
│  └─ qspectro2d/  # Open-quantum-system spectroscopy simulations
├─ scripts/
│  ├─ common/      # Shared workflow helpers
│  ├─ local/       # Local execution, reduction, plotting, analysis
│  └─ hpc/         # SLURM dispatchers and batch runner
├─ environment.yml  # Conda environment that installs both packages editable
├─ pyproject.toml   # Ruff / Black / pytest / coverage configuration
├─ README.md        # Project overview and reproducible workflow
└─ .gitignore       # Local-only outputs, caches, notebooks, and scratch data
```

`jobs/`, `tmp/`, `venv/`, caches, and notebook outputs are treated as local work artifacts and are intentionally excluded from the repository.

## Setup

The environment file already installs both packages in editable mode:

```bash
conda env create -f environment.yml
conda activate m_env
```

If you prefer `pip`, install the two packages from the repository root:

```bash
pip install -e ./packages/qspectro2d -e ./packages/plotstyle
```

The code targets Python 3.11. The `plotstyle` package uses LaTeX when it is available and falls back to Matplotlib’s math rendering when it is not.

## What the code supports

The public simulation code currently supports:

- solvers: `lindblad`, `redfield`, `paper_eqs`
- simulation types: `0d`, `1d`, `2d`
- bath types: `ohmic`, `drudelorentz`, `ohmic+lorentzian`, `drudelorentz+lorentzian`
- batch workflows with strict partial reduction followed by final processing and plotting

Bath and solver defaults live in `packages/qspectro2d/src/qspectro2d/config/defaults.py`. Bath parameters in YAML are interpreted in normalized units relative to the mean atomic transition frequency, and `config.solver_run_kwargs.sec_cutoff` is passed through to QuTiP unchanged.

## Reproducible workflow

This checkout does not ship committed YAML simulation templates. The workflow helpers support a local-only `scripts/simulation_configs/` lookup if you omit `--config`, but for a clean clone the safest option is to pass an explicit config path.

### Local run

Use the local strict runner for smaller jobs or single-machine sweeps:

```bash
python scripts/local/calc_datas.py --sim_type 2d --config /path/to/config.yaml
python scripts/local/process_datas.py --job_dir /path/to/jobs/DD_HHMMSS_config_stem
python scripts/local/plot_datas.py --abs_path /path/to/jobs/DD_HHMMSS_config_stem/data/2d_inhom_averaged.npz
```

Useful flags:

- `scripts/local/calc_datas.py`: `--rng_seed`, `--phase_pool_max_combos`
- `scripts/local/process_datas.py`: `--skip_if_exists`
- `scripts/local/plot_datas.py`: `--time_only`, `--freq_only`, `--apodization_window`

The local runner creates one job directory under `jobs/` with a canonical name of the form `DD_HHMMSS_config_stem` and appends `_NN` on collisions. Each job directory contains the copied config, `job_metadata.json`, `data/`, `figures/`, and logs produced by the workflow.

### HPC run

Use the SLURM dispatcher for larger sweeps:

```bash
python scripts/hpc/calc_dispatcher.py --sim_type 2d --n_batches 8 --config /path/to/config.yaml
python scripts/hpc/plot_dispatcher.py --job_dir /path/to/jobs/DD_HHMMSS_config_stem
```

Useful flags:

- `scripts/hpc/calc_dispatcher.py`: `--rng_seed`, `--time_cut`, `--cpus_per_task`, `--phase_pool_max_combos`, `--partition`, `--no_submit`
- `scripts/hpc/plot_dispatcher.py`: `--skip_if_exists`, `--no_submit`, `--time_only`

`scripts/hpc/calc_dispatcher.py` prepares the job directory, shared metadata, batch manifests, copied config, and SLURM scripts. `scripts/hpc/run_batch.py` is the internal batch entrypoint that each SLURM job executes. After the batches finish, `scripts/hpc/plot_dispatcher.py` queues the strict reduction job and the dependent plotting job.

## Package docs

- `packages/qspectro2d/README.md` documents the simulation library and public API.
- `packages/plotstyle/README.md` documents the plotting helper package and the notebook example.
