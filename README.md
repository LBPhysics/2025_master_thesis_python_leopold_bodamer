# Thesis Spectroscopy Python Toolkit (Minimal Edition)

This repository is a Python-only collection of the full Master Thesis repository.

## Included Packages
- `thesis_paths`: Paths.
- `qspectro2d`: Core quantum spectroscopy simulation framework (1D/2D electronic spectroscopy; system, bath, laser, simulation orchestration, I/O, visualization).
    - `config`: Validates default physics parameters and builds ready-to-run `SimulationModuleOQS` objects from YAML inputs.
    - `core`: Defines the dynamical model (atomic systems, baths, laser pulses) and solver building blocks used across simulations.
    - `spectroscopy`: Provides 1D/2D pulse evolution, polarization pipelines, inhomogeneous sampling, and post-processing transforms. For post-processing it follows https://doi.org/10.1063/5.0214023.
    - `utils`: Shared constants, rotating-wave helpers, file naming, and data I/O utilities.
    - `visualization`: High-level plotting helpers for pulse envelopes, fields, and spectroscopy datasets.
- `plotstyle`: Drop-in Matplotlib styles for publication-ready figures—use these to get consistently “nice” plots that match the thesis aesthetic.
- `scripts/`: *most important* CLI utilities for (batch) simulations and data post-processing (`calc_datas.py`, `plot_datas.py`, etc.).
- `notebooks/`: Demonstration and exploratory notebooks.

## Quick Start (Development Install)
```bash
git clone https://github.com/LBPhysics/2025_master_thesis_python_leopold_bodamer.git
cd 2025_master_thesis_python_leopold_bodamer
conda env create -f environment.yml
conda activate m_env
```

## Workflow summary
1. **Configure** – Copy/edit a YAML under `scripts/simulation_configs/`. A template and the main schema are provided:
    - `template.yaml` explains all options.
    - `monomer.yaml` (reproduces https://pubs.aip.org/jcp/article/124/23/234504/930650/)
    - `[un]coupled_dimer.yaml` (reproduces https://pubs.aip.org/jcp/article/124/23/234505/930637/)
    - The script `calc_datas.py` will select the file that starts with an underscore e.g. `_monomer.yaml`.

All parameters that are not specified in the YAML will take defaults located in `qspectro2d/config`.

2. **Simulate** – Run `calc_datas --sim_type {1d,2d}` locally or via SLURM (`hpc_calc_datas.py` with n_batches).
3. **Aggregate** – Stack inhomogeneous runs (`stack_inhomogenity.py`) OR build 2D datasets (`stack_times`).
4. **Plot** – Use `plot_datas.py` (or `hpc_plot_datas.py`) to create time/frequency figures; outputs land in `figures/figures_from_python/`.

## HPC checklist
```bash
git pull --ff-only
conda activate master_env
conda env update -f environment.yml --prune   # only when dependencies changed
python scripts/hpc_calc_datas.py --n_batches N --sim_type 2d
```

Generated SLURM scripts store logs in `code/python/scripts/batch_jobs/`.


## Current problems (TODOS):

- when trying to recreate Fig. 2. from https://pubs.aip.org/jcp/article/124/23/234504/930650/ -> inhomogeneous broadening doesnt work (I GET NO REPHASING SIGNAL after time t_det ~ t_coh)
- when trying to recreate Fig. 3 of the same article / every other figure also Fig. 3 in https://pubs.aip.org/jcp/article/124/23/234505/930637/: the spectral features are rotated by 90 degrees