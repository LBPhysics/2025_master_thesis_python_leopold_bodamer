# Thesis Spectroscopy Python Toolkit (Minimal Edition)

This directory is a snapshot of the Python-only components extracted from the full Master Thesis repository. It contains only reusable simulation, plotting, and helper code.

## Included Packages
- `thesis_paths`: Paths.
- `qspectro2d`: Core quantum spectroscopy simulation framework (1D/2D electronic spectroscopy; system, bath, laser, simulation orchestration, I/O, visualization).
- `plotstyle`: Matplotlib style helpers tuned for LaTeX-quality figures.
- `scripts/`: *MOST IMPORTANT* CLI utilities for batch simulations and data post-processing (`calc_datas.py`, `plot_datas.py`, etc.).
- `notebooks/`: Demonstration and exploratory notebooks (may be trimmed further for distribution).

## Quick Start (Development Install)
```bash
conda env create -f environment.yml
conda activate m_env
```

## Workflow summary
1. **Configure** – Copy/edit a YAML under `code/python/scripts/simulation_configs/`.
2. **Simulate** – Run `calc_datas --sim_type {1d,2d}` locally or via SLURM (`hpc_calc_datas.py` with n_batches).
3. **Aggregate** – Stack inhomogeneous runs (`stack_inhomogenity.py`) and build 2D datasets (`stack_times`).
4. **Plot** – Use `plot_datas.py` (or `hpc_plot_datas.py`) to create time/frequency figures; outputs land in `figures/figures_from_python/`.

## HPC checklist
```bash
git pull --ff-only
conda activate master_env
conda env update -f environment.yml --prune   # only when dependencies changed
python code/python/scripts/hpc_calc_datas.py --n_batches N --sim_type 2d
```

Generated SLURM scripts store logs in `code/python/scripts/batch_jobs/`.

## References
- `code/python/README.md` – top-level CLI + path instructions
- `code/python/qspectro2d/README.md` – detailed simulation docs and YAML schema
- `code/python/plotstyle/README.md` – plotting style usage
- `latex/` – thesis structure, chapter templates, bibliography