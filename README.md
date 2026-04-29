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

## Branch notes

- **master**: stable baseline.
   - Solvers: `lindblad`, `redfield`, `paper_eqs`.
   - Bath models: `ohmic`, `drudelorentz`, `ohmic+lorentzian`, `drudelorentz+lorentzian`.
- **ideas**: experimental extensions (currently unstable/problematic).
   - Extra solvers: `heom`, `montecarlo`.
   - Extra bath models: `subohmic`, `superohmic`, `subohmic+lorentzian`, `superohmic+lorentzian`.

Each package in `packages/` is a standalone `pyproject`.  When you create the conda environment the packages are installed in editable mode automatically.
## Environment setup

```bash
git clone https://github.com/LBPhysics/2025_master_thesis_python_leopold_bodamer.git
cd 2025_master_thesis_python_leopold_bodamer
conda env create -f environment.yml
conda activate m_env
```

The environment file already performs the editable installs for my custom made `plotstyle` and `qspectro2d`.

## Units: bath parameters

Atomic transition frequencies are specified in cm⁻¹ via `atomic.frequencies_cm`.

By default, bath parameters in YAML are interpreted as **dimensionless multiples** of
$$\bar\omega_0 = \mathrm{mean}(\texttt{atomic.frequencies_cm})$$

That is:
- `bath.bath_temperature` means $T/\bar\omega_0$
- `bath.bath_cutoff` means $\omega_c/\bar\omega_0$
- `bath.sb_coupling` means $\text{coupling}/\bar\omega_0$
- `config.solver_run_kwargs.sec_cutoff` is passed directly to QuTiP for Redfield runs; QuTiP interprets positive values as a dimensionless multiplier of `dw_min`, the smallest nonzero Bohr-frequency spacing of the Hamiltonian, while `-1` disables the secular cutoff

Internally, the code converts `atomic.frequencies_cm` to fs^-1 for the dynamics; the same
$\bar\omega_0$ (in fs^-1) is used to scale these bath parameters. The Redfield `sec_cutoff`
is no longer rescaled by the loader.

### Stable bath types

Supported `bath.bath_type` values now include:
- `ohmic`
- `drudelorentz`
- `ohmic+lorentzian`
- `drudelorentz+lorentzian`

For the Ohmic variants (`ohmic`, `ohmic+lorentzian`), you can optionally set an explicit
power-law exponent via `bath.s` (dimensionless). If omitted, the default is:
- `ohmic`: `s = 1.0`

For the `*+lorentzian` types, the Lorentzian peak is configured via **normalized (dimensionless)** YAML inputs:
- `bath.peak_width`: $\gamma / \bar\omega_0$
- `bath.peak_strength`: $\text{strength} / \text{sb\_coupling}$
- `bath.peak_center` (optional): $\omega_\mathrm{center}/\bar\omega_0$ (default `0.0`)
- `bath.wmax_factor` (optional): sets `wMax = wmax_factor * (bath.bath_cutoff * \bar\omega_0)` (default `10.0`)

Internally, the loader converts these as:
$$\gamma = \texttt{peak_width}\,\bar\omega_0, \quad \text{strength} = \texttt{peak_strength}\,\text{sb\_coupling}.$$

**Known limitation (important):** the `*+lorentzian` bath types are still numerically delicate.
In particular, trying to create a strong peak close to $\omega\approx 0$ (to boost pure dephasing) tends to be numerically fragile in the current QuTiP Bloch–Redfield workflow (rates become very sensitive to low-frequency behavior and the internal spectral integrations / interpolation can become unstable or slow).
If you need robust low-frequency dephasing, prefer the built-in Ohmic or Drude–Lorentz families for now, or use an analytic Drude–Lorentz/underdamped-mode model instead of an ad-hoc near-zero peak.

## Workflow overview

There are two supported workflows: a strict local runner for smaller jobs and an
HPC batching workflow for cluster runs. Both create one dedicated job directory
under `jobs/`.

The canonical job-directory name is:

```text
DD_HHMMSS_config_stem
```

If that exact name already exists, the allocator appends a collision suffix:

```text
DD_HHMMSS_config_stem_01
```

Whenever a command asks for `--job_dir`, pass the full path to that directory.
Do not pass only the config stem.

### Local workflow (run on your machine)

For small test jobs or limited parameter sweeps.

1. **Configure simulation** - Duplicate a template from `scripts/simulation_configs/`
   and adjust the physical parameters. If `--config` is omitted, the local runner
   picks the preferred default config automatically.
2. **Run the strict local workflow** - Use
   `python scripts/local/calc_datas.py --sim_type {0d,1d,2d} [--config /path/to/config.yaml] [--rng_seed S]`.
   The script resolves the config once, runs all `(t_coh, inhomogeneous sample)`
   combinations locally, writes the job artifacts, and immediately performs the
   strict reduction step.
3. **Re-run reduction only if needed** - Use
   `python scripts/local/process_datas.py --job_dir /path/to/jobs/DD_HHMMSS_config_stem [--skip_if_exists]`
   if you already have partial artifacts and want to rebuild the final processed
   file without re-running the simulation.
4. **Plot the processed artifact** - Use
   `python scripts/local/plot_datas.py --abs_path /path/to/jobs/DD_HHMMSS_config_stem/data/2d_inhom_averaged.npz [--time_only]`.

Example local run:

```bash
python scripts/local/calc_datas.py --sim_type 2d --config scripts/simulation_configs/monomer.yaml
```

Typical job layout:

```text
jobs/01_123456_monomer/
  job_metadata.json
  monomer.yaml
  data/
    raw.pkl
    raw_samples.npy
    raw_combos.json
    raw_batch_000.partial.npz
    2d_inhom_averaged.npz
    2d_inhom_averaged.pkl
  figures/
```

### HPC batching workflow (run on a cluster)

For large sweeps with many coherence times and inhomogeneous samples.

1. **Dispatch batches** - Use
   `python scripts/hpc/calc_dispatcher.py --sim_type {0d,1d,2d} --n_batches N [--rng_seed S] [--no_submit] [--config /path/to/config.yaml]`.
   This creates `jobs/DD_HHMMSS_config_stem/`, writes the shared job
   metadata, batch manifests, copied config, and SLURM scripts, and optionally
   submits the batch jobs.
2. **Run batches** - Unless `--no_submit` is used, the dispatcher submits
   `scripts/hpc/run_batch.py` jobs via `sbatch`. Each batch writes exactly one
   strict partial artifact into the job's `data/` directory.
3. **Queue reduction and plotting** - After the batch jobs finish, run
   `python scripts/hpc/plot_dispatcher.py --job_dir /path/to/jobs/DD_HHMMSS_config_stem [--skip_if_exists] [--no_submit] [--time_only]`.
   This creates the reduction and plotting SLURM scripts and optionally submits
   them with the correct dependency chain.

HPC reminder:

```bash
git pull
conda activate m_env
conda env update -f environment.yml --prune  # only when dependencies change
python scripts/hpc/calc_dispatcher.py --n_batches N --sim_type 2d --config /path/to/config.yaml
# Wait for the batch jobs to finish, then replace the job_dir with the actual
# directory created above, e.g. jobs/01_123456_monomer
python scripts/hpc/plot_dispatcher.py --job_dir /path/to/jobs/DD_HHMMSS_config_stem
```

Batch and post-processing logs stay under the corresponding job directory,
typically in `jobs/DD_HHMMSS_config_stem/logs/`.

## Future implementation idea:

### Spectral Diffusion
### 1. What it is

Each molecule has a transition frequency

\[
\omega_{\mathrm{eg}}(t) = \bar{\omega}_{\mathrm{eg}} + \delta \omega(t)
\]

that drifts in time because its local surroundings slightly change — e.g. molecular reorientation, vibrations, or solvent polarisation relaxation.

If \(\delta \omega(t)\) changes slowly, the system is inhomogeneously broadened: each molecule has a fixed but different frequency → static disorder.

If \(\delta \omega(t)\) changes fast, the environment “averages out” → homogeneous broadening.

When \(\delta \omega(t)\) changes on intermediate timescales (tens of fs–ps), the frequency correlation decays in time — that decay is spectral diffusion.

Mathematically it is described by a correlation function

\[
C(t) = \langle \delta \omega(t) \, \delta \omega(0) \rangle
\]

and a correlation time \(\tau_c\).

A fixed homogeneous linewidth corresponds to memoryless (Markovian) dephasing.
Spectral diffusion introduces finite memory: correlation decays on ps or fs scales, affecting the shape and orientation of 2D peaks. Modeling it is therefore essential when:

the bath correlation time ≈ the waiting-time range of the experiment,

you want to extract dynamical information (protein relaxation, solvent reorganization, etc.).

For molecule \( n \): the energy gap \( \omega_{\mathrm{eg}}^{(n)}(t) \) fluctuates with correlation

\[
C(t) = \langle \delta \omega(t) \, \delta \omega(0) \rangle
\]

(their Eq. 30).

Pick a stationary stochastic model with that \( C(t) \). The standard choice is Ornstein–Uhlenbeck (OU) with correlation time \( \tau_{\mathrm{corr}} \):

\[
d\omega = -\frac{\omega - \bar{\omega}}{\tau_{\mathrm{corr}}} \, dt + \sqrt{\frac{2\sigma^2}{\tau_{\mathrm{corr}}}} \, dW_t, \quad C(t) = \sigma^2 e^{-|t|/\tau_{\mathrm{corr}}}.
\]

Build a time-dependent Hamiltonian for each molecule,

\[
H_n(t) = H_0 + \hbar \, \delta \omega_n(t) \, |e\rangle\langle e|.
\]

Compute the third-order response (rephasing and non-rephasing) for your pulse sequence for each realization, then average over many molecules/realizations. As the waiting time \( T \) grows, the distribution “diffuses,” rotating and symmetrizing the 2D lineshape.
