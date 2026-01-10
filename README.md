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

The environment file already performs the editable installs for my custom made `plotstyle` and `qspectro2d`.

## Units: bath parameters

Atomic transition frequencies are specified in cm⁻¹ via `atomic.frequencies_cm`.

By default, bath parameters in YAML are interpreted as **dimensionless multiples** of
$$\bar\omega_0 = \mathrm{mean}(\texttt{atomic.frequencies_cm})$$

That is:
- `bath.temperature` means $T/\bar\omega_0$
- `bath.cutoff` means $\omega_c/\bar\omega_0$
- `bath.coupling` means $\text{coupling}/\bar\omega_0$

Internally, the code converts `atomic.frequencies_cm` to fs⁻¹ for the dynamics; the same
$\bar\omega_0$ (in fs⁻¹) is used to scale these bath parameters.

### Extra bath types: `*+lorentzian`

Supported `bath.bath_type` values now include:
- `ohmic`
- `drudelorentz`
- `ohmic+lorentzian`
- `drudelorentz+lorentzian`

For the `*+lorentzian` types, the Lorentzian peak is configured via **normalized (dimensionless)** YAML inputs:
- `bath.peak_width`: $\gamma / \bar\omega_0$
- `bath.peak_strength`: $\text{strength} / \text{coupling}$
- `bath.peak_center` (optional): $\omega_\mathrm{center}/\bar\omega_0$ (default `0.0`)
- `bath.wmax_factor` (optional): sets `wMax = wmax_factor * (bath.cutoff * \bar\omega_0)` (default `10.0`)

Internally, the loader converts these as:
$$\gamma = \texttt{peak_width}\,\bar\omega_0, \quad \text{strength} = \texttt{peak_strength}\,\text{coupling}.$$

**Known limitation (important):** the `*+lorentzian` bath types are still experimental.
In particular, trying to create a strong peak close to $\omega\approx 0$ (to boost pure dephasing) tends to be numerically fragile in the current QuTiP Bloch–Redfield workflow (rates become very sensitive to low-frequency behavior and the internal spectral integrations / interpolation can become unstable or slow).
If you need robust low-frequency dephasing, prefer the built-in Ohmic/Sub-Ohmic families for now, or use an analytic Drude–Lorentz/underdamped-mode model instead of an ad-hoc near-zero peak.

## Workflow overview

There are two main workflows: **local execution** (for small tests) and **HPC batching** (for large parameter sweeps). Both produce the same plots but differ in scaling and automation.

### Local Workflow (Run on Your Machine)
For small-scale runs (e.g., quick tests or limited parameter sweeps). Generates all combinations of coherence times and inhomogeneous samples, then processes them into final averaged spectra.

1. **Configure simulation** — Duplicate the template in `scripts/simulation_configs/` and adjust physical parameters. `_monomer.yaml` is the default that `calc_datas.py` auto-selects.
   - Always set `config.t_coh` (fs) for 0d/1d runs. If omitted, the code falls back to `t_coh_max` (which defaults to `t_det_max`), but explicit values make intent clear.
   - 2d sweeps ignore `config.t_coh` and instead iterate over the full grid up to `t_coh_max`.

1. **Simulate** — Run `python scripts/calc_datas.py --sim_type {0d,1d,2d}` locally.
   - Generates all combinations of `t_coh` points and inhomogeneous samples.
   - Outputs raw `.npz` files per combination.
   - For a 2D simulation the `t_coh` value in the config will be ignored and instead all t_coh values (same as the detection times `t_det`) will be used.

   **Example:**

   ```bash
   (m_env) path/to/scripts$ python calc_datas.py --sim_type 1d
   ```

   ```
   ================================================================================
   LOCAL ALL-COMBINATIONS RUNNER
   Config path: /home/leopold/Projects/2025_master_thesis_python_leopold_bodamer/scripts/simulation_configs/_monomer.yaml
   ...
   Completed 1 combination(s) in 3.26 s
   Latest artifact:
   /home/leopold/Projects/2025_master_thesis_python_leopold_bodamer/data/1_atoms/lindblad/RWA/t_dm300.0_t_wait10.0_dt_0.2_1/1d_run_t000_s000.npz

   🎯 Next step:
      python process_datas.py --abs_path '/home/leopold/Projects/2025_master_thesis_python_leopold_bodamer/data/1_atoms/lindblad/RWA/t_dm300.0_t_wait10.0_dt_0.2_1/1d_run_t000_s000.npz' --skip_if_exists
   ================================================================================
   DONE
   ```
3. **Process** — Run `python scripts/process_datas.py --abs_path /path/to/any/artifact.npz` to stack (if multiple `t_coh`) and average across samples in one efficient step.

   **Example:**

   ```bash
   (m_env) path/to/scripts$ python process_datas.py --abs_path '/home/leopold/Projects/2025_master_thesis_python_leopold_bodamer/data/1_atoms/lindblad/RWA/t_dm300.0_t_wait10.0_dt_0.2_1/1d_run_t000_s000.npz' --skip_if_exists
   ```

   ```
   ...
   🎯 Plot with:
   python plot_datas.py --abs_path /home/leopold/Projects/2025_master_thesis_python_leopold_bodamer/data/1_atoms/lindblad/RWA/t_dm300.0_t_wait10.0_dt_0.2_1/1d_inhom_averaged.npz
   ```

4. **Visualize** — Run `python scripts/plot_datas.py --abs_path /path/to/processed_artifact.npz` to generate time/frequency-domain plots (e.g., signals, spectra). The script applies zero-padding with a factor of `EXTEND` for frequency-domain plots, crops frequency data to the range `SECTION` [10^4 cm⁻¹], and always generates both time and frequency domains.

   **Example:**

   ```bash
   (m_env) path/to/scripts$ python plot_datas.py --abs_path /home/leopold/Projects/2025_master_thesis_python_leopold_bodamer/data/1_atoms/lindblad/RWA/t_dm300.0_t_wait10.0_dt_0.2_1/1d_inhom_averaged.npz
   ```

   -> find the figures under `figures/...`

Simulation outputs remain under `data/` and plots under `figures/`.

### HPC Batching Workflow (Run on a Cluster)
For large-scale runs (e.g., full sweeps with many inhomogeneous samples and coherence times). Supports all combinations with parallel batching. Processing and plotting are automated.
Is structured similar to the local workflow but splits the simulation into batches that are atomically executed on the cluster:

1. **Dispatch batches** — Run `python scripts/hpc_batch_dispatch.py --sim_type {0d,1d,2d} --n_batches N [--rng_seed S] [--no_submit]`. Generates SLURM jobs that split work across combinations. Validates locally first.

2. **Run batches** — Batches auto-submit via `sbatch` (unless `--no_submit`). Each runs `run_batch.py` on the cluster, producing partial artifacts in `data/...`.

3. **Post-process and plot** — After batches finish, run `python scripts/hpc_plot_datas.py --job_dir scripts/batch_jobs/<label> [--skip_if_exists] [--no_submit]`. Processes data (stacks and averages) and submits a single plotting SLURM job that runs `plot_datas.py`.

## HPC reminder

```bash
git pull
conda activate m_env
conda env update -f environment.yml --prune  # only when dependencies change
python scripts/hpc_batch_dispatch.py --n_batches N --sim_type 2d
# Wait for batches to finish, then:
python scripts/hpc_plot_datas.py --job_dir scripts/batch_jobs/<label>
```

Logs from batch submissions stay in the same directories referenced by the HPC helper scripts.

## Future implementation idea:

### Spectral Diffusion
### 1. What it is

Each molecule has a transition frequency

\[
\omega_{\mathrm{eg}}(t) = \bar{\omega}_{\mathrm{eg}} + \delta \omega(t)
\]

that drifts in time because its local surroundings slightly change — e.g. molecular reorientation, vibrations, or solvent polarization relaxation.

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