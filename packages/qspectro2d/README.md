# qspectro2d

`qspectro2d` is the simulation package that powers the spectroscopy workflow in this repository. It exposes the core open-quantum-system objects at package level and keeps the configuration and artifact helpers in submodules.

## Public API

The top-level package exports:

- `AtomicSystem`
- `LaserPulse`
- `LaserPulseSequence`
- `SimulationConfig`
- `SimulationModuleOQS`
- `compute_spectra`
- `save_run_artifact`
- `load_run_artifact`
- `load_simulation_data`

Configuration helpers live in `qspectro2d.config` and `qspectro2d.config.factory`:

- `resolve_config`
- `validate_config`
- `load_simulation_config`
- `load_simulation`
- `create_base_sim_oqs`

## Installation

From the repository root, either create the conda environment:

```bash
conda env create -f environment.yml
conda activate m_env
```

or install the package directly in editable mode:

```bash
pip install -e ./packages/qspectro2d
```

The package targets Python 3.11 and depends on NumPy, SciPy, QuTiP, Matplotlib, and PyYAML. If you also want thesis-style plots, install the sibling package `plotstyle` from `packages/plotstyle`.

## Quick start

Create a simulation from a YAML configuration file:

```python
from pathlib import Path

from qspectro2d import AtomicSystem, LaserPulseSequence, SimulationConfig, SimulationModuleOQS
from qspectro2d.config.factory import create_base_sim_oqs, load_simulation

sim = load_simulation(Path("config/dimer.yaml"))
print(sim.summary())

sim_checked, time_cut = create_base_sim_oqs(Path("config/dimer.yaml"))
print(f"Usable detection window: {time_cut:.1f} fs")
```

Or build the core objects manually:

```python
from qspectro2d import AtomicSystem, LaserPulseSequence, SimulationConfig, SimulationModuleOQS

system = AtomicSystem(
    n_atoms=2,
    frequencies_cm=[16000.0, 16050.0],
    dip_moments=[1.0, 0.9],
    coupling_cm=150.0,
)

laser = LaserPulseSequence.from_pulse_delays(
    pulse_delays=[0.0, 50.0],
    pulse_fwhm_fs=15.0,
    carrier_freq_cm=16000.0,
)

config = SimulationConfig(
    ode_solver="redfield",
    dt=0.1,
    t_coh=0.0,
    t_wait=100.0,
    t_det=250.0,
)

simulation = SimulationModuleOQS(
    simulation_config=config,
    system=system,
    laser=laser,
)
```

## Configuration

`qspectro2d.config.defaults` defines the supported solver, bath, envelope, and simulation-type options. The shipped defaults currently cover:

- solvers: `lindblad`, `redfield`, `paper_eqs`
- bath types: `ohmic`, `drudelorentz`, `ohmic+lorentzian`, `drudelorentz+lorentzian`
- envelopes: `gaussian`, `cos2`
- simulation types: `0d`, `1d`, `2d`

YAML configuration is merged with those defaults and validated once. The loader also passes `config.solver_run_kwargs.sec_cutoff` through to QuTiP unchanged for Redfield runs.

Bath parameters are normalized against the mean atomic transition frequency. In practice, `bath_temperature`, `bath_cutoff`, and the Lorentzian peak parameters are dimensionless inputs that are converted to internal units by the loader.

## Results and diagnostics

Use `qspectro2d.utils.data_io` to save and load run artifacts, and `qspectro2d.spectroscopy.post_processing.compute_spectra` for FFT-based post-processing. The solver diagnostic in `qspectro2d.diagnostics.check_the_solver` returns the recommended time cut when the evolution becomes unphysical.

The workflow scripts in `scripts/local/` and `scripts/hpc/` both consume the same configuration and artifact format, so results can be processed locally or through SLURM without changing the package API.
