# qspectro2d

Modular tools for simulating 1D and 2D electronic four-wave mixing spectroscopy on open quantum systems, developed as part my  overall Master’s thesis project.

## Table of contents
- [Overview](#overview)
- [Core capabilities](#core-capabilities)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Configuration-driven workflow](#configuration-driven-workflow)
- [Working with results](#working-with-results)
- [Visualization](#visualization)
- [Background literature](#background-literature)

## Overview
`qspectro2d` streamlines the complete workflow of nonlinear spectroscopy simulations:

1. Define excitonic systems with configurable geometries, truncation levels, and inhomogeneous broadening.
2. Specify laser pulses and phase-cycling schemes consistent with four-wave mixing experiments.
3. Couple systems to dissipative environments using Bloch–Redfield, Lindblad, or paper-specific equations of motion.
4. Propagate dynamics, compute polarizations, and perform Fourier-domain post-processing for 1D/2D spectroscopy.

The package underpins the numerical results of the Master’s thesis and is designed to be reusable for related projects.

## Core capabilities

| Domain | Highlights |
| --- | --- |
| Atomic & excitonic models | Cylindrical chain geometries, single/double excitation truncation, history-aware frequency updates |
| Bath descriptions | Ohmic and Drude–Lorentz spectral densities, Qutip `OhmicEnvironment`, configurable coupling strengths |
| Laser pulses | Gaussian and cos² envelopes, automatic pulse-delay synthesis, phase cycling presets |
| Simulation engine | Factory to assemble `SimulationModuleOQS`, solver validation, support for Qutip solvers and paper-derived Liouvillians |
| Spectroscopy utilities | Analytical polarization, solver sanity checks, FFT helpers, signal averaging and component selection |
| Project utilities | Default parameter validation, file I/O helpers, thesis path conventions, plotting helpers |

## Architecture

```
qspectro2d/
├── config/        # Default parameters, YAML loading → SimulationModuleOQS factory
├── core/          # Atomic, laser, bath, and simulation classes
├── spectroscopy/  # Polarization, solver validation, post-processing
├── utils/         # Constants, I/O, file naming, rotating-wave helpers
└── visualization/ # Plotting wrappers (Matplotlib + thesis aesthetics)
```

### Key modules
- `config.create_sim_obj` – high-level entry point that reads YAML config files, applies physics-aware validation, and returns a ready-to-run simulation object with consistent timing arrays.
- `core.atomic_system.system_class.AtomicSystem` – manages basis construction, exciton Hamiltonians, and geometry-dependent couplings.
- `core.simulation.simulation_class.SimulationModuleOQS` – wraps system, laser, bath, and solver settings.
- `spectroscopy.polarization` – provides analytical polarization computation (`complex_polarization`) compatible with solver outputs.
- `utils.data_io` – standardizes how time-domain signals and metadata are saved/loaded to support reproducible thesis figures.

## Installation

The package targets Python ≥ 3.11. It depends on NumPy, SciPy, Matplotlib, Qutip, PyYAML, and optional extras for plotting.


### User installation
```bash
pip install -e qspectro2d
```

Optional extras:
- `plotstyle` (https://github.com/LBPhysics/plotstyle) for LaTeX-quality Matplotlib output.

## Quick start

Discover the public API and assemble basic simulations programmatically:

```python
from qspectro2d import (
		AtomicSystem,
)
from qspectro2d.core.laser_system.laser_class import LaserPulseSequence
from qspectro2d.core.simulation.sim_config import SimulationConfig
from qspectro2d.core.simulation.simulation_class import SimulationModuleOQS

system = AtomicSystem(
		n_atoms=2,
		frequencies_cm=[16000.0, 16050.0],
		dip_moments=[1.0, 0.9],
		coupling_cm=150.0,
)

sequence = LaserPulseSequence.from_pulse_delays(
		pulse_delays=[0.0, 50.0],
		pulse_fwhm_fs=15.0,
		carrier_freq_cm=16000.0,
)

config = SimulationConfig(
		ode_solver="redfield",
		dt=0.1,
		t_coh=0.0,
		t_wait=100.0,
		t_det_max=250.0,
)

simulation = SimulationModuleOQS(
		simulation_config=config,
		system=system,
		laser=sequence,
)
```

## Configuration-driven workflow

1. Describe system, laser, bath, and solver blocks in YAML (values omitted fall back to module defaults in `qspectro2d.config`).
2. Call `create_sim_obj(path_to_yaml)` to obtain a validated `SimulationModuleOQS` instance, with cutoff time (where the simulation becomes unphysical).

### Loading and validating configuration

```python
from pathlib import Path
from qspectro2d.config.create_sim_obj import load_simulation, create_base_sim_oqs

sim = load_simulation(Path("config/dimer.yaml"))
sim_summary = sim.summary()

sim, time_cut = create_base_sim_oqs(Path("config/dimer.yaml"))
print(f"Usable detection window: {time_cut:.1f} fs")
```

The loader automatically respects `SLURM_CPUS_PER_TASK` for parallel averaging and raises actionable errors when parameter combinations are inconsistent (see `qspectro2d.config.validation.validate`).

### Solver options

You can pass solver-specific knobs under `config.solver_options`. These are forwarded to the selected QuTiP solver backend.

Defaults live in `qspectro2d.config.simulation.SOLVER_OPTIONS`, and the allowed keys are validated in `qspectro2d.config.validation.validate`.
Unknown keys raise an error to avoid silently ignored typos.

Common keys (for the solvers that support them):
- `atol`, `rtol`: absolute/relative tolerances (must be > 0)
- `nsteps`: maximum allowed internal steps (must be > 0)
- `method`: ODE method string (e.g. `"bdf"`)
- `max_step`, `min_step`: optional step size limits
- `progress_bar`: show QuTiP progress bar (supported by `montecarlo` and `heom`)

Solver-specific keys:
- `redfield`:
	- `sec_cutoff`: controls the secular approximation in QuTiP `brmesolve`
		- `sec_cutoff = -1` disables the secular approximation
		- `sec_cutoff > 0` enables the secular approximation with that cutoff
- `montecarlo`:
	- `ntraj`: number of trajectories
- `heom`:
	- `max_depth`: hierarchy depth (1 approaches Redfield; higher is more accurate/costly)
	- `approx_method`: bath decomposition method (e.g. `"prony"`)
	- `n_exp`: number of exponentials in the decomposition (when supported by the chosen method)
	- `Ni`, `Nr`: decomposition knobs (method-dependent)
	- `combine`, `separate`: decomposition bookkeeping switches
	- `n_t`, `t_max`: internal bath-fit grid size and max time
	- `tag`: optional solver tag
- `paper_eqs`: no `solver_options` (empty allow-list)

Example (turn off secular approximation for Bloch–Redfield):

```yaml
config:
	solver: redfield

	# Solver-specific knobs. For QuTiP brmesolve:
	#   sec_cutoff = -1 disables the secular approximation
	#   sec_cutoff > 0 applies secular approximation with that cutoff
	solver_options:
		sec_cutoff: -1
```

## Bath configuration (normalized units)

Bath parameters in YAML are interpreted as dimensionless multiples of
$\bar\omega_0 = \mathrm{mean}(\texttt{atomic.frequencies_cm})$ (converted internally to fs⁻¹).

Supported `bath.bath_type` values:
- `ohmic`
- `drudelorentz`
- `ohmic+lorentzian`
- `drudelorentz+lorentzian`

For the `*+lorentzian` types, add a Lorentzian peak to the base spectral density using normalized inputs:
- `bath.peak_width`: $\gamma / \bar\omega_0$
- `bath.peak_strength`: $\text{strength} / \text{coupling}$
- `bath.peak_center` (optional): $\omega_\mathrm{center}/\bar\omega_0$ (default `0.0`)
- `bath.wmax_factor` (optional): `wMax = wmax_factor * cutoff` in internal units (default `10.0`)

**Known limitation:** the `*+lorentzian` bath types are still experimental.
Peaks very close to $\omega\approx 0$ (often used to boost pure dephasing) can make Bloch–Redfield rates highly sensitive to low-frequency details and may lead to slow/unstable numerics.
For stable dephasing control, prefer the built-in Ohmic/Sub-Ohmic families, or analytic Drude–Lorentz / underdamped-mode environments.

## Working with results

- `qspectro2d.utils.data_io` – save raw time-domain polarizations, spectral grids, and metadata.
- `qspectro2d.spectroscopy.post_processing` – apply FFTs, phase matching, and generate absorption/emission maps.
- `qspectro2d.spectroscopy.solver_check.check_the_solver` – detect unphysical behavior and return the recommended time cut.

All outputs are designed to integrate with the thesis data hierarchy (`Master_thesis/data` and `Master_thesis/figures`).

## Visualization

The optional companion package `plotstyle` enforces LaTeX fonts, consistent color palettes, and thesis-ready sizes:

```python
from plotstyle import init_style, save_fig, set_size

init_style()
figsize = set_size(fraction=0.9)

# ... build Matplotlib figure ...

save_fig(fig, "./figures/2d_spectrum", formats=["pdf", "png"])
```

If LaTeX binaries are missing, the style falls back gracefully to non-TeX rendering.

## Background literature
- Monomer response: [J. Chem. Phys. 124, 234504 (2006)](https://pubs.aip.org/jcp/article/124/23/234504/930650/)
- Dimer response (coupled/uncoupled): [J. Chem. Phys. 124, 234505 (2006)](https://pubs.aip.org/jcp/article/124/23/234505/930637/)