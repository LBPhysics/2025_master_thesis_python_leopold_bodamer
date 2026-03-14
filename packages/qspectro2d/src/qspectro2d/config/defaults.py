"""Single source of truth for configuration defaults.

This module keeps all raw defaults in one place so the rest of the config
package can work with one merged config object instead of reconstructing state
from scattered module globals.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np

# --- Generic defaults ---
INITIAL_STATE = "ground"

# --- Time and grid defaults (kept as flat vars for backward compat) ---
T_DET_MAX = 20.0
T_COH_MAX = T_DET_MAX
DT = 0.1
T_WAIT = 0.0

# --- Supported options ---
SUPPORTED_SOLVERS = ["lindblad", "redfield", "paper_eqs"]

SUPPORTED_BATHS = [
    "ohmic",
    "drudelorentz",
    "ohmic+lorentzian",
    "drudelorentz+lorentzian",
]

SUPPORTED_ENVELOPES = ["gaussian", "cos2"]

SUPPORTED_SIM_TYPES = ["0d", "1d", "2d"]

SOLVER_OPTIONS = {
    "paper_eqs": {},
    "redfield": {
        "sec_cutoff": 0,
    },
    "lindblad": {
        "atol": 1e-5,
        "rtol": 1e-3,
        "nsteps": 200000,
        "method": "bdf",
    },
}

ALLOWED_SOLVER_OPTIONS = {
    "paper_eqs": [],
    "lindblad": ["atol", "rtol", "nsteps", "method", "max_step", "min_step"],
    "redfield": [
        "atol",
        "rtol",
        "nsteps",
        "method",
        "max_step",
        "min_step",
        "sec_cutoff",
    ],
}

N_PHASES = 4
DPHI = 2 * np.pi / N_PHASES
PHASE_CYCLING_PHASES = DPHI * np.arange(N_PHASES)

SIGNAL_TYPES = ["rephasing", "nonrephasing"]

COMPONENT_MAP: dict[str, tuple[int, int]] = {
    "average": (0, 0),
    "rephasing": (-1, 1),
    "nonrephasing": (1, -1),
    "doublequantum": (1, 1),
}

NEGATIVE_EIGVAL_THRESHOLD = -1e-3
TRACE_TOLERANCE = 1e-6

DEFAULTS = {
    "atomic": {
        "n_atoms": 1,
        "n_chains": 1,
        "frequencies_cm": [16000.0],
        "dip_moments": [1.0],
        "coupling_cm": 0.0,
        "max_excitation": 1,
        "n_inhomogen": 1,
        "delta_inhomogen_cm": 0.0,
        "deph_rate_fs": 1 / 100,
        "down_rate_fs": 1 / 300,
        "up_rate_fs": 0.0,
    },
    "laser": {
        "pulse_fwhm_fs": None,
        "pulse_amplitudes": [0.01, 0.01, 0.01],
        "envelope_type": "gaussian",
        "carrier_freq_cm": 16000.0,
        "rwa_sl": True,
    },
    "bath": {
        "bath_type": "ohmic",
        "temperature": 1e-2,
        "cutoff": 1e2,
        "coupling": 1e-4,
        "s": None,
        "wmax_factor": 10.0,
        "peak_strength": 0.0,
        "peak_width": 1.0,
        "peak_center": 0.0,
    },
    "config": {
        "solver": "redfield",
        "solver_options": {},
        "sim_type": "1d",
        "t_det_max": 20.0,
        "t_coh_max": None,
        "t_coh": None,
        "t_wait": 0.0,
        "dt": 0.1,
        "n_phases": N_PHASES,
        "signal_types": list(SIGNAL_TYPES),
        "initial_state": INITIAL_STATE,
        "max_workers": None,
    },
}


def default_pulse_fwhm_fs(n_atoms: int) -> float:
    """Return the derived default pulse width for a system size."""
    return 15.0 if int(n_atoms) == 1 else 5.0


def get_defaults() -> dict:
    """Return a deep copy so callers can safely mutate merged configs."""
    return deepcopy(DEFAULTS)


__all__ = [
    "ALLOWED_SOLVER_OPTIONS",
    "COMPONENT_MAP",
    "DEFAULTS",
    "DPHI",
    "INITIAL_STATE",
    "NEGATIVE_EIGVAL_THRESHOLD",
    "N_PHASES",
    "PHASE_CYCLING_PHASES",
    "SIGNAL_TYPES",
    "SOLVER_OPTIONS",
    "SUPPORTED_BATHS",
    "SUPPORTED_ENVELOPES",
    "SUPPORTED_SIM_TYPES",
    "SUPPORTED_SOLVERS",
    "TRACE_TOLERANCE",
    "T_COH_MAX",
    "T_DET_MAX",
    "T_WAIT",
    "DT",
    "default_pulse_fwhm_fs",
    "get_defaults",
]
