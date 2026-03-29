"""Constants and defaults for configuration."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

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

ALLOWED_SOLVER_OPTIONS = {
    "paper_eqs": [],
    "lindblad": ["atol", "rtol", "nsteps", "method", "max_step"],
    "redfield": [
        "atol",
        "rtol",
        "nsteps",
        "method",
        "max_step",
    ],
}
ALLOWED_SOLVER_RUN_KWARGS = {
    "paper_eqs": [],
    "lindblad": [],
    "redfield": ["sec_cutoff"],
}

N_PHASES = 4
DPHI = 2 * np.pi / N_PHASES
PHASE_CYCLING_PHASES = DPHI * np.arange(N_PHASES)

N_PULSES = 3
PULSE_AMPLITUDES = [
    0.01
] * N_PULSES  # ensures that the population of the excited state is less then 1% -> contributions of higher nonlinearities remain negligible
SIGNAL_TYPES = ["rephasing", "nonrephasing"]

COMPONENT_MAP: dict[str, tuple[int, int]] = {
    "average": (0, 0),
    "rephasing": (-1, 1),
    "nonrephasing": (1, -1),
    "doublequantum": (1, 1),
}

NEGATIVE_EIGVAL_THRESHOLD = -1e-3
TRACE_TOLERANCE = 1e-6

# --- Time and grid defaults ---
DT = 0.1
T_DET = 1.0
T_COH = T_DET
T_WAIT = DT
DEFAULT_PULSE_FWHM_FS = 5.0

# --- Generic defaults ---
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
        # Currently mainly kept for paper-reproduction workflows.
        "deph_rate_fs": 1 / 100,
        "down_rate_fs": 1 / 300,
        "up_rate_fs": 0.0,
    },
    "laser": {
        "pulse_fwhm_fs": DEFAULT_PULSE_FWHM_FS,
        "pulse_amplitudes": list(PULSE_AMPLITUDES),
        "envelope_type": "gaussian",
        "carrier_freq_cm": 16000.0,
        "rwa_sl": True,
    },
    "bath": {
        "bath_type": "ohmic",
        "bath_temperature": 1e-2,
        "bath_cutoff": 1e2,
        "sb_coupling": 1e-4,
        "s": 1.0,
        "wmax_factor": 10.0,
        "peak_strength": 0.0,
        "peak_width": 1.0,
        "peak_center": 0.0,
    },
    "config": {
        "solver": "redfield",
        "solver_options": {
            "method": (
                "lsoda"
            ),  # automatically switches between stiff and non-stiff methods, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp
        },
        "solver_run_kwargs": {
            "sec_cutoff": 1.0e-6,
        },
        "sim_type": "1d",
        "t_det": T_DET,
        "t_coh": T_COH,
        "t_wait": T_WAIT,
        "dt": DT,
        "n_phases": N_PHASES,
        "signal_types": list(SIGNAL_TYPES),
        "initial_state": "ground",
        "max_workers": None,  # always check cpu cores
    },
}


def phase_cycling_phases(n_phases: int) -> np.ndarray:
    """Return an equidistant phase-cycling grid over [0, 2π)."""
    n_phases = int(n_phases)
    if n_phases <= 0:
        raise ValueError("n_phases must be >= 1")
    return (2 * np.pi / n_phases) * np.arange(n_phases, dtype=float)


def get_defaults() -> dict:
    """Return a deep copy so callers can safely mutate merged configs."""
    return deepcopy(DEFAULTS)


__all__ = [
    "ALLOWED_SOLVER_OPTIONS",
    "COMPONENT_MAP",
    "DEFAULTS",
    "DEFAULT_PULSE_FWHM_FS",
    "DPHI",
    "DT",
    "get_defaults",
    "NEGATIVE_EIGVAL_THRESHOLD",
    "N_PHASES",
    "N_PULSES",
    "PHASE_CYCLING_PHASES",
    "PULSE_AMPLITUDES",
    "SIGNAL_TYPES",
    "SUPPORTED_BATHS",
    "SUPPORTED_ENVELOPES",
    "SUPPORTED_SIM_TYPES",
    "SUPPORTED_SOLVERS",
    "TRACE_TOLERANCE",
    "T_WAIT",
    "phase_cycling_phases",
]
