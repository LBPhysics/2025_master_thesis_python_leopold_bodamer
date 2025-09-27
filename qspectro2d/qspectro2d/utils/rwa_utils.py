"""Rotating frame helpers (RWA phase transforms)."""

from __future__ import annotations

import numpy as np
from typing import List
from qutip import Qobj, expect

__all__ = [
    "rotating_frame_unitary",
    "to_rotating_frame_op",
    "from_rotating_frame_op",
    "to_rotating_frame_list",
    "from_rotating_frame_list",
    "get_expect_vals_with_RWA",
]


def _excitation_number_vector(dim: int, n_atoms: int) -> np.ndarray:
    """Construct the excitation-number vector n for basis ordering:
    [0-ex], then [1-ex with n_atoms states], then [2-ex remainder].
    """
    if n_atoms < 0:
        raise ValueError("n_atoms must be non-negative")
    if dim <= 0:
        raise ValueError("dim must be positive")

    n = np.zeros(dim, dtype=int)
    # 1-ex manifold: indices [1 .. min(n_atoms, dim-1)]
    one_ex_end = min(n_atoms, dim - 1)
    if one_ex_end >= 1:
        n[1 : one_ex_end + 1] = 1
    # 2-ex manifold: indices [n_atoms+1 .. dim-1]
    two_ex_start = n_atoms + 1
    if dim > two_ex_start:
        n[two_ex_start:] = 2
    return n


# --- Simple unitary-based API -------------------------------------------------------
def rotating_frame_unitary(ref: Qobj, t: float, n_atoms: int, omega_laser: float) -> Qobj:
    """Return U(t) = exp(-i * omega_laser * N * t) matching the dims of `ref`.

    - Basis ordering: [0-ex], [1-ex (n_atoms states)], [2-ex (remainder)].
    - ρ_RWA(t) = U†(t) ρ_lab(t) U(t)
    - ρ_lab(t) = U(t) ρ_RWA(t) U†(t)
    """
    dim = ref.shape[0]
    n = _excitation_number_vector(dim, n_atoms)
    phases = np.exp(-1j * omega_laser * t * n)
    U_full = np.diag(phases)
    return Qobj(U_full, dims=ref.dims)


def to_rotating_frame_op(rho_lab: Qobj, t: float, n_atoms: int, omega_laser: float) -> Qobj:
    """Compute ρ_RWA(t) = U†(t) ρ_lab(t) U(t)."""
    U = rotating_frame_unitary(rho_lab, t, n_atoms, omega_laser)
    return U.dag() * rho_lab * U


def from_rotating_frame_op(rho_rot: Qobj, t: float, n_atoms: int, omega_laser: float) -> Qobj:
    """Compute ρ_lab(t) = U(t) ρ_RWA(t) U†(t)."""
    U = rotating_frame_unitary(rho_rot, t, n_atoms, omega_laser)
    return U * rho_rot * U.dag()


def to_rotating_frame_list(
    states: List[Qobj],
    times: np.ndarray,
    n_atoms: int,
    omega_laser: float,
) -> List[Qobj]:
    """Batch version of ρ_RWA = U† ρ_lab U with simple broadcasting.

    - times: scalar → apply same t to all states
    - times: array-like → must match len(states)
    """
    if not all(isinstance(s, Qobj) for s in states):
        raise TypeError("All states must be Qobj instances.")

    times_arr = np.asarray(times, dtype=float).reshape(-1)
    if len(times_arr) != len(states):
        raise ValueError(f"Length mismatch: {len(states)} states vs {times_arr.shape[0]} times")
    return [
        to_rotating_frame_op(rho, float(t), n_atoms, omega_laser)
        for rho, t in zip(states, times_arr)
    ]


def from_rotating_frame_list(
    states: List[Qobj],
    times: np.ndarray,
    n_atoms: int,
    omega_laser: float,
) -> List[Qobj]:
    """Batch version of ρ_lab = U ρ_RWA U† with simple broadcasting.

    - times: scalar → apply same t to all states
    - times: array-like → must match len(states)
    """
    if not all(isinstance(s, Qobj) for s in states):
        raise TypeError("All states must be Qobj instances.")

    times_arr = np.asarray(times, dtype=float).reshape(-1)
    if len(times_arr) != len(states):
        raise ValueError(f"Length mismatch: {len(states)} states vs {times_arr.shape[0]} times")
    return [
        from_rotating_frame_op(rho, float(t), n_atoms, omega_laser)
        for rho, t in zip(states, times_arr)
    ]


def get_expect_vals_with_RWA(
    states: List[Qobj],
    times: np.ndarray,
    n_atoms: int,
    e_ops: List[Qobj],
    omega_laser: float,
    rwa_sl: bool,
    dipole_op: Qobj = None,
) -> List[np.ndarray]:
    """
    Parameters:
        states: Results of the pulse evolution (data.states from qutip.Result)
        times: Time points at which the expectation values are calculated
        n_atoms: Number of atoms in the system
        e_ops: operators in the eigenbasis of H0
        omega_laser: Frequency of the laser
        rwa_sl: Whether to apply the RWA phase factors
        dipole_op: also in eigenbasis!

    Returns:
        List of arrays containing expectation values for each operator
    """
    if rwa_sl:
        # By default we assume stored states are in the rotating frame and we want lab-frame
        # expectation values. If you need the opposite, call `to_rotating_frame` explicitly
        # at the call site and pass rwa_sl=False here to avoid double transforms.
        states = from_rotating_frame_list(states, times - times[0], n_atoms, omega_laser)
    states_lab = states
    ## Calculate expectation values for each state and each operator
    updated_expects = []
    for e_op in e_ops:
        # Calculate expectation value for each state with this operator
        expect_vals = np.array(np.real(expect(e_op, states_lab)))
        updated_expects.append(expect_vals)
    if dipole_op is not None:
        # Import locally to avoid circular imports and depend directly on polarization module
        from qspectro2d.spectroscopy.polarization import complex_polarization

        # Calculate expectation value for the dipole operator if provided
        expect_vals_dip = np.array(complex_polarization(dipole_op, states_lab))
        updated_expects.append(expect_vals_dip)

    return updated_expects
