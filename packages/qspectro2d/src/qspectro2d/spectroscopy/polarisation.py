"""Polarisation related helper functions.

Pphys(t) = Tr[μ ρ(t)] = ∑_{m,n} μ_{mn} ρ_{nm}(t)

but for emission spectroscopy we are interested in
P^-(t) = Tr[μ^- ρ(t)] = ∑_{m>n} μ_{mn} ρ_{nm}(t)
"""

from __future__ import annotations

from typing import List, Union

import numpy as np
from qutip import Qobj, expect, ket2dm


def complex_polarisation(
    dipole_op: Qobj, state: Union[Qobj, List[Qobj]]
) -> Union[complex, np.ndarray]:
    """Return complex/analytical polarisation(s) P^(-)(t) for given state(s).

    Physics convention:
        - The negative-frequency part of the dipole operator corresponds to
            μ^(-) in this codebase's basis ordering and is represented by the
            strictly upper-triangular part (m < n), selecting |lower⟩⟨higher|.
        - Used for emission spectroscopy: P^(-)(t) ~ exp(-iωt).

    Accepts a single Qobj (ket or density matrix) or list of Qobj.
    """
    if isinstance(state, Qobj):
        return _complex_polarisation_single(dipole_op, state)
    if isinstance(state, list):
        if len(state) == 0:
            return np.array([], dtype=np.complex128)
        return np.array(
            [_complex_polarisation_single(dipole_op, s) for s in state], dtype=np.complex128
        )
    raise TypeError(f"State must be Qobj or list[Qobj], got {type(state)}")


def _complex_polarisation_single(dipole_op: Qobj, state: Qobj) -> complex:
    """
    Calculate polarisation for a single quantum state or density matrix.

    Parameters
    ----------
    dipole_op : Qobj
        Dipole operator
    state : Qobj
        Quantum state (ket) or density matrix.

    Returns
    -------
    complex
        Complex polarisation value.

    Raises
    ------
    TypeError
        If state is not a ket or density matrix.
    """
    rho = ket2dm(state) if state.isket else state
    dipole_op_pos = Qobj(np.tril(dipole_op.full(), k=-1), dims=dipole_op.dims)

    pol = expect(dipole_op_pos, rho)

    return complex(pol)


def time_dependent_polarisation_rwa(
    dipole_op: Qobj,
    state: Qobj,
    t: float,
    n_atoms: int,
    carrier_freq_fs: float,
) -> complex:
    """Compute time-dependent polarisation with RWA correction."""
    from qspectro2d.utils.rwa_utils import from_rotating_frame_op

    state_lab = from_rotating_frame_op(state, t, n_atoms, carrier_freq_fs)
    return complex_polarisation(dipole_op, state_lab)

__all__ = ["complex_polarisation", "time_dependent_polarisation_rwa"]

