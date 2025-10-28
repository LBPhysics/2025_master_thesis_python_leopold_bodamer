"""Polarization related helper functions.

Pphys(t) = Tr[μ ρ(t)] = ∑_{m,n} μ_{mn} ρ_{nm}(t)

but for spectroscopy are onlz interested in
P^+(t) = Tr[μ^+ ρ(t)] = ∑_{m>n} μ_{mn} ρ_{nm}(t)
"""

from __future__ import annotations
from typing import Union, List
import numpy as np
from qutip import Qobj, ket2dm, expect


def complex_polarization(
    dipole_op: Qobj, state: Union[Qobj, List[Qobj]]
) -> Union[complex, np.ndarray]:
    """Return complex/analytical polarization(s) P^(+)(t) for given state(s).

    Physics convention:
    - The positive-frequency part of the dipole operator corresponds to the
      lowering operator μ^(-) in the energy eigenbasis and carries the
      exp(-i ω t) time dependence relevant for emitted fields.
    - Therefore we project the dipole operator onto its strictly upper-triangular
      part (k = +1), which selects matrix elements connecting higher to lower
      energy states (|lower⟩⟨higher|).

    Accepts a single Qobj (ket or density matrix) or list of Qobj.
    """
    if isinstance(state, Qobj):
        return _single_qobj__complex_pol(dipole_op, state)
    if isinstance(state, list):
        if len(state) == 0:
            return np.array([], dtype=np.complex128)
        return np.array(
            [_single_qobj__complex_pol(dipole_op, s) for s in state], dtype=np.complex128
        )
    raise TypeError(f"State must be Qobj or list[Qobj], got {type(state)}")


def _single_qobj__complex_pol(dipole_op: Qobj, state: Qobj) -> complex:
    """
    Calculate polarization for a single quantum state or density matrix.

    Parameters
    ----------
    dipole_op : Qobj
        Dipole operator
    state : Qobj
        Quantum state (ket) or density matrix.

    Returns
    -------
    complex
        Complex polarization value.

    Raises
    ------
    TypeError
        If state is not a ket or density matrix.
    """
    rho = ket2dm(state) if state.isket else state
    # Positive-frequency part for this codebase's basis ordering corresponds to
    # the strictly LOWER-triangular portion (m > n) in the energy eigenbasis.
    # ~ sigma^+ e^[-iwt]
    dipole_op_pos = Qobj(np.tril(dipole_op.full(), k=-1), dims=dipole_op.dims)

    pol = expect(dipole_op_pos, rho)

    return complex(pol)


def time_dependent_polarization_rwa(
    dipole_op: Qobj, 
    state: Qobj, 
    t: float, 
    n_atoms: int, 
    carrier_freq_fs: float
) -> complex:
    """Compute time-dependent polarization with RWA correction."""
    from qspectro2d.utils.rwa_utils import from_rotating_frame_op
    state_lab = from_rotating_frame_op(state, t, n_atoms, carrier_freq_fs)
    return expect(dipole_op, state_lab)

__all__ = ["complex_polarization", "time_dependent_polarization_rwa"]

