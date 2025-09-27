"""Polarization related helper functions."""

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
      part (k = -1), which selects matrix elements connecting higher to lower
      energy states (|lower⟩⟨higher|).

    Accepts a single Qobj (ket or density matrix) or list of Qobj.
    """
    if isinstance(state, Qobj):
        return _single_qobj__complex_pol(dipole_op, state)
    if isinstance(state, list):
        if len(state) == 0:
            return np.array([], dtype=np.complex64)
        return np.array(
            [_single_qobj__complex_pol(dipole_op, s) for s in state], dtype=np.complex64
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
    # the strictly UPPER-triangular portion (i < j) in the energy eigenbasis.
    dipole_op_pos = Qobj(np.triu(dipole_op.full(), k=1), dims=dipole_op.dims)

    pol = expect(dipole_op_pos, rho)

    return complex(pol)


__all__ = ["complex_polarization"]
