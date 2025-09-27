"""Core physical constants and unit conversion helpers.

Lightweight module: safe to import from any layer without triggering
expensive or circular imports. Keep ONLY primitive constants and pure
functions here.
"""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np

__all__ = [
    "HBAR",
    "BOLTZMANN",
    "convert_cm_to_fs",
    "convert_fs_to_cm",
]


# FUNDAMENTAL CONSTANTS (natural units inside project)

HBAR: float = 1.0  # Reduced Planck constant
BOLTZMANN: float = 1.0  # Boltzmann constant

_C_CM_PER_FS: float = 2.998  # speed of light factor in (1e-5 * cm/fs)
_TWOPI: float = 2 * np.pi
_CM_TO_FS_FACTOR: float = _C_CM_PER_FS * _TWOPI * 1e-5
_FS_TO_CM_FACTOR: float = 1.0 / _CM_TO_FS_FACTOR


def _is_qutip_qobj(x: Any) -> bool:
    """Return True if ``x`` looks like a QuTiP ``Qobj`` without importing qutip.

    We avoid importing heavy dependencies at module import time. Detection is
    done via duck-typing on class name and module path.
    """
    cls = x.__class__
    mod = getattr(cls, "__module__", "")
    return getattr(cls, "__name__", "") == "Qobj" and mod.startswith("qutip")


def _apply_conversion(obj: Any, factor: float) -> Any:
    """Apply a scalar conversion factor to scalars, arrays, sequences, or Qobj.

    Supported inputs:
    - float/int/np.scalar → float
    - numpy.ndarray → numpy.ndarray (float dtype)
    - Sequence[Number] → numpy.ndarray (float dtype)
    - QuTiP Qobj (detected lazily) → Qobj (scalar-multiplied)
    """
    # Scalars
    if isinstance(obj, (int, float, np.integer, np.floating)):
        return float(obj) * factor

    # numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.astype(float, copy=False) * factor

    # lightweight detection of QuTiP's Qobj without importing it
    if _is_qutip_qobj(obj):
        return obj * factor  # Qobj supports scalar multiplication

    # Generic sequences (lists/tuples) → return ndarray
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return np.asarray(obj, dtype=float) * factor

    raise TypeError(f"Unsupported type for conversion: {type(obj)!r}")


def convert_cm_to_fs(obj: Any) -> Any:
    """Convert wavenumber values (cm^-1) to angular frequency (fs^-1).

    Accepted input types:
    - float/int/np.scalar → returns float
    - numpy.ndarray → returns numpy.ndarray
    - Sequence of numbers → returns numpy.ndarray
    - QuTiP Qobj → returns Qobj (scaled)
    """
    return _apply_conversion(obj, _CM_TO_FS_FACTOR)


def convert_fs_to_cm(obj: Any) -> Any:
    """Convert angular frequency values (fs^-1) to wavenumber (cm^-1).

    Accepted input types:
    - float/int/np.scalar → returns float
    - numpy.ndarray → returns numpy.ndarray
    - Sequence of numbers → returns numpy.ndarray
    - QuTiP Qobj → returns Qobj (scaled)
    """
    return _apply_conversion(obj, _FS_TO_CM_FACTOR)
