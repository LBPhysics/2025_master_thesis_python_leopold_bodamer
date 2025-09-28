from __future__ import annotations
from typing import Union
import numpy as np

from .laser_class import LaserPulse, LaserPulseSequence


__all__ = [
    "pulse_envelopes",
    "e_pulses",
    "epsilon_pulses",
]


def single_pulse_envelope(t_array: np.ndarray, pulse: "LaserPulse") -> np.ndarray:
    """Compute envelope contribution of a single pulse for provided time array.

    Parameters
    ----------
    t_array : np.ndarray
        1D numpy array of times (already normalized from user input).
    pulse : LaserPulse
        Pulse instance providing cached invariants (_t_start/_t_end/_sigma/_boundary_val).

    Returns
    -------
    np.ndarray
        Envelope values for this single pulse over t_array.
    """
    t_peak = pulse.pulse_peak_time
    fwhm = pulse.pulse_fwhm_fs
    env = pulse.envelope_type

    out = np.zeros_like(t_array, dtype=float)

    # active mask using cached window (gaussian may be > ±FWHM if active_time_range wider)
    active = (t_array >= pulse._t_start) & (t_array <= pulse._t_end)
    if not np.any(active):
        return out

    t_act = t_array[active]
    if env == "cos2":
        arg = np.pi * (t_act - t_peak) / (2 * fwhm)
        out[active] = np.cos(arg) ** 2
    elif env == "gaussian":
        sigma = pulse._sigma
        boundary_val = pulse._boundary_val
        gauss = np.exp(-((t_act - t_peak) ** 2) / (2 * sigma**2))
        # subtract boundary baseline (ensures ~0 at stored window edges) then clamp
        out[active] = np.maximum(gauss - boundary_val, 0.0)
    else:
        raise ValueError(f"Unknown envelope_type: {env}. Use 'cos2' or 'gaussian'.")
    return out


def pulse_envelopes(
    t: Union[float, np.ndarray], pulse_seq: "LaserPulseSequence"
) -> Union[float, np.ndarray]:
    """
    Combined envelope (unitless) for pulses at time(s) t.
    Envelope semantics:
    - 'cos2': Compact support strictly inside [t_peak - FWHM, t_peak + FWHM]; zero outside.
    - 'gaussian': Finite-support approximation: active window extends to ± n_fwhm * FWHM (n_fwhm≈1.094)
       and a constant baseline equal to the Gaussian value at that EXTENDED edge is subtracted, then
       negative values clamped to zero. This preserves smooth Gaussian tails between ±FWHM and the
       extended edge while forcing the envelope ≈ 0 at the window boundaries.

    Args:
        t (Union[float, np.ndarray]): Time value or array of time values
        pulse_seq (LaserPulseSequence): The pulse sequence

    """
    # Import here to avoid circular imports
    from qspectro2d.core.laser_system.laser_class import LaserPulseSequence

    if not isinstance(pulse_seq, LaserPulseSequence):
        raise TypeError("pulse_seq must be a LaserPulseSequence instance.")

    # Normalize input to numpy array for vectorized operations
    t_array = np.asarray(t, dtype=float)
    is_scalar = t_array.ndim == 0
    if is_scalar:
        t_array = t_array[None]  # shape (1,)

    envelope_total = np.zeros_like(t_array, dtype=float)
    for pulse in pulse_seq.pulses:
        envelope_total += single_pulse_envelope(t_array, pulse)

    if is_scalar:
        return float(envelope_total[0])
    return envelope_total


def e_pulses(
    t: Union[float, np.ndarray], pulse_seq: "LaserPulseSequence"
) -> Union[complex, np.ndarray]:
    """Calculate RWA positive freq. electric field: E^(+) = E0 * exp(-i * phi) * envelopes."""
    from qspectro2d.core.laser_system.laser_class import LaserPulseSequence

    if not isinstance(pulse_seq, LaserPulseSequence):
        raise TypeError("pulse_seq must be a LaserPulseSequence instance.")

    t_array = np.asarray(t, dtype=float)
    is_scalar = t_array.ndim == 0
    if is_scalar:
        t_array = t_array[None]

    field_total = np.zeros_like(t_array, dtype=complex)
    for i in range(len(pulse_seq.pulses)):
        phi = pulse_seq.pulse_phases[i]
        E_amp = pulse_seq.pulse_amplitudes[i]
        # print(f"Pulse phase: {phi}, amplitude: {E_amp}")
        if phi is None or E_amp is None:
            continue
        single_env = single_pulse_envelope(t_array, pulse_seq.pulses[i])
        field_total += E_amp * single_env * np.exp(-1j * phi)

    if is_scalar:
        return complex(field_total[0])
    return field_total


def epsilon_pulses(
    t: Union[float, np.ndarray], pulse_seq: "LaserPulseSequence"
) -> Union[complex, np.ndarray]:
    """Calculate total positive freq. electric field: E^(+) = E0 * exp(-i * phi - i omega * t) * envelopes."""
    from qspectro2d.core.laser_system.laser_class import LaserPulseSequence

    if not isinstance(pulse_seq, LaserPulseSequence):
        raise TypeError("pulse_seq must be a LaserPulseSequence instance.")

    t_array = np.asarray(t, dtype=float)
    is_scalar = t_array.ndim == 0
    if is_scalar:
        t_array = t_array[None]

    field_total = np.zeros_like(t_array, dtype=complex)
    omega = pulse_seq.carrier_freq_fs
    carrier = np.exp(-1j * (omega * t_array)) * e_pulses(t_array, pulse_seq)
    field_total = carrier

    return field_total
