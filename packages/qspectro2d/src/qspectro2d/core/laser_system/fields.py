"""Laser field evaluation helpers."""

from __future__ import annotations

from typing import Union

import numpy as np

from .laser import LaserPulse, LaserPulseSequence

__all__ = ["pulse_envelopes", "e_pulses", "epsilon_pulses", "single_pulse_envelope"]


def single_pulse_envelope(t_array: np.ndarray, pulse: LaserPulse) -> np.ndarray:
    """Compute a single pulse envelope on the given time grid.

    The returned envelope is centered at ``pulse.pulse_peak_time`` and truncated
    to the active window ``[pulse._t_start, pulse._t_end]``.
    """
    t_peak = pulse.pulse_peak_time
    fwhm = pulse.pulse_fwhm_fs
    envelope_type = pulse.envelope_type

    result = np.zeros_like(t_array, dtype=float)
    active = (t_array >= pulse._t_start) & (t_array <= pulse._t_end)
    if not np.any(active):
        return result

    active_times = t_array[active]
    if envelope_type == "cos2":
        arg = np.pi * (active_times - t_peak) / (2 * fwhm)
        result[active] = np.cos(arg) ** 2
    elif envelope_type == "gaussian":
        gaussian = np.exp(-((active_times - t_peak) ** 2) / (2 * pulse._sigma**2))
        result[active] = np.maximum(gaussian - pulse._boundary_val, 0.0)
    else:
        raise ValueError(f"Unknown envelope_type: {envelope_type}. Use 'cos2' or 'gaussian'.")

    return result


def pulse_envelopes(
    t: Union[float, np.ndarray], pulse_seq: LaserPulseSequence
) -> Union[float, np.ndarray]:
    """Return the combined real envelope of all pulses."""
    t_array = np.asarray(t, dtype=float)
    is_scalar = t_array.ndim == 0
    if is_scalar:
        t_array = t_array[None]

    total = np.zeros_like(t_array, dtype=float)
    for pulse in pulse_seq.pulses:
        total += single_pulse_envelope(t_array, pulse)

    return float(total[0]) if is_scalar else total


def e_pulses(
    t: Union[float, np.ndarray], pulse_seq: LaserPulseSequence
) -> Union[complex, np.ndarray]:
    """
    Return the slowly varying field envelope ``E(t)`` used in the paper-style RWA.

    Following Mančal/Pisliakov (2006), the lab-frame positive-frequency field is

        epsilon^(+)(t) = exp(-i omega_L t) E(t),

    with
        E(t) = sum_k A_k s_k(t) exp(-i phi_k),

    where the pulse timing is carried by the centred envelope ``s_k(t)``.
    """
    t_array = np.asarray(t, dtype=float)
    is_scalar = t_array.ndim == 0
    if is_scalar:
        t_array = t_array[None]

    field_total = np.zeros_like(t_array, dtype=complex)
    for pulse, phi_k, A_k in zip(
        pulse_seq.pulses,
        pulse_seq.pulse_phases,
        pulse_seq.pulse_amplitudes,
    ):
        envelope = single_pulse_envelope(t_array, pulse)
        field_total += A_k * np.exp(-1j * phi_k) * envelope

    return field_total[0] if is_scalar else field_total


def epsilon_pulses(
    t: Union[float, np.ndarray], pulse_seq: LaserPulseSequence
) -> Union[complex, np.ndarray]:
    """Return the lab-frame positive-frequency field ``epsilon^(+)(t)``."""
    t_array = np.asarray(t, dtype=float)
    is_scalar = t_array.ndim == 0
    if is_scalar:
        t_array = t_array[None]

    field = np.exp(-1j * pulse_seq.carrier_freq_fs * t_array) * e_pulses(t_array, pulse_seq)
    return field[0] if is_scalar else field
