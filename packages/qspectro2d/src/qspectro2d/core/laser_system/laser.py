"""Laser pulse data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ...utils.constants import convert_cm_to_fs, convert_fs_to_cm

DEFAULT_ACTIVE_WINDOW_NFWHM: float = 1.823


@dataclass
class LaserPulse:
    pulse_index: int
    pulse_peak_time: float
    pulse_phase: float
    pulse_fwhm_fs: float
    pulse_amplitude: float
    pulse_freq_cm: float
    envelope_type: str = "gaussian"

    _pulse_freq_fs: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._pulse_freq_fs = float(convert_cm_to_fs(self.pulse_freq_cm))
        self._recompute_envelope_support()

    def _recompute_envelope_support(self) -> None:
        if self.envelope_type == "gaussian":
            self._t_start, self._t_end = self.active_time_range
            self._sigma = self.pulse_fwhm_fs / (2 * np.sqrt(2 * np.log(2)))
            edge_span = self._t_end - self.pulse_peak_time
            self._boundary_val = float(np.exp(-(edge_span**2) / (2 * self._sigma**2)))
            return
        if self.envelope_type == "delta":
            self._t_start = self.pulse_peak_time
            self._t_end = self.pulse_peak_time
            self._sigma = None
            self._boundary_val = None
            return
        self._t_start = self.pulse_peak_time - self.pulse_fwhm_fs
        self._t_end = self.pulse_peak_time + self.pulse_fwhm_fs
        self._sigma = None
        self._boundary_val = None

    @property
    def pulse_freq_fs(self) -> float:
        return self._pulse_freq_fs

    def update_frequency_cm(self, new_freq_cm: float) -> None:
        if new_freq_cm <= 0:
            raise ValueError("new_freq_cm must be positive")
        self.pulse_freq_cm = float(new_freq_cm)
        self._pulse_freq_fs = float(convert_cm_to_fs(self.pulse_freq_cm))

    @property
    def active_time_range(self) -> tuple[float, float]:
        if self.envelope_type == "gaussian":
            duration = DEFAULT_ACTIVE_WINDOW_NFWHM * self.pulse_fwhm_fs
        elif self.envelope_type == "delta":
            return (self.pulse_peak_time, self.pulse_peak_time)
        else:
            duration = self.pulse_fwhm_fs
        return (self.pulse_peak_time - duration, self.pulse_peak_time + duration)

    def summary_line(self) -> str:
        fwhm_str = "N/A" if self.envelope_type == "delta" else f"{self.pulse_fwhm_fs:4.1f} fs"
        return (
            f"Pulse {self.pulse_index:>2}: "
            f"t = {self.pulse_peak_time:6.2f} fs | "
            f"E0 = {self.pulse_amplitude:.3e} | "
            f"FWHM = {fwhm_str} | "
            f"w = {self.pulse_freq_cm:8.2f} cm^-1 | "
            f"phi = {self.pulse_phase:6.3f} rad | "
            f"type = {self.envelope_type:<7}"
        )

    def to_dict(self) -> dict:
        return {
            "pulse_index": self.pulse_index,
            "pulse_peak_time": self.pulse_peak_time,
            "pulse_phase": self.pulse_phase,
            "pulse_fwhm_fs": self.pulse_fwhm_fs,
            "pulse_amplitude": self.pulse_amplitude,
            "pulse_freq_cm": self.pulse_freq_cm,
            "envelope_type": self.envelope_type,
        }


@dataclass
class LaserPulseSequence:
    pulses: list[LaserPulse] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.pulses.sort(key=lambda pulse: pulse.pulse_peak_time)
        if not self.pulses:
            raise ValueError("LaserPulseSequence requires at least one pulse")
        self._E0 = self.pulses[0].pulse_amplitude
        self.carrier_freq_fs = self.pulses[0].pulse_freq_fs
        self.carrier_fwhm_fs = self.pulses[0].pulse_fwhm_fs
        self.carrier_type = self.pulses[0].envelope_type

    @property
    def E0(self) -> float:
        return self._E0

    @property
    def pulse_amplitudes(self) -> list[float]:
        return [pulse.pulse_amplitude for pulse in self.pulses]

    @property
    def pulse_indices(self) -> list[int]:
        return [pulse.pulse_index for pulse in self.pulses]

    @property
    def pulse_peak_times(self) -> list[float]:
        return [pulse.pulse_peak_time for pulse in self.pulses]

    @property
    def pulse_delays(self) -> list[float]:
        if len(self.pulses) <= 1:
            return []
        return list(np.diff(self.pulse_peak_times))

    @pulse_delays.setter
    def pulse_delays(self, new_pulse_delays: list[float]) -> None:
        if not isinstance(new_pulse_delays, (list, tuple, np.ndarray)):
            raise TypeError("new_pulse_delays must be a list/tuple/ndarray of floats")
        delays = list(map(float, list(new_pulse_delays)))
        if len(delays) != len(self.pulses) - 1:
            raise ValueError(
                f"Number of pulse_delays ({len(delays)}) must be one less than number of pulses ({len(self.pulses)})"
            )
        t_forward = np.insert(np.cumsum(delays), 0, 0.0)
        peak_times = t_forward - t_forward[-1]
        for pulse, new_peak in zip(self.pulses, peak_times):
            pulse.pulse_peak_time = float(new_peak)
            pulse._recompute_envelope_support()

    @property
    def pulse_phases(self) -> list[float]:
        return [pulse.pulse_phase for pulse in self.pulses]

    @pulse_phases.setter
    def pulse_phases(self, phases: list[float]) -> None:
        if not isinstance(phases, (list, tuple, np.ndarray)):
            raise TypeError("phases must be a list/tuple/ndarray of floats")
        for pulse, phase in zip(self.pulses, map(float, list(phases))):
            pulse.pulse_phase = phase

    @property
    def carrier_freq_cm(self) -> Optional[float]:
        return convert_fs_to_cm(self.carrier_freq_fs)

    @staticmethod
    def from_pulse_delays(
        pulse_delays: list[float],
        base_amplitude: float = 1,
        pulse_fwhm_fs: float = 15.0,
        carrier_freq_cm: float = 16000.0,
        envelope_type: str = "gaussian",
        pulse_amplitudes: Optional[list[float]] = None,
        phases: Optional[list[float]] = None,
    ) -> "LaserPulseSequence":
        if any(delay < 0 for delay in pulse_delays):
            raise ValueError("All pulse_delays must be non-negative.")
        n_pulses = len(pulse_delays) + 1
        pulse_amplitudes = pulse_amplitudes or [float(base_amplitude)] * n_pulses
        phases = phases or [0.0] * n_pulses
        if not (len(pulse_amplitudes) == len(phases) == n_pulses):
            raise ValueError("Lengths of pulse_delays, pulse_amplitudes, and phases must match")
        t_forward = np.insert(np.cumsum(pulse_delays), 0, 0.0)
        peak_times = t_forward - t_forward[-1]
        pulses = [
            LaserPulse(
                pulse_index=index,
                pulse_peak_time=peak_times[index],
                pulse_phase=phases[index],
                pulse_fwhm_fs=pulse_fwhm_fs,
                pulse_freq_cm=carrier_freq_cm,
                pulse_amplitude=pulse_amplitudes[index],
                envelope_type=envelope_type,
            )
            for index in range(n_pulses)
        ]
        return LaserPulseSequence(pulses=pulses)

    def subset(self, indices: Sequence[int]) -> "LaserPulseSequence":
        n_pulses = len(self.pulses)
        for index in indices:
            if index < 0 or index >= n_pulses:
                raise IndexError(f"Pulse index {index} out of range (0 <= i < {n_pulses})")
        return LaserPulseSequence(pulses=[self.pulses[index] for index in indices])

    def select_pulses(self, indices: Sequence[int]) -> None:
        """Select only the pulses at given indices (in-place mutation). Use subset() for non-mutating alternative."""
        n = len(self.pulses)
        for i in indices:
            if i < 0 or i >= n:
                raise IndexError(f"Pulse index {i} out of range (0 <= i < {n})")
        self.pulses = [self.pulses[i] for i in indices]

    def summary(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return "LaserPulseSequence Summary\n" + "-" * 80 + "\n" + "\n".join(
            pulse.summary_line() for pulse in self.pulses
        )

    def to_dict(self) -> dict:
        return {
            "E_0": self.E0,
            "w_L": self.carrier_freq_cm,
            "FWHM": self.carrier_fwhm_fs,
            "env": self.carrier_type,
        }


__all__ = ["DEFAULT_ACTIVE_WINDOW_NFWHM", "LaserPulse", "LaserPulseSequence"]
