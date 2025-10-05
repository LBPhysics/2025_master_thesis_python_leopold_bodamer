# Pulse and LaserPulseSequence classes for structured pulse handling

from dataclasses import dataclass, field  # for the class definiton
from typing import List, Tuple, Optional, Sequence

import numpy as np
from ...utils.constants import convert_cm_to_fs, convert_fs_to_cm

# Default Gaussian active window size in multiples of FWHM that roughly
# corresponds to ~0.01% envelope cutoff at the boundaries.
DEFAULT_ACTIVE_WINDOW_NFWHM: float = 1.823


@dataclass
class LaserPulse:
    """
    Represents a single optical pulse with its temporal and spectral properties.

    External (public) frequency unit:  cm^-1  (pulse_freq_cm)
    Internal (private) frequency unit: fs^-1  (_pulse_freq_fs)

    Access patterns:
        - Provide frequency in cm^-1 via pulse_freq_cm at construction
        - Use .pulse_freq_cm for human readable output / serialization
        - Use .pulse_freq_fs internally for dynamics
    """

    pulse_index: int
    pulse_peak_time: float  # [fs]
    pulse_phase: float  # [rad]
    pulse_fwhm_fs: float  # [fs]
    pulse_amplitude: float
    pulse_freq_cm: float  # user supplied central frequency [cm^-1]
    envelope_type: str = "gaussian"

    # internal cache (not part of __init__ signature)
    _pulse_freq_fs: float = field(init=False, repr=False)

    def __post_init__(self):

        # UNIT CONVERSION (single source of truth)
        self._pulse_freq_fs = float(convert_cm_to_fs(self.pulse_freq_cm))

        # PRECOMPUTE ENVELOPE SUPPORT
        self._recompute_envelope_support()

    def _recompute_envelope_support(self) -> None:
        """Recompute cached envelope window and Gaussian parameters.

        Must be called whenever attributes affecting timing change, e.g.,
        `pulse_peak_time`, `pulse_fwhm_fs`, or `envelope_type`.
        """
        if self.envelope_type == "gaussian":
            # Use extended active window (≈0.01% cutoff) defined by active_time_range (± n_fwhm * FWHM)
            self._t_start, self._t_end = self.active_time_range  # uses DEFAULT_ACTIVE_WINDOW_NFWHM
            self._sigma = self.pulse_fwhm_fs / (2 * np.sqrt(2 * np.log(2)))
            # Baseline value chosen at EXTENDED window edge (edge_span), not at FWHM, so
            # envelope retains smooth tails between ±FWHM and ±edge_span.
            edge_span = self._t_end - self.pulse_peak_time
            self._boundary_val = float(np.exp(-(edge_span**2) / (2 * self._sigma**2)))
        else:
            self._t_start = self.pulse_peak_time - self.pulse_fwhm_fs
            self._t_end = self.pulse_peak_time + self.pulse_fwhm_fs
            self._sigma = None
            self._boundary_val = None

    @property
    def pulse_freq_fs(self) -> float:
        """Internal frequency in fs^-1 (read-only)."""
        return self._pulse_freq_fs

    def update_frequency_cm(self, new_freq_cm: float) -> None:
        """Update the carrier frequency (cm^-1) and keep internal cache synchronized."""
        if new_freq_cm <= 0:
            raise ValueError("new_freq_cm must be positive")
        self.pulse_freq_cm = float(new_freq_cm)
        self._pulse_freq_fs = float(convert_cm_to_fs(self.pulse_freq_cm))

    @property
    def active_time_range(self) -> Tuple[float, float]:
        """Return (t_min, t_max) where the envelope is effectively non-zero.

        Gaussian: uses ± DEFAULT_ACTIVE_WINDOW_NFWHM × FWHM (~0.01% boundary).
        Other envelopes: uses ± FWHM around the peak.
        """
        if self.envelope_type == "gaussian":
            duration = DEFAULT_ACTIVE_WINDOW_NFWHM * self.pulse_fwhm_fs
        else:
            duration = self.pulse_fwhm_fs
        return (self.pulse_peak_time - duration, self.pulse_peak_time + duration)

    def summary_line(self) -> str:
        return (
            f"Pulse {self.pulse_index:>2}: "
            f"t = {self.pulse_peak_time:6.2f} fs | "
            f"E₀ = {self.pulse_amplitude:.3e} | "
            f"FWHM = {self.pulse_fwhm_fs:4.1f} fs | "
            f"ω = {self.pulse_freq_cm:8.2f} cm^-1 | "
            f"ϕ = {self.pulse_phase:6.3f} rad | "
            f"type = {self.envelope_type:<7}"
        )

    def to_dict(self):
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
    pulses: List[LaserPulse] = field(default_factory=list)

    def __post_init__(self):
        # Keep pulses ordered in time at initialization;
        self.pulses.sort(key=lambda p: p.pulse_peak_time)

        self._E0 = self.pulses[0].pulse_amplitude
        self.carrier_freq_fs = self.pulses[0].pulse_freq_fs
        self.carrier_fwhm_fs = self.pulses[0].pulse_fwhm_fs
        self.carrier_type = self.pulses[0].envelope_type

    @property
    def E0(self) -> float:
        """Reference amplitude of the first pulse (E0)."""
        return self._E0

    @property
    def pulse_amplitudes(self) -> List[float]:
        """List of pulse amplitudes."""
        return [p.pulse_amplitude for p in self.pulses]

    # --- Dynamic Properties ---
    @property
    def pulse_indices(self) -> List[int]:
        """List of pulse indices."""
        return [p.pulse_index for p in self.pulses]

    @property
    def pulse_peak_times(self) -> List[float]:
        """List of pulse peak times."""
        return [p.pulse_peak_time for p in self.pulses]

    # --- Inter-pulse pulse_delays convenience -------------------------------------
    @property
    def pulse_delays(self) -> List[float]:
        """Inter-pulse pulse_delays Δt_i = t_{i} - t_{i-1} (length = n_pulses-1).
        Notes
        -----
        This is a derived quantity from the sorted pulse_peak_times. For a
        single pulse an empty list is returned. To modify, assign to this
        property (``seq.pulse_delays = [...]``); this updates pulse peak times from
        t0 = 0 fs and keeps envelope caches synchronized.
        """
        pts = self.pulse_peak_times
        if len(pts) <= 1:
            return []
        return list(np.diff(pts))

    @pulse_delays.setter
    def pulse_delays(self, new_pulse_delays: List[float]) -> None:
        """Set inter-pulse delays.

        Interpretation (right-to-left):
        For pulses = [p1, p2, ..., pN], the last pulse pN occurs at t=0.
        Let delays = [d1, d2, ..., d_{N-1}] where d_k is the positive time
        between p_k and p_{k+1}. Then:
            t_{pN}      = 0
            t_{pN-1}    = -d_{N-1}
            t_{pN-2}    = -(d_{N-2} + d_{N-1})
            ...
            t_{p1}      = -(d_1 + d_2 + ... + d_{N-1})

        Parameters
        ----------
        new_pulse_delays : list[float]
            Must have length len(pulses)-1. Each entry is the time between
            consecutive pulses (earlier → later), all non-negative.
        """
        if not isinstance(new_pulse_delays, (list, tuple, np.ndarray)):
            raise TypeError("new_pulse_delays must be a list/tuple/ndarray of floats")
        new_pulse_delays = list(map(float, list(new_pulse_delays)))

        n = len(self.pulses)
        if len(new_pulse_delays) != n - 1:
            raise ValueError(
                f"Number of pulse_delays ({len(new_pulse_delays)}) must be one less than number of pulses ({n})"
            )

        # Build forward cumulative times [0, d1, d1+d2, ..., sum(d1..d_{n-1})]
        t_forward = np.insert(np.cumsum(new_pulse_delays), 0, 0.0)
        # Shift so the last pulse is at 0 → earlier pulses become negative
        peak_times = t_forward - t_forward[-1]

        # Apply to pulses
        for pulse, new_peak in zip(self.pulses, peak_times):
            pulse.pulse_peak_time = new_peak
            if hasattr(pulse, "_recompute_envelope_support"):
                pulse._recompute_envelope_support()

    @property
    def pulse_phases(self) -> List[float]:
        return [p.pulse_phase for p in self.pulses]

    @pulse_phases.setter
    def pulse_phases(self, phases: List[float]) -> None:
        """Set phases for the sequence.

        Accepts a sequence of phases that may be shorter than the number of
        pulses; only the first k pulses are updated. Values are cast to float.
        """
        if not isinstance(phases, (list, tuple, np.ndarray)):
            raise TypeError("phases must be a list/tuple/ndarray of floats")
        phases_list = list(map(float, list(phases)))
        k = min(len(phases_list), len(self.pulses))
        for i in range(k):
            self.pulses[i].pulse_phase = phases_list[i]

    @property
    def carrier_freq_cm(self) -> Optional[float]:
        return convert_fs_to_cm(self.carrier_freq_fs)

    # --- Factory Methods ---
    @staticmethod
    def from_pulse_delays(
        pulse_delays: List[float],
        base_amplitude: float = 1,
        pulse_fwhm_fs: float = 15.0,
        carrier_freq_cm: float = 16000.0,
        envelope_type: str = "gaussian",
        relative_E0s: Optional[List[float]] = None,
        phases: Optional[List[float]] = None,
    ) -> "LaserPulseSequence":
        """
        Construct a sequence with right-to-left timing:
        For n pulses, provide pulse_delays = [d1, d2, ..., d_{n-1}] with d_k >= 0.
        Then the absolute peak times are:
            t_n      =  0
            t_{n-1}  = -d_{n-1}
            t_{n-2}  = -(d_{n-2} + d_{n-1})
            ...
            t_1      = -(d_1 + ... + d_{n-1})
        """
        if any(d < 0 for d in pulse_delays):
            raise ValueError("All pulse_delays must be non-negative.")
        n_pulses = len(pulse_delays) + 1
        if relative_E0s is None:
            relative_E0s = [1.0] * n_pulses
            relative_E0s[-1] = 0.1
        if phases is None:
            phases = [0.0] * n_pulses

        if not (len(relative_E0s) == len(phases) == n_pulses):
            raise ValueError("Lengths of pulse_delays, relative_E0s, and phases must match")

        # Build forward cumulative times [0, d1, d1+d2, ..., sum(d1..d_{n-1})]
        t_forward = np.insert(np.cumsum(pulse_delays), 0, 0.0)
        # Shift so the last pulse is at 0 → earlier pulses become negative
        peak_times = t_forward - t_forward[-1]

        pulses = [
            LaserPulse(
                pulse_index=i,
                pulse_peak_time=peak_times[i],
                pulse_phase=phases[i],
                pulse_fwhm_fs=pulse_fwhm_fs,
                pulse_freq_cm=carrier_freq_cm,
                pulse_amplitude=base_amplitude * relative_E0s[i],
                envelope_type=envelope_type,
            )
            for i in range(n_pulses)
        ]

        return LaserPulseSequence(pulses=pulses)

    def select_pulses(self, indices: Sequence[int]) -> None:
        """Restruct the LaserPulseSequence to only the pulses at the given indices."""
        # Ensure indices are valid
        n = len(self.pulses)
        for i in indices:
            if i < 0 or i >= n:
                raise IndexError(f"Pulse index {i} out of range (0 <= i < {n})")

        # Select the pulses (deepcopy if you want independence)
        selected = [self.pulses[i] for i in indices]

        self.pulses = selected

    # Output
    def summary(self):
        print(str(self))

    def __str__(self) -> str:
        header = f"LaserPulseSequence Summary\n{'-' * 80}\n"
        return header + "\n".join(p.summary_line() for p in self.pulses)

    def to_dict(self) -> dict:
        return {
            "LASER": "",
            "E_0": self.E0,
            "w_L": self.carrier_freq_cm,
            "FWHM": self.carrier_fwhm_fs,
            "env": self.carrier_type,
        }
