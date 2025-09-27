# Pulse and LaserPulseSequence classes for structured pulse handling

from dataclasses import dataclass, field  # for the class definiton
from typing import List, Tuple, Optional, Sequence

import numpy as np
from ...utils.constants import convert_cm_to_fs, convert_fs_to_cm

# Default Gaussian active window size in multiples of FWHM that roughly
# corresponds to ~1% envelope cutoff at the boundaries.
DEFAULT_ACTIVE_WINDOW_NFWHM: float = 1.094


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
            # Use extended active window (≈1% cutoff) defined by active_time_range (± n_fwhm * FWHM)
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

        Gaussian: uses ± DEFAULT_ACTIVE_WINDOW_NFWHM × FWHM (~1% boundary).
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
        # Keep pulses ordered in time at initialization; later updates preserve order
        # based on the sequence logic (pulse_delays are cumulative from 0).
        self.pulses.sort(key=lambda p: p.pulse_peak_time)

        # Peak amplitude of the first pulse (E0)
        if not self.pulses:
            self._E0 = 0.0
            self.carrier_freq_fs = None
        else:

            self._E0 = self.pulses[0].pulse_amplitude
            """Common carrier frequency in fs^-1 if all pulses share the same value. Else None."""
            first_fs = self.pulses[0].pulse_freq_fs
            if all(np.isclose(p.pulse_freq_fs, first_fs) for p in self.pulses):
                self.carrier_freq_fs = float(first_fs)

    @property
    def E0(self) -> float:
        """Peak amplitude of the first pulse (E0)."""
        return self._E0

    # --- Dynamic Properties ---
    @property
    def pulse_indices(self) -> List[int]:
        return [p.pulse_index for p in self.pulses]

    @property
    def pulse_peak_times(self) -> List[float]:
        return [p.pulse_peak_time for p in self.pulses]

    # --- Inter-pulse pulse_delays convenience -------------------------------------
    @property
    def pulse_delays(self) -> List[float]:
        """Inter-pulse pulse_delays Δt_i = t_{i} - t_{i-1} (length = n_pulses-1).

        Returns
        -------
        list[float]
            List of positive (or zero) pulse_delays between successive pulse peak times.

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
        """Set inter-pulse pulse_delays
        Parameters
        ----------
        new_pulse_delays : list[float]
            Must have length ``len(pulses)-1``. Represents the time differences
            between consecutive pulse peaks starting from t=0 for the first pulse.

        Raises
        ------
        ValueError
            If the length does not match ``n_pulses - 1``.

        Examples
        --------
        >>> seq.pulse_delays  # [d1, d2, ...]
        >>> ds = seq.pulse_delays; ds[0] = 40.0; seq.pulse_delays = ds  # modify first delay
        """
        # Accept any sequence convertible to list of floats
        if not isinstance(new_pulse_delays, (list, tuple, np.ndarray)):
            raise TypeError("new_pulse_delays must be a list/tuple/ndarray of floats")
        new_pulse_delays = list(map(float, list(new_pulse_delays)))

        # Validate length matches n_pulses - 1
        if len(new_pulse_delays) != len(self.pulses) - 1:
            raise ValueError(
                f"Number of pulse_delays ({len(new_pulse_delays)}) must be one less than number of pulses ({len(self.pulses)})"
            )

        # Ensure non-negative pulse_delays (cumulative schedule assumes forward-in-time spacing)
        if any(d < 0 for d in new_pulse_delays):
            raise ValueError("All pulse_delays must be non-negative.")

        # Compute absolute peak times from t0 = 0 fs and apply
        peak_times = np.insert(np.cumsum(new_pulse_delays), 0, 0.0)
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
    def pulse_fwhms(self) -> List[float]:
        return [p.pulse_fwhm_fs for p in self.pulses]

    @property
    def pulse_freqs_cm(self) -> List[float]:
        return [p.pulse_freq_cm for p in self.pulses]

    @property
    def pulse_freqs_fs(self) -> List[float]:
        return [p.pulse_freq_fs for p in self.pulses]

    @property
    def envelope_types(self) -> List[str]:
        return [p.envelope_type for p in self.pulses]

    @property
    def pulse_amplitudes(self) -> List[float]:
        return [p.pulse_amplitude for p in self.pulses]

    @property
    def carrier_freq_cm(self) -> Optional[float]:
        f_fs = self.carrier_freq_fs
        if f_fs is None:
            return None
        return float(convert_fs_to_cm(f_fs))

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
        n_pulses = len(pulse_delays) + 1
        if relative_E0s is None:
            relative_E0s = [1.0] * n_pulses
            relative_E0s[-1] = 0.1
        if phases is None:
            phases = [0.0] * n_pulses

        if not (len(relative_E0s) == len(phases) == n_pulses):
            raise ValueError("Lengths of pulse_delays, relative_E0s, and phases must match")

        peak_times = np.insert(np.cumsum(pulse_delays), 0, 0.0)

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

    def select_pulses(self, indices: Sequence[int]) -> "LaserPulseSequence":
        """
        Return a new LaserPulseSequence containing only the pulses at the given indices.

        Parameters
        ----------
        indices : sequence of int
            Indices of pulses to keep (0-based). Duplicates are allowed.

        Returns
        -------
        LaserPulseSequence
            A new instance containing only the selected pulses, sorted by peak time.
        """
        if not indices:
            return LaserPulseSequence(pulses=[])

        # Ensure indices are valid
        n = len(self.pulses)
        for i in indices:
            if i < 0 or i >= n:
                raise IndexError(f"Pulse index {i} out of range (0 <= i < {n})")

        # Select the pulses (deepcopy if you want independence)
        selected = [self.pulses[i] for i in indices]

        self.pulses = selected

    # --- Convenience ---
    def __len__(self):
        return len(self.pulses)

    def __getitem__(self, index):
        return self.pulses[index]

    def __iter__(self):
        return iter(self.pulses)

    def summary(self):
        print(str(self))

    def __str__(self) -> str:
        header = f"LaserPulseSequence Summary\n{'-' * 80}\n"
        return header + "\n".join(p.summary_line() for p in self.pulses)

    def to_dict(self) -> dict:
        return {
            "pulses": [p.to_dict() for p in self.pulses],
            "E0": self.E0,
            "carrier_freq_cm": self.carrier_freq_cm,
        }


'''
unused helper functions for pulses

    def get_active_pulses_at_time(self, time: float) -> List[LaserPulse]:
        """Return list of pulses active at given time (within their active_time_range)."""
        active_pulses: List[LaserPulse] = []
        for pulse in self.pulses:
            start_time, end_time = pulse.active_time_range
            if start_time <= time <= end_time:
                active_pulses.append(pulse)

        return active_pulses



    @staticmethod
    def from_general_specs(
        pulse_peak_times: Union[float, List[float]],
        pulse_phases: Union[float, List[float]],
        pulse_amplitudes: Union[float, List[float]],
        pulse_fwhms: Union[float, List[float]],
        pulse_freqs_cm: Union[float, List[float]],  # cm^-1 interface preserved
        envelope_types: Union[str, List[str]],
        pulse_indices: Optional[List[int]] = None,
    ) -> "LaserPulseSequence":
        if isinstance(pulse_peak_times, (float, int)):
            pulse_peak_times = [pulse_peak_times]

        n = len(pulse_peak_times)

        def expand(param, name):
            if isinstance(param, (float, int, str)):
                return [param] * n
            if isinstance(param, list):
                if len(param) != n:
                    raise ValueError(f"{name} must have length {n}")
                return param
            raise TypeError(f"{name} must be float, str, or list")

        pulse_phases = expand(pulse_phases, "pulse_phases")
        amps = expand(pulse_amplitudes, "pulse_amplitudes")
        fwhms = expand(pulse_fwhms, "pulse_fwhms")
        freqs_cm = expand(pulse_freqs_cm, "pulse_freqs_cm (cm^-1)")
        envs = expand(envelope_types, "envelope_types")

        if pulse_indices is None:
            pulse_indices = list(range(n))
        elif len(pulse_indices) != n:
            raise ValueError("pulse_indices must match number of pulses")

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


    def get_total_amplitude_at_time(self, time: float) -> float:
        """
        Calculate the total electric field amplitude (E0) at a given time.
        This is the sum of all active pulse amplitudes at that time.

        Parameters:
            time (float): The time at which to calculate the total amplitude

        Returns:
            float: Total electric field amplitude E0 = sum of all active pulse_amplitudes
        """
        active_pulses = self.get_active_pulses_at_time(time)
        total_amplitude = sum(pulse.pulse_amplitude for pulse in active_pulses)

        return total_amplitude



    # --- Additional dynamic update helpers ---------------------------------
    def set_absolute_peak_times(self, peak_times: List[float]) -> None:
        """Set absolute peak times for all pulses.

        The first pulse can be any time, but by convention sequences use 0 fs.
        Keeps internal envelope caches synchronized.
        """
        if len(peak_times) != len(self.pulses):
            raise ValueError(
                f"Number of peak times ({len(peak_times)}) must equal number of pulses ({len(self.pulses)})."
            )
        # Optionally enforce non-decreasing times to preserve order
        if any(t2 < t1 for t1, t2 in zip(peak_times, peak_times[1:])):
            raise ValueError("Peak times must be non-decreasing.")
        for pulse, t in zip(self.pulses, peak_times):
            pulse.pulse_peak_time = float(t)
            if hasattr(pulse, "_recompute_envelope_support"):
                pulse._recompute_envelope_support()

    def update_fwhms(self, fwhms_fs: List[float]) -> None:
        """Batch update FWHM values (fs) and refresh envelope caches."""
        if len(fwhms_fs) != len(self.pulses):
            raise ValueError(
                f"Number of FWHMs ({len(fwhms_fs)}) must equal number of pulses ({len(self.pulses)})."
            )
        for pulse, fwhm in zip(self.pulses, fwhms_fs):
            if fwhm <= 0:
                raise ValueError("FWHM must be positive.")
            pulse.pulse_fwhm_fs = float(fwhm)
            pulse._recompute_envelope_support()

    def update_amplitudes(self, amplitudes: List[float]) -> None:
        """Batch update pulse amplitudes (E0)."""
        if len(amplitudes) != len(self.pulses):
            raise ValueError(
                f"Number of amplitudes ({len(amplitudes)}) must equal number of pulses ({len(self.pulses)})."
            )
        for pulse, amp in zip(self.pulses, amplitudes):
            pulse.pulse_amplitude = float(amp)

    def update_envelope_types(self, envelope_types: List[str]) -> None:
        """Batch update envelope types and refresh caches."""
        if len(envelope_types) != len(self.pulses):
            raise ValueError(
                f"Number of envelope types ({len(envelope_types)}) must equal number of pulses ({len(self.pulses)})."
            )
        for pulse, env in zip(self.pulses, envelope_types):
            pulse.envelope_type = str(env)
            pulse._recompute_envelope_support()

    def update_frequencies_cm(self, freqs_cm: List[float]) -> None:
        """Batch update carrier frequencies in cm^-1 for all pulses."""
        if len(freqs_cm) != len(self.pulses):
            raise ValueError(
                f"Number of frequencies ({len(freqs_cm)}) must equal number of pulses ({len(self.pulses)})."
            )
        for pulse, f_cm in zip(self.pulses, freqs_cm):
            pulse.update_frequency_cm(float(f_cm))

            
    def get_field_info_at_time(self, time: float) -> dict:
        """
        Get comprehensive information about the electric field at a given time.

        Parameters:
            time (float): The time at which to analyze the field

        Returns:
            dict: Dictionary containing:
                - 'active_pulses': List of (pulse_index, pulse) tuples
                - 'num_active_pulses': Number of active pulses
                - 'total_amplitude': Total E0 = sum of active pulse amplitudes
                - 'individual_amplitudes': List of individual pulse amplitudes
                - 'pulse_indices': List of indices of active pulses
        """
        active = self.get_active_pulses_at_time(time)

        return {
            "active_pulses": active,
            "num_active_pulses": len(active),
            "total_amplitude": sum(p.pulse_amplitude for p in active),
            "individual_amplitudes": [p.pulse_amplitude for p in active],
            "pulse_indices": [p.pulse_index for p in active],
        }


        
###

def identify_non_zero_pulse_regions(times: np.ndarray, pulse_seq: LaserPulseSequence) -> np.ndarray:
    """
    Identify regions where the pulse envelope is non-zero across an array of time values.

    Args:
        times (np.ndarray): Array of time values to evaluate
        pulse_seq (LaserPulseSequence): The pulse sequence to evaluate

    Returns:
        np.ndarray: Boolean array where True indicates times where envelope is non-zero
    """

    if not isinstance(pulse_seq, LaserPulseSequence):
        raise TypeError("pulse_seq must be a LaserPulseSequence instance.")

    # Initialize an array of all False values
    active_regions = np.zeros_like(times, dtype=bool)

    # Vectorized over time array per pulse, then OR-reduce across pulses
    for pulse in pulse_seq.pulses:
        start_time, end_time = pulse.active_time_range
        active_regions |= (times >= start_time) & (times <= end_time)

    return active_regions


def split_by_active_regions(times: np.ndarray, active_regions: np.ndarray) -> List[np.ndarray]:
    """
    Split the time array into segments based on active regions.

    Args:
        times (np.ndarray): Array of time values.
        active_regions (np.ndarray): Boolean array indicating active regions.

    Returns:
        List[np.ndarray]: List of time segments split by active regions.
    """
    # Find where the active_regions changes value
    change_indices = np.where(np.diff(active_regions.astype(int)) != 0)[0] + 1

    # Split the times at those change points
    split_times = np.split(times, change_indices)

    # Return list of time segments
    return split_times

'''
