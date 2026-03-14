"""Compute emitted-field components from phase-cycled polarisation signals."""

from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from ..config.defaults import COMPONENT_MAP, PHASE_CYCLING_PHASES
from ..core.simulation.time_axes import compute_t_det
from .evolution import compute_polarisation_over_window, simulation_with_pulses

__all__ = ["compute_emitted_field_components"]


def _worker_phase_pair(
    config_path: str,
    t_coh: float,
    freq_vector: list[float],
    phi1: float,
    phi2: float,
) -> tuple[float, float, np.ndarray, np.ndarray, float, float, float, list[float]]:
    from qspectro2d.config.factory import load_simulation

    sim_oqs = load_simulation(config_path)
    sim_oqs.update_delays(t_coh=t_coh)
    sim_oqs.system.update_frequencies_cm(freq_vector)
    sim_oqs.laser.pulse_phases = [phi1, phi2, 0.0]

    detection_window = compute_t_det(sim_oqs.simulation_config)
    t_det, polarisation_total = compute_polarisation_over_window(sim_oqs, detection_window)

    polarisation_subtractions = np.zeros_like(polarisation_total, dtype=np.complex128)
    pulse_count = len(sim_oqs.laser.pulses)
    for active_set in itertools.combinations(range(pulse_count), 1):
        sim_subset = simulation_with_pulses(sim_oqs, list(active_set))
        _, polarisation_subset = compute_polarisation_over_window(sim_subset, detection_window)
        polarisation_subtractions += polarisation_subset

    polarisation_component = polarisation_total - polarisation_subtractions
    max_abs_total = float(np.max(np.abs(polarisation_total))) if len(polarisation_total) else 0.0
    max_abs_sub = float(np.max(np.abs(polarisation_subtractions))) if len(polarisation_subtractions) else 0.0
    max_abs_component = float(np.max(np.abs(polarisation_component))) if len(polarisation_component) else 0.0
    pulse_amplitudes = list(sim_oqs.laser.pulse_amplitudes)
    return (
        phi1,
        phi2,
        t_det,
        polarisation_component,
        max_abs_total,
        max_abs_sub,
        max_abs_component,
        pulse_amplitudes,
    )


def compute_emitted_field_components(
    config_path: str,
    t_coh: float,
    freq_vector: list[float],
    *,
    lm: tuple[int, int] | None = None,
    time_cut: float | None = None,
) -> list[np.ndarray]:
    """Compute emitted-field components for the configured signal types."""
    from qspectro2d.config.factory import load_simulation_config

    config = load_simulation_config(config_path)
    t_coh_value = float(t_coh)
    config.t_coh_current = t_coh_value
    if t_coh_value > float(config.t_coh_max):
        config.t_coh_max = t_coh_value

    t_det = compute_t_det(config)
    signal_types = config.signal_types
    component_count = len(t_det)

    time_mask = None
    if time_cut is not None and np.isfinite(time_cut):
        time_mask = (t_det <= time_cut).astype(np.float64)

    accumulated = {
        signal_type: np.zeros(component_count, dtype=np.complex128)
        for signal_type in signal_types
    }

    with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
        futures = {
            executor.submit(_worker_phase_pair, config_path, t_coh, freq_vector, phi1, phi2): (phi1, phi2)
            for phi1 in PHASE_CYCLING_PHASES
            for phi2 in PHASE_CYCLING_PHASES
        }
        failed_phase_pairs = 0
        successful_phase_pairs = 0
        for future in as_completed(futures):
            phi1_requested, phi2_requested = futures[future]
            try:
                (
                    phi1_value,
                    phi2_value,
                    _,
                    polarisation_component,
                    max_abs_total,
                    max_abs_sub,
                    max_abs_component,
                    pulse_amplitudes,
                ) = future.result()
            except Exception as exc:
                failed_phase_pairs += 1
                print(
                    "Warning: phase-pair worker failed; using zero contribution for this pair."
                    f" phi1={phi1_requested:.3f}, phi2={phi2_requested:.3f}, t_coh={t_coh:.3f} fs,"
                    f" error={type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue

            successful_phase_pairs += 1
            for signal_type in signal_types:
                component_indices = COMPONENT_MAP[signal_type] if lm is None else lm
                l_index, m_index = component_indices
                phase_factor = np.exp(-1j * (l_index * phi1_value + m_index * phi2_value))
                accumulated[signal_type] += phase_factor * polarisation_component
            if np.all(polarisation_component == 0):
                reason = ""
                if max_abs_total == 0.0 and max_abs_sub == 0.0:
                    reason = " (P_total and P_sub_sum are zero)"
                elif max_abs_component == 0.0 and max_abs_total > 0.0:
                    reason = " (P_total cancels P_sub_sum)"
                print(
                    "Warning: All zero P_phi detected!"
                    f" phi1={phi1_value:.3f}, phi2={phi2_value:.3f}, t_coh={t_coh:.3f} fs,"
                    f" max|P_total|={max_abs_total:.3e}, max|P_sub|={max_abs_sub:.3e},"
                    f" max|P_phi|={max_abs_component:.3e}, pulse_amps={pulse_amplitudes}{reason}"
                )

        if failed_phase_pairs:
            print(
                "Warning: phase cycling completed with failed workers."
                f" successful_pairs={successful_phase_pairs}, failed_pairs={failed_phase_pairs},"
                f" t_coh={t_coh:.3f} fs",
                flush=True,
            )

    dphi = np.diff(PHASE_CYCLING_PHASES).mean() if len(PHASE_CYCLING_PHASES) > 1 else 1.0
    normalisation_base = (dphi / (2 * np.pi)) ** 2
    total_phase_pairs = len(PHASE_CYCLING_PHASES) ** 2
    if successful_phase_pairs > 0:
        # Keep the same average scale even when some phase-pair workers failed.
        missing_pair_correction = total_phase_pairs / float(successful_phase_pairs)
        normalisation = normalisation_base * missing_pair_correction
    else:
        normalisation = 0.0

    emitted_fields: list[np.ndarray] = []
    for signal_type in signal_types:
        field_component = 1j * accumulated[signal_type] * normalisation
        if time_mask is not None:
            field_component = field_component * time_mask
        emitted_fields.append(field_component)
    return emitted_fields