"""Compute emitted-field components from phase-cycled polarisation signals."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Mapping

import numpy as np

from ..config.defaults import COMPONENT_MAP, PHASE_CYCLING_PHASES
from ..core.simulation.time_axes import compute_t_det
from .evolution import compute_polarisation_over_window, simulation_with_pulses

__all__ = ["compute_emitted_field_components"]


def _worker_polarisation(
    config_source: Mapping[str, Any] | str,
    t_coh: float,
    freq_vector: list[float],
    phi1: float,
    phi2: float,
    active_pulses: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, float, list[float]]:
    from qspectro2d.config.factory import load_simulation

    sim_oqs = load_simulation(config_source)
    sim_oqs.update_delays(t_coh=t_coh)
    sim_oqs.system.update_frequencies_cm(freq_vector)
    sim_oqs.laser.pulse_phases = [phi1, phi2, 0.0]

    detection_window = compute_t_det(sim_oqs.simulation_config, t_coh_override=t_coh)
    if active_pulses is None:
        t_det, polarisation = compute_polarisation_over_window(sim_oqs, detection_window)
    else:
        sim_subset = simulation_with_pulses(sim_oqs, active_pulses)
        t_det, polarisation = compute_polarisation_over_window(sim_subset, detection_window)

    max_abs_polarisation = float(np.max(np.abs(polarisation))) if len(polarisation) else 0.0
    pulse_amplitudes = list(sim_oqs.laser.pulse_amplitudes)
    return (t_det, polarisation, max_abs_polarisation, pulse_amplitudes)


def compute_emitted_field_components(
    config_source: Mapping[str, Any] | str,
    t_coh: float,
    freq_vector: list[float],
    *,
    lm: tuple[int, int] | None = None,
    time_cut: float | None = None,
    detection_window: np.ndarray | list[float] | None = None,
) -> list[np.ndarray]:
    """Compute emitted-field components for the configured signal types."""
    from qspectro2d.config.factory import load_simulation_config

    config = load_simulation_config(config_source)
    t_coh_value = float(t_coh)

    if detection_window is None:
        t_det = compute_t_det(config, t_coh_override=t_coh_value)
    else:
        t_det = np.asarray(detection_window, dtype=float)

    if t_det.ndim != 1:
        raise ValueError("detection_window must be one-dimensional")

    sim_type = str(getattr(config, "sim_type", "1d"))
    if len(t_det) == 0 and sim_type != "0d":
        raise RuntimeError(
            "Invariant violation: empty detection axis for non-0d run "
            f"(sim_type={sim_type}, t_coh={t_coh_value:.6g}, t_det={float(config.t_det):.6g}, dt={float(config.dt):.6g})"
        )

    signal_types = config.signal_types
    component_count = len(t_det)

    time_mask = None
    if time_cut is not None and np.isfinite(time_cut):
        time_mask = (t_det <= time_cut).astype(np.float64)

    accumulated = {
        signal_type: np.zeros(component_count, dtype=np.complex128) for signal_type in signal_types
    }
    zero_component = np.zeros(component_count, dtype=np.complex128)

    total_polarisations: dict[tuple[float, float], np.ndarray] = {}
    single_pulse_1: dict[float, np.ndarray] = {}
    single_pulse_2: dict[float, np.ndarray] = {}
    single_pulse_3: np.ndarray | None = None

    successful_phase_pairs = 0
    failed_phase_pairs = 0
    failed_single_pulse_jobs = 0
    pulse_amplitudes: list[float] = []

    with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
        futures = {}
        for phi1 in PHASE_CYCLING_PHASES:
            for phi2 in PHASE_CYCLING_PHASES:
                futures[
                    executor.submit(
                        _worker_polarisation, config_source, t_coh, freq_vector, phi1, phi2
                    )
                ] = ("total", float(phi1), float(phi2))

        for phi1 in PHASE_CYCLING_PHASES:
            futures[
                executor.submit(
                    _worker_polarisation,
                    config_source,
                    t_coh,
                    freq_vector,
                    float(phi1),
                    0.0,
                    [0],
                )
            ] = ("single_1", float(phi1), 0.0)

        for phi2 in PHASE_CYCLING_PHASES:
            futures[
                executor.submit(
                    _worker_polarisation,
                    config_source,
                    t_coh,
                    freq_vector,
                    0.0,
                    float(phi2),
                    [1],
                )
            ] = ("single_2", 0.0, float(phi2))

        futures[
            executor.submit(_worker_polarisation, config_source, t_coh, freq_vector, 0.0, 0.0, [2])
        ] = ("single_3", 0.0, 0.0)

        for future in as_completed(futures):
            worker_kind, phi1_requested, phi2_requested = futures[future]
            try:
                _, polarisation, _, pulse_amplitudes_value = future.result()
            except Exception as exc:
                if worker_kind == "total":
                    failed_phase_pairs += 1
                else:
                    failed_single_pulse_jobs += 1
                print(
                    "Warning: phase-cycling worker failed; using zero contribution for this task."
                    f" kind={worker_kind},"
                    f" phi1={phi1_requested:.3f}, phi2={phi2_requested:.3f}, t_coh={t_coh:.3f} fs,"
                    f" error={type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue

            if not pulse_amplitudes:
                pulse_amplitudes = pulse_amplitudes_value

            if worker_kind == "total":
                total_polarisations[(phi1_requested, phi2_requested)] = polarisation
            elif worker_kind == "single_1":
                single_pulse_1[phi1_requested] = polarisation
            elif worker_kind == "single_2":
                single_pulse_2[phi2_requested] = polarisation
            else:
                single_pulse_3 = polarisation

        for phi1 in PHASE_CYCLING_PHASES:
            for phi2 in PHASE_CYCLING_PHASES:
                p_total = total_polarisations.get((float(phi1), float(phi2)))
                if p_total is None:
                    continue

                successful_phase_pairs += 1
                p_sub_1 = single_pulse_1.get(float(phi1), zero_component)
                p_sub_2 = single_pulse_2.get(float(phi2), zero_component)
                p_sub_3 = single_pulse_3 if single_pulse_3 is not None else zero_component
                p_sub_sum = p_sub_1 + p_sub_2 + p_sub_3
                polarisation_component = p_total - p_sub_sum

                for signal_type in signal_types:
                    component_indices = COMPONENT_MAP[signal_type] if lm is None else lm
                    l_index, m_index = component_indices
                    phase_factor = np.exp(-1j * (l_index * phi1 + m_index * phi2))
                    accumulated[signal_type] += phase_factor * polarisation_component

    dphi = np.diff(PHASE_CYCLING_PHASES).mean() if len(PHASE_CYCLING_PHASES) > 1 else 1.0
    normalisation_base = (dphi / (2 * np.pi)) ** 2
    total_phase_pairs = len(PHASE_CYCLING_PHASES) ** 2
    if successful_phase_pairs > 0:
        normalisation = normalisation_base * (total_phase_pairs / float(successful_phase_pairs))
    else:
        normalisation = 0.0

    emitted_fields: list[np.ndarray] = []
    for signal_type in signal_types:
        field_component = 1j * accumulated[signal_type] * normalisation
        if time_mask is not None:
            field_component = field_component * time_mask
        emitted_fields.append(field_component)
    return emitted_fields
