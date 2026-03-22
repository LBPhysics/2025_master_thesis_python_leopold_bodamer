"""Compute emitted-field components from phase-cycled polarisation signals."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Mapping

import numpy as np

from ..config.defaults import COMPONENT_MAP, phase_cycling_phases
from ..core.simulation.time_axes import compute_t_det
from .evolution import compute_polarisation_over_window, simulation_with_pulses

__all__ = ["compute_emitted_field_components"]


def _set_phase_subset(
    sim_subset,
    *,
    phi1: float,
    phi2: float,
    active_pulses: list[int] | None,
) -> None:
    """Set pulse phases consistently for a full or reduced pulse sequence."""
    if active_pulses is None:
        sim_subset.laser.pulse_phases = [phi1, phi2, 0.0]
        return

    if active_pulses == [0]:
        sim_subset.laser.pulse_phases = [phi1]
        return
    if active_pulses == [1]:
        sim_subset.laser.pulse_phases = [phi2]
        return
    if active_pulses == [2]:
        sim_subset.laser.pulse_phases = [0.0]
        return

    raise ValueError(f"Unsupported active_pulses selection: {active_pulses}")


def _worker_polarisation(
    config_source: Mapping[str, Any] | str,
    t_coh: float,
    freq_vector: list[float],
    phi1: float,
    phi2: float,
    detection_window: list[float],
    active_pulses: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Worker for one phase-cycling polarisation calculation."""
    from qspectro2d.config.factory import load_simulation

    sim_oqs = load_simulation(config_source)
    sim_oqs.update_delays(t_coh=t_coh)
    sim_oqs.system.update_frequencies_cm(freq_vector)
    sim_oqs.refresh_cache()

    run_sim = sim_oqs if active_pulses is None else simulation_with_pulses(sim_oqs, active_pulses)
    _set_phase_subset(run_sim, phi1=phi1, phi2=phi2, active_pulses=active_pulses)

    t_det = np.asarray(detection_window, dtype=float)
    _, polarisation = compute_polarisation_over_window(run_sim, t_det)
    return t_det, polarisation


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
            f"(sim_type={sim_type}, t_coh={t_coh_value:.6g}, "
            f"t_det={float(config.t_det):.6g}, dt={float(config.dt):.6g})"
        )

    signal_types = config.signal_types
    phase_values = phase_cycling_phases(config.n_phases)

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

    failed_jobs: list[str] = []

    total_tasks = len(phase_values) ** 2 + 2 * len(phase_values) + 1
    detection_window_list = t_det.tolist()

    with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
        futures = {}

        for phi1 in phase_values:
            for phi2 in phase_values:
                fut = executor.submit(
                    _worker_polarisation,
                    config_source,
                    t_coh_value,
                    list(freq_vector),
                    phi1,
                    phi2,
                    detection_window_list,
                    None,
                )
                futures[fut] = ("total", phi1, phi2)

        for phi1 in phase_values:
            fut = executor.submit(
                _worker_polarisation,
                config_source,
                t_coh_value,
                list(freq_vector),
                phi1,
                0.0,
                detection_window_list,
                [0],
            )
            futures[fut] = ("single_1", phi1, 0.0)

        for phi2 in phase_values:
            fut = executor.submit(
                _worker_polarisation,
                config_source,
                t_coh_value,
                list(freq_vector),
                0.0,
                phi2,
                detection_window_list,
                [1],
            )
            futures[fut] = ("single_2", 0.0, phi2)

        fut = executor.submit(
            _worker_polarisation,
            config_source,
            t_coh_value,
            list(freq_vector),
            0.0,
            0.0,
            detection_window_list,
            [2],
        )
        futures[fut] = ("single_3", 0.0, 0.0)

        for future in as_completed(futures):
            worker_kind, phi1_requested, phi2_requested = futures[future]
            try:
                _, polarisation = future.result()
            except Exception as exc:
                failed_jobs.append(
                    f"{worker_kind}(phi1={phi1_requested:.6g}, phi2={phi2_requested:.6g}) "
                    f"failed with {type(exc).__name__}: {exc}"
                )
                continue

            if worker_kind == "total":
                total_polarisations[(phi1_requested, phi2_requested)] = polarisation
            elif worker_kind == "single_1":
                single_pulse_1[phi1_requested] = polarisation
            elif worker_kind == "single_2":
                single_pulse_2[phi2_requested] = polarisation
            elif worker_kind == "single_3":
                single_pulse_3 = polarisation
            else:
                raise RuntimeError(f"Unexpected worker kind: {worker_kind}")

    if failed_jobs:
        details = "\n".join(failed_jobs[:10])
        more = "" if len(failed_jobs) <= 10 else f"\n... and {len(failed_jobs) - 10} more"
        print(
            "WARNING: Phase-cycling worker jobs failed; using zero contribution for "
            "missing phase combinations.\n"
            f"Failed jobs: {len(failed_jobs)} / {total_tasks}\n"
            f"{details}{more}"
        )

    missing_total = len(phase_values) ** 2 - len(total_polarisations)
    missing_single_1 = len(phase_values) - len(single_pulse_1)
    missing_single_2 = len(phase_values) - len(single_pulse_2)
    missing_single_3 = 0 if single_pulse_3 is not None else 1

    if missing_total or missing_single_1 or missing_single_2 or missing_single_3:
        print(
            "WARNING: Incomplete phase-cycling data after worker execution; "
            "missing components will be treated as zero. "
            f"missing total={missing_total}, "
            f"single_1={missing_single_1}, "
            f"single_2={missing_single_2}, "
            f"single_3={missing_single_3}"
        )

    for phi1 in phase_values:
        for phi2 in phase_values:
            p_total = total_polarisations.get((phi1, phi2), zero_component)
            p_sub_1 = single_pulse_1.get(phi1, zero_component)
            p_sub_2 = single_pulse_2.get(phi2, zero_component)
            p_sub_3 = single_pulse_3 if single_pulse_3 is not None else zero_component

            polarisation_component = p_total - p_sub_1 - p_sub_2 - p_sub_3

            for signal_type in signal_types:
                l_index, m_index = COMPONENT_MAP[signal_type] if lm is None else lm
                phase_factor = np.exp(-1j * (l_index * phi1 + m_index * phi2))
                accumulated[signal_type] += phase_factor * polarisation_component

    dphi = 2 * np.pi / len(phase_values)
    normalisation = (dphi / (2 * np.pi)) ** 2

    emitted_fields: list[np.ndarray] = []
    for signal_type in signal_types:
        field_component = 1j * accumulated[signal_type] * normalisation
        if time_mask is not None:
            field_component = field_component * time_mask
        emitted_fields.append(field_component)

    return emitted_fields
