"""Compute emitted-field components from phase-cycled polarisation signals.

The emitted field is assembled using the photon-echo convention compatible
with the ``P^+(t)`` readout used in ``polarisation.py``. After phase selection,
this module returns field components proportional to ``-i P_sig(t)``.
"""

from __future__ import annotations

from concurrent.futures import Executor, ProcessPoolExecutor, as_completed
from typing import Any, Mapping

import numpy as np

from ..config.defaults import COMPONENT_MAP, phase_cycling_phases
from ..config.factory import load_simulation_config
from ..core.simulation.time_axes import compute_t_det
from .evolution import compute_polarisation_over_window

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

    sim_oqs = load_simulation(config_source, emit_runtime_warnings=False)
    sim_oqs.update_delays(t_coh=t_coh)
    sim_oqs.system.update_frequencies_cm(freq_vector)
    sim_oqs.refresh_cache()

    run_sim = sim_oqs if active_pulses is None else sim_oqs.with_pulse_subset(active_pulses)
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
    executor: Executor | None = None,
) -> tuple[list[np.ndarray], str, str | None]:
    """Compute emitted-field components for the configured signal types.

    Returns
    -------
    emitted_fields, run_status, status_message
        ``run_status`` is ``"ok"`` if all phase-cycling jobs completed,
        otherwise ``"phase_cycling_incomplete"``.
    """

    config = load_simulation_config(config_source, emit_runtime_warnings=False)

    detection_window_arr = np.asarray(
        compute_t_det(config) if detection_window is None else detection_window,
        dtype=float,
    )
    if detection_window_arr.ndim != 1:
        raise ValueError(f"detection_window must be 1D, got shape={detection_window_arr.shape}")

    component_count = int(detection_window_arr.size)
    sim_type = str(getattr(config, "sim_type", "1d"))
    t_coh_value = float(t_coh)

    if component_count == 0 and sim_type != "0d":
        raise RuntimeError(
            "Invariant violation: empty detection axis for non-0d run "
            f"(sim_type={sim_type}, t_coh={t_coh_value:.6g}, "
            f"t_det={float(config.t_det):.6g}, dt={float(config.dt):.6g})"
        )

    signal_types = list(config.signal_types)
    phase_values = [float(phi) for phi in phase_cycling_phases(config.n_phases)]
    freq_vector_list = [float(x) for x in freq_vector]
    freq_vector_debug = np.array2string(
        np.asarray(freq_vector_list, dtype=float),
        precision=12,
        separator=", ",
        max_line_width=1000,
    )

    time_mask = None
    if time_cut is not None and np.isfinite(time_cut):
        time_mask = (detection_window_arr <= float(time_cut)).astype(np.float64)

    zero_component = np.zeros(component_count, dtype=np.complex128)
    accumulated = {
        signal_type: np.zeros(component_count, dtype=np.complex128) for signal_type in signal_types
    }

    total_polarisations: dict[tuple[float, float], np.ndarray] = {}
    single_pulse_1: dict[float, np.ndarray] = {}
    single_pulse_2: dict[float, np.ndarray] = {}
    single_pulse_3: np.ndarray | None = None

    issues: list[str] = []

    def _result_key(
        worker_kind: str,
        phi1: float,
        phi2: float,
        active_pulses: list[int] | None,
    ) -> str:
        pulses_label: str | list[int] = "all" if active_pulses is None else list(active_pulses)
        return (
            f"{worker_kind}(phi1={phi1:.6g}, phi2={phi2:.6g}, "
            f"active_pulses={pulses_label})"
        )

    def _normalise_worker_result(
        worker_label: str,
        result: tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        _, polarisation = result
        polarisation = np.asarray(polarisation, dtype=np.complex128)

        if polarisation.shape != (component_count,):
            raise ValueError(
                f"{worker_label} returned shape "
                f"{polarisation.shape}, expected {(component_count,)}"
            )

        if not np.all(np.isfinite(polarisation)):
            first_bad = int(np.argmax(~np.isfinite(polarisation)))
            bad_t = (
                float(detection_window_arr[first_bad])
                if first_bad < component_count
                else float("nan")
            )
            raise ValueError(
                f"{worker_label} returned non-finite values "
                f"starting at index={first_bad}, t_det={bad_t:.6g} fs"
            )

        return polarisation

    job_specs: list[tuple[str, float, float, list[int] | None]] = []

    for phi1 in phase_values:
        for phi2 in phase_values:
            job_specs.append(("total", phi1, phi2, None))

    for phi1 in phase_values:
        job_specs.append(("single_1", phi1, 0.0, [0]))

    for phi2 in phase_values:
        job_specs.append(("single_2", 0.0, phi2, [1]))

    job_specs.append(("single_3", 0.0, 0.0, [2]))

    total_tasks = len(job_specs)
    max_workers = max(1, min(int(config.max_workers), total_tasks))
    detection_window_list = detection_window_arr.tolist()

    owns_executor = executor is None
    if owns_executor:
        executor = ProcessPoolExecutor(max_workers=max_workers)

    assert executor is not None
    try:
        futures: dict[Any, tuple[str, float, float, list[int] | None]] = {}

        for worker_kind, phi1, phi2, active_pulses in job_specs:
            fut = executor.submit(
                _worker_polarisation,
                config_source,
                t_coh_value,
                freq_vector_list,
                phi1,
                phi2,
                detection_window_list,
                active_pulses,
            )
            futures[fut] = (worker_kind, phi1, phi2, active_pulses)

        for future in as_completed(futures):
            worker_kind, phi1_requested, phi2_requested, active_pulses = futures[future]
            worker_label = (
                f"{_result_key(worker_kind, phi1_requested, phi2_requested, active_pulses)} "
                f"@ t_coh={t_coh_value:.6g}, freq_vector={freq_vector_debug}"
            )

            try:
                polarisation = _normalise_worker_result(
                    worker_label,
                    future.result(),
                )
            except Exception as exc:
                issues.append(f"{worker_label}: {exc}")
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
    finally:
        if owns_executor:
            executor.shutdown(wait=True)

    missing_total = len(phase_values) ** 2 - len(total_polarisations)
    missing_single_1 = len(phase_values) - len(single_pulse_1)
    missing_single_2 = len(phase_values) - len(single_pulse_2)
    missing_single_3 = 0 if single_pulse_3 is not None else 1

    incomplete = bool(
        issues or missing_total or missing_single_1 or missing_single_2 or missing_single_3
    )

    status_parts: list[str] = []

    if issues:
        details = "\n".join(issues[:10])
        more = "" if len(issues) <= 10 else f"\n... and {len(issues) - 10} more"
        status_parts.append(f"worker_issues={len(issues)}/{total_tasks}\n{details}{more}")

    if missing_total or missing_single_1 or missing_single_2 or missing_single_3:
        status_parts.append(
            f"missing total={missing_total}, "
            f"single_1={missing_single_1}, "
            f"single_2={missing_single_2}, "
            f"single_3={missing_single_3}"
        )

    status_message = " | ".join(status_parts) if status_parts else None

    if incomplete:
        print(
            "WARNING: Phase-cycling data incomplete or invalid; "
            "continuing with zero-filled missing components.\n"
            f"{status_message}",
            flush=True,
        )
        run_status = "phase_cycling_incomplete"
    else:
        run_status = "ok"

    p_sub_3 = single_pulse_3 if single_pulse_3 is not None else zero_component

    for phi1 in phase_values:
        p_sub_1 = single_pulse_1.get(phi1, zero_component)

        for phi2 in phase_values:
            p_total = total_polarisations.get((phi1, phi2), zero_component)
            p_sub_2 = single_pulse_2.get(phi2, zero_component)

            polarisation_component = p_total - p_sub_1 - p_sub_2 - p_sub_3

            for signal_type in signal_types:
                l_index, m_index = COMPONENT_MAP[signal_type] if lm is None else lm
                phase_factor = np.exp(-1j * (l_index * phi1 + m_index * phi2))
                accumulated[signal_type] += phase_factor * polarisation_component

    dphi = 2 * np.pi / len(phase_values)
    normalisation = (dphi / (2 * np.pi)) ** 2

    emitted_fields: list[np.ndarray] = []
    for signal_type in signal_types:
        field_component = -1j * accumulated[signal_type] * normalisation

        if time_mask is not None:
            field_component = field_component * time_mask

        if field_component.shape != (component_count,):
            raise RuntimeError(
                f"Final emitted field for '{signal_type}' has shape "
                f"{field_component.shape}, expected {(component_count,)}"
            )

        if not np.all(np.isfinite(field_component)):
            first_bad = int(np.argmax(~np.isfinite(field_component)))
            bad_t = float(detection_window_arr[first_bad])
            raise RuntimeError(
                f"Final emitted field for '{signal_type}' contains non-finite values "
                f"starting at index={first_bad}, t_det={bad_t:.6g} fs"
            )

        emitted_fields.append(field_component)

    return emitted_fields, run_status, status_message
