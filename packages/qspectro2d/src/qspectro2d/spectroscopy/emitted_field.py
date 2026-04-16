"""Compute emitted-field components from phase-cycled polarisation signals.

The emitted field is assembled using the photon-echo convention compatible
with the ``P^+(t)`` readout used in ``polarisation.py``. After phase selection,
this module returns field components proportional to ``-i P_sig(t)``.
"""

from __future__ import annotations

import json
from copy import deepcopy
from concurrent.futures import Executor, as_completed
from typing import Any, Mapping

import numpy as np

from ..config import resolve_config
from ..config.defaults import COMPONENT_MAP, phase_cycling_phases
from ..config.factory import load_simulation, load_simulation_config
from ..core.simulation.time_axes import compute_t_det
from ..utils.phase_pool import create_phase_pool_executor, resolve_phase_pool_worker_count
from .evolution import compute_polarisation_over_window

__all__ = ["compute_emitted_field_components"]


_WORKER_RESOLVED_CONFIG_CACHE: dict[str, Mapping[str, Any]] = {}
_WORKER_SIMULATION_CACHE: dict[tuple[str, str | None], Any] = {}


def _clear_worker_caches() -> None:
    """Clear worker-local caches.

    This is mainly intended for tests; production workers reuse these caches
    for the lifetime of the process.
    """
    _WORKER_RESOLVED_CONFIG_CACHE.clear()
    _WORKER_SIMULATION_CACHE.clear()


def _config_cache_key(config_source: Mapping[str, Any] | str) -> str:
    if isinstance(config_source, str):
        return f"path:{config_source}"
    payload = json.dumps(config_source, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f"mapping:{payload}"


def _resolved_worker_config(config_source: Mapping[str, Any] | str) -> tuple[str, Mapping[str, Any]]:
    config_key = _config_cache_key(config_source)
    cached_cfg = _WORKER_RESOLVED_CONFIG_CACHE.get(config_key)
    if cached_cfg is None:
        cached_cfg = resolve_config(config_source, emit_runtime_warnings=False)
        _WORKER_RESOLVED_CONFIG_CACHE[config_key] = cached_cfg
    return config_key, cached_cfg


def _simulation_source_for_method(
    resolved_cfg: Mapping[str, Any],
    method_override: str | None,
) -> Mapping[str, Any]:
    if method_override is None:
        return resolved_cfg

    cfg_with_override = deepcopy(dict(resolved_cfg))
    cfg_with_override.setdefault("config", {}).setdefault("solver_options", {})["method"] = (
        method_override
    )
    return cfg_with_override


def _cached_worker_run_sim(
    config_source: Mapping[str, Any] | str,
    *,
    method_override: str | None,
):
    config_key, resolved_cfg = _resolved_worker_config(config_source)
    cache_key = (config_key, method_override)

    cached_sim = _WORKER_SIMULATION_CACHE.get(cache_key)
    if cached_sim is not None:
        return cached_sim

    simulation_source = _simulation_source_for_method(resolved_cfg, method_override)
    run_sim = load_simulation(simulation_source, emit_runtime_warnings=False)
    _WORKER_SIMULATION_CACHE[cache_key] = run_sim
    return run_sim


def _update_worker_frequencies(system, freq_vector: list[float]) -> None:
    """Update worker-local frequencies without unbounded history growth.

    Some environments may still import an older ``AtomicSystem`` signature that
    lacks the ``track_history`` keyword, so we keep a compatible fallback.
    """
    try:
        system.update_frequencies_cm(freq_vector, track_history=False)
        return
    except TypeError as exc:
        if "track_history" not in str(exc):
            raise

    system.update_frequencies_cm(freq_vector)
    if hasattr(system, "frequencies_cm_history"):
        system.frequencies_cm_history = [list(map(float, freq_vector))]


def _prepare_worker_run_sim(
    base_sim,
    *,
    t_coh: float,
    freq_vector: list[float],
    phi1: float,
    phi2: float,
    active_pulses: list[int] | None,
):
    # Keep the original ordering from the pre-cache implementation:
    # update the full simulation first, then carve out reduced pulse subsets.
    base_sim.update_delays(t_coh=t_coh)
    _update_worker_frequencies(base_sim.system, freq_vector)
    base_sim.refresh_cache()
    run_sim = base_sim if active_pulses is None else base_sim.with_pulse_subset(active_pulses)
    _set_phase_subset(run_sim, phi1=phi1, phi2=phi2, active_pulses=active_pulses)
    return run_sim


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
    detection_window: tuple[float, ...],
    active_pulses: list[int] | None = None,
) -> np.ndarray:
    """Worker for one phase-cycling polarisation calculation."""

    run_sim = _prepare_worker_run_sim(
        _cached_worker_run_sim(
            config_source,
            method_override=None,
        ),
        t_coh=t_coh,
        freq_vector=freq_vector,
        phi1=phi1,
        phi2=phi2,
        active_pulses=active_pulses,
    )
    t_det = np.asarray(detection_window, dtype=float)
    _, polarisation = compute_polarisation_over_window(run_sim, t_det)

    # The detection axis is identical for every worker task, so returning only
    # the polarisation avoids repeated IPC copies of the same array.
    return np.asarray(polarisation, dtype=np.complex128)


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

    accumulated = {
        signal_type: np.zeros(component_count, dtype=np.complex128) for signal_type in signal_types
    }
    total_weights_by_signal: dict[str, dict[tuple[float, float], complex]] = {}
    single_1_weights_by_signal: dict[str, dict[float, complex]] = {}
    single_2_weights_by_signal: dict[str, dict[float, complex]] = {}
    single_3_weights_by_signal: dict[str, complex] = {}

    issues: list[str] = []
    seen_total: set[tuple[float, float]] = set()
    seen_single_1: set[float] = set()
    seen_single_2: set[float] = set()
    seen_single_3 = False

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
        polarisation: np.ndarray,
    ) -> np.ndarray:
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
    max_workers = resolve_phase_pool_worker_count(
        configured_workers=int(config.max_workers),
        phase_jobs=total_tasks,
    )
    detection_window_values = tuple(float(value) for value in detection_window_arr)

    for signal_type in signal_types:
        l_index, m_index = COMPONENT_MAP[signal_type] if lm is None else lm
        phi1_factors = {
            phi1: np.exp(-1j * l_index * phi1) for phi1 in phase_values
        }
        phi2_factors = {
            phi2: np.exp(-1j * m_index * phi2) for phi2 in phase_values
        }
        phi1_sum = sum(phi1_factors.values())
        phi2_sum = sum(phi2_factors.values())
        total_weights_by_signal[signal_type] = {
            (phi1, phi2): phi1_factors[phi1] * phi2_factors[phi2]
            for phi1 in phase_values
            for phi2 in phase_values
        }
        single_1_weights_by_signal[signal_type] = {
            phi1: -phi1_factors[phi1] * phi2_sum for phi1 in phase_values
        }
        single_2_weights_by_signal[signal_type] = {
            phi2: -phi1_sum * phi2_factors[phi2] for phi2 in phase_values
        }
        single_3_weights_by_signal[signal_type] = -phi1_sum * phi2_sum

    owns_executor = executor is None
    if owns_executor:
        executor = create_phase_pool_executor(max_workers=max_workers)

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
                detection_window_values,
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
                seen_total.add((phi1_requested, phi2_requested))
                for signal_type in signal_types:
                    accumulated[signal_type] += (
                        total_weights_by_signal[signal_type][(phi1_requested, phi2_requested)]
                        * polarisation
                    )
            elif worker_kind == "single_1":
                seen_single_1.add(phi1_requested)
                for signal_type in signal_types:
                    accumulated[signal_type] += (
                        single_1_weights_by_signal[signal_type][phi1_requested] * polarisation
                    )
            elif worker_kind == "single_2":
                seen_single_2.add(phi2_requested)
                for signal_type in signal_types:
                    accumulated[signal_type] += (
                        single_2_weights_by_signal[signal_type][phi2_requested] * polarisation
                    )
            elif worker_kind == "single_3":
                seen_single_3 = True
                for signal_type in signal_types:
                    accumulated[signal_type] += (
                        single_3_weights_by_signal[signal_type] * polarisation
                    )
            else:
                raise RuntimeError(f"Unexpected worker kind: {worker_kind}")
    finally:
        if owns_executor:
            executor.shutdown(wait=True)

    missing_total = len(phase_values) ** 2 - len(seen_total)
    missing_single_1 = len(phase_values) - len(seen_single_1)
    missing_single_2 = len(phase_values) - len(seen_single_2)
    missing_single_3 = 0 if seen_single_3 else 1

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
