from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from qspectro2d.config.config import resolve_config
from qspectro2d.config.factory import load_simulation
from qspectro2d.core.simulation.time_axes import compute_t_det, compute_times_local
from qspectro2d.spectroscopy.evolution import compute_polarisation_over_window


WORKER_ACTIVE_PULSES: dict[str, list[int] | None] = {
    "total": None,
    "single_1": [0],
    "single_2": [1],
    "single_3": [2],
}


def _preview_complex(values: np.ndarray, *, limit: int = 3) -> str:
    arr = np.asarray(values, dtype=np.complex128)
    if arr.size == 0:
        return "[]"
    return np.array2string(arr[:limit], precision=6, separator=", ")


def _load_frequency_sample(sample_file: Path, sample_index: int) -> np.ndarray:
    samples = np.asarray(np.load(sample_file), dtype=float)

    if samples.ndim == 1:
        if sample_index != 0:
            raise IndexError(
                f"sample_index={sample_index} is invalid for 1D sample file {sample_file}"
            )
        return samples

    if samples.ndim != 2:
        raise ValueError(
            f"Expected sample file with ndim 1 or 2, got shape={samples.shape} from {sample_file}"
        )

    if sample_index < 0 or sample_index >= samples.shape[0]:
        raise IndexError(
            f"sample_index={sample_index} out of range for sample file with "
            f"{samples.shape[0]} rows"
        )

    return samples[sample_index]


def _apply_worker_phases(
    sim_oqs,
    *,
    worker_kind: str,
    phi1: float,
    phi2: float,
) -> list[int] | None:
    active_pulses = WORKER_ACTIVE_PULSES[worker_kind]

    if active_pulses is None:
        sim_oqs.laser.pulse_phases = [phi1, phi2, 0.0]
        return None
    if active_pulses == [0]:
        sim_oqs.laser.pulse_phases = [phi1]
        return active_pulses
    if active_pulses == [1]:
        sim_oqs.laser.pulse_phases = [phi2]
        return active_pulses
    if active_pulses == [2]:
        sim_oqs.laser.pulse_phases = [0.0]
        return active_pulses

    raise ValueError(f"Unsupported worker_kind={worker_kind!r}")


def _build_solver_times(sim_oqs, *, t0_override: float | None) -> tuple[np.ndarray, np.ndarray]:
    auto_times = compute_times_local(sim_oqs.simulation_config)
    if t0_override is None:
        return auto_times, auto_times

    dt = float(sim_oqs.simulation_config.dt)
    t_det = float(sim_oqs.simulation_config.t_det)
    t0 = float(t0_override)
    n_steps = int(np.floor((t_det - t0) / dt)) + 1
    if n_steps < 2:
        raise ValueError(
            f"t0_override={t0} yields fewer than two solver points for t_det={t_det}, dt={dt}"
        )

    solver_times = t0 + dt * np.arange(n_steps, dtype=float)
    return auto_times, solver_times


def _print_case_header(
    *,
    config_path: Path,
    sample_file: Path,
    sample_index: int,
    freq_vector: np.ndarray,
    worker_kind: str,
    active_pulses: list[int] | None,
    phi1: float,
    phi2: float,
    sim_oqs,
    auto_times: np.ndarray,
    solver_times: np.ndarray,
    method_override: str | None,
) -> None:
    print("=" * 80)
    print("LINDBLAD 2D DEBUG HARNESS")
    print("config:", config_path)
    print("sample_file:", sample_file)
    print("sample_index:", sample_index)
    print(
        "freq_vector_cm^-1:",
        np.array2string(np.asarray(freq_vector, dtype=float), precision=12, separator=", "),
    )
    print("worker_kind:", worker_kind)
    print("active_pulses:", "all" if active_pulses is None else active_pulses)
    print("phi1:", float(phi1))
    print("phi2:", float(phi2))
    print("effective pulse phases:", [float(phase) for phase in sim_oqs.laser.pulse_phases])
    print("solver:", sim_oqs.simulation_config.ode_solver)
    print("method:", sim_oqs.simulation_config.solver_options.get("method"))
    print("method_override:", method_override)
    print("t_wait:", float(sim_oqs.simulation_config.t_wait))
    print("t_coh:", float(sim_oqs.simulation_config.t_coh))
    print(
        "t_wait + t_coh:",
        float(sim_oqs.simulation_config.t_wait + sim_oqs.simulation_config.t_coh),
    )
    print("pulse peak times:", [float(pulse.pulse_peak_time) for pulse in sim_oqs.laser.pulses])
    print(
        "pulse active windows:",
        [
            [float(start), float(end)]
            for start, end in (pulse.active_time_range for pulse in sim_oqs.laser.pulses)
        ],
    )
    print("auto solver t0:", float(auto_times[0]))
    print("actual solver t0:", float(solver_times[0]))
    print("solver_times len:", int(solver_times.size))
    print("solver_times first few:", solver_times[:5].tolist())
    print("solver_times last few:", solver_times[-5:].tolist())
    print("=" * 80)


def _run_one_case(
    *,
    resolved_cfg: dict,
    config_path: Path,
    sample_file: Path,
    sample_index: int,
    freq_vector: np.ndarray,
    worker_kind: str,
    phi1: float,
    phi2: float,
    t_coh_value: float,
    t0_override: float | None,
    method_override: str | None,
) -> bool:
    sim_oqs = load_simulation(resolved_cfg, emit_runtime_warnings=False)
    sim_oqs.update_delays(t_coh=t_coh_value)
    sim_oqs.system.update_frequencies_cm(freq_vector.tolist())
    sim_oqs.refresh_cache()

    active_pulses = WORKER_ACTIVE_PULSES[worker_kind]
    run_sim = sim_oqs if active_pulses is None else sim_oqs.with_pulse_subset(active_pulses)
    active_pulses = _apply_worker_phases(
        run_sim,
        worker_kind=worker_kind,
        phi1=phi1,
        phi2=phi2,
    )

    auto_times, solver_times = _build_solver_times(run_sim, t0_override=t0_override)
    detection_window = compute_t_det(run_sim.simulation_config)

    _print_case_header(
        config_path=config_path,
        sample_file=sample_file,
        sample_index=sample_index,
        freq_vector=freq_vector,
        worker_kind=worker_kind,
        active_pulses=active_pulses,
        phi1=phi1,
        phi2=phi2,
        sim_oqs=run_sim,
        auto_times=auto_times,
        solver_times=solver_times,
        method_override=method_override,
    )

    try:
        _, polarisation = compute_polarisation_over_window(
            run_sim,
            window=detection_window,
            solver_times=solver_times,
        )
    except Exception as exc:
        print(f"FAILED for t_coh={t_coh_value:.6g} fs: {exc}")
        traceback.print_exc()
        return False

    polarisation = np.asarray(polarisation, dtype=np.complex128)
    print(f"SUCCESS for t_coh={t_coh_value:.6g} fs")
    print("detection_window len:", int(detection_window.size))
    print("polarisation shape:", polarisation.shape)
    print("polarisation finite:", bool(np.all(np.isfinite(polarisation))))
    print(
        "polarisation max abs:",
        float(np.max(np.abs(polarisation))) if polarisation.size else 0.0,
    )
    print("polarisation preview:", _preview_complex(polarisation))
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one Lindblad 2D worker in-process with detailed solver diagnostics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True, help="Resolved or source config YAML")
    parser.add_argument("--sample-file", type=Path, required=True, help="Path to *.npy sample file")
    parser.add_argument("--sample-index", type=int, default=0, help="Row index inside sample file")
    parser.add_argument(
        "--t-coh",
        dest="t_coh_values",
        type=float,
        nargs="+",
        required=True,
        help="One or more coherence-delay values in fs",
    )
    parser.add_argument(
        "--worker-kind",
        choices=sorted(WORKER_ACTIVE_PULSES),
        required=True,
        help="Phase-cycling worker selection to reproduce",
    )
    parser.add_argument("--phi1", type=float, default=0.0, help="Phase for pulse 1")
    parser.add_argument("--phi2", type=float, default=0.0, help="Phase for pulse 2")
    parser.add_argument(
        "--method-override",
        type=str,
        default=None,
        help="Override config.solver_options.method for this debug run",
    )
    parser.add_argument(
        "--t0-override",
        type=float,
        default=None,
        help="Force the solver-local start time and build an explicit solver grid",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Override config.config.max_workers before loading the simulation",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.max_workers <= 0:
        raise ValueError("--max-workers must be >= 1")

    config_path = args.config.resolve()
    sample_file = args.sample_file.resolve()

    resolved_cfg = resolve_config(config_path, emit_runtime_warnings=False)
    if str(resolved_cfg["config"]["solver"]) != "lindblad":
        raise ValueError(
            f"This debug harness is for lindblad configs only; got {resolved_cfg['config']['solver']!r}"
        )

    resolved_cfg["config"]["max_workers"] = int(args.max_workers)
    if args.method_override is not None:
        resolved_cfg["config"].setdefault("solver_options", {})["method"] = str(args.method_override)

    freq_vector = _load_frequency_sample(sample_file, args.sample_index)

    any_failed = False
    for t_coh_value in args.t_coh_values:
        ok = _run_one_case(
            resolved_cfg=resolved_cfg,
            config_path=config_path,
            sample_file=sample_file,
            sample_index=int(args.sample_index),
            freq_vector=freq_vector,
            worker_kind=str(args.worker_kind),
            phi1=float(args.phi1),
            phi2=float(args.phi2),
            t_coh_value=float(t_coh_value),
            t0_override=args.t0_override,
            method_override=args.method_override,
        )
        any_failed = any_failed or (not ok)

    raise SystemExit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
