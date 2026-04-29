from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from qutip import ket2dm
from scipy.optimize import minimize
import yaml

from qspectro2d.config.config import resolve_config
from qspectro2d.config.factory import load_simulation
from qspectro2d.spectroscopy.evolution import compute_evolution


def _fit_exp_decay_rate(times: np.ndarray, values: np.ndarray, *, floor: float = 1e-12) -> float:
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(times) & np.isfinite(values)
    times = times[valid]
    values = np.clip(values[valid], floor, None)

    if times.size < 5:
        raise ValueError("Not enough points to fit exponential rate")

    start = max(1, int(0.05 * times.size))
    stop = max(start + 5, int(0.85 * times.size))
    slope, _ = np.polyfit(times[start:stop], np.log(values[start:stop]), 1)
    return float(-slope)


def _estimate_effective_rates(
    cfg: dict,
    *,
    pop_level: int,
    coh_levels: tuple[int, int],
    t_stop_fs: float,
    dt_fs: float,
) -> dict[str, float]:
    sim = load_simulation(cfg, emit_runtime_warnings=False)
    basis = sim.system.basis
    a, b = coh_levels

    t_grid = np.arange(0.0, float(t_stop_fs) + 0.5 * float(dt_fs), float(dt_fs))

    rho_pop = ket2dm(basis[pop_level])
    pop_op = ket2dm(basis[pop_level])
    t_pop, pop = compute_evolution(
        sim,
        e_ops=[pop_op],
        initial_state=rho_pop,
        solver_times=t_grid,
        field_free=True,
    )
    pop = np.real(np.asarray(pop, dtype=complex))
    tail_n = max(6, int(0.15 * pop.size))
    p_inf = float(np.mean(pop[-tail_n:]))
    centered = np.clip(pop - p_inf, 1e-14, None)
    k_pop = _fit_exp_decay_rate(t_pop, centered)

    psi = (basis[a] + basis[b]).unit()
    rho_coh = ket2dm(psi)
    coh_op = basis[a] * basis[b].dag()
    t_coh, coh = compute_evolution(
        sim,
        e_ops=[coh_op],
        initial_state=rho_coh,
        solver_times=t_grid,
        field_free=True,
    )
    coh_mag = np.abs(np.asarray(coh, dtype=complex))
    gamma_coh = _fit_exp_decay_rate(t_coh, coh_mag)

    return {
        "k_pop_fs_inv": float(k_pop),
        "gamma_coh_fs_inv": float(gamma_coh),
        "T1_fs": float(np.inf if k_pop <= 0 else 1.0 / k_pop),
        "T2_fs": float(np.inf if gamma_coh <= 0 else 1.0 / gamma_coh),
        "p_inf": p_inf,
    }


def _objective_from_rates(
    rates: dict[str, float],
    *,
    target_k_pop: float,
    target_gamma_coh: float,
) -> float:
    e_pop = rates["k_pop_fs_inv"] / target_k_pop - 1.0
    e_coh = rates["gamma_coh_fs_inv"] / target_gamma_coh - 1.0
    return float(e_pop * e_pop + e_coh * e_coh)


def _write_fitted_config(base_config_path: Path, fitted_cfg: dict, output_path: Path) -> None:
    raw_cfg = yaml.safe_load(base_config_path.read_text(encoding="utf-8")) or {}
    raw_cfg.setdefault("bath", {})
    raw_cfg["bath"]["bath_temperature"] = float(fitted_cfg["bath"]["bath_temperature"])
    raw_cfg["bath"]["bath_cutoff"] = float(fitted_cfg["bath"]["bath_cutoff"])
    raw_cfg["bath"]["sb_coupling"] = float(fitted_cfg["bath"]["sb_coupling"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(raw_cfg, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit trimer bath_cutoff and sb_coupling so the trimer reproduces dimer-like "
            "field-free population-decay and coherence-decay rates."
        )
    )
    parser.add_argument(
        "--reference-config",
        type=Path,
        default=Path("scripts/simulation_configs/dimer.yaml"),
    )
    parser.add_argument(
        "--target-config",
        type=Path,
        default=Path("scripts/simulation_configs/trimer_ring_single.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp/trimer_bath_fit_like_dimer"),
    )
    parser.add_argument(
        "--fitted-config",
        type=Path,
        default=Path("tmp/trimer_bath_fit_like_dimer/trimer_ring_single_fitted.yaml"),
    )
    parser.add_argument("--reference-pop-level", type=int, default=2)
    parser.add_argument("--reference-coh-levels", type=int, nargs=2, default=(1, 2))
    parser.add_argument("--target-pop-level", type=int, default=3)
    parser.add_argument("--target-coh-levels", type=int, nargs=2, default=(2, 3))
    parser.add_argument(
        "--bath-temperature",
        type=float,
        default=None,
        help="Fixed normalized bath temperature during the fit. Defaults to the reference-config value.",
    )
    parser.add_argument("--cutoff-range", type=float, nargs=2, default=(1e-3, 2.0))
    parser.add_argument("--sb-range", type=float, nargs=2, default=(1e-6, 1.0))
    parser.add_argument("--t-stop-fs", type=float, default=2500.0)
    parser.add_argument("--dt-fs", type=float, default=2.0)
    args = parser.parse_args()

    ref_cfg = resolve_config(args.reference_config, emit_runtime_warnings=False)
    target_cfg_base = resolve_config(args.target_config, emit_runtime_warnings=False)

    bath_temperature = (
        float(ref_cfg["bath"]["bath_temperature"])
        if args.bath_temperature is None
        else float(args.bath_temperature)
    )

    ref_rates = _estimate_effective_rates(
        ref_cfg,
        pop_level=int(args.reference_pop_level),
        coh_levels=(int(args.reference_coh_levels[0]), int(args.reference_coh_levels[1])),
        t_stop_fs=float(args.t_stop_fs),
        dt_fs=float(args.dt_fs),
    )

    cut_lo, cut_hi = map(float, args.cutoff_range)
    sb_lo, sb_hi = map(float, args.sb_range)
    if not (0.0 < cut_lo < cut_hi):
        raise ValueError("cutoff-range must satisfy 0 < lo < hi")
    if not (0.0 < sb_lo < sb_hi):
        raise ValueError("sb-range must satisfy 0 < lo < hi")

    x0 = np.array(
        [
            np.log10(float(target_cfg_base["bath"]["bath_cutoff"])),
            np.log10(float(target_cfg_base["bath"]["sb_coupling"])),
        ],
        dtype=float,
    )
    bounds = [
        (np.log10(cut_lo), np.log10(cut_hi)),
        (np.log10(sb_lo), np.log10(sb_hi)),
    ]

    eval_cache: dict[tuple[float, float], tuple[float, dict, dict[str, float]]] = {}

    def evaluate(log_params: np.ndarray) -> tuple[float, dict, dict[str, float]]:
        key = tuple(np.round(np.asarray(log_params, dtype=float), 10))
        if key in eval_cache:
            return eval_cache[key]

        bath_cutoff, sb_coupling = [float(10.0**value) for value in log_params]
        cfg = resolve_config(target_cfg_base, emit_runtime_warnings=False)
        cfg["bath"]["bath_temperature"] = bath_temperature
        cfg["bath"]["bath_cutoff"] = bath_cutoff
        cfg["bath"]["sb_coupling"] = sb_coupling
        cfg = resolve_config(cfg, emit_runtime_warnings=False)

        rates = _estimate_effective_rates(
            cfg,
            pop_level=int(args.target_pop_level),
            coh_levels=(int(args.target_coh_levels[0]), int(args.target_coh_levels[1])),
            t_stop_fs=float(args.t_stop_fs),
            dt_fs=float(args.dt_fs),
        )
        obj = _objective_from_rates(
            rates,
            target_k_pop=float(ref_rates["k_pop_fs_inv"]),
            target_gamma_coh=float(ref_rates["gamma_coh_fs_inv"]),
        )
        eval_cache[key] = (obj, cfg, rates)
        return eval_cache[key]

    def objective(log_params: np.ndarray) -> float:
        return evaluate(log_params)[0]

    result = minimize(
        objective,
        x0=x0,
        method="Powell",
        bounds=bounds,
        options={"maxiter": 40, "xtol": 1e-3, "ftol": 1e-6},
    )

    obj_best, fitted_cfg, fitted_rates = evaluate(result.x)
    _write_fitted_config(args.target_config.resolve(), fitted_cfg, args.fitted_config.resolve())

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "reference_config": str(args.reference_config.resolve()),
        "target_config": str(args.target_config.resolve()),
        "fitted_config": str(args.fitted_config.resolve()),
        "rate_definition": {
            "reference_pop_level": int(args.reference_pop_level),
            "reference_coh_levels": [int(args.reference_coh_levels[0]), int(args.reference_coh_levels[1])],
            "target_pop_level": int(args.target_pop_level),
            "target_coh_levels": [int(args.target_coh_levels[0]), int(args.target_coh_levels[1])],
            "t_stop_fs": float(args.t_stop_fs),
            "dt_fs": float(args.dt_fs),
        },
        "search_ranges": {
            "bath_temperature": float(bath_temperature),
            "bath_cutoff": [cut_lo, cut_hi],
            "sb_coupling": [sb_lo, sb_hi],
        },
        "optimizer": {
            "success": bool(result.success),
            "message": str(result.message),
            "nfev": int(getattr(result, "nfev", -1)),
            "objective": float(obj_best),
        },
        "reference_effective_rates": ref_rates,
        "fitted_bath": {
            "bath_temperature": float(fitted_cfg["bath"]["bath_temperature"]),
            "bath_cutoff": float(fitted_cfg["bath"]["bath_cutoff"]),
            "sb_coupling": float(fitted_cfg["bath"]["sb_coupling"]),
            "effective_cutoff_cm": float(
                float(fitted_cfg["bath"]["bath_cutoff"])
                * np.mean(np.asarray(fitted_cfg["atomic"]["frequencies_cm"], dtype=float))
            ),
        },
        "target_effective_rates": fitted_rates,
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
