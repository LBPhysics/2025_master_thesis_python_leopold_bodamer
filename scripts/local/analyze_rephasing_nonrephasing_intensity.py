from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common.plot_settings import APODIZATION_WINDOW, PAD_FACTOR
from qspectro2d.config.config import resolve_config
from qspectro2d.spectroscopy.emitted_field import compute_emitted_field_components
from qspectro2d.spectroscopy.post_processing import compute_spectra


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "scripts" / "simulation_configs"
OUTPUT_DIR = PROJECT_ROOT / "notebooks" / "test_Spectroscopy" / "rephasing_nonrephasing_intensity"


def _intensity_metrics(axis: np.ndarray, values: np.ndarray) -> dict[str, float]:
    axis = np.asarray(axis, dtype=float)
    magnitude = np.abs(np.asarray(values, dtype=complex))
    return {
        "peak_abs": float(np.max(magnitude)),
        "l1_abs": float(np.trapezoid(magnitude, axis)),
        "l2_energy": float(np.trapezoid(magnitude**2, axis)),
    }


def _ratio_metrics(reference: dict[str, float], target: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, ref_value in reference.items():
        out[key] = float(target[key] / ref_value) if ref_value != 0 else float("nan")
    return out


def _plot_ratio_summary(summary: dict[str, dict]) -> Path:
    metrics = [
        ("peak_abs", "Peak"),
        ("l1_abs", "L1"),
        ("l2_energy", "L2 energy"),
    ]
    case_order = [
        ("monomer", "Monomer", "#1f77b4"),
        ("dimer", "Dimer", "#ff7f0e"),
    ]
    domains = [
        ("time_domain", "Time domain"),
        ("frequency_domain", "Frequency domain"),
    ]

    x = np.arange(len(metrics), dtype=float)
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    ymax = 0.0

    for ax, (domain_key, domain_title) in zip(axes, domains):
        for idx, (case_key, case_label, color) in enumerate(case_order):
            ratios = summary[case_key][domain_key]["nonrephasing_over_rephasing"]
            values = [float(ratios[key]) for key, _ in metrics]
            offset = (idx - 0.5) * width
            bars = ax.bar(x + offset, values, width=width, label=case_label, color=color, alpha=0.9)
            ymax = max(ymax, max(values, default=0.0))

            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.04,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.2)
        ax.set_xticks(x, [label for _, label in metrics])
        ax.set_title(domain_title)
        ax.set_xlabel("Intensity measure")
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("NR / R ratio")
    axes[0].legend(frameon=False)

    if ymax > 0.0:
        axes[0].set_ylim(0.0, ymax * 1.18 + 0.05)

    fig.suptitle("Rephasing vs nonrephasing intensity ratios\npaper_eqs, rwa_sl=True")
    fig.tight_layout()

    output_path = OUTPUT_DIR / "paper_eqs_rwa_rephasing_nonrephasing_intensity_ratios.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _base_case_config(case_name: str) -> dict:
    config_name = "monomer.yaml" if case_name == "monomer" else "dimer.yaml"
    cfg = resolve_config(CONFIG_DIR / config_name, emit_runtime_warnings=False)

    cfg["config"]["solver"] = "paper_eqs"
    cfg["laser"]["rwa_sl"] = True
    cfg["config"]["max_workers"] = 1
    cfg["config"]["n_phases"] = 4

    if case_name == "monomer":
        cfg["config"]["t_coh"] = 12.0
        cfg["config"]["t_wait"] = 6.0
        cfg["config"]["t_det"] = 60.0
        cfg["config"]["dt"] = 0.25
        return resolve_config(cfg, emit_runtime_warnings=False)

    cfg["config"]["t_coh"] = 10.0
    cfg["config"]["t_wait"] = 5.0
    cfg["config"]["t_det"] = 60.0
    cfg["config"]["dt"] = 0.25
    cfg["laser"]["pulse_amplitudes"] = [0.001, 0.0015, 0.002]
    return resolve_config(cfg, emit_runtime_warnings=False)


def _analyze_case(case_name: str) -> dict:
    cfg = _base_case_config(case_name)

    t_det = np.arange(
        0.0,
        float(cfg["config"]["t_det"]) + 0.5 * float(cfg["config"]["dt"]),
        float(cfg["config"]["dt"]),
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        emitted_fields, run_status, status_message = compute_emitted_field_components(
            cfg,
            float(cfg["config"]["t_coh"]),
            list(cfg["atomic"]["frequencies_cm"]),
            detection_window=t_det,
            executor=executor,
        )

    _, nu_det, datas_nu, out_types = compute_spectra(
        emitted_fields,
        list(cfg["config"]["signal_types"]),
        t_det,
        None,
        pad=PAD_FACTOR,
        apodization=APODIZATION_WINDOW,
    )
    nu_det_cm = (np.asarray(nu_det, dtype=float) + float(cfg["laser"]["carrier_freq_cm"]) * 1e-4) * 1e4

    time_signals = {
        name: np.asarray(field, dtype=np.complex128)
        for name, field in zip(cfg["config"]["signal_types"], emitted_fields)
    }
    freq_signals = {
        name: np.asarray(data, dtype=np.complex128)
        for name, data in zip(out_types, datas_nu)
        if name in cfg["config"]["signal_types"]
    }

    time_metrics = {name: _intensity_metrics(t_det, values) for name, values in time_signals.items()}
    freq_metrics = {name: _intensity_metrics(nu_det_cm, values) for name, values in freq_signals.items()}

    return {
        "status": str(run_status),
        "message": status_message,
        "settings": {
            "solver": str(cfg["config"]["solver"]),
            "rwa_sl": bool(cfg["laser"]["rwa_sl"]),
            "pad_factor": float(PAD_FACTOR),
            "apodization_window": APODIZATION_WINDOW,
            "t_coh_fs": float(cfg["config"]["t_coh"]),
            "t_wait_fs": float(cfg["config"]["t_wait"]),
            "t_det_fs": float(cfg["config"]["t_det"]),
            "dt_fs": float(cfg["config"]["dt"]),
        },
        "time_domain": {
            "rephasing": time_metrics["rephasing"],
            "nonrephasing": time_metrics["nonrephasing"],
            "nonrephasing_over_rephasing": _ratio_metrics(
                time_metrics["rephasing"],
                time_metrics["nonrephasing"],
            ),
        },
        "frequency_domain": {
            "rephasing": freq_metrics["rephasing"],
            "nonrephasing": freq_metrics["nonrephasing"],
            "nonrephasing_over_rephasing": _ratio_metrics(
                freq_metrics["rephasing"],
                freq_metrics["nonrephasing"],
            ),
        },
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "monomer": _analyze_case("monomer"),
        "dimer": _analyze_case("dimer"),
    }

    output_path = OUTPUT_DIR / "paper_eqs_rwa_rephasing_nonrephasing_intensity.json"
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved intensity summary to: {output_path}")

    figure_path = _plot_ratio_summary(summary)
    print(f"Saved intensity ratio figure to: {figure_path}")


if __name__ == "__main__":
    main()
