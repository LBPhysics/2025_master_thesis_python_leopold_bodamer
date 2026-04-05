from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common.plot_settings import APODIZATION_WINDOW
from qspectro2d.config.config import resolve_config
from qspectro2d.spectroscopy.emitted_field import compute_emitted_field_components
from qspectro2d.spectroscopy.post_processing import compute_spectra


plt.rcParams["text.usetex"] = False


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "scripts" / "simulation_configs"
OUTPUT_DIR = PROJECT_ROOT / "notebooks" / "test_Spectroscopy" / "solver_alignment_1d"

SIGNAL_ORDER = ("rephasing", "nonrephasing", "absorptive")
PAD_FACTOR = 16.0

RUN_STYLES = {
    "lindblad_rwa": {
        "label": "Lindblad, RWA",
        "solver": "lindblad",
        "rwa_sl": True,
        "color": "#1f77b4",
        "linestyle": "-",
    },
    "paper_eqs_rwa": {
        "label": "paper_eqs, RWA",
        "solver": "paper_eqs",
        "rwa_sl": True,
        "color": "#111111",
        "linestyle": "--",
    },
    "lindblad_no_rwa": {
        "label": "Lindblad, no RWA",
        "solver": "lindblad",
        "rwa_sl": False,
        "color": "#d62728",
        "linestyle": "-.",
    },
}


def _build_case_config(case_name: str) -> dict:
    config_name = "monomer.yaml" if case_name == "monomer" else "dimer.yaml"
    cfg = resolve_config(CONFIG_DIR / config_name, emit_runtime_warnings=False)

    cfg["config"]["max_workers"] = 1
    cfg["config"]["n_phases"] = 4
    cfg["config"]["dt"] = 0.25
    cfg["config"]["t_det"] = 60.0

    if case_name == "monomer":
        cfg["config"]["t_coh"] = 12.0
        cfg["config"]["t_wait"] = 6.0
        return cfg

    cfg["config"]["t_coh"] = 10.0
    cfg["config"]["t_wait"] = 5.0
    cfg["laser"]["pulse_amplitudes"] = [0.001, 0.0015, 0.002]
    cfg["atomic"]["deph_rate_fs"] = 0.0
    cfg["atomic"]["down_rate_fs"] = 0.0
    cfg["atomic"]["up_rate_fs"] = 0.0
    # Keep config validation happy while isolating the Hamiltonian-driven response.
    cfg["bath"]["sb_coupling"] = 1e-15
    return cfg


def _convert_plot_axis(nu_det: np.ndarray, *, carrier_freq_cm: float) -> np.ndarray:
    return np.asarray(nu_det, dtype=float) + float(carrier_freq_cm) * 1e-4


def _run_1d_spectrum(cfg: dict) -> dict:
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

    if bool(cfg["laser"]["rwa_sl"]):
        nu_det_plot = _convert_plot_axis(
            nu_det,
            carrier_freq_cm=float(cfg["laser"]["carrier_freq_cm"]),
        )
    else:
        nu_det_plot = nu_det

    spectra = {name: np.asarray(data) for name, data in zip(out_types, datas_nu)}
    return {
        "axis_cm": np.asarray(nu_det_plot, dtype=float) * 1e4,
        "spectra": spectra,
        "status": str(run_status),
        "message": status_message,
        "resolved_dt_fs": float(cfg["config"]["dt"]),
        "t_det_fs": t_det.tolist(),
    }


def _normalized_abs(values: np.ndarray) -> np.ndarray:
    values = np.abs(np.asarray(values, dtype=complex))
    vmax = float(np.max(values))
    if vmax <= 0.0:
        return np.zeros_like(values, dtype=float)
    return values / vmax


def _interp_to_reference(ref_x: np.ndarray, x: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.interp(ref_x, x, _normalized_abs(values))


def _summarize_case(case_name: str, runs: dict[str, dict]) -> dict:
    ref_x = np.asarray(runs["lindblad_rwa"]["result"]["axis_cm"], dtype=float)
    summary: dict[str, dict] = {
        "case": {
            "name": case_name,
            "t_coh_fs": float(runs["lindblad_rwa"]["config"]["config"]["t_coh"]),
            "t_wait_fs": float(runs["lindblad_rwa"]["config"]["config"]["t_wait"]),
            "t_det_fs": float(runs["lindblad_rwa"]["config"]["config"]["t_det"]),
        },
        "runs": {},
        "signals": {},
    }

    for run_name, payload in runs.items():
        summary["runs"][run_name] = {
            "label": RUN_STYLES[run_name]["label"],
            "solver": payload["config"]["config"]["solver"],
            "rwa_sl": bool(payload["config"]["laser"]["rwa_sl"]),
            "status": payload["result"]["status"],
            "message": payload["result"]["message"],
            "resolved_dt_fs": float(payload["result"]["resolved_dt_fs"]),
        }

    all_peaks: list[float] = []
    for signal_name in SIGNAL_ORDER:
        signal_summary: dict[str, dict | float] = {
            "peaks_cm^-1": {},
            "normalized_max_diff_vs_lindblad_rwa": {},
        }

        ref_spectrum = runs["lindblad_rwa"]["result"]["spectra"][signal_name]
        ref_peak = float(
            ref_x[int(np.argmax(np.abs(ref_spectrum)))]
        )
        all_peaks.append(ref_peak)

        ref_interp = _interp_to_reference(
            ref_x,
            runs["lindblad_rwa"]["result"]["axis_cm"],
            ref_spectrum,
        )

        for run_name, payload in runs.items():
            x = np.asarray(payload["result"]["axis_cm"], dtype=float)
            spectrum = payload["result"]["spectra"][signal_name]
            peak = float(x[int(np.argmax(np.abs(spectrum)))])
            all_peaks.append(peak)
            signal_summary["peaks_cm^-1"][run_name] = peak

            interp = _interp_to_reference(ref_x, x, spectrum)
            signal_summary["normalized_max_diff_vs_lindblad_rwa"][run_name] = float(
                np.max(np.abs(ref_interp - interp))
            )

        summary["signals"][signal_name] = signal_summary

    summary["plot_window_cm^-1"] = _plot_window(all_peaks)
    return summary


def _plot_window(peak_positions_cm: list[float]) -> list[float]:
    left = float(min(peak_positions_cm))
    right = float(max(peak_positions_cm))
    margin = max(250.0, 0.25 * (right - left))
    return [left - margin, right + margin]


def _plot_case(case_name: str, runs: dict[str, dict], summary: dict) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    x_min, x_max = summary["plot_window_cm^-1"]
    fig, axes = plt.subplots(
        len(SIGNAL_ORDER),
        1,
        figsize=(10, 9),
        sharex=True,
        constrained_layout=True,
    )

    for axis, signal_name in zip(axes, SIGNAL_ORDER):
        for run_name, payload in runs.items():
            style = RUN_STYLES[run_name]
            x = np.asarray(payload["result"]["axis_cm"], dtype=float)
            y = _normalized_abs(payload["result"]["spectra"][signal_name])
            axis.plot(
                x,
                y,
                label=style["label"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.0,
            )

        axis.set_ylabel(f"{signal_name}\nnormalized |S|")
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(-0.02, 1.05)
        axis.grid(True, alpha=0.25)

    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel(r"Detection Frequency $\nu$ (cm$^{-1}$)")

    case_info = summary["case"]
    fig.suptitle(
        (
            f"1D Solver Alignment: {case_name.capitalize()} "
            f"(t_coh={case_info['t_coh_fs']:.1f} fs, "
            f"t_wait={case_info['t_wait_fs']:.1f} fs, "
            f"t_det={case_info['t_det_fs']:.1f} fs)"
        ),
        fontsize=13,
    )

    output_path = OUTPUT_DIR / f"{case_name}_1d_solver_alignment.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    all_summary: dict[str, dict] = {}

    for case_name in ("monomer", "dimer"):
        base_cfg = _build_case_config(case_name)
        runs: dict[str, dict] = {}

        for run_name, style in RUN_STYLES.items():
            run_cfg = deepcopy(base_cfg)
            run_cfg["config"]["solver"] = style["solver"]
            run_cfg["laser"]["rwa_sl"] = bool(style["rwa_sl"])
            resolved = resolve_config(run_cfg, emit_runtime_warnings=False)

            runs[run_name] = {
                "config": resolved,
                "result": _run_1d_spectrum(resolved),
            }

        summary = _summarize_case(case_name, runs)
        plot_path = _plot_case(case_name, runs, summary)
        summary["plot_path"] = str(plot_path)
        all_summary[case_name] = summary

    summary_path = OUTPUT_DIR / "solver_alignment_1d_summary.json"
    summary_path.write_text(json.dumps(all_summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Saved plots and summary to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
