"""Extract four local real-part peak values from dimer 2D spectra.

This is aimed at waiting-time analysis of the rephasing dimer spectra.
For each job it finds the local extrema with largest |Re \tilde{E}| near the
expected peak positions 11, 12, 21, and 22.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_SCRIPTS = PROJECT_ROOT / "scripts" / "local"
QSPECTRO_SRC = PROJECT_ROOT / "packages" / "qspectro2d" / "src"
if str(LOCAL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(LOCAL_SCRIPTS))
if str(QSPECTRO_SRC) not in sys.path:
    sys.path.insert(0, str(QSPECTRO_SRC))

from extract_dimer_linewidths import (
    APODIZATION,
    PAD_FACTOR,
    _metadata_config,
    _peak_specs,
    _stored_section,
)
from qspectro2d.spectroscopy.post_processing import compute_spectra
from qspectro2d.visualization.plotting import convert_plot_axes


DEFAULT_FIG4_JOBS = sorted(
    (
        PROJECT_ROOT
        / "jobs"
        / "dimer"
        / "150426_use_those_in_thesis"
        / "n_inh_1000"
    ).glob("*_dimer_fig4_T*_paper_eqs")
)


def _case_label(config: dict[str, Any]) -> str:
    delta = float(config["atomic"]["delta_inhomogen_cm"])
    return "homogeneous" if abs(delta) < 1e-12 else "inhomogeneous"


def _load_metadata_with_fallback(
    job_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    try:
        metadata, config = _metadata_config(job_dir)
        return metadata, config, "job_metadata.json"
    except Exception:
        yaml_candidates = sorted(job_dir.glob("*.yaml"))
        if not yaml_candidates:
            raise
        with yaml_candidates[0].open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        with np.load(job_dir / "data" / "2d_inhom_averaged.npz", allow_pickle=False) as bundle:
            rephasing = np.asarray(bundle["signal::rephasing"])
        if rephasing.ndim != 2:
            raise RuntimeError("Expected a 2D rephasing signal array")
        n_coh, n_det = rephasing.shape
        dt = float(config["config"]["dt"])
        metadata = {
            "t_coh": (np.arange(1, n_coh + 1, dtype=float) * dt).tolist(),
            "t_det": (np.arange(n_det, dtype=float) * dt).tolist(),
        }
        return metadata, config, f"{yaml_candidates[0].name} + inferred axes"


def _rephasing_real_spectrum(job_dir: Path) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    metadata, config, _metadata_source = _load_metadata_with_fallback(job_dir)
    with np.load(job_dir / "data" / "2d_inhom_averaged.npz", allow_pickle=False) as bundle:
        signal_types = ["rephasing", "nonrephasing"]
        signals = [np.asarray(bundle[f"signal::{name}"]) for name in signal_types]

    t_coh = np.asarray(metadata["t_coh"], dtype=float)
    t_det = np.asarray(metadata["t_det"], dtype=float)
    nu_coh, nu_det, spectra, out_types = compute_spectra(
        signals,
        signal_types,
        t_det=t_det,
        t_coh=t_coh,
        pad=PAD_FACTOR,
        section=_stored_section(config),
        apodization=APODIZATION,
    )
    if nu_coh is None:
        raise RuntimeError("Expected 2D coherence-frequency axis")

    carrier = float(config["laser"]["carrier_freq_cm"])
    nu_coh_plot, nu_det_plot = convert_plot_axes(nu_coh, nu_det, carrier_freq_cm=carrier)
    axis_coh_cm = np.asarray(nu_coh_plot, dtype=float) * 1e4
    axis_det_cm = np.asarray(nu_det_plot, dtype=float) * 1e4
    coh_mask = (axis_coh_cm >= 1.5e4) & (axis_coh_cm <= 1.7e4)
    det_mask = (axis_det_cm >= 1.5e4) & (axis_det_cm <= 1.7e4)
    coh_idx = np.flatnonzero(coh_mask)
    det_idx = np.flatnonzero(det_mask)

    rephasing = spectra[out_types.index("rephasing")]
    if sp.issparse(rephasing):
        block = rephasing.tocsr()[np.ix_(coh_idx, det_idx)].toarray()
    else:
        block = np.asarray(rephasing)[np.ix_(coh_idx, det_idx)]
    return metadata, config, axis_coh_cm[coh_idx], axis_det_cm[det_idx], np.asarray(np.real(block), dtype=float)


def _find_peak(
    real_data: np.ndarray,
    axis_coh_cm: np.ndarray,
    axis_det_cm: np.ndarray,
    expected_coh_cm: float,
    expected_det_cm: float,
    *,
    radius_cm: float,
) -> tuple[float, float, float]:
    coh_grid, det_grid = np.meshgrid(axis_coh_cm, axis_det_cm, indexing="ij")
    mask = np.hypot(coh_grid - expected_coh_cm, det_grid - expected_det_cm) <= radius_cm
    if not np.any(mask):
        raise RuntimeError("Empty search mask for expected peak")
    search = np.where(mask, np.abs(real_data), -np.inf)
    row, col = np.unravel_index(int(np.nanargmax(search)), real_data.shape)
    return float(real_data[row, col]), float(axis_coh_cm[row]), float(axis_det_cm[col])


def _rows_for_job(job_dir: Path) -> list[dict[str, Any]]:
    metadata, config, metadata_source = _load_metadata_with_fallback(job_dir)
    _, _, axis_coh_cm, axis_det_cm, real_data = _rephasing_real_spectrum(job_dir)
    peaks = _peak_specs(config)
    low = float(peaks[0].expected_cm)
    high = float(peaks[1].expected_cm)
    radius = min(float(peaks[0].search_radius_cm), float(peaks[1].search_radius_cm))

    found = {
        "11": _find_peak(real_data, axis_coh_cm, axis_det_cm, low, low, radius_cm=radius),
        "12": _find_peak(real_data, axis_coh_cm, axis_det_cm, low, high, radius_cm=radius),
        "21": _find_peak(real_data, axis_coh_cm, axis_det_cm, high, low, radius_cm=radius),
        "22": _find_peak(real_data, axis_coh_cm, axis_det_cm, high, high, radius_cm=radius),
    }
    abs_found = {key: abs(value[0]) for key, value in found.items()}
    return [
        {
            "job": job_dir.name,
            "job_dir": str(job_dir),
            "case": _case_label(config),
            "t_wait_fs": float(config["config"]["t_wait"]),
            "coupling_cm": float(config["atomic"].get("coupling_cm", 0.0)),
            "delta_inhomogen_cm": float(config["atomic"]["delta_inhomogen_cm"]),
            "peak": key,
            "real_value": found[key][0],
            "abs_real_value": abs_found[key],
            "coh_cm": found[key][1],
            "det_cm": found[key][2],
            "metadata_source": metadata_source,
        }
        for key in ("11", "12", "21", "22")
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "job",
        "case",
        "t_wait_fs",
        "coupling_cm",
        "delta_inhomogen_cm",
        "peak",
        "real_value",
        "abs_real_value",
        "coh_cm",
        "det_cm",
        "metadata_source",
        "job_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_dirs", nargs="*", type=Path)
    parser.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_ROOT / "jobs" / "dimer" / "150426_use_those_in_thesis" / "n_inh_1000" / "fig4_fourpeak_rephasing_realpart.csv",
    )
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for job_dir in (args.job_dirs or DEFAULT_FIG4_JOBS):
        job_dir = Path(job_dir).resolve()
        if "fig4" not in job_dir.name:
            continue
        print(f"Analyzing {job_dir}")
        rows.extend(_rows_for_job(job_dir))
    csv_path = Path(args.csv).resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(csv_path, rows)
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
