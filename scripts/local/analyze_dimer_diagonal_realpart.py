"""Compare the two diagonal dimer peaks in the real-part spectra.

For each selected dimer job and branch, this script:
- reproduces the frequency-domain spectra with the same FFT settings as plotting
- identifies the lower (11) and upper (22) diagonal features near the expected
  single-exciton energies
- finds the local extremum with the largest |Re S| near each feature
- reports the signed real-part value, its magnitude, and the lower/upper ratio

Results are written to per-job summaries and to one aggregate summary.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from extract_dimer_linewidths import (
    APODIZATION,
    PAD_FACTOR,
    PROJECT_ROOT,
    _metadata_config,
    _peak_specs,
    _stored_section,
)

QSPECTRO_SRC = PROJECT_ROOT / "packages" / "qspectro2d" / "src"
if str(QSPECTRO_SRC) not in sys.path:
    sys.path.insert(0, str(QSPECTRO_SRC))

from qspectro2d.spectroscopy.post_processing import compute_spectra
from qspectro2d.visualization.plotting import convert_plot_axes


DEFAULT_JOB_DIRS = [
    PROJECT_ROOT
    / "jobs"
    / "dimer"
    / "150426_use_those_in_thesis"
    / "n_inh_1000"
    / "10000"
    / "19_222916_dimer_fig3a_uncoupled_paper_eqs",
    PROJECT_ROOT
    / "jobs"
    / "dimer"
    / "150426_use_those_in_thesis"
    / "n_inh_1000"
    / "10000"
    / "17_121018_dimer_fig3b_coupled_paper_eqs",
    PROJECT_ROOT
    / "jobs"
    / "dimer"
    / "150426_use_those_in_thesis"
    / "n_inh_1000"
    / "15_152741_dimer_fig4_T046_paper_eqs",
    PROJECT_ROOT
    / "jobs"
    / "dimer"
    / "150426_use_those_in_thesis"
    / "n_inh_1000"
    / "15_152817_dimer_fig4_T062_paper_eqs",
    PROJECT_ROOT
    / "jobs"
    / "dimer"
    / "150426_use_those_in_thesis"
    / "n_inh_1000"
    / "15_162341_dimer_fig6_T046_paper_eqs",
    PROJECT_ROOT
    / "jobs"
    / "dimer"
    / "150426_use_those_in_thesis"
    / "n_inh_1000"
    / "15_162408_dimer_fig6_T062_paper_eqs",
]


def _complex_spectra_for_job(
    job_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    metadata, config = _metadata_config(job_dir)
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
        raise RuntimeError("Expected a 2D spectrum with a coherence-frequency axis")

    carrier = float(config["laser"]["carrier_freq_cm"])
    nu_coh_plot, nu_det_plot = convert_plot_axes(nu_coh, nu_det, carrier_freq_cm=carrier)
    axis_coh_cm = np.asarray(nu_coh_plot, dtype=float) * 1e4
    axis_det_cm = np.asarray(nu_det_plot, dtype=float) * 1e4
    coh_mask = (axis_coh_cm >= 1.5e4) & (axis_coh_cm <= 1.7e4)
    det_mask = (axis_det_cm >= 1.5e4) & (axis_det_cm <= 1.7e4)
    coh_idx = np.flatnonzero(coh_mask)
    det_idx = np.flatnonzero(det_mask)

    cropped: dict[str, np.ndarray] = {}
    for branch, spectrum in zip(out_types, spectra):
        if sp.issparse(spectrum):
            block = spectrum.tocsr()[np.ix_(coh_idx, det_idx)].toarray()
        else:
            block = np.asarray(spectrum)[np.ix_(coh_idx, det_idx)]
        cropped[str(branch)] = np.asarray(block)
    return metadata, config, axis_coh_cm[coh_idx], axis_det_cm[det_idx], cropped


def _case_label(config: dict[str, Any]) -> str:
    delta = float(config["atomic"]["delta_inhomogen_cm"])
    return "homogeneous" if abs(delta) < 1e-12 else "inhomogeneous"


def _find_realpart_peak(
    real_data: np.ndarray,
    axis_coh_cm: np.ndarray,
    axis_det_cm: np.ndarray,
    expected_cm: float,
    radius_cm: float,
) -> dict[str, float]:
    coh_grid, det_grid = np.meshgrid(axis_coh_cm, axis_det_cm, indexing="ij")
    mask = np.hypot(coh_grid - expected_cm, det_grid - expected_cm) <= radius_cm
    if not np.any(mask):
        raise RuntimeError(f"Empty search mask for expected peak {expected_cm:.2f} cm^-1")
    search = np.where(mask, np.abs(real_data), -np.inf)
    row, col = np.unravel_index(int(np.nanargmax(search)), real_data.shape)
    signed = float(real_data[row, col])
    return {
        "coh_cm": float(axis_coh_cm[row]),
        "det_cm": float(axis_det_cm[col]),
        "signed_real": signed,
        "abs_real": abs(signed),
    }


def _extract_job(job_dir: Path) -> list[dict[str, Any]]:
    metadata, config, axis_coh_cm, axis_det_cm, spectra = _complex_spectra_for_job(job_dir)
    peak_specs = _peak_specs(config)
    rows: list[dict[str, Any]] = []
    for branch in ("rephasing", "nonrephasing", "absorptive"):
        if branch not in spectra:
            continue
        real_data = np.asarray(np.real(np.asarray(spectra[branch])), dtype=float)
        peak_11 = _find_realpart_peak(
            real_data,
            axis_coh_cm,
            axis_det_cm,
            peak_specs[0].expected_cm,
            peak_specs[0].search_radius_cm,
        )
        peak_22 = _find_realpart_peak(
            real_data,
            axis_coh_cm,
            axis_det_cm,
            peak_specs[1].expected_cm,
            peak_specs[1].search_radius_cm,
        )
        ratio = np.nan
        if peak_22["abs_real"] > 0.0:
            ratio = peak_11["abs_real"] / peak_22["abs_real"]
        rows.append(
            {
                "job": job_dir.name,
                "job_dir": str(job_dir),
                "case": _case_label(config),
                "branch": branch,
                "t_wait_fs": float(config["config"]["t_wait"]),
                "delta_inhomogen_cm": float(config["atomic"]["delta_inhomogen_cm"]),
                "n_inhomogen": int(config["atomic"].get("n_inhomogen", metadata.get("n_inhom", 1))),
                "coupling_cm": float(config["atomic"].get("coupling_cm", 0.0)),
                "peak_11_coh_cm": peak_11["coh_cm"],
                "peak_11_det_cm": peak_11["det_cm"],
                "peak_11_real": peak_11["signed_real"],
                "peak_11_abs_real": peak_11["abs_real"],
                "peak_22_coh_cm": peak_22["coh_cm"],
                "peak_22_det_cm": peak_22["det_cm"],
                "peak_22_real": peak_22["signed_real"],
                "peak_22_abs_real": peak_22["abs_real"],
                "peak_11_to_22_ratio": ratio,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "job",
        "case",
        "branch",
        "t_wait_fs",
        "delta_inhomogen_cm",
        "n_inhomogen",
        "coupling_cm",
        "peak_11_coh_cm",
        "peak_11_det_cm",
        "peak_11_real",
        "peak_11_abs_real",
        "peak_22_coh_cm",
        "peak_22_det_cm",
        "peak_22_real",
        "peak_22_abs_real",
        "peak_11_to_22_ratio",
        "job_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _markdown_table(rows: list[dict[str, Any]], *, include_job: bool) -> str:
    header = []
    if include_job:
        header.append("Job")
    header.extend(
        [
            "Case",
            "Branch",
            "$t_{wait}$ / fs",
            "$J$ / cm$^{-1}$",
            "11: Re",
            "11: |Re|",
            "22: Re",
            "22: |Re|",
            "|11| / |22|",
        ]
    )
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        values: list[str] = []
        if include_job:
            values.append(str(row["job"]))
        values.extend(
            [
                str(row["case"]),
                str(row["branch"]),
                f"{float(row['t_wait_fs']):.0f}",
                f"{float(row['coupling_cm']):.0f}",
                _fmt(float(row["peak_11_real"])),
                _fmt(float(row["peak_11_abs_real"])),
                _fmt(float(row["peak_22_real"])),
                _fmt(float(row["peak_22_abs_real"])),
                _fmt(float(row["peak_11_to_22_ratio"])),
            ]
        )
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_job_markdown(job_dir: Path, rows: list[dict[str, Any]]) -> None:
    content = [
        "# Dimer Diagonal Real-Part Peak Balance",
        "",
        "Measured from the real-part spectra with the same FFT settings as the plotted figures.",
        "- quantity compared: local diagonal extrema in `Re S` near the 11 (lower) and 22 (upper) diagonal peaks",
        "- reported strength: `|Re S|` at those local extrema",
        "- comparison ratio: `|Re S_11| / |Re S_22|`",
        "",
        _markdown_table(rows, include_job=False),
        "",
    ]
    (job_dir / "diagonal_peak_realpart_summary.md").write_text(
        "\n".join(content),
        encoding="utf-8",
    )
    _write_csv(job_dir / "diagonal_peak_realpart_summary.csv", rows)


def _write_family_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    content = [
        "# Dimer Diagonal Real-Part Peak Balance",
        "",
        "This file compares the lower diagonal peak 11 and the upper diagonal peak 22 in the real-part spectra.",
        "Within each branch, the ratio `|Re S_11| / |Re S_22|` is unchanged by linear normalization, so it matches the visual peak balance in the corresponding real-part panels.",
        "",
        _markdown_table(rows, include_job=True),
        "",
        "Per-job copies are saved as `diagonal_peak_realpart_summary.md` and `diagonal_peak_realpart_summary.csv` inside each job directory.",
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")
    _write_csv(path.with_suffix(".csv"), rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_dirs", nargs="*", type=Path)
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=PROJECT_ROOT / "jobs" / "dimer" / "150426_use_those_in_thesis" / "n_inh_1000",
    )
    args = parser.parse_args()

    job_dirs = [Path(path).resolve() for path in (args.job_dirs or DEFAULT_JOB_DIRS)]
    all_rows: list[dict[str, Any]] = []
    for job_dir in job_dirs:
        print(f"Analyzing {job_dir}")
        rows = _extract_job(job_dir)
        _write_job_markdown(job_dir, rows)
        all_rows.extend(rows)

    summary_dir = Path(args.summary_dir).resolve()
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "dimer_diagonal_realpart_summary.md"
    _write_family_summary(summary_path, all_rows)
    print(f"Saved aggregate summary: {summary_path}")
    print(f"Saved aggregate CSV: {summary_path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
