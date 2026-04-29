"""Extract diagonal and anti-diagonal linewidths for dimer 2D spectra.

The script mirrors the monomer linewidth workflow, with one dimer-specific
addition: two diagonal features are measured separately. Peak search windows
are centred on the two single-exciton energies estimated from the dimer site
frequencies and coupling in each job's metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QSPECTRO_SRC = PROJECT_ROOT / "packages" / "qspectro2d" / "src"
if str(QSPECTRO_SRC) not in sys.path:
    sys.path.insert(0, str(QSPECTRO_SRC))

from qspectro2d.spectroscopy.post_processing import compute_spectra
from qspectro2d.visualization.plotting import convert_plot_axes


PAD_FACTOR = 50.0
APODIZATION = None
SECTION_LAB = ((1.5, 1.7), (1.5, 1.7))
CUT_STEP_CM = 0.5
ANTI_DIAG_HALF_RANGE_CM = 900.0

DEFAULT_JOB_DIRS = [
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
    / "10000"
    / "19_222916_dimer_fig3a_uncoupled_paper_eqs",
]


@dataclass(frozen=True)
class PeakSpec:
    label: str
    expected_cm: float
    search_radius_cm: float


@dataclass(frozen=True)
class FwhmResult:
    width_cm: float | None
    left_cm: float | None
    right_cm: float | None
    half_max: float
    peak_value: float
    status: str


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metadata_config(job_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = _load_json(job_dir / "job_metadata.json")
    config = metadata["merged_config"]
    return metadata, config


def _case_label(delta_inhomogen_cm: float) -> str:
    return "homogeneous" if abs(delta_inhomogen_cm) < 1e-12 else "inhomogeneous"


def _single_exciton_energies_cm(config: dict[str, Any]) -> list[float]:
    atomic = config["atomic"]
    freqs = [float(value) for value in atomic["frequencies_cm"]]
    if len(freqs) != 2:
        raise ValueError(f"Expected two dimer frequencies, got {freqs!r}")
    coupling = float(atomic.get("coupling_cm", 0.0))
    mean = 0.5 * (freqs[0] + freqs[1])
    half_delta = 0.5 * (freqs[0] - freqs[1])
    split = math.sqrt(half_delta * half_delta + coupling * coupling)
    return sorted([mean - split, mean + split])


def _peak_specs(config: dict[str, Any]) -> list[PeakSpec]:
    energies = _single_exciton_energies_cm(config)
    separation = abs(energies[1] - energies[0])
    radius = max(220.0, min(450.0, 0.45 * separation))
    return [
        PeakSpec("lower", energies[0], radius),
        PeakSpec("upper", energies[1], radius),
    ]


def _stored_section(config: dict[str, Any]) -> tuple[tuple[float, float], tuple[float, float]]:
    laser = config["laser"]
    if not bool(laser.get("rwa_sl", False)):
        return SECTION_LAB
    shift = float(laser["carrier_freq_cm"]) * 1e-4
    return (
        (SECTION_LAB[0][0] - shift, SECTION_LAB[0][1] - shift),
        (SECTION_LAB[1][0] - shift, SECTION_LAB[1][1] - shift),
    )


def _spectra_for_job(
    job_dir: Path,
    metadata: dict[str, Any],
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    data_path = job_dir / "data" / "2d_inhom_averaged.npz"
    with np.load(data_path, allow_pickle=False) as bundle:
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
        raise RuntimeError("Expected 2D spectra with a coherence-frequency axis")

    carrier = float(config["laser"]["carrier_freq_cm"])
    nu_coh_plot, nu_det_plot = convert_plot_axes(nu_coh, nu_det, carrier_freq_cm=carrier)
    axis_coh_cm = np.asarray(nu_coh_plot, dtype=float) * 1e4
    axis_det_cm = np.asarray(nu_det_plot, dtype=float) * 1e4

    coh_mask = (axis_coh_cm >= SECTION_LAB[0][0] * 1e4) & (
        axis_coh_cm <= SECTION_LAB[0][1] * 1e4
    )
    det_mask = (axis_det_cm >= SECTION_LAB[1][0] * 1e4) & (
        axis_det_cm <= SECTION_LAB[1][1] * 1e4
    )
    coh_idx = np.flatnonzero(coh_mask)
    det_idx = np.flatnonzero(det_mask)
    if coh_idx.size == 0 or det_idx.size == 0:
        raise RuntimeError("Empty frequency ROI after axis conversion")

    cropped: dict[str, np.ndarray] = {}
    for signal_type, spectrum in zip(out_types, spectra):
        if sp.issparse(spectrum):
            block = spectrum.tocsr()[np.ix_(coh_idx, det_idx)].toarray()
        else:
            block = np.asarray(spectrum)[np.ix_(coh_idx, det_idx)]
        cropped[str(signal_type)] = np.abs(np.asarray(block))

    return axis_coh_cm[coh_idx], axis_det_cm[det_idx], cropped


def _find_peak(
    magnitude: np.ndarray,
    axis_coh_cm: np.ndarray,
    axis_det_cm: np.ndarray,
    spec: PeakSpec,
) -> tuple[int, int, float, float, float]:
    coh_grid, det_grid = np.meshgrid(axis_coh_cm, axis_det_cm, indexing="ij")
    distance = np.hypot(coh_grid - spec.expected_cm, det_grid - spec.expected_cm)
    mask = distance <= spec.search_radius_cm
    if not np.any(mask):
        raise RuntimeError(f"Empty peak-search mask for {spec.label} peak")
    local = np.where(mask, magnitude, -np.inf)
    row, col = np.unravel_index(int(np.nanargmax(local)), magnitude.shape)
    return (
        int(row),
        int(col),
        float(axis_coh_cm[row]),
        float(axis_det_cm[col]),
        float(magnitude[row, col]),
    )


def _interp2(
    values: np.ndarray,
    axis_x: np.ndarray,
    axis_y: np.ndarray,
    xq: np.ndarray,
    yq: np.ndarray,
) -> np.ndarray:
    xq = np.asarray(xq, dtype=float)
    yq = np.asarray(yq, dtype=float)
    out = np.full(xq.shape, np.nan, dtype=float)
    valid = (
        (xq >= axis_x[0])
        & (xq <= axis_x[-1])
        & (yq >= axis_y[0])
        & (yq <= axis_y[-1])
    )
    if not np.any(valid):
        return out

    xv = xq[valid]
    yv = yq[valid]
    ix = np.searchsorted(axis_x, xv, side="right") - 1
    iy = np.searchsorted(axis_y, yv, side="right") - 1
    ix = np.clip(ix, 0, axis_x.size - 2)
    iy = np.clip(iy, 0, axis_y.size - 2)

    x0 = axis_x[ix]
    x1 = axis_x[ix + 1]
    y0 = axis_y[iy]
    y1 = axis_y[iy + 1]
    wx = np.divide(xv - x0, x1 - x0, out=np.zeros_like(xv), where=(x1 != x0))
    wy = np.divide(yv - y0, y1 - y0, out=np.zeros_like(yv), where=(y1 != y0))

    v00 = values[ix, iy]
    v10 = values[ix + 1, iy]
    v01 = values[ix, iy + 1]
    v11 = values[ix + 1, iy + 1]
    out[valid] = (
        (1.0 - wx) * (1.0 - wy) * v00
        + wx * (1.0 - wy) * v10
        + (1.0 - wx) * wy * v01
        + wx * wy * v11
    )
    return out


def _half_crossing(
    s0: float,
    v0: float,
    s1: float,
    v1: float,
    half: float,
) -> float:
    if v1 == v0:
        return float(s0)
    frac = (half - v0) / (v1 - v0)
    return float(s0 + frac * (s1 - s0))


def _fwhm_from_cut(s_values: np.ndarray, cut_values: np.ndarray) -> FwhmResult:
    finite = np.isfinite(cut_values)
    if not np.any(finite):
        return FwhmResult(None, None, None, math.nan, math.nan, "no finite cut samples")

    center_mask = finite & (np.abs(s_values) <= 30.0)
    if not np.any(center_mask):
        return FwhmResult(None, None, None, math.nan, math.nan, "no finite centre samples")
    center_indices = np.flatnonzero(center_mask)
    peak_index = center_indices[int(np.argmax(cut_values[center_indices]))]
    peak_value = float(cut_values[peak_index])
    half = 0.5 * peak_value

    left_cross = None
    for idx in range(peak_index - 1, -1, -1):
        if not np.isfinite(cut_values[idx]):
            continue
        if cut_values[idx] <= half <= cut_values[idx + 1]:
            left_cross = _half_crossing(
                float(s_values[idx]),
                float(cut_values[idx]),
                float(s_values[idx + 1]),
                float(cut_values[idx + 1]),
                half,
            )
            break

    right_cross = None
    for idx in range(peak_index + 1, cut_values.size):
        if not np.isfinite(cut_values[idx]):
            continue
        if cut_values[idx] <= half <= cut_values[idx - 1]:
            right_cross = _half_crossing(
                float(s_values[idx - 1]),
                float(cut_values[idx - 1]),
                float(s_values[idx]),
                float(cut_values[idx]),
                half,
            )
            break

    if left_cross is None or right_cross is None:
        return FwhmResult(
            None,
            left_cross,
            right_cross,
            half,
            peak_value,
            "half maximum not crossed on both sides",
        )
    return FwhmResult(
        float(right_cross - left_cross),
        left_cross,
        right_cross,
        half,
        peak_value,
        "ok",
    )


def _measure_cut(
    magnitude: np.ndarray,
    axis_coh_cm: np.ndarray,
    axis_det_cm: np.ndarray,
    peak_coh_cm: float,
    peak_det_cm: float,
    *,
    direction: str,
    half_range_cm: float,
) -> FwhmResult:
    s_values = np.arange(-half_range_cm, half_range_cm + CUT_STEP_CM, CUT_STEP_CM)
    if direction == "diagonal":
        dx = s_values / math.sqrt(2.0)
        dy = s_values / math.sqrt(2.0)
    elif direction == "anti-diagonal":
        dx = s_values / math.sqrt(2.0)
        dy = -s_values / math.sqrt(2.0)
    else:
        raise ValueError(f"Unsupported cut direction: {direction}")

    cut = _interp2(
        magnitude,
        axis_coh_cm,
        axis_det_cm,
        peak_coh_cm + dx,
        peak_det_cm + dy,
    )
    return _fwhm_from_cut(s_values, cut)


def _format_optional(value: float | None) -> str:
    return "" if value is None or not np.isfinite(value) else f"{value:.2f}"


def _extract_job(job_dir: Path) -> list[dict[str, Any]]:
    metadata, config = _metadata_config(job_dir)
    atomic = config["atomic"]
    t_wait = float(config["config"]["t_wait"])
    delta = float(atomic["delta_inhomogen_cm"])
    coupling = float(atomic.get("coupling_cm", 0.0))
    n_inhom = int(atomic.get("n_inhomogen", metadata.get("n_inhom", 1)))
    case = _case_label(delta)
    peak_specs = _peak_specs(config)
    separation = abs(peak_specs[1].expected_cm - peak_specs[0].expected_cm)
    diag_half_range = min(900.0, max(350.0, 0.48 * math.sqrt(2.0) * separation))

    axis_coh_cm, axis_det_cm, spectra = _spectra_for_job(job_dir, metadata, config)

    rows: list[dict[str, Any]] = []
    for branch in ("rephasing", "nonrephasing", "absorptive"):
        if branch not in spectra:
            continue
        magnitude = spectra[branch]
        for spec in peak_specs:
            _, _, peak_coh, peak_det, peak_height = _find_peak(
                magnitude,
                axis_coh_cm,
                axis_det_cm,
                spec,
            )
            diag = _measure_cut(
                magnitude,
                axis_coh_cm,
                axis_det_cm,
                peak_coh,
                peak_det,
                direction="diagonal",
                half_range_cm=diag_half_range,
            )
            anti = _measure_cut(
                magnitude,
                axis_coh_cm,
                axis_det_cm,
                peak_coh,
                peak_det,
                direction="anti-diagonal",
                half_range_cm=ANTI_DIAG_HALF_RANGE_CM,
            )
            rows.append(
                {
                    "job": job_dir.name,
                    "job_dir": str(job_dir),
                    "case": case,
                    "t_wait_fs": t_wait,
                    "delta_inhomogen_cm": delta,
                    "n_inhomogen": n_inhom,
                    "coupling_cm": coupling,
                    "site_frequencies_cm": ",".join(
                        f"{float(value):.2f}" for value in atomic["frequencies_cm"]
                    ),
                    "branch": branch,
                    "peak": spec.label,
                    "expected_peak_cm": spec.expected_cm,
                    "search_radius_cm": spec.search_radius_cm,
                    "peak_coh_cm": peak_coh,
                    "peak_det_cm": peak_det,
                    "peak_height": peak_height,
                    "diag_fwhm_cm": diag.width_cm,
                    "anti_diag_fwhm_cm": anti.width_cm,
                    "diag_status": diag.status,
                    "anti_diag_status": anti.status,
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "job",
        "case",
        "t_wait_fs",
        "delta_inhomogen_cm",
        "n_inhomogen",
        "coupling_cm",
        "site_frequencies_cm",
        "branch",
        "peak",
        "expected_peak_cm",
        "peak_coh_cm",
        "peak_det_cm",
        "diag_fwhm_cm",
        "anti_diag_fwhm_cm",
        "diag_status",
        "anti_diag_status",
        "search_radius_cm",
        "peak_height",
        "job_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _markdown_table(rows: list[dict[str, Any]], *, include_job: bool) -> str:
    header = [
        "Job" if include_job else None,
        "Case",
        "Branch",
        "Peak",
        "$t_{wait}$ / fs",
        "$J$ / cm$^{-1}$",
        "$\\Delta_{inh}$ / cm$^{-1}$",
        "Peak position / cm$^{-1}$",
        "Diag. FWHM / cm$^{-1}$",
        "Anti-diag. FWHM / cm$^{-1}$",
    ]
    header = [value for value in header if value is not None]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        values = []
        if include_job:
            values.append(str(row["job"]))
        values.extend(
            [
                str(row["case"]),
                str(row["branch"]),
                str(row["peak"]),
                f"{float(row['t_wait_fs']):.0f}",
                f"{float(row['coupling_cm']):.0f}",
                f"{float(row['delta_inhomogen_cm']):.0f}",
                f"{float(row['peak_coh_cm']):.2f}, {float(row['peak_det_cm']):.2f}",
                _format_optional(row["diag_fwhm_cm"]),
                _format_optional(row["anti_diag_fwhm_cm"]),
            ]
        )
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_job_markdown(job_dir: Path, rows: list[dict[str, Any]]) -> None:
    metadata, config = _metadata_config(job_dir)
    atomic = config["atomic"]
    content = [
        "# Dimer Linewidth Summary",
        "",
        "Measured from `data/2d_inhom_averaged.npz` with the same frequency-domain settings as the plotting pipeline:",
        f"- pad factor: `{PAD_FACTOR:g}`",
        "- apodization: `None`",
        "- lab-frame ROI: `1.5..1.7 x 10^4 cm^-1` on both axes",
        "- measured quantity: `|S(omega_coh, omega_det)|` for each branch",
        "- branch handling: `absorptive = Re(rephasing + nonrephasing)`, then FWHM measured on its absolute magnitude",
        "- peak selection: local maxima near the two single-exciton diagonal positions from the job metadata",
        "",
        "Metadata:",
        f"- physical case: `{_case_label(float(atomic['delta_inhomogen_cm']))}`",
        f"- `t_wait`: `{float(config['config']['t_wait']):.0f} fs`",
        f"- `delta_inhomogen_cm`: `{float(atomic['delta_inhomogen_cm']):.2f}`",
        f"- `n_inhomogen`: `{int(atomic.get('n_inhomogen', metadata.get('n_inhom', 1)))}`",
        f"- `coupling_cm`: `{float(atomic.get('coupling_cm', 0.0)):.2f}`",
        f"- `frequencies_cm`: `{atomic['frequencies_cm']}`",
        "",
        _markdown_table(rows, include_job=False),
        "",
        "Statuses are written in `linewidth_summary.csv`; blank FWHM cells indicate that the half-maximum crossing was not found on both sides of the local peak.",
        "",
    ]
    (job_dir / "linewidth_summary.md").write_text("\n".join(content), encoding="utf-8")
    _write_csv(job_dir / "linewidth_summary.csv", rows)


def _write_family_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    seen_jobs: set[str] = set()
    metadata_lines: list[str] = []
    for row in rows:
        job = str(row["job"])
        if job in seen_jobs:
            continue
        seen_jobs.add(job)
        metadata_lines.append(
            "- "
            f"`{job}`: {row['case']}, "
            f"`t_wait = {float(row['t_wait_fs']):.0f} fs`, "
            f"`Delta_inh = {float(row['delta_inhomogen_cm']):.0f} cm^-1`, "
            f"`N_inhom = {int(row['n_inhomogen'])}`, "
            f"`J = {float(row['coupling_cm']):.0f} cm^-1`"
        )

    content = [
        "# Dimer Linewidth Summary",
        "",
        "This file aggregates the dimer two-peak diagonal/anti-diagonal FWHM extraction for the selected thesis jobs.",
        "",
        "Metadata case labels are assigned directly from `delta_inhomogen_cm` in each job's `job_metadata.json`:",
        *metadata_lines,
        "",
        _markdown_table(rows, include_job=True),
        "",
        "Detailed per-job copies are saved as `linewidth_summary.md` and `linewidth_summary.csv` inside each job directory.",
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")
    _write_csv(path.with_suffix(".csv"), rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "job_dirs",
        nargs="*",
        type=Path,
        help="Dimer job directories. Defaults to the four thesis dimer jobs.",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=PROJECT_ROOT
        / "jobs"
        / "dimer"
        / "150426_use_those_in_thesis"
        / "n_inh_1000",
        help="Directory for the aggregate markdown/csv summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    job_dirs = [Path(path).resolve() for path in (args.job_dirs or DEFAULT_JOB_DIRS)]

    all_rows: list[dict[str, Any]] = []
    for job_dir in job_dirs:
        if not job_dir.exists():
            raise FileNotFoundError(job_dir)
        print(f"Extracting {job_dir}")
        rows = _extract_job(job_dir)
        _write_job_markdown(job_dir, rows)
        all_rows.extend(rows)

    summary_dir = Path(args.summary_dir).resolve()
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "dimer_linewidth_summary.md"
    _write_family_summary(summary_path, all_rows)
    print(f"Saved aggregate summary: {summary_path}")
    print(f"Saved aggregate CSV: {summary_path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
