"""Plot frequency distributions from a YAML config.

Primary target: atomic.frequencies_cm (cm^-1).
Optionally, generates an inhomogeneous ensemble using atomic.n_inhomogen and
atomic.delta_inhomogen_cm.

Usage (PowerShell):
  python .\plot_frequencies_yaml.py C:\path\to\config.yaml

Author: workspace utility script (no public API).
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import yaml


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _flatten_numbers(value: Any) -> list[float]:
    if _is_number(value):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        numbers: list[float] = []
        for item in value:
            numbers.extend(_flatten_numbers(item))
        return numbers
    return []


def _walk_find_frequency_lists(data: Any) -> list[float]:
    """Fallback: find numeric lists under keys containing 'freq'/'frequency'."""
    found: list[float] = []
    if isinstance(data, dict):
        for key, value in data.items():
            key_str = str(key).lower()
            if "freq" in key_str:
                found.extend(_flatten_numbers(value))
            found.extend(_walk_find_frequency_lists(value))
    elif isinstance(data, list):
        for item in data:
            found.extend(_walk_find_frequency_lists(item))
    return found


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_atomic_frequencies_cm(doc: Any) -> list[float]:
    if isinstance(doc, dict):
        atomic = doc.get("atomic")
        if isinstance(atomic, dict) and "frequencies_cm" in atomic:
            return _flatten_numbers(atomic.get("frequencies_cm"))
    # fallback
    return _walk_find_frequency_lists(doc)


def generate_inhomogeneous_ensemble(
    centers_cm: Iterable[float],
    n_inhomogen: int,
    delta_inhomogen_cm: float,
    *,
    delta_is_fwhm: bool,
    seed: int | None,
) -> np.ndarray:
    centers = np.asarray(list(centers_cm), dtype=float)
    if centers.size == 0:
        raise ValueError("No center frequencies provided")
    if n_inhomogen <= 0:
        raise ValueError("n_inhomogen must be > 0")
    if not math.isfinite(delta_inhomogen_cm) or delta_inhomogen_cm <= 0:
        raise ValueError("delta_inhomogen_cm must be > 0")

    # Interpret delta either as standard deviation or FWHM.
    sigma = (delta_inhomogen_cm / 2.354820045) if delta_is_fwhm else float(delta_inhomogen_cm)

    rng = np.random.default_rng(seed)

    # Produce n_inhomogen samples per center frequency.
    samples = []
    for center in centers:
        samples.append(rng.normal(loc=center, scale=sigma, size=int(n_inhomogen)))
    return np.concatenate(samples)


def plot_frequency_distribution(
    frequencies_cm: np.ndarray,
    *,
    title: str,
    output_path: Path,
    bins: int | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)

    if frequencies_cm.size == 0:
        raise ValueError("No frequencies found to plot")

    # Small-N: bar chart of unique values (nice for discrete frequencies).
    if frequencies_cm.size < 30:
        counts = Counter(np.round(frequencies_cm, 10))
        xs = np.array(sorted(counts.keys()), dtype=float)
        ys = np.array([counts[x] for x in xs], dtype=int)
        ax.bar(xs, ys, width=max(1.0, 0.002 * (xs.max() - xs.min() + 1.0)))
    else:
        ax.hist(frequencies_cm, bins=bins if bins is not None else "auto", edgecolor="black", linewidth=0.4)

    ax.set_xlabel(r"Frequenz (cm$^{-1}$)")
    ax.set_ylabel("Häufigkeit")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    fig.savefig(output_path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot Häufigkeit vs Frequenz from a YAML config")
    parser.add_argument("yaml", type=Path, help="Path to YAML file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: thesis_python/figures/frequencies_<yamlname>.png)",
    )
    parser.add_argument("--bins", type=int, default=None, help="Histogram bins (only used for many frequencies)")

    parser.add_argument(
        "--inhomogen",
        action="store_true",
        help="Generate and plot an inhomogeneous ensemble if parameters exist in the YAML",
    )
    parser.add_argument(
        "--delta-is-fwhm",
        action="store_true",
        help="Interpret delta_inhomogen_cm as FWHM instead of standard deviation",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (only for --inhomogen)")

    args = parser.parse_args()

    yaml_path: Path = args.yaml
    doc = load_yaml(yaml_path)

    base_frequencies = extract_atomic_frequencies_cm(doc)

    # Optional: inhomogeneous ensemble generation.
    frequencies = np.asarray(base_frequencies, dtype=float)
    title_suffix = ""

    if args.inhomogen and isinstance(doc, dict) and isinstance(doc.get("atomic"), dict):
        atomic = doc["atomic"]
        n_inhomogen = atomic.get("n_inhomogen")
        delta_inhomogen_cm = atomic.get("delta_inhomogen_cm")

        if _is_number(n_inhomogen) and _is_number(delta_inhomogen_cm):
            frequencies = generate_inhomogeneous_ensemble(
                base_frequencies,
                int(n_inhomogen),
                float(delta_inhomogen_cm),
                delta_is_fwhm=bool(args.delta_is_fwhm),
                seed=None if args.seed is None else int(args.seed),
            )
            title_suffix = f" (inhomogen, N={frequencies.size})"

    if args.output is None:
        default_out = Path(__file__).resolve().parents[1] / "figures" / f"frequencies_{yaml_path.stem}.png"
        output_path = default_out
    else:
        output_path = args.output

    title = f"Frequenzverteilung: {yaml_path.stem}{title_suffix}"
    plot_frequency_distribution(frequencies, title=title, output_path=output_path, bins=args.bins)

    print(f"Saved: {output_path}")
    print(f"Count: {frequencies.size}")
    if frequencies.size:
        print(f"Min/Max: {frequencies.min():.6g} / {frequencies.max():.6g} cm^-1")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
