"""Helpers for allocating and managing per-run job directories.

The local and HPC workflows both create a dedicated "job directory" that
collects raw artifacts, processed outputs, figures, and metadata in one
place.  This module provides tiny utilities to derive those paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qspectro2d.utils.file_naming import generate_base_sub_dir


@dataclass(frozen=True)
class JobPaths:
    """Resolved paths for a spectroscopy job run."""

    job_dir: Path
    data_dir: Path
    figures_dir: Path
    base_name: str

    @property
    def data_base_path(self) -> Path:
        """Base path used for raw artifacts (stem only)."""

        return self.data_dir / self.base_name


def ensure_job_layout(job_dir: Path, base_name: str = "run") -> JobPaths:
    """Create the canonical job layout and return the resolved paths."""

    resolved_job = job_dir.resolve()
    data_dir = resolved_job / "jobs"
    figures_dir = resolved_job / "figures"

    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    return JobPaths(
        job_dir=resolved_job,
        data_dir=data_dir,
        figures_dir=figures_dir,
        base_name=base_name,
    )


def allocate_job_dir(root: Path, base_label: str) -> Path:
    """Allocate a unique job directory under ``root`` using ``base_label``."""

    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    candidate = root / base_label
    if not candidate.exists():
        try:
            candidate.mkdir(parents=True)
            return candidate
        except FileExistsError:
            pass

    counter = 1
    while True:
        candidate = root / f"{base_label}_{counter:02d}"
        if not candidate.exists():
            try:
                candidate.mkdir(parents=True)
                return candidate
            except FileExistsError:
                pass
        counter += 1


def job_label_token(sim_config, system, *, sim_type: str | None = None) -> str:
    """Return a flattened identifier derived from the config and system."""

    base = generate_base_sub_dir(sim_config, system)
    parts = [part for part in base.parts if part and str(part).lower() != "none"]
    prefix = (sim_type or getattr(sim_config, "sim_type", None) or "sim").strip()
    token_parts = [prefix, *parts]
    return "_".join(token_parts)
