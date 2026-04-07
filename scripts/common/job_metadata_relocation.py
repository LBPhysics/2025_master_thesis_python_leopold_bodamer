"""Helpers to rewrite stored job metadata after moving job directories."""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

JSON_KEYS = {
    "config_path",
    "config_stem",
    "data_base_name",
    "data_base_path",
    "data_dir",
    "figures_dir",
    "job_dir",
    "job_unique_id",
    "merged_config",
    "n_batches",
    "n_inhom",
    "n_t_coh",
    "signal_types",
    "sim_type",
    "t_coh",
    "t_det",
    "time_cut",
}

_PREFERRED_INFO_FILENAMES = (
    "raw.pkl",
    "2d_inhom_averaged.pkl",
    "1d_inhom_averaged.pkl",
    "0d_inhom_averaged.pkl",
)


@dataclass(frozen=True)
class JobRelocationResult:
    """Summary of the metadata files rewritten for one job directory."""

    job_dir: Path
    updated_json: bool
    created_json: bool
    updated_pkl: int
    slurm_files: tuple[Path, ...]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _sanitize_token(value: object) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    token = re.sub(r"_+", "_", token).strip("._-")
    return token or "run"


def config_stem_token(config_path: Path | str) -> str:
    """Return a filesystem-safe token derived from the config filename stem."""

    return _sanitize_token(Path(config_path).stem)


def extract_job_unique_id(job_dir: Path | str) -> str:
    """Extract the run-unique token from a job directory name."""

    name = Path(job_dir).name
    match = re.match(r"^(\d{2}_\d{6})(?:_|$)", name)
    if match:
        return match.group(1)
    match = re.search(r"(\d{8}_\d{6}(?:_\d{2})?)$", name)
    if match:
        return match.group(1)
    match = re.match(r"^(\d{8}_\d{6})(?:_|$)", name)
    if match:
        return match.group(1)
    match = re.search(r"(\d{2}_\d{6}(?:_\d{2})?)$", name)
    if match:
        return match.group(1)
    if "_" in name:
        return _sanitize_token(name.rsplit("_", 1)[-1])
    return _sanitize_token(name)


def discover_job_dirs(root: Path | str) -> list[Path]:
    """Return candidate job directories found under ``root``."""

    resolved_root = Path(root).expanduser().resolve()
    if not resolved_root.exists():
        raise FileNotFoundError(f"Root path not found: {resolved_root}")
    if not resolved_root.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {resolved_root}")

    job_dirs: set[Path] = set()
    for metadata_path in resolved_root.rglob("job_metadata.json"):
        job_dirs.add(metadata_path.parent.resolve())
    for info_path in resolved_root.rglob("*.pkl"):
        if info_path.parent.name == "data":
            job_dirs.add(info_path.parent.parent.resolve())

    return sorted(job_dirs)


def _candidate_info_paths(job_dir: Path) -> list[Path]:
    data_dir = job_dir / "data"
    if not data_dir.exists():
        return []

    candidates: list[Path] = []
    seen: set[Path] = set()
    for name in _PREFERRED_INFO_FILENAMES:
        candidate = (data_dir / name).resolve()
        if candidate.exists() and candidate not in seen:
            candidates.append(candidate)
            seen.add(candidate)

    for candidate in sorted(data_dir.glob("*.pkl")):
        resolved_candidate = candidate.resolve()
        if resolved_candidate not in seen:
            candidates.append(resolved_candidate)
            seen.add(resolved_candidate)

    return candidates


def _load_pickle_dict(path: Path) -> dict[str, Any] | None:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        return None
    return dict(payload)


def resolve_config_copy_path(
    job_dir: Path,
    *,
    old_config_path: object | None = None,
    config_stem: object | None = None,
) -> Path | None:
    """Resolve the copied YAML path stored inside the moved job directory."""

    if old_config_path:
        candidate = job_dir / Path(str(old_config_path)).name
        if candidate.exists():
            return candidate.resolve()

    if config_stem:
        candidate = job_dir / f"{config_stem}.yaml"
        if candidate.exists():
            return candidate.resolve()

    yamls = sorted(job_dir.glob("*.yaml"))
    if len(yamls) == 1:
        return yamls[0].resolve()
    return None


def rewrite_job_metadata(metadata: Mapping[str, Any], job_dir: Path | str) -> dict[str, Any]:
    """Rewrite job metadata so all stored paths point at ``job_dir``."""

    resolved_job_dir = Path(job_dir).expanduser().resolve()
    data_dir = (resolved_job_dir / "data").resolve()
    figures_dir = (resolved_job_dir / "figures").resolve()

    payload = dict(metadata)
    payload["job_dir"] = str(resolved_job_dir)
    payload["data_dir"] = str(data_dir)
    payload["figures_dir"] = str(figures_dir)

    data_base_name = str(payload.get("data_base_name") or "").strip()
    if data_base_name:
        payload["data_base_path"] = str((data_dir / data_base_name).resolve())

    config_path = resolve_config_copy_path(
        resolved_job_dir,
        old_config_path=payload.get("config_path"),
        config_stem=payload.get("config_stem"),
    )
    if config_path is not None:
        payload["config_path"] = str(config_path)
        if not str(payload.get("config_stem") or "").strip():
            payload["config_stem"] = config_stem_token(config_path)

    if not str(payload.get("job_unique_id") or "").strip():
        payload["job_unique_id"] = extract_job_unique_id(resolved_job_dir)

    return payload


def seed_job_metadata(job_dir: Path | str) -> dict[str, Any]:
    """Seed ``job_metadata.json`` from embedded metadata inside info ``.pkl`` files."""

    resolved_job_dir = Path(job_dir).expanduser().resolve()
    for info_path in _candidate_info_paths(resolved_job_dir):
        payload = _load_pickle_dict(info_path)
        if not payload:
            continue
        seeded = {key: payload[key] for key in JSON_KEYS if key in payload}
        if seeded:
            return seeded

    raise FileNotFoundError(f"No seed metadata found in {resolved_job_dir / 'data'}")


def relocate_job_dir(job_dir: Path | str, *, dry_run: bool = False) -> JobRelocationResult:
    """Rewrite JSON and info metadata for one moved job directory."""

    resolved_job_dir = Path(job_dir).expanduser().resolve()
    json_path = resolved_job_dir / "job_metadata.json"

    updated_json = False
    created_json = False

    if json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        rewritten = rewrite_job_metadata(payload, resolved_job_dir)
        updated_json = True
    else:
        rewritten = rewrite_job_metadata(seed_job_metadata(resolved_job_dir), resolved_job_dir)
        created_json = True

    if not dry_run:
        _write_json(json_path, rewritten)

    updated_pkl = 0
    for info_path in _candidate_info_paths(resolved_job_dir):
        payload = _load_pickle_dict(info_path)
        if not payload or not any(key in payload for key in JSON_KEYS):
            continue
        rewritten_info = rewrite_job_metadata(payload, resolved_job_dir)
        if not dry_run:
            with info_path.open("wb") as handle:
                pickle.dump(rewritten_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
        updated_pkl += 1

    return JobRelocationResult(
        job_dir=resolved_job_dir,
        updated_json=updated_json,
        created_json=created_json,
        updated_pkl=updated_pkl,
        slurm_files=tuple(sorted(resolved_job_dir.glob("*.slurm"))),
    )


def relocate_job_dirs(root: Path | str, *, dry_run: bool = False) -> list[JobRelocationResult]:
    """Rewrite job metadata for every discovered job directory under ``root``."""

    job_dirs = discover_job_dirs(root)
    if not job_dirs:
        resolved_root = Path(root).expanduser().resolve()
        raise FileNotFoundError(f"No job directories found under {resolved_root}")

    return [relocate_job_dir(job_dir, dry_run=dry_run) for job_dir in job_dirs]


def summarize_relocation(results: list[JobRelocationResult]) -> dict[str, int]:
    """Return aggregate counts for a relocation run."""

    return {
        "jobs": len(results),
        "updated_json": sum(int(result.updated_json) for result in results),
        "created_json": sum(int(result.created_json) for result in results),
        "updated_pkl": sum(result.updated_pkl for result in results),
        "slurm_files": sum(len(result.slurm_files) for result in results),
    }


__all__ = [
    "JSON_KEYS",
    "JobRelocationResult",
    "config_stem_token",
    "discover_job_dirs",
    "extract_job_unique_id",
    "relocate_job_dir",
    "relocate_job_dirs",
    "resolve_config_copy_path",
    "rewrite_job_metadata",
    "seed_job_metadata",
    "summarize_relocation",
]
