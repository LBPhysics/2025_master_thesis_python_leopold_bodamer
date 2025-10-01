#!/usr/bin/env python3
"""Post-process generalized batch outputs into averaged 1D and stacked 2D artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

LogFn = Optional[Callable[[str], None]]

import numpy as np

from avg_inhomogenity import average_inhom_1d
from stack_times import stack_artifacts
from qspectro2d.utils.data_io import load_run_artifact

# Keys we want to remove from artifact metadata after post-processing
_POSTPROC_META_KEYS_TO_DROP = {
    "sample_size",
    "sample_id",
    "sample_index",
    "batch_id",
    "combination_index",
    "t_index",
}


def _sanitize_artifact_metadata(path: Path, *, keys_to_drop: set[str]) -> Path:
    """Remove unwanted keys from the metadata_json of a run artifact in-place.

    This rewrites the .npz file preserving all arrays and only updating the
    embedded JSON string under the "metadata_json" key.
    """
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as bundle:
            contents = {key: bundle[key] for key in bundle.files}

        meta_key = "metadata_json"
        if meta_key not in contents:
            return path

        # Parse, drop, and re-serialize
        try:
            meta = json.loads(str(contents[meta_key].item()))
        except Exception:
            return path

        changed = False
        for k in list(keys_to_drop):
            if k in meta:
                meta.pop(k, None)
                changed = True

        if not changed:
            return path

        contents[meta_key] = np.array(json.dumps(meta, separators=(",", ":")), dtype=np.str_)

        tmp_path = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp_path, **contents)
        tmp_path.replace(path)
        return path
    except Exception:
        # Best-effort: ignore failures so post-processing can continue
        return path


def _load_metadata(job_dir: Path) -> dict[str, Any]:
    metadata_path = job_dir / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass(slots=True)
class ArtifactRecord:
    path: Path
    metadata: dict[str, Any]


@dataclass(slots=True)
class PostProcessResult:
    job_dir: Path
    data_dir: Path
    prefix: str
    sim_type: str
    sample_size: int
    averaged_paths: List[Path]
    final_path: Path
    stacked: bool


def _discover_artifacts(base_dir: Path, prefix: str) -> List[ArtifactRecord]:
    artifacts: List[ArtifactRecord] = []
    for candidate in sorted(base_dir.glob(f"{prefix}_run_t*_c*_s*.npz")):
        try:
            artifact = load_run_artifact(candidate)
        except Exception as exc:
            print(f"⚠️  Skipping unreadable artifact: {candidate}\n    {exc}")
            continue

        metadata = dict(artifact.get("metadata", {}))
        if not metadata:
            print(f"⚠️  Missing metadata in artifact: {candidate}")
            continue
        if "t_coh_value" not in metadata:
            raise KeyError(f"Missing 't_coh_value' in artifact metadata: {candidate}")
        metadata["t_coh_value"] = float(np.asarray(metadata["t_coh_value"]))
        artifacts.append(ArtifactRecord(path=candidate, metadata=metadata))

    return artifacts


def _group_by_t_index(records: Iterable[ArtifactRecord]) -> Dict[int, List[ArtifactRecord]]:
    grouped: Dict[int, List[ArtifactRecord]] = {}
    for record in records:
        t_index = int(record.metadata.get("t_index", 0))
        grouped.setdefault(t_index, []).append(record)
    return grouped


def _load_metadata_for_paths(paths: Iterable[Path]) -> List[ArtifactRecord]:
    records: List[ArtifactRecord] = []
    for path in paths:
        artifact = load_run_artifact(path)
        metadata = dict(artifact.get("metadata", {}))
        metadata["t_coh_value"] = float(np.asarray(metadata["t_coh_value"]))
        records.append(ArtifactRecord(path=path, metadata=metadata))
    return records


def _summarize_paths(label: str, paths: Iterable[Path], log: LogFn) -> None:
    if log is None:
        return

    paths_list = list(paths)
    log(f"{label} ({len(paths_list)} paths):")
    for path in paths_list:
        log(f"  - {path}")


def post_process_job(
    job_dir: Path | str,
    *,
    skip_inhom: bool = False,
    skip_stack: bool = False,
    log: LogFn = print,
) -> Optional[PostProcessResult]:
    job_path = Path(job_dir).resolve()

    def _log(message: str) -> None:
        if log is not None:
            log(message)

    _log("=" * 80)
    _log("POST-PROCESS GENERALIZED BATCHES")
    _log(f"Job directory: {job_path}")

    metadata = _load_metadata(job_path)
    sim_type = metadata.get("sim_type", "1d")
    sample_size = int(metadata.get("sample_size", 1) or 1)
    base_path = Path(metadata["data_base_path"]).resolve()
    data_dir = base_path.parent
    prefix = base_path.name

    _log(f"Sim type: {sim_type}")
    _log(f"Sample size: {sample_size}")
    _log(f"Artifact directory: {data_dir}")
    _log(f"Artifact prefix: {prefix}")

    records = _discover_artifacts(data_dir, prefix)
    if not records:
        _log("No run artifacts found. Ensure the batches finished and data_root is correct.")
        return None

    grouped = _group_by_t_index(records)
    _log(f"Discovered {len(grouped)} coherence index group(s).")

    averaged_paths: List[Path] = []
    for t_index in sorted(grouped):
        candidates = grouped[t_index]
        if not candidates:
            continue
        anchor_path = candidates[0].path
        _log("-" * 80)
        _log(f"Processing t_index={t_index} using anchor {anchor_path.name}")
        averaged_path = average_inhom_1d(anchor_path, skip_if_exists=skip_inhom)
        _log(f"  → Averaged artifact: {averaged_path}")
        # Sanitize metadata of the produced averaged artifact
        sanitized_path = _sanitize_artifact_metadata(
            Path(averaged_path), keys_to_drop=_POSTPROC_META_KEYS_TO_DROP
        )
        averaged_paths.append(sanitized_path)

    unique_averaged: List[Path] = []
    seen: set[str] = set()
    for path in averaged_paths:
        resolved = Path(path).resolve()
        resolved_key = str(resolved)
        if resolved_key not in seen:
            unique_averaged.append(resolved)
            seen.add(resolved_key)

    _log("=" * 80)
    _summarize_paths("Averaged artifacts", unique_averaged, log)

    if not unique_averaged:
        _log("No averaged artifacts produced; aborting.")
        return None

    # Determine whether stacking is required
    if sim_type == "1d" and len(grouped) == 1:
        _log("Single coherence point detected; stacking is not required.")
        # Ensure the single averaged artifact is sanitized
        final_path = _sanitize_artifact_metadata(
            unique_averaged[0], keys_to_drop=_POSTPROC_META_KEYS_TO_DROP
        ).resolve()
        _log("=" * 80)
        _log("🎯 To plot the averaged 1D data run:")
        _log(f"python plot_datas.py --abs_path {final_path}")
        return PostProcessResult(
            job_dir=job_path,
            data_dir=data_dir,
            prefix=prefix,
            sim_type=sim_type,
            sample_size=sample_size,
            averaged_paths=unique_averaged,
            final_path=final_path,
            stacked=False,
        )

    stack_anchor_candidates = unique_averaged if unique_averaged else [records[0].path]
    stack_anchor_records = _load_metadata_for_paths(stack_anchor_candidates)
    stack_anchor_records.sort(key=lambda rec: int(rec.metadata.get("t_index", 0)))
    stack_anchor_path = stack_anchor_records[0].path

    _log("Preparing to stack into 2D using anchor:")
    _log(f"  {stack_anchor_path}")

    stacked_path = stack_artifacts(stack_anchor_path, skip_if_exists=skip_stack).resolve()
    # Sanitize metadata of the final stacked artifact
    stacked_path = _sanitize_artifact_metadata(
        stacked_path, keys_to_drop=_POSTPROC_META_KEYS_TO_DROP
    ).resolve()

    _log("=" * 80)
    _log("🎯 To plot the 2D data run:")
    _log(f"python plot_datas.py --abs_path {stacked_path}")

    return PostProcessResult(
        job_dir=job_path,
        data_dir=data_dir,
        prefix=prefix,
        sim_type=sim_type,
        sample_size=sample_size,
        averaged_paths=unique_averaged,
        final_path=stacked_path,
        stacked=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Average + stack artifacts generated by HPC runs")
    parser.add_argument(
        "--job_dir",
        type=str,
        required=True,
        help="Path to batch_jobs_generalized/<job_label> (must contain metadata.json)",
    )
    parser.add_argument(
        "--skip_inhom",
        action="store_true",
        help="Reuse existing averaged artifacts if they already exist",
    )
    parser.add_argument(
        "--skip_stack",
        action="store_true",
        help="Reuse existing 2D artifact if it already exists",
    )
    args = parser.parse_args()

    result = post_process_job(
        args.job_dir,
        skip_inhom=args.skip_inhom,
        skip_stack=args.skip_stack,
    )

    if result is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
