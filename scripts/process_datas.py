"""Post-process generalized batch outputs into averaged 1D and stacked 2D artifacts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

LogFn = Optional[Callable[[str], None]]

import numpy as np

from avg_inhomogenity import average_inhom_1d, _load_entry, SimulationSnapshot, save_run_artifact, save_info_file, _generate_base_stem
from stack_times import stack_artifacts
from qspectro2d.utils.data_io import load_run_artifact


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
    averaged_paths: List[Path]
    final_path: Path
    stacked: bool


def _discover_artifacts(base_dir: Path, prefix: str) -> List[ArtifactRecord]:
    artifacts: List[ArtifactRecord] = []
    for candidate in sorted(base_dir.glob(f"{prefix}_run_t*_c*.npz")):
        try:
            artifact = load_run_artifact(candidate)
        except Exception as exc:
            print(f"⚠️  Skipping unreadable artifact: {candidate}\n    {exc}")
            continue

        metadata = dict(artifact.get("metadata", {}))
        if not metadata:
            print(f"⚠️  Missing metadata in artifact: {candidate}")
            continue
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
        if "t_coh_value" in metadata:
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

    print("=" * 80)
    print("POST-PROCESS GENERALIZED BATCHES")
    print(f"Job directory: {job_path}")

    job_metadata = _load_metadata(job_path)
    sim_type = job_metadata.get("sim_type", "1d")
    base_path = Path(job_metadata["data_base_path"]).resolve()
    data_dir = base_path.parent
    prefix = base_path.name

    print(f"Sim type: {sim_type}")
    print(f"Artifact directory: {data_dir}")
    print(f"Artifact prefix: {prefix}")

    records = _discover_artifacts(data_dir, prefix)
    if not records:
        print("No run artifacts found. Ensure the batches finished and data_root is correct.")
        return None

    grouped = _group_by_t_index(records)
    print(f"Discovered {len(grouped)} coherence index group(s).")

    raw_records = [rec for rec in records if not rec.metadata.get("inhom_averaged", False)]
    grouped_by_t = defaultdict(list)
    for rec in raw_records:
        t_index = rec.metadata.get("t_index")
        grouped_by_t[t_index].append(rec)

    averaged_paths: List[Path] = []
    for t_index in sorted(grouped_by_t):
        group = grouped_by_t[t_index]
        print("-" * 80)
        print(f"Processing t_index={t_index} ({len(group)} artifacts)")
        if len(group) == 1:
            # Single entry
            rec = group[0]
            entry = _load_entry(rec.path)
            signal_types = list(entry.metadata.get("signal_types", entry.signals.keys()))

            metadata_out = dict(entry.metadata)
            metadata_out.update(
                {
                    "inhom_averaged": True,
                    "averaged_count": 1,
                    "source_artifacts": [entry.path.name],
                }
            )

            sim_cfg = replace(
                entry.simulation_config,
                inhom_averaged=True,
            )

            snapshot = SimulationSnapshot(
                system=entry.system,
                simulation_config=sim_cfg,
                laser=entry.laser,
                bath=entry.bath,
            )

            extra_payload: dict[str, Any] | None = None
            if entry.job_metadata or entry.bath:
                extra_payload = {}
                if entry.job_metadata:
                    extra_payload["job_metadata"] = entry.job_metadata
                if entry.bath:
                    extra_payload["bath"] = entry.bath

            expected_dir = entry.path.parent
            prefix = _generate_base_stem(snapshot.simulation_config)
            expected_path = (
                expected_dir / f"{prefix}_run_t{int(t_index):03d}_c{0:04d}.npz"
            )

            if skip_inhom and expected_path.exists():
                averaged_path = expected_path
                print(f"⏭️  Averaged artifact already present: {averaged_path}")
            else:
                save_info_file(
                    expected_dir / f"{prefix}.pkl",
                    snapshot.system,
                    snapshot.simulation_config,
                    bath=snapshot.bath,
                    laser=snapshot.laser,
                    extra_payload=extra_payload,
                )

                out_path = save_run_artifact(
                    snapshot,
                    signal_arrays=[entry.signals[sig] for sig in signal_types],
                    t_det=entry.t_det,
                    metadata=metadata_out,
                    frequency_sample_cm=entry.frequency_sample_cm,
                    data_dir=expected_dir,
                    prefix=prefix,
                    t_coh=metadata_out.get("t_coh_value"),
                )

                averaged_path = out_path
                print(f"⏭️ Single artifact treated as averaged for t_index={t_index} → {averaged_path}")

            sanitized_path = averaged_path
            averaged_paths.append(sanitized_path)
        else:
            # Multi
            anchor_path = group[0].path
            averaged_path = average_inhom_1d(anchor_path, skip_if_exists=skip_inhom)
            print(f"✅ Averaged {len(group)} artifacts for t_index={t_index} → {averaged_path}")
            sanitized_path = averaged_path
            averaged_paths.append(sanitized_path)

    unique_averaged: List[Path] = []
    seen: set[str] = set()
    for path in averaged_paths:
        resolved = Path(path).resolve()
        resolved_key = str(resolved)
        if resolved_key not in seen:
            unique_averaged.append(resolved)
            seen.add(resolved_key)

    print("=" * 80)
    _summarize_paths("Averaged artifacts", unique_averaged, log)

    if not unique_averaged:
        print("No averaged artifacts produced; aborting.")
        return None

    # Determine whether stacking is required
    if sim_type == "1d" and len(grouped) == 1:
        print("Single coherence point detected; stacking is not required.")
        # Ensure the single averaged artifact is sanitized
        final_path = unique_averaged[0].resolve()
        print("=" * 80)
        print("🎯 To plot the averaged 1D data run:")
        print(f"python plot_datas.py --abs_path {final_path}")
        return PostProcessResult(
            job_dir=job_path,
            data_dir=data_dir,
            prefix=prefix,
            sim_type=sim_type,
            averaged_paths=unique_averaged,
            final_path=final_path,
            stacked=False,
        )

    stack_anchor_candidates = unique_averaged if unique_averaged else [records[0].path]
    stack_anchor_records = _load_metadata_for_paths(stack_anchor_candidates)
    stack_anchor_records.sort(key=lambda rec: int(rec.metadata.get("t_index", 0)))
    stack_anchor_path = stack_anchor_records[0].path

    print("Preparing to stack into 2D using anchor:")
    print(f"  {stack_anchor_path}")

    stacked_path = stack_artifacts(stack_anchor_path, skip_if_exists=skip_stack).resolve()
    # Sanitize metadata of the final stacked artifact
    print("=" * 80)
    print("🎯 To plot the 2D data run:")
    print(f"python plot_datas.py --abs_path {stacked_path}")

    return PostProcessResult(
        job_dir=job_path,
        data_dir=data_dir,
        prefix=prefix,
        sim_type=sim_type,
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
        help="Path to batch_jobs/<job_label> (must contain metadata.json)",
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
