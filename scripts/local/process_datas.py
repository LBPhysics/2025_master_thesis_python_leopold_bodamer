"""Process spectroscopy data: stack per sample and average across samples."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import argparse
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from qspectro2d.utils.data_io import (
    load_run_artifact,
    save_info_file,
    save_run_artifact,
    split_prefix,
)

from common.workflow import PROJECT_ROOT, final_processed_filename

print = partial(print, flush=True)


def _format_seconds(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    mins, secs = divmod(seconds, 60)
    if mins < 60:
        return f"{int(mins)}m {secs:04.1f}s"
    hours, mins = divmod(mins, 60)
    return f"{int(hours)}h {int(mins)}m {secs:04.1f}s"


@dataclass(slots=True)
class RunEntry:
    path: Path
    metadata: dict[str, Any]
    signals: dict[str, np.ndarray]
    t_det: np.ndarray
    frequency_sample_cm: np.ndarray
    simulation_config: Any
    system: Any
    laser: Any | None
    bath: Any | None
    t_coh: np.ndarray | None = None
    job_metadata: dict[str, Any] | None = None


def _load_entry(path: Path) -> RunEntry:
    artifact = load_run_artifact(path)
    metadata = dict(artifact["metadata"])
    signals = {key: np.asarray(val) for key, val in artifact["signals"].items()}
    t_det = np.asarray(artifact["t_det"], dtype=float)
    freq_sample = np.asarray(artifact["frequency_sample_cm"], dtype=float)
    t_coh = np.asarray(artifact["t_coh"], dtype=float) if artifact["t_coh"] is not None else None

    sim_cfg = artifact["simulation_config"]
    system = artifact["system"]
    if sim_cfg is None or system is None:
        raise ValueError(f"Artifact {path} is missing simulation context")

    if sim_cfg.sim_type == "1d":
        t_coh = None

    return RunEntry(
        path=path,
        metadata=metadata,
        signals=signals,
        t_det=t_det,
        frequency_sample_cm=freq_sample,
        simulation_config=sim_cfg,
        system=system,
        laser=artifact["laser"],
        bath=artifact["bath"],
        t_coh=t_coh,
        job_metadata=artifact["job_metadata"],
    )


def _discover_entries(anchor: RunEntry) -> list[RunEntry]:
    entries: list[RunEntry] = []
    for candidate in sorted(anchor.path.parent.glob("*_run_t*_s*.npz")):
        entry = _load_entry(candidate)
        if entry.metadata.get("sim_type") == "2d":
            continue
        if entry.metadata.get("run_status", "ok") != "ok":
            print(f"    ⚠️ Skipping {candidate.name}: run_status={entry.metadata.get('run_status')}")
            continue
        entries.append(entry)
    return entries


def _group_by_sample(entries: list[RunEntry]) -> dict[int, list[RunEntry]]:
    groups: dict[int, list[RunEntry]] = defaultdict(list)
    for entry in entries:
        groups[int(entry.metadata["sample_index"])].append(entry)
    return dict(groups)


def _stack_group_to_2d(group: list[RunEntry]) -> RunEntry:
    if len(group) <= 1:
        return group[0]

    group_sorted = sorted(group, key=lambda e: float(e.metadata["t_coh_value"]))
    reference = group_sorted[0]
    signal_types = list(reference.metadata["signal_types"])

    tol = 0.5 * float(reference.simulation_config.dt)
    collapsed: list[RunEntry] = []
    for entry in group_sorted:
        current_t = float(entry.metadata["t_coh_value"])
        if collapsed and np.isclose(
            current_t, float(collapsed[-1].metadata["t_coh_value"]), atol=tol
        ):
            for sig in signal_types:
                collapsed[-1].signals[sig] = 0.5 * (collapsed[-1].signals[sig] + entry.signals[sig])
            continue
        collapsed.append(entry)

    stacked_signals = {
        sig: np.stack([entry.signals[sig] for entry in collapsed], axis=0) for sig in signal_types
    }
    t_coh_axis = np.asarray([entry.metadata["t_coh_value"] for entry in collapsed], dtype=float)

    if not np.all(np.diff(t_coh_axis) >= 0):
        sort_idx = np.argsort(t_coh_axis)
        t_coh_axis = t_coh_axis[sort_idx]
        stacked_signals = {sig: arr[sort_idx] for sig, arr in stacked_signals.items()}

    metadata_2d = dict(reference.metadata)
    metadata_2d.update({"sim_type": "2d", "stacked_points": len(collapsed)})
    metadata_2d.pop("t_coh_value", None)
    metadata_2d.pop("t_index", None)
    metadata_2d.pop("combination_index", None)

    sim_cfg = replace(
        reference.simulation_config,
        sim_type="2d",
        inhom_averaged=bool(reference.metadata.get("inhom_averaged")),
    )

    return RunEntry(
        path=reference.path,
        metadata=metadata_2d,
        signals=stacked_signals,
        t_det=reference.t_det,
        frequency_sample_cm=reference.frequency_sample_cm,
        simulation_config=sim_cfg,
        system=reference.system,
        laser=reference.laser,
        bath=reference.bath,
        t_coh=t_coh_axis,
        job_metadata=reference.job_metadata,
    )


def _average_entries(entries: list[RunEntry]) -> RunEntry:
    if len(entries) == 1:
        single = entries[0]
        if single.simulation_config.inhom_averaged:
            return single
        return RunEntry(
            path=single.path,
            metadata={**single.metadata, "inhom_averaged": True, "averaged_count": 1},
            signals=single.signals,
            t_det=single.t_det,
            frequency_sample_cm=single.frequency_sample_cm,
            simulation_config=replace(single.simulation_config, inhom_averaged=True),
            system=single.system,
            laser=single.laser,
            bath=single.bath,
            t_coh=single.t_coh,
            job_metadata=single.job_metadata,
        )

    reference = entries[0]
    signal_types = list(reference.metadata.get("signal_types", reference.signals.keys()))

    for entry in entries[1:]:
        current = list(entry.metadata.get("signal_types", entry.signals.keys()))
        if current != signal_types:
            raise ValueError(f"Inconsistent signals for {entry.path}")
        if entry.t_coh is None and reference.t_coh is not None:
            raise ValueError(f"Missing t_coh axis in {entry.path}")
        if entry.t_coh is not None and reference.t_coh is not None:
            if entry.t_coh.shape != reference.t_coh.shape or not np.allclose(
                entry.t_coh, reference.t_coh
            ):
                raise ValueError(
                    "Inconsistent t_coh axis across samples; "
                    f"reference={reference.path}, offending={entry.path}"
                )

    averaged_signals = {
        sig: np.mean(np.stack([entry.signals[sig] for entry in entries], axis=0), axis=0)
        for sig in signal_types
    }
    avg_freq = np.mean(np.stack([entry.frequency_sample_cm for entry in entries], axis=0), axis=0)

    metadata_out = dict(reference.metadata)
    metadata_out.update({"inhom_averaged": True, "averaged_count": len(entries)})
    metadata_out.pop("sample_index", None)
    metadata_out.pop("combination_index", None)

    return RunEntry(
        path=reference.path,
        metadata=metadata_out,
        signals=averaged_signals,
        t_det=reference.t_det,
        frequency_sample_cm=avg_freq,
        simulation_config=replace(reference.simulation_config, inhom_averaged=True),
        system=reference.system,
        laser=reference.laser,
        bath=reference.bath,
        t_coh=reference.t_coh,
        job_metadata=reference.job_metadata,
    )


def process_datas(abs_path: Path, *, skip_if_exists: bool = False) -> Path:
    start_all = time.perf_counter()
    abs_path = abs_path.expanduser().resolve()
    print(f"Starting process_datas for: {abs_path}")
    print(f"Skip if exists: {skip_if_exists}")

    anchor = _load_entry(abs_path)
    print(
        "Loaded anchor artifact: "
        f"signals={list(anchor.signals.keys())}, "
        f"t_det={anchor.t_det.shape}, "
        f"t_coh={'None' if anchor.t_coh is None else anchor.t_coh.shape}"
    )

    entries = _discover_entries(anchor)
    print(f"Discovered {len(entries)} input artifacts")
    if not entries:
        raise FileNotFoundError("No raw run artifacts found for processing")

    groups = _group_by_sample(entries)
    print(f"Grouped into {len(groups)} samples: {sorted((k, len(v)) for k, v in groups.items())}")

    processed_per_sample: list[RunEntry] = []
    for sample_idx, group in groups.items():
        group_start = time.perf_counter()
        if len(group) > 1:
            print(f"Stacking sample {sample_idx}: {len(group)} artifacts")
            processed_per_sample.append(_stack_group_to_2d(group))
        else:
            processed_per_sample.append(group[0])
            print(f"Sample {sample_idx}: single artifact (no stacking)")
        print(
            f"Sample {sample_idx} processing time: {_format_seconds(time.perf_counter() - group_start)}"
        )

    final_entry = _average_entries(processed_per_sample)
    final_sim_type = str(final_entry.simulation_config.sim_type)
    final_filename = final_processed_filename(final_sim_type)
    final_path = anchor.path.parent / final_filename

    if skip_if_exists and final_path.exists():
        print(f"⏭️  Final averaged artifact already exists: {final_path}")
        return final_path

    info_path = final_path.with_suffix(".pkl")
    extra_payload: dict[str, Any] = {}
    if final_entry.job_metadata:
        extra_payload.update(final_entry.job_metadata)
    extra_payload.update({"t_det": final_entry.t_det, "t_coh": final_entry.t_coh})

    save_info_file(
        info_path,
        final_entry.system,
        final_entry.simulation_config,
        bath=final_entry.bath,
        laser=final_entry.laser,
        extra_payload=extra_payload,
    )

    out_path = save_run_artifact(
        signal_arrays=[final_entry.signals[sig] for sig in final_entry.signals],
        metadata=final_entry.metadata,
        frequency_sample_cm=final_entry.frequency_sample_cm,
        data_dir=anchor.path.parent,
        filename=final_filename,
    )

    print(f"✅ Processed and saved final averaged artifact: {out_path}")
    print(
        f"Processed {len(entries)} files, averaged {len(processed_per_sample)} samples in "
        f"{_format_seconds(time.perf_counter() - start_all)}"
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process spectroscopy data: stack and average across inhomogeneity samples."
    )
    parser.add_argument("--abs_path", type=str, required=True, help="Path to any artifact (.npz)")
    parser.add_argument(
        "--skip_if_exists",
        action="store_true",
        help="Reuse existing final artifact if present",
    )
    args = parser.parse_args()

    processed_path = process_datas(Path(args.abs_path), skip_if_exists=args.skip_if_exists)
    plot_script = (PROJECT_ROOT / "scripts" / "local" / "plot_datas.py").resolve()
    print(f"Final processed artifact: {processed_path}")
    print("\n🎯 Plot with:")
    print(f'python "{plot_script}" --abs_path {processed_path}')


if __name__ == "__main__":
    main()
