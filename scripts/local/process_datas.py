"""Reduce batch-level partial artifacts into one final processed spectroscopy artifact."""

from __future__ import annotations

import os
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import argparse
import json
import time
from functools import partial
from typing import Any

import numpy as np

from qspectro2d.utils.data_io import (
    load_info_file,
    load_partial_reduction_artifact,
    save_info_file,
    save_run_artifact,
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


def _load_job_metadata(job_dir: Path) -> dict[str, Any]:
    metadata_path = job_dir / "job_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing job metadata: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _partial_paths(data_dir: Path, prefix: str) -> list[Path]:
    return sorted(data_dir.glob(f"{prefix}_batch_*.partial.npz"))


def _resolve_data_dir(job_dir: Path, metadata: dict[str, Any]) -> Path:
    if os.name == "nt" and str(metadata["data_dir"]).startswith("/"):
        fallback_dir = (job_dir / "data").resolve()
        if fallback_dir.exists():
            print(f"Using local data dir for POSIX metadata path: {fallback_dir}")
            return fallback_dir

    configured_dir = Path(metadata["data_dir"]).expanduser()
    if configured_dir.exists():
        return configured_dir.resolve()

    fallback_dir = (job_dir / "data").resolve()
    if fallback_dir.exists():
        print(f"Falling back to local data dir: {fallback_dir}")
        return fallback_dir

    return configured_dir.resolve()


def process_job_dir(job_dir: Path, *, skip_if_exists: bool = False) -> Path:
    start_all = time.perf_counter()
    job_dir = job_dir.expanduser().resolve()
    metadata = _load_job_metadata(job_dir)

    data_dir = _resolve_data_dir(job_dir, metadata)
    prefix = str(metadata["data_base_name"])
    final_filename = final_processed_filename(str(metadata["sim_type"]))
    final_path = data_dir / final_filename

    print(f"Starting strict reduction for job_dir: {job_dir}")
    print(f"Data directory: {data_dir}")

    if skip_if_exists and final_path.exists():
        print(f"⏭️  Final processed artifact already exists: {final_path}")
        return final_path

    partial_paths = _partial_paths(data_dir, prefix)
    print(f"Found {len(partial_paths)} partial artifact(s)")
    if not partial_paths:
        raise FileNotFoundError(
            f"No partial reduction artifacts found in {data_dir} for prefix '{prefix}'"
        )

    info_path = data_dir / f"{prefix}.pkl"
    info = load_info_file(info_path)
    sim_cfg = info["sim_config"]
    system = info["system"]
    bath = info.get("bath")
    laser = info.get("laser")

    expected_signal_types = list(metadata["signal_types"])
    expected_t_det = np.asarray(metadata["t_det"], dtype=float)
    expected_t_coh = np.asarray(metadata.get("t_coh", []), dtype=float)
    expected_n_inhom = int(metadata["n_inhom"])

    total_counts: np.ndarray | None = None
    total_frequency_sum: np.ndarray | None = None
    total_frequency_count = 0
    total_signal_sums: dict[str, np.ndarray] | None = None

    for partial_path in partial_paths:
        loaded = load_partial_reduction_artifact(partial_path)
        part_meta = dict(loaded["metadata"])
        part_signals = loaded["signal_sums"]
        part_counts = np.asarray(loaded["counts_per_t_coh"], dtype=np.int64)
        part_freq_sum = np.asarray(loaded["frequency_sample_sum_cm"], dtype=float)
        part_freq_count = int(loaded["frequency_sample_count"])

        if list(part_meta.get("signal_types", [])) != expected_signal_types:
            raise ValueError(
                f"Signal types mismatch in {partial_path.name}: "
                f"expected {expected_signal_types}, got {part_meta.get('signal_types')}"
            )

        if total_counts is None:
            total_counts = np.zeros_like(part_counts, dtype=np.int64)
        elif part_counts.shape != total_counts.shape:
            raise ValueError(
                f"counts_per_t_coh shape mismatch in {partial_path.name}: "
                f"expected {total_counts.shape}, got {part_counts.shape}"
            )

        if total_frequency_sum is None:
            total_frequency_sum = np.zeros_like(part_freq_sum, dtype=float)
        elif part_freq_sum.shape != total_frequency_sum.shape:
            raise ValueError(
                f"frequency_sample_sum_cm shape mismatch in {partial_path.name}: "
                f"expected {total_frequency_sum.shape}, got {part_freq_sum.shape}"
            )

        if total_signal_sums is None:
            total_signal_sums = {
                sig: np.zeros_like(np.asarray(part_signals[sig]), dtype=np.complex128)
                for sig in expected_signal_types
            }

        for sig in expected_signal_types:
            if sig not in part_signals:
                raise KeyError(f"Missing signal '{sig}' in {partial_path.name}")
            part_array = np.asarray(part_signals[sig])
            if part_array.shape != total_signal_sums[sig].shape:
                raise ValueError(
                    f"Signal shape mismatch for {sig!r} in {partial_path.name}: "
                    f"expected {total_signal_sums[sig].shape}, got {part_array.shape}"
                )
            total_signal_sums[sig] += part_array

        total_counts += part_counts
        total_frequency_sum += part_freq_sum
        total_frequency_count += part_freq_count

    assert (
        total_counts is not None
        and total_signal_sums is not None
        and total_frequency_sum is not None
    )

    if np.any(total_counts != expected_n_inhom):
        bad = np.where(total_counts != expected_n_inhom)[0].tolist()
        preview = bad[:10]
        raise RuntimeError(
            "Strict reduction refused to average incomplete data: "
            f"expected {expected_n_inhom} successful sample(s) per t_coh, "
            f"got counts={total_counts.tolist()} (bad indices preview={preview})"
        )

    if total_frequency_count <= 0:
        raise RuntimeError("No successful combinations contributed to the reduction")

    averaged_signals_2d = {
        sig: total_signal_sums[sig] / total_counts[:, None] for sig in expected_signal_types
    }
    average_frequency_sample = total_frequency_sum / float(total_frequency_count)

    final_sim_type = str(metadata["sim_type"]).strip().lower()
    if final_sim_type == "2d":
        final_t_coh = expected_t_coh
        final_signals = averaged_signals_2d
    else:
        final_t_coh = None
        final_signals = {sig: arr[0] for sig, arr in averaged_signals_2d.items()}

    final_metadata = {
        "signal_types": expected_signal_types,
        "sim_type": final_sim_type,
        "inhom_averaged": True,
        "averaged_count": expected_n_inhom,
    }

    final_info_path = final_path.with_suffix(".pkl")
    extra_payload = dict(metadata)
    extra_payload.update({"t_det": expected_t_det, "t_coh": final_t_coh})
    save_info_file(
        final_info_path,
        system,
        sim_cfg,
        bath=bath,
        laser=laser,
        extra_payload=extra_payload,
    )

    out_path = save_run_artifact(
        signal_arrays=[final_signals[sig] for sig in expected_signal_types],
        metadata=final_metadata,
        frequency_sample_cm=average_frequency_sample,
        data_dir=data_dir,
        filename=final_filename,
    )

    print(f"Final processed artifact written: {out_path}")
    print(f"Reduction completed in {_format_seconds(time.perf_counter() - start_all)}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reduce strict batch partial artifacts into one processed result."
    )
    parser.add_argument(
        "--job_dir",
        type=str,
        required=True,
        help="Path to the job directory, e.g. jobs/01_123456_monomer",
    )
    parser.add_argument(
        "--skip_if_exists",
        action="store_true",
        help="Reuse existing final artifact if present",
    )
    args = parser.parse_args()

    processed_path = process_job_dir(Path(args.job_dir), skip_if_exists=args.skip_if_exists)
    plot_script = (PROJECT_ROOT / "scripts" / "local" / "plot_datas.py").resolve()
    print(f"Final processed artifact: {processed_path}")
    print("\n🎯 Plot with:")
    print(f'python "{plot_script}" --abs_path {processed_path}')


if __name__ == "__main__":
    main()
