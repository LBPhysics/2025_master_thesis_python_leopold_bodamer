"""Run 1D retries for failed phase-cycling points and overwrite originals on success."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from qspectro2d.config.factory import load_simulation_config
from qspectro2d.core.simulation.time_axes import compute_t_det
from qspectro2d.spectroscopy import compute_emitted_field_components
from qspectro2d.utils.data_io import build_run_metadata, load_run_artifact, pad_or_crop_signals, save_run_artifact

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(SCRIPTS_DIR))

from common.retry_queue import ensure_retry_dir, load_retry_candidates


def _load_retry_batch(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    retries = payload.get("retries", payload)
    if not isinstance(retries, list):
        raise TypeError(f"Expected retry list, got {type(retries)!r}")
    return retries


def _load_cfg_dict(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise TypeError(f"Resolved config must be a dict, got {type(cfg)!r}")
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-run failed phase-cycling points as 1D retries and overwrite originals on success."
    )
    parser.add_argument("--retry_file", type=str, required=True)
    parser.add_argument("--batch_id", type=int, default=0)
    args = parser.parse_args()

    retry_file = Path(args.retry_file).resolve()
    retries = _load_retry_batch(retry_file)
    if not retries:
        print("No retries in batch; nothing to do.")
        return

    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    max_workers_override = None
    if slurm_cpus:
        try:
            max_workers_override = int(slurm_cpus)
        except ValueError:
            max_workers_override = None

    print("=" * 80)
    print("PHASE-CYCLING 1D RETRY BATCH")
    print(f"Retry file: {retry_file}")
    print(f"Loaded {len(retries)} retry candidate(s).")

    retry_dir = ensure_retry_dir(retry_file.parent)
    results_path = retry_dir / f"retry_results_batch_{args.batch_id:03d}.jsonl"

    start_all = time.time()
    overwritten = 0
    failed = 0
    skipped = 0

    # Shared pool for repeated phase-cycling solves in this retry batch.
    pool_workers = max_workers_override or 1
    with ProcessPoolExecutor(max_workers=pool_workers) as executor:
        for idx, entry in enumerate(retries, start=1):
            original_path = Path(entry["original_artifact_path"]).resolve()
            resolved_config_path = Path(entry["resolved_config_path"]).resolve()
            t_coh = float(entry["t_coh_value"])
            freq_vector = np.asarray(entry["freq_vector"], dtype=float)
            time_cut = float(entry.get("time_cut", float("inf")))

            print(
                f"\n--- retry {idx} / {len(retries)}: "
                f"t_idx={entry['t_index']}, t_coh={t_coh:.4f} fs, inhom_idx={entry['inhom_index']} ---"
            )

            if not original_path.exists():
                failed += 1
                message = f"Original artifact missing: {original_path}"
                print(f"    ⚠️ {message}", flush=True)
                with results_path.open("a", encoding="utf-8") as handle:
                    json.dump({**entry, "retry_status": "missing_original", "retry_error": message}, handle)
                    handle.write("\n")
                continue

            original_entry = load_run_artifact(original_path)
            target_length = int(next(iter(original_entry["signals"].values())).shape[0])

            cfg_dict = _load_cfg_dict(resolved_config_path)
            cfg_dict.setdefault("config", {})["sim_type"] = "1d"
            cfg_dict["config"]["t_coh"] = t_coh
            if max_workers_override is not None and max_workers_override > 0:
                cfg_dict["config"]["max_workers"] = max_workers_override

            cfg = load_simulation_config(cfg_dict)
            detection_window = np.asarray(compute_t_det(cfg), dtype=float)
            signal_types = list(cfg.signal_types)

            # print("    ▶ entering compute_emitted_field_components() for retry", flush=True)
            call_start = time.time()
            try:
                e_components, run_status, status_message = compute_emitted_field_components(
                    config_source=cfg_dict,
                    t_coh=t_coh,
                    freq_vector=freq_vector.tolist(),
                    time_cut=time_cut,
                    detection_window=detection_window,
                    executor=executor,
                )
                call_elapsed = time.time() - call_start
                print(
                    f"    ✔ retry compute_emitted_field_components() returned in {call_elapsed:.2f} s",
                    flush=True,
                )
            except Exception as exc:
                failed += 1
                message = f"{type(exc).__name__}: {exc}"
                print(f"    ⚠️ retry failed: {message}", flush=True)
                with results_path.open("a", encoding="utf-8") as handle:
                    json.dump({**entry, "retry_status": "failed_exception", "retry_error": message}, handle)
                    handle.write("\n")
                continue

            if run_status != "ok":
                failed += 1
                print(
                    f"    ⚠️ retry still incomplete ({run_status}); original file will not be overwritten",
                    flush=True,
                )
                with results_path.open("a", encoding="utf-8") as handle:
                    json.dump(
                        {
                            **entry,
                            "retry_status": run_status,
                            "retry_error": status_message,
                        },
                        handle,
                    )
                    handle.write("\n")
                continue

            padded_components = pad_or_crop_signals(e_components, target_length)

            metadata = build_run_metadata(
                signal_types=signal_types,
                sim_type="1d",
                sample_index=int(entry["inhom_index"]),
                t_coh_value=t_coh,
                run_status="ok",
                t_index=int(entry["t_index"]),
                global_index=int(original_entry["metadata"].get("global_index", entry["t_index"])),
                retried_from=str(entry.get("original_run_status", "unknown")),
                retry_source="1d",
                retry_original_error=entry.get("original_error"),
                retry_batch_id=int(args.batch_id),
            )

            archive_dir = original_path.parent / "failed_originals"
            archive_dir.mkdir(parents=True, exist_ok=True)
            archive_path = archive_dir / original_path.name
            if not archive_path.exists():
                shutil.copy2(original_path, archive_path)

            tmp_filename = f"{original_path.stem}.retry_tmp.npz"
            tmp_path = save_run_artifact(
                signal_arrays=padded_components,
                metadata=metadata,
                frequency_sample_cm=freq_vector,
                data_dir=original_path.parent,
                filename=tmp_filename,
            )
            os.replace(tmp_path, original_path)
            overwritten += 1
            print(f"    ✅ overwrote {original_path} with successful 1D retry", flush=True)

            with results_path.open("a", encoding="utf-8") as handle:
                json.dump({**entry, "retry_status": "overwritten_ok"}, handle)
                handle.write("\n")

    elapsed = time.time() - start_all
    print("=" * 80)
    print(
        f"Retry batch finished in {elapsed:.2f} s | overwritten={overwritten}, failed={failed}, skipped={skipped}"
    )
    print(f"Retry results log: {results_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
