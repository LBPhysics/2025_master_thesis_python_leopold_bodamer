"""Execute one HPC batch and write exactly one strict partial-reduction artifact."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from qspectro2d.spectroscopy import compute_emitted_field_components
from qspectro2d.utils.data_io import save_partial_reduction_artifact
from qspectro2d.core.simulation.time_axes import compute_t_det
from qspectro2d.config.factory import load_simulation_config
from qspectro2d.utils.phase_pool import (
    create_phase_pool_executor,
    phase_pool_combo_limit,
    resolve_phase_pool_worker_count,
)

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(SCRIPTS_DIR))

def _load_combinations(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    combos = payload["combos"] if isinstance(payload, dict) and "combos" in payload else payload
    if not isinstance(combos, list):
        raise TypeError(f"Expected list of combinations, got {type(combos)!r}")
    return combos


def _iter_combos(subset: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for entry in subset:
        if not isinstance(entry, dict):
            raise TypeError("Each combination must be a dictionary")
        yield entry


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one strict spectroscopy batch and save one partial reduction artifact",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--combos_file", type=str, required=True)
    parser.add_argument("--samples_file", type=str, required=True)
    parser.add_argument("--time_cut", type=float, required=True)
    parser.add_argument("--sim_type", choices=["0d", "1d", "2d"], required=True)
    parser.add_argument("--batch_id", type=int, default=0)
    parser.add_argument("--n_batches", type=int, default=1)
    args = parser.parse_args()

    combos_path = Path(args.combos_file).resolve()
    samples_path = Path(args.samples_file).resolve()
    job_dir = combos_path.parent
    job_metadata_path = job_dir / "job_metadata.json"
    if not job_metadata_path.exists():
        raise FileNotFoundError(f"Missing job metadata: {job_metadata_path}")
    with job_metadata_path.open("r", encoding="utf-8") as handle:
        job_metadata = json.load(handle)

    merged_cfg = job_metadata["merged_config"]
    config_source: str | dict[str, Any] = str(job_metadata.get("config_path", ""))
    if not config_source:
        config_source = merged_cfg

    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None:
        cpus_per_task = int(slurm_cpus)
        if cpus_per_task > 0:
            merged_cfg.setdefault("config", {})["max_workers"] = cpus_per_task
            print(f"Using max_workers={cpus_per_task} from SLURM_CPUS_PER_TASK", flush=True)

    data_dir = Path(job_metadata["data_dir"]).resolve()
    prefix = str(job_metadata["data_base_name"])
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STRICT BATCH RUNNER")
    print(f"Combos file: {combos_path}")
    print(f"Samples file: {samples_path}")
    print(f"Output directory: {data_dir}")

    combinations = _load_combinations(combos_path)
    if not combinations:
        raise ValueError("No combinations provided")

    samples = np.load(samples_path)
    if samples.ndim != 2:
        raise ValueError(f"Expected samples array with shape (n_inhom, n_atoms); got {samples.shape}")
    n_inhom, n_atoms = samples.shape

    samples_target = data_dir / f"{prefix}_samples.npy"
    if not samples_target.exists():
        shutil.copy2(samples_path, samples_target)

    cfg = load_simulation_config(merged_cfg)
    signal_types = list(job_metadata["signal_types"])
    global_t_det = np.asarray(job_metadata.get("t_det", []), dtype=float)
    if global_t_det.size == 0:
        global_t_det = compute_t_det(cfg)
    if global_t_det.size == 0 and args.sim_type != "0d":
        raise RuntimeError(
            "Invariant violation: empty global detection axis for non-0d batch run "
            f"(sim_type={args.sim_type}, t_det={float(cfg.t_det):.6g}, dt={float(cfg.dt):.6g})"
        )

    n_t_coh = int(job_metadata["n_t_coh"])
    global_n_t = int(global_t_det.size)
    if n_t_coh <= 0:
        raise ValueError(f"Invalid n_t_coh={n_t_coh}")

    phase_jobs = int(cfg.n_phases) ** 2 + 2 * int(cfg.n_phases) + 1
    pool_workers = resolve_phase_pool_worker_count(
        configured_workers=int(cfg.max_workers),
        phase_jobs=phase_jobs,
    )
    pool_combo_limit = phase_pool_combo_limit()
    print(
        f"Using one shared phase-cycling process pool: {pool_workers} worker(s) for {phase_jobs} phase jobs",
        flush=True,
    )
    print(
        f"Recycling the phase-cycling worker pool every {pool_combo_limit} combination(s)",
        flush=True,
    )

    signal_sums = {
        sig: np.zeros((n_t_coh, global_n_t), dtype=np.complex128)
        for sig in signal_types
    }
    counts_per_t_coh = np.zeros(n_t_coh, dtype=np.int64)
    frequency_sample_sum_cm = np.zeros(n_atoms, dtype=float)
    frequency_sample_count = 0

    total_global_combinations = int(job_metadata.get("n_t_coh", 0)) * int(job_metadata.get("n_inhom", 0))
    if total_global_combinations <= 0:
        total_global_combinations = len(combinations)

    t_start = time.time()
    executor: ProcessPoolExecutor | None = None
    combos_since_pool_start = 0
    try:
        for combo_idx, combo in enumerate(_iter_combos(combinations), start=1):
            if executor is None or combos_since_pool_start >= pool_combo_limit:
                if executor is not None:
                    print(
                        f"Recycling phase-cycling pool after {combos_since_pool_start} combination(s)",
                        flush=True,
                    )
                    executor.shutdown(wait=True)
                executor = create_phase_pool_executor(max_workers=pool_workers)
                combos_since_pool_start = 0
                print(
                    f"Started phase-cycling pool for combo {combo_idx} / {len(combinations)}",
                    flush=True,
                )

            t_idx = int(combo["t_index"])
            inhom_idx = int(combo["inhom_index"])
            global_idx = int(combo.get("index", combo_idx - 1))
            t_coh_val = float(combo["t_coh_value"])

            if inhom_idx < 0 or inhom_idx >= n_inhom:
                raise IndexError(
                    f"inhom_index {inhom_idx} out of range for {n_inhom} inhomogeneous samples"
                )
            if t_idx < 0 or t_idx >= n_t_coh:
                raise IndexError(f"t_index {t_idx} out of range for n_t_coh={n_t_coh}")

            freq_vector = samples[inhom_idx, :].astype(float)

            print(
                f"\n--- combo {combo_idx} / {len(combinations)} "
                f"(global {global_idx + 1} / {total_global_combinations}): "
                f"t_idx={t_idx}, t_coh={t_coh_val:.4f} fs, inhom_idx={inhom_idx} ---",
                flush=True,
            )

            call_start = time.time()
            e_components, run_status, status_message = compute_emitted_field_components(
                config_source=config_source,
                t_coh=t_coh_val,
                freq_vector=freq_vector.tolist(),
                time_cut=args.time_cut,
                detection_window=global_t_det,
                executor=executor,
            )
            call_elapsed = time.time() - call_start
            print(f"    ✔ compute_emitted_field_components() returned in {call_elapsed:.2f} s", flush=True)

            if run_status != "ok":
                raise RuntimeError(
                    f"Strict batch execution refused incomplete combo output: "
                    f"run_status={run_status}, message={status_message!r}, t_index={t_idx}, inhom_index={inhom_idx}"
                )

            if len(e_components) != len(signal_types):
                raise ValueError(
                    f"Signal count mismatch: expected {len(signal_types)}, got {len(e_components)}"
                )

            for sig, arr in zip(signal_types, e_components):
                arr = np.asarray(arr)
                if arr.shape != (global_n_t,):
                    raise ValueError(
                        f"Signal {sig!r} returned shape {arr.shape}; expected {(global_n_t,)}"
                    )
                signal_sums[sig][t_idx] += arr

            counts_per_t_coh[t_idx] += 1
            frequency_sample_sum_cm += freq_vector
            frequency_sample_count += 1
            combos_since_pool_start += 1
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    partial_metadata = {
        "signal_types": signal_types,
        "sim_type": str(args.sim_type),
        "batch_id": int(args.batch_id),
        "n_batches": int(args.n_batches),
        "n_t_coh": int(n_t_coh),
        "n_t_det": int(global_n_t),
    }
    partial_filename = f"{prefix}_batch_{int(args.batch_id):03d}.partial.npz"
    partial_path = save_partial_reduction_artifact(
        signal_sums=signal_sums,
        counts_per_t_coh=counts_per_t_coh,
        frequency_sample_sum_cm=frequency_sample_sum_cm,
        frequency_sample_count=frequency_sample_count,
        metadata=partial_metadata,
        data_dir=data_dir,
        filename=partial_filename,
    )

    elapsed = time.time() - t_start
    print("=" * 80)
    print(
        f"Completed {len(combinations)} combination(s) in {elapsed:.2f} s | "
        f"batch_id={args.batch_id}"
    )
    print(f"Partial artifact: {partial_path}")
    print(f"counts_per_t_coh={counts_per_t_coh.tolist()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
