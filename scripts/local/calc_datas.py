"""Run all combinations locally and reduce them through the same strict batch workflow used on HPC."""

from __future__ import annotations

import sys
import os
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import argparse
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import numpy as np
import yaml

from qspectro2d.spectroscopy import compute_emitted_field_components
from qspectro2d.utils.data_io import (
    allocate_job_dir,
    ensure_job_layout,
    save_info_file,
    save_partial_reduction_artifact,
)
from qspectro2d.utils.phase_pool import (
    PHASE_POOL_MAX_COMBOS_ENV,
    create_phase_pool_executor,
    phase_pool_combo_limit,
    resolve_phase_pool_worker_count,
)

from common.workflow import (
    PROJECT_ROOT,
    RUNS_ROOT,
    build_job_metadata,
    build_job_dir_label,
    prepare_workflow,
    resolve_allocated_job_unique_id,
    write_json,
)
from local.process_datas import process_job_dir

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*c_ops, e_ops, args and options will be keyword only from qutip 5\.3.*",
    module=r"qutip\.solver\.brmesolve",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all spectroscopy combinations locally with the strict batch-reduction workflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--sim_type", choices=["0d", "1d", "2d"], default=None)
    parser.add_argument("--rng_seed", type=int, default=None)
    parser.add_argument(
        "--phase_pool_max_combos",
        type=int,
        default=None,
        help="Optional pool recycling interval override to bound long-lived worker memory",
    )
    args = parser.parse_args()

    if args.phase_pool_max_combos is not None:
        if args.phase_pool_max_combos <= 0:
            raise ValueError("--phase_pool_max_combos must be positive")
        os.environ[PHASE_POOL_MAX_COMBOS_ENV] = str(int(args.phase_pool_max_combos))

    print("=" * 80)
    print("LOCAL STRICT RUNNER")

    prepared = prepare_workflow(
        config_path=args.config,
        sim_type=args.sim_type,
        rng_seed=args.rng_seed,
        max_workers=None,
        run_solver_check=True,
    )

    print(f"Config path: {prepared.config_path}")
    print("✅ Merged config validated once.")
    print("✅ Simulation object constructed from validated merged config.")
    print(f"✅ Solver validated. time_cut = {prepared.time_cut:.6g}")

    timestamp = datetime.utcnow().strftime("%d_%H%M%S")
    base_label = build_job_dir_label(prepared.config_path, timestamp)
    job_dir = allocate_job_dir(RUNS_ROOT, base_label)
    job_unique_id = resolve_allocated_job_unique_id(
        job_dir,
        base_label=base_label,
        requested_unique_handle=timestamp,
    )
    job_paths = ensure_job_layout(job_dir, base_name="raw")
    data_base_path = job_paths.data_base_path

    config_copy_path = job_paths.job_dir / prepared.config_path.name
    with config_copy_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(prepared.merged_cfg, handle, sort_keys=False)

    job_metadata = build_job_metadata(
        prepared,
        job_dir=job_paths.job_dir,
        data_dir=job_paths.data_dir,
        figures_dir=job_paths.figures_dir,
        data_base_name=job_paths.base_name,
        data_base_path=data_base_path,
        config_path=config_copy_path,
        time_cut=prepared.time_cut,
    )
    job_metadata["n_batches"] = 1
    job_metadata["job_unique_id"] = job_unique_id
    write_json(job_paths.job_dir / "job_metadata.json", job_metadata)

    info_path = data_base_path.parent / f"{data_base_path.name}.pkl"
    save_info_file(
        info_path,
        prepared.sim.system,
        prepared.sim.simulation_config,
        bath=getattr(prepared.sim, "bath", None),
        laser=getattr(prepared.sim, "laser", None),
        extra_payload=job_metadata,
    )

    samples_target = data_base_path.parent / f"{data_base_path.name}_samples.npy"
    np.save(samples_target, prepared.samples.astype(float))

    combos_target = data_base_path.parent / f"{data_base_path.name}_combos.json"
    write_json(combos_target, {"combos": [combo.to_dict() for combo in prepared.combinations]})

    print(f"Artifacts will be saved to {data_base_path.parent}")
    print(
        f"Prepared {len(prepared.combinations)} combination(s) → "
        f"|t_coh|={prepared.t_coh_values.size}, "
        f"n_inhom={int(prepared.sim.simulation_config.n_inhomogen)}"
    )

    signal_types = list(prepared.sim.simulation_config.signal_types)
    detection_times_length = int(prepared.t_det_axis.size)
    n_t_coh = int(prepared.t_coh_values.size)
    signal_sums = {
        sig: np.zeros((n_t_coh, detection_times_length), dtype=np.complex128)
        for sig in signal_types
    }
    counts_per_t_coh = np.zeros(n_t_coh, dtype=np.int64)
    frequency_sample_sum_cm = np.zeros(prepared.samples.shape[1], dtype=float)
    frequency_sample_count = 0

    t_start = time.time()
    phase_jobs = int(prepared.sim.simulation_config.n_phases) ** 2 + 2 * int(
        prepared.sim.simulation_config.n_phases
    ) + 1
    max_workers = resolve_phase_pool_worker_count(
        configured_workers=int(prepared.sim.simulation_config.max_workers),
        phase_jobs=phase_jobs,
    )
    pool_combo_limit = phase_pool_combo_limit()
    print(
        f"Using one shared phase-cycling process pool: {max_workers} worker(s) for {phase_jobs} phase jobs",
        flush=True,
    )
    print(
        f"Recycling the phase-cycling worker pool every {pool_combo_limit} combination(s)",
        flush=True,
    )

    executor: ProcessPoolExecutor | None = None
    combos_since_pool_start = 0
    try:
        for combo in prepared.combinations:
            if executor is None or combos_since_pool_start >= pool_combo_limit:
                if executor is not None:
                    print(
                        f"Recycling phase-cycling pool after {combos_since_pool_start} combination(s)",
                        flush=True,
                    )
                    executor.shutdown(wait=True)
                executor = create_phase_pool_executor(max_workers=max_workers)
                combos_since_pool_start = 0
                print(
                    f"Started phase-cycling pool for combo {combo.index + 1} / {len(prepared.combinations)}",
                    flush=True,
                )

            freq_vector = prepared.samples[combo.inhom_index, :].astype(float)

            print(
                f"\n--- combo {combo.index + 1} / {len(prepared.combinations)}: "
                f"t_idx={combo.t_index}, t_coh={combo.t_coh:.4f} fs, "
                f"inhom_idx={combo.inhom_index} ---",
                flush=True,
            )

            call_start = time.time()
            e_components, run_status, status_message = compute_emitted_field_components(
                config_source=str(config_copy_path),
                t_coh=combo.t_coh,
                freq_vector=freq_vector.tolist(),
                time_cut=prepared.time_cut,
                detection_window=prepared.t_det_axis,
                executor=executor,
            )
            call_elapsed = time.time() - call_start
            print(
                f"    ✔ compute_emitted_field_components() returned in {call_elapsed:.2f} s",
                flush=True,
            )

            if run_status != "ok":
                raise RuntimeError(
                    f"Strict local execution refused incomplete combo output: "
                    f"run_status={run_status}, message={status_message!r}, t_index={combo.t_index}, "
                    f"inhom_index={combo.inhom_index}"
                )

            if len(e_components) != len(signal_types):
                raise ValueError(
                    f"Signal count mismatch: expected {len(signal_types)}, got {len(e_components)}"
                )

            for sig, arr in zip(signal_types, e_components):
                arr = np.asarray(arr)
                if arr.shape != (detection_times_length,):
                    raise ValueError(
                        f"Signal {sig!r} returned shape {arr.shape}; expected {(detection_times_length,)}"
                    )
                signal_sums[sig][combo.t_index] += arr

            counts_per_t_coh[combo.t_index] += 1
            frequency_sample_sum_cm += freq_vector
            frequency_sample_count += 1
            combos_since_pool_start += 1
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    partial_path = save_partial_reduction_artifact(
        signal_sums=signal_sums,
        counts_per_t_coh=counts_per_t_coh,
        frequency_sample_sum_cm=frequency_sample_sum_cm,
        frequency_sample_count=frequency_sample_count,
        metadata={
            "signal_types": signal_types,
            "sim_type": prepared.sim_type,
            "batch_id": 0,
            "n_batches": 1,
            "n_t_coh": n_t_coh,
            "n_t_det": detection_times_length,
        },
        data_dir=data_base_path.parent,
        filename=f"{data_base_path.name}_batch_000.partial.npz",
    )

    elapsed = time.time() - t_start
    print("=" * 80)
    print(f"Completed {len(prepared.combinations)} combination(s) in {elapsed:.2f} s")
    print(f"Partial artifact: {partial_path}")

    processed_path = process_job_dir(job_paths.job_dir, skip_if_exists=False)
    print(f"Final processed artifact: {processed_path}")
    print("\n🎯 Next step:")
    plot_script = (PROJECT_ROOT / "scripts" / "local" / "plot_datas.py").resolve()
    print(f'     python "{plot_script}" --abs_path "{processed_path}"')


if __name__ == "__main__":
    main()
