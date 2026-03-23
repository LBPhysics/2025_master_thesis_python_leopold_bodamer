"""Run all combinations of (t_coh, inhomogeneity) locally without batching.

This version delegates all shared preparation to ``common.workflow`` so the
same config resolution, validation, axis creation, and frequency sampling are
used by both local and HPC workflows.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import argparse
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from qspectro2d.spectroscopy import compute_emitted_field_components
from qspectro2d.utils.data_io import (
    allocate_job_dir,
    build_run_metadata,
    ensure_job_layout,
    job_label_token,
    pad_or_crop_signals,
    save_info_file,
    save_run_artifact,
)

from common.workflow import (
    PROJECT_ROOT,
    RUNS_ROOT,
    build_job_metadata,
    prepare_workflow,
    write_json,
)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*c_ops, e_ops, args and options will be keyword only from qutip 5\.3.*",
    module=r"qutip\.solver\.brmesolve",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all spectroscopy combinations (t_coh × inhom index) locally",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to a YAML simulation config file",
    )
    parser.add_argument(
        "--sim_type",
        choices=["0d", "1d", "2d"],
        default=None,
        help="Simulation dimensionality override",
    )
    parser.add_argument(
        "--rng_seed",
        type=int,
        default=None,
        help="Optional NumPy random seed for reproducible sampling",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("LOCAL ALL-COMBINATIONS RUNNER")

    prepared = prepare_workflow(
        config_path=args.config,
        sim_type=args.sim_type,
        rng_seed=args.rng_seed,
        run_solver_check=True,
    )

    print(f"Config path: {prepared.config_path}")
    print("✅ Merged config validated once.")
    print("✅ Simulation object constructed from validated merged config.")
    print(f"✅ Solver validated. time_cut = {prepared.time_cut:.6g}")

    label_token = job_label_token(
        prepared.sim.simulation_config,
        prepared.sim.system,
        sim_type=prepared.sim_type,
    )
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    job_dir = allocate_job_dir(RUNS_ROOT, f"local_{label_token}_{timestamp}")
    job_paths = ensure_job_layout(job_dir, base_name="raw")
    data_base_path = job_paths.data_base_path

    print(f"Job workspace: {job_paths.job_dir}")

    config_copy_path = job_paths.job_dir / prepared.config_path.name
    with config_copy_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(prepared.merged_cfg, handle, sort_keys=False)
    print(f"✅ Resolved config written to {config_copy_path}")

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
    global_n_t = int(prepared.t_det_axis.size)

    from concurrent.futures import ProcessPoolExecutor

    t_start = time.time()
    saved_paths: list[str] = []

    max_workers = int(prepared.sim.simulation_config.max_workers)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for combo in prepared.combinations:
            freq_vector = prepared.samples[combo.inhom_index, :].astype(float)

            print(
                f"\n--- combo {combo.index + 1} / {len(prepared.combinations)}: "
                f"t_idx={combo.t_index}, t_coh={combo.t_coh:.4f} fs, "
                f"inhom_idx={combo.inhom_index} ---"
            )

            print("    ▶ entering compute_emitted_field_components()", flush=True)
            call_start = time.time()
            e_components = compute_emitted_field_components(
                config_source=prepared.merged_cfg,
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

            padded_components = pad_or_crop_signals(e_components, global_n_t)

            metadata_combo = build_run_metadata(
                signal_types=signal_types,
                sim_type="1d" if prepared.sim_type == "2d" else prepared.sim_type,
                sample_index=combo.inhom_index,
                t_coh_value=combo.t_coh,
                run_status="ok",
                t_index=int(combo.t_index),
                global_index=int(combo.index),
            )

            path = save_run_artifact(
                signal_arrays=padded_components,
                metadata=metadata_combo,
                frequency_sample_cm=freq_vector,
                data_dir=data_base_path.parent,
                filename=f"{data_base_path.name}_run_t{combo.t_index:03d}_s{combo.inhom_index:03d}.npz",
            )
            saved_paths.append(str(path))
            print(f"    ✅ saved {path}")

    elapsed = time.time() - t_start
    print("=" * 80)
    print(f"Completed {len(saved_paths)} combination(s) in {elapsed:.2f} s")

    if saved_paths:
        print("Latest artifact:")
        print(f"  {saved_paths[-1]}")
        print("\n🎯 Next step:")
        process_script = (PROJECT_ROOT / "scripts" / "local" / "process_datas.py").resolve()
        print(f"     python \"{process_script}\" --abs_path '{saved_paths[-1]}' --skip_if_exists")

    print("=" * 80)
    print("DONE")


if __name__ == "__main__":
    main()
