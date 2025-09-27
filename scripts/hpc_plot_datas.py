"""
Generate and (optionally) submit a SLURM job to plot data.

Flow (kept simple to match the new stacking script):
    1) Normalize the provided path to the 1D results directory.
    2) Invoke `stack_times.py --abs_path <1d_dir>` to build/update the 2D dataset.
    3) Derive the 2D path deterministically and submit a job to run
       `plot_datas.py --abs_path <2d_data.npz>`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from subprocess import run, CalledProcessError

from thesis_paths import SCRIPTS_DIR


def _derive_1d_dir(abs_path: str) -> Path:
    """Return the 1D directory given a path that may be a file or a directory.

    Also sanitizes accidental newlines/carriage-returns or stray quotes from copy/paste.
    """
    sanitized = abs_path.strip().strip('"').strip("'").replace("\r", "").replace("\n", "")
    p = Path(sanitized).expanduser().resolve()
    return p if p.is_dir() else p.parent


def _derive_2d_dir(from_1d_dir: Path) -> Path:
    """Map .../data/1d_spectroscopy/... -> .../data/2d_spectroscopy/..."""
    parts = list(from_1d_dir.parts)
    try:
        idx = parts.index("1d_spectroscopy")
    except ValueError as exc:
        raise ValueError("The provided path must include '1d_spectroscopy'") from exc
    parts[idx] = "2d_spectroscopy"
    return Path(*parts)


def ensure_2d_dataset(abs_path: str) -> Path:
    """Run stacking for the given 1D path and return the absolute 2D data file path.

    Since stacking now saves using save_simulation_data (unique filenames), we parse
    the stdout for a line like: "Saved 2D dataset: <abs_path>".
    As a fallback, we search the derived 2D directory for the newest "*_data.npz".
    """
    # Accept both a single 1D data file or a directory containing them.
    sanitized = abs_path.strip().strip('"').strip("'").replace("\r", "").replace("\n", "")
    orig_path = Path(sanitized).expanduser().resolve()
    one_d_dir = _derive_1d_dir(abs_path)

    # Always run stacking (kept simple; idempotent and quick compared to compute)
    # If a data file is provided, pass the file path to stack_times.py
    # (it currently assumes in_dir is a file and uses parent()).
    stack_arg = str(orig_path if orig_path.is_file() else one_d_dir)
    cmd = ["python", "stack_times.py", "--abs_path", stack_arg]
    proc = run(cmd, cwd=SCRIPTS_DIR, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"stack_times.py failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    # Try to parse the saved path from stdout
    m = re.search(r"Saved 2D dataset:\s*(.+)", proc.stdout)
    if m:
        saved_path = Path(m.group(1).strip()).expanduser().resolve()
        if saved_path.exists():
            return saved_path

    # Fallback: discover newest *_data.npz in the derived 2D directory
    base_for_2d = orig_path.parent if orig_path.is_file() else one_d_dir
    two_d_dir = _derive_2d_dir(base_for_2d)
    candidates = sorted(two_d_dir.glob("*_data.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    raise RuntimeError(
        "2D dataset not found after stacking; no explicit path in output and no *_data.npz in 2D dir."
    )


def create_plotting_script(
    abs_path: str,
    job_dir: Path,
) -> Path:
    """Create a SLURM script that runs plot_datas.py with the given abs_path."""
    plot_py = (SCRIPTS_DIR / "plot_datas.py").resolve()
    content = f"""#!/bin/bash
#SBATCH --job-name=plot_data
#SBATCH --chdir={job_dir}
#SBATCH --output=logs/plotting.out
#SBATCH --error=logs/plotting.err
#SBATCH --partition=GPGPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=200G
#SBATCH --time=0-01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leopold.bodamer@student.uni-tuebingen.de  # NOTE: CHANGE TO YOUR MAIL HERE

# Load conda (adjust to your cluster if needed)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/home/$USER/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/home/$USER/miniconda3/etc/profile.d/conda.sh"
fi
conda activate master_env || true

# Execute plot_datas.py from scripts (absolute path)
python {plot_py} --abs_path "{abs_path}"
"""
    path = job_dir / "plotting.slurm"
    # Write with Unix line endings so SLURM doesn't complain on Linux clusters
    try:
        path.write_text(content, newline="\n")
    except TypeError:
        path.write_text(content.replace("\r\n", "\n").replace("\r", "\n"))
    path.chmod(0o755)
    return path


def execute_slurm_script(job_dir: Path) -> None:
    """Submit the generated SLURM script."""
    slurm_script = job_dir / "plotting.slurm"
    try:
        run(["sbatch", str(slurm_script)], check=True)
        print(f"Submitted {slurm_script}")
    except (FileNotFoundError, CalledProcessError) as exc:
        print(f"Failed submitting {slurm_script}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and optionally submit a SLURM job to stack 1D data to 2D and plot it."
    )
    parser.add_argument(
        "--abs_path",
        type=str,
        required=True,
        help=(
            "Absolute path to the 1D directory OR a single 1D data file (e.g., t_coh_50.0_data.npz)."
        ),
    )
    parser.add_argument(
        "--no_submit",
        action="store_true",
        help="Only generate the job script, do not submit.",
    )
    args = parser.parse_args()

    print("🔄 Building 2D dataset (via stacking)...")
    try:
        two_d_file = ensure_2d_dataset(args.abs_path)
        print(f"✅ Dataset ready: {two_d_file}")
    except RuntimeError as e:
        print(f"❌ Stacking failed: {e}")
        return

    # Create a unique job directory under scripts/batch_jobs
    base_name = "plotting"
    job_root = SCRIPTS_DIR / "batch_jobs"
    job_dir = job_root / base_name
    suffix = 0
    while job_dir.exists():
        suffix += 1
        job_dir = job_root / f"{base_name}_{suffix}"

    logs_dir = job_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=False)

    # Create the script
    create_plotting_script(abs_path=str(two_d_file), job_dir=job_dir)

    print(f"Generated plotting script in: {job_dir}")

    # Optionally submit
    if not args.no_submit:
        execute_slurm_script(job_dir)
    else:
        print("Submission skipped (use without --no_submit to sbatch).")


if __name__ == "__main__":
    main()
