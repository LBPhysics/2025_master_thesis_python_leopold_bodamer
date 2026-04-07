"""Rewrite stored job metadata after moving job directories."""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import argparse

from common.job_metadata_relocation import relocate_job_dirs, summarize_relocation

print = partial(print, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite stored job metadata under a moved job directory or subtree.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Moved job directory or parent subtree to scan recursively",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview updates without rewriting any files",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    try:
        results = relocate_job_dirs(root, dry_run=args.dry_run)
    except (FileNotFoundError, NotADirectoryError) as exc:
        parser.exit(2, f"error: {exc}\n")

    summary = summarize_relocation(results)
    mode_label = "DRY RUN" if args.dry_run else "METADATA RELOCATION"

    print("=" * 80)
    print(mode_label)
    print(f"Root: {root}")
    print(f"Discovered {summary['jobs']} job director{'y' if summary['jobs'] == 1 else 'ies'}")

    for result in results:
        json_status = "created" if result.created_json else "updated"
        if args.dry_run:
            json_status = f"would {json_status}"

        print(f"\n{result.job_dir}")
        print(f"  job_metadata.json: {json_status}")
        print(f"  info .pkl files: {result.updated_pkl}")
        if result.slurm_files:
            print(f"  note: {len(result.slurm_files)} .slurm file(s) were left untouched")

    print("\nSummary")
    print(f"  updated_json={summary['updated_json']}")
    print(f"  created_json={summary['created_json']}")
    print(f"  updated_pkl={summary['updated_pkl']}")

    if summary["slurm_files"]:
        print("\nExisting .slurm files still contain old absolute paths.")
        print("Regenerate them with scripts/hpc/plot_dispatcher.py --job_dir <moved_job_dir> --no_submit")


if __name__ == "__main__":
    main()
