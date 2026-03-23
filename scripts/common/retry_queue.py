"""Helpers for collecting and dispatching 1D retries for failed phase-cycling points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def ensure_retry_dir(job_dir: Path | str) -> Path:
    retry_dir = Path(job_dir) / "retries"
    retry_dir.mkdir(parents=True, exist_ok=True)
    return retry_dir


def append_retry_candidate(path: Path | str, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")


def load_retry_candidates(paths: Iterable[Path | str]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for path in paths:
        source = Path(path)
        if not source.exists():
            continue
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise TypeError(f"Retry payload must be a dict, got {type(payload)!r}")
                items.append(payload)
    return items


def dedupe_retry_candidates(candidates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for entry in candidates:
        key = str(entry["original_artifact_path"])
        if key not in deduped:
            deduped[key] = entry
            continue

        incumbent = deduped[key]
        # Prefer the one with a longer status message / more context.
        incumbent_msg = str(incumbent.get("original_error") or "")
        challenger_msg = str(entry.get("original_error") or "")
        if len(challenger_msg) > len(incumbent_msg):
            deduped[key] = entry
    return list(deduped.values())
