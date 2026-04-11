"""Shared helpers for configuring the phase-cycling worker pool.

Pool width is provided by the caller:

* local strict runs use the workflow's configured ``max_workers``
* HPC runs use ``--cpus_per_task``

The only runtime phase-pool knob retained here is:

``QSPECTRO_PHASE_POOL_MAX_COMBOS``
    Rebuild the shared pool after this many outer combinations.
    This is the main safeguard against long-lived worker RSS growth.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

PHASE_POOL_MAX_COMBOS_ENV = "QSPECTRO_PHASE_POOL_MAX_COMBOS"

DEFAULT_PHASE_POOL_COMBO_LIMIT = 250


def _optional_positive_int_from_env(env_var: str) -> int | None:
    raw_value = os.environ.get(env_var, "").strip()
    if not raw_value:
        return None

    parsed_value = int(raw_value)
    if parsed_value <= 0:
        raise ValueError(f"{env_var} must be positive, got {parsed_value}")
    return parsed_value
def phase_pool_combo_limit(*, default: int = DEFAULT_PHASE_POOL_COMBO_LIMIT) -> int:
    raw_limit = _optional_positive_int_from_env(PHASE_POOL_MAX_COMBOS_ENV)
    if raw_limit is None:
        if default <= 0:
            raise ValueError(f"default combo limit must be positive, got {default}")
        return int(default)
    return raw_limit


def resolve_phase_pool_worker_count(*, configured_workers: int, phase_jobs: int) -> int:
    if configured_workers <= 0:
        raise ValueError(f"configured_workers must be positive, got {configured_workers}")
    if phase_jobs <= 0:
        raise ValueError(f"phase_jobs must be positive, got {phase_jobs}")

    return max(1, min(int(configured_workers), int(phase_jobs)))


def create_phase_pool_executor(*, max_workers: int) -> ProcessPoolExecutor:
    if max_workers <= 0:
        raise ValueError(f"max_workers must be positive, got {max_workers}")
    return ProcessPoolExecutor(max_workers=max_workers)


__all__ = [
    "DEFAULT_PHASE_POOL_COMBO_LIMIT",
    "PHASE_POOL_MAX_COMBOS_ENV",
    "create_phase_pool_executor",
    "phase_pool_combo_limit",
    "resolve_phase_pool_worker_count",
]
