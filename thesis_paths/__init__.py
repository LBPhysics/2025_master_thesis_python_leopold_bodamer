"""Project-wide convenience exports (paths, helpers).

Provides canonical directories:
    from thesis_paths import DATA_DIR, SCRIPTS_DIR, ensure_project_directories

Environment override:
    Set THESIS_PROJECT_ROOT to force a different root (used in tests / notebooks).
"""

from __future__ import annotations
from pathlib import Path
import os

_DEF_FILE = Path(__file__).resolve()

_env_override = os.getenv("THESIS_PROJECT_ROOT")
if _env_override:
    PROJECT_ROOT = Path(_env_override).expanduser().resolve()
else:
    for _parent in _DEF_FILE.parents:
        if (_parent / ".git").is_dir():
            PROJECT_ROOT = _parent
            break
    else:
        PROJECT_ROOT = _DEF_FILE.parents[2]

DATA_DIR = (PROJECT_ROOT / "data").resolve()
FIGURES_DIR = (PROJECT_ROOT / "figures").resolve()
PYTHON_CODE_DIR = PROJECT_ROOT / "code" / "python"
SCRIPTS_DIR = PYTHON_CODE_DIR / "scripts"
NOTEBOOKS_DIR = PYTHON_CODE_DIR / "notebooks"
LATEX_DIR = PROJECT_ROOT / "latex"
FIGURES_PYTHON_DIR = FIGURES_DIR / "figures_from_python"
FIGURES_TESTS_DIR = FIGURES_PYTHON_DIR / "tests"
SIM_CONFIGS_DIR = SCRIPTS_DIR / "simulation_configs"


def ensure_project_directories(create_tests: bool = False) -> None:
    """Create standard output directories if missing."""
    targets = [DATA_DIR, FIGURES_DIR, FIGURES_PYTHON_DIR]
    if create_tests:
        targets.append(FIGURES_TESTS_DIR)
    for p in targets:
        p.mkdir(parents=True, exist_ok=True)


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "FIGURES_DIR",
    "FIGURES_PYTHON_DIR",
    "FIGURES_TESTS_DIR",
    "PYTHON_CODE_DIR",
    "SCRIPTS_DIR",
    "SIM_CONFIGS_DIR",
    "NOTEBOOKS_DIR",
    "LATEX_DIR",
    "ensure_project_directories",
]
