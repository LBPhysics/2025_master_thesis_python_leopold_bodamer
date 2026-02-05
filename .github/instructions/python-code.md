# Python Code Guidelines

## Scope
Apply to all Python code, scripts, and notebooks in this workspace.

## Style & Structure
- Follow PEP 8 for formatting and naming.
- Prefer small, testable functions with clear responsibilities.
- Use type hints for public functions and complex data structures.
- Keep modules cohesive; avoid circular imports.

## Reliability & Performance
- Validate inputs and raise clear exceptions.
- Avoid premature optimization; use vectorized NumPy operations when appropriate.
- Log long-running steps; avoid print in library code.

## Tooling
- Prefer existing utilities in the workspace before adding new dependencies.
- If a new package is required, document it in environment.yml or pyproject.toml.
- Avoid modifying generated files and outputs.

## Notebooks
- Keep notebook cells deterministic and ordered.
- Move reusable code into modules under packages/ or scripts/.
- Clear temporary debugging cells before finalizing changes.

## Documentation
- Add concise docstrings to public functions/classes.
- Include usage examples for non-trivial workflows.
