## What this changes

<!-- And why. Link an issue if there is one. -->

## How it was tested

<!-- `pytest -q` alone is enough for python-lib/ changes. Changes to llm.py touch
     DSS's own contract, which the unit tests cannot cover — please say what you
     exercised in a real DSS instance, and with which model. -->

- [ ] `pytest -q` passes
- [ ] `ruff check .` passes
- [ ] Tested in a real DSS instance (required for `llm.py` changes)
- [ ] `CHANGELOG.md` updated under `## [Unreleased]`, if user-visible
