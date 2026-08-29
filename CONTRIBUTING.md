# Contributing to Shadow Hunter

Thanks for taking the time to contribute. This document covers how to get the project
running, the conventions the codebase relies on, and what a good pull request looks like.

By participating you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).

---

## Getting set up

```bash
git clone https://github.com/Jarvis-Mi/shadowhunter.git
cd shadowhunter
python -m venv .venv
```

Activate the environment — `.venv\Scripts\activate` on Windows, `source .venv/bin/activate`
on macOS and Linux — then:

```bash
pip install -e ".[dev]"
```

Add the desktop toolkits only if you are working on a front-end:

```bash
pip install -e ".[ui]"
```

Confirm the environment is sane:

```bash
python run.py doctor
```

---

## Running the checks

```bash
python -m pytest
```

```bash
ruff check .
```

Both must pass before a pull request is ready. The test suite downloads nothing and needs
no GPU; if a test of yours does, it does not belong in this suite.

---

## The two architectural rules

These are the constraints that keep five front-ends and one backend from drifting apart.
A pull request that breaks either one will be asked to change, however good the feature is.

### 1. No view imports `models`

Front-ends talk to the backend over HTTP through `DeckClient`
(`shadowhunter/services/client.py`). They never import from `shadowhunter.models`.

```python
# wrong — a view reaching into the domain layer
from shadowhunter.models.pipeline import analyse

# right — the view goes through the transport
result = self.client.analyze(scene_id, ...)
```

The payoff: add an endpoint once and all five interfaces can gain the feature; swap the
backend for a remote host and nothing in the views changes.

### 2. No view hard-codes a colour

Every colour, font, radius, spacing step and motion curve comes from
`shadowhunter/templates/tokens.json`, rendered per toolkit by `templates/theme.py`.

```python
# wrong
label.setStyleSheet("color: #FFB020;")

# right
label.setStyleSheet(f"color: {theme.color('solar')};")
```

The payoff: change `tokens.json` once and all five interfaces move together.

---

## Code style

- **Formatting and linting** — `ruff`, configured in `pyproject.toml`. Line length 100,
  target Python 3.10.
- **Type hints** — required on public functions. `from __future__ import annotations` at the
  top of every module.
- **Docstrings** — explain *why*, not *what*. The reader can see what the code does.
- **Import order** — on this codebase it is load-bearing. `cv2` and `numpy` must be imported
  before `torch`; see the docstring in `tests/conftest.py` for the DLL conflict this avoids.
  Entry points do this already — preserve it.

---

## Adding a feature

**A new metric or reward term** → `shadowhunter/models/rl/rewards.py`. Keep it computable
from the crop alone; that label-free property is the point of the method. Add a unit test
that pins its behaviour on a synthetic crop.

**A new API endpoint** → a router under `shadowhunter/views/api/routers/`, a pydantic
contract in `shadowhunter/models/schemas.py`, and a method on `DeckClient`. Then the
front-ends can adopt it at their own pace.

**A new design token** → `templates/tokens.json` first, then the per-toolkit emitters in
`templates/theme.py`. Never the other way round.

---

## Pull request checklist

- [ ] `python -m pytest` passes.
- [ ] `ruff check .` is clean.
- [ ] New behaviour has a test.
- [ ] No view imports `shadowhunter.models`.
- [ ] No hard-coded colours, fonts or spacing values.
- [ ] The PR description says *why* the change is needed, not only what it does.
- [ ] Public API or CLI changes are reflected in `README.md`.
- [ ] User-visible changes have a `CHANGELOG.md` entry under `Unreleased`.

---

## Commit messages

Conventional Commits, so the changelog can be assembled mechanically:

```
feat(rl): add compute-cost term to the composite reward
fix(vision): guard shadow_mask against single-channel input
docs(readme): document the sweep endpoint
test(pipeline): cover fusion when sigma is degenerate
refactor(api): extract checkpoint listing into its own router
```

---

## Reporting bugs

Open an issue with the **Bug report** template. The output of `python run.py doctor` is
almost always the first thing needed, so please include it.

For anything with a security dimension, follow [SECURITY.md](SECURITY.md) instead of opening
a public issue.
