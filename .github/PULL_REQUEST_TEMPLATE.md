# Summary

<!-- What does this change, and why is it needed? The "why" is the part a reviewer
     cannot reconstruct from the diff. -->

Closes #

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Refactor (no behaviour change)
- [ ] Documentation
- [ ] Build, CI or tooling

## How this was verified

<!-- Commands run, scenes used, numbers observed. "Tests pass" on its own is not
     verification of a modelling change. -->

```
python -m pytest
ruff check .
```

## Checklist

- [ ] `python -m pytest` passes
- [ ] `ruff check .` is clean
- [ ] New behaviour has a test
- [ ] No view imports `shadowhunter.models` — front-ends go through `DeckClient`
- [ ] No hard-coded colours, fonts or spacing — values come from `templates/tokens.json`
- [ ] Public API or CLI changes are reflected in `README.md`
- [ ] User-visible changes have a `CHANGELOG.md` entry under `Unreleased`

## For changes to the reward or the environment

- [ ] The Stage-1 reward remains computable without ground-truth heights
- [ ] The learned-vs-greedy comparison was re-run, and the numbers are below

<!-- Delete this section if it does not apply. -->

## Screenshots

<!-- Required for front-end changes. Include light and dark if both are affected. -->
