# Contributing

Tripwire is a personal project, built with the same rigor as a real production repo.

## Workflow

1. **Branch:** create a branch off `main` named `feature/<short-description>`, `fix/<short-description>`, or `docs/<short-description>`.
2. **Develop:** follow `docs/CODING_STANDARDS.md`. Run `pytest`, `ruff check`, and `mypy --strict` locally before opening a PR.
3. **Test:** any change to `src/features/` requires a corresponding test in `tests/feature_parity/` (see `docs/TESTING_STRATEGY.md` for why this is non-negotiable).
4. **Commit messages:** Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`).
5. **Dependencies:** always use `uv add <package>` (or `uv add --dev <package>` for dev-only tools) and `uv remove <package>` to change `pyproject.toml`. Never hand-edit dependency versions — let `uv` resolve and pin the latest compatible version.
6. **Open a PR** against `main` with:
   - A summary of what changed and why.
   - A link to the relevant PRD/architecture section if this implements a milestone.
   - Before/after benchmark numbers if the change touches the scoring path's latency.
   - Test results, especially for anything touching feature parity, leakage checks, or the decision engine.
7. **CI must pass** (lint, type-check, unit, feature-parity, leakage tests) before merge.

## Code Review Standards

Reviewers should use the checklist in `docs/CODING_STANDARDS.md` §8. In particular, be skeptical of:
- New features implemented only in one of the online/offline paths.
- Any change to label-joining logic without an explicit discussion of potential leakage.
- Threshold or cost-assumption changes that aren't clearly justified.

## Reporting Issues

Use GitHub Issues. Please include:
- What you expected vs. what happened.
- Whether it's reproducible with the public datasets used in this repo (IEEE-CIS / PaySim).
- Relevant logs (structured logs should make this straightforward — see `docs/CODING_STANDARDS.md` §6).

## Questions

Open a discussion issue or reach out directly — contact info in the main [README](README.md).