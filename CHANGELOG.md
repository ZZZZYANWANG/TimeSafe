# Changelog

## 0.2.0 — 2026-11-10

- Added `INCONCLUSIVE` and strict evidence semantics.
- Added exhaustive, uniform, and explicit checkpoint plans with coverage reporting.
- Excluded the final timestamp from prefix evidence.
- Separated value changes from structural output differences.
- Added fresh-state refit auditing and optional scikit-learn cloning.
- Added event-aware decision, execution, exit, and label-availability validation.
- Applied safety gaps in generated walk-forward masks.
- Added arbitrary split purge plus post-test embargo.
- Rejected silently ignored embargo arguments in past-only walk-forward mode.
- Fixed stage snapshots, in-place mutation localization, and stage-specific exceptions.
- Added adversarial regression tests for every issue found in the v0.1 audit.

## 0.1.0 — 2026-11-10

- Initial black-box prefix-invariance audit.
- Stage localization, determinism audit, label maturity, reports, CLI, examples, tests, and CI.
