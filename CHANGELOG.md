# Changelog

## 0.3.0 — 2026-11-10

- Added checkpoint- and row-referenced availability perturbation audits.
- Added actual-perturbation evidence semantics; already-missing values no longer create a false exercise boundary.
- Added strict temporal input ordering and caller-input isolation.
- Added contract fingerprints plus calendar-based endpoint recomputation.
- Added event-aware holding-period endpoints.
- Fixed multiple disjoint test-block embargo handling and boundary overlap semantics.
- Added empty train/test evidence guards and stricter test-label validity checks.
- Added fresh-state refit determinism and clearer component-level findings.
- Hardened nullable/categorical output comparison and duplicate-label rejection.
- Added a verified ACTR-20 public-model experiment with mandatory leaking negative controls.
- Added isolated model workers, atomic run status, external-reference verification, and dependency-light reports.
- Added generated/adversarial tests and an 85% CI coverage floor.

## 0.2.0 — 2026-11-10

- Added `INCONCLUSIVE` and strict evidence semantics.
- Added exhaustive, uniform, and explicit checkpoint plans with coverage reporting.
- Added fresh-state refit auditing and optional scikit-learn cloning.
- Added event-aware decision, execution, exit, label availability, purge, safety-gap, and embargo logic.
- Fixed stage snapshots and stage-specific exception attribution.

## 0.1.0 — 2026-11-10

- Initial black-box prefix-invariance audit, stage localization, determinism checks, maturity reports, CLI, examples, tests, and CI.
