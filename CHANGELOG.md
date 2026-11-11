# Changelog

## 0.3.1 — 2026-11-11

- Preserved missing source positions during calendar shifts and rejected calendars containing missing sessions.
- Added complex-valued output comparison without discarding imaginary components.
- Rejected estimator factories that reuse object identities across audit checkpoints.
- Distinguished the first non-passing pipeline stage from the actual exception stage.
- Required scalar availability-mask values and marked unexercised perturbations `INCONCLUSIVE`.
- Downgraded maturity checks without an explicit `as_of` from `PASS` to `WARN`.
- Added hash-verified aggregate ACTR-20 reference evidence without publishing private data or model weights.
- Fixed Ruff violations that prevented the v0.3.0 matrix CI from reaching the test stage.
- Added adversarial regressions for all of the above and retained the 85% coverage gate.
- Added a deterministic randomized purge/embargo oracle so boundary validation still runs when Hypothesis is unavailable.
- Bound every isolated ACTR worker artifact to a common benchmark/model identity and rejected stale or mixed evidence during aggregation and report assembly.
- Added atomic worker and aggregate JSON writes so interrupted experiments cannot leave apparently valid partial evidence.
- Anchored the ACTR package manifest to a fixed SHA-256 instead of trusting a self-consistent but replaceable manifest.
- Separated per-stage and whole-experiment timeout ceilings so the documented one-command run can complete under realistic model-fit latency.
- Made core Markdown/JSON evidence-card writes atomic.

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
- Preserved invalid calendar positions during session shifts and rejected missing calendar sessions.
- Preserved complex-valued prediction differences instead of discarding imaginary components.
- Rejected estimator object reuse across checkpoints and clarified the limit of custom-factory introspection.
- Downgraded maturity checks without an explicit `as_of` timestamp to `WARN`.
- Distinguished the first non-passing pipeline stage from the first actual error stage.
- Added committed, hash-verified aggregate ACTR evidence; no benchmark array or private model artifact is vendored.

## 0.2.0 — 2026-11-10

- Added `INCONCLUSIVE` and strict evidence semantics.
- Added exhaustive, uniform, and explicit checkpoint plans with coverage reporting.
- Added fresh-state refit auditing and optional scikit-learn cloning.
- Added event-aware decision, execution, exit, label availability, purge, safety-gap, and embargo logic.
- Fixed stage snapshots and stage-specific exception attribution.

## 0.1.0 — 2026-11-10

- Initial black-box prefix-invariance audit, stage localization, determinism checks, maturity reports, CLI, examples, tests, and CI.
