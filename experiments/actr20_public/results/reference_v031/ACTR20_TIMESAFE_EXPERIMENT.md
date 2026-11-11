# TimeSafe × ACTR-20 public model experiment

This experiment combines the external ACTR privacy-hardened public benchmark with TimeSafe temporal-integrity audits. The anonymous benchmark preserves the public modeling task but is block-shuffled; it is evidence of structural reproducibility, not a substitute for the original chronological market study.

## External reference verification

- Reference-metric status: `PASS`
- External manifest status: `PASS` (76 files verified)
- Benchmark data SHA-256: `dcc663779842c5701440f16015ceb36fe0555bc125f33f8b8f8e53ca9a02e128`
- External manifest SHA-256: `4e482c68c8578983712752c5d3e35afd78c7dc31a03820927edf76e11c559ba7`
- Rows / sessions: `109499` / `1810`
- TimeSafe / experiment contract: `0.3.1` / `actr20-timesafe-audit-v1`
- Experiment run / implementation identity: `6c0563c0ceaa4eff903e47b9dd84229b` / `e729c25d1aff3d56b2d8169db1ce8064a8aa2f58542a0eaa5e04c9f02831391a`
- Audit subset identity: `088889b8071756d3559dc2172e525644f0be09d4b3fb26a697070bddcb805338` (24 assets, 36399 rows)
- Maximum absolute RankIC difference versus the ACTR package reference: `9.02e-17`
- ACTR HAC p-value difference versus reference: `8.65e-17`

## Benchmark reproduction

| model | rankic | hac_se | hac_p | dates |
|---|---|---|---|---|
| Strong CS | 0.051923737 | 0.024323217 | 0.032782276 | 860 |
| Strong TS | 0.050933711 | 0.015519719 | 0.0010312084 | 860 |
| Stable fusion | 0.057074672 | 0.021126408 | 0.0069010975 | 860 |
| Cross anchor | 0.067021227 | 0.02154038 | 0.0018618825 | 860 |
| ACTR benchmark | 0.067189597 | 0.021537741 | 0.001810836 | 860 |

## Paired comparisons

| comparison | increment | hac_se | hac_p | dates |
|---|---|---|---|---|
| ACTR benchmark minus Strong CS | 0.01526586 | 0.0078850028 | 0.052860017 | 860 |
| ACTR benchmark minus Strong TS | 0.016255886 | 0.016812448 | 0.33359612 | 860 |
| ACTR benchmark minus Stable fusion | 0.010114926 | 0.003671546 | 0.0058701111 | 860 |
| ACTR benchmark minus Cross anchor | 0.00016836993 | 6.2457121e-05 | 0.0070226475 | 860 |

Selected anchor alpha: `0.44999999999999996`
Selected residual gamma: `0.09999999999999999`

## TimeSafe verdicts

| Audit | Verdict | Checked / eligible | Changed values | Structural changes | Purpose |
|---|---|---|---|---|---|
| ACTR mature-feedback prefix audit | PASS | 21 / 21 | 0 | 0 | Exact ACTR mature-feedback function must remain prefix invariant |
| Legal fresh-state refit | WARN | 4 / 910 | 0 | 0 | Same legal learner on prefix and full input with an explicit cutoff |
| Full-sample training negative control | FAIL | 4 / 910 | 75 | 0 | Must detect fitting on future rows |
| Legal availability filtering | WARN | 4 / 910 | 0 | 0 | Mask not-yet-mature targets without changing legal predictions |
| Immature-label negative control | FAIL | 4 / 910 | 58 | 0 | Must detect materialized future targets used in training |
| Fresh-state determinism | WARN | 2 / 910 | 0 | 0 | Fresh refits with fixed seeds must reproduce predictions |
| Label maturity and purge | PASS | n/a |  |  | Validate close-to-next-open 20-session labels plus a 4-session gap |

## Interpretation

The full benchmark is recomputed on all 109,499 anonymous rows. The exact ACTR mature-feedback transform is checked exhaustively at all 21 eligible checkpoints in a pre-registered bounded window. The expensive fresh-state model audits use four pre-registered checkpoints and the first 24 anonymous assets across all 1,810 sessions. A clean sampled model audit therefore returns `WARN`, not `PASS`; the exact coverage is retained in JSON. Both deliberately leaking negative controls must return `FAIL`, while the event-aware label contract and purge audit must return `PASS`.

Every model audit is executed in a fresh interpreter and exchanges only JSON artifacts. This avoids process-local native state from one scikit-learn histogram-boosting workload affecting a later audit.

The TimeSafe repository does not redistribute the ACTR benchmark array, private feature definitions, strong-expert checkpoints, instruments, positions, or orders. The external ACTR research-only notice remains controlling.
