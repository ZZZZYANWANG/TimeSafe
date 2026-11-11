# ACTR-20 public-model validation

This experiment applies TimeSafe to the previously published ACTR-20 privacy-hardened benchmark. It combines two distinct forms of evidence:

1. **exact benchmark reproduction** using the external ACTR public framework on all 109,499 anonymous rows;
2. **bounded temporal-integrity audits** using the same public learner family, the exact mature-feedback function, and deliberately leaking negative controls.

The ACTR package remains governed by its own research-only notice and is **not** vendored into TimeSafe's MIT-licensed source tree.

## Input integrity

Before importing external ACTR code, the runner verifies every file listed in the package's `MANIFEST.sha256`, confirms the package verification status, and checks the benchmark identity. The benchmark stage then verifies RankIC, HAC p-value, selected alpha/gamma, row/session counts, and data SHA-256 against the package reference artifacts.

## One-command run

Extract `ACTR20_JQ_Public_Research_Package_20261110.zip`, then run:

```bash
python experiments/actr20_public/run_all.py \
  --actr-root /path/to/ACTR20_JQ_Public_Research_Package_20261110 \
  --output-dir /tmp/timesafe-actr20
```

Use a fresh output directory, or add `--overwrite` to replace a prior generated run.

The orchestrator is synchronous and observable, but every expensive model audit is executed in a **fresh Python interpreter** with native thread pools limited to one thread. This avoids process-local BLAS/OpenMP state leaking from one histogram-boosting workload into the next. Each stage has its own log, and `run_status.json` plus every worker evidence file are written atomically.
The default ceilings are one hour per isolated stage and six hours overall. Use `--stage-timeout-seconds` and `--total-timeout-seconds` (the old `--timeout-seconds` name remains an alias for the total limit) to adapt the run to the host.

A Bash convenience wrapper is also available:

```bash
experiments/actr20_public/run_all.sh \
  --actr-root /path/to/ACTR20_JQ_Public_Research_Package_20261110 \
  --output-dir /tmp/timesafe-actr20
```

## Explicit stages

The stages can be run independently when a hosted notebook imposes a short foreground timeout:

```bash
python experiments/actr20_public/run_benchmark.py --actr-root ... --output-dir ...
python experiments/actr20_public/run_audits.py --actr-root ... --output-dir ...
python experiments/actr20_public/assemble_report.py --results-dir ... --output ...
```

## Audits

| Audit | Expected verdict | What it tests |
|---|---:|---|
| Exact ACTR mature-feedback prefix audit | `PASS` | The public feedback function uses only targets whose 25-session availability lag has matured |
| Legal fresh-state refit | `WARN` or `PASS` | Same public learner on a historical prefix and full input with an explicit legal cutoff |
| Full-sample training negative control | `FAIL` | TimeSafe must detect fitting on future rows |
| Legal availability filtering | `WARN` or `PASS` | Masking not-yet-mature targets must not change a legally trained model |
| Immature-label negative control | `FAIL` | TimeSafe must detect materialized future targets entering training |
| Fresh-state determinism | `WARN` or `PASS` | Fixed-seed fresh refits must reproduce predictions |
| Label maturity and purge | `PASS` | Close-to-next-open 20-session labels plus a four-session safety gap |

The expensive refit audits use four pre-registered checkpoints and the first 24 anonymous assets across all 1,810 sessions. Every worker records the same benchmark SHA-256, TimeSafe version, anonymized-asset-set hash, row/session counts, and checkpoint plan; aggregation fails on any mixed or stale artifact. A clean sampled audit therefore returns `WARN`, not a false full-history `PASS`. Exact coverage remains in `timesafe_audits.json`.

## Committed reference evidence

`results/reference_v031/` contains the aggregate, non-sensitive outputs from an actual full benchmark reproduction and TimeSafe audit run. `MANIFEST.sha256` covers every committed result file. The benchmark array itself remains external under the ACTR research-only notice.

The committed evidence is not treated as a substitute for rerunning the external package. It provides an inspectable record of the model metrics, negative-control verdicts, contract audit, source benchmark identity, and exact reference differences observed in the validated run.

## Output

The generated directory contains:

```text
ACTR20_TIMESAFE_EXPERIMENT.md
benchmark_summary.csv
benchmark_pairs.csv
benchmark_config.json
benchmark_verification.json
timesafe_audits.json
run_status.json
logs/
audit_workers/
```

Only aggregate reference outputs are eligible for committing to TimeSafe. The anonymous benchmark array, daily predictions, private feature definitions, strong-expert checkpoints, instruments, positions, and orders are not included.

## Interpretation boundary

The ACTR anonymous benchmark is block-shuffled. It tests structural reproduction and software behavior, not chronological investment performance. The experiment demonstrates that TimeSafe can audit a realistic trainable joint-model workflow and reject deliberately leaking controls; it does not certify the original economic alpha, costs, capacity, or untouched out-of-sample performance.
