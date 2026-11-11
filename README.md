# TimeSafe Quant

**Executable temporal-integrity contracts for quantitative research pipelines.**

> A backtest can report what a strategy earned. TimeSafe asks a prior question:
> **was every historical feature, fitted state, label, and split legally available at the time?**

TimeSafe is a small, platform-neutral Python library for detecting hidden future-data dependence. It audits four distinct failure surfaces:

1. **Pure transforms** — does appending future rows rewrite a historical factor or prediction?
2. **Trainable pipelines** — does a freshly rebuilt model change when its fitting code can see future rows?
3. **Materialized-but-unavailable fields** — is a value already present in the table even though it was not yet published at the relevant decision point?
4. **Labels and splits** — are decision, execution, exit, label maturity, purge, safety gap, and embargo mutually consistent?

TimeSafe does **not** replace a backtester, CPCV/DSR/PBO library, experiment tracker, optimizer, or execution simulator. It is intended to sit beside JoinQuant, Qlib, vectorbt, skfolio/mlfinpy, MLflow, scikit-learn, or an internal platform.

---

## Why temporal integrity needs executable tests

Look-ahead leakage is often subtler than `shift(-1)`:

- global standardization, PCA, clustering, or imputation fitted on the full history;
- centered windows and two-sided filters;
- retrospective continuous-contract or universe construction;
- revised fundamentals replacing the vintage known at the time;
- a value stored on a historical row before its real publication date;
- same-day data used before its within-session release event;
- pre-fitted model state captured in a closure, cache, or checkpoint;
- forward-return labels entering training before they mature;
- train/test rows whose label information intervals overlap.

For a deterministic transform \(F\), TimeSafe tests prefix invariance:

\[
F(D_{1:t}) \stackrel{?}{=} F(D_{1:T})_t.
\]

For trainable pipelines it rebuilds **fresh estimator state** at every checkpoint. For already materialized values it perturbs the table according to an explicit availability contract. For labels it compiles an event-aware information interval and constructs purge/embargo masks.

---

## Install

```bash
pip install -e .
```

For scikit-learn estimator cloning:

```bash
pip install -e ".[model]"
```

Development:

```bash
pip install -e ".[dev]"
pytest --cov=timesafe_quant --cov-fail-under=85
ruff check .
python -m compileall -q src examples experiments tests
```

---

## 1. Pure-transform prefix audit

```python
import pandas as pd
from timesafe_quant import audit_prefix_invariance


def factor(panel: pd.DataFrame) -> pd.Series:
    return (
        panel["return"]
        .groupby(level="asset", group_keys=False)
        .ewm(span=20, adjust=False)
        .mean()
        .rename("factor")
    )


result = audit_prefix_invariance(
    factor,
    panel,                       # unique, time-sorted (date, asset) MultiIndex
    time_level="date",
    checkpoint_policy="all",     # exhaustive evidence
    min_history=60,
)
```

TimeSafe never sorts research data silently: an unsorted temporal coordinate is rejected because rolling transforms depend on row order. User frames are isolated from in-place mutation.
The CSV CLI follows the same rule and preserves input row order rather than silently sorting it.

### Evidence semantics

| Status | Meaning |
|---|---|
| `PASS` | Every eligible checkpoint was tested, comparable evidence exists, and no violation occurred |
| `WARN` | No violation was found, but evidence is sampled or partially unresolved |
| `INCONCLUSIVE` | The audit produced no comparable evidence or did not exercise the claimed boundary |
| `FAIL` | Values or labeled output structure changed |
| `ERROR` | The audit itself could not complete |

The final timestamp is excluded because no future observation exists beyond it. A clean sampled audit intentionally returns `WARN`, with exact eligible/checked counts and coverage.

---

## 2. Fresh-state refit audit

A transform audit cannot detect a scaler, PCA, or predictive model fitted before the function was called. Refit auditing creates independent, unfitted objects for both paths:

```python
import pandas as pd
from sklearn.linear_model import Ridge
from timesafe_quant import (
    audit_refit_prefix_invariance,
    audit_refit_determinism,
    sklearn_estimator_factory,
)


def fit_predict(estimator, data: pd.DataFrame, checkpoint):
    dates = data.index.get_level_values("date")
    train = dates < checkpoint                 # must enforce the legal cutoff
    current = dates == checkpoint
    estimator.fit(data.loc[train, ["feature"]], data.loc[train, "target"])
    prediction = estimator.predict(data.loc[current, ["feature"]])
    return pd.Series(prediction, index=data.index[current], name="prediction")


factory = sklearn_estimator_factory(Ridge(alpha=10.0))
result = audit_refit_prefix_invariance(
    factory,
    fit_predict,
    panel,
    time_level="date",
    checkpoint_policy="all",
    min_history=60,
)
reproducibility = audit_refit_determinism(
    factory,
    fit_predict,
    panel,
    time_level="date",
    checkpoint_policy="all",
    min_history=60,
)
```

`sklearn_estimator_factory` delegates to the mature `sklearn.base.clone`; TimeSafe does not implement an inferior estimator-copy protocol. Non-scikit-learn models can use any factory that returns a fresh, unfitted object.

All learned state must be created inside the audited factory/path. Hidden remote caches or external pre-fitted checkpoints remain outside the black-box guarantee.

---

## 3. Availability perturbation audit

Prefix invariance alone cannot detect a future label or revised feature that is already materialized on a historical row. TimeSafe can mask values before their declared availability and refit the pipeline.

```python
from timesafe_quant import AvailabilityRule, audit_refit_availability_invariance

rule = AvailabilityRule(
    value_column="future_20d_return",
    available_at_column="target_available_at",
    available_event="after_close",
    reference="checkpoint",  # label must be available by the model-fit checkpoint
)

result = audit_refit_availability_invariance(
    factory,
    fit_predict,
    panel,
    rules=[rule],
    time_level="date",
    checkpoint_policy="all",
    decision_event="close",
    min_history=60,
)
```

Two reference modes are supported:

- `reference="checkpoint"`: labels or training data must be available by the current audit/model-fit point;
- `reference="row"`: each historical feature must have been available at its own row decision point, which is appropriate for announcement-dated fundamentals and other point-in-time features.

The audit is `INCONCLUSIVE` when no value is actually perturbed—for example, all declared unavailable cells were already missing.

---

## 4. Stage-aware localization

```python
from timesafe_quant import CausalPipeline, PipelineStep, audit_pipeline_prefix_invariance

pipeline = CausalPipeline([
    PipelineStep("raw", lambda x: x[["return"]]),
    PipelineStep("standardize", lambda x: (x - x.mean()) / x.std()),
    PipelineStep("smooth", lambda x: x.ewm(span=10, adjust=False).mean()),
])

result = audit_pipeline_prefix_invariance(
    pipeline,
    panel,
    time_level="date",
    checkpoint_policy="all",
    min_history=60,
)
print(result.first_failing_stage)  # standardize
```

Every stage receives a private input and its output is snapshotted. A later in-place mutation cannot rewrite earlier evidence; exceptions are attributed to the true failing stage.

---

## 5. Event-aware label maturity, purge, and embargo

```python
from timesafe_quant import LabelContract, build_label_intervals

contract = LabelContract(
    decision_event="close",
    execution_offset_sessions=1,
    execution_event="open",
    holding_period_sessions=20,
    exit_event="close",
    label_publication_lag_sessions=0,
    label_available_event="after_close",
    safety_gap_sessions=4,
)

intervals = build_label_intervals(
    decision_sessions=trading_calendar,
    calendar=trading_calendar,
    contract=contract,
)
```

Supported within-session order:

```text
pre_open < open < intraday < close < after_close
```

Impossible contracts such as `close_t -> open_t` are rejected. Holding-period endpoints depend on entry and exit events: `open(t+1) -> close`, one session, ends at `close(t+1)`; `close(t+1) -> close`, one complete interval, ends at `close(t+2)`.

Interval tables carry a contract fingerprint, but TimeSafe does not trust the declaration alone: with a calendar it recomputes every endpoint and rejects stale or tampered tables.

```python
from timesafe_quant import purged_walk_forward_masks, purged_train_test_masks

train, test = purged_walk_forward_masks(
    intervals,
    test_start=trading_calendar[-80],
    test_end=trading_calendar[-60],
    contract=contract,
    calendar=trading_calendar,
)

clean_train, test = purged_train_test_masks(
    intervals,
    candidate_train_mask=candidate_train,
    test_mask=test_mask,
    contract=contract,
    embargo_sessions=5,
    calendar=trading_calendar,
)
```

The generic splitter handles multiple disjoint test blocks, applies the safety gap in code, and applies post-test embargo after every merged test-information block. Empty train or test masks do not receive a purge `PASS`.

---

## 6. Evidence cards and CLI

```python
from timesafe_quant import make_report, write_report

report = make_report(prefix=result)
write_report(report, "report.md")
write_report(report, "report.json")
```

TimeSafe reports coverage, changed values, structural differences, masked cells, first violation, contract fingerprint, maturity, overlap, safety-gap, and embargo evidence. It does not issue a mysterious trust score.

```bash
timesafe demo

timesafe prefix \
  --input panel.csv \
  --callable my_module:build_factor \
  --time-col date --asset-col asset \
  --checkpoint-policy all \
  --min-history 60 \
  --output report.md

timesafe availability \
  --input panel.csv \
  --callable my_module:build_factor \
  --time-col date --asset-col asset \
  --rule future_return:available_at:after_close:checkpoint \
  --output availability.md
```

CLI exit codes: `0=PASS`, `1=WARN`, `2=FAIL/INCONCLUSIVE`, `3=ERROR`. The built-in demonstration intentionally contains failing controls and exits zero.

---

## 7. Real public-model experiment: ACTR-20

`experiments/actr20_public/` applies TimeSafe to the previously published, privacy-hardened ACTR temporal–cross-sectional joint-model benchmark.

The external ACTR research package is **not vendored** into this MIT repository. Supply its extracted path:

```bash
python experiments/actr20_public/run_all.py \
  --actr-root /path/to/ACTR20_JQ_Public_Research_Package_20261110 \
  --output-dir experiments/actr20_public/results/generated
```

The experiment:

1. recomputes the full 109,499-row ACTR public benchmark;
2. verifies RankIC, HAC p-value, selected configuration, shape, and data SHA against the external package's signed-off reference files;
3. runs legal fresh-state refits and availability masking;
4. runs deliberately leaking full-sample and immature-label negative controls;
5. checks fresh-state determinism and the 20-session close-to-next-open label contract with a four-session safety gap;
6. executes each expensive model audit in a fresh interpreter and exchanges only atomic JSON artifacts;
7. binds every worker artifact to the same benchmark SHA-256, TimeSafe version, anonymized-asset-set hash, row/session counts, and pre-registered checkpoints before aggregation.

Reference aggregate evidence is retained under `experiments/actr20_public/results/reference_v031/`. The anonymous dataset, private factors, expert checkpoints, instruments, positions, and orders are not included.

---

## Scope

TimeSafe v0.3 covers:

- deterministic pandas transforms with unique labeled output;
- exhaustive, uniform-sampled, or explicit checkpoint plans;
- numeric and structural output comparison;
- stage-level leak localization;
- fresh-state model refitting and determinism;
- scikit-learn cloning;
- availability perturbation at checkpoint or historical-row time;
- within-session decision/execution/exit validation;
- contract fingerprints plus endpoint recomputation;
- label maturity, safety-gap purge, multiple test blocks, and post-test embargo;
- Markdown/JSON evidence cards, CLI, CI, and a real public-model experiment.

A `PASS` does **not** prove:

- that a vendor supplied the correct historical data vintage;
- that user-provided publication timestamps are true;
- that uncontrolled external state was trained legally;
- that a strategy survived multiple-testing correction;
- that fills, costs, impact, or capacity are realistic;
- that the claimed economic mechanism is true.

Recommended complements:

- **skfolio / mlfinpy** for CPCV, PSR, and DSR;
- **MLflow** for experiment tracking and lineage;
- **JoinQuant / Qlib / vectorbt** for research and backtesting;
- market-specific cost, capacity, placebo, and mechanism tests.

TimeSafe determines whether evidence is temporally eligible for evaluation. It does not replace the evaluation.

## License

MIT. The external ACTR package remains governed by its own research-only terms.
