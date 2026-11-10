# TimeSafe Quant

**Temporal integrity, refit, and label-maturity auditing for quantitative research pipelines.**

> A backtest can report what a strategy earned. TimeSafe asks a prior question:
> **was the historical signal allowed to know what it knew?**

TimeSafe is a small, platform-neutral Python library for turning temporal assumptions into executable tests. It audits three distinct objects:

1. **Pure transforms** — would a factor or feature value at time `t` change after future rows are appended?
2. **Trainable pipelines** — would a freshly rebuilt model produce a different historical prediction when its fitting code receives future rows?
3. **Labels and splits** — were targets mature, was the decision/entry/exit sequence possible, and were purge, safety-gap, and embargo rules actually applied?

TimeSafe deliberately does **not** implement backtesting, CPCV, DSR/PBO, experiment tracking, portfolio optimization, or execution simulation. Use it with JoinQuant, Qlib, vectorbt, skfolio/mlfinpy, MLflow, scikit-learn, or an internal research platform.

---

## Why this exists

Look-ahead leakage is often subtler than `shift(-1)`:

- global standardization fitted on the full history;
- PCA, clustering, imputation, or state labels fitted on all dates;
- centered rolling windows or two-sided filters;
- retrospective continuous-contract construction;
- revised fundamentals replacing the vintage known at the time;
- same-day data published after the decision event;
- pre-fitted model state captured in a closure or global object;
- labels entering training before their outcomes matured;
- random validation on overlapping forward-return intervals.

Many such pipelines produce smooth factors and excellent backtests. Static code scanning cannot reliably detect them. TimeSafe uses executable temporal contracts.

For a pure transform \(F\), prefix invariance is:

\[
F(D_{1:t}) \stackrel{?}{=} F(D_{1:T})_t.
\]

For a trainable pipeline, TimeSafe also rebuilds **fresh estimator state** at each checkpoint and compares the prefix-only fit with the same fitting code run against the full input.

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
pytest
ruff check .
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
    panel,                       # unique (date, asset) MultiIndex
    time_level="date",
    checkpoint_policy="all",     # exhaustive evidence
    min_history=60,
)

print(result.status)
print(result.coverage)
print(result.first_violation)
```

### Evidence semantics

- `PASS`: every eligible historical checkpoint was tested and produced comparable output.
- `WARN`: no violation was found, but the audit sampled checkpoints or had unresolved checkpoints.
- `INCONCLUSIVE`: no comparable labeled cells were produced.
- `FAIL`: values or labeled output structure changed.
- `ERROR`: the audit could not complete.

A clean sampled audit is intentionally **not** called `PASS`.

```python
sampled = audit_prefix_invariance(
    factor,
    panel,
    time_level="date",
    checkpoint_policy="uniform",
    n_checkpoints=20,
    min_history=60,
)
# usually WARN, with explicit eligible/checked counts and coverage
```

The final timestamp is excluded because `prefix == full history` there and provides no future-data evidence.

---

## 2. Refit-aware model audit

A transform audit cannot detect trainable state that was fitted before the function was called. Use the refit audit when the research claim includes model training, preprocessing fit, PCA fit, imputation fit, or any learned state.

```python
import pandas as pd
from sklearn.linear_model import Ridge
from timesafe_quant import (
    audit_refit_prefix_invariance,
    sklearn_estimator_factory,
)


def fit_predict(estimator, data: pd.DataFrame, checkpoint):
    dates = data.index.get_level_values("date")

    # The fitting code must enforce the legal training boundary even when
    # TimeSafe deliberately passes the full dataset.
    train = dates < checkpoint
    current = dates == checkpoint

    estimator.fit(data.loc[train, ["feature"]], data.loc[train, "target"])
    prediction = estimator.predict(data.loc[current, ["feature"]])
    return pd.Series(prediction, index=data.index[current], name="prediction")


result = audit_refit_prefix_invariance(
    sklearn_estimator_factory(Ridge(alpha=10.0)),
    fit_predict,
    panel,
    time_level="date",
    checkpoint_policy="all",
    min_history=60,
)
```

`sklearn_estimator_factory` delegates cloning to the mature `sklearn.base.clone` implementation. For non-scikit-learn models, pass any factory that returns a **fresh, unfitted object**.

### Refit contract

All trainable state must be created by the factory. TimeSafe cannot prove the legality of hidden global caches, vendor vintages, remote feature stores, or external pre-fitted objects that are not rebuilt inside the audit.

---

## 3. Stage-aware localization

```python
from timesafe_quant import (
    CausalPipeline,
    PipelineStep,
    audit_pipeline_prefix_invariance,
)

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

Each stage is deep-copied into an immutable snapshot. In-place mutation by a later stage cannot rewrite an earlier stage's evidence. Exceptions are attributed to the actual failing stage; downstream stages become unresolved rather than falsely blamed.

---

## 4. Event-aware label maturity and purge

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

print(contract.recommended_purge_sessions)  # 24
```

Supported within-session order:

```text
pre_open < open < intraday < close < after_close
```

Impossible contracts are rejected. For example, a decision at `close_t` cannot execute at `open_t`.

### Past-only walk-forward split

```python
from timesafe_quant import purged_walk_forward_masks

train_mask, test_mask = purged_walk_forward_masks(
    intervals,
    test_start=trading_calendar[-80],
    test_end=trading_calendar[-60],
    contract=contract,
    calendar=trading_calendar,
)
```

The generated training mask applies the declared `safety_gap_sessions` in code; it is not merely reported.

A post-test embargo is not meaningful for a strictly past-only training set. Passing a non-zero embargo therefore raises instead of being silently ignored. For arbitrary candidate train/test rows, use:

```python
from timesafe_quant import purged_train_test_masks

clean_train, test = purged_train_test_masks(
    intervals,
    candidate_train_mask=candidate_train,
    test_mask=test_mask,
    contract=contract,
    embargo_sessions=5,
    calendar=trading_calendar,
)
```

---

## 5. Evidence cards

```python
from timesafe_quant import audit_determinism, make_report, write_report

prefix = audit_prefix_invariance(
    factor,
    panel,
    time_level="date",
    checkpoint_policy="all",
    min_history=60,
)
determinism = audit_determinism(factor, panel, repetitions=3)

report = make_report(prefix=prefix, determinism=determinism)
write_report(report, "report.md")
write_report(report, "report.json")
```

TimeSafe does not emit a mysterious trust score. It reports:

- exhaustive vs sampled mode;
- eligible and checked checkpoints;
- exact coverage;
- changed values;
- structural additions/removals;
- empty/no-evidence states;
- first violating checkpoint or stage;
- event contract;
- immature labels;
- interval overlap, safety-gap, and embargo violations.

---

## Demonstration gallery

```bash
python examples/leakage_gallery.py
```

| Pipeline | Expected | Reason |
|---|---:|---|
| Causal EMA | PASS | Future observations do not alter past values |
| Expanding z-score | PASS | Parameters use the historical prefix only |
| Full-sample z-score | FAIL | Mean/std include future rows |
| Centered rolling mean | FAIL | Window reaches forward |
| Full-sample PCA | FAIL | Future rows change historical loadings |
| Backfill | FAIL | Missing history is filled from the future |

Additional examples:

- `examples/custom_pipeline.py`
- `examples/refit_model.py`
- `examples/joinquant_adapter.py`

---

## What TimeSafe covers

TimeSafe v0.2 supports:

- deterministic pandas transforms with unique labeled output;
- exhaustive, uniform-sampled, or explicit checkpoint plans;
- value and output-structure comparison without double-counting;
- named-stage leak localization;
- fresh-state refit audits for trainable models;
- optional scikit-learn cloning;
- within-session decision/execution/exit validation;
- label maturity and event-aware `as_of` checks;
- actual safety-gap purge;
- arbitrary split purge and post-test embargo;
- Markdown/JSON evidence reports;
- CLI, tests, CI, and JoinQuant-oriented examples.

## What it does not prove

A `PASS` does not prove:

- that a vendor dataset was point-in-time correct;
- that supplied publication timestamps are true;
- that hidden external state was trained legally;
- that fills, costs, impact, or capacity are realistic;
- that a strategy survived multiple-testing adjustment;
- that a claimed economic mechanism is true.

Recommended complements:

- **skfolio / mlfinpy**: purged model selection, CPCV, PSR, DSR;
- **MLflow**: experiment tracking and lineage;
- **JoinQuant / Qlib / vectorbt**: research and backtesting;
- market-specific cost, capacity, placebo, and mechanism tests.

TimeSafe checks whether a result is temporally eligible for evaluation. It does not replace the evaluation.

---

## CLI

```bash
timesafe demo
```

```bash
timesafe prefix \
  --input panel.csv \
  --callable my_module:build_factor \
  --time-col date \
  --asset-col asset \
  --checkpoint-policy all \
  --min-history 60 \
  --output report.md
```

```bash
timesafe maturity \
  --sessions trading_calendar.csv \
  --time-col date \
  --decision-event close \
  --execution-offset 1 \
  --execution-event open \
  --holding-period 20 \
  --exit-event close \
  --label-available-event after_close \
  --safety-gap 4 \
  --output maturity.json
```

CLI exit codes:

- `0`: PASS;
- `1`: WARN;
- `2`: FAIL or INCONCLUSIVE;
- `3`: ERROR.

The built-in demo intentionally contains failing cases and exits `0`.

---

## License

MIT. See [LICENSE](LICENSE).
