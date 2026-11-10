# TimeSafe Quant

**Black-box temporal integrity and label-maturity auditing for quantitative research pipelines.**

> A backtest platform can tell you how much a strategy earned. TimeSafe asks a more basic question: **was the historical signal allowed to know what it knew?**

TimeSafe is a small, platform-neutral Python library for detecting hidden future-data dependence in factors, feature pipelines, and model outputs. It focuses on three tasks:

1. **Prefix invariance** — would the value published at time `t` change after future rows are appended?
2. **Label maturity and purge** — was a target actually observable when it entered training, and do train/test label intervals overlap?
3. **Evidence reporting** — can the temporal assumptions and violations be rendered as an auditable Markdown/JSON evidence card?

It does **not** replace a backtester, model trainer, experiment tracker, or portfolio optimizer. Use it alongside JoinQuant, Qlib, vectorbt, skfolio, MLflow, or an internal research platform.

---

## Why this exists

Look-ahead leakage is often subtler than `shift(-1)`:

- global mean/std normalization fitted on the full history;
- PCA or clustering fitted on all dates and backfilled into the past;
- centered rolling windows or two-sided filters;
- retrospective continuous-contract construction;
- revised fundamentals replacing the vintage known at the time;
- same-day member-position data published after the decision time;
- labels that had not matured before model fitting;
- random cross-validation on overlapping forward-return labels.

Many of these pipelines produce smooth, plausible signals and excellent backtests. Static source scanning is not enough. TimeSafe treats the pipeline as a black box and tests:

\[
F(D_{1:t}) \stackrel{?}{=} F(D_{1:T})_t.
\]

If adding future rows changes an output that was supposedly known at `t`, the implementation is not temporally invariant under the tested contract.

---

## Quick start

```bash
pip install -e .
timesafe demo
```

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
    panel,                 # unique (date, asset) MultiIndex
    time_level="date",
    n_checkpoints=20,
    min_history=60,
)

print(result.status)
print(result.first_violation)
```

The callable must return a uniquely indexed `pandas.Series` or `pandas.DataFrame`, aligned to the historical observations it describes.

---

## Prefix-invariance audit

For every historical checkpoint, TimeSafe runs the transform on:

- the prefix that was available at the time;
- the full history, then extracts the output at that same historical time.

It reports changed cells, missing rows/columns, numerical differences, and the first violating date.

```python
from timesafe_quant import audit_prefix_invariance

result = audit_prefix_invariance(
    transform=my_pipeline,
    data=panel,
    time_level="date",
    n_checkpoints=12,
    min_history=60,
    atol=1e-10,
    rtol=1e-8,
)
```

### Stage localization

```python
from timesafe_quant import CausalPipeline, PipelineStep, audit_pipeline_prefix_invariance

pipeline = CausalPipeline([
    PipelineStep("raw", lambda x: x[["return"]]),
    PipelineStep("standardize", lambda x: (x - x.mean()) / x.std()),
    PipelineStep("smooth", lambda x: x.ewm(span=10, adjust=False).mean()),
])

result = audit_pipeline_prefix_invariance(pipeline, panel, time_level="date")
print(result.first_failing_stage)  # standardize
```

---

## Label maturity and purge

```python
from timesafe_quant import LabelContract, build_label_intervals

contract = LabelContract(
    execution_offset_sessions=1,       # decide after close t, enter at open t+1
    holding_period_sessions=20,        # exit at close t+20
    label_publication_lag_sessions=0,
    safety_gap_sessions=4,
)

intervals = build_label_intervals(
    decision_sessions=trading_calendar,
    calendar=trading_calendar,
    contract=contract,
)

print(contract.recommended_purge_sessions)  # 24
```

TimeSafe identifies immature labels, detects train/test interval overlap, and builds a past-only purged walk-forward split.

```python
from timesafe_quant import audit_label_maturity, purged_walk_forward_masks

train_mask, test_mask = purged_walk_forward_masks(
    intervals,
    test_start=trading_calendar[-60],
    test_end=trading_calendar[-40],
)

result = audit_label_maturity(
    intervals,
    contract=contract,
    train_mask=train_mask,
    test_mask=test_mask,
    calendar=trading_calendar,
)
```

---

## Evidence card

```python
from timesafe_quant import audit_determinism, make_report, write_report

prefix = audit_prefix_invariance(factor, panel, time_level="date")
determinism = audit_determinism(factor, panel, repetitions=3)
report = make_report(prefix=prefix, determinism=determinism)

write_report(report, "report.md")
write_report(report, "report.json")
```

TimeSafe deliberately avoids a mysterious trust score. It exposes which contract passed, what failed, and the evidence behind each verdict.

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
| Centered rolling mean | FAIL | The window reaches forward in time |
| Full-sample PCA | FAIL | Future rows change historical loadings |
| Backfill | FAIL | Missing historical values are filled from the future |

`examples/joinquant_adapter.py` shows how to convert JoinQuant data to a `(date, asset)` panel and audit a factor in a research notebook.

---

## Scope and limitations

### TimeSafe can detect

- historical outputs rewritten by future rows;
- centered or two-sided transforms;
- full-sample preprocessing and retrospective refits;
- nondeterminism that can masquerade as leakage;
- immature labels and overlapping train/test label intervals;
- the first leaking stage in a named pandas pipeline.

### It cannot prove by itself

- that an upstream vendor dataset was truly point-in-time;
- that a publication timestamp supplied by the user is correct;
- that the backtest has realistic fills, costs, and capacity;
- that a strategy is not overfit after many research trials;
- that an economically stated mechanism is true.

Recommended complements:

- **skfolio / mlfinpy** for purged model selection, CPCV, PSR, and DSR;
- **MLflow** for experiment tracking and lineage;
- **JoinQuant / Qlib / vectorbt** for research and backtesting;
- cost, capacity, placebo, and mechanism-specific falsification tests.

See [design](docs/design.md), [JoinQuant integration](docs/joinquant.md), [limitations](docs/limitations.md), and the [Chinese research note](docs/research_note_zh.md).

---

# 中文说明

量化平台通常能告诉我们“策略历史赚了多少”，却未必能告诉我们：**过去时点的因子值，是否在后来加入未来数据后被悄悄改写。**

TimeSafe只做三件事：

1. 把因子或模型当作黑盒，比较 `F(截至t的数据)` 与 `F(全历史数据)在t时点的输出`；
2. 根据决策、成交和持有期推导标签成熟时间与最小purge；
3. 输出可以审计的Markdown/JSON证据卡。

它特别适合发现全样本标准化、全样本PCA、居中窗口、双向滤波、事后连续合约、未来回填、未成熟标签和重叠标签污染等问题。

本项目不重复实现回测、DSR、PBO、组合优化和实验管理。它应当与聚宽、Qlib、vectorbt、skfolio和MLflow结合使用：**现有工具评价结果，TimeSafe先检查这个结果是否有资格被评价。**

---

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check .
```

## License

MIT. See [LICENSE](LICENSE).
