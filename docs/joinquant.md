# JoinQuant integration

TimeSafe does not call JoinQuant APIs internally. It accepts a labeled pandas panel, which keeps the audit independent of any one research platform.

## Daily panel example

```python
from jqdata import get_price
import pandas as pd

codes = ["000001.XSHE", "600000.XSHG"]
raw = get_price(
    codes,
    start_date="2018-01-01",
    end_date="2025-12-31",
    frequency="daily",
    fields=["close", "volume", "money"],
    panel=False,
)

panel = raw.reset_index().rename(columns={"time": "date", "code": "asset"})
panel["return"] = panel.groupby("asset")["close"].pct_change()
panel = panel.set_index(["date", "asset"]).sort_index()
```

Audit a causal factor:

```python
from timesafe_quant import audit_prefix_invariance


def momentum_20(panel):
    return panel["close"].groupby(level="asset").pct_change(20)

result = audit_prefix_invariance(
    momentum_20,
    panel,
    time_level="date",
    checkpoint_policy="all",
    min_history=80,
)
```

Use sampled mode during development only; a clean sampled run returns `WARN` and reports its coverage.

## What the adapter does not solve

- whether a JoinQuant field was published before the declared decision time;
- historical revisions of fundamentals;
- exact intraday publication timestamps;
- simulated fills, limit-up queues, slippage, or capacity.

Those assumptions must be documented separately. Prefix invariance tests the code path against future-row dependence; it does not reconstruct an unavailable historical vintage. Trainable preprocessing and models should use the refit audit with fresh estimator state.
