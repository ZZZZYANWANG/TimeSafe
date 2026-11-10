# Design: temporal integrity as an executable contract

## 1. Two distinct audit modes

TimeSafe separates two questions that are often conflated.

### Transform audit

For a pure transform \(F\):

\[
F(D_{1:t}) \stackrel{?}{=} F(D_{1:T})_t.
\]

This detects future dependence that enters through the supplied input rows. It is appropriate for factors, rolling transforms, data cleaning, universe construction, and deterministic prediction functions whose complete state is inside the input or function call.

### Refit audit

For trainable state, TimeSafe creates two fresh estimator instances at every checkpoint and executes the same fitting code once on the prefix and once on the full input. The fitting code is expected to enforce the legal historical boundary even when it receives the full dataset.

This catches full-sample fitting of scalers, PCA, imputers, clustering, regressors, and classifiers. All trainable state must be created by the estimator factory.

## 2. Evidence semantics

`PASS` is deliberately narrow:

- every eligible checkpoint was checked;
- at least one labeled cell was compared;
- no value or structural difference occurred;
- no checkpoint errored or remained unresolved.

A uniform sample with no violations returns `WARN`, not `PASS`. Zero comparable output returns `INCONCLUSIVE`.

## 3. Labeled comparison

Value differences are computed only on cells present in both outputs. Structural additions and removals are counted separately. This prevents changed-cell fractions above 100% and makes dynamic-universe changes visible.

## 4. Pipeline localization

Each stage receives a deep copy and its output is snapshotted before the next stage runs. A later in-place mutation therefore cannot rewrite earlier evidence. Exceptions are recorded at the true failing stage; downstream stages are unresolved.

## 5. Event-aware label contract

TimeSafe orders within-session events as:

```text
pre_open < open < intraday < close < after_close
```

The contract rejects impossible sequences such as `close_t -> open_t`. It maps each decision to entry, exit, label availability, and the end of the declared safety gap.

## 6. Purge and embargo

Past-only walk-forward training applies:

\[
label\_available\_with\_gap < test\_decision.
\]

A post-test embargo is irrelevant to a strictly past-only training set and is rejected rather than ignored. Arbitrary candidate train/test masks use `purged_train_test_masks`, which removes interval overlap, safety-gap violations, and post-test embargo rows.

## 7. Mature open-source reuse

TimeSafe does not implement estimator cloning itself. The optional scikit-learn adapter delegates to `sklearn.base.clone`. It also does not reimplement CPCV, PSR, DSR, experiment tracking, or backtesting; those belong to skfolio/mlfinpy, MLflow, JoinQuant, Qlib, vectorbt, or an internal platform.
