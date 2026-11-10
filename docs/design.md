# Design: temporal integrity as an executable contract

## 1. The causal identity

Let `D[:t]` denote the information genuinely available up to time `t`, and let `F` be an arbitrary factor, preprocessing, or prediction pipeline. A necessary condition for a historical output to be causally well-defined is:

\[
F(D_{1:t}) = F(D_{1:T})_t,
\]

subject to numerical tolerance and an explicit data-vintage contract.

This is a **necessary**, not sufficient, condition. A vendor may have already backfilled revised information into `D[:t]`; TimeSafe cannot infer the missing vintage from the final table. But when the identity fails, the implementation itself is demonstrably sensitive to future rows.

## 2. Why static scans are insufficient

A source scanner can detect obvious patterns such as negative shifts or centered windows, but leakage often enters through state fitted on the whole sample:

- scalers, encoders, PCA and clustering;
- imputation and outlier thresholds;
- universe definitions;
- state labels and regime boundaries;
- feature-selection thresholds;
- retrospective contract mappings.

A dynamic prefix test observes the resulting behavior regardless of how that behavior was produced.

## 3. Labeled comparison

TimeSafe requires unique pandas indexes because quantitative data are panels, not anonymous matrices. A silent positional comparison can declare equality after rows have been reordered, assets dropped, or columns changed. The comparison therefore aligns both indexes and columns and counts structural mismatches explicitly.

## 4. Stage localization

A black-box failure proves that the final pipeline is not prefix invariant but may not identify the cause. `CausalPipeline` records each named stage and applies the same prefix/full comparison to every intermediate output. The first failing stage is a practical debugging hypothesis, not a formal causal attribution: a downstream stage may mask or amplify an upstream error.

## 5. Determinism before causality

A stochastic model with an unfixed seed can produce different values across runs even without future leakage. TimeSafe includes a separate determinism audit. A prefix failure paired with a determinism failure should first be debugged as a reproducibility issue.

## 6. Label maturity

For a decision at session `t`, entry offset `e`, and holding period `h`, the label interval ends at:

\[
t + e + h - 1.
\]

If publication lag is `p`, the label is not available until:

\[
t + e + h - 1 + p.
\]

A past training observation is eligible for a walk-forward test beginning at `T_0` only if its label is available strictly before `T_0`. This condition is stronger and clearer than applying a hard-coded purge without checking the actual event interval.

## 7. Scope boundary

TimeSafe deliberately does not implement:

- backtesting or execution simulation;
- CPCV, DSR or PBO;
- experiment tracking;
- point-in-time vendor storage;
- economic mechanism validation;
- portfolio construction.

Those are separate layers. A small tool with a precise contract is easier to validate, reuse, and trust than an incomplete all-in-one platform.
