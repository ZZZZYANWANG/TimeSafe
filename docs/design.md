# Design: temporal integrity as an executable contract

## 1. Four independent audit surfaces

### Transform prefix invariance

\[
F(D_{1:t}) \stackrel{?}{=} F(D_{1:T})_t
\]

This detects future dependence entering through supplied rows. Inputs must be uniquely indexed and time-sorted; TimeSafe never silently reorders them.

### Fresh-state refit invariance

At each checkpoint TimeSafe creates independent, unfitted objects and invokes the same fitting code on the prefix and full input. Correct fitting code must enforce the legal cutoff itself. scikit-learn objects are cloned with `sklearn.base.clone`.

### Availability perturbation

A value can leak even when it is already stored on a historical row. TimeSafe masks declared unavailable cells and reruns the transform or fresh-state model. `reference="checkpoint"` is appropriate for labels available by model-fit time; `reference="row"` is appropriate for features that must have been published by each row's historical decision.

### Label and split contract

A label is represented by event-aware decision, entry, exit, availability, and safety-gap endpoints. Interval tables carry a contract fingerprint and are recomputed from the active contract and calendar before strict splitting.

## 2. Evidence semantics

`PASS` is deliberately narrow: all eligible checkpoints are tested, comparable evidence exists, no structural/value difference occurs, and no checkpoint remains unresolved. Sampled evidence is `WARN`; zero perturbation/output is `INCONCLUSIVE`.

## 3. Labeled comparison

Only cells present in both outputs contribute to value comparisons. Structural additions/removals are separate. Duplicate row/column labels are rejected because alignment would be ambiguous.

## 4. Input and stage isolation

User data is never passed as mutable shared state. Named pipeline stages receive private inputs and produce deep-copied snapshots. A later in-place mutation cannot rewrite earlier evidence.

## 5. Event model

```text
pre_open < open < intraday < close < after_close
```

A contract rejects execution at or before the decision. Holding duration depends on both entry and exit events, not only session offsets.

## 6. Purge and embargo

Each training sample has an information interval from decision through label availability plus the declared safety gap. Each test sample has an interval from decision through label end. Training intervals overlapping any merged test block are removed; a post-test embargo is applied after every test block.

## 7. Mature open-source reuse

TimeSafe delegates estimator cloning to scikit-learn and uses Hypothesis in the development suite for generated boundary cases. It intentionally does not reimplement CPCV, PSR/DSR, experiment tracking, backtesting, optimization, or execution simulation.

## 8. Real-model validation

The ACTR-20 experiment separates the full benchmark reproduction from bounded model audits. Every expensive audit executes in a fresh interpreter and writes a JSON artifact. Negative controls are mandatory: a validation suite that cannot reject deliberate full-sample training or immature labels is not accepted.
