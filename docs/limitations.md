# Limitations

## A pass is conditional

A TimeSafe pass is conditional on:

- the supplied data;
- the selected checkpoints;
- numerical tolerances;
- the declared trading calendar and label contract;
- the behavior exercised by the test inputs.

It is not a proof of universal causality.

## Vintage blindness

If a data provider has overwritten historical records with later revisions, both the prefix and full dataset may contain the same contaminated value. Prefix invariance cannot detect information that is already embedded in the historical prefix. Use point-in-time storage and publication timestamps.

## Stateful external systems

A transform that reads files, databases, model registries, network APIs, global variables, or wall-clock time may not be reproducible from the pandas argument alone. Wrap or freeze external state before auditing.

## Expensive models

Black-box prefix testing reruns the pipeline multiple times. For expensive training loops:

- sample fewer checkpoints;
- audit preprocessing stages separately;
- cache immutable raw inputs outside the transform;
- use a reduced but structurally representative dataset;
- test fitted preprocessing and prediction components independently.

Do not cache outputs in a way that bypasses the historical-prefix refit you intend to test.

## Numerical sensitivity

Parallel linear algebra, GPU kernels, stochastic training, and floating-point reduction order can create tiny differences. Run the determinism audit first and choose tolerances appropriate to the numerical stack. Large or systematic differences should not be dismissed as floating-point noise.

## Label conventions

The session-based contract assumes that the user has correctly specified execution offset, holding-period counting, publication lag, and safety gap. Markets with overnight sessions, auction events, or non-session timestamps may need an event-time adapter.
