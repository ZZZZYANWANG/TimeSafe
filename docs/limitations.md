# Limitations

TimeSafe is an executable temporal-contract auditor, not a proof of strategy validity.

## Data vintages and timestamp authority

TimeSafe cannot reconstruct a historical vendor vintage that was never supplied. Publication timestamps and revision policies must come from authoritative metadata. Availability perturbation tests whether a pipeline depends on the declared unavailable cells; it does not certify that the declaration itself is true.

## External state

All trainable state must be created by the audited factory. Remote caches, databases, feature stores, global objects, and external checkpoints that are not rebuilt remain outside the guarantee.

## Coverage and computation

Exhaustive refit audits may be expensive. Sampled runs return `WARN` and disclose coverage. A negative control can establish sensitivity to a failure mode, but it cannot transform sampled evidence into exhaustive evidence.

## Output and ordering requirements

Transforms must return a uniquely indexed pandas Series/DataFrame with unique columns. Inputs must have a non-missing, non-decreasing temporal coordinate. Empty output or a perturbation that changes no actual cell is `INCONCLUSIVE`.

## Availability semantics

`reference="checkpoint"` and `reference="row"` answer different questions. Choosing the wrong reference can under-mask or over-mask data. Per-row publication/event metadata must be supplied when a constant rule is not valid.

## Calendar and event model

The built-in model uses a user-supplied session calendar and five generic events. Auctions, lunch breaks, night sessions, early closes, time zones, and asset-specific calendars require an adapter or a sufficiently detailed calendar. TimeSafe does not claim official exchange-calendar authority.

## Statistical and economic validity

Temporal legality does not imply absence of selection bias, a valid mechanism, realistic fills/costs/capacity, regime robustness, or production readiness. Use CPCV/DSR/PBO, experiment ledgers, mechanism-specific placebos, and cost-aware simulation separately.

## ACTR public experiment

The ACTR anonymous benchmark is block-shuffled and deidentified. It validates code paths and structural reproduction, not chronological investment performance. The data remains governed by its external research-only terms and is not redistributed by TimeSafe.
