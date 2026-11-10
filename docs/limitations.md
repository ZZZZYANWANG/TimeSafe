# Limitations

TimeSafe is an executable temporal-contract auditor, not a proof of strategy validity.

## Data-vintage limits

Prefix invariance cannot reconstruct historical vendor vintages. If revised fundamentals, corrected trades, or retrospective classifications have already overwritten the past, TimeSafe sees only the supplied version. Publication timestamps and revision policies still require authoritative metadata.

## External state

Transform-only auditing cannot detect a scaler, PCA, model, cache, or database table that was fitted on the full sample before the transform was called. Use `audit_refit_prefix_invariance` and ensure all trainable state is created by the supplied factory. Hidden external state remains outside the guarantee.

## Coverage

A sampled audit can miss a localized violation. TimeSafe therefore reports eligible checkpoints, checked checkpoints, coverage, and policy. Only `checkpoint_policy="all"` can yield an exhaustive `PASS`.

## Output requirements

Transforms must return a uniquely indexed pandas Series or DataFrame. Empty output is `INCONCLUSIVE`, not `PASS`. Opaque arrays, side-effect-only code, and remote systems require an adapter that exposes labeled historical outputs.

## Calendar and event model

The built-in contract uses a user-supplied ordered session calendar and five generic within-session events. Exchange auctions, lunch breaks, night sessions, time zones, early closes, and asset-specific calendars must be represented in the supplied data/calendar or handled by an adapter. TimeSafe does not claim official exchange-calendar authority.

## Statistical and economic validity

Temporal legality does not imply:

- absence of research overfitting;
- valid economic mechanism;
- realistic fills, costs, impact, or capacity;
- robustness across regimes;
- production readiness.

Use purged model selection and DSR/PBO tools, mechanism-specific placebos, and cost-aware simulation separately.

## Computational cost

Exhaustive prefix and refit audits may be expensive. Sampled mode is useful for development but returns `WARN`. CI can combine a fast sampled audit with a smaller exhaustive fixture.
