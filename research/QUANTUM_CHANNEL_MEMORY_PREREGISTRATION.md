# Quantum channel and memory factor round — preregistration

**Frozen before reading any outer results from the measurement-pointer round.**

## Purpose

The first measurement-pointer round treats completed OHLCV observations as a low-dimensional mixed-state representation and tests interactions built from fidelity, Bures displacement, purity and commutator coordinates. This second round moves one level deeper: it estimates a conditional state evolution, separates observed innovations from predicted evolution, and measures how activity-conditioned observation changes contraction, drift and memory.

Quantum mechanics is used as a mathematical design language. No claim is made that a market is literally a quantum system.

## Fixed state representation

For each product and completed bar, define a Bloch-ball coordinate from standardized intrabar body, opening gap and close location:

- `x = tanh(body_z / 2)`
- `y = tanh(gap_z / 2)`
- `z = close_location`

The vector is radially projected into the unit Bloch ball. Measurement strength is a bounded function of positive range and volume surprises. Every rolling coefficient excludes the query bar.

## Preregistered mechanisms

1. **Quantum-filter innovation** — estimate a strictly prior AR state predictor and Kalman-like gain; use the gain-weighted innovation along the return axis.
2. **Liouvillian Zeno/anti-Zeno response** — estimate how the persistence of the local state channel changes with measurement strength. Persistence under stronger measurement loads continuation; accelerated contraction loads reversal.
3. **CPTP-channel nonunitality** — estimate the affine drift left after state-dependent and measurement-dependent propagation. The drift is the local attractor component not explained by the current state.
4. **Nakajima–Zwanzig memory force** — estimate a five-lag strictly prior memory kernel on filter innovations and project the current delayed force.
5. **Memory-release trend switch** — use the magnitude of the memory force to condition trend continuation versus reversal.
6. **Fisher/fidelity susceptibility** — normalize state displacement by change in measurement strength and interact the resulting susceptibility surprise with the filter innovation.
7. **Open-system entropy production** — compare observed state entropy with the entropy of the predicted conditional state and condition the trend by the signed entropy change.
8. **Jump-diffusion backaction** — standardize the filter innovation against its strict prior noise scale and test nonlinear post-jump backaction.
9. **Non-Markovian information backflow** — compare fast and slow filtered states; a positive change in trace-distance proxy is treated as information backflow and oriented by the return-axis separation.
10. **Backflow trend switch** — use the direction and strength of backflow to switch the trend response.

## Frozen windows and horizons

- estimation windows: `63`, `126`, `252` bars;
- backflow filter pairs: `(10, 40)` and `(20, 80)` bars;
- prediction horizons: `1`, `5`, `10`, `20` bars/sessions.

## Parent-set boundary

Every raw state component used by a nonlinear candidate is also added to the strong parent set: filter innovation, persistence response, nonunital drift, memory force, Fisher susceptibility, entropy production, jump score and trace-distance backflow. A candidate must therefore add value beyond its components, rather than receiving credit for relabeling a raw statistic.

## Selection and validation

- Orientation is fixed only on 2008–2013 futures data.
- Replication gates are 2014–2015 and 2016–2018.
- A candidate must have positive direct product-level IC, positive IC against an out-of-sample strong-parent residual, and positive second-half scalar residual gain in both replication blocks.
- At most one span per mechanism and two mechanisms per horizon enter the outer test.
- Outer periods are 2019–2021, 2022–2024, 2025–2026-09-11, and the post-2026-09-11 segment.
- Outer tests include raw and volatility-normalized targets, Ridge and HistGradientBoosting readers, moving-block intervals, held-out products, family max-T correction, costed next-open accounts and a formula-frozen US-stock transfer panel.

## Admission boundary

A passing identity can only be admitted as a historical, default-off factor/model component. `independent_alpha=false` and `default_enabled=false` remain mandatory until the formula is replayed on the private point-in-time roll-safe panel with realistic contract multipliers, margin, exchange fees, slippage, impact and capacity, followed by genuinely untouched forward evidence.
