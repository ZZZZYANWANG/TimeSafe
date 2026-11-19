# Time-series non-normal transport research round22

This experiment reads the local spot-perpetual propagation operator as an asset-internal time-series state rather than a cross-sectional score. Direction and amplitude are separated, positions use a fixed no-trade region, and validation requires parent-residual TSIC, strong-model increment and 5bp net returns.

## Summary
```json
{
  "candidate_columns": 36,
  "confirmed": [],
  "decision": "eligible_for_trade_only_if_confirmed_nonempty",
  "development_tests": 216,
  "discovery_holdout_opened": 0,
  "discovery_holdout_passed": 0,
  "discovery_validation_passed": 0,
  "frozen_identities": 4,
  "raw_primitives": 12,
  "replication_holdout_opened": 0,
  "replication_validation_opened": 0,
  "replication_validation_passed": 0
}
```

## Frozen selection
| factor                           |   sign |   horizon |   development_ts_ic |   development_positive_asset_share |   development_min_year | development_years                                          |   development_annual_log_return_5bp |   development_sharpe_5bp |   development_score | family                        | oriented_factor                           |
|:---------------------------------|-------:|----------:|--------------------:|-----------------------------------:|-----------------------:|:-----------------------------------------------------------|------------------------------------:|-------------------------:|--------------------:|:------------------------------|:------------------------------------------|
| nonnormal_term_slope_ewm24       |     -1 |        72 |           0.0740439 |                           0.916667 |              0.0463395 | {"2021": 0.10181910703041486, "2022": 0.04633952100110022} |                            0.374035 |                  1.7606  |           0.0946588 | intrinsic_transient_direction | oriented_nonnormal_term_slope_ewm24       |
| source_pressure_consensus_ewm24  |     -1 |       120 |           0.0645355 |                           0.75     |              0.0515835 | {"2021": 0.0756045763978075, "2022": 0.05158351482873961}  |                            0.27172  |                  1.35278 |           0.0909837 | source_supported_transport    | oriented_source_pressure_consensus_ewm24  |
| impulse_balance_direction_ewm24  |      1 |       120 |           0.0683568 |                           1        |              0.0525043 | {"2021": 0.05250434388456599, "2022": 0.0900062951239876}  |                            0.291961 |                  1.44528 |           0.0944451 | pressure_supported_transport  | oriented_impulse_balance_direction_ewm24  |
| slow_mode_damped_nonnormal_ewm72 |      1 |        72 |           0.0889285 |                           0.916667 |              0.058333  | {"2021": 0.13020307077274526, "2022": 0.05833296077129025} |                            0.344554 |                  1.1582  |           0.111132  | modal_direction_amplitude     | oriented_slow_mode_damped_nonnormal_ewm72 |

## Discovery validation
| family                        | primitive_factor                 | factor                                    |   sign |   horizon |      ts_ic |   positive_asset_share |   asset_count |   bootstrap_ci_low |   bootstrap_ci_high |   bootstrap_p_nonpositive | yearly_ts_ic                                                 |   residual_ts_ic |   residual_positive_asset_share |   ridge_mse_gain |   max_parent_corr |   annual_log_return_5bp |   sharpe_5bp |   avg_turnover | portfolio_years                                               |   holm_p | gate   |
|:------------------------------|:---------------------------------|:------------------------------------------|-------:|----------:|-----------:|-----------------------:|--------------:|-------------------:|--------------------:|--------------------------:|:-------------------------------------------------------------|-----------------:|--------------------------------:|-----------------:|------------------:|------------------------:|-------------:|---------------:|:--------------------------------------------------------------|---------:|:-------|
| intrinsic_transient_direction | nonnormal_term_slope_ewm24       | oriented_nonnormal_term_slope_ewm24       |     -1 |        72 | 0.00915862 |               0.583333 |            12 |         -0.0147581 |           0.0333525 |                    0.2369 | {"2023": 0.005541034274995238, "2024": 0.013188663928368047} |       -0.0228093 |                        0.333333 |      -0.00910062 |          0.539575 |               0.0221343 |     0.127093 |       0.321015 | {"2023": -0.005078659685577195, "2024": 0.04934734005480526}  |   0.4738 | False  |
| source_supported_transport    | source_pressure_consensus_ewm24  | oriented_source_pressure_consensus_ewm24  |     -1 |       120 | 0.00631848 |               0.583333 |            12 |         -0.0393183 |           0.0465835 |                    0.3746 | {"2023": 0.0614215304107148, "2024": -0.03327728413344853}   |        0.028763  |                        0.666667 |      -0.0142804  |          0.105772 |               0.0297847 |     0.25081  |       0.276104 | {"2023": 0.06585501819005918, "2024": -0.0062855327478893944} |   0.4738 | False  |
| pressure_supported_transport  | impulse_balance_direction_ewm24  | oriented_impulse_balance_direction_ewm24  |      1 |       120 | 0.0562361  |               0.75     |            12 |          0.0120462 |           0.0974282 |                    0.0061 | {"2023": 0.10089901136657019, "2024": 0.02468992965568308}   |        0.0770314 |                        0.916667 |      -0.0168545  |          0.127865 |               0.0943365 |     0.73702  |       0.283731 | {"2023": 0.14383995367126692, "2024": 0.04483297855814611}    |   0.0183 | False  |
| modal_direction_amplitude     | slow_mode_damped_nonnormal_ewm72 | oriented_slow_mode_damped_nonnormal_ewm72 |      1 |        72 | 0.0582428  |               0.916667 |            12 |          0.0340638 |           0.0807452 |                    0      | {"2023": 0.03469080687284315, "2024": 0.07813238341027223}   |        0.0215747 |                        0.666667 |      -0.00432343 |          0.640458 |               0.16118   |     0.801149 |       0.223318 | {"2023": -0.0059560000243742365, "2024": 0.3283159118046227}  |   0      | False  |

## Discovery holdout
Holdout remained closed.

## Replication
Replication remained closed.