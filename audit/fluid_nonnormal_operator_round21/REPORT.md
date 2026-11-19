# State-dependent non-Markovian fluid closure research round20

This experiment tests whether spot-perpetual propagation direction, source-memory location and impedance matching explain the sign instability of earlier fixed fluid closures.

## Summary
```json
{
  "candidate_columns": 68,
  "confirmed": [],
  "decision": "eligible_for_trade_only_if_confirmed_nonempty",
  "development_tests": 952,
  "discovery_holdout_opened": 0,
  "discovery_promoted": [],
  "discovery_validation_passed": 0,
  "frozen_identities": 4,
  "raw_primitives": 17,
  "replication_holdout_opened": 0,
  "replication_validation_passed": 0
}
```

## Frozen selection
| factor                             |   sign |   horizon |   development_cs |   development_ts |   development_min_year | development_years                                           |   development_score | family                  |   multi_horizon_support |   short_support |   slow_support |   robust_multi_horizon_score | oriented_factor                             |
|:-----------------------------------|-------:|----------:|-----------------:|-----------------:|-----------------------:|:------------------------------------------------------------|--------------------:|:------------------------|------------------------:|----------------:|---------------:|-----------------------------:|:--------------------------------------------|
| nonnormal_shear_h24_ewm6           |      1 |        72 |        0.0299956 |        0.0918295 |              0.0266116 | {"2021": 0.03385399647081865, "2022": 0.02661163972639352}  |           0.0447933 | nonnormal_transient     |                       5 |               5 |              1 |                    0.0319034 | oriented_nonnormal_shear_h24_ewm6           |
| operator_cumulative_shear_h6_ewm72 |      1 |        72 |        0.043531  |        0.0931483 |              0.0345692 | {"2021": 0.034569204380524896, "2022": 0.05131739844854553} |           0.0569433 | operator_forecast       |                       7 |               5 |              3 |                    0.0374112 | oriented_operator_cumulative_shear_h6_ewm72 |
| slow_mode_shear_h72_ewm6           |      1 |       120 |        0.0593164 |        0.0779999 |              0.0567168 | {"2021": 0.06228146853146809, "2022": 0.056716799182552194} |           0.0823117 | slow_mode_relaxation    |                       4 |               2 |              3 |                    0.0119167 | oriented_slow_mode_shear_h72_ewm6           |
| pressure_impulse_shear_h72_ewm6    |      1 |        72 |        0.0383511 |        0.0942816 |              0.0344425 | {"2021": 0.04280765963008925, "2022": 0.03444247009820745}  |           0.055376  | source_impulse_response |                       5 |               5 |              1 |                    0.0314162 | oriented_pressure_impulse_shear_h72_ewm6    |

## Discovery validation
| factor                                      |   horizon |   train_end |   start |   end |    cs_rankic |      ts_ic |   ts_positive_share |   ts_products |   bootstrap_low |   bootstrap_high |   bootstrap_p_nonpositive |   ridge_mse_gain |   ridge_base_rankic |   ridge_full_rankic |   residual_cs_rankic |   residual_ts_ic |   max_parent_corr | max_parent_feature   |   annual_log_return_2bp |   annual_log_return_5bp |   sharpe_5bp |   avg_turnover | yearly_rankic                                                 | yearly_5bp                                                  |   n_train |   n_test |   parent_count | family                  | primitive_factor                   |   sign |   development_score |   multi_horizon_support |   robust_multi_horizon_score |   holm_p | validation_gate   |
|:--------------------------------------------|----------:|------------:|--------:|------:|-------------:|-----------:|--------------------:|--------------:|----------------:|-----------------:|--------------------------:|-----------------:|--------------------:|--------------------:|---------------------:|-----------------:|------------------:|:---------------------|------------------------:|------------------------:|-------------:|---------------:|:--------------------------------------------------------------|:------------------------------------------------------------|----------:|---------:|---------------:|:------------------------|:-----------------------------------|-------:|--------------------:|------------------------:|-----------------------------:|---------:|:------------------|
| oriented_nonnormal_shear_h24_ewm6           |        72 |        2022 |    2023 |  2024 | -0.00115849  |  0.0502474 |            0.666667 |            12 |      -0.0405497 |        0.0385527 |                    0.5271 |      -0.00907681 |         -0.00239036 |        -0.0009005   |          9.83754e-05 |        0.0225547 |          0.265948 | basis_z              |               0.0147628 |              -0.0250804 |    -0.131477 |       1.09085  | {"2023": 0.020553594050940517, "2024": -0.02287057205089976}  | {"2023": 0.08890294834141517, "2024": -0.1390637271811743}  |      2742 |     2928 |             36 | nonnormal_transient     | nonnormal_shear_h24_ewm6           |      1 |           0.0447933 |                       5 |                    0.0319034 |        1 | False             |
| oriented_operator_cumulative_shear_h6_ewm72 |        72 |        2022 |    2023 |  2024 |  0.00229435  |  0.0463124 |            0.916667 |            12 |      -0.0382028 |        0.0426195 |                    0.4577 |      -0.0112475  |         -0.00218974 |         0.00380022  |          0.00778108  |        0.0186082 |          0.193207 | account_ratio_z      |              -0.0238935 |              -0.0533455 |    -0.296677 |       0.806352 | {"2023": 0.006881482193135274, "2024": -0.002292789178035064} | {"2023": 0.09747561983307522, "2024": -0.20416670318216468} |      2730 |     2928 |             36 | operator_forecast       | operator_cumulative_shear_h6_ewm72 |      1 |           0.0569433 |                       7 |                    0.0374112 |        1 | False             |
| oriented_slow_mode_shear_h72_ewm6           |       120 |        2022 |    2023 |  2024 | -0.00948367  | -0.0267009 |            0.5      |            12 |      -0.0552747 |        0.0362595 |                    0.6614 |       0.00338163 |         -0.0409043  |        -0.045263    |         -0.00124533  |       -0.0346205 |          0.176267 | account_ratio_z      |              -0.169343  |              -0.195912  |    -1.07656  |       1.21233  | {"2023": 0.012261710891847789, "2024": -0.03122904492767485}  | {"2023": 0.021006158427132722, "2024": -0.4128293038589676} |      1641 |     1752 |             36 | slow_mode_relaxation    | slow_mode_shear_h72_ewm6           |      1 |           0.0823117 |                       4 |                    0.0119167 |        1 | False             |
| oriented_pressure_impulse_shear_h72_ewm6    |        72 |        2022 |    2023 |  2024 | -1.06878e-05 |  0.0417679 |            0.75     |            12 |      -0.0415675 |        0.041669  |                    0.5072 |      -0.00992908 |         -0.00239036 |         0.000762425 |          0.00184944  |        0.0131883 |          0.260989 | basis_z              |              -0.0883351 |              -0.126457  |    -0.661682 |       1.04372  | {"2023": 0.019925890267395142, "2024": -0.01994726584890505}  | {"2023": 0.01563518326155284, "2024": -0.26854880018894123} |      2742 |     2928 |             36 | source_impulse_response | pressure_impulse_shear_h72_ewm6    |      1 |           0.055376  |                       5 |                    0.0314162 |        1 | False             |

## Discovery holdout
Holdout remained closed.

## Replication validation
Replication remained closed.

## Confirmed
```json
[]
```