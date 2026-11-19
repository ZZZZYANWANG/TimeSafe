# Regime-dependent constitutive operator research round24

Round24 tests whether the sign and strength of previously observed non-normal transport depends on a strictly-prior propagation regime rather than on a fixed readout map.

## Summary
```json
{
  "confirmed": [],
  "decision": "eligible_for_trade_only_if_confirmed_nonempty",
  "development_crossfit_rows": 4,
  "discovery_holdout_passed": 0,
  "discovery_validation_passed": 0,
  "frozen_models": 2,
  "model_terms": 9,
  "replication_holdout_passed": 0,
  "replication_validation_passed": 0
}
```

## Frozen coefficients
| factor                           | parent                           |   horizon | term                        |   coefficient |
|:---------------------------------|:---------------------------------|----------:|:----------------------------|--------------:|
| regime_constitutive_impulse_120h | impulse_balance_direction_ewm24  |       120 | parent                      |    0.116572   |
| regime_constitutive_impulse_120h | impulse_balance_direction_ewm24  |       120 | parent_x_lead               |   -0.0199958  |
| regime_constitutive_impulse_120h | impulse_balance_direction_ewm24  |       120 | parent_x_source             |    0.0417351  |
| regime_constitutive_impulse_120h | impulse_balance_direction_ewm24  |       120 | parent_x_impedance          |    0.0345655  |
| regime_constitutive_impulse_120h | impulse_balance_direction_ewm24  |       120 | parent_x_dominance          |   -0.0669865  |
| regime_constitutive_impulse_120h | impulse_balance_direction_ewm24  |       120 | parent_x_lead_x_source      |   -0.0256803  |
| regime_constitutive_impulse_120h | impulse_balance_direction_ewm24  |       120 | parent_x_lead_x_impedance   |    0.0447616  |
| regime_constitutive_impulse_120h | impulse_balance_direction_ewm24  |       120 | parent_x_source_x_impedance |    0.0526104  |
| regime_constitutive_impulse_120h | impulse_balance_direction_ewm24  |       120 | parent_x_lead_x_dominance   |    0.0153176  |
| regime_constitutive_slowmode_72h | slow_mode_damped_nonnormal_ewm72 |        72 | parent                      |    0.0590941  |
| regime_constitutive_slowmode_72h | slow_mode_damped_nonnormal_ewm72 |        72 | parent_x_lead               |    0.0234768  |
| regime_constitutive_slowmode_72h | slow_mode_damped_nonnormal_ewm72 |        72 | parent_x_source             |   -0.029278   |
| regime_constitutive_slowmode_72h | slow_mode_damped_nonnormal_ewm72 |        72 | parent_x_impedance          |    0.0366564  |
| regime_constitutive_slowmode_72h | slow_mode_damped_nonnormal_ewm72 |        72 | parent_x_dominance          |   -0.0742375  |
| regime_constitutive_slowmode_72h | slow_mode_damped_nonnormal_ewm72 |        72 | parent_x_lead_x_source      |    0.025749   |
| regime_constitutive_slowmode_72h | slow_mode_damped_nonnormal_ewm72 |        72 | parent_x_lead_x_impedance   |    0.0240169  |
| regime_constitutive_slowmode_72h | slow_mode_damped_nonnormal_ewm72 |        72 | parent_x_source_x_impedance |    0.00813431 |
| regime_constitutive_slowmode_72h | slow_mode_damped_nonnormal_ewm72 |        72 | parent_x_lead_x_dominance   |    0.00499502 |

## Development cross-fit
| factor                           |   train_year |   test_year | status                                 | reason                                                                                          |       ts_ic |   positive_asset_share |   annual_log_return_5bp |   sharpe_5bp |
|:---------------------------------|-------------:|------------:|:---------------------------------------|:------------------------------------------------------------------------------------------------|------------:|-----------------------:|------------------------:|-------------:|
| regime_constitutive_impulse_120h |         2021 |        2022 | insufficient_nonoverlap_for_diagnostic | insufficient non-overlapping development observations for impulse_balance_direction_ewm24: 106  | nan         |                    nan |             nan         |  nan         |
| regime_constitutive_impulse_120h |         2022 |        2021 | estimated                              |                                                                                                 |  -0.0589622 |                      0 |               0.0232924 |    0.0690721 |
| regime_constitutive_slowmode_72h |         2021 |        2022 | insufficient_nonoverlap_for_diagnostic | insufficient non-overlapping development observations for slow_mode_damped_nonnormal_ewm72: 169 | nan         |                    nan |             nan         |  nan         |
| regime_constitutive_slowmode_72h |         2022 |        2021 | estimated                              |                                                                                                 |   0.0674829 |                      1 |               0.123113  |    0.300322  |

## Discovery validation
| stage                | family                       | parent                           | factor                           |   horizon |     ts_ic |   positive_asset_share |   asset_count |   bootstrap_ci_low |   bootstrap_ci_high |   bootstrap_p_nonpositive | yearly_ts_ic                                                |   residual_ts_ic |   residual_positive_asset_share |   ridge_mse_gain |   max_parent_corr |   annual_log_return_5bp |   sharpe_5bp |   avg_turnover | portfolio_years                                               |   holm_p | gate   |
|:---------------------|:-----------------------------|:---------------------------------|:---------------------------------|----------:|----------:|-----------------------:|--------------:|-------------------:|--------------------:|--------------------------:|:------------------------------------------------------------|-----------------:|--------------------------------:|-----------------:|------------------:|------------------------:|-------------:|---------------:|:--------------------------------------------------------------|---------:|:-------|
| discovery_validation | regime_constitutive_impulse  | impulse_balance_direction_ewm24  | regime_constitutive_impulse_120h |       120 | 0.0642177 |               0.75     |            12 |         0.0219436  |            0.106623 |                    0.0013 | {"2023": 0.07301225917260516, "2024": 0.05669192450854493}  |        0.0418987 |                        0.583333 |      -0.00444364 |          0.690893 |             -0.00516333 |   -0.0322447 |       0.310002 | {"2023": 0.010818067852278746, "2024": -0.021366696702943545} |   0.0026 | False  |
| discovery_validation | regime_constitutive_slowmode | slow_mode_damped_nonnormal_ewm72 | regime_constitutive_slowmode_72h |        72 | 0.0308688 |               0.666667 |            12 |         0.00642392 |            0.057186 |                    0.0073 | {"2023": 0.026001820990132058, "2024": 0.03517367617743693} |        0.0134216 |                        0.583333 |      -0.00228389 |          0.298143 |             -0.0494782  |   -0.232353  |       0.401393 | {"2023": -0.11332728671375118, "2024": 0.014898578084130117}  |   0.0073 | False  |

## Discovery historical holdout
| stage             | family                       | parent                           | factor                           |   horizon |      ts_ic |   positive_asset_share |   asset_count |   bootstrap_ci_low |   bootstrap_ci_high |   bootstrap_p_nonpositive | yearly_ts_ic                                                 |   residual_ts_ic |   residual_positive_asset_share |   ridge_mse_gain |   max_parent_corr |   annual_log_return_5bp |   sharpe_5bp |   avg_turnover | portfolio_years                                             |   holm_p | gate   |
|:------------------|:-----------------------------|:---------------------------------|:---------------------------------|----------:|-----------:|-----------------------:|--------------:|-------------------:|--------------------:|--------------------------:|:-------------------------------------------------------------|-----------------:|--------------------------------:|-----------------:|------------------:|------------------------:|-------------:|---------------:|:------------------------------------------------------------|---------:|:-------|
| discovery_holdout | regime_constitutive_impulse  | impulse_balance_direction_ewm24  | regime_constitutive_impulse_120h |       120 | 0.00108448 |               0.636364 |            11 |        -0.055969   |           0.0540666 |                    0.4712 | {"2025": 0.03786549336999039, "2026": -0.05321747241107482}  |      -0.0319632  |                        0.181818 |      0.000331454 |          0.691875 |              0.00207756 |    0.0130047 |       0.297337 | {"2025": 0.04737361288821183, "2026": -0.05915599450325561} |   0.4712 | False  |
| discovery_holdout | regime_constitutive_slowmode | slow_mode_damped_nonnormal_ewm72 | regime_constitutive_slowmode_72h |        72 | 0.0278817  |               0.666667 |            12 |        -0.00923871 |           0.0682462 |                    0.0771 | {"2025": 0.04314425175900325, "2026": -0.010852785546164635} |       0.00441503 |                        0.583333 |     -0.00135276  |          0.295973 |              0.0811895  |    0.396781  |       0.36656  | {"2025": 0.16890952475372442, "2026": -0.03674520002910423} |   0.1542 | False  |

## Disjoint-universe validation
| stage                  | family                       | parent                           | factor                           |   horizon |     ts_ic |   positive_asset_share |   asset_count |   bootstrap_ci_low |   bootstrap_ci_high |   bootstrap_p_nonpositive | yearly_ts_ic                                                 |   residual_ts_ic |   residual_positive_asset_share |   ridge_mse_gain |   max_parent_corr |   annual_log_return_5bp |   sharpe_5bp |   avg_turnover | portfolio_years                                             |   holm_p | gate   |
|:-----------------------|:-----------------------------|:---------------------------------|:---------------------------------|----------:|----------:|-----------------------:|--------------:|-------------------:|--------------------:|--------------------------:|:-------------------------------------------------------------|-----------------:|--------------------------------:|-----------------:|------------------:|------------------------:|-------------:|---------------:|:------------------------------------------------------------|---------:|:-------|
| replication_validation | regime_constitutive_impulse  | impulse_balance_direction_ewm24  | regime_constitutive_impulse_120h |       120 | 0.0506478 |               0.666667 |            12 |         0.00202293 |           0.0993832 |                    0.0211 | {"2023": 0.03843775272621323, "2024": 0.054723480189828776}  |       0.00636495 |                        0.5      |     -0.0018069   |          0.711344 |             -0.0564501  |   -0.265939  |       0.307821 | {"2023": -0.20603546817348053, "2024": 0.09521291040622784} |   0.0422 | False  |
| replication_validation | regime_constitutive_slowmode | slow_mode_damped_nonnormal_ewm72 | regime_constitutive_slowmode_72h |        72 | 0.0309584 |               0.75     |            12 |        -0.00499735 |           0.0651897 |                    0.046  | {"2023": 0.0017632694949617361, "2024": 0.05276516394091623} |       0.00876393 |                        0.666667 |     -0.000958378 |          0.311656 |             -0.00743675 |   -0.0243623 |       0.408954 | {"2023": -0.26830329582046764, "2024": 0.2555857226871923}  |   0.046  | False  |

## Disjoint-universe historical holdout
| stage               | family                       | parent                           | factor                           |   horizon |     ts_ic |   positive_asset_share |   asset_count |   bootstrap_ci_low |   bootstrap_ci_high |   bootstrap_p_nonpositive | yearly_ts_ic                                                |   residual_ts_ic |   residual_positive_asset_share |   ridge_mse_gain |   max_parent_corr |   annual_log_return_5bp |   sharpe_5bp |   avg_turnover | portfolio_years                                              |   holm_p | gate   |
|:--------------------|:-----------------------------|:---------------------------------|:---------------------------------|----------:|----------:|-----------------------:|--------------:|-------------------:|--------------------:|--------------------------:|:------------------------------------------------------------|-----------------:|--------------------------------:|-----------------:|------------------:|------------------------:|-------------:|---------------:|:-------------------------------------------------------------|---------:|:-------|
| replication_holdout | regime_constitutive_impulse  | impulse_balance_direction_ewm24  | regime_constitutive_impulse_120h |       120 | 0.0490821 |               0.833333 |            12 |         0.00409163 |           0.0870393 |                    0.0173 | {"2025": 0.08263097286732833, "2026": 0.012624580150837202} |        0.0389558 |                        0.666667 |     -0.000377952 |          0.728701 |                0.250898 |     1.25355  |       0.271016 | {"2025": 0.4598431912493044, "2026": -0.03156402419073789}   |    0.026 | False  |
| replication_holdout | regime_constitutive_slowmode | slow_mode_damped_nonnormal_ewm72 | regime_constitutive_slowmode_72h |        72 | 0.0292792 |               0.666667 |            12 |         0.00300939 |           0.0570333 |                    0.013  | {"2025": 0.01001957797156828, "2026": 0.055414677328332135} |        0.0176119 |                        0.666667 |     -0.000397172 |          0.353507 |                0.181828 |     0.675349 |       0.334118 | {"2025": 0.33270854612349254, "2026": -0.021023191883423713} |    0.026 | False  |