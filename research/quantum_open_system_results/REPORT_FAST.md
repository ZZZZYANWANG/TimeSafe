# Quantum open-system factor search — strict fast protocol

Quantum theory is used as a representation and operator-design language. The study does not assert that markets are literal quantum systems.

- futures proxy assets: **35**
- futures rows: **171525**
- stock transfer assets: **30**
- candidate declarations: **144**
- selected finalists: **3**
- historical default-off admissions: **0**
- production independent Alpha admissions: **0**

## Selection contract

Orientation is fixed on 2008–2013. A candidate must then keep positive direct product-level IC, positive IC against an out-of-sample strong-parent residual, and positive second-half scalar residual gain in both 2014–2015 and 2016–2018. Only one span per mechanism and at most two mechanisms per horizon enter the outer test.

## `purity_backaction_trend_s40` — 20 sessions

- mechanism: `purity_backaction`
- development selection: `diagnostic_only_no_full_development_pass`
- stock domain transfer VAL/HIST positive: **False**
- historical default-off admitted: **False**
- production independent Alpha: **False**
- family max-T p: `1.0`

| Period | Futures TSIC | Futures CSIC | Positive products |
|---|---:|---:|---:|
| DEV_A | 0.011172 | 0.018743 | 0.559 |
| DEV_B1 | 0.066238 | 0.043401 | 0.618 |
| DEV_B2 | 0.055532 | 0.026456 | 0.657 |
| VAL | 0.002585 | 0.002592 | 0.543 |
| HIST | 0.017451 | -0.037699 | 0.571 |
| LATE | 0.064557 | 0.011381 | 0.743 |
| FRESH | nan | 0.052194 | nan |

| Target | Model | Period | MSE gain % | Block low | Block high |
|---|---|---|---:|---:|---:|
| raw | ridge | VAL | 0.001125 | -0.006997 | 0.008002 |
| raw | ridge | HIST | 0.015868 | -0.010320 | 0.044390 |
| raw | ridge | LATE | 0.005576 | -0.012244 | 0.021497 |
| raw | ridge | FRESH | 0.009577 | 0.002110 | 0.016977 |
| raw | hgb | VAL | -0.027822 | -0.087034 | 0.026049 |
| raw | hgb | HIST | 0.004162 | -0.055475 | 0.061530 |
| raw | hgb | LATE | 0.122465 | 0.048174 | 0.187770 |
| raw | hgb | FRESH | -0.015684 | -0.150215 | 0.128490 |
| norm | ridge | VAL | -0.000715 | -0.015367 | 0.016397 |
| norm | ridge | HIST | 0.009789 | -0.002461 | 0.021685 |
| norm | ridge | LATE | -0.022573 | -0.056100 | 0.009956 |
| norm | ridge | FRESH | -0.029510 | -0.072310 | 0.015544 |
| norm | hgb | VAL | 0.042649 | -0.041172 | 0.138568 |
| norm | hgb | HIST | -0.212868 | -0.318961 | -0.105656 |
| norm | hgb | LATE | 0.004467 | -0.067823 | 0.058816 |
| norm | hgb | FRESH | 0.008077 | -0.111439 | 0.122396 |

| Stock period | TSIC | Ridge normalized MSE gain % |
|---|---:|---:|
| VAL | 0.071858 | -0.001711 |
| HIST | 0.039049 | 0.004361 |
| LATE | 0.050162 | -0.016804 |
| FRESH | nan | -0.036523 |

## `zeno_pointer_persistence_s40` — 20 sessions

- mechanism: `zeno_persistence`
- development selection: `diagnostic_only_no_full_development_pass`
- stock domain transfer VAL/HIST positive: **True**
- historical default-off admitted: **False**
- production independent Alpha: **False**
- family max-T p: `1.0`

| Period | Futures TSIC | Futures CSIC | Positive products |
|---|---:|---:|---:|
| DEV_A | 0.017315 | 0.019032 | 0.559 |
| DEV_B1 | 0.078952 | 0.050418 | 0.676 |
| DEV_B2 | 0.073874 | 0.026982 | 0.657 |
| VAL | 0.002039 | -0.009919 | 0.486 |
| HIST | 0.021573 | -0.049628 | 0.486 |
| LATE | 0.071193 | 0.003750 | 0.714 |
| FRESH | nan | 0.007811 | nan |

| Target | Model | Period | MSE gain % | Block low | Block high |
|---|---|---|---:|---:|---:|
| raw | ridge | VAL | 0.010731 | -0.016271 | 0.039815 |
| raw | ridge | HIST | 0.021710 | -0.008709 | 0.051398 |
| raw | ridge | LATE | 0.037537 | -0.006429 | 0.084461 |
| raw | ridge | FRESH | 0.081657 | 0.024922 | 0.136010 |
| raw | hgb | VAL | 0.012365 | -0.059690 | 0.091254 |
| raw | hgb | HIST | 0.036293 | -0.027153 | 0.095701 |
| raw | hgb | LATE | 0.009830 | -0.062893 | 0.084407 |
| raw | hgb | FRESH | 0.080201 | -0.075236 | 0.227524 |
| norm | ridge | VAL | -0.004021 | -0.019739 | 0.011740 |
| norm | ridge | HIST | 0.024688 | -0.000057 | 0.049257 |
| norm | ridge | LATE | -0.007795 | -0.037232 | 0.020503 |
| norm | ridge | FRESH | 0.103688 | 0.058078 | 0.150515 |
| norm | hgb | VAL | 0.021288 | -0.072570 | 0.114474 |
| norm | hgb | HIST | -0.147007 | -0.245695 | -0.060240 |
| norm | hgb | LATE | -0.028490 | -0.132487 | 0.075277 |
| norm | hgb | FRESH | -0.035877 | -0.197451 | 0.127596 |

| Stock period | TSIC | Ridge normalized MSE gain % |
|---|---:|---:|
| VAL | 0.080553 | 0.019271 |
| HIST | 0.044492 | 0.006000 |
| LATE | 0.034866 | 0.000940 |
| FRESH | nan | 0.063022 |

## `pointer_leakage_trend_reversal_s10` — 10 sessions

- mechanism: `pointer_state_leakage`
- development selection: `diagnostic_only_no_full_development_pass`
- stock domain transfer VAL/HIST positive: **False**
- historical default-off admitted: **False**
- production independent Alpha: **False**
- family max-T p: `1.0`

| Period | Futures TSIC | Futures CSIC | Positive products |
|---|---:|---:|---:|
| DEV_A | 0.013654 | 0.011934 | 0.559 |
| DEV_B1 | 0.051647 | 0.034584 | 0.647 |
| DEV_B2 | 0.042891 | 0.022444 | 0.657 |
| VAL | -0.003518 | -0.000492 | 0.543 |
| HIST | 0.017003 | -0.022395 | 0.629 |
| LATE | 0.034213 | -0.005313 | 0.629 |
| FRESH | 0.203973 | 0.018445 | 0.771 |

| Target | Model | Period | MSE gain % | Block low | Block high |
|---|---|---|---:|---:|---:|
| raw | ridge | VAL | 0.004172 | -0.018155 | 0.029138 |
| raw | ridge | HIST | 0.002593 | -0.016734 | 0.021132 |
| raw | ridge | LATE | -0.012218 | -0.057737 | 0.022426 |
| raw | ridge | FRESH | -0.008753 | -0.036152 | 0.018649 |
| raw | hgb | VAL | -0.017251 | -0.089002 | 0.042035 |
| raw | hgb | HIST | -0.012074 | -0.080598 | 0.047198 |
| raw | hgb | LATE | 0.001368 | -0.068363 | 0.063304 |
| raw | hgb | FRESH | 0.169803 | 0.031325 | 0.317155 |
| norm | ridge | VAL | -0.000392 | -0.006142 | 0.005224 |
| norm | ridge | HIST | 0.003099 | -0.004899 | 0.010329 |
| norm | ridge | LATE | 0.001997 | -0.002035 | 0.006448 |
| norm | ridge | FRESH | -0.013645 | -0.026422 | -0.000781 |
| norm | hgb | VAL | -0.061617 | -0.140424 | 0.016596 |
| norm | hgb | HIST | -0.004589 | -0.067728 | 0.057270 |
| norm | hgb | LATE | -0.077316 | -0.185188 | 0.025901 |
| norm | hgb | FRESH | -0.288668 | -0.607584 | 0.021188 |

| Stock period | TSIC | Ridge normalized MSE gain % |
|---|---:|---:|
| VAL | 0.041939 | -0.020766 |
| HIST | 0.019877 | -0.001840 |
| LATE | 0.040761 | -0.008284 |
| FRESH | 0.149081 | -0.037747 |

## Boundaries

The futures data are public continuous-contract proxies and can embed vendor roll conventions. The stock transfer panel is a present-day liquid-name universe and therefore survivorship-limited. Costed accounts are fixed-capital diagnostics without contract multipliers, margin, exchange fees, queueing, impact or capacity. Any object admitted here must remain default-off until replayed on the private point-in-time roll-safe production panel.

The post-2026-09-11 interval is displayed separately and is too short to erase the historical adaptive-search debt. Production independent Alpha remains false unless a genuinely untouched interval, canonical data and execution-aware account all pass.
