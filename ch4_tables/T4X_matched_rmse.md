**Table 4.X.** Mean RMSE (m) of rolling-origin forecasts. Rows are comparable within a block but not between blocks; persistence is recomputed on each block's lakes.

| Configuration | h = 1 | h = 3 | h = 6 | h = 12 |
| --- | ---: | ---: | ---: | ---: |
| **SARIMA(X)** (matched on 3 lakes) | | | | |
| No exogenous | 0.070 | 0.109 | 0.130 | 0.140 |
| All vars | 0.067 | 0.104 | 0.121 | 0.128 |
| Stepwise | 0.068 | 0.105 | 0.124 | 0.127 |
| CCM direct | 0.069 | 0.113 | 0.138 | 0.144 |
| CCM ancestors | 0.068 | 0.110 | 0.133 | 0.141 |
| Persistence | 0.075 | 0.124 | 0.154 | 0.152 |
| **XGBoost** (matched on 9 lakes) | | | | |
| AR-only | 0.195 | 0.366 | 0.490 | 0.330 |
| All vars | 0.198 | 0.308 | 0.357 | 0.257 |
| Stepwise | 0.194 | 0.293 | 0.338 | 0.211 |
| CCM direct | 0.192 | 0.310 | 0.365 | 0.248 |
| CCM ancestors | 0.190 | 0.303 | 0.359 | 0.260 |
| Persistence | 0.180 | 0.412 | 0.607 | 0.620 |

CCM neighbour is unmatched (valid lakes change with horizon): SARIMAX 0.082 (1), 0.110 (1), 0.117 (1), 0.093 (1); XGBoost 0.222 (8), 0.332 (8), 0.381 (8), 0.324 (5).
