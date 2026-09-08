**Table 4.X.** Mean RMSE (m) of rolling-origin forecasts by configuration and forecast horizon, with the number of lakes contributing to each mean in parentheses. Means are taken over the lakes for which that configuration produced valid rolling forecasts; because this number differs between configurations, pooled means are not directly comparable across rows and no ranking is implied. Of the 130 lake-by-configuration combinations, 8 were not applicable, 122 were fitted and 93 produced valid rolling forecasts; the shortfall is concentrated in the SARIMAX configurations, for which exogenous predictors were unavailable over the full evaluation period in some lakes. Figure 4.4 shows the fully matched subset.

| Configuration | h = 1 | h = 3 | h = 6 | h = 12 |
| --- | ---: | ---: | ---: | ---: |
| Persistence | 0.167 (10) | 0.376 (10) | 0.553 (10) | 0.566 (10) |
| SARIMA (no exogenous) | 0.161 (10) | 0.333 (10) | 0.449 (10) | 0.383 (10) |
| SARIMAX + all vars | 0.061 (4) | 0.092 (4) | 0.105 (4) | 0.110 (4) |
| SARIMAX + stepwise | 0.062 (4) | 0.093 (4) | 0.107 (4) | 0.110 (4) |
| SARIMAX + CCM direct | 0.070 (4) | 0.130 (4) | 0.168 (4) | 0.161 (4) |
| SARIMAX + CCM ancestors | 0.070 (4) | 0.128 (4) | 0.164 (4) | 0.159 (4) |
| SARIMAX + CCM neighbour | 0.082 (1) | 0.110 (1) | 0.117 (1) | 0.093 (1) |
| XGBoost, AR-only | 0.180 (10) | 0.334 (10) | 0.446 (10) | 0.304 (10) |
| XGBoost + all vars | 0.182 (10) | 0.282 (10) | 0.326 (10) | 0.236 (10) |
| XGBoost + stepwise | 0.179 (10) | 0.269 (10) | 0.309 (10) | 0.196 (10) |
| XGBoost + CCM direct | 0.192 (9) | 0.310 (9) | 0.365 (9) | 0.248 (9) |
| XGBoost + CCM ancestors | 0.190 (9) | 0.303 (9) | 0.359 (9) | 0.260 (9) |
| XGBoost + CCM neighbour | 0.222 (8) | 0.332 (8) | 0.381 (8) | 0.324 (5) |
