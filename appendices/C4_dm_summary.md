**表C4　Diebold–Mariano 检验结果汇总**

14 组预先设定的方法对，逐湖逐时距共 341 项检验，其中 333 项返回可用统计量，BH 校正在这 333 项上一次性施加，42 项在校正后仍显著。

| method_1 | method_2 | n_lakes | n_tested | n_significant | n_favouring_method_1 | n_favouring_method_2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SARIMAX_CCM_ancestors | Persistence | 4 | 16 | 2 | 2 | 0 |
| SARIMAX_CCM_ancestors | SARIMA | 4 | 16 | 2 | 1 | 1 |
| SARIMAX_CCM_ancestors | SARIMAX_CCM_direct | 4 | 16 | 1 | 1 | 0 |
| SARIMAX_CCM_ancestors | SARIMAX_Stepwise | 3 | 12 | 1 | 1 | 0 |
| SARIMAX_CCM_neighbor | SARIMAX_CCM_ancestors | 1 | 4 | 0 | 0 | 0 |
| XGBoost_CCM_ancestors | Persistence | 9 | 34 | 5 | 5 | 0 |
| XGBoost_CCM_ancestors | SARIMAX_CCM_ancestors | 4 | 16 | 4 | 2 | 2 |
| XGBoost_CCM_ancestors | XGBoost_AR_only | 9 | 35 | 6 | 5 | 1 |
| XGBoost_CCM_ancestors | XGBoost_CCM_direct | 9 | 35 | 1 | 0 | 1 |
| XGBoost_CCM_ancestors | XGBoost_Stepwise | 9 | 35 | 5 | 3 | 2 |
| XGBoost_CCM_ancestors | XGBoost_all_vars | 9 | 35 | 3 | 2 | 1 |
| XGBoost_CCM_direct | XGBoost_AR_only | 9 | 35 | 7 | 5 | 2 |
| XGBoost_CCM_neighbor | XGBoost_CCM_ancestors | 8 | 28 | 4 | 2 | 2 |
| XGBoost_all_vars | SARIMAX_all_vars | 4 | 16 | 1 | 1 | 0 |
