**表B3　通过筛选的 26 条湖间关系**

检验族为 45 个湖泊对的双向共 90 条候选边，其中 26 条同时满足 BH-FDR 与收敛诊断。tier 为写作阶段引入的事后描述性水文距离分组。

| cause_lake | effect_lake | obs_rho | obs_lag | obs_n | p_fdr | kendall_tau | lag_resolution | causal_evidence | waterway_connected | tier |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| Split | Sipiwesk | 0.943 | 1 | 225 | 0.0095 | 0.889 | resolved | 是 | 是 | 1_direct |
| Sipiwesk | Split | 0.940 | 2 | 277 | 0.0095 | 0.667 | resolved | 是 | 是 | 1_direct |
| Okanagan | Kalamalka | 0.721 | 2 | 319 | 0.0095 | 0.944 | resolved | 是 | 是 | 1_direct |
| Sipiwesk | Playgreen | 0.678 | 0 | 267 | 0.0095 | 0.944 | unresolved_contemporaneous | 是 | 是 | 1_direct |
| Kalamalka | Okanagan | 0.629 | 1 | 330 | 0.0095 | 0.722 | resolved | 是 | 是 | 1_direct |
| Rainy | Lake of the Woods | 0.589 | 1 | 333 | 0.0095 | 0.944 | resolved | 是 | 是 | 1_direct |
| Lake of the Woods | Rainy | 0.474 | 0 | 333 | 0.0095 | 0.722 | unresolved_contemporaneous | 是 | 是 | 1_direct |
| Skaha | Vaseux | 0.413 | 4 | 321 | 0.0095 | 1.000 | resolved | 是 | 是 | 1_direct |
| Playgreen | Split | 0.779 | 2 | 281 | 0.0095 | 0.778 | resolved | 是 | 否 | 2_same_subsystem_indirect |
| Split | Playgreen | 0.685 | 0 | 282 | 0.0095 | 0.889 | unresolved_contemporaneous | 是 | 否 | 2_same_subsystem_indirect |
| Kiskitto | Split | 0.610 | 2 | 267 | 0.0095 | 1.000 | resolved | 是 | 否 | 2_same_subsystem_indirect |
| Kiskitto | Playgreen | 0.534 | 1 | 267 | 0.0095 | 0.778 | resolved | 是 | 否 | 2_same_subsystem_indirect |
| Kalamalka | Vaseux | 0.492 | 6 | 324 | 0.0095 | 0.778 | resolved | 是 | 否 | 2_same_subsystem_indirect |
| Vaseux | Kalamalka | 0.486 | 1 | 319 | 0.0095 | 0.944 | resolved | 是 | 否 | 2_same_subsystem_indirect |
| Okanagan | Vaseux | 0.475 | 7 | 328 | 0.0399 | 0.889 | resolved | 是 | 否 | 2_same_subsystem_indirect |
| Rainy | Sipiwesk | 0.649 | 6 | 237 | 0.0095 | 1.000 | resolved | 是 | 否 | 3_same_basin_diff_subsystem |
| Rainy | Split | 0.567 | 3 | 303 | 0.0257 | 0.944 | resolved | 是 | 否 | 3_same_basin_diff_subsystem |
| Lake of the Woods | Sipiwesk | 0.509 | 3 | 239 | 0.0399 | 0.833 | resolved | 是 | 否 | 3_same_basin_diff_subsystem |
| Kiskitto | Lake of the Woods | 0.481 | 3 | 295 | 0.0095 | 0.889 | resolved | 是 | 否 | 3_same_basin_diff_subsystem |
| Lake of the Woods | Split | 0.458 | 3 | 303 | 0.0399 | 0.944 | resolved | 是 | 否 | 3_same_basin_diff_subsystem |
| Split | Kalamalka | 0.553 | 2 | 303 | 0.0095 | 1.000 | resolved | 是 | 否 | 4_different_basin |
| Playgreen | Kalamalka | 0.519 | 6 | 290 | 0.0257 | 1.000 | resolved | 是 | 否 | 4_different_basin |
| Sipiwesk | Kalamalka | 0.507 | 3 | 291 | 0.0399 | 1.000 | resolved | 是 | 否 | 4_different_basin |
| Kalamalka | Lake of the Woods | 0.439 | -9 | 320 | 0.0095 | 0.944 | rejected_reverse | 否 | 否 | 4_different_basin |
| Rainy | Okanagan | 0.407 | 6 | 329 | 0.0399 | 1.000 | resolved | 是 | 否 | 4_different_basin |
| Playgreen | Okanagan | 0.401 | 1 | 306 | 0.0399 | 0.944 | resolved | 是 | 否 | 4_different_basin |
