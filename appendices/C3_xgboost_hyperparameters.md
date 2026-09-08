**表C3　XGBoost 超参数候选网格与交叉验证结果**

固定 subsample = 0.8、colsample_bytree = 0.8、random_state = 0；在 10 个湖泊的训练期内以 3 折扩展窗口交叉验证（共 30 折，仅使用 AR-only 特征）比较平均 RMSE，选出一组供所有湖泊与全部策略共用。选中 max_depth = 2、learning_rate = 0.05、n_estimators = 200，平均 CV RMSE = 0.155 m。

| max_depth | learning_rate | n_estimators | cv_rmse_m | selected |
| ---: | ---: | ---: | ---: | --- |
| 2 | 0.05 | 200 | 0.15547 | 是 |
| 3 | 0.05 | 200 | 0.15933 | 否 |
| 3 | 0.10 | 100 | 0.15871 | 否 |
| 4 | 0.05 | 300 | 0.16096 | 否 |
| 3 | 0.03 | 400 | 0.15958 | 否 |
| 5 | 0.05 | 200 | 0.16270 | 否 |
