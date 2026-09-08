**Table C3. XGBoost hyperparameter grid and cross-validation results**

Subsample = 0.8, colsample_bytree = 0.8 and random_state = 0 were fixed. Mean RMSE was compared over three expanding-window folds in each of the ten lakes (30 folds in total) using autoregressive features only, and one configuration was then shared by every lake and strategy.

| max_depth | learning_rate | n_estimators | cv_rmse_m | selected |
| ---: | ---: | ---: | ---: | --- |
| 2 | 0.05 | 200 | 0.15547 | yes |
| 3 | 0.05 | 200 | 0.15933 | no |
| 3 | 0.10 | 100 | 0.15871 | no |
| 4 | 0.05 | 300 | 0.16096 | no |
| 3 | 0.03 | 400 | 0.15958 | no |
| 5 | 0.05 | 200 | 0.16270 | no |
