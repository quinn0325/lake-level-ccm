**表4.2　Diebold–Mariano 检验结果摘要。** 每行为一组预先设定的方法对，逐湖、逐预见期比较滚动起点的平方误差损失差；「significant」为 Benjamini–Hochberg 校正后仍显著的项数，校正在全部 333 项可用检验上一次性施加。「favouring」两列给出显著项中各方向的占比，「significant by horizon」为显著项在 h = 1 / 3 / 6 / 12 上的分布。全部 14 组方法对共 341 项检验（333 项可用、42 项显著），逐项结果见表C5 与补充材料 S4。

| family | strategy | reference | lakes | tests | significant | favouring_strategy | favouring_reference | significant_by_horizon |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| XGBoost | CCM-direct | AR-only | 9 | 35 | 7 | 5 | 2 | 0 / 2 / 1 / 4 |
| XGBoost | CCM-ancestors | AR-only | 9 | 35 | 6 | 5 | 1 | 0 / 2 / 1 / 3 |
| XGBoost | CCM-ancestors | Persistence | 9 | 34 | 5 | 5 | 0 | 0 / 0 / 1 / 4 |
| XGBoost | CCM-ancestors | All-vars | 9 | 35 | 3 | 2 | 1 | 1 / 0 / 0 / 2 |
| XGBoost | CCM-ancestors | Stepwise | 9 | 35 | 5 | 3 | 2 | 0 / 1 / 1 / 3 |
| XGBoost | CCM-neighbour | CCM-ancestors | 8 | 28 | 4 | 2 | 2 | 1 / 0 / 1 / 2 |
| SARIMA(X) | CCM-ancestors | SARIMA | 4 | 16 | 2 | 1 | 1 | 0 / 0 / 1 / 1 |
| SARIMA(X) | CCM-ancestors | Stepwise | 3 | 12 | 1 | 1 | 0 | 1 / 0 / 0 / 0 |
| SARIMA(X) | CCM-ancestors | CCM-direct | 4 | 16 | 1 | 1 | 0 | 1 / 0 / 0 / 0 |
| SARIMA(X) | CCM-neighbour | CCM-ancestors | 1 | 4 | 0 | 0 | 0 | 0 / 0 / 0 / 0 |
