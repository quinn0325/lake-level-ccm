# 代码结构

活跃分析路径与非路径内容已分开。目录顺序即执行顺序。

## 分析路径（产出论文结果）

| 目录 | 脚本 | 产出 |
|---|---|---|
| `00_data_generation/` | `modal_lake_screening.py` | 全加拿大湖泊筛选 |
| | `ccm_modal_app.py`、`ccm_lib.py` | 抓取 WSC/ERA5-Land、空间聚合 → `results/lake_pkls/` |
| | `recompute_embed_params.py` | 只用训练期重算 E → `embed_params_corrected.json` |
| `01_shared/` | `ccm_full_pipeline.py` | 共享库：面板构建、CCM 单边检验、FDR、预测模型与滚动评价 |
| `02_within_lake_ccm/` | `run_within_lake_ccm.py` | 420 条湖内边 → `ccm_all_edges_merged_fdr.csv` |
| `03_inter_lake_ccm/` | `run_inter_lake_ccm.py` | 90 条湖间边 → `connectivity_*.csv` |
| `04_forecast/` | `modal_forecast_synchrony_filtered.py` | 13 方法 × 10 湖 → `forecast_synchrony_filtered_*.csv` |
| `06_figures/` | `figure_3_1_study_area.py` | 图 3.1（第四章的图尚未开始） |

## 不在分析路径上

**`05_diagnostics/`** — 数据质量自查，**其产物不进论文**：

- `verify_10lakes_completeness.py`、`data_qa_outliers.py` — 只打印报告，不留结果文件
- `data_qa_structural_breaks.py` — Pettitt 突变点检验，产出 `data_qa_structural_breaks.csv`。
  该检验假定观测独立，而月水位强自相关，会过度拒绝（20 条序列检出 17 条），
  因此**不作为论文结论使用**；第三章 3.3 对平稳性的假设陈述不依赖它。

**`06_figures/_exploratory/`** — 探索阶段的图（EDA、lasagna、缺测甘特图、相关热图、
水位标准差、早期研究区域图）。均未进入论文，保留作过程记录。

**`_legacy/`** — 2026-09-03 从 `ccm_full_pipeline.py` 移出的 478 行不可达代码
（PCMCI、被专用脚本取代的旧入口、早期单机预测入口、已内嵌于 `ccm_lib` 的数据抓取）。
不可直接运行，仅作记录，详见该文件头部说明。

## 备注

- `config.py` 目前只被 `00_data_generation/` 下的脚本读取；其余脚本各自维护常量副本
  （湖泊清单散落多处）。这是已知的结构性问题，见 `docs/`。
- 第四章的图脚本已于 2026-09-03 全部删除（`figure_4_1_*` 三版、`figure_4_2_*` 两版、
  `figure_4_3_*`、`figure_ccm_networks_10lakes`、`plot_forecast_results_presentation`）。
  它们都写于 2026-08-31 重跑之前，产出的图对应已被取代的结果；第四章用哪些图尚未确定，
  待定稿后重写。旧版本保留在 `../_backup_20260831_pre_ancestor_lag_fix/code/06_figures/`。

## `results/sensitivity/`

三个设计变体的完整重跑结果（`baseline_ccmlag`、`xcorr_d0`、`drop_d0`），2026-08-31 产出，
可由 `--variant` 开关复现。第三章未声明该敏感性分析，故**不进论文**，保留作质控记录；
详见 `_STATUS.md`。
