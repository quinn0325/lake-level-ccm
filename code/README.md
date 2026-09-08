# Code layout

Directory order is execution order. Everything here is on the path that produced
the results in the dissertation; exploratory work that did not make it into the
text is not in this repository.

| Directory | Script | Produces |
|---|---|---|
| `00_data_generation/` | `modal_lake_screening.py` | screens Canadian lakes by record completeness and regulation status |
| | `ccm_modal_app.py`, `ccm_lib.py` | pulls HYDAT and ERA5-Land, averages over catchments → `lake_pkls/` |
| | `recompute_embed_params.py` | embedding dimension E from the training period only → `embed_params_corrected.json` |
| `01_shared/` | `ccm_full_pipeline.py` | the shared library: panel construction, one-edge CCM, surrogates, FDR, forecast models, rolling evaluation |
| `02_within_lake_ccm/` | `run_within_lake_ccm.py` | 420 within-lake edges → `ccm_all_edges_merged_fdr.csv` |
| `03_inter_lake_ccm/` | `run_inter_lake_ccm.py` | 90 between-lake edges → `connectivity_full_pairwise_ccm_results.csv` |
| `04_forecast/` | `modal_forecast_synchrony_filtered.py` | 13 methods × 10 lakes → `forecast_synchrony_filtered_*.csv` |
| `06_figures/` | one script per figure | `results/figures/` |
| `07_tables/` | `build_ch4_tables.py` first, then the rest | `ch4_tables/`, `appendices/` |
| `08_dataset/` | `build_public_dataset.py` | the three dataset layers in `dataset/` |

`07_tables/` has an ordering constraint. `build_ch4_tables.py` derives six
intermediate tables from `results/`; the four `table_*.py` scripts then read
those and write four more; `build_appendices*.py` read both sets. Alphabetical
order gets this wrong — `build_appendices.py` sorts before
`table_C1_supported_drivers.py`, which writes a table it needs. `run_local.py`
hard-codes the correct order.

## A known wart

`config.py` is meant to be the single source of truth, but only the
`00_data_generation/` scripts actually read it. The others keep their own copies
of the lake list and variable list. They agree, and `verify.py` would catch it
if they stopped agreeing, but it is duplication that should have been cleaned up.
