---
title: "Technical Appendix — Reproducing the Analysis"
subtitle: "Causal Exploration and Predictability of Lake Water Level: Evidence from Two Regulated Canadian River Basins"
---

**Code and data repository:** <https://github.com/quinn0325/lake-level-ccm>

This appendix is self-contained: it states where the source data comes from,
what software is required, in what order the code must be run, which parameters
govern the analysis, and what outputs to expect. It is not a summary of the
dissertation's findings, and it does not describe analyses that were explored
but not reported.

---

## 1. What the repository contains

| Directory | Content |
| --- | --- |
| `code/00_data_generation/` | Lake screening, HYDAT and ERA5-Land acquisition, catchment averaging, embedding-parameter selection |
| `code/01_shared/` | `ccm_full_pipeline.py`: preprocessing, CCM, IAAFT surrogates, FDR control, lag rules — shared by every stage |
| `code/02_within_lake_ccm/` | The 420 within-lake directed edges |
| `code/03_inter_lake_ccm/` | The 90 between-lake directed edges |
| `code/04_forecast/` | Rolling-origin forecast evaluation and Diebold–Mariano tests |
| `code/05_diagnostics/` | Outlier screen, structural-break scan, coverage checks |
| `code/06_figures/`, `code/07_tables/` | One script per figure and per table in the dissertation |
| `code/08_dataset/` | Builds the published dataset from the preprocessed panels |
| `ch4_tables/` | Six intermediate tables derived from `results/` by `build_ch4_tables.py`; every figure and appendix script reads these rather than `results/` directly |
| `lake_pkls/` | Preprocessed per-lake panels — the entry point if you skip acquisition |
| `dataset/` | The published monthly dataset (three layers plus a data dictionary) |
| `results/` | Key numerical outputs, for checking a rerun against ours |
| `figures/`, `appendices/` | Every figure and appendix table in the dissertation |
| `run_local.py` | Single local driver for all analysis stages |
| `verify.py` | Four-minute check: regenerates every derived table, dataset file and figure and compares them byte for byte, then recomputes one CCM edge |

Exploratory work that does not appear in the dissertation is not included.

---

## 2. Data sources

No raw data is redistributed. All four sources are public.

| Source | Used for | Access |
| --- | --- | --- |
| Water Survey of Canada (HYDAT) | Monthly water level and regulated outflow by station | National Water Data Archive, <https://www.canada.ca/en/environment-climate-change/services/water-overview/quantity/monitoring/survey/data-products-services/national-archive-hydat.html> |
| ERA5-Land monthly means (Copernicus C3S) | Temperature, precipitation, runoff, snow water equivalent, evaporation | Copernicus Climate Data Store, free CDS API key required: <https://cds.climate.copernicus.eu> |
| HydroLAKES v1.0 | Lake polygons, `Hylak_id` | <https://www.hydrosheds.org/products/hydrolakes> |
| HydroBASINS level 12 (`hybas_na_lev12_v1c`) | Catchment delineation for the ERA5-Land spatial averages | <https://www.hydrosheds.org/products/hydrobasins> |

The gauge numbers used for each of the ten lakes are listed in
`appendices/A1_lakes_and_stations.csv` and in Table A1 of the dissertation.

Because the acquisition stage needs a CDS API key and several hours of
downloads, its analysis-ready output is shipped in `lake_pkls/`. A reader who
wants to reproduce only the analysis can start at step 4 below.

---

## 3. Software environment

Python 3.12, the version every result here was produced on and the only one
this repository has been tested against. 3.13 and later will not work, because
`numpy==1.26.4` has no wheels for them.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Three pinned versions matter:

- **`pyEDM==2.4.0`** — version 2.5.0 removed the `sequential` argument from
  `CCM()`; later versions raise `TypeError` in this code.
- **`pandas==2.2.2`** and **`numpy==1.26.4`** — pandas 3.x changed
  chained-assignment semantics (copy-on-write); this code is not adapted to it.

`scipy`, `statsmodels`, `pmdarima`, `xgboost`, `networkx` and `matplotlib` are
unpinned. `geopandas` is needed only for the study-area map and the acquisition
stage.

`modal` appears in the requirements because the cloud-parallel scripts import
it, **but no Modal account is needed**: `run_local.py` runs every stage on one
machine with multiprocessing, calling the same function bodies.

**Reproduction tolerance.**

Surrogate generation uses fixed seeds, so the CCM stages are deterministic and
reproduce exactly. The forecasting stage does not reproduce to the last decimal
on a different platform. Rerunning Kalamalka at a one-month horizon on macOS /
arm64 with `xgboost` 3.4.1, against the published results produced on Modal's
Linux containers, gives:

| Method family | Largest absolute RMSE difference |
| --- | --- |
| Persistence | 0 (exact) |
| SARIMA, SARIMAX (all five variable sets) | 1.0 x 10^-5 |
| XGBoost, autoregressive features only | 6.7 x 10^-5 |
| XGBoost with exogenous predictors | 7.8 x 10^-3 |

The first three rows are optimiser and floating-point noise. The last row is
larger because `xgboost` was not version-pinned in the original run and its
handling of missing values in exogenous columns has changed across major
versions; no record of the version used survives, so it cannot be pinned here
retrospectively. A difference of this size can reorder two closely spaced
XGBoost variants within a single lake and horizon, so treat the per-cell
forecasting numbers as reproducible to about three decimal places rather than
exactly. Pin `xgboost` yourself if you need bitwise agreement across machines.

---

## 4. Execution order

| Step | Command | Output |
| --- | --- | --- |
| 1. Obtain source data | Download HYDAT, register a CDS key, download the HydroLAKES and HydroBASINS shapefiles | — |
| 2. Build monthly series | `modal run code/00_data_generation/modal_lake_screening.py`, then `ccm_modal_app.py` | `lake_pkls/<lake>_result.pkl` |
| 3. Preprocess | Applied automatically on every panel load (`ccm_full_pipeline.py`); embedding dimensions from `recompute_embed_params.py` | `results/embed_params_corrected.json` |
| 4. Within-lake CCM | `python run_local.py --stage within --workers 8` | `results/ccm_all_edges_merged_fdr.csv` |
| 5. Between-lake CCM | `python run_local.py --stage inter --workers 8` | `results/connectivity_full_pairwise_ccm_results.csv` |
| 6. Forecasting | `python run_local.py --stage forecast` | `results/forecast_synchrony_filtered_*.csv` |
| 7. Figures and tables | `python run_local.py --stage figures` | `ch4_tables/`, `results/figures/`, `appendices/` |

Steps 3–7 also run as one command:

```bash
python run_local.py --stage all --workers 8
```

**Preprocessing (step 3) in detail.** Water level is screened for outliers in
two stages (a month-to-month change with robust *z* > 6, confirmed by robust
*z* > 5 on the level itself); flagged points are set to missing and never
adjusted. Lakes with several gauges have their records combined as anomalies
about each gauge's own mean. Every series is then deseasonalised by subtracting
the **training-period** monthly climatology, so no test-period information
enters the analysis. Embedding dimension *E* is likewise selected on the
training period only.

**Runtime.** The within-lake stage dominates: roughly 8–18 hours on a single
core, 2–4 hours with `--workers` set to the physical core count. It writes each
edge on completion and skips finished edges on a rerun, so it can be interrupted
safely. The between-lake stage is about a fifth as long, forecasting takes
roughly an hour, and figures and tables take under a minute.

---

## 5. Key parameters

All are defined in `code/config.py`; no downstream script hard-codes them.

| Parameter | Value | Name in `config.py` |
| --- | --- | --- |
| Study period | 1994-01 – 2024-12 (372 months) | — |
| Lakes | 10, in two systems | `LAKES` |
| Variables | `WL`, `RegFlow`, `R`, `P`, `SWE`, `Evap`, `T` | `VARIABLES` |
| Train / test split | last 37 months held out | `FORECAST_HORIZON` |
| Embedding delay τ | 1, fixed for all lakes and variables | `EMBED_TAU` |
| Embedding dimension *E* | selected from 2–10 by simplex self-prediction | `EMBED_E_CANDIDATES` |
| Causal lag scan | *d* ∈ [−12, +12], signed | `CAUSAL_LAGS` |
| Forecast-domain lags | *d* ∈ [1, 12] | `FORECAST_LAGS` |
| Surrogates | 500 IAAFT, 100 iterations each | `N_SURROGATES`, `IAAFT_N_ITER` |
| FDR level | α = 0.05, Benjamini–Hochberg | `FDR_ALPHA` |
| Longest fillable gap | 6 months (longer ⇒ variable dropped) | `MAX_FILLABLE_GAP_MONTHS` |
| Forecast horizons | 1, 3, 6, 12 months | `ROLLING_HORIZONS` |
| Waterway-connected pairs | 7 of the 45 | `WATERWAY_CONNECTED_PAIRS` |

FDR correction is applied **separately within three testing families**: the 420
within-lake edges, the 90 between-lake edges, and the 333 usable
Diebold–Mariano tests. It is not applied per lake, nor pooled across families.

---

## 6. Expected outputs

### The published dataset (`dataset/`)

| File | Rows | Content |
| --- | ---: | --- |
| `raw_station_water_level.csv` | 5,208 | Per-station monthly water level before screening, for provenance |
| `lake_monthly.csv` | 3,720 | One row per lake-month: combined `WL` and six drivers; real gaps left blank; not deseasonalised |
| `lake_monthly_deseasonalised.csv` | 3,720 | The same series minus the training-period monthly climatology — what the analyses use |
| `data_dictionary.csv` | 13 | Column names, units, definitions |

### Analysis results (`results/`)

| File | Rows |
| --- | ---: |
| `embed_params_corrected.json` | 10 lakes × up to 7 variables |
| `ccm_all_edges_merged_fdr.csv` | 420 |
| `connectivity_full_pairwise_ccm_results.csv` | 90 |
| `forecast_synchrony_filtered_full_results.csv` | 122 fitted lake × method configurations |
| `forecast_synchrony_filtered_rolling_results.csv` | 372 (93 valid configurations × 4 horizons) |
| `forecast_synchrony_filtered_selected_lags.csv` | 38 |
| `forecast_synchrony_filtered_dm_results.csv` | 341 (333 usable) |

### Numbers to check a rerun against

- 212 of the 420 within-lake edges are supported (150 with a positive lag,
  23 contemporaneous, 39 with a negative lag); 24 of these are direct
  driver-to-water-level relationships.
- 26 of the 90 between-lake edges are supported, spanning 20 of the 45 lake pairs.
- Of 130 lake × method combinations, 122 were fitted and 93 produced valid
  rolling forecasts; 8 were unavailable because no CCM relationship existed
  (Skaha and Kiskitto) and 29 failed on exogenous gaps longer than six months.
- 42 of the 333 usable Diebold–Mariano tests are significant after FDR correction.

### Figures

`figures/` holds every dissertation figure as PDF and PNG. File names follow the
generating script rather than the numbering in the text:

| File | In the dissertation |
| --- | --- |
| `figure_3_1_study_area` | Figure 3.1 |
| `figure_4_1_wl_variability` | Figure 4.1 |
| `figure_4_2_ccm_driver_matrix` | Figure 4.2 |
| `figure_4_3_interlake_network` | Figure 4.3 |
| `figure_4_3_pair_strength` | Figure 4.4 |
| `figure_4_4_per_lake_smallmultiples` | Figure 4.5 |
| `figure_B1_within_lake_network` | Figure B1 |

High-resolution TIFF versions are not committed (24–82 MB each); rerunning the
figure scripts regenerates them.

---

## 7. Licence

Code: MIT. Derived dataset: CC BY 4.0, subject to the licences of the primary
sources (HYDAT under the Open Government Licence – Canada, ERA5-Land under the
Copernicus Licence, HydroLAKES and HydroBASINS under CC BY 4.0).
