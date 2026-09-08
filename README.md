# Causal Exploration and Predictability of Lake Water Level

Reproduction materials for the MSc dissertation *Causal Exploration and
Predictability of Lake Water Level: Evidence from Two Regulated Canadian River
Basins* (MSc Data Science, University of Manchester, 2026).

This repository contains everything needed to reproduce the analysis from the
source data: the code that builds the dataset, the code that runs the causal and
forecasting analyses, the code that produces every figure and table in the
dissertation, and the key numerical outputs so that a reader can check their own
run against ours.

---

## 1. Project overview

The study asks whether convergent cross mapping (CCM) can identify directional
drivers of monthly lake water level in regulated river systems, whether it can
detect connectivity between lakes, and whether the relationships it finds carry
any value for forecasting.

**Study system.** Ten regulated Canadian lakes over 1994-01 to 2024-12
(372 months), in two hydrological systems:

| System | Lakes |
| --- | --- |
| Okanagan (British Columbia) | Kalamalka, Okanagan, Skaha, Vaseux |
| Nelson–Winnipeg (Manitoba / Ontario) | Rainy, Lake of the Woods, Playgreen, Kiskitto, Sipiwesk, Split |

**Variables.** Water level (`WL`) plus six candidate drivers: regulated outflow
(`RegFlow`), runoff (`R`), precipitation (`P`), snow water equivalent (`SWE`),
evaporation (`Evap`) and air temperature (`T`).

**Three analyses.**

1. *Within-lake CCM* — 420 directed candidate edges (10 lakes × 7 variables × 6
   targets), each tested with a signed lag scan over d ∈ [−12, +12].
2. *Between-lake CCM* — 90 directed edges from the 45 lake pairs.
3. *Conditional forecasting* — 13 method configurations per lake (three
   no-exogenous baselines plus five variable-selection strategies × two model
   families), evaluated by rolling origin at horizons of 1, 3, 6 and 12 months
   and compared with Diebold–Mariano tests.

Significance is assessed against IAAFT surrogates (B = 500) with
Benjamini–Hochberg FDR control at α = 0.05, applied separately within each of
three testing families: the 420 within-lake edges, the 90 between-lake edges,
and the 333 usable DM tests.

---

## 2. Data sources

No raw data is redistributed here. All four sources are publicly available and
can be obtained directly from the providers below.

| Source | Used for | Access |
| --- | --- | --- |
| **Water Survey of Canada (HYDAT)** | Monthly lake water level and regulated outflow, by station | <https://www.canada.ca/en/environment-climate-change/services/water-overview/quantity/monitoring/survey/data-products-services/national-archive-hydat.html> |
| **ERA5-Land monthly means** (Copernicus C3S) | `T`, `P`, `R`, `SWE`, `Evap`, area-averaged over each lake's catchment | Copernicus Climate Data Store, requires a free CDS API key: <https://cds.climate.copernicus.eu> |
| **HydroLAKES v1.0** | Lake polygons and `Hylak_id` identifiers | <https://www.hydrosheds.org/products/hydrolakes> (shapefile download) |
| **HydroBASINS level 12 (`hybas_na_lev12_v1c`)** | Catchment delineation for the ERA5-Land spatial averages | <https://www.hydrosheds.org/products/hydrobasins> |

Station identifiers for every lake are listed in `appendices/A1_lakes_and_stations.csv`.

**A processed dataset is provided.** Reproducing the acquisition stage requires
a CDS API key and several hours of downloads, so the analysis-ready product of
that stage is included in this repository (see §6, *Dataset*). Anyone wanting to
reproduce only the analysis can start from `lake_pkls/` and skip steps 1–2.

---

## 3. Software and environment

Python 3.12 (3.10–3.12 all tested). Install with:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Three versions are pinned in `requirements.txt` and must not be changed:

- `pyEDM==2.4.0` — 2.5.0 removed the `sequential` argument from `CCM()`; later
  versions raise `TypeError` in this code.
- `pandas==2.2.2`, `numpy==1.26.4` — pandas 3.x changes chained-assignment
  behaviour (copy-on-write), for which this code is not adapted.

The remaining dependencies (`scipy`, `statsmodels`, `pmdarima`, `xgboost`,
`networkx`, `matplotlib`) are unpinned; any recent version works.

`modal` is imported by the cloud-parallel scripts but a **Modal account is not
required** — `run_local.py` reimplements every stage with local multiprocessing.
`geopandas` is needed only to redraw Figure 3.1 (the study-area map) and to run
the acquisition stage.

### Reproduction tolerance

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

## 4. Repository structure

```
lake-level-ccm/
├── README.md
├── requirements.txt
├── run_local.py                  local driver for every analysis stage
│
├── code/
│   ├── config.py                 single source of truth: lakes, lags, α, split
│   ├── 00_data_generation/       acquisition and monthly-series construction
│   │   ├── modal_lake_screening.py     candidate-lake screening
│   │   ├── ccm_modal_app.py            HYDAT + ERA5-Land download, panel build
│   │   ├── ccm_lib.py                  embedding-parameter selection
│   │   └── recompute_embed_params.py   E chosen on the training period only
│   ├── 01_shared/
│   │   └── ccm_full_pipeline.py  preprocessing, CCM, surrogates, FDR, lag rules
│   ├── 02_within_lake_ccm/       the 420 within-lake edges
│   ├── 03_inter_lake_ccm/        the 90 between-lake edges
│   ├── 04_forecast/              rolling-origin evaluation and DM tests
│   ├── 05_diagnostics/           outlier screen, structural breaks, coverage
│   ├── 06_figures/               one script per dissertation figure
│   ├── 07_tables/                build_ch4_tables.py derives the Chapter 4
│   │                             intermediate tables from results/; the rest
│   │                             build one dissertation or appendix table each
│   └── 08_dataset/
│       └── build_public_dataset.py     builds the three dataset layers
│
├── lake_pkls/                    preprocessed per-lake panels (analysis entry point)
├── dataset/                      the public dataset (see §6)
├── results/                      key numerical outputs
├── ch4_tables/                   intermediate tables the figures and appendices read
├── figures/                      final figures as PDF and PNG
└── appendices/                   the appendix tables as CSV, plus Figure B1
```

Exploratory analyses that do not appear in the dissertation are not included.

---

## 5. How to reproduce the analysis

### The short path (from the provided panels)

If you accept the provided `lake_pkls/`, the whole analysis runs from one entry
point:

```bash
python run_local.py --stage all --workers 8
```

Individual stages:

```bash
python run_local.py --stage embed      # embedding dimension E per lake and variable
python run_local.py --stage within     # 420 within-lake CCM edges
python run_local.py --stage inter      # 90 between-lake CCM edges
python run_local.py --stage figures    # all figures
```

The forecasting stage is run separately because it depends on the two CCM
outputs above:

```bash
python code/04_forecast/modal_forecast_synchrony_filtered.py
```

**Runtime.** The within-lake stage is the bottleneck: roughly 8–18 hours on one
core, 2–4 hours with `--workers` set to the number of physical cores. It writes
each edge as soon as it finishes and skips completed edges on a rerun, so it can
be interrupted safely. The between-lake stage is about a fifth of that; the
forecasting stage takes roughly one hour; figures and tables take under a minute.

### The full path (from the original sources)

1. **Obtain the source data.** Download the HYDAT archive, register for a CDS
   API key, and download the HydroLAKES and HydroBASINS shapefiles (§2).
2. **Construct the monthly series.**
   `code/00_data_generation/modal_lake_screening.py` selects the ten study lakes
   from the HYDAT station network by record completeness and regulation status;
   `ccm_modal_app.py` then pulls the water-level and outflow series, downloads
   ERA5-Land, averages it over each lake's HydroBASINS catchment, and writes
   `lake_pkls/<lake>_result.pkl`.
3. **Run preprocessing.** Outlier screening, multi-station combination and
   training-period deseasonalisation all live in
   `code/01_shared/ccm_full_pipeline.py` and are applied automatically whenever a
   panel is loaded. `code/00_data_generation/recompute_embed_params.py` selects
   the embedding dimension E on the training period only.
4. **Run within-lake CCM** — `run_local.py --stage within`.
5. **Run between-lake CCM** — `run_local.py --stage inter`.
6. **Run the forecasting analysis** —
   `code/04_forecast/modal_forecast_synchrony_filtered.py`.
7. **Generate tables and figures** — `run_local.py --stage figures`. This runs
   `code/07_tables/build_ch4_tables.py` first, which derives the six intermediate
   tables in `ch4_tables/` that every figure and appendix script reads, then runs
   every figure script and every table script in turn.

   Rerunning writes figures to `results/figures/` (including the large TIFF
   versions, which are not committed). The copies in `figures/` are the ones used
   in the dissertation.

Steps 2, 4, 5 and 6 also have `modal run` entry points for cloud parallelism;
those are the paths that produced the results in this repository, and the local
driver reproduces them because the stage functions themselves are backend-neutral.

### Key parameters

All of these live in `code/config.py`; nothing is hard-coded downstream.

| Parameter | Value | Where |
| --- | --- | --- |
| Study period | 1994-01 – 2024-12 (372 months) | — |
| Train / test split | last 37 months held out | `FORECAST_HORIZON` |
| Embedding delay τ | 1 (fixed, all lakes and variables) | `EMBED_TAU` |
| Embedding dimension E | chosen from 2–10 by simplex self-prediction | `EMBED_E_CANDIDATES` |
| Causal lag scan | d ∈ [−12, +12] | `CAUSAL_LAGS` |
| Forecast-domain lags | d ∈ [1, 12] | `FORECAST_LAGS` |
| IAAFT surrogates | 500, 100 iterations each | `N_SURROGATES`, `IAAFT_N_ITER` |
| FDR level | α = 0.05 (Benjamini–Hochberg) | `FDR_ALPHA` |
| Longest fillable gap | 6 months | `MAX_FILLABLE_GAP_MONTHS` |
| Rolling forecast horizons | 1, 3, 6, 12 months | `ROLLING_HORIZONS` |

---

## 6. Expected outputs

### Dataset (`dataset/`)

A by-product of this study is an analysis-ready monthly dataset for the ten
lakes, built by `code/08_dataset/build_public_dataset.py` in three layers:

| File | Rows | Content |
| --- | ---: | --- |
| `raw_station_water_level.csv` | 5,208 | Per-station monthly water level before screening, for provenance |
| `lake_monthly.csv` | 3,720 | One row per lake-month: combined `WL` and the six drivers, real gaps preserved as blanks, not deseasonalised |
| `lake_monthly_deseasonalised.csv` | 3,720 | The same series after subtracting the training-period monthly climatology — what the CCM and forecasting analyses actually use |
| `data_dictionary.csv` | 13 | Column names, units and definitions |

Deseasonalisation uses training-period monthly means only, so the test period
leaks no information into the analysis.

### Results (`results/`)

| File | Rows | Produced by |
| --- | ---: | --- |
| `embed_params_corrected.json` | 10 lakes | embedding stage |
| `ccm_all_edges_merged_fdr.csv` | 420 | within-lake stage |
| `connectivity_full_pairwise_ccm_results.csv` | 90 | between-lake stage |
| `connectivity_connected_vs_unconnected_summary.csv` | 2 | between-lake stage, connected vs unconnected group summary |
| `forecast_synchrony_filtered_rolling_results.csv` | 372 | forecasting stage (93 valid lake × method combinations × 4 horizons) |
| `forecast_synchrony_filtered_full_results.csv` | 122 | forecasting stage, one row per fitted lake x method |
| `forecast_synchrony_filtered_selected_lags.csv` | 38 | predictors and lags entering each model |
| `forecast_synchrony_filtered_dm_results.csv` | 341 | Diebold–Mariano tests (333 usable) |
| `data_qa_structural_breaks.csv` | 17 | diagnostics |

**Headline numbers to check a rerun against.** 212 of the 420 within-lake edges
are supported, of which 24 are direct driver-to-water-level relationships;
26 of the 90 between-lake edges are supported; 42 of the 333 DM tests are
significant after FDR correction.

### Figures and tables

`figures/` holds every figure in the dissertation as PDF and PNG. File names
follow the script names rather than the final numbering in the text:

| File | In the dissertation |
| --- | --- |
| `figure_3_1_study_area` | Figure 3.1 |
| `figure_4_1_wl_variability` | Figure 4.1 |
| `figure_4_2_ccm_driver_matrix` | Figure 4.2 |
| `figure_4_3_interlake_network` | Figure 4.3 |
| `figure_4_3_pair_strength` | Figure 4.4 |
| `figure_4_4_per_lake_smallmultiples` | Figure 4.5 |
| `figure_B1_within_lake_network` | Figure B1 |

`appendices/` holds the eleven appendix tables as CSV and Markdown, regenerated by
`code/07_tables/build_appendices.py` (Chinese titles) and
`build_appendices_en.py` (English titles).

---

## 7. Licence and citation

Code is released under the MIT Licence (`LICENSE`). The derived dataset in
`dataset/` is released under CC BY 4.0; it is built from HYDAT (Open Government
Licence – Canada), ERA5-Land (Copernicus Licence), and HydroLAKES / HydroBASINS
(CC BY 4.0), and users should cite those primary sources as well.

If you use this repository, please cite the dissertation.
