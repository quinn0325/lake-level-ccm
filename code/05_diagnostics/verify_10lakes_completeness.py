"""验证实际10个study lakes的WL/RegFlow是否满足逐年完整度>=90%且无整年缺失。"""
import os
import modal

app = modal.App("verify-10lakes-completeness")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "pandas==2.2.2", "numpy==1.26.4", "scipy", "statsmodels",
        "pmdarima", "xgboost", "networkx", "pyEDM==2.4.0",
    )
    .add_local_python_source("ccm_full_pipeline")
)

volume = modal.Volume.from_name("ccm-data", create_if_missing=False)
DATA_ROOT = "/data"

LAKES = ["Kalamalka_Lake", "Okanagan_Lake", "Skaha_Lake", "Vaseux_Lake",
         "Rainy_Lake", "Lake_of_the_Woods", "Playgreen_Lake", "Kiskitto_Lake",
         "Sipiwesk_Lake", "Split_Lake"]

@app.function(image=image, volumes={DATA_ROOT: volume}, timeout=300)
def check_lake(lake_name: str) -> dict:
    import pickle
    import pandas as pd

    pkl_path = os.path.join(f"{DATA_ROOT}/lake_results", f"{lake_name}_result.pkl")
    if not os.path.exists(pkl_path):
        return {"lake": lake_name, "status": "PKL_NOT_FOUND"}

    with open(pkl_path, "rb") as f:
        cached = pickle.load(f)

    import ccm_full_pipeline as p
    clean_wide = p.clean_wide_wl(cached["wide_wl"], log_prefix=f"[{lake_name}] ")
    wl = p.combine_station_water_levels(clean_wide, method="anomaly_mean").sort_index()
    real_predictors = cached["real_predictors"]
    has_regflow = "RegFlow" in real_predictors.columns and real_predictors["RegFlow"].notna().any()
    rf = real_predictors["RegFlow"].sort_index() if has_regflow else None

    def yearly_completeness(series, y0=1994, y1=2024):
        s = series.dropna()
        rows = []
        for y in range(y0, y1 + 1):
            n_present = int((s.index.year == y).sum())
            rows.append((y, n_present))
        return pd.DataFrame(rows, columns=["year", "n_present"])

    wl_yc = yearly_completeness(wl)
    wl_full_missing = wl_yc[wl_yc["n_present"] == 0]["year"].tolist()
    wl_low = wl_yc[(wl_yc["n_present"] / 12) < 0.9]["year"].tolist()

    result = {
        "lake": lake_name, "status": "OK",
        "wl_full_missing_years": wl_full_missing,
        "wl_low_years(<90%)": wl_low,
        "has_regflow": has_regflow,
    }
    if has_regflow:
        rf_yc = yearly_completeness(rf)
        result["rf_full_missing_years"] = rf_yc[rf_yc["n_present"] == 0]["year"].tolist()
        result["rf_low_years(<90%)"] = rf_yc[(rf_yc["n_present"] / 12) < 0.9]["year"].tolist()
    return result


@app.local_entrypoint()
def main():
    import pandas as pd
    results = list(check_lake.map(LAKES))
    df = pd.DataFrame(results)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 60)
    print(df.to_string(index=False))
