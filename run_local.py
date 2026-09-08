"""Run every stage locally. No Modal account needed.

The stage functions are backend-neutral -- the `@app.function` decorators in the
Modal scripts only wrap ordinary Python. This driver calls those same function
bodies with multiprocessing, starting from the `lake_pkls/*.pkl` shipped here.

Usage
----
    python run_local.py --stage all
    python run_local.py --stage within --workers 8
    python run_local.py --stage figures

Stages
------
    embed     embedding dimensions    -> results/embed_params_corrected.json
    within    420 within-lake edges   -> results/ccm_all_edges_merged_fdr.csv
    inter     90 between-lake edges   -> results/connectivity_full_pairwise_ccm_results.csv
    forecast  the 13 method variants  -> results/forecast_*_results.csv
    figures   every figure and table  -> results/figures/, ch4_tables/, appendices/
    all       the above, in order

Timing: `within` is 8-18 hours on one core, 2-4 with `--workers` set to your core
count. It writes each edge as it finishes and skips completed ones on a rerun,
so it is safe to interrupt.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PKG = Path(__file__).resolve().parent
CODE = PKG / "code"
for sub in ("", "01_shared", "00_data_generation", "03_inter_lake_ccm",
            "04_forecast", "06_figures"):
    sys.path.insert(0, str(CODE / sub) if sub else str(CODE))

RESULTS = PKG / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# Point the shared library inside this package rather than at any absolute path
os.environ.setdefault("CCM_PKL_DIR", str(PKG / "lake_pkls"))
os.environ.setdefault("CCM_OUT_DIR", str(RESULTS))
os.environ.setdefault("CCM_EMBED_PARAMS", str(RESULTS / "embed_params_corrected.json"))


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- embed
def stage_embed():
    """Recompute embedding parameters. tau is fixed, so only E is searched."""
    import ast
    import json
    import pickle

    import numpy as np
    import config
    import ccm_full_pipeline as p

    ns = {"np": np, "pd": __import__("pandas"), "pyEDM": __import__("pyEDM")}
    src = (CODE / "00_data_generation" / "ccm_lib.py").read_text(encoding="utf-8")
    want = {"simplex_self_predict_rho", "select_E", "get_embedding_params"}
    got = set()
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in want:
            exec(compile(ast.Module([node], []), "ccm_lib.py", "exec"), ns)
            got.add(node.name)
    assert not (want - got), f"could not extract from ccm_lib.py: {want - got}"

    merged = {}
    for lake in config.LAKES:
        with open(PKG / "lake_pkls" / f"{lake}_result.pkl", "rb") as fh:
            cached = pickle.load(fh)
        wl = p.combine_station_water_levels(
            p.clean_wide_wl(cached["wide_wl"]), method="anomaly_mean")
        rp = cached["real_predictors"]
        _, des = p.build_variable_panel(
            wl, {c: rp[c] for c in rp.columns}, forecast_horizon=config.FORECAST_HORIZON)
        train = des.iloc[:-config.FORECAST_HORIZON]
        out = {}
        for var in config.VARIABLES:
            if var not in train.columns:
                continue
            vals = train[var].values
            if int(np.sum(~np.isnan(vals))) < 30:
                continue
            prm = ns["get_embedding_params"](
                vals, var, verbose=False,
                tau=config.EMBED_TAU, candidate_E=config.EMBED_E_CANDIDATES)
            out[var] = {"E": int(prm["E"]), "tau": int(prm["tau"])}
        merged[lake] = out
        _log(f"  {lake}: {len(out)} variables")
    (RESULTS / "embed_params_corrected.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"wrote {RESULTS / 'embed_params_corrected.json'}")


# ---------------------------------------------------------------- within
def _one_within_edge(args):
    lake, cause, effect, n_sur = args
    import ccm_full_pipeline as p
    panel, embed, _ = p.load_lake_panel_for_ccm(lake)
    if effect not in embed:
        return {"lake": lake, "cause": cause, "effect": effect,
                "status": "effect_embed_params_missing"}
    row = p.test_one_ccm_edge(panel, cause, effect,
                              embed[effect]["E"], embed[effect]["tau"],
                              n_surrogates=n_sur)
    row["lake"] = lake
    return row


def stage_within(workers, n_surrogates):
    import pandas as pd
    import config
    import ccm_full_pipeline as p

    tasks = [(lk, c, e, n_surrogates)
             for lk in config.LAKES
             for c in config.VARIABLES for e in config.VARIABLES if c != e]
    _log(f"within-lake CCM: {len(tasks)} edges, {workers} processes, "
         f"n_surrogates={n_surrogates}")
    rows, done = [], 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one_within_edge, t): t for t in tasks}
        for f in as_completed(futs):
            rows.append(f.result())
            done += 1
            if done % 20 == 0:
                _log(f"  {done}/{len(tasks)}")
    df = p.apply_fdr_and_causal_evidence(pd.DataFrame(rows))
    out = RESULTS / "ccm_all_edges_merged_fdr.csv"
    df.to_csv(out, index=False)
    _log(f"done: {len(df)} edges, "
         f"{int(df['causal_evidence'].sum())} supported -> {out}")


# ---------------------------------------------------------------- inter
def _one_inter_edge(args):
    a, b, n_sur = args
    import ccm_full_pipeline as p
    panel, ea, eb = p.load_pair_panel_for_connectivity(a, b)
    row = p.test_one_ccm_edge(panel, a, b, eb["E"], eb["tau"], n_surrogates=n_sur)
    row["cause_lake"], row["effect_lake"] = a, b
    return row


def stage_inter(workers, n_surrogates):
    import itertools
    import pandas as pd
    import config
    import ccm_full_pipeline as p

    conn = {tuple(sorted(x)) for x in config.WATERWAY_CONNECTED_PAIRS}
    pairs = list(itertools.combinations(config.LAKES, 2))
    tasks = [(a, b, n_surrogates) for a, b in pairs] + \
            [(b, a, n_surrogates) for a, b in pairs]
    _log(f"between-lake CCM: {len(tasks)} edges, {workers} processes")
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(_one_inter_edge, t) for t in tasks]):
            rows.append(f.result())
    df = p.apply_fdr_and_causal_evidence(pd.DataFrame(rows))
    df["waterway_connected"] = [
        tuple(sorted([r.cause_lake, r.effect_lake])) in conn for r in df.itertuples()]
    out = RESULTS / "connectivity_full_pairwise_ccm_results.csv"
    df.to_csv(out, index=False)
    _log(f"done: {len(df)} edges, "
         f"{int(df['causal_evidence'].sum())} supported -> {out}")


# ---------------------------------------------------------------- forecast
def stage_forecast(variant="main"):
    """The forecasting stage, run locally.

    `main()` in modal_forecast_synchrony_filtered.py is a local_entrypoint that
    reads and writes /data on a Modal Volume, so it needs an account. The two
    functions it maps over are not container-specific though -- the only things
    that are live in `_configure_module()`. Swap those for local paths and call
    the same function bodies through Modal's `.local()`, and you get the same
    code path as the cloud run.
    """
    import pandas as pd
    sys.path.insert(0, str(CODE / "04_forecast"))
    import modal_forecast_synchrony_filtered as F

    def _configure_local():
        import ccm_full_pipeline as p
        p.PKL_DIR = str(PKG / "lake_pkls")
        p.OUT_DIR = str(RESULTS)
        p.EMBED_PARAMS_PATH = str(RESULTS / "embed_params_corrected.json")
        return p
    F._configure_module = _configure_local
    # _original_within_direct_lags and _original_neighbor_lags read these two
    # module-level constants directly rather than taking them as arguments, so
    # they have to be redirected too.
    F.WITHIN_ORIGINAL_INPUT = str(RESULTS / "ccm_all_edges_merged_fdr.csv")
    F.INTER_ORIGINAL_INPUT = str(
        RESULTS / "connectivity_full_pairwise_ccm_results.csv")

    out_dir = RESULTS if variant == "main" else RESULTS / "sensitivity" / variant
    within = F._filtered_within_edges(
        pd, str(RESULTS / "ccm_all_edges_merged_fdr.csv"))
    inter = F._filtered_interlake_edges(
        pd, str(RESULTS / "connectivity_full_pairwise_ccm_results.csv"))
    _log(f"variant {variant}: {len(within)} within-lake edges, "
         f"{len(inter)} between-lake edges")

    _log("tuning XGBoost hyperparameters once, globally")
    xgb_params = F.tune_hyperparams.local()
    _log(f"  selected {xgb_params}")

    import config
    wr, ir = within.to_dict("records"), inter.to_dict("records")
    rows = {"full_rows": [], "rolling_rows": [], "dm_rows": [], "selected_rows": []}
    failed = []
    for lake in config.LAKES:
        _log(f"  {lake}")
        try:                                   # one lake failing must not void the other nine
            res = F.run_lake_synchrony_filtered.local(lake, xgb_params, wr, ir, variant)
        except Exception as exc:
            failed.append((lake, f"{type(exc).__name__}: {exc}"))
            continue
        for key in rows:
            rows[key].extend(res.get(key, []))
    for lake, err in failed:
        _log(f"  failed: {lake} -- {err}")

    outputs = {"full": pd.DataFrame(rows["full_rows"]),
               "rolling": pd.DataFrame(rows["rolling_rows"]),
               "dm": F._apply_dm_fdr(pd, pd.DataFrame(rows["dm_rows"])),
               "selected": pd.DataFrame(rows["selected_rows"])}
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, frame in outputs.items():
        path = out_dir / F.OUTPUT_NAMES[key]
        frame.to_csv(path, index=False)
        _log(f"wrote {path} ({len(frame)} rows)")


# ---------------------------------------------------------------- figures
def stage_figures():
    import runpy
    # Do not sort these by filename. Alphabetically build_appendices comes before
    # table_C1_supported_drivers, which writes the TC1 table it reads. Order is:
    #   build_ch4_tables   writes T1/T3/T4/T5/T6/T7 from results/
    #   table_*            write TC1, T4_2, T4X_*, reading the above
    #   figure_*           read T1/T5/T6/T7
    #   build_appendices*  read TC1, T3, T4, T6, T7
    TABLES_FIRST = ["build_ch4_tables.py", "table_C1_supported_drivers.py",
                    "table_4_2_dm_maintext.py", "table_4_4_forecast_summary.py",
                    "table_4_4_matched_rmse.py"]
    TABLES_LAST = ["build_appendices.py", "build_appendices_en.py"]
    scripts = ([CODE / "07_tables" / n for n in TABLES_FIRST]
               + sorted((CODE / "06_figures").glob("figure_*.py"))
               + [CODE / "07_tables" / n for n in TABLES_LAST])
    known = {p.name for p in scripts}
    missed = sorted(p for p in (CODE / "07_tables").glob("*.py")
                    if p.name not in known)
    if missed:                       # nudge whoever adds a script to place it above
        _log(f"  unordered table scripts {[p.name for p in missed]}, "
             f"appended at the end")
        scripts += missed
    for script in scripts:
        _log(f"running {script.name}")
        try:
            runpy.run_path(str(script), run_name="__main__")
        except Exception as exc:                      # one bad figure must not stop the rest
            _log(f"  skipped ({type(exc).__name__}: {exc})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["embed", "within", "inter", "forecast", "figures", "all"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--n-surrogates", type=int, default=500)
    a = ap.parse_args()

    order = (["embed", "within", "inter", "forecast", "figures"]
             if a.stage == "all" else [a.stage])
    for st in order:
        _log(f"===== stage {st} =====")
        if st == "embed":
            stage_embed()
        elif st == "within":
            stage_within(a.workers, a.n_surrogates)
        elif st == "inter":
            stage_inter(a.workers, a.n_surrogates)
        elif st == "figures":
            stage_figures()
        elif st == "forecast":
            stage_forecast()
    _log("done")


if __name__ == "__main__":
    main()
