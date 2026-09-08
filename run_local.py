"""本地运行入口——不需要 Modal 账户。

各阶段的计算函数本身与后端无关（Modal 脚本里的 `@app.function` 只是并行调度的
包装，函数体是普通 Python）。本驱动直接调用这些函数，用多进程并行，
数据起点为随包提供的 `lake_pkls/*.pkl`。

用法
----
    python run_local.py --stage all
    python run_local.py --stage within --workers 8
    python run_local.py --stage figures

阶段
----
    embed     重算嵌入参数            → results/embed_params_corrected.json
    within    湖内 CCM 420 条边       → results/ccm_all_edges_merged_fdr.csv
    inter     湖间 CCM 90 条边        → results/connectivity_full_pairwise_ccm_results.csv
    forecast  预测对比                → results/forecast_*_results.csv
    figures   全部图与表              → results/figures/、ch4_tables/
    all       依次执行以上全部

耗时（单核参考）：within 约 8–18 小时；`--workers` 开到 CPU 核数可压缩到 2–4 小时。
within 阶段支持断点续跑：已完成的边会跳过（见 --resume）。
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

# 让共享库把输入输出都指向包内，而不是任何绝对路径
os.environ.setdefault("CCM_PKL_DIR", str(PKG / "lake_pkls"))
os.environ.setdefault("CCM_OUT_DIR", str(RESULTS))
os.environ.setdefault("CCM_EMBED_PARAMS", str(RESULTS / "embed_params_corrected.json"))


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- embed
def stage_embed():
    """重算嵌入参数（τ 固定为 config.EMBED_TAU，仅需选 E）。"""
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
    assert not (want - got), f"未能从 ccm_lib.py 抽取: {want - got}"

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
        _log(f"  {lake}: {len(out)} 个变量")
    (RESULTS / "embed_params_corrected.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"已写出 {RESULTS / 'embed_params_corrected.json'}")


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
    _log(f"湖内 CCM：{len(tasks)} 条边，{workers} 进程并行，n_surrogates={n_surrogates}")
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
    _log(f"完成：{len(df)} 条边，{int(df['causal_evidence'].sum())} 条通过 → {out}")


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
    _log(f"湖间 CCM：{len(tasks)} 条边，{workers} 进程并行")
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(_one_inter_edge, t) for t in tasks]):
            rows.append(f.result())
    df = p.apply_fdr_and_causal_evidence(pd.DataFrame(rows))
    df["waterway_connected"] = [
        tuple(sorted([r.cause_lake, r.effect_lake])) in conn for r in df.itertuples()]
    out = RESULTS / "connectivity_full_pairwise_ccm_results.csv"
    df.to_csv(out, index=False)
    _log(f"完成：{len(df)} 条边，{int(df['causal_evidence'].sum())} 条通过 → {out}")


# ---------------------------------------------------------------- forecast
def stage_forecast(variant="main"):
    """预测对比——本地版。

    `modal_forecast_synchrony_filtered.py` 的 `main()` 是 local_entrypoint，
    读写的是 Modal Volume 上的 /data 路径，没有 Modal 账户跑不了。但被 `.map()`
    调度的两个函数体本身与后端无关：容器专用的只有 `_configure_module()` 里的
    三个路径。这里把它换成包内路径，再用 Modal 函数对象的 `.local()`
    直接在本机调用同一份函数体，因此与云端跑法逐行同源。
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

    out_dir = RESULTS if variant == "main" else RESULTS / "sensitivity" / variant
    within = F._filtered_within_edges(
        pd, str(RESULTS / "ccm_all_edges_merged_fdr.csv"))
    inter = F._filtered_interlake_edges(
        pd, str(RESULTS / "connectivity_full_pairwise_ccm_results.csv"))
    _log(f"变体 {variant}：湖内边 {len(within)} 条，湖间边 {len(inter)} 条")

    _log("全局调一次 XGBoost 超参数……")
    xgb_params = F.tune_hyperparams.local()
    _log(f"  选定 {xgb_params}")

    import config
    wr, ir = within.to_dict("records"), inter.to_dict("records")
    rows = {"full_rows": [], "rolling_rows": [], "dm_rows": [], "selected_rows": []}
    failed = []
    for lake in config.LAKES:
        _log(f"  {lake}")
        try:                                   # 单湖失败不作废其余九个湖
            res = F.run_lake_synchrony_filtered.local(lake, xgb_params, wr, ir, variant)
        except Exception as exc:
            failed.append((lake, f"{type(exc).__name__}: {exc}"))
            continue
        for key in rows:
            rows[key].extend(res.get(key, []))
    for lake, err in failed:
        _log(f"  失败：{lake} — {err}")

    outputs = {"full": pd.DataFrame(rows["full_rows"]),
               "rolling": pd.DataFrame(rows["rolling_rows"]),
               "dm": F._apply_dm_fdr(pd, pd.DataFrame(rows["dm_rows"])),
               "selected": pd.DataFrame(rows["selected_rows"])}
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, frame in outputs.items():
        path = out_dir / F.OUTPUT_NAMES[key]
        frame.to_csv(path, index=False)
        _log(f"已写出 {path}（{len(frame)} 行）")


# ---------------------------------------------------------------- figures
def stage_figures():
    import runpy
    scripts = sorted((CODE / "06_figures").glob("figure_*.py")) + \
              sorted((CODE / "07_tables").glob("*.py"))
    for script in scripts:
        _log(f"运行 {script.name}")
        try:
            runpy.run_path(str(script), run_name="__main__")
        except Exception as exc:                      # 单张图失败不中断其余
            _log(f"  跳过（{type(exc).__name__}: {exc}）")


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
        _log(f"===== 阶段 {st} =====")
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
    _log("全部完成")


if __name__ == "__main__":
    main()
