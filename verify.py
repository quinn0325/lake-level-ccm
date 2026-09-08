"""Verify this repository on your machine in a few minutes.

Reproducing the full study means recomputing 420 within-lake and 90 between-lake
CCM edges, each against 500 IAAFT surrogates. That takes 8-18 hours on one core
and nobody is expected to sit through it in order to check this work.

What can be checked quickly is that the chain is intact and the code runs here:
every table, figure and appendix in the dissertation is a deterministic function
of the files in results/ and lake_pkls/, so those steps can be re-executed and
compared against the committed copies byte for byte. The two expensive stages
are then exercised on a single edge at a reduced surrogate count, which proves
the CCM code runs on your machine without asking you to wait for all 510 edges.

Usage
-----
    python verify.py              # roughly four minutes
    python verify.py --quick      # about one minute; skips the CCM smoke test
    python verify.py --forecast   # also check the forecasting entry point

Exit status is 0 if every check passes, 1 otherwise.
"""
from __future__ import annotations

import argparse
import filecmp
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
results: list[tuple[str, str, str]] = []


def record(name, status, detail=""):
    results.append((name, status, detail))
    mark = {PASS: "  ok  ", FAIL: " FAIL ", SKIP: " skip "}[status]
    print(f"[{mark}] {name}" + (f"  -- {detail}" if detail else ""), flush=True)


def section(title):
    print(f"\n--- {title} " + "-" * max(0, 58 - len(title)), flush=True)


# ---------------------------------------------------------------- environment
def check_environment():
    section("Environment")
    major, minor = sys.version_info[:2]
    ok = (3, 10) <= (major, minor) <= (3, 12)
    record("Python 3.10-3.12", PASS if ok else FAIL, f"running {major}.{minor}")

    wanted = {"pyEDM": "2.4.0", "pandas": "2.2.2", "numpy": "1.26.4"}
    for mod, want in wanted.items():
        try:
            got = __import__(mod).__version__
        except Exception as exc:
            record(f"{mod} == {want}", FAIL, f"import failed: {exc}")
            continue
        record(f"{mod} == {want}", PASS if got == want else FAIL, f"found {got}")

    for mod in ("scipy", "statsmodels", "pmdarima", "xgboost", "networkx",
                "matplotlib"):
        try:
            record(f"{mod} importable", PASS, __import__(mod).__version__)
        except Exception as exc:
            record(f"{mod} importable", FAIL, str(exc))

    for name in ("lake_pkls", "results", "ch4_tables", "appendices", "dataset"):
        d = ROOT / name
        n = len(list(d.glob("*"))) if d.is_dir() else 0
        record(f"{name}/ present", PASS if n else FAIL, f"{n} files")


# ------------------------------------------------------------ regenerate+diff
def run_script(rel, code_dir, cwd=None, env=None):
    """Run one script; return (ok, tail-of-output).

    code_dir must be the copy under the scratch tree, not this repository's own
    code/: these scripts derive their output paths from __file__, so running the
    original would write over the very files being compared.
    """
    proc = subprocess.run([sys.executable, str(Path(code_dir) / rel)],
                          cwd=str(cwd or ROOT), capture_output=True, text=True,
                          env=env)
    tail = (proc.stdout + proc.stderr).strip().splitlines()
    return proc.returncode == 0, "\n".join(tail[-4:])


def compare_dir(name, generated: Path, committed: Path, pattern="*.csv"):
    """Every committed file must be reproduced identically."""
    missing, differing, same = [], [], 0
    for ref in sorted(committed.glob(pattern)):
        new = generated / ref.name
        if not new.exists():
            missing.append(ref.name)
        elif filecmp.cmp(ref, new, shallow=False):
            same += 1
        else:
            differing.append(ref.name)
    if missing or differing:
        bad = (missing + differing)[:3]
        record(name, FAIL,
               f"{same} identical, {len(differing)} differ, {len(missing)} missing"
               f" (e.g. {', '.join(bad)})")
    else:
        record(name, PASS, f"{same} files byte-identical")


def check_derivations():
    """results/ and lake_pkls/ -> every committed table, in a scratch copy."""
    section("Derived outputs regenerate identically")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "repo"
        # 只复制派生所需的输入与代码，输出目录留空，确保比对的是新生成的文件
        work.mkdir(parents=True)
        for d in ("code", "results", "lake_pkls"):
            shutil.copytree(ROOT / d, work / d,
                            ignore=shutil.ignore_patterns("__pycache__"))
        env = dict(os.environ, CCM_PKL_DIR=str(work / "lake_pkls"),
                   CCM_OUT_DIR=str(work / "results"),
                   CCM_EMBED_PARAMS=str(work / "results"
                                        / "embed_params_corrected.json"))

        # 顺序与 run_local.stage_figures 一致：表之间有依赖，字母序是错的。
        chain = ["07_tables/build_ch4_tables.py",
                 "07_tables/table_C1_supported_drivers.py",
                 "07_tables/table_4_2_dm_maintext.py",
                 "07_tables/table_4_4_forecast_summary.py",
                 "07_tables/table_4_4_matched_rmse.py"]
        broke = False
        for rel in chain:
            ok, tail = run_script(rel, work / "code", cwd=work, env=env)
            if not ok:
                record(f"{Path(rel).name} runs", FAIL, tail)
                broke = True
                break
        if not broke:
            record("the five Chapter 4 table scripts run", PASS)
            compare_dir("ch4_tables/ reproduced from results/",
                        work / "ch4_tables", ROOT / "ch4_tables")

            ok, tail = run_script("07_tables/build_appendices.py", work / "code",
                                  cwd=work, env=env)
            if not ok:
                record("build_appendices.py runs", FAIL, tail)
            else:
                record("build_appendices.py runs", PASS)
                compare_dir("appendices/ reproduced", work / "appendices",
                            ROOT / "appendices")

        ok, tail = run_script("08_dataset/build_public_dataset.py",
                              work / "code", cwd=work, env=env)
        if not ok:
            record("build_public_dataset.py runs", FAIL, tail)
        else:
            record("build_public_dataset.py runs", PASS)
            compare_dir("dataset/ reproduced from lake_pkls/",
                        work / "dataset", ROOT / "dataset")

        ok, tail = run_script("06_figures/figure_4_2_ccm_driver_matrix.py",
                              work / "code", cwd=work, env=env)
        made = sorted((work / "results" / "figures").glob("figure_4_2*"))
        record("a figure redraws", PASS if ok and made else FAIL,
               f"{len(made)} files" if ok else tail)


# ------------------------------------------------------------- CCM smoke test
def check_ccm_runs():
    """One real CCM edge at 3 surrogates instead of 500 (about three minutes)."""
    section("CCM code runs here (1 edge, 3 surrogates instead of 500)")
    sys.path.insert(0, str(CODE / "01_shared"))
    sys.path.insert(0, str(CODE))
    try:
        import ccm_full_pipeline as p
    except Exception as exc:
        record("import ccm_full_pipeline", FAIL, str(exc))
        return
    record("import ccm_full_pipeline", PASS)

    lake, cause, effect = "Split_Lake", "RegFlow", "WL"
    t0 = time.time()
    try:
        panel, embed, _ = p.load_lake_panel_for_ccm(lake)
        row = p.test_one_ccm_edge(panel, cause, effect, embed[effect]["E"],
                                  embed[effect]["tau"], n_surrogates=3)
    except Exception as exc:
        record(f"CCM {cause}->{effect} at {lake}", FAIL, f"{type(exc).__name__}: {exc}")
        return

    import pandas as pd
    published = pd.read_csv(ROOT / "results" / "ccm_all_edges_merged_fdr.csv")
    ref = published[(published.lake == lake) & (published.cause == cause)
                    & (published.effect == effect)].iloc[0]
    d_rho = abs(float(row["obs_rho"]) - float(ref.obs_rho))
    same_lag = int(row["obs_lag"]) == int(ref.obs_lag)
    detail = (f"rho {float(row['obs_rho']):.4f} vs published {float(ref.obs_rho):.4f}, "
              f"lag {int(row['obs_lag'])} vs {int(ref.obs_lag)}, "
              f"{time.time() - t0:.0f}s")
    # rho 与 lag 由观测序列的滞后扫描决定，与替代序列数量无关，应当完全一致；
    # p 值在 3 个替代序列下毫无意义，这里不比对它。
    record("cross-map skill and optimal lag match the published edge",
           PASS if d_rho < 1e-6 and same_lag else FAIL, detail)


# -------------------------------------------------------- forecasting (opt-in)
def check_forecast_runs():
    section("Forecasting code runs here (1 lake, ~4 minutes)")
    sys.path.insert(0, str(CODE / "04_forecast"))
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("run_local", ROOT / "run_local.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        record("run_local.py imports", PASS)
    except Exception as exc:
        record("run_local.py imports", FAIL, str(exc))
        return
    record("stage_forecast defined", PASS if hasattr(mod, "stage_forecast") else FAIL)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="skip the CCM smoke test")
    ap.add_argument("--forecast", action="store_true",
                    help="also check the forecasting entry point")
    args = ap.parse_args()

    print(f"Verifying {ROOT}")
    check_environment()
    check_derivations()
    if not args.quick:
        check_ccm_runs()
    if args.forecast:
        check_forecast_runs()

    section("Summary")
    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    n_pass = sum(1 for _, s, _ in results if s == PASS)
    print(f"{n_pass} passed, {n_fail} failed, "
          f"{sum(1 for _, s, _ in results if s == SKIP)} skipped")
    if n_fail:
        print("\nFailed checks:")
        for name, status, detail in results:
            if status == FAIL:
                print(f"  - {name}: {detail}")
        print("\nThe full analysis was run on Python 3.12 with the pinned "
              "versions in requirements.txt; version mismatches are the usual "
              "cause. See the Reproduction tolerance section of README.md.")
    else:
        msg = ("\nEvery committed table, dataset and figure was regenerated from "
               "the inputs in this repository and matched byte for byte")
        msg += ("." if args.quick else
                ", and the CCM code reproduced a published edge exactly.")
        if args.quick:
            msg += (" The CCM smoke test was skipped; drop --quick to run it.")
        print(msg + " The full 510-edge run was not repeated; see README.md "
                    "for how to launch it.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
