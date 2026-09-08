"""
数据质量筛查(第3条:异常值/物理合理性检测)——修正版，不再用粗糙的"全年混合中位数+MAD"，
那种方法对R/SWE这类强季节性、右偏变量会把正常的季节性峰值(融雪期径流、冬季积雪)
大量误判成异常值(之前跑出来的假警报已经验证过：R异常值全部落在4/5/6月融雪季，
SWE异常值全部落在1/2/3/4/12月积雪季，都是正常季节性峰值不是数据问题)。

这版按变量类型分别处理：
1. WL(水位)：湖泊水位应该平滑变化，用"月度变化量(一阶差分)"的稳健异常检测，
   抓"孤立骤变、前后又恢复正常"这种模式——这是普通水位季节波动不会有的信号，
   比直接对水位原始值做异常检测更准确。
2. P/R/SWE/Evap/RegFlow(非负物理量)：先查物理边界(不能是负值)，再按"同月份历史"
   (只拿全部5月互相比，不跟全年混在一起比)做稳健异常检测，这样正常的季节性峰值
   不会被误判——因为异常检测的参照系变成了"这个月历史上通常什么样"，不是"全年
   普遍什么样"。
3. T(气温)：同样按同月份历史比较。

只做检测和报告，不自动修改任何缓存数据——每一处发现都打印出来，由你决定怎么处理
(排除、修正、还是保留但标注)，这本身也是"异常处理要留痕"(清单第8条)的一部分。
"""

import pickle

import numpy as np
import pandas as pd

LAKES = [
    "Kalamalka_Lake", "Okanagan_Lake", "Skaha_Lake", "Vaseux_Lake",
    "Rainy_Lake", "Lake_of_the_Woods", "Playgreen_Lake",
    "Kiskitto_Lake", "Sipiwesk_Lake", "Split_Lake",
]

import os
# 早期开发时的临时路径 "/tmp/lake_pkls" 在他人机器上不存在。
# 改为包内 lake_pkls/（可用环境变量 CCM_PKL_DIR 覆盖），使本脚本随包可跑。
PKL_DIR = os.environ.get(
    "CCM_PKL_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "lake_pkls"),
)
NONNEGATIVE_VARS = {"P", "R", "SWE", "Evap", "RegFlow"}
Z_THRESH = 6  # 稳健z分数阈值(用1.4826*MAD近似标准差)，6倍算是比较保守，不轻易误报


def seasonal_robust_outliers(series, z_thresh=Z_THRESH):
    """按月份分别算稳健统计量，只跟同月份历史比。"""
    flagged_parts = []
    for month in range(1, 13):
        month_vals = series[series.index.month == month].dropna()
        if len(month_vals) < 6:  # 同月份历史样本太少，稳健统计量不可靠，跳过
            continue
        med = month_vals.median()
        mad = (month_vals - med).abs().median()
        if mad == 0:
            continue
        robust_z = (month_vals - med).abs() / (1.4826 * mad)
        flagged_parts.append(month_vals[robust_z > z_thresh])
    if flagged_parts:
        return pd.concat(flagged_parts).sort_index()
    return pd.Series(dtype=float)


def wl_jump_outliers(series, z_thresh=Z_THRESH, level_z_thresh=5):
    """WL专用：先找月度变化量(一阶差分)的稳健异常，再用"这个月的绝对水位值相对该站点
    整体历史分布是不是也异常"做二次确认——只标"骤变+数值本身也是历史级异常"的点，
    过滤掉"变化速度快但数值仍在正常范围内"的情况(比如每年融雪期水位快速上涨，
    在多个不同年份的4-6月反复出现同一种模式，这更像是真实的、只是幅度较大的季节性
    现象，不是数据错误——真正的数据错误应该是这个数值本身在站点历史上从未出现过，
    不只是"涨得比平时快")。"""
    diffs = series.diff().dropna()
    if len(diffs) < 10:
        return pd.Series(dtype=float)
    med = diffs.median()
    mad = (diffs - med).abs().median()
    if mad == 0:
        return pd.Series(dtype=float)
    robust_z = (diffs - med).abs() / (1.4826 * mad)
    flagged_diff_dates = set(diffs[robust_z > z_thresh].index)
    flagged_months = set()
    for d in flagged_diff_dates:
        flagged_months.add(d)
        prev = d - pd.DateOffset(months=1)
        flagged_months.add(prev)
    flagged_months = sorted(m for m in flagged_months if m in series.index)
    candidates = series.loc[flagged_months]

    # 二次确认：数值本身相对站点整体历史分布是不是也是异常(不只是变化快)
    level_med = series.median()
    level_mad = (series - level_med).abs().median()
    if level_mad == 0:
        return candidates
    level_robust_z = (candidates - level_med).abs() / (1.4826 * level_mad)
    return candidates[level_robust_z > level_z_thresh]


NEG_TOLERANCE = -0.01  # 小于这个才算真的负值，排除浮点数舍入误差(比如-0.0000这种)


def main():
    print(f"{'='*70}")
    print("数据质量筛查报告(第3条:异常值/物理合理性检测，修正版)")
    print(f"{'='*70}\n")

    total_findings = 0
    all_records = []  # 用来做跨湖泊聚合，找系统性(不是单个湖泊孤立)的问题

    for lake in LAKES:
        with open(f"{PKL_DIR}/{lake}_result.pkl", "rb") as f:
            cached = pickle.load(f)

        lake_findings = []

        # ---- 1. WL站点：月度变化量异常检测(已加数值本身也要历史级异常这道二次确认) ----
        wide = cached["wide_wl"]
        for col in wide.columns:
            s = wide[col].dropna()
            outliers = wl_jump_outliers(s)
            if len(outliers):
                lake_findings.append(("WL站点" + col, "骤变+数值本身也历史级异常", outliers))
                for idx, v in outliers.items():
                    all_records.append({"lake": lake, "var": "WL", "kind": "骤变", "date": idx, "value": v})

        # ---- 2. real_predictors各变量 ----
        rp = cached["real_predictors"]
        for col in rp.columns:
            s = rp[col].dropna()
            if len(s) < 10:
                continue

            # 物理边界(非负变量不能是负值，留一个小容差排除浮点数舍入误差)
            if col in NONNEGATIVE_VARS:
                neg = s[s < NEG_TOLERANCE]
                if len(neg):
                    lake_findings.append((col, "物理边界违反(真负值，非浮点误差)", neg))
                    for idx, v in neg.items():
                        all_records.append({"lake": lake, "var": col, "kind": "负值", "date": idx, "value": v})

            # 同月份历史异常检测
            outliers = seasonal_robust_outliers(s)
            if len(outliers):
                lake_findings.append((col, "同月历史异常", outliers))
                for idx, v in outliers.items():
                    all_records.append({"lake": lake, "var": col, "kind": "同月异常", "date": idx, "value": v})

        if lake_findings:
            print(f"### {lake} ###")
            for var_name, kind, vals in lake_findings:
                print(f"  [{var_name}] {kind}: {len(vals)}个可疑点")
                for idx, v in vals.items():
                    print(f"      {idx.date()}: {v:.4f}")
                total_findings += len(vals)
            print()

    print(f"{'='*70}")
    print(f"总计发现 {total_findings} 个可疑数据点(已排除SWE浮点误差和季节性峰值误报)")
    print(f"{'='*70}\n")

    # ---- 跨湖泊聚合：同一个变量、同一个(年,月)组合，如果在很多个不同湖泊里都被
    # 标记，说明这大概率不是某个湖泊自己的数据问题，而是上游数据源(比如ERA5)在
    # 那个时间段本身有系统性异常，处理方式应该不一样(不是排除某个湖，而是要去查
    # 那批ERA5数据本身) ----
    if all_records:
        df = pd.DataFrame(all_records)
        df["year_month"] = df["date"].dt.to_period("M")
        print("=== 跨湖泊聚合：同一个变量+同一个月份，在几个不同湖泊里同时被标记 ===")
        grp = df.groupby(["var", "year_month"])["lake"].nunique().sort_values(ascending=False)
        systematic = grp[grp >= 4]  # 同一个月份，4个或以上湖泊同时被标记，怀疑是系统性问题
        if len(systematic):
            for (var, ym), n_lakes in systematic.items():
                lakes_involved = sorted(df[(df["var"] == var) & (df["year_month"] == ym)]["lake"].unique())
                print(f"  [{var}] {ym}: {n_lakes}个湖泊同时被标记 -> {lakes_involved}")
            print("\n  这些很可能不是单个湖泊的数据问题，是那个月份/那个变量的上游数据源"
                  "(ERA5等)本身有系统性异常，建议单独去查那段时间的原始ERA5数据，不要"
                  "按'排除单个湖泊'的方式处理。")
        else:
            print("  (没有发现4个以上湖泊同时被标记同一个月份+变量的情况)")


if __name__ == "__main__":
    main()
