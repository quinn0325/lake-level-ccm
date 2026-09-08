"""
数据质量筛查(第6条:结构性断点/趋势检测)——用Pettitt检验(Pettitt, 1979, "A
non-parametric approach to the change-point problem", Journal of the Royal
Statistical Society Series C)，水文时间序列检测突变点的标准非参数方法(比如
Kong, Shi, Yao & Liu 2025那篇我们之前读过的水文因果分析论文所在的领域，change-point
detection也是常见的前置步骤)。

对WL和RegFlow(去季节化后，先剔除季节性再测，不然季节性本身的年度循环会干扰检测)
分别测每个湖泊，找有没有真实的、非季节性的水平阶跃——比如水坝启用/调度规则变化
可能导致RegFlow或WL出现永久性的均值偏移，这种不应该被当成异常值剔除(它是真实的、
持续的信号，不是孤立的坏点)，但也不应该被deseasonalize()的"多年月均值"简单地
平均掉(混合了断点前后两种不同状态的均值，两边都不准)。
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


def combine_station_water_levels(wide, method="anomaly_mean"):
    wide = wide.sort_index()
    station_means = wide.mean(axis=0, skipna=True)
    global_mean = station_means.mean(skipna=True)
    return (wide - station_means).mean(axis=1, skipna=True) + global_mean


def deseasonalize(series):
    monthly_clim = series.groupby(series.index.month).transform("mean")
    return series - monthly_clim


def pettitt_test(x):
    """Pettitt (1979)非参数突变点检验。x是一维、无缺测的序列(有缺测先dropna)。
    返回(最可能的突变点位置索引, K统计量, 近似p值)。"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 20:
        return None, np.nan, np.nan

    # U_t = sum_{i<=t} sum_{j>t} sign(x_i - x_j)，用秩次的高效算法实现(等价于原始定义)
    ranks = pd.Series(x).rank().values
    U = np.cumsum(2 * ranks - n - 1)
    K = np.max(np.abs(U))
    t_star = int(np.argmax(np.abs(U)))  # 0-indexed，最可能的突变点位置

    # Pettitt (1979)原文给出的近似p值公式
    p_approx = 2 * np.exp(-6 * K**2 / (n**3 + n**2))
    p_approx = min(1.0, p_approx)
    return t_star, float(K), float(p_approx)


def main():
    print("=" * 70)
    print("数据质量筛查报告(第6条:结构性断点检测，Pettitt检验)")
    print("=" * 70 + "\n")

    all_rows = []          # 汇总为表，供方法论/局限部分引用
    for lake in LAKES:
        with open(f"{PKL_DIR}/{lake}_result.pkl", "rb") as f:
            cached = pickle.load(f)

        combined_wl = combine_station_water_levels(cached["wide_wl"], method="anomaly_mean")
        wl_deseason = deseasonalize(combined_wl.sort_index().asfreq("MS")).dropna()

        findings = []

        t_star, K, p = pettitt_test(wl_deseason.values)
        if t_star is not None and p < 0.05:
            break_date = wl_deseason.index[t_star]
            before_mean = wl_deseason.iloc[:t_star + 1].mean()
            after_mean = wl_deseason.iloc[t_star + 1:].mean()
            findings.append(("WL", break_date, p, before_mean, after_mean))

        rp = cached["real_predictors"]
        if "RegFlow" in rp.columns:
            rf = rp["RegFlow"].sort_index().asfreq("MS")
            rf_deseason = deseasonalize(rf).dropna()
            t_star, K, p = pettitt_test(rf_deseason.values)
            if t_star is not None and p < 0.05:
                break_date = rf_deseason.index[t_star]
                before_mean = rf_deseason.iloc[:t_star + 1].mean()
                after_mean = rf_deseason.iloc[t_star + 1:].mean()
                findings.append(("RegFlow", break_date, p, before_mean, after_mean))

        if findings:
            print(f"### {lake} ###")
            for var, date, p, before, after in findings:
                print(f"  [{var}] 最可能的突变点: {date.date()}, p={p:.4f}(<0.05显著)")
                print(f"      断点前均值={before:.4f}, 断点后均值={after:.4f}, 偏移量={after-before:.4f}")
                all_rows.append({
                    "lake": lake, "variable": var,
                    "break_date": date.date().isoformat(),
                    "p_value": round(float(p), 6),
                    "mean_before": round(float(before), 4),
                    "mean_after": round(float(after), 4),
                    "shift": round(float(after - before), 4),
                })
            print()

    print("=" * 70)
    print("说明: 只报告p<0.05(Pettitt检验显著)的突变点; 检测对象是去季节化后的序列")
    print("(已经去掉正常季节循环，剩下的突变反映的是真实的、非季节性的水平转变)")
    print("=" * 70)

    # 写出表格：本结果用于方法论中"受调控系统对 CCM 递归性假设的影响"一节，
    # 需要随包提交、可被正文引用，不能只停留在终端输出。
    if all_rows:
        out_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), "results")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, "data_qa_structural_breaks.csv")
        pd.DataFrame(all_rows).to_csv(out, index=False)
        print(f"\n已写出 {out}（{len(all_rows)} 处断点）")

        # 按年月聚集情况——断点若在时间与空间上聚集，指向系统层面的调度规则变更，
        # 而非孤立的数据问题；这正是方法论讨论所需的证据。
        cl = pd.DataFrame(all_rows).groupby(
            pd.DataFrame(all_rows)["break_date"].str[:7]).size().sort_values(ascending=False)
        print("\n按年月聚集(前5):")
        for ym, n in cl.head(5).items():
            print(f"  {ym}: {n} 处")


if __name__ == "__main__":
    main()
