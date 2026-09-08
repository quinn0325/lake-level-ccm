"""全流程唯一配置来源。

所有关于"用哪些湖泊、扫哪些滞后、读哪个结果文件"的选择集中在此。
其他脚本一律从这里读，不再各自写死——这是为了根除
"图用 A 分支、预测用 B 分支"这类不一致（见 docs/排查问题清单.md A-2/A-3）。
"""

from pathlib import Path

# ---------------------------------------------------------------- 路径
CODE_DIR = Path(__file__).resolve().parent
PKG_DIR = CODE_DIR.parent
RESULTS_DIR = PKG_DIR / "results"
PKL_DIR = PKG_DIR / "lake_pkls"

# ---------------------------------------------------------------- 研究对象
LAKES = [
    # Okanagan 系统（不列颠哥伦比亚省）
    "Kalamalka_Lake", "Okanagan_Lake", "Skaha_Lake", "Vaseux_Lake",
    # Nelson–Winnipeg 系统（曼尼托巴省 / 安大略省）
    "Rainy_Lake", "Lake_of_the_Woods", "Playgreen_Lake",
    "Kiskitto_Lake", "Sipiwesk_Lake", "Split_Lake",
]

VARIABLES = ["WL", "T", "P", "R", "SWE", "Evap", "RegFlow"]

# 有直接水道连接的湖泊对（7 对），用于连通性对照
WATERWAY_CONNECTED_PAIRS = [
    ("Kalamalka_Lake", "Okanagan_Lake"),
    ("Okanagan_Lake", "Skaha_Lake"),
    ("Skaha_Lake", "Vaseux_Lake"),
    ("Rainy_Lake", "Lake_of_the_Woods"),
    ("Playgreen_Lake", "Sipiwesk_Lake"),
    ("Kiskitto_Lake", "Sipiwesk_Lake"),
    ("Sipiwesk_Lake", "Split_Lake"),
]

# ---------------------------------------------------------------- 切分
FORECAST_HORIZON = 37          # 约 9:1 切分；训练/测试分界，CCM 也只用训练段
ROLLING_HORIZONS = [1, 3, 6, 12]

# ---------------------------------------------------------------- 嵌入参数
# τ 固定为 1：与 Javier et al. (2022, Physica A 604, 127893) 一致
#   ——"We fixed τ = 1 because this gives the highest resolution for the embedding"。
# 固定 τ 同时消除了 τ 选择环节的测试期信息泄漏，并把嵌入跨度
# (E-1)*τ 从最长 36 个月压到最长 (E_MAX-1)=9 个月，显著提高有缺测序列上的可用样本量。
# E 仍由单变量 simplex 自预测选择（不用交叉映射技巧选 E，避免用被检验量选参数的循环论证）。
EMBED_TAU = 1
EMBED_E_CANDIDATES = range(2, 11)   # E ∈ [2, 10]

# ---------------------------------------------------------------- 滞后族
# 主分析对每条边在 -12..+12 上做一次完整的带符号扫描，湖内湖间统一。
# 观测数据与每个 IAAFT 替代序列走同一滞后族（见 ccm_full_pipeline.CCM_LAGS，
# 那里是 max_over_lags 默认值的唯一来源）。
CAUSAL_LAGS = range(-12, 13)

# 预测特征的滞后域：d=0 需要预测起点当期的真实值，不构成可用预测信息。
# 因果分析与预测特征构建是两个预先定义的优化问题，各自在自己的域内求最优
# （forecast-constrained optimal lag，见 ccm_full_pipeline.forecast_constrained_lag）。
FORECAST_LAGS = range(1, 13)

# ---------------------------------------------------------------- 显著性
N_SURROGATES = 500
IAAFT_N_ITER = 100
FDR_ALPHA = 0.05

# ---------------------------------------------------------------- 缺失值
MAX_FILLABLE_GAP_MONTHS = 6    # 预测阶段：<=6 个月的连续缺口才插补，更长则整个变量剔除
OUTLIER_Z_THRESH = 6
OUTLIER_LEVEL_Z_THRESH = 5

# ---------------------------------------------------------------- 权威结果文件
# 图表与预测一律从这里取文件名，禁止在脚本内写死。
EMBED_PARAMS_FILE = RESULTS_DIR / "embed_params_corrected.json"

# 说明：本文件目前仅由 00_data_generation/ 下的脚本读取（ccm_lib、recompute_embed_params）。
# 其余脚本仍各自维护常量副本——包内湖泊清单有多份、变量清单有多份。
# 这是已知的结构性问题（见 docs/全包审查_修复覆盖面_0828.md），
# 尚未统一，勿误以为此处修改会自动传播到全流程。
