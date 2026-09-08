"""
Modal版CCM湖泊分析流水线——把CCM_Lake_Pipeline_v3_pairwise_complete.ipynb里process_lake()
按湖泊拆成并行Modal任务（15个湖泊，最多15个容器同时跑，比Colab里顺序跑一个湖泊接一个湖泊快得多）。
CCM缺测处理是pairwise-complete版本，见ccm_lib.py顶部说明。

用法：
  modal run ccm_modal_app.py                 # 跑全部15个LAKES
  modal run ccm_modal_app.py --lakes Kalamalka_Lake,Okanagan_Lake   # 只跑指定几个

跑法分两步（main()里自动做，不用手动分开调用）：
  第1步 fetch_era5_modal：低并发(max_containers=4，内部带退避重试)预下载ERA5数据，避免15个湖泊同时打CDS API
        撞上"Number queued requests for this dataset is temporarily limited"账号级限流；
  第2步 process_lake_modal：ERA5已经缓存到Volume，逐湖并行构建面板与嵌入参数，
        写出 {lake}_result.pkl。CCM 与预测不在这里跑，见 02_/03_/04_。

也可以用 fetch_only / process_only 分开手动跑（见各自函数 docstring）。
main() 一条命令跑完两步，但要求本地进程全程存活（配合 caffeinate）。

前置条件（这两步你自己在终端做，不要把密钥发进对话）：
  modal setup
  modal secret create cds-api CDSAPI_URL=... CDSAPI_KEY=...

第一次跑之前，需要把本地HydroLAKES/HydroBASINS shapefile上传到Modal Volume：
  modal volume create ccm-data
  modal volume put ccm-data <local-path> HydroLAKES_polys_v10_shp
  modal volume put ccm-data /path/to/hybas_na_lev12_v1c hybas_na_lev12_v1c
"""
import modal

app = modal.App("ccm-lakes")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgdal-dev", "gdal-bin")
    .pip_install(
        "pandas==2.2.2", "numpy==1.26.4",  # 锁版本跟本机(3.10)一致，否则.map()结果传回本地反序列化会因为
        # pandas内部dtype表示不兼容而报DeserializationError（已经踩过这个坑）
        "geopandas", "shapely", "requests",
        "pmdarima", "statsmodels", "networkx", "xarray", "netcdf4",
        "cdsapi", "matplotlib", "pyEDM", "scipy", "xgboost",
    )
    .add_local_python_source("ccm_lib")  # 显式把ccm_lib.py打进镜像，不依赖自动挂载
)

volume = modal.Volume.from_name("ccm-data", create_if_missing=True)
DATA_ROOT = "/data"

VARS = ["WL", "T", "P", "R", "SWE", "Evap", "RegFlow"]
START_YEAR, END_YEAR = 1994, 2024
PIPELINE_VERSION = "2026-08-09_v14_forecast_horizon_37mo_revert"  # v8有bug：panel_deseason
# 在build_lake_panel/process_lake_modal里还是被整体dropna了一次，ccm_train_panel继承了这个
# listwise交集，pairwise-complete在per-edge层面形同虚设(全部边n_obs一样)。v9修复了这一行。
# v10：修复Lake_Huron的Hylak_id解析错误——之前默认country="Canada"过滤掉了HydroLAKES里
# 正确的跨境湖记录(Hylak_id=8,59399km²)，误匹配到旁边一个不相关的0.2km²小水域(见
# LAKE_RESOLVE_COUNTRY)。升版本号触发全部15个湖泊重算：Lake_Huron用正确掩膜得到新结果，
# 其余14个湖泊掩膜/站点未变，重算结果预期与v9一致，用于交叉验证。
# v11：ccm_lib.py整体重写(2026-08-09)，修复_pair_panel/surrogate_significance/lag_scan/
# get_embedding_params一直在做的dropna+reset_index——这一步会把物理上不相邻(缺测拼接处)
# 的两个观测点错误地当成相邻处理，改成保留完整日历月份索引+真实NaN，靠pyEDM默认的
# ignoreNan=True正确跳过需要缺测点的嵌入/预测(细节见ccm_lib.py顶部说明)。这个改动会让
# 每条边的rho数值发生变化，尤其是缺测拼接严重的边(Crooked_Lake/Dauphin_Lake/Split_Lake/
# Kootenay_Lake等)，升版本号触发全部湖泊重算是必要的，不是可选的。
# v12：FORECAST_HORIZON从12改成37(9:1切分，约3年)，不再只留出2024一年做测试/预测验证——
# 已证实Split_Lake/Rainy_Lake的出口流量站2024整年记录为0，只用2024做测试集风险太大。
# 这个值同时决定ccm_train_panel的训练窗口，CCM本身的训练数据量也跟着变了，升版本号触发
# 全部湖泊重算。
# v13：FORECAST_HORIZON从37改成74(8:2切分)，对比一下换成机器学习里更常见的80/20比例后
# CCM结果稳不稳。74个月的评估区间也给后续12个月预见期的滚动窗口评估留出更多起点
# (h=12时约62个起点，比37个月区间下的25个稳健得多)。
# v14：改回37(9:1切分)——74个月(8:2)训练集数据量偏少，权衡后改回9:1。同时74个月窗口
# 下暴露的两个问题(Okanagan RegFlow滞后由+5翻转为-1触发同步假象过滤、Split_Lake测试集
# 起点撞上2018年数据缺口导致XGBoost全部失败)预期会随窗口缩短而消失。
FORECAST_HORIZON = 37  # 9:1切分，固定月份数、不按各湖面板长度动态算
# 10%，保证全部湖泊用同一条时间线切分，便于跨湖比较预测表现。原来是12(只留最后一年)，
# 单独用2024一年做测试集风险太大——已证实Split_Lake/Rainy_Lake的出口流量站2024整年
# 记录为0，测试集完全押在这一年上不合理。改成37个月后2024年只占测试集约1/3，问题被
# 稀释但没有完全消除，这一点需要在方法论限制里如实说明。
# 这个值同时决定了ccm_train_panel的训练窗口(process_lake_modal里`panel_deseason.iloc[
# :-FORECAST_HORIZON]`)，不只是SARIMAX/XGBoost预测测试集——改这个值会让CCM本身的
# 训练数据量变化，触发下面PIPELINE_VERSION升级、全部湖泊需要重新计算。

# RESERVOIR_LAKES: 查HydroLAKES自己的Lake_type字段(1=天然湖泊, 2=水库/GRanD大坝人工造的,
# 3=天然湖泊但出口被人工控制)核实过，这6个是Lake_type=2的真水库，不是"天然湖泊出口被调控"
# (那种是type=3，比如Rainy_Lake/Southern_Indian_Lake/Kootenay_Lake，仍然算湖泊)。论文范围
# 是天然湖泊，不含水库，这6个留在LAKES/REGULATION_STATIONS里是因为CCM已经跑过、数据本身没问题，
# 只是不进最终论文的湖泊清单——下游生成"官方"结果表时用这个集合过滤掉，不要从字典里物理删除
# (万一以后要做"湖泊vs水库"对比分析还能直接复用已经算好的结果)。
RESERVOIR_LAKES = {
    "Nickle_Lake", "Grant_Devine_Lake", "Lake_of_the_Prairies",       # 原15个里的3个(GRanD id 723/724/713)
    "Tobin_Lake", "Sugar_Lake_Reservoir", "Coquitlam_Lake",           # 新5个里的3个(GRanD id 710/265/34)
}

# 15个湖泊，跟CCM_Lake_Pipeline_v3_pairwise_complete.ipynb的LAKES字典保持一致。
LAKES = {
    "Lake_of_the_Woods":    dict(group="Boreal Shield", rank=1, stations=["05PD029", "05PD008", "05PE014", "05PD011"]),
    "Rainy_Lake":           dict(group="Boreal Shield", rank=2, stations=["05PB024", "05PB007"]),
    "Southern_Indian_Lake": dict(group="Boreal Shield", rank=3, stations=["06EC001", "06EC003", "06EC007"]),
    "Lake_Nipissing":       dict(group="Boreal Shield", rank=4, stations=["02DD006"]),
    "Lake_Huron":           dict(group="Boreal Shield", rank=5, stations=["02EA014"]),
    "Nickle_Lake":          dict(group="Prairie", rank=1, stations=["05NB020"]),
    "Last_Mountain_Lake":   dict(group="Prairie", rank=2, stations=["05JH004"]),
    "Katepwa_Lake":         dict(group="Prairie", rank=3, stations=["05JL004"]),
    "Grant_Devine_Lake":    dict(group="Prairie", rank=4, stations=["05ND012"]),
    "Lake_of_the_Prairies": dict(group="Prairie", rank=5, stations=["05MD009"]),
    "Okanagan_Lake":        dict(group="Montane Cordillera", rank=1, stations=["08NM083"]),
    "Vaseux_Lake":          dict(group="Montane Cordillera", rank=2, stations=["08NM243"]),
    "Kootenay_Lake":        dict(group="Montane Cordillera", rank=3, stations=["08NH067", "08NH064"]),
    "Kalamalka_Lake":       dict(group="Montane Cordillera", rank=4, stations=["08NM143"]),
    "Nicola_Lake":          dict(group="Montane Cordillera", rank=5, stations=["08LG046"]),
    # 2026-08新增5个湖泊：不再按生态区配额选取，改成全加拿大范围内系统性筛选——水位站
    # 逐年完整度≥90%且1994-2024无整年缺失、配对调控流量站在HYDAT里明确标记为regulated、
    # 站名方位关键词确认是目标湖泊自己的出口(而非入流或下游多湖混合)、流量站集水面积与
    # 水位站基本一致。核实过程见audit记录，这里group统一标"New_2026"，不参与生态区连通性
    # 分析(connectivity_ccm_ecoregion按group分组两两配对，这5个互相之间没有已知水力连通)。
    # 原本还有Prosperous_Lake(NT，07SB014)，核实到HydroBASINS把北美大陆按流向分成"na"
    # (排入太平洋/大西洋/墨西哥湾)和"ar"(北极排水区)两个独立shapefile，Prosperous Lake属于
    # Mackenzie River/Great Slave Lake流域，在"ar"区域，项目目前只有"na"区域的shapefile
    # (hybas_na_lev12_v1c)，trace_upstream_basin找不到包含参照点的子流域直接报错——换哪个
    # 站做参照点都没用，是整片流域都不在这个shapefile覆盖范围内，不是坐标选错了。系统性搜索
    # 补位候选未果，最终定为9个新+已有湖泊，不含Prosperous_Lake。
    "Tobin_Lake":           dict(group="New_2026", rank=1, stations=["05KD004"]),
    "Dauphin_Lake":         dict(group="New_2026", rank=2, stations=["05LJ009"]),
    "Nadina_Lake":          dict(group="New_2026", rank=3, stations=["08JB007"]),
    "Sugar_Lake_Reservoir": dict(group="New_2026", rank=4, stations=["08LC041"]),
    "Coquitlam_Lake":       dict(group="New_2026", rank=5, stations=["08MH149"]),
    # 第10个候选：Lake_Manitoba，手动研究定位(不是自动token搜索找到的——Fairford River
    # Water Control Structure这个坝名跟"Manitoba"没有字面关联，token匹配天然找不到)。
    # 流程完全一致：先查HYDAT核实数据(STN_REGULATION=1、集水面积79780 vs 79790km²几乎
    # 完全一致、逐年完整度均>=94%且无整年缺失)，再查Lake_type，最后才跑CCM——不是先看
    # CCM结果再决定要不要留。
    "Lake_Manitoba":        dict(group="New_2026", rank=6, stations=["05LK002"]),
    # 第11-12个候选，流程与Lake_Manitoba一致：先核实HYDAT数据(STN_REGULATION=1、集水
    # 面积比值、逐年完整度>=90%且无整年缺失)，查Lake_type确认非水库，再跑CCM。
    # 注意：另有Prosperous_Lake、Aishihik_Lake两个候选核实数据本身没问题，但用当前
    # hybas_na_lev12_v1c验证坐标未落入任何子流域(跟已有的Prosperous_Lake排除记录同一个
    # 原因)——Prosperous_Lake属Mackenzie/Great Slave Lake流域、Aishihik_Lake属Yukon River
    # 流域，均不在这个"na"区shapefile覆盖范围内，暂不加入LAKES，需要补充对应区域的
    # HydroBASINS shapefile后才能跑。
    "Skaha_Lake":           dict(group="New_2026", rank=7, stations=["08NM084"]),
    "Split_Lake":           dict(group="New_2026", rank=8, stations=["05UF003"]),
    # 第15-17个候选。Deer_Lake核实过完整流程(集水面积比值、REGULATED=1、hybas_na_lev12_v1c
    # 覆盖确认)，配了RegFlow。Buffalo_Pound_Lake不配RegFlow：集水面积比值最接近的候选站
    # (05JG005，3399 vs 3398 km²几乎完全一致)只有H记录且1970年就停测，1994-2024完全没有
    # 数据；其余邻近regulated流量站集水面积比值都在3-5倍以上，是下游合流后的大流域，核实
    # 后判定不存在这个湖自己的可用出口，不是门槛问题，是真的没有站。Crooked_Lake配了
    # RegFlow(05JM013，比值0.973)：1994-2024每年都有数据、无整年缺失，但常年只有4-10个月/
    # 月(冬季结冰期固定停测模式)，完整度66.4%，明确不满足其余湖泊统一使用的90%门槛，是
    # 经用户确认接受的例外，下游分析/论文里需要单独注明。
    "Deer_Lake":            dict(group="New_2026", rank=9, stations=["02YL007"]),
    "Buffalo_Pound_Lake":   dict(group="New_2026", rank=10, stations=["05JG009"]),
    "Crooked_Lake":         dict(group="New_2026", rank=11, stations=["05JM006"]),
    # 第18-20个候选：尼尔森河梯级链 Playgreen/Kiskitto->Sipiwesk Lake->Split Lake的另外3个
    # 环节(Split_Lake已在研究里)，全部由Manitoba Hydro在Jenpeg大坝统一调度(把Lake Winnipeg
    # 水位控制在711-715英尺之间)，用来跟Okanagan链(Kalamalka/Okanagan/Skaha/Vaseux)做
    # "生态区内部梯级调度湖泊"的对称对比，最终定为4湖(Playgreen/Kiskitto/Sipiwesk/Split)。
    # 排除记录：
    # - Cross_Lake：HydroLAKES把它切成好几个不相连的小多边形(测站落进的那块只有64km²，
    #   真实湖泊~755km²)，没有一块能代表真实湖泊面积，直接用会让T/Evap的湖泊掩膜系统性
    #   偏小，改用链最上游的Playgreen Lake(Hylak_id=578，759.81km²，单一多边形干净匹配)
    #   代替，RegFlow同样借用Jenpeg(Jenpeg大坝回水同时顶托Playgreen和Cross Lake两个湖，
    #   见Manitoba Hydro资料"Jenpeg forebay")。
    # - Stephens_Lake：核实HydroLAKES Lake_type=2(真正的水库，1971年Kettle大坝蓄水形成，
    #   不是天然湖)，跟本研究"只用天然湖泊(Lake_type∈{1,3})"的统一标准冲突，排除。
    # - Little Playgreen Lake、Pipestone Lake(东支流上另外两个环节)：HydroLAKES按名字搜索
    #   查无此湖，WSC也没有对应水位站，数据缺口不是我们能补的，排除。
    # Sipiwesk_Lake自己没有坝(是Cross Lake和Split Lake之间的过水段)，RegFlow借用下游
    # Kelsey发电站(05UE005)的实测流量当代理——跟Split_Lake现在的做法(借用下游Kettle站)
    # 是同一个逻辑。
    "Playgreen_Lake":       dict(group="Nelson_River_Chain", rank=1, stations=["05UB005"]),
    "Kiskitto_Lake":        dict(group="Nelson_River_Chain", rank=2, stations=["05UB013"]),
    "Sipiwesk_Lake":        dict(group="Nelson_River_Chain", rank=3, stations=["05UD006"]),
}

STATION_COORDS = {
    "02DD006": (-79.4738082885742, 46.3118286132813),   # LAKE NIPISSING AT NORTH BAY
    "05JH004": (-105.183860778809, 50.9956398010254),   # LAST MOUNTAIN LAKE AT ROWAN'S RAVINE
    "05JL004": (-103.620826721191, 50.6927795410156),   # KATEPWA LAKE AT KATEPWA BEACH
    "05NB020": (-103.769279479981, 49.5753898620606),   # NICKLE LAKE NEAR WEYBURN
    "05PB007": (-93.3206787109375, 48.6491203308106),   # RAINY LAKE NEAR FORT FRANCES
    "05PB024": (-92.9583587646484, 48.7004699707031),   # RAINY LAKE NEAR BEAR PASS
    "05PD008": (-94.2834930419922, 49.1328010559082),   # LAKE OF THE WOODS AT HANSON BAY
    "05PD011": (-94.8102493286133, 49.7126388549805),   # LAKE OF THE WOODS AT CLEARWATER BAY
    "05PD029": (-94.8531112670898, 49.3284683227539),   # LAKE OF THE WOODS AT CYCLONE ISLAND
    "05PE014": (-94.5537185668945, 49.762939453125),    # LAKE OF THE WOODS AT KEEWATIN (出口站)
    "06EC001": (-98.9413604736328, 56.7784996032715),   # SOUTHERN INDIAN LAKE NEAR SOUTH INDIAN LAKE
    "06EC003": (-99.0234832763672, 56.6872711181641),   # SOUTHERN INDIAN LAKE AT SOUTH BAY
    "06EC007": (-99.5252838134766, 56.7832183837891),   # SOUTHERN INDIAN LAKE NEAR OPACHUANAU LAKE
    "08LG046": (-120.525100708008, 50.158390045166),    # NICOLA LAKE NEAR QUILCHENA
    "08NH064": (-116.930320739746, 49.6540412902832),   # KOOTENAY LAKE AT QUEENS BAY (出口站)
    "08NH067": (-116.659561157227, 49.2988204956055),   # KOOTENAY LAKE AT KUSKONOOK
    "08NM083": (-119.499656677246, 49.8861312866211),   # OKANAGAN LAKE AT KELOWNA
    "08NM143": (-119.274421691895, 50.2299194335938),   # KALAMALKA LAKE AT VERNON PUMPHOUSE
    "08NM243": (-119.525512695313, 49.2731895446777),   # VASEUX LAKE NEAR THE OUTLET
    "02EA014": (-80.03578186035156, 45.33855819702149), # LAKE HURON AT PARRY SOUND
    "05ND012": (-102.23078155517578, 49.25899887084961),  # GRANT DEVINE LAKE NEAR ALAMEDA
    "05MD009": (-101.41242980957033, 50.96038055419922),  # LAKE OF THE PRAIRIES NEAR SHELLMOUTH
    "05PE011": (-94.52446746826172, 49.771968841552734),  # LAKE OF THE WOODS WESTERN OUTLET ABOVE NORMAN DAM (RegFlow)
    "05PE006": (-94.50308227539062, 49.77252960205078),   # LAKE OF THE WOODS EASTERN OUTLET AT KENORA POWERHOUSE (RegFlow)
    "05PC019": (-93.4034423828125, 48.60852813720703),    # RAINY RIVER AT FORT FRANCES (RegFlow)
    "06EC002": (-99.04425048828124, 56.67350006103516),   # SOUTH BAY DIVERSION CHANNEL AT SOUTH BAY (RegFlow)
    "02DD017": (-80.02749633789062, 46.12083053588867),   # FRENCH RIVER AT CHAUDIERE DAM (RegFlow)
    "05JK002": (-104.79792022705078, 50.7065315246582),   # QU'APPELLE RIVER BELOW CRAVEN DAM (RegFlow)
    "05ND013": (-102.22930908203124, 49.24803161621094),  # MOOSE MOUNTAIN CREEK BELOW GRANT DEVINE LAKE (RegFlow)
    "05ME001": (-101.43536376953124, 50.81005096435547),  # ASSINIBOINE RIVER NEAR RUSSELL (RegFlow)
    "08NM050": (-119.6153564453125, 49.49851989746094),   # OKANAGAN RIVER AT PENTICTON (RegFlow)
    "08NM247": (-119.5280303955078, 49.25683975219727),   # OKANAGAN RIVER BELOW MCINTYRE DAM (RegFlow)
    "08NJ158": (-117.46499633789062, 49.466941833496094), # KOOTENAY LAKE OUTFLOW NEAR CORRA LINN (RegFlow)
    "08NM065": (-119.2668914794922, 50.238468170166016),  # VERNON CREEK AT OUTLET OF KALAMALKA LAKE (RegFlow)
    "08LG065": (-120.66876220703124, 50.162139892578125), # NICOLA RIVER AT OUTLET OF NICOLA LAKE (RegFlow)
    # --- 2026-08新增6湖泊，坐标取自HYDAT STATIONS表LATITUDE/LONGITUDE字段 ---
    "05KD004": (-103.395606994629, 53.6571884155273),   # TOBIN LAKE AT THE SPILLWAY
    "05KD003": (-103.298721313477, 53.7071418762207),   # SASKATCHEWAN RIVER BELOW TOBIN LAKE (RegFlow)
    "05LJ009": (-99.9503936767578, 51.4399185180664),   # DAUPHIN LAKE AT OUTLET
    "05LJ025": (-99.971076965332, 51.4503288269043),    # MOSSY RIVER BELOW OUTLET OF DAUPHIN LAKE (RegFlow)
    "07SB014": (-114.163528442383, 62.5454711914063),   # PROSPEROUS LAKE NEAR MCMEEKAN BAY
    "07SB002": (-114.220062255859, 62.5589408874512),   # YELLOWKNIFE RIVER AT OUTLET OF PROSPEROUS LAKE (RegFlow)
    "08JB007": (-126.970832824707, 53.9041595458984),   # NADINA LAKE NEAR NORALEE
    "08JB008": (-126.954719543457, 53.9025001525879),   # NADINA RIVER AT OUTLET OF NADINA LAKE (RegFlow)
    "08LC041": (-118.538261413574, 50.3576698303223),   # SUGAR LAKE RESERVOIR AT THE OUTLET
    "08LC018": (-118.544967651367, 50.347110748291),    # SHUSWAP RIVER AT OUTLET OF SUGAR LAKE RESERVOIR (RegFlow)
    "08MH149": (-122.779197692871, 49.3557395935059),   # COQUITLAM LAKE FOREBAY
    "08MH002": (-122.78150177002, 49.2652397155762),    # COQUITLAM RIVER AT PORT COQUITLAM (RegFlow)
    "05LK002": (-98.8026428222656, 51.4438095092773),   # LAKE MANITOBA AT STEEP ROCK
    "05LM001": (-98.7118072509766, 51.587329864502),    # FAIRFORD RIVER NEAR FAIRFORD (RegFlow)
    # --- 第11-14个候选，坐标取自HYDAT STATIONS表LATITUDE/LONGITUDE字段 ---
    "08NM084": (-119.57508087158205, 49.42641830444336),  # SKAHA LAKE AT OKANAGAN FALLS
    "08NM002": (-119.58040618896484, 49.34204864501953),  # OKANAGAN RIVER AT OKANAGAN FALLS (RegFlow)
    "05UF003": (-96.08728790283205, 56.24385833740234),   # SPLIT LAKE AT SPLIT LAKE
    "05UF006": (-94.6338882446289, 56.380279541015625),   # NELSON RIVER AT KETTLE GENERATING STATION (RegFlow)
    # --- 第18-21个候选：尼尔森河梯级链，坐标取自MSC GeoMet hydrometric-stations API ---
    "05UB005": (-97.9749984741211, 53.90277862548828),    # PLAYGREEN LAKE AT ENTRANCE TO EAST NELSON RIVER
    "05UB013": (-98.4383316040039, 54.30305862426758),    # KISKITTO LAKE NEAR NORWAY HOUSE
    "05UB009": (-98.04805755615234, 54.4980583190918),    # NELSON RIVER (WEST CHANNEL) AT JENPEG (RegFlow, Playgreen_Lake/Kiskitto_Lake共用)
    "05UD006": (-97.5, 55.09444046020508),                # SIPIWESK LAKE AT FORESTRY DOCK
    "05UE005": (-96.5250015258789, 56.03889083862305),    # NELSON RIVER AT KELSEY GENERATING STATION (RegFlow, Sipiwesk_Lake下游代理)
    "07SB014": (-114.1635284423828, 62.54547119140625),   # PROSPEROUS LAKE NEAR MCMEEKAN BAY
    "07SB002": (-114.22006225585938, 62.55894088745117),  # YELLOWKNIFE RIVER AT OUTLET OF PROSPEROUS LAKE (RegFlow)
    "08AA005": (-136.99960327148438, 61.19744110107422),  # AISHIHIK LAKE NEAR WHITEHORSE
    "08AA010": (-136.9915771484375, 61.18521881103516),   # AISHIHIK RIVER BELOW AISHIHIK LAKE (RegFlow)
    # --- 第15-17个候选 ---
    "02YL007": (-57.43825149536133, 49.17013931274414),   # DEER LAKE NEAR GENERATING STATION
    "02YL003": (-57.76057815551758, 48.98286056518555),   # HUMBER RIVER AT HUMBER VILLAGE BRIDGE (RegFlow)
    "05JG009": (-105.3861083984375, 50.58332824707031),   # BUFFALO POUND LAKE AT PUMPING STATION (无RegFlow)
    "05JM006": (-102.66777801513672, 50.60778045654297),  # CROOKED LAKE NEAR GRAYSON
    "05JM013": (-102.9175033569336, 50.64139175415039),   # QU'APPELLE RIVER AT HYDE (RegFlow, 完整度66.4%例外)
}

# 出口站：用于确定流域追溯起点的参照坐标（Southern_Indian_Lake三个站名都不能明确指向出水口，
# 留空由代码兜底用第一个站）。
LAKE_OUTLET_STATION = {
    "Lake_of_the_Woods": "05PE014",
    "Rainy_Lake": "05PB007",
    "Southern_Indian_Lake": None,
    "Kootenay_Lake": "08NH064",
}

# 15个湖泊里，人为调控流量(RegFlow)只对以下12个可用（Nickle_Lake、Katepwa_Lake无可用连续流量
# 记录，Lake_Huron不受调控不需要这个变量，三者均不在这个字典里，process_lake_modal会自动跳过RegFlow）。
REGULATION_STATIONS = {
    "Lake_of_the_Woods":    ["05PE011", "05PE006"],  # Norman Dam + Kenora Powerhouse，两站相加才是总放水量
    "Rainy_Lake":           ["05PC019"],
    "Southern_Indian_Lake": ["06EC002"],             # South Bay Diversion Channel
    "Lake_Nipissing":       ["02DD017"],             # 仅2011-2024子区间可用（1999-2010停测12年）
    "Last_Mountain_Lake":   ["05JK002"],             # Craven Dam坝下，已确认为Last Mountain Lake实际调控出口
    "Grant_Devine_Lake":    ["05ND013"],             # 仅2015-2024子区间可用（坝1994年建成，测流站2015年才建）
    "Lake_of_the_Prairies": ["05ME001"],             # 官方指定站05ME012已停测，改用这个仍在运行的替代站
    "Okanagan_Lake":        ["08NM050"],
    "Vaseux_Lake":          ["08NM247"],             # 仅2012-2024子区间可用
    "Kootenay_Lake":        ["08NJ158"],
    "Kalamalka_Lake":       ["08NM065"],             # 正好在Kalamalka Lake出口
    "Nicola_Lake":          ["08LG065"],
    # 2026-08新增5湖泊，全部核实过STN_REGULATION=1且集水面积与湖泊本身基本一致：
    "Tobin_Lake":           ["05KD003"],             # 276100 vs 276000 km²，几乎完全一致
    "Dauphin_Lake":         ["05LJ025"],             # 8730 vs 8709 km²；官方remarks明确写"用Dauphin
                                                      # Lake(Mossy)大坝闸门记录估算"
    "Nadina_Lake":          ["08JB008"],             # 376.1 vs 374.9 km²
    "Sugar_Lake_Reservoir": ["08LC018"],             # 1132 vs 1131 km²
    "Coquitlam_Lake":       ["08MH002"],             # 233.7 vs 186.7 km²(+25%，5个新湖里集水面积
                                                      # 匹配度最松的一个，其余核实条件都满足)
    "Lake_Manitoba":        ["05LM001"],             # 79790 vs 79780 km²，几乎完全一致
    "Skaha_Lake":           ["08NM002"],              # 6686 vs 6684 km²，几乎完全一致
    "Split_Lake":           ["05UF006"],              # 1306000 vs 1302000 km²，几乎完全一致
    "Deer_Lake":            ["02YL003"],              # 7821 vs 7794 km²，几乎完全一致
    "Crooked_Lake":         ["05JM013"],              # 43200 vs 44410 km²，比值0.973；完整度仅66.4%
                                                       # (冬季固定停测)，用户已知情接受，不满足其余
                                                       # 湖泊统一的90%门槛，需在论文里单独注明例外。
    # Buffalo_Pound_Lake不在这个字典里(故意的，非门槛问题)：集水面积比值最接近的候选站
    # 1970年后完全无数据，核实后判定这个湖没有可用的现代出口流量记录。process_lake_modal
    # 会自动跳过RegFlow，只用WL+气候变量。
    "Playgreen_Lake":       ["05UB009"],   # Jenpeg大坝回水同时顶托Playgreen_Lake和下游Cross Lake
    "Kiskitto_Lake":        ["05UB009"],   # 同一西支流，同样经Jenpeg控制
    "Sipiwesk_Lake":        ["05UE005"],   # Sipiwesk_Lake自己没有坝，借用下游Kelsey发电站出流代理
}
# 数据覆盖不全的湖泊，CCM只使用各自子区间：
REGULATION_SUBPERIODS = {
    "Lake_Nipissing": (2011, 2024),
    "Vaseux_Lake": (2012, 2024),
    "Grant_Devine_Lake": (2015, 2024),
}

LAKE_NAME_SEARCH_TERM = {
    "Lake_of_the_Woods": "Woods", "Rainy_Lake": "Rainy", "Southern_Indian_Lake": "Southern Indian",
    "Lake_Nipissing": "Nipissing", "Lake_Huron": "Huron", "Nickle_Lake": "Nickle",
    "Last_Mountain_Lake": "Last Mountain", "Katepwa_Lake": "Katepwa",
    "Grant_Devine_Lake": "Grant Devine", "Lake_of_the_Prairies": "Prairies",
    "Okanagan_Lake": "Okanagan", "Vaseux_Lake": "Vaseux",
    "Kootenay_Lake": "Kootenay", "Kalamalka_Lake": "Kalamalka", "Nicola_Lake": "Nicola",
    "Tobin_Lake": "Tobin", "Dauphin_Lake": "Dauphin",
    "Nadina_Lake": "Nadina", "Sugar_Lake_Reservoir": "Sugar", "Coquitlam_Lake": "Coquitlam",
    "Lake_Manitoba": "Manitoba", "Skaha_Lake": "Skaha", "Split_Lake": "Split",
    "Deer_Lake": "Deer", "Buffalo_Pound_Lake": "Buffalo Pound", "Crooked_Lake": "Crooked",
    "Playgreen_Lake": "Playgreen", "Kiskitto_Lake": "Kiskitto",
    "Sipiwesk_Lake": "Sipiwesk",
}

# resolve_hylak_id默认只在country="Canada"的HydroLAKES记录里找候选，这对14个完全在加拿大
# 境内的湖泊没问题，但Lake_Huron是加美共有的跨境湖，HydroLAKES把整个湖的Country字段标成了
# "United States of America"(Hylak_id=8，面积59399km²，经search_hydrolakes核实过是唯一
# 正确候选)——默认的国家过滤会把这条正确记录排除掉，退而求其次匹配到旁边一个不相关的
# 0.2km²小水域(audit_masks查出来的)。这里对Lake_Huron单独关闭国家过滤；9湖泊版本Modal代码
# 里Osoyoos_Lake(也是跨境湖)有过同样的处理，15湖泊版本迁移时误以为"全部在加拿大境内"漏掉了
# 这一个例外。
LAKE_RESOLVE_COUNTRY = {"Lake_Huron": None}


@app.function(
    image=image,
    volumes={DATA_ROOT: volume},
    secrets=[modal.Secret.from_name("cds-api")],
    timeout=3600,  # 内部退避重试最多等到~10分钟，加上下载本身的时间，放宽一点超时
    max_containers=4,  # 之前retries=2形同虚设(见下面说明)，光调大并发数没有真正的重试兜底，
                       # 15直接全开撞了CDS限流(13/15被拒)。现在函数内部自己带了退避重试，
                       # 4是在"比2快"和"CDS这个账号到底能扛多少并发我们并不确切知道"之间
                       # 选的折中值，不是量出来的最优值——如果这个值还是频繁触发重试，往下调；
                       # 如果重试很少触发，可以再往上试。
)
def fetch_era5_modal(lake_name: str) -> dict:
    """只做ERA5-Land下载+缓存到Volume，不跑CCM。低并发运行，避免撞CDS账号级限流。

    注意：try_fetch_era5_land内部会把cdsapi抛出的异常自己try/except吞掉、返回None，
    不会往外抛异常——这意味着Modal装饰器上的retries=N对这种失败完全不起作用(retries
    只在函数本身抛异常时触发，正常return不算)。所以这里不走try_fetch_era5_land的
    封装，直接调用更底层的fetch_era5_land_monthly，自己捕获异常、按退避策略重试。
    """
    import os
    import glob
    import time
    from pathlib import Path

    import geopandas as gpd

    cds_rc = Path.home() / ".cdsapirc"
    cds_rc.write_text(f"url: {os.environ['CDSAPI_URL']}\nkey: {os.environ['CDSAPI_KEY']}\n")

    ERA5_DIR = Path(DATA_ROOT) / "era5_downloads"
    ERA5_DIR.mkdir(exist_ok=True, parents=True)

    from ccm_lib import resolve_lake_area_bbox, fetch_era5_land_monthly

    hydrolakes_matches = glob.glob(str(Path(DATA_ROOT) / "HydroLAKES_polys_v10_shp" / "**" / "*.shp"), recursive=True)
    hydrobasins_matches = glob.glob(str(Path(DATA_ROOT) / "hybas_na_lev12_v1c" / "*.shp"))
    hydrolakes_gdf = gpd.read_file(hydrolakes_matches[0])
    hydrobasins_gdf = gpd.read_file(hydrobasins_matches[0])

    # resolve_lake_area_bbox内部可能抛异常(比如trace_upstream_basin在参照点没落进任何
    # 子流域多边形时会raise ValueError)——这里必须兜住，否则Modal的.map()会把这一个湖泊的
    # 异常原样往上抛，导致整批其余湖泊(包括本来能秒回缓存的)全部被取消(已经在10湖泊批量跑
    # Prosperous_Lake时踩过一次：其余9个湖泊全部收到cancellation signal)。
    try:
        hylak_id, lake_cells, basin_cells, area_bbox = resolve_lake_area_bbox(lake_name, hydrolakes_gdf, hydrobasins_gdf)
    except Exception as e:
        return {"lake_name": lake_name, "status": f"FAILED_exception_in_resolve_lake_area_bbox: {type(e).__name__}: {e}"}
    if hylak_id is None:
        return {"lake_name": lake_name, "status": "FAILED_hylak_id_not_resolved"}

    out_path = ERA5_DIR / f"{lake_name}_era5land_monthly.nc"
    if out_path.exists():
        print(f"[CACHED] {lake_name} 的ERA5-Land数据已在Volume上: {out_path}")
        return {"lake_name": lake_name, "status": "OK"}

    last_err = None
    for attempt in range(1, 5):
        try:
            fetch_era5_land_monthly(area_bbox, range(START_YEAR, END_YEAR + 1), out_path)
            print(f"[OK] ERA5-Land 下载完成: {out_path}")
            volume.commit()
            return {"lake_name": lake_name, "status": "OK"}
        except Exception as e:
            last_err = e
            wait_s = 30 * attempt  # 30s, 60s, 90s, 120s
            print(f"  [RETRY {attempt}/4] {lake_name} CDS请求失败({e})，{wait_s}秒后重试")
            if attempt < 4:
                time.sleep(wait_s)

    print(f"[SKIP] ERA5-Land 下载未执行 ({lake_name})，重试4次仍失败: {last_err}")
    return {"lake_name": lake_name, "status": f"FAILED_era5_download: {last_err}"}


@app.function(
    image=image,
    volumes={DATA_ROOT: volume},
    secrets=[modal.Secret.from_name("cds-api")],
    timeout=7200,  # pairwise-complete之后不涉及RegFlow的边library size会明显变大(比如从119恢复到接近372个月)，
                   # 单条边CCM+IAAFT替代数据检验比之前listwise版本慢，1小时超时可能不够，放宽到2小时
    max_containers=15,  # 15个湖泊最多同时15个容器并行——ERA5已经被fetch_era5_modal提前缓存好，
                        # 这里try_fetch_era5_land命中Volume缓存直接返回，不会再打CDS，可以放心全速并行。
    retries=1,
)
def process_lake_modal(lake_name: str, n_surrogates: int = 200, sample: int = 100) -> dict:
    """
    单湖泊完整流程：跟CCM_Lake_Pipeline_v3_pairwise_complete.ipynb里的process_lake逻辑一致
    (pairwise-complete CCM，见ccm_lib.py顶部说明)，只是数据源换成Modal Volume，
    ERA5凭证换成Modal Secret（写到容器内~/.cdsapirc，cdsapi库照常读取）。
    """
    import os
    import glob
    import hashlib
    import io
    import json
    import pickle
    import zipfile
    from pathlib import Path

    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import requests

    # ---- CDS凭证：从Modal Secret环境变量写成cdsapi库要求的文件格式 ----
    cds_rc = Path.home() / ".cdsapirc"
    cds_rc.write_text(f"url: {os.environ['CDSAPI_URL']}\nkey: {os.environ['CDSAPI_KEY']}\n")

    RESULTS_DIR = Path(DATA_ROOT) / "lake_results"
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    ERA5_DIR = Path(DATA_ROOT) / "era5_downloads"
    ERA5_DIR.mkdir(exist_ok=True, parents=True)

    # ---- checkpoint失效判断：不再只看PIPELINE_VERSION这一个人工维护的字符串 ----
    # 这个湖泊自己相关的配置(水位站列表/RegFlow站点列表/REGULATION_SUBPERIODS窗口)跟
    # PIPELINE_VERSION一起算一个指纹，三者任何一个变了这个湖的缓存就自动失效重算。
    # 之前的教训：给Crooked_Lake新增REGULATION_STATIONS条目时忘了手动升PIPELINE_VERSION，
    # 导致process_lake_modal读到旧缓存直接返回、RegFlow配置改了跟没改一样，结果悄悄错了
    # 两次重跑都没发现——只有对着日志逐行看才查出来。改成自动指纹之后，任何单个湖泊的配置
    # 改动都会被自动感知，不再依赖人记得同步升版本号。
    config_fingerprint = hashlib.sha256(json.dumps({
        "pipeline_version": PIPELINE_VERSION,
        "stations": LAKES.get(lake_name, {}).get("stations"),
        "regulation_stations": REGULATION_STATIONS.get(lake_name),
        "regulation_subperiod": REGULATION_SUBPERIODS.get(lake_name),
    }, sort_keys=True, default=str).encode()).hexdigest()[:16]

    checkpoint_path = RESULTS_DIR / f"{lake_name}_result.pkl"
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, "rb") as f:
                cached = pickle.load(f)
            if cached.get("config_fingerprint") == config_fingerprint and cached["status"] in ("OK", "OK_no_candidates"):
                return cached
        except Exception as e:
            # 旧checkpoint可能是不同pandas/numpy版本序列化的，读不出来就当没有缓存，重新跑，
            # 不要让一个读不了的旧文件把整个函数炸掉。
            print(f"  [WARN] 旧checkpoint读取失败（可能是版本不兼容），忽略缓存重新计算: {e}")

    # ---- HydroLAKES / HydroBASINS：从Volume读（提前用modal volume put上传好） ----
    hydrolakes_matches = glob.glob(str(Path(DATA_ROOT) / "HydroLAKES_polys_v10_shp" / "**" / "*.shp"), recursive=True)
    hydrobasins_matches = glob.glob(str(Path(DATA_ROOT) / "hybas_na_lev12_v1c" / "*.shp"))
    hydrolakes_gdf = gpd.read_file(hydrolakes_matches[0])
    hydrobasins_gdf = gpd.read_file(hydrobasins_matches[0])

    # ==== 以下逻辑跟notebook cell 10/11/15/33基本一致，按需要粘贴/维护成同一份 ====
    # 为了不在这里重复贴几百行代码、造成两份代码不同步，
    # 实际维护方式建议见文件末尾的"同步策略"说明。
    from ccm_lib import fetch_lake_water_level, fetch_lake_regulation_flow, resolve_lake_area_bbox, \
        try_fetch_era5_land, extract_masked_series, \
        convert_era5_units, build_variable_panel, get_embedding_params

    result = {"lake_name": lake_name, "status": "started", "pipeline_version": PIPELINE_VERSION,
              "config_fingerprint": config_fingerprint}
    try:
        combined_wl, wide_wl, coverage_wl = fetch_lake_water_level(lake_name, method="anomaly_mean")
        result["coverage_mean"] = coverage_wl["coverage_fraction"].mean()
        result["wide_wl"] = wide_wl

        regflow_series = fetch_lake_regulation_flow(lake_name)
        result["has_regflow"] = regflow_series is not None

        hylak_id, lake_cells, basin_cells, area_bbox = resolve_lake_area_bbox(lake_name, hydrolakes_gdf, hydrobasins_gdf)
        if hylak_id is None:
            result["status"] = "FAILED_hylak_id_not_resolved"
            with open(checkpoint_path, "wb") as f:
                pickle.dump(result, f)
            volume.commit()
            return result
        result["hylak_id"] = hylak_id

        # ERA5数据已经由main()先调用fetch_era5_modal低并发预下载/缓存过，这里命中Volume缓存
        # 直接返回，不会再触发CDS请求（否则15个容器同时打CDS会被限流，见fetch_era5_modal注释）。
        nc_path = try_fetch_era5_land(lake_name, area_bbox, era5_dir=ERA5_DIR)
        if nc_path is None:
            result["status"] = "FAILED_era5_download"
            with open(checkpoint_path, "wb") as f:
                pickle.dump(result, f)
            volume.commit()
            return result

        nc_path = Path(nc_path)
        if nc_path.read_bytes()[:4] == b"PK\x03\x04":
            extract_dir = nc_path.parent / (nc_path.stem + "_extracted")
            if not extract_dir.exists():
                with zipfile.ZipFile(nc_path) as zf:
                    zf.extractall(extract_dir)
            nc_path = sorted(extract_dir.glob("*.nc"))[0]

        real_predictors = pd.DataFrame({
            "T": convert_era5_units("T", extract_masked_series(nc_path, "t2m", lake_cells)),
            "Evap": convert_era5_units("Evap", extract_masked_series(nc_path, "e", lake_cells)),
            "P": convert_era5_units("P", extract_masked_series(nc_path, "tp", basin_cells)),
            "R": convert_era5_units("R", extract_masked_series(nc_path, "ro", basin_cells)),
            "SWE": convert_era5_units("SWE", extract_masked_series(nc_path, "sd", basin_cells)),
        })
        if regflow_series is not None:
            real_predictors["RegFlow"] = regflow_series.reindex(real_predictors.index)
        result["real_predictors"] = real_predictors

        _, panel_deseason = build_variable_panel(combined_wl, {c: real_predictors[c] for c in real_predictors.columns}, forecast_horizon=FORECAST_HORIZON)
        # 不再对panel_deseason整体dropna——这是上一版"pairwise-complete"修复漏改的一行：
        # ccm_train_panel是从panel_deseason切出来的，如果这里先按全部变量listwise dropna一次，
        # 后面per-edge的pairwise-complete逻辑(_pair_panel/surrogate_significance/lag_scan)
        # 拿到的panel_df早就已经被砍到RegFlow能用的那个短窗口了，形同虚设——每条边算出来的
        # n_obs会完全一样，就是这个bug的症状。这里只用WL自己的有效月数做健全性检查。
        if panel_deseason["WL"].dropna().shape[0] < 60:
            result["status"] = "FAILED_too_few_overlapping_months"
            with open(checkpoint_path, "wb") as f:
                pickle.dump(result, f)
            volume.commit()
            return result

        # 不再在这里dropna——CCM每条边各自按pairwise-complete原则取数据(见ccm_lib.py里
        # run_pairwise_ccm/surrogate_significance/lag_scan)。这里只按日期切掉最后
        # FORECAST_HORIZON个月留给最终测试，不因为某个变量(通常是RegFlow)局部缺测就把
        # 其他变量的完整数据也一起砍掉(Little & Rubin, 2002)。
        ccm_train_panel = panel_deseason.iloc[:-FORECAST_HORIZON]
        # 用WL自己的有效月数做健全性检查(WL是几乎所有下游分析的核心变量)，而不是要求
        # 全部变量同时有值——后者在pairwise-complete设计下已经不是"够不够跑CCM"的正确判据了。
        if ccm_train_panel["WL"].dropna().shape[0] < 60:
            result["status"] = "FAILED_too_few_training_rows_for_ccm"
            with open(checkpoint_path, "wb") as f:
                pickle.dump(result, f)
            volume.commit()
            return result

        available_vars = [v for v in VARS if v in ccm_train_panel.columns]
        # 每个变量的τ/E用它自己完整的日历月份序列算，真实NaN原样保留、不dropna——
        # get_embedding_params内部靠pyEDM的ignoreNan=True正确跳过缺测点，不会把缺测
        # 拼接处两侧本不相邻的观测错误地当成相邻(细节见ccm_lib.py顶部说明)。
        embed_params = {v: get_embedding_params(ccm_train_panel[v].values, v, verbose=False) for v in available_vars}
        # 这个阶段只负责把面板和嵌入参数存进 pkl。CCM 本身由 02_/03_ 跑，
        # 预测由 04_ 跑，两者都从 results/ 读，不经过这里。
        result.update({"status": "OK", "embed_params": embed_params})
    except Exception as e:
        import traceback
        result["status"] = f"FAILED_exception: {type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
        print(result["traceback"])

    with open(checkpoint_path, "wb") as f:
        pickle.dump(result, f)
    volume.commit()
    return result


CONNECTIVITY_CHAINS = {
    "Montane Cordillera (Okanagan Lake Regulation System)": ["Kalamalka_Lake", "Okanagan_Lake", "Vaseux_Lake"],
    "Prairie (Qu'Appelle River system, via Craven Dam)": ["Last_Mountain_Lake", "Katepwa_Lake"],
    "Boreal Shield (Rainy River)": ["Rainy_Lake", "Lake_of_the_Woods"],
}


# 最终定稿的10湖清单，按真实生态区分组——不能用_all_ecoregion_pairs()/LAKES字典里的
# group字段，那是"原15湖3生态区"设计的老分组，Skaha_Lake/Deer_Lake/Split_Lake在LAKES
# 里都还标着"New_2026"，不是它们现在实际所在的生态区，而且老分组里还混着Southern_
# Indian_Lake/Lake_Nipissing/Lake_Huron/Prairie全组等不在最终10湖名单里的湖泊。

# 湖间连通性CCM(connectivity_ccm_chains)结果里，每个湖泊自己作为effect时FDR显著的
# "cause"邻居湖泊及其l_star_months，用于forecast_compare_modal里新增的
# SARIMAX/XGBoost_neighbor_lake方法(把邻居湖泊水位当额外外生变量喂给预测模型，检验
# "知道邻居湖泊水位能不能帮这个湖泊自己预测得更准")。Vaseux_Lake、Kiskitto_Lake没有
# 任何邻居方向显著，不在这个字典里，process_lake_modal/forecast_compare_modal会自动跳过。
NEIGHBOR_PREDICTORS = {
    "Kalamalka_Lake":    [("Okanagan_Lake", 1)],
    "Okanagan_Lake":     [("Kalamalka_Lake", 0), ("Skaha_Lake", 0)],
    "Skaha_Lake":        [("Vaseux_Lake", 0)],
    "Rainy_Lake":        [("Lake_of_the_Woods", 0)],
    "Lake_of_the_Woods": [("Rainy_Lake", 7)],
    "Playgreen_Lake":    [("Sipiwesk_Lake", 0)],
    "Sipiwesk_Lake":     [("Playgreen_Lake", 5), ("Split_Lake", 6)],
    "Split_Lake":        [("Sipiwesk_Lake", 0)],
}


@app.local_entrypoint()
def main(lakes: str = ""):
    """本地入口：modal run ccm_modal_app.py [--lakes Kalamalka_Lake,Okanagan_Lake]
    分两步跑：第1步低并发预下载/缓存ERA5(避免撞CDS账号级限流)，第2步15个湖泊全速并行跑CCM
    (ERA5已缓存，这一步不再触发CDS请求)。已经缓存过ERA5的湖泊第1步会很快跳过。"""
    target_lakes = lakes.split(",") if lakes else list(LAKES.keys())

    print(f"第1步：低并发预下载/缓存ERA5数据，{len(target_lakes)}个湖泊: {target_lakes}")
    era5_results = list(fetch_era5_modal.map(target_lakes))
    for r in era5_results:
        print(f"  {r['lake_name']}: {r['status']}")
    era5_failed = [r["lake_name"] for r in era5_results if r["status"] != "OK"]
    if era5_failed:
        print(f"\n[WARN] {len(era5_failed)}个湖泊ERA5预下载未成功: {era5_failed}")
        print("       这些湖泊第2步大概率会失败在FAILED_era5_download；如果是CDS限流被拒，"
              "隔几分钟单独重跑一下 --lakes 就行，不用重跑全部。")

    print(f"\n第2步：{len(target_lakes)}个湖泊全速并行跑CCM计算...")
    results = list(process_lake_modal.map(target_lakes))
    print("\n=== 全部完成 ===")
    for r in results:
        print(f"  {r['lake_name']}: {r['status']}")


@app.local_entrypoint()
def fetch_only(lakes: str = ""):
    """只跑第1步(低并发ERA5预下载/缓存)，不跑CCM。

    专门用来配合过夜/合盖场景：
        caffeinate -i modal run --detach ccm_modal_app.py::fetch_only

    main()里"先等第1步跑完、再派发第2步"这个调度逻辑是本地Python代码自己在做的，本地进程
    (你的笔记本)如果在两步之间睡眠/断开，第2步不会被自动触发——--detach只保证"已经派发出去
    的云端任务不会被杀掉"，不能让本地进程凭空继续跑下去。所以要过夜跑，就把两步分开手动触发：
    今晚跑这个（第1步很快，二三十分钟量级，可以直接合盖睡觉），明天起来确认ERA5都缓存好了，
    再跑process_only（第2步，真正耗时的CCM计算，同样用--detach起）。
    """
    target_lakes = lakes.split(",") if lakes else list(LAKES.keys())
    print(f"低并发预下载/缓存ERA5数据，{len(target_lakes)}个湖泊: {target_lakes}")
    era5_results = list(fetch_era5_modal.map(target_lakes))
    print("\n=== ERA5预下载完成 ===")
    for r in era5_results:
        print(f"  {r['lake_name']}: {r['status']}")
    failed = [r["lake_name"] for r in era5_results if r["status"] != "OK"]
    if failed:
        print(f"\n[WARN] {len(failed)}个湖泊没成功: {failed}")
        print("       明天先单独重跑这几个 --lakes，确认都OK了再跑process_only。")
    else:
        print("\n全部成功，可以跑 modal run --detach ccm_modal_app.py::process_only 了。")


@app.local_entrypoint()
def process_only(lakes: str = ""):
    """只跑第2步(15并行CCM计算)，假设ERA5已经用fetch_only预下载/缓存好了。

        caffeinate -i modal run --detach ccm_modal_app.py::process_only

    如果某个湖泊ERA5还没缓存，process_lake_modal里try_fetch_era5_land会尝试临时下载一次
    (走的是process_lake_modal自己那个15并发的通道，不带退避重试)，可能又撞CDS限流——所以
    最好先确认fetch_only跑完全部OK，再跑这个。
    """
    target_lakes = lakes.split(",") if lakes else list(LAKES.keys())
    print(f"{len(target_lakes)}个湖泊全速并行跑CCM计算...")
    results = list(process_lake_modal.map(target_lakes))
    print("\n=== 全部完成 ===")
    for r in results:
        print(f"  {r['lake_name']}: {r['status']}")


# ============================================================
# 同步策略说明：
# 上面import的ccm_lib里那一批函数（fetch_lake_water_level、resolve_hylak_id、
# get_basin_mask_cells、build_variable_panel、get_embedding_params等），
# 原样存成ccm_lib.py放在同一个目录，Modal打包镜像时
# 会自动带上同目录下的其他.py文件。不要在这个文件里重复贴一份，容易改一个忘了改
# 另一个、两边不同步。
# ============================================================
