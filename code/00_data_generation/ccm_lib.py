"""
CCM湖泊研究流水线核心算法函数库（2026-08-09重写版）。

跟旧版(git历史里的ccm_lib.py，本次重写前的最后状态另存为ccm_lib.py.bak_pre_v11)相比，
本次重写只改了"缺测数据怎么喂给pyEDM"这一件事，其余全部函数(WSC/ERA5抓取、HydroLAKES/
HydroBASINS解析、因果网络分析、SARIMAX/XGBoost预测对照)逐行保留，没有改动。

============================================================
本次修复的问题，以及为什么这样修（供以后维护时对照）
============================================================

旧版的_pair_panel/surrogate_significance/lag_scan都是"dropna() -> reset_index(drop=True)"，
把两个变量各自缺测的行删掉、剩下的行重新从0连续编号，再喂给pyEDM。这样做的后果是：
如果两个变量的缺测模式不是"整段连续缺失"而是"分散成好几段"(比如某个RegFlow测站每年冬季
固定停测，30年下来缺测拼成三十多段)，dropna+重编号之后，pyEDM会把物理上相隔好几个月的
两行数据当成"挨着的1个月"来构造延迟嵌入向量——这是编出来的假邻接关系，不是真实的动态。
pyEDM/rEDM的延迟嵌入完全按数据行的物理顺序算，不看time列的实际数值，所以这个bug不会报错、
只会悄悄把结果算错。

这次实测确认了两件事(2026-08-09 Modal容器里用pyEDM 2.5.6验证过)：
  1. pyEDM.CCM()这个版本的函数签名里已经没有旧式的`lib="1 80 81 160"`多段边界写法了，
     换成了`validLib`(布尔数组)。但`validLib=False`只能防止某一行被别的预测借用当近邻，
     没法阻止这一行自己被拿去硬预测——用一个刻意留了缺口的人造数据集测过：不管validLib
     排不排除跨缺口那一行，它自己的预测值完全一样，说明validLib单独用是不完整的。
  2. 真正管用、而且简单得多的做法是：**根本不要dropna**。把两个变量对齐到完整的日历月份
     索引(该有多少行就有多少行，缺测的月份就是真实的NaN，不删行不重新编号)，直接喂给
     pyEDM——它的Simplex/CCM默认`ignoreNan=True`，遇到某个预测需要的历史点是NaN时会
     正确地跳过(返回NaN)，不会拿假邻居瞎猜。同样的人造缺口数据集验证过：完整日历表方案下，
     跨缺口那一行的预测值正确地变成NaN；dropna重编号方案下，那一行会被硬凑出一个错误的
     预测值。

这个修复不是在复现某篇具体论文的方法(比如Clark et al. 2015的multispatial CCM，那篇用的是
NA分段拼接+bootstrap+dewdrop regression，是一套完整的独立算法，我们没有实现它的bootstrap
机制)——这里只是让pyEDM按它自己文档说的方式正确处理缺测，不需要额外的算法。如果论文方法论
里要交代这一步，应该引用pyEDM/rEDM官方文档里"Disjoint prediction sets"那条说明，而不是引用
Clark 2015。

这个修复影响的不只是CCM那一步，连"选嵌入参数(tau, E)"这一步也受影响——average_mutual_
information原来会在算lag pair之前先把NaN删掉(np.isnan过滤)，这跟dropna是同一类bug，这次
一并改成先按位置构造lag pair、再对每一对联合去掉缺测，不提前压缩掉时间结构。**调用方
(ccm_modal_app.py)传参数给get_embedding_params时也不能再传`.dropna().values`了，得传
保留了完整日历索引和真实NaN的Series/array，不然这里的修复等于白做**——这一处改动在
ccm_modal_app.py里，不在这个文件里，改的时候要一起看。

============================================================
LAKES/REGULATION_STATIONS等配置仍从ccm_modal_app导入，不在本文件重复定义。
============================================================
"""
import inspect
import io
import itertools

import geopandas as gpd
import numpy as np
import pandas as pd
import pyEDM
import requests
from scipy import stats
from shapely.geometry import Point, box
from statsmodels.stats.multitest import multipletests

from ccm_modal_app import (
    LAKES, REGULATION_STATIONS, REGULATION_SUBPERIODS, VARS,
    START_YEAR, END_YEAR, STATION_COORDS, LAKE_OUTLET_STATION,
    LAKE_NAME_SEARCH_TERM, LAKE_RESOLVE_COUNTRY,
)

WSC_BASE_URL = "https://wateroffice.ec.gc.ca/services/monthly_data/csv/inline"

_ccm_params = inspect.signature(pyEDM.CCM).parameters
CCM_SEQUENTIAL_KWARGS = {"sequential": True} if "sequential" in _ccm_params else {"parallel": False}

CDS_VARIABLES = {
    "T": "2m_temperature", "P": "total_precipitation", "Evap": "total_evaporation",
    "SWE": "snow_depth_water_equivalent", "R": "runoff",
}


# ============ WSC水位抓取（未改动） ============

def fetch_wsc_station_level(station_id, start_year=START_YEAR, end_year=END_YEAR, timeout=30):
    """Fetch monthly mean water level for one WSC station."""
    params = {"stations[]": station_id, "parameters[]": "level", "start_year": start_year, "end_year": end_year}
    r = requests.get(WSC_BASE_URL, params=params, timeout=timeout)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), encoding_errors="ignore")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "ID": "station_id", "Year/Année": "year", "Month/Mois": "month",
        "Value/Valeur": "water_level", "Symbol/Symbole": "symbol",
    })
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    return df[["date", "station_id", "water_level", "symbol"]].sort_values("date").reset_index(drop=True)


def combine_station_water_levels(wide, method="anomaly_mean", baseline_index=None, min_station_fraction=0.0):
    """合成多站水位；baseline_index用于只用训练窗估计站点基准差。"""
    wide = wide.sort_index()
    min_count = max(1, int(np.ceil(wide.shape[1] * min_station_fraction)))
    enough_coverage = wide.notna().sum(axis=1) >= min_count
    if method == "raw_mean":
        combined = wide.mean(axis=1, skipna=True)
    elif method == "anomaly_mean":
        baseline = wide if baseline_index is None else wide.reindex(pd.DatetimeIndex(baseline_index))
        station_means = baseline.mean(axis=0, skipna=True)
        global_mean = station_means.mean(skipna=True)
        combined = (wide - station_means).mean(axis=1, skipna=True) + global_mean
    else:
        raise ValueError(f"unknown method {method}")
    return combined.where(enough_coverage)


def fetch_lake_water_level(lake_name, method="anomaly_mean", start_year=START_YEAR, end_year=END_YEAR):
    """Fetch + combine all WSC stations for a lake."""
    stations = LAKES[lake_name]["stations"]
    station_dfs = {}
    for sid in stations:
        try:
            station_dfs[sid] = fetch_wsc_station_level(sid, start_year, end_year)
        except Exception as e:
            print(f"  [WARN] station {sid} fetch failed: {e}")
    if not station_dfs:
        raise RuntimeError(f"No WSC data retrieved for {lake_name}")

    wide = pd.concat({sid: df.set_index("date")["water_level"] for sid, df in station_dfs.items()}, axis=1).sort_index()
    full_index = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-01", freq="MS")
    wide = wide.reindex(full_index)

    n_stations_used = wide.notna().sum(axis=1)
    coverage_df = pd.DataFrame({
        "n_stations_reporting": n_stations_used,
        "coverage_fraction": n_stations_used / wide.shape[1],
    })
    combined = combine_station_water_levels(wide, method=method)
    return combined, wide, coverage_df


# ============ WSC调控流量抓取（未改动） ============

def fetch_wsc_station_flow(station_id, start_year=START_YEAR, end_year=END_YEAR, timeout=30):
    """Fetch monthly mean regulated discharge (flow) for one WSC station.
    跟fetch_wsc_station_level结构一致，只是parameters[]换成"flow"。"""
    params = {"stations[]": station_id, "parameters[]": "flow", "start_year": start_year, "end_year": end_year}
    r = requests.get(WSC_BASE_URL, params=params, timeout=timeout)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), encoding_errors="ignore")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "ID": "station_id", "Year/Année": "year", "Month/Mois": "month",
        "Value/Valeur": "regulated_flow", "Symbol/Symbole": "symbol",
    })
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    return df[["date", "station_id", "regulated_flow", "symbol"]].sort_values("date").reset_index(drop=True)


def fetch_lake_regulation_flow(lake_name, start_year=START_YEAR, end_year=END_YEAR):
    """抓取一个湖泊调控出口/调控代理站的实测流量；如一个湖泊映射多个放水通道，则多站相加。
    返回None表示该湖泊没有REGULATION_STATIONS条目，调用方应把RegFlow这一列跳过，
    不要塞NaN占位。"""
    station_ids = REGULATION_STATIONS.get(lake_name)
    if not station_ids:
        return None

    full_index = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-01", freq="MS")
    station_series = {}
    for sid in station_ids:
        try:
            df = fetch_wsc_station_flow(sid, start_year, end_year)
            station_series[sid] = df.set_index("date")["regulated_flow"].reindex(full_index)
        except Exception as e:
            print(f"  [WARN] regulation flow station {sid} fetch failed: {e}")
    if not station_series:
        return None

    wide = pd.DataFrame(station_series)
    total_flow = wide.sum(axis=1, skipna=False)

    sub = REGULATION_SUBPERIODS.get(lake_name)
    if sub is not None:
        valid_start, valid_end = sub
        mask = (total_flow.index.year < valid_start) | (total_flow.index.year > valid_end)
        total_flow = total_flow.mask(mask)

    return total_flow


# ============ ERA5-Land下载（未改动） ============

def build_cds_request(area_bbox, years, variables=None):
    variables = variables or list(CDS_VARIABLES.values())
    return {
        "product_type": ["monthly_averaged_reanalysis"], "variable": variables,
        "year": [str(y) for y in years], "month": [f"{m:02d}" for m in range(1, 13)],
        "time": ["00:00"], "area": area_bbox, "data_format": "netcdf",
    }


def fetch_era5_land_monthly(area_bbox, years, out_nc_path, variables=None):
    import cdsapi
    client = cdsapi.Client()
    client.retrieve("reanalysis-era5-land-monthly-means", build_cds_request(area_bbox, years, variables), str(out_nc_path))
    return out_nc_path


def try_fetch_era5_land(lake_name, area_bbox, era5_dir, years=range(START_YEAR, END_YEAR + 1)):
    out_path = era5_dir / f"{lake_name}_era5land_monthly.nc"
    if out_path.exists():
        print(f"[CACHED] {lake_name} 的ERA5-Land数据已在Volume上: {out_path}")
        return out_path
    try:
        fetch_era5_land_monthly(area_bbox, years, out_path)
        print(f"[OK] ERA5-Land 下载完成: {out_path}")
        return out_path
    except Exception as e:
        print(f"[SKIP] ERA5-Land 下载未执行 ({lake_name}): {e}")
        return None


# ============ 流域/湖泊掩膜、Hylak_id解析（未改动） ============

def grid_cells_intersecting_polygon(poly, res=0.1, buffer_deg=0.15):
    """返回与给定多边形相交的所有 0.1度 ERA5-Land 网格盒子。"""
    minx, miny, maxx, maxy = poly.bounds
    lon0, lon1 = np.floor((minx - buffer_deg) / res) * res, np.ceil((maxx + buffer_deg) / res) * res
    lat0, lat1 = np.floor((miny - buffer_deg) / res) * res, np.ceil((maxy + buffer_deg) / res) * res
    cells = []
    for lo in np.arange(lon0, lon1, res):
        for la in np.arange(lat0, lat1, res):
            cell = box(lo, la, lo + res, la + res)
            if cell.intersects(poly):
                cells.append({"lon_center": round(lo + res / 2, 3), "lat_center": round(la + res / 2, 3), "geometry": cell})
    return gpd.GeoDataFrame(cells, crs="EPSG:4326")


def get_lake_mask_cells(hydrolakes_gdf, hylak_id, id_col="Hylak_id"):
    """T / Evap 用：湖泊本体覆盖到的 ERA5-Land 格点。"""
    poly = hydrolakes_gdf.loc[hydrolakes_gdf[id_col] == hylak_id, "geometry"].iloc[0]
    return grid_cells_intersecting_polygon(poly), poly


def trace_upstream_basin(hydrobasins_gdf, outlet_point, id_col="HYBAS_ID", next_down_col="NEXT_DOWN"):
    """P / R / SWE 用：从出口点反向追溯上游全部子流域，dissolve 成完整流域边界。"""
    containing = hydrobasins_gdf[hydrobasins_gdf.geometry.contains(outlet_point)]
    if containing.empty:
        raise ValueError("出口点不在任何子流域多边形内，请检查坐标或 shapefile 覆盖范围")
    outlet_id = containing.iloc[0][id_col]
    children = {}
    for _, row in hydrobasins_gdf.iterrows():
        children.setdefault(row[next_down_col], []).append(row[id_col])
    upstream_ids, frontier = set(), [outlet_id]
    while frontier:
        current = frontier.pop()
        if current in upstream_ids:
            continue
        upstream_ids.add(current)
        frontier.extend(children.get(current, []))
    sub = hydrobasins_gdf[hydrobasins_gdf[id_col].isin(upstream_ids)]
    return outlet_id, upstream_ids, sub.geometry.union_all()


def get_basin_mask_cells(hydrobasins_gdf, outlet_lonlat):
    outlet_id, upstream_ids, basin_poly = trace_upstream_basin(hydrobasins_gdf, Point(outlet_lonlat))
    return grid_cells_intersecting_polygon(basin_poly), basin_poly, upstream_ids


def extract_masked_series(nc_path, var_name, cells_gdf):
    """从ERA5-Land netCDF按格点坐标提取区域平均月度序列。用共享维度做逐点匹配（不是笛卡尔积）。"""
    import xarray as xr
    ds = xr.open_dataset(nc_path)
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    lat_da = xr.DataArray(cells_gdf["lat_center"].values, dims="points")
    lon_da = xr.DataArray(cells_gdf["lon_center"].values, dims="points")
    sel = ds[var_name].sel({lat_name: lat_da, lon_name: lon_da}, method="nearest")
    s = sel.mean(dim="points").to_series()
    s.index = pd.to_datetime(s.index).to_period("M").to_timestamp()
    return s


def resolve_lake_area_bbox(lake_name, hydrolakes_gdf, hydrobasins_gdf):
    """解析湖泊Hylak_id + 计算ERA5下载用的外接矩形bbox，供fetch_era5_modal（低并发预下载）
    和process_lake_modal（正式流程）共用，避免两边各写一份、容易不同步。
    返回(hylak_id, lake_cells, basin_cells, area_bbox)；hylak_id为None表示解析失败。"""
    ref_lonlat = STATION_COORDS[LAKE_OUTLET_STATION.get(lake_name) or LAKES[lake_name]["stations"][0]]
    country = LAKE_RESOLVE_COUNTRY.get(lake_name, "Canada")
    hylak_id, _candidates = resolve_hylak_id(hydrolakes_gdf, LAKE_NAME_SEARCH_TERM[lake_name], ref_lonlat, country=country)
    if hylak_id is None:
        return None, None, None, None
    pour_row = hydrolakes_gdf.loc[hydrolakes_gdf["Hylak_id"] == hylak_id].iloc[0]
    outlet_lonlat = (pour_row["Pour_long"], pour_row["Pour_lat"])
    lake_cells, _lake_poly = get_lake_mask_cells(hydrolakes_gdf, hylak_id)
    basin_cells, _basin_poly, _upstream_ids = get_basin_mask_cells(hydrobasins_gdf, outlet_lonlat)
    area_bbox = [
        max(lake_cells["lat_center"].max(), basin_cells["lat_center"].max()) + 0.1,
        min(lake_cells["lon_center"].min(), basin_cells["lon_center"].min()) - 0.1,
        min(lake_cells["lat_center"].min(), basin_cells["lat_center"].min()) - 0.1,
        max(lake_cells["lon_center"].max(), basin_cells["lon_center"].max()) + 0.1,
    ]
    return hylak_id, lake_cells, basin_cells, area_bbox


def resolve_hylak_id(hydrolakes_gdf, search_term, ref_lonlat, country="Canada", search_buffer_deg=0.5, max_dist_deg=0.05):
    """按坐标空间位置解析Hylak_id：优先看有没有多边形直接包含参照点，其次找离参照点最近的多边形。
    关键点：这里对全部Lake_type一起解析(不预先筛type∈{1,3})，调用方自己决定要不要按类型过滤——
    如果解析前就只看天然湖多边形，会把本该落在附近水库(type=2)里的测站，误配到旁边一个不相关
    的小天然水体上(已验证案例：Tobin_Lake、Sugar_Lake_Reservoir、Coquitlam_Lake、
    Duncan_Reservoir都曾被这样误判)。
    search_term 参数保留但不参与筛选（历史遗留，仅为兼容旧调用签名）。
    """
    lon, lat = ref_lonlat
    ref_pt = Point(ref_lonlat)
    subset = hydrolakes_gdf.cx[lon - search_buffer_deg: lon + search_buffer_deg, lat - search_buffer_deg: lat + search_buffer_deg]
    if country:
        subset = subset[subset["Country"] == country]
    if subset.empty:
        return None, subset
    containing = subset[subset.geometry.contains(ref_pt)]
    if not containing.empty:
        return int(containing.iloc[0]["Hylak_id"]), containing
    subset = subset.copy()
    subset["dist_to_ref"] = subset.geometry.distance(ref_pt)
    subset = subset.sort_values("dist_to_ref")
    best = subset.iloc[0]
    if best["dist_to_ref"] > max_dist_deg:
        return None, subset.head(10)
    return int(best["Hylak_id"]), subset.head(10)


# ============ 单位换算、去季节化、面板拼装（未改动） ============

def _days_in_month(series):
    idx = pd.DatetimeIndex(series.index)
    return pd.Series(idx.days_in_month, index=series.index, dtype=float)


def convert_era5_units(var_name, series, monthly_total=True):
    """ERA5-Land单位换算。
    T: K -> degC；SWE: m water equivalent -> mm；P/R/Evap: m/day -> mm/month（monthly_total=True）或mm/day。
    ERA5 total_evaporation通常以向下通量为正，蒸发为负，因此取负号让Evap表示正向蒸发量。
    """
    series = series.astype(float)
    if var_name == "T":
        return series - 273.15
    if var_name == "SWE":
        return series * 1000.0
    if var_name in ("P", "R"):
        out = series * 1000.0
        return out * _days_in_month(series) if monthly_total else out
    if var_name == "Evap":
        out = -series * 1000.0
        return out * _days_in_month(series) if monthly_total else out
    raise ValueError(f"unknown variable {var_name}")


def deseasonalize(series, train_end=None):
    """减去逐月气候态，返回距平序列。

    月度气候态均值只用 train_end 之前(训练窗口)的数据计算，同一套均值同时套用到
    训练段与测试段。不能用全部数据(含测试期)一起算月度均值再切分 train/test：
    那样测试期某个月的真实值，在被算进"该月气候态均值"的那一刻，就已经把自己的
    信息用于标准化自己了——这是发生在 train/test 切分之前的信息泄漏。

    本实现与 01_shared/ccm_full_pipeline.py 中的同名函数逐字一致（该处早已修复，
    此处为对齐补上）。train_end=None 时退化为用全部数据计算，仅供探索性场景，
    正式流程必须由调用方传入。
    """
    train_series = series if train_end is None else series.iloc[:train_end]
    monthly_clim = train_series.groupby(train_series.index.month).mean()
    month_of_each_point = pd.Series(series.index.month, index=series.index)
    return series - month_of_each_point.map(monthly_clim)


def build_variable_panel(wl_series, era5_vars=None, forecast_horizon=None):
    """把 WL 与 ERA5-Land 各变量对齐成一张月度面板，逐变量去季节化。返回 (原始面板, 去季节化面板)。
    reindex到完整日历月份范围(min~max, 逐月)——这一步本来就保留了真实NaN，是下游
    _pair_panel等函数能正确工作的前提，不要在这之后的任何环节再对整张面板做listwise dropna。

    forecast_horizon 必须由调用方显式传入：去季节化的月度气候态只用训练窗口
    (排除最后 forecast_horizon 个月)计算，否则测试期信息会经由气候态均值泄漏。
    原实现无此参数、恒用全序列，是已确认的泄漏来源。
    """
    era5_vars = era5_vars or {}
    panel = pd.DataFrame({"WL": wl_series})
    for name, s in era5_vars.items():
        panel[name] = s
    panel.index = pd.DatetimeIndex(panel.index)
    panel = panel.sort_index()
    panel = panel.reindex(pd.date_range(panel.index.min(), panel.index.max(), freq="MS"))
    train_end = len(panel) - forecast_horizon if forecast_horizon else None
    return panel, panel.apply(lambda col: deseasonalize(col, train_end=train_end))


# ============ 嵌入参数选择 ============
#
# 当前生效的做法：**τ 固定为 1**（config.EMBED_TAU），E 由单变量 simplex
# 自预测在 config.EMBED_E_CANDIDATES 内选择。见 get_embedding_params()。
#
# 下面的 average_mutual_information / first_local_minimum / select_tau 是
# **AMI 自动选 τ 的旧实现，当前流程不再调用**（全包搜索确认无调用点）。
# 保留它们只为两个目的：(1) 若要做 τ 的敏感性分析可直接复用；
# (2) 便于复核"固定 τ=1 与数据驱动选 τ 相比改变了什么"。
#
# 注意：方法论中记载的是 τ=1。若读到下面这套 AMI 代码而误以为 τ 是数据驱动
# 选出的，即与方法论矛盾——它们没有在跑。
# 本节是这次重写实际改动的地方之一：average_mutual_information原来会在构造lag pair之前
# 先把NaN整体过滤掉(x = x[~np.isnan(x)])，这跟_pair_panel旧版的dropna是同一类bug——
# 会把时间上不相邻、只是恰好都有观测值的两个点错误地当成相邻。现在改成先按位置构造lag pair、
# 再对每一对(x1[i], x2[i])联合判断是否都非缺测，缺测月份该跳的跳，但不提前压缩时间结构。
#
# simplex_self_predict_rho / select_E 这两个不需要改代码本身——它们已经是"整段序列直接喂给
# pyEDM、靠ignoreNan=True自动处理缺测"的写法，只要调用方(get_embedding_params的调用点，在
# ccm_modal_app.py里)传进来的values保留了完整日历索引和真实NaN、不是提前.dropna()过的数组，
# 这两个函数就是对的。

def average_mutual_information(x, max_lag=6, bins=None):
    """
    [未启用] τ 已固定为 1（config.EMBED_TAU），本函数当前无调用点。
AMI用等occupancy(分位数)分箱，对应Cellucci, Albano & Rapp (2005, Phys. Rev. E 71, 066208)
    Section IV自适应分区算法的第一步。分箱数N_E=floor(sqrt(N_D/5))。只实现到基线版本，
    没有实现论文Section IV后续的递归细分。

    x需要是保留了原始日历月份位置的数组(缺测月份为np.nan，不要提前dropna)——lag pair
    按位置错位构造(x[:-lag] vs x[lag:])，如果x事先被压缩过，缺测两侧的观测会被错误地
    当成相邻处理。这里在错位之后才对每一对联合去掉缺测，缺测模式分散成很多段时尤其重要。"""
    x = np.asarray(x, dtype=float)
    n_obs = int(np.sum(~np.isnan(x)))
    amis = {}
    for lag in range(1, max_lag + 1):
        x1, x2 = x[:-lag], x[lag:]
        pair_mask = ~np.isnan(x1) & ~np.isnan(x2)
        x1, x2 = x1[pair_mask], x2[pair_mask]
        n_bins = bins if bins is not None else max(5, int(np.sqrt(n_obs / 5)))
        if len(x1) < 2:
            amis[lag] = 0.0
            continue
        edges1 = np.unique(np.quantile(x1, np.linspace(0, 1, n_bins + 1)))
        edges2 = np.unique(np.quantile(x2, np.linspace(0, 1, n_bins + 1)))
        if len(edges1) < 2 or len(edges2) < 2:
            amis[lag] = 0.0
            continue
        c_xy, _, _ = np.histogram2d(x1, x2, bins=[edges1, edges2])
        total = c_xy.sum()
        if total == 0:
            amis[lag] = 0.0
            continue
        pxy = c_xy / total
        px, py = pxy.sum(axis=1, keepdims=True), pxy.sum(axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = pxy * np.log(pxy / (px * py))
        amis[lag] = float(np.nan_to_num(terms, nan=0.0, posinf=0.0, neginf=0.0).sum())
    return amis


def calendar_month_offsets(index, start=None, one_based=True):
    """把DatetimeIndex转换成真实日历月偏移；有缺月时time轴会保留跳跃。"""
    idx = pd.DatetimeIndex(index)
    if len(idx) == 0:
        return np.array([], dtype=int)
    start = pd.Timestamp(idx.min() if start is None else start)
    offsets = (idx.year * 12 + idx.month) - (start.year * 12 + start.month)
    return offsets + 1 if one_based else offsets


def first_local_minimum(ami_dict):
    """[未启用] 仅供 select_tau 使用；τ 已固定为 1，本函数当前无调用点。"""
    lags = sorted(ami_dict)
    vals = [ami_dict[l] for l in lags]
    for i in range(1, len(vals) - 1):
        if vals[i] < vals[i - 1] and vals[i] <= vals[i + 1]:
            return lags[i]
    return None


def simplex_self_predict_rho(values, E, tau, exclusion_radius=None):
    """values需保留完整日历位置和真实NaN——本函数不做任何缺测预处理，直接靠pyEDM
    Simplex()默认的ignoreNan=True正确跳过需要缺测点的预测，不需要额外代码。

    如果这段数据缺测拼接过于零散(某个候选E/tau组合下连一个有效近邻都凑不出来)，
    pyEDM底层(scipy cKDTree.query)会直接抛异常而不是返回一个低分——已在Crooked_Lake
    的RegFlow上实测到这个情况(32段，最短的只有1个月)。这里接住异常返回NaN，让调用方
    (select_tau)把这个候选tau当作"此路不通"处理，不要让整个嵌入参数选择跟着崩掉。"""
    n = len(values)
    df = pd.DataFrame({"time": np.arange(n), "v": values})
    full = f"1 {n}"
    exclusion_radius = exclusion_radius if exclusion_radius is not None else max(tau, 1)
    try:
        res = pyEDM.Simplex(dataFrame=df, columns="v", target="v", lib=full, pred=full,
                             E=E, tau=-tau, exclusionRadius=exclusion_radius, embedded=False)
    except Exception:
        return np.nan
    obs, pred = res["Observations"], res["Predictions"]
    mask = obs.notna() & pred.notna()
    if mask.sum() < 2:
        return np.nan
    return np.corrcoef(obs[mask], pred[mask])[0, 1]


def select_tau(values, candidate_taus=(1, 2, 3), max_ami_lag=6, probe_E=2, fallback_tau=1):
    """
    [未启用] τ 已固定为 1（config.EMBED_TAU），本函数当前无调用点。
如果AMI选不出局部极小值、且fallback_selfpred这条路上全部候选tau都因为数据太碎
    而失败(simplex_self_predict_rho全返回NaN)，退回fallback_tau=1，不能让best_tau
    停留在None——那样下游get_embedding_params/ccm_edge_curve会直接因为tau=None崩溃。"""
    ami = average_mutual_information(values, max_lag=max_ami_lag)
    tau_star = first_local_minimum(ami)
    if tau_star is not None:
        return tau_star, {"method": "AMI_local_min", "ami": ami}
    best_tau, best_rho, probe_rhos = None, -np.inf, {}
    for tau in candidate_taus:
        rho = simplex_self_predict_rho(values, E=probe_E, tau=tau)
        probe_rhos[tau] = rho
        if not np.isnan(rho) and rho > best_rho:
            best_tau, best_rho = tau, rho
    if best_tau is None:
        return fallback_tau, {"method": "fallback_default_insufficient_data", "ami": ami, "probe_rhos": probe_rhos}
    return best_tau, {"method": "fallback_selfpred", "ami": ami, "probe_rhos": probe_rhos}


def select_E(values, tau, candidate_E=range(2, 11), fallback_E=2):
    """values需保留完整日历位置和真实NaN，理由同simplex_self_predict_rho。

    原来直接调pyEDM.EmbedDimension()一次性扫描全部候选E——这个函数内部用多进程池
    并行跑每个E，只要其中一个E因为数据太碎、找不到有效近邻而抛异常，整个进程池
    连带崩溃、拿不到任何E的结果(已在Crooked_Lake的RegFlow上实测到)。改成逐个E值
    单独调simplex_self_predict_rho、每个都单独接住异常，一个E失败不影响其他E值
    继续尝试；如果全部候选E都失败，退回fallback_E=2并在返回的curve里标注清楚，
    不能让整个湖泊的计算因为一个变量选不出E就整体失败。"""
    rows = []
    for E in candidate_E:
        rho = simplex_self_predict_rho(values, E=E, tau=tau)
        rows.append({"E": E, "rho": rho})
    curve = pd.DataFrame(rows)
    if curve["rho"].notna().any():
        best_row = curve.loc[curve["rho"].idxmax()]
        return int(best_row["E"]), curve
    curve = curve.copy()
    curve["note"] = "all_candidate_E_failed_insufficient_contiguous_data_using_fallback"
    return fallback_E, curve


def get_embedding_params(values, var_name, verbose=True, tau=None, candidate_E=None):
    """values必须是保留了完整日历月份位置和真实NaN的序列/数组——调用方不能先做
    .dropna()再传进来，否则E的选择会受"缺测拼接"问题影响。

    tau 现固定为 1（由 config.EMBED_TAU 提供），不再用 AMI/simplex 自动选择。三点理由：

    1. 文献先例：Javier et al. (2022, Physica A 604, 127893) 在同类水库系统的 CCM
       分析中即固定 τ=1，理由是"gives the highest resolution for the embedding"。
    2. 消除泄漏：原 select_tau() 依赖去季节化后的序列，而数据生成阶段的去季节化
       曾使用全序列（含测试期）气候态，导致 τ 携带测试期信息。实测 53 个变量中
       20 个（38%）的 τ 会因修正去季节化而改变。固定 τ 后该环节不复存在。
    3. 缩短嵌入跨度：原 τ∈[1,5] 配合 E∈[2,10]，嵌入跨度 (E-1)*τ 最长达 36 个月，
       在约 330 个月度观测且含缺测的序列上代价很大。实测跨度 25–40 个月的边
       有效样本中位数降至 299、显著率降至 30%（跨度 0–6 个月者为 329 / 46%）。
       τ=1 使跨度上限降为 E-1 ≤ 9 个月。

    E 仍由单变量 simplex 自预测选择，**不**改用"使交叉映射技巧最大"的选法——
    后者用被检验的量本身来选参数，存在循环论证问题。

    代价（需在方法论中披露）：月度序列自相关强，τ=1 时相邻滞后坐标高度冗余，
    嵌入向量沿对角线方向退化（Fraser & Swinney, 1986 提出 AMI 选 τ 正是为此）。
    该偏差通过 IAAFT 替代序列控制——替代序列保留原序列功率谱因而保留自相关结构，
    观测数据与零分布走完全相同的嵌入流程，冗余对两者影响一致。
    """
    if tau is None:
        tau = _config_embed_tau()
    if candidate_E is None:
        candidate_E = _config_embed_E_candidates()
    E, e_curve = select_E(values, tau, candidate_E=candidate_E)
    if verbose:
        print(f"[{var_name}] tau={tau} (fixed), E={E}")
    return {"variable": var_name, "tau": tau, "E": E,
            "tau_info": {"method": "fixed", "value": tau}, "E_curve": e_curve}


def _config_embed_tau():
    """从 config.py 读 τ；config 不可导入时退回 1（与 config 默认值一致）。"""
    try:
        import config
        return config.EMBED_TAU
    except Exception:
        return 1


def _config_embed_E_candidates():
    try:
        import config
        return config.EMBED_E_CANDIDATES
    except Exception:
        return range(2, 11)


# ============ CCM边曲线、收敛诊断 ============

def convergence_test(curve_df, alpha=0.05):
    """CCM收敛检验：用Kendall's tau检验rho是否随library size表现出统计显著的单调上升趋势
    (Mann, 1945, Econometrica, 'Nonparametric tests against trend'; Kendall, 1938, Biometrika,
    'A new measure of rank correlation')——这是检验"一个序列相对另一个序列有无显著单调趋势"
    的标准非参数方法，直接对应Sugihara et al.(2012)对CCM收敛的定性定义："rho随library size
    增大而上升、并在library size最大时统计显著"，不是一个凭经验拍出来的固定数值。"""
    curve = curve_df.sort_values("LibSize").dropna(subset=["LibSize", "rho"])
    if len(curve) < 4:
        return {
            "rho_start": np.nan, "rho_full": np.nan, "rho_gain": np.nan,
            "tail_slope": np.nan, "tail_drop": np.nan,
            "kendall_tau": np.nan, "kendall_p": np.nan,
            "convergence_diagnostic_pass": False,
            "convergence_note": "too_few_library_sizes",
        }
    rho_start = float(curve["rho"].iloc[0])
    rho_full = float(curve["rho"].iloc[-1])
    rho_gain = rho_full - rho_start
    tail = curve.tail(min(3, len(curve)))
    tail_slope = float(np.polyfit(tail["LibSize"], tail["rho"], 1)[0]) if len(tail) >= 2 else np.nan
    tail_drop = float(max(0.0, tail["rho"].iloc[0] - tail["rho"].iloc[-1]))
    tau, p = stats.kendalltau(curve["LibSize"], curve["rho"])
    tau, p = float(tau), float(p)
    diagnostic_pass = bool((rho_full > 0) and (tau > 0) and (p < alpha))
    note = "diagnostic_pass" if diagnostic_pass else "diagnostic_fail"
    return {
        "rho_start": rho_start, "rho_full": rho_full, "rho_gain": rho_gain,
        "tail_slope": tail_slope, "tail_drop": tail_drop,
        "kendall_tau": tau, "kendall_p": p,
        "convergence_diagnostic_pass": diagnostic_pass,
        "convergence_note": note,
    }


def ccm_edge_curve(df_time, cause, effect, embed_params, lib_sizes, sample, seed=0):
    """检验 cause -> effect。返回 LibSize/rho 曲线。df_time应已经是只含[time, cause, effect]
    的完整日历月份子表(由调用方按_pair_panel准备好，真实NaN原样保留、不要dropna过)。

    如果这条边的数据缺测拼接过于零散，pyEDM可能连有效近邻都找不到、直接抛异常(同
    select_E里遇到的问题)——这里接住异常，返回一条rho全为NaN的曲线，让下游
    convergence_test(len(curve)<4分支)/surrogate_significance/build_full_ccm_table
    走"数据不足"分支，不要让一条边的失败拖垮整个湖泊其他干净边的计算。"""
    E_eff, tau_eff = embed_params[effect]["E"], embed_params[effect]["tau"]
    try:
        res = pyEDM.CCM(dataFrame=df_time, columns=effect, target=cause, E=E_eff, tau=-tau_eff,
                         libSizes=lib_sizes, sample=sample, seed=seed, **CCM_SEQUENTIAL_KWARGS)
        col = f"{effect}:{cause}"
        out = res[["LibSize", col]].rename(columns={col: "rho"})
    except Exception as e:
        parts = [int(x) for x in str(lib_sizes).split()]
        lib_vals = list(range(parts[0], parts[1] + 1, parts[2])) if len(parts) == 3 else [np.nan]
        out = pd.DataFrame({"LibSize": lib_vals, "rho": np.nan})
        out.attrs["ccm_error"] = f"{type(e).__name__}: {e}"
    out["cause"], out["effect"] = cause, effect
    return out


def _pair_panel(panel_df, cols):
    """取指定的两列，保留完整日历月份索引和真实NaN——**不dropna、不因为缺测而删行**。
    只把DatetimeIndex换算成time列(calendar_month_offsets)、重置成pyEDM要的位置索引，
    位置索引里对应的行该是NaN的还是NaN，行数就是这两列共同覆盖的完整日历月份数(不是
    实际有数据的月份数)。

    这样处理之后，pyEDM.CCM/Simplex默认的ignoreNan=True会自动跳过任何需要缺测点的
    延迟嵌入或预测，不会把物理上不相邻的两个观测点错误地拼成"相邻"——这是这次重写
    唯一改动的核心逻辑，细节见模块顶部说明。"""
    sub = panel_df[cols]
    df_time = sub.reset_index(drop=True)
    df_time.insert(0, "time", calendar_month_offsets(sub.index))
    return df_time


def _segment_stats(panel_df, cols):
    """诊断用，不参与CCM计算本身：统计这两列共同非缺测的行，实际拆成了几段真正连续的
    日历月份区间，以及最长的一段占整体的比例。用于报告数据质量、判断某条边的结果该给
    多大的信心权重，不是用来修复缺测问题的(缺测问题已经靠_pair_panel+pyEDM的ignoreNan
    解决了，这里纯粹是把"这条边的数据碎不碎"这个信息暴露出来，方便写论文时引用)。"""
    both_present = panel_df[cols].notna().all(axis=1)
    n_obs = int(both_present.sum())
    if n_obs == 0:
        return {"n_obs": 0, "num_segments": 0, "max_seg_len": 0, "max_seg_pct": np.nan}
    present_idx = panel_df.index[both_present]
    month_num = present_idx.year * 12 + present_idx.month
    diffs = np.diff(month_num.values)
    seg_breaks = np.where(diffs > 1)[0]
    seg_lengths = np.diff(np.concatenate(([0], seg_breaks + 1, [n_obs])))
    max_seg = int(seg_lengths.max())
    return {
        "n_obs": n_obs, "num_segments": int(len(seg_lengths)),
        "max_seg_len": max_seg, "max_seg_pct": round(100 * max_seg / n_obs, 1),
    }


def run_pairwise_ccm(panel_df, embed_params, sample=50, seed=0, lib_start_frac=1 / 3, lib_step=10):
    """对 panel_df 里出现的全部变量做完整两两有向 CCM。每条边取数据时保留完整日历月份索引
    (_pair_panel)，library size的范围按这两列共同覆盖的日历跨度(n_span)算，不是按实际
    有数据的行数算——这样pyEDM在更大的library size下依然能利用跨越缺口两侧的干净片段，
    不会被人为砍短。"""
    variables = [v for v in VARS if v in panel_df.columns]
    curves = {}
    for cause, effect in itertools.permutations(variables, 2):
        pair_df = _pair_panel(panel_df, [cause, effect])
        n_span = len(pair_df)
        lib_sizes = f"{max(10, int(n_span * lib_start_frac))} {n_span} {lib_step}"
        curves[(cause, effect)] = ccm_edge_curve(pair_df, cause, effect, embed_params, lib_sizes, sample, seed)
    return curves


# ============ IAAFT替代数据显著性检验 + FDR ============

def iaaft_surrogate(x, n_iter=100, rng=None):
    """Schreiber & Schmitz (1996) IAAFT surrogate。x必须是不含NaN的纯观测值数组——
    这个函数本身不处理缺测(FFT要求规则采样)，调用方(surrogate_significance)负责只把
    cause列实际有观测的那些值传进来，再把生成的替代值放回原来的位置，缺测位置保持NaN。"""
    rng = rng or np.random.default_rng()
    x = np.asarray(x, dtype=float)
    n = len(x)
    sorted_x = np.sort(x)
    amp = np.abs(np.fft.rfft(x))
    surrogate = rng.permutation(x)
    for _ in range(n_iter):
        s_fft = np.fft.rfft(surrogate)
        new_fft = amp * np.exp(1j * np.angle(s_fft))
        s = np.fft.irfft(new_fft, n=n)
        surrogate = sorted_x[np.argsort(np.argsort(s))]
    return surrogate


def surrogate_significance(panel_df, cause, effect, embed_params, n_surrogates=200, seed=0):
    """取(cause, effect)两列做IAAFT替代数据检验，跟_pair_panel/ccm_edge_curve用同一套
    "保留完整日历索引、真实NaN不dropna"的逻辑，不再像旧版那样自己单独写一遍dropna。

    IAAFT替代序列只在cause实际有观测的月份上生成(iaaft_surrogate要求规则采样、不能
    直接喂NaN)，生成完之后放回cause原来的观测位置，其余月份维持NaN、effect列完全不动——
    这样替代数据版本跟观测数据版本用的是同一套缺测掩膜，CCM计算时pyEDM会用同样的方式
    跳过缺测，obs_rho和null_rhos才是可比的。"""
    df_time = _pair_panel(panel_df, [cause, effect])
    n_span = len(df_time)
    full_libsize = f"{n_span} {n_span} 1"
    obs_rho = ccm_edge_curve(df_time, cause, effect, embed_params, full_libsize, sample=1, seed=seed)["rho"].iloc[-1]
    seg_stats = _segment_stats(panel_df, [cause, effect])

    # obs_rho是NaN说明这条边数据太碎、pyEDM连观测数据本身都算不出rho(ccm_edge_curve
    # 已经接住了底层异常)——这种情况不能再往下算替代数据检验：`null_rhos >= NaN`在
    # numpy里全部判定为False，会让p_value=(0+1)/(n+1)算出一个看起来"显著"的假结果，
    # 必须在这里提前拦截，明确标注"数据不足"而不是悄悄给一个误导性的小p值。
    if pd.isna(obs_rho):
        return {"cause": cause, "effect": effect, "obs_rho": np.nan, "p_value": np.nan,
                "null_mean": np.nan, "null_std": np.nan, "n_span": n_span,
                "insufficient_data": True, **seg_stats}

    cause_observed_mask = df_time[cause].notna()
    cause_observed_values = df_time.loc[cause_observed_mask, cause].values

    rng = np.random.default_rng(seed)
    null_rhos = np.empty(n_surrogates)
    for i in range(n_surrogates):
        df_s = df_time.copy()
        surrogate_vals = iaaft_surrogate(cause_observed_values, n_iter=100, rng=rng)
        df_s.loc[cause_observed_mask, cause] = surrogate_vals
        null_rhos[i] = ccm_edge_curve(df_s, cause, effect, embed_params, full_libsize, sample=1, seed=seed)["rho"].iloc[-1]
    valid_null = null_rhos[~np.isnan(null_rhos)]
    if len(valid_null) < n_surrogates * 0.5:
        # 太多替代数据也算不出rho(同样是数据太碎的信号)，剩下能用的替代样本不够支撑
        # 一个可信的p值，同样标注成数据不足，不要硬凑一个基于少数几个替代样本的p值。
        return {"cause": cause, "effect": effect, "obs_rho": obs_rho, "p_value": np.nan,
                "null_mean": np.nanmean(null_rhos), "null_std": np.nanstd(null_rhos), "n_span": n_span,
                "insufficient_data": True, "n_valid_surrogates": int(len(valid_null)), **seg_stats}
    p_value = (np.sum(valid_null >= obs_rho) + 1) / (len(valid_null) + 1)
    return {"cause": cause, "effect": effect, "obs_rho": obs_rho, "p_value": p_value,
            "null_mean": float(np.mean(valid_null)), "null_std": float(np.std(valid_null)),
            "n_span": n_span, "n_valid_surrogates": int(len(valid_null)), **seg_stats}


def build_full_ccm_table(panel_df, embed_params, sample=50, n_surrogates=200, seed=0, verbose=True):
    """CCM曲线 -> 替代数据显著性 -> FDR校正，并附收敛曲线诊断 + 数据碎片化诊断(num_segments/
    max_seg_pct，见_segment_stats)。n_obs列现在指这条边实际两列都有观测的月份数(诊断用途)，
    n_span列是这两列共同覆盖的完整日历跨度(CCM library size的分母)——旧版只有一个n_obs，
    含义是"dropna后剩下的行数"，跟n_span混在一起用，这次分开成两列。"""
    curves = run_pairwise_ccm(panel_df, embed_params, sample=sample, seed=seed)
    rows = []
    for (cause, effect), curve in curves.items():
        conv = convergence_test(curve)
        if verbose:
            print(f"  {cause} -> {effect}: rho(full)={curve['rho'].iloc[-1]:.3f}, convergence_diagnostic={conv['convergence_diagnostic_pass']}")
        sig = surrogate_significance(panel_df, cause, effect, embed_params, n_surrogates=n_surrogates, seed=seed)
        rows.append({**sig, **conv, "curve": curve})
    table = pd.DataFrame(rows)
    # 数据太碎的边(surrogate_significance标了insufficient_data=True)p_value是NaN——
    # multipletests不认NaN，只对有效p值的行做FDR校正，NaN行的p_fdr/significant保持
    # NaN/False，不要让这几条边的NaN把FDR校正对其他干净边的结果搞坏。
    table["p_fdr"] = np.nan
    valid = table["p_value"].notna()
    if valid.any():
        _, fdr_p, _, _ = multipletests(table.loc[valid, "p_value"], alpha=0.05, method="fdr_bh")
        table.loc[valid, "p_fdr"] = fdr_p
    table["significant"] = table["p_fdr"] < 0.05  # NaN < 0.05 -> False，天然处理了插不进来的行
    # causal_evidence必须同时满足FDR显著和收敛诊断通过两个条件——之前只用FDR显著一个条件，
    # 会把"rho随library size增大而收敛"这个CCM因果推断的核心证据晾在一边，允许收敛没通过
    # 的边一样被下游build_causal_network/select_sarimax_candidates/find_common_drivers当成
    # 真实因果边使用。收敛诊断本身(convergence_test)在数据不足(len(curve)<4)时已经返回
    # 明确的False、不会是NaN，这里用fillna(False)只是防御性写法。
    table["causal_evidence"] = table["significant"] & table["convergence_diagnostic_pass"].fillna(False)
    table["causal_evidence_note"] = np.where(
        table["causal_evidence"],
        "FDR_significant; convergence diagnostic passed",
        np.where(
            table["significant"],
            "FDR_significant but convergence diagnostic not passed/unclear -- excluded from causal_evidence",
            "not FDR significant",
        ),
    )
    table.loc[~valid, "causal_evidence_note"] = "insufficient_data (data too fragmented for pyEDM to compute)"
    return table


# ============ 滞后扫描 ============

def lag_scan(panel_df, cause, effect, embed_params, lags=None, seed=0):
    """跟_pair_panel同一套逻辑：effect整体平移l个月后，保留完整日历索引、不dropna。"""
    if lags is None:
        lags = CCM_LAGS if "CCM_LAGS" in globals() else range(-12, 13)
    results = []
    for l in lags:
        col_name = f"{effect}__lag{l}"
        pair = panel_df[[cause, effect]].copy()
        pair[col_name] = panel_df[effect].shift(-l)
        pair = pair[[cause, col_name]]
        df_time = pair.reset_index(drop=True)
        df_time.insert(0, "time", calendar_month_offsets(pair.index))
        n_span = len(df_time)
        n_obs = int(pair.notna().all(axis=1).sum())
        ep = dict(embed_params)
        ep[col_name] = embed_params[effect]
        curve = ccm_edge_curve(df_time, cause, col_name, ep, f"{n_span} {n_span} 1", sample=1, seed=seed)
        results.append({"lag": l, "rho": curve["rho"].iloc[-1], "n_obs": n_obs, "n_span": n_span})
    return pd.DataFrame(results)


def run_lag_scans(panel_df, embed_params, ccm_table, lags=range(0, 13), seed=0, verbose=True):
    out = {}
    for _, row in ccm_table[ccm_table["causal_evidence"]].iterrows():
        cause, effect = row["cause"], row["effect"]
        scan = lag_scan(panel_df, cause, effect, embed_params, lags=lags, seed=seed)
        l_star = int(scan.loc[scan["rho"].idxmax(), "lag"])
        out[(cause, effect)] = {"scan": scan, "l_star": l_star, "rho_at_l_star": scan["rho"].max()}
        if verbose:
            print(f"  {cause} -> {effect}: l* = {l_star} 个月 (rho={out[(cause, effect)]['rho_at_l_star']:.3f})")
    return out


# ============ 因果网络、双向边分类、共同驱动、中介链条（未改动） ============

def build_causal_network(ccm_table):
    import networkx as nx
    G = nx.DiGraph()
    for _, row in ccm_table.iterrows():
        if row["causal_evidence"]:
            G.add_edge(row["cause"], row["effect"], rho=row["obs_rho"], p_fdr=row["p_fdr"], convergence_diagnostic_pass=row.get("convergence_diagnostic_pass"))
    return G


def classify_edges(G, lag_results=None):
    """lag_results来自run_lag_scans(..., lags=range(-12,13))：l_star>=0代表"因在前、果在后"，
    是符合因果常识的方向；l_star<0代表要用"未来的值"去解释"过去的值"，按Ye, Deyle, Gilarranz
    & Sugihara (2015, Sci Rep)的判据，这是强制单向驱动造成的广义同步(generalized synchrony)
    假象，不是真实反馈。"""
    pairs = set(tuple(sorted([a, b])) for a, b in itertools.permutations(G.nodes(), 2))
    rows = []
    for a, b in pairs:
        ab, ba = G.has_edge(a, b), G.has_edge(b, a)
        if ab and ba:
            stronger = a if G[a][b]["rho"] > G[b][a]["rho"] else b
            row = {"pair": (a, b), "type": "bidirectional (case i)",
                   "rho_a_to_b": G[a][b]["rho"], "rho_b_to_a": G[b][a]["rho"], "stronger_direction": stronger}
            if lag_results is not None:
                l_ab = lag_results.get((a, b), {}).get("l_star")
                l_ba = lag_results.get((b, a), {}).get("l_star")
                row["l_star_a_to_b"] = l_ab
                row["l_star_b_to_a"] = l_ba
                if l_ab is not None and l_ba is not None:
                    if l_ab >= 0 and l_ba >= 0:
                        row["synchrony_check"] = "confirmed_bidirectional"
                    elif l_ab >= 0 > l_ba:
                        row.update(type="unidirectional (case ii, downgraded from bidirectional)",
                                   driver=a, receiver=b, rho=G[a][b]["rho"],
                                   synchrony_check=f"{b}->{a}方向l_star={l_ba}<0，判定为同步伪影，剔除")
                    elif l_ba >= 0 > l_ab:
                        row.update(type="unidirectional (case ii, downgraded from bidirectional)",
                                   driver=b, receiver=a, rho=G[b][a]["rho"],
                                   synchrony_check=f"{a}->{b}方向l_star={l_ab}<0，判定为同步伪影，剔除")
                    else:
                        row["synchrony_check"] = "both_lags_negative_unusual_manual_review_needed"
                else:
                    row["synchrony_check"] = "missing_lag_scan_data"
            rows.append(row)
        elif ab or ba:
            drv, rcv = (a, b) if ab else (b, a)
            rows.append({"pair": (a, b), "type": "unidirectional (case ii)", "driver": drv, "receiver": rcv,
                         "rho": G[drv][rcv]["rho"]})
    return pd.DataFrame(rows)


def find_common_drivers(G):
    nodes = list(G.nodes())
    rows = []
    for a, b in itertools.combinations(nodes, 2):
        if G.has_edge(a, b) or G.has_edge(b, a):
            continue
        for z in nodes:
            if z in (a, b):
                continue
            if G.has_edge(z, a) and G.has_edge(z, b):
                rows.append({"A": a, "B": b, "common_driver": z,
                             "rho_z_to_A": G[z][a]["rho"], "rho_z_to_B": G[z][b]["rho"]})
    return pd.DataFrame(rows)


def find_transitive_mediation(G):
    nodes = list(G.nodes())
    rows = []
    for p, m, w in itertools.permutations(nodes, 3):
        if G.has_edge(p, m) and G.has_edge(m, w) and G.has_edge(p, w):
            rows.append({"cause": p, "mediator": m, "final_effect": w,
                         "direct_rho_P_W": G[p][w]["rho"], "rho_P_M": G[p][m]["rho"], "rho_M_W": G[m][w]["rho"]})
    return pd.DataFrame(rows)


def find_full_and_partial_mediation(G):
    """识别P->M->W路径共现模式；这是pairwise CCM网络摘要，不是正式中介检验。"""
    nodes = list(G.nodes())
    rows = []
    for p, m, w in itertools.permutations(nodes, 3):
        if G.has_edge(p, m) and G.has_edge(m, w):
            direct_exists = G.has_edge(p, w)
            rows.append({
                "cause": p, "path_variable": m, "final_effect": w,
                "rho_P_M": G[p][m]["rho"], "rho_M_W": G[m][w]["rho"],
                "direct_P_W_exists": direct_exists,
                "direct_rho_P_W": G[p][w]["rho"] if direct_exists else None,
                "pairwise_path_pattern": "间接路径+直接边共现" if direct_exists else "仅检出间接路径",
                "interpretation_note": "两两CCM路径共现；不能视为正式条件化中介检验",
            })
    return pd.DataFrame(rows)


# ============ SARIMAX候选变量筛选（未改动） ============

def select_sarimax_candidates(ccm_table, lag_results, common_drivers_df):
    wl_edges = ccm_table[(ccm_table["effect"] == "WL") & (ccm_table["causal_evidence"])]
    rows = []
    for _, row in wl_edges.iterrows():
        var = row["cause"]
        lag_info = lag_results.get((var, "WL"), {})
        is_common_driver_only = (
            (common_drivers_df["A"] == var).any() or (common_drivers_df["B"] == var).any()
        ) if len(common_drivers_df) else False
        rows.append({
            "variable": var, "rho_to_WL": row["obs_rho"], "p_fdr": row["p_fdr"],
            "convergence_diagnostic_pass": row.get("convergence_diagnostic_pass"),
            "kendall_tau": row.get("kendall_tau"), "kendall_p": row.get("kendall_p"),
            "rho_gain": row.get("rho_gain"), "tail_drop": row.get("tail_drop"),
            "l_star_months": lag_info.get("l_star"), "rho_at_l_star": lag_info.get("rho_at_l_star"),
            "flag_possible_common_driver_artifact": is_common_driver_only,
            "note": "需人工核对 l* 的水文学合理性 + SARIMAX 拟合后的 AIC/BIC/VIF + 样本外回测",
        })
    cols = ["variable", "rho_to_WL", "p_fdr", "convergence_diagnostic_pass", "kendall_tau", "kendall_p",
            "rho_gain", "tail_drop", "l_star_months", "rho_at_l_star", "flag_possible_common_driver_artifact", "note"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols].sort_values("rho_to_WL", ascending=False)


# ============ SARIMA/SARIMAX/XGBoost预测函数（未改动） ============

def _forecast_metrics(actual, predicted):
    """RMSE + MAE + NSE + PBIAS；自动剔除测试集中实际值或预测值缺失的月份。
    PBIAS按Moriasi et al. (2015)定义：100*sum(actual-predicted)/sum(actual)，
    正值=模型系统性低估，负值=系统性高估。注意：WL是去季节化后的距平序列，
    sum(actual)在某些测试窗口里可能接近0(不像原始流量序列恒为正)，此时PBIAS
    分母不稳定，返回NaN而不是给出一个没有意义的极端值——跨湖泊比较PBIAS时
    只看分母不接近0的情形。NSE/KGE不适合跨站点比较精度(Williams, 2025,
    *Environmental Modelling & Software*)，跨湖泊比较应优先看RMSE(原始单位)
    + PBIAS(偏差方向，理论上不受各湖泊自身方差影响)，NSE只用于湖泊内部比较
    不同方法。"""
    actual = pd.Series(np.asarray(actual, dtype=float))
    predicted = pd.Series(np.asarray(predicted, dtype=float))
    mask = actual.notna() & predicted.notna()
    if not mask.any():
        return {"rmse": np.nan, "mae": np.nan, "nse": np.nan, "pbias": np.nan, "n_eval": 0}
    a, p = actual[mask].values, predicted[mask].values
    rmse = np.sqrt(np.mean((p - a) ** 2))
    mae = np.mean(np.abs(p - a))
    denom = np.sum((a - a.mean()) ** 2)
    nse = np.nan if denom == 0 else 1 - np.sum((a - p) ** 2) / denom
    sum_a = np.sum(a)
    pbias_unstable = abs(sum_a) < 1e-6 * max(1.0, np.sum(np.abs(a)))
    pbias = np.nan if pbias_unstable else 100.0 * np.sum(a - p) / sum_a
    return {"rmse": rmse, "mae": mae, "nse": nse, "pbias": pbias, "n_eval": int(mask.sum())}


def _fill_training_block(obj):
    """只用于训练集：线性插值后前后填充，避免跨越train/test边界。"""
    return obj.interpolate(method="linear").ffill().bfill()


def compute_vif(exog_df):
    """计算多外生变量SARIMAX训练矩阵的VIF；单变量时VIF不适用，返回NaN。"""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X = exog_df.dropna()
    if X.shape[1] < 2 or len(X) <= X.shape[1]:
        return pd.Series([np.nan] * X.shape[1], index=X.columns, dtype=float)
    return pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)


def _arima_fit_is_degenerate(model):
    """检测auto_arima选中的模型是否数值退化(协方差矩阵奇异/近奇异)。这类模型的AIC经常
    异常低、看似"最优"，实际上标准误/z值/置信区间全部失真(bse里出现NaN/inf)，出现这种情况
    应当当作拟合失败处理，不能采信它的AIC排序结果(排查Lake_of_the_Prairies时发现的问题)。"""
    try:
        bse = np.asarray(model.arima_res_.bse, dtype=float)
    except Exception:
        return False
    return len(bse) == 0 or not np.all(np.isfinite(bse))


def fit_auto_sarima(wl_series, test_size=12, m=12):
    import pmdarima as pm
    wl_series = wl_series.sort_index().asfreq("MS")
    train_raw, test = wl_series[:-test_size], wl_series[-test_size:]
    train = _fill_training_block(train_raw).dropna()
    model = pm.auto_arima(train, seasonal=True, m=m, stepwise=True, suppress_warnings=True, error_action="ignore")
    if _arima_fit_is_degenerate(model):
        raise ValueError(f"auto_arima选中的模型{model.order}x{model.seasonal_order}协方差矩阵奇异/近奇异，判定为拟合失败")
    preds, conf_int = model.predict(n_periods=test_size, return_conf_int=True)
    fc = pd.Series(np.asarray(preds), index=test.index)
    conf_df = pd.DataFrame(np.asarray(conf_int), index=test.index, columns=["lower", "upper"])
    return {**_forecast_metrics(test.values, fc.values), "order": model.order, "seasonal_order": model.seasonal_order,
            "predicted": fc, "actual": test, "conf_int": conf_df}


def fit_auto_sarimax_multi(wl_series, exog_df, exog_lags, test_size=12, m=12):
    """SARIMAX整块提前预测，hindcast实验设计：外生变量用真实观测值(按各自滞后exog_lags)，
    不要求滞后覆盖整个test_size预测窗口——目的是隔离"变量选择方法本身"的效果，不模拟业务
    预报里"未来气象输入未知"这个约束(observed/perfect meteorological forcing，水文预测
    文献常见做法)。"""
    import pmdarima as pm
    wl_series = wl_series.sort_index().asfreq("MS")
    exog_df = exog_df.sort_index().asfreq("MS")
    exog_shifted = pd.DataFrame({col: exog_df[col].shift(lag) for col, lag in exog_lags.items()}, index=exog_df.index)
    combined = pd.concat([wl_series.rename("WL"), exog_shifted], axis=1).reindex(wl_series.index)
    train_raw, test_raw = combined.iloc[:-test_size], combined.iloc[-test_size:]
    test_wl = test_raw.iloc[:, 0]
    test_exog_raw = test_raw.iloc[:, 1:]

    train = _fill_training_block(train_raw).dropna()
    train_wl, train_exog = train.iloc[:, 0], train.iloc[:, 1:]
    test_exog = test_exog_raw.ffill()
    if test_exog.isna().any().any():
        raise ValueError("预测起点前可用的滞后外生变量不足，SARIMAX无法完成整块预测")
    model = pm.auto_arima(train_wl, X=train_exog.values, seasonal=True, m=m,
                          stepwise=True, suppress_warnings=True, error_action="ignore")
    if _arima_fit_is_degenerate(model):
        raise ValueError(f"auto_arima选中的模型{model.order}x{model.seasonal_order}协方差矩阵奇异/近奇异，判定为拟合失败")
    preds, conf_int = model.predict(n_periods=test_size, X=test_exog.values, return_conf_int=True)
    fc = pd.Series(np.asarray(preds), index=test_raw.index)
    conf_df = pd.DataFrame(np.asarray(conf_int), index=test_raw.index, columns=["lower", "upper"])
    metrics = _forecast_metrics(test_wl.values, fc.values)
    return {**metrics, "order": model.order, "seasonal_order": model.seasonal_order, "vif": compute_vif(train_exog),
            "predicted": fc, "actual": test_wl, "conf_int": conf_df}


def fit_climatology(wl_series, test_size=12, m=12):
    wl_series = wl_series.sort_index().asfreq("MS")
    train_raw, test = wl_series[:-test_size], wl_series[-test_size:]
    train = _fill_training_block(train_raw).dropna()
    monthly_clim = train.groupby(train.index.month).mean()
    fc = pd.Series([monthly_clim.get(mo, np.nan) for mo in test.index.month], index=test.index)
    return {**_forecast_metrics(test.values, fc.values), "predicted": fc, "actual": test}


def fit_xgboost_multi(wl_series, exog_df, exog_lags, test_size=12, wl_ar_lags=(1, 2, 3, 6, 12),
                       n_estimators=200, max_depth=3, learning_rate=0.05):
    """XGBoost对照，跟fit_auto_sarimax_multi同一套hindcast实验设计：外生变量按各自exog_lags
    滞后使用真实观测值，不做递归多步预测(不把上一步的点预测喂回去当下一步输入)——WL自身的
    自回归特征(wl_ar_lags)同样用真实历史值构造，跟外生变量的处理哲学一致：隔离"特征/变量
    选择方法本身"的效果，不模拟操作性预报里"未来完全未知"这个约束(exog_df/exog_lags为空
    dict时就是纯AR基线，对应SARIMA不用外生变量的版本)。

    树模型超参数固定为保守的默认值，不逐湖网格搜索调参——n~350个月的小样本上做超参数搜索
    容易过拟合验证集，固定一组浅树/低学习率的保守配置更适合这个数据规模，这是有意的选择
    不是偷懒。
    """
    import xgboost as xgb

    wl_series = wl_series.sort_index().asfreq("MS")
    exog_df = exog_df.sort_index().asfreq("MS") if len(exog_df.columns) else exog_df
    exog_shifted = pd.DataFrame({col: exog_df[col].shift(lag) for col, lag in exog_lags.items()}, index=wl_series.index)
    wl_ar = pd.DataFrame({f"WL_lag{k}": wl_series.shift(k) for k in wl_ar_lags}, index=wl_series.index)
    features = pd.concat([wl_ar, exog_shifted], axis=1)
    combined = pd.concat([wl_series.rename("WL"), features], axis=1).reindex(wl_series.index)

    train_raw, test_raw = combined.iloc[:-test_size], combined.iloc[-test_size:]
    test_wl = test_raw["WL"]
    test_X_raw = test_raw.drop(columns="WL")

    train = _fill_training_block(train_raw).dropna()
    train_wl, train_X = train["WL"], train.drop(columns="WL")
    test_X = test_X_raw.ffill()
    if test_X.isna().any().any():
        raise ValueError("预测起点前可用的滞后特征不足，XGBoost无法完成整块预测")

    model = xgb.XGBRegressor(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
        subsample=0.8, colsample_bytree=0.8, random_state=0,
    )
    model.fit(train_X, train_wl)
    preds = model.predict(test_X)
    fc = pd.Series(preds, index=test_raw.index)
    metrics = _forecast_metrics(test_wl.values, fc.values)
    importance = pd.Series(model.feature_importances_, index=train_X.columns).sort_values(ascending=False)
    return {**metrics, "predicted": fc, "actual": test_wl, "feature_importance": importance}
