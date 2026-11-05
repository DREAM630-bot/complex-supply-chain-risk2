# -*- coding: utf-8 -*-
"""
气象动态数据模块：Open-Meteo Archive API 真实拉取与安全截断
底层数据源：ECMWF ERA5/ERA5-Land 再分析数据
内置 safe_end_date 函数，利用 ERA5 约5天归档延迟特性截断请求区间
异常时启用文本级兜底字典，遵循零插值/零虚构原则
"""
import requests
from datetime import date, timedelta

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
ERA5_ARCHIVE_DELAY_DAYS = 5  # ERA5 归档延迟（约5天）
DEFAULT_WINDOW_START = "2025-09-01"
DEFAULT_WINDOW_END = "2026-09-30"


def safe_end_date(requested_end: str, today: date = None) -> str:
    """将请求截止日截断在已归档区间内，避免对未归档日期请求产生虚假数值"""
    if today is None:
        today = date.today()
    try:
        end = date.fromisoformat(requested_end)
    except (ValueError, TypeError):
        end = today
    safe = min(end, today - timedelta(days=ERA5_ARCHIVE_DELAY_DAYS))
    return safe.isoformat()


def fetch_node_weather(lat: float, lon: float, node_id: str,
                       start: str = DEFAULT_WINDOW_START,
                       end: str = DEFAULT_WINDOW_END,
                       timeout: int = 20) -> dict:
    """拉取单节点窗口内逐日降水与风速极值，返回结构化结果（含状态标记）"""
    safe_end = safe_end_date(end)
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": safe_end,
        "daily": "precipitation_sum,wind_speed_10m_max",
        "timezone": "Asia/Shanghai"
    }
    result = {"node_id": node_id, "lat": lat, "lon": lon,
              "start": start, "end": safe_end, "status": "ok", "data": []}
    try:
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        j = resp.json().get("daily", {})
        times = j.get("time", [])
        prec = j.get("precipitation_sum", [])
        wind = j.get("wind_speed_10m_max", [])
        for i, t in enumerate(times):
            p = prec[i] if i < len(prec) else None
            w = wind[i] if i < len(wind) else None
            if p is None or w is None:
                continue  # 关键极值缺失标记缺失而非伪造
            result["data"].append({"date": t, "precipitation": p, "wind_max": w})
        if not result["data"]:
            result["status"] = "empty"
    except Exception as e:
        result["status"] = "fallback"
        result["error"] = str(e)
    return result


def summarize_extremes(weather_result: dict) -> dict:
    """汇总节点极端天气指标，用于生成非结构化气象文本"""
    if weather_result["status"] != "ok" or not weather_result["data"]:
        return {"node_id": weather_result["node_id"], "status": weather_result["status"],
                "max_precip": None, "max_wind": None,
                "storm_days": 0, "rainstorm_days": 0}
    data = weather_result["data"]
    precs = [d["precipitation"] for d in data]
    winds = [d["wind_max"] for d in data]
    rainstorm_days = sum(1 for p in precs if p >= 50)      # 暴雨: >=50mm/日
    storm_days = sum(1 for w in winds if w >= 17.2)          # 大风: >=17.2m/s (8级)
    return {"node_id": weather_result["node_id"], "status": "ok",
            "max_precip": max(precs), "max_wind": max(winds),
            "rainstorm_days": rainstorm_days, "storm_days": storm_days,
            "total_days": len(data)}


def generate_weather_text(summary: dict, node_name: str) -> str:
    """基于极端天气统计生成非结构化气象描述文本（供多智能体流水线处理）"""
    if summary["status"] != "ok":
        return (f"{node_name}在观测窗口内气象数据获取异常（{summary['status']}），"
                f"触发文本级兜底机制补充上下文，不参与数值插值。")
    parts = [f"{node_name}在2025年9月至2026年9月观测窗口内"]
    if summary["rainstorm_days"] > 0:
        parts.append(f"出现{summary['rainstorm_days']}天暴雨（日降水≥50mm），"
                     f"最大日降水量达{summary['max_precip']:.1f}mm")
    else:
        parts.append(f"未出现暴雨级别降水，最大日降水量为{summary['max_precip']:.1f}mm")
    if summary["storm_days"] > 0:
        parts.append(f"另有{summary['storm_days']}天出现8级以上大风，"
                     f"最大风速达{summary['max_wind']:.1f}m/s")
    else:
        parts.append(f"最大风速为{summary['max_wind']:.1f}m/s，未达8级大风标准")
    return "，".join(parts) + "。"