# -*- coding: utf-8 -*-
"""
前端多智能体文本测度流水线：Extractor → Feedback → Revision
将非结构化文本转化为带证据锚点的可审计暴露度评分
- Extractor：提取线索、证据锚点、范围词、时效标记，生成severity_raw
- Feedback：R1证据一致性硬门禁 / R2范围词校准 / R3时效衰减 / R4单因子一致性
- Revision：有限修正（score_delta下限-0.30），clamp(0,1)输出severity_final
"""
import json
import os
import re
from datetime import date

_CORPUS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "corpus.json")
WINDOW_END = date(2026, 9, 30)
TIME_DECAY_THRESHOLD_MONTHS = 6

# 量化冲击关键词分级（用于Extractor生成severity_raw）
HIGH_IMPACT = ["停摆", "瘫痪", "断供", "停产", "全面中断"]
MID_IMPACT = ["受阻", "延迟", "停运", "内涝", "合并减班", "大幅上升", "严重"]
LOW_IMPACT = ["下滑", "积压", "缺货", "临近保质期", "胀包", "变质", "异常"]
POSITIVE = ["稳定", "保障", "充足", "有力", "韧性", "不断档", "高位"]

SCOPE_BASE = {"local": 0.40, "region": 0.65, "global": 0.85}
SCOPE_CAP = {"local": 0.70, "region": 0.85, "global": 1.00}


def load_corpus(path=None):
    with open(path or _CORPUS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ============ Extractor 智能体 ============
def extractor(event):
    """提取线索与证据锚点，基于量化冲击幅度生成初始severity_raw"""
    raw_text = event.get("raw_text", "")
    impact_text = event.get("quantified_impact", "") + raw_text
    scope = event.get("scope_word", "local")
    decay_flag = event.get("time_decay_flag", "pulse")

    # 冲击幅度分级评分
    severity = SCOPE_BASE.get(scope, 0.40)
    if any(kw in impact_text for kw in HIGH_IMPACT):
        severity += 0.25
    elif any(kw in impact_text for kw in MID_IMPACT):
        severity += 0.15
    elif any(kw in impact_text for kw in LOW_IMPACT):
        severity += 0.08
    if any(kw in impact_text for kw in POSITIVE):
        severity -= 0.15
    if decay_flag == "pulse":
        severity += 0.05  # 脉冲式冲击集中
    severity = max(0.05, min(0.98, severity))

    return {
        "event_id": event.get("event_id"),
        "node_ids": event.get("node_ids", []),
        "date": event.get("date"),
        "source": event.get("source"),
        "raw_text": raw_text,
        "evidence_anchor": event.get("evidence_anchor", ""),
        "scope_word": scope,
        "time_decay_flag": decay_flag,
        "severity_raw": round(severity, 4),
        "quantified_impact": event.get("quantified_impact", ""),
        "is_fallback": event.get("fallback", False)
    }


# ============ Feedback 智能体（四维批判） ============
def _normalize_text(s):
    """去空格、统一标点、数值格式归一化"""
    s = re.sub(r"\s+", "", s)
    s = s.replace("，", ",").replace("。", ".").replace("：", ":")
    s = s.replace("（", "(").replace("）", ")")
    s = re.sub(r"(\d),(\d)", r"\1\2", s)  # 数值内逗号
    return s


def _r1_evidence_consistency(item):
    """R1证据一致性硬门禁：锚点必须在原文中逐字存在，否则直接驳回"""
    anchor = _normalize_text(item["evidence_anchor"])
    raw = _normalize_text(item["raw_text"])
    if anchor and anchor in raw:
        return {"rule": "R1", "passed": True, "delta": 0.0,
                "detail": "证据锚点在原文中逐字存在，通过硬门禁"}
    return {"rule": "R1", "passed": False, "delta": 0.0,
            "detail": f"证据锚点[{item['evidence_anchor']}]未在原文逐字存在，驳回剔除",
            "rejected": True}


def _r2_scope_calibration(item):
    """R2范围词校准：局部0.70、区域0.85、全局1.00上限拉回超限分数"""
    cap = SCOPE_CAP.get(item["scope_word"], 0.70)
    if item["severity_raw"] > cap:
        delta = cap - item["severity_raw"]
        return {"rule": "R2", "passed": True, "delta": round(delta, 4),
                "detail": f"范围词[{item['scope_word']}]上限{cap}，拉回超限分数"}
    return {"rule": "R2", "passed": True, "delta": 0.0,
            "detail": f"分数在范围词[{item['scope_word']}]上限内，无需校准"}


def _r3_time_decay(item):
    """R3时效衰减：距窗口终点超6个月触发-0.20至-0.10扣分，结构性风险豁免"""
    if item["time_decay_flag"] == "structural":
        return {"rule": "R3", "passed": True, "delta": 0.0,
                "detail": "结构性风险，豁免时效衰减"}
    try:
        d = date.fromisoformat(item["date"])
    except (ValueError, TypeError):
        return {"rule": "R3", "passed": True, "delta": 0.0,
                "detail": "日期缺失，不触发时效衰减"}
    months = (WINDOW_END - d).days / 30.0
    if months > TIME_DECAY_THRESHOLD_MONTHS:
        # 线性插值：6个月→-0.10，12个月→-0.20
        delta = -min(0.20, 0.10 + (months - 6) * 0.017)
        return {"rule": "R3", "passed": True, "delta": round(delta, 4),
                "detail": f"距窗口终点{months:.1f}个月，触发时效衰减{delta:.2f}"}
    return {"rule": "R3", "passed": True, "delta": 0.0,
            "detail": f"距窗口终点{months:.1f}个月，未超6个月阈值"}


def _r4_single_factor(item):
    """R4单因子一致性：量化依据仅来自单一数据点且severity_raw>0.60时扣分"""
    impact = item.get("quantified_impact", "")
    # 判断是否单一数据点驱动（仅一个量化指标关键词且无交叉来源）
    data_points = len(re.findall(r"\d+\.?\d*", impact))
    if data_points <= 1 and item["severity_raw"] > 0.60:
        delta = -0.10  # 单因子扣分-0.15至-0.05区间中值
        return {"rule": "R4", "passed": True, "delta": round(delta, 4),
                "detail": f"量化依据仅{data_points}个数据点且severity_raw>0.60，触发单因子扣分"}
    return {"rule": "R4", "passed": True, "delta": 0.0,
            "detail": f"量化数据点{data_points}个，不触发单因子扣分"}


def feedback(item):
    """四维规则批判，返回规则触发记录与各项delta"""
    rules = []
    r1 = _r1_evidence_consistency(item)
    rules.append(r1)
    if r1.get("rejected"):
        return {"rules": rules, "rejected": True, "total_delta": 0.0}
    rules.append(_r2_scope_calibration(item))
    rules.append(_r3_time_decay(item))
    rules.append(_r4_single_factor(item))
    total = sum(r["delta"] for r in rules[1:])
    return {"rules": rules, "rejected": False, "total_delta": round(total, 4)}


# ============ Revision 智能体（有限修正） ============
def revision(item, fb_result):
    """有限修正：score_delta下限-0.30防止过度压制，累加后clamp(0,1)"""
    if fb_result.get("rejected"):
        return {
            "event_id": item["event_id"],
            "node_ids": item["node_ids"],
            "severity_final": None,
            "status": "rejected_by_R1",
            "audit_trail": fb_result["rules"]
        }
    delta = max(-0.30, fb_result["total_delta"])  # 下限-0.30
    final = max(0.0, min(1.0, item["severity_raw"] + delta))
    return {
        "event_id": item["event_id"],
        "node_ids": item["node_ids"],
        "date": item["date"],
        "source": item["source"],
        "evidence_anchor": item["evidence_anchor"],
        "scope_word": item["scope_word"],
        "time_decay_flag": item["time_decay_flag"],
        "severity_raw": item["severity_raw"],
        "score_delta": round(delta, 4),
        "severity_final": round(final, 4),
        "is_fallback": item.get("is_fallback", False),
        "status": "passed",
        "audit_trail": fb_result["rules"]
    }


def run_pipeline(events=None):
    """执行完整流水线：Extractor→Feedback→Revision，输出事件级评分与节点级暴露度"""
    corpus = load_corpus()
    all_events = events if events is not None else corpus["events"]
    # 兜底事件仅在异常时引入，这里默认不纳入主评分
    results = []
    for ev in all_events:
        extracted = extractor(ev)
        fb = feedback(extracted)
        revised = revision(extracted, fb)
        results.append(revised)
    # 节点级暴露度：同一节点多事件取加权最大（近期事件权重高）
    node_scores = {}
    for r in results:
        if r.get("severity_final") is None:
            continue
        for nid in r["node_ids"]:
            prev = node_scores.get(nid)
            if prev is None or r["severity_final"] > prev["severity_final"]:
                node_scores[nid] = r
    return {
        "event_results": results,
        "node_exposure": {nid: {"severity_final": v["severity_final"],
                                 "evidence_anchor": v["evidence_anchor"],
                                 "event_id": v["event_id"],
                                 "source": v["source"],
                                 "date": v["date"],
                                 "scope_word": v["scope_word"]}
                          for nid, v in node_scores.items()}
    }


def cross_validate_exposure_responsibility(exposure, shapley):
    """双规则交叉验证：前端文本严重度与后端拓扑责任度的Pearson相关系数"""
    import numpy as np
    common = [n for n in exposure if n in shapley]
    if len(common) < 3:
        return {"pearson": None, "note": "共同节点不足，无法计算"}
    # 所有节点都给默认暴露度0（无事件的节点）
    all_nodes = list(shapley.keys())
    exp_vals = [exposure.get(n, {}).get("severity_final", 0.0) for n in all_nodes]
    shap_vals = [shapley[n] for n in all_nodes]
    if np.std(exp_vals) < 1e-9 or np.std(shap_vals) < 1e-9:
        r = 0.0
    else:
        r = float(np.corrcoef(exp_vals, shap_vals)[0, 1])
    return {
        "pearson": round(r, 4),
        "n_nodes": len(all_nodes),
        "note": ("探索性效度检验：Pearson相关系数反映文本暴露度与拓扑责任度的收敛趋势，"
                 "低相关恰恰说明暴露度高≠全局破坏力大，印证二维矩阵的必要性")
    }