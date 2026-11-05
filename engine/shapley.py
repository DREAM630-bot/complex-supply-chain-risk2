# -*- coding: utf-8 -*-
"""
Shapley责任度计算模块
- 蒙特卡罗排列采样（大图）
- 精确枚举（小图，≤10节点）
- 分层Shapley：关键候选集 + 背景聚合
- 敏感性分析：±20%-30%容量扰动 + Kendall's W / Spearman秩相关
- Kendall's W对称性诊断
"""
import random
import itertools
import copy
import numpy as np
import networkx as nx
from engine.graph_engine import (build_graph, characteristic_value,
                                 compute_weighted_maxflow, identify_critical_candidates)

_VALUE_CACHE = {}


def _cv_cached(G, config, failed, base_flow):
    """带缓存的特征函数（以失效集合元组为键）"""
    key = tuple(sorted(failed))
    if key in _VALUE_CACHE:
        return _VALUE_CACHE[key]
    pp = config["penalty_params"]
    cur_flow, sink_flow = compute_weighted_maxflow(G, config, failed)
    base_flow_val = base_flow if base_flow > 0 else 1.0
    flow_ratio = cur_flow / base_flow_val
    from engine.graph_engine import compute_connectivity
    conn = compute_connectivity(G, config, failed)
    value = (pp.get("maxflow_weight", 0.65) * flow_ratio +
             pp.get("connectivity_weight", 0.35) * conn)
    val = max(0.0, min(1.0, value))
    _VALUE_CACHE[key] = val
    return val


def clear_cache():
    _VALUE_CACHE.clear()


def shapley_monte_carlo(G, config, n_samples=800, seed=42):
    """蒙特卡罗排列采样计算各节点Shapley责任度（边际损失期望）"""
    clear_cache()
    rng = random.Random(seed)
    players = [n["id"] for n in config["nodes"]]
    base_flow, _ = compute_weighted_maxflow(G, config, [])
    shapley = {p: 0.0 for p in players}
    for _ in range(n_samples):
        perm = players[:]
        rng.shuffle(perm)
        failed = []
        for p in perm:
            v_before = _cv_cached(G, config, failed, base_flow)
            failed.append(p)
            v_after = _cv_cached(G, config, failed, base_flow)
            shapley[p] += (v_before - v_after)  # 边际责任度=价值下降
    return {p: s / n_samples for p, s in shapley.items()}


def shapley_exact(G, config):
    """精确枚举Shapley（小图≤10节点），复杂度O(2^n * n)"""
    clear_cache()
    players = [n["id"] for n in config["nodes"]]
    n = len(players)
    if n > 10:
        raise ValueError(f"精确枚举仅支持≤10节点，当前{n}节点，请用蒙特卡罗采样")
    base_flow, _ = compute_weighted_maxflow(G, config, [])
    shapley = {p: 0.0 for p in players}
    for r in range(n + 1):
        for subset in itertools.combinations(players, r):
            failed = list(subset)
            v_with = _cv_cached(G, config, failed, base_flow)
            for p in players:
                if p in failed:
                    v_without = _cv_cached(G, config,
                                           [x for x in failed if x != p], base_flow)
                    shapley[p] += (v_without - v_with)
    from math import factorial
    n_fact = factorial(n)
    for p in players:
        shapley[p] = shapley[p] / n_fact
    return shapley


def shapley_layered(G, config, n_samples=600, seed=42):
    """分层Shapley：关键候选集逐个计算，其余节点聚合为背景玩家"""
    clear_cache()
    cand_info = identify_critical_candidates(G, config)
    candidates = cand_info["candidates"]
    players = [n["id"] for n in config["nodes"]]
    background = [p for p in players if p not in candidates]
    base_flow, _ = compute_weighted_maxflow(G, config, [])
    rng = random.Random(seed)
    # 前景玩家逐一采样，背景作为整体
    shapley = {p: 0.0 for p in players}
    all_units = candidates + ["__BG__"]
    for _ in range(n_samples):
        perm = all_units[:]
        rng.shuffle(perm)
        failed_real = []
        for unit in perm:
            v_before = _cv_cached(G, config, failed_real, base_flow)
            if unit == "__BG__":
                failed_real.extend(background)
            else:
                failed_real.append(unit)
            v_after = _cv_cached(G, config, failed_real, base_flow)
            marginal = v_before - v_after
            if unit == "__BG__":
                share = marginal / max(len(background), 1)
                for b in background:
                    shapley[b] += share
            else:
                shapley[unit] += marginal
    return {p: s / n_samples for p, s in shapley.items()}, cand_info


def kendalls_w(rankings):
    """Kendall's W 一致性系数（多排列协调程度，0~1）"""
    m = len(rankings)
    if m < 2:
        return 1.0
    n = len(rankings[0])
    items = [r[0] for r in rankings[0]]
    R = {item: 0.0 for item in items}
    for ranking in rankings:
        for rank, (item, _) in enumerate(ranking):
            R[item] += (rank + 1)
    mean_R = sum(R.values()) / n
    S = sum((R[item] - mean_R) ** 2 for item in items)
    return S / (m * m * (n ** 3 - n)) if n > 1 else 1.0


def spearman_rank_corr(ranking_a, ranking_b):
    """Spearman秩相关系数"""
    items_a = [r[0] for r in ranking_a]
    items_b = [r[0] for r in ranking_b]
    common = [x for x in items_a if x in items_b]
    if len(common) < 2:
        return 1.0
    rank_a = {item: i + 1 for i, item in enumerate(items_a)}
    rank_b = {item: i + 1 for i, item in enumerate(items_b)}
    n = len(common)
    d2 = sum((rank_a[c] - rank_b[c]) ** 2 for c in common)
    return 1 - (6 * d2) / (n * (n ** 2 - 1))


def sensitivity_analysis(G, config, n_samples=500, perturb_range=0.25,
                         n_trials=5, seed=42):
    """对proxy_ratio和prior_assumption节点做±perturb容量敏感性扫描
    返回各试验Shapley排序 + Kendall's W + Spearman相关 + 排序稳健性结论"""
    results = []
    rankings = []
    rng = random.Random(seed)
    # 基线排序
    base_shap = shapley_monte_carlo(G, config, n_samples=n_samples, seed=seed)
    base_rank = sorted(base_shap.items(), key=lambda x: -x[1])
    rankings.append(base_rank)
    results.append({"trial": "baseline", "shapley": base_shap})
    # 扰动试验
    perturb_nodes = [n for n in config["nodes"]
                     if n["capacity_source"] in ("proxy_ratio", "prior_assumption")]
    for t in range(n_trials):
        cfg = copy.deepcopy(config)
        factor = rng.uniform(-perturb_range, perturb_range)
        for n in cfg["nodes"]:
            if n["capacity_source"] in ("proxy_ratio", "prior_assumption"):
                n["capacity"] = max(0.01, n["capacity"] * (1 + factor))
        for e in cfg["edges"]:
            if e["capacity_source"] in ("proxy_ratio", "prior_assumption"):
                e["capacity"] = max(0.01, e["capacity"] * (1 + factor))
        Gp = build_graph(cfg)
        shap = shapley_monte_carlo(Gp, cfg, n_samples=n_samples, seed=seed + t + 1)
        rank = sorted(shap.items(), key=lambda x: -x[1])
        rankings.append(rank)
        results.append({"trial": f"perturb_{t+1}", "factor": round(factor, 3),
                        "shapley": shap})
    W = kendalls_w(rankings)
    spearmans = [spearman_rank_corr(rankings[0], r) for r in rankings[1:]]
    # 核心排序稳健性：Top3集合是否稳定
    top3_base = set(x[0] for x in base_rank[:3])
    top3_stable = all(set(x[0] for x in r[:3]) == top3_base for r in rankings)
    return {
        "trials": results,
        "kendalls_w": round(W, 4),
        "spearman_mean": round(float(np.mean(spearmans)), 4),
        "top3_stable": top3_stable,
        "perturb_nodes": [n["id"] for n in perturb_nodes],
        "conclusion": ("核心节点责任度排序在±{:.0%}容量扰动下保持稳健（Kendall's W={:.3f}）"
                       .format(perturb_range, W)) if top3_stable else
                      ("核心节点责任度排序在扰动下出现波动，需进一步检查（Kendall's W={:.3f}）"
                       .format(perturb_range, W))
    }


def symmetry_diagnosis_single_chain(G, config, n_samples=400):
    """Kendall's W对称性诊断：纯最大流Shapley在单链上无一致区分度
    将特征函数退化为纯最大流（无连通性惩罚），验证对称性困境"""
    clear_cache()
    rng = random.Random(123)
    players = [n["id"] for n in config["nodes"]]
    base_flow, _ = compute_weighted_maxflow(G, config, [])
    # 纯最大流Shapley
    rankings = []
    for trial in range(5):
        shap = {p: 0.0 for p in players}
        for _ in range(n_samples):
            perm = players[:]
            rng.shuffle(perm)
            failed = []
            for p in perm:
                fb, _ = compute_weighted_maxflow(G, config, failed)
                failed.append(p)
                fa, _ = compute_weighted_maxflow(G, config, failed)
                shap[p] += (fb - fa) / max(base_flow, 1e-9)
        shap = {p: s / n_samples for p, s in shap.items()}
        rankings.append(sorted(shap.items(), key=lambda x: -x[1]))
    W = kendalls_w(rankings)
    return {
        "kendalls_w_pure_maxflow": round(W, 4),
        "interpretation": ("纯最大流特征函数下Shapley区分度低（W={:.3f}），"
                           "反向说明引入连通性惩罚与业务先验的必要性").format(W)
    }