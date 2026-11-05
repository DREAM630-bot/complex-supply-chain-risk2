# -*- coding: utf-8 -*-
"""
图博弈归因引擎：配置驱动的通用有向图
特征函数 = 多源多汇加权最大流 + 有向可达性/割集脆弱性惩罚
木桶效应数学化为最小割/瓶颈约束分析
"""
import json
import os
import networkx as nx
import math

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "topology.json")

SUPER_SOURCE = "__SUPER_S__"
SUPER_SINK = "__SUPER_T__"
FLOW_SCALE = 1000  # 容量放大系数，避免浮点最大流精度问题


def load_config(path=None):
    with open(path or _CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_graph(config):
    """从JSON配置构建NetworkX有向图（含节点属性与边容量）"""
    G = nx.DiGraph()
    for n in config["nodes"]:
        G.add_node(n["id"], **{k: v for k, v in n.items() if k != "id"})
    for e in config["edges"]:
        G.add_edge(e["from"], e["to"],
                   capacity=round(e["capacity"] * FLOW_SCALE),
                   mode=e.get("mode", "road"),
                   capacity_source=e.get("capacity_source", "proxy_ratio"),
                   bidirectional=e.get("bidirectional", False))
    return G


def _augment_graph(G, config, failed_nodes):
    """构建含超级源汇的流网络，剔除失效节点"""
    H = G.copy()
    for fn in failed_nodes:
        if fn in H:
            H.remove_node(fn)
    sources = [s for s in config["sources"] if s in H]
    sinks = [k for k in config["sinks"] if k in H]
    dw = config["demand_weights"]
    H.add_node(SUPER_SOURCE)
    H.add_node(SUPER_SINK)
    for s in sources:
        cap = H.nodes[s].get("capacity", 0.5) * FLOW_SCALE
        H.add_edge(SUPER_SOURCE, s, capacity=int(round(cap)))
    for k in sinks:
        cap = dw.get(k, 0.3) * FLOW_SCALE
        H.add_edge(k, SUPER_SINK, capacity=int(round(cap)))
    return H, sources, sinks


def compute_weighted_maxflow(G, config, failed_nodes=None):
    """多源多汇加权最大流：超级源S连供应源、超级汇T连需求点（带权重）"""
    failed_nodes = failed_nodes or []
    H, sources, sinks = _augment_graph(G, config, failed_nodes)
    if not sources or not sinks:
        return 0.0, {}
    try:
        flow_value, flow_dict = nx.maximum_flow(H, SUPER_SOURCE, SUPER_SINK)
    except nx.NetworkXError:
        return 0.0, {}
    # 各需求汇实际接收流量
    sink_flow = {}
    for k in sinks:
        incoming = sum(fd.get(k, 0) for _, fd in flow_dict.items() if isinstance(fd, dict))
        sink_flow[k] = incoming / FLOW_SCALE
    return flow_value / FLOW_SCALE, sink_flow


def compute_connectivity(G, config, failed_nodes=None):
    """连通性指标：需求点可达性 + 源冗余度 + 最大SCC占比 - 割点/桥惩罚 - 最小割下降"""
    failed_nodes = failed_nodes or []
    H, sources, sinks = _augment_graph(G, config, failed_nodes)
    H2 = H.copy()
    H2.remove_nodes_from([SUPER_SOURCE, SUPER_SINK])
    dw = config["demand_weights"]
    pp = config["penalty_params"]
    total_w = sum(dw.get(k, 0) for k in config["sinks"]) or 1.0

    # 1. 需求点可达性：至少一个源能到达该需求汇
    reachable_w = 0.0
    source_redundancy_w = 0.0
    for k in sinks:
        w = dw.get(k, 0.3)
        reaching = 0
        for s in sources:
            if nx.has_path(H2, s, k):
                reaching += 1
        if reaching > 0:
            reachable_w += w
        source_redundancy_w += w * (reaching / max(len(config["sources"]), 1))
    reachability_ratio = reachable_w / total_w
    source_redundancy_ratio = source_redundancy_w / total_w

    # 2. 最大强连通分量占比
    if len(H2) > 0:
        sccs = list(nx.strongly_connected_components(H2))
        max_scc_ratio = max(len(c) for c in sccs) / len(H2)
    else:
        max_scc_ratio = 0.0

    # 3. 割点/桥惩罚（基于无向视图）
    undirected = H2.to_undirected()
    cut_nodes = set(nx.articulation_points(undirected)) if len(undirected) > 0 else set()
    bridges = list(nx.bridges(undirected)) if len(undirected) > 0 else []
    # 基线割点/桥
    base_undirected = G.to_undirected()
    base_cut_nodes = set(nx.articulation_points(base_undirected))
    base_bridges = list(nx.bridges(base_undirected))
    new_cuts = len(cut_nodes - base_cut_nodes)
    new_bridges = max(0, len(bridges) - len(base_bridges))
    cut_penalty = min(1.0, (new_cuts * pp.get("cut_point_penalty", 0.10) +
                            new_bridges * pp.get("bridge_penalty", 0.08)))

    # 4. 最小割容量下降率（S到T）
    mincut_ratio = 1.0
    try:
        if SUPER_SOURCE in H and SUPER_SINK in H:
            base_H, _, _ = _augment_graph(G, config, [])
            base_mincut = nx.minimum_cut_value(base_H, SUPER_SOURCE, SUPER_SINK)
            cur_mincut = nx.minimum_cut_value(H, SUPER_SOURCE, SUPER_SINK)
            if base_mincut > 0:
                mincut_ratio = cur_mincut / base_mincut
    except (nx.NetworkXError, nx.NetworkXUnbounded):
        pass
    mincut_drop = 1.0 - mincut_ratio

    score = (reachability_ratio * pp.get("reachability_weight", 0.35) +
             source_redundancy_ratio * pp.get("source_redundancy_weight", 0.15) +
             max_scc_ratio * pp.get("scc_ratio_weight", 0.15) +
             mincut_ratio * pp.get("mincut_drop_weight", 0.15) -
             cut_penalty * pp.get("cut_penalty_weight", 0.20))
    return max(0.0, min(1.0, score))


def characteristic_value(G, config, failed_nodes=None):
    """特征函数：多源多汇加权最大流 + 连通性惩罚的融合（网络价值，越高越好）"""
    failed_nodes = failed_nodes or []
    pp = config["penalty_params"]
    # 基线值
    base_flow, _ = compute_weighted_maxflow(G, config, [])
    base_flow_val = base_flow if base_flow > 0 else 1.0
    # 当前值
    cur_flow, sink_flow = compute_weighted_maxflow(G, config, failed_nodes)
    flow_ratio = cur_flow / base_flow_val
    conn_score = compute_connectivity(G, config, failed_nodes)
    value = (pp.get("maxflow_weight", 0.65) * flow_ratio +
             pp.get("connectivity_weight", 0.35) * conn_score)
    return {
        "value": max(0.0, min(1.0, value)),
        "maxflow": cur_flow,
        "flow_ratio": flow_ratio,
        "sink_flow": sink_flow,
        "connectivity": conn_score
    }


def analyze_cuts(G, config):
    """木桶效应数学化：最小割/瓶颈约束分析，识别全局瓶颈割集"""
    H, sources, sinks = _augment_graph(G, config, [])
    result = {"mincut_value": 0, "mincut_partition": [], "cut_nodes": [],
              "bridges": [], "bottleneck_edges": []}
    try:
        if SUPER_SOURCE in H and SUPER_SINK in H:
            mc_val = nx.minimum_cut_value(H, SUPER_SOURCE, SUPER_SINK)
            result["mincut_value"] = mc_val / FLOW_SCALE
            _, partition = nx.minimum_cut(H, SUPER_SOURCE, SUPER_SINK)
            reachable, non_reachable = partition
            for u in reachable:
                for v in non_reachable:
                    if H.has_edge(u, v):
                        result["bottleneck_edges"].append({
                            "from": u, "to": v,
                            "capacity": H[u][v]["capacity"] / FLOW_SCALE,
                            "mode": H[u][v].get("mode", "road")
                        })
    except (nx.NetworkXError, nx.NetworkXUnbounded):
        pass
    undirected = G.to_undirected()
    result["cut_nodes"] = sorted(nx.articulation_points(undirected))
    result["bridges"] = [sorted(b) for b in nx.bridges(undirected)]
    return result


def identify_critical_candidates(G, config):
    """关键候选集：割点、桥端点、高吞吐节点、唯一转换点（用于背景聚合分层Shapley）"""
    undirected = G.to_undirected()
    cut_nodes = set(nx.articulation_points(undirected))
    bridge_ends = set()
    for b in nx.bridges(undirected):
        bridge_ends.update(b)
    # 高吞吐：入度+出度加权容量Top
    throughput = {}
    for n in G.nodes():
        in_cap = sum(G[u][n]["capacity"] for u in G.predecessors(n))
        out_cap = sum(G[n][v]["capacity"] for v in G.successors(n))
        throughput[n] = (in_cap + out_cap) / FLOW_SCALE
    sorted_th = sorted(throughput.items(), key=lambda x: -x[1])
    high_throughput = {n for n, _ in sorted_th[:max(3, len(sorted_th) // 3)]}
    # 唯一转换点：key_convert=True的工厂节点
    key_converters = {n for n, d in G.nodes(data=True) if d.get("key_convert")}
    candidates = cut_nodes | bridge_ends | high_throughput | key_converters
    return {
        "candidates": sorted(candidates),
        "cut_nodes": sorted(cut_nodes),
        "bridge_ends": sorted(bridge_ends),
        "high_throughput": sorted(high_throughput),
        "key_converters": sorted(key_converters),
        "throughput": throughput
    }


def node_topological_profile(G, config, node_id):
    """生成单节点的拓扑上下文指标（供解释智能体JSON Context使用）"""
    if node_id not in G:
        return {}
    in_deg = G.in_degree(node_id)
    out_deg = G.out_degree(node_id)
    undirected = G.to_undirected()
    cut_nodes = set(nx.articulation_points(undirected))
    is_cut = node_id in cut_nodes
    # 可达性影响：该节点失效后，需求汇可达性下降
    base_conn = compute_connectivity(G, config, [])
    fail_conn = compute_connectivity(G, config, [node_id])
    reach_impact = base_conn - fail_conn
    # 最小割影响
    H_base, _, _ = _augment_graph(G, config, [])
    H_fail, _, _ = _augment_graph(G, config, [node_id])
    mincut_impact = 0.0
    try:
        base_mc = nx.minimum_cut_value(H_base, SUPER_SOURCE, SUPER_SINK)
        fail_mc = nx.minimum_cut_value(H_fail, SUPER_SOURCE, SUPER_SINK)
        if base_mc > 0:
            mincut_impact = 1.0 - fail_mc / base_mc
    except (nx.NetworkXError, nx.NetworkXUnbounded):
        pass
    # 上下游源汇
    upstream_sources = [s for s in config["sources"]
                        if nx.has_path(G, s, node_id)] if node_id not in config["sources"] else []
    downstream_sinks = [k for k in config["sinks"]
                        if nx.has_path(G, node_id, k)] if node_id not in config["sinks"] else []
    base_flow, _ = compute_weighted_maxflow(G, config, [])
    fail_flow, _ = compute_weighted_maxflow(G, config, [node_id])
    flow_impact = (base_flow - fail_flow) / base_flow if base_flow > 0 else 0
    return {
        "node_id": node_id,
        "name": G.nodes[node_id].get("name", node_id),
        "type": G.nodes[node_id].get("type", ""),
        "role": G.nodes[node_id].get("role", ""),
        "in_degree": in_deg,
        "out_degree": out_deg,
        "is_cut_point": is_cut,
        "capacity": G.nodes[node_id].get("capacity", 0),
        "capacity_source": G.nodes[node_id].get("capacity_source", ""),
        "key_convert": G.nodes[node_id].get("key_convert", False),
        "reachability_impact": round(reach_impact, 4),
        "mincut_impact": round(mincut_impact, 4),
        "maxflow_impact": round(flow_impact, 4),
        "upstream_sources": upstream_sources,
        "downstream_sinks": downstream_sinks
    }