# -*- coding: utf-8 -*-
"""
供应链风险全景测度系统 - Flask后端
离线重加工（多智能体文本测度） + 在线轻交互（受约束解释智能体）
"""
import os
import sys
import json
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.graph_engine import (load_config, build_graph, characteristic_value,
                                 analyze_cuts, identify_critical_candidates,
                                 node_topological_profile, compute_weighted_maxflow)
from engine.shapley import (shapley_monte_carlo, shapley_layered,
                            sensitivity_analysis, symmetry_diagnosis_single_chain)
from agents.pipeline import run_pipeline, cross_validate_exposure_responsibility
from agents import explainer

app = Flask(__name__, static_folder="static", static_url_path="/static")

# ============ 全局预计算缓存 ============
STATE = {"ready": False, "data": {}}


def classify_quadrant(exposure, responsibility, exp_threshold, resp_threshold):
    """二维交叉矩阵象限分类"""
    high_exp = exposure >= exp_threshold
    high_resp = responsibility >= resp_threshold
    if high_exp and high_resp:
        return "dominant_core"
    if not high_exp and high_resp:
        return "hidden_critical"
    if high_exp and not high_resp:
        return "local_disturbance"
    return "routine"


def precompute(n_samples=400):
    """启动时预计算全部核心数据并缓存"""
    config = load_config()
    G = build_graph(config)
    node_map = {n["id"]: n for n in config["nodes"]}

    pipeline_result = run_pipeline()
    exposure = pipeline_result["node_exposure"]

    shapley = shapley_monte_carlo(G, config, n_samples=n_samples, seed=42)
    shap_rank = sorted(shapley.items(), key=lambda x: -x[1])

    cuts = analyze_cuts(G, config)
    candidates = identify_critical_candidates(G, config)

    profiles = {}
    ctx_pool = []
    exp_vals = [exposure.get(n["id"], {}).get("severity_final", 0.0) for n in config["nodes"]]
    resp_vals = [shapley[n["id"]] for n in config["nodes"]]
    exp_threshold = float(np.median(exp_vals)) if exp_vals else 0.3
    resp_threshold = float(np.median(resp_vals)) if resp_vals else 0.05
    exp_threshold = max(exp_threshold, 0.30)

    matrix_data = []
    for n in config["nodes"]:
        nid = n["id"]
        prof = node_topological_profile(G, config, nid)
        profiles[nid] = prof
        exp_score = exposure.get(nid, {}).get("severity_final", 0.0)
        resp_score = shapley[nid]
        quadrant = classify_quadrant(exp_score, resp_score, exp_threshold, resp_threshold)
        ctx = explainer.build_json_context(
            nid, prof, resp_score, exposure.get(nid, {}),
            quadrant, config.get("meta", {}))
        ctx_pool.append(ctx)
        matrix_data.append({
            "node_id": nid, "name": n["name"], "type": n["type"], "role": n["role"],
            "exposure": round(exp_score, 4),
            "responsibility": round(resp_score, 4),
            "quadrant": quadrant,
            "is_cut_point": prof["is_cut_point"],
            "capacity_source": n["capacity_source"],
            "region": next((r for r, members in config.get("regions", {}).items()
                            if nid in members), "")
        })

    baseline = characteristic_value(G, config, [])
    cross_val = cross_validate_exposure_responsibility(exposure, shapley)

    topo_vis = {
        "nodes": [{"id": n["id"], "name": n["name"], "type": n["type"],
                   "lat": n["lat"], "lon": n["lon"],
                   "capacity": n["capacity"], "capacity_source": n["capacity_source"],
                   "shapley": round(shapley[n["id"]], 4),
                   "exposure": round(exposure.get(n["id"], {}).get("severity_final", 0.0), 4),
                   "quadrant": next(m["quadrant"] for m in matrix_data if m["node_id"] == n["id"])}
                  for n in config["nodes"]],
        "edges": [{"from": e["from"], "to": e["to"], "capacity": e["capacity"],
                   "mode": e["mode"], "bidirectional": e.get("bidirectional", False),
                   "capacity_source": e["capacity_source"]}
                  for e in config["edges"]],
        "sources": config["sources"], "sinks": config["sinks"],
        "regions": config.get("regions", {})
    }

    STATE["data"] = {
        "config": config, "pipeline": pipeline_result, "shapley": shapley,
        "shapley_rank": shap_rank, "cuts": cuts, "candidates": candidates,
        "profiles": profiles, "ctx_pool": ctx_pool, "matrix": matrix_data,
        "baseline": baseline, "cross_validation": cross_val,
        "topo_vis": topo_vis, "exp_threshold": exp_threshold,
        "resp_threshold": resp_threshold, "n_samples": n_samples
    }
    STATE["ready"] = True


# ================= 关键改动：模块级启动预计算 =================
# gunicorn 导入 app 时 __name__ == "app"，原 __main__ 块不会执行；
# 放到模块级，保证无论 python app.py 还是 gunicorn 都能初始化。
if not STATE["ready"]:
    print("=" * 60)
    print("供应链风险全景测度系统 - 启动预计算中...")
    print("=" * 60)
    precompute(n_samples=400)
    print(f"预计算完成：{len(STATE['data']['matrix'])}个节点矩阵数据就绪")
    print(f"Shapley Top3: {STATE['data']['shapley_rank'][:3]}")
    dk = os.environ.get("DEEPSEEK_API_KEY", "")
    print(f"DeepSeek API: {'已配置' if dk else '未配置'}")


# ============ API 路由（全部保持原样） ============
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/matrix")
def api_matrix():
    if not STATE["ready"]:
        return jsonify({"error": "系统计算中，请稍后刷新"}), 503
    d = STATE["data"]
    return jsonify({
        "matrix": d["matrix"],
        "exp_threshold": d["exp_threshold"],
        "resp_threshold": d["resp_threshold"],
        "shapley_rank": [{"node_id": k, "value": round(v, 4)} for k, v in d["shapley_rank"]],
        "cross_validation": d["cross_validation"],
        "n_samples": d["n_samples"]
    })


@app.route("/api/topology")
def api_topology():
    if not STATE["ready"]:
        return jsonify({"error": "系统计算中"}), 503
    return jsonify(STATE["data"]["topo_vis"])


@app.route("/api/cuts")
def api_cuts():
    if not STATE["ready"]:
        return jsonify({"error": "系统计算中"}), 503
    d = STATE["data"]
    node_map = {n["id"]: n["name"] for n in d["config"]["nodes"]}
    cuts = d["cuts"]
    cuts["cut_node_names"] = [node_map.get(c, c) for c in cuts["cut_nodes"]]
    return jsonify({"cuts": cuts, "candidates": d["candidates"]})


@app.route("/api/pipeline")
def api_pipeline():
    if not STATE["ready"]:
        return jsonify({"error": "系统计算中"}), 503
    return jsonify(STATE["data"]["pipeline"])


@app.route("/api/node/<node_id>")
def api_node(node_id):
    if not STATE["ready"]:
        return jsonify({"error": "系统计算中"}), 503
    d = STATE["data"]
    if node_id not in d["profiles"]:
        return jsonify({"error": f"节点{node_id}不存在"}), 404
    ctx = next((c for c in d["ctx_pool"] if c["node"]["id"] == node_id), None)
    return jsonify({"profile": d["profiles"][node_id], "context": ctx,
                    "shapley": d["shapley"].get(node_id, 0)})


@app.route("/api/baseline")
def api_baseline():
    if not STATE["ready"]:
        return jsonify({"error": "系统计算中"}), 503
    return jsonify(STATE["data"]["baseline"])


@app.route("/api/sensitivity")
def api_sensitivity():
    """敏感性分析（按需计算，较慢）"""
    if not STATE["ready"]:
        return jsonify({"error": "系统计算中"}), 503
    d = STATE["data"]
    config = d["config"]
    G = build_graph(config)
    result = sensitivity_analysis(G, config, n_samples=200, perturb_range=0.25,
                                  n_trials=3, seed=42)
    sym = symmetry_diagnosis_single_chain(G, config, n_samples=150)
    return jsonify({"sensitivity": result, "symmetry": sym})


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """智能体聊天入口：越界拦截 → 强约束JSON Context → 受约束解释"""
    if not STATE["ready"]:
        return jsonify({"reply": "系统计算中，请稍后再试", "in_scope": True})
    body = request.get_json(force=True)
    question = body.get("question", "").strip()
    if not question:
        return jsonify({"reply": "请输入您的问题。", "in_scope": True})
    boundary = explainer.check_boundary(question)
    if not boundary["in_scope"]:
        reply = explainer._template_out_of_scope(question)
        return jsonify({"reply": reply, "in_scope": False})
    d = STATE["data"]
    all_matches = explainer.NODE_ID_PATTERN.findall(question)
    matched_ctxs = []
    for m in all_matches:
        nid = f"N{int(m)}"
        ctx = next((c for c in d["ctx_pool"] if c["node"]["id"] == nid), None)
        if ctx:
            matched_ctxs.append(ctx)

    api_reply = None
    if matched_ctxs:
        user_prompt = (
            f"问题：{question}\n\n"
            f"以下是与问题相关的供应链节点结构化数据（JSON Context），"
            f"请基于这些事实回答用户的问题，不要编造未提供的信息：\n\n"
            f"{json.dumps(matched_ctxs, ensure_ascii=False, indent=2)}"
        )
        api_reply = explainer.call_deepseek_api(explainer.SYSTEM_PROMPT, user_prompt)

    reply = api_reply or explainer._template_general(
        d["ctx_pool"], question, d["matrix"], d["shapley_rank"])
    return jsonify({"reply": reply, "in_scope": True,
                    "engine": "deepseek_api" if api_reply else "rule_template"})


@app.route("/api/status")
def api_status():
    return jsonify({"ready": STATE["ready"],
                    "deepseek_enabled": bool(os.environ.get("DEEPSEEK_API_KEY", ""))})


# ============ 本地开发入口（生产用 gunicorn，不执行这里） ============
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
