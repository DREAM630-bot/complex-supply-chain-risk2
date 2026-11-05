# -*- coding: utf-8 -*-
"""
受约束解释智能体（业务翻译官）
- 将拓扑指标、Shapley值、暴露度、证据锚点打包为强约束JSON Context
- 大模型只做基于事实的业务解释，不参与数值计算，不自行推断未提供的节点关系
- 内置越界拦截与边界声明，对超出局部灾害级联失效范围的问题明确拒答
- 支持DeepSeek API接入（环境变量DEEPSEEK_API_KEY），默认使用规则模板引擎
"""
import os
import re
import json

# ============ 越界拦截 ============
BOUNDARY_DECLARATION = (
    "【边界声明】本智能体仅回答与供应链局部灾害级联失效风险归因相关的问题，"
    "包括节点暴露度、拓扑责任度、割集脆弱性、防御措施建议。"
    "不回答宏观经济预测、股价走势、企业内部机密、非供应链业务等问题。"
    "所有解释基于已提供的结构化上下文，不自行推断未提供的节点关系，不参与数值计算。"
)

OUT_OF_SCOPE_PATTERNS = [
    r"股票|股价|股市|基金|理财|投资建议|收益率",
    r"宏观经济|GDP|通胀|汇率|加息",
    r"机密|内部数据|财务报表|利润|营收|成本结构",
    r"天气预测|预报|明天.*天气|未来.*气温",
    r"竞争对手|蒙牛|光明|三元.*对比",
    r"写代码|写诗|讲笑话|聊天|你是谁|什么模型",
]

NODE_ID_PATTERN = re.compile(r"N(\d{1,2})")

# ============ JSON Context 构建 ============
def build_json_context(node_id, topo_profile, shapley_val, exposure_info,
                       matrix_quadrant, config_meta):
    """构建强约束JSON Context（喂给大模型的唯一事实来源）"""
    ctx = {
        "role": "业务翻译官",
        "constraint": "只做基于事实的业务解释，不参与数值计算，不推断未提供的节点关系",
        "case": config_meta.get("case", ""),
        "window": config_meta.get("window", ""),
        "node": {
            "id": node_id,
            "name": topo_profile.get("name", ""),
            "type": topo_profile.get("type", ""),
            "role": topo_profile.get("role", ""),
            "in_degree": topo_profile.get("in_degree", 0),
            "out_degree": topo_profile.get("out_degree", 0),
            "is_cut_point": topo_profile.get("is_cut_point", False),
            "capacity_source": topo_profile.get("capacity_source", ""),
            "key_convert": topo_profile.get("key_convert", False),
            "reachability_impact": topo_profile.get("reachability_impact", 0),
            "mincut_impact": topo_profile.get("mincut_impact", 0),
            "maxflow_impact": topo_profile.get("maxflow_impact", 0),
            "upstream_sources": topo_profile.get("upstream_sources", []),
            "downstream_sinks": topo_profile.get("downstream_sinks", [])
        },
        "metrics": {
            "shapley_responsibility": round(shapley_val, 4),
            "exposure_score": exposure_info.get("severity_final", 0.0),
            "matrix_quadrant": matrix_quadrant
        },
        "evidence": {
            "anchor": exposure_info.get("evidence_anchor", "无关联事件"),
            "source": exposure_info.get("source", ""),
            "date": exposure_info.get("date", ""),
            "scope_word": exposure_info.get("scope_word", "")
        },
        "data_precision_note": config_meta.get("disclaimer", "")
    }
    return ctx


def check_boundary(question):
    """越界拦截：检测问题是否超出局部灾害级联失效范围"""
    for pattern in OUT_OF_SCOPE_PATTERNS:
        if re.search(pattern, question):
            return {"in_scope": False}
    # 语义越界：未命中任何意图路由且明显与供应链无关
    if _is_semantically_out_of_scope(question):
        return {"in_scope": False}
    return {"in_scope": True}


def _is_semantically_out_of_scope(question):
    """语义越界检测：未命中正则但明显与业务无关"""
    q = question.lower()
    # 生活/闲聊/非业务关键词
    life_keywords = ["奶茶", "咖啡", "吃饭", "旅游", "旅行", "电影", "游戏",
                     "购物", "逛街", "唱歌", "跳舞", "健身", "跑步", "睡觉",
                     "无聊", "好玩", "推荐", "好吃", "好玩"]
    for kw in life_keywords:
        if kw in q:
            return True
    # 如果问题极短（<=4字）且不含任何供应链相关词
    supply_chain_keywords = ["风险", "节点", "暴露", "责任", "供应链", "割集",
                             "瓶颈", "脆弱", "防御", "措施", "建议", "拓扑",
                             "shapley", "矩阵", "排序", "预算", "投入", "优先",
                             "工厂", "分仓", "牧场", "配送", "终端", "奶源",
                             "n\\d+", "节点"]
    if len(question) <= 4:
        has_sc = any(re.search(kw, q) for kw in supply_chain_keywords)
        if not has_sc:
            return True
    return False


# ============ 规则模板解释引擎（默认，可验证可复现） ============
QUADRANT_DESC = {
    "dominant_core": "显性核心（高暴露—高责任）",
    "hidden_critical": "隐性关键（低暴露—高责任，潜伏炸弹）",
    "local_disturbance": "局部扰动（高暴露—低责任，虚假警报）",
    "routine": "常规节点（低暴露—低责任）"
}

DEFENSE_MEASURES = {
    "factory": "增加物理冗余与备产线、建立缓冲库存、定期开展压力测试与应急演练",
    "warehouse": "建立双仓互备机制、提升库存周转缓冲、预置应急调拨通道",
    "supply": "拓展备选奶源基地、签订应急供应协议、建立奶源多元化组合",
    "delivery": "优化末端配送路由冗余、预置应急运力、建立末端互援机制",
    "terminal": "建立多渠道供货备份、提升终端库存安全水位"
}


def _template_explain_node(ctx, question=""):
    """基于强约束JSON Context生成单节点业务解释"""
    node = ctx["node"]
    m = ctx["metrics"]
    ev = ctx["evidence"]
    quadrant = QUADRANT_DESC.get(m["matrix_quadrant"], m["matrix_quadrant"])
    parts = []

    # 是/否类问题直接回答
    q = question.lower()
    if re.search(r"是.*致命|是.*致命风险|是.*命门|是.*关键|是.*核心", q):
        is_fatal = node["is_cut_point"] or m["shapley_responsibility"] > 0.08
        if is_fatal:
            parts.append(f"【是】{node['name']}（{node['id']}）是致命风险。")
            parts.append(f"理由：该节点责任度（Shapley边际贡献）为{m['shapley_responsibility']:.4f}，"
                         f"{'且为割点，一旦失效网络将被分割。' if node['is_cut_point'] else '失效后对全局破坏力极大。'}")
        else:
            parts.append(f"【否】{node['name']}（{node['id']}）不是致命风险。")
            parts.append(f"理由：该节点责任度（Shapley边际贡献）为{m['shapley_responsibility']:.4f}，"
                         f"{'虽为割点但责任度较低。' if node['is_cut_point'] else '责任度较低，失效影响可控。'}")
        parts.append(f"二维矩阵定位：{quadrant}。")
        parts.append(f"暴露度评分：{m['exposure_score']:.2f}，"
                     f"证据锚点：「{ev['anchor']}」（来源：{ev['source']}，{ev['date']}）。")
        defense = DEFENSE_MEASURES.get(node["type"], "建立冗余与缓冲机制")
        parts.append(f"防御建议：{defense}。")
        return "\n".join(parts)

    parts.append(f"【{node['name']}（{node['id']}）风险解读】")
    parts.append(f"节点类型：{node['type']}，角色：{node['role']}。")
    if node["key_convert"]:
        parts.append("该节点为关键转换点，承担核心产能转换职能。")
    if node["is_cut_point"]:
        parts.append("拓扑分析显示该节点为割点：一旦失效，网络将被分割，"
                     "部分需求汇将失去供应通路，属于结构性命门。")
    parts.append(f"责任度（Shapley边际贡献）：{m['shapley_responsibility']:.4f}，"
                 f"表示该节点失效对全局供应保障的平均破坏力。")
    parts.append(f"暴露度评分：{m['exposure_score']:.2f}，"
                 f"证据锚点：「{ev['anchor']}」（来源：{ev['source']}，{ev['date']}）。")
    parts.append(f"二维矩阵定位：{quadrant}。")
    if m["matrix_quadrant"] == "hidden_critical":
        parts.append("⚠ 该节点属于'隐性关键'：平时暴露度低、不易触发预警，"
                     "但一旦失效对全局破坏极大，是典型的潜伏炸弹，应优先配置韧性投资。")
    elif m["matrix_quadrant"] == "dominant_core":
        parts.append("该节点暴露度与责任度双高，是当前最紧迫的风险焦点，"
                     "应急资源应优先保障。")
    elif m["matrix_quadrant"] == "local_disturbance":
        parts.append("该节点虽然暴露度较高（舆情/气象热度大），但拓扑责任度有限，"
                     "属于'虚假警报'，避免过度投入应急资源造成错配。")
    # 防御建议
    defense = DEFENSE_MEASURES.get(node["type"], "建立冗余与缓冲机制")
    parts.append(f"防御建议：{defense}。")
    if node["mincut_impact"] > 0.1:
        parts.append(f"该节点失效将导致最小割容量下降{node['mincut_impact']:.0%}，"
                     f"建议将其纳入瓶颈割集重点监控清单。")
    return "\n".join(parts)


def _template_explain_matrix(matrix_data):
    """二维矩阵全局解释"""
    parts = ["【暴露度—责任度二维交叉矩阵总览】"]
    for q_key, q_name in QUADRANT_DESC.items():
        nodes = [n for n in matrix_data if n["quadrant"] == q_key]
        if nodes:
            names = "、".join(f"{n['name']}({n['node_id']})" for n in nodes)
            parts.append(f"{q_name}：{names}")
    parts.append("决策建议：应急预算优先投向'显性核心'与'隐性关键'节点；"
                 "'局部扰动'节点保持常规监控即可，避免资源错配；"
                 "'常规节点'纳入标准运维流程。")
    return "\n".join(parts)


def _template_general(ctx_pool, question, matrix_data, shapley_rank):
    """通用问题回答（基于事实，不编造）"""
    # 收集问题中提到的所有节点ID
    all_matches = NODE_ID_PATTERN.findall(question)
    matched_nids = [f"N{int(m)}" for m in all_matches]
    matched_ctxs = [ctx for ctx in ctx_pool if ctx["node"]["id"] in matched_nids]

    # 多节点对比/列举类问题
    if len(matched_ctxs) >= 2 or re.search(r"分别|对比|比较|区别|哪些|分别是什么", question):
        if matched_ctxs:
            parts = ["【节点角色与类型对比】"]
            for ctx in matched_ctxs:
                node = ctx["node"]
                m = ctx["metrics"]
                quadrant = QUADRANT_DESC.get(m["matrix_quadrant"], m["matrix_quadrant"])
                parts.append(
                    f"• {node['name']}（{node['id']}）："
                    f"类型={node['type']}，角色={node['role']}，"
                    f"责任度={m['shapley_responsibility']:.4f}，"
                    f"暴露度={m['exposure_score']:.2f}，"
                    f"矩阵定位={quadrant}"
                )
                if node["is_cut_point"]:
                    parts.append(f"  → 该节点为割点，属于结构性命门")
                if node["key_convert"]:
                    parts.append(f"  → 该节点为关键转换点")
            return "\n".join(parts)

    # 单节点查询
    if len(matched_ctxs) == 1:
        return _template_explain_node(matched_ctxs[0], question)

    # 预算/资源类问题
    if re.search(r"预算|资源|钱|投入|优先", question):
        top3 = shapley_rank[:3]
        parts = ["【应急资源配置建议】基于Shapley责任度排序，预算应优先投向："]
        for i, (nid, val) in enumerate(top3, 1):
            parts.append(f"{i}. {nid}（责任度{val:.4f}）")
        parts.append("责任度反映节点失效对全局供应保障的边际破坏力，"
                     "按责任度配置资源可最大化韧性投资效率。")
        return "\n".join(parts)

    # 隐性关键/潜伏
    if re.search(r"隐性|潜伏|隐藏|不显眼", question):
        hidden = [n for n in matrix_data if n["quadrant"] == "hidden_critical"]
        if hidden:
            names = "、".join(f"{n['name']}({n['node_id']})" for n in hidden)
            parts = [f"【隐性关键节点识别】{names}",
                     "这些节点暴露度低（未触发气象阈值或舆情热度），"
                     "但拓扑责任度高，一旦失效对全局破坏极大，是典型的'沉默炸弹'。",
                     "",
                     "【归因解释】",
                     "• 暴露度低：当前舆情/气象数据未触发阈值，日常监控中不易被察觉。",
                     "• 责任度高：Shapley值显示其失效对全局供应保障的边际破坏力大，"
                     "拓扑上处于关键路径或割点位置。",
                     "• 错配陷阱：若仅按暴露度配置应急资源，这些节点将被遗漏，"
                     "形成'高暴露陷阱'。",
                     "",
                     "【四级应急策略】",
                     "一级（预防）：建立冗余备份（双仓互备/备选奶源/备产线），"
                     "降低单点依赖。",
                     "二级（监测）：部署IoT传感器与舆情爬虫，提升早期预警灵敏度，"
                     "打破'低暴露'盲区。",
                     "三级（响应）：制定专项应急预案，明确失效后的调拨路径与替代方案，"
                     "缩短恢复时间。",
                     "四级（恢复）：预留应急预算与缓冲库存，确保节点失效后"
                     "72小时内恢复主要供应通路。"]
            return "\n".join(parts)
        return "当前矩阵中未识别出隐性关键节点。"

    # 矩阵/全局
    if re.search(r"矩阵|全局|总览|整体|所有节点", question):
        return _template_explain_matrix(matrix_data)

    # 割集/瓶颈/木桶
    if re.search(r"割集|瓶颈|木桶|最短|脆弱", question):
        return ("【木桶效应数学化】全局瓶颈由最小割/瓶颈约束决定，而非单一'最短板'。"
                "系统通过多源多汇最大流与最小割分析识别全局瓶颈割集，"
                "请查看'割集分析'面板获取具体瓶颈边与割点清单。")

    # 未知问题兜底：系统不知道时不编造
    return _template_unknown(question, matrix_data, shapley_rank)


def _template_unknown(question, matrix_data, shapley_rank):
    """对无法识别意图的问题，明确告知系统无法回答，不编造答案"""
    top3 = shapley_rank[:3]
    top3_str = "、".join(f"{nid}（{val:.4f}）" for nid, val in top3)
    return (
        "【系统提示】您的问题超出了当前智能体的回答范围，"
        "我无法基于已计算的拓扑指标和证据锚点给出可靠答案。\n\n"
        "本系统目前支持以下类型的问题：\n"
        "• 单节点风险解读（如：N3风险如何？N6是致命风险吗？）\n"
        "• 多节点对比（如：N3和N5有什么区别？）\n"
        "• 隐性关键节点识别与归因分析\n"
        "• 应急资源配置建议（预算优先投向哪些节点）\n"
        "• 暴露度—责任度二维矩阵总览\n"
        "• 割集脆弱性与瓶颈分析\n\n"
        f"当前责任度最高的三个节点：{top3_str}。\n"
        "您可以尝试围绕以上主题提问。"
    )


def _template_out_of_scope(question):
    """对越界问题的礼貌拒答"""
    return ("【边界声明】本智能体仅回答与供应链局部灾害级联失效风险归因相关的问题，"
            "包括节点暴露度、拓扑责任度、割集脆弱性、防御措施建议。\n\n"
            "您的问题「" + question + "」超出了本系统的业务边界，"
            "我无法回答该问题。请围绕供应链节点风险、暴露度、责任度、"
            "防御措施等主题提问。")


# ============ DeepSeek API 接入（可选） ============
def call_deepseek_api(system_prompt, user_prompt):
    """若配置了DEEPSEEK_API_KEY则调用DeepSeek API，否则返回None"""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        return None
    try:
        import requests
        resp = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={"model": "deepseek-chat",
                  "messages": [{"role": "system", "content": system_prompt},
                               {"role": "user", "content": user_prompt}],
                  "temperature": 0.3, "max_tokens": 800},
            timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception:
        return None


SYSTEM_PROMPT = (
    "你是供应链风险归因系统的'业务翻译官'。你只能基于用户提供的JSON Context中的事实"
    "进行业务解释，严禁自行计算数值，严禁推断Context中未提供的节点关系。"
    "回答需面向管理层，简洁、可操作。对超出局部灾害级联失效范围的问题明确拒答。"
)