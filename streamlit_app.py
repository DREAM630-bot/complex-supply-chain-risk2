# -*- coding: utf-8 -*-
"""
供应链风险全景测度系统 - Streamlit 前端
只负责 UI，通过 HTTP 调用独立部署的 Flask 后端。
"""
import os
import time
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="供应链风险全景测度系统", layout="wide")


def get_secret(key, default=""):
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, default)


API_BASE = get_secret("API_BASE", "").rstrip("/")
if not API_BASE:
    st.error("未配置 API_BASE。请在 Streamlit Secrets 中设置后端地址，例如：")
    st.code('API_BASE = "https://your-backend.onrender.com"', language="toml")
    st.stop()


def api_get(path: str, timeout: int = 120):
    """带缓存的 GET；不要用于 /api/status。"""
    @st.cache_data(ttl=300, show_spinner=False)
    def _cached(base: str, p: str, t: int):
        r = requests.get(f"{base}{p}", timeout=t)
        r.raise_for_status()
        return r.json()
    return _cached(API_BASE, path, timeout)


def api_post(path: str, payload: dict, timeout: int = 120):
    r = requests.post(f"{API_BASE}{path}", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def backend_status():
    """不走缓存，用于冷启动轮询。"""
    r = requests.get(f"{API_BASE}/api/status", timeout=10)
    r.raise_for_status()
    return r.json()


def wait_ready(max_wait=180):
    placeholder = st.empty()
    start = time.time()
    while time.time() - start < max_wait:
        try:
            s = backend_status()
            if s.get("ready"):
                placeholder.empty()
                return s
            placeholder.info("后端预计算中，请稍候…")
        except Exception as e:
            placeholder.warning(f"等待后端启动：{e}")
        time.sleep(3)
    placeholder.error("后端长时间未就绪，请检查后端服务地址或日志。")
    return None


if "status" not in st.session_state:
    st.session_state.status = wait_ready()
if not st.session_state.status:
    st.stop()

status = st.session_state.status

# ============ 侧边栏 ============
st.sidebar.title("供应链风险全景测度系统")
page = st.sidebar.radio(
    "导航",
    ["风险矩阵", "节点详情", "拓扑视图", "割点与候选", "智能问答", "系统状态"],
)
st.sidebar.caption(f"后端：{API_BASE}")
if st.sidebar.button("清除前端缓存并刷新"):
    st.cache_data.clear()
    st.rerun()

# ============ 风险矩阵 ============
if page == "风险矩阵":
    st.header("暴露度 × 责任度 二维矩阵")
    data = api_get("/api/matrix")
    df = pd.DataFrame(data["matrix"])
    exp_th = data["exp_threshold"]
    resp_th = data["resp_threshold"]

    st.caption(
        f"暴露度阈值 {exp_th:.4f}｜责任度阈值 {resp_th:.4f}｜"
        f"Monte Carlo 采样 {data['n_samples']}"
    )
    st.scatter_chart(
        df, x="exposure", y="responsibility", color="quadrant",
        use_container_width=True,
    )
    st.dataframe(
        df[["node_id", "name", "type", "role", "exposure",
            "responsibility", "quadrant", "is_cut_point", "region"]],
        use_container_width=True, hide_index=True,
    )

    st.subheader("Shapley 责任度排名")
    st.dataframe(
        pd.DataFrame(data["shapley_rank"]),
        use_container_width=True, hide_index=True,
    )

    with st.expander("交叉验证结果"):
        st.json(data.get("cross_validation", {}), expanded=True)

# ============ 节点详情 ============
elif page == "节点详情":
    st.header("节点详情")
    matrix = api_get("/api/matrix")["matrix"]
    node_ids = [m["node_id"] for m in matrix]
    name_map = {m["node_id"]: m["name"] for m in matrix}
    nid = st.selectbox(
        "选择节点", node_ids,
        format_func=lambda x: f"{x} - {name_map.get(x, '')}",
    )
    detail = api_get(f"/api/node/{nid}")
    c1, c2, c3 = st.columns(3)
    c1.metric("节点", nid)
    c2.metric("Shapley 责任度", f"{detail.get('shapley', 0):.4f}")
    c3.metric("名称", name_map.get(nid, ""))

    st.subheader("拓扑画像 profile")
    st.json(detail.get("profile", {}), expanded=False)

    if detail.get("context"):
        st.subheader("结构化上下文 context")
        st.json(detail["context"], expanded=False)

# ============ 拓扑视图 ============
elif page == "拓扑视图":
    st.header("供应链拓扑")
    topo = api_get("/api/topology")
    nodes = pd.DataFrame(topo["nodes"])
    edges = pd.DataFrame(topo["edges"])

    st.subheader("节点地理分布")
    if {"lat", "lon"}.issubset(nodes.columns):
        st.map(nodes[["lat", "lon"]].dropna(), use_container_width=True)

    st.subheader("节点表")
    st.dataframe(nodes, use_container_width=True, hide_index=True)

    st.subheader("边表")
    st.dataframe(edges, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("源点 sources")
        st.write(topo.get("sources", []))
    with col2:
        st.subheader("汇点 sinks")
        st.write(topo.get("sinks", []))

    with st.expander("区域 regions"):
        st.json(topo.get("regions", {}), expanded=True)

# ============ 割点与候选 ============
elif page == "割点与候选":
    st.header("割点分析与关键候选")
    data = api_get("/api/cuts")
    cuts = data.get("cuts", {})
    candidates = data.get("candidates", [])

    st.subheader("割点")
    st.json(cuts, expanded=True)

    st.subheader("关键候选")
    if candidates:
        st.dataframe(pd.DataFrame(candidates), use_container_width=True, hide_index=True)
    else:
        st.info("无候选数据")

# ============ 智能问答 ============
elif page == "智能问答":
    st.header("受约束解释智能体")
    st.caption("问题中可包含节点编号，例如 N3、N6；越界问题会被后端拦截。")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("meta"):
                st.caption(msg["meta"])

    q = st.chat_input("请输入问题…")
    if q:
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)

        with st.chat_message("assistant"):
            with st.spinner("思考中…"):
                try:
                    res = api_post("/api/chat", {"question": q}, timeout=120)
                    reply = res.get("reply", "")
                    meta = (
                        f"引擎：{res.get('engine', '')}｜"
                        f"in_scope={res.get('in_scope')}"
                    )
                    st.markdown(reply)
                    st.caption(meta)
                except Exception as e:
                    reply = f"请求后端失败：{e}"
                    meta = ""
                    st.error(reply)

        st.session_state.messages.append(
            {"role": "assistant", "content": reply, "meta": meta}
        )

# ============ 系统状态 ============
else:
    st.header("系统状态")
    st.json(status, expanded=True)

    st.subheader("按需计算：敏感性分析")
    st.caption("该接口较慢，点击按钮后请求，超时 300 秒。")
    if st.button("运行敏感性分析"):
        with st.spinner("计算中，请稍候…"):
            try:
                res = api_get("/api/sensitivity", timeout=300)
                st.subheader("sensitivity")
                st.json(res.get("sensitivity", {}), expanded=False)
                st.subheader("symmetry")
                st.json(res.get("symmetry", {}), expanded=False)
            except Exception as e:
                st.error(f"敏感性分析失败：{e}")

    st.subheader("pipeline")
    if st.button("加载 pipeline"):
        try:
            st.json(api_get("/api/pipeline"), expanded=False)
        except Exception as e:
            st.error(f"加载失败：{e}")

    st.subheader("baseline")
    if st.button("加载 baseline"):
        try:
            st.json(api_get("/api/baseline"), expanded=False)
        except Exception as e:
            st.error(f"加载失败：{e}")
