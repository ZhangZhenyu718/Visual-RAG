# -*- coding: utf-8 -*-
"""Visual RAG demo UI (W10 demo / plan `ui/`): retrieval with timestamp-jump
playback + agent QA.

    streamlit run ui/app.py

Tabs:
  1. 片段检索 — query -> (optional decompose/rerank) -> segments with keyframes,
     each expandable into the source video seeked to the segment start.
  2. 视频问答 — the W4/W7 agents (simple loop / LangGraph), DeepSeek or Claude.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from visualrag.utils.config import load_config


def _load_user_env_keys():
    """Windows: pick up setx-persisted API keys even if the parent shell is stale."""
    if os.name != "nt":
        return
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            for name in ("DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY"):
                if not os.environ.get(name):
                    try:
                        os.environ[name] = winreg.QueryValueEx(k, name)[0]
                    except FileNotFoundError:
                        pass
    except Exception:
        pass


_load_user_env_keys()

VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".avi", ".mov")


@st.cache_resource
def cfg_for(path: str):
    return load_config(path)


@st.cache_resource
def retriever_for(path: str):
    from visualrag.retrieve.retriever import Retriever
    return Retriever(cfg_for(path))


@st.cache_resource
def decomposer_for(path: str):
    from visualrag.retrieve.decompose import QueryDecomposer
    return QueryDecomposer(cfg_for(path), cache_name="ui")


@st.cache_resource
def reranker_for(path: str):
    from visualrag.retrieve.rerank import make_reranker
    return make_reranker(cfg_for(path))


@st.cache_resource
def segments_for(path: str):
    from visualrag.agent.answerer import _SegmentLookup
    return _SegmentLookup(cfg_for(path))


@st.cache_resource
def qa_for(path: str, provider: str, agent_kind: str):
    from visualrag.utils.config import Config
    base = cfg_for(path)
    cfg = Config({**base, "agent": {**base.get("agent", {}), "provider": provider}})
    if provider == "claude":
        cfg["agent"].pop("model", None)
    if agent_kind == "graph":
        from visualrag.agent.graph_agent import GraphVideoQA
        return GraphVideoQA(cfg)
    from visualrag.agent.answerer import VideoQA
    return VideoQA(cfg)


@st.cache_data
def video_index(videos_dir: str) -> dict:
    index = {}
    for root, _dirs, files in os.walk(videos_dir):
        for fn in files:
            stem, ext = os.path.splitext(fn)
            if ext.lower() in VIDEO_EXTS:
                index.setdefault(stem, os.path.join(root, fn))
    return index


def render_hit(cfg_path: str, hit: dict, rank: int):
    m = hit["metadata"]
    seg = segments_for(cfg_path).get(hit["segment_id"])
    kfs = seg.get("keyframe_paths", [])
    transcript = (seg.get("transcript", "") + " " + seg.get("ocr_text", "")).strip()

    left, right = st.columns([1, 2], vertical_alignment="center")
    with left:
        if kfs and os.path.exists(kfs[len(kfs) // 2]):
            st.image(kfs[len(kfs) // 2], width="stretch")
        else:
            st.caption("(无关键帧)")
    with right:
        st.markdown(f"**#{rank} · `{m.get('video_id')}` · "
                    f"{m.get('start'):.1f}–{m.get('end'):.1f}s** · score {hit['score']:.3f}")
        st.caption(transcript if transcript else "(无语音)")
        vids = video_index(cfg_for(cfg_path).get_path("paths.videos"))
        vpath = vids.get(str(m.get("video_id")))
        if vpath:
            with st.expander("▶ 播放该时刻"):
                st.video(vpath, start_time=int(m.get("start", 0)))
    st.divider()


def tab_search(cfg_path: str):
    cfg = cfg_for(cfg_path)
    with st.sidebar:
        st.subheader("检索设置")
        modality = st.selectbox("模态", ["visual", "text", "fused"], index=0)
        k = st.slider("Top-K", 3, 20, 8)
        use_decompose = st.toggle("查询分解 (W5, 需 DeepSeek key)", value=False)
        use_rerank = st.toggle("视觉重排 (W6, ViT-L)", value=True)

    query = st.text_input("描述你要找的画面或台词",
                          placeholder="例如: a boy unwrapping a present on a sofa")
    if not query:
        st.info("输入查询后回车。索引: 567 个 NExT-QA val 视频, 5725 个 8 秒片段。")
        return

    retriever = retriever_for(cfg_path)
    queries = [query]
    if use_decompose:
        with st.spinner("LLM 分解查询中..."):
            queries = decomposer_for(cfg_path).decompose(query)
        st.caption("子查询: " + " | ".join(f"`{q}`" for q in queries))

    pool = int(cfg.get_path("rerank.candidates", 30)) if use_rerank else k
    with st.spinner("检索中..."):
        if len(queries) > 1:
            hits = retriever.search_multi(queries, modality=modality, k=pool)
        else:
            hits = retriever.search(query, modality=modality, k=pool)
        if use_rerank:
            hits = reranker_for(cfg_path).rerank(query, hits, k=k, queries=queries)
        hits = hits[:k]

    for i, h in enumerate(hits, 1):
        render_hit(cfg_path, h, i)


def tab_ask(cfg_path: str):
    with st.sidebar:
        st.subheader("问答设置")
        provider = st.selectbox("LLM", ["deepseek", "claude"], index=0,
                                help="deepseek: 便宜, 纯文本证据 | claude: 关键帧图片进上下文")
        agent_kind = st.selectbox("Agent", ["simple", "graph"], index=1,
                                  help="graph = LangGraph ReAct + 自反思 (W7); claude 仅支持 simple")
        video_scope = st.text_input("限定 video_id (可选)", "")

    if provider == "claude" and agent_kind == "graph":
        st.warning("graph agent 目前仅支持 DeepSeek; 已自动切换为 simple。")
        agent_kind = "simple"
    key_name = "DEEPSEEK_API_KEY" if provider == "deepseek" else "ANTHROPIC_API_KEY"
    if not os.environ.get(key_name):
        st.error(f"缺少 {key_name} 环境变量")
        return

    question = st.text_input("提问", placeholder="例如: what does the white dog do after going to the cushion")
    if not question:
        st.info("提问后回车。Agent 会检索片段、按需做时序推理, 并给出带时间戳引用的回答。")
        return

    qa = qa_for(cfg_path, provider, agent_kind)
    with st.spinner(f"{provider}:{agent_kind} agent 思考中..."):
        result = qa.answer(question, video_id=video_scope.strip() or None)

    st.markdown("### 回答")
    st.markdown(result["answer"] or "*（空回答）*")

    with st.expander("证据轨迹 (工具调用)"):
        for s in result["searches"]:
            if "hits" in s:
                st.markdown(f"🔍 **search** `{s['query']}` ({s['modality']})")
                for h in s["hits"][:4]:
                    m = h["metadata"]
                    st.caption(f"　{h['score']:.3f} · {m.get('video_id')} "
                               f"[{m.get('start')}–{m.get('end')}s]")
            else:
                st.markdown(f"⏱ **{s['tool']}** `{s['args']}`")
    u = result["usage"]
    extra = f" · rounds {result['rounds']} · reflections {result['reflections']}" if "rounds" in result else ""
    st.caption(f"tokens: {u['input_tokens']} in / {u['output_tokens']} out{extra}")


def main():
    st.set_page_config(page_title="Visual RAG", page_icon="🎬", layout="wide")
    st.title("🎬 Visual RAG — 视频片段检索与问答")

    with st.sidebar:
        st.subheader("数据")
        cfg_path = st.selectbox("配置", ["configs/default.yaml", "configs/demo.yaml"],
                                format_func=lambda p: "NExT-QA val (567 视频)"
                                if "default" in p else "Demo (4 视频)")
        st.divider()

    t1, t2 = st.tabs(["🔍 片段检索", "💬 视频问答"])
    with t1:
        tab_search(cfg_path)
    with t2:
        tab_ask(cfg_path)


if __name__ == "__main__":
    main()
