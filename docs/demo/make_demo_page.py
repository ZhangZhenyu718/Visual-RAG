# -*- coding: utf-8 -*-
"""Generate the Visual RAG demo HTML page with embedded keyframes."""
import base64
import html
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # repo root (keyframe paths are repo-relative)
RESULTS = os.path.join(HERE, "demo_results.json")
OUT = os.path.join(HERE, "visualrag_demo.html")

DUR = {"bunny": 10.0, "jellyfish": 10.01, "sintel": 10.0, "nasa_snowflake": 163.93}
VID_LABEL = {
    "bunny": "bunny", "jellyfish": "jellyfish",
    "sintel": "sintel", "nasa_snowflake": "nasa_snowflake",
}

def b64(relpath):
    p = os.path.join(ROOT, relpath)
    with open(p, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()

CORPUS = [
    ("bunny", "artifacts_demo/frames/bunny/00007.00.jpg", "10 秒 · 无音频", "动画:兔子与森林"),
    ("jellyfish", "artifacts_demo/frames/jellyfish/00005.00.jpg", "10 秒 · 无音频", "实拍:水母游动"),
    ("sintel", "artifacts_demo/frames/sintel/00005.00.jpg", "10 秒 · 无音频", "动画:雪地打斗"),
    ("nasa_snowflake", "artifacts_demo/frames/nasa_snowflake/00005.00.jpg", "164 秒 · 英文解说", "NASA:雪花融化 3D 模型科普"),
]

ANNOT = {
    "a cartoon rabbit in a green forest":
        "纯视觉命中。bunny 视频没有声音也没有字幕(索引里 0 条文本向量),检索完全依赖 CLIP 对画面语义的理解。",
    "jellyfish swimming underwater":
        "水母视频的两个片段包揽前二,分数与无关视频拉开明显差距。",
    "a snowy mountain landscape":
        "全场最能说明问题的一条:164 秒的 NASA 视频里,雪山空镜只出现在 68–76 秒附近——检索没有停留在“找对视频”,而是直接命中了那 8 秒窗口。这就是本项目的核心目标:时序精确定位(temporal grounding)。",
    "two people fighting in the snow":
        "sintel 是运动模糊很重的动作镜头,画面检索依然把它排到前二。",
    "scientists explain how snowflakes melt":
        "融合检索(视觉 0.5 + 文本 0.5)。第一名恰好是解说词说出这句话的时刻:“For the first time ever, scientists have created a 3D model of a melting snowflake.”",
    "a 3D computer model":
        "两个重叠滑窗(144–152s / 148–156s)并列第一——都覆盖了解说提到 “computer power to make simulations” 的时刻。重叠窗口的设计保证事件不会被切在边界上。",
    "radar observations of precipitation":
        "纯文本模态(检索 Whisper 转写)。注意:转写里并没有出现 “radar” 这个词,但语义最接近的“降水模型 / 天气预报”解说段仍排在最前——向量检索按含义而非关键词匹配。",
}

HIGHLIGHT = {"a snowy mountain landscape"}

MOD_META = {
    "visual": ("visual", "视觉", "#7fb4e8"),
    "text": ("text", "文本", "#8fd0a0"),
    "fused": ("fused", "融合", "#c6a3e8"),
}

with open(RESULTS, encoding="utf-8") as f:
    results = json.load(f)

def timeline(vid, start, end):
    dur = DUR[vid]
    left = 100.0 * start / dur
    width = max(100.0 * (end - start) / dur, 1.2)
    return (
        f'<div class="tl"><div class="tl-track">'
        f'<div class="tl-band" style="left:{left:.2f}%;width:{width:.2f}%"></div>'
        f'</div><div class="tl-lab"><span>0s</span><span>{dur:.0f}s</span></div></div>'
    )

def hit_html(rank, h):
    img = b64(h["keyframe"]) if h["keyframe"] else ""
    txt = h["text"].strip()
    quote = f'<p class="quote">“{html.escape(txt)}”</p>' if txt else ""
    score_pct = min(h["score"], 1.0) * 100
    return f"""
    <div class="hit">
      <div class="rank">{rank}</div>
      <img class="thumb" src="{img}" alt="keyframe of {h['video_id']} at {h['start']}s">
      <div class="hit-body">
        <div class="hit-meta">
          <span class="vid">{h['video_id']}</span>
          <span class="tc">{h['start']:.1f}s – {h['end']:.1f}s</span>
          <span class="score"><i style="width:{score_pct:.0f}%"></i></span>
          <span class="score-n">{h['score']:.3f}</span>
        </div>
        {timeline(h['video_id'], h['start'], h['end'])}
        {quote}
      </div>
    </div>"""

cards = []
for q in results:
    mod_en, mod_zh, mod_c = MOD_META[q["modality"]]
    hits = "".join(hit_html(i + 1, h) for i, h in enumerate(q["hits"]))
    note = ANNOT.get(q["query"], "")
    hl = " hl" if q["query"] in HIGHLIGHT else ""
    cards.append(f"""
  <section class="card{hl}">
    <div class="card-head">
      <span class="chip" style="--c:{mod_c}">{mod_en} · {mod_zh}</span>
      <h2 class="query">“{html.escape(q['query'])}”</h2>
    </div>
    <div class="hits">{hits}</div>
    <p class="note">{note}</p>
  </section>""")

corpus_cards = "".join(
    f"""<div class="v">
      <img src="{b64(p)}" alt="{vid}">
      <div class="v-name">{vid}</div>
      <div class="v-meta">{meta}</div>
      <div class="v-desc">{desc}</div>
    </div>"""
    for vid, p, meta, desc in CORPUS
)

page = f"""<title>Visual RAG 检索演示</title>
<style>
  :root {{
    --bg: #131720; --panel: #1b2130; --panel2: #171c29; --line: #2a3145;
    --ink: #e9ecf4; --muted: #98a1b6; --amber: #e6b45c;
    --mono: "Cascadia Code", Consolas, "Courier New", monospace;
  }}
  html {{ background: var(--bg); }}
  body {{
    margin: 0; padding: 40px 20px 72px; background: var(--bg); color: var(--ink);
    font-family: "Segoe UI", "Microsoft YaHei UI", "PingFang SC", sans-serif;
    line-height: 1.65;
  }}
  .wrap {{ max-width: 900px; margin: 0 auto; display: flex; flex-direction: column; gap: 22px; }}
  header h1 {{ font-size: 26px; margin: 0 0 6px; letter-spacing: .2px; text-wrap: balance; }}
  header p {{ margin: 0; color: var(--muted); max-width: 68ch; }}
  .stats {{
    display: flex; flex-wrap: wrap; gap: 8px 22px; margin-top: 14px;
    font-family: var(--mono); font-size: 12.5px; color: var(--muted);
    font-variant-numeric: tabular-nums;
  }}
  .stats b {{ color: var(--amber); font-weight: 600; }}
  .flow {{
    display: flex; flex-wrap: wrap; gap: 6px; align-items: center;
    font-size: 12.5px; color: var(--muted);
  }}
  .flow span {{ background: var(--panel2); border: 1px solid var(--line); border-radius: 4px; padding: 3px 10px; }}
  .flow i {{ font-style: normal; color: #4a5470; }}
  .corpus {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
  .v {{ background: var(--panel2); border: 1px solid var(--line); border-radius: 6px; padding: 10px; }}
  .v img {{ width: 100%; border-radius: 4px; display: block; aspect-ratio: 16/9; object-fit: cover; }}
  .v-name {{ font-family: var(--mono); font-size: 13px; margin-top: 8px; }}
  .v-meta {{ font-family: var(--mono); font-size: 11.5px; color: var(--amber); }}
  .v-desc {{ font-size: 12.5px; color: var(--muted); }}
  h3.sec {{ font-size: 13px; text-transform: uppercase; letter-spacing: .12em; color: var(--muted); margin: 10px 0 -8px; font-weight: 600; }}
  .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 18px 20px 14px; }}
  .card.hl {{ border-color: #58502f; box-shadow: 0 0 0 1px #58502f; }}
  .card-head {{ display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; margin-bottom: 12px; }}
  .chip {{
    font-family: var(--mono); font-size: 11.5px; color: var(--c);
    border: 1px solid color-mix(in srgb, var(--c) 45%, transparent);
    background: color-mix(in srgb, var(--c) 10%, transparent);
    border-radius: 999px; padding: 2px 10px; white-space: nowrap;
  }}
  .query {{ font-size: 18px; font-weight: 600; margin: 0; letter-spacing: .1px; }}
  .hits {{ display: flex; flex-direction: column; gap: 12px; }}
  .hit {{ display: flex; gap: 14px; align-items: flex-start; background: var(--panel2); border-radius: 6px; padding: 10px 14px 10px 10px; }}
  .rank {{ font-family: var(--mono); color: #4a5470; font-size: 12px; padding-top: 2px; min-width: 12px; }}
  .thumb {{ width: 150px; aspect-ratio: 16/9; object-fit: cover; border-radius: 4px; flex: none; }}
  .hit-body {{ flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 6px; }}
  .hit-meta {{ display: flex; align-items: center; gap: 12px; flex-wrap: wrap; font-family: var(--mono); font-size: 12.5px; font-variant-numeric: tabular-nums; }}
  .vid {{ color: var(--ink); }}
  .tc {{ color: var(--amber); }}
  .score {{ width: 72px; height: 4px; background: #262d40; border-radius: 2px; overflow: hidden; }}
  .score i {{ display: block; height: 100%; background: var(--muted); }}
  .score-n {{ color: var(--muted); font-size: 12px; }}
  .tl-track {{ position: relative; height: 6px; background: #262d40; border-radius: 3px; }}
  .tl-band {{ position: absolute; top: 0; height: 100%; background: var(--amber); border-radius: 3px; }}
  .tl-lab {{ display: flex; justify-content: space-between; font-family: var(--mono); font-size: 10.5px; color: #4a5470; margin-top: 2px; }}
  .quote {{ margin: 2px 0 0; font-size: 13px; color: var(--muted); display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }}
  .note {{ margin: 14px 2px 2px; font-size: 13.5px; color: var(--muted); border-left: 2px solid var(--line); padding-left: 12px; max-width: 74ch; }}
  .card.hl .note {{ color: #d8c89a; border-left-color: var(--amber); }}
  footer {{ color: #4a5470; font-size: 12.5px; font-family: var(--mono); }}
  @media (max-width: 560px) {{
    .hit {{ flex-wrap: wrap; }}
    .thumb {{ width: 100%; }}
  }}
</style>
<div class="wrap">
  <header>
    <h1>Visual RAG 检索演示</h1>
    <p>端到端流水线在 4 个测试视频上的真实运行结果:文本查询 → CLIP 编码 → ChromaDB
       向量检索 → 返回精确到秒的视频片段。以下每张缩略图都是被命中片段自己的关键帧,
       时间轴亮条标出片段在整个视频里的位置。</p>
    <div class="stats">
      <span><b>4</b> 个视频</span><span><b>46</b> 个滑窗片段</span>
      <span><b>46</b> 视觉向量 + <b>38</b> 文本向量</span>
      <span>CLIP ViT-B-32</span><span>Whisper base</span><span>全程本地 CPU</span>
    </div>
  </header>

  <div class="flow">
    <span>视频</span><i>→</i><span>关键帧 · 场景检测</span><i>+</i><span>ASR · Whisper</span><i>→</i>
    <span>8s 滑窗片段</span><i>→</i><span>CLIP 视觉/文本编码</span><i>→</i>
    <span>ChromaDB 双集合</span><i>→</i><span>late fusion 检索</span>
  </div>

  <h3 class="sec">被索引的 4 个视频</h3>
  <div class="corpus">{corpus_cards}</div>

  <h3 class="sec">检索结果 · 每条查询取 top-3</h3>
  {''.join(cards)}

  <footer>configs/demo.yaml · scripts/ingest_dataset.py → build_index.py → query_index.py · 2026-07-06</footer>
</div>
"""

with open(OUT, "w", encoding="utf-8") as f:
    f.write(page)
print("written", OUT, f"{os.path.getsize(OUT)/1024:.0f} KB")
