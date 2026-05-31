# Visual RAG — 实现方案 (Implementation Plan)

> 基于 MSc Project Plan，结合约束：本地 4050 (6GB VRAM) + 按需云 GPU；LLM 先 API 后开源；先用现成 benchmark。

## 0. 核心架构决策 (一句话总结)

**离线重计算 (云 GPU) → 落盘 embedding + 向量库 → 在线查询期全部轻量 (本地 6GB + LLM API)。**

6GB 显存跑不动 Whisper large-v3 / CLIP ViT-L 的批量任务，但查询期只需要：编码一句 query 文本 (CLIP text encoder 极小) + 向量检索 + cross-encoder 重排 (小模型) + 调 LLM API。重活全在离线一次性完成。

---

## 1. 计算策略 (对应 6GB + 按需云)

| 任务 | 在哪跑 | 模型/工具 |
|------|--------|-----------|
| 帧抽取 + 场景检测 | 本地 CPU | PySceneDetect + 均匀采样兜底 |
| **批量** ASR 转写 | 云 GPU (一次性) | faster-whisper large-v3 (CTranslate2, int8) |
| **批量** 视觉 embedding | 云 GPU (一次性) | open_clip: ViT-B/32 → ViT-L/14 → SigLIP |
| OCR (可选增强) | 本地/云 | PaddleOCR (按需，benchmark 优先级低) |
| 查询期文本编码 | 本地 6GB | CLIP/SigLIP text encoder (轻) |
| 向量检索 | 本地 CPU | ChromaDB (HNSW) |
| Cross-encoder 重排 | 本地 6GB | bge-reranker-base / ms-marco MiniLM |
| Agent 推理 + 答案合成 | LLM API | GPT-4o / Claude (后期换开源对比) |

**云 GPU 用法**：RunPod / Lambda 上租 1×4090 或 A10，跑几小时做全量预计算，embedding 落成 parquet + 灌进 Chroma，下载到本地。整个数据集预计算成本约 $5–15。本地用 ViT-B/32 + faster-whisper small (int8) 做小规模重跑和调试，6GB 够用。

---

## 2. 技术选型 (锁定)

- **向量库：ChromaDB**（嵌入式、零运维、单机够用）。Milvus 在这个规模是过度工程，不用。segment 级 embedding，metadata 存 `video_id / start / end / modality`。
- **视觉模型：open_clip 统一接口**。MVP 用 `ViT-B-32`，最终用 `ViT-L-14` 和 SigLIP (`ViT-SO400M`) 做 RQ2 的消融对比——一套接口切换三个 backbone。
- **ASR：faster-whisper**（比官方 whisper 省显存省时间）。
- **RAG/Agent 框架：不锁死 LangChain**。
  - 检索核心写成薄的自定义模块（可控、好评估）。
  - MVP agent 用 **LLM 原生 function-calling** 跑通单轮工具调用。
  - Phase 2 多步 agent 用 **LangGraph** 做状态机（ReAct + self-reflection），比裸 LangChain 可控。
- **重排：cross-encoder** (`BAAI/bge-reranker-base`)，6GB 本地可跑。

---

## 3. 数据与评估 (先 benchmark)

**主 benchmark：NExT-QA + NExT-GQA**
- NExT-QA：因果/时序多选 QA，视频短 (~44s)、规模可控，迭代快。
- **NExT-GQA**：在 NExT-QA 上加了**时序定位标注**——直接支撑计划里的"Temporal Accuracy"指标（检索片段时间戳 vs 真值区间），这是它比 MSRVTT/ActivityNet 更契合本项目的关键。
- 跑通后再扩到自建小库 (20–30 视频) 做 demo 展示。

**指标 (对应计划 §5)**
- 检索：Recall@K (1/5/10)、MRR、nDCG（用 NExT-GQA 时序真值区间判定命中，IoU 阈值）。
- QA：NExT-QA 多选准确率 + 开放式答案 (faithfulness / relevance，可用 LLM-as-judge)。
- 时序：预测片段与真值区间 IoU / tIoU。
- 系统：端到端延迟、embedding 吞吐、索引大小。

**基线对比 (RQ1)**：纯文本检索 (只用 Whisper 转写文本做 embedding/BM25) vs 多模态。
**消融 (RQ2)**：visual-only / audio-only / text-only / fused，逐模态贡献。

---

## 4. 仓库结构

```
visualrag/
  ingest/        # 帧抽取、场景检测、Whisper、OCR
  embed/         # open_clip / siglip 封装，批量编码脚本
  index/         # ChromaDB 封装，segment schema，灌库
  retrieve/      # 检索 + cross-encoder 重排 + 时序邻居
  agent/         # 查询分解、工具 (search/filter/compare/summarize)、ReAct 循环
  eval/          # NExT-QA/GQA 加载、Recall@K/MRR/nDCG/IoU、基线、消融
  ui/            # Streamlit (查询 + 时间戳跳转播放)
  configs/       # 模型/路径/超参 (yaml)
scripts/         # 一次性云 GPU 预计算入口
data/            # benchmark、预计算 embedding (gitignore)
```

---

## 5. 里程碑 (映射 12 周，强调 MVP-first)

- **W1**：环境 + NExT-QA/GQA 下载解析；ingest 骨架；选定云 GPU 流程。
- **W2**：云 GPU 跑全量 Whisper + 帧 embedding，落盘 + 灌 Chroma。
- **W3**：检索基线评估 (Recall@K/MRR/nDCG)，纯文本 vs 多模态首版对比。
- **W4 ✅ MVP 垂直切片**：query → embed → retrieve → LLM 合成带时间戳引用的答案，单轮 function-calling 跑通。
- **W5**：查询分解 + 多步迭代检索。
- **W6**：cross-encoder 重排 + 时序推理 ("X 之后发生了什么")。
- **W7**：LangGraph agent (工具 + self-reflection)，端到端测试。
- **W8–9**：完整评估、基线对比、逐模态消融 (RQ2)；此处**接入开源 LLM (Llama3/Qwen) 与 API 做对比**。
- **W10**：性能优化 (融合策略、检索速度、延迟)；小型用户研究。
- **W11–12**：论文初稿 + 打磨 demo UI；清理、文档、提交。

---

## 6. 关键风险与处置

- **6GB 跑不动大模型** → 重计算全部上云，本地只做轻量查询期与小模型调试 (已在 §1 解决)。
- **embedding 融合方式 (RQ2)** → 不要一上来就 concat。先做**分模态独立检索 + 分数融合 (late fusion)**，再实验 early fusion，late fusion 更易消融、更稳。
- **OCR 优先级** → benchmark 视频屏幕文字少，OCR 列为可选增强，别卡 Phase 1 主线。
- **scope creep** → W4 的 MVP 垂直切片是硬门槛，先有端到端可评估系统再加 agent 复杂度。
- **API 成本** → 评估期批量查询用缓存 + 开源模型兜底 (与计划风险表一致)。

---

## 7. 待你确认的下一步

1. 立即可做：`pip` 环境 + 下载 NExT-QA/GQA + 搭 ingest 骨架。
2. 你来定云 GPU 平台 (RunPod / Lambda / Colab Pro)，我来写一次性预计算脚本。
3. 确认主 benchmark 用 NExT-QA + NExT-GQA（如果你导师更倾向 MSRVTT/ActivityNet 我再调整）。
```
