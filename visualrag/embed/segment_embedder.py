"""Turn a video's Segments into per-segment visual + text vectors.

- Visual: mean-pool the (normalized) CLIP embeddings of the segment's keyframes,
  then re-normalize. Segments with no keyframe get no visual vector.
- Text: CLIP text embedding of `segment.text` (transcript + OCR). Empty -> none.

Results persist to `artifacts/embeddings/<video_id>.npz` (resumable / portable);
re-runs load the cache unless `overwrite=True`.
"""

from __future__ import annotations

import os
import numpy as np

from visualrag.schema import Segment


def _emb_path(cfg, video_id: str) -> str:
    root = cfg.get_path("paths.embeddings", "artifacts/embeddings")
    return os.path.join(root, f"{video_id}.npz")


def embed_video_segments(video_id: str, segments: list[Segment], encoder, cfg,
                         overwrite: bool = False) -> dict:
    """Return {segment_ids, visual[N,D], text[N,D], has_visual[N], has_text[N]}.

    Missing-modality rows are zero vectors flagged False in the has_* masks.
    """
    out_path = _emb_path(cfg, video_id)
    if os.path.exists(out_path) and not overwrite:
        z = np.load(out_path, allow_pickle=True)
        return {k: z[k] for k in z.files}

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    D = encoder.dim
    n = len(segments)
    seg_ids = np.array([s.segment_id for s in segments], dtype=object)
    visual = np.zeros((n, D), dtype=np.float32)
    text = np.zeros((n, D), dtype=np.float32)
    has_visual = np.zeros(n, dtype=bool)
    has_text = np.zeros(n, dtype=bool)

    # --- Visual: encode each unique frame once, then pool per segment. ---
    unique_frames = sorted({p for s in segments for p in s.keyframe_paths})
    if unique_frames:
        frame_emb = encoder.encode_images(unique_frames)
        idx = {p: i for i, p in enumerate(unique_frames)}
        for i, s in enumerate(segments):
            if not s.keyframe_paths:
                continue
            rows = frame_emb[[idx[p] for p in s.keyframe_paths]]
            rows = rows[np.linalg.norm(rows, axis=1) > 0]  # drop unreadable frames
            if len(rows) == 0:
                continue
            pooled = rows.mean(axis=0)
            norm = np.linalg.norm(pooled)
            if norm > 0:
                visual[i] = pooled / norm
                has_visual[i] = True

    # --- Text: batch-encode non-empty segment texts. ---
    text_idx = [i for i, s in enumerate(segments) if s.text.strip()]
    if text_idx:
        text_emb = encoder.encode_texts([segments[i].text for i in text_idx])
        for j, i in enumerate(text_idx):
            text[i] = text_emb[j]
            has_text[i] = True

    result = {
        "segment_ids": seg_ids,
        "visual": visual,
        "text": text,
        "has_visual": has_visual,
        "has_text": has_text,
    }
    np.savez(out_path, **result)
    return result
