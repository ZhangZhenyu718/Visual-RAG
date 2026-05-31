"""Per-video ingestion orchestrator: video file -> keyframes + transcript + OCR
-> overlapping Segments, all persisted under `artifacts/`.

Designed to be idempotent (skips a video whose segments JSONL already exists) so
batch runs on cloud GPUs can be resumed cheaply.
"""

from __future__ import annotations

import os
from dataclasses import asdict

from visualrag.ingest.frames import extract_keyframes, get_video_duration
from visualrag.ingest.segment import build_segments
from visualrag.schema import (
    Keyframe, TranscriptChunk, OCRSpan, Segment, write_jsonl, read_jsonl,
)


def _seg_path(cfg, video_id: str) -> str:
    root = cfg.get_path("paths.artifacts", "artifacts")
    return os.path.join(root, "segments", f"{video_id}.jsonl")


def ingest_video(video_path: str, cfg, transcriber=None, ocr_reader=None, overwrite: bool = False) -> list[Segment]:
    """Run the full ingest for one video. `transcriber`/`ocr_reader` are passed in
    so heavy models load once across a batch run."""
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    out_seg = _seg_path(cfg, video_id)
    os.makedirs(os.path.dirname(out_seg), exist_ok=True)

    if os.path.exists(out_seg) and not overwrite:
        return [Segment(**r) for r in read_jsonl(out_seg)]

    duration = get_video_duration(video_path)

    # 1) Keyframes
    keyframes = extract_keyframes(video_path, cfg.get_path("paths.frames", "artifacts/frames"), cfg)

    # 2) Transcript (lazy: caller supplies a shared Transcriber)
    transcript: list[TranscriptChunk] = []
    if transcriber is not None:
        transcript = transcriber.transcribe(video_path)
        write_jsonl(transcript, os.path.join(cfg.get_path("paths.transcripts", "artifacts/transcripts"), f"{video_id}.jsonl"))

    # 3) OCR (optional)
    ocr: list[OCRSpan] = []
    if ocr_reader is not None and cfg.get_path("ocr.enabled", False):
        ocr = ocr_reader.read_keyframes(keyframes)
        write_jsonl(ocr, os.path.join(cfg.get_path("paths.ocr", "artifacts/ocr"), f"{video_id}.jsonl"))

    # 4) Segments
    segments = build_segments(video_id, duration, keyframes, transcript, ocr, cfg)
    write_jsonl(segments, out_seg)
    return segments
