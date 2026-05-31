"""Build overlapping temporal Segments from per-video ingest artifacts.

A Segment is a [start, end] sliding window (window_sec / stride_sec). Each window
gathers the keyframes whose timestamp falls inside it and the transcript/OCR text
overlapping it. These windows are the units embedded and indexed in the vector DB.
"""

from __future__ import annotations

from visualrag.schema import Keyframe, TranscriptChunk, OCRSpan, Segment


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return a_start < b_end and b_start < a_end


def build_segments(
    video_id: str,
    duration: float,
    keyframes: list[Keyframe],
    transcript: list[TranscriptChunk],
    ocr: list[OCRSpan],
    cfg,
) -> list[Segment]:
    window = float(cfg.get_path("segment.window_sec", 8.0))
    stride = float(cfg.get_path("segment.stride_sec", 4.0))
    if duration <= 0:
        # Fall back to transcript/keyframe extent if duration is unknown.
        ends = [c.end for c in transcript] + [k.timestamp for k in keyframes]
        duration = max(ends) if ends else window

    segments: list[Segment] = []
    start = 0.0
    while start < duration:
        end = min(start + window, duration)
        seg_kf = [k.path for k in keyframes if start <= k.timestamp < end]
        seg_tx = " ".join(c.text for c in transcript if _overlaps(start, end, c.start, c.end)).strip()
        seg_ocr = " ".join(o.text for o in ocr if start <= o.timestamp < end).strip()

        # Skip fully empty windows (no frame, no text) to keep the index clean.
        if seg_kf or seg_tx or seg_ocr:
            segments.append(Segment(
                segment_id=f"{video_id}::{start:.2f}-{end:.2f}",
                video_id=video_id,
                start=round(start, 3),
                end=round(end, 3),
                keyframe_paths=seg_kf,
                transcript=seg_tx,
                ocr_text=seg_ocr,
            ))
        if end >= duration:
            break
        start += stride
    return segments
