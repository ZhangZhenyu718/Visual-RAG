"""Core data schema shared across ingest / embed / index / retrieve.

A `Segment` is the atomic unit indexed in the vector store: a short overlapping
temporal window of a video, carrying its keyframe(s), transcript text, and
optional OCR text, plus precise [start, end] timestamps for grounded retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class Keyframe:
    """A single extracted frame."""
    video_id: str
    timestamp: float          # seconds
    path: str                 # saved image path (relative to artifacts)
    source: str = "scene"     # "scene" | "uniform"


@dataclass
class TranscriptChunk:
    """A timestamped ASR span from Whisper."""
    video_id: str
    start: float
    end: float
    text: str


@dataclass
class OCRSpan:
    """On-screen text detected at a timestamp."""
    video_id: str
    timestamp: float
    text: str
    confidence: float = 0.0


@dataclass
class Segment:
    """Indexed unit: an overlapping temporal window of one video."""
    segment_id: str           # f"{video_id}::{start:.2f}-{end:.2f}"
    video_id: str
    start: float
    end: float
    keyframe_paths: list[str] = field(default_factory=list)
    transcript: str = ""
    ocr_text: str = ""

    @property
    def text(self) -> str:
        """Concatenated textual modality (transcript + OCR) for text retrieval."""
        parts = [t for t in (self.transcript, self.ocr_text) if t.strip()]
        return " ".join(parts)

    def to_dict(self) -> dict:
        return asdict(self)


def write_jsonl(records: list, path: str) -> None:
    """Dump a list of dataclasses (or dicts) as JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            d = r.to_dict() if hasattr(r, "to_dict") else (asdict(r) if hasattr(r, "__dataclass_fields__") else r)
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
