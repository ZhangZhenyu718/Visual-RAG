"""ASR via faster-whisper (CTranslate2) — int8 for low-VRAM, large-v3 on cloud.

Returns timestamped TranscriptChunk records. The model is loaded lazily and
reused across videos (load once per batch run).
"""

from __future__ import annotations

import os
from typing import Optional

from visualrag.schema import TranscriptChunk
from visualrag.utils.device import resolve_device


class Transcriber:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_size = cfg.get_path("transcribe.model", "large-v3")
        self.compute_type = cfg.get_path("transcribe.compute_type", "int8")
        self.language = cfg.get_path("transcribe.language", None)
        self.vad_filter = bool(cfg.get_path("transcribe.vad_filter", True))
        self.beam_size = int(cfg.get_path("transcribe.beam_size", 5))
        self._model = None

    def _device_args(self) -> tuple[str, str]:
        dev = resolve_device(self.cfg.get_path("device", "auto"))
        if dev == "cuda":
            return "cuda", self.compute_type
        # faster-whisper has no MPS backend -> CPU with int8 is the portable choice.
        return "cpu", "int8"

    @property
    def model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            device, compute = self._device_args()
            print(f"[asr] loading faster-whisper '{self.model_size}' on {device} ({compute})")
            self._model = WhisperModel(self.model_size, device=device, compute_type=compute)
        return self._model

    def transcribe(self, video_path: str) -> list[TranscriptChunk]:
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        segments, _info = self.model.transcribe(
            video_path,
            language=self.language,
            vad_filter=self.vad_filter,
            beam_size=self.beam_size,
        )
        chunks: list[TranscriptChunk] = []
        for seg in segments:
            text = seg.text.strip()
            if text:
                chunks.append(TranscriptChunk(
                    video_id=video_id,
                    start=round(float(seg.start), 3),
                    end=round(float(seg.end), 3),
                    text=text,
                ))
        return chunks
