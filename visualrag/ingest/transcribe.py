"""ASR via faster-whisper (CTranslate2) — int8 for low-VRAM, large-v3 on cloud.

Returns timestamped TranscriptChunk records. The model is loaded lazily and
reused across videos (load once per batch run).
"""

from __future__ import annotations

import os
from typing import Optional

from visualrag.schema import TranscriptChunk
from visualrag.utils.device import resolve_device


def _register_cuda_dlls() -> None:
    """Windows: make pip-installed cuBLAS/cuDNN DLLs visible to CTranslate2
    (`pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`)."""
    if os.name != "nt":
        return
    import importlib.util
    for mod in ("nvidia.cublas", "nvidia.cudnn"):
        spec = importlib.util.find_spec(mod)
        if spec and spec.submodule_search_locations:
            bin_dir = os.path.join(list(spec.submodule_search_locations)[0], "bin")
            if os.path.isdir(bin_dir):
                os.add_dll_directory(bin_dir)
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


def _ct2_cuda_available() -> bool:
    """faster-whisper runs on CTranslate2, not torch — probe its own CUDA
    support (torch may be a CPU-only build while the GPU is still usable)."""
    try:
        import ctranslate2
        _register_cuda_dlls()
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


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
        prefer = self.cfg.get_path("device", "auto")
        if prefer in ("auto", "cuda") and _ct2_cuda_available():
            return "cuda", self.compute_type
        if resolve_device(prefer) == "cuda":
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

    @staticmethod
    def _has_audio(video_path: str) -> bool:
        import av
        with av.open(video_path) as container:
            return len(container.streams.audio) > 0

    def transcribe(self, video_path: str) -> list[TranscriptChunk]:
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        if not self._has_audio(video_path):
            print(f"[asr] {video_id}: no audio stream, skipping transcription")
            return []
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
