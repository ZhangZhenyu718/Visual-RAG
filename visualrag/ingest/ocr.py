"""Optional OCR enrichment over keyframes (EasyOCR).

Disabled by default (`ocr.enabled: false`) — benchmark videos carry little
on-screen text, so this is a lower-priority modality kept modular so it doesn't
block the Phase-1 main line.
"""

from __future__ import annotations

from typing import Optional

from visualrag.schema import Keyframe, OCRSpan
from visualrag.utils.device import resolve_device


class OCRReader:
    def __init__(self, cfg):
        self.cfg = cfg
        self.langs = cfg.get_path("ocr.langs", ["en"])
        self._reader = None

    @property
    def reader(self):
        if self._reader is None:
            import easyocr
            dev = resolve_device(self.cfg.get_path("device", "auto"))
            self._reader = easyocr.Reader(self.langs, gpu=(dev == "cuda"))
        return self._reader

    def read_keyframe(self, kf: Keyframe) -> Optional[OCRSpan]:
        results = self.reader.readtext(kf.path, detail=1)
        if not results:
            return None
        texts = [t for (_box, t, conf) in results if conf >= 0.4]
        if not texts:
            return None
        avg_conf = sum(c for (_b, _t, c) in results) / len(results)
        return OCRSpan(
            video_id=kf.video_id,
            timestamp=kf.timestamp,
            text=" ".join(texts),
            confidence=round(float(avg_conf), 3),
        )

    def read_keyframes(self, keyframes: list[Keyframe]) -> list[OCRSpan]:
        spans = []
        for kf in keyframes:
            span = self.read_keyframe(kf)
            if span:
                spans.append(span)
        return spans
