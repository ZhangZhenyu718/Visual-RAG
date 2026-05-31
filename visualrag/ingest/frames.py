"""Keyframe extraction: PySceneDetect shot boundaries + uniform sampling fallback.

Strategy (matches plan §3.1): detect scene cuts and grab a representative frame
per scene; supplement with uniform temporal sampling so long static shots still
get coverage. Capped at `max_frames_per_video`.
"""

from __future__ import annotations

import os
from typing import Optional

from visualrag.schema import Keyframe


def get_video_duration(video_path: str) -> float:
    """Duration in seconds via PyAV (no full decode)."""
    import av
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        if stream.duration is not None and stream.time_base is not None:
            return float(stream.duration * stream.time_base)
        if container.duration is not None:
            return float(container.duration / 1_000_000)  # AV_TIME_BASE
    return 0.0


def _detect_scene_times(video_path: str, cfg) -> list[float]:
    """Return scene-start timestamps (seconds) using PySceneDetect."""
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import AdaptiveDetector, ContentDetector
    except ImportError:
        return []

    detector_name = cfg.get_path("frames.detector", "adaptive")
    threshold = float(cfg.get_path("frames.threshold", 3.0))
    min_len = int(cfg.get_path("frames.min_scene_len", 15))

    video = open_video(video_path)
    sm = SceneManager()
    if detector_name == "content":
        sm.add_detector(ContentDetector(threshold=threshold * 10, min_scene_len=min_len))
    else:
        sm.add_detector(AdaptiveDetector(adaptive_threshold=threshold, min_scene_len=min_len))
    sm.detect_scenes(video)
    scenes = sm.get_scene_list()
    return [start.get_seconds() for start, _end in scenes]


def _uniform_times(duration: float, fps: float) -> list[float]:
    """Uniformly spaced timestamps at `fps` frames-per-second (e.g. 0.5 => every 2s)."""
    if duration <= 0 or fps <= 0:
        return []
    step = 1.0 / fps
    times, t = [], step / 2.0
    while t < duration:
        times.append(round(t, 3))
        t += step
    return times


def _dedupe_times(times: list[float], min_gap: float = 1.0) -> list[float]:
    times = sorted(t for t in times if t >= 0)
    out: list[float] = []
    for t in times:
        if not out or (t - out[-1]) >= min_gap:
            out.append(t)
    return out


def extract_keyframes(video_path: str, out_dir: str, cfg) -> list[Keyframe]:
    """Extract keyframes for one video, save as JPGs, return Keyframe records."""
    import av
    from PIL import Image

    video_id = os.path.splitext(os.path.basename(video_path))[0]
    frame_dir = os.path.join(out_dir, video_id)
    os.makedirs(frame_dir, exist_ok=True)

    duration = get_video_duration(video_path)
    uniform_fps = float(cfg.get_path("frames.uniform_fps", 0.5))
    max_frames = int(cfg.get_path("frames.max_frames_per_video", 64))
    resize_long = int(cfg.get_path("frames.resize_long_side", 384))

    scene_times = _detect_scene_times(video_path, cfg)
    uniform = _uniform_times(duration, uniform_fps)
    sources = {round(t, 1): "scene" for t in scene_times}
    for t in uniform:
        sources.setdefault(round(t, 1), "uniform")

    target_times = _dedupe_times(list(sources.keys()), min_gap=1.0)
    if len(target_times) > max_frames:  # keep evenly spaced subset
        idx = [round(i * (len(target_times) - 1) / (max_frames - 1)) for i in range(max_frames)]
        target_times = [target_times[i] for i in sorted(set(idx))]

    keyframes: list[Keyframe] = []
    targets = sorted(target_times)
    ti = 0
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            if ti >= len(targets):
                break
            t = float(frame.pts * stream.time_base) if frame.pts is not None else None
            if t is None:
                continue
            if t + 1e-3 >= targets[ti]:
                img = frame.to_image()
                if resize_long and max(img.size) > resize_long:
                    scale = resize_long / max(img.size)
                    img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))
                fname = f"{targets[ti]:08.2f}.jpg"
                fpath = os.path.join(frame_dir, fname)
                img.save(fpath, quality=90)
                keyframes.append(Keyframe(
                    video_id=video_id,
                    timestamp=round(targets[ti], 3),
                    path=os.path.relpath(fpath),
                    source=sources.get(round(targets[ti], 1), "uniform"),
                ))
                ti += 1
    return keyframes
