"""Device auto-detection — keeps the pipeline portable across local Mac (mps),
local 4050 (cuda), and whichever cloud GPU platform we settle on (cuda)."""

from __future__ import annotations


def resolve_device(prefer: str = "auto") -> str:
    """Return one of 'cuda' | 'mps' | 'cpu'.

    `prefer='auto'` picks the best available; an explicit value is honored if
    actually available, otherwise it falls back with a warning.
    """
    try:
        import torch
    except ImportError:
        return "cpu"

    has_cuda = torch.cuda.is_available()
    has_mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()

    if prefer == "auto":
        if has_cuda:
            return "cuda"
        if has_mps:
            return "mps"
        return "cpu"

    if prefer == "cuda" and not has_cuda:
        print("[device] cuda requested but unavailable -> cpu")
        return "cpu"
    if prefer == "mps" and not has_mps:
        print("[device] mps requested but unavailable -> cpu")
        return "cpu"
    return prefer


def describe_device(device: str) -> str:
    try:
        import torch
        if device == "cuda" and torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            return f"cuda ({name}, {mem:.1f} GB)"
    except Exception:
        pass
    return device
