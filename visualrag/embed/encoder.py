"""CLIP/SigLIP encoder (open_clip) — shared image+text embedding space so a text
query can be matched against both visual and text segment vectors.

One interface, three backbones (RQ2 ablation): swap `embed.backbone` /
`embed.pretrained` in config between ViT-B-32, ViT-L-14, ViT-SO400M-14-SigLIP-384.
"""

from __future__ import annotations

import numpy as np

from visualrag.utils.device import resolve_device, describe_device


class CLIPEncoder:
    def __init__(self, cfg):
        self.backbone = cfg.get_path("embed.backbone", "ViT-B-32")
        self.pretrained = cfg.get_path("embed.pretrained", "laion2b_s34b_b79k")
        self.batch_size = int(cfg.get_path("embed.batch_size", 32))
        self.device = resolve_device(cfg.get_path("device", "auto"))
        self._model = None
        self._preprocess = None
        self._tokenizer = None

    def _ensure(self):
        if self._model is not None:
            return
        import open_clip
        import torch
        print(f"[embed] loading {self.backbone}/{self.pretrained} on {describe_device(self.device)}")
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.backbone, pretrained=self.pretrained
        )
        model.eval().to(self.device)
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(self.backbone)
        self._torch = torch

    @property
    def dim(self) -> int:
        self._ensure()
        d = getattr(self._model.visual, "output_dim", None)  # CLIP towers
        if d is None:
            d = getattr(self._model, "embed_dim", None)      # CustomTextCLIP
        if d is None:  # SigLIP timm towers expose neither -> probe with a dummy encode
            torch = self._torch
            with torch.no_grad():
                d = self._model.encode_text(self._tokenizer(["probe"]).to(self.device)).shape[-1]
        return int(d)

    def encode_images(self, paths: list[str]) -> np.ndarray:
        """L2-normalized image embeddings, [N, D] float32. Unreadable files -> zero rows."""
        self._ensure()
        from PIL import Image
        torch = self._torch
        if not paths:
            return np.zeros((0, self.dim), dtype=np.float32)

        out = np.zeros((len(paths), self.dim), dtype=np.float32)
        for start in range(0, len(paths), self.batch_size):
            batch = paths[start:start + self.batch_size]
            tensors, valid = [], []
            for i, p in enumerate(batch):
                try:
                    img = Image.open(p).convert("RGB")
                    tensors.append(self._preprocess(img))
                    valid.append(start + i)
                except Exception as e:
                    print(f"[embed] skip unreadable frame {p}: {e}")
            if not tensors:
                continue
            x = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                feats = self._model.encode_image(x)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            out[valid] = feats.cpu().numpy().astype(np.float32)
        return out

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """L2-normalized text embeddings, [N, D] float32.

        NOTE: CLIP's context length is 77 tokens; the tokenizer truncates longer
        text. Per-window transcripts are short so this is rarely hit, but long OCR
        dumps could be clipped — flagged for the text-modality design.
        """
        self._ensure()
        torch = self._torch
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            tokens = self._tokenizer(batch).to(self.device)
            with torch.no_grad():
                feats = self._model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            out[start:start + len(batch)] = feats.cpu().numpy().astype(np.float32)
        return out

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single natural-language query -> [D] float32."""
        return self.encode_texts([query])[0]
