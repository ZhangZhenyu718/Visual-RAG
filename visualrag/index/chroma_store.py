"""ChromaDB wrapper with one persistent collection per modality.

Two collections — `segments_visual` and `segments_text` — share segment ids and
metadata (video_id, start, end). Keeping them separate lets retrieval query each
modality independently and fuse scores (late fusion), which is exactly what the
RQ2 modality ablation needs. Embeddings are L2-normalized, so cosine space turns
similarity into a dot product.
"""

from __future__ import annotations

from typing import Optional
import numpy as np

MODALITIES = ("visual", "text")


class ChromaStore:
    def __init__(self, cfg):
        import chromadb
        self.path = cfg.get_path("paths.chroma", "artifacts/chroma")
        self.client = chromadb.PersistentClient(path=self.path)
        self._cols: dict = {}

    def collection(self, modality: str):
        if modality not in MODALITIES:
            raise ValueError(f"unknown modality {modality!r}; expected {MODALITIES}")
        if modality not in self._cols:
            self._cols[modality] = self.client.get_or_create_collection(
                name=f"segments_{modality}",
                metadata={"hnsw:space": "cosine"},
            )
        return self._cols[modality]

    def upsert(self, modality: str, ids: list[str], embeddings: np.ndarray,
               metadatas: list[dict], documents: Optional[list[str]] = None) -> int:
        if len(ids) == 0:
            return 0
        col = self.collection(modality)
        col.upsert(
            ids=list(ids),
            embeddings=[e.tolist() for e in embeddings],
            metadatas=metadatas,
            documents=documents if documents is not None else None,
        )
        return len(ids)

    def query(self, modality: str, embedding: np.ndarray, k: int = 10,
              where: Optional[dict] = None) -> list[dict]:
        """Return up to k hits: [{segment_id, score, metadata, document}], score in
        [0,1] (1 - cosine distance), best first."""
        col = self.collection(modality)
        res = col.query(
            query_embeddings=[embedding.tolist()],
            n_results=k,
            where=where,
            include=["metadatas", "distances", "documents"],
        )
        hits = []
        ids = res.get("ids", [[]])[0]
        dists = res.get("distances", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        docs = res.get("documents", [[]])[0]
        for i, sid in enumerate(ids):
            hits.append({
                "segment_id": sid,
                "score": 1.0 - float(dists[i]),
                "metadata": metas[i] if i < len(metas) else {},
                "document": docs[i] if i < len(docs) else "",
            })
        return hits

    def count(self, modality: str) -> int:
        return self.collection(modality).count()
