"""
Vector Store -- Week 4 (RAG)

Unified interface over the most common 2026 vector stores:

  * FAISSStore   - Local, in-memory, fastest. Great for ≤10M vectors.
  * ChromaStore  - Local, persistent, simplest API. Default for prototyping.
  * QdrantStore  - Local server (Rust). Great filters, payload, scale.

All stores accept LangChain-style chunk dicts:
    {"text": "...", "metadata": {"source": "...", "chunk_id": 0, ...}}

Each implements:
    add(chunks, embeddings)       - Index chunks
    search(query_vec, k=5, filter)- Top-k similarity search
    save(path) / load(path)       - Persistence
    __len__                       - Number of vectors
"""

import os
import json
import pickle
from typing import Dict, List, Optional, Any


# ===========================================================================
# FAISS
# ===========================================================================

class FAISSStore:
    """Thin wrapper around faiss.IndexFlatIP (cosine via normalized vectors)."""

    def __init__(self, dim: int, metric: str = "cosine"):
        try:
            import faiss
        except ImportError:
            raise ImportError("Install: pip install faiss-cpu")
        self.faiss = faiss
        self.dim = dim
        self.metric = metric
        if metric == "cosine" or metric == "ip":
            self.index = faiss.IndexFlatIP(dim)
        elif metric == "l2":
            self.index = faiss.IndexFlatL2(dim)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        self.chunks: List[Dict[str, Any]] = []

    def __len__(self):
        return self.index.ntotal

    def _normalize(self, v):
        import numpy as np
        v = np.asarray(v, dtype="float32")
        if self.metric == "cosine":
            v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
        return v

    def add(self, chunks: List[Dict[str, Any]], embeddings):
        """chunks: list of {text, metadata}; embeddings: (n, dim) numpy"""
        import numpy as np
        embeddings = self._normalize(np.array(embeddings))
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        print(f"  ✓ FAISS: {len(chunks)} vectors added (total: {len(self)})")

    def search(self, query_vec, k: int = 5, filter: Optional[Dict] = None):
        import numpy as np
        q = self._normalize(np.array([query_vec])) if np.asarray(query_vec).ndim == 1 else self._normalize(np.array(query_vec))
        # Over-fetch when filtering
        fetch_k = k if filter is None else min(len(self), k * 5)
        scores, idxs = self.index.search(q, fetch_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            if filter and not all(chunk.get("metadata", {}).get(k_, None) == v
                                  for k_, v in filter.items()):
                continue
            results.append({**chunk, "score": float(score), "rank": len(results) + 1})
            if len(results) >= k:
                break
        return results

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "chunks.pkl"), "wb") as f:
            pickle.dump(
                {"chunks": self.chunks, "dim": self.dim, "metric": self.metric},
                f,
            )
        print(f"  ✓ Saved FAISS store to {path} ({len(self)} vectors)")

    @classmethod
    def load(cls, path: str) -> "FAISSStore":
        try:
            import faiss
        except ImportError:
            raise ImportError("Install: pip install faiss-cpu")
        with open(os.path.join(path, "chunks.pkl"), "rb") as f:
            data = pickle.load(f)
        store = cls(dim=data["dim"], metric=data["metric"])
        store.index = faiss.read_index(os.path.join(path, "index.faiss"))
        store.chunks = data["chunks"]
        print(f"  ✓ Loaded FAISS store from {path} ({len(store)} vectors)")
        return store


# ===========================================================================
# Chroma
# ===========================================================================

class ChromaStore:
    """
    Wrapper around the persistent Chroma client (chromadb >= 0.5).
    Handles its own embedding storage and metadata.
    """

    def __init__(self, collection_name: str = "hw4", persist_dir: str = "outputs/chroma_db"):
        try:
            import chromadb
            from chromadb.api.shared_system_client import SharedSystemClient
        except ImportError:
            raise ImportError("Install: pip install chromadb")
        os.makedirs(persist_dir, exist_ok=True)
        # Clear Chroma's cached system so a recreated persist_dir doesn't reuse
        # stale handles from a previous PersistentClient on the same path.
        SharedSystemClient.clear_system_cache()
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.persist_dir = persist_dir
        self.collection_name = collection_name

    def __len__(self):
        return self.collection.count()

    def add(self, chunks: List[Dict[str, Any]], embeddings):
        import numpy as np
        ids = [f"{c.get('metadata', {}).get('source', 'doc')}::{c.get('metadata', {}).get('chunk_id', i)}::{i + len(self)}"
               for i, c in enumerate(chunks)]
        self.collection.add(
            ids=ids,
            embeddings=np.array(embeddings).tolist(),
            documents=[c["text"] for c in chunks],
            metadatas=[c.get("metadata", {}) for c in chunks],
        )
        print(f"  ✓ Chroma: {len(chunks)} vectors added (total: {len(self)})")

    def search(self, query_vec, k: int = 5, filter: Optional[Dict] = None):
        import numpy as np
        kwargs = {
            "query_embeddings": [np.asarray(query_vec).tolist()],
            "n_results": k,
        }
        if filter:
            kwargs["where"] = filter
        r = self.collection.query(**kwargs)
        results = []
        for i in range(len(r["ids"][0])):
            results.append({
                "text": r["documents"][0][i],
                "metadata": r["metadatas"][0][i],
                "score": 1 - r["distances"][0][i],  # cosine -> similarity
                "rank": i + 1,
            })
        return results

    def save(self, path: Optional[str] = None):
        # Chroma is persistent by default; nothing to do
        print(f"  ✓ Chroma store persists at {self.persist_dir}")


# ===========================================================================
# Qdrant (local server or in-memory)
# ===========================================================================

class QdrantStore:
    """Wrapper around qdrant-client (local or in-memory mode)."""

    def __init__(self, collection_name: str = "hw4", dim: int = 384,
                 in_memory: bool = True, url: Optional[str] = None):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            raise ImportError("Install: pip install qdrant-client")
        self.QdrantClient = QdrantClient
        self.Distance = Distance
        self.VectorParams = VectorParams
        if in_memory:
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(url=url or "http://localhost:6333")
        self.collection_name = collection_name
        self.dim = dim
        self._counter = 0
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    def __len__(self):
        return self.client.count(self.collection_name).count

    def add(self, chunks: List[Dict[str, Any]], embeddings):
        from qdrant_client.models import PointStruct
        import numpy as np
        embeddings = np.asarray(embeddings)
        points = []
        for i, (chunk, vec) in enumerate(zip(chunks, embeddings)):
            points.append(PointStruct(
                id=self._counter + i,
                vector=vec.tolist(),
                payload={"text": chunk["text"], **chunk.get("metadata", {})},
            ))
        self.client.upsert(collection_name=self.collection_name, points=points)
        self._counter += len(points)
        print(f"  ✓ Qdrant: {len(points)} vectors added (total: {len(self)})")

    def search(self, query_vec, k: int = 5, filter: Optional[Dict] = None):
        import numpy as np
        # Note: Qdrant filters need a Filter object; for simplicity skipping here
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=np.asarray(query_vec).tolist(),
            limit=k,
        )
        hits = response.points
        results = []
        for i, h in enumerate(hits):
            payload = h.payload or {}
            text = payload.pop("text", "")
            results.append({
                "text": text,
                "metadata": payload,
                "score": float(h.score),
                "rank": i + 1,
            })
        return results
