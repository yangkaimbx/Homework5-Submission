"""
Embeddings -- Week 4 (RAG)

Unified embedding wrapper with multi-backend support.

Backends (best 2026 picks per MTEB leaderboard):
  - sentence-transformers (open source, local CPU/GPU)
      * 'all-MiniLM-L6-v2'      -- 384d, fastest, MTEB ~56
      * 'BAAI/bge-small-en-v1.5'-- 384d, MTEB ~62, recommended starter
      * 'BAAI/bge-m3'           -- 1024d, multilingual, dense+sparse+ColBERT
      * 'Alibaba-NLP/gte-large-en-v1.5' -- 1024d, MTEB ~65
  - voyage-ai           -- API, top-tier (voyage-3-large)
  - openai              -- API ('text-embedding-3-small', '...-large')
  - cohere              -- API ('embed-v4.0' multilingual)
  - jina                -- API ('jina-embeddings-v3', long context)

The class auto-detects which backend to use based on the model_name string.

Functions / classes:
  EmbeddingModel(model_name, backend=None)  - Unified API
  cosine_similarity(a, b)                   - Numpy helper
  compare_models(text_pairs, model_names)   - Side-by-side comparison
"""

import os
import time
from typing import List, Optional, Union, Dict, Any


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors or matrices."""
    import numpy as np
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
    if b.ndim == 1:
        b = b[None, :]
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T


class EmbeddingModel:
    """
    Unified embedding model wrapper.

    Auto-detects backend from model_name, or pass backend= explicitly.
    """

    BACKEND_HINTS = {
        "openai":   ["text-embedding-3", "text-embedding-ada"],
        "cohere":   ["embed-v", "embed-english", "embed-multilingual"],
        "voyage":   ["voyage-"],
        "jina":     ["jina-embeddings"],
    }

    def __init__(self, model_name: str, backend: Optional[str] = None,
                 device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.backend = backend or self._detect_backend(model_name)
        self._client = None
        self._dim = None
        self._init_backend()

    def _detect_backend(self, name: str) -> str:
        low = name.lower()
        for backend, hints in self.BACKEND_HINTS.items():
            if any(h in low for h in hints):
                return backend
        return "sentence-transformers"

    def _init_backend(self):
        if self.backend == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError("Install: pip install sentence-transformers")
            print(f"  Loading {self.model_name} (sentence-transformers, {self.device})...")
            t0 = time.time()
            self._client = SentenceTransformer(self.model_name, device=self.device)
            self._dim = self._client.get_sentence_embedding_dimension()
            print(f"  ✓ Loaded in {time.time()-t0:.1f}s, dim={self._dim}")

        elif self.backend == "openai":
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("Install: pip install openai")
            self._client = OpenAI()  # uses OPENAI_API_KEY
            print(f"  ✓ OpenAI embedding client ready ({self.model_name})")

        elif self.backend == "voyage":
            try:
                import voyageai
            except ImportError:
                raise ImportError("Install: pip install voyageai")
            self._client = voyageai.Client()  # uses VOYAGE_API_KEY
            print(f"  ✓ Voyage embedding client ready ({self.model_name})")

        elif self.backend == "cohere":
            try:
                import cohere
            except ImportError:
                raise ImportError("Install: pip install cohere")
            self._client = cohere.ClientV2()  # uses COHERE_API_KEY
            print(f"  ✓ Cohere embedding client ready ({self.model_name})")

        elif self.backend == "jina":
            # Jina uses requests
            self._jina_key = os.getenv("JINA_API_KEY")
            print(f"  ✓ Jina embedding client ready ({self.model_name})")

        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def encode(self, texts: Union[str, List[str]], batch_size: int = 32):
        """Encode one or more texts. Returns numpy array (n, d)."""
        import numpy as np
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        if self.backend == "sentence-transformers":
            vecs = self._client.encode(
                texts, batch_size=batch_size, show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=False,
            )

        elif self.backend == "openai":
            vecs = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                r = self._client.embeddings.create(model=self.model_name, input=batch)
                vecs.extend([d.embedding for d in r.data])
            vecs = np.array(vecs, dtype=np.float32)
            self._dim = vecs.shape[1]

        elif self.backend == "voyage":
            r = self._client.embed(texts, model=self.model_name, input_type="document")
            vecs = np.array(r.embeddings, dtype=np.float32)
            self._dim = vecs.shape[1]

        elif self.backend == "cohere":
            r = self._client.embed(
                texts=texts, model=self.model_name,
                input_type="search_document", embedding_types=["float"],
            )
            vecs = np.array(r.embeddings.float, dtype=np.float32)
            self._dim = vecs.shape[1]

        elif self.backend == "jina":
            import requests
            url = "https://api.jina.ai/v1/embeddings"
            headers = {"Authorization": f"Bearer {self._jina_key}"}
            payload = {"model": self.model_name, "input": texts, "task": "retrieval.passage"}
            r = requests.post(url, json=payload, headers=headers, timeout=60)
            r.raise_for_status()
            vecs = np.array([d["embedding"] for d in r.json()["data"]], dtype=np.float32)
            self._dim = vecs.shape[1]

        return vecs[0] if single else vecs

    def encode_query(self, query: str):
        """Some backends use a different prefix/input_type for queries."""
        import numpy as np
        if self.backend == "voyage":
            r = self._client.embed([query], model=self.model_name, input_type="query")
            return np.array(r.embeddings[0], dtype=np.float32)
        if self.backend == "cohere":
            r = self._client.embed(
                texts=[query], model=self.model_name,
                input_type="search_query", embedding_types=["float"],
            )
            return np.array(r.embeddings.float[0], dtype=np.float32)
        return self.encode(query)

    @property
    def dim(self) -> int:
        if self._dim is None:
            _ = self.encode("warmup")
        return self._dim


def compare_models(
    text_pairs: List[tuple],
    model_names: List[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Compare embedding models on a list of (text_a, text_b) pairs.

    Returns a dict: {model_name: {"sims": [...], "dim": int, "encode_ms": float}}
    """
    import numpy as np
    results = {}
    for name in model_names:
        try:
            print(f"\n--- {name} ---")
            m = EmbeddingModel(name)
            sims = []
            t0 = time.time()
            for a, b in text_pairs:
                va = m.encode(a)
                vb = m.encode(b)
                sims.append(float(cosine_similarity(va, vb)[0][0]))
            elapsed = (time.time() - t0) * 1000 / max(1, len(text_pairs) * 2)
            results[name] = {
                "sims": sims,
                "dim": m.dim,
                "encode_ms_per_text": round(elapsed, 1),
            }
            for (a, b), s in zip(text_pairs, sims):
                print(f"  cos={s:+.3f} | '{a[:30]}...' vs '{b[:30]}...'")
        except Exception as e:
            print(f"  ⚠ skipped {name}: {e}")
            results[name] = {"error": str(e)}
    return results
