"""
Chunking Strategies -- Week 4 (RAG)

Three approaches:
  1. Recursive character splitting (LangChain-style, the default workhorse)
  2. Semantic chunking (split where embedding similarity drops)
  3. Contextual chunking (Anthropic 2024) — prepend an LLM-generated context
     blurb to each chunk before embedding. Reduces retrieval failures by 35-49%.

Functions:
  recursive_chunk(text, chunk_size, overlap)       - Standard splitter
  sentence_chunk(text, max_tokens, overlap)        - Sentence-boundary aware
  semantic_chunk(text, embed_fn, threshold)        - Embedding-similarity split
  contextual_chunk(chunks, full_doc, llm_client)   - Anthropic contextual retrieval
  attach_metadata(chunks, source, doc_id)          - Add source metadata
"""

import re
from typing import Callable, Dict, List, Optional, Any


# ---------------------------------------------------------------------------
# 1. Recursive character splitter (no LangChain dep required)
# ---------------------------------------------------------------------------

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def recursive_chunk(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    separators: Optional[List[str]] = None,
) -> List[str]:
    """
    Recursive character text splitter.

    Walks a hierarchy of separators, splitting at the highest-level boundary
    that produces small-enough pieces. Same algorithm as LangChain's
    RecursiveCharacterTextSplitter, but no external dep.

    Args:
        text:       Source string
        chunk_size: Target chunk length (characters)
        overlap:    Characters to repeat across chunks
        separators: Hierarchy of separators (default: paragraph -> line -> sentence -> word -> char)

    Returns:
        List of chunk strings
    """
    if separators is None:
        separators = DEFAULT_SEPARATORS
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    sep = separators[0]
    next_seps = separators[1:] if len(separators) > 1 else [""]

    if sep == "":
        # Char-level fallback
        return [text[i:i + chunk_size]
                for i in range(0, len(text), max(1, chunk_size - overlap))]

    parts = text.split(sep) if sep else list(text)
    chunks, buf = [], ""
    for part in parts:
        candidate = buf + (sep if buf else "") + part
        if len(candidate) <= chunk_size:
            buf = candidate
        else:
            if buf:
                chunks.append(buf)
            if len(part) > chunk_size:
                chunks.extend(recursive_chunk(part, chunk_size, overlap, next_seps))
                buf = ""
            else:
                buf = part
    if buf:
        chunks.append(buf)

    # Apply overlap by appending the tail of chunk i to chunk i+1
    if overlap > 0 and len(chunks) > 1:
        out = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = out[-1][-overlap:]
            out.append(tail + chunks[i])
        chunks = out

    print(f"  ✓ recursive_chunk: {len(chunks)} chunks "
          f"(size~{chunk_size}, overlap={overlap})")
    return chunks


# ---------------------------------------------------------------------------
# 2. Sentence-boundary aware chunking
# ---------------------------------------------------------------------------

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def sentence_chunk(
    text: str,
    max_tokens: int = 256,
    overlap_sentences: int = 1,
) -> List[str]:
    """
    Sentence-aware chunking. Approximates tokens as words/0.75.
    """
    sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    chunks, current, current_tokens = [], [], 0
    for s in sentences:
        s_tokens = int(len(s.split()) / 0.75)
        if current_tokens + s_tokens > max_tokens and current:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_tokens = sum(int(len(c.split()) / 0.75) for c in current)
        current.append(s)
        current_tokens += s_tokens
    if current:
        chunks.append(" ".join(current))

    print(f"  ✓ sentence_chunk: {len(chunks)} chunks "
          f"(max_tokens={max_tokens}, overlap_sentences={overlap_sentences})")
    return chunks


# ---------------------------------------------------------------------------
# 3. Semantic chunking
# ---------------------------------------------------------------------------

def semantic_chunk(
    text: str,
    embed_fn: Callable[[List[str]], "np.ndarray"],
    threshold_percentile: float = 90.0,
    min_sentences: int = 2,
) -> List[str]:
    """
    Semantic chunker.

    1. Split on sentences.
    2. Embed each sentence.
    3. Compute cosine distance between consecutive sentences.
    4. Insert a chunk break wherever the distance exceeds the
       Nth percentile (default: 90th).

    Args:
        text:                  Source string
        embed_fn:              Callable that returns (n, d) numpy embeddings
        threshold_percentile:  Higher = fewer breakpoints
        min_sentences:         Minimum sentences per chunk
    """
    try:
        import numpy as np
    except ImportError:
        raise ImportError("Install: pip install numpy")

    sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    if len(sentences) <= min_sentences:
        return [text]

    emb = embed_fn(sentences)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    sims = np.sum(emb[:-1] * emb[1:], axis=1)
    distances = 1 - sims
    threshold = np.percentile(distances, threshold_percentile)
    breakpoints = [i + 1 for i, d in enumerate(distances) if d > threshold]

    chunks, prev = [], 0
    for bp in breakpoints:
        if bp - prev >= min_sentences:
            chunks.append(" ".join(sentences[prev:bp]))
            prev = bp
    chunks.append(" ".join(sentences[prev:]))

    print(f"  ✓ semantic_chunk: {len(chunks)} chunks "
          f"(threshold p{threshold_percentile:.0f}={threshold:.3f}, "
          f"breakpoints={len(breakpoints)})")
    return chunks


# ---------------------------------------------------------------------------
# 4. Contextual chunking (Anthropic, 2024)
# ---------------------------------------------------------------------------

CONTEXTUAL_PROMPT = """<document>
{whole_document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_content}
</chunk>

Please give a short, succinct context (1-2 sentences) to situate this chunk
within the overall document for the purposes of improving search retrieval
of the chunk. Answer ONLY with the succinct context, nothing else."""


def contextual_chunk(
    chunks: List[str],
    full_document: str,
    llm_client,
    model: Optional[str] = None,
    max_chunks: Optional[int] = None,
) -> List[str]:
    """
    Anthropic-style contextual retrieval:
    prepend a 1-2 sentence document-level context blurb to each chunk
    before embedding. Reduces retrieval failure rate by 35-49%.

    Note: For production, use Claude prompt caching on the document
    to amortize cost across all chunks. We use Claude 4.6 Haiku here.

    Args:
        chunks:        List of raw chunks
        full_document: The full source document
        llm_client:    Initialized LLMClient (must have Claude access)
        model:         Override model (default: claude-haiku-4-5-20251001)
        max_chunks:    Cap for cost control during demos

    Returns:
        List of contextualized chunks ("Context: ...\n\n<original chunk>")
    """
    if max_chunks:
        chunks = chunks[:max_chunks]

    contextualized = []
    print(f"  Contextualizing {len(chunks)} chunks via LLM...")
    for i, chunk in enumerate(chunks, 1):
        prompt = CONTEXTUAL_PROMPT.format(
            whole_document=full_document[:50_000],  # safety cap
            chunk_content=chunk,
        )
        resp = llm_client.generate(
            prompt=prompt,
            model=model or "claude-haiku-4-5-20251001",
            max_tokens=150,
            temperature=0.0,
        )
        ctx = resp.get("content", "").strip() if "error" not in resp else ""
        if ctx:
            contextualized.append(f"Context: {ctx}\n\n{chunk}")
        else:
            contextualized.append(chunk)
        if i % 5 == 0:
            print(f"    [{i}/{len(chunks)}] done")

    print(f"  ✓ contextual_chunk: contextualized {len(contextualized)} chunks")
    return contextualized


def attach_metadata(
    chunks: List[str],
    source: str,
    doc_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Wrap raw chunk strings into LangChain-style document dicts."""
    out = []
    for i, c in enumerate(chunks):
        meta = {"source": source, "chunk_id": i}
        if doc_id:
            meta["doc_id"] = doc_id
        if extra:
            meta.update(extra)
        out.append({"text": c, "metadata": meta})
    return out
