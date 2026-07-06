"""
rag.py – Real Retrieval-Augmented Generation
Uses sentence-transformers to embed document chunks,
stores them in a FAISS index, and retrieves top-k chunks
by cosine similarity for any query.
"""

import os
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ── Constants ────────────────────────────────────────────────────────────────
DOCS_DIR = Path(__file__).parent.parent / "docs"
INDEX_PATH = Path(__file__).parent / "faiss_index.pkl"
CHUNK_SIZE = 300          # characters per chunk
CHUNK_OVERLAP = 50        # overlap between consecutive chunks
TOP_K = 4                 # number of chunks to retrieve
EMBED_MODEL = "all-MiniLM-L6-v2"   # fast, high-quality, 384-dim


# ── Text Chunking ─────────────────────────────────────────────────────────────
def chunk_text(text: str, source: str) -> List[dict]:
    """Split text into overlapping chunks with source metadata."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({"text": chunk, "source": source})
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ── Index Building ────────────────────────────────────────────────────────────
def build_index(force_rebuild: bool = False) -> Tuple[faiss.Index, List[dict], SentenceTransformer]:
    """
    Build (or load cached) FAISS index from docs/*.txt files.
    Returns (faiss_index, chunks_list, embedder).
    """
    if INDEX_PATH.exists() and not force_rebuild:
        with open(INDEX_PATH, "rb") as f:
            saved = pickle.load(f)
        embedder = SentenceTransformer(EMBED_MODEL)
        return saved["index"], saved["chunks"], embedder

    print("[RAG] Building FAISS index from docs...")
    embedder = SentenceTransformer(EMBED_MODEL)

    all_chunks: List[dict] = []
    for txt_file in sorted(DOCS_DIR.glob("*.txt")):
        text = txt_file.read_text(encoding="utf-8")
        chunks = chunk_text(text, source=txt_file.name)
        all_chunks.extend(chunks)
        print(f"  Loaded {txt_file.name}: {len(chunks)} chunks")

    if not all_chunks:
        raise FileNotFoundError(f"No .txt files found in {DOCS_DIR}")

    texts = [c["text"] for c in all_chunks]
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype(np.float32)

    # L2-normalise → inner product == cosine similarity
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner Product (cosine after normalisation)
    index.add(embeddings)

    with open(INDEX_PATH, "wb") as f:
        pickle.dump({"index": index, "chunks": all_chunks}, f)

    print(f"[RAG] Index built: {len(all_chunks)} chunks, dim={dim}")
    return index, all_chunks, embedder


# ── Singleton cache so the index is loaded once per process ───────────────────
_cache: dict = {}

def _get_rag():
    if not _cache:
        index, chunks, embedder = build_index()
        _cache["index"] = index
        _cache["chunks"] = chunks
        _cache["embedder"] = embedder
    return _cache["index"], _cache["chunks"], _cache["embedder"]


# ── Public API ────────────────────────────────────────────────────────────────
def retrieve_docs(query: str, top_k: int = TOP_K) -> str:
    """
    Retrieve the most relevant document chunks for `query`.
    Returns a formatted string with source labels and content,
    ready to be injected into an LLM prompt as context.
    """
    index, chunks, embedder = _get_rag()

    query_vec = embedder.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(query_vec)

    scores, indices = index.search(query_vec, top_k)

    results = []
    seen_sources = set()
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx]
        source_label = chunk["source"].replace("_", " ").replace(".txt", "").title()
        results.append(
            f"[Source: {source_label} | Relevance: {score:.2f}]\n{chunk['text']}"
        )
        seen_sources.add(chunk["source"])

    if not results:
        return "No relevant documentation found."

    return "\n\n---\n\n".join(results)


def rebuild_index():
    """Force a full rebuild of the FAISS index (call after updating docs)."""
    _cache.clear()
    if INDEX_PATH.exists():
        INDEX_PATH.unlink()
    build_index(force_rebuild=True)
    _get_rag()  # warm cache
