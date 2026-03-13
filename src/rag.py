"""
RAG (Retrieval-Augmented Generation) for the MIT course catalog.

1. Prepare data: load data.json, structure each course as a text chunk.
2. Generate embeddings: embed each chunk with sentence-transformers, store in FAISS.
3. At query time: embed the user message, search top-k nearest chunks.
4. Build prompt: return retrieved chunks as a string to insert into the system prompt.
"""

import json
import os
from pathlib import Path

import numpy as np

# Cache directory for saved FAISS index and chunks (under project root)
RAG_CACHE_DIR = ".rag_cache"
CACHE_INDEX_FILE = "faiss.index"
CACHE_CHUNKS_FILE = "chunks.json"
CACHE_MTIME_FILE = "data_mtime.txt"

def _cache_dir():
    return _project_root() / RAG_CACHE_DIR

# Lazy imports for optional deps (sentence_transformers, faiss)
_embedding_model = None
_faiss_index = None
_chunks = None
_index_built = False


def _project_root():
    return Path(__file__).resolve().parent.parent


def _resolve_data_path(data_path):
    path = Path(data_path)
    if not path.is_absolute():
        path = _project_root() / data_path
    return path


def _load_cached_index(data_path="data.json"):
    """
    Load FAISS index and chunks from cache if present and still valid.
    Returns (index, chunks) or (None, None) if cache miss or stale.
    """
    path = _resolve_data_path(data_path)
    if not path.exists():
        return None, None
    data_mtime = str(path.stat().st_mtime)
    cache = _cache_dir()
    index_file = cache / CACHE_INDEX_FILE
    chunks_file = cache / CACHE_CHUNKS_FILE
    mtime_file = cache / CACHE_MTIME_FILE
    if not index_file.exists() or not chunks_file.exists() or not mtime_file.exists():
        return None, None
    try:
        with open(mtime_file, "r", encoding="utf-8") as f:
            if f.read().strip() != data_mtime:
                return None, None
    except Exception:
        return None, None
    try:
        faiss = _get_faiss()
        index = faiss.read_index(str(index_file))
        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        if not chunks or index.ntotal != len(chunks):
            return None, None
        print(f"Loaded RAG index from cache ({len(chunks)} chunks)")
        return index, chunks
    except Exception as e:
        print(f"Cache load failed: {e}, rebuilding index")
        return None, None


def _save_index_to_cache(index, chunks, data_path="data.json"):
    """Save FAISS index and chunks to cache and record data.json mtime."""
    path = _resolve_data_path(data_path)
    if not path.exists():
        return
    cache = _cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    faiss = _get_faiss()
    index_file = cache / CACHE_INDEX_FILE
    chunks_file = cache / CACHE_CHUNKS_FILE
    mtime_file = cache / CACHE_MTIME_FILE
    faiss.write_index(index, str(index_file))
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=0)
    with open(mtime_file, "w", encoding="utf-8") as f:
        f.write(str(path.stat().st_mtime))
    print(f"Saved RAG index to cache ({len(chunks)} chunks)")


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            raise ImportError(
                "RAG requires sentence-transformers and faiss-cpu. "
                "Install with: pip install sentence-transformers faiss-cpu"
            )
    return _embedding_model


def _get_faiss():
    try:
        import faiss
        return faiss
    except ImportError:
        raise ImportError("RAG requires faiss-cpu. Install with: pip install faiss-cpu")


# -----------------------------------------------------------------------------
# Step 1: Prepare data – collect and structure domain info, one chunk per entity
# -----------------------------------------------------------------------------


def load_and_chunk_data(data_path="data.json"):
    """
    Load MIT course catalog from data.json and turn each course into a text chunk.

    Args:
        data_path: Path to data.json (relative to cwd or absolute).

    Returns:
        list[str]: One string per course (chunk) for embedding.
    """
    path = Path(data_path)
    if not path.is_absolute():
        # Prefer project root (parent of src/)
        project_root = Path(__file__).resolve().parent.parent
        path = project_root / data_path
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    classes = data.get("classes") or data.get("courses") or {}
    if not isinstance(classes, dict):
        return []

    chunks = []

    for course_id, course in classes.items():
        if not isinstance(course, dict):
            continue

        parts = []

        for key, value in course.items():
            if isinstance(value, list):
                value = ", ".join(map(str, value))
            elif isinstance(value, dict):
                value = json.dumps(value)

            parts.append(f"{key}: {value}")

        chunk_text = "\n".join(parts)
        chunks.append(chunk_text)

    print("================================================")
    print("First chunk:")
    print("================================================")
    print(chunks[:1])

    return chunks


def build_index(chunks, embedding_model=None):
    """
    Embed each chunk and build a FAISS index.

    Args:
        chunks: list of chunk strings.
        embedding_model: optional SentenceTransformer model; uses default if None.

    Returns:
        tuple: (faiss.IndexFlatL2 index, list of chunk strings).
    """
    if not chunks:
        return None, []

    faiss = _get_faiss()
    model = embedding_model or _get_embedding_model()
    vectors = model.encode(chunks, show_progress_bar=False)
    vectors = np.array(vectors, dtype=np.float32)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index, chunks


def ensure_index(data_path="data.json"):
    """
    Ensure the FAISS index and chunk list exist; load from cache if valid, else build once and save.
    """
    global _faiss_index, _chunks, _index_built
    if _index_built and _chunks is not None:
        return _faiss_index, _chunks
    index, chunks = _load_cached_index(data_path)
    if index is not None and chunks:
        _faiss_index, _chunks = index, chunks
        _index_built = True
        return _faiss_index, _chunks
    chunks = load_and_chunk_data(data_path)
    if not chunks:
        _index_built = True
        return None, []
    _faiss_index, _chunks = build_index(chunks)
    _save_index_to_cache(_faiss_index, _chunks, data_path)
    _index_built = True
    return _faiss_index, _chunks

def get_relevant_context(query, top_k=5, data_path="data.json"):
    """
    Convert the user message to an embedding, find top-k nearest chunks, return as string.

    Args:
        query: User's message (string).
        top_k: Number of chunks to retrieve (default 5).
        data_path: Path to data.json for (re)building index if needed.

    Returns:
        str: Retrieved chunks concatenated for insertion into the system prompt.
             Empty string if no data or index.
    """
    index, chunks = ensure_index(data_path)
    if index is None or not chunks:
        return "FAILED in get_relevant_context: Got no chunks from the index"
    model = _get_embedding_model()
    q_vec = model.encode([query], show_progress_bar=False)
    q_vec = np.array(q_vec, dtype=np.float32)
    k = min(top_k, len(chunks))
    distances, indices = index.search(q_vec, k)

    selected = []
    for i in indices[0]:
        if 0 <= i < len(chunks):
            selected.append(chunks[i])
    print(f"Selected chunks: {selected[:1]}")
    return "\n\n---\n\n".join(selected)


# -----------------------------------------------------------------------------
# Step 5 is applied in chat.py: insert retrieved chunks into system prompt
# -----------------------------------------------------------------------------
