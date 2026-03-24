

from __future__ import annotations

import enum
import hashlib
import os
from typing import Any, Dict, Iterable, List, cast
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

# -------------------------------
# Defaults
# -------------------------------
QDRANT_MODE = os.environ.get("QDRANT_MODE", "server").lower()
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "rag_chunks")

# Fields indexed as KEYWORD (exact string match)
_KEYWORD_FIELDS = [
    "file_id",
    "paper_id",
    "source_type",
    "section_title",
    "run_id",
    "chunking.strategy",
]

# Fields indexed as INTEGER (exact int match / range)
_INTEGER_FIELDS = [
    "year",
    "chunk_id",
]

# -------------------------------
# Client singleton
# -------------------------------
_def_client: QdrantClient | None = None


def get_qdrant() -> QdrantClient:
    global _def_client
    if _def_client is not None:
        return _def_client

    timeout = int(os.environ.get("QDRANT_TIMEOUT", "60"))

    if QDRANT_MODE in ("embedded", "local", "path"):
        path = os.environ.get("QDRANT_PATH", "qdrant_data")
        _def_client = QdrantClient(path=path, timeout=timeout)
        return _def_client

    prefer_grpc = os.environ.get("QDRANT_PREFER_GRPC", "false").lower() in ("1", "true", "yes")
    grpc_port = int(os.environ.get("QDRANT_GRPC_PORT", "6334"))

    _def_client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        grpc_port=grpc_port,
        prefer_grpc=prefer_grpc,
        timeout=timeout,
    )
    return _def_client


# -------------------------------
# Collection + indices
# -------------------------------
def ensure_collection(name: str, dimension: int) -> None:
    qc = get_qdrant()
    existing = [c.name for c in qc.get_collections().collections]
    if name not in existing:
        qc.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )


def ensure_payload_indices() -> None:
    """Create keyword and integer payload indices for fast filtering."""
    qc = get_qdrant()

    for fname in _KEYWORD_FIELDS:
        try:
            qc.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name=fname,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass  # already exists

    for fname in _INTEGER_FIELDS:
        try:
            qc.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name=fname,
                field_schema=PayloadSchemaType.INTEGER,
            )
        except Exception:
            pass  # already exists


# -------------------------------
# Payload sanitation
# -------------------------------
def _sanitize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure payload values are JSON-serializable primitives."""

    def normalize(val: Any) -> Any:
        if val is None or isinstance(val, (str, float, bool)):
            return val
        if isinstance(val, int):
            return val  # preserve int — important for year, chunk_id
        if isinstance(val, enum.Enum):
            return getattr(val, "value", str(val))
        if isinstance(val, (list, tuple)):
            return [normalize(v) for v in val]
        if isinstance(val, dict):
            return {str(k): normalize(v) for k, v in val.items()}
        try:
            return str(val)
        except Exception:
            return None

    return {str(k): normalize(v) for k, v in (payload or {}).items()}


# -------------------------------
# Deterministic point ID
# -------------------------------
def make_point_id(file_id: str, chunk_index: int, text_preview: str) -> str:
    """
    Stable ID based on file + chunk position + content.
    Re-uploading the same file produces the same IDs → upsert deduplicates.
    """
    raw = f"{file_id}|{chunk_index}|{text_preview[:120]}"
    return hashlib.md5(raw.encode()).hexdigest()


# -------------------------------
# Build Qdrant filter
# -------------------------------
def build_filter(flt: Dict[str, Any] | None) -> Filter | None:
    """
    Build a Qdrant Filter from a flat dict.
    Integer fields are cast to int; everything else stays as-is.
    """
    if not flt:
        return None

    conditions = []
    for key, val in flt.items():
        if val is None:
            continue
        # Cast to int for known integer fields
        if key in _INTEGER_FIELDS:
            try:
                val = int(val)
            except (TypeError, ValueError):
                continue
        conditions.append(FieldCondition(key=str(key), match=MatchValue(value=val)))

    return Filter(must=conditions) if conditions else None


# -------------------------------
# Public API
# -------------------------------
def upsert_embeddings(vectors: List[Dict[str, Any]]) -> None:
    """
    Upsert vectors into Qdrant.

    Each item must have:
      {
        "id":       str   (optional — generated deterministically if omitted)
        "values":   List[float]
        "metadata": Dict[str, Any]
      }

    If metadata contains "file_id" and "chunk_id", a deterministic ID is
    derived automatically (making re-uploads idempotent).
    """
    qc = get_qdrant()
    if not vectors:
        return

    dim = len(vectors[0].get("values") or [])
    if dim <= 0:
        return

    ensure_collection(QDRANT_COLLECTION, dim)
    ensure_payload_indices()

    points: List[PointStruct] = []
    for idx, v in enumerate(vectors):
        vec = v.get("values")
        if not vec:
            continue

        meta = _sanitize_payload(v.get("metadata", {}))

        # Prefer caller-supplied id; fall back to deterministic id; last resort uuid
        pid = v.get("id")
        if not pid:
            file_id = str(meta.get("file_id") or "")
            chunk_id = str(meta.get("chunk_id") or idx)
            preview = str(meta.get("text") or "")[:120]
            if file_id:
                pid = make_point_id(file_id, int(chunk_id) if chunk_id.isdigit() else idx, preview)
            else:
                pid = str(uuid4())

        points.append(PointStruct(id=pid, vector=vec, payload=meta))

    if not points:
        return

    batch_size = int(os.environ.get("QDRANT_UPSERT_BATCH", "256"))
    for i in range(0, len(points), batch_size):
        qc.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points[i : i + batch_size],
            wait=True,
        )


def query_embeddings(
    embedding: List[float],
    top_k: int = 5,
    filter: Dict[str, Any] | None = None,
    min_score: float = 0.0,
) -> Dict[str, Any]:
    """
    Query Qdrant for nearest neighbours.

    Args:
        embedding:  Query vector.
        top_k:      Max results to return.
        filter:     Flat dict of metadata filters (file_id, source_type, year, …).
        min_score:  Minimum cosine similarity score (0.0 = no threshold).

    Returns:
        {"matches": [{"id": str, "score": float, "metadata": dict}, ...]}
    """
    qc = get_qdrant()
    flt = build_filter(filter)

    res = qc.query_points(
        collection_name=QDRANT_COLLECTION,
        query=embedding,
        query_filter=flt,
        limit=top_k,
        with_payload=True,
    ).points

    matches = []
    for r in cast(Iterable[Any], res):
        score = float(getattr(r, "score", 0.0) or 0.0)
        if score < min_score:
            continue
        matches.append({
            "id": str(r.id),
            "score": score,
            "metadata": getattr(r, "payload", {}) or {},
        })

    return {"matches": matches}


def delete_file_vectors(file_id: str) -> int:
    """
    Delete all vectors belonging to a file_id.
    Useful for re-indexing a file cleanly.
    Returns number of deleted points.
    """
    qc = get_qdrant()
    flt = build_filter({"file_id": file_id})
    if not flt:
        return 0

    result = qc.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=flt,
        wait=True,
    )
    return getattr(result, "operation_id", 0)
