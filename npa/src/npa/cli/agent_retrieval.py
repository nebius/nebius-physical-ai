"""Compatibility shim: retrieval/grounding now lives in the shipped package.

The real implementation lives in ``npa/src/npa/agent_backend/retrieval.py``
(Blueprint Phase H: shipped importable package instead of embed). This shim
preserves the ``npa.cli.agent_retrieval`` import path for callers and tests.
"""

from __future__ import annotations

from npa.agent_backend.retrieval import (  # noqa: F401
    DEFAULT_CHUNK_CHARS,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_MIN_SCORE,
    DEFAULT_TOP_K,
    KIND_DOC,
    KIND_SKILL,
    KIND_WEB,
    Citation,
    InMemoryVectorStore,
    JsonVectorStore,
    build_lance_store,
    chunk_text,
    cosine_similarity,
    format_grounded_answer,
    grounded_reply_from_result,
    index_corpus,
    iter_corpus_documents,
    retrieve,
)

__all__ = [
    "DEFAULT_CHUNK_CHARS",
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_MIN_SCORE",
    "DEFAULT_TOP_K",
    "KIND_DOC",
    "KIND_SKILL",
    "KIND_WEB",
    "Citation",
    "InMemoryVectorStore",
    "JsonVectorStore",
    "build_lance_store",
    "chunk_text",
    "cosine_similarity",
    "format_grounded_answer",
    "grounded_reply_from_result",
    "index_corpus",
    "iter_corpus_documents",
    "retrieve",
]
