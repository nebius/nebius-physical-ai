"""Retrieval / grounding for the NPA agent backend (Blueprint Phase H).

Open-source replacement for the Blueprint reference agent's Pinecone + Tavily
retrieval stack: a **LanceDB**-backed vector store with **Token Factory**
embeddings, plus a provider-agnostic, injected ``web_search`` for live grounding.
Everything here is pure/deterministic given its injected collaborators so it
unit-tests at 0 tokens / no network:

- ``embed(texts) -> list[vector]`` — wraps the Token Factory embeddings endpoint
  on the VM; a deterministic fake in tests.
- ``store`` — a vector store satisfying ``add`` / ``search`` / ``count``. Backed
  by LanceDB (``build_lance_store``) or a pure-python ``JsonVectorStore`` on the
  VM; an ``InMemoryVectorStore`` in tests.
- ``web_search(query) -> list[dict]`` — an optional injected live-search callable
  (SearXNG self-hosted via npa, or a generic fetch tool). Provider-agnostic so a
  hosted search can be swapped in later without touching this module.

Retrieval answers are **extractive** — ``format_grounded_answer`` cites the
retrieved snippets with no generation call — so a retrieval turn spends only
embedding tokens, preserving the cost-discipline invariant.

Phase G: this module is *shipped* to the agent VM as an importable file (see
``npa/src/npa/agent_backend/__init__.py``); the backend imports it via
``from agent_backend.retrieval import ...``. The ``npa/src/npa/cli/agent_retrieval.py``
shim re-exports it for existing import paths/tests.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

DEFAULT_CHUNK_CHARS = 1200
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 5
# Cosine-similarity floor below which a match is treated as noise (so an
# unrelated turn does not get a spurious "grounded" citation in /chat).
DEFAULT_MIN_SCORE = 0.15

KIND_DOC = "doc"
KIND_SKILL = "skill"
KIND_WEB = "web"

_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9._:/-]")
_HEADING_RE = re.compile(r"^#{1,6}\s+(.*)$", re.MULTILINE)


@dataclass
class Citation:
    """A typed, grounded citation returned by :func:`retrieve`."""

    source: str
    title: str
    snippet: str
    score: float
    uri: str = ""
    kind: str = KIND_DOC

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "title": self.title,
            "snippet": self.snippet,
            "score": round(float(self.score), 6),
            "uri": self.uri,
            "kind": self.kind,
        }


# ── chunking + corpus discovery ──────────────────────────────────────────────


def _leading_heading(text: str) -> str:
    match = _HEADING_RE.search(text)
    return match.group(1).strip() if match else ""


def chunk_text(
    text: str,
    *,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    title: str = "",
) -> list[dict[str, Any]]:
    """Split ``text`` into overlapping windows, carrying a best-effort title.

    Chunking is on paragraph boundaries where possible so a citation snippet
    reads as a coherent unit; long paragraphs fall back to fixed windows.
    """
    body = str(text or "").strip()
    if not body:
        return []
    chunk_chars = max(200, int(chunk_chars))
    overlap = max(0, min(int(overlap), chunk_chars // 2))
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(para) > chunk_chars:
            if current:
                chunks.append(current)
                current = ""
            start = 0
            while start < len(para):
                chunks.append(para[start : start + chunk_chars])
                start += chunk_chars - overlap
            continue
        if current and len(current) + len(para) + 2 > chunk_chars:
            chunks.append(current)
            # Keep an overlap tail so a fact spanning a boundary stays retrievable.
            current = (current[-overlap:] + "\n\n" + para) if overlap else para
        else:
            current = (current + "\n\n" + para) if current else para
    if current:
        chunks.append(current)
    resolved_title = title or _leading_heading(body)
    records: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        records.append(
            {
                "index": idx,
                "text": chunk.strip(),
                "title": resolved_title or (chunk.strip().splitlines()[0][:80] if chunk.strip() else ""),
            }
        )
    return records


def iter_corpus_documents(
    roots: Sequence[str],
    *,
    extensions: Sequence[str] = (".md", ".markdown", ".txt", ".rst"),
    max_bytes: int = 500_000,
) -> Iterable[tuple[str, str, str]]:
    """Yield ``(uri, title, text)`` for every corpus file under ``roots``.

    Used to index the repo ``docs/`` + ``skills/`` trees. Binary/oversized files
    are skipped. ``uri`` is the path relative to the first existing root so
    citations do not leak absolute filesystem layout.
    """
    exts = tuple(e.lower() for e in extensions)
    for root in roots:
        base = Path(root)
        if not base.exists():
            continue
        if base.is_file():
            files = [base]
            anchor = base.parent
        else:
            files = sorted(p for p in base.rglob("*") if p.is_file())
            anchor = base
        for path in files:
            if path.suffix.lower() not in exts:
                continue
            try:
                if path.stat().st_size > max_bytes:
                    continue
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if not text.strip():
                continue
            try:
                uri = str(path.relative_to(anchor))
            except ValueError:
                uri = path.name
            yield uri, _leading_heading(text) or path.stem, text


def _classify_kind(uri: str) -> str:
    lowered = str(uri or "").lower()
    if "skill" in lowered or lowered.endswith("skill.md"):
        return KIND_SKILL
    return KIND_DOC


# ── similarity + vector stores ───────────────────────────────────────────────


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity in pure python (no numpy needed on the VM)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        fx = float(x)
        fy = float(y)
        dot += fx * fy
        na += fx * fx
        nb += fy * fy
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


class InMemoryVectorStore:
    """Pure-python cosine vector store (tests + zero-dependency fallback)."""

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []
        self._ids: set[str] = set()

    def add(self, records: Sequence[dict[str, Any]]) -> int:
        added = 0
        for record in records:
            if not isinstance(record, dict):
                continue
            rid = str(record.get("id") or "")
            if rid and rid in self._ids:
                # Idempotent upsert: replace the prior record with the same id.
                self._records = [r for r in self._records if r.get("id") != rid]
            self._records.append(dict(record))
            if rid:
                self._ids.add(rid)
            added += 1
        return added

    def search(self, vector: Sequence[float], k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        scored: list[dict[str, Any]] = []
        for record in self._records:
            score = cosine_similarity(vector, record.get("vector") or [])
            hit = {key: value for key, value in record.items() if key != "vector"}
            hit["score"] = score
            scored.append(hit)
        scored.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return scored[: max(0, int(k))]

    def count(self) -> int:
        return len(self._records)

    def sources(self) -> list[str]:
        return sorted({str(r.get("source") or "") for r in self._records if r.get("source")})


class JsonVectorStore(InMemoryVectorStore):
    """Filesystem-persisted vector store rooted at a JSON file (no bucket/secret).

    Used on the agent VM when LanceDB is unavailable. Keeps the same pure-python
    cosine search as the in-memory store but survives backend restarts.
    """

    def __init__(self, path: str) -> None:
        super().__init__()
        self._path = Path(path)
        self._load()

    def _load(self) -> None:
        try:
            raw = self._path.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError):
            return
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            return
        if isinstance(data, list):
            for record in data:
                if isinstance(record, dict):
                    self._records.append(record)
                    rid = str(record.get("id") or "")
                    if rid:
                        self._ids.add(rid)

    def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._records), encoding="utf-8")

    def add(self, records: Sequence[dict[str, Any]]) -> int:
        added = super().add(records)
        if added:
            self._flush()
        return added


class _LanceVectorStore:
    """Adapter over a LanceDB table satisfying the store protocol.

    LanceDB is the Blueprint-equivalent open-source vector store. The import is
    guarded (see :func:`build_lance_store`) so the module stays importable on a
    VM without the ``lancedb`` extra installed.
    """

    def __init__(self, table: Any, *, dim: int) -> None:
        self._table = table
        self._dim = int(dim)

    def add(self, records: Sequence[dict[str, Any]]) -> int:
        rows = []
        for record in records:
            if not isinstance(record, dict):
                continue
            vector = list(record.get("vector") or [])
            if len(vector) != self._dim:
                continue
            rows.append(
                {
                    "id": str(record.get("id") or ""),
                    "text": str(record.get("text") or ""),
                    "title": str(record.get("title") or ""),
                    "source": str(record.get("source") or ""),
                    "uri": str(record.get("uri") or ""),
                    "kind": str(record.get("kind") or KIND_DOC),
                    "vector": vector,
                }
            )
        if rows:
            self._table.add(rows)
        return len(rows)

    def search(self, vector: Sequence[float], k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        query = self._table.search(list(vector))
        # Use cosine distance so ``score`` lands on the SAME 0..1 cosine scale as
        # the pure-python stores; the shared ``min_score`` floor then behaves
        # identically across backends (LanceDB cosine ``_distance`` == 1 - cosine
        # similarity, so similarity == 1 - distance).
        try:
            query = query.distance_type("cosine")
        except Exception:  # noqa: BLE001 - older lancedb spelling
            try:
                query = query.metric("cosine")
            except Exception:  # noqa: BLE001 - fall back to the backend default
                pass
        results = query.limit(max(1, int(k))).to_list()
        hits: list[dict[str, Any]] = []
        for row in results:
            hit = {key: value for key, value in row.items() if key != "vector"}
            distance = row.get("_distance")
            if isinstance(distance, (int, float)):
                # Clamp: cosine distance is in [0, 2]; map to a [0, 1] similarity.
                hit["score"] = max(0.0, min(1.0, 1.0 - float(distance)))
            else:
                hit["score"] = 0.0
            hits.append(hit)
        return hits

    def count(self) -> int:
        try:
            return int(self._table.count_rows())
        except Exception:  # noqa: BLE001 - count is best-effort telemetry
            return 0

    def sources(self) -> list[str]:
        return []


def build_lance_store(uri: str, table_name: str, *, dim: int) -> _LanceVectorStore:
    """Build a LanceDB-backed store (guarded import; VM/live path only).

    ``uri`` is a LanceDB dataset URI (a local directory on the VM or an
    S3/AI-Cloud path) — never hardcoded here; the backend passes an
    operator/config-resolved value.
    """
    import lancedb  # local import: heavy, VM-only, not needed for pure tests
    import pyarrow as pa

    db = lancedb.connect(uri)
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("title", pa.string()),
            pa.field("source", pa.string()),
            pa.field("uri", pa.string()),
            pa.field("kind", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), int(dim))),
        ]
    )
    if table_name in db.table_names():
        table = db.open_table(table_name)
    else:
        table = db.create_table(table_name, schema=schema)
    return _LanceVectorStore(table, dim=dim)


# ── indexing + retrieval ─────────────────────────────────────────────────────


def _record_id(source: str, uri: str, index: int) -> str:
    raw = f"{source}:{uri}:{index}"
    return _SAFE_ID_RE.sub("_", raw)


def index_corpus(
    documents: Iterable[tuple[str, str, str]],
    *,
    embed: Callable[[Sequence[str]], Sequence[Sequence[float]]],
    store: Any,
    source: str = "repo",
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Chunk + embed + upsert ``documents`` into the injected ``store``.

    ``documents`` is an iterable of ``(uri, title, text)`` (e.g. from
    :func:`iter_corpus_documents`). ``embed`` is the injected embeddings callable.
    Returns ``{ok, chunks_indexed, documents, source}``.
    """
    pending: list[dict[str, Any]] = []
    docs_seen = 0
    chunks_indexed = 0

    def _flush() -> None:
        nonlocal chunks_indexed
        if not pending:
            return
        vectors = list(embed([p["text"] for p in pending]))
        if len(vectors) != len(pending):
            raise ValueError("embed returned a different number of vectors than inputs")
        for record, vector in zip(pending, vectors):
            record["vector"] = list(vector)
        chunks_indexed += store.add(pending)
        pending.clear()

    for uri, title, text in documents:
        docs_seen += 1
        kind = _classify_kind(uri)
        for chunk in chunk_text(text, chunk_chars=chunk_chars, overlap=overlap, title=title):
            pending.append(
                {
                    "id": _record_id(source, uri, chunk["index"]),
                    "text": chunk["text"],
                    "title": chunk.get("title") or title,
                    "source": source,
                    "uri": uri,
                    "kind": kind,
                }
            )
            if len(pending) >= max(1, int(batch_size)):
                _flush()
    _flush()
    return {
        "ok": True,
        "chunks_indexed": chunks_indexed,
        "documents": docs_seen,
        "source": source,
    }


def _snippet(text: str, *, limit: int = 320) -> str:
    body = re.sub(r"\s+", " ", str(text or "")).strip()
    return body if len(body) <= limit else body[: limit - 1].rstrip() + "…"


def retrieve(
    query: str,
    *,
    embed: Callable[[Sequence[str]], Sequence[Sequence[float]]],
    store: Any,
    k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
    web_search: Callable[[str], Sequence[dict[str, Any]]] | None = None,
    index_web: bool = False,
    web_limit: int = 3,
) -> dict[str, Any]:
    """Retrieve grounded citations for ``query`` from the store (+ optional web).

    Embeds the query, searches the injected ``store``, filters by ``min_score``,
    and optionally folds in provider-agnostic ``web_search`` results (ranked by
    the same embedding similarity). Returns ``{ok, query, citations, count,
    used_web}``. Never raises on a collaborator failure — degrades to whatever
    citations it could gather.
    """
    text = str(query or "").strip()
    if not text:
        return {"ok": False, "query": "", "citations": [], "count": 0, "used_web": False, "error": "empty query"}
    try:
        query_vector = list(embed([text])[0])
    except Exception as exc:  # noqa: BLE001 - surface embed failure as a grounded error
        return {
            "ok": False,
            "query": text,
            "citations": [],
            "count": 0,
            "used_web": False,
            "error": f"embedding failed: {exc}",
        }

    citations: list[Citation] = []
    try:
        hits = store.search(query_vector, max(1, int(k)))
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "query": text,
            "citations": [],
            "count": 0,
            "used_web": False,
            "error": f"store search failed: {exc}",
        }
    for hit in hits:
        score = float(hit.get("score") or 0.0)
        if score < float(min_score):
            continue
        citations.append(
            Citation(
                source=str(hit.get("source") or "repo"),
                title=str(hit.get("title") or hit.get("uri") or "untitled"),
                snippet=_snippet(hit.get("text") or ""),
                score=score,
                uri=str(hit.get("uri") or ""),
                kind=str(hit.get("kind") or KIND_DOC),
            )
        )

    used_web = False
    if index_web and web_search is not None:
        try:
            web_results = list(web_search(text) or [])
        except Exception:  # noqa: BLE001 - live search is best-effort
            web_results = []
        web_texts = [str(r.get("snippet") or r.get("content") or r.get("title") or "") for r in web_results[: max(0, int(web_limit))]]
        web_vectors: list[list[float]] = []
        if web_texts:
            try:
                web_vectors = [list(v) for v in embed(web_texts)]
            except Exception:  # noqa: BLE001
                web_vectors = []
        for idx, result in enumerate(web_results[: max(0, int(web_limit))]):
            if not isinstance(result, dict):
                continue
            score = (
                cosine_similarity(query_vector, web_vectors[idx])
                if idx < len(web_vectors)
                else 0.0
            )
            used_web = True
            citations.append(
                Citation(
                    source="web",
                    title=str(result.get("title") or result.get("url") or "web result"),
                    snippet=_snippet(result.get("snippet") or result.get("content") or ""),
                    score=score,
                    uri=str(result.get("url") or result.get("uri") or ""),
                    kind=KIND_WEB,
                )
            )

    citations.sort(key=lambda c: c.score, reverse=True)
    citations = citations[: max(1, int(k))]
    return {
        "ok": True,
        "query": text,
        "citations": [c.to_dict() for c in citations],
        "count": len(citations),
        "used_web": used_web,
    }


def format_grounded_answer(query: str, citations: Sequence[dict[str, Any]]) -> str:
    """Build an extractive, cited markdown answer with **no generation call**.

    The reply is assembled purely from the retrieved snippets, so a retrieval
    turn spends only embedding tokens (0 generation tokens) and cannot fabricate
    content beyond what was indexed.
    """
    cites = [c for c in citations if isinstance(c, dict) and c.get("snippet")]
    if not cites:
        return (
            f"No indexed grounding matched **{str(query or '').strip()}**. "
            "Index a corpus first (`POST /api/agent/retrieval/index`) or ask a "
            "grounded workbench question."
        )
    lines = [f"**Grounded answer for:** {str(query or '').strip()}", ""]
    for idx, cite in enumerate(cites, start=1):
        title = str(cite.get("title") or cite.get("uri") or "source")
        uri = str(cite.get("uri") or "")
        kind = str(cite.get("kind") or KIND_DOC)
        score = cite.get("score")
        loc = f" (`{uri}`)" if uri else ""
        score_str = f" · score `{round(float(score), 3)}`" if isinstance(score, (int, float)) else ""
        lines.append(f"{idx}. **{title}** [{kind}]{loc}{score_str}")
        lines.append(f"   > {str(cite.get('snippet') or '').strip()}")
    lines.append("")
    lines.append("_Grounded on indexed corpus — no generated content beyond the cited snippets._")
    return "\n".join(lines)


def grounded_reply_from_result(
    query: str, result: dict[str, Any], *, min_score: float = DEFAULT_MIN_SCORE
) -> dict[str, Any] | None:
    """Decide whether a :func:`retrieve` result should ground a ``/chat`` turn.

    Returns a ``{answer, citations}`` payload when retrieval succeeded and the
    top citation clears ``min_score`` (the chat confidence floor); otherwise
    ``None`` so the caller falls through to its existing path unchanged. Pure and
    deterministic, so the retrieval fallthrough *decision* is unit-tested here —
    the embedded backend glue (build store → count guard → call retrieve) just
    delegates to this helper.
    """
    if not isinstance(result, dict) or not result.get("ok"):
        return None
    citations = result.get("citations") or []
    if not citations or not isinstance(citations[0], dict):
        return None
    top = citations[0].get("score")
    # Reject missing / NaN / below-floor top scores (NaN != NaN).
    if not isinstance(top, (int, float)) or top != top or top < float(min_score):
        return None
    return {
        "answer": format_grounded_answer(query, citations),
        "citations": citations,
    }
