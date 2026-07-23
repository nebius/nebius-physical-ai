"""Tier-0 tests for retrieval / grounding (Blueprint Phase H).

All collaborators (``embed`` / ``store`` / ``web_search``) are injected fakes so
the suite spends zero tokens and touches no network. The deterministic embedder
is a tiny bag-of-words hasher so cosine ranking is stable and assertable.
"""

from __future__ import annotations

import math

import pytest

from npa.cli import agent_retrieval as R


def _fake_embed(dim: int = 32):
    """Deterministic hashing embedder: shared vocabulary -> high cosine."""

    def embed(texts):
        vectors = []
        for text in texts:
            vec = [0.0] * dim
            for token in str(text).lower().split():
                vec[hash(token) % dim] += 1.0
            vectors.append(vec)
        return vectors

    return embed


def _index(store, docs, embed):
    documents = [(uri, title, text) for uri, title, text in docs]
    return R.index_corpus(documents, embed=embed, store=store, source="repo")


# ── chunking + corpus discovery ──────────────────────────────────────────────


def test_chunk_text_keeps_short_doc_as_one_chunk():
    chunks = R.chunk_text("# Title\n\nA short paragraph.")
    assert len(chunks) == 1
    assert chunks[0]["title"] == "Title"


def test_chunk_text_splits_long_doc_with_overlap():
    body = "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(20))
    chunks = R.chunk_text(body, chunk_chars=400, overlap=80)
    assert len(chunks) > 1
    assert all(c["text"] for c in chunks)


def test_iter_corpus_documents_reads_markdown(tmp_path):
    (tmp_path / "a.md").write_text("# Alpha\n\nAlpha body text.", encoding="utf-8")
    (tmp_path / "skip.bin").write_text("binary-ish", encoding="utf-8")
    docs = list(R.iter_corpus_documents([str(tmp_path)]))
    uris = {uri for uri, _title, _text in docs}
    assert "a.md" in uris
    assert "skip.bin" not in uris


# ── similarity + stores ──────────────────────────────────────────────────────


def test_cosine_similarity_bounds():
    assert R.cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert R.cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert R.cosine_similarity([], [1.0]) == 0.0


def test_in_memory_store_upsert_is_idempotent():
    store = R.InMemoryVectorStore()
    store.add([{"id": "x", "text": "a", "vector": [1.0, 0.0]}])
    store.add([{"id": "x", "text": "a-updated", "vector": [1.0, 0.0]}])
    assert store.count() == 1


def test_json_vector_store_persists(tmp_path):
    path = str(tmp_path / "corpus.json")
    store = R.JsonVectorStore(path)
    store.add([{"id": "x", "text": "hello world", "vector": [1.0, 2.0], "source": "repo"}])
    reopened = R.JsonVectorStore(path)
    assert reopened.count() == 1
    assert reopened.sources() == ["repo"]


# ── indexing + retrieval ─────────────────────────────────────────────────────


def test_index_corpus_reports_chunks():
    embed = _fake_embed()
    store = R.InMemoryVectorStore()
    result = _index(
        store,
        [("docs/genesis.md", "Genesis", "Genesis is a GPU physics simulator for robotics.")],
        embed,
    )
    assert result["ok"] is True
    assert result["chunks_indexed"] >= 1
    assert store.count() >= 1


def test_retrieve_returns_ranked_citations():
    embed = _fake_embed()
    store = R.InMemoryVectorStore()
    _index(
        store,
        [
            ("docs/genesis.md", "Genesis", "Genesis GPU physics simulator for robotics training."),
            ("docs/s3.md", "Storage", "Configure S3 object storage buckets and credentials."),
        ],
        embed,
    )
    result = R.retrieve("genesis physics simulator", embed=embed, store=store, k=2, min_score=0.0)
    assert result["ok"] is True
    assert result["count"] >= 1
    top = result["citations"][0]
    assert top["uri"] == "docs/genesis.md"
    assert top["kind"] == "doc"


def test_retrieve_filters_below_min_score():
    embed = _fake_embed()
    store = R.InMemoryVectorStore()
    _index(store, [("docs/s3.md", "Storage", "S3 buckets and credentials only.")], embed)
    # A totally unrelated query should score below a high floor -> no citations.
    result = R.retrieve("quantum chromodynamics lecture", embed=embed, store=store, min_score=0.9)
    assert result["ok"] is True
    assert result["count"] == 0


def test_retrieve_empty_query_is_grounded_error():
    result = R.retrieve("", embed=_fake_embed(), store=R.InMemoryVectorStore())
    assert result["ok"] is False


def test_retrieve_embed_failure_degrades_gracefully():
    def boom(_texts):
        raise RuntimeError("embed down")

    result = R.retrieve("hi", embed=boom, store=R.InMemoryVectorStore())
    assert result["ok"] is False
    assert "embedding failed" in result["error"]


def test_retrieve_folds_in_injected_web_search():
    embed = _fake_embed()
    store = R.InMemoryVectorStore()
    _index(store, [("docs/genesis.md", "Genesis", "Genesis physics simulator.")], embed)

    def web(query):
        return [{"title": "Genesis release notes", "url": "https://example.test/g", "snippet": "genesis physics simulator update"}]

    result = R.retrieve(
        "genesis physics", embed=embed, store=store, k=5, min_score=0.0,
        web_search=web, index_web=True,
    )
    assert result["used_web"] is True
    kinds = {c["kind"] for c in result["citations"]}
    assert "web" in kinds


def test_retrieve_ignores_web_when_not_requested():
    embed = _fake_embed()
    store = R.InMemoryVectorStore()
    _index(store, [("docs/genesis.md", "Genesis", "Genesis physics simulator.")], embed)
    called = {"n": 0}

    def web(query):
        called["n"] += 1
        return [{"title": "x", "url": "y", "snippet": "z"}]

    R.retrieve("genesis", embed=embed, store=store, min_score=0.0, web_search=web, index_web=False)
    assert called["n"] == 0


# ── grounded answer formatting (0 generation tokens) ─────────────────────────


def test_format_grounded_answer_cites_sources():
    citations = [
        {"title": "Genesis", "uri": "docs/genesis.md", "kind": "doc", "score": 0.91, "snippet": "Genesis is a simulator."},
    ]
    answer = R.format_grounded_answer("what is genesis", citations)
    assert "docs/genesis.md" in answer
    assert "Genesis is a simulator." in answer
    assert "no generated content" in answer.lower()


def test_format_grounded_answer_handles_no_citations():
    answer = R.format_grounded_answer("obscure", [])
    assert "no indexed grounding" in answer.lower()


# ── /chat retrieval-fallthrough gating (pure, mirrors the embedded glue) ─────


def test_grounded_reply_from_result_grounds_above_floor():
    result = {
        "ok": True,
        "citations": [
            {"title": "Genesis", "uri": "docs/genesis.md", "kind": "doc", "score": 0.72, "snippet": "Genesis simulator."},
        ],
    }
    reply = R.grounded_reply_from_result("genesis?", result, min_score=0.35)
    assert reply is not None
    assert reply["citations"][0]["uri"] == "docs/genesis.md"
    assert "docs/genesis.md" in reply["answer"]


def test_grounded_reply_from_result_declines_below_floor():
    result = {"ok": True, "citations": [{"uri": "docs/x.md", "score": 0.20, "snippet": "weak"}]}
    assert R.grounded_reply_from_result("q", result, min_score=0.35) is None


def test_grounded_reply_from_result_declines_when_not_ok_or_empty():
    assert R.grounded_reply_from_result("q", {"ok": False, "citations": []}) is None
    assert R.grounded_reply_from_result("q", {"ok": True, "citations": []}) is None


def test_grounded_reply_from_result_rejects_nan_and_missing_score():
    nan = float("nan")
    assert R.grounded_reply_from_result("q", {"ok": True, "citations": [{"uri": "d", "score": nan, "snippet": "s"}]}) is None
    assert R.grounded_reply_from_result("q", {"ok": True, "citations": [{"uri": "d", "snippet": "s"}]}) is None


def test_index_corpus_raises_on_embed_count_mismatch():
    store = R.InMemoryVectorStore()

    def bad_embed(texts):
        return [[1.0]]  # one vector regardless of input count

    docs = [("a.md", "A", "para one\n\npara two\n\n" + "x " * 400)]
    try:
        R.index_corpus(docs, embed=bad_embed, store=store, chunk_chars=200, overlap=0)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_norm_helper_is_finite():
    # Guard: cosine of a vector with itself is 1 (within fp tolerance).
    vec = [0.1, 0.2, 0.3, 0.4]
    assert math.isclose(R.cosine_similarity(vec, vec), 1.0, rel_tol=1e-9)


def test_lance_store_scores_on_cosine_scale(tmp_path):
    # LanceDB store must return scores on the SAME 0..1 cosine scale as the pure
    # stores so the shared min_score floor behaves identically across backends.
    pytest.importorskip("lancedb")
    store = R.build_lance_store(str(tmp_path / "lance"), "corpus", dim=3)
    store.add(
        [
            {"id": "a", "text": "alpha", "vector": [1.0, 0.0, 0.0], "uri": "a.md"},
            {"id": "b", "text": "beta", "vector": [0.0, 1.0, 0.0], "uri": "b.md"},
        ]
    )
    hits = store.search([1.0, 0.0, 0.0], k=2)
    assert hits[0]["uri"] == "a.md"
    # Exact self-match ~1.0 (cosine); orthogonal ~0.0 — both within [0, 1].
    assert hits[0]["score"] == pytest.approx(1.0, abs=1e-5)
    assert all(0.0 <= h["score"] <= 1.0 for h in hits)
    orthogonal = next(h for h in hits if h["uri"] == "b.md")
    assert orthogonal["score"] < 0.5
