"""Tests for agronomic RAG — uses the built-in knowledge base."""

from __future__ import annotations

import pytest
import numpy as np

from pulse.rag.retriever import AgronomicRAG, compute_rag_adjustment


@pytest.fixture
def rag_with_mock_index():
    """Build a RAG with a simple mock index (no sentence-transformers needed)."""
    rag = AgronomicRAG()
    # Mock the embedding model and build a simple index
    _build_mock_index(rag)
    return rag


def _build_mock_index(rag: AgronomicRAG):
    """Build a mock FAISS index using random embeddings (avoids sentence-transformers)."""
    import faiss

    n_docs = len(rag._documents)
    dim = 384  # MiniLM-L6 dimension
    rng = np.random.RandomState(42)
    embeddings = np.ascontiguousarray(rng.randn(n_docs, dim).astype(np.float32))
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = np.ascontiguousarray((embeddings / norms).astype(np.float32))

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    rag._index = index
    rag._embeddings = embeddings
    rag._built = True

    # Mock the embedding model to return random vectors
    class MockModel:
        def encode(self, texts, normalize_embeddings=False):
            r = np.random.RandomState(hash(str(texts)) % 2**31)
            embs = np.ascontiguousarray(r.randn(len(texts), dim).astype(np.float32))
            if normalize_embeddings:
                n = np.linalg.norm(embs, axis=1, keepdims=True)
                embs = np.ascontiguousarray((embs / n).astype(np.float32))
            return embs

    rag._embedding_model = MockModel()


def test_rag_default_documents():
    rag = AgronomicRAG()
    assert len(rag._documents) > 0
    # Check documents have required keys
    for doc in rag._documents:
        assert "id" in doc
        assert "text" in doc
        assert "tags" in doc


def test_rag_query_returns_results(rag_with_mock_index):
    results = rag_with_mock_index.query("tomato disease", top_k=3)
    assert len(results) == 3
    for r in results:
        assert "id" in r
        assert "text" in r
        assert "score" in r


def test_rag_query_for_treatment(rag_with_mock_index):
    results = rag_with_mock_index.query_for_treatment(
        condition="disease",
        crop="tomato",
        season="summer",
    )
    assert len(results) > 0


def test_rag_add_documents(rag_with_mock_index):
    original_count = len(rag_with_mock_index._documents)
    rag_with_mock_index.add_documents([{
        "id": "custom_doc",
        "text": "Custom agronomic advice for testing.",
        "tags": ["test"],
    }])
    assert len(rag_with_mock_index._documents) == original_count + 1
    # Index should be invalidated
    assert not rag_with_mock_index.is_built


def test_rag_save_load_roundtrip(rag_with_mock_index, tmp_path):
    rag_with_mock_index.save_index(tmp_path)
    loaded = AgronomicRAG.load_index(tmp_path)
    assert loaded.is_built
    assert len(loaded._documents) == len(rag_with_mock_index._documents)


def test_rag_load_nonexistent_returns_empty(tmp_path):
    rag = AgronomicRAG.load_index(tmp_path / "nonexistent")
    assert not rag.is_built


# --- compute_rag_adjustment tests ---

def test_adjustment_for_resistance_warning():
    docs = [{"text": "Resistance to mancozeb documented in 2022.", "score": 0.8}]
    adj = compute_rag_adjustment(docs, "targeted_fungicide")
    assert adj < 0  # should penalise


def test_adjustment_for_threshold_guidance():
    docs = [{"text": "Threshold: 2 lesions per leaf before treatment.", "score": 0.7}]
    adj = compute_rag_adjustment(docs, "targeted_spray")
    assert adj > 0  # supports informed treatment


def test_adjustment_against_no_action_for_serious():
    docs = [{"text": "Highly aggressive under wet conditions.", "score": 0.8}]
    adj = compute_rag_adjustment(docs, "no_action")
    assert adj < 0  # don't do nothing for serious condition


def test_adjustment_zero_for_low_score():
    docs = [{"text": "Some random text.", "score": 0.1}]
    adj = compute_rag_adjustment(docs, "targeted_spray")
    assert adj == 0.0


def test_adjustment_zero_for_empty():
    adj = compute_rag_adjustment([], "targeted_spray")
    assert adj == 0.0


def test_adjustment_bounded():
    docs = [{"text": "Highly aggressive resistance documented threshold organic.", "score": 0.99}]
    adj = compute_rag_adjustment(docs, "targeted_spray")
    assert -0.3 <= adj <= 0.3
