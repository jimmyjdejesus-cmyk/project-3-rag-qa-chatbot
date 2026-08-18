"""
================================================================================
Unit & Integration Tests for Hybrid RAG Engine
================================================================================
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rag_engine import HybridRAGEngine, DocumentChunk


@pytest.fixture
def rag_instance(tmp_path):
    """Instantiates a test RAG engine with controlled mock documents."""
    rag = HybridRAGEngine()
    
    # Create sample text file
    sample_doc = tmp_path / "sample_doc.txt"
    sample_doc.write_text(
        "Quarterly net revenue for Q4 reached $3.8M with a customer retention rate of 82%.\n\n"
        "The newly launched AI document search assistant reduced average support ticket resolution times by 45%.\n\n"
        "Shipping delays accounted for 64% of total churn feedback recorded by the support operations team."
    )
    
    rag.load_document(str(sample_doc))
    rag.chunk_document(sample_doc.read_text(), source_name="sample_doc.txt", chunk_size=120, overlap=30)
    rag.build_hybrid_index()
    return rag


def test_chunking_metadata(rag_instance):
    """Verifies that sliding window chunking preserves IDs and character offsets."""
    assert len(rag_instance.chunks) >= 3
    for chunk in rag_instance.chunks:
        assert isinstance(chunk, DocumentChunk)
        assert chunk.source == "sample_doc.txt"
        assert chunk.word_count > 0
        assert chunk.chunk_id > 0
        assert chunk.end_char > chunk.start_char


def test_hybrid_retrieval_and_rrf(rag_instance):
    """Verifies that hybrid dense+sparse retrieval returns ranked items with valid RRF scores."""
    results = rag_instance.hybrid_retrieve("What reduced support ticket resolution times?", top_k=2)
    
    assert len(results) == 2
    for item in results:
        assert "chunk" in item
        assert "rrf_score" in item
        assert "dense_score" in item
        assert "sparse_score" in item
        assert item["rrf_score"] > 0
        
    # Top result should mention AI document search assistant
    top_chunk_text = results[0]["chunk"].text
    assert "AI document search assistant" in top_chunk_text or "ticket resolution" in top_chunk_text


def test_query_rewriting_context(rag_instance):
    """Verifies conversation history pronoun resolution and query expansion."""
    rag_instance.conversation_history = [
        ("What was the Q4 revenue?", "Revenue was $3.8M.")
    ]
    
    expanded = rag_instance.rewrite_query_with_context("What was its retention rate?")
    assert "What was the Q4 revenue?" in expanded
    assert "retention rate" in expanded


def test_grounding_faithfulness(rag_instance):
    """Verifies citation faithfulness score calculation."""
    context = "Q4 revenue was $3.8M with 82% customer retention rate."
    grounded_answer = "Revenue was $3.8M and customer retention rate was 82%."
    hallucinated_answer = "The spacecraft landed on Mars with twenty astronauts."
    
    score_grounded = rag_instance.verify_grounding(grounded_answer, context)
    score_hallucinated = rag_instance.verify_grounding(hallucinated_answer, context)
    
    assert score_grounded > 0.70
    assert score_hallucinated < 0.20


def test_csv_header_preservation(tmp_path):
    """Verifies that CSV loading includes column headers in the concatenated string."""
    rag = HybridRAGEngine()
    sample_csv = tmp_path / "data.csv"
    sample_csv.write_text("ID,Name,Role\n1,Alice,Engineer\n2,Bob,Manager")
    
    rag.load_document(str(sample_csv))
    content = rag.documents["data.csv"]
    
    assert "ID: 1" in content
    assert "Name: Alice" in content
    assert "Role: Engineer" in content
    assert "ID: 2" in content


def test_chunk_size_bounding():
    """Verifies that chunk character lengths stay within threshold strictly."""
    rag = HybridRAGEngine()
    long_para = "A " * 300 # 600 chars
    rag.chunk_document(long_para, source_name="long_doc.txt", chunk_size=150, overlap=30)
    
    for chunk in rag.chunks:
        assert len(chunk.text) <= 150
