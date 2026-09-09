"""
================================================================================
Test Suite: FastAPI Server & Hybrid RAG REST Endpoints
================================================================================
Validates endpoint availability, document ingestion, hybrid query execution,
faithfulness auditing, and static asset delivery.
================================================================================
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

# Ensure root directory is on Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server import app, manager

client = TestClient(app)

@pytest.fixture(autouse=True)
def reset_engine():
    """Reset the engine state before each test run."""
    manager._initialize_engine()
    manager.indexed_files = []
    yield


def test_status_endpoint():
    """Verify system status reports readiness and initial unindexed state."""
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert "ready" in data
    assert "chunks_count" in data
    assert "mode" in data
    assert isinstance(data["indexed_files"], list)


def test_sample_ingest_and_query_flow():
    """Test full cycle: sample report ingestion -> hybrid query -> faithfulness auditing."""
    # Step 1: Ingest bundled sample document
    ingest_res = client.post(
        "/api/ingest-sample",
        json={"sample_name": "annual_retail_report_2025.txt"}
    )
    assert ingest_res.status_code == 200
    ingest_data = ingest_res.json()
    assert ingest_data["status"] == "success"
    assert ingest_data["chunks_added"] > 0
    assert ingest_data["total_chunks"] > 0

    # Step 2: Verify status updated with indexed file
    status_res = client.get("/api/status")
    status_data = status_res.json()
    assert status_data["ready"] is True
    assert status_data["chunks_count"] > 0
    assert "annual_retail_report_2025.txt" in status_data["indexed_files"]

    # Step 3: Execute query against ingested content
    query_res = client.post(
        "/api/query",
        json={
            "query": "What is the total revenue for FY25?",
            "top_k": 3
        }
    )
    assert query_res.status_code == 200
    query_data = query_res.json()
    assert "answer" in query_data
    assert len(query_data["answer"]) > 0
    assert "faithfulness" in query_data
    assert isinstance(query_data["faithfulness"], (int, float))
    assert len(query_data["sources"]) > 0
    assert "rrf_score" in query_data["sources"][0]


def test_static_html_serving():
    """Verify that the root endpoint serves the modern HTML frontend."""
    response = client.get("/")
    assert response.status_code == 200
    assert "NEURAL" in response.text
    assert "RAG" in response.text


def test_reset_endpoint():
    """Verify that knowledge base reset resets chunks count to zero."""
    # Pre-populate with sample report
    client.post("/api/ingest-sample", json={"sample_name": "annual_retail_report_2025.txt"})
    
    # Execute reset
    reset_res = client.post("/api/reset")
    assert reset_res.status_code == 200
    
    # Confirm empty state
    status_data = client.get("/api/status").json()
    assert status_data["chunks_count"] == 0
    assert len(status_data["indexed_files"]) == 0
