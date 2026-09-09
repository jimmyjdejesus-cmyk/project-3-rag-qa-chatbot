"""
================================================================================
Intelligent Document QA Engine (Hybrid RAG) - FastAPI Backend
================================================================================
Architecture: FastAPI + Google Gemini API + Dense + TF-IDF Reciprocal Rank Fusion
Serves high-recall RAG retrieval, faithfulness guardrails, and modern Glassmorphism UI.
================================================================================
"""

import os
import sys
import tempfile
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Ensure current directory is accessible on Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from rag_engine import HybridRAGEngine

# Initialize FastAPI application with descriptive metadata
app = FastAPI(
    title="Hybrid RAG Document Intelligence API",
    description="Enterprise Hybrid Dense + Sparse BM25 retrieval engine with Grounding Faithfulness Auditing",
    version="2.0.0"
)

# Enable CORS for local dev and cross-origin embedding
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define static web asset directory
WEB_DIR = os.path.join(CURRENT_DIR, "web")
os.makedirs(WEB_DIR, exist_ok=True)

# Global engine state holder
class EngineManager:
    """Manages the in-memory Hybrid RAG engine instance and document index."""
    def __init__(self):
        self.engine: Optional[HybridRAGEngine] = None
        self.api_key: Optional[str] = os.environ.get("GEMINI_API_KEY", "")
        self.indexed_files: List[str] = []
        self._initialize_engine()

    def _initialize_engine(self, custom_key: Optional[str] = None):
        """Instantiate or reconfigure the HybridRAGEngine with credentials."""
        key = custom_key if custom_key is not None else self.api_key
        self.engine = HybridRAGEngine(api_key=key)
        self.api_key = key

    def get_status(self) -> dict:
        """Return real-time telemetry metrics for the system HUD."""
        if not self.engine:
            return {
                "ready": False,
                "chunks_count": 0,
                "indexed_files": [],
                "has_api_key": bool(self.api_key),
                "mode": "Uninitialized"
            }
        
        chunks = len(self.engine.chunks)
        has_index = self.engine.dense_embeddings is not None or self.engine.sparse_matrix is not None
        mode = "Gemini 1.5 Flash (Hybrid API)" if self.engine.use_api else "Local Dense + TF-IDF (Offline Fallback)"
        
        return {
            "ready": has_index and chunks > 0,
            "chunks_count": chunks,
            "indexed_files": self.indexed_files,
            "has_api_key": bool(self.api_key),
            "mode": mode
        }

manager = EngineManager()

# Pydantic schemas for structured request validation
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3
    api_key: Optional[str] = None

class IngestSampleRequest(BaseModel):
    sample_name: Optional[str] = "annual_retail_report_2025.txt"
    api_key: Optional[str] = None


# ==============================================================================
# API ENDPOINTS
# ==============================================================================

@app.get("/api/status")
def get_system_status():
    """Return engine readiness, indexed chunks count, and active retrieval mode."""
    return manager.get_status()


@app.post("/api/configure")
def configure_credentials(api_key: str = Form("")):
    """Dynamically update Gemini API key credentials without restarting the server."""
    manager._initialize_engine(custom_key=api_key.strip())
    return {
        "status": "success",
        "has_api_key": bool(api_key.strip()),
        "mode": "Gemini 1.5 Flash (Hybrid API)" if api_key.strip() else "Local Dense + TF-IDF (Offline Fallback)"
    }


@app.post("/api/ingest-sample")
def ingest_sample_report(req: IngestSampleRequest):
    """Ingest bundled enterprise sample documents to enable instant testing."""
    sample_dir = os.path.join(CURRENT_DIR, "sample_reports")
    sample_path = os.path.join(sample_dir, req.sample_name)
    
    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail=f"Sample report '{req.sample_name}' not found.")

    # Re-initialize engine if requested key differs
    if req.api_key and req.api_key != manager.api_key:
        manager._initialize_engine(custom_key=req.api_key)
    elif manager.engine is None:
        manager._initialize_engine()

    try:
        # Load and chunk document text
        text = manager.engine.load_document(sample_path)
        chunks = manager.engine.chunk_document(text, source_name=req.sample_name)
        # Build multi-vector and lexical TF-IDF indexes
        manager.engine.build_hybrid_index()
        
        if req.sample_name not in manager.indexed_files:
            manager.indexed_files.append(req.sample_name)
            
        return {
            "status": "success",
            "message": f"Successfully indexed sample document '{req.sample_name}'",
            "chunks_added": len(chunks),
            "total_chunks": len(manager.engine.chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/api/ingest")
async def ingest_documents(
    files: List[UploadFile] = File(...),
    api_key: Optional[str] = Form(None)
):
    """Parse, chunk, and index user-uploaded files (.pdf, .txt, .md, .csv)."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    if api_key and api_key != manager.api_key:
        manager._initialize_engine(custom_key=api_key)
    elif manager.engine is None:
        manager._initialize_engine()

    total_new_chunks = 0
    added_files = []

    for upload in files:
        suffix = os.path.splitext(upload.filename)[1].lower()
        if suffix not in [".pdf", ".txt", ".md", ".csv"]:
            continue

        # Write uploaded content to secure temporary file for ingestion
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await upload.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            doc_text = manager.engine.load_document(tmp_path)
            chunks = manager.engine.chunk_document(doc_text, source_name=upload.filename)
            total_new_chunks += len(chunks)
            if upload.filename not in manager.indexed_files:
                manager.indexed_files.append(upload.filename)
            added_files.append(upload.filename)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    if total_new_chunks > 0:
        manager.engine.build_hybrid_index()

    return {
        "status": "success",
        "files_indexed": added_files,
        "chunks_added": total_new_chunks,
        "total_chunks": len(manager.engine.chunks)
    }


@app.post("/api/query")
def execute_hybrid_query(req: QueryRequest):
    """Execute hybrid retrieval (Dense + BM25 via RRF) and generate audited response."""
    if not manager.engine or len(manager.engine.chunks) == 0:
        raise HTTPException(
            status_code=400,
            detail="Knowledge base is empty. Please ingest a document or sample report first."
        )

    # Optional dynamic key override
    if req.api_key and req.api_key != manager.api_key:
        manager.engine.api_key = req.api_key
        manager.engine._init_gemini()

    try:
        # Run hybrid retrieval with reciprocal rank fusion and grounding check
        result = manager.engine.query(user_query=req.query, top_k=req.top_k or 3)
        
        # Serialize retrieved chunks for frontend rendering
        serialized_sources = []
        for item in result.get("retrieved", []):
            chunk_obj = item["chunk"]
            serialized_sources.append({
                "source": chunk_obj.source,
                "chunk_id": chunk_obj.chunk_id,
                "text": chunk_obj.text,
                "rrf_score": round(item.get("rrf_score", 0.0), 4),
                "dense_score": round(item.get("dense_score", 0.0), 4),
                "tfidf_score": round(item.get("tfidf_score", 0.0), 4)
            })

        return {
            "query": req.query,
            "answer": result.get("answer", ""),
            "faithfulness": round(result.get("faithfulness", 0.0), 3),
            "sources": serialized_sources,
            "mode": "Gemini 1.5 Flash" if manager.engine.use_api else "Local Dense+Sparse Fallback"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query execution error: {str(e)}")


@app.post("/api/reset")
def reset_knowledge_base():
    """Clear conversational history and reset indexed memory."""
    manager._initialize_engine()
    manager.indexed_files = []
    return {"status": "success", "message": "Knowledge base reset."}


# ==============================================================================
# STATIC FRONTEND SERVING
# ==============================================================================

@app.get("/")
def serve_index():
    """Serve modern single-page web client."""
    index_file = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return JSONResponse({"status": "API active", "docs": "/docs"})

# Mount web directory for static CSS/JS assets
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    # Preload sample report on launch so the app is instantly usable out-of-the-box
    sample_path = os.path.join(CURRENT_DIR, "sample_reports", "annual_retail_report_2025.txt")
    if os.path.exists(sample_path):
        try:
            print("🚀 Pre-loading annual_retail_report_2025.txt into hybrid index...")
            text = manager.engine.load_document(sample_path)
            manager.engine.chunk_document(text, source_name="annual_retail_report_2025.txt")
            manager.engine.build_hybrid_index()
            manager.indexed_files.append("annual_retail_report_2025.txt")
            print(f"✓ Indexed {len(manager.engine.chunks)} semantic chunks ready for queries.")
        except Exception as e:
            print(f"Warning: Failed to preload sample report: {e}")

    print("⚡ Starting Hybrid RAG Server on http://localhost:8000")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
