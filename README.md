# Production Hybrid RAG Document Intelligence Engine

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Gemini API](https://img.shields.io/badge/Google%20Gemini-1.5%20Flash-4285F4.svg)](https://ai.google.dev/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-ready Retrieval-Augmented Generation (RAG) system combining Dense Vector Embeddings and Sparse TF-IDF lexical search via Reciprocal Rank Fusion (RRF) with conversational memory buffers and grounding faithfulness verification.

---

## 🏛️ RAG Sequence & Architecture

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant App as Streamlit UI
    participant Rewriter as Contextual Query Rewriter
    participant Hybrid as Hybrid Retriever (Dense + TF-IDF)
    participant RRF as Reciprocal Rank Fusion
    participant LLM as Google Gemini 1.5 Flash
    participant Guardrail as Grounding Guardrail Evaluator

    User->>App: Submits Natural Language Query
    App->>Rewriter: Query + Conversation History
    Rewriter-->>Hybrid: Context-Expanded Query
    
    par Dense Search
        Hybrid->>Hybrid: Gemini text-embedding-004 Cosine Sim
    and Sparse Search
        Hybrid->>Hybrid: TF-IDF / TF-IDF Lexical Matching
    end
    
    Hybrid->>RRF: Raw Ranked Retrieval Lists
    RRF-->>App: Top-K Fused Chunks with RRF Weights
    
    App->>LLM: Prompt (Grounded Context Blocks + User Query)
    LLM-->>Guardrail: Raw Candidate Response
    Guardrail-->>App: Audited Response + Grounding Score (0-100%)
    App-->>User: Display Response & Expandable Source Citations
```

---

## 🔬 Technical & Engineering Innovations

### 1. Hybrid Dense + Sparse Retrieval (TF-IDF + Dense)
Dense vector embeddings capture conceptual semantic intent, whereas sparse TF-IDF models excel at exact acronyms, part numbers, and discrete numerical metrics. Combining both guarantees high recall and high precision.

### 2. Reciprocal Rank Fusion (RRF)
Combines disparate scoring scales into an invariant rank-based score:
$$RRF\_Score(d \in D) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
where $k = 60$, $M = \{\text{Dense}, \text{Sparse}\}$, and $r_m(d)$ is the document rank within model $m$.

### 3. Contextual Query Rewriting & Conversation Buffer
Resolves conversational ambiguity (e.g. *"What is its growth rate?"* following *"Tell me about Electronics revenue"*) by injecting dialogue turns into the search vector query.

### 4. Grounding Faithfulness & Hallucination Guardrail
Computes lexical citation overlap between the candidate LLM response and the retrieved source text to verify output adherence.

---

## 🚀 Quickstart & Setup

### Local Execution
```bash
cd project-3-rag-qa-chatbot

# Install dependencies
pip install -r requirements.txt

# Run automated pytest suite
pytest tests/ -v

# (Optional) Export your Gemini API Key
export GEMINI_API_KEY="your-gemini-api-key"

# Launch Streamlit web interface
streamlit run app.py
```

### Docker Container Deployment
```bash
docker build -t hybrid-rag-qa .
docker run -p 8501:8501 -e GEMINI_API_KEY="your-key" hybrid-rag-qa
```

---

## 🧪 Automated Testing Suite

```bash
$ pytest tests/ -v
============================= test session starts ==============================
collected 4 items

tests/test_rag_engine.py::test_chunking_metadata PASSED                  [ 25%]
tests/test_rag_engine.py::test_hybrid_retrieval_and_rrf PASSED          [ 50%]
tests/test_rag_engine.py::test_query_rewriting_context PASSED           [ 75%]
tests/test_rag_engine.py::test_grounding_faithfulness PASSED            [100%]

============================== 4 passed in 0.89s ===============================
```
