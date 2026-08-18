"""
================================================================================
Intelligent Document QA Chatbot (Hybrid RAG)
================================================================================
Framework: Streamlit & Google Gemini API
Architecture: Dense + TF-IDF Reciprocal Rank Fusion Retrieval
================================================================================
"""

import os
import sys
import tempfile
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rag_engine import HybridRAGEngine

# Streamlit Page Config
st.set_page_config(
    page_title="Hybrid RAG Document Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .source-box {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 6px;
        padding: 0.8rem;
        margin-bottom: 0.6rem;
    }
    .score-badge {
        background-color: #059669;
        color: white;
        padding: 0.15rem 0.45rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Intelligent Document QA Engine (Hybrid RAG)")
st.caption("Production-grade Retrieval-Augmented Generation system combining Dense Vector Embeddings and Sparse TF-IDF Keyword Search via Reciprocal Rank Fusion (RRF) with Grounding Faithfulness Auditing.")

# Session State Initialization
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = []
if "last_query_result" not in st.session_state:
    st.session_state.last_query_result = None

# ==============================================================================
# SIDEBAR SETUP (Credentials & Document Management)
# ==============================================================================
st.sidebar.header("⚙️ System Configuration")

api_key = st.sidebar.text_input(
    "Google Gemini API Key (Optional)",
    type="password",
    help="Enter your Gemini API Key. If omitted, the engine executes in local hybrid Dense+TF-IDF mode.",
    value=os.environ.get("GEMINI_API_KEY", "")
)

if st.session_state.rag_engine is not None and st.session_state.rag_engine.api_key != api_key:
    st.session_state.rag_engine.api_key = api_key
    if api_key:
        import google.generativeai as genai
        try:
            genai.configure(api_key=api_key)
            st.session_state.rag_engine.llm = genai.GenerativeModel("gemini-1.5-flash")
            st.session_state.rag_engine.use_api = True
        except Exception:
            st.session_state.rag_engine.use_api = False
    else:
        st.session_state.rag_engine.use_api = False

st.sidebar.markdown("---")
st.sidebar.subheader("📚 Document Knowledge Base")

# Preloaded Sample Reports Selector
sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_reports")
sample_files = os.listdir(sample_dir) if os.path.exists(sample_dir) else []

load_sample = st.sidebar.checkbox("Load Sample Annual Retail Report 2025", value=True)

uploaded_files = st.sidebar.file_uploader(
    "Upload Custom Documents (PDF, TXT, MD, CSV)",
    type=["pdf", "txt", "md", "csv"],
    accept_multiple_files=True
)

if st.sidebar.button("⚡ Ingest & Build Hybrid Index", use_container_width=True):
    with st.spinner("Parsing documents, chunking semantic windows, and building Hybrid Index..."):
        rag = HybridRAGEngine(api_key=api_key)
        indexed_names = []
        
        # Load sample report if selected
        if load_sample:
            for s_file in sample_files:
                s_path = os.path.join(sample_dir, s_file)
                text = rag.load_document(s_path)
                rag.chunk_document(text, source_name=s_file)
                indexed_names.append(f"{s_file} (Sample)")
                
        # Load user uploaded files
        if uploaded_files:
            for u_file in uploaded_files:
                suffix = os.path.splitext(u_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(u_file.read())
                    tmp_path = tmp.name
                try:
                    text = rag.load_document(tmp_path)
                    rag.documents[u_file.name] = text
                    if os.path.basename(tmp_path) in rag.documents:
                        del rag.documents[os.path.basename(tmp_path)]
                    rag.chunk_document(text, source_name=u_file.name)
                    indexed_names.append(u_file.name)
                except Exception as e:
                    st.error(f"Error reading {u_file.name}: {e}")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                        
        if rag.chunks:
            rag.build_hybrid_index()
            st.session_state.rag_engine = rag
            st.session_state.indexed_docs = indexed_names
            st.sidebar.success(f"✓ Indexed {len(indexed_names)} document(s) into {len(rag.chunks)} semantic chunks.")
        else:
            st.sidebar.error("No valid text content extracted to index.")

# Auto-initialize with sample data if first load
if st.session_state.rag_engine is None and load_sample and sample_files:
    rag = HybridRAGEngine(api_key=api_key)
    for s_file in sample_files:
        s_path = os.path.join(sample_dir, s_file)
        text = rag.load_document(s_path)
        rag.chunk_document(text, source_name=s_file)
    rag.build_hybrid_index()
    st.session_state.rag_engine = rag
    st.session_state.indexed_docs = [f"{s} (Preloaded)" for s in sample_files]

if st.session_state.indexed_docs:
    st.sidebar.info("📂 **Active Documents:**\n" + "\n".join([f"- {d}" for d in st.session_state.indexed_docs]))

# ==============================================================================
# MAIN CONSOLE: CHAT & GROUNDING INSPECTOR
# ==============================================================================
col_chat, col_sources = st.columns([1.6, 1])

with col_chat:
    st.subheader("💬 Dialogue Interface")
    
    # Prompt suggestions
    st.markdown("**Sample Prompts:**")
    pcols = st.columns(3)
    with pcols[0]:
        if st.button("📊 Summarize FY25 Revenue", use_container_width=True):
            user_input = "What was the total revenue and order volume in FY25?"
        else:
            user_input = None
    with pcols[1]:
        if st.button("💻 Electronics Category CAC", use_container_width=True):
            user_input = "What is the revenue and customer acquisition cost for Electronics?"
        else:
            user_input = user_input
    with pcols[2]:
        if st.button("⚠️ Churn Drivers & Negative Sentiment", use_container_width=True):
            user_input = "What causes churn in the At-Risk customer cohort?"
        else:
            user_input = user_input

    # Render Dialogue History
    for role, msg in st.session_state.chat_messages:
        with st.chat_message(role):
            st.markdown(msg)
            
    # Chat Input
    query_input = st.chat_input("Ask a question about your documents...") or user_input
    
    if query_input and st.session_state.rag_engine:
        # Display user question
        with st.chat_message("user"):
            st.markdown(query_input)
        st.session_state.chat_messages.append(("user", query_input))
        
        # Execute RAG query
        with st.spinner("Executing Dense + TF-IDF Hybrid Retrieval and Grounding Synthesis..."):
            result = st.session_state.rag_engine.query(query_input, top_k=3)
            st.session_state.last_query_result = result
            
        with st.chat_message("assistant"):
            st.markdown(result["answer"])
            if result.get("faithfulness") is not None:
                st.caption(f"🛡 Grounding Faithfulness Score: `{result['faithfulness']*100:.0f}%`")
                
        st.session_state.chat_messages.append(("assistant", result["answer"]))
        st.rerun()

with col_sources:
    st.subheader("🔍 Grounding Citations & Fusion Telemetry")
    st.markdown("Inspect retrieved source chunks, dense/sparse scores, and RRF rank fusion weights.")
    
    res = st.session_state.last_query_result
    if res and res.get("retrieved"):
        for i, item in enumerate(res["retrieved"], 1):
            chunk = item["chunk"]
            with st.expander(f"Citation {i}: {chunk.source} (Chunk {chunk.chunk_id})", expanded=(i == 1)):
                st.markdown(f"**RRF Combined Score:** `{item['rrf_score']}`")
                st.markdown(f"**Dense Similarity:** `{item['dense_score']}` | **Sparse TF-IDF:** `{item['sparse_score']}`")
                st.markdown(f"> *\"{chunk.text}\"*")
                st.caption(f"Word Count: {chunk.word_count} words | Char Span: [{chunk.start_char}:{chunk.end_char}]")
    else:
        st.info("Submit a question to inspect retrieved source grounding excerpts.")
