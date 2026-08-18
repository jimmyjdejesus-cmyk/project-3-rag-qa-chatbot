"""
================================================================================
Production Hybrid RAG (Retrieval-Augmented Generation) Engine
================================================================================
Author: AI & Data Science Engineer
Architecture:
  - Multi-Format Parsing: PDF, TXT, Markdown, CSV
  - Context-Aware Overlapping Chunking with Document Metadata
  - Hybrid Search: Dense Vector Embeddings + Sparse TF-IDF Lexical Search
  - Reciprocal Rank Fusion (RRF) for Multi-Modal Ranking
  - Conversation Context Memory & Query Rewriting
  - Grounding Faithfulness Verification & Hallucination Guardrails
================================================================================
"""

import os
import re
import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai


class DocumentChunk:
    """Represents an enriched text segment with structural metadata."""
    def __init__(self, text: str, source: str, chunk_id: int, start_char: int, end_char: int):
        self.text = text
        self.source = source
        self.chunk_id = chunk_id
        self.start_char = start_char
        self.end_char = end_char
        self.word_count = len(text.split())
        self.dense_embedding = None

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "word_count": self.word_count,
            "text": self.text
        }


class HybridRAGEngine:
    def __init__(self, api_key: str = None, rrf_k: int = 60):
        """
        Args:
            api_key: Optional Gemini API Key. If absent, executes in local hybrid mode.
            rrf_k: Constant parameter for Reciprocal Rank Fusion (standard default: 60).
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.rrf_k = rrf_k
        self.chunks = []
        self.documents = {}
        self.dense_embeddings = None
        self.sparse_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.sparse_matrix = None
        self.conversation_history = []
        
        # Configure Gemini if available
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.llm = genai.GenerativeModel("gemini-1.5-flash")
                self.use_api = True
            except Exception as e:
                print(f"API Configuration Warning: {e}. Falling back to local synthesis.")
                self.use_api = False
        else:
            self.use_api = False

    # ==========================================================================
    # 1. MULTI-FORMAT DOCUMENT INGESTION
    # ==========================================================================

    def load_document(self, file_path: str) -> str:
        """Extracts plain text from PDF, TXT, MD, and CSV files."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        ext = os.path.splitext(file_path)[1].lower()
        file_name = os.path.basename(file_path)
        
        if ext in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        elif ext == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(file_path)
                content = "\n\n".join([page.extract_text() or "" for page in reader.pages])
            except ImportError:
                raise ImportError("pypdf is required for PDF parsing. Please run 'pip install pypdf'.")
        elif ext == ".csv":
            content_lines = []
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                headers = next(reader, [])
                for row in reader:
                    content_lines.append(", ".join([f"{col}: {val}" for col, val in zip(headers, row) if val]))
            content = "\n".join(content_lines)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
            
        self.documents[file_name] = content
        return content

    # ==========================================================================
    # 2. CONTEXT-AWARE SLIDING WINDOW CHUNKING
    # ==========================================================================

    def chunk_document(self, text: str, source_name: str, chunk_size: int = 500, overlap: int = 100) -> list:
        """
        Splits text into overlapping semantic segments preserving paragraph and sentence boundaries.
        """
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""
        current_start = 0
        chunk_idx = len(self.chunks) + 1
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            if len(para) > chunk_size:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                        current_chunk += (" " if current_chunk else "") + sentence
                    else:
                        if current_chunk:
                            chunks.append(DocumentChunk(
                                text=current_chunk.strip(),
                                source=source_name,
                                chunk_id=chunk_idx,
                                start_char=current_start,
                                end_char=current_start + len(current_chunk)
                            ))
                            chunk_idx += 1
                            
                        # Find nearest word boundary for clean overlap
                        overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                        space_idx = overlap_text.find(' ')
                        if space_idx > 0:
                            overlap_text = overlap_text[space_idx + 1:]
                        overlap_offset = len(current_chunk) - len(overlap_text)
                        current_chunk = (overlap_text + ' ' + sentence).strip()
                        current_start += overlap_offset
                        
                        # Enforce strict chunk_size limit if sentence itself is longer than chunk_size
                        while len(current_chunk) > chunk_size:
                            split_idx = current_chunk.rfind(' ', 0, chunk_size)
                            if split_idx == -1:
                                split_idx = chunk_size
                            
                            chunks.append(DocumentChunk(
                                text=current_chunk[:split_idx].strip(),
                                source=source_name,
                                chunk_id=chunk_idx,
                                start_char=current_start,
                                end_char=current_start + split_idx
                            ))
                            chunk_idx += 1
                            
                            overlap_text = current_chunk[:split_idx][-overlap:] if split_idx > overlap else current_chunk[:split_idx]
                            space_idx = overlap_text.find(' ')
                            if space_idx > 0:
                                overlap_text = overlap_text[space_idx + 1:]
                            overlap_offset = split_idx - len(overlap_text)
                            
                            current_start += overlap_offset
                            current_chunk = (overlap_text + current_chunk[split_idx:]).strip()
            else:
                if len(current_chunk) + len(para) + 2 <= chunk_size:
                    current_chunk += ("\n\n" if current_chunk else "") + para
                else:
                    if current_chunk:
                        chunks.append(DocumentChunk(
                            text=current_chunk.strip(),
                            source=source_name,
                            chunk_id=chunk_idx,
                            start_char=current_start,
                            end_char=current_start + len(current_chunk)
                        ))
                        chunk_idx += 1
                        
                    # Find clean word boundaries for paragraph overlap
                    overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                    space_idx = overlap_text.find(' ')
                    if space_idx > 0:
                        overlap_text = overlap_text[space_idx + 1:]
                    overlap_offset = len(current_chunk) - len(overlap_text)
                    current_chunk = overlap_text + "\n\n" + para
                    current_start += overlap_offset
                    
                    # Enforce strict chunk_size limit
                    while len(current_chunk) > chunk_size:
                        split_idx = current_chunk.rfind(' ', 0, chunk_size)
                        if split_idx == -1:
                            split_idx = chunk_size
                        
                        chunks.append(DocumentChunk(
                            text=current_chunk[:split_idx].strip(),
                            source=source_name,
                            chunk_id=chunk_idx,
                            start_char=current_start,
                            end_char=current_start + split_idx
                        ))
                        chunk_idx += 1
                        
                        overlap_text = current_chunk[:split_idx][-overlap:] if split_idx > overlap else current_chunk[:split_idx]
                        space_idx = overlap_text.find(' ')
                        if space_idx > 0:
                            overlap_text = overlap_text[space_idx + 1:]
                        overlap_offset = split_idx - len(overlap_text)
                        
                        current_start += overlap_offset
                        current_chunk = overlap_text + current_chunk[split_idx:]
                    
        if current_chunk:
            chunks.append(DocumentChunk(
                text=current_chunk.strip(),
                source=source_name,
                chunk_id=chunk_idx,
                start_char=current_start,
                end_char=current_start + len(current_chunk)
            ))
            
        self.chunks.extend(chunks)
        return chunks

    # ==========================================================================
    # 3. HYBRID INDEXING PIPELINE
    # ==========================================================================

    def build_hybrid_index(self):
        """
        Builds both Dense Vector Embeddings and Sparse TF-IDF lexical matrices.
        """
        if not self.chunks:
            raise ValueError("No text chunks available to index. Ingest and chunk documents first.")
            
        chunk_texts = [c.text for c in self.chunks]
        
        # 3.1 Build Sparse Index
        self.sparse_matrix = self.sparse_vectorizer.fit_transform(chunk_texts)
        
        # 3.2 Build Dense Index (API Embeddings or local TF-IDF vector matrix)
        if self.use_api:
            try:
                dense_list = []
                for chunk in self.chunks:
                    resp = genai.embed_content(
                        model="models/text-embedding-004",
                        content=chunk.text,
                        task_type="retrieval_document"
                    )
                    chunk.dense_embedding = resp["embedding"]
                    dense_list.append(resp["embedding"])
                self.dense_embeddings = np.array(dense_list)
            except Exception as e:
                print(f"API embedding failed ({e}), using sparse vectors as dense fallback.")
                self.dense_embeddings = self.sparse_matrix.toarray()
                self.use_api = False
        else:
            self.dense_embeddings = self.sparse_matrix.toarray()

    # ==========================================================================
    # 4. HYBRID RETRIEVAL & RECIPROCAL RANK FUSION (RRF)
    # ==========================================================================

    def hybrid_retrieve(self, query: str, top_k: int = 3) -> list:
        """
        Performs Dense and Sparse retrieval and fuses ranking via Reciprocal Rank Fusion:
          RRF_Score(d) = 1/(k + Rank_Dense) + 1/(k + Rank_TF-IDF)
        """
        if not self.chunks:
            return []
            
        num_chunks = len(self.chunks)
        
        # 4.1 Sparse Retrieval Ranking
        query_sparse = self.sparse_vectorizer.transform([query])
        sparse_scores = cosine_similarity(query_sparse, self.sparse_matrix).flatten()
        sparse_ranks = {idx: rank for rank, idx in enumerate(np.argsort(sparse_scores)[::-1], 1)}
        
        # 4.2 Dense Retrieval Ranking
        if self.use_api and self.dense_embeddings is not None:
            try:
                q_resp = genai.embed_content(
                    model="models/text-embedding-004",
                    content=query,
                    task_type="retrieval_query"
                )
                q_emb = np.array(q_resp["embedding"]).reshape(1, -1)
                dot_prod = np.dot(self.dense_embeddings, q_emb.T).squeeze()
                norm_docs = np.linalg.norm(self.dense_embeddings, axis=1)
                norm_q = np.linalg.norm(q_emb)
                dense_scores = dot_prod / (norm_docs * norm_q + 1e-9)
            except Exception:
                dense_scores = sparse_scores
        else:
            dense_scores = sparse_scores
            
        dense_ranks = {idx: rank for rank, idx in enumerate(np.argsort(dense_scores)[::-1], 1)}
        
        # 4.3 Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        for idx in range(num_chunks):
            r_dense = dense_ranks[idx]
            r_sparse = sparse_ranks[idx]
            score = (1.0 / (self.rrf_k + r_dense)) + (1.0 / (self.rrf_k + r_sparse))
            rrf_scores[idx] = score
            
        # Top-K Selection
        sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in sorted_indices:
            chunk = self.chunks[idx]
            results.append({
                "chunk": chunk,
                "rrf_score": round(float(rrf_scores[idx]), 4),
                "dense_score": round(float(dense_scores[idx]), 4),
                "sparse_score": round(float(sparse_scores[idx]), 4)
            })
        return results

    # ==========================================================================
    # 5. CONVERSATION CONTEXT & GROUNDING FAITHFULNESS
    # ==========================================================================

    def rewrite_query_with_context(self, current_query: str) -> str:
        """
        Resolves ambiguous pronouns (e.g. 'what about its revenue?') using prior dialogue turns.
        """
        if not self.conversation_history:
            return current_query
        # Use last 2 turns for richer context resolution
        context_parts = []
        for user_q, bot_a in self.conversation_history[-2:]:
            context_parts.append(user_q)
            # Extract key nouns/entities from bot response for pronoun resolution
            bot_words = [w for w in re.findall(r"\w+", bot_a) if len(w) > 4 and w.lower() not in {"the", "is", "was", "at", "which", "on", "a", "an", "and", "or", "in", "to", "for", "of", "with", "from", "by", "be", "has", "have", "it", "its", "this", "that"}]
            context_parts.extend(bot_words[:10])
        return ' '.join(context_parts) + ' ' + current_query

    def verify_grounding(self, answer: str, context: str) -> float:
        """
        Evaluates lexical citation overlap to calculate a hallucination guardrail score [0.0, 1.0].
        """
        answer_words = set(re.findall(r"\w+", answer.lower()))
        context_words = set(re.findall(r"\w+", context.lower()))
        
        stop_words = {"the", "is", "was", "at", "which", "on", "a", "an", "and", "or", "in", "to", "for", "of", "with", "from", "by", "be", "has", "have", "it", "its", "this", "that"}
        ans_keywords = answer_words - stop_words
        
        if not ans_keywords:
            return 1.0
            
        grounded_count = sum(1 for w in ans_keywords if w in context_words)
        faithfulness_score = round(grounded_count / len(ans_keywords), 2)
        return min(1.0, max(0.0, faithfulness_score))

    # ==========================================================================
    # 6. MAIN RAG EXECUTION PIPELINE
    # ==========================================================================

    def query(self, user_query: str, top_k: int = 3) -> dict:
        """
        Executes end-to-end RAG pipeline: Query Rewriting -> Hybrid Retrieval -> Context Synthesis -> Faithfulness Audit.
        """
        rewritten_q = self.rewrite_query_with_context(user_query)
        retrieved_items = self.hybrid_retrieve(rewritten_q, top_k=top_k)
        
        if not retrieved_items:
            return {
                "answer": "No relevant document excerpts found in the indexed repository.",
                "retrieved": [],
                "faithfulness": 0.0
            }
            
        context_blocks = [
            f"[SOURCE: {item['chunk'].source} | CHUNK-{item['chunk'].chunk_id}]\n{item['chunk'].text}"
            for item in retrieved_items
        ]
        full_context = "\n\n---\n\n".join(context_blocks)
        
        if self.use_api:
            prompt = (
                "You are an expert AI Analyst. Answer the user question accurately using ONLY the provided context.\n"
                "Provide precise metrics and cite source chunks where applicable.\n"
                "If the context does not contain the answer, state that clearly.\n\n"
                f"CONTEXT:\n{full_context}\n\n"
                f"QUESTION:\n{rewritten_q}\n\n"
                "ANSWER:"
            )
            try:
                response = self.llm.generate_content(prompt)
                answer_text = response.text.strip()
            except Exception as e:
                self.use_api = False
                answer_text = (
                    f"API generation failed ({e}). Fallback to retrieved synthesis.\n\n"
                    f"🔍 **Local Hybrid Retrieval (Dense + TF-IDF RRF)**\n\n"
                    f"Based on the indexed document sources:\n\n"
                )
                for idx, item in enumerate(retrieved_items, 1):
                    chunk = item["chunk"]
                    answer_text += (
                        f"**[{idx}] {chunk.source} (Chunk {chunk.chunk_id})** — *RRF Match: {item['rrf_score']}*\n"
                        f"> {chunk.text}\n\n"
                    )
        else:
            # High-quality local synthesized response
            answer_text = (
                f"🔍 **Local Hybrid Retrieval (Dense + TF-IDF RRF)**\n\n"
                f"Based on the indexed document sources:\n\n"
            )
            for idx, item in enumerate(retrieved_items, 1):
                chunk = item["chunk"]
                answer_text += (
                    f"**[{idx}] {chunk.source} (Chunk {chunk.chunk_id})** — *RRF Match: {item['rrf_score']}*\n"
                    f"> {chunk.text}\n\n"
                )
                
        # Calculate Grounding Faithfulness
        faithfulness = self.verify_grounding(answer_text, full_context)
        
        # Save to memory buffer
        self.conversation_history.append((user_query, answer_text))
        
        return {
            "answer": answer_text,
            "retrieved": retrieved_items,
            "faithfulness": faithfulness,
            "context": full_context
        }


if __name__ == "__main__":
    print("Testing Hybrid RAG Engine...")
    rag = HybridRAGEngine()
    sample_text = (
        "Projected FY25 enterprise revenue is $12.4M with a 14.5% year-over-year growth rate. "
        "The highest grossing product division is Electronics contributing $5.2M."
    )
    rag.chunk_document(sample_text, source_name="annual_report_2025.txt")
    rag.build_hybrid_index()
    res = rag.query("What is the projected revenue for FY25?")
    print(f"Answer:\n{res['answer']}")
    print(f"Faithfulness Score: {res['faithfulness']}")
