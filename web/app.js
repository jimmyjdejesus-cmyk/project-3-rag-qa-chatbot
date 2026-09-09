/**
 * ==============================================================================
 * NeuralRAG Studio - Frontend Application Logic
 * ==============================================================================
 * Connects modern Glassmorphism UI to the FastAPI Hybrid RAG endpoints.
 * Handles drag-and-drop document upload, prompt chips, and telemetry rendering.
 * ==============================================================================
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Element Selectors
  const hudStatusText = document.getElementById('hudStatusText');
  const hudChunkCount = document.getElementById('hudChunkCount');
  const hudModeText = document.getElementById('hudModeText');
  const btnPreloadSample = document.getElementById('btnPreloadSample');
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  const indexedFileList = document.getElementById('indexedFileList');
  const indexedFilesBadge = document.getElementById('indexedFilesBadge');
  const messagesStream = document.getElementById('messagesStream');
  const chatForm = document.getElementById('chatForm');
  const queryInput = document.getElementById('queryInput');
  const btnSend = document.getElementById('btnSend');
  const btnClearHistory = document.getElementById('btnClearHistory');
  const btnConfigKey = document.getElementById('btnConfigKey');
  const keyModal = document.getElementById('keyModal');
  const btnCloseModal = document.getElementById('btnCloseModal');
  const btnCancelKey = document.getElementById('btnCancelKey');
  const btnSaveKey = document.getElementById('btnSaveKey');
  const apiKeyInput = document.getElementById('apiKeyInput');
  const promptChips = document.querySelectorAll('.prompt-chip');

  // Load persisted API key from localStorage if available
  let userApiKey = localStorage.getItem('gemini_api_key') || '';
  if (userApiKey && apiKeyInput) {
    apiKeyInput.value = userApiKey;
  }

  // ============================================================================
  // TELEMETRY HUD & STATUS POLLING
  // ============================================================================

  async function updateSystemStatus() {
    try {
      const res = await fetch('/api/status');
      if (!res.ok) throw new Error('Status endpoint unavailable');
      const data = await res.json();

      // Update chunk metrics & HUD labels
      hudChunkCount.textContent = data.chunks_count || 0;
      hudModeText.textContent = data.has_api_key ? 'GEMINI 1.5 HYBRID' : 'OFFLINE FALLBACK';
      
      if (data.ready) {
        hudStatusText.textContent = 'SYSTEM READY';
        hudStatusText.style.color = 'var(--accent-emerald)';
      } else {
        hudStatusText.textContent = 'NO INDEX LOADED';
        hudStatusText.style.color = 'var(--accent-amber)';
      }

      // Render indexed documents registry
      renderIndexedFiles(data.indexed_files || []);
    } catch (err) {
      console.warn('Telemetry update failed:', err);
      hudStatusText.textContent = 'OFFLINE';
      hudStatusText.style.color = 'var(--accent-rose)';
    }
  }

  function renderIndexedFiles(files) {
    indexedFilesBadge.textContent = `${files.length} files`;
    if (!files.length) {
      indexedFileList.innerHTML = '<li class="file-empty-state">No documents indexed yet.</li>';
      return;
    }

    indexedFileList.innerHTML = files.map(file => `
      <li class="file-item">
        <span class="file-item-name" title="${file}">📄 ${file}</span>
        <span class="badge-emerald" style="font-size:0.6rem;">INDEXED</span>
      </li>
    `).join('');
  }

  // ============================================================================
  // DOCUMENT INGESTION HANDLERS
  // ============================================================================

  // 1. One-click Preload Sample Document
  btnPreloadSample.addEventListener('click', async () => {
    btnPreloadSample.disabled = true;
    btnPreloadSample.innerHTML = `<span>Ingesting Sample Report...</span>`;

    try {
      const res = await fetch('/api/ingest-sample', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sample_name: 'annual_retail_report_2025.txt',
          api_key: userApiKey || undefined
        })
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Failed to ingest sample');

      addSystemMessage(`✓ Successfully indexed sample report. Added ${data.chunks_added} semantic chunks.`);
      await updateSystemStatus();
    } catch (err) {
      alert(`Ingest error: ${err.message}`);
    } finally {
      btnPreloadSample.disabled = false;
      btnPreloadSample.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
        <span>Pre-load Sample Document</span>
      `;
    }
  });

  // 2. Drag & Drop File Uploads
  dropZone.addEventListener('click', () => fileInput.click());

  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
    });
  });

  dropZone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length) handleFileUpload(files);
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) handleFileUpload(fileInput.files);
  });

  async function handleFileUpload(fileList) {
    const formData = new FormData();
    for (const file of fileList) {
      formData.append('files', file);
    }
    if (userApiKey) {
      formData.append('api_key', userApiKey);
    }

    addSystemMessage(`Parsing and indexing ${fileList.length} uploaded document(s)...`);

    try {
      const res = await fetch('/api/ingest', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Ingestion failed');

      addSystemMessage(`✓ Indexed: ${data.files_indexed.join(', ')} (+${data.chunks_added} chunks).`);
      await updateSystemStatus();
    } catch (err) {
      alert(`Upload error: ${err.message}`);
    } finally {
      fileInput.value = '';
    }
  }

  // ============================================================================
  // CONVERSATION & QUERY EXECUTION
  // ============================================================================

  // Handle Quick-Query Prompt Chips
  promptChips.forEach(chip => {
    chip.addEventListener('click', () => {
      const q = chip.getAttribute('data-query');
      if (q) {
        queryInput.value = q;
        chatForm.dispatchEvent(new Event('submit'));
      }
    });
  });

  // Handle Query Submission
  chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (!query) return;

    // Append User Message Card
    addUserMessage(query);
    queryInput.value = '';
    btnSend.disabled = true;

    // Append Loading Thinking Card
    const loadingCardId = 'loading-' + Date.now();
    addLoadingMessage(loadingCardId);

    try {
      const res = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query,
          api_key: userApiKey || undefined,
          top_k: 3
        })
      });

      const data = await res.json();
      removeLoadingMessage(loadingCardId);

      if (!res.ok) {
        addAssistantMessage(`⚠️ **Query Error**: ${data.detail || 'Unable to retrieve answer. Please ensure knowledge base is ingested.'}`, null);
        return;
      }

      // Render Assistant Response with Telemetry & Citations
      addAssistantMessage(data.answer, data);
    } catch (err) {
      removeLoadingMessage(loadingCardId);
      addAssistantMessage(`⚠️ **Network Error**: Failed to communicate with RAG backend. (${err.message})`, null);
    } finally {
      btnSend.disabled = false;
      queryInput.focus();
    }
  });

  // Message Rendering Functions
  function addUserMessage(text) {
    const card = document.createElement('div');
    card.className = 'message-card user';
    card.innerHTML = `
      <div class="msg-avatar user">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>
      </div>
      <div class="msg-body">
        <div class="msg-header">
          <span class="msg-author">USER</span>
          <span class="msg-time">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
        </div>
        <p>${escapeHTML(text)}</p>
      </div>
    `;
    messagesStream.appendChild(card);
    scrollToBottom();
  }

  function addAssistantMessage(text, meta) {
    const card = document.createElement('div');
    card.className = 'message-card';
    
    // Determine Faithfulness Color & Label
    let faithfulnessHtml = '';
    if (meta && typeof meta.faithfulness === 'number') {
      const pct = Math.round(meta.faithfulness * 100);
      const tierClass = pct >= 75 ? 'high' : pct >= 50 ? 'medium' : 'low';
      
      faithfulnessHtml = `
        <div class="telemetry-card">
          <div class="telemetry-meta">
            <div class="faithfulness-badge ${tierClass}">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
              <span>${pct}% Grounding Faithfulness</span>
            </div>
            <span class="retrieval-mode-badge">${escapeHTML(meta.mode || 'Dense + TF-IDF RRF')}</span>
          </div>

          ${renderCitations(meta.sources || [])}
        </div>
      `;
    }

    card.innerHTML = `
      <div class="msg-avatar assistant">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>
      </div>
      <div class="msg-body">
        <div class="msg-header">
          <span class="msg-author">NEURAL RAG ASSISTANT</span>
          <span class="msg-time">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
        </div>
        <div class="msg-content">${formatAnswerText(text)}</div>
        ${faithfulnessHtml}
      </div>
    `;
    messagesStream.appendChild(card);
    scrollToBottom();
  }

  function renderCitations(sources) {
    if (!sources || !sources.length) return '';

    const items = sources.map((s, i) => `
      <div class="citation-item">
        <div class="citation-header">
          <span>SOURCE [${i + 1}]: ${escapeHTML(s.source)} (Chunk #${s.chunk_id})</span>
          <div class="citation-scores">
            <span>RRF: ${s.rrf_score}</span>
            <span>Dense: ${s.dense_score}</span>
            <span>BM25: ${s.tfidf_score}</span>
          </div>
        </div>
        <p class="citation-text">"${escapeHTML(s.text)}"</p>
      </div>
    `).join('');

    return `
      <details class="citations-details">
        <summary class="citations-summary">Inspect ${sources.length} Retrieved Knowledge Excerpts</summary>
        <div class="citations-list">${items}</div>
      </details>
    `;
  }

  function addLoadingMessage(id) {
    const card = document.createElement('div');
    card.className = 'message-card';
    card.id = id;
    card.innerHTML = `
      <div class="msg-avatar assistant">
        <span class="pulse-dot"></span>
      </div>
      <div class="msg-body">
        <div class="msg-header">
          <span class="msg-author">NEURAL RAG ASSISTANT</span>
          <span class="msg-time">QUERYING HYBRID INDEX...</span>
        </div>
        <p style="color:var(--text-muted); font-style:italic;">Executing Dense Embeddings + BM25 Reciprocal Rank Fusion...</p>
      </div>
    `;
    messagesStream.appendChild(card);
    scrollToBottom();
  }

  function removeLoadingMessage(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
  }

  function addSystemMessage(text) {
    const card = document.createElement('div');
    card.className = 'message-card system-welcome';
    card.innerHTML = `
      <div class="msg-avatar system">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
      </div>
      <div class="msg-body">
        <div class="msg-header">
          <span class="msg-author">SYSTEM NOTIFICATION</span>
          <span class="msg-time">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
        </div>
        <p>${escapeHTML(text)}</p>
      </div>
    `;
    messagesStream.appendChild(card);
    scrollToBottom();
  }

  // Clear Chat History
  btnClearHistory.addEventListener('click', async () => {
    messagesStream.innerHTML = '';
    addSystemMessage('Chat session cleared.');
    try {
      await fetch('/api/reset', { method: 'POST' });
      await updateSystemStatus();
    } catch (_) {}
  });

  // ============================================================================
  // API KEY MODAL LOGIC
  // ============================================================================

  btnConfigKey.addEventListener('click', () => keyModal.showModal());
  btnCloseModal.addEventListener('click', () => keyModal.close());
  btnCancelKey.addEventListener('click', () => keyModal.close());

  btnSaveKey.addEventListener('click', async () => {
    const newKey = apiKeyInput.value.trim();
    userApiKey = newKey;
    localStorage.setItem('gemini_api_key', newKey);
    keyModal.close();

    try {
      const formData = new FormData();
      formData.append('api_key', newKey);
      await fetch('/api/configure', { method: 'POST', body: formData });
      addSystemMessage(newKey ? '✓ Gemini API Key configured. Switching to Gemini 1.5 Flash.' : 'API Key cleared. Running in local fallback mode.');
      await updateSystemStatus();
    } catch (e) {
      console.error(e);
    }
  });

  // Helpers
  function scrollToBottom() {
    messagesStream.scrollTop = messagesStream.scrollHeight;
  }

  function escapeHTML(str) {
    return str.replace(/[&<>'"]/g, tag => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      "'": '&#39;',
      '"': '&quot;'
    }[tag] || tag));
  }

  function formatAnswerText(text) {
    // Basic Markdown bullet point & bold parser for clean layout
    let formatted = escapeHTML(text);
    formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    formatted = formatted.replace(/\n\n/g, '<br/><br/>');
    formatted = formatted.replace(/\n\* /g, '<br/>&bull; ');
    formatted = formatted.replace(/\n- /g, '<br/>&bull; ');
    return formatted;
  }

  // Initial Telemetry Kickoff
  updateSystemStatus();
});
