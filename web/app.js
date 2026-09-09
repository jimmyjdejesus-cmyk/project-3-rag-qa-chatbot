/**
 * ==============================================================================
 * Document Intelligence - Client Application Logic
 * ==============================================================================
 * Clean, consumer-friendly state management, API querying, and document handling.
 * ==============================================================================
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const hudStatusText = document.getElementById('hudStatusText');
  const hudChunkCount = document.getElementById('hudChunkCount');
  const statusDot = document.getElementById('statusDot');
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
  const suggestionPills = document.querySelectorAll('.suggestion-pill');

  let userApiKey = localStorage.getItem('gemini_api_key') || '';
  if (userApiKey && apiKeyInput) {
    apiKeyInput.value = userApiKey;
  }

  // ============================================================================
  // STATUS & TELEMETRY
  // ============================================================================

  async function updateSystemStatus() {
    try {
      const res = await fetch('/api/status');
      if (!res.ok) throw new Error('Status unavailable');
      const data = await res.json();

      hudChunkCount.textContent = data.chunks_count || 0;
      
      if (data.ready) {
        hudStatusText.textContent = data.has_api_key ? 'Ready (Gemini 1.5)' : 'Ready (Offline mode)';
        if (statusDot) statusDot.className = 'status-dot';
      } else {
        hudStatusText.textContent = 'No documents loaded';
        if (statusDot) statusDot.className = 'status-dot idle';
      }

      renderIndexedFiles(data.indexed_files || []);
    } catch (err) {
      hudStatusText.textContent = 'Offline';
      if (statusDot) statusDot.className = 'status-dot idle';
    }
  }

  function renderIndexedFiles(files) {
    indexedFilesBadge.textContent = files.length;
    if (!files.length) {
      indexedFileList.innerHTML = '<li class="doc-empty">No files loaded yet.</li>';
      return;
    }

    indexedFileList.innerHTML = files.map(file => `
      <li class="doc-item">
        <span class="doc-name" title="${file}">📄 ${escapeHTML(file)}</span>
        <span class="pill-success" style="font-size: 0.65rem;">Loaded</span>
      </li>
    `).join('');
  }

  // ============================================================================
  // DOCUMENT INGESTION
  // ============================================================================

  // Preload Sample Document
  btnPreloadSample.addEventListener('click', async () => {
    btnPreloadSample.disabled = true;
    btnPreloadSample.innerHTML = `<span>Loading Report...</span>`;

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
      if (!res.ok) throw new Error(data.detail || 'Ingestion failed');

      addSystemMessage(`Annual Retail Report (2025) has been loaded with ${data.chunks_added} searchable passages.`);
      await updateSystemStatus();
    } catch (err) {
      alert(`Could not load sample: ${err.message}`);
    } finally {
      btnPreloadSample.disabled = false;
      btnPreloadSample.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
        <span>Load Sample Document</span>
      `;
    }
  });

  // Drag and Drop File Ingest
  dropZone.addEventListener('click', () => fileInput.click());

  ['dragenter', 'dragover'].forEach(name => {
    dropZone.addEventListener(name, (e) => {
      e.preventDefault();
      dropZone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(name => {
    dropZone.addEventListener(name, (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
    });
  });

  dropZone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length) handleFiles(files);
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) handleFiles(fileInput.files);
  });

  async function handleFiles(fileList) {
    const formData = new FormData();
    for (const f of fileList) formData.append('files', f);
    if (userApiKey) formData.append('api_key', userApiKey);

    addSystemMessage(`Processing ${fileList.length} document(s)...`);

    try {
      const res = await fetch('/api/ingest', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Upload failed');

      addSystemMessage(`Successfully added: ${data.files_indexed.join(', ')} (${data.chunks_added} passages parsed).`);
      await updateSystemStatus();
    } catch (err) {
      alert(`Upload error: ${err.message}`);
    } finally {
      fileInput.value = '';
    }
  }

  // ============================================================================
  // QUESTION / ANSWER INTERACTION
  // ============================================================================

  // Suggestion Pills
  suggestionPills.forEach(pill => {
    pill.addEventListener('click', () => {
      const q = pill.getAttribute('data-query');
      if (q) {
        queryInput.value = q;
        chatForm.dispatchEvent(new Event('submit'));
      }
    });
  });

  // Submit Query
  chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (!query) return;

    addUserMessage(query);
    queryInput.value = '';
    btnSend.disabled = true;

    const loadingId = 'loading-' + Date.now();
    addLoadingMessage(loadingId);

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
      removeElement(loadingId);

      if (!res.ok) {
        addAssistantMessage(`Could not complete search: ${data.detail || 'Please verify documents are loaded.'}`, null);
        return;
      }

      addAssistantMessage(data.answer, data);
    } catch (err) {
      removeElement(loadingId);
      addAssistantMessage(`Connection error: ${err.message}`, null);
    } finally {
      btnSend.disabled = false;
      queryInput.focus();
    }
  });

  // Message Renderers
  function addUserMessage(text) {
    const row = document.createElement('div');
    row.className = 'message-row user';
    row.innerHTML = `
      <div class="message-avatar user">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>
      </div>
      <div class="message-content-wrapper">
        <div class="message-meta">
          <span class="message-sender">You</span>
          <span class="message-time">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
        </div>
        <div class="message-bubble">${escapeHTML(text)}</div>
      </div>
    `;
    messagesStream.appendChild(row);
    scrollToBottom();
  }

  function addAssistantMessage(text, meta) {
    const row = document.createElement('div');
    row.className = 'message-row';

    let groundingHtml = '';
    if (meta && typeof meta.faithfulness === 'number') {
      const pct = Math.round(meta.faithfulness * 100);
      groundingHtml = `
        <div class="grounding-box">
          <div class="grounding-header">
            <span class="fidelity-pill">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
              ${pct}% Source Grounded
            </span>
            <span class="retrieval-caption">Retrieved from ${meta.sources?.length || 0} excerpts</span>
          </div>
          ${renderSources(meta.sources || [])}
        </div>
      `;
    }

    row.innerHTML = `
      <div class="message-avatar assistant">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line></svg>
      </div>
      <div class="message-content-wrapper">
        <div class="message-meta">
          <span class="message-sender">Document Assistant</span>
          <span class="message-time">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
        </div>
        <div class="message-bubble">${cleanAnswerFormat(text)}</div>
        ${groundingHtml}
      </div>
    `;
    messagesStream.appendChild(row);
    scrollToBottom();
  }

  function renderSources(sources) {
    if (!sources || !sources.length) return '';

    const list = sources.map((s, i) => `
      <div class="source-card">
        <div class="source-card-header">
          <span>Source ${i + 1}: ${escapeHTML(s.source)} (Passage #${s.chunk_id})</span>
        </div>
        <div class="source-card-text">"${escapeHTML(s.text)}"</div>
      </div>
    `).join('');

    return `
      <details class="sources-details">
        <summary class="sources-summary">View source excerpts (${sources.length})</summary>
        <div class="sources-list">${list}</div>
      </details>
    `;
  }

  function addLoadingMessage(id) {
    const row = document.createElement('div');
    row.className = 'message-row';
    row.id = id;
    row.innerHTML = `
      <div class="message-avatar assistant">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 14 14"></polyline></svg>
      </div>
      <div class="message-content-wrapper">
        <div class="message-meta">
          <span class="message-sender">Document Assistant</span>
          <span class="message-time">Searching...</span>
        </div>
        <div class="message-bubble" style="color: var(--text-secondary); font-style: italic;">
          Reading passages and formulating answer...
        </div>
      </div>
    `;
    messagesStream.appendChild(row);
    scrollToBottom();
  }

  function addSystemMessage(text) {
    const row = document.createElement('div');
    row.className = 'message-row';
    row.innerHTML = `
      <div class="message-avatar assistant" style="background: rgba(255,255,255,0.05); color: var(--text-secondary);">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
      </div>
      <div class="message-content-wrapper">
        <div class="message-bubble" style="padding: 0.75rem 1rem; font-size: 0.85rem; color: var(--text-secondary); background: transparent; border-style: dashed;">
          ${escapeHTML(text)}
        </div>
      </div>
    `;
    messagesStream.appendChild(row);
    scrollToBottom();
  }

  function removeElement(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
  }

  // Clear Chat
  btnClearHistory.addEventListener('click', async () => {
    messagesStream.innerHTML = '';
    addSystemMessage('Conversation cleared.');
    try {
      await fetch('/api/reset', { method: 'POST' });
      await updateSystemStatus();
    } catch (_) {}
  });

  // API Key Modal
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
      addSystemMessage(newKey ? 'Gemini API Key configured.' : 'API Key removed (running in offline mode).');
      await updateSystemStatus();
    } catch (_) {}
  });

  // Formatting Utilities
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

  function cleanAnswerFormat(text) {
    let formatted = escapeHTML(text);
    formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    formatted = formatted.replace(/\n\n/g, '<br/><br/>');
    formatted = formatted.replace(/\n\* /g, '<br/>&bull; ');
    formatted = formatted.replace(/\n- /g, '<br/>&bull; ');
    return formatted;
  }

  // Initialize
  updateSystemStatus();
});
