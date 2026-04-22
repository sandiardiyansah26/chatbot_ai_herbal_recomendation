const LOCAL_HOST_ALIASES = new Set(["localhost", "127.0.0.1", "0.0.0.0", ""]);
const pageHost = window.location.hostname || "127.0.0.1";
const preferredApiHost = LOCAL_HOST_ALIASES.has(pageHost) ? "127.0.0.1" : pageHost;
const storedApiBase = localStorage.getItem("HERBAL_API_BASE");
const API_BASES = [
  storedApiBase,
  `http://${preferredApiHost}:8000`,
  "http://127.0.0.1:8000",
  "http://localhost:8000",
].filter(Boolean);
let apiBase = [...new Set(API_BASES)][0];

const STORAGE_KEYS = {
  conversations: "HERBAL_CHAT_CONVERSATIONS",
  activeConversation: "HERBAL_ACTIVE_CONVERSATION_ID",
  legacySession: "HERBAL_CHAT_SESSION_ID",
};

const messagesEl = document.querySelector("#messages");
const formEl = document.querySelector("#chat-form");
const inputEl = document.querySelector("#message-input");
const dynamicRepliesEl = document.querySelector("#dynamic-replies");
const quickPromptEls = document.querySelectorAll("[data-prompt]");
const sendButtonEl = formEl.querySelector("button");
const historyListEl = document.querySelector("#history-list");
const newChatButtonEl = document.querySelector("#new-chat-button");
const newChatInlineButtonEl = document.querySelector("#new-chat-inline-button");
const clearHistoryButtonEl = document.querySelector("#clear-history-button");
const activeChatTitleEl = document.querySelector("#active-chat-title");

let conversations = loadConversations();
let activeConversationId = localStorage.getItem(STORAGE_KEYS.activeConversation);
let isResponding = false;

function createMessageShell(type = "bot", status = "") {
  const rowEl = document.createElement("div");
  rowEl.className = `message-row ${type}`;

  const avatarEl = document.createElement("div");
  avatarEl.className = "avatar";
  avatarEl.textContent = type === "user" ? "U" : "AI";

  const bubbleEl = document.createElement("div");
  bubbleEl.className = `message ${type}`;

  if (status) {
    const statusEl = document.createElement("div");
    statusEl.className = "message-status";
    statusEl.textContent = status;
    bubbleEl.appendChild(statusEl);
  }

  if (type === "user") {
    rowEl.append(bubbleEl, avatarEl);
  } else {
    rowEl.append(avatarEl, bubbleEl);
  }

  messagesEl.appendChild(rowEl);
  scrollToBottom();
  return { rowEl, bubbleEl };
}

function appendUserMessage(content) {
  const { bubbleEl } = createMessageShell("user");
  bubbleEl.textContent = content;
}

function appendMetaMessage(content) {
  const { bubbleEl } = createMessageShell("meta");
  bubbleEl.textContent = content;
}

function appendThinkingMessage() {
  const { rowEl, bubbleEl } = createMessageShell("bot");
  bubbleEl.classList.add("thinking");
  bubbleEl.innerHTML = `
    <span class="thinking-dot"></span>
    <span class="thinking-dot"></span>
    <span class="thinking-dot"></span>
    <span class="thinking-text">Menganalisis gejala dan knowledge base herbal...</span>
  `;
  return rowEl;
}

async function renderAssistantResponse(data, options = {}) {
  const animate = options.animate !== false;
  const { bubbleEl } = createMessageShell("bot", responseLabel(data.response_type));
  bubbleEl.classList.add(`response-${data.response_type}`);

  const contentEl = document.createElement("div");
  contentEl.className = "assistant-content";
  bubbleEl.appendChild(contentEl);

  await typeBlocks(contentEl, buildResponseBlocks(data), animate);

  const cardsEl = buildStructuredCards(data);
  if (cardsEl) {
    bubbleEl.appendChild(cardsEl);
  }

  const comparisonEl = buildModelComparison(data.model_comparison);
  if (comparisonEl) {
    bubbleEl.appendChild(comparisonEl);
  }

  const contextEl = buildContextDetails(data.retrieved_context || []);
  if (contextEl) {
    bubbleEl.appendChild(contextEl);
  }
}

function buildResponseBlocks(data) {
  if (data.model_comparison && data.model_comparison.selected_reply) {
    return textToParagraphBlocks(data.reply);
  }

  if (data.response_type === "follow_up") {
    return [
      {
        kind: "paragraph",
        text: `Saya menangkap keluhan ini mengarah ke ${quote(
          (data.anamnesis_summary && data.anamnesis_summary.keluhan_ringan) || "keluhan ringan",
        )}. Sebelum memberi rekomendasi ramuan, saya perlu memastikan dulu apakah ada tanda bahaya.`,
      },
      {
        kind: "list",
        title: "Pertanyaan anamnesis yang perlu kamu jawab:",
        items: splitQuestions(data.follow_up_question || data.reply),
      },
      {
        kind: "paragraph",
        text: "Jawab singkat saja, misalnya: tidak demam, tidak sesak, ruam muncul sejak pagi, dan tidak ada bengkak wajah.",
      },
    ];
  }

  if (data.response_type === "recommendation" && data.recommendation) {
    return [
      {
        kind: "paragraph",
        text: `Keluhan masih saya posisikan sebagai ${quote(
          data.recommendation.keluhan_ringan,
        )} selama tidak ada tanda bahaya.`,
      },
      {
        kind: "paragraph",
        text: "Berikut rekomendasi ramuan herbal awal yang paling relevan dari knowledge base:",
      },
    ];
  }

  if (data.response_type === "red_flag") {
    return [
      {
        kind: "paragraph",
        text: `Saya mendeteksi tanda yang perlu diwaspadai: ${(data.red_flags || ["tanda bahaya"]).join(", ")}.`,
      },
      {
        kind: "paragraph",
        text: "Untuk keamanan, ramuan herbal tidak saya posisikan sebagai penanganan utama pada kondisi ini. Sebaiknya segera konsultasi ke tenaga kesehatan atau fasilitas kesehatan terdekat, terutama bila gejala berat, menetap, atau memburuk.",
      },
      {
        kind: "paragraph",
        text: "Informasi ini bersifat rekomendasi awal dan edukasi, bukan diagnosis medis final.",
      },
    ];
  }

  return [
    {
      kind: "paragraph",
      text: data.reply,
    },
  ];
}

function textToParagraphBlocks(text) {
  const normalized = String(text || "").trim();
  if (!normalized) {
    return [{ kind: "paragraph", text: "Respons belum tersedia." }];
  }

  return normalized
    .split(/\n{2,}/)
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => ({ kind: "paragraph", text: part }));
}

async function typeBlocks(container, blocks, animate = true) {
  for (const block of blocks) {
    if (block.kind === "list") {
      const wrapperEl = document.createElement("div");
      wrapperEl.className = "answer-section";
      const titleEl = document.createElement("p");
      titleEl.className = "section-title";
      wrapperEl.appendChild(titleEl);
      container.appendChild(wrapperEl);
      await typeText(titleEl, block.title, animate);

      const listEl = document.createElement("ol");
      wrapperEl.appendChild(listEl);
      for (const item of block.items) {
        const itemEl = document.createElement("li");
        listEl.appendChild(itemEl);
        await typeText(itemEl, item, animate);
      }
      continue;
    }

    const paragraphEl = document.createElement("p");
    paragraphEl.className = "assistant-paragraph";
    container.appendChild(paragraphEl);
    await typeText(paragraphEl, block.text, animate);
  }
}

async function typeText(element, text, animate = true) {
  const shouldAnimate = animate && !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (!shouldAnimate || text.length > 900) {
    element.textContent = text;
    scrollToBottom();
    return;
  }

  const tokens = text.match(/\S+\s*/g) || [text];
  for (const token of tokens) {
    element.textContent += token;
    scrollToBottom();
    await wait(Math.min(34, 10 + token.length * 1.4));
  }
}

function buildStructuredCards(data) {
  if (data.response_type !== "recommendation" || !data.recommendation) {
    return null;
  }

  const recommendation = data.recommendation;
  const cardsEl = document.createElement("div");
  cardsEl.className = "recommendation-card";

  cardsEl.innerHTML = `
    <div class="card-kicker">Rekomendasi Herbal</div>
    <h3>${escapeHtml(recommendation.ramuan)}</h3>
    <div class="recipe-grid">
      <div>
        <span>Bahan</span>
        <p>${escapeHtml(recommendation.bahan.join(", "))}</p>
      </div>
      <div>
        <span>Dosis</span>
        <p>${escapeHtml(recommendation.dosis_penggunaan)}</p>
      </div>
    </div>
    <div class="recipe-step">
      <span>Cara Pengolahan</span>
      <p>${escapeHtml(recommendation.cara_pengolahan)}</p>
    </div>
    <div class="safety-note">
      <span>Catatan Kewaspadaan</span>
      <p>${escapeHtml(recommendation.catatan_kewaspadaan)}</p>
    </div>
    <p class="source-line">Sumber ringkas: ${escapeHtml(recommendation.sumber_ringkas)}</p>
    <p class="disclaimer">${escapeHtml(recommendation.disclaimer)}</p>
  `;

  return cardsEl;
}

function buildContextDetails(contexts = []) {
  if (!contexts.length) return null;

  const detailsEl = document.createElement("details");
  detailsEl.className = "context-details";
  const summaryEl = document.createElement("summary");
  summaryEl.textContent = `Lihat konteks RAG yang dipakai (${contexts.length})`;
  detailsEl.appendChild(summaryEl);

  const listEl = document.createElement("div");
  listEl.className = "context-list";
  contexts.slice(0, 6).forEach((item) => {
    const itemEl = document.createElement("div");
    itemEl.className = "context-item";
    itemEl.innerHTML = `
      <strong>${escapeHtml(item.type)}: ${escapeHtml(item.title)}</strong>
      <span>score ${Number(item.score).toFixed(3)}${item.evidence_level ? ` - ${escapeHtml(item.evidence_level)}` : ""}</span>
    `;
    listEl.appendChild(itemEl);
  });
  detailsEl.appendChild(listEl);
  return detailsEl;
}

function buildModelComparison(comparison) {
  if (!comparison || !comparison.enabled) return null;

  const detailsEl = document.createElement("details");
  detailsEl.className = "model-comparison-card";
  if (comparison.selected_model || comparison.note) {
    detailsEl.open = true;
  }

  const summaryEl = document.createElement("summary");
  summaryEl.textContent = comparison.selected_model
    ? `Komparasi GenAI: pemenang ${comparison.selected_model}`
    : "Komparasi GenAI: kandidat belum tersedia";
  detailsEl.appendChild(summaryEl);

  const bodyEl = document.createElement("div");
  bodyEl.className = "model-comparison-body";

  if (comparison.note) {
    const noteEl = document.createElement("p");
    noteEl.className = "comparison-note";
    noteEl.textContent = comparison.note;
    bodyEl.appendChild(noteEl);
  }

  if (comparison.selected_model) {
    const winnerEl = document.createElement("div");
    winnerEl.className = "comparison-winner";
    winnerEl.innerHTML = `
      <span>Jawaban Terpilih</span>
      <strong>${escapeHtml(comparison.selected_model)}</strong>
      <p>Jawaban utama di atas adalah kandidat dengan skor tertinggi dari proses scoring berbasis RAG dan safety.</p>
    `;
    bodyEl.appendChild(winnerEl);
  }

  const candidatesEl = document.createElement("div");
  candidatesEl.className = "candidate-list";
  (comparison.candidates || []).forEach((candidate) => {
    const candidateEl = document.createElement("div");
    candidateEl.className = `candidate-item ${candidate.status === "ok" ? "ok" : "error"}`;
    const breakdown = candidate.scoring_breakdown || {};
    const score = Number(candidate.score || 0).toFixed(3);
    const latency = candidate.latency_ms ? `${candidate.latency_ms} ms` : "-";
    candidateEl.innerHTML = `
      <div class="candidate-head">
        <strong>${escapeHtml(candidate.model)}</strong>
        <span>${escapeHtml(candidate.status)} | score ${score} | ${escapeHtml(latency)}</span>
      </div>
      ${candidate.error ? `<p class="candidate-error">${escapeHtml(candidate.error)}</p>` : ""}
      ${
        candidate.status === "ok"
          ? `<div class="score-breakdown">
              <span>Safety ${formatScore(breakdown.safety)}</span>
              <span>Grounding ${formatScore(breakdown.grounding)}</span>
              <span>Completeness ${formatScore(breakdown.completeness)}</span>
              <span>Language ${formatScore(breakdown.language)}</span>
            </div>`
          : ""
      }
    `;
    candidatesEl.appendChild(candidateEl);
  });
  bodyEl.appendChild(candidatesEl);

  if (comparison.learning_log_id) {
    const learningEl = document.createElement("p");
    learningEl.className = "learning-log";
    learningEl.textContent = `Learning log: ${comparison.learning_log_id}`;
    bodyEl.appendChild(learningEl);
  }

  detailsEl.appendChild(bodyEl);
  return detailsEl;
}

function clearDynamicReplies() {
  dynamicRepliesEl.innerHTML = "";
}

function renderQuickReplies(replies = []) {
  clearDynamicReplies();
  replies.forEach((reply, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = index === 0 ? "suggested" : "";
    button.textContent = reply;
    button.addEventListener("click", () => {
      if (reply.toLowerCase() === "mulai sesi baru") {
        startNewConversation();
        return;
      }
      sendMessage(reply);
    });
    dynamicRepliesEl.appendChild(button);
  });
}

function loadConversations() {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEYS.conversations) || "[]");
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter((item) => item && item.id)
      .map((item) => ({
        id: item.id,
        sessionId: item.sessionId || null,
        title: item.title || "Chat baru",
        messages: Array.isArray(item.messages) ? item.messages : [],
        createdAt: item.createdAt || new Date().toISOString(),
        updatedAt: item.updatedAt || item.createdAt || new Date().toISOString(),
      }));
  } catch {
    return [];
  }
}

function saveConversations() {
  conversations = conversations
    .filter((conversation) => conversation && conversation.id)
    .sort((a, b) => new Date(b.updatedAt || 0) - new Date(a.updatedAt || 0))
    .slice(0, 40);
  localStorage.setItem(STORAGE_KEYS.conversations, JSON.stringify(conversations));
  if (activeConversationId) {
    localStorage.setItem(STORAGE_KEYS.activeConversation, activeConversationId);
  }
}

function createConversation() {
  const now = new Date().toISOString();
  const conversation = {
    id: makeId(),
    sessionId: null,
    title: "Chat baru",
    messages: [],
    createdAt: now,
    updatedAt: now,
  };
  conversations.unshift(conversation);
  activeConversationId = conversation.id;
  saveConversations();
  return conversation;
}

function getActiveConversation() {
  let conversation = conversations.find((item) => item.id === activeConversationId);
  if (!conversation) {
    conversation = createConversation();
  }
  return conversation;
}

function touchConversation(conversation) {
  conversation.updatedAt = new Date().toISOString();
  saveConversations();
  renderHistoryList();
  renderActiveTitle();
}

function startNewConversation() {
  if (isResponding) return;
  conversations = conversations.filter((conversation) => {
    return conversation.messages.length > 0 || conversation.id !== activeConversationId;
  });
  const conversation = createConversation();
  inputEl.value = "";
  renderHistoryList();
  renderConversation(conversation, { fresh: true });
  inputEl.focus();
}

async function clearAllConversations() {
  if (isResponding) return;
  const hasHistory = conversations.some((conversation) => conversation.messages.length > 0);
  if (hasHistory && !window.confirm("Hapus semua riwayat percakapan lokal?")) {
    return;
  }

  const sessionIds = conversations.map((conversation) => conversation.sessionId).filter(Boolean);
  conversations = [];
  activeConversationId = null;
  localStorage.removeItem(STORAGE_KEYS.conversations);
  localStorage.removeItem(STORAGE_KEYS.activeConversation);
  localStorage.removeItem(STORAGE_KEYS.legacySession);
  createConversation();
  renderHistoryList();
  await renderConversation(getActiveConversation());

  sessionIds.forEach((sessionId) => {
    fetchWithFallback(`/api/session/${sessionId}`, { method: "DELETE" }).catch(() => {});
  });
}

function renderHistoryList() {
  historyListEl.innerHTML = "";
  conversations.forEach((conversation) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `history-item${conversation.id === activeConversationId ? " active" : ""}`;
    button.innerHTML = `
      <span class="history-title">${escapeHtml(conversation.title || "Chat baru")}</span>
      <span class="history-meta">${conversation.messages.length} pesan - ${formatDate(conversation.updatedAt)}</span>
    `;
    button.addEventListener("click", () => {
      if (isResponding || conversation.id === activeConversationId) return;
      activeConversationId = conversation.id;
      saveConversations();
      renderHistoryList();
      renderConversation(conversation);
      inputEl.focus();
    });
    historyListEl.appendChild(button);
  });
}

async function renderConversation(conversation, options = {}) {
  messagesEl.innerHTML = "";
  clearDynamicReplies();
  renderActiveTitle();

  if (!conversation.messages.length) {
    renderWelcomeState(options.fresh);
    return;
  }

  for (const message of conversation.messages) {
    if (message.role === "user") {
      appendUserMessage(message.content);
    } else if (message.role === "assistant" && message.data) {
      await renderAssistantResponse(message.data, { animate: false });
    } else if (message.role === "meta") {
      appendMetaMessage(message.content);
    }
  }

  const lastAssistant = [...conversation.messages].reverse().find((message) => message.role === "assistant");
  const replies = lastAssistant && lastAssistant.data ? lastAssistant.data.quick_replies : [];
  renderQuickReplies(replies || []);
  scrollToBottom();
}

function renderWelcomeState(fresh = false) {
  messagesEl.innerHTML = `
    <div class="empty-state">
      <div class="empty-state-card">
        <span>Anamnesis awal</span>
        <h2>${fresh ? "Chat baru siap." : "Mulai dari keluhan yang paling terasa."}</h2>
        <p>
          Tulis keluhan ringan yang ingin dicek. Chatbot akan bertanya dulu seperti anamnesis,
          lalu memberi rekomendasi ramuan herbal hanya bila konteksnya cukup dan tidak ada tanda bahaya.
        </p>
      </div>
    </div>
  `;
}

function renderActiveTitle() {
  const conversation = conversations.find((item) => item.id === activeConversationId);
  activeChatTitleEl.textContent = (conversation && conversation.title) || "Chat baru";
}

async function sendMessage(message) {
  if (isResponding) return;
  isResponding = true;

  const conversation = getActiveConversation();
  if (!conversation.messages.length) {
    messagesEl.innerHTML = "";
  }

  appendUserMessage(message);
  conversation.messages.push({
    id: makeId(),
    role: "user",
    content: message,
    createdAt: new Date().toISOString(),
  });
  if (!conversation.title || conversation.title === "Chat baru") {
    conversation.title = titleFromMessage(message);
  }
  touchConversation(conversation);

  clearDynamicReplies();
  inputEl.value = "";
  setInputDisabled(true);
  const thinkingEl = appendThinkingMessage();

  try {
    const response = await fetchWithFallback("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: conversation.sessionId }),
    });

    if (!response.ok) {
      throw new Error(`Backend memberi status ${response.status}`);
    }

    const data = await response.json();
    conversation.sessionId = data.session_id;
    conversation.messages.push({
      id: makeId(),
      role: "assistant",
      data,
      createdAt: new Date().toISOString(),
    });
    touchConversation(conversation);

    thinkingEl.remove();
    await renderAssistantResponse(data);
    renderQuickReplies(data.quick_replies || []);
  } catch (error) {
    thinkingEl.remove();
    const { bubbleEl } = createMessageShell("bot", "Koneksi backend");
    bubbleEl.textContent = `Maaf, backend belum bisa dihubungi. Pastikan service berjalan di ${apiBase}.\nDetail: ${error.message}`;
  } finally {
    isResponding = false;
    setInputDisabled(false);
    inputEl.focus();
  }
}

async function fetchWithFallback(path, options = {}) {
  const bases = [...new Set([apiBase, ...API_BASES])].filter(Boolean);
  let lastError;

  for (const base of bases) {
    try {
      const response = await fetch(`${base}${path}`, options);
      apiBase = base;
      localStorage.setItem("HERBAL_API_BASE", base);
      return response;
    } catch (error) {
      lastError = error;
    }
  }

  throw lastError || new Error("Backend tidak dapat dihubungi");
}

function setInputDisabled(disabled) {
  inputEl.disabled = disabled;
  sendButtonEl.disabled = disabled;
  sendButtonEl.textContent = disabled ? "Menulis..." : "Kirim";
}

function splitQuestions(text) {
  return text
    .split("\n")
    .map((line) => line.replace(/^\d+\.\s*/, "").trim())
    .filter(Boolean)
    .slice(0, 7);
}

function responseLabel(type) {
  const labels = {
    follow_up: "Anamnesis lanjutan",
    recommendation: "Rekomendasi terkurasi",
    red_flag: "Tanda bahaya",
    out_of_scope: "Butuh konteks tambahan",
  };
  return labels[type] || "Respons";
}

function quote(text) {
  return `"${text}"`;
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function makeId() {
  if (window.crypto && typeof window.crypto.randomUUID === "function") {
    return window.crypto.randomUUID();
  }
  return `chat_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

function titleFromMessage(message) {
  const normalized = message.replace(/\s+/g, " ").trim();
  if (normalized.length <= 46) return normalized;
  return `${normalized.slice(0, 43).trim()}...`;
}

function formatDate(value) {
  if (!value) return "baru";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "baru";
  return date.toLocaleDateString("id-ID", { day: "2-digit", month: "short" });
}

function formatScore(value) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(2);
}

formEl.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = inputEl.value.trim();
  if (message) sendMessage(message);
});

inputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    formEl.requestSubmit();
  }
});

quickPromptEls.forEach((button) => {
  button.addEventListener("click", () => {
    const prompt = button.dataset.prompt;
    if (prompt) sendMessage(prompt);
  });
});

newChatButtonEl.addEventListener("click", startNewConversation);
newChatInlineButtonEl.addEventListener("click", startNewConversation);
clearHistoryButtonEl.addEventListener("click", clearAllConversations);

async function initializeApp() {
  localStorage.removeItem(STORAGE_KEYS.legacySession);
  if (!conversations.length || !conversations.some((item) => item.id === activeConversationId)) {
    createConversation();
  }
  renderHistoryList();
  await renderConversation(getActiveConversation());
}

initializeApp();
