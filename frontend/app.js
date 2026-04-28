const LOCAL_HOST_ALIASES = new Set(["localhost", "127.0.0.1", "0.0.0.0", ""]);
const runtimeConfig = window.HERBAL_APP_CONFIG || {};
const pageHost = window.location.hostname || "127.0.0.1";
const preferredApiHost = LOCAL_HOST_ALIASES.has(pageHost) ? "127.0.0.1" : pageHost;
const storedApiBase = localStorage.getItem("HERBAL_API_BASE");
const configuredApiBase = runtimeConfig.apiBase || runtimeConfig.API_BASE || "";
const API_BASES = [
  configuredApiBase,
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
const exportConversationButtonEl = document.querySelector("#export-conversation-button");
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
  if (data.response_type === "follow_up") {
    return buildFollowUpBlocks(data);
  }

  if (data.response_type === "recommendation") {
    return buildRecommendationBlocks(data);
  }

  if (data.response_type === "red_flag") {
    return buildRedFlagBlocks(data);
  }

  if (data.response_type === "medical_referral") {
    return buildMedicalReferralBlocks(data);
  }

  if (data.response_type === "out_of_scope") {
    return buildOutOfScopeBlocks(data);
  }

  if (data.response_type === "feedback") {
    return [
      {
        kind: "section",
        tone: "feedback",
        title: "Umpan balik tersimpan",
        text: data.reply,
      },
    ];
  }

  if (data.model_comparison && data.model_comparison.selected_reply) {
    return textToParagraphBlocks(data.reply);
  }

  return [
    {
      kind: "paragraph",
      text: data.reply,
    },
  ];
}

function buildFollowUpBlocks(data) {
  const assessment = getSelectedAssessment(data);
  const questions = buildFollowUpQuestions(data, assessment);
  return [
    {
      kind: "field",
      tone: "summary",
      label: "Ringkasan anamnesis saat ini:",
      text: buildFollowUpSummary(data, assessment),
    },
    {
      kind: "field",
      tone: "question",
      label: "Pertanyaan anamnesis yang perlu kamu jawab:",
      items: questions,
      ordered: true,
    },
    {
      kind: "field",
      tone: "note",
      label: "Jawab singkat saja, misalnya:",
      text: buildFollowUpExample(data, assessment),
    },
  ];
}

function buildRecommendationBlocks(data) {
  const assessment = getSelectedAssessment(data);
  const recommendation = data.recommendation || null;
  const suspected = compactList([
    ...(data.suspected_conditions || []),
    ...((assessment && assessment.suspected_conditions) || []),
    recommendation && recommendation.keluhan_ringan,
  ]).slice(0, 5);
  const finalAnswer = (assessment && assessment.final_answer) || extractSentence(data.reply) || "Berdasarkan anamnesis, sistem sudah menyusun jawaban awal dan rekomendasi yang paling sesuai dengan knowledge base.";
  const reasoning = (assessment && assessment.reasoning) || extractLineByPrefix(data.reply, "Pertimbangan utama") || "";
  const selfCare = buildSelfCareItems(data, assessment);
  const warningItems = buildWarningItems(data, assessment);

  const blocks = [
    {
      kind: "section",
      tone: "summary",
      title: "Ringkasan hasil anamnesis",
      text: finalAnswer,
    },
    {
      kind: "section",
      tone: "primary",
      title: "Kemungkinan penyebab / dugaan kondisi",
      text: reasoning || "Dugaan ini bersifat awal dan belum menggantikan diagnosis tenaga kesehatan.",
      items: suspected,
      ordered: false,
    },
    {
      kind: "section",
      tone: "action",
      title: "Yang bisa Anda lakukan sekarang",
      items: selfCare,
      ordered: false,
    },
    {
      kind: "section",
      tone: "safety",
      title: "Segera periksa ke dokter / IGD jika ada salah satu ini",
      items: warningItems,
      ordered: false,
    },
    {
      kind: "section",
      tone: "note",
      title: "Catatan",
      text: "Informasi ini adalah rekomendasi awal berbasis anamnesis dan knowledge base, bukan diagnosis medis final.",
    },
  ];

  if (data.feedback_prompt) {
    blocks.push({
      kind: "section",
      tone: "feedback",
      title: "Umpan balik",
      text: data.feedback_prompt,
      items: data.feedback_options || [],
      ordered: false,
    });
  }

  return blocks;
}

function buildRedFlagBlocks(data) {
  return [
    {
      kind: "section",
      tone: "danger",
      title: "Tanda bahaya terdeteksi",
      text: "Untuk keamanan, ramuan herbal tidak diposisikan sebagai penanganan utama.",
      items: data.red_flags || ["tanda bahaya"],
      ordered: false,
    },
    {
      kind: "section",
      tone: "safety",
      title: "Arahan sekarang",
      items: [
        "Segera konsultasi ke tenaga kesehatan atau fasilitas kesehatan terdekat.",
        "Jangan menunda bila gejala berat, menetap, atau memburuk.",
        "Gunakan informasi ini sebagai edukasi awal, bukan pengganti pemeriksaan medis.",
      ],
      ordered: false,
    },
  ];
}

function buildMedicalReferralBlocks(data) {
  const assessment = getSelectedAssessment(data);
  const suspected = compactList([
    ...(data.suspected_conditions || []),
    ...((assessment && assessment.suspected_conditions) || []),
  ]);
  return [
    {
      kind: "section",
      tone: "danger",
      title: "Keluhan perlu evaluasi medis",
      text: (assessment && (assessment.scope_reason || assessment.reasoning)) || extractSentence(data.reply) || "Keluhan ini tidak aman ditangani sebagai keluhan ringan mandiri.",
      items: suspected,
      ordered: false,
    },
    {
      kind: "section",
      tone: "safety",
      title: "Langkah aman",
      items: buildWarningItems(data, assessment).slice(0, 5),
      ordered: false,
    },
  ];
}

function buildOutOfScopeBlocks(data) {
  return [
    {
      kind: "section",
      tone: "note",
      title: "Butuh konteks keluhan yang lebih jelas",
      text: extractSentence(data.reply) || "Saya belum menemukan konteks keluhan yang cukup relevan dengan knowledge base saat ini.",
    },
    {
      kind: "section",
      tone: "question",
      title: "Coba tulis dengan format ini",
      items: [
        "Keluhan utama yang paling terasa.",
        "Sejak kapan muncul.",
        "Ada atau tidak demam, sesak, nyeri berat, muntah terus, darah, atau tanda bahaya lain.",
      ],
      ordered: false,
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
    if (block.kind === "field") {
      await renderFieldBlock(container, block, animate);
      continue;
    }

    if (block.kind === "section") {
      await renderStructuredSection(container, block, animate);
      continue;
    }

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

async function renderFieldBlock(container, block, animate = true) {
  const wrapperEl = document.createElement("div");
  wrapperEl.className = `answer-field ${block.tone ? `field-${block.tone}` : ""}`;
  container.appendChild(wrapperEl);

  const labelEl = document.createElement("strong");
  labelEl.className = "field-label";
  wrapperEl.appendChild(labelEl);
  await typeText(labelEl, block.label || "", animate);

  const items = compactList(block.items || []);
  if (items.length) {
    const listEl = document.createElement(block.ordered ? "ol" : "ul");
    listEl.className = "field-list";
    wrapperEl.appendChild(listEl);
    for (const item of items) {
      const itemEl = document.createElement("li");
      listEl.appendChild(itemEl);
      await typeText(itemEl, item, animate);
    }
    return;
  }

  const bodyEl = document.createElement("p");
  bodyEl.className = "field-body";
  wrapperEl.appendChild(bodyEl);
  await typeText(bodyEl, block.text || "-", animate);
}

async function renderStructuredSection(container, block, animate = true) {
  const wrapperEl = document.createElement("div");
  wrapperEl.className = `answer-section structured-section ${block.tone ? `section-${block.tone}` : ""}`;
  container.appendChild(wrapperEl);

  const titleEl = document.createElement("p");
  titleEl.className = "section-title";
  wrapperEl.appendChild(titleEl);
  await typeText(titleEl, block.title || "Ringkasan", animate);

  if (block.text) {
    const bodyEl = document.createElement("p");
    bodyEl.className = "section-body";
    wrapperEl.appendChild(bodyEl);
    await typeText(bodyEl, block.text, animate);
  }

  const items = compactList(block.items || []);
  if (!items.length) {
    return;
  }

  const listEl = document.createElement(block.ordered ? "ol" : "ul");
  wrapperEl.appendChild(listEl);
  for (const item of items) {
    const itemEl = document.createElement("li");
    listEl.appendChild(itemEl);
    await typeText(itemEl, item, animate);
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
    const breakdownEntries = Object.entries(breakdown);
    const score = Number(candidate.score || 0).toFixed(3);
    const latency = formatLatency(candidate.latency_ms);
    const answer = formatCandidateAnswer(candidate);
    const assessment = candidate.assessment || {};
    const suspected = compactList(assessment.suspected_conditions || []).slice(0, 3);
    const headMeta = compactList([
      candidate.provider ? `provider ${candidate.provider}` : "",
      candidate.status || "",
      `score ${score}`,
      `waktu respons ${latency}`,
    ]).join(" | ");
    const inferenceMetricsHtml = buildInferenceMetricsMarkup(candidate);
    candidateEl.innerHTML = `
      <div class="candidate-head">
        <strong>${escapeHtml(candidate.model)}</strong>
        <span>${escapeHtml(headMeta)}</span>
      </div>
      ${candidate.error ? `<p class="candidate-error">${escapeHtml(candidate.error)}</p>` : ""}
      ${
        candidate.status === "ok"
          ? `
            <div class="candidate-answer">
              <strong>Jawaban model:</strong>
              <p>${escapeHtml(answer)}</p>
            </div>
            ${
              suspected.length || assessment.reasoning || assessment.scope
                ? `<div class="candidate-assessment">
                    ${assessment.scope ? `<span>Scope: ${escapeHtml(formatScopeLabel(assessment.scope))}</span>` : ""}
                    ${suspected.length ? `<span>Dugaan: ${escapeHtml(suspected.join(", "))}</span>` : ""}
                    ${assessment.reasoning ? `<p>${escapeHtml(assessment.reasoning)}</p>` : ""}
                  </div>`
                : ""
            }
            <div class="score-breakdown">
              ${breakdownEntries
                .map(
                  ([key, value]) =>
                    `<span>${escapeHtml(formatMetricLabel(key))} ${formatScore(value)}</span>`,
                )
                .join("")}
            </div>
            ${inferenceMetricsHtml}`
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
  updateConversationActionState();
}

function updateConversationActionState() {
  const conversation = conversations.find((item) => item.id === activeConversationId);
  const hasMessages = Boolean(conversation && conversation.messages.length);
  exportConversationButtonEl.disabled = isResponding || !hasMessages;
}

function exportActiveConversation() {
  if (isResponding) return;

  const conversation = getActiveConversation();
  if (!conversation.messages.length) {
    window.alert("Belum ada percakapan untuk diexport.");
    return;
  }

  const workbookXml = buildConversationWorkbookXml(conversation);
  const fileName = `${sanitizeFileName(conversation.title || "percakapan-herbal")}-${formatExportTimestamp(new Date())}.xls`;
  const blob = new Blob([workbookXml], {
    type: "application/vnd.ms-excel;charset=utf-8;",
  });
  downloadBlob(blob, fileName);
}

function buildConversationWorkbookXml(conversation) {
  const worksheets = [
    buildWorksheetXml("Ringkasan", buildConversationSummaryRows(conversation), [180, 480]),
    buildWorksheetXml("Percakapan", buildConversationExportRows(conversation), [
      45,
      130,
      70,
      145,
      120,
      380,
      180,
      220,
      220,
      160,
      220,
      120,
      120,
    ]),
  ];

  const comparisonRows = buildComparisonExportRows(conversation);
  if (comparisonRows.length > 1) {
    worksheets.push(
      buildWorksheetXml("Komparasi Model", comparisonRows, [
        40,
        130,
        120,
        90,
        120,
        90,
        75,
        110,
        110,
        200,
        240,
        300,
        380,
      ]),
    );
  }

  return `<?xml version="1.0" encoding="UTF-8"?>
<?mso-application progid="Excel.Sheet"?>
<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet"
 xmlns:o="urn:schemas-microsoft-com:office:office"
 xmlns:x="urn:schemas-microsoft-com:office:excel"
 xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet"
 xmlns:html="http://www.w3.org/TR/REC-html40">
  <DocumentProperties xmlns="urn:schemas-microsoft-com:office:office">
    <Author>Herbal Chat</Author>
    <Created>${escapeXml(new Date().toISOString())}</Created>
    <Company>AI Chatbot Rekomendasi Ramuan Herbal</Company>
    <Version>16.00</Version>
  </DocumentProperties>
  <Styles>
    <Style ss:ID="Default" ss:Name="Normal">
      <Alignment ss:Vertical="Top" ss:WrapText="1"/>
      <Font ss:FontName="Calibri" ss:Size="11"/>
    </Style>
    <Style ss:ID="header">
      <Alignment ss:Vertical="Top" ss:WrapText="1"/>
      <Borders>
        <Border ss:Position="Bottom" ss:LineStyle="Continuous" ss:Weight="1"/>
        <Border ss:Position="Left" ss:LineStyle="Continuous" ss:Weight="1"/>
        <Border ss:Position="Right" ss:LineStyle="Continuous" ss:Weight="1"/>
        <Border ss:Position="Top" ss:LineStyle="Continuous" ss:Weight="1"/>
      </Borders>
      <Font ss:FontName="Calibri" ss:Size="11" ss:Bold="1"/>
      <Interior ss:Color="#EDE2CC" ss:Pattern="Solid"/>
    </Style>
    <Style ss:ID="cell">
      <Alignment ss:Vertical="Top" ss:WrapText="1"/>
      <Borders>
        <Border ss:Position="Bottom" ss:LineStyle="Continuous" ss:Weight="1"/>
        <Border ss:Position="Left" ss:LineStyle="Continuous" ss:Weight="1"/>
        <Border ss:Position="Right" ss:LineStyle="Continuous" ss:Weight="1"/>
        <Border ss:Position="Top" ss:LineStyle="Continuous" ss:Weight="1"/>
      </Borders>
      <Font ss:FontName="Calibri" ss:Size="11"/>
    </Style>
  </Styles>
  ${worksheets.join("")}
</Workbook>`;
}

function buildWorksheetXml(name, rows, columnWidths = []) {
  const safeRows = rows.length ? rows : [["Belum ada data"]];
  const columnCount = Math.max(...safeRows.map((row) => row.length), 1);
  const columnsXml = Array.from({ length: columnCount }, (_, index) => {
    const width = columnWidths[index] || 140;
    return `<Column ss:AutoFitWidth="1" ss:Width="${width}"/>`;
  }).join("");
  const rowsXml = safeRows
    .map((row, index) => buildWorksheetRowXml(row, index === 0 ? "header" : "cell"))
    .join("");

  return `<Worksheet ss:Name="${escapeXml(sanitizeSheetName(name))}">
    <Table ss:ExpandedColumnCount="${columnCount}" ss:ExpandedRowCount="${safeRows.length}">
      ${columnsXml}
      ${rowsXml}
    </Table>
    <WorksheetOptions xmlns="urn:schemas-microsoft-com:office:excel">
      <FreezePanes/>
      <FrozenNoSplit/>
      <SplitHorizontal>1</SplitHorizontal>
      <TopRowBottomPane>1</TopRowBottomPane>
      <ActivePane>2</ActivePane>
      <ProtectObjects>False</ProtectObjects>
      <ProtectScenarios>False</ProtectScenarios>
    </WorksheetOptions>
  </Worksheet>`;
}

function buildWorksheetRowXml(values, styleId) {
  const cellsXml = values
    .map((value) => `<Cell ss:StyleID="${styleId}"><Data ss:Type="String">${escapeXml(value)}</Data></Cell>`)
    .join("");
  return `<Row>${cellsXml}</Row>`;
}

function buildConversationSummaryRows(conversation) {
  const userCount = conversation.messages.filter((message) => message.role === "user").length;
  const assistantCount = conversation.messages.filter((message) => message.role === "assistant").length;
  const comparisonCount = conversation.messages.filter(
    (message) => message.role === "assistant" && message.data && message.data.model_comparison,
  ).length;
  return [
    ["Field", "Value"],
    ["Judul percakapan", conversation.title || "Chat baru"],
    ["Session ID", conversation.sessionId || "-"],
    ["Dibuat", formatDateTime(conversation.createdAt)],
    ["Terakhir diperbarui", formatDateTime(conversation.updatedAt)],
    ["Diexport pada", formatDateTime(new Date().toISOString())],
    ["Jumlah pesan user", String(userCount)],
    ["Jumlah respons assistant", String(assistantCount)],
    ["Jumlah blok komparasi model", String(comparisonCount)],
  ];
}

function buildConversationExportRows(conversation) {
  const rows = [
    [
      "No",
      "Waktu",
      "Peran",
      "Tahap",
      "Tipe respons",
      "Isi",
      "Dugaan kondisi",
      "Pertanyaan follow-up",
      "Red flags",
      "Ramuan",
      "Bahan",
      "Dosis",
      "Model terpilih",
    ],
  ];

  conversation.messages.forEach((message, index) => {
    if (message.role === "user") {
      rows.push([
        String(index + 1),
        formatDateTime(message.createdAt),
        "User",
        "-",
        "-",
        normalizeExportText(message.content),
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
      ]);
      return;
    }

    if (message.role !== "assistant" || !message.data) {
      rows.push([
        String(index + 1),
        formatDateTime(message.createdAt),
        capitalizeRole(message.role),
        "-",
        "-",
        normalizeExportText(message.content || ""),
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
      ]);
      return;
    }

    const data = message.data;
    const assessment = getSelectedAssessment(data);
    const recommendation = data.recommendation || null;
    const suspected = compactList([
      ...(data.suspected_conditions || []),
      ...((assessment && assessment.suspected_conditions) || []),
    ]).join("; ");
    const redFlags = compactList([
      ...(data.red_flags || []),
      ...((assessment && assessment.red_flags) || []),
    ]).join("; ");

    rows.push([
      String(index + 1),
      formatDateTime(message.createdAt),
      "Assistant",
      data.conversation_stage || "-",
      responseLabel(data.response_type),
      normalizeExportText(data.reply),
      suspected || "-",
      normalizeExportText(data.follow_up_question || (assessment && assessment.follow_up_question) || "-"),
      redFlags || "-",
      recommendation ? recommendation.ramuan : "-",
      recommendation ? compactList(recommendation.bahan || []).join(", ") || "-" : "-",
      recommendation ? recommendation.dosis_penggunaan || "-" : "-",
      (data.model_comparison && data.model_comparison.selected_model) || "-",
    ]);
  });

  return rows;
}

function buildComparisonExportRows(conversation) {
  const rows = [
    [
      "No respons",
      "Waktu",
      "Tipe respons",
      "Provider",
      "Model",
      "Status",
      "Score",
      "Waktu respons",
      "Scope",
      "Dugaan kondisi",
      "Scoring breakdown",
      "Metrik inferensi",
      "Jawaban model",
    ],
  ];

  let responseIndex = 0;
  conversation.messages.forEach((message) => {
    if (message.role !== "assistant" || !message.data || !message.data.model_comparison) {
      return;
    }

    const comparison = message.data.model_comparison;
    const candidates = Array.isArray(comparison.candidates) ? comparison.candidates : [];
    if (!candidates.length) {
      return;
    }

    responseIndex += 1;
    candidates.forEach((candidate) => {
      const assessment = candidate.assessment || {};
      rows.push([
        String(responseIndex),
        formatDateTime(message.createdAt),
        responseLabel(message.data.response_type),
        candidate.provider || "-",
        candidate.model || "-",
        candidate.status || "-",
        candidate.score === undefined || candidate.score === null ? "-" : Number(candidate.score).toFixed(3),
        formatLatency(candidate.latency_ms),
        assessment.scope ? formatScopeLabel(assessment.scope) : "-",
        compactList(assessment.suspected_conditions || []).join("; ") || "-",
        flattenBreakdown(candidate.scoring_breakdown),
        flattenInferenceMetrics(candidate.inference_metrics),
        normalizeExportText(formatCandidateAnswer(candidate)),
      ]);
    });
  });

  return rows;
}

function flattenBreakdown(breakdown) {
  const entries = Object.entries(breakdown || {});
  if (!entries.length) return "-";
  return entries
    .map(([key, value]) => `${formatMetricLabel(key)}=${formatScore(value)}`)
    .join("; ");
}

function flattenInferenceMetrics(metrics) {
  const entries = buildInferenceMetricEntries(metrics || {});
  if (!entries.length) return "-";
  return entries.map(({ label, value }) => `${label}=${value}`).join("; ");
}

function normalizeExportText(value) {
  return String(value || "")
    .replace(/\r/g, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim() || "-";
}

function downloadBlob(blob, filename) {
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => window.URL.revokeObjectURL(url), 1000);
}

function sanitizeFileName(value) {
  const sanitized = String(value || "")
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-zA-Z0-9\s-]+/g, "")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "")
    .toLowerCase();
  return sanitized || "percakapan-herbal";
}

function sanitizeSheetName(value) {
  return String(value || "Sheet")
    .replace(/[:\\/?*\[\]]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 31) || "Sheet";
}

function formatExportTimestamp(value) {
  const date = value instanceof Date ? value : new Date(value);
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  const hours = String(date.getHours()).padStart(2, "0");
  const minutes = String(date.getMinutes()).padStart(2, "0");
  const seconds = String(date.getSeconds()).padStart(2, "0");
  return `${year}${month}${day}-${hours}${minutes}${seconds}`;
}

function formatDateTime(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "-";
  return date.toLocaleString("id-ID", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function capitalizeRole(role) {
  const value = String(role || "").trim();
  if (!value) return "-";
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function escapeXml(value) {
  return String(value || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;")
    .replace(/\n/g, "&#10;");
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
    const sessionSync = buildSessionSyncPayload(conversation);
    const response = await fetchWithFallback("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: conversation.sessionId, session_sync: sessionSync }),
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

function buildSessionSyncPayload(conversation) {
  const turns = (conversation.messages || [])
    .map((message) => {
      if (message.role === "assistant" && message.data) {
        return {
          role: "assistant",
          content: message.data.reply || "",
        };
      }
      return {
        role: message.role || "user",
        content: message.content || "",
      };
    })
    .filter((turn) => String(turn.content || "").trim());

  const followUpMessages = (conversation.messages || []).filter(
    (message) => message.role === "assistant" && message.data && message.data.response_type === "follow_up",
  );
  const lastAssistantData = [...(conversation.messages || [])]
    .reverse()
    .find((message) => message.role === "assistant" && message.data)?.data || null;
  const lastRecommendationData = [...(conversation.messages || [])]
    .reverse()
    .find((message) => message.role === "assistant" && message.data && message.data.response_type === "recommendation")?.data || null;

  return {
    turns,
    question_count: followUpMessages.length,
    conversation_stage: lastRecommendationData ? "final_recommendation" : (lastAssistantData && lastAssistantData.conversation_stage) || "initial",
    completed: Boolean(lastRecommendationData),
    suspected_conditions: (lastAssistantData && lastAssistantData.suspected_conditions) || [],
    asked_follow_up_questions: followUpMessages
      .map((message) => message.data.follow_up_question || "")
      .filter((question) => String(question || "").trim()),
    last_recommendation: lastRecommendationData ? lastRecommendationData.recommendation || null : null,
  };
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
  updateConversationActionState();
}

function splitQuestions(text) {
  const normalized = String(text || "")
    .replace(/\r/g, "\n")
    .replace(/\s+(?=\d+[\).\s-]+\S)/g, "\n")
    .trim();

  return normalized
    .split(/\n+/)
    .flatMap((line) => line.split(/(?<=[?])\s+(?=[A-ZÀ-ÝA-Za-z0-9])/))
    .map((line) =>
      line
        .replace(/^(pertanyaan(?: anamnesis)?(?: yang perlu (?:kamu )?jawab| yang perlu dijawab)?|pertanyaan ke-\d+[^:]*):/i, "")
        .replace(/^\d+[\).\s-]+/, "")
        .replace(/^[-•]\s*/, "")
        .trim(),
    )
    .map((line) => {
      const questionMarkIndex = line.indexOf("?");
      return questionMarkIndex >= 0 ? line.slice(0, questionMarkIndex + 1).trim() : line;
    })
    .filter(Boolean)
    .filter((line) => line.length > 6)
    .slice(0, 3);
}

function buildFollowUpQuestions(data, assessment) {
  const sources = compactList([
    data.follow_up_question,
    assessment && assessment.follow_up_question,
    extractQuestionSection(data.reply),
  ]);

  for (const source of sources) {
    const questions = splitQuestions(source);
    if (questions.length) return questions;
  }

  return ["Sejak kapan keluhan muncul, apakah memburuk, dan apakah ada demam, sesak, muntah, darah, atau tanda bahaya lain?"];
}

function extractQuestionSection(text) {
  const normalized = String(text || "").replace(/\r/g, "\n");
  const match = normalized.match(
    /(?:Pertanyaan(?: anamnesis)?(?: yang perlu (?:kamu )?jawab| yang perlu dijawab)?|Pertanyaan ke-\d+[^:\n]*):\s*([\s\S]*?)(?:\n\n|Alasan pertanyaan|Jawab singkat|Informasi ini|$)/i,
  );
  return match ? match[1].trim() : normalized;
}

function getSelectedAssessment(data) {
  return data && data.model_comparison ? data.model_comparison.selected_assessment || null : null;
}

function buildFollowUpSummary(data, assessment) {
  const summary = data.anamnesis_summary || {};
  const suspected = compactList([
    ...(data.suspected_conditions || []),
    ...((assessment && assessment.suspected_conditions) || []),
    summary.keluhan_ringan,
  ]).slice(0, 3);
  const symptoms = compactList(summary.detected_symptoms || []).slice(0, 5);
  const parts = [];
  const mainCondition = summary.keluhan_ringan || suspected[0] || "keluhan yang kamu ceritakan";

  parts.push(
    `Saya menangkap keluhan ini mengarah ke "${mainCondition}". Sebelum memberi rekomendasi ramuan, saya perlu memastikan dulu apakah ada tanda bahaya dan memperjelas pola gejalanya.`,
  );

  if (symptoms.length) {
    parts.push(`Gejala terdeteksi: ${symptoms.join(", ")}.`);
  }
  parts.push(`Tahap anamnesis: pertanyaan ${data.questions_asked || 1} dari maksimal ${data.max_questions || 3}.`);

  const rationale = assessment && (assessment.follow_up_rationale || assessment.reasoning);
  if (rationale) {
    parts.push(`Alasan pertanyaan: ${rationale}`);
  }

  return parts.join(" ");
}

function buildFollowUpExample(data, assessment) {
  const question = normalizeForMatch(data.follow_up_question || "");
  const suspected = normalizeForMatch([...(data.suspected_conditions || []), ...((assessment && assessment.suspected_conditions) || [])].join(" "));

  if (question.includes("demam") || question.includes("sesak") || question.includes("tanda bahaya")) {
    return "Tidak ada demam, tidak sesak, tidak muntah terus, keluhan muncul sejak pagi, dan nyerinya ringan.";
  }
  if (question.includes("muntah") || question.includes("diare") || suspected.includes("mual")) {
    return "Tidak muntah, tidak diare, mual sejak tadi pagi, masih bisa minum, dan tidak ada nyeri berat.";
  }
  if (question.includes("ulu hati") || question.includes("makan") || suspected.includes("dispepsia")) {
    return "Perih lebih terasa saat perut kosong, tidak muntah, tidak ada tinja hitam, dan nyerinya tidak berat.";
  }
  if (question.includes("batuk") || question.includes("pilek")) {
    return "Ada batuk ringan sejak kemarin, tidak sesak, tidak demam tinggi, dan tidak ada batuk darah.";
  }
  if (question.includes("ruam") || question.includes("gatal")) {
    return "Ruam muncul sejak pagi, terasa gatal, tidak melepuh, tidak ada bengkak wajah, dan tidak sesak.";
  }

  return "Keluhan muncul sejak kapan, tingkat beratnya ringan/sedang/berat, ada atau tidak demam, sesak, muntah, darah, atau nyeri hebat.";
}

function compactList(items) {
  const seen = new Set();
  return (items || [])
    .map((item) => String(item || "").replace(/\s+/g, " ").trim())
    .filter(Boolean)
    .filter((item) => {
      const key = item.toLowerCase();
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
}

function normalizeForMatch(value) {
  return String(value || "").toLowerCase().replace(/\s+/g, " ").trim();
}

function buildSelfCareItems(data, assessment) {
  const recommendation = data.recommendation || null;
  const items = [
    "Istirahat cukup dan pantau perubahan gejala.",
    "Cukupi cairan, terutama bila ada demam, mual, diare, atau tubuh terasa lemas.",
    "Makan ringan dan hindari pemicu yang membuat keluhan memburuk.",
  ];

  if (recommendation) {
    items.push(`Bila tidak ada alergi atau kondisi khusus, ramuan ${recommendation.ramuan} dapat dipertimbangkan sesuai dosis/kisaran penggunaan di kartu rekomendasi.`);
  }

  if (assessment && assessment.warning_notes) {
    items.push(`Perhatikan kewaspadaan: ${assessment.warning_notes}`);
  }

  return compactList(items).slice(0, 6);
}

function buildWarningItems(data, assessment) {
  const recommendation = data.recommendation || null;
  return compactList([
    ...((data && data.red_flags) || []),
    ...((assessment && assessment.red_flags) || []),
    assessment && assessment.warning_notes,
    recommendation && recommendation.catatan_kewaspadaan,
    "Demam tinggi lebih dari 39°C atau demam lebih dari 3 hari.",
    "Sesak napas, nyeri hebat, pingsan, atau sangat lemas.",
    "Muntah terus, sulit minum, jarang kencing, mulut kering, atau tanda dehidrasi.",
    "Muncul darah, tinja hitam, ruam/bintik merah luas, atau kondisi cepat memburuk.",
  ]).slice(0, 7);
}

function extractSentence(text) {
  const normalized = String(text || "").replace(/\s+/g, " ").trim();
  if (!normalized) return "";
  const withoutLabels = normalized
    .replace(/Ringkasan anamnesis:.*?(?=Dugaan|Pertimbangan|Berdasarkan|Untuk saat ini|$)/i, "")
    .trim();
  const match = withoutLabels.match(/^(.+?[.!?])(\s|$)/);
  const sentence = match ? match[1] : withoutLabels;
  return sentence && sentence.length < 280 ? sentence : normalized.slice(0, 260).trim();
}

function extractLineByPrefix(text, prefix) {
  const pattern = new RegExp(`${prefix}\\s*:\\s*([^\\n]+)`, "i");
  const match = String(text || "").match(pattern);
  return match ? match[1].trim() : "";
}

function responseLabel(type) {
  const labels = {
    follow_up: "Anamnesis lanjutan",
    recommendation: "Rekomendasi terkurasi",
    medical_referral: "Perlu evaluasi medis",
    red_flag: "Tanda bahaya",
    out_of_scope: "Butuh konteks tambahan",
    feedback: "Umpan balik",
    preparation_detail: "Detail pengolahan",
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

function formatMetricLabel(key) {
  return String(key || "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function buildInferenceMetricsMarkup(candidate) {
  const entries = buildInferenceMetricEntries((candidate && candidate.inference_metrics) || {});
  if (!entries.length) {
    return "";
  }

  return `
    <div class="candidate-metrics">
      <strong>Metrik inferensi:</strong>
      <div class="inference-metrics-grid">
        ${entries
          .map(
            ({ label, value }) =>
              `<span><b>${escapeHtml(label)}</b><small>${escapeHtml(value)}</small></span>`,
          )
          .join("")}
      </div>
    </div>
  `;
}

function buildInferenceMetricEntries(metrics) {
  if (!metrics || typeof metrics !== "object") {
    return [];
  }

  const entries = [];
  const pushEntry = (key, label, formatter = (value) => String(value)) => {
    if (!(key in metrics) || metrics[key] === undefined || metrics[key] === null || metrics[key] === "") {
      return;
    }
    entries.push({ label, value: formatter(metrics[key]) });
  };

  pushEntry("total_duration", "total_duration", formatDurationFromNanoseconds);
  pushEntry("load_duration", "load_duration", formatDurationFromNanoseconds);
  pushEntry("prompt_eval_count", "prompt_eval_count", formatTokenCount);
  pushEntry("prompt_eval_duration", "prompt_eval_duration", formatDurationFromNanoseconds);
  pushEntry("prompt_eval_rate_tps", "prompt_eval_rate", formatTokenRate);
  pushEntry("eval_count", "eval_count", formatTokenCount);
  pushEntry("eval_duration", "eval_duration", formatDurationFromNanoseconds);
  pushEntry("eval_rate_tps", "eval_rate", formatTokenRate);
  pushEntry("prompt_tokens", "prompt_tokens", formatTokenCount);
  pushEntry("completion_tokens", "completion_tokens", formatTokenCount);
  pushEntry("total_tokens", "total_tokens", formatTokenCount);
  pushEntry("reasoning_tokens", "reasoning_tokens", formatTokenCount);
  pushEntry("cached_prompt_tokens", "cached_prompt_tokens", formatTokenCount);
  pushEntry("done_reason", "done_reason");
  pushEntry("finish_reason", "finish_reason");

  return entries;
}

function formatLatency(value) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) {
    return "-";
  }
  const ms = Number(value);
  if (ms >= 1000) {
    return `${(ms / 1000).toFixed(2)} detik (${Math.round(ms)} ms)`;
  }
  return `${Math.round(ms)} ms`;
}

function formatDurationFromNanoseconds(value) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) {
    return "-";
  }

  const nanoseconds = Number(value);
  if (nanoseconds >= 1_000_000_000) {
    return `${(nanoseconds / 1_000_000_000).toFixed(2)} detik`;
  }
  if (nanoseconds >= 1_000_000) {
    return `${(nanoseconds / 1_000_000).toFixed(2)} ms`;
  }
  if (nanoseconds >= 1_000) {
    return `${(nanoseconds / 1_000).toFixed(2)} μs`;
  }
  return `${Math.round(nanoseconds)} ns`;
}

function formatTokenCount(value) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) {
    return "-";
  }
  const count = Number(value);
  return `${Math.round(count)} token`;
}

function formatTokenRate(value) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) {
    return "-";
  }
  return `${Number(value).toFixed(2)} token/detik`;
}

function formatCandidateAnswer(candidate) {
  if (!candidate) return "Jawaban belum tersedia.";
  if (candidate.reply) return candidate.reply;
  const assessment = candidate.assessment || {};
  return (
    assessment.final_answer ||
    assessment.follow_up_question ||
    assessment.scope_reason ||
    assessment.reasoning ||
    "Jawaban belum tersedia."
  );
}

function formatScopeLabel(scope) {
  const labels = {
    supported: "Didukung",
    internal_medicine: "Penyakit dalam / rujukan",
    critical: "Kritis / red flag",
    unsupported: "Di luar cakupan",
  };
  return labels[scope] || scope;
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
exportConversationButtonEl.addEventListener("click", exportActiveConversation);
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
