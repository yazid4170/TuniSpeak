const DEFAULT_API_ORIGIN = "http://127.0.0.1:8000";
const apiOrigin = window.location?.origin || DEFAULT_API_ORIGIN;
const API_BASE = `${apiOrigin.replace(/\/?$/, "")}/api/v1`;
const QA_URL = `${API_BASE}/qa`;
const FEEDBACK_URL = `${API_BASE}/feedback`;

let currentInteractionId = null;
let selectedRating = null;
let speechRecognition = null;
let isListening = false;
let speechSynthesisSupported = false;

function setLoading(isLoading) {
  const askBtn = document.getElementById("ask");
  if (!askBtn) return;
  askBtn.disabled = isLoading;
  if (isLoading) {
    askBtn.classList.add("loading");
  } else {
    askBtn.classList.remove("loading");
  }
}

function updateCharCount() {
  const textarea = document.getElementById("question");
  const counter = document.getElementById("char-count");
  if (!textarea || !counter) return;
  const len = textarea.value.length;
  counter.textContent = `${len} caractère${len > 1 ? "s" : ""}`;
}

function updateAskDisabled() {
  const askBtn = document.getElementById("ask");
  const textarea = document.getElementById("question");
  if (!askBtn || !textarea) return;
  askBtn.disabled = !textarea.value.trim();
}

function isRTLText(s) {
  // Basic detection for Arabic script
  return /[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]/.test(s);
}

function applyDir(node, rtl) {
  if (!node) return;
  node.setAttribute("dir", rtl ? "rtl" : "ltr");
}

function toggleSpeechButtons(listening) {
  const startBtn = document.getElementById("speech-start");
  const stopBtn = document.getElementById("speech-stop");
  if (!startBtn || !stopBtn) {
    return;
  }
  startBtn.disabled = listening;
  stopBtn.disabled = !listening;
}

function setReadButtonEnabled(enabled) {
  const readBtn = document.getElementById("speech-read");
  if (readBtn) {
    readBtn.disabled = !(speechSynthesisSupported && enabled);
  }
}

function updateSpeechStatus(message) {
  const statusNode = document.getElementById("speech-status");
  if (!statusNode) {
    return;
  }
  if (message) {
    statusNode.textContent = message;
    statusNode.hidden = false;
  } else {
    statusNode.hidden = true;
    statusNode.textContent = "";
  }
}

function initializeSpeech() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  speechSynthesisSupported = "speechSynthesis" in window;

  if (speechSynthesisSupported) {
    setReadButtonEnabled(false);
  } else {
    const readBtn = document.getElementById("speech-read");
    if (readBtn) {
      readBtn.disabled = true;
    }
  }

  if (!SpeechRecognition) {
    const unsupported = document.getElementById("speech-unsupported");
    if (unsupported) {
      unsupported.hidden = false;
    }
    return;
  }

  speechRecognition = new SpeechRecognition();
  speechRecognition.continuous = false;
  speechRecognition.interimResults = true;
  speechRecognition.maxAlternatives = 1;

  speechRecognition.addEventListener("result", (event) => {
    const textarea = document.getElementById("question");
    if (!textarea) {
      return;
    }
    let transcript = "";
    for (let i = event.resultIndex; i < event.results.length; i += 1) {
      transcript += event.results[i][0].transcript;
    }
    textarea.value = transcript.trim();
    const lastResult = event.results[event.results.length - 1];
    if (lastResult?.isFinal) {
      isListening = false;
      toggleSpeechButtons(false);
      updateSpeechStatus("Dictée terminée. Vérifiez la question avant envoi.");
      try {
        speechRecognition.stop();
      } catch (err) {
        // Ignore stop errors triggered by state transitions.
      }
      textarea.focus();
      const end = textarea.value.length;
      textarea.setSelectionRange(end, end);
    }
  });

  speechRecognition.addEventListener("error", (event) => {
    updateSpeechStatus(`Erreur dictée : ${event.error}`);
    isListening = false;
    toggleSpeechButtons(false);
  });

  speechRecognition.addEventListener("end", () => {
    if (isListening) {
      speechRecognition.start();
    }
  });

  const controls = document.getElementById("speech-controls");
  if (controls) {
    controls.hidden = false;
  }
  toggleSpeechButtons(false);
  updateSpeechStatus(null);
}

function startSpeech() {
  if (!speechRecognition || isListening) {
    return;
  }
  const langSelect = document.getElementById("speech-lang");
  const language = langSelect ? langSelect.value : "fr-FR";
  speechRecognition.lang = language;
  isListening = true;
  toggleSpeechButtons(true);
  updateSpeechStatus("Dictée en cours...");
  try {
    speechRecognition.start();
  } catch (error) {
    updateSpeechStatus(error.message);
    isListening = false;
    toggleSpeechButtons(false);
  }
}

function stopSpeech() {
  if (!speechRecognition || !isListening) {
    return;
  }
  isListening = false;
  speechRecognition.stop();
  toggleSpeechButtons(false);
  updateSpeechStatus("Dictée interrompue.");
}

function readAnswerAloud() {
  if (!speechSynthesisSupported) {
    return;
  }
  const answerNode = document.getElementById("answer");
  if (!answerNode?.textContent) {
    return;
  }
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(answerNode.textContent);
  const langSelect = document.getElementById("speech-lang");
  if (langSelect) {
    utterance.lang = langSelect.value;
  }
  window.speechSynthesis.speak(utterance);
}

function resetFeedback() {
  selectedRating = null;
  document.getElementById("feedback").hidden = true;
  document.getElementById("feedback-comment").value = "";
  document.getElementById("feedback-correction").value = "";
  document.getElementById("feedback-status").hidden = true;
  document
    .querySelectorAll(".feedback-btn")
    .forEach((btn) => btn.classList.remove("active"));
}

async function askQuestion() {
  const questionInput = document.getElementById("question");
  const responsePanel = document.getElementById("response");
  const answerNode = document.getElementById("answer");
  const sourcesNode = document.getElementById("sources");
  const metaNode = document.getElementById("meta");
  const noticeNode = document.getElementById("abstain");

  resetFeedback();
  setReadButtonEnabled(false);
  if (speechSynthesisSupported) {
    window.speechSynthesis.cancel();
  }
  updateSpeechStatus(null);
  const payload = {
    question: questionInput.value.trim(),
    top_k: 5,
  };

  try {
    setLoading(true);
    if (!payload.question) {
      setLoading(false);
      return;
    }
    const response = await fetch(`${QA_URL}/answer`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Erreur inconnue");
    }

    const result = await response.json();
    answerNode.textContent = result.answer;
    applyDir(answerNode, result.language && (result.language === "ar" || result.language === "aeb"));
    const confidencePercent = ((result.confidence || 0) * 100).toFixed(1);
    const normalized = result.normalized_query || payload.question;
    const normalizedLabel =
      normalized && normalized !== payload.question
        ? ` • Requête normalisée : ${normalized}`
        : "";
    metaNode.textContent = `Langue détectée : ${result.language?.toUpperCase() || ""} • Confiance : ${confidencePercent}%${normalizedLabel}`;
    if (result.abstain) {
      noticeNode.textContent = result.reason || "Réponse fournie avec une faible confiance.";
      noticeNode.hidden = false;
    } else {
      noticeNode.hidden = true;
      noticeNode.textContent = "";
    }
    sourcesNode.innerHTML = "";
    result.sources.forEach((source) => {
      const li = document.createElement("li");
      li.className = "source-card";
      const title = document.createElement("div");
      title.className = "source-title";
      title.textContent = source.title || source.document_id || "Source";
      const scoreWrap = document.createElement("div");
      scoreWrap.className = "score-wrap";
      const scoreBar = document.createElement("div");
      scoreBar.className = "score-bar";
      const fill = document.createElement("div");
      fill.className = "score-fill";
      const score = typeof source.score === "number" ? source.score : 0;
      const pct = Math.max(0, Math.min(100, Math.round(score * 100)));
      fill.style.width = `${pct}%`;
      const label = document.createElement("span");
      label.textContent = `score ${score.toFixed ? score.toFixed(3) : score}`;
      scoreBar.appendChild(fill);
      scoreWrap.appendChild(scoreBar);
      scoreWrap.appendChild(label);
      li.appendChild(title);
      li.appendChild(scoreWrap);
      sourcesNode.appendChild(li);
    });
    responsePanel.hidden = false;
    currentInteractionId = result.interaction_id;
    document.getElementById("feedback").hidden = !currentInteractionId;
    document.getElementById("feedback-status").hidden = true;
    document.getElementById("feedback-comment").value = "";
    document.getElementById("feedback-correction").value = "";
    document
      .querySelectorAll(".feedback-btn")
      .forEach((btn) => btn.classList.remove("active"));
    selectedRating = null;
    if (speechSynthesisSupported && result.answer) {
      setReadButtonEnabled(true);
    }
  } catch (error) {
    answerNode.textContent = error.message;
    sourcesNode.innerHTML = "";
    metaNode.textContent = "";
    noticeNode.hidden = true;
    noticeNode.textContent = "";
    responsePanel.hidden = false;
    resetFeedback();
  } finally {
    setLoading(false);
  }
}

async function submitFeedback() {
  if (!currentInteractionId) {
    return;
  }
  const statusNode = document.getElementById("feedback-status");
  if (!selectedRating) {
    statusNode.textContent = "Sélectionnez d'abord si la réponse était utile ou non.";
    statusNode.hidden = false;
    return;
  }
  const payload = {
    interaction_id: currentInteractionId,
    rating: selectedRating,
    comment: document.getElementById("feedback-comment").value.trim() || null,
    correction: document.getElementById("feedback-correction").value.trim() || null,
  };

  try {
    const response = await fetch(FEEDBACK_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Impossible d'enregistrer le feedback");
    }
    statusNode.textContent = "Merci ! Votre feedback a été pris en compte.";
    statusNode.hidden = false;
    document.getElementById("feedback").hidden = true;
  } catch (err) {
    statusNode.textContent = err.message;
    statusNode.hidden = false;
  }
}

function selectRating(rating) {
  selectedRating = rating;
  document
    .querySelectorAll(".feedback-btn")
    .forEach((btn) => btn.classList.remove("active"));
  if (rating === "helpful") {
    document.getElementById("feedback-helpful").classList.add("active");
  } else {
    document.getElementById("feedback-unhelpful").classList.add("active");
  }
  const statusNode = document.getElementById("feedback-status");
  statusNode.hidden = true;
}

document.getElementById("ask").addEventListener("click", askQuestion);
document
  .getElementById("feedback-helpful")
  .addEventListener("click", () => selectRating("helpful"));
document
  .getElementById("feedback-unhelpful")
  .addEventListener("click", () => selectRating("unhelpful"));
document.getElementById("submit-feedback").addEventListener("click", submitFeedback);

const startBtn = document.getElementById("speech-start");
const stopBtn = document.getElementById("speech-stop");
const readBtn = document.getElementById("speech-read");

if (startBtn && stopBtn) {
  startBtn.addEventListener("click", startSpeech);
  stopBtn.addEventListener("click", stopSpeech);
}
if (readBtn) {
  readBtn.addEventListener("click", readAnswerAloud);
}

// UX wiring: character counter, keyboard shortcuts, examples
function initializeUX() {
  const textarea = document.getElementById("question");
  if (textarea) {
    textarea.addEventListener("input", updateCharCount);
    textarea.addEventListener("input", updateAskDisabled);
    textarea.addEventListener("keydown", (e) => {
      const isSubmitCombo = (e.ctrlKey || e.metaKey) && e.key === "Enter";
      if (isSubmitCombo) {
        e.preventDefault();
        askQuestion();
      }
    });
    updateCharCount();
    updateAskDisabled();
    // Auto RTL for Arabic input
    textarea.addEventListener("input", () => applyDir(textarea, isRTLText(textarea.value)));
  }

  document.querySelectorAll(".example-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const text = btn.getAttribute("data-example") || "";
      if (textarea) {
        textarea.value = text;
        updateCharCount();
        updateAskDisabled();
        applyDir(textarea, isRTLText(text));
      }
      askQuestion();
    });
  });

  const copyBtn = document.getElementById("copy-answer");
  if (copyBtn) {
    copyBtn.addEventListener("click", async () => {
      const answerNode = document.getElementById("answer");
      const text = answerNode?.textContent || "";
      if (!text) return;
      try {
        if (navigator.clipboard?.writeText) {
          await navigator.clipboard.writeText(text);
          copyBtn.textContent = "✅ Copié";
          setTimeout(() => (copyBtn.textContent = "📋 Copier"), 1200);
        } else {
          const area = document.createElement("textarea");
          area.value = text; document.body.appendChild(area); area.select();
          document.execCommand("copy"); document.body.removeChild(area);
        }
      } catch (_) {
        // ignore
      }
    });
  }

  const themeToggle = document.getElementById("theme-toggle");
  const savedTheme = localStorage.getItem("tunispeak.theme");
  if (savedTheme) {
    document.body.setAttribute("data-theme", savedTheme);
  }
  if (themeToggle) {
    themeToggle.addEventListener("click", () => {
      const current = document.body.getAttribute("data-theme") === "dark" ? "light" : "dark";
      document.body.setAttribute("data-theme", current);
      localStorage.setItem("tunispeak.theme", current);
      themeToggle.textContent = current === "dark" ? "☀️" : "🌙";
    });
    // initialize icon
    themeToggle.textContent = (document.body.getAttribute("data-theme") === "dark") ? "☀️" : "🌙";
  }
}

initializeSpeech();
initializeUX();
