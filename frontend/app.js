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
    question: questionInput.value,
    top_k: 5,
  };

  try {
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
      const item = document.createElement("li");
      const scoreLabel =
        typeof source.score === "number" ? source.score.toFixed(3) : "N/A";
      item.textContent = `${source.title || source.document_id} – score: ${scoreLabel}`;
      sourcesNode.appendChild(item);
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

initializeSpeech();
