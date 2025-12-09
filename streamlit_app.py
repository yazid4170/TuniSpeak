"""Streamlit interface for the TuniSpeak QA assistant."""
from __future__ import annotations

import base64
import html
import os
import queue
import time
from collections import deque
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

DEFAULT_API_URL = os.getenv("TUNISPEAK_API_URL", "http://127.0.0.1:8000")
EXAMPLES = [
    "Quels sont les délais pour l'inscription tardive ?",
    "Comment obtenir une attestation de réussite ?",
    "شنو يلزمني باش نبدل الشعبة؟",
    "ما هي الوثائق اللازمة للتسجيل؟",
]
RTC_CONFIGURATION = {"iceServers": []}
MAX_VOICE_HISTORY = 5
MAX_VOICE_SECONDS = 30
DEFAULT_COMPLEXITY = "Bronze"
COMPLEXITY_LEVELS: dict[str, dict[str, Any]] = {
    "Bronze": {
        "label": "🥉 Bronze",
        "pipeline": "TF-IDF + QA extractif",
        "description": "Baseline end-to-end avec réponse concise et feedback rapide.",
        "top_k": 3,
        "show_metrics": False,
        "show_normalized": False,
        "show_sources": False,
        "max_sources": 0,
        "show_feedback": True,
        "voice_enabled": False,
    },
    "Silver": {
        "label": "🥈 Silver",
        "pipeline": "Dense retrieval + reranking",
        "description": "Ajoute reranking dense et feedback contextualisé.",
        "top_k": 5,
        "show_metrics": True,
        "show_normalized": False,
        "show_sources": True,
        "max_sources": 3,
        "show_feedback": True,
        "voice_enabled": False,
    },
    "Gold": {
        "label": "🥇 Gold",
        "pipeline": "Encodeur multilingue fine-tuné",
        "description": "Adaptation domaine + métriques complètes et feedback.",
        "top_k": 7,
        "show_metrics": True,
        "show_normalized": True,
        "show_sources": True,
        "max_sources": None,
        "show_feedback": True,
        "voice_enabled": False,
    },
    "Platinum": {
        "label": "💎 Platinum",
        "pipeline": "RAG distillé + calibration",
        "description": "Expérience complète avec calibration, abstention et voix.",
        "top_k": 10,
        "show_metrics": True,
        "show_normalized": True,
        "show_sources": True,
        "max_sources": None,
        "show_feedback": True,
        "voice_enabled": True,
    },
}


def render_complexity_static_overview(selected_level: str) -> None:
    rows: list[dict[str, Any]] = []
    for key, cfg in COMPLEXITY_LEVELS.items():
        rows.append(
            {
                "Niveau": cfg["label"],
                "Pipeline": cfg.get("pipeline", "—"),
                "Top-k": cfg["top_k"],
                "Sources": "✅" if cfg["show_sources"] else "—",
                "Métriques": "✅" if cfg["show_metrics"] else "—",
                "Feedback": "✅" if cfg["show_feedback"] else "—",
                "Voix": "✅" if cfg["voice_enabled"] else "—",
                "Description": cfg["description"],
            }
        )

    df = pd.DataFrame(rows, index=COMPLEXITY_LEVELS.keys())
    df.index.name = "Mode"

    styled = df.style.apply(
        lambda row: [
            "background-color: rgba(16, 58, 113, 0.12)" if row.name == selected_level else ""
            for _ in row
        ],
        axis=1,
    )

    st.dataframe(styled, use_container_width=True)


# Execute the QA endpoint for a specific complexity mode and collect timing metadata.
def evaluate_complexity_level(
    qa_url: str,
    question: str,
    level_key: str,
    config: dict[str, Any],
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    payload = {
        "question": question,
        "top_k": config.get("top_k", 5),
    }
    if client is None:
        response = call_api(qa_url, payload)
    else:
        http_response = client.post(qa_url, json=payload)
        http_response.raise_for_status()
        response = http_response.json()
    latency_ms = (time.perf_counter() - start) * 1000
    return {
        "ok": True,
        "data": response,
        "latency_ms": latency_ms,
        "used_top_k": payload["top_k"],
        "level": level_key,
    }

# Render a comparison grid showing live metrics for every complexity level.
def render_complexity_metrics(
    results: dict[str, dict[str, Any]] | None,
    selected_level: str,
    available_only: bool = False,
) -> None:
    if not results:
        st.info("Posez une question pour comparer les niveaux de complexité.")
        return

    rows: list[dict[str, Any]] = []
    keys = results.keys() if available_only else COMPLEXITY_LEVELS.keys()
    for key in keys:
        cfg = COMPLEXITY_LEVELS[key]
        entry = (results or {}).get(key)
        base_row: dict[str, Any] = {
            "Niveau": cfg["label"],
            "Pipeline": cfg.get("pipeline", "—"),
            "Top-k": (entry or {}).get("used_top_k", cfg.get("top_k")),
            "Sélection": "⭐" if key == selected_level else "",
        }
        if not entry:
            base_row.update(
                {
                    "Confiance": "—",
                    "Sources": "—",
                    "Langue": "—",
                    "Abstention": "—",
                    "Latence (ms)": "—",
                    "Longueur": "—",
                    "Statut": "En attente",
                }
            )
        elif not entry.get("ok"):
            base_row.update(
                {
                    "Confiance": "—",
                    "Sources": "—",
                    "Langue": "—",
                    "Abstention": "—",
                    "Latence (ms)": "—",
                    "Longueur": "—",
                    "Statut": entry.get("error", "Erreur"),
                }
            )
        else:
            data = entry.get("data", {})
            sources = data.get("sources") or []
            confidence = data.get("confidence")
            conf_display = (
                f"{confidence * 100:.1f}%"
                if isinstance(confidence, (int, float))
                else "N/A"
            )
            latency_ms = entry.get("latency_ms")
            latency_display = (
                f"{latency_ms:.0f}"
                if isinstance(latency_ms, (int, float))
                else "N/A"
            )
            base_row.update(
                {
                    "Confiance": conf_display,
                    "Sources": len(sources),
                    "Langue": (data.get("language") or "-").upper(),
                    "Abstention": "✅" if data.get("abstain") else "—",
                    "Latence (ms)": latency_display,
                    "Longueur": len((data.get("answer") or "").strip()),
                    "Statut": "OK",
                }
            )
        rows.append(base_row)

    df = pd.DataFrame(rows, index=list(keys))
    df.index.name = "Mode"

    styled = df.style.apply(
        lambda row: [
            "background-color: rgba(16, 58, 113, 0.12)" if row.name == selected_level else ""
            for _ in row
        ],
        axis=1,
    )

    st.dataframe(styled, use_container_width=True)


def ensure_voice_state() -> None:
    if "voice_frames" not in st.session_state:
        st.session_state.voice_frames: deque[np.ndarray] = deque()
        st.session_state.voice_total_samples = 0
        st.session_state.voice_sample_rate = 16000
    if "voice_history" not in st.session_state:
        st.session_state.voice_history = []


def _append_audio_frame(frame) -> None:
    ensure_voice_state()
    samples = frame.to_ndarray()
    if samples.dtype != np.int16:
        samples = (samples * np.iinfo(np.int16).max).astype(np.int16)
    st.session_state.voice_frames.append(samples)
    st.session_state.voice_total_samples += samples.shape[1]
    st.session_state.voice_sample_rate = frame.sample_rate
    _trim_voice_buffer()


def _trim_voice_buffer() -> None:
    limit = int(st.session_state.voice_sample_rate * MAX_VOICE_SECONDS)
    while st.session_state.voice_total_samples > limit and st.session_state.voice_frames:
        removed = st.session_state.voice_frames.popleft()
        st.session_state.voice_total_samples -= removed.shape[1]


def _clear_voice_buffer() -> None:
    st.session_state.voice_frames.clear()
    st.session_state.voice_total_samples = 0


def _voice_duration_seconds() -> float:
    rate = st.session_state.voice_sample_rate or 1
    return st.session_state.voice_total_samples / float(rate)


def _flush_voice_buffer() -> bytes:
    if not st.session_state.voice_frames:
        return b""
    audio = np.concatenate(list(st.session_state.voice_frames), axis=1)
    _clear_voice_buffer()
    samples = audio.T.astype(np.int16, copy=False)
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    buffer = BytesIO()
    sf.write(
        buffer,
        samples,
        st.session_state.voice_sample_rate,
        format="wav",
        subtype="PCM_16",
    )
    buffer.seek(0)
    return buffer.read()


def _has_voice_audio() -> bool:
    return bool(st.session_state.voice_total_samples)


def build_api_base(raw_url: str) -> str:
    base = raw_url.rstrip("/") or DEFAULT_API_URL.rstrip("/")
    return f"{base}/api/v1"


def call_api(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    with httpx.Client(timeout=30) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        return response.json()


def call_voice_api(url: str, audio_bytes: bytes, top_k: int) -> dict[str, Any]:
    files = {"file": ("query.wav", audio_bytes, "audio/wav")}
    params = {"top_k": top_k}
    with httpx.Client(timeout=90) as client:
        response = client.post(url, params=params, files=files)
        response.raise_for_status()
        payload = response.json()

    audio_b64 = payload.get("audio_base64")
    if audio_b64:
        try:
            payload["audio_bytes"] = base64.b64decode(audio_b64)
        except Exception:
            payload["audio_bytes"] = None
    else:
        payload["audio_bytes"] = None
    return payload


def send_feedback(url: str, payload: dict[str, Any]) -> None:
    with httpx.Client(timeout=15) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()


def render_feedback_form(endpoint: str, interaction_id: str, form_key: str) -> None:
    if not interaction_id:
        st.info("Identifiant de l'interaction indisponible pour le feedback.")
        return

    submissions: dict[str, bool] = st.session_state.setdefault("feedback_submissions", {})
    if submissions.get(interaction_id):
        st.success("✅ Merci pour votre retour !")
        return

    rating_options = ("helpful", "unhelpful")
    prefix = form_key.replace(" ", "-")
    form_id = f"feedback-form-{prefix}-{interaction_id}"

    with st.form(form_id):
        rating = st.radio(
            "Cette réponse vous a-t-elle aidé ?",
            rating_options,
            index=0,
            format_func=lambda x: "👍 Oui" if x == "helpful" else "👎 Non",
            key=f"radio-{form_id}",
        )
        comment = st.text_area(
            "Commentaire (optionnel)",
            height=80,
            key=f"comment-{form_id}",
        )
        correction = st.text_area(
            "Suggestion de réponse (optionnel)",
            height=80,
            key=f"correction-{form_id}",
        )
        submitted = st.form_submit_button("📤 Envoyer le feedback", use_container_width=True)

    if submitted:
        payload = {
            "interaction_id": interaction_id,
            "rating": rating,
            "comment": comment.strip() or None,
            "correction": correction.strip() or None,
        }
        try:
            with st.spinner("Envoi en cours..."):
                send_feedback(endpoint, payload)
            submissions[interaction_id] = True
            st.success("✅ Merci pour votre retour !")
        except httpx.HTTPStatusError as exc:
            st.error(f"❌ Erreur: {exc.response.text}")
        except httpx.RequestError as exc:
            st.error(f"❌ Connexion impossible: {exc}")


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        st.info("Aucune source fournie.")
        return
    for idx, source in enumerate(sources, 1):
        title = source.get("title") or source.get("document_id", "Source")
        score = source.get("score")
        snippet = source.get("snippet") or source.get("text") or ""
        document_id = source.get("document_id")
        url = source.get("url")

        snippet_html = html.escape(snippet).replace("\n", "<br>")
        title_html = html.escape(str(title))
        document_html = html.escape(str(document_id)) if document_id else ""
        score_html = (
            f'<div class="source-score">Score: {score:.3f}</div>'
            if isinstance(score, (int, float))
            else ""
        )
        url_html = (
            f'<a href="{html.escape(url)}" target="_blank" class="source-link">Ouvrir le document</a>'
            if url
            else ""
        )

        st.markdown(
            f"""
            <div class="source-card">
                <div class="source-header">
                    <div class="source-index">{idx}</div>
                    <div class="source-meta">
                        <div class="source-title">{title_html}</div>
                        {'<div class="source-doc">' + document_html + '</div>' if document_html else ''}
                        {url_html}
                    </div>
                    {score_html}
                </div>
                <div class="source-snippet">{snippet_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


CUSTOM_CSS = """
<style>
/* Global Variables */
:root {
    --primary-blue: #103a71;
    --primary-hover: #0c2d59;
    --accent-gold: #f2b441;
    --text-dark: #0b1731;
    --text-medium: #1f2e4a;
    --text-light: #4a5c7f;
    --border-color: rgba(11, 23, 49, 0.18);
    --bg-light: #f5f7fb;
    --bg-shell: #e6ecf7;
    --success-green: #10b981;
    --warning-orange: #f2b441;
    --error-red: #ef4444;
}

/* Base Styles */
.stApp {
    background-color: var(--bg-light);
}

body, .stApp, .stApp p, .stApp span, .stApp label {
    color: var(--text-dark);
}

.stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown li {
    color: var(--text-dark) !important;
}

.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
    color: var(--text-dark) !important;
}

/* Streamlit native header */
[data-testid="stHeader"] {
    background: linear-gradient(90deg, #103a71 0%, #f2f4fb 100%);
    border-bottom: 1px solid var(--border-color);
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.07);
}

[data-testid="stHeader"] * {
    color: var(--text-dark) !important;
}

/* Header */
.app-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 16px 0;
    border-bottom: 1px solid var(--border-color);
    margin-bottom: 32px;
}

.logo-img {
    width: 200px;
    height: 200px;
    object-fit: contain;
}

.header-text h1 {
    margin: 0;
    font-size: 24px;
    font-weight: 600;
    color: var(--text-dark);
    line-height: 1.2;
}

.header-text p {
    margin: 0;
    font-size: 14px;
    color: var(--text-medium);
}

/* Hero Section */
.hero-section {
    text-align: center;
    padding: 48px 0;
    max-width: 800px;
    margin: 0 auto;
    color: var(--text-dark);
}

.hero-section-title {
    font-size: 20px;
    font-weight: 600;
    color: var(--text-dark) !important;
    margin-bottom: 0.6rem;
}

.hero-section h2 {
    font-size: 36px;
    font-weight: 700;
    color: var(--text-dark);
    margin-bottom: 16px;
    line-height: 1.3;
}

.hero-section p {
    font-size: 18px;
    color: var(--text-medium);
    line-height: 1.6;
    margin-bottom: 32px;
}

.section-title {
    font-size: 20px;
    font-weight: 600;
    color: var(--text-dark) !important;
    margin-bottom: 0.6rem;
}

/* Stats Bar */
.stats-container {
    display: flex;
    gap: 16px;
    justify-content: center;
    margin-bottom: 48px;
}

.stat-box {
    background: #fff;
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 20px 32px;
    text-align: center;
    min-width: 140px;
}

.stat-value {
    display: block;
    font-size: 28px;
    font-weight: 700;
    color: var(--primary-blue);
    margin-bottom: 4px;
}

.stat-label {
    display: block;
    font-size: 13px;
    color: var(--text-medium);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Settings Panel */
.settings-panel {
    background: var(--bg-light);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 32px;
}

/* Example Chips */
.examples-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 8px;
    margin-top: 8px;
}

/* Answer Card */
.answer-box {
    background: rgba(16, 58, 113, 0.08);
    border-left: 4px solid var(--primary-blue);
    border-radius: 8px;
    padding: 20px;
    margin: 24px 0;
    font-size: 16px;
    line-height: 1.7;
    color: var(--text-dark);
}

.source-card {
    background: #fff;
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 16px;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
}

.source-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 10px;
}

.source-index {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: var(--primary-blue);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    font-size: 14px;
    flex-shrink: 0;
}

.source-meta {
    flex: 1;
}

.source-title {
    font-weight: 600;
    font-size: 16px;
    color: var(--text-dark);
    margin-bottom: 2px;
}

.source-doc {
    font-size: 12px;
    color: var(--text-light);
    letter-spacing: 0.4px;
    text-transform: uppercase;
}

.source-score {
    font-size: 13px;
    font-weight: 600;
    color: var(--primary-blue);
    margin-top: 4px;
}

.source-link {
    display: inline-block;
    margin-top: 6px;
    font-size: 12px;
    color: var(--primary-blue);
    text-decoration: none;
}

.source-link:hover {
    text-decoration: underline;
}

.source-snippet {
    background: rgba(16, 58, 113, 0.05);
    border-radius: 8px;
    padding: 12px 14px;
    color: var(--text-dark);
    line-height: 1.6;
    white-space: pre-wrap;
}

/* Metric Pills */
.metric-row {
    display: flex;
    gap: 12px;
    margin-bottom: 20px;
}

.metric-pill {
    flex: 1;
    background: white;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 16px;
    text-align: center;
}

.pill-value {
    display: block;
    font-size: 20px;
    font-weight: 600;
    color: var(--text-dark);
    margin-bottom: 4px;
}

.pill-label {
    display: block;
    font-size: 12px;
    color: var(--text-medium);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 2px solid var(--border-color);
}

.stTabs [data-baseweb="tab"] {
    padding: 12px 24px;
    background: transparent;
    border: none;
    border-bottom: 3px solid transparent;
    color: var(--text-medium);
    font-weight: 500;
}

.stTabs [aria-selected="true"] {
    border-bottom-color: var(--primary-blue);
    color: var(--primary-blue);
}

/* Buttons */
.stButton button {
    border-radius: 999px;
    font-weight: 500;
    padding: 10px 20px;
    transition: all 0.2s;
    background-color: rgba(16, 58, 113, 0.08);
    color: var(--text-dark);
    border: 1px solid rgba(11, 23, 49, 0.25);
}

.stButton button:hover {
    background-color: rgba(16, 58, 113, 0.18);
    border-color: var(--primary-blue);
}

.stButton button[kind="primary"] {
    background: linear-gradient(120deg, var(--primary-blue), var(--accent-gold));
    border: none;
    color: #ffffff;
}

.stButton button[kind="primary"]:hover {
    filter: brightness(0.95);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(16, 58, 113, 0.25);
}

/* Text Inputs */
.stTextInput input, .stTextArea textarea {
    border-radius: 12px;
    border: 1px solid rgba(11, 23, 49, 0.2);
    padding: 12px;
    font-size: 15px;
    background-color: #ffffff;
    color: var(--text-dark);
}

.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: var(--text-light);
    opacity: 0.9;
}

.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--primary-blue);
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
}

/* Sliders */
.stSlider {
    padding: 8px 0;
}

/* Radio Buttons */
.stRadio > div {
    gap: 16px;
}

.stRadio label {
    color: var(--text-dark) !important;
    font-weight: 500;
}

.stTextArea label,
.stTextInput label,
.stSelectbox label,
.stForm label {
    color: var(--text-dark) !important;
    font-weight: 500;
}

small[data-testid="stCaption"],
span[data-testid="stCaption"] {
    color: var(--text-medium) !important;
}

/* Expander */
.streamlit-expanderHeader {
    background-color: var(--bg-light);
    border-radius: 8px;
    font-weight: 500;
}

/* Footer */
.app-footer {
    text-align: center;
    padding: 32px 0;
    border-top: 1px solid var(--border-color);
    margin-top: 64px;
    color: var(--text-light);
    font-size: 14px;
}

/* Responsive */
@media (max-width: 768px) {
    .hero-section h2 {
        font-size: 28px;
    }
    
    .stats-container {
        flex-direction: column;
    }
    
    .stat-box {
        width: 100%;
    }
}

/* Sidebar */
[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, #103a71 0%, #f2ddb6 120%);
    padding: 16px;
}

.sidebar-card {
    background: rgba(255, 255, 255, 0.9);
    border: 1px solid rgba(255, 255, 255, 0.4);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 18px 35px rgba(11, 23, 49, 0.25);
}

.sidebar-card h3 {
    margin: 0 0 4px;
    font-size: 16px;
    color: var(--text-dark);
}

.sidebar-card p {
    margin: 0 0 12px;
    color: var(--text-medium);
    font-size: 13px;
}

/* Lighter buttons */
.stButton button {
    background-color: #f7f7f7;
    border-color: #ddd;
    color: #333;
}

.stButton button:hover {
    background-color: #f2f2f2;
    border-color: #ccc;
}

.stButton button[kind="primary"] {
    background-color: #4CAF50;
    border: none;
    color: #ffffff;
}

.stButton button[kind="primary"]:hover {
    background-color: #3e8e41;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(62, 142, 65, 0.3);
}
</style>
"""


def inject_global_styles() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data
def load_logo_b64() -> str | None:
    logo_path = Path(__file__).resolve().parent / "frontend" / "logo.png"
    if not logo_path.exists():
        return None
    data = logo_path.read_bytes()
    return base64.b64encode(data).decode("utf-8")


# Page Configuration
st.set_page_config(
    page_title="TuniSpeak - Assistant Q&A Multilingue",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_styles()

# Header
logo_b64 = load_logo_b64()
logo_html = f'<img src="data:image/png;base64,{logo_b64}" class="logo-img" alt="TuniSpeak"/>' if logo_b64 else '💬'

st.markdown(
    f"""
    <div class="app-header">
        {logo_html}
        <div class="header-text">
            <h1>TuniSpeak</h1>
            <p>Assistant Q&A pour les universités tunisiennes</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Hero Section
st.markdown(
    """
    <div class="hero-section">
        <h2>Obtenez des réponses fiables instantanément</h2>
        <p>Posez vos questions sur les procédures universitaires en français, arabe ou darija. Notre système vous fournit des réponses vérifiées avec sources et niveau de confiance.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Stats
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        '<div class="stat-box"><span class="stat-value">3</span><span class="stat-label">Langues</span></div>',
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        '<div class="stat-box"><span class="stat-value">< 2s</span><span class="stat-label">Latence</span></div>',
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        '<div class="stat-box"><span class="stat-value">24/7</span><span class="stat-label">Disponible</span></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("### ⚙️ Niveaux de complexité")
complexity_keys = list(COMPLEXITY_LEVELS.keys())
default_complexity = st.session_state.get("complexity_level", DEFAULT_COMPLEXITY)
default_index = (
    complexity_keys.index(default_complexity)
    if default_complexity in complexity_keys
    else 0
)
selected_complexity = st.radio(
    "Sélectionnez le niveau de réponse souhaité",
    options=complexity_keys,
    index=default_index,
    horizontal=True,
    format_func=lambda key: COMPLEXITY_LEVELS[key]["label"],
)
st.session_state["complexity_level"] = selected_complexity
render_complexity_static_overview(selected_complexity)
complexity_config = COMPLEXITY_LEVELS[selected_complexity]
st.caption(f"Mode actif : {complexity_config['label']} · {complexity_config['description']}")

st.markdown("<br>", unsafe_allow_html=True)

# Sidebar examples
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-card">
            <h3>💡 Questions fréquentes</h3>
            <p>Sélectionnez une suggestion pour pré-remplir le champ de question.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    for idx, example in enumerate(EXAMPLES):
        if st.button(example, key=f"sidebar-example-{idx}", use_container_width=True):
            st.session_state["question"] = example

st.markdown("<br>", unsafe_allow_html=True)

# API Setup
api_host = DEFAULT_API_URL
top_k = complexity_config["top_k"]
api_base = build_api_base(api_host)

QA_ENDPOINT = f"{api_base}/qa/answer"
FEEDBACK_ENDPOINT = f"{api_base}/feedback"
VOICE_ENDPOINT = f"{api_base}/qa/voice"

# Main Tabs
tab_text, tab_voice = st.tabs(["📝 Mode Texte", "🎤 Mode Vocal"])

# Text Mode
with tab_text:
    st.markdown("<h3 class='section-title'>Posez votre question</h3>", unsafe_allow_html=True)

    question = st.text_area(
        "Votre question",
        key="question",
        placeholder="Ex: Comment demander une attestation d'inscription ?",
        height=120,
        label_visibility="collapsed",
    )

    answer_clicked = st.button("🔍 Obtenir une réponse", type="primary", use_container_width=True)
    compare_clicked = st.button("📊 Comparer tous les niveaux", use_container_width=True)

    if answer_clicked:
        question_clean = question.strip() if question else ""
        if len(question_clean) < 3:
            st.warning("⚠️ Veuillez formuler une question plus précise (minimum 3 caractères).")
        else:
            with st.spinner(f"{complexity_config['label']} · génération en cours..."):
                results_by_level: dict[str, dict[str, Any]] = {}
                try:
                    entry = evaluate_complexity_level(
                        QA_ENDPOINT,
                        question_clean,
                        selected_complexity,
                        complexity_config,
                    )
                    results_by_level[selected_complexity] = entry
                except httpx.HTTPStatusError as exc:
                    results_by_level[selected_complexity] = {
                        "ok": False,
                        "error": f"HTTP {exc.response.status_code}",
                        "details": exc.response.text,
                    }
                except httpx.RequestError as exc:
                    results_by_level[selected_complexity] = {
                        "ok": False,
                        "error": "Connexion API",
                        "details": str(exc),
                    }

            st.session_state["results_by_level"] = results_by_level
            st.session_state["selected_question"] = question_clean

    if compare_clicked:
        question_clean = question.strip() if question else ""
        if len(question_clean) < 3:
            st.warning("⚠️ Veuillez formuler une question plus précise (minimum 3 caractères).")
        else:
            total_levels = len(COMPLEXITY_LEVELS)
            progress_text = st.empty()
            progress_bar = st.progress(0)
            results_by_level: dict[str, dict[str, Any]] = {}

            with httpx.Client(timeout=30) as shared_client:
                for idx, (level_key, level_cfg) in enumerate(COMPLEXITY_LEVELS.items(), start=1):
                    progress_text.info(
                        f"{level_cfg['label']} · exécution ({idx}/{total_levels})"
                    )
                    try:
                        entry = evaluate_complexity_level(
                            QA_ENDPOINT,
                            question_clean,
                            level_key,
                            level_cfg,
                            client=shared_client,
                        )
                        results_by_level[level_key] = entry
                    except httpx.HTTPStatusError as exc:
                        results_by_level[level_key] = {
                            "ok": False,
                            "error": f"HTTP {exc.response.status_code}",
                            "details": exc.response.text,
                        }
                    except httpx.RequestError as exc:
                        results_by_level[level_key] = {
                            "ok": False,
                            "error": "Connexion API",
                            "details": str(exc),
                        }
                    progress_bar.progress(idx / total_levels)

            progress_bar.empty()
            if any(entry.get("ok") for entry in results_by_level.values()):
                progress_text.success("Comparaison terminée ✅")
            else:
                progress_text.warning("Comparaison terminée avec des erreurs")
            time.sleep(0.4)
            progress_text.empty()

            st.session_state["results_by_level"] = results_by_level
            st.session_state["selected_question"] = question_clean

    results_by_level = st.session_state.get("results_by_level")
    st.markdown("### 📈 Comparaison temps réel")
    if results_by_level:
        last_question = st.session_state.get("selected_question")
        if last_question:
            st.caption(f'Question analysée : "{last_question}"')
    available_only = bool(results_by_level) and len(results_by_level) != len(COMPLEXITY_LEVELS)
    render_complexity_metrics(results_by_level, selected_complexity, available_only=available_only)
    st.markdown("<br>", unsafe_allow_html=True)

    selected_entry = None
    if results_by_level:
        selected_entry = results_by_level.get(selected_complexity)

    result_error = None
    if selected_entry and not selected_entry.get("ok"):
        result_error = selected_entry.get("error")

    result = selected_entry.get("data") if selected_entry and selected_entry.get("ok") else None

    if result_error:
        st.error(f"❌ {result_error}. Consultez les autres niveaux pour comparer les résultats.")
        details = selected_entry.get("details") if selected_entry else None
        if details:
            with st.expander("Détails de l'erreur"):
                st.code(details)

    if result:
        st.markdown("---")
        st.markdown("### 📊 Résultats")
        st.caption(f"{complexity_config['label']} · {complexity_config['description']}")

        lang = result.get("language", "N/A").upper()
        confidence = result.get("confidence")
        normalized_full = result.get("normalized_query") or "—"
        normalized_display = (
            normalized_full[:30] + "..." if len(normalized_full) > 30 else normalized_full
        )

        if complexity_config["show_metrics"]:
            if complexity_config["show_normalized"]:
                metric_col1, metric_col2, metric_col3 = st.columns(3)
            else:
                metric_col1, metric_col2 = st.columns(2)

            with metric_col1:
                st.markdown(
                    f'<div class="metric-pill"><span class="pill-value">{lang}</span><span class="pill-label">Langue détectée</span></div>',
                    unsafe_allow_html=True,
                )

            with metric_col2:
                conf_text = (
                    f"{confidence*100:.1f}%" if isinstance(confidence, (int, float)) else "N/A"
                )
                st.markdown(
                    f'<div class="metric-pill"><span class="pill-value">{conf_text}</span><span class="pill-label">Confiance</span></div>',
                    unsafe_allow_html=True,
                )

            if complexity_config["show_normalized"]:
                with metric_col3:
                    st.markdown(
                        f'<div class="metric-pill"><span class="pill-value" style="font-size: 14px;">{html.escape(normalized_display)}</span><span class="pill-label">Requête normalisée</span></div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.caption(f"Langue détectée : {lang}")
            if isinstance(confidence, (int, float)):
                st.caption(f"Confiance estimée : {confidence*100:.1f}%")

        # Answer
        st.markdown("### 💬 Réponse")
        answer_text = result.get("answer") or "Aucune réponse disponible"
        st.markdown(
            f'<div class="answer-box">{html.escape(answer_text)}</div>',
            unsafe_allow_html=True,
        )

        if result.get("abstain"):
            st.warning(f"⚠️ {result.get('reason', 'Réponse fournie avec prudence.')}")

        sources = result.get("sources", [])
        if complexity_config["show_sources"]:
            sources_to_render = sources
            max_sources = complexity_config.get("max_sources")
            if isinstance(max_sources, int) and max_sources > 0:
                sources_to_render = sources_to_render[:max_sources]
            with st.expander("📚 Sources utilisées", expanded=bool(sources_to_render)):
                if sources_to_render:
                    render_sources(sources_to_render)
                else:
                    st.info("Aucune source disponible pour cette réponse.")
        else:
            st.caption("Les sources détaillées sont disponibles à partir du niveau Silver.")

        interaction_id = result.get("interaction_id")
        if complexity_config["show_feedback"]:
            st.markdown("---")
            st.markdown("### 📝 Votre avis")
            render_feedback_form(
                endpoint=FEEDBACK_ENDPOINT,
                interaction_id=interaction_id,
                form_key=f"text-{selected_complexity}",
            )

# Voice Mode
with tab_voice:
    if not complexity_config["voice_enabled"]:
        st.info(
            "L'assistant vocal est disponible au niveau Platinum. Sélectionnez \"Platinum\" dans les niveaux de complexité pour l'activer."
        )
    else:
        ensure_voice_state()

        st.markdown("### 🎤 Assistant vocal")
        st.info(
            "Cliquez sur **Start** pour commencer l'enregistrement, puis utilisez le bouton ci-dessous pour envoyer votre question."
        )

        webrtc_ctx = webrtc_streamer(
            key="voice-chat",
            mode=WebRtcMode.SENDONLY,
            media_stream_constraints={"audio": True, "video": False},
            async_processing=False,
            rtc_configuration=RTC_CONFIGURATION,
            audio_receiver_size=1024,
        )

        if webrtc_ctx.state.playing and webrtc_ctx.audio_receiver:
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                audio_frames = []
            for audio_frame in audio_frames:
                _append_audio_frame(audio_frame)

        st.caption(f"⏱️ Durée enregistrée: {_voice_duration_seconds():.1f}s")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Réinitialiser", disabled=not _has_voice_audio(), use_container_width=True):
                _clear_voice_buffer()
                st.success("Enregistrement effacé")

        with col2:
            if st.button(
                "🚀 Envoyer",
                type="primary",
                disabled=not _has_voice_audio(),
                use_container_width=True,
            ):
                audio_bytes = _flush_voice_buffer()
                if not audio_bytes:
                    st.warning("Aucun audio détecté")
                else:
                    with st.spinner("Traitement en cours..."):
                        try:
                            voice_result = call_voice_api(VOICE_ENDPOINT, audio_bytes, top_k)
                            history = st.session_state.voice_history
                            history.insert(0, voice_result)
                            del history[MAX_VOICE_HISTORY:]
                            st.success("✅ Réponse reçue !")
                        except httpx.HTTPStatusError as exc:
                            st.error(f"❌ Erreur: {exc.response.text}")
                        except httpx.RequestError as exc:
                            st.error(f"❌ Connexion impossible: {exc}")

        history = st.session_state.voice_history
        if history:
            st.markdown("---")
            st.markdown("### 📜 Historique")
            for idx, entry in enumerate(history):
                with st.container():
                    st.markdown(f"**Vous:** {entry.get('question', '')}")

                    confidence = entry.get("confidence")
                    if isinstance(confidence, (int, float)):
                        st.caption(f"Confiance: {confidence*100:.1f}%")

                    st.markdown(f"**TuniSpeak:** {entry.get('answer', '')}")

                    if entry.get("abstain"):
                        st.warning(f"⚠️ {entry.get('reason', '')}")

                    audio_bytes = entry.get("audio_bytes")
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3")

                    if entry.get("sources"):
                        with st.expander("📚 Sources", expanded=False):
                            render_sources(entry.get("sources", []))

                    if entry.get("interaction_id"):
                        with st.expander("📝 Donner votre avis", expanded=False):
                            render_feedback_form(
                                endpoint=FEEDBACK_ENDPOINT,
                                interaction_id=entry.get("interaction_id"),
                                form_key=f"voice-{idx}",
                            )

                    if idx < len(history) - 1:
                        st.markdown("---")

# Footer
st.markdown(
    """
    <div class="app-footer">
        <p>TuniSpeak by FazaAI © 2025 · Propulsé par FastAPI et Streamlit</p>
        <p style="font-size: 12px; margin-top: 8px;">Assistance multilingue pour les étudiants tunisiens</p>
    </div>
    """,
    unsafe_allow_html=True,
)