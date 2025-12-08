"""Streamlit interface for the TuniSpeak QA assistant."""
from __future__ import annotations

import base64
import os
import queue
from collections import deque
from io import BytesIO
from typing import Any

import httpx
import numpy as np
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
        # Convert to 16-bit PCM for consistent encoding.
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
        except Exception:  # noqa: BLE001
            payload["audio_bytes"] = None
    else:
        payload["audio_bytes"] = None
    return payload


def send_feedback(url: str, payload: dict[str, Any]) -> None:
    with httpx.Client(timeout=15) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        st.info("Aucune source fournie.")
        return
    for source in sources:
        title = source.get("title") or source.get("document_id", "Source")
        score = source.get("score")
        snippet = source.get("snippet") or source.get("text")
        cols = st.columns([3, 1])
        with cols[0]:
            st.markdown(f"**{title}**")
            if snippet:
                st.write(snippet)
        with cols[1]:
            st.caption(f"Score : {score:.3f}" if isinstance(score, (int, float)) else "Score : N/A")
        st.divider()
st.set_page_config(page_title="TuniSpeak", page_icon="💬", layout="wide")
st.title("TuniSpeak – Interface Streamlit")
st.caption("Assistant Q/R trilingue • Français · Arabe · Darija")

with st.sidebar:
    st.header("Configuration")
    api_host = st.text_input(
        "FastAPI URL",
        value=DEFAULT_API_URL,
        help="Point d'accès FastAPI (sans /api/v1)",
    )
    top_k = st.slider("Nombre de sources (top_k)", min_value=1, max_value=10, value=5)
    st.markdown("---")
    st.subheader("Exemples")
    selected_example = st.radio("Choisir un exemple", EXAMPLES, index=0)
    if st.button("Utiliser l'exemple"):
        st.session_state["question"] = selected_example

api_base = build_api_base(api_host)
QA_ENDPOINT = f"{api_base}/qa/answer"
FEEDBACK_ENDPOINT = f"{api_base}/feedback"
VOICE_ENDPOINT = f"{api_base}/qa/voice"

tab_text, tab_voice = st.tabs(["Mode texte", "Mode vocal (bêta)"])

with tab_text:
    question = st.text_area(
        "Posez votre question",
        key="question",
        placeholder="Ex: Comment demander une attestation d'inscription ?",
        height=140,
    )

    if st.button("Obtenir une réponse", type="primary"):
        if not question or len(question.strip()) < 3:
            st.warning("Veuillez formuler une question plus précise (≥ 3 caractères).")
        else:
            with st.spinner("Recherche en cours..."):
                try:
                    payload = {"question": question.strip(), "top_k": top_k}
                    result = call_api(QA_ENDPOINT, payload)
                    st.session_state["result"] = result
                    st.session_state["feedback_sent"] = False
                except httpx.HTTPStatusError as exc:
                    st.error(f"Erreur API ({exc.response.status_code}) : {exc.response.text}")
                except httpx.RequestError as exc:
                    st.error(f"Connexion impossible : {exc}")

    result = st.session_state.get("result")

    if result:
        st.subheader("Réponse")
        badge_cols = st.columns(3)
        badge_cols[0].metric("Langue détectée", result.get("language", "-"))
        confidence = result.get("confidence")
        badge_cols[1].metric(
            "Confiance",
            f"{confidence*100:.1f}%" if isinstance(confidence, (int, float)) else "-",
        )
        normalized = result.get("normalized_query")
        badge_cols[2].metric("Requête normalisée", normalized or "—")

        st.success(result.get("answer") or "Aucune réponse")
        if result.get("abstain"):
            st.warning(result.get("reason") or "Réponse fournie avec prudence.")

        with st.expander("Sources", expanded=True):
            render_sources(result.get("sources", []))

        interaction_id = result.get("interaction_id")
        if interaction_id:
            st.divider()
            st.subheader("Feedback")
            rating = st.radio(
                "La réponse vous a-t-elle aidé ?",
                ("helpful", "unhelpful"),
                format_func=lambda x: "👍 Oui" if x == "helpful" else "👎 Non",
                horizontal=True,
            )
            comment = st.text_area("Commentaire (optionnel)", key="feedback_comment", height=80)
            correction = st.text_area("Réponse attendue (optionnel)", key="feedback_correction", height=80)
            if st.button("Envoyer le feedback") and not st.session_state.get("feedback_sent"):
                payload = {
                    "interaction_id": interaction_id,
                    "rating": rating,
                    "comment": comment.strip() or None,
                    "correction": correction.strip() or None,
                }
                with st.spinner("Envoi du feedback..."):
                    try:
                        send_feedback(FEEDBACK_ENDPOINT, payload)
                        st.success("Merci ! Votre feedback a été enregistré.")
                        st.session_state["feedback_sent"] = True
                    except httpx.HTTPStatusError as exc:
                        st.error(f"Erreur API ({exc.response.status_code}) : {exc.response.text}")
                    except httpx.RequestError as exc:
                        st.error(f"Connexion impossible : {exc}")

with tab_voice:
    ensure_voice_state()

    st.subheader("Communication vocale")
    st.write(
        "Cliquez sur *Start* puis posez votre question. Lorsque vous avez terminé,"
        " utilisez le bouton ci-dessous pour envoyer la capture audio."
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

    st.caption(
        f"Durée capturée : {_voice_duration_seconds():.1f} s"
        f" · État WebRTC : {webrtc_ctx.state}"
    )
    if webrtc_ctx.state.playing and not webrtc_ctx.audio_receiver:
        st.info("Connexion audio en cours… autorisez le micro dans votre navigateur.")

    col_left, col_right = st.columns(2)
    with col_left:
        if st.button("Réinitialiser la capture", disabled=not _has_voice_audio()):
            _clear_voice_buffer()
            st.info("Capture audio effacée.")

    with col_right:
        disabled = not _has_voice_audio()
        if st.button("Envoyer la question audio", type="primary", disabled=disabled):
            audio_bytes = _flush_voice_buffer()
            if not audio_bytes:
                st.warning("Aucune capture audio détectée.")
            else:
                with st.spinner("Transcription et réponse en cours..."):
                    try:
                        voice_result = call_voice_api(VOICE_ENDPOINT, audio_bytes, top_k)
                        history: list[dict[str, Any]] = st.session_state.voice_history
                        history.insert(0, voice_result)
                        del history[MAX_VOICE_HISTORY:]
                    except httpx.HTTPStatusError as exc:
                        st.error(f"Erreur API ({exc.response.status_code}) : {exc.response.text}")
                    except httpx.RequestError as exc:
                        st.error(f"Connexion impossible : {exc}")

    history = st.session_state.voice_history
    if history:
        st.markdown("---")
        st.subheader("Historique vocal")
        for entry in history:
            st.markdown(f"**Vous :** {entry.get('question', '')}")
            confidence = entry.get("confidence")
            if isinstance(confidence, (int, float)):
                st.caption(f"Confiance : {confidence*100:.1f}%")
            st.markdown(f"**TuniSpeak :** {entry.get('answer', '')}")
            if entry.get("abstain"):
                st.warning(entry.get("reason") or "Réponse fournie avec prudence.")
            audio_bytes = entry.get("audio_bytes")
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3")
            if entry.get("sources"):
                with st.expander("Sources", expanded=False):
                    render_sources(entry.get("sources", []))
            st.divider()

st.markdown("---")
st.caption("Déployé avec Streamlit · Backend FastAPI TuniSpeak")
