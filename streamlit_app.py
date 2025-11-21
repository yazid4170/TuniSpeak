"""Streamlit interface for the TuniSpeak QA assistant."""
from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

DEFAULT_API_URL = os.getenv("TUNISPEAK_API_URL", "http://127.0.0.1:8000")
EXAMPLES = [
    "Quels sont les délais pour l'inscription tardive ?",
    "Comment obtenir une attestation de réussite ?",
    "شنو يلزمني باش نبدل الشعبة؟",
    "ما هي الوثائق اللازمة للتسجيل؟",
]


def build_api_base(raw_url: str) -> str:
    base = raw_url.rstrip("/") or DEFAULT_API_URL.rstrip("/")
    return f"{base}/api/v1"


def call_api(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    with httpx.Client(timeout=30) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        return response.json()


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
    api_host = st.text_input("FastAPI URL", value=DEFAULT_API_URL, help="Point d'accès FastAPI (sans /api/v1)")
    top_k = st.slider("Nombre de sources (top_k)", min_value=1, max_value=10, value=5)
    st.markdown("---")
    st.subheader("Exemples")
    selected_example = st.radio("Choisir un exemple", EXAMPLES, index=0)
    if st.button("Utiliser l'exemple"):
        st.session_state["question"] = selected_example

api_base = build_api_base(api_host)
QA_ENDPOINT = f"{api_base}/qa/answer"
FEEDBACK_ENDPOINT = f"{api_base}/feedback"

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
    badge_cols[1].metric("Confiance", f"{confidence*100:.1f}%" if isinstance(confidence, (int, float)) else "-")
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
        rating = st.radio("La réponse vous a-t-elle aidé ?", ("helpful", "unhelpful"), format_func=lambda x: "👍 Oui" if x == "helpful" else "👎 Non", horizontal=True)
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

st.markdown("---")
st.caption("Déployé avec Streamlit · Backend FastAPI TuniSpeak")
