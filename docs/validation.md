# TuniSpeak Validation Briefing

This document condenses the work completed so far so you can confidently walk through the project with your teacher. It is organised into three sections:

1. End-to-end system view
2. Implementation timeline (milestones to reach the Gold phase)
3. Validation checklist (artifacts, tests, and demo steps)

---

## 1. End-to-End System View

### Functional Scope
- **Goal**: Trilingual question-answering assistant (French, Arabic, Darija) for student services.
- **Modalities**: Text chat + full voice loop (speech-to-text + text-to-speech).
- **Knowledge Base**: University regulations, FAQs, procedures loaded from PDF/Word/text files.
- **Outputs**: Natural language answers, supporting citations, optional audio playback, feedback capture.

### High-Level Pipeline
1. **Document ingestion** (`scripts/ingest_documents.py`)
   - Loads raw files from `data/raw/`, normalises text (Unicode cleanup + language hints), splits into overlapping chunks, and writes `data/processed/chunks.jsonl`.
   - Stores metadata (document id, source path, language tag) for transparency.

2. **Hybrid retrieval index** (`app/retrieval/hybrid.py` + `scripts/build_hybrid_index.py`)
   - Sparse side: BM25 using `rank_bm25` over chunk texts.
   - Dense side: Sentence Transformer embeddings (`app/services/embedding.py`).
   - Combines BM25 + dense scores; optional cross-encoder reranking (`app/services/reranker.py`).

3. **Question answering service** (`app/services/qa.py`)
   - Detects language (`app/services/language.py` → fastText light model).
   - Normalises Darija variants (`app/utils/normalization.py`).
   - Runs retrieval (sparse + dense + rerank) to get top-k contexts.
   - Generation route: hybrid pipeline feeds a fine-tuned generative model (currently LoRA-adapted LLaMA) or falls back to extractive QA (fine-tuned DistilBERT) when needed.
   - Returns answer text, confidence, language tag, retrieved sources, and abstains when confidence < threshold.

4. **Voice loop** (`app/services/speech.py`)
   - Faster-Whisper for streaming transcription.
   - gTTS for text-to-speech.
   - Endpoint `/api/v1/qa/voice` handles audio upload, returns transcript, answer, base64 audio.

5. **API surface** (FastAPI in `app/api/routes.py`)
   - `/api/v1/qa/answer`: text QA.
   - `/api/v1/qa/voice`: voice QA.
   - `/api/v1/feedback`: persistence of user feedback (`app/services/feedback.py`).
   - `/api/v1/metrics`: surfaces evaluation metrics for dashboards.

6. **Frontend** (`frontend/`)
   - Minimal text UI (current state) with textarea + speech controls + feedback form.
   - Voice-first UI iterations archived in git history; currently reverted to classic layout for clarity.

7. **Evaluation tooling** (`app/evaluation/` + `scripts/evaluate_qa.py`)
   - Dataset loader wraps dev/test splits from `data/processed/eval/`.
   - Metrics: EM, F1, Recall@k, nDCG, calibration (ECE/Brier). Results persisted to `models/*/eval_results.json`.

### Data & Config Management
- `pyproject.toml`: Poetry-managed dependencies (FastAPI, torch, sentence-transformers, faster-whisper, gTTS, etc.).
- `app/core/config.py`: Centralised environment variables (paths, model ids, thresholds).
- Model assets stored under `models/` (`llama-lora-*`, `base/tinyllama` checkpoints).

---

## 2. Implementation Timeline (Gold Phase Milestones)

| Milestone | Description | Key Files |
|-----------|-------------|-----------|
| **Initial ingestion & retrieval** | Built document ingestion, chunking, and hybrid BM25 + dense retrieval. | `scripts/ingest_documents.py`, `app/retrieval/hybrid.py`, `app/services/embedding.py` |
| **QA baseline** | Added extractive QA service using a pre-trained multilingual encoder. | `app/services/qa.py`, `app/models/schemas.py` |
| **Voice integration** | Implemented speech-to-text and text-to-speech endpoints, added voice route and service. | `app/services/speech.py`, `app/api/routes.py`, `app/services/qa.py` |
| **Frontend iterations** | Modern voice-first UI with glassmorphism → reverted to teacher-friendly classic layout. | `frontend/index.html`, `frontend/styles.css`, `frontend/app.js` |
| **Fine-tuning QA encoder** | Created script to adapt QA model on in-domain FAQ dataset (SQuAD format). | `scripts/train_qa_model.py`, `models/llama-lora-*/all_results.json` |
| **LoRA fine-tuning for generation** | LoRA adapters on LLaMA for domain-specific answers; training utilities + multi-run experiment folders. | `scripts/train_llama_lora.py`, `models/llama-lora-*/` |
| **Evaluation suite** | Added calibration metrics (ECE, Brier), evaluation runner, FastAPI metrics endpoint, pytest coverage. | `app/evaluation/metrics.py`, `app/evaluation/runner.py`, `tests/api/test_feedback_and_metrics.py` |
| **Feedback loop** | Feedback API + persistence (`data/feedback/feedback.jsonl`), integration in frontend. | `app/services/feedback.py`, `frontend/app.js` |

Outcome: We satisfy the “Gold” spec: hybrid retrieval + reranking, fine-tuned multilingual QA encoder, domain adaptation for generation, calibration metrics, and guardrails.

---

## 3. Validation Checklist

### Artifacts to Show Your Teacher
- **System architecture**: Present this document and `docs/ARCHITECTURE.md` (if updated) to explain the pipeline.
- **API proof**: Run `poetry run uvicorn app.main:app --reload` and use Swagger UI at `http://127.0.0.1:8000/docs` to show endpoints.
- **Frontend demo**: Open `frontend/index.html` via the local server; demonstrate text question, voice dictation, and feedback submission.
- **Model cards**: Review `docs/MODEL_CARD.md` and `docs/DATA_CARD.md` for summary of datasets + models.
- **Evaluation results**: Share latest `evaluate_qa.py` outputs and `models/*/eval_results.json` files.
- **Feedback log**: `data/feedback/feedback.jsonl` to highlight user feedback capture.

### Tests & Quality Gates
- **Unit tests**: `poetry run pytest` (covers API feedback, metrics computations, QA span selection logic, etc.).
- **Lint/format**: `ruff` + `mypy` (configs in `ruff.toml`, `mypy.ini`).
- **Manual voice test**: With the API running, hit `/api/v1/qa/voice` using curl or the frontend.
- **Calibration sanity**: Inspect metrics endpoint, ensure ECE/Brier < 0.2 (example numbers from latest run).

### Talking Points for Q&A with Teacher
- **Why hybrid retrieval?** Sparse BM25 handles keyword-heavy French legal wording; dense embeddings catch paraphrases/Arabic transliteration. We fuse scores + rerank for best of both worlds.
- **Language handling**: Lightweight detection (fastText) routes queries; Darija normalisation strips Latin/Arabic diacritics so embeddings stay stable.
- **Voice loop**: Faster-Whisper chosen for quality vs. latency; gTTS ensures we can answer in the same language the user spoke.
- **Fine-tuning**: QA encoder adapted on internal FAQ to boost EM/F1; LoRA adapters avoid full model retrain while capturing domain phrasing.
- **Calibration & Abstention**: Confidence derived from retrieval/generator logits; we compute ECE/Brier and when low we surface an abstain notice rather than a hallucinated answer.
- **Privacy & ethics**: Documents remain on-premise, no third-party API calls beyond optional Hugging Face downloads; we log feedback without personal identifiers.

### Ready for Platinum?
To move forward: choose student model, implement teacher→student distillation, integrate calibrated refusal thresholds for the student, document improvements.

---

**Next Steps**
- Review this sheet with your teacher.
- Gather feedback, prioritise remaining gaps before jumping to Platinum.
- Once approved, start the distillation plan outlined in our prior conversation.
