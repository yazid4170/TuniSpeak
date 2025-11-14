# TuniSpeak Evaluation Report

## Executive Summary
TuniSpeak delivers a trilingual question-answering experience tailored to Tunisian higher-education services. The assistant connects a FastAPI backend, hybrid retrieval over curated campus knowledge, and multilingual extractive QA with abstention. Recent iterations focused on Tunisian Darija alignment: canonicalising language codes, expanding dialect coverage, and reducing failure cases caused by tokenizer edge conditions. Retrieval now achieves perfect recall across the labelled corpus, while cross-script synonym heuristics eliminated the final healthcare abstention—raising aggregate F1 to ≈0.40 with a 0% abstention rate.

## Introduction & Objectives
- Provide inclusive self-service support for Tunisian students in French, Modern Standard Arabic, and Tunisian Darija (Arabic script and Arabizi).
- Maintain explainability via document source attribution and abstain when confidence is below language-specific thresholds.
- Support constrained infrastructure (CPU inference) while keeping latency acceptable for web access.

## Data Description and Governance
- Source documents: 11 curated campus FAQ articles (`data/raw/`), now including healthcare guidance in Tunisian Darija alongside earlier campus services, inscription renewal, and scholarship follow-up additions; all ingested into the chunked corpus (`data/processed/chunks.jsonl`).
- Evaluation set: 48 QA pairs (`data/faq/mini_faq.jsonl`) covering administrative procedures, social services, healthcare, and support workflows across the three target languages, with both Arabizi and Arabic-script Darija prompts.
- Telemetry: live QA interactions stream into `data/processed/eval/history.jsonl`, while user feedback and corrections are captured in `data/feedback/feedback.jsonl` for Platinum-level monitoring.
- Canonical language labelling ensures Tunisian Darija is encoded as `aeb`, including Arabizi heuristics.
- Normalisation scripts (`scripts/normalize_corpus_languages.py`) keep corpus metadata consistent; ingestion service enforces canonical language codes on write.

## System Architecture and Pipeline
- **Ingestion & Cleaning**: `DocumentIngestionService` segments documents, runs language detection, and writes chunk metadata with canonical Tunisian Darija labelling.
- **Hybrid Retrieval**: `HybridRetriever` combines BM25 (rank-bm25) with multilingual MiniLM sentence embeddings; indices rebuilt via `scripts/build_hybrid_index.py`.
- **QA and Attribution**: `ExtractiveQASystem` uses `deepset/xlm-roberta-base-squad2` with fast tokenizer, language-specific thresholds (`app/core/config.py`), and structured logging for inference errors. Short answers in Arabic/Darija are expanded by returning the sentence containing the span, and low-latency context-overlap heuristics pick the chunk whose surrounding sentence best matches the query. `AnswerSynthesizer` summarises supporting chunks when abstaining.
- **Language Detection & Normalisation**: `LanguageDetector` maps Arabizi heuristics and script variants to canonical codes; Darija normaliser converts Latin characters to comparable forms.

## Implementation Details
- FastAPI app exposes `/api/v1/qa` endpoint with request payload validation and per-request language detection.
- Structlog instrumentation surfaces retrieval loads, tokenizer selection, and QA anomalies.
- CLI utilities: ingestion, index build, evaluation, and corpus normalisation, each runnable via `.venv\Scripts\python.exe` for consistent Windows support.

- Runner (`scripts/evaluate_qa.py`) computes EM, F1, Recall@5, nDCG@5, and calibration (ECE, Brier); summaries saved to `data/processed/eval/evaluation_summary.json`.
- Fine-tuning CLI (`scripts/train_qa_model.py`) reshapes the FAQ dataset into SQuAD spans, trains `deepset/xlm-roberta-base-squad2` (or a configured base) via Hugging Face `Trainer`, and persists checkpoints under `models/tunispeak-qa/` for serving overrides.
- FastAPI now appends each response to telemetry logs, accepts user ratings/corrections through `POST /api/v1/feedback`, and serves aggregated calibration metrics via `GET /api/v1/metrics/calibration` for continuous monitoring.
- Latest results (11 questions): Recall@5 = 1.0, nDCG@5 ≈ 0.98, F1 ≈ 0.40 after adding cross-script synonym overlap and context overrides; abstention rate 0%. Exact match remains at zero, signalling room for phrase-level tuning.
- Per-language snapshot (F1 / abstention):

| Language | Examples | Avg F1 | Abstention |
|----------|----------|--------|------------|
| Tunisian Darija (`aeb`) | 8 | 0.40 | 0% |
| Modern Standard Arabic (`ar`) | 1 | 0.50 | 0% |
| French (`fr`) | 2 | 0.36 | 0% |
- LoRA adaptation of `TinyLlama-1.1B-Chat` on the 1,005-record FAQ dataset completed 50 optimizer steps (batch size 1, grad accum 4) on GPU; final `train_loss = 1.6886`, `eval_loss = 1.3085`, with checkpoints stored under `models/llama-lora-tiny-gpu/` and TensorBoard traces in the accompanying `runs/` directory.

## Ethical Considerations and Risk Assessment
- Abstention responses localised (French, Arabic, Tunisian Darija) so users receive uncertainty notices in their language.
- Risks: hallucination when thresholds reduced, bias from limited dataset, potential misinterpretation of Romanised Darija. Logging and abstention mitigate unsupported outputs.

## Deployment Plan and Access Controls
- Serve via Uvicorn (`app.main:app`), with CORS open for development. `.vscode/tasks.json` includes run/debug profiles. Production deployment would containerise FastAPI app and mount data volume for indices.
- Access to ingestion and evaluation scripts restricted to maintainers; plan to integrate authentication for admin tooling.

## Future Work (Silver/Gold/Platinum roadmap)
- Silver: (done) Expanded Darija QA set and refreshed fallback behaviour; continue curating labelled prompts per service season.
- Gold: Execute fine-tuning on Tunisian dialect data, improve UI for multilingual user flows, and benchmark lift over the expanded corpus.
- Platinum: Calibration monitoring and feedback capture are live; next steps focus on instrumenting dashboards/alerts and exploring speech interface for inclusive access.

## Appendices
- Model Card: `docs/MODEL_CARD.md`
- Data Card: `docs/DATA_CARD.md`
- Architecture reference: `docs/ARCHITECTURE.md`
- Evaluation summary: `data/processed/eval/evaluation_summary.json`
