# Architecture Overview

## Pipeline Layers
1. **Ingestion**: `DocumentIngestionService` converts raw documents (PDF, DOCX, TXT) into JSON chunks under `data/processed/`.
2. **Retrieval**: `HybridRetriever` combines a persisted BM25 index (`bm25.pkl`) with multilingual dense embeddings and a cross-encoder re-ranker for contextual ordering.
3. **Language**: `LanguageDetector` identifies French/Arabic/Darija and applies light normalisation (`normalize_darija`).
4. **QA**: `QAPipeline` combines the `ExtractiveQASystem`—now augmenting token overlap with bidirectional Arabic↔French medical synonyms and context-overlap overrides to avoid mixed-script abstentions—with an Ollama-backed `AnswerSynthesizer` that polishes confident answers and produces grounded fallbacks while returning confidence scores and attributions.
5. **API/UI**: FastAPI endpoints expose QA and health endpoints; the minimal frontend consumes `/api/v1/qa/answer` and surfaces confidence, normalised queries, and citations.
6. **Evaluation**: `EvaluationRunner` executes labelled QA sets, aggregates EM/F1, retrieval Recall@k/nDCG, and calibration (ECE/Brier) into persisted summaries; the latest baseline run produced Recall@5 = 1.0, nDCG@5 ≈ 0.98, F1 ≈ 0.40, with 0% abstention across 11 multilingual queries. The labelled dataset now holds 40+ prompts to support periodic fine-tuning sweeps.
7. **Fine-Tuning**: `scripts/train_qa_model.py` converts the expanded FAQ dataset into SQuAD-style spans, tokenises with the configured base model, and fine-tunes via Hugging Face `Trainer`, persisting checkpoints under `models/tunispeak-qa/` for deployment overrides.
8. **Telemetry & Feedback**: `TelemetryLogger` appends every QA interaction to `data/processed/eval/history.jsonl`, `FeedbackRepository` captures helpful/unhelpful votes plus corrections in `data/feedback/feedback.jsonl`, and FastAPI exposes `/api/v1/metrics/calibration` for live ECE/Brier monitoring.

## Storage
- `data/raw`: original corpus artefacts.
- `data/processed`: chunked JSON and hybrid indices.
- `data/feedback`: user feedback ledger for calibration and quality review.
- `data/faq`: annotated QA pairs.

## Roadmap
- Replace retrieval placeholders with BM25 via ElasticSearch/Whoosh and dense embeddings via SentenceTransformers + FAISS.
- Integrate OCR (Tesseract) for scanned PDFs.
- Add calibration metrics and refine abstention thresholds.
- Extend evaluation suite with QA metrics and retrieval nDCG computation (done).
- Maintain weekly fine-tuning runs to track EM/F1 lift and calibration drift once Gold baseline is reached.
- Wire streaming responses + tool calling once the generative layer proves stable (e.g., integrate function calling for timetable lookup).
- Expand telemetry dashboarding with alerting thresholds once sufficient feedback volume accumulates.
