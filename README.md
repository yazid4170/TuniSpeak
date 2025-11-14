# TuniSpeak

TuniSpeak is a trilingual (French, Arabic, Darija) question answering assistant designed for university student services. It provides language-aware retrieval augmented generation with source attribution over internal regulations, FAQs, and procedures.

## Features
- FastAPI backend that orchestrates ingestion, retrieval, and QA pipelines
- Hybrid search combining BM25 and dense embeddings with contextual re-ranking
- Multilingual extractive QA with confidence scoring, abstention, and fallback summarisation
- Optional Ollama-powered LLaMA layer that rewrites answers conversationally while preserving citations
- LoRA fine-tuning script for adapting LLaMA chat weights on local instruction data
- Lightweight Darija normalisation and language detection
- Mini web UI for multilingual queries and citation display
- Evaluation utilities for QA (EM/F1) and retrieval (Recall@k, nDCG)
- User feedback loop with live calibration metrics (ECE/Brier) exposed via `/api/v1/metrics/calibration`
- Fine-tuning CLI to adapt the QA model on expanded labelled FAQs

## Getting Started

### Quickstart (command by command)
1. Clone the repository and enter the project folder:
	```powershell
	git clone https://github.com/<your-org>/tunispeak.git
	Set-Location tunispeak
	```
2. Install dependencies with Poetry:
	```powershell
	poetry install
	```
3. Create your environment file from the template:
	```powershell
	Copy-Item .env.example .env
	```
4. Download the required model weights (TinyLlama base, QA checkpoint, LoRA adapters):
	```powershell
	poetry run python scripts/download_models.py
	```
5. (Optional) Ingest raw documents and build the hybrid index:
	```powershell
	poetry run tunispeak-ingest data/raw
	poetry run tunispeak-index
	```
6. Launch the API (serves both backend and frontend):
	```powershell
	poetry run uvicorn app.main:app --host 127.0.0.1 --port 5500
	```
	Then browse to `http://127.0.0.1:5500/`.
7. Run the automated test suite:
	```powershell
	poetry run pytest
	```
8. Evaluate QA quality against the sample dataset:
	```powershell
	poetry run tunispeak-eval --dataset data/faq/mini_faq.jsonl
	```

### Prerequisites
- Python 3.11+
- Poetry 1.8+

### Installation
```powershell
poetry install
```

### Environment
Copy `.env.example` to `.env` and adjust secrets:
```powershell
Copy-Item .env.example .env
```

### Model Assets
- The TinyLlama base checkpoint is fetched on demand from the official Hugging Face repository (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`).
- Custom QA weights (`models/tunispeak-qa-sfax/`) and LoRA adapters (`models/llama-lora-tiny-gpu/`) are versioned in the repo via Git LFS, so teammates receive them automatically after cloning.
- If the TinyLlama download fails (e.g. network interruption), re-run `poetry run python scripts/download_models.py --force` to retry.


Generative rephrasing can now run either through Ollama or the locally fine-tuned TinyLlama adapter:

- **Local LoRA adapter** (preferred for offline operation)

	```properties
	GENERATIVE_LORA_BASE=./models/base/tinyllama-1.1b-chat
	GENERATIVE_LORA_ADAPTER=./models/llama-lora-tiny-gpu
	GENERATIVE_MAX_NEW_TOKENS=220
	GENERATIVE_TEMPERATURE=0.2
	GENERATIVE_TOP_P=0.9
	GENERATIVE_DO_SAMPLE=true
	GENERATIVE_REPETITION_PENALTY=1.05
	```

	The service loads the base checkpoint, merges the LoRA weights at runtime, and keeps responses constrained to the retrieved context. GPU acceleration is used automatically when available.

- **Ollama fallback**

	Install [Ollama](https://ollama.ai/download), pull a model (e.g. `ollama pull llama3.1:8b`), then set:

```properties
GENERATIVE_MODEL=llama3.1:8b
OLLAMA_BASE_URL=http://localhost:11434
```

### Corpus Preparation
Place institutional documents (PDF, DOCX, TXT, MD) into `data/raw/` and run the ingestion + indexing utilities:
```powershell
poetry run tunispeak-ingest data/raw
poetry run tunispeak-index
```
Repeat the ingestion command when new documents are added, then rebuild the index.
`tunispeak-index` writes the sparse BM25 index (`data/processed/hybrid_index/bm25.pkl`) and dense embeddings; the first run will download the configured encoder and cross-encoder models.

🆕 A synthetic Tunisian university QA corpus (`data/raw/dataset_final_universite_tunisie.json`) is already staged; it has been converted into retrieval chunks (`data/processed/chunks.jsonl`) and a FAQ dataset (`data/faq/dataset_final_universite_tunisie.jsonl`). Re-run `poetry run tunispeak-index` after any edits to keep the hybrid index fresh.

### Development Server
Run the API (which now also serves the frontend):
```powershell
poetry run uvicorn app.main:app --host 127.0.0.1 --port 5500
```
Open `http://127.0.0.1:5500/` in your browser; the UI and API share the same origin, so no extra static server is required.

Need hot-reload during active development? Append `--reload`, but keep in mind it spawns an additional watcher process and can surface GPU-loading issues on Windows.

### Tests
```powershell
poetry run pytest
```

### Evaluation
Prepare a labelled FAQ dataset at `data/faq/mini_faq.jsonl` (see `data/faq/README.md` for the format). A multilingual sample (FR/AR/Darija) is included; update it alongside new documents. After rebuilding the hybrid index, run:
```powershell
poetry run tunispeak-eval --dataset data/faq/mini_faq.jsonl
```
The command writes `evaluation_summary.json` under `data/processed/eval/` with aggregate metrics (EM, F1, Recall@5, nDCG@5, calibration) and per-example breakdowns.

### Fine-Tuning
The dataset now ships with 40+ multilingual QA pairs to support supervised adaptation. Train a local QA checkpoint (saved to `models/tunispeak-qa/` by default) with:
```powershell
poetry run tunispeak-train --dataset data/faq/mini_faq.jsonl --output-dir models/tunispeak-qa
```
Override `--base-model`, `--eval-split`, or other hyper-parameters as needed. Point `QA_MODEL` in `.env` to the resulting directory to serve the fine-tuned weights.
To leverage the synthetic corpus, swap in `data/faq/dataset_final_universite_tunisie.jsonl` or merge it with your curated `mini_faq.jsonl` before launching the training command.

Need additional supervised data? Fetch open QA corpora with:
```powershell
poetry run tunispeak-fetch-external
```
This writes curated Arabic and French question answering datasets (MLQA, TyDi QA, PIAF) to `data/external/` with licensing metadata for provenance. Limit the sample if needed:
```powershell
poetry run tunispeak-fetch-external --dataset mlqa --limit 250
```

#### LoRA fine-tuning for LLaMA

Use `scripts/train_llama_lora.py` to adapt a LLaMA-family chat checkpoint with LoRA adapters. A GPU with at least 4 GB VRAM (e.g. GTX 1650) is sufficient for the TinyLlama variant used in this project:

```powershell
$env:TEMP='D:/NLP/.tmp'; $env:TMP='D:/NLP/.tmp'
python scripts/train_llama_lora.py \
	--dataset-path data/faq/dataset_finetune.jsonl \
	--base-model models/base/tinyllama-1.1b-chat \
	--output-dir models/llama-lora-tiny-gpu \
	--per-device-batch-size 1 \
	--gradient-accumulation-steps 4 \
	--max-seq-length 512 \
	--save-steps 20 \
	--logging-steps 5 \
	--eval-ratio 0.1 \
	--max-train-steps 50
```

The command saves LoRA adapter weights and tokenizer metadata under `models/llama-lora-tiny-gpu`. Point `GENERATIVE_LORA_BASE` and `GENERATIVE_LORA_ADAPTER` in `.env` to these paths to activate local generation. Adjust learning rate, accumulation, and steps to fit your GPU or CPU budget; the default hyper-parameters mirror the run documented in `docs/REPORT_DRAFT.md` (train loss ≈ 1.69, eval loss ≈ 1.31).

### Feedback & Monitoring
Each answer response includes an `interaction_id`; the frontend surfaces a feedback widget (helpful / not helpful, optional correction). Submissions hit `POST /api/v1/feedback` and are archived under `data/feedback/feedback.jsonl`. Calibration metrics (ECE, Brier, per-bin summaries) aggregate telemetry + feedback at `GET /api/v1/metrics/calibration`.

## Project Structure
```
app/                # FastAPI application and domain modules
scripts/            # CLI utilities for ingestion, indexing, evaluation
tests/              # Automated tests
data/               # Raw, processed, and FAQ datasets (placeholders)
docs/               # Model card, data card, and report outline
frontend/           # Minimal web UI
```

## Documentation
- `docs/MODEL_CARD.md`
- `docs/DATA_CARD.md`
- `docs/REPORT_OUTLINE.md`

## License
All rights reserved. Internal academic use only.
