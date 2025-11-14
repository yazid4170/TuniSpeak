# Data Card: TuniSpeak Corpora

## Summary
- **Sources:** Internal regulations, procedures, FAQs, annotated Q/A pairs (1k-3k).
- **Formats:** PDF, DOCX, TXT, structured JSON.
- **Languages:** French, Modern Standard Arabic, Darija (Moroccan Arabic).

## Collection Process
- OCR pipeline for scanned PDFs (planned integration).
- Manual annotation workflow for FAQ pairs with double review.

## Preprocessing
- Document segmentation into semantic chunks.
- Language identification and Darija normalisation heuristics.
- Metadata preservation for attribution (document id, section, page).

## Risks & Mitigations
- Confidential content handled via secure storage with audit logs.
- Language bias addressed with balanced sampling and subgroup evaluation.
- External augmentation corpora tracked with explicit licensing metadata (see below).

## External Augmentation Sources
- **MLQA (ar/ar split)** — multilingual Wikipedia QA pairs (Arabic questions & contexts). License: CC BY-SA 3.0. Source: https://github.com/facebookresearch/MLQA.
- **TyDi QA (secondary task, Arabic subset)** — cross-lingual QA with naturally occurring questions. License: CC BY 4.0. Source: https://huggingface.co/datasets/tydiqa.
- **PIAF (plain_text split)** — crowdsourced French QA pairs curated by Etalab. License: Licence Ouverte / Open Licence 2.0. Source: https://huggingface.co/datasets/etalab-ia/piaf.
- **dataset_final_universite_tunisie** — corpus synthétique FR/AR/Darija couvrant démarches administratives (1 082 paires QA). Source interne, créé pour l'expérimentation; voir `data/raw/dataset_final_universite_tunisie.json`.
