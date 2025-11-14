# Mini FAQ Dataset Format

Create a JSON Lines file (one JSON object per line) with the following keys:

```json
{
  "question": "Comment obtenir une attestation de scolarité ?",
  "answers": [
    "Vous pouvez demander une attestation de scolarité via le portail étudiant ou au guichet du service scolarité."
  ],
  "relevant_documents": ["doc-attestation", "doc-etudiants"],
  "relevance_gains": {
    "doc-attestation": 3,
    "doc-etudiants": 1
  },
  "metadata": {
    "category": "administratif",
    "language": "fr"
  }
}
```

- `question` (string): User question in French, Arabic, or Darija.
- `answers` (list[str]): One or more acceptable reference answers.
- `relevant_documents` (list[str]): Document or chunk identifiers considered relevant.
- `relevance_gains` (object, optional): Graded relevance values for nDCG; defaults to 1 for listed documents.
- `metadata` (object, optional): Arbitrary additional information (e.g., language, tags).

Store the file at `data/faq/mini_faq.jsonl` before running `poetry run tunispeak-eval`. A starter dataset covering French, Arabic, and Darija examples is provided; extend it alongside new documents you ingest. The repository now bundles 40+ prompts spanning administrative, social, and support themes as a baseline for evaluation and fine-tuning.
