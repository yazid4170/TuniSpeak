# Model Card: TuniSpeak QA Pipeline

## Model Details
- **Developers:** TuniSpeak Team
- **Model Type:** Retrieval-augmented QA pipeline (hybrid retrieval + abstractive synthesis)
- **Languages:** French (fr), Arabic (ar), Darija (ary)
- **Version:** 0.1.0 prototype

## Intended Use
- Answer institutional questions from students and staff using internal policies, FAQs, and procedures.
- Provide source citations for every answer to ensure traceability.

## Limitations
- Placeholder retrieval/QA components during the prototype phase; accuracy metrics pending.
- Darija normalisation heuristics remain preliminary.

## Ethical Considerations
- Documents may contain confidential data; access control required.
- Risk of language bias; plan subgroup evaluation and red-teaming in Darija.

## Metrics
- Target: EM/F1 for QA, Recall@k and nDCG for retrieval, calibration (ECE/Brier), latency, correct attribution rate.
