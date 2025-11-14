from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_reload: bool = Field(default=False)
    environment: str = Field(default="development")
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    data_root: str = Field(default="./data")
    raw_data_dir: str = Field(default="./data/raw")
    processed_data_dir: str = Field(default="./data/processed")
    faq_data_path: str = Field(default="./data/faq/mini_faq.jsonl")
    hybrid_index_path: str = Field(default="./data/processed/hybrid_index")

    dense_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    bm25_tokenizer: str = Field(default="arabic")
    cross_encoder_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    qa_model: str = Field(default="deepset/xlm-roberta-base-squad2")
    qa_confidence_threshold: float = Field(default=0.15)
    qa_language_thresholds: dict[str, float] = Field(
        default_factory=lambda: {"fr": 0.05, "ar": 0.02, "aeb": 0.005}
    )
    qa_language_overlap: dict[str, float] = Field(
        default_factory=lambda: {"default": 0.5, "fr": 0.55, "ar": 0.4, "aeb": 0.3}
    )
    qa_language_similarity: dict[str, float] = Field(
        default_factory=lambda: {"aeb": 0.55}
    )
    qa_language_context_override: dict[str, float] = Field(
        default_factory=lambda: {"aeb": 0.2}
    )
    qa_max_chunks: int = Field(default=5)
    qa_abstain_message: str = Field(
        default="Je ne suis pas suffisamment certain pour fournir une réponse fiable."
    )
    qa_abstain_messages: dict[str, str] = Field(
        default_factory=lambda: {
            "fr": "Je ne suis pas suffisamment certain pour fournir une réponse fiable.",
            "ar": "لست واثقًا بدرجة كافية لتقديم إجابة دقيقة حاليًا.",
            "aeb": "مع الأسف ما عنديش معلومة مؤكدة تو."}
    )
    generative_model: str | None = Field(default=None)
    generative_lora_base: str | None = Field(default=None)
    generative_lora_adapter: str | None = Field(default=None)
    generative_max_context_chars: int = Field(default=2400)
    generative_timeout: float = Field(default=30.0)
    generative_enable_rewrite: bool = Field(default=True)
    generative_max_new_tokens: int = Field(default=256)
    generative_temperature: float = Field(default=0.3)
    generative_top_p: float = Field(default=0.9)
    generative_do_sample: bool = Field(default=True)
    generative_repetition_penalty: float = Field(default=1.05)
    ollama_base_url: str = Field(default="http://localhost:11434")

    eval_output_dir: str = Field(default="./data/processed/eval")
    telemetry_log_path: str = Field(default="./data/processed/eval/history.jsonl")
    feedback_log_path: str = Field(default="./data/feedback/feedback.jsonl")
    retriever_curated_prefixes: list[str] = Field(default_factory=lambda: ["doc-"])
    retriever_curated_boost: float = Field(default=0.8)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()
