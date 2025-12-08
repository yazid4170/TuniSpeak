from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import httpx
import re
import structlog
import torch
from peft import PeftModel

from app.utils.torchvision_stub import ensure_torchvision_stub

from app.core.config import get_settings
from app.models.schemas import DocumentChunk
from app.services.ollama import OllamaClient
from app.utils.text import sentences

ensure_torchvision_stub()

from transformers import (  # noqa: E402  # import after stub ensures safe lazy load
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


ROLE_TOKENS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
}


class AnswerSynthesizer:
    def __init__(self) -> None:
        settings = get_settings()
        self._logger = structlog.get_logger(__name__)
        self._max_context_chars = settings.generative_max_context_chars
        self._max_new_tokens = settings.generative_max_new_tokens
        self._temperature = settings.generative_temperature
        self._top_p = settings.generative_top_p
        self._do_sample = settings.generative_do_sample
        self._repetition_penalty = settings.generative_repetition_penalty
        self._allow_generative = settings.generative_enable_rewrite
        self._device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self._local_tokenizer: PreTrainedTokenizerBase | None = None
        self._local_model: PreTrainedModel | None = None

        # Prefer a local LoRA adapter if the checkpoints are available on disk.
        if settings.generative_lora_base and settings.generative_lora_adapter:
            self._initialise_local_generator(
                base_path=Path(settings.generative_lora_base),
                adapter_path=Path(settings.generative_lora_adapter),
            )

        self._ollama: OllamaClient | None = None
        if not self._local_model and settings.generative_model:
            base_url = settings.ollama_base_url.rstrip("/")
            if self._ping_ollama(base_url, settings.generative_timeout):
                self._ollama = OllamaClient(
                    base_url=base_url,
                    model=settings.generative_model,
                    timeout=settings.generative_timeout,
                )
            else:
                self._logger.warning(
                    "synthesizer.ollama_unavailable",
                    base_url=base_url,
                )

    def summarise(self, question: str, chunks: Iterable[DocumentChunk]) -> str:
        top_chunks: list[DocumentChunk] = list(chunks)
        if not top_chunks:
            return "Aucune réponse trouvée."
        # Fast extractive summary gives the user *something* even if generation is disabled.
        sentences_pool: list[str] = []
        for chunk in top_chunks[:3]:
            sentences_pool.extend(sentences(chunk.text)[:3])
        if not sentences_pool:
            return top_chunks[0].text
        summary = " ".join(sentences_pool[:5])
        return summary.strip()

    def generate(
        self,
        question: str,
        chunks: Sequence[DocumentChunk],
        draft: str | None = None,
        language: str | None = None,
    ) -> str | None:
        context = self._build_context(chunks)
        if not context:
            return None
        # System prompt switches by language so citations + tone match the question.
        system_prompt = (
            "Tu es un assistant universitaire bénévole. Réponds de manière concise, "
            "structurée et en citant les sources entre crochets (ex. [Source 1]). "
            "Utilise uniquement les informations du contexte fourni."
        )
        if language == "ar":
            system_prompt = (
                "أنت مساعد جامعي. أجب بإيجاز وباستناد حصري إلى السياق المرفق، "
                "واذكر المصادر بين أقواس مثل [Source 1]."
            )
        elif language == "aeb":
            system_prompt = (
                "إنت مساعد للطلبة في الجامعة. جاوب باختصار وبالدارجة التونسية، "
                "واعتمد كان على المعلومات اللي في السياق وتذكر المصادر بين []" 
                "مثال [Source 1]."
            )

        prompt_sections: list[str] = [
            "Contexte:",
            context,
            "",
            f"Question: {question.strip()}",
        ]
        if draft:
            prompt_sections.extend(["", f"Réponse extraite: {draft.strip()}"])
        prompt_sections.append("")
        prompt_sections.append(
            "Consigne: Fournis une réponse courte (max 3 phrases) dans la langue de la question."
        )
        prompt = "\n".join(prompt_sections).strip()

        if self._allow_generative and self._local_model and self._local_tokenizer:
            # First attempt is offline to avoid latency and data leakage.
            generated = self._generate_with_local_model(system_prompt, prompt)
            if generated and self._is_rewrite_acceptable(draft, generated, question):
                self._logger.info("synthesizer.generative_success", backend="local")
                return generated
            if generated:
                self._logger.info("synthesizer.generative_reject", backend="local")

        if self._allow_generative and self._ollama:
            # Ollama serves as remote fallback when local weights are missing.
            generated = self._ollama.generate(prompt=prompt, system_prompt=system_prompt)
            if generated and self._is_rewrite_acceptable(draft, generated, question):
                self._logger.info("synthesizer.generative_success", backend="ollama")
                return generated
            if generated:
                self._logger.info("synthesizer.generative_reject", backend="ollama")

        return self._deterministic_rewrite(question, chunks, draft, language)

    def _build_context(self, chunks: Sequence[DocumentChunk]) -> str:
        if not chunks:
            return ""
        # Cap the prompt budget so llama-based models stay within VRAM constraints.
        budget = max(200, self._max_context_chars)
        collected: list[str] = []
        used = 0
        for index, chunk in enumerate(chunks[:5], start=1):
            text = chunk.text.strip()
            if not text:
                continue
            remaining = budget - used
            if remaining <= 0:
                break
            snippet = text if len(text) <= remaining else text[:remaining]
            collected.append(f"[Source {index}] {snippet}")
            used += len(snippet)
        return "\n\n".join(collected)

    def _compose_conversation(self, system_prompt: str, prompt: str) -> str:
        segments = [
            f"{ROLE_TOKENS['system']}\n{system_prompt.strip()}\n",
            f"{ROLE_TOKENS['user']}\n{prompt.strip()}\n",
            f"{ROLE_TOKENS['assistant']}\n",
        ]
        return "".join(segments)

    def _generate_with_local_model(self, system_prompt: str, prompt: str) -> str | None:
        if not self._local_model or not self._local_tokenizer:
            return None

        conversation = self._compose_conversation(system_prompt, prompt)
        try:
            inputs = self._local_tokenizer(
                conversation,
                return_tensors="pt",
                padding=False,
            )
        except Exception as exc:
            self._logger.warning(
                "synthesizer.tokenization_failed",
                backend="local",
                error_message=str(exc),
                exception_type=exc.__class__.__name__,
            )
            return None

        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        generate_kwargs = {
            "max_new_tokens": self._max_new_tokens,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "do_sample": self._do_sample,
            "repetition_penalty": self._repetition_penalty,
            "pad_token_id": self._local_tokenizer.pad_token_id,
            "eos_token_id": self._local_tokenizer.eos_token_id,
        }

        try:
            with torch.no_grad():
                output = self._local_model.generate(**inputs, **generate_kwargs)
        except RuntimeError as exc:
            self._logger.warning(
                "synthesizer.generation_failed",
                backend="local",
                error_message=str(exc),
                exception_type=exc.__class__.__name__,
            )
            return None

        input_length = inputs["input_ids"].shape[1]
        generated_ids = output[0, input_length:]
        if generated_ids.numel() == 0:
            return None

        text = self._local_tokenizer.decode(generated_ids, skip_special_tokens=True)
        cleaned = text.replace("<|end|>", "").replace("</|end|>", "").strip()
        return cleaned or None

    def _is_rewrite_acceptable(
        self,
        draft: str | None,
        generated: str,
        question: str,
    ) -> bool:
        if not generated:
            return False
        # Reject outputs that look like a prompt leak or are too far from the draft.
        if "<|" in generated or "|>" in generated:
            return False
        if draft:
            overlap = self._token_overlap_ratio(draft, generated)
            if overlap < 0.4:
                return False
        question_overlap = self._token_overlap_ratio(question, generated)
        if question_overlap <= 0.05:
            return False
        return True

    @staticmethod
    def _token_overlap_ratio(text_a: str, text_b: str) -> float:
        tokens_a = AnswerSynthesizer._significant_tokens(text_a)
        if not tokens_a:
            return 0.0
        tokens_b = AnswerSynthesizer._significant_tokens(text_b)
        if not tokens_b:
            return 0.0
        shared = tokens_a & tokens_b
        return len(shared) / float(len(tokens_a))

    @staticmethod
    def _significant_tokens(text: str) -> set[str]:
        if not text:
            return set()
        return {token for token in re.findall(r"\w+", text.lower()) if len(token) >= 4}

    def _deterministic_rewrite(
        self,
        question: str,
        chunks: Sequence[DocumentChunk],
        draft: str | None,
        language: str | None,
    ) -> str | None:
        base_answer = (draft or "").strip()
        if base_answer:
            return base_answer

        summary = self.summarise(question, chunks).strip()
        return summary or None

    def _initialise_local_generator(self, base_path: Path, adapter_path: Path) -> None:
        resolved_base = base_path.expanduser()
        resolved_adapter = adapter_path.expanduser()
        if not resolved_base.exists() or not resolved_adapter.exists():
            self._logger.warning(
                "synthesizer.local_missing_paths",
                base=str(resolved_base),
                adapter=str(resolved_adapter),
            )
            return

        try:
            tokenizer = AutoTokenizer.from_pretrained(resolved_adapter, use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

            model = AutoModelForCausalLM.from_pretrained(
                resolved_base,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            model = PeftModel.from_pretrained(model, resolved_adapter)
            model.to(self._device)
            model.eval()
            model.config.pad_token_id = tokenizer.pad_token_id

            self._local_tokenizer = tokenizer
            self._local_model = model
            self._logger.info(
                "synthesizer.local_loaded",
                base=str(resolved_base),
                adapter=str(resolved_adapter),
                device=str(self._device),
            )
        except Exception as exc:
            self._logger.warning(
                "synthesizer.local_load_failed",
                base=str(resolved_base),
                adapter=str(resolved_adapter),
                error_message=str(exc),
                exception_type=exc.__class__.__name__,
            )

    def _ping_ollama(self, base_url: str, timeout: float) -> bool:
        try:
            response = httpx.get(f"{base_url}/api/tags", timeout=min(timeout, 3.0))
            response.raise_for_status()
            return True
        except httpx.HTTPError as exc:
            self._logger.info(
                "synthesizer.ollama_ping_failed",
                base_url=base_url,
                error_message=str(exc),
            )
            return False
