from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from io import BytesIO
import structlog
import torch
from faster_whisper import WhisperModel
from gtts import gTTS

from app.core.config import get_settings

logger = structlog.get_logger(__name__)


@dataclass
class SpeechTranscription:
    text: str
    language: str | None
    duration: float | None


class SpeechService:
    def __init__(
        self,
        model_size: str | None = None,
        device: str | None = None,
    ) -> None:
        settings = get_settings()
        self._model_size = model_size or settings.speech_model_size
        resolved_device = device or settings.speech_device
        if resolved_device is None:
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        # Whisper runs in mixed precision on GPU, quantised int8 on CPU to keep latency manageable.
        compute_type = "float16" if resolved_device != "cpu" else "int8"
        self._model = WhisperModel(
            self._model_size,
            device=resolved_device,
            compute_type=compute_type,
        )
        self._enable_tts = settings.speech_enable_tts
        self._tts_default_language = settings.speech_tts_default_language
        self._tts_language_map = settings.speech_tts_language_map

    def transcribe(self, audio_bytes: bytes, language: str | None = None) -> SpeechTranscription:
        if not audio_bytes:
            raise ValueError("Audio payload is empty")

        tmp_path: str | None = None
        try:
            # Faster-Whisper expects a real file handle, so stream bytes into a temp WAV.
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_file.flush()
                tmp_path = tmp_file.name

            segments, info = self._model.transcribe(
                tmp_path,
                language=language,
                beam_size=5,
            )
            text_segments: list[str] = []
            for segment in segments:
                text = segment.text.strip()
                if text:
                    text_segments.append(text)
            transcript = " ".join(text_segments).strip()
            detected_language = getattr(info, "language", None)
            duration = getattr(info, "duration", None)
            return SpeechTranscription(
                text=transcript,
                language=detected_language,
                duration=duration,
            )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    logger.warning("speech.cleanup_failed", path=tmp_path)

    def synthesize(self, text: str, language: str | None = None) -> bytes | None:
        if not self._enable_tts:
            return None
        if not text.strip():
            return None
        target_language = self._resolve_tts_language(language)
        try:
            # gTTS writes MP3 bytes into an in-memory buffer; caller returns it as base64.
            tts = gTTS(text=text, lang=target_language)
            buffer = BytesIO()
            tts.write_to_fp(buffer)
            return buffer.getvalue()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "speech.tts_failed",
                language=target_language,
                error=str(exc),
            )
            return None

    def _resolve_tts_language(self, language: str | None) -> str:
        if language is None:
            return self._tts_default_language
        return self._tts_language_map.get(language, self._tts_default_language)
