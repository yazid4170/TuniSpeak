"""TuniSpeak FastAPI application package."""

from __future__ import annotations

import os

# Transformers pulls in torchvision by default; disabling it prevents missing CUDA ops on Windows.
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("DISABLE_TRANSFORMERS_IMAGE_FEATURES", "1")
os.environ.setdefault("DISABLE_TRANSFORMERS_IMAGE_IMPORTS", "1")
os.environ.setdefault("DISABLE_TRANSFORMERS_AUDIO_IMPORTS", "1")
