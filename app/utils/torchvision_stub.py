from __future__ import annotations

import os
import sys
import types
from enum import Enum
from importlib.machinery import ModuleSpec


def ensure_torchvision_stub() -> None:
    """Ensure Transformers can import without real torchvision ops present."""
    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    os.environ.setdefault("DISABLE_TRANSFORMERS_IMAGE_FEATURES", "1")
    os.environ.setdefault("DISABLE_TRANSFORMERS_IMAGE_IMPORTS", "1")
    os.environ.setdefault("DISABLE_TRANSFORMERS_AUDIO_IMPORTS", "1")

    if "torchvision" in sys.modules:
        return

    torchvision_stub = types.ModuleType("torchvision")
    transforms_stub = types.ModuleType("torchvision.transforms")

    class _InterpolationMode(Enum):
        NEAREST = 0
        BILINEAR = 2
        BICUBIC = 3
        LANCZOS = 4
        BOX = 5
        HAMMING = 6

    class _Compose(list):
        def __call__(self, value):
            result = value
            for transform in self:
                result = transform(result)
            return result

    transforms_stub.Compose = _Compose
    transforms_stub.InterpolationMode = _InterpolationMode
    transforms_stub.__spec__ = ModuleSpec(name="torchvision.transforms", loader=None)
    transforms_stub.__path__ = []
    torchvision_stub.transforms = transforms_stub
    torchvision_stub.__spec__ = ModuleSpec(name="torchvision", loader=None)
    torchvision_stub.__path__ = []  # mark as package for importlib

    v2_stub = types.ModuleType("torchvision.transforms.v2")
    v2_functional = types.ModuleType("torchvision.transforms.v2.functional")
    v2_stub.functional = v2_functional
    v2_stub.__spec__ = ModuleSpec(name="torchvision.transforms.v2", loader=None)
    v2_stub.__path__ = []
    v2_functional.__spec__ = ModuleSpec(
        name="torchvision.transforms.v2.functional",
        loader=None,
    )

    sys.modules["torchvision"] = torchvision_stub
    sys.modules["torchvision.transforms"] = transforms_stub
    sys.modules["torchvision.transforms.v2"] = v2_stub
    sys.modules["torchvision.transforms.v2.functional"] = v2_functional

    for submodule in ("datasets", "io", "models", "ops", "utils", "_meta_registrations"):
        module_name = f"torchvision.{submodule}"
        stub = types.ModuleType(module_name)
        stub.__spec__ = ModuleSpec(name=module_name, loader=None)
        stub.__path__ = []
        sys.modules[module_name] = stub
        setattr(torchvision_stub, submodule, stub)

    torchvision_stub.transforms = transforms_stub
