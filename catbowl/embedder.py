"""Turning a cropped cat into a vector.

Rather than fine-tuning a network on three cats (you would need thousands of
photos), we use a frozen ImageNet backbone as a feature extractor and train a
small classifier on top of its embeddings. Fifty to a hundred crops per cat is
plenty, and retraining takes seconds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .config import RecognitionConfig

log = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class EmbedderSpec:
    """Identity of an embedder, stored alongside a trained classifier.

    A classifier trained on one backbone is meaningless when fed another one's
    vectors, so the runtime refuses to load a mismatched pair.
    """

    backend: str
    model: str
    input_size: int
    dim: int


class Embedder:
    spec: EmbedderSpec

    def embed(self, image: np.ndarray) -> np.ndarray:
        return self.embed_batch([image])[0]

    def embed_batch(self, images: list[np.ndarray]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


def _prepare(image: np.ndarray, size: int) -> np.ndarray:
    """BGR uint8 crop -> normalised float32 RGB square."""
    import cv2

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
    return (rgb - IMAGENET_MEAN) / IMAGENET_STD


def _l2(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-8)


class TorchEmbedder(Embedder):
    """torchvision backbone, pooled to a single vector per image."""

    SUPPORTED = {
        "mobilenet_v3_small": ("MobileNet_V3_Small_Weights", 576),
        "mobilenet_v3_large": ("MobileNet_V3_Large_Weights", 960),
        "mobilenet_v2": ("MobileNet_V2_Weights", 1280),
    }

    def __init__(self, cfg: RecognitionConfig, threads: int = 2):
        import torch
        import torchvision.models as models

        if cfg.model not in self.SUPPORTED:
            raise ValueError(
                f"unsupported torch model {cfg.model!r}; pick one of {sorted(self.SUPPORTED)}"
            )
        self._torch = torch
        torch.set_num_threads(threads)
        weights_name, dim = self.SUPPORTED[cfg.model]
        weights = getattr(models, weights_name).DEFAULT
        net = getattr(models, cfg.model)(weights=weights)
        self.features = net.features
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.features.eval()
        self.size = cfg.input_size
        self.spec = EmbedderSpec("torch", cfg.model, cfg.input_size, dim)

    def embed_batch(self, images: list[np.ndarray]) -> np.ndarray:
        torch = self._torch
        batch = np.stack([_prepare(img, self.size) for img in images])
        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous()
        with torch.inference_mode():
            out = self.pool(self.features(tensor)).flatten(1)
        return _l2(out.numpy().astype(np.float32))


class TfliteEmbedder(Embedder):
    """A .tflite feature extractor, for when torch is too heavy for the Pi."""

    def __init__(self, cfg: RecognitionConfig):
        if not cfg.tflite_model_path:
            raise ValueError("recognition.tflite_model_path is required for the tflite backend")
        interpreter_cls = _load_tflite_interpreter()
        self.interp = interpreter_cls(model_path=cfg.tflite_model_path, num_threads=2)
        self.interp.allocate_tensors()
        self._in = self.interp.get_input_details()[0]
        self._out = self.interp.get_output_details()[0]
        self.size = int(self._in["shape"][1])
        dim = int(np.prod(self._out["shape"][1:]))
        self.spec = EmbedderSpec("tflite", cfg.tflite_model_path, self.size, dim)

    def embed_batch(self, images: list[np.ndarray]) -> np.ndarray:
        vectors = []
        for image in images:
            data = _prepare(image, self.size)[None, ...]
            if self._in["dtype"] == np.uint8:      # quantised model
                scale, zero = self._in["quantization"]
                data = np.clip(data / (scale or 1.0) + zero, 0, 255).astype(np.uint8)
            self.interp.set_tensor(self._in["index"], data.astype(self._in["dtype"]))
            self.interp.invoke()
            vectors.append(self.interp.get_tensor(self._out["index"]).reshape(-1))
        return _l2(np.stack(vectors).astype(np.float32))


def _load_tflite_interpreter():
    for module, attr in (
        ("ai_edge_litert.interpreter", "Interpreter"),
        ("tflite_runtime.interpreter", "Interpreter"),
        ("tensorflow.lite", "Interpreter"),
    ):
        try:
            mod = __import__(module, fromlist=[attr])
            return getattr(mod, attr)
        except ImportError:
            continue
    raise ImportError(
        "no TFLite runtime found - install ai-edge-litert (or tflite-runtime)"
    )


class MockEmbedder(Embedder):
    """Downsampled colour layout: a real, if weak, descriptor.

    No dependencies beyond OpenCV, deterministic, microseconds per image. Used by
    the tests and by ``catbowl selftest``, and good enough for three cats of
    clearly different colour if you never get the heavier backends installed.
    """

    def __init__(self, cfg: RecognitionConfig | None = None, grid: int = 8):
        self.grid = grid
        self.size = grid
        self.spec = EmbedderSpec("mock", f"colorgrid{grid}", grid, grid * grid * 3)

    def embed_batch(self, images: list[np.ndarray]) -> np.ndarray:
        import cv2

        rows = []
        for image in images:
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            small = cv2.resize(image, (self.grid, self.grid), interpolation=cv2.INTER_AREA)
            rows.append(small.astype(np.float32).reshape(-1) / 255.0)
        return _l2(np.stack(rows))


def build_embedder(cfg: RecognitionConfig) -> Embedder:
    if cfg.backend == "mock":
        return MockEmbedder(cfg)
    if cfg.backend == "tflite":
        return TfliteEmbedder(cfg)
    return TorchEmbedder(cfg)
