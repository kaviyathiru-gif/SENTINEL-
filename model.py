"""
model.py
--------
PyTorch anomaly-detection autoencoder plus an async batching inference engine.

Design notes:
- Autoencoder trained on benign flow features; high reconstruction error => anomalous.
  This is standard for unsupervised network IDS since labeled attack data is scarce.
- The InferenceEngine batches concurrent requests from many HTTP/WebSocket callers into
  a single forward pass, so the GPU/CPU stays utilized efficiently under load without
  blocking the FastAPI event loop (all torch calls run in a worker thread via to_thread).
- Falls back cleanly to CPU on devices with no CUDA (Raspberry Pi / Jetson Nano / gateways).
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn

from config import get_settings

logger = logging.getLogger("sentinel.model")
settings = get_settings()


class NeuralNetwork(nn.Module):
    """
    Lightweight fully-connected autoencoder for flow-feature anomaly detection.
    Small enough to run in real time on edge hardware (CPU-only Pi/Jetson) while
    still being expressive enough for typical NetFlow/CICFlowMeter-style feature sets.
    """

    def __init__(self, input_dim: int = 32, latent_dim: int = 8):
        super().__init__()
        hidden_dim = max(latent_dim * 2, input_dim // 2)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample mean squared error between input and reconstruction."""
        recon = self.forward(x)
        return torch.mean((recon - x) ** 2, dim=1)


def select_device(preference: str = "auto") -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available; falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class InferenceRequest:
    features: torch.Tensor
    future: "asyncio.Future"


class InferenceEngine:
    """
    Owns the model and a background asyncio task that batches incoming inference
    requests. Callers `await engine.infer(vector)` from any endpoint/coroutine;
    the engine coalesces concurrent calls into batched GPU/CPU forward passes.
    """

    def __init__(self):
        self.device = select_device(settings.MODEL_DEVICE)
        self.model = NeuralNetwork(
            input_dim=settings.MODEL_INPUT_DIM,
            latent_dim=settings.MODEL_LATENT_DIM,
        ).to(self.device)
        self.model.eval()

        self._queue: "asyncio.Queue[InferenceRequest]" = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self._batch_size = settings.INFERENCE_BATCH_SIZE
        self._batch_timeout = settings.INFERENCE_BATCH_TIMEOUT_MS / 1000.0

        # Simple counters for the /metrics endpoint.
        self.total_processed = 0
        self._latency_ema_ms = 0.0
        self._start_time = time.monotonic()

        self._load_weights_if_present()

    # -- lifecycle -----------------------------------------------------

    def _load_weights_if_present(self) -> None:
        path = Path(settings.MODEL_PATH)
        if path.exists():
            try:
                state = torch.load(path, map_location=self.device)
                self.model.load_state_dict(state)
                logger.info("Loaded model weights from %s", path)
            except Exception as exc:  # noqa: BLE001 — log and continue with random init
                logger.error("Failed to load model weights (%s); using randomly-initialized model.", exc)
        else:
            logger.warning("No model weights found at %s; using randomly-initialized model. "
                            "Train and export weights before production use.", path)

    def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._batch_worker(), name="inference-batch-worker")
            logger.info("Inference engine started on device=%s", self.device)

    async def stop(self) -> None:
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

    # -- public API ------------------------------------------------------

    async def infer(self, feature_vector: List[float]) -> Tuple[float, bool, float]:
        """
        Submit one feature vector for inference. Returns
        (reconstruction_error, is_anomalous, confidence).
        Awaits until the batch worker processes it — typically sub-30ms.
        """
        tensor = torch.tensor(feature_vector, dtype=torch.float32)
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        await self._queue.put(InferenceRequest(features=tensor, future=future))
        return await future

    # -- internals ---------------------------------------------------------

    async def _batch_worker(self) -> None:
        """Continuously drains the queue, forming batches up to batch_size or batch_timeout."""
        while True:
            requests: List[InferenceRequest] = []
            try:
                first = await self._queue.get()
                requests.append(first)
                deadline = asyncio.get_running_loop().time() + self._batch_timeout
                while len(requests) < self._batch_size:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        break
                    try:
                        req = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                        requests.append(req)
                    except asyncio.TimeoutError:
                        break

                await self._run_batch(requests)
            except asyncio.CancelledError:
                # Fail any in-flight requests so callers don't hang on shutdown.
                for req in requests:
                    if not req.future.done():
                        req.future.set_exception(RuntimeError("Inference engine shutting down"))
                raise
            except Exception as exc:  # noqa: BLE001 — never let the worker die silently
                logger.exception("Batch inference failed: %s", exc)
                for req in requests:
                    if not req.future.done():
                        req.future.set_exception(exc)

    async def _run_batch(self, requests: List[InferenceRequest]) -> None:
        start = time.perf_counter()
        # Run the actual torch forward pass off the event loop thread.
        batch = torch.stack([r.features for r in requests]).to(self.device)
        errors = await asyncio.to_thread(self._forward_sync, batch)

        threshold = settings.ANOMALY_THRESHOLD
        for req, err in zip(requests, errors.tolist()):
            is_anomalous = err > threshold
            # Confidence: how far past (or under) the threshold, squashed to [0, 1].
            confidence = min(1.0, abs(err - threshold) / max(threshold, 1e-6))
            if not req.future.done():
                req.future.set_result((err, is_anomalous, confidence))

        elapsed_ms = (time.perf_counter() - start) * 1000
        self.total_processed += len(requests)
        # Exponential moving average for a smooth latency metric.
        alpha = 0.2
        self._latency_ema_ms = (
            elapsed_ms if self._latency_ema_ms == 0 else alpha * elapsed_ms + (1 - alpha) * self._latency_ema_ms
        )

    def _forward_sync(self, batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model.reconstruction_error(batch).cpu()

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def avg_latency_ms(self) -> float:
        return round(self._latency_ema_ms, 3)

    @property
    def uptime_seconds(self) -> float:
        return time.monotonic() - self._start_time


# Module-level singleton, created once at import time, started in the FastAPI lifespan.
inference_engine = InferenceEngine()
