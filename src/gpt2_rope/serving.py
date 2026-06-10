"""FastAPI inference service with micro-batched KV-cached decoding.

Requests queue into an async worker that groups compatible requests (same
prompt token length and sampling settings) into one batched ``generate`` call.
Latency and throughput metrics stream to the local ``metrics.jsonl`` format.
Production servers add paged KV caches and continuous batching (vLLM-style);
the queueing, batching, and observability contracts are the same.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from gpt2_rope.config import GenerationConfig
from gpt2_rope.generation import generate
from gpt2_rope.model import GPT
from gpt2_rope.monitoring import MetricLogger
from gpt2_rope.tokenizer import ByteBPETokenizer


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(default=64, ge=1)
    temperature: float = Field(default=1.0, ge=0)
    top_k: int | None = Field(default=None, ge=1)
    top_p: float | None = Field(default=None, gt=0, le=1)
    repetition_penalty: float = Field(default=1.0, gt=0)
    seed: int = 0


class GenerateResponse(BaseModel):
    completion: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    batch_size: int


@dataclass(slots=True)
class _Pending:
    token_ids: list[int]
    settings: GenerationConfig
    enqueued_at: float
    future: asyncio.Future[GenerateResponse]

    def batch_key(self) -> tuple[object, ...]:
        return (
            len(self.token_ids),
            self.settings.max_new_tokens,
            self.settings.temperature,
            self.settings.top_k,
            self.settings.top_p,
            self.settings.repetition_penalty,
            self.settings.seed,
        )


def group_compatible(pending: list[_Pending]) -> list[list[_Pending]]:
    """Group requests that can share one batched forward pass."""
    # Stringified keys sort uniformly even when fields mix None with numbers.
    ordered = sorted(enumerate(pending), key=lambda item: (str(item[1].batch_key()), item[0]))
    groups: list[list[_Pending]] = []
    for _, members in groupby(ordered, key=lambda item: str(item[1].batch_key())):
        groups.append([item for _, item in members])
    return groups


class InferenceService:
    """Single-model queueing service with same-shape micro-batching."""

    def __init__(
        self,
        model: GPT,
        tokenizer: ByteBPETokenizer,
        device: torch.device,
        *,
        max_batch_size: int = 8,
        batch_window_ms: float = 10.0,
        metrics_dir: Path | None = None,
    ) -> None:
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.max_batch_size = max_batch_size
        self.batch_window_s = batch_window_ms / 1000.0
        self._queue: asyncio.Queue[_Pending] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self._batches_served = 0
        self._logger = (
            MetricLogger(metrics_dir, tensorboard=False) if metrics_dir is not None else None
        )

    def validate(self, request: GenerateRequest) -> list[int]:
        token_ids = self.tokenizer.encode(request.prompt)
        if not token_ids:
            raise HTTPException(status_code=422, detail="prompt produced no tokens")
        budget = self.model.config.context_length - request.max_new_tokens
        if budget < 1:
            raise HTTPException(status_code=422, detail="max_new_tokens exceeds model context")
        if len(token_ids) > budget:
            raise HTTPException(
                status_code=422,
                detail=f"prompt has {len(token_ids)} tokens; budget is {budget}",
            )
        return token_ids

    async def submit(self, request: GenerateRequest) -> GenerateResponse:
        token_ids = self.validate(request)
        settings = GenerationConfig(
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            seed=request.seed,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        loop = asyncio.get_running_loop()
        pending = _Pending(token_ids, settings, time.perf_counter(), loop.create_future())
        await self._queue.put(pending)
        return await pending.future

    async def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._serve_forever())

    async def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
            self._worker = None
        if self._logger is not None:
            self._logger.close()

    async def _serve_forever(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            first = await self._queue.get()
            batch = [first]
            deadline = loop.time() + self.batch_window_s
            while len(batch) < self.max_batch_size:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self._queue.get(), remaining))
                except TimeoutError:
                    break
            for group in group_compatible(batch):
                try:
                    responses = await asyncio.to_thread(self._run_batch, group)
                except Exception as error:  # surface worker faults to callers
                    for pending in group:
                        if not pending.future.done():
                            pending.future.set_exception(error)
                    continue
                for pending, response in zip(group, responses, strict=True):
                    pending.future.set_result(response)

    def _run_batch(self, group: list[_Pending]) -> list[GenerateResponse]:
        started = time.perf_counter()
        input_ids = torch.tensor([pending.token_ids for pending in group], device=self.device)
        settings = group[0].settings
        output = generate(self.model, input_ids, settings)
        latency_ms = (time.perf_counter() - started) * 1000.0
        prompt_length = input_ids.size(1)
        responses: list[GenerateResponse] = []
        completion_tokens = 0
        for row in range(len(group)):
            generated = output[row, prompt_length:].tolist()
            completion_tokens += len(generated)
            responses.append(
                GenerateResponse(
                    completion=self.tokenizer.decode(generated),
                    prompt_tokens=prompt_length,
                    completion_tokens=len(generated),
                    latency_ms=latency_ms,
                    batch_size=len(group),
                )
            )
        if self._logger is not None:
            self._batches_served += 1
            self._logger.log(
                self._batches_served,
                {
                    "serve/batch_size": len(group),
                    "serve/latency_ms": latency_ms,
                    "serve/completion_tokens_per_second": completion_tokens
                    / max(latency_ms / 1000.0, 1e-9),
                },
            )
        return responses


def create_app(service: InferenceService) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await service.start()
        try:
            yield
        finally:
            await service.stop()

    app = FastAPI(title="gpt2-rope", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, object]:
        return {
            "status": "ok",
            "device": str(service.device),
            "parameters": service.model.parameter_count(),
            "context_length": service.model.config.context_length,
        }

    @app.post("/generate")
    async def generate_endpoint(request: GenerateRequest) -> GenerateResponse:
        return await service.submit(request)

    return app
