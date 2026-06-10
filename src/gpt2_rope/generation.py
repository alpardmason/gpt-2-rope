"""KV-cached autoregressive decoding and common sampling controls."""

from __future__ import annotations

import torch
from torch import Tensor

from gpt2_rope.config import GenerationConfig
from gpt2_rope.model import GPT


def _apply_repetition_penalty(logits: Tensor, tokens: Tensor, penalty: float) -> Tensor:
    if penalty == 1.0:
        return logits
    gathered = torch.gather(logits, 1, tokens)
    adjusted = torch.where(gathered < 0, gathered * penalty, gathered / penalty)
    return logits.scatter(1, tokens, adjusted)


def sample_next_token(
    logits: Tensor,
    tokens: Tensor,
    config: GenerationConfig,
    generator: torch.Generator,
) -> Tensor:
    logits = _apply_repetition_penalty(logits, tokens, config.repetition_penalty)
    if config.temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / config.temperature
    if config.top_k is not None:
        threshold = torch.topk(logits, min(config.top_k, logits.size(-1))).values[:, -1:]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    if config.top_p is not None:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probabilities = torch.softmax(sorted_logits, dim=-1)
        remove = probabilities.cumsum(dim=-1) - probabilities > config.top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(
            1, sorted_indices, sorted_logits
        )
    probabilities = torch.softmax(logits, dim=-1)
    return torch.multinomial(probabilities, 1, generator=generator)


@torch.inference_mode()
def generate(model: GPT, input_ids: Tensor, config: GenerationConfig) -> Tensor:
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    if input_ids.size(1) + config.max_new_tokens > model.config.context_length:
        raise ValueError("prompt plus generated tokens exceed model context")
    generator = torch.Generator(device=input_ids.device)
    generator.manual_seed(config.seed)
    output = input_ids
    model_output = model(output, use_cache=True)
    cache = model_output.past_key_values
    logits = model_output.logits[:, -1]
    finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)

    for _ in range(config.max_new_tokens):
        next_token = sample_next_token(logits, output, config, generator)
        if config.eos_token_id is not None:
            eos = torch.full_like(next_token, config.eos_token_id)
            next_token = torch.where(finished.unsqueeze(1), eos, next_token)
        output = torch.cat((output, next_token), dim=1)
        if config.eos_token_id is not None:
            finished |= next_token.squeeze(1).eq(config.eos_token_id)
            if bool(finished.all()):
                break
        model_output = model(next_token, past_key_values=cache, use_cache=True)
        cache = model_output.past_key_values
        logits = model_output.logits[:, -1]
    return output

