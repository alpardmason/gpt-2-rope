"""Evaluation harness: perplexity suites, logprob-scored tasks, and passkey.

This is a deliberately small, transparent harness. The industry-standard tool
for benchmark breadth is lm-evaluation-harness; the scoring rules implemented
here (windowed perplexity, length-normalized choice logprobs, exact-match
retrieval) are the same primitives it uses.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as functional

from gpt2_rope.config import GenerationConfig
from gpt2_rope.generation import generate
from gpt2_rope.model import GPT
from gpt2_rope.tokenizer import ByteBPETokenizer


@dataclass(slots=True)
class MultipleChoiceExample:
    question: str
    choices: list[str]
    answer: int


def load_multiple_choice_tasks(path: Path) -> list[MultipleChoiceExample]:
    """Read JSONL rows of ``{"question", "choices", "answer"}`` with validation."""
    examples: list[MultipleChoiceExample] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("question"), str)
            or not isinstance(record.get("choices"), list)
            or not isinstance(record.get("answer"), int)
        ):
            raise ValueError(f"{path}:{line_number} requires question, choices, answer")
        choices = [str(choice) for choice in record["choices"]]
        answer = record["answer"]
        if len(choices) < 2:
            raise ValueError(f"{path}:{line_number} needs at least two choices")
        if not 0 <= answer < len(choices):
            raise ValueError(f"{path}:{line_number} answer index out of range")
        examples.append(MultipleChoiceExample(record["question"], choices, answer))
    if not examples:
        raise ValueError(f"{path} contains no examples")
    return examples


@torch.inference_mode()
def continuation_logprob(
    model: GPT,
    token_ids: Sequence[int],
    prefix_length: int,
    device: torch.device,
) -> float:
    """Sum of ``log p(token_t | tokens_<t)`` for every token after the prefix."""
    if not 0 < prefix_length < len(token_ids):
        raise ValueError("prefix must be non-empty and shorter than the sequence")
    input_ids = torch.tensor([token_ids], device=device)
    logits = model(input_ids, use_cache=False).logits[0]
    log_probs = functional.log_softmax(logits.float(), dim=-1)
    total = 0.0
    for position in range(prefix_length, len(token_ids)):
        total += float(log_probs[position - 1, token_ids[position]])
    return total


@torch.inference_mode()
def evaluate_multiple_choice(
    model: GPT,
    tokenizer: ByteBPETokenizer,
    examples: Sequence[MultipleChoiceExample],
    device: torch.device,
    length_normalize: bool = True,
) -> dict[str, float]:
    """Score each choice as a continuation of the question; argmax wins."""
    model.eval()
    correct = 0
    for example in examples:
        prefix = tokenizer.encode(example.question)
        scores: list[float] = []
        for choice in example.choices:
            tokens = prefix + tokenizer.encode(choice)
            if len(tokens) > model.config.context_length:
                raise ValueError("question plus choice exceeds model context")
            score = continuation_logprob(model, tokens, len(prefix), device)
            if length_normalize:
                score /= max(len(tokens) - len(prefix), 1)
            scores.append(score)
        if scores.index(max(scores)) == example.answer:
            correct += 1
    return {
        "task/multiple_choice_accuracy": correct / len(examples),
        "task/multiple_choice_examples": float(len(examples)),
    }


@torch.inference_mode()
def evaluate_perplexity_files(
    model: GPT,
    tokenizer: ByteBPETokenizer,
    paths: Sequence[Path],
    device: torch.device,
    sequence_length: int | None = None,
) -> dict[str, float]:
    """Windowed held-out perplexity per file plus a token-weighted aggregate."""
    model.eval()
    window = min(sequence_length or model.config.context_length, model.config.context_length)
    metrics: dict[str, float] = {}
    total_loss = 0.0
    total_tokens = 0
    for path in paths:
        token_ids = tokenizer.encode(path.read_text(encoding="utf-8"))
        if len(token_ids) < 2:
            raise ValueError(f"{path} has too few tokens to evaluate")
        file_loss = 0.0
        file_tokens = 0
        for start in range(0, len(token_ids) - 1, window):
            chunk = token_ids[start : start + window + 1]
            if len(chunk) < 2:
                break
            input_ids = torch.tensor([chunk], device=device)
            output = model(input_ids, labels=input_ids, use_cache=False)
            if output.loss is None:
                raise RuntimeError("evaluation model did not return loss")
            count = len(chunk) - 1
            file_loss += float(output.loss) * count
            file_tokens += count
        mean = file_loss / file_tokens
        metrics[f"perplexity/{path.stem}"] = float(torch.exp(torch.tensor(min(mean, 20.0))))
        total_loss += file_loss
        total_tokens += file_tokens
    aggregate = total_loss / total_tokens
    metrics["perplexity/aggregate"] = float(torch.exp(torch.tensor(min(aggregate, 20.0))))
    metrics["perplexity/tokens"] = float(total_tokens)
    return metrics


@dataclass(slots=True)
class PasskeySample:
    prompt: str
    passkey: str


def build_passkey_samples(
    count: int,
    *,
    filler_sentences: int = 8,
    seed: int = 0,
) -> list[PasskeySample]:
    """Synthetic needle-in-a-haystack retrieval probes for context usage."""
    if count < 1:
        raise ValueError("count must be positive")
    generator = random.Random(seed)
    filler = "The grass is green and the sky is blue here. "
    samples: list[PasskeySample] = []
    for _ in range(count):
        passkey = str(generator.randint(10_000, 99_999))
        prompt = (
            f"The pass key is {passkey}. Remember it. "
            + filler * filler_sentences
            + "What is the pass key? The pass key is"
        )
        samples.append(PasskeySample(prompt=prompt, passkey=passkey))
    return samples


@torch.inference_mode()
def evaluate_passkey(
    model: GPT,
    tokenizer: ByteBPETokenizer,
    samples: Sequence[PasskeySample],
    device: torch.device,
    max_new_tokens: int = 8,
) -> dict[str, float]:
    """Greedy-decode each probe and require the passkey verbatim in the output."""
    model.eval()
    settings = GenerationConfig(max_new_tokens=max_new_tokens, temperature=0.0)
    correct = 0
    for sample in samples:
        prompt_ids = tokenizer.encode(sample.prompt)
        budget = model.config.context_length - max_new_tokens
        if budget < 1:
            raise ValueError("max_new_tokens leaves no room for the prompt")
        prompt_ids = prompt_ids[-budget:]
        input_ids = torch.tensor([prompt_ids], device=device)
        output = generate(model, input_ids, settings)
        completion = tokenizer.decode(output[0, len(prompt_ids) :].tolist())
        if sample.passkey in completion:
            correct += 1
    return {
        "task/passkey_accuracy": correct / len(samples),
        "task/passkey_samples": float(len(samples)),
    }
