"""Command-line workflows for tokenization, training, evaluation, and inference."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import torch
import typer

from gpt2_rope.assets import download_gpt2_tokenizer
from gpt2_rope.benchmarking import benchmark_inference
from gpt2_rope.checkpoint import export_safetensors
from gpt2_rope.config import GenerationConfig, ProfilingConfig
from gpt2_rope.config_io import load_experiment_config
from gpt2_rope.data import MemmapTokenDataset, prepare_corpus, read_documents
from gpt2_rope.data_quality import (
    FilterThresholds,
    deduplicate_documents,
    filter_documents,
    write_shards,
)
from gpt2_rope.dpo import train_dpo
from gpt2_rope.evaluation import (
    build_passkey_samples,
    evaluate_multiple_choice,
    evaluate_passkey,
    evaluate_perplexity_files,
    load_multiple_choice_tasks,
)
from gpt2_rope.generation import generate
from gpt2_rope.lora import apply_lora, load_lora, set_lora_merged
from gpt2_rope.model import GPT
from gpt2_rope.quantization import quantization_report, quantize_model, save_quantized
from gpt2_rope.sweeps import load_sweep_config, run_sweep
from gpt2_rope.tokenizer import ByteBPETokenizer
from gpt2_rope.training import (
    evaluate,
    resolve_device,
    run_profiler,
    train_finetuning,
    train_pretraining,
)

app = typer.Typer(help="GPT-2 with grouped-query attention and rotary embeddings.")
tokenizer_app = typer.Typer(help="Train, download, and inspect byte-level BPE tokenizers.")
data_app = typer.Typer(help="Prepare, clean, and shard token datasets.")
checkpoint_app = typer.Typer(help="Inspect, export, and quantize checkpoints.")
eval_app = typer.Typer(help="Perplexity suites, logprob tasks, and passkey probes.")
sweep_app = typer.Typer(help="Local grid/random hyperparameter sweeps.")
benchmark_app = typer.Typer(help="Reproducible model performance benchmarks.")
app.add_typer(tokenizer_app, name="tokenizer")
app.add_typer(data_app, name="data")
app.add_typer(checkpoint_app, name="checkpoint")
app.add_typer(eval_app, name="eval")
app.add_typer(sweep_app, name="sweep")
app.add_typer(benchmark_app, name="benchmark")


def _tokenizer(directory: Path) -> ByteBPETokenizer:
    return ByteBPETokenizer.from_files(
        directory / "vocab.json",
        directory / "merges.txt",
    )


def _load_model(config_path: Path, checkpoint: Path, device: torch.device) -> GPT:
    config = load_experiment_config(config_path)
    model = GPT(config.model)
    model.load_state_dict(
        torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=True)
    )
    return model.to(device).eval()


@tokenizer_app.command("train")
def tokenizer_train(
    inputs: Annotated[list[Path], typer.Argument(help="UTF-8 text corpus files.")],
    output_dir: Annotated[Path, typer.Argument(help="Tokenizer output directory.")],
    vocab_size: Annotated[int, typer.Option(min=257)] = 50_257,
    eos_token: Annotated[str, typer.Option()] = "<|endoftext|>",
) -> None:
    documents = (
        line
        for path in inputs
        for line in path.read_text(encoding="utf-8").splitlines()
    )
    tokenizer = ByteBPETokenizer.train(
        documents,
        vocab_size=vocab_size,
        special_tokens=[eos_token],
    )
    tokenizer.save(output_dir)
    typer.echo(json.dumps(tokenizer.identity(), sort_keys=True))


@tokenizer_app.command("download")
def tokenizer_download(
    output_dir: Annotated[Path, typer.Argument(help="Tokenizer output directory.")],
) -> None:
    checksums = download_gpt2_tokenizer(output_dir)
    typer.echo(json.dumps(checksums, indent=2, sort_keys=True))


@tokenizer_app.command("inspect")
def tokenizer_inspect(
    tokenizer_dir: Annotated[Path, typer.Argument()],
    text: Annotated[str, typer.Option()] = "Hello, RoPE!",
) -> None:
    tokenizer = _tokenizer(tokenizer_dir)
    token_ids = tokenizer.encode(text)
    typer.echo(
        json.dumps(
            {
                "identity": tokenizer.identity(),
                "token_ids": token_ids,
                "tokens": [tokenizer.decoder[token_id] for token_id in token_ids],
                "round_trip": tokenizer.decode(token_ids),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@data_app.command("prepare")
def data_prepare(
    inputs: Annotated[list[Path], typer.Argument()],
    output_dir: Annotated[Path, typer.Argument()],
    tokenizer_dir: Annotated[Path, typer.Argument()],
    validation_fraction: Annotated[float, typer.Option(min=0.0, max=0.99)] = 0.01,
) -> None:
    manifest = prepare_corpus(
        inputs,
        output_dir,
        _tokenizer(tokenizer_dir),
        validation_fraction,
    )
    typer.echo(json.dumps(manifest, indent=2, sort_keys=True))


@data_app.command("dedup")
def data_dedup(
    inputs: Annotated[list[Path], typer.Argument(help="Text or JSONL corpus files.")],
    output: Annotated[Path, typer.Argument(help="Deduplicated one-document-per-line file.")],
    near_threshold: Annotated[
        float | None,
        typer.Option(min=0.0, max=1.0, help="MinHash Jaccard threshold; omit for exact-only."),
    ] = 0.9,
    num_hashes: Annotated[int, typer.Option(min=8)] = 64,
    shingle_size: Annotated[int, typer.Option(min=1)] = 3,
) -> None:
    documents, report = deduplicate_documents(
        read_documents(inputs),
        near_duplicate_threshold=near_threshold,
        num_hashes=num_hashes,
        shingle_size=shingle_size,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(documents) + "\n", encoding="utf-8")
    typer.echo(json.dumps(report.as_dict(), indent=2, sort_keys=True))


@data_app.command("filter")
def data_filter(
    inputs: Annotated[list[Path], typer.Argument(help="Text or JSONL corpus files.")],
    output: Annotated[Path, typer.Argument(help="Filtered one-document-per-line file.")],
    min_chars: Annotated[int, typer.Option(min=1)] = 8,
    max_chars: Annotated[int, typer.Option(min=1)] = 100_000,
    max_repetition: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.6,
    max_non_alpha: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.6,
) -> None:
    documents, report = filter_documents(
        read_documents(inputs),
        FilterThresholds(
            min_chars=min_chars,
            max_chars=max_chars,
            max_word_repetition_ratio=max_repetition,
            max_non_alpha_ratio=max_non_alpha,
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(documents) + "\n", encoding="utf-8")
    typer.echo(json.dumps(report.as_dict(), indent=2, sort_keys=True))


@data_app.command("shard")
def data_shard(
    inputs: Annotated[list[Path], typer.Argument(help="Text or JSONL corpus files.")],
    output_dir: Annotated[Path, typer.Argument(help="Shard output directory.")],
    documents_per_shard: Annotated[int, typer.Option(min=1)] = 10_000,
) -> None:
    manifest = write_shards(
        list(read_documents(inputs)),
        output_dir,
        documents_per_shard=documents_per_shard,
    )
    typer.echo(json.dumps(manifest, indent=2, sort_keys=True))


@app.command()
def pretrain(
    config_path: Annotated[Path, typer.Argument()],
    set_value: Annotated[
        list[str] | None,
        typer.Option("--set", help="Dotted override such as training.max_steps=10."),
    ] = None,
) -> None:
    train_pretraining(load_experiment_config(config_path, set_value))


@app.command()
def finetune(
    config_path: Annotated[Path, typer.Argument()],
    set_value: Annotated[list[str] | None, typer.Option("--set")] = None,
) -> None:
    train_finetuning(load_experiment_config(config_path, set_value))


@app.command()
def dpo(
    config_path: Annotated[Path, typer.Argument()],
    set_value: Annotated[list[str] | None, typer.Option("--set")] = None,
) -> None:
    train_dpo(load_experiment_config(config_path, set_value))


@app.command("evaluate")
def evaluate_command(
    config_path: Annotated[Path, typer.Argument()],
    checkpoint: Annotated[Path, typer.Argument()],
    batches: Annotated[int, typer.Option(min=1)] = 50,
) -> None:
    config = load_experiment_config(config_path)
    device = resolve_device(config.training.device)
    model = _load_model(config_path, checkpoint, device)
    validation_path = config.data.validation_path or config.data.train_path
    loader = torch.utils.data.DataLoader(
        MemmapTokenDataset(validation_path, config.data.sequence_length),
        batch_size=config.training.micro_batch_size,
    )
    metrics = evaluate(model, loader, device, batches, "fp32")
    typer.echo(json.dumps(metrics, indent=2, sort_keys=True))


@app.command("generate")
def generate_text(
    config_path: Annotated[Path, typer.Argument()],
    checkpoint: Annotated[Path, typer.Argument()],
    prompt: Annotated[list[str], typer.Option("--prompt")],
    max_new_tokens: Annotated[int, typer.Option(min=1)] = 64,
    temperature: Annotated[float, typer.Option(min=0)] = 1.0,
    top_k: Annotated[int | None, typer.Option(min=1)] = None,
    top_p: Annotated[float | None, typer.Option(min=0, max=1)] = None,
    repetition_penalty: Annotated[float, typer.Option(min=0.01)] = 1.0,
    seed: Annotated[int, typer.Option()] = 1337,
    lora: Annotated[
        Path | None,
        typer.Option(help="LoRA adapter safetensors to apply on top of the checkpoint."),
    ] = None,
) -> None:
    config = load_experiment_config(config_path)
    device = resolve_device(config.training.device)
    model = _load_model(config_path, checkpoint, device)
    if lora is not None:
        finetuning = config.finetuning
        apply_lora(
            model,
            rank=finetuning.lora_rank if finetuning else 8,
            alpha=finetuning.lora_alpha if finetuning else 16.0,
            target_modules=(
                finetuning.lora_targets
                if finetuning
                else ("q_proj", "k_proj", "v_proj", "out_proj", "fc", "proj")
            ),
        )
        load_lora(model, lora)
        # Merge so decoding pays no adapter overhead per token.
        set_lora_merged(model, True)
        model.to(device).eval()
    tokenizer = _tokenizer(config.data.tokenizer_dir)
    encoded = [tokenizer.encode(text) for text in prompt]
    if len({len(tokens) for tokens in encoded}) != 1:
        raise typer.BadParameter("batched prompts must have equal token lengths")
    input_ids = torch.tensor(encoded, device=device)
    settings = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        seed=seed,
        eos_token_id=tokenizer.eos_token_id,
    )
    outputs = generate(model, input_ids, settings)
    for row in outputs.tolist():
        typer.echo(tokenizer.decode(row))


@eval_app.command("suite")
def eval_suite(
    config_path: Annotated[Path, typer.Argument()],
    checkpoint: Annotated[Path, typer.Argument()],
    perplexity_file: Annotated[
        list[Path] | None,
        typer.Option("--perplexity-file", help="Held-out UTF-8 text files."),
    ] = None,
    task_file: Annotated[
        list[Path] | None,
        typer.Option("--task-file", help="Multiple-choice JSONL task files."),
    ] = None,
    passkey_samples: Annotated[
        int, typer.Option(min=0, help="Synthetic passkey probes; 0 disables.")
    ] = 0,
    passkey_filler: Annotated[int, typer.Option(min=0)] = 8,
    output: Annotated[
        Path | None,
        typer.Option(help="Optional JSON file for the metric report."),
    ] = None,
) -> None:
    config = load_experiment_config(config_path)
    device = resolve_device(config.training.device)
    model = _load_model(config_path, checkpoint, device)
    tokenizer = _tokenizer(config.data.tokenizer_dir)
    metrics: dict[str, float] = {}
    if perplexity_file:
        metrics.update(
            evaluate_perplexity_files(
                model,
                tokenizer,
                perplexity_file,
                device,
                config.data.sequence_length,
            )
        )
    for path in task_file or []:
        task_metrics = evaluate_multiple_choice(
            model,
            tokenizer,
            load_multiple_choice_tasks(path),
            device,
        )
        metrics.update({f"{key}/{path.stem}": value for key, value in task_metrics.items()})
    if passkey_samples:
        metrics.update(
            evaluate_passkey(
                model,
                tokenizer,
                build_passkey_samples(passkey_samples, filler_sentences=passkey_filler),
                device,
            )
        )
    if not metrics:
        raise typer.BadParameter("select at least one evaluation source")
    report = json.dumps(metrics, indent=2, sort_keys=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report + "\n", encoding="utf-8")
    typer.echo(report)


@sweep_app.command("run")
def sweep_run(
    sweep_path: Annotated[Path, typer.Argument(help="Sweep specification YAML.")],
) -> None:
    sweep = load_sweep_config(sweep_path)
    results = run_sweep(sweep)
    typer.echo(
        json.dumps(
            {
                "trials": len(results),
                "summary": str(sweep.output_dir / "sweep_summary.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.command("profile")
def profile_model(
    config_path: Annotated[Path, typer.Argument()],
    output_dir: Annotated[Path, typer.Argument()] = Path("profiler"),
) -> None:
    config = load_experiment_config(config_path)
    device = resolve_device(config.training.device)
    model = GPT(config.model).to(device).train()
    input_ids = torch.randint(
        0,
        config.model.vocab_size,
        (config.training.micro_batch_size, config.data.sequence_length),
        device=device,
    )
    optimizer = model.configure_optimizer(config.training.learning_rate, 0.0)

    def work() -> None:
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids, labels=input_ids, use_cache=False)
        if output.loss is None:
            raise RuntimeError("profiler model did not return loss")
        output.loss.backward()
        optimizer.step()

    run_profiler(
        work,
        output_dir,
        ProfilingConfig(),
        use_cuda=device.type == "cuda",
    )


@benchmark_app.command("inference")
def benchmark_inference_command(
    config_path: Annotated[Path, typer.Argument()],
    checkpoint: Annotated[Path, typer.Argument()],
    output: Annotated[
        Path | None,
        typer.Option(help="Optional JSON output for comparison across devices."),
    ] = None,
    batch_size: Annotated[int, typer.Option(min=1)] = 1,
    prompt_tokens: Annotated[int, typer.Option(min=1)] = 128,
    generated_tokens: Annotated[int, typer.Option(min=2)] = 32,
    warmup_runs: Annotated[int, typer.Option(min=0)] = 2,
    measured_runs: Annotated[int, typer.Option(min=1)] = 5,
) -> None:
    """Benchmark prefill, cached decoding, KV-cache bytes, and peak memory."""
    config = load_experiment_config(config_path)
    device = resolve_device(config.training.device)
    report = benchmark_inference(
        _load_model(config_path, checkpoint, device),
        device,
        batch_size=batch_size,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
        precision=config.training.precision,
        seed=config.training.seed,
    )
    payload = json.dumps(report.as_dict(), indent=2, sort_keys=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    typer.echo(payload)


@app.command("serve")
def serve(
    config_path: Annotated[Path, typer.Argument()],
    checkpoint: Annotated[Path, typer.Argument()],
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65_535)] = 8000,
    max_batch_size: Annotated[int, typer.Option(min=1)] = 8,
    batch_window_ms: Annotated[float, typer.Option(min=0)] = 10.0,
    metrics_dir: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Serve KV-cached generation over HTTP (requires the 'serving' extra)."""
    try:
        import uvicorn

        from gpt2_rope.serving import InferenceService, create_app
    except ModuleNotFoundError as error:
        raise typer.BadParameter(
            "serving dependencies are missing; install with 'uv sync --extra serving'"
        ) from error
    config = load_experiment_config(config_path)
    device = resolve_device(config.training.device)
    service = InferenceService(
        _load_model(config_path, checkpoint, device),
        _tokenizer(config.data.tokenizer_dir),
        device,
        max_batch_size=max_batch_size,
        batch_window_ms=batch_window_ms,
        metrics_dir=metrics_dir,
    )
    uvicorn.run(create_app(service), host=host, port=port)


@checkpoint_app.command("inspect")
def checkpoint_inspect(checkpoint: Annotated[Path, typer.Argument()]) -> None:
    typer.echo((checkpoint / "metadata.json").read_text(encoding="utf-8"))


@checkpoint_app.command("export")
def checkpoint_export(
    config_path: Annotated[Path, typer.Argument()],
    checkpoint: Annotated[Path, typer.Argument()],
    output: Annotated[Path, typer.Argument()],
) -> None:
    model = _load_model(config_path, checkpoint, torch.device("cpu"))
    export_safetensors(model, output)
    typer.echo(str(output))


@checkpoint_app.command("quantize")
def checkpoint_quantize(
    config_path: Annotated[Path, typer.Argument()],
    checkpoint: Annotated[Path, typer.Argument()],
    output: Annotated[Path, typer.Argument(help="Quantized safetensors output path.")],
) -> None:
    model = _load_model(config_path, checkpoint, torch.device("cpu"))
    quantize_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_quantized(model, output)
    typer.echo(json.dumps(quantization_report(model), indent=2, sort_keys=True))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    app()


if __name__ == "__main__":
    main()
