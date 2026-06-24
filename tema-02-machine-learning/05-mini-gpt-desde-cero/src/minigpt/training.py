"""Bucle de entrenamiento, evaluación, scheduler e historial."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from minigpt.checkpoint import load_payload, save_checkpoint
from minigpt.config import ModelConfig, TrainingConfig
from minigpt.data import get_batch
from minigpt.model import MiniGPT
from minigpt.tokenizer import CharacterTokenizer


@torch.no_grad()
def estimate_losses(
    model: MiniGPT,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    config: TrainingConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    results: dict[str, float] = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(config.eval_iters):
            x, y = get_batch(
                split,
                train_tokens,
                val_tokens,
                config.batch_size,
                model.config.block_size,
                device,
            )
            _, loss = model(x, y)
            assert loss is not None
            losses.append(loss.item())
        results[split] = sum(losses) / len(losses)
    model.train()
    return results


def cosine_factor(iteration: int, config: TrainingConfig) -> float:
    if not config.use_scheduler:
        return 1.0
    if config.warmup_iters and iteration < config.warmup_iters:
        return (iteration + 1) / config.warmup_iters
    remaining = max(1, config.max_iters - config.warmup_iters)
    progress = min(1.0, (iteration - config.warmup_iters) / remaining)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return config.min_lr_ratio + (1 - config.min_lr_ratio) * cosine


def train_model(
    model: MiniGPT,
    tokenizer: CharacterTokenizer,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    model_config: ModelConfig,
    config: TrainingConfig,
    output_dir: Path,
    device: torch.device,
    resume: Path | None = None,
) -> tuple[list[dict[str, float | int]], float]:
    """Entrena, evalúa, registra historial y guarda best.pt/last.pt."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    start_iteration = 0
    best_val_loss = float("inf")
    history: list[dict[str, float | int]] = []
    if resume:
        payload = load_payload(resume, device)
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        start_iteration = int(payload["iteration"]) + 1
        best_val_loss = float(payload["best_val_loss"])
        history = list(payload["history"])

    print(
        f"Loss inicial de referencia ~= log(vocab_size) = {math.log(model_config.vocab_size):.4f}"
    )
    started = time.perf_counter()
    model.train()
    for iteration in range(start_iteration, config.max_iters):
        factor = cosine_factor(iteration, config)
        learning_rate = config.learning_rate * factor
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        x, y = get_batch(
            "train",
            train_tokens,
            val_tokens,
            config.batch_size,
            model_config.block_size,
            device,
        )
        _, loss = model(x, y)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if config.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()

        should_evaluate = iteration % config.eval_interval == 0 or iteration == config.max_iters - 1
        if should_evaluate:
            losses = estimate_losses(model, train_tokens, val_tokens, config, device)
            elapsed = time.perf_counter() - started
            record: dict[str, float | int] = {
                "iteration": iteration,
                "train_loss": losses["train"],
                "val_loss": losses["val"],
                "learning_rate": learning_rate,
                "elapsed_seconds": elapsed,
                "seconds_per_iteration": elapsed / (iteration - start_iteration + 1),
            }
            history.append(record)
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(
                    output_dir / "best.pt",
                    model,
                    optimizer,
                    model_config,
                    config,
                    tokenizer,
                    iteration,
                    best_val_loss,
                    history,
                    device,
                )
            print(
                f"iter {iteration:5d} | train {losses['train']:.4f} | "
                f"val {losses['val']:.4f} | lr {learning_rate:.2e}"
            )

    save_checkpoint(
        output_dir / "last.pt",
        model,
        optimizer,
        model_config,
        config,
        tokenizer,
        config.max_iters - 1,
        best_val_loss,
        history,
        device,
    )
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return history, best_val_loss


def checkpoint_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_config": payload["model_config"],
        "training_config": payload["training_config"],
        "vocab_size": len(payload["vocabulary"]),
        "iteration": payload["iteration"],
        "best_val_loss": payload["best_val_loss"],
        "seed": payload["seed"],
        "device": payload["device"],
    }
