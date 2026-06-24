"""Persistencia completa de entrenamiento y carga para inferencia."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

from minigpt.config import ModelConfig, TrainingConfig
from minigpt.model import MiniGPT
from minigpt.tokenizer import CharacterTokenizer


def save_checkpoint(
    path: Path,
    model: MiniGPT,
    optimizer: torch.optim.Optimizer,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    tokenizer: CharacterTokenizer,
    iteration: int,
    best_val_loss: float,
    history: list[dict[str, float | int]],
    device: torch.device,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_config": model_config.to_dict(),
            "training_config": training_config.to_dict(),
            "vocabulary": tokenizer.characters,
            "iteration": iteration,
            "best_val_loss": best_val_loss,
            "history": history,
            "seed": training_config.seed,
            "device": str(device),
        },
        path,
    )


def load_payload(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def _filter_dataclass_values(cls: type, values: dict[str, Any]) -> dict[str, Any]:
    valid = {field.name for field in fields(cls)}
    return {key: value for key, value in values.items() if key in valid}


def load_model(
    path: Path, device: torch.device
) -> tuple[MiniGPT, CharacterTokenizer, dict[str, Any]]:
    payload = load_payload(path, device)
    config = ModelConfig(**_filter_dataclass_values(ModelConfig, payload["model_config"]))
    model = MiniGPT(config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    tokenizer = CharacterTokenizer(payload["vocabulary"])
    return model, tokenizer, payload
