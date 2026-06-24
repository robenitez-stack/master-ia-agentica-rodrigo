"""Codificación, partición y batches auto-supervisados."""

from __future__ import annotations

import torch

from minigpt.tokenizer import CharacterTokenizer


def encode_and_split(
    text: str, tokenizer: CharacterTokenizer, train_fraction: float = 0.9
) -> tuple[torch.Tensor, torch.Tensor]:
    """Codifica y divide tokens de forma determinista en 90/10."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction debe estar entre 0 y 1")
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split_index = int(len(tokens) * train_fraction)
    return tokens[:split_index], tokens[split_index:]


def get_batch(
    split: str,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crea entradas y targets desplazados un token."""
    if split not in {"train", "val"}:
        raise ValueError("split debe ser 'train' o 'val'")
    data = train_tokens if split == "train" else val_tokens
    if len(data) <= block_size:
        raise ValueError(
            f"El corpus de {split} tiene {len(data)} tokens; necesita más de {block_size}"
        )
    starts = torch.randint(
        0,
        len(data) - block_size,
        (batch_size,),
        generator=generator,
    )
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x.to(device=device, dtype=torch.long), y.to(device=device, dtype=torch.long)
