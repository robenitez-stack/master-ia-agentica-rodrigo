"""Visualizaciones opcionales para entrenamiento e interpretabilidad."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from minigpt.generation import generate_text
from minigpt.model import MiniGPT
from minigpt.tokenizer import CharacterTokenizer


def plot_losses(history: list[dict[str, float | int]]) -> plt.Figure:
    if not history:
        raise ValueError("El historial está vacío")
    figure, axis = plt.subplots(figsize=(8, 4.5))
    iterations = [int(record["iteration"]) for record in history]
    axis.plot(iterations, [record["train_loss"] for record in history], label="train")
    axis.plot(iterations, [record["val_loss"] for record in history], label="val")
    axis.set(xlabel="Iteración", ylabel="Cross-entropy", title="Curva de entrenamiento")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    return figure


@torch.no_grad()
def plot_attention(
    model: MiniGPT,
    tokenizer: CharacterTokenizer,
    prompt: str,
    layer: int,
    head: int,
    device: torch.device,
) -> plt.Figure:
    if not prompt:
        raise ValueError("El prompt de atención no puede estar vacío")
    ids = tokenizer.encode(prompt)
    if len(ids) > model.config.block_size:
        raise ValueError("El prompt supera block_size")
    model.eval()
    tensor = torch.tensor([ids], dtype=torch.long, device=device)
    model(tensor)
    attention = model.attention_weights(layer, head)[0].cpu().numpy()
    labels = [char.replace(" ", "␣").replace("\n", "⏎") for char in prompt]
    figure, axis = plt.subplots(figsize=(max(6, len(labels) * 0.35), 5))
    image = axis.imshow(attention, cmap="viridis", vmin=0)
    axis.set_xticks(range(len(labels)), labels=labels)
    axis.set_yticks(range(len(labels)), labels=labels)
    axis.set_title(f"Atención: capa {layer}, cabeza {head}")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    return figure


@torch.no_grad()
def plot_next_token_distribution(
    model: MiniGPT,
    tokenizer: CharacterTokenizer,
    prompt: str,
    device: torch.device,
    temperature: float = 1.0,
    top_n: int = 10,
) -> plt.Figure:
    if not prompt:
        raise ValueError("El prompt no puede estar vacío")
    if temperature <= 0:
        raise ValueError("temperature debe ser positiva")
    ids = tokenizer.encode(prompt)
    context = torch.tensor([ids[-model.config.block_size :]], device=device)
    logits, _ = model(context)
    probabilities = F.softmax(logits[0, -1] / temperature, dim=-1)
    values, indices = probabilities.topk(min(top_n, tokenizer.vocab_size))
    labels = [tokenizer.decode([index]) for index in indices.cpu().tolist()]
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(labels, values.cpu().tolist())
    axis.set(ylabel="Probabilidad", title=f"Siguiente carácter (T={temperature})")
    figure.tight_layout()
    return figure


def compare_temperatures(
    model: MiniGPT,
    tokenizer: CharacterTokenizer,
    prompt: str,
    temperatures: Iterable[float],
    max_new_tokens: int,
    device: torch.device,
) -> dict[float, str]:
    return {
        temperature: generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens,
            device,
            temperature=temperature,
        )
        for temperature in temperatures
    }


def embedding_neighbors(
    model: MiniGPT, tokenizer: CharacterTokenizer, character: str, k: int = 5
) -> list[tuple[str, float]]:
    if len(character) != 1:
        raise ValueError("Se requiere exactamente un carácter")
    token_id = tokenizer.encode(character)[0]
    embeddings = model.token_embedding.weight.detach().float()
    normalized = F.normalize(embeddings, dim=1)
    similarities = normalized @ normalized[token_id]
    values, indices = torch.topk(similarities, min(k + 1, tokenizer.vocab_size))
    neighbors = []
    for value, index in zip(values.tolist(), indices.tolist(), strict=True):
        if index != token_id:
            neighbors.append((tokenizer.decode([index]), float(value)))
    return neighbors[:k]


def attention_array(model: MiniGPT, layer: int, head: int) -> np.ndarray:
    """Acceso validado para pruebas y análisis sin graficar."""
    return model.attention_weights(layer, head)[0].detach().cpu().numpy()
