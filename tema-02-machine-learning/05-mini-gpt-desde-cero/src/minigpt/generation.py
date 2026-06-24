"""Generación autoregresiva greedy y por muestreo."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from minigpt.model import MiniGPT
from minigpt.tokenizer import CharacterTokenizer


def validate_sampling(temperature: float, top_k: int | None, top_p: float | None) -> None:
    if temperature <= 0:
        raise ValueError("temperature debe ser mayor que cero")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k debe ser positivo")
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError("top_p debe cumplir 0 < top_p <= 1")


def filter_top_k(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    if top_k is None:
        return logits
    top_k = min(top_k, logits.shape[-1])
    threshold = torch.topk(logits, top_k).values[:, -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def filter_top_p(logits: torch.Tensor, top_p: float | None) -> torch.Tensor:
    if top_p is None or top_p >= 1:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    remove = cumulative > top_p
    remove[:, 1:] = remove[:, :-1].clone()
    remove[:, 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    result = torch.full_like(logits, float("-inf"))
    return result.scatter(1, sorted_indices, sorted_logits)


@torch.no_grad()
def generate_ids(
    model: MiniGPT,
    prompt_ids: list[int],
    max_new_tokens: int,
    device: torch.device,
    temperature: float = 0.8,
    top_k: int | None = None,
    top_p: float | None = None,
    greedy: bool = False,
) -> list[int]:
    """Genera exactamente max_new_tokens y conserva el prompt."""
    validate_sampling(temperature, top_k, top_p)
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens no puede ser negativo")
    model.eval()
    seed_ids = prompt_ids or [0]
    tokens = torch.tensor([seed_ids], dtype=torch.long, device=device)
    prompt_was_empty = not prompt_ids
    for _ in range(max_new_tokens):
        context = tokens[:, -model.config.block_size :]
        logits, _ = model(context)
        next_logits = logits[:, -1, :]
        if greedy or temperature < 1e-5:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        else:
            next_logits = next_logits / temperature
            next_logits = filter_top_k(next_logits, top_k)
            next_logits = filter_top_p(next_logits, top_p)
            probabilities = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
        tokens = torch.cat((tokens, next_token), dim=1)
    result = tokens[0].tolist()
    return result[1:] if prompt_was_empty else result


def generate_text(
    model: MiniGPT,
    tokenizer: CharacterTokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    **sampling: float | int | bool | None,
) -> str:
    ids = generate_ids(
        model,
        tokenizer.encode(prompt),
        max_new_tokens,
        device,
        **sampling,
    )
    return tokenizer.decode(ids)
