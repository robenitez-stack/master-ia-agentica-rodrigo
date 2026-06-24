"""Arquitectura decoder-only implementada sin torch.nn.Transformer."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from minigpt.config import ModelConfig


class CausalSelfAttentionHead(nn.Module):
    """Una cabeza de self-attention causal con Q, K y V explícitos."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        head_size = config.n_embed // config.n_head
        self.key = nn.Linear(config.n_embed, head_size, bias=False)
        self.query = nn.Linear(config.n_embed, head_size, bias=False)
        self.value = nn.Linear(config.n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool)),
        )
        self.last_attention: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, time, _ = x.shape
        key = self.key(x)
        query = self.query(x)
        value = self.value(x)
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(key.shape[-1])
        scores = scores.masked_fill(~self.causal_mask[:time, :time], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        self.last_attention = weights.detach()
        return self.dropout(weights) @ value


class MultiHeadAttention(nn.Module):
    """Concatena varias cabezas y proyecta al ancho del embedding."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.heads = nn.ModuleList([CausalSelfAttentionHead(config) for _ in range(config.n_head)])
        self.projection = nn.Linear(config.n_embed, config.n_embed)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        concatenated = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.projection(concatenated))


class FeedForward(nn.Module):
    """MLP por posición con expansión 4x y activación ReLU."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(config.n_embed, 4 * config.n_embed),
            nn.ReLU(),
            nn.Linear(4 * config.n_embed, config.n_embed),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TransformerBlock(nn.Module):
    """Bloque pre-norm con atención, MLP y conexiones residuales."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.norm_attention = nn.LayerNorm(config.n_embed)
        self.norm_feed_forward = nn.LayerNorm(config.n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm_attention(x))
        return x + self.feed_forward(self.norm_feed_forward(x))


class MiniGPT(nn.Module):
    """Modelo de lenguaje autoregresivo decoder-only."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embed)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embed)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.final_norm = nn.LayerNorm(config.n_embed)
        self.language_head = nn.Linear(config.n_embed, config.vocab_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, token_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, time = token_ids.shape
        if time > self.config.block_size:
            raise ValueError(
                f"Secuencia de {time} tokens supera block_size={self.config.block_size}"
            )
        positions = torch.arange(time, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        logits = self.language_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(batch * time, self.config.vocab_size),
                targets.reshape(batch * time),
            )
        return logits, loss

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def attention_weights(self, layer: int, head: int) -> torch.Tensor:
        """Devuelve la última matriz de atención calculada."""
        if not 0 <= layer < len(self.blocks):
            raise ValueError("Capa fuera de rango")
        heads = self.blocks[layer].attention.heads
        if not 0 <= head < len(heads):
            raise ValueError("Cabeza fuera de rango")
        weights = heads[head].last_attention
        if weights is None:
            raise RuntimeError("Ejecuta un forward antes de consultar la atención")
        return weights
