from __future__ import annotations

import pytest
import torch

from minigpt.config import ModelConfig
from minigpt.model import MiniGPT
from minigpt.tokenizer import CharacterTokenizer


@pytest.fixture
def tokenizer() -> CharacterTokenizer:
    return CharacterTokenizer.from_text("abc \n")


@pytest.fixture
def tiny_model(tokenizer: CharacterTokenizer) -> MiniGPT:
    torch.manual_seed(7)
    return MiniGPT(
        ModelConfig(
            vocab_size=tokenizer.vocab_size,
            n_embed=16,
            n_head=4,
            n_layer=2,
            block_size=8,
            dropout=0.0,
        )
    )
