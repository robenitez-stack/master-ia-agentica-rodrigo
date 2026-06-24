import pytest
import torch

from minigpt.config import ModelConfig
from minigpt.model import MiniGPT


def test_logits_loss_and_backward(tiny_model: MiniGPT) -> None:
    inputs = torch.randint(0, tiny_model.config.vocab_size, (3, 6))
    targets = torch.randint(0, tiny_model.config.vocab_size, (3, 6))
    logits, no_loss = tiny_model(inputs)
    assert logits.shape == (3, 6, tiny_model.config.vocab_size)
    assert no_loss is None
    _, loss = tiny_model(inputs, targets)
    assert loss is not None and loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in tiny_model.parameters())


def test_invalid_head_divisibility() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(vocab_size=10, n_embed=15, n_head=4)
