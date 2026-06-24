import pytest
import torch

from minigpt.generation import filter_top_p, generate_ids, validate_sampling
from minigpt.model import MiniGPT


@pytest.mark.parametrize(
    ("top_k", "top_p"),
    [(None, None), (2, None), (None, 0.8)],
)
def test_generation_length(tiny_model: MiniGPT, top_k: int | None, top_p: float | None) -> None:
    ids = generate_ids(
        tiny_model,
        [1, 2],
        5,
        torch.device("cpu"),
        top_k=top_k,
        top_p=top_p,
    )
    assert len(ids) == 7


def test_empty_prompt(tiny_model: MiniGPT) -> None:
    ids = generate_ids(tiny_model, [], 4, torch.device("cpu"), greedy=True)
    assert len(ids) == 4


def test_invalid_temperature() -> None:
    with pytest.raises(ValueError, match="temperature"):
        validate_sampling(0, None, None)


def test_top_p_keeps_at_least_one_candidate() -> None:
    filtered = filter_top_p(torch.tensor([[10.0, 9.0, 8.0]]), 0.01)
    assert torch.isfinite(filtered).sum().item() >= 1
