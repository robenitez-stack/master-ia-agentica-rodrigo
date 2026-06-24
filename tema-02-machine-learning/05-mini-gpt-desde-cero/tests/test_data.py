import pytest
import torch

from minigpt.data import get_batch


def test_batch_shape_dtype_and_shift() -> None:
    train = torch.arange(100, dtype=torch.long)
    val = torch.arange(30, dtype=torch.long)
    generator = torch.Generator().manual_seed(1)
    x, y = get_batch("train", train, val, 4, 8, torch.device("cpu"), generator)
    assert x.shape == y.shape == (4, 8)
    assert x.dtype == y.dtype == torch.long
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_short_corpus_error() -> None:
    data = torch.arange(4)
    with pytest.raises(ValueError, match="necesita más"):
        get_batch("val", data, data, 2, 4, torch.device("cpu"))
