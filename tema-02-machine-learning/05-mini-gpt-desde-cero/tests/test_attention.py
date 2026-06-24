import torch

from minigpt.config import ModelConfig
from minigpt.model import CausalSelfAttentionHead


def test_causal_attention_shape_and_probabilities() -> None:
    config = ModelConfig(vocab_size=5, n_embed=16, n_head=4, n_layer=1, block_size=8, dropout=0.0)
    head = CausalSelfAttentionHead(config).eval()
    output = head(torch.randn(2, 6, 16))
    assert output.shape == (2, 6, 4)
    weights = head.last_attention
    assert weights is not None
    future = torch.triu(weights, diagonal=1)
    assert torch.all(future < 1e-7)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 6), atol=1e-6)
