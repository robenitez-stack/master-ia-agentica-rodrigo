import torch

from minigpt.checkpoint import load_model
from minigpt.config import ModelConfig, TrainingConfig
from minigpt.data import encode_and_split
from minigpt.generation import generate_text
from minigpt.model import MiniGPT
from minigpt.tokenizer import CharacterTokenizer
from minigpt.training import train_model


def test_offline_cpu_smoke_training(tmp_path) -> None:
    text = "En un lugar de la Mancha abcdefghijklmnopqrstuvwxyz.\n" * 20
    tokenizer = CharacterTokenizer.from_text(text)
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        n_embed=16,
        n_head=4,
        n_layer=1,
        block_size=8,
        dropout=0.0,
    )
    training_config = TrainingConfig(
        batch_size=4,
        learning_rate=1e-3,
        max_iters=3,
        eval_interval=1,
        eval_iters=1,
        seed=11,
    )
    train_tokens, val_tokens = encode_and_split(text, tokenizer)
    model = MiniGPT(model_config)
    history, best_loss = train_model(
        model,
        tokenizer,
        train_tokens,
        val_tokens,
        model_config,
        training_config,
        tmp_path,
        torch.device("cpu"),
    )
    assert history and torch.isfinite(torch.tensor(best_loss))
    assert (tmp_path / "best.pt").exists() and (tmp_path / "last.pt").exists()
    restored, restored_tokenizer, _ = load_model(tmp_path / "best.pt", torch.device("cpu"))
    generated = generate_text(
        restored,
        restored_tokenizer,
        "En ",
        5,
        torch.device("cpu"),
        top_k=3,
    )
    assert len(generated) == 8
