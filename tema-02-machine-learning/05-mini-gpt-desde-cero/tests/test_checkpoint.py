import torch

from minigpt.checkpoint import load_model, save_checkpoint
from minigpt.config import TrainingConfig
from minigpt.model import MiniGPT
from minigpt.tokenizer import CharacterTokenizer


def test_checkpoint_round_trip(
    tmp_path, tiny_model: MiniGPT, tokenizer: CharacterTokenizer
) -> None:
    optimizer = torch.optim.AdamW(tiny_model.parameters())
    path = tmp_path / "model.pt"
    save_checkpoint(
        path,
        tiny_model,
        optimizer,
        tiny_model.config,
        TrainingConfig(max_iters=2, eval_interval=1, eval_iters=1, batch_size=2),
        tokenizer,
        1,
        1.23,
        [{"iteration": 1, "train_loss": 1.2, "val_loss": 1.23}],
        torch.device("cpu"),
    )
    restored, restored_tokenizer, payload = load_model(path, torch.device("cpu"))
    tiny_model.eval()
    sample = torch.tensor([[0, 1, 2]])
    assert torch.allclose(tiny_model(sample)[0], restored(sample)[0])
    assert restored.config == tiny_model.config
    assert restored_tokenizer.characters == tokenizer.characters
    assert payload["best_val_loss"] == 1.23
