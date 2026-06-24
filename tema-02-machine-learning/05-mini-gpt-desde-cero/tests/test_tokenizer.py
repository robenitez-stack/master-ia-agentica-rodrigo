import pytest

from minigpt.tokenizer import CharacterTokenizer


def test_round_trip(tokenizer: CharacterTokenizer) -> None:
    text = "ab cab\n"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_serialization_preserves_vocabulary(tmp_path, tokenizer: CharacterTokenizer) -> None:
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    restored = CharacterTokenizer.load(path)
    assert restored.characters == tokenizer.characters
    assert restored.decode(restored.encode("abc")) == "abc"


def test_unknown_character_has_clear_error(tokenizer: CharacterTokenizer) -> None:
    with pytest.raises(ValueError, match="desconocidos"):
        tokenizer.encode("z")
