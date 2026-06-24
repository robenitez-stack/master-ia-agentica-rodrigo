"""Tokenizador reversible a nivel de carácter."""

from __future__ import annotations

import json
from pathlib import Path


class CharacterTokenizer:
    """Mapea cada carácter conocido a un entero."""

    def __init__(self, characters: list[str]) -> None:
        unique = sorted(set(characters))
        if not unique:
            raise ValueError("El vocabulario no puede estar vacío")
        self._characters = unique
        self._stoi = {char: index for index, char in enumerate(unique)}

    @classmethod
    def from_text(cls, text: str) -> CharacterTokenizer:
        if not text:
            raise ValueError("No se puede construir el tokenizador desde texto vacío")
        return cls(list(text))

    @property
    def vocab_size(self) -> int:
        return len(self._characters)

    @property
    def characters(self) -> list[str]:
        return list(self._characters)

    def encode(self, text: str) -> list[int]:
        unknown = sorted(set(text) - self._stoi.keys())
        if unknown:
            raise ValueError(f"Caracteres desconocidos: {unknown!r}")
        return [self._stoi[char] for char in text]

    def decode(self, ids: list[int]) -> str:
        try:
            return "".join(self._characters[index] for index in ids)
        except (IndexError, TypeError) as exc:
            raise ValueError("La secuencia contiene IDs fuera del vocabulario") from exc

    def save(self, path: Path) -> None:
        Path(path).write_text(
            json.dumps({"characters": self._characters}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> CharacterTokenizer:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload["characters"])
