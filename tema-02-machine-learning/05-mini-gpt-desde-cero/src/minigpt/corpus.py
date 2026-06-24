"""Carga y descarga explícita de corpus de texto."""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

QUIJOTE_URL = "https://www.gutenberg.org/cache/epub/2000/pg2000.txt"


def download_corpus(output: Path, url: str = QUIJOTE_URL) -> Path:
    """Descarga el corpus solo cuando todavía no existe."""
    output = Path(output)
    if output.exists():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, output)
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(
            f"No se pudo descargar el corpus. Copia manualmente un UTF-8 en {output}"
        ) from exc
    return output


def load_corpus(path: Path) -> str:
    """Lee un corpus UTF-8 local y valida que contenga texto."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el corpus {path}. Usa `minigpt download --output {path}` "
            "o copia allí un archivo .txt UTF-8."
        )
    if path.suffix.lower() != ".txt":
        raise ValueError("El corpus debe ser un archivo .txt")
    text = path.read_text(encoding="utf-8")
    if not text:
        raise ValueError("El corpus está vacío")
    return text
