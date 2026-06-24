"""Configuración validada para el modelo y el entrenamiento."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Hiperparámetros estructurales del mini-GPT."""

    vocab_size: int
    n_embed: int = 256
    n_head: int = 4
    n_layer: int = 4
    block_size: int = 128
    dropout: float = 0.1

    def __post_init__(self) -> None:
        for name in ("vocab_size", "n_embed", "n_head", "n_layer", "block_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} debe ser positivo")
        if self.n_embed % self.n_head != 0:
            raise ValueError("n_embed debe ser divisible por n_head")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout debe cumplir 0 <= dropout < 1")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingConfig:
    """Hiperparámetros del proceso de optimización."""

    batch_size: int = 64
    learning_rate: float = 3e-4
    max_iters: int = 5000
    eval_interval: int = 250
    eval_iters: int = 50
    seed: int = 1337
    gradient_clip: float | None = 1.0
    use_scheduler: bool = False
    warmup_iters: int = 100
    min_lr_ratio: float = 0.1

    def __post_init__(self) -> None:
        for name in ("batch_size", "max_iters", "eval_interval", "eval_iters"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} debe ser positivo")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate debe ser positivo")
        if self.gradient_clip is not None and self.gradient_clip <= 0:
            raise ValueError("gradient_clip debe ser positivo")
        if self.warmup_iters < 0:
            raise ValueError("warmup_iters no puede ser negativo")
        if not 0 < self.min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio debe cumplir 0 < ratio <= 1")

    def to_dict(self) -> dict[str, int | float | bool | None]:
        return asdict(self)


def get_profile(name: str, vocab_size: int) -> tuple[ModelConfig, TrainingConfig]:
    """Construye uno de los perfiles académicos solicitados."""
    profiles = {
        "academic": (
            dict(n_embed=256, n_head=4, n_layer=4, block_size=128, dropout=0.1),
            dict(
                batch_size=64,
                learning_rate=3e-4,
                max_iters=5000,
                eval_interval=250,
                eval_iters=50,
            ),
        ),
        "demo": (
            dict(n_embed=256, n_head=4, n_layer=4, block_size=128, dropout=0.1),
            dict(
                batch_size=64,
                learning_rate=3e-4,
                max_iters=500,
                eval_interval=250,
                eval_iters=50,
            ),
        ),
        "cpu_light": (
            dict(n_embed=128, n_head=4, n_layer=3, block_size=64, dropout=0.1),
            dict(
                batch_size=32,
                learning_rate=3e-4,
                max_iters=2000,
                eval_interval=100,
                eval_iters=20,
            ),
        ),
    }
    if name not in profiles:
        raise ValueError(f"Perfil desconocido: {name}. Opciones: {', '.join(profiles)}")
    model_values, training_values = profiles[name]
    return ModelConfig(vocab_size=vocab_size, **model_values), TrainingConfig(**training_values)
