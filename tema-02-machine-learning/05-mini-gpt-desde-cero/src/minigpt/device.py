"""Selección de dispositivo y semillas."""

from __future__ import annotations

import random

import torch


def select_device(requested: str = "auto") -> torch.device:
    """Selecciona CUDA > MPS > CPU o valida un dispositivo forzado."""
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("device debe ser auto, cpu, cuda o mps")
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA fue solicitado, pero no está disponible")
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS fue solicitado, pero no está disponible")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    """Configura semillas de Python, PyTorch y CUDA."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
