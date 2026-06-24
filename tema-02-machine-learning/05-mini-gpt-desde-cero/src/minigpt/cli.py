"""Interfaz de línea de comandos del proyecto."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minigpt.checkpoint import load_model, load_payload
from minigpt.config import get_profile
from minigpt.corpus import download_corpus, load_corpus
from minigpt.data import encode_and_split
from minigpt.device import seed_everything, select_device
from minigpt.generation import generate_text
from minigpt.model import MiniGPT
from minigpt.tokenizer import CharacterTokenizer
from minigpt.training import checkpoint_summary, train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mini-GPT decoder-only educativo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Descargar Don Quijote")
    download.add_argument("--output", type=Path, required=True)

    train = subparsers.add_parser("train", help="Entrenar un mini-GPT")
    train.add_argument("--corpus", type=Path, required=True)
    train.add_argument("--profile", choices=["academic", "demo", "cpu_light"], default="demo")
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    train.add_argument("--resume", type=Path)
    train.add_argument("--max-iters", type=int, help="Override útil para smoke tests")

    generate = subparsers.add_parser("generate", help="Generar texto desde un checkpoint")
    generate.add_argument("--checkpoint", type=Path, required=True)
    generate.add_argument("--prompt", default="")
    generate.add_argument("--max-new-tokens", type=int, default=300)
    generate.add_argument("--temperature", type=float, default=0.8)
    generate.add_argument("--top-k", type=int)
    generate.add_argument("--top-p", type=float)
    generate.add_argument("--greedy", action="store_true")
    generate.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")

    inspect = subparsers.add_parser("inspect", help="Inspeccionar un checkpoint")
    inspect.add_argument("--checkpoint", type=Path, required=True)
    return parser


def run_train(args: argparse.Namespace) -> None:
    text = load_corpus(args.corpus)
    tokenizer = CharacterTokenizer.from_text(text)
    model_config, training_config = get_profile(args.profile, tokenizer.vocab_size)
    if args.max_iters is not None:
        if args.max_iters <= 0:
            raise ValueError("--max-iters debe ser positivo")
        values = training_config.to_dict()
        values["max_iters"] = args.max_iters
        values["eval_interval"] = min(int(values["eval_interval"]), args.max_iters)
        values["eval_iters"] = min(int(values["eval_iters"]), 5)
        training_config = type(training_config)(**values)
    device = select_device(args.device)
    seed_everything(training_config.seed)
    train_tokens, val_tokens = encode_and_split(text, tokenizer)
    model = MiniGPT(model_config).to(device)
    print(f"Dispositivo: {device}; parámetros: {model.count_parameters():,}")
    train_model(
        model,
        tokenizer,
        train_tokens,
        val_tokens,
        model_config,
        training_config,
        args.output_dir,
        device,
        args.resume,
    )


def run_generate(args: argparse.Namespace) -> None:
    device = select_device(args.device)
    model, tokenizer, _ = load_model(args.checkpoint, device)
    print(
        generate_text(
            model,
            tokenizer,
            args.prompt,
            args.max_new_tokens,
            device,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            greedy=args.greedy,
        )
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "download":
        print(download_corpus(args.output))
    elif args.command == "train":
        run_train(args)
    elif args.command == "generate":
        run_generate(args)
    elif args.command == "inspect":
        payload = load_payload(args.checkpoint)
        summary = checkpoint_summary(payload)
        config, _ = get_profile("demo", summary["vocab_size"])
        config_values = dict(summary["model_config"])
        config = type(config)(**config_values)
        parameter_count = MiniGPT(config).count_parameters()
        summary["parameters"] = parameter_count
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
