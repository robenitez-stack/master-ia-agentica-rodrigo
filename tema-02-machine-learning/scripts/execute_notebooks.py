import argparse
import os
import subprocess
import sys
from pathlib import Path


TEMA2 = Path(__file__).resolve().parents[1]
MODULE2 = TEMA2 / "02-deep-learning-redes-neuronales"

NOTEBOOKS = {
    "02-01": {
        "path": MODULE2 / "notebooks" / "01_neurona_manual_numpy_mnist.ipynb",
        "results": MODULE2 / "results" / "01_neurona_manual_numpy_mnist",
    },
    "02-02": {
        "path": MODULE2 / "notebooks" / "02_red_neuronal_desde_cero_pytorch.ipynb",
        "results": MODULE2 / "results" / "02_red_neuronal_desde_cero_pytorch",
    },
}


def execute(notebook_id: str, mode: str, timeout: int, kernel: str) -> None:
    config = NOTEBOOKS[notebook_id]
    output_name = "notebook_executed.fast.ipynb" if mode == "fast" else "notebook_executed.ipynb"
    config["results"].mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["RUN_MODE"] = mode
    command = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(config["path"]),
        "--output",
        output_name,
        "--output-dir",
        str(config["results"]),
        f"--ExecutePreprocessor.timeout={timeout}",
        f"--ExecutePreprocessor.kernel_name={kernel}",
    ]
    print(f"Ejecutando {notebook_id} en modo {mode}...")
    subprocess.run(command, cwd=TEMA2.parent, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ejecuta notebooks del Tema 2 desde un kernel limpio")
    parser.add_argument("--target", choices=["02-01", "02-02", "module2", "all"], required=True)
    parser.add_argument("--mode", choices=["fast", "full"], required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--kernel", default="master-ia-agentica")
    args = parser.parse_args()

    selected = [args.target] if args.target in NOTEBOOKS else list(NOTEBOOKS)
    for notebook_id in selected:
        execute(notebook_id, args.mode, args.timeout, args.kernel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
