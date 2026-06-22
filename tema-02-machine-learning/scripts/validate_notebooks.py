import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import nbformat


TEMA2 = Path(__file__).resolve().parents[1]
MODULE2_RESULTS = TEMA2 / "02-deep-learning-redes-neuronales" / "results"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_hash_manifest(manifest: Path, base: Path) -> list[str]:
    errors = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, filename = line.split(maxsplit=1)
        path = base / filename.strip()
        if not path.exists():
            errors.append(f"Falta archivo protegido: {path}")
        elif sha256(path) != expected:
            errors.append(f"Hash modificado: {path}")
    return errors


def validate_notebook(path: Path, require_outputs: bool) -> list[str]:
    errors = []
    try:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
    except Exception as exc:
        return [f"Notebook inválido {path}: {exc}"]

    source = "\n".join(cell.source for cell in notebook.cells)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]

    if re.search(r"(?i)(?:[A-Z]:\\\\Users\\\\|/home/[^/]+/|/Users/[^/]+/)", source):
        errors.append(f"Ruta absoluta Windows detectada: {path}")
    if re.search(r"(?i)(api[_-]?key|secret|password)\s*=\s*['\"][^'\"]+", source):
        errors.append(f"Posible secreto detectado: {path}")
    if re.search(r"(?m)^\s*[!%]pip\s+install", source):
        errors.append(f"Instalación automática detectada: {path}")

    required_terms = ["RUN_MODE", "validation", "test", "Marina del Sol"]
    for term in required_terms:
        if term not in source:
            errors.append(f"Falta contenido requerido '{term}': {path}")

    output_errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    if output_errors:
        errors.append(f"Hay {len(output_errors)} outputs con error: {path}")

    if require_outputs:
        missing_execution = [
            index for index, cell in enumerate(code_cells, start=1)
            if cell.get("execution_count") is None
        ]
        if missing_execution:
            errors.append(f"Celdas de código sin ejecutar {missing_execution}: {path}")

    if path.stat().st_size > 50 * 1024 * 1024:
        errors.append(f"Archivo mayor de 50 MB: {path}")
    return errors


def validate_nb01(mode: str) -> list[str]:
    errors = []
    notebook_dir = TEMA2 / "02-deep-learning-redes-neuronales" / "notebooks"
    result_dir = MODULE2_RESULTS / "01_neurona_manual_numpy_mnist"
    canonical = notebook_dir / "01_neurona_manual_numpy_mnist.ipynb"
    executed_name = "notebook_executed.fast.ipynb" if mode == "fast" else "notebook_executed.ipynb"
    executed = result_dir / executed_name
    summary_name = "run_summary.fast.json" if mode == "fast" else "run_summary.json"
    summary_path = result_dir / summary_name

    errors.extend(validate_notebook(canonical, require_outputs=False))
    if not executed.exists():
        errors.append(f"Falta notebook ejecutado: {executed}")
    else:
        errors.extend(validate_notebook(executed, require_outputs=True))

    if not summary_path.exists():
        errors.append(f"Falta resumen: {summary_path}")
        return errors

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("run_mode") != mode:
        errors.append("run_mode inconsistente en run_summary")
    if summary.get("publishable") is not (mode == "full"):
        errors.append("publishable inconsistente en run_summary")
    if summary.get("status") != "COMPLETED":
        errors.append("La ejecución no figura como COMPLETED")

    required_metric_keys = {
        "experiment_id", "task", "classes", "train_samples",
        "validation_samples", "test_samples", "parameters",
        "training_seconds", "best_validation_accuracy", "test_accuracy",
        "test_precision", "test_recall", "test_f1",
        "test_balanced_accuracy", "test_specificity",
    }
    required_artifacts = {
        "metrics.json", "training_history.csv", "learning_curves.png",
        "confusion_matrix.png", "classification_report.csv",
        "sample_predictions.png", "misclassified_examples.png", "weights.png",
    }
    run_dir = result_dir / "experiments" / mode
    for experiment_id in summary.get("experiments", []):
        experiment_dir = run_dir / experiment_id
        missing = [name for name in required_artifacts if not (experiment_dir / name).exists()]
        if missing:
            errors.append(f"Artefactos ausentes en {experiment_id}: {missing}")
            continue
        metrics = json.loads((experiment_dir / "metrics.json").read_text(encoding="utf-8"))
        absent_keys = required_metric_keys - metrics.keys()
        if absent_keys:
            errors.append(f"Métricas ausentes en {experiment_id}: {sorted(absent_keys)}")
        for key in ["test_accuracy", "test_precision", "test_recall", "test_f1"]:
            value = metrics.get(key)
            if value is None or not 0 <= value <= 1:
                errors.append(f"Métrica inválida {experiment_id}.{key}: {value}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook-id", choices=["02-01"], required=True)
    parser.add_argument("--mode", choices=["fast", "full"], required=True)
    args = parser.parse_args()

    errors = []
    errors.extend(validate_hash_manifest(
        TEMA2 / "input_notebooks" / "SHA256SUMS.txt",
        TEMA2 / "input_notebooks",
    ))
    errors.extend(validate_hash_manifest(
        TEMA2 / "MODULE1_SHA256SUMS.txt",
        TEMA2,
    ))
    errors.extend(validate_nb01(args.mode))

    if errors:
        print("VALIDATION_FAILED")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"VALIDATION_OK notebook={args.notebook_id} mode={args.mode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
