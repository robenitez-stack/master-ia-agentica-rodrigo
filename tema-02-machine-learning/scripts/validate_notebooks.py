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


def validate_nb02(mode: str) -> list[str]:
    errors = []
    notebook_dir = TEMA2 / "02-deep-learning-redes-neuronales" / "notebooks"
    result_dir = MODULE2_RESULTS / "02_red_neuronal_desde_cero_pytorch"
    canonical = notebook_dir / "02_red_neuronal_desde_cero_pytorch.ipynb"
    executed_name = "notebook_executed.fast.ipynb" if mode == "fast" else "notebook_executed.ipynb"
    summary_name = "run_summary.fast.json" if mode == "fast" else "run_summary.json"
    executed = result_dir / executed_name
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
        errors.append("run_mode inconsistente en run_summary 02-02")
    if summary.get("publishable") is not (mode == "full"):
        errors.append("publishable inconsistente en run_summary 02-02")
    if summary.get("status") != "COMPLETED":
        errors.append("La ejecución 02-02 no figura como COMPLETED")
    if len(summary.get("experiments", [])) != 3:
        errors.append("02-02 debe registrar exactamente tres experimentos")

    required_keys = {
        "experiment_id", "model_type", "classes", "train_samples",
        "validation_samples", "test_samples", "parameters", "training_seconds",
        "best_validation_accuracy", "test_accuracy", "test_precision",
        "test_recall", "test_f1", "test_balanced_accuracy", "test_specificity",
    }
    required_artifacts = {
        "metrics.json", "training_history.csv", "learning_curves.png",
        "confusion_matrix.png", "classification_report.csv",
        "sample_predictions.png", "misclassified_examples.png",
        "validation_candidates.csv",
    }
    run_dir = result_dir / "experiments" / mode
    for experiment_id in summary.get("experiments", []):
        experiment_dir = run_dir / experiment_id
        missing = [name for name in required_artifacts if not (experiment_dir / name).exists()]
        if missing:
            errors.append(f"Artefactos ausentes en {experiment_id}: {missing}")
            continue
        metrics = json.loads((experiment_dir / "metrics.json").read_text(encoding="utf-8"))
        absent = required_keys - metrics.keys()
        if absent:
            errors.append(f"Métricas ausentes en {experiment_id}: {sorted(absent)}")
        for key in ["test_accuracy", "test_precision", "test_recall", "test_f1"]:
            value = metrics.get(key)
            if value is None or not 0 <= value <= 1:
                errors.append(f"Métrica inválida {experiment_id}.{key}: {value}")
    return errors


def validate_nb03_01(mode: str) -> list[str]:
    errors = []
    module = TEMA2 / "03-redes-multicapa-convolucionales-vision"
    result_dir = module / "results" / "01_mlp_mnist_pytorch"
    canonical = module / "notebooks" / "01_mlp_mnist_pytorch.ipynb"
    executed_name = "notebook_executed.fast.ipynb" if mode == "fast" else "notebook_executed.ipynb"
    summary_name = "run_summary.fast.json" if mode == "fast" else "run_summary.json"
    executed = result_dir / executed_name
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
    if summary.get("notebook_id") != "03-01":
        errors.append("notebook_id inconsistente en 03-01")
    if summary.get("run_mode") != mode:
        errors.append("run_mode inconsistente en 03-01")
    if summary.get("publishable") is not (mode == "full"):
        errors.append("publishable inconsistente en 03-01")
    if summary.get("experiments") != ["mnist_multiclass"]:
        errors.append("03-01 debe registrar mnist_multiclass")

    experiment_dir = result_dir / "experiments" / mode / "mnist_multiclass"
    required = {
        "metrics.json", "training_history.csv", "classification_report.csv",
        "learning_curves.png", "confusion_matrix.png",
        "sample_predictions.png", "misclassified_examples.png",
    }
    missing = [name for name in required if not (experiment_dir / name).exists()]
    if missing:
        errors.append(f"Artefactos ausentes en 03-01: {missing}")
        return errors

    metrics = json.loads((experiment_dir / "metrics.json").read_text(encoding="utf-8"))
    required_keys = {
        "experiment_id", "task", "classes", "train_samples",
        "validation_samples", "test_samples", "parameters",
        "training_seconds", "best_validation_accuracy", "test_accuracy",
        "test_precision", "test_recall", "test_f1",
        "test_macro_precision", "test_macro_recall", "test_macro_f1",
    }
    absent = required_keys - metrics.keys()
    if absent:
        errors.append(f"Métricas ausentes en 03-01: {sorted(absent)}")
    if len(metrics.get("classes", [])) != 10:
        errors.append("03-01 debe registrar diez clases")
    for key in ["test_accuracy", "test_precision", "test_recall", "test_f1"]:
        value = metrics.get(key)
        if value is None or not 0 <= value <= 1:
            errors.append(f"Métrica inválida 03-01.{key}: {value}")
    return errors


def validate_nb03_02(mode: str) -> list[str]:
    errors = []
    module = TEMA2 / "03-redes-multicapa-convolucionales-vision"
    result_dir = module / "results" / "02_cnn_mnist_keras"
    canonical = module / "notebooks" / "02_cnn_mnist_keras.ipynb"
    executed_name = "notebook_executed.fast.ipynb" if mode == "fast" else "notebook_executed.ipynb"
    summary_name = "run_summary.fast.json" if mode == "fast" else "run_summary.json"
    executed = result_dir / executed_name
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
    if summary.get("notebook_id") != "03-02":
        errors.append("notebook_id inconsistente en 03-02")
    if summary.get("run_mode") != mode:
        errors.append("run_mode inconsistente en 03-02")
    if summary.get("publishable") is not (mode == "full"):
        errors.append("publishable inconsistente en 03-02")
    if summary.get("experiments") != ["mnist_cnn"]:
        errors.append("03-02 debe registrar mnist_cnn")

    experiment_dir = result_dir / "experiments" / mode / "mnist_cnn"
    required = {
        "metrics.json", "training_history.csv", "classification_report.csv",
        "learning_curves.png", "confusion_matrix.png",
        "sample_predictions.png", "misclassified_examples.png",
    }
    missing = [name for name in required if not (experiment_dir / name).exists()]
    if missing:
        errors.append(f"Artefactos ausentes en 03-02: {missing}")
        return errors
    metrics = json.loads((experiment_dir / "metrics.json").read_text(encoding="utf-8"))
    if len(metrics.get("classes", [])) != 10:
        errors.append("03-02 debe registrar diez clases")
    for key in ["test_accuracy", "test_precision", "test_recall", "test_f1"]:
        value = metrics.get(key)
        if value is None or not 0 <= value <= 1:
            errors.append(f"Métrica inválida 03-02.{key}: {value}")
    return errors


def validate_nb03_03(mode: str) -> list[str]:
    module = TEMA2 / "03-redes-multicapa-convolucionales-vision"
    result_dir = module / "results" / "03_cnn_mnist_pytorch_base"
    canonical = module / "notebooks" / "03_cnn_mnist_pytorch_base.ipynb"
    executed = result_dir / ("notebook_executed.fast.ipynb" if mode == "fast" else "notebook_executed.ipynb")
    summary_path = result_dir / ("run_summary.fast.json" if mode == "fast" else "run_summary.json")
    errors = validate_notebook(canonical, False)
    if not executed.exists(): errors.append(f"Falta notebook ejecutado: {executed}")
    else: errors.extend(validate_notebook(executed, True))
    if not summary_path.exists(): return errors + [f"Falta resumen: {summary_path}"]
    summary=json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("notebook_id")!="03-03": errors.append("notebook_id inconsistente en 03-03")
    if summary.get("run_mode")!=mode: errors.append("run_mode inconsistente en 03-03")
    if summary.get("publishable") is not (mode=="full"): errors.append("publishable inconsistente en 03-03")
    exp=result_dir/"experiments"/mode/"mnist_cnn_base"
    required={"metrics.json","training_history.csv","classification_report.csv","learning_curves.png","confusion_matrix.png","sample_predictions.png","misclassified_examples.png"}
    missing=[x for x in required if not (exp/x).exists()]
    if missing: errors.append(f"Artefactos ausentes en 03-03: {missing}")
    else:
        metrics=json.loads((exp/"metrics.json").read_text(encoding="utf-8"))
        if len(metrics.get("classes",[]))!=10: errors.append("03-03 debe registrar diez clases")
        for key in ["test_accuracy","test_precision","test_recall","test_f1"]:
            value=metrics.get(key)
            if value is None or not 0<=value<=1: errors.append(f"Métrica inválida 03-03.{key}: {value}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--notebook-id",
        choices=["02-01", "02-02", "03-01", "03-02", "03-03"],
        required=True,
    )
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
    if args.notebook_id == "02-01":
        errors.extend(validate_nb01(args.mode))
    elif args.notebook_id == "02-02":
        errors.extend(validate_nb02(args.mode))
    elif args.notebook_id == "03-01":
        errors.extend(validate_nb03_01(args.mode))
    elif args.notebook_id == "03-02":
        errors.extend(validate_nb03_02(args.mode))
    else:
        errors.extend(validate_nb03_03(args.mode))

    if errors:
        print("VALIDATION_FAILED")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"VALIDATION_OK notebook={args.notebook_id} mode={args.mode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
