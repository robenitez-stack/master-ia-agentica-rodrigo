import json
from pathlib import Path


TEMA2 = Path(__file__).resolve().parents[1]
MODULE2 = TEMA2 / "02-deep-learning-redes-neuronales"

NOTEBOOKS = [
    {
        "id": "02-01",
        "title": "Neurona manual NumPy sobre MNIST",
        "source": "1_manual_neural_network_model.ipynb",
        "canonical": "01_neurona_manual_numpy_mnist.ipynb",
        "framework": "NumPy",
        "architecture": "Neurona logística manual: 784 pesos y un sesgo",
        "results": MODULE2 / "results" / "01_neurona_manual_numpy_mnist",
        "command": "python tema-02-machine-learning/scripts/execute_notebooks.py --target 02-01 --mode full",
    },
    {
        "id": "02-02",
        "title": "Neurona y MLP desde cero con PyTorch",
        "source": "1_red_neuronal_desde_cero.ipynb",
        "canonical": "02_red_neuronal_desde_cero_pytorch.ipynb",
        "framework": "PyTorch",
        "architecture": "Neurona explícita con autograd y MLP 784→128→1",
        "results": MODULE2 / "results" / "02_red_neuronal_desde_cero_pytorch",
        "command": "python tema-02-machine-learning/scripts/execute_notebooks.py --target 02-02 --mode full",
    },
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def percentage(value: float) -> str:
    return f"{100 * value:.2f} %"


def replace_section(path: Path, marker: str, content: str) -> None:
    start = f"<!-- {marker}:START -->"
    end = f"<!-- {marker}:END -->"
    text = path.read_text(encoding="utf-8")
    section = f"{start}\n{content.rstrip()}\n{end}"
    if start in text and end in text:
        prefix, rest = text.split(start, 1)
        _, suffix = rest.split(end, 1)
        text = prefix.rstrip() + "\n\n" + section + suffix
    else:
        text = text.rstrip() + "\n\n" + section + "\n"
    path.write_text(text, encoding="utf-8")


def result_readme(config: dict, summary: dict, metrics: list[dict]) -> str:
    rows = "\n".join(
        f"| `{item['experiment_id']}` | {item['model_type'] if 'model_type' in item else 'neurona NumPy'} "
        f"| {item['parameters']:,} | {item['best_epoch_or_iteration']} | "
        f"{percentage(item['best_validation_accuracy'])} | {percentage(item['test_accuracy'])} | "
        f"{percentage(item['test_precision'])} | {percentage(item['test_recall'])} | "
        f"{percentage(item['test_f1'])} | {item['training_seconds']:.2f} s |"
        for item in metrics
    )
    return f"""# {config['id']} — {config['title']}

## Estado

- Estado: `COMPLETED`
- Modo: `full`
- Publicable: `true`
- Semilla: `{summary['seed']}`
- Dispositivo: `{summary['device']}`
- Dataset: `{summary['dataset']}`
- Framework: `{config['framework']}`

## Origen y cambios

- Original protegido: `input_notebooks/{config['source']}`
- Notebook canónico: `notebooks/{config['canonical']}`
- Arquitectura: {config['architecture']}
- Se añadieron train, validation y test separados, selección por validation, evaluación final única de test, métricas, curvas y ejemplos de error.

## Ejecución

```powershell
{config['command']}
```

## Resultados regenerados

| Experimento | Modelo | Parámetros | Mejor época | Mejor val. accuracy | Test accuracy | Precisión | Recall | F1 | Tiempo |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{rows}

Los valores proceden directamente de `experiments/full/*/metrics.json`.

## Interpretación y limitaciones

Los pares `0 vs 1` son casi linealmente separables. `5 vs 6` comparte trazos y resulta más exigente. Los tiempos corresponden a CPU en este equipo y no deben generalizarse a otros entornos.

El ejercicio utiliza MNIST, un dataset pequeño y limpio. No evalúa deriva, sesgo operacional, seguridad, latencia productiva ni mantenimiento.

## Aplicación empresarial

El flujo es trasladable conceptualmente a baselines de clasificación de incidencias en un futuro MDS AI Operations Center. Antes de producción serían necesarios datos corporativos gobernados, validación temporal, explicabilidad, monitorización y revisión humana.
"""


def main() -> int:
    all_rows = []
    for config in NOTEBOOKS:
        summary = load_json(config["results"] / "run_summary.json")
        if summary["run_mode"] != "full" or not summary["publishable"]:
            raise RuntimeError(f"{config['id']} no tiene resultados full publicables")
        metrics = [
            load_json(config["results"] / "experiments" / "full" / experiment / "metrics.json")
            for experiment in summary["experiments"]
        ]
        (config["results"] / "README.md").write_text(
            result_readme(config, summary, metrics), encoding="utf-8"
        )
        for item in metrics:
            all_rows.append((config, item))

    table_rows = "\n".join(
        f"| {config['id']} | `{item['experiment_id']}` | "
        f"{item.get('model_type', 'neurona NumPy')} | {item['parameters']:,} | "
        f"{percentage(item['test_accuracy'])} | {percentage(item['test_f1'])} | "
        f"{item['training_seconds']:.2f} s |"
        for config, item in all_rows
    )
    table = f"""## Resultados full del módulo 2

| Notebook | Experimento | Modelo | Parámetros | Test accuracy | Test F1 | Tiempo CPU |
|---|---|---|---:|---:|---:|---:|
{table_rows}

Fuente: archivos `run_summary.json` y `experiments/full/*/metrics.json`. Los smoke tests `fast` no se publican como resultados académicos.
"""
    replace_section(MODULE2 / "README.md", "FULL_RESULTS", table)

    general = """## Estado del módulo 2

Los dos notebooks del módulo 2 están desarrollados, ejecutados de principio a fin en modo `full` y validados en CPU. La documentación detallada y los artefactos se encuentran en `02-deep-learning-redes-neuronales/results/`.

""" + table
    replace_section(TEMA2 / "README.md", "MODULE2_RESULTS", general)
    print("Documentación regenerada desde JSON.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
