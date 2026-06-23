from pathlib import Path

import nbformat as nbf


TEMA2 = Path(__file__).resolve().parents[1]
OUTPUT = TEMA2 / "02-deep-learning-redes-neuronales" / "notebooks" / "02_red_neuronal_desde_cero_pytorch.ipynb"


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md(r"""
# Red neuronal desde cero con PyTorch

## Objetivo

Hacer explícitas las cuatro fases del aprendizaje —**forward, loss, backward y update**— usando tensores PyTorch, y comparar una neurona binaria con un MLP sencillo.

Experimentos:

1. Neurona `0 vs 1`.
2. Neurona `5 vs 6`.
3. MLP `5 vs 6`.

Los parámetros y el bucle de la neurona son explícitos, pero los gradientes se calculan con **autograd**. El MLP utiliza componentes de alto nivel de PyTorch. Test se reserva para una evaluación final única.

> Instalar desde `requirements/common.txt` y `requirements/pytorch.txt`; el notebook no instala paquetes.
"""),
    md(r"""
## 1. Configuración, reproducibilidad y dispositivo

`RUN_MODE=fast` ejecuta un smoke test no publicable. `RUN_MODE=full` utiliza todos los datos y produce resultados publicables únicamente si finaliza y supera las validaciones.
"""),
    code(r"""
import copy
import gzip
import json
import os
import platform
import random
import struct
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

RUN_MODE = os.getenv("RUN_MODE", "full").lower()
if RUN_MODE not in {"fast", "full"}:
    raise ValueError("RUN_MODE debe ser 'fast' o 'full'")

SEED = 42


def fijar_semillas(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def elegir_dispositivo():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def localizar_tema2():
    cwd = Path.cwd().resolve()
    for candidato in [cwd, cwd / "tema-02-machine-learning", *cwd.parents]:
        if candidato.name == "tema-02-machine-learning" and (candidato / "AGENTS.md").exists():
            return candidato
    raise FileNotFoundError("No se pudo localizar tema-02-machine-learning")


fijar_semillas()
DEVICE = elegir_dispositivo()
TEMA2_ROOT = localizar_tema2()
DATA_DIR = TEMA2_ROOT / ".data" / "mnist"
RESULTS_ROOT = TEMA2_ROOT / "02-deep-learning-redes-neuronales" / "results" / "02_red_neuronal_desde_cero_pytorch"
RUN_DIR = RESULTS_ROOT / "experiments" / RUN_MODE
DATA_DIR.mkdir(parents=True, exist_ok=True)
RUN_DIR.mkdir(parents=True, exist_ok=True)
PUBLISHABLE = RUN_MODE == "full"

print(f"RUN_MODE={RUN_MODE} | publishable={str(PUBLISHABLE).lower()}")
print(f"PyTorch={torch.__version__} | dispositivo={DEVICE} | semilla={SEED}")
"""),
    md(r"""
## 2. Dataset y particiones

Se reutilizan los mismos archivos MNIST de `.data/mnist` que el notebook NumPy. Las imágenes permanecen en CPU y solo cada minibatch se transfiere al dispositivo, reduciendo el consumo de memoria GPU.
"""),
    code(r"""
URLS = {
    "train-images-idx3-ubyte.gz": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


def descargar(nombre, url):
    destino = DATA_DIR / nombre
    if not destino.exists():
        temporal = destino.with_suffix(destino.suffix + ".tmp")
        urllib.request.urlretrieve(url, temporal)
        temporal.replace(destino)
    return destino


def leer_imagenes(path):
    with gzip.open(path, "rb") as stream:
        magic, n, rows, cols = struct.unpack(">IIII", stream.read(16))
        if magic != 2051:
            raise ValueError("Formato IDX de imágenes inválido")
        return np.frombuffer(stream.read(), dtype=np.uint8).reshape(n, rows, cols)


def leer_etiquetas(path):
    with gzip.open(path, "rb") as stream:
        magic, n = struct.unpack(">II", stream.read(8))
        if magic != 2049:
            raise ValueError("Formato IDX de etiquetas inválido")
        values = np.frombuffer(stream.read(), dtype=np.uint8)
    if len(values) != n:
        raise ValueError("Cantidad de etiquetas inconsistente")
    return values


paths = {name: descargar(name, url) for name, url in URLS.items()}
X_train_raw = leer_imagenes(paths["train-images-idx3-ubyte.gz"])
y_train_raw = leer_etiquetas(paths["train-labels-idx1-ubyte.gz"])
X_test_raw = leer_imagenes(paths["t10k-images-idx3-ubyte.gz"])
y_test_raw = leer_etiquetas(paths["t10k-labels-idx1-ubyte.gz"])
print("Train original:", X_train_raw.shape, "| Test original:", X_test_raw.shape)
"""),
    code(r"""
def limitar(X, y, n, seed):
    if n is None or len(y) <= n:
        return X, y
    X_keep, _, y_keep, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=seed)
    return X_keep, y_keep


def preparar_par(negativa, positiva):
    train_mask = np.isin(y_train_raw, [negativa, positiva])
    test_mask = np.isin(y_test_raw, [negativa, positiva])
    X_base = X_train_raw[train_mask]
    y_base = (y_train_raw[train_mask] == positiva).astype(np.float32)
    X_test = X_test_raw[test_mask]
    y_test = (y_test_raw[test_mask] == positiva).astype(np.float32)
    X_train, X_val, y_train, y_val = train_test_split(
        X_base, y_base, test_size=0.20, stratify=y_base, random_state=SEED
    )
    if RUN_MODE == "fast":
        X_train, y_train = limitar(X_train, y_train, 1600, SEED)
        X_val, y_val = limitar(X_val, y_val, 400, SEED + 1)
        X_test, y_test = limitar(X_test, y_test, 500, SEED + 2)

    to_tensor = lambda X: torch.from_numpy(X.reshape(len(X), -1).astype(np.float32) / 255.0)
    return {
        "train": TensorDataset(to_tensor(X_train), torch.from_numpy(y_train)),
        "val": TensorDataset(to_tensor(X_val), torch.from_numpy(y_val)),
        "test": TensorDataset(to_tensor(X_test), torch.from_numpy(y_test)),
        "test_images": X_test,
        "test_labels": y_test.astype(np.int64),
    }


def crear_loader(dataset, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=128, shuffle=shuffle, generator=generator,
        num_workers=0, pin_memory=DEVICE.type == "cuda",
    )


for pair in [(0, 1), (5, 6)]:
    data = preparar_par(*pair)
    print(f"{pair}: train={len(data['train'])}, validation={len(data['val'])}, test={len(data['test'])}")
"""),
    md(r"""
## 3. Modelos

La neurona declara `w` y `b` con `requires_grad=True`; el forward y la actualización son explícitos, mientras `loss.backward()` delega la diferenciación a autograd. `BCEWithLogitsLoss` combina sigmoide y entropía cruzada de manera estable.

El MLP usa `nn.Sequential`, una capa oculta ReLU y Dropout. Su optimización utiliza Adam, mostrando el salto hacia APIs de alto nivel.
"""),
    code(r"""
class NeuronaAutograd:
    def __init__(self, n_features):
        self.w = torch.zeros(n_features, device=DEVICE, requires_grad=True)
        self.b = torch.zeros(1, device=DEVICE, requires_grad=True)

    def __call__(self, X):
        return X @ self.w + self.b

    def state(self):
        return self.w.detach().clone(), self.b.detach().clone()

    def load_state(self, state):
        with torch.no_grad():
            self.w.copy_(state[0]); self.b.copy_(state[1])

    def parameter_count(self):
        return self.w.numel() + self.b.numel()


class MLPBinario(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 128), nn.ReLU(), nn.Dropout(0.20), nn.Linear(128, 1)
        )

    def forward(self, X):
        return self.network(X).squeeze(1)


CRITERION = nn.BCEWithLogitsLoss()
"""),
    md(r"""
## 4. Entrenamiento, validation y checkpoint en memoria

Cada learning rate parte de la misma semilla. El mejor estado se conserva según `validation_loss`. Ningún resultado de test interviene en la selección.
"""),
    code(r"""
def forward(model, model_type, X):
    return model(X) if model_type == "neuron" else model(X)


def evaluate(model, model_type, loader):
    if model_type == "mlp":
        model.eval()
    total_loss, labels, predictions = 0.0, [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            logits = forward(model, model_type, Xb)
            total_loss += CRITERION(logits, yb).item() * len(yb)
            predictions.extend((torch.sigmoid(logits) >= 0.5).long().cpu().numpy())
            labels.extend(yb.long().cpu().numpy())
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    return {
        "loss": total_loss / len(labels),
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, zero_division=0),
        "labels": labels,
        "predictions": predictions,
    }


def train_candidate(data, model_type, learning_rate, epochs):
    fijar_semillas()
    model = NeuronaAutograd(784) if model_type == "neuron" else MLPBinario(784).to(DEVICE)
    optimizer = None if model_type == "neuron" else torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = crear_loader(data["train"], True, SEED)
    val_loader = crear_loader(data["val"], False, SEED)
    history, best_loss, best_epoch, best_state = [], np.inf, 0, None

    for epoch in range(1, epochs + 1):
        if model_type == "mlp":
            model.train()
        running_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            logits = forward(model, model_type, Xb)
            loss = CRITERION(logits, yb)
            loss.backward()
            if model_type == "neuron":
                with torch.no_grad():
                    model.w -= learning_rate * model.w.grad
                    model.b -= learning_rate * model.b.grad
                    model.w.grad.zero_(); model.b.grad.zero_()
            else:
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            running_loss += loss.item() * len(yb)

        val = evaluate(model, model_type, val_loader)
        history.append({
            "epoch": epoch,
            "train_loss": running_loss / len(data["train"]),
            "validation_loss": val["loss"],
            "validation_accuracy": val["accuracy"],
            "validation_f1": val["f1"],
        })
        if val["loss"] < best_loss:
            best_loss, best_epoch = val["loss"], epoch
            best_state = model.state() if model_type == "neuron" else copy.deepcopy(model.state_dict())

    if model_type == "neuron":
        model.load_state(best_state)
    else:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history), best_epoch
"""),
    code(r"""
def save_figure(fig, path):
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


def sample_figure(images, true_binary, pred_binary, indices, negative, positive, title, path):
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for ax in axes.ravel(): ax.axis("off")
    for ax, idx in zip(axes.ravel(), indices[:8]):
        actual = positive if true_binary[idx] else negative
        predicted = positive if pred_binary[idx] else negative
        ax.imshow(images[idx], cmap="gray"); ax.set_title(f"Real={actual} | Pred={predicted}"); ax.axis("off")
    fig.suptitle(title); fig.tight_layout(); save_figure(fig, path)


def run_experiment(experiment_id, negative, positive, model_type, learning_rates):
    data = preparar_par(negative, positive)
    epochs = 2 if RUN_MODE == "fast" else (15 if model_type == "neuron" else 12)
    candidates, start_total = [], time.perf_counter()
    for lr in learning_rates:
        start = time.perf_counter()
        model, history, best_epoch = train_candidate(data, model_type, lr, epochs)
        best = history.loc[best_epoch - 1]
        candidates.append({
            "learning_rate": lr, "model": model, "history": history,
            "best_epoch": best_epoch, "validation_loss": float(best["validation_loss"]),
            "validation_accuracy": float(best["validation_accuracy"]),
            "validation_f1": float(best["validation_f1"]),
            "training_seconds": time.perf_counter() - start,
        })
    selected = min(candidates, key=lambda item: item["validation_loss"])
    model, history = selected["model"], selected["history"]

    # Test se consulta una única vez, después de cerrar la selección.
    test = evaluate(model, model_type, crear_loader(data["test"], False, SEED))
    y_true, y_pred = test["labels"], test["predictions"]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]); tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if tn + fp else None
    parameters = model.parameter_count() if model_type == "neuron" else sum(p.numel() for p in model.parameters())
    metrics = {
        "experiment_id": experiment_id, "task": "binary_classification",
        "model_type": model_type, "classes": [str(negative), str(positive)], "positive_class": str(positive),
        "train_samples": len(data["train"]), "validation_samples": len(data["val"]), "test_samples": len(data["test"]),
        "epochs_or_iterations_completed": epochs, "best_epoch_or_iteration": selected["best_epoch"],
        "parameters": parameters, "training_seconds": time.perf_counter() - start_total,
        "selected_learning_rate": selected["learning_rate"], "best_validation_accuracy": selected["validation_accuracy"],
        "test_accuracy": accuracy_score(y_true, y_pred),
        "test_precision": precision_score(y_true, y_pred, zero_division=0),
        "test_recall": recall_score(y_true, y_pred, zero_division=0),
        "test_f1": f1_score(y_true, y_pred, zero_division=0),
        "test_macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "test_macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "test_macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "test_balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "test_specificity": specificity,
        "true_negatives": int(tn), "false_positives": int(fp), "false_negatives": int(fn), "true_positives": int(tp),
    }
    out = RUN_DIR / experiment_id; out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    history.to_csv(out / "training_history.csv", index=False)
    report = pd.DataFrame(classification_report(
        y_true, y_pred, target_names=[str(negative), str(positive)], output_dict=True, zero_division=0
    )).T
    report.to_csv(out / "classification_report.csv")
    comparison = pd.DataFrame([{k: v for k, v in c.items() if k not in {"model", "history"}} for c in candidates])
    comparison.to_csv(out / "validation_candidates.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(history.epoch, history.train_loss, label="train"); ax.plot(history.epoch, history.validation_loss, label="validation"); ax.axvline(selected["best_epoch"], ls="--", c="gray", label="mejor época"); ax.set(xlabel="Época", ylabel="Loss", title=f"Curvas — {experiment_id}"); ax.legend(); save_figure(fig, out / "learning_curves.png")
    fig, ax = plt.subplots(figsize=(5, 4)); ConfusionMatrixDisplay(cm, display_labels=[str(negative), str(positive)]).plot(cmap="Blues", ax=ax); ax.set_title("Matriz de confusión"); save_figure(fig, out / "confusion_matrix.png")
    correct = np.flatnonzero(y_true == y_pred); wrong = np.flatnonzero(y_true != y_pred)
    sample_figure(data["test_images"], y_true, y_pred, correct, negative, positive, "Predicciones correctas", out / "sample_predictions.png")
    sample_figure(data["test_images"], y_true, y_pred, wrong, negative, positive, "Errores", out / "misclassified_examples.png")
    return metrics, comparison, report


EXPERIMENTS = [
    ("neuron_mnist_0_vs_1", 0, 1, "neuron", [0.05, 0.10]),
    ("neuron_mnist_5_vs_6", 5, 6, "neuron", [0.05, 0.10]),
    ("mlp_mnist_5_vs_6", 5, 6, "mlp", [0.001, 0.003]),
]
results = []
for config in EXPERIMENTS:
    metrics, comparison, report = run_experiment(*config)
    results.append(metrics)
    print(f"\n{metrics['experiment_id']}")
    display(comparison[["learning_rate", "validation_loss", "validation_accuracy", "validation_f1"]])
    display(pd.Series(metrics, name="valor").to_frame()); display(report)
"""),
    md(r"""
## 5. Interpretación y comparación

`0 vs 1` suele admitir una separación casi lineal. `5 vs 6` comparte trazos y exige una frontera más flexible; el MLP introduce no linealidad mediante ReLU. Una mejora no debe atribuirse solo a la arquitectura: también influyen inicialización, optimizador, regularización y presupuesto de entrenamiento.

Overfitting aparece si train mejora mientras validation empeora; underfitting, si ambas métricas permanecen pobres. La mejor época se decide con validation, nunca con test.
"""),
    code(r"""
try:
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=TEMA2_ROOT.parent, capture_output=True, text=True, check=True).stdout.strip()
except Exception:
    git_commit = "UNAVAILABLE"

summary = {
    "notebook_id": "02-02", "source_notebook": "1_red_neuronal_desde_cero.ipynb",
    "canonical_notebook": "02_red_neuronal_desde_cero_pytorch.ipynb",
    "module": "02-deep-learning-redes-neuronales", "status": "COMPLETED",
    "run_mode": RUN_MODE, "publishable": PUBLISHABLE, "seed": SEED,
    "device": str(DEVICE), "frameworks": ["pytorch"], "dataset": "MNIST",
    "run_timestamp_utc": datetime.now(timezone.utc).isoformat(), "git_commit": git_commit,
    "experiments": [item["experiment_id"] for item in results],
}
summary_name = "run_summary.json" if RUN_MODE == "full" else "run_summary.fast.json"
(RESULTS_ROOT / summary_name).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
environment_name = "environment.txt" if RUN_MODE == "full" else "environment.fast.txt"
(RESULTS_ROOT / environment_name).write_text("\n".join([
    f"platform={platform.platform()}", f"python={sys.version}", f"torch={torch.__version__}",
    f"numpy={np.__version__}", f"scikit-learn={sklearn.__version__}", f"device={DEVICE}",
    f"cuda_available={torch.cuda.is_available()}",
]), encoding="utf-8")
display(pd.DataFrame(results).set_index("experiment_id")[["model_type", "parameters", "test_accuracy", "test_f1", "training_seconds"]])
print(json.dumps(summary, indent=2, ensure_ascii=False))
"""),
    md(r"""
## 6. Conclusiones

### Técnica

Autograd calcula gradientes, pero la neurona mantiene visibles forward, loss, backward y update. El MLP usa abstracciones de alto nivel y puede representar fronteras no lineales. Los datos permanecen en CPU y solo cada minibatch viaja al dispositivo.

### Ejecutiva

La comparación ilustra cuándo un baseline lineal deja de ser suficiente. Conceptualmente, Marina del Sol podría contrastar modelos lineales y MLP para clasificar incidencias operativas, siempre con datos reales, trazabilidad, seguridad, monitorización y revisión humana. Este ejercicio académico no está listo para producción.
"""),
]

metadata = {"kernelspec": {"display_name": "Python - Master IA Agentica", "language": "python", "name": "master-ia-agentica"}, "language_info": {"name": "python", "version": "3.10"}}
notebook = nbf.v4.new_notebook(cells=cells, metadata=metadata)
nbf.validate(notebook)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(f"Notebook generado: {OUTPUT}")
