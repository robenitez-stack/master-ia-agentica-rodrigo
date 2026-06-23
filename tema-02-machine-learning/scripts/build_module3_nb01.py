from pathlib import Path

import nbformat as nbf


TEMA2 = Path(__file__).resolve().parents[1]
OUTPUT = (
    TEMA2
    / "03-redes-multicapa-convolucionales-vision"
    / "notebooks"
    / "01_mlp_mnist_pytorch.ipynb"
)


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md(r"""
# MLP multiclase sobre MNIST con PyTorch

## Objetivo

Construir una red neuronal multicapa para clasificar los diez dígitos de MNIST, separando correctamente **train, validation y test**.

Conceptos principales:

- `Flatten`: convierte 28×28 píxeles en 784 características.
- Capas densas: combinan globalmente todas las entradas.
- ReLU: introduce no linealidad.
- Logits: puntuaciones sin normalizar para las diez clases.
- Softmax: interpretación probabilística; no se aplica antes de `CrossEntropyLoss`.
- Validation: selecciona la mejor época.
- Test: evaluación final única.

> Dependencias: `requirements/common.txt` y `requirements/pytorch.txt`. El notebook no instala paquetes.
"""),
    md(r"""
## 1. Configuración reproducible

`RUN_MODE=fast` ejecuta un smoke test no publicable. `RUN_MODE=full` utiliza el conjunto completo y genera resultados definitivos si todas las validaciones pasan.
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
    ConfusionMatrixDisplay, accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
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
        if candidato.name == "tema-02-machine-learning" and (candidato / ".tema2-root").exists():
            return candidato
    raise FileNotFoundError("No se pudo localizar tema-02-machine-learning")


fijar_semillas()
DEVICE = elegir_dispositivo()
TEMA2_ROOT = localizar_tema2()
DATA_DIR = TEMA2_ROOT / ".data" / "mnist"
RESULTS_ROOT = (
    TEMA2_ROOT / "03-redes-multicapa-convolucionales-vision"
    / "results" / "01_mlp_mnist_pytorch"
)
RUN_DIR = RESULTS_ROOT / "experiments" / RUN_MODE / "mnist_multiclass"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RUN_DIR.mkdir(parents=True, exist_ok=True)
PUBLISHABLE = RUN_MODE == "full"

print(f"RUN_MODE={RUN_MODE} | publishable={str(PUBLISHABLE).lower()}")
print(f"PyTorch={torch.__version__} | dispositivo={DEVICE} | semilla={SEED}")
"""),
    md(r"""
## 2. Dataset y separación

El train oficial de MNIST se divide de forma estratificada en train y validation. El test oficial permanece aislado hasta cerrar la mejor época.
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

X_train, X_val, y_train, y_val = train_test_split(
    X_train_raw, y_train_raw, test_size=0.15,
    stratify=y_train_raw, random_state=SEED,
)


def limitar(X, y, n, seed):
    if n is None or len(y) <= n:
        return X, y
    X_keep, _, y_keep, _ = train_test_split(
        X, y, train_size=n, stratify=y, random_state=seed
    )
    return X_keep, y_keep


if RUN_MODE == "fast":
    X_train, y_train = limitar(X_train, y_train, 4000, SEED)
    X_val, y_val = limitar(X_val, y_val, 1000, SEED + 1)
    X_test, y_test = limitar(X_test_raw, y_test_raw, 1200, SEED + 2)
else:
    X_test, y_test = X_test_raw, y_test_raw


def tensor_dataset(X, y):
    images = torch.from_numpy(X.astype(np.float32) / 255.0).unsqueeze(1)
    labels = torch.from_numpy(y.astype(np.int64))
    return TensorDataset(images, labels)


train_ds = tensor_dataset(X_train, y_train)
val_ds = tensor_dataset(X_val, y_val)
test_ds = tensor_dataset(X_test, y_test)


def loader(dataset, shuffle, seed):
    return DataLoader(
        dataset, batch_size=128, shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0, pin_memory=DEVICE.type == "cuda",
    )


print(f"Train={len(train_ds):,} | Validation={len(val_ds):,} | Test={len(test_ds):,}")
print("Distribución train:", np.bincount(y_train))
"""),
    code(r"""
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for digit, ax in enumerate(axes.ravel()):
    index = np.flatnonzero(y_train == digit)[0]
    ax.imshow(X_train[index], cmap="gray")
    ax.set_title(f"Clase {digit}")
    ax.axis("off")
plt.suptitle("Ejemplos de MNIST")
plt.tight_layout()
plt.show()
"""),
    md(r"""
## 3. Arquitectura

El modelo aplana la imagen, utiliza dos capas ocultas ReLU y genera diez logits. `CrossEntropyLoss` aplica internamente `log_softmax`, por lo que añadir softmax al forward sería incorrecto y numéricamente menos estable.
"""),
    code(r"""
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 10),
        )

    def forward(self, X):
        return self.network(X)


model = MLP().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
parameter_count = sum(parameter.numel() for parameter in model.parameters())
print(model)
print(f"Parámetros entrenables: {parameter_count:,}")
"""),
    md(r"""
## 4. Entrenamiento y checkpoint por validation

Validation determina la mejor época. El checkpoint se conserva en memoria para evitar versionar pesos grandes. Test no se consulta dentro del bucle.
"""),
    code(r"""
def evaluar(model, data_loader):
    model.eval()
    total_loss, true, pred = 0.0, [], []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            total_loss += criterion(logits, labels).item() * len(labels)
            true.extend(labels.cpu().numpy())
            pred.extend(logits.argmax(dim=1).cpu().numpy())
    true, pred = np.asarray(true), np.asarray(pred)
    return {
        "loss": total_loss / len(true),
        "accuracy": accuracy_score(true, pred),
        "macro_f1": f1_score(true, pred, average="macro"),
        "true": true,
        "pred": pred,
    }


epochs = 2 if RUN_MODE == "fast" else 15
train_loader = loader(train_ds, True, SEED)
val_loader = loader(val_ds, False, SEED)
history, best_loss, best_epoch, best_state = [], np.inf, 0, None
start = time.perf_counter()

for epoch in range(1, epochs + 1):
    model.train()
    train_loss, train_true, train_pred = 0.0, [], []
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(labels)
        train_true.extend(labels.cpu().numpy())
        train_pred.extend(logits.argmax(dim=1).detach().cpu().numpy())

    validation = evaluar(model, val_loader)
    row = {
        "epoch": epoch,
        "train_loss": train_loss / len(train_ds),
        "validation_loss": validation["loss"],
        "train_accuracy": accuracy_score(train_true, train_pred),
        "validation_accuracy": validation["accuracy"],
        "validation_macro_f1": validation["macro_f1"],
    }
    history.append(row)
    if row["validation_loss"] < best_loss:
        best_loss, best_epoch = row["validation_loss"], epoch
        best_state = copy.deepcopy(model.state_dict())
    print(
        f"Época {epoch:02d}/{epochs} | train loss={row['train_loss']:.4f} "
        f"| val loss={row['validation_loss']:.4f} "
        f"| val accuracy={row['validation_accuracy']:.4f}"
    )

training_seconds = time.perf_counter() - start
model.load_state_dict(best_state)
history = pd.DataFrame(history)
print(f"Mejor época: {best_epoch} | tiempo: {training_seconds:.2f} s")
"""),
    md(r"""
## 5. Evaluación final de test

Tras restaurar la mejor época se consulta test una sola vez. Se generan métricas macro y por clase, matriz de confusión, curvas y ejemplos.
"""),
    code(r"""
test = evaluar(model, loader(test_ds, False, SEED))
y_true, y_pred = test["true"], test["pred"]
report = pd.DataFrame(classification_report(
    y_true, y_pred, output_dict=True, zero_division=0
)).T

metrics = {
    "experiment_id": "mnist_multiclass",
    "task": "multiclass_classification",
    "classes": [str(i) for i in range(10)],
    "train_samples": len(train_ds),
    "validation_samples": len(val_ds),
    "test_samples": len(test_ds),
    "epochs_or_iterations_completed": epochs,
    "best_epoch_or_iteration": best_epoch,
    "parameters": parameter_count,
    "training_seconds": training_seconds,
    "best_validation_accuracy": float(history.loc[best_epoch - 1, "validation_accuracy"]),
    "test_accuracy": accuracy_score(y_true, y_pred),
    "test_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
    "test_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    "test_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    "test_macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
    "test_macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    "test_macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    "test_balanced_accuracy": recall_score(y_true, y_pred, average="macro"),
    "test_specificity": None,
}

(RUN_DIR / "metrics.json").write_text(
    json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
)
history.to_csv(RUN_DIR / "training_history.csv", index=False)
report.to_csv(RUN_DIR / "classification_report.csv")

display(pd.Series(metrics, name="valor").to_frame())
display(report)
"""),
    code(r"""
def guardar(fig, name):
    fig.savefig(RUN_DIR / name, dpi=140, bbox_inches="tight")
    plt.close(fig)


fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.epoch, history.train_loss, label="train")
axes[0].plot(history.epoch, history.validation_loss, label="validation")
axes[0].axvline(best_epoch, ls="--", color="gray", label="mejor época")
axes[0].set(title="Pérdida", xlabel="Época", ylabel="Loss")
axes[0].legend()
axes[1].plot(history.epoch, history.train_accuracy, label="train")
axes[1].plot(history.epoch, history.validation_accuracy, label="validation")
axes[1].set(title="Accuracy", xlabel="Época", ylabel="Accuracy")
axes[1].legend()
fig.tight_layout()
guardar(fig, "learning_curves.png")

cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
fig, ax = plt.subplots(figsize=(8, 7))
ConfusionMatrixDisplay(cm, display_labels=list(range(10))).plot(
    cmap="Blues", ax=ax, values_format="d"
)
ax.set_title("Matriz de confusión — MLP MNIST")
guardar(fig, "confusion_matrix.png")


def ejemplos(indices, title, filename):
    fig, axes = plt.subplots(2, 5, figsize=(11, 5))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, idx in zip(axes.ravel(), indices[:10]):
        ax.imshow(X_test[idx], cmap="gray")
        ax.set_title(f"Real={y_true[idx]} | Pred={y_pred[idx]}")
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    guardar(fig, filename)


ejemplos(np.flatnonzero(y_true == y_pred), "Predicciones correctas", "sample_predictions.png")
ejemplos(np.flatnonzero(y_true != y_pred), "Ejemplos mal clasificados", "misclassified_examples.png")
"""),
    md(r"""
## 6. Interpretación, generalización y comparación con CNN

Un MLP pierde la estructura espacial al aplanar la imagen: un píxel vecino no recibe tratamiento especial. Una CNN reutiliza filtros locales, necesita menos conexiones para explotar bordes y patrones y suele generalizar mejor en visión.

- **Underfitting**: train y validation permanecen bajos.
- **Overfitting**: train mejora mientras validation se degrada.
- Las clases más confundidas deben interpretarse mediante la matriz y los ejemplos mal clasificados.
"""),
    code(r"""
try:
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=TEMA2_ROOT.parent,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
except Exception:
    git_commit = "UNAVAILABLE"

summary = {
    "notebook_id": "03-01",
    "source_notebook": "1_red_neuronal_multicapa.ipynb",
    "canonical_notebook": "01_mlp_mnist_pytorch.ipynb",
    "module": "03-redes-multicapa-convolucionales-vision",
    "status": "COMPLETED",
    "run_mode": RUN_MODE,
    "publishable": PUBLISHABLE,
    "seed": SEED,
    "device": str(DEVICE),
    "frameworks": ["pytorch"],
    "dataset": "MNIST",
    "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "git_commit": git_commit,
    "experiments": ["mnist_multiclass"],
}
summary_name = "run_summary.json" if RUN_MODE == "full" else "run_summary.fast.json"
(RESULTS_ROOT / summary_name).write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
)
environment_name = "environment.txt" if RUN_MODE == "full" else "environment.fast.txt"
(RESULTS_ROOT / environment_name).write_text("\n".join([
    f"platform={platform.platform()}", f"python={sys.version}",
    f"torch={torch.__version__}", f"numpy={np.__version__}",
    f"scikit-learn={sklearn.__version__}", f"device={DEVICE}",
]), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
"""),
    md(r"""
## 7. Conclusiones

### Técnica

El MLP aprende una frontera no lineal multiclase con logits y entropía cruzada. Validation selecciona el checkpoint y test permanece no contaminado. Su limitación principal es ignorar explícitamente la geometría local de la imagen.

### Ejecutiva

El ejercicio muestra cómo una red densa supera baselines lineales, pero también por qué la arquitectura debe reflejar la estructura del dato. Conceptualmente, Marina del Sol podría emplear MLP sobre variables tabulares de incidencias; para imágenes o mapas operativos serían preferibles arquitecturas espaciales. Este notebook académico no está listo para producción.
"""),
]

metadata = {
    "kernelspec": {
        "display_name": "Python - Master IA Agentica",
        "language": "python",
        "name": "master-ia-agentica",
    },
    "language_info": {"name": "python", "version": "3.10"},
}
notebook = nbf.v4.new_notebook(cells=cells, metadata=metadata)
nbf.validate(notebook)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(f"Notebook generado: {OUTPUT}")
