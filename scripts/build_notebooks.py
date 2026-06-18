from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tema-02-machine-learning"

KERNEL = {
    "kernelspec": {
        "display_name": "Python - Master IA Agentica",
        "language": "python",
        "name": "master-ia-agentica",
    },
    "language_info": {"name": "python", "version": "3.10"},
}


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


supervised_cells = [
    md("""
# Clasificación supervisada: ingresos con XGBoost

Este notebook desarrolla un flujo reproducible de clasificación binaria sobre el dataset **Adult** de UCI. El objetivo es predecir si una persona supera los 50 000 USD anuales.

La solución separa entrenamiento y prueba antes del preprocesamiento, evita fuga de información y evalúa el modelo con métricas apropiadas para clases desbalanceadas.
"""),
    md("""
## 1. Preparación

Las dependencias se administran desde `requirements.txt`; no se instalan paquetes dentro del notebook.
"""),
    code("""
from pathlib import Path
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=FutureWarning)
SEED = 42
rng = np.random.default_rng(SEED)
pd.set_option("display.max_columns", 100)
"""),
    md("## 2. Carga y comprensión de los datos"),
    code("""
adult = fetch_ucirepo(id=2)
X_raw = adult.data.features.copy()
y_raw = adult.data.targets["income"].copy()

# UCI combina etiquetas con y sin punto final; se normalizan a una sola forma.
y = y_raw.astype(str).str.strip().str.rstrip(".").map({"<=50K": 0, ">50K": 1})

# Los signos ? representan valores ausentes.
X_raw = X_raw.replace("?", np.nan)

print(f"Filas: {X_raw.shape[0]:,} | Variables: {X_raw.shape[1]}")
print(f"Valores objetivo ausentes: {y.isna().sum()}")
display(X_raw.head())
"""),
    code("""
class_distribution = (
    y.value_counts()
    .rename(index={0: "<=50K", 1: ">50K"})
    .to_frame("cantidad")
)
class_distribution["porcentaje"] = 100 * class_distribution["cantidad"] / len(y)
display(class_distribution.round(2))

ax = class_distribution["cantidad"].plot(kind="bar", color=["#4C78A8", "#F58518"], rot=0)
ax.set_title("Distribución de la variable objetivo")
ax.set_ylabel("Personas")
plt.tight_layout()
plt.show()
"""),
    code("""
missing = X_raw.isna().sum().sort_values(ascending=False)
missing = missing[missing > 0].to_frame("valores_ausentes")
missing["porcentaje"] = 100 * missing["valores_ausentes"] / len(X_raw)
display(missing.round(2))

numeric_preview = X_raw.select_dtypes(include=np.number).describe().T
display(numeric_preview)
"""),
    md("""
### Lectura inicial

La clase de ingresos altos es minoritaria, por lo que *accuracy* no basta. Se reportan también precisión, *recall*, F1 y ROC-AUC. Los valores ausentes se imputarán dentro del pipeline usando únicamente el conjunto de entrenamiento.
"""),
    md("## 3. Separación y preprocesamiento sin fuga de información"),
    code("""
valid_rows = y.notna()
X = X_raw.loc[valid_rows].reset_index(drop=True)
y = y.loc[valid_rows].astype(int).reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=SEED,
)

numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_features = X_train.select_dtypes(exclude=np.number).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

print(f"Entrenamiento: {len(X_train):,} filas")
print(f"Prueba: {len(X_test):,} filas")
print(f"Numéricas: {len(numeric_features)} | Categóricas: {len(categorical_features)}")
"""),
    md("## 4. Entrenamiento y búsqueda de hiperparámetros"),
    code("""
model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    random_state=SEED,
    n_jobs=4,
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model),
])

parameter_space = {
    "model__n_estimators": [200, 350, 500],
    "model__max_depth": [3, 5, 7],
    "model__learning_rate": [0.03, 0.05, 0.10],
    "model__min_child_weight": [1, 5, 10],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.8, 1.0],
}

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=parameter_space,
    n_iter=8,
    scoring="roc_auc",
    cv=3,
    random_state=SEED,
    n_jobs=1,
    verbose=1,
    refit=True,
)

search.fit(X_train, y_train)
best_model = search.best_estimator_

print(f"Mejor ROC-AUC de validación cruzada: {search.best_score_:.4f}")
print("Mejores hiperparámetros:")
display(pd.Series(search.best_params_, name="valor").to_frame())
"""),
    md("## 5. Evaluación sobre datos no vistos"),
    code("""
y_probability = best_model.predict_proba(X_test)[:, 1]
y_prediction = (y_probability >= 0.50).astype(int)

metrics = pd.Series({
    "accuracy": accuracy_score(y_test, y_prediction),
    "precision": precision_score(y_test, y_prediction),
    "recall": recall_score(y_test, y_prediction),
    "f1": f1_score(y_test, y_prediction),
    "roc_auc": roc_auc_score(y_test, y_probability),
}, name="resultado").to_frame()

display(metrics.style.format("{:.4f}"))
print(classification_report(y_test, y_prediction, target_names=["<=50K", ">50K"]))
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_prediction,
    display_labels=["<=50K", ">50K"],
    cmap="Blues",
    ax=axes[0],
)
axes[0].set_title("Matriz de confusión")
RocCurveDisplay.from_predictions(y_test, y_probability, ax=axes[1])
axes[1].plot([0, 1], [0, 1], "--", color="gray")
axes[1].set_title("Curva ROC")
plt.tight_layout()
plt.show()
"""),
    md("## 6. Variables más influyentes"),
    code("""
feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
importance = pd.Series(
    best_model.named_steps["model"].feature_importances_,
    index=feature_names,
).sort_values(ascending=False)

top_features = importance.head(20).sort_values()
ax = top_features.plot(kind="barh", figsize=(9, 7), color="#4C78A8")
ax.set_title("20 variables con mayor importancia en XGBoost")
ax.set_xlabel("Importancia")
plt.tight_layout()
plt.show()

display(importance.head(20).to_frame("importancia"))
"""),
    md("""
## 7. Conclusiones

- El pipeline separa los datos antes de imputar y codificar, evitando contaminación del conjunto de prueba.
- La validación cruzada selecciona los hiperparámetros usando ROC-AUC, una métrica adecuada para el desbalance observado.
- El conjunto de prueba se utiliza una sola vez para estimar el rendimiento final.
- En un entorno productivo convendría estudiar el umbral de decisión según el coste de falsos positivos y falsos negativos, además de vigilar deriva y equidad entre grupos.
"""),
]


unsupervised_cells = [
    md("""
# Clasificación no supervisada: segmentación de vinos

Este notebook aplica **UMAP + HDBSCAN** al dataset Wine de UCI. Las etiquetas reales se reservan exclusivamente para la evaluación final: no intervienen en el escalado, la reducción dimensional, la búsqueda de hiperparámetros ni el entrenamiento del clustering.
"""),
    md("## 1. Preparación"),
    code("""
import warnings

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap.umap_ as umap

from scipy.optimize import linear_sum_assignment
from ucimlrepo import fetch_ucirepo
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    adjusted_rand_score,
    confusion_matrix,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
SEED = 42
pd.set_option("display.max_columns", 100)
"""),
    md("## 2. Carga y análisis exploratorio"),
    code("""
wine = fetch_ucirepo(id=109)
X_raw = wine.data.features.copy()
y_true = wine.data.targets["class"].astype(int).copy()

print(f"Filas: {X_raw.shape[0]} | Variables: {X_raw.shape[1]}")
print(f"Valores ausentes: {X_raw.isna().sum().sum()}")
display(X_raw.head())
"""),
    code("""
display(X_raw.describe().T)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
X_raw[["Alcohol", "Color_intensity", "Proline"]].plot(kind="box", ax=axes[0])
axes[0].set_title("Escalas originales de variables seleccionadas")
axes[0].tick_params(axis="x", rotation=25)

y_true.value_counts().sort_index().plot(kind="bar", ax=axes[1], color=["#4C78A8", "#F58518", "#54A24B"], rot=0)
axes[1].set_title("Clases reales (solo referencia)")
axes[1].set_xlabel("Clase")
axes[1].set_ylabel("Vinos")
plt.tight_layout()
plt.show()
"""),
    md("""
## 3. Preprocesamiento

Todas las variables usadas por el algoritmo son características químicas. La etiqueta `class` permanece fuera de `X`. La imputación y el escalado se encapsulan en un pipeline reproducible.
"""),
    code("""
feature_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

X_scaled = feature_pipeline.fit_transform(X_raw)
print(f"Matriz de modelado: {X_scaled.shape}")
print(f"Media absoluta después de escalar: {np.abs(X_scaled.mean(axis=0)).max():.2e}")
"""),
    md("## 4. Proyección bidimensional con UMAP"),
    code("""
reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.05,
    n_components=2,
    metric="euclidean",
    random_state=SEED,
)
X_umap = reducer.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_umap[:, 0], X_umap[:, 1], s=30, alpha=0.75, color="#4C78A8")
plt.title("Proyección UMAP sin utilizar etiquetas")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.tight_layout()
plt.show()
"""),
    md("""
## 5. Selección no supervisada de HDBSCAN

Se explora una rejilla pequeña. La selección usa solamente dos señales internas: `relative_validity_` de HDBSCAN y la cobertura de puntos no clasificados como ruido. Las etiquetas reales no se consultan.
"""),
    code("""
search_rows = []

for min_cluster_size in [3, 5, 8, 10, 12, 15, 20]:
    for min_samples in [1, 3, 5, 8, 10]:
        candidate = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
            gen_min_span_tree=True,
        )
        labels = candidate.fit_predict(X_umap)
        mask = labels != -1
        n_clusters = len(set(labels[mask]))
        coverage = mask.mean()
        validity = float(candidate.relative_validity_)
        internal_score = validity * coverage if 2 <= n_clusters <= 6 else -np.inf

        search_rows.append({
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "clusters": n_clusters,
            "coverage": coverage,
            "relative_validity": validity,
            "internal_score": internal_score,
        })

search_results = pd.DataFrame(search_rows).sort_values("internal_score", ascending=False)
display(search_results.head(10).style.format({
    "coverage": "{:.3f}",
    "relative_validity": "{:.3f}",
    "internal_score": "{:.3f}",
}))
"""),
    code("""
best = search_results.iloc[0]
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=int(best["min_cluster_size"]),
    min_samples=int(best["min_samples"]),
    metric="euclidean",
    cluster_selection_method="eom",
    gen_min_span_tree=True,
    prediction_data=True,
)
cluster_labels = clusterer.fit_predict(X_umap)
valid_mask = cluster_labels != -1

n_clusters = len(set(cluster_labels[valid_mask]))
noise_ratio = 1 - valid_mask.mean()
silhouette = silhouette_score(X_umap[valid_mask], cluster_labels[valid_mask]) if n_clusters > 1 else np.nan

print(f"Parámetros seleccionados: min_cluster_size={int(best['min_cluster_size'])}, min_samples={int(best['min_samples'])}")
print(f"Clústeres encontrados: {n_clusters}")
print(f"Ruido: {noise_ratio:.2%}")
print(f"Silhouette sin ruido: {silhouette:.4f}")
"""),
    code("""
plt.figure(figsize=(8, 6))
for label in sorted(np.unique(cluster_labels)):
    mask = cluster_labels == label
    name = "Ruido" if label == -1 else f"Clúster {label}"
    color = "#BDBDBD" if label == -1 else None
    plt.scatter(X_umap[mask, 0], X_umap[mask, 1], s=35, alpha=0.8, label=name, color=color)

plt.title("Clústeres HDBSCAN sobre la proyección UMAP")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.legend()
plt.tight_layout()
plt.show()
"""),
    md("""
## 6. Evaluación externa

Solo después de cerrar el modelo se comparan los clústeres con las clases reales. ARI y NMI son invariantes al nombre asignado a cada clúster. Para visualizar una matriz de confusión, la asignación clúster-clase se obtiene mediante el algoritmo húngaro y los puntos de ruido permanecen como `-1`.
"""),
    code("""
ari = adjusted_rand_score(y_true, cluster_labels)
nmi = normalized_mutual_info_score(y_true, cluster_labels)

valid_clusters = sorted(set(cluster_labels[valid_mask]))
true_classes = sorted(y_true.unique())
contingency = pd.crosstab(
    pd.Series(y_true[valid_mask].to_numpy(), name="clase"),
    pd.Series(cluster_labels[valid_mask], name="cluster"),
).reindex(index=true_classes, columns=valid_clusters, fill_value=0)

row_ind, col_ind = linear_sum_assignment(-contingency.to_numpy())
cluster_to_class = {
    contingency.columns[col]: contingency.index[row]
    for row, col in zip(row_ind, col_ind)
}
mapped_labels = np.array([cluster_to_class.get(label, -1) for label in cluster_labels])
accuracy_without_noise = (mapped_labels[valid_mask] == y_true.to_numpy()[valid_mask]).mean()

external_metrics = pd.Series({
    "adjusted_rand_index": ari,
    "normalized_mutual_information": nmi,
    "accuracy_sin_ruido_tras_mapeo": accuracy_without_noise,
    "cobertura": valid_mask.mean(),
    "silhouette_sin_ruido": silhouette,
}, name="resultado").to_frame()
display(external_metrics.style.format("{:.4f}"))
print("Mapeo clúster → clase:", cluster_to_class)
"""),
    code("""
labels_for_matrix = true_classes + [-1]
matrix = confusion_matrix(y_true, mapped_labels, labels=labels_for_matrix)
disp = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=[str(x) for x in true_classes] + ["Ruido"],
)
disp.plot(cmap="Blues")
plt.title("Clústeres mapeados frente a clases reales")
plt.tight_layout()
plt.show()
"""),
    md("""
## 7. Conclusiones

- La etiqueta real quedó completamente aislada del proceso de entrenamiento y selección.
- UMAP facilita una representación local compacta; HDBSCAN determina automáticamente el número de clústeres y puede identificar ruido.
- ARI y NMI permiten evaluar la correspondencia con las clases conocidas sin depender de la numeración arbitraria de los clústeres.
- El resultado es exploratorio: para uso empresarial habría que comprobar estabilidad entre semillas, validar el significado operativo de cada segmento y monitorizar cambios en la distribución.
"""),
]


def write_notebook(filename, cells):
    notebook = nbf.v4.new_notebook(cells=cells, metadata=KERNEL)
    notebook["nbformat"] = 4
    notebook["nbformat_minor"] = 5
    nbf.write(notebook, OUT / filename)


write_notebook("1_supervised_classification_model.ipynb", supervised_cells)
write_notebook("2_unsupervised_classification_model.ipynb", unsupervised_cells)
print("Notebooks reconstruidos correctamente.")
