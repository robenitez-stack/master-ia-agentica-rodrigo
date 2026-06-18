# Máster en IA Agéntica

Repositorio de Rodrigo Benítez para los trabajos prácticos del Máster en Ingeniería de Automatización con IA Agéntica.

## Estructura

- `tema-01-entorno/`: preparación y ejercicios del entorno.
- `tema-02-machine-learning/`: notebooks de clasificación supervisada y no supervisada.
- `datasets/`: datos locales de trabajo (los archivos pesados no se versionan).
- `docs/`: documentación y guías del proyecto.

## Preparación en Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

En VS Code, selecciona el intérprete de `.venv` y el kernel **Python - Master IA Agentica**.

## Notebooks

1. `tema-02-machine-learning/1_supervised_classification_model.ipynb`
2. `tema-02-machine-learning/2_unsupervised_classification_model.ipynb`

## Resultados reproducidos

Los dos notebooks fueron ejecutados de principio a fin con Python 3.10:

- **Clasificación supervisada (Adult + XGBoost):** ROC-AUC de prueba 0,9318, accuracy 0,8788 y F1 0,7247 para la clase de ingresos altos.
- **Clasificación no supervisada (Wine + UMAP/HDBSCAN):** 3 clústeres, silhouette 0,7302, ARI 0,8025 y NMI 0,7955.

La etiqueta real del dataset Wine se mantiene fuera del entrenamiento y se utiliza únicamente en la evaluación externa final.

El script `scripts/build_notebooks.py` permite reconstruir las versiones limpias de ambos notebooks; después deben ejecutarse para regenerar sus resultados y gráficos.
