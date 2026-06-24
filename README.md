# Máster en Ingeniería de Automatización con IA Agéntica

Repositorio de Rodrigo Benítez para los trabajos prácticos del Máster en Ingeniería de Automatización con IA Agéntica.

## Estructura

- `tema-01-entorno/`: preparación y ejercicios del entorno.
- `tema-02-machine-learning/`: cuatro módulos de Machine Learning, Deep Learning, visión artificial y NLP.
- `datasets/`: datos locales de trabajo (los archivos pesados no se versionan).
- `docs/`: documentación y guías del proyecto.

## Preparación en Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

En VS Code, selecciona el intérprete de `.venv` y el kernel **Python - Master IA Agentica**.

## Tema 2 — módulos desarrollados

| Módulo | Contenido | Estado |
|---|---|---|
| 1 | Clasificación supervisada y no supervisada | Completado y protegido |
| 2 | Redes neuronales desde cero con NumPy y PyTorch | Ejecutado y validado en modo `full` |
| 3 | MLP, CNN, MNIST y CIFAR-10 con PyTorch y Keras | Ejecutado y validado en modo `full` |
| 4 | Tokenización, embeddings, similitud coseno, GloVe y PCA | Ejecutado y validado |

La documentación completa, los notebooks y los resultados reproducibles están en [`tema-02-machine-learning/`](tema-02-machine-learning/README.md).

## Resultados reproducidos

Los dos notebooks fueron ejecutados de principio a fin con Python 3.10:

- **Clasificación supervisada (Adult + XGBoost):** ROC-AUC de prueba 0,9318, accuracy 0,8788 y F1 0,7247 para la clase de ingresos altos.
- **Clasificación no supervisada (Wine + UMAP/HDBSCAN):** 3 clústeres, silhouette 0,7302, ARI 0,8025 y NMI 0,7955.

La etiqueta real del dataset Wine se mantiene fuera del entrenamiento y se utiliza únicamente en la evaluación externa final.

Los módulos posteriores incorporan separación `train/validation/test`, modos de ejecución `fast` y `full`, métricas, historiales, matrices de confusión, ejemplos de predicción y documentación reproducible.
