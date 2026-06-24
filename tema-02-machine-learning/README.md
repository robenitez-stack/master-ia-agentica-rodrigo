# Tema 2 — Machine Learning, Deep Learning y LLMs

Este directorio organiza de forma progresiva los contenidos del Tema 2 del Máster en Ingeniería de Automatización con IA Agéntica.

## Mapa modular

| Módulo | Contenido | Estado |
|---|---|---|
| 1 | Introducción al Machine Learning | Completado; ubicación histórica protegida |
| 2 | Fundamentos de Deep Learning y redes neuronales | Completado y validado en modo full |
| 3 | MLP, CNN y visión artificial | Completado y validado en modo full |
| 4 | Transformers y procesamiento de lenguaje natural | Pendiente de materiales |
| 5 | Introducción a los LLMs | Pendiente de materiales |

## Módulo 1 — ubicación histórica

Estos notebooks permanecen intactos en la raíz del Tema 2:

- `1_supervised_classification_model.ipynb`
- `2_unsupervised_classification_model.ipynb`

## Inventario de Deep Learning

- Módulo 2: dos notebooks sobre neuronas binarias, NumPy, PyTorch y autograd.
- Módulo 3: seis notebooks sobre MLP y CNN con MNIST y CIFAR-10, usando PyTorch y TensorFlow/Keras.
- Los originales se conservan en `input_notebooks/` y se verifican mediante SHA-256.

## Instalación

Desde la raíz del repositorio:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r tema-02-machine-learning/requirements/common.txt
python -m pip install -r tema-02-machine-learning/requirements/pytorch.txt
python -m pip install -r tema-02-machine-learning/requirements/tensorflow.txt
```

El kernel recomendado es `Python - Master IA Agentica`.

## Modos de ejecución

```powershell
$env:RUN_MODE = "fast"  # smoke test; resultados no publicables
$env:RUN_MODE = "full"  # entrenamiento y resultados definitivos
```

Los datasets se descargan una sola vez en `.data/`, carpeta excluida de Git.

## Política de datos

```text
train      → aprender parámetros
validation → seleccionar configuración y checkpoint
test       → evaluación final única
```

Los resultados `fast` se identifican con `publishable: false`. Solo una ejecución `full` finalizada y validada puede incorporarse a una tabla comparativa.

## Relación conceptual

Machine Learning introduce modelado y evaluación; Deep Learning amplía esa base mediante representaciones aprendidas; las CNN explotan estructura espacial; Transformers extienden el aprendizaje de representaciones a secuencias; los LLMs escalan esa arquitectura para lenguaje y razonamiento asistido.

En un futuro **MDS AI Operations Center**, estos conceptos pueden apoyar clasificación de incidencias, detección de anomalías, análisis visual, procesamiento de registros y agentes de asistencia. Los ejercicios académicos no constituyen por sí solos soluciones listas para producción.

<!-- MODULE2_RESULTS:START -->
## Estado del módulo 2

Los dos notebooks del módulo 2 están desarrollados, ejecutados de principio a fin en modo `full` y validados en CPU. La documentación detallada y los artefactos se encuentran en `02-deep-learning-redes-neuronales/results/`.

## Resultados full del módulo 2

| Notebook | Experimento | Modelo | Parámetros | Test accuracy | Test F1 | Tiempo CPU |
|---|---|---|---:|---:|---:|---:|
| 02-01 | `mnist_0_vs_1` | neurona NumPy | 785 | 99.95 % | 99.96 % | 3.38 s |
| 02-01 | `mnist_5_vs_6` | neurona NumPy | 785 | 98.00 % | 98.08 % | 3.42 s |
| 02-02 | `neuron_mnist_0_vs_1` | neuron | 785 | 99.95 % | 99.96 % | 8.60 s |
| 02-02 | `neuron_mnist_5_vs_6` | neuron | 785 | 97.89 % | 97.97 % | 9.08 s |
| 02-02 | `mlp_mnist_5_vs_6` | mlp | 100,609 | 99.08 % | 99.11 % | 14.62 s |

Fuente: archivos `run_summary.json` y `experiments/full/*/metrics.json`. Los smoke tests `fast` no se publican como resultados académicos.
<!-- MODULE2_RESULTS:END -->

<!-- MODULE3_RESULTS:START -->
## Estado del módulo 3

Los seis notebooks del módulo 3 están desarrollados, ejecutados de principio a fin en modo `full` y validados en CPU. Cada ejecución conserva métricas, historial, reporte por clase, curvas de aprendizaje, matriz de confusión y ejemplos de predicción.

## Resultados full del módulo 3

| Notebook | Dataset | Framework/modelo | Parámetros | Test accuracy | Test F1 | Tiempo CPU |
|---|---|---|---:|---:|---:|---:|
| 03-01 | MNIST | PyTorch MLP | 235,146 | 97.89 % | 97.87 % | 37.79 s |
| 03-02 | MNIST | Keras CNN | 422,026 | 99.28 % | 99.27 % | 210.15 s |
| 03-03 | MNIST | PyTorch CNN base | 31,530 | 97.98 % | 97.97 % | 88.07 s |
| 03-04 | CIFAR-10 | Keras CNN | 591,658 | 76.78 % | 76.51 % | 999.89 s |
| 03-05 | MNIST | PyTorch CNN mejorada | 468,458 | 99.60 % | 99.60 % | 1,387.56 s |
| 03-06 | CIFAR-10 | PyTorch CNN | 815,530 | 88.37 % | 88.28 % | 5,946.31 s |

Fuente: `03-redes-multicapa-convolucionales-vision/results/*/run_summary.json` y `experiments/full/*/metrics.json`. Los resultados `fast` se conservan solamente como evidencia técnica.
<!-- MODULE3_RESULTS:END -->
