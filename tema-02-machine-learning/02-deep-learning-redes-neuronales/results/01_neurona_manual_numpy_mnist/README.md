# 02-01 — Neurona manual NumPy sobre MNIST

## Estado

- Estado: `COMPLETED`
- Modo: `full`
- Publicable: `true`
- Semilla: `42`
- Dispositivo: `cpu`
- Dataset: `MNIST`
- Framework: `NumPy`

## Origen y cambios

- Original protegido: `input_notebooks/1_manual_neural_network_model.ipynb`
- Notebook canónico: `notebooks/01_neurona_manual_numpy_mnist.ipynb`
- Arquitectura: Neurona logística manual: 784 pesos y un sesgo
- Se añadieron train, validation y test separados, selección por validation, evaluación final única de test, métricas, curvas y ejemplos de error.

## Ejecución

```powershell
python tema-02-machine-learning/scripts/execute_notebooks.py --target 02-01 --mode full
```

## Resultados regenerados

| Experimento | Modelo | Parámetros | Mejor época | Mejor val. accuracy | Test accuracy | Precisión | Recall | F1 | Tiempo |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mnist_0_vs_1` | neurona NumPy | 785 | 25 | 99.80 % | 99.95 % | 99.91 % | 100.00 % | 99.96 % | 3.38 s |
| `mnist_5_vs_6` | neurona NumPy | 785 | 25 | 97.75 % | 98.00 % | 97.72 % | 98.43 % | 98.08 % | 3.42 s |

Los valores proceden directamente de `experiments/full/*/metrics.json`.

## Interpretación y limitaciones

Los pares `0 vs 1` son casi linealmente separables. `5 vs 6` comparte trazos y resulta más exigente. Los tiempos corresponden a CPU en este equipo y no deben generalizarse a otros entornos.

El ejercicio utiliza MNIST, un dataset pequeño y limpio. No evalúa deriva, sesgo operacional, seguridad, latencia productiva ni mantenimiento.

## Aplicación empresarial

El flujo es trasladable conceptualmente a baselines de clasificación de incidencias en un futuro MDS AI Operations Center. Antes de producción serían necesarios datos corporativos gobernados, validación temporal, explicabilidad, monitorización y revisión humana.
