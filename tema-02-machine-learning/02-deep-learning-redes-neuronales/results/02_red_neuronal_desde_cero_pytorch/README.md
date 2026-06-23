# 02-02 — Neurona y MLP desde cero con PyTorch

## Estado

- Estado: `COMPLETED`
- Modo: `full`
- Publicable: `true`
- Semilla: `42`
- Dispositivo: `cpu`
- Dataset: `MNIST`
- Framework: `PyTorch`

## Origen y cambios

- Original protegido: `input_notebooks/1_red_neuronal_desde_cero.ipynb`
- Notebook canónico: `notebooks/02_red_neuronal_desde_cero_pytorch.ipynb`
- Arquitectura: Neurona explícita con autograd y MLP 784→128→1
- Se añadieron train, validation y test separados, selección por validation, evaluación final única de test, métricas, curvas y ejemplos de error.

## Ejecución

```powershell
python tema-02-machine-learning/scripts/execute_notebooks.py --target 02-02 --mode full
```

## Resultados regenerados

| Experimento | Modelo | Parámetros | Mejor época | Mejor val. accuracy | Test accuracy | Precisión | Recall | F1 | Tiempo |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `neuron_mnist_0_vs_1` | neuron | 785 | 15 | 99.80 % | 99.95 % | 99.91 % | 100.00 % | 99.96 % | 8.07 s |
| `neuron_mnist_5_vs_6` | neuron | 785 | 15 | 97.75 % | 97.89 % | 97.72 % | 98.23 % | 97.97 % | 17.58 s |
| `mlp_mnist_5_vs_6` | mlp | 100,609 | 11 | 99.38 % | 99.08 % | 98.96 % | 99.27 % | 99.11 % | 28.08 s |

Los valores proceden directamente de `experiments/full/*/metrics.json`.

## Interpretación y limitaciones

Los pares `0 vs 1` son casi linealmente separables. `5 vs 6` comparte trazos y resulta más exigente. Los tiempos corresponden a CPU en este equipo y no deben generalizarse a otros entornos.

El ejercicio utiliza MNIST, un dataset pequeño y limpio. No evalúa deriva, sesgo operacional, seguridad, latencia productiva ni mantenimiento.

## Aplicación empresarial

El flujo es trasladable conceptualmente a baselines de clasificación de incidencias en un futuro MDS AI Operations Center. Antes de producción serían necesarios datos corporativos gobernados, validación temporal, explicabilidad, monitorización y revisión humana.
