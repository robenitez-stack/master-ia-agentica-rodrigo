# Módulo 2 — Fundamentos de Deep Learning y redes neuronales

## Objetivos

- Implementar una neurona binaria manual con NumPy.
- Hacer explícitas las fases forward, loss, backward y update.
- Comprender el papel de autograd en PyTorch.
- Comparar una frontera lineal con un MLP sencillo.

## Notebooks

| Orden | Notebook | Framework | Estado |
|---:|---|---|---|
| 01 | `01_neurona_manual_numpy_mnist.ipynb` | NumPy | Validado en modo fast; full pendiente |
| 02 | `02_red_neuronal_desde_cero_pytorch.ipynb` | PyTorch | Validado en modo fast; full pendiente |

Los resultados se generan automáticamente dentro de `results/`. Las ejecuciones `fast` son pruebas técnicas y no resultados académicos publicables.

<!-- FULL_RESULTS:START -->
## Resultados full del módulo 2

| Notebook | Experimento | Modelo | Parámetros | Test accuracy | Test F1 | Tiempo CPU |
|---|---|---|---:|---:|---:|---:|
| 02-01 | `mnist_0_vs_1` | neurona NumPy | 785 | 99.95 % | 99.96 % | 3.06 s |
| 02-01 | `mnist_5_vs_6` | neurona NumPy | 785 | 98.00 % | 98.08 % | 4.33 s |
| 02-02 | `neuron_mnist_0_vs_1` | neuron | 785 | 99.95 % | 99.96 % | 8.07 s |
| 02-02 | `neuron_mnist_5_vs_6` | neuron | 785 | 97.89 % | 97.97 % | 17.58 s |
| 02-02 | `mlp_mnist_5_vs_6` | mlp | 100,609 | 99.08 % | 99.11 % | 28.08 s |

Fuente: archivos `run_summary.json` y `experiments/full/*/metrics.json`. Los smoke tests `fast` no se publican como resultados académicos.
<!-- FULL_RESULTS:END -->
