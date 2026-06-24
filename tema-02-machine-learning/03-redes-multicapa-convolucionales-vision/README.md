# Módulo 3 — Redes multicapas, convolucionales y visión artificial

Este módulo desarrolla seis notebooks canónicos sobre MLP y CNN con MNIST y CIFAR-10, utilizando PyTorch y TensorFlow/Keras.

| Orden | Notebook | Framework | Estado |
|---:|---|---|---|
| 01 | `01_mlp_mnist_pytorch.ipynb` | PyTorch | Full validado — accuracy 97.89 % |
| 02 | `02_cnn_mnist_keras.ipynb` | TensorFlow/Keras | Full validado — accuracy 99.28 % |
| 03 | `03_cnn_mnist_pytorch_base.ipynb` | PyTorch | Full validado — accuracy 97.98 % |
| 04 | `04_cnn_cifar10_keras.ipynb` | TensorFlow/Keras | Full validado — accuracy 76.78 % |
| 05 | `05_cnn_mnist_pytorch_mejorada.ipynb` | PyTorch | Full validado — accuracy 99.60 % |
| 06 | `06_cnn_cifar10_pytorch.ipynb` | PyTorch | Full validado — accuracy 88.37 % |

Los resultados `full` son los resultados académicos publicados. Los artefactos detallados están en `results/`; los resultados `fast` se conservan como smoke tests no publicables.

## Comparación

En MNIST, la CNN mejorada con PyTorch obtuvo el mejor resultado (99.60 %). En CIFAR-10, la CNN con PyTorch alcanzó 88.37 %, frente a 76.78 % de la CNN con Keras. La comparación refleja estas configuraciones concretas y no una superioridad general entre frameworks.
