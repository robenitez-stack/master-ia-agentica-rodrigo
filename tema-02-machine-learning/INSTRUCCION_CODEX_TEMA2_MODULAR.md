## Tema 2 — Introducción al Machine Learning, Deep Learning y LLMs
### Versión modular actualizada

---

# 0. Propósito de esta instrucción

Debes organizar, corregir, documentar, ejecutar y preparar para Git los notebooks pendientes del Tema 2 del Máster en Ingeniería de Automatización con IA Agéntica.

Esta instrucción reemplaza las versiones anteriores relacionadas con el Tema 2. Si encuentras indicaciones antiguas que contradicen este documento, aplica este documento como fuente principal para este trabajo.

No debes comenzar modificando archivos. La primera intervención es exclusivamente una auditoría.

---

# 1. Contexto del repositorio

Estás trabajando dentro de un monorepositorio existente llamado:

```text
master-ia-agentica
```

La raíz del repositorio contiene una estructura equivalente a:

```text
master-ia-agentica/
├── .git/
├── .venv/
├── datasets/
├── docs/
├── scripts/
├── tema-01-entorno/
├── tema-02-machine-learning/
├── .gitignore
├── README.md
└── requirements.txt
```

## Reglas de repositorio

- No crees un repositorio nuevo.
- No ejecutes `git init`.
- No elimines ni reemplaces la carpeta `.git`.
- No modifiques `tema-01-entorno`.
- No reescribas el historial existente.
- No uses comandos Git destructivos.
- No hagas `git push` sin autorización expresa.
- El trabajo debe integrarse en:

```text
tema-02-machine-learning/
```

---

# 2. Estructura académica del Tema 2

El Tema 2 se divide en cinco módulos progresivos:

1. Introducción al Machine Learning.
2. Fundamentos de Deep Learning y redes neuronales.
3. Redes neuronales multicapas, convolucionales y visión artificial.
4. Transformers y procesamiento de lenguaje natural.
5. Introducción a los LLMs.

## Estado actual

```text
Módulo 1 — Introducción al Machine Learning: ya desarrollado
Módulo 2 — Fundamentos de Deep Learning y redes neuronales: por desarrollar
Módulo 3 — Redes multicapas, CNN y visión artificial: por desarrollar
Módulo 4 — Transformers y NLP: pendiente de materiales
Módulo 5 — Introducción a los LLMs: pendiente de materiales
```

## Protección del módulo 1

El módulo 1 ya fue creado y avanzado.

Reglas obligatorias:

- Detecta el nombre real de su carpeta.
- No lo recrees.
- No lo borres.
- No lo renombres sin autorización.
- No reorganices sus notebooks.
- No modifiques su código ni sus resultados.
- No cambies sus dependencias de forma que dejen de funcionar.
- Solo puedes actualizar enlaces o referencias en el README general del Tema 2.
- Antes de cada commit, comprueba que el módulo 1 no tenga cambios accidentales.

Los módulos 4 y 5 no deben desarrollarse todavía. Solo deben aparecer como pendientes en la documentación general.

---

# 3. Inventario definitivo: ocho notebooks

Debes trabajar con exactamente estos ocho notebooks:

```text
1_manual_neural_network_model.ipynb
1_red_neuronal_desde_cero.ipynb
1_red_neuronal_multicapa.ipynb
2_convolutional_neural_network_model.ipynb
2_red_convolucional_mnist.ipynb
3_convolutional_neural_network_model_animals.ipynb
3_red_convolucional_mnist_mejorada.ipynb
4_red_convolucional_cifar10.ipynb
```

No vuelvas a referirte a ellos como “seis notebooks”.

## Distribución definitiva

### Módulo 2 — Fundamentos de Deep Learning y redes neuronales

```text
1_manual_neural_network_model.ipynb
1_red_neuronal_desde_cero.ipynb
```

### Módulo 3 — Redes multicapas, convolucionales y visión artificial

```text
1_red_neuronal_multicapa.ipynb
2_convolutional_neural_network_model.ipynb
2_red_convolucional_mnist.ipynb
3_convolutional_neural_network_model_animals.ipynb
3_red_convolucional_mnist_mejorada.ipynb
4_red_convolucional_cifar10.ipynb
```

No dupliques los notebooks del módulo 2 dentro del módulo 3.

---

# 4. Nombres canónicos propuestos

Conserva los originales sin modificar y crea copias desarrolladas con nombres consistentes.

## Módulo 2

| Orden | Notebook original | Nombre desarrollado recomendado |
|---:|---|---|
| 01 | `1_manual_neural_network_model.ipynb` | `01_neurona_manual_numpy_mnist.ipynb` |
| 02 | `1_red_neuronal_desde_cero.ipynb` | `02_red_neuronal_desde_cero_pytorch.ipynb` |

## Módulo 3

| Orden | Notebook original | Nombre desarrollado recomendado |
|---:|---|---|
| 01 | `1_red_neuronal_multicapa.ipynb` | `01_mlp_mnist_pytorch.ipynb` |
| 02 | `2_convolutional_neural_network_model.ipynb` | `02_cnn_mnist_keras.ipynb` |
| 03 | `2_red_convolucional_mnist.ipynb` | `03_cnn_mnist_pytorch_base.ipynb` |
| 04 | `3_convolutional_neural_network_model_animals.ipynb` | `04_cnn_cifar10_keras.ipynb` |
| 05 | `3_red_convolucional_mnist_mejorada.ipynb` | `05_cnn_mnist_pytorch_mejorada.ipynb` |
| 06 | `4_red_convolucional_cifar10.ipynb` | `06_cnn_cifar10_pytorch.ipynb` |

Si el repositorio ya utiliza otra convención clara, presenta la alternativa durante la auditoría y espera aprobación antes de renombrar.

---

# 5. Primera fase obligatoria: auditoría sin modificaciones

En la primera interacción no debes modificar ningún archivo.

## Acciones de auditoría

1. Inspecciona la estructura completa de `tema-02-machine-learning`.
2. Identifica el nombre y la ubicación real del módulo 1.
3. Localiza los ocho notebooks.
4. Identifica si alguno ya tiene una copia desarrollada.
5. Inspecciona el `README` del Tema 2.
6. Inspecciona el `requirements.txt` de la raíz y cualquier archivo de dependencias existente.
7. Inspecciona `.gitignore`.
8. Inspecciona el estado del entorno virtual sin modificarlo.
9. Ejecuta:

```bash
git status --short
git branch --show-current
git log --oneline -10
```

10. Revisa cada notebook:
   - framework;
   - dataset;
   - arquitectura;
   - propósito pedagógico;
   - imports;
   - dependencias;
   - instalación de paquetes dentro de celdas;
   - rutas locales;
   - rutas de datasets;
   - uso de CPU, CUDA o MPS;
   - semilla;
   - train, validation y test;
   - posible fuga de datos;
   - checkpoints;
   - callbacks;
   - celdas duplicadas;
   - variables inconsistentes;
   - errores lógicos;
   - outputs existentes;
   - métricas existentes;
   - tamaño del notebook;
   - tiempo y hardware estimados.
11. No entrenes modelos.
12. No instales paquetes.
13. No copies ni renombres archivos.
14. No hagas commits.

## Respuesta obligatoria de auditoría

Entrega:

1. árbol actual relevante;
2. identificación del módulo 1;
3. ubicación de los ocho notebooks;
4. estado Git inicial;
5. problemas encontrados por notebook;
6. estructura objetivo adaptada al repositorio real;
7. estrategia de respaldo de originales;
8. estrategia de dependencias;
9. estrategia de datasets;
10. riesgos de CPU, GPU, memoria y tiempo;
11. plan de implementación;
12. plan de validación;
13. plan de commits;
14. preguntas o bloqueos.

Después detente y espera aprobación.

---

# 6. Estructura objetivo corregida

Adapta la estructura a las convenciones existentes, pero debe quedar conceptualmente equivalente a:

```text
tema-02-machine-learning/
├── README.md
├── AGENTS.md
├── INSTRUCCION_CODEX_TEMA2_MODULAR.md
│
├── input_notebooks/
│   ├── 1_manual_neural_network_model.ipynb
│   ├── 1_red_neuronal_desde_cero.ipynb
│   ├── 1_red_neuronal_multicapa.ipynb
│   ├── 2_convolutional_neural_network_model.ipynb
│   ├── 2_red_convolucional_mnist.ipynb
│   ├── 3_convolutional_neural_network_model_animals.ipynb
│   ├── 3_red_convolucional_mnist_mejorada.ipynb
│   └── 4_red_convolucional_cifar10.ipynb
│
├── 01-introduccion-machine-learning/
│   └── CONTENIDO EXISTENTE — NO MODIFICAR
│
├── 02-deep-learning-redes-neuronales/
│   ├── README.md
│   ├── notebooks/
│   │   ├── 01_neurona_manual_numpy_mnist.ipynb
│   │   └── 02_red_neuronal_desde_cero_pytorch.ipynb
│   └── results/
│       ├── 01_neurona_manual_numpy_mnist/
│       └── 02_red_neuronal_desde_cero_pytorch/
│
├── 03-redes-multicapa-convolucionales-vision/
│   ├── README.md
│   ├── notebooks/
│   │   ├── 01_mlp_mnist_pytorch.ipynb
│   │   ├── 02_cnn_mnist_keras.ipynb
│   │   ├── 03_cnn_mnist_pytorch_base.ipynb
│   │   ├── 04_cnn_cifar10_keras.ipynb
│   │   ├── 05_cnn_mnist_pytorch_mejorada.ipynb
│   │   └── 06_cnn_cifar10_pytorch.ipynb
│   └── results/
│       ├── 01_mlp_mnist_pytorch/
│       ├── 02_cnn_mnist_keras/
│       ├── 03_cnn_mnist_pytorch_base/
│       ├── 04_cnn_cifar10_keras/
│       ├── 05_cnn_mnist_pytorch_mejorada/
│       └── 06_cnn_cifar10_pytorch/
│
├── requirements/
│   ├── common.txt
│   ├── pytorch.txt
│   └── tensorflow.txt
│
└── scripts/
    ├── execute_notebooks.py
    ├── validate_notebooks.py
    ├── collect_environment.py
    └── summarize_results.py
```

## Consideraciones

- `input_notebooks/` es la ubicación recomendada para originales inmutables.
- Si los originales ya están protegidos en otra carpeta o mediante una convención existente, no los dupliques automáticamente: presenta la alternativa.
- No crees carpetas físicas para los módulos 4 y 5 si el repositorio no usa placeholders.
- Documenta los módulos 4 y 5 como pendientes en el README del Tema 2.

---

# 7. Protección de originales e historial

Antes de refactorizar:

- Confirma que existe una copia intacta de cada notebook original.
- Calcula y registra opcionalmente su hash SHA-256.
- No sobrescribas la única copia.
- Trabaja sobre copias desarrolladas.
- No elimines los originales después.
- No cambies outputs originales para “limpiarlos”.
- No uses:
  - `git reset --hard`;
  - `git clean -fd`;
  - `git checkout -- .`;
  - `git restore .` sin revisión;
  - `git push --force`;
  - rebase de commits compartidos.
- No reviertas cambios del usuario que no estén relacionados con este trabajo.

---

# 8. Reglas comunes para los ocho notebooks

Cada notebook desarrollado debe:

1. Estar escrito y comentado en español.
2. Mantener el objetivo pedagógico original.
3. Tener títulos y explicaciones Markdown.
4. Incluir:
   - objetivo;
   - conceptos técnicos;
   - dataset;
   - exploración mínima;
   - preparación de datos;
   - separación de datos;
   - arquitectura;
   - función de pérdida;
   - optimizador o algoritmo de actualización;
   - hiperparámetros;
   - entrenamiento;
   - validación;
   - evaluación;
   - interpretación;
   - limitaciones;
   - aplicación empresarial;
   - posible aplicación conceptual para Marina del Sol;
   - conclusiones.
5. Usar semilla `42` en Python, NumPy, PyTorch y TensorFlow cuando corresponda.
6. Intentar habilitar operaciones deterministas cuando sea razonable, documentando sus límites.
7. Detectar CPU, CUDA o MPS.
8. Registrar el dispositivo utilizado.
9. Evitar rutas absolutas.
10. Usar `pathlib.Path`.
11. Reutilizar la carpeta común de datasets cuando sea compatible.
12. No instalar paquetes silenciosamente en cada ejecución.
13. Sustituir celdas `!pip install` por:
    - una celda Markdown de instalación; o
    - una celda opcional claramente marcada.
14. Separar correctamente train, validation y test.
15. No usar test para:
    - elegir hiperparámetros;
    - seleccionar checkpoint;
    - decidir early stopping;
    - ajustar el learning rate;
    - comparar variantes durante el desarrollo.
16. Evaluar test una sola vez al final de cada experimento definitivo.
17. Registrar:
    - loss de train;
    - loss de validation, cuando aplique;
    - métricas por época;
    - tiempo;
    - número de parámetros;
    - mejor época;
    - configuración;
    - resultado final.
18. Generar:
    - curvas de aprendizaje;
    - matriz de confusión;
    - classification report;
    - ejemplos correctos;
    - ejemplos mal clasificados.
19. Explicar:
    - overfitting;
    - underfitting;
    - generalización;
    - clases confundidas;
    - limitaciones del experimento.
20. Ejecutarse desde un kernel limpio, de arriba abajo y sin errores.
21. No contener celdas duplicadas.
22. No dejar código abandonado.
23. No inventar resultados.
24. No copiar manualmente cifras antiguas.
25. Distinguir resultados históricos de resultados regenerados.
26. Guardar outputs y artefactos de forma ordenada.
27. Añadir conclusión técnica y conclusión ejecutiva.
28. No afirmar que el ejercicio académico está listo para producción.

---

# 9. Política de train, validation y test

## Principio general

```text
train      → aprender parámetros
validation → elegir configuración y checkpoint
test       → evaluación final no contaminada
```

## Reglas

- Crea validation desde el train original cuando el dataset no la proporcione.
- Usa partición estratificada en problemas binarios cuando sea posible.
- Conserva la distribución de clases.
- Registra tamaños y proporciones.
- Usa una semilla fija para el split.
- No apliques data augmentation aleatorio a validation o test.
- Ajusta normalización con estadísticas conocidas o derivadas solo de train.
- No vuelvas a consultar test después de ajustar el modelo por sus resultados.
- Si se realizan varios experimentos dentro de un notebook, cada uno debe conservar la misma política.

---

# 10. Modos de ejecución

Todos los notebooks deben soportar:

```python
import os

RUN_MODE = os.getenv("RUN_MODE", "full").lower()

if RUN_MODE not in {"fast", "full"}:
    raise ValueError("RUN_MODE debe ser 'fast' o 'full'")
```

## Modo `fast`

Se utiliza para:

- comprobar imports;
- comprobar descarga o acceso a datos;
- validar rutas;
- ejecutar smoke tests;
- usar pocas épocas o iteraciones;
- usar un subconjunto estratificado si es necesario;
- validar la generación de artefactos.

Sus métricas deben marcarse como no publicables:

```text
publishable: false
run_mode: fast
```

## Modo `full`

Se utiliza para:

- entrenamiento completo;
- métricas oficiales;
- gráficos definitivos;
- notebook ejecutado;
- resultados publicados en README.

Debe registrar:

```text
publishable: true
run_mode: full
```

## Fallo de ejecución completa

Si no se puede finalizar por hardware, memoria o tiempo:

- no inventes métricas;
- no reutilices outputs antiguos como si fueran nuevos;
- conserva el smoke test por separado;
- marca:

```text
status: FULL_RUN_PENDING
publishable: false
```

- explica la causa;
- entrega el comando exacto para continuar.

---

# 11. Resultados y artefactos

Cada notebook debe tener su propia carpeta:

```text
results/<notebook_id>/
├── notebook_executed.ipynb
├── run_summary.json
├── environment.txt
├── README.md
└── experiments/
    └── <experiment_id>/
        ├── metrics.json
        ├── training_history.csv
        ├── learning_curves.png
        ├── confusion_matrix.png
        ├── classification_report.csv
        ├── sample_predictions.png
        └── misclassified_examples.png
```

Esto permite que los notebooks con más de un experimento, como `0 vs 1` y `5 vs 6`, mantengan resultados separados.

## Esquema mínimo de `run_summary.json`

```json
{
  "notebook_id": "02-01",
  "source_notebook": "1_manual_neural_network_model.ipynb",
  "canonical_notebook": "01_neurona_manual_numpy_mnist.ipynb",
  "module": "02-deep-learning-redes-neuronales",
  "status": "COMPLETED",
  "run_mode": "full",
  "publishable": true,
  "seed": 42,
  "device": "cpu",
  "frameworks": ["numpy"],
  "dataset": "MNIST",
  "run_timestamp_utc": "GENERATED_AT_RUNTIME",
  "git_commit": "GENERATED_AT_RUNTIME",
  "experiments": [
    "mnist_0_vs_1",
    "mnist_5_vs_6"
  ]
}
```

## Esquema mínimo de `metrics.json`

```json
{
  "experiment_id": "mnist_0_vs_1",
  "task": "binary_classification",
  "classes": ["0", "1"],
  "train_samples": 0,
  "validation_samples": 0,
  "test_samples": 0,
  "epochs_or_iterations_completed": 0,
  "best_epoch_or_iteration": 0,
  "parameters": 0,
  "training_seconds": 0.0,
  "best_validation_accuracy": 0.0,
  "test_accuracy": 0.0,
  "test_precision": 0.0,
  "test_recall": 0.0,
  "test_f1": 0.0,
  "test_macro_precision": 0.0,
  "test_macro_recall": 0.0,
  "test_macro_f1": 0.0,
  "test_balanced_accuracy": 0.0,
  "test_specificity": null
}
```

- Usa `null` cuando una métrica no aplique.
- No escribas cifras manualmente.
- Genera JSON desde las variables reales de la ejecución.
- Los README deben tomar sus resultados de estos archivos, no de valores copiados.

---

# 12. Requisitos específicos: módulo 2

## 12.1 `1_manual_neural_network_model.ipynb`
### Nombre desarrollado: `01_neurona_manual_numpy_mnist.ipynb`

Propósito: explicar una neurona o clasificador lineal binario implementado manualmente con NumPy sobre MNIST.

Acciones obligatorias:

- Mantén el enfoque manual con NumPy.
- Explica que el modelo equivale conceptualmente a regresión logística o una neurona binaria.
- Conserva los experimentos:
  - MNIST `0 vs 1`;
  - MNIST `5 vs 6`.
- Añade train, validation y test para cada experimento.
- Usa validation para comparar learning rates o número de iteraciones.
- Elimina imports no utilizados.
- Revisa y corrige estabilidad numérica:
  - evita `log(0)`;
  - aplica clipping o formulación estable;
  - controla overflow de la sigmoide.
- Revisa la consistencia de dimensiones de `X`, `Y`, `w` y `b`.
- Elimina o convierte en comparación explícita las celdas duplicadas de entrenamiento.
- No entrenes dos veces sin explicar qué hiperparámetro se compara.
- Corrige la nomenclatura y el cálculo de TP, TN, FP y FN.
- Define claramente cuál clase se considera positiva.
- Valida las métricas con `sklearn.metrics`.
- Genera:
  - curva de coste;
  - representación visual de pesos;
  - matriz de confusión;
  - métricas binarias;
  - ejemplos de errores.
- No uses el conjunto test para escoger learning rate.
- Sustituye instalaciones `!pip install` por documentación de entorno.
- Registra los dos experimentos por separado.
- Explica por qué `5 vs 6` es más difícil que `0 vs 1`.
- No presentes este notebook como una red profunda.

Resultado histórico detectado: existen outputs previos cercanos al 97 % en uno de los experimentos. Trátalos solo como referencia histórica; vuelve a ejecutar y regenera todas las métricas.

---

## 12.2 `1_red_neuronal_desde_cero.ipynb`
### Nombre desarrollado: `02_red_neuronal_desde_cero_pytorch.ipynb`

Propósito: mostrar explícitamente forward, loss, backward y update usando tensores PyTorch, y luego comparar una neurona con un MLP sencillo.

Acciones obligatorias:

- Conserva la progresión pedagógica:
  - sigmoide;
  - neurona binaria;
  - `0 vs 1`;
  - `5 vs 6`;
  - comparación con MLP.
- Explica con precisión que:
  - los parámetros y el bucle son explícitos;
  - los gradientes se obtienen con autograd;
  - la sección MLP utiliza componentes de alto nivel de PyTorch.
- No afirmes que todo el cálculo diferencial está implementado manualmente si se usa autograd.
- Crea validation desde train para ambos pares de clases.
- Usa validation para elegir learning rate e iteraciones.
- Reserva test para la evaluación final.
- Evita cargar todo innecesariamente en GPU si afecta memoria; documenta la decisión.
- Usa `BCEWithLogitsLoss` cuando corresponda para mayor estabilidad.
- Asegura que las etiquetas tengan el dtype correcto.
- Registra por separado:
  - neurona `0 vs 1`;
  - neurona `5 vs 6`;
  - MLP `5 vs 6`.
- Compara número de parámetros, accuracy, F1 y tiempo.
- Genera curvas, matrices de confusión y errores.
- Usa rutas compartidas de datasets.
- Mantén compatibilidad CPU, CUDA y MPS.
- Añade una conclusión sobre cuándo una frontera lineal resulta insuficiente.

---

# 13. Requisitos específicos: módulo 3

## 13.1 `1_red_neuronal_multicapa.ipynb`
### Nombre desarrollado: `01_mlp_mnist_pytorch.ipynb`

- Conserva el enfoque MLP multiclase.
- Crea validation desde train.
- No consultes test en cada época.
- Usa validation para checkpoint y selección de época.
- Explica:
  - Flatten;
  - capas densas;
  - ReLU;
  - logits;
  - softmax conceptual;
  - CrossEntropyLoss.
- No apliques softmax antes de `CrossEntropyLoss`.
- Registra parámetros y tiempo.
- Genera métricas por clase.
- Compara MLP frente a CNN respecto de estructura espacial.
- Regenera los resultados.
- La cifra histórica aproximada de 97,62 % no debe copiarse como resultado final.

---

## 13.2 `2_convolutional_neural_network_model.ipynb`
### Nombre desarrollado: `02_cnn_mnist_keras.ipynb`

- Traduce títulos y comentarios relevantes al español.
- Corrige nombres inconsistentes de modelo y checkpoint.
- Usa un único identificador de experimento.
- Evita entrenar arquitecturas diferentes sin una comparación A/B explícita.
- Usa un validation split razonable y reproducible.
- Incorpora:
  - EarlyStopping;
  - ModelCheckpoint;
  - ReduceLROnPlateau.
- Asegura que todos monitoricen una métrica coherente.
- Restaura los mejores pesos.
- No evalúes test hasta el final.
- Guarda el modelo o checkpoint fuera de Git si supera la política de tamaño.
- Genera métricas por clase y ejemplos de error.
- Regenera resultados.
- La cifra histórica aproximada de 97,37 % no debe copiarse como resultado final.

---

## 13.3 `2_red_convolucional_mnist.ipynb`
### Nombre desarrollado: `03_cnn_mnist_pytorch_base.ipynb`

- Conserva una CNN base, pequeña y pedagógica.
- Explica:
  - Conv2D;
  - filtros;
  - kernels;
  - padding;
  - ReLU;
  - pooling;
  - flatten;
  - capa lineal.
- Mantén train, validation y test.
- Usa validation para guardar el mejor checkpoint.
- Evalúa test una sola vez al final.
- Registra curvas completas de train y validation.
- Añade classification report, matriz de confusión y errores.
- Regenera resultados.
- La cifra histórica aproximada de 98,00 % es solo referencia.

---

## 13.4 `3_convolutional_neural_network_model_animals.ipynb`
### Nombre desarrollado: `04_cnn_cifar10_keras.ipynb`

- Renombra el entregable: CIFAR-10 no contiene únicamente animales.
- Explica las diez clases:
  - avión;
  - automóvil;
  - pájaro;
  - gato;
  - ciervo;
  - perro;
  - rana;
  - caballo;
  - barco;
  - camión.
- Elimina celdas duplicadas de entrenamiento.
- Traduce la documentación relevante al español.
- Normaliza las imágenes correctamente.
- Aplica augmentation solo a train si se utiliza.
- Usa callbacks coherentes.
- No selecciones modelo con test.
- Genera métricas por clase.
- Analiza confusiones como gato/perro, automóvil/camión o avión/barco.
- Regenera resultados.
- La cifra histórica aproximada de 80,84 % es solo referencia.

---

## 13.5 `3_red_convolucional_mnist_mejorada.ipynb`
### Nombre desarrollado: `05_cnn_mnist_pytorch_mejorada.ipynb`

- Conserva:
  - mayor profundidad;
  - BatchNorm;
  - Dropout;
  - data augmentation;
  - scheduler;
  - early stopping.
- Aplica augmentation solo a train.
- Validation y test deben usar transformaciones deterministas.
- Usa validation para scheduler, checkpoint y early stopping.
- Evalúa test solo al final.
- Compara con:
  - MLP;
  - CNN base.
- No atribuyas toda mejora a una sola técnica.
- Regenera resultados reproducibles.
- La cifra histórica aproximada de 99,60 % es solo referencia.

---

## 13.6 `4_red_convolucional_cifar10.ipynb`
### Nombre desarrollado: `06_cnn_cifar10_pytorch.ipynb`

- Corrige la fuga metodológica.
- Divide el train original en train y validation.
- No evalúes test por época.
- No selecciones checkpoint con test.
- Usa validation para:
  - scheduler;
  - early stopping;
  - checkpoint.
- Evalúa test una única vez al final.
- Aplica augmentation solo a train.
- Mantén transformaciones deterministas en validation y test.
- Configura `num_workers` de forma segura para Windows.
- Usa `pin_memory` solo cuando tenga sentido.
- Protege la ejecución multiproceso si es necesario.
- Documenta tiempos esperados en CPU y GPU.
- Genera métricas y errores por clase.
- El notebook original no contiene resultados finales ejecutados confiables.
- Si no se completa full, usa `FULL_RUN_PENDING`.

---

# 14. Dependencias y entornos

Antes de modificar dependencias:

1. Inspecciona los archivos existentes.
2. No sobrescribas el `requirements.txt` raíz.
3. No actualices masivamente paquetes.
4. No rompas el módulo 1.
5. Propón una estrategia compatible con el repositorio.

Estructura sugerida:

```text
tema-02-machine-learning/requirements/
├── common.txt
├── pytorch.txt
└── tensorflow.txt
```

## Dependencias comunes esperables

```text
jupyter
ipykernel
nbclient
nbformat
numpy
pandas
matplotlib
scikit-learn
```

## PyTorch

```text
torch
torchvision
```

## TensorFlow/Keras

```text
tensorflow
keras
```

No fijes versiones al azar. Registra las versiones que realmente funcionen en el entorno.

## Kernels

Documenta la creación o selección de kernels. No crees múltiples entornos sin necesidad.

## Instalaciones dentro de notebooks

- Evita `!pip install` ejecutado automáticamente.
- Usa `%pip` solo en una celda opcional claramente marcada, si es imprescindible.
- La forma principal de instalación debe estar en README y requirements.

---

# 15. Datasets, checkpoints y archivos grandes

## Datasets

- Reutiliza `datasets/` de la raíz si esa es la convención existente.
- Usa rutas relativas calculadas con `pathlib`.
- No dupliques MNIST o CIFAR-10 entre notebooks.
- No subas datasets a Git.
- Documenta que la primera ejecución puede descargarlos.

## Checkpoints

- Guarda checkpoints en una carpeta local coherente.
- No los mezcles con los resultados publicables.
- No subas archivos grandes al Git normal.
- Considera Git LFS solo con autorización.
- No es obligatorio versionar pesos para considerar completado un notebook.

## `.gitignore`

- Inspecciona primero el archivo existente.
- No lo reemplaces.
- Añade solo reglas necesarias.
- Verifica que excluya:
  - `.venv`;
  - cachés;
  - datasets;
  - checkpoints;
  - modelos;
  - secretos;
  - archivos temporales.

---

# 16. Documentación

## README general del Tema 2

Actualízalo sin destruir contenido útil.

Debe incluir:

- propósito del Tema 2;
- mapa de los cinco módulos;
- estado de cada módulo;
- inventario de notebooks;
- frameworks;
- datasets;
- instalación;
- kernels;
- ejecución fast y full;
- política train/validation/test;
- tabla comparativa de resultados;
- limitaciones;
- relación entre ML, Deep Learning, CNN, Transformers y LLMs;
- relación conceptual con el TFM MDS AI Operations Center.

## README del módulo 2

Debe incluir:

- neurona manual con NumPy;
- PyTorch y autograd;
- clasificación binaria;
- forward, loss, backward y update;
- comparación neurona frente a MLP;
- resultados por experimento.

## README del módulo 3

Debe incluir:

- MLP;
- CNN;
- MNIST;
- CIFAR-10;
- PyTorch;
- TensorFlow/Keras;
- regularización;
- augmentation;
- callbacks;
- comparación de resultados.

## README por notebook o por carpeta de resultados

Debe incluir:

- propósito;
- origen;
- cambios realizados;
- dataset;
- arquitectura;
- hiperparámetros;
- comando de ejecución;
- hardware;
- tiempo;
- resultados;
- interpretación;
- limitaciones;
- aplicación empresarial.

---

# 17. Automatización

Después de estabilizar manualmente los notebooks, crea o adapta scripts.

## `execute_notebooks.py`

Debe:

- ejecutar uno, un módulo o todos;
- aceptar `--mode fast|full`;
- aceptar timeout configurable;
- ejecutar desde kernel limpio;
- guardar `notebook_executed.ipynb`;
- detenerse con código distinto de cero ante errores;
- no sobrescribir resultados full con fast.

## `validate_notebooks.py`

Debe validar:

- JSON válido;
- ausencia de outputs con error;
- celdas requeridas;
- existencia de artefactos;
- esquema de JSON;
- `run_mode`;
- estado publishable;
- ausencia de rutas absolutas evidentes;
- ausencia de secretos comunes;
- ausencia de archivos demasiado grandes;
- documentación de train/validation/test.

## `collect_environment.py`

Debe guardar:

- sistema operativo;
- versión de Python;
- versión de librerías;
- dispositivo;
- disponibilidad de CUDA/MPS;
- información relevante reproducible.

## `summarize_results.py`

Debe:

- leer los JSON;
- construir la tabla comparativa;
- evitar resultados escritos manualmente;
- distinguir fast, full y pendientes.

## CI opcional

Si el repositorio ya usa GitHub Actions o se autoriza:

- ejecuta solo smoke tests fast;
- no entrenes los ocho modelos completos;
- valida estructura y notebooks;
- no descargues artefactos innecesarios.

---

# 18. Validaciones obligatorias

Antes de considerar terminado un notebook:

1. Reinicia el kernel.
2. Ejecuta todas las celdas en orden.
3. Confirma cero excepciones.
4. Confirma que test se usa solo al final.
5. Confirma que augmentation no afecta validation/test.
6. Confirma que checkpoint usa validation.
7. Confirma que métricas JSON coinciden con outputs.
8. Confirma que figuras fueron regeneradas.
9. Confirma que el notebook ejecutado tiene outputs.
10. Confirma que no contiene rutas absolutas del desarrollador.
11. Confirma que no contiene secretos.
12. Confirma que no versiona datasets.
13. Confirma que no versiona pesos grandes.
14. Ejecuta el validador.
15. Ejecuta:

```bash
git diff --check
git status --short
```

16. Verifica que el módulo 1 siga intacto.
17. Revisa el tamaño de los archivos antes del commit.

---

# 19. Plan de commits actualizado para ocho notebooks

No combines todo en un único commit.

Secuencia recomendada:

```text
chore(t2): scaffold modules 2 and 3 without modifying module 1

feat(t2-m2-nb01): document and execute manual NumPy neuron on MNIST

feat(t2-m2-nb02): document and execute PyTorch neural network from first principles

feat(t2-m3-nb01): fix and execute PyTorch MNIST MLP

feat(t2-m3-nb02): fix and execute Keras MNIST CNN

feat(t2-m3-nb03): document and execute baseline PyTorch MNIST CNN

feat(t2-m3-nb04): fix and execute Keras CIFAR-10 CNN

feat(t2-m3-nb05): document and execute improved PyTorch MNIST CNN

feat(t2-m3-nb06): fix validation split and execute PyTorch CIFAR-10 CNN

test(t2): add notebook execution and validation tooling

docs(t2): publish modular index and comparative results
```

## Antes de cada commit

- Ejecuta validaciones aplicables.
- Revisa `git diff`.
- Revisa `git diff --stat`.
- Confirma que el módulo 1 no fue modificado.
- Confirma que no hay datasets, secretos o pesos grandes.
- Usa `git add` selectivo.
- No uses `git add .` sin revisar.

## Push

No hagas push automáticamente.

Cuando todo esté aprobado, muestra:

```bash
git status
git log --oneline --decorate -15
git remote -v
```

Y espera autorización.

---

# 20. Restricciones operativas

- Limita los cambios a `tema-02-machine-learning`.
- Solicita autorización para modificar:
  - README raíz;
  - requirements raíz;
  - scripts compartidos;
  - `.gitignore` raíz.
- No modifiques `tema-01-entorno`.
- No desarrolles módulos 4 o 5.
- No borres resultados existentes.
- No reemplaces el trabajo del módulo 1.
- No inventes métricas.
- No publiques resultados fast como finales.
- No consideres completado un notebook con `FULL_RUN_PENDING`.
- No hagas push.
- No ejecutes comandos destructivos.
- No ocultes errores de ejecución.
- No continúes automáticamente tras la auditoría inicial.

---

# 21. Definición de terminado

El trabajo se considera terminado cuando:

- el módulo 1 permanece intacto;
- existen exactamente dos notebooks desarrollados en el módulo 2;
- existen exactamente seis notebooks desarrollados en el módulo 3;
- los ocho originales están protegidos;
- los ocho notebooks ejecutan correctamente en modo fast;
- los resultados publicados proceden de modo full;
- cualquier full pendiente está claramente marcado y no publicado;
- train, validation y test están correctamente separados;
- test se usa solo al final;
- cada notebook tiene resultados ordenados;
- cada notebook tiene métricas generadas por código;
- cada notebook tiene documentación;
- los README reflejan los cinco módulos;
- módulos 4 y 5 aparecen como pendientes;
- las validaciones pasan;
- los commits están separados;
- el módulo 1 no tiene cambios accidentales;
- no hay datasets, secretos ni pesos grandes versionados;
- no se ha realizado push sin autorización.

---

# 22. Respuesta esperada en la primera interacción

En la primera interacción:

- no modifiques archivos;
- no entrenes;
- no instales;
- no crees carpetas;
- no hagas commits.

Responde únicamente con:

1. estructura encontrada;
2. módulo 1 identificado;
3. ubicación de los ocho notebooks;
4. estado Git;
5. auditoría individual de los ocho notebooks;
6. inconsistencias detectadas;
7. estructura objetivo adaptada;
8. estrategia para proteger originales;
9. estrategia de dependencias;
10. estrategia de datasets y checkpoints;
11. riesgos;
12. estimación de esfuerzo y orden de ejecución;
13. plan de validación;
14. plan de commits;
15. preguntas o bloqueos.

Después detente y espera aprobación.

---

# 23. Instrucción inicial para pegar en Codex

```text
Lee completamente AGENTS.md e INSTRUCCION_CODEX_TEMA2_MODULAR.md.

Esta instrucción consolidada reemplaza las versiones anteriores para el Tema 2.

El repositorio ya existe. No ejecutes git init.

El Tema 2 está dividido en cinco módulos. El módulo 1 ya fue desarrollado
y debe permanecer intacto.

Debes trabajar con ocho notebooks:
- dos en el módulo 2;
- seis en el módulo 3.

Ejecuta únicamente la fase de auditoría descrita en la instrucción.

En esta primera interacción:
- no modifiques archivos;
- no instales dependencias;
- no entrenes modelos;
- no crees carpetas;
- no hagas commits;
- no hagas push.

Entrega el informe solicitado y detente hasta recibir aprobación.
```

---

# 24. Segunda instrucción, después de aprobar la auditoría

No la ejecutes hasta que el usuario apruebe explícitamente el informe.

```text
Apruebo la auditoría y la estructura propuesta.

Implementa primero el scaffolding y el módulo 2.
Protege los notebooks originales.
No modifiques el módulo 1.
No trabajes todavía en el módulo 3.

Desarrolla los dos notebooks del módulo 2 uno por uno.

Para cada notebook:
1. muestra los cambios propuestos;
2. ejecuta modo fast;
3. ejecuta las validaciones;
4. muestra git diff y resultados;
5. espera aprobación antes del commit;
6. no hagas push.
```
