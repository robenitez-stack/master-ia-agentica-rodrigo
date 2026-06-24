# Módulo 5 — Mini-GPT desde cero con PyTorch

Proyecto académico modular que implementa un modelo de lenguaje decoder-only desde cero. El objetivo es comprender tokenización, embeddings, atención causal, entrenamiento autoregresivo y generación, no competir con GPT-4.

## Requisitos

- Python 3.11 o superior.
- CPU; CUDA o MPS son opcionales.
- Espacio adicional si se descarga Don Quijote.

## Instalación

Desde esta carpeta:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Estructura

```text
.
├── data/.gitkeep
├── docs/
│   ├── 1 - Diapositivas_Tema2_Clase5.pdf
│   └── Resumen Clase 4-5 Tema 2.pdf
├── notebooks/
│   ├── mini-gpt-desde-cero.ipynb
│   └── mini-gpt-desde-cero.original.ipynb
├── src/minigpt/
│   ├── config.py
│   ├── device.py
│   ├── corpus.py
│   ├── tokenizer.py
│   ├── data.py
│   ├── model.py
│   ├── training.py
│   ├── generation.py
│   ├── checkpoint.py
│   ├── visualization.py
│   └── cli.py
└── tests/
```

El notebook original queda conservado sin cambios. El notebook canónico importa la implementación modular.

## Arquitectura

```text
Texto
  │
  ▼
CharacterTokenizer
  │ IDs
  ▼
Token Embedding + Positional Embedding
  │
  ▼
┌─────────────────────────────┐
│ Transformer Block × N       │
│  ├─ LayerNorm               │
│  ├─ Masked Multi-Head Attn  │
│  ├─ Residual                │
│  ├─ LayerNorm               │
│  ├─ Feed Forward            │
│  └─ Residual                │
└─────────────────────────────┘
  │
  ▼
LayerNorm → Linear Head → Logits
  │
  ▼
Probabilidad del siguiente carácter
```

No se usa `torch.nn.Transformer`: Q, K, V, scores escalados, softmax, máscara causal, cabezas, residuales y bloques están implementados explícitamente.

## Corpus

Descarga Don Quijote una sola vez:

```powershell
python -m minigpt.cli download --output data/quijote.txt
```

Si la red, el proxy o el certificado bloquean la descarga, copia manualmente cualquier `.txt` UTF-8 a esa ruta. Los corpus completos no se versionan. Las pruebas usan texto sintético y no requieren Internet.

## Entrenamiento

```powershell
python -m minigpt.cli train `
  --corpus data/quijote.txt `
  --profile demo `
  --output-dir runs/quijote-demo `
  --device auto
```

Perfiles:

- `academic`: arquitectura 256/4/4, contexto 128 y 5000 iteraciones.
- `demo`: misma arquitectura y 500 iteraciones.
- `cpu_light`: arquitectura 128/4/3, contexto 64 y 2000 iteraciones.

El dispositivo se selecciona como CUDA > MPS > CPU. Puede forzarse con `--device`. La partición es determinista: 90 % entrenamiento y 10 % validación.

La tarea es auto-supervisada:

```python
x = tokens[i : i + block_size]
y = tokens[i + 1 : i + block_size + 1]
```

AdamW minimiza cross-entropy. Esta pérdida penaliza la probabilidad insuficiente asignada al carácter correcto. Antes de entrenar se muestra `log(vocab_size)` como referencia aproximada, no como prueba estricta.

## Checkpoints y reanudación

Cada run guarda:

- `best.pt`: mejor pérdida de validación.
- `last.pt`: último estado.
- `history.json`: iteración, train/val loss, learning rate y tiempos.

Los checkpoints contienen pesos, optimizador, configuraciones, vocabulario, iteración, mejor val loss, historial, semilla y dispositivo original.

```powershell
python -m minigpt.cli train `
  --corpus data/quijote.txt `
  --profile demo `
  --output-dir runs/quijote-demo `
  --resume runs/quijote-demo/last.pt
```

## Generación e inspección

Top-k:

```powershell
python -m minigpt.cli generate `
  --checkpoint runs/quijote-demo/best.pt `
  --prompt "En un lugar de la Mancha" `
  --max-new-tokens 300 `
  --temperature 0.8 `
  --top-k 40
```

Top-p:

```powershell
python -m minigpt.cli generate `
  --checkpoint runs/quijote-demo/best.pt `
  --prompt "En un lugar" `
  --max-new-tokens 300 `
  --temperature 0.9 `
  --top-p 0.9
```

```powershell
python -m minigpt.cli inspect --checkpoint runs/quijote-demo/best.pt
```

La inferencia recorta el contexto a `block_size` y genera token por token. Greedy toma siempre el máximo. Temperature baja concentra la distribución; temperature alta la aplana. Top-k conserva los `k` candidatos más probables. Top-p conserva el conjunto mínimo cuya probabilidad acumulada alcanza `p`.

## Conceptos fundamentales

### Tokenización y BPE

`CharacterTokenizer` usa un ID por carácter y conserva exactamente el texto conocido. Es transparente pero produce secuencias largas. BPE fusiona fragmentos frecuentes y reduce longitud a cambio de un vocabulario mayor; queda documentado como mejora futura, no reemplaza la implementación principal.

### Embeddings

Los embeddings de token convierten IDs en vectores aprendidos. Los embeddings posicionales aportan el orden, ya que la atención por sí sola no conoce posiciones.

### Atención, Q, K y V

Cada token genera:

- Query: qué información busca.
- Key: qué información ofrece.
- Value: contenido que se combinará.

La similitud se calcula con `Q @ Kᵀ / sqrt(head_size)`. La máscara causal pone `-inf` sobre posiciones futuras antes del softmax. Varias cabezas aprenden relaciones distintas y sus salidas se concatenan.

### Transformer decoder-only

Cada bloque usa LayerNorm pre-norm, masked multi-head attention, conexión residual, otra LayerNorm, feed-forward con ReLU y otra residual. La cabeza lineal final produce logits sobre el vocabulario.

### Entrenamiento frente a inferencia

Durante entrenamiento se conocen los targets, se calcula cross-entropy y se actualizan pesos mediante gradientes. Durante inferencia no hay targets ni actualización: el modelo reutiliza sus pesos y agrega un carácter muestreado en cada paso.

## Visualizaciones

El módulo ofrece funciones opcionales para:

- train loss frente a val loss;
- atención por capa y cabeza;
- distribución del siguiente token;
- comparación de temperaturas;
- vecinos de embeddings por similitud coseno.

No son necesarias para entrenar o generar.

## Tests y calidad

```powershell
python -m pytest -q
python -m minigpt.cli --help
python -m minigpt.cli train --help
python -m minigpt.cli generate --help
ruff check .
ruff format --check .
```

La suite valida tokenización, batches, causalidad, shapes, pérdida, backward, sampling, checkpoints y un entrenamiento pequeño offline en CPU.

Estado verificado en Python 3.11 y CPU:

- 16 pruebas aprobadas.
- Ruff lint y format aprobados.
- Notebook modular ejecutado de principio a fin.
- CLI de entrenamiento, reanudación, inspección y generación comprobada.
- Smoke CLI: modelo de 609.308 parámetros, 3 iteraciones iniciales y una iteración reanudada.

## Reproducibilidad y limitaciones

- Se fijan semillas de Python, PyTorch y CUDA.
- Los resultados exactos pueden variar entre CPU, CUDA y MPS.
- El tokenizador rechaza caracteres fuera del vocabulario.
- El contexto, corpus y número de parámetros son pequeños.
- No hay preentrenamiento masivo, instruction tuning, RLHF/DPO ni alineamiento.
- No hay APIs externas, modelos preentrenados, LangChain o servicios cloud.

Comparte con GPT-4 la familia decoder-only, la atención causal, el objetivo de siguiente token y la generación autoregresiva. No comparte escala, datos, infraestructura, alineamiento ni capacidades.
