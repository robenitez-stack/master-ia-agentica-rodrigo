# Demo · Tema 2 · Sesión 4 · NLP

Notebook que da soporte a la sesión 4 del Tema 2 del Máster. Sirve dos demos durante la clase:

- **Mini-demo en el slide 9** — solo la sección 1 (tokenización con `tiktoken`). 3 minutos.
- **Demo de cierre en el slide 25** — secciones 2, 3 y 4 (embeddings, aritmética semántica, visualización). 10 minutos.

## Qué hay en el notebook

1. **Tokenización con `tiktoken`** — cómo GPT-4 trocea texto en subwords. Comparativa español-inglés, casos divertidos (emojis, palabras técnicas).
2. **Embeddings de frase con `sentence-transformers`** — modelo MiniLM-L6-v2 (~80 MB), matriz de similitud coseno, heatmap.
3. **Aritmética semántica con GloVe** — la analogía clásica rey−hombre+mujer ≈ reina, y otras (capitales, plurales, verbos).
4. **Visualización 2D con PCA** — proyectar embeddings de 100 dimensiones a 2D y ver los clusters semánticos en un plot.

## Requisitos

- Python 3.10+
- ~300 MB libres en disco (modelos cacheados después de la primera descarga).
- 4 GB de RAM libres.

Funciona en cualquier plataforma. Apple Silicon va acelerado con MPS.

## Setup

Desde esta carpeta:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pip check
jupyter lab nlp-fundamentos.ipynb
```

En Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pip check
jupyter lab nlp-fundamentos.ipynb
```

El notebook no instala paquetes desde sus celdas. Comprueba que el kernel seleccionado corresponde al mismo entorno `.venv`.

## ⚠️ Importante: ejecutar antes de clase

**Ejecuta el notebook entero una vez antes de impartir la sesión.** La primera ejecución descarga aproximadamente 200 MB de modelos (`all-MiniLM-L6-v2` y `glove-wiki-gigaword-100`). Las ejecuciones posteriores reutilizan las cachés estándar de Hugging Face y Gensim.

Si lo descargas en clase con WiFi mala, esperarás minutos en silencio incómodo.

## Tiempo de ejecución (segunda ejecución, modelos en cache)

- Sección 1 (tokenización) — 3 s
- Sección 2 (embeddings + heatmap) — 5 s
- Sección 3 (analogías GloVe) — 8 s (la carga es lo que tarda)
- Sección 4 (PCA + plot) — 2 s

**Total: ~20 segundos de ejecución pura. El resto es narración.**

## Solución de problemas

- **`ModuleNotFoundError: sentence_transformers`** — falta instalar dependencias: `pip install -r requirements.txt`.
- **GloVe no se descarga** — comprueba conexión, o ejecútalo en un terminal fuera del notebook: `python -c "import gensim.downloader as api; api.load('glove-wiki-gigaword-100')"`.
- **Hugging Face está bloqueado por proxy o certificado** — configura el proxy/certificado corporativo antes de precargar `all-MiniLM-L6-v2`; no borres una caché válida.
- **El heatmap se ve negro** — actualiza `matplotlib >= 3.8` o cambia `cmap="viridis"` por `cmap="Blues"`.
- **Una palabra no existe en GloVe** — usa una alternativa inglesa presente en el vocabulario; el notebook identifica las palabras ausentes.
- **Una analogía no devuelve el término esperado** — es un resultado aproximado y dependiente del corpus, no una regla lógica. Revisa también los cinco vecinos mostrados.
- **`jupyter` no se reconoce** — activa `.venv` y vuelve a ejecutar `pip install -r requirements.txt`.
- **El notebook usa otro kernel** — selecciona el intérprete de `.venv` desde Jupyter y reinicia el kernel.

### Precarga para una sesión sin conexión

Con conexión disponible, abre y ejecuta el notebook completo una vez usando el mismo usuario y entorno que se empleará en clase. No copies modelos al repositorio: conserva las cachés estándar y verifica después una segunda ejecución.

## Conexión conceptual

Las diapositivas continúan desde embeddings hacia RNN, LSTM, atención y Transformers. Este notebook se limita deliberadamente a los cuatro fundamentos reproducibles anteriores; no entrena ni implementa esos modelos.

## Ampliaciones opcionales (fuera de esta demo)

Si quieres profundizar, en una segunda iteración se podría añadir:

- **Visualización de atención** con `bertviz` sobre un BERT pequeño. Muy visual.
- **Embeddings en español** con modelos como `paraphrase-multilingual-MiniLM-L12-v2` o `BAAI/bge-m3`.
- **Comparativa MiniLM vs un encoder más grande** (e.g. `all-mpnet-base-v2`) — para ver el tradeoff calidad/tamaño.
