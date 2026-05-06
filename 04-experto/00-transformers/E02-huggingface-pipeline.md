# E02: Hugging Face Pipeline — NLP en 3 Líneas

## Objetivo
Dominar la API `pipeline()` de Hugging Face para resolver tareas de NLP aplicadas al comercio electrónico: clasificación de reseñas, extracción de entidades, respuesta a preguntas, resumen y traducción de catálogos.

---

## 1. Fundamentos Teóricos

### 1.1 ¿Qué es `pipeline()`?

`pipeline()` es la abstracción de más alto nivel en la biblioteca `transformers`. Conecta automáticamente:
1. **Tokenizer**: Convierte texto a IDs de tokens.
2. **Modelo**: Red neuronal pre-entrenada.
3. **Post-procesamiento**: Convierte logits a etiquetas legibles.

### 1.2 Tareas Soportadas

| Pipeline Tag | Modelo típico | Aplicación en ventas |
|---|---|---|
| `sentiment-analysis` | BERT, RoBERTa | Clasificar reseñas (+/-) |
| `zero-shot-classification` | BART, NLI | Categorizar sin entrenar |
| `text-generation` | GPT-2, Llama | Descripciones automáticas |
| `fill-mask` | BERT, RoBERTa | Completar descripciones |
| `ner` | BERT, DistilBERT | Extraer marcas, precios |
| `question-answering` | BERT, RoBERTa | FAQ de productos |
| `summarization` | BART, T5 | Resumir reseñas largas |
| `translation` | T5, mBART | Traducir catálogos |
| `feature-extraction` | BERT | Embeddings para búsqueda |

### 1.3 Formatos de Entrada

- **Texto único**: `pipeline(tarea)("texto")`
- **Lista de textos**: `pipeline(tarea)(lista, batch_size=N)`
- **Pares de texto**: `question-answering` usa dict `{"question": ..., "context": ...}`

### 1.4 Parámetros Clave

- `model`: Nombre del modelo en Hugging Face Hub
- `tokenizer`: Tokenizador específico (por defecto usa el del modelo)
- `device`: `0` para GPU, `-1` para CPU
- `batch_size`: Procesar múltiples textos en lote
- `truncation`: `True` para truncar textos largos
- `max_length`: Longitud máxima de tokens
- `padding`: `True` para padding automático

---

## 2. Ejemplos Prácticos

### Ejemplo 1: Sentiment Analysis — Clasificar Reseña

```python
from transformers import pipeline

clasificador = pipeline("sentiment-analysis")

reseña = "Este producto es excelente, superó todas mis expectativas. Muy recomendado."
resultado = clasificador(reseña)
print(resultado)
# [{'label': 'POSITIVE', 'score': 0.9998}]

reseña_mala = "Llegó roto y el servicio al cliente fue pésimo."
resultado = clasificador(reseña_mala)
print(resultado)
# [{'label': 'NEGATIVE', 'score': 0.9995}]

print("✅ pipeline('sentiment-analysis') clasifica reseñas automáticamente")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: Sentiment Analysis — Clasificar Reseña.*

1. [{'label': 'POSITIVE', 'score': 0.9998}]
2. [{'label': 'NEGATIVE', 'score': 0.9995}]

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: Zero-Shot Classification — Categorizar Producto

```python
from transformers import pipeline

clasificador_zsl = pipeline("zero-shot-classification",
                            model="facebook/bart-large-mnli")

descripcion = "Laptop gaming con RTX 3060, 16GB RAM y SSD 512GB"
candidatos = ["electrónica", "mueble", "ropa", "alimentos", "juguetes"]

resultado = clasificador_zsl(descripcion, candidate_labels=candidatos)
print(f"Producto: {descripcion}")
print(f"Categoría: {resultado['labels'][0]} (confianza: {resultado['scores'][0]:.3f})")
print(f"Todas las categorías: {dict(zip(resultado['labels'], resultado['scores']))}")
# Categoría: electrónica (confianza: 0.987)

print("✅ Zero-shot: clasificar sin entrenamiento previo")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: Zero-Shot Classification — Categorizar Producto.*

1. Categoría: electrónica (confianza: 0.987)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: Text Generation — Generar Descripción

```python
from transformers import pipeline

generador = pipeline("text-generation", model="gpt2")

prompt = "El nuevo monitor ultrawide de 49 pulgadas ofrece"
resultado = generador(prompt, max_new_tokens=50, num_return_sequences=1)
print(resultado[0]['generated_text'])
# "El nuevo monitor ultrawide de 49 pulgadas ofrece una experiencia inmersiva..."

print("\n✅ pipeline('text-generation') crea descripciones automáticas")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: Text Generation — Generar Descripción.*

1. "El nuevo monitor ultrawide de 49 pulgadas ofrece una experiencia inmersiva..."

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: Fill-Mask — Completar Descripción

```python
from transformers import pipeline

llenador = pipeline("fill-mask", model="bert-base-multilingual-cased")

frase = "Este [MASK] es excelente para oficina y trabajo remoto."
resultados = llenador(frase)

print(f"Frase: {frase}")
for r in resultados:
    print(f"  Token: '{r['token_str']}', score: {r['score']:.4f}")
# Token: 'producto', score: 0.1234
# Token: 'equipo', score: 0.0987
# Token: 'monitor', score: 0.0765

print("\n✅ Fill-mask sugiere palabras para completar descripciones")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: Fill-Mask — Completar Descripción.*

1. Token: 'producto', score: 0.1234
2. Token: 'equipo', score: 0.0987
3. Token: 'monitor', score: 0.0765

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: Named Entity Recognition — Extraer Marcas y Productos

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

texto = "La nueva laptop Dell XPS 15 cuesta $1,499 en Amazon y tiene garantía de 2 años."
entidades = ner(texto)

print("Entidades encontradas:")
for e in entidades:
    print(f"  {e['entity_group']}: '{e['word']}' (confianza: {e['score']:.3f})")
# ORG: 'Dell' (0.999)
# MISC: 'XPS 15' (0.995)
# MISC: 'Amazon' (0.998)

print("\n✅ NER extrae marcas, productos y organizaciones de textos")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: Named Entity Recognition — Extraer Marcas y Productos.*

1. ORG: 'Dell' (0.999)
2. MISC: 'XPS 15' (0.995)
3. MISC: 'Amazon' (0.998)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: Question Answering — Responder sobre Productos

```python
from transformers import pipeline

qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

contexto = """
El monitor LG Ultragear 27" tiene resolución 4K, tasa de refresco 144Hz,
tiempo de respuesta 1ms, panel IPS y precio de $449. Incluye garantía de 3 años.
Viene con cable HDMI 2.1 y soporte VESA.
"""

preguntas = [
    "¿Cuál es el precio del monitor?",
    "¿Qué tamaño tiene la pantalla?",
    "¿Cuántos años de garantía incluye?"
]

for pregunta in preguntas:
    respuesta = qa(question=pregunta, context=contexto)
    print(f"P: {pregunta}")
    print(f"R: {respuesta['answer']} (confianza: {respuesta['score']:.3f})\n")

print("✅ Question-Answering responde preguntas sobre productos")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: Question Answering — Responder sobre Productos.*

1. `from transformers import pipeline` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: Summarization — Resumir Reseña Larga

```python
from transformers import pipeline

resumidor = pipeline("summarization", model="facebook/bart-large-cnn")

reseña_larga = """
Compré este monitor la semana pasada y estoy muy impresionado con la calidad de imagen.
Los colores son vibrantes y precisos, perfectos para edición de fotos. La tasa de
refresco de 144Hz hace que los juegos se vean fluidos. El único inconveniente es que
los parlantes integrados son regulares. El menú OSD es intuitivo y fácil de navegar.
La construcción es sólida, aunque la base ocupa bastante espacio. Por el precio,
considero que es una excelente opción para gaming y trabajo creativo. Lo recomendaría
a cualquiera que busque un monitor versátil.
"""

resumen = resumidor(reseña_larga, max_length=50, min_length=20)
print(f"Resumen: {resumen[0]['summary_text']}")
# "Monitor con excelente calidad de imagen y 144Hz para gaming. Parlantes regulares."

print("\n✅ Summarization condensa reseñas largas en resúmenes útiles")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: Summarization — Resumir Reseña Larga.*

1. "Monitor con excelente calidad de imagen y 144Hz para gaming. Parlantes regulares."

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: Translation — Traducir Descripción

```python
from transformers import pipeline

traductor = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")

descripcion = """
Laptop gaming con procesador Intel Core i7, 16GB de RAM,
tarjeta gráfica NVIDIA RTX 3060 y SSD de 512GB.
"""

resultado = traductor(descripcion)
print(f"Original (ES): {descripcion.strip()}")
print(f"Traducción (EN): {resultado[0]['translation_text']}")
# "Gaming laptop with Intel Core i7 processor, 16GB RAM..."

print("\n✅ Traducción automática de catálogos a múltiples idiomas")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: Translation — Traducir Descripción.*

1. "Gaming laptop with Intel Core i7 processor, 16GB RAM..."

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: Feature Extraction — Embeddings de Productos

```python
from transformers import pipeline

extractor = pipeline("feature-extraction", model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

descripciones = [
    "Laptop gaming con RTX 3060",
    "Mouse inalámbrico ergonómico",
    "Teclado mecánico RGB"
]

# Obtener embeddings
embeddings = []
for desc in descripciones:
    emb = extractor(desc, pooling="mean")
    embeddings.append(emb[0])  # (1, num_tokens, dim)

import numpy as np
print(f"Embedding de '{descripciones[0]}': dimensiones {len(embeddings[0][0])}")
print(f"Shape del primer embedding: {np.array(embeddings[0]).shape}")

# Similitud: laptop vs mouse
sim = np.dot(embeddings[0][0], embeddings[1][0]) / (
    np.linalg.norm(embeddings[0][0]) * np.linalg.norm(embeddings[1][0])
)
print(f"Similitud '{descripciones[0]}' vs '{descripciones[1]}': {sim:.3f}")
# ~0.3 (baja, son categorías diferentes)

print("\n✅ Feature extraction: embeddings para búsqueda semántica y clustering")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: Feature Extraction — Embeddings de Productos.*

1. Obtener embeddings
2. Similitud: laptop vs mouse
3. ~0.3 (baja, son categorías diferentes)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: Text2Text Generation — Tareas con T5

```python
from transformers import pipeline

t5 = pipeline("text2text-generation", model="google/flan-t5-small")

# Varias tareas con el mismo modelo
tareas = [
    "Translate English to Spanish: Gaming laptop with RGB keyboard",
    "Summarize: The product arrived on time and works perfectly. Great quality.",
    "Classify: This laptop has excellent battery life. Options: positive, negative.",
]

for tarea in tareas:
    resultado = t5(tarea, max_new_tokens=30)
    print(f"Input: {tarea}")
    print(f"Output: {resultado[0]['generated_text']}\n")

print("✅ T5 unifica múltiples tareas en text-to-text generation")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: Text2Text Generation — Tareas con T5.*

1. Varias tareas con el mismo modelo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: Table Question Answering — Consultar Tabla

```python
from transformers import pipeline

tqa = pipeline("table-question-answering", model="google/tapas-base-finetuned-wtq")

tabla = {
    "Producto": ["Laptop A", "Monitor B", "Teclado C", "Mouse D"],
    "Precio": ["$1200", "$450", "$89", "$35"],
    "Stock": ["15", "8", "50", "100"],
    "Categoría": ["Electrónica", "Electrónica", "Periférico", "Periférico"]
}

preguntas = [
    "¿Cuál es el producto más caro?",
    "¿Cuántos monitores hay en stock?",
    "¿Qué productos son periféricos?"
]

for pregunta in preguntas:
    respuesta = tqa(table=tabla, query=pregunta)
    print(f"P: {pregunta}")
    print(f"R: {respuesta['answer']}\n")

print("✅ Table-QA permite consultar catálogos en formato tabla")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: Table Question Answering — Consultar Tabla.*

1. `from transformers import pipeline` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: Text Classification — Clasificar Quejas

```python
from transformers import pipeline

clasificador = pipeline("text-classification",
                        model="finiteautomata/bertweet-base-sentiment-analysis")

quejas = [
    "El producto llegó defectuoso y no funciona",
    "La entrega fue rápida y el producto es de buena calidad",
    "El servicio al cliente no resolvió mi problema",
    "Excelente relación calidad-precio, muy contento"
]

for queja in quejas:
    resultado = clasificador(queja)
    print(f"'{queja[:40]}...' => {resultado[0]['label']} ({resultado[0]['score']:.3f})")

print("\n✅ Text classification para monitorear quejas de clientes")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: Text Classification — Clasificar Quejas.*

1. `from transformers import pipeline` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: Token Classification — Etiquetar Tokens

```python
from transformers import pipeline

token_classifier = pipeline("token-classification",
                            model="dbmdz/bert-large-cased-finetuned-conll03-english",
                            aggregation_strategy="simple")

texto = "Apple lanzó la MacBook Pro M3 en su tienda de Nueva York por $2,499"
entidades = token_classifier(texto)

print("Tokens etiquetados:")
for e in entidades:
    print(f"  '{e['word']}' -> {e['entity_group']} (score: {e['score']:.3f})")

print("\n✅ Token classification: etiquetado a nivel de token")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: Token Classification — Etiquetar Tokens.*

1. `from transformers import pipeline` — Importa las librerías necesarias para el análisis.
2. `print("Tokens etiquetados:")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: Cambiar Modelo en Pipeline

```python
from transformers import pipeline

# Modelo especializado para español
clasificador_es = pipeline(
    "sentiment-analysis",
    model="bertin-project/bertin-roberta-base-sentiment"
)

reseñas_es = [
    "El producto es una maravilla, funciona perfectamente",
    "Horrible, se rompió al primer uso, no lo recomiendo"
]

for res in reseñas_es:
    resultado = clasificador_es(res)
    print(f"'{res[:30]}...' => {resultado[0]['label']} ({resultado[0]['score']:.3f})")

print("\n✅ Podemos cambiar el modelo a uno especializado para español")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: Cambiar Modelo en Pipeline.*

1. Modelo especializado para español

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: Batching — Procesar en Lote

```python
from transformers import pipeline

clasificador = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# 20 reseñas de ejemplo
reseñas = [
    "Great product, very satisfied",
    "Terrible quality, broke in a week",
    "Good value for the price",
    "Amazing battery life and fast shipping",
    "Not worth the money, disappointed",
    "Perfect for my needs, would buy again",
    "Average quality, nothing special",
    "Excellent customer service, fast refund",
    "Poor packaging, item arrived damaged",
    "Highly recommended for professionals",
    "Better than expected, great purchase",
    "Works as described, no complaints",
    "Cheap materials, feels fragile",
    "Outstanding performance, very happy",
    "Difficult to set up, instructions unclear",
    "Love this product, best purchase this year",
    "Decent quality but overpriced",
    "Exactly what I needed, fast delivery",
    "Faulty unit, but replacement was smooth",
    "Superb build quality and design",
] * 2  # 40 reseñas

# Sin batch
import time
start = time.time()
resultados = [clasificador(r) for r in reseñas[:10]]
t_sin_batch = time.time() - start
print(f"Sin batch (10 items): {t_sin_batch:.3f}s")

# Con batch
start = time.time()
resultados = clasificador(reseñas, batch_size=8)
t_con_batch = time.time() - start
print(f"Con batch_size=8 (40 items): {t_con_batch:.3f}s")
print(f"Speedup: {t_sin_batch * 4 / t_con_batch:.1f}x (estimado)")

print("\n✅ Batching acelera el procesamiento de múltiples textos")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: Batching — Procesar en Lote.*

1. 20 reseñas de ejemplo
2. Sin batch
3. Con batch

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 16: Parámetros — Truncation, Padding, Max Length

```python
from transformers import pipeline

clasificador = pipeline("sentiment-analysis")

# Texto muy largo
reseña_larga = "Este producto es " * 100 + "excelente."

# Sin truncation (puede fallar por exceder límite)
try:
    resultado = clasificador(reseña_larga, truncation=False)
except Exception as e:
    print(f"Sin truncation: ERROR — {type(e).__name__}")

# Con truncation
resultado = clasificador(reseña_larga, truncation=True, max_length=512)
print(f"Con truncation: {resultado[0]['label']} ({resultado[0]['score']:.3f})")

# Padding automático para batch
textos = ["Buen producto", "Excelente calidad", "Malo"]
resultados = clasificador(textos, padding=True, truncation=True, max_length=128)
for texto, res in zip(textos, resultados):
    print(f"  '{texto}' => {res['label']}")

print("\n✅ truncation, padding, max_length controlan el preprocesamiento")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: Parámetros — Truncation, Padding, Max Length.*

1. Texto muy largo
2. Sin truncation (puede fallar por exceder límite)
3. Con truncation
4. Padding automático para batch

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 17: GPU — device=0

```python
from transformers import pipeline
import torch

# Detectar GPU
device = 0 if torch.cuda.is_available() else -1
print(f"Usando device={device} ({'GPU' if device == 0 else 'CPU'})")

clasificador = pipeline("sentiment-analysis", device=device)

reseñas = ["Great laptop!", "Terrible mouse", "Amazing monitor"]
resultados = clasificador(reseñas)
for res, r in zip(resultados, reseñas):
    print(f"  '{r}' => {res['label']} ({res['score']:.3f})")

if device == 0:
    print("\n✅ GPU acelera significativamente la inferencia")
else:
    print("\n⚠️ No hay GPU disponible — el pipeline corre en CPU")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: GPU — device=0.*

1. Detectar GPU

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 18: Integrador — Pipeline para 100 Reseñas

```python
from transformers import pipeline
import time
import numpy as np

# ===== CONFIGURACIÓN =====
modelo_sentiment = "distilbert-base-uncased-finetuned-sst-2-english"
clasificador = pipeline("sentiment-analysis", model=modelo_sentiment, device=0 if __import__('torch').cuda.is_available() else -1)

modelo_ner = "dslim/bert-base-NER"
ner = pipeline("ner", model=modelo_ner, aggregation_strategy="simple", device=0 if __import__('torch').cuda.is_available() else -1)

modelo_qa = "distilbert-base-cased-distilled-squad"
qa = pipeline("question-answering", model=modelo_qa, device=0 if __import__('torch').cuda.is_available() else -1)

# ===== DATOS: 100 reseñas sintéticas =====
productos = ["laptop", "monitor", "teclado", "mouse", "audífonos"]
calificaciones = ["excelente", "bueno", "regular", "malo", "pésimo"]
reseñas = [
    f"Este {np.random.choice(productos)} es {np.random.choice(calificaciones)}. "
    f"{'Lo recomiendo totalmente.' if np.random.random() > 0.5 else 'No lo recomiendo.'}"
    for _ in range(100)
]

# ===== 1. CLASIFICAR SENTIMIENTO =====
start = time.time()
sentimientos = clasificador(reseñas, batch_size=16, truncation=True, max_length=128)
t_sent = time.time() - start
positivos = sum(1 for s in sentimientos if s['label'] == 'POSITIVE')
negativos = len(sentimientos) - positivos
print(f"Sentimiento: {positivos} positivas, {negativos} negativas ({t_sent:.2f}s)")

# ===== 2. EXTRAER ENTIDADES (primeras 10 reseñas) =====
start = time.time()
entidades = ner(reseñas[:10])
t_ner = time.time() - start
for i, ents in enumerate(entidades):
    if ents:
        print(f"  Reseña {i+1}: {[e['word'] for e in ents[:3]]}")
print(f"NER en 10 reseñas: {t_ner:.2f}s")

# ===== 3. RESPONDER PREGUNTAS (sobre reseñas positivas) =====
reseñas_positivas = [r for r, s in zip(reseñas, sentimientos) if s['label'] == 'POSITIVE']
if reseñas_positivas:
    contexto = " ".join(reseñas_positivas[:3])
    preguntas = ["What product is mentioned?", "Is it recommended?"]
    for pregunta in preguntas:
        respuesta = qa(question=pregunta, context=contexto)
        print(f"  QA '{pregunta}': {respuesta['answer']} ({respuesta['score']:.2f})")

# ===== 4. RESUMEN =====
print(f"\n✅ INTEGRADOR: Pipeline completo procesó {len(reseñas)} reseñas")
print(f"   Tasks: sentiment-analysis + NER + question-answering")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador — Pipeline para 100 Reseñas.*

1. ===== CONFIGURACIÓN =====
2. ===== DATOS: 100 reseñas sintéticas =====
3. ===== 1. CLASIFICAR SENTIMIENTO =====
4. ===== 2. EXTRAER ENTIDADES (primeras 10 reseñas) =====
5. ===== 3. RESPONDER PREGUNTAS (sobre reseñas positivas) =====
6. ===== 4. RESUMEN =====

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 3. Ejercicios Propuestos

1. **Pipeline personalizado**: Crea un pipeline que reciba una reseña de producto y devuelva: sentimiento (positivo/negativo), entidades (marcas, modelos) y un resumen de 1 línea.

2. **Zero-shot multi-etiqueta**: Usa zero-shot-classification para asignar 3 categorías a un producto de electrónica. Evalúa si las categorías son correctas.

3. **NER para catálogo**: Usa NER para extraer todos los productos, marcas y precios de un párrafo que describa 5 productos. Muestra los resultados en tabla.

4. **QA sobre catálogo**: Crea un contexto con 3 productos (nombre, precio, características). Haz 5 preguntas en lenguaje natural. Evalúa la precisión.

5. **Traducción masiva**: Traduce 20 descripciones de productos de español a inglés. Calcula el tiempo total y por elemento.

6. **Comparación de modelos**: Compara 3 modelos de sentiment-analysis (distilbert, roberta, bertin-roberta-es) en 20 reseñas en español. Reporta diferencias.

7. **Pipeline con fallback**: Crea un pipeline que intente con GPU, y si falla, use CPU automáticamente.

8. **Dashboard de reseñas**: Procesa 50 reseñas con sentiment-analysis + NER y genera un resumen estadístico (% positivo, marcas más mencionadas, productos populares).

---

## 4. Resumen

| Pipeline | Una línea | Aplicación |
|---|---|---|
| `sentiment-analysis` | `pipeline("sentiment-analysis")(texto)` | Monitorear reseñas |
| `zero-shot-classification` | `pipeline("zero-shot-classification")(texto, candidatos)` | Categorizar sin datos |
| `text-generation` | `pipeline("text-generation", model)(prompt)` | Generar descripciones |
| `fill-mask` | `pipeline("fill-mask")(texto_con_[MASK])` | Completar textos |
| `ner` | `pipeline("ner")(texto)` | Extraer entidades |
| `question-answering` | `pipeline("question-answering")(question, context)` | FAQ inteligente |
| `summarization` | `pipeline("summarization")(texto)` | Resumir reseñas |
| `translation` | `pipeline("translation", model)(texto)` | Traducir catálogos |
| `feature-extraction` | `pipeline("feature-extraction")(texto)` | Embeddings |

**Conclusión**: `pipeline()` permite resolver tareas complejas de NLP con 1-3 líneas de código. Es la herramienta ideal para prototipado rápido y aplicaciones de ventas que requieren análisis de texto sin infraestructura compleja.
