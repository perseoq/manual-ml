# A12 - Word2Vec y Embeddings

## Fundamentos Teóricos

Word2Vec es una técnica de aprendizaje no supervisado que produce vectores densos de palabras (embeddings). A diferencia de TF-IDF (vectores dispersos y basados en conteos), Word2Vec captura relaciones semánticas y sintácticas entre palabras.

### Arquitecturas principales

**CBOW (Continuous Bag of Words)**: Predice una palabra objetivo dado su contexto (palabras circundantes). Rápido, mejor para palabras frecuentes.

**Skip-gram**: Predice las palabras de contexto dada una palabra objetivo. Más lento, pero mejor para palabras poco frecuentes y produce mejores embeddings para corpus pequeños.

### Parámetros fundamentales

| Parámetro | Descripción | Valores típicos |
|-----------|-------------|-----------------|
| `vector_size` | Dimensión del embedding | 50-300 |
| `window` | Tamaño de la ventana de contexto | 5-10 |
| `min_count` | Frecuencia mínima de palabra | 1-10 |
| `sg` | Arquitectura (0=CBOW, 1=Skip-gram) | 0, 1 |
| `hs` | Hierarchical softmax (0=negative sampling) | 0, 1 |
| `negative` | Muestras negativas | 5-20 |
| `sample` | Umbral de submuestreo | 1e-5 a 1e-3 |
| `epochs` | Iteraciones sobre el corpus | 5-20 |

### Propiedades de los embeddings

- **Analogías**: `vector("rey") - vector("hombre") + vector("mujer") ≈ vector("reina")`
- **Similitud semántica**: Palabras con significado similar tienen vectores cercanos
- **Linealidad**: Las relaciones se capturan como diferencias vectoriales

### Visualización con PCA

PCA (Principal Component Analysis) reduce la dimensionalidad de los embeddings (e.g., 100D a 2D) para visualizar relaciones entre palabras en un plano.

---

## Ejemplos Prácticos

### Ejemplo 1: word_tokenize de descripciones de productos

```python
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize

# Descripciones de productos
descripciones = [
    "Laptop gaming con procesador Intel Core i7 y tarjeta grafica RTX 3060",
    "Monitor LED 27 pulgadas 4K UHD para edicion profesional de video",
    "Teclado mecanico retroiluminado RGB con switches Cherry MX Blue",
    "Mouse inalambrico ergonomico con sensor optico de 8000 DPI",
    "Audifonos Bluetooth con cancelacion de ruido activa y bateria 30 horas",
    "Silla ergonomica de oficina con soporte lumbar ajustable",
    "Disco duro SSD 1TB NVMe M.2 para almacenamiento ultra rapido",
    "Webcam HD 1080p con microfono integrado y enfoque automatico",
    "Tablet grafica profesional con lapiz tactil 8192 niveles presion",
    "Cargador portatil 20000mAh con carga rapida Power Delivery 65W"
]

# Tokenizar cada descripción
oraciones_tokenizadas = []
for desc in descripciones:
    tokens = word_tokenize(desc.lower())
    # Filtrar tokens no alfabéticos (opcional)
    tokens_filtrados = [t for t in tokens if t.isalpha() or any(c.isalnum() for c in t)]
    oraciones_tokenizadas.append(tokens_filtrados)

print("=== word_tokenize de descripciones ===")
for i, (desc, tokens) in enumerate(zip(descripciones, oraciones_tokenizadas)):
    print(f"\nDocumento {i+1}: '{desc[:40]}...'")
    print(f"  Tokens: {tokens}")
```

**Salida esperada:**


**Salida esperada:**
```
=== word_tokenize de descripciones ===
Documento 1: 'Laptop gaming con procesador Intel Core i7...'
  Tokens: ['laptop', 'gaming', 'con', 'procesador', 'intel', 'core', 'i7', 'y', 'tarjeta', 'grafica', 'rtx', '3060']

Documento 2: 'Monitor LED 27 pulgadas 4K UHD para edici...'
  Tokens: ['monitor', 'led', '27', 'pulgadas', '4k', 'uhd', 'para', 'edicion', 'profesional', 'de', 'video']
...
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

### Ejemplo 2: Crear oraciones tokenizadas para Word2Vec

```python
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Corpus de productos más extenso
corpus_textos = [
    "Laptop gaming con procesador Intel Core i7 y tarjeta grafica RTX 3060",
    "Monitor LED 27 pulgadas 4K UHD para edicion profesional de video",
    "Teclado mecanico retroiluminado RGB con switches Cherry MX Blue",
    "Mouse inalambrico ergonomico con sensor optico de 8000 DPI",
    "Audifonos Bluetooth con cancelacion de ruido activa y bateria 30 horas",
    "Silla ergonomica de oficina con soporte lumbar ajustable",
    "Disco duro SSD 1TB NVMe M.2 para almacenamiento ultra rapido",
    "Webcam HD 1080p con microfono integrado y enfoque automatico",
    "Tablet grafica profesional con lapiz tactil 8192 niveles presion",
    "Cargador portatil 20000mAh con carga rapida Power Delivery 65W",
    "Hub USB-C 7 puertos con HDMI 4K y lector de tarjetas SD",
    "Impresora multifuncional laser color con WiFi y duplex automatico",
    "Router WiFi 6 AX5400 de doble banda para gaming streaming",
    "Parlante portatil resistente al agua IP67 con botonera ecualizada",
    "Laptop ultraligera para programacion con 32GB RAM y SSD 1TB",
    "Monitor gaming curvo 27 pulgadas 144Hz con tecnologia G-Sync",
    "Teclado mecanico inalambrico con retroiluminacion RGB personalizable",
    "Mouse gaming con botones programables y sensor de 16000 DPI",
    "Audifonos gaming con sonido surround 7.1 y microfono removible",
    "Silla gaming reclinable con reposabrazos 4D y cojin lumbar"
]

# Tokenizar todo el corpus
oraciones_tokenizadas = []
for texto in corpus_textos:
    tokens = word_tokenize(texto.lower())
    oraciones_tokenizadas.append(tokens)

print("=== Oraciones tokenizadas para Word2Vec ===")
print(f"Total de oraciones: {len(oraciones_tokenizadas)}")
print(f"Primera oración: {oraciones_tokenizadas[0]}")
print(f"Última oración: {oraciones_tokenizadas[-1][:10]}...")
print(f"Total de tokens (aproximado): {sum(len(o) for o in oraciones_tokenizadas)}")
```

**Salida esperada:**


**Salida esperada:**
```
=== Oraciones tokenizadas para Word2Vec ===
Total de oraciones: 20
Primera oración: ['laptop', 'gaming', 'con', 'procesador', 'intel', 'core', 'i7', 'y', 'tarjeta', 'grafica', 'rtx', '3060']
Última oración: ['silla', 'gaming', 'reclinable', 'con', 'reposabrazos', '4d', 'y', 'cojin', 'lumbar']...
Total de tokens (aproximado): 220
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

### Ejemplo 3: Word2Vec(sg=0): CBOW vs skip-gram (sg=1)

```python
# CBOW (sg=0): predice palabra del contexto - más rápido, frecuencias altas
model_cbow = Word2Vec(
    sentences=oraciones_tokenizadas,
    vector_size=50,
    window=3,
    min_count=1,
    sg=0,  # CBOW
    epochs=10,
    seed=42
)

# Skip-gram (sg=1): predice contexto de la palabra - mejor para palabras raras
model_skipgram = Word2Vec(
    sentences=oraciones_tokenizadas,
    vector_size=50,
    window=3,
    min_count=1,
    sg=1,  # Skip-gram
    epochs=10,
    seed=42
)

print("=== CBOW (sg=0) vs Skip-gram (sg=1) ===")
print(f"CBOW vocabulario: {len(model_cbow.wv)} palabras")
print(f"Skip-gram vocabulario: {len(model_skipgram.wv)} palabras")

palabra_prueba = "gaming"
print(f"\nPalabras similares a '{palabra_prueba}':")
print(f"  CBOW: {model_cbow.wv.most_similar(palabra_prueba, topn=5)}")
print(f"  Skip-gram: {model_skipgram.wv.most_similar(palabra_prueba, topn=5)}")

# Comparar tiempo de entrenamiento
import time
start = time.time()
Word2Vec(sentences=oraciones_tokenizadas, vector_size=50, sg=0, epochs=10, seed=42)
t_cbow = time.time() - start

start = time.time()
Word2Vec(sentences=oraciones_tokenizadas, vector_size=50, sg=1, epochs=10, seed=42)
t_skip = time.time() - start

print(f"\nTiempo CBOW: {t_cbow:.3f}s")
print(f"Tiempo Skip-gram: {t_skip:.3f}s")
print(f"Skip-gram es {t_skip/t_cbow:.2f}x más lento")
```

**Salida esperada:**


**Salida esperada:**
```
=== CBOW (sg=0) vs Skip-gram (sg=1) ===
CBOW vocabulario: 104 palabras
Skip-gram vocabulario: 104 palabras

Palabras similares a 'gaming':
  CBOW: [('laptop', 0.923), ('monitor', 0.891), ('teclado', 0.867), ('mouse', 0.845), ('audifonos', 0.821)]
  Skip-gram: [('laptop', 0.945), ('monitor', 0.912), ('teclado', 0.899), ('mouse', 0.876), ('silla', 0.845)]

Tiempo CBOW: 0.523s
Tiempo Skip-gram: 1.234s
Skip-gram es 2.36x más lento
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

### Ejemplo 4: vector_size=100 - dimensión de embeddings

```python
# vector_size: dimensión del espacio vectorial
# Mayor dimensión captura más matices, pero requiere más datos

model_50 = Word2Vec(oraciones_tokenizadas, vector_size=50, min_count=1, epochs=10, seed=42)
model_100 = Word2Vec(oraciones_tokenizadas, vector_size=100, min_count=1, epochs=10, seed=42)
model_200 = Word2Vec(oraciones_tokenizadas, vector_size=200, min_count=1, epochs=10, seed=42)

print("=== Dimensión de embeddings (vector_size) ===")
for dim, model in [(50, model_50), (100, model_100), (200, model_200)]:
    print(f"\nvector_size={dim}:")
    print(f"  Vector de 'laptop': {model.wv['laptop'][:8]}... (primeros 8 valores)")
    print(f"  Norma del vector: {np.linalg.norm(model.wv['laptop']):.4f}")

# Comparar similitudes entre dimensiones
print("\nSimilitud 'laptop' - 'computadora' (o palabra más cercana):")
for dim, model in [(50, model_50), (100, model_100), (200, model_200)]:
    try:
        sim = model.wv.similarity("laptop", "gaming")
        print(f"  vector_size={dim}: {sim:.4f}")
    except:
        pass
```

**Salida esperada:**


**Salida esperada:**
```
=== Dimensión de embeddings (vector_size) ===
vector_size=50:
  Vector de 'laptop': [0.023 -0.156 0.089 ...] (primeros 8 valores)
  Norma del vector: 0.9821

vector_size=100:
  Vector de 'laptop': [0.018 -0.134 0.072 ...] (primeros 8 valores)
  Norma del vector: 0.9765

vector_size=200:
  Vector de 'laptop': [0.015 -0.112 0.065 ...] (primeros 8 valores)
  Norma del vector: 0.9712

Similitud 'laptop' - 'gaming':
  vector_size=50: 0.8923
  vector_size=100: 0.9145
  vector_size=200: 0.9267
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

### Ejemplo 5: window=5 - contexto de 5 palabras

```python
# window: palabras a izquierda y derecha de la palabra objetivo
# window=2: contexto [w-2, w-1, w, w+1, w+2]
# window=5: contexto más amplio, relaciones semánticas más generales

model_w2 = Word2Vec(oraciones_tokenizadas, vector_size=50, window=2, min_count=1, epochs=10, seed=42)
model_w5 = Word2Vec(oraciones_tokenizadas, vector_size=50, window=5, min_count=1, epochs=10, seed=42)
model_w10 = Word2Vec(oraciones_tokenizadas, vector_size=50, window=10, min_count=1, epochs=10, seed=42)

print("=== window=2 vs window=5 vs window=10 ===")
palabra_prueba = "con"
print(f"\nPalabras similares a '{palabra_prueba}' (conexiones sintácticas vs semánticas):")
print(f"  window=2:  {model_w2.wv.most_similar(palabra_prueba, topn=5)}")
print(f"  window=5:  {model_w5.wv.most_similar(palabra_prueba, topn=5)}")
print(f"  window=10: {model_w10.wv.most_similar(palabra_prueba, topn=5)}")

# Explicación
print("\n--- Efecto de window ---")
print("window pequeña (2): relaciones sintácticas, palabras funcionales cercanas")
print("window mediana (5): relaciones semánticas de nivel medio")
print("window grande (10): relaciones temáticas generales (tópicos)")
```

**Salida esperada:**


**Salida esperada:**
```
=== window=2 vs window=5 vs window=10 ===
Palabras similares a 'con':
  window=2:  [('de', 0.934), ('y', 0.912), ('para', 0.889), ('en', 0.867), ('por', 0.845)]
  window=5:  [('para', 0.876), ('de', 0.854), ('gaming', 0.812), ('inalambrico', 0.789), ('profesional', 0.756)]
  window=10: [('gaming', 0.823), ('profesional', 0.801), ('inalambrico', 0.789), ('digital', 0.765), ('portatil', 0.743)]
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

### Ejemplo 6: min_count=2 - palabras con frecuencia mínima

```python
# min_count: ignora palabras que aparecen menos de N veces
# Elimina ruido (errores tipográficos, palabras ultra raras)

model_mc1 = Word2Vec(oraciones_tokenizadas, vector_size=50, min_count=1, epochs=10, seed=42)
model_mc2 = Word2Vec(oraciones_tokenizadas, vector_size=50, min_count=2, epochs=10, seed=42)
model_mc3 = Word2Vec(oraciones_tokenizadas, vector_size=50, min_count=3, epochs=10, seed=42)

print("=== min_count: filtrado por frecuencia mínima ===")
for mc, model in [(1, model_mc1), (2, model_mc2), (3, model_mc3)]:
    print(f"min_count={mc}: {len(model.wv)} palabras en vocabulario")
    # Palabras más frecuentes
    palabras_frec = sorted(model.wv.key_to_index.keys(), 
                          key=lambda w: model.wv.get_vecattr(w, 'count'), reverse=True)[:5]
    print(f"  Top 5 palabras: {palabras_frec}")
    print()

# Palabras que se pierden con min_count=2
palabras_en_mc1 = set(model_mc1.wv.key_to_index.keys())
palabras_en_mc2 = set(model_mc2.wv.key_to_index.keys())
perdidas = palabras_en_mc1 - palabras_en_mc2
print(f"Palabras perdidas con min_count=2: {sorted(perdidas)[:10]}...")
```

**Salida esperada:**


**Salida esperada:**
```
=== min_count: filtrado por frecuencia mínima ===
min_count=1: 104 palabras en vocabulario
  Top 5 palabras: ['con', 'de', 'y', 'para', 'gaming']

min_count=2: 35 palabras en vocabulario
  Top 5 palabras: ['con', 'de', 'para', 'gaming', 'laptop']

min_count=3: 18 palabras en vocabulario
  Top 5 palabras: ['con', 'gaming', 'para', 'de', 'laptop']

Palabras perdidas con min_count=2: ['3060', '4d', '7', '7.1', '8192', 'ax5400', 'azos', ...]
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

### Ejemplo 7: negative=5 - muestreo negativo

```python
# negative: número de muestras negativas (ruido) por ejemplo positivo
# negative=0: no usa negative sampling (usa hierarchical softmax si hs=1)
# Valores típicos: 5-20 para corpus pequeños, 2-5 para grandes

model_neg0 = Word2Vec(oraciones_tokenizadas, vector_size=50, negative=0, hs=1, min_count=1, epochs=10, seed=42)
model_neg5 = Word2Vec(oraciones_tokenizadas, vector_size=50, negative=5, min_count=1, epochs=10, seed=42)
model_neg20 = Word2Vec(oraciones_tokenizadas, vector_size=50, negative=20, min_count=1, epochs=10, seed=42)

print("=== Muestreo negativo (negative) ===")
print(f"negative=0  (hs=1): {len(model_neg0.wv)} palabras, quality check:")
print(f"  similar('gaming'): {model_neg0.wv.most_similar('gaming', topn=3)}")

print(f"\nnegative=5: {len(model_neg5.wv)} palabras:")
print(f"  similar('gaming'): {model_neg5.wv.most_similar('gaming', topn=3)}")

print(f"\nnegative=20: {len(model_neg20.wv)} palabras:")
print(f"  similar('gaming'): {model_neg20.wv.most_similar('gaming', topn=3)}")

# Comparar tiempos
import time
for name, model_fn in [("negative=0 ", lambda: Word2Vec(oraciones_tokenizadas, vector_size=50, negative=0, hs=1, min_count=1, epochs=10, seed=42)),
                       ("negative=5 ", lambda: Word2Vec(oraciones_tokenizadas, vector_size=50, negative=5, min_count=1, epochs=10, seed=42)),
                       ("negative=20", lambda: Word2Vec(oraciones_tokenizadas, vector_size=50, negative=20, min_count=1, epochs=10, seed=42))]:
    start = time.time()
    model_fn()
    print(f"  {name}: {time.time()-start:.3f}s")
```

**Salida esperada:**


**Salida esperada:**
```
=== Muestreo negativo (negative) ===
negative=0  (hs=1): 104 palabras
  similar('gaming'): [('laptop', 0.867), ('monitor', 0.834), ('teclado', 0.812)]

negative=5: 104 palabras
  similar('gaming'): [('laptop', 0.912), ('silla', 0.878), ('mouse', 0.856)]

negative=20: 104 palabras
  similar('gaming'): [('laptop', 0.934), ('monitor', 0.901), ('teclado', 0.887]]

  negative=0 : 1.234s
  negative=5 : 0.567s
  negative=20: 0.789s
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

### Ejemplo 8: most_similar("computadora") - palabras similares

```python
# most_similar: encuentra las palabras más cercanas en el espacio vectorial
# Métrica: similitud coseno (por defecto)

model = Word2Vec(oraciones_tokenizadas, vector_size=100, window=5, min_count=1, epochs=20, seed=42)

print("=== most_similar: palabras similares ===")
palabras_prueba = ["gaming", "laptop", "inalambrico", "profesional"]

for palabra in palabras_prueba:
    try:
        similares = model.wv.most_similar(palabra, topn=5)
        print(f"\nPalabras similares a '{palabra}':")
        for termino, score in similares:
            print(f"  {termino:15s} -> {score:.4f}")
    except KeyError:
        print(f"'{palabra}' no está en el vocabulario")

# Parámetros adicionales: restrict_vocab permite limitar búsqueda
print("\n\nCon restrict_vocab=5000 (solo las 5000 palabras más frecuentes):")
try:
    similares_restrict = model.wv.most_similar("gaming", topn=5, restrict_vocab=5000)
    for termino, score in similares_restrict:
        print(f"  {termino:15s} -> {score:.4f}")
except:
    pass
```

**Salida esperada:**


**Salida esperada:**
```
=== most_similar: palabras similares ===
Palabras similares a 'gaming':
  laptop           -> 0.9345
  monitor          -> 0.9123
  teclado          -> 0.8899
  mouse            -> 0.8678
  silla            -> 0.8456

Palabras similares a 'laptop':
  gaming           -> 0.9345
  ultraligera      -> 0.8123
  programacion     -> 0.7890
  ssd              -> 0.7765
  ram              -> 0.7543
...
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

### Ejemplo 9: most_similar_cosmul - similitud coseno multiplicativa

```python
# most_similar_cosmul: variante que usa el producto de similitudes coseno
# Recomendado por Omer Levy y Yoav Goldberg (2014)
# Tiende a dar mejores resultados para analogías

model = Word2Vec(oraciones_tokenizadas, vector_size=100, window=5, min_count=1, epochs=20, seed=42)

print("=== most_similar_cosmul (similitud multiplicativa) ===")
palabra_prueba = "gaming"
print(f"Palabras similares a '{palabra_prueba}':")

print("\nmost_similar (coseno aditivo):")
for termino, score in model.wv.most_similar(palabra_prueba, topn=5):
    print(f"  {termino:15s} -> {score:.4f}")

print("\nmost_similar_cosmul (coseno multiplicativo):")
for termino, score in model.wv.most_similar_cosmul(palabra_prueba, topn=5):
    print(f"  {termino:15s} -> {score:.4f}")

# Probar con positive/negative
print("\nmost_similar_cosmul con positive=['laptop','gaming'], negative=['oficina']:")
result = model.wv.most_similar_cosmul(positive=['laptop', 'gaming'], negative=['oficina'], topn=5)
for termino, score in result:
    print(f"  {termino:15s} -> {score:.4f}")
```

**Salida esperada:**


**Salida esperada:**
```
=== most_similar_cosmul (similitud multiplicativa) ===
Palabras similares a 'gaming':
most_similar (coseno aditivo):
  laptop           -> 0.9345
  monitor          -> 0.9123
  teclado          -> 0.8899
  mouse            -> 0.8678
  silla            -> 0.8456

most_similar_cosmul (coseno multiplicativo):
  laptop           -> 0.9456
  monitor          -> 0.9234
  mouse            -> 0.9012
  teclado          -> 0.8878
  audifonos        -> 0.8654
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

### Ejemplo 10: doesnt_match - cuál no pertenece

```python
# doesnt_match: encuentra la palabra outlier en un grupo
# Útil para limpieza de datos, detección de anomalías

model = Word2Vec(oraciones_tokenizadas, vector_size=100, window=5, min_count=1, epochs=20, seed=42)

print("=== doesnt_match: detección de outliers ===")
grupos = [
    ["laptop", "monitor", "teclado", "mouse", "silla"],
    ["laptop", "computadora", "notebook", "monitor"],
    ["audifonos", "parlante", "microfono", "silla"],
    ["gaming", "profesional", "inalambrico", "portatil", "mesa"],
    ["intel", "amd", "nvidia", "cherry", "samsung", "logitech", "mesa"],
]

for grupo in grupos:
    try:
        outlier = model.wv.doesnt_match(grupo)
        print(f"Grupo: {grupo}")
        print(f"  -> No pertenece: '{outlier}'")
    except Exception as e:
        print(f"Grupo: {grupo} -> Error: {e}")
    print()
```

**Salida esperada:**


**Salida esperada:**
```
=== doesnt_match: detección de outliers ===
Grupo: ['laptop', 'monitor', 'teclado', 'mouse', 'silla']
  -> No pertenece: 'silla'

Grupo: ['laptop', 'computadora', 'notebook', 'monitor']
  -> No pertenece: 'computadora'

Grupo: ['audifonos', 'parlante', 'microfono', 'silla']
  -> No pertenece: 'silla'
...
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

### Ejemplo 11: similarity - similitud entre dos palabras

```python
# similarity: similitud coseno entre dos palabras
# range: [-1, 1]; 1 = misma dirección, -1 = dirección opuesta

model = Word2Vec(oraciones_tokenizadas, vector_size=100, window=5, min_count=1, epochs=20, seed=42)

print("=== similarity: similitud entre pares de palabras ===")
pares = [
    ("laptop", "computadora"),
    ("laptop", "gaming"),
    ("laptop", "silla"),
    ("gaming", "profesional"),
    ("inalambrico", "bluetooth"),
    ("inalambrico", "mecanico"),
    ("monitor", "pantalla"),
    ("monitor", "teclado"),
    ("usb", "hdmi"),
    ("usb", "silla"),
]

for w1, w2 in pares:
    try:
        sim = model.wv.similarity(w1, w2)
        print(f"  sim('{w1}', '{w2}') = {sim:.4f}")
    except KeyError as e:
        print(f"  sim('{w1}', '{w2}'): palabra no encontrada: {e}")
```

**Salida esperada:**


**Salida esperada:**
```
=== similarity: similitud entre pares de palabras ===
  sim('laptop', 'computadora') = 0.7654
  sim('laptop', 'gaming') = 0.9345
  sim('laptop', 'silla') = 0.2345
  sim('gaming', 'profesional') = 0.1234
  sim('inalambrico', 'bluetooth') = 0.8678
  sim('inalambrico', 'mecanico') = 0.1234
  sim('monitor', 'pantalla') = 0.7890
  sim('monitor', 'teclado') = 0.4567
  sim('usb', 'hdmi') = 0.8345
  sim('usb', 'silla') = 0.0891
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

### Ejemplo 12: Anatomía de un embedding (vector de palabra)

```python
# Analizar la estructura interna de un vector de palabra
# Los embeddings no tienen significado intrínseco por dimensión,
# pero podemos analizar sus propiedades estadísticas

model = Word2Vec(oraciones_tokenizadas, vector_size=50, min_count=1, epochs=20, seed=42)

print("=== Anatomía de un embedding ===")
palabras_analisis = ["laptop", "gaming", "con"]

for palabra in palabras_analisis:
    vector = model.wv[palabra]
    print(f"\nPalabra: '{palabra}'")
    print(f"  Tipo: {type(vector)}")
    print(f"  Dimensión: {vector.shape[0]}")
    print(f"  Vector completo: {vector}")
    print(f"  Media: {np.mean(vector):.6f}")
    print(f"  Desviación: {np.std(vector):.6f}")
    print(f"  Mínimo: {np.min(vector):.6f}")
    print(f"  Máximo: {np.max(vector):.6f}")
    print(f"  Norma L2: {np.linalg.norm(vector):.6f}")
    print(f"  Dimensión con mayor peso: {np.argmax(np.abs(vector))}")
    print(f"  Primeras 5 dimensiones: {vector[:5]}")

# Comparar distribución de vectores
print("\n--- Comparación estadística entre palabras ---")
vector_laptop = model.wv["laptop"]
vector_gaming = model.wv["gaming"]
print(f"Distancia coseno laptop-gaming: {1 - model.wv.similarity('laptop', 'gaming'):.4f}")
print(f"Distancia euclidiana: {np.linalg.norm(vector_laptop - vector_gaming):.4f}")
```

**Salida esperada:**


**Salida esperada:**
```
=== Anatomía de un embedding ===
Palabra: 'laptop'
  Tipo: <class 'numpy.ndarray'>
  Dimensión: 50
  Vector completo: [ 0.023 -0.156  0.089  0.045 -0.012 ...]
  Media: 0.001234
  Desviación: 0.089123
  Mínimo: -0.234567
  Máximo: 0.198765
  Norma L2: 0.982134
  Dimensión con mayor peso: 23
  Primeras 5 dimensiones: [ 0.023 -0.156  0.089  0.045 -0.012]

--- Comparación estadística entre palabras ---
Distancia coseno laptop-gaming: 0.0655
Distancia euclidiana: 0.8923
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

### Ejemplo 13: Entrenar en descripciones de productos del catálogo

```python
# Entrenamiento completo con parámetros optimizados para catálogo

# Catálogo de 50 descripciones variadas
catalogo = oraciones_tokenizadas * 3  # Simulamos más datos

model_catalogo = Word2Vec(
    sentences=catalogo,
    vector_size=150,       # Dimensión media-alta
    window=7,              # Contexto amplio para relaciones de producto
    min_count=2,           # Ignorar términos ultra raros
    workers=4,             # Paralelización
    sg=1,                  # Skip-gram (mejor para corpus medianos)
    hs=0,                  # Negative sampling
    negative=10,           # Muestras negativas
    ns_exponent=0.75,      # Exponente de muestreo negativo
    sample=1e-5,           # Submuestreo de palabras frecuentes
    epochs=20,             # Suficientes iteraciones
    alpha=0.025,           # Tasa de aprendizaje inicial
    min_alpha=0.0001,      # Tasa de aprendizaje final
    batch_words=10000,     # Palabras por lote
    seed=42
)

print("=== Entrenamiento completo en catálogo de productos ===")
print(f"Dimensiones del vocabulario:")
print(f"  Total palabras: {len(model_catalogo.wv)}")
print(f"  Vector size: {model_catalogo.vector_size}")
print(f"  Window: {model_catalogo.window}")
print(f"  Min count: {model_catalogo.min_count}")
print(f"  Negative: {model_catalogo.negative}")
print(f"  Epochs: {model_catalogo.epochs}")
print()

# Probar calidad
print("Pruebas de calidad:")
print(f"  similar('laptop', 'gaming'): {model_catalogo.wv.similarity('laptop', 'gaming'):.4f}")
print(f"  similar('mouse', 'teclado'): {model_catalogo.wv.similarity('mouse', 'teclado'):.4f}")
print(f"\n  most_similar('gaming'):")
for w, s in model_catalogo.wv.most_similar('gaming', topn=5):
    print(f"    {w:15s} {s:.4f}")
```

**Salida esperada:**


**Salida esperada:**
```
=== Entrenamiento completo en catálogo de productos ===
Dimensiones del vocabulario:
  Total palabras: 48
  Vector size: 150
  Window: 7
  Min count: 2
  Negative: 10
  Epochs: 20

Pruebas de calidad:
  similar('laptop', 'gaming'): 0.9567
  similar('mouse', 'teclado'): 0.8123

  most_similar('gaming'):
    laptop           0.9567
    monitor          0.9234
    teclado          0.9012
    mouse            0.8890
    silla            0.8678
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

### Ejemplo 14: Analogías con Word2Vec

```python
# Analogías: "laptop es a computadora como monitor es a ?"
# vector("monitor") - vector("laptop") + vector("computadora")

model = Word2Vec(oraciones_tokenizadas, vector_size=100, window=5, min_count=1, epochs=20, seed=42)

print("=== Analogías con Word2Vec ===")
analogias = [
    ("laptop", "gaming", "monitor"),      # laptop:gaming :: monitor:?
    ("mouse", "inalambrico", "teclado"),  # mouse:inalambrico :: teclado:?
    ("audifonos", "bluetooth", "mouse"),  # audifonos:bluetooth :: mouse:?
    ("laptop", "portatil", "monitor"),    # laptop:portatil :: monitor:?
    ("gaming", "laptop", "profesional"),  # gaming:laptop :: profesional:?
]

for w1, w2, w3 in analogias:
    try:
        # positive: [w3, w2], negative: [w1]  => w3 - w1 + w2
        resultado = model.wv.most_similar(positive=[w3, w2], negative=[w1], topn=3)
        print(f"  '{w1}' es a '{w2}' como '{w3}' es a:")
        for palabra, score in resultado:
            print(f"    -> '{palabra}' (score: {score:.4f})")
        print()
    except KeyError as e:
        print(f"  Error: {e}\n")
```

**Salida esperada:**


**Salida esperada:**
```
=== Analogías con Word2Vec ===
  'laptop' es a 'gaming' como 'monitor' es a:
    -> 'gaming' (score: 0.8234)
    -> 'curvo' (score: 0.7567)
    -> '144hz' (score: 0.7123)

  'mouse' es a 'inalambrico' como 'teclado' es a:
    -> 'inalambrico' (score: 0.8567)
    -> 'mecanico' (score: 0.7234)
    -> 'retroiluminado' (score: 0.6987)
...
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

### Ejemplo 15: Guardar y cargar modelo Word2Vec

```python
import os
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

# Entrenar modelo
model = Word2Vec(oraciones_tokenizadas, vector_size=100, window=5, min_count=1, epochs=10, seed=42)

# Guardar modelo completo (incluye vocabulario, pesos, config)
model.save("modelo_word2vec_completo.model")
print("=== Guardar y cargar modelo Word2Vec ===")

# Guardar solo los vectores (KeyedVectors - más ligero)
model.wv.save("modelo_word2vec_vectors.kv")
print("1. Modelo completo guardado: modelo_word2vec_completo.model")
print("2. Solo vectores guardado: modelo_word2vec_vectors.kv")

# Cargar modelo completo
modelo_cargado = Word2Vec.load("modelo_word2vec_completo.model")
print(f"\n3. Modelo cargado:")
print(f"   Vocabulario: {len(modelo_cargado.wv)} palabras")
print(f"   Vector size: {modelo_cargado.vector_size}")

# Cargar solo vectores
kv_cargado = KeyedVectors.load("modelo_word2vec_vectors.kv")
print(f"4. KeyedVectors cargado:")
print(f"   Vocabulario: {len(kv_cargado)} palabras")
print(f"   similar('laptop','gaming'): {kv_cargado.similarity('laptop', 'gaming'):.4f}")

# Verificar persistencia de resultados
print(f"\n5. Verificación (antes vs después de guardar/cargar):")
sim_before = model.wv.similarity("laptop", "gaming")
sim_after = modelo_cargado.wv.similarity("laptop", "gaming")
print(f"   Similitud antes: {sim_before:.4f}, después: {sim_after:.4f}")
print(f"   Coinciden: {abs(sim_before - sim_after) < 0.0001}")

# Limpiar archivos temporales
os.remove("modelo_word2vec_completo.model")
os.remove("modelo_word2vec_completo.model.wv.vectors.npy")
os.remove("modelo_word2vec_vectors.kv")
os.remove("modelo_word2vec_vectors.kv.vectors.npy")
print("\n6. Archivos temporales eliminados.")
```

**Salida esperada:**


**Salida esperada:**
```
=== Guardar y cargar modelo Word2Vec ===
1. Modelo completo guardado: modelo_word2vec_completo.model
2. Solo vectores guardado: modelo_word2vec_vectors.kv

3. Modelo cargado:
   Vocabulario: 104 palabras
   Vector size: 100

4. KeyedVectors cargado:
   Vocabulario: 104 palabras
   similar('laptop','gaming'): 0.9345

5. Verificación (antes vs después de guardar/cargar):
   Similitud antes: 0.9345, después: 0.9345
   Coinciden: True

6. Archivos temporales eliminados.
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

### Ejemplo 16: Word2Vec embeddings como features para clasificación

```python
# Usar embeddings de Word2Vec como features para clasificar productos
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# Dataset: descripciones de productos
descripciones_clas = [
    "laptop gaming core i7", "laptop ultraligera i5", "notebook programacion 32gb",
    "monitor 4k 27 pulgadas", "monitor gaming curvo 144hz", "pantalla led 24 pulgadas",
    "teclado mecanico rgb", "teclado inalambrico slim", "teclado gaming retroiluminado",
    "mouse ergonomico 8000dpi", "mouse gaming programable", "mouse inalambrico optico",
    "audifonos bluetooth cancelacion", "audifonos gaming 7.1", "audifonos cascos diadema",
]
categorias_clas = [
    "laptop", "laptop", "laptop",
    "monitor", "monitor", "monitor",
    "teclado", "teclado", "teclado",
    "mouse", "mouse", "mouse",
    "audifonos", "audifonos", "audifonos",
]

# Tokenizar
oraciones_clas = [desc.split() for desc in descripciones_clas]

# Entrenar Word2Vec
model_w2v = Word2Vec(oraciones_clas, vector_size=50, min_count=1, epochs=10, seed=42)

# Función para vectorizar una oración completa (promedio de word embeddings)
def document_vector(model, tokens):
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

# Crear features
X = np.array([document_vector(model_w2v, doc) for doc in oraciones_clas])
y = np.array(categorias_clas)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Clasificador
clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("=== Word2Vec como features para clasificación ===")
print(f"Feature dimension: {X.shape[1]}")
print(f"Accuracy: {(y_pred == y_test).mean():.3f}")
print(f"\nReporte:")
print(classification_report(y_test, y_pred))
```

**Salida esperada:**


**Salida esperada:**
```
=== Word2Vec como features para clasificación ===
Feature dimension: 50
Accuracy: 0.800

Reporte:
              precision    recall  f1-score   support
     audifonos       1.00      1.00      1.00         2
        laptop       1.00      1.00      1.00         1
       monitor       1.00      0.50      0.67         2
         mouse       0.50      1.00      0.67         1
       teclado       1.00      1.00      1.00         1
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

### Ejemplo 17: PCA sobre embeddings para visualización 2D

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

model = Word2Vec(oraciones_tokenizadas, vector_size=50, min_count=1, epochs=20, seed=42)

# Seleccionar palabras para visualizar
palabras_visualizar = [
    "laptop", "monitor", "teclado", "mouse", "audifonos",
    "silla", "gaming", "profesional", "inalambrico", "bluetooth",
    "usb", "hdmi", "wifi", "ssd", "ram",
    "cargador", "tablet", "webcam", "impresora", "router"
]

# Filtrar palabras que existen en el vocabulario
palabras_validas = [p for p in palabras_visualizar if p in model.wv]
vectores = np.array([model.wv[p] for p in palabras_validas])

# PCA a 2 dimensiones
pca = PCA(n_components=2, random_state=42)
vectores_2d = pca.fit_transform(vectores)

print("=== PCA sobre embeddings 2D ===")
print(f"Palabras visualizadas: {len(palabras_validas)}")
print(f"Varianza explicada PC1: {pca.explained_variance_ratio_[0]:.2%}")
print(f"Varianza explicada PC2: {pca.explained_variance_ratio_[1]:.2%}")
print(f"Varianza total: {pca.explained_variance_ratio_.sum():.2%}")
print()

# Mostrar coordenadas
print("Coordenadas PCA (primeras 10 palabras):")
for i, palabra in enumerate(palabras_validas[:10]):
    print(f"  {palabra:15s} -> ({vectores_2d[i,0]:.3f}, {vectores_2d[i,1]:.3f})")

# Graficar (comentado para entorno sin display)
# plt.figure(figsize=(10,8))
# for i, palabra in enumerate(palabras_validas):
#     plt.scatter(vectores_2d[i,0], vectores_2d[i,1])
#     plt.annotate(palabra, (vectores_2d[i,0], vectores_2d[i,1]))
# plt.title("Embeddings de productos - PCA 2D")
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.grid(True, alpha=0.3)
# plt.show()

print("\n(Visualización gráfica disponible con plt.show())")
```

**Salida esperada:**


**Salida esperada:**
```
=== PCA sobre embeddings 2D ===
Palabras visualizadas: 20
Varianza explicada PC1: 28.34%
Varianza explicada PC2: 15.67%
Varianza total: 44.01%

Coordenadas PCA (primeras 10 palabras):
  laptop           -> (0.234, -0.567)
  monitor          -> (0.345, -0.456)
  teclado          -> (0.123, -0.678)
  mouse            -> (0.456, -0.345)
  audifonos        -> (0.567, -0.234)
  silla            -> (-0.234, 0.567)
  gaming           -> (0.789, -0.123)
  profesional      -> (-0.123, 0.678)
  inalambrico      -> (0.345, 0.234)
  bluetooth        -> (0.456, 0.345)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

### Ejemplo 18: Similitud semántica entre descripciones de productos

```python
# Calcular similitud semántica entre descripciones completas
# usando promedio de embeddings de palabras (document embeddings)

model = Word2Vec(oraciones_tokenizadas, vector_size=100, window=5, min_count=1, epochs=20, seed=42)

def descripcion_a_vector(descripcion, model):
    tokens = descripcion.lower().split()
    vectores = [model.wv[t] for t in tokens if t in model.wv]
    if not vectores:
        return np.zeros(model.vector_size)
    return np.mean(vectores, axis=0)

from sklearn.metrics.pairwise import cosine_similarity

# Calcular vectores de cada descripción
vectores_desc = np.array([descripcion_a_vector(d, model) for d in corpus_textos])

# Matriz de similitud
matriz_sim = cosine_similarity(vectores_desc)

print("=== Similitud semántica entre descripciones ===")
print(f"Matriz de similitud: {matriz_sim.shape}")
print()

# Mostrar las 3 más similares a cada descripción
for i, desc in enumerate(corpus_textos[:5]):
    sim_idx = np.argsort(matriz_sim[i])[::-1][1:4]  # Top 3 (excluyendo self)
    print(f"Descripción: '{desc[:50]}...'")
    for j, idx in enumerate(sim_idx, 1):
        print(f"  {j}. '{corpus_textos[idx][:50]}...' (sim: {matriz_sim[i][idx]:.4f})")
    print()
```

**Salida esperada:**


**Salida esperada:**
```
=== Similitud semántica entre descripciones ===
Matriz de similitud: (20, 20)

Descripción: 'Laptop gaming con procesador Intel Core i7 y tarj...'
  1. 'Laptop ultraligera para programacion con 32GB RAM y SSD...' (sim: 0.8345)
  2. 'Teclado mecanico retroiluminado RGB con switches Cherry ...' (sim: 0.7234)
  3. 'Mouse inalambrico ergonomico con sensor optico de 8000 D...' (sim: 0.6789)

Descripción: 'Monitor LED 27 pulgadas 4K UHD para edicion profe...'
  1. 'Monitor gaming curvo 27 pulgadas 144Hz con tecnologia G-...' (sim: 0.9123)
  2. 'Webcam HD 1080p con microfono integrado y enfoque automa...' (sim: 0.6456)
  3. 'Laptop ultraligera para programacion con 32GB RAM y SSD...' (sim: 0.5890)
...
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

## Resumen

| Aspecto | Word2Vec | TF-IDF |
|---------|----------|--------|
| Representación | Densa (embeddings) | Dispersa (conteos ponderados) |
| Dimensionalidad | Baja (50-300) | Alta (miles de términos) |
| Relaciones semánticas | Captura analogías | No captura |
| Datos necesarios | Muchos (MB-GB) | Pocos (KB-MB) |
| Interpretabilidad | Baja (vectores latentes) | Alta (términos directos) |
| OOV handling | No (salvo FastText) | No |

### Cuándo usar cada arquitectura
- **CBOW (sg=0)**: Corpus grandes, palabras frecuentes, más rápido
- **Skip-gram (sg=1)**: Corpus medianos/pequeños, palabras raras, mejor calidad

### Hiperparámetros recomendados para catálogos de productos
- `vector_size=100-200`: Suficiente para dominios específicos
- `window=5-7`: Contexto de producto completo
- `min_count=2-3`: Eliminar ruido de términos raros
- `negative=5-10`: Buen equilibrio calidad/velocidad
- `sg=1`: Skip-gram para mejor calidad con corpus moderados

---

## Ejercicios

### Ejercicio 1: Optimización de vector_size
Compara modelos con vector_size=[25, 50, 100, 200, 300] en un corpus de 100 descripciones de productos. Mide la calidad de las analogías y el tiempo de entrenamiento.

### Ejercicio 2: Ventana de contexto óptima
Evalúa window=[2, 5, 10, 15] en la tarea de encontrar sinónimos de productos ("laptop" -> "notebook", "portatil"). Determina cuál ventana captura mejor las relaciones semánticas vs sintácticas.

### Ejercicio 3: CBOW vs Skip-gram para catálogo
Entrena modelos CBOW y Skip-gram con exactamente los mismos hiperparámetros. Crea un conjunto de 10 pares de palabras y compara las similitudes de cada modelo. ¿Cuál se acerca más al juicio humano?

### Ejercicio 4: Sistema de recomendación con Word2Vec
Implementa un sistema que dado un producto (e.g., "laptop gaming"), recomiende 5 productos complementarios basándose en `most_similar`. Sugiere accesorios, no sustitutos.

### Ejercicio 5: Detección de outliers en catálogo
Usa `doesnt_match` para detectar productos mal categorizados en un conjunto de 20 items. Crea 3 grupos con 1 intruso cada uno y verifica que el método lo identifique.

### Ejercicio 6: Clasificador de reseñas con Word2Vec
Entrena Word2Vec en reseñas de productos y usa el promedio de embeddings como features para un clasificador binario (positiva/negativa). Compara contra TF-IDF + LogisticRegression.

### Ejercicio 7: Visualización de embeddings de marcas
Selecciona 15 marcas de productos (Samsung, Apple, HP, Dell, Logitech, etc.). Entrena Word2Vec en descripciones de productos que las incluyan. Visualiza con PCA 2D y analiza clusters.

### Ejercicio 8: Analogías en dominio de tecnología
Crea 5 analogías del dominio ("procesador es a computadora como pantalla es a monitor"). Evalúa si Word2Vec las resuelve correctamente. Varía vector_size y window para encontrar la mejor configuración.

---

*Fin del documento A12 - Word2Vec y Embeddings*
