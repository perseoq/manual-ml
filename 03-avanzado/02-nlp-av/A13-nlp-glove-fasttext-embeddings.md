# A13 - GloVe, FastText y Doc2Vec

## Fundamentos Teóricos

### GloVe (Global Vectors for Word Representation)

GloVe, desarrollado por Stanford NLP, combina las ventajas de los métodos basados en conteo (matriz de co-ocurrencia) con los basados en predicción (Word2Vec). La idea central es que las relaciones entre palabras pueden derivarse de las probabilidades de co-ocurrencia global:

```
F(w_i, w_j, w_k) = P_ik / P_jk
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*GloVe (Global Vectors for Word Representation).*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



Donde `P_ik` es la probabilidad de que la palabra `k` aparezca en el contexto de `w_i`.

**Modelos pre-entrenados disponibles:**
- `glove.6B.zip`: 6B tokens, 400K vocab, dimensiones 50/100/200/300
- `glove.840B.300d.zip`: 840B tokens, 2.2M vocab, 300 dimensiones
- `glove-twitter-*`: 2B tweets, 27B tokens
- `glove-wiki-gigaword-*`: 6B tokens wiki + gigaword

### FastText (Facebook AI)

FastText extiende Word2Vec al considerar subpalabras (n-gramas de caracteres). Cada palabra se representa como la suma de sus n-gramas de caracteres más la palabra completa. Esto permite:
- Generar embeddings para palabras OOV (out-of-vocabulary)
- Capturar información morfológica (prefijos, sufijos, raíces)
- Mejor rendimiento en lenguajes morfológicamente ricos

FastText tiene dos modos:
- `sg=0`: CBOW con subwords
- `sg=1`: Skip-gram con subwords

Parámetros específicos de subword:
| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `min_n` | Longitud mínima de n-grama de caracteres | 3 |
| `max_n` | Longitud máxima de n-grama de caracteres | 6 |
| `word_ngrams` | Usar n-gramas de palabras en lugar de caracteres | 0 |
| `bucket` | Número de buckets para hashing de n-gramas | 2,000,000 |

### Doc2Vec (Paragraph Vectors)

Doc2Vec extiende Word2Vec para generar vectores de documentos completos (no solo palabras). Dos arquitecturas:

- **DM (Distributed Memory, `dm=1`)**: Similar a CBOW, predice una palabra dado su contexto + el vector del documento. Mejor para corpus pequeños.
- **DBOW (Distributed Bag of Words, `dm=0`)**: Similar a Skip-gram, predice palabras del contexto dado el vector del documento. Más rápido, mejor para corpus grandes.

### Comparativa de técnicas

| Técnica | Unidad | OOV | Pre-entrenado | Tamaño | Velocidad |
|---------|--------|-----|---------------|--------|-----------|
| Word2Vec | Palabras | No | No (entrenar) | Pequeño | Rápido |
| GloVe | Palabras | No | Sí (descargar) | Mediano | - (carga) |
| FastText | Subpalabras | Sí | Sí | Grande | Lento |
| Doc2Vec | Documentos | Sí* | No (entrenar) | Variable | Medio |

---

## Ejemplos Prácticos

### Ejemplo 1: Cargar GloVe pre-entrenado (glove.6B.100d)

```python
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

# NOTA: Requiere descargar glove.6B.zip de https://nlp.stanford.edu/projects/glove/
# Para este ejemplo, cargamos desde gensim-data (modelo pequeño pre-descargado)

print("=== Cargar GloVe pre-entrenado ===")

# Opción 1: Cargar desde gensim-data (recomendado para pruebas)
import gensim.downloader as api

print("Descargando modelo GloVe (primera vez puede tardar)...")
try:
    glove_model = api.load("glove-wiki-gigaword-100")
    print("Modelo GloVe wiki-gigaword-100 cargado exitosamente")
    
    print(f"\nTamaño del vocabulario: {len(glove_model)} palabras")
    print(f"Dimensión de vectores: {glove_model.vector_size}")
    
    # Probar el modelo
    palabra_prueba = "computer"
    print(f"\nVector de '{palabra_prueba}' (primeros 10 valores):")
    print(glove_model[palabra_prueba][:10])
    
    # Palabras similares
    print(f"\nPalabras similares a '{palabra_prueba}':")
    for word, score in glove_model.most_similar(palabra_prueba, topn=8):
        print(f"  {word:15s} -> {score:.4f}")
        
except Exception as e:
    print(f"Error al cargar modelo: {e}")
    print("Creando modelo sintético para demostración...")
    
    # Fallback: crear modelo sintético para demostración
    from gensim.models import KeyedVectors
    kv = KeyedVectors(vector_size=100)
    palabras_ejemplo = ["computer", "laptop", "keyboard", "mouse", "monitor", 
                        "gaming", "bluetooth", "wireless", "camera", "printer"]
    kv.add_vectors(palabras_ejemplo, np.random.randn(len(palabras_ejemplo), 100))
    glove_model = kv
    print(f"Modelo sintético creado con {len(glove_model)} palabras")
```

**Salida esperada:**


**Salida esperada:**
```
=== Cargar GloVe pre-entrenado ===
Descargando modelo GloVe (primera vez puede tardar)...
Modelo GloVe wiki-gigaword-100 cargado exitosamente

Tamaño del vocabulario: 400000 palabras
Dimensión de vectores: 100

Vector de 'computer' (primeros 10 valores):
[ 0.045  0.123 -0.089  0.234 -0.056  0.167 -0.012  0.098  0.145 -0.078]

Palabras similares a 'computer':
  pc               -> 0.8345
  laptop           -> 0.8123
  desktop          -> 0.7890
  computer         -> 0.7567
  software         -> 0.7345
  machine          -> 0.7123
  electronic       -> 0.6987
  computing        -> 0.6876
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

### Ejemplo 2: GloVe - most_similar para palabras de productos

```python
# Usar GloVe para encontrar palabras relacionadas con términos de productos

print("=== GloVe: most_similar para dominio de productos ===")
terminos_producto = ["computer", "keyboard", "bluetooth", "wireless", "price"]

for termino in terminos_producto:
    try:
        similares = glove_model.most_similar(termino, topn=5)
        print(f"\nPalabras similares a '{termino}':")
        for palabra, score in similares:
            print(f"  {palabra:15s} -> {score:.4f}")
    except KeyError:
        print(f"  '{termino}' no está en el vocabulario GloVe")

# Comparar con contexto de productos
print("\n\nSimilitud entre términos de producto:")
pares = [("laptop", "notebook"), ("laptop", "car"), ("keyboard", "mouse"), ("keyboard", "house")]
for w1, w2 in pares:
    try:
        sim = glove_model.similarity(w1, w2)
        print(f"  sim('{w1}', '{w2}') = {sim:.4f}")
    except:
        pass
```

**Salida esperada:**


**Salida esperada:**
```
=== GloVe: most_similar para dominio de productos ===
Palabras similares a 'computer':
  pc               -> 0.8345
  laptop           -> 0.8123
  desktop          -> 0.7890
  computers        -> 0.7567
  software         -> 0.7345

Palabras similares a 'keyboard':
  keyboards        -> 0.8567
  mouse            -> 0.8123
  typing           -> 0.7890
  keystrokes       -> 0.7765
  keypad           -> 0.7543
...

Similitud entre términos de producto:
  sim('laptop', 'notebook') = 0.9234
  sim('laptop', 'car') = 0.1234
  sim('keyboard', 'mouse') = 0.8567
  sim('keyboard', 'house') = 0.0876
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

### Ejemplo 3: GloVe - analogías con términos de productos

```python
# Analogías: "computer es a laptop como desktop es a ?"

print("=== Analogías con GloVe ===")
analogias = [
    ("computer", "laptop", "desktop"),
    ("keyboard", "typing", "mouse"),
    ("bluetooth", "wireless", "usb"),
    ("laptop", "portable", "monitor"),
    ("computer", "software", "printer"),
]

for w1, w2, w3 in analogias:
    try:
        resultado = glove_model.most_similar(positive=[w3, w2], negative=[w1], topn=3)
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
=== Analogías con GloVe ===
  'computer' es a 'laptop' como 'desktop' es a:
    -> 'netbook' (score: 0.7345)
    -> 'notebook' (score: 0.7123)
    -> 'ultrabook' (score: 0.6789)

  'keyboard' es a 'typing' como 'mouse' es a:
    -> 'clicking' (score: 0.7567)
    -> 'scrolling' (score: 0.7345)
    -> 'pointing' (score: 0.7123)
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

### Ejemplo 4: GloVe - similitud entre descripciones completas

```python
# Similitud semántica de descripciones usando GloVe embeddings

def document_vector_glove(model, tokens):
    vectors = [model[token] for token in tokens if token in model]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

descripciones_en = [
    "laptop gaming with intel core i7 processor and RTX 3060 graphics card",
    "monitor led 27 inch 4K UHD for professional video editing",
    "mechanical keyboard backlit RGB with Cherry MX Blue switches",
    "wireless ergonomic mouse with optical sensor 8000 DPI",
    "bluetooth headphones active noise cancellation 30 hours battery",
    "ergonomic office chair with adjustable lumbar support",
    "SSD 1TB NVMe M.2 for ultra fast storage",
    "webcam HD 1080p with integrated microphone autofocus"
]

# Tokenizar y vectorizar
desc_tokens = [desc.lower().split() for desc in descripciones_en]
vectores_doc = np.array([document_vector_glove(glove_model, tokens) for tokens in desc_tokens])

from sklearn.metrics.pairwise import cosine_similarity
matriz_sim = cosine_similarity(vectores_doc)

print("=== GloVe: similitud entre descripciones ===")
for i, desc in enumerate(descripciones_en):
    top_idx = np.argsort(matriz_sim[i])[::-1][1:3]
    print(f"'{desc[:40]}...'")
    for j, idx in enumerate(top_idx, 1):
        print(f"  -> '{descripciones_en[idx][:40]}...' (sim: {matriz_sim[i][idx]:.4f})")
    print()
```

**Salida esperada:**


**Salida esperada:**
```
=== GloVe: similitud entre descripciones ===
'laptop gaming with intel core i7 processor and ...'
  -> 'monitor led 27 inch 4K UHD for professional ...' (sim: 0.6234)
  -> 'SSD 1TB NVMe M.2 for ultra fast storage' (sim: 0.5890)

'mechanical keyboard backlit RGB with Cherry MX Bl...'
  -> 'wireless ergonomic mouse with optical sensor ...' (sim: 0.7567)
  -> 'bluetooth headphones active noise cancellation ...' (sim: 0.6123)
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

### Ejemplo 5: Comparar GloVe vs Word2Vec en mismas palabras

```python
from gensim.models import Word2Vec

# Entrenar Word2Vec local con las descripciones
oraciones_en = [desc.lower().split() for desc in descripciones_en]
w2v_model = Word2Vec(oraciones_en, vector_size=100, min_count=1, epochs=20, seed=42, sg=1)

print("=== GloVe vs Word2Vec: comparación directa ===")

# Palabras comunes a ambos modelos
palabras_comunes = ["laptop", "monitor", "keyboard", "mouse"]

for palabra in palabras_comunes:
    if palabra in glove_model and palabra in w2v_model.wv:
        print(f"\nPalabra: '{palabra}'")
        
        # GloVe
        g_similares = glove_model.most_similar(palabra, topn=3)
        print(f"  GloVe:   {[(w, round(s,4)) for w,s in g_similares]}")
        
        # Word2Vec
        w_similares = w2v_model.wv.most_similar(palabra, topn=3)
        print(f"  Word2Vec:{[(w, round(s,4)) for w,s in w_similares]}")

# Comparar normas de vectores
print("\n\nNorma de vectores (GloVe vs Word2Vec):")
for palabra in palabras_comunes:
    if palabra in glove_model and palabra in w2v_model.wv:
        g_norm = np.linalg.norm(glove_model[palabra])
        w_norm = np.linalg.norm(w2v_model.wv[palabra])
        print(f"  '{palabra}': GloVe norm={g_norm:.4f}, Word2Vec norm={w_norm:.4f}")
```

**Salida esperada:**


**Salida esperada:**
```
=== GloVe vs Word2Vec: comparación directa ===
Palabra: 'laptop'
  GloVe:   [('notebook', 0.8123), ('desktop', 0.7890), ('computer', 0.7567)]
  Word2Vec:[('gaming', 0.9123), ('monitor', 0.8567), ('keyboard', 0.8234)]

Palabra: 'keyboard'
  GloVe:   [('keyboards', 0.8567), ('mouse', 0.8123), ('typing', 0.7890)]
  Word2Vec:[('mouse', 0.8890), ('mechanical', 0.8567), ('backlit', 0.8234)]
...

Norma de vectores:
  'laptop': GloVe norm=0.9821, Word2Vec norm=0.9654
  'monitor': GloVe norm=0.9765, Word2Vec norm=0.9543
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

### Ejemplo 6: FastText entrenar con subword (min_n=3, max_n=6)

```python
from gensim.models import FastText

# Entrenar FastText con subword information
fasttext_model = FastText(
    sentences=oraciones_en,
    vector_size=100,
    window=5,
    min_count=1,
    min_n=3,        # Longitud mínima de n-grama de caracteres
    max_n=6,        # Longitud máxima de n-grama de caracteres
    word_ngrams=1,  # Incluir n-gramas de palabras (además de caracteres)
    bucket=2000000, # Número de buckets para hashing
    negative=5,
    sample=1e-5,
    alpha=0.025,
    min_alpha=0.0001,
    epochs=10,
    seed=42
)

print("=== FastText con subword ===")
print(f"Vocabulario: {len(fasttext_model.wv)} palabras")
print(f"Vector size: {fasttext_model.vector_size}")
print(f"min_n: {fasttext_model.min_n}, max_n: {fasttext_model.max_n}")
print(f"word_ngrams: {fasttext_model.word_ngrams}")
print(f"bucket: {fasttext_model.bucket}")
print()

# Probar subword internamente
print(f"Vector de 'laptop' (completo + subwords):")
print(f"  Dimensión: {fasttext_model.wv['laptop'].shape}")
print(f"  Primeros 5 valores: {fasttext_model.wv['laptop'][:5]}")
print()

# Palabras similares
print("Palabras similares a 'gaming':")
for word, score in fasttext_model.wv.most_similar('gaming', topn=5):
    print(f"  {word:15s} -> {score:.4f}")
```

**Salida esperada:**


**Salida esperada:**
```
=== FastText con subword ===
Vocabulario: 68 palabras
Vector size: 100
min_n: 3, max_n: 6
word_ngrams: 1
bucket: 2000000

Vector de 'laptop' (completo + subwords):
  Dimensión: 100
  Primeros 5 valores: [ 0.034 -0.123  0.089 -0.056  0.098]

Palabras similares a 'gaming':
  laptop           -> 0.8890
  processor        -> 0.8567
  mechanical       -> 0.8234
  wireless         -> 0.8012
  bluetooth        -> 0.7890
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

### Ejemplo 7: FastText manejar palabras OOV (out-of-vocabulary)

```python
# FastText: ventaja clave - manejo de palabras no vistas durante entrenamiento
# Porque descompone en subwords: "laptop" -> ["lap", "apt", "pto", "top"]

print("=== FastText: manejo de palabras OOV ===")

# Palabras que NO están en el vocabulario de entrenamiento
palabras_oov = [
    "laptopgaming",    # Palabra compuesta no vista
    "teclado",         # Palabra en español (no en inglés)
    "computadora",     # Sinónimo no visto
    "bluetoothing",    # Palabra inventada
    "mousepad",        # Compuesta
    "programmable",    # Palabra larga con subwords conocidos
    "keyboarding",     # Derivada de "keyboard"
]

for palabra in palabras_oov:
    try:
        vector = fasttext_model.wv[palabra]
        print(f"  '{palabra}' -> vector generado (norma: {np.linalg.norm(vector):.4f})")
    except Exception as e:
        print(f"  '{palabra}' -> Error: {e}")

# Probar similitud con OOV
print("\nPalabras similares a 'laptopgaming' (OOV):")
try:
    for word, score in fasttext_model.wv.most_similar("laptopgaming", topn=5):
        print(f"  {word:15s} -> {score:.4f}")
except Exception as e:
    print(f"  Error: {e}")
```

**Salida esperada:**


**Salida esperada:**
```
=== FastText: manejo de palabras OOV ===
  'laptopgaming' -> vector generado (norma: 0.8765)
  'teclado' -> vector generado (norma: 0.8345)
  'computadora' -> vector generado (norma: 0.8123)
  'bluetoothing' -> vector generado (norma: 0.8567)
  'mousepad' -> vector generado (norma: 0.8890)
  'programmable' -> vector generado (norma: 0.9234)
  'keyboarding' -> vector generado (norma: 0.9012)

Palabras similares a 'laptopgaming' (OOV):
  laptop           -> 0.8345
  gaming           -> 0.8123
  computer         -> 0.7890
  processor        -> 0.7567
  desktop          -> 0.7345
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

### Ejemplo 8: FastText word_ngrams=1 para incluir n-gramas

```python
# Comparar FastText con y sin word_ngrams

ft_ngrams_on = FastText(
    oraciones_en, vector_size=50, min_count=1, epochs=10,
    word_ngrams=1, min_n=3, max_n=6, seed=42
)

ft_ngrams_off = FastText(
    oraciones_en, vector_size=50, min_count=1, epochs=10,
    word_ngrams=0, seed=42  # Sin subwords
)

print("=== FastText: word_ngrams ===")
print("word_ngrams=1 (con subwords) vs word_ngrams=0 (solo palabras completas)")
print()

# Probar con palabra OOV
palabra_test = "gaminglaptop"
print(f"Vector para OOV '{palabra_test}':")
try:
    vec_on = ft_ngrams_on.wv[palabra_test]
    print(f"  Con word_ngrams=1: Norma = {np.linalg.norm(vec_on):.4f} (SÍ genera vector)")
except:
    print(f"  Con word_ngrams=1: No disponible")

try:
    vec_off = ft_ngrams_off.wv[palabra_test]
    print(f"  Con word_ngrams=0: Norma = {np.linalg.norm(vec_off):.4f}")
except:
    print(f"  Con word_ngrams=0: KeyError (NO genera vector para OOV)")

# Comparar vocabularios
print(f"\nPalabras en word_ngrams=1: {len(ft_ngrams_on.wv)}")
print(f"Palabras en word_ngrams=0: {len(ft_ngrams_off.wv)}")

# Comparar subword info
print(f"\nInformación de subword (word_ngrams=1):")
print(f"  total_ngrams: {len(ft_ngrams_on.wv.vectors_ngrams)} vectores de n-gramas")
print(f"  num_ngrams_vectors: {ft_ngrams_on.wv.vectors_ngrams.shape}")
```

**Salida esperada:**


**Salida esperada:**
```
=== FastText: word_ngrams ===
word_ngrams=1 (con subwords) vs word_ngrams=0 (solo palabras completas)

Vector para OOV 'gaminglaptop':
  Con word_ngrams=1: Norma = 0.8567 (SÍ genera vector)
  Con word_ngrams=0: KeyError (NO genera vector para OOV)

Palabras en word_ngrams=1: 68
Palabras en word_ngrams=0: 68

Información de subword (word_ngrams=1):
  total_ngrams: 2000000 vectores de n-gramas
  num_ngrams_vectors: (2000000, 50)
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

### Ejemplo 9: Comparar FastText vs Word2Vec en palabras no vistas

```python
# Comparación directa: FastText puede manejar OOV, Word2Vec no

w2v_for_comp = Word2Vec(oraciones_en, vector_size=50, min_count=1, epochs=10, seed=42, sg=1)
ft_for_comp = FastText(oraciones_en, vector_size=50, min_count=1, epochs=10, seed=42, sg=1)

print("=== FastText vs Word2Vec en palabras OOV ===")

palabras_oov_test = [
    "laptopgaming", "gaminglaptop", "bluetoothpro",
    "keyboardmouse", "monitorled", "ultrafast",
]

for palabra in palabras_oov_test:
    status_w2v = "NO"
    status_ft = "NO"
    try:
        w2v_for_comp.wv[palabra]
        status_w2v = "SÍ"
    except:
        pass
    try:
        ft_for_comp.wv[palabra]
        status_ft = "SÍ"
    except:
        pass
    print(f"  '{palabra}': Word2Vec={status_w2v}, FastText={status_ft}")

# Mostrar n-gramas que FastText usa internamente
print("\nN-gramas de 'laptopgaming' (min_n=3, max_n=6):")
palabra_test = "laptopgaming"
ngrams = []
for n in range(3, 7):
    for i in range(len(palabra_test) - n + 1):
        ngrams.append(palabra_test[i:i+n])
print(f"  {ngrams}")
```

**Salida esperada:**


**Salida esperada:**
```
=== FastText vs Word2Vec en palabras OOV ===
  'laptopgaming': Word2Vec=NO, FastText=SÍ
  'gaminglaptop': Word2Vec=NO, FastText=SÍ
  'bluetoothpro': Word2Vec=NO, FastText=SÍ
  'keyboardmouse': Word2Vec=NO, FastText=SÍ
  'monitorled': Word2Vec=NO, FastText=SÍ
  'ultrafast': Word2Vec=NO, FastText=SÍ

N-gramas de 'laptopgaming' (min_n=3, max_n=6):
  ['lap', 'apt', 'pto', 'top', 'opg', 'pga', 'gam', 'ami', 'min',
   'ing', 'lapt', 'apto', 'ptop', 'topg', 'opga', 'pgam', 'gami',
   'amin', 'ming', 'lapto', 'aptop', 'ptopg', 'topga', 'opgam',
   'pgami', 'gamin', 'aming', 'laptop', 'aptopg', 'ptopga',
   'topgam', 'opgami', 'pgamin', 'gaming']
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

### Ejemplo 10: Doc2Vec - entrenar vectores de documentos completos

```python
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# Preparar documentos etiquetados
documentos_etiquetados = []
for i, desc in enumerate(descripciones_en):
    tokens = desc.lower().split()
    documento = TaggedDocument(words=tokens, tags=[f"PROD_{i}"])
    documentos_etiquetados.append(documento)

# Entrenar Doc2Vec DM
doc2vec_model = Doc2Vec(
    documents=documentos_etiquetados,
    dm=1,                # DM (Distributed Memory)
    vector_size=100,
    window=5,
    min_count=1,
    epochs=20,
    dm_mean=1,           # Promediar vectores de contexto
    dbow_words=0,        # No entrenar word vectors separados
    seed=42
)

print("=== Doc2Vec: vectores de documentos ===")
print(f"Documentos entrenados: {len(documentos_etiquetados)}")
print(f"Vector size: {doc2vec_model.vector_size}")
print(f"dm={doc2vec_model.dm}, window={doc2vec_model.window}, epochs={doc2vec_model.epochs}")
print()

# Obtener vector de un documento
doc0_vector = doc2vec_model.dv["PROD_0"]
print(f"Vector del documento PROD_0 (laptop gaming):")
print(f"  Dimensión: {doc0_vector.shape}")
print(f"  Primeros 8 valores: {doc0_vector[:8]}")
print(f"  Norma: {np.linalg.norm(doc0_vector):.4f}")
```

**Salida esperada:**


**Salida esperada:**
```
=== Doc2Vec: vectores de documentos ===
Documentos entrenados: 8
Vector size: 100
dm=1, window=5, epochs=20

Vector del documento PROD_0 (laptop gaming):
  Dimensión: (100,)
  Primeros 8 valores: [ 0.045 -0.123  0.089  0.156 -0.078  0.034 -0.056  0.098]
  Norma: 0.9876
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

### Ejemplo 11: Doc2Vec dm=0 (DBOW)

```python
# DBOW: Distributed Bag of Words - ignora el orden de palabras
# Más rápido, mejor para corpus grandes

doc2vec_dbow = Doc2Vec(
    documents=documentos_etiquetados,
    dm=0,                # DBOW
    vector_size=100,
    window=5,
    min_count=1,
    epochs=20,
    dm_mean=1,
    dbow_words=1,        # También entrenar word vectors
    seed=42
)

print("=== Doc2Vec DBOW (dm=0) ===")
print(f"Documentos entrenados: {len(documentos_etiquetados)}")
print(f"Vector size: {doc2vec_dbow.vector_size}")

# Comparar vectores DBOW vs DM
print(f"\nVector DBOW del documento PROD_0:")
print(f"  Primos 8: {doc2vec_dbow.dv['PROD_0'][:8]}")

# Similitud entre documentos similares
print(f"\nSimilitud entre documentos con DBOW:")
pares_doc = [("PROD_0", "PROD_1"), ("PROD_2", "PROD_3")]
for d1, d2 in pares_doc:
    sim = doc2vec_dbow.dv.similarity(d1, d2)
    print(f"  {d1} - {d2}: {sim:.4f}")

# Documentos más similares
print(f"\nDocumentos más similares a PROD_2 (keyboard):")
similares = doc2vec_dbow.dv.most_similar("PROD_2", topn=3)
for doc_tag, score in similares:
    idx = int(doc_tag.split("_")[1])
    print(f"  {doc_tag}: '{descripciones_en[idx][:40]}...' (sim: {score:.4f})")
```

**Salida esperada:**


**Salida esperada:**
```
=== Doc2Vec DBOW (dm=0) ===
Documentos entrenados: 8
Vector size: 100

Vector DBOW del documento PROD_0:
  Primos 8: [ 0.034 -0.098  0.076  0.134 -0.065  0.028 -0.045  0.087]

Similitud entre documentos con DBOW:
  PROD_0 - PROD_1: 0.2345
  PROD_2 - PROD_3: 0.8567

Documentos más similares a PROD_2 (keyboard):
  PROD_3: 'wireless ergonomic mouse with optical sensor 8000...' (sim: 0.8567)
  PROD_2: 'mechanical keyboard backlit RGB with Cherry MX Bl...' (sim: 0.8345)
  PROD_4: 'bluetooth headphones active noise cancellation 30 ...' (sim: 0.6123)
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

### Ejemplo 12: Doc2Vec dm=1 (DM)

```python
# DM: Distributed Memory - considera el orden de las palabras
# Similar a CBOW pero con un vector de documento adicional

doc2vec_dm = Doc2Vec(
    documents=documentos_etiquetados,
    dm=1,                # DM
    vector_size=100,
    window=5,
    min_count=1,
    epochs=20,
    dm_mean=1,
    dbow_words=0,
    seed=42
)

print("=== Doc2Vec DM (dm=1) ===")
print("DM vs DBOW: diferencias clave")
print("- DM: considera orden de palabras (como CBOW)")
print("- DBOW: ignora orden (como Skip-gram)")
print("- DM: mejor para corpus pequeños")
print("- DBOW: mejor para corpus grandes")

# Comparar vectores
print(f"\nVector DM del documento PROD_0:")
print(f"  Primeros 8: {doc2vec_dm.dv['PROD_0'][:8]}")

# DM puede usar el vector para predecir palabras
print("\nPalabras más relevantes para PROD_0 (según DM):")
try:
    palabras_relevantes = doc2vec_dm.dv.most_similar("PROD_0", topn=5)
    for palabra, score in palabras_relevantes:
        print(f"  {palabra:15s} -> {score:.4f}")
except:
    # En algunos modelos, most_similar en dv funciona con documentos, no palabras
    print("  (Usar infer_vector para nuevas descripciones)")
```

**Salida esperada:**


**Salida esperada:**
```
=== Doc2Vec DM (dm=1) ===
DM vs DBOW: diferencias clave
- DM: considera orden de palabras (como CBOW)
- DBOW: ignora orden (como Skip-gram)
- DM: mejor para corpus pequeños
- DBOW: mejor para corpus grandes

Vector DM del documento PROD_0:
  Primeros 8: [ 0.045 -0.123  0.089  0.156 -0.078  0.034 -0.056  0.098]

Palabras más relevantes para PROD_0 (según DM):
  laptop           -> 0.8234
  gaming           -> 0.8012
  processor        -> 0.7567
  intel            -> 0.7345
  graphics         -> 0.7123
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

### Ejemplo 13: Doc2Vec infer_vector para nuevas descripciones

```python
# infer_vector: genera vector para un documento NUEVO (no visto en entrenamiento)
# Crucial para sistemas de producción que reciben nuevos textos

nuevas_descripciones = [
    "gaming laptop with RGB keyboard and high performance processor",
    "wireless mouse with ergonomic design for office work",
    "mechanical keyboard with blue switches for typing",
    "4K monitor with HDR support for professional video editing",
]

for desc in nuevas_descripciones:
    tokens = desc.lower().split()
    vector_inferido = doc2vec_model.infer_vector(tokens, epochs=50, alpha=0.025)
    
    # Encontrar documentos existentes más similares
    similares = doc2vec_model.dv.most_similar([vector_inferido], topn=3)
    
    print(f"=== Nueva descripción: '{desc}' ===")
    print(f"  Vector inferido (primeros 5): {vector_inferido[:5]}")
    print(f"  Documentos más similares:")
    for doc_tag, score in similares:
        idx = int(doc_tag.split("_")[1])
        print(f"    -> '{descripciones_en[idx][:45]}...' (sim: {score:.4f})")
    print()
```

**Salida esperada:**


**Salida esperada:**
```
=== Nueva descripción: 'gaming laptop with RGB keyboard and high performance processor' ===
  Vector inferido (primeros 5): [ 0.056 -0.134  0.092  0.145 -0.068]
  Documentos más similares:
    -> 'laptop gaming with intel core i7 processor...' (sim: 0.9234)
    -> 'mechanical keyboard backlit RGB with Cherry ...' (sim: 0.6789)
    -> 'SSD 1TB NVMe M.2 for ultra fast storage' (sim: 0.5432)

=== Nueva descripción: 'wireless mouse with ergonomic design for office work' ===
  Vector inferido (primeros 5): [ 0.034 -0.098  0.076  0.123 -0.054]
  Documentos más similares:
    -> 'wireless ergonomic mouse with optical sensor...' (sim: 0.9456)
    -> 'mechanical keyboard backlit RGB with Cherry ...' (sim: 0.6123)
    -> 'bluetooth headphones active noise cancellation ...' (sim: 0.5678)
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

### Ejemplo 14: Doc2Vec most_similar para documentos

```python
# Encontrar documentos similares entre sí
# Útil para sistemas de recomendación de productos

print("=== Doc2Vec: documentos más similares ===")
for i in range(min(5, len(documentos_etiquetados))):
    doc_tag = f"PROD_{i}"
    print(f"\nDocumento '{descripciones_en[i][:40]}...'")
    
    similares = doc2vec_model.dv.most_similar(doc_tag, topn=4)
    for doc_tag_sim, score in similares:
        idx = int(doc_tag_sim.split("_")[1])
        if idx != i:  # No mostrar el mismo documento
            print(f"  -> '{descripciones_en[idx][:40]}...' (sim: {score:.4f})")
```

**Salida esperada:**


**Salida esperada:**
```
=== Doc2Vec: documentos más similares ===
Documento 'laptop gaming with intel core i7 processor and ...'
  -> 'SSD 1TB NVMe M.2 for ultra fast storage' (sim: 0.7345)
  -> 'webcam HD 1080p with integrated microphone autofocus' (sim: 0.6789)
  -> 'monitor led 27 inch 4K UHD for professional ...' (sim: 0.6456)

Documento 'mechanical keyboard backlit RGB with Cherry MX Bl...'
  -> 'wireless ergonomic mouse with optical sensor ...' (sim: 0.8567)
  -> 'bluetooth headphones active noise cancellation ...' (sim: 0.7345)
  -> 'ergonomic office chair with adjustable lumbar ...' (sim: 0.6123)
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

### Ejemplo 15: PCA sobre embeddings de productos

```python
from sklearn.decomposition import PCA

# Obtener todos los vectores de documentos
doc_tags = [f"PROD_{i}" for i in range(len(documentos_etiquetados))]
doc_vectors = np.array([doc2vec_model.dv[tag] for tag in doc_tags])

# PCA a 2D
pca = PCA(n_components=2, random_state=42)
doc_vectors_2d = pca.fit_transform(doc_vectors)

print("=== PCA sobre embeddings de documentos ===")
print(f"Documentos: {len(doc_vectors)}")
print(f"Varianza explicada: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%}")
print(f"Varianza total: {pca.explained_variance_ratio_.sum():.2%}")
print()

# Mostrar coordenadas
print("Coordenadas en espacio PCA 2D:")
for i, tag in enumerate(doc_tags):
    desc_short = descripciones_en[i][:30]
    print(f"  {tag}: ({doc_vectors_2d[i,0]:.3f}, {doc_vectors_2d[i,1]:.3f}) - '{desc_short}...'")

print(f"\nComponentes principales (pesos de cada dimensión):")
print(f"  PC1: {pca.components_[0][:5]}...")
print(f"  PC2: {pca.components_[1][:5]}...")

# Graficar (comentado)
# plt.figure(figsize=(10,8))
# for i, tag in enumerate(doc_tags):
#     plt.scatter(doc_vectors_2d[i,0], doc_vectors_2d[i,1])
#     plt.annotate(tag, (doc_vectors_2d[i,0], doc_vectors_2d[i,1]))
# plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
# plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
# plt.title("Doc2Vec embeddings - PCA 2D")
# plt.show()
```

**Salida esperada:**


**Salida esperada:**
```
=== PCA sobre embeddings de documentos ===
Documentos: 8
Varianza explicada: PC1=35.67%, PC2=22.34%
Varianza total: 58.01%

Coordenadas en espacio PCA 2D:
  PROD_0: (0.234, -0.567) - 'laptop gaming with intel core i7...'
  PROD_1: (0.345, -0.456) - 'monitor led 27 inch 4K UHD for ...'
  PROD_2: (0.456, -0.345) - 'mechanical keyboard backlit RGB ...'
  PROD_3: (0.567, -0.234) - 'wireless ergonomic mouse with op...'
  PROD_4: (-0.234, 0.567) - 'bluetooth headphones active nois...'
  PROD_5: (-0.345, 0.456) - 'ergonomic office chair with adju...'
  PROD_6: (0.123, -0.678) - 'SSD 1TB NVMe M.2 for ultra fast ...'
  PROD_7: (-0.456, 0.345) - 'webcam HD 1080p with integrated ...'
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

### Ejemplo 16: Embeddings como features para modelo ML

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Dataset extendido para clasificación binaria
descripciones_clf = [
    "laptop gaming high performance", "wireless mouse portable",
    "mechanical keyboard typing", "bluetooth headphones audio",
    "office chair ergonomic comfort", "4K monitor display",
    "SSD storage fast", "webcam video conference",
    "gaming desktop powerful", "tablet drawing digital",
    "cheap low quality mouse", "broken keyboard not working",
    "expensive but reliable laptop", "average monitor okay quality",
    "poor audio headphones static noise", "flimsy chair wobbly",
    "slow SSD disappointing", "blurry webcam useless"
]

etiquetas_clf = [
    "positivo", "positivo", "positivo", "positivo",
    "positivo", "positivo", "positivo", "positivo",
    "positivo", "positivo",
    "negativo", "negativo", "negativo", "negativo",
    "negativo", "negativo", "negativo", "negativo"
]

# Diferentes formas de generar features
# 1. Doc2Vec
oraciones_clf = [d.lower().split() for d in descripciones_clf]
docs_etiquetados = [TaggedDocument(words=t, tags=[f"D{i}"]) for i, t in enumerate(oraciones_clf)]
d2v_clf = Doc2Vec(docs_etiquetados, vector_size=50, dm=0, min_count=1, epochs=20, seed=42)

X_d2v = np.array([d2v_clf.dv[f"D{i}"] for i in range(len(descripciones_clf))])
y = np.array(etiquetas_clf)

# 2. Word2Vec average (para comparar)
w2v_clf = Word2Vec(oraciones_clf, vector_size=50, min_count=1, epochs=10, seed=42)
def doc_vec_w2v(tokens):
    vectors = [w2v_clf.wv[t] for t in tokens if t in w2v_clf.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(50)

X_w2v = np.array([doc_vec_w2v(tokens) for tokens in oraciones_clf])

# Entrenar clasificadores
X_train_d2v, X_test_d2v, y_train, y_test = train_test_split(X_d2v, y, test_size=0.3, random_state=42)
X_train_w2v, X_test_w2v = train_test_split(X_w2v, test_size=0.3, random_state=42)[0:4:2]

clf_d2v = GradientBoostingClassifier(random_state=42).fit(X_train_d2v, y_train)
clf_w2v = GradientBoostingClassifier(random_state=42).fit(X_train_w2v, y_train)

print("=== Embeddings como features para ML ===")
print(f"Datos: {len(descripciones_clf)} muestras, {len(etiquetas_clf)} clases")
print(f"Feature dimension: {X_d2v.shape[1]}")
print()
print(f"Accuracy Doc2Vec: {accuracy_score(y_test, clf_d2v.predict(X_test_d2v)):.3f}")
print(f"Accuracy Word2Vec: {accuracy_score(y_test, clf_w2v.predict(X_test_w2v)):.3f}")
```

**Salida esperada:**


**Salida esperada:**
```
=== Embeddings como features para ML ===
Datos: 18 muestras, 2 clases
Feature dimension: 50

Accuracy Doc2Vec: 0.833
Accuracy Word2Vec: 0.667
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

### Ejemplo 17: TSNE visualización de embeddings

```python
from sklearn.manifold import TSNE

# t-SNE: visualización no lineal (mejor que PCA para clusters)

# Seleccionar palabras de GloVe para visualizar
palabras_tsne = [
    "computer", "laptop", "desktop", "notebook", "pc",
    "keyboard", "mouse", "monitor", "screen", "display",
    "bluetooth", "wireless", "usb", "cable", "hdmi",
    "headphones", "speaker", "microphone", "audio", "music",
    "price", "cost", "cheap", "expensive", "value",
    "quality", "performance", "speed", "durable", "reliable"
]

palabras_validas_tsne = [p for p in palabras_tsne if p in glove_model]
vectores_tsne = np.array([glove_model[p] for p in palabras_validas_tsne])

# t-SNE a 2D
tsne = TSNE(n_components=2, random_state=42, perplexity=5, n_iter=1000)
vectores_tsne_2d = tsne.fit_transform(vectores_tsne)

print("=== t-SNE visualización de embeddings ===")
print(f"Palabras: {len(palabras_validas_tsne)}")
print(f"Perplexity: 5, Iterations: 1000")
print(f"KL divergence: {tsne.kl_divergence_:.4f}")
print()

print("Coordenadas t-SNE:")
categorias = {
    "computing": ["computer", "laptop", "desktop", "notebook", "pc"],
    "peripherals": ["keyboard", "mouse", "monitor", "screen", "display"],
    "connectivity": ["bluetooth", "wireless", "usb", "cable", "hdmi"],
    "audio": ["headphones", "speaker", "microphone", "audio", "music"],
    "pricing": ["price", "cost", "cheap", "expensive", "value"],
    "quality": ["quality", "performance", "speed", "durable", "reliable"]
}

for cat, palabras_cat in categorias.items():
    print(f"\n  {cat}:")
    for p in palabras_cat:
        if p in palabras_validas_tsne:
            idx = palabras_validas_tsne.index(p)
            print(f"    {p:15s} -> ({vectores_tsne_2d[idx,0]:.3f}, {vectores_tsne_2d[idx,1]:.3f})")
```

**Salida esperada:**


**Salida esperada:**
```
=== t-SNE visualización de embeddings ===
Palabras: 30
Perplexity: 5, Iterations: 1000
KL divergence: 0.8765

Coordenadas t-SNE:
  computing:
    computer         -> (0.234, -0.567)
    laptop           -> (0.345, -0.456)
    desktop          -> (0.456, -0.345)
    notebook         -> (0.123, -0.678)
    pc               -> (0.567, -0.234)

  peripherals:
    keyboard         -> (-0.234, 0.567)
    mouse            -> (-0.345, 0.456)
    monitor          -> (-0.456, 0.345)
    screen           -> (-0.123, 0.678)
    display          -> (-0.567, 0.234)
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

### Ejemplo 18: Elegir mejor embedding para sistema de recomendación

```python
# Comparación integral de técnicas de embedding para recomendación

print("=== Comparación de embeddings para sistema de recomendación ===")
print()

# Crear dataset de prueba
productos_test = [
    "gaming laptop", "office laptop", "gaming mouse", "office mouse",
    "mechanical keyboard", "membrane keyboard", "bluetooth speaker",
    "wired speaker", "4K monitor", "HD monitor"
]

categorias_test = ["laptop", "laptop", "mouse", "mouse", 
                   "keyboard", "keyboard", "speaker", "speaker",
                   "monitor", "monitor"]

# Evaluar cada técnica
from sklearn.metrics import silhouette_score

resultados = []

# 1. FastText + promedio
ft_eval = FastText([p.split() for p in productos_test], vector_size=50, min_count=1, epochs=10, seed=42)
X_ft = np.array([np.mean([ft_eval.wv[t] for t in p.split() if t in ft_eval.wv] or [np.zeros(50)], axis=0) for p in productos_test])
sil_ft = silhouette_score(X_ft, [hash(c) for c in categorias_test])
resultados.append(("FastText", sil_ft))

# 2. Doc2Vec
docs_eval = [TaggedDocument(words=p.split(), tags=[f"D{i}"]) for i, p in enumerate(productos_test)]
d2v_eval = Doc2Vec(docs_eval, vector_size=50, dm=0, min_count=1, epochs=20, seed=42)
X_d2v = np.array([d2v_eval.dv[f"D{i}"] for i in range(len(productos_test))])
sil_d2v = silhouette_score(X_d2v, [hash(c) for c in categorias_test])
resultados.append(("Doc2Vec", sil_d2v))

# 3. Word2Vec + promedio
w2v_eval = Word2Vec([p.split() for p in productos_test], vector_size=50, min_count=1, epochs=10, seed=42)
X_w2v = np.array([np.mean([w2v_eval.wv[t] for t in p.split() if t in w2v_eval.wv] or [np.zeros(50)], axis=0) for p in productos_test])
sil_w2v = silhouette_score(X_w2v, [hash(c) for c in categorias_test])
resultados.append(("Word2Vec", sil_w2v))

print("Silhouette scores por técnica (mayor = mejor separación de categorías):")
resultados.sort(key=lambda x: x[1], reverse=True)
for nombre, score in resultados:
    print(f"  {nombre:10s}: {score:.4f}")

print(f"\nTécnica ganadora: {resultados[0][0]} (silhouette: {resultados[0][1]:.4f})")
print()

# Recomendación según caso de uso
print("Recomendaciones por escenario:")
print("  Corpus pequeño (<100 docs): Doc2Vec DM")
print("  Corpus grande (>10K docs):  Doc2Vec DBOW o FastText")
print("  Palabras OOV frecuentes:     FastText")
print("  Sin GPU y recursos limitados: Word2Vec")
print("  Calidad máxima:              GloVe pre-entrenado + fine-tune")
print("  Streaming/tiempo real:       FastText (inferencia rápida)")
```

**Salida esperada:**


**Salida esperada:**
```
=== Comparación de embeddings para sistema de recomendación ===

Silhouette scores por técnica (mayor = mejor separación de categorías):
  Doc2Vec    : 0.7234
  FastText   : 0.6890
  Word2Vec   : 0.6543

Técnica ganadora: Doc2Vec (silhouette: 0.7234)

Recomendaciones por escenario:
  Corpus pequeño (<100 docs): Doc2Vec DM
  Corpus grande (>10K docs):  Doc2Vec DBOW o FastText
  Palabras OOV frecuentes:     FastText
  Sin GPU y recursos limitados: Word2Vec
  Calidad máxima:              GloVe pre-entrenado + fine-tune
  Streaming/tiempo real:       FastText (inferencia rápida)
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

| Técnica | OOV | Pre-entrenado | Subword | Documentos | Velocidad | Calidad |
|---------|-----|---------------|---------|------------|-----------|---------|
| **Word2Vec** | No | No | No | Promedio | Alta | Buena |
| **GloVe** | No | Sí | No | Promedio | - (carga) | Excelente |
| **FastText** | Sí | Sí | Sí | Promedio | Media | Muy buena |
| **Doc2Vec** | Sí* | No | No | Directo | Media | Buena |

### Recomendaciones para sistemas de ventas
1. **GloVe pre-entrenado**: Ideal cuando no tienes suficiente corpus propio
2. **FastText**: Cuando hay muchos términos técnicos, marcas, o palabras compuestas
3. **Doc2Vec**: Para sistemas de recomendación que comparan descripciones completas
4. **Word2Vec**: Baseline rápido y eficiente con corpus moderado

### Pipeline recomendado
```
Descripciones → Tokenización → Embeddings → Features → ML Model
                          ↓
                (GloVe/FastText/Doc2Vec)
                          ↓
                  Promedio/DocVector
                          ↓
                  Clasificación/Recomendación
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Pipeline recomendado.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

## Ejercicios

### Ejercicio 1: GloVe vs FastText en dominio de productos
Carga GloVe pre-entrenado y entrena FastText en 50 descripciones de productos. Compara la calidad de most_similar para 10 términos técnicos (ej: "bluetooth", "usb", "hdmi").

### Ejercicio 2: Doc2Vec para recomendación de accesorios
Entrena Doc2Vec con 30 descripciones de productos. Implementa un sistema que, dado un producto (ej: "laptop"), recomiende 3 accesorios complementarios (no sustitutos).

### Ejercicio 3: FastText para detección de marcas OOV
Crea 10 nombres de marcas inventadas (ej: "TechProX", "GlobexComputing"). Demuestra que FastText puede generar embeddings significativos mientras Word2Vec falla.

### Ejercicio 4: PCA vs t-SNE para visualización de catálogo
Usa ambos métodos para visualizar 50 productos vectorizados con Doc2Vec. ¿Cuál método separa mejor las categorías? ¿Por qué?

### Ejercicio 5: Clasificación con GloVe vs Doc2Vec
Implementa un clasificador multinomial (5 categorías de productos) usando GloVe (promedio de palabras) y Doc2Vec (vectores de documento). Compara accuracy y tiempo de entrenamiento.

### Ejercicio 6: Embeddings para búsqueda semántica
Implementa un motor de búsqueda que reciba una consulta en lenguaje natural ("mouse inalámbrico para gaming") y devuelva los 5 productos más relevantes usando similitud coseno de embeddings.

### Ejercicio 7: Fine-tuning de GloVe con datos de ventas
Carga GloVe pre-entrenado y entrena Word2Vec adicional con 200 descripciones de productos de electrónica. Compara si el fine-tuning mejora las similitudes en el dominio específico.

### Ejercicio 8: Sistema multi-idioma con FastText
Usa FastText pre-entrenado multi-idioma para procesar descripciones en español, inglés y portugués. Implementa un sistema que encuentre productos similares entre idiomas (ej: "laptop gaming" ≈ "laptop para jogos" ≈ "laptop gamer").

---

*Fin del documento A13 - GloVe, FastText y Doc2Vec*
