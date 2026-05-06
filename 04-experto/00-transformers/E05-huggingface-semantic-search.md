# E05: Hugging Face — Búsqueda Semántica con Sentence-Transformers y FAISS

## Objetivo
Construir sistemas de búsqueda semántica para catálogos de productos usando embeddings (sentence-transformers) e indexación eficiente (FAISS). Aplicado a búsqueda en lenguaje natural sobre descripciones de productos, reseñas y catálogos B2B.

---

## 1. Fundamentos Teóricos

### 1.1 Embeddings de Texto (Sentence-Transformers)

Los modelos de sentence-transformers convierten texto en vectores densos de ~384-768 dimensiones que capturan el significado semántico.

**Modelos populares**:
- `paraphrase-multilingual-MiniLM-L12-v2`: Multilingüe, 384d, rápido.
- `distiluse-base-multilingual-cased-v2`: Multilingüe, 512d, balance calidad/velocidad.
- `all-MiniLM-L6-v2`: Inglés, 384d, muy rápido.

### 1.2 Búsqueda Semántica vs Búsqueda por Palabras Clave

| Característica | Palabras Clave (BM25) | Búsqueda Semántica |
|---|---|---|
| Matching | Token exacto | Significado |
| Sinónimos | No captura | Sí captura |
| "laptop barata" | Busca "laptop", "barata" | Encuentra "notebook económica" |
| Multilingüe | Requiere traducción | Embeddings alineados |
| Contexto | No | Sí |

### 1.3 FAISS (Facebook AI Similarity Search)

FAISS es la biblioteca estándar para búsqueda de similitud en espacios vectoriales.

**Índices principales**:

| Índice | Tipo | Velocidad | Precisión | Uso |
|---|---|---|---|---|
| `IndexFlatIP` | Fuerza bruta | Lenta (O(n)) | Exacta | < 10k vectores |
| `IndexFlatL2` | Fuerza bruta L2 | Lenta | Exacta | Distancia euclídea |
| `IndexIVFFlat` | Cuantización IVF | Rápida | Alta | 10k-1M vectores |
| `IndexHNSWFlat` | Grafos jerárquicos | Muy rápida | Alta | > 100k vectores |

### 1.4 Normalización y Similitud

- **Producto punto (coseno)**: Normalizar embeddings a norma L2 = 1, luego producto punto = coseno.
- **FAISS normalización**: `faiss.normalize_L2(x)` o `normalize_embeddings=True` en `model.encode()`.
- **util.cos_sim**: Calcula similitud coseno entre dos conjuntos de embeddings.
- **util.semantic_search**: Wrapper de búsqueda semántica con sentence-transformers.

---

## 2. Ejemplos Prácticos

### Ejemplo 1: Cargar Modelo Sentence-Transformers Multilingüe

```python
from sentence_transformers import SentenceTransformer

# Modelo multilingüe: funciona con español, inglés, etc.
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

print(f"Modelo: {model}")
print(f"Dimensión de embeddings: {model.get_sentence_embedding_dimension()}")
print(f"Max sequence length: {model.max_seq_length}")

# Probar codificación
texto = "Laptop gaming con RTX 3060"
embedding = model.encode(texto)
print(f"\nTexto: '{texto}'")
print(f"Embedding shape: {embedding.shape}")
print(f"Embedding (primeros 5 valores): {embedding[:5]}")
print(f"Norma L2: {sum(embedding**2)**0.5:.4f}")

print("\n✅ SentenceTransformer multilingüe cargado correctamente")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: Cargar Modelo Sentence-Transformers Multilingüe.*

1. Modelo multilingüe: funciona con español, inglés, etc.
2. Probar codificación

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 2: Codificar Catálogo de Productos en Embeddings

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Catálogo de productos
catalogo = [
    {"id": 1, "nombre": "Laptop Gaming Pro", "descripcion": "Laptop gaming con RTX 3060, i7, 16GB RAM"},
    {"id": 2, "nombre": "Mouse Ergonómico", "descripcion": "Mouse inalámbrico ergonómico con 6 botones"},
    {"id": 3, "nombre": "Teclado Mecánico RGB", "descripcion": "Teclado mecánico Cherry MX Brown retroiluminado"},
    {"id": 4, "nombre": "Monitor 4K 27\"", "descripcion": "Monitor IPS 4K 27 pulgadas 144Hz"},
    {"id": 5, "nombre": "Silla Ergonómica", "descripcion": "Silla de oficina ergonómica ajustable"},
    {"id": 6, "nombre": "Auriculares Bluetooth", "descripcion": "Auriculares inalámbricos cancelación ruido"},
    {"id": 7, "nombre": "Hub USB-C", "descripcion": "Hub USB C 7 en 1 con HDMI y SD"},
    {"id": 8, "nombre": "SSD 1TB NVMe", "descripcion": "Disco SSD NVMe 1TB lectura 3500MB/s"},
]

# Extraer textos
textos = [p["nombre"] + " " + p["descripcion"] for p in catalogo]
nombres = [p["nombre"] for p in catalogo]

# Codificar
embeddings = model.encode(textos, convert_to_tensor=True)
print(f"Embeddings shape: {embeddings.shape}")
print(f"{len(catalogo)} productos codificados como vectores de {embeddings.shape[1]} dimensiones")

for i, (nombre, emb) in enumerate(zip(nombres, embeddings)):
    print(f"  {nombre}: {emb.shape}")

print("\n✅ Catálogo completo codificado como embeddings")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: Codificar Catálogo de Productos en Embeddings.*

1. Catálogo de productos
2. Extraer textos
3. Codificar

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 3: Búsqueda Semántica — Encontrar Productos por Consulta

```python
from sentence_transformers import SentenceTransformer, util
import numpy as np

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Catálogo
catalogo = [
    "Laptop gaming con RTX 3060, i7, 16GB RAM, SSD 512GB",
    "Mouse inalámbrico ergonómico con 6 botones programables",
    "Teclado mecánico Cherry MX Brown retroiluminado RGB",
    "Monitor IPS 4K 27 pulgadas 144Hz 1ms",
    "Silla de oficina ergonómica ajustable soporte lumbar",
    "Auriculares inalámbricos bluetooth cancelación ruido activa",
    "Hub USB C 7 en 1 con HDMI 4K, SD, USB 3.0",
    "Disco SSD NVMe 1TB lectura 3500MB/s escritura 3000MB/s",
    "Impresora láser wifi dúplex automático a color",
    "Router wifi 6 doble banda gigabit 3000Mbps",
]
corpus_embeddings = model.encode(catalogo, convert_to_tensor=True)

# Consulta en lenguaje natural
consulta = "quiero un teclado para jugar con luces"
query_embedding = model.encode(consulta, convert_to_tensor=True)

# Calcular similitud coseno
scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
top_k = 5
top_results = scores.topk(top_k)

print(f"Consulta: '{consulta}'\n")
print(f"Top {top_k} productos más similares:")
for score, idx in zip(top_results[0], top_results[1]):
    print(f"  [{score:.3f}] {catalogo[idx]}")

# Consulta en inglés (modelo multilingüe)
consulta_en = "cheap wireless mouse for office"
query_emb_en = model.encode(consulta_en, convert_to_tensor=True)
scores_en = util.cos_sim(query_emb_en, corpus_embeddings)[0]
top_en = scores_en.topk(3)
print(f"\nConsulta (EN): '{consulta_en}'")
for score, idx in zip(top_en[0], top_en[1]):
    print(f"  [{score:.3f}] {catalogo[idx]}")

print("\n✅ Búsqueda semántica: encuentra productos relevantes aunque no haya matching exacto")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: Búsqueda Semántica — Encontrar Productos por Consulta.*

1. Catálogo
2. Consulta en lenguaje natural
3. Calcular similitud coseno
4. Consulta en inglés (modelo multilingüe)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 4: util.semantic_search — API de Alto Nivel

```python
from sentence_transformers import SentenceTransformer, util
import torch

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

catalogo = [
    "Laptop gaming con RTX 3060, i7, 16GB RAM",
    "Mouse inalámbrico ergonómico 6 botones",
    "Teclado mecánico Cherry MX Brown RGB",
    "Monitor IPS 4K 27 pulgadas 144Hz",
    "Silla ergonómica oficina ajustable lumbar",
    "Auriculares bluetooth cancelación ruido",
    "Hub USB C 7 puertos HDMI SD",
    "SSD NVMe 1TB 3500MB/s",
]
corpus_emb = model.encode(catalogo, convert_to_tensor=True)

consulta = "necesito un monitor para diseño gráfico"
query_emb = model.encode(consulta, convert_to_tensor=True)

# util.semantic_search hace todo: cos_sim + top_k sorting
resultados = util.semantic_search(query_emb, corpus_emb, top_k=3)

print(f"Consulta: '{consulta}'")
for i, r in enumerate(resultados[0]):
    idx = r['corpus_id']
    score = r['score']
    print(f"  #{i+1} (score={score:.3f}): {catalogo[idx]}")

print("\n✅ util.semantic_search simplifica la búsqueda semántica")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: util.semantic_search — API de Alto Nivel.*

1. util.semantic_search hace todo: cos_sim + top_k sorting

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 5: FAISS IndexFlatIP — Índice de Producto Punto

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
dim = model.get_sentence_embedding_dimension()

# Crear índice de producto punto (similitud coseno si los vectores están normalizados)
index = faiss.IndexFlatIP(dim)
print(f"Índice FAISS FlatIP creado: dimensión={dim}, tipo={type(index).__name__}")

# Indexar productos
catalogo = [
    "Laptop gaming con RTX 3060",
    "Mouse inalámbrico ergonómico",
    "Teclado mecánico RGB",
    "Monitor 4K 27 pulgadas",
    "Silla ergonómica oficina",
]
embeddings = np.array([model.encode(p) for p in catalogo]).astype('float32')
faiss.normalize_L2(embeddings)  # Normalizar para coseno

index.add(embeddings)
print(f"Vectores indexados: {index.ntotal}")

# Buscar
consulta = "monitor para gaming"
query_emb = np.array([model.encode(consulta)]).astype('float32')
faiss.normalize_L2(query_emb)

D, I = index.search(query_emb, k=3)
print(f"\nConsulta: '{consulta}'")
for i, (dist, idx) in enumerate(zip(D[0], I[0])):
    print(f"  #{i+1} (score={dist:.4f}): {catalogo[idx]}")

print("\n✅ FAISS IndexFlatIP: búsqueda exacta por similitud coseno")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: FAISS IndexFlatIP — Índice de Producto Punto.*

1. Crear índice de producto punto (similitud coseno si los vectores están normalizados)
2. Indexar productos
3. Buscar

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 6: FAISS add/search — Añadir y Buscar Vectores

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
dim = model.get_sentence_embedding_dimension()

index = faiss.IndexFlatIP(dim)

# Catálogo grande (simulado)
np.random.seed(42)
productos = [f"Producto {i}: descripción detallada del producto número {i}" for i in range(1000)]
embeddings = np.array([model.encode(p) for p in productos]).astype('float32')
faiss.normalize_L2(embeddings)

# Añadir vectores al índice
index.add(embeddings)
print(f"Indexados {index.ntotal} vectores en FAISS")

# Buscar
consultas = ["laptop gaming barata", "mouse inalámbrico", "monitor 4K"]
for consulta in consultas:
    query_emb = np.array([model.encode(consulta)]).astype('float32')
    faiss.normalize_L2(query_emb)
    D, I = index.search(query_emb, k=3)
    print(f"\nConsulta: '{consulta}'")
    for i, (dist, idx) in enumerate(zip(D[0], I[0])):
        print(f"  [{dist:.4f}] {productos[idx]}")

print("\n✅ FAISS add/search: API básica para indexar y buscar")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: FAISS add/search — Añadir y Buscar Vectores.*

1. Catálogo grande (simulado)
2. Añadir vectores al índice
3. Buscar

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 7: FAISS IndexIVFFlat — Índice Cuantizado (Rápido para >100k)

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
dim = model.get_sentence_embedding_dimension()

# Catálogo grande (100k productos sintéticos)
np.random.seed(42)
n_total = 100000
descripciones = [f"Producto {i}: categoría {i % 10}, precio ${i % 500 + 1}" for i in range(n_total)]
embeddings = np.array([model.encode(d) for d in descripciones[:1000]]).astype('float32')
faiss.normalize_L2(embeddings)

# El dataset completo simulado
embeddings_full = np.vstack([
    embeddings,
    np.random.randn(n_total - 1000, dim).astype('float32')
])
faiss.normalize_L2(embeddings_full)

# IndexIVFFlat: más rápido para grandes volúmenes
nlist = 100  # número de centroides (clusters)
quantizer = faiss.IndexFlatIP(dim)
index_ivf = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
index_ivf.train(embeddings_full)
index_ivf.add(embeddings_full)
index_ivf.nprobe = 10  # número de clusters a explorar

print(f"IndexIVFFlat: {index_ivf.ntotal} vectores, {nlist} centroides, nprobe={index_ivf.nprobe}")

# Búsqueda
consulta = "laptop gaming barata"
query_emb = np.array([model.encode(consulta)]).astype('float32')
faiss.normalize_L2(query_emb)

D, I = index_ivf.search(query_emb, k=5)
print(f"\nConsulta: '{consulta}'")
for i, (dist, idx) in enumerate(zip(D[0], I[0])):
    print(f"  [{dist:.4f}] Producto {idx}")

print("\n✅ IndexIVFFlat: búsqueda aproximada rápida para grandes volúmenes")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: FAISS IndexIVFFlat — Índice Cuantizado (Rápido para >100k).*

1. Catálogo grande (100k productos sintéticos)
2. El dataset completo simulado
3. IndexIVFFlat: más rápido para grandes volúmenes
4. Búsqueda

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 8: FAISS nprobe — Precisión vs Velocidad

```python
import faiss
import numpy as np
import time
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
dim = model.get_sentence_embedding_dimension()

# 50k vectores sintéticos
np.random.seed(42)
n = 50000
embeddings = np.random.randn(n, dim).astype('float32')
faiss.normalize_L2(embeddings)

# Construir índice IVFFlat
nlist = 50
quantizer = faiss.IndexFlatIP(dim)
index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
index.train(embeddings)
index.add(embeddings)

# Probar diferentes nprobe
query = np.random.randn(1, dim).astype('float32')
faiss.normalize_L2(query)

for nprobe in [1, 5, 10, 20, 50]:
    index.nprobe = nprobe
    start = time.time()
    D, I = index.search(query, k=10)
    t = time.time() - start
    print(f"nprobe={nprobe:3d}: {t*1000:.2f}ms | scores={D[0][0]:.3f}")

print("\n✅ nprobe controla el balance velocidad vs precisión")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: FAISS nprobe — Precisión vs Velocidad.*

1. 50k vectores sintéticos
2. Construir índice IVFFlat
3. Probar diferentes nprobe

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 9: FAISS IndexHNSWFlat — Índice de Grafos Jerárquicos

```python
import faiss
import numpy as np
import time
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
dim = model.get_sentence_embedding_dimension()

# 10k vectores
np.random.seed(42)
n = 10000
embeddings = np.random.randn(n, dim).astype('float32')
faiss.normalize_L2(embeddings)

# HNSW: Hierarchical Navigable Small World
index_hnsw = faiss.IndexHNSWFlat(dim, 32)  # 32 vecinos en construcción
index_hnsw.hnsw.efConstruction = 40
index_hnsw.add(embeddings)

print(f"IndexHNSWFlat: {index_hnsw.ntotal} vectores")
print(f"  efConstruction: {index_hnsw.hnsw.efConstruction}")

# Búsqueda con diferentes efSearch
query = np.random.randn(1, dim).astype('float32')
faiss.normalize_L2(query)

for ef_search in [16, 32, 64, 128]:
    index_hnsw.hnsw.efSearch = ef_search
    start = time.time()
    D, I = index_hnsw.search(query, k=10)
    t = time.time() - start
    print(f"efSearch={ef_search:3d}: {t*1000:.2f}ms | score={D[0][0]:.3f}")

print("\n✅ HNSW: índices de última generación para búsqueda en alta dimensión")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: FAISS IndexHNSWFlat — Índice de Grafos Jerárquicos.*

1. 10k vectores
2. HNSW: Hierarchical Navigable Small World
3. Búsqueda con diferentes efSearch

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 10: faiss.write_index/read_index — Guardar y Cargar

```python
import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
dim = model.get_sentence_embedding_dimension()

# Crear e indexar
productos = [
    "Laptop gaming con RTX 3060",
    "Mouse inalámbrico ergonómico",
    "Teclado mecánico RGB",
]
embeddings = np.array([model.encode(p) for p in productos]).astype('float32')
faiss.normalize_L2(embeddings)

index = faiss.IndexFlatIP(dim)
index.add(embeddings)

# Guardar a disco
faiss.write_index(index, "/tmp/indice_productos.faiss")
print(f"Índice guardado: {os.path.getsize('/tmp/indice_productos.faiss') / 1024:.1f} KB")

# Cargar desde disco
index_cargado = faiss.read_index("/tmp/indice_productos.faiss")
print(f"Índice cargado: {index_cargado.ntotal} vectores, dim={index_cargado.d}")

# Verificar búsqueda
query_emb = np.array([model.encode("laptop para gaming")]).astype('float32')
faiss.normalize_L2(query_emb)
D, I = index_cargado.search(query_emb, k=2)
for dist, idx in zip(D[0], I[0]):
    print(f"  [{dist:.4f}] {productos[idx]}")

print("\n✅ FAISS índices persistentes: guardar y cargar")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: faiss.write_index/read_index — Guardar y Cargar.*

1. Crear e indexar
2. Guardar a disco
3. Cargar desde disco
4. Verificar búsqueda

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 11: faiss.normalize_L2 — Normalizar Embeddings

```python
import faiss
import numpy as np

# Embeddings sin normalizar
embeddings = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
], dtype='float32')

print("Embeddings originales (normas L2):")
for i, emb in enumerate(embeddings):
    norm = np.linalg.norm(emb)
    print(f"  Vector {i}: norma={norm:.4f}")

# Normalizar
faiss.normalize_L2(embeddings)
print("\nEmbeddings normalizados (normas L2 = 1):")
for i, emb in enumerate(embeddings):
    norm = np.linalg.norm(emb)
    print(f"  Vector {i}: norma={norm:.4f}, valores={emb}")

# Verificar: producto punto = coseno
cos_01 = np.dot(embeddings[0], embeddings[1])
print(f"\nProducto punto (coseno) entre vectores 0 y 1: {cos_04f}")
print("✅ faiss.normalize_L2 asegura que producto punto = similitud coseno")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: faiss.normalize_L2 — Normalizar Embeddings.*

1. Embeddings sin normalizar
2. Normalizar
3. Verificar: producto punto = coseno

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 12: normalize_embeddings=True en encode

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

textos = [
    "Laptop gaming con RTX 3060",
    "Mouse inalámbrico ergonómico",
    "Teclado mecánico RGB"
]

# Sin normalizar
emb_sin_norm = model.encode(textos, normalize_embeddings=False)
print("Sin normalizar:")
for i, emb in enumerate(emb_sin_norm):
    print(f"  '{textos[i][:20]}': norma={np.linalg.norm(emb):.4f}")

# Con normalizar
emb_con_norm = model.encode(textos, normalize_embeddings=True)
print("\nCon normalize_embeddings=True:")
for i, emb in enumerate(emb_con_norm):
    print(f"  '{textos[i][:20]}': norma={np.linalg.norm(emb):.4f}")

# Verificar similitud
cos_sim = np.dot(emb_con_norm[0], emb_con_norm[1])
print(f"\nSimilitud coseno 'laptop' vs 'mouse': {cos_sim:.4f}")
print("✅ normalize_embeddings=True produce vectores normalizados listos para FAISS")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: normalize_embeddings=True en encode.*

1. Sin normalizar
2. Con normalizar
3. Verificar similitud

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 13: Búsqueda Asimétrica — Query Corta vs Descripción Larga

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Descripciones largas de productos (corpus)
descripciones = [
    "Laptop gaming de alto rendimiento con procesador Intel Core i7-12700H, "
    "16GB de RAM DDR5, tarjeta gráfica NVIDIA RTX 3060 de 6GB, SSD NVMe de 512GB, "
    "pantalla de 15.6 pulgadas Full HD 144Hz, teclado retroiluminado RGB.",
    "Ratón inalámbrico ergonómico con 6 botones programables, sensor óptico de 16000 DPI, "
    "batería recargable de 70 horas de duración, compatible con Windows y Mac.",
    "Teclado mecánico gaming con switches Cherry MX Brown, retroiluminación RGB personalizable, "
    "construcción en aluminio, reposamuñecas magnético, cable USB-C desmontable.",
]

# Consultas cortas (lenguaje natural)
consultas = [
    "laptop para jugar",
    "mouse cómodo para oficina",
    "teclado con luces RGB",
]

corpus_emb = model.encode(descripciones, convert_to_tensor=True)

for consulta in consultas:
    query_emb = model.encode(consulta, convert_to_tensor=True)
    scores = util.cos_sim(query_emb, corpus_emb)[0]
    best_idx = scores.argmax().item()
    print(f"\nConsulta: '{consulta}'")
    print(f"  Mejor match: {descripciones[best_idx][:50]}...")
    print(f"  Score: {scores[best_idx]:.3f}")

print("\n✅ Búsqueda asimétrica: consultas cortas encuentran descripciones largas")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: Búsqueda Asimétrica — Query Corta vs Descripción Larga.*

1. Descripciones largas de productos (corpus)
2. Consultas cortas (lenguaje natural)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 14: Búsqueda Multilingüe — Query en Inglés, Productos en Español

```python
from sentence_transformers import SentenceTransformer, util

# Modelo multilingüe
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Catálogo en español
catalogo_es = [
    "Laptop gaming con RTX 3060 y procesador i7",
    "Ratón inalámbrico ergonómico con 6 botones",
    "Teclado mecánico retroiluminado RGB",
    "Monitor 4K de 27 pulgadas para diseño",
    "Silla de oficina ergonómica ajustable",
    "Auriculares con cancelación de ruido activa",
]

corpus_emb = model.encode(catalogo_es, convert_to_tensor=True)

# Consultas en diferentes idiomas
consultas = {
    "EN": "gaming laptop with good graphics",
    "ES": "computadora portátil para juegos",
    "FR": "ordinateur portable de jeu",
    "DE": "Gaming-Laptop mit guter Grafik",
}

for lang, consulta in consultas.items():
    query_emb = model.encode(consulta, convert_to_tensor=True)
    scores = util.cos_sim(query_emb, corpus_emb)[0]
    best_idx = scores.argmax().item()
    print(f"[{lang}] '{consulta}'")
    print(f"  -> {catalogo_es[best_idx]} (score: {scores[best_idx]:.3f})")

print("\n✅ Búsqueda multilingüe: embeddings alineados entre idiomas")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: Búsqueda Multilingüe — Query en Inglés, Productos en Español.*

1. Modelo multilingüe
2. Catálogo en español
3. Consultas en diferentes idiomas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 15: Híbrido — Búsqueda Semántica + Filtro por Precio

```python
from sentence_transformers import SentenceTransformer, util
import numpy as np

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Catálogo con precios
catalogo = [
    {"nombre": "Laptop Gaming Pro", "descripcion": "Laptop gaming RTX 3060 i7 16GB", "precio": 1200},
    {"nombre": "Mouse Ergonómico", "descripcion": "Mouse inalámbrico ergonómico 6 botones", "precio": 45},
    {"nombre": "Teclado Mecánico RGB", "descripcion": "Teclado mecánico Cherry MX Brown RGB", "precio": 120},
    {"nombre": "Monitor 4K 27\"", "descripcion": "Monitor 4K IPS 27 pulgadas 144Hz", "precio": 450},
    {"nombre": "Silla Ergonómica", "descripcion": "Silla oficina ergonómica ajustable lumbar", "precio": 350},
    {"nombre": "Auriculares BT", "descripcion": "Auriculares bluetooth cancelación ruido", "precio": 80},
    {"nombre": "Hub USB-C", "descripcion": "Hub USB C 7 puertos HDMI SD lector", "precio": 35},
    {"nombre": "SSD 1TB", "descripcion": "Disco SSD NVMe 1TB 3500MB/s", "precio": 150},
]

def buscar_con_filtro(consulta, precio_max=None, precio_min=None, top_k=3):
    textos = [p["descripcion"] for p in catalogo]
    corpus_emb = model.encode(textos, convert_to_tensor=True)
    query_emb = model.encode(consulta, convert_to_tensor=True)

    scores = util.cos_sim(query_emb, corpus_emb)[0]

    # Aplicar filtros
    indices_filtrados = []
    for i, p in enumerate(catalogo):
        if precio_max is not None and p["precio"] > precio_max:
            continue
        if precio_min is not None and p["precio"] < precio_min:
            continue
        indices_filtrados.append(i)

    # Ordenar por score solo los que pasan filtro
    resultados = [(scores[i].item(), i) for i in indices_filtrados]
    resultados.sort(reverse=True)
    return resultados[:top_k]

# Búsqueda sin filtro
print("Búsqueda: 'teclado para gaming'")
resultados = buscar_con_filtro("teclado para gaming")
for score, idx in resultados:
    p = catalogo[idx]
    print(f"  [{score:.3f}] {p['nombre']} - ${p['precio']}")

# Búsqueda con filtro de precio < $100
print("\nBúsqueda: 'teclado para gaming' con precio < $100")
resultados = buscar_con_filtro("teclado para gaming", precio_max=100)
for score, idx in resultados:
    p = catalogo[idx]
    print(f"  [{score:.3f}] {p['nombre']} - ${p['precio']}")

print("\n✅ Búsqueda híbrida: semántica + filtros estructurados")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: Híbrido — Búsqueda Semántica + Filtro por Precio.*

1. Catálogo con precios
2. Aplicar filtros
3. Ordenar por score solo los que pasan filtro
4. Búsqueda sin filtro
5. Búsqueda con filtro de precio < $100

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 16: Evaluar Búsqueda — Recall@k, Precision@k

```python
from sentence_transformers import SentenceTransformer, util
import numpy as np

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Dataset de evaluación: pares (consulta, producto_relevante)
pares_eval = [
    ("laptop gaming", "Laptop gaming con RTX 3060"),
    ("mouse oficina", "Mouse inalámbrico ergonómico"),
    ("teclado mecánico", "Teclado mecánico Cherry MX Brown"),
    ("monitor 4K", "Monitor IPS 4K 27 pulgadas 144Hz"),
    ("silla cómoda", "Silla de oficina ergonómica ajustable"),
]
catalogo = [par[1] for par in pares_eval] + [
    "Auriculares bluetooth cancelación ruido",
    "Hub USB C 7 puertos HDMI SD",
    "SSD NVMe 1TB 3500MB/s",
    "Impresora láser wifi dúplex",
    "Router wifi 6 gigabit",
]

corpus_emb = model.encode(catalogo, convert_to_tensor=True)

def evaluar_busqueda(consultas_relevantes, k=3):
    recalls = []
    precisions = []
    for consulta, producto_relevante in consultas_relevantes:
        query_emb = model.encode(consulta, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, corpus_emb)[0]
        top_k_idx = scores.topk(k).indices.tolist()

        # Recall: ¿está el relevante en top-k?
        idx_relevante = catalogo.index(producto_relevante)
        relevante_encontrado = idx_relevante in top_k_idx
        recalls.append(1.0 if relevante_encontrado else 0.0)

        # Precision: ¿cuántos de top-k son realmente relevantes?
        # (asumimos solo 1 relevante por consulta)
        precisión = 1.0 / k if relevante_encontrado else 0.0
        precisions.append(precisión)

    return {
        "recall@k": np.mean(recalls),
        "precision@k": np.mean(precisions),
    }

resultados = evaluar_busqueda(pares_eval, k=3)
print("Evaluación de búsqueda semántica:")
for metrica, valor in resultados.items():
    print(f"  {metrica}: {valor:.2%}")

print("\n✅ Evaluación cuantitativa de la búsqueda")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: Evaluar Búsqueda — Recall@k, Precision@k.*

1. Dataset de evaluación: pares (consulta, producto_relevante)
2. Recall: ¿está el relevante en top-k?
3. Precision: ¿cuántos de top-k son realmente relevantes?
4. (asumimos solo 1 relevante por consulta)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 17: Comparar FAISS — Brute Force vs IVFFlat vs HNSW

```python
import faiss
import numpy as np
import time

dim = 384
n = 100000
k = 10

# Datos sintéticos
np.random.seed(42)
xb = np.random.randn(n, dim).astype('float32')
faiss.normalize_L2(xb)
xq = np.random.randn(1, dim).astype('float32')
faiss.normalize_L2(xq)

# 1. Brute Force (IndexFlatIP)
index_flat = faiss.IndexFlatIP(dim)
index_flat.add(xb)
start = time.time()
D_flat, I_flat = index_flat.search(xq, k)
t_flat = time.time() - start

# 2. IVFFlat con nprobe=10
nlist = 100
quantizer = faiss.IndexFlatIP(dim)
index_ivf = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
index_ivf.train(xb)
index_ivf.add(xb)
index_ivf.nprobe = 10
start = time.time()
D_ivf, I_ivf = index_ivf.search(xq, k)
t_ivf = time.time() - start

# 3. HNSWFlat
index_hnsw = faiss.IndexHNSWFlat(dim, 32)
index_hnsw.add(xb)
index_hnsw.hnsw.efSearch = 64
start = time.time()
D_hnsw, I_hnsw = index_hnsw.search(xq, k)
t_hnsw = time.time() - start

print("Comparación de índices FAISS (100k vectores, dim=384):")
print(f"{'Índice':<15} {'Tiempo':<12} {'Score':<12} {'Recíproco exacto':<15}")
print(f"{'FlatIP':<15} {t_flat*1000:<9.2f}ms {D_flat[0][0]:<12.4f} {'1.0000':<15}")
print(f"{'IVFFlat':<15} {t_ivf*1000:<9.2f}ms {D_ivf[0][0]:<12.4f} {float(len(set(I_flat[0]) & set(I_ivf[0])))/k:<15.2%}")
print(f"{'HNSWFlat':<15} {t_hnsw*1000:<9.2f}ms {D_hnsw[0][0]:<12.4f} {float(len(set(I_flat[0]) & set(I_hnsw[0])))/k:<15.2%}")

print("\n✅ Comparación: FlatIP es exacto, IVFFlat y HNSW son aproximados pero más rápidos")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: Comparar FAISS — Brute Force vs IVFFlat vs HNSW.*

1. Datos sintéticos
2. 1. Brute Force (IndexFlatIP)
3. 2. IVFFlat con nprobe=10
4. 3. HNSWFlat

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 18: Integrador — Sistema de Búsqueda Semántica para Catálogo

```python
from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np
import pandas as pd
import time
import os

# ===== 1. DATOS =====
catalogo = pd.DataFrame({
    "id": range(1, 13),
    "nombre": [
        "Laptop Gaming Pro", "Mouse Ergonómico", "Teclado Mecánico RGB",
        "Monitor 4K 27\"", "Silla Ergonómica", "Auriculares Bluetooth",
        "Hub USB-C", "SSD 1TB NVMe", "Impresora Láser",
        "Router WiFi 6", "Webcam 4K", "Cable HDMI 2.1",
    ],
    "descripcion": [
        "Laptop gaming con RTX 3060, i7-12700H, 16GB RAM, SSD 512GB, pantalla 15.6\" 144Hz",
        "Mouse inalámbrico ergonómico con 6 botones programables, sensor 16000 DPI, batería 70h",
        "Teclado mecánico Cherry MX Brown, retroiluminación RGB, construcción aluminio",
        "Monitor IPS 4K 27 pulgadas, 144Hz, 1ms, HDR400, altavoces integrados",
        "Silla oficina ergonómica ajustable en altura, soporte lumbar, reposabrazos 3D",
        "Auriculares bluetooth 5.0, cancelación ruido activa, 30h batería, plegables",
        "Hub USB C 7 en 1: HDMI 4K, 2xUSB 3.0, SD, microSD, PD 100W",
        "SSD NVMe M.2 1TB, lectura 3500MB/s, escritura 3000MB/s, 5 años garantía",
        "Impresora láser color WiFi, dúplex automático, 25ppm, bandeja 250 hojas",
        "Router WiFi 6 AX3000, doble banda, 4 antenas, puerto gigabit, Mesh compatible",
        "Webcam 4K con micrófono estéreo, enfoque automático, corrección de luz",
        "Cable HDMI 2.1 de 3 metros, 48Gbps, 8K@60Hz, compatible con eARC",
    ],
    "precio": [1200, 45, 120, 450, 350, 80, 35, 150, 280, 90, 130, 25],
    "categoria": ["laptop", "periférico", "periférico", "monitor", "mueble", "audio",
                  "accesorio", "almacenamiento", "impresora", "red", "periférico", "cable"],
})

# ===== 2. MODELO E ÍNDICE =====
modelo = 'paraphrase-multilingual-MiniLM-L12-v2'
model = SentenceTransformer(modelo)
dim = model.get_sentence_embedding_dimension()

textos = (catalogo['nombre'] + " " + catalogo['descripcion']).tolist()
embeddings = np.array([model.encode(t, normalize_embeddings=True) for t in textos]).astype('float32')

# FAISS IndexFlatIP
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

# ===== 3. FUNCIÓN DE BÚSQUEDA =====
def buscar_productos(consulta, top_k=5, precio_max=None, categoria=None):
    query_emb = np.array([model.encode(consulta, normalize_embeddings=True)]).astype('float32')
    D, I = index.search(query_emb, top_k * 3)

    resultados = []
    for dist, idx in zip(D[0], I[0]):
        if idx >= len(catalogo):
            continue
        prod = catalogo.iloc[idx]
        if precio_max is not None and prod['precio'] > precio_max:
            continue
        if categoria is not None and prod['categoria'] != categoria:
            continue
        resultados.append({
            'id': prod['id'],
            'nombre': prod['nombre'],
            'precio': prod['precio'],
            'categoria': prod['categoria'],
            'descripcion': prod['descripcion'],
            'score': float(dist),
        })
        if len(resultados) == top_k:
            break
    return resultados

# ===== 4. DEMO =====
consultas = [
    "computadora portátil para juegos potente",
    "ratón cómodo para trabajar",
    "monitor grande para edición de video",
    "disco duro rápido para almacenamiento",
    "accesorio para conectar múltiples dispositivos",
]

print("=== SISTEMA DE BÚSQUEDA SEMÁNTICA PARA CATÁLOGO ===\n")
for consulta in consultas:
    start = time.time()
    resultados = buscar_productos(consulta, top_k=3)
    t = time.time() - start
    print(f"Consulta: '{consulta}' ({t*1000:.1f}ms)")
    for i, r in enumerate(resultados, 1):
        print(f"  #{i} {r['nombre']} (${r['precio']}) [{r['score']:.3f}]")
        print(f"      {r['descripcion'][:70]}...")
    print()

# ===== 5. EVALUACIÓN =====
print("=== EVALUACIÓN EN 10 CONSULTAS ===")
consultas_eval = [
    "laptop gaming", "mouse oficina", "teclado rgb", "monitor 4k",
    "silla cómoda", "audífonos inalámbricos", "hub usb", "ssd rápido",
    "impresora color", "router wifi",
]
tiempos = []
for c in consultas_eval:
    start = time.time()
    res = buscar_productos(c, top_k=1)
    tiempos.append(time.time() - start)
print(f"Tiempo promedio: {np.mean(tiempos)*1000:.1f}ms por consulta")
print(f"Tiempo total: {sum(tiempos)*1000:.1f}ms para {len(consultas_eval)} consultas")

# ===== 6. GUARDAR =====
faiss.write_index(index, "/tmp/index_catalogo.faiss")
catalogo.to_csv("/tmp/catalogo_productos.csv", index=False)
print("\nÍndice y catálogo guardados en /tmp/")

# ===== 7. BÚSQUEDA MULTILINGÜE =====
consultas_multi = {
    "EN": "gaming laptop",
    "ES": "computadora para juegos",
    "FR": "ordinateur de jeu",
}
print("\n=== BÚSQUEDA MULTILINGÜE ===")
for lang, consulta in consultas_multi.items():
    res = buscar_productos(consulta, top_k=1)
    print(f"[{lang}] '{consulta}' -> {res[0]['nombre']} (score: {res[0]['score']:.3f})")

# ===== 8. FILTROS =====
print("\n=== BÚSQUEDA CON FILTROS ===")
res = buscar_productos("periférico para computadora", top_k=3, precio_max=100)
print(f"Periféricos con precio < $100:")
for r in res:
    print(f"  {r['nombre']} - ${r['precio']}")

print("\n✅ INTEGRADOR COMPLETO: Sistema de búsqueda semántica multilingüe con filtros")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador — Sistema de Búsqueda Semántica para Catálogo.*

1. ===== 1. DATOS =====
2. ===== 2. MODELO E ÍNDICE =====
3. FAISS IndexFlatIP
4. ===== 3. FUNCIÓN DE BÚSQUEDA =====
5. ===== 4. DEMO =====
6. ===== 5. EVALUACIÓN =====
7. ===== 6. GUARDAR =====
8. ===== 7. BÚSQUEDA MULTILINGÜE =====

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

## 3. Ejercicios Propuestos

1. **Elegir el mejor modelo**: Compara 3 modelos (paraphrase-multilingual-MiniLM-L12-v2, distiluse-base-multilingual-cased-v2, all-MiniLM-L6-v2) en 10 consultas de ejemplo. ¿Cuál da mejores resultados para tu catálogo?

2. **FAISS con 1M vectores**: Genera 1M de vectores sintéticos (dim=384). Construye índices FlatIP, IVFFlat (nprobe=10, 20, 50) y HNSW. Mide tiempo de búsqueda y precisión relativa a FlatIP.

3. **Búsqueda híbrida avanzada**: Combina búsqueda semántica (peso 0.7) + búsqueda por palabras clave BM25 (peso 0.3). ¿Mejora los resultados? Implementa usando rank_bm25.

4. **Re-ranking**: Después de obtener top-20 con FAISS, re-rank usando cross-encoder (modelo de similitud textual más preciso). Compara top-5 antes y después del re-ranking.

5. **Sistema completo con API**: Crea una API REST con Flask/FastAPI que exponga: `GET /buscar?q=laptop+gaming&top_k=5&precio_max=1000`. Incluye documentación.

6. **Evaluación con juicio humano**: Pide a 3 personas que etiqueten 20 consultas con los productos relevantes. Calcula recall@k y precision@k para k=1,3,5,10.

7. **Búsqueda multimodal**: Extiende el sistema para incluir embeddings de imágenes (usando CLIP) y buscar tanto por texto como por imagen. Compara vs solo texto.

8. **Proyecto final**: Construye un sistema completo con: (a) carga de catálogo desde CSV, (b) indexación con FAISS (HNSW), (c) búsqueda semántica con filtros, (d) API REST, (e) evaluación de calidad, (f) interfaz simple en Streamlit.

---

## 4. Resumen

| Componente | Propósito | Aplicación |
|---|---|---|
| SentenceTransformer | Embeddings de texto | Codificar descripciones |
| util.cos_sim / util.semantic_search | Similitud coseno | Buscar productos similares |
| IndexFlatIP | Búsqueda exacta | Catálogos < 10k |
| IndexIVFFlat | Búsqueda aproximada rápida | Catálogos 10k-1M |
| IndexHNSWFlat | Búsqueda en grafos | Catálogos > 100k |
| faiss.normalize_L2 | Normalizar para coseno | Pre-procesamiento |
| normalize_embeddings=True | Normalización en encode | Conveniencia |
| Búsqueda multilingüe | Queries en cualquier idioma | Catálogos internacionales |
| Filtros híbridos | Semántica + estructurado | Precio, categoría, stock |

**Conclusión**: La combinación de Sentence-Transformers para embeddings + FAISS para indexación permite construir sistemas de búsqueda semántica que entienden el significado de las consultas y encuentran productos relevantes incluso cuando no hay coincidencia exacta de palabras clave. Es la base de los motores de búsqueda modernos en e-commerce.
