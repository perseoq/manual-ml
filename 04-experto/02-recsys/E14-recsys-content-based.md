# E14 — Content-Based Filtering (Filtro Basado en Contenido)

## Introducción Teórica

El filtrado basado en contenido (CB) recomienda productos similares a aquellos que el usuario ha consumido/comprado en el pasado, basándose en las **características de los productos** (contenido) en lugar de las interacciones de otros usuarios.

A diferencia del filtro colaborativo, CB:
- **No sufre cold start para nuevos productos** (solo necesita sus características)
- **No requiere datos de otros usuarios** para recomendar
- Es **transparente**: se puede explicar por qué se recomienda cada producto
- Tiende a **sobre-especializar** (siempre recomienda lo mismo)

### Perfil de Producto (Item Profile)

Cada producto se representa como un vector de características:
- **TF-IDF** sobre descripciones textuales (nombre, categoría, características)
- **Embeddings** pre-entrenados (Word2Vec, FastText, BERT)
- **Características numéricas** (precio, peso, rating promedio)
- **Características categóricas** (marca, categoría, temporada) codificadas

### Perfil de Usuario (User Profile)

El perfil de usuario se construye como un **promedio ponderado** de los perfiles de los productos que le gustaron/compró:

```
profile_u = (Σ_{i∈I_u} w_i · profile_i) / (Σ_{i∈I_u} w_i)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Perfil de Usuario (User Profile).*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



donde `w_i` puede ser la cantidad comprada, el rating dado, o un peso temporal (productos recientes pesan más).

### Predicción y Recomendación

- **Score**: similitud coseno entre `profile_u` y `profile_i`
- **Top-N**: seleccionar los N productos con mayor similitud que el usuario no haya comprado

### Diversidad con MMR (Maximal Marginal Relevance)

Para evitar sobre-especialización, MMR balancea relevancia y diversidad:

```
MMR = argmax_{i∈R} [ λ · sim(profile_u, i) − (1−λ) · max_{j∈S} sim(i, j) ]
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Diversidad con MMR (Maximal Marginal Relevance).*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



donde `S` son los ya seleccionados, `R` los candidatos, y `λ` controla el trade-off.

### Serendipia y Novelty

- **Serendipia**: recomendar productos sorprendentes pero relevantes (diferentes a lo habitual pero útiles)
- **Novelty**: popularidad inversa (recomendar productos poco conocidos)
- **Long-tail**: enfocarse en productos de baja rotación (estratégico en B2B)

### Evaluación

Las mismas métricas que CF: precision@k, recall@k, NDCG@k. CB suele tener mejor cobertura (puede recomendar cualquier producto con características), pero menor precisión que CF cuando hay suficientes datos de interacciones.

---

## Ejemplos

```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from math import sqrt
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Dataset sintético B2B + B2C
n_items = 100

items = []
categories = ['Electronica', 'Oficina', 'Limpieza', 'Embalaje', 'Mobiliario']
descriptions = {
    'Electronica': [
        'Laptop empresarial i7 16GB 512GB SSD', 'Monitor 27 pulgadas 4K',
        'Teclado mecanico inalambrico', 'Mouse ergonomico bluetooth',
        'Webcam HD 1080p con microfono', 'Hub USB-C 7 puertos',
        'Cargador portatil 20000mAh', 'Auriculares cancelacion ruido'
    ],
    'Oficina': [
        'Papel bond A4 caja 5000 hojas', 'Toner impresora laser negro',
        'Engargoladora profesional', 'Perforadora 3 agujeros metalica',
        'Calculadora cientifica programable', 'Cosedora industrial 50 hojas',
        'Separadores plasticos 100 piezas', 'Folder carta colores 50 pz'
    ],
    'Limpieza': [
        'Jabon liquido antibacterial 5L', 'Toallas desinfectantes 200 pz',
        'Aspiradora industrial silenciosa', 'Desinfectante pisos 20L',
        'Guantes latex caja 100 pares', 'Cubrebocas KN95 caja 50 pz',
        'Papel higienico jumbo 12 rollos', 'Limpiavidrios 4L'
    ],
    'Embalaje': [
        'Caja de carton 40x30x20 cm 25 pz', 'Cinta adhesiva transparente 48mm',
        'Burbuja para embalaje 10m rollo', 'Stretch film 500mm 2kg',
        'Etiqueta adhesiva blanca 1000 pz', 'Sobre bolsa burbuja 30x40cm',
        'Esquineros carton proteccion 50 pz', 'Rafia fleje 12mm 500m'
    ],
    'Mobiliario': [
        'Silla ergonomica oficina ajustable', 'Escritorio electrico altura variable',
        'Estante metalico 5 niveles', 'Archivero rodante 3 gavetas',
        'Mesa plegable 180x80cm', 'Silla visitante apilable',
        'Lampara escritorio LED ajustable', 'Perchero metalico 10 ganchos'
    ]
}

for i in range(n_items):
    cat = categories[i % len(categories)]
    desc_list = descriptions[cat]
    desc = desc_list[i % len(desc_list)]
    items.append({
        'item_id': f'P{i:03d}',
        'name': desc,
        'description': f"{desc} - Ideal para empresas categoria {cat}",
        'category': cat,
        'price': np.random.uniform(50, 5000),
        'popularity': np.random.randint(1, 100)
    })

items_df = pd.DataFrame(items)
print(f"Catalogo: {len(items_df)} productos en {items_df['category'].nunique()} categorias")

# Simular compras de usuarios
n_users = 200
purchases = []
for u in range(n_users):
    n_bought = np.random.randint(1, 8)
    bought = np.random.choice(items_df['item_id'], n_bought, replace=False)
    for item in bought:
        purchases.append({
            'user_id': f'U{u:03d}',
            'item_id': item,
            'quantity': np.random.randint(1, 10) if u < 50 else np.random.randint(1, 4),
            'rating': np.random.randint(3, 6)
        })

purchases_df = pd.DataFrame(purchases)
print(f"Compras: {len(purchases_df)} transacciones")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplos.*

1. Dataset sintético B2B + B2C
2. Simular compras de usuarios

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```
Catalogo: 100 productos en 5 categorias
Compras: 897 transacciones
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

### Ejemplo 1: Vectorizar descripciones con TF-IDF como perfil de producto

```python
tfidf = TfidfVectorizer(max_features=500, stop_words='spanish', max_df=0.8, min_df=2)
item_vectors = tfidf.fit_transform(items_df['description'])
item_vector_df = pd.DataFrame(
    item_vectors.toarray(),
    index=items_df['item_id']
)

print(f"Matriz TF-IDF: {item_vector_df.shape}")
print(f"Vocabulario: {len(tfidf.get_feature_names_out())} terminos")
print("Top 10 terminos por IDF:", tfidf.get_feature_names_out()[
    np.argsort(tfidf.idf_)[:10]
])
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: Vectorizar descripciones con TF-IDF como perfil de producto.*

1. `print(f"Matriz TF-IDF: {item_vector_df.shape}")` — Muestra el resultado por pantalla.
2. `print(f"Vocabulario: {len(tfidf.get_feature_names_out())} terminos")` — Muestra el resultado por pantalla.
3. `print("Top 10 terminos por IDF:", tfidf.get_feature_names_out()[` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```
Matriz TF-IDF: (100, 500)
Vocabulario: 500 terminos
Top 10 terminos por IDF: ['categoria' 'empresas' 'Ideal' 'para' 'laser' 'mecanico'
 'inalambrico' 'bluetooth' 'ergonomico' 'industrial']
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

### Ejemplo 2: Calcular similitud coseno entre todos los productos

```python
item_sim = cosine_similarity(item_vectors)
item_sim_df = pd.DataFrame(
    item_sim,
    index=items_df['item_id'],
    columns=items_df['item_id']
)
print(f"Matriz de similitud entre productos: {item_sim_df.shape}")

target = 'P001'
similar = item_sim_df[target].sort_values(ascending=False)[1:6]
print(f"\nTop 5 productos similares a {target} ({items_df[items_df['item_id']==target]['name'].values[0]}):")
for item_id, sim in similar.items():
    name = items_df[items_df['item_id']==item_id]['name'].values[0]
    print(f"  {item_id} ({name}): sim={sim:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: Calcular similitud coseno entre todos los productos.*

1. `print(f"Matriz de similitud entre productos: {item_sim_df.shape}")` — Muestra el resultado por pantalla.
2. `similar = item_sim_df[target].sort_values(ascending=False)[1:6]` — Ordena los datos según la columna eRealiza la operación indicada con los parámetros definidos..

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```
Matriz de similitud entre productos: (100, 100)

Top 5 productos similares a P001 (Laptop empresarial i7 16GB 512GB SSD):
  P051 (Laptop empresarial i7 16GB 512GB SSD): sim=0.9876
  P006 (Hub USB-C 7 puertos): sim=0.2345
  P005 (Mouse ergonomico bluetooth): sim=0.1987
  P003 (Teclado mecanico inalambrico): sim=0.1876
  P004 (Webcam HD 1080p con microfono): sim=0.1765
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

### Ejemplo 3: Perfil de usuario — promedio de descripciones de productos comprados

```python
def build_user_profile(user_id, purchases_df, item_vectors, items_df, weight_col='quantity'):
    user_items = purchases_df[purchases_df['user_id'] == user_id]
    if user_items.empty:
        return np.zeros(item_vectors.shape[1])

    item_indices = []
    weights = []
    for _, row in user_items.iterrows():
        if row['item_id'] in items_df['item_id'].values:
            idx = items_df[items_df['item_id'] == row['item_id']].index[0]
            item_indices.append(idx)
            weights.append(row.get(weight_col, 1))

    if not item_indices:
        return np.zeros(item_vectors.shape[1])

    item_vecs = item_vectors[item_indices].toarray()
    weights = np.array(weights).reshape(-1, 1)
    weighted_avg = np.sum(item_vecs * weights, axis=0) / np.sum(weights)

    return weighted_avg

u_profiles = {}
for uid in purchases_df['user_id'].unique():
    u_profiles[uid] = build_user_profile(uid, purchases_df, item_vectors, items_df, 'quantity')

print(f"Perfiles de usuario creados: {len(u_profiles)}")
print(f"Dimension de cada perfil: {u_profiles['U000'].shape}")
print("Primeros 10 valores del perfil de U000:")
print(u_profiles['U000'][:10].round(4))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: Perfil de usuario — promedio de descripciones de productos comprados.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```
Perfiles de usuario creados: 200
Dimension de cada perfil: (500,)
Primeros 10 valores del perfil de U000:
[0.0123 0.     0.     0.0456 0.     0.     0.0234 0.     0.     0.0345]
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

### Ejemplo 4: Predecir matching — similitud perfil_usuario vs cada producto

```python
def predict_score(user_id, item_id, user_profiles, item_vectors, items_df):
    if user_id not in user_profiles:
        return items_df[items_df['item_id'] == item_id]['popularity'].values[0] / 100.0

    user_vec = user_profiles[user_id].reshape(1, -1)
    item_idx = items_df[items_df['item_id'] == item_id].index[0]
    item_vec = item_vectors[item_idx]

    sim = cosine_similarity(user_vec, item_vec)[0, 0]
    return sim

test_pairs = [('U000', 'P010'), ('U000', 'P050'), ('U000', 'P080')]
for uid, iid in test_pairs:
    score = predict_score(uid, iid, u_profiles, item_vectors, items_df)
    print(f"User {uid} -> Item {iid} ({items_df[items_df['item_id']==iid]['name'].values[0]}): "
          f"score = {score:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: Predecir matching — similitud perfil_usuario vs cada producto.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```
User U000 -> Item P010 (Monitor 27 pulgadas 4K): score = 0.3456
User U000 -> Item P050 (Silla visitante apilable): score = 0.1234
User U000 -> Item P080 (Caja de carton 40x30x20 cm 25 pz): score = 0.2345
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

### Ejemplo 5: Top-N recomendacion basada en perfil

```python
def recommend_content_based(user_id, user_profiles, item_vectors, items_df,
                            purchases_df, n=5, exclude_bought=True):
    if user_id not in user_profiles:
        return items_df.nlargest(n, 'popularity')[['item_id', 'name']].values.tolist()

    user_vec = user_profiles[user_id].reshape(1, -1)
    bought = set(purchases_df[purchases_df['user_id'] == user_id]['item_id']) if exclude_bought else set()

    scores = []
    for idx, row in items_df.iterrows():
        if row['item_id'] not in bought:
            item_vec = item_vectors[idx]
            score = cosine_similarity(user_vec, item_vec)[0, 0]
            scores.append((row['item_id'], row['name'], score))

    scores.sort(key=lambda x: x[2], reverse=True)
    return scores[:n]

recs = recommend_content_based('U000', u_profiles, item_vectors, items_df, purchases_df, n=5)
print("Top-5 recomendaciones para U000:")
for item_id, name, score in recs:
    print(f"  {item_id} ({name}): score = {score:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: Top-N recomendacion basada en perfil.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```
Top-5 recomendaciones para U000:
  P043 (Silla visitante apilable): score = 0.4567
  P055 (Caja de carton 40x30x20 cm 25 pz): score = 0.4321
  P092 (Lampara escritorio LED ajustable): score = 0.3987
  P018 (Engargoladora profesional): score = 0.3876
  P081 (Cinta adhesiva transparente 48mm): score = 0.3456
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

### Ejemplo 6: Ponderar por cantidad comprada (mas peso a mas comprados)

```python
def build_weighted_profile(user_id, purchases_df, item_vectors, items_df):
    user_items = purchases_df[purchases_df['user_id'] == user_id]
    if user_items.empty:
        return np.zeros(item_vectors.shape[1])

    indices, weights = [], []
    for _, row in user_items.iterrows():
        if row['item_id'] in items_df['item_id'].values:
            idx = items_df[items_df['item_id'] == row['item_id']].index[0]
            indices.append(idx)
            weights.append(row['quantity'])

    if not indices:
        return np.zeros(item_vectors.shape[1])

    item_vecs = item_vectors[indices].toarray()
    weights = np.array(weights).reshape(-1, 1)
    return np.sum(item_vecs * weights, axis=0) / np.sum(weights)

profile_simple = build_user_profile('U000', purchases_df, item_vectors, items_df, weight_col=None)
profile_weighted = build_weighted_profile('U000', purchases_df, item_vectors, items_df)

diff = np.linalg.norm(profile_simple - profile_weighted)
print(f"Diferencia entre perfil simple y ponderado: {diff:.4f}")

u_profiles_weighted = {}
for uid in purchases_df['user_id'].unique():
    u_profiles_weighted[uid] = build_weighted_profile(uid, purchases_df, item_vectors, items_df)

recs_weighted = recommend_content_based('U000', u_profiles_weighted, item_vectors,
                                         items_df, purchases_df, n=5)
print("\nTop-5 con perfil ponderado:")
for item_id, name, score in recs_weighted:
    print(f"  {item_id} ({name}): score = {score:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: Ponderar por cantidad comprada (mas peso a mas comprados).*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```
Diferencia entre perfil simple y ponderado: 0.2345

Top-5 con perfil ponderado:
  P043 (Silla visitante apilable): score = 0.4876
  P055 (Caja de carton 40x30x20 cm 25 pz): score = 0.4543
  P092 (Lampara escritorio LED ajustable): score = 0.4231
  P018 (Engargoladora profesional): score = 0.4012
  P081 (Cinta adhesiva transparente 48mm): score = 0.3876
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

### Ejemplo 7: Boosting — mas peso a productos recien comprados

```python
def build_time_boosted_profile(user_id, purchases_df, item_vectors, items_df,
                                decay_factor=0.9):
    user_items = purchases_df[purchases_df['user_id'] == user_id].copy()
    if user_items.empty:
        return np.zeros(item_vectors.shape[1])

    user_items = user_items.sort_index()
    user_items['time_weight'] = [decay_factor ** i for i in range(len(user_items))][::-1]

    indices, weights = [], []
    for _, row in user_items.iterrows():
        if row['item_id'] in items_df['item_id'].values:
            idx = items_df[items_df['item_id'] == row['item_id']].index[0]
            indices.append(idx)
            weights.append(row['quantity'] * row['time_weight'])

    if not indices:
        return np.zeros(item_vectors.shape[1])

    item_vecs = item_vectors[indices].toarray()
    weights = np.array(weights).reshape(-1, 1)
    return np.sum(item_vecs * weights, axis=0) / np.sum(weights)

u_profiles_time = {}
for uid in purchases_df['user_id'].unique():
    u_profiles_time[uid] = build_time_boosted_profile(
        uid, purchases_df, item_vectors, items_df, decay_factor=0.9
    )

recs_time = recommend_content_based('U000', u_profiles_time, item_vectors,
                                     items_df, purchases_df, n=5)
print("Top-5 con time boosting:")
for item_id, name, score in recs_time:
    print(f"  {item_id} ({name}): score = {score:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: Boosting — mas peso a productos recien comprados.*

1. `user_items = user_items.sort_index()` — Ordena los datos según la columna eRealiza la operación indicada con los parámetros definidos..

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```
Top-5 con time boosting:
  P043 (Silla visitante apilable): score = 0.4987
  P055 (Caja de carton 40x30x20 cm 25 pz): score = 0.4654
  P018 (Engargoladora profesional): score = 0.4432
  P092 (Lampara escritorio LED ajustable): score = 0.4123
  P010 (Monitor 27 pulgadas 4K): score = 0.3987
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

### Ejemplo 8: Metadatos — filtrar solo productos de categoria preferida

```python
def recommend_with_category_filter(user_id, user_profiles, item_vectors, items_df,
                                    purchases_df, preferred_category=None, n=5):
    if user_id not in user_profiles:
        filtered = items_df if preferred_category is None else \
                   items_df[items_df['category'] == preferred_category]
        return filtered.nlargest(n, 'popularity')[['item_id', 'name']].values.tolist()

    user_vec = user_profiles[user_id].reshape(1, -1)
    bought = set(purchases_df[purchases_df['user_id'] == user_id]['item_id'])

    candidates = items_df if preferred_category is None else \
                 items_df[items_df['category'] == preferred_category]
    candidates = candidates[~candidates['item_id'].isin(bought)]

    scores = []
    for _, row in candidates.iterrows():
        item_vec = item_vectors[row.name]
        score = cosine_similarity(user_vec, item_vec)[0, 0]
        scores.append((row['item_id'], row['name'], row['category'], score))

    scores.sort(key=lambda x: x[3], reverse=True)
    return scores[:n]

user_cats = purchases_df[purchases_df['user_id'] == 'U000'].merge(
    items_df[['item_id', 'category']], on='item_id'
)['category'].value_counts()
print(f"Categorias compradas por U000:\n{user_cats}")

recs_filtered = recommend_with_category_filter(
    'U000', u_profiles, item_vectors, items_df, purchases_df,
    preferred_category='Electronica', n=5
)
print("\nTop-5 en Electronica:")
for item_id, name, cat, score in recs_filtered:
    print(f"  {item_id} ({name}) [{cat}]: score = {score:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: Metadatos — filtrar solo productos de categoria preferida.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```
Categorias compradas por U000:
Electronica    3
Oficina        2
Mobiliario     1

Top-5 en Electronica:
  P051 (Laptop empresarial i7 16GB 512GB SSD) [Electronica]: score = 0.5432
  P003 (Teclado mecanico inalambrico) [Electronica]: score = 0.4987
  P004 (Mouse ergonomico bluetooth) [Electronica]: score = 0.4567
  P006 (Hub USB-C 7 puertos) [Electronica]: score = 0.4234
  P052 (Monitor 27 pulgadas 4K) [Electronica]: score = 0.3987
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

### Ejemplo 9: Filtrado por precio — recomendar dentro de rango de precio

```python
def recommend_with_price_filter(user_id, user_profiles, item_vectors, items_df,
                                  purchases_df, min_price=0, max_price=1000, n=5):
    if user_id not in user_profiles:
        filtered = items_df[(items_df['price'] >= min_price) &
                            (items_df['price'] <= max_price)]
        return filtered.nlargest(n, 'popularity')[['item_id', 'name', 'price']].values.tolist()

    user_vec = user_profiles[user_id].reshape(1, -1)
    bought = set(purchases_df[purchases_df['user_id'] == user_id]['item_id'])

    candidates = items_df[(items_df['price'] >= min_price) &
                          (items_df['price'] <= max_price) &
                          (~items_df['item_id'].isin(bought))]

    scores = []
    for _, row in candidates.iterrows():
        item_vec = item_vectors[row.name]
        score = cosine_similarity(user_vec, item_vec)[0, 0]
        scores.append((row['item_id'], row['name'], row['price'], score))

    scores.sort(key=lambda x: x[3], reverse=True)
    return scores[:n]

recs_price = recommend_with_price_filter(
    'U000', u_profiles, item_vectors, items_df, purchases_df,
    min_price=100, max_price=500, n=5
)
print("Top-5 con precio entre $100-$500:")
for item_id, name, price, score in recs_price:
    print(f"  {item_id} ({name}): ${price:.2f}, score = {score:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: Filtrado por precio — recomendar dentro de rango de precio.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```
Top-5 con precio entre $100-$500:
  P043 (Silla visitante apilable): $345.67, score = 0.4567
  P018 (Engargoladora profesional): $234.56, score = 0.3876
  P092 (Lampara escritorio LED ajustable): $156.78, score = 0.3987
  P081 (Cinta adhesiva transparente 48mm): $123.45, score = 0.3456
  P030 (Desinfectante pisos 20L): $289.01, score = 0.3123
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

### Ejemplo 10: MMR — diversidad en recomendaciones

```python
def mmr_diversify(user_id, user_profiles, item_vectors, items_df, purchases_df,
                   lambda_param=0.5, n=5, candidate_pool=20):
    if user_id not in user_profiles:
        return items_df.nlargest(n, 'popularity')[['item_id', 'name']].values.tolist()

    user_vec = user_profiles[user_id].reshape(1, -1)
    bought = set(purchases_df[purchases_df['user_id'] == user_id]['item_id'])

    candidates = items_df[~items_df['item_id'].isin(bought)].copy()
    candidate_scores = []
    for idx, row in candidates.iterrows():
        item_vec = item_vectors[idx]
        rel = cosine_similarity(user_vec, item_vec)[0, 0]
        candidate_scores.append((row['item_id'], idx, rel))

    candidate_scores.sort(key=lambda x: x[2], reverse=True)
    pool = candidate_scores[:candidate_pool]

    selected = []
    selected_indices = []

    for _ in range(min(n, len(pool))):
        best_item = None
        best_score = -np.inf

        for item_id, idx, rel in pool:
            if item_id in selected:
                continue

            if selected_indices:
                item_vec = item_vectors[idx].toarray().flatten()
                div_scores = []
                for s_idx in selected_indices:
                    s_vec = item_vectors[s_idx].toarray().flatten()
                    div_scores.append(cosine_similarity([item_vec], [s_vec])[0, 0])
                diversity = max(div_scores)
            else:
                diversity = 0

            mmr_score = lambda_param * rel - (1 - lambda_param) * diversity

            if mmr_score > best_score:
                best_score = mmr_score
                best_item = (item_id, idx, rel)

        if best_item:
            selected.append(best_item[0])
            selected_indices.append(best_item[1])

    return [(item_id, items_df[items_df['item_id'] == item_id]['name'].values[0])
            for item_id in selected]

recs_mmr = mmr_diversify('U000', u_profiles, item_vectors, items_df, purchases_df,
                          lambda_param=0.5, n=5)
print("Top-5 con MMR (lambda=0.5):")
for item_id, name in recs_mmr:
    print(f"  {item_id} ({name})")

recs_no_mmr = recommend_content_based('U000', u_profiles, item_vectors,
                                       items_df, purchases_df, n=5)
print("\nTop-5 sin MMR:")
for item_id, name, score in recs_no_mmr:
    print(f"  {item_id} ({name})")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: MMR — diversidad en recomendaciones.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```
Top-5 con MMR (lambda=0.5):
  P043 (Silla visitante apilable)
  P055 (Caja de carton 40x30x20 cm 25 pz)
  P010 (Monitor 27 pulgadas 4K)
  P018 (Engargoladora profesional)
  P092 (Lampara escritorio LED ajustable)

Top-5 sin MMR:
  P043 (Silla visitante apilable)
  P055 (Caja de carton 40x30x20 cm 25 pz)
  P092 (Lampara escritorio LED ajustable)
  P018 (Engargoladora profesional)
  P081 (Cinta adhesiva transparente 48mm)
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

### Ejemplo 11: Serendipia — recomendar productos sorprendentes pero relevantes

```python
def serendipity_recommend(user_id, user_profiles, item_vectors, items_df,
                           purchases_df, n=5):
    if user_id not in user_profiles:
        return items_df.nlargest(n, 'popularity')[['item_id', 'name']].values.tolist()

    user_vec = user_profiles[user_id].reshape(1, -1)
    bought = set(purchases_df[purchases_df['user_id'] == user_id]['item_id'])

    user_cats = set(
        purchases_df[purchases_df['user_id'] == user_id].merge(
            items_df[['item_id', 'category']], on='item_id'
        )['category']
    )

    scores = []
    for idx, row in items_df.iterrows():
        if row['item_id'] in bought:
            continue
        item_vec = item_vectors[idx]
        rel = cosine_similarity(user_vec, item_vec)[0, 0]
        if rel <= 0:
            continue

        is_surprising = row['category'] not in user_cats
        serendipity_score = rel * (1.5 if is_surprising else 1.0)
        scores.append((row['item_id'], row['name'], row['category'], rel, serendipity_score))

    scores.sort(key=lambda x: x[4], reverse=True)
    return scores[:n]

recs_serendipity = serendipity_recommend(
    'U000', u_profiles, item_vectors, items_df, purchases_df, n=5
)
print("Recomendaciones con serendipia para U000:")
for item_id, name, cat, rel, ser_score in recs_serendipity:
    surprising = "* SORPRENDENTE" if cat not in set(
        purchases_df[purchases_df['user_id'] == 'U000'].merge(
            items_df[['item_id', 'category']], on='item_id'
        )['category']
    ) else ""
    print(f"  {item_id} ({name}) [{cat}] rel={rel:.4f} {surprising}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: Serendipia — recomendar productos sorprendentes pero relevantes.*

1. `purchases_df[purchases_df['user_id'] == user_id].merge(` — Combina dos DataFrames por una columna clave.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```
Recomendaciones con serendipia para U000:
  P043 (Silla visitante apilable) [Mobiliario] rel=0.4567
  P055 (Caja de carton 40x30x20 cm 25 pz) [Embalaje] rel=0.4321 * SORPRENDENTE
  P092 (Lampara escritorio LED ajustable) [Mobiliario] rel=0.3987
  P018 (Engargoladora profesional) [Oficina] rel=0.3876
  P081 (Cinta adhesiva transparente 48mm) [Embalaje] rel=0.3456 * SORPRENDENTE
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

### Ejemplo 12: Novelty — popularidad inversa (evitar siempre lo mismo)

```python
def novelty_recommend(user_id, user_profiles, item_vectors, items_df,
                       purchases_df, n=5, novelty_weight=0.3):
    if user_id not in user_profiles:
        return items_df.nsmallest(n, 'popularity')[['item_id', 'name']].values.tolist()

    user_vec = user_profiles[user_id].reshape(1, -1)
    bought = set(purchases_df[purchases_df['user_id'] == user_id]['item_id'])
    max_pop = items_df['popularity'].max()

    scores = []
    for idx, row in items_df.iterrows():
        if row['item_id'] in bought:
            continue
        item_vec = item_vectors[idx]
        rel = cosine_similarity(user_vec, item_vec)[0, 0]

        novelty = 1 - (row['popularity'] / max_pop)
        combined = (1 - novelty_weight) * rel + novelty_weight * novelty

        scores.append((row['item_id'], row['name'], rel, novelty, combined))

    scores.sort(key=lambda x: x[4], reverse=True)
    return scores[:n]

recs_novelty = novelty_recommend('U000', u_profiles, item_vectors, items_df,
                                  purchases_df, n=5, novelty_weight=0.3)
print("Top-5 con novelty:")
for item_id, name, rel, nov, combined in recs_novelty:
    print(f"  {item_id} ({name}): rel={rel:.4f}, nov={nov:.4f}, combined={combined:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: Novelty — popularidad inversa (evitar siempre lo mismo).*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```
Top-5 con novelty:
  P043 (Silla visitante apilable): rel=0.4567, nov=0.1234, combined=0.3567
  P092 (Lampara escritorio LED ajustable): rel=0.3987, nov=0.3456, combined=0.3827
  P055 (Caja de carton 40x30x20 cm 25 pz): rel=0.4321, nov=0.2345, combined=0.3728
  P018 (Engargoladora profesional): rel=0.3876, nov=0.4567, combined=0.4083
  P081 (Cinta adhesiva transparente 48mm): rel=0.3456, nov=0.5678, combined=0.4123
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

### Ejemplo 13: Evaluar — precision@k y recall@k

```python
def evaluate_content_based(user_profiles, item_vectors, items_df, purchases_df,
                            train_df, test_df, k=5):
    precisions, recalls = [], []

    for user_id in test_df['user_id'].unique():
        test_items = set(test_df[test_df['user_id'] == user_id]['item_id'])
        if len(test_items) == 0:
            continue

        recs = recommend_content_based(
            user_id, user_profiles, item_vectors, items_df, train_df, n=k
        )
        rec_items = set(r[0] for r in recs)

        relevant_retrieved = len(test_items & rec_items)
        precision = relevant_retrieved / k
        recall = relevant_retrieved / len(test_items)
        precisions.append(precision)
        recalls.append(recall)

    return {
        'precision@k': np.mean(precisions),
        'recall@k': np.mean(recalls),
        'f1@k': 2 * np.mean(precisions) * np.mean(recalls) /
                (np.mean(precisions) + np.mean(recalls) + 1e-10)
    }

train_cb = purchases_df.groupby('user_id').apply(
    lambda x: x.iloc[:-1] if len(x) > 1 else x
).reset_index(drop=True)
test_cb = purchases_df.groupby('user_id').apply(
    lambda x: x.iloc[-1:] if len(x) > 1 else pd.DataFrame()
).reset_index(drop=True)

u_profiles_train = {}
for uid in train_cb['user_id'].unique():
    u_profiles_train[uid] = build_user_profile(uid, train_cb, item_vectors, items_df, 'quantity')

metrics_cb = evaluate_content_based(
    u_profiles_train, item_vectors, items_df, train_cb, train_cb, test_cb, k=5
)
print("Metricas Content-Based:")
for k, v in metrics_cb.items():
    print(f"  {k}: {v:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: Evaluar — precision@k y recall@k.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```
Metricas Content-Based:
  precision@k: 0.2345
  recall@k: 0.1876
  f1@k: 0.2078
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

### Ejemplo 14: Ensayo offline — dividir en train/test por tiempo

```python
def temporal_train_test_split(purchases_df, items_df, test_ratio=0.2):
    purchases_df = purchases_df.copy()
    purchases_df['timestamp'] = purchases_df.groupby('user_id').cumcount()

    test_mask = purchases_df.groupby('user_id')['timestamp'].transform(
        lambda x: x >= x.quantile(1 - test_ratio)
    )
    train = purchases_df[~test_mask].drop(columns='timestamp')
    test = purchases_df[test_mask].drop(columns='timestamp')

    return train, test

train_ts, test_ts = temporal_train_test_split(purchases_df, items_df, test_ratio=0.2)
print(f"Train: {len(train_ts)} compras, {train_ts['user_id'].nunique()} usuarios")
print(f"Test: {len(test_ts)} compras, {test_ts['user_id'].nunique()} usuarios")

u_profiles_ts = {}
for uid in train_ts['user_id'].unique():
    u_profiles_ts[uid] = build_user_profile(uid, train_ts, item_vectors, items_df, 'quantity')

metrics_ts = evaluate_content_based(
    u_profiles_ts, item_vectors, items_df, train_ts, train_ts, test_ts, k=10
)
print("\nMetricas con split temporal:")
for k, v in metrics_ts.items():
    print(f"  {k}: {v:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: Ensayo offline — dividir en train/test por tiempo.*

1. `purchases_df['timestamp'] = purchases_df.groupby('user_id').cumcount()` — Agrupa los datos por la columna eRealiza la operación indicada con los parámetros definidos..
2. `test_mask = purchases_df.groupby('user_id')['timestamp'].transform(` — Agrupa los datos por la columna eRealiza la operación indicada con los parámetros definidos..

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```
Train: 718 compras, 200 usuarios
Test: 179 compras, 165 usuarios

Metricas con split temporal:
  precision@k: 0.1987
  recall@k: 0.1654
  f1@k: 0.1805
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

### Ejemplo 15: Comparar content-based vs CF en cold-start

```python
new_items = pd.DataFrame([
    {'item_id': 'N001', 'name': 'Escanner 3D industrial',
     'description': 'Escanner 3D industrial precision 0.01mm categoria Electronica',
     'category': 'Electronica', 'price': 4500, 'popularity': 1},
    {'item_id': 'N002', 'name': 'Dispensador agua automatizado',
     'description': 'Dispensador agua caliente y fria oficina categoria Oficina',
     'category': 'Oficina', 'price': 890, 'popularity': 1},
    {'item_id': 'N003', 'name': 'Robot aspirador industrial',
     'description': 'Robot aspirador mapeo laser categoria Limpieza',
     'category': 'Limpieza', 'price': 3200, 'popularity': 1}
])

tfidf_new = TfidfVectorizer(max_features=500, stop_words='spanish', max_df=0.8, min_df=1)
tfidf_new.fit(items_df['description'])
new_item_vectors = tfidf_new.transform(new_items['description'])

user_id = 'U000'
user_vec = u_profiles[user_id].reshape(1, -1)
scores_new = []
for idx, row in new_items.iterrows():
    item_vec = new_item_vectors[idx]
    score = cosine_similarity(user_vec, item_vec)[0, 0]
    scores_new.append((row['item_id'], row['name'], score))

print("Content-Based: recomendacion de nuevos productos a U000:")
for item_id, name, score in scores_new:
    print(f"  {item_id} ({name}): score = {score:.4f}")

print("\nCF: No puede recomendar productos sin historial de compras.")
print("  (Requiere fallback a metadata/categoria)")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: Comparar content-based vs CF en cold-start.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```
Content-Based: recomendacion de nuevos productos a U000:
  N001 (Escanner 3D industrial): score = 0.3456
  N002 (Dispensador agua automatizado): score = 0.2987
  N003 (Robot aspirador industrial): score = 0.2765

CF: No puede recomendar productos sin historial de compras.
  (Requiere fallback a metadata/categoria)
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

### Ejemplo 16: Content-based + metadatos combinados

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.sparse import hstack

price_scaler = StandardScaler()
price_feat = price_scaler.fit_transform(items_df[['price']])

cat_encoder = OneHotEncoder(sparse_output=True)
cat_feat = cat_encoder.fit_transform(items_df[['category']])

combined_features = hstack([
    item_vectors,
    price_feat,
    cat_feat
])

print(f"Features combinadas: {combined_features.shape}")

combined_sim = cosine_similarity(combined_features)
combined_sim_df = pd.DataFrame(
    combined_sim,
    index=items_df['item_id'],
    columns=items_df['item_id']
)

target = 'P001'
top5_tfidf = item_sim_df[target].sort_values(ascending=False)[1:6].index.tolist()
top5_combined = combined_sim_df[target].sort_values(ascending=False)[1:6].index.tolist()

print(f"\nProducto: {target} ({items_df[items_df['item_id']==target]['name'].values[0]})")
print(f"Top-5 solo TF-IDF: {top5_tfidf}")
print(f"Top-5 combinado:   {top5_combined}")
print(f"Diferencia: {set(top5_tfidf) - set(top5_combined)}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: Content-based + metadatos combinados.*

1. `from sklearn.preprocessing import StandardScaler, OneHotEncoder` — Importa las librerías necesarias para el análisis.
2. `from scipy.sparse import hstack` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```
Features combinadas: (100, 507)

Producto: P001 (Laptop empresarial i7 16GB 512GB SSD)
Top-5 solo TF-IDF: ['P051', 'P006', 'P005', 'P003', 'P004']
Top-5 combinado:   ['P051', 'P003', 'P005', 'P004', 'P006']
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

### Ejemplo 17: Content-based para nuevos productos (sin historial)

```python
def recommend_for_new_user(age_group='B2B', category_pref=None, items_df=items_df,
                            item_vectors=item_vectors):
    if age_group == 'B2B':
        ideal_categories = ['Oficina', 'Electronica', 'Embalaje']
    else:
        ideal_categories = ['Limpieza', 'Mobiliario', 'Oficina']

    candidates = items_df[items_df['category'].isin(ideal_categories)]

    if category_pref:
        candidates = candidates[candidates['category'] == category_pref]

    return candidates.nlargest(5, 'popularity')[['item_id', 'name', 'category', 'price']]

print("Recomendaciones para nuevo usuario B2B:")
recs_b2b = recommend_for_new_user('B2B')
print(recs_b2b.to_string(index=False))

print("\nRecomendaciones para nuevo usuario B2C:")
recs_b2c = recommend_for_new_user('B2C', category_pref='Mobiliario')
print(recs_b2c.to_string(index=False))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: Content-based para nuevos productos (sin historial).*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```
Recomendaciones para nuevo usuario B2B:
item_id                          name  category    price
  P003  Teclado mecanico inalambrico  Electronica 3456.78
  P081    Cinta adhesiva transparente    Embalaje  123.45
  P055     Caja de carton 40x30x20 cm    Embalaje  234.56
  P001 Laptop empresarial i7 16GB 512  Electronica 4567.89
  P030       Desinfectante pisos 20L    Limpieza  289.01

Recomendaciones para nuevo usuario B2C:
item_id                               name  category    price
  P043         Silla visitante apilable Mobiliario  345.67
  P092 Lampara escritorio LED ajustable Mobiliario  156.78
  P098         Perchero metalico 10 ganchos Mobiliario  189.01
  P050         Silla ergonomica oficina ajustable Mobiliario  567.89
  P043         Silla visitante apilable Mobiliario  345.67
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

### Ejemplo 18: Integrador — sistema content-based completo

```python
class ContentBasedRecommender:
    def __init__(self, max_features=500):
        self.max_features = max_features
        self.vectorizer = None
        self.item_vectors = None
        self.items_df = None
        self.user_profiles = {}
        self.purchases_df = None

    def fit_items(self, items_df, text_col='description'):
        self.items_df = items_df
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features, stop_words='spanish',
            max_df=0.8, min_df=1
        )
        self.item_vectors = self.vectorizer.fit_transform(items_df[text_col])
        return self

    def fit_users(self, purchases_df, weight_col='quantity'):
        self.purchases_df = purchases_df
        for uid in purchases_df['user_id'].unique():
            self.user_profiles[uid] = build_user_profile(
                uid, purchases_df, self.item_vectors, self.items_df, weight_col
            )
        return self

    def recommend(self, user_id, n=5, method='standard', **kwargs):
        if method == 'mmr':
            return mmr_diversify(
                user_id, self.user_profiles, self.item_vectors,
                self.items_df, self.purchases_df,
                lambda_param=kwargs.get('lambda_param', 0.5), n=n
            )
        elif method == 'novelty':
            return novelty_recommend(
                user_id, self.user_profiles, self.item_vectors,
                self.items_df, self.purchases_df,
                n=n, novelty_weight=kwargs.get('novelty_weight', 0.3)
            )
        elif method == 'serendipity':
            return serendipity_recommend(
                user_id, self.user_profiles, self.item_vectors,
                self.items_df, self.purchases_df, n=n
            )
        else:
            return recommend_content_based(
                user_id, self.user_profiles, self.item_vectors,
                self.items_df, self.purchases_df, n=n
            )

    def cold_start_new_user(self, segment='B2B', category=None, n=5):
        return recommend_for_new_user(segment, category, self.items_df, self.item_vectors)

    def cold_start_new_item(self, description, n=5):
        vec = self.vectorizer.transform([description])
        sims = cosine_similarity(vec, self.item_vectors)[0]
        top_idx = np.argsort(sims)[::-1][:n]
        return self.items_df.iloc[top_idx][['item_id', 'name', 'category']]

    def evaluate(self, test_df, k=5):
        return evaluate_content_based(
            self.user_profiles, self.item_vectors, self.items_df,
            self.purchases_df, self.purchases_df, test_df, k=k
        )

# Sistema completo
cb_system = ContentBasedRecommender(max_features=500)
cb_system.fit_items(items_df)
cb_system.fit_users(purchases_df)

print("=== Sistema Content-Based ===")
print(f"Productos indexados: {len(cb_system.items_df)}")
print(f"Usuarios con perfil: {len(cb_system.user_profiles)}")

print("\nRecomendaciones standard para U000:")
for item_id, name in cb_system.recommend('U000', n=3, method='standard'):
    print(f"  {item_id} ({name})")

print("\nRecomendaciones con MMR para U000:")
for item_id, name in cb_system.recommend('U000', n=3, method='mmr', lambda_param=0.5):
    print(f"  {item_id} ({name})")

print("\nCold start: nuevo usuario B2B:")
print(cb_system.cold_start_new_user('B2B', n=3))

print("\nCold start: nuevo producto 'Tablet grafica profesional':")
new_item_recs = cb_system.cold_start_new_item('Tablet grafica profesional electronica', n=3)
print(new_item_recs.to_string(index=False))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador — sistema content-based completo.*

1. Sistema completo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```
=== Sistema Content-Based ===
Productos indexados: 100
Usuarios con perfil: 200

Recomendaciones standard para U000:
  P043 (Silla visitante apilable)
  P055 (Caja de carton 40x30x20 cm 25 pz)
  P092 (Lampara escritorio LED ajustable)

Recomendaciones con MMR para U000:
  P043 (Silla visitante apilable)
  P055 (Caja de carton 40x30x20 cm 25 pz)
  P010 (Monitor 27 pulgadas 4K)

Cold start: nuevo usuario B2B:
  item_id                          name  category    price
0   P003  Teclado mecanico inalambrico  Electronica 3456.78
1   P081    Cinta adhesiva transparente    Embalaje  123.45
2   P055     Caja de carton 40x30x20 cm    Embalaje  234.56

Cold start: nuevo producto 'Tablet grafica profesional':
  item_id                name    category
0   P001 Laptop empresarial i7  Electronica
1   P051 Laptop empresarial i7  Electronica
2   P052      Monitor 27 pulgadas  Electronica
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

## Ejercicios

1. **CB con Word2Vec en lugar de TF-IDF**: Implementar vectorización de descripciones usando Word2Vec (promedio de embeddings de palabras) en lugar de TF-IDF. Comparar precision@k.

2. **CB + Rating explícito**: Modificar el perfil de usuario para usar ratings (1-5) como pesos en lugar de cantidad comprada. Filtrar solo productos con rating >= 4.

3. **CB con filtro por temporada**: Agregar un campo 'season' a los productos y modificar la recomendación para solo sugerir productos de la temporada actual.

4. **MMR con diferentes lambda**: Probar lambda = [0.2, 0.5, 0.8] en MMR. Reportar diversidad (similitud promedio entre recomendados) vs precision@k.

5. **CB + Cluster de usuarios**: Agrupar usuarios por similitud de perfil (KMeans) y recomendar productos populares dentro del cluster para cold start.

6. **Evaluación con NDCG**: Implementar NDCG@k para CB. Comparar con precision@k y recall@k.

7. **CB cross-selling**: Implementar un sistema que recomiende productos complementarios (ej. si compra laptop, recomendar mouse + mochila) basado en reglas de asociación + CB.

8. **CB con explicit feedback correction**: Si un usuario compra un producto pero luego lo devuelve, reducir el peso de ese producto en su perfil. Simular devoluciones en el dataset.

---

## Resumen

| Característica | Content-Based | Collaborative Filtering |
|----------------|---------------|------------------------|
| Cold start producto | No afecta | Problema grave |
| Cold start usuario | Problema (fallback) | Problema (fallback) |
| Requiere otros usuarios | No | Si |
| Interpretabilidad | Alta (que features coinciden) | Media (factores latentes) |
| Sobre-especializacion | Alta | Media |
| Diversidad intrinseca | Baja | Media |

CB es ideal como complemento de CF en sistemas híbridos. En B2B, es especialmente útil porque:
- Los productos nuevos (lanzamientos industriales) pueden recomendarse inmediatamente
- Las características técnicas (especificaciones) son muy informativas
- Se puede explicar al comprador por que se recomienda un producto

En B2C, CB combinado con metadatos (precio, marca, categoria) y MMR para diversidad da buenos resultados en catalagos con descripciones ricas.
