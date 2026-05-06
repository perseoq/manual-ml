# E16 — Sistemas Híbridos de Recomendación y Evaluación

## 1. Introducción a Sistemas Híbridos

Los sistemas híbridos combinan múltiples técnicas de recomendación (CF, content-based, DL) para superar las limitaciones individuales: cold start, sparse data, y falta de diversidad.

## 2. Estrategias de Combinación

### 2.1 Weighted Hybrid
Combina predicciones de varios modelos usando pesos aprendidos o fijos.

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Scores de 3 modelos: CF, Content, DL
cf_scores = np.array([4.2, 3.8, 2.1, 5.0])
cb_scores = np.array([3.9, 4.0, 2.5, 4.5])
dl_scores = np.array([4.5, 3.5, 2.3, 4.8])

# Pesos fijos manuales
pesos = [0.4, 0.3, 0.3]
final_weighted = (pesos[0] * cf_scores + pesos[1] * cb_scores + pesos[2] * dl_scores)
print("Weighted hybrid:", final_weighted)

# Pesos aprendidos con regresión lineal
X = np.column_stack([cf_scores, cb_scores, dl_scores])
y = np.array([4.3, 3.9, 2.4, 4.9])  # ratings reales
model = LinearRegression().fit(X, y)
print("Pesos aprendidos:", model.coef_)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*2.1 Weighted Hybrid.*

1. Scores de 3 modelos: CF, Content, DL
2. Pesos fijos manuales
3. Pesos aprendidos con regresión lineal

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### 2.2 Switching Hybrid
Selecciona el modelo según la situación (ej. cold-start → content, warm → CF).

```python
def switching_recommend(user_id, item_id, n_interactions):
    if n_interactions < 5:
        return content_based_score(user_id, item_id)
    elif n_interactions < 20:
        return weighted_hybrid_score(user_id, item_id, w_cf=0.3, w_cb=0.7)
    else:
        return cf_score(user_id, item_id)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*2.2 Switching Hybrid.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### 2.3 Cascade Hybrid
Aplica modelos en secuencia, refinando resultados.

```python
def cascade_hybrid(user_id, top_n=50):
    candidates = content_based_candidates(user_id, n=200)
    candidates = cf_filter(user_id, candidates, n=100)
    candidates = dl_rank(user_id, candidates, n=top_n)
    return candidates
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*2.3 Cascade Hybrid.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### 2.4 Feature Augmentation
Usa salida de un modelo como feature de entrada para otro.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Augment: añadir embedding latente como feature
tfidf = TfidfVectorizer(max_features=5000)
item_features = tfidf.fit_transform(item_descriptions)
svd = TruncatedSVD(n_components=50)
latent_features = svd.fit_transform(item_features)

# Concatenar con features originales
augmented_features = np.hstack([original_features, latent_features])
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*2.4 Feature Augmentation.*

1. Augment: añadir embedding latente como feature
2. Concatenar con features originales

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### 2.5 Meta-Level Hybrid
Entrena un modelo con la salida (features latentes) de otro modelo más complejo.

```python
# DL extrae embeddings
user_emb = dl_model.get_layer('user_embedding').predict(user_ids)
item_emb = dl_model.get_layer('item_embedding').predict(item_ids)

# Meta-modelo más rápido sobre los embeddings
from sklearn.ensemble import GradientBoostingRegressor
meta_model = GradientBoostingRegressor()
meta_model.fit(np.hstack([user_emb, item_emb]), ratings)
```

**Salida:**

```
# El modelo ajusta sus parámetros durante el entrenamiento.
# Los parámetros aprendidos quedan disponibles en el objeto.
```

**Explicación línea por línea:**

*2.5 Meta-Level Hybrid.*

1. DL extrae embeddings
2. Meta-modelo más rápido sobre los embeddings

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 3. Ensemble CF + DL + Content

```python
import tensorflow as tf
from tensorflow import keras

# Modelo ensemble híbrido
user_input = keras.Input(shape=(1,), name='user')
item_input = keras.Input(shape=(1,), name='item')
content_input = keras.Input(shape=(100,), name='content')

user_emb = keras.layers.Embedding(n_users, 50)(user_input)
item_emb = keras.layers.Embedding(n_items, 50)(item_input)
user_vec = keras.layers.Flatten()(user_emb)
item_vec = keras.layers.Flatten()(item_emb)
content_dense = keras.layers.Dense(32, activation='relu')(content_input)

concat = keras.layers.Concatenate()([user_vec, item_vec, content_dense])
output = keras.layers.Dense(1, activation='sigmoid')(concat)

model = keras.Model(inputs=[user_input, item_input, content_input], outputs=output)
model.compile(optimizer='adam', loss='mse')
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*3. Ensemble CF + DL + Content.*

1. Modelo ensemble híbrido

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 4. Cold Start Strategy

### 4.1 User Cold Start
```python
def cold_start_user(user_profile):
    if not user_profile:
        return popular_items(n=10)
    elif len(user_profile) < 3:
        cb = content_based(user_profile['demographics'])
        pop = popular_items(n=5)
        return interleave(cb, pop)
    else:
        return hybrid_recommend(user_profile['id'])
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*4.1 User Cold Start.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### 4.2 Item Cold Start
```python
def cold_start_item(item_metadata):
    similar_items = content_similarity(item_metadata, k=10)
    for item in similar_items:
        pred_rating = cf_model.predict(user_id, item)
        if pred_rating > threshold:
            return True
    return False
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*4.2 Item Cold Start.*

1. `pred_rating = cf_model.predict(user_id, item)` — Genera predicciones sobre nuevos datos.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 5. Evaluación Online (A/B Testing)

```python
import scipy.stats as stats

def ab_test(control_ratings, treatment_ratings):
    t_stat, p_value = stats.ttest_ind(control_ratings, treatment_ratings)
    lift = (np.mean(treatment_ratings) - np.mean(control_ratings)) / np.mean(control_ratings)
    return {
        'p_value': p_value,
        'lift': lift,
        'significant': p_value < 0.05
    }

# Simulación
np.random.seed(42)
control = np.random.normal(3.5, 0.8, 1000)
treatment = np.random.normal(3.7, 0.8, 1000)
print(ab_test(control, treatment))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*5. Evaluación Online (A/B Testing).*

1. Simulación

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### 5.1 Métricas de Negocio en A/B Test
```python
def business_metrics(users):
    return {
        'ctr': np.mean(users['clicked']),
        'conversion': np.mean(users['purchased']),
        'avg_order_value': np.mean(users['spent']),
        'retention': np.mean(users['returned_next_week']),
        'dwell_time': np.mean(users['seconds_viewed'])
    }
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*5.1 Métricas de Negocio en A/B Test.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 6. Offline Métricas

### 6.1 Hit Rate (@k)
```python
def hit_rate(recommendations, held_out, k=10):
    hits = 0
    for user in held_out:
        if held_out[user] in recommendations[user][:k]:
            hits += 1
    return hits / len(held_out)

# Ejemplo
recs = {'u1': [11, 22, 33, 44], 'u2': [55, 66, 77]}
held = {'u1': 33, 'u2': 99}
print("Hit Rate@4:", hit_rate(recs, held, k=4))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*6.1 Hit Rate (@k).*

1. Ejemplo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### 6.2 Coverage
```python
def catalog_coverage(recommendations, total_items):
    recommended_set = set()
    for user_recs in recommendations.values():
        recommended_set.update(user_recs[:10])
    return len(recommended_set) / total_items

def user_coverage(users_with_recs, total_users):
    return len(users_with_recs) / total_users
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*6.2 Coverage.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### 6.3 Diversity (Pairwise Distance)
```python
from itertools import combinations

def intra_list_diversity(recommendations, item_similarity_matrix):
    diversities = []
    for user_recs in recommendations.values():
        if len(user_recs) < 2:
            continue
        dists = []
        for i, j in combinations(user_recs[:10], 2):
            dists.append(1 - item_similarity_matrix[i][j])
        diversities.append(np.mean(dists) if dists else 0)
    return np.mean(diversities)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*6.3 Diversity (Pairwise Distance).*

1. `from itertools import combinations` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### 6.4 Novelty
```python
def average_novelty(recommendations, item_popularity):
    novelties = []
    for user_recs in recommendations.values():
        pops = [item_popularity[item] for item in user_recs[:10]]
        novelties.append(-np.log2(np.mean(pops) + 1e-10))
    return np.mean(novelties)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*6.4 Novelty.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### 6.5 Serendipity
```python
def serendipity(recommendations, expected_recs, actual_ratings):
    scores = []
    for user in recommendations:
        for item in recommendations[user][:10]:
            is_unexpected = item not in expected_recs[user]
            is_positive = actual_ratings.get((user, item), 0) > 3.5
            scores.append(is_unexpected and is_positive)
    return np.mean(scores)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*6.5 Serendipity.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### 6.6 Long-tail Coverage
```python
def long_tail_coverage(recommendations, item_popularity, threshold_percentile=80):
    cutoff = np.percentile(list(item_popularity.values()), threshold_percentile)
    long_tail_items = {i for i, p in item_popularity.items() if p < cutoff}
    recommended = set()
    for user_recs in recommendations.values():
        recommended.update(user_recs[:10])
    long_tail_recs = recommended & long_tail_items
    return len(long_tail_recs) / len(long_tail_items) if long_tail_items else 0
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*6.6 Long-tail Coverage.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### 6.7 Métricas Agregadas
```python
def full_evaluation(recommendations, held_out, item_sim, item_pop):
    return {
        'hit_rate_10': hit_rate(recommendations, held_out, k=10),
        'coverage': catalog_coverage(recommendations, len(item_pop)),
        'diversity': intra_list_diversity(recommendations, item_sim),
        'novelty': average_novelty(recommendations, item_pop),
        'long_tail': long_tail_coverage(recommendations, item_pop),
        'serendipity': serendipity(recommendations, held_out, item_pop)
    }
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*6.7 Métricas Agregadas.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 7. Fairness y Bias Amplification

### 7.1 Bias Amplification Detection
```python
def bias_amplification(training_ratio, recommendation_ratio):
    return (recommendation_ratio - training_ratio) / training_ratio

# Ejemplo: categorías populares
train_popular_ratio = 0.3
rec_popular_ratio = 0.6
amp = bias_amplification(train_popular_ratio, rec_popular_ratio)
print(f"Bias amplification: {amp:.2%}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*7.1 Bias Amplification Detection.*

1. Ejemplo: categorías populares

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### 7.2 Fairness Metrics
```python
def demographic_parity(recommendations, user_groups):
    group_exposure = {}
    for group, users in user_groups.items():
        total_recs = sum(len(recommendations[u][:10]) for u in users if u in recommendations)
        group_exposure[group] = total_recs / len(users) if users else 0
    return group_exposure

def equality_of_opportunity(recommendations, relevant_items, user_groups):
    tpr_by_group = {}
    for group, users in user_groups.items():
        tp = sum(1 for u in users if u in recommendations and
                 any(item in relevant_items[u] for item in recommendations[u][:10]))
        total_rel = sum(1 for u in users if u in relevant_items and len(relevant_items[u]) > 0)
        tpr_by_group[group] = tp / total_rel if total_rel else 0
    return tpr_by_group
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*7.2 Fairness Metrics.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### 7.3 Debiasing Techniques

#### 7.3.1 Inverse Propensity Scoring (IPS)
```python
def ips_estimator(ratings, propensities):
    return np.mean(ratings / propensities)

# Propensity = probabilidad de ser observado
observed_ratings = np.array([4, 5, 3, 2])
propensity = np.array([0.8, 0.6, 0.9, 0.3])
debias_rating = ips_estimator(observed_ratings, propensity)
print(f"Debiased rating: {debias_rating:.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*7.3.1 Inverse Propensity Scoring (IPS).*

1. Propensity = probabilidad de ser observado

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



#### 7.3.2 Causal Embedding
```python
# Añadir término de debiasing en loss
def debiased_loss(y_true, y_pred, item_popularity):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    pop_penalty = tf.reduce_mean(item_popularity * y_pred)
    return mse + 0.01 * pop_penalty
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*7.3.2 Causal Embedding.*

1. Añadir término de debiasing en loss

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 8. Recomendación en Cola Larga (Long-tail)

### 8.1 Estrategias para Promover Cola Larga
```python
def long_tail_reranker(recommendations, item_popularity, alpha=0.5):
    reranked = {}
    for user, items in recommendations.items():
        scores = []
        for rank, item in enumerate(items):
            pop_score = 1 - item_popularity.get(item, 0)
            relevance = 1 / (rank + 1)
            scores.append((item, relevance * (pop_score ** alpha)))
        reranked[user] = [item for item, _ in sorted(scores, key=lambda x: -x[1])]
    return reranked

def xcl_penalty(recommendations, item_pop, gamma=0.1):
    for user, items in recommendations.items():
        rec_items = items[:10]
        penalty = gamma * sum(np.log(item_pop[i] + 1) for i in rec_items)
        print(f"User {user} long-tail penalty: {penalty:.3f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*8.1 Estrategias para Promover Cola Larga.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### 8.2 XCL (eXtreme Cold-start Long-tail)
```python
def xcl_strategy(user_meta, item_meta):
    item_embed = item_meta_features(item_meta)
    user_embed = user_meta_features(user_meta)
    score = cosine_similarity(user_embed.reshape(1, -1), item_embed)
    return score.flatten() + 0.05 * np.random.random(score.shape[1])
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*8.2 XCL (eXtreme Cold-start Long-tail).*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 9. Ejemplos Completos

### Ejemplo 1: Weighted Hybrid con Optimización de Pesos
```python
from sklearn.model_selection import GridSearchCV

def optimize_weights(cf_scores, cb_scores, dl_scores, y_true):
    X = np.column_stack([cf_scores, cb_scores, dl_scores])
    param_grid = {'positive': [True, False]}
    base = LinearRegression()
    grid = GridSearchCV(base, param_grid, cv=3)
    grid.fit(X, y_true)
    return grid.best_estimator_.coef_

# Uso
cf = np.random.rand(1000)
cb = np.random.rand(1000)
dl = np.random.rand(1000)
y = 0.4*cf + 0.35*cb + 0.25*dl + 0.1*np.random.randn(1000)
w = optimize_weights(cf, cb, dl, y)
print(f"Optimized weights: CF={w[0]:.3f}, CB={w[1]:.3f}, DL={w[2]:.3f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: Weighted Hybrid con Optimización de Pesos.*

1. Uso

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 2: Switching Hybrid con Reglas
```python
def switching_rules(user):
    n = user['n_interactions']
    age_days = user['days_since_signup']
    if n == 0:
        return 'popular'
    elif n < 5:
        return 'content'
    elif n < 20 or age_days < 7:
        return 'weighted'
    elif n > 100:
        return 'cf'
    else:
        return 'ensemble'
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Ejemplo 2: Switching Hybrid con Reglas.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 3: Cascade Hybrid Multi-etapa
```python
def cascade_multistage(user_id, items, stages):
    candidates = items.copy()
    for stage_name, stage_fn, top_k in stages:
        candidates = stage_fn(user_id, candidates, top_k)
        print(f"Stage {stage_name}: {len(candidates)} candidates")
    return candidates

stages = [
    ('popular_filter', lambda u, i, k: sorted(i, key=lambda x: -x.popularity)[:k], 500),
    ('content_match', lambda u, i, k: sorted(i, key=lambda x: content_sim(u, x))[:k], 100),
    ('cf_rank', lambda u, i, k: sorted(i, key=lambda x: cf_predict(u, x.id))[:k], 10)
]
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: Cascade Hybrid Multi-etapa.*

1. `print(f"Stage {stage_name}: {len(candidates)} candidates")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 4: Feature Augmentation con Deep Features
```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image

cnn = ResNet50(weights='imagenet', include_top=False, pooling='avg')
img = image.load_img('product.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
deep_features = cnn.predict(img_array)

item_orig_features = np.random.rand(50)
item_augmented = np.hstack([item_orig_features, deep_features.flatten()])
print(f"Augmented feature dim: {item_augmented.shape[0]}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: Feature Augmentation con Deep Features.*

1. `from tensorflow.keras.applications import ResNet50` — Importa las librerías necesarias para el análisis.
2. `from tensorflow.keras.preprocessing import image` — Importa las librerías necesarias para el análisis.
3. `deep_features = cnn.predict(img_array)` — Genera predicciones sobre nuevos datos.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 5: Métricas de Evaluación Completas
```python
def comprehensive_eval(recs, held_out, item_sim, item_pop, user_groups):
    metrics = {}
    metrics['hit_rate@5'] = hit_rate(recs, held_out, k=5)
    metrics['hit_rate@10'] = hit_rate(recs, held_out, k=10)
    metrics['hit_rate@20'] = hit_rate(recs, held_out, k=20)
    metrics['coverage'] = catalog_coverage(recs, len(item_pop))
    metrics['diversity'] = intra_list_diversity(recs, item_sim)
    metrics['novelty'] = average_novelty(recs, item_pop)
    metrics['long_tail'] = long_tail_coverage(recs, item_pop)
    metrics['serendipity'] = serendipity(recs, {u: [] for u in recs}, {})
    metrics['fairness'] = demographic_parity(recs, user_groups)
    return metrics
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Ejemplo 5: Métricas de Evaluación Completas.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 6: A/B Test con Múltiples Variantes
```python
def ab_test_multivariate(variants_data, alpha=0.05):
    from scipy.stats import f_oneway
    groups = [data for name, data in variants_data]
    f_stat, p_value = f_oneway(*groups)
    return {
        'n_variants': len(variants_data),
        'f_stat': f_stat,
        'p_value': p_value,
        'significant': p_value < alpha,
        'best_variant': max(variants_data, key=lambda x: np.mean(x[1]))[0]
    }

variants = [
    ('control', np.random.normal(3.5, 0.8, 1000)),
    ('weighted', np.random.normal(3.7, 0.9, 1000)),
    ('cascade', np.random.normal(3.6, 0.85, 1000)),
    ('ensemble', np.random.normal(3.8, 0.8, 1000))
]
print(ab_test_multivariate(variants))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: A/B Test con Múltiples Variantes.*

1. `from scipy.stats import f_oneway` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 7: Cold Start con Meta-learning
```python
def meta_learning_cold_start(support_set, query_item):
    # Support set: pocos items del nuevo usuario
    support_emb = embed_items(support_set)
    query_emb = embed_items([query_item])
    similarities = cosine_similarity(query_emb, support_emb).flatten()
    weights = softmax(similarities / 0.1)
    return np.dot(weights, np.array([r for _, r in support_set]))
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Ejemplo 7: Cold Start con Meta-learning.*

1. Support set: pocos items del nuevo usuario

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 8: Reranking por Diversidad
```python
def mmr_rerank(scores, similarities, lambda_param=0.5, k=10):
    selected = []
    candidates = list(range(len(scores)))
    for _ in range(k):
        mmr_scores = []
        for i in candidates:
            sim_to_selected = max([similarities[i][j] for j in selected]) if selected else 0
            mmr = lambda_param * scores[i] - (1 - lambda_param) * sim_to_selected
            mmr_scores.append(mmr)
        best = candidates[np.argmax(mmr_scores)]
        selected.append(best)
        candidates.remove(best)
    return selected
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Ejemplo 8: Reranking por Diversidad.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 9: Evaluación de Serendipia vs Precisión
```python
def precision_vs_serendipity_tradeoff(recs_base, recs_diverse, held_out):
    p_base = hit_rate(recs_base, held_out, k=10)
    p_diverse = hit_rate(recs_diverse, held_out, k=10)
    s_base = serendipity(recs_base, held_out, {})
    s_diverse = serendipity(recs_diverse, held_out, {})
    return {
        'precision_base': p_base, 'serendipity_base': s_base,
        'precision_diverse': p_diverse, 'serendipity_diverse': s_diverse
    }
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Ejemplo 9: Evaluación de Serendipia vs Precisión.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 10: Debiasing con Sample Weight
```python
def sample_weight_debias(interactions, item_pop):
    weights = []
    for (user, item, rating) in interactions:
        weight = 1.0 / (item_pop.get(item, 1.0) ** 0.5)
        weights.append(weight)
    return np.array(weights)

# Usar en entrenamiento
from sklearn.linear_model import Ridge
model = Ridge()
model.fit(X_train, y_train, sample_weight=weights_train)
```

**Salida:**

```
# El modelo ajusta sus parámetros durante el entrenamiento.
# Los parámetros aprendidos quedan disponibles en el objeto.
```

**Explicación línea por línea:**

*Ejemplo 10: Debiasing con Sample Weight.*

1. Usar en entrenamiento

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 11: Evaluación de Fairness por Grupo
```python
def fairness_report(recs, user_attributes):
    groups = set(user_attributes.values())
    report = {}
    for g in groups:
        users_g = [u for u, a in user_attributes.items() if a == g]
        avg_recs = np.mean([len(recs[u][:10]) for u in users_g if u in recs])
        avg_rating = np.mean([r for u in users_g for r in ratings_for_user(u)])
        report[g] = {'avg_recs': avg_recs, 'avg_rating': avg_rating}
    return report
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Ejemplo 11: Evaluación de Fairness por Grupo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 12: Online Metrics Dashboard
```python
def online_dashboard(stream_data):
    metrics = {'ctr': [], 'conversion': [], 'revenue_per_user': []}
    for batch in stream_data:
        metrics['ctr'].append(np.mean(batch['clicked']))
        metrics['conversion'].append(np.mean(batch['purchased']))
        metrics['revenue_per_user'].append(np.mean(batch['revenue']))
    return {k: {'mean': np.mean(v), 'std': np.std(v), 'trend': v[-5:]}
            for k, v in metrics.items()}
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Ejemplo 12: Online Metrics Dashboard.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 13: Comparación de Estrategias de Recomendación
```python
strategies = {
    'popular': lambda u: popular_items(10),
    'cf': lambda u: cf_recommend(u, 10),
    'content': lambda u: content_recommend(u, 10),
    'weighted': lambda u: weighted_hybrid(u, w=[0.3, 0.3, 0.4]),
    'cascade': lambda u: cascade_hybrid(u, [200, 50, 10]),
    'ensemble_dl': lambda u: ensemble_dl_recommend(u, 10)
}

def compare_strategies(strategies, test_users, held_out, item_sim, item_pop):
    results = {}
    for name, strategy in strategies.items():
        recs = {u: strategy(u) for u in test_users}
        results[name] = {
            'hit_rate': hit_rate(recs, held_out, k=10),
            'diversity': intra_list_diversity(recs, item_sim),
            'coverage': catalog_coverage(recs, len(item_pop))
        }
    return pd.DataFrame(results).T
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Ejemplo 13: Comparación de Estrategias de Recomendación.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 14: Optimización de Pesos con Búsqueda Bayesiana
```python
from skopt import gp_minimize

def objective(weights):
    w1, w2, w3 = weights
    combo = w1 * cf_scores + w2 * cb_scores + w3 * dl_scores
    return -np.mean([hit_rate_at_k(combo, held_out, k=10)])

result = gp_minimize(objective, [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], n_calls=50)
print(f"Best weights: {result.x} with hit rate {-result.fun:.3f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: Optimización de Pesos con Búsqueda Bayesiana.*

1. `from skopt import gp_minimize` — Importa las librerías necesarias para el análisis.
2. `print(f"Best weights: {result.x} with hit rate {-result.fun:.3f}")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 15: Recomendación con Restricciones de Negocio
```python
def business_constrained_recs(user_id, constraints):
    candidates = hybrid_recommend(user_id, n=100)
    if constraints.get('min_margin'):
        candidates = [c for c in candidates if c.margin > constraints['min_margin']]
    if constraints.get('exclude_categories'):
        candidates = [c for c in candidates if c.category not in constraints['exclude_categories']]
    if constraints.get('inventory_only'):
        candidates = [c for c in candidates if c.in_stock]
    return candidates[:10]
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Ejemplo 15: Recomendación con Restricciones de Negocio.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 16: Evaluación Temporal (Time-aware)
```python
def time_based_evaluation(interactions, recs_fn, train_cutoff, test_cutoff):
    train = interactions[interactions['timestamp'] < train_cutoff]
    test = interactions[interactions['timestamp'] >= test_cutoff]
    model = train_model(train)
    recs = {u: recs_fn(model, u) for u in test['user'].unique()}
    return hit_rate(recs, test, k=10)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Ejemplo 16: Evaluación Temporal (Time-aware).*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 17: Ensemble de Modelos Heterogéneos
```python
from sklearn.ensemble import VotingRegressor

cf_model = Ridge(alpha=1.0)
cb_model = RandomForestRegressor(n_estimators=100)
dl_model = keras.Sequential([...])

ensemble = VotingRegressor([
    ('cf', cf_model),
    ('cb', cb_model),
    ('dl', dl_model)
])
ensemble.fit(X_train, y_train)
```

**Salida:**

```
# El modelo ajusta sus parámetros durante el entrenamiento.
# Los parámetros aprendidos quedan disponibles en el objeto.
```

**Explicación línea por línea:**

*Ejemplo 17: Ensemble de Modelos Heterogéneos.*

1. `from sklearn.ensemble import VotingRegressor` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 18: Monitoreo de Métricas en Producción
```python
def monitor_metrics(production_recs, production_feedback, baseline_metrics):
    current = {
        'hit_rate': hit_rate(production_recs, production_feedback, k=10),
        'ctr': np.mean(production_feedback['clicked']),
        'diversity': intra_list_diversity(production_recs, item_sim_matrix)
    }
    alerts = []
    for metric, value in current.items():
        baseline = baseline_metrics.get(metric, 0)
        if abs(value - baseline) / max(baseline, 0.001) > 0.1:
            alerts.append(f"ALERT: {metric} changed by {(value-baseline)/baseline:.1%}")
    return current, alerts
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Ejemplo 18: Monitoreo de Métricas en Producción.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 10. Ejercicios

### Ejercicio 1
Implementa un weighted hybrid que combine CF, content-based y DL usando pesos [0.2, 0.5, 0.3]. Genera scores aleatorios para 10 usuarios y 20 items, calcula el score final.

### Ejercicio 2
Diseña un switching hybrid con 4 niveles: cold-start (0 interacciones), warm-up (1-5), growth (6-20), mature (21+). Define qué modelo usar en cada nivel.

### Ejercicio 3
Implementa cascade hybrid en 3 etapas: (1) filtrar top 1000 por popularidad, (2) top 100 por content similarity, (3) top 10 por CF score. Simula con datos sintéticos.

### Ejercicio 4
Calcula hit rate@10, coverage, y diversity para dos estrategias de recomendación. Genera datos sintéticos para 50 usuarios y 100 items.

### Ejercicio 5
Crea una función que detecte bias amplification en recomendaciones comparando la proporción de una categoría en entrenamiento vs recomendaciones.

### Ejercicio 6
Implementa un reranker MMR que balancee relevancia y diversidad. Prueba con lambda=0.3, 0.5, y 0.7.

### Ejercicio 7
Simula un A/B test con 2 variantes (control y tratamiento). Calcula p-value usando t-test y determina si hay diferencia significativa.

### Ejercicio 8
Implementa una estrategia de debiasing IPS y compárala con un estimador naive (simple average) sobre datos con propensity sesgada.

---

## Referencias

- Burke, R. (2002). Hybrid Recommender Systems: Survey and Experiments.
- McNee, S. M., Riedl, J., & Konstan, J. A. (2006). Being accurate is not enough.
- Castells, P. et al. (2015). Novelty and Diversity in Recommender Systems.
- Abdollahpouri, H. et al. (2020). Fairness in Recommender Systems.
- Koren, Y. (2008). Factorization Meets the Neighborhood.
