# I23: Clustering Avanzado con scikit-learn

## Aplicación a Ventas, Compras e Inventarios

El clustering avanzado abarca técnicas que superan las limitaciones de K-Means: **detectar formas arbitrarias, manejar ruido, determinar el número de clusters automáticamente y proporcionar jerarquías**. En el contexto comercial, permiten **descubrir segmentos naturales de productos, identificar anomalías en inventarios y agrupar clientes por patrones complejos de compra**.

### Fundamentos Teóricos

**DBSCAN (Density-Based Spatial Clustering):** agrupa puntos densamente conectados, marcando como ruido los puntos aislados.
- `eps`: radio de vecindad (distancia máxima para considerar vecinos)
- `min_samples`: mínimo de puntos en un vecindario para ser núcleo
- `metric`: métrica de distancia (`euclidean`, `manhattan`, `cosine`)
- `algorithm`: método de búsqueda de vecinos (`auto`, `ball_tree`, `kd_tree`, `brute`)
- No requiere especificar número de clusters
- Detecta clusters de forma arbitraria y outliers

**HDBSCAN:** extensión jerárquica de DBSCAN que encuentra clusters de densidad variable. No necesita el parámetro `eps`.

**AgglomerativeClustering:** clustering jerárquico ascendente que fusiona puntos/clusters iterativamente.
- `linkage`: criterio de fusión — `ward` (min varianza), `complete` (max distancia), `average` (media), `single` (min distancia)
- `distance_threshold`: umbral para cortar el dendrograma
- Se puede visualizar con dendrograma de `scipy.cluster.hierarchy`

**SpectralClustering:** usa la descomposición espectral de la matriz de similitud (basada en grafos). Bueno para formas complejas.
- `affinity`: tipo de similitud (`rbf`, `nearest_neighbors`)
- `gamma`: parámetro del kernel RBF
- `n_neighbors`: vecinos para grafo de k-vecinos

**MeanShift:** estima modas de densidad desplazando puntos hacia regiones de alta densidad.
- `bandwidth`: tamaño de ventana de kernel (automático si no se especifica)
- `bin_seeding`: acelera usando bins
- No requiere especificar número de clusters

**OPTICS:** similar a DBSCAN pero produce un ordenamiento de puntos que revela la estructura de clustering a diferentes escalas.
- `min_samples`, `max_eps`, `xi` (umbral de pendiente para clusters jerárquicos)

**BIRCH:** clustering incremental diseñado para grandes datasets, construye un árbol de características (CF Tree).
- `threshold`: radio máximo del subcluster
- `branching_factor`: máximo de subclusters por nodo

**AffinityPropagation:** intercambia mensajes entre puntos para determinar ejemplares (centros de clusters).
- `damping`: factor de amortiguamiento (0.5-1)
- `preference`: preferencia de cada punto para ser ejemplar

**GaussianMixture:** modelo probabilístico que asume que los datos provienen de una mezcla de Gaussianas.
- `n_components`: número de componentes
- `covariance_type`: `'full'` (cada una su propia matriz), `'tied'` (compartida), `'diag'` (diagonal), `'spherical'` (esférica)
- BIC/AIC: criterios para seleccionar número de componentes

---

## Ejemplos Prácticos con sklearn

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import (DBSCAN, AgglomerativeClustering, SpectralClustering,
                              MeanShift, OPTICS, Birch, AffinityPropagation,
                              estimate_bandwidth)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs, make_moons, make_circles
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# DATASET SIMULADO DE PRODUCTOS
# ============================================================
np.random.seed(42)
n = 400

df = pd.DataFrame({
    'precio': np.random.uniform(10, 200, n),
    'margen': np.random.uniform(5, 60, n),
    'descuento': np.random.uniform(0, 0.5, n),
    'dias_stock': np.random.randint(1, 120, n),
    'ventas_semana': np.random.poisson(50, n),
    'devoluciones': np.random.poisson(3, n),
})

df['margen'] = df['margen'] + 0.2 * df['precio'] + np.random.normal(0, 3, n)
df['ventas_semana'] = (df['ventas_semana'] - 0.3 * df['precio']
                        + 2 * df['descuento'] * 100 + np.random.normal(0, 10, n)).clip(0)
df['dias_stock'] = (df['dias_stock'] - 0.1 * df['ventas_semana']
                     + np.random.normal(0, 5, n)).clip(1)

X_ventas = df[['precio', 'margen', 'descuento', 'ventas_semana']].values

# Escalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_ventas)

print('Dataset de ventas creado.')
print(f'Shape: {X_scaled.shape}')
print(f'Features: precio, margen, descuento, ventas_semana')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplos Prácticos con sklearn.*

1. ============================================================
2. DATASET SIMULADO DE PRODUCTOS
3. ============================================================
4. Escalar

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 1: DBSCAN — detectar clusters de formas arbitrarias en datos de productos

```python
# Probar con datos con formas no convexas
X_moons, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
X_circles, _ = make_circles(n_samples=300, noise=0.05, factor=0.3, random_state=42)
X_blobs, _ = make_blobs(n_samples=300, n_features=2, centers=3, random_state=42)

# DBSCAN en lunas
dbscan_moons = DBSCAN(eps=0.2, min_samples=5)
labels_moons = dbscan_moons.fit_predict(X_moons)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, X, labels, title in zip(axes,
                                 [X_moons, X_circles, X_blobs],
                                 [dbscan_moons.fit_predict(X_moons),
                                  DBSCAN(eps=0.15, min_samples=5).fit_predict(X_circles),
                                  DBSCAN(eps=0.3, min_samples=5).fit_predict(X_blobs)],
                                 ['Moons', 'Circles', 'Blobs']):
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7, edgecolors='k')
    ax.set_title(f'{title}: {n_clusters} clusters, {n_noise} outliers')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

plt.tight_layout()
plt.show()

print('DBSCAN detecta formas arbitrarias sin necesidad de especificar k.')
print('Puntos etiquetados como -1 son outliers/ruido.')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: DBSCAN — detectar clusters de formas arbitrarias en datos de productos.*

1. Probar con datos con formas no convexas
2. DBSCAN en lunas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 2: DBSCAN eps — efecto del radio (0.3, 0.5, 1.0)

```python
X = df[['precio', 'margen']].values
scaler = StandardScaler()
X_s = scaler.fit_transform(X)

eps_values = [0.2, 0.4, 0.6, 1.0]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, eps in zip(axes.flat, eps_values):
    dbscan = DBSCAN(eps=eps, min_samples=5)
    labels = dbscan.fit_predict(X_s)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    ax.scatter(X_s[:, 0], X_s[:, 1], c=labels, cmap='viridis', alpha=0.7, edgecolors='k')
    ax.set_title(f'eps={eps}: {n_clusters} clusters, {n_noise} outliers')
    ax.set_xlabel('Precio (escalado)')
    ax.set_ylabel('Margen (escalado)')

plt.tight_layout()
plt.show()

print('eps pequeño → más clusters pequeños y más outliers.')
print('eps grande → menos clusters, más puntos fusionados.')
print('El parámetro eps es crítico y requiere ajuste.')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: DBSCAN eps — efecto del radio (0.3, 0.5, 1.0).*

1. `labels = dbscan.fit_predict(X_s)` — Genera predicciones sobre nuevos datos.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 3: DBSCAN min_samples — mínimo puntos para cluster

```python
X_s = scaler.fit_transform(df[['precio', 'margen']].values)

min_samples_vals = [3, 5, 10, 20]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, ms in zip(axes.flat, min_samples_vals):
    dbscan = DBSCAN(eps=0.3, min_samples=ms)
    labels = dbscan.fit_predict(X_s)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    ax.scatter(X_s[:, 0], X_s[:, 1], c=labels, cmap='viridis', alpha=0.7, edgecolors='k')
    ax.set_title(f'min_samples={ms}: {n_clusters} clusters, {n_noise} outliers')
    ax.set_xlabel('Precio (escalado)')
    ax.set_ylabel('Margen (escalado)')

plt.tight_layout()
plt.show()

print('min_samples alto → más restrictivo, menos clusters, más ruido.')
print('Regla general: min_samples >= dimensionality + 1')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: DBSCAN min_samples — mínimo puntos para cluster.*

1. `labels = dbscan.fit_predict(X_s)` — Genera predicciones sobre nuevos datos.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 4: DBSCAN metric='cosine' — para datos normalizados

```python
# Datos normalizados (vectores unitarios)
from sklearn.preprocessing import normalize

X_norm = normalize(X_scaled, norm='l2')

# DBSCAN con cosine
dbscan_cos = DBSCAN(eps=0.3, min_samples=5, metric='cosine')
labels_cos = dbscan_cos.fit_predict(X_norm)

# DBSCAN con euclidean (sobre datos normalizados)
dbscan_euc = DBSCAN(eps=0.3, min_samples=5, metric='euclidean')
labels_euc = dbscan_euc.fit_predict(X_norm)

# PCA para visualizar
pca = PCA(n_components=2)
X_viz = pca.fit_transform(X_norm)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, labels, metric in zip(axes, [labels_cos, labels_euc], ['cosine', 'euclidean']):
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    ax.scatter(X_viz[:, 0], X_viz[:, 1], c=labels, cmap='viridis', alpha=0.7, edgecolors='k')
    ax.set_title(f'metric={metric}: {n_clusters} clusters, {n_noise} outliers')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

plt.tight_layout()
plt.show()

print('La métrica cosine es adecuada para datos normalizados.')
print('Mide similitud por ángulo, no por magnitud.')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: DBSCAN metric='cosine' — para datos normalizados.*

1. Datos normalizados (vectores unitarios)
2. DBSCAN con cosine
3. DBSCAN con euclidean (sobre datos normalizados)
4. PCA para visualizar

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 5: HDBSCAN — versión jerárquica (no necesita eps)

```python
try:
    import hdbscan
    HDBSCAN_INSTALLED = True
except ImportError:
    HDBSCAN_INSTALLED = False
    print('HDBSCAN no instalado. Instalar con: pip install hdbscan')

if HDBSCAN_INSTALLED:
    X_s = scaler.fit_transform(df[['precio', 'margen', 'descuento', 'ventas_semana']].values)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5, metric='euclidean')
    labels_hdb = clusterer.fit_predict(X_s)

    n_clusters = len(set(labels_hdb)) - (1 if -1 in labels_hdb else 0)
    n_noise = list(labels_hdb).count(-1)

    # PCA para visualizar
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_s)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_hdb, cmap='viridis',
                alpha=0.7, edgecolors='k')
    plt.title(f'HDBSCAN: {n_clusters} clusters, {n_noise} outliers')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()

    print(f'Clusters encontrados: {n_clusters}')
    print(f'Outliers: {n_noise}')
    print(f'Probabilidades de pertenencia: {clusterer.probabilities_[:5].round(3)}')
    print('HDBSCAN es más robusto que DBSCAN: no requiere eps.')
else:
    print('\nNOTA: Para ejecutar este ejemplo, instale hdbscan:')
    print('  pip install hdbscan')
    print('HDBSCAN extiende DBSCAN con jerarquía de densidad.')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: HDBSCAN — versión jerárquica (no necesita eps).*

1. PCA para visualizar

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 6: AgglomerativeClustering — clustering jerárquico aglomerativo

```python
X_s = scaler.fit_transform(df[['precio', 'margen']].values)

agglo = AgglomerativeClustering(n_clusters=4, linkage='ward')
labels_agg = agglo.fit_predict(X_s)

plt.figure(figsize=(8, 6))
plt.scatter(X_s[:, 0], X_s[:, 1], c=labels_agg, cmap='viridis', alpha=0.7, edgecolors='k')
plt.xlabel('Precio (escalado)')
plt.ylabel('Margen (escalado)')
plt.title(f'AgglomerativeClustering (ward, k=4)')
plt.show()

print('Distribución de clusters:')
print(pd.Series(labels_agg).value_counts().sort_index())
print('\nAgglomerativeClustering fusiona puntos iterativamente.')
print('No requiere especificar k si se usa distance_threshold.')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: AgglomerativeClustering — clustering jerárquico aglomerativo.*

1. `labels_agg = agglo.fit_predict(X_s)` — Genera predicciones sobre nuevos datos.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 7: AgglomerativeClustering linkage='ward' vs 'complete' vs 'average'

```python
X_s = scaler.fit_transform(df[['precio', 'margen', 'descuento']].values)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, linkage_type in zip(axes, ['ward', 'complete', 'average']):
    agglo = AgglomerativeClustering(n_clusters=4, linkage=linkage_type)
    labels = agglo.fit_predict(X_s)

    # PCA para visualizar en 2D
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_s)
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis', alpha=0.7, edgecolors='k')
    ax.set_title(f'linkage={linkage_type}')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

plt.tight_layout()
plt.show()

print('Comparación de linkages:')
for linkage_type in ['ward', 'complete', 'average']:
    agglo = AgglomerativeClustering(n_clusters=4, linkage=linkage_type)
    labels = agglo.fit_predict(X_s)
    sil = silhouette_score(X_s, labels)
    print(f'  {linkage_type}: silhouette={sil:.3f}, '
          f'n={pd.Series(labels).value_counts().to_dict()}')

print('\nward: minimiza varianza intra-cluster (similar a KMeans)')
print('complete: maximiza distancia entre clusters')
print('average: compromiso entre single y complete')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: AgglomerativeClustering linkage='ward' vs 'complete' vs 'average'.*

1. PCA para visualizar en 2D

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 8: Dendrograma — visualizar jerarquía de clusters

```python
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Usar una muestra pequeña para visualización clara
np.random.seed(42)
idx_sample = np.random.choice(len(df), 30, replace=False)
X_sample = scaler.fit_transform(df[['precio', 'margen']].values)[idx_sample]

# Calcular matriz de enlace
Z = linkage(X_sample, method='ward')

plt.figure(figsize=(12, 6))
dn = dendrogram(Z, labels=idx_sample, leaf_rotation=90, leaf_font_size=8,
                color_threshold=2.5)
plt.title('Dendrograma - Clustering Jerárquico (ward)')
plt.xlabel('Índice de producto')
plt.ylabel('Distancia')
plt.axhline(y=2.5, color='r', linestyle='--', alpha=0.5, label='Corte k=2')
plt.axhline(y=1.8, color='g', linestyle='--', alpha=0.5, label='Corte k=3')
plt.legend()
plt.show()

# Obtener clusters del dendrograma
clusters_dendro = fcluster(Z, t=2.5, criterion='distance')
print(f'Clusters con corte en distancia=2.5: {np.unique(clusters_dendro)}')
print(f'Distribución: {pd.Series(clusters_dendro).value_counts().sort_index().to_dict()}')

# Interpretación
print('\nEl dendrograma permite visualizar las fusiones jerárquicas.')
print('Cortar horizontalmente = elegir número de clusters.')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: Dendrograma — visualizar jerarquía de clusters.*

1. Usar una muestra pequeña para visualización clara
2. Calcular matriz de enlace
3. Obtener clusters del dendrograma
4. Interpretación

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 9: SpectralClustering — clustering basado en grafos

```python
# Dataset con formas anidadas (círculos concéntricos)
X_circ, y_circ = make_circles(n_samples=300, noise=0.05, factor=0.3, random_state=42)

# KMeans falla en formas anidadas
from sklearn.cluster import KMeans
kmeans_circ = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_kmeans = kmeans_circ.fit_predict(X_circ)

# Spectral clustering
spectral = SpectralClustering(n_clusters=2, affinity='rbf', gamma=1.0, random_state=42)
labels_spec = spectral.fit_predict(X_circ)

# Spectral con nearest neighbors
spectral_nn = SpectralClustering(n_clusters=2, affinity='nearest_neighbors',
                                  n_neighbors=10, random_state=42)
labels_spec_nn = spectral_nn.fit_predict(X_circ)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].scatter(X_circ[:, 0], X_circ[:, 1], c=labels_kmeans, cmap='viridis', alpha=0.7)
axes[0].set_title('KMeans (falla en formas anidadas)')
axes[1].scatter(X_circ[:, 0], X_circ[:, 1], c=labels_spec, cmap='viridis', alpha=0.7)
axes[1].set_title('Spectral (affinity=rbf)')
axes[2].scatter(X_circ[:, 0], X_circ[:, 1], c=labels_spec_nn, cmap='viridis', alpha=0.7)
axes[2].set_title('Spectral (affinity=nn)')

for ax in axes:
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

plt.tight_layout()
plt.show()

print('SpectralClustering funciona bien con formas anidadas y complejas.')
print('KMeans no puede separar círculos concéntricos.')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: SpectralClustering — clustering basado en grafos.*

1. Dataset con formas anidadas (círculos concéntricos)
2. KMeans falla en formas anidadas
3. Spectral clustering
4. Spectral con nearest neighbors

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 10: MeanShift — estimación de densidad (bandwidth automático)

```python
X_s = scaler.fit_transform(df[['precio', 'margen']].values)

# Estimar bandwidth automáticamente
bandwidth = estimate_bandwidth(X_s, quantile=0.2, n_samples=len(X_s))
print(f'Bandwidth estimado: {bandwidth:.4f}')

meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True)
labels_ms = meanshift.fit_predict(X_s)

n_clusters_ms = len(set(labels_ms))
centros_ms = meanshift.cluster_centers_

plt.figure(figsize=(8, 6))
plt.scatter(X_s[:, 0], X_s[:, 1], c=labels_ms, cmap='viridis', alpha=0.7, edgecolors='k')
plt.scatter(centros_ms[:, 0], centros_ms[:, 1], c='red', marker='X', s=200, label='Centros')
plt.xlabel('Precio (escalado)')
plt.ylabel('Margen (escalado)')
plt.title(f'MeanShift: {n_clusters_ms} clusters (bandwidth={bandwidth:.3f})')
plt.legend()
plt.show()

print(f'Clusters encontrados: {n_clusters_ms}')
print(f'Centroides:\n{centros_ms}')
print('MeanShift no requiere especificar k; lo determina por densidad.')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: MeanShift — estimación de densidad (bandwidth automático).*

1. Estimar bandwidth automáticamente

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 11: OPTICS — ordenamiento de puntos para visualizar estructura

```python
X_s = scaler.fit_transform(df[['precio', 'margen', 'descuento', 'ventas_semana']].values)

optics = OPTICS(min_samples=10, xi=0.05, min_cluster_size=0.05)
labels_optics = optics.fit_predict(X_s)

n_clusters_opt = len(set(labels_optics)) - (1 if -1 in labels_optics else 0)
n_noise_opt = list(labels_optics).count(-1)

# PCA para visualizar
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_s)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_optics, cmap='viridis', alpha=0.7, edgecolors='k')
plt.title(f'OPTICS: {n_clusters_opt} clusters, {n_noise_opt} outliers')
plt.xlabel('PC1')
plt.ylabel('PC2')

# Reachability plot
plt.subplot(1, 2, 2)
space = np.arange(len(X_s))
reachability = optics.reachability_[optics.ordering_]
plt.plot(space, reachability, 'b-', alpha=0.7)
plt.xlabel('Orden de puntos')
plt.ylabel('Distancia de alcanzabilidad')
plt.title('Reachability plot (OPTICS)')
plt.show()

print(f'Clusters encontrados: {n_clusters_opt}')
print(f'Outliers: {n_noise_opt}')
print('El reachability plot muestra la estructura de clustering:')
print('valles = clusters, picos = separaciones entre clusters.')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: OPTICS — ordenamiento de puntos para visualizar estructura.*

1. PCA para visualizar
2. Reachability plot

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 12: BIRCH — clustering incremental para grandes datos

```python
# Simular dataset grande
n_grande = 2000
X_grande = np.random.randn(n_grande, 4)

birch = Birch(threshold=0.5, branching_factor=50, n_clusters=4)
labels_birch = birch.fit_predict(X_grande)

# Submuestra para visualizar
idx_viz = np.random.choice(n_grande, 500, replace=False)
pca = PCA(n_components=2)
X_viz = pca.fit_transform(X_grande[idx_viz])

plt.figure(figsize=(8, 6))
plt.scatter(X_viz[:, 0], X_viz[:, 1], c=labels_birch[idx_viz], cmap='viridis', alpha=0.7)
plt.title(f'BIRCH (threshold={0.5}, branching={50}) — {len(set(labels_birch))} clusters')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

print('BIRCH procesa datos incrementalmente sin cargar todo en RAM.')
print(f'Subclusters formados: {birch.subcluster_centers_.shape[0]}')
print(f'Clusters finales: {len(set(labels_birch))}')
print('Ideal para streaming o datasets que no caben en memoria.')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: BIRCH — clustering incremental para grandes datos.*

1. Simular dataset grande
2. Submuestra para visualizar

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 13: AffinityPropagation — propagación de afinidad

```python
X_s = scaler.fit_transform(df[['precio', 'margen']].values)

# Usar muestra pequeña (AffinityPropagation es computacionalmente intensivo)
idx_ap = np.random.choice(len(X_s), 150, replace=False)
X_ap = X_s[idx_ap]

ap = AffinityPropagation(damping=0.9, preference=-50, random_state=42, max_iter=500)
labels_ap = ap.fit_predict(X_ap)

n_clusters_ap = len(set(labels_ap))
ejemplares = ap.cluster_centers_indices_

plt.figure(figsize=(8, 6))
plt.scatter(X_ap[:, 0], X_ap[:, 1], c=labels_ap, cmap='viridis', alpha=0.7, edgecolors='k')
plt.scatter(X_ap[ejemplares, 0], X_ap[ejemplares, 1],
            c='red', marker='X', s=200, label='Ejemplares')
plt.xlabel('Precio (escalado)')
plt.ylabel('Margen (escalado)')
plt.title(f'AffinityPropagation: {n_clusters_ap} clusters')
plt.legend()
plt.show()

print(f'Clusters encontrados: {n_clusters_ap}')
print(f'Ejemplares (centros de clusters): {len(ejemplares)}')
print(f'Preference usado: -50')
print('Cada cluster está representado por un punto real (ejemplar).')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: AffinityPropagation — propagación de afinidad.*

1. Usar muestra pequeña (AffinityPropagation es computacionalmente intensivo)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 14: GaussianMixture — clustering probabilístico

```python
X_s = scaler.fit_transform(df[['precio', 'margen', 'descuento', 'ventas_semana']].values)

gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42, max_iter=200)
gmm.fit(X_s)
labels_gmm = gmm.predict(X_s)
probs_gmm = gmm.predict_proba(X_s)

print('GaussianMixture: clustering probabilístico')
print(f'Distribución de clusters: {pd.Series(labels_gmm).value_counts().sort_index().to_dict()}')
print(f'\nProbabilidades de pertenencia (primeros 5 puntos):')
print(pd.DataFrame(probs_gmm[:5], columns=[f'Cluster {i}' for i in range(4)]).round(3))

# Punto con máxima incertidumbre
incertidumbre = 1 - probs_gmm.max(axis=1)
idx_incierto = np.argmax(incertidumbre)
print(f'\nPunto más incierto (idx={idx_incierto}): '
      f'probabilidad max={probs_gmm[idx_incierto].max():.3f}')
print('GMM asigna probabilidades, no etiquetas duras.')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: GaussianMixture — clustering probabilístico.*

1. Punto con máxima incertidumbre

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 15: GaussianMixture covariance_type — 'full' vs 'tied' vs 'diag' vs 'spherical'

```python
X_s = scaler.fit_transform(df[['precio', 'margen']].values)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
cov_types = ['full', 'tied', 'diag', 'spherical']

for ax, cov_type in zip(axes.flat, cov_types):
    gmm = GaussianMixture(n_components=3, covariance_type=cov_type, random_state=42, max_iter=200)
    labels = gmm.fit_predict(X_s)

    ax.scatter(X_s[:, 0], X_s[:, 1], c=labels, cmap='viridis', alpha=0.7, edgecolors='k')
    ax.set_title(f'covariance_type={cov_type}')
    ax.set_xlabel('Precio (escalado)')
    ax.set_ylabel('Margen (escalado)')

plt.tight_layout()
plt.show()

print('Comparación de tipos de covarianza:')
for cov_type in cov_types:
    gmm = GaussianMixture(n_components=3, covariance_type=cov_type, random_state=42, max_iter=200)
    labels = gmm.fit_predict(X_s)
    bic = gmm.bic(X_s)
    aic = gmm.aic(X_s)
    sil = silhouette_score(X_s, labels) if len(set(labels)) > 1 else -1
    print(f'  {cov_type}: BIC={bic:.0f}, AIC={aic:.0f}, silhouette={sil:.3f}')

print('\nfull: más flexible (cada cluster su propia elipse)')
print('tied: misma forma para todos')
print('diag: ejes alineados a coordenadas')
print('spherical: esferas (todas las direcciones igual)')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: GaussianMixture covariance_type — 'full' vs 'tied' vs 'diag' vs 'spherical'.*

1. `labels = gmm.fit_predict(X_s)` — Genera predicciones sobre nuevos datos.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 16: BIC/AIC para seleccionar número de componentes en GMM

```python
X_s = scaler.fit_transform(df[['precio', 'margen', 'descuento', 'ventas_semana']].values)

bic_scores = []
aic_scores = []
K_range = range(1, 11)

for k in K_range:
    gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42, max_iter=200)
    gmm.fit(X_s)
    bic_scores.append(gmm.bic(X_s))
    aic_scores.append(gmm.aic(X_s))

plt.figure(figsize=(10, 5))
plt.plot(K_range, bic_scores, 'bo-', label='BIC', linewidth=2)
plt.plot(K_range, aic_scores, 'rs-', label='AIC', linewidth=2)
plt.xlabel('Número de componentes (k)')
plt.ylabel('Valor del criterio')
plt.title('BIC y AIC para seleccionar k en GaussianMixture')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

mejor_k_bic = K_range[np.argmin(bic_scores)]
mejor_k_aic = K_range[np.argmin(aic_scores)]
print(f'Mejor k según BIC: {mejor_k_bic} (BIC={min(bic_scores):.0f})')
print(f'Mejor k según AIC: {mejor_k_aic} (AIC={min(aic_scores):.0f})')
print('\nBIC penaliza más la complejidad (prefiere modelos más simples).')
print('AIC puede sobreajustar ligeramente.')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: BIC/AIC para seleccionar número de componentes en GMM.*

1. `gmm.fit(X_s)` — Entrena el modelo con los datos de entrenamiento.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 17: Comparar DBSCAN vs KMeans vs Agglomerative en mismo dataset

```python
X_s = scaler.fit_transform(df[['precio', 'margen', 'descuento']].values)

# KMeans
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels_km = kmeans.fit_predict(X_s)

# DBSCAN
dbscan = DBSCAN(eps=0.4, min_samples=5)
labels_db = dbscan.fit_predict(X_s)

# Agglomerative
agglo = AgglomerativeClustering(n_clusters=4, linkage='ward')
labels_ag = agglo.fit_predict(X_s)

# PCA para visualizar
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_s)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
titles = ['KMeans', 'DBSCAN', 'Agglomerative']
labels_list = [labels_km, labels_db, labels_ag]

for ax, labels, title in zip(axes, labels_list, titles):
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis', alpha=0.7, edgecolors='k')
    ax.set_title(f'{title} ({n_clusters} clusters)')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

plt.tight_layout()
plt.show()

print('Comparación de algoritmos:')
metricas = []
for name, labels in zip(['KMeans', 'DBSCAN', 'Agglomerative'], labels_list):
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters > 1:
        sil = silhouette_score(X_s, labels)
    else:
        sil = -1
    metricas.append({'Algoritmo': name, 'Clusters': n_clusters, 'Silhouette': round(sil, 3)})

print(pd.DataFrame(metricas).to_string(index=False))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: Comparar DBSCAN vs KMeans vs Agglomerative en mismo dataset.*

1. KMeans
2. DBSCAN
3. Agglomerative
4. PCA para visualizar

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 18: Integrador — segmentación avanzada de productos con DBSCAN + PCA

```python
# Dataset completo
df_prod = df.copy()
df_prod['costo_logistico'] = np.random.uniform(2, 20, n)
df_prod['rating'] = np.random.uniform(1, 5, n)
df_prod['peso'] = np.random.uniform(0.1, 25, n)

features_all = ['precio', 'margen', 'descuento', 'dias_stock', 'ventas_semana',
                'devoluciones', 'costo_logistico', 'rating', 'peso']
X_all = scaler.fit_transform(df_prod[features_all].values)

# PCA para reducir dimensionalidad
pca = PCA(n_components=0.90, random_state=42)
X_pca = pca.fit_transform(X_all)

print(f'Features originales: {X_all.shape[1]}')
print(f'Features tras PCA (90% varianza): {X_pca.shape[1]}')
print(f'Varianza explicada: {sum(pca.explained_variance_ratio_)*100:.1f}%')

# DBSCAN en espacio PCA
dbscan_final = DBSCAN(eps=0.5, min_samples=5, metric='euclidean')
labels_final = dbscan_final.fit_predict(X_pca)

n_clusters = len(set(labels_final)) - (1 if -1 in labels_final else 0)
n_noise = list(labels_final).count(-1)
print(f'Clusters encontrados: {n_clusters}')
print(f'Puntos de ruido: {n_noise} ({n_noise/len(labels_final)*100:.1f}%)')

# Visualizar en 2D
pca_2d = PCA(n_components=2, random_state=42)
X_2d = pca_2d.fit_transform(X_all)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_final,
                      cmap='tab10', alpha=0.7, edgecolors='k', s=40)
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
plt.title(f'DBSCAN + PCA: {n_clusters} clusters, {n_noise} outliers')
plt.colorbar(scatter, label='Cluster')
plt.show()

# Perfiles de los clusters
df_prod['cluster_final'] = labels_final
print('\n--- Perfiles de los clusters ---')
for cluster_id in sorted(set(labels_final)):
    miembros = df_prod[df_prod['cluster_final'] == cluster_id]
    if cluster_id == -1:
        print(f'\n--- RUIDO (n={len(miembros)}) ---')
    else:
        print(f'\n--- Cluster {cluster_id} (n={len(miembros)}) ---')
    print(miembros[features_all].mean().round(2))

# Métricas (solo clusters, excluyendo ruido)
idx_valid = labels_final != -1
if sum(idx_valid) > 0 and n_clusters > 1:
    sil = silhouette_score(X_pca[idx_valid], labels_final[idx_valid])
    ch = calinski_harabasz_score(X_pca[idx_valid], labels_final[idx_valid])
    db = davies_bouldin_score(X_pca[idx_valid], labels_final[idx_valid])
    print(f'\n--- Métricas de validación ---')
    print(f'Silhouette Score: {sil:.3f}')
    print(f'Calinski-Harabasz: {ch:.2f}')
    print(f'Davies-Bouldin: {db:.3f}')

# Recomendación por cluster
print('\n--- Recomendaciones de negocio ---')
for cluster_id in sorted(set(labels_final)):
    if cluster_id == -1:
        continue
    miembros = df_prod[df_prod['cluster_final'] == cluster_id]
    precio_prom = miembros['precio'].mean()
    ventas_prom = miembros['ventas_semana'].mean()
    margen_prom = miembros['margen'].mean()
    if precio_prom > 100 and margen_prom > 30:
        print(f'  Cluster {cluster_id}: Productos PREMIUM (alto valor y margen)')
    elif ventas_prom > 60 and margen_prom < 20:
        print(f'  Cluster {cluster_id}: Productos de VOLUMEN (bajo margen)')
    elif precio_prom < 50:
        print(f'  Cluster {cluster_id}: Productos ECONÓMICOS (bajo precio)')
    else:
        print(f'  Cluster {cluster_id}: Productos ESTÁNDAR')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador — segmentación avanzada de productos con DBSCAN + PCA.*

1. Dataset completo
2. PCA para reducir dimensionalidad
3. DBSCAN en espacio PCA
4. Visualizar en 2D
5. Perfiles de los clusters
6. Métricas (solo clusters, excluyendo ruido)
7. Recomendación por cluster

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Resumen

| Algoritmo | Tipo | Ventajas | Desventajas |
|-----------|------|----------|-------------|
| **DBSCAN** | Densidad | No requiere k, detecta outliers, formas arbitrarias | Sensible a eps, falla con densidades variables |
| **HDBSCAN** | Densidad jerárquica | No requiere eps, densidades variables | Requiere instalación adicional |
| **Agglomerative** | Jerárquico | Dendrograma interpretable, varios linkages | Costoso O(n³), difícil con grandes datos |
| **SpectralClustering** | Grafos | Formas complejas (anilladas, anidadas) | Costoso, requiere elegir affinity |
| **MeanShift** | Densidad | No requiere k, encuentra modas | Costoso O(n²), bandwidth sensible |
| **OPTICS** | Densidad | Reachability plot, múltiples escalas | Complejo de parametrizar |
| **BIRCH** | Incremental | Escalable, bajo consumo de memoria | Sensible al orden de datos |
| **AffinityPropagation** | Propagación | No requiere k, ejemplares reales | Costoso O(n²), damping sensible |
| **GaussianMixture** | Probabilístico | Probabilidades, covarianza flexible | Asume Gaussianas, sensible a inicialización |

**Recomendación:** Usar DBSCAN para descubrimiento exploratorio de segmentos. Agglomerative cuando se necesita interpretabilidad jerárquica. GaussianMixture cuando se requieren probabilidades de pertenencia. Spectral para formas complejas.

---

## Ejercicios

1. Aplica DBSCAN con `eps=0.3, 0.5, 0.8` y `min_samples=5` a las variables `precio`, `margen`, `descuento` de ventas. ¿Cómo cambia el número de clusters y outliers? ¿Qué eps produce la segmentación más útil desde el punto de vista comercial?

2. Usa `AgglomerativeClustering` con `linkage='ward'` y `n_clusters=4`. Calcula silhouette_score para los clusters. Visualiza el dendrograma con una submuestra de 20 productos. ¿A qué distancia se forman los 4 clusters?

3. Compara `KMeans`, `DBSCAN` y `AgglomerativeClustering` en el mismo dataset de ventas (4 features). Crea una tabla comparativa con número de clusters, silhouette score y tiempo de ejecución. ¿Qué algoritmo produce segmentos más homogéneos?

4. Genera un dataset con `make_moons` (300 puntos, noise=0.1). Aplica `SpectralClustering` con `affinity='rbf'` (gamma=0.5, 1, 5, 10) y `affinity='nearest_neighbors'` (k=5, 10, 20). ¿Qué combinación separa mejor las dos lunas?

5. Usa `GaussianMixture` con `covariance_type='full'` para segmentar productos. Calcula BIC y AIC para k=1 a 8. ¿Cuál es el k óptimo según cada criterio? ¿Coinciden? Visualiza las elipses de covarianza de los componentes.

6. Aplica `OPTICS` a las 4 variables de ventas. Genera el reachability plot. ¿Cuántos clusters identificas visualmente en el plot? ¿Coinciden con los que OPTICS detecta automáticamente?

7. Simula un dataset con 20% de puntos de ruido (outliers). Compara cómo manejan el ruido DBSCAN, KMeans y AgglomerativeClustering. ¿Cuál de los tres es más robusto a outliers? ¿Por qué?

8. Proyecta los datos de ventas con PCA a 2 componentes y luego aplica DBSCAN, Agglomerative y GaussianMixture. Visualiza los clusters en el espacio PCA. ¿Qué algoritmo produce la segmentación más natural? Interpreta los clusters en términos de negocio.

---

*Teoría y práctica de clustering avanzado con scikit-learn aplicada a segmentación de productos y detección de patrones en ventas, compras e inventarios.*
