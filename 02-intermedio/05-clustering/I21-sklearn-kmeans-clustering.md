# I21: K-Means y Clustering Particional con scikit-learn

## Aplicación a Ventas, Compras e Inventarios

El clustering particional divide los datos en **K grupos** (clusters) donde cada punto pertenece al cluster con el centroide más cercano. En el contexto comercial, K-Means permite **segmentar clientes, agrupar productos por comportamiento de venta, identificar patrones de compra y optimizar inventarios**.

### Fundamentos Teóricos

**K-Means** busca minimizar la **inercia** (suma de distancias al cuadrado dentro de cada cluster):
- Inicializa K centroides aleatorios
- Asigna cada punto al centroide más cercano
- Recalcula centroides como la media de los puntos asignados
- Repite hasta convergencia (los centroides no cambian)

**Parámetros clave de KMeans:**
- `n_clusters`: número de grupos a formar
- `init`: método de inicialización (`'k-means++'` — inteligente, o `'random'`)
- `n_init`: veces que se ejecuta con diferentes semillas (se queda con la mejor)
- `max_iter`: máximo de iteraciones por ejecución
- `tol`: tolerancia para declarar convergencia
- `random_state`: semilla para reproducibilidad

**MiniBatchKMeans:** versión optimizada que procesa mini-lotes en vez de todo el dataset. Escala a millones de filas. `batch_size` controla el tamaño del lote.

**Métricas de evaluación:**
- **Inercia** (`inertia_`): suma de distancias intra-cluster (menor = mejor, pero monótona decreciente con K)
- **Silhouette Score** (`silhouette_score`): mide cohesión intra-cluster vs separación inter-cluster ([-1, 1], mejor > 0.5)
- **Silhouette Samples** (`silhouette_samples`): coeficiente por punto individual
- **Calinski-Harabasz Score** (`calinski_harabasz_score`): razón de varianza entre clusters vs intra-cluster (mayor = mejor)
- **Davies-Bouldin Score** (`davies_bouldin_score`): similitud promedio entre clusters (menor = mejor)
- **Método del Codo (Elbow)**: gráfico de inercia vs K para encontrar el punto de inflexión

**RFM:** segmentación clásica por Recencia (última compra), Frecuencia (cuántas compras) y Monto (gasto total).

---

## Ejemplos Prácticos con sklearn

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (silhouette_score, silhouette_samples,
                             calinski_harabasz_score, davies_bouldin_score)
from sklearn.datasets import make_blobs
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# DATASET SIMULADO DE VENTAS
# ============================================================
np.random.seed(42)
n = 500

df = pd.DataFrame({
    'precio': np.random.uniform(10, 200, n),
    'margen': np.random.uniform(5, 60, n),
    'descuento': np.random.uniform(0, 0.5, n),
    'dias_stock': np.random.randint(1, 120, n),
    'ventas_semana': np.random.poisson(50, n),
    'devoluciones': np.random.poisson(3, n),
    'categoria': np.random.choice(['A', 'B', 'C'], n),
})

# Relaciones realistas entre variables
df['margen'] = df['margen'] + 0.2 * df['precio'] + np.random.normal(0, 3, n)
df['ventas_semana'] = (df['ventas_semana'] - 0.3 * df['precio']
                        + 2 * df['descuento'] * 100 + np.random.normal(0, 10, n)).clip(0)
df['dias_stock'] = (df['dias_stock'] - 0.1 * df['ventas_semana']
                     + np.random.normal(0, 5, n)).clip(1)

# Dataset de clientes para RFM
clientes_rfm = pd.DataFrame({
    'cliente_id': range(1, 201),
    'recencia_dias': np.random.exponential(30, 200).astype(int),
    'frecuencia_mes': np.random.exponential(3, 200),
    'monto_promedio': np.random.lognormal(4, 0.8, 200),
})
# Correlaciones realistas
clientes_rfm['frecuencia_mes'] = clientes_rfm['frecuencia_mes'] + 0.01 * clientes_rfm['monto_promedio']
clientes_rfm['recencia_dias'] = clientes_rfm['recencia_dias'] - 0.5 * clientes_rfm['frecuencia_mes']
clientes_rfm['recencia_dias'] = clientes_rfm['recencia_dias'].clip(1)

print('Dataset de ventas creado:')
print(df.head())
print(f'\nShape: {df.shape}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
# Las primeras filas muestran la estructura de los datos procesados:
#    Las columnas dependen del DataFrame utilizado.
# 0    Los valores dependen de los datos de entrada.
# 1    valor4    valor5    valor6
```

**Explicación línea por línea:**

*Ejemplos Prácticos con sklearn.*

1. ============================================================
2. DATASET SIMULADO DE VENTAS
3. ============================================================
4. Relaciones realistas entre variables
5. Dataset de clientes para RFM
6. Correlaciones realistas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 1: KMeans básico — segmentar productos por precio y margen (k=3)

```python
X = df[['precio', 'margen']].values

kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

plt.figure(figsize=(8, 5))
scatter = plt.scatter(df['precio'], df['margen'], c=df['cluster'],
                      cmap='viridis', alpha=0.7, edgecolors='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='red', marker='X', s=200, label='Centroides')
plt.xlabel('Precio ($)')
plt.ylabel('Margen (%)')
plt.title('KMeans: Segmentación de productos (k=3)')
plt.colorbar(scatter, label='Cluster')
plt.legend()
plt.show()

print('Distribución por cluster:')
print(df['cluster'].value_counts().sort_index())
print(f'\nInercia: {kmeans.inertia_:.2f}')
print(f'Iteraciones: {kmeans.n_iter_}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: KMeans básico — segmentar productos por precio y margen (k=3).*

1. `df['cluster'] = kmeans.fit_predict(X)` — Genera predicciones sobre nuevos datos.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 2: KMeans++ init vs random init

```python
X = df[['precio', 'margen']].values

kmeans_pp = KMeans(n_clusters=4, init='k-means++', n_init=1, random_state=42)
kmeans_rnd = KMeans(n_clusters=4, init='random', n_init=1, random_state=42)

kmeans_pp.fit(X)
kmeans_rnd.fit(X)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, km, title in zip(axes, [kmeans_pp, kmeans_rnd],
                          ['k-means++ init', 'random init']):
    ax.scatter(X[:, 0], X[:, 1], c=km.labels_, cmap='viridis', alpha=0.7, edgecolors='k')
    ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
               c='red', marker='X', s=200)
    ax.set_title(f'{title}\nInercia: {km.inertia_:.2f}')
    ax.set_xlabel('Precio')
    ax.set_ylabel('Margen')

plt.tight_layout()
plt.show()

print(f'k-means++ inercia: {kmeans_pp.inertia_:.2f}')
print(f'random init inercia: {kmeans_rnd.inertia_:.2f}')
print('k-means++ suele converger a óptimo global con menor inercia')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: KMeans++ init vs random init.*

1. `kmeans_pp.fit(X)` — Entrena el modelo con los datos de entrenamiento.
2. `kmeans_rnd.fit(X)` — Entrena el modelo con los datos de entrenamiento.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 3: n_init — convergencia consistente

```python
X = df[['precio', 'margen', 'descuento']].values

resultados = []
for n in [1, 5, 10, 20]:
    inertias = []
    for _ in range(30):
        km = KMeans(n_clusters=4, n_init=n, random_state=None).fit(X)
        inertias.append(km.inertia_)
    resultados.append({
        'n_init': n,
        'media': np.mean(inertias),
        'std': np.std(inertias),
        'min': np.min(inertias),
        'max': np.max(inertias),
    })

df_result = pd.DataFrame(resultados)
print('Efecto de n_init en estabilidad:')
print(df_result.to_string(index=False))
print('\nMayor n_init → menor varianza, pero más cómputo')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: n_init — convergencia consistente.*

1. `km = KMeans(n_clusters=4, n_init=n, random_state=None).fit(X)` — Entrena el modelo con los datos de entrenamiento.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 4: Elbow method — inercia vs k (1 a 10)

```python
X = df[['precio', 'margen', 'descuento', 'ventas_semana']].values

inertias = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Número de clusters (k)')
plt.ylabel('Inercia')
plt.title('Método del Codo (Elbow) para elegir k')
plt.axvline(x=3, color='r', linestyle='--', alpha=0.5, label='Posible codo')
plt.axvline(x=4, color='g', linestyle='--', alpha=0.5, label='Otro codo')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print('Inercias por k:')
for k, inercia in zip(K_range, inertias):
    print(f'  k={k}: {inercia:.2f}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: Elbow method — inercia vs k (1 a 10).*

1. `km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)` — Entrena el modelo con los datos de entrenamiento.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 5: Silhouette score — evaluar k (1 a 10)

```python
X = df[['precio', 'margen', 'descuento', 'ventas_semana']].values

sil_scores = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    sil = silhouette_score(X, km.labels_)
    sil_scores.append(sil)

plt.figure(figsize=(8, 5))
plt.plot(K_range, sil_scores, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Número de clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score para elegir k')
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Bueno (>0.5)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

mejor_k = K_range[np.argmax(sil_scores)]
print(f'Mejor k según silhouette: {mejor_k} (score={max(sil_scores):.3f})')
for k, s in zip(K_range, sil_scores):
    print(f'  k={k}: {s:.3f}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: Silhouette score — evaluar k (1 a 10).*

1. `km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)` — Entrena el modelo con los datos de entrenamiento.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 6: Silhouette plot por cluster

```python
from matplotlib.ticker import FixedLocator, FixedFormatter

X = df[['precio', 'margen', 'descuento', 'ventas_semana']].values
k = 3
km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)

sil_vals = silhouette_samples(X, km.labels_)
y_lower = 10

plt.figure(figsize=(10, 6))
for i in range(k):
    cluster_vals = sil_vals[km.labels_ == i]
    cluster_vals.sort()
    y_upper = y_lower + len(cluster_vals)
    color = plt.cm.viridis(i / k)
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_vals,
                      facecolor=color, alpha=0.7, edgecolor='k')
    plt.text(-0.05, y_lower + 0.5 * len(cluster_vals), str(i))
    y_lower = y_upper + 10

sil_avg = silhouette_score(X, km.labels_)
plt.axvline(x=sil_avg, color='red', linestyle='--', label=f'Media: {sil_avg:.3f}')
plt.xlabel('Coeficiente Silhouette')
plt.ylabel('Cluster')
plt.title('Silhouette Plot por cluster (k=3)')
plt.legend()
plt.show()

print(f'Silhouette promedio: {sil_avg:.3f}')
for i in range(k):
    print(f'  Cluster {i}: media={sil_vals[km.labels_==i].mean():.3f}, '
          f'n={sum(km.labels_==i)}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: Silhouette plot por cluster.*

1. `from matplotlib.ticker import FixedLocator, FixedFormatter` — Importa las librerías necesarias para el análisis.
2. `km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)` — Entrena el modelo con los datos de entrenamiento.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 7: Calinski-Harabasz score — razón de varianza

```python
X = df[['precio', 'margen', 'descuento', 'ventas_semana']].values

ch_scores = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    ch = calinski_harabasz_score(X, km.labels_)
    ch_scores.append(ch)

plt.figure(figsize=(8, 5))
plt.plot(K_range, ch_scores, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Número de clusters (k)')
plt.ylabel('Calinski-Harabasz Score')
plt.title('Calinski-Harabasz: mayor es mejor (varianza entre / varianza intra)')
plt.grid(alpha=0.3)
plt.show()

mejor_k = K_range[np.argmax(ch_scores)]
print(f'Mejor k según Calinski-Harabasz: {mejor_k}')
for k, s in zip(K_range, ch_scores):
    print(f'  k={k}: {s:.2f}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: Calinski-Harabasz score — razón de varianza.*

1. `km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)` — Entrena el modelo con los datos de entrenamiento.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 8: Davies-Bouldin score — similitud entre clusters

```python
X = df[['precio', 'margen', 'descuento', 'ventas_semana']].values

db_scores = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    db = davies_bouldin_score(X, km.labels_)
    db_scores.append(db)

plt.figure(figsize=(8, 5))
plt.plot(K_range, db_scores, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Número de clusters (k)')
plt.ylabel('Davies-Bouldin Score')
plt.title('Davies-Bouldin: menor es mejor (baja similitud entre clusters)')
plt.grid(alpha=0.3)
plt.show()

mejor_k = K_range[np.argmin(db_scores)]
print(f'Mejor k según Davies-Bouldin: {mejor_k}')
for k, s in zip(K_range, db_scores):
    print(f'  k={k}: {s:.3f}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: Davies-Bouldin score — similitud entre clusters.*

1. `km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)` — Entrena el modelo con los datos de entrenamiento.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 9: Visualizar clusters 2D con scatterplot (coloreado por cluster)

```python
X = df[['precio', 'margen', 'descuento', 'ventas_semana']].values
k = 4
km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
df['cluster_4'] = km.labels_

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
pairs = [('precio', 'margen'), ('precio', 'descuento'), ('precio', 'ventas_semana'),
         ('margen', 'descuento'), ('margen', 'ventas_semana'), ('descuento', 'ventas_semana')]

for ax, (x_col, y_col) in zip(axes.flat, pairs):
    scatter = ax.scatter(df[x_col], df[y_col], c=df['cluster_4'],
                         cmap='viridis', alpha=0.7, edgecolors='k', s=30)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)

plt.suptitle('Pares de features coloreados por cluster (k=4)', fontsize=14)
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 9: Visualizar clusters 2D con scatterplot (coloreado por cluster).*

1. `km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)` — Entrena el modelo con los datos de entrenamiento.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 10: Interpretar centroides (perfil de cada cluster)

```python
X = df[['precio', 'margen', 'descuento', 'ventas_semana', 'dias_stock']].values
k = 3
km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
df['cluster_perfil'] = km.labels_

centroides = pd.DataFrame(
    km.cluster_centers_,
    columns=['precio', 'margen', 'descuento', 'ventas_semana', 'dias_stock']
)
centroides.index = [f'Cluster {i}' for i in range(k)]
print('Centroides (perfil de cada cluster):')
print(centroides.round(2))

# Perfil descriptivo
for i in range(k):
    miembros = df[df['cluster_perfil'] == i]
    print(f'\n--- Cluster {i} (n={len(miembros)}) ---')
    print(f'  Precio medio: ${miembros["precio"].mean():.1f}')
    print(f'  Margen medio: {miembros["margen"].mean():.1f}%')
    print(f'  Descuento medio: {miembros["descuento"].mean():.2f}')
    print(f'  Ventas/semana: {miembros["ventas_semana"].mean():.0f}')
    print(f'  Días en stock: {miembros["dias_stock"].mean():.0f}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: Interpretar centroides (perfil de cada cluster).*

1. Perfil descriptivo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 11: MiniBatchKMeans — escalar a grandes datasets

```python
# Simular dataset grande
n_grande = 10000
X_grande = np.column_stack([
    np.random.uniform(10, 200, n_grande),
    np.random.uniform(5, 60, n_grande),
    np.random.uniform(0, 0.5, n_grande),
])

mbk = MiniBatchKMeans(n_clusters=5, batch_size=100, random_state=42, n_init=3)
mbk.fit(X_grande)

print('MiniBatchKMeans en dataset grande (10000 filas):')
print(f'  Clusters: {mbk.n_clusters}')
print(f'  Batch size: {mbk.batch_size}')
print(f'  Inercia: {mbk.inertia_:.2f}')
print(f'  Iteraciones: {mbk.n_iter_}')
print(f'  N_init realizado: {mbk.n_init}')
print(f'\nCentroides:')
print(pd.DataFrame(mbk.cluster_centers_, columns=['precio', 'margen', 'descuento']).round(2))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: MiniBatchKMeans — escalar a grandes datasets.*

1. Simular dataset grande

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 12: Comparar KMeans vs MiniBatchKMeans (velocidad y calidad)

```python
import time

X = df[['precio', 'margen', 'descuento', 'ventas_semana']].values

resultados_comp = []
for n_muestras in [500, 2000, 5000]:
    X_sub = X[:min(n_muestras, len(X))]

    t0 = time.time()
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    km.fit(X_sub)
    t_km = time.time() - t0

    t0 = time.time()
    mbk = MiniBatchKMeans(n_clusters=4, random_state=42, batch_size=100, n_init=10)
    mbk.fit(X_sub)
    t_mbk = time.time() - t0

    resultados_comp.append({
        'n_muestras': n_muestras,
        'KMeans_inercia': km.inertia_,
        'MiniBatch_inercia': mbk.inertia_,
        'KMeans_tiempo_s': round(t_km, 3),
        'MiniBatch_tiempo_s': round(t_mbk, 3),
    })

df_comp = pd.DataFrame(resultados_comp)
print('Comparación KMeans vs MiniBatchKMeans:')
print(df_comp.to_string(index=False))
print('\nMiniBatch es más rápido pero puede tener mayor inercia.')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: Comparar KMeans vs MiniBatchKMeans (velocidad y calidad).*

1. `import time` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 13: predict — asignar nuevo producto a cluster existente

```python
X = df[['precio', 'margen']].values
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X)

# Nuevos productos
nuevos = np.array([
    [50, 25],
    [150, 45],
    [25, 10],
    [200, 55],
])

clusters_nuevos = kmeans.predict(nuevos)
distancias = kmeans.transform(nuevos).min(axis=1)

print('Asignación de nuevos productos a clusters existentes:')
for i, (precio, margen) in enumerate(nuevos):
    print(f'  Producto (${precio}, {margen}%) → Cluster {clusters_nuevos[i]} '
          f'(distancia al centroide: {distancias[i]:.2f})')

# Visualizar
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.5, edgecolors='k')
plt.scatter(nuevos[:, 0], nuevos[:, 1], c=clusters_nuevos, cmap='viridis',
            marker='s', s=150, edgecolors='red', linewidths=2, label='Nuevos productos')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='red', marker='X', s=200, label='Centroides')
plt.xlabel('Precio')
plt.ylabel('Margen')
plt.title('Asignación de nuevos productos a clusters')
plt.legend()
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: predict — asignar nuevo producto a cluster existente.*

1. Nuevos productos
2. Visualizar

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 14: transform — distancia de cada punto a cada centro

```python
X = df[['precio', 'margen']].values[:10]  # 10 productos
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X)

distancias = kmeans.transform(X)

print('Distancia de cada producto a cada centroide:')
print(f'{"Producto":>10} {"C_0":>8} {"C_1":>8} {"C_2":>8} {"Cluster":>8} {"Min_dist":>8}')
for i in range(len(X)):
    cluster_asignado = kmeans.labels_[i]
    min_dist = distancias[i].min()
    print(f'{i:>10} {distancias[i,0]:>8.2f} {distancias[i,1]:>8.2f} '
          f'{distancias[i,2]:>8.2f} {cluster_asignado:>8} {min_dist:>8.2f}')

# Interpretación: la distancia mínima determina la asignación
print('\nLa columna con distancia mínima determina el cluster asignado.')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: transform — distancia de cada punto a cada centro.*

1. Interpretación: la distancia mínima determina la asignación

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 15: Escalar datos antes de KMeans (StandardScaler)

```python
from sklearn.preprocessing import StandardScaler

X = df[['precio', 'margen', 'descuento', 'ventas_semana', 'dias_stock']].values

# Sin escalar
km_sin = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X)

# Con escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
km_con = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(X[:, 0], X[:, 1], c=km_sin.labels_, cmap='viridis', alpha=0.7)
axes[0].set_title('Sin escalar (precio domina)')
axes[0].set_xlabel('Precio')
axes[0].set_ylabel('Margen')

axes[1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=km_con.labels_, cmap='viridis', alpha=0.7)
axes[1].set_title('Con StandardScaler')
axes[1].set_xlabel('Precio (std)')
axes[1].set_ylabel('Margen (std)')

plt.tight_layout()
plt.show()

print('Sin escalar - distribución clusters:')
print(pd.Series(km_sin.labels_).value_counts().sort_index())
print('Con escalado - distribución clusters:')
print(pd.Series(km_con.labels_).value_counts().sort_index())
print('\nEscalar es crítico cuando las variables tienen escalas distintas.')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: Escalar datos antes de KMeans (StandardScaler).*

1. Sin escalar
2. Con escalado

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 16: Pipeline: StandardScaler + KMeans

```python
from sklearn.pipeline import Pipeline

X = df[['precio', 'margen', 'descuento', 'ventas_semana']].values

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=3, random_state=42, n_init=10)),
])

pipeline.fit(X)
labels = pipeline.predict(X)
centroides_scaled = pipeline.named_steps['kmeans'].cluster_centers_

print('Pipeline (StandardScaler + KMeans) ejecutado.')
print(f'Inercia: {pipeline.named_steps["kmeans"].inertia_:.2f}')
print(f'Distribución de clusters:')
print(pd.Series(labels).value_counts().sort_index())

# Score: negativo de la inercia (mayor = mejor)
score = pipeline.score(X)
print(f'Score (negativo de inercia): {score:.2f}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: Pipeline: StandardScaler + KMeans.*

1. Score: negativo de la inercia (mayor = mejor)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 17: Segmentación RFM con KMeans (recencia, frecuencia, monto)

```python
# Usar dataset clientes_rfm creado arriba
X_rfm = clientes_rfm[['recencia_dias', 'frecuencia_mes', 'monto_promedio']].values

# Escalar RFM (las escalas son muy diferentes)
scaler_rfm = StandardScaler()
X_rfm_scaled = scaler_rfm.fit_transform(X_rfm)

# Elbow para RFM
inertias_rfm = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_rfm_scaled)
    inertias_rfm.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertias_rfm, 'bo-')
plt.xlabel('k')
plt.ylabel('Inercia')
plt.title('Elbow para segmentación RFM')
plt.grid(alpha=0.3)
plt.show()

# Elegir k=4 y segmentar
k = 4
km_rfm = KMeans(n_clusters=k, random_state=42, n_init=10)
clientes_rfm['cluster_rfm'] = km_rfm.fit_predict(X_rfm_scaled)

# Perfiles RFM
perfiles = clientes_rfm.groupby('cluster_rfm').agg({
    'recencia_dias': 'mean',
    'frecuencia_mes': 'mean',
    'monto_promedio': 'mean',
    'cliente_id': 'count',
}).rename(columns={'cliente_id': 'count'})

print('Perfiles RFM por cluster:')
print(perfiles.round(2))

# Nombrar segmentos
for i in perfiles.index:
    r = perfiles.loc[i, 'recencia_dias']
    f = perfiles.loc[i, 'frecuencia_mes']
    m = perfiles.loc[i, 'monto_promedio']
    if r < 15 and f > 4 and m > 100:
        label = '🟢 Clientes Premium (alta frecuencia y gasto)'
    elif r < 30 and f > 2:
        label = '🔵 Clientes Regulares'
    elif r > 60 and f < 1:
        label = '🔴 Clientes Perdidos (riesgo)'
    else:
        label = '🟡 Ocasionales / Potenciales'
    print(f'  Cluster {i}: {label}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: Segmentación RFM con KMeans (recencia, frecuencia, monto).*

1. Usar dataset clientes_rfm creado arriba
2. Escalar RFM (las escalas son muy diferentes)
3. Elbow para RFM
4. Elegir k=4 y segmentar
5. Perfiles RFM
6. Nombrar segmentos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 18: Integrador — segmentar clientes para campañas de marketing

```python
# Dataset integrado de clientes con más variables
clientes_mkt = clientes_rfm.copy()
clientes_mkt['gasto_total'] = clientes_mkt['frecuencia_mes'] * clientes_mkt['monto_promedio'] * 12
clientes_mkt['antiguedad_meses'] = np.random.randint(1, 60, 200)
clientes_mkt['canal'] = np.random.choice(['web', 'app', 'tienda', 'telefono'], 200)
clientes_mkt['devoluciones'] = np.random.poisson(0.5, 200)
clientes_mkt['email_abierto'] = np.random.uniform(0, 1, 200)
clientes_mkt['cupones_usados'] = np.random.randint(0, 10, 200)

# Features para clustering
features_mkt = ['recencia_dias', 'frecuencia_mes', 'monto_promedio',
                'gasto_total', 'antiguedad_meses', 'devoluciones',
                'email_abierto', 'cupones_usados']
X_mkt = clientes_mkt[features_mkt].values

# Pipeline completo
pipeline_mkt = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=4, random_state=42, n_init=10)),
])

clientes_mkt['segmento'] = pipeline_mkt.fit_predict(X_mkt)

# Análisis de segmentos
segmentos = clientes_mkt.groupby('segmento').agg({
    'recencia_dias': 'mean',
    'frecuencia_mes': 'mean',
    'monto_promedio': 'mean',
    'gasto_total': 'mean',
    'antiguedad_meses': 'mean',
    'devoluciones': 'mean',
    'email_abierto': 'mean',
    'cupones_usados': 'mean',
    'cliente_id': 'count',
}).rename(columns={'cliente_id': 'n_clientes'})

print('Segmentación de clientes para campañas:')
print(segmentos.round(2))

# Recomendación de estrategia por segmento
print('\n--- Estrategias de Marketing por Segmento ---')
for i in segmentos.index:
    r = clientes_mkt[clientes_mkt['segmento'] == i]
    print(f'\nSegmento {i} (n={len(r)}):')
    if r['recencia_dias'].mean() < 20 and r['frecuencia_mes'].mean() > 3:
        print('  ˃ Campaña: Programa de fidelización y cross-selling')
    elif r['recencia_dias'].mean() < 40:
        print('  ˃ Campaña: Ofertas personalizadas para aumentar frecuencia')
    elif r['recencia_dias'].mean() < 80:
        print('  ˃ Campaña: Email reactivación con descuento')
    else:
        print('  ˃ Campaña: Oferta de bienvenida para re-enganche')

# Evaluación final
sil_mkt = silhouette_score(X_mkt, clientes_mkt['segmento'])
ch_mkt = calinski_harabasz_score(X_mkt, clientes_mkt['segmento'])
db_mkt = davies_bouldin_score(X_mkt, clientes_mkt['segmento'])
print(f'\n--- Métricas de la segmentación ---')
print(f'Silhouette Score: {sil_mkt:.3f}')
print(f'Calinski-Harabasz: {ch_mkt:.2f}')
print(f'Davies-Bouldin: {db_mkt:.3f}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador — segmentar clientes para campañas de marketing.*

1. Dataset integrado de clientes con más variables
2. Features para clustering
3. Pipeline completo
4. Análisis de segmentos
5. Recomendación de estrategia por segmento
6. Evaluación final

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Resumen

| Técnica | Cuándo usarla | Parámetros clave |
|---------|--------------|------------------|
| **KMeans** | Segmentación general, datos numéricos | `n_clusters`, `n_init`, `init` |
| **MiniBatchKMeans** | Grandes volúmenes (>100k filas) | `batch_size`, `n_init` |
| **Elbow method** | Elegir k visualmente | Rango de k, inercia |
| **Silhouette Score** | Validar calidad clusters | k, sample size |
| **Calinski-Harabasz** | Comparar densidad clusters | k, varianza intra/inter |
| **Davies-Bouldin** | Evaluar separación | k, similitud entre clusters |
| **StandardScaler + KMeans** | Variables en distintas escalas | escalado antes de cluster |
| **Pipeline** | Flujo reproducible | scaler + cluster |
| **RFM** | Segmentación de clientes | recencia, frecuencia, monto |
| **predict** | Asignar nuevos puntos | modelo entrenado |
| **transform** | Distancia a todos los centroides | matriz de distancias |

**Recomendación:** Para datos comerciales (ventas, clientes, inventarios) siempre escalar antes de KMeans. Usar al menos `n_init=10` para convergencia confiable. Combinar Elbow + Silhouette para elegir k. Interpretar siempre los centroides en términos de negocio.

---

## Ejercicios

1. Usando el dataset de ventas (`df`), aplica KMeans con `k=5` sobre `precio`, `margen`, `descuento`, `ventas_semana` y `dias_stock`. Calcula las 4 métricas de evaluación (silhouette, CH, DB, inercia). ¿Qué cluster tiene el margen más alto?

2. Implementa el método del codo para elegir el mejor k entre 1 y 15. ¿En qué k se produce el punto de inflexión? Compáralo con el k óptimo según silhouette_score.

3. Escala los datos con `StandardScaler` y repite el ejercicio 1. ¿Cambian los clusters? ¿Por qué? Compara la distribución de clusters antes y después.

4. Simula un dataset de 500 productos con `make_blobs(n_samples=500, centers=4, n_features=3)`. Aplica KMeans con `init='random'` y `n_init=1` diez veces. ¿Cuánta variabilidad hay en la inercia? Repite con `n_init=10` y compara.

5. Crea un `Pipeline` con `StandardScaler` + `KMeans(n_clusters=3)`. Usa `predict` para asignar 5 nuevos productos inventados. ¿A qué cluster pertenece cada uno?

6. Usando el dataset de clientes RFM, segmenta con k=3, 4, 5. Calcula silhouette_score para cada k. Interpreta los perfiles de los centroides en términos de negocio. ¿Qué k recomiendas para una campaña de marketing?

7. Compara `KMeans` vs `MiniBatchKMeans` en un dataset de 20000 filas sintéticas (5 features). Mide tiempo de entrenamiento e inercia para `batch_size=50, 100, 500, 1000`. ¿Cuál es el mejor trade-off velocidad/calidad?

8. Usa `transform` en KMeans para obtener la distancia de cada producto a los 3 centroides. Identifica los 5 productos más cercanos al centroide del cluster 0 y los 5 más lejanos. Interpreta sus características comerciales.

---

*Teoría y práctica de K-Means con scikit-learn aplicada a segmentación de productos, clientes e inventarios en el dominio de ventas.*
