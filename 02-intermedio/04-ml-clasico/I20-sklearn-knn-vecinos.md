# I20: KNN y Métodos de Vecinos con scikit-learn

## 1. Fundamentos Teóricos

**K-Nearest Neighbors (KNN)** es un algoritmo no paramétrico que clasifica/regresa basándose en los k vecinos más cercanos en el espacio de features.

### Principio básico
- Para clasificar un nuevo punto, se encuentran los k vecinos más cercanos
- La clase se determina por **voto mayoritario** (clasificación) o **promedio** (regresión)
- La distancia se mide con alguna métrica (Euclidean, Manhattan, cosine, etc.)

### Parámetros clave
| Parámetro | Descripción |
|-----------|-------------|
| n_neighbors | Número de vecinos (k) |
| weights | 'uniform' (mismo peso) o 'distance' (inverso de distancia) |
| algorithm | 'auto', 'ball_tree', 'kd_tree', 'brute' |
| leaf_size | Tamaño de hoja para árboles (velocidad vs memoria) |
| p | Potencia de Minkowski: 1=Manhattan, 2=Euclidean |
| metric | 'euclidean', 'manhattan', 'cosine', 'minkowski' |

### Variantes
- **KNeighborsClassifier/Regressor**: KNN clásico
- **NearestNeighbors**: Encontrar vecinos sin supervisión
- **RadiusNeighborsClassifier**: Vecinos dentro de un radio fijo
- **NearestCentroid**: Clasifica al centroide más cercano
- **NeighborhoodComponentsAnalysis**: Aprende una métrica de distancia

### IMPORTANTE: Escalado
KNN es **extremadamente sensible** a la escala. **Siempre escalar** antes de KNN.

---

## 2. Ejemplos Prácticos con sklearn

```python
# Configuracion inicial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import (KNeighborsClassifier, KNeighborsRegressor,
                                NearestNeighbors, RadiusNeighborsClassifier,
                                NearestCentroid, NeighborhoodComponentsAnalysis)
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings('ignore')
import time

np.random.seed(42)

# ============================================================
# EJEMPLO 1: KNeighborsClassifier — clasificar productos (k=3,5,7)
# ============================================================
print("="*60)
print("EJEMPLO 1: KNeighborsClassifier — clasificar productos")
print("="*60)

n = 500
df = pd.DataFrame({
    'precio': np.random.uniform(10, 500, n),
    'margen': np.random.uniform(5, 40, n),
    'stock': np.random.randint(0, 200, n),
    'rotacion': np.random.uniform(0, 100, n),
    'perecible': np.random.choice([0, 1], n),
})
df['alta_rotacion'] = (
    (df['rotacion'] > 30) | (df['precio'] < 100) & (df['margen'] > 20)
).astype(int)
mask_ruido = np.random.random(n) < 0.1
df.loc[mask_ruido, 'alta_rotacion'] = 1 - df.loc[mask_ruido, 'alta_rotacion']

X = df.drop('alta_rotacion', axis=1)
y = df['alta_rotacion']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

for k in [3, 5, 7, 11]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_s, y_train)
    train_acc = (knn.predict(X_train_s) == y_train).mean()
    test_acc = (knn.predict(X_test_s) == y_test).mean()
    print(f"k={k:2d} | train={train_acc:.3f} test={test_acc:.3f}")

print("k pequeno -> sobreajuste (frontera muy detallada)")
print("k grande -> suavizado, posible underfitting")

# ============================================================
# EJEMPLO 2: weights='distance' — ponderar por distancia inversa
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 2: weights='distance' vs 'uniform'")
print("="*60)

for k in [5, 11, 21]:
    for w in ['uniform', 'distance']:
        knn_w = KNeighborsClassifier(n_neighbors=k, weights=w)
        knn_w.fit(X_train_s, y_train)
        test_acc = (knn_w.predict(X_test_s) == y_test).mean()
        print(f"k={k:2d}, weights={w:10s} | test={test_acc:.3f}")

print("weights=distance: vecinos mas cercanos tienen mas peso en el voto")
print("Reduce el efecto de 'suavizado' de k grande")

# ============================================================
# EJEMPLO 3: p=1 (Manhattan) vs p=2 (Euclidean)
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 3: Manhattan (p=1) vs Euclidean (p=2)")
print("="*60)

for p in [1, 2, 3]:
    knn_p = KNeighborsClassifier(n_neighbors=5, p=p)
    knn_p.fit(X_train_s, y_train)
    test_acc = (knn_p.predict(X_test_s) == y_test).mean()
    nombre = 'Manhattan' if p == 1 else ('Euclidean' if p == 2 else f'Minkowski p={p}')
    print(f"{nombre:15s} | p={p} | test={test_acc:.3f}")

print("Manhattan es mas robusto en espacios de alta dimension")
print("Euclidean es la metrica por defecto y mas comun")

# ============================================================
# EJEMPLO 4: n_neighbors — comparar k=1 a k=20 (curva de validacion)
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 4: Curva de validacion — k de 1 a 20")
print("="*60)

ks = range(1, 21)
train_scores = []
test_scores = []

for k in ks:
    knn_k = KNeighborsClassifier(n_neighbors=k)
    knn_k.fit(X_train_s, y_train)
    train_scores.append((knn_k.predict(X_train_s) == y_train).mean())
    test_scores.append((knn_k.predict(X_test_s) == y_test).mean())

best_k = ks[np.argmax(test_scores)]
print(f"Mejor k: {best_k} (test accuracy={max(test_scores):.3f})")
print(f"\nk=1: train={train_scores[0]:.3f}, test={test_scores[0]:.3f}")
print(f"k={best_k}: train={train_scores[best_k-1]:.3f}, test={test_scores[best_k-1]:.3f}")
print(f"k=20: train={train_scores[-1]:.3f}, test={test_scores[-1]:.3f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ks, train_scores, 'b-o', label='Train')
ax.plot(ks, test_scores, 'r-o', label='Test')
ax.axvline(best_k, color='g', linestyle='--', label=f'Mejor k={best_k}')
ax.set_xlabel('k (n_neighbors)')
ax.set_ylabel('Accuracy')
ax.set_title('Curva de validacion: k vs accuracy')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig('I20_k_curve.png', dpi=100)
plt.close()
print("Grafico guardado: I20_k_curve.png")

# ============================================================
# EJEMPLO 5: KNeighborsRegressor — predecir cantidad vendida
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 5: KNeighborsRegressor — predecir cantidad vendida")
print("="*60)

n_reg = 400
df_reg = pd.DataFrame({
    'precio': np.random.uniform(10, 200, n_reg),
    'stock': np.random.randint(0, 300, n_reg),
    'promocion': np.random.choice([0, 1], n_reg, p=[0.7, 0.3]),
    'temp': np.random.choice([0, 1], n_reg, p=[0.6, 0.4]),
    'clientes_diarios': np.random.poisson(40, n_reg)
})
df_reg['cantidad'] = (
    20 - 0.08*df_reg['precio'] + 0.03*df_reg['stock']
    + 12*df_reg['promocion'] + 8*df_reg['temp']
    + 0.2*df_reg['clientes_diarios'] + np.random.normal(0, 10, n_reg)
).clip(0)

Xr = df_reg.drop('cantidad', axis=1)
yr = df_reg['cantidad']
Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.3, random_state=42)
scaler_r = StandardScaler()
Xr_train_s = scaler_r.fit_transform(Xr_train)
Xr_test_s = scaler_r.transform(Xr_test)

for k in [3, 5, 7, 11, 15]:
    knn_r = KNeighborsRegressor(n_neighbors=k)
    knn_r.fit(Xr_train_s, yr_train)
    yp = knn_r.predict(Xr_test_s)
    r2 = r2_score(yr_test, yp)
    mae = mean_absolute_error(yr_test, yp)
    print(f"k={k:2d} | R2={r2:.3f} | MAE={mae:.2f}")

# ============================================================
# EJEMPLO 6: NearestNeighbors — encontrar los 5 productos mas similares
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 6: NearestNeighbors — productos similares")
print("="*60)

X_sim = X.values
scaler_sim = StandardScaler()
X_sim_s = scaler_sim.fit_transform(X_sim)

nn = NearestNeighbors(n_neighbors=5, metric='euclidean')
nn.fit(X_sim_s)

# Producto de consulta (el primero)
query_idx = 0
query = X_sim_s[query_idx].reshape(1, -1)
distancias, indices = nn.kneighbors(query)

print(f"Producto de consulta (idx={query_idx}):")
print(f"  {dict(X.iloc[query_idx])}")
print(f"\nLos 5 productos mas similares:")
for i, (dist, idx) in enumerate(zip(distancias[0], indices[0])):
    if idx == query_idx:
        print(f"  #{i+1}: Producto {idx} (el mismo) — distancia=0")
    else:
        print(f"  #{i+1}: Producto {idx} — distancia={dist:.3f}")
        print(f"         {dict(X.iloc[idx])}")

# ============================================================
# EJEMPLO 7: NearestNeighbors con metric='cosine' para descripciones
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 7: Metrica cosine para similitud de descripciones")
print("="*60)

# Simular features de descripcion (TF-IDF-like)
np.random.seed(42)
n_prod = 100
n_words = 50
desc_features = np.random.dirichlet(np.ones(n_words)*0.1, n_prod)
# Hacer algunos productos similares
desc_features[:5] += np.random.normal(0, 0.01, (5, n_words))
desc_features[5:10] += np.random.normal(0, 0.01, (5, n_words))
desc_features = np.abs(desc_features)
desc_features = desc_features / desc_features.sum(axis=1, keepdims=True)

productos = [f"Prod_{i}" for i in range(n_prod)]

nn_cos = NearestNeighbors(n_neighbors=5, metric='cosine')
nn_cos.fit(desc_features)

print("Productos similares por descripcion (cosine):")
for query_p in [0, 5]:
    dist, idx = nn_cos.kneighbors(desc_features[query_p].reshape(1, -1))
    print(f"\nProducto consulta: {productos[query_p]}")
    for i, (d, ix) in enumerate(zip(dist[0], idx[0])):
        sim = 1 - d
        print(f"  #{i+1}: {productos[ix]} — cosine_sim={sim:.4f}")

# ============================================================
# EJEMPLO 8: RadiusNeighborsClassifier — clasificacion por radio fijo
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 8: RadiusNeighborsClassifier")
print("="*60)

for radius in [0.5, 1.0, 1.5, 2.0]:
    rnc = RadiusNeighborsClassifier(radius=radius, weights='distance')
    rnc.fit(X_train_s, y_train)
    yp = rnc.predict(X_test_s)
    acc = (yp == y_test).mean()
    # Contar outliers (puntos sin vecinos en el radio)
    n_outliers = 0
    dist, idx = rnc.radius_neighbors(X_test_s, return_distance=True)
    for i, neighbors in enumerate(idx):
        if len(neighbors) == 0:
            n_outliers += 1
    print(f"radius={radius:.1f} | test={acc:.3f} | outliers={n_outliers}/{len(X_test_s)}")

print("RadiusNeighbors: radio fijo en lugar de k vecinos")
print("Puntos sin vecinos en el radio quedan sin clasificar")

# ============================================================
# EJEMPLO 9: NearestCentroid — clasificar al centroide mas cercano
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 9: NearestCentroid")
print("="*60)

nc = NearestCentroid()
nc.fit(X_train_s, y_train)
yp_nc = nc.predict(X_test_s)

print(f"NearestCentroid test accuracy: {(yp_nc == y_test).mean():.3f}")
print(f"Centroides de cada clase:")
for i, centroid in enumerate(nc.centroids_):
    print(f"  Clase {i}: {np.round(centroid, 3)}")

print("\nNearestCentroid: simple y rapido, pero asume clases esfericas")

# ============================================================
# EJEMPLO 10: NeighborhoodComponentsAnalysis — aprendizaje de metrica
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 10: NeighborhoodComponentsAnalysis (NCA)")
print("="*60)

# NCA aprende una transformacion lineal para mejorar KNN
from sklearn.pipeline import make_pipeline

nca_pipe = make_pipeline(
    StandardScaler(),
    NeighborhoodComponentsAnalysis(n_components=3, random_state=42),
    KNeighborsClassifier(n_neighbors=5)
)
nca_pipe.fit(X_train, y_train)
acc_nca = (nca_pipe.predict(X_test) == y_test).mean()

# KNN sin NCA
knn_base = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5)
)
knn_base.fit(X_train, y_train)
acc_base = (knn_base.predict(X_test) == y_test).mean()

print(f"KNN sin NCA:       {acc_base:.3f}")
print(f"KNN con NCA (dim=3): {acc_nca:.3f}")
print(f"Mejora: {acc_nca - acc_base:+.3f}")

# ============================================================
# EJEMPLO 11: Algoritmo ball_tree vs kd_tree vs brute (comparar velocidad)
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 11: Comparacion de algoritmos de busqueda")
print("="*60)

# Dataset mas grande
X_big, y_big = make_classification(
    n_samples=5000, n_features=20, n_informative=10, random_state=42
)
Xb_train, Xb_test, yb_train, yb_test = train_test_split(X_big, y_big, test_size=0.3, random_state=42)
scaler_b = StandardScaler()
Xb_train_s = scaler_b.fit_transform(Xb_train)

for algo in ['brute', 'kd_tree', 'ball_tree']:
    start = time.time()
    knn_a = KNeighborsClassifier(n_neighbors=5, algorithm=algo, leaf_size=30)
    knn_a.fit(Xb_train_s, yb_train)
    fit_time = time.time() - start

    start = time.time()
    yp = knn_a.predict(scaler_b.transform(Xb_test))
    pred_time = time.time() - start

    acc = (yp == yb_test).mean()
    print(f"{algo:10s} | fit={fit_time:.3f}s | predict={pred_time:.3f}s | acc={acc:.3f}")

print("brute: O(n*d) predict, bueno para pocos datos")
print("kd_tree: bueno para baja dimension (d<20)")
print("ball_tree: bueno para alta dimension")

# ============================================================
# EJEMPLO 12: leaf_size — optimizacion de arbol
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 12: Efecto de leaf_size")
print("="*60)

for leaf_size in [5, 20, 50, 100]:
    start = time.time()
    knn_l = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree', leaf_size=leaf_size)
    knn_l.fit(Xb_train_s, yb_train)
    fit_time = time.time() - start
    start = time.time()
    acc = (knn_l.predict(scaler_b.transform(Xb_test)) == yb_test).mean()
    pred_time = time.time() - start
    print(f"leaf_size={leaf_size:3d} | fit={fit_time:.3f}s | predict={pred_time:.3f}s | acc={acc:.3f}")

print("leaf_size pequeno -> arbol mas profundo, prediccion mas rapida")
print("leaf_size grande -> arbol mas plano, menos memoria")

# ============================================================
# EJEMPLO 13: Estandarizar datos antes de KNN (IMPORTANTE)
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 13: Importancia de estandarizar en KNN")
print("="*60)

X_scale = pd.DataFrame({
    'precio': np.random.uniform(10, 500, 300),
    'unidades': np.random.randint(1, 10000, 300),
    'tasa': np.random.uniform(0.01, 0.99, 300),
    'puntaje': np.random.uniform(0, 1000, 300)
})
y_scale = (X_scale['precio']*0.01 + X_scale['unidades']*0.0001
           - X_scale['tasa']*0.5 + np.random.normal(0, 0.3, 300) > 0.5).astype(int)

Xs_train, Xs_test, ys_train, ys_test = train_test_split(
    X_scale, y_scale, test_size=0.3, random_state=42
)

knn_no = KNeighborsClassifier(n_neighbors=5)
knn_no.fit(Xs_train, ys_train)
acc_no = (knn_no.predict(Xs_test) == ys_test).mean()

scaler_k = StandardScaler()
Xs_train_s = scaler_k.fit_transform(Xs_train)
Xs_test_s = scaler_k.transform(Xs_test)
knn_sc = KNeighborsClassifier(n_neighbors=5)
knn_sc.fit(Xs_train_s, ys_train)
acc_sc = (knn_sc.predict(Xs_test_s) == ys_test).mean()

print(f"Sin escalar:  {acc_no:.3f}")
print(f"Con escalado: {acc_sc:.3f}")
print(f"Diferencia:   {acc_sc-acc_no:+.3f}")
print("CRITICO: KNN usa distancias. Features con escalas mayores dominan")

# ============================================================
# EJEMPLO 14: Pipeline — StandardScaler + KNeighborsClassifier
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 14: Pipeline StandardScaler + KNN")
print("="*60)

pipeline_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])
pipeline_knn.fit(X_train, y_train)
print(f"Pipeline accuracy: {(pipeline_knn.predict(X_test)==y_test).mean():.3f}")
print(pipeline_knn)

# ============================================================
# EJEMPLO 15: GridSearchCV — busqueda de n_neighbors, weights, p
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 15: GridSearchCV para KNN")
print("="*60)

param_grid_knn = {
    'knn__n_neighbors': [3, 5, 7, 11, 15],
    'knn__weights': ['uniform', 'distance'],
    'knn__p': [1, 2],
    'knn__metric': ['euclidean', 'manhattan']
}

pipe_gs_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

gs_knn = GridSearchCV(
    pipe_gs_knn, param_grid_knn, cv=5, scoring='accuracy', n_jobs=-1
)
gs_knn.fit(X_train, y_train)

print(f"Mejores parametros: {gs_knn.best_params_}")
print(f"Mejor score CV: {gs_knn.best_score_:.4f}")
print(f"Test accuracy: {(gs_knn.predict(X_test)==y_test).mean():.4f}")

results_knn = pd.DataFrame(gs_knn.cv_results_)
top5_knn = results_knn.sort_values('rank_test_score').head(5)
print("\nTop 5 combinaciones:")
for i, row in top5_knn.iterrows():
    print(f"  {row['params']} | mean={row['mean_test_score']:.4f} | std={row['std_test_score']:.4f}")

# ============================================================
# EJEMPLO 16: Visualizar limite de decision 2D para KNN
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 16: Limite de decision 2D para KNN")
print("="*60)

X_2d = df[['precio', 'margen']].values
y_2d = df['alta_rotacion'].values
X_2d_train, X_2d_test, y_2d_train, y_2d_test = train_test_split(
    X_2d, y_2d, test_size=0.3, random_state=42
)
scaler_2d = StandardScaler()
X_2d_train_s = scaler_2d.fit_transform(X_2d_train)
X_2d_test_s = scaler_2d.transform(X_2d_test)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, k in enumerate([1, 5, 15]):
    knn_viz = KNeighborsClassifier(n_neighbors=k)
    knn_viz.fit(X_2d_train_s, y_2d_train)

    xx, yy = np.meshgrid(
        np.linspace(X_2d_train_s[:, 0].min()-0.5, X_2d_train_s[:, 0].max()+0.5, 200),
        np.linspace(X_2d_train_s[:, 1].min()-0.5, X_2d_train_s[:, 1].max()+0.5, 200)
    )
    Z = knn_viz.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    axes[i].contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    axes[i].scatter(X_2d_test_s[:, 0], X_2d_test_s[:, 1], c=y_2d_test,
                    cmap='coolwarm', edgecolors='k', linewidth=0.5)
    axes[i].set_xlabel('Precio (std)')
    axes[i].set_ylabel('Margen (std)')
    axes[i].set_title(f'KNN k={k}')
plt.tight_layout()
plt.savefig('I20_knn_boundaries.png', dpi=100)
plt.close()
print("Grafico guardado: I20_knn_boundaries.png")
print("k=1: frontera muy detallada (overfitting). k=15: frontera mas suave")

# ============================================================
# EJEMPLO 17: Comparar KNN vs LogisticRegression vs SVM
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 17: Comparacion KNN vs LogisticRegression vs SVM")
print("="*60)

modelos_comp = {
    'KNN (k=5)': Pipeline([('scaler', StandardScaler()),
                           ('clf', KNeighborsClassifier(n_neighbors=5))]),
    'KNN (k=15)': Pipeline([('scaler', StandardScaler()),
                            ('clf', KNeighborsClassifier(n_neighbors=15))]),
    'LogisticRegression': Pipeline([('scaler', StandardScaler()),
                                    ('clf', LogisticRegression(max_iter=1000, random_state=42))]),
    'SVM (rbf)': Pipeline([('scaler', StandardScaler()),
                           ('clf', SVC(kernel='rbf', random_state=42))]),
}

for nombre, pipe in modelos_comp.items():
    pipe.fit(X_train, y_train)
    yp = pipe.predict(X_test)
    train_acc = (pipe.predict(X_train) == y_train).mean()
    test_acc = (yp == y_test).mean()
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5)
    print(f"{nombre:25s} | train={train_acc:.3f} test={test_acc:.3f} "
          f"| CV={cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

print("KNN es bueno para fronteras complejas pero sensible a k y scaling")
print("SVM suele generalizar mejor con pocos datos")
print("LogReg es el mas simple e interpretable")

# ============================================================
# EJEMPLO 18: Integrador — sistema de recomendacion simple con NN
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 18: Sistema de recomendacion simple con NearestNeighbors")
print("="*60)

# Catalogo de productos
n_cat = 200
np.random.seed(42)
catalogo = pd.DataFrame({
    'precio': np.random.uniform(10, 500, n_cat),
    'categoria': np.random.choice(['electronica', 'ropa', 'alimentos', 'hogar'], n_cat),
    'rating': np.random.uniform(1, 5, n_cat),
    'ventas_mensuales': np.random.poisson(100, n_cat),
    'descuento': np.random.choice([0, 5, 10, 15, 20, 30], n_cat, p=[0.5, 0.2, 0.1, 0.1, 0.05, 0.05]),
})

# Codificar categoria
cat_dummies = pd.get_dummies(catalogo['categoria'], prefix='cat')
X_cat = pd.concat([catalogo[['precio', 'rating', 'ventas_mensuales', 'descuento']],
                   cat_dummies], axis=1)

scaler_cat = StandardScaler()
X_cat_s = scaler_cat.fit_transform(X_cat)

# Sistema de recomendacion
rec_system = NearestNeighbors(n_neighbors=6, metric='euclidean')
rec_system.fit(X_cat_s)

# Recomendar para un producto de consulta
query_prod_idx = 10
query_prod = catalogo.iloc[query_prod_idx]
dist, indices = rec_system.kneighbors(X_cat_s[query_prod_idx].reshape(1, -1))

print(f"Producto de consulta: {query_prod_idx}")
print(f"  Categoria: {query_prod['categoria']}, Precio: ${query_prod['precio']:.0f}, "
      f"Rating: {query_prod['rating']:.1f}, Ventas: {query_prod['ventas_mensuales']:.0f}")
print(f"\nProductos recomendados:")
for i, (d, idx) in enumerate(zip(dist[0], indices[0])):
    if idx == query_prod_idx:
        continue
    prod = catalogo.iloc[idx]
    print(f"  #{i}: {prod['categoria']:12s} | ${prod['precio']:5.0f} | "
          f"Rating: {prod['rating']:.1f} | Desc: {prod['descuento']:2d}% | "
          f"Dist: {d:.3f}")

# Funcion de recomendacion
def recomendar(producto_id, n_recom=5):
    dist, indices = rec_system.kneighbors(X_cat_s[producto_id].reshape(1, -1))
    print(f"\nRecomendaciones para producto #{producto_id}:")
    for i in range(1, min(n_recom+1, len(indices[0]))):
        idx = indices[0][i]
        d = dist[0][i]
        prod = catalogo.iloc[idx]
        print(f"  -> #{idx}: {prod['categoria']:12s} ${prod['precio']:5.0f} "
              f"(rating={prod['rating']:.1f}, sim={1-d:.3f})")

recomendar(5)
recomendar(50)

print("\n=== FIN EJEMPLOS KNN Y VECINOS ===")
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

*2. Ejemplos Prácticos con sklearn.*

1. Configuracion inicial
2. ============================================================
3. EJEMPLO 1: KNeighborsClassifier — clasificar productos (k=3,5,7)
4. ============================================================
5. ============================================================
6. EJEMPLO 2: weights='distance' — ponderar por distancia inversa
7. ============================================================
8. ============================================================

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

## 3. Ejercicios Propuestos

1. **k óptimo con validación cruzada**: Usa `cross_val_score` para encontrar el k óptimo (1-30) en el dataset de productos con 10-fold CV. Grafica media y desviación estándar.

2. **Pesos y métricas**: En el dataset de cantidad vendida (Ejemplo 5), prueba las 4 combinaciones de weights (uniform/distance) × p (1/2). ¿Cuál da mejor MAE?

3. **Detección de outliers con NearestNeighbors**: Calcula la distancia promedio a los 5 vecinos más cercanos para cada punto. Los puntos con mayor distancia promedio son potenciales outliers. Encuentra el top 5% de outliers en el dataset.

4. **KNN con Cosine metric para texto**: Crea 50 descripciones de productos con 20 palabras clave. Usa NearestNeighbors(metric='cosine') para agrupar productos similares. ¿La metrica cosine captura mejor la similitud semántica que euclidean?

5. **RadiusNeighbors con diferentes radios**: En el dataset de productos, varía radius de 0.1 a 3.0. ¿Qué porcentaje de puntos queda sin clasificar en cada caso? ¿Cuál es el mejor radio?

6. **NCA para reducción de dimensionalidad**: Usa NCA para reducir a 2 dimensiones y visualiza el dataset. ¿Las clases se separan mejor que con PCA?

7. **KNN en datos no estandarizados**: Demuestra cuantitativamente cómo KNN empeora sin escalado, usando features con escalas muy diferentes (precio en $ vs unidades en miles).

8. **Sistema de recomendación mejorado**: Al recomendador del Ejemplo 18, añade ponderación: que el rating tenga el doble de peso que las otras features. ¿Cómo cambian las recomendaciones?

---

## 4. Resumen

- **KNN** clasifica por voto mayoritario de los k vecinos más cercanos
- **k pequeño** → modelo complejo, overfitting; **k grande** → modelo simple, underfitting
- **weights='distance'** pondera vecinos por proximidad
- **p=1** (Manhattan) es más robusto en alta dimensión; **p=2** (Euclidean) es el default
- **NearestNeighbors** encuentra vecinos sin supervisión (recomendación, búsqueda)
- **RadiusNeighbors** clasifica por radio fijo (útil con densidad variable)
- **NearestCentroid** es rápido pero asume clusters esféricos
- **NCA** aprende una métrica de distancia para mejorar KNN
- **Siempre escalar** datos antes de KNN (critical)
- **algorithm** afecta velocidad: brute (pocos datos), kd_tree (baja dim), ball_tree (alta dim)

### Cuándo usar KNN
- El **límite de decisión** es muy irregular o complejo
- Tienes **pocos datos** pero muchas features relevantes
- Necesitas un sistema de **recomendación** o búsqueda de similitud
- Quieres un baseline **no paramétrico** rápido

### Limitaciones
- **Maldición de la dimensionalidad**: KNN funciona mal con muchas features
- Computacionalmente costoso en **predicción** (O(n*d) por consulta)
- Muy sensible a **escalado** y features irrelevantes
- No da **probabilidades** bien calibradas
- Almacena **todos los datos** de entrenamiento (alta memoria)
