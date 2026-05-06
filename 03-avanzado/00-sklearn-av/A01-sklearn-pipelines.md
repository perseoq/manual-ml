# A01: Pipelines en scikit-learn para Ventas, Compras e Inventarios

## Introducción Teórica

Los **Pipelines** de scikit-learn permiten encadenar múltiples pasos de transformación y un estimador final en un solo objeto. Esto garantiza que las mismas transformaciones aplicadas durante el entrenamiento se apliquen exactamente igual durante la predicción, eliminando el riesgo de data leakage.

### Componentes principales:

1. **Pipeline**: Secuencia de pasos (transformaciones + estimador final). Cada paso es un tuple `(nombre, transformador)` excepto el último que es un estimador.
2. **make_pipeline**: Función simplificada que asigna nombres automáticamente.
3. **ColumnTransformer**: Aplica diferentes transformaciones a diferentes columnas.
4. **FeatureUnion**: Concatena resultados de múltiples transformaciones paralelas.
5. **FunctionTransformer**: Convierte funciones arbitrarias en transformadores de sklearn.

### Ventajas en negocio:
- **Ventas**: Pipeline completo: limpiar precios → escalar montos → predecir si una venta se concretará.
- **Compras**: Pipeline: imputar proveedores faltantes → codificar categorías → predecir lead time.
- **Inventarios**: Pipeline: escalar rotación → PCA de productos → clasificar ABC de inventario.

---

## Ejemplos

### Ejemplo 1: Pipeline básico — StandardScaler → LogisticRegression para clasificar ventas exitosas

```python
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Datos simulados de ventas
np.random.seed(42)
n = 1000
df_ventas = pd.DataFrame({
    'monto': np.random.exponential(500, n),
    'descuento': np.random.uniform(0, 0.3, n),
    'dias_ultima_compra': np.random.randint(1, 365, n),
    'exito': np.random.binomial(1, 0.7, n)
})

X = df_ventas[['monto', 'descuento', 'dias_ultima_compra']]
y = df_ventas['exito']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression())
])

pipe.fit(X_train, y_train)
print(f"Accuracy: {pipe.score(X_test, y_test):.3f}")
print(f"Coeficientes: {pipe.named_steps['clf'].coef_}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: Pipeline básico — StandardScaler → LogisticRegression para clasificar ventas exitosas.*

1. Datos simulados de ventas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: Pipeline con PCA y RandomForest para segmentar productos por rotación

```python
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# Datos de rotación de inventarios
df_inv = pd.DataFrame({
    'rotacion_mensual': np.random.exponential(10, 500),
    'costo_unitario': np.random.uniform(5, 500, 500),
    'margen': np.random.uniform(0.1, 0.8, 500),
    'stock_seguridad': np.random.randint(10, 500, 500),
    'clase_abc': np.random.choice(['A', 'B', 'C'], 500)
})

X2 = df_inv[['rotacion_mensual', 'costo_unitario', 'margen', 'stock_seguridad']]
y2 = df_inv['clase_abc']

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

pipe2 = Pipeline([
    ('scaler', MinMaxScaler()),
    ('pca', PCA(n_components=2)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipe2.fit(X2_train, y2_train)
print(f"Accuracy con PCA: {pipe2.score(X2_test, y2_test):.3f}")
print(f"Componentes PCA: {pipe2.named_steps['pca'].explained_variance_ratio_}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: Pipeline con PCA y RandomForest para segmentar productos por rotación.*

1. Datos de rotación de inventarios

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: Pipeline con predict_proba para estimar probabilidad de compra

```python
# Mismos datos del ejemplo 1
pipe3 = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression())
])

pipe3.fit(X_train, y_train)
probas = pipe3.predict_proba(X_test)[:, 1]

print(f"Primeras 5 probabilidades de éxito: {probas[:5].round(3)}")
print(f"Predicciones: {pipe3.predict(X_test[:5])}")

# Decisión de negocio: solo enviar oferta si prob > 0.8
ofertas_automaticas = (probas > 0.8).sum()
print(f"Ofertas automáticas recomendadas: {ofertas_automaticas} de {len(probas)}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: Pipeline con predict_proba para estimar probabilidad de compra.*

1. Mismos datos del ejemplo 1
2. Decisión de negocio: solo enviar oferta si prob > 0.8

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: make_pipeline simplificado para predicción de lead time

```python
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

# Datos de compras
df_compras = pd.DataFrame({
    'dias_entrega_hist': np.random.randint(1, 60, 300),
    'costo_envio': np.random.uniform(10, 200, 300),
    'volumen': np.random.exponential(100, 300),
    'lead_time': np.random.randint(3, 45, 300)
})

Xc = df_compras[['dias_entrega_hist', 'costo_envio', 'volumen']]
yc = df_compras['lead_time']

Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.2, random_state=42)

pipe4 = make_pipeline(
    SimpleImputer(strategy='median'),
    RandomForestRegressor(n_estimators=100, random_state=42)
)

pipe4.fit(Xc_train, yc_train)
print(f"R²: {pipe4.score(Xc_test, yc_test):.3f}")
print(f"Pasos: {pipe4.steps}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: make_pipeline simplificado para predicción de lead time.*

1. Datos de compras

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: ColumnTransformer — numéricas escaladas, categóricas codificadas

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

df_mixto = pd.DataFrame({
    'monto': np.random.exponential(500, 1000),
    'dias_ultima_compra': np.random.randint(1, 365, 1000),
    'categoria': np.random.choice(['electronica', 'ropa', 'hogar', 'deportes'], 1000),
    'metodo_pago': np.random.choice(['tarjeta', 'efectivo', 'transferencia'], 1000),
    'compro': np.random.binomial(1, 0.6, 1000)
})

X_mix = df_mixto.drop('compro', axis=1)
y_mix = df_mixto['compro']

Xm_train, Xm_test, ym_train, ym_test = train_test_split(X_mix, y_mix, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['monto', 'dias_ultima_compra']),
    ('cat', OneHotEncoder(drop='first'), ['categoria', 'metodo_pago'])
])

pipe5 = Pipeline([
    ('prep', preprocessor),
    ('clf', LogisticRegression(max_iter=1000))
])

pipe5.fit(Xm_train, ym_train)
print(f"Accuracy con ColumnTransformer: {pipe5.score(Xm_test, ym_test):.3f}")
print(f"Features de salida: {pipe5.named_steps['prep'].get_feature_names_out()}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: ColumnTransformer — numéricas escaladas, categóricas codificadas.*

1. `from sklearn.compose import ColumnTransformer` — Importa las librerías necesarias para el análisis.
2. `from sklearn.preprocessing import OneHotEncoder` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: ColumnTransformer con remainder='passthrough'

```python
# Mantener columnas no transformadas intactas
preprocessor6 = ColumnTransformer([
    ('num', StandardScaler(), ['monto', 'dias_ultima_compra']),
], remainder='passthrough')

pipe6 = Pipeline([
    ('prep', preprocessor6),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipe6.fit(Xm_train, ym_train)
print(f"Accuracy con passthrough: {pipe6.score(Xm_test, ym_test):.3f}")

# Ver cuántas features resultan (2 escaladas + 2 passthrough)
X_transformed = preprocessor6.fit_transform(Xm_train)
print(f"Shape transformado: {X_transformed.shape}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: ColumnTransformer con remainder='passthrough'.*

1. Mantener columnas no transformadas intactas
2. Ver cuántas features resultan (2 escaladas + 2 passthrough)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: ColumnTransformer con remainder que estima (SimpleImputer)

```python
from sklearn.impute import SimpleImputer

# Datos con valores faltantes
df_faltantes = pd.DataFrame({
    'monto': np.random.exponential(500, 1000),
    'descuento': np.where(np.random.random(1000) < 0.1, np.nan, np.random.uniform(0, 0.3, 1000)),
    'categoria': np.where(np.random.random(1000) < 0.15, np.nan, np.random.choice(['A', 'B', 'C'], 1000)),
    'compro': np.random.binomial(1, 0.6, 1000)
})

X_falt = df_faltantes.drop('compro', axis=1)
y_falt = df_faltantes['compro']

preprocessor7 = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), ['monto', 'descuento']),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first'))
    ]), ['categoria'])
])

pipe7 = Pipeline([
    ('prep', preprocessor7),
    ('clf', LogisticRegression(max_iter=1000))
])

pipe7.fit(X_falt, y_falt)
print(f"Accuracy con imputación: {pipe7.score(X_falt, y_falt):.3f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: ColumnTransformer con remainder que estima (SimpleImputer).*

1. Datos con valores faltantes

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: ColumnTransformer con verbose_feature_names_out=False

```python
preprocessor8 = ColumnTransformer([
    ('num', StandardScaler(), ['monto', 'dias_ultima_compra']),
    ('cat', OneHotEncoder(drop='first'), ['categoria', 'metodo_pago'])
], verbose_feature_names_out=False)

pipe8 = Pipeline([
    ('prep', preprocessor8),
    ('clf', LogisticRegression(max_iter=1000))
])

pipe8.fit(Xm_train, ym_train)
print(f"Features (sin prefijo): {pipe8.named_steps['prep'].get_feature_names_out()}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: ColumnTransformer con verbose_feature_names_out=False.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: FeatureUnion — concatenar PCA + SelectKBest

```python
from sklearn.pipeline import FeatureUnion
from sklearn.feature_selection import SelectKBest, f_classif

# Datos de ventas con muchas features
df_muchas = pd.DataFrame({
    f'feature_{i}': np.random.randn(500) for i in range(20)
})
df_muchas['target'] = (df_muchas['feature_0'] + df_muchas['feature_5'] + np.random.randn(500) * 0.5 > 0).astype(int)

X_much = df_muchas.drop('target', axis=1)
y_much = df_muchas['target']

union = FeatureUnion([
    ('pca', PCA(n_components=3)),
    ('kbest', SelectKBest(score_func=f_classif, k=5))
])

pipe9 = Pipeline([
    ('union', union),
    ('clf', LogisticRegression())
])

pipe9.fit(X_much, y_much)
print(f"Accuracy con FeatureUnion: {pipe9.score(X_much, y_much):.3f}")
print(f"Dim total: {pipe9.named_steps['union'].transform(X_much).shape[1]} (PCA:3 + KBest:5)")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: FeatureUnion — concatenar PCA + SelectKBest.*

1. Datos de ventas con muchas features

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: Pipeline + ColumnTransformer + FeatureUnion completo

```python
# Dataset mixto completo
df_completo = pd.DataFrame({
    'monto': np.random.exponential(500, 1000),
    'frecuencia': np.random.randint(1, 30, 1000),
    'recencia': np.random.randint(1, 365, 1000),
    'categoria': np.random.choice(['electronica', 'ropa', 'hogar'], 1000),
    'zona': np.random.choice(['norte', 'sur', 'este', 'oeste'], 1000),
    'target': np.random.binomial(1, 0.6, 1000)
})

X_full = df_completo.drop('target', axis=1)
y_full = df_completo['target']

num_features = ['monto', 'frecuencia', 'recencia']
cat_features = ['categoria', 'zona']

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), num_features),
    ('cat', OneHotEncoder(drop='first'), cat_features)
])

feature_union = FeatureUnion([
    ('main', 'passthrough'),
    ('pca', PCA(n_components=2))
])

pipe10 = Pipeline([
    ('preprocessor', preprocessor),
    ('union', feature_union),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipe10.fit(X_full, y_full)
print(f"Score completo: {pipe10.score(X_full, y_full):.3f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: Pipeline + ColumnTransformer + FeatureUnion completo.*

1. Dataset mixto completo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: Pipeline con memoria (cache de transformaciones)

```python
from tempfile import mkdtemp
from shutil import rmtree
import os

cachedir = mkdtemp()

pipe_cache = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=3)),
    ('clf', LogisticRegression(max_iter=1000))
], memory=cachedir)

# Primera ejecución: calcula y cachea las transformaciones
import time
t0 = time.time()
pipe_cache.fit(Xm_train, ym_train)
t1 = time.time()
print(f"Primer fit: {t1 - t0:.3f}s")

# Segunda ejecución: reusa las transformaciones cacheadas
t0 = time.time()
pipe_cache.fit(Xm_train, ym_train)
t1 = time.time()
print(f"Segundo fit (cache): {t1 - t0:.3f}s")

rmtree(cachedir)
print(f"Memoria cacheada en: {cachedir}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: Pipeline con memoria (cache de transformaciones).*

1. Primera ejecución: calcula y cachea las transformaciones
2. Segunda ejecución: reusa las transformaciones cacheadas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: FunctionTransformer personalizado — log transform de montos

```python
from sklearn.preprocessing import FunctionTransformer

def log_transform(X):
    return np.log1p(np.abs(X))

def log_inverse(X):
    return np.expm1(X)

log_transformer = FunctionTransformer(func=log_transform, inverse_func=log_inverse)

pipe12 = Pipeline([
    ('log', log_transformer),
    ('scaler', StandardScaler()),
    ('clf', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Datos con skew (montos típicos de ventas)
df_skew = pd.DataFrame({
    'monto': np.random.lognormal(mean=6, sigma=1.5, size=1000),
    'unidades': np.random.poisson(5, 1000),
    'demanda': np.random.poisson(10, 1000)
})
y_skew = df_skew['demanda'] + np.random.randn(1000) * 2

pipe12.fit(df_skew[['monto', 'unidades']], y_skew)
print(f"R² con log-transform: {pipe12.score(df_skew[['monto', 'unidades']], y_skew):.3f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: FunctionTransformer personalizado — log transform de montos.*

1. Datos con skew (montos típicos de ventas)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: Pipeline anidado (Pipeline como paso de otro Pipeline)

```python
inner_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2))
])

outer_pipe = Pipeline([
    ('inner', inner_pipe),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

outer_pipe.fit(X2_train, y2_train)
print(f"Pipeline anidado - Accuracy: {outer_pipe.score(X2_test, y2_test):.3f}")
print(f"Pasos del pipeline interno: {outer_pipe.named_steps['inner'].named_steps}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: Pipeline anidado (Pipeline como paso de otro Pipeline).*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: GridSearchCV sobre Pipeline (parámetros con __)

```python
from sklearn.model_selection import GridSearchCV

pipe_gs = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(random_state=42))
])

param_grid = {
    'clf__n_estimators': [50, 100],
    'clf__max_depth': [None, 10, 20],
    'clf__min_samples_split': [2, 5]
}

grid = GridSearchCV(pipe_gs, param_grid, cv=3, scoring='accuracy', verbose=1)
grid.fit(Xm_train, ym_train)

print(f"Mejores params: {grid.best_params_}")
print(f"Mejor score: {grid.best_score_:.3f}")
print(f"Test score: {grid.score(Xm_test, ym_test):.3f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: GridSearchCV sobre Pipeline (parámetros con __).*

1. `from sklearn.model_selection import GridSearchCV` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: RandomizedSearchCV sobre Pipeline

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

pipe_rs = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(random_state=42))
])

param_dist = {
    'clf__n_estimators': randint(50, 300),
    'clf__max_depth': randint(3, 30),
    'clf__min_samples_split': randint(2, 20),
}

random_search = RandomizedSearchCV(pipe_rs, param_dist, n_iter=20, cv=3, scoring='accuracy', random_state=42, verbose=1)
random_search.fit(Xm_train, ym_train)

print(f"RandomizedSearch - Mejores params: {random_search.best_params_}")
print(f"Mejor score: {random_search.best_score_:.3f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: RandomizedSearchCV sobre Pipeline.*

1. `from sklearn.model_selection import RandomizedSearchCV` — Importa las librerías necesarias para el análisis.
2. `from scipy.stats import randint, uniform` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 16: cross_val_score sobre Pipeline completo

```python
from sklearn.model_selection import cross_val_score

pipe_cv = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000))
])

scores = cross_val_score(pipe_cv, X, y, cv=5, scoring='accuracy')
print(f"Scores CV: {scores}")
print(f"Media ± std: {scores.mean():.3f} ± {scores.std():.3f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: cross_val_score sobre Pipeline completo.*

1. `from sklearn.model_selection import cross_val_score` — Importa las librerías necesarias para el análisis.
2. `print(f"Scores CV: {scores}")` — Muestra el resultado por pantalla.
3. `print(f"Media ± std: {scores.mean():.3f} ± {scores.std():.3f}")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 17: Nested cross-validation con Pipeline

```python
from sklearn.model_selection import cross_val_score, KFold

# Inner CV: selección de hiperparámetros
param_grid_inner = {
    'clf__n_estimators': [50, 100],
    'clf__max_depth': [5, 10, None]
}

pipe_nested = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(random_state=42))
])

inner_cv = KFold(3, shuffle=True, random_state=42)
outer_cv = KFold(3, shuffle=True, random_state=42)

grid_inner = GridSearchCV(pipe_nested, param_grid_inner, cv=inner_cv, scoring='accuracy')

# Nested CV: outer loop evalúa el pipeline completo
nested_scores = cross_val_score(grid_inner, X, y, cv=outer_cv, scoring='accuracy')
print(f"Nested CV scores: {nested_scores}")
print(f"Nested CV media ± std: {nested_scores.mean():.3f} ± {nested_scores.std():.3f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: Nested cross-validation con Pipeline.*

1. Inner CV: selección de hiperparámetros
2. Nested CV: outer loop evalúa el pipeline completo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 18: Integrador — Pipeline completo de principio a fin

```python
"""
Pipeline completo para predicción de demanda de productos:
1. Carga y split
2. Preprocesamiento (numérico y categórico)
3. Feature engineering (PCA + selección)
4. Modelo con tuning
5. Evaluación
"""

# 1. Datos
np.random.seed(42)
n = 2000
df_demanda = pd.DataFrame({
    'precio': np.random.uniform(10, 500, n),
    'descuento': np.random.uniform(0, 0.5, n),
    'gasto_marketing': np.random.exponential(1000, n),
    'temporada': np.random.choice(['alta', 'baja', 'media'], n),
    'categoria': np.random.choice(['electronica', 'ropa', 'hogar', 'deportes'], n),
    'region': np.random.choice(['norte', 'sur', 'este', 'oeste'], n),
    'demanda_real': np.random.poisson(50, n)
})

X_dem = df_demanda.drop('demanda_real', axis=1)
y_dem = df_demanda['demanda_real']

X_dem_train, X_dem_test, y_dem_train, y_dem_test = train_test_split(X_dem, y_dem, test_size=0.2, random_state=42)

# 2. Preprocesamiento
num_feats = ['precio', 'descuento', 'gasto_marketing']
cat_feats = ['temporada', 'categoria', 'region']

preprocessor_dem = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), num_feats),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', sparse_output=False))
    ]), cat_feats)
])

# 3. Feature union
feature_union_dem = FeatureUnion([
    ('preprocessed', 'passthrough'),
    ('pca', PCA(n_components=3))
])

# 4. Pipeline completo
pipeline_final = Pipeline([
    ('preprocessor', preprocessor_dem),
    ('features', feature_union_dem),
    ('model', RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42))
])

# 5. Entrenamiento
pipeline_final.fit(X_dem_train, y_dem_train)

# 6. Evaluación
y_pred = pipeline_final.predict(X_dem_test)
from sklearn.metrics import mean_absolute_error, r2_score

print(f"R² test: {r2_score(y_dem_test, y_pred):.3f}")
print(f"MAE test: {mean_absolute_error(y_dem_test, y_pred):.1f}")
print(f"Pasos del pipeline: {[s[0] for s in pipeline_final.steps]}")

# 7. Predicción para nuevo producto
nuevo_producto = pd.DataFrame([{
    'precio': 250, 'descuento': 0.2, 'gasto_marketing': 1500,
    'temporada': 'alta', 'categoria': 'electronica', 'region': 'norte'
}])
print(f"Predicción demanda nuevo producto: {pipeline_final.predict(nuevo_producto)[0]:.0f} unidades")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador — Pipeline completo de principio a fin.*

1. 1. Datos
2. 2. Preprocesamiento
3. 3. Feature union
4. 4. Pipeline completo
5. 5. Entrenamiento
6. 6. Evaluación
7. 7. Predicción para nuevo producto

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejercicios

1. **Pipeline de imputación + escalado + regresión**: Crea un pipeline que impute valores faltantes con la mediana, escale con StandardScaler y entrene un Ridge regressor para predecir el monto de compra. Usa datos simulados con 10% de NaN.

2. **ColumnTransformer con 3 tipos de columnas**: Diseña un ColumnTransformer que: (a) escale las numéricas, (b) one-hot encode las categóricas con drop='first', (c) pase las columnas binarias intactas. Aplica a un dataset de ventas.

3. **FeatureUnion con 3 transformadores**: Crea un FeatureUnion que combine PCA(3), SelectKBest(k=5) y un transformador de polinomios (PolynomialFeatures(degree=2, interaction_only=True)). Evalúa cómo cambia el rendimiento.

4. **Pipeline con cache**: Implementa un pipeline con `memory` que procese 10,000 registros de inventarios. Mide el tiempo con y sin cache para demostrar la mejora.

5. **FunctionTransformer de winsorización**: Crea un FunctionTransformer que recorte outliers al percentil 1 y 99 (winsorización). Incorpóralo en un pipeline de predicción de ventas.

6. **Pipeline anidado con 3 niveles**: Construye un pipeline de 3 niveles: inner (scaler + PCA), middle (FeatureUnion con inner + selectivo), outer (modelo final). Documenta la estructura con `print(pipe)`.

7. **GridSearchCV sobre Pipeline de 4 pasos**: Optimiza un pipeline completo (imputación → escalado → selección → modelo) con GridSearch variando parámetros de al menos 3 pasos diferentes.

8. **Pipeline completo para clasificación ABC**: Diseña un pipeline que tome features de producto (rotación, margen, costo, lead time) y clasifique en categoría A/B/C usando ColumnTransformer + FeatureUnion + RandomForest. Incluye evaluación con nested CV.

---

## Resumen

- **Pipeline** encadena transformaciones + modelo, evitando data leakage.
- **ColumnTransformer** aplica transformaciones distintas por tipo de columna (numéricas, categóricas, texto).
- **FeatureUnion** combina múltiples transformaciones en paralelo.
- **FunctionTransformer** envuelve funciones Python como transformadores sklearn.
- **memory** cachea transformaciones intermedias para acelerar reentrenamientos.
- La sintaxis `__` permite acceder a parámetros de pasos internos desde GridSearchCV.
- Pipelines son estimadores completos: sirven con `.fit()`, `.predict()`, `.score()` y son compatibles con `cross_val_score()` y `GridSearchCV()`.
- En negocio de ventas/compras/inventarios, los pipelines garantizan consistencia entre desarrollo y producción.
