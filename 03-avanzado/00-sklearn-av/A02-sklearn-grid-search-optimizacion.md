# A02: Grid Search y Optimización de Hiperparámetros para Ventas, Compras e Inventarios

## Introducción Teórica

La **optimización de hiperparámetros** es el proceso de encontrar la combinación de parámetros que maximiza el rendimiento de un modelo. scikit-learn ofrece múltiples estrategias:

### Estrategias de búsqueda:

1. **GridSearchCV**: Búsqueda exhaustiva sobre una grilla de parámetros. Garantiza encontrar la mejor combinación dentro de la grilla, pero es costoso computacionalmente.
2. **RandomizedSearchCV**: Búsqueda aleatoria sobre distribuciones de parámetros. Más eficiente que GridSearch cuando hay muchos parámetros o algunos son poco influyentes.
3. **HalvingGridSearchCV**: Búsqueda sucesiva que descarta combinaciones malas rápidamente usando menos recursos en iteraciones tempranas.
4. **HalvingRandomSearchCV**: Versión aleatoria de HalvingSearch.

### Componentes clave:
- **param_grid / param_distributions**: Diccionario con parámetros a probar.
- **cv**: Estrategia de validación cruzada.
- **scoring**: Métrica de evaluación.
- **refit**: Reentrenar automáticamente con los mejores parámetros en todo el dataset.
- **return_train_score**: Comparar train vs test para detectar overfitting.

### Aplicación en negocio:
- **Ventas**: Optimizar umbrales de descuento y modelo para predecir conversión.
- **Compras**: Encontrar mejor configuración para predecir lead time de proveedores.
- **Inventarios**: Ajustar modelo de clasificación ABC con parámetros óptimos.

---

## Ejemplos

### Ejemplo 1: GridSearchCV básico para RandomForest en ventas

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n = 1000
df = pd.DataFrame({
    'monto': np.random.exponential(500, n),
    'frecuencia': np.random.randint(1, 30, n),
    'recencia': np.random.randint(1, 365, n),
    'conversion': np.random.binomial(1, 0.6, n)
})

X = df[['monto', 'frecuencia', 'recencia']]
y = df['conversion']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid,
                    cv=3, scoring='accuracy', verbose=0)
grid.fit(X_train_s, y_train)

print(f"Mejores parámetros: {grid.best_params_}")
print(f"Mejor accuracy (CV): {grid.best_score_:.3f}")
print(f"Accuracy en test: {grid.score(X_test_s, y_test):.3f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: GridSearchCV básico para RandomForest en ventas.*

1. `import pandas as pd` — Importa las librerías necesarias para el análisis.
2. `import numpy as np` — Importa las librerías necesarias para el análisis.
3. `from sklearn.ensemble import RandomForestClassifier` — Importa las librerías necesarias para el análisis.
4. `from sklearn.model_selection import GridSearchCV, train_test_split` — Importa las librerías necesarias para el análisis.
5. `from sklearn.preprocessing import StandardScaler` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: GridSearchCV con scoring='f1' para clases desbalanceadas

```python
# Dataset desbalanceado: solo 15% de conversiones
df_desb = pd.DataFrame({
    'monto': np.random.exponential(500, 2000),
    'descuento': np.random.uniform(0, 0.5, 2000),
    'visitas': np.random.poisson(10, 2000),
    'conversion': np.random.binomial(1, 0.15, 2000)
})

X_desb = df_desb[['monto', 'descuento', 'visitas']]
y_desb = df_desb['conversion']
X_desb_train, X_desb_test, y_desb_train, y_desb_test = train_test_split(
    X_desb, y_desb, test_size=0.2, random_state=42, stratify=y_desb)

scaler = StandardScaler()
X_desb_train_s = scaler.fit_transform(X_desb_train)
X_desb_test_s = scaler.transform(X_desb_test)

grid_f1 = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_grid={'n_estimators': [50, 100], 'max_depth': [5, 10, None]},
    cv=3, scoring='f1', verbose=0
)
grid_f1.fit(X_desb_train_s, y_desb_train)

print(f"Mejores params (F1): {grid_f1.best_params_}")
print(f"Mejor F1 (CV): {grid_f1.best_score_:.3f}")
print(f"F1 en test: {grid_f1.score(X_desb_test_s, y_desb_test):.3f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: GridSearchCV con scoring='f1' para clases desbalanceadas.*

1. Dataset desbalanceado: solo 15% de conversiones

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: GridSearchCV con cv=5 y verbose=2

```python
grid_v = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid={'n_estimators': [50, 100], 'max_depth': [5, 10]},
    cv=5, scoring='accuracy', verbose=2
)
grid_v.fit(X_train_s, y_train)
print(f"\nMejores params (verbose): {grid_v.best_params_}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: GridSearchCV con cv=5 y verbose=2.*

1. `grid_v.fit(X_train_s, y_train)` — Entrena el modelo con los datos de entrenamiento.
2. `print(f"\nMejores params (verbose): {grid_v.best_params_}")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: RandomizedSearchCV con distribuciones uniforme y log-uniforme

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform, loguniform

# Distribución log-uniforme para parámetros de regularización
param_dist = {
    'C': loguniform(0.001, 100),
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

random_search = RandomizedSearchCV(
    LogisticRegression(max_iter=1000),
    param_dist, n_iter=20, cv=3, scoring='accuracy',
    random_state=42, verbose=0
)
random_search.fit(X_train_s, y_train)

print(f"Mejores params (Randomized): {random_search.best_params_}")
print(f"Mejor score: {random_search.best_score_:.3f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: RandomizedSearchCV con distribuciones uniforme y log-uniforme.*

1. Distribución log-uniforme para parámetros de regularización

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: RandomizedSearchCV para RandomForest con randint

```python
param_dist_rf = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(3, 30),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': uniform(0.3, 0.7)
}

random_rf = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist_rf, n_iter=30, cv=3, scoring='accuracy',
    random_state=42, verbose=0, n_jobs=-1
)
random_rf.fit(X_train_s, y_train)

print(f"RandomForest RandomSearch - Mejores params: {random_rf.best_params_}")
print(f"Mejor score: {random_rf.best_score_:.3f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: RandomizedSearchCV para RandomForest con randint.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: ParameterGrid — generar todas las combinaciones manualmente

```python
from sklearn.model_selection import ParameterGrid

param_grid_manual = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, None],
    'criterion': ['gini', 'entropy']
}

combos = list(ParameterGrid(param_grid_manual))
print(f"Total combinaciones: {len(combos)}")
for i, combo in enumerate(combos[:5]):
    print(f"  {i+1}. {combo}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: ParameterGrid — generar todas las combinaciones manualmente.*

1. `from sklearn.model_selection import ParameterGrid` — Importa las librerías necesarias para el análisis.
2. `print(f"Total combinaciones: {len(combos)}")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: ParameterSampler — muestra aleatoria de parámetros

```python
from sklearn.model_selection import ParameterSampler

param_dist_sampler = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 10)
}

samples = list(ParameterSampler(param_dist_sampler, n_iter=10, random_state=42))
print(f"Muestras generadas: {len(samples)}")
for i, s in enumerate(samples):
    print(f"  {i+1}. {s}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: ParameterSampler — muestra aleatoria de parámetros.*

1. `from sklearn.model_selection import ParameterSampler` — Importa las librerías necesarias para el análisis.
2. `print(f"Muestras generadas: {len(samples)}")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: HalvingGridSearchCV — búsqueda sucesiva más rápida

```python
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

halving_grid = HalvingGridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid={'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None]},
    cv=3, factor=2, min_resources='smallest',
    scoring='accuracy', verbose=0, random_state=42
)
halving_grid.fit(X_train_s, y_train)

print(f"HalvingGrid - Mejores params: {halving_grid.best_params_}")
print(f"Mejor score: {halving_grid.best_score_:.3f}")
print(f"Iteraciones realizadas: {len(halving_grid.cv_results_['params'])}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: HalvingGridSearchCV — búsqueda sucesiva más rápida.*

1. `from sklearn.experimental import enable_halving_search_cv` — Importa las librerías necesarias para el análisis.
2. `from sklearn.model_selection import HalvingGridSearchCV` — Importa las librerías necesarias para el análisis.
3. `halving_grid.fit(X_train_s, y_train)` — Entrena el modelo con los datos de entrenamiento.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: HalvingRandomSearchCV — aún más rápido

```python
halving_random = HalvingRandomSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions={
        'n_estimators': randint(50, 300),
        'max_depth': randint(3, 30),
        'min_samples_split': randint(2, 15)
    },
    n_candidates='exhaust', factor=2, cv=3,
    scoring='accuracy', random_state=42, verbose=0
)
halving_random.fit(X_train_s, y_train)

print(f"HalvingRandom - Mejores params: {halving_random.best_params_}")
print(f"Mejor score: {halving_random.best_score_:.3f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: HalvingRandomSearchCV — aún más rápido.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: refit=True — reentrenar con mejores params en todo el dataset

```python
grid_refit = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid={'n_estimators': [50, 100], 'max_depth': [5, 10]},
    cv=3, scoring='accuracy', refit=True
)
grid_refit.fit(X_train_s, y_train)

# El modelo ya está reentrenado con los mejores parámetros en X_train_s
print(f"Modelo reentrenado con: {grid_refit.best_params_}")
print(f"Predict en test: {grid_refit.predict(X_test_s)[:5]}")
print(f"Probabilidad: {grid_refit.predict_proba(X_test_s)[:5, 1].round(3)}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: refit=True — reentrenar con mejores params en todo el dataset.*

1. El modelo ya está reentrenado con los mejores parámetros en X_train_s

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: return_train_score=True — comparar train vs test

```python
grid_train = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid={'n_estimators': [50, 100], 'max_depth': [3, 5, 10, None]},
    cv=3, scoring='accuracy', return_train_score=True
)
grid_train.fit(X_train_s, y_train)

results = pd.DataFrame(grid_train.cv_results_)
results['gap'] = results['mean_train_score'] - results['mean_test_score']
cols = ['param_n_estimators', 'param_max_depth', 'mean_train_score', 'mean_test_score', 'gap']
print(results[cols].round(3))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: return_train_score=True — comparar train vs test.*

1. `grid_train.fit(X_train_s, y_train)` — Entrena el modelo con los datos de entrenamiento.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: cv_results_ DataFrame — analizar resultados detallados

```python
grid_cv = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid={'n_estimators': [50, 100], 'max_depth': [5, 10, None]},
    cv=3, scoring='accuracy', return_train_score=True
)
grid_cv.fit(X_train_s, y_train)

df_results = pd.DataFrame(grid_cv.cv_results_)
print("Columnas disponibles:")
print(df_results.columns.tolist())
print("\nResultados ordenados por rank:")
print(df_results[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']]
      .sort_values('rank_test_score').to_string())
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: cv_results_ DataFrame — analizar resultados detallados.*

1. `grid_cv.fit(X_train_s, y_train)` — Entrena el modelo con los datos de entrenamiento.
2. `print("Columnas disponibles:")` — Muestra el resultado por pantalla.
3. `print(df_results.columns.tolist())` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: best_params_ y best_score_

```python
print(f"Mejores parámetros: {grid_cv.best_params_}")
print(f"Mejor score CV: {grid_cv.best_score_:.4f}")
print(f"Score en test: {grid_cv.score(X_test_s, y_test):.4f}")

# Índice del mejor estimador
print(f"Índice del mejor: {grid_cv.best_index_}")
print(f"Mejor estimador: {type(grid_cv.best_estimator_).__name__}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: best_params_ y best_score_.*

1. Índice del mejor estimador

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: Múltiples métricas — scoring=['accuracy','f1','roc_auc']

```python
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score

grid_multi = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid={'n_estimators': [50, 100], 'max_depth': [5, 10]},
    cv=3, scoring=['accuracy', 'f1', 'roc_auc'],
    refit='roc_auc',  # refit usando roc_auc como criterio
    return_train_score=True
)
grid_multi.fit(X_train_s, y_train)

df_multi = pd.DataFrame(grid_multi.cv_results_)
cols_multi = ['params', 'mean_test_accuracy', 'mean_test_f1', 'mean_test_roc_auc']
print(df_multi[cols_multi].round(3))
print(f"\nRefit basado en roc_auc. Mejores params: {grid_multi.best_params_}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: Múltiples métricas — scoring=['accuracy','f1','roc_auc'].*

1. `from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score` — Importa las librerías necesarias para el análisis.
2. `grid_multi.fit(X_train_s, y_train)` — Entrena el modelo con los datos de entrenamiento.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: Custom scorer — make_scorer para profit_score

```python
def profit_score(y_true, y_pred):
    """
    Métrica de negocio: ganancia neta por predicción correcta.
    - VP (acertar conversión): +$50 de ganancia
    - FP (falso positivo): -$10 (costo de oferta)
    - FN (falso negativo): -$30 (pérdida de venta potencial)
    - VN (correcto no conversión): +$0
    """
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    ganancia = tp * 50 - fp * 10 - fn * 30
    return ganancia / len(y_true)

profit_scorer = make_scorer(profit_score, greater_is_better=True)

grid_profit = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid={'n_estimators': [50, 100], 'max_depth': [5, 10, None]},
    cv=3, scoring=profit_scorer
)
grid_profit.fit(X_train_s, y_train)

print(f"Mejores params (profit): {grid_profit.best_params_}")
print(f"Mejor profit CV: ${grid_profit.best_score_:.2f} por cliente")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: Custom scorer — make_scorer para profit_score.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 16: Custom scorer con needs_proba=True para AUC

```python
def ganancia_esperada(y_true, y_proba):
    """Ganancia basada en probabilidad, umbral óptimo."""
    umbral = 0.3  # umbral de decisión de negocio
    y_pred = (y_proba[:, 1] >= umbral).astype(int)
    return profit_score(y_true, y_pred)

ganancia_scorer = make_scorer(ganancia_esperada, greater_is_better=True, needs_proba=True)

grid_proba = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid={'n_estimators': [50, 100], 'max_depth': [5, 10]},
    cv=3, scoring=ganancia_scorer
)
grid_proba.fit(X_train_s, y_train)

print(f"Mejores params (prob ganancia): {grid_proba.best_params_}")
print(f"Ganancia CV: ${grid_proba.best_score_:.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: Custom scorer con needs_proba=True para AUC.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 17: Comparar GridSearch vs RandomizedSearch vs HalvingSearch (tiempo)

```python
import time

param_grid_comp = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10, 15, None]}
param_dist_comp = {'n_estimators': randint(50, 200), 'max_depth': randint(3, 15)}

strategies = {}

# GridSearch
t0 = time.time()
gs = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_comp,
                  cv=3, scoring='accuracy', n_jobs=-1)
gs.fit(X_train_s, y_train)
strategies['GridSearch'] = {'time': time.time() - t0, 'score': gs.best_score_}

# RandomizedSearch
t0 = time.time()
rs = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_dist_comp,
                        n_iter=15, cv=3, scoring='accuracy', n_jobs=-1, random_state=42)
rs.fit(X_train_s, y_train)
strategies['Randomized'] = {'time': time.time() - t0, 'score': rs.best_score_}

# HalvingGrid
t0 = time.time()
hs = HalvingGridSearchCV(RandomForestClassifier(random_state=42), param_grid_comp,
                         cv=3, factor=2, scoring='accuracy', n_jobs=-1, random_state=42)
hs.fit(X_train_s, y_train)
strategies['HalvingGrid'] = {'time': time.time() - t0, 'score': hs.best_score_}

for name, res in strategies.items():
    print(f"{name:15s} | Score: {res['score']:.4f} | Tiempo: {res['time']:.2f}s")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: Comparar GridSearch vs RandomizedSearch vs HalvingSearch (tiempo).*

1. GridSearch
2. RandomizedSearch
3. HalvingGrid

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 18: Integrador — optimizar pipeline completo para predicción de demanda

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor

# Datos de demanda
np.random.seed(42)
n = 2000
df_dem = pd.DataFrame({
    'precio': np.random.uniform(10, 500, n),
    'descuento': np.random.uniform(0, 0.5, n),
    'gasto_mk': np.random.exponential(1000, n),
    'temporada': np.random.choice(['alta', 'baja', 'media'], n),
    'cat': np.random.choice(['elec', 'ropa', 'hogar'], n),
    'demanda': np.random.poisson(50, n)
})
X_dem = df_dem.drop('demanda', axis=1)
y_dem = df_dem['demanda']
X_dem_train, X_dem_test, y_dem_train, y_dem_test = train_test_split(
    X_dem, y_dem, test_size=0.2, random_state=42)

preprocessor_dem = ColumnTransformer([
    ('num', StandardScaler(), ['precio', 'descuento', 'gasto_mk']),
    ('cat', OneHotEncoder(drop='first'), ['temporada', 'cat'])
])

pipe_opt = Pipeline([
    ('prep', preprocessor_dem),
    ('model', GradientBoostingRegressor(random_state=42))
])

param_grid_opt = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [3, 5],
    'model__learning_rate': [0.05, 0.1],
    'model__subsample': [0.8, 1.0]
}

grid_opt = GridSearchCV(pipe_opt, param_grid_opt, cv=3, scoring='neg_mean_squared_error', verbose=1)
grid_opt.fit(X_dem_train, y_dem_train)

from sklearn.metrics import mean_absolute_error, r2_score
y_pred_opt = grid_opt.predict(X_dem_test)

print(f"\nMejores params: {grid_opt.best_params_}")
print(f"R²: {r2_score(y_dem_test, y_pred_opt):.3f}")
print(f"MAE: {mean_absolute_error(y_dem_test, y_pred_opt):.1f} unidades")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador — optimizar pipeline completo para predicción de demanda.*

1. Datos de demanda

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejercicios

1. **GridSearchCV para GradientBoosting**: Optimiza un GradientBoostingRegressor para predecir lead time de compras. Varía learning_rate, n_estimators, max_depth y subsample. Reporta los 5 mejores resultados.

2. **RandomizedSearchCV para XGBoost**: Usa RandomizedSearchCV con distribuciones (randint, uniform, loguniform) para optimizar un XGBClassifier en datos de conversión de ventas con 30 iteraciones.

3. **HalvingGridSearchCV vs GridSearchCV**: Compara tiempo y resultado de HalvingGridSearchCV vs GridSearchCV para el mismo problema de clasificación ABC de inventarios. Usa al menos 4 parámetros.

4. **Multi-metric optimization**: Usa GridSearchCV con scoring=['precision', 'recall', 'f1'] y refit='f1' para un problema de predicción de productos defectuosos en compras. Muestra todas las métricas.

5. **Custom scorer de negocio**: Crea un scorer personalizado que calcule el beneficio neto de una campaña de marketing: beneficio = VP * $100 - FP * $20 - FN * $50. Optimiza un pipeline con este scorer.

6. **Análisis de cv_results_**: Ejecuta GridSearchCV con return_train_score=True para un RandomForest. Crea un DataFrame con los resultados, calcula el gap train-test y encuentra los parámetros con menor overfitting.

7. **ParameterGrid y ParameterSampler**: Genera todas las combinaciones de {n_estimators, max_depth, min_samples_split, criterion} para RandomForest. Luego muestrea 20 aleatoriamente. Compara las coberturas.

8. **Optimización completa de pipeline**: Diseña un pipeline con ColumnTransformer (3 tipos de columnas) + FeatureUnion (PCA + SelectKBest) + RandomForest. Optimiza parámetros de todos los pasos con GridSearchCV. Mide R² en test.

---

## Resumen

- **GridSearchCV**: Exhaustivo, garantiza óptimo dentro de la grilla, pero lento con muchos parámetros.
- **RandomizedSearchCV**: Muestrea aleatoriamente, mejor para espacios grandes o cuando algunos parámetros importan poco.
- **HalvingGridSearchCV / HalvingRandomSearchCV**: Búsqueda sucesiva, más rápida, descarta malas combinaciones temprano.
- **ParameterGrid / ParameterSampler**: Útiles para inspeccionar espacios de búsqueda antes de ejecutar.
- **Custom scorers** con `make_scorer` permiten optimizar métricas de negocio reales (ganancia, costo).
- **Múltiples métricas** con `scoring=['acc', 'f1', 'roc_auc']` y `refit='f1'` permiten optimizar una métrica mientras se monitorean otras.
- **Sintaxis `__`** para parámetros de pasos internos en pipelines (ej: `model__n_estimators`).
- El análisis de `cv_results_` revela overfitting (gap train-test) y estabilidad de parámetros.
- En ventas/compras/inventarios, la optimización debe alinearse con métricas de negocio, no solo accuracy.
