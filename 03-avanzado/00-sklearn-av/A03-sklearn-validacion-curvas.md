# A03: Validación, Curvas de Validación y Aprendizaje para Ventas, Compras e Inventarios

## Introducción Teórica

La **validación de modelos** va más allá del simple train/test split. scikit-learn proporciona herramientas avanzadas para entender el comportamiento del modelo:

### Herramientas de diagnóstico:

1. **validation_curve**: Evalúa el efecto de un hiperparámetro sobre el rendimiento (train vs test) variando su valor. Ayuda a detectar overfitting/underfitting.
2. **learning_curve**: Muestra cómo el rendimiento mejora (o se estanca) al aumentar el tamaño del dataset. Indica si conviene recolectar más datos.
3. **permutation_importance**: Mide importancia de features permutando aleatoriamente sus valores. No depende del modelo, solo de la caída en rendimiento.
4. **partial_dependence**: Muestra cómo cambia la predicción al variar una feature, manteniendo las demás constantes (efecto marginal).
5. **cross_validate**: cross_val_score mejorado que puede retornar múltiples métricas y estimadores entrenados.

### Estrategias de Cross-Validation:

- **StratifiedKFold**: Mantiene proporción de clases en cada fold (clasificación).
- **TimeSeriesSplit**: Respeta orden temporal (ventas por fecha).
- **RepeatedKFold**: Repite K-Fold para estimación más robusta.
- **GroupKFold**: Grupos completos no se dividen entre train/test (mismo cliente).
- **LeaveOneOut**: n modelos, cada uno con una muestra de test diferente.
- **ShuffleSplit**: Mezcla aleatoria repetida, útil para datasets grandes.

### Aplicación en negocio:
- **Ventas**: Validación temporal (TimeSeriesSplit) porque los patrones de compra cambian con el tiempo.
- **Compras**: GroupKFold por proveedor (no entrenar con datos de un proveedor y testear con otro).
- **Inventarios**: StratifiedKFold para clasificación ABC desbalanceada.

---

## Ejemplos

### Ejemplo 1: validation_curve — efecto de max_depth en DecisionTree

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve, train_test_split
from sklearn.tree import DecisionTreeClassifier
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

scaler = StandardScaler()
X_s = scaler.fit_transform(X)

param_range = [1, 3, 5, 10, 15, 20, 30]
train_scores, test_scores = validation_curve(
    DecisionTreeClassifier(random_state=42),
    X_s, y, param_name="max_depth", param_range=param_range,
    cv=5, scoring="accuracy"
)

train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

print("max_depth | Train Acc | Test Acc")
for i, d in enumerate(param_range):
    print(f"     {d:2d}    |   {train_mean[i]:.3f}  |   {test_mean[i]:.3f}")

# Detectar punto de overfitting
gap = train_mean - test_mean
best_depth = param_range[np.argmax(test_mean)]
print(f"\nMejor max_depth: {best_depth} (test acc: {test_mean.max():.3f})")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: validation_curve — efecto de max_depth en DecisionTree.*

1. Detectar punto de overfitting

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: validation_curve — efecto de n_estimators en RandomForest

```python
from sklearn.ensemble import RandomForestClassifier

param_range_n = [10, 50, 100, 200, 300]
train_scores_n, test_scores_n = validation_curve(
    RandomForestClassifier(random_state=42),
    X_s, y, param_name="n_estimators", param_range=param_range_n,
    cv=3, scoring="accuracy"
)

train_mean_n = train_scores_n.mean(axis=1)
test_mean_n = test_scores_n.mean(axis=1)

print("n_estimators | Train Acc | Test Acc")
for i, n in enumerate(param_range_n):
    print(f"     {n:3d}     |   {train_mean_n[i]:.3f}  |   {test_mean_n[i]:.3f}")

mejor_n = param_range_n[np.argmax(test_mean_n)]
print(f"\nMejor n_estimators: {mejor_n}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: validation_curve — efecto de n_estimators en RandomForest.*

1. `from sklearn.ensemble import RandomForestClassifier` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: validation_curve — efecto de C en LogisticRegression

```python
from sklearn.linear_model import LogisticRegression

param_range_c = [0.001, 0.01, 0.1, 1, 10, 100]
train_scores_c, test_scores_c = validation_curve(
    LogisticRegression(max_iter=1000, random_state=42),
    X_s, y, param_name="C", param_range=param_range_c,
    cv=5, scoring="accuracy"
)

train_mean_c = train_scores_c.mean(axis=1)
test_mean_c = test_scores_c.mean(axis=1)

print("     C     | Train Acc | Test Acc")
for i, c in enumerate(param_range_c):
    print(f" {c:8.3f} |   {train_mean_c[i]:.3f}  |   {test_mean_c[i]:.3f}")

mejor_c = param_range_c[np.argmax(test_mean_c)]
print(f"\nMejor C: {mejor_c}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: validation_curve — efecto de C en LogisticRegression.*

1. `from sklearn.linear_model import LogisticRegression` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: learning_curve — rendimiento vs tamaño de dataset

```python
from sklearn.model_selection import learning_curve

train_sizes_abs, train_scores_lc, test_scores_lc = learning_curve(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X_s, y, train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring='accuracy'
)

train_mean_lc = train_scores_lc.mean(axis=1)
test_mean_lc = test_scores_lc.mean(axis=1)

print("Tamaño train | Train Acc | Test Acc")
for i, size in enumerate(train_sizes_abs):
    print(f"   {size:4d}      |   {train_mean_lc[i]:.3f}  |   {test_mean_lc[i]:.3f}")

# ¿Conviene más datos?
gap_lc = train_mean_lc[-1] - test_mean_lc[-1]
print(f"\nGap final train-test: {gap_lc:.3f}")
print(f"Mejora al aumentar datos: {test_mean_lc[-1] - test_mean_lc[0]:.3f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: learning_curve — rendimiento vs tamaño de dataset.*

1. ¿Conviene más datos?

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: learning_curve — detectar overfitting

```python
# Modelo propenso a overfitting: árbol profundo
train_sizes_of, train_of, test_of = learning_curve(
    DecisionTreeClassifier(max_depth=20, random_state=42),
    X_s, y, train_sizes=np.linspace(0.1, 1.0, 8),
    cv=5, scoring='accuracy'
)

train_mean_of = train_of.mean(axis=1)
test_mean_of = test_of.mean(axis=1)

print("Tamaño | Train | Test | Gap")
for i, s in enumerate(train_sizes_of):
    gap = train_mean_of[i] - test_mean_of[i]
    print(f"  {s:4d} | {train_mean_of[i]:.3f} | {test_mean_of[i]:.3f} | {gap:.3f}")

print(f"\nOverfitting severo si gap > 0.1: {train_mean_of[-1] - test_mean_of[-1]:.3f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: learning_curve — detectar overfitting.*

1. Modelo propenso a overfitting: árbol profundo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: learning_curve con shuffle=True

```python
train_sizes_sh, train_sh, test_sh = learning_curve(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X_s, y, train_sizes=np.linspace(0.1, 1.0, 8),
    cv=5, scoring='accuracy', shuffle=True, random_state=42
)

train_mean_sh = train_sh.mean(axis=1)
test_mean_sh = test_sh.mean(axis=1)

print("Tamaño | Train | Test (shuffle)")
for i, s in enumerate(train_sizes_sh):
    print(f"  {s:4d} | {train_mean_sh[i]:.3f} | {test_mean_sh[i]:.3f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: learning_curve con shuffle=True.*

1. `print("Tamaño | Train | Test (shuffle)")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: permutation_importance — importancia por permutación

```python
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingClassifier

X_train, X_test, y_train, y_test = train_test_split(X_s, y, test_size=0.2, random_state=42)

model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, scoring='accuracy')

print("Feature | Importancia | std")
for i, col in enumerate(['monto', 'frecuencia', 'recencia']):
    print(f" {col:12s} |   {result.importances_mean[i]:.4f}   | {result.importances_std[i]:.4f}")

# Features importantes vs no importantes
sorted_idx = result.importances_mean.argsort()
print(f"\nFeature más importante: {['monto','frecuencia','recencia'][sorted_idx[-1]]}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: permutation_importance — importancia por permutación.*

1. Features importantes vs no importantes

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: partial_dependence — efecto del precio en predicción de demanda

```python
from sklearn.inspection import partial_dependence
from sklearn.ensemble import GradientBoostingRegressor

# Datos de demanda vs precio
df_dem = pd.DataFrame({
    'precio': np.random.uniform(10, 500, 2000),
    'descuento': np.random.uniform(0, 0.5, 2000),
    'gasto_mk': np.random.exponential(1000, 2000),
    'demanda': np.random.poisson(50, 2000)
})
X_dem = df_dem[['precio', 'descuento', 'gasto_mk']]
y_dem = df_dem['demanda']

scaler_dem = StandardScaler()
X_dem_s = scaler_dem.fit_transform(X_dem)

model_dem = GradientBoostingRegressor(n_estimators=100, random_state=42)
model_dem.fit(X_dem_s, y_dem)

# PDP para precio
pdp_precio = partial_dependence(model_dem, X_dem_s, features=[0], kind='average', grid_resolution=20)

# Reconstruir valores originales de precio
precio_vals = np.linspace(10, 500, 20)
print("Precio | Demanda promedio (PDP)")
for i, p in enumerate(precio_vals):
    print(f" {p:6.1f} | {pdp_precio['average'][0][i]:.1f}")

print(f"\nInterpretación: a mayor precio, la demanda { 'sube' if pdp_precio['average'][0][-1] > pdp_precio['average'][0][0] else 'baja' }")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: partial_dependence — efecto del precio en predicción de demanda.*

1. Datos de demanda vs precio
2. PDP para precio
3. Reconstruir valores originales de precio

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: plot_partial_dependence — visualización PDP (textual)

```python
# PDP para todas las features
pdp_all = partial_dependence(model_dem, X_dem_s, features=[0, 1, 2], kind='average')

feature_names = ['precio', 'descuento', 'gasto_mk']
feature_vals = {
    'precio': np.linspace(10, 500, 20),
    'descuento': np.linspace(0, 0.5, 20),
    'gasto_mk': np.linspace(0, 5000, 20)
}

for idx, name in enumerate(feature_names):
    vals = feature_vals[name]
    pdp_vals = pdp_all['average'][idx]
    print(f"\nPDP - {name}:")
    print(f"  Rango: {vals[0]:.1f} a {vals[-1]:.1f}")
    print(f"  Demanda min: {pdp_vals.min():.1f}, max: {pdp_vals.max():.1f}")
    print(f"  Efecto neto: {pdp_vals[-1] - pdp_vals[0]:.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: plot_partial_dependence — visualización PDP (textual).*

1. PDP para todas las features

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: cross_validate con return_train_score=True

```python
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC

cv_results = cross_validate(
    SVC(kernel='rbf', gamma='scale', random_state=42),
    X_s, y, cv=5,
    scoring=['accuracy', 'f1'],
    return_train_score=True,
    return_estimator=True
)

print("Fold | Train Acc | Test Acc | Train F1 | Test F1")
for i in range(5):
    print(f"  {i+1}  |   {cv_results['train_accuracy'][i]:.3f}   |  {cv_results['test_accuracy'][i]:.3f}   |  {cv_results['train_f1'][i]:.3f}  |  {cv_results['test_f1'][i]:.3f}")

gap = cv_results['train_accuracy'].mean() - cv_results['test_accuracy'].mean()
print(f"\nGap medio accuracy: {gap:.3f}")
print(f"Estimadores guardados: {len(cv_results['estimator'])}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: cross_validate con return_train_score=True.*

1. `from sklearn.model_selection import cross_validate` — Importa las librerías necesarias para el análisis.
2. `from sklearn.svm import SVC` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: StratifiedKFold — mantener proporción de clases

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_skf = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_s, y), 1):
    X_tr, X_val = X_s[train_idx], X_s[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_tr, y_tr)
    score = model.score(X_val, y_val)
    scores_skf.append(score)

    # Proporción de clases en validación
    prop_val = y_val.mean()
    print(f"Fold {fold}: Acc={score:.3f}, %positivos val={prop_val:.2f} (global={y.mean():.2f})")

print(f"\nStratifiedKFold mean: {np.mean(scores_skf):.3f} ± {np.std(scores_skf):.3f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: StratifiedKFold — mantener proporción de clases.*

1. Proporción de clases en validación

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: TimeSeriesSplit — validación temporal (ventas por mes)

```python
from sklearn.model_selection import TimeSeriesSplit

# Datos temporales de ventas
np.random.seed(42)
fechas = pd.date_range('2024-01-01', periods=200, freq='D')
df_ts = pd.DataFrame({
    'fecha': fechas,
    'dia_semana': fechas.dayofweek,
    'mes': fechas.month,
    'ventas': np.random.poisson(100, 200) + np.sin(np.arange(200) * 0.1) * 20
})

X_ts = df_ts[['dia_semana', 'mes']]
y_ts = df_ts['ventas']

tscv = TimeSeriesSplit(n_splits=5)
scores_ts = []

print("TimeSeriesSplit:")
for fold, (train_idx, test_idx) in enumerate(tscv.split(X_ts), 1):
    X_tr, X_te = X_ts.iloc[train_idx], X_ts.iloc[test_idx]
    y_tr, y_te = y_ts.iloc[train_idx], y_ts.iloc[test_idx]

    model = GradientBoostingRegressor(n_estimators=50, random_state=42)
    model.fit(X_tr, y_tr)
    from sklearn.metrics import r2_score
    score = r2_score(y_te, model.predict(X_te))
    scores_ts.append(score)

    print(f"  Fold {fold}: Train hasta {df_ts.iloc[train_idx[-1]]['fecha'].date()}, "
          f"Test {df_ts.iloc[test_idx[0]]['fecha'].date()} - {df_ts.iloc[test_idx[-1]]['fecha'].date()}, "
          f"R²={score:.3f}")

print(f"\nTimeSeriesSplit mean R²: {np.mean(scores_ts):.3f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: TimeSeriesSplit — validación temporal (ventas por mes).*

1. Datos temporales de ventas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: RepeatedKFold — repetir K-Fold para estimación robusta

```python
from sklearn.model_selection import RepeatedKFold

rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
scores_rkf = []

for train_idx, test_idx in rkf.split(X_s):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_s[train_idx], y[train_idx])
    scores_rkf.append(model.score(X_s[test_idx], y[test_idx]))

print(f"RepeatedKFold (5x10=50 ejecuciones):")
print(f"  Accuracy: {np.mean(scores_rkf):.3f} ± {np.std(scores_rkf):.4f}")
print(f"  IC 95%: [{np.percentile(scores_rkf, 2.5):.3f}, {np.percentile(scores_rkf, 97.5):.3f}]")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: RepeatedKFold — repetir K-Fold para estimación robusta.*

1. `from sklearn.model_selection import RepeatedKFold` — Importa las librerías necesarias para el análisis.
2. `model.fit(X_s[train_idx], y[train_idx])` — Entrena el modelo con los datos de entrenamiento.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: LeaveOneOut — cada muestra como test (n=1)

```python
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Usamos un subset pequeño para LOOCV (n=100)
X_loo = X_s[:100]
y_loo = y[:100]

loo = LeaveOneOut()
y_pred_loo = []
y_true_loo = []

for train_idx, test_idx in loo.split(X_loo):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_loo[train_idx], y_loo[train_idx])
    y_pred_loo.append(model.predict(X_loo[test_idx])[0])
    y_true_loo.append(y_loo[test_idx])

acc_loo = accuracy_score(y_true_loo, y_pred_loo)
print(f"LeaveOneOut accuracy: {acc_loo:.3f}")
print(f"LOOCV es costoso: {len(X_loo)} modelos entrenados")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: LeaveOneOut — cada muestra como test (n=1).*

1. Usamos un subset pequeño para LOOCV (n=100)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: ShuffleSplit — mezcla aleatoria repetida

```python
from sklearn.model_selection import ShuffleSplit

ss = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
scores_ss = []

for train_idx, test_idx in ss.split(X_s):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_s[train_idx], y[train_idx])
    scores_ss.append(model.score(X_s[test_idx], y[test_idx]))

print(f"ShuffleSplit (10 splits, 80/20):")
print(f"  Accuracy: {np.mean(scores_ss):.3f} ± {np.std(scores_ss):.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: ShuffleSplit — mezcla aleatoria repetida.*

1. `from sklearn.model_selection import ShuffleSplit` — Importa las librerías necesarias para el análisis.
2. `model.fit(X_s[train_idx], y[train_idx])` — Entrena el modelo con los datos de entrenamiento.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 16: Comparar CV strategies en mismo modelo

```python
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit

model_comp = RandomForestClassifier(n_estimators=100, random_state=42)
strategies = {
    'KFold(5)': KFold(5, shuffle=True, random_state=42),
    'StratifiedKFold(5)': StratifiedKFold(5, shuffle=True, random_state=42),
    'RepeatedKFold(5x3)': RepeatedKFold(5, 3, random_state=42),
    'ShuffleSplit(10)': ShuffleSplit(10, test_size=0.2, random_state=42)
}

print("Estrategia       | Score medio | std")
for name, cv in strategies.items():
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model_comp, X_s, y, cv=cv, scoring='accuracy')
    print(f" {name:20s} |   {scores.mean():.3f}    | {scores.std():.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: Comparar CV strategies en mismo modelo.*

1. `from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 17: GroupKFold — no mezclar mismo grupo (ej: mismo cliente)

```python
from sklearn.model_selection import GroupKFold

# Datos con grupos: mismo cliente no debe estar en train y test
np.random.seed(42)
n_clientes = 50
grupos = np.random.randint(0, n_clientes, 1000)

df_grupos = pd.DataFrame({
    'monto': np.random.exponential(500, 1000),
    'cliente_id': grupos,
    'compro': np.random.binomial(1, 0.5, 1000)
})

X_grp = df_grupos[['monto']].values
y_grp = df_grupos['compro'].values
groups_grp = df_grupos['cliente_id'].values

gkf = GroupKFold(n_splits=5)
scores_grp = []

for fold, (train_idx, test_idx) in enumerate(gkf.split(X_grp, y_grp, groups_grp), 1):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_grp[train_idx], y_grp[train_idx])
    score = model.score(X_grp[test_idx], y_grp[test_idx])

    # Verificar que no hay clientes compartidos
    train_clients = set(groups_grp[train_idx])
    test_clients = set(groups_grp[test_idx])
    overlap = train_clients & test_clients

    print(f"Fold {fold}: Acc={score:.3f}, Clientes compartidos={len(overlap)}")
    scores_grp.append(score)

print(f"\nGroupKFold mean: {np.mean(scores_grp):.3f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: GroupKFold — no mezclar mismo grupo (ej: mismo cliente).*

1. Datos con grupos: mismo cliente no debe estar en train y test
2. Verificar que no hay clientes compartidos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 18: Integrador — validación exhaustiva del mejor modelo

```python
"""
Validación exhaustiva para modelo de predicción de demanda:
1. Learning curve para decidir si recolectar más datos
2. Validation curve para elegir hiperparámetros
3. Permutation importance para features clave
4. TimeSeriesSplit por ser datos temporales
5. Partial dependence para interpretar precio
"""

from sklearn.metrics import mean_squared_error, r2_score

# 1. Datos
np.random.seed(42)
n = 2000
df_val = pd.DataFrame({
    'precio': np.random.uniform(10, 500, n),
    'descuento': np.random.uniform(0, 0.5, n),
    'gasto_mk': np.random.exponential(1000, n),
    'temporada': np.random.choice(['alta', 'baja'], n),
    'dia_semana': np.random.randint(0, 7, n),
    'demanda': np.random.poisson(50, n)
})

X_val = df_val[['precio', 'descuento', 'gasto_mk', 'temporada', 'dia_semana']]
y_val = df_val['demanda']

from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

prep_val = ColumnTransformer([
    ('num', StandardScaler(), ['precio', 'descuento', 'gasto_mk', 'dia_semana']),
    ('cat', OneHotEncoder(drop='first'), ['temporada'])
])

pipe_val = Pipeline([
    ('prep', prep_val),
    ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))
])

tscv = TimeSeriesSplit(n_splits=5)
scores_val = {'R2': [], 'RMSE': [], 'MAE': []}

from sklearn.metrics import mean_absolute_error

for fold, (tr_idx, te_idx) in enumerate(tscv.split(X_val), 1):
    X_tr, X_te = X_val.iloc[tr_idx], X_val.iloc[te_idx]
    y_tr, y_te = y_val.iloc[tr_idx], y_val.iloc[te_idx]

    pipe_val.fit(X_tr, y_tr)
    y_pred = pipe_val.predict(X_te)

    scores_val['R2'].append(r2_score(y_te, y_pred))
    scores_val['RMSE'].append(np.sqrt(mean_squared_error(y_te, y_pred)))
    scores_val['MAE'].append(mean_absolute_error(y_te, y_pred))

print("Validación Temporal (TimeSeriesSplit 5 folds):")
print(f"  R²:  {np.mean(scores_val['R2']):.3f} ± {np.std(scores_val['R2']):.3f}")
print(f"  RMSE: {np.mean(scores_val['RMSE']):.1f} ± {np.std(scores_val['RMSE']):.1f}")
print(f"  MAE:  {np.mean(scores_val['MAE']):.1f} ± {np.std(scores_val['MAE']):.1f}")

# Permutation importance
from sklearn.inspection import permutation_importance
pipe_val.fit(X_val, y_val)
X_val_proc = prep_val.fit_transform(X_val)
model_only = pipe_val.named_steps['model']

perm = permutation_importance(model_only, X_val_proc, y_val, n_repeats=10, random_state=42)
feat_names = prep_val.get_feature_names_out()
sorted_idx = perm.importances_mean.argsort()[::-1]

print("\nImportancia por permutación:")
for i in sorted_idx:
    print(f"  {feat_names[i]:30s} {perm.importances_mean[i]:.3f} ± {perm.importances_std[i]:.3f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador — validación exhaustiva del mejor modelo.*

1. 1. Datos
2. Permutation importance

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejercicios

1. **Validation curve para GradientBoosting**: Genera validation_curve variando `learning_rate` en [0.01, 0.05, 0.1, 0.2, 0.5] para predecir demanda. Identifica el punto de overfitting con el gap train-test.

2. **Learning curve para clasificación de ventas**: Genera learning_curve para LogisticRegression en datos de 5000 ventas. ¿A partir de qué tamaño deja de mejorar el rendimiento? ¿Conviene recolectar más datos?

3. **Permutation importance en compras**: Aplica permutation_importance a un modelo que predice lead time de proveedores. Identifica las 3 features más importantes (días_entrega_hist, costo_envio, volumen, pais_origen).

4. **Partial Dependence para descuento**: Calcula y reporta el PDP del descuento sobre la probabilidad de conversión en ventas. Interpreta: ¿descuentos altos siempre mejoran conversión?

5. **TimeSeriesSplit para predicción de inventarios**: Implementa validación temporal con TimeSeriesSplit(5) para predecir stock mínimo requerido la próxima semana. Muestra R² por fold y verifica que no hay data leakage temporal.

6. **GroupKFold por proveedor**: Simula datos de compras donde cada fila es una orden de compra y el grupo es el proveedor. Demuestra que GroupKFold evita que el mismo proveedor esté en train y test.

7. **Comparación de 5 estrategias CV**: Compara KFold(5), StratifiedKFold(5), RepeatedKFold(5x5), ShuffleSplit(20) y TimeSeriesSplit(5) para el mismo modelo y dataset. Crea una tabla con media y std de accuracy.

8. **Validación exhaustiva completa**: Implementa un pipeline completo para clasificación ABC de inventarios (3 clases). Aplica: (a) learning_curve, (b) validation_curve para max_depth, (c) permutation_importance, (d) cross_validate con 3 métricas. Reporta conclusiones de negocio.

---

## Resumen

- **validation_curve**: Diagnostica overfitting/underfitting variando un hiperparámetro. El punto donde test_score deja de mejorar es el valor óptimo.
- **learning_curve**: Determina si más datos mejorarán el modelo. Si train_score y test_score convergen, más datos no ayudarán.
- **permutation_importance**: Importancia de features independiente del modelo. Mide caída en rendimiento al permutar cada feature.
- **partial_dependence**: Efecto marginal de una feature en la predicción. Útil para interpretabilidad de negocio ("¿subir precio reduce demanda?").
- **cross_validate**: cross_val_score extendido con múltiples métricas y acceso a estimadores entrenados.
- **TimeSeriesSplit**: Crucial para ventas/negocio donde el orden temporal importa.
- **StratifiedKFold**: Para clasificación con clases desbalanceadas.
- **GroupKFold**: Cuando hay grupos naturales (clientes, proveedores) que no deben dividirse.
- **RepeatedKFold**: Estimación más robusta de la varianza del modelo.
