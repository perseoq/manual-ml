# A08 — Feature Selection para Ventas, Compras e Inventarios

## Fundamentos Teóricos

La **selección de características** elige un subconjunto relevante de variables predictoras descartando las redundantes o ruidosas. Beneficios: reduce overfitting, mejora interpretabilidad, acelera entrenamiento y puede aumentar precisión.

### Categorías principales

| Tipo | Métodos | Ejemplo |
|------|---------|---------|
| **Filtro** | VarianceThreshold, SelectKBest, SelectPercentile, GenericUnivariateSelect | Estadísticos univariados independientes del modelo |
| **Wrapper** | RFE, RFECV, SequentialFeatureSelector | Entrena modelo con subconjuntos y evalúa |
| **Embedded** | SelectFromModel, Lasso, TreeImportance | La selección es parte del entrenamiento |

### Métricas de score univariado

- `f_classif`: ANOVA F-value (target categórico)
- `chi2`: Chi-cuadrado (features no negativas)
- `mutual_info_classif`: Información mutua (clasificación)
- `f_regression`: F-test (target continuo)
- `mutual_info_regression`: Información mutua (regresión)

### RFE (Recursive Feature Elimination)

Entrena modelo con todas las features, ordena por importancia, elimina la(s) menos importante(s) y repite hasta `n_features_to_select`.

### SelectFromModel

Usa el atributo `coef_` o `feature_importances_` de un estimador entrenado y selecciona aquellas por encima de un `threshold`.

### RFECV

Igual que RFE pero usa validación cruzada para determinar automáticamente el número óptimo de features.

### SequentialFeatureSelector

Construye (forward) o elimina (backward) features secuencialmente evaluando con CV cada paso.

---

## Configuración

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, SelectPercentile, RFE, RFECV,
    SelectFromModel, SequentialFeatureSelector, GenericUnivariateSelect,
    f_classif, chi2, mutual_info_classif, f_regression,
    mutual_info_regression
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, mean_squared_error, r2_score
)

np.random.seed(42)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Configuración.*

1. `import numpy as np` — Importa las librerías necesarias para el análisis.
2. `import pandas as pd` — Importa las librerías necesarias para el análisis.
3. `import matplotlib.pyplot as plt` — Importa las librerías necesarias para el análisis.
4. `import seaborn as sns` — Importa las librerías necesarias para el análisis.
5. `from sklearn.datasets import make_classification, make_regression` — Importa las librerías necesarias para el análisis.
6. `from sklearn.model_selection import (` — Importa las librerías necesarias para el análisis.
7. `from sklearn.preprocessing import StandardScaler, LabelEncoder` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejemplo 1: VarianceThreshold — eliminar columnas con varianza casi 0

```python
# Simulamos datos de ventas: columna 'total_impuesto' casi constante
np.random.seed(42)
n = 500
df_ventas = pd.DataFrame({
    'precio': np.random.uniform(10, 500, n),
    'cantidad': np.random.randint(1, 20, n),
    'descuento_pct': np.random.uniform(0, 30, n),
    # Casi constante (99.9% del tiempo vale 0.21)
    'tasa_impuesto': [0.21] * 499 + [0.22],
    # Constante
    'moneda': ['USD'] * n,
    'es_interno': [1] * n,
})
print(df_ventas.head())
print(f"\nVarianzas:\n{df_ventas.var(numeric_only=True)}")

selector = VarianceThreshold(threshold=0.01)
X_sel = selector.fit_transform(df_ventas.select_dtypes(include=[np.number]))

cols_eliminadas = df_ventas.columns[
    ~selector.get_support()
]
print(f"\nColumnas eliminadas por baja varianza: {list(cols_eliminadas)}")
print(f"Shape original: {df_ventas.shape[1]} → Shape después: {X_sel.shape[1]}")
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

*Ejemplo 1: VarianceThreshold — eliminar columnas con varianza casi 0.*

1. Simulamos datos de ventas, compras o inventarios: columna 'total_impuesto' casi constante
2. Casi constante (99.9% del tiempo vale 0.21)
3. Constante

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejemplo 2: SelectKBest con f_classif — top 5 features para clasificación de productos

```python
# Clasificación: producto es "alta rotacion" (1) o "baja rotacion" (0)
np.random.seed(42)
n = 1000
X_ventas = pd.DataFrame({
    'precio_promedio': np.random.uniform(5, 500, n),
    'unidades_vendidas': np.random.poisson(50, n),
    'dias_stock': np.random.exponential(30, n),
    'margen_pct': np.random.uniform(5, 60, n),
    'num_proveedores': np.random.randint(1, 6, n),
    'peso_kg': np.random.uniform(0.1, 25, n),
    'volumen_cm3': np.random.uniform(100, 5000, n),
    'rating': np.random.uniform(1, 5, n),
})

y_rotacion = (
    (X_ventas['unidades_vendidas'] > 40) &
    (X_ventas['dias_stock'] < 20)
).astype(int)

selector = SelectKBest(score_func=f_classif, k=5)
X_sel = selector.fit_transform(X_ventas, y_rotacion)

scores = pd.DataFrame({
    'feature': X_ventas.columns,
    'score': selector.scores_,
    'pvalue': selector.pvalues_,
    'seleccionada': selector.get_support()
}).sort_values('score', ascending=False)

print("Top 5 features según f_classif:")
print(scores[scores['seleccionada']])
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: SelectKBest con f_classif — top 5 features para clasificación de productos.*

1. Clasificación: producto es "alta rotacion" (1) o "baja rotacion" (0)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejemplo 3: SelectKBest con mutual_info_classif — información mutua con target

```python
selector_mi = SelectKBest(score_func=mutual_info_classif, k=5)
X_sel_mi = selector_mi.fit_transform(X_ventas, y_rotacion)

scores_mi = pd.DataFrame({
    'feature': X_ventas.columns,
    'score_mi': selector_mi.scores_,
    'seleccionada': selector_mi.get_support()
}).sort_values('score_mi', ascending=False)

print("Top 5 features según información mutua:")
print(scores_mi[scores_mi['seleccionada']])

# Comparar rankings f_classif vs mutual_info
comparacion = scores.merge(scores_mi, on='feature', suffixes=('_f', '_mi'))
print("\nComparación de scores:")
print(comparacion[['feature', 'score', 'score_mi']])
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: SelectKBest con mutual_info_classif — información mutua con target.*

1. Comparar rankings f_classif vs mutual_info

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejemplo 4: SelectPercentile — top 20% de mejores features

```python
selector_pct = SelectPercentile(score_func=f_classif, percentile=20)
X_sel_pct = selector_pct.fit_transform(X_ventas, y_rotacion)

n_total = X_ventas.shape[1]
n_seleccionadas = X_sel_pct.shape[1]
print(f"Total features: {n_total}")
print(f"Seleccionadas (top 20%): {n_seleccionadas}")
print(f"Features: {list(X_ventas.columns[selector_pct.get_support()])}")

# Mostrar scores de todas
scores_pct = pd.DataFrame({
    'feature': X_ventas.columns,
    'score': selector_pct.scores_,
    'seleccionada': selector_pct.get_support()
}).sort_values('score', ascending=False)
print(scores_pct)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: SelectPercentile — top 20% de mejores features.*

1. Mostrar scores de todas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejemplo 5: RFE con LogisticRegression — eliminar recursivamente features

```python
X_ventas_bin = X_ventas.copy()
# Asegurar todas positivas para demostración
for col in X_ventas_bin.columns:
    X_ventas_bin[col] = X_ventas_bin[col] - X_ventas_bin[col].min() + 1e-6

estimator = LogisticRegression(max_iter=1000, random_state=42)
rfe = RFE(estimator=estimator, n_features_to_select=4, step=1)
X_rfe = rfe.fit_transform(X_ventas_bin, y_rotacion)

rfe_ranking = pd.DataFrame({
    'feature': X_ventas_bin.columns,
    'rank': rfe.ranking_,
    'selected': rfe.support_
}).sort_values('rank')

print("Ranking RFE (LogisticRegression):")
print(rfe_ranking)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: RFE con LogisticRegression — eliminar recursivamente features.*

1. Asegurar todas positivas para demostración

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejemplo 6: RFECV con RandomForest — número óptimo automático

```python
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rfecv = RFECV(
    estimator=rf,
    step=1,
    cv=StratifiedKFold(5),
    scoring='accuracy',
    min_features_to_select=2,
    n_jobs=-1,
)
X_rfecv = rfecv.fit_transform(X_ventas, y_rotacion)

print(f"Número óptimo de features: {rfecv.n_features_}")
print(f"Features seleccionadas: {list(X_ventas.columns[rfecv.support_])}")

# Grid search-like: accuracy vs número de features
plt.figure(figsize=(10, 5))
plt.plot(
    range(1, len(rfecv.cv_results_['mean_test_score']) + 1),
    rfecv.cv_results_['mean_test_score'],
    marker='o'
)
plt.axvline(rfecv.n_features_, color='r', linestyle='--',
            label=f'Óptimo: {rfecv.n_features_} features')
plt.xlabel('Número de features')
plt.ylabel('Accuracy CV')
plt.title('RFECV: Accuracy vs Número de Features')
plt.legend()
plt.grid(True)
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: RFECV con RandomForest — número óptimo automático.*

1. Grid search-like: accuracy vs número de features

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejemplo 7: SelectFromModel con RandomForest threshold='mean'

```python
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_ventas, y_rotacion)

sfm_mean = SelectFromModel(
    estimator=rf,
    threshold='mean',
    prefit=True
)
X_sfm_mean = sfm_mean.transform(X_ventas)

importancias = rf.feature_importances_
print(f"Threshold 'mean' = {importancias.mean():.4f}")
print(f"Features seleccionadas: {list(X_ventas.columns[sfm_mean.get_support()])}")
print(f"Número: {X_sfm_mean.shape[1]} de {X_ventas.shape[1]}")

# Mostrar importancias
imp_df = pd.DataFrame({
    'feature': X_ventas.columns,
    'importancia': importancias,
}).sort_values('importancia', ascending=False)
print(imp_df)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: SelectFromModel con RandomForest threshold='mean'.*

1. Mostrar importancias

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejemplo 8: SelectFromModel con threshold='median'

```python
sfm_median = SelectFromModel(
    estimator=rf,
    threshold='median',
    prefit=True
)
X_sfm_median = sfm_median.transform(X_ventas)

print(f"Threshold 'median' = {np.median(importancias):.4f}")
print(f"Features seleccionadas: {list(X_ventas.columns[sfm_median.get_support()])}")
print(f"Número: {X_sfm_median.shape[1]} de {X_ventas.shape[1]}")

# Comparar mean vs median
print(f"\n'mean' selecciona {X_sfm_mean.shape[1]} features")
print(f"'median' selecciona {X_sfm_median.shape[1]} features")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: SelectFromModel con threshold='median'.*

1. Comparar mean vs median

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejemplo 9: SequentialFeatureSelector forward — construir features uno a uno

```python
# Mismo dataset de clasificación binaria de ventas
sfs_forward = SequentialFeatureSelector(
    estimator=RandomForestClassifier(n_estimators=50, random_state=42),
    n_features_to_select=4,
    direction='forward',
    scoring='accuracy',
    cv=3,
    n_jobs=-1,
)
X_sfs_fwd = sfs_forward.fit_transform(X_ventas, y_rotacion)

print("SFS Forward:")
print(f"Features seleccionadas: {list(X_ventas.columns[sfs_forward.get_support()])}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: SequentialFeatureSelector forward — construir features uno a uno.*

1. Mismo dataset de clasificación binaria de ventas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejemplo 10: SequentialFeatureSelector backward — eliminar uno a uno

```python
sfs_back = SequentialFeatureSelector(
    estimator=RandomForestClassifier(n_estimators=50, random_state=42),
    n_features_to_select=4,
    direction='backward',
    scoring='accuracy',
    cv=3,
    n_jobs=-1,
)
X_sfs_back = sfs_back.fit_transform(X_ventas, y_rotacion)

print("SFS Backward:")
print(f"Features seleccionadas: {list(X_ventas.columns[sfs_back.get_support()])}")

# Comparar forward vs backward
print(f"\nForward: {list(X_ventas.columns[sfs_forward.get_support()])}")
print(f"Backward: {list(X_ventas.columns[sfs_back.get_support()])}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: SequentialFeatureSelector backward — eliminar uno a uno.*

1. Comparar forward vs backward

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejemplo 11: GenericUnivariateSelect mode='fpr' — controlar tasa de falsos positivos

```python
gus_fpr = GenericUnivariateSelect(
    score_func=f_classif,
    mode='fpr',
    param=0.05  # alpha = 0.05
)
X_gus_fpr = gus_fpr.fit_transform(X_ventas, y_rotacion)

print("GenericUnivariateSelect mode='fpr' (alpha=0.05):")
print(f"Features seleccionadas: {list(X_ventas.columns[gus_fpr.get_support()])}")
print(f"Número: {X_gus_fpr.shape[1]} de {X_ventas.shape[1]}")

# Probar otros modos
for mode, param in [('fdr', 0.05), ('fwe', 0.05)]:
    gus = GenericUnivariateSelect(
        score_func=f_classif, mode=mode, param=param
    )
    gus.fit(X_ventas, y_rotacion)
    print(f"mode={mode}: {gus.get_support().sum()} features seleccionadas")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: GenericUnivariateSelect mode='fpr' — controlar tasa de falsos positivos.*

1. Probar otros modos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejemplo 12: Comparar métodos — cuántas y cuáles features selecciona cada uno

```python
metodos = {}

# VarianceThreshold
vt = VarianceThreshold(threshold=0.01)
vt.fit(X_ventas)
metodos['VarianceThreshold'] = vt.get_support()

# SelectKBest f_classif
skb = SelectKBest(f_classif, k=4)
skb.fit(X_ventas, y_rotacion)
metodos['SelectKBest_f'] = skb.get_support()

# SelectKBest mutual_info
skb_mi = SelectKBest(mutual_info_classif, k=4)
skb_mi.fit(X_ventas, y_rotacion)
metodos['SelectKBest_MI'] = skb_mi.get_support()

# RFE
rfe = RFE(LogisticRegression(max_iter=1000), n_features_to_select=4)
rfe.fit(X_ventas, y_rotacion)
metodos['RFE_LR'] = rfe.get_support()

# RFECV
rfecv = RFECV(RandomForestClassifier(n_estimators=50, random_state=42),
              step=1, cv=3)
rfecv.fit(X_ventas, y_rotacion)
metodos['RFECV_RF'] = rfecv.get_support()

# SelectFromModel
sfm = SelectFromModel(
    RandomForestClassifier(n_estimators=50, random_state=42),
    threshold='mean'
)
sfm.fit(X_ventas, y_rotacion)
metodos['SelectFromModel'] = sfm.get_support()

comparativa = pd.DataFrame(
    metodos, index=X_ventas.columns
)
comparativa['total_si'] = comparativa.sum(axis=1)
print("Comparativa por método:\n")
print(comparativa)
print(f"\nResumen:")
for nombre, mask in metodos.items():
    print(f"  {nombre}: {mask.sum()} features → {list(X_ventas.columns[mask])}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: Comparar métodos — cuántas y cuáles features selecciona cada uno.*

1. VarianceThreshold
2. SelectKBest f_classif
3. SelectKBest mutual_info
4. RFE
5. RFECV
6. SelectFromModel

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejemplo 13: Feature selection dentro de Pipeline completo

```python
# Pipeline: Escalado → Selección → Clasificación
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('select', SelectKBest(score_func=f_classif, k=5)),
    ('clf', LogisticRegression(max_iter=1000, random_state=42)),
])

X_train, X_test, y_train, y_test = train_test_split(
    X_ventas, y_rotacion, test_size=0.3, random_state=42
)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print("Pipeline completo con feature selection:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Ver qué features se seleccionaron dentro del pipeline
selector = pipeline.named_steps['select']
print(f"Features seleccionadas: {list(X_ventas.columns[selector.get_support()])}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: Feature selection dentro de Pipeline completo.*

1. Pipeline: Escalado → Selección → Clasificación
2. Ver qué features se seleccionaron dentro del pipeline

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejemplo 14: Pipeline: SelectKBest + LogisticRegression con GridSearch

```python
pipeline_gs = Pipeline([
    ('scaler', StandardScaler()),
    ('select', SelectKBest(score_func=f_classif)),
    ('clf', LogisticRegression(max_iter=1000, random_state=42)),
])

param_grid = {
    'select__k': [2, 4, 6, 8],
    'clf__C': [0.01, 0.1, 1, 10],
}

gs = GridSearchCV(
    pipeline_gs, param_grid, cv=5, scoring='accuracy', n_jobs=-1
)
gs.fit(X_train, y_train)

print("GridSearchCV sobre Pipeline con feature selection:")
print(f"Mejores parámetros: {gs.best_params_}")
print(f"Mejor accuracy CV: {gs.best_score_:.4f}")
print(f"Accuracy en test: {gs.score(X_test, y_test):.4f}")

# Features seleccionadas por el mejor modelo
best_selector = gs.best_estimator_.named_steps['select']
print(f"k óptimo = {gs.best_params_['select__k']}")
print(f"Features: {list(X_ventas.columns[best_selector.get_support()])}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: Pipeline: SelectKBest + LogisticRegression con GridSearch.*

1. Features seleccionadas por el mejor modelo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejemplo 15: Validación — evaluar modelo con y sin feature selection

```python
# Sin feature selection
rf_full = RandomForestClassifier(n_estimators=100, random_state=42)
score_full = cross_val_score(rf_full, X_ventas, y_rotacion, cv=5,
                              scoring='accuracy')

# Con feature selection (SelectFromModel)
sfm_opt = SelectFromModel(
    RandomForestClassifier(n_estimators=100, random_state=42),
    threshold='mean'
)
X_sel_opt = sfm_opt.fit_transform(X_ventas, y_rotacion)
score_sel = cross_val_score(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X_sel_opt, y_rotacion, cv=5, scoring='accuracy'
)

print("Comparación CON vs SIN feature selection:")
print(f"Sin FS:   mean={score_full.mean():.4f} (±{score_full.std():.4f})")
print(f"Con FS:   mean={score_sel.mean():.4f} (±{score_sel.std():.4f})")

features_sel = list(X_ventas.columns[sfm_opt.get_support()])
print(f"\nFeatures originales: {X_ventas.shape[1]}")
print(f"Features seleccionadas: {len(features_sel)} → {features_sel}")

# Si el accuracy es similar con menos features, el modelo es más simple
if score_sel.mean() >= score_full.mean() * 0.95:
    print("✓ El modelo con FS es igual de bueno (o mejor) con menos features")
else:
    print("⚠ El modelo completo rinde mejor, pero usa más features")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: Validación — evaluar modelo con y sin feature selection.*

1. Sin feature selection
2. Con feature selection (SelectFromModel)
3. Si el accuracy es similar con menos features, el modelo es más simple

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejemplo 16: Visualizar scores de features seleccionados

```python
metodos_vis = ['f_classif', 'mutual_info_classif']
colores = ['steelblue', 'coral']

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, score_func, color in zip(axes, metodos_vis, colores):
    skb = SelectKBest(score_func=eval(score_func), k='all')
    skb.fit(X_ventas, y_rotacion)

    orden = np.argsort(skb.scores_)
    ax.barh(range(len(skb.scores_)), skb.scores_[orden], color=color)
    ax.set_yticks(range(len(skb.scores_)))
    ax.set_yticklabels(X_ventas.columns[orden])
    ax.set_title(f'Scores: {score_func}')
    ax.axvline(0, color='gray', linewidth=0.5)

plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 16: Visualizar scores de features seleccionados.*

1. `skb.fit(X_ventas, y_rotacion)` — Entrena el modelo con los datos de entrenamiento.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejemplo 17: Eliminar features altamente correlacionadas manualmente

```python
# Añadimos features correlacionadas para demostración
df_corr = X_ventas.copy()
df_corr['precio_dolar'] = df_corr['precio_promedio'] * 1.08 + np.random.normal(0, 0.5, n)
df_corr['unidades_kg'] = df_corr['unidades_vendidas'] / (df_corr['peso_kg'] + 1)
df_corr['volumen_rating'] = df_corr['volumen_cm3'] * df_corr['rating'] / 1000

# Calcular matriz de correlación
corr_matrix = df_corr.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Identificar pares con correlación > 0.95
threshold = 0.95
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

print(f"Columnas a eliminar por correlación > {threshold}: {to_drop}")
print(f"Shape original: {df_corr.shape[1]}")
df_reduced = df_corr.drop(columns=to_drop)
print(f"Shape después: {df_reduced.shape[1]}")

# Mostrar pares correlacionados
high_corr = [(col, row) for col in upper.columns
             for row in upper.index if upper.loc[row, col] > threshold]
print(f"\nPares altamente correlacionados:")
for c1, c2 in high_corr:
    print(f"  {c1} ↔ {c2}: r={corr_matrix.loc[c1, c2]:.3f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: Eliminar features altamente correlacionadas manualmente.*

1. Añadimos features correlacionadas para demostración
2. Calcular matriz de correlación
3. Identificar pares con correlación > 0.95
4. Mostrar pares correlacionados

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejemplo 18: Integrador — pipeline completo con selección automática + validación

```python
# === SIMULACIÓN DATOS INVENTARIOS ===
np.random.seed(42)
n = 2000
df_inv = pd.DataFrame({
    'costo_unitario': np.random.uniform(1, 200, n),
    'precio_venta': np.random.uniform(5, 500, n),
    'stock_actual': np.random.poisson(100, n),
    'stock_minimo': np.random.randint(5, 50, n),
    'stock_maximo': np.random.randint(100, 500, n),
    'rotacion_mensual': np.random.exponential(20, n),
    'dias_entrega_proveedor': np.random.randint(1, 30, n),
    'demanda_promedio': np.random.poisson(30, n),
    'merma_pct': np.random.uniform(0, 5, n),
    'lead_time_dias': np.random.exponential(7, n),
    'costo_almacenamiento': np.random.uniform(0.5, 5, n),
    'cantidad_pedido': np.random.poisson(50, n),
})
# Añadir features correlacionadas
df_inv['precio_sin_iva'] = df_inv['precio_venta'] / 1.21
df_inv['costo_logistico'] = (
    df_inv['costo_almacenamiento'] * df_inv['dias_entrega_proveedor'] / 30
)
df_inv['stock_seguridad'] = (
    df_inv['stock_maximo'] - df_inv['stock_minimo']
) * np.random.uniform(0.8, 1.2, n)

# Target: necesita reposición urgente (1) o no (0)
y_reposicion = (
    (df_inv['stock_actual'] < df_inv['stock_minimo'] * 1.2) &
    (df_inv['demanda_promedio'] > df_inv['rotacion_mensual'] * 0.5)
).astype(int)

X_inv = df_inv.copy()

# Pipeline con selección automática
from sklearn.ensemble import GradientBoostingClassifier

pipeline_final = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=42),
        threshold='mean'
    )),
    ('clf', GradientBoostingClassifier(n_estimators=100, random_state=42)),
])

X_train, X_test, y_train, y_test = train_test_split(
    X_inv, y_reposicion, test_size=0.3, random_state=42, stratify=y_reposicion
)

pipeline_final.fit(X_train, y_train)
y_pred_final = pipeline_final.predict(X_test)

print("=== Pipeline Integrador: Selección Automática ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_final):.4f}")
print(f"\nReporte de clasificación:")
print(classification_report(y_test, y_pred_final, target_names=['No urgente', 'Urgente']))

# Mostrar features seleccionadas
selector_final = pipeline_final.named_steps['selector']
features_finales = list(X_inv.columns[selector_final.get_support()])
print(f"Features originales: {X_inv.shape[1]}")
print(f"Features seleccionadas: {len(features_finales)} → {features_finales}")

# Importancias del modelo final
clf_final = pipeline_final.named_steps['clf']
imp_final = pd.DataFrame({
    'feature': features_finales,
    'importancia': clf_final.feature_importances_
}).sort_values('importancia', ascending=False)
print("\nImportancias (GradientBoosting):")
print(imp_final)

# Barplot
plt.figure(figsize=(10, 5))
plt.barh(imp_final['feature'], imp_final['importancia'], color='teal')
plt.xlabel('Importancia')
plt.title('Importancia de Features Seleccionadas')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador — pipeline completo con selección automática + validación.*

1. === SIMULACIÓN DATOS INVENTARIOS ===
2. Añadir features correlacionadas
3. Target: necesita reposición urgente (1) o no (0)
4. Pipeline con selección automática
5. Mostrar features seleccionadas
6. Importancias del modelo final
7. Barplot

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Resumen

| Método | Cuándo usarlo | Parámetros clave |
|--------|---------------|-------------------|
| **VarianceThreshold** | Features constantes o cuasi-constantes | `threshold` |
| **SelectKBest** | Ranking rápido, target conocido | `score_func`, `k` |
| **SelectPercentile** | % fijo de features | `percentile` |
| **RFE** | Importancia por modelo, número fijo | `estimator`, `n_features_to_select` |
| **RFECV** | Número óptimo automático | `estimator`, `cv`, `scoring` |
| **SelectFromModel** | Post-entrenamiento, threshold | `estimator`, `threshold` |
| **SequentialFeatureSelector** | Búsqueda greedy | `direction`, `scoring`, `cv` |
| **GenericUnivariateSelect** | Control estadístico | `mode`, `param` |

**Recomendación práctica:** RFECV con RandomForest suele dar buen balance entre automatización y calidad. Para pipelines productivos, SelectFromModel es rápido y efectivo.

---

## Ejercicios

1. Genera un dataset de compras con 15 features (incluye 2 constantes y 3 con varianza < 0.001). Aplica VarianceThreshold con diferentes thresholds y documenta cuántas features se eliminan en cada caso.

2. Usando el dataset de ventas del ejemplo 2, compara SelectKBest con `f_classif`, `mutual_info_classif` y `chi2` para k=4. ¿Cuáles son las diferencias en las features seleccionadas? ¿Por qué chi2 requiere features no negativas?

3. Para un dataset de inventarios con 12 features y target binario (reposición urgente), aplica RFE con LogisticRegression, RandomForest y SVM. Compara las features seleccionadas por cada estimador. ¿Coinciden?

4. Implementa RFECV con scoring='roc_auc' para el dataset de clasificación de productos. Grafica el AUC promedio vs número de features y determina el número óptimo automáticamente.

5. Crea dos modelos: (a) RandomForest con todas las features, (b) RandomForest con SelectFromModel threshold='1.25*mean'. Compara accuracy, precision y recall en test. ¿Cuántas features eliminaste?

6. Usa SequentialFeatureSelector en modo backward para un dataset de regresión (target: monto total de ventas). Evalúa con RMSE. ¿Qué features son las más importantes?

7. Para el ejercicio 6, aplica GenericUnivariateSelect con mode='fdr' y param=0.1. Compara las features seleccionadas con las de SFS backward usando un diagrama de Venn (puedes usar matplotlib para dibujar círculos).

8. **Integrador:** Diseña un experimento completo con el dataset de inventarios: (a) Divide en train/test, (b) Aplica 3 métodos de feature selection (RFE, SelectFromModel, RFECV), (c) Entrena un GradientBoostingClassifier con cada subconjunto, (d) Compara accuracy, número de features y tiempo de entrenamiento, (e) Concluye cuál método recomiendas para este problema y por qué.
