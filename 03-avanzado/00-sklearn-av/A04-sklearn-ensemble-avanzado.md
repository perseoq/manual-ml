# A04: Ensemble Avanzado — GradientBoosting, XGBoost, LightGBM, CatBoost, HistGB, Stacking para Ventas, Compras e Inventarios

## Introducción Teórica

Los **métodos ensemble avanzados** combinan múltiples modelos débiles (generalmente árboles) para crear un predictor robusto. Son el estado del arte para datos tabulares.

### Algoritmos cubiertos:

1. **GradientBoosting (sklearn)**: Construye árboles secuencialmente, cada uno corrige errores del anterior. Parámetros clave: learning_rate, n_estimators, subsample (stochastic GB), max_depth, validation_fraction, n_iter_no_change (early stopping).

2. **XGBoost**: Implementación optimizada con regularización L1/L2, soporte nativo para missing values, early stopping, y evaluación personalizada. Parámetros: gamma, colsample_bytree, reg_alpha/reg_lambda, scale_pos_weight.

3. **LightGBM**: Basado en histogramas, extremadamente rápido. Usa GOSS (muestreo) y EFB (features mutuamente exclusivas). Parámetros: num_leaves, min_child_samples, categorical_feature nativo.

4. **CatBoost**: Manejo automático de features categóricas (sin OneHotEncoding). Usa Ordered Boosting para reducir overfitting. Parámetros: cat_features, auto_class_weights, depth.

5. **HistGradientBoosting**: Versión de sklearn basada en histogramas, rápida para grandes datasets. Soporta categorical_features y early stopping nativo.

6. **StackingClassifier/Regressor**: Combina múltiples estimadores base con un meta-estimador. Parámetros: estimators, final_estimator, cv, stack_method, passthrough.

### Aplicación en negocio:
- **Ventas**: Predecir conversión con XGBoost (balanceo con scale_pos_weight).
- **Compras**: LightGBM con categorical_feature para países/proveedores.
- **Inventarios**: CatBoost sin preprocesamiento para clasificación ABC multiclase.

---

## Ejemplos

### Ejemplo 1: GradientBoosting — learning_rate vs n_estimators (trade-off)

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)
n = 2000
df = pd.DataFrame({
    'monto': np.random.exponential(500, n),
    'frecuencia': np.random.randint(1, 30, n),
    'recencia': np.random.randint(1, 365, n),
    'conversion': np.random.binomial(1, 0.6, n)
})
X = df[['monto', 'frecuencia', 'recencia']]
y = df['conversion']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

configs = [
    {'lr': 0.01, 'n': 500},
    {'lr': 0.05, 'n': 200},
    {'lr': 0.1, 'n': 100},
    {'lr': 0.2, 'n': 50},
]

print("learning_rate | n_estimators | Train Acc | Test Acc")
for cfg in configs:
    gb = GradientBoostingClassifier(
        n_estimators=cfg['n'], learning_rate=cfg['lr'],
        max_depth=3, random_state=42
    )
    gb.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, gb.predict(X_train))
    test_acc = accuracy_score(y_test, gb.predict(X_test))
    print(f"     {cfg['lr']:.2f}      |     {cfg['n']:3d}     |   {train_acc:.3f}  |   {test_acc:.3f}")
    print(f"  Gap: {train_acc - test_acc:.3f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: GradientBoosting — learning_rate vs n_estimators (trade-off).*

1. `import pandas as pd` — Importa las librerías necesarias para el análisis.
2. `import numpy as np` — Importa las librerías necesarias para el análisis.
3. `from sklearn.ensemble import GradientBoostingClassifier` — Importa las librerías necesarias para el análisis.
4. `from sklearn.model_selection import train_test_split` — Importa las librerías necesarias para el análisis.
5. `from sklearn.metrics import accuracy_score` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: GradientBoosting con early stopping

```python
from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

gb_es = GradientBoostingClassifier(
    n_estimators=1000, learning_rate=0.1, max_depth=3,
    validation_fraction=0.2, n_iter_no_change=10, tol=1e-4,
    random_state=42
)
gb_es.fit(X_tr, y_tr)

print(f"n_estimators reales (early stop): {gb_es.n_estimators_}")
print(f"Mejor iteración: {gb_es.n_estimators_}")
print(f"Train acc: {accuracy_score(y_tr, gb_es.predict(X_tr)):.3f}")
print(f"Val acc: {accuracy_score(y_val, gb_es.predict(X_val)):.3f}")
print(f"Test acc: {accuracy_score(y_test, gb_es.predict(X_test)):.3f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: GradientBoosting con early stopping.*

1. `from sklearn.model_selection import train_test_split` — Importa las librerías necesarias para el análisis.
2. `gb_es.fit(X_tr, y_tr)` — Entrena el modelo con los datos de entrenamiento.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: GradientBoosting con subsample (stochastic)

```python
configs_subsample = [0.5, 0.7, 1.0]

print("subsample | Train Acc | Test Acc")
for ss in configs_subsample:
    gb_s = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=3,
        subsample=ss, random_state=42
    )
    gb_s.fit(X_train, y_train)
    print(f"   {ss:.1f}    |   {accuracy_score(y_train, gb_s.predict(X_train)):.3f}  |   {accuracy_score(y_test, gb_s.predict(X_test)):.3f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: GradientBoosting con subsample (stochastic).*

1. `print("subsample | Train Acc | Test Acc")` — Muestra el resultado por pantalla.
2. `gb_s.fit(X_train, y_train)` — Entrena el modelo con los datos de entrenamiento.
3. `print(f"   {ss:.1f}    |   {accuracy_score(y_train, gb_s.predict(X_train)):.3f}  |   {accuracy_score(y_test, gb_s.predict(X_test)):.3f}")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: XGBoost — clasificar productos con eval_set

```python
try:
    import xgboost as xgb

    xgb_model = xgb.XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        gamma=0, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
        eval_metric='logloss', random_state=42
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    print(f"XGBoost Test Acc: {accuracy_score(y_test, xgb_model.predict(X_test)):.3f}")
    print(f"Best iteration: {xgb_model.best_iteration if hasattr(xgb_model, 'best_iteration') else 'N/A'}")

except ImportError:
    print("XGBoost no instalado. Instalar: pip install xgboost")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: XGBoost — clasificar productos con eval_set.*

1. `import xgboost as xgb` — Importa las librerías necesarias para el análisis.
2. `xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)` — Entrena el modelo con los datos de entrenamiento.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: XGBoost — feature importance (weight, gain, cover)

```python
try:
    import xgboost as xgb

    xgb_model2 = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    xgb_model2.fit(X_train, y_train)

    print("Tipo de importancia:")
    for imp_type in ['weight', 'gain', 'cover']:
        importance = xgb_model2.get_booster().get_score(importance_type=imp_type)
        print(f"\n  {imp_type}:")
        for feat, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"    {feat}: {score:.3f}")

except ImportError:
    print("XGBoost no instalado.")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: XGBoost — feature importance (weight, gain, cover).*

1. `import xgboost as xgb` — Importa las librerías necesarias para el análisis.
2. `xgb_model2.fit(X_train, y_train)` — Entrena el modelo con los datos de entrenamiento.
3. `print("Tipo de importancia:")` — Muestra el resultado por pantalla.
4. `print(f"\n  {imp_type}:")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: XGBoost — early_stopping_rounds

```python
try:
    import xgboost as xgb

    xgb_es = xgb.XGBClassifier(
        n_estimators=1000, max_depth=4, learning_rate=0.05,
        early_stopping_rounds=20, eval_metric='logloss', random_state=42
    )
    xgb_es.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    print(f"XGBoost early stopping - mejor iteración: {xgb_es.best_iteration}")
    print(f"Mejor score: {xgb_es.best_score:.4f}")
    print(f"Test acc: {accuracy_score(y_test, xgb_es.predict(X_test)):.3f}")

except ImportError:
    print("XGBoost no instalado.")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: XGBoost — early_stopping_rounds.*

1. `import xgboost as xgb` — Importa las librerías necesarias para el análisis.
2. `xgb_es.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)` — Entrena el modelo con los datos de entrenamiento.
3. `print(f"XGBoost early stopping - mejor iteración: {xgb_es.best_iteration}")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: LightGBM — clasificación con categorical_feature

```python
try:
    import lightgbm as lgb

    # Datos con features categóricas
    df_lgb = pd.DataFrame({
        'monto': np.random.exponential(500, 2000),
        'categoria': np.random.choice(['electronica', 'ropa', 'hogar', 'deportes'], 2000),
        'zona': np.random.choice(['norte', 'sur', 'este', 'oeste'], 2000),
        'conversion': np.random.binomial(1, 0.6, 2000)
    })

    # Codificar categóricas como enteros
    cat_cols = ['categoria', 'zona']
    for col in cat_cols:
        df_lgb[col] = df_lgb[col].astype('category').cat.codes

    X_lgb = df_lgb[['monto', 'categoria', 'zona']]
    y_lgb = df_lgb['conversion']
    Xl_tr, Xl_te, yl_tr, yl_te = train_test_split(X_lgb, y_lgb, test_size=0.2, random_state=42)

    lgb_model = lgb.LGBMClassifier(
        n_estimators=100, num_leaves=31, learning_rate=0.1,
        categorical_feature=[1, 2], random_state=42
    )
    lgb_model.fit(Xl_tr, yl_tr, eval_set=[(Xl_te, yl_te)], verbose=False)

    print(f"LightGBM Test Acc: {accuracy_score(yl_te, lgb_model.predict(Xl_te)):.3f}")
    print(f"Features importantes: {list(zip(X_lgb.columns, lgb_model.feature_importances_))}")

except ImportError:
    print("LightGBM no instalado. Instalar: pip install lightgbm")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: LightGBM — clasificación con categorical_feature.*

1. Datos con features categóricas
2. Codificar categóricas como enteros

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: LightGBM — comparar velocidad vs XGBoost

```python
try:
    import lightgbm as lgb
    import xgboost as xgb
    import time

    n_pts = 5000
    X_big = np.random.randn(n_pts, 20)
    y_big = (X_big[:, 0] + X_big[:, 1] > 0).astype(int)

    # LightGBM
    t0 = time.time()
    lgb_fast = lgb.LGBMClassifier(n_estimators=100, num_leaves=31, random_state=42, verbose=-1)
    lgb_fast.fit(X_big, y_big)
    lgb_time = time.time() - t0

    # XGBoost
    t0 = time.time()
    xgb_fast = xgb.XGBClassifier(n_estimators=100, max_depth=5, random_state=42, verbosity=0)
    xgb_fast.fit(X_big, y_big)
    xgb_time = time.time() - t0

    print(f"LightGBM: {lgb_time:.3f}s, Acc: {accuracy_score(y_big, lgb_fast.predict(X_big)):.3f}")
    print(f"XGBoost:  {xgb_time:.3f}s, Acc: {accuracy_score(y_big, xgb_fast.predict(X_big)):.3f}")
    print(f"LightGBM es {xgb_time / max(lgb_time, 0.001):.1f}x más rápido")

except ImportError:
    print("LightGBM o XGBoost no instalado.")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: LightGBM — comparar velocidad vs XGBoost.*

1. LightGBM
2. XGBoost

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: CatBoost — manejo automático de categóricas

```python
try:
    from catboost import CatBoostClassifier

    df_cat = pd.DataFrame({
        'monto': np.random.exponential(500, 2000),
        'categoria': np.random.choice(['electronica', 'ropa', 'hogar', 'deportes'], 2000),
        'zona': np.random.choice(['norte', 'sur', 'este', 'oeste'], 2000),
        'conversion': np.random.binomial(1, 0.6, 2000)
    })

    X_cat = df_cat[['monto', 'categoria', 'zona']]
    y_cat = df_cat['conversion']
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(X_cat, y_cat, test_size=0.2, random_state=42)

    cb_model = CatBoostClassifier(
        iterations=200, learning_rate=0.1, depth=6,
        cat_features=['categoria', 'zona'],
        verbose=False, random_seed=42
    )
    cb_model.fit(Xc_tr, yc_tr)

    print(f"CatBoost Test Acc: {accuracy_score(yc_te, cb_model.predict(Xc_te)):.3f}")
    print(f"Sin OneHotEncoding necesario!")
    print(f"Feature importances: {cb_model.get_feature_importance()}")

except ImportError:
    print("CatBoost no instalado. Instalar: pip install catboost")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: CatBoost — manejo automático de categóricas.*

1. `from catboost import CatBoostClassifier` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: CatBoost — class_weight para desbalanceo

```python
try:
    from catboost import CatBoostClassifier

    # Dataset desbalanceado
    df_cb_desb = pd.DataFrame({
        'monto': np.random.exponential(500, 2000),
        'categoria': np.random.choice(['A', 'B', 'C'], 2000),
        'conversion': np.random.binomial(1, 0.1, 2000)  # 10% positivos
    })

    X_cb_d = df_cb_desb[['monto', 'categoria']]
    y_cb_d = df_cb_desb['conversion']
    Xcd_tr, Xcd_te, ycd_tr, ycd_te = train_test_split(
        X_cb_d, y_cb_d, test_size=0.2, random_state=42, stratify=y_cb_d)

    cb_balanced = CatBoostClassifier(
        iterations=200, learning_rate=0.1, depth=6,
        cat_features=['categoria'], auto_class_weights='Balanced',
        verbose=False, random_seed=42
    )
    cb_balanced.fit(Xcd_tr, ycd_tr)

    from sklearn.metrics import f1_score
    y_pred_cb = cb_balanced.predict(Xcd_te)
    print(f"CatBoost balanceado - F1: {f1_score(ycd_te, y_pred_cb):.3f}")
    print(f"Accuracy: {accuracy_score(ycd_te, y_pred_cb):.3f}")
    print(f"Proporción positiva predicha: {y_pred_cb.mean():.3f} (real: {ycd_te.mean():.3f})")

except ImportError:
    print("CatBoost no instalado.")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: CatBoost — class_weight para desbalanceo.*

1. Dataset desbalanceado

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: HistGradientBoosting — rápida para grandes datasets

```python
from sklearn.ensemble import HistGradientBoostingClassifier

# Dataset grande
n_big = 10000
X_hgb = np.random.randn(n_big, 50)
y_hgb = (X_hgb[:, 0] + X_hgb[:, 2] > 0).astype(int)
Xh_tr, Xh_te, yh_tr, yh_te = train_test_split(X_hgb, y_hgb, test_size=0.2, random_state=42)

import time
t0 = time.time()
hgb = HistGradientBoostingClassifier(
    max_iter=200, max_leaf_nodes=31, learning_rate=0.1,
    max_depth=None, l2_regularization=0.0, early_stopping=True,
    random_state=42
)
hgb.fit(Xh_tr, yh_tr)
t_hgb = time.time() - t0

print(f"HistGradientBoosting: {t_hgb:.2f}s")
print(f"Test Acc: {accuracy_score(yh_te, hgb.predict(Xh_te)):.3f}")
print(f"Iteraciones reales: {hgb.n_iter_}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: HistGradientBoosting — rápida para grandes datasets.*

1. Dataset grande

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: HistGradientBoosting — categorical_features

```python
# HistGB también soporta categóricas
X_mix_hgb = pd.DataFrame({
    'monto': np.random.exponential(500, 2000),
    'categoria': np.random.choice([0, 1, 2, 3], 2000),  # ya codificada
    'zona': np.random.choice([0, 1, 2], 2000),
    'conversion': np.random.binomial(1, 0.6, 2000)
})

y_hgb_cat = X_mix_hgb.pop('conversion').values

hgb_cat = HistGradientBoostingClassifier(
    max_iter=100, max_leaf_nodes=31, categorical_features=[1, 2],  # índices de categóricas
    early_stopping=True, random_state=42
)
Xmh_tr, Xmh_te, ymh_tr, ymh_te = train_test_split(X_mix_hgb, y_hgb_cat, test_size=0.2, random_state=42)

hgb_cat.fit(Xmh_tr, ymh_tr)
print(f"HistGB con categóricas - Test Acc: {accuracy_score(ymh_te, hgb_cat.predict(Xmh_te)):.3f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: HistGradientBoosting — categorical_features.*

1. HistGB también soporta categóricas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: StackingClassifier — combinar RF + SVM + Logistic

```python
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
    ('lr', LogisticRegression(max_iter=1000, random_state=42))
]

meta_model = LogisticRegression(max_iter=1000, random_state=42)

stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=3,
    stack_method='predict_proba',
    passthrough=False
)

Xst_tr, Xst_te, yst_tr, yst_te = train_test_split(X, y, test_size=0.2, random_state=42)
stacking.fit(Xst_tr, yst_tr)

print(f"Stacking Test Acc: {accuracy_score(yst_te, stacking.predict(Xst_te)):.3f}")
print(f"Modelos base: {[name for name, _ in base_models]}")
print(f"Meta model: LogisticRegression")

# Comparar con modelos individuales
for name, model in base_models:
    model.fit(Xst_tr, yst_tr)
    acc = accuracy_score(yst_te, model.predict(Xst_te))
    print(f"  {name}: {acc:.3f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: StackingClassifier — combinar RF + SVM + Logistic.*

1. Comparar con modelos individuales

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: StackingRegressor — combinar RF + GB + Ridge

```python
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

# Datos de demanda
df_dem_st = pd.DataFrame({
    'precio': np.random.uniform(10, 500, 2000),
    'descuento': np.random.uniform(0, 0.5, 2000),
    'gasto_mk': np.random.exponential(1000, 2000),
    'demanda': np.random.poisson(50, 2000)
})
X_dem_st = df_dem_st[['precio', 'descuento', 'gasto_mk']]
y_dem_st = df_dem_st['demanda']

base_reg = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ('ridge', Ridge(alpha=1.0, random_state=42))
]

meta_reg = Ridge(alpha=0.5)

stacking_reg = StackingRegressor(
    estimators=base_reg,
    final_estimator=meta_reg,
    cv=3,
    passthrough=True  # incluir features originales
)

Xd_st_tr, Xd_st_te, yd_st_tr, yd_st_te = train_test_split(X_dem_st, y_dem_st, test_size=0.2, random_state=42)
stacking_reg.fit(Xd_st_tr, yd_st_tr)

from sklearn.metrics import r2_score
y_pred_st = stacking_reg.predict(Xd_st_te)
print(f"Stacking Regressor R²: {r2_score(yd_st_te, y_pred_st):.3f}")

for name, model in base_reg:
    model.fit(Xd_st_tr, yd_st_tr)
    r2 = r2_score(yd_st_te, model.predict(Xd_st_te))
    print(f"  {name}: R²={r2:.3f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: StackingRegressor — combinar RF + GB + Ridge.*

1. Datos de demanda

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: Calibración de probabilidades — CalibratedClassifierCV

```python
from sklearn.calibration import CalibratedClassifierCV

# Modelo sin calibrar
rf_base = RandomForestClassifier(n_estimators=100, random_state=42)
rf_base.fit(X_train, y_train)
probs_raw = rf_base.predict_proba(X_test)[:, 1]

# Modelo calibrado (Platt scaling)
rf_calibrated = CalibratedClassifierCV(rf_base, cv=3, method='sigmoid')
rf_calibrated.fit(X_train, y_train)
probs_cal = rf_calibrated.predict_proba(X_test)[:, 1]

from sklearn.metrics import brier_score_loss
brier_raw = brier_score_loss(y_test, probs_raw)
brier_cal = brier_score_loss(y_test, probs_cal)

print(f"Brier score raw: {brier_raw:.4f}")
print(f"Brier score calibrado: {brier_cal:.4f}")
print(f"Mejora: {(brier_raw - brier_cal) / brier_raw * 100:.1f}%")

# Decisión de negocio basada en probabilidades calibradas
print(f"\nCon umbral 0.7:")
print(f"  Ofertas raw: {(probs_raw > 0.7).sum()}")
print(f"  Ofertas cal: {(probs_cal > 0.7).sum()}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: Calibración de probabilidades — CalibratedClassifierCV.*

1. Modelo sin calibrar
2. Modelo calibrado (Platt scaling)
3. Decisión de negocio basada en probabilidades calibradas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 16: Comparar GB vs XGB vs LGBM vs CatBoost (accuracy y tiempo)

```python
import time

results_comp = []

# GradientBoosting
t0 = time.time()
gb_c = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)
gb_c.fit(X_train, y_train)
results_comp.append(('GB', time.time() - t0, accuracy_score(y_test, gb_c.predict(X_test))))

try:
    import xgboost as xgb
    t0 = time.time()
    xgb_c = xgb.XGBClassifier(n_estimators=200, max_depth=5, random_state=42, verbosity=0)
    xgb_c.fit(X_train, y_train)
    results_comp.append(('XGB', time.time() - t0, accuracy_score(y_test, xgb_c.predict(X_test))))
except ImportError:
    pass

try:
    import lightgbm as lgb
    t0 = time.time()
    lgb_c = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, max_depth=5, random_state=42, verbose=-1)
    lgb_c.fit(X_train, y_train)
    results_comp.append(('LGBM', time.time() - t0, accuracy_score(y_test, lgb_c.predict(X_test))))
except ImportError:
    pass

try:
    from catboost import CatBoostClassifier
    t0 = time.time()
    cb_c = CatBoostClassifier(iterations=200, depth=5, learning_rate=0.1, verbose=False, random_seed=42)
    cb_c.fit(X_train, y_train)
    results_comp.append(('CatB', time.time() - t0, accuracy_score(y_test, cb_c.predict(X_test))))
except ImportError:
    pass

print("Modelo | Tiempo | Accuracy")
for name, t, acc in results_comp:
    print(f" {name:5s} | {t:.2f}s | {acc:.3f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: Comparar GB vs XGB vs LGBM vs CatBoost (accuracy y tiempo).*

1. GradientBoosting

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 17: GridSearchCV para XGBoost (max_depth, learning_rate)

```python
try:
    import xgboost as xgb
    from sklearn.model_selection import GridSearchCV

    param_grid_xgb = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 1.0]
    }

    grid_xgb = GridSearchCV(
        xgb.XGBClassifier(n_estimators=100, random_state=42, verbosity=0, eval_metric='logloss'),
        param_grid_xgb, cv=3, scoring='accuracy', verbose=0
    )
    grid_xgb.fit(X_train, y_train)

    print(f"XGBoost GridSearch - Mejores params: {grid_xgb.best_params_}")
    print(f"Mejor score CV: {grid_xgb.best_score_:.3f}")
    print(f"Test acc: {accuracy_score(y_test, grid_xgb.predict(X_test)):.3f}")

    df_xgb = pd.DataFrame(grid_xgb.cv_results_)
    print(df_xgb[['params', 'mean_test_score']].to_string())

except ImportError:
    print("XGBoost no instalado.")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: GridSearchCV para XGBoost (max_depth, learning_rate).*

1. `import xgboost as xgb` — Importa las librerías necesarias para el análisis.
2. `from sklearn.model_selection import GridSearchCV` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 18: Integrador — ensemble campeón para predicción de demanda

```python
"""
Ensemble campeón: Stacking de múltiples gradient boosters para demanda.
Pipeline completo: preprocesamiento → stacking → evaluación
"""
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Datos
np.random.seed(42)
n = 3000
df_ens = pd.DataFrame({
    'precio': np.random.uniform(10, 500, n),
    'descuento': np.random.uniform(0, 0.5, n),
    'gasto_mk': np.random.exponential(1000, n),
    'temporada': np.random.choice(['alta', 'baja', 'media'], n),
    'cat': np.random.choice(['electronica', 'ropa', 'hogar'], n),
    'region': np.random.choice(['norte', 'sur', 'este'], n),
    'demanda': np.random.poisson(50, n)
})
X_ens = df_ens.drop('demanda', axis=1)
y_ens = df_ens['demanda']
Xe_tr, Xe_te, ye_tr, ye_te = train_test_split(X_ens, y_ens, test_size=0.2, random_state=42)

# Preprocesador
prep_ens = ColumnTransformer([
    ('num', StandardScaler(), ['precio', 'descuento', 'gasto_mk']),
    ('cat', OneHotEncoder(drop='first'), ['temporada', 'cat', 'region'])
])

# Stacking de regresores
base_ensemble = [
    ('gb', GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)),
    ('ridge', Ridge(alpha=1.0))
]

try:
    import xgboost as xgb
    base_ensemble.append(('xgb', xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, verbosity=0)))
except ImportError:
    pass

try:
    import lightgbm as lgb
    base_ensemble.append(('lgbm', lgb.LGBMRegressor(n_estimators=200, num_leaves=15, max_depth=4, random_state=42, verbose=-1)))
except ImportError:
    pass

stack_ens = StackingRegressor(
    estimators=base_ensemble,
    final_estimator=Ridge(alpha=0.3),
    cv=3, passthrough=False
)

pipe_ens = Pipeline([
    ('prep', prep_ens),
    ('stack', stack_ens)
])

pipe_ens.fit(Xe_tr, ye_tr)
y_ens_pred = pipe_ens.predict(Xe_te)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print("Ensemble Stacking - Predicción de Demanda:")
print(f"  R²:  {r2_score(ye_te, y_ens_pred):.3f}")
print(f"  MAE: {mean_absolute_error(ye_te, y_ens_pred):.1f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(ye_te, y_ens_pred)):.1f}")
print(f"  Modelos base: {[name for name, _ in base_ensemble]}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador — ensemble campeón para predicción de demanda.*

1. Datos
2. Preprocesador
3. Stacking de regresores

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejercicios

1. **GradientBoosting early stopping en ventas**: Implementa GradientBoosting con n_iter_no_change=10 para clasificar conversiones. Encuentra el número óptimo de iteraciones y compara con sin early stopping.

2. **XGBoost con scale_pos_weight**: Usa XGBoost para un dataset de ventas con 5% de conversiones. Ajusta scale_pos_weight para balancear. Compara F1 y ROC-AUC con/sin balanceo.

3. **LightGBM categorical_feature para zonas de venta**: Simula datos con 10 zonas geográficas y usa categorical_feature de LightGBM para clasificar éxito de venta. Compara con OneHotEncoding.

4. **CatBoost multiclase para ABC**: Usa CatBoost para clasificación multiclase (A/B/C) de inventarios. Usa auto_class_weights='Balanced'. Reporta matriz de confusión.

5. **HistGradientBoosting vs GradientBoosting en velocidad**: Genera dataset de 50,000 filas y 30 features. Compara HistGradientBoosting vs GradientBoosting en tiempo de entrenamiento y accuracy.

6. **StackingClassifier con 5 modelos base**: Combina RF, SVM, LR, XGB, LightGBM como base y LogisticRegression como meta. Evalúa si mejora al mejor modelo individual.

7. **Calibración para decisión de negocio**: Usa CalibratedClassifierCV en RandomForest para ventas. Define 3 umbrales de probabilidad (0.3, 0.5, 0.7) y calcula ganancia esperada para cada uno.

8. **Torneo de ensembles**: Compara GradientBoosting, XGBoost, LightGBM, CatBoost, HistGradientBoosting y Stacking en un dataset de predicción de demanda. Tabla con accuracy/tiempo. ¿Cuál recomiendas?

---

## Resumen

- **GradientBoosting**: Clásico, robusto, con early stopping nativo. Bueno para empezar.
- **XGBoost**: El estándar industrial. Regularización L1/L2, manejo nativo de missing values, scale_pos_weight para desbalanceo.
- **LightGBM**: El más rápido (GOSS/EFB). categorical_feature nativo sin preprocesamiento. Ideal para datasets grandes.
- **CatBoost**: Mejor manejo de categóricas (sin OneHotEncoding). Ordered Boosting reduce overfitting.
- **HistGradientBoosting**: La opción de sklearn, rápida y sin dependencias externas.
- **Stacking**: Combina fortalezas de múltiples modelos. Útil cuando modelos individuales tienen sesgos distintos.
- **CalibratedClassifierCV**: Ajusta probabilidades para mejor calibración (importante para decisiones de negocio con umbrales).
- En ventas/compras/inventarios, LightGBM y CatBoost destacan por su velocidad y manejo nativo de datos heterogéneos.
