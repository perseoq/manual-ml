# I15: Regresión Lineal y Regularización con scikit-learn

## Aplicación a Ventas, Compras e Inventarios

La regresión lineal es la base del aprendizaje supervisado para variables continuas. En el contexto comercial, se usa para **predecir cantidades vendidas, demanda de productos, precios óptimos, rotación de inventarios**, entre otros.

### Fundamentos Teóricos

**Regresión Lineal Simple:** `y = β₀ + β₁x + ε`
- β₀: intercepto (valor base)
- β₁: coeficiente (impacto de la variable x)
- ε: error aleatorio

**Regresión Lineal Múltiple:** `y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε`
- Cada βᵢ representa el cambio en y por cada unidad de xᵢ, manteniendo lo demás constante.

**Métrica de ajuste:** Mínimos Cuadrados Ordinarios (OLS) minimiza Σ(yᵢ - ŷᵢ)²

**Regularización:** Cuando hay muchas features o multicolinealidad, se añade un término de penalización:
- **Ridge (L2):** J(β) = Σ(yᵢ - ŷᵢ)² + α Σβⱼ² — encoge coeficientes pero no los lleva a cero
- **Lasso (L1):** J(β) = Σ(yᵢ - ŷᵢ)² + α Σ|βⱼ| — puede llevar coeficientes exactamente a cero (selección de features)
- **ElasticNet:** J(β) = Σ(yᵢ - ŷᵢ)² + α[l1_ratio·L1 + (1-l1_ratio)·L2] — combina ambos

**Regresores Robustos:** Resistentes a outliers
- **HuberRegressor:** combina pérdida cuadrática (errores pequeños) con lineal (errores grandes)
- **RANSAC:** elimina outliers iterativamente usando un subconjunto aleatorio
- **TheilSen:** usa medianas de pendientes entre pares de puntos
- **BayesianRidge:** enfoque probabilístico con priors sobre los coeficientes

**PolynomialFeatures:** transforma features a potencias e interacciones para capturar relaciones no lineales.

---

## Ejemplos Prácticos con sklearn

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import (LinearRegression, Ridge, RidgeCV, Lasso,
                                   LassoCV, ElasticNet, ElasticNetCV,
                                   HuberRegressor, RANSACRegressor,
                                   TheilSenRegressor, BayesianRidge)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# DATASET SIMULADO DE VENTAS
# ============================================================
np.random.seed(42)
n = 500

df = pd.DataFrame({
    'precio': np.random.uniform(10, 200, n),
    'descuento': np.random.uniform(0, 0.5, n),
    'día_semana': np.random.randint(0, 7, n),
    'es_finde': np.random.choice([0, 1], n, p=[0.7, 0.3]),
    'temperatura': np.random.uniform(5, 40, n),
    'stock_dias': np.random.randint(1, 90, n),
    'competencia_precio': np.random.uniform(8, 220, n),
    'gasto_publicidad': np.random.uniform(0, 5000, n),
})

# Variable target: cantidad_vendida con relación realista
df['cantidad_vendida'] = (
    100
    - 0.4 * df['precio']
    + 30 * df['descuento']
    + 5 * df['es_finde']
    + 0.02 * df['gasto_publicidad']
    - 0.1 * df['competencia_precio']
    + np.random.normal(0, 15, n)
)
df['cantidad_vendida'] = df['cantidad_vendida'].clip(0)

# DataFrame para clasificación (usado en ejemplos de clasificación)
df['rentable'] = (df['cantidad_vendida'] > 80).astype(int)

print('Dataset de ventas creado:')
print(df.head())
print(f'\nShape: {df.shape}')
print(f'Cantidad vendida - media: {df.cantidad_vendida.mean():.1f}, std: {df.cantidad_vendida.std():.1f}')
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
4. Variable target: cantidad_vendida con relación realista
5. DataFrame para clasificación (usado en ejemplos de clasificación)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 1: LinearRegression — Predecir cantidad vendida según precio

```python
X = df[['precio']]
y = df['cantidad_vendida']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print('=== Regresión Lineal Simple ===')
print(f'Coeficiente (pendiente): {lr.coef_[0]:.4f}')
print(f'Intercepto: {lr.intercept_:.4f}')
print(f'R² en test: {lr.score(X_test, y_test):.4f}')
print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}')
print(f'MAE: {mean_absolute_error(y_test, y_pred):.2f}')
print('\nInterpretación: Por cada unidad que sube el precio,')
print(f'la cantidad vendida disminuye en {abs(lr.coef_[0]):.2f} unidades.')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: LinearRegression — Predecir cantidad vendida según precio.*

1. `lr.fit(X_train, y_train)` — Entrena el modelo con los datos de entrenamiento.
2. `y_pred = lr.predict(X_test)` — Genera predicciones sobre nuevos datos.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 2: LinearRegression — Múltiples features (precio, descuento, día_semana)

```python
features = ['precio', 'descuento', 'día_semana', 'es_finde', 'temperatura']
X = df[features]
y = df['cantidad_vendida']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_multi = LinearRegression()
lr_multi.fit(X_train, y_train)

y_pred = lr_multi.predict(X_test)

print('=== Regresión Lineal Múltiple ===')
print('Coeficientes:')
for feat, coef in zip(features, lr_multi.coef_):
    print(f'  {feat}: {coef:.4f}')
print(f'Intercepto: {lr_multi.intercept_:.4f}')
print(f'R² en test: {lr_multi.score(X_test, y_test):.4f}')
print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: LinearRegression — Múltiples features (precio, descuento, día_semana).*

1. `lr_multi.fit(X_train, y_train)` — Entrena el modelo con los datos de entrenamiento.
2. `y_pred = lr_multi.predict(X_test)` — Genera predicciones sobre nuevos datos.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 3: Ridge — Regularización L2 con diferentes alpha

```python
X = df[['precio', 'descuento', 'día_semana', 'es_finde', 'temperatura',
        'stock_dias', 'competencia_precio', 'gasto_publicidad']]
y = df['cantidad_vendida']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar datos (importante para Ridge)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

alphas = [0.01, 0.1, 1, 10, 100, 1000]
print('=== Ridge con diferentes alpha ===')
print(f'{"alpha":<10} {"R² train":<12} {"R² test":<12} {"Coefs no cero":<15}')
print('-' * 50)
for a in alphas:
    ridge = Ridge(alpha=a)
    ridge.fit(X_train_scaled, y_train)
    r2_train = ridge.score(X_train_scaled, y_train)
    r2_test = ridge.score(X_test_scaled, y_test)
    nz = np.sum(np.abs(ridge.coef_) > 1e-10)
    print(f'{a:<10} {r2_train:<12.4f} {r2_test:<12.4f} {nz:<15}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: Ridge — Regularización L2 con diferentes alpha.*

1. Escalar datos (importante para Ridge)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 4: RidgeCV — Búsqueda del mejor alpha automática

```python
X = df[['precio', 'descuento', 'día_semana', 'es_finde', 'temperatura',
        'stock_dias', 'competencia_precio', 'gasto_publicidad']]
y = df['cantidad_vendida']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

alphas = np.logspace(-3, 3, 20)
ridge_cv = RidgeCV(alphas=alphas, store_cv_values=True)
ridge_cv.fit(X_train_scaled, y_train)

print('=== RidgeCV ===')
print(f'Mejor alpha: {ridge_cv.alpha_:.4f}')
print(f'R² en test: {ridge_cv.score(X_test_scaled, y_test):.4f}')
print(f'Coeficientes: {ridge_cv.coef_}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: RidgeCV — Búsqueda del mejor alpha automática.*

1. `X_test_scaled = scaler.transform(X_test)` — Aplica función preservando la forma original de los datos.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 5: Lasso — Selección de features (lleva coeficientes a cero)

```python
X = df[['precio', 'descuento', 'día_semana', 'es_finde', 'temperatura',
        'stock_dias', 'competencia_precio', 'gasto_publicidad']]
y = df['cantidad_vendida']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

alphas = [10, 1, 0.1, 0.01, 0.001]
print('=== Lasso: selección de features ===')
print(f'{"alpha":<10} {"R² test":<12} {"Coefs ≠ 0":<12} {"Features activas":<30}')
print('-' * 65)
for a in alphas:
    lasso = Lasso(alpha=a, max_iter=10000, tol=0.001)
    lasso.fit(X_train_scaled, y_train)
    r2 = lasso.score(X_test_scaled, y_test)
    coef_mask = np.abs(lasso.coef_) > 1e-10
    active_feats = [f for f, m in zip(X.columns, coef_mask) if m]
    print(f'{a:<10} {r2:<12.4f} {coef_mask.sum():<12} {", ".join(active_feats):<30}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: Lasso — Selección de features (lleva coeficientes a cero).*

1. `X_test_scaled = scaler.transform(X_test)` — Aplica función preservando la forma original de los datos.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 6: LassoCV — Búsqueda del mejor alpha automática

```python
X = df[['precio', 'descuento', 'día_semana', 'es_finde', 'temperatura',
        'stock_dias', 'competencia_precio', 'gasto_publicidad']]
y = df['cantidad_vendida']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

alphas = np.logspace(-3, 1, 50)
lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=10000, random_state=42)
lasso_cv.fit(X_train_scaled, y_train)

print('=== LassoCV ===')
print(f'Mejor alpha: {lasso_cv.alpha_:.4f}')
print(f'R² en test: {lasso_cv.score(X_test_scaled, y_test):.4f}')
print(f'Número de features con coef ≠ 0: {np.sum(lasso_cv.coef_ != 0)}')
print(f'Coeficientes: {lasso_cv.coef_}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: LassoCV — Búsqueda del mejor alpha automática.*

1. `X_test_scaled = scaler.transform(X_test)` — Aplica función preservando la forma original de los datos.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 7: ElasticNet — Combinación L1+L2 con diferentes l1_ratio

```python
X = df[['precio', 'descuento', 'día_semana', 'es_finde', 'temperatura',
        'stock_dias', 'competencia_precio', 'gasto_publicidad']]
y = df['cantidad_vendida']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
print('=== ElasticNet con diferentes l1_ratio ===')
print(f'{"l1_ratio":<12} {"R² test":<12} {"Coefs ≠ 0":<12}')
print('-' * 36)
for l1 in l1_ratios:
    en = ElasticNet(alpha=0.01, l1_ratio=l1, max_iter=10000, random_state=42)
    en.fit(X_train_scaled, y_train)
    r2 = en.score(X_test_scaled, y_test)
    nz = np.sum(np.abs(en.coef_) > 1e-10)
    print(f'{l1:<12.1f} {r2:<12.4f} {nz:<12}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: ElasticNet — Combinación L1+L2 con diferentes l1_ratio.*

1. `X_test_scaled = scaler.transform(X_test)` — Aplica función preservando la forma original de los datos.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 8: ElasticNetCV — Búsqueda automática de alpha y l1_ratio

```python
X = df[['precio', 'descuento', 'día_semana', 'es_finde', 'temperatura',
        'stock_dias', 'competencia_precio', 'gasto_publicidad']]
y = df['cantidad_vendida']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
alphas = np.logspace(-3, 1, 30)
en_cv = ElasticNetCV(alphas=alphas, l1_ratio=l1_ratios, cv=5,
                     max_iter=10000, random_state=42)
en_cv.fit(X_train_scaled, y_train)

print('=== ElasticNetCV ===')
print(f'Mejor alpha: {en_cv.alpha_:.4f}')
print(f'Mejor l1_ratio: {en_cv.l1_ratio_:.2f}')
print(f'R² en test: {en_cv.score(X_test_scaled, y_test):.4f}')
print(f'Número de features activas: {np.sum(np.abs(en_cv.coef_) > 1e-10)}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: ElasticNetCV — Búsqueda automática de alpha y l1_ratio.*

1. `X_test_scaled = scaler.transform(X_test)` — Aplica función preservando la forma original de los datos.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 9: HuberRegressor — Robusto a outliers en datos de ventas

```python
# Introducimos outliers artificiales
df_outliers = df.copy()
outlier_idx = np.random.choice(df.index, 20, replace=False)
df_outliers.loc[outlier_idx, 'cantidad_vendida'] *= 5  # multiplicamos x5

X = df_outliers[['precio', 'descuento', 'día_semana', 'es_finde']]
y = df_outliers['cantidad_vendida']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

huber = HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=100)
huber.fit(X_train_scaled, y_train)

print('=== HuberRegressor vs LinearRegression (con outliers) ===')
print(f'LinearRegression R²: {lr.score(X_test_scaled, y_test):.4f}')
print(f'HuberRegressor R²:   {huber.score(X_test_scaled, y_test):.4f}')
print(f'\nHuber parámetros: epsilon={huber.epsilon}, alpha={huber.alpha}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: HuberRegressor — Robusto a outliers en datos de ventas, compras o inventarios.*

1. Introducimos outliers artificiales

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 10: RANSACRegressor — Robusto eliminando outliers automáticamente

```python
X = df_outliers[['precio', 'descuento', 'día_semana', 'es_finde']]
y = df_outliers['cantidad_vendida']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ransac = RANSACRegressor(min_samples=0.5, residual_threshold=20, max_trials=100,
                         random_state=42)
ransac.fit(X_train_scaled, y_train)

inlier_mask = ransac.inlier_mask_
print(f'=== RANSACRegressor ===')
print(f'R² en test: {ransac.score(X_test_scaled, y_test):.4f}')
print(f'Muestras totales train: {len(y_train)}')
print(f'Inliers detectados: {inlier_mask.sum()}')
print(f'Outliers detectados: {(~inlier_mask).sum()} ({((~inlier_mask).sum()/len(y_train))*100:.1f}%)')
print(f'Coeficientes: {ransac.estimator_.coef_}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: RANSACRegressor — Robusto eliminando outliers automáticamente.*

1. `X_test_scaled = scaler.transform(X_test)` — Aplica función preservando la forma original de los datos.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 11: TheilSenRegressor — Robusto no paramétrico

```python
X = df_outliers[['precio', 'descuento', 'día_semana', 'es_finde']]
y = df_outliers['cantidad_vendida']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

theil = TheilSenRegressor(max_iter=300, n_subsamples=100, random_state=42)
theil.fit(X_train_scaled, y_train)

print('=== TheilSenRegressor ===')
print(f'R² en test: {theil.score(X_test_scaled, y_test):.4f}')
print(f'N_iteraciones: {theil.n_iter_}')
print(f'Coeficientes: {theil.coef_}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: TheilSenRegressor — Robusto no paramétrico.*

1. `X_test_scaled = scaler.transform(X_test)` — Aplica función preservando la forma original de los datos.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 12: BayesianRidge — Regresión bayesiana con intervalos de confianza

```python
X = df[['precio', 'descuento', 'día_semana', 'es_finde', 'temperatura']]
y = df['cantidad_vendida']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

br = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6,
                   fit_intercept=True, max_iter=300)
br.fit(X_train_scaled, y_train)

y_pred, y_std = br.predict(X_test_scaled, return_std=True)

print('=== BayesianRidge ===')
print(f'R² en test: {br.score(X_test_scaled, y_test):.4f}')
print(f'Alpha (precisión del ruido): {br.alpha_:.4f}')
print(f'Lambda (precisión de pesos): {br.lambda_:.4f}')
print(f'\nPrimeras 5 predicciones con intervalo de confianza del 95%:')
for i in range(5):
    lower = y_pred[i] - 1.96 * y_std[i]
    upper = y_pred[i] + 1.96 * y_std[i]
    real = y_test.values[i]
    print(f'  Pred: {y_pred[i]:.1f} ± {1.96*y_std[i]:.1f}  (IC95%: [{lower:.1f}, {upper:.1f}])  Real: {real:.1f}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: BayesianRidge — Regresión bayesiana con intervalos de confianza.*

1. `X_test_scaled = scaler.transform(X_test)` — Aplica función preservando la forma original de los datos.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 13: PolynomialFeatures — Regresión polinómica para relación no lineal

```python
# Simulamos relación no lineal: rendimiento de publicidad con saturación
np.random.seed(42)
X_poly = np.linspace(0, 100, 200).reshape(-1, 1)
y_poly = 50 + 3 * X_poly.ravel() - 0.03 * X_poly.ravel()**2 + np.random.normal(0, 20, 200)

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_poly, y_poly, test_size=0.2, random_state=42)

print('=== Regresión Polinómica ===')
for degree in [1, 2, 3, 4]:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_p)
    X_test_poly = poly.transform(X_test_p)

    lr_poly = LinearRegression()
    lr_poly.fit(X_train_poly, y_train_p)
    y_pred_poly = lr_poly.predict(X_test_poly)

    r2 = r2_score(y_test_p, y_pred_poly)
    rmse = np.sqrt(mean_squared_error(y_test_p, y_pred_poly))
    print(f'  Grado {degree}: R²={r2:.4f}, RMSE={rmse:.2f}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: PolynomialFeatures — Regresión polinómica para relación no lineal.*

1. Simulamos relación no lineal: rendimiento de publicidad con saturación

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 14: Comparar modelos — MSE, MAE, R² de cada uno

```python
X = df[['precio', 'descuento', 'día_semana', 'es_finde', 'temperatura',
        'stock_dias', 'competencia_precio', 'gasto_publicidad']]
y = df['cantidad_vendida']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

modelos = {
    'LinearRegression': LinearRegression(),
    'Ridge(alpha=1)': Ridge(alpha=1),
    'Lasso(alpha=0.1)': Lasso(alpha=0.1, max_iter=10000),
    'ElasticNet(α=0.1,l1=0.5)': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000),
    'HuberRegressor': HuberRegressor(epsilon=1.35),
    'RANSAC': RANSACRegressor(random_state=42),
    'TheilSen': TheilSenRegressor(random_state=42),
    'BayesianRidge': BayesianRidge(),
}

print('=== Comparación de Modelos ===')
print(f'{"Modelo":<30} {"R²":<10} {"RMSE":<10} {"MAE":<10}')
print('-' * 60)
for nombre, model in modelos.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    print(f'{nombre:<30} {r2:<10.4f} {rmse:<10.2f} {mae:<10.2f}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: Comparar modelos — MSE, MAE, R² de cada uno.*

1. `X_test_scaled = scaler.transform(X_test)` — Aplica función preservando la forma original de los datos.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 15: Residual plot — Analizar patrones en residuos

```python
X = df[['precio', 'descuento', 'día_semana', 'es_finde', 'temperatura',
        'stock_dias', 'competencia_precio', 'gasto_publicidad']]
y = df['cantidad_vendida']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred = lr.predict(X_test_scaled)
residuos = y_test - y_pred

print('=== Análisis de Residuos ===')
print(f'Media de residuos: {residuos.mean():.4f} (ideal: 0)')
print(f'Desviación estándar: {residuos.std():.4f}')
print(f'Sesgo (skewness): {residuos.skew():.4f} (ideal: 0)')
print(f'Curtosis: {residuos.kurtosis():.4f} (ideal ~3 para normal)')

# Prueba de normalidad (Shapiro-Wilk)
from scipy import stats
stat, p_value = stats.shapiro(residuos.sample(100))
print(f'Shapiro-Wilk p-valor: {p_value:.4f} (si > 0.05, residuos ∼ normal)')

# Durbin-Watson para autocorrelación
# (aproximación: buscar correlación entre residuo t y t-1)
resid_series = residuos.values
dw = np.sum(np.diff(resid_series)**2) / np.sum(resid_series**2)
print(f'Durbin-Watson: {dw:.4f} (ideal ≈ 2, rango [0,4])')

print('\nInterpretación:')
print('  - Si media ≈ 0: modelo no sesgado')
print('  - Si Durbin-Watson ≈ 2: no hay autocorrelación')
print('  - Si residuos ∼ normal: intervalo de confianza válido')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: Residual plot — Analizar patrones en residuos.*

1. Prueba de normalidad (Shapiro-Wilk)
2. Durbin-Watson para autocorrelación
3. (aproximación: buscar correlación entre residuo t y t-1)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---


**Nota:** Este análisis es puramente demostrativo. En la práctica, estos tests requieren supuestos más estrictos.

---

### Ejemplo 16: Interpretación de coeficientes — ¿Qué feature tiene más impacto?

```python
X = df[['precio', 'descuento', 'día_semana', 'es_finde', 'temperatura',
        'stock_dias', 'competencia_precio', 'gasto_publicidad']]
y = df['cantidad_vendida']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

# Importancia = |coef| * std(X) (efecto en unidades originales)
importancia = pd.DataFrame({
    'feature': X.columns,
    'coeficiente': lr.coef_,
    'abs_coef': np.abs(lr.coef_),
}).sort_values('abs_coef', ascending=False)

print('=== Importancia de Features (coeficientes estandarizados) ===')
print(importancia.to_string(index=False))
print(f'\nFeature más importante: {importancia.iloc[0]["feature"]}')
print(f'Feature menos importante: {importancia.iloc[-1]["feature"]}')
print('\nInterpretación (datos estandarizados):')
print('  El coeficiente representa el cambio en y por cada desviación estándar de x.')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: Interpretación de coeficientes — ¿Qué feature tiene más impacto?.*

1. Importancia = |coef| * std(X) (efecto en unidades originales)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 17: Pipeline — PolynomialFeatures + StandardScaler + Ridge

```python
# Relación no lineal: ventas en función de gasto en publicidad (con saturación)
np.random.seed(42)
n = 300
X_adv = np.random.uniform(0, 100, n).reshape(-1, 1)
y_adv = 30 + 5 * X_adv.ravel() - 0.04 * X_adv.ravel()**2 + np.random.normal(0, 15, n)

X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_adv, y_adv, test_size=0.2, random_state=42)

pipe = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=1))
])
pipe.fit(X_train_a, y_train_a)
y_pred_a = pipe.predict(X_test_a)

print('=== Pipeline: PolynomialFeatures + StandardScaler + Ridge ===')
print(f'R² en test: {pipe.score(X_test_a, y_test_a):.4f}')
print(f'RMSE: {np.sqrt(mean_squared_error(y_test_a, y_pred_a)):.2f}')
print(f'Nuevas features (poly degree=2): de 1 feature → {pipe.named_steps["poly"].n_output_features_} features')

# Comparar con lineal simple
lr_simple = LinearRegression()
lr_simple.fit(X_train_a, y_train_a)
print(f'\nComparación: LinearRegression R²={lr_simple.score(X_test_a, y_test_a):.4f}')
print(f'  Pipeline (polynomial) R²={pipe.score(X_test_a, y_test_a):.4f}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: Pipeline — PolynomialFeatures + StandardScaler + Ridge.*

1. Relación no lineal: ventas en función de gasto en publicidad (con saturación)
2. Comparar con lineal simple

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 18: Integrador — Predecir demanda semanal con 5 features

```python
# Simulamos datos semanales de demanda
np.random.seed(42)
n_semanas = 104  # 2 años
df_demanda = pd.DataFrame({
    'precio_promedio': np.random.uniform(15, 180, n_semanas),
    'descuento_promedio': np.random.uniform(0, 0.4, n_semanas),
    'gasto_publicidad': np.random.uniform(100, 8000, n_semanas),
    'temperatura_promedio': np.random.uniform(5, 35, n_semanas),
    'num_competidores': np.random.randint(2, 15, n_semanas),
})

df_demanda['demanda'] = (
    200
    - 0.8 * df_demanda['precio_promedio']
    + 80 * df_demanda['descuento_promedio']
    + 0.03 * df_demanda['gasto_publicidad']
    + 2 * df_demanda['temperatura_promedio']
    - 5 * df_demanda['num_competidores']
    + np.random.normal(0, 20, n_semanas)
)
df_demanda['demanda'] = df_demanda['demanda'].clip(20)

features_dem = ['precio_promedio', 'descuento_promedio', 'gasto_publicidad',
                'temperatura_promedio', 'num_competidores']
X_dem = df_demanda[features_dem]
y_dem = df_demanda['demanda']

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_dem, y_dem, test_size=0.2, random_state=42)

# Pipeline con búsqueda de alpha para Ridge
scaler = StandardScaler()
X_train_d_scaled = scaler.fit_transform(X_train_d)
X_test_d_scaled = scaler.transform(X_test_d)

ridge_cv_d = RidgeCV(alphas=np.logspace(-2, 2, 20), store_cv_values=True)
ridge_cv_d.fit(X_train_d_scaled, y_train_d)
y_pred_d = ridge_cv_d.predict(X_test_d_scaled)

lasso_cv_d = LassoCV(alphas=np.logspace(-3, 1, 30), cv=5, max_iter=10000, random_state=42)
lasso_cv_d.fit(X_train_d_scaled, y_train_d)
y_pred_l = lasso_cv_d.predict(X_test_d_scaled)

print('=== Integrador: Predicción de Demanda Semanal ===')
print(f'Ridge: alpha={ridge_cv_d.alpha_:.4f}, R²={ridge_cv_d.score(X_test_d_scaled, y_test_d):.4f}')
print(f'Lasso: alpha={lasso_cv_d.alpha_:.4f}, R²={lasso_cv_d.score(X_test_d_scaled, y_test_d):.4f}')
print(f'Features activas en Lasso: {np.sum(lasso_cv_d.coef_ != 0)} de {len(features_dem)}')

# Interpretación de resultados
print('\nCoeficientes Lasso (features importantes):')
for feat, coef in zip(features_dem, lasso_cv_d.coef_):
    if abs(coef) > 0.01:
        impacto = 'positivo' if coef > 0 else 'negativo'
        print(f'  {feat}: {coef:.4f} (impacto {impacto})')

print('\nRecomendaciones basadas en el modelo:')
coef_dict = dict(zip(features_dem, ridge_cv_d.coef_))
if coef_dict['precio_promedio'] < 0:
    print('  ✓ Reducir precio aumenta la demanda')
if coef_dict['descuento_promedio'] > 0:
    print('  ✓ Ofrecer descuentos impulsa ventas')
if coef_dict['gasto_publicidad'] > 0:
    print('  ✓ Invertir en publicidad es efectivo')
if coef_dict['num_competidores'] < 0:
    print('  ⚠ La competencia afecta negativamente: diferenciarse')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador — Predecir demanda semanal con 5 features.*

1. Simulamos datos semanales de demanda
2. Pipeline con búsqueda de alpha para Ridge
3. Interpretación de resultados

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Resumen

| Modelo | Cuándo usarlo | Parámetros clave |
|--------|--------------|------------------|
| **LinearRegression** | Datos limpios, sin multicolinealidad | — |
| **Ridge** | Muchas features correlacionadas | `alpha` (mayor = más regularización) |
| **Lasso** | Selección de features automática | `alpha`, `max_iter` |
| **ElasticNet** | Balance L1/L2 | `alpha`, `l1_ratio` |
| **HuberRegressor** | Outliers moderados | `epsilon` (transición L2→L1) |
| **RANSAC** | Muchos outliers | `min_samples`, `residual_threshold` |
| **TheilSen** | Outliers, no paramétrico | `n_subsamples` |
| **BayesianRidge** | Inferencia probabilística | `alpha_1`, `alpha_2`, `lambda_1`, `lambda_2` |
| **PolynomialFeatures** | Relaciones no lineales | `degree`, `interaction_only` |

**Recomendación:** Siempre estandarizar datos antes de Ridge/Lasso/ElasticNet. Usar validación cruzada para elegir `alpha`. Preferir modelos robustos si hay outliers.

---

## Ejercicios

1. Usando el dataset de ventas, entrena un `LinearRegression` con las features `precio`, `descuento`, `gasto_publicidad` y `temperatura`. Calcula R², RMSE y MAE. ¿Qué feature tiene el coeficiente más grande?

2. Aplica `RidgeCV` con `alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]` sobre 5 features. ¿Cuál es el mejor alpha? ¿Cómo cambia R² train vs test?

3. Usa `Lasso` con `alpha=0.5` sobre todas las features del dataset. ¿Cuántos coeficientes quedan en cero? ¿Qué features fueron seleccionadas?

4. Simula un dataset con 10 outliers extremos (multiplica `cantidad_vendida` × 10 en 10 filas). Compara `LinearRegression` vs `HuberRegressor` vs `RANSACRegressor`. ¿Cuál resiste mejor?

5. Usa `PolynomialFeatures(degree=3)` para modelar la relación entre `gasto_publicidad` y `cantidad_vendida`. Compara R² contra una regresión lineal simple. ¿Mejora significativamente?

6. Crea un `Pipeline` con `StandardScaler` + `ElasticNetCV`. Entrénalo con 8 features. ¿Qué `alpha` y `l1_ratio` selecciona automáticamente?

7. Genera un dataset sintético semanal de demanda (20 registros). Usa `BayesianRidge` para predecir la demanda de la semana 21 con intervalo de confianza del 95%.

8. Entrena todos los modelos vistos en 100 muestras de entrenamiento y compáralos con una tabla de R², RMSE, MAE. ¿Qué modelo gana? ¿Cuál es más rápido en entrenar?

---

*Teoría y práctica de regresión lineal con scikit-learn aplicada al dominio de ventas, compras e inventarios.*
