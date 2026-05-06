# A05: Modelos Lineales Avanzados para Ventas, Compras e Inventarios

## Introducción Teórica

Los **modelos lineales avanzados** extienden la regresión/logística clásica con regularización, robustez a outliers, estimación bayesiana, aprendizaje online, y soporte para múltiples tareas. Son ideales cuando la interpretabilidad y la velocidad son prioritarias.

### Familias de modelos:

1. **SGD-based (Stochastic Gradient Descent)**: Escalables a datasets masivos. Soporta múltiples pérdidas (hinge, log, huber, perceptron) y penalizaciones (L1, L2, ElasticNet).
2. **PassiveAggressive**: Algoritmo online que se adapta agresivamente cuando hay error de clasificación.
3. **Perceptron**: Clasificador lineal simple, precursor de las redes neuronales.
4. **Regularizados (Ridge/Lasso/ElasticNet)**: Con validación cruzada automática (CV) para encontrar alpha óptimo.
5. **MultiTask**: Features compartidas entre múltiples tareas de regresión.
6. **Bayesianos (ARD/BayesianRidge)**: Estiman distribuciones de parámetros, no solo puntos. Proporcionan intervalos de confianza.
7. **Robustos (RANSAC/TheilSen/Huber)**: Resistentes a outliers en los datos.
8. **QuantileRegressor**: Predice percentiles específicos (P10, P50, P90) para análisis de riesgo.

### Aplicación en negocio:
- **Ventas**: SGDClassifier para clasificar millones de transacciones en tiempo real.
- **Compras**: RANSACRegressor para precios de proveedores ignorando outliers.
- **Inventarios**: QuantileRegressor P90 para stock de seguridad.

---

## Ejemplos

### Ejemplo 1: SGDClassifier con loss='log' — logistic regression escalable

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n = 5000
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

sgd_log = SGDClassifier(loss='log', penalty='l2', alpha=0.0001, max_iter=1000, random_state=42)
sgd_log.fit(X_train_s, y_train)

print(f"SGDClassifier(log) Test Acc: {accuracy_score(y_test, sgd_log.predict(X_test_s)):.3f}")
print(f"Probabilidades (primeras 5): {sgd_log.predict_proba(X_test_s)[:5, 1].round(3)}")
print(f"Coeficientes: {sgd_log.coef_}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: SGDClassifier con loss='log' — logistic regression escalable.*

1. `import pandas as pd` — Importa las librerías necesarias para el análisis.
2. `import numpy as np` — Importa las librerías necesarias para el análisis.
3. `from sklearn.linear_model import SGDClassifier` — Importa las librerías necesarias para el análisis.
4. `from sklearn.model_selection import train_test_split` — Importa las librerías necesarias para el análisis.
5. `from sklearn.metrics import accuracy_score` — Importa las librerías necesarias para el análisis.
6. `from sklearn.preprocessing import StandardScaler` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: SGDClassifier con diferentes learning_rate

```python
learning_rates = ['optimal', 'invscaling', 'adaptive']

print("learning_rate | Test Acc | Iteraciones")
for lr in learning_rates:
    sgd = SGDClassifier(loss='log', penalty='l2', learning_rate=lr,
                        eta0=0.01, power_t=0.5, max_iter=1000, random_state=42)
    sgd.fit(X_train_s, y_train)
    acc = accuracy_score(y_test, sgd.predict(X_test_s))
    print(f"  {lr:12s} |   {acc:.3f}  |     {sgd.n_iter_}")

print("\n  'adaptive': reduce learning rate si loss no mejora")
print("  'invscaling': eta = eta0 / pow(t, power_t)")
print("  'optimal': eta = 1.0 / (alpha * (t + t0))")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: SGDClassifier con diferentes learning_rate.*

1. `print("learning_rate | Test Acc | Iteraciones")` — Muestra el resultado por pantalla.
2. `sgd.fit(X_train_s, y_train)` — Entrena el modelo con los datos de entrenamiento.
3. `acc = accuracy_score(y_test, sgd.predict(X_test_s))` — Genera predicciones sobre nuevos datos.
4. `print(f"  {lr:12s} |   {acc:.3f}  |     {sgd.n_iter_}")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: SGDClassifier con early_stopping en datos de ventas

```python
sgd_es = SGDClassifier(
    loss='log', penalty='l2', alpha=0.0001,
    max_iter=10000, early_stopping=True,
    validation_fraction=0.2, n_iter_no_change=5,
    random_state=42
)
sgd_es.fit(X_train_s, y_train)

print(f"SGD early stopping - iteraciones reales: {sgd_es.n_iter_}")
print(f"Test Acc: {accuracy_score(y_test, sgd_es.predict(X_test_s)):.3f}")
print(f"Pérdida final: {sgd_es.loss_function_}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: SGDClassifier con early_stopping en datos de ventas, compras o inventarios.*

1. `sgd_es.fit(X_train_s, y_train)` — Entrena el modelo con los datos de entrenamiento.
2. `print(f"SGD early stopping - iteraciones reales: {sgd_es.n_iter_}")` — Muestra el resultado por pantalla.
3. `print(f"Test Acc: {accuracy_score(y_test, sgd_es.predict(X_test_s)):.3f}")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: SGDRegressor con loss='huber' (robusto a outliers)

```python
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Datos con outliers (errores de tipeo en montos)
df_out = pd.DataFrame({
    'monto': np.random.exponential(500, 2000),
    'frecuencia': np.random.randint(1, 30, 2000),
})
df_out.loc[::50, 'monto'] = 50000  # outliers

y_out = df_out['monto'] * 0.3 + np.random.randn(2000) * 50
X_out = df_out[['monto', 'frecuencia']]

Xo_tr, Xo_te, yo_tr, yo_te = train_test_split(X_out, y_out, test_size=0.2, random_state=42)
scaler_out = StandardScaler()
Xo_tr_s = scaler_out.fit_transform(Xo_tr)
Xo_te_s = scaler_out.transform(Xo_te)

losses = ['squared_error', 'huber', 'epsilon_insensitive']
print("loss                | MAE  | R²")
for loss in losses:
    sgd_r = SGDRegressor(loss=loss, max_iter=1000, random_state=42, tol=1e-3)
    sgd_r.fit(Xo_tr_s, yo_tr)
    yp = sgd_r.predict(Xo_te_s)
    print(f"  {loss:20s} | {mean_absolute_error(yo_te, yp):.1f} | {r2_score(yo_te, yp):.3f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: SGDRegressor con loss='huber' (robusto a outliers).*

1. Datos con outliers (errores de tipeo en montos)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: PassiveAggressiveClassifier — online learning para clasificar productos

```python
from sklearn.linear_model import PassiveAggressiveClassifier

# Simular data streaming (lote por lote)
pac = PassiveAggressiveClassifier(C=1.0, max_iter=1000, random_state=42, tol=1e-3)

batch_size = 200
n_batches = 10
scores_stream = []

for batch in range(n_batches):
    idx_start = batch * batch_size
    idx_end = (batch + 1) * batch_size

    X_batch = X_train_s[idx_start:idx_end] if idx_end <= len(X_train_s) else X_train_s[idx_start:]
    y_batch = y_train[idx_start:idx_end] if idx_end <= len(y_train) else y_train[idx_start:]

    pac.partial_fit(X_batch, y_batch, classes=[0, 1])
    score = accuracy_score(y_test, pac.predict(X_test_s))
    scores_stream.append(score)
    print(f"Batch {batch+1}: Test Acc = {score:.3f}")

print(f"\nPassiveAggressive final: {scores_stream[-1]:.3f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: PassiveAggressiveClassifier — online learning para clasificar productos.*

1. Simular data streaming (lote por lote)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: Perceptron — clasificador lineal simple

```python
from sklearn.linear_model import Perceptron

perceptron = Perceptron(penalty='l2', alpha=0.0001, max_iter=1000, shuffle=True,
                        class_weight='balanced', early_stopping=True,
                        validation_fraction=0.2, n_iter_no_change=5, random_state=42)
perceptron.fit(X_train_s, y_train)

print(f"Perceptron Test Acc: {accuracy_score(y_test, perceptron.predict(X_test_s)):.3f}")
print(f"Iteraciones: {perceptron.n_iter_}")
print(f"Coeficientes: {perceptron.coef_}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: Perceptron — clasificador lineal simple.*

1. `from sklearn.linear_model import Perceptron` — Importa las librerías necesarias para el análisis.
2. `perceptron.fit(X_train_s, y_train)` — Entrena el modelo con los datos de entrenamiento.
3. `print(f"Perceptron Test Acc: {accuracy_score(y_test, perceptron.predict(X_test_s)):.3f}")` — Muestra el resultado por pantalla.
4. `print(f"Iteraciones: {perceptron.n_iter_}")` — Muestra el resultado por pantalla.
5. `print(f"Coeficientes: {perceptron.coef_}")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: RidgeCV — búsqueda automática de alpha

```python
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

# Datos de demanda
df_ridge = pd.DataFrame({
    'precio': np.random.uniform(10, 500, 2000),
    'descuento': np.random.uniform(0, 0.5, 2000),
    'gasto_mk': np.random.exponential(1000, 2000),
    'demanda': np.random.poisson(50, 2000)
})
X_r = df_ridge[['precio', 'descuento', 'gasto_mk']].values
y_r = df_ridge['demanda'].values

scaler_r = StandardScaler()
X_r_s = scaler_r.fit_transform(X_r)

alphas = np.logspace(-3, 3, 20)
ridge_cv = RidgeCV(alphas=alphas, cv=5, fit_intercept=True)
ridge_cv.fit(X_r_s, y_r)

print(f"RidgeCV - Mejor alpha: {ridge_cv.alpha_:.4f}")
print(f"R²: {ridge_cv.score(X_r_s, y_r):.3f}")
print(f"Coeficientes: {ridge_cv.coef_}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: RidgeCV — búsqueda automática de alpha.*

1. Datos de demanda

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: LassoCV — selección automática de features

```python
from sklearn.linear_model import LassoCV

# Features muchas, solo algunas relevantes
X_lasso = np.random.randn(2000, 30)
y_lasso = X_lasso[:, 0] * 3 + X_lasso[:, 5] * (-2) + X_lasso[:, 10] * 1.5 + np.random.randn(2000) * 0.5

Xl_tr, Xl_te, yl_tr, yl_te = train_test_split(X_lasso, y_lasso, test_size=0.2, random_state=42)

lasso_cv = LassoCV(alphas=np.logspace(-4, 1, 50), cv=5, max_iter=10000, random_state=42)
lasso_cv.fit(Xl_tr, yl_tr)

print(f"LassoCV - Mejor alpha: {lasso_cv.alpha_:.6f}")
print(f"Features activas (≠0): {(lasso_cv.coef_ != 0).sum()} de {len(lasso_cv.coef_)}")
print(f"R² test: {lasso_cv.score(Xl_te, yl_te):.3f}")

# Features seleccionadas
selected = np.where(lasso_cv.coef_ != 0)[0]
print(f"Features seleccionadas: {selected}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: LassoCV — selección automática de features.*

1. Features muchas, solo algunas relevantes
2. Features seleccionadas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: ElasticNetCV — búsqueda de alpha y l1_ratio

```python
from sklearn.linear_model import ElasticNetCV

elastic_cv = ElasticNetCV(
    alphas=np.logspace(-4, 1, 30),
    l1_ratio=np.linspace(0.1, 0.9, 9),
    cv=5, max_iter=10000, random_state=42
)
elastic_cv.fit(Xl_tr, yl_tr)

print(f"ElasticNetCV - Mejor alpha: {elastic_cv.alpha_:.6f}")
print(f"Mejor l1_ratio: {elastic_cv.l1_ratio_:.2f}")
print(f"Features activas: {(elastic_cv.coef_ != 0).sum()} de {len(elastic_cv.coef_)}")
print(f"R² test: {elastic_cv.score(Xl_te, yl_te):.3f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: ElasticNetCV — búsqueda de alpha y l1_ratio.*

1. `from sklearn.linear_model import ElasticNetCV` — Importa las librerías necesarias para el análisis.
2. `elastic_cv.fit(Xl_tr, yl_tr)` — Entrena el modelo con los datos de entrenamiento.
3. `print(f"ElasticNetCV - Mejor alpha: {elastic_cv.alpha_:.6f}")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: MultiTaskLasso — features compartidas entre múltiples tareas

```python
from sklearn.linear_model import MultiTaskLasso

# Múltiples tareas: predecir demanda en 3 regiones simultáneamente
n_tasks = 3
X_mt = np.random.randn(1000, 20)
y_mt = np.column_stack([
    X_mt[:, 0] * 2 + X_mt[:, 3] * (-1) + np.random.randn(1000) * 0.3,
    X_mt[:, 0] * 1.5 + X_mt[:, 7] * 2 + np.random.randn(1000) * 0.3,
    X_mt[:, 0] * (-1) + X_mt[:, 12] * 1.5 + np.random.randn(1000) * 0.3
])

Xmt_tr, Xmt_te, ymt_tr, ymt_te = train_test_split(X_mt, y_mt, test_size=0.2, random_state=42)

mt_lasso = MultiTaskLasso(alpha=0.1, max_iter=10000, random_state=42)
mt_lasso.fit(Xmt_tr, ymt_tr)

print(f"MultiTaskLasso R² (tarea 1): {mt_lasso.score(Xmt_te, ymt_te):.3f}")
print(f"Features activas por tarea:")
for i in range(n_tasks):
    n_active = (mt_lasso.coef_[i] != 0).sum()
    print(f"  Tarea {i+1}: {n_active} features activas")

# Features compartidas (todas ≠ 0)
shared = np.all(mt_lasso.coef_ != 0, axis=0).sum()
print(f"Features compartidas (todas tareas): {shared}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: MultiTaskLasso — features compartidas entre múltiples tareas.*

1. Múltiples tareas: predecir demanda en 3 regiones simultáneamente
2. Features compartidas (todas ≠ 0)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: OrthogonalMatchingPursuit — selección sparse de features

```python
from sklearn.linear_model import OrthogonalMatchingPursuit

omp = OrthogonalMatchingPursuit(n_nonzero_coefs=5, tol=None)
omp.fit(Xl_tr, yl_tr)

print(f"OMP - Features seleccionadas (n_nonzero=5): {np.where(omp.coef_ != 0)[0]}")
print(f"R² test: {omp.score(Xl_te, yl_te):.3f}")
print(f"Coeficientes no cero: {omp.coef_[omp.coef_ != 0]}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: OrthogonalMatchingPursuit — selección sparse de features.*

1. `from sklearn.linear_model import OrthogonalMatchingPursuit` — Importa las librerías necesarias para el análisis.
2. `omp.fit(Xl_tr, yl_tr)` — Entrena el modelo con los datos de entrenamiento.
3. `print(f"OMP - Features seleccionadas (n_nonzero=5): {np.where(omp.coef_ != 0)[0]}")` — Muestra el resultado por pantalla.
4. `print(f"R² test: {omp.score(Xl_te, yl_te):.3f}")` — Muestra el resultado por pantalla.
5. `print(f"Coeficientes no cero: {omp.coef_[omp.coef_ != 0]}")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: ARDRegression — regresión bayesiana automática

```python
from sklearn.linear_model import ARDRegression

ard = ARDRegression(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6,
                    n_iter=300, tol=1e-3, fit_intercept=True)
ard.fit(Xl_tr, yl_tr)

print(f"ARDRegression - R² test: {ard.score(Xl_te, yl_te):.3f}")
print(f"Coeficientes estimados: {ard.coef_[:10]}")
print(f"Alpha estimado: {ard.alpha_:.4f}")
print(f"Lambda estimado: {ard.lambda_:.4f}")
print(f"Iteraciones: {ard.n_iter_}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: ARDRegression — regresión bayesiana automática.*

1. `from sklearn.linear_model import ARDRegression` — Importa las librerías necesarias para el análisis.
2. `ard.fit(Xl_tr, yl_tr)` — Entrena el modelo con los datos de entrenamiento.
3. `print(f"ARDRegression - R² test: {ard.score(Xl_te, yl_te):.3f}")` — Muestra el resultado por pantalla.
4. `print(f"Coeficientes estimados: {ard.coef_[:10]}")` — Muestra el resultado por pantalla.
5. `print(f"Alpha estimado: {ard.alpha_:.4f}")` — Muestra el resultado por pantalla.
6. `print(f"Lambda estimado: {ard.lambda_:.4f}")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: BayesianRidge con intervalos de confianza

```python
from sklearn.linear_model import BayesianRidge

br = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6,
                   n_iter=300, tol=1e-3, fit_intercept=True, alpha_init=None, lambda_init=None)
br.fit(Xl_tr, yl_tr)

y_pred_br = br.predict(Xl_te, return_std=True)

print(f"BayesianRidge - R²: {r2_score(yl_te, y_pred_br[0]):.3f}")
print(f"\nPredicciones con intervalo de confianza:")
for i in range(5):
    mean, std = y_pred_br[0][i], y_pred_br[1][i]
    print(f"  Muestra {i+1}: {mean:.1f} ± {1.96*std:.1f} (IC 95%)")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: BayesianRidge con intervalos de confianza.*

1. `from sklearn.linear_model import BayesianRidge` — Importa las librerías necesarias para el análisis.
2. `br.fit(Xl_tr, yl_tr)` — Entrena el modelo con los datos de entrenamiento.
3. `y_pred_br = br.predict(Xl_te, return_std=True)` — Genera predicciones sobre nuevos datos.
4. `print(f"BayesianRidge - R²: {r2_score(yl_te, y_pred_br[0]):.3f}")` — Muestra el resultado por pantalla.
5. `print(f"\nPredicciones con intervalo de confianza:")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: QuantileRegressor — predecir percentiles (P10, P50, P90)

```python
from sklearn.linear_model import QuantileRegressor

# Datos de demanda con heterocedasticidad
n_q = 2000
X_q = np.random.uniform(10, 500, (n_q, 1))
y_q = 50 + 0.1 * X_q[:, 0] + X_q[:, 0] * np.random.randn(n_q) * 0.05

Xq_tr, Xq_te, yq_tr, yq_te = train_test_split(X_q, y_q, test_size=0.2, random_state=42)

quantiles = [0.1, 0.5, 0.9]
models_q = {}
for q in quantiles:
    qr = QuantileRegressor(quantile=q, alpha=0, solver='highs')
    qr.fit(Xq_tr, yq_tr)
    models_q[q] = qr

print("Percentil | Coef | Intercept")
for q in quantiles:
    m = models_q[q]
    print(f"   P{int(q*100):2d}     | {m.coef_[0]:.4f} | {m.intercept_:.2f}")

# Predicciones
print("\nPredicciones para precio=250:")
for q in quantiles:
    pred = models_q[q].predict([[250]])[0]
    print(f"  P{int(q*100)}: {pred:.1f} unidades")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: QuantileRegressor — predecir percentiles (P10, P50, P90).*

1. Datos de demanda con heterocedasticidad
2. Predicciones

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: RANSACRegressor — robusto eliminando outliers

```python
from sklearn.linear_model import RANSACRegressor

# Datos con 20% de outliers
X_rans = np.random.uniform(10, 500, (500, 1))
y_rans = 50 + 0.8 * X_rans[:, 0] + np.random.randn(500) * 30
outlier_idx = np.random.choice(500, 100, replace=False)
y_rans[outlier_idx] = y_rans[outlier_idx] + np.random.uniform(-500, 500, 100)

ransac = RANSACRegressor(min_samples=0.5, max_trials=100, residual_threshold=50, random_state=42)
ransac.fit(X_rans, y_rans)

print(f"RANSAC - R²: {ransac.score(X_rans, y_rans):.3f}")
print(f"Coef: {ransac.estimator_.coef_[0]:.4f}, Intercept: {ransac.estimator_.intercept_:.2f}")
print(f"Inliers: {ransac.inlier_mask_.sum()} de {len(y_rans)} ({ransac.inlier_mask_.mean()*100:.0f}%)")

# Comparar con OLS
from sklearn.linear_model import LinearRegression
ols = LinearRegression()
ols.fit(X_rans, y_rans)
print(f"OLS - Coef: {ols.coef_[0]:.4f}, Intercept: {ols.intercept_:.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: RANSACRegressor — robusto eliminando outliers.*

1. Datos con 20% de outliers
2. Comparar con OLS

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 16: TheilSenRegressor — no paramétrico robusto

```python
from sklearn.linear_model import TheilSenRegressor

theil = TheilSenRegressor(random_state=42, max_subpopulation=10000, n_subsamples=None)
theil.fit(X_rans, y_rans)

print(f"TheilSen - R²: {theil.score(X_rans, y_rans):.3f}")
print(f"Coef: {theil.coef_[0]:.4f}, Intercept: {theil.intercept_:.2f}")
print(f"Iteraciones: {theil.n_iter_}")
print("Basado en medianas de pendientes por pares — no paramétrico")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: TheilSenRegressor — no paramétrico robusto.*

1. `from sklearn.linear_model import TheilSenRegressor` — Importa las librerías necesarias para el análisis.
2. `theil.fit(X_rans, y_rans)` — Entrena el modelo con los datos de entrenamiento.
3. `print(f"TheilSen - R²: {theil.score(X_rans, y_rans):.3f}")` — Muestra el resultado por pantalla.
4. `print(f"Coef: {theil.coef_[0]:.4f}, Intercept: {theil.intercept_:.2f}")` — Muestra el resultado por pantalla.
5. `print(f"Iteraciones: {theil.n_iter_}")` — Muestra el resultado por pantalla.
6. `print("Basado en medianas de pendientes por pares — no paramétrico")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 17: HuberRegressor — ajuste robusto de errores

```python
from sklearn.linear_model import HuberRegressor

huber = HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=100, tol=1e-5)
huber.fit(X_rans, y_rans)

print(f"HuberRegressor - R²: {huber.score(X_rans, y_rans):.3f}")
print(f"Coef: {huber.coef_[0]:.4f}, Intercept: {huber.intercept_:.2f}")
print(f"Outlier weight: {huber.outlier_weight:.3f}")
print("Huber combina squared loss (inliers) + absolute loss (outliers)")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: HuberRegressor — ajuste robusto de errores.*

1. `from sklearn.linear_model import HuberRegressor` — Importa las librerías necesarias para el análisis.
2. `huber.fit(X_rans, y_rans)` — Entrena el modelo con los datos de entrenamiento.
3. `print(f"HuberRegressor - R²: {huber.score(X_rans, y_rans):.3f}")` — Muestra el resultado por pantalla.
4. `print(f"Coef: {huber.coef_[0]:.4f}, Intercept: {huber.intercept_:.2f}")` — Muestra el resultado por pantalla.
5. `print(f"Outlier weight: {huber.outlier_weight:.3f}")` — Muestra el resultado por pantalla.
6. `print("Huber combina squared loss (inliers) + absolute loss (outliers)")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 18: Integrador — comparar todos los modelos lineales en datos de ventas

```python
"""
Comparación exhaustiva de todos los modelos lineales en datos de ventas
con outliers para evaluar robustez.
"""

np.random.seed(42)
n_comp = 2000
X_comp = np.random.uniform(10, 500, (n_comp, 5))
y_comp = 100 + 2*X_comp[:, 0] - 1.5*X_comp[:, 1] + 0.5*X_comp[:, 2] + np.random.randn(n_comp) * 30

# Agregar 10% outliers
n_out = int(n_comp * 0.1)
out_idx = np.random.choice(n_comp, n_out, replace=False)
y_comp[out_idx] = y_comp[out_idx] * 3

Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(X_comp, y_comp, test_size=0.2, random_state=42)

scaler_comp = StandardScaler()
Xc_tr_s = scaler_comp.fit_transform(Xc_tr)
Xc_te_s = scaler_comp.transform(Xc_te)

models = {
    'RidgeCV': RidgeCV(alphas=[0.1, 1, 10], cv=3),
    'LassoCV': LassoCV(alphas=[0.01, 0.1, 1], cv=3, max_iter=5000),
    'ElasticNetCV': ElasticNetCV(alphas=[0.01, 0.1, 1], l1_ratio=[0.3, 0.7], cv=3, max_iter=5000),
    'BayesianRidge': BayesianRidge(),
    'ARDRegression': ARDRegression(),
    'SGDRegressor(l2)': SGDRegressor(loss='squared_error', penalty='l2', max_iter=1000, random_state=42),
    'SGDRegressor(huber)': SGDRegressor(loss='huber', max_iter=1000, random_state=42),
    'HuberRegressor': HuberRegressor(max_iter=100),
    'RANSACRegressor': RANSACRegressor(random_state=42),
    'TheilSenRegressor': TheilSenRegressor(random_state=42),
}

print(f"{'Modelo':25s} | {'R²':>6s} | {'MAE':>8s}")
print("-" * 44)
results_comp = []
for name, m in models.items():
    try:
        m.fit(Xc_tr_s, yc_tr)
        yp = m.predict(Xc_te_s)
        r2 = r2_score(yc_te, yp)
        mae = mean_absolute_error(yc_te, yp)
        results_comp.append((name, r2, mae))
        print(f" {name:25s} | {r2:6.3f} | {mae:8.1f}")
    except Exception as e:
        print(f" {name:25s} | ERROR: {str(e)[:30]}")

best_r2 = max(results_comp, key=lambda x: x[1])
best_mae = min(results_comp, key=lambda x: x[2])
print(f"\nMejor R²: {best_r2[0]} ({best_r2[1]:.3f})")
print(f"Mejor MAE: {best_mae[0]} ({best_mae[2]:.1f})")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador — comparar todos los modelos lineales en datos de ventas, compras o inventarios.*

1. Agregar 10% outliers

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejercicios

1. **SGDClassifier para streaming de ventas**: Simula 50 batches de ventas en tiempo real. Usa SGDClassifier con partial_fit para actualizar el modelo incrementalmente. Muestra accuracy acumulado.

2. **RidgeCV vs LassoCV vs ElasticNetCV para demanda**: Compara los 3 modelos con CV automática para predecir demanda de productos. Reporta alpha óptimo, features seleccionadas y R².

3. **QuantileRegressor para stock de seguridad**: Usa QuantileRegressor(quantile=0.9) para predecir el percentil 90 de demanda (stock de seguridad). Compara con regresión lineal estándar. ¿Cuál da más cobertura?

4. **RANSAC vs Huber vs TheilSen en compras**: Simula datos de compras con errores típicos (precios mal tipeados, devoluciones). Compara RANSAC, Huber y TheilSen en MAE y tiempo de ejecución.

5. **BayesianRidge con intervalos para presupuesto**: Usa BayesianRidge para predecir gasto mensual en compras. Genera predicciones con intervalo de confianza del 95% para planificar presupuesto.

6. **MultiTaskLasso para demanda por región**: Simula datos de demanda en 4 regiones. Usa MultiTaskLasso para identificar features compartidas entre regiones. Reporta las features más importantes globalmente.

7. **Perceptron vs SGDClassifier vs PassiveAggressive**: Compara los 3 clasificadores online en velocidad y accuracy para clasificar ventas exitosas. Usa un dataset de 10,000 muestras.

8. **Integrador: pipeline lineal para predicción de churn**: Diseña un pipeline con ColumnTransformer + selección de features (LassoCV) + clasificador lineal (SGDClassifier con early stopping). Optimiza hiperparámetros con GridSearchCV.

---

## Resumen

- **SGDClassifier/Regressor**: Escalables a millones de muestras. Soporte para múltiples pérdidas (log, hinge, huber) y penalizaciones (L1, L2, ElasticNet). Con early stopping para convergencia automática.
- **PassiveAggressive**: Ideal para aprendizaje online (streaming). Se adapta agresivamente a errores.
- **Perceptron**: Clasificador lineal simple con regularización y early stopping.
- **RidgeCV/LassoCV/ElasticNetCV**: Regularización con búsqueda automática de alpha. Lasso selecciona features (coeficientes exactamente cero). ElasticNet mezcla L1+L2.
- **MultiTaskLasso**: Features compartidas entre tareas de regresión múltiple.
- **OrdinaryMatchingPursuit**: Selección sparse de features con control exacto de no-ceros.
- **ARDRegression/BayesianRidge**: Enfoque bayesiano que estima distribuciones. Proporciona intervalos de confianza.
- **QuantileRegressor**: Predice percentiles específicos. Crucial para gestión de inventarios (stock de seguridad).
- **RANSAC/TheilSen/Huber**: Robustos a outliers. RANSAC identifica y excluye outliers explícitamente. Huber mezcla pérdida cuadrática y absoluta.
- En negocio de ventas/compras/inventarios, la combinación de interpretabilidad (modelos lineales) + robustez (outliers) + escalabilidad (SGD) es invaluable para producción.
