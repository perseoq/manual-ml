# B30 — Introducción a `scikit-learn`: model selection y métricas

`scikit-learn` (sklearn) es la biblioteca estándar de machine learning en Python. Este módulo cubre dos pilares fundamentales: **model selection** (cómo dividir datos, validar modelos) y **métricas** (cómo medir rendimiento). Todo aplicado al contexto de ventas.

## Tabla de funciones cubiertas

| Función / Clase | Propósito | Categoría |
|---|---|---|
| `train_test_split` | Dividir datos en entrenamiento/prueba | Model selection |
| `cross_val_score` | Validación cruzada | Model selection |
| `KFold` | Particionamiento K-Folds | Model selection |
| `StratifiedKFold` | K-Folds estratificado (clasificación) | Model selection |
| `ShuffleSplit` | Muestreo aleatorio repetido | Model selection |
| `RepeatedKFold` | K-Folds repetido | Model selection |
| `accuracy_score` | Precisión de clasificación | Métrica clasificación |
| `confusion_matrix` | Matriz de confusión | Métrica clasificación |
| `classification_report` | Reporte completo de clasificación | Métrica clasificación |
| `precision_score` | Precisión (TP / TP+FP) | Métrica clasificación |
| `recall_score` | Recall (TP / TP+FN) | Métrica clasificación |
| `f1_score` | Media armónica precision+recall | Métrica clasificación |
| `roc_auc_score` | Área bajo la curva ROC | Métrica clasificación |
| `roc_curve` | Puntos de la curva ROC | Métrica clasificación |
| `ConfusionMatrixDisplay` | Visualización de matriz de confusión | Métrica clasificación |
| `RocCurveDisplay` | Visualización de curva ROC | Métrica clasificación |
| `mean_squared_error` | Error cuadrático medio (MSE) | Métrica regresión |
| `mean_absolute_error` | Error absoluto medio (MAE) | Métrica regresión |
| `r2_score` | Coeficiente de determinación R² | Métrica regresión |
| `explained_variance_score` | Varianza explicada | Métrica regresión |
| `max_error` | Error máximo residual | Métrica regresión |
| `mean_absolute_percentage_error` | Error porcentual absoluto medio (MAPE) | Métrica regresión |

---

## Configuración inicial

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.model_selection import StratifiedKFold, ShuffleSplit, RepeatedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_percentage_error
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

ventas = pd.read_csv("../datos/ventas.csv")
inventario = pd.read_csv("../datos/inventario.csv")
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Configuración inicial.*

1. `import pandas as pd` — Importa las librerías necesarias para el análisis.
2. `import numpy as np` — Importa las librerías necesarias para el análisis.
3. `from sklearn.model_selection import train_test_split, cross_val_score, KFold` — Importa las librerías necesarias para el análisis.
4. `from sklearn.model_selection import StratifiedKFold, ShuffleSplit, RepeatedKFold` — Importa las librerías necesarias para el análisis.
5. `from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score` — Importa las librerías necesarias para el análisis.
6. `from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve` — Importa las librerías necesarias para el análisis.
7. `from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score` — Importa las librerías necesarias para el análisis.
8. `from sklearn.metrics import explained_variance_score, max_error, mean_absolute_percentage_error` — Importa las librerías necesarias para el análisis.
9. `from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay` — Importa las librerías necesarias para el análisis.
10. `from sklearn.dummy import DummyRegressor, DummyClassifier` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejemplo 1 — `train_test_split` básico: X = precio, y = cantidad

```python
X = ventas[["precio_unitario"]].values
y = ventas["cantidad"].values

X_ent, X_prue, y_ent, y_prue = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"X entrenamiento: {X_ent.shape}, X prueba: {X_prue.shape}")
print(f"y entrenamiento: {y_ent.shape}, y prueba: {y_prue.shape}")

# Salida:
# X entrenamiento: (930, 1), X prueba: (400, 1)
# y entrenamiento: (930,), y prueba: (400,)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1 — `train_test_split` básico: X = precio, y = cantidad.*

1. Salida:
2. X entrenamiento: (930, 1), X prueba: (400, 1)
3. y entrenamiento: (930,), y prueba: (400,)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** Dividimos precio vs cantidad en 70% entrenamiento (930 registros) y 30% prueba (400). `random_state=42` asegura reproducibilidad. Ahora podemos entrenar un modelo en X_ent→y_ent y evaluar en datos no vistos.

---

## Ejemplo 2 — `train_test_split` con `stratify` para clasificación desbalanceada

```python
# Clasificación binaria: margen alto (> 30%) vs bajo (<= 30%)
ventas["margen_alto"] = (ventas["margen_pct"] > 30).astype(int)

X = ventas[["precio_unitario", "descuento"]].values
y = ventas["margen_alto"].values

X_ent, X_prue, y_ent, y_prue = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

print(f"Proporción en y original:  {y.mean():.3f}")
print(f"Proporción en y_entrenamiento: {y_ent.mean():.3f}")
print(f"Proporción en y_prueba:    {y_prue.mean():.3f}")

# Salida:
# Proporción en y original:  0.681
# Proporción en y_entrenamiento: 0.681
# Proporción en y_prueba:    0.680
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2 — `train_test_split` con `stratify` para clasificación desbalanceada.*

1. Clasificación binaria: margen alto (> 30%) vs bajo (<= 30%)
2. Salida:
3. Proporción en y original:  0.681
4. Proporción en y_entrenamiento: 0.681
5. Proporción en y_prueba:    0.680

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** `stratify=y` mantiene la misma proporción de clases (68% margen alto) en ambas particiones. Sin stratify, una partición podría tener 75% y otra 60%, sesgando la evaluación.

---

## Ejemplo 3 — `train_test_split` con `random_state` para reproducibilidad

```python
X = ventas[["precio_unitario"]].values
y = ventas["cantidad"].values

X1, _, y1, _ = train_test_split(X, y, test_size=0.3, random_state=42)
X2, _, y2, _ = train_test_split(X, y, test_size=0.3, random_state=42)
X3, _, y3, _ = train_test_split(X, y, test_size=0.3, random_state=99)

print(f"Misma semilla (42):  índices iguales = {(y1 == y2).all()}")
print(f"Distinta semilla (99): índices iguales = {(y1 == y3).all()}")

# Salida:
# Misma semilla (42):  índices iguales = True
# Distinta semilla (99): índices iguales = False
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3 — `train_test_split` con `random_state` para reproducibilidad.*

1. Salida:
2. Misma semilla (42):  índices iguales = True
3. Distinta semilla (99): índices iguales = False

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** `random_state` fijo = misma partición siempre. Sin él, cada ejecución produce resultados distintos, imposibilitando la depuración y comparación de modelos.

---

## Ejemplo 4 — `cross_val_score` con `DummyRegressor` (baseline)

```python
X = ventas[["precio_unitario"]].values
y = ventas["cantidad"].values

modelo = DummyRegressor(strategy="mean")
puntajes = cross_val_score(modelo, X, y, cv=5, scoring="r2")

print(f"R² por fold: {puntajes}")
print(f"R² promedio: {puntajes.mean():.4f}")

# Salida:
# R² por fold: [-0.0032 -0.0058 -0.0006 -0.0046 -0.0031]
# R² promedio: -0.0035
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4 — `cross_val_score` con `DummyRegressor` (baseline).*

1. Salida:
2. R² por fold: [-0.0032 -0.0058 -0.0006 -0.0046 -0.0031]
3. R² promedio: -0.0035

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** DummyRegressor predice siempre la media. R² ≈ 0 (en realidad ligeramente negativo por fluctuaciones de los folds). Este es el **baseline**: cualquier modelo útil debe superar este R².

---

## Ejemplo 5 — `cross_val_score` con `cv=5` y `scoring="r2"`

```python
X = ventas[["precio_unitario"]].values
y = ventas["cantidad"].values

modelo = LinearRegression()
puntajes = cross_val_score(modelo, X, y, cv=5, scoring="r2")

print(f"R² por fold (5 folds): {puntajes}")
print(f"R² promedio: {puntajes.mean():.4f} +/- {puntajes.std():.4f}")

# Salida:
# R² por fold (5 folds): [0.0123 0.0087 0.0156 0.0112 0.0095]
# R² promedio: 0.0114 +/- 0.0026
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5 — `cross_val_score` con `cv=5` y `scoring="r2"`.*

1. Salida:
2. R² por fold (5 folds): [0.0123 0.0087 0.0156 0.0112 0.0095]
3. R² promedio: 0.0114 +/- 0.0026

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** Con validación cruzada de 5 folds, el R² es apenas 0.011 — el precio solo explica ~1% de la variación en cantidad. La desviación estándar baja (0.0026) sugiere que el rendimiento es consistente entre folds.

---

## Ejemplo 6 — `KFold` básico (5 folds) manual

```python
X = ventas[["precio_unitario", "descuento"]].values
y = ventas["cantidad"].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)
modelo = LinearRegression()
puntajes = []

for fold, (idx_ent, idx_prue) in enumerate(kf.split(X), 1):
    modelo.fit(X[idx_ent], y[idx_ent])
    r2 = modelo.score(X[idx_prue], y[idx_prue])
    puntajes.append(r2)
    print(f"Fold {fold}: R² = {r2:.4f}")

print(f"\nR² promedio: {np.mean(puntajes):.4f}")

# Salida:
# Fold 1: R² = 0.0142
# Fold 2: R² = 0.0098
# Fold 3: R² = 0.0121
# Fold 4: R² = 0.0105
# Fold 5: R² = 0.0133
#
# R² promedio: 0.0120
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6 — `KFold` básico (5 folds) manual.*

1. Salida:
2. Fold 1: R² = 0.0142
3. Fold 2: R² = 0.0098
4. Fold 3: R² = 0.0121
5. Fold 4: R² = 0.0105
6. Fold 5: R² = 0.0133
7. R² promedio: 0.0120

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** KFold divide en 5 folds y rotamos cuál es prueba. `shuffle=True` mezcla antes de partir para evitar sesgos por orden de filas. Cada fold entrena en 80% y prueba en 20%.

---

## Ejemplo 7 — `StratifiedKFold` para clasificación (proveedor bueno/malo)

```python
ventas["proveedor_bueno"] = (ventas["margen_pct"] > ventas["margen_pct"].median()).astype(int)

X = ventas[["precio_unitario", "descuento", "cantidad"]].values
y = ventas["proveedor_bueno"].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (idx_ent, idx_prue) in enumerate(skf.split(X, y), 1):
    y_ent, y_prue = y[idx_ent], y[idx_prue]
    print(f"Fold {fold}: % clase 1 en entrenamiento = {y_ent.mean():.3f}, en prueba = {y_prue.mean():.3f}")

# Salida:
# Fold 1: % clase 1 en entrenamiento = 0.500, en prueba = 0.500
# Fold 2: % clase 1 en entrenamiento = 0.500, en prueba = 0.500
# Fold 3: % clase 1 en entrenamiento = 0.500, en prueba = 0.500
# Fold 4: % clase 1 en entrenamiento = 0.500, en prueba = 0.500
# Fold 5: % clase 1 en entrenamiento = 0.500, en prueba = 0.500
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7 — `StratifiedKFold` para clasificación (proveedor bueno/malo).*

1. Salida:
2. Fold 1: % clase 1 en entrenamiento = 0.500, en prueba = 0.500
3. Fold 2: % clase 1 en entrenamiento = 0.500, en prueba = 0.500
4. Fold 3: % clase 1 en entrenamiento = 0.500, en prueba = 0.500
5. Fold 4: % clase 1 en entrenamiento = 0.500, en prueba = 0.500
6. Fold 5: % clase 1 en entrenamiento = 0.500, en prueba = 0.500

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** `StratifiedKFold` mantiene la proporción de clases (50/50) en cada fold. Esencial cuando las clases están desbalanceadas. KFold normal podría crear un fold con 60/40.

---

## Ejemplo 8 — `accuracy_score`: precisión de clasificación

```python
y_real = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
y_pred = np.array([1, 0, 1, 0, 0, 1, 0, 1, 1, 0])

precision = accuracy_score(y_real, y_pred)
print(f"Accuracy: {precision:.2f} ({sum(y_real == y_pred)}/{len(y_real)} correctas)")

# Salida:
# Accuracy: 0.80 (8/10 correctas)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8 — `accuracy_score`: precisión de clasificación.*

1. Salida:
2. Accuracy: 0.80 (8/10 correctas)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** Accuracy = (VP + VN) / total. Dos errores: predijo 0 cuando era 1 (FN) y 1 cuando era 0 (FP). 80% parece bueno, pero si hubiera 90% de una clase, un clasificador tonto daría 90%.

---

## Ejemplo 9 — `confusion_matrix`: matriz de confusión

```python
y_real = ventas["margen_alto"].values
np.random.seed(42)
y_pred = (np.random.rand(len(y_real)) > 0.3).astype(int)  # clasificador aleatorio

cm = confusion_matrix(y_real, y_pred)
tn, fp, fn, tp = cm.ravel()

print("Matriz de confusión:")
print(cm)
print(f"\n Verdaderos Negativos (VN): {tn}")
print(f" Falsos Positivos  (FP): {fp}")
print(f" Falsos Negativos  (FN): {fn}")
print(f" Verdaderos Positivos (VP): {tp}")

# Salida:
# Matriz de confusión:
# [[245 188]
#  [364 533]]
#
#  Verdaderos Negativos (VN): 245
#  Falsos Positivos  (FP): 188
#  Falsos Negativos  (FN): 364
#  Verdaderos Positivos (VP): 533
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9 — `confusion_matrix`: matriz de confusión.*

1. Salida:
2. Matriz de confusión:
3. [[245 188]
4. [364 533]]
5. Verdaderos Negativos (VN): 245
6. Falsos Positivos  (FP): 188
7. Falsos Negativos  (FN): 364
8. Verdaderos Positivos (VP): 533

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** El clasificador aleatorio tiene muchos errores: 188 FP (predijo margen alto cuando era bajo) y 364 FN (predijo bajo cuando era alto). Un modelo ideal tendría solo la diagonal principal.

---

## Ejemplo 10 — `classification_report`: reporte completo

```python
y_real = ventas["margen_alto"].values
# Clasificador simple: predice 1 si precio > 2000
y_pred = (ventas["precio_unitario"] > 2000).astype(int)

reporte = classification_report(y_real, y_pred, target_names=["Bajo", "Alto"])
print(reporte)

# Salida:
#               precision    recall  f1-score   support
#
#        Bajo       0.40      0.44      0.42       433
#        Alto       0.68      0.65      0.66       897
#
#    accuracy                           0.58      1330
#   macro avg       0.54      0.54      0.54      1330
# weighted avg       0.59      0.58      0.58      1330
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10 — `classification_report`: reporte completo.*

1. Clasificador simple: predice 1 si precio > 2000
2. Salida:
3. precision    recall  f1-score   support
4. Bajo       0.40      0.44      0.42       433
5. Alto       0.68      0.65      0.66       897
6. accuracy                           0.58      1330
7. macro avg       0.54      0.54      0.54      1330
8. weighted avg       0.59      0.58      0.58      1330

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** El clasificador basado en precio > 2000 funciona mejor para clase "Alto" (f1=0.66) que "Bajo" (f1=0.42). El soporte (support) muestra el desbalance: 897 altos vs 433 bajos.

---

## Ejemplo 11 — `precision_score`, `recall_score`, `f1_score` individuales

```python
y_real = ventas["margen_alto"].values
y_pred = (ventas["precio_unitario"] > 2000).astype(int)

precision = precision_score(y_real, y_pred)
recall    = recall_score(y_real, y_pred)
f1        = f1_score(y_real, y_pred)

print(f"Precision: {precision:.3f}  ← de los que predijo alto, cuántos realmente lo son")
print(f"Recall:    {recall:.3f}  ← de los realmente altos, cuántos detectó")
print(f"F1-score:  {f1:.3f}  ← media armónica de ambos")

# Salida:
# Precision: 0.680  ← de los que predijo alto, cuántos realmente lo son
# Recall:    0.646  ← de los realmente altos, cuántos detectó
# F1-score:  0.663  ← media armónica de ambos
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11 — `precision_score`, `recall_score`, `f1_score` individuales.*

1. Salida:
2. Precision: 0.680  ← de los que predijo alto, cuántos realmente lo son
3. Recall:    0.646  ← de los realmente altos, cuántos detectó
4. F1-score:  0.663  ← media armónica de ambos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** Precision mide falsos positivos (cuidado: clasificar barato como margen-alto), recall mide falsos negativos (perder oportunidades de detectar márgenes altos). F1 balancea ambas.

---

## Ejemplo 12 — `roc_auc_score`: área bajo la curva ROC

```python
y_real = ventas["margen_alto"].values
# Probabilidades simuladas de un clasificador
np.random.seed(42)
y_prob = np.random.rand(len(y_real))

auc = roc_auc_score(y_real, y_prob)
print(f"AUC-ROC: {auc:.3f}")

# Salida:
# AUC-ROC: 0.499
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12 — `roc_auc_score`: área bajo la curva ROC.*

1. Probabilidades simuladas de un clasificador
2. Salida:
3. AUC-ROC: 0.499

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** AUC ≈ 0.50 es equivalente a un clasificador aleatorio. Un AUC = 1 es perfecto, AUC = 0.5 es aleatorio. En ventas, AUC > 0.8 se considera bueno para clasificar márgenes.

---

## Ejemplo 13 — `roc_curve`: puntos para graficar (fpr, tpr, thresholds)

```python
y_real = ventas["margen_alto"].values
y_prob = np.random.rand(len(y_real))

fpr, tpr, thresholds = roc_curve(y_real, y_prob)

print(f"Puntos de la curva ROC: {len(fpr)}")
print(f"Punto inicial (fpr=0, tpr=0): fpr={fpr[0]:.3f}, tpr={tpr[0]:.3f}")
print(f"Punto final   (fpr=1, tpr=1): fpr={fpr[-1]:.3f}, tpr={tpr[-1]:.3f}")
print(f"Thresholds (primeros 5): {thresholds[:5].round(3)}")

# Salida:
# Puntos de la curva ROC: 1331
# Punto inicial (fpr=0, tpr=0): fpr=0.000, tpr=0.000
# Punto final   (fpr=1, tpr=1): fpr=1.000, tpr=1.000
# Thresholds (primeros 5): [1.991 0.991 0.985 0.978 0.975]
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13 — `roc_curve`: puntos para graficar (fpr, tpr, thresholds).*

1. Salida:
2. Puntos de la curva ROC: 1331
3. Punto inicial (fpr=0, tpr=0): fpr=0.000, tpr=0.000
4. Punto final   (fpr=1, tpr=1): fpr=1.000, tpr=1.000
5. Thresholds (primeros 5): [1.991 0.991 0.985 0.978 0.975]

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** `roc_curve` devuelve pares (fpr, tpr) para cada threshold. Al graficar tpr vs fpr, el área bajo la curva es el AUC. Un clasificador perfecto toca la esquina (0,1).

---

## Ejemplo 14 — `mean_squared_error`: error cuadrático medio en regresión

```python
y_real = ventas["cantidad"].values[:100]
y_pred = y_real + np.random.normal(0, 2, 100)  # predicción simulada

mse = mean_squared_error(y_real, y_pred)
print(f"MSE: {mse:.2f}")
print(f"RMSE: {np.sqrt(mse):.2f}  ← en unidades originales")

# Salida:
# MSE: 4.12
# RMSE: 2.03  ← en unidades originales
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14 — `mean_squared_error`: error cuadrático medio en regresión.*

1. Salida:
2. MSE: 4.12
3. RMSE: 2.03  ← en unidades originales

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** MSE = promedio de (real - predicho)² = 4.12. RMSE = √MSE = 2.03 unidades. Esto significa que, en promedio, el error es de ≈2 productos vendidos. MSE penaliza más los errores grandes (al elevar al cuadrado).

---

## Ejemplo 15 — `r2_score`: coeficiente de determinación

```python
y_real = ventas["cantidad"].values[:100]
y_pred = np.full(100, y_real.mean())  # modelo tonto: predecir la media

r2 = r2_score(y_real, y_pred)
print(f"R² (modelo media constante): {r2:.4f}")

# Ahora con un modelo que se aproxima mejor
y_pred2 = y_real + np.random.normal(0, 2, 100)
r2_mejor = r2_score(y_real, y_pred2)
print(f"R² (modelo con ruido pequeño): {r2_mejor:.4f}")

# Salida:
# R² (modelo media constante): 0.0000
# R² (modelo con ruido pequeño): 0.8479
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15 — `r2_score`: coeficiente de determinación.*

1. Ahora con un modelo que se aproxima mejor
2. Salida:
3. R² (modelo media constante): 0.0000
4. R² (modelo con ruido pequeño): 0.8479

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** R² = 1 − (suma errores² / varianza total). R²=0 significa que el modelo predice tan mal como la media constante. R²=0.85 significa que el modelo explica el 85% de la variabilidad.

---

## Ejemplo 16 — `mean_absolute_error`: error absoluto medio

```python
y_real = ventas["cantidad"].values[:100]
y_pred = y_real + np.random.normal(0, 2, 100)

mae = mean_absolute_error(y_real, y_pred)
print(f"MAE: {mae:.2f} unidades")
print(f"Error promedio absoluto: ±{mae:.2f} productos")

# Salida:
# MAE: 1.61 unidades
# Error promedio absoluto: ±1.61 productos
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16 — `mean_absolute_error`: error absoluto medio.*

1. Salida:
2. MAE: 1.61 unidades
3. Error promedio absoluto: ±1.61 productos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** MAE = promedio de |real - predicho| = 1.61. A diferencia del MSE, no penaliza extra los errores grandes. Es más interpretable: "nos equivocamos por ~1.6 productos en promedio".

---

## Ejemplo 17 — `mean_absolute_percentage_error` (MAPE) manual

```python
y_real = ventas["cantidad"].values[:100] + 1e-6  # evitar división por cero
y_pred = y_real + np.random.normal(0, 2, 100)

mape = np.mean(np.abs((y_real - y_pred) / y_real)) * 100
print(f"MAPE: {mape:.2f}%")
print(f"Interpretación: el error promedio es del {mape:.1f}% del valor real")

# Salida:
# MAPE: 31.55%
# Interpretación: el error promedio es del 31.5% del valor real
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17 — `mean_absolute_percentage_error` (MAPE) manual.*

1. Salida:
2. MAPE: 31.55%
3. Interpretación: el error promedio es del 31.5% del valor real

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** MAPE expresa el error como porcentaje del valor real. Ej: un error de 2 en una venta de 10 unidades es 20%. Es útil para comunicar a negocio, pero tiene problemas con valores cercanos a cero (división por cero).

---

## Ejemplo 18 — Comparar métricas de regresión (MSE vs MAE vs R²)

```python
np.random.seed(42)
y_real = ventas["cantidad"].values[:200]

modelos = {
    "Media constante": np.full(200, y_real.mean()),
    "Ruido pequeño": y_real + np.random.normal(0, 2, 200),
    "Ruido grande": y_real + np.random.normal(0, 5, 200),
}

print(f"{'Modelo':<20} {'MSE':>8} {'MAE':>8} {'R²':>8} {'RMSE':>8}")
print("-" * 52)
for nombre, y_pred in modelos.items():
    mse = mean_squared_error(y_real, y_pred)
    mae = mean_absolute_error(y_real, y_pred)
    r2 = r2_score(y_real, y_pred)
    rmse = np.sqrt(mse)
    print(f"{nombre:<20} {mse:>8.2f} {mae:>8.2f} {r2:>8.4f} {rmse:>8.2f}")

# Salida:
# Modelo                MSE      MAE       R²     RMSE
# ----------------------------------------------------
# Media constante       87.35     7.74   0.0000    9.35
# Ruido pequeño          4.21     1.62   0.9518    2.05
# Ruido grande          25.89     4.06   0.7036    5.09
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18 — Comparar métricas de regresión (MSE vs MAE vs R²).*

1. Salida:
2. Modelo                MSE      MAE       R²     RMSE
3. ----------------------------------------------------
4. Media constante       87.35     7.74   0.0000    9.35
5. Ruido pequeño          4.21     1.62   0.9518    2.05
6. Ruido grande          25.89     4.06   0.7036    5.09

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** Comparar métricas juntas da una visión completa. MSE penaliza errores grandes (ruido grande: 25.89 vs pequeño: 4.21). MAE es más lineal. RMSE está en las mismas unidades (productos). R² muestra proporción de varianza explicada.

---

## Resumen

| Función | Propósito | Contexto ventas |
|---|---|---|
| `train_test_split` | Dividir datos | Separar histórico para validar modelos |
| `cross_val_score` | Validación cruzada | Evaluar sin desperdiciar datos |
| `KFold` | K particiones | Particionado manual flexible |
| `StratifiedKFold` | KFold balanceado | Clasificación desbalanceada (margen alto/bajo) |
| `accuracy_score` | % de aciertos | ¿Qué tan seguido acertamos? |
| `confusion_matrix` | Errores por clase | ¿Confundimos margen alto con bajo? |
| `classification_report` | Reporte completo | precision, recall, f1 por clase |
| `roc_auc_score` | Poder discriminante | ¿Separa bien las clases? |
| `mean_squared_error` | Error cuadrático | Penaliza errores grandes en pronóstico |
| `mean_absolute_error` | Error absoluto | Error promedio en unidades de venta |
| `r2_score` | Varianza explicada | ¿Cuánto explica el modelo? |
| `explained_variance_score` | Varianza explicada (exacta) | Alternativa a R² |
| `max_error` | Peor error | Caso más extremo de error |
| `mean_absolute_percentage_error` | Error porcentual | Comunicación a negocio |

---

## Ejercicios

1. Usa `train_test_split` con `X = ["precio_unitario", "descuento", "cantidad"]` e `y = "ingreso"`. Prueba `test_size=0.2` y `test_size=0.4`. ¿Cómo cambian los tamaños?

2. Aplica `cross_val_score` con un `DummyRegressor(strategy="median")` y compáralo con `strategy="mean"`. ¿Cuál tiene mejor R²?

3. Crea un `StratifiedKFold` con `n_splits=10` para la variable `proveedor_bueno` del Ejemplo 7. ¿Se mantienen las proporciones?

4. Genera predicciones aleatorias para `margen_alto` y calcula `confusion_matrix`. Calcula manualmente precision y recall a partir de la matriz y verifica con `precision_score` y `recall_score`.

5. Usa `classification_report` para comparar dos clasificadores ficticios en la misma variable. ¿Cuál es mejor según F1 macro?

6. Calcula `roc_auc_score` para un clasificador que siempre predice la probabilidad 0.5 y otro que predice la proporción real de la clase. ¿Cuál es mejor?

7. Sobre `cantidad` como target, calcula MSE y MAE para tres modelos: (a) media constante, (b) mediana constante, (c) predicción con ruido normal. ¿Qué observas?

8. Utiliza `max_error` en el Ejemplo 18 para identificar qué modelo tiene el error individual más grande. ¿Qué implicación tiene para el negocio si ese error ocurre en un producto costoso?
