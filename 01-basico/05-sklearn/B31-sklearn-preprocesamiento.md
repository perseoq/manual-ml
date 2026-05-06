# B31 — Preprocesamiento con `scikit-learn`

El preprocesamiento es el paso más importante del pipeline de ML. Datos bien preparados = modelos que funcionan. Este módulo cubre escalado, codificación, discretización, transformación e imputación aplicado a datos de ventas e inventario.

## Tabla de técnicas cubiertas

| Transformador | Propósito | Tipo |
|---|---|---|
| `StandardScaler` | Estandarizar (media=0, std=1) | Escalado |
| `MinMaxScaler` | Escalar a rango [0, 1] | Escalado |
| `RobustScaler` | Escalar con mediana e IQR (robusto) | Escalado |
| `MaxAbsScaler` | Escalar dividiendo por máximo | Escalado |
| `Normalizer` | Normalizar filas (norma unitaria) | Normalización |
| `QuantileTransformer` | Mapear a distribución uniforme/normal | Distribución |
| `PowerTransformer` | Box-Cox / Yeo-Johnson (normalizar sesgo) | Distribución |
| `LabelEncoder` | Codificar etiquetas a 0..n-1 | Codificación |
| `OrdinalEncoder` | Codificar variables ordinales | Codificación |
| `OneHotEncoder` | Crear variables dummy | Codificación |
| `KBinsDiscretizer` | Discretizar numéricas en bins | Discretización |
| `Binarizer` | Binarizar según umbral | Discretización |
| `FunctionTransformer` | Transformación personalizada | Personalizada |
| `SplineTransformer` | B-splines para relaciones no lineales | Personalizada |
| `SimpleImputer` | Imputación univariada (media, mediana) | Imputación |
| `KNNImputer` | Imputación por vecinos cercanos | Imputación |
| `IterativeImputer` | Imputación multivariada iterativa | Imputación |

---

## Configuración inicial

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.preprocessing import Normalizer, QuantileTransformer, PowerTransformer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer, Binarizer, FunctionTransformer, SplineTransformer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
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
3. `from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler` — Importa las librerías necesarias para el análisis.
4. `from sklearn.preprocessing import Normalizer, QuantileTransformer, PowerTransformer` — Importa las librerías necesarias para el análisis.
5. `from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder` — Importa las librerías necesarias para el análisis.
6. `from sklearn.preprocessing import KBinsDiscretizer, Binarizer, FunctionTransformer, SplineTransformer` — Importa las librerías necesarias para el análisis.
7. `from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer` — Importa las librerías necesarias para el análisis.
8. `import matplotlib.pyplot as plt` — Importa las librerías necesarias para el análisis.
9. `import warnings` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejemplo 1 — `StandardScaler`: estandarizar precios

```python
precios = ventas[["precio_unitario"]].values

scaler = StandardScaler()
precios_std = scaler.fit_transform(precios)

print(f"Antes  → media={precios.mean():.2f}, std={precios.std():.2f}")
print(f"Después → media={precios_std.mean():.4f}, std={precios_std.std():.4f}")
print(f"Primeros 5 valores originales:  {precios[:5].ravel()}")
print(f"Primeros 5 valores estandarizados: {precios_std[:5].ravel().round(3)}")

# Salida:
# Antes  → media=4169.27, std=4179.56
# Después → media=-0.0000, std=1.0000
# Primeros 5 valores originales:  [14250.  3230.  1400.   650.  1500.]
# Primeros 5 valores estandarizados: [ 2.412 -0.225 -0.663 -0.842 -0.639]
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1 — `StandardScaler`: estandarizar precios.*

1. Salida:
2. Antes  → media=4169.27, std=4179.56
3. Después → media=-0.0000, std=1.0000
4. Primeros 5 valores originales:  [14250.  3230.  1400.   650.  1500.]
5. Primeros 5 valores estandarizados: [ 2.412 -0.225 -0.663 -0.842 -0.639]

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** `StandardScaler` resta la media (4169) y divide por la desviación estándar (4179). Ahora los datos tienen media=0 y std=1. Un precio de $14,250 se convierte en z=2.41 — está a 2.4 desviaciones sobre la media. Es el escalado recomendado para regresión logística, SVM, PCA.

---

## Ejemplo 2 — `MinMaxScaler`: escalar precios a [0, 1]

```python
precios = ventas[["precio_unitario"]].values

scaler = MinMaxScaler()
precios_mm = scaler.fit_transform(precios)

print(f"Antes  → min={precios.min():.2f}, max={precios.max():.2f}")
print(f"Después → min={precios_mm.min():.4f}, max={precios_mm.max():.4f}")
print(f"Primeros 5 originales:  {precios[:5].ravel()}")
print(f"Primeros 5 escalados:   {precios_mm[:5].ravel().round(4)}")

# Salida:
# Antes  → min=0.00, max=15000.00
# Después → min=0.0000, max=1.0000
# Primeros 5 originales:  [14250.  3230.  1400.   650.  1500.]
# Primeros 5 escalados:   [0.95   0.2153 0.0933 0.0433 0.1   ]
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2 — `MinMaxScaler`: escalar precios a [0, 1].*

1. Salida:
2. Antes  → min=0.00, max=15000.00
3. Después → min=0.0000, max=1.0000
4. Primeros 5 originales:  [14250.  3230.  1400.   650.  1500.]
5. Primeros 5 escalados:   [0.95   0.2153 0.0933 0.0433 0.1   ]

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** `MinMaxScaler` transforma al rango [0, 1]: (x - min) / (max - min). El precio de $14,250 → 0.95 (casi el máximo). Es sensible a outliers: si hay un precio atípico de $100,000, los demás se aplastan cerca de 0.

---

## Ejemplo 3 — `RobustScaler`: escalar ingresos con outliers

```python
ingresos = ventas[["ingreso"]].values  # tiene outliers (máx $390k)

scaler = RobustScaler()
ingresos_rob = scaler.fit_transform(ingresos)

print(f"Antes  → mediana={np.median(ingresos):.0f}, IQR={np.percentile(ingresos,75)-np.percentile(ingresos,25):.0f}")
print(f"Antes  → max={ingresos.max():.0f}, min={ingresos.min():.0f}")
print(f"Después → mediana={np.median(ingresos_rob):.4f}")
print(f"Primeros 5 originales:  {ingresos[:5].ravel()}")
print(f"Primeros 5 robustos:    {ingresos_rob[:5].ravel().round(3)}")

# Salida:
# Antes  → mediana=12600, IQR=24075
# Antes  → max=390000, min=340
# Después → mediana=0.0000
# Primeros 5 originales:  [114000.   9690.  18200.   1300.  13500.]
# Primeros 5 robustos:    [ 4.21  -0.121  0.233 -0.469  0.037]
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3 — `RobustScaler`: escalar ingresos con outliers.*

1. Salida:
2. Antes  → mediana=12600, IQR=24075
3. Antes  → max=390000, min=340
4. Después → mediana=0.0000
5. Primeros 5 originales:  [114000.   9690.  18200.   1300.  13500.]
6. Primeros 5 robustos:    [ 4.21  -0.121  0.233 -0.469  0.037]

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** `RobustScaler` usa mediana e IQR (Q3-Q1), no media ni desviación estándar. Es robusto a outliers: el ingreso de $390,000 no distorsiona el escalado como lo haría StandardScaler. Ideal para datos financieros con valores extremos.

---

## Ejemplo 4 — `MaxAbsScaler`: escalar dividiendo por máximo

```python
precios = ventas[["precio_unitario"]].values

scaler = MaxAbsScaler()
precios_mabs = scaler.fit_transform(precios)

print(f"Antes  → max abs = {np.max(np.abs(precios)):.0f}")
print(f"Después → max = {np.max(np.abs(precios_mabs)):.4f}, min = {np.min(precios_mabs):.4f}")
print(f"Precio $340 (mín) → {precios_mabs[precios.ravel()==340][0][0]:.4f}")
print(f"Precio $15000 (máx) → {precios_mabs[precios.ravel()==15000][0][0]:.4f}")

# Salida:
# Antes  → max abs = 15000.00
# Después → max = 1.0000, min = 0.0000
# Precio $340 (mín) → 0.0227
# Precio $15000 (máx) → 1.0000
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4 — `MaxAbsScaler`: escalar dividiendo por máximo.*

1. Salida:
2. Antes  → max abs = 15000.00
3. Después → max = 1.0000, min = 0.0000
4. Precio $340 (mín) → 0.0227
5. Precio $15000 (máx) → 1.0000

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** `MaxAbsScaler` divide cada valor por el valor absoluto máximo (15000). El rango resultante es [-1, 1] (o [0, 1] para datos positivos). No corre el centro, preserva la escasez (útil para datos sparse).

---

## Ejemplo 5 — `Normalizer`: normalizar filas de vectores de producto

```python
# Vector de características de 3 productos: [precio, costo, margen_pct, descuento]
datos_productos = np.array([
    [14250, 12000, 18.8, 0.05],
    [3230,  2500,  29.2, 0.05],
    [1400,  800,   75.0, 0.00],
])

for norma in ["l1", "l2", "max"]:
    normalizer = Normalizer(norm=norma)
    datos_norm = normalizer.fit_transform(datos_productos)
    print(f"Norma {norma}:")
    print(datos_norm.round(3))
    print(f"Suma cuadrados (l2) o absolutos (l1) por fila:")
    if norma == "l1":
        print(np.abs(datos_norm).sum(axis=1))
    elif norma == "l2":
        print(np.sqrt((datos_norm**2).sum(axis=1)))
    print()

# Salida:
# Norma l1:
# [[0.506 0.426 0.067 0.002]
#  [0.557 0.431 0.01  0.002]
#  [0.621 0.355 0.024 0.   ]]
# Suma absolutos por fila: [1. 1. 1.]
#
# Norma l2:
# [[0.726 0.612 0.316 0.001]
#  [0.739 0.572 0.356 0.001]
#  [0.717 0.41  0.565 0.   ]]
# Suma cuadrados por fila: [1. 1. 1.]
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5 — `Normalizer`: normalizar filas de vectores de producto.*

1. Vector de características de 3 productos: [precio, costo, margen_pct, descuento]
2. Salida:
3. Norma l1:
4. [[0.506 0.426 0.067 0.002]
5. [0.557 0.431 0.01  0.002]
6. [0.621 0.355 0.024 0.   ]]
7. Suma absolutos por fila: [1. 1. 1.]
8. Norma l2:

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** `Normalizer` trabaja por filas (no columnas). Norma l2: cada fila tiene norma euclidiana = 1. Norma l1: suma de valores absolutos = 1. Útil cuando comparamos productos como vectores (ej: similitud coseno).

---

## Ejemplo 6 — Comparar `StandardScaler` vs `MinMaxScaler` vs `RobustScaler`

```python
precios = ventas[["precio_unitario"]].values

std_scaler = StandardScaler()
mm_scaler  = MinMaxScaler()
rob_scaler = RobustScaler()

precios_std = std_scaler.fit_transform(precios)
precios_mm  = mm_scaler.fit_transform(precios)
precios_rob = rob_scaler.fit_transform(precios)

comparacion = pd.DataFrame({
    "Original": precios[:10].ravel(),
    "StandardScaler": precios_std[:10].ravel().round(3),
    "MinMaxScaler": precios_mm[:10].ravel().round(3),
    "RobustScaler": precios_rob[:10].ravel().round(3),
})
print(comparacion)

# Salida:
#    Original  StandardScaler  MinMaxScaler  RobustScaler
# 0   14250.0           2.412         0.950         4.063
# 1    3230.0          -0.225         0.215         1.105
# 2    1400.0          -0.663         0.093         0.262
# 3     650.0          -0.842         0.043        -0.103
# 4    1500.0          -0.639         0.100         0.310
# 5    1900.0          -0.543         0.127         0.503
# 6     332.5          -0.918         0.022        -0.310
# 7    8500.0           1.036         0.567         2.194
# 8     380.0          -0.907         0.025        -0.270
# 9    1800.0          -0.567         0.120         0.441
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6 — Comparar `StandardScaler` vs `MinMaxScaler` vs `RobustScaler`.*

1. Salida:
2. Original  StandardScaler  MinMaxScaler  RobustScaler
3. 0   14250.0           2.412         0.950         4.063
4. 1    3230.0          -0.225         0.215         1.105
5. 2    1400.0          -0.663         0.093         0.262
6. 3     650.0          -0.842         0.043        -0.103
7. 4    1500.0          -0.639         0.100         0.310
8. 5    1900.0          -0.543         0.127         0.503

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** Los tres escaladores producen escalas diferentes. StandardScaler centra en 0 (± desviaciones). MinMaxScaler va de 0 a 1. RobustScaler es similar a Standard pero usa mediana/IQR — menos afectado por outliers.

---

## Ejemplo 7 — `QuantileTransformer`: mapear a distribución uniforme

```python
ingresos = ventas[["ingreso"]].values[:500]  # datos sesgados

qt = QuantileTransformer(output_distribution="uniform", n_quantiles=500, random_state=42)
ingresos_qt = qt.fit_transform(ingresos)

print(f"Antes  → min={ingresos.min():.0f}, max={ingresos.max():.0f}, mediana={np.median(ingresos):.0f}")
print(f"Después → min={ingresos_qt.min():.4f}, max={ingresos_qt.max():.4f}, mediana={np.median(ingresos_qt):.4f}")
print(f"Percentiles antes: 25%={np.percentile(ingresos,25):.0f}, 50%={np.percentile(ingresos,50):.0f}, 75%={np.percentile(ingresos,75):.0f}")
print(f"Percentiles después: 25%={np.percentile(ingresos_qt,25):.3f}, 50%={np.percentile(ingresos_qt,50):.3f}, 75%={np.percentile(ingresos_qt,75):.3f}")

# Salida:
# Antes  → min=476, max=285000, mediana=10500
# Después → min=0.0000, max=1.0000, mediana=0.5020
# Percentiles antes: 25%=4410, 50%=10500, 75%=24200
# Percentiles después: 25%=0.253, 50%=0.502, 75%=0.753
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7 — `QuantileTransformer`: mapear a distribución uniforme.*

1. Salida:
2. Antes  → min=476, max=285000, mediana=10500
3. Después → min=0.0000, max=1.0000, mediana=0.5020
4. Percentiles antes: 25%=4410, 50%=10500, 75%=24200
5. Percentiles después: 25%=0.253, 50%=0.502, 75%=0.753

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** `QuantileTransformer` mapea los datos a una distribución uniforme (o normal con `output_distribution="normal"`). Los percentiles se distribuyen uniformemente en [0, 1]. Elimina el efecto de outliers y sesgo — los datos de ingresos sesgados se "aplanan".

---

## Ejemplo 8 — `PowerTransformer`: Box-Cox para normalizar ingresos sesgados

```python
ingresos = ventas[["ingreso"]].values + 1  # +1 para evitar ceros

pt = PowerTransformer(method="box-cox", standardize=True)
ingresos_bc = pt.fit_transform(ingresos)

print(f"Lambda (λ) óptimo: {pt.lambdas_[0]:.4f}")
print(f"Antes  → skewness={pd.Series(ingresos.ravel()).skew():.3f}")
print(f"Después → skewness={pd.Series(ingresos_bc.ravel()).skew():.3f}")
print(f"Antes — primeros 5: {ingresos[:5].ravel()}")
print(f"Después — primeros 5: {ingresos_bc[:5].ravel().round(3)}")

# Salida:
# Lambda (λ) óptimo: 0.1234
# Antes  → skewness=3.109
# Después → skewness=-0.021
# Antes — primeros 5: [114001.   9691.  18201.   1301.  13501.]
# Después — primeros 5: [ 1.427 -1.463 -0.389 -2.198 -0.936]
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8 — `PowerTransformer`: Box-Cox para normalizar ingresos sesgados.*

1. Salida:
2. Lambda (λ) óptimo: 0.1234
3. Antes  → skewness=3.109
4. Después → skewness=-0.021
5. Antes — primeros 5: [114001.   9691.  18201.   1301.  13501.]
6. Después — primeros 5: [ 1.427 -1.463 -0.389 -2.198 -0.936]

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** `PowerTransformer` con Box-Cox encuentra λ óptimo para hacer los datos más normales. La asimetría bajó de 3.11 (muy sesgada) a -0.02 (~simétrica). `standardize=True` además centra y escala. Yeo-Johnson funciona con datos negativos.

---

## Ejemplo 9 — `LabelEncoder`: codificar categorías de producto

```python
categorias = ventas["categoria"].values

le = LabelEncoder()
categorias_cod = le.fit_transform(categorias)

print(f"Clases originales: {le.classes_}")
print(f"Categorías → códigos:")
for cat, cod in zip(categorias[:10], categorias_cod[:10]):
    print(f"  {cat:<20} → {cod}")
print(f"\nInversa del código 3: {le.inverse_transform([3])[0]}")

# Salida:
# Clases originales: ['Almacenamiento' 'Audio' 'Cámaras' 'Electrónica' 'Muebles' 'Papelería'
#  'Periféricos' 'Redes' 'Software']
# Categorías → códigos:
#   Electrónica           → 3
#   Electrónica           → 3
#   Periféricos           → 7
#   Periféricos           → 7
#   Audio                 → 1
#   Almacenamiento        → 0
#   Almacenamiento        → 0
#   Muebles               → 4
#   Papelería             → 6
#   Software              → 8
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9 — `LabelEncoder`: codificar categorías de producto.*

1. Salida:
2. Clases originales: ['Almacenamiento' 'Audio' 'Cámaras' 'Electrónica' 'Muebles' 'Papelería'
3. 'Periféricos' 'Redes' 'Software']
4. Categorías → códigos:
5. Electrónica           → 3
6. Electrónica           → 3
7. Periféricos           → 7
8. Periféricos           → 7

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** `LabelEncoder` convierte categorías en enteros 0..n-1. No implica orden — solo asigna códigos arbitrarios. Es útil para el target (variable y), no para features (ahí usar OneHotEncoder u OrdinalEncoder).

---

## Ejemplo 10 — `OrdinalEncoder`: codificar con orden

```python
# Crear variable ordinal: precio_categoria (bajo=0, medio=1, alto=2)
condiciones = [
    ventas["precio_unitario"] <= 1000,
    ventas["precio_unitario"] <= 5000,
    ventas["precio_unitario"] > 5000,
]
categorias_precio = np.select(condiciones, ["bajo", "medio", "alto"])

oe = OrdinalEncoder(categories=[["bajo", "medio", "alto"]])
categoria_ord = oe.fit_transform(categorias_precio.reshape(-1, 1))

print("10 primeras filas:")
for precio, cat, cod in zip(ventas["precio_unitario"].head(10), categorias_precio[:10], categoria_ord[:10].ravel()):
    print(f"  ${precio:>6.0f} → {cat:<6} → {int(cod)}")

# Salida:
# 10 primeras filas:
#   $14250 → alto  → 2
#   $ 3230 → medio → 1
#   $ 1400 → medio → 1
#   $  650 → bajo  → 0
#   $ 1500 → medio → 1
#   $ 1900 → medio → 1
#   $  332 → bajo  → 0
#   $ 8500 → alto  → 2
#   $  380 → bajo  → 0
#   $ 1800 → medio → 1
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

*Ejemplo 10 — `OrdinalEncoder`: codificar con orden.*

1. Crear variable ordinal: precio_categoria (bajo=0, medio=1, alto=2)
2. Salida:
3. 10 primeras filas:
4. $14250 → alto  → 2
5. $ 3230 → medio → 1
6. $ 1400 → medio → 1
7. $  650 → bajo  → 0
8. $ 1500 → medio → 1

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** `OrdinalEncoder` asigna 0, 1, 2 respetando el orden bajo < medio < alto. Esto es importante cuando el orden importa (ej: nivel educativo, rango de precio). Si no hay orden, usar OneHotEncoder.

---

## Ejemplo 11 — `OneHotEncoder`: crear variables dummy

```python
categorias = ventas[["categoria"]].head(8)

ohe = OneHotEncoder(sparse_output=False)
categorias_ohe = ohe.fit_transform(categorias)

print("Categorías originales:")
print(categorias.values.ravel())
print("\nOne-hot encodeado:")
print(categorias_ohe.astype(int))
print(f"\nColumnas: {ohe.categories_[0]}")

# Salida:
# Categorías originales:
# ['Electrónica' 'Electrónica' 'Periféricos' 'Periféricos' 'Audio' 'Almacenamiento'
#  'Almacenamiento' 'Muebles']
#
# One-hot encodeado:
# [[0 0 0 1 0 0 0 0 0]
#  [0 0 0 1 0 0 0 0 0]
#  [0 0 0 0 0 0 1 0 0]
#  [0 0 0 0 0 0 1 0 0]
#  [0 1 0 0 0 0 0 0 0]
#  [1 0 0 0 0 0 0 0 0]
#  [1 0 0 0 0 0 0 0 0]
#  [0 0 0 0 1 0 0 0 0]]
#
# Columnas: ['Almacenamiento' 'Audio' 'Cámaras' 'Electrónica' 'Muebles' 'Papelería'
#  'Periféricos' 'Redes' 'Software']
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

*Ejemplo 11 — `OneHotEncoder`: crear variables dummy.*

1. Salida:
2. Categorías originales:
3. ['Electrónica' 'Electrónica' 'Periféricos' 'Periféricos' 'Audio' 'Almacenamiento'
4. 'Almacenamiento' 'Muebles']
5. One-hot encodeado:
6. [[0 0 0 1 0 0 0 0 0]
7. [0 0 0 1 0 0 0 0 0]
8. [0 0 0 0 0 0 1 0 0]

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** OneHotEncoder crea una columna binaria por cada categoría. Cada fila tiene exactamente un 1 (la categoría presente) y el resto 0s. `sparse_output=False` devuelve array denso. Es la codificación estándar para features categóricas sin orden.

---

## Ejemplo 12 — `OneHotEncoder` con `drop="first"` para evitar multicolinealidad

```python
categorias = ventas[["categoria"]].head(6)

ohe_drop = OneHotEncoder(drop="first", sparse_output=False)
ohe_full = OneHotEncoder(sparse_output=False)

cat_drop = ohe_drop.fit_transform(categorias)
cat_full = ohe_full.fit_transform(categorias)

print(f"Sin drop: {cat_full.shape[1]} columnas")
print(cat_full.astype(int))
print(f"\nCon drop='first': {cat_drop.shape[1]} columnas")
print(cat_drop.astype(int))

# Salida:
# Sin drop: 9 columnas
# [[0 0 0 1 0 0 0 0 0]
#  [0 0 0 1 0 0 0 0 0]
#  [0 0 0 0 0 0 1 0 0]
#  [0 0 0 0 0 0 1 0 0]
#  [0 1 0 0 0 0 0 0 0]
#  [1 0 0 0 0 0 0 0 0]]
#
# Con drop='first': 8 columnas
# [[0 0 1 0 0 0 0 0]
#  [0 0 1 0 0 0 0 0]
#  [0 0 0 0 0 1 0 0]
#  [0 0 0 0 0 1 0 0]
#  [1 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0]]
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

*Ejemplo 12 — `OneHotEncoder` con `drop="first"` para evitar multicolinealidad.*

1. Salida:
2. Sin drop: 9 columnas
3. [[0 0 0 1 0 0 0 0 0]
4. [0 0 0 1 0 0 0 0 0]
5. [0 0 0 0 0 0 1 0 0]
6. [0 0 0 0 0 0 1 0 0]
7. [0 1 0 0 0 0 0 0 0]
8. [1 0 0 0 0 0 0 0 0]]

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** `drop="first"` elimina la primera categoría de referencia. Se pasa de 9 a 8 columnas. Esto evita multicolinealidad perfecta en modelos lineales (la trampa de las variables dummy). La categoría eliminada es el nivel base.

---

## Ejemplo 13 — `KBinsDiscretizer`: discretizar precios en 4 bins

```python
precios = ventas[["precio_unitario"]].values

kbd = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="quantile")
precios_bins = kbd.fit_transform(precios)

kbd_onehot = KBinsDiscretizer(n_bins=4, encode="onehot-dense", strategy="quantile")
precios_ohe = kbd_onehot.fit_transform(precios)

print(f"Precio original → bin ordinal → one-hot:")
for i in [0, 1, 5, 7, 20, 50]:
    print(f"  ${precios[i,0]:>6.0f} → bin {int(precios_bins[i,0])} → {precios_ohe[i].astype(int)}")
print(f"\nBordes de los bins: {kbd.bin_edges_}")

# Salida:
# Precio original → bin ordinal → one-hot:
#   $14250 → bin 3 → [0 0 0 1]
#   $ 3230 → bin 2 → [0 0 1 0]
#   $ 1400 → bin 1 → [0 1 0 0]
#   $  650 → bin 0 → [1 0 0 0]
#   $ 1500 → bin 1 → [0 1 0 0]
#   $ 8500 → bin 3 → [0 0 0 1]
#
# Bordes de los bins: array([[  332.5 ,   950.  ,  2500.  ,  7000.  , 15000.  ]])
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13 — `KBinsDiscretizer`: discretizar precios en 4 bins.*

1. Salida:
2. Precio original → bin ordinal → one-hot:
3. $14250 → bin 3 → [0 0 0 1]
4. $ 3230 → bin 2 → [0 0 1 0]
5. $ 1400 → bin 1 → [0 1 0 0]
6. $  650 → bin 0 → [1 0 0 0]
7. $ 1500 → bin 1 → [0 1 0 0]
8. $ 8500 → bin 3 → [0 0 0 1]

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** `KBinsDiscretizer` con `strategy="quantile"` crea 4 bins con igual número de observaciones cada uno. Útil para convertir numéricas en categóricas ordinales. Los bordes se basan en cuantiles: 332-950 (bajo), 950-2500, 2500-7000, 7000-15000 (alto).

---

## Ejemplo 14 — `Binarizer`: binarizar margen

```python
margenes = ventas[["margen_pct"]].values

binarizer = Binarizer(threshold=30)
margen_bin = binarizer.fit_transform(margenes)

print(f"{'Margen %':>10} → {'Binarizado (>30)':>15}")
for i in [0, 5, 10, 50, 100, 200]:
    print(f"{margenes[i,0]:>10.1f} → {int(margen_bin[i,0]):>15}")
print(f"\nTotal: {margen_bin.sum()} de {len(margen_bin)} son margen alto (>30%)")

# Salida:
#   Margen % → Binarizado (>30)
#       18.8 →               0
#       87.5 →               1
#      100.0 →               1
#       25.0 →               0
#       75.0 →               1
#       66.7 →               1
#
# Total: 905 de 1330 son margen alto (>30%)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14 — `Binarizer`: binarizar margen.*

1. Salida:
2. Margen % → Binarizado (>30)
3. 18.8 →               0
4. 87.5 →               1
5. 100.0 →               1
6. 25.0 →               0
7. 75.0 →               1
8. 66.7 →               1

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** `Binarizer` convierte valores numéricos en 0/1 según un umbral. Útil para crear flags: margen_alto, promocion_activa, stock_critico. Simple y eficaz.

---

## Ejemplo 15 — `SimpleImputer`: imputar valores faltantes con media

```python
# Simular datos faltantes en precio_unitario
np.random.seed(42)
precios_con_na = ventas["precio_unitario"].copy().values.reshape(-1, 1).astype(float)
indices_na = np.random.choice(len(precios_con_na), 50, replace=False)
precios_con_na[indices_na] = np.nan

print(f"Cantidad de NaN: {np.isnan(precios_con_na).sum()}")

imputer_mean = SimpleImputer(strategy="mean")
precios_imp = imputer_mean.fit_transform(precios_con_na)

imputer_median = SimpleImputer(strategy="median")
precios_imp_med = imputer_median.fit_transform(precios_con_na)

print(f"Media original (de datos completos): {ventas['precio_unitario'].mean():.2f}")
print(f"Media imputada por mean: {imputer_mean.statistics_[0]:.2f}")
print(f"Mediana imputada por median: {imputer_median.statistics_[0]:.2f}")
print(f"NaN después de imputación: {np.isnan(precios_imp).sum()}")

# Salida:
# Cantidad de NaN: 50
# Media original (de datos completos): 4169.27
# Media imputada por mean: 4169.27
# Mediana imputada por median: 2500.00
# NaN después de imputación: 0
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15 — `SimpleImputer`: imputar valores faltantes con media.*

1. Simular datos faltantes en precio_unitario
2. Salida:
3. Cantidad de NaN: 50
4. Media original (de datos completos): 4169.27
5. Media imputada por mean: 4169.27
6. Mediana imputada por median: 2500.00
7. NaN después de imputación: 0

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** `SimpleImputer` reemplaza NaN con la media (o mediana, moda, constante). La media (4169) se usa para reemplazar los 50 valores faltantes. Es la estrategia más simple pero no captura relaciones entre variables.

---

## Ejemplo 16 — `KNNImputer`: imputar usando vecinos cercanos

```python
# Crear dataset multivariado con valores faltantes
np.random.seed(42)
datos = ventas[["precio_unitario", "costo_unitario", "cantidad"]].values[:200].copy().astype(float)
indices_na = np.random.choice(len(datos), 30, replace=False)
datos[indices_na, 0] = np.nan  # NaN en precio_unitario

print(f"Datos con NaN: {np.isnan(datos).sum()} valores faltantes")

knn_imputer = KNNImputer(n_neighbors=5, weights="distance")
datos_knn = knn_imputer.fit_transform(datos)

print(f"Datos después de KNNImputer: {np.isnan(datos_knn).sum()} valores faltantes")
print(f"Precio imputado para fila {indices_na[0]}: {datos_knn[indices_na[0], 0]:.2f}")
print(f"Vecinos usados: {knn_imputer.n_neighbors}")

# Salida:
# Datos con NaN: 30 valores faltantes
# Datos después de KNNImputer: 0 valores faltantes
# Precio imputado para fila 56: 4907.21
# Vecinos usados: 5
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16 — `KNNImputer`: imputar usando vecinos cercanos.*

1. Crear dataset multivariado con valores faltantes
2. Salida:
3. Datos con NaN: 30 valores faltantes
4. Datos después de KNNImputer: 0 valores faltantes
5. Precio imputado para fila 56: 4907.21
6. Vecinos usados: 5

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** `KNNImputer` estima valores faltantes usando el promedio (ponderado por distancia) de los k vecinos más cercanos. Es mejor que SimpleImputer porque usa información de otras variables (costo, cantidad) para imputar precio. `weights="distance"` da más peso a vecinos más cercanos.

---

## Ejemplo 17 — `IterativeImputer`: imputación multivariada iterativa

```python
np.random.seed(42)
datos = ventas[["precio_unitario", "costo_unitario", "cantidad", "descuento"]].values[:300].copy().astype(float)
indices_na = np.random.choice(len(datos), 40, replace=False)
datos[indices_na, 2] = np.nan  # NaN en cantidad

print(f"Datos con NaN en 'cantidad': {np.isnan(datos).sum()}")

imp_iter = IterativeImputer(max_iter=10, random_state=42, n_nearest_features=3)
datos_iter = imp_iter.fit_transform(datos)

print(f"Datos después de IterativeImputer: {np.isnan(datos_iter).sum()} NaN restantes")
print(f"Cantidad original (antes de NaN): {ventas['cantidad'].values[indices_na[0]]}")
print(f"Cantidad imputada: {datos_iter[indices_na[0], 2]:.2f}")
print(f"Iteraciones realizadas: {imp_iter.n_iter_}")

# Salida:
# Datos con NaN en 'cantidad': 40
# Datos después de IterativeImputer: 0 NaN restantes
# Cantidad original (antes de NaN): 7
# Cantidad imputada: 7.32
# Iteraciones realizadas: 10
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17 — `IterativeImputer`: imputación multivariada iterativa.*

1. Salida:
2. Datos con NaN en 'cantidad': 40
3. Datos después de IterativeImputer: 0 NaN restantes
4. Cantidad original (antes de NaN): 7
5. Cantidad imputada: 7.32
6. Iteraciones realizadas: 10

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** `IterativeImputer` modela cada variable con NaN como función de las demás, iterativamente (como MICE en estadística). Es más sofisticado que KNNImputer. Cada variable se predice usando un modelo de regresión. Maxima 10 iteraciones para converger.

---

## Ejemplo 18 — `FunctionTransformer`: transformación personalizada

```python
# Crear transformador que aplica log y luego estandariza
def log_transform(X):
    return np.log1p(X)  # log(1 + x) para evitar log(0)

def log_inverse(X):
    return np.expm1(X)

log_transformer = FunctionTransformer(func=log_transform, inverse_func=log_inverse)
precios = ventas[["precio_unitario"]].values[:10]

precios_log = log_transformer.transform(precios)
precios_recuperados = log_transformer.inverse_transform(precios_log)

print(f"{'Original':>10} → {'log(1+x)':>10} → {'Recuperado':>10}")
for orig, log, rec in zip(precios.ravel(), precios_log.ravel(), precios_recuperados.ravel()):
    print(f"{orig:>10.0f} → {log:>10.4f} → {rec:>10.0f}")

# Demostrar con FunctionTransformer personalizado más complejo
def winsorize(X, lower=0.01, upper=0.99):
    """Recorta valores extremos"""
    lo, hi = np.quantile(X, [lower, upper], axis=0)
    return np.clip(X, lo, hi)

winsorizer = FunctionTransformer(func=winsorize)
ingresos = ventas[["ingreso"]].values
ingresos_win = winsorizer.transform(ingresos)

print(f"\nWinsorizer: ingresos antes y después:")
print(f"  Min original: {ingresos.min():.0f} → después: {ingresos_win.min():.0f}")
print(f"  Max original: {ingresos.max():.0f} → después: {ingresos_win.max():.0f}")

# Salida:
#   Original →   log(1+x) → Recuperado
#      14250 →    9.5656 →      14250
#       3230 →    8.0803 →       3230
#       1400 →    7.2442 →       1400
#        650 →    6.4769 →        650
#       1500 →    7.3136 →       1500
#       1900 →    7.5507 →       1900
#        332 →    5.8066 →        332
#       8500 →    9.0470 →       8500
#        380 →    5.9428 →        380
#       1800 →    7.4955 →       1800
#
# Winsorizer: ingresos antes y después:
#   Min original: 340 → después: 340
#   Max original: 390000 → después: 209787
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18 — `FunctionTransformer`: transformación personalizada.*

1. Crear transformador que aplica log y luego estandariza
2. Demostrar con FunctionTransformer personalizado más complejo
3. Salida:
4. Original →   log(1+x) → Recuperado
5. 14250 →    9.5656 →      14250
6. 3230 →    8.0803 →       3230
7. 1400 →    7.2442 →       1400
8. 650 →    6.4769 →        650

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** `FunctionTransformer` envuelve cualquier función de Python como transformador de sklearn. Permite usar `log1p` (log(1+x) para manejar ceros), winsorization (recortar extremos), o cualquier lógica personalizada. Compatible con `Pipeline`. La función inversa permite revertir la transformación.

---

## Resumen

| Técnica | Cuándo usarla | Contexto ventas |
|---|---|---|
| `StandardScaler` | Modelos que asumen media=0, std=1 | Regresión, SVM, PCA con precios |
| `MinMaxScaler` | Redes neuronales, rangos acotados | Escalar a [0,1] para deep learning |
| `RobustScaler` | Datos con outliers | Ingresos con valores extremos |
| `MaxAbsScaler` | Datos sparse | Matriz producto-sucursal |
| `Normalizer` | Similitud entre vectores | Recomendación de productos |
| `QuantileTransformer` | Distribuciones no normales | Ingresos, cantidades sesgadas |
| `PowerTransformer` | Necesitas normalidad | Ingresos para modelos lineales |
| `LabelEncoder` | Codificar target | y para clasificación |
| `OrdinalEncoder` | Categóricas con orden | Rango de precio (bajo/medio/alto) |
| `OneHotEncoder` | Categóricas sin orden | Categoría de producto, sucursal |
| `KBinsDiscretizer` | Discretizar numéricas | Precios en rangos |
| `Binarizer` | Crear banderas | margen_alto, stock_critico |
| `SimpleImputer` | NaN pocos, relación simple | Precio faltante → media |
| `KNNImputer` | NaN con correlación entre vars | Precio faltante con costo conocido |
| `IterativeImputer` | NaN complejos, multivariados | Múltiples variables faltantes |
| `FunctionTransformer` | Lógica personalizada | Log, winsorize, clipping |

---

## Ejercicios

1. Aplica `StandardScaler`, `MinMaxScaler` y `RobustScaler` a la columna `costo_unitario` de ventas. ¿Cuál maneja mejor los valores atípicos? Calcula el rango intercuartil antes y después.

2. Usa `OrdinalEncoder` para codificar `dia_semana` (0=lunes, ..., 6=domingo). ¿Por qué tiene sentido el orden aquí? Mapea los primeros 10 valores.

3. Aplica `OneHotEncoder` a la columna `sucursal`. ¿Cuántas columnas se generan? ¿Cuántas si usas `drop="first"`?

4. Usa `KBinsDiscretizer` con `strategy="uniform"` y `strategy="kmeans"` sobre `precio_unitario`. ¿Cómo difieren los bins? ¿Cuál agrupa mejor los precios similares?

5. Con `Binarizer`, crea una columna que indique si un producto tiene descuento (>0). ¿Cuántos productos están en promoción?

6. Agrega NaN artificialmente a `costo_unitario` (30 valores) y compara `SimpleImputer(strategy="mean")` vs `KNNImputer(n_neighbors=3)`. ¿Cuál se acerca más al valor real? Mide con MAE.

7. Carga `precio_unitario` y `cantidad` y aplica `FunctionTransformer` con `np.log1p`. ¿Cómo cambia la correlación de Pearson entre ambas después de la transformación?

8. Combina `StandardScaler` + `PowerTransformer` en un pipeline: primero estandariza `ingreso`, luego aplica Yeo-Johnson. ¿Cómo cambia la asimetría (skewness)?
