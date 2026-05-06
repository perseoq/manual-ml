# A10 — Feature Scaling y Transformación para Ventas, Compras e Inventarios

## Fundamentos Teóricos

El **escalado** y la **transformación** son pasos críticos cuando las features tienen escalas, distribuciones o rangos diferentes. Muchos modelos (SVM, KNN, regresión logística, PCA) asumen que los datos están centrados y escalados.

### Tipos de escalado

| Método | Fórmula | Rango | Robusto a outliers |
|--------|---------|-------|-------------------|
| **StandardScaler** | (x - μ) / σ | Ilimitado | No |
| **MinMaxScaler** | (x - min) / (max - min) | [0, 1] (o configurable) | No |
| **RobustScaler** | (x - Q₂) / (Q₃ - Q₁) | Ilimitado | Sí |
| **MaxAbsScaler** | x / max\|x\| | [-1, 1] | No (si hay outlier extremo) |
| **Normalizer** | x / \|x\|ₚ | Norma = 1 | N/A (por fila) |

### Transformaciones de distribución

| Método | Output | Ideal para |
|--------|--------|-----------|
| **QuantileTransformer** | Distribución uniforme o normal | Datos con forma arbitraria |
| **PowerTransformer (Box-Cox)** | Más normal | Datos positivos estrictos |
| **PowerTransformer (Yeo-Johnson)** | Más normal | Datos con negativos |

### Discretización y splines

| Método | Uso |
|--------|-----|
| **KBinsDiscretizer** | Convertir continua a categórica ordinal |
| **Binarizer** | Umbral booleano |
| **SplineTransformer** | Bases B-spline para efectos no lineales |

---

## Configuración

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,
    Normalizer, QuantileTransformer, PowerTransformer,
    KBinsDiscretizer, Binarizer, SplineTransformer, FunctionTransformer,
    LabelEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance

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
5. `from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV` — Importa las librerías necesarias para el análisis.
6. `from sklearn.preprocessing import (` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejemplo 1: StandardScaler — estandarizar ingresos (media=0, std=1)

```python
# Simulamos ingresos de ventas (sesgo a la derecha)
np.random.seed(42)
n = 1000
ingresos = np.random.lognormal(mean=10, sigma=1.2, size=n)
df_ingresos = pd.DataFrame({'ingreso': ingresos})

scaler_std = StandardScaler()
ingresos_scaled = scaler_std.fit_transform(df_ingresos)

print("StandardScaler — Ingresos")
print(f"Antes:  mean={df_ingresos['ingreso'].mean():.2f}, "
      f"std={df_ingresos['ingreso'].std():.2f}")
print(f"Después: mean={ingresos_scaled.mean():.4f}, "
      f"std={ingresos_scaled.std():.4f}")

# Visualización
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df_ingresos['ingreso'], bins=50, color='steelblue', edgecolor='white')
axes[0].set_title('Distribución original (ingresos)')
axes[1].hist(ingresos_scaled, bins=50, color='coral', edgecolor='white')
axes[1].set_title('StandardScaler (media=0, std=1)')
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: StandardScaler — estandarizar ingresos (media=0, std=1).*

1. Simulamos ingresos de ventas (sesgo a la derecha)
2. Visualización

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejemplo 2: MinMaxScaler — escalar precios a [0,1]

```python
# Precios de productos en inventario
precios = np.concatenate([
    np.random.uniform(5, 50, 700),   # productos baratos
    np.random.uniform(100, 500, 250), # productos medios
    np.random.uniform(1000, 5000, 50) # productos caros
])
df_precios = pd.DataFrame({'precio': precios})

scaler_mm = MinMaxScaler()
precios_mm = scaler_mm.fit_transform(df_precios)

print("MinMaxScaler — Precios")
print(f"Antes:  min={df_precios['precio'].min():.2f}, "
      f"max={df_precios['precio'].max():.2f}")
print(f"Después: min={precios_mm.min():.4f}, max={precios_mm.max():.4f}")

# Verificar primera fila
print(f"\nPrimer precio: {df_precios['precio'].iloc[0]:.2f} → {precios_mm[0,0]:.4f}")
# Fórmula manual: (x - min) / (max - min)
x = df_precios['precio'].iloc[0]
manual = (x - df_precios['precio'].min()) / (df_precios['precio'].max() - df_precios['precio'].min())
print(f"Verificación manual: {manual:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: MinMaxScaler — escalar precios a [0,1].*

1. Precios de productos en inventario
2. Verificar primera fila
3. Fórmula manual: (x - min) / (max - min)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejemplo 3: MinMaxScaler con feature_range=(-1,1)

```python
# Cantidades de productos vendidos
cantidades = np.random.poisson(50, 1000)
df_cant = pd.DataFrame({'cantidad': cantidades})

scaler_mm_neg = MinMaxScaler(feature_range=(-1, 1))
cant_mm_neg = scaler_mm_neg.fit_transform(df_cant)

print("MinMaxScaler con feature_range=(-1, 1)")
print(f"Antes:  min={df_cant['cantidad'].min()}, max={df_cant['cantidad'].max()}")
print(f"Después: min={cant_mm_neg.min():.4f}, max={cant_mm_neg.max():.4f}")
print(f"Media: {cant_mm_neg.mean():.4f}")

# Primeras 10 transformaciones
comparacion_mm = pd.DataFrame({
    'original': df_cant['cantidad'].head(10),
    'escalado': cant_mm_neg[:10, 0]
})
print("\nPrimeras 10 transformaciones:")
print(comparacion_mm.to_string(index=False))
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

*Ejemplo 3: MinMaxScaler con feature_range=(-1,1).*

1. Cantidades de productos vendidos
2. Primeras 10 transformaciones

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejemplo 4: RobustScaler — escalar usando mediana e IQR (robusto a outliers en ingresos)

```python
# Ingresos con outliers extremos (algunas ventas muy grandes)
ingresos_outliers = np.concatenate([
    np.random.lognormal(mean=10, sigma=0.8, size=950),
    np.random.uniform(500000, 2000000, 50)  # outliers
])
df_robust = pd.DataFrame({'ingreso': ingresos_outliers})

scaler_std = StandardScaler()
scaler_rb = RobustScaler()  # usa mediana e IQR

ing_std = scaler_std.fit_transform(df_robust)
ing_rb = scaler_rb.fit_transform(df_robust)

print("RobustScaler vs StandardScaler con outliers:")
print(f"Original: mean={df_robust['ingreso'].mean():.2f}, "
      f"std={df_robust['ingreso'].std():.2f}")
print(f"Standard: mean={ing_std.mean():.4f}, std={ing_std.std():.4f}")
print(f"Robust:   mean={ing_rb.mean():.4f}, std={ing_rb.std():.4f}")

# Ver rango después del escalado
print(f"\nStandard: min={ing_std.min():.2f}, max={ing_std.max():.2f}")
print(f"Robust:   min={ing_rb.min():.2f}, max={ing_rb.max():.2f}")

# El RobustScaler comprime mejor los outliers
df_comp = pd.DataFrame({
    'original': df_robust['ingreso'].values,
    'standard': ing_std[:, 0],
    'robust': ing_rb[:, 0]
})
print("\nTop 5 valores más extremos:")
outlier_idx = df_robust['ingreso'].nlargest(5).index
print(df_comp.loc[outlier_idx].to_string())
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: RobustScaler — escalar usando mediana e IQR (robusto a outliers en ingresos).*

1. Ingresos con outliers extremos (algunas ventas muy grandes)
2. Ver rango después del escalado
3. El RobustScaler comprime mejor los outliers

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejemplo 5: MaxAbsScaler — mantener 0 y escalar por máximo

```python
# Datos de compras: montos positivos y negativos (devoluciones)
montos = np.concatenate([
    np.random.uniform(-500, -10, 100),  # devoluciones
    np.random.uniform(10, 1000, 900),   # compras normales
])
df_montos = pd.DataFrame({'monto': montos})

scaler_maxabs = MaxAbsScaler()
montos_maxabs = scaler_maxabs.fit_transform(df_montos)

print("MaxAbsScaler — Montos con negativos:")
print(f"Antes:  min={df_montos['monto'].min():.2f}, max={df_montos['monto'].max():.2f}")
print(f"Después: min={montos_maxabs.min():.4f}, max={montos_maxabs.max():.4f}")

# Verificar que el 0 se mantiene
idx_cero = np.abs(df_montos['monto'].values).argmin()
print(f"Valor original cercano a 0: {df_montos['monto'].iloc[idx_cero]:.2f} → {montos_maxabs[idx_cero, 0]:.4f}")

# Histograma
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df_montos['monto'], bins=50, color='steelblue')
axes[0].set_title('Original')
axes[1].hist(montos_maxabs, bins=50, color='coral')
axes[1].set_title('MaxAbsScaler')
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: MaxAbsScaler — mantener 0 y escalar por máximo.*

1. Datos de compras: montos positivos y negativos (devoluciones)
2. Verificar que el 0 se mantiene
3. Histograma

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejemplo 6: Normalizer l2 — normalizar filas a norma 1 (útil para embeddings)

```python
# Características de productos: vector de atributos
df_productos = pd.DataFrame({
    'precio_relativo': np.random.uniform(0, 1, 500),
    'peso_relativo': np.random.uniform(0, 1, 500),
    'popularidad': np.random.uniform(0, 1, 500),
    'calidad': np.random.uniform(0, 1, 500),
})

# Normalizer L2 (cada fila tiene norma euclidiana = 1)
norm_l2 = Normalizer(norm='l2')
X_l2 = norm_l2.fit_transform(df_productos)

# Verificar norma L2 de cada fila
normas = np.linalg.norm(X_l2, axis=1)
print("Normalizer L2 — Norma por fila:")
print(f"  min={normas.min():.6f}, max={normas.max():.6f}")
print(f"  primeras 5 normas: {normas[:5].round(6)}")

# Comparar L1, L2 y Max
norm_l1 = Normalizer(norm='l1')
norm_max = Normalizer(norm='max')
X_l1 = norm_l1.fit_transform(df_productos)
X_max = norm_max.fit_transform(df_productos)

print(f"\nComparación de normas:")
print(f"  L1:  suma absoluta por fila = {X_l1.sum(axis=1)[:5].round(4)}")
print(f"  L2:  norma euclidiana = {np.linalg.norm(X_l2[:5], axis=1).round(4)}")
print(f"  Max: max por fila = {X_max.max(axis=1)[:5].round(4)}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: Normalizer l2 — normalizar filas a norma 1 (útil para embeddings).*

1. Características de productos: vector de atributos
2. Normalizer L2 (cada fila tiene norma euclidiana = 1)
3. Verificar norma L2 de cada fila
4. Comparar L1, L2 y Max

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejemplo 7: QuantileTransformer uniform — mapear a distribución uniforme

```python
# Días de entrega (distribución exponencial)
dias_entrega = np.random.exponential(scale=15, size=2000)
df_dias = pd.DataFrame({'dias_entrega': dias_entrega})

qt_uniform = QuantileTransformer(
    n_quantiles=1000,
    output_distribution='uniform'
)
dias_qt_uniform = qt_uniform.fit_transform(df_dias)

print("QuantileTransformer → Uniforme:")
print(f"Original: min={dias_entrega.min():.2f}, max={dias_entrega.max():.2f}")
print(f"Quantile: min={dias_qt_uniform.min():.4f}, max={dias_qt_uniform.max():.4f}")

# La transformación "aplana" la distribución exponencial a uniforme
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(dias_entrega, bins=50, density=True, color='steelblue')
axes[0].set_title(f'Original (exponencial)')
axes[1].hist(dias_qt_uniform, bins=50, density=True, color='coral')
axes[1].set_title('QuantileTransformer → Uniforme')
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: QuantileTransformer uniform — mapear a distribución uniforme.*

1. Días de entrega (distribución exponencial)
2. La transformación "aplana" la distribución exponencial a uniforme

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejemplo 8: QuantileTransformer normal — mapear a distribución normal

```python
qt_normal = QuantileTransformer(
    n_quantiles=1000,
    output_distribution='normal'
)
dias_qt_normal = qt_normal.fit_transform(df_dias)

print("QuantileTransformer → Normal:")
print(f"  mean={dias_qt_normal.mean():.4f}, std={dias_qt_normal.std():.4f}")

# Comparar distribuciones
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(dias_entrega, bins=50, density=True, color='steelblue')
axes[0].set_title('Original (exponencial)')
axes[1].hist(dias_qt_normal, bins=50, density=True, color='forestgreen')
axes[1].set_title('QuantileTransformer → Normal')
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: QuantileTransformer normal — mapear a distribución normal.*

1. Comparar distribuciones

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejemplo 9: PowerTransformer Box-Cox — normalizar datos positivos sesgados

```python
# Costos de inventario (sesgados a la derecha)
costos = np.random.lognormal(mean=8, sigma=1.5, size=2000)
df_costos = pd.DataFrame({'costo': costos})

pt_bc = PowerTransformer(method='box-cox', standardize=False)
costos_bc = pt_bc.fit_transform(df_costos)

# Box-Cox solo acepta valores positivos (> 0)
print("PowerTransformer Box-Cox:")
print(f"Original:  min={costos.min():.2f}, max={costos.max():.2f}")
print(f"Box-Cox:   min={costos_bc.min():.4f}, max={costos_bc.max():.4f}")
print(f"Lambda óptimo: {pt_bc.lambdas_[0]:.4f}")

# Estadísticos de asimetría (skewness)
from scipy.stats import skew
print(f"\nSkewness original: {skew(costos):.4f}")
print(f"Skewness Box-Cox:  {skew(costos_bc):.4f}")

# Visualización
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(costos, bins=50, density=True, color='steelblue')
axes[0].set_title(f'Original (skew={skew(costos):.2f})')
axes[1].hist(costos_bc, bins=50, density=True, color='purple')
axes[1].set_title(f'Box-Cox (skew={skew(costos_bc):.2f})')
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: PowerTransformer Box-Cox — normalizar datos positivos sesgados.*

1. Costos de inventario (sesgados a la derecha)
2. Box-Cox solo acepta valores positivos (> 0)
3. Estadísticos de asimetría (skewness)
4. Visualización

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejemplo 10: PowerTransformer Yeo-Johnson — normalizar datos con negativos

```python
# Margen de ganancia (puede ser negativo)
margen = np.concatenate([
    np.random.normal(loc=-5, scale=3, size=200),   # márgenes negativos
    np.random.normal(loc=25, scale=10, size=1800), # márgenes positivos
])
df_margen = pd.DataFrame({'margen': margen})

pt_yj = PowerTransformer(method='yeo-johnson', standardize=True)
margen_yj = pt_yj.fit_transform(df_margen)

print("PowerTransformer Yeo-Johnson:")
print(f"Original: min={margen.min():.2f}, max={margen.max():.2f}")
print(f"Yeo-Johnson: min={margen_yj.min():.4f}, max={margen_yj.max():.4f}")
print(f"Lambda óptimo: {pt_yj.lambdas_[0]:.4f}")
print(f"Skewness original: {skew(margen):.4f}")
print(f"Skewness Yeo-Johnson: {skew(margen_yj):.4f}")

# Yeo-Johnson maneja valores negativos (a diferencia de Box-Cox)
print(f"\nHay valores negativos: {(margen < 0).sum()} de {len(margen)}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: PowerTransformer Yeo-Johnson — normalizar datos con negativos.*

1. Margen de ganancia (puede ser negativo)
2. Yeo-Johnson maneja valores negativos (a diferencia de Box-Cox)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejemplo 11: Comparar Standard vs Robust vs Quantile vs Power (con outliers)

```python
# Datos de ventas diarias con outliers
ventas_diarias = np.concatenate([
    np.random.normal(loc=1000, scale=200, size=950),
    np.random.uniform(5000, 20000, 50)  # outliers: días promocionales
]).reshape(-1, 1)

scalers = {
    'StandardScaler': StandardScaler(),
    'RobustScaler': RobustScaler(),
    'QuantileUniform': QuantileTransformer(output_distribution='uniform'),
    'QuantileNormal': QuantileTransformer(output_distribution='normal'),
    'PowerTransformer(YJ)': PowerTransformer(method='yeo-johnson'),
}

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

axes[0].hist(ventas_diarias, bins=50, color='gray', edgecolor='white')
axes[0].set_title('Original')

for i, (nombre, scaler) in enumerate(scalers.items(), 1):
    X_scaled = scaler.fit_transform(ventas_diarias)
    axes[i].hist(X_scaled, bins=50, color='steelblue', edgecolor='white')
    axes[i].set_title(f'{nombre}\n(min={X_scaled.min():.2f}, max={X_scaled.max():.2f})')

plt.tight_layout()
plt.show()

# Tabla comparativa
print(f"{'Método':25s} {'Min':>8s} {'Max':>8s} {'Mean':>8s} {'Std':>8s}")
print("-" * 60)
print(f"{'Original':25s} {ventas_diarias.min():>8.1f} {ventas_diarias.max():>8.1f} "
      f"{ventas_diarias.mean():>8.1f} {ventas_diarias.std():>8.1f}")
for nombre, scaler in scalers.items():
    X_s = scaler.fit_transform(ventas_diarias)
    print(f"{nombre:25s} {X_s.min():>8.4f} {X_s.max():>8.4f} "
          f"{X_s.mean():>8.4f} {X_s.std():>8.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: Comparar Standard vs Robust vs Quantile vs Power (con outliers).*

1. Datos de ventas diarias con outliers
2. Tabla comparativa

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejemplo 12: KBinsDiscretizer strategy='uniform' — bins de igual ancho

```python
# Precios de productos
precios_prod = np.random.lognormal(mean=4.5, sigma=0.8, size=1000)
df_precios = pd.DataFrame({'precio': precios_prod})

kbd_uniform = KBinsDiscretizer(
    n_bins=5, strategy='uniform', encode='ordinal'
)
precios_bins = kbd_uniform.fit_transform(df_precios)

print("KBinsDiscretizer — stratégia 'uniform' (igual ancho):")
print(f"Bins: 5")
print(f"Límites de los bins:")
for i, (edge_l, edge_r) in enumerate(zip(
    kbd_uniform.bin_edges_[0][:-1], kbd_uniform.bin_edges_[0][1:]
)):
    count = ((precios_prod >= edge_l) & (precios_prod < edge_r)).sum()
    print(f"  Bin {i}: [{edge_l:.2f}, {edge_r:.2f}) → {count} obs")

# Verificar con encode='onehot' para crear dummies
kbd_ohe = KBinsDiscretizer(
    n_bins=5, strategy='uniform', encode='onehot-dense'
)
precios_ohe = kbd_ohe.fit_transform(df_precios)
print(f"\nOneHot shape: {precios_ohe.shape} (5 columnas binarias)")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: KBinsDiscretizer strategy='uniform' — bins de igual ancho.*

1. Precios de productos
2. Verificar con encode='onehot' para crear dummies

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejemplo 13: KBinsDiscretizer strategy='quantile' — bins de igual frecuencia

```python
kbd_quantile = KBinsDiscretizer(
    n_bins=5, strategy='quantile', encode='ordinal'
)
precios_quantile = kbd_quantile.fit_transform(df_precios)

print("KBinsDiscretizer — stratégia 'quantile' (igual frecuencia):")
print(f"Bins: 5")
for i, (edge_l, edge_r) in enumerate(zip(
    kbd_quantile.bin_edges_[0][:-1], kbd_quantile.bin_edges_[0][1:]
)):
    count = ((precios_prod >= edge_l) & (precios_prod < edge_r)).sum()
    print(f"  Bin {i}: [{edge_l:.2f}, {edge_r:.2f}) → {count} obs")

# Comparar estratégias
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(precios_prod, bins=50, color='lightgray', edgecolor='white')
# superponer límites de bins uniformes
for e in kbd_uniform.bin_edges_[0]:
    axes[0].axvline(e, color='red', linestyle='--', alpha=0.7)
axes[0].set_title('Uniforme (igual ancho)')

axes[1].hist(precios_prod, bins=50, color='lightgray', edgecolor='white')
for e in kbd_quantile.bin_edges_[0]:
    axes[1].axvline(e, color='green', linestyle='--', alpha=0.7)
axes[1].set_title('Quantile (igual frecuencia)')

plt.tight_layout()
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: KBinsDiscretizer strategy='quantile' — bins de igual frecuencia.*

1. Comparar estratégias
2. superponer límites de bins uniformes

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejemplo 14: Binarizer — flag booleano (ej: margen > 30%)

```python
# Márgenes de ganancia
margenes = np.random.normal(loc=25, scale=12, size=1000)
df_margenes = pd.DataFrame({'margen_pct': margenes})

# Binarizer: margen > 30% → 1, sino → 0
binarizer = Binarizer(threshold=30)
margen_bin = binarizer.fit_transform(df_margenes)

print("Binarizer — threshold=30 (margen > 30%)")
print(f"Media original: {margenes.mean():.2f}%")
print(f"Proporción > 30%: {margen_bin.sum() / len(margen_bin):.2%}")

# Comparar con variable continua
df_bin = pd.DataFrame({
    'margen_original': margenes,
    'margen_alto': margen_bin.flatten().astype(int)
})
print("\nPrimeras 10 filas:")
print(df_bin.head(10))

# Si el threshold cambia
for thr in [10, 20, 30, 40, 50]:
    b = Binarizer(threshold=thr)
    m_bin = b.fit_transform(df_margenes)
    print(f"  threshold={thr:2d} → {m_bin.sum() / len(m_bin):.1%} son 'altos'")
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

*Ejemplo 14: Binarizer — flag booleano (ej: margen > 30%).*

1. Márgenes de ganancia
2. Binarizer: margen > 30% → 1, sino → 0
3. Comparar con variable continua
4. Si el threshold cambia

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejemplo 15: SplineTransformer — bases spline para modelar no linealidad

```python
# Relación no lineal: precio vs demanda
np.random.seed(42)
X_precio = np.linspace(1, 100, 500).reshape(-1, 1)
# Demanda = 100 - 0.5*precio + 0.003*precio² + ruido
demanda = (
    100 - 0.5 * X_precio.ravel() + 0.003 * X_precio.ravel()**2
    + np.random.normal(0, 5, 500)
)

spline = SplineTransformer(
    n_knots=8,
    degree=3,
    knots='quantile'  # ubicar knots en cuantiles
)
X_spline = spline.fit_transform(X_precio)
print(f"SplineTransformer: {X_precio.shape[1]} feature → {X_spline.shape[1]} bases")
print(f"n_knots=8, degree=3 → {8 + 3 - 1} bases (fórmula: n_knots + degree - 1)")

# Mostrar las bases
plt.figure(figsize=(12, 5))
for i in range(X_spline.shape[1]):
    plt.plot(X_precio, X_spline[:, i], label=f'Base {i}', alpha=0.7)
plt.xlabel('Precio')
plt.ylabel('Valor base spline')
plt.title(f'Bases Spline (n_knots=8, degree=3)')
plt.legend(loc='best', ncol=2)
plt.grid(True)
plt.show()

# Entrenar regresión lineal con bases spline
from sklearn.linear_model import LinearRegression
lr_spline = LinearRegression()
lr_spline.fit(X_spline, demanda)

X_pred = np.linspace(1, 100, 200).reshape(-1, 1)
X_pred_spline = spline.transform(X_pred)
y_pred_spline = lr_spline.predict(X_pred_spline)

plt.figure(figsize=(10, 6))
plt.scatter(X_precio, demanda, alpha=0.3, label='Datos reales')
plt.plot(X_pred, y_pred_spline, 'r-', linewidth=3, label='Regresión con Splines')
plt.xlabel('Precio')
plt.ylabel('Demanda')
plt.title('Modelo no lineal con SplineTransformer + LinearRegression')
plt.legend()
plt.grid(True)
plt.show()

print(f"R² del modelo spline: {lr_spline.score(X_spline, demanda):.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: SplineTransformer — bases spline para modelar no linealidad.*

1. Relación no lineal: precio vs demanda
2. Demanda = 100 - 0.5*precio + 0.003*precio² + ruido
3. Mostrar las bases
4. Entrenar regresión lineal con bases spline

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejemplo 16: FunctionTransformer — log transformación personalizada

```python
# Función personalizada: log(1+x) para datos con ceros
def log_transform(x):
    return np.log1p(x)  # log(1+x)

def inverse_log_transform(x):
    return np.expm1(x)  # exp(x) - 1

log_transformer = FunctionTransformer(
    func=log_transform,
    inverse_func=inverse_log_transform,
    validate=True,
    kw_args={},
    inv_kw_args={}
)

# Montos de ventas (sesgados, algunos muy grandes)
montos_venta = np.random.lognormal(mean=8, sigma=2, size=2000).reshape(-1, 1)
montos_log = log_transformer.fit_transform(montos_venta)

print("FunctionTransformer — log(1+x):")
print(f"Original: min={montos_venta.min():.2f}, max={montos_venta.max():.2f}, "
      f"skew={skew(montos_venta.ravel()):.2f}")
print(f"Log:      min={montos_log.min():.4f}, max={montos_log.max():.4f}, "
      f"skew={skew(montos_log.ravel()):.2f}")

# Verificar inversa
montos_recuperados = log_transformer.inverse_transform(montos_log)
print(f"\nMáximo error de inversa: {np.abs(montos_venta - montos_recuperados).max():.6f}")

# Crear transformador personalizado con parámetros
def winsorize_transform(x, lower=0.01, upper=0.99):
    """Recorta outliers en los percentiles especificados."""
    lo = np.percentile(x, lower * 100)
    hi = np.percentile(x, upper * 100)
    return np.clip(x, lo, hi)

winsorizer = FunctionTransformer(
    func=winsorize_transform,
    kw_args={'lower': 0.02, 'upper': 0.98}
)
montos_winsor = winsorizer.fit_transform(montos_venta)
print(f"\nWinsorizer (2%-98%):")
print(f"Original: min={montos_venta.min():.2f}, max={montos_venta.max():.2f}")
print(f"Winsor:   min={montos_winsor.min():.2f}, max={montos_winsor.max():.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: FunctionTransformer — log transformación personalizada.*

1. Función personalizada: log(1+x) para datos con ceros
2. Montos de ventas (sesgados, algunos muy grandes)
3. Verificar inversa
4. Crear transformador personalizado con parámetros

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejemplo 17: ColumnTransformer con diferentes scalers para diferentes columnas

```python
# Dataset mixto: columnas con diferentes distribuciones
np.random.seed(42)
n = 1000
df_mixto = pd.DataFrame({
    'precio': np.random.lognormal(mean=4, sigma=0.5, n),       # sesgada positiva
    'cantidad': np.random.poisson(30, n),                       # discreta
    'margen_pct': np.random.normal(loc=25, scale=10, n),       # normal
    'dias_entrega': np.random.exponential(scale=10, n),        # exponencial
    'es_importante': np.random.choice([0, 1], n, p=[0.7, 0.3]), # binaria
})
target = (
    df_mixto['precio'] * df_mixto['cantidad'] * df_mixto['margen_pct'] / 100
    > np.percentile(df_mixto['precio'] * df_mixto['cantidad'] * df_mixto['margen_pct'] / 100, 75)
).astype(int)

col_transformer = ColumnTransformer([
    ('log_scale', FunctionTransformer(np.log1p, validate=True), ['precio']),
    ('standard', StandardScaler(), ['cantidad']),
    ('robust', RobustScaler(), ['margen_pct']),
    ('quantile_uniform', QuantileTransformer(output_distribution='uniform'), ['dias_entrega']),
    ('passthrough', 'passthrough', ['es_importante']),
])

X_transformed = col_transformer.fit_transform(df_mixto)

print("ColumnTransformer — diferentes scalers por columna:")
print(f"Shape original: {df_mixto.shape}")
print(f"Shape transformado: {X_transformed.shape}")
print(f"Columnas: precio(log) | cantidad(std) | margen(robust) | dias(uniform) | es_importante")

# Pipeline completo
pipeline_ct = Pipeline([
    ('preprocessor', col_transformer),
    ('clf', LogisticRegression(max_iter=1000, random_state=42)),
])

X_train, X_test, y_train, y_test = train_test_split(
    df_mixto, target, test_size=0.3, random_state=42
)
pipeline_ct.fit(X_train, y_train)
print(f"\nAccuracy Pipeline con ColumnTransformer: "
      f"{pipeline_ct.score(X_test, y_test):.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: ColumnTransformer con diferentes scalers para diferentes columnas.*

1. Dataset mixto: columnas con diferentes distribuciones
2. Pipeline completo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejemplo 18: Integrador — comparar efecto de cada scaler en modelo final

```python
# === DATOS DE INVENTARIOS ===
np.random.seed(42)
n = 2000
df_inv_model = pd.DataFrame({
    'costo_unitario': np.random.lognormal(mean=3, sigma=0.8, n),
    'precio_venta': np.random.lognormal(mean=4.5, sigma=0.6, n),
    'stock_actual': np.random.poisson(80, n),
    'rotacion_mensual': np.random.exponential(scale=25, n),
    'dias_entrega': np.random.exponential(scale=12, n),
    'demanda_promedio': np.random.poisson(35, n),
    'merma_pct': np.random.uniform(0, 8, n),
    'lead_time': np.random.exponential(scale=7, n),
})
# Target: predicción de ganancia mensual (regresión)
ganancia = (
    (df_inv_model['precio_venta'] - df_inv_model['costo_unitario'])
    * df_inv_model['stock_actual'] * df_inv_model['rotacion_mensual']
    / (df_inv_model['dias_entrega'] + 1)
    + np.random.normal(0, 500, n)
)

# Evaluar diferentes scalers con KNN (sensible a escala)
metodos_scalers = {
    'Sin escalar': None,
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler(),
    'QuantileUniform': QuantileTransformer(output_distribution='uniform'),
    'QuantileNormal': QuantileTransformer(output_distribution='normal'),
    'PowerTransform(YJ)': PowerTransformer(method='yeo-johnson'),
    'MaxAbsScaler': MaxAbsScaler(),
}

resultados_modelo = []

for nombre, scaler in metodos_scalers.items():
    pipeline = Pipeline([
        ('scaler', scaler) if scaler else ('identity', 'passthrough'),
        ('model', KNeighborsRegressor(n_neighbors=10)),
    ])

    # Evitar nombre None
    if scaler is None:
        pipeline = Pipeline([
            ('identity', 'passthrough'),
            ('model', KNeighborsRegressor(n_neighbors=10)),
        ])

    scores_r2 = cross_val_score(
        pipeline, df_inv_model, ganancia, cv=5, scoring='r2'
    )
    scores_rmse = np.sqrt(-cross_val_score(
        pipeline, df_inv_model, ganancia, cv=5,
        scoring='neg_mean_squared_error'
    ))

    resultados_modelo.append({
        'scaler': nombre,
        'r2_mean': scores_r2.mean(),
        'r2_std': scores_r2.std(),
        'rmse_mean': scores_rmse.mean(),
        'rmse_std': scores_rmse.std(),
    })

# También probar con RandomForest (no requiere escalado)
rf_pipeline = Pipeline([
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])
scores_rf_r2 = cross_val_score(
    rf_pipeline, df_inv_model, ganancia, cv=5, scoring='r2'
)
scores_rf_rmse = np.sqrt(-cross_val_score(
    rf_pipeline, df_inv_model, ganancia, cv=5,
    scoring='neg_mean_squared_error'
))

resultados_modelo.append({
    'scaler': 'RandomForest (ref)',
    'r2_mean': scores_rf_r2.mean(),
    'r2_std': scores_rf_r2.std(),
    'rmse_mean': scores_rf_rmse.mean(),
    'rmse_std': scores_rf_rmse.std(),
})

df_resultados = pd.DataFrame(resultados_modelo).sort_values('r2_mean', ascending=False)
print("=== Efecto del escalado en KNN (Regresión de ganancia) ===")
print(df_resultados.to_string(index=False))

# Gráfico
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
bars = plt.barh(df_resultados['scaler'], df_resultados['r2_mean'],
                xerr=df_resultados['r2_std'], capsize=5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.xlabel('R² (CV)')
plt.title('Efecto del Scaler en R²')
plt.grid(axis='x', alpha=0.3)

plt.subplot(1, 2, 2)
bars2 = plt.barh(df_resultados['scaler'], df_resultados['rmse_mean'],
                 xerr=df_resultados['rmse_std'], capsize=5, color='coral')
plt.xlabel('RMSE (CV)')
plt.title('Efecto del Scaler en RMSE')
plt.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✓ Conclusión: modelos basados en distancias (KNN, SVM) REQUIEREN escalado")
print("  Modelos basados en árboles (RandomForest) NO requieren escalado")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador — comparar efecto de cada scaler en modelo final.*

1. === DATOS DE INVENTARIOS ===
2. Target: predicción de ganancia mensual (regresión)
3. Evaluar diferentes scalers con KNN (sensible a escala)
4. Evitar nombre None
5. También probar con RandomForest (no requiere escalado)
6. Gráfico

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Resumen

| Método | Cuándo usarlo | Parámetros clave | Sensible a outliers |
|--------|---------------|------------------|---------------------|
| **StandardScaler** | Datos aproximadamente normales | — | Sí |
| **MinMaxScaler** | Rangos acotados, redes neuronales | `feature_range` | Sí |
| **RobustScaler** | Datos con outliers | `quantile_range`, `unit_variance` | No |
| **MaxAbsScaler** | Datos sparse (no shift) | — | Moderado |
| **Normalizer** | Embeddings, similitud por filas | `norm='l1'/'l2'/'max'` | N/A |
| **QuantileTransformer** | Distribución arbitraria → uniforme/normal | `n_quantiles`, `output_distribution` | Muy robusto |
| **PowerTransformer** | Normalizar datos sesgados | `method`, `standardize` | Moderado |
| **KBinsDiscretizer** | Discretizar continuas | `n_bins`, `strategy`, `encode` | Depende de strategy |
| **Binarizer** | Flags booleanos | `threshold` | Depende de threshold |
| **SplineTransformer** | No linealidad suave | `n_knots`, `degree`, `knots` | Moderado |
| **FunctionTransformer** | Transformaciones ad-hoc | `func`, `inverse_func` | Depende de la función |
| **ColumnTransformer** | Diferentes scalers por columna | `transformers` | Depende del scaler |

**Regla general:**
- **Árboles**: no necesitan escalado (RandomForest, XGBoost, LightGBM)
- **Distancia**: necesitan escalado (KNN, SVM, K-Means, PCA)
- **Lineales**: se benefician del escalado (Regresión, Ridge, Lasso, LogisticRegression)
- **Redes neuronales**: requieren escalado (MinMaxScaler o StandardScaler típicamente)

---

## Ejercicios

1. Genera un dataset de ventas con 3 features numéricas de diferentes distribuciones: normal, exponencial y uniforme. Aplica StandardScaler, MinMaxScaler y RobustScaler. Compara los resultados visualmente con histogramas. ¿Qué scaler preserva mejor la forma de la distribución?

2. Crea un dataset de compras con outliers intencionales (5% de valores extremos en el monto). Compara StandardScaler vs RobustScaler mostrando: (a) el rango después del escalado, (b) el efecto en un modelo KNN (accuracy con y sin escalado).

3. Para datos de inventarios con distribución sesgada (stock_actual con cola larga), aplica PowerTransformer Box-Cox y QuantileTransformer(output_distribution='normal'). Calcula el skewness antes y después. ¿Cuál reduce más la asimetría?

4. Implementa un ColumnTransformer que aplique: StandardScaler a las numéricas normales, RobustScaler a las numéricas con outliers, y FunctionTransformer(log1p) a las sesgadas positivas. Úsalo en un Pipeline con Ridge Regression y evalúa con R² en CV.

5. Usando el dataset de ventas, aplica KBinsDiscretizer con strategy='uniform', 'quantile' y 'kmeans' para n_bins=5. Compara: (a) distribución de observaciones por bin, (b) accuracy de un modelo LogisticRegression con cada encoding.

6. Genera una relación no lineal simulada (ej: demanda en función del precio con un término cuadrático). Usa SplineTransformer con diferentes n_knots (3, 6, 10, 15) combinado con LinearRegression. ¿Cómo cambia el R² al aumentar los knots? ¿Hay overfitting?

7. Crea tu propio FunctionTransformer que aplique una transformación de raíz cuadrada (sqrt) y su inversa (cuadrado). Aplícalo a datos de monto de compras y compara la distribución antes y después. Verifica que la transformación inversa recupera los valores originales.

8. **Integrador:** Diseña un experimento completo para evaluar el impacto del escalado en la predicción de ganancia de inventarios: (a) Usa el dataset del ejemplo 18, (b) Evalúa 5 scalers (Standard, MinMax, Robust, QuantileNormal, PowerTransformer) + sin escalar usando 3 modelos (KNN, SVM, RandomForest), (c) Para cada combinación, reporta R² medio de 5-fold CV, (d) Crea un heatmap (scalers × modelos) con los resultados, (e) Concluye: qué scaler recomiendas para cada tipo de modelo.
