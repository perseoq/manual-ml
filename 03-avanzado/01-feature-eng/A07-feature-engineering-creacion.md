# A07: Feature Engineering — Creación de Features

## 1. Introducción Teórica

Feature Engineering es el proceso de transformar datos crudos en variables (features) que maximizan el poder predictivo de modelos de Machine Learning. En contextos de ventas, compras e inventarios, la calidad de los features suele importar más que la complejidad del modelo.

**Principios clave:**
- **Dominio del negocio:** Conocer el ciclo de ventas, estacionalidad, promociones y comportamiento de inventario guía la creación de features.
- **Dimensionalidad:** Más features no siempre es mejor; el overfitting crece con features irrelevantes.
- **Información temporal:** En series de ventas, el orden temporal es crítico — no se debe hacer leakage usando información futura.

**Tipos de features:**

| Tipo | Ejemplo en ventas |
|------|-------------------|
| Polinomiales | precio², precio × cantidad |
| Splines | Transformaciones no lineales de precio |
| Temporales | día_semana, mes, trimestre |
| Rezago (lag) | demanda_1, demanda_7, demanda_30 |
| Ventana móvil | mean_7, std_7 |
| Ratio | margen/precio, precio/costo |
| Interacción | precio × categoría |
| Encoding | target encoding, frequency encoding |
| Agregaciones | precio medio por categoría |

---

## 2. Ejemplos Ejecutables

### 2.1. Configuración Inicial

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import (
    PolynomialFeatures, SplineTransformer, KBinsDiscretizer,
    MinMaxScaler, LabelEncoder
)
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

fechas = pd.date_range('2023-01-01', '2024-12-31', freq='D')
n = len(fechas)

ventas = pd.DataFrame({
    'fecha': fechas,
    'precio': np.random.uniform(10, 500, n),
    'costo': np.random.uniform(5, 300, n),
    'cantidad': np.random.poisson(30, n),
    'categoria': np.random.choice(['electronica', 'ropa', 'hogar', 'deportes', 'libros'], n),
    'sku': np.random.choice([f'SKU_{i:04d}' for i in range(200)], n),
    'promocion': np.random.choice([0, 1], n, p=[0.85, 0.15]),
    'stock': np.random.randint(0, 500, n)
})

ventas['ingreso'] = ventas['precio'] * ventas['cantidad']
ventas['margen'] = ventas['precio'] - ventas['costo']
print(ventas.shape)
ventas.head()
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

*2.1. Configuración Inicial.*

1. `import numpy as np` — Importa las librerías necesarias para el análisis.
2. `import pandas as pd` — Importa las librerías necesarias para el análisis.
3. `import matplotlib.pyplot as plt` — Importa las librerías necesarias para el análisis.
4. `import seaborn as sns` — Importa las librerías necesarias para el análisis.
5. `from sklearn.preprocessing import (` — Importa las librerías necesarias para el análisis.
6. `from sklearn.feature_selection import SelectKBest, f_regression` — Importa las librerías necesarias para el análisis.
7. `from sklearn.model_selection import train_test_split` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### 2.2. PolynomialFeatures — Grado 2

```python
from sklearn.preprocessing import PolynomialFeatures

X_poly = ventas[['precio', 'cantidad']].values[:1000]
poly = PolynomialFeatures(degree=2, include_bias=True, interaction_only=False)
X_poly_transformed = poly.fit_transform(X_poly)

df_poly = pd.DataFrame(
    X_poly_transformed,
    columns=poly.get_feature_names_out(['precio', 'cantidad'])
)
df_poly.head()
```

**Salida esperada:**
| 1 | precio | cantidad | precio^2 | precio cantidad | cantidad^2 |
|---|---|---|---|---|---|

---

### 2.3. PolynomialFeatures — Solo Interacciones



**Salida esperada:**
| 1 | precio | cantidad | precio^2 | precio cantidad | cantidad^2 |
|---|---|---|---|---|---|

---

### 2.3. PolynomialFeatures — Solo Interacciones

```python
poly_inter = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_inter = poly_inter.fit_transform(X_poly)

df_inter = pd.DataFrame(
    X_inter,
    columns=poly_inter.get_feature_names_out(['precio', 'cantidad'])
)
df_inter.head()
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*2.3. PolynomialFeatures — Solo Interacciones.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



Genera solo `precio * cantidad` (sin términos cuadráticos).

---

### 2.4. SplineTransformer — Bases Spline para Precio

```python
spline = SplineTransformer(n_knots=5, degree=3, knots='quantile', extrapolation='constant', include_bias=True)
X_spline = spline.fit_transform(ventas[['precio']].values[:1000])

df_spline = pd.DataFrame(
    X_spline,
    columns=[f'spline_precio_b{i}' for i in range(X_spline.shape[1])]
)
df_spline.head()
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*2.4. SplineTransformer — Bases Spline para Precio.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



Los spline basis permiten modelar relaciones no lineales sin forzar una forma polinomial específica.

---

### 2.5. Lag Features — Shift(1), Shift(7), Shift(30)

```python
ventas_ordenado = ventas.sort_values('fecha').copy()

ventas_ordenado['demanda_lag_1'] = ventas_ordenado['cantidad'].shift(1)
ventas_ordenado['demanda_lag_7'] = ventas_ordenado['cantidad'].shift(7)
ventas_ordenado['demanda_lag_30'] = ventas_ordenado['cantidad'].shift(30)

ventas_ordenado[['fecha', 'cantidad', 'demanda_lag_1', 'demanda_lag_7', 'demanda_lag_30']].head(35)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*2.5. Lag Features — Shift(1), Shift(7), Shift(30).*

1. `ventas_ordenado = ventas.sort_values('fecha').copy()` — Ordena los datos según la columna eRealiza la operación indicada con los parámetros definidos..

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



> ⚠️ **Precaución:** Los primeros N valores serán NaN (N = lag). Se deben eliminar o imputar.

---

### 2.6. Rolling Features — Media, Std, Min, Max(7)

```python
ventas_ordenado['rolling_mean_7'] = ventas_ordenado['cantidad'].rolling(window=7).mean()
ventas_ordenado['rolling_std_7'] = ventas_ordenado['cantidad'].rolling(window=7).std()
ventas_ordenado['rolling_min_7'] = ventas_ordenado['cantidad'].rolling(window=7).min()
ventas_ordenado['rolling_max_7'] = ventas_ordenado['cantidad'].rolling(window=7).max()

ventas_ordenado[['fecha', 'cantidad', 'rolling_mean_7', 'rolling_std_7', 'rolling_min_7', 'rolling_max_7']].dropna().head()
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*2.6. Rolling Features — Media, Std, Min, Max(7).*

1. `ventas_ordenado['rolling_mean_7'] = ventas_ordenado['cantidad'].rolling(window=7).mean()` — Crea una ventana deslizante para cálculos móviles.
2. `ventas_ordenado['rolling_std_7'] = ventas_ordenado['cantidad'].rolling(window=7).std()` — Crea una ventana deslizante para cálculos móviles.
3. `ventas_ordenado['rolling_min_7'] = ventas_ordenado['cantidad'].rolling(window=7).min()` — Crea una ventana deslizante para cálculos móviles.
4. `ventas_ordenado['rolling_max_7'] = ventas_ordenado['cantidad'].rolling(window=7).max()` — Crea una ventana deslizante para cálculos móviles.
5. `ventas_ordenado[['fecha', 'cantidad', 'rolling_mean_7', 'rolling_std_7', 'rolling_min_7', 'rolling_max_7']].dropna().head()` — Elimina las filas con valores nulos.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### 2.7. Expanding Features — Media y Máximo Acumulados

```python
ventas_ordenado['expanding_mean'] = ventas_ordenado['cantidad'].expanding().mean()
ventas_ordenado['expanding_max'] = ventas_ordenado['cantidad'].expanding().max()

ventas_ordenado[['fecha', 'cantidad', 'expanding_mean', 'expanding_max']].head()
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*2.7. Expanding Features — Media y Máximo Acumulados.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### 2.8. Ratio Features — Margen/ Precio, Precio/ Costo, Cantidad/ Stock

```python
ventas_ordenado['ratio_margen_precio'] = ventas_ordenado['margen'] / (ventas_ordenado['precio'] + 1e-6)
ventas_ordenado['ratio_precio_costo'] = ventas_ordenado['precio'] / (ventas_ordenado['costo'] + 1e-6)
ventas_ordenado['ratio_cantidad_stock'] = ventas_ordenado['cantidad'] / (ventas_ordenado['stock'] + 1e-6)

ventas_ordenado[['fecha', 'ratio_margen_precio', 'ratio_precio_costo', 'ratio_cantidad_stock']].head()
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*2.8. Ratio Features — Margen/ Precio, Precio/ Costo, Cantidad/ Stock.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### 2.9. Features Temporales desde Fecha

```python
ventas_ordenado['dia_semana'] = ventas_ordenado['fecha'].dt.dayofweek  # 0=lunes
ventas_ordenado['mes'] = ventas_ordenado['fecha'].dt.month
ventas_ordenado['trimestre'] = ventas_ordenado['fecha'].dt.quarter
ventas_ordenado['fin_de_semana'] = (ventas_ordenado['dia_semana'] >= 5).astype(int)
ventas_ordenado['dia_del_ano'] = ventas_ordenado['fecha'].dt.dayofyear
ventas_ordenado['semana_del_ano'] = ventas_ordenado['fecha'].dt.isocalendar().week.astype(int)

ventas_ordenado[['fecha', 'dia_semana', 'mes', 'trimestre', 'fin_de_semana', 'semana_del_ano']].head()
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*2.9. Features Temporales desde Fecha.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### 2.10. Días desde Último Evento (Promoción)

```python
ultima_promo = None
dias_desde_promo = []

for p in ventas_ordenado['promocion']:
    if p == 1:
        ultima_promo = 0
    elif ultima_promo is not None:
        ultima_promo += 1
    dias_desde_promo.append(ultima_promo if ultima_promo is not None else np.nan)

ventas_ordenado['dias_desde_promocion'] = dias_desde_promo
ventas_ordenado[['fecha', 'promocion', 'dias_desde_promocion']].head(20)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*2.10. Días desde Último Evento (Promoción).*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### 2.11. Target Encoding — Media de Ventas por Categoría

```python
from category_encoders import TargetEncoder

X = ventas_ordenado[['categoria', 'precio', 'cantidad']].dropna()
y = ventas_ordenado.loc[X.index, 'ingreso']

encoder_target = TargetEncoder(cols=['categoria'], smoothing=10.0)
X_encoded = encoder_target.fit_transform(X[['categoria']], y)

ventas_ordenado.loc[X.index, 'target_enc_categoria'] = X_encoded.values
ventas_ordenado[['categoria', 'target_enc_categoria']].drop_duplicates().head()
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*2.11. Target Encoding — Media de Ventas por Categoría.*

1. `from category_encoders import TargetEncoder` — Importa las librerías necesarias para el análisis.
2. `X = ventas_ordenado[['categoria', 'precio', 'cantidad']].dropna()` — Elimina las filas con valores nulos.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### 2.12. Frequency Encoding — Frecuencia de Cada Categoría

```python
freq_categoria = ventas_ordenado['categoria'].value_counts() / len(ventas_ordenado)
ventas_ordenado['freq_categoria'] = ventas_ordenado['categoria'].map(freq_categoria)

freq_categoria
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*2.12. Frequency Encoding — Frecuencia de Cada Categoría.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### 2.13. Count Encoding — Número de Apariciones de Cada SKU

```python
count_sku = ventas_ordenado['sku'].value_counts()
ventas_ordenado['count_sku'] = ventas_ordenado['sku'].map(count_sku)

ventas_ordenado[['sku', 'count_sku']].drop_duplicates().head(10)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*2.13. Count Encoding — Número de Apariciones de Cada SKU.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### 2.14. Aggregations por Categoría — Precio Medio y Std

```python
agg_categoria = ventas_ordenado.groupby('categoria')['precio'].agg(['mean', 'std', 'min', 'max'])
agg_categoria.columns = ['precio_mean_cat', 'precio_std_cat', 'precio_min_cat', 'precio_max_cat']

ventas_ordenado = ventas_ordenado.merge(agg_categoria, on='categoria', how='left')
ventas_ordenado[['categoria', 'precio', 'precio_mean_cat', 'precio_std_cat']].head()
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*2.14. Aggregations por Categoría — Precio Medio y Std.*

1. `agg_categoria = ventas_ordenado.groupby('categoria')['precio'].agg(['mean', 'std', 'min', 'max'])` — Agrupa los datos por la columna eRealiza la operación indicada con los parámetros definidos..
2. `ventas_ordenado = ventas_ordenado.merge(agg_categoria, on='categoria', how='left')` — Combina dos DataFrames por una columna clave.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### 2.15. Interacción: Precio × Categoría OneHot

```python
cat_dummies = pd.get_dummies(ventas_ordenado['categoria'], prefix='cat')
for col in cat_dummies.columns:
    ventas_ordenado[f'interac_precio_{col}'] = ventas_ordenado['precio'] * cat_dummies[col]

interac_cols = [c for c in ventas_ordenado.columns if 'interac_precio_cat' in c]
ventas_ordenado[['precio'] + interac_cols].head()
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*2.15. Interacción: Precio × Categoría OneHot.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### 2.16. Binarización — Margen > 30%

```python
ventas_ordenado['flag_margen_alto'] = (ventas_ordenado['margen'] / (ventas_ordenado['precio'] + 1e-6) > 0.30).astype(int)
ventas_ordenado[['precio', 'costo', 'margen', 'flag_margen_alto']].head()
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*2.16. Binarización — Margen > 30%.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### 2.17. Binning con KBinsDiscretizer + OneHot

```python
from sklearn.preprocessing import KBinsDiscretizer

kbd = KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='quantile')
precio_bins = kbd.fit_transform(ventas_ordenado[['precio']].values[:1000])

df_bins = pd.DataFrame(
    precio_bins,
    columns=[f'precio_bin_{i}' for i in range(precio_bins.shape[1])]
)
df_bins.head()
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*2.17. Binning con KBinsDiscretizer + OneHot.*

1. `from sklearn.preprocessing import KBinsDiscretizer` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### 2.18. Encoding Cíclico — Seno/Coseno para Día de Semana y Mes

```python
ventas_ordenado['dia_semana_sin'] = np.sin(2 * np.pi * ventas_ordenado['dia_semana'] / 7)
ventas_ordenado['dia_semana_cos'] = np.cos(2 * np.pi * ventas_ordenado['dia_semana'] / 7)
ventas_ordenado['mes_sin'] = np.sin(2 * np.pi * ventas_ordenado['mes'] / 12)
ventas_ordenado['mes_cos'] = np.cos(2 * np.pi * ventas_ordenado['mes'] / 12)

ventas_ordenado[['dia_semana', 'dia_semana_sin', 'dia_semana_cos', 'mes', 'mes_sin', 'mes_cos']].head()
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*2.18. Encoding Cíclico — Seno/Coseno para Día de Semana y Mes.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### 2.19. Integrador: Crear 30+ Features desde ventas.csv

```python
def crear_features_ventas(df):
    df = df.sort_values('fecha').copy()

    # Temporales
    df['dia_semana'] = df['fecha'].dt.dayofweek
    df['mes'] = df['fecha'].dt.month
    df['trimestre'] = df['fecha'].dt.quarter
    df['fin_de_semana'] = (df['dia_semana'] >= 5).astype(int)
    df['dia_del_ano'] = df['fecha'].dt.dayofyear

    # Cíclicos
    df['dia_semana_sin'] = np.sin(2 * np.pi * df['dia_semana'] / 7)
    df['dia_semana_cos'] = np.cos(2 * np.pi * df['dia_semana'] / 7)
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)

    # Ratios
    df['ratio_margen_precio'] = df['margen'] / (df['precio'] + 1e-6)
    df['ratio_precio_costo'] = df['precio'] / (df['costo'] + 1e-6)
    df['ratio_cantidad_stock'] = df['cantidad'] / (df['stock'] + 1e-6)

    # Lags
    df['lag_1'] = df['cantidad'].shift(1)
    df['lag_7'] = df['cantidad'].shift(7)
    df['lag_30'] = df['cantidad'].shift(30)

    # Rolling
    df['rolling_mean_7'] = df['cantidad'].rolling(7).mean()
    df['rolling_std_7'] = df['cantidad'].rolling(7).std()
    df['rolling_max_7'] = df['cantidad'].rolling(7).max()
    df['rolling_min_7'] = df['cantidad'].rolling(7).min()

    # Expanding
    df['expanding_mean'] = df['cantidad'].expanding().mean()
    df['expanding_max'] = df['cantidad'].expanding().max()

    # Flags
    df['flag_margen_alto'] = (df['ratio_margen_precio'] > 0.30).astype(int)
    df['flag_promocion'] = df['promocion']

    # Frecuencia categoría
    freq = df['categoria'].value_counts() / len(df)
    df['freq_categoria'] = df['categoria'].map(freq)

    # Count SKU
    cnt = df['sku'].value_counts()
    df['count_sku'] = df['sku'].map(cnt)

    # Agregaciones por categoría
    aggs = df.groupby('categoria')['precio'].agg(['mean', 'std']).rename(
        columns={'mean': 'precio_mean_cat', 'std': 'precio_std_cat'}
    )
    df = df.merge(aggs, on='categoria', how='left')

    return df

ventas_con_features = crear_features_ventas(ventas_ordenado)
print(f"Features creadas: {ventas_con_features.shape[1]}")
list(ventas_con_features.columns)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*2.19. Integrador: Crear 30+ Features desde ventas.csv.*

1. Temporales
2. Cíclicos
3. Ratios
4. Lags
5. Rolling
6. Expanding
7. Flags
8. Frecuencia categoría

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 3. Resumen

| Técnica | Uso principal | Consideraciones |
|---------|---------------|-----------------|
| PolynomialFeatures | Relaciones cuadráticas/exponenciales | Puede causar overfitting si degree es alto |
| SplineTransformer | No linealidad suave | Elegir bien número de knots |
| Lag features | Dependencia temporal | Cuidado con leakage si no se ordena |
| Rolling features | Tendencia y volatilidad | Ventana debe reflejar ciclo de negocio |
| Ratio features | Eficiencia y rentabilidad | Posible división por cero |
| Target Encoding | Alta cardinalidad | Riesgo de overfitting sin smoothing |
| Frequency/Count | Popularidad de categorías | Pierde relación con target |
| Cíclico sin/cos | Periodicidad | Mantiene distancia circular |
| Binning | No linealidad | Pierde granularidad |

**Regla de oro:** Validar siempre con temporal train/test split. No usar información futura.

---

## 4. Ejercicios

1. Crea un feature `rolling_median_14` para cantidad con ventana de 14 días.
2. Genera PolynomialFeatures con degree=3 para precio, cantidad y costo. ¿Cuántos features se crean?
3. Implementa un feature `dias_desde_cambio_precio` que cuente días desde la última vez que el precio cambió más de 10%.
4. Usa SplineTransformer con n_knots=10 y degree=5 sobre la columna precio. Compara visualmente con degree=3.
5. Crea features de interacción triple: precio × cantidad × promoción usando PolynomialFeatures.
6. Implementa Target Encoding con smoothing=0.0 y smoothing=100.0. ¿Cómo cambian los valores codificados?
7. Genera rolling features para 3, 7, 14, 30 días. Calcula la correlación de cada una con la demanda futura (shift -1).
8. Diseña un feature compuesto: `score_rotacion = cantidad / (stock + 1) * (1 + promocion * 0.5)`. ¿Qué información captura?

---

> **Siguiente tema:** [A08 — Feature Selection](A08-feature-selection.md)
