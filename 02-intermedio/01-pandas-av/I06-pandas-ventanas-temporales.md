# I06 — Ventanas Temporales en Pandas (Rolling, Expanding, EWM)

## 1. Introducción Teórica

Las ventanas temporales permiten calcular estadísticas sobre subconjuntos deslizantes de datos, esenciales para analizar tendencias de ventas, suavizar ruido y detectar cambios.

### Tipos de ventanas

**Rolling (ventana deslizante fija):**
- `rolling(window)`: Ventana de tamaño fijo
- `min_periods`: Mínimo de observaciones requeridas
- `center=True`: Ventana centrada (usa datos pasados y futuros)
- `win_type`: Tipo de ponderación (triang, blackman, hamming, bartlett, hanning)
- `on`: Columna de fecha para ventana basada en tiempo

**Expanding (ventana expansiva):**
- `expanding()`: Crece desde el inicio hasta el punto actual
- `min_periods`: Mínimo de observaciones para comenzar

**EWM — Exponentially Weighted (suavizado exponencial):**
- `ewm(span=)`: Periodo equivalente a media móvil simple
- `ewm(alpha=)`: Factor de suavizado directo (0 < alpha ≤ 1)
- `ewm(halflife=)`: Tiempo para que el peso se reduzca a la mitad
- `ewm(com=)`: Centro de masa
- `adjust=True/False`: Controla cómo se normalizan los pesos

### Operaciones derivadas

- **shift(n):** Desplaza datos n períodos (lag/lead)
- **diff(n):** Diferencia con n períodos atrás
- **pct_change(n):** Cambio porcentual vs n períodos atrás
- **rolling().corr():** Correlación móvil entre dos series
- **rolling().cov():** Covarianza móvil

---

## 2. Ejemplos Prácticos

### Ejemplo 1: rolling(7).mean() — media móvil semanal

```python
import pandas as pd
import numpy as np

np.random.seed(42)
fechas = pd.date_range('2024-01-01', periods=100, freq='D')
ventas = pd.DataFrame({
    'fecha': fechas,
    'ingreso': np.random.normal(5000, 1000, 100).cumsum() + 10000
})

ventas['media_7'] = ventas['ingreso'].rolling(7).mean()

print(ventas.head(14))
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

*Ejemplo 1: rolling(7).mean() — media móvil semanal.*

1. `import pandas as pd` — Importa las librerías necesarias para el análisis.
2. `import numpy as np` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: rolling(30).mean() — media móvil mensual

```python
ventas['media_30'] = ventas['ingreso'].rolling(30).mean()

print(ventas[['fecha', 'ingreso', 'media_30']].iloc[25:35])
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: rolling(30).mean() — media móvil mensual.*

1. `ventas['media_30'] = ventas['ingreso'].rolling(30).mean()` — Crea una ventana deslizante para cálculos móviles.
2. `print(ventas[['fecha', 'ingreso', 'media_30']].iloc[25:35])` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: rolling con min_periods

```python
ventas['media_7_parcial'] = ventas['ingreso'].rolling(7, min_periods=3).mean()

print(ventas[['fecha', 'ingreso', 'media_7', 'media_7_parcial']].head(10))
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

*Ejemplo 3: rolling con min_periods.*

1. `ventas['media_7_parcial'] = ventas['ingreso'].rolling(7, min_periods=3).mean()` — Crea una ventana deslizante para cálculos móviles.
2. `print(ventas[['fecha', 'ingreso', 'media_7', 'media_7_parcial']].head(10))` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: rolling(7, center=True) — ventana centrada

```python
ventas['media_7_center'] = ventas['ingreso'].rolling(7, center=True).mean()

print(ventas[['fecha', 'ingreso', 'media_7', 'media_7_center']].iloc[3:11])
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: rolling(7, center=True) — ventana centrada.*

1. `ventas['media_7_center'] = ventas['ingreso'].rolling(7, center=True).mean()` — Crea una ventana deslizante para cálculos móviles.
2. `print(ventas[['fecha', 'ingreso', 'media_7', 'media_7_center']].iloc[3:11])` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: rolling con win_type="triang"

```python
ventas['media_triang'] = ventas['ingreso'].rolling(7, win_type='triang').mean()
ventas['media_hamming'] = ventas['ingreso'].rolling(7, win_type='hamming').mean()
ventas['media_blackman'] = ventas['ingreso'].rolling(7, win_type='blackman').mean()

print(ventas[['fecha', 'ingreso', 'media_triang', 'media_hamming', 'media_blackman']].head(10))
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

*Ejemplo 5: rolling con win_type="triang".*

1. `ventas['media_triang'] = ventas['ingreso'].rolling(7, win_type='triang').mean()` — Crea una ventana deslizante para cálculos móviles.
2. `ventas['media_hamming'] = ventas['ingreso'].rolling(7, win_type='hamming').mean()` — Crea una ventana deslizante para cálculos móviles.
3. `ventas['media_blackman'] = ventas['ingreso'].rolling(7, win_type='blackman').mean()` — Crea una ventana deslizante para cálculos móviles.
4. `print(ventas[['fecha', 'ingreso', 'media_triang', 'media_hamming', 'media_blackman']].head(10))` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: rolling(7).apply() — función personalizada

```python
def rango_ventana(x):
    return x.max() - x.min()

def cv_ventana(x):
    return np.std(x) / np.mean(x) * 100

ventas['rango_7'] = ventas['ingreso'].rolling(7).apply(rango_ventana)
ventas['cv_7'] = ventas['ingreso'].rolling(7).apply(cv_ventana)

print(ventas[['fecha', 'ingreso', 'rango_7', 'cv_7']].head(10))
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

*Ejemplo 6: rolling(7).apply() — función personalizada.*

1. `ventas['rango_7'] = ventas['ingreso'].rolling(7).apply(rango_ventana)` — Crea una ventana deslizante para cálculos móviles.
2. `ventas['cv_7'] = ventas['ingreso'].rolling(7).apply(cv_ventana)` — Crea una ventana deslizante para cálculos móviles.
3. `print(ventas[['fecha', 'ingreso', 'rango_7', 'cv_7']].head(10))` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: expanding().sum() — suma acumulada

```python
ventas['acumulado'] = ventas['ingreso'].expanding().sum()

print(ventas[['fecha', 'ingreso', 'acumulado']].head(10))
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

*Ejemplo 7: expanding().sum() — suma acumulada.*

1. `print(ventas[['fecha', 'ingreso', 'acumulado']].head(10))` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: expanding().max() — máximo acumulado

```python
ventas['max_acum'] = ventas['ingreso'].expanding().max()

print(ventas[['fecha', 'ingreso', 'max_acum']].head(15))
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

*Ejemplo 8: expanding().max() — máximo acumulado.*

1. `print(ventas[['fecha', 'ingreso', 'max_acum']].head(15))` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: expanding().mean() con min_periods

```python
ventas['media_exp'] = ventas['ingreso'].expanding(min_periods=5).mean()

print(ventas[['fecha', 'ingreso', 'media_exp']].head(12))
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

*Ejemplo 9: expanding().mean() con min_periods.*

1. `print(ventas[['fecha', 'ingreso', 'media_exp']].head(12))` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: ewm(span=7).mean() — suavizado exponencial

```python
ventas['ewm_span7'] = ventas['ingreso'].ewm(span=7).mean()

print(ventas[['fecha', 'ingreso', 'media_7', 'ewm_span7']].head(15))
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

*Ejemplo 10: ewm(span=7).mean() — suavizado exponencial.*

1. `print(ventas[['fecha', 'ingreso', 'media_7', 'ewm_span7']].head(15))` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: ewm(alpha=0.3).mean()

```python
ventas['ewm_alpha03'] = ventas['ingreso'].ewm(alpha=0.3).mean()
ventas['ewm_alpha07'] = ventas['ingreso'].ewm(alpha=0.7).mean()

print(ventas[['fecha', 'ingreso', 'ewm_alpha03', 'ewm_alpha07']].head(10))
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

*Ejemplo 11: ewm(alpha=0.3).mean().*

1. `print(ventas[['fecha', 'ingreso', 'ewm_alpha03', 'ewm_alpha07']].head(10))` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: ewm(halflife=3).mean()

```python
ventas['ewm_halflife3'] = ventas['ingreso'].ewm(halflife=3).mean()
ventas['ewm_halflife7'] = ventas['ingreso'].ewm(halflife=7).mean()

print(ventas[['fecha', 'ingreso', 'ewm_halflife3', 'ewm_halflife7']].head(10))
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

*Ejemplo 12: ewm(halflife=3).mean().*

1. `print(ventas[['fecha', 'ingreso', 'ewm_halflife3', 'ewm_halflife7']].head(10))` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: ewm(adjust=False).mean()

```python
ventas['ewm_adj_true'] = ventas['ingreso'].ewm(span=7, adjust=True).mean()
ventas['ewm_adj_false'] = ventas['ingreso'].ewm(span=7, adjust=False).mean()

print(ventas[['fecha', 'ingreso', 'ewm_adj_true', 'ewm_adj_false']].head(10))
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

*Ejemplo 13: ewm(adjust=False).mean().*

1. `print(ventas[['fecha', 'ingreso', 'ewm_adj_true', 'ewm_adj_false']].head(10))` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: ewm().std() — desviación exponencial

```python
ventas['ewm_std'] = ventas['ingreso'].ewm(span=30).std()
ventas['banda_sup'] = ventas['ewm_span7'] + 2 * ventas['ewm_std']
ventas['banda_inf'] = ventas['ewm_span7'] - 2 * ventas['ewm_std']

print(ventas[['fecha', 'ingreso', 'ewm_span7', 'banda_sup', 'banda_inf']].head(10))
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

*Ejemplo 14: ewm().std() — desviación exponencial.*

1. `print(ventas[['fecha', 'ingreso', 'ewm_span7', 'banda_sup', 'banda_inf']].head(10))` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: ewm().corr() entre dos series

```python
# Correlación exponencial entre ingresos y cantidad
np.random.seed(42)
n = 100
df = pd.DataFrame({
    'fecha': pd.date_range('2024-01-01', periods=n, freq='D'),
    'ingreso': np.random.normal(5000, 1000, n).cumsum() + 10000,
    'cantidad': np.random.normal(50, 10, n).cumsum() + 200
})

df['ewm_corr'] = df['ingreso'].ewm(span=30).corr(df['cantidad'])
print(df[['fecha', 'ingreso', 'cantidad', 'ewm_corr']].head(15))
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

*Ejemplo 15: ewm().corr() entre dos series.*

1. Correlación exponencial entre ingresos y cantidad

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 16: shift(1) — comparar día actual vs anterior

```python
ventas['ingreso_ayer'] = ventas['ingreso'].shift(1)
ventas['cambio_abs'] = ventas['ingreso'] - ventas['ingreso_ayer']

print(ventas[['fecha', 'ingreso', 'ingreso_ayer', 'cambio_abs']].head(10))
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

*Ejemplo 16: shift(1) — comparar día actual vs anterior.*

1. `print(ventas[['fecha', 'ingreso', 'ingreso_ayer', 'cambio_abs']].head(10))` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 17: shift(-1) — "adelantar" datos (lead)

```python
ventas['ingreso_manana'] = ventas['ingreso'].shift(-1)
ventas['cambio_manana'] = ventas['ingreso_manana'] - ventas['ingreso']

print(ventas[['fecha', 'ingreso', 'ingreso_manana', 'cambio_manana']].head(10))
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

*Ejemplo 17: shift(-1) — "adelantar" datos (lead).*

1. `print(ventas[['fecha', 'ingreso', 'ingreso_manana', 'cambio_manana']].head(10))` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 18: diff(1) — cambio día a día

```python
ventas['diff_1'] = ventas['ingreso'].diff(1)
ventas['diff_7'] = ventas['ingreso'].diff(7)

print(ventas[['fecha', 'ingreso', 'diff_1', 'diff_7']].head(14))
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

*Ejemplo 18: diff(1) — cambio día a día.*

1. `print(ventas[['fecha', 'ingreso', 'diff_1', 'diff_7']].head(14))` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 19: pct_change(7) — cambio % vs semana anterior

```python
ventas['pct_1d'] = ventas['ingreso'].pct_change(1) * 100
ventas['pct_7d'] = ventas['ingreso'].pct_change(7) * 100

print(ventas[['fecha', 'ingreso', 'pct_1d', 'pct_7d']].head(14))
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

*Ejemplo 19: pct_change(7) — cambio % vs semana anterior.*

1. `print(ventas[['fecha', 'ingreso', 'pct_1d', 'pct_7d']].head(14))` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 20: rolling(30).corr() — correlación móvil

```python
df['rolling_corr_30'] = df['ingreso'].rolling(30).corr(df['cantidad'])

print(df[['fecha', 'ingreso', 'cantidad', 'rolling_corr_30']].dropna().head(10))
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

*Ejemplo 20: rolling(30).corr() — correlación móvil.*

1. `df['rolling_corr_30'] = df['ingreso'].rolling(30).corr(df['cantidad'])` — Crea una ventana deslizante para cálculos móviles.
2. `print(df[['fecha', 'ingreso', 'cantidad', 'rolling_corr_30']].dropna().head(10))` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 3. Resumen Teórico

| Función | Aplicación en Ventas/Compras/Inventarios |
|---------|----------------------------------------|
| rolling().mean() | Tendencia de ventas, suavizar ruido diario |
| rolling().apply() | Métricas personalizadas (rango, CV) por ventana |
| expanding().sum() | Ventas acumuladas YTD, MTD |
| ewm().mean() | Suavizado exponencial, ponderar más datos recientes |
| ewm().std() | Bandas de volatilidad para inventario de seguridad |
| shift() | Lag/lead para modelos predictivos, comparación vs ayer |
| diff() | Cambios absolutos diarios/semanales |
| pct_change() | Tasas de crecimiento, comparación interanual |
| rolling().corr() | Correlación dinámica entre precio y demanda |

---

## 4. Ejercicios Propuestos

**Ejercicio 1:** Calcula la media móvil de 14 días para ventas diarias y compárala con la media móvil de 7 días. ¿Cuál suaviza más?

**Ejercicio 2:** Usa rolling(30).apply() para calcular el coeficiente de variación mensual del ingreso.

**Ejercicio 3:** Implementa un sistema de alerta: marca días donde el ingreso cae por debajo de `media_movil_7 - 2*std_movil_7`.

**Ejercicio 4:** Calcula el acumulado de ventas por mes usando expanding().sum() con reinicio al inicio de cada mes.

**Ejercicio 5:** Compara ewm(span=7) vs rolling(7).mean(): ¿cuál reacciona más rápido a cambios recientes?

**Ejercicio 6:** Usa shift() para calcular si las ventas de hoy son mayores que las de ayer y las de la semana pasada.

**Ejercicio 7:** Calcula la correlación móvil de 60 días entre el precio de un producto y su cantidad vendida.

**Ejercicio 8:** Diferencia temporal: usa diff(7) para eliminar la estacionalidad semanal de las ventas y analiza la serie resultante.

---

*Fin del documento I06 — Ventanas Temporales en Pandas*
