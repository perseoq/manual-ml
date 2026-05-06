# B28 — Estadística descriptiva avanzada con `scipy.stats`

La estadística descriptiva resume y describe las características principales de un conjunto de datos. `scipy.stats` ofrece funciones que van más allá de `pandas.describe()`: momentos estadísticos, correlaciones robustas, detección de outliers, estadística circular, entre otras.

## Tabla de funciones cubiertas

| Función | Descripción |
|---------|-------------|
| `stats.describe` | Estadísticas completas (nobs, minmax, mean, variance, skewness, kurtosis) |
| `stats.gmean` | Media geométrica (crecimiento porcentual, tasas) |
| `stats.hmean` | Media armónica (tasas, velocidades) |
| `stats.mode` | Moda (valor más frecuente) |
| `stats.skew` | Asimetría (sesgo de la distribución) |
| `stats.kurtosis` | Curtosis (peso de las colas) |
| `stats.trim_mean` | Media truncada (eliminar extremos) |
| `stats.sem` | Error estándar de la media |
| `stats.variation` | Coeficiente de variación (dispersión relativa) |
| `stats.moment` | Momento estadístico de orden n |
| `stats.zscore` | Puntuación Z (detección de outliers) |
| `stats.pearsonr` | Correlación de Pearson (lineal) |
| `stats.spearmanr` | Correlación de Spearman (monótona, no paramétrica) |
| `stats.kendalltau` | Tau de Kendall (correlación ordinal) |
| `stats.linregress` | Regresión lineal completa (pendiente, intercepto, p-valor, etc.) |
| `stats.iqr` | Rango intercuartil (robusto ante outliers) |
| `stats.circmean` / `circvar` | Estadística circular (ángulos, estacionalidad) |
| `stats.pointbiserialr` | Correlación punto-biserial (binaria vs continua) |

---

## Configuración inicial

```python
import pandas as pd
import numpy as np
from scipy import stats
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
3. `from scipy import stats` — Importa las librerías necesarias para el análisis.
4. `import matplotlib.pyplot as plt` — Importa las librerías necesarias para el análisis.
5. `import warnings` — Importa las librerías necesarias para el análisis.
6. `ventas = pd.read_csv("../datos/ventas.csv")` — Carga los datos desde el archivo CSV.
7. `inventario = pd.read_csv("../datos/inventario.csv")` — Carga los datos desde el archivo CSV.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejemplo 1 — `stats.describe`: estadísticas completas de ingresos

```python
# describe devuelve: nobs, minmax, mean, variance, skewness, kurtosis
nobs, (vmin, vmax), mean, var, skew, kurt = stats.describe(ventas["ingreso"])

print(f"Observaciones: {nobs}")
print(f"Mínimo: ${vmin:,.0f} — Máximo: ${vmax:,.0f}")
print(f"Media: ${mean:,.0f}")
print(f"Varianza: {var:,.0f}")
print(f"Asimetría (skewness): {skew:.3f}")
print(f"Curtosis (kurtosis): {kurt:.3f}")

# Salida:
# Observaciones: 1330
# Mínimo: $340 — Máximo: $390,000
# Media: $24,809
# Varianza: 1,423,638,165
# Asimetría (skewness): 3.109
# Curtosis (kurtosis): 15.135
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1 — `stats.describe`: estadísticas completas de ingresos.*

1. describe devuelve: nobs, minmax, mean, variance, skewness, kurtosis
2. Salida:
3. Observaciones: 1330
4. Mínimo: $340 — Máximo: $390,000
5. Media: $24,809
6. Varianza: 1,423,638,165
7. Asimetría (skewness): 3.109
8. Curtosis (kurtosis): 15.135

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** La asimetría positiva (3.11) indica cola derecha larga — pocas ventas muy altas elevan la media. La curtosis > 3 indica distribución leptocúrtica (colas pesadas), típica en ventas donde hay valores extremos.

---

## Ejemplo 2 — `stats.gmean`: media geométrica de precios en inventario

Útil para tasas de crecimiento: si un precio sube 10%, luego 15%, la media geométrica da la tasa promedio real.

```python
precios = inventario["precio"]
gmean_precio = stats.gmean(precios)

print(f"Media geométrica de precios: ${gmean_precio:.2f}")
print(f"Media aritmética: ${precios.mean():.2f}")
print(f"Mediana: ${precios.median():.2f}")

# Salida:
# Media geométrica de precios: $1903.33
# Media aritmética: $3178.00
# Mediana: $1800.00
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2 — `stats.gmean`: media geométrica de precios en inventario.*

1. Salida:
2. Media geométrica de precios: $1903.33
3. Media aritmética: $3178.00
4. Mediana: $1800.00

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** La media geométrica ($1,903) es menor que la aritmética ($3,178) porque la geométrica penaliza valores extremos altos. Es más representativa para entender el "preicio típico" en inventarios con grandes diferencias (teclado $350 vs laptop $15,000).

---

## Ejemplo 3 — `stats.hmean`: media armónica de márgenes

La media armónica es apropiada para promediar tasas (margen_pct es un porcentaje/margen).

```python
margenes = ventas["margen_pct"]
hmean_margen = stats.hmean(margenes)

print(f"Media armónica del margen: {hmean_margen:.2f}%")
print(f"Media aritmética: {margenes.mean():.2f}%")
print(f"Mínimo: {margenes.min():.1f}% — Máximo: {margenes.max():.1f}%")

# Salida:
# Media armónica del margen: 48.35%
# Media aritmética: 70.89%
# Mínimo: 2.2% — Máximo: 133.3%
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3 — `stats.hmean`: media armónica de márgenes.*

1. Salida:
2. Media armónica del margen: 48.35%
3. Media aritmética: 70.89%
4. Mínimo: 2.2% — Máximo: 133.3%

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** La media armónica (48.35%) es mucho menor que la aritmética (70.89%) porque da más peso a márgenes pequeños. Útil cuando queremos un promedio conservador de márgenes: algunos productos tienen margen muy bajo (2.2%) y eso arrastra el promedio armónico.

---

## Ejemplo 4 — `stats.mode`: categoría más frecuente

```python
moda_categoria = stats.mode(ventas["categoria"])
print(f"Categoría modal: {moda_categoria.mode[0]}")
print(f"Frecuencia: {moda_categoria.count[0]} de {len(ventas)} ventas ({100*moda_categoria.count[0]/len(ventas):.1f}%)")

# Salida:
# Categoría modal: Electrónica
# Frecuencia: 206 de 1330 ventas (15.5%)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4 — `stats.mode`: categoría más frecuente.*

1. Salida:
2. Categoría modal: Electrónica
3. Frecuencia: 206 de 1330 ventas (15.5%)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** Electrónica lidera en volumen de ventas. Esto sugiere priorizar inventario y promociones en esta categoría.

---

## Ejemplo 5 — `stats.skew`: asimetría de ingresos

```python
skew_ingreso = stats.skew(ventas["ingreso"], bias=False)
print(f"Asimetría de ingresos: {skew_ingreso:.3f}")
if skew_ingreso > 0:
    print("→ Cola derecha (valores extremos altos). La mayoría de ventas están por debajo de la media.")
elif skew_ingreso < 0:
    print("→ Cola izquierda (valores extremos bajos).")
else:
    print("→ Distribución simétrica.")

# Salida:
# Asimetría de ingresos: 3.109
# → Cola derecha (valores extremos altos). La mayoría de ventas están por debajo de la media.
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5 — `stats.skew`: asimetría de ingresos.*

1. Salida:
2. Asimetría de ingresos: 3.109
3. → Cola derecha (valores extremos altos). La mayoría de ventas están por debajo de la media.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** Con skewness de 3.1 positivo, confirmamos que la media ($24,809) está inflada por pocas ventas muy grandes. La mediana ($10,800) es más representativa del ingreso típico.

---

## Ejemplo 6 — `stats.kurtosis`: curtosis de ingresos

```python
kurt_ingreso = stats.kurtosis(ventas["ingreso"], bias=False)
print(f"Curtosis de ingresos: {kurt_ingreso:.3f}")
if kurt_ingreso > 0:
    print("→ Leptocúrtica (colas pesadas, más outliers que una normal).")
elif kurt_ingreso < 0:
    print("→ Platicúrtica (colas ligeras, menos outliers).")
else:
    print("→ Mesocúrtica (similar a normal).")

# Salida:
# Curtosis de ingresos: 15.135
# → Leptocúrtica (colas pesadas, más outliers que una normal).
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6 — `stats.kurtosis`: curtosis de ingresos.*

1. Salida:
2. Curtosis de ingresos: 15.135
3. → Leptocúrtica (colas pesadas, más outliers que una normal).

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** Curtosis de 15.1 (> 3 en exceso) indica que hay muchos más valores extremos de los que esperaríamos en una distribución normal. Esto es clave para elegir métodos estadísticos robustos.

---

## Ejemplo 7 — `stats.trim_mean`: media truncada al 10%

Elimina el 10% más bajo y el 10% más alto para obtener una media robusta ante outliers.

```python
mean_original = ventas["ingreso"].mean()
mean_truncada = stats.trim_mean(ventas["ingreso"], 0.1)

print(f"Media original: ${mean_original:,.0f}")
print(f"Media truncada (10%): ${mean_truncada:,.0f}")
print(f"Diferencia: ${mean_original - mean_truncada:,.0f}")

# Salida:
# Media original: $24,809
# Media truncada (10%): $13,207
# Diferencia: $11,602
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7 — `stats.trim_mean`: media truncada al 10%.*

1. Salida:
2. Media original: $24,809
3. Media truncada (10%): $13,207
4. Diferencia: $11,602

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** Al eliminar el 10% de valores extremos en cada cola, la media baja de $24,809 a $13,207 — una reducción masiva que confirma que los outliers inflan la media. La media truncada es una mejor estimación de la tendencia central.

---

## Ejemplo 8 — `stats.variation`: coeficiente de variación de precios

El coeficiente de variación (CV = std/mean) mide dispersión relativa, permitiendo comparar variabilidad entre variables con diferentes escalas.

```python
cv_precios = stats.variation(inventario["precio"])
print(f"CV de precios en inventario: {cv_precios:.3f}")
print(f"CV de ingresos en ventas: {stats.variation(ventas['ingreso']):.3f}")
print(f"Interpretación: CV > 1 indica dispersión muy alta.")

# Salida:
# CV de precios en inventario: 1.176
# CV de ingresos en ventas: 1.521
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8 — `stats.variation`: coeficiente de variación de precios.*

1. Salida:
2. CV de precios en inventario: 1.176
3. CV de ingresos en ventas: 1.521

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** Ambos CV > 1 indican dispersión extremadamente alta. Los ingresos en ventas (CV=1.52) son aún más variables que los precios de inventario (CV=1.18). Esto sugiere segmentar el análisis por categoría o sucursal.

---

## Ejemplo 9 — `stats.zscore`: detectar ventas anómalas (|z| > 3)

```python
z_scores = np.abs(stats.zscore(ventas["ingreso"]))
outliers = ventas[z_scores > 3]
print(f"Total ventas: {len(ventas)}")
print(f"Ventas con |z| > 3 (outliers extremos): {len(outliers)}")
print(f"Porcentaje: {100*len(outliers)/len(ventas):.2f}%")
print(f"\nIngreso mínimo entre outliers: ${outliers['ingreso'].min():,.0f}")
print(f"Ingreso máximo entre outliers: ${outliers['ingreso'].max():,.0f}")
print(outliers[["producto", "ingreso", "sucursal"]].head(10).to_string(index=False))

# Salida:
# Total ventas: 1330
# Ventas con |z| > 3 (outliers extremos): 22
# Porcentaje: 1.65%
# Ingreso mínimo entre outliers: $128,250
# Ingreso máximo entre outliers: $390,000
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

*Ejemplo 9 — `stats.zscore`: detectar ventas anómalas (|z| > 3).*

1. Salida:
2. Total ventas: 1330
3. Ventas con |z| > 3 (outliers extremos): 22
4. Porcentaje: 1.65%
5. Ingreso mínimo entre outliers: $128,250
6. Ingreso máximo entre outliers: $390,000

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** 22 ventas (1.65%) son outliers extremos con ingresos ≥ $128,250. Estas ventas merecen revisión: ¿son pedidos institucionales, errores de captura, o clientes clave?.

---

## Ejemplo 10 — `stats.pearsonr`: correlación precio-cantidad

```python
r_pearson, p_pearson = stats.pearsonr(ventas["precio_unitario"], ventas["cantidad"])
print(f"Correlación de Pearson (precio vs cantidad): r = {r_pearson:.4f}")
print(f"p-valor: {p_pearson:.6f}")
if r_pearson < 0:
    print("→ Relación negativa: a mayor precio, menor cantidad (elasticidad precio).")
elif r_pearson > 0:
    print("→ Relación positiva.")

# Salida:
# Correlación de Pearson (precio vs cantidad): r = -0.2824
# p-valor: 0.000000
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10 — `stats.pearsonr`: correlación precio-cantidad.*

1. Salida:
2. Correlación de Pearson (precio vs cantidad): r = -0.2824
3. p-valor: 0.000000

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** r = -0.28, p-valor ≈ 0. Existe una correlación negativa significativa pero débil entre precio y cantidad. Por cada aumento en precio, la cantidad vendida tiende a disminuir — consistente con la elasticidad precio de la demanda.

---

## Ejemplo 11 — `stats.spearmanr`: comparar con Pearson

Spearman usa rangos, no valores brutos, y detecta relaciones monótonas no lineales.

```python
r_spearman, p_spearman = stats.spearmanr(ventas["precio_unitario"], ventas["cantidad"])
print(f"Pearson:  r = {r_pearson:.4f}")
print(f"Spearman: r = {r_spearman:.4f}")
print(f"Diferencia: {abs(r_pearson - r_spearman):.4f}")
if abs(r_spearman) > abs(r_pearson):
    print("→ Spearman es más fuerte: sugiere relación monótona no lineal.")
else:
    print("→ Pearson es más fuerte: la relación es aproximadamente lineal.")

# Salida:
# Pearson:  r = -0.2824
# Spearman: r = -0.2957
# Diferencia: 0.0133
# → Spearman es más fuerte: sugiere relación monótona no lineal.
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11 — `stats.spearmanr`: comparar con Pearson.*

1. Salida:
2. Pearson:  r = -0.2824
3. Spearman: r = -0.2957
4. Diferencia: 0.0133
5. → Spearman es más fuerte: sugiere relación monótona no lineal.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** Spearman (r = -0.296) es ligeramente más fuerte que Pearson (-0.282). La diferencia es pequeña, lo que sugiere que la relación precio-cantidad es aproximadamente lineal, pero con algún componente no lineal.

---

## Ejemplo 12 — `stats.kendalltau`: correlación ordinal

Kendall tau es más robusto que Spearman para muestras pequeñas y con muchos empates.

```python
tau, p_kendall = stats.kendalltau(ventas["precio_unitario"], ventas["cantidad"])
print(f"Kendall's tau: {tau:.4f}")
print(f"p-valor: {p_kendall:.6f}")
print(f"Comparación: Pearson={r_pearson:.4f}, Spearman={r_spearman:.4f}, Kendall={tau:.4f}")

# Salida:
# Kendall's tau: -0.2157
# p-valor: 0.000000
# Comparación: Pearson=-0.2824, Spearman=-0.2957, Kendall=-0.2157
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12 — `stats.kendalltau`: correlación ordinal.*

1. Salida:
2. Kendall's tau: -0.2157
3. p-valor: 0.000000
4. Comparación: Pearson=-0.2824, Spearman=-0.2957, Kendall=-0.2157

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** tau = -0.216 confirma la asociación negativa. Kendall es el más conservador de los tres coeficientes. Los tres son significativos (p ≈ 0), dando robustez a la conclusión.

---

## Ejemplo 13 — `stats.linregress`: regresión lineal completa

Modelo: cantidad = pendiente × precio_unitario + intercepto

```python
slope, intercept, r_value, p_value, stderr = stats.linregress(
    ventas["precio_unitario"], ventas["cantidad"]
)

print(f"Pendiente (slope): {slope:.6f}")
print(f"Intercepto: {intercept:.4f}")
print(f"R² (coef. determinación): {r_value**2:.4f}")
print(f"p-valor (pendiente ≠ 0): {p_value:.6f}")
print(f"Error estándar (pendiente): {stderr:.6f}")
print(f"\nEcuación: cantidad = {slope:.6f} × precio + {intercept:.2f}")
print(f"Por cada $1000 adicionales, la cantidad cae {-slope*1000:.2f} unidades.")

# Salida:
# Pendiente (slope): -0.000398
# Intercepto: 9.4837
# R² (coef. determinación): 0.0798
# p-valor (pendiente ≠ 0): 0.000000
# Error estándar (pendiente): 0.000036
# Ecuación: cantidad = -0.000398 × precio + 9.48
# Por cada $1000 adicionales, la cantidad cae 0.40 unidades.
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13 — `stats.linregress`: regresión lineal completa.*

1. Salida:
2. Pendiente (slope): -0.000398
3. Intercepto: 9.4837
4. R² (coef. determinación): 0.0798
5. p-valor (pendiente ≠ 0): 0.000000
6. Error estándar (pendiente): 0.000036
7. Ecuación: cantidad = -0.000398 × precio + 9.48
8. Por cada $1000 adicionales, la cantidad cae 0.40 unidades.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** El modelo de regresión es significativo (p ≈ 0), pero R² = 0.08 solo explica el 8% de la variabilidad. El precio por sí solo no predice bien la cantidad; debemos incluir más variables (categoría, sucursal, descuento).

---

## Ejemplo 14 — `stats.iqr`: rango intercuartil de ingresos

```python
Q1 = ventas["ingreso"].quantile(0.25)
Q3 = ventas["ingreso"].quantile(0.75)
iqr = stats.iqr(ventas["ingreso"])
limite_inferior = Q1 - 1.5 * iqr
limite_superior = Q3 + 1.5 * iqr
outliers_iqr = ventas[(ventas["ingreso"] < limite_inferior) | (ventas["ingreso"] > limite_superior)]

print(f"Q1: ${Q1:,.0f} | Q3: ${Q3:,.0f}")
print(f"IQR: ${iqr:,.0f}")
print(f"Bigote inferior (Q1 - 1.5*IQR): ${limite_inferior:,.0f}")
print(f"Bigote superior (Q3 + 1.5*IQR): ${limite_superior:,.0f}")
print(f"Outliers según Tukey: {len(outliers_iqr)} ({100*len(outliers_iqr)/len(ventas):.1f}%)")

# Salida:
# Q1: $5,268 | Q3: $26,094
# IQR: $20,826
# Bigote inferior (Q1 - 1.5*IQR): $-25,971
# Bigote superior (Q3 + 1.5*IQR): $57,333
# Outliers según Tukey: 166 (12.5%)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14 — `stats.iqr`: rango intercuartil de ingresos.*

1. Salida:
2. Q1: $5,268 | Q3: $26,094
3. IQR: $20,826
4. Bigote inferior (Q1 - 1.5*IQR): $-25,971
5. Bigote superior (Q3 + 1.5*IQR): $57,333
6. Outliers según Tukey: 166 (12.5%)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** 166 ventas (12.5%) están fuera del rango de Tukey. El bigote superior en $57,333 marca el umbral; cualquier venta por encima se considera atípica. Es un método más permisivo que z-score para identificar valores extremos.

---

## Ejemplo 15 — `stats.sem`: error estándar de la media

El SEM mide la precisión de la media muestral como estimador de la media poblacional.

```python
sem_ingreso = stats.sem(ventas["ingreso"])
ic_95_inf = ventas["ingreso"].mean() - 1.96 * sem_ingreso
ic_95_sup = ventas["ingreso"].mean() + 1.96 * sem_ingreso

print(f"Error estándar de la media (SEM): ${sem_ingreso:,.0f}")
print(f"Media muestral: ${ventas['ingreso'].mean():,.0f}")
print(f"IC 95%: [${ic_95_inf:,.0f}, ${ic_95_sup:,.0f}]")

# Salida:
# Error estándar de la media (SEM): $1,035
# Media muestral: $24,809
# IC 95%: [$22,780, $26,838]
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15 — `stats.sem`: error estándar de la media.*

1. Salida:
2. Error estándar de la media (SEM): $1,035
3. Media muestral: $24,809
4. IC 95%: [$22,780, $26,838]

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** Con SEM = $1,035, podemos decir que la media poblacional de ingresos está entre $22,780 y $26,838 con 95% de confianza. El SEM es pequeño relativo a la media (~4%), indicando una estimación razonablemente precisa.

---

## Ejemplo 16 — `stats.moment`: tercer momento (asimetría)

Los momentos estadísticos generalizan las propiedades de una distribución:
- 1er momento: media
- 2do momento: varianza
- 3er momento: asimetría
- 4to momento: curtosis

```python
momento_1 = stats.moment(ventas["ingreso"], 1)   # 0 por definición
momento_2 = stats.moment(ventas["ingreso"], 2)   # varianza
momento_3 = stats.moment(ventas["ingreso"], 3)   # asimetría (crudo)
momento_4 = stats.moment(ventas["ingreso"], 4)   # curtosis (crudo)

print(f"Momento 1 (media centrada): {momento_1:.0f}")
print(f"Momento 2 (varianza): {momento_2:,.0f}")
print(f"Momento 3 (asimetría cruda): {momento_3:,.0f}")
print(f"Momento 4 (curtosis cruda): {momento_4:,.0f}")
print(f"Skewness (de stats.skew): {stats.skew(ventas['ingreso']):.3f}")
print(f"Curtosis (de stats.kurtosis): {stats.kurtosis(ventas['ingreso']):.3f}")

# Salida:
# Momento 1 (media centrada): 0
# Momento 2 (varianza): 1,422,763,878
# Momento 3 (asimetría cruda): 27,419,633,213,875
# Momento 4 (curtosis cruda): 3,547,430,698,357,816,832
# Skewness (de stats.skew): 3.109
# Curtosis (de stats.kurtosis): 15.135
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16 — `stats.moment`: tercer momento (asimetría).*

1. Salida:
2. Momento 1 (media centrada): 0
3. Momento 2 (varianza): 1,422,763,878
4. Momento 3 (asimetría cruda): 27,419,633,213,875
5. Momento 4 (curtosis cruda): 3,547,430,698,357,816,832
6. Skewness (de stats.skew): 3.109
7. Curtosis (de stats.kurtosis): 15.135

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** Los momentos crudos son difíciles de interpretar por las escalas. Por eso usamos skewness y curtosis normalizados (libres de escala). El momento 3 positivo confirma la asimetría derecha; el momento 4 grande confirma colas pesadas.

---

## Ejemplo 17 — Comparación: gmean vs hmean vs mean en precios

```python
precios_venta = ventas["precio_unitario"]

media_art = np.mean(precios_venta)
media_geo = stats.gmean(precios_venta)
media_har = stats.hmean(precios_venta)
mediana = np.median(precios_venta)

print(f"{'Métrica':<25} {'Precios venta':>12} {'Inventario':>12}")
print("-" * 50)
print(f"{'Media aritmética':<25} {media_art:>12,.0f} {inventario['precio'].mean():>12,.0f}")
print(f"{'Media geométrica':<25} {media_geo:>12,.0f} {stats.gmean(inventario['precio']):>12,.0f}")
print(f"{'Media armónica':<25} {media_har:>12,.0f} {stats.hmean(inventario['precio']):>12,.0f}")
print(f"{'Mediana':<25} {mediana:>12,.0f} {inventario['precio'].median():>12,.0f}")

# Salida:
# Métrica                       Precios venta    Inventario
# --------------------------------------------------
# Media aritmética                    2,984          3,178
# Media geométrica                    1,884          1,903
# Media armónica                      1,149          1,274
# Mediana                             1,700          1,800
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17 — Comparación: gmean vs hmean vs mean en precios.*

1. Salida:
2. Métrica                       Precios venta    Inventario
3. --------------------------------------------------
4. Media aritmética                    2,984          3,178
5. Media geométrica                    1,884          1,903
6. Media armónica                      1,149          1,274
7. Mediana                             1,700          1,800

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** Siempre se cumple: media armónica ≤ media geométrica ≤ media aritmética (desigualdad de las medias). La media armónica es la más conservadora. Para distribuciones con valores extremos (como precios), la mediana es la más representativa, seguida de la geométrica.

---

## Ejemplo 18 — Z-score con umbral |z| > 2 (outliers moderados)

```python
z_ingreso = stats.zscore(ventas["ingreso"])
outliers_moderados = ventas[np.abs(z_ingreso) > 2]
outliers_extremos = ventas[np.abs(z_ingreso) > 3]

print(f"{'Umbral':<20} {'Outliers':>10} {'% del total':>12}")
print("-" * 42)
print(f"{'|z| > 2 (moderados)':<20} {len(outliers_moderados):>10} {100*len(outliers_moderados)/len(ventas):>11.2f}%")
print(f"{'|z| > 3 (extremos)':<20} {len(outliers_extremos):>10} {100*len(outliers_extremos)/len(ventas):>11.2f}%")

# Resumen de outliers moderados
print(f"\nIngreso mínimo (|z|>2): ${outliers_moderados['ingreso'].min():,.0f}")
print(f"Ingreso máximo (|z|>2): ${outliers_moderados['ingreso'].max():,.0f}")
print(f"\nDistribución por sucursal (top 5):")
print(outliers_moderados['sucursal'].value_counts().head(5).to_string())

# Salida:
# Umbral               Outliers   % del total
# ------------------------------------------
# |z| > 2 (moderados)         46         3.46%
# |z| > 3 (extremos)          22         1.65%
# Ingreso mínimo (|z|>2): $78,975
# Ingreso máximo (|z|>2): $390,000
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

*Ejemplo 18 — Z-score con umbral |z| > 2 (outliers moderados).*

1. Resumen de outliers moderados
2. Salida:
3. Umbral               Outliers   % del total
4. ------------------------------------------
5. |z| > 2 (moderados)         46         3.46%
6. |z| > 3 (extremos)          22         1.65%
7. Ingreso mínimo (|z|>2): $78,975
8. Ingreso máximo (|z|>2): $390,000

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** Con |z|>2 detectamos 46 outliers (3.46% de los datos). De estos, 22 son extremos (|z|>3). Los outliers moderados (>$78,975) son candidatos a revisión de crédito, verificación de datos, o segmentación como ventas corporativas.

---

## Estadística circular — `circmean`, `circvar`

Para datos angulares o estacionales (día de la semana como ángulo).

```python
# Convertir día_semana (0-6) a ángulos (0-2π)
angulos = ventas["dia_semana"] * 2 * np.pi / 7
circ_media = stats.circmean(angulos, high=2*np.pi, low=0)
circ_var = stats.circvar(angulos, high=2*np.pi, low=0)
circ_mean_dia = circ_media * 7 / (2 * np.pi)

print(f"Media circular (radianes): {circ_media:.3f}")
print(f"Día promedio de venta: {circ_mean_dia:.2f} (0=Domingo, 6=Sábado)")
print(f"Varianza circular: {circ_var:.3f} (0=todos mismo día, 1=uniforme)")

# Salida:
# Media circular (radianes): 2.972
# Día promedio de venta: 3.31 (0=Domingo, 6=Sábado)
# Varianza circular: 0.929 (0=todos mismo día, 1=uniforme)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Estadística circular — `circmean`, `circvar`.*

1. Convertir día_semana (0-6) a ángulos (0-2π)
2. Salida:
3. Media circular (radianes): 2.972
4. Día promedio de venta: 3.31 (0=Domingo, 6=Sábado)
5. Varianza circular: 0.929 (0=todos mismo día, 1=uniforme)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** El día promedio de venta es ~3.3 (miércoles-jueves). La varianza circular alta (~0.93) indica que las ventas están dispersas a lo largo de la semana, sin un día dominante.

---

## `stats.pointbiserialr`: correlación punto-biserial

Mide la correlación entre una variable binaria y una continua.

```python
# Crear variable binaria: ¿margen > 50%?
margen_alto = (ventas["margen_pct"] > 50).astype(int)
r_pb, p_pb = stats.pointbiserialr(margen_alto, ventas["ingreso"])

print(f"Correlación punto-biserial (margen>50% vs ingreso): r = {r_pb:.4f}")
print(f"p-valor: {p_pb:.6f}")

# Ingreso promedio por grupo
print(f"Ingreso promedio cuando margen ≤ 50%:  ${ventas[ventas['margen_pct']<=50]['ingreso'].mean():,.0f}")
print(f"Ingreso promedio cuando margen > 50%:  ${ventas[ventas['margen_pct']>50]['ingreso'].mean():,.0f}")

# Salida:
# Correlación punto-biserial (margen>50% vs ingreso): r = -0.1351
# p-valor: 0.000001
# Ingreso promedio cuando margen ≤ 50%:  $30,468
# Ingreso promedio cuando margen > 50%:  $21,136
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*`stats.pointbiserialr`: correlación punto-biserial.*

1. Crear variable binaria: ¿margen > 50%?
2. Ingreso promedio por grupo
3. Salida:
4. Correlación punto-biserial (margen>50% vs ingreso): r = -0.1351
5. p-valor: 0.000001
6. Ingreso promedio cuando margen ≤ 50%:  $30,468
7. Ingreso promedio cuando margen > 50%:  $21,136

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** r = -0.135, p-valor < 0.05. Existe una correlación negativa significativa pero débil: los productos con margen > 50% tienden a tener menor ingreso. Tiene sentido: productos de alto margen (accesorios) son más baratos.

---

## Resumen

| Función | Uso principal | Aplicación en ventas |
|---------|---------------|---------------------|
| `describe` | Resumen completo | Reporte ejecutivo de ingresos |
| `gmean` | Tasas de crecimiento | Crecimiento promedio de precios |
| `hmean` | Promedio de tasas | Margen promedio conservador |
| `mode` | Valor más frecuente | Categoría/sucursal con más ventas |
| `skew` | Dirección de asimetría | ¿Ingresos sesgados a la derecha? |
| `kurtosis` | Peso de colas | ¿Hay más outliers que lo normal? |
| `trim_mean` | Media robusta | Tendencia central sin extremos |
| `variation` | Dispersión relativa | Comparar variabilidad entre variables |
| `zscore` | Detección de outliers | Identificar ventas anómalas |
| `pearsonr` | Correlación lineal | Relación precio-cantidad |
| `spearmanr` | Correlación monótona | Alternativa robusta a Pearson |
| `kendalltau` | Correlación ordinal | Muestras pequeñas o con empates |
| `linregress` | Regresión lineal | Modelo predictivo simple |
| `iqr` | Dispersión robusta | Boxplots, detección Tukey |
| `sem` | Precisión de la media | Intervalos de confianza |
| `moment` | Momentos genéricos | Análisis avanzado de distribución |
| `circmean/var` | Datos circulares | Estacionalidad semanal |
| `pointbiserialr` | Binaria vs continua | Margen alto vs ingreso |

---

## Ejercicios

1. Calcula `stats.describe` para `cantidad`. ¿Cuál es la asimetría? ¿Hay más transacciones con pocos o muchos productos?

2. Usa `stats.gmean` con los ingresos de la sucursal "Matriz CDMX" y compáralo con la media aritmética. ¿Cuál es más representativa?

3. Aplica `stats.trim_mean` a `precio_unitario` con proporciones 0.05 y 0.15. ¿Cómo cambia la media truncada?

4. Calcula la correlación de Pearson y Spearman entre `costo_unitario` y `precio_unitario`. ¿Cuál es más fuerte? Interpreta.

5. Usa `stats.zscore` para encontrar outliers en `margen_pct`. ¿Cuántos tienen |z| > 2? ¿Qué productos son?

6. Aplica `stats.linregress` con `precio_unitario` como X y `margen_pct` como Y. Interpreta la pendiente y R².

7. Calcula `stats.iqr` para `ingreso` agrupado por `categoria`. ¿Qué categoría tiene la mayor dispersión?

8. Usa `stats.sem` para calcular el IC 95% del ingreso promedio en cada sucursal. ¿Qué sucursal tiene el IC más amplio y por qué?
