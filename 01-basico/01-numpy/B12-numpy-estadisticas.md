# B12 – Estadísticas con NumPy

## Introducción

Las funciones estadísticas de NumPy permiten analizar distribuciones de precios, ventas, costos e inventarios. Calcular promedios, desviaciones, percentiles y correlaciones es fundamental para la toma de decisiones comerciales: ¿cuál es la venta promedio? ¿qué tan variables son los precios? ¿qué producto tiene mayor demanda? Este capítulo cubre las principales funciones estadísticas aplicadas al contexto de ventas y compras.

---

## 1. `np.mean` — Media (venta promedio por producto)

```python
import numpy as np

ventas_diarias = np.array([1200, 890, 1500, 2100, 980, 3100, 1800])
promedio_ventas = np.mean(ventas_diarias)
print("Venta promedio diaria:", promedio_ventas)
```

**Salida esperada:**
```
Venta promedio diaria: 1667.14
```

**Explicación:**
- `np.mean(ventas_diarias)`: calcula la media aritmética.
- Suma todos los valores y divide entre el número total de elementos.

---

## 2. `np.std` — Desviación estándar (variabilidad de precios)

```python
import numpy as np

precios = np.array([150.0, 230.0, 85.5, 420.0, 67.0, 310.0])
desviacion = np.std(precios)
print("Desviación estándar de precios:", round(desviacion, 2))
```

**Salida esperada:**
```
Desviación estándar de precios: 124.83
```

**Explicación:**
- `np.std(precios)`: mide la dispersión de los precios respecto a la media.
- Un valor alto indica precios muy diferentes entre sí.

---

## 3. `np.var` — Varianza (dispersión de inventarios)

```python
import numpy as np

stocks = np.array([45, 12, 8, 90, 3, 67, 0, 22])
varianza_stock = np.var(stocks)
print("Varianza del stock:", round(varianza_stock, 2))
```

**Salida esperada:**
```
Varianza del stock: 974.75
```

**Explicación:**
- `np.var(stocks)`: varianza poblacional (divide por n).
- Mide qué tan dispersos están los niveles de inventario.

---

## 4. `np.min` y `np.max` — Mínimo y máximo

```python
import numpy as np

precios = np.array([150.0, 230.0, 85.5, 420.0, 67.0, 310.0])
print("Precio mínimo:", np.min(precios))
print("Precio máximo:", np.max(precios))
print("Rango:", np.max(precios) - np.min(precios))
```

**Salida esperada:**
```
Precio mínimo: 67.0
Precio máximo: 420.0
Rango: 353.0
```

**Explicación:**
- `np.min` y `np.max`: valor mínimo y máximo del array.
- El rango (max - min) da una idea rápida de la dispersión.

---

## 5. `np.argmin` y `np.argmax` — Índice del mínimo/máximo

```python
import numpy as np

productos = np.array(["Leche", "Pan", "Huevos", "Arroz", "Frijoles"])
precios = np.array([150.0, 230.0, 85.5, 420.0, 67.0])
idx_caro = np.argmax(precios)
idx_barato = np.argmin(precios)
print("Producto más caro:", productos[idx_caro], "-", precios[idx_caro])
print("Producto más barato:", productos[idx_barato], "-", precios[idx_barato])
```

**Salida esperada:**
```
Producto más caro: Arroz - 420.0
Producto más barato: Frijoles - 67.0
```

**Explicación:**
- `np.argmax`: índice del valor máximo.
- `np.argmin`: índice del valor mínimo.
- Útil para identificar qué producto es el más caro/barato.

---

## 6. `np.sum` — Suma total (ingresos totales)

```python
import numpy as np

ingresos = np.array([1500, 1150, 1710, 1260, 1005])
total_ingresos = np.sum(ingresos)
print("Ingresos totales:", total_ingresos)
```

**Salida esperada:**
```
Ingresos totales: 6625
```

**Explicación:**
- `np.sum(ingresos)`: suma todos los elementos.
- A diferencia de `np.mean`, devuelve el total acumulado.

---

## 7. `np.cumsum` — Suma acumulada (ingresos acumulados)

```python
import numpy as np

ingresos = np.array([1500, 1150, 1710, 1260, 1005])
ingresos_acum = np.cumsum(ingresos)
print("Ingresos acumulados:", ingresos_acum)
```

**Salida esperada:**
```
Ingresos acumulados: [1500 2650 4360 5620 6625]
```

**Explicación:**
- `np.cumsum`: suma acumulativa elemento por elemento.
- Útil para reportes de ventas progresivas en el tiempo.

---

## 8. `np.percentile` — Percentiles (análisis de distribución de precios)

```python
import numpy as np

precios = np.array([150.0, 230.0, 85.5, 420.0, 67.0, 310.0, 180.0, 95.0])
p25 = np.percentile(precios, 25)
p50 = np.percentile(precios, 50)  # mediana
p75 = np.percentile(precios, 75)
print("Percentil 25:", p25)
print("Mediana (P50):", p50)
print("Percentil 75:", p75)
```

**Salida esperada:**
```
Percentil 25: 90.0
Mediana (P50): 165.0
Percentil 75: 260.0
```

**Explicación:**
- `np.percentile(array, q)`: calcula el percentil `q` del array.
- P25: 25% de los precios están por debajo.
- P50: mediana (50% por debajo, 50% por encima).
- P75: 75% de los precios están por debajo.

---

## 9. `np.median` — Mediana (precio típico)

```python
import numpy as np

precios = np.array([150.0, 230.0, 85.5, 420.0, 67.0, 310.0])
mediana = np.median(precios)
media = np.mean(precios)
print("Mediana:", mediana)
print("Media:", round(media, 2))
```

**Salida esperada:**
```
Mediana: 190.0
Media: 210.42
```

**Explicación:**
- `np.median`: valor central cuando los datos están ordenados.
- Menos sensible a valores extremos que la media.

---

## 10. `np.average` con pesos — Precio promedio ponderado por cantidad

```python
import numpy as np

precios = np.array([150.0, 230.0, 85.5, 420.0, 67.0])
cantidades = np.array([10, 5, 20, 3, 15])
precio_prom_ponderado = np.average(precios, weights=cantidades)
print("Precio promedio ponderado:", round(precio_prom_ponderado, 2))
```

**Salida esperada:**
```
Precio promedio ponderado: 124.53
```

**Explicación:**
- `np.average(precios, weights=cantidades)`: media ponderada.
- Los productos con mayor cantidad vendida tienen más peso en el promedio.
- Útil porque el precio promedio simple no refleja la cantidad vendida.

---

## 11. `np.corrcoef` — Correlación entre precio y cantidad vendida

```python
import numpy as np

precios = np.array([150, 230, 85, 420, 67, 310, 180, 95])
cantidades = np.array([10, 5, 20, 3, 15, 7, 12, 18])
corr = np.corrcoef(precios, cantidades)
print("Matriz de correlación:\n", corr)
print("Correlación precio-cantidad:", round(corr[0, 1], 3))
```

**Salida esperada:**
```
Matriz de correlación:
 [[ 1.     -0.928]
 [-0.928   1.   ]]
Correlación precio-cantidad: -0.928
```

**Explicación:**
- `np.corrcoef(x, y)`: matriz 2×2 con correlaciones.
- `corr[0,1]` es la correlación entre precio y cantidad.
- Valor negativo (−0.928): a mayor precio, menor cantidad vendida (ley de demanda).

---

## 12. `np.ptp` — Rango (peak-to-peak)

```python
import numpy as np

precios = np.array([150.0, 230.0, 85.5, 420.0, 67.0, 310.0])
rango = np.ptp(precios)
print("Rango de precios:", rango)
```

**Salida esperada:**
```
Rango de precios: 353.0
```

**Explicación:**
- `np.ptp`: `max - min`, el rango total de los datos.
- Mide la amplitud total de los precios.

---

## 13. `np.histogram` — Histograma de frecuencias de precios

```python
import numpy as np

precios = np.array([150, 230, 85, 420, 67, 310, 180, 95, 250, 130])
hist, bins = np.histogram(precios, bins=5)
print("Frecuencias:", hist)
print("Bins:", bins)
```

**Salida esperada:**
```
Frecuencias: [3 2 2 1 2]
Bins: [ 67.  137.6 208.2 278.8 349.4 420. ]
```

**Explicación:**
- `np.histogram(precios, bins=5)`: divide el rango en 5 intervalos y cuenta cuántos precios caen en cada uno.
- `hist`: frecuencias (conteos).
- `bins`: límites de los intervalos.
- Útil para entender la distribución de precios.

---

## 14. `np.bincount` — Conteo de frecuencias de enteros (ventas por producto)

```python
import numpy as np

# IDs de productos vendidos (0-4)
ventas_productos = np.array([0, 2, 1, 3, 2, 0, 4, 1, 2, 3, 3, 0])
conteo = np.bincount(ventas_productos)
print("Conteo de ventas por producto:", conteo)
```

**Salida esperada:**
```
Conteo de ventas por producto: [3 2 3 3 1]
```

**Explicación:**
- `np.bincount`: cuenta cuántas veces aparece cada entero no negativo.
- El índice del resultado es el ID del producto; el valor es la frecuencia.
- Producto 0 se vendió 3 veces, producto 1→2, etc.

---

## 15. `np.nanmean` — Media ignorando NaN (datos faltantes)

```python
import numpy as np

ventas_con_faltantes = np.array([1500, np.nan, 1710, 1260, np.nan, 1005])
promedio_sin_nan = np.nanmean(ventas_con_faltantes)
print("Promedio (ignorando NaN):", round(promedio_sin_nan, 2))
```

**Salida esperada:**
```
Promedio (ignorando NaN): 1368.75
```

**Explicación:**
- `np.nanmean(array)`: calcula la media ignorando valores `NaN`.
- Cuando hay datos faltantes en ventas, `np.mean` devolvería `NaN`.

---

## 16. `np.nansum` — Suma ignorando NaN

```python
import numpy as np

ventas_con_faltantes = np.array([1500, np.nan, 1710, 1260, np.nan, 1005])
total_sin_nan = np.nansum(ventas_con_faltantes)
print("Total (ignorando NaN):", total_sin_nan)
```

**Salida esperada:**
```
Total (ignorando NaN): 5475.0
```

**Explicación:**
- `np.nansum`: suma ignorando valores `NaN`.
- Sin esta función, el resultado sería `NaN`.

---

## 17. `np.average` vs `np.mean` — Diferencia práctica

```python
import numpy as np

precios = np.array([150, 230, 85, 420, 67])
cantidades = np.array([10, 5, 20, 3, 15])
media_simple = np.mean(precios)
media_pond = np.average(precios, weights=cantidades)
print("Media simple:", media_simple)
print("Media ponderada:", round(media_pond, 2))
print("Diferencia:", round(media_simple - media_pond, 2))
```

**Salida esperada:**
```
Media simple: 190.4
Media ponderada: 124.53
Diferencia: 65.87
```

**Explicación:**
- La media simple da el mismo peso a cada producto.
- La media ponderada da más peso a los productos con mayor cantidad vendida.
- La gran diferencia indica que los productos baratos se venden más.

---

## 18. Estadísticas por eje (axis) — Análisis por sucursal

```python
import numpy as np

# 3 sucursales × 7 días
ventas_semana = np.array([
    [1200, 890, 1500, 2100, 980, 3100, 1800],
    [950, 1100, 1300, 1700, 880, 2500, 1600],
    [1500, 1400, 1200, 1900, 1050, 2800, 2000]
])
promedio_por_sucursal = np.mean(ventas_semana, axis=1)
promedio_por_dia = np.mean(ventas_semana, axis=0)
print("Promedio por sucursal:", promedio_por_sucursal)
print("Promedio por día:", promedio_por_dia)
```

**Salida esperada:**
```
Promedio por sucursal: [1652.86 1432.86 1692.86]
Promedio por día: [1216.67 1130.  1333.33 1900.   970.  2800.  1800.  ]
```

**Explicación:**
- `axis=1`: promedio a lo largo de las columnas (por sucursal).
- `axis=0`: promedio a lo largo de las filas (por día).
- La sucursal 2 tiene el promedio más alto; el día 5 (sábado) es el de mayores ventas.

---

## Resumen

| Función | Propósito | Aplicación |
|---------|-----------|------------|
| `np.mean` | Media aritmética | Venta promedio |
| `np.std` | Desviación estándar | Variabilidad de precios |
| `np.var` | Varianza | Dispersión de inventarios |
| `np.min` / `np.max` | Mínimo y máximo | Producto más barato/caro |
| `np.argmin` / `np.argmax` | Índice del min/max | Identificar producto extremo |
| `np.sum` | Suma total | Ingresos totales |
| `np.cumsum` | Suma acumulada | Ventas acumuladas en el mes |
| `np.percentile` | Percentiles | Distribución de precios |
| `np.median` | Mediana | Precio típico |
| `np.average` (weights) | Media ponderada | Precio promedio ponderado |
| `np.corrcoef` | Correlación | Relación precio-demanda |
| `np.ptp` | Rango total | Amplitud de precios |
| `np.histogram` | Histograma | Distribución de precios |
| `np.bincount` | Conteo de enteros | Frecuencia de ventas |
| `np.nanmean` / `np.nansum` | Ignorar NaN | Datos faltantes |
| `axis=` | Estadísticas por eje | Por sucursal, por día |

---

## Ejercicios

1. Dado el array `precios = [150, 230, 85, 420, 67, 310]`, calcula la media, mediana, desviación estándar y rango.

2. Encuentra el producto más caro y el más barato usando `argmax` / `argmin` con los arrays `productos = ["A","B","C","D","E","F"]` y `precios = [150, 230, 85, 420, 67, 310]`.

3. Calcula el precio promedio ponderado de `precios = [150, 230, 85, 420, 67]` con `cantidades = [10, 5, 20, 3, 15]`.

4. Dados `precios = [150, 230, 85, 420, 67, 310, 180, 95]` y `cantidades = [10, 5, 20, 3, 15, 7, 12, 18]`, calcula la correlación entre precio y cantidad.

5. Calcula los percentiles 10, 25, 50, 75 y 90 de `precios = [150, 230, 85, 420, 67, 310, 180, 95, 250, 130]`.

6. Usa `np.histogram` con 4 bins para el array de precios del ejercicio 5 y muestra las frecuencias.

7. Dada la matriz 3×7 de ventas semanales, calcula el promedio, mínimo y máximo por sucursal (axis=1) y por día (axis=0).

8. Crea un array con valores `NaN` en algunas posiciones (ventas faltantes) y demuestra el uso de `np.nanmean` y `np.nansum` para obtener estadísticas ignorando los datos faltantes.
