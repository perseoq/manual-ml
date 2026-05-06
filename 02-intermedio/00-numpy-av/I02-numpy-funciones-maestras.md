# I02 — Funciones Maestras de NumPy (Estadísticas, Análisis, Señales)

## 1. Introducción Teórica

NumPy proporciona un conjunto de funciones especializadas para análisis estadístico robusto, procesamiento de señales y manipulación de datos. Este módulo cubre las funciones más útiles para el análisis de ventas, compras e inventarios.

### Funciones NaN-safe (tolerantes a valores faltantes)

Cuando los datos reales contienen valores faltantes (NaN), las funciones estándar devuelven NaN. Las variantes `np.nan*` ignoran los NaN automáticamente:

- **np.nanmean(arr):** Promedio ignorando NaN
- **np.nansum(arr):** Suma ignorando NaN
- **np.nanstd(arr):** Desviación estándar ignorando NaN
- **np.nanmin(arr), np.nanmax(arr):** Mínimo/Máximo ignorando NaN
- **np.nanpercentile(arr, q):** Percentil ignorando NaN

### Funciones de análisis y señal

- **np.trapz(y, x):** Integración numérica por regla trapezoidal (área bajo curva)
- **np.gradient(f, *spacing):** Gradiente o diferencias centradas (1ra derivada)
- **np.diff(a, n=1):** Diferencias discretas (n-ésima)
- **np.ediff1d(ary):** Diferencias entre elementos consecutivos (1D)
- **np.cross(a, b):** Producto cruzado (útil para vectores de precio-cantidad)
- **np.cumsum(a):** Suma acumulada

### Funciones de conteo y distribución

- **np.unique(ar, return_counts=True):** Valores únicos y sus frecuencias
- **np.bincount(x):** Cuenta enteros no negativos (histograma rápido)
- **np.histogram(a, bins=10):** Histograma con bins personalizados
- **np.digitize(x, bins):** Asigna valores a bins
- **np.interp(x, xp, fp):** Interpolación lineal 1D

---

## 2. Ejemplos Prácticos

### Ejemplo 1: np.nanmean — promedio de ventas con datos faltantes

```python
import numpy as np

# Ventas diarias de una semana con datos faltantes (NaN)
ventas = np.array([1200.0, np.nan, 980.0, 1500.0, np.nan, 2100.0, 1750.0])

promedio_simple = np.mean(ventas) if not np.any(np.isnan(ventas)) else np.nan
promedio_robusto = np.nanmean(ventas)

print(f"Ventas: {ventas}")
print(f"Promedio con mean(): {promedio_simple}")
print(f"Promedio con nanmean(): {promedio_robusto:.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: np.nanmean — promedio de ventas con datos faltantes.*

1. Ventas diarias de una semana con datos faltantes (NaN)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: np.nansum — suma total ignorando NaN

```python
# Compras mensuales con meses sin datos
compras = np.array([5000, 7200, np.nan, 4800, 6100, np.nan, np.nan, 8300])
total_compras = np.nansum(compras)
dias_con_datos = np.sum(~np.isnan(compras))
print(f"Total compras (ignorando NaN): ${total_compras:,.0f}")
print(f"Meses con datos: {dias_con_datos} de {len(compras)}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: np.nansum — suma total ignorando NaN.*

1. Compras mensuales con meses sin datos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: np.nanpercentile — percentiles robustos

```python
# Precios de 50 productos con algunos missing
np.random.seed(42)
precios = np.random.uniform(50, 500, 50)
precios[np.random.choice(50, 5, replace=False)] = np.nan

percentiles = np.nanpercentile(precios, [10, 25, 50, 75, 90])
print(f"P10: ${percentiles[0]:.0f}")
print(f"P25: ${percentiles[1]:.0f}")
print(f"P50 (mediana): ${percentiles[2]:.0f}")
print(f"P75: ${percentiles[3]:.0f}")
print(f"P90: ${percentiles[4]:.0f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: np.nanpercentile — percentiles robustos.*

1. Precios de 50 productos con algunos missing

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: np.trapz — área bajo curva de demanda

```python
# Curva de demanda: precio vs cantidad vendida
precios = np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10])
cantidades = np.array([10, 15, 22, 35, 50, 75, 110, 160, 230, 320])

area_bajo_curva = np.trapz(cantidades, precios)
print(f"Área bajo curva demanda: {area_bajo_curva:.0f} (unidades·precio)")
# Representa el "valor total del mercado" a diferentes precios
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: np.trapz — área bajo curva de demanda.*

1. Curva de demanda: precio vs cantidad vendida
2. Representa el "valor total del mercado" a diferentes precios

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: np.gradient — tasa de cambio de ventas diarias

```python
# Ventas diarias durante 10 días
ventas = np.array([100, 120, 115, 140, 160, 155, 180, 200, 195, 220])

cambio_diario = np.gradient(ventas)
print("Día   Ventas  Cambio")
for i in range(len(ventas)):
    print(f"{i+1:2d}   {ventas[i]:5.0f}  {cambio_diario[i]:+7.1f}")

# np.gradient usa diferencias centradas (más preciso que diff)
print(f"\nCambio promedio: {np.mean(cambio_diario):.1f} unds/día")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: np.gradient — tasa de cambio de ventas diarias.*

1. Ventas diarias durante 10 días
2. np.gradient usa diferencias centradas (más preciso que diff)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: np.gradient — aceleración (segunda derivada)

```python
# Ventas acumuladas mostrando aceleración/desaceleración
dias = np.arange(1, 11)
ventas_acum = np.array([100, 220, 340, 480, 640, 800, 970, 1150, 1340, 1550])

velocidad = np.gradient(ventas_acum, dias)     # primera derivada
aceleracion = np.gradient(velocidad, dias)       # segunda derivada

print("Día  Ventas  Velocidad  Aceleración")
for i in range(len(dias)):
    print(f"{dias[i]:2d}  {ventas_acum[i]:5.0f}  {velocidad[i]:8.1f}  {aceleracion[i]:9.1f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: np.gradient — aceleración (segunda derivada).*

1. Ventas acumuladas mostrando aceleración/desaceleración

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: np.unique + return_counts — frecuencia de productos vendidos

```python
# IDs de productos vendidos en un día
productos = np.array([101, 203, 101, 305, 203, 101, 402, 203, 203, 305])

codigos, conteos = np.unique(productos, return_counts=True)
print("Producto  Frecuencia")
for cod, cnt in zip(codigos, conteos):
    print(f"   {cod}       {cnt}")

mas_vendido = codigos[np.argmax(conteos)]
print(f"\nProducto más vendido: {mas_vendido} ({max(conteos)} unidades)")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: np.unique + return_counts — frecuencia de productos vendidos.*

1. IDs de productos vendidos en un día

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: np.bincount — histograma rápido de categorías

```python
# Categorías de producto (0: Electrónicos, 1: Ropa, 2: Hogar, 3: Deportes)
categorias = np.array([0, 2, 1, 0, 3, 1, 2, 0, 1, 3, 3, 0, 2, 1])
conteos = np.bincount(categorias, minlength=4)

nombres = ["Electrónicos", "Ropa", "Hogar", "Deportes"]
print("Categoría       Unidades vendidas")
for cat, cnt in enumerate(conteos):
    print(f"{nombres[cat]:15s} {cnt}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: np.bincount — histograma rápido de categorías.*

1. Categorías de producto (0: Electrónicos, 1: Ropa, 2: Hogar, 3: Deportes)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: np.histogram — distribución de precios con bins personalizados

```python
precios = np.array([12, 45, 78, 120, 35, 89, 150, 200, 65, 180,
                    25, 95, 110, 55, 140, 170, 30, 75, 130, 190])

bins = [0, 50, 100, 150, 200, 250]
conteos, bordes = np.histogram(precios, bins=bins)

print("Rango precio     Productos")
for i in range(len(conteos)):
    print(f"${bordes[i]:3.0f} - ${bordes[i+1]:3.0f}      {conteos[i]}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: np.histogram — distribución de precios con bins personalizados.*

1. `print("Rango precio     Productos")` — Muestra el resultado por pantalla.
2. `print(f"${bordes[i]:3.0f} - ${bordes[i+1]:3.0f}      {conteos[i]}")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: np.digitize — asignar productos a bins de precio

```python
precios_productos = np.array([25, 180, 65, 220, 45, 150, 90, 300])
bins_precio = [0, 50, 100, 200, 500]

indices_bin = np.digitize(precios_productos, bins_precio)
etiquetas = ["Económico", "Popular", "Estándar", "Premium", "Lujo"]

print("Prod  Precio  Categoría")
for i, (p, b) in enumerate(zip(precios_productos, indices_bin)):
    print(f"{i+1:3d}  ${p:4.0f}  {etiquetas[b-1]}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: np.digitize — asignar productos a bins de precio.*

1. `print("Prod  Precio  Categoría")` — Muestra el resultado por pantalla.
2. `print(f"{i+1:3d}  ${p:4.0f}  {etiquetas[b-1]}")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: np.interp — interpolar demanda faltante entre días

```python
# Demanda conocida en ciertos días
dias_conocidos = np.array([0, 3, 5, 7, 10])
demanda_conocida = np.array([100, 130, 120, 150, 140])

# Días a interpolar
dias_a_interpolar = np.arange(0, 11)
demanda_interpolada = np.interp(dias_a_interpolar, dias_conocidos, demanda_conocida)

print("Día  Demanda")
for d, dem in zip(dias_a_interpolar, demanda_interpolada):
    print(f"{d:2d}   {dem:.0f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: np.interp — interpolar demanda faltante entre días.*

1. Demanda conocida en ciertos días
2. Días a interpolar

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: np.cross — producto cruzado de vectores de precio y cantidad

```python
# Vector de precios y vector de cantidades en 3D
precios = np.array([120, 250, 180])    # 3 productos
cantidades = np.array([50, 30, 45])

# Momento económico = cross(precios, cantidades)
momento = np.cross(precios, cantidades)
print(f"Precios: {precios}")
print(f"Cantidades: {cantidades}")
print(f"Producto cruzado: {momento}")

# En 2D: np.cross devuelve escalar
p2d = np.array([120, 250])
c2d = np.array([50, 30])
print(f"Cross 2D (momento): {np.cross(p2d, c2d)}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: np.cross — producto cruzado de vectores de precio y cantidad.*

1. Vector de precios y vector de cantidades en 3D
2. Momento económico = cross(precios, cantidades)
3. En 2D: np.cross devuelve escalar

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: np.trapz con múltiples segmentos

```python
# Ventas en 4 trimestres (3 segmentos)
trimestres = np.array([0, 3, 6, 9, 12])
ventas = np.array([100, 150, 130, 170, 160])

# Área total bajo curva (ventas totales ponderadas en tiempo)
area_total = np.trapz(ventas, trimestres)

# Área por segmento
area_anual = 0
for i in range(4):
    seg = np.trapz(ventas[i:i+2], trimestres[i:i+2])
    area_anual += seg
    print(f"Segmento {i+1}: {seg:.1f}")

print(f"Área total: {area_total:.1f}")
print(f"Ventas promedio ponderadas: {area_total/12:.1f}/mes")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: np.trapz con múltiples segmentos.*

1. Ventas en 4 trimestres (3 segmentos)
2. Área total bajo curva (ventas totales ponderadas en tiempo)
3. Área por segmento

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: np.gradient para detectar cambios bruscos en ventas

```python
ventas = np.array([100, 105, 102, 500, 510, 508, 110, 115, 112])
cambio = np.gradient(ventas)

# Detectar cambios > 3 desviaciones estándar
umbral = np.std(cambio) * 3
anomalias = np.where(np.abs(cambio) > umbral)[0]

print(f"Cambios: {cambio}")
print(f"Umbral anomalía: ±{umbral:.1f}")
if len(anomalias) > 0:
    print(f"⚠ Posibles anomalías en índices: {anomalias}")
    for idx in anomalias:
        print(f"  Día {idx}: ventas={ventas[idx]}, cambio={cambio[idx]:+.0f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: np.gradient para detectar cambios bruscos en ventas.*

1. Detectar cambios > 3 desviaciones estándar

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: np.nan funciones vs funciones normales

```python
arr = np.array([10, 20, np.nan, 40, np.nan, 60])

print("Función     Normal  NaN-safe")
print(f"mean        {np.mean(arr):.1f}    {np.nanmean(arr):.1f}")
print(f"sum         {np.sum(arr):.0f}     {np.nansum(arr):.0f}")
print(f"std         {np.std(arr):.1f}    {np.nanstd(arr):.1f}")
print(f"min         {np.min(arr):.0f}     {np.nanmin(arr):.0f}")
print(f"max         {np.max(arr):.0f}     {np.nanmax(arr):.0f}")

# Con nanpercentile
print(f"percentile  {np.percentile(arr, 50):.1f}    {np.nanpercentile(arr, 50):.1f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: np.nan funciones vs funciones normales.*

1. Con nanpercentile

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 16: np.diff — diferencias entre días consecutivos

```python
# Ventas diarias de una semana
ventas = np.array([1200, 1350, 1100, 1500, 1600, 1450, 1700])

diferencias = np.diff(ventas)
print("Día  Ventas  vs día ant.")
for i in range(len(ventas)):
    if i == 0:
        print(f"{i+1}   {ventas[i]:5.0f}      ---")
    else:
        print(f"{i+1}   {ventas[i]:5.0f}   {diferencias[i-1]:+6.0f}")

print(f"\nMejor día: +{diferencias.max():.0f} en día {diferencias.argmax()+2}")
print(f"Peor día:  {diferencias.min():.0f} en día {diferencias.argmin()+2}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: np.diff — diferencias entre días consecutivos.*

1. Ventas diarias de una semana

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 17: np.ediff1d — diferencias como 1D

```python
# matriz de precios: [sucursales, 4 semanas]
precios = np.array([
    [100, 105, 102, 108],
    [200, 195, 210, 205],
    [150, 152, 148, 155]
])

diffs = np.ediff1d(precios)
print(f"Precios matrix shape: {precios.shape}")
print(f"Diferencias 1D: {diffs}")
print(f"Total diferencias: {len(diffs)}")
print(f"Máximo cambio: {diffs.max()}")
print(f"Mínimo cambio: {diffs.min()}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: np.ediff1d — diferencias como 1D.*

1. matriz de precios: [sucursales, 4 semanas]

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 18: np.cumsum con máscara para reset acumulado

```python
# Ventas por día, queremos acumulado por semana
ventas = np.array([10, 20, 15, 30, 25, 40, 35, 50, 45, 60, 55, 70, 65, 80])

# Máscara: primer día de cada semana
dias_semana = np.arange(len(ventas)) % 7
inicio_semana = dias_semana == 0

# Acumulado con reset por semana
acumulado = np.cumsum(ventas)
for idx in np.where(inicio_semana)[0][1:]:
    acumulado[idx:] -= acumulado[idx-1]

print("Día  Ventas  Acum. semana")
for i in range(len(ventas)):
    print(f"{i+1:2d}   {ventas[i]:4.0f}   {acumulado[i]:6.0f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: np.cumsum con máscara para reset acumulado.*

1. Ventas por día, queremos acumulado por semana
2. Máscara: primer día de cada semana
3. Acumulado con reset por semana

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 3. Resumen Teórico

| Función | Aplicación en Ventas/Compras/Inventarios |
|---------|----------------------------------------|
| nanmean, nansum, nanstd | Métricas robustas con datos incompletos |
| nanpercentile | Análisis de distribución sin sesgo por NaN |
| trapz | Área bajo curva de demanda, ingresos acumulados |
| gradient | Velocidad de cambio en ventas, tendencias |
| diff, ediff1d | Cambios día a día, detección de picos |
| unique + return_counts | Productos más vendidos, frecuencia de proveedores |
| bincount | Conteo rápido de categorías de productos |
| histogram | Distribución de precios, montos de factura |
| digitize | Segmentación automática en rangos de precio |
| interp | Rellenar datos faltantes de demanda |
| cross | Análisis de portafolio, correlación precio-cantidad |
| cumsum | Totales acumulados, tendencias a largo plazo |

---

## 4. Ejercicios Propuestos

**Ejercicio 1:** Dado un array de ventas mensuales con 3 valores NaN, calcula promedio, mediana y desviación estándar usando funciones NaN-safe.

**Ejercicio 2:** Genera una curva de demanda con np.trapz para estimar el ingreso total potencial entre precios de $10 y $100.

**Ejercicio 3:** Usa np.gradient para identificar el día con mayor crecimiento en ventas de un mes de 30 días.

**Ejercicio 4:** Con np.unique + return_counts, determina los 3 productos más vendidos de una lista de 100 transacciones.

**Ejercicio 5:** Crea un histograma con np.histogram para analizar la distribución de montos de compra, usando 5 bins personalizados.

**Ejercicio 6:** Usa np.interp para estimar el valor de inventario en días sin conteo físico conocido.

**Ejercicio 7:** Con np.cumsum y una máscara de reinicio, calcula el acumulado de ventas por mes para 60 días de datos.

**Ejercicio 8:** Aplica np.digitize para clasificar 30 productos en 4 segmentos de precio (bajo, medio, alto, premium) y reporta cuántos caen en cada uno.

---

*Fin del documento I02 — Funciones Maestras de NumPy*
