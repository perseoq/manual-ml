# I01 — Broadcasting y Funciones Avanzadas de NumPy

## 1. Introducción Teórica

### Broadcasting

Broadcasting es el mecanismo de NumPy para trabajar con arrays de diferentes formas durante operaciones aritméticas. En lugar de expandir físicamente los arrays (lo que consumiría memoria), NumPy "transmite" virtualmente las dimensiones más pequeñas sobre las más grandes.

**Reglas de broadcasting:**
1. Si los arrays tienen diferente número de dimensiones, se agregan dimensiones de tamaño 1 al frente del array más pequeño.
2. Los arrays son compatibles si en cada dimensión ambos tienen el mismo tamaño, o uno de ellos tiene tamaño 1.
3. El resultado tiene el tamaño máximo en cada dimensión.

**Ejemplo visual:** `(3,4) + (4,)` → `(3,4) + (1,4)` → `(3,4) + (3,4)` → `(3,4)`

### Funciones avanzadas

- **np.where(condition, x, y):** Selecciona elementos de `x` o `y` según `condition`.
- **np.clip(a, min, max):** Limita valores a un rango.
- **np.extract(condition, arr):** Extrae elementos donde condition es True (devuelve 1D).
- **np.piecewise(x, condlist, funclist):** Evalúa diferentes funciones según condiciones.
- **np.select(condlist, choicelist, default=0):** Toma decisiones múltiples vectorizadas.
- **np.choose(a, choices):** Construye array seleccionando de múltiples opciones.
- **np.vectorize(pyfunc):** Vectoriza una función Python (nota: no mejora rendimiento real).
- **np.frompyfunc(func, nin, nout):** Crea ufunc desde función Python.
- **np.apply_along_axis(func1d, axis, arr):** Aplica función a lo largo de un eje.
- **np.apply_over_axes(func, a, axes):** Aplica función repetidamente sobre ejes.
- **np.nditer:** Iterador eficiente sobre arrays multidimensionales.
- **np.ndenumerate:** Itera con índices al estilo Python.

### Strides y Memory Layout

Los strides son los saltos en bytes entre elementos consecutivos a lo largo de cada dimensión. El orden C (row-major) es el default; el orden F (column-major) puede ser más rápido para ciertas operaciones columna por columna.

---

## 2. Ejemplos Prácticos

### Ejemplo 1: Broadcasting array+scalar (descuento %)

```python
import numpy as np

# Precios originales de 12 productos
precios = np.array([150.0, 280.0, 95.0, 410.0, 230.0, 175.0,
                    320.0, 60.0, 510.0, 185.0, 275.0, 390.0])
descuento = 0.15  # 15% de descuento

precios_final = precios * (1 - descuento)
print("Precios originales:", precios)
print("Precios con 15% desc.:", precios_final)
print("Broadcasting: scalar se transmite a cada elemento")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: Broadcasting array+scalar (descuento %).*

1. Precios originales de 12 productos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 2: Broadcasting 2D+1D (sucursales x productos + costos)

```python
# ingresos[3 sucursales, 5 productos]
ingresos = np.array([
    [1200, 3400, 2800, 1500, 4200],
    [2100, 1800, 3900, 2700, 3100],
    [1500, 2600, 2100, 3800, 1900]
])

# costo_base por producto (5,)
costo_base = np.array([300, 500, 400, 350, 600])

# margen = ingresos - costo_base (broadcasting (3,5) - (5,) -> (3,5))
margen = ingresos - costo_base
print("Margen bruto por sucursal-producto:\n", margen)
print("Forma resultado:", margen.shape)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: Broadcasting 2D+1D (sucursales x productos + costos).*

1. ingresos[3 sucursales, 5 productos]
2. costo_base por producto (5,)
3. margen = ingresos - costo_base (broadcasting (3,5) - (5,) -> (3,5))

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 3: Broadcasting 3D (sucursales x días x productos)

```python
# ventas[2 sucursales, 7 días, 4 productos]
np.random.seed(42)
ventas = np.random.randint(10, 100, size=(2, 7, 4)).astype(float)

# factor_estacional[4 productos] - se transmite a (1,1,4) -> (2,7,4)
factor_estacional = np.array([1.2, 0.8, 1.1, 0.9])

ventas_ajustadas = ventas * factor_estacional
print("Ventas originales (suc1, día1):", ventas[0, 0])
print("Ventas ajustadas (suc1, día1):", ventas_ajustadas[0, 0])
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: Broadcasting 3D (sucursales x días x productos).*

1. ventas[2 sucursales, 7 días, 4 productos]
2. factor_estacional[4 productos] - se transmite a (1,1,4) -> (2,7,4)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 4: np.where — marcar productos con margen > 30%

```python
precios = np.array([150, 280, 95, 410, 230, 175, 320, 60, 510])
costos = np.array([100, 200, 80, 300, 180, 140, 250, 50, 400])
margen_pct = (precios - costos) / precios * 100

etiqueta = np.where(margen_pct > 30, "Alto margen", "Bajo margen")
print("Margen %:", margen_pct)
print("Etiqueta:", etiqueta)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: np.where — marcar productos con margen > 30%.*

1. `print("Margen %:", margen_pct)` — Muestra el resultado por pantalla.
2. `print("Etiqueta:", etiqueta)` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 5: np.where anidado — clasificar margen en bajo/medio/alto

```python
margen_pct = np.array([15, 35, 42, 8, 28, 55, 18, 32, 48])

clasificacion = np.where(
    margen_pct < 20, "Bajo",
    np.where(margen_pct < 40, "Medio", "Alto")
)
print("Margen %:", margen_pct)
print("Clasificación:", clasificacion)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: np.where anidado — clasificar margen en bajo/medio/alto.*

1. `print("Margen %:", margen_pct)` — Muestra el resultado por pantalla.
2. `print("Clasificación:", clasificacion)` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 6: np.clip — limitar precios entre mínimo y máximo

```python
precios = np.array([5.0, 120.0, 2500.0, 45.0, 18.0, 9999.0, 0.50])
precio_min, precio_max = 10.0, 2000.0

precios_limitados = np.clip(precios, precio_min, precio_max)
print("Original:", precios)
print("Limitado:", precios_limitados)
# Útil para evitar outliers o errores de carga
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: np.clip — limitar precios entre mínimo y máximo.*

1. Útil para evitar outliers o errores de carga

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 7: np.extract — extraer ventas > $10,000

```python
ventas = np.array([8500, 12000, 5400, 15000, 7200, 11000, 3000, 18500])
ventas_altas = np.extract(ventas > 10000, ventas)
print("Ventas > $10,000:", ventas_altas)
print("Cantidad:", len(ventas_altas))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: np.extract — extraer ventas > $10,000.*

1. `print("Ventas > $10,000:", ventas_altas)` — Muestra el resultado por pantalla.
2. `print("Cantidad:", len(ventas_altas))` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 8: np.piecewise — múltiples condiciones de precio

```python
precios = np.array([25, 80, 150, 300, 500, 750, 1000])

def rango_bajo(x): return "Económico"
def rango_medio(x): return "Estándar"
def rango_alto(x): return "Premium"
def rango_lujo(x): return "Lujo"

etiquetas = np.piecewise(
    precios,
    [precios < 50, (precios >= 50) & (precios < 200),
     (precios >= 200) & (precios < 500), precios >= 500],
    [rango_bajo, rango_medio, rango_alto, rango_lujo]
)
print("Precios:", precios)
print("Etiquetas:", etiquetas)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: np.piecewise — múltiples condiciones de precio.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 9: np.select — múltiples condiciones con default

```python
compras = np.array([5, 50, 120, 500, 1000, 8, 300])

condiciones = [
    compras < 10,
    (compras >= 10) & (compras < 100),
    (compras >= 100) & (compras < 400),
    compras >= 400
]
categorias = ["Minorista", "Mayorista", "Distribuidor", "Corporativo"]
resultado = np.select(condiciones, categorias, default="Desconocido")
print("Compras:", compras)
print("Categoría:", resultado)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: np.select — múltiples condiciones con default.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 10: np.choose — seleccionar de múltiples arrays

```python
# 3 estrategias de precio
estrategia_a = np.array([100, 200, 300, 400, 500])
estrategia_b = np.array([110, 190, 310, 390, 510])
estrategia_c = np.array([105, 205, 295, 405, 495])

# Elegir estrategia por producto (índices 0,1,2)
selector = np.array([0, 1, 2, 0, 1])
precios_final = np.choose(selector, [estrategia_a, estrategia_b, estrategia_c])
print("Selector:", selector)
print("Precios finales:", precios_final)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: np.choose — seleccionar de múltiples arrays.*

1. 3 estrategias de precio
2. Elegir estrategia por producto (índices 0,1,2)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 11: np.vectorize — función personalizada sobre array

```python
def calcular_envio(precio):
    if precio < 100:
        return precio * 0.15
    elif precio < 500:
        return precio * 0.10
    else:
        return 0  # envío gratis

precios = np.array([50, 200, 600, 80, 350])
costo_envio = np.vectorize(calcular_envio)(precios)
print("Precios:", precios)
print("Costo envío:", costo_envio)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: np.vectorize — función personalizada sobre array.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 12: np.apply_along_axis — aplicar función por fila/columna

```python
# ventas[4 sucursales, 6 meses]
ventas = np.random.randint(1000, 5000, size=(4, 6))

def crecimiento(serie):
    return (serie[-1] - serie[0]) / serie[0] * 100

# Aplicar a lo largo de axis=1 (por sucursal / por fila)
crecimientos = np.apply_along_axis(crecimiento, axis=1, arr=ventas)
print("Ventas:\n", ventas)
print("Crecimiento % por sucursal:", crecimientos)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: np.apply_along_axis — aplicar función por fila/columna.*

1. ventas[4 sucursales, 6 meses]
2. Aplicar a lo largo de axis=1 (por sucursal / por fila)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 13: np.apply_over_axes — múltiples ejes

```python
# inventario[2 almacenes, 3 categorías, 4 meses]
inventario = np.random.randint(50, 200, size=(2, 3, 4))

# Suma sobre ejes (0,2): total por categoría sumando almacenes y meses
resultado = np.apply_over_axes(np.sum, inventario, axes=[0, 2])
print("Forma original:", inventario.shape)
print("Forma después de apply_over_axes:", resultado.shape)
print("Total por categoría (sobre almacenes y meses):\n", resultado.ravel())
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: np.apply_over_axes — múltiples ejes.*

1. inventario[2 almacenes, 3 categorías, 4 meses]
2. Suma sobre ejes (0,2): total por categoría sumando almacenes y meses

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 14: Comparar velocidad — loop vs vectorization vs broadcasting

```python
import time

N = 1_000_000
precios = np.random.uniform(10, 1000, N)
descuento = np.random.uniform(0.05, 0.30, N)

# Loop Python puro (lento)
t0 = time.time()
result_loop = np.empty(N)
for i in range(N):
    result_loop[i] = precios[i] * (1 - descuento[i])
t_loop = time.time() - t0

# Vectorización
t0 = time.time()
result_vec = precios * (1 - descuento)
t_vec = time.time() - t0

print(f"Loop:     {t_loop:.4f}s")
print(f"Vectoriz: {t_vec:.4f}s")
print(f"Speedup:  {t_loop/t_vec:.1f}x")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: Comparar velocidad — loop vs vectorization vs broadcasting.*

1. Loop Python puro (lento)
2. Vectorización

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 15: np.nditer — iteración manual sobre array multidimensional

```python
# inventario 3D [2 almacenes, 3 pasillos, 4 estantes]
inventario = np.arange(24).reshape(2, 3, 4)

print("Iterando con nditer:")
for val in np.nditer(inventario, order='C'):
    print(val, end=' ')
print()

# Usando op_flags para modificar
it = np.nditer(inventario, op_flags=['readwrite'])
for val in it:
    val[...] = val * 1.1  # incrementar 10%
print("Inventario después de 10%:\n", inventario)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: np.nditer — iteración manual sobre array multidimensional.*

1. inventario 3D [2 almacenes, 3 pasillos, 4 estantes]
2. Usando op_flags para modificar

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 16: np.ndenumerate — iterar con índices

```python
# stock crítico: marcar posiciones con bajo stock
stock = np.array([[15, 3, 42], [8, 0, 25], [30, 12, 5]])
print("Stock:\n", stock)

print("Posiciones con stock crítico (< 10):")
for idx, valor in np.ndenumerate(stock):
    if valor < 10:
        print(f"  Almacén {idx[0]}, pasillo {idx[1]}: {valor} unidades")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: np.ndenumerate — iterar con índices.*

1. stock crítico: marcar posiciones con bajo stock

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 17: Memory layout (C vs Fortran order) y performance

```python
N = 5000
A = np.random.rand(N, N)

# Operaciones por filas (C-order → rápido)
A_c = np.ascontiguousarray(A)
t0 = time.time()
suma_filas = A_c.sum(axis=1)
t_c = time.time() - t0

# Operaciones por columnas (F-order → rápido)
A_f = np.asfortranarray(A)
t0 = time.time()
suma_cols = A_f.sum(axis=0)
t_f = time.time() - t0

print(f"Strides C-order: {A_c.strides}")
print(f"Strides F-order: {A_f.strides}")
print(f"Suma filas (C):  {t_c:.4f}s")
print(f"Suma cols (F):   {t_f:.4f}s")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: Memory layout (C vs Fortran order) y performance.*

1. Operaciones por filas (C-order → rápido)
2. Operaciones por columnas (F-order → rápido)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 18: Ejercicio integrador — procesar matriz de ventas

```python
# Matriz ventas[12 meses, 5 categorías, 4 sucursales]
np.random.seed(123)
ventas = np.random.normal(10000, 2000, size=(12, 5, 4)).round(2)

# 1. Aplicar crecimiento estacional (broadcasting)
factor_mes = np.array([0.8, 0.9, 1.0, 1.1, 1.2, 1.3,
                        1.3, 1.2, 1.1, 1.0, 0.9, 0.8])
ventas_est = ventas * factor_mes[:, np.newaxis, np.newaxis]

# 2. np.where: marcar meses con ventas sobre promedio
promedio_global = np.mean(ventas_est)
mascara_alto = np.where(ventas_est > promedio_global, "Alto", "Bajo")

# 3. np.clip: evitar valores negativos o extremos
ventas_limpias = np.clip(ventas_est, 1000, 30000)

# 4. np.extract: sucursales con ventas altas en diciembre
dic = ventas_limpias[11]
altas = np.extract(dic > 15000, dic)
print(f"Ventas totales: {ventas_limpias.sum():.0f}")
print(f"Ventas altas en diciembre: {len(altas)} registros")
print(f"Promedio global: {promedio_global:.0f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Ejercicio integrador — procesar matriz de ventas.*

1. Matriz ventas[12 meses, 5 categorías, 4 sucursales]
2. 1. Aplicar crecimiento estacional (broadcasting)
3. 2. np.where: marcar meses con ventas sobre promedio
4. 3. np.clip: evitar valores negativos o extremos
5. 4. np.extract: sucursales con ventas altas en diciembre

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

## 3. Resumen Teórico

| Concepto | Uso en Ventas/Compras/Inventarios |
|----------|----------------------------------|
| Broadcasting | Aplicar descuentos, costos, impuestos a matrices sin loops |
| np.where | Clasificar productos, marcar umbrales, segmentar clientes |
| np.clip | Limpiar outliers en precios, cantidades, montos |
| np.extract | Filtrar transacciones que cumplen condiciones |
| np.piecewise | Segmentación por rangos de precio o cantidad |
| np.select | Múltiples condiciones categóricas (tipo cliente, canal) |
| np.choose | Seleccionar estrategias de precio según perfil |
| np.vectorize | Aplicar lógica de negocio personalizada |
| np.apply_along_axis | Métricas por sucursal, vendedor, producto |
| np.apply_over_axes | Agregaciones multi-eje (año, categoría, almacén) |
| np.nditer | Procesamiento manual cuando no hay función vectorizada |
| np.ndenumerate | Localizar posiciones con bajo stock, anomalías |
| Strides | Optimizar layout según acceso a datos |

---

## 4. Ejercicios Propuestos

**Ejercicio 1:** Dada una matriz `compras[4 trimestres, 6 proveedores]`, usa broadcasting para aplicar un incremento de precio del 8% a todos los valores.

**Ejercicio 2:** Con `np.where` anidado, clasifica productos en "Bajo stock" (< 20), "Stock medio" (20-100), "Stock alto" (> 100) dado un array de existencias.

**Ejercicio 3:** Usa `np.clip` para limitar montos de factura entre $50 y $5000, simulando validación de cargos.

**Ejercicio 4:** Extrae con `np.extract` las ventas de los 3 mejores meses de un año dado.

**Ejercicio 5:** Usa `np.piecewise` para asignar categorías de descuento (5%, 10%, 15%, 20%) según el monto de compra.

**Ejercicio 6:** Implementa una función con `np.vectorize` que calcule el impuesto (16% si precio < 1000, 8% si >= 1000).

**Ejercicio 7:** Dada una matriz `inventario[10 almacenes, 20 productos]`, usa `np.apply_along_axis` para encontrar el producto más vendido por almacén.

**Ejercicio 8:** Crea un array 3D de ventas (4 semanas × 7 días × 5 productos) y aplica al menos 4 técnicas de este módulo para procesarlo. Documenta cada paso.

---

*Fin del documento I01 — Broadcasting y Funciones Avanzadas de NumPy*
