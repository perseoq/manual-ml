# B09 – Creación de Arrays en NumPy

## Introducción

NumPy es la librería fundamental para computación numérica en Python. Su estructura principal es el **ndarray** (n-dimensional array), que permite operaciones vectorizadas eficientes. En el contexto de ventas, compras e inventarios, los arrays nos permiten manejar grandes volúmenes de datos numéricos (precios, cantidades, costos) sin usar bucles explícitos.

---

## 1. `np.array` — Crear array desde lista

```python
import numpy as np

# Precios de 5 productos en una tienda
precios = np.array([150.0, 230.0, 85.5, 420.0, 67.0])
print("Precios:", precios)
print("Tipo:", type(precios))
print("Dimensiones:", precios.ndim)
print("Forma:", precios.shape)
```

**Salida esperada:**
```
Precios: [150.  230.   85.5 420.   67. ]
Tipo: <class 'numpy.ndarray'>
Dimensiones: 1
Forma: (5,)
```

**Explicación línea por línea:**
- `import numpy as np`: importa NumPy con el alias estándar `np`.
- `np.array([...])`: crea un ndarray unidimensional a partir de una lista de Python.
- `precios.ndim`: devuelve el número de dimensiones (1 → vector).
- `precios.shape`: devuelve una tupla con el tamaño en cada dimensión `(5,)`.

---

## 2. `np.zeros` — Array de ceros

```python
import numpy as np

# Stock inicial de 4 sucursales, cada una con 3 productos: cero stock
stock_inicial = np.zeros((4, 3), dtype=int)
print("Stock inicial:\n", stock_inicial)
print("Forma:", stock_inicial.shape)
```

**Salida esperada:**
```
Stock inicial:
 [[0 0 0]
 [0 0 0]
 [0 0 0]
 [0 0 0]]
Forma: (4, 3)
```

**Explicación:**
- `np.zeros((4,3), dtype=int)`: crea array 4×3 lleno de ceros con tipo entero.
- Cada fila representa una sucursal; cada columna un producto diferente.

---

## 3. `np.ones` — Array de unos

```python
import numpy as np

# Factor de markup del 100% (1.0) para calcular precio venta desde costo
markup = np.ones((3, 3))
print("Markup:\n", markup)
```

**Salida esperada:**
```
Markup:
 [[1. 1. 1.]
 [1. 1. 1.]
 [1. 1. 1.]]
```

**Explicación:**
- `np.ones((3,3))`: crea array 3×3 con valores 1.0 de tipo float64 por defecto.
- Útil para inicializar matrices de factores multiplicativos.

---

## 4. `np.full` — Array con valor constante

```python
import numpy as np

# Precio fijo de 99.90 para todos los productos en una promoción
precio_promo = np.full(8, 99.90)
print("Precios promoción:", precio_promo)
```

**Salida esperada:**
```
Precios promoción: [99.9 99.9 99.9 99.9 99.9 99.9 99.9 99.9]
```

**Explicación:**
- `np.full(8, 99.90)`: crea un vector de 8 elementos todos con valor 99.90.

---

## 5. `np.arange` — Secuencia con paso

```python
import numpy as np

# IDs de productos del 1001 al 1020 (excluye 1020)
ids_productos = np.arange(1001, 1020, 3)
print("IDs:", ids_productos)
```

**Salida esperada:**
```
IDs: [1001 1004 1007 1010 1013 1016 1019]
```

**Explicación:**
- `np.arange(inicio, fin, paso)`: genera valores desde `inicio` hasta `fin-1` con incremento de `paso`.
- Similar a `range()` pero devuelve un ndarray.

---

## 6. `np.linspace` — Secuencia lineal

```python
import numpy as np

# 5 rangos de precios equiespaciados entre 10 y 500
rangos_precios = np.linspace(10, 500, 5)
print("Rangos:", rangos_precios)
```

**Salida esperada:**
```
Rangos: [ 10.   132.5  255.   377.5  500. ]
```

**Explicación:**
- `np.linspace(10, 500, 5)`: genera 5 valores equiespaciados entre 10 y 500 (inclusive ambos extremos).
- Difiere de `arange` en que se especifica la cantidad de puntos, no el paso.

---

## 7. `np.random.rand` — Números aleatorios uniformes [0,1)

```python
import numpy as np

# Simular 5 descuentos aleatorios entre 0% y 100%
descuentos = np.random.rand(5)
print("Descuentos:", descuentos)
```

**Salida esperada:**
```
Descuentos: [0.456 0.789 0.123 0.987 0.345]
```

**Explicación:**
- `np.random.rand(5)`: genera 5 valores aleatorios uniformes en [0, 1).
- Cada valor puede representar un porcentaje de descuento aplicado a un producto.

---

## 8. `np.identity` — Matriz identidad

```python
import numpy as np

# Matriz identidad 4x4 para representar 4 productos sin relación entre sí
identidad = np.identity(4)
print("Identidad:\n", identidad)
```

**Salida esperada:**
```
Identidad:
 [[1. 0. 0. 0.]
 [0. 1. 0. 0.]
 [0. 0. 1. 0.]
 [0. 0. 0. 1.]]
```

**Explicación:**
- `np.identity(4)`: matriz cuadrada con unos en la diagonal principal.
- Útil en álgebra lineal para representar transformaciones sin efecto.

---

## 9. `np.diag` — Diagonal de una matriz o crear matriz diagonal

```python
import numpy as np

# Precios base de 4 productos en la diagonal
precios_base = np.array([120, 340, 65, 210])
matriz_costos = np.diag(precios_base)
print("Matriz diagonal de costos:\n", matriz_costos)
```

**Salida esperada:**
```
Matriz diagonal de costos:
 [[120   0   0   0]
 [  0 340   0   0]
 [  0   0  65   0]
 [  0   0   0 210]]
```

**Explicación:**
- `np.diag(vector)`: crea una matriz con el vector en la diagonal principal y ceros fuera.
- Útil para modelar costos unitarios donde cada producto tiene precio independiente.

---

## 10. `np.fromstring` — Crear array desde string

```python
import numpy as np

# Datos de ventas del día separados por comas
ventas_str = "45, 78, 23, 91, 56, 34"
ventas = np.fromstring(ventas_str, dtype=int, sep=",")
print("Ventas del día:", ventas)
print("Total:", ventas.sum())
```

**Salida esperada:**
```
Ventas del día: [45 78 23 91 56 34]
Total: 327
```

**Explicación:**
- `np.fromstring(ventas_str, dtype=int, sep=",")`: parsea el string usando coma como separador y lo convierte en array de enteros.
- Ideal para leer datos copiados desde Excel o CSV en formato texto.

---

## 11. `np.loadtxt` — Cargar desde archivo de texto

```python
import numpy as np
import tempfile, os

# Crear archivo temporal con datos de ventas
tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
tmp.write("producto,precio,cantidad\n")
tmp.write("1,150.0,10\n")
tmp.write("2,230.0,5\n")
tmp.write("3,85.5,20\n")
tmp.close()

# Cargar saltando la cabecera
datos = np.loadtxt(tmp.name, delimiter=",", skiprows=1, usecols=(1,2))
os.unlink(tmp.name)
print("Datos cargados:\n", datos)
```

**Salida esperada:**
```
Datos cargados:
 [[150.  10.]
 [230.   5.]
 [ 85.5 20.]]
```

**Explicación:**
- `np.loadtxt(archivo, delimiter=",", skiprows=1)`: carga datos numéricos desde archivo de texto.
- `skiprows=1`: omite la primera línea (cabecera).
- `usecols=(1,2)`: carga solo columnas 1 y 2 (precio y cantidad).

---

## 12. `np.empty` — Array sin inicializar

```python
import numpy as np

# Buffer rápido para 1000 transacciones (valores basura)
buffer_transacciones = np.empty((4, 3))
print("Buffer:\n", buffer_transacciones)
```

**Salida esperada:**
```
Buffer:
 [[2.3e-316 0.0e+000 0.0e+000]
 [0.0e+000 0.0e+000 0.0e+000]
 [0.0e+000 0.0e+000 0.0e+000]
 [0.0e+000 0.0e+000 0.0e+000]]
```

**Explicación:**
- `np.empty((4,3))`: crea un array sin inicializar los valores (contiene lo que haya en memoria).
- Más rápido que `zeros` cuando se va a sobrescribir inmediatamente.
- Advertencia: los valores son impredecibles (basura).

---

## 13. `np.repeat` — Repetir elementos

```python
import numpy as np

# Códigos de 3 productos repetidos para 4 sucursales cada uno
codigos = np.array([101, 102, 103])
codigos_repetidos = np.repeat(codigos, 4)
print("Códigos por sucursal:", codigos_repetidos)
```

**Salida esperada:**
```
Códigos por sucursal: [101 101 101 101 102 102 102 102 103 103 103 103]
```

**Explicación:**
- `np.repeat(codigos, 4)`: repite cada elemento del array 4 veces consecutivas.
- Útil para expandir IDs de productos al número de sucursales o días.

---

## 14. `np.tile` — Repetir array completo

```python
import numpy as np

# Vector de precios de 3 productos, repetido para 4 sucursales
precios = np.array([150.0, 230.0, 85.5])
precios_sucursales = np.tile(precios, (4, 1))
print("Precios en 4 sucursales:\n", precios_sucursales)
```

**Salida esperada:**
```
Precios en 4 sucursales:
 [[150.  230.   85.5]
 [150.  230.   85.5]
 [150.  230.   85.5]
 [150.  230.   85.5]]
```

**Explicación:**
- `np.tile(precios, (4,1))`: repite el array completo como un mosaico 4 veces en filas y 1 vez en columnas.
- A diferencia de `repeat`, replica el array completo en bloque.

---

## 15. `np.meshgrid` — Mallas de coordenadas

```python
import numpy as np

# Sucursales y productos para crear matriz de ventas
sucursales = np.array([1, 2, 3])
productos = np.array([101, 102])
S, P = np.meshgrid(sucursales, productos, indexing='ij')
print("Malla sucursales:\n", S)
print("Malla productos:\n", P)
```

**Salida esperada:**
```
Malla sucursales:
 [[1 1]
 [2 2]
 [3 3]]
Malla productos:
 [[101 102]
 [101 102]
 [101 102]]
```

**Explicación:**
- `np.meshgrid(sucursales, productos, indexing='ij')`: genera matrices de coordenadas para evaluar funciones en una malla.
- Útil para crear combinaciones de sucursal × producto.
- `S` tiene las sucursales repetidas en columnas; `P` los productos repetidos en filas.

---

## Resumen

| Función | Propósito | Aplicación en Ventas |
|---------|-----------|----------------------|
| `np.array` | Desde lista Python | Precios de productos |
| `np.zeros` | Array de ceros | Stock inicial vacío |
| `np.ones` | Array de unos | Factores markup |
| `np.full` | Valor constante | Precios promocionales |
| `np.arange` | Secuencia con paso | IDs de productos |
| `np.linspace` | Puntos equiespaciados | Rangos de precios |
| `np.random.rand` | Aleatorios uniformes | Descuentos aleatorios |
| `np.identity` | Matriz identidad | Costos sin relación cruzada |
| `np.diag` | Diagonal de matriz | Costos unitarios |
| `np.fromstring` | Desde string CSV | Datos desde clipboard |
| `np.loadtxt` | Desde archivo texto | Carga de datos reales |
| `np.empty` | Sin inicializar | Buffer temporal |
| `np.repeat` | Repetir elementos | IDs expandidos |
| `np.tile` | Repetir array completo | Precios por sucursal |
| `np.meshgrid` | Coordenadas de malla | Combinaciones sucursal×producto |

---

## Ejercicios

1. Crea un array con los precios `[99.9, 149.9, 299.9, 49.9, 199.9]` y muestra su forma y dimensiones.

2. Crea una matriz de 6×4 llena de ceros que represente 6 productos y 4 días de stock inicial.

3. Genera un vector con 10 valores equiespaciados entre 0 y 100 que representen porcentajes de descuento.

4. Usando `np.arange`, genera IDs de producto desde 2000 hasta 2010.

5. Crea una matriz identidad de 5×5 que represente 5 productos donde cada producto solo se relaciona consigo mismo.

6. Dado el string `"120,340,65,210,89"`, usa `np.fromstring` para convertirlo en un array de enteros y calcula el precio promedio.

7. Usa `np.tile` para replicar el vector `[10, 20, 30]` en 3 filas, creando una matriz de 3×3.

8. Simula 8 descuentos aleatorios usando `np.random.rand` y multiplícalos por 100 para obtener porcentajes.
