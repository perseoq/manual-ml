# AP02 — Cheatsheet NumPy

## 1. Creación de Arrays

```python
import numpy as np

# Desde lista
arr = np.array([1, 2, 3, 4, 5])
matriz = np.array([[1, 2, 3], [4, 5, 6]])

# Funciones de creación
np.zeros(5)                    # [0., 0., 0., 0., 0.]
np.zeros((3, 4))               # matriz 3x4 de ceros
np.ones((2, 3))                # matriz 2x3 de unos
np.eye(4)                      # identidad 4x4
np.full((2, 2), 7)             # [[7,7],[7,7]]
np.empty((3, 2))               # sin inicializar (valores basura)
np.arange(10)                  # [0..9]
np.arange(2, 10, 2)            # [2,4,6,8]
np.linspace(0, 1, 5)           # [0., 0.25, 0.5, 0.75, 1.]
np.logspace(0, 3, 4)           # [1., 10., 100., 1000.]

# Arrays aleatorios
np.random.seed(42)              # reproducibilidad
np.random.rand(5)               # uniforme [0,1) (5,)
np.random.rand(3, 4)            # uniforme [0,1) 3x4
np.random.randn(5)              # normal estándar
np.random.randint(0, 10, 5)    # enteros aleatorios [0,10)
np.random.uniform(0, 1, 5)     # distribución uniforme
np.random.normal(0, 1, (3, 3)) # distribución normal
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*1. Creación de Arrays.*

1. Desde lista
2. Funciones de creación
3. Arrays aleatorios

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 2. Atributos de Arrays

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

arr.shape            # (2, 3)
arr.ndim             # 2
arr.size             # 6 (total elementos)
arr.dtype            # dtype('int64')
arr.itemsize         # bytes por elemento
arr.nbytes           # bytes totales
arr.T                # transpuesta (3x2)
type(arr)            # <class 'numpy.ndarray'>
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*2. Atributos de Arrays.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 3. Indexación y Slicing

```python
arr = np.array([10, 20, 30, 40, 50])
mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Indexación básica
arr[0]               # 10
arr[-1]              # 50
mat[0, 0]            # 1
mat[1, 2]            # 6
mat[0][0]            # 1 (equivalente)

# Slicing
arr[1:4]             # [20, 30, 40]
arr[:3]              # [10, 20, 30]
arr[::2]             # [10, 30, 50]
arr[::-1]            # [50, 40, 30, 20, 10]
mat[:2, :2]          # [[1,2],[4,5]]
mat[:, 1]            # [2, 5, 8] (columna 1)
mat[1, :]            # [4, 5, 6] (fila 1)

# Indexación booleana
mask = arr > 25
arr[mask]            # [30, 40, 50]
arr[arr % 20 == 0]   # [20, 40]

# Indexación con arrays de enteros (fancy indexing)
arr[[0, 2, 4]]       # [10, 30, 50]
mat[[0, 2], [0, 2]]  # [1, 9] (esquinas)

# np.where
np.where(arr > 25)           # índices: (array([2,3,4]),)
np.where(arr > 25, 1, 0)     # [0,0,1,1,1]
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*3. Indexación y Slicing.*

1. Indexación básica
2. Slicing
3. Indexación booleana
4. Indexación con arrays de enteros (fancy indexing)
5. np.where

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 4. Operaciones Aritméticas

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
escalar = 2

a + b                # [5, 7, 9]
a - b                # [-3, -3, -3]
a * b                # [4, 10, 18]
a / b                # [0.25, 0.4, 0.5]
a ** 2               # [1, 4, 9]
a % 2                # [1, 0, 1]

a + escalar          # [3, 4, 5] (broadcasting)
a * escalar          # [2, 4, 6]

# Operaciones in-place
a += 10              # modifica a
np.add(a, 10)        # equivalente
np.multiply(a, 2)    # equivalente
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*4. Operaciones Aritméticas.*

1. Operaciones in-place

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 5. Broadcasting

```python
# Broadcasting: operaciones entre arrays de diferentes formas
a = np.array([[1, 2, 3], [4, 5, 6]])   # (2, 3)
b = np.array([10, 20, 30])              # (3,)
c = np.array([[10], [20]])              # (2, 1)

a + b                # cada fila suma b: [[11,22,33],[14,25,36]]
a * c                # cada columna multiplica: [[10,20,30],[80,100,120]]
a + 1                # broadcasting con escalar

# Regla: las dimensiones se comparan de derecha a izquierda
# Son compatibles si son iguales o una es 1
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*5. Broadcasting.*

1. Broadcasting: operaciones entre arrays de diferentes formas
2. Regla: las dimensiones se comparan de derecha a izquierda
3. Son compatibles si son iguales o una es 1

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 6. Operaciones Estadísticas

```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
mat = np.array([[1, 2, 3], [4, 5, 6]])

np.mean(arr)           # 5.5
np.median(arr)         # 5.5
np.std(arr)            # desviación estándar (~2.87)
np.var(arr)            # varianza (~8.25)
np.min(arr)            # 1
np.max(arr)            # 10
np.sum(arr)            # 55
np.prod(arr)           # producto
np.cumsum(arr)         # suma acumulativa [1,3,6,10,...]
np.cumprod(arr)        # producto acumulativo

# Estadísticas por eje
np.sum(mat, axis=0)    # suma por columna: [5, 7, 9]
np.sum(mat, axis=1)    # suma por fila: [6, 15]
np.mean(mat, axis=0)   # media por columna: [2.5, 3.5, 4.5]
np.mean(mat, axis=1)   # media por fila: [2., 5.]

# Otras
np.percentile(arr, 25)  # Q1: 3.25
np.percentile(arr, [25, 50, 75])  # cuartiles
np.quantile(arr, 0.25)  # equivalente
np.ptp(arr)             # range (max-min): 9
np.average(arr, weights=np.arange(1, 11))  # media ponderada

# Correlación y covarianza
np.corrcoef(x, y)       # matriz de correlación
np.cov(x, y)            # matriz de covarianza
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*6. Operaciones Estadísticas.*

1. Estadísticas por eje
2. Otras
3. Correlación y covarianza

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 7. Álgebra Lineal

```python
from numpy.linalg import inv, det, eig, svd, qr, norm

A = np.array([[1, 2], [3, 4]])
b = np.array([5, 6])

# Producto matricial
A @ b                # [17, 39] (Python 3.5+)
np.dot(A, b)         # [17, 39]
np.matmul(A, b)      # [17, 39]

# Inversa
inv(A)               # [[-2., 1.],[1.5, -0.5]]

# Determinante
det(A)               # -2.0

# Autovalores y autovectores
eigvals, eigvecs = eig(A)

# Descomposición SVD
U, S, Vt = svd(A)

# Descomposición QR
Q, R = qr(A)

# Normas
norm(A)              # norma Frobenius
norm(A, ord=1)       # norma 1
norm(A, ord=np.inf)  # norma infinito
norm(b, ord=2)       # norma euclidiana (L2)

# Solución de sistemas lineales
np.linalg.solve(A, b)  # [-4., 4.5]

# Traza
np.trace(A)          # 5

# Producto exterior
np.outer([1, 2], [3, 4])  # [[3,4],[6,8]]
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*7. Álgebra Lineal.*

1. Producto matricial
2. Inversa
3. Determinante
4. Autovalores y autovectores
5. Descomposición SVD
6. Descomposición QR
7. Normas
8. Solución de sistemas lineales

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 8. Cambio de Forma (Reshape)

```python
arr = np.arange(12)

arr.reshape(3, 4)          # 3 filas, 4 columnas
arr.reshape(2, -1)         # 2 filas, columna automática (6)
arr.reshape(-1, 4)         # filas automáticas, 4 columnas (3)
arr.reshape(2, 2, 3)       # 3D: 2x2x3

arr.flatten()              # copia 1D
arr.ravel()                # vista 1D (si es posible)
np.ravel(arr)              # equivalente

# Expandir/eliminar dimensiones
np.expand_dims(arr, axis=0)  # (1, 12)
np.expand_dims(arr, axis=1)  # (12, 1)
np.squeeze(arr_3d)           # elimina dims de tamaño 1
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*8. Cambio de Forma (Reshape).*

1. Expandir/eliminar dimensiones

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 9. Concatenación y Split

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# Concatenación
np.concatenate([a, b])            # vertical: (4,2)
np.concatenate([a, b], axis=1)    # horizontal: (2,4)

np.vstack([a, b])                 # stack vertical (equivalente)
np.hstack([a, b])                 # stack horizontal (equivalente)
np.column_stack([a, b])           # como hstack para 1D

np.stack([a, b], axis=0)          # nuevo eje: (2,2,2)
np.stack([a, b], axis=2)          # nuevo eje: (2,2,2) en eje 2

# Split
np.split(arr, 3)                  # divide en 3 partes
np.vsplit(mat, 2)                 # split vertical
np.hsplit(mat, 3)                 # split horizontal
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*9. Concatenación y Split.*

1. Concatenación
2. Split

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 10. Operaciones con Arrays Booleanos

```python
arr = np.array([1, 2, 3, 4, 5])

# Comparaciones
arr > 3                # [False, False, False, True, True]
arr == 3               # [False, False, True, False, False]
(arr > 2) & (arr < 5)  # [False, False, True, True, False]
(arr <= 2) | (arr >= 4) # [True, True, False, True, True]
~(arr == 3)            # [True, True, False, True, True]

# Métodos booleanos
np.any(arr > 4)        # True
np.all(arr > 0)        # True
np.sum(arr > 2)        # 3 (cuenta Trues)
np.count_nonzero(arr > 2)  # 3
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*10. Operaciones con Arrays Booleanos.*

1. Comparaciones
2. Métodos booleanos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 11. Funciones Universales (ufunc)

```python
arr = np.array([1, 2, 3, 4, 5])

# Trigonométricas
np.sin(arr), np.cos(arr), np.tan(arr)
np.arcsin(arr * 0.1), np.arccos(arr * 0.1)

# Exponenciales y logarítmicas
np.exp(arr)            # e^arr
np.log(arr)            # log natural
np.log10(arr)          # log base 10
np.log2(arr)           # log base 2
np.power(2, arr)       # 2^arr

# Redondeo
np.round([1.23, 1.78])     # [1., 2.]
np.ceil([1.23, 1.78])      # [2., 2.]
np.floor([1.23, 1.78])     # [1., 1.]
np.trunc([1.23, 1.78])     # [1., 1.]

# Signo y valor absoluto
np.abs([-1, 2, -3])        # [1, 2, 3]
np.sign([-5, 0, 5])        # [-1, 0, 1]

# Comparación
np.maximum(arr, 3)         # element-wise max con 3: [3,3,3,4,5]
np.minimum(arr, 3)         # element-wise min con 3: [1,2,3,3,3]
np.clip(arr, 2, 4)         # [2,2,3,4,4] (recorta valores)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*11. Funciones Universales (ufunc).*

1. Trigonométricas
2. Exponenciales y logarítmicas
3. Redondeo
4. Signo y valor absoluto
5. Comparación

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 12. Ordenamiento

```python
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])

np.sort(arr)               # [1, 1, 2, 3, 4, 5, 6, 9]
np.argsort(arr)            # índices ordenados
np.argmin(arr)             # índice del mínimo: 1
np.argmax(arr)             # índice del máximo: 5

# Ordenar matrices por eje
mat = np.array([[3, 1], [2, 4]])
np.sort(mat, axis=0)       # [[2,1],[3,4]]
np.sort(mat, axis=1)       # [[1,3],[2,4]]
np.argsort(mat, axis=1)    # índices ordenados por fila

# Partición (encuentra top k)
np.partition(arr, 3)       # los 3 más pequeños primero
np.argpartition(arr, 3)    # índices de partición

# Únicos y conteo
np.unique(arr)             # valores únicos
np.unique(arr, return_counts=True)  # con frecuencias
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*12. Ordenamiento.*

1. Ordenar matrices por eje
2. Partición (encuentra top k)
3. Únicos y conteo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 13. Copias y Vistas

```python
arr = np.array([1, 2, 3, 4, 5])

# Vista (comparte memoria)
vista = arr[1:4]
vista[0] = 99
print(arr)                 # [1, 99, 3, 4, 5]  # modificado!

# Copia (memoria independiente)
copia = arr[1:4].copy()
copia[0] = 0
print(arr)                 # sin cambios

# Verificar si comparten memoria
np.shares_memory(arr, vista)  # True
np.shares_memory(arr, copia)  # False
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*13. Copias y Vistas.*

1. Vista (comparte memoria)
2. Copia (memoria independiente)
3. Verificar si comparten memoria

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 14. Manejo de NaN e Inf

```python
arr = np.array([1, 2, np.nan, 4, np.inf, 6])

np.isnan(arr)              # [False, False, True, False, False, False]
np.isinf(arr)              # [False, False, False, False, True, False]
np.isfinite(arr)           # [True, True, False, True, False, True]

# Ignorar NaN en estadísticas
np.nanmean(arr)            # 3.25 (ignora NaN y Inf)
np.nansum(arr)             # 13.0
np.nanstd(arr)             # desviación ignorando NaN
np.nanmin(arr)             # 1.0
np.nanmax(arr)             # 6.0
np.nan_to_num(arr)         # reemplaza NaN por 0, Inf por grandes

arr[np.isnan(arr)] = 0     # reemplazar NaN manualmente
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*14. Manejo de NaN e Inf.*

1. Ignorar NaN en estadísticas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 15. Entrada/Salida

```python
# Guardar y cargar arrays (.npy)
np.save("ventas.npy", arr)
arr_cargado = np.load("ventas.npy")

# Guardar múltiples arrays (.npz)
np.savez("datos.npz", ventas=arr, precios=mat)
data = np.load("datos.npz")
data["ventas"], data["precios"]

# Texto CSV/TSV
np.savetxt("ventas.csv", arr, delimiter=",", fmt="%.2f")
np.savetxt("ventas.csv", arr, delimiter=",",
           header="ventas", comments="")
arr_csv = np.loadtxt("ventas.csv", delimiter=",")
arr_csv = np.genfromtxt("ventas.csv", delimiter=",", skip_header=1)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*15. Entrada/Salida.*

1. Guardar y cargar arrays (.npy)
2. Guardar múltiples arrays (.npz)
3. Texto CSV/TSV

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## Referencia Rápida

| Categoría | Función | Descripción |
|-----------|---------|-------------|
| Creación | `array()`, `zeros()`, `ones()`, `arange()`, `linspace()` | Crear arrays |
| Estadística | `mean()`, `median()`, `std()`, `var()`, `percentile()` | Métricas |
| Álgebra | `dot()`, `inv()`, `det()`, `eig()`, `svd()`, `solve()` | Operaciones matriciales |
| Reducción | `sum()`, `prod()`, `min()`, `max()`, `cumsum()` | Agregación |
| Forma | `reshape()`, `flatten()`, `ravel()`, `expand_dims()` | Cambio de dimensión |
| Combinar | `concatenate()`, `vstack()`, `hstack()`, `stack()` | Unir arrays |
| Ordenar | `sort()`, `argsort()`, `partition()`, `unique()` | Ordenamiento |
| Búsqueda | `where()`, `argmin()`, `argmax()`, `nonzero()` | Encontrar elementos |
| I/O | `save()`, `load()`, `savetxt()`, `loadtxt()` | Archivos |
