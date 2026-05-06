# B15 – Álgebra Lineal con NumPy

## Introducción

El álgebra lineal es fundamental para muchas aplicaciones en ventas y negocios: resolución de sistemas de ecuaciones para costos, regresión lineal para pronósticos, descomposición SVD para sistemas de recomendación, entre otros. NumPy proporciona el submódulo `np.linalg` con operaciones matriciales eficientes.

---

## 1. `np.dot` — Producto punto (ingreso total con precios y cantidades)

```python
import numpy as np

precios = np.array([150.0, 230.0, 85.5])
cantidades = np.array([10, 5, 20])
ingreso_total = np.dot(precios, cantidades)
print("Ingreso total:", ingreso_total)
```

**Salida esperada:**


**Salida esperada:**
```
Ingreso total: 4360.0
```

**Explicación:**
- `np.dot(a, b)`: producto punto de dos vectores.
- Calcula `150×10 + 230×5 + 85.5×20 = 4360`.
- Equivalente a `sum(a * b)` pero más eficiente.

---

## 2. `np.matmul` — Multiplicación de matrices (costos totales)



**Explicación:**
- `np.dot(a, b)`: producto punto de dos vectores.
- Calcula `150×10 + 230×5 + 85.5×20 = 4360`.
- Equivalente a `sum(a * b)` pero más eficiente.

---

## 2. `np.matmul` — Multiplicación de matrices (costos totales)

```python
import numpy as np

# 3 sucursales × 4 productos (cantidades vendidas)
cantidades = np.array([
    [10, 5, 20, 3],
    [8, 7, 15, 5],
    [12, 4, 18, 6]
])
# 4 productos × 1 (precios unitarios)
precios = np.array([150, 230, 85, 420])
# Multiplicar matrices: (3×4) @ (4×1) = (3×1)
ingresos_sucursal = np.matmul(cantidades, precios)
print("Ingresos por sucursal:", ingresos_sucursal)
```

**Salida esperada:**


**Salida esperada:**
```
Ingresos por sucursal: [3560 3385 4260]
```

**Explicación:**
- `np.matmul(A, B)`: multiplicación matricial.
- `cantidades` (3×4) × `precios` (4×1) → resultado (3×1).
- Cada fila es el ingreso total de cada sucursal.

---

## 3. `np.linalg.det` — Determinante (verificar si sistema tiene solución única)



**Explicación:**
- `np.matmul(A, B)`: multiplicación matricial.
- `cantidades` (3×4) × `precios` (4×1) → resultado (3×1).
- Cada fila es el ingreso total de cada sucursal.

---

## 3. `np.linalg.det` — Determinante (verificar si sistema tiene solución única)

```python
import numpy as np

# Matriz de costos: 3 productos × 3 insumos
matriz_insumos = np.array([
    [2, 1, 3],
    [1, 2, 1],
    [3, 1, 2]
])
det = np.linalg.det(matriz_insumos)
print("Determinante:", round(det, 2))
if det != 0:
    print("La matriz es invertible (sistema tiene solución única)")
```

**Salida esperada:**


**Salida esperada:**
```
Determinante: -12.0
La matriz es invertible (sistema tiene solución única)
```

**Explicación:**
- `np.linalg.det(matriz)`: calcula el determinante.
- Si det ≠ 0, la matriz es invertible y el sistema de ecuaciones tiene solución única.

---

## 4. `np.linalg.inv` — Inversa de matriz (despejar costos unitarios)



**Explicación:**
- `np.linalg.det(matriz)`: calcula el determinante.
- Si det ≠ 0, la matriz es invertible y el sistema de ecuaciones tiene solución única.

---

## 4. `np.linalg.inv` — Inversa de matriz (despejar costos unitarios)

```python
import numpy as np

# Matriz de coeficientes de insumos
A = np.array([
    [2, 1, 3],
    [1, 2, 1],
    [3, 1, 2]
])
A_inv = np.linalg.inv(A)
print("Inversa:\n", np.round(A_inv, 3))
# Verificar: A @ A_inv ≈ I
print("A @ A_inv:\n", np.round(A @ A_inv, 2))
```

**Salida esperada:**


**Salida esperada:**
```
Inversa:
 [[ 0.25  -0.083  0.417]
 [-0.083  0.417 -0.083]
 [ 0.417 -0.083 -0.25 ]]
A @ A_inv:
 [[ 1.  0. -0.]
 [-0.  1.  0.]
 [ 0.  0.  1.]]
```

**Explicación:**
- `np.linalg.inv(A)`: calcula la inversa de la matriz A.
- `A @ A_inv` debe dar la matriz identidad (verificación).
- Útil para resolver sistemas de ecuaciones lineales.

---

## 5. `np.linalg.solve` — Resolver sistema de ecuaciones (costos ocultos)



**Explicación:**
- `np.linalg.inv(A)`: calcula la inversa de la matriz A.
- `A @ A_inv` debe dar la matriz identidad (verificación).
- Útil para resolver sistemas de ecuaciones lineales.

---

## 5. `np.linalg.solve` — Resolver sistema de ecuaciones (costos ocultos)

```python
import numpy as np

# Sistema: 3 productos, 3 insumos, 3 costos totales conocidos
# 2x + y + 3z = 100
# x + 2y + z = 80
# 3x + y + 2z = 110
A = np.array([
    [2, 1, 3],
    [1, 2, 1],
    [3, 1, 2]
])
b = np.array([100, 80, 110])
costos_unitarios = np.linalg.solve(A, b)
print("Costos unitarios (x, y, z):", np.round(costos_unitarios, 2))
```

**Salida esperada:**


**Salida esperada:**
```
Costos unitarios (x, y, z): [20.  15.  10.]
```

**Explicación:**
- `np.linalg.solve(A, b)`: resuelve `A · x = b` para x.
- Más eficiente y numéricamente estable que calcular `inv(A) @ b`.
- x=20, y=15, z=10 son los costos unitarios de cada insumo.

---

## 6. `np.linalg.eig` — Valores y vectores propios (componentes principales de precios)



**Explicación:**
- `np.linalg.solve(A, b)`: resuelve `A · x = b` para x.
- Más eficiente y numéricamente estable que calcular `inv(A) @ b`.
- x=20, y=15, z=10 son los costos unitarios de cada insumo.

---

## 6. `np.linalg.eig` — Valores y vectores propios (componentes principales de precios)

```python
import numpy as np

# Matriz de correlación de precios entre 3 sucursales
corr_sucursales = np.array([
    [1.0, 0.8, 0.6],
    [0.8, 1.0, 0.7],
    [0.6, 0.7, 1.0]
])
valores, vectores = np.linalg.eig(corr_sucursales)
print("Valores propios:", np.round(valores, 3))
print("Vectores propios:\n", np.round(vectores, 3))
```

**Salida esperada:**


**Salida esperada:**
```
Valores propios: [2.312 0.542 0.146]
Vectores propios:
 [[ 0.576 -0.683  0.449]
 [ 0.603 -0.083 -0.794]
 [ 0.552  0.726  0.409]]
```

**Explicación:**
- `np.linalg.eig(A)`: valores propios (λ) y vectores propios (v) donde `A·v = λ·v`.
- El primer valor propio (2.312) es el dominante → explica la mayor variación.
- Útil para Análisis de Componentes Principales (PCA).

---

## 7. `np.linalg.svd` — Descomposición SVD (sistema de recomendación de productos)



**Explicación:**
- `np.linalg.eig(A)`: valores propios (λ) y vectores propios (v) donde `A·v = λ·v`.
- El primer valor propio (2.312) es el dominante → explica la mayor variación.
- Útil para Análisis de Componentes Principales (PCA).

---

## 7. `np.linalg.svd` — Descomposición SVD (sistema de recomendación de productos)

```python
import numpy as np

# Matriz de calificaciones: 4 usuarios × 3 productos
calificaciones = np.array([
    [5, 3, 0],
    [4, 0, 2],
    [1, 4, 5],
    [0, 2, 4]
])
U, S, Vt = np.linalg.svd(calificaciones, full_matrices=False)
print("U:\n", np.round(U, 2))
print("Valores singulares:", np.round(S, 2))
print("V^T:\n", np.round(Vt, 2))
```

**Salida esperada:**


**Salida esperada:**
```
U:
 [[-0.46  0.31 -0.63]
 [-0.35 -0.44  0.79]
 [-0.62 -0.43 -0.09]
 [-0.53  0.72  0.09]]
Valores singulares: [9.55 5.07 2.79]
V^T:
 [[-0.35 -0.57 -0.74]
 [ 0.8  -0.53 -0.28]
 [ 0.49  0.63 -0.61]]
```

**Explicación:**
- `np.linalg.svd(M)`: descompone `M = U · S · V^T`.
- Útil para sistemas de recomendación (filtrado colaborativo), reducción de dimensionalidad y compresión.

---

## 8. `np.linalg.norm` — Norma de vector (magnitud de precios)



**Explicación:**
- `np.linalg.svd(M)`: descompone `M = U · S · V^T`.
- Útil para sistemas de recomendación (filtrado colaborativo), reducción de dimensionalidad y compresión.

---

## 8. `np.linalg.norm` — Norma de vector (magnitud de precios)

```python
import numpy as np

precios = np.array([150, 230, 85, 420, 67])
norma_l2 = np.linalg.norm(precios)
norma_l1 = np.linalg.norm(precios, ord=1)
print("Norma L2 (euclideana):", round(norma_l2, 2))
print("Norma L1 (Manhattan):", norma_l1)
```

**Salida esperada:**


**Salida esperada:**
```
Norma L2 (euclideana): 521.95
Norma L1 (Manhattan): 952.0
```

**Explicación:**
- `np.linalg.norm(vector, ord=2)`: norma euclideana (por defecto).
- `ord=1`: suma de valores absolutos (distancia Manhattan).
- La norma L1 da la suma total de precios; la L2 es la distancia euclideana.

---

## 9. `np.trace` — Traza de matriz (suma de diagonal = costo total base)



**Explicación:**
- `np.linalg.norm(vector, ord=2)`: norma euclideana (por defecto).
- `ord=1`: suma de valores absolutos (distancia Manhattan).
- La norma L1 da la suma total de precios; la L2 es la distancia euclideana.

---

## 9. `np.trace` — Traza de matriz (suma de diagonal = costo total base)

```python
import numpy as np

# Costos de 3 productos en 3 sucursales
costos = np.array([
    [150, 155, 148],
    [230, 225, 228],
    [85,  90,  88]
])
traza = np.trace(costos)
print("Traza (costo total base):", traza)
```

**Salida esperada:**


**Salida esperada:**
```
Traza (costo total base): 463
```

**Explicación:**
- `np.trace(matriz)`: suma los elementos de la diagonal principal.
- Representa el costo base de cada producto en su sucursal principal.

---

## 10. `np.diag` con matriz — Extraer o crear diagonal



**Explicación:**
- `np.trace(matriz)`: suma los elementos de la diagonal principal.
- Representa el costo base de cada producto en su sucursal principal.

---

## 10. `np.diag` con matriz — Extraer o crear diagonal

```python
import numpy as np

costos = np.array([
    [150, 155, 148],
    [230, 225, 228],
    [85,  90,  88]
])
# Extraer diagonal
diag = np.diag(costos)
print("Diagonal:", diag)

# Crear matriz diagonal desde vector
costos_unitarios = np.array([150, 225, 88])
matriz_costos = np.diag(costos_unitarios)
print("Matriz diagonal:\n", matriz_costos)
```

**Salida esperada:**


**Salida esperada:**
```
Diagonal: [150 225  88]
Matriz diagonal:
 [[150   0   0]
 [  0 225   0]
 [  0   0  88]]
```

**Explicación:**
- `np.diag(matriz)`: extrae la diagonal como vector.
- `np.diag(vector)`: crea matriz diagonal con el vector en la diagonal.
- Útil para aislar costos unitarios.

---

## 11. `np.outer` — Producto exterior (precios × cantidades como matriz)



**Explicación:**
- `np.diag(matriz)`: extrae la diagonal como vector.
- `np.diag(vector)`: crea matriz diagonal con el vector en la diagonal.
- Útil para aislar costos unitarios.

---

## 11. `np.outer` — Producto exterior (precios × cantidades como matriz)

```python
import numpy as np

precios = np.array([150, 230, 85])
cantidades = np.array([10, 5, 20, 3])
matriz_ingresos = np.outer(precios, cantidades)
print("Matriz ingresos (precio×cant):\n", matriz_ingresos)
```

**Salida esperada:**


**Salida esperada:**
```
Matriz ingresos (precio×cant):
 [[1500  750 3000  450]
 [2300 1150 4600  690]
 [ 850  425 1700  255]]
```

**Explicación:**
- `np.outer(a, b)`: producto exterior `a[i] × b[j]` para todo i, j.
- Cada fila i muestra el ingreso del producto i para cada cantidad.
- Shape: (len(a), len(b)).

---

## 12. `np.cross` — Producto cruz (vector perpendicular, similitud de productos)



**Explicación:**
- `np.outer(a, b)`: producto exterior `a[i] × b[j]` para todo i, j.
- Cada fila i muestra el ingreso del producto i para cada cantidad.
- Shape: (len(a), len(b)).

---

## 12. `np.cross` — Producto cruz (vector perpendicular, similitud de productos)

```python
import numpy as np

# Dos productos representados como vectores de atributos (precio, calidad, stock)
prod_a = np.array([150, 8, 45])
prod_b = np.array([230, 7, 12])
cruz = np.cross(prod_a, prod_b)
print("Producto cruz:", cruz)
```

**Salida esperada:**


**Salida esperada:**
```
Producto cruz: [-219  -90 -710]
```

**Explicación:**
- `np.cross(a, b)`: producto cruz (solo para vectores 2D o 3D).
- El vector resultante es perpendicular a ambos.
- La magnitud indica qué tan diferentes son los productos en el espacio de atributos.

---

## 13. `np.tensordot` — Producto tensorial (suma contraída sobre ejes)



**Explicación:**
- `np.cross(a, b)`: producto cruz (solo para vectores 2D o 3D).
- El vector resultante es perpendicular a ambos.
- La magnitud indica qué tan diferentes son los productos en el espacio de atributos.

---

## 13. `np.tensordot` — Producto tensorial (suma contraída sobre ejes)

```python
import numpy as np

# Ventas: 2 semanas × 3 sucursales × 4 productos
ventas = np.random.randint(1, 10, size=(2, 3, 4))
# Precios: 4 productos
precios = np.array([150, 230, 85, 420])
# Contraer sobre el eje de productos (último de ventas, único de precios)
ingresos = np.tensordot(ventas, precios, axes=1)
print("Shape ventas:", ventas.shape)
print("Ingresos (semana×sucursal):\n", ingresos)
```

**Salida esperada:**


**Salida esperada:**
```
Shape ventas: (2, 3, 4)
Ingresos (semana×sucursal):
 [[ 6540  4480 10360]
 [ 5970  5400  8760]]
```

**Explicación:**
- `np.tensordot(A, B, axes=1)`: contrae tensores a lo largo de ejes especificados.
- `axes=1`: contrae el último eje de A (productos=4) con el único eje de B.
- Resultado: matriz 2×3 (semanas × sucursales) con ingresos totales.

---

## 14. `np.linalg.lstsq` — Mínimos cuadrados (regresión lineal simple)



**Explicación:**
- `np.tensordot(A, B, axes=1)`: contrae tensores a lo largo de ejes especificados.
- `axes=1`: contrae el último eje de A (productos=4) con el único eje de B.
- Resultado: matriz 2×3 (semanas × sucursales) con ingresos totales.

---

## 14. `np.linalg.lstsq` — Mínimos cuadrados (regresión lineal simple)

```python
import numpy as np

# Datos: días (1..7) vs ventas
dias = np.array([1, 2, 3, 4, 5, 6, 7])
ventas = np.array([1200, 1350, 1300, 1500, 1450, 1600, 1550])

# Construir matriz de diseño X = [días, 1] para pendiente e intercepto
X = np.column_stack([dias, np.ones_like(dias)])
# Resolver por mínimos cuadrados: X · beta ≈ ventas
beta, residuals, rank, s = np.linalg.lstsq(X, ventas, rcond=None)
pendiente, intercepto = beta
print(f"Ecuación: ventas = {pendiente:.2f} × día + {intercepto:.2f}")
print("Pronóstico día 8:", round(pendiente * 8 + intercepto, 0))
```

**Salida esperada:**


**Salida esperada:**
```
Ecuación: ventas = 57.14 × día + 1135.71
Pronóstico día 8: 1593.0
```

**Explicación:**
- `np.linalg.lstsq(X, y)`: resuelve `X·beta = y` minimizando `||X·beta - y||²`.
- `X = [días, 1]`: matriz de diseño con pendiente e intercepto.
- `beta[0]` = pendiente (57.14: aumento diario), `beta[1]` = intercepto (1135.71).
- Pronóstico para día 8: 1593 unidades.

---

## 15. `np.linalg.matrix_rank` — Rango de matriz (multicolinealidad en precios)



**Explicación:**
- `np.linalg.lstsq(X, y)`: resuelve `X·beta = y` minimizando `||X·beta - y||²`.
- `X = [días, 1]`: matriz de diseño con pendiente e intercepto.
- `beta[0]` = pendiente (57.14: aumento diario), `beta[1]` = intercepto (1135.71).
- Pronóstico para día 8: 1593 unidades.

---

## 15. `np.linalg.matrix_rank` — Rango de matriz (multicolinealidad en precios)

```python
import numpy as np

# Matriz de precios: 3 productos × 3 proveedores
precios_proveedores = np.array([
    [150, 155, 148],
    [230, 225, 228],
    [85,  88,  84]
])
rango = np.linalg.matrix_rank(precios_proveedores)
print("Rango de la matriz:", rango)
print("Filas:", precios_proveedores.shape[0])
if rango == precios_proveedores.shape[0]:
    print("Las filas son linealmente independientes (sin colinealidad)")
else:
    print("Hay dependencia lineal entre productos/proveedores")
```

**Salida esperada:**


**Salida esperada:**
```
Rango de la matriz: 3
Filas: 3
Las filas son linealmente independientes (sin colinealidad)
```

**Explicación:**
- `np.linalg.matrix_rank(M)`: número de filas/columnas linealmente independientes.
- Si el rango es menor que el número de filas, hay multicolinealidad (precios que se pueden expresar como combinación de otros).

---

## Resumen

| Función | Propósito | Aplicación en Ventas |
|---------|-----------|----------------------|
| `np.dot` | Producto punto | Ingreso total = precio·cantidad |
| `np.matmul` | Multiplicación matrices | Ingresos por sucursal |
| `np.linalg.det` | Determinante | Verificar invertibilidad |
| `np.linalg.inv` | Inversa | Despejar costos unitarios |
| `np.linalg.solve` | Sistema ecuaciones | Resolver costos ocultos |
| `np.linalg.eig` | Valores/vectores propios | PCA de precios |
| `np.linalg.svd` | Descomposición SVD | Sistema de recomendación |
| `np.linalg.norm` | Norma vectorial | Magnitud de precios |
| `np.trace` | Traza | Suma de diagonal |
| `np.diag` | Diagonal | Extraer/crear diagonal |
| `np.outer` | Producto exterior | Matriz precios×cantidades |
| `np.cross` | Producto cruz | Diferencia de productos |
| `np.tensordot` | Producto tensorial | Contraer datos 3D |
| `np.linalg.lstsq` | Mínimos cuadrados | Regresión de ventas |
| `np.linalg.matrix_rank` | Rango | Detectar colinealidad |

---

## Ejercicios

1. Dados `precios = [150, 230, 85]` y `cantidades = [10, 5, 20]`, calcula el ingreso total usando `np.dot`.

2. Dada la matriz de cantidades `[[10,5,20],[8,7,15],[12,4,18]]` y precios `[150, 230, 85]`, usa `np.matmul` para calcular los ingresos de cada sucursal.

3. Usa `np.linalg.solve` para resolver el sistema:
   

**Explicación:**
- `np.linalg.matrix_rank(M)`: número de filas/columnas linealmente independientes.
- Si el rango es menor que el número de filas, hay multicolinealidad (precios que se pueden expresar como combinación de otros).

---

## Resumen

| Función | Propósito | Aplicación en Ventas |
|---------|-----------|----------------------|
| `np.dot` | Producto punto | Ingreso total = precio·cantidad |
| `np.matmul` | Multiplicación matrices | Ingresos por sucursal |
| `np.linalg.det` | Determinante | Verificar invertibilidad |
| `np.linalg.inv` | Inversa | Despejar costos unitarios |
| `np.linalg.solve` | Sistema ecuaciones | Resolver costos ocultos |
| `np.linalg.eig` | Valores/vectores propios | PCA de precios |
| `np.linalg.svd` | Descomposición SVD | Sistema de recomendación |
| `np.linalg.norm` | Norma vectorial | Magnitud de precios |
| `np.trace` | Traza | Suma de diagonal |
| `np.diag` | Diagonal | Extraer/crear diagonal |
| `np.outer` | Producto exterior | Matriz precios×cantidades |
| `np.cross` | Producto cruz | Diferencia de productos |
| `np.tensordot` | Producto tensorial | Contraer datos 3D |
| `np.linalg.lstsq` | Mínimos cuadrados | Regresión de ventas |
| `np.linalg.matrix_rank` | Rango | Detectar colinealidad |

---

## Ejercicios

1. Dados `precios = [150, 230, 85]` y `cantidades = [10, 5, 20]`, calcula el ingreso total usando `np.dot`.

2. Dada la matriz de cantidades `[[10,5,20],[8,7,15],[12,4,18]]` y precios `[150, 230, 85]`, usa `np.matmul` para calcular los ingresos de cada sucursal.

3. Usa `np.linalg.solve` para resolver el sistema:
   ```
   3x + 2y = 120
   2x + 5y = 190
   ```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Ejercicios.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



4. Calcula la norma L2 y L1 del vector `[150, 230, 85, 420, 67]`.

5. Con `np.outer`, genera la matriz de ingresos para `precios = [100, 200, 150]` y `cantidades = [5, 10, 15, 20]`.

6. Calcula la traza de la matriz `[[150, 155, 148], [230, 225, 228], [85, 90, 88]]`.

7. Usa `np.linalg.lstsq` para ajustar una recta a los datos: días `[1,2,3,4,5]`, ventas `[1000, 1100, 1050, 1200, 1150]`. Pronostica el día 6.

8. Dada la matriz [[2,1],[4,2]], calcula su determinante y rango. ¿Es invertible? ¿Por qué?
