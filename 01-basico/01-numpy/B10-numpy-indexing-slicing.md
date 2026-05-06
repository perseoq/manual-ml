# B10 – Indexing y Slicing en NumPy

## Introducción

El indexing y slicing en NumPy permite acceder y modificar subconjuntos de datos de forma eficiente. En ventas e inventarios, esto es fundamental para filtrar productos caros, días con bajas ventas, sucursales con stock crítico, etc. NumPy ofrece indexación básica, por rebanadas (slices), fancy indexing, boolean indexing y funciones especializadas.

---

## 1. Index básico — Acceso por posición

```python
import numpy as np

# Ventas diarias de lunes a domingo
ventas = np.array([1200, 890, 1500, 2100, 980, 3100, 1800])
print("Ventas lunes:", ventas[0])
print("Ventas viernes:", ventas[4])
```

**Salida esperada:**
```
Ventas lunes: 1200
Ventas viernes: 980
```

**Explicación:**
- `ventas[0]`: accede al primer elemento (lunes, índice 0).
- `ventas[4]`: accede al quinto elemento (viernes, índice 4).
- Los índices comienzan en 0 como en Python estándar.

---

## 2. Slice 1D — Subvector por rangos

```python
import numpy as np

ventas = np.array([1200, 890, 1500, 2100, 980, 3100, 1800])
# Ventas de miércoles a viernes (índices 2 al 5, exclusivo)
print("Miércoles a viernes:", ventas[2:5])
# Primeros 3 días
print("Primeros 3 días:", ventas[:3])
# Últimos 2 días
print("Últimos 2 días:", ventas[-2:])
```

**Salida esperada:**
```
Miércoles a viernes: [1500 2100  980]
Primeros 3 días: [1200  890 1500]
Últimos 2 días: [3100 1800]
```

**Explicación:**
- `ventas[2:5]`: elementos desde índice 2 hasta 4 (5 exclusivo).
- `ventas[:3]`: desde el inicio hasta índice 2 (3 exclusivo).
- `ventas[-2:]`: últimos 2 elementos (usando índices negativos).

---

## 3. Slice 2D — Submatriz

```python
import numpy as np

# Ventas: 3 sucursales × 7 días
ventas_semana = np.array([
    [1200, 890, 1500, 2100, 980, 3100, 1800],
    [950, 1100, 1300, 1700, 880, 2500, 1600],
    [1500, 1400, 1200, 1900, 1050, 2800, 2000]
])
# Sucursales 0 y 1, días 3 a 5
print("Submatriz:\n", ventas_semana[0:2, 3:6])
```

**Salida esperada:**
```
Submatriz:
 [[2100  980 3100]
 [1700  880 2500]]
```

**Explicación:**
- `ventas_semana[0:2, 3:6]`: filas 0 y 1, columnas 3, 4, 5.
- Formato: `[filas, columnas]`.

---

## 4. Index con step — Cada n elementos

```python
import numpy as np

# Ventas de 30 días
ventas_mes = np.array([1200, 890, 1500, 2100, 980, 3100, 1800,
                       950, 1100, 1300, 1700, 880, 2500, 1600,
                       1500, 1400, 1200, 1900, 1050, 2800, 2000,
                       1100, 1300, 900, 2100, 1500, 1200, 1800, 1600, 1400])
# Cada 7 días (los domingos, índice 6, 13, 20, 27)
print("Ventas domingos:", ventas_mes[6::7])
```

**Salida esperada:**
```
Ventas domingos: [1800 1600 2000 1800]
```

**Explicación:**
- `ventas_mes[6::7]`: desde índice 6 hasta el final, step de 7.
- Selecciona el séptimo día de cada semana (domingo).

---

## 5. Índices negativos

```python
import numpy as np

precios = np.array([150.0, 230.0, 85.5, 420.0, 67.0, 310.0])
print("Último precio:", precios[-1])
print("Penúltimo:", precios[-2])
print("Tres últimos:", precios[-3:])
```

**Salida esperada:**
```
Último precio: 310.0
Penúltimo: 67.0
Tres últimos: [ 67.  310.  420.]
```

**Explicación:**
- `precios[-1]`: último elemento.
- `precios[-2]`: segundo desde el final.
- `precios[-3:]`: últimos 3 elementos.

---

## 6. Fancy indexing — Index con lista de posiciones

```python
import numpy as np

precios = np.array([150.0, 230.0, 85.5, 420.0, 67.0, 310.0])
# Seleccionar productos en posiciones 0, 2, 4 (1ro, 3ro, 5to)
seleccion = precios[[0, 2, 4]]
print("Precios seleccionados:", seleccion)
```

**Salida esperada:**
```
Precios seleccionados: [150.   85.5  67. ]
```

**Explicación:**
- `precios[[0, 2, 4]]`: pasa una lista de índices para seleccionar posiciones específicas.
- El resultado es un nuevo array con los elementos en esas posiciones.

---

## 7. Boolean indexing — Filtro por condición

```python
import numpy as np

precios = np.array([150.0, 230.0, 85.5, 420.0, 67.0, 310.0])
# Productos con precio mayor a 200
caros = precios[precios > 200]
print("Productos caros:", caros)
```

**Salida esperada:**
```
Productos caros: [230.  420.  310.]
```

**Explicación:**
- `precios > 200`: genera una máscara booleana `[False, True, False, True, False, True]`.
- `precios[mascara]`: filtra solo los elementos donde la máscara es `True`.

---

## 8. `np.where` — Índices de elementos que cumplen condición

```python
import numpy as np

stocks = np.array([45, 12, 8, 90, 3, 67, 0, 22])
# Posiciones donde stock < 10 (reordenar urgente)
indices_bajo = np.where(stocks < 10)
print("Índices con stock bajo:", indices_bajo)
print("Valores:", stocks[indices_bajo])
```

**Salida esperada:**
```
Índices con stock bajo: (array([2, 4, 6]),)
Valores: [8 3 0]
```

**Explicación:**
- `np.where(stocks < 10)`: devuelve una tupla de arrays con los índices donde la condición es verdadera.
- Se puede usar directamente como índice para obtener los valores.

---

## 9. `np.where` con 3 argumentos — Reemplazo condicional

```python
import numpy as np

precios = np.array([150.0, 230.0, 85.5, 420.0, 67.0, 310.0])
# Aplicar 10% descuento si precio > 200, si no mantener
precios_final = np.where(precios > 200, precios * 0.9, precios)
print("Precios con descuento:", precios_final)
```

**Salida esperada:**
```
Precios con descuento: [150.  207.   85.5 378.   67.  279. ]
```

**Explicación:**
- `np.where(condición, valor_si_true, valor_si_false)`: aplica descuento del 10% solo a productos caros.

---

## 10. `np.take` — Tomar elementos por índices

```python
import numpy as np

productos = np.array(["Leche", "Pan", "Huevos", "Arroz", "Frijoles"])
# Seleccionar productos para una promoción (índices 0, 2, 4)
promo = np.take(productos, [0, 2, 4])
print("Promoción:", promo)
```

**Salida esperada:**
```
Promoción: ['Leche' 'Huevos' 'Frijoles']
```

**Explicación:**
- `np.take(productos, [0, 2, 4])`: extrae elementos por sus índices, similar a fancy indexing.

---

## 11. `np.compress` — Filtrar con máscara booleana

```python
import numpy as np

ventas = np.array([1200, 890, 1500, 2100, 980, 3100, 1800])
# Días que superaron los 1500
buenos_dias = np.compress(ventas > 1500, ventas)
print("Ventas > 1500:", buenos_dias)
```

**Salida esperada:**
```
Ventas > 1500: [2100 3100 1800]
```

**Explicación:**
- `np.compress(condición, datos)`: equivalente a `datos[condición]`, filtra usando máscara booleana.

---

## 12. Selección de diagonal

```python
import numpy as np

# Costo de 3 productos en 3 sucursales (matriz 3×3)
costos = np.array([
    [150, 155, 148],
    [230, 225, 228],
    [85,  90,  88]
])
diag_principal = np.diag(costos)
print("Costo base por producto:", diag_principal)
```

**Salida esperada:**
```
Costo base por producto: [150 225  88]
```

**Explicación:**
- `np.diag(costos)`: extrae la diagonal principal de la matriz.
- Cada valor representa el costo base de cada producto en su sucursal principal.

---

## 13. `np.put` — Modificar elementos por índices

```python
import numpy as np

stocks = np.array([45, 12, 8, 90, 3, 67, 0, 22])
# Reponer stock en posiciones 2, 4, 6 con 50 unidades cada una
np.put(stocks, [2, 4, 6], [50, 50, 50])
print("Stock después de reposición:", stocks)
```

**Salida esperada:**
```
Stock después de reposición: [45 12 50 90 50 67 50 22]
```

**Explicación:**
- `np.put(array, índices, valores)`: reemplaza valores en las posiciones indicadas.
- Modifica el array original (in-place).

---

## 14. Índices múltiples en 2D — Seleccionar filas y columnas específicas

```python
import numpy as np

ventas_semana = np.array([
    [1200, 890, 1500, 2100, 980, 3100, 1800],
    [950, 1100, 1300, 1700, 880, 2500, 1600],
    [1500, 1400, 1200, 1900, 1050, 2800, 2000]
])
# Sucursales 0 y 2, días 1 y 4
seleccion = ventas_semana[[0, 2]][:, [1, 4]]
print("Selección:\n", seleccion)
```

**Salida esperada:**
```
Selección:
 [[ 890  980]
 [1400 1050]]
```

**Explicación:**
- `ventas_semana[[0, 2]]`: selecciona filas 0 y 2.
- `[:, [1, 4]]`: de esas filas, selecciona columnas 1 y 4.

---

## 15. Máscaras booleanas compuestas

```python
import numpy as np

precios = np.array([150.0, 230.0, 85.5, 420.0, 67.0, 310.0])
stocks = np.array([45, 12, 8, 90, 3, 67])
# Productos con precio entre 100 y 300 Y stock > 10
mascara = (precios > 100) & (precios < 300) & (stocks > 10)
print("Máscara:", mascara)
print("Productos a promover:", precios[mascara])
```

**Salida esperada:**
```
Máscara: [ True  True False False False False]
Productos a promover: [150. 230.]
```

**Explicación:**
- Se construye una máscara combinando múltiples condiciones con `&` (AND).
- Solo los productos que cumplen todas las condiciones son seleccionados.
- `|` (OR) y `~` (NOT) también están disponibles para composición lógica.

---

## Resumen

| Operación | Sintaxis | Uso en Ventas |
|-----------|----------|---------------|
| Index básico | `arr[i]` | Un día específico |
| Slice 1D | `arr[i:j]` | Rango de días |
| Slice 2D | `arr[f1:f2, c1:c2]` | Submatriz sucursales×días |
| Step | `arr[i:j:k]` | Cada n días |
| Índices negativos | `arr[-i]` | Últimos elementos |
| Fancy indexing | `arr[[i,j,k]]` | Productos específicos |
| Boolean indexing | `arr[cond]` | Filtrar por condición |
| `np.where` | `np.where(cond)` | Índices de stock bajo |
| `np.where` 3 args | `np.where(cond, x, y)` | Descuentos condicionales |
| `np.take` | `np.take(arr, idx)` | Promociones |
| `np.compress` | `np.compress(cond, arr)` | Filtrar días buenos |
| `np.diag` | `np.diag(mat)` | Costos base |
| `np.put` | `np.put(arr, idx, val)` | Reposición de stock |
| Máscaras compuestas | `(a>0)&(b<5)` | Filtros avanzados |

---

## Ejercicios

1. Dado el array `[340, 210, 560, 120, 780, 450, 300]`, selecciona los elementos con valor > 400 usando boolean indexing.

2. Con `np.where`, encuentra los índices donde el array `[45, 12, 8, 90, 3, 67]` tiene stock < 15.

3. Usando la matriz 3×7 de ventas semanales, extrae la submatriz de las sucursales 1 y 2 para los días 2, 3 y 4.

4. Aplica un descuento del 15% a los productos con precio > 200 usando `np.where` con 3 argumentos.

5. Dado `stocks = [45, 12, 8, 90, 3, 67]`, usa `np.put` para reemplazar los valores en posiciones 1, 3, 5 con 100.

6. Selecciona los elementos en posiciones pares (0, 2, 4, 6) de un array de 8 elementos usando fancy indexing.

7. Crea una máscara compuesta para filtrar productos donde precio > 100 Y stock < 50 de los arrays: `precios = [150, 230, 85, 420, 67]`, `stocks = [45, 12, 80, 90, 3]`.

8. Usa slicing con step para extraer cada 3er elemento del array `[10, 20, 30, 40, 50, 60, 70, 80, 90]`.
