# B13 – Manipulación de Arrays en NumPy

## Introducción

La manipulación de arrays permite reorganizar, combinar, ordenar y transformar datos para adaptarlos a las necesidades del análisis. En ventas e inventarios, estas operaciones son cruciales para cambiar la forma de los datos (por ejemplo, de una lista plana a una matriz sucursales×días), combinar tablas, ordenar productos por precio o stock, y eliminar duplicados.

---

## 1. `np.reshape` — Cambiar forma (lista plana a matriz sucursales×días)

```python
import numpy as np

# Ventas totales de 12 días en una lista plana
ventas_12dias = np.array([1200, 890, 1500, 2100, 980, 3100,
                          1800, 950, 1100, 1300, 1700, 880])
# Reorganizar en 3 sucursales × 4 días
ventas_matriz = ventas_12dias.reshape(3, 4)
print("Ventas por sucursal×días:\n", ventas_matriz)
```

**Salida esperada:**
```
Ventas por sucursal×días:
 [[1200  890 1500 2100]
 [ 980 3100 1800  950]
 [1100 1300 1700  880]]
```

**Explicación:**
- `ventas_12dias.reshape(3, 4)`: reorganiza los 12 elementos en una matriz 3×4.
- El número total de elementos debe mantenerse (3×4 = 12).
- El orden se mantiene por filas (row-major).

---

## 2. `np.resize` — Redimensionar (ajusta repitiendo o truncando)

```python
import numpy as np

precios = np.array([150, 230, 85, 420])
# Redimensionar a 6 elementos (rellena repitiendo)
precios_resized = np.resize(precios, 6)
print("Resize a 6:", precios_resized)

# Redimensionar a 3 elementos (trunca)
precios_trunc = np.resize(precios, 3)
print("Resize a 3:", precios_trunc)
```

**Salida esperada:**
```
Resize a 6: [150 230  85 420 150 230]
Resize a 3: [150 230  85]
```

**Explicación:**
- `np.resize(array, (nuevas_dimensiones))`: si el nuevo tamaño es mayor, repite el array cíclicamente; si es menor, trunca.
- A diferencia de `reshape`, no requiere que el número de elementos coincida.

---

## 3. `np.flatten` — Aplanar a 1D (copia)

```python
import numpy as np

ventas_semana = np.array([
    [1200, 890, 1500],
    [2100, 980, 3100],
    [1800, 950, 1100]
])
plano = ventas_semana.flatten()
print("Aplanado:", plano)
```

**Salida esperada:**
```
Aplanado: [1200  890 1500 2100  980 3100 1800  950 1100]
```

**Explicación:**
- `ventas_semana.flatten()`: convierte cualquier array multidimensional en 1D.
- Devuelve una **copia** del array (modificar no afecta al original).

---

## 4. `np.ravel` — Aplanar a 1D (vista)

```python
import numpy as np

ventas_semana = np.array([
    [1200, 890, 1500],
    [2100, 980, 3100],
    [1800, 950, 1100]
])
plano_vista = ventas_semana.ravel()
plano_vista[0] = 9999
print("Original modificado:\n", ventas_semana)
```

**Salida esperada:**
```
Original modificado:
 [[9999  890 1500]
 [2100  980 3100]
 [1800  950 1100]]
```

**Explicación:**
- `np.ravel()`: devuelve una **vista** del array (referencia), no una copia.
- Modificar `plano_vista` afecta al array original.
- Más eficiente en memoria que `flatten`.

---

## 5. `np.transpose` y `.T` — Transponer matriz

```python
import numpy as np

# 3 sucursales × 4 productos
ventas_suc_producto = np.array([
    [120, 230, 85, 310],
    [95, 180, 120, 250],
    [150, 200, 90, 280]
])
print("Original (3×4):\n", ventas_suc_producto)
print("Transpuesta (4×3):\n", ventas_suc_producto.T)
```

**Salida esperada:**
```
Original (3×4):
 [[120 230  85 310]
 [ 95 180 120 250]
 [150 200  90 280]]
Transpuesta (4×3):
 [[120  95 150]
 [230 180 200]
 [ 85 120  90]
 [310 250 280]]
```

**Explicación:**
- `.T` o `np.transpose()`: invierte filas y columnas.
- Las sucursales pasan a ser columnas y los productos filas.

---

## 6. `np.concatenate` — Concatenar arrays (ventas de dos semanas)

```python
import numpy as np

semana1 = np.array([1200, 890, 1500, 2100, 980, 3100, 1800])
semana2 = np.array([950, 1100, 1300, 1700, 880, 2500, 1600])
quincena = np.concatenate([semana1, semana2])
print("Ventas quincena:", quincena)
print("Días totales:", len(quincena))
```

**Salida esperada:**
```
Ventas quincena: [1200  890 1500 2100  980 3100 1800  950 1100 1300 1700  880 2500 1600]
Días totales: 14
```

**Explicación:**
- `np.concatenate([arr1, arr2])`: une arrays a lo largo de un eje existente.
- Los arrays deben tener las mismas dimensiones (excepto el eje de concatenación).

---

## 7. `np.stack` — Apilar arrays (nuevo eje)

```python
import numpy as np

semana1 = np.array([1200, 890, 1500, 2100, 980, 3100, 1800])
semana2 = np.array([950, 1100, 1300, 1700, 880, 2500, 1600])
# Apilar en un nuevo eje (2 semanas como 2 filas)
dos_semanas = np.stack([semana1, semana2], axis=0)
print("Stack (2×7):\n", dos_semanas)
```

**Salida esperada:**
```
Stack (2×7):
 [[1200  890 1500 2100  980 3100 1800]
 [ 950 1100 1300 1700  880 2500 1600]]
```

**Explicación:**
- `np.stack([arr1, arr2], axis=0)`: crea un nuevo eje (dimensión 0) para apilar los arrays.
- `axis=1` apilaría como columnas (7×2).

---

## 8. `np.hstack` — Apilar horizontalmente (columnas)

```python
import numpy as np

precios = np.array([150, 230, 85])
cantidades = np.array([10, 5, 20])
tabla = np.hstack([precios.reshape(-1, 1), cantidades.reshape(-1, 1)])
print("Tabla precio|cant:\n", tabla)
```

**Salida esperada:**
```
Tabla precio|cant:
 [[150  10]
 [230   5]
 [ 85  20]]
```

**Explicación:**
- `np.hstack([...])`: apila arrays horizontalmente (más columnas).
- Necesita que las filas coincidan; por eso se usa `reshape(-1, 1)` para convertir vectores 1D en columnas 2D.

---

## 9. `np.vstack` — Apilar verticalmente (filas)

```python
import numpy as np

suc1 = np.array([1200, 890, 1500])
suc2 = np.array([2100, 980, 3100])
suc3 = np.array([1800, 950, 1100])
todas_suc = np.vstack([suc1, suc2, suc3])
print("Todas sucursales:\n", todas_suc)
```

**Salida esperada:**
```
Todas sucursales:
 [[1200  890 1500]
 [2100  980 3100]
 [1800  950 1100]]
```

**Explicación:**
- `np.vstack([arr1, arr2, ...])`: apila arrays verticalmente (más filas).
- Los arrays deben tener el mismo número de columnas.

---

## 10. `np.split` — Dividir array en subarrays

```python
import numpy as np

ventas_30dias = np.array([1200, 890, 1500, 2100, 980, 3100, 1800,
                          950, 1100, 1300, 1700, 880, 2500, 1600])
# Dividir en 2 quincenas
quincenas = np.split(ventas_30dias, 2)
print("Quincena 1:", quincenas[0])
print("Quincena 2:", quincenas[1])
```

**Salida esperada:**
```
Quincena 1: [1200  890 1500 2100  980 3100 1800]
Quincena 2: [ 950 1100 1300 1700  880 2500 1600]
```

**Explicación:**
- `np.split(array, n)`: divide el array en `n` partes iguales.
- El array debe ser divisible exactamente.

---

## 11. `np.hsplit` — Dividir horizontalmente (columnas)

```python
import numpy as np

datos = np.array([
    [150, 10, 1200],
    [230, 5, 950],
    [85, 20, 1500]
])
precio, cant, ventas = np.hsplit(datos, 3)
print("Solo precios:\n", precio)
print("Solo cantidades:\n", cant)
```

**Salida esperada:**
```
Solo precios:
 [[150]
 [230]
 [ 85]]
Solo cantidades:
 [[10]
 [ 5]
 [20]]
```

**Explicación:**
- `np.hsplit(matriz, n)`: divide la matriz verticalmente (por columnas).
- Útil para separar columnas de una tabla de datos.

---

## 12. `np.sort` — Ordenar array (precios de menor a mayor)

```python
import numpy as np

precios = np.array([150, 230, 85, 420, 67, 310])
precios_ordenados = np.sort(precios)
print("Precios ordenados:", precios_ordenados)
```

**Salida esperada:**
```
Precios ordenados: [ 67  85 150 230 310 420]
```

**Explicación:**
- `np.sort(array)`: devuelve una copia ordenada de menor a mayor.
- No modifica el array original.

---

## 13. `np.argsort` — Índices que ordenarían el array

```python
import numpy as np

productos = np.array(["Leche", "Pan", "Huevos", "Arroz", "Frijoles"])
precios = np.array([150, 230, 85, 420, 67])
indices = np.argsort(precios)
print("Índices ordenados:", indices)
print("Productos por precio (menor a mayor):", productos[indices])
```

**Salida esperada:**
```
Índices ordenados: [4 2 0 1 3]
Productos por precio (menor a mayor): ['Frijoles' 'Huevos' 'Leche' 'Pan' 'Arroz']
```

**Explicación:**
- `np.argsort(array)`: devuelve los índices que ordenarían el array.
- Útil para ordenar múltiples arrays relacionados (ej. nombres + precios).

---

## 14. `np.lexsort` — Ordenación por múltiples criterios

```python
import numpy as np

productos = np.array(["Leche", "Pan", "Huevos", "Arroz", "Pan"])
precios = np.array([150, 230, 85, 420, 67])
stocks = np.array([45, 12, 8, 90, 30])
# Ordenar por producto (alfabético) y luego por precio
indices = np.lexsort((precios, productos))  # primero productos, luego precios
print("Orden lexsort:", indices)
print("Productos:", productos[indices])
print("Precios:", precios[indices])
```

**Salida esperada:**
```
Orden lexsort: [3 4 2 0 1]
Productos: ['Arroz' 'Pan' 'Huevos' 'Leche' 'Pan']
Precios: [420  67  85 150 230]
```

**Explicación:**
- `np.lexsort((precios, productos))`: ordena primero por `productos` (alfabético), luego por `precios`.
- El último argumento tiene prioridad de ordenación.

---

## 15. `np.flip`, `np.roll`, `np.unique` — Invertir, rotar y elementos únicos

```python
import numpy as np

ventas = np.array([1200, 890, 1500, 2100, 980, 3100, 1800])
print("Original:", ventas)
print("Invertido:", np.flip(ventas))
print("Rotado 2 posiciones:", np.roll(ventas, 2))

# IDs únicos de productos vendidos
ids_vendidos = np.array([101, 102, 101, 103, 102, 101, 104, 103])
print("IDs únicos:", np.unique(ids_vendidos))
```

**Salida esperada:**
```
Original: [1200  890 1500 2100  980 3100 1800]
Invertido: [1800 3100  980 2100 1500  890 1200]
Rotado 2 posiciones: [3100 1800 1200  890 1500 2100  980]
IDs únicos: [101 102 103 104]
```

**Explicación:**
- `np.flip(ventas)`: invierte el orden de los elementos.
- `np.roll(ventas, 2)`: desplaza elementos 2 posiciones a la derecha (los que salen por la derecha entran por la izquierda).
- `np.unique(ids_vendidos)`: devuelve los valores únicos ordenados (los productos que se han vendido al menos una vez).

---

## Resumen

| Función | Propósito | Aplicación en Ventas |
|---------|-----------|----------------------|
| `reshape` | Cambiar forma | Plano → matriz sucursales×días |
| `resize` | Redimensionar | Ajustar tamaño de array |
| `flatten` | Aplanar (copia) | Convertir matriz a 1D |
| `ravel` | Aplanar (vista) | Vista 1D eficiente |
| `transpose` / `.T` | Transponer | Filas ↔ columnas |
| `concatenate` | Concatenar | Unir semanas |
| `stack` | Apilar con nuevo eje | 2 arrays → 3D |
| `hstack` | Apilar horizontal | Agregar columna de datos |
| `vstack` | Apilar vertical | Agregar sucursal |
| `split` | Dividir | Separar en quincenas |
| `hsplit` | Dividir columnas | Separar precio, cant, ventas |
| `sort` | Ordenar | Precios ordenados |
| `argsort` | Índices de orden | Productos por precio |
| `lexsort` | Orden múltiple | Producto + precio |
| `flip` / `roll` / `unique` | Invertir / rotar / únicos | Análisis exploratorio |

---

## Ejercicios

1. Dado el array `[1200, 890, 1500, 2100, 980, 3100, 1800, 950, 1100]`, usa `reshape` para convertirlo en una matriz 3×3.

2. Con `vstack`, apila los vectores `[150, 230, 85]`, `[420, 67, 310]`, `[180, 95, 250]` en una matriz 3×3.

3. Con `hstack`, combina `precios = [150, 230, 85]` y `cantidades = [10, 5, 20]` (previamente convertidos a columnas) en una matriz 3×2.

4. Usa `np.sort` para ordenar `[150, 230, 85, 420, 67, 310]` y `np.argsort` para obtener los índices correspondientes.

5. Dados `productos = ["A","B","C","D","E","F"]` y `precios = [150, 230, 85, 420, 67, 310]`, usa `argsort` para listar los productos en orden ascendente de precio.

6. Divide el array `[1200, 890, 1500, 2100, 980, 3100, 1800, 950, 1100, 1300, 1700, 880]` en 3 partes iguales usando `np.split`.

7. Usa `np.unique` para encontrar los IDs únicos en `[101, 102, 101, 103, 102, 101, 104, 103, 105, 102]` y cuenta cuántos productos distintos se vendieron.

8. Dada la matriz 2D `[[120, 230, 85], [95, 180, 120], [150, 200, 90]]`, transpónela y muestra el resultado.
