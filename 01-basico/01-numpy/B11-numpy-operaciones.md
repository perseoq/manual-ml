# B11 – Operaciones Aritméticas en NumPy

## Introducción

NumPy permite realizar operaciones aritméticas elemento por elemento y broadcasting, lo cual es esencial para cálculos en ventas: suma de ingresos e impuestos, multiplicación de precio por cantidad, cálculo de márgenes, etc. Las operaciones vectorizadas eliminan la necesidad de bucles y son órdenes de magnitud más rápidas.

---

## 1. Suma — Ingresos + impuestos

```python
import numpy as np

ingresos = np.array([1500, 2300, 850, 4200, 670])
impuestos = np.array([300, 460, 170, 840, 134])
total_factura = ingresos + impuestos
print("Total factura:", total_factura)
```

**Salida esperada:**
```
Total factura: [1800 2760 1020 5040  804]
```

**Explicación:**
- `ingresos + impuestos`: suma elemento a elemento.
- Cada posición corresponde al mismo producto o transacción.

---

## 2. Resta — Precio - costo (margen bruto)

```python
import numpy as np

precios = np.array([150.0, 230.0, 85.5, 420.0, 67.0])
costos = np.array([90.0, 140.0, 50.0, 280.0, 40.0])
margen = precios - costos
print("Margen bruto:", margen)
```

**Salida esperada:**
```
Margen bruto: [ 60.  90.  35.5 140.  27. ]
```

**Explicación:**
- `precios - costos`: calcula la ganancia bruta por producto.

---

## 3. Multiplicación — Precio × cantidad (ingreso total por producto)

```python
import numpy as np

precios = np.array([150.0, 230.0, 85.5, 420.0, 67.0])
cantidades = np.array([10, 5, 20, 3, 15])
ingreso_total = precios * cantidades
print("Ingreso por producto:", ingreso_total)
print("Ingreso total:", ingreso_total.sum())
```

**Salida esperada:**
```
Ingreso por producto: [1500. 1150. 1710. 1260. 1005.]
Ingreso total: 6625.0
```

**Explicación:**
- `precios * cantidades`: multiplica elemento a elemento.
- `ingreso_total.sum()`: suma todos los ingresos individuales.

---

## 4. División — Costo / precio (ratio de costo)

```python
import numpy as np

costos = np.array([90.0, 140.0, 50.0, 280.0, 40.0])
precios = np.array([150.0, 230.0, 85.5, 420.0, 67.0])
ratio_costo = costos / precios
print("Ratio costo/precio:", ratio_costo)
```

**Salida esperada:**
```
Ratio costo/precio: [0.6  0.6087 0.5848 0.6667 0.597 ]
```

**Explicación:**
- `costos / precios`: qué fracción del precio representa el costo.
- Valores < 1 indican ganancia positiva.

---

## 5. Broadcasting — Descuento porcentual a toda una matriz

```python
import numpy as np

precios = np.array([150.0, 230.0, 85.5, 420.0, 67.0])
descuento = 0.15  # 15% de descuento
precios_oferta = precios * (1 - descuento)
print("Precios con 15% desc.:", precios_oferta)
```

**Salida esperada:**
```
Precios con 15% desc.: [127.5  195.5   72.675 357.    56.95 ]
```

**Explicación:**
- Broadcasting: el escalar `0.15` se "expande" automáticamente a todo el array.
- `precios * (1 - descuento)` aplica el descuento a todos los elementos.

---

## 6. Broadcasting 2D + 1D — Agregar costo de envío por sucursal

```python
import numpy as np

# 3 sucursales × 4 productos
costos_base = np.array([
    [90, 140, 50, 280],
    [95, 145, 48, 275],
    [88, 138, 52, 285]
])
envio_sucursal = np.array([10, 15, 12])  # costo envío por sucursal
costo_total = costos_base + envio_sucursal.reshape(3, 1)
print("Costo total c/envío:\n", costo_total)
```

**Salida esperada:**
```
Costo total c/envío:
 [[100 150  60 290]
 [110 160  63 290]
 [100 150  64 297]]
```

**Explicación:**
- `envio_sucursal.reshape(3, 1)`: convierte el vector 1D en columna 3×1 para broadcasting.
- NumPy expande automáticamente la dimensión para alinearse con la matriz 3×4.

---

## 7. `np.sqrt` — Raíz cuadrada (desviación estándar manual)

```python
import numpy as np

varianzas_precios = np.array([2500, 3600, 1225, 4900, 2025])
desv_estandar = np.sqrt(varianzas_precios)
print("Desviación estándar:", desv_estandar)
```

**Salida esperada:**
```
Desviación estándar: [50. 60. 35. 70. 45.]
```

**Explicación:**
- `np.sqrt`: calcula la raíz cuadrada elemento a elemento.
- Útil para convertir varianza a desviación estándar.

---

## 8. `np.power` — Potencia (crecimiento compuesto de ventas)

```python
import numpy as np

crecimiento_mensual = np.array([1.05, 1.03, 1.08, 1.02, 1.06])
# Crecimiento acumulado después de 6 meses
crecimiento_6m = np.power(crecimiento_mensual, 6)
print("Factor crecimiento 6 meses:", crecimiento_6m)
```

**Salida esperada:**
```
Factor crecimiento 6 meses: [1.3401 1.1941 1.5869 1.1262 1.4185]
```

**Explicación:**
- `np.power(base, exponente)`: eleva cada elemento a la potencia indicada.
- Modela crecimiento compuesto: `(1 + tasa)^n`.

---

## 9. `np.exp` — Exponencial (modelo de demanda)

```python
import numpy as np

descuentos = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
factor_demanda = np.exp(descuentos * 2)
print("Factor demanda esperado:", factor_demanda)
```

**Salida esperada:**
```
Factor demanda esperado: [1.2214 1.4918 1.8221 2.2255 2.7183]
```

**Explicación:**
- `np.exp(x)`: calcula `e^x` para cada elemento.
- Modela crecimiento exponencial de la demanda ante descuentos.

---

## 10. `np.log` — Logaritmo natural (elasticidad precio)

```python
import numpy as np

precios = np.array([150, 230, 85, 420, 67])
log_precios = np.log(precios)
print("Log precios:", log_precios)
```

**Salida esperada:**
```
Log precios: [5.0106 5.4381 4.4427 6.0403 4.2047]
```

**Explicación:**
- `np.log`: logaritmo natural (base e).
- Útil para transformar datos con distribución asimétrica y calcular elasticidades.

---

## 11. `np.clip` — Recortar valores (precio mínimo y máximo)

```python
import numpy as np

precios = np.array([150.0, -20.0, 85.5, 420.0, 67.0, -5.0])
precios_limpios = np.clip(precios, 0, 400)
print("Precios limpios:", precios_limpios)
```

**Salida esperada:**
```
Precios limpios: [150.   0.  85.5 400.  67.   0. ]
```

**Explicación:**
- `np.clip(array, min, max)`: recorta valores menores que `min` a `min` y mayores que `max` a `max`.
- Elimina precios negativos (errores de datos) y limita valores atípicos.

---

## 12. `np.mod` — Módulo (agrupación en lotes)

```python
import numpy as np

transacciones = np.array([45, 78, 23, 91, 56, 34, 67, 89])
# Agrupar en lotes de 30 unidades
lotes = np.mod(transacciones, 30)
print("Sobrantes por lote:", lotes)
```

**Salida esperada:**
```
Sobrantes por lote: [15 18 23  1 26  4  7 29]
```

**Explicación:**
- `np.mod(dividendo, divisor)`: resto de la división entera.
- Útil para calcular sobrantes al empaquetar en lotes.

---

## 13. `np.floor` — Redondear hacia abajo (precios sin decimales)

```python
import numpy as np

precios = np.array([149.99, 229.50, 84.99, 419.49, 66.99])
precios_piso = np.floor(precios)
print("Precios redondeados abajo:", precios_piso)
```

**Salida esperada:**
```
Precios redondeados abajo: [149. 229.  84. 419.  66.]
```

**Explicación:**
- `np.floor`: redondea al entero inferior más cercano.
- Útil para estrategias de precios psicológicos (99 → 99, 149.99 → 149).

---

## 14. `np.ceil` — Redondear hacia arriba (precios al alza)

```python
import numpy as np

costos = np.array([90.10, 140.50, 50.01, 280.70, 40.30])
precios_sugeridos = np.ceil(costos * 1.3)
print("Precio sugerido (ceil):", precios_sugeridos)
```

**Salida esperada:**
```
Precio sugerido (ceil): [118. 183.  66. 366.  53.]
```

**Explicación:**
- `np.ceil`: redondea al entero superior más cercano.
- Útil para fijar precios finales redondeados al alza.

---

## 15. `np.prod` — Producto acumulado (crecimiento total)

```python
import numpy as np

factores_crecimiento = np.array([1.1, 1.05, 0.98, 1.12, 1.03])
crecimiento_total = np.prod(factores_crecimiento)
print("Crecimiento acumulado total:", crecimiento_total)
```

**Salida esperada:**
```
Crecimiento acumulado total: 1.3045
```

**Explicación:**
- `np.prod(array)`: producto de todos los elementos.
- Multiplica factores de crecimiento secuenciales (ej. inflación + aumento real).

---

## 16. `np.cumsum` — Suma acumulada (ventas acumuladas)

```python
import numpy as np

ventas_diarias = np.array([1200, 890, 1500, 2100, 980, 3100, 1800])
ventas_acumuladas = np.cumsum(ventas_diarias)
print("Ventas acumuladas:", ventas_acumuladas)
```

**Salida esperada:**
```
Ventas acumuladas: [1200 2090 3590 5690 6670 9770 11570]
```

**Explicación:**
- `np.cumsum`: suma acumulativa.
- Cada posición i contiene la suma desde el inicio hasta i.
- Útil para ver ventas acumuladas en el mes.

---

## 17. `np.diff` — Diferencias consecutivas (cambio diario)

```python
import numpy as np

ventas = np.array([1200, 890, 1500, 2100, 980, 3100, 1800])
cambio_diario = np.diff(ventas)
print("Cambio diario en ventas:", cambio_diario)
```

**Salida esperada:**
```
Cambio diario en ventas: [-310  610  600 -1120 2120 -1300]
```

**Explicación:**
- `np.diff(ventas)`: calcula `ventas[i+1] - ventas[i]` para cada i.
- El resultado tiene n-1 elementos.
- Valores positivos = aumento, negativos = disminución.

---

## 18. `np.floor` + `np.ceil` combinados — Precios terminados en 9

```python
import numpy as np

precios = np.array([150.0, 230.0, 85.5, 420.0, 67.0, 310.0])
# Redondear a entero y restar 0.01 para precio psicológico
precios_psicologicos = np.floor(precios) - 0.01
print("Precios psicológicos:", precios_psicologicos)
```

**Salida esperada:**
```
Precios psicológicos: [149.99 229.99  84.99 419.99  66.99 309.99]
```

**Explicación:**
- Estrategia de precios psicológicos: `$149.99` en lugar de `$150`.
- Se usa `np.floor` para bajar al entero y se resta 1 centavo.

---

## Resumen

| Operación | Sintaxis | Aplicación en Ventas |
|-----------|----------|----------------------|
| Suma | `a + b` | Ingresos + impuestos |
| Resta | `a - b` | Precio - costo |
| Multiplicación | `a * b` | Precio × cantidad |
| División | `a / b` | Costo / precio |
| Broadcasting | escalar + array | Descuento porcentual |
| `np.sqrt` | Raíz cuadrada | Desviación estándar |
| `np.power` | Potencia | Crecimiento compuesto |
| `np.exp` | Exponencial | Modelo de demanda |
| `np.log` | Logaritmo | Elasticidad precio |
| `np.clip` | Recortar valores | Limpiar precios atípicos |
| `np.mod` | Módulo | Agrupar en lotes |
| `np.floor` | Redondear abajo | Precios psicológicos |
| `np.ceil` | Redondear arriba | Precios sugeridos |
| `np.prod` | Producto total | Crecimiento total |
| `np.cumsum` | Suma acumulada | Ventas acumuladas |
| `np.diff` | Diferencias | Cambio diario |

---

## Ejercicios

1. Dados `precios = [150, 230, 85, 420, 67]` y `cantidades = [10, 5, 20, 3, 15]`, calcula el ingreso total (precio × cantidad) y la suma de todos los ingresos.

2. Calcula el margen bruto (precio - costo) para `precios = [150, 230, 85, 420, 67]` y `costos = [90, 140, 50, 280, 40]`.

3. Aplica un descuento del 20% a todos los precios del array `[150, 230, 85, 420, 67]` usando broadcasting.

4. Usa `np.clip` para limpiar el array `[150, -30, 85, 500, 67, -10]` con límites entre 0 y 400.

5. Dado `ventas = [1200, 890, 1500, 2100, 980]`, calcula las ventas acumuladas con `np.cumsum` y el cambio diario con `np.diff`.

6. Con `np.power`, calcula el crecimiento de ventas después de 12 meses si la tasa mensual es `[1.03, 1.02, 1.04, 1.01, 1.05]`.

7. Convierte los precios `[149.99, 229.50, 84.99, 419.49, 66.99]` a precios psicológicos terminados en 0.99 usando `np.floor`.

8. Calcula el ratio costo/precio (costo / precio) para `costos = [90, 140, 50, 280, 40]` y `precios = [150, 230, 85, 420, 67]`.
