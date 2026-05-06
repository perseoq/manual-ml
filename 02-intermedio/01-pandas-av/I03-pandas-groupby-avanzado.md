# I03 — Groupby Avanzado en Pandas

## 1. Introducción Teórica

El método `groupby` en pandas sigue el patrón **Split-Apply-Combine**:
1. **Split:** Dividir datos en grupos según una o más columnas
2. **Apply:** Aplicar una función a cada grupo
3. **Combine:** Combinar resultados en un nuevo objeto

### Operaciones avanzadas

- **agg(funcs_list):** Múltiples funciones de agregación simultáneas
- **transform(func):** Devuelve objeto del mismo tamaño que el original (útil para z-scores, % del total)
- **filter(func):** Filtra grupos completos según condición
- **apply(func):** Aplica función arbitraria a cada grupo (más flexible pero más lento)
- **nunique():** Cuenta valores únicos por grupo
- **first(), last(), nth(n):** Primer/último/n-ésimo elemento por grupo
- **rank(method):** Ranking dentro del grupo (average, min, max, first, dense)
- **cumcount():** Enumeración secuencial dentro del grupo
- **ngroup():** Etiqueta numérica de grupo
- **pipe():** Encadenar operaciones groupby
- **Named aggregation:** `agg(nueva_columna=('columna', 'funcion'))`
- **Grouped rolling:** Rolling window dentro de cada grupo
- **Resample + groupby:** Combinar remuestreo temporal con agrupación

---

## 2. Ejemplos Prácticos

### Ejemplo 1: agg — [sum, mean, std, min, max] por categoría

```python
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: agg — [sum, mean, std, min, max] por categoría*


---
 — Realiza la operación indicada con los parámetros definidos..

---

---


### Ejemplo 2: agg con funciones personalizadas

```python
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# **Explicación línea por línea:**

*Ejemplo 2: agg con funciones personalizadas*


---
 para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


### Ejemplo 3: transform — z-score de precios dentro de cada categoría

```python
```

**Salida:**

```
**Explicación línea por línea:**

*Ejemplo 3: transform — z-score de precios dentro de cada categoría*


---
nan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


### E**Explicación línea por línea:**

*E*

1. `**Explicación línea por línea:**` — Realiza la operación indicada con los parámetros definidos..
2. `*Ejemplo 3: transform — z-score de precios dentro de cada categoría*` — Realiza la operación indicada con los parámetros definidos..
3. `---` — Realiza la operación indicada con los parámetros definidos..
4. `nan en las variables definidas.**Explicación línea por línea:**

1. `**Explicación línea por línea:**` — Realiza la operación indicada con los parámetros definidos..
2. `*Ejemplo 3: transform — z-score de precios dentro de cada categoría*` — Realiza la operación indicada con los parámetros definidos..
3. `---` — Realiza la operación indicada con los parámetros definidos..
4. `nan en las variables definidas.` — **Explicación línea por línea:**

1. `**Explicación línea por línea:**` — Realiza la operación indicada con los parámetros definidos..
2. `*Ejemplo 3: transform — z-score de precios dentro de cada categoría*` — Realiza la operación indicada con los parámetros definidos..
3. `---` — Realiza la operación indicada con los parámetros definidos..
4. `nan en las variables definidas.` — **Explicación línea por línea:**

1. `**Explicación línea por línea:**` — Realiza la operación indicada con los parámetros definidos..
2. `*Ejemplo 3: transform — z-score de precios dentro de cada categoría*` — Realiza la operación indicada con los parámetros definidos..
3. `---` — Realiza la operación indicada con los parámetros definidos..
4. `nan en las variables**Explicación línea por línea:**

1. `**Explicación línea por línea:**` — Realiza la operación indicada con los parámetros definidos..
2. `*Ejemplo 3: transform — z-score de precios dentro de cada categoría*` — Realiza la operación indicada con los parámetros definidos..
3. `---` — Realiza la operación indicada con los parámetros definidos..
4. `nan en las variables defi**Explicación línea por línea:**

1. `**Explicación línea por línea:**` — Realiza la operación indicada con los parámetros definidos..
2. `*Ejemplo 3: transform — z-score de precios dentro de cada categoría*` — Realiza la operación indicada con los parámetros definidos..
3. `---` — Realiza la operación indicada con los parámetros definidos..
4. `nan en las variables definidas.` — Ejec**Explicación línea por línea:**

1. `**Explicación línea por línea:**` — Realiza la operación indicada con los parámetros definidos..
2. `*Ejemplo 3: transform — z-score de precios dentro de cada categoría*` — Realiza la operación indicada con los parámetros definidos..
3. `---` — Realiza la operación indicada con los parámetros definidos..
4. `nan en las variables definida**Explicación línea por línea:**

1. `**Explicación línea por línea:**` — Realiza la operación indicada con los parámetros definidos..
2. `*Ejemplo 3: transform — z-score de precios dentro de cada categoría*` — Realiza la operación indicada con los parámetros definidos..
3. `---` — Realiza la operación indicada con los parámetros definidos..
4. `nan en las variables defin**Explicación línea por línea:**

1. `**Explicación línea por línea:**` — Realiza la operación indicada con los parámetros definidos..
2. `*Ejemplo 3: transform — z-score de precios dentro de cada categoría*` — Realiza la operación indicada con los parámetros definidos..
3. `---` — Realiza la operación indicada con los parámetros definidos..
4. `nan en **Explicación línea por línea:**

1. `**Explicación línea por línea:**` — Realiza la operación indicada con los parámetros definidos..
2. `*Ejemplo 3: transform — z-score de precios dentro de cada categoría*` — Realiza la operación indicada con los parámetros definidos..
3. `---` — Realiza la operación indicada con los parámetros definidos..
4. `nan en las variables definidas.`**Explicación línea por línea:**

1. `**Explicación línea por línea:**` — Realiza la operación indicada con los parámetros definidos..
2. `*Ejemplo 3: transform — z-score de precios dentro de cada categoría*` — Realiza la operación indicada con los parámetros definidos..
3. `---` — Realiza la operación indicada con los parámetros definidos..
4. `nan en las variables definidas.` — Ejecut**Explicación línea por línea:**

1. `**Explicación línea por línea:**` — Realiza la operación indicada con los parámetros definidos..
2. `*Ejemplo 3: transform — z-score de precios dentro de cada categoría*` — Realiza la operación indicada con los parámetros definidos..
3. `---` — Realiza la operación indicada con los parámetros definidos..
4. `nan en las variables definidas.` — Eje**Explicación línea por línea:**

1. `**Explicación línea por línea:**` — Realiza la operación indicada con los parámetros definidos..
2. `*Ejemplo 3: transform — z-score de precios dentro de cada categoría*` — Realiza la operación indicada con los parámetros definidos..
3. `---` — Realiza la operación indicada con los parámetros definidos..
4. `nan en las variables definidas.` — E**Explicación línea por línea:**

1. `**Explicación línea por línea:**` — Realiza la operación indicada con los parámetros definidos..
2. `*Ejemplo 3: transform — z-score de precios dentro de cada categoría*` — Realiza la operación indicada con los parámetros definidos..
3. `---` — Realiza la operación indicada con los parámetros definidos..
4. `nan en las variables defin**Explicación línea por línea:**

1. `**Explicación línea por línea:**` — Realiza la operación indicada con los parámetros definidos..
2. `*Ejemplo 3: transform — z-score de precios dentro de cada categoría*` — Realiza la operación indicada con los parámetros definidos..
3. `---` — Realiza la operación indicada con los parámetros definidos..
4. `nan en las variables definidas.` — Ejecuta la op**Explicación línea por línea:**

1. `**Explicación línea por línea:**` — Realiza la operación indicada con los parámetros definidos..
2. `*Ejemplo 3: transform — z-score de precios dentro de cada categoría*` — Realiza la operación indicada con los parámetros definidos..
3. `---` — Realiza la operación indicada con los parámetros definidos..
4. `nan en las variables definidas.` — Ejec**Explicación línea por línea:**

1. `**Explicación línea por línea:**` — Realiza la operación indicada con los parámetros definidos..
2. `*Ejemplo 3: transform — z-score de precios dentro de cada categoría*` — Realiza la operación indicada con los parámetros definidos..
3. `---` — Realiza la operación indicada con los parámetros definidos..
4. `nan en las variables definidas.` — Realiza la operación indicada con los parámetros definidos..

---
ada.

---

ión eRealiza la operación indicada con los parámetros definidos..

---
ficada.

---
cada.

---
a.

---
peRealiza la operación indicada con los parámetros definidos..

---
— Realiza la operación indicada con los parámetros definidos..

---
ión eRealiza la operación indicada con los parámetros definidos..

---
 eRealiza la operación indicada con los parámetros definidos..

---
ada.

---
ción eRealiza la operación indicada con los parámetros definidos..

---
Realiza la operación indicada con los parámetros definidos.

---
Realiza la operación indicada con los parámetros definidos.

---
Realiza la operación indicada con los parámetros definidos.

---
Realiza la operación indicada con los parámetros definidos..

---
l total dentro de grupo

```python
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


### Ejemplo 5: filter — categorías con ingreso total > $50,000

```python
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


### Ejemplo 6: apply — regresión lineal por grupo (pendiente)

```python
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


### Ejemplo 7: rank — ranking por ingreso dentro de categoría

```python
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


### Ejemplo 8: dense_rank — ranking sin huecos

```python
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


### Ejemplo 9: pct_rank — percentil dentro de grupo

```python
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


### Ejemplo 10: cumcount — enumerar ventas por producto por fecha

```python
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


### Ejemplo 11: ngroup — etiquetar grupos numéricamente

```python
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


### Ejemplo 12: pipe — encadenar operaciones groupby

```python
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


### Ejemplo 13: named aggregation

```python
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


### Ejemplo 14: grouped rolling — media móvil por producto

```python
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


### Ejemplo 15: resample + groupby — tendencia semanal por sucursal

```python
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


### Ejemplo 16: nunique — productos únicos vendidos por sucursal

```python
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


### Ejemplo 17: first/last — primera/última venta por producto

```python
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


### Ejemplo 18: nth — n-ésima venta de cada producto

```python
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


### Ejemplo 19: value_counts por grupo — producto más vendido por sucursal

```python
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


### Ejemplo 20: idxmax — producto con mayor ingreso por categoría

```python
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


---

## 3. Resumen Teórico

| Operación | Uso en Ventas/Compras/Inventarios |
|-----------|----------------------------------|
| agg | Resúmenes múltiples por categoría/sucursal |
| transform | Normalización, % del total, z-scores intra-grupo |
| filter | Seleccionar grupos que cumplen condición |
| apply | Modelos por grupo (regresión, clustering) |
| rank | Ranking de productos dentro de categoría |
| cumcount | Enumerar transacciones secuenciales |
| ngroup | Identificación numérica de grupos |
| pipe | Pipelines de procesamiento groupby |
| named agg | Código limpio y auto-documentado |
| grouped rolling | Tendencias móviles por producto |
| resample + groupby | Análisis temporal por sucursal |

---

## 4. Ejercicios Propuestos

**Ejercicio 1:** Dado un DataFrame de compras con columnas `[proveedor, monto, fecha]`, calcula suma, promedio y desviación por proveedor usando `agg`.

**Ejercicio 2:** Usa `transform` para calcular el porcentaje que cada compra representa del total por proveedor.

**Ejercicio 3:** Filtra los proveedores cuyo monto total de compras supere los $100,000 usando `filter`.

**Ejercicio 4:** Aplica una regresión lineal a las ventas diarias de cada producto y reporta la pendiente (tendencia alcista o bajista).

**Ejercicio 5:** Calcula el ranking de productos por margen de ganancia dentro de cada categoría usando `rank(method='dense')`.

**Ejercicio 6:** Usa `named aggregation` para crear un resumen con columnas: `total, promedio, max, min, conteo` por sucursal.

**Ejercicio 7:** Calcula la media móvil de 7 días para las ventas de cada producto usando `groupby + rolling`.

**Ejercicio 8:** Encuentra el producto con mayor ingreso en cada categoría usando `idxmax` y reporta el resultado.

---

*Fin del documento I03 — Groupby Avanzado en Pandas*
