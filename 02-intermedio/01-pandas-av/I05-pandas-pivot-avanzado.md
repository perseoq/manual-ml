# I05 — Pivot, Reshape y Tablas Dinámicas en Pandas

## 1. Introducción Teórica

La reestructuración de datos (reshape) es una operación fundamental en el análisis de ventas, permitiendo transformar datos entre formato largo (tidy) y ancho (para reportes).

### Operaciones clave

- **pivot_table():** Tabla dinámica estilo Excel, con multiíndice y agregaciones
- **crosstab():** Tabla de contingencia para frecuencias cruzadas
- **melt():** Unpivot: de ancho a largo (columnas → filas)
- **wide_to_long():** Versión especializada de melt para patrones de nombres
- **stack():** De ancho a largo con MultiIndex
- **unstack():** De largo a ancho con MultiIndex
- **explode():** Expandir listas/celdas en múltiples filas
- **get_dummies():** One-hot encoding de variables categóricas

### Parámetros importantes de pivot_table

- **index:** Filas de la tabla
- **columns:** Columnas de la tabla
- **values:** Valores a agregar
- **aggfunc:** Función(es) de agregación (default: mean)
- **margins=True:** Agrega totales parciales y gran total
- **fill_value:** Valor para celdas vacías

---

## 2. Ejemplos Prácticos

### Ejemplo 1: pivot_table — ingresos por sucursal × categoría

```python
import pandas as pd
import numpy as np

np.random.seed(42)
ventas = pd.DataFrame({
    'sucursal': np.random.choice(['Norte', 'Sur', 'Este', 'Oeste'], 200),
    'categoria': np.random.choice(['Electrónicos', 'Ropa', 'Hogar',
                                    'Deportes', 'Libros'], 200),
    'ingreso': np.random.uniform(100, 1000, 200).round(2),
    'cantidad': np.random.randint(1, 20, 200)
})

pivot = pd.pivot_table(ventas,
                       index='sucursal',
                       columns='categoria',
                       values='ingreso',
                       aggfunc='sum')
print(pivot.round(2))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: pivot_table — ingresos por sucursal × categoría.*

1. `import pandas as pd` — Importa las librerías necesarias para el análisis.
2. `import numpy as np` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: pivot_table con margins

```python
pivot_margins = pd.pivot_table(ventas,
                               index='sucursal',
                               columns='categoria',
                               values='ingreso',
                               aggfunc='sum',
                               margins=True,
                               margins_name='Total')
print(pivot_margins.round(2))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: pivot_table con margins.*

1. `pivot_margins = pd.pivot_table(ventas,` — Reorganiza los datos de formato largo a ancho.
2. `print(pivot_margins.round(2))` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: pivot_table con aggfunc múltiple

```python
pivot_multi = pd.pivot_table(ventas,
                             index='sucursal',
                             columns='categoria',
                             values='ingreso',
                             aggfunc=['sum', 'mean', 'count'])
print(pivot_multi.round(2))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: pivot_table con aggfunc múltiple.*

1. `pivot_multi = pd.pivot_table(ventas,` — Reorganiza los datos de formato largo a ancho.
2. `print(pivot_multi.round(2))` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: pivot_table multiíndice

```python
ventas['mes'] = np.random.choice(['Ene', 'Feb', 'Mar', 'Abr'], 200)
ventas['trimestre'] = ventas['mes'].map({
    'Ene': 'Q1', 'Feb': 'Q1', 'Mar': 'Q1', 'Abr': 'Q2'
})

pivot_multi = pd.pivot_table(ventas,
                             index=['sucursal', 'mes'],
                             columns='categoria',
                             values='ingreso',
                             aggfunc='sum')
print(pivot_multi.round(2))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: pivot_table multiíndice.*

1. `pivot_multi = pd.pivot_table(ventas,` — Reorganiza los datos de formato largo a ancho.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: crosstab — frecuencia producto × día_semana

```python
ventas['dia_semana'] = np.random.choice(
    ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom'], 200
)

crosstab = pd.crosstab(ventas['categoria'], ventas['dia_semana'])
print(crosstab)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: crosstab — frecuencia producto × día_semana.*

1. `print(crosstab)` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: crosstab con normalize="index"

```python
crosstab_pct_row = pd.crosstab(ventas['categoria'], ventas['dia_semana'],
                                normalize='index')
print((crosstab_pct_row * 100).round(1))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: crosstab con normalize="index".*

1. `print((crosstab_pct_row * 100).round(1))` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: crosstab con normalize="columns"

```python
crosstab_pct_col = pd.crosstab(ventas['categoria'], ventas['dia_semana'],
                                normalize='columns')
print((crosstab_pct_col * 100).round(1))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: crosstab con normalize="columns".*

1. `print((crosstab_pct_col * 100).round(1))` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: crosstab 3 vías

```python
crosstab_3 = pd.crosstab(
    [ventas['categoria'], ventas['sucursal']],
    ventas['dia_semana']
)
print(crosstab_3)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: crosstab 3 vías.*

1. `print(crosstab_3)` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: melt — unpivot de ancho a largo

```python
# Datos en formato ancho (típico de reporte)
ventas_anchas = pd.DataFrame({
    'sucursal': ['Norte', 'Sur', 'Este'],
    'Enero': [10000, 12000, 9000],
    'Febrero': [11000, 13000, 9500],
    'Marzo': [10500, 12500, 9200]
})

ventas_largas = ventas_anchas.melt(
    id_vars=['sucursal'],
    var_name='mes',
    value_name='ingreso'
)
print(ventas_largas)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: melt — unpivot de ancho a largo.*

1. Datos en formato ancho (típico de reporte)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: melt con id_vars y value_vars específicos

```python
ventas_largas_parcial = ventas_anchas.melt(
    id_vars=['sucursal'],
    value_vars=['Enero', 'Marzo'],
    var_name='mes',
    value_name='ingreso'
)
print(ventas_largas_parcial)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: melt con id_vars y value_vars específicos.*

1. `ventas_largas_parcial = ventas_anchas.melt(` — Reorganiza los datos de formato ancho a largo.
2. `print(ventas_largas_parcial)` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: wide_to_long — con stubnames y i, j

```python
# Datos con patrón: ventas_ene, ventas_feb, gastos_ene, gastos_feb
df = pd.DataFrame({
    'sucursal': ['Norte', 'Sur'],
    'region': ['A', 'B'],
    'ventas_ene': [100, 200],
    'ventas_feb': [110, 210],
    'gastos_ene': [30, 50],
    'gastos_feb': [35, 55]
})

largo = pd.wide_to_long(
    df,
    stubnames=['ventas', 'gastos'],
    i=['sucursal', 'region'],
    j='mes',
    sep='_',
    suffix='\\w+'
).reset_index()
print(largo)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: wide_to_long — con stubnames y i, j.*

1. Datos con patrón: ventas_ene, ventas_feb, gastos_ene, gastos_feb

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: stack — de ancho a largo (multiindex)

```python
# pivot_table genera MultiIndex en columnas
pivot = pd.pivot_table(ventas,
                       index='sucursal',
                       columns='categoria',
                       values='ingreso',
                       aggfunc='sum')

apilado = pivot.stack()
print(apilado.head(10))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
# Las primeras filas muestran la estructura de los datos procesados:
#    Las columnas dependen del DataFrame utilizado.
# 0    Los valores dependen de los datos de entrada.
# 1    valor4    valor5    valor6
```

**Explicación línea por línea:**

*Ejemplo 12: stack — de ancho a largo (multiindex).*

1. pivot_table genera MultiIndex en columnas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: unstack — de largo a ancho

```python
# Des-apilar el resultado anterior
desapilado = apilado.unstack()
print(desapilado)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: unstack — de largo a ancho.*

1. Des-apilar el resultado anterior

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: unstack con fill_value

```python
# DataFrame con datos dispersos
disperso = ventas.groupby(['sucursal', 'categoria'])['ingreso'].sum()
print("Original (puede tener NaN):\n", disperso.unstack())

# Con fill_value
print("\nCon fill_value=0:\n", disperso.unstack(fill_value=0))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: unstack con fill_value.*

1. DataFrame con datos dispersos
2. Con fill_value

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: stack con level específico

```python
# pivot_table multiíndice
pivot_multi = pd.pivot_table(ventas,
                             index=['sucursal', 'mes'],
                             columns='categoria',
                             values='ingreso',
                             aggfunc='sum')

# Stackear solo un nivel
stack_level_0 = pivot_multi.stack(level=0)
print(stack_level_0.head(10))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
# Las primeras filas muestran la estructura de los datos procesados:
#    Las columnas dependen del DataFrame utilizado.
# 0    Los valores dependen de los datos de entrada.
# 1    valor4    valor5    valor6
```

**Explicación línea por línea:**

*Ejemplo 15: stack con level específico.*

1. pivot_table multiíndice
2. Stackear solo un nivel

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 16: explode — expandir listas en filas

```python
# Productos con múltiples categorías
productos = pd.DataFrame({
    'producto': ['Laptop', 'Mouse', 'Teclado'],
    'categorias': [['Electrónica', 'Cómputo'],
                   ['Periféricos', 'Cómputo'],
                   ['Periféricos']],
    'precio': [15000, 500, 800]
})

expandido = productos.explode('categorias')
print(expandido)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: explode — expandir listas en filas.*

1. Productos con múltiples categorías

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 17: get_dummies con prefix y drop_first

```python
clientes = pd.DataFrame({
    'cliente_id': [1, 2, 3, 4],
    'tipo': ['Premium', 'Regular', 'Premium', 'VIP'],
    'region': ['Norte', 'Sur', 'Norte', 'Este']
})

dummies = pd.get_dummies(clientes, columns=['tipo', 'region'],
                          prefix=['tipo', 'reg'],
                          drop_first=True)
print(dummies)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: get_dummies con prefix y drop_first.*

1. `print(dummies)` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 18: get_dummies con dummy_na

```python
datos = pd.DataFrame({
    'producto': ['A', 'B', 'C', 'D'],
    'categoria': ['Electrónica', 'Hogar', np.nan, 'Electrónica']
})

dummies_na = pd.get_dummies(datos, columns=['categoria'],
                             dummy_na=True)
print(dummies_na)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: get_dummies con dummy_na.*

1. `print(dummies_na)` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 19: pivot con aggfunc personalizado

```python
def rango_intercuartil(x):
    return x.quantile(0.75) - x.quantile(0.25)

pivot_custom = pd.pivot_table(ventas,
                              index='sucursal',
                              columns='categoria',
                              values='ingreso',
                              aggfunc=rango_intercuartil)
print(pivot_custom.round(2))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 19: pivot con aggfunc personalizado.*

1. `pivot_custom = pd.pivot_table(ventas,` — Reorganiza los datos de formato largo a ancho.
2. `print(pivot_custom.round(2))` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 20: Ejercicio integrador — reshape completo

```python
# 1. Datos originales: ventas diarias por sucursal en ancho
raw = pd.DataFrame({
    'sucursal': ['Norte', 'Sur', 'Este'],
    '2024-01-01': [1200, 1500, 900],
    '2024-01-02': [1300, 1400, 950],
    '2024-01-03': [1250, 1600, 920],
    'categoria': ['Mix', 'Mix', 'Mix']
})

# 2. Melt a formato largo
largo = raw.melt(id_vars=['sucursal', 'categoria'],
                 var_name='fecha',
                 value_name='ingreso')
largo['fecha'] = pd.to_datetime(largo['fecha'])

# 3. Extraer día de semana
largo['dia'] = largo['fecha'].dt.day_name()

# 4. Pivot table por sucursal y día
pivot_final = pd.pivot_table(largo,
                             index='sucursal',
                             columns='dia',
                             values='ingreso',
                             aggfunc='mean',
                             margins=True)
print("Pivot final:\n", pivot_final.round(2))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 20: Ejercicio integrador — reshape completo.*

1. 1. Datos originales: ventas diarias por sucursal en ancho
2. 2. Melt a formato largo
3. 3. Extraer día de semana
4. 4. Pivot table por sucursal y día

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 3. Resumen Teórico

| Operación | Uso en Ventas/Compras/Inventarios |
|-----------|----------------------------------|
| pivot_table | Reportes de ventas por sucursal × categoría |
| crosstab | Frecuencia de productos por día, clientes por zona |
| melt | Convertir reportes mensuales a formato analizable |
| wide_to_long | Datos con patrones (venta_ene, gasto_ene, ...) |
| stack/unstack | Navegar entre niveles de agregación |
| explode | Productos con múltiples categorías o proveedores |
| get_dummies | Preparar variables categóricas para ML |
| margins | Totales y subtotales automáticos |

---

## 4. Ejercicios Propuestos

**Ejercicio 1:** Crea una pivot_table que muestre el total de ventas por sucursal (filas) y mes (columnas) con totals.

**Ejercicio 2:** Usa crosstab con normalize para mostrar la distribución porcentual de productos por día de la semana.

**Ejercicio 3:** Dado un DataFrame ancho con columnas `ene_ventas, feb_ventas, ene_gastos, feb_gastos`, usa melt para llevarlo a formato largo.

**Ejercicio 4:** Toma un DataFrame con 3 columnas categóricas y usa get_dummies con drop_first para prepararlo para regresión.

**Ejercicio 5:** Aplica stack y unstack a una pivot_table multiíndice para cambiar qué niveles están en filas vs columnas.

**Ejercicio 6:** Usa explode para expandir una columna de listas de productos comprados en cada transacción.

**Ejercicio 7:** Emplea wide_to_long para transformar datos de inventario con columnas `stock_ene, stock_feb, precio_ene, precio_feb`.

**Ejercicio 8:** Crea un pipeline que: importe datos anchos → melt → agregue fecha → pivot_table con margins → exporte.

---

*Fin del documento I05 — Pivot, Reshape y Tablas Dinámicas en Pandas*
