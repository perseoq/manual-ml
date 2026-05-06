# I04 — Merge y Combinación Avanzada en Pandas

## 1. Introducción Teórica

Pandas ofrece múltiples formas de combinar DataFrames, cada una con casos de uso específicos en el contexto de ventas, compras e inventarios.

### Tipos de combinación

- **merge():** Combinación estilo SQL (inner, left, right, outer)
- **merge_asof():** Combinación por proximidad en clave temporal (sin coincidencia exacta)
- **merge_ordered():** Combina series temporales llenando gaps
- **join():** Conveniente para combinar por índice
- **concat():** Apila DataFrames vertical u horizontalmente
- **compare():** Encuentra diferencias elemento a elemento
- **combine_first():** Rellena NaN con valores de otro DataFrame
- **update():** Modifica valores in-place basado en otro DataFrame

### Parámetros clave de merge

- **validate:** Verifica relaciones ("one_to_one", "one_to_many", "many_to_one", "many_to_many")
- **indicator:** Agrega columna `_merge` indicando origen de cada fila
- **suffixes:** Personaliza sufijos para columnas duplicadas
- **left_on/right_on, left_index/right_index:** Flexibilidad total en keys

---

## 2. Ejemplos Prácticos

### Ejemplo 1: merge_asof — ventas con precios más cercanos en fecha

```python
import pandas as pd
import numpy as np

# Ventas: fecha y cantidad
ventas = pd.DataFrame({
    'fecha': pd.to_datetime(['2024-01-03', '2024-01-07', '2024-01-10',
                             '2024-01-14', '2024-01-18']),
    'producto': ['A', 'A', 'A', 'A', 'A'],
    'cantidad': [10, 15, 8, 20, 12]
})

# Precios: cambios de precio en fechas específicas
precios = pd.DataFrame({
    'fecha': pd.to_datetime(['2024-01-01', '2024-01-05', '2024-01-12',
                             '2024-01-15', '2024-01-20']),
    'producto': ['A', 'A', 'A', 'A', 'A'],
    'precio': [100, 110, 105, 115, 120]
})

# merge_asof: asigna el precio más cercano ANTERIOR a cada venta
resultado = pd.merge_asof(ventas.sort_values('fecha'),
                          precios.sort_values('fecha'),
                          on='fecha', by='producto')
print(resultado)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: merge_asof — ventas con precios más cercanos en fecha.*

1. Ventas: fecha y cantidad
2. Precios: cambios de precio en fechas específicas
3. merge_asof: asigna el precio más cercano ANTERIOR a cada venta

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: merge_asof con direction

```python
# Forward (default): busca la clave más cercana hacia adelante en el tiempo
fwd = pd.merge_asof(ventas.sort_values('fecha'),
                    precios.sort_values('fecha'),
                    on='fecha', by='producto', direction='forward')
print("Forward:\n", fwd)

# Backward: busca hacia atrás
bwd = pd.merge_asof(ventas.sort_values('fecha'),
                    precios.sort_values('fecha'),
                    on='fecha', by='producto', direction='backward')
print("\nBackward:\n", bwd)

# Nearest: el más cercano en cualquier dirección
near = pd.merge_asof(ventas.sort_values('fecha'),
                     precios.sort_values('fecha'),
                     on='fecha', by='producto', direction='nearest')
print("\nNearest:\n", near)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: merge_asof con direction.*

1. Forward (default): busca la clave más cercana hacia adelante en el tiempo
2. Backward: busca hacia atrás
3. Nearest: el más cercano en cualquier dirección

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: validate — verificar relaciones

```python
# one_to_one: cada key aparece una vez en ambos
df1 = pd.DataFrame({'key': [1, 2, 3], 'valor': ['a', 'b', 'c']})
df2 = pd.DataFrame({'key': [1, 2, 3], 'precio': [100, 200, 300]})

try:
    m = pd.merge(df1, df2, on='key', validate='one_to_one')
    print("Merge one_to_one exitoso:\n", m)
except Exception as e:
    print(f"Error: {e}")

# many_to_one: muchas filas en df1, una en df2
df1_m = pd.DataFrame({'key': [1, 1, 2, 3], 'venta': [10, 20, 30, 40]})
df2_o = pd.DataFrame({'key': [1, 2, 3], 'nombre': ['A', 'B', 'C']})
m = pd.merge(df1_m, df2_o, on='key', validate='many_to_one')
print("\nMerge many_to_one:\n", m)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: validate — verificar relaciones.*

1. one_to_one: cada key aparece una vez en ambos
2. many_to_one: muchas filas en df1, una en df2

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: indicator=True — origen de cada fila

```python
clientes_norte = pd.DataFrame({
    'cliente_id': [101, 102, 103, 104],
    'nombre': ['Ana', 'Luis', 'Carlos', 'Marta']
})

clientes_sur = pd.DataFrame({
    'cliente_id': [103, 104, 105, 106],
    'nombre': ['Carlos', 'Marta', 'Pedro', 'Lucía']
})

merged = pd.merge(clientes_norte, clientes_sur, on='cliente_id',
                  how='outer', indicator=True)
print(merged)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: indicator=True — origen de cada fila.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: suffixes personalizados para columnas duplicadas

```python
ventas_ene = pd.DataFrame({
    'producto': ['A', 'B', 'C'],
    'precio': [100, 200, 150],
    'cantidad': [10, 20, 15]
})

ventas_feb = pd.DataFrame({
    'producto': ['A', 'B', 'D'],
    'precio': [110, 190, 160],
    'cantidad': [12, 18, 20]
})

comparacion = pd.merge(ventas_ene, ventas_feb, on='producto',
                       how='outer', suffixes=('_ene', '_feb'))
print(comparacion)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: suffixes personalizados para columnas duplicadas.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: merge con index

```python
productos = pd.DataFrame({
    'producto_id': [1, 2, 3, 4],
    'nombre': ['Laptop', 'Mouse', 'Teclado', 'Monitor']
}).set_index('producto_id')

inventario = pd.DataFrame({
    'producto_id': [1, 2, 3, 4],
    'stock': [50, 200, 150, 30]
}).set_index('producto_id')

resultado = pd.merge(productos, inventario,
                     left_index=True, right_index=True)
print(resultado)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: merge con index.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: merge con MultiIndex

```python
ventas_multi = pd.DataFrame({
    'sucursal': ['Norte', 'Norte', 'Sur', 'Sur'],
    'producto': ['A', 'B', 'A', 'B'],
    'ingreso': [1000, 1500, 1200, 1800]
})

costos_multi = pd.DataFrame({
    'sucursal': ['Norte', 'Norte', 'Sur', 'Sur', 'Norte'],
    'producto': ['A', 'B', 'A', 'B', 'C'],
    'costo': [600, 900, 700, 1100, 300]
})

# merge con múltiples keys
resultado = pd.merge(ventas_multi, costos_multi,
                     on=['sucursal', 'producto'], how='left')
resultado['margen'] = resultado['ingreso'] - resultado['costo']
print(resultado)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: merge con MultiIndex.*

1. merge con múltiples keys

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: join con lista de DataFrames

```python
enero = pd.DataFrame({'ventas': [100, 200]}, index=['A', 'B'])
febrero = pd.DataFrame({'ventas': [150, 250]}, index=['A', 'B'])
marzo = pd.DataFrame({'ventas': [130, 220]}, index=['A', 'B'])

trimestre = enero.join([febrero, marzo], how='outer', lsuffix='_ene',
                        rsuffix='_feb')
# Nota: join con lista requiere sufijos manuales
print(trimestre)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: join con lista de DataFrames.*

1. Nota: join con lista requiere sufijos manuales

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: concat con keys para identificar origen

```python
q1 = pd.DataFrame({'producto': ['A', 'B'], 'ventas': [300, 500]})
q2 = pd.DataFrame({'producto': ['A', 'B'], 'ventas': [400, 600]})
q3 = pd.DataFrame({'producto': ['A', 'B'], 'ventas': [350, 550]})

anual = pd.concat([q1, q2, q3], keys=['Q1', 'Q2', 'Q3'])
print(anual)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: concat con keys para identificar origen.*

1. `print(anual)` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: merge_ordered — combinar series temporales

```python
# Datos de ventas no uniformes
df1 = pd.DataFrame({
    'fecha': pd.to_datetime(['2024-01-01', '2024-01-05', '2024-01-10']),
    'ventas': [100, 150, 130]
})
df2 = pd.DataFrame({
    'fecha': pd.to_datetime(['2024-01-03', '2024-01-08', '2024-01-12']),
    'gastos': [30, 40, 35]
})

combinado = pd.merge_ordered(df1, df2, on='fecha', fill_method='ffill')
print(combinado)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: merge_ordered — combinar series temporales.*

1. Datos de ventas no uniformes

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: compare — encontrar diferencias

```python
inventario_ayer = pd.DataFrame({
    'producto': ['A', 'B', 'C'],
    'stock': [100, 200, 150],
    'precio': [10, 20, 15]
}).set_index('producto')

inventario_hoy = pd.DataFrame({
    'producto': ['A', 'B', 'C'],
    'stock': [95, 200, 160],
    'precio': [10, 22, 15]
}).set_index('producto')

diferencias = inventario_ayer.compare(inventario_hoy)
print("Cambios en inventario:\n", diferencias)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: compare — encontrar diferencias.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: combine_first — rellenar NaN con valores de otro

```python
# Datos incompletos de clientes
df_principal = pd.DataFrame({
    'cliente': ['Ana', 'Luis', 'Carlos', 'Marta'],
    'email': ['ana@mail.com', np.nan, 'carlos@mail.com', np.nan],
    'telefono': ['555-0101', '555-0102', np.nan, np.nan]
})

df_secundario = pd.DataFrame({
    'cliente': ['Luis', 'Marta', 'Ana'],
    'email': ['luis@mail.com', 'marta@mail.com', np.nan],
    'telefono': [np.nan, '555-0104', '555-0100']
})

completo = df_principal.set_index('cliente').combine_first(
    df_secundario.set_index('cliente')
).reset_index()
print(completo)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: combine_first — rellenar NaN con valores de otro.*

1. Datos incompletos de clientes

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: update — modificar valores in-place

```python
inventario = pd.DataFrame({
    'producto': ['A', 'B', 'C', 'D'],
    'stock': [100, 200, 150, 300]
}).set_index('producto')

ajuste = pd.DataFrame({
    'producto': ['B', 'D'],
    'stock': [180, 280]
}).set_index('producto')

inventario.update(ajuste)
print("Inventario actualizado:\n", inventario)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: update — modificar valores in-place.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: merge con diferentes dtypes

```python
df_int = pd.DataFrame({'key': [1, 2, 3], 'valor': [10, 20, 30]})
df_float = pd.DataFrame({'key': [1, 2, 3], 'valor': [10.5, 20.5, 30.5]})

resultado = pd.merge(df_int, df_float, on='key', suffixes=('_int', '_float'))
print(resultado.dtypes)
print(resultado)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: merge con diferentes dtypes.*

1. `resultado = pd.merge(df_int, df_float, on='key', suffixes=('_int', '_float'))` — Combina dos DataFrames por una columna clave.
2. `print(resultado.dtypes)` — Muestra el resultado por pantalla.
3. `print(resultado)` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: merge_asof con tolerance

```python
import datetime

ventas = pd.DataFrame({
    'fecha': pd.to_datetime(['2024-01-03', '2024-01-08']),
    'cantidad': [10, 15]
})

precios = pd.DataFrame({
    'fecha': pd.to_datetime(['2024-01-01', '2024-01-10']),
    'precio': [100, 110]
})

resultado = pd.merge_asof(
    ventas.sort_values('fecha'),
    precios.sort_values('fecha'),
    on='fecha',
    tolerance=pd.Timedelta('3 days')
)
print(resultado)  # fila con diferencia > 3d tendrá NaN
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: merge_asof con tolerance.*

1. `import datetime` — Importa las librerías necesarias para el análisis.
2. `'fecha': pd.to_datetime(['2024-01-03', '2024-01-08']),` — Convierte la columna a formato datetime.
3. `'fecha': pd.to_datetime(['2024-01-01', '2024-01-10']),` — Convierte la columna a formato datetime.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 16: merge con left_on y right_index

```python
df_left = pd.DataFrame({
    'producto_id': [1, 2, 3, 4],
    'ventas': [100, 200, 150, 300]
})

df_right = pd.DataFrame({
    'nombre': ['Laptop', 'Mouse', 'Teclado', 'Monitor'],
    'costo': [800, 20, 50, 300]
}, index=[1, 2, 3, 4])

resultado = pd.merge(df_left, df_right,
                     left_on='producto_id', right_index=True)
print(resultado)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: merge con left_on y right_index.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 17: concat axis=1 con join="inner"

```python
df_ingresos = pd.DataFrame({'A': [100, 200], 'B': [150, 250]}, index=[1, 2])
df_costos = pd.DataFrame({'A': [60, 120], 'B': [90, 150]}, index=[2, 3])
df_margen = pd.DataFrame({'A': [40, 80], 'B': [60, 100]}, index=[1, 2])

combinado = pd.concat([df_ingresos, df_costos, df_margen],
                      axis=1, join='inner')
combinado.columns = ['ingreso', 'costo', 'margen']
print(combinado)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: concat axis=1 con join="inner".*

1. `print(combinado)` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 18: merge con duplicados en key

```python
df_ventas = pd.DataFrame({
    'pedido': [1, 1, 2, 3],
    'producto': ['A', 'B', 'A', 'C']
})

df_info = pd.DataFrame({
    'producto': ['A', 'A', 'B', 'C'],
    'precio': [100, 110, 200, 150]
})

resultado = pd.merge(df_ventas, df_info, on='producto')
print("Merge con duplicados en ambos lados:\n", resultado)
print(f"Filas originales: {len(df_ventas)}, resultado: {len(resultado)}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: merge con duplicados en key.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 19: merge con NaN en key

```python
df1 = pd.DataFrame({'key': [1, 2, np.nan, 4], 'valor': ['a', 'b', 'c', 'd']})
df2 = pd.DataFrame({'key': [1, np.nan, 3, 4], 'precio': [100, 200, 300, 400]})

resultado = pd.merge(df1, df2, on='key', how='outer')
print("NaNs en key no se emparejan:\n", resultado)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 19: merge con NaN en key.*

1. `resultado = pd.merge(df1, df2, on='key', how='outer')` — Combina dos DataFrames por una columna clave.
2. `print("NaNs en key no se emparejan:\n", resultado)` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 20: Integrador — unir ventas+productos+proveedores

```python
np.random.seed(42)

ventas = pd.DataFrame({
    'venta_id': range(1, 11),
    'producto_id': np.random.choice([1, 2, 3, 4], 10),
    'fecha': pd.date_range('2024-01-01', periods=10, freq='D'),
    'cantidad': np.random.randint(1, 10, 10),
    'precio_unitario': np.random.uniform(10, 100, 10).round(2)
})

productos = pd.DataFrame({
    'producto_id': [1, 2, 3, 4],
    'nombre': ['Laptop', 'Mouse', 'Teclado', 'Monitor'],
    'proveedor_id': [101, 102, 101, 103],
    'categoria': ['Electrónica', 'Periféricos', 'Periféricos', 'Electrónica']
})

proveedores = pd.DataFrame({
    'proveedor_id': [101, 102, 103],
    'nombre_prov': ['TechSupply', 'InputMasters', 'DisplayCo'],
    'tiempo_entrega': [5, 3, 7]  # días
})

# Unir todo
resultado = (ventas
             .merge(productos, on='producto_id')
             .merge(proveedores, on='proveedor_id'))

resultado['total'] = resultado['cantidad'] * resultado['precio_unitario']
print(resultado[['venta_id', 'nombre', 'nombre_prov', 'cantidad',
                 'precio_unitario', 'total', 'fecha']])
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 20: Integrador — unir ventas+productos+proveedores.*

1. Unir todo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 3. Resumen Teórico

| Operación | Cuándo usarla |
|-----------|---------------|
| merge | Combinación estándar de DataFrames por columnas clave |
| merge_asof | Unir series temporales sin coincidencia exacta |
| merge_ordered | Series temporales con relleno de gaps |
| validate | Verificar supuestos de integridad referencial |
| indicator | Auditoría: saber de dónde vino cada fila |
| suffixes | Manejar columnas con nombres duplicados |
| compare | Detectar cambios en inventarios o precios |
| combine_first | Rellenar datos faltantes de clientes/productos |
| update | Actualizar stocks, precios, estados |
| concat | Apilar datos de diferentes períodos o sucursales |

---

## 4. Ejercicios Propuestos

**Ejercicio 1:** Usa `merge_asof` para asignar el precio más cercano a cada compra, donde los precios cambian semanalmente.

**Ejercicio 2:** Valida con `validate='one_to_many'` que cada pedido de compra tenga uno o más productos asociados.

**Ejercicio 3:** Usa `indicator=True` en un merge de clientes activos vs antiguos para identificar quiénes están en ambos.

**Ejercicio 4:** Combina ventas de 4 trimestres con `concat` y `keys` para identificar el origen de cada registro.

**Ejercicio 5:** Usa `compare` para detectar diferencias en el inventario entre dos fechas de conteo.

**Ejercicio 6:** Aplica `combine_first` para llenar datos de contacto de clientes desde una base secundaria.

**Ejercicio 7:** Usa `merge_ordered` para combinar ventas diarias con días feriados, rellenando hacia adelante.

**Ejercicio 8:** Crea un pipeline merge que una: ventas → productos → categorías → proveedores, calculando el margen total.

---

*Fin del documento I04 — Merge y Combinación Avanzada en Pandas*
