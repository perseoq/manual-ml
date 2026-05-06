# Módulo B21 — Pandas: Merge y Join

## Teoría

Combinar DataFrames es esencial para trabajar con datos relacionales. Pandas ofrece:

- `merge`: combinación estilo SQL (inner, left, right, outer)
- `join`: combinación por índice
- `concat`: concatenación por filas o columnas
- `merge_asof`: combinación por proximidad temporal
- `merge_ordered`: combinación con orden
- `compare`: diferencias entre DataFrames
- `update`, `combine_first`: fusión de valores

Aplicado a: unir ventas con catálogo, compras con proveedores.

## Setup

import pandas as pd
import numpy as np

ventas = pd.read_csv("../datos/ventas.csv")
compras = pd.read_csv("../datos/compras.csv")
inventario = pd.read_csv("../datos/inventario.csv")
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La operación transforma o filtra los datos según la lógica implementada.


---

## Ejemplos

### 1. merge inner — intersección

df = pd.merge(ventas.head(10), inventario[["sku", "stock_actual"]], on="sku", how="inner")
print(df[["sku", "producto", "stock_actual"]])
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 1. merge inner — intersección*


### 2. merge left — mantener todas las filas izquierda

df = pd.merge(ventas.head(10), inventario[["sku", "stock_actual"]], on="sku", how="left")
print(df[["sku", "producto", "stock_actual"]])
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 2. merge left — mantener todas las filas izquierda*


### 3. merge right — mantener todas las filas derecha

df = pd.merge(ventas.head(10), inventario[["sku", "stock_actual"]], on="sku", how="right")
print(df[["sku", "stock_actual"]].head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 3. merge right — mantener todas las filas derecha*


### 4. merge outer — unión completa

df = pd.merge(ventas.head(10), inventario[["sku", "stock_actual"]], on="sku", how="outer")
print(df.shape)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 4. merge outer — unión completa*


### 5. merge con left_on / right_on (columnas diferentes)

compras_renamed = compras.rename(columns={"sku": "sku_compra"})
df = pd.merge(
    ventas.head(10),
    compras_renamed[["sku_compra", "proveedor", "costo_unitario"]].head(5),
    left_on="sku",
    right_on="sku_compra",
    how="left"
)
print(df[["sku", "sku_compra", "proveedor"]])
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 5. merge con left_on / right_on (columnas diferentes)*


### 6. merge con índice

ventas_idx = ventas.head(10).set_index("sku")
inventario_idx = inventario.set_index("sku")
df = pd.merge(ventas_idx, inventario_idx[["stock_actual"]], left_index=True, right_index=True, how="left")
print(df[["producto", "stock_actual"]])
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 6. merge con índice*


### 7. join — combinación por índice

ventas_idx = ventas.head(10).set_index("sku")
inventario_idx = inventario.set_index("sku")
df = ventas_idx.join(inventario_idx[["stock_actual", "stock_minimo"]], how="left")
print(df[["producto", "stock_actual", "stock_minimo"]])
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 7. join — combinación por índice*


### 8. concat axis=0 — concatenar filas

df1 = ventas.head(3)
df2 = ventas.iloc[3:6]
df_concat = pd.concat([df1, df2], ignore_index=True)
print(df_concat.shape)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 8. concat axis=0 — concatenar filas*


### 9. concat axis=1 — concatenar columnas

df_precios = ventas[["sku", "precio_unitario"]].head(5)
df_margen = ventas[["sku", "margen"]].head(5)
df_concat = pd.concat([df_precios, df_margen], axis=1)
print(df_concat)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 9. concat axis=1 — concatenar columnas*


### 10. merge_asof — combinación por proximidad

# Ordenar por fecha
ventas_ord = ventas.head(20).sort_values("fecha").copy()
compras_ord = compras.head(10).sort_values("fecha_orden").copy()
ventas_ord["fecha"] = pd.to_datetime(ventas_ord["fecha"])
compras_ord["fecha_orden"] = pd.to_datetime(compras_ord["fecha_orden"])

df = pd.merge_asof(
    ventas_ord,
    compras_ord[["fecha_orden", "proveedor"]],
    left_on="fecha",
    right_on="fecha_orden",
    direction="backward"
)
print(df[["fecha", "producto", "proveedor"]].head(10))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 10. merge_asof — combinación por proximidad*


### 11. compare — diferencias entre DataFrames

df1 = ventas.head(3)[["producto", "ingreso"]]
df2 = ventas.head(3)[["producto", "ingreso"]].copy()
df2.loc[0, "ingreso"] = 99999
print(df1.compare(df2))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 11. compare — diferencias entre DataFrames*


### 12. update — actualizar valores

df1 = ventas.head(5)[["sku", "producto", "ingreso"]].copy()
df2 = pd.DataFrame({"sku": ["LAP001"], "ingreso": [999999]})
df1 = df1.set_index("sku")
df2 = df2.set_index("sku")
df1.update(df2)
print(df1.reset_index())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 12. update — actualizar valores*


### 13. combine_first — rellenar nulos

df1 = ventas.head(5)[["sku", "producto", "ingreso"]].copy()
df1.loc[0, "ingreso"] = np.nan
df2 = ventas.head(5)[["sku", "ingreso"]].copy()
df2.loc[0, "ingreso"] = 50000
result = df1.set_index("sku").combine_first(df2.set_index("sku"))
print(result.reset_index().head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 13. combine_first — rellenar nulos*


### 14. merge_ordered — combinación con orden

ventas["fecha"] = pd.to_datetime(ventas["fecha"])
compras["fecha_orden"] = pd.to_datetime(compras["fecha_orden"])
df = pd.merge_ordered(
    ventas.head(10),
    compras.head(10),
    left_on="fecha",
    right_on="fecha_orden",
    fill_method="ffill"
)
print(df[["fecha", "fecha_orden", "producto_x", "proveedor"]])
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 14. merge_ordered — combinación con orden*


### 15. merge con MultiIndex

ventas_mi = ventas.head(10).set_index(["sucursal", "categoria"])
inventario_mi = inventario.set_index(["categoria", "producto"])
# Necesitamos resetear niveles para merge
df = pd.merge(
    ventas_mi.reset_index(),
    inventario_mi.reset_index()[["categoria", "stock_actual"]],
    on="categoria",
    how="left"
)
print(df[["sucursal", "categoria", "stock_actual"]].drop_duplicates())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 15. merge con MultiIndex*


### 16. validate — validar relaciones

try:
    pd.merge(
        ventas.head(10),
        inventario[["sku", "stock_actual"]],
        on="sku",
        validate="m:1"
    )
    print("Merge válido (muchos a uno)")
except Exception as e:
    print(f"Error: {e}")
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 16. validate — validar relaciones*


### 17. indicator — origen de cada fila

df = pd.merge(
    ventas.head(10),
    inventario[["sku", "stock_actual"]],
    on="sku",
    how="outer",
    indicator=True
)
print(df["_merge"].value_counts())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 17. indicator — origen de cada fila*


### 18. suffixes — manejar columnas duplicadas

df = pd.merge(
    ventas.head(5),
    ventas.head(5)[["sku", "ingreso", "margen"]],
    on="sku",
    how="left",
    suffixes=("_venta", "_extra")
)
print(df.columns.tolist())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 18. suffixes — manejar columnas duplicadas*


### 19. merge con múltiples keys

df = pd.merge(
    ventas.head(10),
    inventario[["sku", "categoria", "stock_actual"]],
    on=["sku", "categoria"],
    how="left"
)
print(df[["sku", "categoria", "stock_actual"]])
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 19. merge con múltiples keys*


### 20. concat con keys — mantener origen

df_ene = ventas[ventas["mes"] == 1].head(3)
df_feb = ventas[ventas["mes"] == 2].head(3)
df_concat = pd.concat([df_ene, df_feb], keys=["Enero", "Febrero"])
print(df_concat.index)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 20. concat con keys — mantener origen*


---

## Ejercicios

1. Combina ventas con inventario usando merge en "sku" (left join).
2. Usa merge para traer "stock_actual" y "stock_minimo" a ventas.
3. Concatena las primeras 50 filas con las últimas 50 de ventas.
4. Usa join para combinar dos DataFrames por índice.
5. Con merge_asof une la compra más cercana anterior a cada venta.
6. Usa indicator para saber cuántos skus están en ventas pero no en inventario.
7. Con combine_first rellena valores nulos de ingreso en una copia.
8. Usa merge_ordered para combinar ventas y compras ordenadas por fecha.

---

## Resumen

- `merge`: inner/left/right/outer con on, left_on, right_on, left_index, right_index
- `join`: atajo para merge por índice
- `concat`: axis=0 (filas) o axis=1 (columnas), con keys
- `merge_asof`: combinación por proximidad temporal
- `merge_ordered`: combinación ordenada con fill
- `compare`: encuentra diferencias entre DataFrames
- `update`: modifica en el lugar
- `combine_first`: rellena nulos desde otro DataFrame
- `validate`: verifica relaciones (1:1, 1:m, m:1)
- `indicator`: columna _merge que indica origen