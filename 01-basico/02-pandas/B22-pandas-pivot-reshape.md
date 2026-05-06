# Módulo B22 — Pandas: Pivot y Reshape

## Teoría

Reestructurar datos es crucial para cambiar entre formatos **ancho** (wide) y **largo** (long):

- **pivot**: pasar de largo a ancho
- **pivot_table**: pivot con agregación
- **melt**: pasar de ancho a largo
- **stack / unstack**: apilar/desapilar niveles de índice
- **crosstab**: tablas de contingencia
- **explode**: expandir listas en filas
- **get_dummies**: codificación one-hot
- **wide_to_long**: formato largo desde wide estructurado

Aplicado a: tabla ventas × sucursal, frecuencias por categoría.

## Setup

import pandas as pd
import numpy as np

ventas = pd.read_csv("../datos/ventas.csv")
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La operación transforma o filtra los datos según la lógica implementada.


---

## Ejemplos

### 1. pivot — de largo a ancho

df_pivot = ventas.head(20).pivot(
    index="fecha",
    columns="sucursal",
    values="ingreso"
)
print(df_pivot)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 1. pivot — de largo a ancho*


### 2. pivot_table con aggfunc

pt = pd.pivot_table(
    ventas,
    index="categoria",
    columns="sucursal",
    values="ingreso",
    aggfunc="sum"
)
print(pt)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 2. pivot_table con aggfunc*


### 3. pivot_table con margins (totales)

pt = pd.pivot_table(
    ventas,
    index="categoria",
    columns="sucursal",
    values="ingreso",
    aggfunc="sum",
    margins=True,
    margins_name="Total"
)
print(pt)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 3. pivot_table con margins (totales)*


### 4. pivot_table con múltiples values y aggfuncs

pt = pd.pivot_table(
    ventas,
    index="categoria",
    columns="sucursal",
    values=["ingreso", "cantidad"],
    aggfunc={"ingreso": "sum", "cantidad": "mean"}
)
print(pt.head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 4. pivot_table con múltiples values y aggfuncs*


### 5. crosstab — frecuencias

ct = pd.crosstab(ventas["categoria"], ventas["sucursal"])
print(ct)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 5. crosstab — frecuencias*


### 6. crosstab con normalize

ct = pd.crosstab(ventas["categoria"], ventas["sucursal"], normalize="index")
print(ct.round(3))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 6. crosstab con normalize*


### 7. crosstab con margins y valores

ct = pd.crosstab(
    ventas["categoria"],
    ventas["sucursal"],
    values=ventas["ingreso"],
    aggfunc="sum",
    margins=True
)
print(ct)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 7. crosstab con margins y valores*


### 8. melt — de ancho a largo

pt = pd.pivot_table(
    ventas,
    index="categoria",
    columns="sucursal",
    values="ingreso",
    aggfunc="sum"
).reset_index()

df_melted = pt.melt(
    id_vars=["categoria"],
    var_name="sucursal",
    value_name="ingreso_total"
)
print(df_melted.head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 8. melt — de ancho a largo*


### 9. melt con value_vars

df_melted = pt.melt(
    id_vars=["categoria"],
    value_vars=["Matriz CDMX", "Sucursal Mérida"],
    var_name="sucursal",
    value_name="ingreso_total"
)
print(df_melted.head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 9. melt con value_vars*


### 10. wide_to_long

# Simular datos wide: ingresos por sucursal como columnas
df_wide = ventas.groupby(["fecha", "sucursal"])["ingreso"].sum().unstack().head(5).reset_index()
df_wide.columns.name = None
# Renombrar sucursales para que tengan prefijo
df_wide = df_wide.rename(columns=lambda x: f"suc_{x}" if x != "fecha" else x)

df_long = pd.wide_to_long(
    df_wide,
    stubnames="suc",
    i="fecha",
    j="sucursal",
    sep="_",
    suffix=".*"
).reset_index()
print(df_long.head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 10. wide_to_long*


### 11. stack — apilar niveles

pt = pd.pivot_table(
    ventas,
    index="categoria",
    columns="sucursal",
    values="ingreso",
    aggfunc="sum"
)
print(pt.stack().head(10))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 11. stack — apilar niveles*


### 12. unstack — desapilar niveles

df_grouped = ventas.groupby(["categoria", "sucursal"])["ingreso"].sum()
print(df_grouped.unstack().head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 12. unstack — desapilar niveles*


### 13. stack con MultiIndex

df_multi = ventas.groupby(["categoria", "sucursal", "mes"])["ingreso"].sum()
print(df_multi.unstack(level=[0, 1]).head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 13. stack con MultiIndex*


### 14. explode — expandir listas

df = pd.DataFrame({
    "producto": ["Laptop", "Monitor"],
    "colores": [["Negro", "Plata"], ["Negro"]]
})
print(df.explode("colores"))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 14. explode — expandir listas*


### 15. get_dummies — one-hot encoding

dummies = pd.get_dummies(ventas["sucursal"].head(5))
print(dummies)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 15. get_dummies — one-hot encoding*


### 16. factorize — codificación numérica

codes, uniques = pd.factorize(ventas["sucursal"])
print("Únicos:", uniques)
print("Códigos (primeros 10):", codes[:10])
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 16. factorize — codificación numérica*


### 17. pivot con aggfunc explícito

pt = ventas.head(50).pivot_table(
    index="producto",
    columns="dia_semana",
    values="ingreso",
    aggfunc=np.sum,
    fill_value=0
)
print(pt)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 17. pivot con aggfunc explícito*


### 18. rename_axis después de pivot

pt = pd.pivot_table(
    ventas,
    index="categoria",
    columns="sucursal",
    values="ingreso",
    aggfunc="sum"
)
pt = pt.rename_axis(columns=None).reset_index()
print(pt.head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 18. rename_axis después de pivot*


### 19. reset_index después de pivot

pt = pd.pivot_table(
    ventas,
    index="categoria",
    columns="sucursal",
    values="ingreso",
    aggfunc="sum"
).reset_index()
print(pt.head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 19. reset_index después de pivot*


### 20. pivot_table con fill_value

pt = pd.pivot_table(
    ventas,
    index="categoria",
    columns="sucursal",
    values="ingreso",
    aggfunc="sum",
    fill_value=0
)
print(pt)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 20. pivot_table con fill_value*


### 21. pivot_table multi-index

pt = pd.pivot_table(
    ventas,
    index=["categoria", "producto"],
    columns="sucursal",
    values="ingreso",
    aggfunc="sum"
)
print(pt.head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 21. pivot_table multi-index*


### 22. crosstab con múltiples filas

ct = pd.crosstab(
    [ventas["categoria"], ventas["dia_semana"]],
    ventas["sucursal"]
)
print(ct.head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 22. crosstab con múltiples filas*


### 23. crosstab con normalize="all"

ct = pd.crosstab(
    ventas["categoria"],
    ventas["sucursal"],
    normalize="all"
)
print(ct.round(4))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 23. crosstab con normalize="all"*


### 24. pivot con index compuesto y dropna

pt = ventas.head(100).pivot_table(
    index=["categoria", "sucursal"],
    columns="dia_semana",
    values="ingreso",
    aggfunc="sum",
    dropna=False
)
print(pt.head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 24. pivot con index compuesto y dropna*


### 25. melt con múltiples id_vars

pt = pd.pivot_table(
    ventas,
    index=["categoria", "producto"],
    columns="sucursal",
    values="ingreso",
    aggfunc="sum"
).reset_index()

df_long = pt.melt(
    id_vars=["categoria", "producto"],
    var_name="sucursal",
    value_name="ingreso"
)
print(df_long.head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 25. melt con múltiples id_vars*


---

## Ejercicios

1. Usa pivot_table para crear matriz categoría × sucursal con suma de ingreso.
2. Añade margins para ver totales por fila y columna.
3. Con crosstab, calcula frecuencias de productos por sucursal.
4. Normaliza el crosstab para ver porcentajes por fila.
5. Con melt, convierte la tabla anterior a formato largo.
6. Usa stack para apilar sucursales como filas adicionales.
7. Aplica get_dummies a la columna "categoria".
8. Usa factorize para codificar "sucursal" numéricamente.

---

## Resumen

- `pivot`: reestructura de largo a ancho (requiere combinación única)
- `pivot_table`: pivot con aggfunc, fill_value, margins
- `crosstab`: tabla de contingencia con normalize, margins
- `melt`: de ancho a largo (id_vars, var_name, value_name)
- `wide_to_long`: para datos wide con prefijos
- `stack`/`unstack`: apilar/desapilar niveles MultiIndex
- `explode`: expandir celdas con listas
- `get_dummies`: one-hot encoding
- `factorize`: codificación entera de categorías