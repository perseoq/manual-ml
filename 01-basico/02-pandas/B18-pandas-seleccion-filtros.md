# Módulo B18 — Pandas: Selección y Filtros

## Teoría

Pandas ofrece múltiples formas de seleccionar filas y columnas:

- `[]` — selección por nombre de columna
- `[[]]` — selección múltiple de columnas
- `loc` — selección por etiqueta
- `iloc` — selección por posición entera
- `query` — filtro estilo SQL
- `filter` — filtro por nombre/patrón
- `isin`, `between`, `str.contains` — filtros condicionales
- `isnull`, `dropna`, `duplicated` — manejo de nulos y duplicados

Aplicado a filtrar ventas por sucursal y detectar productos agotados.

## Setup

import pandas as pd
import numpy as np

ventas = pd.read_csv("../datos/ventas.csv")
inventario = pd.read_csv("../datos/inventario.csv")
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La operación transforma o filtra los datos según la lógica implementada.


---

## Ejemplos

### 1. Seleccionar una columna con []

print(ventas["producto"].head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 1. Seleccionar una columna con []*


### 2. Seleccionar múltiples columnas con [[]]

print(ventas[["producto", "ingreso", "margen"]].head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 2. Seleccionar múltiples columnas con [[]]*


### 3. loc — selección por etiqueta de índice

df_idx = ventas.set_index("sku")
print(df_idx.loc["LAP001"])
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 3. loc — selección por etiqueta de índice*


### 4. loc con rango de etiquetas

df_idx = ventas.set_index("sku")
print(df_idx.loc["LAP001":"LAP002", ["producto", "ingreso"]].head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 4. loc con rango de etiquetas*


### 5. iloc — selección por posición

print(ventas.iloc[0:5, [0, 3, 8]])
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 5. iloc — selección por posición*


### 6. iloc con slices

print(ventas.iloc[:5, :4])
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 6. iloc con slices*


### 7. query — filtro estilo SQL

print(ventas.query("ingreso > 50000 and categoria == 'Electrónica'").head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 7. query — filtro estilo SQL*


### 8. filter — columnas por patrón

# El índice debe ser string para filter
df = ventas.head()
print(df.filter(like="precio"))
print(df.filter(regex="^m"))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 8. filter — columnas por patrón*


### 9. isin — filtro por lista

sucursales_filtro = ["Matriz CDMX", "Sucursal Mérida"]
print(ventas[ventas["sucursal"].isin(sucursales_filtro)].head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 9. isin — filtro por lista*


### 10. between — rango de valores

print(ventas[ventas["precio_unitario"].between(3000, 8000)].head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 10. between — rango de valores*


### 11. str.contains — texto en columnas

print(ventas[ventas["producto"].str.contains("Laptop", case=False)].head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 11. str.contains — texto en columnas*


### 12. isnull — detectar nulos

# Compras tiene nulos
print(compras[compras["fecha_entrega"].isnull()].head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 12. isnull — detectar nulos*


### 13. dropna — eliminar nulos

print(compras.dropna(subset=["fecha_entrega"]).head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 13. dropna — eliminar nulos*


### 14. duplicated — duplicados

print(ventas[ventas.duplicated(subset=["sku", "sucursal"])].head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 14. duplicated — duplicados*


### 15. head / tail

print(ventas.head(3))
print(ventas.tail(3))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 15. head / tail*


### 16. sort_values y head combinado

print(ventas.sort_values("ingreso", ascending=False).head(5))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 16. sort_values y head combinado*


### 17. nlargest — top N

print(ventas.nlargest(5, "ingreso")[["producto", "ingreso"]])
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 17. nlargest — top N*


### 18. idxmin / idxmax

print("Producto con mayor margen:", ventas.loc[ventas["margen"].idxmax(), "producto"])
print("Producto con menor margen:", ventas.loc[ventas["margen"].idxmin(), "producto"])
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 18. idxmin / idxmax*


### 19. at / iat — acceso rápido

print(ventas.at[0, "producto"])
print(ventas.iat[0, 2])
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 19. at / iat — acceso rápido*


### 20. take — selección por posiciones

print(ventas.take([0, 5, 10])[["producto", "ingreso"]])
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 20. take — selección por posiciones*


### 21. xs — cross-section en MultiIndex

idx = pd.MultiIndex.from_frame(ventas.head(10)[["sucursal", "producto"]])
df_multi = pd.DataFrame({"ingreso": ventas.head(10)["ingreso"].values}, index=idx)
print(df_multi.xs("Sucursal Mérida", level="sucursal"))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 21. xs — cross-section en MultiIndex*


### 22. Filtro booleano compuesto

filtro = (ventas["categoria"] == "Electrónica") & (ventas["ingreso"] > 50000)
print(ventas[filtro].head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 22. Filtro booleano compuesto*


### 23. Filtro con NOT

print(ventas[~(ventas["categoria"] == "Papelería")].head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 23. Filtro con NOT*


### 24. Seleccionar filas con loc y condición

print(ventas.loc[ventas["sucursal"] == "Matriz CDMX", ["producto", "ingreso", "margen"]].head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 24. Seleccionar filas con loc y condición*


### 25. Eliminar filas con drop

print(ventas.drop([0, 1, 2]).head(3))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 25. Eliminar filas con drop*


### 26. Seleccionar con .loc y slice de índice

df_idx = ventas.set_index("fecha")
print(df_idx.loc["2024-01-01":"2024-01-03", ["producto", "ingreso"]].head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 26. Seleccionar con .loc y slice de índice*


### 27. Filtro combinado con OR

filtro = (ventas["categoria"] == "Electrónica") | (ventas["categoria"] == "Audio")
print(ventas[filtro].head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 27. Filtro combinado con OR*


### 28. Filtro con ~ para negar

print(ventas[~(ventas["descuento"] > 0)].head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 28. Filtro con ~ para negar*


### 29. Seleccionar columnas con .loc y lista

print(ventas.loc[:, ["producto", "categoria", "ingreso", "margen"]].head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 29. Seleccionar columnas con .loc y lista*


### 30. Comparar con query avanzado

print(ventas.query("categoria in ['Electrónica', 'Audio'] and ingreso > 30000").head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 30. Comparar con query avanzado*


---

## Ejercicios

1. Filtra ventas donde la sucursal sea "Matriz CDMX" y el ingreso > 30000.
2. Usa query para obtener filas con margen_pct > 50 y categoría "Electrónica".
3. Selecciona las 5 ventas con mayor ingreso usando nlargest.
4. Encuentra índices de fila con el precio_unitario mínimo y máximo.
5. Usa str.contains para buscar productos que contengan "Teclado".
6. Filtra ventas cuyo precio_unitario esté entre 1000 y 5000 con between.
7. Usa iloc para seleccionar las primeras 5 filas y las columnas 2 a 4.
8. Con at, obtén el producto de la fila 0.

---

## Resumen

- `loc`: selección por etiqueta (incluye slices con etiquetas)
- `iloc`: selección por posición entera
- `[]`/`[[]]`: selección simple/múltiple de columnas
- `query`: filtros SQL-like
- `isin`, `between`, `str.contains`: filtros condicionales
- `isnull`/`dropna`: manejo de nulos
- `duplicated`: identificación de duplicados
- `nlargest`, `idxmin`/`idxmax`: top/bottom valores
- `at`/`iat`: acceso rápido a una celda