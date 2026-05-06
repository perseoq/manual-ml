# Módulo B17 — Pandas DataFrame: Creación

## Teoría

Un **DataFrame** es la estructura bidimensional de pandas: filas y columnas con índice etiquetado. Es similar a una hoja de cálculo o tabla SQL.

Formas de crear un DataFrame:
- Desde estructuras Python (dict de listas, lista de dicts, array 2D)
- Desde archivos (CSV, Excel, JSON, HTML, clipboard)
- Desde bases de datos SQL
- Combinando Series o DataFrames existentes
- Con constructores como `pd.concat`, `assign`, `MultiIndex`

Usaremos datos del catálogo de productos y ventas diarias.

## Setup

import pandas as pd
import numpy as np

ventas = pd.read_csv("../datos/ventas.csv")
inventario = pd.read_csv("../datos/inventario.csv")
compras = pd.read_csv("../datos/compras.csv")
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La operación transforma o filtra los datos según la lógica implementada.


---

## Ejemplos

### 1. Desde dict de listas

data = {
    "producto": ["Laptop Pro 15", "Monitor 27 4K", "Teclado Mecánico"],
    "precio": [15000, 7200, 1400],
    "stock": [17, 123, 60]
}
df = pd.DataFrame(data)
print(df)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 1. Desde dict de listas*


### 2. Desde lista de dicts

data = [
    {"producto": "Laptop Pro 15", "precio": 15000, "stock": 17},
    {"producto": "Monitor 27 4K", "precio": 7200, "stock": 123},
    {"producto": "Teclado Mecánico", "precio": 1400, "stock": 60},
]
df = pd.DataFrame(data)
print(df)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 2. Desde lista de dicts*


### 3. Desde CSV con read_csv

df = pd.read_csv("../datos/ventas.csv")
print(df.head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 3. Desde CSV con read_csv*


### 4. Desde Excel (read_excel)

# df = pd.read_excel("../datos/ventas.xlsx", sheet_name="Ventas")
# print(df.head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 4. Desde Excel (read_excel)*


### 5. Desde JSON (read_json)

# Simulación: convertir a JSON y leer de vuelta
df_json = ventas.head(5).to_json(orient="records")
df = pd.read_json(df_json, orient="records")
print(df)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 5. Desde JSON (read_json)*


### 6. Desde SQL (read_sql)

# import sqlite3
# conn = sqlite3.connect("ventas.db")
# df = pd.read_sql("SELECT * FROM ventas LIMIT 5", conn)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La operación transforma o filtra los datos según la lógica implementada.

*Contexto: 6. Desde SQL (read_sql)*


### 7. Desde array 2D de NumPy

arr = np.array([[15000, 17], [7200, 123], [1400, 60]])
df = pd.DataFrame(arr, columns=["precio", "stock"], index=["LAP", "MON", "TEC"])
print(df)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 7. Desde array 2D de NumPy*


### 8. Desde dict de Series

precios = pd.Series([15000, 7200, 1400], name="precio")
stocks = pd.Series([17, 123, 60], name="stock")
df = pd.DataFrame({"precio": precios, "stock": stocks})
print(df)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 8. Desde dict de Series*


### 9. read_html — leer tablas HTML

# dfs = pd.read_html("https://ejemplo.com/tabla.html")
# print(dfs[0].head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 9. read_html — leer tablas HTML*


### 10. read_clipboard

# Copia datos con Ctrl+C y luego:
# df = pd.read_clipboard()
# print(df)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 10. read_clipboard*


### 11. concat — concatenar DataFrames

df1 = ventas.head(3)
df2 = ventas.iloc[3:6]
df_concat = pd.concat([df1, df2], ignore_index=True)
print(df_concat)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 11. concat — concatenar DataFrames*


### 12. assign — agregar columna calculada

df = ventas.head(5).assign(
    iva=lambda x: x["ingreso"] * 0.16,
    total_con_iva=lambda x: x["ingreso"] + x["iva"]
)
print(df[["ingreso", "iva", "total_con_iva"]])
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 12. assign — agregar columna calculada*


### 13. MultiIndex desde producto y sucursal

arrays = [["LAP", "LAP", "MON", "MON"], ["CDMX", "MTY", "CDMX", "MTY"]]]
index = pd.MultiIndex.from_arrays(arrays, names=["producto", "sucursal"])
df = pd.DataFrame({"ventas": [100, 150, 80, 120]}, index=index)
print(df)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 13. MultiIndex desde producto y sucursal*


### 14. reindex — reordenar filas

df = ventas.head(5).copy()
df_reindex = df.reindex([4, 3, 2, 1, 0])
print(df_reindex[["producto", "ingreso"]])
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 14. reindex — reordenar filas*


### 15. set_index — establecer columna como índice

df = ventas.head(5).set_index("sku")
print(df)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 15. set_index — establecer columna como índice*


### 16. reset_index — restaurar índice por defecto

df = ventas.head(5).set_index("sku").reset_index()
print(df)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 16. reset_index — restaurar índice por defecto*


### 17. rename — renombrar columnas

df = ventas.head(3).rename(columns={
    "precio_unitario": "precio",
    "costo_unitario": "costo"
})
print(df[["producto", "precio", "costo"]])
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 17. rename — renombrar columnas*


### 18. drop — eliminar columnas

df = ventas.head(3).drop(columns=["dia_semana", "mes"])
print(df.columns.tolist())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 18. drop — eliminar columnas*


### 19. sample — muestra aleatoria

print(ventas.sample(n=5, random_state=42)[["producto", "ingreso"]])
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 19. sample — muestra aleatoria*


### 20. from_dict con orient

data = {
    "Ene": {"LAP": 100, "MON": 80},
    "Feb": {"LAP": 120, "MON": 90},
}
df = pd.DataFrame.from_dict(data, orient="index")
print(df)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 20. from_dict con orient*


### 21. Constructor con dtype

df = pd.DataFrame({
    "producto": pd.Categorical(["LAP", "MON", "TEC"]),
    "precio": [15000, 7200, 1400],
    "fecha": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
})
print(df.dtypes)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 21. Constructor con dtype*


### 22. DataFrame desde rangos de fechas

fechas = pd.date_range("2024-01-01", periods=5, freq="D")
df = pd.DataFrame({"fecha": fechas, "ventas": [100, 200, 150, 300, 250]})
print(df)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 22. DataFrame desde rangos de fechas*


### 23. from_records — desde lista de tuplas

data = [("LAP001", 15000, 17), ("MON001", 7200, 123), ("TEC001", 1400, 60)]
df = pd.DataFrame.from_records(data, columns=["sku", "precio", "stock"])
print(df)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 23. from_records — desde lista de tuplas*


### 24. read_fwf — ancho fijo

# from io import StringIO
# data = "LAP001 15000 17\nMON001  7200 123\n"
# df = pd.read_fwf(StringIO(data), widths=[6, 5, 3], names=["sku", "precio", "stock"])
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La operación transforma o filtra los datos según la lógica implementada.

*Contexto: 24. read_fwf — ancho fijo*


### 25. copy — copiar DataFrame

df_original = ventas.head(5)
df_copia = df_original.copy()
df_copia.loc[0, "ingreso"] = 999999
print("Original:\n", df_original[["producto", "ingreso"]].head(1))
print("Copia:\n", df_copia[["producto", "ingreso"]].head(1))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 25. copy — copiar DataFrame*


### 26. filter — filtrar columnas al crear

columnas = ["fecha", "sku", "producto", "ingreso"]
df = ventas[columnas].head(5)
print(df)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 26. filter — filtrar columnas al crear*


### 27. info — inspeccionar DataFrame

print(ventas.info())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 27. info — inspeccionar DataFrame*


### 28. columns — lista de columnas

print(ventas.columns.tolist())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 28. columns — lista de columnas*


### 29. shape — dimensiones

print(f"Filas: {ventas.shape[0]}, Columnas: {ventas.shape[1]}")
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 29. shape — dimensiones*


### 30. dtypes — tipos de datos

print(ventas.dtypes)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 30. dtypes — tipos de datos*


---

## Ejercicios

1. Crea un DataFrame desde un dict de listas con 5 productos del inventario.
2. Lee ventas.csv y muestra las primeras 10 filas.
3. Usa assign para añadir columna "ingreso_neto" = ingreso - costo_total.
4. Concatena las primeras 3 y las últimas 3 filas de ventas.
5. Usa set_index con la columna "sku".
6. Cambia el nombre de la columna "margen_pct" a "margen_porcentual".
7. Elimina las columnas "dia_semana" y "mes" con drop.
8. Toma una muestra aleatoria de 10 filas con sample.

---

## Resumen

- DataFrame es la tabla bidimensional de pandas.
- Se crea desde: dict de listas, lista de dicts, CSV, Excel, JSON, SQL, arrays, concat.
- Manipulación de estructura: `set_index`, `reset_index`, `rename`, `drop`, `assign`, `reindex`.
- MultiIndex permite índices jerárquicos.
- `sample` para obtener subconjuntos aleatorios.
- Diferentes `orient` para `from_dict` y `read_json`.