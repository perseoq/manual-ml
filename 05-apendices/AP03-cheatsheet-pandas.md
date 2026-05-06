# AP03 — Cheatsheet Pandas

## 1. Series

```python
import pandas as pd
import numpy as np

# Creación
s = pd.Series([10, 20, 30, 40])
s = pd.Series([10, 20, 30], index=["a", "b", "c"])
s = pd.Series({"ene": 100, "feb": 200, "mar": 300})
s = pd.Series(np.random.randn(100))

# Atributos
s.values             # array numpy
s.index              # índice
s.dtype              # tipo de datos
s.shape              # (len,)
s.size               # número de elementos
s.name               # nombre de la serie

# Acceso
s["a"]               # por label
s[0]                 # por posición
s["a":"c"]           # slicing por label (inclusive)
s[0:2]               # slicing por posición (exclusive)
s.loc["a"]           # por label
s.iloc[0]            # por posición
s.take([0, 2])       # por lista de posiciones
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*1. Series.*

1. Creación
2. Atributos
3. Acceso

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 2. DataFrame — Creación

```python
# Desde dict
df = pd.DataFrame({
    "producto": ["Laptop", "Mouse", "Teclado"],
    "precio": [899.99, 25.50, 45.00],
    "stock": [15, 100, 50]
})

# Desde lista de dicts
df = pd.DataFrame([
    {"producto": "Laptop", "precio": 899.99},
    {"producto": "Mouse", "precio": 25.50}
])

# Desde numpy array
df = pd.DataFrame(np.random.randn(100, 4),
                  columns=["a", "b", "c", "d"])

# Desde CSV
df = pd.read_csv("ventas.csv")
df = pd.read_csv("ventas.csv", encoding="utf-8")
df = pd.read_csv("ventas.csv", sep=";", decimal=",")
df = pd.read_csv("ventas.csv", parse_dates=["fecha"])
df = pd.read_csv("ventas.csv", index_col=0)

# Desde Excel
df = pd.read_excel("ventas.xlsx", sheet_name="Enero")
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*2. DataFrame — Creación.*

1. Desde dict
2. Desde lista de dicts
3. Desde numpy array
4. Desde CSV
5. Desde Excel

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 3. DataFrame — Exploración

```python
df.head(10)          # primeras 10 filas
df.tail(5)           # últimas 5 filas
df.sample(3)         # 3 filas aleatorias
df.info()            # resumen: tipos, nulos, memoria
df.describe()        # estadísticas descriptivas
df.describe(include="object")  # para columnas categóricas
df.dtypes            # tipos de cada columna
df.columns           # nombres de columnas
df.index             # índice
df.shape             # (filas, columnas)
df.values            # numpy array subyacente
df.T                 # transpuesta
df.memory_usage()    # memoria por columna
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*3. DataFrame — Exploración.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 4. DataFrame — Selección

```python
# Columnas
df["producto"]           # Series
df[["producto", "precio"]]  # DataFrame
df.precio                # acceso como atributo (si nombre válido)

# Filas por índice
df[0:5]                  # slicing por posición
df[:10]                  # primeras 10
df[-5:]                  # últimas 5

# .loc — por label
df.loc[0]                # fila con índice 0
df.loc[[0, 2, 4]]        # filas específicas
df.loc[0:5]              # rango inclusive
df.loc[0:5, "producto"]  # filas 0-5, columna producto
df.loc[0:5, ["producto", "precio"]]

# .iloc — por posición
df.iloc[0]               # primera fila
df.iloc[[0, 2, 3]]       # filas específicas
df.iloc[0:5]             # rango exclusive
df.iloc[0:5, 0]          # primeras 5 filas, primera columna
df.iloc[0:5, 0:2]        # primeras 5 filas, primeras 2 cols
df.iloc[:, -1]           # última columna

# .at / .iat — acceso rápido a un valor
df.at[0, "producto"]     # valor escalar por label
df.iat[0, 0]             # valor escalar por posición
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*4. DataFrame — Selección.*

1. Columnas
2. Filas por índice
3. .loc — por label
4. .iloc — por posición
5. .at / .iat — acceso rápido a un valor

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 5. DataFrame — Filtros

```python
# Filtros booleanos
df[df["precio"] > 100]
df[(df["precio"] > 50) & (df["stock"] < 30)]
df[(df["precio"] > 500) | (df["categoria"] == "tech")]
df[~df["producto"].isin(["Mouse", "Teclado"])]

# .query (más legible)
df.query("precio > 100")
df.query("precio > 100 and stock < 30")
df.query("categoria == 'tech'")
df.query("precio > precio.mean()")

# .filter (por nombre)
df.filter(items=["producto", "precio"])
df.filter(regex="^pre")        # columnas que empiezan con "pre"
df.filter(like="pre")           # columnas que contienen "pre"

# .where / .mask
df.where(df > 0, 0)            # reemplaza False con 0
df.mask(df < 0, 0)             # reemplaza True con 0
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*5. DataFrame — Filtros.*

1. Filtros booleanos
2. .query (más legible)
3. .filter (por nombre)
4. .where / .mask

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 6. DataFrame — Ordenamiento

```python
# Ordenar por una columna
df.sort_values("precio")
df.sort_values("precio", ascending=False)
df.sort_values(["categoria", "precio"], ascending=[True, False])

# Ordenar por índice
df.sort_index()
df.sort_index(ascending=False)

# rank
df["rank_precio"] = df["precio"].rank()
df["rank_precio"] = df["precio"].rank(method="dense")
df["rank_precio"] = df["precio"].rank(ascending=False)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*6. DataFrame — Ordenamiento.*

1. Ordenar por una columna
2. Ordenar por índice
3. rank

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 7. groupby — Agrupación

```python
# Agrupación simple
df.groupby("categoria")["precio"].mean()
df.groupby("categoria")["precio"].agg(["mean", "std", "count"])

# Múltiples agregaciones
df.groupby("categoria").agg({
    "precio": ["mean", "max", "min"],
    "stock": "sum",
    "producto": "count"
})

# groupby con transform
df["precio_normalizado"] = df.groupby("categoria")["precio"].transform(
    lambda x: (x - x.mean()) / x.std()
)

# groupby con apply
df.groupby("categoria").apply(lambda g: g.sort_values("precio").head(3))

# groupby con múltiples columnas
df.groupby(["categoria", "marca"])["precio"].mean()

# groupby con filtro
df.groupby("categoria").filter(lambda g: g["precio"].mean() > 100)

# groupby + nlargest
df.groupby("categoria")["precio"].nlargest(3)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*7. groupby — Agrupación.*

1. Agrupación simple
2. Múltiples agregaciones
3. groupby con transform
4. groupby con apply
5. groupby con múltiples columnas
6. groupby con filtro
7. groupby + nlargest

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 8. merge — Unión de DataFrames

```python
clientes = pd.DataFrame({"id": [1, 2, 3], "nombre": ["Ana", "Luis", "Pablo"]})
compras = pd.DataFrame({"cliente_id": [1, 2, 1, 3], "monto": [100, 200, 150, 300]})

# Inner join (default)
pd.merge(compras, clientes, left_on="cliente_id", right_on="id")

# Left join
pd.merge(compras, clientes, left_on="cliente_id", right_on="id", how="left")

# Right join
pd.merge(compras, clientes, left_on="cliente_id", right_on="id", how="right")

# Outer join
pd.merge(compras, clientes, left_on="cliente_id", right_on="id", how="outer")

# merge por índice
pd.merge(df1, df2, left_index=True, right_index=True)

# merge con sufijos para columnas duplicadas
pd.merge(df1, df2, on="id", suffixes=("_izq", "_der"))

# .join (merge por índice)
df1.join(df2, how="inner")

# .concat (apilar)
pd.concat([df1, df2])              # apilar filas
pd.concat([df1, df2], axis=1)      # apilar columnas
pd.concat([df1, df2], ignore_index=True)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*8. merge — Unión de DataFrames.*

1. Inner join (default)
2. Left join
3. Right join
4. Outer join
5. merge por índice
6. merge con sufijos para columnas duplicadas
7. .join (merge por índice)
8. .concat (apilar)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 9. pivot — Tablas Dinámicas

```python
# Pivot simple
df.pivot(index="fecha", columns="producto", values="ventas")

# Pivot table (puede agregar)
pd.pivot_table(df, values="ventas", index="fecha",
               columns="producto", aggfunc="sum")
pd.pivot_table(df, values="ventas", index=["año", "mes"],
               columns="categoria", aggfunc=["sum", "mean"])
pd.pivot_table(df, values="ventas", index="fecha",
               columns="producto", fill_value=0, margins=True)

# melt (inverso de pivot)
pd.melt(df, id_vars=["fecha"], value_vars=["Laptop", "Mouse"],
        var_name="producto", value_name="ventas")

# stack / unstack
df.stack()           # columnas a filas
df.unstack()         # filas a columnas
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*9. pivot — Tablas Dinámicas.*

1. Pivot simple
2. Pivot table (puede agregar)
3. melt (inverso de pivot)
4. stack / unstack

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 10. Fechas y Tiempos

```python
# Parsear fechas
df["fecha"] = pd.to_datetime(df["fecha"])
df["fecha"] = pd.to_datetime(df["fecha"], format="%Y-%m-%d")
df["fecha"] = pd.to_datetime(df[["año", "mes", "dia"]])

# Crear rangos de fechas
pd.date_range("2024-01-01", periods=10, freq="D")     # diario
pd.date_range("2024-01-01", "2024-12-31", freq="W")   # semanal
pd.date_range("2024-01-01", periods=12, freq="M")     # mensual
pd.date_range("2024-01-01", periods=4, freq="Q")      # trimestral
pd.date_range("2024-01-01", periods=3, freq="Y")      # anual
pd.bdate_range("2024-01-01", periods=10)               # días hábiles

# Componentes de fecha
df["fecha"].dt.year
df["fecha"].dt.month
df["fecha"].dt.day
df["fecha"].dt.dayofweek        # lunes=0
df["fecha"].dt.day_name()       # nombre del día
df["fecha"].dt.quarter           # trimestre
df["fecha"].dt.is_month_start
df["fecha"].dt.is_month_end

# Setear como índice
df = df.set_index("fecha")
df.index.year
df.index.month

# Resample
df.resample("D").sum()           # diario
df.resample("W").mean()          # semanal
df.resample("M").agg({"ventas": "sum", "precio": "mean"})
df.resample("Q").max()
df.resample("Y").first()

# Shift y diff
df["ventas_lag1"] = df["ventas"].shift(1)
df["ventas_diff"] = df["ventas"].diff()
df["pct_change"] = df["ventas"].pct_change()
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*10. Fechas y Tiempos.*

1. Parsear fechas
2. Crear rangos de fechas
3. Componentes de fecha
4. Setear como índice
5. Resample
6. Shift y diff

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 11. Valores Nulos

```python
# Detectar nulos
df.isna()                # DataFrame booleano
df.isnull()              # alias
df.notna()
df.isna().sum()          # nulos por columna
df.isna().sum().sum()    # total nulos

# Eliminar nulos
df.dropna()              # filas con cualquier nulo
df.dropna(how="all")     # filas completamente nulas
df.dropna(thresh=3)      # filas con al menos 3 no nulos
df.dropna(subset=["precio"])  # nulos solo en columnas específicas
df.dropna(axis=1)        # columnas con nulos

# Rellenar nulos
df.fillna(0)
df.fillna(df.mean())
df.fillna(df.median())
df.fillna(method="ffill")   # forward fill
df.fillna(method="bfill")   # backward fill
df.interpolate()            # interpolación lineal
df["precio"].fillna(df.groupby("categoria")["precio"].transform("mean"))
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*11. Valores Nulos.*

1. Detectar nulos
2. Eliminar nulos
3. Rellenar nulos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 12. Columnas — Crear y Modificar

```python
# Crear nuevas columnas
df["total"] = df["precio"] * df["cantidad"]
df["descuento"] = df["precio"] * 0.1
df["con_iva"] = df["total"] * 1.16

# .assign (encadenable)
df.assign(
    total=lambda d: d.precio * d.cantidad,
    con_iva=lambda d: d.total * 1.16
)

# Condicional
df["segmento"] = np.where(df["precio"] > 100, "alto", "bajo")
df["segmento"] = df["precio"].apply(
    lambda x: "alto" if x > 100 else "medio" if x > 50 else "bajo"
)

# map
df["categoria_num"] = df["categoria"].map({"tech": 1, "ropa": 2, "hogar": 3})

# Renombrar
df.rename(columns={"precio": "precio_unitario"}, inplace=True)
df.columns = [c.upper() for c in df.columns]
df.rename(index={0: "primero"})

# Eliminar columnas
df.drop("columna", axis=1, inplace=True)
df.drop(["col1", "col2"], axis=1)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*12. Columnas — Crear y Modificar.*

1. Crear nuevas columnas
2. .assign (encadenable)
3. Condicional
4. map
5. Renombrar
6. Eliminar columnas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 13. apply — Aplicar Funciones

```python
# apply a una columna
df["precio_iva"] = df["precio"].apply(lambda x: x * 1.16)

# apply a múltiples columnas
df["total"] = df.apply(lambda row: row["precio"] * row["cantidad"], axis=1)

# apply a todo el DataFrame
df.apply(np.sum)           # suma por columna
df.apply(np.sum, axis=1)   # suma por fila
df.apply(pd.Series.min)    # mínimo por columna

# applymap (element-wise)
df[["precio", "cantidad"]].applymap(lambda x: f"${x:.2f}")

# map (solo Series)
df["mes"].map({"ene": 1, "feb": 2})

# transform
df.groupby("categoria")["precio"].transform("mean")
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*13. apply — Aplicar Funciones.*

1. apply a una columna
2. apply a múltiples columnas
3. apply a todo el DataFrame
4. applymap (element-wise)
5. map (solo Series)
6. transform

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 14. Duplicados

```python
# Detectar duplicados
df.duplicated()                  # filas duplicadas
df.duplicated(subset=["producto"])  # por columna específica
df.duplicated(keep=False)        # marcar todos los duplicados

# Eliminar duplicados
df.drop_duplicates()
df.drop_duplicates(subset=["producto"])
df.drop_duplicates(keep="last")  # mantener última ocurrencia
df.drop_duplicates(keep=False)   # eliminar todos los duplicados
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*14. Duplicados.*

1. Detectar duplicados
2. Eliminar duplicados

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 15. Entrada/Salida

```python
# CSV
df.to_csv("ventas.csv", index=False)
df.to_csv("ventas.csv", sep=";", decimal=",")
df.to_csv("ventas.csv", encoding="utf-8-sig")  # para Excel

# Excel
df.to_excel("ventas.xlsx", sheet_name="Enero", index=False)
with pd.ExcelWriter("reporte.xlsx") as writer:
    df_ene.to_excel(writer, sheet_name="Enero")
    df_feb.to_excel(writer, sheet_name="Febrero")

# JSON
df.to_json("ventas.json", orient="records", indent=2)
df.to_json("ventas.json", orient="columns")
df.to_json("ventas.json", orient="split")

# Parquet (eficiente para grandes volúmenes)
df.to_parquet("ventas.parquet")
df = pd.read_parquet("ventas.parquet")

# Feather
df.to_feather("ventas.feather")
df = pd.read_feather("ventas.feather")

# HTML (para reportes)
df.to_html("reporte.html")

# Clipboard (copiar/pegar)
df.to_clipboard()
df = pd.read_clipboard()
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*15. Entrada/Salida.*

1. CSV
2. Excel
3. JSON
4. Parquet (eficiente para grandes volúmenes)
5. Feather
6. HTML (para reportes)
7. Clipboard (copiar/pegar)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## Referencia Rápida

| Operación | Código |
|-----------|--------|
| Leer CSV | `pd.read_csv("archivo.csv")` |
| Guardar CSV | `df.to_csv("archivo.csv", index=False)` |
| Ver primeras filas | `df.head()` |
| Info del DataFrame | `df.info()` |
| Estadísticas | `df.describe()` |
| Seleccionar columna | `df["col"]` o `df.col` |
| Filtrar filas | `df[df["col"] > valor]` |
| Groupby + agg | `df.groupby("col")["val"].agg(["mean","sum"])` |
| Merge | `pd.merge(df1, df2, on="id")` |
| Pivot table | `pd.pivot_table(df, values="v", index="i", columns="c")` |
| Fecha a datetime | `pd.to_datetime(df["col"])` |
| Resample | `df.resample("M").sum()` |
| Fill NA | `df.fillna(df.mean())` |
| Drop NA | `df.dropna()` |
| apply | `df["col"].apply(lambda x: x*2)` |
