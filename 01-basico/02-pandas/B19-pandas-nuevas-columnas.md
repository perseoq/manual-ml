# Módulo B19 — Pandas: Nuevas Columnas

## Teoría

Crear nuevas columnas es una operación fundamental en el análisis de datos. Pandas ofrece múltiples enfoques:

- **Asignación directa**: `df["nueva"] = valor`
- **Operaciones vectorizadas**: `df["total"] = df["precio"] * df["cant"]`
- `assign`: método encadenable para crear columnas
- `apply`: aplicar función por fila o columna
- `map`, `transform`: transformaciones con diccionarios/funciones
- `cut`, `qcut`: discretización en bins
- `where`, `masked`, `replace`: reemplazo condicional
- `fillna`, `bfill`, `ffill`, `interpolate`: manejo de nulos
- `clip`, `cumsum`, `diff`: operaciones acumulativas

Aplicado a: margen, IVA, categorías de precio.

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

### 1. Columna calculada directa

ventas["ingreso_bruto"] = ventas["precio_unitario"] * ventas["cantidad"]
print(ventas[["producto", "cantidad", "precio_unitario", "ingreso_bruto"]].head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 1. Columna calculada directa*


### 2. Columna con operación entre columnas

ventas["margen_calculado"] = ventas["ingreso"] - ventas["costo_total"]
print(ventas[["producto", "ingreso", "costo_total", "margen_calculado"]].head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 2. Columna con operación entre columnas*


### 3. assign — método encadenable

df = ventas.head(5).assign(
    iva=lambda x: round(x["ingreso"] * 0.16, 2),
    total_factura=lambda x: x["ingreso"] + x["iva"]
)
print(df[["producto", "ingreso", "iva", "total_factura"]])
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 3. assign — método encadenable*


### 4. assign con múltiples dependencias

df = ventas.head(5).assign(
    descuento_aplicado=lambda x: x["precio_unitario"] * x["descuento"],
    precio_final=lambda x: x["precio_unitario"] - x["descuento_aplicado"]
)
print(df[["producto", "precio_unitario", "descuento", "precio_final"]])
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 4. assign con múltiples dependencias*


### 5. apply en columna

ventas["categoria_margen"] = ventas["margen_pct"].apply(
    lambda x: "Alto" if x > 70 else ("Medio" if x > 40 else "Bajo")
)
print(ventas[["producto", "margen_pct", "categoria_margen"]].head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 5. apply en columna*


### 6. apply en filas (axis=1)

ventas["rentabilidad"] = ventas.apply(
    lambda row: "Alta" if row["margen"] > 10000 and row["margen_pct"] > 50 else "Normal",
    axis=1
)
print(ventas[["producto", "margen", "margen_pct", "rentabilidad"]].head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 6. apply en filas (axis=1)*


### 7. map con diccionario

mapa_descuento = {0: "Sin desc.", 0.05: "5%", 0.10: "10%", 0.15: "15%", 0.20: "20%"}
ventas["desc_etiqueta"] = ventas["descuento"].map(mapa_descuento)
print(ventas[["descuento", "desc_etiqueta"]].head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 7. map con diccionario*


### 8. map con función

ventas["sucursal_corta"] = ventas["sucursal"].map(lambda x: x.replace("Sucursal ", ""))
print(ventas[["sucursal", "sucursal_corta"]].head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 8. map con función*


### 9. transform — estadísticas por grupo

ventas["promedio_categoria"] = ventas.groupby("categoria")["ingreso"].transform("mean")
print(ventas[["categoria", "ingreso", "promedio_categoria"]].head(10))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 9. transform — estadísticas por grupo*


### 10. transform con función personalizada

ventas["desviacion_categoria"] = ventas.groupby("categoria")["ingreso"].transform(lambda x: x - x.mean())
print(ventas[["categoria", "ingreso", "desviacion_categoria"]].head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 10. transform con función personalizada*


### 11. pipe — pipeline de funciones

def add_iva(df):
    return df.assign(iva=df["ingreso"] * 0.16)

def add_total(df):
    return df.assign(total=df["ingreso"] + df["iva"])

resultado = ventas.head(5).pipe(add_iva).pipe(add_total)
print(resultado[["producto", "ingreso", "iva", "total"]])
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 11. pipe — pipeline de funciones*


### 12. eval — expresiones rápidas

df = ventas.head(5).copy()
df.eval("margen_estimado = ingreso - costo_total", inplace=True)
print(df[["producto", "ingreso", "costo_total", "margen_estimado"]])
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 12. eval — expresiones rápidas*


### 13. cut — discretizar en bins

ventas["rango_precio"] = pd.cut(
    ventas["precio_unitario"],
    bins=[0, 1000, 5000, 10000, 20000],
    labels=["Económico", "Medio", "Premium", "Ultra Premium"]
)
print(ventas[["producto", "precio_unitario", "rango_precio"]].head(10))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 13. cut — discretizar en bins*


### 14. qcut — cuantiles

ventas["cuartil_ingreso"] = pd.qcut(
    ventas["ingreso"],
    q=4,
    labels=["Q1", "Q2", "Q3", "Q4"]
)
print(ventas[["producto", "ingreso", "cuartil_ingreso"]].head(10))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 14. qcut — cuantiles*


### 15. where — reemplazo condicional

ventas["ingreso_con_descuento"] = ventas["ingreso"].where(
    ventas["descuento"] == 0,
    ventas["ingreso"] * (1 - ventas["descuento"])
)
print(ventas[["ingreso", "descuento", "ingreso_con_descuento"]].head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 15. where — reemplazo condicional*


### 16. masked — reemplazo inverso a where

ventas["precio_promocion"] = ventas["precio_unitario"].mask(
    ventas["precio_unitario"] < 5000,
    ventas["precio_unitario"] * 0.9
)
print(ventas[["producto", "precio_unitario", "precio_promocion"]].head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 16. masked — reemplazo inverso a where*


### 17. replace — reemplazar valores

df = ventas.head(5).copy()
df["sucursal"] = df["sucursal"].replace({"Matriz CDMX": "CDMX", "Sucursal Mérida": "MID"})
print(df[["sucursal"]])
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 17. replace — reemplazar valores*


### 18. fillna — rellenar nulos

compras["retraso_sin_nulos"] = compras["retraso"].fillna(0)
print(compras[["retraso", "retraso_sin_nulos"]].head(10))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 18. fillna — rellenar nulos*


### 19. bfill / ffill

s = pd.Series([100, np.nan, np.nan, 200, np.nan, 300])
print("ffill:", s.ffill().tolist())
print("bfill:", s.bfill().tolist())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 19. bfill / ffill*


### 20. interpolate

s = pd.Series([100, np.nan, np.nan, 200])
print(s.interpolate().tolist())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 20. interpolate*


### 21. clip — acotar valores

ventas["margen_acotado"] = ventas["margen"].clip(lower=0, upper=50000)
print(ventas[["margen", "margen_acotado"]].head(10))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 21. clip — acotar valores*


### 22. cumsum — acumulado

ventas["ingreso_acumulado"] = ventas["ingreso"].cumsum()
print(ventas[["fecha", "ingreso", "ingreso_acumulado"]].head(10))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 22. cumsum — acumulado*


### 23. diff — diferencia entre filas consecutivas

ventas["cambio_ingreso"] = ventas["ingreso"].diff()
print(ventas[["fecha", "ingreso", "cambio_ingreso"]].head(10))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 23. diff — diferencia entre filas consecutivas*


### 24. Columna con categorías y apply

def clasificar_producto(row):
    if row["precio_unitario"] > 10000:
        return "Gama Alta"
    elif row["precio_unitario"] > 3000:
        return "Gama Media"
    else:
        return "Gama Baja"

ventas["gama"] = ventas.apply(clasificar_producto, axis=1)
print(ventas[["producto", "precio_unitario", "gama"]].head(10))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 24. Columna con categorías y apply*


### 25. Asignación múltiple con assign

df = ventas.head(5).assign(
    precio_sin_iva=lambda x: round(x["precio_unitario"] / 1.16, 2),
    iva_unitario=lambda x: round(x["precio_unitario"] - x["precio_sin_iva"], 2)
)
print(df[["producto", "precio_unitario", "precio_sin_iva", "iva_unitario"]])
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 25. Asignación múltiple con assign*


---

## Ejercicios

1. Crea columna "total_linea" = cantidad * precio_unitario.
2. Usa assign para añadir "iva" (16% del ingreso) y "total_con_iva".
3. Con cut, categoriza "precio_unitario" en 3 categorías: Bajo, Medio, Alto.
4. Con apply, crea columna "es_caro" que sea True si precio > 5000.
5. Usa groupby + transform para añadir el promedio de ingreso por categoría.
6. Reemplaza con fillna los nulos de la columna "retraso" en compras con 0.
7. Aplica clip al margen para que no baje de 0 ni supere 100000.
8. Usa cumsum para calcular el ingreso acumulado por fecha.

---

## Resumen

- **Asignación directa**: `df["col"] = expr`
- `assign`: método funcional encadenable
- `apply`: flexibilidad total por fila/columna
- `map`: transformación uno-a-uno con dict/función
- `transform`: estadísticas por grupo que preservan forma
- `pipe`: composición de funciones
- `eval`: expresiones rápidas
- `cut`/`qcut`: discretización en bins uniformes o por cuantil
- `where`/`masked`: reemplazo condicional
- `fillna`/`bfill`/`ffill`/`interpolate`: manejo de valores nulos
- `clip`, `cumsum`, `diff`: operaciones acumulativas y de rango