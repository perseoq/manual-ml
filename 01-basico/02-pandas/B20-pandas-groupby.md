# Módulo B20 — Pandas: GroupBy

## Teoría

GroupBy implementa el patrón **split-apply-combine**:
1. **Split**: dividir datos en grupos según una o más variables
2. **Apply**: aplicar una función a cada grupo
3. **Combine**: combinar resultados

Operaciones clave:
- `groupby` una o múltiples columnas
- `agg` con funciones predefinidas o personalizadas
- `transform` — devuelve datos con la misma forma original
- `filter` — filtrar grupos completos
- `apply` — función arbitraria por grupo
- `resample` — agrupación temporal

Aplicado a: ventas por categoría, por sucursal, por mes.

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

### 1. groupby una columna — sum()

print(ventas.groupby("categoria")["ingreso"].sum())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 1. groupby una columna — sum()*


### 2. groupby múltiples columnas

print(ventas.groupby(["categoria", "sucursal"])["ingreso"].sum().head(10))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 2. groupby múltiples columnas*


### 3. agg con múltiples funciones

print(ventas.groupby("categoria")["ingreso"].agg(["sum", "mean", "std", "count"]))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 3. agg con múltiples funciones*


### 4. agg con diccionario

print(ventas.groupby("categoria").agg({
    "ingreso": ["sum", "mean"],
    "cantidad": "sum",
    "margen": "mean"
}).head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 4. agg con diccionario*


### 5. transform — estadísticas por grupo (misma forma)

ventas["promedio_categoria"] = ventas.groupby("categoria")["ingreso"].transform("mean")
print(ventas[["categoria", "ingreso", "promedio_categoria"]].head(10))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 5. transform — estadísticas por grupo (misma forma)*


### 6. transform con función personalizada

ventas["desv_categoria"] = ventas.groupby("categoria")["ingreso"].transform(lambda x: x - x.mean())
print(ventas[["categoria", "ingreso", "desv_categoria"]].head(10))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 6. transform con función personalizada*


### 7. filter — filtrar grupos

# Categorías con ingreso total > 500000
filtro = ventas.groupby("categoria").filter(lambda x: x["ingreso"].sum() > 500000)
print(filtro["categoria"].value_counts())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 7. filter — filtrar grupos*


### 8. apply — función por grupo

def top_productos(grupo):
    return grupo.nlargest(3, "ingreso")[["producto", "ingreso"]]

print(ventas.groupby("categoria").apply(top_productos))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 8. apply — función por grupo*


### 9. Grouper temporal — agrupar por fecha

ventas["fecha"] = pd.to_datetime(ventas["fecha"])
print(ventas.groupby(pd.Grouper(key="fecha", freq="W"))["ingreso"].sum().head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 9. Grouper temporal — agrupar por fecha*


### 10. resample — remuestreo temporal

ventas["fecha"] = pd.to_datetime(ventas["fecha"])
ventas_diarias = ventas.set_index("fecha").resample("W")["ingreso"].sum()
print(ventas_diarias.head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 10. resample — remuestreo temporal*


### 11. first / last / nth

print(ventas.groupby("categoria")["producto"].first())
print(ventas.groupby("categoria")["producto"].last())
print(ventas.groupby("categoria")["producto"].nth(1))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 11. first / last / nth*


### 12. describe por grupo

print(ventas.groupby("categoria")["ingreso"].describe())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 12. describe por grupo*


### 13. nunique — valores únicos por grupo

print(ventas.groupby("categoria")["producto"].nunique())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 13. nunique — valores únicos por grupo*


### 14. value_counts por grupo

print(ventas.groupby("categoria")["sucursal"].value_counts().head(10))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 14. value_counts por grupo*


### 15. idxmax por grupo

idx = ventas.groupby("categoria")["ingreso"].idxmax()
print(ventas.loc[idx, ["categoria", "producto", "ingreso"]])
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 15. idxmax por grupo*


### 16. rank por grupo

ventas["rank_ingreso_categoria"] = ventas.groupby("categoria")["ingreso"].rank(ascending=False)
print(ventas[ventas["rank_ingreso_categoria"] <= 3][["categoria", "producto", "ingreso", "rank_ingreso_categoria"]])
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 16. rank por grupo*


### 17. cumcount — contador por grupo

ventas["orden_venta_categoria"] = ventas.groupby("categoria").cumcount() + 1
print(ventas[["categoria", "producto", "orden_venta_categoria"]].head(10))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 17. cumcount — contador por grupo*


### 18. ngroup — etiqueta de grupo

ventas["grupo_id"] = ventas.groupby("categoria").ngroup()
print(ventas[["categoria", "grupo_id"]].drop_duplicates())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 18. ngroup — etiqueta de grupo*


### 19. agg con funciones personalizadas

def rango_intercuartil(x):
    return x.quantile(0.75) - x.quantile(0.25)

print(ventas.groupby("categoria")["ingreso"].agg(["mean", "std", rango_intercuartil]))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 19. agg con funciones personalizadas*


### 20. groupby + agg con NamedAgg

print(ventas.groupby("categoria").agg(
    ingreso_total=pd.NamedAgg(column="ingreso", aggfunc="sum"),
    margen_prom=pd.NamedAgg(column="margen_pct", aggfunc="mean"),
    ventas_count=pd.NamedAgg(column="sku", aggfunc="count")
))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 20. groupby + agg con NamedAgg*


### 21. groupby + transform con función compleja

ventas["pct_categoria"] = ventas.groupby("categoria")["ingreso"].transform(
    lambda x: x / x.sum() * 100
)
print(ventas[["categoria", "producto", "ingreso", "pct_categoria"]].head(10))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 21. groupby + transform con función compleja*


### 22. groupby con keys múltiples y sort

print(ventas.groupby(["categoria", "sucursal"], sort=False)["ingreso"].sum().head(10))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 22. groupby con keys múltiples y sort*


### 23. groupby por día de semana

print(ventas.groupby("dia_semana")["ingreso"].agg(["sum", "mean"]))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 23. groupby por día de semana*


### 24. groupby con observed=True

ventas["categoria"] = pd.Categorical(ventas["categoria"])
print(ventas.groupby("categoria", observed=True)["ingreso"].sum())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 24. groupby con observed=True*


### 25. groupby anual

ventas["fecha"] = pd.to_datetime(ventas["fecha"])
print(ventas.set_index("fecha").groupby(pd.Grouper(freq="ME"))["ingreso"].sum().head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 25. groupby anual*


### 26. agg con quantile por grupo

print(ventas.groupby("categoria")["ingreso"].agg(["min", "max", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 26. agg con quantile por grupo*


### 27. transform con z-score

ventas["zscore_categoria"] = ventas.groupby("categoria")["ingreso"].transform(
    lambda x: (x - x.mean()) / x.std()
)
print(ventas[["categoria", "ingreso", "zscore_categoria"]].head(10))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 27. transform con z-score*


### 28. value_counts por grupo normalizado

print(ventas.groupby("categoria")["sucursal"].value_counts(normalize=True).head(10))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 28. value_counts por grupo normalizado*


### 29. first y last con timestamp

ventas["fecha"] = pd.to_datetime(ventas["fecha"])
print(ventas.groupby("categoria")["fecha"].first())
print(ventas.groupby("categoria")["fecha"].last())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 29. first y last con timestamp*


### 30. apply con función que devuelve DataFrame

def resumen_grupo(grupo):
    return pd.DataFrame({
        "total_ingreso": [grupo["ingreso"].sum()],
        "promedio": [grupo["ingreso"].mean()],
        "num_ventas": [len(grupo)]
    }, index=[grupo.name])

print(ventas.groupby("categoria").apply(resumen_grupo))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 30. apply con función que devuelve DataFrame*


---

## Ejercicios

1. Calcula el ingreso total por categoría con groupby + sum.
2. Usa agg para obtener sum, mean y count de ingreso por sucursal.
3. Con transform, añade columna con el promedio de margen por categoría.
4. Encuentra el producto de mayor ingreso en cada categoría con idxmax.
5. Filtra las sucursales con ingreso total > 300000.
6. Usa resample semanal para calcular ingreso por semana.
7. Calcula el rank de cada venta dentro de su categoría por ingreso.
8. Usa cumcount para numerar las ventas dentro de cada sucursal.

---

## Resumen

- `groupby`: split-apply-combine para agregación por grupos
- `agg`: múltiples funciones de agregación
- `transform`: mantiene la forma original, útil para añadir columnas
- `filter`: elimina grupos que no cumplen condición
- `apply`: máxima flexibilidad por grupo
- `resample` / `Grouper`: agrupación temporal
- `first`, `last`, `nth`, `nunique`, `value_counts` por grupo
- `idxmax`, `rank`, `cumcount`, `ngroup`: operaciones intra-grupo