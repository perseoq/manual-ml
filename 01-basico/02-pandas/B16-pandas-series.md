# Módulo B16 — Pandas Series

## Teoría

Una **Series** es la estructura fundamental de pandas: un arreglo unidimensional etiquetado. Puede contener cualquier tipo de dato (int, float, str, object, etc.) y sus elementos se acceden mediante un índice.

Características clave:
- **Index**: cada elemento tiene una etiqueta única (por defecto 0..n-1)
- **name**: la serie puede tener un nombre que la identifica
- **Operaciones vectorizadas**: las operaciones aritméticas se aplican elemento a elemento
- **Métodos integrados** para análisis estadístico, ordenamiento, filtrado

Usaremos datos del dataset de ventas para practicar con precios, cantidades y márgenes.

## Setup

import pandas as pd
import numpy as np

# Cargar datos reales
ventas = pd.read_csv("../datos/ventas.csv")
print(ventas.shape)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.


---

## Ejemplos

### 1. Crear Series desde una lista

precios = [15000, 7200, 3400, 1400, 650]
s = pd.Series(precios, name="precios")
print(s)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 1. Crear Series desde una lista*


### 2. Crear Series desde un diccionario

inventario = {"Laptop Pro 15": 17, "Monitor 27 4K": 123, "Teclado Mecánico": 60}
s = pd.Series(inventario, name="stock_actual")
print(s)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 2. Crear Series desde un diccionario*


### 3. Crear Series desde un array de NumPy

arr = np.array([15000, 7200, 3400, 1400, 650])
s = pd.Series(arr, index=["A", "B", "C", "D", "E"], name="precios")
print(s)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 3. Crear Series desde un array de NumPy*


### 4. El atributo index

precios = ventas["precio_unitario"].head(10)
print("Índice:", precios.index)
print("Valores:", precios.values)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 4. El atributo index*


### 5. El atributo name

cantidades = ventas["cantidad"]
cantidades.name = "unidades_vendidas"
print(cantidades.name)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 5. El atributo name*


### 6. Operaciones aritméticas entre series

ingreso = ventas["ingreso"].head(5)
costo = ventas["costo_total"].head(5)
margen_calculado = ingreso - costo
print(margen_calculado)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 6. Operaciones aritméticas entre series*


### 7. Multiplicar series (precio × cantidad)

precio = ventas["precio_unitario"].head(5)
cant = ventas["cantidad"].head(5)
total_estimado = precio * cant
print(total_estimado)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 7. Multiplicar series (precio × cantidad)*


### 8. map — transformar valores

sucursales = ventas["sucursal"].head(10)
abreviado = sucursales.map(lambda x: x.replace("Sucursal ", ""))
print(abreviado)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 8. map — transformar valores*


### 9. apply — función sobre cada elemento

margen = ventas["margen_pct"].head(10)
categoria_margen = margen.apply(lambda x: "Alto" if x > 50 else "Bajo")
print(categoria_margen)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 9. apply — función sobre cada elemento*


### 10. sort_values — ordenar

precios = ventas["precio_unitario"]
print(precios.sort_values(ascending=False).head(10))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 10. sort_values — ordenar*


### 11. head / tail

print(ventas["ingreso"].head(3))
print(ventas["ingreso"].tail(3))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 11. head / tail*


### 12. value_counts — frecuencias

print(ventas["categoria"].value_counts())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 12. value_counts — frecuencias*


### 13. unique — valores únicos

print(ventas["sucursal"].unique())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 13. unique — valores únicos*


### 14. isin — filtro por pertenencia

sucursales_filtro = ventas["sucursal"].isin(["Matriz CDMX", "Sucursal Mérida"])
print(sucursales_filtro.head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 14. isin — filtro por pertenencia*


### 15. between — filtro por rango

precios_medios = ventas["precio_unitario"].between(1000, 5000)
print(precios_medios.head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 15. between — filtro por rango*


### 16. clip — acotar valores

margenes = ventas["margen_pct"].head(20)
acotado = margenes.clip(lower=10, upper=100)
print(acotado)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 16. clip — acotar valores*


### 17. rolling — ventana móvil

ingresos = ventas.groupby("fecha")["ingreso"].sum().head(30)
media_movil = ingresos.rolling(window=7).mean()
print(media_movil.head(10))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 17. rolling — ventana móvil*


### 18. expanding — ventana expansiva

ingresos = ventas.groupby("fecha")["ingreso"].sum().head(30)
acumulado = ingresos.expanding().mean()
print(acumulado.head(10))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 18. expanding — ventana expansiva*


### 19. shift — desplazar valores

precios = ventas["precio_unitario"].head(10)
print(precios.shift(1))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 19. shift — desplazar valores*


### 20. diff — diferencias

precios = ventas["precio_unitario"].head(10)
print(precios.diff())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 20. diff — diferencias*


### 21. pct_change — cambio porcentual

ingresos_diarios = ventas.groupby("fecha")["ingreso"].sum().head(20)
print(ingresos_diarios.pct_change())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 21. pct_change — cambio porcentual*


### 22. Operadores de comparación

cantidades = ventas["cantidad"]
alta_demanda = cantidades > 15
print(alta_demanda.value_counts())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 22. Operadores de comparación*


### 23. Filtro booleano con Series

ingresos = ventas["ingreso"]
print(ingresos[ingresos > 50000].head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 23. Filtro booleano con Series*


### 24. Redondear valores

margen_pct = ventas["margen_pct"].head(5)
print(margen_pct.round(0))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 24. Redondear valores*


### 25. Series con MultiIndex

s = pd.Series(
    [100, 200, 300],
    index=pd.MultiIndex.from_tuples([("Ene", "LAP"), ("Ene", "MON"), ("Feb", "LAP")]),
    name="ventas"
)
print(s)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 25. Series con MultiIndex*


### 26. astype — cambiar tipo

cant = ventas["cantidad"].head(5)
print(cant.astype(float))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 26. astype — cambiar tipo*


### 27. isna / notna — detectar nulos

# Simular nulo
s = pd.Series([100, np.nan, 200, None, 300])
print(s.isna())
print(s.notna())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 27. isna / notna — detectar nulos*


### 28. fillna — rellenar nulos

s = pd.Series([100, np.nan, 200])
print(s.fillna(s.mean()))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 28. fillna — rellenar nulos*


### 29. idxmin / idxmax

precios = ventas["precio_unitario"]
print("Índice del precio mínimo:", precios.idxmin())
print("Índice del precio máximo:", precios.idxmax())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 29. idxmin / idxmax*


### 30. Resumen estadístico

print(ventas["ingreso"].describe())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 30. Resumen estadístico*


---

## Ejercicios

1. Crea una Series con los primeros 10 precios_unitario de ventas. Calcula la media.
2. Usa value_counts para mostrar las 3 categorías con más ventas.
3. Con between, filtra las cantidades entre 5 y 15.
4. Crea una Series de margen_pct y aplica clip entre 0 y 100.
5. Calcula la media móvil de 3 días de los ingresos diarios usando rolling.
6. Usa map para convertir los nombres de sucursal a mayúsculas.
7. Con diff, calcula el cambio día a día de los ingresos diarios.
8. Usa pct_change para ver el crecimiento porcentual diario de ingresos.

---

## Resumen

- **Series**: arreglo 1D etiquetado, base de pandas.
- Se crean desde listas, dicts, arrays, o extrayendo columnas de DataFrame.
- Soportan operaciones vectorizadas (+, -, *, /).
- Métodos clave: `map`, `apply`, `sort_values`, `value_counts`, `unique`, `isin`, `between`, `clip`, `rolling`, `expanding`, `shift`, `diff`, `pct_change`.
- Aplicación directa a columnas de ventas: precios, cantidades, ingresos, márgenes.