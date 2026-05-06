# I08 — Funciones Avanzadas de Pandas

## 1. Introducción Teórica

Pandas ofrece un conjunto de funciones avanzadas que permiten escribir código más rápido, eficiente en memoria y legible. Este módulo cubre desde evaluación de expresiones hasta optimización de tipos de datos, pasando por encadenamiento de operaciones, mapeo de valores, y discretización.

### Funciones clave:

- **`pd.eval()`**: Evalúa expresiones de cadena usando `numexpr` (más rápido que operaciones nativas en DataFrames grandes)
- **`df.query()`**: Filtra filas usando expresiones de cadena, soporta variables externas con `@`
- **`df.pipe()`**: Encadena funciones personalizadas en un pipeline fluido
- **`df.assign()`**: Crea múltiples columnas nuevas en cadena
- **`df.transform()`**: Aplica funciones que devuelven el mismo tamaño que el input (por grupo o columna)
- **`df.map()` / `Series.map()`**: Transforma valores uno a uno mediante un diccionario o función
- **`df.applymap()`**: Aplica función a cada celda del DataFrame
- **`df.categorize()`**: Convierte columnas a tipo category (ahorra memoria)
- **`df.infer_objects()`** / **`df.convert_dtypes()`**: Inferencia y conversión de tipos moderna
- **`df.filter()`**: Selecciona columnas por nombre, like o regex
- **`df.select_dtypes()`**: Selecciona columnas por tipo de dato
- **`df.memory_usage()`**: Mide consumo de memoria por columna
- **`pd.cut()`** / **`pd.qcut()`**: Discretización en intervalos fijos o por cuantiles

---

## 2. Ejemplos Prácticos

### Ejemplo 1: pd.eval — Expresiones rápidas en grandes DataFrames

```python
import pandas as pd
import numpy as np

# Cargar datos
ventas = pd.read_csv("../../datos/ventas.csv")

# pd.eval evalúa expresiones sin crear DataFrames intermedios
# Útil para operaciones en tablas grandes
resultado = pd.eval("ventas['ingreso'] - ventas['costo_total']")
print("Margen calculado con pd.eval (primeros 5):", resultado.head().values)
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

*Ejemplo 1: pd.eval — Expresiones rápidas en grandes DataFrames.*

1. Cargar datos
2. pd.eval evalúa expresiones sin crear DataFrames intermedios
3. Útil para operaciones en tablas grandes

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: df.query con @ para variables externas

```python
# Variable externa definida en Python
margen_minimo = 1000

# Usamos @ para referenciar variable externa
altos_margenes = ventas.query("margen > @margen_minimo and cantidad > 5")
print(f"Ventas con margen > ${margen_minimo} y cantidad > 5: {len(altos_margenes)} registros")
print(altos_margenes[["producto", "margen", "cantidad"]].head())
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

*Ejemplo 2: df.query con @ para variables externas.*

1. Variable externa definida en Python
2. Usamos @ para referenciar variable externa

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: df.pipe — Pipeline de transformaciones

```python
def limpiar_precios(df):
    return df[df["precio_unitario"] > 0].copy()

def filtrar_margen_alto(df, umbral=0.15):
    return df[df["margen_pct"] > umbral * 100].copy()

def agregar_metricas(df):
    return df.assign(
        valor_total=df["precio_unitario"] * df["cantidad"],
        rentabilidad=df["margen"] / (df["ingreso"] + 1)
    )

resultado_pipe = (ventas
    .pipe(limpiar_precios)
    .pipe(filtrar_margen_alto, umbral=0.20)
    .pipe(agregar_metricas)
)
print(f"Pipeline: {len(resultado_pipe)} registros filtrados")
print(resultado_pipe[["producto", "valor_total", "rentabilidad"]].head(3))
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

*Ejemplo 3: df.pipe — Pipeline de transformaciones.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: df.assign con múltiples columnas en cadena

```python
ventas_enriquecido = ventas.assign(
    ingreso_neto=ventas["ingreso"] - ventas["costo_total"],
    iva=ventas["ingreso"] * 0.16,
    utilidad_neta=lambda df: df["ingreso_neto"] - df["iva"],
    eficiencia=lambda df: df["utilidad_neta"] / (df["ingreso"] + 1)
)
print(ventas_enriquecido[["producto", "ingreso", "ingreso_neto", "iva", "utilidad_neta", "eficiencia"]].head(3))
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

*Ejemplo 4: df.assign con múltiples columnas en cadena.*

1. `print(ventas_enriquecido[["producto", "ingreso", "ingreso_neto", "iva", "utilidad_neta", "eficiencia"]].head(3))` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: df.transform con función lambda personalizada

```python
# Normalización z-score sobre margen
ventas["margen_zscore"] = ventas.groupby("categoria")["margen"].transform(
    lambda x: (x - x.mean()) / x.std()
)
print(ventas[["producto", "categoria", "margen", "margen_zscore"]].head(5))
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

*Ejemplo 5: df.transform con función lambda personalizada.*

1. Normalización z-score sobre margen

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: df.map + Series.map para mapear valores

```python
# Mapear días de semana numéricos a nombres
mapa_dias = {0: "Domingo", 1: "Lunes", 2: "Martes", 3: "Miércoles",
             4: "Jueves", 5: "Viernes", 6: "Sábado"}
ventas["dia_nombre"] = ventas["dia_semana"].map(mapa_dias)

# Mapear sku a nombre de producto
mapa_productos = ventas.set_index("sku")["producto"].to_dict()
ventas["producto_desde_sku"] = ventas["sku"].map(mapa_productos)

print(ventas[["sku", "producto", "producto_desde_sku", "dia_semana", "dia_nombre"]].head(3))
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

*Ejemplo 6: df.map + Series.map para mapear valores.*

1. Mapear días de semana numéricos a nombres
2. Mapear sku a nombre de producto

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: df.applymap — Limpiar formato de todas las celdas

```python
# Seleccionar solo columnas numéricas
numéricas = ventas.select_dtypes(include=[np.number])

# Redondear todas las celdas numéricas a 1 decimal
redondeado = numéricas.iloc[:5].applymap(lambda x: round(x, 1))
print(redondeado.head())
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

*Ejemplo 7: df.applymap — Limpiar formato de todas las celdas.*

1. Seleccionar solo columnas numéricas
2. Redondear todas las celdas numéricas a 1 decimal

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: .categorize — Reducir memoria con tipos categóricos

```python
print("Memoria antes:", ventas.memory_usage(deep=True).sum(), "bytes")

ventas_opt = ventas.copy()
for col in ["categoria", "sucursal", "producto", "sku", "dia_semana", "mes"]:
    if col in ventas_opt.columns:
        ventas_opt[col] = ventas_opt[col].astype("category")

print("Memoria después:", ventas_opt.memory_usage(deep=True).sum(), "bytes")
print(f"Ahorro: {(1 - ventas_opt.memory_usage(deep=True).sum() / ventas.memory_usage(deep=True).sum()) * 100:.1f}%")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: .categorize — Reducir memoria con tipos categóricos.*

1. `print("Memoria antes:", ventas.memory_usage(deep=True).sum(), "bytes")` — Muestra el resultado por pantalla.
2. `print("Memoria después:", ventas_opt.memory_usage(deep=True).sum(), "bytes")` — Muestra el resultado por pantalla.
3. `print(f"Ahorro: {(1 - ventas_opt.memory_usage(deep=True).sum() / ventas.memory_usage(deep=True).sum()) * 100:.1f}%")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: df.infer_objects — Inferir tipos de objeto

```python
df_mixto = pd.DataFrame({
    "precio": ["15000", "3200", "1400", "650"],
    "cantidad": ["8", "3", "13", "2"],
    "fecha": ["2024-01-01", "2024-01-01", "2024-01-01", "2024-01-02"]
})
print("Tipos originales:")
print(df_mixto.dtypes)

df_inferido = df_mixto.infer_objects()
print("\nTipos inferidos:")
print(df_inferido.dtypes)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: df.infer_objects — Inferir tipos de objeto.*

1. `print("Tipos originales:")` — Muestra el resultado por pantalla.
2. `print(df_mixto.dtypes)` — Muestra el resultado por pantalla.
3. `print("\nTipos inferidos:")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: df.convert_dtypes — Convertir a tipos modernos

```python
# convert_dtypes usa los tipos nullable de pandas (Int64, string, etc.)
ventas_moderno = ventas.convert_dtypes()
print("Tipos convertidos:")
print(ventas_moderno.dtypes)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: df.convert_dtypes — Convertir a tipos modernos.*

1. convert_dtypes usa los tipos nullable de pandas (Int64, string, etc.)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: Boolean indexing — Combinación de múltiples condiciones

```python
# Combinación avanzada con & (y), | (o), ~ (no)
alto_valor = ventas["ingreso"] > 50000
alta_cantidad = ventas["cantidad"] > 10
electronica = ventas["categoria"] == "Electrónica"
no_perifericos = ventas["categoria"] != "Periféricos"

seleccion = ventas[alto_valor & alta_cantidad & (electronica | no_perifericos)]
print(f"Registros que cumplen todas las condiciones: {len(seleccion)}")
print(seleccion[["producto", "categoria", "ingreso", "cantidad"]].head())
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

*Ejemplo 11: Boolean indexing — Combinación de múltiples condiciones.*

1. Combinación avanzada con & (y), | (o), ~ (no)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: df.filter con like y regex

```python
# like: columnas que contienen cierta cadena
columnas_precio = ventas.filter(like="precio")
print("Columnas con 'precio':", list(columnas_precio.columns))

# regex: columnas que terminan con 'o' o contienen 'margen'
columnas_regex = ventas.filter(regex="(o$|margen)")
print("Columnas con regex (terminan en 'o' o contienen 'margen'):", list(columnas_regex.columns))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: df.filter con like y regex.*

1. like: columnas que contienen cierta cadena
2. regex: columnas que terminan con 'o' o contienen 'margen'

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: df.select_dtypes — Seleccionar columnas por tipo

```python
numéricas = ventas.select_dtypes(include=[np.number])
print(f"Columnas numéricas ({len(numéricas.columns)}):", list(numéricas.columns))

object_cols = ventas.select_dtypes(include=["object"])
print(f"Columnas tipo object ({len(object_cols.columns)}):", list(object_cols.columns))

# Excluir algunos tipos
sin_int = ventas.select_dtypes(exclude=["int64"])
print(f"Columnas excluyendo int64 ({len(sin_int.columns)}):", list(sin_int.columns))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: df.select_dtypes — Seleccionar columnas por tipo.*

1. Excluir algunos tipos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: df.memory_usage(deep=True) — Memoria real de cada columna

```python
memoria = ventas.memory_usage(deep=True)
print("Memoria por componente:")
print(memoria)
print(f"\nMemoria total: {memoria.sum():,} bytes = {memoria.sum() / 1024:.1f} KB")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: df.memory_usage(deep=True) — Memoria real de cada columna.*

1. `print("Memoria por componente:")` — Muestra el resultado por pantalla.
2. `print(memoria)` — Muestra el resultado por pantalla.
3. `print(f"\nMemoria total: {memoria.sum():,} bytes = {memoria.sum() / 1024:.1f} KB")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: Comparar memoria antes/después de optimizar tipos

```python
antes = ventas.memory_usage(deep=True)

ventas_opt = ventas.copy()
# Optimizar enteros
ventas_opt["cantidad"] = pd.to_numeric(ventas_opt["cantidad"], downcast="integer")
# Optimizar flotantes
for col in ["precio_unitario", "costo_unitario", "ingreso", "costo_total", "margen", "margen_pct"]:
    ventas_opt[col] = pd.to_numeric(ventas_opt[col], downcast="float")
# Optimizar categóricas
ventas_opt["categoria"] = ventas_opt["categoria"].astype("category")
ventas_opt["sucursal"] = ventas_opt["sucursal"].astype("category")

despues = ventas_opt.memory_usage(deep=True)
comparacion = pd.DataFrame({"Antes (bytes)": antes, "Después (bytes)": despues},
                           index=antes.index)
comparacion["Ahorro %"] = ((1 - despues / antes) * 100).round(1)
print(comparacion)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: Comparar memoria antes/después de optimizar tipos.*

1. Optimizar enteros
2. Optimizar flotantes
3. Optimizar categóricas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 16: pd.cut con labels personalizados para rangos de precio

```python
bins = [0, 500, 2000, 5000, 10000, 50000]
labels = ["Económico", "Medio", "Premium", "Alta gama", "Lujo"]

ventas["rango_precio"] = pd.cut(ventas["precio_unitario"],
                                bins=bins, labels=labels)

print(ventas["rango_precio"].value_counts().sort_index())
print("\nMuestra:")
print(ventas[["producto", "precio_unitario", "rango_precio"]].head(8))
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

*Ejemplo 16: pd.cut con labels personalizados para rangos de precio.*

1. `print(ventas["rango_precio"].value_counts().sort_index())` — Muestra el resultado por pantalla.
2. `print("\nMuestra:")` — Muestra el resultado por pantalla.
3. `print(ventas[["producto", "precio_unitario", "rango_precio"]].head(8))` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 17: pd.qcut con duplicates="drop" para manejar empates

```python
# qcut divide en cuantiles (misma cantidad de datos por bin)
try:
    ventas["cuartil_margen"] = pd.qcut(ventas["margen"], q=4,
                                       labels=["Bajo", "Medio", "Alto", "Muy alto"],
                                       duplicates="drop")
    print("Distribución por cuartil de margen:")
    print(ventas["cuartil_margen"].value_counts().sort_index())
except Exception as e:
    print(f"Error con qcut: {e}")
    # Si hay empates, usamos rank basado en percentiles
    ventas["cuartil_margen"] = pd.qcut(ventas["margen"].rank(method="first"), q=4,
                                       labels=["Bajo", "Medio", "Alto", "Muy alto"])
    print(ventas["cuartil_margen"].value_counts().sort_index())
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: pd.qcut con duplicates="drop" para manejar empates.*

1. qcut divide en cuantiles (misma cantidad de datos por bin)
2. Si hay empates, usamos rank basado en percentiles

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 18: Integrador — Optimizar DataFrame de ventas completo

```python
def optimizar_dataframe(df):
    """Pipeline completo de optimización."""
    df_opt = df.copy()

    # 1. Columnas de texto a category
    for col in df_opt.select_dtypes(include="object").columns:
        n_unicos = df_opt[col].nunique()
        if n_unicos < len(df_opt) * 0.5:  # Si hay repetición significativa
            df_opt[col] = df_opt[col].astype("category")

    # 2. Downcast numéricos
    for col in df_opt.select_dtypes(include="int").columns:
        df_opt[col] = pd.to_numeric(df_opt[col], downcast="integer")

    for col in df_opt.select_dtypes(include="float").columns:
        df_opt[col] = pd.to_numeric(df_opt[col], downcast="float")

    # 3. Columnas booleanas
    for col in df_opt.select_dtypes(include="bool").columns:
        df_opt[col] = df_opt[col].astype("bool")

    return df_opt

antes = ventas.memory_usage(deep=True).sum()
ventas_optimizado = optimizar_dataframe(ventas)
despues = ventas_optimizado.memory_usage(deep=True).sum()

print(f"Memoria original: {antes:,} bytes ({antes/1024:.1f} KB)")
print(f"Memoria optimizada: {despues:,} bytes ({despues/1024:.1f} KB)")
print(f"Ahorro total: {(1 - despues/antes) * 100:.1f}%")
print("\nTipos optimizados:")
print(ventas_optimizado.dtypes)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador — Optimizar DataFrame de ventas completo.*

1. 1. Columnas de texto a category
2. 2. Downcast numéricos
3. 3. Columnas booleanas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 3. Resumen

| Función | Propósito | Aplicación en Ventas/Compras/Inventarios |
|---------|-----------|------------------------------------------|
| `pd.eval` | Expresiones rápidas sin DataFrames intermedios | Calcular métricas en lotes grandes |
| `df.query` | Filtrado con sintaxis de cadena + `@variables` | Filtrar ventas por umbral dinámico |
| `df.pipe` | Encadenar transformaciones en pipeline | Pipeline de limpieza → filtro → agregación |
| `df.assign` | Crear columnas en cadena | Añadir impuestos, descuentos, utilidades |
| `df.transform` | Operaciones que preservan tamaño | Normalización intra-grupo |
| `df.map` / `.map` | Mapeo diccionario/función | Traducir códigos a nombres |
| `df.applymap` | Operación elemento por elemento | Formatear/redondear celdas |
| `.astype("category")` | Tipos categóricos | Reducir memoria en columnas repetitivas |
| `infer_objects` / `convert_dtypes` | Inferencia moderna de tipos | Tipos nullable y strings |
| Boolean indexing | Filtros con & \| ~ | Segmentación multicondición |
| `df.filter(like/regex)` | Selección por nombre de columna | Elegir columnas por patrón |
| `df.select_dtypes` | Selección por tipo de dato | Separar numéricas de categóricas |
| `df.memory_usage(deep=True)` | Medición de memoria | Auditoría de consumo |
| `pd.cut` / `pd.qcut` | Discretización en bins | Segmentar precios o márgenes |

---

## 4. Ejercicios Propuestos

**Ejercicio 1:** Usa `pd.eval` para calcular `ingreso - costo_total - (ingreso * 0.16)` como utilidad neta en todo el DataFrame de ventas. Compara la velocidad con la operación nativa.

**Ejercicio 2:** Usa `df.query` con una variable `@precio_min` que definas tú. Filtra todas las ventas con `precio_unitario > @precio_min` y `cantidad >= 5`.

**Ejercicio 3:** Crea un pipeline con `df.pipe` que: (1) filtre precios mayores a 0, (2) calcule el margen porcentual, (3) etiquete productos como "Caro" o "Barato" según si superan el precio promedio.

**Ejercicio 4:** Usa `df.assign` en cadena para crear 4 nuevas columnas: `margen_bruto = ingreso - costo_total`, `impuesto = ingreso * 0.16`, `utilidad = margen_bruto - impuesto`, `rentabilidad = utilidad / ingreso`.

**Ejercicio 5:** Convierte las columnas `categoria` y `sucursal` a tipo `category`. Mide la memoria antes y después con `memory_usage(deep=True)`. Calcula el ahorro total.

**Ejercicio 6:** Usa `pd.cut` para crear 5 rangos de descuento (`descuento`) con labels: "Nulo", "Bajo", "Medio", "Alto", "Muy alto". Muestra cuántas ventas caen en cada rango.

**Ejercicio 7:** Del DataFrame de inventario (`pd.read_csv("../../datos/inventario.csv")`), usa `df.filter(regex="stock|precio")` para seleccionar solo las columnas relacionadas a stock y precio. Muestra las primeras 5 filas.

**Ejercicio 8:** Combina boolean indexing con al menos 4 condiciones diferentes sobre ventas para encontrar los registros de "Electrónica" con ingreso > $30,000, margen_pct > 20%, y que no sean de "Sucursal Mérida". ¿Cuántos registros cumplen?

---

*Fin del documento I08 — Funciones Avanzadas de Pandas*
