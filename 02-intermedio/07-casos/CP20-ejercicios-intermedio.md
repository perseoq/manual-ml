# CP20: 35 Ejercicios Integradores de Nivel Intermedio

## Contexto de Negocio
Este compendio de ejercicios consolida todos los temas del nivel intermedio: NumPy avanzado, Pandas experto, Seaborn, SciPy, ML clásico, clustering, reducción de dimensionalidad, NLP y pipelines integradores. Cada ejercicio incluye enunciado, pista y solución.

```python
# ============================================================
# BLOQUE 1: NUMPY AVANZADO (Ejercicios 1-5)
# ============================================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import classification_report
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.figsize": (12, 6), "font.size": 12})

ventas = pd.read_csv("../datos/ventas.csv")
inventario = pd.read_csv("../datos/inventario.csv")
compras = pd.read_csv("../datos/compras.csv")
resenas = pd.read_csv("../datos/resenas.csv")
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Contexto de Negocio.*

1. ============================================================
2. BLOQUE 1: NUMPY AVANZADO (Ejercicios 1-5)
3. ============================================================

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
print("=" * 70)
print("EJERCICIO 1: BROADCASTING — Descuentos progresivos")
print("=" * 70)
print("Enunciado: Aplica descuentos del [5%, 10%, 15%, 20%] a las cantidades")
print("de cada producto usando broadcasting de NumPy. Calcula el ingreso")
print("proyectado para cada nivel de descuento.")
print("Pista: usa ventas['ingreso'].values * np.array([...]).reshape(-1, 1)")
print("")

ingresos = ventas["ingreso"].values
descuentos = np.array([0.95, 0.90, 0.85, 0.80])
ingresos_con_descuento = ingresos * descuentos.reshape(-1, 1)
df_descuentos = pd.DataFrame(
    ingresos_con_descuento.T,
    columns=[f"{int((1-d)*100)}% desc" for d in descuentos]
)
print(df_descuentos.head())
print(f"\nIngreso total sin descuento: ${ingresos.sum():,.2f}")
for i, d in enumerate(descuentos):
    print(f"Ingreso con {int((1-d)*100)}% desc: ${ingresos_con_descuento[i].sum():,.2f}")
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

*Este ejemplo.*

1. `print("=" * 70)` — Muestra el resultado por pantalla.
2. `print("EJERCICIO 1: BROADCASTING — Descuentos progresivos")` — Muestra el resultado por pantalla.
3. `print("=" * 70)` — Muestra el resultado por pantalla.
4. `print("Enunciado: Aplica descuentos del [5%, 10%, 15%, 20%] a las cantidades")` — Muestra el resultado por pantalla.
5. `print("de cada producto usando broadcasting de NumPy. Calcula el ingreso")` — Muestra el resultado por pantalla.
6. `print("proyectado para cada nivel de descuento.")` — Muestra el resultado por pantalla.
7. `print("Pista: usa ventas['ingreso'].values * np.array([...]).reshape(-1, 1)")` — Muestra el resultado por pantalla.
8. `print("")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
print("=" * 70)
print("EJERCICIO 2: np.where — Clasificar productos según rendimiento")
print("=" * 70)
print("Enunciado: Crea una columna 'rendimiento' usando np.where:")
print("Alto si ingreso > Q3, Bajo si ingreso < Q1, Medio en otro caso.")
print("Pista: Q1 = np.percentile(ventas['ingreso'], 25)")
print("")

Q1 = np.percentile(ventas["ingreso"], 25)
Q3 = np.percentile(ventas["ingreso"], 75)
ventas["rendimiento"] = np.where(
    ventas["ingreso"] > Q3, "Alto",
    np.where(ventas["ingreso"] < Q1, "Bajo", "Medio")
)
print(ventas["rendimiento"].value_counts())
sns.countplot(data=ventas, x="rendimiento", order=["Bajo", "Medio", "Alto"],
              palette="viridis", hue="rendimiento", legend=False)
plt.title("Clasificación de Rendimiento por Ingreso")
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. `print("=" * 70)` — Muestra el resultado por pantalla.
2. `print("EJERCICIO 2: np.where — Clasificar productos según rendimiento")` — Muestra el resultado por pantalla.
3. `print("=" * 70)` — Muestra el resultado por pantalla.
4. `print("Enunciado: Crea una columna 'rendimiento' usando np.where:")` — Muestra el resultado por pantalla.
5. `print("Alto si ingreso > Q3, Bajo si ingreso < Q1, Medio en otro caso.")` — Muestra el resultado por pantalla.
6. `print("Pista: Q1 = np.percentile(ventas['ingreso'], 25)")` — Muestra el resultado por pantalla.
7. `print("")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
print("=" * 70)
print("EJERCICIO 3: FUNCIONES MAESTRAS — Estadísticas por sucursal")
print("=" * 70)
print("Enunciado: Usa np.mean, np.std, np.min, np.max para calcular")
print("estadísticas de ingreso agrupadas por sucursal. Compara resultados")
print("con el método groupby de pandas.")
print("Pista: unique() para sucursales, luego boolean indexing.")
print("")

sucursales = ventas["sucursal"].unique()
stats_np = []
for suc in sucursales:
    mask = ventas["sucursal"] == suc
    ing = ventas.loc[mask, "ingreso"].values
    stats_np.append({
        "sucursal": suc, "media": np.mean(ing),
        "std": np.std(ing), "min": np.min(ing), "max": np.max(ing)
    })
df_stats_np = pd.DataFrame(stats_np)
stats_pd = ventas.groupby("sucursal")["ingreso"].agg(["mean", "std", "min", "max"]).reset_index()
print("Diferencia máxima (media):", abs(df_stats_np["media"] - stats_pd["mean"]).max())
print("(Debe ser 0 — ambos métodos equivalentes)")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. `print("=" * 70)` — Muestra el resultado por pantalla.
2. `print("EJERCICIO 3: FUNCIONES MAESTRAS — Estadísticas por sucursal")` — Muestra el resultado por pantalla.
3. `print("=" * 70)` — Muestra el resultado por pantalla.
4. `print("Enunciado: Usa np.mean, np.std, np.min, np.max para calcular")` — Muestra el resultado por pantalla.
5. `print("estadísticas de ingreso agrupadas por sucursal. Compara resultados")` — Muestra el resultado por pantalla.
6. `print("con el método groupby de pandas.")` — Muestra el resultado por pantalla.
7. `print("Pista: unique() para sucursales, luego boolean indexing.")` — Muestra el resultado por pantalla.
8. `print("")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
print("=" * 70)
print("EJERCICIO 4: OPERACIONES VECTORIZADAS — Cálculo de márgenes")
print("=" * 70)
print("Enunciado: Calcula margen bruto, neto (después de descuento) y %.")
print("Costo = 60% del ingreso. Usa operaciones vectorizadas.")
print("Pista: opera directamente sobre columnas del DataFrame.")
print("")

costo = ventas["ingreso"] * 0.6
descuento = ventas["ingreso"] * ventas["descuento"]
margen_bruto = ventas["ingreso"] - costo
margen_neto = ventas["ingreso"] - costo - descuento
margen_pct = (margen_neto / ventas["ingreso"] * 100)
print(f"Margen bruto promedio: ${margen_bruto.mean():.2f}")
print(f"Margen neto promedio: ${margen_neto.mean():.2f}")
print(f"Margen %% promedio: {margen_pct.mean():.1f}%%")
print(f"Pérdida por descuentos: ${descuento.sum():,.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. `print("=" * 70)` — Muestra el resultado por pantalla.
2. `print("EJERCICIO 4: OPERACIONES VECTORIZADAS — Cálculo de márgenes")` — Muestra el resultado por pantalla.
3. `print("=" * 70)` — Muestra el resultado por pantalla.
4. `print("Enunciado: Calcula margen bruto, neto (después de descuento) y %.")` — Muestra el resultado por pantalla.
5. `print("Costo = 60% del ingreso. Usa operaciones vectorizadas.")` — Muestra el resultado por pantalla.
6. `print("Pista: opera directamente sobre columnas del DataFrame.")` — Muestra el resultado por pantalla.
7. `print("")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
print("=" * 70)
print("EJERCICIO 5: ÁLGEBRA LINEAL — Producto punto para ventas totales")
print("=" * 70)
print("Enunciado: Usa np.dot para calcular ingreso total multiplicando")
print("cantidades por precios unitarios. Compara con sum tradicional.")
print("Pista: np.dot(ventas['cantidad'], ventas['precio_unitario'])")
print("")

if "precio_unitario" not in ventas.columns:
    ventas["precio_unitario"] = ventas["ingreso"] / ventas["cantidad"].replace(0, np.nan)
    ventas["precio_unitario"] = ventas["precio_unitario"].fillna(0)
total_dot = np.dot(ventas["cantidad"].values, ventas["precio_unitario"].values)
total_sum = ventas["ingreso"].sum()
print(f"Total con np.dot: ${total_dot:,.2f}")
print(f"Total con sum:    ${total_sum:,.2f}")
print(f"¿Coinciden?: {np.isclose(total_dot, total_sum, rtol=0.01)}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. `print("=" * 70)` — Muestra el resultado por pantalla.
2. `print("EJERCICIO 5: ÁLGEBRA LINEAL — Producto punto para ventas totales")` — Muestra el resultado por pantalla.
3. `print("=" * 70)` — Muestra el resultado por pantalla.
4. `print("Enunciado: Usa np.dot para calcular ingreso total multiplicando")` — Muestra el resultado por pantalla.
5. `print("cantidades por precios unitarios. Compara con sum tradicional.")` — Muestra el resultado por pantalla.
6. `print("Pista: np.dot(ventas['cantidad'], ventas['precio_unitario'])")` — Muestra el resultado por pantalla.
7. `print("")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
print("=" * 70)
print("BLOQUE 2: PANDAS AVANZADO (Ejercicios 6-10)")
print("=" * 70)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. `print("=" * 70)` — Muestra el resultado por pantalla.
2. `print("BLOQUE 2: PANDAS AVANZADO (Ejercicios 6-10)")` — Muestra el resultado por pantalla.
3. `print("=" * 70)` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
print("=" * 70)
print("EJERCICIO 6: GROUPBY — Top 3 productos por sucursal")
print("=" * 70)
print("Enunciado: Por cada sucursal, los 3 productos con mayor ingreso.")
print("Pista: groupby('sucursal').apply(lambda x: x.nlargest(3, 'ingreso'))")
print("")

top_por_sucursal = (
    ventas.groupby(["sucursal", "producto"])["ingreso"].sum().reset_index()
    .sort_values(["sucursal", "ingreso"], ascending=[True, False])
    .groupby("sucursal").head(3)
)
print(top_por_sucursal.to_string(index=False))
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

*Este ejemplo.*

1. `print("=" * 70)` — Muestra el resultado por pantalla.
2. `print("EJERCICIO 6: GROUPBY — Top 3 productos por sucursal")` — Muestra el resultado por pantalla.
3. `print("=" * 70)` — Muestra el resultado por pantalla.
4. `print("Enunciado: Por cada sucursal, los 3 productos con mayor ingreso.")` — Muestra el resultado por pantalla.
5. `print("Pista: groupby('sucursal').apply(lambda x: x.nlargest(3, 'ingreso'))")` — Muestra el resultado por pantalla.
6. `print("")` — Muestra el resultado por pantalla.
7. `ventas.groupby(["sucursal", "producto"])["ingreso"].sum().reset_index()` — Agrupa los datos por la columna eRealiza la operación indicada con los parámetros definidos..
8. `.sort_values(["sucursal", "ingreso"], ascending=[True, False])` — Ordena los datos según la columna eRealiza la operación indicada con los parámetros definidos..

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
print("=" * 70)
print("EJERCICIO 7: MERGE — Ventas con inventario")
print("=" * 70)
print("Enunciado: Merge ventas con inventario. Identifica productos")
print("sin inventario. Calcula rotación = ventas / stock.")
print("Pista: pd.merge(..., on='producto', how='left')")
print("")

ventas_prod = ventas.groupby("producto")["cantidad"].sum().reset_index()
merged = pd.merge(ventas_prod, inventario, on="producto", how="left")
merged["rotacion"] = merged["cantidad"] / merged["stock_disponible"].replace(0, np.nan)
print(merged.head(10))
sin_inv = merged[merged["stock_disponible"].isna()]
print(f"Productos sin inventario: {len(sin_inv)}")
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

*Este ejemplo.*

1. `print("=" * 70)` — Muestra el resultado por pantalla.
2. `print("EJERCICIO 7: MERGE — Ventas con inventario")` — Muestra el resultado por pantalla.
3. `print("=" * 70)` — Muestra el resultado por pantalla.
4. `print("Enunciado: Merge ventas con inventario. Identifica productos")` — Muestra el resultado por pantalla.
5. `print("sin inventario. Calcula rotación = ventas / stock.")` — Muestra el resultado por pantalla.
6. `print("Pista: pd.merge(..., on='producto', how='left')")` — Muestra el resultado por pantalla.
7. `print("")` — Muestra el resultado por pantalla.
8. `ventas_prod = ventas.groupby("producto")["cantidad"].sum().reset_index()` — Agrupa los datos por la columna eRealiza la operación indicada con los parámetros definidos..
9. `merged = pd.merge(ventas_prod, inventario, on="producto", how="left")` — Combina dos DataFrames por una columna clave.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
print("=" * 70)
print("EJERCICIO 8: PIVOT — Tabla dinámica mensual")
print("=" * 70)
print("Enunciado: Pivot con meses como filas, productos como columnas")
print("e ingreso como valores. NaN a 0. Encuentra producto líder en enero.")
print("Pista: pd.pivot_table(values='ingreso', index='mes', columns='producto')")
print("")

ventas["mes"] = pd.to_datetime(ventas["fecha"]).dt.month
pivot = pd.pivot_table(ventas, values="ingreso", index="mes",
                        columns="producto", aggfunc="sum").fillna(0)
print(pivot.head())
print(f"Shape: {pivot.shape}")
print(f"Líder enero: {pivot.loc[1].idxmax()} = ${pivot.loc[1].max():,.0f}")
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

*Este ejemplo.*

1. `print("=" * 70)` — Muestra el resultado por pantalla.
2. `print("EJERCICIO 8: PIVOT — Tabla dinámica mensual")` — Muestra el resultado por pantalla.
3. `print("=" * 70)` — Muestra el resultado por pantalla.
4. `print("Enunciado: Pivot con meses como filas, productos como columnas")` — Muestra el resultado por pantalla.
5. `print("e ingreso como valores. NaN a 0. Encuentra producto líder en enero.")` — Muestra el resultado por pantalla.
6. `print("Pista: pd.pivot_table(values='ingreso', index='mes', columns='producto')")` — Muestra el resultado por pantalla.
7. `print("")` — Muestra el resultado por pantalla.
8. `ventas["mes"] = pd.to_datetime(ventas["fecha"]).dt.month` — Convierte la columna a formato datetime.
9. `pivot = pd.pivot_table(ventas, values="ingreso", index="mes",` — Reorganiza los datos de formato largo a ancho.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
print("=" * 70)
print("EJERCICIO 9: VENTANAS — Picos en ingresos diarios")
print("=" * 70)
print("Enunciado: Promedio móvil 7d + std móvil. Detecta días con")
print("ingreso > media + 2*std (picos).")
print("Pista: rolling(7).mean(), rolling(7).std()")
print("")

ventas_diarias = ventas.groupby("fecha")["ingreso"].sum().reset_index()
ventas_diarias["fecha"] = pd.to_datetime(ventas_diarias["fecha"])
ventas_diarias = ventas_diarias.set_index("fecha").asfreq("D").fillna(0)
ventas_diarias["mm7"] = ventas_diarias["ingreso"].rolling(7).mean()
ventas_diarias["std7"] = ventas_diarias["ingreso"].rolling(7).std()
ventas_diarias["pico"] = ventas_diarias["ingreso"] > (ventas_diarias["mm7"] + 2 * ventas_diarias["std7"])
print(f"Picos detectados: {ventas_diarias['pico'].sum()}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. `print("=" * 70)` — Muestra el resultado por pantalla.
2. `print("EJERCICIO 9: VENTANAS — Picos en ingresos diarios")` — Muestra el resultado por pantalla.
3. `print("=" * 70)` — Muestra el resultado por pantalla.
4. `print("Enunciado: Promedio móvil 7d + std móvil. Detecta días con")` — Muestra el resultado por pantalla.
5. `print("ingreso > media + 2*std (picos).")` — Muestra el resultado por pantalla.
6. `print("Pista: rolling(7).mean(), rolling(7).std()")` — Muestra el resultado por pantalla.
7. `print("")` — Muestra el resultado por pantalla.
8. `ventas_diarias = ventas.groupby("fecha")["ingreso"].sum().reset_index()` — Agrupa los datos por la columna eRealiza la operación indicada con los parámetros definidos..
9. `ventas_diarias["fecha"] = pd.to_datetime(ventas_diarias["fecha"])` — Convierte la columna a formato datetime.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
print("=" * 70)
print("EJERCICIO 10: FECHAS — Cohortes semanales")
print("=" * 70)
print("Enunciado: Asigna cada transacción a su cohorte semanal.")
print("Calcula ingreso total por cohorte. Identifica la mejor semana.")
print("Pista: dt.isocalendar().week, groupby por año y semana.")
print("")

ventas["semana"] = pd.to_datetime(ventas["fecha"]).dt.isocalendar().week.astype(int)
ventas["año"] = pd.to_datetime(ventas["fecha"]).dt.year
cohortes = ventas.groupby(["año", "semana"])["ingreso"].sum().reset_index()
mejor = cohortes.loc[cohortes["ingreso"].idxmax()]
print(f"Semanas con datos: {len(cohortes)}")
print(f"Mejor semana: año {mejor['año']}, semana {mejor['semana']} = ${mejor['ingreso']:,.0f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. `print("=" * 70)` — Muestra el resultado por pantalla.
2. `print("EJERCICIO 10: FECHAS — Cohortes semanales")` — Muestra el resultado por pantalla.
3. `print("=" * 70)` — Muestra el resultado por pantalla.
4. `print("Enunciado: Asigna cada transacción a su cohorte semanal.")` — Muestra el resultado por pantalla.
5. `print("Calcula ingreso total por cohorte. Identifica la mejor semana.")` — Muestra el resultado por pantalla.
6. `print("Pista: dt.isocalendar().week, groupby por año y semana.")` — Muestra el resultado por pantalla.
7. `print("")` — Muestra el resultado por pantalla.
8. `ventas["semana"] = pd.to_datetime(ventas["fecha"]).dt.isocalendar().week.astype(int)` — Convierte la columna a formato datetime.
9. `ventas["año"] = pd.to_datetime(ventas["fecha"]).dt.year` — Convierte la columna a formato datetime.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
print("=" * 70)
print("BLOQUE 3: SEABORN EXPERTO (Ejercicios 11-15)")
print("=" * 70)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. `print("=" * 70)` — Muestra el resultado por pantalla.
2. `print("BLOQUE 3: SEABORN EXPERTO (Ejercicios 11-15)")` — Muestra el resultado por pantalla.
3. `print("=" * 70)` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
print("=" * 70)
print("EJERCICIO 11: PERSONALIZACIÓN — Estilo corporativo")
print("=" * 70)
print("Enunciado: Gráfico de ingresos por sucursal con colores")
print("corporativos. Usa rc_context para fondo, fuente y grid.")
print("Pista: with plt.rc_context({'axes.facecolor': '...'}):")
print("")

with plt.rc_context({
    "axes.facecolor": "#f0f0f0", "axes.edgecolor": "#333333",
    "axes.grid": True, "grid.alpha": 0.3, "font.family": "serif",
    "font.size": 13, "legend.frameon": True, "legend.facecolor": "white"
}):
    plt.figure(figsize=(10, 5))
    ventas_suc = ventas.groupby("sucursal")["ingreso"].sum().reset_index()
    colores_corp = ["#1a5276", "#2e86c1", "#85c1e9", "#d4e6f1"]
    bars = plt.bar(ventas_suc["sucursal"], ventas_suc["ingreso"], color=colores_corp, edgecolor="white")
    plt.title("Ingresos por Sucursal — Estilo Corporativo", fontsize=15, fontweight="bold")
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                 f"${bar.get_height():,.0f}", ha="center", fontsize=10)
    plt.tight_layout(); plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. `print("=" * 70)` — Muestra el resultado por pantalla.
2. `print("EJERCICIO 11: PERSONALIZACIÓN — Estilo corporativo")` — Muestra el resultado por pantalla.
3. `print("=" * 70)` — Muestra el resultado por pantalla.
4. `print("Enunciado: Gráfico de ingresos por sucursal con colores")` — Muestra el resultado por pantalla.
5. `print("corporativos. Usa rc_context para fondo, fuente y grid.")` — Muestra el resultado por pantalla.
6. `print("Pista: with plt.rc_context({'axes.facecolor': '...'}):")` — Muestra el resultado por pantalla.
7. `print("")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
print("=" * 70)
print("EJERCICIO 12: FACETAS — Distribución por producto")
print("=" * 70)
print("Enunciado: FacetGrid con boxplot de ingresos para top 6 productos.")
print("Cada faceta con su propia escala Y.")
print("Pista: sns.catplot(col='producto', kind='box', sharey=False)")
print("")

top6 = ventas.groupby("producto")["ingreso"].sum().nlargest(6).index
g = sns.catplot(data=ventas[ventas["producto"].isin(top6)],
                x="producto", y="ingreso", col="producto", kind="box",
                sharey=False, height=4, aspect=0.8, palette="Set2", col_wrap=3)
g.set_titles("{col_name}")
g.set_axis_labels("", "Ingreso ($)")
plt.suptitle("Distribución por Producto (faceta individual)", y=1.02, fontsize=14)
plt.tight_layout(); plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. `print("=" * 70)` — Muestra el resultado por pantalla.
2. `print("EJERCICIO 12: FACETAS — Distribución por producto")` — Muestra el resultado por pantalla.
3. `print("=" * 70)` — Muestra el resultado por pantalla.
4. `print("Enunciado: FacetGrid con boxplot de ingresos para top 6 productos.")` — Muestra el resultado por pantalla.
5. `print("Cada faceta con su propia escala Y.")` — Muestra el resultado por pantalla.
6. `print("Pista: sns.catplot(col='producto', kind='box', sharey=False)")` — Muestra el resultado por pantalla.
7. `print("")` — Muestra el resultado por pantalla.
8. `top6 = ventas.groupby("producto")["ingreso"].sum().nlargest(6).index` — Agrupa los datos por la columna eRealiza la operación indicada con los parámetros definidos..

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
print("=" * 70)
print("EJERCICIO 13: HEATMAPS — Correlaciones con máscara")
print("=" * 70)
print("Enunciado: Heatmap triangular con paleta divergente, anotaciones")
print("y barra de color personalizada.")
print("Pista: mask = np.triu(np.ones_like(corr, dtype=bool))")
print("")

cols_num = ventas.select_dtypes(include=[np.number]).columns[:8]
corr = ventas[cols_num].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
plt.figure(figsize=(10, 8))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Correlación"})
plt.title("Matriz de Correlaciones (triangular superior oculta)", fontsize=14)
plt.tight_layout(); plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. `print("=" * 70)` — Muestra el resultado por pantalla.
2. `print("EJERCICIO 13: HEATMAPS — Correlaciones con máscara")` — Muestra el resultado por pantalla.
3. `print("=" * 70)` — Muestra el resultado por pantalla.
4. `print("Enunciado: Heatmap triangular con paleta divergente, anotaciones")` — Muestra el resultado por pantalla.
5. `print("y barra de color personalizada.")` — Muestra el resultado por pantalla.
6. `print("Pista: mask = np.triu(np.ones_like(corr, dtype=bool))")` — Muestra el resultado por pantalla.
7. `print("")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
print("=" * 70)
print("EJERCICIO 14: GRÁFICOS COMPUESTOS — 3 paneles")
print("=" * 70)
print("Enunciado: Tres subplots (ingreso, volumen, margen) para top 10")
print("productos. Comparten eje X.")
print("Pista: plt.subplots(3, 1, sharex=True, figsize=(12, 10))")
print("")

prod_stats = ventas.groupby("producto").agg(
    ingreso_total=("ingreso", "sum"),
    cantidad_total=("cantidad", "sum"),
    margen_prom=("ingreso", lambda x: (x * 0.4).mean())
).sort_values("ingreso_total", ascending=False).head(10)
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 10))
axes[0].bar(prod_stats.index, prod_stats["ingreso_total"], color="steelblue")
axes[0].set_title("Top 10 — Ingreso, Volumen, Margen")
axes[0].set_ylabel("Ingreso ($)")
axes[1].bar(prod_stats.index, prod_stats["cantidad_total"], color="green")
axes[1].set_ylabel("Cantidad")
axes[2].bar(prod_stats.index, prod_stats["margen_prom"], color="orange")
axes[2].set_ylabel("Margen prom ($)")
for ax in axes:
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="x", rotation=45)
plt.tight_layout(); plt.show()
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

*Este ejemplo.*

1. `print("=" * 70)` — Muestra el resultado por pantalla.
2. `print("EJERCICIO 14: GRÁFICOS COMPUESTOS — 3 paneles")` — Muestra el resultado por pantalla.
3. `print("=" * 70)` — Muestra el resultado por pantalla.
4. `print("Enunciado: Tres subplots (ingreso, volumen, margen) para top 10")` — Muestra el resultado por pantalla.
5. `print("productos. Comparten eje X.")` — Muestra el resultado por pantalla.
6. `print("Pista: plt.subplots(3, 1, sharex=True, figsize=(12, 10))")` — Muestra el resultado por pantalla.
7. `print("")` — Muestra el resultado por pantalla.
8. `prod_stats = ventas.groupby("producto").agg(` — Agrupa los datos por la columna eRealiza la operación indicada con los parámetros definidos..

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
print("=" * 70)
print("EJERCICIO 15: ESTILOS Y TEMAS — Dashboard compuesto")
print("=" * 70)
print("Enunciado: Barras + línea + scatter en un gráfico con twinx.")
print("Incluye leyenda unificada y anotaciones en picos.")
print("Pista: ax.twinx() para dos escalas Y.")
print("")

fig, ax1 = plt.subplots(figsize=(14, 6))
resumen = ventas_diarias.tail(30)
ax1.bar(resumen.index, resumen["ingreso"], alpha=0.5, color="gray", label="Ingreso diario")
ax1.set_ylabel("Ingreso ($)", color="gray")
ax2 = ax1.twinx()
ax2.plot(resumen.index, resumen["mm7"], color="red", linewidth=2, label="MM7")
ax2.set_ylabel("MM7 ($)", color="red")
picos = resumen[resumen["pico"]]
ax2.scatter(picos.index, picos["mm7"], color="orange", s=100, zorder=5, label="Picos")
l1, lb1 = ax1.get_legend_handles_labels()
l2, lb2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, lb1 + lb2, loc="upper left")
plt.title("Dashboard: Ingreso + MM7 + Picos", fontsize=14)
plt.tight_layout(); plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. `print("=" * 70)` — Muestra el resultado por pantalla.
2. `print("EJERCICIO 15: ESTILOS Y TEMAS — Dashboard compuesto")` — Muestra el resultado por pantalla.
3. `print("=" * 70)` — Muestra el resultado por pantalla.
4. `print("Enunciado: Barras + línea + scatter en un gráfico con twinx.")` — Muestra el resultado por pantalla.
5. `print("Incluye leyenda unificada y anotaciones en picos.")` — Muestra el resultado por pantalla.
6. `print("Pista: ax.twinx() para dos escalas Y.")` — Muestra el resultado por pantalla.
7. `print("")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
print("=" * 70)
print("BLOQUE 4: SCIPY APLICADO (Ejercicios 16-20)")
print("=" * 70)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. `print("=" * 70)` — Muestra el resultado por pantalla.
2. `print("BLOQUE 4: SCIPY APLICADO (Ejercicios 16-20)")` — Muestra el resultado por pantalla.
3. `print("=" * 70)` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
print("=" * 70)
print("EJERCICIO 16: OPTIMIZACIÓN — Precio óptimo")
print("=" * 70)
print("Enunciado: scipy.optimize.minimize_scalar para precio que maximiza")
print("ingreso. Relación lineal precio-demanda estimada de datos.")
print("Pista: ingreso(p) = p * demanda(p), minimizar -ingreso(p)")
print("")

from scipy.optimize import minimize_scalar
from scipy.stats import linregress
if "precio_unitario" not in ventas.columns:
    ventas["precio_unitario"] = ventas["ingreso"] / ventas["cantidad"].replace(0, np.nan)
precios = ventas["precio_unitario"].dropna().values
cantidades = ventas.loc[ventas["precio_unitario"].notna(), "cantidad"].values
slope, intercept, _, _, _ = linregress(precios, cantidades)
def demanda(p): return max(0, intercept + slope * p)
def ingreso_func(p): return -p * demanda(p)
res = minimize_scalar(ingreso_func, bounds=(precios.min()*0.5, precios.max()*1.5), method="bounded")
print(f"Precio óptimo: ${res.x:.2f}, Ingreso max: ${-res.fun:.2f}")
print(f"Precio actual prom: ${precios.mean():.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. `print("=" * 70)` — Muestra el resultado por pantalla.
2. `print("EJERCICIO 16: OPTIMIZACIÓN — Precio óptimo")` — Muestra el resultado por pantalla.
3. `print("=" * 70)` — Muestra el resultado por pantalla.
4. `print("Enunciado: scipy.optimize.minimize_scalar para precio que maximiza")` — Muestra el resultado por pantalla.
5. `print("ingreso. Relación lineal precio-demanda estimada de datos.")` — Muestra el resultado por pantalla.
6. `print("Pista: ingreso(p) = p * demanda(p), minimizar -ingreso(p)")` — Muestra el resultado por pantalla.
7. `print("")` — Muestra el resultado por pantalla.
8. `from scipy.optimize import minimize_scalar` — Importa las librerías necesarias para el análisis.
9. `from scipy.stats import linregress` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
print("=" * 70)
print("EJERCICIO 17: INTERPOLACIÓN — Rellenar valores faltantes")
print("=" * 70)
print("Enunciado: Simula 10% NaN en ingresos diarios. Compara")
print("interpolación lineal, cuadrática y cúbica con scipy.")
 print("Pista: scipy.interpolate.interp1d(kind='linear')")
print("")

from scipy.interpolate import interp1d
np.random.seed(42)
serie = ventas_diarias["ingreso"].values
serie_nan = serie.copy()
nan_idx = np.random.choice(len(serie), int(len(serie)*0.1), replace=False)
serie_nan[nan_idx] = np.nan
x_known = np.where(~np.isnan(serie_nan))[0]
y_known = serie_nan[~np.isnan(serie_nan)]
x_all = np.arange(len(serie_nan))
for kind, color in [("linear", "green"), ("quadratic", "orange"), ("cubic", "red")]:
    f = interp1d(x_known, y_known, kind=kind, fill_value="extrapolate")
    mae = np.mean(np.abs(f(x_all) - serie))
    print(f"{kind}: MAE = {mae:.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. `print("=" * 70)` — Muestra el resultado por pantalla.
2. `print("EJERCICIO 17: INTERPOLACIÓN — Rellenar valores faltantes")` — Muestra el resultado por pantalla.
3. `print("=" * 70)` — Muestra el resultado por pantalla.
4. `print("Enunciado: Simula 10% NaN en ingresos diarios. Compara")` — Muestra el resultado por pantalla.
5. `print("interpolación lineal, cuadrática y cúbica con scipy.")` — Muestra el resultado por pantalla.
6. `print("Pista: scipy.interpolate.interp1d(kind='linear')")` — Muestra el resultado por pantalla.
7. `print("")` — Muestra el resultado por pantalla.
8. `from scipy.interpolate import interp1d` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
print("=" * 70)
print("EJERCICIO 18: SEÑALES — Suavizado con convolución")
print("=" * 70)
print("Enunciado: Aplica np.convolve y savgol_filter a la serie de")
print("ingresos. Compara con rolling().mean() visualmente.")
 print("Pista: np.convolve(serie, np.ones(7)/7, mode='same')")
print("")

from scipy.signal import savgol_filter
serie = ventas_diarias["ingreso"].values
kernel = np.ones(7) / 7
conv = np.convolve(serie, kernel, mode="same")
savgol = savgol_filter(serie, 11, 3)
plt.figure(figsize=(14, 4))
plt.plot(serie, alpha=0.3, label="Original")
plt.plot(conv, label="Convolución 7d", linewidth=2)
plt.plot(savgol, label="Savitzky-Golay", linewidth=2, linestyle="--")
plt.legend(); plt.title("Suavizado de Señales"); plt.tight_layout(); plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. `print("=" * 70)` — Muestra el resultado por pantalla.
2. `print("EJERCICIO 18: SEÑALES — Suavizado con convolución")` — Muestra el resultado por pantalla.
3. `print("=" * 70)` — Muestra el resultado por pantalla.
4. `print("Enunciado: Aplica np.convolve y savgol_filter a la serie de")` — Muestra el resultado por pantalla.
5. `print("ingresos. Compara con rolling().mean() visualmente.")` — Muestra el resultado por pantalla.
6. `print("Pista: np.convolve(serie, np.ones(7)/7, mode='same')")` — Muestra el resultado por pantalla.
7. `print("")` — Muestra el resultado por pantalla.
8. `from scipy.signal import savgol_filter` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
print("=" * 70)
print("EJERCICIO 19: DISTRIBUCIONES — Test de normalidad")
print("=" * 70)
print("Enunciado: Aplica scipy.stats.normaltest a ingresos. Identifica")
 print("si la distribución es normal. Prueba transformación Box-Cox.")
print("Pista: stats.boxcox(ventas['ingreso'] + 1)")
 print("")

from scipy.stats import normaltest, boxcox
stat, p_valor = normaltest(ventas["ingreso"])
print(f"Normal test: stat={stat:.2f}, p={p_valor:.4f}")
print(f"¿Normal? {'Sí' if p_valor > 0.05 else 'No'}")
ventas["ingreso_bc"], _ = boxcox(ventas["ingreso"] + 1)
stat_bc, p_bc = normaltest(ventas["ingreso_bc"])
print(f"Box-Cox transform: stat={stat_bc:.2f}, p={p_bc:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. `print("=" * 70)` — Muestra el resultado por pantalla.
2. `print("EJERCICIO 19: DISTRIBUCIONES — Test de normalidad")` — Muestra el resultado por pantalla.
3. `print("=" * 70)` — Muestra el resultado por pantalla.
4. `print("Enunciado: Aplica scipy.stats.normaltest a ingresos. Identifica")` — Muestra el resultado por pantalla.
5. `print("si la distribución es normal. Prueba transformación Box-Cox.")` — Muestra el resultado por pantalla.
6. `print("Pista: stats.boxcox(ventas['ingreso'] + 1)")` — Muestra el resultado por pantalla.
7. `print("")` — Muestra el resultado por pantalla.
8. `from scipy.stats import normaltest, boxcox` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
print("=" * 70)
print("EJERCICIO 20: PRUEBAS ESTADÍSTICAS — Diferencia entre sucursales")
print("=" * 70)
print("Enunciado: Usa stats.f_oneway (ANOVA) para probar si hay")
print("diferencia significativa en ingreso entre sucursales.")
print("Pista: agrupa ingresos por sucursal y desempaqueta en *grupos")
print("")

from scipy.stats import f_oneway
grupos = [ventas[ventas["sucursal"] == s]["ingreso"].values for s in ventas["sucursal"].unique()]
f_stat, p_val = f_oneway(*grupos)
print(f"ANOVA: F={f_stat:.4f}, p={p_val:.6f}")
print(f"Diferencias significativas: {'Sí' if p_val < 0.05 else 'No'}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. `print("=" * 70)` — Muestra el resultado por pantalla.
2. `print("EJERCICIO 20: PRUEBAS ESTADÍSTICAS — Diferencia entre sucursales")` — Muestra el resultado por pantalla.
3. `print("=" * 70)` — Muestra el resultado por pantalla.
4. `print("Enunciado: Usa stats.f_oneway (ANOVA) para probar si hay")` — Muestra el resultado por pantalla.
5. `print("diferencia significativa en ingreso entre sucursales.")` — Muestra el resultado por pantalla.
6. `print("Pista: agrupa ingresos por sucursal y desempaqueta en *grupos")` — Muestra el resultado por pantalla.
7. `print("")` — Muestra el resultado por pantalla.
8. `from scipy.stats import f_oneway` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
print("=" * 70)
print("BLOQUE 5: ML CLÁSICO (Ejercicios 21-25)")
print("=" * 70)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. `print("=" * 70)` — Muestra el resultado por pantalla.
2. `print("BLOQUE 5: ML CLÁSICO (Ejercicios 21-25)")` — Muestra el resultado por pantalla.
3. `print("=" * 70)` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
print("=" * 70)
print("EJERCICIO 21: REGRESIÓN LINEAL — Predecir ingreso desde cantidad")
print("=" * 70)
print("Enunciado: LinearRegression para predecir ingreso con cantidad.")
print("Evalúa con R², MAE, RMSE. Visualiza recta de regresión.")
print("Pista: LinearRegression().fit(X.reshape(-1,1), y)")
print("")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
X_ej21 = ventas[["cantidad"]].values
y_ej21 = ventas["ingreso"].values
lr21 = LinearRegression().fit(X_ej21, y_ej21)
y_pred21 = lr21.predict(X_ej21)
print(f"Coef: {lr21.coef_[0]:.2f}, Intercept: {lr21.intercept_:.2f}")
print(f"R²: {r2_score(y_ej21, y_pred21):.4f}")
print(f"MAE: ${mean_absolute_error(y_ej21, y_pred21):.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. `print("=" * 70)` — Muestra el resultado por pantalla.
2. `print("EJERCICIO 21: REGRESIÓN LINEAL — Predecir ingreso desde cantidad")` — Muestra el resultado por pantalla.
3. `print("=" * 70)` — Muestra el resultado por pantalla.
4. `print("Enunciado: LinearRegression para predecir ingreso con cantidad.")` — Muestra el resultado por pantalla.
5. `print("Evalúa con R², MAE, RMSE. Visualiza recta de regresión.")` — Muestra el resultado por pantalla.
6. `print("Pista: LinearRegression().fit(X.reshape(-1,1), y)")` — Muestra el resultado por pantalla.
7. `print("")` — Muestra el resultado por pantalla.
8. `from sklearn.linear_model import LinearRegression` — Importa las librerías necesarias para el análisis.
9. `from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
print("=" * 70)
print("EJERCICIO 22: REGRESIÓN LOGÍSTICA — Alto vs Bajo rendimiento")
print("=" * 70)
print("Enunciado: Clasifica productos como 'Alto' o 'Bajo' rendimiento")
 print("usando LogisticRegression. Evalúa con classification_report.")
print("Pista: y_bin = (ventas['ingreso'] > mediana).astype(int)")
print("")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
mediana = ventas["ingreso"].median()
ventas["alto_rendimiento"] = (ventas["ingreso"] > mediana).astype(int)
X_lr = ventas[["cantidad", "descuento"]].fillna(0)
y_lr = ventas["alto_rendimiento"]
X_train, X_test, y_train, y_test = train_test_split(X_lr, y_lr, test_size=0.3, random_state=42)
logreg = LogisticRegression().fit(X_train, y_train)
print(classification_report(y_test, logreg.predict(X_test), target_names=["Bajo", "Alto"]))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. `print("=" * 70)` — Muestra el resultado por pantalla.
2. `print("EJERCICIO 22: REGRESIÓN LOGÍSTICA — Alto vs Bajo rendimiento")` — Muestra el resultado por pantalla.
3. `print("=" * 70)` — Muestra el resultado por pantalla.
4. `print("Enunciado: Clasifica productos como 'Alto' o 'Bajo' rendimiento")` — Muestra el resultado por pantalla.
5. `print("usando LogisticRegression. Evalúa con classification_report.")` — Muestra el resultado por pantalla.
6. `print("Pista: y_bin = (ventas['ingreso'] > mediana).astype(int)")` — Muestra el resultado por pantalla.
7. `print("")` — Muestra el resultado por pantalla.
8. `from sklearn.linear_model import LogisticRegression` — Importa las librerías necesarias para el análisis.
9. `from sklearn.model_selection import train_test_split` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
print("=" * 70)
print("EJERCICIO 23: ÁRBOL DE DECISIÓN — Interpretabilidad")
print("=" * 70)
print("Enunciado: Entrena DecisionTreeClassifier (max_depth=3) para")
print("clasificar rendimiento. Visualiza el árbol e interpreta.")
print("Pista: plot_tree() de sklearn.tree")
print("")

from sklearn.tree import DecisionTreeClassifier, plot_tree
X_tree = ventas[["ingreso", "cantidad", "descuento"]].fillna(0)
y_tree = ventas["alto_rendimiento"]
tree = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_tree, y_tree)
plt.figure(figsize=(16, 8))
plot_tree(tree, feature_names=X_tree.columns, class_names=["Bajo", "Alto"], filled=True)
plt.title("Árbol de Decisión (profundidad 3)"); plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. `print("=" * 70)` — Muestra el resultado por pantalla.
2. `print("EJERCICIO 23: ÁRBOL DE DECISIÓN — Interpretabilidad")` — Muestra el resultado por pantalla.
3. `print("=" * 70)` — Muestra el resultado por pantalla.
4. `print("Enunciado: Entrena DecisionTreeClassifier (max_depth=3) para")` — Muestra el resultado por pantalla.
5. `print("clasificar rendimiento. Visualiza el árbol e interpreta.")` — Muestra el resultado por pantalla.
6. `print("Pista: plot_tree() de sklearn.tree")` — Muestra el resultado por pantalla.
7. `print("")` — Muestra el resultado por pantalla.
8. `from sklearn.tree import DecisionTreeClassifier, plot_tree` — Importa las librerías necesarias para el análisis.
9. `X_tree = ventas[["ingreso", "cantidad", "descuento"]].fillna(0)` — Rellena los valores faltantes con el valor indicado.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
print("=" * 70)
print("EJERCICIO 24: RANDOM FOREST — Comparar con árbol simple")
print("=" * 70)
print("Enunciado: RandomForestClassifier con 100 árboles. Compara ROC AUC")
print("con el árbol de decisión simple. Analiza feature importance.")
print("Pista: RandomForestClassifier(n_estimators=100, random_state=42)")
print("")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
rf24 = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
y_prob_rf24 = rf24.predict_proba(X_test)[:, 1]
auc_rf = roc_auc_score(y_test, y_prob_rf24)
fi24 = pd.DataFrame({"feature": X_train.columns, "importance": rf24.feature_importances_})
print(f"Random Forest AUC: {auc_rf:.4f}")
print("Feature importance:")
print(fi24.sort_values("importance", ascending=False).to_string(index=False))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. `print("=" * 70)` — Muestra el resultado por pantalla.
2. `print("EJERCICIO 24: RANDOM FOREST — Comparar con árbol simple")` — Muestra el resultado por pantalla.
3. `print("=" * 70)` — Muestra el resultado por pantalla.
4. `print("Enunciado: RandomForestClassifier con 100 árboles. Compara ROC AUC")` — Muestra el resultado por pantalla.
5. `print("con el árbol de decisión simple. Analiza feature importance.")` — Muestra el resultado por pantalla.
6. `print("Pista: RandomForestClassifier(n_estimators=100, random_state=42)")` — Muestra el resultado por pantalla.
7. `print("")` — Muestra el resultado por pantalla.
8. `from sklearn.ensemble import RandomForestClassifier` — Importa las librerías necesarias para el análisis.
9. `from sklearn.metrics import roc_auc_score` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
print("=" * 70)
print("EJERCICIO 25: KNN — Vecinos para clasificación")
print("=" * 70)
print("Enunciado: KNeighborsClassifier con k=3,5,7,9. Encuentra el mejor k")
 print("por validación cruzada. Visualiza fronteras de decisión.")
print("Pista: KNeighborsClassifier(n_neighbors=k)")
print("")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
k_vals = [3, 5, 7, 9]
scores_knn = []
scaler25 = StandardScaler()
X_train_scaled = scaler25.fit_transform(X_train)
X_test_scaled = scaler25.transform(X_test)
for k in k_vals:
    knn = KNeighborsClassifier(n_neighbors=k).fit(X_train_scaled, y_train)
    score = knn.score(X_test_scaled, y_test)
    scores_knn.append({"k": k, "accuracy": score})
    print(f"k={k}: accuracy={score:.4f}")
print(f"Mejor k: {max(scores_knn, key=lambda x: x['accuracy'])}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. `print("=" * 70)` — Muestra el resultado por pantalla.
2. `print("EJERCICIO 25: KNN — Vecinos para clasificación")` — Muestra el resultado por pantalla.
3. `print("=" * 70)` — Muestra el resultado por pantalla.
4. `print("Enunciado: KNeighborsClassifier con k=3,5,7,9. Encuentra el mejor k")` — Muestra el resultado por pantalla.
5. `print("por validación cruzada. Visualiza fronteras de decisión.")` — Muestra el resultado por pantalla.
6. `print("Pista: KNeighborsClassifier(n_neighbors=k)")` — Muestra el resultado por pantalla.
7. `print("")` — Muestra el resultado por pantalla.
8. `from sklearn.neighbors import KNeighborsClassifier` — Importa las librerías necesarias para el análisis.
9. `from sklearn.preprocessing import StandardScaler` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
print("=" * 70)
print("BLOQUE 6: CLUSTERING Y REDUCCIÓN (Ejercicios 26-30)")
print("=" * 70)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. `print("=" * 70)` — Muestra el resultado por pantalla.
2. `print("BLOQUE 6: CLUSTERING Y REDUCCIÓN (Ejercicios 26-30)")` — Muestra el resultado por pantalla.
3. `print("=" * 70)` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
print("=" * 70)
print("EJERCICIO 26: K-MEANS — Segmentación de productos")
print("=" * 70)
print("Enunciado: Segmenta productos por ingreso y cantidad con K-Means.")
print("Prueba k=2 a 6. Usa silhouette_score para elegir el mejor k.")
print("Pista: KMeans(n_clusters=k, random_state=42)")
print("")

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
X_kmeans = ventas[["ingreso", "cantidad"]].sample(1000, random_state=42)
scaler_k = StandardScaler().fit(X_kmeans)
X_k_scaled = scaler_k.transform(X_kmeans)
sil_scores = []
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_k_scaled)
    sil = silhouette_score(X_k_scaled, km.labels_)
    sil_scores.append({"k": k, "silhouette": sil})
    print(f"k={k}: silhouette={sil:.4f}")
print(f"Mejor k: {max(sil_scores, key=lambda x: x['silhouette'])}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. `print("=" * 70)` — Muestra el resultado por pantalla.
2. `print("EJERCICIO 26: K-MEANS — Segmentación de productos")` — Muestra el resultado por pantalla.
3. `print("=" * 70)` — Muestra el resultado por pantalla.
4. `print("Enunciado: Segmenta productos por ingreso y cantidad con K-Means.")` — Muestra el resultado por pantalla.
5. `print("Prueba k=2 a 6. Usa silhouette_score para elegir el mejor k.")` — Muestra el resultado por pantalla.
6. `print("Pista: KMeans(n_clusters=k, random_state=42)")` — Muestra el resultado por pantalla.
7. `print("")` — Muestra el resultado por pantalla.
8. `from sklearn.cluster import KMeans` — Importa las librerías necesarias para el análisis.
9. `from sklearn.metrics import silhouette_score` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
print("=" * 70)
print("EJERCICIO 27: PCA — Reducción a 2D para visualización")
print("=" * 70)
print("Enunciado: Aplica PCA a features numéricas de ventas. Visualiza")
print("los primeros 2 componentes. Explica varianza explicada.")
print("Pista: PCA(n_components=2).fit_transform(X_scaled)")
print("")

from sklearn.decomposition import PCA
X_pca = ventas.select_dtypes(include=[np.number]).dropna().sample(500, random_state=42)
X_pca = X_pca.loc[:, X_pca.std() > 0]
scaler_pca = StandardScaler().fit(X_pca)
X_pca_scaled = scaler_pca.transform(X_pca)
pca = PCA(n_components=2).fit(X_pca_scaled)
X_pca_2d = pca.transform(X_pca_scaled)
print(f"Varianza explicada: {pca.explained_variance_ratio_}")
print(f"Total 2D: {pca.explained_variance_ratio_.sum():.2%}")
plt.figure(figsize=(10, 6))
plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], alpha=0.5, c=ventas.loc[X_pca.index, "ingreso"], cmap="viridis")
plt.colorbar(label="Ingreso")
plt.title("PCA 2D — Proyección de Ventas"); plt.xlabel("PC1"); plt.ylabel("PC2")
plt.tight_layout(); plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. `print("=" * 70)` — Muestra el resultado por pantalla.
2. `print("EJERCICIO 27: PCA — Reducción a 2D para visualización")` — Muestra el resultado por pantalla.
3. `print("=" * 70)` — Muestra el resultado por pantalla.
4. `print("Enunciado: Aplica PCA a features numéricas de ventas. Visualiza")` — Muestra el resultado por pantalla.
5. `print("los primeros 2 componentes. Explica varianza explicada.")` — Muestra el resultado por pantalla.
6. `print("Pista: PCA(n_components=2).fit_transform(X_scaled)")` — Muestra el resultado por pantalla.
7. `print("")` — Muestra el resultado por pantalla.
8. `from sklearn.decomposition import PCA` — Importa las librerías necesarias para el análisis.
9. `X_pca = ventas.select_dtypes(include=[np.number]).dropna().sample(500, random_state=42)` — Elimina las filas con valores nulos.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
print("=" * 70)
print("EJERCICIO 28: DBSCAN — Clustering basado en densidad")
print("=" * 70)
print("Enunciado: Aplica DBSCAN a ventas escaladas. Encuentra clusters")
print("y puntos ruido. Compara con K-Means.")
print("Pista: DBSCAN(eps=0.5, min_samples=10)")
print("")

from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=10).fit(X_k_scaled)
n_clusters = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
n_noise = list(dbscan.labels_).count(-1)
print(f"DBSCAN clusters: {n_clusters}, ruido: {n_noise}")
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.scatter(X_k_scaled[:, 0], X_k_scaled[:, 1], c=KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_k_scaled), cmap="Set2")
plt.title("K-Means (k=3)")
plt.subplot(122)
plt.scatter(X_k_scaled[:, 0], X_k_scaled[:, 1], c=dbscan.labels_, cmap="Set2")
plt.title(f"DBSCAN (clusters={n_clusters})")
plt.tight_layout(); plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. `print("=" * 70)` — Muestra el resultado por pantalla.
2. `print("EJERCICIO 28: DBSCAN — Clustering basado en densidad")` — Muestra el resultado por pantalla.
3. `print("=" * 70)` — Muestra el resultado por pantalla.
4. `print("Enunciado: Aplica DBSCAN a ventas escaladas. Encuentra clusters")` — Muestra el resultado por pantalla.
5. `print("y puntos ruido. Compara con K-Means.")` — Muestra el resultado por pantalla.
6. `print("Pista: DBSCAN(eps=0.5, min_samples=10)")` — Muestra el resultado por pantalla.
7. `print("")` — Muestra el resultado por pantalla.
8. `from sklearn.cluster import DBSCAN` — Importa las librerías necesarias para el análisis.
9. `dbscan = DBSCAN(eps=0.5, min_samples=10).fit(X_k_scaled)` — Entrena el modelo con los datos de entrenamiento.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
print("=" * 70)
print("EJERCICIO 29: T-SNE — Visualización no lineal")
print("=" * 70)
print("Enunciado: t-SNE para visualizar estructura de datos en 2D.")
print("Compara con PCA. ¿t-SNE revela patrones que PCA no muestra?")
 print("Pista: TSNE(n_components=2, perplexity=30, random_state=42)")
print("")

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_k_scaled)
plt.figure(figsize=(10, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.6, cmap="viridis", c=ventas.loc[X_kmeans.index, "ingreso"])
plt.colorbar(label="Ingreso")
plt.title("t-SNE 2D — Estructura de Datos de Ventas")
plt.tight_layout(); plt.show()
print("t-SNE revela agrupaciones que PCA lineal no puede capturar.")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. `print("=" * 70)` — Muestra el resultado por pantalla.
2. `print("EJERCICIO 29: T-SNE — Visualización no lineal")` — Muestra el resultado por pantalla.
3. `print("=" * 70)` — Muestra el resultado por pantalla.
4. `print("Enunciado: t-SNE para visualizar estructura de datos en 2D.")` — Muestra el resultado por pantalla.
5. `print("Compara con PCA. ¿t-SNE revela patrones que PCA no muestra?")` — Muestra el resultado por pantalla.
6. `print("Pista: TSNE(n_components=2, perplexity=30, random_state=42)")` — Muestra el resultado por pantalla.
7. `print("")` — Muestra el resultado por pantalla.
8. `from sklearn.manifold import TSNE` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
print("=" * 70)
print("EJERCICIO 30: EVALUACIÓN DE CLUSTERING — Métricas internas")
print("=" * 70)
print("Enunciado: Calcula silhouette_score, calinski_harabasz_score y")
print("davies_bouldin_score para K-Means (k=3). Interpreta cada métrica.")
print("Pista: metrics.silhouette_score, .calinski_harabasz_score")
print("")

from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
km30 = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_k_scaled)
labels30 = km30.labels_
sil = silhouette_score(X_k_scaled, labels30)
ch = calinski_harabasz_score(X_k_scaled, labels30)
db = davies_bouldin_score(X_k_scaled, labels30)
print(f"Silhouette: {sil:.4f} (más cerca de 1 = mejor)")
print(f"Calinski-Harabasz: {ch:.2f} (más alto = mejor)")
print(f"Davies-Bouldin: {db:.4f} (más bajo = mejor)")
print(f"Interpretación: k=3 genera clusters {'bien' if sil > 0.5 else 'moderadamente'} separados.")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. `print("=" * 70)` — Muestra el resultado por pantalla.
2. `print("EJERCICIO 30: EVALUACIÓN DE CLUSTERING — Métricas internas")` — Muestra el resultado por pantalla.
3. `print("=" * 70)` — Muestra el resultado por pantalla.
4. `print("Enunciado: Calcula silhouette_score, calinski_harabasz_score y")` — Muestra el resultado por pantalla.
5. `print("davies_bouldin_score para K-Means (k=3). Interpreta cada métrica.")` — Muestra el resultado por pantalla.
6. `print("Pista: metrics.silhouette_score, .calinski_harabasz_score")` — Muestra el resultado por pantalla.
7. `print("")` — Muestra el resultado por pantalla.
8. `from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score` — Importa las librerías necesarias para el análisis.
9. `km30 = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_k_scaled)` — Entrena el modelo con los datos de entrenamiento.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
print("=" * 70)
print("BLOQUE 7: NLP (Ejercicios 31-33)")
print("=" * 70)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. `print("=" * 70)` — Muestra el resultado por pantalla.
2. `print("BLOQUE 7: NLP (Ejercicios 31-33)")` — Muestra el resultado por pantalla.
3. `print("=" * 70)` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
print("=" * 70)
print("EJERCICIO 31: TOKENIZACIÓN BÁSICA — Frecuencia de palabras")
print("=" * 70)
print("Enunciado: Usa nltk o split() para tokenizar reseñas. Calcula")
print("las 20 palabras más frecuentes (excluyendo stopwords).")
print("Pista: Counter(palabras).most_common(20)")
print("")

from collections import Counter
import re
try:
    from nltk.corpus import stopwords
    import nltk; nltk.download("stopwords", quiet=True)
    stops = set(stopwords.words("spanish"))
except:
    stops = {"el", "la", "los", "las", "un", "una", "y", "e", "o", "a", "de", "del",
             "para", "por", "con", "en", "es", "muy", "que", "se", "no", "lo", "su"}
texto_completo = " ".join(resenas["resena"].dropna())
palabras = [p.lower() for p in re.findall(r'\b\w+\b', texto_completo) if p.lower() not in stops and len(p) > 3]
top20 = Counter(palabras).most_common(20)
print("Top 20 palabras en reseñas:")
for palabra, count in top20:
    print(f"  {palabra}: {count}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. `print("=" * 70)` — Muestra el resultado por pantalla.
2. `print("EJERCICIO 31: TOKENIZACIÓN BÁSICA — Frecuencia de palabras")` — Muestra el resultado por pantalla.
3. `print("=" * 70)` — Muestra el resultado por pantalla.
4. `print("Enunciado: Usa nltk o split() para tokenizar reseñas. Calcula")` — Muestra el resultado por pantalla.
5. `print("las 20 palabras más frecuentes (excluyendo stopwords).")` — Muestra el resultado por pantalla.
6. `print("Pista: Counter(palabras).most_common(20)")` — Muestra el resultado por pantalla.
7. `print("")` — Muestra el resultado por pantalla.
8. `from collections import Counter` — Importa las librerías necesarias para el análisis.
9. `import re` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
print("=" * 70)
print("EJERCICIO 32: ANÁLISIS DE SENTIMIENTO — Clasificador simple")
print("=" * 70)
print("Enunciado: Crea un clasificador Naive Bayes con CountVectorizer")
print("para predecir si una reseña es positiva (>=4) o negativa (<=2).")
print("Pista: MultinomialNB(), train_test_split 70-30")
print("")

try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    resenas_bin = resenas[resenas["puntuacion"] != 3].copy()
    resenas_bin["sentimiento"] = (resenas_bin["puntuacion"] >= 4).astype(int)
    vectorizer = CountVectorizer(max_features=500, stop_words=list(stops))
    X_vec = vectorizer.fit_transform(resenas_bin["resena"].fillna(""))
    y_sent = resenas_bin["sentimiento"]
    X_tr, X_te, y_tr, y_te = train_test_split(X_vec, y_sent, test_size=0.3, random_state=42)
    nb = MultinomialNB().fit(X_tr, y_tr)
    print(classification_report(y_te, nb.predict(X_te), target_names=["Negativa", "Positiva"]))
except Exception as e:
    print(f"Error: {e}. Asegúrate de tener sklearn.")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. `print("=" * 70)` — Muestra el resultado por pantalla.
2. `print("EJERCICIO 32: ANÁLISIS DE SENTIMIENTO — Clasificador simple")` — Muestra el resultado por pantalla.
3. `print("=" * 70)` — Muestra el resultado por pantalla.
4. `print("Enunciado: Crea un clasificador Naive Bayes con CountVectorizer")` — Muestra el resultado por pantalla.
5. `print("para predecir si una reseña es positiva (>=4) o negativa (<=2).")` — Muestra el resultado por pantalla.
6. `print("Pista: MultinomialNB(), train_test_split 70-30")` — Muestra el resultado por pantalla.
7. `print("")` — Muestra el resultado por pantalla.
8. `from sklearn.feature_extraction.text import CountVectorizer` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
print("=" * 70)
print("EJERCICIO 33: CLASIFICACIÓN DE TEXTO — TF-IDF + LogisticRegression")
print("=" * 70)
print("Enunciado: Usa TfidfVectorizer + LogisticRegression para clasificar")
print("reseñas en 5 categorías (puntuaciones 1-5). Evalúa con matriz de confusión.")
print("Pista: TfidfVectorizer(max_features=1000)")
print("")

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=1000, stop_words=list(stops))
X_tfidf = tfidf.fit_transform(resenas["resena"].fillna(""))
y_tfidf = resenas["puntuacion"]
X_tr_t, X_te_t, y_tr_t, y_te_t = train_test_split(X_tfidf, y_tfidf, test_size=0.3, random_state=42)
lr33 = LogisticRegression(max_iter=1000).fit(X_tr_t, y_tr_t)
y_pred33 = lr33.predict(X_te_t)
print(classification_report(y_te_t, y_pred33))
cm33 = confusion_matrix(y_te_t, y_pred33)
sns.heatmap(cm33, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusión — Clasificación de Puntuaciones")
plt.tight_layout(); plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. `print("=" * 70)` — Muestra el resultado por pantalla.
2. `print("EJERCICIO 33: CLASIFICACIÓN DE TEXTO — TF-IDF + LogisticRegression")` — Muestra el resultado por pantalla.
3. `print("=" * 70)` — Muestra el resultado por pantalla.
4. `print("Enunciado: Usa TfidfVectorizer + LogisticRegression para clasificar")` — Muestra el resultado por pantalla.
5. `print("reseñas en 5 categorías (puntuaciones 1-5). Evalúa con matriz de confusión.")` — Muestra el resultado por pantalla.
6. `print("Pista: TfidfVectorizer(max_features=1000)")` — Muestra el resultado por pantalla.
7. `print("")` — Muestra el resultado por pantalla.
8. `from sklearn.feature_extraction.text import TfidfVectorizer` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
print("=" * 70)
print("BLOQUE 8: INTEGRADORES (Ejercicios 34-35)")
print("=" * 70)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. `print("=" * 70)` — Muestra el resultado por pantalla.
2. `print("BLOQUE 8: INTEGRADORES (Ejercicios 34-35)")` — Muestra el resultado por pantalla.
3. `print("=" * 70)` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
print("=" * 70)
print("EJERCICIO 34: PIPELINE ML COMPLETO — Predicción de ingreso")
print("=" * 70)
print("Enunciado: Crea un pipeline completo que incluya: preprocesamiento,")
print("selección de features, RandomForestRegressor, evaluación y")
print("visualización de resultados. Incluye validación cruzada.")
print("Pista: make_pipeline(StandardScaler(), RandomForestRegressor())")
print("")

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
features_34 = ["cantidad", "descuento"]
if "precio_unitario" in ventas.columns:
    features_34.append("precio_unitario")
X_34 = ventas[features_34].fillna(0)
y_34 = ventas["ingreso"]
pipe = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100, random_state=42))
cv_scores = cross_val_score(pipe, X_34, y_34, cv=5, scoring="r2")
pipe.fit(X_34, y_34)
y_pred_34 = pipe.predict(X_34)
print(f"CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"Train R²: {r2_score(y_34, y_pred_34):.4f}")
print(f"MAE: ${mean_absolute_error(y_34, y_pred_34):.2f}")
plt.figure(figsize=(10, 6))
plt.scatter(y_34, y_pred_34, alpha=0.3, color="steelblue")
plt.plot([y_34.min(), y_34.max()], [y_34.min(), y_34.max()], "r--", linewidth=2)
plt.xlabel("Ingreso real ($)"); plt.ylabel("Ingreso predicho ($)")
plt.title("Pipeline ML: Real vs Predicho"); plt.tight_layout(); plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. `print("=" * 70)` — Muestra el resultado por pantalla.
2. `print("EJERCICIO 34: PIPELINE ML COMPLETO — Predicción de ingreso")` — Muestra el resultado por pantalla.
3. `print("=" * 70)` — Muestra el resultado por pantalla.
4. `print("Enunciado: Crea un pipeline completo que incluya: preprocesamiento,")` — Muestra el resultado por pantalla.
5. `print("selección de features, RandomForestRegressor, evaluación y")` — Muestra el resultado por pantalla.
6. `print("visualización de resultados. Incluye validación cruzada.")` — Muestra el resultado por pantalla.
7. `print("Pista: make_pipeline(StandardScaler(), RandomForestRegressor())")` — Muestra el resultado por pantalla.
8. `print("")` — Muestra el resultado por pantalla.
9. `from sklearn.pipeline import make_pipeline` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
print("=" * 70)
print("EJERCICIO 35: INTEGRADOR FINAL — Análisis completo de ventas")
print("=" * 70)
print("Enunciado: Combina todo lo aprendido en un solo análisis:")
print("1. Limpieza y EDA con Pandas + Seaborn")
print("2. Feature engineering con NumPy")
print("3. Modelo predictivo (Random Forest)")
print("4. Clustering de productos (K-Means)")
 print("5. Visualización integrada con interpretación de negocio")
print("Pista: organiza en funciones reutilizables.")
print("")

def reporte_ventas(df):
    print("=== REPORTE INTEGRAL DE VENTAS ===")
    print(f"Total transacciones: {len(df)}")
    print(f"Ingreso total: ${df['ingreso'].sum():,.2f}")
    print(f"Ingreso promedio: ${df['ingreso'].mean():.2f}")
    print(f"Ticket promedio: ${df.groupby('fecha')['ingreso'].sum().mean():.2f}/día")
    top_prod = df.groupby("producto")["ingreso"].sum().nlargest(3)
    print(f"Top 3 productos:")
    for prod, ing in top_prod.items():
        print(f"  {prod}: ${ing:,.2f}")
    print(f"Mejor sucursal: {df.groupby('sucursal')['ingreso'].sum().idxmax()}")

def segmentar_productos(df):
    print("\n=== SEGMENTACIÓN DE PRODUCTOS ===")
    agg = df.groupby("producto").agg(ingreso_total=("ingreso", "sum"), cantidad_total=("cantidad", "sum")).reset_index()
    scaler = StandardScaler()
    X_seg = scaler.fit_transform(agg[["ingreso_total", "cantidad_total"]])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    agg["segmento"] = kmeans.fit_predict(X_seg)
    for seg in sorted(agg["segmento"].unique()):
        prods = agg[agg["segmento"] == seg]["producto"].tolist()
        print(f"Segmento {seg}: {len(prods)} productos")
        print(f"  Ejemplos: {prods[:3]}")
    return agg

def modelo_predictivo(df):
    print("\n=== MODELO PREDICTIVO ===")
    X_mod = df[["cantidad", "descuento"]].fillna(0)
    y_mod = df["ingreso"]
    X_tr, X_te, y_tr, y_te = train_test_split(X_mod, y_mod, test_size=0.3, random_state=42)
    rf_mod = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_tr, y_tr)
    y_pred = rf_mod.predict(X_te)
    print(f"R² test: {r2_score(y_te, y_pred):.4f}")
    print(f"MAE test: ${mean_absolute_error(y_te, y_pred):.2f}")
    fi = pd.DataFrame({"feature": X_mod.columns, "importance": rf_mod.feature_importances_}).sort_values("importance", ascending=False)
    print("Importancia:")
    print(fi.to_string(index=False))
    return rf_mod

reporte_ventas(ventas)
segmentar_productos(ventas)
modelo_predictivo(ventas)

print("\n=== INTERPRETACIÓN DE NEGOCIO ===")
print("1. Los productos de alto ingreso deben priorizarse en inventario")
print("2. La segmentación revela grupos con diferente comportamiento de compra")
print("3. El modelo predictivo permite anticipar ingresos por transacción")
print("4. Recomendación: implementar dashboard con estos 3 componentes")
print("5. Próximo paso: incorporar datos de clientes para segmentación RFM")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. `print("=" * 70)` — Muestra el resultado por pantalla.
2. `print("EJERCICIO 35: INTEGRADOR FINAL — Análisis completo de ventas")` — Muestra el resultado por pantalla.
3. `print("=" * 70)` — Muestra el resultado por pantalla.
4. `print("Enunciado: Combina todo lo aprendido en un solo análisis:")` — Muestra el resultado por pantalla.
5. `print("1. Limpieza y EDA con Pandas + Seaborn")` — Muestra el resultado por pantalla.
6. `print("2. Feature engineering con NumPy")` — Muestra el resultado por pantalla.
7. `print("3. Modelo predictivo (Random Forest)")` — Muestra el resultado por pantalla.
8. `print("4. Clustering de productos (K-Means)")` — Muestra el resultado por pantalla.
9. `print("5. Visualización integrada con interpretación de negocio")` — Muestra el resultado por pantalla.
10. `print("Pista: organiza en funciones reutilizables.")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# EJERCICIOS COMPLEMENTARIOS
# ============================================================
print("=" * 70)
print("5 EJERCICIOS EXTRAS PARA PRACTICAR")
print("=" * 70)
print("""
Ejercicio 36: Implementa GradientBoostingRegressor y compara con RandomForest.
Ejercicio 37: Crea un sistema de recomendación simple (correlación entre productos).
Ejercicio 38: Analiza la estacionalidad con gráficos de violín por mes.
Ejercicio 39: Implementa detección de anomalías con IsolationForest en ventas.
Ejercicio 40: Construye un dashboard con plotly express de todas las métricas.
""")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. EJERCICIOS COMPLEMENTARIOS
3. ============================================================

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# RESUMEN EJECUTIVO
# ============================================================
print("=" * 70)
print("RESUMEN EJECUTIVO — 35 EJERCICIOS INTEGRADORES")
print("=" * 70)
print("""
Este compendio cubre 8 bloques temáticos del nivel intermedio:

Bloque 1: NumPy Avanzado (broadcasting, where, funciones maestras, álgebra lineal)
Bloque 2: Pandas Avanzado (groupby, merge, pivot, ventanas, fechas)
Bloque 3: Seaborn Experto (personalización, facetas, heatmaps, compuestos)
Bloque 4: SciPy Aplicado (optimización, interpolación, señales, estadística)
Bloque 5: ML Clásico (regresión, logística, árboles, Random Forest, KNN)
Bloque 6: Clustering (K-Means, DBSCAN, PCA, t-SNE, métricas)
Bloque 7: NLP (tokenización, Naive Bayes, TF-IDF, clasificación)
Bloque 8: Integradores (pipeline completo, reporte integral)

Habilidades desarrolladas:
- Manipulación eficiente de datos con NumPy y Pandas
- Visualización profesional con Seaborn y Matplotlib
- Modelos de machine learning supervisado y no supervisado
- Procesamiento de lenguaje natural básico
- Construcción de pipelines completos de datos a negocio
""")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. RESUMEN EJECUTIVO
3. ============================================================

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---


