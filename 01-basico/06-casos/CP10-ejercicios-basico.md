# CP10 — 30 Ejercicios Integradores de Nivel Básico

## 🎯 Contexto de Negocio

Este caso práctico consolida todo lo aprendido en los módulos anteriores (Python, NumPy, Pandas, Seaborn, SciPy, Sklearn) mediante 30 ejercicios progresivos aplicados al contexto de ventas, compras e inventarios.

Cada ejercicio incluye: enunciado de negocio, pista técnica, código de solución y explicación del resultado. Los ejercicios están ordenados por dificultad y cubren desde tipos básicos de Python hasta un pipeline completo de análisis.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style='whitegrid', palette='Set2')
plt.rcParams['figure.figsize'] = (10, 5)

ventas = pd.read_csv("../datos/ventas.csv")
inventario = pd.read_csv("../datos/inventario.csv")
compras = pd.read_csv("../datos/compras.csv")
clientes = pd.read_csv("../datos/clientes.csv")
ventas['fecha'] = pd.to_datetime(ventas['fecha'])
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*🎯 Contexto de Negocio.*

1. `import pandas as pd` — Importa las librerías necesarias para el análisis.
2. `import numpy as np` — Importa las librerías necesarias para el análisis.
3. `import matplotlib.pyplot as plt` — Importa las librerías necesarias para el análisis.
4. `import seaborn as sns` — Importa las librerías necesarias para el análisis.
5. `from scipy import stats` — Importa las librerías necesarias para el análisis.
6. `import warnings` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 📘 EJERCICIOS 1-5: Python (Tipos, Listas, Dicts, Funciones)

### Ejercicio 1: Calcular Ticket Promedio

**Enunciado de negocio:** El gerente quiere saber el ticket promedio de una transacción. Calcula el ingreso total y la cantidad de transacciones, luego divide.

**Pista:** Usa `len()` para transacciones y `sum()` con lista de ingresos.

```python
# Solución
ingresos = ventas['ingreso'].tolist()
total_ingresos = sum(ingresos)
num_transacciones = len(ingresos)
ticket_promedio = total_ingresos / num_transacciones
print(f"Ingreso total: ${total_ingresos:,.0f}")
print(f"Transacciones: {num_transacciones}")
print(f"Ticket promedio: ${ticket_promedio:,.0f}")
```

**Explicación:** Usamos listas de Python para calcular métricas básicas. Este cálculo simple pero fundamental es la base de cualquier análisis de ventas.

---

### Ejercicio 2: Filtrar Productos con List Comprehension

**Enunciado de negocio:** Crea una lista de productos cuyo precio_unitario sea mayor a 5000. ¿Cuántos productos premium se vendieron?

**Pista:** Usa list comprehension: `[item for item in lista if condición]`



**Explicación:** Usamos listas de Python para calcular métricas básicas. Este cálculo simple pero fundamental es la base de cualquier análisis de ventas.

---

### Ejercicio 2: Filtrar Productos con List Comprehension

**Enunciado de negocio:** Crea una lista de productos cuyo precio_unitario sea mayor a 5000. ¿Cuántos productos premium se vendieron?

**Pista:** Usa list comprehension: `[item for item in lista if condición]`

```python
# Solución
precios = ventas['precio_unitario'].tolist()
productos_premium = [p for p in precios if p > 5000]
print(f"Total transacciones: {len(precios)}")
print(f"Transacciones premium (>$5000): {len(productos_premium)}")
print(f"Porcentaje premium: {len(productos_premium)/len(precios)*100:.1f}%")
```

**Explicación:** List comprehension es una forma eficiente y pythonica de filtrar datos. Aquí identificamos cuántas transacciones son de productos de alto valor (premium).

---

### Ejercicio 3: Diccionario de Métricas por Sucursal

**Enunciado de negocio:** Crea un diccionario donde cada sucursal tenga su ingreso total, transacciones y ticket promedio.

**Pista:** Usa un diccionario anidado: `{'sucursal': {'ingreso': X, 'transacciones': Y}}`



**Explicación:** List comprehension es una forma eficiente y pythonica de filtrar datos. Aquí identificamos cuántas transacciones son de productos de alto valor (premium).

---

### Ejercicio 3: Diccionario de Métricas por Sucursal

**Enunciado de negocio:** Crea un diccionario donde cada sucursal tenga su ingreso total, transacciones y ticket promedio.

**Pista:** Usa un diccionario anidado: `{'sucursal': {'ingreso': X, 'transacciones': Y}}`

```python
# Solución
sucursales = ventas['sucursal'].unique()
metricas = {}
for suc in sucursales:
    filtro = ventas[ventas['sucursal'] == suc]
    metricas[suc] = {
        'ingreso_total': filtro['ingreso'].sum(),
        'transacciones': len(filtro),
        'ticket_promedio': filtro['ingreso'].mean()
    }

for suc, m in sorted(metricas.items(), key=lambda x: x[1]['ingreso_total'], reverse=True):
    print(f"{suc}: ${m['ingreso_total']:,.0f} | {m['transacciones']} trans | ${m['ticket_promedio']:,.0f} ticket")
```

**Explicación:** Los diccionarios permiten estructurar datos jerárquicamente. Este patrón es útil para construir tableros de control en código puro Python antes de usar Pandas.

---

### Ejercicio 4: Función para Calcular Margen

**Enunciado de negocio:** Escribe una función que calcule el margen porcentual dado ingreso y costo. Si el costo es 0, devuelve 0.

**Pista:** `def calcular_margen(ingreso, costo): return ((ingreso - costo) / ingreso) * 100 if ingreso > 0 else 0`



**Explicación:** Los diccionarios permiten estructurar datos jerárquicamente. Este patrón es útil para construir tableros de control en código puro Python antes de usar Pandas.

---

### Ejercicio 4: Función para Calcular Margen

**Enunciado de negocio:** Escribe una función que calcule el margen porcentual dado ingreso y costo. Si el costo es 0, devuelve 0.

**Pista:** `def calcular_margen(ingreso, costo): return ((ingreso - costo) / ingreso) * 100 if ingreso > 0 else 0`

```python
# Solución
def calcular_margen(ingreso, costo):
    if ingreso <= 0:
        return 0.0
    return round((ingreso - costo) / ingreso * 100, 2)

# Probar con primeros 5 registros
for i in range(5):
    row = ventas.iloc[i]
    margen = calcular_margen(row['ingreso'], row['costo_total'])
    print(f"{row['producto']}: ingreso=${row['ingreso']:.0f}, margen={margen}%")
```

**Explicación:** Las funciones permiten reutilizar lógica de negocio. Esta función de margen puede aplicarse a cualquier fila o agregación.

---

### Ejercicio 5: Procesar por Lotes con Map

**Enunciado de negocio:** Aplica la función de margen a todas las transacciones usando `map()` para crear una lista de márgenes.

**Pista:** `list(map(funcion, lista1, lista2))` o usa lambda.



**Explicación:** Las funciones permiten reutilizar lógica de negocio. Esta función de margen puede aplicarse a cualquier fila o agregación.

---

### Ejercicio 5: Procesar por Lotes con Map

**Enunciado de negocio:** Aplica la función de margen a todas las transacciones usando `map()` para crear una lista de márgenes.

**Pista:** `list(map(funcion, lista1, lista2))` o usa lambda.

```python
# Solución
ingresos = ventas['ingreso'].tolist()
costos = ventas['costo_total'].tolist()
margenes = list(map(lambda i, c: round((i - c) / i * 100, 2) if i > 0 else 0, ingresos, costos))
print(f"Mínimo margen: {min(margenes):.1f}%")
print(f"Máximo margen: {max(margenes):.1f}%")
print(f"Margen promedio: {sum(margenes)/len(margenes):.1f}%")
```

**Explicación:** `map()` aplica una función a múltiples iterables sin usar loops explícitos. Es más eficiente y expresivo para transformaciones de datos.

---

## 📘 EJERCICIOS 6-10: NumPy (Arrays, Operaciones, Estadísticas)

### Ejercicio 6: Array de Ingresos y Estadísticas Básicas

**Enunciado de negocio:** Convierte la columna de ingresos a un array de NumPy y calcula media, mediana, desviación estándar, mínimo y máximo.

**Pista:** `np.mean()`, `np.median()`, `np.std()`, `np.min()`, `np.max()`



**Explicación:** `map()` aplica una función a múltiples iterables sin usar loops explícitos. Es más eficiente y expresivo para transformaciones de datos.

---

## 📘 EJERCICIOS 6-10: NumPy (Arrays, Operaciones, Estadísticas)

### Ejercicio 6: Array de Ingresos y Estadísticas Básicas

**Enunciado de negocio:** Convierte la columna de ingresos a un array de NumPy y calcula media, mediana, desviación estándar, mínimo y máximo.

**Pista:** `np.mean()`, `np.median()`, `np.std()`, `np.min()`, `np.max()`

```python
# Solución
ingresos_np = np.array(ventas['ingreso'])
print(f"Media: ${ingresos_np.mean():,.0f}")
print(f"Mediana: ${np.median(ingresos_np):,.0f}")
print(f"Desviación estándar: ${ingresos_np.std():,.0f}")
print(f"Mínimo: ${ingresos_np.min():,.0f}")
print(f"Máximo: ${ingresos_np.max():,.0f}")
print(f"Diferencia media vs mediana: ${ingresos_np.mean() - np.median(ingresos_np):,.0f}")
```

**Explicación:** NumPy ofrece operaciones vectorizadas mucho más rápidas que Python puro. La diferencia entre media y mediana indica asimetría en la distribución de ingresos.

---

### Ejercicio 7: Filtro Boolean Indexing

**Enunciado de negocio:** Encuentra las transacciones con ingreso mayor al percentil 90. ¿Cuántas hay y cuál es su valor total?

**Pista:** `arr[arr > np.percentile(arr, 90)]`



**Explicación:** NumPy ofrece operaciones vectorizadas mucho más rápidas que Python puro. La diferencia entre media y mediana indica asimetría en la distribución de ingresos.

---

### Ejercicio 7: Filtro Boolean Indexing

**Enunciado de negocio:** Encuentra las transacciones con ingreso mayor al percentil 90. ¿Cuántas hay y cuál es su valor total?

**Pista:** `arr[arr > np.percentile(arr, 90)]`

```python
# Solución
p90 = np.percentile(ingresos_np, 90)
transacciones_top = ingresos_np[ingresos_np > p90]
print(f"Percentil 90: ${p90:,.0f}")
print(f"Transacciones top: {len(transacciones_top)} ({len(transacciones_top)/len(ingresos_np)*100:.1f}%)")
print(f"Valor total top: ${transacciones_top.sum():,.0f}")
print(f"Promedio top: ${transacciones_top.mean():,.0f}")
```

**Explicación:** Boolean indexing permite filtrar arrays de forma eficiente. Identificar las transacciones del top 10% ayuda a enfocar esfuerzos en los clientes y productos de mayor valor.

---

### Ejercicio 8: Operaciones Vectorizadas para Métricas

**Enunciado de negocio:** Calcula el margen en dólares y porcentaje para todas las transacciones usando operaciones vectorizadas de NumPy.

**Pista:** `margen = ingreso - costo; margen_pct = (margen / ingreso) * 100`



**Explicación:** Boolean indexing permite filtrar arrays de forma eficiente. Identificar las transacciones del top 10% ayuda a enfocar esfuerzos en los clientes y productos de mayor valor.

---

### Ejercicio 8: Operaciones Vectorizadas para Métricas

**Enunciado de negocio:** Calcula el margen en dólares y porcentaje para todas las transacciones usando operaciones vectorizadas de NumPy.

**Pista:** `margen = ingreso - costo; margen_pct = (margen / ingreso) * 100`

```python
# Solución
ingreso_arr = ventas['ingreso'].values
costo_arr = ventas['costo_total'].values
margen_arr = ingreso_arr - costo_arr
margen_pct_arr = np.where(ingreso_arr > 0, margen_arr / ingreso_arr * 100, 0)

print(f"Margen total: ${margen_arr.sum():,.0f}")
print(f"Margen promedio: ${margen_arr.mean():,.0f}")
print(f"Margen% promedio: {margen_pct_arr.mean():.1f}%")
print(f"Transacciones con margen negativo: {(margen_arr < 0).sum()}")
```

**Explicación:** Las operaciones vectorizadas evitan loops y son órdenes de magnitud más rápidas. Con una línea calculamos el margen de las 1330 transacciones simultáneamente.

---

### Ejercicio 9: Matriz de Correlación con NumPy

**Enunciado de negocio:** Crea una matriz NumPy con las columnas numéricas relevantes y calcula su matriz de correlación.

**Pista:** `np.corrcoef(matriz.T)`



**Explicación:** Las operaciones vectorizadas evitan loops y son órdenes de magnitud más rápidas. Con una línea calculamos el margen de las 1330 transacciones simultáneamente.

---

### Ejercicio 9: Matriz de Correlación con NumPy

**Enunciado de negocio:** Crea una matriz NumPy con las columnas numéricas relevantes y calcula su matriz de correlación.

**Pista:** `np.corrcoef(matriz.T)`

```python
# Solución
cols_num = ['cantidad', 'precio_unitario', 'costo_unitario', 'ingreso', 'costo_total', 'margen', 'descuento']
matriz = ventas[cols_num].values
corr = np.corrcoef(matriz.T)

print("Matriz de correlación:")
print(pd.DataFrame(corr, index=cols_num, columns=cols_num).round(3))

corr_flat = corr.flatten()
corr_sin_diag = corr_flat[corr_flat < 1]
print(f"Correlación máxima (excluyendo diagonal): {corr_sin_diag.max():.3f}")
```

**Explicación:** La matriz de correlación revela relaciones lineales entre variables. Saber que ingreso y cantidad tienen alta correlación es esperable, pero detectar correlaciones inesperadas guía el análisis.

---

### Ejercicio 10: Simulación de Monte Carlo para Ingresos

**Enunciado de negocio:** Simula 10,000 escenarios de ingreso total anual usando bootstrap (remuestreo con reemplazo) para estimar el rango probable.

**Pista:** `np.random.choice(arr, size=(n_scenarios, len(arr))).sum(axis=1)`



**Explicación:** La matriz de correlación revela relaciones lineales entre variables. Saber que ingreso y cantidad tienen alta correlación es esperable, pero detectar correlaciones inesperadas guía el análisis.

---

### Ejercicio 10: Simulación de Monte Carlo para Ingresos

**Enunciado de negocio:** Simula 10,000 escenarios de ingreso total anual usando bootstrap (remuestreo con reemplazo) para estimar el rango probable.

**Pista:** `np.random.choice(arr, size=(n_scenarios, len(arr))).sum(axis=1)`

```python
# Solución
np.random.seed(42)
n_simulaciones = 10000
simulaciones = np.random.choice(ingresos_np, size=(n_simulaciones, len(ingresos_np)), replace=True)
ingresos_simulados = simulaciones.sum(axis=1)

p5 = np.percentile(ingresos_simulados, 5)
p50 = np.percentile(ingresos_simulados, 50)
p95 = np.percentile(ingresos_simulados, 95)

print(f"Ingreso real: ${ingresos_np.sum():,.0f}")
print(f"Simulación (P50): ${p50:,.0f}")
print(f"Intervalo 90%: ${p5:,.0f} - ${p95:,.0f}")
```

**Explicación:** El bootstrap permite estimar incertidumbre sin asumir distribución teórica. Útil para rangos de ingresos esperados en lugar de un número puntual.

---

## 📘 EJERCICIOS 11-15: Pandas (DataFrame, Filtros, Groupby)

### Ejercicio 11: Resumen Descriptivo por Categoría

**Enunciado de negocio:** Agrupa ventas por categoría y calcula ingreso total, promedio, transacciones y margen promedio.

**Pista:** `ventas.groupby('categoria').agg({'ingreso': ['sum', 'mean', 'count'], 'margen_pct': 'mean'})`



**Explicación:** El bootstrap permite estimar incertidumbre sin asumir distribución teórica. Útil para rangos de ingresos esperados en lugar de un número puntual.

---

## 📘 EJERCICIOS 11-15: Pandas (DataFrame, Filtros, Groupby)

### Ejercicio 11: Resumen Descriptivo por Categoría

**Enunciado de negocio:** Agrupa ventas por categoría y calcula ingreso total, promedio, transacciones y margen promedio.

**Pista:** `ventas.groupby('categoria').agg({'ingreso': ['sum', 'mean', 'count'], 'margen_pct': 'mean'})`

```python
# Solución
resumen_cat = ventas.groupby('categoria').agg(
    ingreso_total=('ingreso', 'sum'),
    ingreso_promedio=('ingreso', 'mean'),
    transacciones=('ingreso', 'count'),
    margen_promedio=('margen_pct', 'mean')
).reset_index().sort_values('ingreso_total', ascending=False)

print("Resumen por categoría:")
print(resumen_cat.to_string(index=False))
print(f"Categoría con mayor ingreso: {resumen_cat.iloc[0]['categoria']}")
print(f"Categoría con mayor margen: {resumen_cat.sort_values('margen_promedio', ascending=False).iloc[0]['categoria']}")
```

**Explicación:** El groupby con agg es la herramienta más poderosa de Pandas para resumir datos. Permite múltiples agregaciones en una sola línea.

---

### Ejercicio 12: Filtrar con Múltiples Condiciones

**Enunciado de negocio:** Encuentra transacciones de Electrónica con ingreso > 5000 y descuento < 0.10. ¿Cuántas hay y cuál es su ingreso total?

**Pista:** `ventas[(cond1) & (cond2) & (cond3)]`



**Explicación:** El groupby con agg es la herramienta más poderosa de Pandas para resumir datos. Permite múltiples agregaciones en una sola línea.

---

### Ejercicio 12: Filtrar con Múltiples Condiciones

**Enunciado de negocio:** Encuentra transacciones de Electrónica con ingreso > 5000 y descuento < 0.10. ¿Cuántas hay y cuál es su ingreso total?

**Pista:** `ventas[(cond1) & (cond2) & (cond3)]`

```python
# Solución
filtro = ventas[
    (ventas['categoria'] == 'Electrónica') &
    (ventas['ingreso'] > 5000) &
    (ventas['descuento'] < 0.10)
]
print(f"Transacciones: {len(filtro)}")
print(f"Ingreso total: ${filtro['ingreso'].sum():,.0f}")
print(f"Ingreso promedio: ${filtro['ingreso'].mean():,.0f}")
```

**Explicación:** El filtrado con múltiples condiciones es esencial para responder preguntas de negocio específicas. Cada paréntesis es una condición independiente.

---

### Ejercicio 13: Top 5 Productos con rank()

**Enunciado de negocio:** Encuentra el top 5 de productos por ingreso total usando `rank()`.

**Pista:** Agrupa, suma, usa `rank(method='dense', ascending=False)`



**Explicación:** El filtrado con múltiples condiciones es esencial para responder preguntas de negocio específicas. Cada paréntesis es una condición independiente.

---

### Ejercicio 13: Top 5 Productos con rank()

**Enunciado de negocio:** Encuentra el top 5 de productos por ingreso total usando `rank()`.

**Pista:** Agrupa, suma, usa `rank(method='dense', ascending=False)`

```python
# Solución
prod_ingreso = ventas.groupby(['producto', 'categoria']).agg(ingreso_total=('ingreso', 'sum')).reset_index()
prod_ingreso['ranking'] = prod_ingreso['ingreso_total'].rank(method='dense', ascending=False)
top5 = prod_ingreso[prod_ingreso['ranking'] <= 5].sort_values('ranking')
print(top5[['ranking', 'producto', 'categoria', 'ingreso_total']].to_string(index=False))
```

**Explicación:** `rank()` asigna posición según una columna. Es más flexible que sort+head porque maneja empates y permite filtrar dinámicamente.

---

### Ejercicio 14: Porcentaje con transform()

**Enunciado de negocio:** Calcula qué % del ingreso de su categoría representa cada producto usando `transform('sum')`.

**Pista:** `ventas.groupby('categoria')['ingreso'].transform('sum')`



**Explicación:** `rank()` asigna posición según una columna. Es más flexible que sort+head porque maneja empates y permite filtrar dinámicamente.

---

### Ejercicio 14: Porcentaje con transform()

**Enunciado de negocio:** Calcula qué % del ingreso de su categoría representa cada producto usando `transform('sum')`.

**Pista:** `ventas.groupby('categoria')['ingreso'].transform('sum')`

```python
# Solución
ventas['ingreso_categoria'] = ventas.groupby('categoria')['ingreso'].transform('sum')
ventas['porcentaje_categoria'] = ventas['ingreso'] / ventas['ingreso_categoria'] * 100
resumen_pct = ventas.groupby('producto').agg(
    ingreso_producto=('ingreso', 'sum'),
    pct_categoria=('porcentaje_categoria', 'mean')
).reset_index().sort_values('pct_categoria', ascending=False)
print(resumen_pct.head(10).to_string(index=False))
```

**Explicación:** `transform()` propaga la agregación a cada fila sin colapsar el DataFrame. Ideal para cálculos de contribución porcentual.

---

### Ejercicio 15: Clientes Multi-Categoría

**Enunciado de negocio:** Encuentra clientes que han comprado en más de 2 categorías diferentes.

**Pista:** `ventas.groupby('cliente')['categoria'].nunique()`



**Explicación:** `transform()` propaga la agregación a cada fila sin colapsar el DataFrame. Ideal para cálculos de contribución porcentual.

---

### Ejercicio 15: Clientes Multi-Categoría

**Enunciado de negocio:** Encuentra clientes que han comprado en más de 2 categorías diferentes.

**Pista:** `ventas.groupby('cliente')['categoria'].nunique()`

```python
# Solución
cats_por_cliente = ventas.groupby('cliente')['categoria'].nunique().reset_index()
cats_por_cliente.columns = ['cliente', 'categorias_distintas']
multicategoria = cats_por_cliente[cats_por_cliente['categorias_distintas'] > 2]
print(f"Total clientes: {len(cats_por_cliente)}")
print(f"Clientes multi-categoría: {len(multicategoria)} ({len(multicategoria)/len(cats_por_cliente)*100:.1f}%)")
```

**Explicación:** Los clientes que compran en múltiples categorías tienen mayor valor potencial. Son candidatos a programas de fidelización cruzada.

---

## 📘 EJERCICIOS 16-20: Pandas (Merge, Pivot, Fechas)

### Ejercicio 16: Merge Ventas con Inventario

**Enunciado de negocio:** Combina ventas con inventario para añadir el costo unitario desde la tabla de productos.

**Pista:** `pd.merge(ventas, inventario[['sku', 'costo', 'stock_actual']], on='sku', how='left')`



**Explicación:** Los clientes que compran en múltiples categorías tienen mayor valor potencial. Son candidatos a programas de fidelización cruzada.

---

## 📘 EJERCICIOS 16-20: Pandas (Merge, Pivot, Fechas)

### Ejercicio 16: Merge Ventas con Inventario

**Enunciado de negocio:** Combina ventas con inventario para añadir el costo unitario desde la tabla de productos.

**Pista:** `pd.merge(ventas, inventario[['sku', 'costo', 'stock_actual']], on='sku', how='left')`

```python
# Solución
ventas_con_inv = ventas.merge(
    inventario[['sku', 'costo', 'stock_actual', 'producto']],
    on='sku', how='left', suffixes=('_venta', '_inv')
)
print(ventas_con_inv[['sku', 'producto_venta', 'producto_inv', 'costo', 'stock_actual']].head())
print(f"\nFilas con producto sin match: {ventas_con_inv['producto_inv'].isna().sum()}")
```

**Explicación:** Merge es esencial para enriquecer datos de transacciones con información maestra de productos. El `how='left'` preserva todas las transacciones aunque falten productos en inventario.

---

### Ejercicio 17: Pivot Table de Ventas por Sucursal y Mes

**Enunciado de negocio:** Crea una tabla pivote con sucursales como filas, meses como columnas e ingreso total como valores.

**Pista:** `pd.pivot_table(ventas, values='ingreso', index='sucursal', columns='mes', aggfunc='sum')`



**Explicación:** Merge es esencial para enriquecer datos de transacciones con información maestra de productos. El `how='left'` preserva todas las transacciones aunque falten productos en inventario.

---

### Ejercicio 17: Pivot Table de Ventas por Sucursal y Mes

**Enunciado de negocio:** Crea una tabla pivote con sucursales como filas, meses como columnas e ingreso total como valores.

**Pista:** `pd.pivot_table(ventas, values='ingreso', index='sucursal', columns='mes', aggfunc='sum')`

```python
# Solución
ventas['mes'] = ventas['fecha'].dt.month
pivot_mes = pd.pivot_table(ventas, values='ingreso', index='sucursal',
                           columns='mes', aggfunc='sum', fill_value=0)
pivot_mes.columns = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                     'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
print(pivot_mes.round(0))
```

**Explicación:** Pivot tables transforman datos largos a anchos para comparar dimensiones cruzadas. Útil para reportes ejecutivos y heatmaps.

---

### Ejercicio 18: Cliente con Mayor Gasto por Mes

**Enunciado de negocio:** Para cada mes, encuentra el cliente que más gastó.

**Pista:** Agrupa por mes y cliente, suma, luego usa `idxmax()` o `rank()`.



**Explicación:** Pivot tables transforman datos largos a anchos para comparar dimensiones cruzadas. Útil para reportes ejecutivos y heatmaps.

---

### Ejercicio 18: Cliente con Mayor Gasto por Mes

**Enunciado de negocio:** Para cada mes, encuentra el cliente que más gastó.

**Pista:** Agrupa por mes y cliente, suma, luego usa `idxmax()` o `rank()`.

```python
# Solución
gasto_mes_cliente = ventas.groupby(['mes', 'cliente'])['ingreso'].sum().reset_index()
idx_max = gasto_mes_cliente.groupby('mes')['ingreso'].idxmax()
top_cliente_mes = gasto_mes_cliente.loc[idx_max].sort_values('mes')
top_cliente_mes['mes_nombre'] = top_cliente_mes['mes'].map({
    1:'Ene',2:'Feb',3:'Mar',4:'Abr',5:'May',6:'Jun',
    7:'Jul',8:'Ago',9:'Sep',10:'Oct',11:'Nov',12:'Dic'})
print(top_cliente_mes[['mes_nombre', 'cliente', 'ingreso']].to_string(index=False))
```

**Explicación:** Encontrar el cliente estrella de cada mes permite diseñar programas de reconocimiento y analizar estacionalidad de clientes top.

---

### Ejercicio 19: Días Entre Compras (Cliente)

**Enunciado de negocio:** Para un cliente específico, calcula los días transcurridos entre cada compra consecutiva.

**Pista:** Filtra por cliente, ordena por fecha, usa `diff()`.



**Explicación:** Encontrar el cliente estrella de cada mes permite diseñar programas de reconocimiento y analizar estacionalidad de clientes top.

---

### Ejercicio 19: Días Entre Compras (Cliente)

**Enunciado de negocio:** Para un cliente específico, calcula los días transcurridos entre cada compra consecutiva.

**Pista:** Filtra por cliente, ordena por fecha, usa `diff()`.

```python
# Solución
cliente_ejemplo = ventas['cliente'].value_counts().index[0]
compras_cliente = ventas[ventas['cliente'] == cliente_ejemplo].copy()
compras_cliente = compras_cliente.sort_values('fecha')
compras_cliente['dias_entre_compras'] = compras_cliente['fecha'].diff().dt.days

print(f"Cliente: {cliente_ejemplo}")
print(f"Total compras: {len(compras_cliente)}")
print(f"Días promedio entre compras: {compras_cliente['dias_entre_compras'].mean():.1f}")
print(compras_cliente[['fecha', 'ingreso', 'dias_entre_compras']].head(10).to_string(index=False))
```

**Explicación:** La frecuencia de compra es un indicador de lealtad. Días entre compras con poca variación indica un hábito de compra estable, ideal para programas de suscripción.

---

### Ejercicio 20: Cohortes Mensuales de Retención

**Enunciado de negocio:** Crea una matriz de cohortes mostrando cuántos clientes compraron en el mes 0 (primera compra) y en meses siguientes.

**Pista:** Asigna mes de primera compra con `transform('min')`, luego cruza con tabla pivote.



**Explicación:** La frecuencia de compra es un indicador de lealtad. Días entre compras con poca variación indica un hábito de compra estable, ideal para programas de suscripción.

---

### Ejercicio 20: Cohortes Mensuales de Retención

**Enunciado de negocio:** Crea una matriz de cohortes mostrando cuántos clientes compraron en el mes 0 (primera compra) y en meses siguientes.

**Pista:** Asigna mes de primera compra con `transform('min')`, luego cruza con tabla pivote.

```python
# Solución
ventas_cohorte = ventas.copy()
ventas_cohorte['mes_compra'] = ventas_cohorte['fecha'].dt.to_period('M')
ventas_cohorte['mes_primera_compra'] = ventas_cohorte.groupby('cliente')['mes_compra'].transform('min')
ventas_cohorte['cohorte'] = (ventas_cohorte['mes_compra'] - ventas_cohorte['mes_primera_compra']).apply(lambda x: x.n)

cohorte = ventas_cohorte.groupby(['mes_primera_compra', 'cohorte'])['cliente'].nunique().reset_index()
cohorte_pivot = cohorte.pivot_table(index='mes_primera_compra', columns='cohorte', values='cliente', fill_value=0)

# Retención porcentual
cohorte_pct = cohorte_pivot.divide(cohorte_pivot[0], axis=0) * 100
print("Matriz de retención de cohortes (%):")
print(cohorte_pct.round(1))
```

**Explicación:** El análisis de cohortes revela la retención de clientes a lo largo del tiempo. Cohortes con alta retención indican campañas exitosas o temporadas favorables.

---

## 📘 EJERCICIOS 21-25: Seaborn (Gráficos, Distribuciones, Categorías)

### Ejercicio 21: Histograma de Ingresos

**Enunciado de negocio:** Visualiza la distribución de ingresos por transacción con un histograma. Identifica si hay asimetría.

**Pista:** `sns.histplot(data=ventas, x='ingreso', bins=30, kde=True)`



**Explicación:** El análisis de cohortes revela la retención de clientes a lo largo del tiempo. Cohortes con alta retención indican campañas exitosas o temporadas favorables.

---

## 📘 EJERCICIOS 21-25: Seaborn (Gráficos, Distribuciones, Categorías)

### Ejercicio 21: Histograma de Ingresos

**Enunciado de negocio:** Visualiza la distribución de ingresos por transacción con un histograma. Identifica si hay asimetría.

**Pista:** `sns.histplot(data=ventas, x='ingreso', bins=30, kde=True)`

```python
# Solución
plt.figure(figsize=(10, 4))
sns.histplot(data=ventas, x='ingreso', bins=40, kde=True, color='steelblue', edgecolor='white')
plt.axvline(ventas['ingreso'].mean(), color='red', linestyle='--', label=f'Media: ${ventas["ingreso"].mean():.0f}')
plt.axvline(ventas['ingreso'].median(), color='green', linestyle='--', label=f'Mediana: ${ventas["ingreso"].median():.0f}')
plt.title('Distribución de Ingresos por Transacción', fontweight='bold')
plt.xlabel('Ingreso ($)')
plt.ylabel('Frecuencia')
plt.legend()
plt.tight_layout()
plt.show()
```

**Explicación:** El histograma muestra la forma de la distribución. Si la cola derecha es larga (asimetría positiva), hay pocas transacciones muy grandes que jalan la media hacia arriba.

---

### Ejercicio 22: Boxplot de Ingresos por Categoría

**Enunciado de negocio:** Compara la distribución de ingresos entre categorías usando un boxplot. ¿Qué categoría tiene mayor mediana?

**Pista:** `sns.boxplot(data=ventas, x='categoria', y='ingreso')`



**Explicación:** El histograma muestra la forma de la distribución. Si la cola derecha es larga (asimetría positiva), hay pocas transacciones muy grandes que jalan la media hacia arriba.

---

### Ejercicio 22: Boxplot de Ingresos por Categoría

**Enunciado de negocio:** Compara la distribución de ingresos entre categorías usando un boxplot. ¿Qué categoría tiene mayor mediana?

**Pista:** `sns.boxplot(data=ventas, x='categoria', y='ingreso')`

```python
# Solución
plt.figure(figsize=(10, 4))
order = ventas.groupby('categoria')['ingreso'].median().sort_values(ascending=False).index
sns.boxplot(data=ventas, x='categoria', y='ingreso', order=order, palette='Set2', hue='categoria', legend=False)
plt.title('Distribución de Ingresos por Categoría', fontweight='bold')
plt.xlabel('Categoría')
plt.ylabel('Ingreso ($)')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

medianas = ventas.groupby('categoria')['ingreso'].median().sort_values(ascending=False)
print("Medianas por categoría:")
print(medianas)
```

**Explicación:** El boxplot revela diferencias en distribución y outliers. Categorías con cajas más altas tienen tickets más caros. Outliers son transacciones excepcionales que merecen investigación.

---

### Ejercicio 23: Barplot de Ingreso por Sucursal

**Enunciado de negocio:** Crea un barplot del ingreso total por sucursal, ordenado de mayor a menor.

**Pista:** `sns.barplot(data=agrupado, y='sucursal', x='ingreso')` con datos pre-agrupados.



**Explicación:** El boxplot revela diferencias en distribución y outliers. Categorías con cajas más altas tienen tickets más caros. Outliers son transacciones excepcionales que merecen investigación.

---

### Ejercicio 23: Barplot de Ingreso por Sucursal

**Enunciado de negocio:** Crea un barplot del ingreso total por sucursal, ordenado de mayor a menor.

**Pista:** `sns.barplot(data=agrupado, y='sucursal', x='ingreso')` con datos pre-agrupados.

```python
# Solución
ingreso_suc = ventas.groupby('sucursal')['ingreso'].sum().reset_index().sort_values('ingreso')
plt.figure(figsize=(10, 4))
sns.barplot(data=ingreso_suc, y='sucursal', x='ingreso', palette='viridis', hue='sucursal', legend=False)
for _, row in ingreso_suc.iterrows():
    plt.text(row['ingreso'] + 50000, row['sucursal'], f'${row["ingreso"]/1e6:.1f}M', va='center')
plt.title('Ingreso Total por Sucursal', fontweight='bold')
plt.xlabel('Ingreso ($)')
plt.ylabel('Sucursal')
plt.tight_layout()
plt.show()
```

**Explicación:** Barplots ordenados permiten comparar visualmente el desempeño. La diferencia entre la primera y última sucursal es la brecha de mejora potencial.

---

### Ejercicio 24: Scatterplot Precio vs Ingreso

**Enunciado de negocio:** Grafica la relación entre precio_unitario e ingreso. ¿Son directamente proporcionales?

**Pista:** `sns.scatterplot(data=ventas, x='precio_unitario', y='ingreso', alpha=0.5)`



**Explicación:** Barplots ordenados permiten comparar visualmente el desempeño. La diferencia entre la primera y última sucursal es la brecha de mejora potencial.

---

### Ejercicio 24: Scatterplot Precio vs Ingreso

**Enunciado de negocio:** Grafica la relación entre precio_unitario e ingreso. ¿Son directamente proporcionales?

**Pista:** `sns.scatterplot(data=ventas, x='precio_unitario', y='ingreso', alpha=0.5)`

```python
# Solución
plt.figure(figsize=(10, 4))
sns.scatterplot(data=ventas.sample(500), x='precio_unitario', y='ingreso',
                hue='categoria', alpha=0.6, size='cantidad', sizes=(20, 200))
plt.title('Relación Precio Unitario vs Ingreso', fontweight='bold')
plt.xlabel('Precio Unitario ($)')
plt.ylabel('Ingreso ($)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

corr = ventas['precio_unitario'].corr(ventas['ingreso'])
print(f"Correlación precio vs ingreso: {corr:.3f}")
```

**Explicación:** El scatterplot muestra relación no siempre lineal. Productos caros no siempre generan más ingreso porque se venden en menor cantidad. El color por categoría añade contexto.

---

### Ejercicio 25: Pairplot de Variables Numéricas

**Enunciado de negocio:** Crea un pairplot de las principales variables numéricas para visualizar relaciones multivariadas.

**Pista:** `sns.pairplot(ventas[cols], diag_kind='kde', hue='categoria')`



**Explicación:** El scatterplot muestra relación no siempre lineal. Productos caros no siempre generan más ingreso porque se venden en menor cantidad. El color por categoría añade contexto.

---

### Ejercicio 25: Pairplot de Variables Numéricas

**Enunciado de negocio:** Crea un pairplot de las principales variables numéricas para visualizar relaciones multivariadas.

**Pista:** `sns.pairplot(ventas[cols], diag_kind='kde', hue='categoria')`

```python
# Solución
cols_pair = ['cantidad', 'precio_unitario', 'ingreso', 'margen_pct', 'descuento']
sample = ventas.sample(300, random_state=42)
g = sns.pairplot(sample[cols_pair + ['categoria']], diag_kind='kde',
                 hue='categoria', palette='Set2', height=2.5)
g.fig.suptitle('Pairplot de Variables Numéricas', y=1.02, fontweight='bold')
plt.tight_layout()
plt.show()
```

**Explicación:** El pairplot muestra todas las relaciones bivariadas en una matriz. Revela clusters, outliers y correlaciones no evidentes en análisis univariados.

---

## 📘 EJERCICIOS 26-27: SciPy (Estadística, Tests)

### Ejercicio 26: Prueba t-test: Diferencia de Ingresos entre Dos Sucursales

**Enunciado de negocio:** ¿Hay diferencia significativa en el ingreso promedio entre las dos sucursales con mayor y menor ingreso?

**Pista:** `stats.ttest_ind(muestra1, muestra2)`



**Explicación:** El pairplot muestra todas las relaciones bivariadas en una matriz. Revela clusters, outliers y correlaciones no evidentes en análisis univariados.

---

## 📘 EJERCICIOS 26-27: SciPy (Estadística, Tests)

### Ejercicio 26: Prueba t-test: Diferencia de Ingresos entre Dos Sucursales

**Enunciado de negocio:** ¿Hay diferencia significativa en el ingreso promedio entre las dos sucursales con mayor y menor ingreso?

**Pista:** `stats.ttest_ind(muestra1, muestra2)`

```python
# Solución
suc_ingresos = ventas.groupby('sucursal')['ingreso'].sum().sort_values(ascending=False)
suc_alta = suc_ingresos.index[0]
suc_baja = suc_ingresos.index[-1]

ingresos_alta = ventas[ventas['sucursal'] == suc_alta]['ingreso']
ingresos_baja = ventas[ventas['sucursal'] == suc_baja]['ingreso']

t_stat, p_val = stats.ttest_ind(ingresos_alta, ingresos_baja, equal_var=False)

print(f"Sucursal alta ({suc_alta}): media=${ingresos_alta.mean():,.0f}")
print(f"Sucursal baja ({suc_baja}): media=${ingresos_baja.mean():,.0f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-valor: {p_val:.6f}")
print("Diferencia significativa (p<0.05):", "Sí" if p_val < 0.05 else "No")
```

**Explicación:** El t-test confirma estadísticamente si la diferencia entre sucursales es real o aleatoria. p<0.05 indica que la brecha es significativa y requiere atención gerencial.

---

### Ejercicio 27: Correlación de Spearman entre Descuento y Margen

**Enunciado de negocio:** ¿Los descuentos altos reducen el margen porcentual? Usa correlación de Spearman (no paramétrica).

**Pista:** `stats.spearmanr(ventas['descuento'], ventas['margen_pct'])`



**Explicación:** El t-test confirma estadísticamente si la diferencia entre sucursales es real o aleatoria. p<0.05 indica que la brecha es significativa y requiere atención gerencial.

---

### Ejercicio 27: Correlación de Spearman entre Descuento y Margen

**Enunciado de negocio:** ¿Los descuentos altos reducen el margen porcentual? Usa correlación de Spearman (no paramétrica).

**Pista:** `stats.spearmanr(ventas['descuento'], ventas['margen_pct'])`

```python
# Solución
rho, p_val = stats.spearmanr(ventas['descuento'], ventas['margen_pct'])

plt.figure(figsize=(10, 4))
sns.scatterplot(data=ventas.sample(300), x='descuento', y='margen_pct',
                alpha=0.5, hue='categoria')
plt.title(f'Correlación de Spearman: Descuento vs Margen (rho={rho:.3f}, p={p_val:.4f})', fontweight='bold')
plt.xlabel('Descuento (%)')
plt.ylabel('Margen (%)')
plt.tight_layout()
plt.show()

print(f"Spearman rho: {rho:.4f}")
print(f"p-valor: {p_val:.6f}")
if p_val < 0.05:
    print(f"Conclusión: Existe correlación {'negativa' if rho < 0 else 'positiva'} significativa.")
    print(f"Los descuentos {'reducen' if rho < 0 else 'aumentan'} el margen.")
else:
    print("Conclusión: No hay evidencia de correlación significativa.")
```

**Explicación:** Spearman mide relaciones monotónicas (no necesariamente lineales). Si rho es negativa y significativa, dar descuentos erosiona la rentabilidad. Útil para fijar política de descuentos máximos.

---

## 📘 EJERCICIOS 28-29: Sklearn (Train/Test, Métricas, Preprocesamiento)

### Ejercicio 28: Regresión Lineal para Predecir Ingreso

**Enunciado de negocio:** Crea un modelo de regresión lineal que prediga el ingreso usando cantidad, precio unitario y descuento como features.

**Pista:** `from sklearn.linear_model import LinearRegression`; usa `train_test_split` y `r2_score`.



**Explicación:** Spearman mide relaciones monotónicas (no necesariamente lineales). Si rho es negativa y significativa, dar descuentos erosiona la rentabilidad. Útil para fijar política de descuentos máximos.

---

## 📘 EJERCICIOS 28-29: Sklearn (Train/Test, Métricas, Preprocesamiento)

### Ejercicio 28: Regresión Lineal para Predecir Ingreso

**Enunciado de negocio:** Crea un modelo de regresión lineal que prediga el ingreso usando cantidad, precio unitario y descuento como features.

**Pista:** `from sklearn.linear_model import LinearRegression`; usa `train_test_split` y `r2_score`.

```python
# Solución
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

features = ['cantidad', 'precio_unitario', 'descuento']
X = ventas[features]
y = ventas['ingreso']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

print("=== REGRESIÓN LINEAL ===")
print(f"R² en test: {r2_score(y_test, y_pred):.4f}")
print(f"MAE: ${mean_absolute_error(y_test, y_pred):,.0f}")
print(f"Coeficientes:")
for feat, coef in zip(features, modelo.coef_):
    print(f"  {feat}: {coef:.4f}")
print(f"  intercept: {modelo.intercept_:.4f}")
```

**Explicación:** La regresión lineal cuantifica el impacto de cada variable en el ingreso. R² indica qué % de la variabilidad explica el modelo. MAE da el error promedio en dólares.

---

### Ejercicio 29: Escalar y Normalizar con StandardScaler

**Enunciado de negocio:** Prepara las features numéricas para un modelo escalándolas con StandardScaler (media=0, std=1). Verifica antes y después.

**Pista:** `from sklearn.preprocessing import StandardScaler`



**Explicación:** La regresión lineal cuantifica el impacto de cada variable en el ingreso. R² indica qué % de la variabilidad explica el modelo. MAE da el error promedio en dólares.

---

### Ejercicio 29: Escalar y Normalizar con StandardScaler

**Enunciado de negocio:** Prepara las features numéricas para un modelo escalándolas con StandardScaler (media=0, std=1). Verifica antes y después.

**Pista:** `from sklearn.preprocessing import StandardScaler`

```python
# Solución
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_escalado = scaler.fit_transform(ventas[features])

print("=== ANTES DEL ESCALADO ===")
print(ventas[features].describe().round(2))

print("\n=== DESPUÉS DEL ESCALADO ===")
df_escalado = pd.DataFrame(X_escalado, columns=features)
print(df_escalado.describe().round(4))

# Verificar media ~0 y std ~1
print(f"\nMedias (deberían ser ~0): {df_escalado.mean().round(6).tolist()}")
print(f"Desviaciones (deberían ser ~1): {df_escalado.std().round(6).tolist()}")
```

**Explicación:** El escalado es necesario para algoritmos sensibles a magnitudes (SVM, KNN, PCA, redes neuronales). Convierte todas las variables a la misma escala sin perder información.

---

## 📘 EJERCICIO 30: Integrador (Pipeline Completo)

### Ejercicio 30: Pipeline Completo de Análisis

**Enunciado de negocio:** Ejecuta un pipeline completo: carga, limpia, analiza, grafica y reporta en un solo flujo. El objetivo es responder: ¿Qué sucursal tiene mejor desempeño general y por qué?

**Pista:** Combina carga, merge, groupby, visualización y modelo simple en una secuencia.



**Explicación:** El escalado es necesario para algoritmos sensibles a magnitudes (SVM, KNN, PCA, redes neuronales). Convierte todas las variables a la misma escala sin perder información.

---

## 📘 EJERCICIO 30: Integrador (Pipeline Completo)

### Ejercicio 30: Pipeline Completo de Análisis

**Enunciado de negocio:** Ejecuta un pipeline completo: carga, limpia, analiza, grafica y reporta en un solo flujo. El objetivo es responder: ¿Qué sucursal tiene mejor desempeño general y por qué?

**Pista:** Combina carga, merge, groupby, visualización y modelo simple en una secuencia.

```python
# Solución completa integradora
print("=" * 60)
print("PIPELINE COMPLETO: ANÁLISIS DE SUCURSALES")
print("=" * 60)

# 1. Cargar y preparar
df = pd.read_csv("../datos/ventas.csv")
df['fecha'] = pd.to_datetime(df['fecha'])
df['mes'] = df['fecha'].dt.month
print(f"\n1. DATOS CARGADOS: {len(df)} transacciones")

# 2. Limpiar (verificar nulos)
nulos = df.isna().sum().sum()
print(f"2. VALORES NULOS: {nulos}")

# 3. Analizar: métricas por sucursal
metricas = df.groupby('sucursal').agg(
    ingresos=('ingreso', 'sum'),
    transacciones=('ingreso', 'count'),
    ticket_prom=('ingreso', 'mean'),
    margen_prom=('margen_pct', 'mean'),
    clientes=('cliente', 'nunique')
).reset_index()

# 4. Calcular puntaje compuesto
for col in ['ingresos', 'transacciones', 'ticket_prom', 'margen_prom', 'clientes']:
    metricas[f'{col}_norm'] = (metricas[col] - metricas[col].min()) / (metricas[col].max() - metricas[col].min())
metricas['score'] = (
    metricas['ingresos_norm'] * 0.30 +
    metricas['transacciones_norm'] * 0.20 +
    metricas['ticket_prom_norm'] * 0.20 +
    metricas['margen_prom_norm'] * 0.20 +
    metricas['clientes_norm'] * 0.10
) * 100
metricas = metricas.sort_values('score', ascending=False)

print("\n3. RANKING DE SUCURSALES:")
print(metricas[['sucursal', 'score', 'ingresos', 'transacciones', 'margen_prom']].to_string(index=False))

# 5. Visualizar
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.barplot(data=metricas, y='sucursal', x='score', palette='RdYlGn', ax=axes[0], hue='sucursal', legend=False)
axes[0].set_title('Score Compuesto', fontweight='bold')
sns.barplot(data=metricas, y='sucursal', x='ingresos', palette='viridis', ax=axes[1], hue='sucursal', legend=False)
axes[1].set_title('Ingresos', fontweight='bold')
sns.barplot(data=metricas, y='sucursal', x='margen_prom', palette='coolwarm', ax=axes[2], hue='sucursal', legend=False)
axes[2].set_title('Margen Promedio (%)', fontweight='bold')
plt.tight_layout()
plt.show()

# 6. Modelo simple: predecir score con una variable
from sklearn.linear_model import LinearRegression
X = metricas[['ingresos']]
y = metricas['score']
model = LinearRegression().fit(X, y)
print(f"\n4. RELACIÓN INGRESOS vs SCORE: R² = {model.score(X, y):.3f}")

# 7. Reporte final
mejor = metricas.iloc[0]
peor = metricas.iloc[-1]
print(f"\n5. REPORTE EJECUTIVO:")
print(f"   Mejor sucursal: {mejor['sucursal']} (score: {mejor['score']:.1f})")
print(f"   - Ingresos: ${mejor['ingresos']:,.0f}")
print(f"   - Transacciones: {mejor['transacciones']}")
print(f"   - Ticket prom: ${mejor['ticket_prom']:,.0f}")
print(f"   - Margen prom: {mejor['margen_prom']:.1f}%")
print(f"   Peor sucursal: {peor['sucursal']} (score: {peor['score']:.1f})")
print(f"   Brecha total: {mejor['score'] - peor['score']:.1f} puntos")
print("=" * 60)
```

**Explicación:** Este pipeline integra todas las habilidades: carga (Pandas), limpieza (nulos), análisis (groupby), modelo (sklearn) y visualización (Seaborn). Es el entregable mínimo que un analista debe producir para responder una pregunta de negocio compleja.

---

## 📝 5 Ejercicios Extra (Desafío)

**Ejercicio Extra 1 — Pronóstico con Prophet:** Usa la serie diaria de ingresos para pronosticar los siguientes 30 días usando la columna `fecha` e `ingreso`. (Pista: instala fbprophet o usa statsmodels con SARIMA)

**Ejercicio Extra 2 — Cluster de Clientes con K-Means:** Usa las variables de RFM (recencia, frecuencia, monto) escaladas para segmentar clientes en 4 clusters con K-Means. ¿Qué perfil tiene cada cluster? (Pista: `from sklearn.cluster import KMeans`)

**Ejercicio Extra 3 — Detección de Anomalías con Isolation Forest:** Identifica transacciones anómalas en ingresos usando Isolation Forest. ¿Qué porcentaje son anomalías? (Pista: `from sklearn.ensemble import IsolationForest`)

**Ejercicio Extra 4 — Análisis de Texto en Nombres de Producto:** Extrae palabras clave de los nombres de producto y crea una nube de palabras. ¿Qué términos son más frecuentes? (Pista: `from wordcloud import WordCloud`)

**Ejercicio Extra 5 — Dashboard Interactivo con Plotly:** Crea un dashboard interactivo con 3 gráficos (ingresos por sucursal, tendencia mensual, top productos) usando Plotly Express. (Pista: `import plotly.express as px`)

---

## 📌 Resumen de Habilidades Cubiertas

| Ejercicios | Habilidad | Librería |
|------------|-----------|----------|
| 1-5 | Tipos, listas, dicts, funciones, map | Python base |
| 6-10 | Arrays, indexing, operaciones vectorizadas, bootstrap | NumPy |
| 11-15 | Groupby, filtros, rank, transform, nunique | Pandas |
| 16-20 | Merge, pivot, fechas, cohortes | Pandas |
| 21-25 | Histplot, boxplot, barplot, scatter, pairplot | Seaborn |
| 26-27 | t-test, Spearman | SciPy |
| 28-29 | Regresión, escalado | Sklearn |
| 30 | Pipeline completo (carga->limpia->analiza->grafica->reporta) | Todo |

## 🔗 Enlaces Relacionados
- [CP01 a CP05 — Casos previos](.)
- [Módulo 00 — Base Python](../00-python/)
- [Módulo 01 — NumPy](../01-numpy/)
- [Módulo 02 — Pandas](../02-pandas/)
- [Módulo 03 — Seaborn](../03-seaborn/)
- [Módulo 04 — SciPy](../04-scipy/)
- [Módulo 05 — Sklearn](../05-sklearn/)


**Explicación:** Este pipeline integra todas las habilidades: carga (Pandas), limpieza (nulos), análisis (groupby), modelo (sklearn) y visualización (Seaborn). Es el entregable mínimo que un analista debe producir para responder una pregunta de negocio compleja.

---

## 📝 5 Ejercicios Extra (Desafío)

**Ejercicio Extra 1 — Pronóstico con Prophet:** Usa la serie diaria de ingresos para pronosticar los siguientes 30 días usando la columna `fecha` e `ingreso`. (Pista: instala fbprophet o usa statsmodels con SARIMA)

**Ejercicio Extra 2 — Cluster de Clientes con K-Means:** Usa las variables de RFM (recencia, frecuencia, monto) escaladas para segmentar clientes en 4 clusters con K-Means. ¿Qué perfil tiene cada cluster? (Pista: `from sklearn.cluster import KMeans`)

**Ejercicio Extra 3 — Detección de Anomalías con Isolation Forest:** Identifica transacciones anómalas en ingresos usando Isolation Forest. ¿Qué porcentaje son anomalías? (Pista: `from sklearn.ensemble import IsolationForest`)

**Ejercicio Extra 4 — Análisis de Texto en Nombres de Producto:** Extrae palabras clave de los nombres de producto y crea una nube de palabras. ¿Qué términos son más frecuentes? (Pista: `from wordcloud import WordCloud`)

**Ejercicio Extra 5 — Dashboard Interactivo con Plotly:** Crea un dashboard interactivo con 3 gráficos (ingresos por sucursal, tendencia mensual, top productos) usando Plotly Express. (Pista: `import plotly.express as px`)

---

## 📌 Resumen de Habilidades Cubiertas

| Ejercicios | Habilidad | Librería |
|------------|-----------|----------|
| 1-5 | Tipos, listas, dicts, funciones, map | Python base |
| 6-10 | Arrays, indexing, operaciones vectorizadas, bootstrap | NumPy |
| 11-15 | Groupby, filtros, rank, transform, nunique | Pandas |
| 16-20 | Merge, pivot, fechas, cohortes | Pandas |
| 21-25 | Histplot, boxplot, barplot, scatter, pairplot | Seaborn |
| 26-27 | t-test, Spearman | SciPy |
| 28-29 | Regresión, escalado | Sklearn |
| 30 | Pipeline completo (carga->limpia->analiza->grafica->reporta) | Todo |

## 🔗 Enlaces Relacionados
- [CP01 a CP05 — Casos previos](.)
- [Módulo 00 — Base Python](../00-python/)
- [Módulo 01 — NumPy](../01-numpy/)
- [Módulo 02 — Pandas](../02-pandas/)
- [Módulo 03 — Seaborn](../03-seaborn/)
- [Módulo 04 — SciPy](../04-scipy/)
- [Módulo 05 — Sklearn](../05-sklearn/)
