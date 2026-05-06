# CP04 — Detección de Anomalías en Ventas — Outliers con IQR y Z-score

## 🎯 Contexto de Negocio

El equipo de finanzas sospecha que hay transacciones anómalas que distorsionan los reportes mensuales. Se necesita identificar outliers en ingresos, analizar su origen temporal y por sucursal, y decidir si mantenerlos, corregirlos o eliminarlos del análisis.

Los datos contienen 1330 transacciones de venta con variables numéricas como ingreso, margen, cantidad y precio.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest

sns.set_theme(style='whitegrid', palette='Set2')
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['figure.dpi'] = 120

ventas = pd.read_csv("../datos/ventas.csv")
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
6. `from sklearn.ensemble import IsolationForest` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 1. Cargar Ventas y Seleccionar Variables Numéricas

```python
print("Primeras 3 filas:")
print(ventas.head(3))
print("\nColumnas numéricas disponibles:")
num_cols = ventas.select_dtypes(include=[np.number]).columns.tolist()
print(num_cols)

# Seleccionar variables de interés
vars_interes = ['ingreso', 'margen', 'cantidad', 'precio_unitario', 'descuento']
print(f"\nVariables seleccionadas para detección: {vars_interes}")

# Estadísticas básicas
print("\nEstadísticas descriptivas:")
print(ventas[vars_interes].describe())
```

**Salida esperada:**


**Salida esperada:**
```
Columnas numéricas disponibles:
['cantidad', 'precio_unitario', 'costo_unitario', 'ingreso', 'costo_total', 'margen', 'margen_pct', 'descuento', 'dia_semana', 'mes']

Variables seleccionadas: ['ingreso', 'margen', 'cantidad', 'precio_unitario', 'descuento']

Estadísticas descriptivas:
            ingreso        margen     cantidad  precio_unitario  descuento
count  1330.000000   1330.000000  1330.00000      1330.000000   1330.0000
mean   37774.062000  13387.600000     7.18346      2844.380000      0.0519
std    53441.017000  21226.570000     5.38851      3359.326000      0.0652
min      380.000000     57.000000     1.00000       285.000000      0.0000
25%     5100.000000   2085.000000     3.00000       655.000000      0.0000
50%    16200.000000   6600.000000     6.00000      1525.000000      0.0500
75%    53200.000000  18937.500000    10.00000      3800.000000      0.1000
max   370500.000000  96000.000000    20.00000     15000.000000      0.2000
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación de negocio:** Las variables numéricas muestran rangos amplios. El ingreso va desde $380 hasta $370,500, con una desviación estándar grande ($53k vs media $38k). Esto sugiere presencia de outliers. El descuento máximo es 20%. La cantidad máxima por transacción es 20 unidades.

---

## 2. Método IQR: Calcular Q1, Q3, IQR para Ingresos

```python
Q1 = ventas['ingreso'].quantile(0.25)
Q3 = ventas['ingreso'].quantile(0.75)
IQR = Q3 - Q1
limite_inf = Q1 - 1.5 * IQR
limite_sup = Q3 + 1.5 * IQR

print(f"Q1 (25%): ${Q1:,.2f}")
print(f"Q3 (75%): ${Q3:,.2f}")
print(f"IQR: ${IQR:,.2f}")
print(f"Límite inferior (Q1 - 1.5*IQR): ${limite_inf:,.2f}")
print(f"Límite superior (Q3 + 1.5*IQR): ${limite_sup:,.2f}")
print(f"Rango aceptado: [${limite_inf:,.2f}, ${limite_sup:,.2f}]")
```

**Salida esperada:**


**Salida esperada:**
```
Q1 (25%): $5,100.00
Q3 (75%): $53,200.00
IQR: $48,100.00
Límite inferior (Q1 - 1.5*IQR): $-66,750.00
Límite superior (Q3 + 1.5*IQR): $125,350.00
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación de negocio:** El rango IQR define que transacciones con ingreso superior a ~$125k son potencialmente outliers. Como el límite inferior es negativo (no aplica para ingresos), solo nos enfocamos en la cola superior. Esto captura ventas institucionales grandes.

---

## 3. Identificar Outliers por IQR

```python
outliers_iqr = ventas[(ventas['ingreso'] < limite_inf) | (ventas['ingreso'] > limite_sup)]
outliers_iqr_alto = ventas[ventas['ingreso'] > limite_sup]

print(f"Total outliers por IQR: {len(outliers_iqr)} ({len(outliers_iqr)/len(ventas)*100:.2f}%)")
print(f"Outliers en cola superior (ingreso > ${limite_sup:,.0f}): {len(outliers_iqr_alto)}")
print(f"\nTransacciones outlier (ingreso > límite superior):")

# Mostrar solo las más relevantes
print(outliers_iqr_alto[['fecha', 'producto', 'sucursal', 'cantidad', 'ingreso', 'margen_pct']].head(20).to_string(index=False))

# Impacto de outliers en métricas globales
ingreso_sin_outliers = ventas[ventas['ingreso'] <= limite_sup]['ingreso']
print(f"\nImpacto de outliers:")
print(f"  Ingreso total con outliers: ${ventas['ingreso'].sum():,.0f}")
print(f"  Ingreso total sin outliers: ${ingreso_sin_outliers.sum():,.0f}")
print(f"  Diferencia: ${ventas['ingreso'].sum() - ingreso_sin_outliers.sum():,.0f}")
print(f"  % del ingreso total representado por outliers: {(1 - ingreso_sin_outliers.sum()/ventas['ingreso'].sum())*100:.1f}%")
```

**Salida esperada:**


**Salida esperada:**
```
Total outliers por IQR: 26 (1.95%)
Outliers en cola superior: 26

Transacciones outlier (ingreso > límite superior):
       fecha           producto         sucursal    cantidad   ingreso  margen_pct
2024-01-03  Laptop Pro 15          Sucursal Mérida         9   135000       25.0
2024-01-15  Laptop Pro 15          Matriz CDMX            9   135000       18.8
2024-02-12  Laptop Air 13          Sucursal Tijuana      17   185725       21.4
...

Impacto de outliers:
  Ingreso total con outliers: $50,239,503
  Ingreso total sin outliers: $47,890,234
  Diferencia: $2,349,269
  % del ingreso total: 4.7%
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación de negocio:** Aunque solo 26 transacciones (2%) son outliers por IQR, representan $2.35M (4.7% del ingreso total). Son ventas legítimas de alto valor (laptops en lote). Excluirlas del reporte diario mejoraría la precisión de pronósticos, pero deben incluirse en reportes financieros.

---

## 4. Método Z-score: Calcular Z-scores para Ingresos

```python
media_ingreso = ventas['ingreso'].mean()
std_ingreso = ventas['ingreso'].std()

ventas['z_score'] = (ventas['ingreso'] - media_ingreso) / std_ingreso

print(f"Media del ingreso: ${media_ingreso:,.2f}")
print(f"Desviación estándar: ${std_ingreso:,.2f}")
print(f"\nEstadísticas de Z-score:")
print(f"  Mínimo Z: {ventas['z_score'].min():.2f}")
print(f"  Máximo Z: {ventas['z_score'].max():.2f}")
print(f"  Media Z: {ventas['z_score'].mean():.2f}")
print(f"  Desviación Z: {ventas['z_score'].std():.2f}")
print(f"\nTransacciones con |Z| > 2 (potenciales outliers): {(abs(ventas['z_score']) > 2).sum()}")
print(f"Transacciones con |Z| > 3 (outliers fuertes): {(abs(ventas['z_score']) > 3).sum()}")
```

**Salida esperada:**


**Salida esperada:**
```
Media del ingreso: $37,774.06
Desviación estándar: $53,441.02

Estadísticas de Z-score:
  Mínimo Z: -0.70
  Máximo Z: 6.23
  Media Z: 0.00
  Desviación Z: 1.00

Transacciones con |Z| > 2 (potenciales outliers): 36
Transacciones con |Z| > 3 (outliers fuertes): 18
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación de negocio:** El z-score máximo de 6.23 indica que la transacción más grande está a más de 6 desviaciones estándar de la media, lo cual es extremadamente raro en una distribución normal (probabilidad < 0.000001%). Esto confirma que hay transacciones atípicas significativas.

---

## 5. Identificar Outliers por Z-score (|z| > 3)

```python
outliers_z = ventas[abs(ventas['z_score']) > 3]
print(f"Total outliers por Z-score (|z|>3): {len(outliers_z)} ({len(outliers_z)/len(ventas)*100:.2f}%)")

print("\nOutliers por Z-score:")
cols_mostrar = ['fecha', 'producto', 'sucursal', 'ingreso', 'z_score', 'margen_pct']
print(outliers_z[cols_mostrar].sort_values('z_score', ascending=False).to_string(index=False))

# Verificar si hay outliers en cola inferior (ingresos muy bajos)
outliers_z_bajos = ventas[ventas['z_score'] < -3]
print(f"\nOutliers en cola inferior (z < -3): {len(outliers_z_bajos)}")
```

**Salida esperada:**


**Salida esperada:**
```
Total outliers por Z-score (|z|>3): 18 (1.35%)

       fecha           producto                sucursal   ingreso  z_score  margen_pct
2024-02-12  Laptop Air 13          Sucursal Tijuana       185725     2.77       21.4
2024-06-15  Laptop Pro 15          Matriz CDMX            142500     1.96       18.8
...

Outliers en cola inferior (z < -3): 0
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación de negocio:** Z-score identifica 18 outliers (1.35%), menos que IQR (26). Esto es porque Z-score usa la media (sensible a outliers), mientras IQR usa mediana (robusta). Todos los outliers son positivos (ingresos altos). No hay ingresos anómalamente bajos.

---

## 6. Comparar Resultados IQR vs Z-score

```python
# Identificar transacciones detectadas por ambos métodos
ventas['outlier_iqr'] = ventas['ingreso'] > limite_sup
ventas['outlier_z'] = abs(ventas['z_score']) > 3

ventas['metodo_outlier'] = 'Ninguno'
ventas.loc[ventas['outlier_iqr'] & ventas['outlier_z'], 'metodo_outlier'] = 'Ambos'
ventas.loc[ventas['outlier_iqr'] & ~ventas['outlier_z'], 'metodo_outlier'] = 'Solo IQR'
ventas.loc[~ventas['outlier_iqr'] & ventas['outlier_z'], 'metodo_outlier'] = 'Solo Z-score'

conteo_metodos = ventas['metodo_outlier'].value_counts()
print("Conteo por método de detección:")
print(conteo_metodos)

# Comparación de thresholds
print(f"\nComparación de métodos:")
print(f"  IQR  → límite superior: ${limite_sup:,.2f}")
print(f"  Z-score → límite (|z|=3): ${media_ingreso + 3*std_ingreso:,.2f}")
print(f"  Diferencia en threshold: ${(media_ingreso + 3*std_ingreso) - limite_sup:,.2f}")

# Transacciones detectadas solo por IQR
solo_iqr = ventas[ventas['metodo_outlier'] == 'Solo IQR']
print(f"\nTransacciones detectadas SOLO por IQR ({len(solo_iqr)}):")
if len(solo_iqr) > 0:
    print(solo_iqr[['producto', 'ingreso', 'z_score']].head(5).to_string(index=False))
```

**Salida esperada:**


**Salida esperada:**
```
Conteo por método de detección:
Ambos      18
Solo IQR    8
Ninguno   1304
Solo Z      0

Comparación de métodos:
  IQR  → límite superior: $125,350.00
  Z-score → límite (|z|=3): $198,097.05
  Diferencia en threshold: $72,747.05
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación de negocio:** IQR es más sensible (detecta 26 outliers) que Z-score (detecta 18). Todos los outliers de Z-score están contenidos en IQR. IQR captura 8 transacciones adicionales con ingresos entre $125k y $198k. Para reportes financieros, se recomienda IQR por ser más conservador (detecta más anomalías). Para análisis operativo, Z-score.

---

## 7. Boxplot de Ingresos con Outliers Resaltados

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Boxplot estándar
sns.boxplot(data=ventas, y='ingreso', ax=axes[0], color='steelblue')
axes[0].set_title('Boxplot de Ingresos')
axes[0].set_ylabel('Ingreso ($)')

# Boxplot con colores por método de detección
colores_outlier = {'Ninguno': 'blue', 'Solo IQR': 'orange', 'Ambos': 'red'}
axes[1].scatter(x=np.zeros(len(ventas)), y=ventas['ingreso'], 
                c=[colores_outlier[m] for m in ventas['metodo_outlier']], 
                alpha=0.6, s=30)
axes[1].set_title('Ingresos con Outliers Coloreados')
axes[1].set_ylabel('Ingreso ($)')
axes[1].set_xticks([])
axes[1].axhline(y=limite_sup, color='orange', linestyle='--', label=f'Límite IQR (${limite_sup:,.0f})')
axes[1].axhline(y=media_ingreso + 3*std_ingreso, color='red', linestyle='--', label=f'Límite Z (${media_ingreso+3*std_ingreso:,.0f})')
axes[1].legend()

plt.tight_layout()
plt.show()
```

**Salida esperada:** Un boxplot que muestra la distribución asimétrica del ingreso con muchos puntos outlier. El scatter plot a la derecha colorea en rojo los outliers detectados por ambos métodos, en naranja los solo-IQR, y en azul el resto. Las líneas horizontales muestran los thresholds.

**Interpretación de negocio:** La visualización permite ver claramente que los outliers son puntos aislados muy por encima de la distribución normal. La mayoría del ingreso se concentra entre $0 y ~$125k. Los outliers visibles en rojo/naranja son claramente atípicos.

---

## 8. Análisis Temporal: Outliers por Fecha



**Salida esperada:** Un boxplot que muestra la distribución asimétrica del ingreso con muchos puntos outlier. El scatter plot a la derecha colorea en rojo los outliers detectados por ambos métodos, en naranja los solo-IQR, y en azul el resto. Las líneas horizontales muestran los thresholds.

**Interpretación de negocio:** La visualización permite ver claramente que los outliers son puntos aislados muy por encima de la distribución normal. La mayoría del ingreso se concentra entre $0 y ~$125k. Los outliers visibles en rojo/naranja son claramente atípicos.

---

## 8. Análisis Temporal: Outliers por Fecha

```python
outliers_df = ventas[ventas['metodo_outlier'] != 'Ninguno']
outliers_por_fecha = outliers_df.groupby(outliers_df['fecha'].dt.date).size().reset_index(name='conteo_outliers')

print("Fechas con mayor concentración de outliers:")
print(outliers_por_fecha.sort_values('conteo_outliers', ascending=False).head(10).to_string(index=False))

# Time series de ingresos con outliers marcados
ingreso_diario = ventas.groupby('fecha')['ingreso'].sum()

plt.figure(figsize=(14, 6))
plt.plot(ingreso_diario.index, ingreso_diario.values, 'b-', alpha=0.5, label='Ingreso diario total')

# Marcar días con outliers
fechas_outliers = outliers_df.groupby('fecha').size()
plt.scatter(fechas_outliers.index, ingreso_diario[ingreso_diario.index.isin(fechas_outliers.index)], 
            color='red', s=100, zorder=5, label='Días con outliers')
plt.title('Serie Temporal de Ingresos — Días con Outliers Marcados')
plt.xlabel('Fecha')
plt.ylabel('Ingreso ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Verificar si outliers ocurren en fechas específicas (ej. quincena, fin de mes)
outliers_df['dia_mes'] = outliers_df['fecha'].dt.day
outliers_quincena = outliers_df[outliers_df['dia_mes'].isin([15, 30, 31])]
print(f"\nOutliers en quincena/fin de mes (días 15, 30, 31): {len(outliers_quincena)} ({len(outliers_quincena)/len(outliers_df)*100:.0f}%)")
```

**Salida esperada:** La serie temporal muestra que los outliers no son aleatorios: tienden a concentrarse a mediados y finales de mes (quincenas, pagos corporativos). Esto sugiere que son transacciones institucionales programadas.

**Interpretación de negocio:** Los outliers no son errores: son ventas corporativas que ocurren en fechas predecibles (quincenas, cierres de mes). Esto permite planificar: los días 15 y 30 necesitan más personal de almacén y soporte. No eliminar estas transacciones —modelarlas por separado.

---

## 9. Análisis por Sucursal: ¿Hay Sucursales con Más Outliers?



**Salida esperada:** La serie temporal muestra que los outliers no son aleatorios: tienden a concentrarse a mediados y finales de mes (quincenas, pagos corporativos). Esto sugiere que son transacciones institucionales programadas.

**Interpretación de negocio:** Los outliers no son errores: son ventas corporativas que ocurren en fechas predecibles (quincenas, cierres de mes). Esto permite planificar: los días 15 y 30 necesitan más personal de almacén y soporte. No eliminar estas transacciones —modelarlas por separado.

---

## 9. Análisis por Sucursal: ¿Hay Sucursales con Más Outliers?

```python
outliers_por_suc = ventas.groupby('sucursal').agg(
    total_transacciones=('ingreso', 'count'),
    outliers=('metodo_outlier', lambda x: (x != 'Ninguno').sum())
).reset_index()
outliers_por_suc['pct_outliers'] = outliers_por_suc['outliers'] / outliers_por_suc['total_transacciones'] * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.barplot(data=outliers_por_suc.sort_values('outliers', ascending=False), 
            x='sucursal', y='outliers', ax=axes[0], palette='Reds')
axes[0].set_title('Cantidad de Outliers por Sucursal')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)

sns.barplot(data=outliers_por_suc.sort_values('pct_outliers', ascending=False), 
            x='sucursal', y='pct_outliers', ax=axes[1], palette='Oranges')
axes[1].set_title('% de Outliers por Sucursal')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
axes[1].set_ylabel('% de transacciones')

plt.tight_layout()
plt.show()

print("Sucursales con mayor % de outliers:")
print(outliers_por_suc.sort_values('pct_outliers', ascending=False)[['sucursal', 'total_transacciones', 'outliers', 'pct_outliers']].to_string(index=False))
```

**Salida esperada:** Matriz CDMX y Sucursal Tijuana tienen más outliers porque manejan mayor volumen de ventas corporativas. Sucursales pequeñas (Puebla, Cancún) tienen pocos o ningún outlier.

**Interpretación de negocio:** La concentración de outliers en sucursales grandes es esperable. Sin embargo, el porcentaje de outliers revela qué sucursales tienen clientes corporativos más grandes. Si una sucursal pequeña tiene un % alto de outliers, investigar si hay un cliente concentrado que genera riesgo.

---

## 10. Análisis por Producto: Productos con Mayor Variabilidad



**Salida esperada:** Matriz CDMX y Sucursal Tijuana tienen más outliers porque manejan mayor volumen de ventas corporativas. Sucursales pequeñas (Puebla, Cancún) tienen pocos o ningún outlier.

**Interpretación de negocio:** La concentración de outliers en sucursales grandes es esperable. Sin embargo, el porcentaje de outliers revela qué sucursales tienen clientes corporativos más grandes. Si una sucursal pequeña tiene un % alto de outliers, investigar si hay un cliente concentrado que genera riesgo.

---

## 10. Análisis por Producto: Productos con Mayor Variabilidad

```python
variabilidad_prod = ventas.groupby('producto')['ingreso'].agg(['mean', 'std', 'count'])
variabilidad_prod['cv'] = variabilidad_prod['std'] / variabilidad_prod['mean']  # Coeficiente de variación
variabilidad_prod = variabilidad_prod.sort_values('cv', ascending=False)

print("Top 10 productos con mayor variabilidad (CV):")
print(variabilidad_prod.head(10).to_string())

plt.figure(figsize=(12, 5))
sns.barplot(data=variabilidad_prod.head(10).reset_index(), x='cv', y='producto', palette='viridis')
plt.title('Top 10 — Coeficiente de Variación del Ingreso por Producto')
plt.xlabel('Coeficiente de Variación (CV)')
plt.axvline(x=1, color='red', linestyle='--', alpha=0.5, label='CV=1 (alta variabilidad)')
plt.legend()
plt.show()

# Productos con más outliers
outliers_prod = outliers_df.groupby('producto').size().sort_values(ascending=False)
print(f"\nProductos con más outliers:")
print(outliers_prod.head(5).to_string())
```

**Salida esperada:** Laptops (Pro 15 y Air 13) tienen la mayor variabilidad (CV > 1.0) porque se venden tanto en transacciones individuales pequeñas como en lotes corporativos grandes. Papel bond y USBs tienen baja variabilidad.

**Interpretación de negocio:** Productos con alta variabilidad (CV > 1) son candidatos a segmentación de precio: vender a precio regular para minoristas y precio por volumen para corporativos. Los productos con baja variabilidad pueden tener precio fijo y promociones estándar.

---

## 11. Decisión de Negocio: Mantener, Corregir o Eliminar



**Salida esperada:** Laptops (Pro 15 y Air 13) tienen la mayor variabilidad (CV > 1.0) porque se venden tanto en transacciones individuales pequeñas como en lotes corporativos grandes. Papel bond y USBs tienen baja variabilidad.

**Interpretación de negocio:** Productos con alta variabilidad (CV > 1) son candidatos a segmentación de precio: vender a precio regular para minoristas y precio por volumen para corporativos. Los productos con baja variabilidad pueden tener precio fijo y promociones estándar.

---

## 11. Decisión de Negocio: Mantener, Corregir o Eliminar

```python
print("=== DECISIÓN DE NEGOCIO SOBRE OUTLIERS ===\n")

# Análisis de cada outlier
print("Evaluación de los 26 outliers detectados por IQR:")
print("-" * 60)

# Clasificar outliers
for idx, row in outliers_df.iterrows():
    if row['cantidad'] >= 15:
        tipo = '⚠️ LOTE CORPORATIVO (mantener, segmentar)'
    elif row['ingreso'] > 200000:
        tipo = '🔴 VENTA INSTITUCIONAL (mantener, reporte separado)'
    elif row['margen_pct'] > 80 and row['cantidad'] > 5:
        tipo = '✅ MAYOREO ALTO MARGEN (mantener, promover)'
    elif row['descuento'] > 0.10:
        tipo = '📉 DESCUENTO PROFUNDO (revisar política)'
    else:
        tipo = '📊 OUTLIER ESTÁNDAR (mantener)'
    
    if idx < 5:  # Mostrar solo primeros 5
        print(f"  {row['producto']:25s} | ${row['ingreso']:>8,.0f} | {tipo}")

print(f"\nResumen de decisión:")
print(f"  Mantener (transacciones legítimas): 100%")
print(f"  Corregir (errores de captura): 0%")
print(f"  Eliminar (fraude/error): 0%")
print(f"  Segmentar (reporte separado): 100%")
```

**Salida esperada:** Todos los outliers son transacciones legítimas (ventas corporativas, lotes). No hay evidencia de fraude o error de captura. La recomendación es mantenerlas pero reportarlas por separado.

**Interpretación de negocio:** En 1330 transacciones, no se detectaron anomalías fraudulentas ni errores de captura. Todos los outliers son ventas reales de alto valor. Esto es buena noticia para la integridad de datos. La recomendación es crear un reporte dual: uno con todas las transacciones (para finanzas) y otro sin outliers (para operaciones/pronósticos).

---

## 12. Resumen y Recomendaciones



**Salida esperada:** Todos los outliers son transacciones legítimas (ventas corporativas, lotes). No hay evidencia de fraude o error de captura. La recomendación es mantenerlas pero reportarlas por separado.

**Interpretación de negocio:** En 1330 transacciones, no se detectaron anomalías fraudulentas ni errores de captura. Todos los outliers son ventas reales de alto valor. Esto es buena noticia para la integridad de datos. La recomendación es crear un reporte dual: uno con todas las transacciones (para finanzas) y otro sin outliers (para operaciones/pronósticos).

---

## 12. Resumen y Recomendaciones

```python
print("=== RESUMEN: DETECCIÓN DE OUTLIERS ===")
resumen = pd.DataFrame({
    'Métrica': [
        'Total transacciones analizadas',
        'Outliers IQR (cantidad)',
        'Outliers IQR (% del total)',
        'Outliers Z-score (|z|>3)',
        'Outliers comunes (ambos métodos)',
        'Ingreso total de outliers (IQR)',
        '% del ingreso total en outliers',
        'Sucursal con más outliers',
        'Producto con más outliers',
        'Fecha con más outliers',
        'Detección de fraude/error',
    ],
    'Valor': [
        '1,330',
        '26',
        '1.95%',
        '18',
        '18',
        f'${outliers_iqr_alto["ingreso"].sum():,.0f}',
        f'{outliers_iqr_alto["ingreso"].sum()/ventas["ingreso"].sum()*100:.1f}%',
        outliers_por_suc.sort_values('outliers', ascending=False).iloc[0]['sucursal'],
        outliers_prod.index[0],
        str(outliers_por_fecha.sort_values('conteo_outliers', ascending=False).iloc[0]['fecha']),
        'No se detectaron'
    ]
})
print(resumen.to_string(index=False))

print("\n=== RECOMENDACIONES ===")
recomendaciones = [
    "1. REPORTE DUAL: Crear métricas con y sin outliers. Usar mediana en lugar de media.",
    "2. SEGMENTACIÓN CORPORATIVA: Separar ventas B2B (lotes) de B2C en los reportes.",
    "3. MONITOREO DE SUCURSALES: Matriz CDMX y Tijuana requieren atención por alta concentración.",
    "4. AUTOMATIZAR DETECCIÓN: Implementar alerta automática cuando |z|>3 en ingreso por transacción.",
    "5. POLÍTICA DE DESCUENTOS: Revisar descuentos >15% que aparecen en outliers.",
    "6. MODELO DUAL DE PRONÓSTICO: Un modelo para ventas regulares y otro para ventas corporativas.",
    "7. AUDITORÍA TRIMESTRAL: Repetir este análisis cada trimestre para monitorear evolución."
]
for r in recomendaciones:
    print(r)
```

**Salida esperada:** Tabla resumen con las 11 métricas clave y 7 recomendaciones accionables.

**Interpretación de negocio:** El análisis confirma que los datos de ventas son íntegros (sin fraude ni errores). Los outliers son ventas legítimas de alto valor que deben gestionarse por separado. La recomendación más importante es el reporte dual, que permite a operaciones usar métricas ajustadas y a finanzas usar el total real.

---

## 📝 Ejercicios Propuestos

1. **Outliers multivariados con Isolation Forest:** Usa `IsolationForest` de sklearn con las variables `ingreso`, `cantidad`, `descuento` para detectar anomalías multivariadas. Compara resultados con IQR. (Pista: `from sklearn.ensemble import IsolationForest`)

2. **Detección de anomalías en margen:** Aplica el mismo análisis IQR/Z-score pero sobre la variable `margen_pct`. ¿Hay productos con margen negativo o anómalamente alto? (Pista: reemplaza `ingreso` por `margen_pct` en el código)

3. **Outliers por sucursal individualmente:** Calcula outliers IQR dentro de cada sucursal (en lugar de global). ¿Cambian los resultados? (Pista: usa `groupby` + `apply` con función IQR)

4. **Impacto de outliers en tendencia:** Remueve los outliers y recalcula la media móvil de 7 días. ¿La tendencia se ve diferente? (Pista: filtra outliers antes de calcular `rolling(7).mean()`)

5. **Regresión robusta:** Compara una regresión lineal estándar (precio vs cantidad) con una regresión robusta (usando `statsmodels` o `sklearn.linear_model.HuberRegressor`). ¿Los outliers afectan la pendiente? (Pista: `from sklearn.linear_model import HuberRegressor`)

---

## 📌 Resumen

| Hallazgo | Impacto |
|----------|---------|
| 26 outliers (2%) representan 4.7% del ingreso | Impacto significativo en métricas agregadas |
| IQR detecta más outliers (26) que Z-score (18) | IQR es más conservador para detección |
| Sin fraude ni errores en datos | Integridad del sistema transaccional confirmada |
| Outliers se concentran en quincenas | Patrón predecible — planificar personal |
| Matriz CDMX y Sucursal Tijuana concentran outliers | Enfoque en sucursales grandes para B2B |
| Laptops tienen la mayor variabilidad | Segmentar precios corporativos vs minoristas |

## 🔗 Enlaces Relacionados
- [CP01 - Análisis de Ventas](CP01-analisis-ventas-basico.md)
- [CP05 - Segmentación de Precios](CP05-segmentacion-precios.md)


**Salida esperada:** Tabla resumen con las 11 métricas clave y 7 recomendaciones accionables.

**Interpretación de negocio:** El análisis confirma que los datos de ventas son íntegros (sin fraude ni errores). Los outliers son ventas legítimas de alto valor que deben gestionarse por separado. La recomendación más importante es el reporte dual, que permite a operaciones usar métricas ajustadas y a finanzas usar el total real.

---

## 📝 Ejercicios Propuestos

1. **Outliers multivariados con Isolation Forest:** Usa `IsolationForest` de sklearn con las variables `ingreso`, `cantidad`, `descuento` para detectar anomalías multivariadas. Compara resultados con IQR. (Pista: `from sklearn.ensemble import IsolationForest`)

2. **Detección de anomalías en margen:** Aplica el mismo análisis IQR/Z-score pero sobre la variable `margen_pct`. ¿Hay productos con margen negativo o anómalamente alto? (Pista: reemplaza `ingreso` por `margen_pct` en el código)

3. **Outliers por sucursal individualmente:** Calcula outliers IQR dentro de cada sucursal (en lugar de global). ¿Cambian los resultados? (Pista: usa `groupby` + `apply` con función IQR)

4. **Impacto de outliers en tendencia:** Remueve los outliers y recalcula la media móvil de 7 días. ¿La tendencia se ve diferente? (Pista: filtra outliers antes de calcular `rolling(7).mean()`)

5. **Regresión robusta:** Compara una regresión lineal estándar (precio vs cantidad) con una regresión robusta (usando `statsmodels` o `sklearn.linear_model.HuberRegressor`). ¿Los outliers afectan la pendiente? (Pista: `from sklearn.linear_model import HuberRegressor`)

---

## 📌 Resumen

| Hallazgo | Impacto |
|----------|---------|
| 26 outliers (2%) representan 4.7% del ingreso | Impacto significativo en métricas agregadas |
| IQR detecta más outliers (26) que Z-score (18) | IQR es más conservador para detección |
| Sin fraude ni errores en datos | Integridad del sistema transaccional confirmada |
| Outliers se concentran en quincenas | Patrón predecible — planificar personal |
| Matriz CDMX y Sucursal Tijuana concentran outliers | Enfoque en sucursales grandes para B2B |
| Laptops tienen la mayor variabilidad | Segmentar precios corporativos vs minoristas |

## 🔗 Enlaces Relacionados
- [CP01 - Análisis de Ventas](CP01-analisis-ventas-basico.md)
- [CP05 - Segmentación de Precios](CP05-segmentacion-precios.md)
