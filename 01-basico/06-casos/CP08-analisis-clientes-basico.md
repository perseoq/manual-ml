# CP08 — Perfil de Clientes — Frecuencia, Ticket Promedio y Segmentación

## 🎯 Contexto de Negocio

El equipo de ventas quiere entender quiénes son los mejores clientes, cómo se comportan y cómo segmentarlos para diseñar estrategias de fidelización, reactivación y venta cruzada. Conocer el perfil del cliente permite personalizar ofertas y optimizar el gasto en marketing.

Los datos incluyen información de clientes y sus transacciones durante 2024.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style='whitegrid', palette='Set2')
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['figure.dpi'] = 120

clientes = pd.read_csv("../datos/clientes.csv")
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
6. `import warnings` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 1. Cargar y Explorar Clientes

```python
print("Shape clientes:", clientes.shape)
print("\nColumnas:", list(clientes.columns))
print("\nPrimeras 5 filas:")
print(clientes.head())
print("\nInfo:")
print(clientes.info())
print("\nDescribe:")
print(clientes.describe())

print("\nValores únicos por columna:")
for col in clientes.columns:
    print(f"  {col}: {clientes[col].nunique()} únicos")
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

*1. Cargar y Explorar Clientes.*

1. `print("Shape clientes:", clientes.shape)` — Muestra el resultado por pantalla.
2. `print("\nColumnas:", list(clientes.columns))` — Muestra el resultado por pantalla.
3. `print("\nPrimeras 5 filas:")` — Muestra el resultado por pantalla.
4. `print(clientes.head())` — Muestra el resultado por pantalla.
5. `print("\nInfo:")` — Muestra el resultado por pantalla.
6. `print(clientes.info())` — Muestra el resultado por pantalla.
7. `print("\nDescribe:")` — Muestra el resultado por pantalla.
8. `print(clientes.describe())` — Muestra el resultado por pantalla.
9. `print("\nValores únicos por columna:")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación de negocio:** El dataset de clientes contiene información demográfica y de segmento. Conocer la cantidad de clientes, su distribución por tipo (persona física, empresa, gobierno) y su ubicación permite diseñar estrategias comerciales segmentadas. Los clientes empresariales suelen tener tickets más altos pero menor frecuencia.

---

## 2. Estadísticas Descriptivas de Clientes (RFM)

```python
# Calcular Recencia (días desde última compra), Frecuencia, Monto
fecha_maxima = ventas['fecha'].max()

rfm = ventas.groupby('cliente').agg(
    recencia=('fecha', lambda x: (fecha_maxima - x.max()).days),
    frecuencia=('fecha', 'count'),
    monto_total=('ingreso', 'sum'),
    ticket_promedio=('ingreso', 'mean'),
    productos_distintos=('sku', 'nunique')
).reset_index()

print("=== ESTADÍSTICAS RFM ===")
print(rfm[['recencia', 'frecuencia', 'monto_total', 'ticket_promedio', 'productos_distintos']].describe())

# Merge con datos de clientes
clientes_rfm = clientes.merge(rfm, left_on='cliente', right_on='cliente', how='right')

print(f"\nClientes con transacciones: {len(clientes_rfm)}")
print(f"Clientes en tabla clientes: {len(clientes)}")
print(f"Clientes sin match: {clientes_rfm['tipo_cliente'].isna().sum()}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*2. Estadísticas Descriptivas de Clientes (RFM).*

1. Calcular Recencia (días desde última compra), Frecuencia, Monto
2. Merge con datos de clientes

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación de negocio:** El análisis RFM (Recencia, Frecuencia, Monto) es el estándar de la industria para segmentación de clientes. La recencia indica hace cuánto compraron (bajo = mejor). La frecuencia muestra lealtad. El monto refleja valor. Juntos permiten identificar clientes VIP (frecuencia alta + monto alto), clientes en riesgo (recencia alta) y clientes nuevos (frecuencia baja + recencia baja).

---

## 3. Distribución de Recencia

```python
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(rfm['recencia'], bins=30, kde=True, color='coral', edgecolor='white')
plt.axvline(rfm['recencia'].median(), color='red', linestyle='--', label=f'Mediana: {rfm["recencia"].median():.0f} días')
plt.title('Distribución de Recencia (días desde última compra)', fontweight='bold')
plt.xlabel('Días desde última compra')
plt.ylabel('Cantidad de Clientes')
plt.legend()

plt.subplot(1, 2, 2)
rfm['segmento_recencia'] = pd.cut(rfm['recencia'],
                                    bins=[-1, 30, 90, 180, 365, 9999],
                                    labels=['<30d (Activos)', '30-90d (Recientes)',
                                            '90-180d (Medio)', '180-365d (Inactivos)', '>365d (Perdidos)'])
recencia_counts = rfm['segmento_recencia'].value_counts()
colors_recencia = ['#2ecc71', '#27ae60', '#f39c12', '#e67e22', '#e74c3c']
plt.pie(recencia_counts.values, labels=recencia_counts.index, autopct='%1.1f%%',
        colors=colors_recencia, startangle=90)
plt.title('Composición por Recencia', fontweight='bold')
plt.tight_layout()
plt.show()

print(f"Clientes activos (<30 días): {(rfm['recencia'] <= 30).sum()}")
print(f"Clientes perdidos (>365 días): {(rfm['recencia'] > 365).sum()}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*3. Distribución de Recencia.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación de negocio:** La distribución de recencia revela la salud de la cartera de clientes. Idealmente, la mayoría debería estar en <30 días (activos). Si hay muchos clientes con >180 días, hay una oportunidad de campaña de reactivación. Los clientes con >365 días probablemente están perdidos y requieren una estrategia de recuperación más agresiva (descuentos fuertes o llamado directo).

---

## 4. Distribución de Frecuencia

```python
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(rfm['frecuencia'], bins=20, kde=True, color='steelblue', edgecolor='white')
plt.axvline(rfm['frecuencia'].median(), color='red', linestyle='--', label=f'Mediana: {rfm["frecuencia"].median():.0f} compras')
plt.title('Distribución de Frecuencia (número de compras)', fontweight='bold')
plt.xlabel('Número de Compras')
plt.ylabel('Cantidad de Clientes')
plt.legend()

plt.subplot(1, 2, 2)
# Boxplot por segmento de recencia (si hay datos de clientes)
if 'tipo_cliente' in clientes.columns:
    clientes_rfm_box = clientes_rfm.dropna(subset=['tipo_cliente', 'frecuencia'])
    sns.boxplot(data=clientes_rfm_box, x='tipo_cliente', y='frecuencia', palette='Set2')
    plt.title('Frecuencia por Tipo de Cliente', fontweight='bold')
    plt.xlabel('Tipo de Cliente')
    plt.ylabel('Frecuencia de Compra')

plt.tight_layout()
plt.show()

print(f"\nEstadísticas de frecuencia:")
print(f"  Media: {rfm['frecuencia'].mean():.1f}")
print(f"  Mediana: {rfm['frecuencia'].median():.1f}")
print(f"  Mínimo: {rfm['frecuencia'].min()}")
print(f"  Máximo: {rfm['frecuencia'].max()}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*4. Distribución de Frecuencia.*

1. Boxplot por segmento de recencia (si hay datos de clientes)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación de negocio:** La frecuencia de compra mide la lealtad del cliente. Una media baja con algunos clientes de alta frecuencia sugiere una base grande de clientes esporádicos y pocos clientes recurrentes. El objetivo debe ser mover clientes de frecuencia baja a media mediante programas de fidelización. Los clientes con frecuencia muy alta son candidatos a programas VIP.

---

## 5. Distribución de Monto Total

```python
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
sns.histplot(rfm['monto_total'], bins=30, kde=True, color='green', edgecolor='white')
plt.axvline(rfm['monto_total'].median(), color='red', linestyle='--',
            label=f'Mediana: ${rfm["monto_total"].median():,.0f}')
plt.axvline(rfm['monto_total'].mean(), color='blue', linestyle='--',
            label=f'Media: ${rfm["monto_total"].mean():,.0f}')
plt.title('Distribución del Monto Total por Cliente', fontweight='bold')
plt.xlabel('Monto Total ($)')
plt.ylabel('Cantidad de Clientes')
plt.legend()

plt.subplot(1, 2, 2)
sns.boxplot(data=rfm, y='monto_total', color='lightgreen')
plt.title('Boxplot del Monto Total', fontweight='bold')
plt.ylabel('Monto Total ($)')
plt.tight_layout()
plt.show()

print("Percentiles de monto total:")
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"  Percentil {p}: ${rfm['monto_total'].quantile(p/100):,.0f}")

# Índice de concentración (top 20% de clientes qué % del valor generan)
rfm_ordenado = rfm.sort_values('monto_total', ascending=False)
top20_pct = int(len(rfm_ordenado) * 0.2)
valor_top20 = rfm_ordenado.head(top20_pct)['monto_total'].sum() / rfm_ordenado['monto_total'].sum() * 100
print(f"\nTop 20% de clientes genera: {valor_top20:.1f}% del valor total")
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

*5. Distribución de Monto Total.*

1. Índice de concentración (top 20% de clientes qué % del valor generan)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación de negocio:** La distribución del monto total suele ser asimétrica (pocos clientes generan la mayoría del valor). Si el top 20% genera >60% de los ingresos, se confirma la regla de Pareto aplicada a clientes. Estos clientes de alto valor merecen gestión dedicada (ejecutivo de cuenta, atención preferencial, términos de pago flexibles).

---

## 6. Top 10 Clientes por Monto Total

```python
top10_monto = rfm.sort_values('monto_total', ascending=False).head(10).copy()
top10_monto['rank'] = range(1, 11)

# Merge con datos de clientes
top10_monto = top10_monto.merge(clientes[['cliente', 'tipo_cliente', 'ciudad']], on='cliente', how='left')

plt.figure(figsize=(12, 5))
colors = sns.color_palette('RdYlGn', n_colors=10)[::-1]
ax = sns.barplot(data=top10_monto, y='cliente', x='monto_total', palette=colors, hue='cliente', legend=False)
for i, (_, row) in enumerate(top10_monto.iterrows()):
    ax.text(row['monto_total'] + 5000, i, f'${row["monto_total"]:,.0f}', va='center', fontweight='bold')
plt.title('Top 10 Clientes por Monto Total', fontsize=14, fontweight='bold')
plt.xlabel('Monto Total ($)')
plt.ylabel('Cliente')
plt.tight_layout()
plt.show()

print("Top 10 Clientes por Monto:")
print(top10_monto[['rank', 'cliente', 'tipo_cliente', 'ciudad', 'monto_total', 'frecuencia']].to_string(index=False))
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

*6. Top 10 Clientes por Monto Total.*

1. Merge con datos de clientes

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación de negocio:** El top 10 de clientes por monto revela quiénes son los más valiosos. Si son empresas, merecen visitas comerciales periódicas. Si son individuos, programas de fidelización premium. La pérdida de un solo cliente de este top 10 puede impactar significativamente los ingresos. Se recomienda tener un plan de retención específico para cada uno.

---

## 7. Top 10 Clientes por Frecuencia

```python
top10_frec = rfm.sort_values('frecuencia', ascending=False).head(10).copy()
top10_frec['rank'] = range(1, 11)
top10_frec = top10_frec.merge(clientes[['cliente', 'tipo_cliente']], on='cliente', how='left')

plt.figure(figsize=(12, 5))
ax = sns.barplot(data=top10_frec, y='cliente', x='frecuencia', palette='Blues_r', hue='cliente', legend=False)
for i, (_, row) in enumerate(top10_frec.iterrows()):
    ax.text(row['frecuencia'] + 0.5, i, f'{int(row["frecuencia"])} compras\n${row["monto_total"]:,.0f}',
            va='center', fontsize=9, fontweight='bold')
plt.title('Top 10 Clientes por Frecuencia de Compra', fontsize=14, fontweight='bold')
plt.xlabel('Número de Compras')
plt.ylabel('Cliente')
plt.tight_layout()
plt.show()

print("Top 10 Clientes por Frecuencia:")
print(top10_frec[['rank', 'cliente', 'tipo_cliente', 'frecuencia', 'monto_total']].to_string(index=False))

# ¿Hay clientes que aparecen en ambos top 10?
top10_monto_set = set(top10_monto['cliente'])
top10_frec_set = set(top10_frec['cliente'])
interseccion = top10_monto_set & top10_frec_set
print(f"\nClientes en top 10 de monto Y frecuencia: {len(interseccion)}")
print(interseccion if interseccion else "Ninguno — sugiere que valor y lealtad no siempre coinciden")
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

*7. Top 10 Clientes por Frecuencia.*

1. ¿Hay clientes que aparecen en ambos top 10?

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación de negocio:** Los clientes con mayor frecuencia son los más leales, pero no necesariamente los de mayor monto. Un cliente que compra seguido pero tickets pequeños puede ser igual de valioso que uno que compra mucho pero rara vez. Idealmente, se busca identificar clientes que aparecen en ambas listas (alta frecuencia + alto monto): son los clientes estrella.

---

## 8. Relación Frecuencia vs Monto

```python
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(data=rfm, x='frecuencia', y='monto_total', alpha=0.6, s=80, color='purple', edgecolor='white')
# Añadir línea de tendencia
z = np.polyfit(rfm['frecuencia'], rfm['monto_total'], 1)
p = np.poly1d(z)
plt.plot(rfm['frecuencia'], p(rfm['frecuencia']), color='red', linestyle='--', linewidth=2, label='Tendencia lineal')
plt.title('Relación Frecuencia vs Monto Total', fontweight='bold')
plt.xlabel('Frecuencia (número de compras)')
plt.ylabel('Monto Total ($)')
plt.legend()

plt.subplot(1, 2, 2)
sns.lmplot(data=rfm, x='frecuencia', y='monto_total', scatter_kws={'alpha':0.5, 's':50},
           line_kws={'color': 'red'}, height=4, aspect=1.3)
plt.title('Regresión: Frecuencia vs Monto', fontweight='bold')
plt.xlabel('Frecuencia')
plt.ylabel('Monto Total ($)')
plt.tight_layout()
plt.show()

corr_frec_monto = rfm['frecuencia'].corr(rfm['monto_total'])
print(f"Correlación frecuencia vs monto: {corr_frec_monto:.4f}")
print("Interpretación:", "Correlación fuerte positiva" if corr_frec_monto > 0.7 else
      "Correlación moderada" if corr_frec_monto > 0.4 else "Correlación débil")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*8. Relación Frecuencia vs Monto.*

1. Añadir línea de tendencia

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación de negocio:** La relación entre frecuencia y monto es clave: si hay alta correlación positiva, significa que los clientes leales también gastan más. Si la correlación es débil, hay clientes que compran mucho en cada visita pero rara vez (potencial para aumentar frecuencia) y clientes que compran seguido pero tickets pequeños (potencial para upselling).

---

## 9. Distribución del Ticket Promedio

```python
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
sns.histplot(rfm['ticket_promedio'], bins=30, kde=True, color='teal', edgecolor='white')
plt.axvline(rfm['ticket_promedio'].median(), color='red', linestyle='--',
            label=f'Mediana: ${rfm["ticket_promedio"].median():,.0f}')
plt.axvline(rfm['ticket_promedio'].mean(), color='blue', linestyle='--',
            label=f'Media: ${rfm["ticket_promedio"].mean():,.0f}')
plt.title('Distribución del Ticket Promedio por Cliente', fontweight='bold')
plt.xlabel('Ticket Promedio ($)')
plt.ylabel('Cantidad de Clientes')
plt.legend()

plt.subplot(1, 2, 2)
sns.boxplot(data=rfm, y='ticket_promedio', color='lightseagreen')
plt.title('Boxplot del Ticket Promedio', fontweight='bold')
plt.ylabel('Ticket Promedio ($)')
plt.tight_layout()
plt.show()

# Prueba de normalidad (Shapiro-Wilk) en ticket promedio
stat_sw, p_sw = stats.shapiro(rfm['ticket_promedio'].sample(min(100, len(rfm['ticket_promedio']))))
print(f"Shapiro-Wilk (normalidad): estadístico={stat_sw:.4f}, p-valor={p_sw:.4f}")
print("Distribución normal:", "Sí" if p_sw > 0.05 else "No (distribución asimétrica)")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*9. Distribución del Ticket Promedio.*

1. Prueba de normalidad (Shapiro-Wilk) en ticket promedio

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación de negocio:** El ticket promedio revela cuánto gasta un cliente típico por transacción. Una distribución asimétrica con cola derecha indica que pocos clientes tienen tickets muy altos. La mediana es mejor referencia que la media para definir estrategias de precios y descuentos. Tickets promedio bajos sugieren oportunidad de upselling al momento de compra.

---

## 10. Clientes con Mayor Ticket Promedio (Top 10)

```python
top10_ticket = rfm.sort_values('ticket_promedio', ascending=False).head(10).copy()
top10_ticket['rank'] = range(1, 11)
top10_ticket = top10_ticket.merge(clientes[['cliente', 'tipo_cliente', 'ciudad']], on='cliente', how='left')

plt.figure(figsize=(12, 5))
ax = sns.barplot(data=top10_ticket, y='cliente', x='ticket_promedio', palette='magma_r', hue='cliente', legend=False)
for i, (_, row) in enumerate(top10_ticket.iterrows()):
    ax.text(row['ticket_promedio'] + 500, i, f'${row["ticket_promedio"]:,.0f}\n({int(row["frecuencia"])} compras)',
            va='center', fontsize=9, fontweight='bold')
plt.title('Top 10 Clientes por Ticket Promedio', fontsize=14, fontweight='bold')
plt.xlabel('Ticket Promedio ($)')
plt.ylabel('Cliente')
plt.tight_layout()
plt.show()

print("Top 10 Clientes por Ticket Promedio:")
print(top10_ticket[['rank', 'cliente', 'tipo_cliente', 'ciudad', 'ticket_promedio', 'frecuencia', 'monto_total']].to_string(index=False))
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

*10. Clientes con Mayor Ticket Promedio (Top 10).*

1. `top10_ticket = rfm.sort_values('ticket_promedio', ascending=False).head(10).copy()` — Ordena los datos según la columna eRealiza la operación indicada con los parámetros definidos..
2. `top10_ticket = top10_ticket.merge(clientes[['cliente', 'tipo_cliente', 'ciudad']], on='cliente', how='left')` — Combina dos DataFrames por una columna clave.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación de negocio:** Los clientes con mayor ticket promedio son estratégicos porque maximizan el valor por transacción. Si tienen frecuencia baja, el foco debe ser aumentar su frecuencia de compra (programa de referidos, descuentos por recompra). Si ya tienen frecuencia alta, son clientes premium que merecen atención personalizada y acceso exclusivo a nuevos productos.

---

## 11. Segmentación Simple: Frecuencia × Monto

```python
# Clasificar frecuencia y monto como Alto/Bajo según mediana
mediana_frec = rfm['frecuencia'].median()
mediana_monto = rfm['monto_total'].median()

rfm['segmento_frec'] = np.where(rfm['frecuencia'] > mediana_frec, 'Alta Frecuencia', 'Baja Frecuencia')
rfm['segmento_monto'] = np.where(rfm['monto_total'] > mediana_monto, 'Alto Monto', 'Bajo Monto')
rfm['segmento'] = rfm['segmento_frec'] + ' + ' + rfm['segmento_monto']

print("=== DISTRIBUCIÓN DE SEGMENTOS ===")
segmentos = rfm.groupby('segmento').agg(
    clientes=('cliente', 'count'),
    monto_promedio=('monto_total', 'mean'),
    frecuencia_promedio=('frecuencia', 'mean'),
    ticket_promedio=('ticket_promedio', 'mean'),
    ingresos_totales=('monto_total', 'sum')
).reset_index()
segmentos['% clientes'] = segmentos['clientes'] / segmentos['clientes'].sum() * 100
segmentos['% ingresos'] = segmentos['ingresos_totales'] / segmentos['ingresos_totales'].sum() * 100
print(segmentos.sort_values('ingresos_totales', ascending=False).to_string(index=False))

# Visualizar segmentos
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.countplot(data=rfm, x='segmento', hue='segmento', palette='Set2', legend=False, order=rfm['segmento'].value_counts().index)
plt.title('Cantidad de Clientes por Segmento', fontweight='bold')
plt.xticks(rotation=30, ha='right')
plt.ylabel('Clientes')

plt.subplot(1, 2, 2)
segmentos_plot = rfm.groupby('segmento')['monto_total'].sum().reset_index().sort_values('monto_total', ascending=False)
sns.barplot(data=segmentos_plot, x='segmento', y='monto_total', hue='segmento', palette='Set2', legend=False)
plt.title('Ingresos Totales por Segmento', fontweight='bold')
plt.xticks(rotation=30, ha='right')
plt.ylabel('Ingresos ($)')
plt.tight_layout()
plt.show()

# Mapa de calor de segmentos (Frecuencia vs Monto como variables continuas)
pivot_segmentos = rfm.pivot_table(
    values='monto_total', index='segmento_frec', columns='segmento_monto',
    aggfunc=['count', 'sum'], fill_value=0
)
print("\nMatriz de segmentos (clientes / ingresos):")
print(pivot_segmentos)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*11. Segmentación Simple: Frecuencia × Monto.*

1. Clasificar frecuencia y monto como Alto/Bajo según mediana
2. Visualizar segmentos
3. Mapa de calor de segmentos (Frecuencia vs Monto como variables continuas)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación de negocio:** La segmentación 2×2 (Frecuencia × Monto) produce 4 cuadrantes con estrategias distintas:
- **Alta Frecuencia + Alto Monto (VIP):** Clientes estrella. Programa de lealtad premium, atención dedicada.
- **Alta Frecuencia + Bajo Monto (Leales):** Compran seguido pero poco. Oportunidad de upselling y venta cruzada.
- **Baja Frecuencia + Alto Monto (Ocasionales):** Gastan mucho pero rara vez. Estrategia de reactivación y频次 incremento.
- **Baja Frecuencia + Bajo Monto (Transaccionales):** Base amplia. Campañas automatizadas de bajo costo.

---

## 12. Recomendaciones: Clientes VIP, en Riesgo y a Reactivar

```python
print("=" * 70)
print("RECOMENDACIONES POR SEGMENTO DE CLIENTE")
print("=" * 70)

# Identificar clientes VIP (Alta Frecuencia + Alto Monto)
vip = rfm[(rfm['segmento_frec'] == 'Alta Frecuencia') & (rfm['segmento_monto'] == 'Alto Monto')]
print(f"\n🔴 CLIENTES VIP ({len(vip)} clientes, ${vip['monto_total'].sum():,.0f} ingresos):")
print("  - Asignar ejecutivo de cuenta personal")
print("  - Descuentos por volumen y términos de pago flexibles")
print("  - Acceso anticipado a nuevos productos")
print("  - Eventos exclusivos para clientes premium")
print("  - Encuesta de satisfacción trimestral")
print(vip[['cliente', 'monto_total', 'frecuencia', 'ticket_promedio']].sort_values('monto_total', ascending=False).head().to_string(index=False))

# Identificar clientes en riesgo (recencia alta pero antes eran buenos)
en_riesgo = rfm[
    (rfm['recencia'] > 90) & (rfm['recencia'] <= 180) &
    (rfm['monto_total'] > rfm['monto_total'].median())
]
print(f"\n🟡 CLIENTES EN RIESGO ({len(en_riesgo)} clientes, ${en_riesgo['monto_total'].sum():,.0f} ingresos):")
print("  - Campaña de reactivación por email con descuento del 15%")
print("  - Recordatorio de productos vistos anteriormente")
print("  - Oferta personalizada basada en historial de compras")
print("  - Llamada de seguimiento por parte de ventas")
print(en_riesgo[['cliente', 'recencia', 'monto_total', 'frecuencia']].head().to_string(index=False))

# Identificar clientes a reactivar (recencia > 180)
a_reactivar = rfm[
    (rfm['recencia'] > 180) & (rfm['recencia'] <= 365) &
    (rfm['monto_total'] > rfm['monto_total'].quantile(0.25))
]
print(f"\n🟠 CLIENTES A REACTIVAR ({len(a_reactivar)} clientes, ${a_reactivar['monto_total'].sum():,.0f} ingresos):")
print("  - Oferta de bienvenida de regreso: 20% descuento")
print("  - Email preguntando por qué dejaron de comprar")
print("  - Campaña de retargeting en redes sociales")
print("  - Cupón de envío gratis en su próxima compra")
print(a_reactivar[['cliente', 'recencia', 'monto_total', 'frecuencia']].head().to_string(index=False))

# Clientes perdidos (> 365 días)
perdidos = rfm[rfm['recencia'] > 365]
print(f"\n⚪ CLIENTES PERDIDOS ({len(perdidos)}):")
print("  - Campaña de recuperación agresiva: 30-50% descuento")
print("  - Encuesta de salida para entender motivo")
print("  - Evaluar si vale la pena invertir en recuperarlos")

# Resumen general
print("\n" + "=" * 70)
print("PLAN DE ACCIÓN — GESTIÓN DE CLIENTES")
print("=" * 70)
acciones = [
    ("VIP", len(vip), vip['monto_total'].sum(), "Mantener y mimar"),
    ("En Riesgo", len(en_riesgo), en_riesgo['monto_total'].sum(), "Reactivar urgente"),
    ("A Reactivar", len(a_reactivar), a_reactivar['monto_total'].sum(), "Campaña de recuperación"),
    ("Perdidos", len(perdidos), perdidos['monto_total'].sum(), "Evaluar recuperación"),
]

print(f"{'Segmento':<20} {'Clientes':>10} {'Ingresos':>15} {'Acción':<30}")
print("-" * 75)
for seg, cnt, ing, acc in acciones:
    print(f"{seg:<20} {cnt:>10} {ing:>15,.0f} {acc:<30}")
print("-" * 75)
print(f"{'TOTAL':<20} {len(rfm):>10} {rfm['monto_total'].sum():>15,.0f}")
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

*12. Recomendaciones: Clientes VIP, en Riesgo y a Reactivar.*

1. Identificar clientes VIP (Alta Frecuencia + Alto Monto)
2. Identificar clientes en riesgo (recencia alta pero antes eran buenos)
3. Identificar clientes a reactivar (recencia > 180)
4. Clientes perdidos (> 365 días)
5. Resumen general

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación de negocio:** La segmentación de clientes no es solo un ejercicio académico: debe traducirse en acciones concretas de marketing y ventas. Cada segmento requiere un tratamiento diferente:
- **VIP:** Retener cueste lo que cueste.
- **En riesgo:** Actuar rápido antes de que se pierdan.
- **A reactivar:** Campaña automatizada con incentivo.
- **Perdidos:** Inversión selectiva, solo si el valor lo justifica.

El plan de acción debe implementarse en el sistema CRM y monitorearse mensualmente.

---

## 📝 Ejercicios Propuestos

1. **Segmentación RFM completa:** Implementa cuartiles para recencia, frecuencia y monto (1-4). Crea segmentos combinados (111 = mejor, 444 = peor). (Pista: usa `pd.qcut` para cada variable)

2. **Análisis de cohortes:** Crea cohortes por mes de primera compra y calcula la retención en los meses siguientes. ¿Qué cohorte tiene mejor retención? (Pista: agrupa por mes de primera compra y mes de compra)

3. **CLV simple:** Calcula el Customer Lifetime Value como monto_total / recencia * 365 (estimación anual). ¿Cuáles son los top 10 por CLV? (Pista: filtra recencia > 0)

4. **Segmentación por categoría favorita:** Para cada cliente, encuentra la categoría que más compra. ¿Qué segmentos prefieren qué categorías? (Pista: usa `mode` dentro de `groupby`)

5. **Churn prediction simple:** Define churn como recencia > 90 días. Crea un modelo de regresión logística para predecir churn usando frecuencia, ticket_promedio y productos_distintos. (Pista: usa `from sklearn.linear_model import LogisticRegression`)

---

## 📌 Resumen

| Hallazgo | Impacto |
|----------|---------|
| Top 20% clientes genera mayor parte de ingresos | Priorizar retención de alto valor |
| Distribución asimétrica de recencia | Campañas de reactivación necesarias |
| Correlación frecuencia-monto revela perfil de compra | Estrategia de upselling/venta cruzada |
| Segmentación 2×2 produce 4 tipos de clientes | Acciones específicas por cuadrante |
| Clientes VIP merecen gestión dedicada | Ejecutivo de cuenta, términos flexibles |
| Clientes en riesgo requieren acción inmediata | Campaña automatizada de reactivación |

## 🔗 Enlaces Relacionados
- [CP06 - Análisis de Estacionalidad](CP06-analisis-estacionalidad.md)
- [CP01 - Análisis de Ventas Básico](CP01-analisis-ventas-basico.md)
- [CP09 - Comparativa de Sucursales](CP09-comparativa-sucursales.md)
