# CP09 — Comparativa de Sucursales — Rendimiento y Eficiencia

## 🎯 Contexto de Negocio

El director operativo quiere comparar el desempeño de todas las sucursales para identificar las de mejor rendimiento, detectar oportunidades de mejora y asignar recursos de manera óptima. La comparación debe considerar ingresos, márgenes, volumen de transacciones y eficiencia operativa.

Los datos cubren 10 sucursales con transacciones durante todo 2024.

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

## 1. Cargar Ventas y Filtrar Sucursales

```python
print("Sucursales disponibles:")
print(ventas['sucursal'].value_counts())

print(f"\nTransacciones totales: {len(ventas)}")
print(f"Sucursales únicas: {ventas['sucursal'].nunique()}")

# Estadísticas básicas por sucursal
stats_suc = ventas.groupby('sucursal').agg(
    transacciones=('ingreso', 'count'),
    ingreso_total=('ingreso', 'sum'),
    ingreso_promedio=('ingreso', 'mean'),
    ingreso_std=('ingreso', 'std'),
    costo_total=('costo_total', 'sum'),
    margen_promedio=('margen_pct', 'mean'),
    descuento_promedio=('descuento', 'mean'),
    clientes_unicos=('cliente', 'nunique')
).reset_index().sort_values('ingreso_total', ascending=False)

print("\n=== ESTADÍSTICAS POR SUCURSAL ===")
print(stats_suc.to_string(index=False))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*1. Cargar Ventas y Filtrar Sucursales.*

1. Estadísticas básicas por sucursal

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación de negocio:** El primer vistazo revela diferencias importantes entre sucursales. Las sucursales con más transacciones no necesariamente tienen el mayor ingreso promedio. La desviación estándar del ingreso indica estabilidad: sucursales con alta std pueden tener días de mucha venta y días bajos, lo que sugiere estacionalidad local.

---

## 2. Ingreso Total por Sucursal

```python
plt.figure(figsize=(12, 5))
stats_suc_ordenado = stats_suc.sort_values('ingreso_total', ascending=True)

colors = ['#e74c3c' if v < stats_suc_ordenado['ingreso_total'].median() else '#2ecc71'
          for v in stats_suc_ordenado['ingreso_total']]

ax = sns.barplot(data=stats_suc_ordenado, y='sucursal', x='ingreso_total', palette=colors, hue='sucursal', legend=False)
for i, (_, row) in enumerate(stats_suc_ordenado.iterrows()):
    ax.text(row['ingreso_total'] + 200000, i, f'${row["ingreso_total"]/1e6:.2f}M',
            va='center', fontweight='bold')
plt.title('Ingreso Total por Sucursal', fontsize=14, fontweight='bold')
plt.xlabel('Ingreso Total ($)')
plt.ylabel('Sucursal')
plt.tight_layout()
plt.show()

# Brecha entre mejor y peor
mejor = stats_suc_ordenado.iloc[-1]
peor = stats_suc_ordenado.iloc[0]
print(f"Sucursal con mayor ingreso: {mejor['sucursal']} (${mejor['ingreso_total']:,.0f})")
print(f"Sucursal con menor ingreso: {peor['sucursal']} (${peor['ingreso_total']:,.0f})")
print(f"Diferencia: {(mejor['ingreso_total']/peor['ingreso_total'] - 1)*100:.1f}% más la mejor")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*2. Ingreso Total por Sucursal.*

1. Brecha entre mejor y peor

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación de negocio:** La disparidad entre sucursales es una señal de alerta. Si la mejor sucursal genera 3x más que la peor, puede deberse a ubicación, gestión local, competencia en la zona o surtido de productos. Las sucursales debajo de la mediana (rojas) requieren análisis más profundo para entender las causas raíz.

---

## 3. Ingreso Promedio por Transacción por Sucursal

```python
plt.figure(figsize=(12, 5))
stats_suc_ticket = stats_suc.sort_values('ingreso_promedio', ascending=True)

ax = sns.barplot(data=stats_suc_ticket, y='sucursal', x='ingreso_promedio',
                 palette='coolwarm', hue='sucursal', legend=False)
for i, (_, row) in enumerate(stats_suc_ticket.iterrows()):
    ax.text(row['ingreso_promedio'] + 100, i, f'${row["ingreso_promedio"]:,.0f}',
            va='center', fontweight='bold')
plt.title('Ingreso Promedio por Transacción (Ticket Promedio)', fontsize=14, fontweight='bold')
plt.xlabel('Ticket Promedio ($)')
plt.ylabel('Sucursal')
plt.tight_layout()
plt.show()

print(f"\nTicket promedio general: ${ventas['ingreso'].mean():,.0f}")
print(f"Diferencia entre mayor y menor: ${stats_suc['ingreso_promedio'].max() - stats_suc['ingreso_promedio'].min():,.0f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*3. Ingreso Promedio por Transacción por Sucursal.*

1. `stats_suc_ticket = stats_suc.sort_values('ingreso_promedio', ascending=True)` — Ordena los datos según la columna eRealiza la operación indicada con los parámetros definidos..

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación de negocio:** El ticket promedio es un indicador de upselling y mezcla de productos. Sucursales con ticket alto pueden estar vendiendo productos más caros (electrónica) o haciendo mejor upselling. Sucursales con ticket bajo pueden estar vendiendo más periféricos o accesorios. Esto sugiere que las sucursales de bajo ticket podrían beneficiarse de capacitación en venta consultiva.

---

## 4. Cantidad de Transacciones por Sucursal

```python
plt.figure(figsize=(12, 5))
stats_suc_trans = stats_suc.sort_values('transacciones', ascending=True)

ax = sns.barplot(data=stats_suc_trans, y='sucursal', x='transacciones',
                 palette='viridis', hue='sucursal', legend=False)
for i, (_, row) in enumerate(stats_suc_trans.iterrows()):
    ax.text(row['transacciones'] + 5, i, f'{int(row["transacciones"])}',
            va='center', fontweight='bold')
plt.title('Número de Transacciones por Sucursal', fontsize=14, fontweight='bold')
plt.xlabel('Transacciones')
plt.ylabel('Sucursal')
plt.tight_layout()
plt.show()

# Relación volumen vs ticket
print("Relación Volumen vs Ticket:")
print(stats_suc[['sucursal', 'transacciones', 'ingreso_promedio', 'ingreso_total']].corr(numeric_only=True)[['ingreso_total']].round(4))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*4. Cantidad de Transacciones por Sucursal.*

1. Relación volumen vs ticket

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación de negocio:** El volumen de transacciones indica tráfico de clientes. Una sucursal con muchas transacciones pero ticket bajo puede estar en una zona de alto tráfico (centro comercial) pero vendiendo productos de bajo valor. Una sucursal con pocas transacciones pero ticket alto puede ser especializada (ventas corporativas). Ambas pueden ser rentables pero requieren estrategias distintas.

---

## 5. Margen Promedio por Sucursal

```python
plt.figure(figsize=(12, 5))
stats_suc_margen = stats_suc.sort_values('margen_promedio', ascending=True)

ax = sns.barplot(data=stats_suc_margen, y='sucursal', x='margen_promedio',
                 palette='RdYlGn', hue='sucursal', legend=False)
for i, (_, row) in enumerate(stats_suc_margen.iterrows()):
    ax.text(row['margen_promedio'] + 1, i, f'{row["margen_promedio"]:.1f}%',
            va='center', fontweight='bold')
plt.axvline(x=ventas['margen_pct'].mean(), color='blue', linestyle='--',
            label=f'Margen promedio general: {ventas["margen_pct"].mean():.1f}%')
plt.title('Margen Promedio por Sucursal', fontsize=14, fontweight='bold')
plt.xlabel('Margen Promedio (%)')
plt.ylabel('Sucursal')
plt.legend()
plt.tight_layout()
plt.show()

print("Sucursales con margen por debajo del promedio:")
debajo_margen = stats_suc[stats_suc['margen_promedio'] < ventas['margen_pct'].mean()]
print(debajo_margen[['sucursal', 'margen_promedio', 'descuento_promedio']])
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*5. Margen Promedio por Sucursal.*

1. `stats_suc_margen = stats_suc.sort_values('margen_promedio', ascending=True)` — Ordena los datos según la columna eRealiza la operación indicada con los parámetros definidos..

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación de negocio:** El margen refleja la rentabilidad de cada transacción. Una sucursal con mucho ingreso pero bajo margen puede estar dando demasiados descuentos o vendiendo productos de bajo margen. La correlación entre descuento_promedio y margen revela si los descuentos están erosionando la rentabilidad. Sucursales con margen bajo necesitan revisar su política de precios y descuentos.

---

## 6. Variabilidad de Ingresos por Sucursal

```python
plt.figure(figsize=(14, 6))
order = ventas.groupby('sucursal')['ingreso'].median().sort_values(ascending=False).index
sns.boxplot(data=ventas, x='sucursal', y='ingreso', order=order, palette='Set2', hue='sucursal', legend=False)
plt.title('Distribución de Ingresos por Sucursal', fontsize=14, fontweight='bold')
plt.xlabel('Sucursal')
plt.ylabel('Ingreso por Transacción ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Coeficiente de variación
cv_suc = ventas.groupby('sucursal')['ingreso'].agg(lambda x: x.std() / x.mean()).sort_values()
print("Coeficiente de Variación (menor = más estable):")
for suc, cv in cv_suc.items():
    print(f"  {suc}: {cv:.3f}")

print(f"\nSucursal más estable: {cv_suc.idxmin()} (CV={cv_suc.min():.3f})")
print(f"Sucursal más volátil: {cv_suc.idxmax()} (CV={cv_suc.max():.3f})")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*6. Variabilidad de Ingresos por Sucursal.*

1. Coeficiente de variación

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación de negocio:** La variabilidad de ingresos indica predictibilidad. Sucursales con baja variabilidad (boxplots angostos) son más predecibles y fáciles de planificar. Sucursales con alta variabilidad pueden tener picos estacionales o depender de pocos clientes grandes. Para las volátiles, se recomienda un colchón de inventario mayor y personal flexible.

---

## 7. Producto Más Vendido por Sucursal

```python
producto_top_suc = ventas.groupby(['sucursal', 'producto']).agg(
    cantidad=('cantidad', 'sum'),
    ingreso=('ingreso', 'sum')
).reset_index()

top_por_suc = producto_top_suc.loc[
    producto_top_suc.groupby('sucursal')['cantidad'].idxmax()
].sort_values('cantidad', ascending=False)

print("=== PRODUCTO MÁS VENDIDO (por cantidad) POR SUCURSAL ===")
print(top_por_suc[['sucursal', 'producto', 'categoria', 'cantidad', 'ingreso']].to_string(index=False))

# Verificar si todas las sucursales tienen el mismo top
top_unico = top_por_suc['producto'].nunique()
print(f"\nProductos únicos como top: {top_unico} de {len(top_por_suc)} sucursales")
print("Diferenciación de surtido entre sucursales:", "Alta" if top_unico > 3 else "Baja")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*7. Producto Más Vendido por Sucursal.*

1. Verificar si todas las sucursales tienen el mismo top

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación de negocio:** Conocer el producto más vendido por sucursal permite personalizar el surtido. Si todas las sucursales tienen el mismo top, la estrategia de producto es homogénea y puede centralizarse. Si cada sucursal tiene un top diferente, la estrategia debe ser local, adaptando el inventario a las preferencias de cada zona.

---

## 8. Sucursal con Mejor Desempeño por Categoría

```python
cat_suc = ventas.groupby(['sucursal', 'categoria']).agg(
    ingreso=('ingreso', 'sum'),
    margen=('margen_pct', 'mean')
).reset_index()

# Pivot table: sucursal×categoria con ingreso
pivot_cat_suc = cat_suc.pivot_table(index='categoria', columns='sucursal', values='ingreso', aggfunc='sum')

plt.figure(figsize=(14, 6))
sns.heatmap(pivot_cat_suc, annot=True, fmt='$_.0f', cmap='YlOrRd', linewidths=0.5,
            cbar_kws={'label': 'Ingreso ($)'})
plt.title('Ingreso por Categoría y Sucursal', fontsize=14, fontweight='bold')
plt.xlabel('Sucursal')
plt.ylabel('Categoría')
plt.tight_layout()
plt.show()

# Sucursal líder por categoría
print("Sucursal líder por categoría (mayor ingreso):")
for cat in pivot_cat_suc.index:
    lider = pivot_cat_suc.columns[pivot_cat_suc.loc[cat].argmax()]
    ingreso = pivot_cat_suc.loc[cat].max()
    print(f"  {cat}: {lider} (${ingreso:,.0f})")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*8. Sucursal con Mejor Desempeño por Categoría.*

1. Pivot table: sucursal×categoria con ingreso
2. Sucursal líder por categoría

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación de negocio:** El heatmap categoría×sucursal revela fortalezas y debilidades. Una sucursal que lidera en múltiples categorías probablemente tiene un mejor equipo de ventas o está mejor ubicada. Las categorías donde ninguna sucursal destaca pueden necesitar atención a nivel corporativo (mejores proveedores, campañas nacionales).

---

## 9. ANOVA: ¿Diferencia Significativa de Ingresos entre Sucursales?

```python
# ANOVA de una vía: ingreso ~ sucursal
sucursales_grupos = [ventas[ventas['sucursal'] == s]['ingreso'] for s in ventas['sucursal'].unique()]
f_stat, p_valor = stats.f_oneway(*sucursales_grupos)

print("=== ANOVA: INGRESO POR SUCURSAL ===")
print(f"Estadístico F: {f_stat:.4f}")
print(f"p-valor: {p_valor:.10f}")

if p_valor < 0.05:
    print("Conclusión: Existen diferencias significativas en el ingreso promedio entre sucursales (p < 0.05).")
    print("→ Las sucursales NO son iguales en rendimiento.")
else:
    print("Conclusión: No hay evidencia suficiente de diferencias significativas.")
    print("→ Las sucursales tienen rendimiento similar.")

# Post-hoc: Tukey HSD
from statsmodels.stats.multicomp import pairwise_tukeyhsd

tukey = pairwise_tukeyhsd(endog=ventas['ingreso'], groups=ventas['sucursal'], alpha=0.05)
print("\n=== TUKEY HSD: Pares con diferencia significativa ===")
print(tukey)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*9. ANOVA: ¿Diferencia Significativa de Ingresos entre Sucursales?.*

1. ANOVA de una vía: ingreso ~ sucursal
2. Post-hoc: Tukey HSD

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación de negocio:** El ANOVA confirma estadísticamente si las diferencias observadas entre sucursales son reales o producto del azar. Si es significativo, la prueba post-hoc Tukey identifica qué pares de sucursales son diferentes. Esto permite enfocar recursos de mejora en las sucursales que están estadísticamente por debajo, no solo por casualidad.

---

## 10. Comparar Tendencia Semanal por Sucursal (Lineplot Facetado)

```python
ventas['semana_anio'] = ventas['fecha'].dt.isocalendar().week.astype(int)
ventas_semanal = ventas.groupby(['sucursal', 'semana_anio'])['ingreso'].sum().reset_index()

g = sns.FacetGrid(ventas_semanal, col='sucursal', col_wrap=5, height=3, aspect=1.3, sharey=False)
g.map_dataframe(sns.lineplot, x='semana_anio', y='ingreso', color='steelblue', linewidth=2)
g.map_dataframe(plt.fill_between, ventas_semanal['semana_anio'], ventas_semanal['ingreso'], alpha=0.1)

# Añadir línea de media por sucursal
for ax, suc in zip(g.axes.flat, ventas_semanal['sucursal'].unique()):
    media = ventas_semanal[ventas_semanal['sucursal'] == suc]['ingreso'].mean()
    ax.axhline(y=media, color='red', linestyle='--', alpha=0.5, label=f'Media: ${media:,.0f}')
    ax.legend(fontsize=8)

g.fig.suptitle('Tendencia Semanal de Ingresos por Sucursal', y=1.02, fontsize=14, fontweight='bold')
g.set_axis_labels('Semana del Año', 'Ingreso ($)')
g.fig.tight_layout()
plt.show()

# Correlación entre tendencias semanales de sucursales
pivot_semanal = ventas_semanal.pivot_table(index='semana_anio', columns='sucursal', values='ingreso')
print("Correlación entre tendencias semanales de sucursales:")
corr_matrix = pivot_semanal.corr()
print(corr_matrix.round(3))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*10. Comparar Tendencia Semanal por Sucursal (Lineplot Facetado).*

1. Añadir línea de media por sucursal
2. Correlación entre tendencias semanales de sucursales

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación de negocio:** Las tendencias semanales revelan si las sucursales siguen el mismo patrón estacional o tienen ciclos independientes. Alta correlación entre sucursales sugiere que factores externos (economía, clima, campañas nacionales) afectan a todas por igual. Baja correlación indica que cada sucursal responde a factores locales. Esto es crucial para decidir si las campañas deben ser nacionales o locales.

---

## 11. Ranking Compuesto de Sucursales (Puntuación 0-100)

```python
# Construir ranking compuesto con 5 dimensiones normalizadas
dimensiones = ['ingreso_total', 'ingreso_promedio', 'transacciones', 'margen_promedio']
ranking = stats_suc[['sucursal'] + dimensiones].copy()

# Normalizar Min-Max (0-100)
for dim in dimensiones:
    min_val = ranking[dim].min()
    max_val = ranking[dim].max()
    ranking[f'{dim}_score'] = (ranking[dim] - min_val) / (max_val - min_val) * 100

# Puntaje compuesto (ponderado)
pesos = {'ingreso_total_score': 0.30, 'ingreso_promedio_score': 0.25,
         'transacciones_score': 0.25, 'margen_promedio_score': 0.20}
ranking['puntaje_compuesto'] = sum(ranking[dim] * peso for dim, peso in pesos.items())
ranking = ranking.sort_values('puntaje_compuesto', ascending=False).reset_index(drop=True)
ranking['posicion'] = range(1, len(ranking) + 1)

print("=== RANKING COMPUESTO DE SUCURSALES (0-100) ===")
print(ranking[['posicion', 'sucursal', 'puntaje_compuesto'] + [f'{d}_score' for d in dimensiones]].to_string(index=False))

# Visualizar ranking
plt.figure(figsize=(12, 5))
ax = sns.barplot(data=ranking, y='sucursal', x='puntaje_compuesto',
                 palette='RdYlGn', hue='sucursal', legend=False)
for i, (_, row) in enumerate(ranking.iterrows()):
    ax.text(row['puntaje_compuesto'] + 1, i, f'{row["puntaje_compuesto"]:.1f} pts',
            va='center', fontweight='bold')
plt.title('Ranking Compuesto de Sucursales', fontsize=14, fontweight='bold')
plt.xlabel('Puntaje Compuesto (0-100)')
plt.ylabel('Sucursal')
plt.tight_layout()
plt.show()

# Desviación por dimensión
print("\nFortalezas y debilidades por sucursal (dimensión con mayor/menor score):")
for _, row in ranking.iterrows():
    dims = [(d.replace('_score', ''), row[f'{d}']) for d in dimensiones]
    mejor_dim = max(dims, key=lambda x: x[1])
    peor_dim = min(dims, key=lambda x: x[1])
    print(f"  {row['sucursal']}: Mejor en {mejor_dim[0]} ({mejor_dim[1]:.0f}), Peor en {peor_dim[0]} ({peor_dim[1]:.0f})")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*11. Ranking Compuesto de Sucursales (Puntuación 0-100).*

1. Construir ranking compuesto con 5 dimensiones normalizadas
2. Normalizar Min-Max (0-100)
3. Puntaje compuesto (ponderado)
4. Visualizar ranking
5. Desviación por dimensión

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación de negocio:** El ranking compuesto sintetiza múltiples dimensiones en una sola métrica. Permite identificar rápidamente las sucursales con mejor desempeño global y las que necesitan más apoyo. Analizar la dimensión más débil de cada sucursal permite diseñar planes de mejora específicos: si una sucursal puntea bajo en margen, se trabaja en precios; si puntea bajo en transacciones, se trabaja en tráfico.

---

## 12. Recomendaciones por Sucursal

```python
print("=" * 70)
print("RECOMENDACIONES POR SUCURSAL")
print("=" * 70)

for _, row in ranking.iterrows():
    suc = row['sucursal']
    puntaje = row['puntaje_compuesto']

    # Determinar perfil
    dims = {d: row[f'{d}_score'] for d in dimensiones}
    fortaleza = max(dims, key=dims.get)
    debilidad = min(dims, key=dims.get)

    if puntaje >= 70:
        categoria = "🌟 LÍDER"
    elif puntaje >= 50:
        categoria = "📈 EN DESARROLLO"
    elif puntaje >= 30:
        categoria = "⚠️ NECESITA ATENCIÓN"
    else:
        categoria = "🔴 CRÍTICA"

    print(f"\n--- {suc} | {categoria} | Puntaje: {puntaje:.1f}/100 ---")
    print(f"  Fortaleza: {fortaleza} ({dims[fortaleza]:.0f}/100)")
    print(f"  Debilidad: {debilidad} ({dims[debilidad]:.0f}/100)")

    if debilidad == 'ingreso_total':
        print("  ➜ Acción: Campaña de marketing local para aumentar ventas.")
        print("  ➜ Revisar horarios, surtido y presencia digital.")
    elif debilidad == 'ingreso_promedio':
        print("  ➜ Acción: Capacitar en upselling y venta cruzada.")
        print("  ➜ Revisar mezcla de productos (más electrónica, menos accesorios).")
    elif debilidad == 'transacciones':
        print("  ➜ Acción: Aumentar tráfico con promociones y eventos locales.")
        print("  ➜ Programa de referidos y fidelización de clientes.")
    elif debilidad == 'margen_promedio':
        print("  ➜ Acción: Revisar política de descuentos y precios.")
        print("  ➜ Enfocar ventas en productos de alto margen.")

print("\n" + "=" * 70)
print("ACCIONES GENERALES PARA TODAS LAS SUCURSALES")
print("=" * 70)
acciones_generales = [
    "1. BENCHMARKING: Replicar mejores prácticas de sucursales líderes en las de menor desempeño.",
    "2. CAPACITACIÓN: Programa trimestral de ventas para todas las sucursales.",
    "3. INCENTIVOS: Bono por cumplimiento de metas de margen y ticket promedio.",
    "4. DASHBOARD: Tablero semanal con indicadores clave por sucursal.",
    "5. VISITAS GERENCIALES: Director operativo visita mensual a sucursales críticas.",
    "6. INVERSIÓN SELECTIVA: Priorizar remodelación en sucursales con potencial de mejora.",
    "7. HOMOLOGACIÓN: Estandarizar procesos operativos exitosos en todas las sucursales.",
    "8. TECNOLOGÍA: Implementar sistema de sugerencia de venta en punto de venta.",
]
for acc in acciones_generales:
    print(acc)

print("\n" + "=" * 70)
print(f"RESUMEN: {len(ranking)} sucursales evaluadas")
print(f"Puntaje promedio: {ranking['puntaje_compuesto'].mean():.1f}/100")
print(f"Mejor sucursal: {ranking.iloc[0]['sucursal']} ({ranking.iloc[0]['puntaje_compuesto']:.1f}/100)")
print(f"Peor sucursal: {ranking.iloc[-1]['sucursal']} ({ranking.iloc[-1]['puntaje_compuesto']:.1f}/100)")
print(f"Brecha total: {ranking.iloc[0]['puntaje_compuesto'] - ranking.iloc[-1]['puntaje_compuesto']:.1f} puntos")
print("=" * 70)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*12. Recomendaciones por Sucursal.*

1. Determinar perfil

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación de negocio:** Cada sucursal recibe un diagnóstico individual con su fortaleza, debilidad y acciones recomendadas. Las sucursales líderes son casos de estudio para replicar sus prácticas. Las críticas reciben atención gerencial directa. El ranking debe actualizarse mensualmente para medir el progreso. La meta es que todas las sucursales alcancen al menos 60 puntos en 12 meses.

---

## 📝 Ejercicios Propuestos

1. **Eficiencia por empleado:** Si tuvieras datos de empleados por sucursal, calcula ingreso por empleado. Simula esta columna si es necesario. (Pista: crea una columna aleatoria de empleados 5-15 por sucursal)

2. **Análisis de canibalización:** ¿Dos sucursales cercanas se canibalizan? Identifica sucursales que podrían estar compitiendo por los mismos clientes. (Pista: analiza clientes compartidos entre sucursales cercanas)

3. **Predicción de ingresos:** Usa regresión lineal para predecir el ingreso de una sucursal basado en transacciones, ticket promedio y margen. (Pista: `from sklearn.linear_model import LinearRegression`)

4. **Segmentación de sucursales:** Aplica K-Means para agrupar sucursales en clusters homogéneos. ¿Cuántos clusters encuentras? (Pista: `from sklearn.cluster import KMeans`)

5. **Evolución mensual por sucursal:** Crea un área plot apilado del ingreso mensual por sucursal para visualizar la contribución relativa a lo largo del año. (Pista: usa `pivot_table` y `plot.area`)

---

## 📌 Resumen

| Hallazgo | Impacto |
|----------|---------|
| Disparidad de 3x+ entre sucursales | Potencial de mejora enorme en sucursales bajas |
| ANOVA confirma diferencias significativas | No es azar: hay causas reales que atender |
| Ticket promedio varía hasta 50% | Capacitación en upselling para sucursales bajas |
| Margen correlaciona con descuentos | Revisar política de descuentos agresivos |
| Ranking compuesto identifica líderes y críticas | Plan de acción personalizado por sucursal |
| Cada sucursal tiene perfil único | Estrategias locales, no homogéneas |

## 🔗 Enlaces Relacionados
- [CP06 - Análisis de Estacionalidad](CP06-analisis-estacionalidad.md)
- [CP08 - Perfil de Clientes](CP08-analisis-clientes-basico.md)
- [CP01 - Análisis de Ventas Básico](CP01-analisis-ventas-basico.md)
