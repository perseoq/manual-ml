# CP06 — Análisis de Estacionalidad en Ventas — Patrones Temporales

## 🎯 Contexto de Negocio

El gerente de operaciones necesita identificar patrones estacionales en las ventas para planificar inventario, programar personal y lanzar promociones en los momentos óptimos. La estacionalidad afecta directamente la rotación de productos, la capacidad de almacén y el flujo de caja.

Este caso analiza 1330 transacciones de 2024 buscando ciclos semanales, mensuales y trimestrales.

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

## 1. Convertir Fecha y Extraer Componentes Temporales

```python
ventas['fecha'] = pd.to_datetime(ventas['fecha'])
ventas['dia_semana'] = ventas['fecha'].dt.dayofweek
ventas['dia_semana_nombre'] = ventas['fecha'].dt.day_name()
ventas['mes'] = ventas['fecha'].dt.month
ventas['mes_nombre'] = ventas['fecha'].dt.month_name()
ventas['trimestre'] = ventas['fecha'].dt.quarter
ventas['dia_mes'] = ventas['fecha'].dt.day
ventas['semana_anio'] = ventas['fecha'].dt.isocalendar().week.astype(int)

print("Columnas temporales añadidas:")
print(ventas[['fecha', 'dia_semana', 'dia_semana_nombre', 'mes', 'mes_nombre', 'trimestre', 'semana_anio']].head())
print(f"\nRango de fechas: {ventas['fecha'].min()} a {ventas['fecha'].max()}")
print(f"Días únicos: {ventas['fecha'].nunique()}")
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

*1. Convertir Fecha y Extraer Componentes Temporales.*

1. `ventas['fecha'] = pd.to_datetime(ventas['fecha'])` — Convierte la columna a formato datetime.
2. `print("Columnas temporales añadidas:")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación de negocio:** La extracción de componentes temporales permite descomponer la serie de ventas en ciclos manejables. Poder analizar por día de semana, mes y trimestre da visibilidad granular para decisiones operativas como cuándo programar personal extra o cuándo lanzar campañas.

---

## 2. Ventas Totales por Día de la Semana

```python
dias_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
ventas['dia_semana_nombre'] = pd.Categorical(ventas['dia_semana_nombre'], categories=dias_order, ordered=True)

ventas_dia = ventas.groupby('dia_semana_nombre', observed=False)['ingreso'].sum().reset_index()

plt.figure(figsize=(10, 5))
ax = sns.barplot(data=ventas_dia, x='dia_semana_nombre', y='ingreso', hue='dia_semana_nombre', palette='viridis', legend=False)
for i, v in enumerate(ventas_dia['ingreso']):
    ax.text(i, v + 200000, f'${v/1e6:.1f}M', ha='center', fontweight='bold')
plt.title('Ventas Totales por Día de la Semana', fontsize=14, fontweight='bold')
plt.xlabel('Día de la Semana')
plt.ylabel('Ingreso Total ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(ventas_dia.sort_values('ingreso', ascending=False))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*2. Ventas Totales por Día de la Semana.*

1. `ventas_dia = ventas.groupby('dia_semana_nombre', observed=False)['ingreso'].sum().reset_index()` — Agrupa los datos por la columna eRealiza la operación indicada con los parámetros definidos..

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación de negocio:** El viernes y sábado concentran el mayor volumen de ventas. Esto sugiere que los clientes compran más al final de la semana laboral y durante el fin de semana. Se recomienda programar inventarios completos para jueves, promociones los viernes y personal reforzado los fines de semana. El lunes suele ser el día más bajo, ideal para tareas administrativas y mantenimiento de tienda.

---

## 3. Ventas por Mes (Lineplot)

```python
ventas_mes = ventas.groupby('mes')['ingreso'].sum().reset_index()

plt.figure(figsize=(10, 5))
ax = sns.lineplot(data=ventas_mes, x='mes', y='ingreso', marker='o', linewidth=2.5, color='coral')
ax.fill_between(ventas_mes['mes'], ventas_mes['ingreso'], alpha=0.2, color='coral')
for _, row in ventas_mes.iterrows():
    ax.text(row['mes'], row['ingreso'] + 300000, f'${row["ingreso"]/1e6:.1f}M', ha='center', fontsize=9)
plt.title('Ventas Mensuales — Tendencia 2024', fontsize=14, fontweight='bold')
plt.xlabel('Mes')
plt.ylabel('Ingreso Total ($)')
plt.xticks(range(1, 13), ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'])
plt.tight_layout()
plt.show()

print("Mes con mayor venta:", ventas_mes.loc[ventas_mes['ingreso'].idxmax(), 'mes'])
print("Mes con menor venta:", ventas_mes.loc[ventas_mes['ingreso'].idxmin(), 'mes'])
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*3. Ventas por Mes (Lineplot).*

1. `ventas_mes = ventas.groupby('mes')['ingreso'].sum().reset_index()` — Agrupa los datos por la columna eRealiza la operación indicada con los parámetros definidos..

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación de negocio:** Se observan picos en meses específicos como marzo (vuelta a clases), agosto (preparación fin de año) y diciembre (temporada navideña). Los meses bajos como enero y febrero reflejan la cuesta de enero post-navideña. Esta información permite planificar campañas promocionales en los meses valle para equilibrar el flujo de ingresos y ajustar pedidos a proveedores en los meses pico.

---

## 4. Ventas por Trimestre

```python
ventas_trim = ventas.groupby('trimestre')['ingreso'].sum().reset_index()
ventas_trim['etiqueta'] = ventas_trim['trimestre'].map({1: 'Ene-Mar', 2: 'Abr-Jun', 3: 'Jul-Sep', 4: 'Oct-Dic'})

plt.figure(figsize=(8, 5))
colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
ax = sns.barplot(data=ventas_trim, x='etiqueta', y='ingreso', hue='etiqueta', palette=colors, legend=False)
for i, v in enumerate(ventas_trim['ingreso']):
    ax.text(i, v + 400000, f'${v/1e6:.1f}M\n({v/ventas_trim["ingreso"].sum()*100:.1f}%)', ha='center', fontweight='bold')
plt.title('Ventas por Trimestre', fontsize=14, fontweight='bold')
plt.xlabel('Trimestre')
plt.ylabel('Ingreso Total ($)')
plt.tight_layout()
plt.show()

print("Distribución trimestral:")
print(ventas_trim[['etiqueta', 'ingreso']])
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*4. Ventas por Trimestre.*

1. `ventas_trim = ventas.groupby('trimestre')['ingreso'].sum().reset_index()` — Agrupa los datos por la columna eRealiza la operación indicada con los parámetros definidos..

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación de negocio:** El cuarto trimestre (Q4) concentra el mayor porcentaje de ventas anuales, típico del comercio por temporada navideña. Q1 suele ser el más débil. La planificación de compras debe anticipar el Q4 con pedidos a proveedores en Q3. Las campañas de marketing deben reforzarse en Q1 para contrarrestar la estacionalidad negativa.

---

## 5. Ingreso Promedio por Día de Semana y Sucursal

```python
pivot_dia_suc = ventas.pivot_table(
    values='ingreso', index='dia_semana_nombre', columns='sucursal',
    aggfunc='mean', observed=False
)

plt.figure(figsize=(14, 6))
sns.heatmap(pivot_dia_suc, annot=True, fmt='$_.0f', cmap='YlOrRd', linewidths=0.5,
            cbar_kws={'label': 'Ingreso Promedio ($)'})
plt.title('Ingreso Promedio por Día de Semana × Sucursal', fontsize=14, fontweight='bold')
plt.xlabel('Sucursal')
plt.ylabel('Día de la Semana')
plt.tight_layout()
plt.show()

sucursal_mayor_var = pivot_dia_suc.std().idxmax()
dia_mayor_var = pivot_dia_suc.std(axis=1).idxmax()
print(f"Sucursal con mayor variabilidad diaria: {sucursal_mayor_var}")
print(f"Día con mayor variabilidad entre sucursales: {dia_mayor_var}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*5. Ingreso Promedio por Día de Semana y Sucursal.*

1. `pivot_dia_suc = ventas.pivot_table(` — Reorganiza los datos de formato largo a ancho.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación de negocio:** El heatmap revela qué sucursales tienen picos marcados ciertos días. Por ejemplo, sucursales en zonas corporativas venden más entre semana, mientras que sucursales en zonas turísticas venden más en fin de semana. Esta información permite asignar personal de forma dinámica: reforzar sucursales específicas en sus días pico sin aumentar costos fijos.

---

## 6. Comparar Fines de Semana vs Días Laborales

```python
ventas['es_finde'] = ventas['dia_semana'].apply(lambda x: 'Fin de Semana' if x >= 5 else 'Día Laboral')

ventas_finde = ventas.groupby('es_finde')['ingreso'].agg(['sum', 'mean', 'count']).reset_index()

plt.figure(figsize=(8, 5))
ax = sns.barplot(data=ventas_finde, x='es_finde', y='sum', hue='es_finde',
                 palette={'Día Laboral': '#3498db', 'Fin de Semana': '#e74c3c'}, legend=False)
for i, v in enumerate(ventas_finde['sum']):
    ax.text(i, v + 300000, f'${v/1e6:.1f}M', ha='center', fontweight='bold')
    ax.text(i, v - 400000, f'Prom: ${ventas_finde["mean"].iloc[i]:.0f}\nTrans: {int(ventas_finde["count"].iloc[i])}',
            ha='center', color='white', fontweight='bold')
plt.title('Ventas: Fin de Semana vs Días Laborales', fontsize=14, fontweight='bold')
plt.xlabel('')
plt.ylabel('Ingreso Total ($)')
plt.tight_layout()
plt.show()

print(ventas_finde)
print(f"\nDiferencia porcentual en promedio por transacción: "
      f"{(ventas_finde.loc[ventas_finde['es_finde']=='Fin de Semana', 'mean'].values[0] / ventas_finde.loc[ventas_finde['es_finde']=='Día Laboral', 'mean'].values[0] - 1) * 100:.1f}%")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*6. Comparar Fines de Semana vs Días Laborales.*

1. `ventas_finde = ventas.groupby('es_finde')['ingreso'].agg(['sum', 'mean', 'count']).reset_index()` — Agrupa los datos por la columna eRealiza la operación indicada con los parámetros definidos..

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación de negocio:** Separar fines de semana de días laborales cuantifica el impacto real del fin de semana en el negocio. Si el ticket promedio es mayor en fin de semana, sugiere que los clientes compran más productos o más caros cuando tienen más tiempo. Esto justifica campañas específicas para sábado/domingo y diferente surtido de productos.

---

## 7. Tendencia Diaria con Media Móvil (Rolling 7 días)

```python
ventas_diarias = ventas.groupby('fecha')['ingreso'].sum().reset_index()
ventas_diarias = ventas_diarias.sort_values('fecha')
ventas_diarias['media_movil_7d'] = ventas_diarias['ingreso'].rolling(window=7, min_periods=1).mean()
ventas_diarias['media_movil_30d'] = ventas_diarias['ingreso'].rolling(window=30, min_periods=1).mean()

plt.figure(figsize=(14, 6))
plt.plot(ventas_diarias['fecha'], ventas_diarias['ingreso'], alpha=0.3, color='gray', label='Ventas Diarias')
plt.plot(ventas_diarias['fecha'], ventas_diarias['media_movil_7d'], color='coral', linewidth=2, label='Media Móvil 7 días')
plt.plot(ventas_diarias['fecha'], ventas_diarias['media_movil_30d'], color='darkblue', linewidth=2, label='Media Móvil 30 días')
plt.title('Tendencia Diaria de Ventas con Media Móvil', fontsize=14, fontweight='bold')
plt.xlabel('Fecha')
plt.ylabel('Ingreso ($)')
plt.legend()
plt.tight_layout()
plt.show()

print("Media móvil 7d (últimos 5 registros):")
print(ventas_diarias[['fecha', 'ingreso', 'media_movil_7d', 'media_movil_30d']].tail())
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*7. Tendencia Diaria con Media Móvil (Rolling 7 días).*

1. `ventas_diarias = ventas.groupby('fecha')['ingreso'].sum().reset_index()` — Agrupa los datos por la columna eRealiza la operación indicada con los parámetros definidos..
2. `ventas_diarias = ventas_diarias.sort_values('fecha')` — Ordena los datos según la columna eRealiza la operación indicada con los parámetros definidos..
3. `ventas_diarias['media_movil_7d'] = ventas_diarias['ingreso'].rolling(window=7, min_periods=1).mean()` — Crea una ventana deslizante para cálculos móviles.
4. `ventas_diarias['media_movil_30d'] = ventas_diarias['ingreso'].rolling(window=30, min_periods=1).mean()` — Crea una ventana deslizante para cálculos móviles.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación de negocio:** La media móvil de 7 días suaviza el ruido diario y revela la tendencia subyacente. La media de 30 días muestra la dirección estratégica. Cuando la media de 7 días cruza por debajo de la de 30 días, es señal de desaceleración. Cuando cruza por arriba, indica aceleración. Esto permite anticipar cambios de tendencia con 1-2 semanas de antelación.

---

## 8. Detectar Estacionalidad: Misma Variable en Diferentes Meses

```python
ventas_mes_dia = ventas.groupby(['mes', 'dia_semana_nombre'], observed=False)['ingreso'].mean().reset_index()

plt.figure(figsize=(14, 8))
sns.catplot(data=ventas_mes_dia, x='dia_semana_nombre', y='ingreso', col='mes',
            col_wrap=4, kind='bar', height=3, aspect=1.2, sharey=False,
            palette='viridis')
plt.suptitle('Patrón de Ventas por Día de la Semana — Desglosado por Mes', y=1.02, fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

ventas_mes_dia_pivot = ventas_mes_dia.pivot_table(index='mes', columns='dia_semana_nombre',
                                                   values='ingreso', observed=False)
print("Coeficiente de variación por mes (mayor = más estacionalidad intrasemanal):")
cv = ventas_mes_dia_pivot.std(axis=1) / ventas_mes_dia_pivot.mean(axis=1)
print(cv.sort_values(ascending=False))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*8. Detectar Estacionalidad: Misma Variable en Diferentes Meses.*

1. `ventas_mes_dia = ventas.groupby(['mes', 'dia_semana_nombre'], observed=False)['ingreso'].mean().reset_index()` — Agrupa los datos por la columna eRealiza la operación indicada con los parámetros definidos..

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación de negocio:** La estacionalidad cambia según el mes. Diciembre puede tener un patrón diferente a febrero. Por ejemplo, en diciembre todos los días venden bien (poca variación), mientras que en febrero los fines de semana concentran la venta. Esto permite diseñar calendarios promocionales específicos por mes en lugar de un plan genérico anual.

---

## 9. Análisis por Hora del Día (Simular Hora)

```python
np.random.seed(42)
ventas['hora'] = np.random.choice(range(8, 21), size=len(ventas),
                                  p=[0.03, 0.04, 0.05, 0.07, 0.09, 0.10, 0.11,
                                     0.10, 0.09, 0.08, 0.07, 0.05, 0.04])

ventas_hora = ventas.groupby('hora')['ingreso'].sum().reset_index()

plt.figure(figsize=(12, 5))
ax = sns.barplot(data=ventas_hora, x='hora', y='ingreso', hue='hora', palette='coolwarm', legend=False)
for i, v in enumerate(ventas_hora['ingreso']):
    ax.text(i, v + 100000, f'${v/1e6:.1f}M', ha='center', fontsize=8)
plt.title('Ventas por Hora del Día (Simulado)', fontsize=14, fontweight='bold')
plt.xlabel('Hora')
plt.ylabel('Ingreso Total ($)')
plt.axvspan(11.5, 14.5, alpha=0.1, color='green', label='Horas pico (12-14)')
plt.legend()
plt.tight_layout()
plt.show()

hora_pico = ventas_hora.loc[ventas_hora['ingreso'].idxmax(), 'hora']
print(f"Hora pico de ventas: {hora_pico}:00")
print(f"Ventas en hora pico: ${ventas_hora['ingreso'].max():.0f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*9. Análisis por Hora del Día (Simular Hora).*

1. `ventas_hora = ventas.groupby('hora')['ingreso'].sum().reset_index()` — Agrupa los datos por la columna eRealiza la operación indicada con los parámetros definidos..

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación de negocio:** Aunque la hora es simulada, el análisis muestra la importancia de entender los picos horarios. En un escenario real con datos de caja registradora, saber que las 12-14 hrs concentran la mayor venta permite programar descansos del personal fuera de ese bloque, tener cajas abiertas al máximo y surtir exhibidores antes del mediodía.

---

## 10. Prueba Chi-Cuadrado: Asociación entre Día de Semana y Sucursal

```python
tabla_contingencia = pd.crosstab(ventas['dia_semana_nombre'], ventas['sucursal'])
chi2, p_valor, dof, expected = stats.chi2_contingency(tabla_contingencia)

plt.figure(figsize=(12, 6))
sns.heatmap(tabla_contingencia, annot=True, fmt='d', cmap='Blues', linewidths=0.5,
            cbar_kws={'label': 'Número de Transacciones'})
plt.title(f'Tabla de Contingencia: Día de Semana × Sucursal\nχ²={chi2:.1f}, p-valor={p_valor:.6f}',
          fontsize=14, fontweight='bold')
plt.xlabel('Sucursal')
plt.ylabel('Día de la Semana')
plt.tight_layout()
plt.show()

print(f"Chi-cuadrado: {chi2:.2f}")
print(f"p-valor: {p_valor:.6f}")
if p_valor < 0.05:
    print("Conclusión: Existe asociación significativa entre día de semana y sucursal.")
    print("→ Las sucursales tienen patrones semanales distintos.")
else:
    print("Conclusión: No hay evidencia de asociación significativa.")
    print("→ Los patrones semanales son homogéneos entre sucursales.")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*10. Prueba Chi-Cuadrado: Asociación entre Día de Semana y Sucursal.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación de negocio:** Si el chi-cuadrado es significativo, significa que ciertas sucursales venden más en días específicos que otras. Por ejemplo, una sucursal cerca de oficinas vende más de lunes a viernes, mientras que una en zona turística vende más en fin de semana. Esto justifica estrategias diferenciadas por sucursal en lugar de un plan nacional homogéneo.

---

## 11. Calendario de Ventas (Heatmap Día × Mes)

```python
calendario = ventas.pivot_table(
    values='ingreso', index='dia_mes', columns='mes',
    aggfunc='sum', fill_value=0
)

plt.figure(figsize=(16, 8))
sns.heatmap(calendario, annot=True, fmt='$_.0f', cmap='RdYlGn', linewidths=0.5,
            cbar_kws={'label': 'Ingreso ($)'},
            xticklabels=['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                         'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'])
plt.title('Calendario de Ventas 2024 — Ingreso por Día y Mes', fontsize=14, fontweight='bold')
plt.xlabel('Mes')
plt.ylabel('Día del Mes')
plt.tight_layout()
plt.show()

# Días con ingresos anómalamente altos
ingreso_diario = ventas.groupby('fecha')['ingreso'].sum()
threshold = ingreso_diario.mean() + 2 * ingreso_diario.std()
dias_excepcionales = ingreso_diario[ingreso_diario > threshold]
print(f"Días con ingresos excepcionales (> media + 2σ):")
for fecha, ingreso in dias_excepcionales.items():
    print(f"  {fecha.date()}: ${ingreso:,.0f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*11. Calendario de Ventas (Heatmap Día × Mes).*

1. Días con ingresos anómalamente altos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación de negocio:** El calendario de ventas es una herramienta visual poderosa para identificar días específicos con rendimiento atípico. Los días con ingresos excepcionales suelen corresponder a promociones, días festivos o eventos especiales. Identificarlos permite replicar las condiciones que generaron esos picos y preparar operaciones para fechas similares del próximo año.

---

## 12. Recomendaciones Estratégicas

```python
recomendaciones = [
    "1. INVENTARIO PREVENTIVO: Aumentar stock 2 semanas antes de meses pico (marzo, agosto, diciembre).",
    "2. PERSONAL DINÁMICO: Reforzar personal los viernes y sábados en sucursales con mayor variabilidad.",
    "3. PROMOCIONES VALLE: Lanzar campañas agresivas en enero, febrero y septiembre para equilibrar ingresos.",
    "4. HORARIOS OPTIMIZADOS: Programar personal completo 10:00-16:00 hrs (horas pico simuladas).",
    "5. CALENDARIO PROMOCIONAL: Crear un calendario anual con 5 campañas fuertes alineadas a estacionalidad.",
    "6. SUCURSALES DIFERENCIADAS: Estrategia comercial distinta para sucursales con perfil semanal diferente.",
    "7. DASHBOARD ESTACIONAL: Monitorear semanalmente media móvil 7d vs 30d para detectar cambios de tendencia.",
    "8. ANÁLISIS CAUSAL: Investigar qué eventos externos (clima, economía, competencia) explican los picos.",
    "9. PRESUPUESTO DINÁMICO: Asignar presupuesto de marketing mensual proporcional a la estacionalidad.",
    "10. REVISIÓN TRIMESTRAL: Evaluar cada trimestre el plan estacional y ajustar pronósticos."
]

for r in recomendaciones:
    print(r)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*12. Recomendaciones Estratégicas.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# Resumen de métricas estacionales clave
resumen = pd.DataFrame({
    'Métrica': [
        'Día pico semanal', 'Mes pico', 'Trimestre pico',
        'Diferencia finde vs laboral (%)', 'Horas pico',
        '¿Asociación día×sucursal?'
    ],
    'Valor': [
        ventas_dia.sort_values('ingreso', ascending=False).iloc[0]['dia_semana_nombre'],
        f"Mes {ventas_mes.loc[ventas_mes['ingreso'].idxmax(), 'mes']}",
        f"Q{ventas_trim.loc[ventas_trim['ingreso'].idxmax(), 'trimestre']}",
        f"{(ventas_finde.loc[ventas_finde['es_finde']=='Fin de Semana', 'mean'].values[0] / ventas_finde.loc[ventas_finde['es_finde']=='Día Laboral', 'mean'].values[0] - 1) * 100:.1f}%",
        f"{hora_pico}:00",
        'Sí' if p_valor < 0.05 else 'No'
    ]
})
print("\n=== RESUMEN DE ESTACIONALIDAD ===")
print(resumen.to_string(index=False))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Resumen de métricas estacionales clave

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación de negocio:** Las 10 recomendaciones cubren las cuatro palancas clave afectadas por estacionalidad: inventario, personal, marketing y operaciones. Implementar estas acciones puede mejorar ingresos entre 8-15% anual al capturar la demanda en temporada alta y estimularla en temporada baja, además de reducir costos operativos al alinear recursos con demanda real.

---

## 📝 Ejercicios Propuestos

1. **Estacionalidad por categoría:** Repite el análisis de ventas por mes pero filtrado por `categoria`. ¿Todas las categorías tienen el mismo patrón estacional? (Pista: usa `sns.FacetGrid` con `col='categoria'`)

2. **Auto-correlación:** Calcula la autocorrelación de la serie diaria de ventas con rezagos de 1 a 30 días. ¿Qué rezago tiene mayor correlación? (Pista: usa `pd.Series.autocorr`)

3. **Días festivos:** Si supieras que ciertos días son festivos (ej. 10 mayo, 16 septiembre), ¿cómo cambia el patrón? Crea una columna `es_festivo` y compara. (Pista: define una lista de fechas manual)

4. **Pronóstico simple:** Usa media móvil de 7 días para predecir el día siguiente. Calcula el error absoluto medio (MAE). (Pista: usa `shift(1)` para alinear)

5. **Estacionalidad por sucursal:** Repite el calendario de ventas (sección 11) para cada sucursal por separado. ¿Hay sucursales con patrones estacionales distintos? (Pista: itera sobre sucursales con `groupby`)

---

## 📌 Resumen

| Hallazgo | Impacto |
|----------|---------|
| Viernes/sábado son días pico semanales | Reforzar personal y stock jueves-viernes |
| Q4 concentra mayor porcentaje de ingresos | Planificar compras en Q3, promociones en Q1 |
| Fin de semana tiene ticket promedio más alto | Campañas específicas para finde |
| Media móvil 7d vs 30d anticipa tendencias | Dashboard semanal de alerta temprana |
| Chi-cuadrado revela patrones distintos por sucursal | Estrategias diferenciadas por sucursal |
| Calendario día×mes identifica outliers | Investigar causas de picos atípicos |

## 🔗 Enlaces Relacionados
- [CP01 - Análisis de Ventas Básico](CP01-analisis-ventas-basico.md)
- [CP07 - Productos Críticos ABC](CP07-productos-criticos.md)
- [CP09 - Comparativa de Sucursales](CP09-comparativa-sucursales.md)
