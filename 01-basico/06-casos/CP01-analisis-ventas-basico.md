# CP01 — Análisis Exploratorio de Ventas Diarias — Reporte Semanal

## 🎯 Contexto de Negocio

El gerente de ventas de la cadena de tiendas solicita un reporte semanal automatizado que muestre ingresos, márgenes, tendencias por sucursal y categoría, así como detección de outliers. El objetivo es tomar decisiones informadas sobre inventario, promociones y personal.

Los datos provienen de 1330 transacciones reales durante enero-diciembre 2024 en 10 sucursales.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Configuración visual
sns.set_theme(style='whitegrid', palette='Set2')
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['figure.dpi'] = 120

# Carga de datos
ventas = pd.read_csv("../datos/ventas.csv")
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*🎯 Contexto de Negocio.*

1. Configuración visual
2. Carga de datos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 1. Cargar y Explorar Datos

```python
print("Shape:", ventas.shape)
print("\nPrimeras 5 filas:")
print(ventas.head())
print("\nInfo:")
print(ventas.info())
print("\nDescribe:")
print(ventas.describe())
```

**Salida esperada:**


**Salida esperada:**
```
Shape: (1330, 16)

Primeras 5 filas:
       fecha      sku           producto     categoria        sucursal  \
0 2024-01-01  LAP001    Laptop Pro 15   Electrónica  Sucursal Mérida
1 2024-01-01  MON002  Monitor 24 HD     Electrónica  Sucursal Querétaro
2 2024-01-01  TEC001  Teclado Mecánico  Periféricos  Sucursal Mérida
3 2024-01-01  MOU001  Mouse Ergonómico  Periféricos  Sucursal Querétaro
4 2024-01-01  AUD002  Parlante Portátil Audio        Sucursal Toluca

         cliente  cantidad  precio_unitario  costo_unitario   ingreso  \
0  Comercial MX         8          14250.0           12000  114000.0
1  Soluciones Inc       3           3230.0            2500    9690.0
2  Distribuidora ABC   13           1400.0             800   18200.0
3  Empresa Beta         2            650.0             350    1300.0
4  Empresa Beta         9           1500.0             800   13500.0

   costo_total   margen  margen_pct  descuento  dia_semana  mes
0        96000  18000.0        18.8       0.05           0    1
1         7500   2190.0        29.2       0.05           0    1
2        10400   7800.0        75.0       0.00           0    1
3          700    600.0        85.7       0.00           0    1
4         7200   6300.0        87.5       0.00           0    1
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



**Interpretación de negocio:** El dataset contiene 1330 transacciones con 16 columnas. Las variables numéricas clave son `ingreso` (media ~37k), `margen` (media ~18k) y `cantidad`. Hay columnas categóricas como `sucursal`, `categoria`, `producto` que permiten segmentar el análisis. El rango de fegos abarca un año completo, lo que permite análisis temporal robusto.

---

## 2. Limpiar Datos

```python
# Duplicados
dup = ventas.duplicated().sum()
print(f"Filas duplicadas: {dup}")

# Nulos
nulos = ventas.isnull().sum()
print(f"\nValores nulos por columna:\n{nulos[nulos > 0]}")

# Tipos de datos
print(f"\nTipos de dato:\n{ventas.dtypes}")

# Convertir fecha a datetime si no lo está
ventas['fecha'] = pd.to_datetime(ventas['fecha'])

# Verificar rangos lógicos
print(f"\nIngresos negativos: {(ventas['ingreso'] < 0).sum()}")
print(f"Cantidades cero o negativas: {(ventas['cantidad'] <= 0).sum()}")
```

**Salida esperada:**


**Salida esperada:**
```
Filas duplicadas: 0

Valores nulos por columna:
Series([], dtype: int64)

Tipos de dato:
fecha              object
sku                object
producto           object
categoria          object
sucursal           object
cliente            object
cantidad            int64
precio_unitario   float64
costo_unitario    float64
ingreso           float64
costo_total       float64
margen            float64
margen_pct        float64
descuento         float64
dia_semana          int64
mes                 int64

Ingresos negativos: 0
Cantidades cero o negativas: 0
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



**Interpretación de negocio:** Los datos están limpios: sin duplicados, sin nulos y sin valores negativos. Esto indica que el proceso de captura de ventas es confiable. Podemos proceder directamente al análisis sin transformaciones correctivas.

---

## 3. Ingreso Total del Período

```python
ingreso_total = ventas['ingreso'].sum()
costo_total = ventas['costo_total'].sum()
margen_total = ventas['margen'].sum()
margen_pct_global = (margen_total / ingreso_total) * 100

print(f"Ingreso total: ${ingreso_total:,.2f}")
print(f"Costo total: ${costo_total:,.2f}")
print(f"Margen total: ${margen_total:,.2f}")
print(f"Margen % global: {margen_pct_global:.1f}%")
print(f"Número de transacciones: {len(ventas)}")
print(f"Ingreso promedio por transacción: ${ventas['ingreso'].mean():,.2f}")
```

**Salida esperada:**


**Salida esperada:**
```
Ingreso total: $50,239,503.00
Costo total: $32,433,956.00
Margen total: $17,805,547.00
Margen % global: 35.4%
Número de transacciones: 1330
Ingreso promedio por transacción: $37,774.06
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



**Interpretación de negocio:** La cadena generó ~$50.2M en ingresos con un margen del 35.4%. Esto significa que por cada $100 vendidos, $35.4 quedan como ganancia bruta. Es un margen saludable para retail de tecnología.

---

## 4. Ingreso por Categoría

```python
ingreso_cat = ventas.groupby('categoria')['ingreso'].sum().sort_values(ascending=False)
print("Ingreso por categoría:")
print(ingreso_cat)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Barplot
sns.barplot(x=ingreso_cat.values, y=ingreso_cat.index, ax=axes[0], palette='viridis')
axes[0].set_title('Ingreso Total por Categoría')
axes[0].set_xlabel('Ingreso ($)')
# Pie chart
ingreso_cat.plot.pie(ax=axes[1], autopct='%1.1f%%', startangle=90, colors=sns.color_palette('viridis', len(ingreso_cat)))
axes[1].set_ylabel('')
axes[1].set_title('Distribución % de Ingreso por Categoría')
plt.tight_layout()
plt.show()
```

**Salida esperada:** Las categorías Electrónica y Muebles dominan los ingresos, seguidas de Audio y Periféricos. Las categorías con menor contribución son Papelería y Cámaras.

**Interpretación de negocio:** Electrónica genera ~35% del ingreso total. Muebles le sigue con ~20%. Esto sugiere concentrar esfuerzos de marketing y negociación con proveedores en estas categorías. Sin embargo, categorías como Papelería, aunque de bajo ingreso, pueden tener alto margen y merecen análisis separado.

---

## 5. Ingreso por Sucursal



**Salida esperada:** Las categorías Electrónica y Muebles dominan los ingresos, seguidas de Audio y Periféricos. Las categorías con menor contribución son Papelería y Cámaras.

**Interpretación de negocio:** Electrónica genera ~35% del ingreso total. Muebles le sigue con ~20%. Esto sugiere concentrar esfuerzos de marketing y negociación con proveedores en estas categorías. Sin embargo, categorías como Papelería, aunque de bajo ingreso, pueden tener alto margen y merecen análisis separado.

---

## 5. Ingreso por Sucursal

```python
ingreso_suc = ventas.groupby('sucursal')['ingreso'].sum().sort_values(ascending=False)
print("Ingreso por sucursal:")
print(ingreso_suc)

plt.figure(figsize=(12, 5))
sns.barplot(x=ingreso_suc.index, y=ingreso_suc.values, palette='coolwarm')
plt.title('Ingreso Total por Sucursal')
plt.xlabel('Sucursal')
plt.ylabel('Ingreso ($)')
plt.xticks(rotation=45)
plt.show()
```

**Salida esperada:** Las sucursales Matriz CDMX, Sucursal Mérida y Sucursal Tijuana encabezan los ingresos. Las sucursales con menor ingreso son Sucursal Puebla y Sucursal Cancún.

**Interpretación de negocio:** Hay una disparidad significativa entre sucursales. Matriz CDMX genera ~3x más que la sucursal de menor rendimiento. Esto puede deberse a ubicación, tamaño de tienda o afluencia de clientes. Se recomienda investigar prácticas de la sucursal top para replicarlas.

---

## 6. Margen Promedio por Producto



**Salida esperada:** Las sucursales Matriz CDMX, Sucursal Mérida y Sucursal Tijuana encabezan los ingresos. Las sucursales con menor ingreso son Sucursal Puebla y Sucursal Cancún.

**Interpretación de negocio:** Hay una disparidad significativa entre sucursales. Matriz CDMX genera ~3x más que la sucursal de menor rendimiento. Esto puede deberse a ubicación, tamaño de tienda o afluencia de clientes. Se recomienda investigar prácticas de la sucursal top para replicarlas.

---

## 6. Margen Promedio por Producto

```python
margen_prod = ventas.groupby('producto').agg(
    margen_promedio=('margen_pct', 'mean'),
    ingresos_totales=('ingreso', 'sum'),
    transacciones=('ingreso', 'count')
).sort_values('margen_promedio', ascending=False)

print("Top 10 productos por margen % promedio:")
print(margen_prod.head(10))
```

**Salida esperada:**


**Salida esperada:**
```
                    margen_promedio  ingresos_totales  transacciones
USB 64GB                    98.5            48500             40
Mouse Ergonómico            85.7            23500             35
Parlante Portátil           83.2            67800             52
Papel Bond 5000 hojas       81.0            32000             28
...
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



**Interpretación de negocio:** Productos pequeños como USB y mouse tienen márgenes altos (~85-98%) aunque bajo valor absoluto. Electrónica tiene márgenes más bajos (~18-25%) pero alto volumen. Estrategia: promocionar accesorios de alto margen como upselling en cada venta de laptop.

---

## 7. Top 10 Productos Más Vendidos

```python
top_qty = ventas.groupby('producto')['cantidad'].sum().sort_values(ascending=False).head(10)
print("Top 10 productos más vendidos (por cantidad):")
print(top_qty)

plt.figure(figsize=(10, 5))
sns.barplot(x=top_qty.values, y=top_qty.index, palette='mako')
plt.title('Top 10 Productos Más Vendidos por Cantidad')
plt.xlabel('Cantidad Total Vendida')
plt.show()
```

**Salida esperada:** Los productos más vendidos en volumen son accesorios: USB, mouse, papel bond, teclados. Las laptops aparecen pero con menor frecuencia.

**Interpretación de negocio:** Los productos de bajo costo unitario dominan en volumen. Sin embargo, su contribución al ingreso total es menor que la de electrónica. Esto es típico en retail: la regla 80/20 aplica (20% de productos generan 80% del ingreso).

---

## 8. Distribución de Precios



**Salida esperada:** Los productos más vendidos en volumen son accesorios: USB, mouse, papel bond, teclados. Las laptops aparecen pero con menor frecuencia.

**Interpretación de negocio:** Los productos de bajo costo unitario dominan en volumen. Sin embargo, su contribución al ingreso total es menor que la de electrónica. Esto es típico en retail: la regla 80/20 aplica (20% de productos generan 80% del ingreso).

---

## 8. Distribución de Precios

```python
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(ventas['precio_unitario'], bins=50, kde=True, color='steelblue')
plt.title('Distribución de Precios Unitarios')
plt.xlabel('Precio ($)')
plt.ylabel('Frecuencia')

plt.subplot(1, 2, 2)
sns.boxplot(x=ventas['precio_unitario'], color='coral')
plt.title('Boxplot de Precios')
plt.xlabel('Precio ($)')

plt.tight_layout()
plt.show()

print(f"Precio promedio: ${ventas['precio_unitario'].mean():.2f}")
print(f"Precio mediano: ${ventas['precio_unitario'].median():.2f}")
print(f"Precio min: ${ventas['precio_unitario'].min():.2f}")
print(f"Precio max: ${ventas['precio_unitario'].max():.2f}")
```

**Salida esperada:** La distribución es asimétrica positiva (sesgada a la derecha). La mayoría de productos están entre $300 y $3,000, con una cola larga hasta $15,000 (laptops). El precio mediano (~$1,500) es más representativo que la media (~$2,800).

**Interpretación de negocio:** La cadena vende principalmente productos de gama media. Los pocos productos de alto precio (laptops, monitores 4K, escritorios) generan picos en la distribución. Para campañas, conviene segmentar por rangos de precio.

---

## 9. Tendencia Diaria de Ingresos



**Salida esperada:** La distribución es asimétrica positiva (sesgada a la derecha). La mayoría de productos están entre $300 y $3,000, con una cola larga hasta $15,000 (laptops). El precio mediano (~$1,500) es más representativo que la media (~$2,800).

**Interpretación de negocio:** La cadena vende principalmente productos de gama media. Los pocos productos de alto precio (laptops, monitores 4K, escritorios) generan picos en la distribución. Para campañas, conviene segmentar por rangos de precio.

---

## 9. Tendencia Diaria de Ingresos

```python
ingreso_diario = ventas.groupby('fecha')['ingreso'].sum()

plt.figure(figsize=(14, 5))
sns.lineplot(x=ingreso_diario.index, y=ingreso_diario.values, color='darkgreen', linewidth=1.5)
plt.title('Tendencia Diaria de Ingresos')
plt.xlabel('Fecha')
plt.ylabel('Ingreso ($)')
plt.grid(True, alpha=0.3)

# Media móvil de 7 días
media_movil = ingreso_diario.rolling(window=7).mean()
sns.lineplot(x=media_movil.index, y=media_movil.values, color='red', linewidth=2, label='Media móvil 7d')
plt.legend()
plt.show()
```

**Salida esperada:** La serie muestra fluctuaciones diarias con una media ~$165k. Se observan picos periódicos posiblemente asociados a días de quincena o promociones. La media móvil suaviza la tendencia mostrando estabilidad general.

**Interpretación de negocio:** Los ingresos diarios son estables pero con picos marcados. Identificar las fechas de esos picos (ej. quincena, fin de mes) permite planificar promociones y dotación de personal.

---

## 10. Tendencia por Sucursal (con hue)



**Salida esperada:** La serie muestra fluctuaciones diarias con una media ~$165k. Se observan picos periódicos posiblemente asociados a días de quincena o promociones. La media móvil suaviza la tendencia mostrando estabilidad general.

**Interpretación de negocio:** Los ingresos diarios son estables pero con picos marcados. Identificar las fechas de esos picos (ej. quincena, fin de mes) permite planificar promociones y dotación de personal.

---

## 10. Tendencia por Sucursal (con hue)

```python
ingreso_diario_suc = ventas.groupby(['fecha', 'sucursal'])['ingreso'].sum().reset_index()

plt.figure(figsize=(14, 6))
sns.lineplot(data=ingreso_diario_suc, x='fecha', y='ingreso', hue='sucursal', linewidth=1, alpha=0.7)
plt.title('Tendencia Diaria de Ingresos por Sucursal')
plt.xlabel('Fecha')
plt.ylabel('Ingreso ($)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.show()
```

**Salida esperada:** Se ven 10 líneas de colores, una por sucursal. Matriz CDMX muestra los valores más altos consistentemente. Algunas sucursales tienen patrones estacionales sincronizados.

**Interpretación de negocio:** Todas las sucursales siguen un patrón similar pero con escalas distintas. Esto sugiere que factores externos (estacionalidad, economía) afectan a todas por igual, pero la diferencia en magnitud se debe a factores locales (ubicación, competencia).

---

## 11. Día de la Semana con Más Ventas



**Salida esperada:** Se ven 10 líneas de colores, una por sucursal. Matriz CDMX muestra los valores más altos consistentemente. Algunas sucursales tienen patrones estacionales sincronizados.

**Interpretación de negocio:** Todas las sucursales siguen un patrón similar pero con escalas distintas. Esto sugiere que factores externos (estacionalidad, economía) afectan a todas por igual, pero la diferencia en magnitud se debe a factores locales (ubicación, competencia).

---

## 11. Día de la Semana con Más Ventas

```python
dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
ventas['dia_nombre'] = ventas['dia_semana'].map({i: d for i, d in enumerate(dias)})

ingreso_dia = ventas.groupby('dia_nombre')['ingreso'].sum()
ingreso_dia = ingreso_dia.reindex(dias)

plt.figure(figsize=(10, 5))
ax = sns.barplot(x=ingreso_dia.index, y=ingreso_dia.values, palette='Blues_d')
plt.title('Ingreso Total por Día de la Semana')
plt.xlabel('Día')
plt.ylabel('Ingreso ($)')
for i, v in enumerate(ingreso_dia.values):
    ax.text(i, v + 500, f'${v:,.0f}', ha='center', fontsize=9)
plt.show()

conteo_dia = ventas.groupby('dia_nombre')['ingreso'].count().reindex(dias)
print("Transacciones por día:")
print(conteo_dia)
```

**Salida esperada:** Los viernes y lunes tienen los mayores ingresos. Los domingos y miércoles los menores. Esto refleja el comportamiento de compra B2B (días laborales).

**Interpretación de negocio:** Los clientes compran más a inicios (lunes) y finales (viernes) de semana. Sugerencia: programar lanzamientos de productos para lunes y promociones flash para viernes. Los domingos podrían usarse para mantenimiento de sistemas.

---

## 12. Correlación Precio-Cantidad



**Salida esperada:** Los viernes y lunes tienen los mayores ingresos. Los domingos y miércoles los menores. Esto refleja el comportamiento de compra B2B (días laborales).

**Interpretación de negocio:** Los clientes compran más a inicios (lunes) y finales (viernes) de semana. Sugerencia: programar lanzamientos de productos para lunes y promociones flash para viernes. Los domingos podrían usarse para mantenimiento de sistemas.

---

## 12. Correlación Precio-Cantidad

```python
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(data=ventas, x='precio_unitario', y='cantidad', alpha=0.4, hue='categoria')
plt.title('Precio vs Cantidad Vendida')
plt.xlabel('Precio Unitario ($)')
plt.ylabel('Cantidad')

plt.subplot(1, 2, 2)
sns.lmplot(data=ventas, x='precio_unitario', y='cantidad', scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
plt.title('Regresión Lineal: Precio vs Cantidad')
plt.xlabel('Precio Unitario ($)')
plt.ylabel('Cantidad')
plt.show()

corr = ventas[['precio_unitario', 'cantidad', 'ingreso', 'margen', 'descuento']].corr()
print("Matriz de correlación:")
print(corr)
```

**Salida esperada:** La correlación precio-cantidad es negativa (-0.35): a mayor precio, menor cantidad. La correlación descuento-cantidad es positiva (+0.28). La categoría colorea los clusters: electrónica (alto precio, baja cantidad) vs papelería (bajo precio, alta cantidad).

**Interpretación de negocio:** Existe una relación inversa esperada entre precio y cantidad. Sin embargo, el descuento estimula la cantidad vendida. Esto valida que las promociones selectivas pueden aumentar volumen sin erosionar margen si se aplican a productos de alto margen.

---

## 13. Detectar Outliers en Ingresos



**Salida esperada:** La correlación precio-cantidad es negativa (-0.35): a mayor precio, menor cantidad. La correlación descuento-cantidad es positiva (+0.28). La categoría colorea los clusters: electrónica (alto precio, baja cantidad) vs papelería (bajo precio, alta cantidad).

**Interpretación de negocio:** Existe una relación inversa esperada entre precio y cantidad. Sin embargo, el descuento estimula la cantidad vendida. Esto valida que las promociones selectivas pueden aumentar volumen sin erosionar margen si se aplican a productos de alto margen.

---

## 13. Detectar Outliers en Ingresos

```python
Q1 = ventas['ingreso'].quantile(0.25)
Q3 = ventas['ingreso'].quantile(0.75)
IQR = Q3 - Q1
limite_inf = Q1 - 1.5 * IQR
limite_sup = Q3 + 1.5 * IQR

outliers_iqr = ventas[(ventas['ingreso'] < limite_inf) | (ventas['ingreso'] > limite_sup)]
print(f"Outliers por IQR: {len(outliers_iqr)} ({len(outliers_iqr)/len(ventas)*100:.1f}%)")

z_scores = np.abs((ventas['ingreso'] - ventas['ingreso'].mean()) / ventas['ingreso'].std())
outliers_z = ventas[z_scores > 3]
print(f"Outliers por Z-score (|z|>3): {len(outliers_z)} ({len(outliers_z)/len(ventas)*100:.1f}%)")

plt.figure(figsize=(10, 4))
sns.boxplot(x=ventas['ingreso'], color='tomato')
plt.title('Boxplot de Ingresos — Outliers')
plt.xlabel('Ingreso ($)')
plt.show()
```

**Salida esperada:** Se detectan ~15-25 outliers (1-2% de los datos). Son transacciones de alto valor (laptops vendidas en lote, escritorios eléctricos) que, aunque legítimas, distorsionan los promedios.

**Interpretación de negocio:** Los outliers corresponden a ventas corporativas de alto volumen. No deben eliminarse (son transacciones reales) pero sí identificarse para reportes separados. Se recomienda reportar mediana en lugar de media para métricas de tendencia central.

---

## 14. Resumen Ejecutivo — Tabla de Métricas Clave



**Salida esperada:** Se detectan ~15-25 outliers (1-2% de los datos). Son transacciones de alto valor (laptops vendidas en lote, escritorios eléctricos) que, aunque legítimas, distorsionan los promedios.

**Interpretación de negocio:** Los outliers corresponden a ventas corporativas de alto volumen. No deben eliminarse (son transacciones reales) pero sí identificarse para reportes separados. Se recomienda reportar mediana en lugar de media para métricas de tendencia central.

---

## 14. Resumen Ejecutivo — Tabla de Métricas Clave

```python
resumen = pd.DataFrame({
    'Métrica': [
        'Ingreso Total', 'Costo Total', 'Margen Total', 'Margen %',
        'Transacciones', 'Ticket Promedio', 'Ingreso Promedio Diario',
        'Producto Estrella (ingreso)', 'Sucursal Estrella', 'Día Pico'
    ],
    'Valor': [
        f'${ingreso_total:,.0f}',
        f'${costo_total:,.0f}',
        f'${margen_total:,.0f}',
        f'{margen_pct_global:.1f}%',
        f'{len(ventas):,}',
        f'${ventas["ingreso"].mean():,.2f}',
        f'${ingreso_diario.mean():,.0f}',
        ingreso_cat.index[0],
        ingreso_suc.index[0],
        ingreso_dia.idxmax()
    ]
})
print("=== RESUMEN EJECUTIVO ===")
print(resumen.to_string(index=False))
```

**Salida esperada:**


**Salida esperada:**
```
=== RESUMEN EJECUTIVO ===
                   Métrica          Valor
             Ingreso Total    $50,239,503
               Costo Total    $32,433,956
              Margen Total    $17,805,547
                 Margen %          35.4%
            Transacciones          1,330
          Ticket Promedio    $37,774.06
 Ingreso Promedio Diario      $165,435
   Producto Estrella (ing)     Electrónica
     Sucursal Estrella       Matriz CDMX
                 Día Pico           Viernes
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



**Interpretación de negocio:** Esta tabla consolida los hallazgos clave en un solo vistazo. El ticket promedio de $37.7k sugiere que es mayorista/B2B. Electrónica es la categoría reina. Matriz CDMX lidera.

---

## 15. Recomendaciones de Negocio

Basado en el análisis completo:

```python
recomendaciones = [
    "1. ENFOQUE EN ELECTRÓNICA: Concentrar promociones y negociación con proveedores en laptops y monitores (35% del ingreso).",
    "2. PROGRAMA DE MEJORES PRÁCTICAS: Documentar operación de Matriz CDMX y replicar en sucursales de bajo rendimiento.",
    "3. UPSELLING DE ACCESORIOS: Capacitar vendedores para ofrecer USB/mouse/teclados (alto margen) en cada venta de laptop.",
    "4. PROMOCIONES PROGRAMADAS: Lanzar campañas los viernes (día pico) y ofertas flash los miércoles (día valle).",
    "5. SEGMENTAR REPORTES: Reportar mediana de ingresos (no media) por sucursal para evitar distorsión por outliers.",
    "6. ANÁLISIS DE ESTACIONALIDAD: Profundizar en tendencias mensuales para planificar inventario antes de temporadas altas.",
    "7. DASHBOARD EN TIEMPO REAL: Automatizar este reporte con actualización diaria conectada al sistema transaccional."
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

*15. Recomendaciones de Negocio.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación de negocio:** Siete acciones concretas emergen del análisis. La más impactante a corto plazo es el programa de mejores prácticas entre sucursales, que no requiere inversión y puede aumentar ingresos entre 10-15%.

---

## 📝 Ejercicios Propuestos

1. **Análisis por cliente:** Agrupa por `cliente` y encuentra el top 5 por ingreso total. ¿Qué concentración existe? (Pista: usa `groupby` + `sum` + `sort_values`)

2. **Correlación descuento-margen:** Calcula si los descuentos altos reducen significativamente el margen porcentual. Haz un scatterplot con `descuento` vs `margen_pct`. (Pista: usa `sns.scatterplot`)

3. **Estacionalidad mensual:** Crea un `barplot` de ingreso por mes. ¿Qué meses tienen mayor y menor venta? (Pista: agrupa por la columna `mes`)

4. **Productos con margen negativo:** Filtra productos donde `margen_pct` sea menor a 0. ¿Existen? ¿Qué los causa? (Pista: usa `ventas[ventas['margen_pct'] < 0]`)

5. **Segmentación RFM simple:** Usa `clientes.csv`, haz un merge con ventas y calcula recencia, frecuencia y monto monetario por cliente. (Pista: usa `pd.merge` y `groupby` con `nunique` y `sum`)

---

## 📌 Resumen

| Hallazgo | Impacto |
|----------|---------|
| Ingreso total: $50.2M con margen 35.4% | Negocio rentable con margen saludable |
| Electrónica domina (35% ingresos) | Priorizar abasto y promociones en esa categoría |
| Disparidad 3x entre sucursales | Oportunidad de mejora de 200% en sucursales bajas |
| Outliers son ventas corporativas | No eliminar, reportar mediana |
| Viernes es el día pico | Programar promociones y personal |
| Accesorios tienen margen >80% | Estrategia de upselling obligatoria |

## 🔗 Enlaces Relacionados
- [CP02 - Dashboard de Inventario](CP02-analisis-inventario-basico.md)
- [CP04 - Detección de Outliers](CP04-deteccion-outliers.md)
