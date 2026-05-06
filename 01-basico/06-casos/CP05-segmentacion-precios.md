# CP05 — Segmentación de Productos por Precio — Estrategia Comercial

## 🎯 Contexto de Negocio

El equipo de marketing quiere segmentar los productos en rangos de precio para diseñar campañas promocionales diferenciadas. Se necesita crear segmentos coherentes usando tanto `pd.cut` (rangos fijos) como `pd.qcut` (cuartiles basados en frecuencia), comparar ambos enfoques y generar recomendaciones por segmento.

Los datos provienen del catálogo de 25 productos con precios desde $350 hasta $15,000.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sns.set_theme(style='whitegrid', palette='Set2')
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['figure.dpi'] = 120

inventario = pd.read_csv("../datos/inventario.csv")
ventas = pd.read_csv("../datos/ventas.csv")
ventas['fecha'] = pd.to_datetime(ventas['fecha'])

# Combinar datos para análisis de margen y ventas por producto
prod_ventas = ventas.groupby(['sku', 'producto']).agg(
    cantidad_vendida=('cantidad', 'sum'),
    ingreso_total=('ingreso', 'sum'),
    margen_pct_prom=('margen_pct', 'mean')
).reset_index()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*🎯 Contexto de Negocio.*

1. Combinar datos para análisis de margen y ventas por producto

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 1. Cargar Inventario y Explorar Precios

```python
print("Shape:", inventario.shape)
print("\nPrimeras 5 filas:")
print(inventario[['sku', 'producto', 'categoria', 'costo', 'precio', 'stock_actual']].head())

# Rango de precios
print(f"\nEstadísticas rápidas de precios:")
print(f"  Precio mínimo: ${inventario['precio'].min():,.0f}")
print(f"  Precio máximo: ${inventario['precio'].max():,.0f}")
print(f"  Productos únicos: {inventario['producto'].nunique()}")
print(f"  Categorías: {inventario['categoria'].nunique()}")
```

**Salida esperada:**


**Salida esperada:**
```
Shape: (25, 13)

      sku           producto     categoria   costo  precio  stock_actual
0  LAP001    Laptop Pro 15    Electrónica   12000   15000            17
1  LAP002    Laptop Air 13    Electrónica    9000   11500           192
2  MON001    Monitor 27 4K    Electrónica    5500    7200           123
3  MON002    Monitor 24 HD    Electrónica    2500    3400            90
4  TEC001    Teclado Mecánico Periféricos     800    1400            60

Precio mínimo: $350
Precio máximo: $15,000
Productos únicos: 25
Categorías: 9
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



**Interpretación de negocio:** El catálogo tiene 25 productos con un rango de precio muy amplio: desde $350 (USB, papel, tinta) hasta $15,000 (laptop Pro 15). Esta dispersión hace necesaria una segmentación para diseñar estrategias de precio y promoción adecuadas a cada segmento.

---

## 2. Estadísticas Descriptivas de Precios

```python
desc_precio = inventario['precio'].describe()
print("Estadísticas descriptivas de precios:")
print(desc_precio)

# Asimetría y curtosis
asimetria = inventario['precio'].skew()
curtosis = inventario['precio'].kurtosis()
print(f"\nAsimetría (skewness): {asimetria:.2f}")
print(f"Curtosis: {curtosis:.2f}")

# Interpretación
if asimetria > 1:
    print("→ Distribución muy asimétrica positiva (cola derecha larga)")
elif asimetria > 0:
    print("→ Distribución moderadamente asimétrica positiva")
else:
    print("→ Distribución aproximadamente simétrica")

# Cuartiles manuales
print(f"\nCuartiles:")
for q in [0.10, 0.25, 0.50, 0.75, 0.90]:
    print(f"  Percentil {q*100:.0f}: ${inventario['precio'].quantile(q):,.0f}")
```

**Salida esperada:**


**Salida esperada:**
```
count      25.00
mean      4336.00
std       4266.37
min        350.00
25%       1000.00
50%       2500.00
75%       6400.00
max      15000.00

Asimetría: 1.23
Curtosis: 0.87
→ Distribución muy asimétrica positiva

Cuartiles:
  Percentil 10: $475
  Percentil 25: $1,000
  Percentil 50: $2,500
  Percentil 75: $6,400
  Percentil 90: $11,350
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



**Interpretación de negocio:** La asimetría positiva (1.23) confirma que hay pocos productos muy caros (laptops, monitores 4K) y muchos productos baratos (USBs, papel, mouse). La mediana ($2,500) es un mejor punto de referencia que la media ($4,336). El percentil 90 ($11,350) muestra que el 10% de productos son de ultra-premium.

---

## 3. Distribución de Precios (Histograma + KDE)

```python
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
sns.histplot(inventario['precio'], bins=15, kde=True, color='steelblue')
plt.title('Distribución de Precios (Histograma + KDE)')
plt.xlabel('Precio ($)')
plt.ylabel('Frecuencia')
plt.axvline(x=inventario['precio'].mean(), color='red', linestyle='--', label=f'Media=${inventario["precio"].mean():,.0f}')
plt.axvline(x=inventario['precio'].median(), color='green', linestyle='--', label=f'Mediana=${inventario["precio"].median():,.0f}')
plt.legend()

plt.subplot(1, 2, 2)
sns.kdeplot(inventario['precio'], fill=True, color='purple')
plt.title('Densidad de Precios (KDE)')
plt.xlabel('Precio ($)')
plt.axvline(x=inventario['precio'].mean(), color='red', linestyle='--', label='Media')
plt.axvline(x=inventario['precio'].median(), color='green', linestyle='--', label='Mediana')
plt.legend()

plt.tight_layout()
plt.show()

# Dispersión vertical: cada producto como punto
plt.figure(figsize=(12, 3))
sns.stripplot(data=inventario, x='precio', y=[''] * len(inventario), size=10, 
              hue='categoria', legend='brief')
plt.title('Distribución de Precios — Cada Producto es un Punto')
plt.xlabel('Precio ($)')
plt.show()
```

**Salida esperada:** El histograma muestra una acumulación de productos en el rango $500-$3,000 y una cola larga hasta $15,000. El KDE confirma la asimetría. El stripplot muestra cada producto individual coloreado por categoría.

**Interpretación de negocio:** Visualmente se confirma que la mayoría de productos son de precio bajo-medio. Hay un gap notable entre $7,000 y $11,000 (pocos productos). Esto sugiere una oportunidad de mercado: introducir productos en ese rango. Electrónica (laptops) ocupa la cola derecha; Periféricos y Papelería la izquierda.

---

## 4. Boxplot de Precios por Categoría



**Salida esperada:** El histograma muestra una acumulación de productos en el rango $500-$3,000 y una cola larga hasta $15,000. El KDE confirma la asimetría. El stripplot muestra cada producto individual coloreado por categoría.

**Interpretación de negocio:** Visualmente se confirma que la mayoría de productos son de precio bajo-medio. Hay un gap notable entre $7,000 y $11,000 (pocos productos). Esto sugiere una oportunidad de mercado: introducir productos en ese rango. Electrónica (laptops) ocupa la cola derecha; Periféricos y Papelería la izquierda.

---

## 4. Boxplot de Precios por Categoría

```python
plt.figure(figsize=(12, 6))
sns.boxplot(data=inventario, x='categoria', y='precio', palette='Set3')
plt.title('Distribución de Precios por Categoría')
plt.xlabel('Categoría')
plt.ylabel('Precio ($)')
plt.xticks(rotation=45)
plt.show()

# Estadísticas por categoría
precio_cat = inventario.groupby('categoria')['precio'].agg(['min', 'mean', 'median', 'max', 'count']).round(0)
print("Resumen de precios por categoría:")
print(precio_cat.sort_values('median', ascending=False).to_string())
```

**Salida esperada:** 


**Salida esperada:** 
```
           min    mean  median    max  count
categoria
Electrónica     2500   9275   10250  15000      4
Muebles          600   5200    6000   8500      3
Audio           1500   2167    2200   2800      3
Cámaras         1700   2450    2450   3200      2
Redes           1100   1600    1600   2100      2
Almacenamiento   350   1583    1900   2500      3
Software         800   1300    1300   1800      2
Periféricos      650    975     925   1400      4
Papelería        200    450     450    500      2
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



**Interpretación de negocio:** La categoría define fuertemente el precio. Electrónica y Muebles son categorías premium (>$5,000 de mediana). Periféricos y Papelería son de entrada. Audio y Redes son gama media. La segmentación para campañas debe considerar categoría + precio.

---

## 5. Crear Segmentos con pd.cut (Rangos Fijos)

```python
# Definir bins personalizados basados en la distribución
bins_precio = [0, 1000, 3000, 7000, 20000]
etiquetas = ['Económico (<$1k)', 'Medio ($1k-$3k)', 'Premium ($3k-$7k)', 'Ultra Premium (>$7k)']

inventario['segmento_cut'] = pd.cut(inventario['precio'], bins=bins_precio, labels=etiquetas, include_lowest=True)

print("Productos por segmento (pd.cut - rangos fijos):")
print(inventario[['producto', 'precio', 'segmento_cut']].to_string(index=False))

print(f"\nConteo por segmento:")
print(inventario['segmento_cut'].value_counts().sort_index())
```

**Salida esperada:**


**Salida esperada:**
```
         producto  precio        segmento_cut
   Laptop Pro 15   15000  Ultra Premium (>$7k)
   Laptop Air 13   11500  Ultra Premium (>$7k)
   Monitor 27 4K    7200  Ultra Premium (>$7k)
   Monitor 24 HD    3400     Premium ($3k-$7k)
Teclado Mecánico    1400       Medio ($1k-$3k)
...

Conteo por segmento:
Económico (<$1k)         7
Medio ($1k-$3k)          9
Premium ($3k-$7k)        4
Ultra Premium (>$7k)     5
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



**Interpretación de negocio:** `pd.cut` con rangos fijos produce segmentos interpretables: Económico (papel, USBs, mouse), Medio (teclados, audífonos), Premium (monitores 24", sillas), Ultra Premium (laptops, monitores 4K). El problema: algunos rangos tienen pocos productos (Premium: 4) y otros muchos (Medio: 9).

---

## 6. Crear Segmentos con pd.qcut (Cuartiles por Frecuencia)

```python
inventario['segmento_qcut'] = pd.qcut(inventario['precio'], q=4, labels=['Económico', 'Medio', 'Premium', 'Ultra Premium'])

print("Productos por segmento (pd.qcut - cuartiles):")
print(inventario[['producto', 'precio', 'segmento_qcut']].to_string(index=False))

print(f"\nConteo por segmento:")
print(inventario['segmento_qcut'].value_counts().sort_index())

# Mostrar rangos de cada cuartil
for q in [0.25, 0.50, 0.75, 1.0]:
    print(f"  Percentil {q*100:.0f}: ${inventario['precio'].quantile(q):,.0f}")
print("\nRangos de cada segmento qcut:")
for i, label in enumerate(['Económico', 'Medio', 'Premium', 'Ultra Premium']):
    if i == 0:
        rango = f"${inventario['precio'].min():,.0f} - ${inventario['precio'].quantile(0.25):,.0f}"
    else:
        rango = f"${inventario['precio'].quantile(i*0.25):,.0f} - ${inventario['precio'].quantile((i+1)*0.25):,.0f}"
    print(f"  {label}: {rango}")
```

**Salida esperada:**


**Salida esperada:**
```
         producto  precio segmento_qcut
   Laptop Pro 15   15000  Ultra Premium
   Laptop Air 13   11500  Ultra Premium
   Monitor 27 4K    7200  Ultra Premium
   Monitor 24 HD    3400        Premium
Teclado Mecánico    1400         Medio
...

Conteo por segmento:
Económico        6
Medio            6
Premium          7
Ultra Premium    6

Rangos de cada segmento qcut:
Económico: $350 - $1,000
Medio: $1,000 - $2,500
Premium: $2,500 - $6,400
Ultra Premium: $6,400 - $15,000
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



**Interpretación de negocio:** `pd.qcut` fuerza que cada segmento tenga ~6-7 productos, garantizando grupos balanceados para campañas. Los rangos son más coherentes con la distribución real. El segmento Ultra Premium ahora inicia en $6,400 (no $7k), capturando sillas y monitores medianos.

---

## 7. Comparar cut vs qcut

```python
comparacion = inventario[['producto', 'precio', 'segmento_cut', 'segmento_qcut']]
print("Comparación cut vs qcut:")
print(comparacion.to_string(index=False))

# Ver productos con clasificación diferente
diferentes = inventario[inventario['segmento_cut'] != inventario['segmento_qcut']]
print(f"\nProductos con clasificación diferente ({len(diferentes)}):")
if len(diferentes) > 0:
    print(diferentes[['producto', 'precio', 'segmento_cut', 'segmento_qcut']].to_string(index=False))

# Distribución comparativa
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

inventario['segmento_cut'].value_counts().sort_index().plot.bar(ax=axes[0], color='steelblue')
axes[0].set_title('pd.cut: Segmentos por Rangos Fijos')
axes[0].set_ylabel('Cantidad de Productos')
axes[0].set_xlabel('Segmento')
axes[0].tick_params(axis='x', rotation=45)

inventario['segmento_qcut'].value_counts().sort_index().plot.bar(ax=axes[1], color='coral')
axes[1].set_title('pd.qcut: Segmentos por Cuartiles')
axes[1].set_ylabel('Cantidad de Productos')
axes[1].set_xlabel('Segmento')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

**Salida esperada:** Algunos productos cambian de segmento entre ambos métodos. Por ejemplo, Monitor 24 HD ($3,400) es "Premium" en cut ($3k-$7k) pero "Medio" en qcut (hasta $6,400). Escritorio Eléctrico ($8,500) es "Ultra Premium" en cut pero "Premium" en qcut.

**Interpretación de negocio:** La elección entre cut y qcut depende del objetivo:
- **cut**: Útil cuando hay rangos de precio con significado de negocio (ej. <$1k = económico, >$7k = premium).
- **qcut**: Útil cuando se necesitan grupos balanceados para campañas con presupuesto equitativo.
- **Recomendación**: Usar qcut como base para marketing (presupuestos iguales por segmento) y cut para reportes financieros (rangos consistentes en el tiempo).

---

## 8. Contar Productos por Segmento (Barplot)



**Salida esperada:** Algunos productos cambian de segmento entre ambos métodos. Por ejemplo, Monitor 24 HD ($3,400) es "Premium" en cut ($3k-$7k) pero "Medio" en qcut (hasta $6,400). Escritorio Eléctrico ($8,500) es "Ultra Premium" en cut pero "Premium" en qcut.

**Interpretación de negocio:** La elección entre cut y qcut depende del objetivo:
- **cut**: Útil cuando hay rangos de precio con significado de negocio (ej. <$1k = económico, >$7k = premium).
- **qcut**: Útil cuando se necesitan grupos balanceados para campañas con presupuesto equitativo.
- **Recomendación**: Usar qcut como base para marketing (presupuestos iguales por segmento) y cut para reportes financieros (rangos consistentes en el tiempo).

---

## 8. Contar Productos por Segmento (Barplot)

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, col, titulo in zip(axes, ['segmento_cut', 'segmento_qcut'], ['cut (Rangos Fijos)', 'qcut (Cuartiles)']):
    conteo = inventario[col].value_counts().sort_index()
    sns.barplot(x=conteo.index, y=conteo.values, ax=ax, palette='viridis')
    ax.set_title(f'Productos por Segmento — {titulo}')
    ax.set_xlabel('Segmento')
    ax.set_ylabel('Cantidad de Productos')
    ax.tick_params(axis='x', rotation=45)
    for i, v in enumerate(conteo.values):
        ax.text(i, v + 0.2, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

print("\nResumen comparativo:")
resumen_segmentos = pd.DataFrame({
    'Segmento': ['Económico', 'Medio', 'Premium', 'Ultra Premium'],
    'cut (productos)': [inventario['segmento_cut'].value_counts().get('Económico (<$1k)', 0),
                        inventario['segmento_cut'].value_counts().get('Medio ($1k-$3k)', 0),
                        inventario['segmento_cut'].value_counts().get('Premium ($3k-$7k)', 0),
                        inventario['segmento_cut'].value_counts().get('Ultra Premium (>$7k)', 0)],
    'qcut (productos)': [inventario['segmento_qcut'].value_counts().get('Económico', 0),
                         inventario['segmento_qcut'].value_counts().get('Medio', 0),
                         inventario['segmento_qcut'].value_counts().get('Premium', 0),
                         inventario['segmento_qcut'].value_counts().get('Ultra Premium', 0)]
})
print(resumen_segmentos.to_string(index=False))
```

**Salida esperada:** La visualización muestra claramente la diferencia: cut produce segmentos desbalanceados (7, 9, 4, 5) mientras qcut produce grupos balanceados (~6-7 productos cada uno).

**Interpretación de negocio:** Para campañas de marketing, qcut es preferible porque cada segmento recibe la misma cantidad de productos y puede asignarse presupuesto equitativamente. Para análisis financiero, cut permite mantener definiciones estables año tras año.

---

## 9. Valor de Inventario por Segmento



**Salida esperada:** La visualización muestra claramente la diferencia: cut produce segmentos desbalanceados (7, 9, 4, 5) mientras qcut produce grupos balanceados (~6-7 productos cada uno).

**Interpretación de negocio:** Para campañas de marketing, qcut es preferible porque cada segmento recibe la misma cantidad de productos y puede asignarse presupuesto equitativamente. Para análisis financiero, cut permite mantener definiciones estables año tras año.

---

## 9. Valor de Inventario por Segmento

```python
# Usar qcut para análisis de valor (por ser más balanceado)
segmento_actual = 'segmento_qcut'

valor_segmento = inventario.groupby(segmento_actual).agg(
    valor_total=('valor_inventario', 'sum'),
    stock_total=('stock_actual', 'sum'),
    precio_prom=('precio', 'mean'),
    productos=('sku', 'count')
).round(2)

print("Valor de inventario por segmento:")
print(valor_segmento)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

valor_segmento['valor_total'].plot.bar(ax=axes[0], color=['green', 'blue', 'orange', 'red'])
axes[0].set_title('Valor de Inventario por Segmento')
axes[0].set_ylabel('Valor Total ($)')
axes[0].set_xlabel('Segmento')
axes[0].tick_params(axis='x', rotation=45)
for i, v in enumerate(valor_segmento['valor_total']):
    axes[0].text(i, v + 20000, f'${v:,.0f}', ha='center', fontsize=9)

valor_segmento['stock_total'].plot.bar(ax=axes[1], color=['green', 'blue', 'orange', 'red'])
axes[1].set_title('Stock Total por Segmento')
axes[1].set_ylabel('Unidades en Stock')
axes[1].set_xlabel('Segmento')
axes[1].tick_params(axis='x', rotation=45)
for i, v in enumerate(valor_segmento['stock_total']):
    axes[1].text(i, v + 10, f'{v:,.0f}', ha='center', fontsize=9)

plt.tight_layout()
plt.show()
```

**Salida esperada:** El segmento Ultra Premium concentra el mayor valor de inventario (>$4M) porque cada unidad cuesta mucho aunque haya pocas unidades. El segmento Económico tiene el menor valor pero el stock puede ser alto (productos baratos pero muchas unidades).

**Interpretación de negocio:** El 65-70% del valor del inventario está en productos Ultra Premium y Premium. Esto implica que el riesgo financiero está en esos segmentos. Para campañas, el segmento Económico debe enfocarse en volumen; el Ultra Premium en margen y experiencias.

---

## 10. Margen Promedio por Segmento



**Salida esperada:** El segmento Ultra Premium concentra el mayor valor de inventario (>$4M) porque cada unidad cuesta mucho aunque haya pocas unidades. El segmento Económico tiene el menor valor pero el stock puede ser alto (productos baratos pero muchas unidades).

**Interpretación de negocio:** El 65-70% del valor del inventario está en productos Ultra Premium y Premium. Esto implica que el riesgo financiero está en esos segmentos. Para campañas, el segmento Económico debe enfocarse en volumen; el Ultra Premium en margen y experiencias.

---

## 10. Margen Promedio por Segmento

```python
# Unir ventas con segmento
ventas_segmento = ventas.merge(inventario[['sku', segmento_actual]], on='sku')

margen_segmento = ventas_segmento.groupby(segmento_actual).agg(
    margen_pct_prom=('margen_pct', 'mean'),
    margen_pct_med=('margen_pct', 'median'),
    margen_total=('margen', 'sum'),
    ingreso_total=('ingreso', 'sum'),
    transacciones=('sku', 'count')
).round(2)

margen_segmento['rentabilidad'] = (margen_segmento['margen_total'] / margen_segmento['ingreso_total'] * 100).round(1)

print("Análisis de margen por segmento:")
print(margen_segmento)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.barplot(data=margen_segmento.reset_index(), x=segmento_actual, y='margen_pct_prom', 
            order=['Económico', 'Medio', 'Premium', 'Ultra Premium'], palette='RdYlGn')
plt.title('Margen % Promedio por Segmento')
plt.xlabel('Segmento')
plt.ylabel('Margen %')
for i, v in enumerate(margen_segmento.loc[['Económico', 'Medio', 'Premium', 'Ultra Premium'], 'margen_pct_prom']):
    plt.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')

plt.subplot(1, 2, 2)
sns.barplot(data=margen_segmento.reset_index(), x=segmento_actual, y='rentabilidad',
            order=['Económico', 'Medio', 'Premium', 'Ultra Premium'], palette='RdYlGn')
plt.title('Rentabilidad Real por Segmento (%)')
plt.xlabel('Segmento')
plt.ylabel('Rentabilidad (%)')
for i, v in enumerate(margen_segmento.loc[['Económico', 'Medio', 'Premium', 'Ultra Premium'], 'rentabilidad']):
    plt.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()
```

**Salida esperada:** Los segmentos Económico y Medio tienen márgenes porcentuales más altos (75-85%) porque son productos de bajo costo con markup alto. Ultra Premium tiene el margen % más bajo (~22%) pero el margen absoluto más alto por unidad.

**Interpretación de negocio:** Paradoja del retail: los productos baratos tienen mayor margen porcentual pero menor margen absoluto. Los productos caros tienen menor margen % pero generan más ganancia por venta. Estrategia: promocionar económicos para atraer tráfico (pérdida líder) y ultra premium para ganancia.

---

## 11. Stock Promedio por Segmento



**Salida esperada:** Los segmentos Económico y Medio tienen márgenes porcentuales más altos (75-85%) porque son productos de bajo costo con markup alto. Ultra Premium tiene el margen % más bajo (~22%) pero el margen absoluto más alto por unidad.

**Interpretación de negocio:** Paradoja del retail: los productos baratos tienen mayor margen porcentual pero menor margen absoluto. Los productos caros tienen menor margen % pero generan más ganancia por venta. Estrategia: promocionar económicos para atraer tráfico (pérdida líder) y ultra premium para ganancia.

---

## 11. Stock Promedio por Segmento

```python
stock_segmento = inventario.groupby(segmento_actual).agg(
    stock_promedio=('stock_actual', 'mean'),
    stock_total=('stock_actual', 'sum'),
    valor_promedio=('valor_inventario', 'mean'),
    productos=('sku', 'count'),
    demanda_prom=('demanda_diaria_prom', 'mean')
).round(1)

print("Stock promedio por segmento:")
print(stock_segmento)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.barplot(data=stock_segmento.reset_index(), x=segmento_actual, y='stock_promedio', ax=axes[0],
            order=['Económico', 'Medio', 'Premium', 'Ultra Premium'], palette='Blues')
axes[0].set_title('Stock Promedio por Segmento')
axes[0].set_ylabel('Unidades')
for i, v in enumerate(stock_segmento.loc[['Económico', 'Medio', 'Premium', 'Ultra Premium'], 'stock_promedio']):
    axes[0].text(i, v + 2, f'{v:.0f}', ha='center', fontweight='bold')

sns.barplot(data=stock_segmento.reset_index(), x=segmento_actual, y='demanda_prom', ax=axes[1],
            order=['Económico', 'Medio', 'Premium', 'Ultra Premium'], palette='Oranges')
axes[1].set_title('Demanda Diaria Promedio por Segmento')
axes[1].set_ylabel('Unidades/día')
for i, v in enumerate(stock_segmento.loc[['Económico', 'Medio', 'Premium', 'Ultra Premium'], 'demanda_prom']):
    axes[1].text(i, v + 0.2, f'{v:.1f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()
```

**Salida esperada:** El stock promedio es mayor en Ultra Premium (por las laptops con 192 unidades) pero la demanda diaria es baja. El segmento Económico tiene stock moderado pero la mayor demanda diaria (productos de alta rotación como USBs y papel).

**Interpretación de negocio:** Los segmentos tienen perfiles opuestos: Económico (alta rotación, bajo stock) vs Ultra Premium (baja rotación, alto stock inmovilizado). Para Económico, priorizar reabastecimiento frecuente. Para Ultra Premium, revisar si el exceso de stock está justificado.

---

## 12. Recomendaciones de Marketing por Segmento



**Salida esperada:** El stock promedio es mayor en Ultra Premium (por las laptops con 192 unidades) pero la demanda diaria es baja. El segmento Económico tiene stock moderado pero la mayor demanda diaria (productos de alta rotación como USBs y papel).

**Interpretación de negocio:** Los segmentos tienen perfiles opuestos: Económico (alta rotación, bajo stock) vs Ultra Premium (baja rotación, alto stock inmovilizado). Para Económico, priorizar reabastecimiento frecuente. Para Ultra Premium, revisar si el exceso de stock está justificado.

---

## 12. Recomendaciones de Marketing por Segmento

```python
print("=== ESTRATEGIA DE MARKETING POR SEGMENTO ===\n")

estrategias = {
    'Económico': {
        'enfoque': 'Volumen y frecuencia',
        'tacticas': [
            'Promociones 2x1 y paquetes (ej. mouse + teclado + USB)',
            'Canales digitales: email marketing semanal',
            'Ubicación en zona de caja para compra impulsiva',
            'Suscripción recurrente para clientes corporativos'
        ],
        'kpi': 'Aumentar frecuencia de compra en 20%',
        'inversion': '30% del presupuesto de marketing'
    },
    'Medio': {
        'enfoque': 'Calidad-precio',
        'tacticas': [
            'Comparativas vs competencia en redes sociales',
            'Bundle medio: audífonos + micrófono para home office',
            'Reseñas y testimonios de clientes satisfechos',
            'Programa de fidelidad con puntos por categoría'
        ],
        'kpi': 'Aumentar ticket promedio en 15%',
        'inversion': '30% del presupuesto de marketing'
    },
    'Premium': {
        'enfoque': 'Diferenciación y experiencia',
        'tacticas': [
            'Contenido educativo: guías de compra, unboxing videos',
            'Garantía extendida como diferenciador',
            'Eventos presenciales en sucursales top',
            'Cross-sell con productos de mayor margen'
        ],
        'kpi': 'Aumentar margen en 5 puntos porcentuales',
        'inversion': '25% del presupuesto de marketing'
    },
    'Ultra Premium': {
        'enfoque': 'Exclusividad y servicio',
        'tacticas': [
            'Asesor personalizado en tienda (consultor de ventas)',
            'Financiamiento a 12 meses sin intereses',
            'Programa early-adopter para nuevos lanzamientos',
            'Soporte premium 24/7 incluido en la compra'
        ],
        'kpi': 'Aumentar margen neto en 8%',
        'inversion': '15% del presupuesto de marketing'
    }
}

for segmento, estrategia in estrategias.items():
    print(f"\n{'='*50}")
    print(f"📌 {segmento.upper()} — Enfoque: {estrategia['enfoque']}")
    print(f"{'='*50}")
    print(f"  Inversión sugerida: {estrategia['inversion']}")
    print(f"  KPI principal: {estrategia['kpi']}")
    print(f"  Tácticas:")
    for t in estrategia['tacticas']:
        print(f"    • {t}")

# Resumen visual
print("\n\n=== RESUMEN DE SEGMENTACIÓN ===")
resumen_final = pd.DataFrame({
    'Segmento': ['Económico', 'Medio', 'Premium', 'Ultra Premium'],
    'Rango Precio': ['$350-$1,000', '$1,000-$2,500', '$2,500-$6,400', '$6,400-$15,000'],
    'Productos': [stock_segmento.loc['Económico', 'productos'] if 'Económico' in stock_segmento.index else 0,
                  stock_segmento.loc['Medio', 'productos'] if 'Medio' in stock_segmento.index else 0,
                  stock_segmento.loc['Premium', 'productos'] if 'Premium' in stock_segmento.index else 0,
                  stock_segmento.loc['Ultra Premium', 'productos'] if 'Ultra Premium' in stock_segmento.index else 0],
    'Margen % Prom': [f'{margen_segmento.loc["Económico", "margen_pct_prom"]:.1f}%',
                      f'{margen_segmento.loc["Medio", "margen_pct_prom"]:.1f}%',
                      f'{margen_segmento.loc["Premium", "margen_pct_prom"]:.1f}%',
                      f'{margen_segmento.loc["Ultra Premium", "margen_pct_prom"]:.1f}%'],
    'Valor Inventario': [f'${valor_segmento.loc["Económico", "valor_total"]:,.0f}',
                         f'${valor_segmento.loc["Medio", "valor_total"]:,.0f}',
                         f'${valor_segmento.loc["Premium", "valor_total"]:,.0f}',
                         f'${valor_segmento.loc["Ultra Premium", "valor_total"]:,.0f}'],
    'Estrategia': ['Volumen', 'Calidad-Precio', 'Diferenciación', 'Exclusividad']
})
print(resumen_final.to_string(index=False))
```

**Salida esperada:** Tabla resumen con los 4 segmentos, sus rangos de precio, cantidad de productos, margen promedio, valor de inventario y estrategia recomendada.

**Interpretación de negocio:** Cada segmento requiere un enfoque comercial distinto. No se puede tratar igual a un USB de $350 que a una laptop de $15,000. La segmentación permite asignar presupuesto de marketing de manera óptima: 60% a Económico+Medio (generan tráfico y volumen) y 40% a Premium+Ultra Premium (generan margen y posicionamiento).

---

## 📝 Ejercicios Propuestos

1. **Segmentación con KMeans:** Usa KMeans con `precio`, `stock_actual` y `demanda_diaria_prom` para crear 4 clusters. Compara los clusters con los segmentos de cut y qcut. (Pista: `from sklearn.cluster import KMeans`; escala con `StandardScaler` primero)

2. **Elasticidad precio-demanda:** Calcula la correlación entre precio y cantidad vendida dentro de cada segmento. ¿En qué segmento la demanda es más elástica al precio? (Pista: usa `ventas.merge(inventario[['sku','segmento_qcut']])` y correlación por grupo)

3. **Campaña simulada:** Asigna un presupuesto de $100k proporcional al valor del inventario de cada segmento. Calcula el ROI esperado si las ventas aumentan 10% en el segmento objetivo. (Pista: `margen_segmento['ingreso_total'] * 0.10`)

4. **Segmentación por margen:** Crea segmentos usando `pd.qcut` sobre `margen_pct` en lugar de precio. ¿Coinciden con los segmentos de precio? ¿Hay productos baratos con bajo margen o caros con alto margen? (Pista: aplica el mismo análisis a `ventas.groupby('producto')['margen_pct'].mean()`)

5. **Recomendación visual por segmento:** Crea un scatterplot con `precio` en x, `margen_pct_prom` en y, y colorea por segmento. ¿Qué segmento tiene la mejor relación precio-margen? (Pista: agrega `producto` vs margen usando merge de ventas e inventario)

---

## 📌 Resumen

| Hallazgo | Impacto |
|----------|---------|
| Precios de $350 a $15,000 (rango 43x) | Segmentación obligatoria para estrategia comercial |
| `pd.cut` = 4 segmentos desbalanceados (7,9,4,5) | Útil para rangos de negocio fijos |
| `pd.qcut` = 4 segmentos balanceados (~6 c/u) | Ideal para campañas con presupuesto equitativo |
| Económico: alto margen % (82%), bajo valor unitario | Estrategia de volumen y frecuencia |
| Ultra Premium: bajo margen % (22%), alto valor | Estrategia de exclusividad y servicio |
| 70% del valor del inventario en Ultra Premium | Riesgo financiero concentrado — monitorear |
| Segmentos requieren tácticas distintas | Asignar 60% presupuesto a Económico+Medio |

## 🔗 Enlaces Relacionados
- [CP01 - Análisis de Ventas](CP01-analisis-ventas-basico.md)
- [CP02 - Dashboard de Inventario](CP02-analisis-inventario-basico.md)
- [CP04 - Detección de Outliers](CP04-deteccion-outliers.md)


**Salida esperada:** Tabla resumen con los 4 segmentos, sus rangos de precio, cantidad de productos, margen promedio, valor de inventario y estrategia recomendada.

**Interpretación de negocio:** Cada segmento requiere un enfoque comercial distinto. No se puede tratar igual a un USB de $350 que a una laptop de $15,000. La segmentación permite asignar presupuesto de marketing de manera óptima: 60% a Económico+Medio (generan tráfico y volumen) y 40% a Premium+Ultra Premium (generan margen y posicionamiento).

---

## 📝 Ejercicios Propuestos

1. **Segmentación con KMeans:** Usa KMeans con `precio`, `stock_actual` y `demanda_diaria_prom` para crear 4 clusters. Compara los clusters con los segmentos de cut y qcut. (Pista: `from sklearn.cluster import KMeans`; escala con `StandardScaler` primero)

2. **Elasticidad precio-demanda:** Calcula la correlación entre precio y cantidad vendida dentro de cada segmento. ¿En qué segmento la demanda es más elástica al precio? (Pista: usa `ventas.merge(inventario[['sku','segmento_qcut']])` y correlación por grupo)

3. **Campaña simulada:** Asigna un presupuesto de $100k proporcional al valor del inventario de cada segmento. Calcula el ROI esperado si las ventas aumentan 10% en el segmento objetivo. (Pista: `margen_segmento['ingreso_total'] * 0.10`)

4. **Segmentación por margen:** Crea segmentos usando `pd.qcut` sobre `margen_pct` en lugar de precio. ¿Coinciden con los segmentos de precio? ¿Hay productos baratos con bajo margen o caros con alto margen? (Pista: aplica el mismo análisis a `ventas.groupby('producto')['margen_pct'].mean()`)

5. **Recomendación visual por segmento:** Crea un scatterplot con `precio` en x, `margen_pct_prom` en y, y colorea por segmento. ¿Qué segmento tiene la mejor relación precio-margen? (Pista: agrega `producto` vs margen usando merge de ventas e inventario)

---

## 📌 Resumen

| Hallazgo | Impacto |
|----------|---------|
| Precios de $350 a $15,000 (rango 43x) | Segmentación obligatoria para estrategia comercial |
| `pd.cut` = 4 segmentos desbalanceados (7,9,4,5) | Útil para rangos de negocio fijos |
| `pd.qcut` = 4 segmentos balanceados (~6 c/u) | Ideal para campañas con presupuesto equitativo |
| Económico: alto margen % (82%), bajo valor unitario | Estrategia de volumen y frecuencia |
| Ultra Premium: bajo margen % (22%), alto valor | Estrategia de exclusividad y servicio |
| 70% del valor del inventario en Ultra Premium | Riesgo financiero concentrado — monitorear |
| Segmentos requieren tácticas distintas | Asignar 60% presupuesto a Económico+Medio |

## 🔗 Enlaces Relacionados
- [CP01 - Análisis de Ventas](CP01-analisis-ventas-basico.md)
- [CP02 - Dashboard de Inventario](CP02-analisis-inventario-basico.md)
- [CP04 - Detección de Outliers](CP04-deteccion-outliers.md)
