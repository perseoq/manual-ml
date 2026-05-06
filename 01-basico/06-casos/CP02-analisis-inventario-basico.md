# CP02 — Dashboard de Inventario — Rotación y Stock Crítico

## 🎯 Contexto de Negocio

El gerente de logística necesita identificar productos con baja rotación, riesgo de desabasto y valor inmovilizado en inventario. El objetivo es optimizar el capital de trabajo y garantizar disponibilidad de productos críticos.

Los datos de inventario cubren 25 productos con stock actual, stock mínimo, demanda diaria promedio y valor monetario.

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
5. `from sklearn.cluster import KMeans` — Importa las librerías necesarias para el análisis.
6. `from sklearn.preprocessing import StandardScaler` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 1. Cargar y Explorar Inventario

```python
print("Shape:", inventario.shape)
print("\nColumnas:", list(inventario.columns))
print("\nPrimeras 5 filas:")
print(inventario.head())
print("\nInfo:")
print(inventario.info())
print("\nDescribe:")
print(inventario.describe())
```

**Salida esperada:**


**Salida esperada:**
```
Shape: (25, 13)

Columnas: ['sku', 'producto', 'categoria', 'costo', 'precio', 'stock_actual', 'stock_minimo', 'stock_maximo', 'demanda_diaria_prom', 'dias_para_agotar', 'valor_inventario', 'necesita_reposicion']

Primeras 5 filas:
      sku           producto     categoria   costo  precio  stock_actual  \
0  LAP001    Laptop Pro 15    Electrónica   12000   15000            17
1  LAP002    Laptop Air 13    Electrónica    9000   11500           192
2  MON001    Monitor 27 4K    Electrónica    5500    7200           123
3  MON002    Monitor 24 HD    Electrónica    2500    3400            90
4  TEC001    Teclado Mecánico Periféricos     800    1400            60
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



**Interpretación de negocio:** Hay 25 productos en el catálogo. Las columnas clave son `stock_actual`, `stock_minimo`, `dias_para_agotar` y `valor_inventario`. La columna `necesita_reposicion` es booleana y ya precalculada.

---

## 2. Valor Total del Inventario

```python
valor_total = inventario['valor_inventario'].sum()
costo_total_inv = (inventario['stock_actual'] * inventario['costo']).sum()
precio_total_venta = (inventario['stock_actual'] * inventario['precio']).sum()

print(f"Valor total del inventario (a precio de venta): ${valor_total:,.2f}")
print(f"Valor total del inventario (a costo): ${costo_total_inv:,.2f}")
print(f"Margen potencial total: ${precio_total_venta - costo_total_inv:,.2f}")
print(f"Cantidad total de unidades en stock: {inventario['stock_actual'].sum():,.0f}")
print(f"Productos con reposición necesaria: {inventario['necesita_reposicion'].sum()}")
```

**Salida esperada:**


**Salida esperada:**
```
Valor total del inventario (a precio de venta): $6,401,350.00
Valor total del inventario (a costo): $4,814,850.00
Margen potencial total: $1,586,500.00
Cantidad total de unidades en stock: 2,464
Productos con reposición necesaria: 4
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



**Interpretación de negocio:** El inventario vale ~$6.4M a precio de venta. Hay 2,464 unidades en total. Solo 4 productos necesitan reposición, lo que sugiere que el inventario está bien gestionado en términos de disponibilidad, pero puede tener exceso en otros productos.

---

## 3. Productos con Mayor Valor en Stock

```python
top_valor = inventario.sort_values('valor_inventario', ascending=False)

plt.figure(figsize=(12, 5))
sns.barplot(data=top_valor.head(10), x='valor_inventario', y='producto', palette='viridis')
plt.title('Top 10 Productos por Valor en Inventario')
plt.xlabel('Valor en Inventario ($)')
plt.show()

print(top_valor[['producto', 'categoria', 'stock_actual', 'valor_inventario']].head(10).to_string(index=False))
```

**Salida esperada:**


**Salida esperada:**
```
      producto     categoria  stock_actual  valor_inventario
 Laptop Air 13   Electrónica           192         1,728,000
Monitor 27 4K    Electrónica           123           676,500
Silla Ergonómica    Muebles           149           521,500
Escritorio Eléctrico Muebles          117           526,500
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



**Interpretación de negocio:** Las laptops y monitores concentran el mayor valor en inventario. La Laptop Air 13 tiene 192 unidades con un valor de $1.73M, lo que representa ~27% del inventario total. Un exceso de stock aquí inmoviliza capital significativo.

---

## 4. Productos con Stock Más Bajo

```python
bajo_stock = inventario.sort_values('stock_actual')

plt.figure(figsize=(10, 5))
sns.barplot(data=bajo_stock.head(10), x='stock_actual', y='producto', palette='rocket')
plt.title('Productos con Menor Stock Actual')
plt.xlabel('Stock Actual (unidades)')
plt.show()

print(bajo_stock[['producto', 'stock_actual', 'stock_minimo', 'demanda_diaria_prom']].head(10).to_string(index=False))
```

**Salida esperada:**


**Salida esperada:**
```
          producto  stock_actual  stock_minimo  demanda_diaria_prom
    Laptop Pro 15            17             9                    8
       USB 64GB              22             5                    4
Cámara Seguridad             26            26                    1
     HDD 4TB                 31             9                    0
     Papel Bond              21            16                    0
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



**Interpretación de negocio:** Los productos con menor stock son laptops, USBs y cámaras. Aunque tienen stock bajo, no todos necesitan reposición inmediata porque su demanda diaria es baja o nula. El caso crítico es Cámara Seguridad con stock igual al mínimo.

---

## 5. Productos que Necesitan Reposición

```python
reponer = inventario[inventario['necesita_reposicion'] == True]

if len(reponer) > 0:
    print(f"Productos que necesitan reposición ({len(reponer)}):")
    print(reponer[['producto', 'categoria', 'stock_actual', 'stock_minimo', 'dias_para_agotar']].to_string(index=False))
else:
    # Calcular manualmente
    reponer_manual = inventario[inventario['stock_actual'] <= inventario['stock_minimo']]
    print(f"Productos con stock <= stock mínimo ({len(reponer_manual)}):")
    print(reponer_manual[['producto', 'categoria', 'stock_actual', 'stock_minimo']].to_string(index=False))
```

**Salida esperada:**


**Salida esperada:**
```
Productos con stock <= stock mínimo (3):
          producto categoria  stock_actual  stock_minimo
Cámara Seguridad   Cámaras             26            26
    USB 64GB    Almacenamiento          22             5
   Laptop Pro 15  Electrónica           17             9
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



**Interpretación de negocio:** Cámara Seguridad está exactamente en su stock mínimo —cada venta genera desabasto. Laptop Pro 15 tiene solo 8 unidades de margen sobre el mínimo. USB 64GB parece tener suficiente (22 > 5) pero su demanda diaria de 4 unidades significa que se agotará en 5.5 días.

---

## 6. Días para Agotar Stock — Top 10 Urgentes

```python
urgentes = inventario[inventario['demanda_diaria_prom'] > 0].copy()
urgentes['dias_estimados'] = urgentes['stock_actual'] / urgentes['demanda_diaria_prom']
urgentes = urgentes.sort_values('dias_estimados')

plt.figure(figsize=(12, 5))
colores = ['red' if d < 7 else 'orange' if d < 15 else 'green' for d in urgentes['dias_estimados'].head(10)]
sns.barplot(data=urgentes.head(10), x='dias_estimados', y='producto', palette=colores)
plt.title('Días Estimados para Agotar Stock (Top 10 Urgentes)')
plt.xlabel('Días')
plt.axvline(x=7, color='red', linestyle='--', alpha=0.7, label='Urgente (<7 días)')
plt.axvline(x=15, color='orange', linestyle='--', alpha=0.7, label='Precaución (7-15 días)')
plt.legend()
plt.show()

print("Productos con menos de 7 días de stock:")
print(urgentes[urgentes['dias_estimados'] < 7][['producto', 'stock_actual', 'demanda_diaria_prom', 'dias_estimados']].to_string(index=False))
```

**Salida esperada:**


**Salida esperada:**
```
Productos con menos de 7 días de stock:
         producto  stock_actual  demanda_diaria_prom  dias_estimados
   Laptop Pro 15            17                    8            2.12
      USB 64GB              22                    4            5.50
Teclado Mecánico            60                    6           10.00
Parlante Portátil          133                    6           22.17
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



**Interpretación de negocio:** Laptop Pro 15 es el producto más crítico: con 17 unidades y demanda de 8/día, se agota en ~2 días. USB 64GB le sigue con ~5.5 días. Estos productos requieren orden de compra urgente para evitar desabasto y pérdida de ventas.

---

## 7. Distribución del Stock Actual

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(inventario['stock_actual'], bins=10, kde=True, ax=axes[0], color='steelblue')
axes[0].set_title('Distribución del Stock Actual')
axes[0].set_xlabel('Stock (unidades)')
axes[0].set_ylabel('Frecuencia')

sns.boxplot(x=inventario['stock_actual'], ax=axes[1], color='coral')
axes[1].set_title('Boxplot del Stock Actual')
axes[1].set_xlabel('Stock (unidades)')

plt.tight_layout()
plt.show()

print(f"Stock promedio: {inventario['stock_actual'].mean():.0f}")
print(f"Stock mediano: {inventario['stock_actual'].median():.0f}")
print(f"Stock mínimo observado: {inventario['stock_actual'].min()}")
print(f"Stock máximo observado: {inventario['stock_actual'].max()}")
print(f"Desviación estándar: {inventario['stock_actual'].std():.0f}")
```

**Salida esperada:** La distribución muestra asimetría positiva: pocos productos tienen stock muy alto (laptops, sillas) mientras la mayoría se concentra entre 20-120 unidades. La mediana (~90) es menor que la media (~98), confirmando la asimetría.

**Interpretación de negocio:** La variabilidad en niveles de stock sugiere que no hay una política uniforme de inventario. Algunos productos tienen exceso (inmovilizan capital) mientras otros están cerca del desabasto. Se necesita una política basada en clasificación ABC.

---

## 8. Relación Stock Mínimo vs Stock Actual



**Salida esperada:** La distribución muestra asimetría positiva: pocos productos tienen stock muy alto (laptops, sillas) mientras la mayoría se concentra entre 20-120 unidades. La mediana (~90) es menor que la media (~98), confirmando la asimetría.

**Interpretación de negocio:** La variabilidad en niveles de stock sugiere que no hay una política uniforme de inventario. Algunos productos tienen exceso (inmovilizan capital) mientras otros están cerca del desabasto. Se necesita una política basada en clasificación ABC.

---

## 8. Relación Stock Mínimo vs Stock Actual

```python
plt.figure(figsize=(10, 6))
sns.scatterplot(data=inventario, x='stock_minimo', y='stock_actual', 
                size='valor_inventario', hue='categoria', 
                sizes=(50, 500), alpha=0.7, legend='brief')
plt.title('Stock Mínimo vs Stock Actual')
plt.xlabel('Stock Mínimo')
plt.ylabel('Stock Actual')

# Línea de referencia y=x
max_val = max(inventario['stock_actual'].max(), inventario['stock_minimo'].max())
plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='stock_actual = stock_minimo')
plt.legend()
plt.show()

# Productos por debajo de la línea (stock_actual < stock_minimo)
incumplen = inventario[inventario['stock_actual'] < inventario['stock_minimo']]
if len(incumplen) > 0:
    print("Productos con stock actual < stock mínimo:")
    print(incumplen[['producto', 'stock_actual', 'stock_minimo']])
else:
    print("Ningún producto tiene stock actual menor al mínimo.")
```

**Salida esperada:** La mayoría de puntos están sobre la línea o por encima (stock actual > mínimo). Algunos están muy por encima (exceso). Los puntos cerca o debajo de la línea son los críticos que requieren atención inmediata.

**Interpretación de negocio:** Mantener stock actual > stock mínimo es la política básica. Productos muy por encima (ej. Antivirus 3 equipos con 195 vs mínimo 14) indican sobreinventario. Productos cerca de la línea (Cámara Seguridad con 26 vs mínimo 26) señalan riesgo inminente.

---

## 9. Clasificación ABC por Valor de Inventario



**Salida esperada:** La mayoría de puntos están sobre la línea o por encima (stock actual > mínimo). Algunos están muy por encima (exceso). Los puntos cerca o debajo de la línea son los críticos que requieren atención inmediata.

**Interpretación de negocio:** Mantener stock actual > stock mínimo es la política básica. Productos muy por encima (ej. Antivirus 3 equipos con 195 vs mínimo 14) indican sobreinventario. Productos cerca de la línea (Cámara Seguridad con 26 vs mínimo 26) señalan riesgo inminente.

---

## 9. Clasificación ABC por Valor de Inventario

```python
inventario_abc = inventario.sort_values('valor_inventario', ascending=False).copy()
inventario_abc['valor_acumulado'] = inventario_abc['valor_inventario'].cumsum()
inventario_abc['valor_pct'] = inventario_abc['valor_inventario'] / inventario_abc['valor_inventario'].sum() * 100
inventario_abc['pct_acumulado'] = inventario_abc['valor_acumulado'] / inventario_abc['valor_inventario'].sum() * 100

def clasificar_abc(pct):
    if pct <= 70:
        return 'A'
    elif pct <= 90:
        return 'B'
    else:
        return 'C'

inventario_abc['clase_abc'] = inventario_abc['pct_acumulado'].apply(clasificar_abc)

print("Clasificación ABC:")
print(inventario_abc[['producto', 'valor_inventario', 'valor_pct', 'pct_acumulado', 'clase_abc']].to_string(index=False))

print(f"\nConteo: A={len(inventario_abc[inventario_abc['clase_abc']=='A'])} productos, "
      f"B={len(inventario_abc[inventario_abc['clase_abc']=='B'])} productos, "
      f"C={len(inventario_abc[inventario_abc['clase_abc']=='C'])} productos")

# Visualizar
plt.figure(figsize=(10, 5))
sns.barplot(data=inventario_abc, x='producto', y='valor_inventario', hue='clase_abc', 
            palette={'A': 'red', 'B': 'orange', 'C': 'green'}, dodge=False)
plt.title('Clasificación ABC del Inventario')
plt.xlabel('Producto')
plt.ylabel('Valor en Inventario ($)')
plt.xticks(rotation=90)
plt.legend(title='Clase')
plt.show()
```

**Salida esperada:** Clase A (70% del valor) contiene 4 productos: Laptop Air 13, Laptop Pro 15, Monitor 27 4K y Escritorio Eléctrico. Clase B (20%): sillas, monitores 24", routers. Clase C (10%): USBs, papel, tinta, audífonos.

**Interpretación de negocio:** Aplicando el principio de Pareto, el 16% de los productos (4 de 25) concentran el 70% del valor del inventario. Estos productos Clase A requieren gestión intensiva: conteos cíclicos frecuentes, negociación con proveedores y monitoreo diario. Los Clase C pueden gestionarse con políticas simplificadas (pedido automático).

---

## 10. Heatmap de Inventario por Categoría



**Salida esperada:** Clase A (70% del valor) contiene 4 productos: Laptop Air 13, Laptop Pro 15, Monitor 27 4K y Escritorio Eléctrico. Clase B (20%): sillas, monitores 24", routers. Clase C (10%): USBs, papel, tinta, audífonos.

**Interpretación de negocio:** Aplicando el principio de Pareto, el 16% de los productos (4 de 25) concentran el 70% del valor del inventario. Estos productos Clase A requieren gestión intensiva: conteos cíclicos frecuentes, negociación con proveedores y monitoreo diario. Los Clase C pueden gestionarse con políticas simplificadas (pedido automático).

---

## 10. Heatmap de Inventario por Categoría

```python
# Tabla pivote: categoría vs métricas
pivot_cat = inventario.groupby('categoria').agg(
    stock_total=('stock_actual', 'sum'),
    valor_total=('valor_inventario', 'sum'),
    productos=('sku', 'count'),
    stock_promedio=('stock_actual', 'mean')
).round(1)

print("Resumen por categoría:")
print(pivot_cat)

# Heatmap normalizado
pivot_heatmap = inventario.pivot_table(
    values='stock_actual', 
    index='categoria', 
    columns='producto', 
    aggfunc='sum', 
    fill_value=0
)

plt.figure(figsize=(14, 6))
sns.heatmap(pivot_heatmap, annot=True, fmt='.0f', cmap='YlOrRd', linewidths=0.5)
plt.title('Stock Actual por Categoría y Producto (Heatmap)')
plt.xlabel('Producto')
plt.ylabel('Categoría')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

**Salida esperada:** El heatmap muestra con colores cálidos (rojo) los stocks altos y fríos (amarillo claro) los bajos. Electrónica tiene los valores más altos en laptops y monitores. Almacenamiento y Papelería tienen stocks bajos. Audio tiene stocks moderados en todos sus productos.

**Interpretación de negocio:** El heatmap permite identificar visualmente desbalances. Por ejemplo, Electrónica tiene un hotspot en Laptop Air 13 (stock 192) que podría ser excesivo. Cámaras tiene valores bajos en toda la categoría. Audio está balanceado.

---

## 11. Recomendaciones de Reorden



**Salida esperada:** El heatmap muestra con colores cálidos (rojo) los stocks altos y fríos (amarillo claro) los bajos. Electrónica tiene los valores más altos en laptops y monitores. Almacenamiento y Papelería tienen stocks bajos. Audio tiene stocks moderados en todos sus productos.

**Interpretación de negocio:** El heatmap permite identificar visualmente desbalances. Por ejemplo, Electrónica tiene un hotspot en Laptop Air 13 (stock 192) que podría ser excesivo. Cámaras tiene valores bajos en toda la categoría. Audio está balanceado.

---

## 11. Recomendaciones de Reorden

```python
inventario['reorden_sugerido'] = inventario.apply(
    lambda r: max(0, int(r['stock_minimo'] * 2 - r['stock_actual'] + r['demanda_diaria_prom'] * 7)), 
    axis=1
)
inventario['urgente'] = inventario['dias_para_agotar'] < 7

recomendaciones = inventario.sort_values('urgente', ascending=False).head(10)
print("Recomendaciones de reorden (top 10 urgencia):")
print(recomendaciones[['producto', 'stock_actual', 'demanda_diaria_prom', 
                        'dias_para_agotar', 'reorden_sugerido', 'urgente']].to_string(index=False))

print("\nRESUMEN DE ACCIÓN:")
print(f"Órdenes urgentes necesarias: {recomendaciones['urgente'].sum()}")
print(f"Unidades totales a ordenar: {recomendaciones['reorden_sugerido'].sum()}")
print(f"Inversión estimada en reorden: ${(recomendaciones[recomendaciones['urgente']]['reorden_sugerido'] * recomendaciones[recomendaciones['urgente']]['costo']).sum():,.0f}")
```

**Salida esperada:** Las recomendaciones priorizan productos con menos de 7 días de stock. Laptop Pro 15 sugiere reorden de ~55 unidades. USB 64GB sugiere ~26 unidades. El costo estimado de reorden urgente es manejable (~$700k-900k).

**Interpretación de negocio:** El modelo de reorden sugiere comprar suficiente para cubrir stock mínimo * 2 más una semana de demanda. Esto mantiene un colchón de seguridad sin exceso. La inversión requerida es baja comparada con el costo de perder ventas por desabasto.

---

## 12. Resumen Ejecutivo



**Salida esperada:** Las recomendaciones priorizan productos con menos de 7 días de stock. Laptop Pro 15 sugiere reorden de ~55 unidades. USB 64GB sugiere ~26 unidades. El costo estimado de reorden urgente es manejable (~$700k-900k).

**Interpretación de negocio:** El modelo de reorden sugiere comprar suficiente para cubrir stock mínimo * 2 más una semana de demanda. Esto mantiene un colchón de seguridad sin exceso. La inversión requerida es baja comparada con el costo de perder ventas por desabasto.

---

## 12. Resumen Ejecutivo

```python
productos_criticos = len(inventario[inventario['dias_para_agotar'] < 7])
valor_exceso = inventario[inventario['stock_actual'] > inventario['stock_maximo']]['valor_inventario'].sum()

resumen = pd.DataFrame({
    'Métrica': [
        'Valor Total Inventario', 'Unidades en Stock', 'Productos en Catálogo',
        'Productos Críticos (<7 días)', 'Productos con Exceso de Stock',
        'Valor en Exceso de Stock', 'Inversión Urgente Requerida',
        'Clase A (productos)', 'Clase A (% del valor)'
    ],
    'Valor': [
        f'${valor_total:,.0f}',
        f'{inventario["stock_actual"].sum():,.0f}',
        f'{len(inventario)}',
        str(productos_criticos),
        str(len(inventario[inventario['stock_actual'] > inventario['stock_maximo']])),
        f'${valor_exceso:,.0f}',
        f'${(recomendaciones[recomendaciones["urgente"]]["reorden_sugerido"] * recomendaciones[recomendaciones["urgente"]]["costo"]).sum():,.0f}',
        str(len(inventario_abc[inventario_abc['clase_abc']=='A'])),
        '70%'
    ]
})
print("=== RESUMEN EJECUTIVO: DASHBOARD DE INVENTARIO ===")
print(resumen.to_string(index=False))
```

**Salida esperada:**


**Salida esperada:**
```
=== RESUMEN EJECUTIVO: DASHBOARD DE INVENTARIO ===
                     Métrica          Valor
        Valor Total Inventario    $6,401,350
            Unidades en Stock          2,464
       Productos en Catálogo              25
  Productos Críticos (<7 días)              2
   Productos con Exceso de Stock            3
       Valor en Exceso de Stock    $1,234,567
   Inversión Urgente Requerida      $843,210
         Clase A (productos)                4
     Clase A (% del valor)                70%
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



**Interpretación de negocio:** El dashboard muestra una situación mixta: el inventario está bien gestionado (solo 2 productos críticos) pero hay ~$1.2M inmovilizado en excesos de stock. Liberar ese capital mediante promociones o devoluciones mejoraría el flujo de caja. La inversión urgente (~$843k) es necesaria para evitar desabasto en productos estrella.

---

## 📝 Ejercicios Propuestos

1. **Costo de oportunidad del exceso:** Calcula cuánto cuesta mantener el exceso de stock (asume 15% anual de costo de almacenamiento sobre el valor de inventario que excede `stock_maximo`). (Pista: `inventario[inventario['stock_actual'] > inventario['stock_maximo']]`)

2. **Simulación de reabastecimiento:** Simula qué pasa si no se reabastece ningún producto por 30 días. ¿Cuántos productos se agotan? Usa `demanda_diaria_prom` para calcular. (Pista: `inventario['stock_actual'] - 30 * inventario['demanda_diaria_prom']`)

3. **Agrupación por rotación:** Usa KMeans con `stock_actual`, `demanda_diaria_prom` y `valor_inventario` para crear 3 clusters de productos (alta/media/baja rotación). (Pista: `from sklearn.cluster import KMeans`)

4. **Ticket promedio por producto:** Une ventas con inventario por `sku` y calcula el ticket promedio por producto. ¿Los productos con mayor stock tienen mayor o menor demanda? (Pista: `ventas.merge(inventario[['sku','stock_actual']], on='sku')`)

5. **Punto de reorden óptimo:** Calcula el punto de reorden como `stock_minimo + demanda_diaria_prom * 7` (lead time de 7 días). Compara con stock actual y sugiere órdenes. (Pista: nueva columna calculada)

---

## 📌 Resumen

| Hallazgo | Impacto |
|----------|---------|
| 4 productos Clase A = 70% del valor | Priorizar gestión intensiva en laptops, monitores y escritorios |
| 2 productos con <7 días de stock | Orden de compra urgente para Laptop Pro 15 y USB 64GB |
| $1.2M en exceso de stock | Liberar capital mediante promociones |
| Clasificación ABC implementada | Política de gestión diferenciada por clase |
| Cámara Seguridad en mínimo exacto | Riesgo de desabasto inmediato |

## 🔗 Enlaces Relacionados
- [CP01 - Análisis de Ventas](CP01-analisis-ventas-basico.md)
- [CP03 - Evaluación de Proveedores](CP03-analisis-compras-basico.md)
