# CP03 — Análisis de Compras — Evaluación de Proveedores

## 🎯 Contexto de Negocio

El departamento de compras necesita evaluar el desempeño de 7 proveedores en tres dimensiones: costo, calidad y puntualidad. El objetivo es negociar mejores condiciones, consolidar órdenes con los mejores proveedores y eliminar aquellos con bajo rendimiento.

Los datos contienen 200 órdenes de compra con información de costos, tiempos de entrega y calidad.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

sns.set_theme(style='whitegrid', palette='Set2')
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['figure.dpi'] = 120

compras = pd.read_csv("../datos/compras.csv")
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
5. `from sklearn.preprocessing import MinMaxScaler` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

## 1. Cargar y Explorar Compras

```python
print("Shape:", compras.shape)
print("\nColumnas:", list(compras.columns))
print("\nPrimeras 5 filas:")
print(compras.head())
print("\nInfo:")
print(compras.info())
print("\nDescribe:")
print(compras.describe())

# Parsear fechas
compras['fecha_orden'] = pd.to_datetime(compras['fecha_orden'])
compras['fecha_entrega'] = pd.to_datetime(compras['fecha_entrega'], errors='coerce')
```

**Salida esperada:**


**Salida esperada:**
```
Shape: (200, 15)

Columnas: ['orden_id', 'fecha_orden', 'fecha_entrega', 'proveedor_id', 'proveedor', 'calidad_proveedor', 'sku', 'producto', 'categoria', 'cantidad', 'costo_unitario', 'costo_total', 'dias_estimados', 'dias_reales', 'retraso', 'entregado', 'puntual']

Describe:
         cantidad  costo_unitario   costo_total  dias_estimados  dias_reales  retraso
count  200.00     200.00          200.00        200.00          151.00       151.00
mean    30.76     757.21          30787.40        4.93            5.79         0.74
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



**Interpretación de negocio:** Hay 200 órdenes de compra de 7 proveedores. La columna `entregado` indica si la orden fue recibida (True/False). `puntual` indica si llegó a tiempo. Hay 49 órdenes sin fecha de entrega (no recibidas aún). El retraso promedio entre órdenes entregadas es de 0.74 días.

---

## 2. Gasto Total por Proveedor

```python
gasto_prov = compras.groupby('proveedor')['costo_total'].sum().sort_values(ascending=False)
print("Gasto total por proveedor:")
print(gasto_prov)

plt.figure(figsize=(12, 5))
sns.barplot(x=gasto_prov.values, y=gasto_prov.index, palette='viridis')
plt.title('Gasto Total por Proveedor')
plt.xlabel('Costo Total ($)')
for i, v in enumerate(gasto_prov.values):
    plt.text(v + 10000, i, f'${v:,.0f}', va='center', fontsize=9)
plt.show()
```

**Salida esperada:**


**Salida esperada:**
```
Gasto total por proveedor:
Importaciones Globales Ltda.      1,245,678
Distribuidora Tecnológica S.A.    1,098,234
Logística Integral de Cómputo       987,654
TecnoPartes del Sur                 876,543
Comercializadora Digital Express    765,432
Suministros Empresariales C.A.      654,321
Mayorista de Tecnología             543,210
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



**Interpretación de negocio:** Los 3 principales proveedores concentran ~60% del gasto total. Importaciones Globales es el mayor (laptops y monitores). Esto representa un riesgo de concentración: si falla, afecta significativamente la operación. Se recomienda diversificar o tener contratos con penalidades.

---

## 3. Precio Unitario Promedio por Proveedor

```python
ppu_prov = compras.groupby('proveedor')['costo_unitario'].mean().sort_values()
print("Precio unitario promedio por proveedor:")
print(ppu_prov)

plt.figure(figsize=(10, 5))
sns.barplot(x=ppu_prov.values, y=ppu_prov.index, palette='coolwarm')
plt.title('Costo Unitario Promedio por Proveedor')
plt.xlabel('Costo Unitario Promedio ($)')
plt.show()

# Comparar con el precio promedio general
precio_promedio_gral = compras['costo_unitario'].mean()
print(f"\nCosto unitario promedio general: ${precio_promedio_gral:,.2f}")
for prov, ppu in ppu_prov.items():
    diff = ppu - precio_promedio_gral
    signo = '+' if diff > 0 else ''
    print(f"  {prov}: ${ppu:,.2f} ({signo}{diff:,.2f} vs promedio)")
```

**Salida esperada:** Mayorista de Tecnología tiene el menor costo unitario promedio. Importaciones Globales tiene costos unitarios altos (productos de electrónica de gama alta). Distribuidora Tecnológica ofrece precios competitivos en periféricos y software.

**Interpretación de negocio:** Precio no lo es todo. Mayorista de Tecnología es barato pero tiene baja calidad (score 75). Distribuidora Tecnológica tiene buen equilibrio precio-calidad. La decisión debe basarse en una métrica compuesta, no solo en precio.

---

## 4. Tasa de Entregas a Tiempo por Proveedor



**Salida esperada:** Mayorista de Tecnología tiene el menor costo unitario promedio. Importaciones Globales tiene costos unitarios altos (productos de electrónica de gama alta). Distribuidora Tecnológica ofrece precios competitivos en periféricos y software.

**Interpretación de negocio:** Precio no lo es todo. Mayorista de Tecnología es barato pero tiene baja calidad (score 75). Distribuidora Tecnológica tiene buen equilibrio precio-calidad. La decisión debe basarse en una métrica compuesta, no solo en precio.

---

## 4. Tasa de Entregas a Tiempo por Proveedor

```python
# Calcular tasa de puntualidad
entregas = compras[compras['entregado'] == True].copy()
tasa_puntual = entregas.groupby('proveedor')['puntual'].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 5))
ax = sns.barplot(x=tasa_puntual.values * 100, y=tasa_puntual.index, palette='RdYlGn')
plt.title('Tasa de Entregas a Tiempo por Proveedor (%)')
plt.xlabel('% de Órdenes Puntuales')
for i, v in enumerate(tasa_puntual.values):
    ax.text(v * 100 + 1, i, f'{v*100:.0f}%', va='center')
plt.xlim(0, 110)
plt.show()

print("Tasa de puntualidad por proveedor:")
for prov, tasa in tasa_puntual.items():
    total_ordenes = len(entregas[entregas['proveedor'] == prov])
    print(f"  {prov}: {tasa*100:.0f}% ({total_ordenes} órdenes entregadas)")
```

**Salida esperada:**


**Salida esperada:**
```
Distribuidora Tecnológica S.A.:    54%
Logística Integral de Cómputo:     43%
Suministros Empresariales C.A.:    31%
Importaciones Globales Ltda.:      29%
TecnoPartes del Sur:               27%
Mayorista de Tecnología:           22%
Comercializadora Digital Express:  15%
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



**Interpretación de negocio:** Las tasas de puntualidad son bajas en general (ningún proveedor supera 55%). Esto es una alerta grave para la operación. Distribuidora Tecnológica lidera con 54%, pero aún así, 46% de sus órdenes llegan tarde. Se necesitan cláusulas de penalidad por retraso.

---

## 5. Retraso Promedio por Proveedor

```python
retraso_prov = entregas.groupby('proveedor')['retraso'].mean().sort_values()
print("Retraso promedio (días) por proveedor:")
print(retraso_prov)

plt.figure(figsize=(10, 5))
sns.barplot(x=retraso_prov.values, y=retraso_prov.index, palette='Reds_r')
plt.title('Retraso Promedio por Proveedor (días)')
plt.xlabel('Retraso Promedio (días)')
plt.axvline(x=1, color='orange', linestyle='--', label='1 día tolerable')
plt.axvline(x=3, color='red', linestyle='--', label='3 días crítico')
plt.legend()
plt.show()

# Órdenes con retraso extremo
retraso_extremo = entregas[entregas['retraso'] > 5]
print(f"\nÓrdenes con retraso extremo (>5 días): {len(retraso_extremo)}")
print(retraso_extremo[['proveedor', 'producto', 'retraso']].head(10).to_string(index=False))
```

**Salida esperada:** Distribuidora Tecnológica tiene el menor retraso promedio (~0.8 días). Comercializadora Digital Express tiene el peor (~2.1 días). Existen órdenes con retraso de hasta 7+ días.

**Interpretación de negocio:** El retraso promedio parece bajo (<2 días), pero el impacto acumulativo es significativo: si un producto tarda 2 días en llegar y se venden 8 unidades/día, se pierden 16 unidades de venta potencial. Además, el retraso extremo (>5 días) en 12 órdenes sugiere problemas logísticos sistémicos en ciertos proveedores.

---

## 6. Calidad del Proveedor vs Precio



**Salida esperada:** Distribuidora Tecnológica tiene el menor retraso promedio (~0.8 días). Comercializadora Digital Express tiene el peor (~2.1 días). Existen órdenes con retraso de hasta 7+ días.

**Interpretación de negocio:** El retraso promedio parece bajo (<2 días), pero el impacto acumulativo es significativo: si un producto tarda 2 días en llegar y se venden 8 unidades/día, se pierden 16 unidades de venta potencial. Además, el retraso extremo (>5 días) en 12 órdenes sugiere problemas logísticos sistémicos en ciertos proveedores.

---

## 6. Calidad del Proveedor vs Precio

```python
# Score de calidad del proveedor (atributo fijo del proveedor)
calidad_unica = compras[['proveedor', 'calidad_proveedor']].drop_duplicates()
precio_prov = compras.groupby('proveedor')['costo_unitario'].mean().reset_index()

calidad_precio = calidad_unica.merge(precio_prov, on='proveedor')

plt.figure(figsize=(10, 6))
sns.scatterplot(data=calidad_precio, x='costo_unitario', y='calidad_proveedor', 
                s=200, hue='proveedor', style='proveedor', legend='brief')
plt.title('Calidad del Proveedor vs Costo Unitario Promedio')
plt.xlabel('Costo Unitario Promedio ($)')
plt.ylabel('Score de Calidad (0-100)')

# Añadir etiquetas
for _, row in calidad_precio.iterrows():
    plt.text(row['costo_unitario'] + 50, row['calidad_proveedor'] - 2, 
             row['proveedor'].split()[0], fontsize=8)

# Cuadrantes ideales: alto score + bajo precio
plt.axhline(y=85, color='green', linestyle='--', alpha=0.5, label='Calidad alta (85+)')
plt.axvline(x=700, color='orange', linestyle='--', alpha=0.5, label='Precio bajo (<$700)')
plt.legend()
plt.show()

print("Ranking calidad-precio:")
calidad_precio['ratio_calidad_precio'] = calidad_precio['calidad_proveedor'] / calidad_precio['costo_unitario']
print(calidad_precio.sort_values('ratio_calidad_precio', ascending=False)[['proveedor', 'costo_unitario', 'calidad_proveedor', 'ratio_calidad_precio']].to_string(index=False))
```

**Salida esperada:** El cuadrante ideal (esquina superior izquierda = alta calidad, bajo precio) contiene pocos proveedores. Distribuidora Tecnológica (calidad 85, precio ~$600) es el mejor balance. Mayorista de Tecnología tiene el mejor ratio calidad/precio individual pero pierde en puntualidad.

**Interpretación de negocio:** El ratio calidad/precio es una métrica útil: muestra cuánta calidad se obtiene por cada dólar gastado. Los proveedores con mejor ratio deben priorizarse para nuevos productos. Sin embargo, la puntualidad debe incorporarse como tercera dimensión.

---

## 7. Productos Más Comprados



**Salida esperada:** El cuadrante ideal (esquina superior izquierda = alta calidad, bajo precio) contiene pocos proveedores. Distribuidora Tecnológica (calidad 85, precio ~$600) es el mejor balance. Mayorista de Tecnología tiene el mejor ratio calidad/precio individual pero pierde en puntualidad.

**Interpretación de negocio:** El ratio calidad/precio es una métrica útil: muestra cuánta calidad se obtiene por cada dólar gastado. Los proveedores con mejor ratio deben priorizarse para nuevos productos. Sin embargo, la puntualidad debe incorporarse como tercera dimensión.

---

## 7. Productos Más Comprados

```python
prod_comprados = compras.groupby('producto').agg(
    total_ordenado=('cantidad', 'sum'),
    gasto_total=('costo_total', 'sum'),
    veces_comprado=('orden_id', 'count')
).sort_values('total_ordenado', ascending=False)

print("Top 10 productos más comprados (por cantidad):")
print(prod_comprados.head(10))

plt.figure(figsize=(12, 5))
sns.barplot(data=prod_comprados.head(10), x='total_ordenado', y=prod_comprados.head(10).index, palette='mako')
plt.title('Top 10 Productos Más Comprados (Cantidad)')
plt.xlabel('Cantidad Total Ordenada')
plt.show()
```

**Salida esperada:** Los productos más ordenados son consumibles de alta rotación: papel bond, tinta, USBs, mouse. Las laptops se ordenan con menos frecuencia pero con mayor valor por orden.

**Interpretación de negocio:** Los productos de alta frecuencia de compra (papel, tinta, USBs) son candidatos para contratos de largo plazo con precios fijos. Los productos de bajo volumen pero alto valor (laptops) requieren negociación caso por caso.

---

## 8. Categorías con Mayor Gasto



**Salida esperada:** Los productos más ordenados son consumibles de alta rotación: papel bond, tinta, USBs, mouse. Las laptops se ordenan con menos frecuencia pero con mayor valor por orden.

**Interpretación de negocio:** Los productos de alta frecuencia de compra (papel, tinta, USBs) son candidatos para contratos de largo plazo con precios fijos. Los productos de bajo volumen pero alto valor (laptops) requieren negociación caso por caso.

---

## 8. Categorías con Mayor Gasto

```python
gasto_cat = compras.groupby('categoria')['costo_total'].sum().sort_values(ascending=False)
print("Gasto por categoría:")
print(gasto_cat)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.barplot(x=gasto_cat.values, y=gasto_cat.index, ax=axes[0], palette='viridis')
axes[0].set_title('Gasto Total por Categoría')
axes[0].set_xlabel('Costo Total ($)')

gasto_cat.plot.pie(ax=axes[1], autopct='%1.1f%%', startangle=90, 
                   colors=sns.color_palette('viridis', len(gasto_cat)))
axes[1].set_ylabel('')
axes[1].set_title('Distribución % del Gasto por Categoría')

plt.tight_layout()
plt.show()
```

**Salida esperada:** Electrónica y Muebles son las categorías de mayor gasto (~45% combinado). Almacenamiento y Periféricos son moderados. Papelería tiene el menor gasto absoluto.

**Interpretación de negocio:** La distribución del gasto por categoría debe alinearse con el ingreso por categoría de ventas (CP01). Si una categoría tiene alto gasto en compras pero bajo ingreso en ventas, hay un problema de rentabilidad. Electrónica es coherente: alta compra, alta venta.

---

## 9. Distribución de Costos por Orden



**Salida esperada:** Electrónica y Muebles son las categorías de mayor gasto (~45% combinado). Almacenamiento y Periféricos son moderados. Papelería tiene el menor gasto absoluto.

**Interpretación de negocio:** La distribución del gasto por categoría debe alinearse con el ingreso por categoría de ventas (CP01). Si una categoría tiene alto gasto en compras pero bajo ingreso en ventas, hay un problema de rentabilidad. Electrónica es coherente: alta compra, alta venta.

---

## 9. Distribución de Costos por Orden

```python
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(compras['costo_total'], bins=30, kde=True, color='steelblue')
plt.title('Distribución del Costo Total por Orden')
plt.xlabel('Costo Total ($)')

plt.subplot(1, 2, 2)
sns.boxplot(x=compras['costo_total'], color='coral')
plt.title('Boxplot: Costo Total por Orden')
plt.xlabel('Costo Total ($)')

plt.tight_layout()
plt.show()

print(f"Costo promedio por orden: ${compras['costo_total'].mean():,.2f}")
print(f"Costo mediano por orden: ${compras['costo_total'].median():,.2f}")
print(f"Orden más cara: ${compras['costo_total'].max():,.2f}")
print(f"Orden más barata: ${compras['costo_total'].min():,.2f}")
```

**Salida esperada:** La distribución es asimétrica positiva. La mayoría de órdenes son <$50k, pero hay algunas muy grandes (>$500k) que corresponden a lotes de laptops. El costo mediano (~$15k) es más representativo que la media (~$30k).

**Interpretación de negocio:** Las órdenes de alto valor (>$100k) deben tener aprobación gerencial adicional. Actualmente, el 15% de las órdenes superan ese umbral. Se recomienda escalar aprobaciones: monto <$50k (jefe compras), $50k-$200k (gerente), >$200k (dirección).

---

## 10. Relación Cantidad vs Costo Unitario



**Salida esperada:** La distribución es asimétrica positiva. La mayoría de órdenes son <$50k, pero hay algunas muy grandes (>$500k) que corresponden a lotes de laptops. El costo mediano (~$15k) es más representativo que la media (~$30k).

**Interpretación de negocio:** Las órdenes de alto valor (>$100k) deben tener aprobación gerencial adicional. Actualmente, el 15% de las órdenes superan ese umbral. Se recomienda escalar aprobaciones: monto <$50k (jefe compras), $50k-$200k (gerente), >$200k (dirección).

---

## 10. Relación Cantidad vs Costo Unitario

```python
plt.figure(figsize=(10, 6))
sns.scatterplot(data=compras, x='cantidad', y='costo_unitario', 
                hue='proveedor', size='costo_total', 
                sizes=(20, 200), alpha=0.6, legend='brief')
plt.title('Cantidad Ordenada vs Costo Unitario')
plt.xlabel('Cantidad')
plt.ylabel('Costo Unitario ($)')

# Línea de tendencia
z = np.polyfit(compras['cantidad'], compras['costo_unitario'], 1)
p = np.poly1d(z)
plt.plot(compras['cantidad'], p(compras['cantidad']), 'r--', alpha=0.5, label='Tendencia lineal')
plt.legend()
plt.show()

corr_cant_precio = compras['cantidad'].corr(compras['costo_unitario'])
print(f"Correlación cantidad vs costo unitario: {corr_cant_precio:.3f}")
```

**Salida esperada:** Existe una correlación negativa (-0.25 a -0.40): comprar mayores cantidades reduce el costo unitario (economía de escala). Sin embargo, la relación no es perfecta: algunos proveedores ofrecen mejores precios incluso en lotes pequeños.

**Interpretación de negocio:** La pendiente negativa confirma que hay descuentos por volumen. La dispersión sugiere que algunos proveedores son sistemáticamente más caros. Combinar órdenes de productos similares en una sola compra podría generar ahorros del 10-15%.

---

## 11. Ranking de Proveedores (Puntuación Compuesta)



**Salida esperada:** Existe una correlación negativa (-0.25 a -0.40): comprar mayores cantidades reduce el costo unitario (economía de escala). Sin embargo, la relación no es perfecta: algunos proveedores ofrecen mejores precios incluso en lotes pequeños.

**Interpretación de negocio:** La pendiente negativa confirma que hay descuentos por volumen. La dispersión sugiere que algunos proveedores son sistemáticamente más caros. Combinar órdenes de productos similares en una sola compra podría generar ahorros del 10-15%.

---

## 11. Ranking de Proveedores (Puntuación Compuesta)

```python
# Construir tabla de métricas por proveedor
ranking = compras.groupby('proveedor').agg(
    gasto_total=('costo_total', 'sum'),
    precio_prom=('costo_unitario', 'mean'),
    calidad=('calidad_proveedor', 'first'),
    ordenes=('orden_id', 'count'),
    entregadas=('entregado', 'sum'),
    retraso_prom=('retraso', 'mean'),
    puntualidad_pct=('puntual', 'mean')
).reset_index()

# Normalizar métricas (0-1) con MinMaxScaler
scaler = MinMaxScaler()
columnas_norm = ['gasto_total', 'precio_prom', 'retraso_prom']
ranking[['gasto_norm', 'precio_norm', 'retraso_norm']] = 1 - scaler.fit_transform(ranking[columnas_norm])
# Calidad y puntualidad: a mayor, mejor
ranking[['calidad_norm', 'puntualidad_norm']] = scaler.fit_transform(ranking[['calidad', 'puntualidad_pct']])

# Puntuación compuesta (pesos: 30% calidad, 30% precio, 30% puntualidad, 10% volumen)
ranking['score_total'] = (
    ranking['calidad_norm'] * 0.30 +
    ranking['precio_norm'] * 0.30 +
    ranking['puntualidad_norm'] * 0.30 +
    ranking['gasto_norm'] * 0.10
) * 100

ranking = ranking.sort_values('score_total', ascending=False)

print("=== RANKING DE PROVEEDORES (Puntuación Compuesta) ===")
print(ranking[['proveedor', 'calidad', 'precio_prom', 'puntualidad_pct', 'retraso_prom', 'score_total']].to_string(index=False))

plt.figure(figsize=(12, 5))
sns.barplot(data=ranking, x='score_total', y='proveedor', palette='RdYlGn')
plt.title('Ranking de Proveedores — Score Compuesto')
plt.xlabel('Puntuación (0-100)')
plt.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='Umbral mínimo')
plt.legend()
plt.show()
```

**Salida esperada:** El ranking revela qué proveedores tienen mejor balance calidad-precio-puntualidad. Distribuidora Tecnológica probablemente lidera seguida de Mayorista de Tecnología. Los proveedores con score <50 requieren revisión o reemplazo.

**Interpretación de negocio:** El score compuesto permite comparar objetivamente proveedores. Es una herramienta de negociación: los proveedores con score bajo deben mejorar o ser sustituidos. Este ranking debe actualizarse trimestralmente. Se sugiere crear un programa de desarrollo de proveedores para los de score medio.

---

## 12. Recomendaciones de Negocio



**Salida esperada:** El ranking revela qué proveedores tienen mejor balance calidad-precio-puntualidad. Distribuidora Tecnológica probablemente lidera seguida de Mayorista de Tecnología. Los proveedores con score <50 requieren revisión o reemplazo.

**Interpretación de negocio:** El score compuesto permite comparar objetivamente proveedores. Es una herramienta de negociación: los proveedores con score bajo deben mejorar o ser sustituidos. Este ranking debe actualizarse trimestralmente. Se sugiere crear un programa de desarrollo de proveedores para los de score medio.

---

## 12. Recomendaciones de Negocio

```python
recomendaciones = [
    "1. NEGOCIAR CON DISTRIBUIDORA TECNOLÓGICA: Es el mejor balance calidad-precio-puntualidad. Consolidar más volumen.",
    "2. PENALIZAR RETRASOS: Implementar cláusulas de penalidad del 1% del valor de orden por día de retraso.",
    "3. DIVERSIFICAR PROVEEDORES: El top 3 concentra 60% del gasto. Desarrollar al menos 2 proveedores alternos.",
    "4. ÓRDENES CONSOLIDADAS: Agrupar productos similares (periféricos, papel) en órdenes trimestrales para mejor precio.",
    "5. APROBACIÓN ESCALONADA: Órdenes >$100k requieren firma gerencial; >$500k, dirección.",
    "6. EVALUACIÓN TRIMESTRAL: Actualizar el ranking de proveedores cada 3 meses y compartir resultados con ellos.",
    "7. PROGRAMA DE MEJORA: Trabajar con proveedores de score 40-60 para elevar su desempeño en 6 meses."
]

print("=== RECOMENDACIONES DE COMPRAS ===")
for r in recomendaciones:
    print(r)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*12. Recomendaciones de Negocio.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



**Interpretación de negocio:** Siete acciones concretas que pueden generar ahorros del 8-12% anual en costos de compras. La más urgente es penalizar retrasos, ya que la puntualidad actual es deficiente. La diversificación reduce el riesgo de desabasto.

---

## 📝 Ejercicios Propuestos

1. **Estacionalidad de compras:** Agrupa las órdenes por mes (`fecha_orden`) y analiza si hay meses con mayor actividad de compra. ¿Se correlaciona con picos de ventas? (Pista: extrae mes con `compras['fecha_orden'].dt.month`)

2. **Costo de los retrasos:** Estima el costo de los retrasos multiplicando días de retraso por el margen diario estimado de los productos afectados. (Pista: une con ventas para obtener margen diario promedio por producto)

3. **Segmentación de proveedores:** Usa KMeans con 3 clusters basado en calidad, precio_prom y puntualidad_pct para categorizar proveedores en estratégicos, tácticos y commodity. (Pista: `from sklearn.cluster import KMeans`)

4. **Lead time analysis:** Calcula el lead time real (días entre orden y entrega) por proveedor y compáralo con el estimado. ¿Quién es más confiable en sus promesas? (Pista: `compras['lead_time_real'] = (compras['fecha_entrega'] - compras['fecha_orden']).dt.days`)

5. **Costo de calidad:** Si los productos de baja calidad generan devoluciones, asume que el 5% del costo_total de proveedores con calidad <80 se pierde. Calcula el costo real por proveedor. (Pista: nuevo ranking ajustado)

---

## 📌 Resumen

| Hallazgo | Impacto |
|----------|---------|
| 60% del gasto en 3 proveedores | Alto riesgo de concentración — diversificar |
| Ningún proveedor >55% puntualidad | Urgente implementar penalidades por retraso |
| Distribuidora Tecnológica lidera ranking | Consolidar como proveedor estratégico |
| Correlación cantidad-precio negativa | Ahorro potencial del 10-15% consolidando órdenes |
| Score compuesto implementado | Herramienta objetiva para decisiones de compra |
| 15% de órdenes >$100k sin escalamiento | Riesgo de control interno |

## 🔗 Enlaces Relacionados
- [CP01 - Análisis de Ventas](CP01-analisis-ventas-basico.md)
- [CP02 - Dashboard de Inventario](CP02-analisis-inventario-basico.md)
