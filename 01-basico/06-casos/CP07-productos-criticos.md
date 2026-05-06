# CP07 — Identificación de Productos Críticos — Clasificación ABC + Stock

## 🎯 Contexto de Negocio

El gerente de logística necesita priorizar productos según su contribución al valor total de ventas (ABC) y su rotación de inventario. El objetivo es mantener stock óptimo de productos de alto valor, identificar riesgos de desabasto y liberar capital inmovilizado en productos de baja rotación.

Según el principio de Pareto, el 20% de los productos genera el 80% del valor. Este caso aplica ese principio a los datos reales de inventario y ventas.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style='whitegrid', palette='Set2')
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['figure.dpi'] = 120

ventas = pd.read_csv("../datos/ventas.csv")
inventario = pd.read_csv("../datos/inventario.csv")
ventas['fecha'] = pd.to_datetime(ventas['fecha'])
```

**Salida:**

```
# La salida muestra los resultados del procesamiento de datos.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación:**

Cada línea del código realiza una operación específica sobre los datos. Ejecutar para ver resultados.

---


---

## 1. Merge de Inventario y Ventas por SKU

```python
# Agregar ventas a nivel SKU
ventas_sku = ventas.groupby('sku').agg(
    cantidad_vendida=('cantidad', 'sum'),
    ingreso_total=('ingreso', 'sum'),
    transacciones=('sku', 'count'),
    clientes_distintos=('cliente', 'nunique')
).reset_index()

# Merge con inventario
inv_ventas = inventario.merge(ventas_sku, on='sku', how='left')

# Llenar nulos (productos sin ventas)
inv_ventas['cantidad_vendida'] = inv_ventas['cantidad_vendida'].fillna(0)
inv_ventas['ingreso_total'] = inv_ventas['ingreso_total'].fillna(0)
inv_ventas['transacciones'] = inv_ventas['transacciones'].fillna(0)
inv_ventas['clientes_distintos'] = inv_ventas['clientes_distintos'].fillna(0)

print(f"Shape merge: {inv_ventas.shape}")
print(f"Productos sin ventas: {(inv_ventas['cantidad_vendida']==0).sum()}")
print(inv_ventas[['sku', 'producto', 'categoria', 'stock_actual', 'cantidad_vendida', 'ingreso_total']].head())
```

**Interpretación de negocio:** El merge revela qué productos tienen ventas y cuáles no. Los productos sin ventas representan capital inmovilizado. Si un producto tiene stock pero cero ventas, es candidato a liquidación o devolución a proveedor. La integración de ambas tablas es el paso fundamental para cualquier análisis de criticidad.

---

## 2. Calcular Valor de Ventas por Producto

```python
inv_ventas['valor_ventas'] = inv_ventas['cantidad_vendida'] * inv_ventas['precio']
inv_ventas['costo_ventas'] = inv_ventas['cantidad_vendida'] * inv_ventas['costo']
inv_ventas['margen_total'] = inv_ventas['valor_ventas'] - inv_ventas['costo_ventas']
inv_ventas['margen_pct'] = (inv_ventas['margen_total'] / inv_ventas['valor_ventas'] * 100).fillna(0)

print("Top 10 productos por valor de ventas:")
print(inv_ventas.sort_values('valor_ventas', ascending=False)[
    ['producto', 'categoria', 'valor_ventas', 'margen_pct', 'stock_actual']
].head(10))

valor_total = inv_ventas['valor_ventas'].sum()
print(f"\nValor total de ventas (cartera): ${valor_total:,.0f}")
```

**Interpretación de negocio:** Calcular el valor de ventas por producto permite ordenar el portafolio por importancia económica. Los productos de la cima merecen atención prioritaria en disponibilidad, calidad y promoción. Los del fondo pueden requerir decisiones de descontinuación. Este ranking es la base para cualquier sistema de clasificación.

---

## 3. Clasificación ABC (80/15/5)

```python
inv_ventas_ordenado = inv_ventas.sort_values('valor_ventas', ascending=False).copy()
inv_ventas_ordenado['valor_acumulado'] = inv_ventas_ordenado['valor_ventas'].cumsum()
inv_ventas_ordenado['porcentaje_acumulado'] = inv_ventas_ordenado['valor_acumulado'] / inv_ventas_ordenado['valor_ventas'].sum() * 100

def clasificar_abc(porcentaje):
    if porcentaje <= 80:
        return 'A'
    elif porcentaje <= 95:
        return 'B'
    else:
        return 'C'

inv_ventas_ordenado['clase_abc'] = inv_ventas_ordenado['porcentaje_acumulado'].apply(clasificar_abc)

print(inv_ventas_ordenado[['producto', 'valor_ventas', 'porcentaje_acumulado', 'clase_abc']])

resumen_abc = inv_ventas_ordenado.groupby('clase_abc').agg(
    productos=('sku', 'count'),
    valor_total=('valor_ventas', 'sum'),
    porcentaje_valor=('valor_ventas', lambda x: x.sum() / inv_ventas_ordenado['valor_ventas'].sum() * 100)
).reset_index()
print("\n=== RESUMEN CLASIFICACIÓN ABC ===")
print(resumen_abc)
```

**Interpretación de negocio:** La clasificación ABC sigue el principio de Pareto: los productos A (20% del catálogo) generan el 80% del valor. Los productos C (50%+ del catálogo) apenas contribuyen al 5%. La estrategia debe ser: productos A = disponibilidad absoluta, productos B = gestión normal, productos C = minimizar inventario o eliminar.

---

## 4. Visualizar Clasificación ABC

```python
plt.figure(figsize=(12, 6))
ax = sns.barplot(data=inv_ventas_ordenado, x=range(len(inv_ventas_ordenado)),
                 y='valor_ventas', hue='clase_abc',
                 palette={'A': '#e74c3c', 'B': '#f39c12', 'C': '#3498db'},
                 dpi=False, legend=True)

# Línea de porcentaje acumulado
ax2 = ax.twinx()
ax2.plot(range(len(inv_ventas_ordenado)), inv_ventas_ordenado['porcentaje_acumulado'],
         color='darkgreen', linewidth=2, marker='o', markersize=4, label='% Acumulado')
ax2.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80%')
ax2.axhline(y=95, color='orange', linestyle='--', alpha=0.7, label='95%')
ax2.set_ylabel('% Acumulado', color='darkgreen')

plt.title('Clasificación ABC de Productos — Curva de Pareto', fontsize=14, fontweight='bold')
plt.xlabel('Productos (ordenados por valor descendente)')
plt.ylabel('Valor de Ventas ($)')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
```

**Interpretación de negocio:** La curva de Pareto muestra visualmente la concentración del valor. La línea roja al 80% marca el límite de productos A. La distancia vertical entre la curva y la línea diagonal representa la desigualdad. Cuanto más pronunciada la curva, más críticos son los pocos productos A. Esto ayuda a comunicar a la dirección por qué ciertos productos merecen más inversión en inventario.

---

## 5. Identificar Productos A con Bajo Stock (Críticos)

```python
productos_a = inv_ventas_ordenado[inv_ventas_ordenado['clase_abc'] == 'A'].copy()
productos_a['stock_dias'] = productos_a['stock_actual'] / productos_a['demanda_diaria_prom']
productos_a['stock_dias'] = productos_a['stock_dias'].replace([np.inf, -np.inf], np.nan)

criticos = productos_a[productos_a['stock_dias'] < 30].sort_values('stock_dias')

print(f"Productos A con stock < 30 días de demanda:")
print(criticos[['producto', 'categoria', 'valor_ventas', 'stock_actual',
                 'demanda_diaria_prom', 'stock_dias']])

print(f"\nTotal de productos A críticos: {len(criticos)}")
print(f"Valor en riesgo: ${criticos['valor_ventas'].sum():,.0f}")
```

**Interpretación de negocio:** Los productos A con bajo stock representan el mayor riesgo operativo. Si un producto que genera el 80% del valor se agota, el impacto en ingresos es inmediato. Estos productos requieren orden de compra urgente, renegociación con proveedores y posiblemente envíos express. La métrica `stock_dias` debe monitorearse diariamente.

---

## 6. Calcular Rotación de Inventario

```python
# Rotación anualizada = ventas totales / stock_promedio
# stock promedio estimado como stock_actual (asumiendo inventario estable)
inv_ventas_ordenado['rotacion'] = inv_ventas_ordenado['cantidad_vendida'] / inv_ventas_ordenado['stock_actual'].replace(0, np.nan)

# Clasificar rotación
inv_ventas_ordenado['clase_rotacion'] = pd.cut(
    inv_ventas_ordenado['rotacion'],
    bins=[-np.inf, 1, 6, np.inf],
    labels=['Baja', 'Media', 'Alta']
)

print("Rotación por producto (top y bottom 5):")
print(inv_ventas_ordenado.sort_values('rotacion', ascending=False)[
    ['producto', 'cantidad_vendida', 'stock_actual', 'rotacion', 'clase_rotacion']
].head(10))

print("\nProductos con rotación baja o nula:")
print(inv_ventas_ordenado[inv_ventas_ordenado['clase_rotacion'] == 'Baja'][
    ['producto', 'stock_actual', 'rotacion', 'valor_inventario']
])
```

**Interpretación de negocio:** La rotación de inventario mide cuántas veces se vende y repone el stock en un período. Rotación alta (>6) significa producto popular que necesita reabastecimiento frecuente. Rotación baja (<1) significa producto estancado que inmoviliza capital. La meta es tener rotación alta en productos A y rotación controlada en B/C.

---

## 7. Productos de Alta Rotación vs Baja Rotación

```python
alta_rot = inv_ventas_ordenado[inv_ventas_ordenado['clase_rotacion'] == 'Alta']
baja_rot = inv_ventas_ordenado[inv_ventas_ordenado['clase_rotacion'] == 'Baja']

comparacion_rotacion = pd.DataFrame({
    'Métrica': ['Cantidad Productos', 'Valor Ventas Total', 'Stock Actual Total',
                'Valor Inventario', 'Margen Promedio (%)'],
    'Alta Rotación': [
        len(alta_rot),
        f"${alta_rot['valor_ventas'].sum():,.0f}",
        f"{alta_rot['stock_actual'].sum():.0f}",
        f"${alta_rot['valor_inventario'].sum():,.0f}",
        f"{alta_rot['margen_pct'].mean():.1f}%"
    ],
    'Baja Rotación': [
        len(baja_rot),
        f"${baja_rot['valor_ventas'].sum():,.0f}",
        f"{baja_rot['stock_actual'].sum():.0f}",
        f"${baja_rot['valor_inventario'].sum():,.0f}",
        f"{baja_rot['margen_pct'].mean():.1f}%"
    ]
}).T
print(comparacion_rotacion)

# Gráfico comparativo
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, col, titulo in zip(axes, ['valor_inventario', 'stock_actual'],
                            ['Valor en Inventario por Rotación', 'Stock Actual por Rotación']):
    sns.barplot(data=inv_ventas_ordenado, x='clase_rotacion', y=col,
                hue='clase_rotacion', palette='viridis', ax=ax, legend=False)
    ax.set_title(titulo, fontweight='bold')
    ax.set_xlabel('Clase de Rotación')
plt.tight_layout()
plt.show()
```

**Interpretación de negocio:** Los productos de baja rotación suelen tener mayor valor de inventario inmovilizado. Si un producto de baja rotación además es categoría C, es candidato directo a liquidación. Los de alta rotación que son categoría A requieren reorden frecuente y posiblemente descuentos por volumen con proveedores.

---

## 8. Matriz: Valor ABC × Rotación (Scatterplot con Cuadrantes)

```python
plt.figure(figsize=(12, 7))

colors = {'A': '#e74c3c', 'B': '#f39c12', 'C': '#3498db'}
for clase in ['A', 'B', 'C']:
    subset = inv_ventas_ordenado[inv_ventas_ordenado['clase_abc'] == clase]
    plt.scatter(subset['rotacion'], subset['valor_ventas'],
                c=colors[clase], label=f'Clase {clase}', s=subset['stock_actual']*3 + 50, alpha=0.7, edgecolors='black', linewidth=0.5)

plt.axhline(y=inv_ventas_ordenado['valor_ventas'].median(), color='gray', linestyle='--', alpha=0.5, label='Mediana valor ventas')
plt.axvline(x=inv_ventas_ordenado['rotacion'].median(), color='gray', linestyle='--', alpha=0.5, label='Mediana rotación')

# Anotar cuadrantes
plt.text(inv_ventas_ordenado['rotacion'].max()*0.8, inv_ventas_ordenado['valor_ventas'].max()*0.9,
         'ESTRELLAS\n(Alto valor, Alta rotación)', fontsize=9, bbox=dict(facecolor='green', alpha=0.1))
plt.text(inv_ventas_ordenado['rotacion'].max()*0.8, inv_ventas_ordenado['valor_ventas'].min(),
         'PERROS\n(Bajo valor, Alta rotación)', fontsize=9, bbox=dict(facecolor='orange', alpha=0.1))
plt.text(0, inv_ventas_ordenado['valor_ventas'].max()*0.9,
         'VACAS LECHERAS\n(Alto valor, Baja rotación)', fontsize=9, bbox=dict(facecolor='blue', alpha=0.1))
plt.text(0, inv_ventas_ordenado['valor_ventas'].min(),
         'LASTRE\n(Bajo valor, Baja rotación)', fontsize=9, bbox=dict(facecolor='red', alpha=0.1))

plt.xlabel('Rotación (veces/año)')
plt.ylabel('Valor de Ventas ($)')
plt.title('Matriz Valor × Rotación — Cuadrantes Estratégicos', fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()
```

**Interpretación de negocio:** La matriz de cuatro cuadrantes clasifica productos en:
- **Estrellas** (Alto valor + Alta rotación): Prioridad máxima, mantener stock óptimo.
- **Vacas lecheras** (Alto valor + Baja rotación): Productos caros que se venden lento. No sobre-stockear.
- **Perros** (Bajo valor + Alta rotación): Productos baratos que se venden rápido. Gestión eficiente de reorden.
- **Lastre** (Bajo valor + Baja rotación): Candidatos a descontinuación o liquidación.

---

## 9. Productos con Riesgo de Desabasto (A + Stock < 2× Demanda Diaria)

```python
inv_ventas_ordenado['demanda_diaria_estimada'] = inv_ventas_ordenado['cantidad_vendida'] / 365
inv_ventas_ordenado['dias_stock'] = inv_ventas_ordenado['stock_actual'] / inv_ventas_ordenado['demanda_diaria_estimada'].replace(0, np.nan)

riesgo_desabasto = inv_ventas_ordenado[
    (inv_ventas_ordenado['clase_abc'] == 'A') &
    (inv_ventas_ordenado['stock_actual'] < 2 * inv_ventas_ordenado['demanda_diaria_estimada'])
].sort_values('dias_stock')

print(f"Productos con RIESGO DE DESABASTO (A + stock < 2 días de demanda):")
print(riesgo_desabasto[['producto', 'categoria', 'stock_actual',
                          'demanda_diaria_estimada', 'dias_stock', 'valor_ventas']])

riesgo_perdida = riesgo_desabasto['valor_ventas'].sum()
print(f"\nValor de ventas en riesgo de pérdida: ${riesgo_perdida:,.0f}")
print(f"Pérdida estimada si se agotan por 1 semana: ${riesgo_perdida / 52:,.0f}")
```

**Interpretación de negocio:** El desabasto de productos A es crítico porque no solo se pierde la venta, sino que el cliente puede cambiar de proveedor para siempre. Mantener menos de 2 días de stock es una señal de alerta roja. La acción inmediata es colocar órdenes de compra exprés y contactar al proveedor para priorizar estos SKUs.

---

## 10. Productos Sobre-stockeados (C + Stock > 6 Meses de Venta)

```python
sobre_stock = inv_ventas_ordenado[
    (inv_ventas_ordenado['clase_abc'] == 'C') &
    (inv_ventas_ordenado['dias_stock'] > 180)  # 6 meses
].sort_values('dias_stock', ascending=False)

print(f"Productos SOBRE-STOCKEADOS (C + stock > 6 meses de venta):")
print(sobre_stock[['producto', 'categoria', 'stock_actual',
                     'demanda_diaria_estimada', 'dias_stock', 'valor_inventario']])

capital_inmovilizado = sobre_stock['valor_inventario'].sum()
print(f"\nCapital inmovilizado en productos sobre-stockeados: ${capital_inmovilizado:,.0f}")
print(f"Costo de oportunidad (12% anual): ${capital_inmovilizado * 0.12:,.0f}")
```

**Interpretación de negocio:** Tener productos C con más de 6 meses de inventario significa que el capital está inmovilizado sin retorno. Cada dólar invertido ahí podría estar generando rendimiento en otro lado. Se recomienda: liquidación con descuento, devolución a proveedor, donación fiscal o paquetes promocionales para liberar espacio y capital.

---

## 11. Tabla de Acciones Recomendadas por Cuadrante

```python
def generar_recomendacion(row):
    abc = row['clase_abc']
    rot = row['clase_rotacion']
    if abc == 'A' and rot == 'Alta':
        return 'Mantener stock óptimo. Reorden automático cada semana. Negociar precio con proveedor.'
    elif abc == 'A' and rot == 'Media':
        return 'Monitorear demanda. Aumentar promoción si rotación baja. Stock de seguridad alto.'
    elif abc == 'A' and rot == 'Baja':
        return 'Revisar precio. ¿Está muy caro? Evaluar promociones o empaques más pequeños.'
    elif abc == 'B' and rot == 'Alta':
        return 'Gestión estándar. Revisar si puede subir a categoría A con más promoción.'
    elif abc == 'B' and rot == 'Media':
        return 'Stock controlado. Reorden cada 30 días. Sin inversión extra.'
    elif abc == 'B' and rot == 'Baja':
        return 'Evaluar descontinuación. Liquidar stock actual con descuento moderado.'
    elif abc == 'C' and rot == 'Alta':
        return 'Producto popular pero barato. Automatizar reorden para evitar desabasto.'
    elif abc == 'C' and rot == 'Media':
        return 'Reducir stock al mínimo. Consolidar con otros productos C.'
    else:
        return 'LIQUIDAR INMEDIATAMENTE. Devolver a proveedor o donar.'

inv_ventas_ordenado['accion_recomendada'] = inv_ventas_ordenado.apply(generar_recomendacion, axis=1)

print("=== ACCIONES RECOMENDADAS POR PRODUCTO ===")
print(inv_ventas_ordenado[['producto', 'clase_abc', 'clase_rotacion', 'accion_recomendada']].to_string())

# Resumen de acciones
resumen_acciones = inv_ventas_ordenado.groupby('accion_recomendada').agg(
    productos=('sku', 'count'),
    valor_ventas=('valor_ventas', 'sum'),
    valor_inventario=('valor_inventario', 'sum')
).reset_index().sort_values('valor_ventas', ascending=False)
print("\n=== IMPACTO ECONÓMICO POR ACCIÓN ===")
print(resumen_acciones)
```

**Interpretación de negocio:** Cada cuadrante de la matriz requiere una estrategia distinta. No se puede gestionar un producto A-alta-rotación igual que uno C-baja-rotación. Asignar acciones específicas permite al equipo de compras ejecutar decisiones estandarizadas sin necesidad de revisión gerencial para cada caso, agilizando la operación.

---

## 12. Resumen Ejecutivo y Dashboard de Prioridades

```python
print("=" * 70)
print("DASHBOARD DE PRIORIDADES — GESTIÓN DE PRODUCTOS CRÍTICOS")
print("=" * 70)

prioridades = [
    ("🔴 CRÍTICO - Desabasto inminente", len(riesgo_desabasto), riesgo_desabasto['valor_ventas'].sum()),
    ("🟡 CRÍTICO - Productos A con bajo stock (<30d)", len(criticos), criticos['valor_ventas'].sum()),
    ("🟠 ATENCIÓN - Sobre-stock C (>6 meses)", len(sobre_stock), sobre_stock['valor_inventario'].sum()),
    ("🔵 MONITOREO - Productos B con rotación baja",
     len(inv_ventas_ordenado[(inv_ventas_ordenado['clase_abc']=='B') & (inv_ventas_ordenado['clase_rotacion']=='Baja')]),
     inv_ventas_ordenado[(inv_ventas_ordenado['clase_abc']=='B') & (inv_ventas_ordenado['clase_rotacion']=='Baja')]['valor_ventas'].sum()),
    ("🟢 ESTABLE - Productos estrella (A + alta rotación)",
     len(inv_ventas_ordenado[(inv_ventas_ordenado['clase_abc']=='A') & (inv_ventas_ordenado['clase_rotacion']=='Alta')]),
     inv_ventas_ordenado[(inv_ventas_ordenado['clase_abc']=='A') & (inv_ventas_ordenado['clase_rotacion']=='Alta')]['valor_ventas'].sum()),
]

print(f"{'Prioridad':<40} {'Qty':>6} {'Valor ($)':>15}")
print("-" * 65)
for nivel, qty, valor in prioridades:
    print(f"{nivel:<40} {qty:>6} {valor:>15,.0f}")

print("\n" + "=" * 70)
print("RECOMENDACIONES GENERALES")
print("=" * 70)

recomendaciones = [
    "1. ACTIVAR ÓRDENES DE COMPRA: Para todos los productos en riesgo de desabasto.",
    "2. PROGRAMA DE LIQUIDACIÓN: Para productos sobre-stockeados, descuento progresivo.",
    "3. REVISIÓN DE PROVEEDORES: Negociar mejores plazos para productos A de alta rotación.",
    "4. SISTEMA DE ALERTAS: Configurar notificación automática cuando stock < 30 días en productos A.",
    "5. REVISIÓN TRIMESTRAL: Re-clasificar ABC cada 3 meses con datos actualizados.",
    "6. POLÍTICA DE STOCK MUERTO: Dar de baja productos sin ventas en 6 meses.",
    "7. CAPACITACIÓN: Entrenar al equipo de compras en la metodología ABC.",
    "8. KPIs: Establecer metas de rotación mínima (6x año) y días stock máximos (90 días)."
]

for r in recomendaciones:
    print(r)

print("\n" + "=" * 70)
print(f"VALOR TOTAL BAJO GESTIÓN: ${inv_ventas_ordenado['valor_ventas'].sum():,.0f}")
print(f"CAPITAL INMOVILIZADO EN INVENTARIO: ${inv_ventas_ordenado['valor_inventario'].sum():,.0f}")
print(f"ROTACIÓN PROMEDIO: {inv_ventas_ordenado['rotacion'].mean():.1f}x año")
print("=" * 70)
```

**Interpretación de negocio:** Este dashboard de prioridades permite al gerente de logística saber exactamente dónde enfocar su atención. Las acciones están ordenadas por criticidad: primero evitar desabasto de productos clave, luego liberar capital de productos sobre-stockeados. La rotación promedio es un KPI maestro: mejorarla en 1 punto puede liberar cientos de miles de dólares en capital de trabajo.

---

## 📝 Ejercicios Propuestos

1. **ABC por sucursal:** Calcula la clasificación ABC para cada sucursal por separado. ¿Los productos prioritarios cambian según la sucursal? (Pista: agrupa por `sucursal` antes de calcular)

2. **Días para agotar:** Crea una columna `dias_para_agotar_simulado` dividiendo stock_actual entre demanda_diaria. ¿Qué productos se agotarán primero? Combínalo con ABC. (Pista: usa `stock_actual / demanda_diaria_prom`)

3. **Volumen óptimo de pedido:** Usa la fórmula EOQ (Economic Order Quantity) para calcular el tamaño óptimo de pedido de productos A. Supón costo de pedido=500, costo de mantener=0.15*año. (Pista: EOQ = sqrt(2*D*S/H))

4. **Concentración por categoría:** Agrupa la clasificación ABC por categoría. ¿Qué categoría tiene mayor proporción de productos A? (Pista: usa `groupby` con dos niveles)

5. **Simulación de quiebre de stock:** Si un proveedor falla 15 días, ¿qué productos se agotan primero? Modela el impacto en ingresos. (Pista: stock / demanda_diaria < 15)

---

## 📌 Resumen

| Hallazgo | Impacto |
|----------|---------|
| Productos A generan 80% del valor | Prioridad absoluta en disponibilidad |
| Productos C son mayoría pero aportan 5% | Liberar capital, liquidar o eliminar |
| Matriz valor×rotación revela 4 cuadrantes | Estrategia diferenciada por cuadrante |
| Riesgo de desabasto en productos A críticos | Órdenes de compra urgentes requeridas |
| Capital inmovilizado en sobre-stock C | Descuentos progresivos para liberar |
| Rotación promedio es KPI maestro | Meta: mejorar 1 punto la rotación anual |

## 🔗 Enlaces Relacionados
- [CP06 - Análisis de Estacionalidad](CP06-analisis-estacionalidad.md)
- [CP05 - Segmentación de Precios](CP05-segmentacion-precios.md)
- [CP02 - Dashboard de Inventario](CP02-analisis-inventario-basico.md)
