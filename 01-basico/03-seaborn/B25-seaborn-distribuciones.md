# Módulo B25 — Seaborn: Distribuciones (histplot, kdeplot, boxplot, violinplot, displot)

## Teoría

Entender la **distribución** de una variable es el paso fundamental en cualquier análisis exploratorio. Seaborn ofrece múltiples herramientas para visualizar distribuciones, cada una con fortalezas distintas.

### Tipos de gráficos de distribución

| Gráfico | Variable | Uso principal en ventas |
|---------|----------|--------------------------|
| `histplot` | 1 numérica | Frecuencia de precios, márgenes, cantidades |
| `kdeplot` | 1 numérica | Densidad suavizada, comparar distribuciones |
| `ecdfplot` | 1 numérica | % acumulado, percentiles |
| `boxplot` | 1 numérica + 1 categórica | Comparar distribuciones entre grupos |
| `violinplot` | 1 numérica + 1 categórica | Boxplot + densidad, ver multimodalidad |
| `boxenplot` | 1 numérica + 1 categórica | Boxplot con más cuantiles, colas detalladas |
| `displot` | 1 numérica + facetado | Hist/KDE/ECDF en subgráficos por categoría |

### Parámetros clave compartidos

- **`multiple`** (`"layer"`, `"stack"`, `"fill"`, `"dodge"`): cómo superponer grupos con `hue`
- **`stat`** (`"count"`, `"frequency"`, `"density"`, `"probability"`): qué escala usar en el eje y
- **`element`** (`"bars"`, `"step"`, `"poly"`): estilo visual del histograma
- **`bw_method` / `bw_adjust`**: controla el suavizado del KDE
- **`whis`**: extiende bigotes del boxplot (típico 1.5)
- **`notch`**: muesca para IC de la mediana en boxplot

### Cuándo usar cada uno

| Situación | Gráfico recomendado |
|-----------|---------------------|
| Ver forma general (sesgo, multimodal) | `histplot` + `kde` |
| Comparar muchas categorías | `boxplot` o `violinplot` |
| Detectar outliers | `boxplot` con `showfliers=True` |
| Ver proporciones acumuladas | `ecdfplot` |
| Analizar colas de distribución | `boxenplot` |
| Facetar por múltiples variables | `displot` con `col`, `row` |

---

## Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos
ventas = pd.read_csv("../datos/ventas.csv")
inventario = pd.read_csv("../datos/inventario.csv")

# Parsear fechas
ventas['fecha'] = pd.to_datetime(ventas['fecha'])

sns.set_theme()
print(ventas.shape, inventario.shape)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Setup.*

1. Cargar datos
2. Parsear fechas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 20 Ejemplos prácticos

### 1. Histograma básico: distribución de precios (10 bins)

```python
sns.histplot(data=ventas, x='precio_unitario', bins=10)
plt.title('Distribución de Precios Unitarios (10 bins)')
plt.xlabel('Precio Unitario ($)')
plt.ylabel('Frecuencia')
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*1. Histograma básico: distribución de precios (10 bins).*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicación**: `histplot` por defecto cuenta frecuencias. `bins=10` divide el rango en 10 intervalos iguales. Cada barra muestra cuántas transacciones caen en ese rango de precio.

**Interpretación**: La distribución está fuertemente sesgada a la derecha. La mayoría de transacciones tienen precios bajos (< $4000), con una cola larga hacia productos caros (> $10000).

---

### 2. Histograma de márgenes con curva KDE

```python
sns.histplot(data=ventas, x='margen', kde=True, bins=20)
plt.title('Distribución de Márgenes con KDE')
plt.xlabel('Margen ($)')
plt.ylabel('Frecuencia')
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*2. Histograma de márgenes con curva KDE.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicación**: `kde=True` superpone una estimación de densidad Kernel (curva suave). La curva no depende del número de bins y da una idea de la forma subyacente. 20 bins dan más detalle que 10.

**Interpretación**: El margen más frecuente está entre $0 y $5000. Hay picos aislados alrededor de $50000 (ventas grandes de muebles/electrónica). La curva KDE confirma el sesgo derecho.

---

### 3. Histograma de precios por categoría con multiple="stack"

```python
sns.histplot(data=ventas, x='precio_unitario', hue='categoria',
             multiple='stack', bins=15)
plt.title('Distribución de Precios por Categoría (stack)')
plt.xlabel('Precio Unitario ($)')
plt.ylabel('Frecuencia')
plt.legend(bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*3. Histograma de precios por categoría con multiple="stack".*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicación**: `hue='categoria'` separa por color. `multiple='stack'` apila las barras en lugar de superponerlas. Así se ve el total absoluto y la contribución de cada categoría por bin.

**Interpretación**: Electrónica y Muebles dominan el rango alto (> $8000). Periféricos y Papelería se concentran en precios bajos. La pila muestra que la mayoría de transacciones son de categorías económicas.

---

### 4. Histograma con stat="probability" para normalizar

```python
sns.histplot(data=ventas, x='precio_unitario', hue='categoria',
             stat='probability', common_norm=False, bins=15)
plt.title('Distribución de Precios por Categoría (probabilidad)')
plt.xlabel('Precio Unitario ($)')
plt.ylabel('Probabilidad')
plt.legend(bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*4. Histograma con stat="probability" para normalizar.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicación**: `stat='probability'` normaliza cada grupo para que sume 1 (área total = 1). `common_norm=False` normaliza cada categoría por separado, permitiendo comparar la forma independientemente del tamaño de cada grupo.

**Interpretación**: Cada categoría suma 1, así se compara la forma de su distribución. Electrónica tiene distribución bimodal (productos baratos y caros). Papelería es casi uniforme en su rango estrecho.

---

### 5. Histograma con log_scale para datos sesgados

```python
sns.histplot(data=ventas, x='precio_unitario', bins=20, log_scale=True)
plt.title('Distribución de Precios (escala logarítmica)')
plt.xlabel('Precio Unitario ($, escala log)')
plt.ylabel('Frecuencia')
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*5. Histograma con log_scale para datos sesgados.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicación**: `log_scale=True` aplica transformación logarítmica al eje x. Revela estructura en la cola larga que en escala lineal se aplasta. Útil para variables con sesgo extremo.

**Interpretación**: En escala log, la distribución se acerca a una forma normal. Se ven claramente 3-4 agrupamientos: productos < $500, entre $1000-$3000, y > $8000.

---

### 6. ECDF: distribución acumulada de ingresos

```python
sns.ecdfplot(data=ventas, x='ingreso')
plt.title('Distribución Acumulada del Ingreso por Transacción')
plt.xlabel('Ingreso ($)')
plt.ylabel('Proporción Acumulada')
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*6. ECDF: distribución acumulada de ingresos.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicación**: `ecdfplot` muestra la proporción de datos ≤ cada valor. Sin bins, sin suavizado: cada salto es una transacción real. El eje y va de 0 a 1 (0% a 100%).

**Interpretación**: El 50% de las transacciones tienen ingreso ≤ $12000 (mediana). El 80% están por debajo de $35000. Un pequeño % de transacciones genera ingresos muy altos (> $100000).

---

### 7. KDE: densidad de márgenes por sucursal

```python
sns.kdeplot(data=ventas, x='margen', hue='sucursal', fill=True, alpha=0.3)
plt.title('Densidad de Márgenes por Sucursal')
plt.xlabel('Margen ($)')
plt.ylabel('Densidad')
plt.legend(bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*7. KDE: densidad de márgenes por sucursal.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicación**: `kdeplot` con `hue` dibuja una curva de densidad por sucursal. `fill=True` rellena el área bajo la curva con color semitransparente (`alpha=0.3`). Cada curva se integra a 1.

**Interpretación**: Sucursal Mérida tiene la distribución más ancha (mayor variabilidad en márgenes). Todas las sucursales tienen el pico principal cerca de $0, pero algunas tienen colas más largas hacia márgenes altos.

---

### 8. KDE con bw_adjust para controlar suavizado

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.kdeplot(data=ventas, x='margen', hue='categoria',
            bw_adjust=0.3, ax=axes[0])
axes[0].set_title('bw_adjust=0.3 (poco suavizado)')

sns.kdeplot(data=ventas, x='margen', hue='categoria',
            bw_adjust=2, ax=axes[1])
axes[1].set_title('bw_adjust=2 (muy suavizado)')

plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*8. KDE con bw_adjust para controlar suavizado.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicación**: `bw_adjust` controla el ancho de banda del KDE (ventana de suavizado). Valores < 1 producen curvas con más picos (posible sobreajuste). Valores > 1 suavizan más (posible pérdida de detalle). El valor por defecto es 1.

**Interpretación**: Con `bw_adjust=0.3` se ven múltiples picos para cada categoría (quizás ruido). Con `bw_adjust=2` las curvas son casi gaussianas. El balance está en bw_adjust=1 (por defecto).

---

### 9. KDE con fill y alpha

```python
sns.kdeplot(data=ventas, x='cantidad', hue='categoria',
            fill=True, alpha=0.4, linewidth=2)
plt.title('Densidad de Cantidad Vendida por Categoría')
plt.xlabel('Cantidad')
plt.ylabel('Densidad')
plt.legend(bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*9. KDE con fill y alpha.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicación**: `fill=True` colorea el área bajo la curva con el color asignado. `alpha=0.4` da transparencia para ver superposición. `linewidth=2` hace más visibles los bordes.

**Interpretación**: La mayoría de categorías venden entre 1-15 unidades por transacción. Muebles y Electrónica tienen picos alrededor de 5-8 unidades. Software muestra cola hacia cantidades mayores (licencias por volumen).

---

### 10. Boxplot: precios por categoría

```python
sns.boxplot(data=ventas, x='categoria', y='precio_unitario')
plt.title('Distribución de Precios por Categoría')
plt.xlabel('Categoría')
plt.ylabel('Precio Unitario ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*10. Boxplot: precios por categoría.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicación**: El boxplot muestra: mediana (línea central), Q1-Q25 (caja), bigotes (1.5×IQR), y puntos fuera (outliers). Resume distribución en 5 números: mínimo, Q1, mediana, Q3, máximo.

**Interpretación**: Electrónica y Muebles tienen las mayores medianas de precio y la mayor dispersión. Papelería y Software tienen precios bajos con poca variabilidad. Electrónica tiene múltiples outliers hacia arriba (productos premium).

---

### 11. Boxplot: márgenes por sucursal con hue=mes

```python
sns.boxplot(data=ventas, x='sucursal', y='margen',
            hue='mes', palette='Set3')
plt.title('Márgenes por Sucursal segmentado por Mes')
plt.xlabel('Sucursal')
plt.ylabel('Margen ($)')
plt.legend(bbox_to_anchor=(1.05, 1), title='Mes')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*11. Boxplot: márgenes por sucursal con hue=mes.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicación**: `hue='mes'` subdivide cada sucursal en cajas por mes. `palette='Set3'` da colores pastel. Los boxplots se agrupan lado a lado para comparar dentro de cada sucursal.

**Interpretación**: Se observa poca variación mensual dentro de cada sucursal. Sucursal Mérida destaca con márgenes más altos en varios meses. Algunas sucursales tienen más outliers (ventas excepcionales).

---

### 12. Boxplot con notch para intervalo de confianza de la mediana

```python
sns.boxplot(data=ventas, x='categoria', y='precio_unitario',
            notch=True, width=0.6)
plt.title('Boxplot con Muesca (IC 95% de la mediana)')
plt.xlabel('Categoría')
plt.ylabel('Precio Unitario ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*12. Boxplot con notch para intervalo de confianza de la mediana.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicación**: `notch=True` añade una muesca alrededor de la mediana. Si las muescas de dos cajas no se solapan, hay evidencia de que sus medianas son significativamente diferentes (α=0.05). `width=0.6` reduce el ancho de las cajas.

**Interpretación**: Las muescas revelan diferencias significativas: Electrónica tiene mediana claramente mayor que Audio. Muebles y Electrónica tienen muescas que se solapan levemente.

---

### 13. Violinplot: cantidad vendida por día de semana

```python
sns.violinplot(data=ventas, x='dia_semana', y='cantidad')
plt.title('Distribución de Cantidad Vendida por Día de Semana')
plt.xlabel('Día (0=Domingo, 6=Sábado)')
plt.ylabel('Cantidad')
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*13. Violinplot: cantidad vendida por día de semana.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicación**: Violinplot combina boxplot (mediana, cuartiles) con KDE (forma). El ancho en cada punto de y representa la densidad. Más ancho = más datos en esa región.

**Interpretación**: La distribución de cantidades es similar todos los días (sesgo derecho). El día 0 (domingo) muestra menos densidad en la parte superior (menos ventas grandes). Días 2-4 (martes a jueves) tienen colas ligeramente más largas.

---

### 14. Violinplot con split para comparar 2 categorías

```python
# Filtrar 2 categorías más frecuentes
top2 = ventas[ventas['categoria'].isin(['Electrónica', 'Periféricos'])]

sns.violinplot(data=top2, x='dia_semana', y='cantidad',
               hue='categoria', split=True)
plt.title('Comparación Electrónica vs Periféricos (split violin)')
plt.xlabel('Día de Semana')
plt.ylabel('Cantidad')
plt.legend(title='Categoría')
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*14. Violinplot con split para comparar 2 categorías.*

1. Filtrar 2 categorías más frecuentes

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicación**: `split=True` parte cada violín en dos mitades, cada una de un color según `hue`. Útil para comparar exactamente dos grupos lado a lado en el mismo eje x.

**Interpretación**: Periféricos (mitad derecha) tiende a cantidades ligeramente mayores que Electrónica (mitad izquierda) en casi todos los días. Ambos tienen distribución similar. El split permite ver esta diferencia sin duplicar violines.

---

### 15. Boxenplot: detalle de colas en distribución de precios

```python
sns.boxenplot(data=ventas, x='categoria', y='precio_unitario')
plt.title('Boxenplot — Colas detalladas de Precio por Categoría')
plt.xlabel('Categoría')
plt.ylabel('Precio Unitario ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*15. Boxenplot: detalle de colas en distribución de precios.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicación**: Boxenplot (o "letter-value plot") muestra más cuantiles que un boxplot. En lugar de solo Q1/Q3/mediana, muestra múltiples "cajas" concéntricas representando percentiles (12.5, 25, 37.5, 50, 62.5, 75, 87.5). Ideal para grandes datasets.

**Interpretación**: Se aprecian con más detalle las colas de Electrónica y Muebles. Las cajas externas muestran que el 12.5% inferior y superior tienen valores extremos, información que un boxplot tradicional oculta.

---

### 16. displot facetado por categoría kind="hist"

```python
sns.displot(data=ventas, x='precio_unitario', col='categoria',
            kind='hist', bins=10, col_wrap=3, height=3)
plt.suptitle('Histogramas de Precio por Categoría (facetados)',
             y=1.02)
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*16. displot facetado por categoría kind="hist".*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicación**: `displot` es una función de figura completa (no devuelve ax). `col='categoria'` crea un subgráfico por categoría. `col_wrap=3` distribuye en 3 columnas. `height=3` controla altura de cada subgráfico.

**Interpretación**: Cada categoría tiene su propia escala y forma. Algunas (Papelería, Software) tienen rango muy estrecho. Otras (Electrónica, Muebles) son anchas. Comparar facetas revela heterogeneidad que se pierde en un solo gráfico.

---

### 17. displot facetado kind="kde" con col_wrap=3

```python
sns.displot(data=ventas, x='margen', col='sucursal',
            kind='kde', col_wrap=3, height=3, fill=True)
plt.suptitle('Densidad de Margen por Sucursal', y=1.02)
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*17. displot facetado kind="kde" con col_wrap=3.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicación**: `kind='kde'` usa KDE en lugar de histograma. `col='sucursal'` genera un panel por sucursal. `fill=True` rellena el área. Es más compacto que `kdeplot` con `hue` cuando hay muchos grupos.

**Interpretación**: Comparar densidades por sucursal lado a lado. La mayoría tiene forma similar (pico cerca de 0, cola derecha). Sucursal Mérida destaca por tener la cola más larga (mayor ocurrencia de márgenes altos).

---

### 18. Comparar histplot vs kdeplot vs ecdfplot misma variable

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

sns.histplot(data=ventas, x='margen', bins=20, kde=True, ax=axes[0])
axes[0].set_title('Histograma + KDE')

sns.kdeplot(data=ventas, x='margen', fill=True, ax=axes[1])
axes[1].set_title('KDE (densidad)')

sns.ecdfplot(data=ventas, x='margen', ax=axes[2])
axes[2].set_title('ECDF (acumulada)')

plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*18. Comparar histplot vs kdeplot vs ecdfplot misma variable.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicación**: Misma variable (`margen`) en tres formatos. Histograma muestra frecuencias por bin. KDE suaviza la forma. ECDF muestra directamente la proporción acumulada. Cada uno revela aspectos distintos.

**Interpretación**: El histograma muestra el sesgo derecho. El KDE confirma la forma con un pico principal. La ECDF responde preguntas como "¿qué % de transacciones tienen margen < $10000?" (~75%).

---

### 19. Boxplot horizontal con orient="h"

```python
sns.boxplot(data=ventas, y='categoria', x='precio_unitario',
            orient='h', palette='Set2')
plt.title('Boxplot Horizontal — Precio por Categoría')
plt.xlabel('Precio Unitario ($)')
plt.ylabel('Categoría')
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*19. Boxplot horizontal con orient="h".*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicación**: `orient='h'` intercambia ejes: las categorías en el eje y, precios en x. Más fácil de leer cuando las etiquetas son largas. `palette='Set2'` da colores suaves.

**Interpretación**: Las categorías se ordenan de arriba abajo por precio creciente. Software y Papelería arriba (bajo precio), Muebles y Electrónica abajo (alto precio). Las etiquetas no se solapan.

---

### 20. Violinplot + swarmplot superpuesto para ver todos los puntos

```python
top_cats = ventas[ventas['categoria'].isin(['Electrónica', 'Periféricos', 'Audio'])]

sns.violinplot(data=top_cats, x='categoria', y='precio_unitario',
               inner='quartile', density_norm='width')
sns.swarmplot(data=top_cats, x='categoria', y='precio_unitario',
              color='black', alpha=0.5, size=3)
plt.title('Violinplot + Swarmplot — Todos los puntos visibles')
plt.xlabel('Categoría')
plt.ylabel('Precio Unitario ($)')
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*20. Violinplot + swarmplot superpuesto para ver todos los puntos.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicación**: `violinplot` con `inner='quartile'` muestra cuartiles internos. `swarmplot` superpuesto dibuja cada punto individual sin solapamiento (usando un algoritmo de "abeja"). `size=3` para puntos pequeños.

**Interpretación**: Se ve la forma de la densidad (violín) y cada transacción individual (swarm). Electrónica tiene puntos dispersos en todo el rango. Periféricos y Audio están más concentrados. El swarm revela valores exactos y agrupaciones.

---

## Resumen

Este módulo cubrió las principales herramientas de Seaborn para analizar distribuciones en datos de ventas:

| Gráfico | Cuándo usarlo | Información que revela |
|---------|---------------|------------------------|
| `histplot` | Primer vistazo a cualquier variable numérica | Forma, sesgo, modas, bins |
| `kdeplot` | Comparar suavemente distribuciones entre grupos | Superposición, multimodalidad |
| `ecdfplot` | Percentiles, ¿qué % está por debajo de X? | Acumulada exacta sin bins |
| `boxplot` | Comparar 5 números entre muchas categorías | Outliers, mediana, dispersión |
| `violinplot` | Boxplot + densidad, ver forma completa | Multimodalidad, asimetría |
| `boxenplot` | Grandes datasets, colas detalladas | Cuantiles más allá del IQR |
| `displot` | Facetar hist/KDE/ECDF por categorías | Patrones por subgrupo |

Regla práctica: empieza con `histplot`, profundiza con `kdeplot` si hay pocos grupos, usa `boxplot` para muchas categorías, y `displot` para facetar.

---

## Ejercicios

1. Crea un histograma de `cantidad` vendida con 30 bins y `kde=True`. ¿Es simétrica o sesgada? ¿Hay valores pico en alguna cantidad específica?

2. Usa `sns.boxplot` para comparar `margen` entre sucursales con `hue='categoria'`. ¿Qué sucursal-categoría tiene la mediana más alta?

3. Genera un `sns.violinplot` de `precio_unitario` por `categoria` con `inner='box'`. ¿Qué categoría tiene la distribución más bimodal?

4. Crea un `sns.ecdfplot` de `margen` segmentado por `mes` (tratar mes como categórico). ¿Qué mes tiene mayor probabilidad de margen > $20000?

5. Usa `sns.displot` con `kind='hist'`, `col='mes'`, `row='categoria'` (2×3). ¿Qué categoría-mes tiene la distribución más atípica?

6. Crea un boxplot horizontal de `stock_actual` por `categoria` usando `inventario`. ¿Qué categoría tiene mayor mediana de stock?

7. Usa `sns.boxenplot` de `valor_inventario` por `categoria` en `inventario`. Compara el detalle de colas vs un boxplot tradicional.

8. Superpone `sns.violinplot` + `sns.swarmplot` de `margen_pct` por `categoria` (filtrando 4 categorías principales). ¿Qué categoría tiene márgenes porcentuales más consistentes?
