# I11 — Gráficos Avanzados en Seaborn

## 1. Introducción Teórica

Seaborn ofrece gráficos especializados para análisis exploratorio avanzado: mapas de calor con clustering jerárquico, densidades 2D, distribuciones acumulativas, boxplots enriquecidos, y visualizaciones conjuntas. Estos gráficos son ideales para detectar patrones en datos de ventas, compras e inventarios.

### Gráficos cubiertos:

- **`sns.clustermap`**: Heatmap con dendrogramas (agrupa filas/columnas similares)
- **`sns.heatmap`**: Mapas de calor con máscaras triangulares, anotaciones, y centros divergentes
- **`sns.kdeplot` 2D**: Densidad bidimensional con contornos y relleno
- **`sns.ecdfplot`**: Distribución acumulativa empírica
- **`sns.boxenplot`**: Boxplot enriquecido (letter-value plot) que muestra más cuantiles
- **`sns.regplot` + `sns.residplot`**: Regresión con análisis de residuos
- **`sns.lineplot`**: Líneas con bandas de confianza o desviación estándar
- **`sns.scatterplot` con size y hue continuo**: Burbujas con 3 dimensiones
- **`sns.jointplot` kind="hex"**: Hexbin para grandes volúmenes
- **`FacetGrid` multi-map**: Múltiples tipos de gráfico superpuestos

---

## 2. Ejemplos Prácticos

### Ejemplo 1: clustermap — Agrupar productos similares por perfil de ventas

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

ventas = pd.read_csv("../../datos/ventas.csv")
inventario = pd.read_csv("../../datos/inventario.csv")

# Crear tabla de productos × meses con ingreso total
tabla = ventas.pivot_table(values="ingreso", index="producto",
                           columns="mes", aggfunc=sum, fill_value=0)

# clustermap con z-score (normaliza por fila) y cmap divergente
g = sns.clustermap(tabla, z_score=1, cmap="vlag",
                   figsize=(10, 8), method="ward",
                   metric="euclidean",
                   dendrogram_ratio=(0.1, 0.2),
                   cbar_pos=(0.02, 0.8, 0.03, 0.2))
g.fig.suptitle("Clustering de Productos por Patrón de Ventas Mensual (z_score)",
               y=1.02, fontsize=14)
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 1: clustermap — Agrupar productos similares por perfil de ventas.*

1. Crear tabla de productos × meses con ingreso total
2. clustermap con z-score (normaliza por fila) y cmap divergente

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: clustermap con col_colors — Colorear columnas por categoría

```python
# Mapa de colores para cada mes según categoría
colores_mes = {1: "red", 2: "blue", 3: "green", 4: "orange",
               5: "purple", 6: "brown", 7: "pink", 8: "gray",
               9: "olive", 10: "cyan", 11: "magenta", 12: "yellow"}
# Asignar un color por mes
lookup = pd.Series(colores_mes)
col_colors = lookup[tabla.columns]

g = sns.clustermap(tabla, z_score=1, cmap="vlag",
                   col_colors=col_colors,  # Colorea cabeceras de columna
                   figsize=(11, 8), method="average")
g.fig.suptitle("Clustermap con col_colors por mes", y=1.02)
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 2: clustermap con col_colors — Colorear columnas por categoría.*

1. Mapa de colores para cada mes según categoría
2. Asignar un color por mes

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: heatmap con mask triangular — Correlaciones (solo mitad inferior)

```python
numéricas = ventas.select_dtypes(include=[np.number])
corr = numéricas.corr()

# Crear máscara triangular superior
mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title("Matriz de Correlación (mitad inferior)")
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 3: heatmap con mask triangular — Correlaciones (solo mitad inferior).*

1. Crear máscara triangular superior

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: heatmap con center=0 y cmap divergente

```python
corr = ventas.select_dtypes(include=[np.number]).corr()

sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0,  # 0 = blanco (neutral)
            vmin=-1, vmax=1,  # correlación va de -1 a 1
            square=True, linewidths=0.5)
plt.title("Correlaciones: Positivas (rojo) vs Negativas (azul)")
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 4: heatmap con center=0 y cmap divergente.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: heatmap con annot_kws — Números legibles

```python
corr = ventas.select_dtypes(include=[np.number]).corr()

sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square=True,
            annot_kws={"size": 9, "weight": "bold"},  # Texto en negrita
            cbar_kws={"label": "Coeficiente de Correlación", "shrink": 0.8})
plt.title("Correlaciones con anotaciones personalizadas")
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 5: heatmap con annot_kws — Números legibles.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: kdeplot 2D — Densidad conjunta precio-margen

```python
sns.kdeplot(x=ventas["precio_unitario"], y=ventas["margen"],
            levels=10,  # Número de contornos
            fill=True,  # Rellenar áreas
            thresh=0.05,  # Ignorar regiones con densidad < 5% del pico
            cmap="viridis", cbar=True)
plt.title("Densidad 2D: Precio vs Margen")
plt.xlabel("Precio Unitario ($)")
plt.ylabel("Margen ($)")
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 6: kdeplot 2D — Densidad conjunta precio-margen.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: kdeplot 2D con hue — Densidad por categoría

```python
sns.kdeplot(x="precio_unitario", y="margen", hue="categoria",
            data=ventas, alpha=0.6, thresh=0.1,
            palette="Set2", fill=True)
plt.title("Densidad 2D Precio-Margen por Categoría")
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 7: kdeplot 2D con hue — Densidad por categoría.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: ecdfplot — Distribución acumulativa de ingresos

```python
# ECDF: Empirical Cumulative Distribution Function
sns.ecdfplot(data=ventas, x="ingreso", hue="categoria",
             stat="proportion",  # También: "count", "percent"
             complementary=False,  # False = P(X<=x), True = P(X>x)
             palette="Set2", linewidth=2)
plt.title("Distribución Acumulativa de Ingresos por Categoría")
plt.xlabel("Ingreso ($)")
plt.ylabel("Proporción Acumulada")
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 8: ecdfplot — Distribución acumulativa de ingresos.*

1. ECDF: Empirical Cumulative Distribution Function

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: boxenplot — Distribución detallada de precios

```python
# Boxenplot (letter-value plot): muestra más cuantiles que boxplot
# Especialmente útil para datasets grandes
sns.boxenplot(x="categoria", y="precio_unitario", data=ventas,
              k_depth="trustworthy",  # Método para profundidad de cajas
              outlier_prop=0.01,  # Proporción esperada de outliers
              palette="Set2")
plt.title("Distribución de Precios por Categoría (Boxenplot)")
plt.xlabel("Categoría")
plt.ylabel("Precio Unitario ($)")
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

*Ejemplo 9: boxenplot — Distribución detallada de precios.*

1. Boxenplot (letter-value plot): muestra más cuantiles que boxplot
2. Especialmente útil para datasets grandes

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: regplot con logx — Relación log-lineal

```python
# logx=True transforma eje x a escala logarítmica
# Útil cuando la relación es multiplicativa (ej: precio vs cantidad)
sns.regplot(x="precio_unitario", y="cantidad", data=ventas,
            logx=True, scatter_kws={"alpha": 0.4},
            line_kws={"color": "red", "linewidth": 2})
plt.title("Regresión log-lineal: Precio vs Cantidad")
plt.xlabel("Precio Unitario ($) [escala log]")
plt.ylabel("Cantidad")
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 10: regplot con logx — Relación log-lineal.*

1. logx=True transforma eje x a escala logarítmica
2. Útil cuando la relación es multiplicativa (ej: precio vs cantidad)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: residplot — Patrones en residuos

```python
# residplot: gráfico de residuos vs predictor
# lowess=True añade curva suavizada para ver patrones
sns.residplot(x="precio_unitario", y="margen", data=ventas,
              lowess=True,  # Curva suavizada de residuos
              scatter_kws={"alpha": 0.4},
              line_kws={"color": "red", "linewidth": 2})
plt.title("Residuos de Regresión: Precio vs Margen (con lowess)")
plt.xlabel("Precio Unitario ($)")
plt.ylabel("Residuos")
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 11: residplot — Patrones en residuos.*

1. residplot: gráfico de residuos vs predictor
2. lowess=True añade curva suavizada para ver patrones

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: lineplot con ci="sd" — Desviación estándar

```python
# Por defecto ci=95 (intervalo de confianza bootstrap)
# ci="sd" muestra ±1 desviación estándar
sns.lineplot(x="mes", y="ingreso", hue="categoria", data=ventas,
             ci="sd",  # Banda = media ± 1 desviación estándar
             estimator=sum,  # Agrega por mes
             palette="Set2", linewidth=2)
plt.title("Tendencia Mensual de Ingresos por Categoría (ci='sd')")
plt.xlabel("Mes")
plt.ylabel("Ingreso Total ($)")
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 12: lineplot con ci="sd" — Desviación estándar.*

1. Por defecto ci=95 (intervalo de confianza bootstrap)
2. ci="sd" muestra ±1 desviación estándar

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: scatterplot con size y hue continuo

```python
# size = 3ª dimensión, hue = 4ª dimensión (color continuo)
sns.scatterplot(x="precio_unitario", y="cantidad",
                size="margen",  # Tamaño del punto = margen
                hue="margen",   # Color del punto = margen
                data=ventas, alpha=0.6,
                sizes=(20, 300),  # Rango de tamaño
                palette="viridis", legend="brief")
plt.title("Precio vs Cantidad (tamaño y color = margen)")
plt.xlabel("Precio Unitario ($)")
plt.ylabel("Cantidad")
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 13: scatterplot con size y hue continuo.*

1. size = 3ª dimensión, hue = 4ª dimensión (color continuo)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: jointplot kind="hex" — Hexbin para grandes datos

```python
# Hexbin: divide el plano en hexágonos, color = densidad
# Ideal para grandes volúmenes de datos (evita overplotting)
sns.jointplot(x="precio_unitario", y="margen", data=ventas,
              kind="hex", gridsize=20,  # Tamaño de hexágonos
              cmap="Blues", marginal_kws={"bins": 20, "kde": True})
plt.suptitle("Jointplot Hexbin: Precio vs Margen", y=1.02)
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 14: jointplot kind="hex" — Hexbin para grandes datos.*

1. Hexbin: divide el plano en hexágonos, color = densidad
2. Ideal para grandes volúmenes de datos (evita overplotting)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: FacetGrid con múltiples map (hist + kde)

```python
# Superponer histograma y kde en cada panel facetado
g = sns.FacetGrid(ventas, col="categoria", col_wrap=3, height=3, aspect=1.3)
g.map(sns.histplot, "margen", stat="density", alpha=0.5, bins=20, color="steelblue")
g.map(sns.kdeplot, "margen", color="red", linewidth=2)
g.set_titles(col_template="{col_name}")
g.set_axis_labels("Margen ($)", "Densidad")
plt.suptitle("Histograma + KDE de Margen por Categoría", y=1.02)
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 15: FacetGrid con múltiples map (hist + kde).*

1. Superponer histograma y kde en cada panel facetado

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 16: heatmap con cbar_kws — Personalizar barra de color

```python
corr = ventas.select_dtypes(include=[np.number]).corr()

sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, linewidths=0.5,
            cbar_kws={"label": "Correlación", "shrink": 0.8,
                       "orientation": "vertical"})
plt.title("Heatmap con barra de color personalizada")
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 16: heatmap con cbar_kws — Personalizar barra de color.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 17: clustermap con figsize y dendrogram_ratio

```python
from matplotlib.colors import LinearSegmentedColormap

# Crear tabla de margen por producto y mes
tabla_margen = ventas.pivot_table(values="margen", index="producto",
                                   columns="mes", aggfunc=sum, fill_value=0)

g = sns.clustermap(tabla_margen, z_score=1,
                   cmap="RdBu_r", center=0,
                   figsize=(12, 10),
                   dendrogram_ratio=(0.1, 0.2),  # 10% ancho dendro filas, 20% alto dendro cols
                   method="ward", metric="euclidean",
                   linewidths=0.5)
g.fig.suptitle("Clustermap: Margen por Producto y Mes (z-score)", y=1.02, fontsize=14)
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 17: clustermap con figsize y dendrogram_ratio.*

1. Crear tabla de margen por producto y mes

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 18: lmplot con robust=True y order=2 — Regresión robusta polinómica

```python
# robust=True: usa estimador M (menos sensible a outliers)
# order=2: regresión polinómica de grado 2 (cuadrática)
sns.lmplot(x="precio_unitario", y="margen", data=ventas,
           robust=True, order=2,
           scatter_kws={"alpha": 0.4, "s": 30},
           line_kws={"color": "darkred", "linewidth": 2},
           height=5, aspect=1.5)
plt.title("Regresión Robusta Polinómica (order=2): Precio vs Margen")
plt.xlabel("Precio Unitario ($)")
plt.ylabel("Margen ($)")
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 18: lmplot con robust=True y order=2 — Regresión robusta polinómica.*

1. robust=True: usa estimador M (menos sensible a outliers)
2. order=2: regresión polinómica de grado 2 (cuadrática)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 3. Resumen

| Gráfico | Propósito | Aplicación |
|---------|-----------|------------|
| `clustermap` | Agrupar filas/columnas por similitud | Segmentar productos por patrón de ventas |
| `heatmap` + mask | Correlaciones sin duplicados | Relaciones entre métricas |
| `heatmap` + center=0 | Separar positivo/negativo | Análisis de márgenes vs costos |
| `kdeplot` 2D | Densidad conjunta | Clusters precio-margen |
| `ecdfplot` | Distribución acumulativa | % de ventas bajo cierto precio |
| `boxenplot` | Distribución detallada | Comparar precios entre categorías |
| `regplot` + `logx` | Relaciones log-lineales | Ley de demanda (precio→cantidad) |
| `residplot` | Patrones en residuos | Validar supuestos de regresión |
| `lineplot` `ci="sd"` | Tendencia con variabilidad | Estacionalidad de ventas |
| `scatterplot` + size + hue | 4 dimensiones en 2D | Burbujas rentabilidad-volumen |
| `jointplot` `kind="hex"` | Grandes volúmenes | Evitar overplotting |
| `FacetGrid` multi-map | Superponer tipos | Histograma + KDE por grupo |
| `lmplot` robust + order | Regresión resistente | Relaciones curvilíneas |

---

## 4. Ejercicios Propuestos

**Ejercicio 1:** Carga el inventario y crea un `clustermap` de productos por columnas numéricas (`stock_actual`, `precio`, `costo`, `demanda_diaria_prom`). Normaliza con `z_score=1`. ¿Qué productos se agrupan juntos?

**Ejercicio 2:** Calcula la matriz de correlación de todas las columnas numéricas de ventas. Crea un heatmap triangular inferior con `cmap="coolwarm"`, `center=0`, y anotaciones en negrita. ¿Cuál es la correlación más fuerte?

**Ejercicio 3:** Usa `sns.kdeplot` 2D con `hue="categoria"`, `fill=True`, y `alpha=0.4` para visualizar la densidad conjunta de `precio_unitario` y `cantidad` para cada categoría en el DataFrame de ventas.

**Ejercicio 4:** Crea un `boxenplot` de `margen` por `sucursal` con `k_depth="trustworthy"`. Compara visualmente con un `boxplot` estándar — ¿qué información adicional muestra el boxenplot?

**Ejercicio 5:** Usa `sns.ecdfplot` para mostrar la distribución acumulativa de `precio_unitario` coloreada por `categoria`. Usa `complementary=True` para mostrar la curva complementaria (P(X > x)). ¿Qué categoría tiene los precios más altos?

**Ejercicio 6:** Crea un scatterplot con `size="cantidad"` y `hue="margen"` para visualizar 4 dimensiones: precio_unitario en x, ingreso en y, tamaño = cantidad, color = margen. Usa `palette="plasma"`.

**Ejercicio 7:** Usa `sns.jointplot(kind="hex", gridsize=15)` para graficar `precio_unitario` vs `cantidad` del DataFrame de ventas. Luego compara con `kind="scatter"`. ¿Cuál prefieres para 1331 puntos y por qué?

**Ejercicio 8:** Del inventario, grafica `precio` vs `stock_actual` con `sns.regplot`. Añade `logx=True`. ¿Tiene sentido usar escala logarítmica para el precio en este caso? ¿Por qué?

---

*Fin del documento I11 — Gráficos Avanzados en Seaborn*
