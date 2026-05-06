# AP04 — Cheatsheet Seaborn

## 1. Configuración y Estilos

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Temas disponibles
sns.set_theme()                     # tema por defecto (v0.11+)
sns.set_style("whitegrid")          # whitegrid, darkgrid, white, dark, ticks
sns.set_style("ticks")
sns.set_style({"axes.facecolor": "white"})

# Paletas de colores
sns.set_palette("deep")             # deep, muted, bright, pastel, dark, colorblind
sns.set_palette("husl")             # husl, hls
sns.set_palette("Set2")             # Set1, Set2, Set3, Paired, tab10
sns.set_palette("Blues")            # secuenciales: Blues, Reds, Greens
sns.set_palette("viridis")          # viridis, plasma, inferno, magma
sns.color_palette("Blues", 3)       # seleccionar n colores de paleta

# Contexto (tamaño de texto y elementos)
sns.set_context("paper")            # paper, notebook, talk, poster
sns.set_context("talk", font_scale=1.5)

# sns.set (todo en uno)
sns.set(style="whitegrid", palette="muted", font_scale=1.2)

# Desactivar warnings de seaborn
sns.set(color_codes=True)
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*1. Configuración y Estilos.*

1. Temas disponibles
2. Paletas de colores
3. Contexto (tamaño de texto y elementos)
4. sns.set (todo en uno)
5. Desactivar warnings de seaborn

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 2. Gráficos de Distribución

```python
# Histograma + KDE
sns.histplot(data=df, x="precio")
sns.histplot(data=df, x="precio", bins=30)
sns.histplot(data=df, x="precio", kde=True)
sns.histplot(data=df, x="precio", hue="categoria")

# KDE (Kernel Density Estimate)
sns.kdeplot(data=df, x="precio")
sns.kdeplot(data=df, x="precio", fill=True)
sns.kdeplot(data=df, x="precio", hue="categoria", common_norm=False)
sns.kdeplot(data=df, x="precio", y="descuento")  # 2D KDE

# ECDF
sns.ecdfplot(data=df, x="precio")
sns.ecdfplot(data=df, x="precio", hue="categoria")

# Rug plot
sns.rugplot(data=df, x="precio")

# Distribución conjunta (bivariada)
sns.jointplot(data=df, x="precio", y="ventas")
sns.jointplot(data=df, x="precio", y="ventas", kind="hex")
sns.jointplot(data=df, x="precio", y="ventas", kind="kde")
sns.jointplot(data=df, x="precio", y="ventas", kind="reg")
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*2. Gráficos de Distribución.*

1. Histograma + KDE
2. KDE (Kernel Density Estimate)
3. ECDF
4. Rug plot
5. Distribución conjunta (bivariada)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 3. Gráficos Categóricos

```python
# Bar plot
sns.barplot(data=df, x="categoria", y="ventas")
sns.barplot(data=df, x="categoria", y="ventas", hue="region")
sns.barplot(data=df, x="categoria", y="ventas", estimator=np.median)
sns.barplot(data=df, x="categoria", y="ventas", ci=95)

# Count plot
sns.countplot(data=df, x="categoria")
sns.countplot(data=df, x="categoria", hue="region")
sns.countplot(data=df, y="categoria")  # horizontal

# Box plot
sns.boxplot(data=df, x="categoria", y="precio")
sns.boxplot(data=df, x="categoria", y="precio", hue="region")
sns.boxenplot(data=df, x="categoria", y="precio")  # boxplot mejorado

# Violin plot (boxplot + KDE)
sns.violinplot(data=df, x="categoria", y="precio")
sns.violinplot(data=df, x="categoria", y="precio", inner="quartile")
sns.violinplot(data=df, x="categoria", y="precio", split=True)  # comparar grupos

# Strip plot
sns.stripplot(data=df, x="categoria", y="precio")
sns.stripplot(data=df, x="categoria", y="precio", jitter=True)

# Swarm plot (strip plot sin superposición)
sns.swarmplot(data=df, x="categoria", y="precio")

# Point plot
sns.pointplot(data=df, x="categoria", y="ventas")
sns.pointplot(data=df, x="categoria", y="ventas", hue="region")

# Combinados
sns.boxplot(data=df, x="categoria", y="precio")
sns.swarmplot(data=df, x="categoria", y="precio", color="black", alpha=0.5)
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*3. Gráficos Categóricos.*

1. Bar plot
2. Count plot
3. Box plot
4. Violin plot (boxplot + KDE)
5. Strip plot
6. Swarm plot (strip plot sin superposición)
7. Point plot
8. Combinados

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 4. Gráficos Relacionales

```python
# Scatter plot
sns.scatterplot(data=df, x="precio", y="ventas")
sns.scatterplot(data=df, x="precio", y="ventas", hue="categoria")
sns.scatterplot(data=df, x="precio", y="ventas", size="stock")
sns.scatterplot(data=df, x="precio", y="ventas", style="region")
sns.scatterplot(data=df, x="precio", y="ventas", alpha=0.5)

# Line plot
sns.lineplot(data=df, x="fecha", y="ventas")
sns.lineplot(data=df, x="fecha", y="ventas", hue="categoria")
sns.lineplot(data=df, x="fecha", y="ventas", style="region")
sns.lineplot(data=df, x="fecha", y="ventas", marker="o")
sns.lineplot(data=df, x="fecha", y="ventas", dashes=False)
sns.lineplot(data=df, x="fecha", y="ventas", errorbar="sd")

# Regresión lineal
sns.regplot(data=df, x="precio", y="ventas")
sns.regplot(data=df, x="precio", y="ventas", order=2)  # polinomial
sns.regplot(data=df, x="precio", y="ventas", logistic=True)
sns.regplot(data=df, x="precio", y="ventas", lowess=True)
sns.regplot(data=df, x="precio", y="ventas", robust=True)
sns.regplot(data=df, x="precio", y="ventas", ci=68)

# lmplot (regplot + facetas)
sns.lmplot(data=df, x="precio", y="ventas", col="categoria")
sns.lmplot(data=df, x="precio", y="ventas", hue="region", col="categoria")
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*4. Gráficos Relacionales.*

1. Scatter plot
2. Line plot
3. Regresión lineal
4. lmplot (regplot + facetas)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 5. Matriz de Correlación — Heatmap

```python
# Calcular correlación
corr = df[["precio", "ventas", "stock", "descuento"]].corr()

# Heatmap básico
sns.heatmap(corr)

# Heatmap personalizado
sns.heatmap(corr, annot=True)
sns.heatmap(corr, annot=True, fmt=".2f")
sns.heatmap(corr, cmap="coolwarm")
sns.heatmap(corr, cmap="RdBu_r", vmin=-1, vmax=1)
sns.heatmap(corr, square=True)
sns.heatmap(corr, linewidths=0.5, linecolor="white")
sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool)))  # triángulo superior
sns.heatmap(corr, cbar_kws={"shrink": 0.8})

# Cluster map
sns.clustermap(corr)
sns.clustermap(corr, cmap="coolwarm", standard_scale=1)
sns.clustermap(df, z_score=1, cmap="viridis")
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*5. Matriz de Correlación — Heatmap.*

1. Calcular correlación
2. Heatmap básico
3. Heatmap personalizado
4. Cluster map

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 6. Facetas — FacetGrid

```python
# FacetGrid básico
g = sns.FacetGrid(df, col="categoria")
g.map(sns.histplot, "precio")

# FacetGrid con filas y columnas
g = sns.FacetGrid(df, col="region", row="categoria")
g.map(sns.scatterplot, "precio", "ventas")

# FacetGrid con hue
g = sns.FacetGrid(df, col="categoria", hue="region")
g.map(sns.scatterplot, "precio", "ventas").add_legend()

# Personalizar facetas
g = sns.FacetGrid(df, col="categoria", col_wrap=3)
g.map(sns.histplot, "precio", kde=True)
g.set_titles("{col_name}")
g.set_axis_labels("Precio ($)", "Frecuencia")
g.tight_layout()

# pairplot (todas las combinaciones)
sns.pairplot(df)
sns.pairplot(df, hue="categoria")
sns.pairplot(df, vars=["precio", "ventas", "stock"])
sns.pairplot(df, kind="kde")
sns.pairplot(df, diag_kind="hist")
sns.PairGrid(df).map_diag(sns.histplot).map_offdiag(sns.scatterplot)
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*6. Facetas — FacetGrid.*

1. FacetGrid básico
2. FacetGrid con filas y columnas
3. FacetGrid con hue
4. Personalizar facetas
5. pairplot (todas las combinaciones)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 7. Personalización

```python
# Títulos y etiquetas
sns.scatterplot(data=df, x="precio", y="ventas")
plt.title("Ventas vs Precio por Categoría")
plt.xlabel("Precio ($)")
plt.ylabel("Unidades Vendidas")

# Modificar después de crear el plot
ax = sns.boxplot(data=df, x="categoria", y="precio")
ax.set_title("Distribución de Precios por Categoría")
ax.set_xlabel("Categoría")
ax.set_ylabel("Precio ($)")
plt.xticks(rotation=45)

# Límites de ejes
plt.xlim(0, 1000)
plt.ylim(0, 500)

# Leyenda personalizada
plt.legend(title="Región", loc="upper right")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[1:], labels=labels[1:], title="Región")

# Tamaño de figura
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x="categoria", y="precio")

# Guardar
plt.savefig("grafico.png", dpi=300, bbox_inches="tight")
plt.savefig("grafico.pdf", bbox_inches="tight")
plt.show()

# Subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
sns.histplot(data=df, x="precio", ax=axes[0, 0])
sns.boxplot(data=df, x="categoria", y="precio", ax=axes[0, 1])
sns.scatterplot(data=df, x="precio", y="ventas", ax=axes[1, 0])
sns.heatmap(corr, annot=True, ax=axes[1, 1])
plt.tight_layout()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*7. Personalización.*

1. Títulos y etiquetas
2. Modificar después de crear el plot
3. Límites de ejes
4. Leyenda personalizada
5. Tamaño de figura
6. Guardar
7. Subplots

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 8. Temas y Personalización Avanzada

```python
# Colores personalizados
sns.set_palette(["#FF6B6B", "#4ECDC4", "#45B7D1"])
custom_colors = {"Tech": "blue", "Ropa": "green", "Hogar": "orange"}

# rcParams
sns.set_theme(rc={
    "figure.figsize": (10, 6),
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})

# Grid personalizado
sns.set_style({
    "axes.facecolor": "#f5f5f5",
    "grid.color": "white",
    "grid.linestyle": "-",
    "axes.spines.right": False,
    "axes.spines.top": False
})
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*8. Temas y Personalización Avanzada.*

1. Colores personalizados
2. rcParams
3. Grid personalizado

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 9. Ejemplos de Dashboard

```python
# Ejemplo: reporte de ventas completo
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Ventas por categoría
sns.barplot(data=df, x="categoria", y="ventas", ax=axes[0, 0])
axes[0, 0].set_title("Ventas por Categoría")

# 2. Distribución de precios
sns.histplot(data=df, x="precio", kde=True, ax=axes[0, 1])
axes[0, 1].set_title("Distribución de Precios")

# 3. Ventas en el tiempo
sns.lineplot(data=df, x="fecha", y="ventas", ax=axes[0, 2])
axes[0, 2].set_title("Ventas en el Tiempo")

# 4. Boxplot por categoría
sns.boxplot(data=df, x="categoria", y="precio", ax=axes[1, 0])
axes[1, 0].set_title("Precios por Categoría")

# 5. Scatter precio vs ventas
sns.scatterplot(data=df, x="precio", y="ventas", hue="categoria", ax=axes[1, 1])
axes[1, 1].set_title("Precio vs Ventas")

# 6. Matriz de correlación
corr = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=axes[1, 2])
axes[1, 2].set_title("Correlaciones")

plt.tight_layout()
plt.savefig("dashboard_ventas.png", dpi=150)
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*9. Ejemplos de Dashboard.*

1. Ejemplo: reporte de ventas completo
2. 1. Ventas por categoría
3. 2. Distribución de precios
4. 3. Ventas en el tiempo
5. 4. Boxplot por categoría
6. 5. Scatter precio vs ventas
7. 6. Matriz de correlación

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 10. Displot — Unificación de Distribuciones

```python
# displot (versión moderna, reemplaza distplot)
sns.displot(data=df, x="precio")                         # histograma
sns.displot(data=df, x="precio", kde=True)               # histograma + kde
sns.displot(data=df, x="precio", kind="kde")             # solo kde
sns.displot(data=df, x="precio", kind="ecdf")            # ecdf
sns.displot(data=df, x="precio", hue="categoria")        # por categoría
sns.displot(data=df, x="precio", col="categoria")        # facetas
sns.displot(data=df, x="precio", rug=True)               # con rug
sns.displot(data=df, x="precio", bins=20, binwidth=10)   # control de bins

# Bivariado
sns.displot(data=df, x="precio", y="ventas")             # hex bin
sns.displot(data=df, x="precio", y="ventas", kind="kde") # kde 2D
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*10. Displot — Unificación de Distribuciones.*

1. displot (versión moderna, reemplaza distplot)
2. Bivariado

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 11. Relplot — Unificación de Relacionales

```python
# relplot (scatter + facets)
sns.relplot(data=df, x="precio", y="ventas")
sns.relplot(data=df, x="precio", y="ventas", hue="categoria")
sns.relplot(data=df, x="precio", y="ventas", size="stock")
sns.relplot(data=df, x="precio", y="ventas", col="categoria")
sns.relplot(data=df, x="precio", y="ventas", col="categoria", row="region")
sns.relplot(data=df, x="fecha", y="ventas", kind="line")  # line plot
sns.relplot(data=df, x="fecha", y="ventas", kind="line",
            hue="categoria", marker="o")
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*11. Relplot — Unificación de Relacionales.*

1. relplot (scatter + facets)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 12. Catplot — Unificación de Categóricos

```python
# catplot (todos los categóricos en una función)
sns.catplot(data=df, x="categoria", y="ventas", kind="bar")
sns.catplot(data=df, x="categoria", y="ventas", kind="box")
sns.catplot(data=df, x="categoria", y="ventas", kind="violin")
sns.catplot(data=df, x="categoria", y="ventas", kind="strip")
sns.catplot(data=df, x="categoria", y="ventas", kind="swarm")
sns.catplot(data=df, x="categoria", y="ventas", kind="point")
sns.catplot(data=df, x="categoria", y="ventas", kind="boxen")
sns.catplot(data=df, x="categoria", y="ventas", hue="region")
sns.catplot(data=df, x="categoria", y="ventas", col="region")
sns.catplot(data=df, x="categoria", y="ventas", kind="bar",
            order=["Tech", "Hogar", "Ropa"])  # orden personalizado
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*12. Catplot — Unificación de Categóricos.*

1. catplot (todos los categóricos en una función)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 13. Formato de Ejes y Anotaciones

```python
# Escalas
sns.scatterplot(data=df, x="precio", y="ventas")
plt.xscale("log")
plt.yscale("log")

# Formato de ticks
import matplotlib.ticker as ticker
ax = sns.barplot(data=df, x="categoria", y="ventas")
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("$%d"))
ax.xaxis.set_major_formatter(ticker.NullFormatter())

# Anotaciones en barras
ax = sns.barplot(data=df, x="categoria", y="ventas")
for i, v in enumerate(df.groupby("categoria")["ventas"].mean()):
    ax.text(i, v + 0.5, f"{v:.0f}", ha="center", va="bottom")

# Líneas de referencia
plt.axhline(y=df["ventas"].mean(), color="red", linestyle="--", label="Promedio")
plt.axvline(x=100, color="gray", linestyle=":", alpha=0.5)
plt.axhline(y=0, color="black", linewidth=0.5)

# Sombreado de regiones
plt.axhspan(ymin=100, ymax=200, alpha=0.2, color="yellow")
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*13. Formato de Ejes y Anotaciones.*

1. Escalas
2. Formato de ticks
3. Anotaciones en barras
4. Líneas de referencia
5. Sombreado de regiones

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 14. Paletas de Colores Detalladas

```python
# Ver paletas disponibles
sns.palettes.SEABORN_PALETTES.keys()

# Paletas cualitativas
sns.color_palette("deep", 10)
sns.color_palette("muted", 10)
sns.color_palette("bright", 10)
sns.color_palette("pastel", 10)
sns.color_palette("dark", 10)
sns.color_palette("Set1", 9)
sns.color_palette("tab10", 10)

# Paletas secuenciales (para datos continuos)
sns.color_palette("Blues", 10)
sns.color_palette("viridis", 10)
sns.color_palette("rocket", 10)
sns.color_palette("mako", 10)

# Paletas divergentes (datos con centro significativo)
sns.color_palette("coolwarm", 10)
sns.color_palette("RdBu", 10)
sns.color_palette("vlag", 10)

# Crear paleta personalizada
sns.set_palette(sns.dark_palette("purple", 4))
sns.set_palette(sns.light_palette("green", 4))
sns.set_palette(sns.diverging_palette(240, 10, n=9))

# light_palette / dark_palette para colormaps
sns.light_palette("blue", as_cmap=True)
sns.dark_palette("red", as_cmap=True)
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*14. Paletas de Colores Detalladas.*

1. Ver paletas disponibles
2. Paletas cualitativas
3. Paletas secuenciales (para datos continuos)
4. Paletas divergentes (datos con centro significativo)
5. Crear paleta personalizada
6. light_palette / dark_palette para colormaps

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 15. Comparación con Matplotlib

```python
# Seaborn trabaja directamente con DataFrames
# Matplotlib requiere más código manual

# Matplotlib puro
fig, ax = plt.subplots()
ax.scatter(df["precio"], df["ventas"])
ax.set_xlabel("Precio")
ax.set_ylabel("Ventas")
ax.set_title("Ventas vs Precio")

# Seaborn (1 línea vs ~5 de matplotlib)
sns.scatterplot(data=df, x="precio", y="ventas")

# Seaborn añade automáticamente:
#   - Leyenda para hue
#   - Estilo consistente
#   - Estadísticas (ci en lineplot, etc.)
#   - Manejo de facetas integrado
#   - Distribuciones (kde en histplot)
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*15. Comparación con Matplotlib.*

1. Seaborn trabaja directamente con DataFrames
2. Matplotlib requiere más código manual
3. Matplotlib puro
4. Seaborn (1 línea vs ~5 de matplotlib)
5. Seaborn añade automáticamente:
6. - Leyenda para hue
7. - Estilo consistente
8. - Estadísticas (ci en lineplot, etc.)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## Referencia Rápida

| Tipo | Función | Uso |
|------|---------|-----|
| Distribución | `histplot`, `kdeplot`, `ecdfplot`, `displot` | Univariado |
| Categórico | `barplot`, `boxplot`, `violinplot`, `catplot` | Por categoría |
| Relacional | `scatterplot`, `lineplot`, `relplot`, `regplot` | Dos variables |
| Matriz | `heatmap`, `clustermap` | Correlación |
| Facetas | `FacetGrid`, `pairplot`, `jointplot` | Múltiples subplots |
| Personalizar | `set_theme`, `set_style`, `set_palette` | Estética global |
