# I10 — Facetas y Gráficos Multipanel en Seaborn

## 1. Introducción Teórica

Los gráficos facetados (multipanel) permiten visualizar subconjuntos de datos en paneles separados, facilitando la comparación entre categorías, sucursales, meses, etc. Seaborn ofrece varias clases y funciones para crear estos gráficos de forma declarativa.

### Clases y funciones principales:

- **`sns.FacetGrid`**: Clase base para crear una grilla de subgráficos según `row` y/o `col`. Se usa con `.map()` para dibujar en cada panel.
- **`sns.lmplot`**: Versión facetada de `regplot` (regresión lineal con scatter).
- **`sns.catplot`**: Versión facetada de todos los categorical plots (bar, box, violin, point, strip, swarm, boxen).
- **`sns.pairplot`**: Matriz de dispersión para múltiples variables (todos contra todos).
- **`sns.PairGrid`**: Clase base para matrices de dispersión con control fino (`map_upper`, `map_lower`, `map_diag`).
- **`sns.JointGrid`**: Gráfico conjunto con distribución marginal (histograma/kde en bordes).

### Parámetros clave de FacetGrid:

| Parámetro | Descripción |
|-----------|-------------|
| `row`, `col` | Variables para dividir filas/columnas |
| `hue` | Variable de color dentro de cada panel |
| `col_wrap` | Número de columnas antes de saltar a siguiente fila |
| `height`, `aspect` | Tamaño y proporción de cada panel |
| `sharex`, `sharey` | Ejes compartidos (por defecto True) |
| `palette` | Paleta de colores |
| `despine` | Quitar ejes sobrantes automáticamente |

---

## 2. Ejemplos Prácticos

### Ejemplo 1: FacetGrid(row, col) — Histogramas facetados

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

ventas = pd.read_csv("../../datos/ventas.csv")

# FacetGrid con fila=categoría, columna=mes
g = sns.FacetGrid(ventas, row="categoria", col="mes", height=2.5, aspect=1.5)
g.map(sns.histplot, "ingreso", bins=20)
g.set_titles(row_template="{row_name}", col_template="Mes {col_name}")
g.set_axis_labels("Ingreso ($)", "Frecuencia")
g.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 1: FacetGrid(row, col) — Histogramas facetados.*

1. FacetGrid con fila=categoría, columna=mes

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: FacetGrid(col, col_wrap) — Scatterplots por sucursal

```python
g = sns.FacetGrid(ventas, col="sucursal", col_wrap=3, height=3, aspect=1.3)
g.map(sns.scatterplot, "precio_unitario", "cantidad", alpha=0.5)
g.set_titles(col_template="{col_name}")
g.set_axis_labels("Precio unitario", "Cantidad")
g.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 2: FacetGrid(col, col_wrap) — Scatterplots por sucursal.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: FacetGrid(hue) + KDE + add_legend

```python
g = sns.FacetGrid(ventas, hue="categoria", height=4, aspect=1.5)
g.map(sns.kdeplot, "margen", fill=True, alpha=0.3)
g.add_legend(title="Categoría")
g.set_axis_labels("Margen ($)")
g.set_titles("Distribución de margen por categoría")
g.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 3: FacetGrid(hue) + KDE + add_legend.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: FacetGrid con set_titles y set_axis_labels

```python
g = sns.FacetGrid(ventas, col="sucursal", col_wrap=3, height=3, aspect=1.2)
g.map(sns.scatterplot, "precio_unitario", "cantidad", alpha=0.4)
g.set_titles(col_template="{col_name} — Precio vs Cantidad")
g.set_axis_labels("Precio ($)", "Cantidad (uds)")
g.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 4: FacetGrid con set_titles y set_axis_labels.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: lmplot — Regresión lineal facetada

```python
sns.lmplot(x="precio_unitario", y="ingreso", col="sucursal", col_wrap=3,
           data=ventas, scatter_kws={"alpha": 0.5}, height=3, aspect=1.2,
           line_kws={"color": "red"})
plt.suptitle("Regresión: Precio vs Ingreso por Sucursal", y=1.02)
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 5: lmplot — Regresión lineal facetada.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: catplot(kind="bar") — Barras facetadas

```python
sns.catplot(x="categoria", y="ingreso", col="sucursal", col_wrap=3,
            kind="bar", data=ventas, estimator=sum, ci=None, sharey=False,
            height=3.5, aspect=1.2)
plt.suptitle("Ingreso por Categoría y Sucursal", y=1.02)
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 6: catplot(kind="bar") — Barras facetadas.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: catplot(kind="box") — Boxplots con fila y columna

```python
sns.catplot(x="categoria", y="margen", row="mes", col="sucursal",
            kind="box", data=ventas, height=3, aspect=1.5)
plt.suptitle("Distribución de Margen: Categoría × Sucursal × Mes", y=1.02)
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 7: catplot(kind="box") — Boxplots con fila y columna.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: catplot(kind="violin") — Violines facetados

```python
# Violinplot: combina boxplot + kde en cada categoría
sns.catplot(x="categoria", y="margen", col="sucursal", col_wrap=3,
            kind="violin", data=ventas, height=3, aspect=1.2,
            inner="quartile")
plt.suptitle("Distribución de Margen por Categoría (violin)", y=1.02)
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 8: catplot(kind="violin") — Violines facetados.*

1. Violinplot: combina boxplot + kde en cada categoría

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: pairplot con corner=True — Matriz de dispersión triangular

```python
# corner=True: solo mitad inferior (evita duplicados)
sns.pairplot(ventas, vars=["ingreso", "margen", "cantidad", "precio_unitario"],
             hue="categoria", corner=True, palette="Set2", height=2.5)
plt.suptitle("Pairplot: Métricas de Ventas por Categoría (triangular)",
             y=1.02, fontsize=14)
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 9: pairplot con corner=True — Matriz de dispersión triangular.*

1. corner=True: solo mitad inferior (evita duplicados)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: pairplot con kind="kde" — Densidad 2D en matriz

```python
sns.pairplot(ventas, vars=["ingreso", "margen", "precio_unitario"],
             kind="kde", diag_kind="hist", palette="husl", height=3)
plt.suptitle("Pairplot tipo KDE con diagonales en histograma", y=1.02)
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 10: pairplot con kind="kde" — Densidad 2D en matriz.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: PairGrid — Control fino (map_upper, map_lower, map_diag)

```python
# PairGrid permite diferentes tipos de gráfico en cada sección
g = sns.PairGrid(ventas, vars=["ingreso", "margen", "cantidad"],
                 hue="categoria", palette="Set1", height=2.5)
g.map_upper(sns.scatterplot, alpha=0.6)
g.map_lower(sns.kdeplot, fill=True, alpha=0.3)
g.map_diag(sns.histplot, kde=True)
g.add_legend(title="Categoría")
plt.suptitle("PairGrid: scatter (arriba) + kde (abajo) + hist (diag)", y=1.02)
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 11: PairGrid — Control fino (map_upper, map_lower, map_diag).*

1. PairGrid permite diferentes tipos de gráfico en cada sección

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: JointGrid — Scatter + histogramas marginales

```python
g = sns.JointGrid(data=ventas, x="precio_unitario", y="margen", height=6)
g.plot_joint(sns.scatterplot, alpha=0.4, s=30)
g.plot_marginals(sns.histplot, kde=True, bins=30)
g.set_axis_labels("Precio Unitario ($)", "Margen ($)")
plt.suptitle("JointGrid: Precio vs Margen + marginales", y=1.02)
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 12: JointGrid — Scatter + histogramas marginales.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: JointGrid con hue — Grupos coloreados

```python
# JointGrid con hue colorea puntos pero marginales son totales
g = sns.JointGrid(data=ventas, x="precio_unitario", y="cantidad", height=6)
g.plot_joint(sns.scatterplot, hue=ventas["categoria"], alpha=0.5, palette="Set2")
g.plot_marginals(sns.histplot, kde=True, bins=25)
g.set_axis_labels("Precio Unitario ($)", "Cantidad")
g.add_legend(title="Categoría")
plt.suptitle("JointGrid con hue (marginal sin hue)", y=1.02)
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 13: JointGrid con hue — Grupos coloreados.*

1. JointGrid con hue colorea puntos pero marginales son totales

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: FacetGrid con map_dataframe

```python
# map_dataframe pasa el DataFrame completo a la función
# Útil para funciones que necesitan acceder a múltiples columnas
def regresion_lineal(data, color=None, **kwargs):
    sns.regplot(data=data, x="precio_unitario", y="ingreso",
                scatter_kws={"alpha": 0.4}, color=color, **kwargs)

g = sns.FacetGrid(ventas, col="categoria", col_wrap=3, height=3, aspect=1.2)
g.map_dataframe(regresion_lineal)
g.set_titles(col_template="{col_name}")
g.set_axis_labels("Precio unitario", "Ingreso")
plt.suptitle("Regresión lineal por categoría (map_dataframe)", y=1.02)
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 14: FacetGrid con map_dataframe.*

1. map_dataframe pasa el DataFrame completo a la función
2. Útil para funciones que necesitan acceder a múltiples columnas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: FacetGrid con set(xlim, ylim) — Ejes consistentes

```python
g = sns.FacetGrid(ventas, col="sucursal", col_wrap=3, height=3, aspect=1.2)
g.map(sns.scatterplot, "precio_unitario", "margen", alpha=0.4)
g.set(xlim=(0, 10000), ylim=(0, 5000))  # Mismos límites en todos
g.set_titles(col_template="{col_name}")
g.set_axis_labels("Precio ($)", "Margen ($)")
plt.suptitle("Precio vs Margen con ejes consistentes", y=1.02)
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 15: FacetGrid con set(xlim, ylim) — Ejes consistentes.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 16: pairplot con plot_kws y diag_kws

```python
sns.pairplot(ventas, vars=["ingreso", "margen", "cantidad", "precio_unitario"],
             hue="categoria", palette="Dark2", height=2.5,
             plot_kws={"s": 50, "alpha": 0.6, "edgecolor": "none"},
             diag_kws={"kde": True, "alpha": 0.5})
plt.suptitle("pairplot con plot_kws (tamaño/transparencia) y diag_kws", y=1.02)
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 16: pairplot con plot_kws y diag_kws.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 17: lmplot con col, row, hue y col_order

```python
# col_order personalizado
orden_sucursales = ["Sucursal Mérida", "Sucursal Querétaro",
                    "Sucursal CDMX", "Sucursal Guadalajara", "Sucursal Monterrey"]

sns.lmplot(x="precio_unitario", y="ingreso",
           col="sucursal", row="mes", hue="categoria",
           data=ventas[ventas["mes"] <= 2],  # solo enero y febrero
           col_order=orden_sucursales,
           height=3, aspect=1.3,
           scatter_kws={"alpha": 0.4},
           palette="Set1")
plt.suptitle("lmplot: Precio vs Ingreso (col=sucursal, row=mes, hue=categoría)",
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

*Ejemplo 17: lmplot con col, row, hue y col_order.*

1. col_order personalizado

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 18: FacetGrid con regplot + add_legend

```python
# Seaborn no tiene regplot facetado nativo, lo simulamos con FacetGrid
def dibujar_regresion(data, color=None, **kwargs):
    sns.regplot(data=data, x="precio_unitario", y="margen",
                scatter_kws={"alpha": 0.3, "s": 20},
                line_kws={"linewidth": 2},
                color=color, **kwargs)

g = sns.FacetGrid(ventas, col="categoria", hue="categoria",
                  col_wrap=3, height=3, aspect=1.3, palette="Set2")
g.map_dataframe(dibujar_regresion)
g.add_legend(title="Categoría")
g.set_titles(col_template="{col_name}")
g.set_axis_labels("Precio unitario", "Margen")
plt.suptitle("Regresión Precio vs Margen por Categoría", y=1.02)
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 18: FacetGrid con regplot + add_legend.*

1. Seaborn no tiene regplot facetado nativo, lo simulamos con FacetGrid

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 3. Resumen

| Función/Clase | Tipo de Faceta | Cuándo usarla |
|--------------|----------------|---------------|
| `FacetGrid` + `.map()` | Grilla genérica row/col | Cuando necesitas máxima flexibilidad |
| `FacetGrid` + `.map_dataframe()` | Grilla con función que necesita datos completos | Regresiones, cálculos por grupo |
| `catplot(kind=...)` | Categórico facetado | Box, violin, bar, point por categoría |
| `lmplot(col=...)` | Regresión lineal facetada | Relación precio-margen por sucursal |
| `pairplot(vars=..., hue=...)` | Matriz de dispersión | Exploración multivariable rápida |
| `PairGrid(map_upper/map_lower/map_diag)` | Matriz con controles finos | Diferentes visualizaciones por sección |
| `JointGrid(plot_joint/plot_marginal)` | Joint + marginales | Relación bivariable + distribución |

---

## 4. Ejercicios Propuestos

**Ejercicio 1:** Usa `FacetGrid` con `row="categoria"` y `col="mes"` para crear histogramas de `margen`. ¿Cuántos paneles se generan? Usa `col_wrap` para controlar el layout.

**Ejercicio 2:** Crea un `catplot(kind="boxen", col="sucursal", col_wrap=2)` mostrando la distribución de `precio_unitario` por `categoria`. Compara con `kind="box"`.

**Ejercicio 3:** Usa `sns.lmplot` con `col="categoria"`, `hue="sucursal"`, y `scatter_kws={"alpha": 0.4}` para explorar la relación entre `precio_unitario` y `margen`.

**Ejercicio 4:** Carga el DataFrame de inventario. Usa `sns.pairplot` con todas las columnas numéricas (`stock_actual`, `stock_minimo`, `stock_maximo`, `precio`, `costo`) con `hue="categoria"` y `corner=True`.

**Ejercicio 5:** Con `PairGrid`, crea una matriz donde el triángulo superior tenga scatterplots, el inferior tenga kdeplots (contornos), y la diagonal tenga histogramas con kde. Usa `hue="categoria"`.

**Ejercicio 6:** Usa `JointGrid` para graficar `precio` vs `stock_actual` del inventario, con marginales en histograma. Colorea los puntos por `categoria`.

**Ejercicio 7:** Crea un `FacetGrid` de 2 filas y 2 columnas. Arriba a la izquierda: scatter precio vs margen. Arriba derecha: boxplot de margen por categoría. Abajo izquierda: histograma de ingreso. Abajo derecha: countplot de sucursales. (Pista: usa `ax=axes[i,j]` de matplotlib directamente.)

**Ejercicio 8:** Usa `sns.catplot(kind="point", col="mes", col_wrap=4)` para mostrar la tendencia mensual del margen promedio por categoría. ¿Qué insights puedes extraer sobre la estacionalidad?

---

*Fin del documento I10 — Facetas y Multipanel en Seaborn*
