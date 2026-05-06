# I09 — Personalización de Gráficos en Seaborn

## 1. Introducción Teórica

Seaborn permite personalizar cada aspecto visual de los gráficos: paletas de colores, estilos de fondo, contexto (tamaño de fuente/ejes), y decoración de ejes. Una personalización adecuada mejora la comunicación de insights en ventas, compras e inventarios.

### Componentes de personalización:

- **Paletas de colores** (`color_palette`, `set_palette`, `light_palette`, `dark_palette`, `diverging_palette`, `blend_palette`, `cubehelix_palette`, `xkcd_palette`): Controlan los colores de las series
- **Estilos** (`set_style`, `axes_style`): "white", "dark", "whitegrid", "darkgrid", "ticks"
- **Contexto** (`set_context`): "paper" (artículo), "notebook" (por defecto), "talk" (presentación), "poster" (póster)
- **Tema general** (`set_theme`): Configura todo de una vez (style, palette, font_scale)
- **Despine** (`despine`): Elimina/desplaza ejes superiores y derechos
- **Fuentes** (`set(font_scale)`, `set(font=...)`): Escala y tipo de letra

### Tipos de paletas:

| Tipo | Función | Uso |
|------|---------|-----|
| **Qualitative** | `"deep"`, `"pastel"`, `"Set2"`, `"Paired"` | Categorías sin orden |
| **Sequential** | `"Blues"`, `"light_palette"`, `"dark_palette"` | Datos ordinales (0→alto) |
| **Diverging** | `"RdBu"`, `"diverging_palette"` | Datos con punto medio (correlación) |
| **Circular** | `"husl"`, `"hls"` | Colores uniformemente distribuidos |
| **Personalizada** | `blend_palette`, `xkcd_palette` | Marca corporativa o preferencia |

---

## 2. Ejemplos Prácticos

### Ejemplo 1: set_theme — Configuración completa inicial

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

ventas = pd.read_csv("../../datos/ventas.csv")

# Tema global: whitegrid, paleta muted, fuente 1.2x
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

sns.barplot(x="categoria", y="ingreso", data=ventas, estimator=sum, ci=None)
plt.title("Ingreso total por categoría")
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

*Ejemplo 1: set_theme — Configuración completa inicial.*

1. Tema global: whitegrid, paleta muted, fuente 1.2x

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: light_palette — Secuencial claro con reverse

```python
# light_palette: de blanco/color claro a color intenso
paleta = sns.light_palette("blue", n_colors=6, reverse=True)
sns.set_palette(paleta)

sns.barplot(x="categoria", y="margen", data=ventas, estimator=sum, ci=None)
plt.title("Margen total por categoría (light_palette blue reverse)")
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

*Ejemplo 2: light_palette — Secuencial claro con reverse.*

1. light_palette: de blanco/color claro a color intenso

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: dark_palette — Secuencial oscuro como cmap para heatmap

```python
# dark_palette como colormap para heatmap
paleta_cmap = sns.dark_palette("green", as_cmap=True)

tabla = ventas.pivot_table(values="ingreso", index="categoria",
                           columns="sucursal", aggfunc=sum, fill_value=0)
sns.heatmap(tabla, cmap=paleta_cmap, annot=True, fmt=".0f")
plt.title("Ingresos por categoría y sucursal (dark_palette green)")
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 3: dark_palette — Secuencial oscuro como cmap para heatmap.*

1. dark_palette como colormap para heatmap

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: diverging_palette — Heatmap de correlación con centro iluminado

```python
# diverging: dos colores en extremos, centro claro
numéricas = ventas.select_dtypes(include=[np.number])
corr = numéricas.corr()

paleta_div = sns.diverging_palette(250, 15, s=75, l=40, center="light", as_cmap=True)
sns.heatmap(corr, cmap=paleta_div, center=0, annot=True, fmt=".2f",
            square=True, linewidths=0.5)
plt.title("Matriz de correlación (diverging_palette 250-15)")
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 4: diverging_palette — Heatmap de correlación con centro iluminado.*

1. diverging: dos colores en extremos, centro claro

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: blend_palette — Mezcla personalizada de 3 colores

```python
paleta_blend = sns.blend_palette(["red", "yellow", "green"], n_colors=6)
sns.set_palette(paleta_blend)

sns.scatterplot(x="precio_unitario", y="margen", data=ventas, s=50, alpha=0.6)
plt.title("Precio vs Margen (blend_palette red-yellow-green)")
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 5: blend_palette — Mezcla personalizada de 3 colores.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: set_context("talk") — Para presentaciones

```python
sns.set_context("talk")  # Fuentes más grandes, ejes más gruesos
sns.set_style("whitegrid")

sns.barplot(x="categoria", y="cantidad", data=ventas, estimator=sum, ci=None)
plt.title("Unidades vendidas por categoría (context=talk)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Restaurar contexto
sns.set_context("notebook")
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 6: set_context("talk") — Para presentaciones.*

1. Restaurar contexto

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: set_context("poster") — Para pósters

```python
sns.set_context("poster", font_scale=0.9)  # poster es muy grande, reducimos scale
sns.set_style("white")

sns.boxplot(x="categoria", y="margen", data=ventas)
plt.title("Distribución de margen por categoría (poster)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

sns.set_context("notebook")
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 7: set_context("poster") — Para pósters.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: set_style("darkgrid") + despine

```python
sns.set_style("darkgrid")
# despine quita ejes superiores y derechos
sns.despine(left=True, bottom=True)  # Quita también ejes izquierdo e inferior

sns.scatterplot(x="precio_unitario", y="cantidad", data=ventas, alpha=0.5)
plt.title("Precio vs Cantidad (darkgrid + despine left+bottom)")
plt.tight_layout()
plt.show()

sns.set_style("whitegrid")  # restaurar
sns.despine()  # restaurar despine default
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 8: set_style("darkgrid") + despine.*

1. despine quita ejes superiores y derechos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: axes_style — Contexto temporal con with

```python
# Usar un estilo temporalmente dentro de un bloque with
with sns.axes_style("white"):
    fig, ax = plt.subplots()
    sns.scatterplot(x="precio_unitario", y="margen", data=ventas, alpha=0.5, ax=ax)
    ax.set_title("Estilo 'white' temporal (axes_style)")
plt.show()

# Verificar que volvió al estilo anterior
sns.scatterplot(x="precio_unitario", y="margen", data=ventas, alpha=0.5)
plt.title("Estilo restaurado automáticamente")
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 9: axes_style — Contexto temporal con with.*

1. Usar un estilo temporalmente dentro de un bloque with
2. Verificar que volvió al estilo anterior

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: color_palette("husl", 8) — Colores uniformemente distribuidos

```python
paleta_husl = sns.color_palette("husl", n_colors=8)
sns.set_palette(paleta_husl)

sns.barplot(x="sucursal", y="ingreso", data=ventas, estimator=sum, ci=None)
plt.title("Ingreso por sucursal (husl 8 colores)")
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

*Ejemplo 10: color_palette("husl", 8) — Colores uniformemente distribuidos.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: color_palette("Set2") — Paleta cualitativa suave

```python
sns.set_palette("Set2")

sns.catplot(x="categoria", y="margen", col="sucursal", col_wrap=3,
            data=ventas, kind="bar", estimator=sum, ci=None, sharey=False)
plt.suptitle("Margen por categoría y sucursal (Set2)", y=1.02)
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 11: color_palette("Set2") — Paleta cualitativa suave.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: despine(offset=10, trim=True) — Ejes desplazados

```python
sns.set_style("ticks")
fig, ax = plt.subplots()
sns.scatterplot(x="precio_unitario", y="cantidad", data=ventas, alpha=0.5, ax=ax)
sns.despine(offset=10, trim=True)  # Desplaza ejes 10pts del borde, recorta
ax.set_title("despine con offset=10, trim=True")
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 12: despine(offset=10, trim=True) — Ejes desplazados.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: cubehelix_palette — Escala secuencial versátil

```python
paleta_cube = sns.cubehelix_palette(start=2, rot=0, dark=0, light=1,
                                    reverse=True, as_cmap=True)

tabla = ventas.pivot_table(values="margen", index="categoria",
                           columns="mes", aggfunc=sum, fill_value=0)
sns.heatmap(tabla, cmap=paleta_cube, annot=True, fmt=".0f")
plt.title("Margen por categoría y mes (cubehelix)")
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 13: cubehelix_palette — Escala secuencial versátil.*

1. `tabla = ventas.pivot_table(values="margen", index="categoria",` — Reorganiza los datos de formato largo a ancho.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: xkcd_palette — Colores con nombres descriptivos

```python
paleta_xkcd = sns.xkcd_palette(["windows blue", "amber", "faded green",
                                 "dusty purple", "pale red"])
sns.set_palette(paleta_xkcd)

sns.barplot(x="sucursal", y="ingreso", data=ventas, estimator=sum, ci=None)
plt.title("Ingreso por sucursal (xkcd_palette)")
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

*Ejemplo 14: xkcd_palette — Colores con nombres descriptivos.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: set_palette("Paired") + countplot

```python
sns.set_palette("Paired")

sns.countplot(x="categoria", data=ventas, order=ventas["categoria"].value_counts().index)
plt.title("Frecuencia de ventas por categoría (Paired)")
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

*Ejemplo 15: set_palette("Paired") + countplot.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 16: Personalización de fuente (font_scale, font)

```python
sns.set(font_scale=1.5, font="serif")
sns.set_style("whitegrid")

sns.scatterplot(x="precio_unitario", y="margen", data=ventas, alpha=0.5)
plt.title("Precio vs Margen (font=serif, scale=1.5)")
plt.tight_layout()
plt.show()

# Restaurar
sns.set(font_scale=1.0, font="sans-serif")
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 16: Personalización de fuente (font_scale, font).*

1. Restaurar

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 17: Comparar paletas sequential vs diverging vs qualitative

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
categorias = ventas.groupby("categoria")["ingreso"].sum().sort_values()

# Sequential Blues
sns.heatmap(categorias.to_frame().T, cmap="Blues", annot=True, fmt=".0f",
            ax=axes[0], cbar=False)
axes[0].set_title("Sequential (Blues)")

# Diverging RdBu
sns.heatmap(categorias.to_frame().T, cmap="RdBu_r", annot=True, fmt=".0f",
            ax=axes[1], cbar=False)
axes[1].set_title("Diverging (RdBu_r)")

# Qualitative Set2
sns.barplot(x=categorias.index, y=categorias.values, palette="Set2", ax=axes[2])
axes[2].set_title("Qualitative (Set2)")
axes[2].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 17: Comparar paletas sequential vs diverging vs qualitative.*

1. Sequential Blues
2. Diverging RdBu
3. Qualitative Set2

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 18: Personalización completa — tema + paleta + contexto + despine

```python
# Configuración global
sns.set_theme(style="whitegrid", palette="Set2", font="DejaVu Sans", font_scale=1.3)
sns.set_context("talk")

# Crear gráfico
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x="categoria", y="margen", data=ventas, estimator=sum, ci=None, ax=ax)

# Personalización adicional
sns.despine(left=True, offset=5, trim=True)
ax.set_title("Margen Total por Categoría — Reporte Ejecutivo", fontsize=16, pad=15)
ax.set_xlabel("Categoría de Producto", fontsize=13)
ax.set_ylabel("Margen Total ($)", fontsize=13)
ax.tick_params(axis="x", rotation=45)

# Añadir valores sobre las barras
for i, v in enumerate(ventas.groupby("categoria")["margen"].sum().values):
    ax.text(i, v + 500, f"${v:,.0f}", ha="center", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 18: Personalización completa — tema + paleta + contexto + despine.*

1. Configuración global
2. Crear gráfico
3. Personalización adicional
4. Añadir valores sobre las barras

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 3. Resumen

| Componente | Función | Uso típico |
|-----------|---------|------------|
| `set_theme(style, palette, font_scale)` | Configuración global | Arranque de análisis |
| `color_palette()` / `set_palette()` | Paleta de colores cualitativa | Categorías (Set2, Paired, husl) |
| `light_palette()` / `dark_palette()` | Paleta secuencial | Heatmaps, gradientes |
| `diverging_palette()` | Paleta divergente | Correlaciones, desviaciones |
| `blend_palette()` | Mezcla personalizada | Colores corporativos |
| `cubehelix_palette()` | Secuencial perceptualmente uniforme | Mapas de calor |
| `xkcd_palette()` | Nombres intuitivos | Prototipos rápidos |
| `set_context("talk"/"poster")` | Tamaño para audiencia | Presentaciones, pósters |
| `set_style("whitegrid"/"darkgrid")` | Fondo y rejilla | Legibilidad |
| `despine(offset, trim)` | Decoración de ejes | Gráficos minimalistas |
| `set(font=..., font_scale=...)` | Tipografía | Coherencia visual |

---

## 4. Ejercicios Propuestos

**Ejercicio 1:** Configura un tema global con `set_theme(style="darkgrid", palette="pastel", font_scale=1.1)` y crea un scatterplot de `precio_unitario` vs `cantidad` coloreado por categoría.

**Ejercicio 2:** Usa `sns.light_palette("purple", reverse=True, as_cmap=True)` como cmap en un heatmap de ventas por categoría y sucursal (pivot_table). ¿Qué cambia al usar `reverse=False`?

**Ejercicio 3:** Crea un `diverging_palette` con colores 220 y 20, s=80, l=50, center="light". Úsalo en un heatmap de la matriz de correlación del DataFrame numérico de ventas.

**Ejercicio 4:** Usa `sns.set_context("poster")` para crear un boxplot de `margen` por `categoria`. Luego cambia a `"paper"` y observa las diferencias en tamaño de fuentes y ejes.

**Ejercicio 5:** Crea una paleta blend con `["navy", "white", "crimson"]` con 5 colores. Úsala en un barplot de ingreso por sucursal.

**Ejercicio 6:** Usa `axes_style` con `with` para crear un gráfico temporal con estilo `"dark"` sin rejilla. Dentro del bloque crea un histograma de `ingreso`. Fuera del bloque, crea otro gráfico para verificar que el estilo se restauró.

**Ejercicio 7:** Carga el DataFrame de inventario (`pd.read_csv("../../datos/inventario.csv")`). Usa `cubehelix_palette(start=0.5, rot=-0.5, as_cmap=True)` para un heatmap de `stock_actual` por `categoria` y `producto`.

**Ejercicio 8:** Crea una personalización completa (tema, paleta, contexto, despine, títulos, valores sobre barras) para un countplot de las categorías en inventario. El gráfico debe verse profesional para un reporte ejecutivo.

---

*Fin del documento I09 — Personalización de Seaborn*
