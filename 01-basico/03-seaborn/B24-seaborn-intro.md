# Módulo B24 — Seaborn: Introducción y gráficos de dispersión y líneas

## Teoría

**Seaborn** es una biblioteca de visualización estadística construida sobre Matplotlib. Proporciona una API de alto nivel para crear gráficos atractivos con poco código. Está diseñada para trabajar de forma natural con DataFrames de pandas.

Ventajas principales:
- **Interfaz concisa**: un gráfico completo en una sola llamada (`sns.scatterplot(data=df, x='col1', y='col2')`)
- **Estética superior**: temas y paletas por defecto más atractivos que Matplotlib base
- **Agregación automática**: `lineplot` y `barplot` calculan promedios e intervalos de confianza solos
- **Paletas integradas**: perceptualmente uniformes (viridis, magma), para daltonismo (colorblind), cualitativas (Set2, Paired)
- **Facetado**: `displot`, `relplot`, `catplot` permiten dividir en subgráficos con una sola línea

Conceptos clave:

| Concepto | Descripción |
|----------|-------------|
| `sns.set_theme()` | Configuración global: estilo, paleta, contexto |
| `sns.set_style()` | Fondo: `white`, `dark`, `whitegrid`, `darkgrid`, `ticks` |
| `sns.set_context()` | Tamaño etiquetas/líneas: `paper`, `notebook`, `talk`, `poster` |
| `sns.color_palette()` | Genera o lista paletas de colores |
| `sns.despine()` | Quita bordes superior/derecho del eje |
| `hue` | Colorea puntos/líneas según una variable categórica |
| `size` | Varía tamaño de marcadores según variable numérica |
| `style` | Varía estilo (círculo, triángulo, línea punteada) según categoría |

Usaremos datos de `ventas.csv` y `inventario.csv` con 1330 registros de transacciones reales de una cadena de tiendas.

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

# Tema base
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
3. Tema base

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 18 Ejemplos prácticos

---

### 1. scatterplot básico: precio vs cantidad vendida

```python
sns.scatterplot(data=ventas, x='precio_unitario', y='cantidad')
plt.title('Precio vs Cantidad Vendida')
plt.xlabel('Precio Unitario ($)')
plt.ylabel('Cantidad')
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*1. scatterplot básico: precio vs cantidad vendida.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicación**: `sns.scatterplot` recibe un DataFrame y los nombres de columnas. Cada punto es una transacción. Se observa que a precios bajos (< $2000) las cantidades son más altas y dispersas; a precios altos (> $10000) las cantidades rara vez superan 10 unidades.

**Interpretación**: Productos baratos se venden en volumen; productos caros tienen demanda limitada. Esto es esperable en un negocio de retail.

---

### 2. scatterplot con hue por categoría

```python
sns.scatterplot(data=ventas, x='precio_unitario', y='cantidad', hue='categoria')
plt.title('Precio vs Cantidad por Categoría')
plt.xlabel('Precio Unitario ($)')
plt.ylabel('Cantidad')
plt.legend(bbox_to_anchor=(1.05, 1))
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*2. scatterplot con hue por categoría.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicación**: `hue='categoria'` asigna un color distinto a cada categoría. Seaborn elige automáticamente la paleta. La leyenda se mueve fuera con `bbox_to_anchor` para no tapar datos.

**Interpretación**: Electrónica y Muebles concentran los precios más altos; Papelería y Software están en el rango bajo. Cada categoría ocupa una zona distinta del gráfico.

---

### 3. scatterplot con size = margen

```python
sns.scatterplot(data=ventas, x='precio_unitario', y='cantidad',
                size='margen', sizes=(20, 400), alpha=0.6)
plt.title('Precio vs Cantidad — Tamaño = Margen ($)')
plt.xlabel('Precio Unitario ($)')
plt.ylabel('Cantidad')
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*3. scatterplot con size = margen.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicación**: `size='margen'` hace que el diámetro del punto sea proporcional al margen de esa venta. `sizes` controla el rango mínimo/máximo del tamaño. `alpha=0.6` da transparencia para ver superposición.

**Interpretación**: Los puntos más grandes (alto margen) se concentran en productos de precio medio-alto con cantidades moderadas. Los productos baratos tienen márgenes pequeños aunque vendan muchas unidades.

---

### 4. lineplot: tendencia de ingresos diarios

```python
diario = ventas.groupby('fecha')['ingreso'].sum().reset_index()

sns.lineplot(data=diario, x='fecha', y='ingreso')
plt.title('Evolución del Ingreso Diario')
plt.xlabel('Fecha')
plt.ylabel('Ingreso ($)')
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

*4. lineplot: tendencia de ingresos diarios.*

1. `diario = ventas.groupby('fecha')['ingreso'].sum().reset_index()` — Agrupa los datos por la columna eRealiza la operación indicada con los parámetros definidos..

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicación**: Agrupamos por fecha y sumamos ingresos. `lineplot` conecta los puntos diarios mostrando la tendencia temporal. `xticks(rotation=45)` evita que las fechas se solapen.

**Interpretación**: Se observan picos y valles semanales. Los ingresos fluctúan típicamente entre $50K y $400K, con algunos días de alta facturación.

---

### 5. lineplot por sucursal con hue

```python
diario_suc = ventas.groupby(['fecha', 'sucursal'])['ingreso'].sum().reset_index()

sns.lineplot(data=diario_suc, x='fecha', y='ingreso', hue='sucursal')
plt.title('Ingreso Diario por Sucursal')
plt.xlabel('Fecha')
plt.ylabel('Ingreso ($)')
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

*5. lineplot por sucursal con hue.*

1. `diario_suc = ventas.groupby(['fecha', 'sucursal'])['ingreso'].sum().reset_index()` — Agrupa los datos por la columna eRealiza la operación indicada con los parámetros definidos..

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicación**: `hue='sucursal'` dibuja una línea por cada sucursal con distinto color. Seaborn automáticamente calcula el hue basado en valores únicos.

**Interpretación**: Sucursal Mérida y Matriz CDMX suelen tener los ingresos más altos. Sucursal Tijuana y Cancún muestran menor volumen. Las tendencias son paralelas en la mayoría de sucursales.

---

### 6. lineplot con intervalo de confianza

```python
sns.lineplot(data=ventas, x='dia_semana', y='ingreso', ci=95)
plt.title('Ingreso Promedio por Día de Semana (IC 95%)')
plt.xlabel('Día (0=Domingo, 6=Sábado)')
plt.ylabel('Ingreso Promedio ($)')
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*6. lineplot con intervalo de confianza.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicación**: Sin agrupar previamente, `lineplot` calcula el promedio de `ingreso` para cada `dia_semana` y muestra una banda de confianza del 95% (sombreado). `ci=95` es el valor por defecto.

**Interpretación**: Los días 0-1 (domingo-lunes) tienen ingreso promedio más bajo. Los días 3-4 (miércoles-jueves) muestran picos. La banda ancha indica alta variabilidad entre semanas.

---

### 7. Personalizar figura con fig, ax = plt.subplots()

```python
fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(data=ventas, x='precio_unitario', y='cantidad',
                hue='categoria', ax=ax)
ax.set_title('Personalizado con fig, ax', fontsize=14)
ax.set_xlabel('Precio Unitario ($)')
ax.set_ylabel('Cantidad')
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*7. Personalizar figura con fig, ax = plt.subplots().*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicación**: `plt.subplots(figsize=(10,5))` crea una figura de 10×5 pulgadas. El objeto `ax` permite personalizar título, etiquetas y más usando la API de Matplotlib. `ax=ax` pasa el eje a Seaborn.

**Interpretación**: Control explícito del tamaño y estilo. Útil cuando se necesita ajustar múltiples elementos en una misma figura.

---

### 8. Cambiar paleta a "viridis"

```python
sns.set_palette("viridis")

sns.scatterplot(data=ventas, x='precio_unitario', y='cantidad',
                hue='categoria')
plt.title('Paleta Viridis')
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

*8. Cambiar paleta a "viridis".*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicación**: `sns.set_palette("viridis")` cambia la paleta global. Viridis es una paleta secuencial perceptualmente uniforme, ideal para datos que van de menor a mayor (aunque aquí se usa como cualitativa por ser pocas categorías).

**Interpretación**: Viridis asigna colores del púrpura oscuro al amarillo claro. Es útil para presentaciones porque se distingue bien en blanco y negro.

---

### 9. sns.despine() para quitar bordes

```python
sns.scatterplot(data=ventas, x='precio_unitario', y='cantidad', hue='categoria')
sns.despine(top=True, right=True, left=False, bottom=False)
plt.title('Sin borde superior ni derecho')
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*9. sns.despine() para quitar bordes.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicación**: `sns.despine()` elimina los bordes ("spines") del gráfico. Por defecto quita superior y derecho. Se puede controlar cada borde con `top`, `right`, `left`, `bottom`.

**Interpretación**: Un gráfico sin bordes superfluos se ve más limpio y moderno, enfocando la atención en los datos.

---

### 10. sns.set_context("talk") para presentaciones

```python
sns.set_context("talk")

sns.scatterplot(data=ventas, x='precio_unitario', y='cantidad', hue='categoria')
plt.title('Contexto "talk" — texto más grande')
plt.legend(bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()

sns.set_context("notebook")  # restaurar
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*10. sns.set_context("talk") para presentaciones.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicación**: `sns.set_context("talk")` escala etiquetas, leyendas y líneas para ser legibles en presentaciones. Opciones: `paper` (artículos), `notebook` (por defecto), `talk` (charlas), `poster` (pósters). Siempre restaurar al final.

**Interpretación**: Ideal para exportar figuras a diapositivas sin tener que ajustar manualmente cada tamaño de fuente.

---

### 11. sns.set_style("whitegrid")

```python
sns.set_style("whitegrid")

sns.scatterplot(data=ventas, x='precio_unitario', y='cantidad', hue='categoria')
plt.title('Estilo whitegrid — cuadrícula blanca')
plt.legend(bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()

sns.set_style("white")  # restaurar
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*11. sns.set_style("whitegrid").*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicación**: `"whitegrid"` añade una cuadrícula blanca horizontal (y vertical) sobre fondo blanco. Facilita leer valores contra los ejes. Otros: `"darkgrid"`, `"dark"`, `"white"`, `"ticks"`.

**Interpretación**: La cuadrícula ayuda a estimar visualmente la densidad de puntos en cada región del gráfico.

---

### 12. scatterplot con alpha para transparencia

```python
sns.scatterplot(data=ventas, x='precio_unitario', y='cantidad',
                hue='categoria', alpha=0.3)
plt.title('Transparencia alpha=0.3 — zonas densas más oscuras')
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

*12. scatterplot con alpha para transparencia.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicación**: `alpha` controla la opacidad (0=invisible, 1=sólido). Con 1330 puntos, alpha bajo revela zonas de alta densidad que de otro modo serían una mancha negra.

**Interpretación**: Las zonas más oscuras (muchos puntos superpuestos) indican combinaciones precio-cantidad frecuentes: ej. productos baratos (<$1000) con cantidades entre 5-15 unidades.

---

### 13. lineplot con markers en puntos

```python
diario = ventas.groupby('fecha')['ingreso'].sum().reset_index()

sns.lineplot(data=diario.iloc[:14], x='fecha', y='ingreso',
             marker='o', markersize=8, linestyle='--')
plt.title('Ingresos diarios primeras 2 semanas con marcadores')
plt.xlabel('Fecha')
plt.ylabel('Ingreso ($)')
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

*13. lineplot con markers en puntos.*

1. `diario = ventas.groupby('fecha')['ingreso'].sum().reset_index()` — Agrupa los datos por la columna eRealiza la operación indicada con los parámetros definidos..

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicación**: `marker='o'` añade círculos en cada punto de datos. `markersize=8` controla su tamaño. `linestyle='--'` usa línea punteada. Tomamos solo 14 días para que los marcadores sean legibles.

**Interpretación**: Cada día se marca explícitamente. Se observa el patrón semanal con caídas los fines de semana.

---

### 14. sns.set_palette("Paired") + scatterplot

```python
sns.set_palette("Paired")

sns.scatterplot(data=ventas, x='precio_unitario', y='cantidad',
                hue='sucursal', style='sucursal')
plt.title('Paleta Paired — colores emparejados')
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

*14. sns.set_palette("Paired") + scatterplot.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicación**: `"Paired"` es una paleta cualitativa de colores emparejados (claro-oscuro). `style='sucursal'` asigna marcadores distintos (círculo, triángulo, cuadrado) a cada sucursal, mejorando la discriminación.

**Interpretación**: Con 9 sucursales, el `style` adicional ayuda a distinguir grupos cuando los colores son similares.

---

### 15. Múltiples subplots con plt.subplots(1,2)

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: scatter precio vs cantidad
sns.scatterplot(data=ventas, x='precio_unitario', y='cantidad',
                hue='categoria', ax=axes[0])
axes[0].set_title('Precio vs Cantidad')

# Subplot 2: scatter margen vs cantidad
sns.scatterplot(data=ventas, x='margen', y='cantidad',
                hue='categoria', ax=axes[1])
axes[1].set_title('Margen vs Cantidad')

plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*15. Múltiples subplots con plt.subplots(1,2).*

1. Subplot 1: scatter precio vs cantidad
2. Subplot 2: scatter margen vs cantidad

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicación**: `plt.subplots(1, 2)` crea 2 ejes en una fila. Cada `sns.scatterplot` recibe su `ax` correspondiente. `figsize` debe ser más ancho para albergar ambos.

**Interpretación**: Comparar ambas variables. Precio y margen están correlacionados pero no idénticos. Algunos productos de bajo precio tienen margen alto (% alto) aunque margen absoluto bajo.

---

### 16. Guardar figura con plt.savefig

```python
sns.scatterplot(data=ventas, x='precio_unitario', y='cantidad',
                hue='categoria', alpha=0.5)
plt.title('Scatterplot guardado a archivo')
plt.legend(bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig('scatter_ventas.png', dpi=150, bbox_inches='tight')
print("Figura guardada como scatter_ventas.png")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*16. Guardar figura con plt.savefig.*

1. `print("Figura guardada como scatter_ventas.png")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicación**: `plt.savefig` guarda a archivo. `dpi=150` define resolución. `bbox_inches='tight'` recorta bordes blancos sobrantes. Debe llamarse antes de `plt.show()` porque `show()` limpia la figura.

**Interpretación**: Exportar figuras permite usarlas en informes, dashboards o presentaciones.

---

### 17. Título con plt.title y etiquetas con plt.xlabel/ylabel

```python
fig, ax = plt.subplots(figsize=(10, 6))

sns.lineplot(data=diario_suc, x='fecha', y='ingreso',
             hue='sucursal', alpha=0.7, ax=ax)

ax.set_title('Ingresos Diarios por Sucursal — Primer Semestre 2024',
             fontsize=16, fontweight='bold')
ax.set_xlabel('Fecha', fontsize=13)
ax.set_ylabel('Ingreso ($)', fontsize=13)
ax.legend(bbox_to_anchor=(1.05, 1), title='Sucursal')
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*17. Título con plt.title y etiquetas con plt.xlabel/ylabel.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicación**: Personalización completa: título con negrita, etiquetas con fontSize mayor, leyenda con `title`. `ax.set_` es el equivalente a `plt.` pero sobre el objeto Axes.

**Interpretación**: Un gráfico bien etiquetado comunica la información sin necesidad de texto adicional.

---

### 18. Leyenda personalizada (loc, title, fuera del gráfico)

```python
sns.scatterplot(data=ventas, x='precio_unitario', y='cantidad',
                hue='categoria', style='categoria',
                palette='Set2')
plt.title('Leyenda personalizada — fuera del gráfico')
plt.legend(loc='upper left', title='Categoría',
           frameon=True, shadow=True,
           bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*18. Leyenda personalizada (loc, title, fuera del gráfico).*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicación**: `plt.legend(loc='upper left', bbox_to_anchor=(1,1))` coloca la leyenda a la derecha. `frameon=True` con `shadow=True` le da un recuadro con sombra. `palette='Set2'` es una paleta suave para categorías.

**Interpretación**: Mover la leyenda fuera evita que oculte datos. Ideal para gráficos con muchas categorías.

---

## Resumen

En este módulo aprendiste a usar Seaborn para crear gráficos de dispersión y líneas aplicados a datos de ventas:

| Herramienta | Propósito en ventas |
|-------------|---------------------|
| `sns.scatterplot` | Explorar relación precio-cantidad, detectar outliers, segmentar por categoría |
| `sns.lineplot` | Visualizar tendencias temporales de ingresos, comparar sucursales, mostrar IC |
| `sns.set_theme / style / context` | Controlar apariencia global de todos los gráficos |
| `sns.color_palette / set_palette` | Elegir paletas que comuniquen mejor (divergentes, secuenciales, cualitativas) |
| `sns.despine` | Eliminar bordes innecesarios para un look más limpio |
| `fig, ax + plt.subplots` | Control fino sobre layout, múltiples gráficos, guardado |

Los scatterplots son ideales para **analizar correlaciones** entre variables numéricas. Los lineplots revelan **patrones temporales** y **estacionalidades**. La personalización (temas, paletas, leyendas) transforma gráficos por defecto en visualizaciones profesionales listas para informes.

---

## Ejercicios

1. Crea un scatterplot de `precio_unitario` vs `margen` con `hue='categoria'`. ¿Qué categoría tiene la relación más lineal?

2. Usa `sns.lineplot` para mostrar la tendencia de `cantidad` vendida promedio por día de la semana, con IC del 90%.

3. Genera una figura de 2 filas × 1 columna: arriba scatter precio vs margen; abajo scatter precio vs cantidad. Usa `fig, axes = plt.subplots(2, 1)`.

4. Crea un scatterplot con `sns.set_context("poster")` y `sns.set_style("darkgrid")`. Observa el cambio visual.

5. Usa `sns.color_palette("magma", 5)` para generar 5 colores y asígnalos manualmente a 5 sucursales en un scatterplot.

6. Crea un lineplot del `ingreso` diario promedio con `style='mes'` (como categórico) y `marker='s'`. ¿Se ven patrones mensuales?

7. Guarda un scatterplot de margen vs cantidad con `hue='descuento'` (tratar descuento como categórico con `hue=ventas['descuento'].astype(str)`) a 200 dpi.

8. Reproduce el ejemplo 15 pero invierte: precio vs margen a la izquierda, margen vs cantidad a la derecha. Usa paleta "dark" para el primero y "muted" para el segundo.
