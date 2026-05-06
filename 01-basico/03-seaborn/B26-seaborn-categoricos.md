# Modulo B26 — Seaborn: Graficos categoricos (barplot, countplot, catplot, stripplot, swarmplot, pointplot)

## Teoria

Los **graficos categoricos** de Seaborn permiten visualizar la relacion entre una variable numerica y una o mas variables categoricas. Son esenciales en analytics de ventas para comparar metricas (ingreso, margen, cantidad) entre grupos (sucursales, categorias, meses).

### Principales funciones

| Funcion | Tipo de grafico | Uso tipico en ventas |
|---------|-----------------|----------------------|
| `sns.barplot` | Barras con agregacion estadistica | Ingreso total por categoria, promedio por sucursal |
| `sns.countplot` | Conteo de observaciones por categoria | Frecuencia de ventas por dia, sucursal mas activa |
| `sns.catplot` | Figura completa (facetado) con kind= | Bar/box/violin/point facetados por col/row |
| `sns.stripplot` | Puntos individuales con jitter | Dispersion de precios por categoria |
| `sns.swarmplot` | Puntos sin solapamiento | Distribucion detallada con muestra pequena |
| `sns.pointplot` | Puntos con linea de tendencia | Evolucion de metricas entre categorias |

### Parametros clave transversales

- **`estimator`**: funcion de agregacion (`sum`, `mean`, `median`, `count`, `std`)
- **`errorbar`**: tipo de barra de error (`("ci", 95)`, `("sd")`, `None`)
- **`hue`**: segunda variable categorica para agrupar
- **`order` / `hue_order`**: control manual del orden de categorias
- **`dodge`**: separar barras con hue (`True`) o apilarlas (`False`)
- **`orient`**: `"v"` (vertical) o `"h"` (horizontal)
- **`palette`**: paleta de colores
- **`capsize`**, **`errwidth`**: ajuste de barras de error
- **`saturation`**: saturacion de color (1.0 = maximo)

### Cuando usar cada uno

| Situacion | Grafico |
|-----------|---------|
| Comparar totales entre grupos | `barplot(estimator=sum)` |
| Comparar promedios con IC | `barplot(estimator=mean, errorbar=("ci",95))` |
| Ver frecuencia de ocurrencias | `countplot` |
| Explorar distribucion + tendencia | `catplot(kind="box"/"violin")` |
| Ver tendencia entre categorias ordenadas | `pointplot` |
| Visualizar puntos individuales | `stripplot` o `swarmplot` |

---

## Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ventas = pd.read_csv("../datos/ventas.csv")
inventario = pd.read_csv("../datos/inventario.csv")

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

1. `import pandas as pd` — Importa las librerías necesarias para el análisis.
2. `import numpy as np` — Importa las librerías necesarias para el análisis.
3. `import matplotlib.pyplot as plt` — Importa las librerías necesarias para el análisis.
4. `import seaborn as sns` — Importa las librerías necesarias para el análisis.
5. `ventas = pd.read_csv("../datos/ventas.csv")` — Carga los datos desde el archivo CSV.
6. `inventario = pd.read_csv("../datos/inventario.csv")` — Carga los datos desde el archivo CSV.
7. `ventas['fecha'] = pd.to_datetime(ventas['fecha'])` — Convierte la columna a formato datetime.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 18 Ejemplos practicos

---

### 1. barplot: ingreso total por categoria de producto

```python
sns.barplot(data=ventas, x='categoria', y='ingreso', estimator=sum)
plt.title('Ingreso Total por Categoria')
plt.xlabel('Categoria')
plt.ylabel('Ingreso Total ($)')
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

*1. barplot: ingreso total por categoria de producto.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicacion**: `estimator=sum` hace que la altura de cada barra sea la suma de `ingreso` para esa categoria. Seaborn calcula automaticamente el total agrupando por `x`. `xticks(rotation=45)` evita que las etiquetas largas se solapen.

**Interpretacion**: Electronica y Muebles generan el mayor ingreso total. Papeleria y Redes estan en la parte baja. Esto refleja tanto el precio unitario como el volumen de ventas de cada categoria.

---

### 2. barplot: cantidad promedio vendida por sucursal

```python
sns.barplot(data=ventas, x='sucursal', y='cantidad', estimator=mean,
            errorbar=("ci", 95))
plt.title('Cantidad Promedio Vendida por Sucursal (IC 95%)')
plt.xlabel('Sucursal')
plt.ylabel('Cantidad Promedio')
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

*2. barplot: cantidad promedio vendida por sucursal.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicacion**: `estimator=mean` (valor por defecto) muestra el promedio. `errorbar=("ci", 95)` agrega intervalos de confianza del 95% como lineas verticales sobre cada barra, indicando la precision de la estimacion.

**Interpretacion**: La mayoria de sucursales tienen promedios entre 7 y 10 unidades por venta. Las barras de error se solapan, sugiriendo que no hay diferencias significativas entre sucursales en cantidad promedio.

---

### 3. barplot: margen promedio por categoria con errorbar

```python
sns.barplot(data=ventas, x='categoria', y='margen',
            errorbar=("sd"), capsize=0.2, errwidth=2)
plt.title('Margen Promedio por Categoria (desv. estandar)')
plt.xlabel('Categoria')
plt.ylabel('Margen Promedio ($)')
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

*3. barplot: margen promedio por categoria con errorbar.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicacion**: `errorbar=("sd")` muestra una desviacion estandar en lugar de IC. `capsize=0.2` anade pequenas tapas en los extremos de las barras de error. `errwidth=2` hace las barras de error mas gruesas.

**Interpretacion**: Muebles y Electronica tienen los margenes promedio mas altos pero tambien la mayor variabilidad (barras de error grandes). Papeleria y Software tienen margenes bajos y estables.

---

### 4. barplot: ingreso por sucursal con hue=categoria (barras agrupadas)

```python
sns.barplot(data=ventas, x='sucursal', y='ingreso', hue='categoria',
            estimator=sum, palette='Set2')
plt.title('Ingreso por Sucursal y Categoria')
plt.xlabel('Sucursal')
plt.ylabel('Ingreso Total ($)')
plt.legend(bbox_to_anchor=(1.05, 1), title='Categoria')
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

*4. barplot: ingreso por sucursal con hue=categoria (barras agrupadas).*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicacion**: `hue='categoria'` divide cada sucursal en barras adyacentes, una por categoria. `dodge=True` (valor por defecto) separa las barras. `palette='Set2'` es una paleta cualitativa suave.

**Interpretacion**: Sucursal Merida lidera en Electronica y Muebles. Matriz CDMX tiene una distribucion mas equilibrada entre categorias. Algunas sucursales no venden ciertas categorias (barras ausentes).

---

### 5. countplot: frecuencia de ventas por dia_semana

```python
sns.countplot(data=ventas, x='dia_semana', palette='viridis')
plt.title('Frecuencia de Ventas por Dia de Semana')
plt.xlabel('Dia (0=Domingo, 6=Sabado)')
plt.ylabel('Numero de Transacciones')
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*5. countplot: frecuencia de ventas por dia_semana.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicacion**: `countplot` cuenta cuantas filas del DataFrame caen en cada categoria de `x`. No necesita `y`. `palette='viridis'` asigna colores secuenciales.

**Interpretacion**: Los dias 2-3 (martes-miercoles) concentran la mayor cantidad de transacciones. El dia 0 (domingo) tiene menor actividad. Esto sugiere que la demanda es estacional dentro de la semana.

---

### 6. countplot: frecuencia por sucursal (horizontal con orient="h")

```python
sns.countplot(data=ventas, y='sucursal', orient='h', palette='muted',
              order=ventas['sucursal'].value_counts().index)
plt.title('Frecuencia de Ventas por Sucursal')
plt.xlabel('Numero de Transacciones')
plt.ylabel('Sucursal')
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*6. countplot: frecuencia por sucursal (horizontal con orient="h").*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicacion**: `orient='h'` intercambia los ejes: sucursales en el eje y, conteo en x. `order` ordena las barras de mayor a menor frecuencia usando `value_counts().index`.

**Interpretacion**: Sucursal Merida y Matriz CDMX son las que mas transacciones registran. Sucursal Cancun y Tijuana estan al final. El orden descendente facilita la comparacion visual.

---

### 7. countplot: frecuencia por categoria con hue=mes

```python
sns.countplot(data=ventas, x='categoria', hue='mes', palette='Set1')
plt.title('Frecuencia de Ventas por Categoria y Mes')
plt.xlabel('Categoria')
plt.ylabel('Conteo')
plt.legend(title='Mes', bbox_to_anchor=(1.05, 1))
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

*7. countplot: frecuencia por categoria con hue=mes.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicacion**: `hue='mes'` desglosa cada barra en segmentos de color por mes. `dodge=True` separa las barras por mes lado a lado. `palette='Set1'` tiene colores vivos para distinguir los 6 meses.

**Interpretacion**: Electronica y Perifericos mantienen alta frecuencia todos los meses. Algunas categorias como Redes y Camaras tienen pocas transacciones y no aparecen en todos los meses.

---

### 8. catplot kind="bar": ingreso por sucursal facetado por categoria

```python
sns.catplot(data=ventas, x='sucursal', y='ingreso', col='categoria',
            kind='bar', estimator=sum, col_wrap=3, height=3,
            sharex=False)
plt.suptitle('Ingreso por Sucursal facetado por Categoria', y=1.02)
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

*8. catplot kind="bar": ingreso por sucursal facetado por categoria.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicacion**: `catplot` crea una figura con multiples subplots. `col='categoria'` genera un panel por categoria. `col_wrap=3` organiza en 3 columnas. `sharex=False` permite que cada panel tenga su propia escala en x. `kind='bar'` usa barplot internamente.

**Interpretacion**: Se observa que Electronica y Muebles dominan en la mayoria de sucursales. Cada categoria tiene un patron distinto de distribucion geografica. Algunas categorias (Redes, Camaras) solo aparecen en sucursales especificas.

---

### 9. catplot kind="box": distribucion precios por categoria

```python
sns.catplot(data=ventas, x='categoria', y='precio_unitario',
            kind='box', height=4, aspect=2)
plt.title('Distribucion de Precios por Categoria')
plt.xlabel('Categoria')
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

*9. catplot kind="box": distribucion precios por categoria.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicacion**: `kind='box'' utiliza boxplot internamente. `aspect=2` hace el grafico mas ancho que alto (proporcion 2:1). El boxplot muestra mediana, cuartiles y outliers.

**Interpretacion**: Electronica y Muebles tienen los precios mas altos y mayor dispersion. Papeleria y Software tienen precios bajos y concentrados. Electronica presenta multiples outliers (productos premium).

---

### 10. catplot kind="violin": distribucion margenes por sucursal

```python
sns.catplot(data=ventas, x='sucursal', y='margen',
            kind='violin', height=4, aspect=2, palette='pastel')
plt.title('Distribucion de Margenes por Sucursal')
plt.xlabel('Sucursal')
plt.ylabel('Margen ($)')
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

*10. catplot kind="violin": distribucion margenes por sucursal.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicacion**: `kind='violin'` combina boxplot con KDE: el ancho representa densidad. `palette='pastel'` da colores suaves. El violin muestra la forma completa de la distribucion (multimodalidad, asimetria).

**Interpretacion**: Todas las sucursales tienen distribucion sesgada a la derecha (mayoria de margenes bajos, algunos altos). Sucursal Merida muestra una cola mas larga hacia margenes altos. Algunas sucursales tienen distribuciones bimodales.

---

### 11. catplot kind="point": tendencia de ventas por mes

```python
sns.catplot(data=ventas, x='mes', y='ingreso',
            kind='point', estimator=sum, capsize=0.2,
            height=4, aspect=2)
plt.title('Tendencia de Ingreso Total por Mes')
plt.xlabel('Mes')
plt.ylabel('Ingreso Total ($)')
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*11. catplot kind="point": tendencia de ventas por mes.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicacion**: `kind='point'` muestra puntos conectados por una linea para cada categoria de `x`. La altura del punto representa el estimador (suma total aqui). `capsize=0.2` anade barras de error. Es util para ver tendencias en datos categoricos ordenados.

**Interpretacion**: El ingreso total muestra una tendencia creciente del mes 1 al 6, con un pico en el mes 5. Esto podria deberse a estacionalidad o crecimiento del negocio.

---

### 12. stripplot: dispersion de precios por categoria (jitter=True)

```python
plt.figure(figsize=(10, 5))
sns.stripplot(data=ventas, x='categoria', y='precio_unitario',
              jitter=True, alpha=0.5, size=4, palette='Set2')
plt.title('Dispersion de Precios por Categoria')
plt.xlabel('Categoria')
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

*12. stripplot: dispersion de precios por categoria (jitter=True).*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicacion**: `stripplot` dibuja cada punto individual. `jitter=True` agrega ruido horizontal para evitar superposicion. `alpha=0.5` da transparencia. `size=4` controla el diametro del punto.

**Interpretacion**: Se ve claramente la concentracion de Electronica y Muebles en precios altos. Papeleria y Software tienen todos sus puntos en el rango bajo (< $2000). La dispersion dentro de cada categoria revela la variedad de productos.

---

### 13. swarmplot: distribucion de cantidades por dia_semana

```python
plt.figure(figsize=(10, 5))
sns.swarmplot(data=ventas, x='dia_semana', y='cantidad',
              palette='husl', size=3, alpha=0.7)
plt.title('Distribucion de Cantidades por Dia de Semana')
plt.xlabel('Dia (0=Domingo, 6=Sabado)')
plt.ylabel('Cantidad')
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*13. swarmplot: distribucion de cantidades por dia_semana.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicacion**: `swarmplot` organiza los puntos sin solapamiento usando un algoritmo de "abeja". Cada punto es visible sin superposicion. `palette='husl'` usa el espacio de color HUSL (matiz, saturacion, luminosidad).

**Interpretacion**: La distribucion de cantidades es similar todos los dias. La mayoria de transacciones son de 1-15 unidades. Los dias 3-4 (miercoles-jueves) tienen mas puntos en el rango medio-alto. Con 1330 puntos el swarmplot puede ser lento.

---

### 14. pointplot: evolucion de ventas por mes con markers

```python
sns.pointplot(data=ventas, x='mes', y='ingreso', estimator=sum,
              markers='o', linestyles='--', capsize=0.1,
              color='crimson', errwidth=1.5)
plt.title('Evolucion de Ingreso Total por Mes')
plt.xlabel('Mes')
plt.ylabel('Ingreso Total ($)')
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*14. pointplot: evolucion de ventas por mes con markers.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicacion**: `pointplot` funciona como barplot pero con puntos y lineas. `markers='o'` usa circulos. `linestyles='--'` linea punteada. `color='crimson'` color unico. Es ideal para series ordenadas (meses, etapas).

**Interpretacion**: La linea ascendente muestra crecimiento mes a mes. Los markers en cada mes permiten leer el valor exacto. La pendiente es mas pronunciada entre los meses 3-5, indicando un periodo de expansion.

---

### 15. catplot kind="bar" con col_wrap=3

```python
sns.catplot(data=ventas, x='sucursal', y='cantidad',
            col='mes', kind='bar', col_wrap=3,
            height=3, sharey=False, palette='coolwarm')
plt.suptitle('Cantidad Promedio por Sucursal y Mes', y=1.02)
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

*15. catplot kind="bar" con col_wrap=3.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicacion**: `col='mes'` genera 6 paneles (uno por mes). `col_wrap=3` los organiza en 2 filas de 3. `sharey=False` permite que cada panel tenga su propia escala y, revelando patrones que la escala comunitaria ocultaria.

**Interpretacion**: Algunos meses muestran sucursales con picos de cantidad (ej. mes 5 en Merida). La escala independiente ayuda a ver la estructura interna de cada mes sin distorsion por valores extremos de otros meses.

---

### 16. Comparar barplot(estimator=sum) vs barplot(estimator=mean)

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.barplot(data=ventas, x='categoria', y='ingreso',
            estimator=sum, ax=axes[0], palette='Blues_d')
axes[0].set_title('Ingreso TOTAL por Categoria')
axes[0].set_xlabel('Categoria')
axes[0].set_ylabel('Suma de Ingreso ($)')
axes[0].tick_params(axis='x', rotation=45)

sns.barplot(data=ventas, x='categoria', y='ingreso',
            estimator=mean, ax=axes[1], palette='Reds_d')
axes[1].set_title('Ingreso PROMEDIO por Categoria')
axes[1].set_xlabel('Categoria')
axes[1].set_ylabel('Media de Ingreso ($)')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*16. Comparar barplot(estimator=sum) vs barplot(estimator=mean).*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicacion**: Dos barplots lado a lado: izquierda con `estimator=sum` (total), derecha con `estimator=mean` (promedio). `palette='Blues_d'` y `'Reds_d'` son paletas secuenciales oscuras. El mismo dato produce lecturas muy distintas.

**Interpretacion**: Electronica domina en suma (muchas transacciones), pero Muebles tiene el mayor promedio (cada venta es cara). La suma favorece a categorias con muchas ventas; el promedio revela el ticket promedio. Ambos son necesarios segun la pregunta de negocio.

---

### 17. barplot horizontal con catplot kind="bar" orient="h"

```python
sns.catplot(data=ventas, y='sucursal', x='ingreso',
            kind='bar', estimator=sum, orient='h',
            height=5, aspect=1.5, palette='magma')
plt.title('Ingreso Total por Sucursal (horizontal)')
plt.xlabel('Ingreso Total ($)')
plt.ylabel('Sucursal')
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*17. barplot horizontal con catplot kind="bar" orient="h".*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicacion**: `orient='h'` hace barras horizontales. `y='sucursal'` y `x='ingreso'` intercambian los ejes. Las barras horizontales son preferibles cuando las etiquetas del eje y son largas (como nombres de sucursales). `palette='magma'` es una paleta secuencial perceptualmente uniforme.

**Interpretacion**: La orientacion horizontal facilita leer los nombres de las sucursales sin rotacion. El orden de arriba a abajo es el de los datos; se puede combinar con `order` para ordenar por ingreso descendente.

---

### 18. stripplot + boxplot superpuesto en misma figura

```python
plt.figure(figsize=(12, 5))

sns.boxplot(data=ventas, x='categoria', y='precio_unitario',
            palette='Set3', width=0.5, fliersize=0)
sns.stripplot(data=ventas, x='categoria', y='precio_unitario',
              color='black', alpha=0.3, size=3, jitter=True)
plt.title('Distribucion de Precios: Boxplot + Stripplot')
plt.xlabel('Categoria')
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

*18. stripplot + boxplot superpuesto en misma figura.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicacion**: Se superpone un `boxplot` (sin outliers con `fliersize=0`) y un `stripplot` (puntos individuales). El boxplot da el resumen estadistico; el stripplot muestra cada transaccion. `width=0.5` reduce el ancho del boxplot para que los puntos sean visibles.

**Interpretacion**: Se combina lo mejor de ambos: el resumen estadistico (mediana, cuartiles) del boxplot y la granularidad de los puntos individuales. Se aprecia la densidad real detras de cada caja. Electronica tiene cola larga visible tanto en caja como en puntos.

---

## Resumen

Este modulo cubrio las principales herramientas de Seaborn para visualizar relaciones entre variables numericas y categoricas en datos de ventas:

| Funcion | Tipo | Uso principal |
|---------|------|---------------|
| `sns.barplot` | Barras | Comparar metricas agregadas entre grupos |
| `sns.countplot` | Conteo | Frecuencia de ocurrencias por categoria |
| `sns.catplot` | Figura completa | Facetado con kind=bar/box/violin/point/strip/swarm |
| `sns.stripplot` | Puntos con jitter | Ver todos los puntos individuales |
| `sns.swarmplot` | Puntos ordenados | Distribucion detallada sin solapamiento |
| `sns.pointplot` | Puntos + linea | Tendencias entre categorias ordenadas |

Conceptos clave:
- **`estimator`** cambia la pregunta de negocio: `sum` para totales, `mean` para promedios, `median` para valores tipicos
- **`errorbar`** comunica incertidumbre: IC 95% para significancia, SD para variabilidad
- **`hue`** permite segmentar por una segunda variable categorica
- **`catplot` con `col`/`row`** permite facetar y descubrir patrones por subgrupo
- Superponer stripplot + boxplot/violinplot da la maxima informacion: resumen + detalle

---

## Ejercicios

1. Usa `sns.barplot` para mostrar el margen total por sucursal con `estimator=sum` y `palette='Blues_d'`. Ordena las barras de mayor a menor usando `order`.

2. Crea un `sns.countplot` de la columna `mes` con `hue='categoria'` y `stat='percent'` para ver la proporcion de cada categoria por mes. Usa `dodge=False` para barras apiladas.

3. Genera un `sns.catplot` con `kind='violin'` de `precio_unitario` por `categoria` y `row='mes'` (6 filas). ¿Que categoria mantiene su distribucion constante todos los meses?

4. Usa `sns.pointplot` para comparar la `cantidad` promedio por `dia_semana` con `hue='categoria'`, `dodge=True` y `markers` distintos por categoria.

5. Crea un `sns.stripplot` de `margen` por `sucursal` con `jitter=True`, `alpha=0.4`, `size=3`, y `palette='viridis'`. Identifica que sucursal tiene la mayor dispersion.

6. Superpone `sns.violinplot` + `sns.swarmplot` de `precio_unitario` por `categoria` filtrando solo 4 categorias (Electronica, Muebles, Papeleria, Software). Usa `inner='quartile'` en el violin.

7. Usa `sns.catplot` con `kind='bar'` para comparar `margen` promedio por `sucursal` facetado por `mes` con `col_wrap=3` y `sharey=False`. ¿Que sucursal lidera cada mes?

8. Crea un `sns.barplot` de `stock_actual` promedio por `categoria` usando `inventario` como datos. Anade `errorbar=("ci", 95)` y compara con la desviacion estandar con `errorbar=("sd")`.
