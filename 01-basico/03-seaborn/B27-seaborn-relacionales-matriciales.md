# Modulo B27 — Seaborn: Graficos relacionales y matriciales (scatterplot, lineplot, relplot, lmplot, regplot, residplot, jointplot, pairplot, heatmap)

## Teoria

Los **graficos relacionales** exploran la relacion entre dos o mas variables numericas. Los **graficos matriciales** visualizan datos en formato de cuadricula (matriz de correlacion, tablas pivote). Ambos son fundamentales en el analisis exploratorio de ventas.

### Graficos relacionales

| Funcion | Tipo | Uso en ventas |
|---------|------|---------------|
| `sns.scatterplot` | Dispersion 2D | Precio vs cantidad, margen vs ingreso |
| `sns.lineplot` | Lineas con tendencia | Ingreso diario, promedios por mes |
| `sns.relplot` | Figura completa | Scatter/line facetado por col/row |
| `sns.lmplot` | Regresion lineal facetada | Relacion precio-cantidad por categoria |
| `sns.regplot` | Regresion en un solo eje | Relacion lineal/polinomica entre 2 vars |
| `sns.residplot` | Residuales de regresion | Diagnosticos del modelo lineal |
| `sns.jointplot` | Bivariado + marginales | Distribucion conjunta con histogramas/KDE |
| `sns.pairplot` | Matriz de dispersion | Todas las variables numericas en cuadricula |

### Graficos matriciales

| Funcion | Tipo | Uso en ventas |
|---------|------|---------------|
| `sns.heatmap` | Mapa de calor | Correlaciones, tablas pivote sucursal x mes |

### Parametros clave

- **`hue`**, **`size`**, **`style`**: variables adicionales para color, tamano y forma de marcadores
- **`ci`**: intervalo de confianza (`"sd"`, `95`, `None`)
- **`order`**: grado polinomico en regplot/lmplot/residplot
- **`logistic`**, **`robust`**, **`logx`**: variantes de regresion
- **`kind`** en jointplot: `"scatter"`, `"kde"`, `"hex"`, `"reg"`
- **`diag_kind`** en pairplot: `"hist"`, `"kde"`, `"auto"`
- **`corner=True`**: solo triangular inferior en pairplot
- **`annot`**, **`fmt`**, **`cmap`**, **`mask`**, **`center`**: personalizacion de heatmap

### Cuando usar cada uno

| Situacion | Grafico |
|-----------|---------|
| Relacion entre 2 variables numericas | `scatterplot` o `regplot` |
| Tendencia temporal con incertidumbre | `lineplot` con `ci` |
| Comparar relaciones entre grupos | `lmplot` con `hue` o `col` |
| Distribucion bivariada + marginales | `jointplot` |
| Vision general de todas las variables | `pairplot` |
| Correlaciones entre multiples variables | `heatmap` de matriz de correlacion |
| Tabla resumen con colores | `heatmap` con `annot` y `fmt` |

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

## 20 Ejemplos practicos

---

### 1. scatterplot: precio_unitario vs cantidad (cada venta)

```python
sns.scatterplot(data=ventas, x='precio_unitario', y='cantidad', alpha=0.5)
plt.title('Precio Unitario vs Cantidad Vendida')
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

*1. scatterplot: precio_unitario vs cantidad (cada venta).*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicacion**: Cada punto es una transaccion. `alpha=0.5` da transparencia para ver zonas de alta densidad. `scatterplot` no agrega los datos; muestra cada observacion individual.

**Interpretacion**: Existe una relacion inversa: a mayor precio, menor cantidad. La mayoria de puntos se concentran en precios bajos (< $4000) con cantidades entre 1 y 15 unidades. Hay pocas transacciones de alto precio con alta cantidad.

---

### 2. scatterplot: precio vs cantidad con hue=categoria y size=margen

```python
sns.scatterplot(data=ventas, x='precio_unitario', y='cantidad',
                hue='categoria', size='margen', sizes=(20, 400),
                alpha=0.6, palette='deep')
plt.title('Precio vs Cantidad: color=Categoria, tamano=Margen')
plt.xlabel('Precio Unitario ($)')
plt.ylabel('Cantidad')
plt.legend(bbox_to_anchor=(1.05, 1), title='Categoria')
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*2. scatterplot: precio vs cantidad con hue=categoria y size=margen.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicacion**: `hue='categoria'` colorea por categoria. `size='margen'` escala el punto segun el margen absoluto. `sizes=(20, 400)` define el rango de tamanos. `palette='deep'` da colores intensos.

**Interpretacion**: Electronica (naranja) y Muebles (rojo) tienen puntos grandes (alto margen) y precios altos. Papeleria (verde) y Software (purpura) tienen puntos pequenos (bajo margen) y precios bajos. El tamano anade una tercera dimension visual.

---

### 3. lineplot: tendencia de ingreso diario con ci="sd"

```python
diario = ventas.groupby('fecha')['ingreso'].sum().reset_index()

sns.lineplot(data=diario, x='fecha', y='ingreso', ci='sd')
plt.title('Ingreso Diario con Desviacion Estandar')
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

*3. lineplot: tendencia de ingreso diario con ci="sd".*

1. `diario = ventas.groupby('fecha')['ingreso'].sum().reset_index()` — Agrupa los datos por la columna eRealiza la operación indicada con los parámetros definidos..

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicacion**: `lineplot` con `ci='sd'` muestra una banda de +/- 1 desviacion estandar alrededor de la linea. Como hay un solo valor por fecha, la banda es cero (sin variabilidad). Si hubiera multiples observaciones por fecha, mostraria la dispersion.

**Interpretacion**: El ingreso diario oscila entre $50K y $400K con picos en ciertos dias. No hay una tendencia clara de crecimiento, sino fluctuaciones diarias. La banda `ci='sd'` no se ve porque solo hay un valor por fecha.

---

### 4. lineplot: tendencia por sucursal con style diferentes

```python
diario_suc = ventas.groupby(['fecha', 'sucursal'])['ingreso'].sum().reset_index()

plt.figure(figsize=(12, 5))
sns.lineplot(data=diario_suc, x='fecha', y='ingreso',
             hue='sucursal', style='sucursal',
             markers=False, dashes=False, alpha=0.7)
plt.title('Ingreso Diario por Sucursal')
plt.xlabel('Fecha')
plt.ylabel('Ingreso ($)')
plt.legend(bbox_to_anchor=(1.05, 1), title='Sucursal')
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*4. lineplot: tendencia por sucursal con style diferentes.*

1. `diario_suc = ventas.groupby(['fecha', 'sucursal'])['ingreso'].sum().reset_index()` — Agrupa los datos por la columna eRealiza la operación indicada con los parámetros definidos..

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicacion**: `style='sucursal'` asigna un estilo de linea distinto (solida, punteada, rayada) a cada sucursal ademas del color. `dashes=False` evita lineas punteadas (usa estilos por defecto). `markers=False` quita marcadores en cada punto.

**Interpretacion**: Las sucursales siguen tendencias paralelas en el tiempo. Sucursal Merida y Matriz CDMX estan consistentemente arriba. Sucursal Cancun y Tijuana tienen los ingresos mas bajos. El estilo de linea ayuda a distinguir sucursales en blanco y negro.

---

### 5. relplot: scatter facetado por categoria

```python
sns.relplot(data=ventas, x='precio_unitario', y='cantidad',
            col='categoria', col_wrap=3, height=3,
            hue='sucursal', alpha=0.5, s=20)
plt.suptitle('Precio vs Cantidad facetado por Categoria', y=1.02)
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*5. relplot: scatter facetado por categoria.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicacion**: `relplot` con `kind='scatter'` (valor por defecto). `col='categoria'` crea un panel por categoria. `col_wrap=3` en 3 columnas. `s=20` controla el tamano de los puntos. `hue='sucursal'` colorea por sucursal dentro de cada panel.

**Interpretacion**: Cada categoria tiene un rango de precios distinto. Electronica y Muebles abarcan todo el rango. Papeleria y Software solo aparecen en precios bajos. El facetado evita la saturacion visual de tenerlo todo en un solo grafico.

---

### 6. relplot: line facetado por sucursal con col_wrap=3

```python
diario_suc = ventas.groupby(['fecha', 'sucursal'])['ingreso'].sum().reset_index()

sns.relplot(data=diario_suc, x='fecha', y='ingreso',
            col='sucursal', col_wrap=3, kind='line',
            height=3, aspect=1.5, color='steelblue')
plt.suptitle('Ingreso Diario por Sucursal (facetado)', y=1.02)
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

*6. relplot: line facetado por sucursal con col_wrap=3.*

1. `diario_suc = ventas.groupby(['fecha', 'sucursal'])['ingreso'].sum().reset_index()` — Agrupa los datos por la columna eRealiza la operación indicada con los parámetros definidos..

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicacion**: `kind='line'` en `relplot` produce lineplots facetados. `col='sucursal'` separa cada sucursal en su propio panel. `aspect=1.5` hace cada panel mas ancho que alto. `color='steelblue'` color unico para todos.

**Interpretacion**: Cada sucursal tiene su propia dinamica diaria. Sucursal Merida muestra mayor volatilidad. Sucursal Cancun tiene ingresos mas estables pero bajos. El facetado permite ver cada serie sin el ruido visual de las demas.

---

### 7. lmplot: regresion precio vs cantidad con IC

```python
sns.lmplot(data=ventas, x='precio_unitario', y='cantidad',
           height=5, aspect=1.5, ci=95)
plt.title('Regresion Lineal: Precio vs Cantidad (IC 95%)')
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

*7. lmplot: regresion precio vs cantidad con IC.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicacion**: `lmplot` ajusta una regresion lineal y la dibuja con su intervalo de confianza (sombreado). `ci=95` muestra el IC de la pendiente. `height` y `aspect` controlan el tamano de la figura.

**Interpretacion**: La recta de regresion decreciente confirma la relacion inversa entre precio y cantidad. El IC es estrecho, indicando que la pendiente esta bien estimada. Sin embargo, la dispersion alrededor de la recta es alta (R^2 bajo), lo que sugiere que el precio solo explica parte de la variacion en cantidad.

---

### 8. lmplot: regresion por categoria con hue

```python
sns.lmplot(data=ventas, x='precio_unitario', y='cantidad',
           hue='categoria', height=5, aspect=1.5,
           palette='Set2', ci=None)
plt.title('Regresion Precio vs Cantidad por Categoria')
plt.xlabel('Precio Unitario ($)')
plt.ylabel('Cantidad')
plt.legend(bbox_to_anchor=(1.05, 1), title='Categoria')
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*8. lmplot: regresion por categoria con hue.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicacion**: `hue='categoria'` ajusta una recta de regresion independiente para cada categoria. `ci=None` elimina el sombreado de IC para mayor claridad. Cada categoria tiene su propio color y recta.

**Interpretacion**: La pendiente negativa es consistente en todas las categorias, pero la magnitud varia. Electronica tiene la pendiente mas pronunciada (el precio afecta fuertemente la cantidad). Papeleria tiene pendiente casi plana (rango de precio estrecho). Cada categoria ocupa una region distinta del espacio precio-cantidad.

---

### 9. regplot: con order=2 (polinomica) para relacion no lineal

```python
sns.regplot(data=ventas, x='precio_unitario', y='cantidad',
            order=2, ci=95, scatter_kws={'alpha': 0.3},
            line_kws={'color': 'red', 'linewidth': 2})
plt.title('Regresion Polinomica (orden 2): Precio vs Cantidad')
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

*9. regplot: con order=2 (polinomica) para relacion no lineal.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicacion**: `order=2` ajusta un polinomio de grado 2 (parabola). `scatter_kws` pasa parametros a `plt.scatter` (puntos). `line_kws` pasa parametros a `plt.plot` (linea de regresion). `ci=95` muestra el IC de la curva.

**Interpretacion**: La curva polinomica captura mejor la relacion que una recta: las cantidades caen rapidamente al aumentar el precio desde 0, luego la pendiente se estabiliza. El IC se ensancha en los extremos donde hay menos datos. Esto sugiere que la relacion no es lineal.

---

### 10. residplot: analizar residuos de regresion lineal

```python
sns.residplot(data=ventas, x='precio_unitario', y='cantidad',
              lowess=True, scatter_kws={'alpha': 0.4},
              line_kws={'color': 'red', 'linewidth': 2})
plt.title('Residuos de Regresion Lineal con LOWESS')
plt.xlabel('Precio Unitario ($)')
plt.ylabel('Residuos')
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*10. residplot: analizar residuos de regresion lineal.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicacion**: `residplot` calcula los residuos (observado - predicho) de una regresion lineal. `lowess=True` superpone una curva suavizada (LOWESS) para detectar patrones en los residuos. Si los residuos son aleatorios, la curva LOWESS debe ser plana cerca de 0.

**Interpretacion**: La curva LOWESS muestra un patron: residuos positivos a precios bajos (subestimacion), luego negativos (sobrestimacion) y de nuevo positivos. Este patron sistematico indica que la relacion no es estrictamente lineal y justifica el modelo polinomico del ejemplo anterior.

---

### 11. jointplot kind="scatter": precio vs margen con histogramas marginales

```python
sns.jointplot(data=ventas, x='precio_unitario', y='margen',
              kind='scatter', alpha=0.4, height=6)
plt.suptitle('Precio vs Margen (scatter + histogramas)', y=1.02)
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*11. jointplot kind="scatter": precio vs margen con histogramas marginales.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicacion**: `jointplot` combina un scatterplot central con histogramas (o KDE) en los margenes superior y derecho. `kind='scatter'` usa dispersion. `height=6` controla el tamano total.

**Interpretacion**: Precio y margen estan fuertemente correlacionados (los productos caros generan mas margen absoluto). Los histogramas marginales muestran que ambas variables tienen distribucion sesgada a la derecha. Algunos puntos se desvian de la tendencia (productos con margen alto relativo a su precio).

---

### 12. jointplot kind="kde": densidad conjunta precio-margen

```python
sns.jointplot(data=ventas, x='precio_unitario', y='margen',
              kind='kde', fill=True, cmap='viridis', height=6)
plt.suptitle('Densidad Conjunta Precio-Margen (KDE)', y=1.02)
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*12. jointplot kind="kde": densidad conjunta precio-margen.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicacion**: `kind='kde'` estima la densidad 2D con curvas de nivel. `fill=True` rellena las areas con color. `cmap='viridis'` mapa de color. Las lineas de nivel conectan puntos de igual densidad. Los marginales muestran KDE 1D.

**Interpretacion**: La densidad se concentra en precios < $6000 y margenes < $15000 (zona amarilla). Hay un "brazo" de densidad que se extiende hacia precios y margenes altos, correspondiente a Electronica y Muebles. La densidad revela donde estan la mayoria de las transacciones.

---

### 13. jointplot kind="hex": hexbin para muchos puntos

```python
sns.jointplot(data=ventas, x='precio_unitario', y='margen',
              kind='hex', gridsize=20, cmap='Blues', height=6)
plt.suptitle('Hexbin: Precio vs Margen (para 1330 puntos)', y=1.02)
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*13. jointplot kind="hex": hexbin para muchos puntos.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicacion**: `kind='hex'` usa hexbin (hexagonos) en lugar de puntos. Cada hexagono cuenta cuantos puntos caen dentro. `gridsize=20` controla el numero de hexagonos por eje. El color del hexagono indica la densidad local.

**Interpretacion**: Los hexagonos mas oscuros (alta densidad) se concentran en la esquina inferior izquierda (precios y margenes bajos). Es mas efectivo que scatterplot con 1330 puntos porque evita la superposicion total y revela la densidad real.

---

### 14. jointplot kind="reg": con recta de regresion

```python
sns.jointplot(data=ventas, x='precio_unitario', y='margen',
              kind='reg', height=6, ci=95,
              joint_kws={'scatter_kws': {'alpha': 0.3},
                         'line_kws': {'color': 'red'}})
plt.suptitle('Precio vs Margen con Regresion Lineal', y=1.02)
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*14. jointplot kind="reg": con recta de regresion.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicacion**: `kind='reg'` muestra scatterplot + recta de regresion + histogramas marginales. `joint_kws` pasa parametros al grafico central. `ci=95` muestra el IC de la regresion.

**Interpretacion**: La recta de regresion confirma la correlacion positiva entre precio y margen. El IC es estrecho, indicando estimacion precisa de la pendiente. Los histogramas marginales confirman la asimetria de ambas variables.

---

### 15. pairplot: todas las variables numericas de ventas

```python
num_vars = ['precio_unitario', 'cantidad', 'margen', 'descuento']
sns.pairplot(ventas[num_vars], height=2.5, diag_kind='hist')
plt.suptitle('Matriz de Dispersion: Variables Numericas', y=1.02)
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*15. pairplot: todas las variables numericas de ventas.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicacion**: `pairplot` crea una matriz de graficos: scatterplots (fuera de la diagonal) e histogramas (diagonal). Con 4 variables genera 4x4 = 16 paneles. `diag_kind='hist'` usa histogramas en la diagonal.

**Interpretacion**: La diagonal muestra la distribucion individual de cada variable. `precio_unitario` y `margen` tienen alta correlacion (grafico superior derecha). `cantidad` tiene baja correlacion con las demas. `descuento` tiene valores discretos (0, 0.05, 0.1, 0.15, 0.2). La matriz da una vision rapida de las relaciones.

---

### 16. pairplot: con hue=categoria para ver separacion

```python
sns.pairplot(ventas, vars=['precio_unitario', 'cantidad', 'margen'],
             hue='categoria', palette='tab10', height=2.5,
             diag_kind='kde', plot_kws={'alpha': 0.5, 's': 10})
plt.suptitle('Matriz de Dispersion por Categoria', y=1.02)
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*16. pairplot: con hue=categoria para ver separacion.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicacion**: `hue='categoria'` colorea cada punto segun su categoria y anade KDE en la diagonal por categoria. `vars` selecciona solo 3 columnas (3x3 = 9 paneles). `plot_kws` pasa `alpha` y `s` (tamano) a los scatterplots.

**Interpretacion**: Las categorias se separan naturalmente en el espacio precio-margen. Electronica y Muebles ocupan la region superior derecha. Papeleria y Software estan en la esquina inferior izquierda. La diagonal KDE muestra la forma de distribucion de cada variable por categoria.

---

### 17. pairplot: corner=True para solo triangular inferior

```python
corner_vars = ['precio_unitario', 'cantidad', 'margen', 'descuento']
sns.pairplot(ventas[corner_vars], corner=True, height=2.5,
             diag_kind='kde', plot_kws={'alpha': 0.4})
plt.suptitle('Pairplot Triangular Inferior (corner=True)', y=1.02)
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*17. pairplot: corner=True para solo triangular inferior.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicacion**: `corner=True` muestra solo la diagonal y la parte inferior de la matriz, eliminando paneles redundantes (la mitad superior es el espejo). Reduce el espacio ocupado y evita graficos duplicados.

**Interpretacion**: Con 4 variables, un pairplot completo tiene 16 paneles. Con `corner=True`, solo 10 paneles (4 de diagonal + 6 inferiores). La informacion es la misma pero en menos espacio. Ideal para explorar muchas variables sin distraccion.

---

### 18. heatmap: matriz de correlacion de variables numericas

```python
corr = ventas[['precio_unitario', 'cantidad', 'margen',
               'margen_pct', 'descuento', 'ingreso']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r',
            vmin=-1, vmax=1, square=True,
            linewidths=0.5, linecolor='white')
plt.title('Matriz de Correlacion - Variables Numericas')
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*18. heatmap: matriz de correlacion de variables numericas.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicacion**: `corr()` calcula la matriz de correlacion de Pearson. `annot=True` muestra los valores numericos. `fmt='.2f'` formato con 2 decimales. `cmap='RdBu_r'` mapa divergente (rojo=positivo, azul=negativo). `vmin=-1, vmax=1` escala fija. `square=True` celdas cuadradas. `linewidths=0.5` separadores blancos.

**Interpretacion**: `precio_unitario`, `margen` e `ingreso` tienen correlaciones altas (>0.8) entre si. `cantidad` tiene correlacion negativa debil con precio (-0.25). `descuento` tiene correlacion casi nula con todas. La visualizacion con colores permite identificar patrones de un vistazo.

---

### 19. heatmap: tabla pivote ventas por sucursal x mes (annot=True, fmt=".0f")

```python
pivot = ventas.pivot_table(values='ingreso', index='sucursal',
                           columns='mes', aggfunc='sum')

plt.figure(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd',
            linewidths=1, linecolor='white')
plt.title('Ingreso Total por Sucursal y Mes')
plt.xlabel('Mes')
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

*19. heatmap: tabla pivote ventas por sucursal x mes (annot=True, fmt=".0f").*

1. `pivot = ventas.pivot_table(values='ingreso', index='sucursal',` — Reorganiza los datos de formato largo a ancho.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicacion**: `pivot_table` crea una tabla con sucursales en filas y meses en columnas, llenada con suma de ingreso. `annot=True, fmt='.0f'` muestra los valores enteros. `cmap='YlOrRd'` escala de amarillo (bajo) a rojo (alto). `linewidths=1` separa las celdas.

**Interpretacion**: Sucursal Merida domina consistentemente todos los meses. Algunas sucursales tienen meses con valores atipicamente altos o bajos. Matriz CDMX muestra un crecimiento gradual del mes 1 al 6. Sucursal Cancun tiene los valores mas bajos todos los meses.

---

### 20. heatmap: matriz de correlacion con mask triangular superior y cmap="coolwarm" center=0

```python
corr = ventas[['precio_unitario', 'cantidad', 'margen',
               'margen_pct', 'descuento', 'ingreso']].corr()

mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

plt.figure(figsize=(8, 6))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, vmin=-1, vmax=1, square=True,
            linewidths=0.5, linecolor='white',
            cbar_kws={'shrink': 0.8, 'label': 'Pearson r'})
plt.title('Matriz de Correlacion (triangular inferior)')
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*20. heatmap: matriz de correlacion con mask triangular superior y cmap="coolwarm" center=0.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Explicacion**: `np.triu(np.ones_like(corr, dtype=bool), k=1)` crea una mascara booleana que oculta la triangular superior (incluyendo la diagonal con `k=1`). `center=0` centra el colormap en 0 (valores positivos en rojo, negativos en azul). `cbar_kws` personaliza la barra de color con `shrink` (tamano) y `label`.

**Interpretacion**: La mascara evita informacion redundante (la matriz es simetrica). Con `center=0`, los colores divergen desde el blanco (correlacion 0) hacia rojo (positiva) y azul (negativa). La correlacion mas fuerte es entre precio-margen-ingreso. No hay correlaciones negativas fuertes en estos datos.

---

## Resumen

Este modulo cubrio graficos relacionales y matriciales en Seaborn para analisis de ventas:

| Funcion | Proposito | Ejemplo tipico |
|---------|-----------|----------------|
| `sns.scatterplot` | Relacion 2 variables continuas | Precio vs cantidad |
| `sns.lineplot` | Tendencia temporal con IC | Ingreso diario |
| `sns.relplot` | Facetado de scatter/line | Precio-cantidad por categoria |
| `sns.lmplot` / `sns.regplot` | Regresion lineal/polinomica | Modelar precio vs cantidad |
| `sns.residplot` | Diagnosticos de regresion | Verificar linealidad |
| `sns.jointplot` | Distribucion bivariada + marginal | Correlacion precio-margen |
| `sns.pairplot` | Matriz de dispersion multiple | Explorar todas las variables |
| `sns.heatmap` | Mapa de calor para matrices | Correlaciones, tablas pivote |

Conceptos clave:
- Los scatterplots revelan **correlaciones** y **outliers**
- Los lineplots muestran **tendencias temporales** con incertidumbre
- Los modelos de regresion (`lmplot`, `regplot`) cuantifican **relaciones funcionales**
- Los residuales (`residplot`) evaluan la **calidad del ajuste**
- Las matrices (`pairplot`, `heatmap`) dan una **vision panoramica** de multiples variables
- `jointplot` combina **distribucion bivariada y marginal** en un unico grafico

---

## Ejercicios

1. Crea un `sns.scatterplot` de `margen` vs `cantidad` con `hue='categoria'` y `style='mes'`. ?Que patrones observas entre margen y cantidad?

2. Usa `sns.lineplot` para mostrar el `margen` promedio diario (agrupa por fecha). Anade `ci=95` y describe la tendencia semanal.

3. Genera un `sns.lmplot` con `hue='mes'` para la relacion `precio_unitario` vs `margen`. ?La pendiente cambia entre meses?

4. Crea un `sns.regplot` con `order=3` para `precio_unitario` vs `margen` y comparalo visualmente con `order=1`. ?Cual se ajusta mejor en los extremos?

5. Usa `sns.jointplot` con `kind='hex'` para `stock_actual` vs `valor_inventario` del DataFrame `inventario`. ?Que patron de stock-valor predomina?

6. Genera un `sns.pairplot` de `ventas[['precio_unitario', 'cantidad', 'margen', 'margen_pct']]` con `diag_kind='kde'` y `corner=True`. Identifica que par de variables tiene la correlacion mas alta.

7. Crea un `sns.heatmap` con `pivot_table` de `ventas` con `index='categoria'`, `columns='mes'`, `values='margen'`, `aggfunc='mean'`. Anade `annot=True, fmt='.0f', cmap='Greens'`. ?Que categoria tiene el margen promedio mas alto cada mes?

8. Usa `sns.relplot` con `kind='line'` para mostrar el `ingreso` promedio por `dia_semana` facetado por `sucursal` (5 columnas con `col_wrap=5`). Describe las diferencias entre sucursales en el patron semanal.
