# I13 — Interpolación con SciPy aplicado a Ventas/Compras/Inventarios

## 1. Introducción Teórica

La interpolación estima valores desconocidos entre puntos conocidos. En datos de ventas, compras e inventarios, es útil para:

- **Rellenar datos faltantes**: días sin registro de ventas.
- **Suavizar tendencias**: convertir datos discretos en curvas continuas.
- **Derivar tasas de cambio**: velocidad de cambio en demanda diaria.
- **Integrar acumulados**: ventas totales en un período continuo.
- **Superficies 2D**: demanda como función de precio y descuento.

### Funciones principales de `scipy.interpolate`

| Función | Descripción | Uso típico |
|---|---|---|
| `interp1d` | Interpolación 1D (linear, cubic, nearest, etc.) | Rellenar serie temporal |
| `UnivariateSpline` | Spline suave con parámetro de suavizado s | Curvas suaves con control de rugosidad |
| `splev`, `splrep`, `splder`, `splint` | Evaluación, derivada e integral de splines | Análisis de splines B |
| `interp2d` | Interpolación 2D | Superficie precio×descuento |
| `griddata` | Datos dispersos a grilla regular | Mapa de calor de ventas |
| `RBFInterpolator` | Interpolación radial | Datos muy irregulares |
| `RegularGridInterpolator` | Grilla regular rápida | Series temporales 2D |
| `LinearNDInterpolator` | Interpolación lineal ND | Datos multidimensionales |
| `CloughTocher2DInterpolator` | Interpolación cúbica 2D | Superficies suaves |
| `PchipInterpolator` | Monótona (sin overshoot) | Precios siempre crecientes |
| `Akima1DInterpolator` | Suave sin overshoot | Demanda con picos |

---

## 2. Ejemplos Prácticos

### Ejemplo 1: interp1d linear — Interpolar demanda faltante entre días conocidos

```python
import pandas as pd
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

ventas = pd.read_csv("../../datos/ventas.csv")
inventario = pd.read_csv("../../datos/inventario.csv")

# Simular demanda diaria para 10 días con 3 faltantes
dias = np.array([1, 2, 3, 5, 6, 8, 9, 10])
demanda = np.array([120, 135, 110, 150, 145, 170, 160, 155])

f_linear = interpolate.interp1d(dias, demanda, kind="linear")
dias_completos = np.arange(1, 11)
demanda_completa = f_linear(dias_completos)

print("Días completos con interpolación linear:")
for d, q in zip(dias_completos, demanda_completa):
    print(f"  Día {d}: {q:.1f} unidades")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: interp1d linear — Interpolar demanda faltante entre días conocidos.*

1. Simular demanda diaria para 10 días con 3 faltantes

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: interp1d cubic — Interpolación suave de tendencia de ventas

```python
# Ventas semanales con 4 semanas conocidas
semanas = np.array([1, 3, 5, 8])
ventas_sem = np.array([1200, 1350, 1100, 1500])

f_cubic = interpolate.interp1d(semanas, ventas_sem, kind="cubic")
semanas_detalle = np.linspace(1, 8, 50)
ventas_detalle = f_cubic(semanas_detalle)

print(f"Ventas estimadas semana 4: {f_cubic(4):.1f}")
print(f"Ventas estimadas semana 6: {f_cubic(6):.1f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: interp1d cubic — Interpolación suave de tendencia de ventas.*

1. Ventas semanales con 4 semanas conocidas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: interp1d nearest — Rellenar con último valor conocido

```python
f_nearest = interpolate.interp1d(dias, demanda, kind="nearest", fill_value="extrapolate")
demanda_nearest = f_nearest(np.arange(1, 13))

print("Interpolación nearest (último valor conocido):")
for d, q in zip(np.arange(1, 13), demanda_nearest):
    print(f"  Día {d}: {q:.0f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: interp1d nearest — Rellenar con último valor conocido.*

1. `print("Interpolación nearest (último valor conocido):")` — Muestra el resultado por pantalla.
2. `print(f"  Día {d}: {q:.0f}")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: interp1d con fill_value=0 y bounds_error=False

```python
f_fill = interpolate.interp1d(dias, demanda, kind="linear",
                              fill_value=0, bounds_error=False)
dias_ext = np.arange(0, 15)
demanda_fill = f_fill(dias_ext)

df_fill = pd.DataFrame({"día": dias_ext, "demanda": demanda_fill})
print("Fuera del rango → 0:")
print(df_fill.to_string(index=False))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: interp1d con fill_value=0 y bounds_error=False.*

1. `print("Fuera del rango → 0:")` — Muestra el resultado por pantalla.
2. `print(df_fill.to_string(index=False))` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: UnivariateSpline — Spline suave de ventas diarias (parámetro s)

```python
from scipy.interpolate import UnivariateSpline

# Datos ruidosos de 30 días
np.random.seed(42)
dias_30 = np.arange(1, 31)
ventas_30 = 100 + 5 * dias_30 + 30 * np.sin(dias_30 / 3) + np.random.normal(0, 15, 30)

spline_s = UnivariateSpline(dias_30, ventas_30, s=200)
print(f"Suavizado s=200: ventas día 15 = {spline_s(15):.1f}")
print(f"Coeficientes del spline: {spline_s.get_coeffs().shape}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: UnivariateSpline — Spline suave de ventas diarias (parámetro s).*

1. Datos ruidosos de 30 días

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: UnivariateSpline con s=0 (interpola exactamente)

```python
spline_exact = UnivariateSpline(dias_30, ventas_30, s=0)
print(f"Interpolación exacta s=0: ventas día 15 = {spline_exact(15):.1f}")
print(f"Diferencia con dato real: {spline_exact(15) - ventas_30[14]:.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: UnivariateSpline con s=0 (interpola exactamente).*

1. `print(f"Interpolación exacta s=0: ventas día 15 = {spline_exact(15):.1f}")` — Muestra el resultado por pantalla.
2. `print(f"Diferencia con dato real: {spline_exact(15) - ventas_30[14]:.2f}")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: splev — Evaluar spline en nuevos puntos

```python
from scipy.interpolate import splev

# Crear spline con splrep
tck = interpolate.splrep(dias_30, ventas_30, s=100)
dias_nuevos = np.linspace(1, 30, 100)
ventas_suaves = interpolate.splev(dias_nuevos, tck)

print(f"Ventas suavizadas día 7.5: {interpolate.splev(7.5, tck):.1f}")
print(f"Ventas suavizadas día 22.3: {interpolate.splev(22.3, tck):.1f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: splev — Evaluar spline en nuevos puntos.*

1. Crear spline con splrep

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: splder — Derivada del spline (tasa de cambio de ventas)

```python
from scipy.interpolate import splder

tck = interpolate.splrep(dias_30, ventas_30, s=100)
tck_deriv = splder(tck)
tck_deriv2 = splder(tck, n=2)

print(f"Tasa de cambio día 10: {interpolate.splev(10, tck_deriv):.2f} unds/día")
print(f"Aceleración día 10: {interpolate.splev(10, tck_deriv2):.2f} unds/día²")

# Máxima tasa de cambio
tasas = interpolate.splev(dias_30, tck_deriv)
print(f"Tasa máxima: {tasas.max():.2f} el día {dias_30[tasas.argmax()]}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: splder — Derivada del spline (tasa de cambio de ventas).*

1. Máxima tasa de cambio

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: splint — Integral del spline (ventas acumuladas)

```python
from scipy.interpolate import splint

tck = interpolate.splrep(dias_30, ventas_30, s=100)
integral = interpolate.splint(1, 30, tck)
print(f"Ventas acumuladas (día 1-30): {integral:.0f} unidades")

integral_semanal = interpolate.splint(1, 7, tck)
print(f"Ventas acumuladas semana 1: {integral_semanal:.0f} unidades")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: splint — Integral del spline (ventas acumuladas).*

1. `from scipy.interpolate import splint` — Importa las librerías necesarias para el análisis.
2. `print(f"Ventas acumuladas (día 1-30): {integral:.0f} unidades")` — Muestra el resultado por pantalla.
3. `print(f"Ventas acumuladas semana 1: {integral_semanal:.0f} unidades")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: interp2d — Superficie precio×descuento → demanda

```python
from scipy.interpolate import interp2d

# Datos de demanda según precio y descuento
precios_vals = np.array([500, 1000, 1500, 2000, 2500])
descuentos_vals = np.array([0.0, 0.1, 0.2, 0.3])
P, D = np.meshgrid(precios_vals, descuentos_vals)

# Demanda simulada: menor precio + mayor descuento = más demanda
Z = 3000 - 0.8 * P + 5000 * D + np.random.normal(0, 50, P.shape)

f_2d = interp2d(precios_vals, descuentos_vals, Z, kind="cubic")

precio_test, desc_test = 1200, 0.15
demanda_est = f_2d(precio_test, desc_test)
print(f"Demanda estimada (P=${precio_test}, D={desc_test:.0%}): {demanda_est[0]:.0f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: interp2d — Superficie precio×descuento → demanda.*

1. Datos de demanda según precio y descuento
2. Demanda simulada: menor precio + mayor descuento = más demanda

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: griddata — Interpolar datos dispersos de ventas por ubicación

```python
from scipy.interpolate import griddata

# Coordenadas ficticias de sucursales
np.random.seed(42)
n_puntos = 50
x = np.random.uniform(0, 100, n_puntos)
y = np.random.uniform(0, 100, n_puntos)
ventas_puntos = 200 + 0.5 * x + 0.3 * y + np.random.normal(0, 20, n_puntos)

# Grilla regular
xi = np.linspace(0, 100, 20)
yi = np.linspace(0, 100, 20)
xi_grid, yi_grid = np.meshgrid(xi, yi)

zi_linear = griddata((x, y), ventas_puntos, (xi_grid, yi_grid), method="linear")
zi_cubic = griddata((x, y), ventas_puntos, (xi_grid, yi_grid), method="cubic")
zi_nearest = griddata((x, y), ventas_puntos, (xi_grid, yi_grid), method="nearest")

print(f"Ventas interpoladas (linear) en (50,50): {zi_linear[10, 10]:.1f}")
print(f"Ventas interpoladas (cubic) en (50,50): {zi_cubic[10, 10]:.1f}")
print(f"Ventas interpoladas (nearest) en (50,50): {zi_nearest[10, 10]:.1f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: griddata — Interpolar datos dispersos de ventas por ubicación.*

1. Coordenadas ficticias de sucursales
2. Grilla regular

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: RBFInterpolator — Interpolación radial para datos irregulares

```python
from scipy.interpolate import RBFInterpolator

# Datos muy irregulares (coordenadas aleatorias)
np.random.seed(123)
x_irr = np.random.uniform(0, 50, 30)
y_irr = np.random.uniform(0, 50, 30)
z_irr = 100 + 2 * x_irr - 1.5 * y_irr + 10 * np.sin(x_irr / 5) + np.random.normal(0, 5, 30)

rbf = RBFInterpolator(np.column_stack([x_irr, y_irr]), z_irr, kernel="thin_plate_spline")

puntos_test = np.array([[25, 25], [10, 40], [40, 10]])
z_pred = rbf(puntos_test)
for pt, zp in zip(puntos_test, z_pred):
    print(f"  Ubicación ({pt[0]:.0f}, {pt[1]:.0f}): ventas ≈ {zp:.0f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: RBFInterpolator — Interpolación radial para datos irregulares.*

1. Datos muy irregulares (coordenadas aleatorias)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: PchipInterpolator — Interpolación monótona (precios siempre crecientes)

```python
from scipy.interpolate import PchipInterpolator

# Precios de producto en diferentes semanas (siempre deben crecer)
semanas_p = np.array([1, 2, 4, 5, 7, 8])
precios_p = np.array([100, 105, 108, 110, 115, 120])

pchip = PchipInterpolator(semanas_p, precios_p)
semanas_det = np.linspace(1, 8, 30)
precios_det = pchip(semanas_det)

# Verificar monotonicidad
diff = np.diff(precios_det)
print(f"¿Es monótona (todas diferencias >= 0)? {np.all(diff >= 0)}")
print(f"Precio semana 3: ${pchip(3):.2f}")
print(f"Precio semana 6: ${pchip(6):.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: PchipInterpolator — Interpolación monótona (precios siempre crecientes).*

1. Precios de producto en diferentes semanas (siempre deben crecer)
2. Verificar monotonicidad

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: Akima1DInterpolator — Suave sin overshoot

```python
from scipy.interpolate import Akima1DInterpolator

# Demanda con picos pronunciados
dias_pico = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
demanda_pico = np.array([100, 95, 300, 310, 105, 100, 350, 340, 110, 105])

akima = Akima1DInterpolator(dias_pico, demanda_pico)
dias_det = np.linspace(1, 10, 100)
demanda_akima = akima(dias_det)

print(f"Demanda Akima día 3.5: {akima(3.5):.1f}")
print(f"Demanda Akima día 7.5: {akima(7.5):.1f}")
# Comparar con cúbico
f_cub = interpolate.interp1d(dias_pico, demanda_pico, kind="cubic")
print(f"Demanda cúbica día 3.5: {f_cub(3.5):.1f}")
print(f"Demanda cúbica día 7.5: {f_cub(7.5):.1f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: Akima1DInterpolator — Suave sin overshoot.*

1. Demanda con picos pronunciados
2. Comparar con cúbico

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: Comparar linear vs cubic vs spline en mismos datos

```python
dias_base = np.array([1, 3, 5, 7, 10, 12, 15])
ventas_base = np.array([200, 180, 220, 210, 250, 240, 280])

f_lin = interpolate.interp1d(dias_base, ventas_base, kind="linear")
f_cub = interpolate.interp1d(dias_base, ventas_base, kind="cubic")
f_spl = UnivariateSpline(dias_base, ventas_base, s=50)

dias_test = np.array([2, 4, 6, 8, 11, 13])

print(f"{'Día':>5} {'Linear':>8} {'Cubic':>8} {'Spline':>8}")
for d in dias_test:
    print(f"{d:5.0f} {f_lin(d):8.1f} {f_cub(d):8.1f} {f_spl(d):8.1f}")

# Error cuadrático medio respecto a spline (referencia)
ref = f_spl(dias_test)
print(f"\nRMSE linear: {np.sqrt(np.mean((f_lin(dias_test) - ref)**2)):.2f}")
print(f"RMSE cubic: {np.sqrt(np.mean((f_cub(dias_test) - ref)**2)):.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: Comparar linear vs cubic vs spline en mismos datos.*

1. Error cuadrático medio respecto a spline (referencia)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 16: Extrapolación con spline (ext='extrapolate')

```python
spline_ext = UnivariateSpline(dias_base, ventas_base, s=50, ext="extrapolate")

print(f"Ventas extrapoladas día 20: {spline_ext(20):.1f}")
print(f"Ventas extrapoladas día 0: {spline_ext(0):.1f}")
print(f"Ventas extrapoladas día -5: {spline_ext(-5):.1f}")

# Con interp1d
f_ext = interpolate.interp1d(dias_base, ventas_base, kind="cubic",
                             fill_value="extrapolate")
print(f"interp1d extrapolate día 20: {f_ext(20):.1f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: Extrapolación con spline (ext='extrapolate').*

1. Con interp1d

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 17: Smoothing spline — Diferentes s y su efecto en suavizado

```python
np.random.seed(7)
dias_s = np.arange(1, 51)
ventas_s = 200 + 30 * np.sin(dias_s / 5) + np.random.normal(0, 25, 50)

s_valores = [0, 50, 200, 1000, 5000]
resultados = {}

for s in s_valores:
    spl = UnivariateSpline(dias_s, ventas_s, s=s)
    resultados[s] = spl(dias_s)
    # Grado de suavizado: diferencia cuadrática media con datos
    error = np.sqrt(np.mean((spl(dias_s) - ventas_s)**2))
    print(f"s={s:5d}: RMSE={error:.2f}, #nudos={spl.get_knots().shape[0]}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: Smoothing spline — Diferentes s y su efecto en suavizado.*

1. Grado de suavizado: diferencia cuadrática media con datos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 18: Integrador — Rellenar 30 días faltantes de demanda con diferentes métodos

```python
np.random.seed(42)
# 10 días conocidos, 20 faltantes (simulamos datos reales solo 10 días)
dias_conocidos = np.sort(np.random.choice(np.arange(1, 31), 10, replace=False))
ventas_conocidas = 150 + 3 * dias_conocidos + 20 * np.sin(dias_conocidos / 2) + np.random.normal(0, 10, 10)

todos_dias = np.arange(1, 31)

# Método 1: Linear
f_lin = interpolate.interp1d(dias_conocidos, ventas_conocidas, kind="linear",
                             fill_value="extrapolate")
v_lin = f_lin(todos_dias)

# Método 2: Cubic
f_cub = interpolate.interp1d(dias_conocidos, ventas_conocidas, kind="cubic",
                             fill_value="extrapolate")
v_cub = f_cub(todos_dias)

# Método 3: Spline suave
f_spl = UnivariateSpline(dias_conocidos, ventas_conocidas, s=50, ext="extrapolate")
v_spl = f_spl(todos_dias)

# Método 4: Pchip monótono
f_pchip = PchipInterpolator(dias_conocidos, ventas_conocidas)
v_pchip = f_pchip(todos_dias)

print(f"{'Día':>4} {'Real':>7} {'Linear':>8} {'Cubic':>8} {'Spline':>8} {'Pchip':>8}")
for d in todos_dias:
    real = ventas_conocidas[dias_conocidos.tolist().index(d)] if d in dias_conocidos else np.nan
    r_str = f"{real:7.0f}" if not np.isnan(real) else "     NA"
    print(f"{d:4d} {r_str} {v_lin[d-1]:8.1f} {v_cub[d-1]:8.1f} {v_spl[d-1]:8.1f} {v_pchip[d-1]:8.1f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador — Rellenar 30 días faltantes de demanda con diferentes métodos.*

1. 10 días conocidos, 20 faltantes (simulamos datos reales solo 10 días)
2. Método 1: Linear
3. Método 2: Cubic
4. Método 3: Spline suave
5. Método 4: Pchip monótono

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 3. Resumen

| Técnica | Cuándo usarla | Limitación |
|---|---|---|
| `interp1d linear` | Datos sin ruido, relleno rápido | No suaviza, no extrapola bien |
| `interp1d cubic` | Curvas suaves entre puntos | Overshoot en cambios bruscos |
| `interp1d nearest` | Relleno conservador | Escalones, no continuo |
| `UnivariateSpline` s>0 | Datos ruidosos, control de suavizado | Elegir s requiere validación |
| `UnivariateSpline` s=0 | Interpolación exacta | Sobreajuste con ruido |
| `PchipInterpolator` | Datos monótonos (precios) | Menos suave que spline cúbico |
| `Akima1DInterpolator` | Picos sin overshoot | Puede ser menos preciso |
| `RBFInterpolator` | Datos irregulares ND | Costoso computacionalmente |
| `griddata` | Disperso → grilla regular | No extrapola |
| `interp2d` | Superficies 2D regulares | Obsoleto, preferir RegularGridInterpolator |

---

## 4. Ejercicios Propuestos

1. Usa `interp1d` con kind="quadratic" para interpolar la demanda diaria de un producto que tiene registros solo los días 1, 4, 7, 10, 13, 16, 19, 22, 25, 28 de un mes. Estima la demanda para todos los días y calcula el total mensual.

2. Con `UnivariateSpline`, ajusta una curva suave a los datos de ventas diarias de la sucursal "Mérida" (agrupando por fecha). Prueba 3 valores de s (10, 100, 1000) y muestra cómo cambia el suavizado.

3. Usa `splder` para calcular la derivada primera y segunda del spline ajustado en el ejercicio anterior. Identifica los días de máxima aceleración de ventas (puntos de inflexión).

4. Aplica `PchipInterpolator` para interpolar la evolución del precio de un producto que sube de $500 a $1200 en 12 meses. Verifica que la interpolación sea monótona creciente (sin bajadas artificiales).

5. Con `griddata`, interpola las ventas totales por sucursal (usa latitud/longitud ficticia) para crear un mapa de calor de ventas en una grilla 50×50. Compara method="linear" vs "cubic".

6. Usa `RBFInterpolator` para interpolar la demanda basada en 3 variables: precio, descuento y gasto en marketing. Genera datos sintéticos y evalúa la precisión en 10 puntos de prueba.

7. Con `interp2d`, construye la superficie de ingreso total como función de precio y cantidad de unidades ofertadas. Encuentra el punto aproximado de máximo ingreso sobre la superficie.

8. **Integrador**: Toma los datos de ventas de un producto específico durante 90 días (agrega por fecha fila), identifica qué días faltan, y rellena los faltantes con 4 métodos distintos (linear, cubic, spline, pchip). Calcula el total trimestral estimado por cada método y compáralos.
