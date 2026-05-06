# I12 — Optimización con SciPy aplicado a Ventas/Compras/Inventarios

## 1. Introducción Teórica

La optimización busca minimizar (o maximizar) una función objetivo sujeto a restricciones. En el contexto de ventas, compras e inventarios, permite resolver problemas como:

- **Precio óptimo**: maximizar ingreso dado una función de demanda-precio.
- **Lote económico de pedido (EOQ)**: minimizar costo total (ordenar + almacenar).
- **Asignación de presupuesto**: maximizar margen con restricción de gasto.
- **Ajuste de curvas**: modelar demanda real a partir de datos históricos.
- **Punto de equilibrio**: encontrar precio donde oferta = demanda.

### Funciones principales de `scipy.optimize`

| Función | Uso | Métodos comunes |
|---|---|---|
| `minimize` | Optimización multivariable con/sin restricciones | Nelder-Mead, BFGS, L-BFGS-B, TNC, SLSQP, Powell, CG |
| `minimize_scalar` | Optimización 1D (una variable) | bounded, golden, bracket |
| `linprog` | Programación lineal (restricciones lineales) | highs, interior-point |
| `curve_fit` | Ajuste de curvas por mínimos cuadrados | Levenberg-Marquardt, trust-region |
| `least_squares` | Mínimos cuadrados no lineales robustos | trf, dogbox, lm |
| `differential_evolution` | Optimización global evolutiva | — |
| `shgo` | Optimización global (homología simplicial) | — |
| `dual_annealing` | Recocido simulado generalizado | — |
| `direct` | Búsqueda directa de Lipschitz | — |
| `root`, `fsolve`, `newton`, `brentq`, `bisect` | Encontrar raíces de ecuaciones | — |

---

## 2. Ejemplos Prácticos

### Ejemplo 1: minimize_scalar — Precio que maximiza ingreso

```python
import pandas as pd
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

ventas = pd.read_csv("../../datos/ventas.csv")
inventario = pd.read_csv("../../datos/inventario.csv")

# Función de ingreso: I(p) = p * q(p) donde q(p) = a - b*p
# Ajustamos con datos reales (aproximación lineal)
precios = ventas.groupby("producto")["precio_unitario"].mean().values
cantidades = ventas.groupby("producto")["cantidad"].sum().values
coef = np.polyfit(precios, cantidades, 1)
a, b = coef[1], -coef[0]

def ingreso(p):
    q = max(0, a - b * p)
    return -p * q  # Negativo para minimizar

res = optimize.minimize_scalar(ingreso)
precio_optimo = res.x
print(f"Precio óptimo teórico: ${precio_optimo:.2f}")
print(f"Ingreso máximo: ${-res.fun:.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: minimize_scalar — Precio que maximiza ingreso.*

1. Función de ingreso: I(p) = p * q(p) donde q(p) = a - b*p
2. Ajustamos con datos reales (aproximación lineal)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: minimize_scalar bounded — Precio óptimo entre $10 y $1000

```python
# Misma función pero con rango realista
res = optimize.minimize_scalar(ingreso, bounds=(10, 1000), method="bounded")
precio_optimo = res.x
print(f"Precio óptimo (acotado $10-$1000): ${precio_optimo:.2f}")
print(f"Ingreso máximo: ${-res.fun:.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: minimize_scalar bounded — Precio óptimo entre $10 y $1000.*

1. Misma función pero con rango realista

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: minimize (Nelder-Mead) — Minimizar costo total EOQ

```python
# Costo total = (D/Q)*S + (Q/2)*H
# D=demanda anual, S=costo por pedido, H=costo almacenamiento unitario
D = ventas["cantidad"].sum()  # Demanda total
S = 1500  # Costo por pedido ($)
H = 200   # Costo almacenamiento unitario ($)

def costo_total(Q):
    return (D / Q) * S + (Q / 2) * H

res = optimize.minimize(costo_total, x0=[500], method="Nelder-Mead")
Q_opt = res.x[0]
print(f"Lote óptimo (EOQ): {Q_opt:.0f} unidades")
print(f"Costo total mínimo: ${res.fun:.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: minimize (Nelder-Mead) — Minimizar costo total EOQ.*

1. Costo total = (D/Q)*S + (Q/2)*H
2. D=demanda anual, S=costo por pedido, H=costo almacenamiento unitario

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: minimize (L-BFGS-B) — Con bounds en variables

```python
# Variables: [cantidad_pedido, precio_venta]
# Restricciones: cantidad entre 100 y 5000, precio entre 50 y 2000
def beneficio_neg(x):
    Q, p = x
    costo_unitario = 300
    demanda = max(0, 10000 - 2 * p)
    if Q <= 0 or demanda <= 0:
        return 1e12
    return -(p * min(Q, demanda) - costo_unitario * min(Q, demanda) - 1500 * (demanda / Q) - 200 * (Q / 2))

res = optimize.minimize(beneficio_neg, x0=[1000, 500],
                        method="L-BFGS-B",
                        bounds=[(100, 5000), (50, 2000)])
print(f"Lote: {res.x[0]:.0f}, Precio: ${res.x[1]:.2f}")
print(f"Beneficio máximo: ${-res.fun:.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: minimize (L-BFGS-B) — Con bounds en variables.*

1. Variables: [cantidad_pedido, precio_venta]
2. Restricciones: cantidad entre 100 y 5000, precio entre 50 y 2000

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: minimize (SLSQP) — Con restricciones de desigualdad

```python
# Minimizar costo sujeto a stock >= stock_mínimo
stock_min = inventario.set_index("sku")["stock_minimo"].to_dict()

def costo_compra(x):
    # x: array de cantidades a comprar por SKU
    skus = list(stock_min.keys())[:5]
    costos = inventario.set_index("sku")["costo"].to_dict()
    return sum(x[i] * costos[skus[i]] for i in range(len(skus)))

def restriccion_stock(x):
    skus = list(stock_min.keys())[:5]
    actual = inventario.set_index("sku")["stock_actual"].to_dict()
    return [actual[skus[i]] + x[i] - stock_min[skus[i]] for i in range(len(skus))]

n = min(5, len(stock_min))
x0 = np.ones(n) * 10
restricciones = [{"type": "ineq", "fun": lambda x, i=i: restriccion_stock(x)[i]} for i in range(n)]
res = optimize.minimize(costo_compra, x0, method="SLSQP", constraints=restricciones, bounds=[(0, 200)] * n)
print(f"Cantidades óptimas a comprar: {res.x.round(0)}")
print(f"Costo mínimo: ${res.fun:.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: minimize (SLSQP) — Con restricciones de desigualdad.*

1. Minimizar costo sujeto a stock >= stock_mínimo
2. x: array de cantidades a comprar por SKU

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: linprog — Minimizar costo de compra con restricciones de proveedores

```python
# 3 proveedores, cada uno ofrece precio distinto por SKU
# Minimizar: 12*x1 + 10*x2 + 15*x3
# Sujeto a: x1 + x2 + x3 >= 500 (demanda mínima)
#           x1 <= 200 (capacidad proveedor 1), x2 <= 300, x3 <= 250
c = [12, 10, 15]
A_ub = [[1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]]
b_ub = [200, 300, 250]
A_eq = [[-1, -1, -1]]
b_eq = [-500]

res = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method="highs")
print(f"Cantidades a comprar: {res.x}")
print(f"Costo mínimo: ${res.fun:.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: linprog — Minimizar costo de compra con restricciones de proveedores.*

1. 3 proveedores, cada uno ofrece precio distinto por SKU
2. Minimizar: 12*x1 + 10*x2 + 15*x3
3. Sujeto a: x1 + x2 + x3 >= 500 (demanda mínima)
4. x1 <= 200 (capacidad proveedor 1), x2 <= 300, x3 <= 250

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: curve_fit — Ajustar modelo de demanda (precio → cantidad)

```python
from scipy.optimize import curve_fit

# Modelo: q(p) = a * p^b  (elasticidad constante)
def demanda(p, a, b):
    return a * p ** b

precios_data = ventas["precio_unitario"].values
cantidades_data = ventas["cantidad"].values

popt, pcov = curve_fit(demanda, precios_data, cantidades_data, p0=[10000, -1])
a_opt, b_opt = popt
print(f"Parámetros: a={a_opt:.2f}, b={b_opt:.2f}")
print(f"Elasticidad precio-demanda: {b_opt:.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: curve_fit — Ajustar modelo de demanda (precio → cantidad).*

1. Modelo: q(p) = a * p^b  (elasticidad constante)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: curve_fit con sigma — Pesos para heterocedasticidad

```python
# Productos con mayor precio tienen mayor varianza en cantidad
precios_data = ventas["precio_unitario"].values
cantidades_data = ventas["cantidad"].values

# Agrupar por precio para estimar desviación estándar
df_agrup = ventas.groupby("precio_unitario")["cantidad"].agg(["mean", "std"]).reset_index()
p_vals = df_agrup["precio_unitario"].values
q_vals = df_agrup["mean"].values
q_std = df_agrup["std"].values.fillna(1)

popt, pcov = curve_fit(demanda, p_vals, q_vals, sigma=q_std, absolute_sigma=True)
print(f"Con pesos (heterocedasticidad): a={popt[0]:.2f}, b={popt[1]:.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: curve_fit con sigma — Pesos para heterocedasticidad.*

1. Productos con mayor precio tienen mayor varianza en cantidad
2. Agrupar por precio para estimar desviación estándar

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: least_squares — Ajuste robusto de curva precio-demanda

```python
from scipy.optimize import least_squares

def residuals(params, p, q):
    a, b = params
    return q - (a * p ** b)

precios_data = ventas["precio_unitario"].values
cantidades_data = ventas["cantidad"].values

res = least_squares(residuals, x0=[10000, -1], args=(precios_data, cantidades_data), loss="soft_l1")
print(f"Robusto: a={res.x[0]:.2f}, b={res.x[1]:.2f}")
print(f"Costo final: {res.cost:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: least_squares — Ajuste robusto de curva precio-demanda.*

1. `from scipy.optimize import least_squares` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: differential_evolution — Optimización global multimodal

```python
from scipy.optimize import differential_evolution

# Función costo multimodal: elegir descuento que maximiza margen
def margen_neg(descuento):
    precio_base = 1500
    costo = 800
    precio_final = precio_base * (1 - descuento)
    demanda_est = 500 + 3000 * descuento - 5000 * descuento ** 2
    if demanda_est <= 0 or precio_final <= 0:
        return 0
    return -(precio_final - costo) * demanda_est

res = differential_evolution(margen_neg, bounds=[(0, 0.5)], seed=42)
print(f"Descuento óptimo: {res.x[0]:.2%}")
print(f"Margen máximo: ${-res.fun:.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: differential_evolution — Optimización global multimodal.*

1. Función costo multimodal: elegir descuento que maximiza margen

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: shgo — Optimización global con topología

```python
from scipy.optimize import shgo

# Función de beneficio con múltiples picos locales
def beneficio(x):
    p, q = x
    if p <= 0 or q <= 0:
        return 1e6
    costo = 200
    ingreso = p * q
    costo_total = costo * q + 5000 + 0.01 * q ** 2
    return -(ingreso - costo_total)

res = shgo(beneficio, bounds=[(10, 1000), (10, 5000)], n=30, iters=5)
print(f"SHGO - Precio: ${res.x[0]:.2f}, Cantidad: {res.x[1]:.0f}")
print(f"Beneficio máximo: ${-res.fun:.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: shgo — Optimización global con topología.*

1. Función de beneficio con múltiples picos locales

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: dual_annealing — Recocido simulado para inventario

```python
from scipy.optimize import dual_annealing

# Decidir punto de reorden y cantidad a pedir
def costo_inventario(x):
    reorder, cantidad = x
    if reorder <= 0 or cantidad <= 0:
        return 1e6
    demanda_diaria = 50
    costo_pedido = 2000
    costo_almacen = 5
    dias_anuales = 365
    num_pedidos = demanda_diaria * dias_anuales / cantidad
    stock_prom = cantidad / 2 + reorder
    return num_pedidos * costo_pedido + stock_prom * costo_almacen

res = dual_annealing(costo_inventario, bounds=[(10, 500), (100, 3000)], seed=42)
print(f"Punto de reorden: {res.x[0]:.0f}, Cantidad: {res.x[1]:.0f}")
print(f"Costo anual: ${res.fun:.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: dual_annealing — Recocido simulado para inventario.*

1. Decidir punto de reorden y cantidad a pedir

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: fsolve — Punto de equilibrio oferta-demanda

```python
from scipy.optimize import fsolve

# Oferta: q_s(p) = 50*p - 2000
# Demanda: q_d(p) = 10000 - 30*p
def equilibrio(p):
    qs = 50 * p - 2000
    qd = 10000 - 30 * p
    return qs - qd

p_eq = fsolve(equilibrio, x0=100)[0]
q_eq = 50 * p_eq - 2000
print(f"Precio equilibrio: ${p_eq:.2f}")
print(f"Cantidad equilibrio: {q_eq:.0f} unidades")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: fsolve — Punto de equilibrio oferta-demanda.*

1. Oferta: q_s(p) = 50*p - 2000
2. Demanda: q_d(p) = 10000 - 30*p

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: newton — Raíz de función de beneficio marginal

```python
from scipy.optimize import newton

# Ingreso marginal - Costo marginal = 0
# I(q) = 5000*ln(q+1), C(q) = 200*q + 1000
def beneficio_marginal(q):
    return 5000 / (q + 1) - 200

q_opt = newton(beneficio_marginal, x0=10)
print(f"Cantidad óptima (beneficio marginal = 0): {q_opt:.0f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: newton — Raíz de función de beneficio marginal.*

1. Ingreso marginal - Costo marginal = 0
2. I(q) = 5000*ln(q+1), C(q) = 200*q + 1000

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: brentq — Raíz de función de costos en intervalo [a,b]

```python
from scipy.optimize import brentq

# Encontrar q donde costo_total = ingreso_total
def costo_total(q):
    return 200 * q + 5000 + 0.5 * q ** 2

def ingreso_total(q):
    return 1500 * q ** 0.8

def punto_equilibrio(q):
    return costo_total(q) - ingreso_total(q)

q_break = brentq(punto_equilibrio, 10, 5000)
print(f"Punto de equilibrio (q): {q_break:.0f} unidades")
print(f"Ingreso = Costo = ${ingreso_total(q_break):.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: brentq — Raíz de función de costos en intervalo [a,b].*

1. Encontrar q donde costo_total = ingreso_total

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 16: root — Sistema de ecuaciones (precio y cantidad de equilibrio)

```python
from scipy.optimize import root

# Sistema: 1) Demanda: q = 10000 - 30*p, 2) Oferta: q = 50*p - 2000
def sistema(vars):
    p, q = vars
    return [q - (10000 - 30 * p), q - (50 * p - 2000)]

sol = root(sistema, x0=[100, 3000])
print(f"Precio equilibrio: ${sol.x[0]:.2f}")
print(f"Cantidad equilibrio: {sol.x[1]:.0f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: root — Sistema de ecuaciones (precio y cantidad de equilibrio).*

1. Sistema: 1) Demanda: q = 10000 - 30*p, 2) Oferta: q = 50*p - 2000

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 17: Comparar métodos — Precisión vs velocidad

```python
import time

def funcion_prueba(x):
    return (x[0] - 150) ** 2 + (x[1] - 3000) ** 2 + 100 * np.sin(x[0]) * np.cos(x[1])

metodos = ["Nelder-Mead", "BFGS", "L-BFGS-B", "Powell", "CG", "TNC"]
resultados = []

for met in metodos:
    t0 = time.time()
    res = optimize.minimize(funcion_prueba, x0=[0, 0], method=met)
    t = time.time() - t0
    resultados.append((met, res.x, res.fun, res.nfev, t))

df_res = pd.DataFrame(resultados, columns=["Método", "Solución", "f(x)", "Evaluaciones", "Tiempo (s)"])
print(df_res.to_string(index=False))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: Comparar métodos — Precisión vs velocidad.*

1. `import time` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 18: Integrador — Optimizar EOQ con demanda estacional

```python
# EOQ con demanda variable mensual y descuento por volumen
demanda_mensual = ventas.groupby("mes")["cantidad"].sum().values
costo_pedido = 1200
costo_almacen = 150
costo_unitario = 500

# Descuento: si Q > 2000, costo_unitario baja 10%
def costo_total_anual(x):
    Q = x[0]
    cu = costo_unitario * (0.9 if Q > 2000 else 1.0)
    D = demanda_mensual.sum()
    return (D / Q) * costo_pedido + (Q / 2) * costo_almacen + D * cu

res = optimize.minimize(costo_total_anual, x0=[1000], method="L-BFGS-B", bounds=[(50, 5000)])
Q_final = res.x[0]
cu_final = costo_unitario * (0.9 if Q_final > 2000 else 1.0)
print(f"EOQ óptimo: {Q_final:.0f} unidades")
print(f"Costo unitario con descuento: ${cu_final:.2f}")
print(f"Costo total anual: ${res.fun:.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador — Optimizar EOQ con demanda estacional.*

1. EOQ con demanda variable mensual y descuento por volumen
2. Descuento: si Q > 2000, costo_unitario baja 10%

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 3. Resumen

| Función | Aplicación en Ventas/Compras/Inventarios |
|---|---|
| `minimize_scalar` | Precio óptimo univariable |
| `minimize` (Nelder-Mead, BFGS, SLSQP, etc.) | EOQ, beneficio con restricciones |
| `linprog` | Asignación de compras entre proveedores |
| `curve_fit` / `least_squares` | Ajuste de modelos de demanda |
| `differential_evolution` / `shgo` / `dual_annealing` | Optimización global multimodal |
| `fsolve` / `newton` / `brentq` / `bisect` | Puntos de equilibrio, raíces de beneficio marginal |

**Criterios de selección**:
- Sin derivadas → Nelder-Mead, Powell, differential_evolution
- Con derivadas → BFGS, L-BFGS-B, CG
- Restricciones lineales → SLSQP, linprog
- Restricciones no lineales → SLSQP, COBYLA, trust-constr
- Una variable → minimize_scalar, brentq, bisect

---

## 4. Ejercicios Propuestos

1. Usando `minimize_scalar`, encuentra el precio que maximiza el margen (ingreso - costo) para la categoría "Electrónica". Usa los datos de `ventas.csv` para estimar la relación precio-demanda.

2. Aplica `linprog` para minimizar el costo de compra de 5 SKUs diferentes, donde cada SKU tiene al menos 3 proveedores con precios distintos. Incluye restricciones de capacidad por proveedor.

3. Usa `curve_fit` para ajustar un modelo log-log (`log(q) = a + b*log(p)`) a los datos de ventas. Reporta la elasticidad precio de la demanda e interpreta el resultado.

4. Resuelve con `differential_evolution` el problema de encontrar el descuento óptimo (0-50%) y la inversión en marketing ($0-$10000) que maximicen el margen neto, donde la demanda responde a ambas variables.

5. Implementa con `least_squares` (loss="huber") un ajuste robusto de la relación entre descuento y cantidad vendida. Compara los coeficientes con los de `curve_fit` estándar.

6. Usa `fsolve` para encontrar el precio donde la elasticidad precio de la demanda es exactamente -1 (elasticidad unitaria) a partir de la curva ajustada en el ejercicio 1.

7. Con `brentq`, encuentra la cantidad de inventario donde el costo de almacenar iguala al costo de pedir (sin considerar demanda) en el intervalo [10, 10000].

8. **Integrador**: Diseña una función que, dado un producto específico, optimice simultáneamente precio, cantidad a pedir y descuento usando `minimize` con SLSQP. Incluye restricciones de stock mínimo y margen mínimo del 20%.
