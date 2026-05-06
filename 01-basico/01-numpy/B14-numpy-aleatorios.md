# B14 – Números Aleatorios en NumPy

## Introducción

La generación de números aleatorios es fundamental para simular escenarios de ventas, demanda de productos, tiempos de entrega y riesgos de inventario. NumPy ofrece una amplia gama de distribuciones de probabilidad que permiten modelar fenómenos reales del negocio: demanda Poisson (llegada de clientes), tiempos exponenciales (entre compras), precios normales (variación de costos), etc.

---

## 1. `np.random.rand` — Distribución uniforme [0,1) (descuentos aleatorios)

```python
import numpy as np

# Descuentos aleatorios para 6 productos
descuentos = np.random.rand(6)
print("Descuentos:", descuentos)
print("En %:", (descuentos * 100).astype(int))
```

**Salida esperada:**
```
Descuentos: [0.456 0.789 0.123 0.987 0.345 0.654]
En %: [45 78 12 98 34 65]
```

**Explicación:**
- `np.random.rand(6)`: 6 valores aleatorios uniformes en [0, 1).
- Cada valor representa un porcentaje de descuento para un producto.

---

## 2. `np.random.randn` — Distribución normal estándar (variación de precios)

```python
import numpy as np

# Variación aleatoria de precios (media 0, std 1)
variacion = np.random.randn(5)
print("Variación de precios:", variacion)
```

**Salida esperada:**
```
Variación de precios: [ 0.456 -1.234  0.789 -0.345  1.567]
```

**Explicación:**
- `np.random.randn(5)`: 5 valores de una distribución normal con media 0 y desviación 1.
- Útil para modelar fluctuaciones aleatorias de precios alrededor de un valor base.

---

## 3. `np.random.randint` — Enteros aleatorios (ID de productos)

```python
import numpy as np

# IDs de productos vendidos en el día (entre 1001 y 1010)
ids_vendidos = np.random.randint(1001, 1011, size=8)
print("IDs vendidos:", ids_vendidos)
```

**Salida esperada:**
```
IDs vendidos: [1005 1002 1009 1001 1007 1003 1010 1004]
```

**Explicación:**
- `np.random.randint(1001, 1011, size=8)`: 8 enteros aleatorios entre 1001 (inclusive) y 1011 (exclusive).
- Simula la venta de 8 productos con IDs en ese rango.

---

## 4. `np.random.uniform` — Uniforme en un rango (precios aleatorios)

```python
import numpy as np

# Precios aleatorios entre 50 y 500 para 6 productos
precios_simulados = np.random.uniform(50, 500, size=6)
print("Precios simulados:", np.round(precios_simulados, 2))
```

**Salida esperada:**
```
Precios simulados: [123.45 456.78 89.12 345.67 234.56 78.90]
```

**Explicación:**
- `np.random.uniform(50, 500, size=6)`: 6 valores uniformes entre 50 y 500.
- Todos los valores en el rango tienen la misma probabilidad.

---

## 5. `np.random.normal` — Normal con media y std específicas (distribución de demanda)

```python
import numpy as np

# Demanda diaria promedio 200 unidades, desviación 30
demanda = np.random.normal(loc=200, scale=30, size=10)
print("Demanda simulada 10 días:", np.round(demanda, 0).astype(int))
```

**Salida esperada:**
```
Demanda simulada 10 días: [195 220 178 210 185 230 165 205 240 190]
```

**Explicación:**
- `np.random.normal(loc=200, scale=30, size=10)`: 10 valores de una normal con media 200 y desviación 30.
- Modela la demanda diaria alrededor de un promedio con variabilidad natural.

---

## 6. `np.random.choice` — Muestra aleatoria de un conjunto (seleccionar productos)

```python
import numpy as np

productos = np.array(["Leche", "Pan", "Huevos", "Arroz", "Frijoles", "Azúcar"])
# Simular compra de 4 productos (con reposición)
compra = np.random.choice(productos, size=4, replace=True)
print("Compra del cliente:", compra)
```

**Salida esperada:**
```
Compra del cliente: ['Arroz' 'Leche' 'Pan' 'Leche']
```

**Explicación:**
- `np.random.choice(productos, size=4, replace=True)`: selecciona 4 productos aleatorios del conjunto.
- `replace=True` permite que un producto sea elegido más de una vez.

---

## 7. `np.random.shuffle` — Mezclar array in-place (orden de atención)

```python
import numpy as np

clientes = np.array(["Cliente1", "Cliente2", "Cliente3", "Cliente4", "Cliente5"])
np.random.shuffle(clientes)
print("Orden de atención:", clientes)
```

**Salida esperada:**
```
Orden de atención: ['Cliente3' 'Cliente5' 'Cliente1' 'Cliente4' 'Cliente2']
```

**Explicación:**
- `np.random.shuffle(clientes)`: mezcla aleatoriamente el array **in-place** (modifica el original).
- Simula el orden aleatorio en que llegan los clientes.

---

## 8. `np.random.permutation` — Permutación aleatoria (nuevo orden sin modificar original)

```python
import numpy as np

precios = np.array([150, 230, 85, 420, 67])
orden_aleatorio = np.random.permutation(precios)
print("Original:", precios)
print("Permutado:", orden_aleatorio)
```

**Salida esperada:**
```
Original: [150 230  85 420  67]
Permutado: [ 67 230 150  85 420]
```

**Explicación:**
- `np.random.permutation(precios)`: devuelve una **copia** permutada aleatoriamente.
- No modifica el array original.

---

## 9. `np.random.seed` — Fijar semilla para reproducibilidad

```python
import numpy as np

np.random.seed(42)
print("Primera llamada:", np.random.rand(3))
np.random.seed(42)
print("Segunda llamada (misma seed):", np.random.rand(3))
```

**Salida esperada:**
```
Primera llamada: [0.37454012 0.95071431 0.73199394]
Segunda llamada (misma seed): [0.37454012 0.95071431 0.73199394]
```

**Explicación:**
- `np.random.seed(42)`: inicializa el generador con una semilla fija.
- Con la misma semilla, los números aleatorios generados son idénticos.
- Esencial para reproducibilidad en simulaciones y pruebas.

---

## 10. `np.random.poisson` — Distribución Poisson (demanda de productos)

```python
import numpy as np

# Demanda de un producto: promedio 5 unidades por hora
demanda_hora = np.random.poisson(lam=5, size=12)
print("Demanda por hora (12h):", demanda_hora)
print("Total del día:", demanda_hora.sum())
```

**Salida esperada:**
```
Demanda por hora (12h): [4 7 3 5 6 8 2 5 4 6 7 5]
Total del día: 62
```

**Explicación:**
- `np.random.poisson(lam=5, size=12)`: genera 12 valores de una Poisson con media 5.
- Modela llegada de clientes o demanda por hora (eventos independientes en el tiempo).

---

## 11. `np.random.exponential` — Exponencial (tiempo entre compras)

```python
import numpy as np

# Tiempo entre llegadas de clientes (media 10 minutos)
tiempos = np.random.exponential(scale=10, size=8)
print("Tiempos entre compras (min):", np.round(tiempos, 2))
```

**Salida esperada:**
```
Tiempos entre compras (min): [12.34  5.67 23.45  8.90 15.67  3.45 18.90  7.89]
```

**Explicación:**
- `np.random.exponential(scale=10, size=8)`: genera 8 tiempos exponenciales con media 10.
- Modela el tiempo entre llegadas de clientes (proceso Poisson).

---

## 12. `np.random.binomial` — Binomial (probabilidad de compra)

```python
import numpy as np

# 100 clientes, cada uno con 30% de probabilidad de comprar
compras = np.random.binomial(n=100, p=0.3, size=10)
print("Clientes que compran (10 simulaciones):", compras)
```

**Salida esperada:**
```
Clientes que compran (10 simulaciones): [28 32 25 31 29 27 33 26 30 28]
```

**Explicación:**
- `np.random.binomial(n=100, p=0.3, size=10)`: simula 10 días, cada día 100 clientes, 30% probabilidad de compra.
- Modela el número de compras exitosas entre n intentos independientes.

---

## 13. `np.random.beta` — Beta (incertidumbre sobre tasa de conversión)

```python
import numpy as np

# Tasa de conversión incierta para 5 productos (prior alpha=2, beta=8)
tasas_conversion = np.random.beta(a=2, b=8, size=5)
print("Tasas de conversión simuladas:", np.round(tasas_conversion, 3))
```

**Salida esperada:**
```
Tasas de conversión simuladas: [0.234 0.189 0.312 0.156 0.267]
```

**Explicación:**
- `np.random.beta(a=2, b=8)`: genera tasas de conversión entre 0 y 1.
- La distribución Beta es ideal para modelar probabilidades inciertas.
- `a` y `b` representan éxitos y fracasos previos (información a priori).

---

## 14. `np.random.gamma` — Gamma (tiempo de entrega de proveedores)

```python
import numpy as np

# Tiempo de entrega: forma k=2, escala theta=3 (media 6 días)
tiempo_entrega = np.random.gamma(shape=2, scale=3, size=8)
print("Tiempos de entrega (días):", np.round(tiempo_entrega, 1))
```

**Salida esperada:**
```
Tiempos de entrega (días): [5.2 7.8 3.4 9.1 6.5 4.3 8.9 5.7]
```

**Explicación:**
- `np.random.gamma(shape=2, scale=3)`: genera tiempos de entrega.
- La Gamma generaliza la exponencial y modela tiempos de espera para múltiples eventos.
- Media = shape × scale = 2 × 3 = 6 días.

---

## 15. `np.random.triangular` — Triangular (estimación pesimista, probable, optimista de ventas)

```python
import numpy as np

# Venta de un producto nuevo: mín 50, más probable 120, máx 200
ventas_nuevo = np.random.triangular(left=50, mode=120, right=200, size=10)
print("Ventas simuladas producto nuevo:", np.round(ventas_nuevo, 0).astype(int))
```

**Salida esperada:**
```
Ventas simuladas producto nuevo: [115 145  98 132 160 110 128 150 105 140]
```

**Explicación:**
- `np.random.triangular(left=50, mode=120, right=200)`: distribución triangular.
- Útil cuando se tiene una estimación de tres puntos: mínimo, más probable y máximo.
- Ideal para simulaciones de Monte Carlo rápidas sin datos históricos.

---

## Resumen

| Función | Distribución | Aplicación en Ventas |
|---------|-------------|----------------------|
| `rand` | Uniforme [0,1) | Descuentos aleatorios |
| `randn` | Normal estándar | Variación de precios |
| `randint` | Enteros uniformes | IDs de productos |
| `uniform` | Uniforme [a,b) | Precios aleatorios |
| `normal` | Normal (μ,σ) | Distribución de demanda |
| `choice` | Muestra de conjunto | Selección de productos |
| `shuffle` | Permutación in-place | Orden de atención |
| `permutation` | Permutación copia | Reordenar datos |
| `seed` | Reproducibilidad | Resultados consistentes |
| `poisson` | Poisson (λ) | Demanda por hora |
| `exponential` | Exponencial (1/λ) | Tiempo entre compras |
| `binomial` | Binomial (n,p) | Clientes que compran |
| `beta` | Beta (α,β) | Tasa de conversión |
| `gamma` | Gamma (k,θ) | Tiempo de entrega |
| `triangular` | Triangular (a,m,b) | Estimación de ventas |

---

## Ejercicios

1. Simula 10 valores de demanda diaria con `np.random.normal` con media 300 y desviación 40. Redondea a enteros.

2. Genera 6 precios aleatorios entre 50 y 500 usando `np.random.uniform` y redondéalos a 2 decimales.

3. Usa `np.random.choice` para simular la compra de 5 productos del conjunto `["A","B","C","D","E","F","G"]` con reposición.

4. Simula la demanda por hora de un producto usando `np.random.poisson` con lam=8 para 10 horas. Calcula el total.

5. Genera 5 tiempos entre compras con `np.random.exponential` con scale=15 (minutos) y redondea a 2 decimales.

6. Fija la semilla en 123, genera 4 números aleatorios uniformes. Vuelve a fijar la semilla en 123 y genera otros 4. Verifica que son idénticos.

7. Simula 8 escenarios de ventas para un producto nuevo usando `np.random.triangular` con mínimo 30, más probable 80, máximo 150.

8. Usa `np.random.binomial` para simular 12 días, cada día 200 clientes, 25% de probabilidad de compra. ¿Cuántos clientes compran cada día?
