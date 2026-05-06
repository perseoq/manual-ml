# A21: Autograd — Diferenciación Automática para Ventas, Compras e Inventarios

## Introducción Teórica

**Autograd** es el motor de diferenciación automática de PyTorch. Construye un **grafo computacional dinámico** (DAG) que registra cada operación sobre tensores con `requires_grad=True`. Cuando se llama a `.backward()`, Autograd recorre el grafo en reversa calculando gradientes mediante la regla de la cadena.

### Conceptos clave:

1. **requires_grad=True**: Activa el seguimiento de operaciones en un tensor. Los gradientes se acumulan en el atributo `.grad`.
2. **backward()**: Dispara el cómputo de gradientes desde la salida hasta las hojas del grafo. Solo válido para salidas escalares; para tensores se requiere `gradient` o `jacobian`.
3. **grad_fn**: Atributo que almacena la función que generó el tensor. Útil para inspeccionar el grafo: `x.grad_fn` → `None` si es hoja; `loss.grad_fn` → `<MseLossBackward>`.
4. **detach()**: Crea un nuevo tensor que **no** forma parte del grafo computacional. Ambos tensores comparten memoria pero el nuevo no tiene `requires_grad`. Esencial para extraer valores como NumPy.
5. **with torch.no_grad()**: Context manager que deshabilita la construcción del grafo temporalmente. Se usa en evaluación/inferencia para ahorrar memoria y acelerar cómputo.
6. **torch.autograd.Function**: Clase base para definir operaciones diferenciables personalizadas. Hay que implementar `forward(ctx, ...)` y `backward(ctx, grad_output)`. `ctx` (context) guarda tensores con `save_for_backward()`.
7. **torch.autograd.grad**: Calcula gradientes directamente sin `.backward()`. Devuelve una tupla de tensores. Útil para gradientes de salidas no escalares.
8. **retain_graph=True**: En `.backward()`, retiene el grafo después del backward. Necesario para múltiples llamadas a backward. Consume memoria.
9. **create_graph=True**: Construye un grafo de segundo orden para calcular derivadas de derivadas (Hessian, meta-learning).
10. **register_hook**: Adjunta una función que se ejecuta durante el backward. Puede inspeccionar o modificar gradientes. Recibe `grad` y debe devolver un tensor (o `None`).
11. **torch.autograd.profiler**: Herramienta para medir tiempo y memoria de operaciones. Se usa como `with torch.autograd.profiler.profile()`.
12. **set_grad_enabled(False)**: Deshabilita gradientes globalmente. Útil para scripts que intercalan entrenamiento y evaluación.

### Aplicaciones en negocio:

- **Ventas**: Calcular derivadas de pérdida MSE respecto a pesos para ajustar predicciones de demanda.
- **Compras**: Propagación de errores en modelos de costos con múltiples variables (precio, cantidad, lead time).
- **Inventarios**: Optimización de parámetros en funciones de costo logístico (stock out + holding).

---

## Ejemplos

### Ejemplo 1: requires_grad=True — Tensor de pesos que necesita gradiente

```python
import torch
import numpy as np

# Simulamos 5 features de productos: precio, costo, descuento, rating, demanda_hist
torch.manual_seed(42)
pesos = torch.randn(5, 1, requires_grad=True)
print(f"Pesos (requires_grad=True):\n{pesos}")
print(f"requires_grad: {pesos.requires_grad}")

# Cada peso tendrá un gradiente después de backward
print(f"Grad inicial: {pesos.grad}")  # None
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: requires_grad=True — Tensor de pesos que necesita gradiente.*

1. Simulamos 5 features de productos: precio, costo, descuento, rating, demanda_hist
2. Cada peso tendrá un gradiente después de backward

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: backward() — Calcular gradiente de pérdida

```python
import torch

# Modelo lineal: y = X @ w + b
X = torch.tensor([[1.0, 0.5], [2.0, 1.5], [3.0, 2.5]])  # 3 muestras, 2 features
y_real = torch.tensor([[10.0], [20.0], [30.0]])            # ventas reales

w = torch.randn(2, 1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

y_pred = X @ w + b
loss = ((y_pred - y_real) ** 2).mean()

loss.backward()  # calcula gradientes
print(f"Gradiente de w:\n{w.grad}")
print(f"Gradiente de b: {b.grad}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: backward() — Calcular gradiente de pérdida.*

1. Modelo lineal: y = X @ w + b

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: .grad — Acceder al gradiente calculado

```python
import torch

x = torch.tensor([3.0, 4.0, 5.0], requires_grad=True)
y = (x ** 2).sum()
y.backward()

print(f"x: {x}")
print(f"Gradientes de x (derivada de x^2 = 2x): {x.grad}")
# x = [3,4,5] → dy/dx = 2*[3,4,5] = [6,8,10]

# Aplicación: ¿cuánto cambia el costo total si sube el precio de cada producto?
precios = torch.tensor([100.0, 200.0, 150.0], requires_grad=True)
costo_total = (precios * 0.85).sum()  # 15% descuento
costo_total.backward()
print(f"Sensibilidad del costo al precio: {precios.grad}")  # [0.85, 0.85, 0.85]
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: .grad — Acceder al gradiente calculado.*

1. x = [3,4,5] → dy/dx = 2*[3,4,5] = [6,8,10]
2. Aplicación: ¿cuánto cambia el costo total si sube el precio de cada producto?

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: grad_fn — Función que generó el tensor

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
y = x ** 3
z = y + 5

print(f"x.grad_fn: {x.grad_fn}")  # None (hoja/leaf)
print(f"y.grad_fn: {y.grad_fn}")  # <PowBackward0>
print(f"z.grad_fn: {z.grad_fn}")  # <AddBackward0>

# Aplicación: inspeccionar el grafo computacional de un modelo
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: grad_fn — Función que generó el tensor.*

1. Aplicación: inspeccionar el grafo computacional de un modelo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: detach() — Desconectar tensor del grafo computacional

```python
import torch

x = torch.tensor([100.0, 200.0, 300.0], requires_grad=True)
y = x * 1.21  # IVA 21%

y_desconectado = y.detach()
print(f"y (con grad): {y}")
print(f"y_desconectado (sin grad): {y_desconectado}")
print(f"y_desconectado.requires_grad: {y_desconectado.requires_grad}")

# Útil para: extraer valores a NumPy sin gradiente, o para RL (stop-gradient)
import numpy as np
y_numpy = y.detach().numpy()
print(f"Como numpy: {y_numpy}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: detach() — Desconectar tensor del grafo computacional.*

1. Útil para: extraer valores a NumPy sin gradiente, o para RL (stop-gradient)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: with torch.no_grad() — Deshabilitar gradiente (inferencia)

```python
import torch

x = torch.tensor([150.0, 230.0, 89.0], requires_grad=True)

# Inferencia sin construir grafo
with torch.no_grad():
    y = x * 0.9 + 10  # predicción de precio con descuento
    print(f"Predicción (sin grafo): {y}")
    print(f"y.requires_grad: {y.requires_grad}")  # False

# Comparación: sin no_grad() sí crea grafo
y_con_grad = x * 0.9 + 10
print(f"y_con_grad.requires_grad: {y_con_grad.requires_grad}")  # True
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: with torch.no_grad() — Deshabilitar gradiente (inferencia).*

1. Inferencia sin construir grafo
2. Comparación: sin no_grad() sí crea grafo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: torch.autograd.Function — Función personalizada con forward/backward

```python
import torch

class CostoLogistico(torch.autograd.Function):
    """Costo logístico personalizado: penaliza stock-out y exceso de inventario"""
    @staticmethod
    def forward(ctx, demanda, inventario, costo_unitario):
        exceso = inventario - demanda
        ctx.save_for_backward(demanda, inventario, costo_unitario, exceso)
        # Costo: holding por exceso + penalización por faltante
        costo_holding = torch.where(exceso > 0, exceso * costo_unitario * 0.1, torch.tensor(0.0))
        costo_faltante = torch.where(exceso < 0, -exceso * costo_unitario * 0.5, torch.tensor(0.0))
        return costo_holding.sum() + costo_faltante.sum()

    @staticmethod
    def backward(ctx, grad_output):
        demanda, inventario, costo_unitario, exceso = ctx.saved_tensors
        grad_inventario = torch.where(exceso > 0, 0.1 * costo_unitario,
                                      torch.where(exceso < 0, -0.5 * costo_unitario, torch.tensor(0.0)))
        return None, grad_inventario * grad_output, None

costo_fn = CostoLogistico.apply
demanda = torch.tensor([100.0, 200.0, 150.0])
inventario = torch.tensor([120.0, 180.0, 140.0], requires_grad=True)
costo_unitario = torch.tensor(50.0)

costo = costo_fn(demanda, inventario, costo_unitario)
costo.backward()
print(f"Gradiente del inventario (sensibilidad al costo): {inventario.grad}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: torch.autograd.Function — Función personalizada con forward/backward.*

1. Costo: holding por exceso + penalización por faltante

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: torch.autograd.grad — Calcular gradiente de salida respecto a entrada

```python
import torch

x = torch.tensor([100.0, 200.0, 300.0], requires_grad=True)
y = x ** 2 + 3 * x + 10  # función de costo cuadrática

grad_y = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=False)
print(f"dy/dx (autograd.grad): {grad_y}")  # 2x + 3

# Aplicación: calcular gradiente de pérdida sin modificar el grafo
X = torch.randn(10, 4)
w = torch.randn(4, 1, requires_grad=True)
y_pred = X @ w
loss = (y_pred ** 2).mean()
grad_w = torch.autograd.grad(loss, w, retain_graph=True)[0]
print(f"Gradiente de pérdida respecto a w: {grad_w.shape}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: torch.autograd.grad — Calcular gradiente de salida respecto a entrada.*

1. Aplicación: calcular gradiente de pérdida sin modificar el grafo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: retain_graph=True — Retener grafo para múltiples backward

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
z = y ** 2  # z = x^4

# Primer backward
z.backward(retain_graph=True)
print(f"Gradiente tras primer backward: {x.grad}")  # 4*x^3 = 32

# Segundo backward (necesita retain_graph=True)
z.backward(retain_graph=True)
print(f"Gradiente tras segundo backward: {x.grad}")  # 32 + 32 = 64 (se acumula)

# Tercer backward
z.backward()
print(f"Gradiente tras tercer backward: {x.grad}")  # 96
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: retain_graph=True — Retener grafo para múltiples backward.*

1. Primer backward
2. Segundo backward (necesita retain_graph=True)
3. Tercer backward

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: create_graph=True — Crear grafo de segundo orden

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
y = x ** 3  # y = x^3

# Primer gradiente (dy/dx = 3x^2)
grad1 = torch.autograd.grad(y, x, create_graph=True)[0]
print(f"dy/dx = 3x^2 = {grad1}")  # 12

# Segunda derivada (d²y/dx² = 6x)
grad2 = torch.autograd.grad(grad1, x)[0]
print(f"d²y/dx² = 6x = {grad2}")  # 12

# Aplicación: meta-learning (MAML), optimización de hiperparámetros
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: create_graph=True — Crear grafo de segundo orden.*

1. Primer gradiente (dy/dx = 3x^2)
2. Segunda derivada (d²y/dx² = 6x)
3. Aplicación: meta-learning (MAML), optimización de hiperparámetros

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: register_hook — Inspeccionar/modificar gradientes

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# Hook para inspeccionar gradientes
def hook_fn(grad):
    print(f"Gradiente original: {grad}")
    # Modificar: clip gradientes para estabilidad
    grad_clipped = torch.clamp(grad, -1.0, 1.0)
    print(f"Gradiente clipped: {grad_clipped}")
    return grad_clipped

x.register_hook(hook_fn)

y = (x ** 2).sum() * 0.5
y.backward()
print(f"Gradiente final de x: {x.grad}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: register_hook — Inspeccionar/modificar gradientes.*

1. Hook para inspeccionar gradientes
2. Modificar: clip gradientes para estabilidad

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: Gradiente de MSE respecto a pesos

```python
import torch

# Datos: features de producto (precio, descuento, rating)
X = torch.tensor([[150.0, 0.10, 4.5],
                   [230.0, 0.05, 3.8],
                   [89.0, 0.20, 4.2],
                   [420.0, 0.15, 4.9],
                   [310.0, 0.08, 4.0]])
y_real = torch.tensor([[120.0], [180.0], [70.0], [350.0], [250.0]])  # unidades vendidas

w = torch.randn(3, 1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

y_pred = X @ w + b
loss = ((y_pred - y_real) ** 2).mean()

loss.backward()
print(f"Gradiente de MSE respecto a w:\n{w.grad}")
print(f"Gradiente de MSE respecto a b: {b.grad}")

# Interpretación: signo y magnitud indican dirección de ajuste
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: Gradiente de MSE respecto a pesos.*

1. Datos: features de producto (precio, descuento, rating)
2. Interpretación: signo y magnitud indican dirección de ajuste

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: Gradiente de CrossEntropyLoss

```python
import torch
import torch.nn.functional as F

# Clasificación de productos en categorías premium, estándar, económico
X = torch.tensor([[150.0, 4.5], [230.0, 3.8], [89.0, 4.2], [420.0, 4.9], [50.0, 3.0]])
y = torch.tensor([1, 1, 2, 0, 2])  # 0=premium, 1=estándar, 2=económico

w = torch.randn(2, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)

logits = X @ w + b
loss = F.cross_entropy(logits, y)

loss.backward()
print(f"Gradiente de CrossEntropy respecto a w:\n{w.grad}")
print(f"Gradiente de CrossEntropy respecto a b: {b.grad}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: Gradiente de CrossEntropyLoss.*

1. Clasificación de productos en categorías premium, estándar, económico

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: Gradiente de ReLU — Zona muerta cuando x < 0

```python
import torch

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
y = torch.relu(x)  # max(0, x)

loss = y.sum()
loss.backward()

print(f"x: {x}")
print(f"ReLU(x): {y}")
print(f"Gradiente de ReLU: {x.grad}")
# dy/dx ReLU: 1 si x>0, 0 si x<0 (indefinido en x=0, PyTorch usa 0)
# Muestra la "zona muerta": para x < 0 el gradiente es 0

# Aplicación en ventas: neuronas con ReLU mueren si el precio estandarizado es muy bajo
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: Gradiente de ReLU — Zona muerta cuando x < 0.*

1. dy/dx ReLU: 1 si x>0, 0 si x<0 (indefinido en x=0, PyTorch usa 0)
2. Muestra la "zona muerta": para x < 0 el gradiente es 0
3. Aplicación en ventas: neuronas con ReLU mueren si el precio estandarizado es muy bajo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: Chain rule manual — dy/dx = dy/du * du/dx con autograd

```python
import torch

x = torch.tensor([3.0], requires_grad=True)
u = x ** 2 + 2 * x + 1  # u = x² + 2x + 1
y = u ** 3              # y = u³

# dy/dx = dy/du * du/dx = 3u² * (2x + 2)
y.backward()

# Verificación manual
du_dx = 2 * x + 2  # = 8
u_val = x ** 2 + 2 * x + 1  # = 16
dy_du = 3 * u_val ** 2  # = 3*256 = 768
dy_dx_manual = dy_du * du_dx  # = 768*8 = 6144

print(f"dy/dx (autograd): {x.grad}")         # 6144
print(f"dy/dx (manual): {dy_dx_manual}")     # 6144

# Aplicación: cadenas de operaciones en redes profundas
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: Chain rule manual — dy/dx = dy/du * du/dx con autograd.*

1. dy/dx = dy/du * du/dx = 3u² * (2x + 2)
2. Verificación manual
3. Aplicación: cadenas de operaciones en redes profundas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 16: torch.autograd.profiler — Perfilar uso de memoria de gradientes

```python
import torch

def modelo_ventas(x, w1, w2):
    h = torch.relu(x @ w1)
    return h @ w2

torch.manual_seed(42)
x = torch.randn(1000, 50, requires_grad=True)
w1 = torch.randn(50, 128, requires_grad=True)
w2 = torch.randn(128, 1, requires_grad=True)

with torch.autograd.profiler.profile(use_cpu=True) as prof:
    y = modelo_ventas(x, w1, w2)
    loss = y.sum()
    loss.backward()

print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
# Muestra qué operaciones consumen más tiempo
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: torch.autograd.profiler — Perfilar uso de memoria de gradientes.*

1. Muestra qué operaciones consumen más tiempo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 17: set_grad_enabled(False) — Deshabilitar globalmente

```python
import torch

# Deshabilitar gradientes globalmente
torch.autograd.set_grad_enabled(False)

x = torch.tensor([5.0], requires_grad=True)  # requires_grad=True se ignora
y = x ** 2
print(f"y.requires_grad: {y.requires_grad}")  # False

# Rehabilitar
torch.autograd.set_grad_enabled(True)
x2 = torch.tensor([5.0], requires_grad=True)
y2 = x2 ** 2
print(f"y2.requires_grad: {y2.requires_grad}")  # True

# Aplicación: alternar entre entrenamiento/evaluación en loops largos
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: set_grad_enabled(False) — Deshabilitar globalmente.*

1. Deshabilitar gradientes globalmente
2. Rehabilitar
3. Aplicación: alternar entre entrenamiento/evaluación en loops largos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 18: Integrador — Gradiente descendiente manual con autograd

```python
import torch
import matplotlib.pyplot as plt

# Predicción de ventas con modelo lineal usando GD manual
torch.manual_seed(42)
X = torch.randn(100, 1) * 10 + 50  # precios
y_real = 3.0 * X + 15.0 + torch.randn(100, 1) * 5  # ventas

w = torch.randn(1, 1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
lr = 0.001
epochs = 500
losses = []

for epoch in range(epochs):
    y_pred = X @ w + b
    loss = ((y_pred - y_real) ** 2).mean()

    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()

    losses.append(loss.item())

print(f"Peso final: {w.item():.4f}, sesgo final: {b.item():.4f}")
print(f"Pérdida final: {losses[-1]:.4f}")
print(f"w real debería ser ~3.0, b real ~15.0")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador — Gradiente descendiente manual con autograd.*

1. Predicción de ventas con modelo lineal usando GD manual

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejercicios

1. Crea un tensor `precios` con `requires_grad=True` que contenga [50, 100, 150, 200, 250]. Define una función de costo `costo = sum(precios**2)`. Calcula el gradiente y explica qué significa.

2. Usa `torch.autograd.Function` para implementar una función de activación personalizada "ventas_activation" que en forward haga `x**2` si x>0 y `x*0.1` si x<=0 (LeakyReLU variante). Implementa backward manual.

3. Dado el grafo: `a = torch.tensor([2.0], requires_grad=True)`, `b = a**3 + 2*a + 1`, `c = b**2`. Calcula dc/da usando autograd y verifica manualmente con la regla de la cadena.

4. Crea un modelo lineal `y = Xw + b` para 100 muestras de ventas con 3 features. Usa `torch.autograd.grad` (no `.backward()`) para obtener los gradientes sin modificar el grafo principal.

5. Implementa gradient clipping usando `register_hook` que limite los gradientes al rango [-10, 10] para un modelo simple de predicción de demanda.

6. Usa `with torch.no_grad()` para calcular predicciones de un modelo de ventas sin construir el grafo, y compara el tiempo de ejecución con/sin `no_grad()` usando `timeit`.

7. Utiliza `torch.autograd.profiler` para perfilart el cómputo forward+backward de una red con 3 capas lineales con 64, 128 y 1 neurona(s). Identifica la operación más lenta.

8. Implementa un paso de gradiente descendiente manual (sin optimizador) para un modelo de predicción de compras con 5 features y 500 epochs. Usa `retain_graph=True` para llamar a backward dos veces: una para pérdida principal y otra para una pérdida auxiliar (regularización L2 manual).

---

## Resumen

Autograd es el motor que hace posible el entrenamiento de redes neuronales en PyTorch. Construye un grafo dinámico que registra cada operación y, al llamar `.backward()`, calcula gradientes automáticamente mediante diferenciación en modo reverso.

| Concepto | Uso principal |
|----------|---------------|
| `requires_grad=True` | Activar gradientes en pesos |
| `backward()` | Calcular gradientes |
| `.grad` | Leer gradientes |
| `grad_fn` | Inspeccionar origen del tensor |
| `detach()` | Extraer sin gradiente (NumPy, logging) |
| `with torch.no_grad()` | Inferencia eficiente |
| `autograd.Function` | Operaciones personalizadas diferenciables |
| `retain_graph` | Múltiples backward |
| `create_graph` | Derivadas de segundo orden |
| `register_hook` | Inspeccionar/modificar gradientes |
| `set_grad_enabled` | Control global de gradientes |

**En negocio**: Autograd permite optimizar cualquier función diferenciable: desde modelos lineales de ventas hasta redes profundas para predicción de demanda, clasificación de productos y optimización de inventarios.
