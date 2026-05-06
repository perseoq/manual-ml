# A20: Introducción a PyTorch — Tensores y Operaciones para Ventas, Compras e Inventarios

## Introducción Teórica

**PyTorch** es un framework de deep learning desarrollado por Meta (Facebook) que se caracteriza por su computación dinámica (define-by-run), su integración natural con Python y su ecosistema flexible. Es ampliamente utilizado tanto en investigación como en producción.

### Conceptos clave:

1. **torch.Tensor**: Estructura multidimensional similar a ndarray de NumPy pero con soporte para GPU y diferenciación automática. Atributos: `shape`, `dtype`, `device`.
2. **Creación de tensores**: `torch.tensor()`, `torch.zeros()`, `torch.ones()`, `torch.rand()`, `torch.randn()`, `torch.arange()`, `torch.linspace()`.
3. **Device**: Los tensores pueden estar en CPU o GPU (`tensor.to(device='cuda'/'cpu')`). Mover tensores entre dispositivos.
4. **Manipulación**: `view()`, `reshape()`, `stack()`, `cat()`, `chunk()`, `split()` para reorganizar tensores.
5. **Álgebra lineal**: `torch.mm()` (matrix multiply), `torch.matmul()`, `torch.bmm()` (batch matmul).
6. **Estadísticas**: `torch.mean()`, `torch.sum()`, `torch.max()`, `torch.min()`, `torch.argmax()`, `torch.argmin()`.
7. **Operaciones condicionales**: `torch.where()`, `torch.clamp()` para filtrar y limitar valores.
8. **Funciones de activación**: `torch.sigmoid()`, `torch.relu()`, `torch.softmax()`, `torch.tanh()`.
9. **Operaciones matemáticas**: `torch.exp()`, `torch.log()`, `torch.sqrt()`, `torch.abs()`.
10. **Conversión**: `tensor.numpy()` para convertir a NumPy; `torch.from_numpy()` para NumPy a tensor.
11. **requires_grad**: Flag que activa el seguimiento de gradientes para diferenciación automática.

### Aplicaciones en negocio:

- **Ventas**: Tensores de precios y cantidades; álgebra lineal para predicciones; funciones de activación para clasificación.
- **Compras**: Stack/cat de órdenes de múltiples proveedores; where para filtrar condiciones; clamp para límites de precios.
- **Inventarios**: Tensores 3D (producto × almacén × tiempo); operaciones de reducción para rotación.

---

## Ejemplos

### Ejemplo 1: torch.tensor — Tensor de precios de productos

```python
import torch
import numpy as np

# Tensor desde lista Python
precios = torch.tensor([150.0, 230.0, 89.0, 420.0, 310.0], dtype=torch.float32)
print(f"Tensor de precios: {precios}")
print(f"Shape: {precios.shape}, dtype: {precios.dtype}, device: {precios.device}")

# Tensor 2D: 3 sucursales × 5 productos
precios_sucursales = torch.tensor([
    [150.0, 230.0, 89.0, 420.0, 310.0],
    [145.0, 225.0, 92.0, 415.0, 305.0],
    [160.0, 240.0, 85.0, 430.0, 320.0]
])
print(f"Precios por sucursal:\n{precios_sucursales}")
print(f"Shape: {precios_sucursales.shape}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: torch.tensor — Tensor de precios de productos.*

1. Tensor desde lista Python
2. Tensor 2D: 3 sucursales × 5 productos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: torch.zeros/ones/rand — Crear tensores inicializados para inventarios

```python
# Inventario inicial cero para 10 productos en 4 almacenes
inventario_cero = torch.zeros((10, 4), dtype=torch.int32)
print(f"Inventario cero (10 prod × 4 almacenes):\n{inventario_cero}")

# Matriz de unos para sesgos de red neuronal
sesgos = torch.ones(3, dtype=torch.float32)
print(f"Sesgos inicializados en 1: {sesgos}")

# Precios aleatorios uniformes [10, 500)
precios_aleatorios = torch.rand(8) * 490 + 10
print(f"Precios aleatorios: {precios_aleatorios}")

# Pesos aleatorios con distribución normal (media=0, std=0.1)
pesos_iniciales = torch.randn(5, 3) * 0.1
print(f"Pesos iniciales normal:\n{pesos_iniciales}")

# Arange y linspace
dias = torch.arange(1, 32, dtype=torch.float32)
precios_lin = torch.linspace(100.0, 500.0, steps=10)
print(f"Días del mes: {dias}")
print(f"10 precios equiespaciados: {precios_lin}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: torch.zeros/ones/rand — Crear tensores inicializados para inventarios.*

1. Inventario inicial cero para 10 productos en 4 almacenes
2. Matriz de unos para sesgos de red neuronal
3. Precios aleatorios uniformes [10, 500)
4. Pesos aleatorios con distribución normal (media=0, std=0.1)
5. Arange y linspace

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: tensor.shape, dtype, device — Propiedades del tensor

```python
tensor_ventas = torch.tensor([
    [[150.0, 230.0], [89.0, 420.0]],
    [[310.0, 175.0], [520.0, 67.0]]
])
print(f"Tensor: {tensor_ventas}")
print(f"shape (dimensiones): {tensor_ventas.shape}")
print(f"dtype (tipo de dato): {tensor_ventas.dtype}")
print(f"device (dispositivo): {tensor_ventas.device}")
print(f"ndim (número de dims): {tensor_ventas.ndim}")
print(f"numel (elementos totales): {tensor_ventas.numel()}")
print(f"Element size (bytes): {tensor_ventas.element_size()}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: tensor.shape, dtype, device — Propiedades del tensor.*

1. `print(f"Tensor: {tensor_ventas}")` — Muestra el resultado por pantalla.
2. `print(f"shape (dimensiones): {tensor_ventas.shape}")` — Muestra el resultado por pantalla.
3. `print(f"dtype (tipo de dato): {tensor_ventas.dtype}")` — Muestra el resultado por pantalla.
4. `print(f"device (dispositivo): {tensor_ventas.device}")` — Muestra el resultado por pantalla.
5. `print(f"ndim (número de dims): {tensor_ventas.ndim}")` — Muestra el resultado por pantalla.
6. `print(f"numel (elementos totales): {tensor_ventas.numel()}")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: tensor.to(device='cuda') — Mover a GPU (si disponible)

```python
tensor_cpu = torch.tensor([150.0, 230.0, 89.0], dtype=torch.float32)
print(f"Tensor en CPU: {tensor_cpu.device}")

# Verificar CUDA
if torch.cuda.is_available():
    tensor_gpu = tensor_cpu.to('cuda')
    print(f"Tensor en GPU: {tensor_gpu.device}")

    tensor_back = tensor_gpu.to('cpu')
    print(f"Tensor de vuelta a CPU: {tensor_back.device}")

    # Crear directamente en GPU
    tensor_directo = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    print(f"Tensor creado en GPU: {tensor_directo.device}")
else:
    print("CUDA no disponible. El tensor permanece en CPU.")
    print(f"Para usar GPU: tensor.to('cuda') cuando esté disponible")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: tensor.to(device='cuda') — Mover a GPU (si disponible).*

1. Verificar CUDA
2. Crear directamente en GPU

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: torch.view — Reshape tensor de precios de productos

```python
# 12 precios en vector plano
precios_flat = torch.tensor([
    150.0, 230.0, 89.0, 420.0, 310.0, 175.0,
    520.0, 67.0, 890.0, 120.0, 450.0, 380.0
])
print(f"Original: shape={precios_flat.shape}")

# Reorganizar como 3 sucursales × 4 productos
precios_3x4 = precios_flat.view(3, 4)
print(f"View 3×4:\n{precios_3x4}")

# Reorganizar como 2 grupos × 2 sucursales × 3 productos
precios_2x2x3 = precios_flat.view(2, 2, 3)
print(f"View 2×2×3:\n{precios_2x2x3}")

# -1: inferir dimensión automáticamente
precios_auto = precios_flat.view(-1, 6)
print(f"View (-1, 6): shape={precios_auto.shape}")

# reshape (similar a view pero crea copia si es necesario)
precios_reshaped = precios_flat.reshape(4, 3)
print(f"Reshape 4×3:\n{precios_reshaped}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: torch.view — Reshape tensor de precios de productos.*

1. 12 precios en vector plano
2. Reorganizar como 3 sucursales × 4 productos
3. Reorganizar como 2 grupos × 2 sucursales × 3 productos
4. -1: inferir dimensión automáticamente
5. reshape (similar a view pero crea copia si es necesario)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: torch.stack — Apilar tensores de múltiples sucursales

```python
# Ventas diarias de 3 sucursales (cada una 7 días)
sucursal_1 = torch.tensor([1200, 1500, 1100, 1300, 1600, 1400, 1550])
sucursal_2 = torch.tensor([900, 950, 1000, 850, 1100, 1050, 980])
sucursal_3 = torch.tensor([2000, 2100, 1900, 2200, 2050, 2150, 1950])

# Apilar: crea nueva dimensión en axis=0
ventas_stack = torch.stack([sucursal_1, sucursal_2, sucursal_3])
print(f"Stack (3 sucursales × 7 días):\n{ventas_stack}")
print(f"Shape: {ventas_stack.shape}")

# Stack en axis=1: 7 días × 3 sucursales
ventas_stack_axis1 = torch.stack([sucursal_1, sucursal_2, sucursal_3], dim=1)
print(f"Stack axis=1 (7 días × 3 sucursales):\n{ventas_stack_axis1}")
print(f"Shape: {ventas_stack_axis1.shape}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: torch.stack — Apilar tensores de múltiples sucursales.*

1. Ventas diarias de 3 sucursales (cada una 7 días)
2. Apilar: crea nueva dimensión en axis=0
3. Stack en axis=1: 7 días × 3 sucursales

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: torch.cat — Concatenar tensores de diferentes días

```python
# Ventas de semana 1 y semana 2
semana1 = torch.tensor([[1200, 1500], [900, 950]], dtype=torch.float32)  # 2 suc × 2 días
semana2 = torch.tensor([[1100, 1300], [1000, 850]], dtype=torch.float32)
semana3 = torch.tensor([[1600, 1400], [1100, 1050]], dtype=torch.float32)

# Concatenar en axis=1 (días)
tres_semanas = torch.cat([semana1, semana2, semana3], dim=1)
print(f"3 semanas concatenadas (2 suc × 6 días):\n{tres_semanas}")
print(f"Shape: {tres_semanas.shape}")

# Concatenar en axis=0 (sucursales)
nueva_sucursal = torch.tensor([[2000, 2100], [1900, 2200], [2050, 2150]], dtype=torch.float32)
print(f"Nueva sucursal shape: {nueva_sucursal.shape}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: torch.cat — Concatenar tensores de diferentes días.*

1. Ventas de semana 1 y semana 2
2. Concatenar en axis=1 (días)
3. Concatenar en axis=0 (sucursales)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: torch.mm — Multiplicación de matrices (features × pesos) para predicción

```python
# 5 productos con 3 features: [precio, descuento, stock_dias]
features = torch.tensor([
    [150.0, 0.10, 5.0],
    [230.0, 0.05, 2.0],
    [89.0,  0.20, 10.0],
    [420.0, 0.15, 1.0],
    [310.0, 0.08, 3.0]
])

pesos = torch.tensor([[0.5], [1.2], [-0.3]])
sesgo = torch.tensor([10.0])

# y = X @ w + b (mm = matrix multiply)
predicciones = torch.mm(features, pesos) + sesgo
print(f"Predicciones de ventas:\n{predicciones}")

# matmul (más flexible)
pred_matmul = torch.matmul(features, pesos) + sesgo
print(f"Con matmul:\n{pred_matmul}")

# Batch matmul (bmm): múltiples matrices a la vez
batches = features.unsqueeze(0).repeat(3, 1, 1)  # 3 batches
pesos_batch = pesos.unsqueeze(0).repeat(3, 1, 1)
pred_batch = torch.bmm(batches, pesos_batch)
print(f"Batch matmul shape: {pred_batch.shape}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: torch.mm — Multiplicación de matrices (features × pesos) para predicción.*

1. 5 productos con 3 features: [precio, descuento, stock_dias]
2. y = X @ w + b (mm = matrix multiply)
3. matmul (más flexible)
4. Batch matmul (bmm): múltiples matrices a la vez

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: torch.mean — Promedio de ingresos en tensor

```python
# Ingresos: 4 sucursales × 7 días
ingresos = torch.tensor([
    [1200., 1500., 1100., 1300., 1600., 1400., 1550.],
    [900.,  950.,  1000., 850.,  1100., 1050., 980.],
    [2000., 2100., 1900., 2200., 2050., 2150., 1950.],
    [600.,  650.,  700.,  550.,  800.,  750.,  620.]
])

promedio_global = torch.mean(ingresos)
promedio_sucursal = torch.mean(ingresos, dim=1)
promedio_dia = torch.mean(ingresos, dim=0)

print(f"Ingreso promedio global: ${promedio_global.item():.2f}")
print(f"Ingreso promedio por sucursal: {promedio_sucursal}")
print(f"Ingreso promedio por día: {promedio_dia}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: torch.mean — Promedio de ingresos en tensor.*

1. Ingresos: 4 sucursales × 7 días

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: torch.sum — Suma total de ventas

```python
# Ventas 10 productos × 30 días
ventas = torch.randint(0, 100, (10, 30), dtype=torch.float32)
total = torch.sum(ventas)
total_por_producto = torch.sum(ventas, dim=1)
total_por_dia = torch.sum(ventas, dim=0)

print(f"Total ventas mes: {total.item():.0f} unidades")
print(f"Producto con más ventas: producto {torch.argmax(total_por_producto).item() + 1} "
      f"({total_por_producto.max().item():.0f} uds)")
print(f"Día con más ventas: día {torch.argmax(total_por_dia).item() + 1} "
      f"({total_por_dia.max().item():.0f} uds)")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: torch.sum — Suma total de ventas.*

1. Ventas 10 productos × 30 días

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: torch.max/torch.min — Valor máximo/mínimo de ingresos

```python
ingresos = torch.tensor([[1200., 1500., 1100.], [900., 950., 1000.]])
max_valor, max_indices = torch.max(ingresos, dim=1)
min_valor, min_indices = torch.min(ingresos, dim=1)

print(f"Matrix:\n{ingresos}")
print(f"Máximo por sucursal: {max_valor}, en índices: {max_indices}")
print(f"Mínimo por sucursal: {min_valor}, en índices: {min_indices}")

# Global max/min
print(f"Máximo global: {torch.max(ingresos).item()}")
print(f"Mínimo global: {torch.min(ingresos).item()}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: torch.max/torch.min — Valor máximo/mínimo de ingresos.*

1. Global max/min

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: torch.argmax — Índice del valor máximo (categoría de producto más probable)

```python
# Probabilidades predichas para 5 productos en 3 categorías
prob_categorias = torch.tensor([
    [0.2, 0.7, 0.1],
    [0.8, 0.1, 0.1],
    [0.1, 0.3, 0.6],
    [0.4, 0.4, 0.2],
    [0.05, 0.15, 0.8]
])

categoria_predicha = torch.argmax(prob_categorias, dim=1)
print(f"Probabilidades:\n{prob_categorias}")
print(f"Categoría predicha por producto: {categoria_predicha}")

# También argmin
categoria_menos_probable = torch.argmin(prob_categorias, dim=1)
print(f"Categoría menos probable: {categoria_menos_probable}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: torch.argmax — Índice del valor máximo (categoría de producto más probable).*

1. Probabilidades predichas para 5 productos en 3 categorías
2. También argmin

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: torch.where — Filtrar productos con margen > 0

```python
# Márgenes de 8 productos (pueden ser negativos)
margenes = torch.tensor([0.25, -0.05, 0.15, 0.30, -0.10, 0.05, 0.20, -0.02])
umbral = 0.0

# where(condition, x, y) → x donde condition=True, y donde False
rentables = torch.where(margenes > umbral, margenes, torch.tensor(0.0))
print(f"Márgenes originales: {margenes}")
print(f"Rentables (>0): {rentables}")
print(f"Productos rentables: {(margenes > umbral).sum().item()} de {margenes.numel()}")

# Usar where para asignar valores
etiquetas = torch.where(margenes > 0.1, torch.tensor(1.0), torch.tensor(0.0))
print(f"Etiquetas (margen>0.1 = 1): {etiquetas}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: torch.where — Filtrar productos con margen > 0.*

1. Márgenes de 8 productos (pueden ser negativos)
2. where(condition, x, y) → x donde condition=True, y donde False
3. Usar where para asignar valores

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: torch.clamp — Limitar precios entre min y max

```python
precios = torch.tensor([150.0, -230.0, 89.0, 420.0, -310.0, 175.0, 520.0, -67.0])
print(f"Precios originales (con negativos): {precios}")

# Limitar entre 0 y 500
precios_limitados = torch.clamp(precios, min=0.0, max=500.0)
print(f"Precios clamp [0, 500]: {precios_limitados}")

# Solo mínimo
precios_sin_negativos = torch.clamp(precios, min=0.0)
print(f"Sin negativos: {precios_sin_negativos}")

# Solo máximo
precios_sin_excesivos = torch.clamp(precios, max=400.0)
print(f"Sin > 400: {precios_sin_excesivos}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: torch.clamp — Limitar precios entre min y max.*

1. Limitar entre 0 y 500
2. Solo mínimo
3. Solo máximo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: torch.sigmoid — Convertir logits a probabilidades de venta

```python
# Logits de clasificación (puntajes antes de activación)
logits = torch.tensor([2.0, -1.5, 0.0, 3.2, -0.5, 1.8, -3.0])
probabilidades = torch.sigmoid(logits)

print(f"Logits: {logits}")
print(f"Probabilidades de venta: {probabilidades}")
print(f"Decisión (p > 0.5): {(probabilidades > 0.5).int()}")

# Confianza promedio
print(f"Confianza promedio: {torch.mean(probabilidades).item():.4f}")
print(f"Mínima confianza: {torch.min(probabilidades).item():.4f}")
print(f"Máxima confianza: {torch.max(probabilidades).item():.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: torch.sigmoid — Convertir logits a probabilidades de venta.*

1. Logits de clasificación (puntajes antes de activación)
2. Confianza promedio

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 16: torch.relu — Activación ReLU (ventas no negativas)

```python
# Predicciones lineales (pueden ser negativas)
predicciones_lineales = torch.tensor([-5.0, 12.0, -3.5, 8.0, 0.0, -10.0, 25.0])
predicciones_relu = torch.relu(predicciones_lineales)

print(f"Predicciones lineales: {predicciones_lineales}")
print(f"Después de ReLU (>=0): {predicciones_relu}")

# Leaky ReLU
leaky_relu = torch.nn.functional.leaky_relu(predicciones_lineales, negative_slope=0.01)
print(f"Leaky ReLU: {leaky_relu}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: torch.relu — Activación ReLU (ventas no negativas).*

1. Predicciones lineales (pueden ser negativas)
2. Leaky ReLU

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 17: torch.softmax — Distribución de probabilidad entre categorías

```python
# 6 productos, 4 categorías de producto
logits_categorias = torch.tensor([
    [2.0, 1.0, 0.1, 0.5],
    [0.5, 2.5, 1.0, 0.2],
    [3.0, 0.5, 0.8, 1.2],
    [1.0, 1.0, 2.0, 0.8],
    [0.1, 0.5, 3.0, 2.0],
    [1.5, 2.0, 0.5, 1.0]
])

prob_softmax = torch.softmax(logits_categorias, dim=1)
print(f"Probabilidades (softmax):\n{prob_softmax}")
print(f"Suma por fila (debe ser 1): {torch.sum(prob_softmax, dim=1)}")

categorias = ['Electrónica', 'Ropa', 'Alimentos', 'Hogar']
predichas = torch.argmax(prob_softmax, dim=1)
for i, cat in enumerate(predichas):
    print(f"  Producto {i+1}: {categorias[cat]} (confianza: {prob_softmax[i, cat].item():.1%})")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: torch.softmax — Distribución de probabilidad entre categorías.*

1. 6 productos, 4 categorías de producto

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 18: Integrador — Regresión lineal con tensores de PyTorch desde cero

```python
import torch

torch.manual_seed(42)

# Datos sintéticos: precio → cantidad vendida
n = 200
precios_reales = torch.rand(n, 1) * 90 + 10  # [10, 100]
w_real, b_real = 2.5, 50.0
ruido = torch.randn(n, 1) * 15
cantidades_reales = w_real * precios_reales + b_real + ruido

# Modelo lineal con requires_grad para autograd
w = torch.randn(1, 1, requires_grad=True) * 0.5
b = torch.zeros(1, requires_grad=True)

def modelo(X):
    return X @ w + b

def loss(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)

optimizer = torch.optim.SGD([w, b], lr=0.0001)

epochs = 500
for ep in range(epochs):
    optimizer.zero_grad()
    predicciones = modelo(precios_reales)
    perdida = loss(predicciones, cantidades_reales)
    perdida.backward()
    optimizer.step()

    if (ep + 1) % 100 == 0:
        print(f"Epoch {ep+1}: loss={perdida.item():.4f}, w={w.item():.4f}, b={b.item():.4f}")

print(f"\nw aprendido: {w.item():.4f} (real: {w_real})")
print(f"b aprendido: {b.item():.4f} (real: {b_real})")

# Predicción para nuevos precios
nuevos_precios = torch.tensor([[45.0], [80.0], [25.0]])
predicciones_finales = modelo(nuevos_precios)
print(f"Predicciones: {predicciones_finales.detach().numpy().flatten()}")

# Convertir a numpy
w_np = w.detach().numpy()
b_np = b.detach().numpy()
print(f"Pesos como numpy: {w_np.flatten()}, {b_np.flatten()}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador — Regresión lineal con tensores de PyTorch desde cero.*

1. Datos sintéticos: precio → cantidad vendida
2. Modelo lineal con requires_grad para autograd
3. Predicción para nuevos precios
4. Convertir a numpy

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejercicios

1. **Tensores de inventario**: Crea un tensor 3D (5 productos × 3 almacenes × 7 días) con valores aleatorios de stock entre 0 y 100. Calcula: stock promedio por producto, stock total por almacén, stock máximo por día.

2. **Multiplicación de matrices para predicción**: Genera una matriz X de 100 filas × 4 columnas (features de ventas) y un vector w de 4×1 (pesos). Calcula y = X @ w + b. Luego suma sesgo y aplica ReLU.

3. **Stack y cat de sucursales**: Crea 4 tensores 1D de 10 elementos cada uno (ventas de 4 sucursales). Apílalos con stack para obtener (4, 10). Luego concatena con otro tensor de 10 elementos para obtener (5, 10).

4. **Filtrado con where**: Genera un tensor de 20 márgenes aleatorios entre -0.2 y 0.5. Usa `torch.where` para crear un tensor de etiquetas: 1 si margen > 0.15, 0 si está entre 0 y 0.15, -1 si es negativo.

5. **Softmax para recomendación**: Crea logits para 8 productos en 5 categorías. Aplica softmax y encuentra la categoría más probable para cada producto. Muestra la confianza promedio de las predicciones.

6. **Regresión con SGD y Adam**: Implementa el modelo de regresión lineal del ejemplo 18 pero usando `torch.optim.Adam` en lugar de SGD. Compara la convergencia (¿cuántos epochs necesita cada uno para loss < 10?).

7. **Manipulación avanzada**: Crea un tensor (4, 6) de precios. Usa view para reshape a (2, 12). Luego clamp con min=10 y max=100. Finalmente calcula mean, std y argmax por fila.

8. **Integrador completo con GPU**: Genera un dataset sintético de 500 puntos con 3 features y un target continuo. Implementa regresión lineal con PyTorch usando: (a) tensor con requires_grad, (b) función forward, (c) MSE loss, (d) optimizador Adam. Mueve todo a GPU si está disponible. Entrena hasta convergencia y reporta loss final y parámetros aprendidos.

---

## Resumen

Hemos cubierto las operaciones fundamentales de PyTorch para análisis de ventas, compras e inventarios:

- **Creación de tensores**: `torch.tensor` (desde datos), `torch.zeros/ones/rand/randn` (inicializados), `torch.arange/linspace` (secuencias).
- **Propiedades**: `shape`, `dtype`, `device` para inspeccionar tensores; `to(device)` para mover entre CPU/GPU.
- **Manipulación**: `view/reshape` (cambiar forma), `stack` (apilar con nueva dimensión), `cat` (concatenar en dimensión existente).
- **Álgebra lineal**: `mm/matmul` (multiplicación de matrices), `bmm` (batch matmul) para predicciones.
- **Estadísticas**: `mean`, `sum`, `max/min` (con y sin `dim`), `argmax/argmin` para análisis y clasificación.
- **Operaciones condicionales**: `where` (selección condicional), `clamp` (limitar rangos de precios o valores).
- **Funciones de activación**: `sigmoid` (probabilidades binarias), `relu` (valores no negativos), `softmax` (distribución multiclase).
- **Conversión**: `tensor.numpy()` para interoperar con NumPy; `requires_grad` para diferenciación automática.

PyTorch ofrece un enfoque más "pythonico" y dinámico que TensorFlow, con diferenciación automática vía autograd, ideal para prototipado rápido y experimentación en problemas de negocio.
