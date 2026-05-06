# A15: Introducción a TensorFlow — Operaciones Fundamentales para Ventas, Compras e Inventarios

## Introducción Teórica

**TensorFlow** es un framework de código abierto desarrollado por Google para cómputo numérico y machine learning a gran escala. Su unidad básica es el **tensor**, una generalización de matrices a N dimensiones.

### Conceptos clave:

1. **tf.Tensor**: Estructura de datos inmutable. Puede ser constante (tf.constant) o variable (tf.Variable). Los tensores fluyen a través de un grafo de operaciones.
2. **tf.GradientTape**: Contexto que registra operaciones para calcular gradientes automáticamente (diferenciación automática). Esencial para entrenar modelos.
3. **tf.function / @tf.function**: Decorador que compila una función Python en un grafo de TensorFlow, mejorando la velocidad de ejecución.
4. **tf.nn**: Módulo con funciones de activación (relu, sigmoid, softmax, tanh) esenciales para redes neuronales.
5. **tf.losses**: Funciones de pérdida para medir error entre predicción y valor real.
6. **tf.optimizers**: Algoritmos de optimización (Adam, SGD) que actualizan pesos usando gradientes.
7. **tf.data**: API para construir pipelines eficientes de datos.
8. **tf.math**: Operaciones matemáticas (exp, log, sqrt, etc.).
9. **tf.cast**: Conversión de tipos de datos en tensores.

### Aplicaciones en negocio:

- **Ventas**: Tensores de precios, cantidades, montos; redes que predicen demanda.
- **Compras**: Features numéricas de proveedores; clasificación de lead time.
- **Inventarios**: Matrices de rotación; optimización de stock con gradientes.

---

## Ejemplos

### Ejemplo 1: tf.constant — Tensor constante de precios de productos

```python
import tensorflow as tf
import numpy as np

precios = tf.constant([150.0, 230.0, 89.0, 420.0, 310.0], dtype=tf.float32)
print(f"Tensor de precios: {precios.numpy()}")
print(f"Shape: {precios.shape}, dtype: {precios.dtype}")

# Precios de 3 sucursales en 5 productos
precios_matriz = tf.constant([
    [150.0, 230.0, 89.0, 420.0, 310.0],
    [145.0, 225.0, 92.0, 415.0, 305.0],
    [160.0, 240.0, 85.0, 430.0, 320.0]
])
print(f"Matriz de precios por sucursal:\n{precios_matriz.numpy()}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: tf.constant — Tensor constante de precios de productos.*

1. Precios de 3 sucursales en 5 productos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: tf.Variable — Variable de pesos inicializados para un modelo de ventas

```python
# Pesos inicializados aleatoriamente para un modelo lineal: 3 features -> 1 predicción
tf.random.set_seed(42)
pesos = tf.Variable(tf.random.normal([3, 1], mean=0.0, stddev=0.1, dtype=tf.float32))
sesgo = tf.Variable(tf.zeros([1], dtype=tf.float32))

print(f"Pesos iniciales:\n{pesos.numpy()}")
print(f"Sesgo inicial: {sesgo.numpy()}")

# Actualizar pesos manualmente (simula un paso de entrenamiento)
pesos.assign(pesos * 0.99)
print(f"Pesos después de decaimiento:\n{pesos.numpy()}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: tf.Variable — Variable de pesos inicializados para un modelo de ventas.*

1. Pesos inicializados aleatoriamente para un modelo lineal: 3 features -> 1 predicción
2. Actualizar pesos manualmente (simula un paso de entrenamiento)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: tf.cast — Convertir tipos de datos (int32 a float32) en montos de ventas

```python
montos_enteros = tf.constant([150, 230, 89, 420, 310], dtype=tf.int32)
print(f"Original dtype: {montos_enteros.dtype}")

montos_flotantes = tf.cast(montos_enteros, dtype=tf.float32)
print(f"Convertido dtype: {montos_flotantes.dtype}")
print(f"Montos: {montos_flotantes.numpy()}")

# También útil para booleanos
flags = tf.constant([True, False, True, False, False])
print(f"Boolean a float32: {tf.cast(flags, tf.float32).numpy()}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: tf.cast — Convertir tipos de datos (int32 a float32) en montos de ventas.*

1. También útil para booleanos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: tf.reduce_mean — Promedio de ingresos por sucursal

```python
# Ingresos diarios de 4 sucursales durante 7 días
ingresos = tf.constant([
    [1200.0, 1500.0, 1100.0, 1300.0, 1600.0, 1400.0, 1550.0],
    [900.0,  950.0,  1000.0, 850.0,  1100.0, 1050.0, 980.0],
    [2000.0, 2100.0, 1900.0, 2200.0, 2050.0, 2150.0, 1950.0],
    [600.0,  650.0,  700.0,  550.0,  800.0,  750.0,  620.0]
], dtype=tf.float32)

promedio_global = tf.reduce_mean(ingresos)
promedio_por_sucursal = tf.reduce_mean(ingresos, axis=1)
promedio_por_dia = tf.reduce_mean(ingresos, axis=0)

print(f"Ingreso promedio global: ${promedio_global.numpy():.2f}")
print(f"Ingreso promedio por sucursal: {promedio_por_sucursal.numpy()}")
print(f"Ingreso promedio por día (L-D): {promedio_por_dia.numpy()}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: tf.reduce_mean — Promedio de ingresos por sucursal.*

1. Ingresos diarios de 4 sucursales durante 7 días

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: tf.reduce_sum — Suma total de ventas del mes

```python
# Ventas diarias de 3 productos durante 30 días
ventas_mes = tf.random.uniform([30, 3], minval=0, maxval=100, dtype=tf.int32)
total_mes = tf.reduce_sum(ventas_mes)
total_por_producto = tf.reduce_sum(ventas_mes, axis=0)
total_por_dia = tf.reduce_sum(ventas_mes, axis=1)

print(f"Total ventas del mes: {total_mes.numpy()} unidades")
print(f"Total por producto: {total_por_producto.numpy()}")
print(f"Día con más ventas: día {tf.argmax(total_por_dia).numpy() + 1}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: tf.reduce_sum — Suma total de ventas del mes.*

1. Ventas diarias de 3 productos durante 30 días

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: tf.matmul — Multiplicación de matrices (features × pesos) para predicción lineal

```python
# 5 productos con 3 features: [precio, descuento, días_sin_stock]
features = tf.constant([
    [150.0, 0.10, 5.0],
    [230.0, 0.05, 2.0],
    [89.0,  0.20, 10.0],
    [420.0, 0.15, 1.0],
    [310.0, 0.08, 3.0]
], dtype=tf.float32)

pesos_modelo = tf.constant([[0.5], [1.2], [-0.3]], dtype=tf.float32)

# y = X @ w + b
predicciones = tf.matmul(features, pesos_modelo)
print(f"Predicciones (ventas estimadas):\n{predicciones.numpy().flatten()}")

# Con sesgo
b = tf.constant([10.0], dtype=tf.float32)
predicciones_con_sesgo = tf.matmul(features, pesos_modelo) + b
print(f"Predicciones con sesgo:\n{predicciones_con_sesgo.numpy().flatten()}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: tf.matmul — Multiplicación de matrices (features × pesos) para predicción lineal.*

1. 5 productos con 3 features: [precio, descuento, días_sin_stock]
2. y = X @ w + b
3. Con sesgo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: tf.nn.relu — Activación ReLU sobre predicciones de ventas

```python
# Predicciones lineales (pueden ser negativas si el modelo no ha aprendido bien)
pred_lineales = tf.constant([-5.0, 12.0, -3.5, 8.0, 0.0, -10.0, 25.0], dtype=tf.float32)
print(f"Predicciones lineales: {pred_lineales.numpy()}")

# ReLU: max(0, x) — las ventas no pueden ser negativas
pred_ventas = tf.nn.relu(pred_lineales)
print(f"Predicciones con ReLU (ventas >= 0): {pred_ventas.numpy()}")

# Relu6: min(max(0, x), 6) — útil para limitar salidas
pred_ventas_limitadas = tf.nn.relu6(pred_lineales / 5.0)
print(f"Con relu6 (normalizado a [0,6]): {pred_ventas_limitadas.numpy()}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: tf.nn.relu — Activación ReLU sobre predicciones de ventas.*

1. Predicciones lineales (pueden ser negativas si el modelo no ha aprendido bien)
2. ReLU: max(0, x) — las ventas no pueden ser negativas
3. Relu6: min(max(0, x), 6) — útil para limitar salidas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: tf.nn.sigmoid — Activación sigmoide para clasificación binaria (¿se vende?)

```python
# Logits de clasificación: ¿el producto se venderá hoy?
logits = tf.constant([2.0, -1.5, 0.0, 3.2, -0.5, 1.8, -3.0], dtype=tf.float32)

probabilidades = tf.nn.sigmoid(logits)
print(f"Logits: {logits.numpy()}")
print(f"Probabilidades de venta: {probabilidades.numpy()}")

# Decisión: > 0.5 se vende
decision = probabilidades >= 0.5
print(f"Decisión (vende=1): {tf.cast(decision, tf.int32).numpy()}")
print(f"Confianza más baja: {tf.reduce_min(probabilidades).numpy():.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: tf.nn.sigmoid — Activación sigmoide para clasificación binaria (¿se vende?).*

1. Logits de clasificación: ¿el producto se venderá hoy?
2. Decisión: > 0.5 se vende

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: tf.nn.softmax — Distribución de probabilidad entre categorías de productos

```python
# Logits para 5 productos en 3 categorías: [Electrónica, Ropa, Alimentos]
logits_categorias = tf.constant([
    [2.0, 1.0, 0.1],
    [0.5, 2.5, 1.0],
    [3.0, 0.5, 0.8],
    [1.0, 1.0, 2.0],
    [0.1, 0.5, 3.0]
], dtype=tf.float32)

prob_categorias = tf.nn.softmax(logits_categorias, axis=1)
print(f"Probabilidades por categoría:\n{prob_categorias.numpy()}")

categoria_predicha = tf.argmax(prob_categorias, axis=1)
categorias = ['Electrónica', 'Ropa', 'Alimentos']
print(f"Categoría predicha por producto: {[categorias[i] for i in categoria_predicha.numpy()]}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: tf.nn.softmax — Distribución de probabilidad entre categorías de productos.*

1. Logits para 5 productos en 3 categorías: [Electrónica, Ropa, Alimentos]

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: tf.losses.MeanSquaredError — Error cuadrático medio entre predicción y real

```python
mse = tf.losses.MeanSquaredError()

# Predicciones de ventas vs valores reales
ventas_reales = tf.constant([100.0, 200.0, 150.0, 300.0, 250.0], dtype=tf.float32)
ventas_predichas = tf.constant([110.0, 180.0, 160.0, 290.0, 240.0], dtype=tf.float32)

perdida_mse = mse(ventas_reales, ventas_predichas)
print(f"MSE: {perdida_mse.numpy():.4f}")

# Loss con diferentes formas: (batch,) vs (batch, 1)
ventas_reales_col = tf.reshape(ventas_reales, [-1, 1])
ventas_predichas_col = tf.reshape(ventas_predichas, [-1, 1])
perdida_mse_col = mse(ventas_reales_col, ventas_predichas_col)
print(f"MSE (columnas): {perdida_mse_col.numpy():.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: tf.losses.MeanSquaredError — Error cuadrático medio entre predicción y real.*

1. Predicciones de ventas vs valores reales
2. Loss con diferentes formas: (batch,) vs (batch, 1)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: tf.optimizers.Adam — Optimizador Adam con learning_rate

```python
# Configurar optimizador Adam
optimizador = tf.optimizers.Adam(learning_rate=0.01)

# Variable a optimizar
x = tf.Variable(10.0)

# Función objetivo simple: (x - 3)^2
for paso in range(50):
    with tf.GradientTape() as tape:
        perdida = (x - 3.0) ** 2
    gradientes = tape.gradient(perdida, [x])
    optimizador.apply_gradients(zip(gradientes, [x]))

print(f"Valor óptimo de x (debería ser 3): {x.numpy():.4f}")

# Optimizador con learning rate schedule
optimizador_decaimiento = tf.optimizers.Adam(learning_rate=0.001)
print(f"LR inicial: {optimizador_decaimiento.learning_rate.numpy()}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: tf.optimizers.Adam — Optimizador Adam con learning_rate.*

1. Configurar optimizador Adam
2. Variable a optimizar
3. Función objetivo simple: (x - 3)^2
4. Optimizador con learning rate schedule

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: tf.GradientTape — Calcular gradientes automáticamente

```python
# Datos de ventas: 1 feature (precio), 1 target (cantidad vendida)
precios = tf.constant([10.0, 20.0, 30.0, 40.0, 50.0], dtype=tf.float32)
cantidades = tf.constant([100.0, 80.0, 60.0, 40.0, 20.0], dtype=tf.float32)

# Modelo lineal simple: y = w * x + b
w = tf.Variable(0.5, dtype=tf.float32)
b = tf.Variable(10.0, dtype=tf.float32)

with tf.GradientTape() as tape:
    predicciones = w * precios + b
    perdida = tf.reduce_mean((predicciones - cantidades) ** 2)

grad_w, grad_b = tape.gradient(perdida, [w, b])
print(f"Gradiente de w: {grad_w.numpy():.4f}")
print(f"Gradiente de b: {grad_b.numpy():.4f}")
print(f"Pérdida inicial: {perdida.numpy():.4f}")

# Actualizar pesos manualmente
lr = 0.01
w.assign_sub(lr * grad_w)
b.assign_sub(lr * grad_b)
print(f"w después de actualizar: {w.numpy():.4f}")
print(f"b después de actualizar: {b.numpy():.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: tf.GradientTape — Calcular gradientes automáticamente.*

1. Datos de ventas: 1 feature (precio), 1 target (cantidad vendida)
2. Modelo lineal simple: y = w * x + b
3. Actualizar pesos manualmente

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: @tf.function — Compilar función para velocidad en inferencia de ventas

```python
@tf.function
def predecir_ventas(precios, descuentos, w, b):
    precio_efectivo = precios * (1.0 - descuentos)
    return w * precio_efectivo + b

# Sin @tf.function (modo eager)
def predecir_ventas_eager(precios, descuentos, w, b):
    precio_efectivo = precios * (1.0 - descuentos)
    return w * precio_efectivo + b

precios_t = tf.constant([100.0, 200.0, 150.0], dtype=tf.float32)
descuentos_t = tf.constant([0.1, 0.05, 0.2], dtype=tf.float32)
w_t = tf.constant(0.8, dtype=tf.float32)
b_t = tf.constant(5.0, dtype=tf.float32)

resultado = predecir_ventas(precios_t, descuentos_t, w_t, b_t)
print(f"Predicción con tf.function: {resultado.numpy()}")

# Verificar el grafo compilado
print(f"¿Función compilada?: {predecir_ventas.function is not None}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: @tf.function — Compilar función para velocidad en inferencia de ventas.*

1. Sin @tf.function (modo eager)
2. Verificar el grafo compilado

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: tf.data.Dataset — Crear dataset desde tensores de ventas

```python
# Datos de 10 transacciones: [precio, descuento, cantidad]
features_ds = tf.constant([
    [150.0, 0.10, 2.0], [230.0, 0.05, 1.0], [89.0, 0.20, 5.0],
    [420.0, 0.15, 1.0], [310.0, 0.08, 3.0], [175.0, 0.12, 2.0],
    [520.0, 0.03, 1.0], [67.0, 0.25, 10.0], [890.0, 0.00, 1.0],
    [120.0, 0.18, 4.0]
], dtype=tf.float32)

targets_ds = tf.constant([300.0, 230.0, 445.0, 420.0, 930.0,
                          350.0, 520.0, 670.0, 890.0, 480.0], dtype=tf.float32)

dataset = tf.data.Dataset.from_tensor_slices((features_ds, targets_ds))
for i, (f, t) in enumerate(dataset.take(3)):
    print(f"Transacción {i+1}: features={f.numpy()}, target={t.numpy()}")

print(f"Número total de elementos: {len(list(dataset))}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: tf.data.Dataset — Crear dataset desde tensores de ventas.*

1. Datos de 10 transacciones: [precio, descuento, cantidad]

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: tf.data.Dataset — Pipeline con batch, shuffle y prefetch

```python
# Crear un dataset de 1000 transacciones sintéticas
n_trans = 1000
X_sintetico = tf.random.normal([n_trans, 5], dtype=tf.float32)  # 5 features
y_sintetico = tf.random.uniform([n_trans, 1], 0, 100, dtype=tf.float32)

dataset_pipeline = tf.data.Dataset.from_tensor_slices((X_sintetico, y_sintetico))
dataset_pipeline = dataset_pipeline.shuffle(buffer_size=200)
dataset_pipeline = dataset_pipeline.batch(batch_size=32)
dataset_pipeline = dataset_pipeline.prefetch(buffer_size=tf.data.AUTOTUNE)

for batch_x, batch_y in dataset_pipeline.take(3):
    print(f"Batch: X shape {batch_x.shape}, y shape {batch_y.shape}")

print(f"Total de batches: {len(list(dataset_pipeline))}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: tf.data.Dataset — Pipeline con batch, shuffle y prefetch.*

1. Crear un dataset de 1000 transacciones sintéticas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 16: tf.math — Operaciones matemáticas (exp, log, sqrt) sobre precios y cantidades

```python
precios_base = tf.constant([100.0, 200.0, 300.0, 400.0, 500.0], dtype=tf.float32)

# Transformaciones comunes en ventas
log_precios = tf.math.log(precios_base + 1.0)  # log(1+x) para evitar log(0)
sqrt_precios = tf.math.sqrt(precios_base)
exp_precios = tf.math.exp(precios_base / 500.0)  # Normalizado

print(f"Precios originales: {precios_base.numpy()}")
print(f"log(1+precios): {log_precios.numpy()}")
print(f"sqrt(precios): {sqrt_precios.numpy()}")
print(f"exp(precios/500): {exp_precios.numpy()}")

# Crecimiento porcentual
anterior = tf.constant([100.0, 150.0, 200.0, 250.0, 300.0])
actual = tf.constant([110.0, 145.0, 220.0, 240.0, 310.0])
crecimiento = tf.math.divide(actual - anterior, anterior) * 100.0
print(f"Crecimiento %: {crecimiento.numpy()}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: tf.math — Operaciones matemáticas (exp, log, sqrt) sobre precios y cantidades.*

1. Transformaciones comunes en ventas
2. Crecimiento porcentual

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 17: tf.convert_to_tensor — Convertir numpy array a tensor de TensorFlow

```python
# Datos de compras desde numpy
np_precios = np.array([15.50, 23.00, 8.90, 42.00, 31.00], dtype=np.float32)
np_cantidades = np.array([10, 5, 20, 3, 8], dtype=np.int32)

tensor_precios = tf.convert_to_tensor(np_precios, dtype=tf.float32)
tensor_cantidades = tf.cast(
    tf.convert_to_tensor(np_cantidades, dtype=tf.int32),
    dtype=tf.float32
)

montos = tensor_precios * tensor_cantidades
print(f"Precios: {tensor_precios.numpy()}")
print(f"Cantidades: {tensor_cantidades.numpy()}")
print(f"Monto total por producto: {montos.numpy()}")
print(f"Monto total compra: ${tf.reduce_sum(montos).numpy():.2f}")

# También funciona con listas Python
lista_precios = [10.0, 20.0, 30.0]
tensor_lista = tf.convert_to_tensor(lista_precios, dtype=tf.float32)
print(f"Desde lista: {tensor_lista.numpy()}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: tf.convert_to_tensor — Convertir numpy array a tensor de TensorFlow.*

1. Datos de compras desde numpy
2. También funciona con listas Python

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 18: Integrador — Regresión lineal con TensorFlow desde cero para predecir ventas

```python
# Datos sintéticos: precio -> cantidad vendida
tf.random.set_seed(42)
n_datos = 200
precios_reales = tf.random.uniform([n_datos, 1], minval=10.0, maxval=100.0)
w_real = 2.5
b_real = 50.0
ruido = tf.random.normal([n_datos, 1], mean=0.0, stddev=15.0)
cantidades_reales = w_real * precios_reales + b_real + ruido

# Modelo lineal
w = tf.Variable(tf.random.normal([1, 1], stddev=0.5))
b = tf.Variable(tf.zeros([1]))
optimizer = tf.optimizers.Adam(learning_rate=0.01)

def modelo(X):
    return tf.matmul(X, w) + b

def loss(y_pred, y_true):
    return tf.reduce_mean((y_pred - y_true) ** 2)

# Entrenamiento
epochs = 500
for ep in range(epochs):
    with tf.GradientTape() as tape:
        preds = modelo(precios_reales)
        perdida = loss(preds, cantidades_reales)
    grad_w, grad_b = tape.gradient(perdida, [w, b])
    optimizer.apply_gradients(zip([grad_w, grad_b], [w, b]))

print(f"w aprendido: {w.numpy().flatten()[0]:.4f} (real: {w_real})")
print(f"b aprendido: {b.numpy().flatten()[0]:.4f} (real: {b_real})")
print(f"Pérdida final: {perdida.numpy():.4f}")

# Predicción para nuevos precios
nuevos_precios = tf.constant([[45.0], [80.0], [25.0]])
predicciones = modelo(nuevos_precios)
print(f"Predicciones para precios {nuevos_precios.numpy().flatten()}: {predicciones.numpy().flatten()}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador — Regresión lineal con TensorFlow desde cero para predecir ventas.*

1. Datos sintéticos: precio -> cantidad vendida
2. Modelo lineal
3. Entrenamiento
4. Predicción para nuevos precios

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejercicios

1. **Tensor de ingresos por sucursal**: Crea un tensor `tf.constant` con los ingresos semanales de 5 sucursales (7 días). Calcula el promedio por sucursal y el día de mayor ingreso global.

2. **Gradiente de función de costo**: Usa `tf.GradientTape` para calcular el gradiente de la función `L(w,b) = (w*X + b - y)^2` con X=[10,20,30], y=[50,45,40]. Imprime los gradientes y actualiza w,b manualmente.

3. **Pipeline de datos**: Crea un `tf.data.Dataset` con 500 transacciones (features: precio, descuento, cantidad). Aplica shuffle(100), batch(16), prefetch(AUTOTUNE). Itera 2 batches y muestra sus formas.

4. **Regresión con SGD**: Implementa regresión lineal con `tf.optimizers.SGD(learning_rate=0.001, momentum=0.9)` en lugar de Adam usando los datos del ejemplo 18. Compara la convergencia.

5. **Clasificación binaria con sigmoid**: Genera 200 datos sintéticos con 2 features (monto, frecuencia) y target binario (compra/no compra). Define un modelo lineal y entrena usando `BinaryCrossentropy` como loss.

6. **Softmax para categorías**: Crea un tensor 10×4 de logits (10 productos, 4 categorías). Aplica softmax y determina la categoría más probable para cada producto. Usa `tf.argmax`.

7. **Compilación con @tf.function**: Define una función que calcule el precio con IVA (precio * 1.21) y descuento escalonado (>100 → 10%, >500 → 20%). Decórala con `@tf.function` y mide su velocidad contra la versión eager con `%timeit`.

8. **Tensor de inventario**: Crea un tensor 3D (10 productos × 4 semanas × 6 días) con stocks diarios. Calcula usando `tf.reduce_mean`, `tf.reduce_sum` y `tf.reduce_max`: stock promedio por producto, stock total por semana, stock máximo por día.

---

## Resumen

Hemos cubierto las operaciones fundamentales de TensorFlow para análisis de ventas, compras e inventarios:

- **Creación de tensores**: `tf.constant`, `tf.Variable`, `tf.convert_to_tensor` para representar datos de negocio (precios, cantidades, montos).
- **Operaciones de reducción**: `tf.reduce_mean`, `tf.reduce_sum`, `tf.reduce_max/min` para agregar métricas por sucursal, producto o período.
- **Álgebra lineal**: `tf.matmul` para predicciones lineales (features × pesos).
- **Funciones de activación**: `tf.nn.relu` (ventas ≥ 0), `tf.nn.sigmoid` (clasificación binaria), `tf.nn.softmax` (categorías múltiples).
- **Diferenciación automática**: `tf.GradientTape` para calcular gradientes de la función de pérdida.
- **Optimización**: `tf.optimizers.Adam` y `SGD` para actualizar pesos.
- **Pipeline de datos**: `tf.data.Dataset.from_tensor_slices` con `shuffle`, `batch`, `prefetch` para alimentar modelos eficientemente.
- **Compilación**: `@tf.function` para acelerar la inferencia convirtiendo Python a grafos de TensorFlow.

Estas herramientas son la base para construir modelos de deep learning aplicados a problemas reales de negocio: predicción de demanda, clasificación de productos, optimización de inventarios y análisis de compras.
