# A19: tf.data — Pipeline de Datos Eficiente para Ventas, Compras e Inventarios

## Introducción Teórica

**tf.data** es la API de TensorFlow para construir pipelines de datos eficientes, escalables y paralelizables. Permite cargar, transformar y alimentar datos a modelos sin cuellos de botella de E/S.

### Componentes principales:

1. **Fuentes de datos**:
   - `from_tensor_slices`: Crea dataset desde tensores/arrays en memoria.
   - `from_generator`: Crea dataset desde un generador Python (útil para datos que no caben en RAM).
   - `from_tensors`: Crea dataset desde uno o más tensores como un solo elemento.
2. **Transformaciones**:
   - `map(func, num_parallel_calls)`: Aplica función a cada elemento. Paralelizable.
   - `filter(predicate)`: Filtra elementos según condición.
   - `batch(batch_size, drop_remainder)`: Agrupa elementos en lotes.
   - `shuffle(buffer_size, seed, reshuffle_each_iteration)`: Mezcla aleatoriamente.
   - `repeat(count)`: Repite el dataset N veces.
   - `take(count)`: Toma los primeros N elementos.
   - `skip(count)`: Salta los primeros N elementos.
   - `prefetch(buffer_size)`: Precarga lotes en background.
   - `cache(filename)`: Cachea el dataset en memoria o archivo.
   - `interleave(map_func, cycle_length, block_length, num_parallel_calls)`: Mezcla múltiples datasets.
   - `window(size, shift, stride, drop_remainder)`: Ventanas deslizantes.
   - `padded_batch(batch_size, padded_shapes, padding_values)`: Lotes con padding.
   - `reduce(initial_state, reduce_func)`: Reduce a un solo valor.
3. **TFRecord**: Formato binario eficiente de TensorFlow.
   - `TFRecordWriter`: Escribe datos en formato TFRecord.
   - `TFRecordDataset`: Lee datos desde TFRecord.
   - `parse_single_example`: Parsea un ejemplo individual.
   - `feature_description`: Describe el esquema de los features.

### Aplicaciones en negocio:

- **Ventas**: Pipeline de millones de transacciones con normalización, filtrado y batching eficiente.
- **Compras**: Procesar archivos TFRecord de órdenes de compra; windowing para series temporales.
- **Inventarios**: Shuffle y cache de datos de stock; interleave de múltiples fuentes (sucursales, almacenes).

---

## Ejemplos

### Ejemplo 1: from_tensor_slices — Crear dataset desde arrays de precios

```python
import tensorflow as tf
import numpy as np

precios = np.array([150.0, 230.0, 89.0, 420.0, 310.0, 175.0, 520.0, 67.0, 890.0, 120.0], dtype=np.float32)
cantidades = np.array([2, 1, 5, 1, 3, 2, 1, 10, 1, 4], dtype=np.int32)
montos = precios * cantidades

dataset = tf.data.Dataset.from_tensor_slices((precios, cantidades, montos))
print("Elementos del dataset:")
for i, (p, c, m) in enumerate(dataset):
    print(f"  {i+1}: precio={p.numpy():.1f}, cant={c.numpy()}, monto={m.numpy():.1f}")

print(f"Cardinalidad: {dataset.cardinality().numpy()}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: from_tensor_slices — Crear dataset desde arrays de precios.*

1. `import tensorflow as tf` — Importa las librerías necesarias para el análisis.
2. `import numpy as np` — Importa las librerías necesarias para el análisis.
3. `print("Elementos del dataset:")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: batch(32) — Agrupar en lotes de 32 para entrenamiento

```python
n = 200
X = np.random.rand(n, 5).astype(np.float32)
y = np.random.rand(n, 1).astype(np.float32)

dataset_batch = tf.data.Dataset.from_tensor_slices((X, y))
dataset_batch = dataset_batch.batch(32)

total_batches = 0
for batch_x, batch_y in dataset_batch:
    total_batches += 1
    print(f"Batch {total_batches}: X shape={batch_x.shape}, y shape={batch_y.shape}")

print(f"Total batches (n=200, batch=32): {total_batches}")
print(f"Último batch tiene {batch_x.shape[0]} elementos (los últimos 8)")

# con drop_remainder=True
dataset_drop = tf.data.Dataset.from_tensor_slices((X, y)).batch(32, drop_remainder=True)
print(f"Batches con drop_remainder=True: {len(list(dataset_drop))} (200/32=6.25 → 6)")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: batch(32) — Agrupar en lotes de 32 para entrenamiento.*

1. con drop_remainder=True

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: shuffle(1000) — Mezclar datos de transacciones

```python
transacciones = tf.range(1, 21, dtype=tf.int32)  # IDs de transacción 1..20
print(f"Orden original: {transacciones.numpy()}")

dataset_shuffled = tf.data.Dataset.from_tensor_slices(transacciones)
dataset_shuffled = dataset_shuffled.shuffle(buffer_size=10, seed=42, reshuffle_each_iteration=True)

print(f"Orden después de shuffle:")
for i, t in enumerate(dataset_shuffled):
    print(f"  {t.numpy()}", end="")
print()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: shuffle(1000) — Mezclar datos de transacciones.*

1. `print(f"Orden original: {transacciones.numpy()}")` — Muestra el resultado por pantalla.
2. `print(f"Orden después de shuffle:")` — Muestra el resultado por pantalla.
3. `print(f"  {t.numpy()}", end="")` — Muestra el resultado por pantalla.
4. `print()` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: prefetch(tf.data.AUTOTUNE) — Precargar siguiente lote durante entrenamiento

```python
n = 500
X_pf = np.random.rand(n, 10).astype(np.float32)
y_pf = np.random.rand(n, 1).astype(np.float32)

dataset_pf = tf.data.Dataset.from_tensor_slices((X_pf, y_pf))
dataset_pf = dataset_pf.batch(32).prefetch(tf.data.AUTOTUNE)

print(f"Dataset con prefetch configurado")
print(f"Prefetch buffer: {dataset_pf._options().deterministic}")

# Simular entrenamiento
for step, (x_b, y_b) in enumerate(dataset_pf.take(3)):
    print(f"Batch {step+1}: X={x_b.shape}, y={y_b.shape} (precargado mientras se procesa)")

# Sin prefetch vs con prefetch
dataset_no_pf = tf.data.Dataset.from_tensor_slices((X_pf, y_pf)).batch(32)
dataset_pf2 = tf.data.Dataset.from_tensor_slices((X_pf, y_pf)).batch(32).prefetch(tf.data.AUTOTUNE)
print(f"Sin prefetch: {len(list(dataset_no_pf))} batches")
print(f"Con prefetch: {len(list(dataset_pf2))} batches (igual número, pero faster)")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: prefetch(tf.data.AUTOTUNE) — Precargar siguiente lote durante entrenamiento.*

1. Simular entrenamiento
2. Sin prefetch vs con prefetch

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: map — Normalizar precios con función personalizada

```python
def normalizar_precio(precio, cantidad, monto):
    precio_norm = (precio - tf.reduce_mean(precio)) / (tf.math.reduce_std(precio) + 1e-8)
    return precio_norm, cantidad, monto

precios_raw = tf.random.uniform([100], minval=10.0, maxval=1000.0, dtype=tf.float32)
cantidades_raw = tf.random.uniform([100], minval=1, maxval=50, dtype=tf.int32)
montos_raw = precios_raw * tf.cast(cantidades_raw, tf.float32)

dataset_map = tf.data.Dataset.from_tensor_slices((precios_raw, cantidades_raw, montos_raw))
dataset_map = dataset_map.map(normalizar_precio)

print("Primeros 5 elementos normalizados:")
for p, c, m in dataset_map.take(5):
    print(f"  precio_norm={p.numpy():.4f}, cant={c.numpy()}, monto={m.numpy():.1f}")

print(f"Media de precios normalizados (debe ser ~0): {tf.reduce_mean(list(dataset_map.map(lambda p,c,m: p))):.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: map — Normalizar precios con función personalizada.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: map(num_parallel_calls=4) — Paralelizar transformación de datos de ventas

```python
def transformar_venta(features, target):
    precio, descuento, cantidad = features[0], features[1], features[2]
    precio_final = precio * (1.0 - descuento)
    features_transformadas = tf.stack([precio, descuento, cantidad, precio_final])
    return features_transformadas, target

n_ventas = 1000
X_ventas = np.column_stack([
    np.random.uniform(10, 500, n_ventas),
    np.random.uniform(0, 0.3, n_ventas),
    np.random.randint(1, 20, n_ventas)
]).astype(np.float32)
y_ventas = np.random.rand(n_ventas, 1).astype(np.float32)

dataset_par = tf.data.Dataset.from_tensor_slices((X_ventas, y_ventas))
dataset_par = dataset_par.map(transformar_venta, num_parallel_calls=tf.data.AUTOTUNE)

for feat, tgt in dataset_par.take(3):
    print(f"Features (precio, desc, cant, precio_final): {feat.numpy()}, target={tgt.numpy()[0]:.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: map(num_parallel_calls=4) — Paralelizar transformación de datos de ventas, compras o inventarios.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: cache — Cachear dataset en memoria después de procesar

```python
dataset_cache = tf.data.Dataset.from_tensor_slices((X_ventas, y_ventas))
dataset_cache = dataset_cache.map(lambda x, y: (x * 2.0, y), num_parallel_calls=4)
dataset_cache = dataset_cache.cache()  # Cachea en memoria después del map
dataset_cache = dataset_cache.shuffle(100).batch(32).prefetch(tf.data.AUTOTUNE)

print(f"Dataset cacheado en memoria")
print(f"Primer batch: X shape={next(iter(dataset_cache))[0].shape}")

# Segunda iteración (usa cache, no re-ejecuta map)
for batch_x, batch_y in dataset_cache.take(2):
    pass
print("Segunda iteración desde cache (más rápida)")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: cache — Cachear dataset en memoria después de procesar.*

1. Segunda iteración (usa cache, no re-ejecuta map)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: repeat — Repetir dataset para múltiples epochs

```python
datos_pequenos = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
dataset_repeat = datos_pequenos.repeat(3)  # 3 epochs

elementos = list(dataset_repeat.as_numpy_iterator())
print(f"Elementos con repeat(3): {elementos}")
print(f"Cada epoch: {elementos[:5]}, {elementos[5:10]}, {elementos[10:]}")

# Combinado con batch
dataset_epoch = tf.data.Dataset.from_tensor_slices(np.random.rand(100, 3).astype(np.float32))
dataset_epoch = dataset_epoch.repeat(3).batch(10)
total_batches = len(list(dataset_epoch))
print(f"100 elementos × 3 epochs ÷ batch=10 = {total_batches} batches")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: repeat — Repetir dataset para múltiples epochs.*

1. Combinado con batch

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: filter — Filtrar productos con precio > 100

```python
productos = tf.constant([
    [150.0, 0.10, 2.0],
    [230.0, 0.05, 1.0],
    [89.0,  0.20, 5.0],
    [420.0, 0.15, 1.0],
    [310.0, 0.08, 3.0],
    [45.0,  0.30, 10.0],
    [520.0, 0.03, 1.0],
    [67.0,  0.25, 10.0],
    [890.0, 0.00, 1.0],
    [120.0, 0.18, 4.0]
], dtype=tf.float32)

def precio_mayor_100(producto):
    return producto[0] > 100.0

dataset_filtrado = tf.data.Dataset.from_tensor_slices(productos)
dataset_filtrado = dataset_filtrado.filter(precio_mayor_100)

print("Productos con precio > 100:")
for p in dataset_filtrado:
    print(f"  precio={p[0].numpy():.1f}, desc={p[1].numpy():.2f}, cant={int(p[2].numpy())}")

print(f"Original: {len(productos.numpy())}, Filtrados: {len(list(dataset_filtrado))}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: filter — Filtrar productos con precio > 100.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: take(5) — Tomar primeros 5 elementos de un dataset de ventas

```python
dataset_grande = tf.data.Dataset.from_tensor_slices(tf.range(1, 1001, dtype=tf.int32))
dataset_sample = dataset_grande.take(5)

print("Primeros 5 elementos:")
for elem in dataset_sample:
    print(f"  {elem.numpy()}")

# Útil para inspección rápida
dataset_ventas = tf.data.Dataset.from_tensor_slices({
    'producto_id': tf.range(1, 1001),
    'precio': tf.random.uniform([1000], 10, 500, dtype=tf.int32),
    'cantidad': tf.random.uniform([1000], 1, 20, dtype=tf.int32)
})
for batch in dataset_ventas.batch(5).take(1):
    print(f"Muestra de 5 ventas:\n  IDs: {batch['producto_id'].numpy()}")
    print(f"  Precios: {batch['precio'].numpy()}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: take(5) — Tomar primeros 5 elementos de un dataset de ventas.*

1. Útil para inspección rápida

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: skip(10) — Saltar primeros 10 elementos

```python
dataset_skip = tf.data.Dataset.from_tensor_slices(tf.range(1, 21, dtype=tf.int32))
dataset_saltado = dataset_skip.skip(10)

print(f"Original: {list(dataset_skip.as_numpy_iterator())}")
print(f"Después de skip(10): {list(dataset_saltado.as_numpy_iterator())}")

# Combinado: skip(5).take(3) → elementos 6,7,8
dataset_comb = dataset_skip.skip(5).take(3)
print(f"skip(5).take(3): {list(dataset_comb.as_numpy_iterator())}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: skip(10) — Saltar primeros 10 elementos.*

1. Combinado: skip(5).take(3) → elementos 6,7,8

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: interleave — Mezclar múltiples archivos de datos de sucursales

```python
# Simular datos de 3 sucursales
sucursal_a = tf.data.Dataset.range(1, 11).map(lambda x: f"A_{x}")
sucursal_b = tf.data.Dataset.range(1, 11).map(lambda x: f"B_{x}")
sucursal_c = tf.data.Dataset.range(1, 11).map(lambda x: f"C_{x}")

# interleave: mezcla elementos de las 3 fuentes
dataset_interleaved = tf.data.Dataset.from_tensor_slices([sucursal_a, sucursal_b, sucursal_c])
dataset_interleaved = dataset_interleaved.interleave(
    lambda x: x,
    cycle_length=3,
    block_length=2,
    num_parallel_calls=tf.data.AUTOTUNE,
    deterministic=True
)

resultado = [elem.numpy().decode() for elem in dataset_interleaved]
print(f"Interleave (cycle=3, block=2): {resultado}")
print(f"Patrón: A_1,A_2,B_1,B_2,C_1,C_2,A_3,A_4,...")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: interleave — Mezclar múltiples archivos de datos de sucursales.*

1. Simular datos de 3 sucursales
2. interleave: mezcla elementos de las 3 fuentes

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: padded_batch — Lotes con padding para secuencias de diferente longitud

```python
# Secuencias de compras de diferente longitud
compras = [
    [150.0, 230.0, 89.0],                    # 3 items
    [420.0, 310.0],                            # 2 items
    [175.0, 520.0, 67.0, 890.0, 120.0],      # 5 items
    [300.0, 450.0],                            # 2 items
    [80.0, 95.0, 110.0, 200.0]                # 4 items
]

dataset_pad = tf.data.Dataset.from_tensor_slices(compras)
dataset_pad = dataset_pad.padded_batch(
    batch_size=3,
    padded_shapes=[None],  # pad a la longitud máxima del batch
    padding_values=0.0,
    drop_remainder=False
)

for i, batch in enumerate(dataset_pad):
    print(f"Batch {i+1} (padded):\n{batch.numpy()}")

# Padding con valores específicos
dataset_pad2 = tf.data.Dataset.from_tensor_slices(compras)
dataset_pad2 = dataset_pad2.padded_batch(
    batch_size=3,
    padded_shapes=[5],    # pad a longitud 5
    padding_values=-1.0,  # valor de padding = -1
)
for i, batch in enumerate(dataset_pad2):
    print(f"Batch {i+1} (pad=5, value=-1):\n{batch.numpy()}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: padded_batch — Lotes con padding para secuencias de diferente longitud.*

1. Secuencias de compras de diferente longitud
2. Padding con valores específicos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: from_generator — Crear dataset desde generador Python (lectura de archivos)

```python
def generador_ventas():
    """Genera lotes de ventas simuladas (simula lectura de archivo)"""
    productos = ['Laptop', 'Mouse', 'Teclado', 'Monitor', 'Audífonos']
    for i in range(20):
        yield {
            'producto': np.random.choice(productos),
            'precio': np.random.uniform(10, 1000),
            'cantidad': np.random.randint(1, 10)
        }

# Crear dataset desde generador
dataset_gen = tf.data.Dataset.from_generator(
    generador_ventas,
    output_types={'producto': tf.string, 'precio': tf.float32, 'cantidad': tf.int32},
    output_shapes={'producto': (), 'precio': (), 'cantidad': ()}
)

print("Primeras 3 ventas del generador:")
for i, venta in enumerate(dataset_gen.take(3)):
    print(f"  {venta['producto'].numpy().decode()}: ${venta['precio'].numpy():.2f} × {venta['cantidad'].numpy()}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: from_generator — Crear dataset desde generador Python (lectura de archivos).*

1. Crear dataset desde generador

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: window — Crear ventanas deslizantes para series temporales de ventas

```python
# Serie temporal de ventas diarias (60 días)
ventas_diarias = tf.random.uniform([60], minval=100, maxval=500, dtype=tf.int32)
dataset_ts = tf.data.Dataset.from_tensor_slices(ventas_diarias)

# Ventanas de 7 días (una semana) con desplazamiento de 1 día
dataset_window = dataset_ts.window(size=7, shift=1, stride=1, drop_remainder=True)

print("Primeras 3 ventanas de 7 días:")
for i, window in enumerate(dataset_window.take(3)):
    window_list = list(window.as_numpy_iterator())
    print(f"  Semana {i+1}: {window_list}")

# Convertir ventanas a batches para entrenamiento
dataset_window_batch = dataset_ts.window(7, shift=1, drop_remainder=True)
dataset_window_batch = dataset_window_batch.flat_map(lambda w: w.batch(7))
dataset_window_batch = dataset_window_batch.map(lambda w: (w[:-1], w[-1]))  # X=6 días, y=1 día

for X_w, y_w in dataset_window_batch.take(3):
    print(f"X (6 días): {X_w.numpy()}, y (día 7): {y_w.numpy()}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: window — Crear ventanas deslizantes para series temporales de ventas.*

1. Serie temporal de ventas diarias (60 días)
2. Ventanas de 7 días (una semana) con desplazamiento de 1 día
3. Convertir ventanas a batches para entrenamiento

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 16: TFRecord — Escribir datos de ventas en formato TFRecord

```python
import tempfile, os

ruta_tfr = os.path.join(tempfile.gettempdir(), 'ventas.tfrecord')

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# 10 transacciones de ejemplo
with tf.io.TFRecordWriter(ruta_tfr) as writer:
    for i in range(10):
        producto = f"Producto_{i}"
        precio = np.random.uniform(10, 500)
        cantidad = np.random.randint(1, 20)
        monto = precio * cantidad

        feature = {
            'producto': _bytes_feature(producto),
            'precio': _float_feature(precio),
            'cantidad': _int64_feature(cantidad),
            'monto': _float_feature(monto)
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

print(f"TFRecord escrito en: {ruta_tfr}")
print(f"Tamaño: {os.path.getsize(ruta_tfr)} bytes")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: TFRecord — Escribir datos de ventas, compras o inventarios en formato TFRecord.*

1. 10 transacciones de ejemplo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 17: TFRecord — Leer y parsear TFRecord de ventas

```python
# Leer TFRecord del ejemplo anterior
dataset_tfr = tf.data.TFRecordDataset([ruta_tfr])

# Definir descripción de features
feature_description = {
    'producto': tf.io.FixedLenFeature([], tf.string),
    'precio': tf.io.FixedLenFeature([], tf.float32),
    'cantidad': tf.io.FixedLenFeature([], tf.int64),
    'monto': tf.io.FixedLenFeature([], tf.float32)
}

def parsear_ejemplo(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)

dataset_parsed = dataset_tfr.map(parsear_ejemplo, num_parallel_calls=tf.data.AUTOTUNE)

print("Datos leídos desde TFRecord:")
for i, elem in enumerate(dataset_parsed):
    print(f"  {i+1}: {elem['producto'].numpy().decode()}, "
          f"precio={elem['precio'].numpy():.1f}, "
          f"cant={elem['cantidad'].numpy()}, "
          f"monto={elem['monto'].numpy():.1f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: TFRecord — Leer y parsear TFRecord de ventas.*

1. Leer TFRecord del ejemplo anterior
2. Definir descripción de features

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 18: Integrador — Pipeline completo de datos de ventas con todas las transformaciones

```python
# Generar datos sintéticos de 5000 transacciones
np.random.seed(42)
n = 5000
features = np.column_stack([
    np.random.uniform(10, 500, n),      # precio
    np.random.uniform(0, 0.3, n),       # descuento
    np.random.randint(1, 365, n),       # días_ultima_compra
    np.random.randint(0, 5, n),         # categoría (0-4)
    np.random.uniform(0.1, 0.9, n),     # margen
]).astype(np.float32)
targets = np.random.rand(n, 1).astype(np.float32)

# Pipeline completo
def preprocess(features, target):
    precio, desc, dias, cat, margen = (
        features[0], features[1], features[2], features[3], features[4]
    )
    # Normalizar precio
    precio_norm = (precio - 255.0) / 141.0
    # Aplicar descuento
    precio_final = precio_norm * (1.0 - desc)
    # One-hot encoding de categoría (simplificado)
    cat_one_hot = tf.one_hot(tf.cast(cat, tf.int32), depth=5)
    # Stack final
    features_proc = tf.concat([
        [precio_norm, desc, dias / 365.0, margen, precio_final],
        cat_one_hot
    ], axis=0)
    return features_proc, target

def filtrar_margen_positivo(features, target):
    return features[3] > 0.1

pipeline_completo = (
    tf.data.Dataset.from_tensor_slices((features, targets))
    .shuffle(buffer_size=1000, seed=42, reshuffle_each_iteration=True)
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()
    .filter(filtrar_margen_positivo)
    .batch(64, drop_remainder=True)
    .prefetch(tf.data.AUTOTUNE)
)

print("=== PIPELINE COMPLETO ===")
print(f"Cardinalidad: {pipeline_completo.cardinality().numpy()}")

total_ejemplos = 0
for batch_x, batch_y in pipeline_completo.take(5):
    total_ejemplos += batch_x.shape[0]
    print(f"Batch: X={batch_x.shape} (features={batch_x.shape[1]}), y={batch_y.shape}")

print(f"Total ejemplos en 5 batches: {total_ejemplos}")
print(f"Pipeline ready para entrenamiento: {len(list(pipeline_completo))} batches total")

# Integración con modelo
modelo_pipeline = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])
modelo_pipeline.compile(optimizer='adam', loss='mse', metrics=['mae'])

hist_pipe = modelo_pipeline.fit(
    pipeline_completo,
    epochs=5,
    verbose=0,
    steps_per_epoch=20
)
print(f"Entrenamiento con pipeline: MAE final = {hist_pipe.history['mae'][-1]:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador — Pipeline completo de datos de ventas, compras o inventarios con todas las transformaciones.*

1. Generar datos sintéticos de 5000 transacciones
2. Pipeline completo
3. Normalizar precio
4. Aplicar descuento
5. One-hot encoding de categoría (simplificado)
6. Stack final
7. Integración con modelo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejercicios

1. **Pipeline de compras**: Crea un pipeline que cargue 3000 órdenes de compra desde arrays numpy (5 features: precio_unitario, cantidad, costo_envío, días_entrega, confianza_proveedor). Aplica shuffle(500), batch(128), prefetch(AUTOTUNE). Verifica que los batches tengan el shape correcto.

2. **Normalización con map**: Implementa una función map que normalice 4 features de ventas (precio, descuento, margen, costo) usando z-score: (x - mean) / std. Calcula la media y std de cada feature antes de aplicar el map.

3. **Filtrado de productos no rentables**: Usa filter para eliminar transacciones con margen < 0.05. Genera 2000 transacciones sintéticas y muestra cuántas pasan el filtro.

4. **Ventanas temporales para forecasting**: Crea un dataset con 365 días de ventas. Usa window(size=30, shift=1) para crear ventanas de 30 días. Divide cada ventana en X (29 días) e y (día 30). Formatea para entrenamiento supervisado.

5. **Pipeline con cache y repeat**: Diseña un pipeline que: cargue 1000 muestras → map (normalización) → cache → shuffle(200) → batch(32) → repeat(10). Explica por qué el cache va antes del shuffle. Verifica el número total de batches.

6. **TFRecord de inventarios**: Escribe 500 registros de productos en formato TFRecord con los campos: producto_id (int64), nombre (bytes), stock_actual (int64), precio (float), categoría (int64). Luego lee y parsea 3 registros.

7. **Interleave de sucursales**: Simula 4 sucursales, cada una con 15 transacciones diarias. Usa interleave con cycle_length=2, block_length=3 para mezclar las transacciones. Imprime el orden resultante.

8. **Pipeline integrador completo**: Construye un pipeline end-to-end que: (a) genere 10000 muestras con 8 features mezclando numéricas y categóricas, (b) normalice con map paralelo, (c) filtre outliers (precio entre 10 y 1000), (d) aplique one-hot encoding a 3 categorías, (e) cachee, (f) shuffle, (g) batch de 128, (h) prefetch. Entrena un modelo Sequential con este pipeline durante 10 epochs y reporta la pérdida.

---

## Resumen

La API tf.data es fundamental para construir pipelines de datos eficientes y escalables en proyectos de deep learning aplicados a ventas, compras e inventarios:

- **Creación de datasets**: `from_tensor_slices` para datos en memoria, `from_generator` para datos que no caben en RAM, `TFRecordDataset` para formato binario.
- **Transformaciones**: `map` (paralelizable con `num_parallel_calls`), `filter`, `batch`, `shuffle`, `repeat`.
- **Optimización**: `prefetch(buffer_size=tf.data.AUTOTUNE)` oculta la latencia de E/S; `cache()` evita reprocesar transformaciones costosas.
- **Series temporales**: `window` para crear ventanas deslizantes, `padded_batch` para lotes con secuencias de diferente longitud.
- **Múltiples fuentes**: `interleave` para mezclar datos de múltiples sucursales, archivos o categorías.
- **TFRecord**: Formato binario compacto y rápido para almacenar grandes volúmenes de datos estructurados.
- **Pipeline completo**: shuffle → map (paralelo) → cache → filter → batch → prefetch es el orden recomendado para máxima eficiencia.

Un pipeline bien construido elimina los cuellos de botella de E/S y permite que la GPU o TPU se mantenga ocupada durante el entrenamiento, maximizando el throughput.
