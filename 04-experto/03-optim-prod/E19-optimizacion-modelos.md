# E19: Optimización de Modelos — Pruning, Quantization, Distillation y Deploy

## Objetivo
Dominar técnicas de optimización de modelos ML (pruning, cuantización, destilación, TensorRT, ONNX, TF Lite) para reducir tamaño y latencia en producción de ventas/compras/inventarios.

---

## 1. Fundamentos Teóricos

### 1.1 Weight Pruning
Elimina pesos con magnitud cercana a cero. Tipos:
- **Magnitude-based**: Podar pesos individuales con |w| < threshold
- **Polynomial decay**: Sparsity crece durante entrenamiento: s_t = s_f + (s_i - s_f) * (1 - t/T)^3
- **Constant sparsity**: Sparsity fija desde el inicio

### 1.2 Unit Pruning
Elimina neuronas completas (todas sus conexiones) cuando todas sus salidas son cercanas a cero.

### 1.3 L1 Unstructured Pruning
Aplica regularización L1 durante entrenamiento para forzar pesos pequeños a cero, luego los poda.

### 1.4 Quantization
Reduce precisión numérica de pesos y/o activaciones:
- **float16**: Pesos en fp16 → tamaño ~50% menor
- **int8 simétrico**: Pesos en int8 con rango simétrico [-127, 127]
- **int8 asimétrico**: Pesos en int8 con zero-point
- **Dynamic range**: Pesos en int8, activaciones en float32 durante inferencia
- **Full integer**: Pesos y activaciones en int8

### 1.5 Quantization Aware Training (QAT)
Simula cuantización durante entrenamiento (forward con pesos cuantizados, backward con gradientes en fp32). El modelo aprende a compensar la pérdida de precisión.

### 1.6 TF Lite
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*1.6 TF Lite.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### 1.7 ONNX / ONNX Runtime
Formato abierto para representar modelos. ONNX Runtime optimiza ejecución:
```python
ort_session = onnxruntime.InferenceSession('model.onnx')
outputs = ort_session.run(None, {'input': data})
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*1.7 ONNX / ONNX Runtime.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### 1.8 TensorRT
Optimización específica de NVIDIA para GPU:
- Fusión de capas (layer fusion)
- Auto-tuning de kernels
- Calibración INT8
- FP16/INT8 precision modes
- Dynamic batching

### 1.9 Knowledge Distillation
Entrenar modelo pequeño (student) usando salidas de modelo grande (teacher):
```python
loss = α * CE(student_logits, teacher_logits / T) + (1-α) * CE(student_logits, true_labels)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*1.9 Knowledge Distillation.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---


- **Temperature (T)**: Suaviza distribuciones de probabilidad del teacher
- **α**: Peso relativo entre teacher y true labels

---

## 2. Ejemplos Prácticos

### Ejemplo 1: Weight pruning — Podar 50% de pesos más pequeños en modelo Dense

```python
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

# Datos sintéticos de ventas
np.random.seed(42)
n = 5000
X_ventas = np.random.randn(n, 10)
y_ventas = np.random.randn(n)

# Modelo sin podar
model = Sequential([
    Dense(128, activation='relu', input_shape=(10,)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_ventas, y_ventas, epochs=5, verbose=0, validation_split=0.2)

# Aplicar pruning
pruning_params = {'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.5, 0)}
model_pruned = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
model_pruned.compile(optimizer='adam', loss='mse')

# Entrenar con pruning
callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
model_pruned.fit(X_ventas, y_ventas, epochs=3, verbose=0, callbacks=callbacks)

# Comparar tamaño
import tempfile, os
_, keras_file = tempfile.mkstemp('.h5')
_, pruned_file = tempfile.mkstemp('.h5')
tf.keras.models.save_model(model, keras_file, include_optimizer=False)
tf.keras.models.save_model(model_pruned, pruned_file, include_optimizer=False)
print(f"Tamaño original: {os.path.getsize(keras_file)/1024:.1f} KB")
print(f"Tamaño podado: {os.path.getsize(pruned_file)/1024:.1f} KB")
print(f"Reducción: {(1 - os.path.getsize(pruned_file)/os.path.getsize(keras_file))*100:.1f}%")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: Weight pruning — Podar 50% de pesos más pequeños en modelo Dense.*

1. Datos sintéticos de ventas
2. Modelo sin podar
3. Aplicar pruning
4. Entrenar con pruning
5. Comparar tamaño

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: Polynomial decay sparsity — Poda progresiva durante entrenamiento

```python
pruning_params_poly = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.1, final_sparsity=0.7,
        begin_step=0, end_step=1000
    )
}
model_poly = Sequential([
    Dense(128, activation='relu', input_shape=(10,)),
    Dense(1)
])
model_poly = tfmot.sparsity.keras.prune_low_magnitude(model_poly, **pruning_params_poly)
model_poly.compile(optimizer='adam', loss='mse')
callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
model_poly.fit(X_ventas, y_ventas, epochs=10, verbose=0, callbacks=callbacks)
print("Polynomial decay sparsity completado")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: Polynomial decay sparsity — Poda progresiva durante entrenamiento.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: Unit pruning — Eliminar neuronas completas

```python
from tensorflow_model_optimization.sparsity.keras import prune_low_magnitude

# Unit pruning usando Weight Pruner con axis=[1] (eliminar neuronas completas)
unit_pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.4, 0),
    'block_size': (1, -1),  # Blocks de tamaño (1, toda la fila)
    'block_pooling_type': 'AVG'
}
model_unit = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(1)
])
model_unit = tfmot.sparsity.keras.prune_low_magnitude(model_unit, **unit_pruning_params)
model_unit.compile(optimizer='adam', loss='mse')
model_unit.fit(X_ventas, y_ventas, epochs=5, verbose=0,
               callbacks=[tfmot.sparsity.keras.UpdatePruningStep()])
print("Unit pruning completado")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: Unit pruning — Eliminar neuronas completas.*

1. Unit pruning usando Weight Pruner con axis=[1] (eliminar neuronas completas)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: float16 quantization — Reducir modelo a mitad de tamaño

```python
# Convertir modelo a TFLite con float16
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_fp16 = converter.convert()

with tempfile.NamedTemporaryFile(suffix='.tflite', delete=False) as f:
    f.write(tflite_fp16)
    fp16_file = f.name

print(f"Tamaño float16: {os.path.getsize(fp16_file)/1024:.1f} KB")
print(f"Reducción respecto a original: {(1 - os.path.getsize(fp16_file)/os.path.getsize(keras_file))*100:.1f}%")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: float16 quantization — Reducir modelo a mitad de tamaño.*

1. Convertir modelo a TFLite con float16

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: int8 quantization — Cuantizar pesos a enteros

```python
# int8 quantization con representative dataset
def representative_dataset():
    for _ in range(100):
        data = np.random.randn(1, 10).astype(np.float32)
        yield [data]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_int8 = converter.convert()

with tempfile.NamedTemporaryFile(suffix='.tflite', delete=False) as f:
    f.write(tflite_int8)
    int8_file = f.name

print(f"Tamaño int8: {os.path.getsize(int8_file)/1024:.1f} KB")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: int8 quantization — Cuantizar pesos a enteros.*

1. int8 quantization con representative dataset

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: Dynamic range quantization — Pesos int8 + activaciones float

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_dynamic = converter.convert()

with tempfile.NamedTemporaryFile(suffix='.tflite', delete=False) as f:
    f.write(tflite_dynamic)
    dynamic_file = f.name

print(f"Tamaño dynamic range: {os.path.getsize(dynamic_file)/1024:.1f} KB")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: Dynamic range quantization — Pesos int8 + activaciones float.*

1. `print(f"Tamaño dynamic range: {os.path.getsize(dynamic_file)/1024:.1f} KB")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: QAT — Entrenar con simulación de cuantización

```python
import tensorflow_model_optimization as tfmot

qat_model = Sequential([
    Dense(128, activation='relu', input_shape=(10,)),
    Dense(1)
])

# Aplicar QAT
qat_model = tfmot.quantization.keras.quantize_model(qat_model)
qat_model.compile(optimizer='adam', loss='mse')
qat_model.fit(X_ventas, y_ventas, epochs=5, verbose=0, validation_split=0.2)

# Convertir a TFLite después de QAT
converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_qat = converter.convert()

with tempfile.NamedTemporaryFile(suffix='.tflite', delete=False) as f:
    f.write(tflite_qat)
    qat_file = f.name

print(f"Tamaño QAT: {os.path.getsize(qat_file)/1024:.1f} KB")
print("QAT completado: modelo entrenado para compensar cuantización")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: QAT — Entrenar con simulación de cuantización.*

1. Aplicar QAT
2. Convertir a TFLite después de QAT

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: TF Lite — Convertir modelo a .tflite

```python
# Crear modelo simple
model_tflite = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(1)
])
model_tflite.compile(optimizer='adam', loss='mse')
model_tflite.fit(X_ventas[:100], y_ventas[:100], epochs=3, verbose=0)

converter = tf.lite.TFLiteConverter.from_keras_model(model_tflite)
tflite_model = converter.convert()

with open('modelo_ventas.tflite', 'wb') as f:
    f.write(tflite_model)

# Inferencia con TFLite
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

test_input = np.random.randn(1, 10).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], test_input)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
print(f"Predicción TFLite: {output[0][0]:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: TF Lite — Convertir modelo a .tflite.*

1. Crear modelo simple
2. Inferencia con TFLite

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: TF Lite con representative_dataset para calibración

```python
def rep_dataset():
    for _ in range(200):
        yield [np.random.randn(1, 10).astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model_tflite)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_dataset
tflite_calibrated = converter.convert()

# Evaluar precisión
interpreter = tf.lite.Interpreter(model_content=tflite_calibrated)
interpreter.allocate_tensors()
preds = []
for i in range(50):
    x_test = np.random.randn(1, 10).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], x_test)
    interpreter.invoke()
    preds.append(interpreter.get_tensor(output_details[0]['index'])[0, 0])
print(f"Predictions con calibración: {np.mean(preds):.4f} ± {np.std(preds):.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: TF Lite con representative_dataset para calibración.*

1. Evaluar precisión

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: ONNX — Convertir modelo Keras a ONNX

```python
import tf2onnx

model_onnx = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(1)
])
model_onnx.compile(optimizer='adam', loss='mse')
model_onnx.fit(X_ventas[:100], y_ventas[:100], epochs=3, verbose=0)

spec = (tf.TensorSpec((None, 10), tf.float32, name="input"),)
output_path = "modelo_ventas.onnx"
model_proto, _ = tf2onnx.convert.from_keras(model_onnx, input_signature=spec,
                                             output_path=output_path)
print(f"Modelo ONNX guardado en: {output_path}")
print(f"Tamaño ONNX: {os.path.getsize(output_path)/1024:.1f} KB")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: ONNX — Convertir modelo Keras a ONNX.*

1. `import tf2onnx` — Importa las librerías necesarias para el análisis.
2. `model_onnx.fit(X_ventas[:100], y_ventas[:100], epochs=3, verbose=0)` — Entrena el modelo con los datos de entrenamiento.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: ONNX Runtime — Inferencia con ONNX

```python
import onnxruntime as ort

ort_session = ort.InferenceSession(output_path)
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

test_input = np.random.randn(1, 10).astype(np.float32)
outputs = ort_session.run([output_name], {input_name: test_input})
print(f"Predicción ONNX Runtime: {outputs[0][0, 0]:.4f}")

# Benchmark
import time
times = []
for _ in range(100):
    data = np.random.randn(1, 10).astype(np.float32)
    start = time.perf_counter()
    ort_session.run([output_name], {input_name: data})
    times.append((time.perf_counter() - start) * 1000)
print(f"Latencia ONNX: {np.mean(times):.2f} ms ± {np.std(times):.2f} ms")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: ONNX Runtime — Inferencia con ONNX.*

1. Benchmark

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: TensorRT — Construir engine optimizado para GPU

```python
# Requiere NVIDIA TensorRT instalado
try:
    import tensorrt as trt

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(output_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    serialized_engine = builder.build_serialized_network(network, config)
    with open('modelo_ventas.trt', 'wb') as f:
        f.write(serialized_engine)

    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    print("TensorRT engine construido exitosamente")
except ImportError:
    print("TensorRT no disponible (requiere NVIDIA GPU y TensorRT SDK)")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: TensorRT — Construir engine optimizado para GPU.*

1. Requiere NVIDIA TensorRT instalado

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: FP16 TensorRT — Inferencia con media precisión

```python
try:
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(output_path, 'rb') as model:
        parser.parse(model.read())

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 mode activado")
    else:
        print("FP16 no soportado en esta plataforma")

    serialized_engine = builder.build_serialized_network(network, config)
    with open('modelo_ventas_fp16.trt', 'wb') as f:
        f.write(serialized_engine)
    print("TensorRT FP16 engine construido")
except ImportError:
    print("TensorRT no disponible")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: FP16 TensorRT — Inferencia con media precisión.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: Distillation — Teacher modelo grande, student pequeño

```python
import tensorflow as tf

# Teacher (modelo grande)
teacher = Sequential([
    Dense(256, activation='relu', input_shape=(10,)),
    Dense(128, activation='relu'),
    Dense(1)
])
teacher.compile(optimizer='adam', loss='mse')
teacher.fit(X_ventas, y_ventas, epochs=10, verbose=0)

# Student (modelo pequeño)
student = Sequential([
    Dense(32, activation='relu', input_shape=(10,)),
    Dense(1)
])

# Distillation training
temperature = 5.0
alpha = 0.7

def distillation_loss(y_true, y_pred, teacher_pred):
    loss_teacher = tf.keras.losses.mean_squared_error(
        teacher_pred / temperature, y_pred / temperature)
    loss_true = tf.keras.losses.mean_squared_error(y_true, y_pred)
    return alpha * loss_teacher * (temperature ** 2) + (1 - alpha) * loss_true

teacher_preds = teacher.predict(X_ventas, verbose=0)
student.compile(optimizer='adam', loss=lambda y, p: distillation_loss(y, p, teacher_preds))
student.fit(X_ventas, y_ventas, epochs=10, verbose=0)

print("Distillation completada")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: Distillation — Teacher modelo grande, student pequeño.*

1. Teacher (modelo grande)
2. Student (modelo pequeño)
3. Distillation training

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: Distillation loss — α=0.7 (teacher) + 0.3 (true labels)

```python
# Comparar student entrenado con y sin distillation
student_scratch = Sequential([
    Dense(32, activation='relu', input_shape=(10,)),
    Dense(1)
])
student_scratch.compile(optimizer='adam', loss='mse')
student_scratch.fit(X_ventas, y_ventas, epochs=10, verbose=0)

# Evaluar
X_test = np.random.randn(200, 10)
y_test = np.random.randn(200)
pred_distill = student.predict(X_test, verbose=0)
pred_scratch = student_scratch.predict(X_test, verbose=0)
mse_distill = ((pred_distill.flatten() - y_test)**2).mean()
mse_scratch = ((pred_scratch.flatten() - y_test)**2).mean()
latencia_distill = np.mean([time.perf_counter() - start for _ in range(100)])
latencia_scratch = np.mean([time.perf_counter() - start for _ in range(100)])

print(f"MSE con distillation: {mse_distill:.4f}")
print(f"MSE sin distillation: {mse_scratch:.4f}")
print(f"Mejora: {(1 - mse_distill/mse_scratch)*100:.1f}%")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: Distillation loss — α=0.7 (teacher) + 0.3 (true labels).*

1. Comparar student entrenado con y sin distillation
2. Evaluar

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 16: Comparar tamaño/velocidad/precisión — original vs pruned vs quantized vs distilled

```python
# Benchmark completo
modelos = {
    'Original': (model, None, None),
    'Pruned (50%)': (model_pruned, None, None),
}
resultados = []

for name, (mdl, _conv, _opt) in modelos.items():
    # Tamaño
    _, tmp = tempfile.mkstemp('.h5')
    tf.keras.models.save_model(mdl, tmp, include_optimizer=False)
    size = os.path.getsize(tmp)

    # Latencia
    test_x = np.random.randn(100, 10).astype(np.float32)
    start = time.perf_counter()
    for i in range(100):
        mdl.predict(test_x[i:i+1], verbose=0)
    latency = (time.perf_counter() - start) / 100 * 1000

    # Precisión
    preds = mdl.predict(X_test, verbose=0).flatten()
    mse = ((preds - y_test)**2).mean()

    resultados.append({'modelo': name, 'tamaño_kb': size/1024,
                       'latencia_ms': latency, 'mse': mse})

resultados_df = pd.DataFrame(resultados)
print(resultados_df.to_string(index=False))
print("\nResumen:")
for _, r in resultados_df.iterrows():
    print(f"  {r['modelo']:20s} | {r['tamaño_kb']:8.1f} KB | {r['latencia_ms']:6.2f} ms | MSE={r['mse']:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: Comparar tamaño/velocidad/precisión — original vs pruned vs quantized vs distilled.*

1. Benchmark completo
2. Tamaño
3. Latencia
4. Precisión

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 17: Benchmark — Latencia (ms) y throughput (predicciones/s)

```python
def benchmark_model(model, input_shape=(1, 10), n_runs=500):
    data = np.random.randn(*input_shape).astype(np.float32)
    # Warmup
    for _ in range(10):
        model.predict(data, verbose=0)
    # Benchmark
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model.predict(data, verbose=0)
        times.append(time.perf_counter() - start)
    avg_ms = np.mean(times) * 1000
    throughput = 1 / np.mean(times)
    return avg_ms, throughput

lat_ms, thr = benchmark_model(model)
print(f"Modelo original: {lat_ms:.2f} ms, {thr:.0f} pred/s")

# Batch prediction
data_batch = np.random.randn(32, 10).astype(np.float32)
start = time.perf_counter()
model.predict(data_batch, verbose=0)
batch_time = time.perf_counter() - start
print(f"Batch 32: {batch_time*1000:.2f} ms, {32/batch_time:.0f} pred/s")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: Benchmark — Latencia (ms) y throughput (predicciones/s).*

1. Warmup
2. Benchmark
3. Batch prediction

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 18: Integrador — Pipeline de optimización para deploy

```python
def pipeline_optimizacion(model, X_train, y_train, X_test, y_test):
    resultados = {}

    # 1. Original
    mse_orig = ((model.predict(X_test, verbose=0).flatten() - y_test)**2).mean()
    _, tmp = tempfile.mkstemp('.h5')
    tf.keras.models.save_model(model, tmp, include_optimizer=False)
    size_orig = os.path.getsize(tmp)
    resultados['Original'] = {'MSE': mse_orig, 'Size_KB': size_orig/1024}

    # 2. Pruning 60%
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.6, 0)
    }
    model_p = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
    model_p.compile(optimizer='adam', loss='mse')
    model_p.fit(X_train, y_train, epochs=3, verbose=0,
                callbacks=[tfmot.sparsity.keras.UpdatePruningStep()])
    mse_p = ((model_p.predict(X_test, verbose=0).flatten() - y_test)**2).mean()
    _, tmp_p = tempfile.mkstemp('.h5')
    tf.keras.models.save_model(model_p, tmp_p, include_optimizer=False)
    resultados['Pruned'] = {'MSE': mse_p, 'Size_KB': os.path.getsize(tmp_p)/1024}

    # 3. QAT → TFLite
    qat_m = tfmot.quantization.keras.quantize_model(model)
    qat_m.compile(optimizer='adam', loss='mse')
    qat_m.fit(X_train, y_train, epochs=3, verbose=0)
    converter = tf.lite.TFLiteConverter.from_keras_model(qat_m)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_data = converter.convert()
    interpreter = tf.lite.Interpreter(model_content=tflite_data)
    interpreter.allocate_tensors()
    preds_qat = []
    for i in range(len(X_test)):
        interpreter.set_tensor(0, X_test[i:i+1].astype(np.float32))
        interpreter.invoke()
        preds_qat.append(interpreter.get_tensor(7)[0, 0])
    mse_qat = ((np.array(preds_qat) - y_test)**2).mean()
    resultados['QAT_TFLite'] = {'MSE': mse_qat, 'Size_KB': len(tflite_data)/1024}

    # 4. Resumen
    print(f"{'Método':15s} {'MSE':10s} {'Tamaño':10s} {'Reducción':10s}")
    orig_size = resultados['Original']['Size_KB']
    for name, res in resultados.items():
        reduction = (1 - res['Size_KB']/orig_size) * 100
        print(f"{name:15s} {res['MSE']:10.4f} {res['Size_KB']:10.1f}KB {reduction:9.1f}%")
    print(f"\nMejor relación tamaño-precisión: QAT_TFLite")
    return resultados

# Ejecutar pipeline
results = pipeline_optimizacion(model, X_ventas[:1000], y_ventas[:1000], X_test, y_test)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador — Pipeline de optimización para deploy.*

1. 1. Original
2. 2. Pruning 60%
3. 3. QAT → TFLite
4. 4. Resumen
5. Ejecutar pipeline

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 3. Ejercicios Propuestos

1. **Weight pruning en red LSTM para forecasting de inventarios**: Crea un modelo LSTM con 2 capas (64 y 32 unidades) para predecir demanda semanal. Aplica pruning al 70% de los pesos. Compara pérdida de precisión vs reducción de tamaño.

2. **Comparar dynamic range vs full int8 quantization**: Para un modelo Dense de predicción de ventas (5 features de entrada), convierte a TFLite con dynamic range y full int8. Mide MSE en 1000 muestras de test y tamaño de archivo.

3. **QAT para recuperar precisión tras cuantización**: Entrena un modelo con QAT y compáralo con post-training quantization. Muestra que QAT recupera 1-3% de precisión en un problema de clasificación binaria de productos.

4. **ONNX Runtime vs TensorFlow nativo**: Convierte un modelo Keras a ONNX y mide latencia ONNX Runtime vs TensorFlow nativo para 1000 predicciones. Calcula speedup. Si no tienes GPU, usa solo CPU.

5. **Knowledge distillation en clasificador de productos**: Teacher: Red de 3 capas densas (256-128-64). Student: Red de 2 capas (32-16). Entrena con distillation (T=5, α=0.7) y sin distillation. Muestra mejora en accuracy y reducción de parámetros.

6. **Pipeline completo pruning → quantization → TFLite**: Crea un pipeline que: (1) entrene modelo original, (2) aplique pruning 50%, (3) convierta a TFLite int8, (4) mida latencia y precisión. Genera tabla comparativa.

7. **TensorRT FP16 benchmark**: Si tienes GPU NVIDIA, construye un engine TensorRT FP16 para un modelo de demanda. Compara latencia vs TensorFlow nativo. Si no tienes GPU, simula los resultados esperados con una tabla teórica.

8. **Optimización para deploy en edge device (Raspberry Pi)**: Toma un modelo de clasificación de productos (10 clases, 20 features) y aplícale: pruning 60% → QAT → TFLite int8. Calcula: tamaño final, latencia esperada en ARM Cortex, y precisión final.

---

## 4. Resumen

| Técnica | Reducción tamaño | Pérdida precisión | Latencia | Complejidad |
|---|---|---|---|---|
| **Weight Pruning** | 30-80% (sparsity) | <1% (10-50%) | Similar | Baja |
| **float16** | ~50% | <0.5% | 1-2x speed | Muy baja |
| **int8 (post-training)** | ~75% | 1-3% | 2-4x speed | Baja |
| **QAT int8** | ~75% | <1% | 2-4x speed | Media |
| **TensorRT FP16** | ~50% | <0.5% | 3-5x speed | Alta |
| **TensorRT INT8** | ~75% | 1-2% | 5-10x speed | Muy alta |
| **Distillation** | 80-90% (parámetros) | 1-5% | 5-10x speed | Alta |

Estrategia recomendada para ventas: pruning 50% + QAT int8 + TFLite para mobile/edge, o TensorRT FP16 para servidores NVIDIA.
