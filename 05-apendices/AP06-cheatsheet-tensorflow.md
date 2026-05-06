# AP06 — Cheatsheet TensorFlow / Keras

## 1. Instalación y Configuración

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Verificar versión
print(tf.__version__)
print(keras.__version__)

# GPU disponible
print("GPU disponible:", tf.config.list_physical_devices("GPU"))

# Fijar semilla para reproducibilidad
tf.random.set_seed(42)

# Configurar memoria GPU (evitar OOM)
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
    )

# Modo eager (por defecto en TF 2.x)
tf.executing_eagerly()  # True
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*1. Instalación y Configuración.*

1. Verificar versión
2. GPU disponible
3. Fijar semilla para reproducibilidad
4. Configurar memoria GPU (evitar OOM)
5. Modo eager (por defecto en TF 2.x)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 2. tf.constant y tf.Variable

```python
# Constantes
a = tf.constant(42)
b = tf.constant([1, 2, 3, 4, 5])
c = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
d = tf.constant("Hola TensorFlow")

# Propiedades
c.shape          # TensorShape([2, 2])
c.dtype          # tf.float32
c.numpy()        # convierte a numpy: [[1., 2.], [3., 4.]]

# Variables (para pesos entrenables)
w = tf.Variable(tf.random.normal([10, 5]))
b = tf.Variable(tf.zeros([5]))
w.assign(tf.ones([10, 5]))
w.assign_add(tf.ones([10, 5]) * 0.01)

# Operaciones con tensores
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])
tf.matmul(a, b)              # producto matricial
a @ b                        # equivalente (Python 3.5+)
tf.add(a, b)                 # suma
tf.reduce_mean(a)            # media
tf.reduce_sum(a, axis=1)     # suma por fila
tf.reshape(a, [4, 1])        # reshape
tf.transpose(a)              # transpuesta
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*2. tf.constant y tf.Variable.*

1. Constantes
2. Propiedades
3. Variables (para pesos entrenables)
4. Operaciones con tensores

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 3. Modelo Sequential

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

# Crear modelo secuencial
model = Sequential([
    Dense(128, input_shape=(100,), activation="relu"),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])

# Compilar
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC()]
)

# Resumen
model.summary()

# Entrenar
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=50,
    validation_split=0.2,
    callbacks=[...],
    verbose=1
)

# Evaluar
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

# Predecir
y_pred = model.predict(X_test)
y_pred_proba = model.predict(X_test)
```

**Salida:**

```
# El modelo ajusta sus parámetros durante el entrenamiento.
# Los parámetros aprendidos quedan disponibles en el objeto.
```

**Explicación línea por línea:**

*3. Modelo Sequential.*

1. Crear modelo secuencial
2. Compilar
3. Resumen
4. Entrenar
5. Evaluar
6. Predecir

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 4. Modelo Funcional API

```python
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model

# Entradas
input_1 = Input(shape=(50,), name="numericas")
input_2 = Input(shape=(20,), name="categoricas")

# Ramas
x1 = Dense(32, activation="relu")(input_1)
x1 = Dropout(0.3)(x1)
x2 = Dense(16, activation="relu")(input_2)

# Concatenación
concat = Concatenate()([x1, x2])
output = Dense(1, activation="sigmoid", name="output")(concat)

# Modelo
model = Model(inputs=[input_1, input_2], outputs=output)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Entrenar con múltiples entradas
model.fit(
    {"numericas": X_num, "categoricas": X_cat},
    y_train,
    epochs=30,
    batch_size=32
)
```

**Salida:**

```
# El modelo ajusta sus parámetros durante el entrenamiento.
# Los parámetros aprendidos quedan disponibles en el objeto.
```

**Explicación línea por línea:**

*4. Modelo Funcional API.*

1. Entradas
2. Ramas
3. Concatenación
4. Modelo
5. Entrenar con múltiples entradas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 5. Capas Especializadas

```python
from tensorflow.keras.layers import (
    Conv1D, Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    LSTM, GRU, SimpleRNN, Bidirectional,
    Embedding, BatchNormalization, LayerNormalization,
    Reshape, Flatten, Concatenate, Add, Average
)

# Convolucionales 2D (imágenes)
Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3))
MaxPooling2D((2, 2))
GlobalAveragePooling2D()

# Convolucionales 1D (series temporales)
Conv1D(64, 3, activation="relu", input_shape=(100, 1))

# Recurrentes
LSTM(64, return_sequences=True)
LSTM(32)
GRU(64)
SimpleRNN(32)
Bidirectional(LSTM(64))

# Embedding
Embedding(input_dim=10000, output_dim=128, input_length=100)

# Normalización
BatchNormalization()
LayerNormalization()

# Skip connections (ResNet)
Add()([x, shortcut])
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*5. Capas Especializadas.*

1. Convolucionales 2D (imágenes)
2. Convolucionales 1D (series temporales)
3. Recurrentes
4. Embedding
5. Normalización
6. Skip connections (ResNet)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 6. Callbacks

```python
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    TensorBoard, CSVLogger, LearningRateScheduler
)

# Early stopping
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    min_delta=1e-4
)

# Checkpoint
checkpoint = ModelCheckpoint(
    "mejor_modelo.h5",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)

# Reducir learning rate
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6
)

# TensorBoard
tensorboard = TensorBoard(log_dir="logs", histogram_freq=1)

# CSV Logger
csv_logger = CSVLogger("training_log.csv")

# Usar juntos
callbacks = [early_stop, checkpoint, reduce_lr, tensorboard]

model.fit(X_train, y_train, epochs=100, callbacks=callbacks,
          validation_data=(X_val, y_val))
```

**Salida:**

```
# El modelo ajusta sus parámetros durante el entrenamiento.
# Los parámetros aprendidos quedan disponibles en el objeto.
```

**Explicación línea por línea:**

*6. Callbacks.*

1. Early stopping
2. Checkpoint
3. Reducir learning rate
4. TensorBoard
5. CSV Logger
6. Usar juntos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 7. tf.data — Pipelines de Datos

```python
# Dataset desde numpy
dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))

# Operaciones del pipeline
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(32)
dataset = dataset.prefetch(tf.data.AUTOTUNE)
dataset = dataset.repeat(2)              # repetir 2 épocas
dataset = dataset.map(lambda x, y: (x * 2, y))  # transformación

# Dataset desde CSV
dataset = tf.data.experimental.make_csv_dataset(
    "ventas.csv",
    batch_size=32,
    label_name="target",
    num_epochs=1
)

# Dataset desde imágenes
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "imagenes/",
    batch_size=32,
    image_size=(224, 224),
    validation_split=0.2,
    subset="training",
    seed=42
)

# Pipeline eficiente
dataset = dataset.cache()                # cache en memoria
dataset = dataset.prefetch(tf.data.AUTOTUNE)  # prefetch automático

# Entrenar con dataset
model.fit(dataset, epochs=10, steps_per_epoch=100)
```

**Salida:**

```
# El modelo ajusta sus parámetros durante el entrenamiento.
# Los parámetros aprendidos quedan disponibles en el objeto.
```

**Explicación línea por línea:**

*7. tf.data — Pipelines de Datos.*

1. Dataset desde numpy
2. Operaciones del pipeline
3. Dataset desde CSV
4. Dataset desde imágenes
5. Pipeline eficiente
6. Entrenar con dataset

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 8. Custom Training Loop

```python
# Pérdida y optimizador
loss_fn = keras.losses.BinaryCrossentropy()
optimizer = keras.optimizers.Adam(learning_rate=0.001)

# Métricas
train_acc = keras.metrics.BinaryAccuracy()
val_acc = keras.metrics.BinaryAccuracy()

# Loop de entrenamiento
for epoch in range(epochs):
    # Training
    for batch, (X_batch, y_batch) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits = model(X_batch, training=True)
            loss = loss_fn(y_batch, logits)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_acc.update_state(y_batch, logits)

    # Validation
    for X_batch, y_batch in val_dataset:
        logits = model(X_batch, training=False)
        val_acc.update_state(y_batch, logits)

    print(f"Epoch {epoch}: acc={train_acc.result():.3f}, "
          f"val_acc={val_acc.result():.3f}")
    train_acc.reset_state()
    val_acc.reset_state()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*8. Custom Training Loop.*

1. Pérdida y optimizador
2. Métricas
3. Loop de entrenamiento
4. Training
5. Validation

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 9. Transfer Learning

```python
from tensorflow.keras.applications import ResNet50, VGG16, EfficientNetB0
from tensorflow.keras.applications.resnet50 import preprocess_input

# Cargar modelo pre-entrenado
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Congelar capas base
base_model.trainable = False

# Añadir cabecera personalizada
inputs = keras.Input(shape=(224, 224, 3))
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
outputs = Dense(10, activation="softmax")(x)

model = keras.Model(inputs, outputs)
model.compile(optimizer="adam", loss="categorical_crossentropy")

# Fine-tuning
base_model.trainable = True
model.compile(optimizer=keras.optimizers.Adam(1e-5), loss="categorical_crossentropy")
model.fit(dataset, epochs=10)
```

**Salida:**

```
# El modelo ajusta sus parámetros durante el entrenamiento.
# Los parámetros aprendidos quedan disponibles en el objeto.
```

**Explicación línea por línea:**

*9. Transfer Learning.*

1. Cargar modelo pre-entrenado
2. Congelar capas base
3. Añadir cabecera personalizada
4. Fine-tuning

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 10. Guardar y Cargar Modelos

```python
# Guardar pesos
model.save_weights("pesos.h5")
model.load_weights("pesos.h5")

# Guardar modelo completo
model.save("modelo_completo.h5")
model.save("modelo_savedmodel", save_format="tf")  # formato TF SavedModel

# Cargar modelo completo
model = keras.models.load_model("modelo_completo.h5")
model = keras.models.load_model("modelo_savedmodel")

# Exportar a TFLite (para móvil/edge)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("modelo.tflite", "wb") as f:
    f.write(tflite_model)

# Exportar a TensorFlow.js
# !tensorflowjs_converter --input_format=keras modelo.h5 tfjs_model/
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*10. Guardar y Cargar Modelos.*

1. Guardar pesos
2. Guardar modelo completo
3. Cargar modelo completo
4. Exportar a TFLite (para móvil/edge)
5. Exportar a TensorFlow.js
6. !tensorflowjs_converter --input_format=keras modelo.h5 tfjs_model/

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 11. Regularización

```python
from tensorflow.keras.regularizers import l1, l2, l1_l2

# L1/L2 en capas
Dense(64, activation="relu", kernel_regularizer=l2(0.01))
Dense(64, activation="relu", bias_regularizer=l2(0.01))
Dense(64, activation="relu", activity_regularizer=l1(0.001))
Dense(64, activation="relu", kernel_regularizer=l1_l2(l1=0.01, l2=0.01))

# Dropout
Dropout(0.5)      # apaga 50% de neuronas aleatoriamente

# Batch Normalization (también regulariza)
BatchNormalization()
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*11. Regularización.*

1. L1/L2 en capas
2. Dropout
3. Batch Normalization (también regulariza)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 12. Learning Rate Schedulers

```python
# Exponencial
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=1000,
    decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Piecewise
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[10000, 20000],
    values=[0.01, 0.001, 0.0001]
)

# Warm-up + Cosine Decay
def lr_warmup_cosine(epoch, lr):
    warmup_epochs = 5
    total_epochs = 100
    if epoch < warmup_epochs:
        return lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return lr * 0.5 * (1 + np.cos(np.pi * progress))

callback_lr = LearningRateScheduler(lr_warmup_cosine, verbose=1)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*12. Learning Rate Schedulers.*

1. Exponencial
2. Piecewise
3. Warm-up + Cosine Decay

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 13. Custom Layers

```python
# Capa personalizada simple
class MyDense(keras.layers.Layer):
    def __init__(self, units=32, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer="glorot_uniform",
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer="zeros",
                                 trainable=True)

    def call(self, inputs):
        return self.activation(inputs @ self.w + self.b)

# Usar capa personalizada
model = Sequential([
    MyDense(64, activation="relu"),
    MyDense(1, activation="sigmoid")
])
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*13. Custom Layers.*

1. Capa personalizada simple
2. Usar capa personalizada

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 14. Custom Loss Functions

```python
# Función de pérdida personalizada
def weighted_binary_crossentropy(weights_pos=5.0):
    def loss(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        weights = weights_pos * y_true + (1 - y_true)
        return tf.reduce_mean(weights * bce)
    return loss

model.compile(optimizer="adam", loss=weighted_binary_crossentropy(10))

# Huber loss
def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    is_small = tf.abs(error) < delta
    squared = 0.5 * tf.square(error)
    linear = delta * (tf.abs(error) - 0.5 * delta)
    return tf.where(is_small, squared, linear)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*14. Custom Loss Functions.*

1. Función de pérdida personalizada
2. Huber loss

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 15. Distribución y Estrategias

```python
# Entrenamiento distribuido en múltiples GPUs
strategy = tf.distribute.MirroredStrategy()
print(f"Número de GPUs: {strategy.num_replicas_in_sync}")

with strategy.scope():
    model = Sequential([...])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# TPU Strategy
resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*15. Distribución y Estrategias.*

1. Entrenamiento distribuido en múltiples GPUs
2. TPU Strategy

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## Referencia Rápida

| Operación | Código |
|-----------|--------|
| Constante | `tf.constant([1, 2, 3])` |
| Variable | `tf.Variable(tf.random.normal([10, 5]))` |
| Sequential | `Sequential([Dense(...), Dropout(...)])` |
| Functional | `Model(inputs=[...], outputs=[...])` |
| Compilar | `model.compile(optimizer="adam", loss="mse")` |
| Entrenar | `model.fit(X, y, epochs=10, batch_size=32)` |
| Predecir | `model.predict(X)` |
| Callbacks | `EarlyStopping, ModelCheckpoint, ReduceLROnPlateau` |
| Guardar | `model.save("modelo.h5")` |
| Cargar | `load_model("modelo.h5")` |
| Dataset | `tf.data.Dataset.from_tensor_slices((X, y)).batch(32)` |
| Custom loop | `GradientTape()`, `tape.gradient()`, `apply_gradients()` |
