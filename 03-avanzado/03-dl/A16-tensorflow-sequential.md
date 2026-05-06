# A16: TensorFlow Sequential API — Redes Neuronales para Ventas, Compras e Inventarios

## Introducción Teórica

La **API Sequential** de tf.keras permite construir modelos de redes neuronales apilando capas de forma secuencial. Es la forma más simple de crear una red: cada capa tiene exactamente un tensor de entrada y uno de salida.

### Componentes principales:

1. **tf.keras.Sequential**: Contenedor que agrupa una pila lineal de capas. Ideal para la mayoría de problemas feedforward.
2. **tf.keras.layers.Dense**: Capa completamente conectada. Parámetros clave:
   - `units`: número de neuronas.
   - `activation`: función de activación ('relu', 'sigmoid', 'softmax', 'linear', 'tanh').
   - `kernel_initializer`: inicialización de pesos ('glorot_uniform', 'he_normal', 'zeros').
   - `bias_initializer`: inicialización del sesgo ('zeros', 'ones').
   - `kernel_regularizer`, `bias_regularizer`, `activity_regularizer`: regularización L1/L2/L1L2.
   - `kernel_constraint`, `bias_constraint`: restricciones (MaxNorm, MinMaxNorm, NonNeg).
   - `use_bias`: incluir o no el sesgo.
3. **Dropout(rate)**: Apaga aleatoriamente un porcentaje de neuronas durante entrenamiento para evitar overfitting.
4. **BatchNormalization**: Normaliza las activaciones de la capa anterior (media=0, var=1). Usa `momentum`, `epsilon`, `center`, `scale`.
5. **Compilación**: `model.compile(optimizer, loss, metrics)` define el algoritmo de optimización, la función de pérdida y las métricas de evaluación.
6. **Entrenamiento y evaluación**: `model.fit()`, `model.evaluate()`, `model.predict()`.
7. **Serialización**: `model.save()` y `tf.keras.models.load_model()` para guardar/cargar modelos completos.
8. **Inspección**: `model.summary()`, `model.get_weights()`, `model.set_weights()`.

### Aplicaciones en negocio:

- **Ventas**: Predecir cantidad vendida (regresión), clasificar si un producto tendrá éxito (binaria), categorizar productos (multiclase).
- **Compras**: Predecir lead time de proveedores, clasificar confiabilidad.
- **Inventarios**: Predecir rotación de productos, clasificar ABC multiclase.

---

## Ejemplos

### Ejemplo 1: Sequential para regresión — Predecir cantidad vendida (1 capa Dense)

```python
import tensorflow as tf
import numpy as np

# Datos: precio, descuento -> cantidad vendida
np.random.seed(42)
n = 1000
X = np.column_stack([
    np.random.uniform(10, 500, n),    # precio
    np.random.uniform(0, 0.3, n)      # descuento
]).astype(np.float32)
y = (200 - 0.3 * X[:, 0] + 150 * X[:, 1] + np.random.normal(0, 20, n)).astype(np.float32)

modelo_reg = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(2,), activation='linear')
])
modelo_reg.compile(optimizer='adam', loss='mse', metrics=['mae'])
modelo_reg.fit(X, y, epochs=50, batch_size=32, verbose=0, validation_split=0.2)
perdida, mae = modelo_reg.evaluate(X, y, verbose=0)
print(f"Regresión — MSE: {perdida:.2f}, MAE: {mae:.2f}")
print(f"Predicción para [precio=300, descuento=0.1]: {modelo_reg.predict([[300.0, 0.1]], verbose=0).flatten()[0]:.1f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: Sequential para regresión — Predecir cantidad vendida (1 capa Dense).*

1. Datos: precio, descuento -> cantidad vendida

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: Sequential para clasificación binaria — ¿El producto se vende?

```python
# Clasificación binaria: 1 = producto se vende, 0 = no se vende
np.random.seed(42)
n = 1000
X_bin = np.column_stack([
    np.random.uniform(10, 500, n),
    np.random.uniform(0, 0.3, n),
    np.random.randint(1, 365, n)  # días desde última compra
]).astype(np.float32)
logits = -2.0 + 0.01 * X_bin[:, 0] + 3.0 * X_bin[:, 1] - 0.005 * X_bin[:, 2]
prob_bin = 1 / (1 + np.exp(-logits))
y_bin = (prob_bin > 0.5).astype(np.float32)

modelo_bin = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
modelo_bin.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
modelo_bin.fit(X_bin, y_bin, epochs=30, batch_size=32, verbose=0, validation_split=0.2)
_, acc = modelo_bin.evaluate(X_bin, y_bin, verbose=0)
print(f"Clasificación binaria — Accuracy: {acc:.3f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: Sequential para clasificación binaria — ¿El producto se vende?.*

1. Clasificación binaria: 1 = producto se vende, 0 = no se vende

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: Sequential para clasificación multiclase — Categoría del producto

```python
np.random.seed(42)
n = 2000
X_mc = np.column_stack([
    np.random.uniform(10, 1000, n),  # precio
    np.random.uniform(0, 0.5, n),    # descuento
    np.random.uniform(0.1, 0.9, n),  # margen
]).astype(np.float32)

# 3 categorías: 0=Economía, 1=Estándar, 2=Premium
y_mc = np.zeros((n, 3), dtype=np.float32)
for i in range(n):
    if X_mc[i, 0] < 200:    cat = 0
    elif X_mc[i, 0] < 500: cat = 1
    else:                   cat = 2
    y_mc[i, cat] = 1.0

modelo_mc = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])
modelo_mc.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
modelo_mc.fit(X_mc, y_mc, epochs=30, batch_size=32, verbose=0, validation_split=0.2)
_, acc_mc = modelo_mc.evaluate(X_mc, y_mc, verbose=0)
print(f"Clasificación multiclase — Accuracy: {acc_mc:.3f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: Sequential para clasificación multiclase — Categoría del producto.*

1. 3 categorías: 0=Economía, 1=Estándar, 2=Premium

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: Dense(units=64, activation='relu') — Capa oculta para características de ventas

```python
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,),
                          kernel_initializer='he_normal',
                          bias_initializer='zeros'),
    tf.keras.layers.Dense(1)
])
modelo.compile(optimizer='adam', loss='mse')
print(f"Pesos de capa 1: {modelo.layers[0].kernel.shape}")
print(f"Sesgos de capa 1: {modelo.layers[0].bias.shape}")
print(f"Total parámetros: {modelo.count_params()}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: Dense(units=64, activation='relu') — Capa oculta para características de ventas.*

1. `print(f"Pesos de capa 1: {modelo.layers[0].kernel.shape}")` — Muestra el resultado por pantalla.
2. `print(f"Sesgos de capa 1: {modelo.layers[0].bias.shape}")` — Muestra el resultado por pantalla.
3. `print(f"Total parámetros: {modelo.count_params()}")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: Dense(units=32, activation='relu') — Segunda capa oculta profunda

```python
modelo_profundo = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(15,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])
modelo_profundo.compile(optimizer='adam', loss='mse')
modelo_profundo.build()
print(f"Arquitectura: {[layer.output_shape for layer in modelo_profundo.layers]}")
print(f"Total parámetros: {modelo_profundo.count_params()}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: Dense(units=32, activation='relu') — Segunda capa oculta profunda.*

1. `print(f"Arquitectura: {[layer.output_shape for layer in modelo_profundo.layers]}")` — Muestra el resultado por pantalla.
2. `print(f"Total parámetros: {modelo_profundo.count_params()}")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: Dropout(0.2) — Regularización para evitar overfitting en ventas

```python
modelo_dropout = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(20,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1)
])
modelo_dropout.compile(optimizer='adam', loss='mse')
modelo_dropout.build()
print(f"Dropout 0.2 aplicado después de capa 1, Dropout 0.3 después de capa 2")
print(f"En entrenamiento: {modelo_dropout.layers[1].rate} de neuronas apagadas")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: Dropout(0.2) — Regularización para evitar overfitting en ventas.*

1. `print(f"Dropout 0.2 aplicado después de capa 1, Dropout 0.3 después de capa 2")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: BatchNormalization() — Normalizar activaciones en red de inventarios

```python
modelo_bn = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(8,)),
    tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1)
])
modelo_bn.compile(optimizer='adam', loss='mse')
modelo_bn.build((None, 8))
print(f"BatchNormalization params por capa: {[layer.count_params() if hasattr(layer, 'count_params') else 0 for layer in modelo_bn.layers]}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: BatchNormalization() — Normalizar activaciones en red de inventarios.*

1. `print(f"BatchNormalization params por capa: {[layer.count_params() if hasattr(layer, 'count_params') else 0 for layer in modelo_bn.layers]}")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: Compilar con optimizer='adam', loss='mse', metrics=['mae']

```python
modelo_comp = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(1)
])
modelo_comp.compile(optimizer='adam', loss='mse', metrics=['mae'])
print(f"Optimizer: {modelo_comp.optimizer.__class__.__name__}")
print(f"Loss: {modelo_comp.loss}")
print(f"Métricas: {[m.name for m in modelo_comp.metrics]}")

# Datos de ejemplo
X_ej = np.random.rand(100, 5).astype(np.float32)
y_ej = np.random.rand(100, 1).astype(np.float32)
hist = modelo_comp.fit(X_ej, y_ej, epochs=10, batch_size=16, verbose=0)
print(f"MAE final: {hist.history['mae'][-1]:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: Compilar con optimizer='adam', loss='mse', metrics=['mae'].*

1. Datos de ejemplo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: Compilar con loss='binary_crossentropy', metrics=['accuracy']

```python
modelo_bin_comp = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
modelo_bin_comp.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)
X_bin2 = np.random.rand(200, 4).astype(np.float32)
y_bin2 = np.random.randint(0, 2, (200, 1)).astype(np.float32)
hist_bin = modelo_bin_comp.fit(X_bin2, y_bin2, epochs=20, batch_size=16, verbose=0)
print(f"Accuracy final: {hist_bin.history['accuracy'][-1]:.4f}")
print(f"Precision final: {hist_bin.history['precision'][-1]:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: Compilar con loss='binary_crossentropy', metrics=['accuracy'].*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: fit con validation_split=0.2 (20% para validación)

```python
modelo_val = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(1)
])
modelo_val.compile(optimizer='adam', loss='mse', metrics=['mae'])
np.random.seed(42)
X_val = np.random.rand(500, 5).astype(np.float32)
y_val = np.random.rand(500, 1).astype(np.float32)

hist_val = modelo_val.fit(X_val, y_val, epochs=50, batch_size=32,
                          validation_split=0.2, verbose=0)
print(f"Train MAE final: {hist_val.history['mae'][-1]:.4f}")
print(f"Val MAE final: {hist_val.history['val_mae'][-1]:.4f}")
print(f"Train epochs: {len(hist_val.history['loss'])}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: fit con validation_split=0.2 (20% para validación).*

1. `hist_val = modelo_val.fit(X_val, y_val, epochs=50, batch_size=32,` — Entrena el modelo con los datos de entrenamiento.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: fit con epochs=100, batch_size=32

```python
modelo_ep = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(1)
])
modelo_ep.compile(optimizer='adam', loss='mse')
X_ep = np.random.rand(1000, 3).astype(np.float32)
y_ep = np.random.rand(1000, 1).astype(np.float32)

hist_ep = modelo_ep.fit(X_ep, y_ep, epochs=100, batch_size=32, verbose=0)
print(f"Loss final después de 100 epochs: {hist_ep.history['loss'][-1]:.6f}")
print(f"Loss primera época: {hist_ep.history['loss'][0]:.6f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: fit con epochs=100, batch_size=32.*

1. `hist_ep = modelo_ep.fit(X_ep, y_ep, epochs=100, batch_size=32, verbose=0)` — Entrena el modelo con los datos de entrenamiento.
2. `print(f"Loss final después de 100 epochs: {hist_ep.history['loss'][-1]:.6f}")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: predict — Hacer predicciones sobre nuevos datos de productos

```python
# Entrenar modelo simple
modelo_pred = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(1)
])
modelo_pred.compile(optimizer='adam', loss='mse')

X_train_pred = np.random.rand(500, 3).astype(np.float32)
y_train_pred = np.random.rand(500, 1).astype(np.float32)
modelo_pred.fit(X_train_pred, y_train_pred, epochs=10, batch_size=32, verbose=0)

# Nuevos productos
nuevos_productos = np.array([
    [250.0, 0.15, 30.0],  # precio, descuento, días inventario
    [80.0,  0.25, 5.0],
    [450.0, 0.05, 60.0]
], dtype=np.float32)

predicciones = modelo_pred.predict(nuevos_productos, verbose=0)
for i, p in enumerate(predicciones.flatten()):
    print(f"Producto {i+1}: venta estimada = {p:.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: predict — Hacer predicciones sobre nuevos datos de productos.*

1. Entrenar modelo simple
2. Nuevos productos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: evaluate — Evaluar en test set de compras

```python
# Dividir datos
np.random.seed(42)
X_eval = np.random.rand(1000, 4).astype(np.float32)
y_eval = np.random.rand(1000, 1).astype(np.float32)
split = 800
X_train_ev, X_test_ev = X_eval[:split], X_eval[split:]
y_train_ev, y_test_ev = y_eval[:split], y_eval[split:]

modelo_ev = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(1)
])
modelo_ev.compile(optimizer='adam', loss='mse', metrics=['mae'])
modelo_ev.fit(X_train_ev, y_train_ev, epochs=30, batch_size=32, verbose=0)

test_loss, test_mae = modelo_ev.evaluate(X_test_ev, y_test_ev, verbose=0)
print(f"Test Loss (MSE): {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: evaluate — Evaluar en test set de compras.*

1. Dividir datos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: summary — Ver arquitectura del modelo de clasificación

```python
modelo_sum = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,), name='capa_entrada'),
    tf.keras.layers.Dropout(0.3, name='dropout1'),
    tf.keras.layers.Dense(32, activation='relu', name='capa_oculta'),
    tf.keras.layers.Dense(3, activation='softmax', name='salida')
])
modelo_sum.compile(optimizer='adam', loss='categorical_crossentropy')
modelo_sum.summary()
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Ejemplo 14: summary — Ver arquitectura del modelo de clasificación.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: model.save y load_model — Guardar y cargar modelo de predicción de demanda

```python
import tempfile, os

modelo_guardar = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(1)
])
modelo_guardar.compile(optimizer='adam', loss='mse')
X_s = np.random.rand(100, 5).astype(np.float32)
y_s = np.random.rand(100, 1).astype(np.float32)
modelo_guardar.fit(X_s, y_s, epochs=5, batch_size=16, verbose=0)

ruta_temp = os.path.join(tempfile.gettempdir(), 'modelo_demanda.keras')
modelo_guardar.save(ruta_temp)
print(f"Modelo guardado en: {ruta_temp}")

modelo_cargado = tf.keras.models.load_model(ruta_temp)
pred_original = modelo_guardar.predict(X_s[:3], verbose=0)
pred_cargada = modelo_cargado.predict(X_s[:3], verbose=0)
print(f"Predicciones originales: {pred_original.flatten()}")
print(f"Predicciones cargadas:   {pred_cargada.flatten()}")
print(f"¿Coinciden?: {np.allclose(pred_original, pred_cargada)}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: model.save y load_model — Guardar y cargar modelo de predicción de demanda.*

1. `import tempfile, os` — Importa las librerías necesarias para el análisis.
2. `modelo_guardar.fit(X_s, y_s, epochs=5, batch_size=16, verbose=0)` — Entrena el modelo con los datos de entrenamiento.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 16: get_weights/set_weights — Inspeccionar pesos de un modelo entrenado

```python
modelo_pesos = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(1)
])
modelo_pesos.compile(optimizer='adam', loss='mse')
X_p = np.random.rand(50, 3).astype(np.float32)
y_p = np.random.rand(50, 1).astype(np.float32)
modelo_pesos.fit(X_p, y_p, epochs=10, batch_size=8, verbose=0)

pesos = modelo_pesos.get_weights()
print(f"Número de arrays de pesos: {len(pesos)}")
print(f"Pesos capa 1 (kernel): shape {pesos[0].shape}, min={pesos[0].min():.4f}, max={pesos[0].max():.4f}")
print(f"Sesgos capa 1: shape {pesos[1].shape}")
print(f"Pesos capa 2 (kernel): shape {pesos[2].shape}")

# Modificar pesos y recargar
pesos[0] = pesos[0] * 0.5
modelo_pesos.set_weights(pesos)
print("Pesos después de set_weights (modificados)")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: get_weights/set_weights — Inspeccionar pesos de un modelo entrenado.*

1. Modificar pesos y recargar

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 17: Diferentes inicializadores: glorot_uniform vs he_normal

```python
modelo_glorot = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(10,),
                          kernel_initializer='glorot_uniform',
                          bias_initializer='zeros',
                          name='dense_glorot'),
    tf.keras.layers.Dense(1, name='salida')
])

modelo_he = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(10,),
                          kernel_initializer='he_normal',
                          bias_initializer='zeros',
                          name='dense_he'),
    tf.keras.layers.Dense(1, name='salida')
])

w_glorot = modelo_glorot.get_weights()[0]
w_he = modelo_he.get_weights()[0]
print(f"Glorot uniform — media: {w_glorot.mean():.6f}, std: {w_glorot.std():.6f}")
print(f"He normal — media: {w_he.mean():.6f}, std: {w_he.std():.6f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: Diferentes inicializadores: glorot_uniform vs he_normal.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 18: Integrador — Red neuronal para predicción de demanda con regularización

```python
# Dataset de demanda: precio, descuento, días_promoción, competencia_precio, temporada
np.random.seed(42)
n = 5000
X_dem = np.column_stack([
    np.random.uniform(10, 500, n),
    np.random.uniform(0, 0.4, n),
    np.random.randint(0, 30, n),
    np.random.uniform(10, 500, n),
    np.random.randint(0, 4, n)
]).astype(np.float32)

y_dem = (100 - 0.2*X_dem[:, 0] + 80*X_dem[:, 1] + 3*X_dem[:, 2]
         - 0.1*X_dem[:, 3] + 10*X_dem[:, 4] + np.random.normal(0, 15, n)).astype(np.float32)

split = 4000
X_train_d, X_test_d = X_dem[:split], X_dem[split:]
y_train_d, y_test_d = y_dem[:split], y_dem[split:]

modelo_demanda = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(5,),
                          kernel_initializer='he_normal',
                          kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu',
                          kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
modelo_demanda.compile(optimizer='adam', loss='mse', metrics=['mae'])
hist_dem = modelo_demanda.fit(X_train_d, y_train_d, epochs=100, batch_size=64,
                              validation_split=0.2, verbose=0)

test_loss_d, test_mae_d = modelo_demanda.evaluate(X_test_d, y_test_d, verbose=0)
print(f"Test MSE: {test_loss_d:.2f}, Test MAE: {test_mae_d:.2f}")

# Predicción para un nuevo producto
nuevo_prod = np.array([[300.0, 0.15, 7.0, 280.0, 2.0]], dtype=np.float32)
pred_dem = modelo_demanda.predict(nuevo_prod, verbose=0)
print(f"Demanda estimada: {pred_dem.flatten()[0]:.1f} unidades")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador — Red neuronal para predicción de demanda con regularización.*

1. Dataset de demanda: precio, descuento, días_promoción, competencia_precio, temporada
2. Predicción para un nuevo producto

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejercicios

1. **Regresión con Early Stopping**: Crea un Sequential con 3 capas Dense (64, 32, 1), compílalo con Adam y MSE, y entrénalo con 2000 datos de ventas (precio, gasto_marketing, competencia). Usa EarlyStopping callback con patience=10.

2. **Clasificación binaria de proveedores confiables**: Genera 1000 muestras con 4 features (años_relación, entregas_tarde %, precio_promedio, calidad_promedio) y target binario (confiable=1/no=0). Crea y entrena un Sequential con 2 capas ocultas y sigmoid final.

3. **Multiclase para segmentación ABC**: Genera datos con 3 features (rotación, margen, costo) y 3 clases (A, B, C). Usa Sequential con softmax, categorical_crossentropy y reporta la matriz de confusión.

4. **Comparación de inicializadores**: Crea dos modelos idénticos cambiando `kernel_initializer` entre 'random_uniform' y 'he_normal'. Entrena en el mismo dataset de ventas y compara la convergencia (pérdida por época).

5. **Dropout vs sin Dropout**: Genera un dataset pequeño (200 muestras, 20 features) con ruido. Entrena un modelo grande (256-128-64-1) con y sin Dropout(0.5). Compara train loss vs val loss para detectar overfitting.

6. **Guardar y cargar con predict**: Entrena un modelo para predecir margen de ganancia (features: precio_compra, precio_venta, cantidad, costo_logístico). Guarda el modelo, cárgalo y haz 5 predicciones nuevas.

7. **Regularización L2**: Añade `kernel_regularizer=l2(0.01)` a todas las capas Dense de un modelo de 4 capas. Compara la magnitud de los pesos (get_weights) contra un modelo sin regularización.

8. **Integrador completo**: Construye un modelo Sequential para predecir la rotación de inventario (30 features sintéticas). Debe incluir: BatchNormalization, Dropout(0.2-0.3), regularización L2, varias capas ocultas, y usar 'mae' como métrica. Reporta test MAE.

---

## Resumen

La API Sequential de tf.keras permite construir rápida y eficientemente redes neuronales feedforward para problemas de ventas, compras e inventarios:

- **Regresión**: Predecir cantidades, montos, márgenes con `loss='mse'` y `activation='linear'` en la salida.
- **Clasificación binaria**: Predecir si un producto se vende o proveedor es confiable con `loss='binary_crossentropy'` y `activation='sigmoid'`.
- **Clasificación multiclase**: Categorizar productos (ABC, segmentos) con `loss='categorical_crossentropy'` y `activation='softmax'`.
- **Regularización**: `Dropout` apaga neuronas aleatoriamente; `BatchNormalization` acelera y estabiliza el entrenamiento; regularizadores L1/L2 controlan la magnitud de pesos.
- **Inicializadores**: `glorot_uniform` (default para tanh/sigmoid), `he_normal` (ReLU) mejoran la convergencia.
- **Ciclo de vida**: `compile` → `fit` → `evaluate` → `predict`. Guardar con `save` y cargar con `load_model`.
- **Inspección**: `summary()` para arquitectura y parámetros; `get_weights()/set_weights()` para acceder y modificar pesos entrenados.

Esta API es el punto de partida ideal para la mayoría de problemas de deep learning en el contexto de negocio, antes de migrar a arquitecturas más complejas con la API Funcional.
