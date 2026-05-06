# A26 - Deep Learning: CNN para Ventas, Compras e Inventarios

## Fundamentos Teóricos

Las redes neuronales convolucionales (CNN) están diseñadas para procesar datos con estructura de cuadrícula. Aunque son famosas por imágenes, las Conv1D son ideales para series temporales de ventas.

### Operación de Convolución 1D
```
y[t] = Σᵢ x[t+i] * w[i]   para i = 0, ..., kernel_size-1
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Operación de Convolución 1D.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---


- **filters**: número de detectores de patrones
- **kernel_size**: ventana temporal de análisis
- **strides**: paso del deslizamiento
- **padding**: 'valid' (reduce), 'same' (misma longitud), 'causal' (futuro no visto)
- **dilation_rate**: espacio entre elementos del kernel (campo receptivo expandido)

### Capas de Pooling
| Tipo | Operación | Efecto |
|------|-----------|--------|
| MaxPooling1D | máximo en ventana | Detecta características dominantes |
| AveragePooling1D | promedio en ventana | Suaviza representación |
| GlobalAvgPooling1D | promedio global | Reduce secuencia a vector |
| GlobalMaxPooling1D | máximo global | Detecta activación más fuerte |

### Tipos de Convolución
| Tipo | Descripción | Parámetros |
|------|-------------|-----------|
| Conv1D | Convolución estándar 1D | filters × kernel_size × channels |
| SeparableConv1D | Depthwise + Pointwise | Más eficiente, menos params |
| DepthwiseConv1D | Convolución por canal | Un kernel por canal de entrada |
| Conv1DTranspose | Deconvolución | Aumenta dimensionalidad |
| Conv2D | Convolución 2D | Para datos con 2 dimensiones espaciales |

### Padding
- **valid**: No padding, salida se reduce: `(W - K + 1) / S`
- **same**: Padding para misma longitud: `W / S`
- **causal**: Padding solo a la izquierda (no usa futuro)

### Arquitectura CNN típica
```
Input → [Conv1D + BN + ReLU + Pooling] × N → GlobalPooling → Dense → Output
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Arquitectura CNN típica.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejemplos Prácticos

### Ejemplo 1: Conv1D para Series de Ventas (kernel_size=3, patrones de 3 días)

```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten
from tensorflow.keras.optimizers import Adam

np.random.seed(42)
tf.random.set_seed(42)

# Serie temporal de ventas diarias (365 días)
dias = 365
t = np.arange(dias)
ventas = (100 + 20*np.sin(2*np.pi*t/7) + 10*np.sin(2*np.pi*t/30) +
          5*np.sin(2*np.pi*t/365) + np.random.normal(0, 5, dias))

# Ventana deslizante: 7 días → predecir día 8
def crear_ventanas(series, ventana=7):
    X, y = [], []
    for i in range(len(series) - ventana):
        X.append(series[i:i+ventana])
        y.append(series[i+ventana])
    return np.array(X), np.array(y)

X, y = crear_ventanas(ventas, ventana=7)
X = X.reshape(-1, 7, 1)

X_train, X_val = X[:300], X[300:]
y_train, y_val = y[:300], y[300:]

modelo = Sequential([
    Conv1D(filters=16, kernel_size=3, activation='relu',
           input_shape=(7, 1), padding='valid'),
    Flatten(),
    Dense(1)
])
modelo.compile(optimizer=Adam(0.01), loss='mse')
modelo.summary()

history = modelo.fit(X_train, y_train, epochs=50, batch_size=16,
                     validation_data=(X_val, y_val), verbose=1)

# Predicciones
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('Curva de pérdida')
plt.legend()

plt.subplot(1, 2, 2)
y_pred = modelo.predict(X_val, verbose=0)
plt.plot(y_val[:50], label='Real', marker='o')
plt.plot(y_pred[:50], label='Predicho', marker='x')
plt.title('Predicción de ventas (50 días)')
plt.legend()
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 1: Conv1D para Series de Ventas (kernel_size=3, patrones de 3 días).*

1. Serie temporal de ventas diarias (365 días)
2. Ventana deslizante: 7 días → predecir día 8
3. Predicciones

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: Conv1D + MaxPooling1D — Reducir Dimensionalidad Temporal

```python
from tensorflow.keras.layers import MaxPooling1D

modelo_pool = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(30, 1)),
    MaxPooling1D(pool_size=2, strides=2),  # 30 → 15
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),             # 15 → 7
    Flatten(),
    Dense(32, activation='relu'),
    Dense(1)
])
modelo_pool.compile(optimizer=Adam(0.001), loss='mse')
modelo_pool.summary()
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Ejemplo 2: Conv1D + MaxPooling1D — Reducir Dimensionalidad Temporal.*

1. `from tensorflow.keras.layers import MaxPooling1D` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: Conv1D con padding='same' — Mantener Longitud

```python
modelo_same = Sequential([
    Conv1D(filters=32, kernel_size=5, padding='same', activation='relu',
           input_shape=(30, 1)),
    # output sigue siendo 30 pasos
    Conv1D(filters=32, kernel_size=5, padding='same', activation='relu'),
    # output sigue siendo 30 pasos
    GlobalAvgPooling1D(),
    Dense(1)
])

# Comparar shapes con padding='valid'
modelo_valid = Sequential([
    Conv1D(filters=32, kernel_size=5, padding='valid', activation='relu',
           input_shape=(30, 1)),
    # output: 30-5+1 = 26
    Conv1D(filters=32, kernel_size=5, padding='valid', activation='relu'),
    # output: 26-5+1 = 22
    GlobalAvgPooling1D(),
    Dense(1)
])

print("Padding 'same':")
modelo_same.build((None, 30, 1))
print(f"  Input: (None, 30, 1)")
print(f"  After 2 Conv1D same: mantiene 30 pasos")

print("\nPadding 'valid':")
modelo_valid.build((None, 30, 1))
print(f"  Input: (None, 30, 1)")
print(f"  After 2 Conv1D valid: 30→26→22 pasos")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: Conv1D con padding='same' — Mantener Longitud.*

1. output sigue siendo 30 pasos
2. output sigue siendo 30 pasos
3. Comparar shapes con padding='valid'
4. output: 30-5+1 = 26
5. output: 26-5+1 = 22

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: Conv1D con dilation_rate=2 — Campo Receptivo Expandido

```python
# Dilation_rate=2: kernel_size=3 ve posiciones [0, 2, 4]
# Campo receptivo más grande sin aumentar parámetros
modelo_dil = Sequential([
    Conv1D(filters=16, kernel_size=3, dilation_rate=2,
           padding='causal', activation='relu', input_shape=(30, 1)),
    # Campo receptivo: 1 + 2*(3-1) = 5
    Conv1D(filters=16, kernel_size=3, dilation_rate=4,
           padding='causal', activation='relu'),
    # Capa 2: 1 + 4*(3-1) = 9
    GlobalAvgPooling1D(),
    Dense(1)
])
modelo_dil.compile(optimizer=Adam(0.001), loss='mse')
modelo_dil.summary()

# Simular datos de ventas con dependencias largas
t = np.arange(500)
ventas_largas = (100 + 30*np.sin(2*np.pi*t/30) +
                 15*np.sin(2*np.pi*t/90) +
                 np.random.normal(0, 8, 500))

X, y = crear_ventanas(ventas_largas, ventana=30)
X = X.reshape(-1, 30, 1)
X_train, X_val = X[:400], X[400:]
y_train, y_val = y[:400], y[400:]

h_dil = modelo_dil.fit(X_train, y_train, epochs=30, batch_size=32,
                       validation_data=(X_val, y_val), verbose=0)

print(f"Val loss (dilation): {min(h_dil.history['val_loss']):.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: Conv1D con dilation_rate=2 — Campo Receptivo Expandido.*

1. Dilation_rate=2: kernel_size=3 ve posiciones [0, 2, 4]
2. Campo receptivo más grande sin aumentar parámetros
3. Campo receptivo: 1 + 2*(3-1) = 5
4. Capa 2: 1 + 4*(3-1) = 9
5. Simular datos de ventas, compras o inventarios con dependencias largas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: Múltiples Filtros — 32, 64, 128 (Efecto en Capacidad)

```python
filtros_lista = [8, 32, 128]
histories_filt = {}

for f in filtros_lista:
    modelo = Sequential([
        Conv1D(filters=f, kernel_size=3, activation='relu', input_shape=(30, 1)),
        MaxPooling1D(2),
        Conv1D(filters=f*2, kernel_size=3, activation='relu'),
        GlobalAvgPooling1D(),
        Dense(1)
    ])
    modelo.compile(optimizer=Adam(0.001), loss='mse')
    n_params = modelo.count_params()
    h = modelo.fit(X_train, y_train, epochs=30, batch_size=32,
                   validation_data=(X_val, y_val), verbose=0)
    histories_filt[f] = h
    print(f"Filtros iniciales: {f:3d} | Parámetros: {n_params:>8,d} | Val loss: {min(h.history['val_loss']):.4f}")

for f in filtros_lista:
    plt.plot(histories_filt[f].history['val_loss'], label=f'{f} filtros')
plt.xlabel('Epoch')
plt.ylabel('Val Loss')
plt.title('Efecto del número de filtros')
plt.legend()
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: Múltiples Filtros — 32, 64, 128 (Efecto en Capacidad).*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: Conv1D Stack — 3 Capas Consecutivas

```python
modelo_stack = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', padding='same', input_shape=(30, 1)),
    Conv1D(32, kernel_size=3, activation='relu', padding='same'),
    Conv1D(64, kernel_size=3, activation='relu', padding='same'),
    GlobalAvgPooling1D(),
    Dense(1)
])
modelo_stack.compile(optimizer=Adam(0.001), loss='mse')
modelo_stack.summary()
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Ejemplo 6: Conv1D Stack — 3 Capas Consecutivas.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: GlobalAvgPooling1D — Resumir Toda la Secuencia

```python
from tensorflow.keras.layers import GlobalAveragePooling1D

# Comparar GlobalAvgPooling vs Flatten
modelo_gap = Sequential([
    Conv1D(32, 3, activation='relu', input_shape=(30, 1)),
    Conv1D(64, 3, activation='relu'),
    GlobalAvgPooling1D(),  # Resumen: promedio de toda la secuencia
    Dense(1)
])
modelo_gap.compile(optimizer=Adam(0.001), loss='mse')

modelo_flat = Sequential([
    Conv1D(32, 3, activation='relu', input_shape=(30, 1)),
    Conv1D(64, 3, activation='relu'),
    Flatten(),  # Aplana la secuencia
    Dense(1)
])
modelo_flat.compile(optimizer=Adam(0.001), loss='mse')

print("GlobalAvgPooling params:", modelo_gap.count_params())
print("Flatten params:", modelo_flat.count_params())
# GlobalAvgPooling tiene menos parámetros porque no depende de la longitud
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: GlobalAvgPooling1D — Resumir Toda la Secuencia.*

1. Comparar GlobalAvgPooling vs Flatten
2. GlobalAvgPooling tiene menos parámetros porque no depende de la longitud

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: Conv1D para Clasificación de Productos

```python
from tensorflow.keras.utils import to_categorical

# 4 categorías de productos, series de 20 features
n_prod = 1000
X_prod = np.random.randn(n_prod, 20, 5)  # 20 días, 5 features
y_prod = np.random.randint(0, 4, n_prod)
y_prod_cat = to_categorical(y_prod)

Xp_train, Xp_val = X_prod[:800], X_prod[800:]
yp_train, yp_val = y_prod_cat[:800], y_prod_cat[800:]

modelo_clasif = Sequential([
    Conv1D(32, 3, activation='relu', padding='same', input_shape=(20, 5)),
    Conv1D(64, 3, activation='relu', padding='same'),
    GlobalAvgPooling1D(),
    Dense(32, activation='relu'),
    Dense(4, activation='softmax')
])
modelo_clasif.compile(optimizer=Adam(0.001), loss='categorical_crossentropy',
                      metrics=['accuracy'])
modelo_clasif.summary()

h_clas = modelo_clasif.fit(Xp_train, yp_train, epochs=30, batch_size=32,
                           validation_data=(Xp_val, yp_val), verbose=1)
print(f"Accuracy: {max(h_clas.history['val_accuracy']):.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: Conv1D para Clasificación de Productos.*

1. 4 categorías de productos, series de 20 features

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: SeparableConv1D — Menos Parámetros

```python
from tensorflow.keras.layers import SeparableConv1D

# SeparableConv1D: depthwise + pointwise
# Parámetros = 3*1*32 + 32*64 ≈ mucho menos que Conv1D normal: 3*32*64
modelo_sep = Sequential([
    SeparableConv1D(32, 3, activation='relu', input_shape=(30, 1)),
    SeparableConv1D(64, 3, activation='relu'),
    GlobalAvgPooling1D(),
    Dense(1)
])

modelo_std = Sequential([
    Conv1D(32, 3, activation='relu', input_shape=(30, 1)),
    Conv1D(64, 3, activation='relu'),
    GlobalAvgPooling1D(),
    Dense(1)
])

print(f"SeparableConv1D params: {modelo_sep.count_params():,}")
print(f"Conv1D estándar params:  {modelo_std.count_params():,}")
print(f"Reducción: {(1 - modelo_sep.count_params()/modelo_std.count_params())*100:.1f}%")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: SeparableConv1D — Menos Parámetros.*

1. SeparableConv1D: depthwise + pointwise
2. Parámetros = 3*1*32 + 32*64 ≈ mucho menos que Conv1D normal: 3*32*64

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: DepthwiseConv1D — Convolución por Canal

```python
from tensorflow.keras.layers import DepthwiseConv1D

# Depthwise: un kernel independiente por canal de entrada
# Input (30, 4) → DepthwiseConv1D con 4 canales → cada canal se convoluciona por separado
X_multi = np.random.randn(500, 30, 4)  # 4 features de ventas
y_multi = np.random.randn(500)

modelo_depth = Sequential([
    DepthwiseConv1D(kernel_size=3, activation='relu', input_shape=(30, 4)),
    # Output: (30, 4) mismo número de canales
    Conv1D(16, kernel_size=3, activation='relu'),
    GlobalAvgPooling1D(),
    Dense(1)
])
modelo_depth.compile(optimizer=Adam(0.001), loss='mse')
modelo_depth.summary()
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Ejemplo 10: DepthwiseConv1D — Convolución por Canal.*

1. Depthwise: un kernel independiente por canal de entrada
2. Input (30, 4) → DepthwiseConv1D con 4 canales → cada canal se convoluciona por separado
3. Output: (30, 4) mismo número de canales

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: Conv2D para Datos Tabulares como Imagen

```python
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten

# Matriz producto × día como "imagen" 2D
# 10 productos × 30 días = imagen de 10×30
X_2d = np.random.randn(500, 10, 30, 1)  # 500 muestras, 10 prod, 30 días, 1 canal
y_2d = np.random.randn(500)

modelo_2d = Sequential([
    Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(10, 30, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    GlobalAvgPooling1D(data_format='channels_last'),
    Dense(1)
])
modelo_2d.compile(optimizer=Adam(0.001), loss='mse')
modelo_2d.summary()
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Ejemplo 11: Conv2D para Datos Tabulares como Imagen.*

1. Matriz producto × día como "imagen" 2D
2. 10 productos × 30 días = imagen de 10×30

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: Conv2D + MaxPooling2D para Patrones 2D

```python
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Patrones 2D: semana × producto (7 días × 20 productos)
X_semanal = np.random.randn(1000, 7, 20, 1)   # 7 días, 20 productos
y_semanal = np.random.randn(1000)

modelo_2d_full = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(7, 20, 1)),
    MaxPooling2D(pool_size=(2, 2), padding='same'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2), padding='same'),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(1)
])
modelo_2d_full.compile(optimizer=Adam(0.001), loss='mse')
modelo_2d_full.summary()
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Ejemplo 12: Conv2D + MaxPooling2D para Patrones 2D.*

1. Patrones 2D: semana × producto (7 días × 20 productos)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: Comparar Conv1D vs Dense para Series de Ventas

```python
# Dataset sintético: ventas últimas 30 días → predecir día 31
X_comp, y_comp = crear_ventanas(ventas, ventana=30)
X_comp = X_comp.reshape(-1, 30, 1)
Xc_train, Xc_val = X_comp[:300], X_comp[300:]
yc_train, yc_val = y_comp[:300], y_comp[300:]

# Modelo Dense (trata cada día como feature independiente)
model_dense = Sequential([
    Flatten(input_shape=(30, 1)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

# Modelo Conv1D (respeta estructura temporal)
model_cnn = Sequential([
    Conv1D(32, 3, activation='relu', input_shape=(30, 1)),
    Conv1D(32, 3, activation='relu'),
    GlobalAvgPooling1D(),
    Dense(1)
])

model_dense.compile(optimizer=Adam(0.001), loss='mse')
model_cnn.compile(optimizer=Adam(0.001), loss='mse')

print(f"Params Dense: {model_dense.count_params():,}")
print(f"Params Conv1D: {model_cnn.count_params():,}")

h_dense = model_dense.fit(Xc_train, yc_train, epochs=50, batch_size=16,
                          validation_data=(Xc_val, yc_val), verbose=0)
h_cnn = model_cnn.fit(Xc_train, yc_train, epochs=50, batch_size=16,
                      validation_data=(Xc_val, yc_val), verbose=0)

plt.plot(h_dense.history['val_loss'], label='Dense', linestyle='--')
plt.plot(h_cnn.history['val_loss'], label='Conv1D', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Conv1D vs Dense para series temporales')
plt.legend()
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: Comparar Conv1D vs Dense para Series de Ventas.*

1. Dataset sintético: ventas últimas 30 días → predecir día 31
2. Modelo Dense (trata cada día como feature independiente)
3. Modelo Conv1D (respeta estructura temporal)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: BatchNormalization después de Conv1D

```python
from tensorflow.keras.layers import BatchNormalization, Activation

modelo_cnn_bn = Sequential([
    Conv1D(32, 3, input_shape=(30, 1), use_bias=False),
    BatchNormalization(),
    Activation('relu'),
    Conv1D(64, 3, use_bias=False),
    BatchNormalization(),
    Activation('relu'),
    GlobalAvgPooling1D(),
    Dense(1)
])
modelo_cnn_bn.compile(optimizer=Adam(0.001), loss='mse')
modelo_cnn_bn.summary()
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Ejemplo 14: BatchNormalization después de Conv1D.*

1. `from tensorflow.keras.layers import BatchNormalization, Activation` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: Dropout después de Convolución

```python
from tensorflow.keras.layers import SpatialDropout1D, Dropout

modelo_cnn_drop = Sequential([
    Conv1D(32, 3, activation='relu', input_shape=(30, 1)),
    SpatialDropout1D(0.2),  # Dropout de canales enteros
    Conv1D(64, 3, activation='relu'),
    Dropout(0.3),            # Dropout estándar
    GlobalAvgPooling1D(),
    Dense(1)
])
modelo_cnn_drop.compile(optimizer=Adam(0.001), loss='mse')

# SpatialDropout1D apaga canales completos → mejor para datos correlacionados
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Ejemplo 15: Dropout después de Convolución.*

1. SpatialDropout1D apaga canales completos → mejor para datos correlacionados

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 16: Arquitectura CNN Completa para Predicción de Demanda Semanal

```python
X_sem = np.random.randn(800, 28, 3)   # 28 días, 3 features (ventas, precio, promoción)
y_sem = np.random.randn(800)          # demanda semanal

Xs_train, Xs_val = X_sem[:600], X_sem[600:]
ys_train, ys_val = y_sem[:600], y_sem[600:]

cnn_demanda = Sequential([
    Conv1D(32, 3, padding='same', input_shape=(28, 3)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling1D(2),  # 28 → 14

    Conv1D(64, 3, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling1D(2),  # 14 → 7

    Conv1D(128, 3, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    GlobalAvgPooling1D(),

    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1)
])
cnn_demanda.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
cnn_demanda.summary()

h_sem = cnn_demanda.fit(Xs_train, ys_train, epochs=50, batch_size=32,
                        validation_data=(Xs_val, ys_val), verbose=1)

plt.plot(h_sem.history['loss'], label='train')
plt.plot(h_sem.history['val_loss'], label='val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('CNN para predicción de demanda semanal')
plt.legend()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 16: Arquitectura CNN Completa para Predicción de Demanda Semanal.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 17: Visualizar Filtros Aprendidos

```python
# Entrenar modelo pequeño para visualizar filtros
modelo_viz = Sequential([
    Conv1D(8, 3, activation='relu', input_shape=(30, 1), name='conv1'),
    Conv1D(16, 3, activation='relu', name='conv2'),
    GlobalAvgPooling1D(),
    Dense(1)
])
modelo_viz.compile(optimizer=Adam(0.001), loss='mse')

# Entrenar con datos
Xv, yv = crear_ventanas(ventas, ventana=30)
Xv = Xv.reshape(-1, 30, 1)
modelo_viz.fit(Xv[:400], yv[:400], epochs=20, batch_size=32, verbose=0)

# Obtener pesos de la primera capa convolucional
pesos, bias = modelo_viz.get_layer('conv1').get_weights()
print(f"Forma de los pesos: {pesos.shape}")  # (kernel_size, input_channels, filters)

# Visualizar los 8 filtros (kernel_size=3)
fig, axes = plt.subplots(2, 4, figsize=(12, 4))
for i, ax in enumerate(axes.flat):
    if i < pesos.shape[2]:
        ax.stem(pesos[:, 0, i])
        ax.set_title(f'Filtro {i+1}')
        ax.set_ylim(pesos.min()-0.1, pesos.max()+0.1)
plt.suptitle('Filtros aprendidos por Conv1D (kernel_size=3)')
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: Visualizar Filtros Aprendidos.*

1. Entrenar modelo pequeño para visualizar filtros
2. Entrenar con datos
3. Obtener pesos de la primera capa convolucional
4. Visualizar los 8 filtros (kernel_size=3)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 18: Integrador — CNN para Pronóstico de Series de Ventas

```python
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Dataset realista: 2000 días de ventas con tendencia + estacionalidad
np.random.seed(42)
tf.random.set_seed(42)

dias_total = 2000
t = np.arange(dias_total)

# Componentes: tendencia + semanal + mensual + anual + ruido
tendencia = 0.02 * t
semanal = 25 * np.sin(2 * np.pi * t / 7)
mensual = 15 * np.sin(2 * np.pi * t / 30)
anual = 10 * np.sin(2 * np.pi * t / 365)
ruido = np.random.normal(0, 8, dias_total)

ventas_completas = 100 + tendencia + semanal + mensual + anual + ruido
ventas_completas = np.maximum(ventas_completas, 0)  # no negativas

# Preparar datos: ventana de 14 días → predecir 1 día
X_final, y_final = crear_ventanas(ventas_completas, ventana=14)
X_final = X_final.reshape(-1, 14, 1)

split = int(0.8 * len(X_final))
Xf_train, Xf_val = X_final[:split], X_final[split:]
yf_train, yf_val = y_final[:split], y_final[split:]

# CNN integradora
cnn_integrador = Sequential([
    Conv1D(32, 3, padding='causal', activation='relu', input_shape=(14, 1)),
    BatchNormalization(),
    MaxPooling1D(2),

    Conv1D(64, 3, padding='causal', activation='relu'),
    BatchNormalization(),
    MaxPooling1D(2),

    Conv1D(128, 3, padding='causal', activation='relu'),
    BatchNormalization(),

    GlobalAvgPooling1D(),
    Dense(64, activation='relu', kernel_regularizer=l2(1e-4)),
    Dropout(0.25),
    Dense(1)
])

optimizer = Adam(learning_rate=0.001)
cnn_integrador.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
cnn_integrador.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
]

h_final = cnn_integrador.fit(Xf_train, yf_train, epochs=100, batch_size=32,
                             validation_data=(Xf_val, yf_val),
                             callbacks=callbacks, verbose=1)

# Evaluación
y_pred_final = cnn_integrador.predict(Xf_val, verbose=0)
from sklearn.metrics import mean_squared_error, r2_score
r2 = r2_score(yf_val, y_pred_final)
rmse = np.sqrt(mean_squared_error(yf_val, y_pred_final))
print(f"\n=== Resultados CNN Integradora ===")
print(f"R²: {r2:.4f}")
print(f"RMSE: {rmse:.2f} unidades")

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(h_final.history['loss'], label='Train')
plt.plot(h_final.history['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Curva de aprendizaje')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(yf_val[:100], label='Real', marker='o', alpha=0.7)
plt.plot(y_pred_final[:100], label='Predicho', marker='x', alpha=0.7)
plt.xlabel('Día')
plt.ylabel('Ventas')
plt.title('Predicción CNN (100 días de validación)')
plt.legend()
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador — CNN para Pronóstico de Series de Ventas.*

1. Dataset realista: 2000 días de ventas con tendencia + estacionalidad
2. Componentes: tendencia + semanal + mensual + anual + ruido
3. Preparar datos: ventana de 14 días → predecir 1 día
4. CNN integradora
5. Evaluación

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejercicios

1. **Conv1D para compras**: Construye un modelo Conv1D que prediga el volumen de compras diarias usando ventana de 10 días. Usa kernel_size=4, 32 filtros, padding='same' y una capa Dense de salida. Features: precio, stock_actual, lead_time.

2. **MaxPooling vs AveragePooling**: Para un dataset de ventas con ventana de 60 días, compara MaxPooling1D vs AveragePooling1D después de Conv1D(32,5). ¿Cuál retiene mejor la información para predecir ventas futuras?

3. **Padding comparison**: Implementa 3 modelos Conv1D idénticos pero con padding='valid', 'same', 'causal'. Entrénalos en datos de ventas con ventana=20 y kernel_size=5. Compara val_loss y el tamaño de salida de cada capa convolucional.

4. **Clasificación de tendencia de productos**: Usa Conv1D para clasificar si un producto tendrá tendencia al alza (1), estable (0) o a la baja (-1) basado en 30 días de historia con 4 features. Salida softmax 3 clases.

5. **SeparableConv1D para eficiencia**: Implementa un modelo de predicción de inventario usando SeparableConv1D. Compara número de parámetros y rendimiento contra un Conv1D estándar con la misma arquitectura. Reporta la reducción porcentual de parámetros.

6. **CNN 2D para matriz producto×día**: Genera datos de 15 productos × 21 días. Usa Conv2D con kernel (3,3) para predecir ventas del día 22. Incluye BatchNormalization y MaxPooling2D. ¿Cómo interpretas los patrones 2D?

7. **Dilated convolutions para dependencias largas**: Implementa una CNN con 3 capas Conv1D usando dilation_rates = [1, 2, 4] y padding='causal' para capturar dependencias de hasta 15 días en una secuencia de ventas de 60 días de entrada. Compara con una CNN sin dilación.

8. **Integrador: CNN para detección de anomalías en inventario**: Diseña una CNN 1D que tome 30 días de datos (ventas, devoluciones, stock, precio) y detecte si un día es anómalo. Usa: 3 bloques Conv1D + BN + ReLU + MaxPooling, GlobalAvgPooling, Dropout(0.3), salida sigmoid binaria. Incluye early stopping y ReduceLROnPlateau.

---

## Resumen

- **Conv1D** extrae patrones locales en series temporales de ventas (kernel_size define la ventana de análisis).
- **MaxPooling/AveragePooling** reducen dimensionalidad temporal y agregan invarianza a traslaciones pequeñas.
- **padding='same'** mantiene la longitud; **padding='valid'** reduce; **padding='causal'** evita usar información futura.
- **dilation_rate** expande el campo receptivo sin añadir parámetros, ideal para dependencias largas.
- **SeparableConv1D** reduce drásticamente parámetros vs Conv1D estándar.
- **GlobalAvgPooling1D** reemplaza a Flatten, reduciendo parámetros y previniendo overfitting.
- **Conv2D** modela interacciones producto×día como imágenes 2D.
- Las CNN son más eficientes que Dense para series temporales porque comparten pesos y respetan la estructura temporal.
- La arquitectura típica: Conv1D → BN → ReLU → Pooling repetido → GlobalPooling → Dense.
- Combinar CNN con dropout, BN, L2, early stopping da modelos robustos para pronóstico de ventas.
