# A29 - Deep Learning: Regularización y Optimización para Ventas

## Fundamentos Teóricos

La regularización y la optimización son dos pilares del entrenamiento exitoso de redes neuronales. Sin regularización, los modelos sobreajustan; sin un buen optimizador, no convergen.

### Regularización de Pesos

| Tipo | Fórmula | Efecto |
|------|---------|--------|
| L1 (Lasso) | λ·Σ|w| | Pesos dispersos (algunos a cero) |
| L2 (Ridge) | λ·Σw² | Pesos pequeños pero no cero |
| L1+L2 (ElasticNet) | λ₁·Σ|w| + λ₂·Σw² | Combina ambos efectos |

Se aplica vía `kernel_regularizer`, `bias_regularizer`, `activity_regularizer`.

### Dropout
- **Dropout(rate)**: apaga neuronas con probabilidad `rate` en cada paso
- **SpatialDropout1D**: apaga canales completos (para CNNs)
- **SpatialDropout2D**: apaga mapas de activación completos
- `noise_shape`: permite broadcasting del dropout mask
- Efecto: entrenamiento ruidoso → inferencia robusta (ensemble implícito)

### Normalización
| Tipo | Normaliza sobre | Parámetros |
|------|----------------|------------|
| BatchNormalization | eje del batch | momentum, epsilon, gamma, beta |
| LayerNormalization | eje de features | epsilon, center, scale |

- **BatchNormalization**: `γ·(x-μ_B)/σ_B + β`. Normaliza por batch, mantiene medias móviles.
- **LayerNormalization**: normaliza por features, útil en RNN/Transformers.

### Early Stopping
```
if val_loss no mejora por 'patience' epochs:
    detener entrenamiento
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Early Stopping.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---


- `restore_best_weights=True`: vuelve a los mejores pesos
- `min_delta`: mejora mínima requerida
- `start_from_epoch`: empezar a monitorear desde cierta época

### Learning Rate Scheduling
- **ReduceLROnPlateau**: reduce lr cuando métrica se estanca
  - `factor`: multiplicador del lr (típico 0.1-0.5)
  - `patience`: épocas sin mejora antes de reducir
  - `min_lr`: lr mínimo
  - `cooldown`: épocas de espera después de reducir
  - `mode`: 'min'/'max'/auto
- **LearningRateScheduler**: función personalizada por época
- **ExponentialDecay**: `lr = initial_lr * decay_rate^(step/decay_steps)`

### Optimizadores

| Optimizador | Característica | Hiperparámetros |
|-------------|---------------|-----------------|
| SGD | Descenso gradiente clásico | lr, momentum, nesterov |
| Adam | Adaptativo + momentum | lr, beta_1=0.9, beta_2=0.999 |
| AdamW | Adam con weight_decay | lr, weight_decay |
| Nadam | Adam + Nesterov | lr, beta_1, beta_2 |
| RMSprop | Raíz cuadrada del gradiente | lr, rho=0.9, momentum |
| Adagrad | Acumula gradientes al cuadrado | lr, initial_accumulator_value |
| Adadelta | Mejora Adagrad (no necesita lr inicial) | rho=0.95, epsilon |

### Gradient Clipping
- **clipnorm**: norma máxima del gradiente (L2)
- **clipvalue**: valor absoluto máximo por componente
- Previene exploding gradients en RNNs y redes profundas

### Weight Decay (Decaimiento de Pesos)
- Penaliza pesos grandes similar a L2
- En AdamW: weight_decay se aplica directamente, no como regularización L2 clásica
- Diferencia sutil: weight_decay en actualización vs L2 en loss

---

## Ejemplos Prácticos

### Ejemplo 1: L2 Regularization — kernel_regularizer=l2(0.01)

```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
tf.random.set_seed(42)

# Dataset de ventas
n = 2000
X = np.random.randn(n, 10)
y = (2*X[:, 0] - 1.5*X[:, 1] + 0.5*X[:, 2]**2 + np.sin(X[:, 3]) +
     np.random.normal(0, 0.3, n))
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Sin regularización
modelo_no_reg = Sequential([
    Dense(128, activation='relu', input_shape=(10,)),
    Dense(64, activation='relu'),
    Dense(1)
])

# Con L2
modelo_l2 = Sequential([
    Dense(128, activation='relu', input_shape=(10,),
          kernel_regularizer=l2(0.01)),
    Dense(64, activation='relu',
          kernel_regularizer=l2(0.01)),
    Dense(1)
])

modelo_no_reg.compile(optimizer=Adam(0.001), loss='mse')
modelo_l2.compile(optimizer=Adam(0.001), loss='mse')

h_no = modelo_no_reg.fit(X_train, y_train, epochs=100, batch_size=32,
                          validation_data=(X_val, y_val), verbose=0)
h_l2 = modelo_l2.fit(X_train, y_train, epochs=100, batch_size=32,
                     validation_data=(X_val, y_val), verbose=0)

plt.plot(h_no.history['val_loss'], label='Sin L2', linestyle='--')
plt.plot(h_l2.history['val_loss'], label='Con L2 (0.01)', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Efecto de regularización L2')
plt.legend()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 1: L2 Regularization — kernel_regularizer=l2(0.01).*

1. Dataset de ventas
2. Sin regularización
3. Con L2

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: L1 Regularization — Pesos Dispersos

```python
from tensorflow.keras.regularizers import l1

modelo_l1 = Sequential([
    Dense(64, activation='relu', input_shape=(10,),
          kernel_regularizer=l1(0.005)),
    Dense(32, activation='relu',
          kernel_regularizer=l1(0.005)),
    Dense(1)
])
modelo_l1.compile(optimizer=Adam(0.001), loss='mse')
h_l1 = modelo_l1.fit(X_train, y_train, epochs=100, batch_size=32,
                     validation_data=(X_val, y_val), verbose=0)

# Ver esparcidad de pesos
pesos_l1 = modelo_l1.get_weights()
for i, w in enumerate(pesos_l1):
    if w.ndim > 1:  # matrices de pesos (no biases)
        frac_cero = (np.abs(w) < 0.01).mean()
        print(f"Capa {i//2}: fracción de pesos ≈0: {frac_cero:.1%}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: L1 Regularization — Pesos Dispersos.*

1. Ver esparcidad de pesos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: L1+L2 — Elastic Net Regularization

```python
from tensorflow.keras.regularizers import l1_l2

modelo_en = Sequential([
    Dense(64, activation='relu', input_shape=(10,),
          kernel_regularizer=l1_l2(l1=0.001, l2=0.001)),
    Dense(32, activation='relu',
          kernel_regularizer=l1_l2(l1=0.001, l2=0.001)),
    Dense(1)
])
modelo_en.compile(optimizer=Adam(0.001), loss='mse')
h_en = modelo_en.fit(X_train, y_train, epochs=100, batch_size=32,
                     validation_data=(X_val, y_val), verbose=0)

print(f"ElasticNet val_loss: {min(h_en.history['val_loss']):.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: L1+L2 — Elastic Net Regularization.*

1. `from tensorflow.keras.regularizers import l1_l2` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: Dropout(0.2) vs Dropout(0.5) en Capas Ocultas

```python
from tensorflow.keras.layers import Dropout

rates = [0.0, 0.2, 0.5, 0.7]
histories_drop = {}

for rate in rates:
    modelo = Sequential([
        Dense(256, activation='relu', input_shape=(10,)),
        Dropout(rate),
        Dense(256, activation='relu'),
        Dropout(rate),
        Dense(1)
    ])
    modelo.compile(optimizer=Adam(0.001), loss='mse')
    h = modelo.fit(X_train, y_train, epochs=80, batch_size=32,
                   validation_data=(X_val, y_val), verbose=0)
    histories_drop[rate] = h
    print(f"Dropout(rate={rate:.1f}) | Train: {h.history['loss'][-1]:.4f} | Val: {h.history['val_loss'][-1]:.4f}")

for rate in rates:
    plt.plot(histories_drop[rate].history['val_loss'], label=f'dropout={rate}')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Dropout: comparación de tasas')
plt.legend()
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: Dropout(0.2) vs Dropout(0.5) en Capas Ocultas.*

1. `from tensorflow.keras.layers import Dropout` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: SpatialDropout1D — Dropear Canales Enteros para Series

```python
from tensorflow.keras.layers import SpatialDropout1D, Conv1D, GlobalAvgPooling1D

# SpatialDropout1D: apaga canales completos (feature maps)
# vs Dropout normal: apaga elementos individuales
X_series = np.random.randn(500, 30, 4)
y_series = np.random.randn(500)

modelo_spatial = Sequential([
    Conv1D(32, 3, activation='relu', input_shape=(30, 4)),
    SpatialDropout1D(0.3),  # apaga canales completos de 32
    Conv1D(64, 3, activation='relu'),
    SpatialDropout1D(0.3),
    GlobalAvgPooling1D(),
    Dense(1)
])
modelo_spatial.compile(optimizer=Adam(0.001), loss='mse')

modelo_drop_normal = Sequential([
    Conv1D(32, 3, activation='relu', input_shape=(30, 4)),
    Dropout(0.3),  # apaga elementos individuales
    Conv1D(64, 3, activation='relu'),
    Dropout(0.3),
    GlobalAvgPooling1D(),
    Dense(1)
])
modelo_drop_normal.compile(optimizer=Adam(0.001), loss='mse')

print("SpatialDropout1D: útil cuando features vecinos están correlacionados")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: SpatialDropout1D — Dropear Canales Enteros para Series.*

1. SpatialDropout1D: apaga canales completos (feature maps)
2. vs Dropout normal: apaga elementos individuales

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: BatchNormalization — Acelerar Convergencia

```python
from tensorflow.keras.layers import BatchNormalization, Activation

modelo_bn = Sequential([
    Dense(64, input_shape=(10,), use_bias=False),
    BatchNormalization(momentum=0.99, epsilon=0.001),
    Activation('relu'),
    Dense(32, use_bias=False),
    BatchNormalization(),
    Activation('relu'),
    Dense(1)
])

modelo_no_bn = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(32, activation='relu'),
    Dense(1)
])

modelo_bn.compile(optimizer=Adam(0.01), loss='mse')
modelo_no_bn.compile(optimizer=Adam(0.01), loss='mse')

h_bn = modelo_bn.fit(X_train, y_train, epochs=50, batch_size=32,
                     validation_data=(X_val, y_val), verbose=0)
h_no_bn = modelo_no_bn.fit(X_train, y_train, epochs=50, batch_size=32,
                            validation_data=(X_val, y_val), verbose=0)

plt.plot(h_no_bn.history['val_loss'], label='Sin BN', linestyle='--')
plt.plot(h_bn.history['val_loss'], label='Con BN', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('BatchNormalization: convergencia más rápida')
plt.legend()
plt.show()

# Ver parámetros de BN
print("Parámetros BN (gamma, beta, moving_mean, moving_variance):")
for layer in modelo_bn.layers:
    if isinstance(layer, BatchNormalization):
        print(f"  gamma: {layer.gamma.numpy()[:3]}, beta: {layer.beta.numpy()[:3]}")
        print(f"  moving_mean: {layer.moving_mean.numpy()[:3]}, moving_variance: {layer.moving_variance.numpy()[:3]}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: BatchNormalization — Acelerar Convergencia.*

1. Ver parámetros de BN

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: LayerNormalization — Alternativa para RNNs

```python
from tensorflow.keras.layers import LayerNormalization, LSTM, RNN

modelo_ln = Sequential([
    LSTM(64, return_sequences=True, input_shape=(14, 1)),
    LayerNormalization(epsilon=1e-6),
    LSTM(32, return_sequences=False),
    LayerNormalization(epsilon=1e-6),
    Dense(1)
])
modelo_ln.compile(optimizer=Adam(0.001), loss='mse')
modelo_ln.summary()

# LayerNormalization: no depende del batch size, ideal para RNNs
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Ejemplo 7: LayerNormalization — Alternativa para RNNs.*

1. LayerNormalization: no depende del batch size, ideal para RNNs

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: EarlyStopping — patience=5 vs patience=20

```python
from tensorflow.keras.callbacks import EarlyStopping

modelo_es5 = Sequential([Dense(64, activation='relu', input_shape=(10,)),
                         Dense(32, activation='relu'), Dense(1)])
modelo_es20 = Sequential([Dense(64, activation='relu', input_shape=(10,)),
                          Dense(32, activation='relu'), Dense(1)])

modelo_es5.compile(optimizer=Adam(0.001), loss='mse')
modelo_es20.compile(optimizer=Adam(0.001), loss='mse')

es5 = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, min_delta=1e-4)
es20 = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, min_delta=1e-4)

h5 = modelo_es5.fit(X_train, y_train, epochs=500, batch_size=32,
                     validation_data=(X_val, y_val), callbacks=[es5], verbose=0)
h20 = modelo_es20.fit(X_train, y_train, epochs=500, batch_size=32,
                      validation_data=(X_val, y_val), callbacks=[es20], verbose=0)

print(f"EarlyStopping patience=5  | Paró en época {len(h5.history['loss'])} | "
      f"Mejor val_loss: {min(h5.history['val_loss']):.4f}")
print(f"EarlyStopping patience=20 | Paró en época {len(h20.history['loss'])} | "
      f"Mejor val_loss: {min(h20.history['val_loss']):.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: EarlyStopping — patience=5 vs patience=20.*

1. `from tensorflow.keras.callbacks import EarlyStopping` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: ReduceLROnPlateau — Reducir lr cuando val_loss se Estanca

```python
from tensorflow.keras.callbacks import ReduceLROnPlateau

modelo_rlr = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(32, activation='relu'),
    Dense(1)
])
modelo_rlr.compile(optimizer=Adam(0.01), loss='mse')

rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                        min_lr=1e-6, min_delta=1e-4, cooldown=3, verbose=1)

h_rlr = modelo_rlr.fit(X_train, y_train, epochs=100, batch_size=32,
                       validation_data=(X_val, y_val), callbacks=[rlr], verbose=1)

# Mostrar cómo varió el lr
print(f"LR final: {modelo_rlr.optimizer.lr.numpy():.2e}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: ReduceLROnPlateau — Reducir lr cuando val_loss se Estanca.*

1. Mostrar cómo varió el lr

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: SGD con momentum=0.9 + nesterov=True

```python
from tensorflow.keras.optimizers import SGD

modelo_sgd = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(32, activation='relu'),
    Dense(1)
])
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
modelo_sgd.compile(optimizer=sgd, loss='mse')

h_sgd = modelo_sgd.fit(X_train, y_train, epochs=50, batch_size=32,
                       validation_data=(X_val, y_val), verbose=0)

# SGD con momentum acelera convergencia en direcciones consistentes
# nesterov: "mirada al futuro" → corrige overshooting
print(f"SGD + Nesterov val_loss: {min(h_sgd.history['val_loss']):.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: SGD con momentum=0.9 + nesterov=True.*

1. SGD con momentum acelera convergencia en direcciones consistentes
2. nesterov: "mirada al futuro" → corrige overshooting

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: Adam vs AdamW (con weight_decay)

```python
from tensorflow.keras.optimizers import AdamW

modelo_adam = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(32, activation='relu'),
    Dense(1)
])
modelo_adamw = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(32, activation='relu'),
    Dense(1)
])

modelo_adam.compile(optimizer=Adam(0.001), loss='mse')
modelo_adamw.compile(optimizer=AdamW(learning_rate=0.001, weight_decay=0.01), loss='mse')

h_adam = modelo_adam.fit(X_train, y_train, epochs=50, batch_size=32,
                         validation_data=(X_val, y_val), verbose=0)
h_adamw = modelo_adamw.fit(X_train, y_train, epochs=50, batch_size=32,
                           validation_data=(X_val, y_val), verbose=0)

print(f"Adam  val_loss: {min(h_adam.history['val_loss']):.4f}")
print(f"AdamW val_loss: {min(h_adamw.history['val_loss']):.4f}")
print("AdamW: weight_decay desacoplado del gradiente, mejor regularización")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: Adam vs AdamW (con weight_decay).*

1. `from tensorflow.keras.optimizers import AdamW` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: Adam vs Nadam vs RMSprop

```python
from tensorflow.keras.optimizers import Nadam, RMSprop

optimizadores = {
    'Adam': Adam(0.001),
    'Nadam': Nadam(0.001),
    'RMSprop': RMSprop(0.001)
}

histories_opt = {}
for nombre, opt in optimizadores.items():
    modelo = Sequential([
        Dense(64, activation='relu', input_shape=(10,)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    modelo.compile(optimizer=opt, loss='mse')
    h = modelo.fit(X_train, y_train, epochs=50, batch_size=32,
                   validation_data=(X_val, y_val), verbose=0)
    histories_opt[nombre] = h
    print(f"{nombre:10s} | Val loss: {min(h.history['val_loss']):.4f}")

for nombre in optimizadores:
    plt.plot(histories_opt[nombre].history['val_loss'], label=nombre)
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Comparación de optimizadores')
plt.legend()
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: Adam vs Nadam vs RMSprop.*

1. `from tensorflow.keras.optimizers import Nadam, RMSprop` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: Gradient Clipping — clipnorm=1.0 (Evitar Exploding Gradients)

```python
from tensorflow.keras.optimizers import SGD

# Red profunda propensa a exploding gradients
modelo_clip = Sequential([
    Dense(256, activation='relu', input_shape=(10,)),
    Dense(256, activation='relu'),
    Dense(256, activation='relu'),
    Dense(256, activation='relu'),
    Dense(1)
])
modelo_noclip = Sequential([
    Dense(256, activation='relu', input_shape=(10,)),
    Dense(256, activation='relu'),
    Dense(256, activation='relu'),
    Dense(256, activation='relu'),
    Dense(1)
])

optimizer_clip = Adam(learning_rate=0.01, clipnorm=1.0)
optimizer_noclip = Adam(learning_rate=0.01)

modelo_clip.compile(optimizer=optimizer_clip, loss='mse')
modelo_noclip.compile(optimizer=optimizer_noclip, loss='mse')

h_clip = modelo_clip.fit(X_train, y_train, epochs=30, batch_size=32,
                         validation_data=(X_val, y_val), verbose=0)
h_noclip = modelo_noclip.fit(X_train, y_train, epochs=30, batch_size=32,
                             validation_data=(X_val, y_val), verbose=0)

print(f"Con  clipnorm=1.0 | Val loss: {min(h_clip.history['val_loss']):.4f}")
print(f"Sin clipnorm     | Val loss: {min(h_noclip.history['val_loss']):.4f}")
# Sin clipping puede diverger (loss = NaN o valores enormes)

plt.plot(h_noclip.history['val_loss'], label='Sin clip', linestyle='--')
plt.plot(h_clip.history['val_loss'], label='Con clipnorm=1', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Gradient Clipping: previene exploding gradients')
plt.legend()
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: Gradient Clipping — clipnorm=1.0 (Evitar Exploding Gradients).*

1. Red profunda propensa a exploding gradients
2. Sin clipping puede diverger (loss = NaN o valores enormes)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: weight_decay en Optimizador (AdamW)

```python
# Comparar weight_decay en AdamW vs L2 regularización
from tensorflow.keras.regularizers import l2

# AdamW con weight_decay
modelo_wd = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(32, activation='relu'),
    Dense(1)
])
adamw = AdamW(learning_rate=0.001, weight_decay=0.01)
modelo_wd.compile(optimizer=adamw, loss='mse')

# Adam con L2 en kernel_regularizer
modelo_l2_adam = Sequential([
    Dense(64, activation='relu', input_shape=(10,),
          kernel_regularizer=l2(0.01)),
    Dense(32, activation='relu',
          kernel_regularizer=l2(0.01)),
    Dense(1)
])
modelo_l2_adam.compile(optimizer=Adam(0.001), loss='mse')

h_wd = modelo_wd.fit(X_train, y_train, epochs=50, batch_size=32,
                     validation_data=(X_val, y_val), verbose=0)
h_l2a = modelo_l2_adam.fit(X_train, y_train, epochs=50, batch_size=32,
                           validation_data=(X_val, y_val), verbose=0)

print(f"AdamW (weight_decay) val_loss: {min(h_wd.history['val_loss']):.4f}")
print(f"Adam + L2 regularizer val_loss: {min(h_l2a.history['val_loss']):.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: weight_decay en Optimizador (AdamW).*

1. Comparar weight_decay en AdamW vs L2 regularización
2. AdamW con weight_decay
3. Adam con L2 en kernel_regularizer

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: Comparar — Sin Regularización vs L2 vs Dropout vs BN vs Todas

```python
configs = {
    'Sin reg': {
        'layers': [Dense(256, activation='relu'), Dense(256, activation='relu')]
    },
    'L2': {
        'layers': [
            Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
            Dense(256, activation='relu', kernel_regularizer=l2(0.01))
        ]
    },
    'Dropout': {
        'layers': [
            Dense(256, activation='relu'), Dropout(0.3),
            Dense(256, activation='relu'), Dropout(0.3)
        ]
    },
    'BN': {
        'layers': [
            Dense(256, use_bias=False), BatchNormalization(), Activation('relu'),
            Dense(256, use_bias=False), BatchNormalization(), Activation('relu')
        ]
    },
    'Todas': {
        'layers': [
            Dense(256, use_bias=False, kernel_regularizer=l2(1e-4)),
            BatchNormalization(), Activation('relu'), Dropout(0.3),
            Dense(256, use_bias=False, kernel_regularizer=l2(1e-4)),
            BatchNormalization(), Activation('relu'), Dropout(0.3)
        ]
    }
}

histories_configs = {}
for nombre, cfg in configs.items():
    modelo = Sequential(cfg['layers'] + [Dense(1)])
    modelo.compile(optimizer=Adam(0.001), loss='mse')
    h = modelo.fit(X_train, y_train, epochs=80, batch_size=32,
                   validation_data=(X_val, y_val), verbose=0)
    histories_configs[nombre] = h
    print(f"{nombre:10s} | Train: {h.history['loss'][-1]:.4f} | Val: {h.history['val_loss'][-1]:.4f}")

for nombre in configs:
    plt.plot(histories_configs[nombre].history['val_loss'], label=nombre)
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Comparación de técnicas de regularización')
plt.legend()
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: Comparar — Sin Regularización vs L2 vs Dropout vs BN vs Todas.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 16: Curvas de Pérdida — Train vs Validation con Diferentes Regularizaciones

```python
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.ravel()

for i, (nombre, cfg) in enumerate(configs.items()):
    ax = axes[i]
    h = histories_configs[nombre]
    ax.plot(h.history['loss'], label='Train', alpha=0.7)
    ax.plot(h.history['val_loss'], label='Validation', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(nombre)
    ax.legend()
    ax.set_yscale('log')

axes[-1].axis('off')
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 16: Curvas de Pérdida — Train vs Validation con Diferentes Regularizaciones.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 17: Grid Search de Hiperparámetros de Regularización

```python
import itertools
from sklearn.metrics import mean_squared_error

# Grid search sobre dropout_rate y l2_lambda
dropout_rates = [0.0, 0.2, 0.4]
l2_lambdas = [0.0, 0.001, 0.01]

mejores_params = None
mejor_val_loss = float('inf')

for dropout_rate, l2_lambda in itertools.product(dropout_rates, l2_lambdas):
    modelo = Sequential([
        Dense(128, activation='relu', input_shape=(10,),
              kernel_regularizer=l2(l2_lambda)),
        Dropout(dropout_rate),
        Dense(64, activation='relu',
              kernel_regularizer=l2(l2_lambda)),
        Dropout(dropout_rate),
        Dense(1)
    ])
    modelo.compile(optimizer=Adam(0.001), loss='mse')
    h = modelo.fit(X_train, y_train, epochs=50, batch_size=32,
                   validation_data=(X_val, y_val), verbose=0)
    val_loss = min(h.history['val_loss'])
    print(f"dropout={dropout_rate:.1f}, L2={l2_lambda:.3f} → Val loss: {val_loss:.4f}")

    if val_loss < mejor_val_loss:
        mejor_val_loss = val_loss
        mejores_params = (dropout_rate, l2_lambda)

print(f"\nMejores parámetros: dropout={mejores_params[0]:.1f}, L2={mejores_params[1]:.3f}")
print(f"Mejor val_loss: {mejor_val_loss:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: Grid Search de Hiperparámetros de Regularización.*

1. Grid search sobre dropout_rate y l2_lambda

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 18: Integrador — Modelo Regularizado Óptimo

```python
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

np.random.seed(42)
tf.random.set_seed(42)

# Dataset completo de ventas con 15 features
n = 5000
X_full = np.random.randn(n, 15)
y_full = (2*X_full[:, 0] + 1.5*np.sin(X_full[:, 1]) +
          0.5*X_full[:, 2]**2 - X_full[:, 3] + 0.3*X_full[:, 4]*X_full[:, 5] +
          np.random.normal(0, 0.4, n))

Xf_train, Xf_val, yf_train, yf_val = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
scaler = StandardScaler()
Xf_train = scaler.fit_transform(Xf_train)
Xf_val = scaler.transform(Xf_val)

modelo_optimo = Sequential([
    Dense(128, input_shape=(15,),
          kernel_initializer='he_normal',
          kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
          use_bias=False),
    BatchNormalization(),
    Activation('selu'),
    Dropout(0.2),

    Dense(64,
          kernel_initializer='he_normal',
          kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
          use_bias=False),
    BatchNormalization(),
    Activation('selu'),
    Dropout(0.2),

    Dense(32,
          kernel_initializer='he_normal',
          kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
          use_bias=False),
    BatchNormalization(),
    Activation('selu'),

    Dense(1)
])

optimizer = AdamW(learning_rate=0.001, weight_decay=0.001)
modelo_optimo.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
modelo_optimo.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=15,
                  restore_best_weights=True, min_delta=1e-5,
                  start_from_epoch=10),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                      min_lr=1e-7, min_delta=1e-5, cooldown=3, verbose=1)
]

h_opt = modelo_optimo.fit(Xf_train, yf_train, epochs=200, batch_size=32,
                          validation_data=(Xf_val, yf_val),
                          callbacks=callbacks, verbose=1)

# Evaluación final
from sklearn.metrics import r2_score
y_pred_opt = modelo_optimo.predict(Xf_val, verbose=0)
r2 = r2_score(yf_val, y_pred_opt)
mse = mean_squared_error(yf_val, y_pred_opt)
print(f"\n=== Resultados Modelo Óptimo ===")
print(f"R²: {r2:.4f}")
print(f"MSE: {mse:.4f}")

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(h_opt.history['loss'], label='Train', alpha=0.7)
plt.plot(h_opt.history['val_loss'], label='Validation', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Curva de aprendizaje (modelo óptimo)')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(yf_val, y_pred_opt, alpha=0.4)
plt.plot([yf_val.min(), yf_val.max()], [yf_val.min(), yf_val.max()], 'r--')
plt.xlabel('Valor real')
plt.ylabel('Predicción')
plt.title(f'R² = {r2:.4f}')
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador — Modelo Regularizado Óptimo.*

1. Dataset completo de ventas con 15 features
2. Evaluación final

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejercicios

1. **Regularización L1 para selección de features en compras**: Entrena un modelo con 30 features de compras usando kernel_regularizer=l1(0.01). Después del entrenamiento, identifica qué features tienen pesos cercanos a cero (|w| < 0.01). ¿Cuántas features se "eliminaron" automáticamente?

2. **Comparación de dropout en capas de ventas**: Construye un MLP con 3 capas ocultas de 512 neuronas para predecir ventas. Prueba dropout rates [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6] y encuentra el rate óptimo. Grafica val_loss final vs dropout rate.

3. **BatchNormalization para acelerar entrenamiento en inventarios**: Entrena dos modelos idénticos (4 capas, 128 neuronas) en 20 features de inventario, uno con BN después de cada capa y otro sin BN. Usa learning_rate=0.01 y grafica la pérdida. ¿Cuántas épocas necesita cada uno para alcanzar val_loss < 0.1?

4. **Early stopping con diferentes patience**: Implementa 5 modelos con patience = [3, 5, 10, 20, 50]. Entrena cada uno en datos de ventas con máximo 500 épocas. Reporta: (a) época de parada, (b) val_loss final, (c) ¿paciencia muy pequeña causa parada prematura?

5. **ReduceLROnPlateau para optimizar learning rate**: Entrena un modelo de predicción de demanda con lr inicial=0.1. Usa ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-6). Grafica el learning rate vs epoch. ¿En qué épocas se redujo el lr? ¿Ayudó a mejorar val_loss?

6. **Optimizadores para clasificación de productos**: Construye un clasificador de 5 categorías con 3 capas ocultas. Compara SGD+momentum, Adam, Nadam, RMSprop en términos de accuracy de validación y velocidad de convergencia. ¿Cuál converge más rápido? ¿Cuál da mejor accuracy final?

7. **Gradient clipping para RNN de inventarios**: Crea un LSTM de 3 capas (128 cada una) para predicción de inventario. Entrena con y sin gradient clipping (clipnorm=1.0, clipnorm=5.0). ¿Hay diferencias en estabilidad? ¿Aparecen NaN sin clipping?

8. **Integrador: optimización completa de modelo de ventas**: Diseña un experimento completo: 4 capas ocultas (256→128→64→32), inicialización he_normal, regularización L1+L2 (1e-5, 1e-4), BatchNormalization, Dropout(0.25), EarlyStopping(patience=12), ReduceLROnPlateau(factor=0.3, patience=4), AdamW(weight_decay=0.001). Features: 20 de ventas/compras/inventario. Target: demanda semanal. Reporta: mejor val_loss, R², curvas de pérdida, y número de épocas reales de entrenamiento.

---

## Resumen

- **L1**: pesos dispersos (selección automática de features). **L2**: pesos pequeños. **L1+L2**: lo mejor de ambos.
- **Dropout**: ensemble implícito. rate=0.2-0.5 típico. SpatialDropout para datos correlacionados (CNNs).
- **BatchNormalization**: acelera convergencia, permite lr más altos, reduce dependencia de inicialización.
- **LayerNormalization**: alternativa para RNNs donde batch_size puede ser pequeño.
- **EarlyStopping**: patience=10-15 con restore_best_weights=True es práctica estándar.
- **ReduceLROnPlateau**: reduce lr automáticamente cuando validación se estanca.
- **Adam** es el optimizador default; **AdamW** mejora con weight_decay desacoplado.
- **Nadam** (Adam + Nesterov) converge más rápido en algunos problemas.
- **Gradient clipping** (clipnorm=1.0) esencial para RNNs y redes profundas.
- **Weight decay en optimizador** vs L2 en regularizer: AdamW aplica weight decay directamente al gradiente, separado del loss.
- La combinación sinérgica de múltiples técnicas (L2 + Dropout + BN + Early stopping + LR scheduling) da los mejores resultados.
- Siempre monitorear curvas train/val loss para diagnosticar overfitting y convergencia.
