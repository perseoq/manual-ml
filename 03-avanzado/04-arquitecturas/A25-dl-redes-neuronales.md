# A25 - Deep Learning: Redes Neuronales para Ventas, Compras e Inventarios

## Fundamentos Teóricos

Las redes neuronales artificiales (ANN) son modelos computacionales inspirados en el cerebro biológico. Consisten en neuronas artificiales organizadas en capas que aprenden representaciones jerárquicas de los datos.

### Perceptrón (Rosenblatt, 1958)
- Unidad básica: `y = σ(w·x + b)`
- Salida lineal o escalón para regresión/binaria
- Limitación: solo resuelve problemas linealmente separables

### MLP (Multi-Layer Perceptron)
- Múltiples capas ocultas con activaciones no lineales
- Capa de entrada (features), capas ocultas (transformaciones), capa de salida (predicción)
- Aprendizaje: backpropagation + gradiente descendente

### Funciones de Activación
| Activación | Ecuación | Rango | Uso |
|------------|----------|-------|-----|
| ReLU | max(0, x) | [0, ∞) | Capas ocultas (default) |
| Sigmoid | 1/(1+e⁻ˣ) | (0, 1) | Salida binaria |
| Tanh | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | (-1, 1) | Capas ocultas (RNN) |
| Linear | x | (-∞, ∞) | Regresión (salida) |
| Softmax | eˣ/Σeˣ | (0,1), suma=1 | Clasificación multiclase |
| ELU | x si x≥0, α(eˣ-1) si x<0 | (-α, ∞) | Alternativa a ReLU |
| SELU | Escalado de ELU | (-λα, ∞) | Auto-normalizante |
| GELU | x·Φ(x) | ≈(-∞, ∞) | State-of-art en transformers |
| Swish | x·σ(x) | (-∞, ∞) | Google Brain |

### Inicialización de Pesos
- **glorot_uniform**: √(6/(fan_in+fan_out)) — para tanh/sigmoid
- **he_normal**: √(2/fan_in) — para ReLU
- **lecun_normal**: √(1/fan_in) — para SELU
- **orthogonal**: matriz ortogonal — para RNNs

### Hiperparámetros Clave
- **batch_size**: muestras por actualización de gradiente
- **epochs**: pases completos sobre el dataset
- **learning_rate (lr)**: tamaño del paso de gradiente
- **Overfitting**: train_loss << val_loss
- **Underfitting**: train_loss alta, no aprende patrón
- **Vanishing gradient**: gradientes se hacen 0 en capas profundas
- **Exploding gradient**: gradientes crecen exponencialmente

### Técnicas de Regularización
- **Dropout(rate)**: apaga neuronas aleatoriamente en cada paso
- **BatchNormalization**: normaliza activaciones por batch
- **L2 regularization**: penaliza pesos grandes (weight decay)
- **Early stopping**: detiene entrenamiento cuando val_loss deja de mejorar

---

## Ejemplos Prácticos

### Ejemplo 1: Perceptrón — Regresión Lineal (Precio → Demanda)

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

np.random.seed(42)
tf.random.set_seed(42)

# Datos: precio → demanda (relación lineal negativa)
precio = np.linspace(10, 100, 500).reshape(-1, 1)
demanda = 500 - 4.5 * precio + np.random.normal(0, 25, (500, 1))

# Normalizar
precio_norm = (precio - precio.mean()) / precio.std()
demanda_norm = (demanda - demanda.mean()) / demanda.std()

# Perceptrón: 1 neurona, sin activación (regresión lineal)
modelo = Sequential([
    Dense(1, activation='linear', input_shape=(1,),
          kernel_initializer='glorot_uniform')
])
modelo.compile(optimizer=SGD(learning_rate=0.01), loss='mse')
modelo.summary()

history = modelo.fit(precio_norm, demanda_norm, epochs=50,
                     batch_size=32, validation_split=0.2, verbose=0)

# Predicción
pred = modelo.predict(precio_norm, verbose=0)
plt.scatter(precio, demanda, alpha=0.3, label='Real')
plt.plot(precio, pred * demanda.std() + demanda.mean(), 'r-', linewidth=2, label='Perceptrón')
plt.xlabel('Precio ($)')
plt.ylabel('Demanda (unidades)')
plt.title('Perceptrón: Precio → Demanda')
plt.legend()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 1: Perceptrón — Regresión Lineal (Precio → Demanda).*

1. Datos: precio → demanda (relación lineal negativa)
2. Normalizar
3. Perceptrón: 1 neurona, sin activación (regresión lineal)
4. Predicción

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: MLP 2 Capas [64 relu, 1 linear] — Predicción de Ventas

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

np.random.seed(42)
tf.random.set_seed(42)

# Dataset sintético de ventas: [precio, gasto_publicidad, día_semana, temporada]
n = 2000
X = np.random.randn(n, 8)
y = (3*X[:, 0] - 1.5*X[:, 1] + 2*X[:, 2]**2 + 0.5*X[:, 3] +
     np.sin(X[:, 4]) + np.random.normal(0, 0.5, n))

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

modelo = Sequential([
    Dense(64, activation='relu', input_shape=(8,),
          kernel_initializer='he_normal'),
    Dense(1, activation='linear')
])
modelo.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
modelo.summary()

history = modelo.fit(X_train, y_train, epochs=100, batch_size=32,
                     validation_data=(X_val, y_val), verbose=1)
```

**Salida:**

```
# El modelo ajusta sus parámetros durante el entrenamiento.
# Los parámetros aprendidos quedan disponibles en el objeto.
```

**Explicación línea por línea:**

*Ejemplo 2: MLP 2 Capas [64 relu, 1 linear] — Predicción de Ventas.*

1. Dataset sintético de ventas: [precio, gasto_publicidad, día_semana, temporada]

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: MLP 3 Capas [128 relu, 64 relu, 1 linear] — Más Profunda

```python
modelo_profundo = Sequential([
    Dense(128, activation='relu', input_shape=(8,),
          kernel_initializer='he_normal'),
    Dense(64, activation='relu', kernel_initializer='he_normal'),
    Dense(1, activation='linear')
])
modelo_profundo.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
modelo_profundo.summary()

history_profundo = modelo_profundo.fit(
    X_train, y_train, epochs=100, batch_size=32,
    validation_data=(X_val, y_val), verbose=1
)
```

**Salida:**

```
# El modelo ajusta sus parámetros durante el entrenamiento.
# Los parámetros aprendidos quedan disponibles en el objeto.
```

**Explicación línea por línea:**

*Ejemplo 3: MLP 3 Capas [128 relu, 64 relu, 1 linear] — Más Profunda.*

1. `history_profundo = modelo_profundo.fit(` — Entrena el modelo con los datos de entrenamiento.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: Comparar Activaciones — ReLU vs Tanh vs Sigmoid

```python
activaciones = ['relu', 'tanh', 'sigmoid']
histories_act = {}

for act in activaciones:
    modelo = Sequential([
        Dense(64, activation=act, input_shape=(8,)),
        Dense(32, activation=act),
        Dense(1, activation='linear')
    ])
    modelo.compile(optimizer=Adam(0.001), loss='mse')
    h = modelo.fit(X_train, y_train, epochs=50, batch_size=32,
                   validation_data=(X_val, y_val), verbose=0)
    histories_act[act] = h

plt.figure(figsize=(12, 4))
for i, act in enumerate(activaciones):
    plt.subplot(1, 3, i+1)
    plt.plot(histories_act[act].history['loss'], label='train')
    plt.plot(histories_act[act].history['val_loss'], label='val')
    plt.title(f'Activación: {act}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
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

*Ejemplo 4: Comparar Activaciones — ReLU vs Tanh vs Sigmoid.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: Inicialización — glorot_uniform vs he_normal

```python
inits = ['glorot_uniform', 'he_normal']
histories_init = {}

for init in inits:
    modelo = Sequential([
        Dense(64, activation='relu', input_shape=(8,),
              kernel_initializer=init),
        Dense(32, activation='relu', kernel_initializer=init),
        Dense(1, activation='linear')
    ])
    modelo.compile(optimizer=Adam(0.001), loss='mse')
    h = modelo.fit(X_train, y_train, epochs=50, batch_size=32,
                   validation_data=(X_val, y_val), verbose=0)
    histories_init[init] = h

for init in inits:
    plt.plot(histories_init[init].history['val_loss'], label=init)
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Efecto de la inicialización en convergencia')
plt.legend()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 5: Inicialización — glorot_uniform vs he_normal.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: learning_rate — 0.1 vs 0.01 vs 0.001 vs 0.0001

```python
lrs = [0.1, 0.01, 0.001, 0.0001]
histories_lr = {}

for lr in lrs:
    modelo = Sequential([
        Dense(64, activation='relu', input_shape=(8,)),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    modelo.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    h = modelo.fit(X_train, y_train, epochs=30, batch_size=32,
                   validation_data=(X_val, y_val), verbose=0)
    histories_lr[lr] = h

for lr in lrs:
    plt.plot(histories_lr[lr].history['val_loss'], label=f'lr={lr}')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Efecto del learning rate en convergencia')
plt.yscale('log')
plt.legend()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 6: learning_rate — 0.1 vs 0.01 vs 0.001 vs 0.0001.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: batch_size — 8 vs 32 vs 128 vs 512

```python
batch_sizes = [8, 32, 128, 512]
histories_bs = {}

for bs in batch_sizes:
    modelo = Sequential([
        Dense(64, activation='relu', input_shape=(8,)),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    modelo.compile(optimizer=Adam(0.001), loss='mse')
    h = modelo.fit(X_train, y_train, epochs=30, batch_size=bs,
                   validation_data=(X_val, y_val), verbose=0)
    histories_bs[bs] = h
    print(f"batch_size={bs:3d} | Train loss final: {h.history['loss'][-1]:.4f} | Val loss: {h.history['val_loss'][-1]:.4f}")

for bs in batch_sizes:
    plt.plot(histories_bs[bs].history['val_loss'], label=f'batch={bs}')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Efecto del batch size')
plt.legend()
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: batch_size — 8 vs 32 vs 128 vs 512.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: Epochs — Suficientes vs Overfitting (Curva de Pérdida)

```python
modelo_over = Sequential([
    Dense(256, activation='relu', input_shape=(8,)),
    Dense(256, activation='relu'),
    Dense(256, activation='relu'),
    Dense(1, activation='linear')
])
modelo_over.compile(optimizer=Adam(0.001), loss='mse')
h_over = modelo_over.fit(X_train, y_train, epochs=200, batch_size=16,
                         validation_data=(X_val, y_val), verbose=0)

plt.plot(h_over.history['loss'], label='train_loss', alpha=0.8)
plt.plot(h_over.history['val_loss'], label='val_loss', alpha=0.8)
plt.axvline(x=np.argmin(h_over.history['val_loss']), color='k', linestyle='--',
            label=f'best val_loss epoch={np.argmin(h_over.history["val_loss"])}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Overfitting: train_loss << val_loss')
plt.legend()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 8: Epochs — Suficientes vs Overfitting (Curva de Pérdida).*

1. `h_over = modelo_over.fit(X_train, y_train, epochs=200, batch_size=16,` — Entrena el modelo con los datos de entrenamiento.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: Dropout(0.2) vs Dropout(0.5) — Regularización

```python
from tensorflow.keras.layers import Dropout

rates = [0.0, 0.2, 0.5]
histories_drop = {}

for rate in rates:
    modelo = Sequential([
        Dense(256, activation='relu', input_shape=(8,)),
        Dropout(rate),
        Dense(256, activation='relu'),
        Dropout(rate),
        Dense(1, activation='linear')
    ])
    modelo.compile(optimizer=Adam(0.001), loss='mse')
    h = modelo.fit(X_train, y_train, epochs=100, batch_size=32,
                   validation_data=(X_val, y_val), verbose=0)
    histories_drop[rate] = h

for rate in rates:
    plt.plot(histories_drop[rate].history['val_loss'], label=f'dropout={rate}')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Dropout: regularización')
plt.legend()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 9: Dropout(0.2) vs Dropout(0.5) — Regularización.*

1. `from tensorflow.keras.layers import Dropout` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: BatchNormalization — Antes vs Después de Activación

```python
from tensorflow.keras.layers import BatchNormalization

# BN antes de activación
model_bn_before = Sequential([
    Dense(64, input_shape=(8,), use_bias=False),
    BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    Dense(32, use_bias=False),
    BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    Dense(1, activation='linear')
])

# BN después de activación
model_bn_after = Sequential([
    Dense(64, activation='relu', input_shape=(8,)),
    BatchNormalization(),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dense(1, activation='linear')
])

model_bn_before.compile(optimizer=Adam(0.001), loss='mse')
model_bn_after.compile(optimizer=Adam(0.001), loss='mse')

h_before = model_bn_before.fit(X_train, y_train, epochs=50, batch_size=32,
                               validation_data=(X_val, y_val), verbose=0)
h_after = model_bn_after.fit(X_train, y_train, epochs=50, batch_size=32,
                             validation_data=(X_val, y_val), verbose=0)

plt.plot(h_before.history['val_loss'], label='BN antes de activación')
plt.plot(h_after.history['val_loss'], label='BN después de activación')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.legend()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 10: BatchNormalization — Antes vs Después de Activación.*

1. BN antes de activación
2. BN después de activación

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: Early Stopping — Patience=10

```python
from tensorflow.keras.callbacks import EarlyStopping

modelo_es = Sequential([
    Dense(128, activation='relu', input_shape=(8,)),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')
])
modelo_es.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])

early_stop = EarlyStopping(monitor='val_loss', patience=10,
                           restore_best_weights=True, verbose=1)

h_es = modelo_es.fit(X_train, y_train, epochs=500, batch_size=32,
                     validation_data=(X_val, y_val),
                     callbacks=[early_stop], verbose=1)

print(f"Entrenamiento detenido en epoch {len(h_es.history['loss'])}")
print(f"Mejor val_loss: {min(h_es.history['val_loss']):.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: Early Stopping — Patience=10.*

1. `from tensorflow.keras.callbacks import EarlyStopping` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: Underfitting — Red Pequeña (4 Neuronas)

```python
# Red con solo 4 neuronas → demasiado simple para el patrón
modelo_under = Sequential([
    Dense(4, activation='relu', input_shape=(8,)),
    Dense(4, activation='relu'),
    Dense(1, activation='linear')
])
modelo_under.compile(optimizer=Adam(0.01), loss='mse')
h_under = modelo_under.fit(X_train, y_train, epochs=100, batch_size=32,
                           validation_data=(X_val, y_val), verbose=0)

print(f"Underfitting - Train loss: {h_under.history['loss'][-1]:.4f}, "
      f"Val loss: {h_under.history['val_loss'][-1]:.4f}")
print(f"Esperado: pérdida relativamente alta (no captura patrón)")

plt.plot(h_under.history['loss'], label='train')
plt.plot(h_under.history['val_loss'], label='val')
plt.title('Underfitting: red con 4 neuronas')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: Underfitting — Red Pequeña (4 Neuronas).*

1. Red con solo 4 neuronas → demasiado simple para el patrón

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: Overfitting — Red Grande (1024 Neuronas) con Pocos Datos

```python
# Pocos datos para exagerar overfitting
X_peq, _, y_peq, _ = train_test_split(X, y, train_size=100, random_state=42)
X_peq = scaler.fit_transform(X_peq)

modelo_grande = Sequential([
    Dense(1024, activation='relu', input_shape=(8,)),
    Dense(1024, activation='relu'),
    Dense(512, activation='relu'),
    Dense(1, activation='linear')
])
modelo_grande.compile(optimizer=Adam(0.001), loss='mse')
h_grande = modelo_grande.fit(X_peq, y_peq, epochs=200, batch_size=8,
                             validation_data=(X_val, y_val), verbose=0)

print(f"Train loss final: {h_grande.history['loss'][-1]:.4f}")
print(f"Val loss final: {h_grande.history['val_loss'][-1]:.4f}")
print("Diferencia grande indica overfitting severo")

plt.plot(h_grande.history['loss'], label='train')
plt.plot(h_grande.history['val_loss'], label='val')
plt.title('Overfitting: 1024 neuronas con pocos datos')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: Overfitting — Red Grande (1024 Neuronas) con Pocos Datos.*

1. Pocos datos para exagerar overfitting

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: Learning Rate Schedule — Reducir lr Durante Entrenamiento

```python
from tensorflow.keras.callbacks import LearningRateScheduler

def scheduler(epoch, lr):
    if epoch < 30:
        return lr
    elif epoch < 60:
        return lr * 0.5
    elif epoch < 80:
        return lr * 0.1
    else:
        return lr * 0.01

modelo_lrs = Sequential([
    Dense(64, activation='relu', input_shape=(8,)),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])
modelo_lrs.compile(optimizer=Adam(0.01), loss='mse')

lr_scheduler = LearningRateScheduler(scheduler, verbose=1)
h_lrs = modelo_lrs.fit(X_train, y_train, epochs=100, batch_size=32,
                       validation_data=(X_val, y_val),
                       callbacks=[lr_scheduler], verbose=1)
```

**Salida:**

```
# El modelo ajusta sus parámetros durante el entrenamiento.
# Los parámetros aprendidos quedan disponibles en el objeto.
```

**Explicación línea por línea:**

*Ejemplo 14: Learning Rate Schedule — Reducir lr Durante Entrenamiento.*

1. `from tensorflow.keras.callbacks import LearningRateScheduler` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: L2 Regularization — kernel_regularizer

```python
from tensorflow.keras.regularizers import l2

modelo_l2 = Sequential([
    Dense(64, activation='relu', input_shape=(8,),
          kernel_regularizer=l2(0.01)),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(1, activation='linear')
])
modelo_l2.compile(optimizer=Adam(0.001), loss='mse')
h_l2 = modelo_l2.fit(X_train, y_train, epochs=100, batch_size=32,
                     validation_data=(X_val, y_val), verbose=0)

plt.plot(h_l2.history['loss'], label='train (L2)')
plt.plot(h_l2.history['val_loss'], label='val (L2)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Regularización L2')
plt.legend()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 15: L2 Regularization — kernel_regularizer.*

1. `from tensorflow.keras.regularizers import l2` — Importa las librerías necesarias para el análisis.
2. `h_l2 = modelo_l2.fit(X_train, y_train, epochs=100, batch_size=32,` — Entrena el modelo con los datos de entrenamiento.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 16: Comparar — Con y Sin Regularización (Dropout + BN + L2)

```python
# Modelo sin regularización
model_sin_reg = Sequential([
    Dense(256, activation='relu', input_shape=(8,)),
    Dense(256, activation='relu'),
    Dense(1, activation='linear')
])

# Modelo con regularización completa
model_con_reg = Sequential([
    Dense(256, input_shape=(8,), kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    Dropout(0.3),
    Dense(256, kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    Dropout(0.3),
    Dense(1, activation='linear')
])

model_sin_reg.compile(optimizer=Adam(0.001), loss='mse')
model_con_reg.compile(optimizer=Adam(0.001), loss='mse')

h_sin = model_sin_reg.fit(X_train, y_train, epochs=100, batch_size=32,
                          validation_data=(X_val, y_val), verbose=0)
h_con = model_con_reg.fit(X_train, y_train, epochs=100, batch_size=32,
                          validation_data=(X_val, y_val), verbose=0)

plt.plot(h_sin.history['val_loss'], label='Sin regularización', linestyle='--')
plt.plot(h_con.history['val_loss'], label='Con regularización', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Comparación: con vs sin regularización')
plt.legend()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 16: Comparar — Con y Sin Regularización (Dropout + BN + L2).*

1. Modelo sin regularización
2. Modelo con regularización completa

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 17: Curva de Aprendizaje — Pérdida Train vs Validation

```python
from sklearn.metrics import mean_squared_error, r2_score

# Entrenar modelo final
modelo_final = Sequential([
    Dense(64, activation='relu', input_shape=(8,),
          kernel_initializer='he_normal'),
    Dense(32, activation='relu', kernel_initializer='he_normal'),
    Dense(1, activation='linear')
])
modelo_final.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])

h_final = modelo_final.fit(X_train, y_train, epochs=80, batch_size=32,
                           validation_data=(X_val, y_val), verbose=0)

# Curvas
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(h_final.history['loss'], label='Train')
axes[0].plot(h_final.history['val_loss'], label='Validation')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (MSE)')
axes[0].set_title('Curva de pérdida')
axes[0].legend()

axes[1].plot(h_final.history['mae'], label='Train')
axes[1].plot(h_final.history['val_mae'], label='Validation')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MAE')
axes[1].set_title('Error absoluto medio')
axes[1].legend()
plt.tight_layout()
plt.show()

# Evaluación final
y_pred = modelo_final.predict(X_val, verbose=0)
print(f"R² Score: {r2_score(y_val, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_val, y_pred):.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: Curva de Aprendizaje — Pérdida Train vs Validation.*

1. Entrenar modelo final
2. Curvas
3. Evaluación final

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 18: Integrador — MLP Optimizada para Predicción de Demanda

```python
"""
Modelo integrador: MLP optimizada para predicción de demanda
Aplica todas las técnicas aprendidas de forma combinada.
"""
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Dataset completo: 20 features de ventas
n_muestras = 5000
X_full = np.random.randn(n_muestras, 20)
y_full = (2.5 * X_full[:, 0] - 1.8 * X_full[:, 1] + 3.0 * X_full[:, 2]**2 +
          0.7 * np.sin(X_full[:, 3]) + 0.5 * X_full[:, 4] * X_full[:, 5] +
          np.random.normal(0, 0.4, n_muestras))

Xtr, Xte, ytr, yte = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
scaler = StandardScaler()
Xtr = scaler.fit_transform(Xtr)
Xte = scaler.transform(Xte)

modelo_integrador = Sequential([
    Dense(128, input_shape=(20,),
          kernel_initializer='he_normal',
          kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
    BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    Dropout(0.2),

    Dense(64, kernel_initializer='he_normal',
          kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
    BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    Dropout(0.2),

    Dense(32, kernel_initializer='he_normal',
          kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
    BatchNormalization(),
    tf.keras.layers.Activation('relu'),

    Dense(1, activation='linear')
])

optimizer = Adam(learning_rate=0.001)
modelo_integrador.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
modelo_integrador.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                      min_lr=1e-6, verbose=1)
]

h_int = modelo_integrador.fit(Xtr, ytr, epochs=200, batch_size=32,
                              validation_data=(Xte, yte),
                              callbacks=callbacks, verbose=1)

# Resultados
y_pred_int = modelo_integrador.predict(Xte, verbose=0)
r2 = r2_score(yte, y_pred_int)
mse = mean_squared_error(yte, y_pred_int)
print(f"\n=== Resultados Modelo Integrador ===")
print(f"R²: {r2:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {np.sqrt(mse):.4f}")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(h_int.history['loss'], label='Train', alpha=0.7)
plt.plot(h_int.history['val_loss'], label='Validation', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Curva de aprendizaje')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(yte, y_pred_int, alpha=0.4)
plt.plot([yte.min(), yte.max()], [yte.min(), yte.max()], 'r--')
plt.xlabel('Valor real')
plt.ylabel('Predicción')
plt.title(f'Predicciones vs Reales (R²={r2:.3f})')
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador — MLP Optimizada para Predicción de Demanda.*

1. Dataset completo: 20 features de ventas
2. Resultados

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejercicios

1. **Perceptrón para compras**: Construye un perceptrón que prediga el volumen de compras basado en el precio unitario usando datos simulados con relación `compras = 1000 - 8 * precio + ruido`. Normaliza los datos y entrena con SGD.

2. **MLP para inventario**: Crea un MLP de 3 capas ([64, 32, 16] con ReLU) que prediga el nivel óptimo de inventario usando 6 features: ventas_históricas, lead_time, costo_pedido, costo_mantenimiento, demanda_promedio, variabilidad_demanda.

3. **Comparación de activaciones en clasificación**: Usando un dataset de 4 categorías de productos (electrónicos, ropa, alimentos, libros) con 12 features, compara ReLU vs Tanh vs Swish en capas ocultas con salida softmax. Reporta accuracy de validación.

4. **Exploración de learning rate**: Implementa un grid search sobre learning_rates = [0.1, 0.05, 0.01, 0.005, 0.001] para un MLP que predice ventas semanales. Encuentra el lr óptimo y muestra las curvas de pérdida.

5. **Dropout en red de inventarios**: Construye un MLP con 3 capas ocultas de 256 neuronas. Prueba dropout rates [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] en cada capa. ¿Cuál da mejor val_loss? ¿Hay overfitting sin dropout?

6. **Early stopping + BatchNormalization**: Para un modelo de predicción de compras con 100 epochs máximo, usa EarlyStopping(patience=8) y BatchNormalization después de cada capa oculta. Compara con una versión sin BN ni early stopping.

7. **Diagnóstico de underfitting/overfitting**: Genera 3 modelos con 4, 64 y 1024 neuronas por capa. Entrena cada uno en solo 200 muestras. Grafica train_loss vs val_loss para cada uno. Identifica underfitting y overfitting.

8. **Integrador de punto de reorden**: Diseña un MLP completo para predecir el punto de reorden óptimo. Incluye: 3 capas ocultas, BatchNormalization, Dropout(0.25), regularización L2(1e-4), early stopping con patience=12, ReduceLROnPlateau, y inicialización he_normal. Features: ventas_diarias_promedio, lead_time, desviación_estándar_demanda, costo_pedido, costo_mantenimiento, días_stock_seguridad.

---

## Resumen

- El **perceptrón** es la unidad fundamental, limitado a problemas lineales.
- **MLP** con múltiples capas y activaciones no lineales resuelve problemas complejos de ventas/compras/inventarios.
- **ReLU** es la activación default para capas ocultas; **softmax** para clasificación; **linear** para regresión.
- La **inicialización** correcta (he_normal para ReLU, glorot_uniform para tanh/sigmoid) es crítica para convergencia.
- **batch_size** pequeños (8-32) convergen más rápido pero son ruidosos; grandes (128-512) son estables pero lentos.
- **learning_rate** balancea velocidad y estabilidad: muy alto diverge, muy bajo no converge.
- **Overfitting** se detecta cuando train_loss << val_loss; se combate con dropout, BN, L2, early stopping.
- **Underfitting** ocurre con redes muy pequeñas o entrenamiento insuficiente.
- **BatchNormalization** acelera convergencia y permite learning rates más altos.
- **Dropout** es regularización por ensemble implícito; típicamente 0.2-0.5.
- **Early stopping** con patience=10-15 es práctica estándar.
- La combinación de múltiples técnicas de regularización da los mejores resultados en datos reales de ventas.
