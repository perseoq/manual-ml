# E10: LSTM para Pronóstico de Demanda y Ventas

## 1. Fundamentos Teóricos

### 1.1 ¿Por qué LSTM para series temporales?

Las redes LSTM (Long Short-Term Memory) son ideales para secuencias porque:
- Capturan dependencias de largo plazo (a diferencia de ARIMA que tiene orden fijo)
- Modelan relaciones no lineales complejas
- Manejan múltiples features de entrada (precio, promociones, temporada)
- Aprenden representaciones directamente de los datos

### 1.2 Arquitecturas principales

| Arquitectura | Uso |
|-------------|-----|
| **Vanilla LSTM** | Forecast 1 paso adelante (one-step) |
| **Stacked LSTM** | Múltiples capas para mayor capacidad |
| **Encoder-Decoder** | Multi-step forecast (secuencia a secuencia) |
| **Bidirectional LSTM** | Contexto pasado y futuro |
| **LSTM + Attention** | Enfoque selectivo en partes de la secuencia |
| **ConvLSTM** | Patrones espaciales + temporales |

### 1.3 Estrategias de multi-step forecast

- **Direct**: un modelo por cada paso futuro (n modelos)
- **Recursive**: usar predicción como input para el siguiente paso
- **Direct-Recursive Hybrid**: combina ambos enfoques
- **Seq2Seq**: encoder-decoder genera toda la secuencia de una vez

### 1.4 Preprocesamiento clave

- Normalización de features (MinMaxScaler recomendado)
- Creación de ventanas deslizantes (lookback window)
- División temporal (no aleatoria) en train/validation/test

---

## 2. Ejemplos Prácticos

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, RepeatVector, TimeDistributed,
    Bidirectional, Input, Concatenate, Attention, Flatten,
    BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 5)

np.random.seed(42)
tf.random.set_seed(42)

# Simular datos de ventas
n = 365 * 3  # 3 años
dates = pd.date_range('2022-01-01', periods=n, freq='D')
t = np.arange(n)

tendencia = 100 + 0.05 * t + 0.0001 * t**2
estacional = 25 * np.sin(2 * np.pi * t / 7) + 30 * np.sin(2 * np.pi * t / 365.25)
efecto_precio = -0.2 * (50 + 10 * np.sin(2 * np.pi * t / 180) + np.random.normal(0, 2, n) - 50)
ruido = np.random.normal(0, 12, n)

ventas = tendencia + estacional + efecto_precio + ruido
ventas = np.maximum(ventas, 5)

df = pd.DataFrame({
    'fecha': dates,
    'ventas': ventas,
    'precio': 50 + 10 * np.sin(2 * np.pi * t / 180) + np.random.normal(0, 2, n),
    'dia_semana': dates.dayofweek,
    'mes': dates.month
})

print(f"Datos: {len(df)} días")
print(f"Ventas: media={df['ventas'].mean():.1f}, std={df['ventas'].std():.1f}")

# Preprocesamiento
scaler_y = MinMaxScaler()
scaler_X = MinMaxScaler()

y_scaled = scaler_y.fit_transform(df[['ventas']]).flatten()
features = ['precio', 'dia_semana', 'mes']
X_raw = df[features].values
X_scaled = scaler_X.fit_transform(X_raw)

print(f"Features escaladas: {features}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*2. Ejemplos Prácticos.*

1. Simular datos de ventas, compras o inventarios
2. Preprocesamiento

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 1: Crear ventana deslizante de 30 días

```python
def crear_ventanas(data, window_size=30, step=1):
    """Crear secuencias de ventanas deslizantes para LSTM."""
    X, y = [], []
    for i in range(0, len(data) - window_size, step):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

window = 30
X, y = crear_ventanas(y_scaled, window_size=window)

print("=== Ventana Deslizante ===")
print(f"Window size: {window} días")
print(f"Input shape: {X.shape}")
print(f"Output shape: {y.shape}")
print(f"X[0] shape: {X[0].shape} (30 días, 1 feature)")
print(f"y[0]: {y[0]:.4f} (día 31)")

# División temporal
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

# Reshape para LSTM: (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
print(f"Train shape after reshape: {X_train.shape} (samples, timesteps, features)")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: Crear ventana deslizante de 30 días.*

1. División temporal
2. Reshape para LSTM: (samples, timesteps, features)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: Dataset de secuencias - estructura

```python
print("=== Estructura del Dataset LSTM ===")
print(f"X.shape = (n_muestras, n_timesteps, n_features)")
print(f"X.shape = {X_train.shape}")
print(f"  → {X_train.shape[0]} muestras de entrenamiento")
print(f"  → {X_train.shape[1]} pasos temporales cada una")
print(f"  → {X_train.shape[2]} feature(s) por paso")

print(f"\ny.shape = (n_muestras,)")
print(f"y.shape = {y_train.shape}")

# Mostrar una secuencia de ejemplo
print("\nEjemplo de secuencia (primeros 10 valores de la muestra 0):")
for i in range(10):
    print(f"  t={i}: ventas_norm={X_train[0, i, 0]:.4f}")
print(f"  → target: y={y_train[0]:.4f}")

# Revertir escala para entender
y_real = scaler_y.inverse_transform(y_train[0].reshape(-1, 1))[0, 0]
print(f"\n  → Target (desescalado): {y_real:.1f} unidades")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: Dataset de secuencias - estructura.*

1. Mostrar una secuencia de ejemplo
2. Revertir escala para entender

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: LSTM básico para forecast

```python
modelo_lstm = Sequential([
    LSTM(50, activation='tanh', return_sequences=False, input_shape=(window, 1)),
    Dense(1)
])

modelo_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

print("=== LSTM Básico ===")
modelo_lstm.summary()

history = modelo_lstm.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    verbose=0,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
)

# Evaluar
pred_train = modelo_lstm.predict(X_train, verbose=0)
pred_test = modelo_lstm.predict(X_test, verbose=0)

pred_train_real = scaler_y.inverse_transform(pred_train)
pred_test_real = scaler_y.inverse_transform(pred_test)
y_train_real = scaler_y.inverse_transform(y_train.reshape(-1, 1))
y_test_real = scaler_y.inverse_transform(y_test.reshape(-1, 1))

rmse_train = np.sqrt(mean_squared_error(y_train_real, pred_train_real))
rmse_test = np.sqrt(mean_squared_error(y_test_real, pred_test_real))
mape_test = np.mean(np.abs((y_test_real - pred_test_real) / y_test_real)) * 100

print(f"RMSE Train: {rmse_train:.2f}")
print(f"RMSE Test: {rmse_test:.2f}")
print(f"MAPE Test: {mape_test:.2f}%")

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
axes[0].plot(history.history['loss'], label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Val Loss')
axes[0].set_title('Curva de Aprendizaje')
axes[0].legend()

axes[1].plot(y_test_real[:100], label='Real', linewidth=2)
axes[1].plot(pred_test_real[:100], label='Predicho', linewidth=2, alpha=0.8)
axes[1].set_title('LSTM: Predicciones vs Reales (primeros 100 test)')
axes[1].legend()
plt.tight_layout()
plt.savefig('e10_ej3_lstm_basic.png', dpi=100)
plt.close()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: LSTM básico para forecast.*

1. Evaluar

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: LSTM con return_sequences para apilar capas

```python
modelo_stacked = Sequential([
    LSTM(50, activation='tanh', return_sequences=True, input_shape=(window, 1)),
    LSTM(50, activation='tanh', return_sequences=False),
    Dense(1)
])

modelo_stacked.compile(optimizer=Adam(0.001), loss='mse')

print("=== LSTM Apilado (Stacked) ===")
modelo_stacked.summary()

history2 = modelo_stacked.fit(
    X_train, y_train,
    epochs=30, batch_size=32,
    validation_split=0.1, verbose=0,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
)

pred_test_s = scaler_y.inverse_transform(modelo_stacked.predict(X_test, verbose=0))
rmse_s = np.sqrt(mean_squared_error(y_test_real, pred_test_s))
print(f"RMSE Test (stacked): {rmse_s:.2f}")

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(history2.history['loss'], label='Stacked LSTM loss')
ax.plot(history.history['loss'], label='Simple LSTM loss')
ax.set_title('Comparación de Convergencia')
ax.legend()
plt.tight_layout()
plt.savefig('e10_ej4_stacked.png', dpi=100)
plt.close()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: LSTM con return_sequences para apilar capas.*

1. `print("=== LSTM Apilado (Stacked) ===")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: Stacked LSTM con 2 capas

```python
modelo_2lstm = Sequential([
    LSTM(100, activation='tanh', return_sequences=True, input_shape=(window, 1)),
    Dropout(0.2),
    LSTM(50, activation='tanh', return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)
])

modelo_2lstm.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])

print("=== Stacked LSTM con 2 Capas + Dropout ===")
modelo_2lstm.summary()

history3 = modelo_2lstm.fit(
    X_train, y_train,
    epochs=40, batch_size=32,
    validation_split=0.1, verbose=0,
    callbacks=[EarlyStopping(patience=8, restore_best_weights=True)]
)

pred_test_2 = scaler_y.inverse_transform(modelo_2lstm.predict(X_test, verbose=0))
rmse_2 = np.sqrt(mean_squared_error(y_test_real, pred_test_2))
mape_2 = np.mean(np.abs((y_test_real - pred_test_2) / y_test_real)) * 100

print(f"RMSE Test: {rmse_2:.2f}")
print(f"MAPE Test: {mape_2:.2f}%")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: Stacked LSTM con 2 capas.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: Walk-forward validation

```python
print("=== Walk-Forward Validation ===")

def walk_forward_lstm(data, window=30, train_size=500, step=7):
    """Walk-forward validation para LSTM."""
    errors = []
    current_train = data[:train_size]
    
    for i in range(train_size, len(data) - window, step):
        # Crear ventanas de entrenamiento
        X_wf, y_wf = crear_ventanas(current_train, window_size=window)
        X_wf = X_wf.reshape((X_wf.shape[0], X_wf.shape[1], 1))
        
        # Entrenar modelo
        model = Sequential([
            LSTM(30, input_shape=(window, 1), return_sequences=False),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_wf, y_wf, epochs=10, batch_size=32, verbose=0)
        
        # Predecir next step
        last_window = data[i:i+window].reshape(1, window, 1)
        pred = model.predict(last_window, verbose=0)[0, 0]
        actual = data[i+window]
        
        error = abs(actual - pred)
        errors.append(error)
        current_train = np.concatenate([current_train, [actual]])
        
        if (i - train_size) % (step * 10) == 0:
            print(f"  Paso {i-train_size}: MAE acumulado={np.mean(errors):.4f}")
    
    return errors

# Usar submuestra para rapidez
errors_wf = walk_forward_lstm(y_scaled[:800], window=30, train_size=400, step=14)
print(f"\nWalk-Forward MAE promedio: {np.mean(errors_wf):.4f} (normalizado)")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: Walk-forward validation.*

1. Crear ventanas de entrenamiento
2. Entrenar modelo
3. Predecir next step
4. Usar submuestra para rapidez

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: Multi-step direct (predecir 7 días)

```python
def crear_ventanas_multistep(data, window=30, horizon=7, step=1):
    """Crear ventanas para multi-step forecast."""
    X, y = [], []
    for i in range(0, len(data) - window - horizon + 1, step):
        X.append(data[i:i + window])
        y.append(data[i + window:i + window + horizon])
    return np.array(X), np.array(y)

horizon = 7
X_ms, y_ms = crear_ventanas_multistep(y_scaled, window=window, horizon=horizon)
print(f"X shape: {X_ms.shape} (muestras, timesteps, features)")
print(f"y shape: {y_ms.shape} (muestras, horizonte)")

split = int(len(X_ms) * 0.8)
X_ms_train, X_ms_test = X_ms[:split], X_ms[split:]
y_ms_train, y_ms_test = y_ms[:split], y_ms[split:]

X_ms_train = X_ms_train.reshape((-1, window, 1))
X_ms_test = X_ms_test.reshape((-1, window, 1))

modelo_multistep = Sequential([
    LSTM(50, input_shape=(window, 1), return_sequences=False),
    Dense(25, activation='relu'),
    Dense(horizon)  # 7 salidas
])

modelo_multistep.compile(optimizer=Adam(0.001), loss='mse')
modelo_multistep.fit(X_ms_train, y_ms_train, epochs=30, batch_size=32, 
                     validation_split=0.1, verbose=0,
                     callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])

pred_ms = scaler_y.inverse_transform(modelo_multistep.predict(X_ms_test, verbose=0))
y_ms_real = scaler_y.inverse_transform(y_ms_test)

rmse_ms = np.sqrt(mean_squared_error(y_ms_real.flatten(), pred_ms.flatten()))
print(f"RMSE Multi-step (7 días): {rmse_ms:.2f}")

# Visualizar una predicción multi-step
fig, ax = plt.subplots(figsize=(14, 4))
sample = 0
ax.plot(range(horizon), y_ms_real[sample], 'bo-', label='Real', linewidth=2)
ax.plot(range(horizon), pred_ms[sample], 'ro--', label='Predicho', linewidth=2)
ax.set_title(f'Ejemplo: Predicción Directa de {horizon} días')
ax.set_xlabel('Día en el futuro')
ax.set_ylabel('Ventas')
ax.legend()
plt.tight_layout()
plt.savefig('e10_ej7_multistep_direct.png', dpi=100)
plt.close()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: Multi-step direct (predecir 7 días).*

1. Visualizar una predicción multi-step

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: Multi-step recursive

```python
print("=== Multi-Step Recursive Forecast ===")

modelo_rec = Sequential([
    LSTM(50, input_shape=(window, 1), return_sequences=False),
    Dense(1)
])
modelo_rec.compile(optimizer=Adam(0.001), loss='mse')

# Entrenar para 1-step
X_rec_train, y_rec_train = crear_ventanas(y_scaled, window_size=window)
split_r = int(len(X_rec_train) * 0.8)
X_rec_train, X_rec_val = X_rec_train[:split_r], X_rec_train[split_r:]
y_rec_train, y_rec_val = y_rec_train[:split_r], y_rec_train[split_r:]

X_rec_train = X_rec_train.reshape((-1, window, 1))
X_rec_val = X_rec_val.reshape((-1, window, 1))

modelo_rec.fit(X_rec_train, y_rec_train, epochs=20, batch_size=32, verbose=0)

# Recursivo: predecir 7 días
def recursive_forecast(model, last_window, n_steps=7):
    """Generar pronóstico recursivo."""
    predictions = []
    current = last_window.copy()
    
    for _ in range(n_steps):
        pred = model.predict(current.reshape(1, window, 1), verbose=0)[0, 0]
        predictions.append(pred)
        current = np.roll(current, -1)
        current[-1] = pred
    
    return np.array(predictions)

# Test en una muestra
test_idx = 0
last_win = X_rec_val[test_idx].flatten()
pred_rec = recursive_forecast(modelo_rec, last_win, n_steps=7)
pred_rec_real = scaler_y.inverse_transform(pred_rec.reshape(-1, 1)).flatten()

# Real de los siguientes 7 días
real_rec = y_scaled[split_r * window + (test_idx + 1) * window:
                     split_r * window + (test_idx + 1) * window + 7]
real_rec_real = scaler_y.inverse_transform(real_rec.reshape(-1, 1)).flatten()

print(f"Recursive Forecast (7 días):")
for d in range(7):
    print(f"  Día {d+1}: pred={pred_rec_real[d]:.1f}, real={real_rec_real[d]:.1f}")

mape_rec = np.mean(np.abs((real_rec_real - pred_rec_real) / real_rec_real)) * 100
print(f"MAPE recursive: {mape_rec:.2f}%")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: Multi-step recursive.*

1. Entrenar para 1-step
2. Recursivo: predecir 7 días
3. Test en una muestra
4. Real de los siguientes 7 días

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: Encoder-Decoder LSTM

```python
print("=== Encoder-Decoder LSTM para Seq2Seq ===")

# Encoder
encoder_inputs = Input(shape=(window, 1))
encoder = LSTM(50, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = RepeatVector(horizon)(encoder_outputs)
decoder_lstm = LSTM(50, return_sequences=True)
decoder_outputs = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = TimeDistributed(Dense(1))
decoder_outputs = decoder_dense(decoder_outputs)

modelo_seq2seq = Model(encoder_inputs, decoder_outputs)
modelo_seq2seq.compile(optimizer=Adam(0.001), loss='mse')

print("Encoder-Decoder Architecture:")
modelo_seq2seq.summary()

# Preparar datos: decoder target debe ser secuencia
y_seq2seq = y_ms.reshape((-1, horizon, 1))

X_seq2seq_train = X_ms_train[:200]  # submuestra
y_seq2seq_train = y_seq2seq[:200]
X_seq2seq_test = X_ms_test[:50]
y_seq2seq_test = y_seq2seq[:50]

history_seq2seq = modelo_seq2seq.fit(
    X_seq2seq_train, y_seq2seq_train,
    epochs=30, batch_size=16,
    validation_split=0.1, verbose=0,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
)

pred_seq2seq = modelo_seq2seq.predict(X_seq2seq_test, verbose=0)
pred_s2s_real = scaler_y.inverse_transform(pred_seq2seq.reshape(-1, horizon))
y_s2s_real = scaler_y.inverse_transform(y_seq2seq_test.reshape(-1, horizon))

rmse_s2s = np.sqrt(np.mean((y_s2s_real - pred_s2s_real)**2))
print(f"RMSE Seq2Seq: {rmse_s2s:.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: Encoder-Decoder LSTM.*

1. Encoder
2. Decoder
3. Preparar datos: decoder target debe ser secuencia

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: LSTM con Attention

```python
print("=== LSTM con Mecanismo de Atención ===")

def build_lstm_attention(timesteps, features):
    """Construir LSTM con capa de atención."""
    inputs = Input(shape=(timesteps, features))
    
    # Encoder LSTM
    lstm_out = LSTM(50, return_sequences=True)(inputs)
    
    # Atención simple (score = softmax sobre timesteps)
    attention = Dense(1, activation='tanh')(lstm_out)
    attention = Flatten()(attention)
    attention = tf.keras.layers.Activation('softmax')(attention)
    attention = tf.keras.layers.RepeatVector(50)(attention)
    attention = tf.keras.layers.Permute([2, 1])(attention)
    
    # Context vector
    context = tf.keras.layers.Multiply()([lstm_out, attention])
    context = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(context)
    
    output = Dense(1)(context)
    
    model = Model(inputs, output)
    return model

modelo_attention = build_lstm_attention(window, 1)
modelo_attention.compile(optimizer=Adam(0.001), loss='mse')

print("LSTM with Attention:")
modelo_attention.summary()

modelo_attention.fit(
    X_train[:500], y_train[:500],
    epochs=20, batch_size=32,
    validation_split=0.1, verbose=0,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
)

pred_att = scaler_y.inverse_transform(modelo_attention.predict(X_test[:200], verbose=0))
rmse_att = np.sqrt(mean_squared_error(y_test_real[:200], pred_att))
print(f"RMSE con Attention: {rmse_att:.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: LSTM con Attention.*

1. Encoder LSTM
2. Atención simple (score = softmax sobre timesteps)
3. Context vector

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: Bidirectional LSTM

```python
print("=== Bidirectional LSTM ===")

modelo_bidi = Sequential([
    Bidirectional(LSTM(50, return_sequences=False), input_shape=(window, 1)),
    Dropout(0.2),
    Dense(1)
])

modelo_bidi.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
print("Bidirectional LSTM:")
modelo_bidi.summary()

history_bidi = modelo_bidi.fit(
    X_train, y_train,
    epochs=30, batch_size=32,
    validation_split=0.1, verbose=0,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
)

pred_bidi = scaler_y.inverse_transform(modelo_bidi.predict(X_test, verbose=0))
rmse_bidi = np.sqrt(mean_squared_error(y_test_real, pred_bidi))
mape_bidi = np.mean(np.abs((y_test_real - pred_bidi) / y_test_real)) * 100

print(f"RMSE Bi-LSTM: {rmse_bidi:.2f}")
print(f"MAPE Bi-LSTM: {mape_bidi:.2f}%")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: Bidirectional LSTM.*

1. `print("=== Bidirectional LSTM ===")` — Muestra el resultado por pantalla.
2. `print("Bidirectional LSTM:")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: LSTM con features estáticas

```python
print("=== LSTM con Features Estáticas ===")

# Features estáticas: mes, dia_semana (no cambian en la ventana)
features_estaticas = ['dia_semana', 'mes']
static_features = df[features_estaticas].values.astype(float)
static_scaler = MinMaxScaler()
static_scaled = static_scaler.fit_transform(static_features)

# Crear ventanas con features estáticas
def crear_ventanas_con_estaticas(data_seq, data_static, window=30, step=1):
    X_seq, X_static, y = [], [], []
    for i in range(0, len(data_seq) - window, step):
        X_seq.append(data_seq[i:i + window])
        X_static.append(data_static[i + window])
        y.append(data_seq[i + window])
    return np.array(X_seq), np.array(X_static), np.array(y)

X_seq, X_static, y_seq = crear_ventanas_con_estaticas(
    y_scaled, static_scaled, window=window
)

split = int(len(X_seq) * 0.8)
X_seq_train, X_seq_test = X_seq[:split], X_seq[split:]
X_static_train, X_static_test = X_static[:split], X_static[split:]
y_seq_train, y_seq_test = y_seq[:split], y_seq[split:]

# Modelo con dos inputs
input_seq = Input(shape=(window, 1))
input_static = Input(shape=(X_static.shape[1],))

lstm_out = LSTM(50, return_sequences=False)(input_seq)
merged = Concatenate()([lstm_out, input_static])
dense_out = Dense(25, activation='relu')(merged)
output = Dense(1)(dense_out)

modelo_mixed = Model(inputs=[input_seq, input_static], outputs=output)
modelo_mixed.compile(optimizer=Adam(0.001), loss='mse')

modelo_mixed.fit(
    [X_seq_train.reshape(-1, window, 1), X_static_train],
    y_seq_train,
    epochs=20, batch_size=32, verbose=0,
    validation_split=0.1
)

pred_mixed = scaler_y.inverse_transform(
    modelo_mixed.predict(
        [X_seq_test.reshape(-1, window, 1), X_static_test], verbose=0
    )
)
y_seq_test_real = scaler_y.inverse_transform(y_seq_test.reshape(-1, 1))
rmse_mixed = np.sqrt(mean_squared_error(y_seq_test_real, pred_mixed))
print(f"RMSE con features estáticas: {rmse_mixed:.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: LSTM con features estáticas.*

1. Features estáticas: mes, dia_semana (no cambian en la ventana)
2. Crear ventanas con features estáticas
3. Modelo con dos inputs

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: LSTM con features dinámicas

```python
print("=== LSTM con Features Dinámicas ===")

# Features dinámicas que cambian en cada paso temporal: precio
# Crear ventanas de features
X_price, _ = crear_ventanas(X_scaled[:, 0], window_size=window)  # precio

# Combinar en tensor 3D: [ventas, precio] por timestep
X_combined = np.stack([X, X_price], axis=-1)  # (samples, timesteps, 2)
print(f"X_combined shape: {X_combined.shape}")

split_c = int(len(X_combined) * 0.8)
Xc_train, Xc_test = X_combined[:split_c], X_combined[split_c:]
yc_train, yc_test = y[:split_c], y[split_c:]

modelo_dinamico = Sequential([
    LSTM(50, input_shape=(window, 2), return_sequences=False),
    Dense(1)
])
modelo_dinamico.compile(optimizer=Adam(0.001), loss='mse')

modelo_dinamico.fit(
    Xc_train, yc_train,
    epochs=25, batch_size=32, verbose=0,
    validation_split=0.1,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
)

pred_din = scaler_y.inverse_transform(modelo_dinamico.predict(Xc_test, verbose=0))
yc_test_real = scaler_y.inverse_transform(yc_test.reshape(-1, 1))
rmse_din = np.sqrt(mean_squared_error(yc_test_real, pred_din))
print(f"RMSE con features dinámicas (precio): {rmse_din:.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: LSTM con features dinámicas.*

1. Features dinámicas que cambian en cada paso temporal: precio
2. Crear ventanas de features
3. Combinar en tensor 3D: [ventas, precio] por timestep

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: Backtest temporal

```python
print("=== Backtest Temporal (Expanding Window) ===")

def backtest_lstm(data, window=30, initial_train=400, step=30, horizon=30):
    """Backtest con ventana expandible."""
    results = []
    train_end = initial_train
    
    while train_end + horizon <= len(data):
        train_data = data[:train_end]
        test_start = train_end
        test_end = train_end + horizon
        
        X_bt, y_bt = crear_ventanas(train_data, window_size=window)
        X_bt = X_bt.reshape((-1, window, 1))
        
        model = Sequential([
            LSTM(30, input_shape=(window, 1), return_sequences=False),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_bt, y_bt, epochs=15, batch_size=32, verbose=0)
        
        # Predecir horizonte recursivamente
        last_win = data[train_end-window:train_end]
        preds = []
        for h in range(horizon):
            pred = model.predict(last_win.reshape(1, window, 1), verbose=0)[0, 0]
            preds.append(pred)
            last_win = np.roll(last_win, -1)
            last_win[-1] = pred
        
        actuals = data[test_start:test_end]
        mape = np.mean(np.abs((actuals - preds) / (actuals + 1e-8))) * 100
        
        results.append({
            'train_end': train_end,
            'mape': mape,
            'preds': preds,
            'actuals': actuals
        })
        
        train_end += step
        print(f"  Ventana {train_end//step-1}: train hasta {train_end}, MAPE={mape:.2f}%")
    
    return results

# Backtest rápido con submuestra
results_bt = backtest_lstm(y_scaled[:600], window=30, initial_train=300, step=50, horizon=14)

if results_bt:
    mapes = [r['mape'] for r in results_bt]
    print(f"\nResumen Backtest:")
    print(f"  MAPE promedio: {np.mean(mapes):.2f}%")
    print(f"  MAPE std: {np.std(mapes):.2f}%")
    print(f"  MAPE min: {np.min(mapes):.2f}% / max: {np.max(mapes):.2f}%")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: Backtest temporal.*

1. Predecir horizonte recursivamente
2. Backtest rápido con submuestra

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: Comparar one-step vs multi-step direct vs recursive

```python
print("=== Comparación de Estrategias Multi-Step ===")

# 1. One-step (re-entrenar para comparación justa)
model_1s = Sequential([LSTM(50, input_shape=(window, 1)), Dense(1)])
model_1s.compile(optimizer='adam', loss='mse')
model_1s.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

# 2. Multi-step direct
model_direct = Sequential([LSTM(50, input_shape=(window, 1)), Dense(horizon)])
model_direct.compile(optimizer='adam', loss='mse')
model_direct.fit(X_ms_train, y_ms_train, epochs=20, batch_size=32, verbose=0)

# Evaluar en test
print(f"\n{'Estrategia':<25} {'RMSE':<10} {'MAPE':<10}")
print("-" * 45)

# One-step recursive
preds_1s = []
for i in range(len(X_test[:100])):
    p = model_1s.predict(X_test[i:i+1], verbose=0)[0, 0]
    preds_1s.append(p)
pred_1s_real = scaler_y.inverse_transform(np.array(preds_1s).reshape(-1, 1))
rmse_1s = np.sqrt(mean_squared_error(y_test_real[:100], pred_1s_real))
mape_1s = np.mean(np.abs((y_test_real[:100] - pred_1s_real) / y_test_real[:100])) * 100
print(f"{'One-step (recursive)':<25} {rmse_1s:<10.2f} {mape_1s:<10.2f}")

# Multi-step direct
pred_direct = scaler_y.inverse_transform(model_direct.predict(X_ms_test[:50], verbose=0))
y_direct_real = scaler_y.inverse_transform(y_ms_test[:50])
rmse_direct = np.sqrt(mean_squared_error(y_direct_real.flatten(), pred_direct.flatten()))
mape_direct = np.mean(np.abs((y_direct_real - pred_direct) / y_direct_real)) * 100
print(f"{'Multi-step direct':<25} {rmse_direct:<10.2f} {mape_direct:<10.2f}")

print(f"\n→ Recomendación: direct para horizonte fijo, recursive para flexibilidad")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: Comparar one-step vs multi-step direct vs recursive.*

1. 1. One-step (re-entrenar para comparación justa)
2. 2. Multi-step direct
3. Evaluar en test
4. One-step recursive
5. Multi-step direct

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 16: Regularización en LSTM

```python
print("=== Regularización en LSTM ===")

modelos_reg = {
    'Sin regularización': Sequential([
        LSTM(100, input_shape=(window, 1)), Dense(1)
    ]),
    'Dropout(0.3)': Sequential([
        LSTM(100, dropout=0.3, input_shape=(window, 1)), Dense(1)
    ]),
    'Recurrent Dropout(0.3)': Sequential([
        LSTM(100, recurrent_dropout=0.3, input_shape=(window, 1)), Dense(1)
    ]),
    'Dropout+BatchNorm': Sequential([
        LSTM(100, dropout=0.3, return_sequences=True, input_shape=(window, 1)),
        BatchNormalization(),
        LSTM(50, dropout=0.3),
        BatchNormalization(),
        Dense(1)
    ])
}

results_reg = {}
for name, model in modelos_reg.items():
    model.compile(optimizer=Adam(0.001), loss='mse')
    
    h = model.fit(
        X_train, y_train,
        epochs=30, batch_size=32,
        validation_split=0.15, verbose=0,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
    )
    
    pred = scaler_y.inverse_transform(model.predict(X_test, verbose=0))
    rmse = np.sqrt(mean_squared_error(y_test_real, pred))
    results_reg[name] = {
        'rmse': rmse,
        'train_loss': h.history['loss'][-1],
        'val_loss': h.history['val_loss'][-1]
    }
    print(f"{name:<25} RMSE={rmse:.2f} train_loss={h.history['loss'][-1]:.6f} "
          f"val_loss={h.history['val_loss'][-1]:.6f}")

# Identificar mejor
best_reg = min(results_reg, key=lambda x: results_reg[x]['rmse'])
print(f"\n→ Mejor regularización: {best_reg}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: Regularización en LSTM.*

1. Identificar mejor

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 17: Early stopping con validation

```python
print("=== Early Stopping + Learning Rate Scheduling ===")

modelo_es = Sequential([
    LSTM(100, dropout=0.2, recurrent_dropout=0.2, input_shape=(window, 1)),
    Dense(1)
])
modelo_es.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
]

history_es = modelo_es.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.15,
    verbose=0,
    callbacks=callbacks
)

print(f"Épocas ejecutadas: {len(history_es.history['loss'])} de 100 max")
print(f"Mejor val_loss: {min(history_es.history['val_loss']):.6f}")
print(f"LR final: {history_es.history['lr'][-1] if 'lr' in history_es.history else 'N/A'}")

pred_es = scaler_y.inverse_transform(modelo_es.predict(X_test, verbose=0))
rmse_es = np.sqrt(mean_squared_error(y_test_real, pred_es))
print(f"RMSE con early stopping: {rmse_es:.2f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(history_es.history['loss'], label='Train')
axes[0].plot(history_es.history['val_loss'], label='Validation')
axes[0].axvline(x=history_es.stop_trained_epoch if hasattr(history_es, 'stop_trained_epoch') 
                else len(history_es.history['loss'])-1, color='red', linestyle='--', label='Early Stop')
axes[0].set_title('Loss con Early Stopping')
axes[0].legend()

if 'lr' in history_es.history:
    axes[1].plot(history_es.history['lr'])
    axes[1].set_title('Learning Rate')
    axes[1].set_yscale('log')
plt.tight_layout()
plt.savefig('e10_ej17_earlystopping.png', dpi=100)
plt.close()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: Early stopping con validation.*

1. `print("=== Early Stopping + Learning Rate Scheduling ===")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 18: Integrador - LSTM backtest completo 30 días

```python
print("=" * 70)
print("LSTM INTEGRAL: BACKTEST COMPLETO PARA FORECAST 30 DÍAS")
print("=" * 70)

# 1. Preparación completa de datos
print("\n[1] PREPARACIÓN DE DATOS")
window_final = 60
horizon_final = 30

X_full, y_full = crear_ventanas_multistep(y_scaled, window=window_final, horizon=horizon_final)
print(f"Dataset: X={X_full.shape}, y={y_full.shape}")

# 2. Backtest temporal
print("\n[2] BACKTEST TEMPORAL (expanding window)")
n_total = len(X_full)
test_windows = []

for test_start in range(n_total - 100, n_total, 30):
    train_end = test_start
    
    X_bt = X_full[:train_end]
    y_bt = y_full[:train_end]
    X_bt_t = X_bt.reshape((-1, window_final, 1))
    
    model_bt = Sequential([
        LSTM(64, return_sequences=True, input_shape=(window_final, 1)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(horizon_final)
    ])
    model_bt.compile(optimizer=Adam(0.001), loss='mse')
    model_bt.fit(X_bt_t, y_bt, epochs=30, batch_size=32, verbose=0,
                 validation_split=0.1,
                 callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])
    
    # Evaluar
    X_eval = X_full[test_start:test_start+1].reshape(1, window_final, 1)
    y_pred = model_bt.predict(X_eval, verbose=0)[0]
    y_true = y_full[test_start]
    
    test_windows.append({
        'start': test_start,
        'pred': scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten(),
        'true': scaler_y.inverse_transform(y_true.reshape(-1, 1)).flatten()
    })
    
    mape_w = np.mean(np.abs((test_windows[-1]['true'] - test_windows[-1]['pred']) / 
                             test_windows[-1]['true'])) * 100
    print(f"  Ventana {test_start}: MAPE={mape_w:.2f}%")

# 3. Modelo final con todos los datos
print("\n[3] MODELO FINAL")
model_final = Sequential([
    LSTM(64, return_sequences=True, input_shape=(window_final, 1)),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(horizon_final)
])
model_final.compile(optimizer=Adam(0.001), loss='mse')

split_final = int(len(X_full) * 0.85)
Xf_train, Xf_test = X_full[:split_final], X_full[split_final:]
yf_train, yf_test = y_full[:split_final], y_full[split_final:]

model_final.fit(
    Xf_train.reshape((-1, window_final, 1)), yf_train,
    epochs=40, batch_size=32, verbose=0,
    validation_split=0.1,
    callbacks=[EarlyStopping(patience=8, restore_best_weights=True)]
)

# 4. Evaluación final
print("\n[4] EVALUACIÓN FINAL")
pred_final = scaler_y.inverse_transform(model_final.predict(
    Xf_test.reshape((-1, window_final, 1)), verbose=0
))
yf_test_real = scaler_y.inverse_transform(yf_test)

rmse_final = np.sqrt(mean_squared_error(yf_test_real.flatten(), pred_final.flatten()))
mape_final = np.mean(np.abs((yf_test_real - pred_final) / yf_test_real)) * 100
mae_final = np.mean(np.abs(yf_test_real - pred_final))

print(f"RMSE (30 días): {rmse_final:.2f}")
print(f"MAPE (30 días): {mape_final:.2f}%")
print(f"MAE (30 días): {mae_final:.2f}")

# 5. Visualización
print("\n[5] VISUALIZACIÓN")
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Una predicción de ejemplo
sample_idx = 0
days = range(horizon_final)
axes[0].plot(days, yf_test_real[sample_idx], 'ko-', label='Real', linewidth=2, markersize=6)
axes[0].plot(days, pred_final[sample_idx], 'rs--', label='Predicho', linewidth=2)
axes[0].fill_between(days, 
                      pred_final[sample_idx] - 1.96*np.std(yf_test_real - pred_final),
                      pred_final[sample_idx] + 1.96*np.std(yf_test_real - pred_final),
                      alpha=0.15, color='red', label='IC 95%')
axes[0].set_title(f'Ejemplo: Forecast 30 días (MAPE={mape_final:.1f}%)')
axes[0].set_ylabel('Ventas')
axes[0].legend()

# Residuos
residuos = (yf_test_real - pred_final).flatten()
axes[1].hist(residuos, bins=40, density=True, alpha=0.7, color='steelblue')
from scipy import stats
x = np.linspace(residuos.min(), residuos.max(), 100)
axes[1].plot(x, stats.norm.pdf(x, residuos.mean(), residuos.std()), 'r-', lw=2)
axes[1].set_title(f'Distribución de Errores (μ={residuos.mean():.1f}, σ={residuos.std():.1f})')
axes[1].set_xlabel('Error de pronóstico')

plt.suptitle('Pipeline LSTM Completo para Forecast de Demanda 30 Días', fontsize=15)
plt.tight_layout()
plt.savefig('e10_ej18_integrador.png', dpi=100, bbox_inches='tight')
plt.close()
print("\n✓ Pipeline LSTM completado")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador - LSTM backtest completo 30 días.*

1. 1. Preparación completa de datos
2. 2. Backtest temporal
3. Evaluar
4. 3. Modelo final con todos los datos
5. 4. Evaluación final
6. 5. Visualización
7. Una predicción de ejemplo
8. Residuos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 3. Ejercicios

1. **Ventana óptima**: Prueba window=7, 14, 30, 60, 90. ¿Cuál da mejor RMSE en test? ¿Cuál es el trade-off?

2. **Arquitectura profunda**: Agrega una tercera capa LSTM. ¿Mejora el error? ¿Hay overfitting? Monitorea train vs val loss.

3. **Features de calendario**: Agrega one-hot encoding del mes y día_semana como features dinámicas. ¿Mejora el forecast estacional?

4. **Multi-step con horizonte variable**: Implementa un modelo que prediga 1, 7, 14, 30 días. ¿Cómo cambia el error con el horizonte?

5. **Comparar optimizadores**: Prueba Adam, RMSprop, SGD con momentum. ¿Cuál converge más rápido? ¿Cuál da menor error final?

6. **Stacked LSTM con residual connections**: Implementa skip connections entre capas LSTM. Compara convergencia con stacked LSTM normal.

7. **Attention visualizada**: Extrae los pesos de atención del modelo del Ejemplo 10 y visualiza en qué timesteps se enfoca más el modelo.

8. **Pipeline de producción LSTM**: Crea una clase `LSTMForecaster` con métodos `fit()`, `predict()`, `backtest()`, que normalice automáticamente, busque window_size óptimo, entrene con early stopping y grafique resultados.

---

## 4. Resumen

| Concepto | Aplicación práctica |
|----------|---------------------|
| **Ventana deslizante** | Transformar serie temporal en aprendizaje supervisado |
| **LSTM básico** | Forecast one-step: rápido, preciso para horizonte corto |
| **Stacked LSTM** | Mayor capacidad, útil para patrones complejos |
| **Multi-step direct** | Un modelo por horizonte: estable pero costoso |
| **Multi-step recursive** | Un modelo, propaga error: barato pero menos preciso |
| **Encoder-Decoder** | Seq2Seq: ideal para forecast de múltiples pasos |
| **Attention** | Ayuda a interpretar qué pasados son más relevantes |
| **Bidirectional** | Usa contexto pasado y futuro (requiere shift) |
| **Regularización** | Dropout, recurrent_dropout, BatchNorm para evitar overfitting |
| **Backtest temporal** | Evaluación realista con ventana expandible |
