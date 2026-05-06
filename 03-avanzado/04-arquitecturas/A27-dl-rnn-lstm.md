# A27 - Deep Learning: RNN y LSTM para Ventas, Compras e Inventarios

## Fundamentos Teóricos

Las Redes Neuronales Recurrentes (RNN) están diseñadas para procesar secuencias manteniendo un estado oculto que "recuerda" información de pasos anteriores.

### SimpleRNN
- Estado oculto: `h_t = tanh(W_h · h_{t-1} + W_x · x_t + b)`
- Problema: **vanishing gradient** en secuencias largas (gradiente se desvanece)
- No puede capturar dependencias > 10 pasos

### LSTM (Long Short-Term Memory, Hochreater & Schmidhuber 1997)
- **Forget gate**: qué olvidar de la celda anterior
- **Input gate**: qué nueva información almacenar
- **Cell state**: memoria a largo plazo (flujo constante con gradiente)
- **Output gate**: qué parte de la celda emitir como salida
- Resuelve el vanishing gradient con la cell state (carretera de gradiente)
- `unit_forget_bias=True`: inicializa forget gate bias en 1 (mejor práctica)

```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)    # forget gate
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)    # input gate
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C) # candidate cell
C_t = f_t * C_{t-1} + i_t * C̃_t         # cell state
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)    # output gate
h_t = o_t * tanh(C_t)                    # hidden state
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*LSTM (Long Short-Term Memory, Hochreater & Schmidhuber 1997).*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### GRU (Gated Recurrent Unit, Cho 2014)
- Combina forget e input gate en **update gate**
- Menos parámetros que LSTM (3 gates → 2 gates)
- Rendimiento similar a LSTM en muchos problemas
- `reset_after=True`: comportamiento original (default en TF)

### Parámetros Clave
| Parámetro | Descripción |
|-----------|-------------|
| return_sequences | Output en cada paso (True para apilar) |
| return_state | Devolver estado oculto/celda además de output |
| go_backwards | Procesar secuencia al revés |
| stateful | Mantener estado entre batches (secuencias largas) |
| unroll | Desenrollar el bucle (más rápido pero más memoria) |
| dropout | Dropout en input |
| recurrent_dropout | Dropout en estado recurrente |

### Bidirectional RNN
- Procesa la secuencia en ambas direcciones (forward + backward)
- `merge_mode`: 'concat' (default), 'sum', 'mul', 'ave'
- Captura contexto pasado y futuro

### Stacked RNN
- Múltiples capas recurrentes apiladas
- Cada capa aprende representaciones de diferente abstracción temporal
- Solo la última capa puede tener return_sequences=False

### Atención (Attention)
- Mecanismo que pondera la importancia de cada paso temporal
- Permite al modelo "enfocarse" en partes relevantes de la secuencia
- `attention_score = softmax(W_a · h_t)`
- `context = Σ attention_score · h_t`

---

## Ejemplos Prácticos

### Ejemplo 1: SimpleRNN — Unidad Recurrente Básica sobre Secuencia de Ventas

```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam

np.random.seed(42)
tf.random.set_seed(42)

# Serie de ventas diarias: 1000 días con estacionalidad semanal y mensual
dias = 1000
t = np.arange(dias)
ventas = (100 + 20*np.sin(2*np.pi*t/7) + 10*np.sin(2*np.pi*t/30) +
          5*np.sin(2*np.pi*t/365) + np.random.normal(0, 5, dias))

def crear_secuencias(series, input_len=14, output_len=1):
    X, y = [], []
    for i in range(len(series) - input_len - output_len + 1):
        X.append(series[i:i+input_len])
        y.append(series[i+input_len:i+input_len+output_len])
    return np.array(X), np.array(y)

X, y = crear_secuencias(ventas, input_len=14, output_len=1)
X = X.reshape(-1, 14, 1)
y = y.reshape(-1, 1)

split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

modelo = Sequential([
    SimpleRNN(32, activation='tanh', input_shape=(14, 1)),
    Dense(1)
])
modelo.compile(optimizer=Adam(0.001), loss='mse')
modelo.summary()

history = modelo.fit(X_train, y_train, epochs=50, batch_size=32,
                     validation_data=(X_val, y_val), verbose=1)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('SimpleRNN - Predicción de ventas')
plt.legend()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 1: SimpleRNN — Unidad Recurrente Básica sobre Secuencia de Ventas.*

1. Serie de ventas diarias: 1000 días con estacionalidad semanal y mensual

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: SimpleRNN con return_sequences=True para Apilar

```python
modelo_stack = Sequential([
    SimpleRNN(32, activation='tanh', return_sequences=True, input_shape=(14, 1)),
    SimpleRNN(16, activation='tanh', return_sequences=False),
    Dense(1)
])
modelo_stack.compile(optimizer=Adam(0.001), loss='mse')
modelo_stack.summary()

h_stack = modelo_stack.fit(X_train, y_train, epochs=30, batch_size=32,
                           validation_data=(X_val, y_val), verbose=0)
print(f"Stacked SimpleRNN val_loss: {min(h_stack.history['val_loss']):.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: SimpleRNN con return_sequences=True para Apilar.*

1. `h_stack = modelo_stack.fit(X_train, y_train, epochs=30, batch_size=32,` — Entrena el modelo con los datos de entrenamiento.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: LSTM — Unidad con Forget Gate (Ventas Diarias → Próxima Semana)

```python
from tensorflow.keras.layers import LSTM

X_lstm, y_lstm = crear_secuencias(ventas, input_len=14, output_len=7)
X_lstm = X_lstm.reshape(-1, 14, 1)
y_lstm = y_lstm.reshape(-1, 7)

Xl_train, Xl_val = X_lstm[:split], X_lstm[split:]
yl_train, yl_val = y_lstm[:split], y_lstm[split:]

modelo_lstm = Sequential([
    LSTM(64, activation='tanh', recurrent_activation='sigmoid',
         input_shape=(14, 1)),
    Dense(7)  # predecir 7 días
])
modelo_lstm.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
modelo_lstm.summary()

h_lstm = modelo_lstm.fit(Xl_train, yl_train, epochs=50, batch_size=32,
                         validation_data=(Xl_val, yl_val), verbose=1)
```

**Salida:**

```
# El modelo ajusta sus parámetros durante el entrenamiento.
# Los parámetros aprendidos quedan disponibles en el objeto.
```

**Explicación línea por línea:**

*Ejemplo 3: LSTM — Unidad con Forget Gate (Ventas Diarias → Próxima Semana).*

1. `from tensorflow.keras.layers import LSTM` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: LSTM return_sequences=True — Output de Cada Paso

```python
# Modelo con return_sequences para apilar LSTM
modelo_lstm_stack = Sequential([
    LSTM(64, return_sequences=True, input_shape=(14, 1)),
    LSTM(32, return_sequences=False),
    Dense(1)
])
modelo_lstm_stack.compile(optimizer=Adam(0.001), loss='mse')
modelo_lstm_stack.summary()
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Ejemplo 4: LSTM return_sequences=True — Output de Cada Paso.*

1. Modelo con return_sequences para apilar LSTM

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: LSTM return_state — Devolver Estado Oculto y Celda

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

# Modelo que devuelve estados (útil para decodificación)
entrada = Input(shape=(14, 1))
lstm_out, estado_h, estado_c = LSTM(32, return_sequences=True,
                                     return_state=True)(entrada)

print(f"Output shape: {lstm_out.shape}")
print(f"Estado oculto shape: {estado_h.shape}")
print(f"Estado celda shape: {estado_c.shape}")

# Crear modelo funcional para estados
modelo_estados = Model(inputs=entrada, outputs=[lstm_out, estado_h, estado_c])
modelo_estados.compile(optimizer=Adam(0.001), loss='mse')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: LSTM return_state — Devolver Estado Oculto y Celda.*

1. Modelo que devuelve estados (útil para decodificación)
2. Crear modelo funcional para estados

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: GRU — Alternativa más Ligera a LSTM

```python
from tensorflow.keras.layers import GRU

modelo_gru = Sequential([
    GRU(64, activation='tanh', recurrent_activation='sigmoid',
        reset_after=True, input_shape=(14, 1)),
    Dense(1)
])
modelo_gru.compile(optimizer=Adam(0.001), loss='mse')

print(f"Parámetros LSTM: {modelo_lstm_stack.count_params():,}")
print(f"Parámetros GRU:  {modelo_gru.count_params():,}")

h_gru = modelo_gru.fit(X_train, y_train, epochs=30, batch_size=32,
                       validation_data=(X_val, y_val), verbose=0)
print(f"GRU val_loss: {min(h_gru.history['val_loss']):.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: GRU — Alternativa más Ligera a LSTM.*

1. `from tensorflow.keras.layers import GRU` — Importa las librerías necesarias para el análisis.
2. `print(f"Parámetros LSTM: {modelo_lstm_stack.count_params():,}")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: Stacked LSTM — 2 y 3 Capas Apiladas

```python
# 2 capas LSTM apiladas
modelo_2stack = Sequential([
    LSTM(64, return_sequences=True, input_shape=(14, 1)),
    LSTM(32, return_sequences=False),
    Dense(1)
])

# 3 capas LSTM apiladas
modelo_3stack = Sequential([
    LSTM(64, return_sequences=True, input_shape=(14, 1)),
    LSTM(48, return_sequences=True),
    LSTM(32, return_sequences=False),
    Dense(1)
])

modelo_2stack.compile(optimizer=Adam(0.001), loss='mse')
modelo_3stack.compile(optimizer=Adam(0.001), loss='mse')

h_2s = modelo_2stack.fit(X_train, y_train, epochs=50, batch_size=32,
                          validation_data=(X_val, y_val), verbose=0)
h_3s = modelo_3stack.fit(X_train, y_train, epochs=50, batch_size=32,
                          validation_data=(X_val, y_val), verbose=0)

print(f"2-LSTM stack val_loss: {min(h_2s.history['val_loss']):.4f}")
print(f"3-LSTM stack val_loss: {min(h_3s.history['val_loss']):.4f}")

plt.plot(h_2s.history['val_loss'], label='2 capas LSTM')
plt.plot(h_3s.history['val_loss'], label='3 capas LSTM')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.legend()
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: Stacked LSTM — 2 y 3 Capas Apiladas.*

1. 2 capas LSTM apiladas
2. 3 capas LSTM apiladas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: Bidirectional LSTM — Procesar en Ambas Direcciones

```python
from tensorflow.keras.layers import Bidirectional

modelo_bi = Sequential([
    Bidirectional(LSTM(32, return_sequences=True), input_shape=(14, 1)),
    Bidirectional(LSTM(16, return_sequences=False)),
    Dense(1)
])
modelo_bi.compile(optimizer=Adam(0.001), loss='mse')
modelo_bi.summary()

h_bi = modelo_bi.fit(X_train, y_train, epochs=30, batch_size=32,
                     validation_data=(X_val, y_val), verbose=0)
print(f"Bidirectional LSTM val_loss: {min(h_bi.history['val_loss']):.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: Bidirectional LSTM — Procesar en Ambas Direcciones.*

1. `from tensorflow.keras.layers import Bidirectional` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: Bidirectional — merge_mode='concat' vs 'sum' vs 'ave'

```python
merge_modes = ['concat', 'sum', 'ave']
histories_merge = {}

for mode in merge_modes:
    modelo = Sequential([
        Bidirectional(
            LSTM(32, return_sequences=True),
            merge_mode=mode, input_shape=(14, 1)
        ),
        Bidirectional(
            LSTM(16, return_sequences=False),
            merge_mode=mode
        ),
        Dense(1)
    ])
    modelo.compile(optimizer=Adam(0.001), loss='mse')
    h = modelo.fit(X_train, y_train, epochs=30, batch_size=32,
                   validation_data=(X_val, y_val), verbose=0)
    histories_merge[mode] = h
    print(f"merge_mode={mode:7s} | Val loss: {min(h.history['val_loss']):.4f}")

for mode in merge_modes:
    plt.plot(histories_merge[mode].history['val_loss'], label=mode)
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.legend()
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: Bidirectional — merge_mode='concat' vs 'sum' vs 'ave'.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: LSTM con Dropout y Recurrent Dropout

```python
modelo_lstm_reg = Sequential([
    LSTM(64, dropout=0.2, recurrent_dropout=0.2,
         return_sequences=True, input_shape=(14, 1)),
    LSTM(32, dropout=0.2, recurrent_dropout=0.2,
         return_sequences=False),
    Dense(1)
])
modelo_lstm_reg.compile(optimizer=Adam(0.001), loss='mse')
modelo_lstm_reg.summary()

h_lstm_reg = modelo_lstm_reg.fit(X_train, y_train, epochs=50, batch_size=32,
                                  validation_data=(X_val, y_val), verbose=1)

# Comparar con versión sin dropout
modelo_lstm_no_reg = Sequential([
    LSTM(64, return_sequences=True, input_shape=(14, 1)),
    LSTM(32, return_sequences=False),
    Dense(1)
])
modelo_lstm_no_reg.compile(optimizer=Adam(0.001), loss='mse')
h_no_reg = modelo_lstm_no_reg.fit(X_train, y_train, epochs=50, batch_size=32,
                                   validation_data=(X_val, y_val), verbose=0)

plt.plot(h_no_reg.history['val_loss'], label='Sin regularización', linestyle='--')
plt.plot(h_lstm_reg.history['val_loss'], label='Con dropout', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Regularización en LSTM')
plt.legend()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 10: LSTM con Dropout y Recurrent Dropout.*

1. Comparar con versión sin dropout

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: LSTM stateful=True — Mantener Estado entre Batches

```python
# Para stateful, debemos especificar batch_size en input
# y resetear estados manualmente entre épocas
batch_size = 32
X_state = X_train[:batch_size * (len(X_train) // batch_size)]
y_state = y_train[:batch_size * (len(y_train) // batch_size)]
X_state = X_state.reshape(-1, batch_size, 14, 1)
y_state = y_state.reshape(-1, batch_size, 1)

modelo_stateful = Sequential([
    LSTM(32, stateful=True, batch_input_shape=(batch_size, 14, 1)),
    Dense(1)
])
modelo_stateful.compile(optimizer=Adam(0.001), loss='mse')

for epoch in range(20):
    for i in range(X_state.shape[0]):
        modelo_stateful.fit(X_state[i], y_state[i], epochs=1,
                            batch_size=batch_size, verbose=0)
    modelo_stateful.reset_states()
    if epoch % 5 == 0:
        val_loss = modelo_stateful.evaluate(X_val, y_val, verbose=0)
        print(f"Epoch {epoch:3d} | Val loss: {val_loss:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: LSTM stateful=True — Mantener Estado entre Batches.*

1. Para stateful, debemos especificar batch_size en input
2. y resetear estados manualmente entre épocas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: LSTM para Multi-step Forecast (7 Días)

```python
X_multi, y_multi = crear_secuencias(ventas, input_len=28, output_len=7)
X_multi = X_multi.reshape(-1, 28, 1)
y_multi = y_multi.reshape(-1, 7)

Xm_train, Xm_val = X_multi[:700], X_multi[700:]
ym_train, ym_val = y_multi[:700], y_multi[700:]

modelo_multistep = Sequential([
    LSTM(64, return_sequences=True, input_shape=(28, 1)),
    LSTM(32, return_sequences=False),
    Dense(7)  # salida: 7 días
])
modelo_multistep.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
modelo_multistep.summary()

h_multi = modelo_multistep.fit(Xm_train, ym_train, epochs=50, batch_size=32,
                               validation_data=(Xm_val, ym_val), verbose=1)

# Visualizar predicción de 7 días
i = 0
pred_7 = modelo_multistep.predict(Xm_val[i:i+1], verbose=0)[0]
plt.figure(figsize=(10, 4))
plt.plot(range(1, 29), Xm_val[i, :, 0], 'b-', label='Input (28 días)')
plt.plot(range(29, 36), ym_val[i], 'go-', label='Real (7 días)')
plt.plot(range(29, 36), pred_7, 'rs--', label='Predicho (7 días)')
plt.xlabel('Día')
plt.ylabel('Ventas')
plt.title('Multi-step forecast: predicción de 7 días')
plt.legend()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 12: LSTM para Multi-step Forecast (7 Días).*

1. Visualizar predicción de 7 días

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: LSTM con Variable Length Sequences (Masking)

```python
from tensorflow.keras.layers import Masking

# Datos con longitudes variables: algunas secuencias más cortas (rellenadas con 0)
# Máscara: ignora los timesteps donde todas las features son 0
X_var = np.random.randn(500, 30, 3)
for i in range(500):
    fin = np.random.randint(10, 30)
    X_var[i, fin:] = 0  # rellenar con 0
y_var = np.random.randn(500, 1)

modelo_mask = Sequential([
    Masking(mask_value=0.0, input_shape=(30, 3)),
    LSTM(32, return_sequences=False),
    Dense(1)
])
modelo_mask.compile(optimizer=Adam(0.001), loss='mse')
print("Masking: ignora timesteps con valor 0")
modelo_mask.summary()

# Ver máscara en acción
mascara = modelo_mask.layers[0].compute_mask(X_var[:1])
print(f"Máscara generada (True = mantener): {mascara[0].numpy()[:15]}...")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: LSTM con Variable Length Sequences (Masking).*

1. Datos con longitudes variables: algunas secuencias más cortas (rellenadas con 0)
2. Máscara: ignora los timesteps donde todas las features son 0
3. Ver máscara en acción

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: CuDNNLSTM — Aceleración GPU

```python
# TensorFlow automáticamente usa CuDNNLSTM cuando:
# 1. activation='tanh' (default)
# 2. recurrent_activation='sigmoid' (default)
# 3. recurrent_dropout=0
# 4. unroll=False
# 5. use_bias=True
# 6. GPU disponible

modelo_cudnn = Sequential([
    LSTM(64, return_sequences=True, input_shape=(14, 1)),  # CuDNNLSTM en GPU
    LSTM(32, return_sequences=False),                       # CuDNNLSTM en GPU
    Dense(1)
])

# Verificar si usa GPU
if tf.config.list_physical_devices('GPU'):
    print("GPU disponible - LSTM usará CuDNNLSTM automáticamente")
else:
    print("GPU no disponible - usando CPU")

modelo_cudnn.compile(optimizer=Adam(0.001), loss='mse')

# Entrenamiento (más rápido en GPU)
import time
start = time.time()
h_cudnn = modelo_cudnn.fit(X_train, y_train, epochs=30, batch_size=32,
                           validation_data=(X_val, y_val), verbose=0)
print(f"Tiempo de entrenamiento: {time.time() - start:.2f}s")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: CuDNNLSTM — Aceleración GPU.*

1. TensorFlow automáticamente usa CuDNNLSTM cuando:
2. 1. activation='tanh' (default)
3. 2. recurrent_activation='sigmoid' (default)
4. 3. recurrent_dropout=0
5. 4. unroll=False
6. 5. use_bias=True
7. 6. GPU disponible
8. Verificar si usa GPU

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: Comparar SimpleRNN vs LSTM vs GRU en Convergencia

```python
configs = {
    'SimpleRNN': SimpleRNN(32, return_sequences=True),
    'LSTM': LSTM(32, return_sequences=True),
    'GRU': GRU(32, return_sequences=True)
}

histories_rnn = {}
for nombre, capa in configs.items():
    modelo = Sequential([
        capa,
        LSTM(16, return_sequences=False),  # segunda capa siempre LSTM para consistencia
        Dense(1)
    ])
    # Rebuild: el input_shape lo toma del primer layer
    modelo.compile(optimizer=Adam(0.001), loss='mse')
    h = modelo.fit(X_train, y_train, epochs=30, batch_size=32,
                   validation_data=(X_val, y_val), verbose=0)
    histories_rnn[nombre] = h
    print(f"{nombre:15s} | Val loss: {min(h.history['val_loss']):.4f}")

for nombre in configs:
    plt.plot(histories_rnn[nombre].history['val_loss'], label=nombre)
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Comparación: SimpleRNN vs LSTM vs GRU')
plt.legend()
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: Comparar SimpleRNN vs LSTM vs GRU en Convergencia.*

1. Rebuild: el input_shape lo toma del primer layer

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 16: LSTM + Attention — Capa de Atención sobre Salidas LSTM

```python
from tensorflow.keras.layers import Layer, Attention

# Capa de atención personalizada
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer='glorot_uniform',
                                 trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch, timesteps, features)
        scores = tf.matmul(inputs, self.W)  # (batch, timesteps, 1)
        scores = tf.nn.softmax(scores, axis=1)  # (batch, timesteps, 1)
        context = tf.reduce_sum(inputs * scores, axis=1)  # (batch, features)
        return context

# Modelo LSTM + Attention
entrada = Input(shape=(14, 1))
lstm_out = LSTM(64, return_sequences=True)(entrada)
atencion = AttentionLayer()(lstm_out)
salida = Dense(1)(atencion)

modelo_attention = Model(inputs=entrada, outputs=salida)
modelo_attention.compile(optimizer=Adam(0.001), loss='mse')
modelo_attention.summary()

h_att = modelo_attention.fit(X_train, y_train, epochs=50, batch_size=32,
                             validation_data=(X_val, y_val), verbose=1)
print(f"LSTM+Attention val_loss: {min(h_att.history['val_loss']):.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: LSTM + Attention — Capa de Atención sobre Salidas LSTM.*

1. Capa de atención personalizada
2. inputs shape: (batch, timesteps, features)
3. Modelo LSTM + Attention

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 17: Curva de Aprendizaje — Loss en Train y Validation

```python
# Entrenar modelo LSTM completo y graficar curvas
modelo_curvas = Sequential([
    LSTM(64, return_sequences=True, input_shape=(14, 1)),
    LSTM(32, return_sequences=False),
    Dense(1)
])
modelo_curvas.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])

h_curvas = modelo_curvas.fit(X_train, y_train, epochs=80, batch_size=32,
                              validation_data=(X_val, y_val), verbose=1)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(h_curvas.history['loss'], label='Train')
axes[0].plot(h_curvas.history['val_loss'], label='Validation')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (MSE)')
axes[0].set_title('Curva de pérdida')
axes[0].legend()

axes[1].plot(h_curvas.history['mae'], label='Train')
axes[1].plot(h_curvas.history['val_mae'], label='Validation')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MAE')
axes[1].set_title('Error absoluto medio')
axes[1].legend()
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 17: Curva de Aprendizaje — Loss en Train y Validation.*

1. Entrenar modelo LSTM completo y graficar curvas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 18: Integrador — LSTM para Predicción de Demanda Diaria

```python
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_squared_error, r2_score

# Dataset realista de demanda diaria con múltiples features
np.random.seed(42)
tf.random.set_seed(42)

n_dias = 1500
t = np.arange(n_dias)

# Features: precio, publicidad, promoción, día_semana, temporada
precio = 50 + 10*np.sin(2*np.pi*t/90) + np.random.normal(0, 3, n_dias)
publicidad = np.random.uniform(0, 100, n_dias)
promocion = np.random.binomial(1, 0.15, n_dias)
dia_semana = t % 7
temporada = np.sin(2*np.pi*t/365)

# Demanda (target) con dependencias temporales
demanda = (200 - 1.5*precio + 0.3*publicidad + 30*promocion +
           10*np.sin(2*np.pi*t/7) + 5*np.sin(2*np.pi*t/30) +
           np.random.normal(0, 15, n_dias))
demanda = np.maximum(demanda, 0)

# Stack: 5 features → shape (n_dias, 5)
features = np.column_stack([precio, publicidad, promocion, dia_semana, temporada])

# Ventanas de 21 días (3 semanas)
def crear_ventanas_multifeature(X, y, ventana=21):
    X_w, y_w = [], []
    for i in range(len(X) - ventana):
        X_w.append(X[i:i+ventana])
        y_w.append(y[i+ventana])
    return np.array(X_w), np.array(y_w)

X_final, y_final = crear_ventanas_multifeature(features, demanda, ventana=21)
split = int(0.8 * len(X_final))
Xf_train, Xf_val = X_final[:split], X_final[split:]
yf_train, yf_val = y_final[:split], y_final[split:]

# Normalizar
from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
scaler_y = StandardScaler()

shape_train = Xf_train.shape
shape_val = Xf_val.shape
Xf_train_reshaped = Xf_train.reshape(-1, Xf_train.shape[-1])
Xf_val_reshaped = Xf_val.reshape(-1, Xf_val.shape[-1])
Xf_train = scaler_X.fit_transform(Xf_train_reshaped).reshape(shape_train)
Xf_val = scaler_X.transform(Xf_val_reshaped).reshape(shape_val)
yf_train = scaler_y.fit_transform(yf_train.reshape(-1, 1)).ravel()
yf_val = scaler_y.transform(yf_val.reshape(-1, 1)).ravel()

# Modelo LSTM integrador
lstm_integrador = Sequential([
    LSTM(128, return_sequences=True, input_shape=(21, 5)),
    BatchNormalization(),
    Dropout(0.2),

    LSTM(64, return_sequences=True),
    BatchNormalization(),
    Dropout(0.2),

    LSTM(32, return_sequences=False),
    BatchNormalization(),
    Dropout(0.2),

    Dense(16, activation='relu', kernel_regularizer=l2(1e-4)),
    Dense(1)
])

optimizer = Adam(learning_rate=0.001)
lstm_integrador.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
lstm_integrador.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
]

h_int = lstm_integrador.fit(Xf_train, yf_train, epochs=150, batch_size=32,
                            validation_data=(Xf_val, yf_val),
                            callbacks=callbacks, verbose=1)

# Evaluación
y_pred_int = lstm_integrador.predict(Xf_val, verbose=0)
y_pred_int_inv = scaler_y.inverse_transform(y_pred_int)
yf_val_inv = scaler_y.inverse_transform(yf_val.reshape(-1, 1))
r2 = r2_score(yf_val_inv, y_pred_int_inv)
rmse = np.sqrt(mean_squared_error(yf_val_inv, y_pred_int_inv))
print(f"\n=== Resultados LSTM Integrador ===")
print(f"R²: {r2:.4f}")
print(f"RMSE: {rmse:.2f}")

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(h_int.history['loss'], label='Train', alpha=0.7)
plt.plot(h_int.history['val_loss'], label='Validation', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Curva de aprendizaje')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(yf_val_inv, y_pred_int_inv, alpha=0.4)
plt.plot([yf_val_inv.min(), yf_val_inv.max()],
         [yf_val_inv.min(), yf_val_inv.max()], 'r--')
plt.xlabel('Valor real')
plt.ylabel('Predicción')
plt.title(f'LSTM Integrador (R²={r2:.3f})')
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador — LSTM para Predicción de Demanda Diaria.*

1. Dataset realista de demanda diaria con múltiples features
2. Features: precio, publicidad, promoción, día_semana, temporada
3. Demanda (target) con dependencias temporales
4. Stack: 5 features → shape (n_dias, 5)
5. Ventanas de 21 días (3 semanas)
6. Normalizar
7. Modelo LSTM integrador
8. Evaluación

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejercicios

1. **SimpleRNN para compras**: Construye un modelo SimpleRNN con 32 unidades que prediga compras diarias basado en 10 días de historia. Entrena con datos simulados (compras con tendencia + ruido). ¿Notas vanishing gradient con secuencias largas?

2. **LSTM para inventario**: Diseña un LSTM con 64 unidades que prediga el nivel de stock óptimo para 7 días adelante. Features: ventas_últimos_30_días, lead_time, stock_actual, pedidos_pendientes. Usa return_sequences=True para apilar 2 LSTM.

3. **GRU vs LSTM para predicción de demanda**: Implementa modelos GRU y LSTM con la misma arquitectura (2 capas, 64 y 32 unidades). Compáralos en términos de val_loss, número de parámetros y tiempo de entrenamiento para predicción de demanda semanal.

4. **Bidirectional LSTM para clasificación de productos**: Usa secuencias de 20 días con 6 features para clasificar productos en 3 categorías (alta/media/baja rotación) usando Bidirectional LSTM. Salida softmax 3 clases. Compara forward vs bidirectional.

5. **Multi-step forecast con LSTM**: Genera un dataset de ventas diarias y entrena un LSTM que prediga los próximos 14 días usando input de 28 días. Evalúa el error por cada día del horizonte (¿el error crece con el horizonte?).

6. **Stateful LSTM para series muy largas**: Implementa un LSTM stateful con batch_size=16 sobre 5000 días de ventas. Usa reset_states() entre épocas. Compara tiempo de entrenamiento y rendimiento contra un LSTM stateless equivalente.

7. **LSTM con masking para productos con historias diferentes**: Simula 800 productos con historias de ventas de longitud variable (entre 15 y 60 días). Usa Masking(mask_value=0) y entrena un LSTM para predecir ventas futuras. ¿Cómo maneja el modelo las longitudes variables?

8. **Integrador: LSTM con atención para pronóstico de demanda**: Combina LSTM apilado (3 capas: 128, 64, 32) con una capa de atención y regularización completa (dropout=0.25, recurrent_dropout=0.15, L2=1e-5) usando 8 features de ventas/compras/inventario con ventana de 30 días para predecir 1 día. Incluye early stopping (patience=15) y ReduceLROnPlateau.

---

## Resumen

- **SimpleRNN** es la unidad recurrente básica pero sufre vanishing gradient en secuencias largas (>10 pasos).
- **LSTM** resuelve el vanishing gradient con la cell state y sus 3 gates (forget, input, output). Es la opción más robusta para series de ventas.
- **GRU** es una alternativa con menos parámetros (2 gates) que suele rendir similar a LSTM.
- **return_sequences=True** permite apilar capas recurrentes; solo la última puede tener return_sequences=False.
- **Bidirectional LSTM** procesa en ambas direcciones, capturando contexto pasado y futuro. merge_mode='concat' es el default.
- **Regularización**: dropout (en input) + recurrent_dropout (en estado recurrente) son críticos para evitar overfitting en RNNs.
- **Stateful LSTM** mantiene estado entre batches, útil para secuencias muy largas donde no caben en un batch.
- **Atención** mejora el enfoque del modelo en pasos relevantes de la secuencia.
- **Multi-step forecast**: LSTM puede predecir múltiples pasos (7-14 días) configurando Dense con output_size = horizonte.
- La arquitectura recomendada para ventas: Stacked LSTM (2-3 capas) + dropout + early stopping + atención.
