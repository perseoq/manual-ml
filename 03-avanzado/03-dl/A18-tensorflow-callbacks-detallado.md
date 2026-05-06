# A18: Callbacks en TensorFlow — Control de Entrenamiento para Ventas, Compras e Inventarios

## Introducción Teórica

Los **callbacks** en tf.keras son objetos que se ejecutan durante el entrenamiento en puntos específicos (inicio/fin de epoch, inicio/fin de batch, inicio/fin de entrenamiento). Permiten monitorear, controlar y modificar el entrenamiento sin modificar el bucle principal.

### Callbacks principales:

1. **EarlyStopping**: Detiene el entrenamiento cuando una métrica deja de mejorar. Parámetros: `monitor`, `patience`, `min_delta`, `restore_best_weights`, `start_from_epoch`, `mode` ('auto', 'min', 'max'), `baseline`.
2. **ModelCheckpoint**: Guarda el modelo (o solo pesos) periódicamente o cuando mejora. Parámetros: `filepath`, `monitor`, `save_best_only`, `save_weights_only`, `mode`, `save_freq`, `options`, `initial_value_threshold`.
3. **ReduceLROnPlateau**: Reduce la tasa de aprendizaje cuando una métrica se estanca. Parámetros: `monitor`, `factor`, `patience`, `min_lr`, `min_delta`, `cooldown`, `verbose`, `mode`.
4. **LearningRateScheduler**: Aplica una función personalizada para modificar el learning rate en cada epoch.
5. **CSVLogger**: Registra métricas de entrenamiento en un archivo CSV.
6. **TensorBoard**: Visualiza gráficos, histogramas de pesos, embeddings en TensorBoard.
7. **LambdaCallback**: Ejecuta funciones arbitrarias en cada evento del entrenamiento.
8. **TerminateOnNaN**: Detiene el entrenamiento si la pérdida se vuelve NaN.
9. **ProgbarLogger**: Muestra la barra de progreso (usado internamente por verbose=1).
10. **BackupAndRestore**: Guarda el progreso para recuperar entrenamiento interrumpido.
11. **RemoteMonitor**: Envía métricas a un servidor remoto.

### Aplicaciones en negocio:

- **Ventas**: EarlyStopping para evitar overfitting en predicción de demanda; ModelCheckpoint para guardar el mejor modelo.
- **Compras**: ReduceLROnPlateau para afinar aprendizaje en lead time; CSVLogger para auditoría.
- **Inventarios**: BackupAndRestore para entrenamientos largos; TensorBoard para monitorear convergencia.

---

## Ejemplos

### Ejemplo 1: EarlyStopping — Detener si val_loss no mejora por 10 epochs

```python
import tensorflow as tf
import numpy as np

np.random.seed(42)
tf.random.set_seed(42)

X = np.random.rand(1000, 5).astype(np.float32)
y = np.random.rand(1000, 1).astype(np.float32)

modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
modelo.compile(optimizer='adam', loss='mse')

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1,
    mode='min'
)

hist = modelo.fit(X, y, epochs=500, batch_size=32, validation_split=0.2,
                  callbacks=[early_stop], verbose=0)
print(f"Entrenamiento detenido en epoch {len(hist.history['loss'])}")
print(f"Val loss final: {hist.history['val_loss'][-1]:.6f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: EarlyStopping — Detener si val_loss no mejora por 10 epochs.*

1. `import tensorflow as tf` — Importa las librerías necesarias para el análisis.
2. `import numpy as np` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: EarlyStopping con restore_best_weights=True

```python
modelo_rs = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(1)
])
modelo_rs.compile(optimizer='adam', loss='mse')

early_rs = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

hist_rs = modelo_rs.fit(X, y, epochs=500, batch_size=32, validation_split=0.2,
                        callbacks=[early_rs], verbose=0)
best_epoch = np.argmin(hist_rs.history['val_loss'])
print(f"Mejor val_loss en epoch {best_epoch+1}: {hist_rs.history['val_loss'][best_epoch]:.6f}")
print(f"Val_loss final (puede ser peor): {hist_rs.history['val_loss'][-1]:.6f} (pero pesos del mejor)")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: EarlyStopping con restore_best_weights=True.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: EarlyStopping con min_delta=0.001 (ignorar mejoras mínimas)

```python
modelo_md = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(1)
])
modelo_md.compile(optimizer='adam', loss='mse')

early_md = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    min_delta=0.001,
    patience=5,
    verbose=1,
    mode='min'
)

hist_md = modelo_md.fit(X, y, epochs=300, batch_size=32,
                        callbacks=[early_md], verbose=0)
print(f"Detenido en epoch {len(hist_md.history['loss'])} con loss={hist_md.history['loss'][-1]:.6f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: EarlyStopping con min_delta=0.001 (ignorar mejoras mínimas).*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: ModelCheckpoint — Guardar mejor modelo en cada epoch para predicción de demanda

```python
import tempfile, os

modelo_ckpt = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(1)
])
modelo_ckpt.compile(optimizer='adam', loss='mse')

ruta_ckpt = os.path.join(tempfile.gettempdir(), 'mejor_modelo_demanda.keras')
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=ruta_ckpt,
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

hist_ckpt = modelo_ckpt.fit(X, y, epochs=50, batch_size=32, validation_split=0.2,
                            callbacks=[checkpoint], verbose=0)
print(f"Mejor modelo guardado en: {ruta_ckpt}")
print(f"Tamaño: {os.path.getsize(ruta_ckpt) / 1024:.1f} KB")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: ModelCheckpoint — Guardar mejor modelo en cada epoch para predicción de demanda.*

1. `import tempfile, os` — Importa las librerías necesarias para el análisis.
2. `ruta_ckpt = os.path.join(tempfile.gettempdir(), 'mejor_modelo_demanda.keras')` — Combina dos DataFrames por una columna clave.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: ModelCheckpoint con save_weights_only=True (solo pesos)

```python
modelo_w = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(1)
])
modelo_w.compile(optimizer='adam', loss='mse')

ruta_weights = os.path.join(tempfile.gettempdir(), 'pesos_demanda.weights.h5')
checkpoint_w = tf.keras.callbacks.ModelCheckpoint(
    filepath=ruta_weights,
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=True,
    mode='min',
    verbose=1
)

hist_w = modelo_w.fit(X, y, epochs=30, batch_size=32, validation_split=0.2,
                      callbacks=[checkpoint_w], verbose=0)
print(f"Pesos guardados en: {ruta_weights}")
print(f"Tamaño: {os.path.getsize(ruta_weights) / 1024:.1f} KB (solo pesos, sin arquitectura)")

# Cargar pesos en un modelo nuevo
modelo_nuevo = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(1)
])
modelo_nuevo.compile(optimizer='adam', loss='mse')
modelo_nuevo.load_weights(ruta_weights)
print("Pesos cargados en modelo nuevo")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: ModelCheckpoint con save_weights_only=True (solo pesos).*

1. Cargar pesos en un modelo nuevo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: ModelCheckpoint + EarlyStopping combinados

```python
modelo_comb = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
modelo_comb.compile(optimizer='adam', loss='mse', metrics=['mae'])

ruta_comb = os.path.join(tempfile.gettempdir(), 'modelo_combinado.keras')
callbacks_comb = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(filepath=ruta_comb, monitor='val_loss', save_best_only=True, verbose=1)
]

hist_comb = modelo_comb.fit(X, y, epochs=200, batch_size=32, validation_split=0.2,
                            callbacks=callbacks_comb, verbose=0)
print(f"Entrenamiento completo. Epochs ejecutados: {len(hist_comb.history['loss'])}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: ModelCheckpoint + EarlyStopping combinados.*

1. `ruta_comb = os.path.join(tempfile.gettempdir(), 'modelo_combinado.keras')` — Combina dos DataFrames por una columna clave.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: ReduceLROnPlateau — Reducir lr si val_loss estanca

```python
modelo_rl = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
modelo_rl.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

hist_rl = modelo_rl.fit(X, y, epochs=100, batch_size=32, validation_split=0.2,
                        callbacks=[reduce_lr], verbose=0)
print(f"LR final: {tf.keras.backend.get_value(modelo_rl.optimizer.learning_rate):.8f}")
print(f"LR inicial era 0.01")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: ReduceLROnPlateau — Reducir lr si val_loss estanca.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: ReduceLROnPlateau con factor=0.5, patience=3

```python
modelo_rl2 = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(1)
])
modelo_rl2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')

reduce_lr2 = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1,
    min_delta=0.0001,
    cooldown=2
)

X2 = np.random.rand(800, 5).astype(np.float32)
y2 = np.random.rand(800, 1).astype(np.float32)
hist_rl2 = modelo_rl2.fit(X2, y2, epochs=80, batch_size=32, validation_split=0.2,
                          callbacks=[reduce_lr2], verbose=0)
print(f"LR final después de reducciones: {tf.keras.backend.get_value(modelo_rl2.optimizer.learning_rate):.8f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: ReduceLROnPlateau con factor=0.5, patience=3.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: LearningRateScheduler — Schedule personalizado (decaimiento)

```python
def schedule_ventas(epoch, lr):
    """Reduce LR cada 10 epochs: lr * 0.9^(epoch/10)"""
    if epoch < 10:
        return lr
    return lr * 0.9

modelo_lrs = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(1)
])
modelo_lrs.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(schedule_ventas, verbose=1)

X_lrs = np.random.rand(600, 5).astype(np.float32)
y_lrs = np.random.rand(600, 1).astype(np.float32)
hist_lrs = modelo_lrs.fit(X_lrs, y_lrs, epochs=30, batch_size=32,
                          callbacks=[lr_scheduler], verbose=0)
print(f"LR histórico (primeros 5 epochs): {[round(lr, 6) for lr in hist_lrs.history.get('lr', ['N/A'])[:5]]}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: LearningRateScheduler — Schedule personalizado (decaimiento).*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: CSVLogger — Registrar historial de entrenamiento en CSV

```python
modelo_csv = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(1)
])
modelo_csv.compile(optimizer='adam', loss='mse', metrics=['mae'])

ruta_csv = os.path.join(tempfile.gettempdir(), 'historial_ventas.csv')
csv_logger = tf.keras.callbacks.CSVLogger(
    filename=ruta_csv,
    separator=',',
    append=False
)

X_csv = np.random.rand(500, 5).astype(np.float32)
y_csv = np.random.rand(500, 1).astype(np.float32)
hist_csv = modelo_csv.fit(X_csv, y_csv, epochs=20, batch_size=32, validation_split=0.2,
                          callbacks=[csv_logger], verbose=0)

with open(ruta_csv, 'r') as f:
    lineas = f.readlines()
print(f"CSV generado con {len(lineas)} líneas (1 header + {len(lineas)-1} epochs)")
print(f"Header: {lineas[0].strip()}")
print(f"Primer epoch: {lineas[1].strip()}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: CSVLogger — Registrar historial de entrenamiento en CSV.*

1. `ruta_csv = os.path.join(tempfile.gettempdir(), 'historial_ventas.csv')` — Combina dos DataFrames por una columna clave.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: TensorBoard — Visualizar gráficos de entrenamiento

```python
modelo_tb = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(1)
])
modelo_tb.compile(optimizer='adam', loss='mse')

log_dir = os.path.join(tempfile.gettempdir(), 'logs_ventas', 'fit')
tensorboard_cb = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=0,
    write_graph=True,
    update_freq='epoch'
)

X_tb = np.random.rand(300, 5).astype(np.float32)
y_tb = np.random.rand(300, 1).astype(np.float32)
hist_tb = modelo_tb.fit(X_tb, y_tb, epochs=10, batch_size=32, validation_split=0.2,
                        callbacks=[tensorboard_cb], verbose=0)
print(f"Logs de TensorBoard en: {log_dir}")
print(f"Para visualizar: tensorboard --logdir={log_dir}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: TensorBoard — Visualizar gráficos de entrenamiento.*

1. `log_dir = os.path.join(tempfile.gettempdir(), 'logs_ventas', 'fit')` — Combina dos DataFrames por una columna clave.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: TensorBoard con histogram_freq=1 (ver distribución de pesos)

```python
modelo_tb2 = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])
modelo_tb2.compile(optimizer='adam', loss='mse')

log_dir2 = os.path.join(tempfile.gettempdir(), 'logs_histogramas')
tensorboard_hist = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir2,
    histogram_freq=1,
    write_images=True,
    embeddings_freq=0
)

X_tb2 = np.random.rand(300, 5).astype(np.float32)
y_tb2 = np.random.rand(300, 1).astype(np.float32)
hist_tb2 = modelo_tb2.fit(X_tb2, y_tb2, epochs=5, batch_size=32,
                          callbacks=[tensorboard_hist], verbose=0)
print(f"Histogramas de pesos en: {log_dir2}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: TensorBoard con histogram_freq=1 (ver distribución de pesos).*

1. `log_dir2 = os.path.join(tempfile.gettempdir(), 'logs_histogramas')` — Combina dos DataFrames por una columna clave.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: LambdaCallback — Imprimir métricas personalizadas de ventas

```python
modelo_lambda = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(1)
])
modelo_lambda.compile(optimizer='adam', loss='mse')

def imprimir_metricas(epoch, logs):
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}: loss={logs['loss']:.4f}, val_loss={logs.get('val_loss', 'N/A')}")

lambda_cb = tf.keras.callbacks.LambdaCallback(on_epoch_end=imprimir_metricas)

X_lam = np.random.rand(200, 5).astype(np.float32)
y_lam = np.random.rand(200, 1).astype(np.float32)
hist_lam = modelo_lambda.fit(X_lam, y_lam, epochs=20, batch_size=16, validation_split=0.2,
                             callbacks=[lambda_cb], verbose=0)
print("Callback Lambda ejecutado cada 5 epochs")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: LambdaCallback — Imprimir métricas personalizadas de ventas.*

1. `print(f"Epoch {epoch+1}: loss={logs['loss']:.4f}, val_loss={logs.get('val_loss', 'N/A')}")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: LambdaCallback on_epoch_end — Enviar alerta si pérdida alta

```python
modelo_alerta = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(1)
])
modelo_alerta.compile(optimizer='adam', loss='mse')

umbral_perdida = 5.0

def alerta_perdida(epoch, logs):
    if logs['loss'] > umbral_perdida:
        print(f"⚠️ ALERTA: Pérdida alta ({logs['loss']:.2f}) en epoch {epoch+1}")
    if 'val_loss' in logs and logs['val_loss'] > umbral_perdida:
        print(f"⚠️ ALERTA: Val loss alto ({logs['val_loss']:.2f}) en epoch {epoch+1}")

alerta_cb = tf.keras.callbacks.LambdaCallback(on_epoch_end=alerta_perdida)

X_al = np.random.rand(100, 5).astype(np.float32)
y_al = np.random.rand(100, 1).astype(np.float32) * 10
hist_al = modelo_alerta.fit(X_al, y_al, epochs=5, batch_size=16,
                            callbacks=[alerta_cb], verbose=0)
print("Callback de alerta ejecutado")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: LambdaCallback on_epoch_end — Enviar alerta si pérdida alta.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: TerminateOnNaN — Detener si pérdida es NaN

```python
modelo_nan = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(1)
])
modelo_nan.compile(optimizer='adam', loss='mse')

terminate_nan = tf.keras.callbacks.TerminateOnNaN()

# Datos normales (sin NaN)
X_nan = np.random.rand(500, 5).astype(np.float32)
y_nan = np.random.rand(500, 1).astype(np.float32)
hist_nan = modelo_nan.fit(X_nan, y_nan, epochs=30, batch_size=32,
                          callbacks=[terminate_nan], verbose=0)
print(f"Entrenamiento completó {len(hist_nan.history['loss'])} epochs sin NaN")
print(f"Loss final: {hist_nan.history['loss'][-1]:.6f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: TerminateOnNaN — Detener si pérdida es NaN.*

1. Datos normales (sin NaN)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 16: BackupAndRestore — Recuperar entrenamiento interrumpido

```python
modelo_backup = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(1)
])
modelo_backup.compile(optimizer='adam', loss='mse')

backup_dir = os.path.join(tempfile.gettempdir(), 'backup_entrenamiento')
backup_cb = tf.keras.callbacks.BackupAndRestore(
    backup_dir=backup_dir,
    save_freq='epoch'
)

X_bk = np.random.rand(400, 5).astype(np.float32)
y_bk = np.random.rand(400, 1).astype(np.float32)
hist_bk = modelo_backup.fit(X_bk, y_bk, epochs=15, batch_size=32,
                            callbacks=[backup_cb], verbose=0)
print(f"Backup guardado en: {backup_dir}")
print(f"Archivos: {os.listdir(backup_dir)}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: BackupAndRestore — Recuperar entrenamiento interrumpido.*

1. `backup_dir = os.path.join(tempfile.gettempdir(), 'backup_entrenamiento')` — Combina dos DataFrames por una columna clave.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 17: Múltiples callbacks combinados en lista para entrenamiento robusto

```python
modelo_multi_cb = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
modelo_multi_cb.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='mse',
    metrics=['mae']
)

ruta_multi = os.path.join(tempfile.gettempdir(), 'modelo_multi_cb.keras')
ruta_csv_multi = os.path.join(tempfile.gettempdir(), 'historial_multi.csv')
log_dir_multi = os.path.join(tempfile.gettempdir(), 'logs_multi')

callbacks_multi = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(ruta_multi, monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
    tf.keras.callbacks.CSVLogger(ruta_csv_multi),
    tf.keras.callbacks.TensorBoard(log_dir=log_dir_multi),
    tf.keras.callbacks.TerminateOnNaN()
]

X_mc = np.random.rand(1000, 5).astype(np.float32)
y_mc = np.random.rand(1000, 1).astype(np.float32)
hist_mc = modelo_multi_cb.fit(X_mc, y_mc, epochs=200, batch_size=32, validation_split=0.2,
                              callbacks=callbacks_multi, verbose=0)
print(f"Epochs ejecutados: {len(hist_mc.history['loss'])} (early stopping detuvo)")
print(f"Mejor val_loss: {min(hist_mc.history['val_loss']):.6f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: Múltiples callbacks combinados en lista para entrenamiento robusto.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 18: Integrador — Entrenamiento completo con todos los callbacks para predicción de demanda

```python
# Dataset de demanda con 8 features
np.random.seed(42)
tf.random.set_seed(42)
n = 3000
X_dem = np.random.randn(n, 8).astype(np.float32)
y_dem = (3 * X_dem[:, 0] - 2 * X_dem[:, 2] + 0.5 * X_dem[:, 4] +
         np.random.normal(0, 0.5, n)).astype(np.float32)

modelo_final = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(8,),
                          kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
modelo_final.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
    loss='mse',
    metrics=['mae']
)

# Directorios
base_dir = tempfile.gettempdir()
ruta_final = os.path.join(base_dir, 'modelo_demanda_final.keras')
ruta_csv_final = os.path.join(base_dir, 'historial_demanda.csv')
log_dir_final = os.path.join(base_dir, 'logs_demanda_final')
backup_dir_final = os.path.join(base_dir, 'backup_demanda')

def lr_schedule_demanda(epoch, lr):
    if epoch < 20:
        return lr
    return lr * 0.95

callbacks_final = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=20, min_delta=0.0001,
        restore_best_weights=True, mode='min', verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        ruta_final, monitor='val_loss', save_best_only=True, mode='min', verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=8,
        min_lr=1e-7, cooldown=3, verbose=1
    ),
    tf.keras.callbacks.LearningRateScheduler(lr_schedule_demanda, verbose=0),
    tf.keras.callbacks.CSVLogger(ruta_csv_final, append=False),
    tf.keras.callbacks.TensorBoard(log_dir=log_dir_final, histogram_freq=1, write_graph=True),
    tf.keras.callbacks.BackupAndRestore(backup_dir_final),
    tf.keras.callbacks.TerminateOnNaN(),
    tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: print(
            f"Ep.{epoch+1}: loss={logs['loss']:.4f}, val_loss={logs['val_loss']:.4f}, "
            f"mae={logs['mae']:.4f}"
        ) if (epoch + 1) % 20 == 0 else None
    )
]

hist_final = modelo_final.fit(
    X_dem, y_dem, epochs=300, batch_size=64,
    validation_split=0.2, callbacks=callbacks_final, verbose=0
)

print(f"\n--- RESULTADOS FINALES ---")
print(f"Epochs totales: {len(hist_final.history['loss'])}")
best_epoch = np.argmin(hist_final.history['val_loss'])
print(f"Mejor val_loss en epoch {best_epoch+1}: {hist_final.history['val_loss'][best_epoch]:.6f}")
print(f"Mejor val_mae: {hist_final.history['val_mae'][best_epoch]:.6f}")

# Evaluación final
X_test_f = np.random.randn(500, 8).astype(np.float32)
y_test_f = (3 * X_test_f[:, 0] - 2 * X_test_f[:, 2] + 0.5 * X_test_f[:, 4] +
            np.random.normal(0, 0.5, 500)).astype(np.float32)
test_loss, test_mae = modelo_final.evaluate(X_test_f, y_test_f, verbose=0)
print(f"Test MSE: {test_loss:.4f}, Test MAE: {test_mae:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador — Entrenamiento completo con todos los callbacks para predicción de demanda.*

1. Dataset de demanda con 8 features
2. Directorios
3. Evaluación final

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejercicios

1. **EarlyStopping para clasificación de productos**: Entrena un modelo Sequential de 3 capas para clasificar productos en 3 categorías. Usa EarlyStopping con patience=8, monitor='val_accuracy', mode='max', restore_best_weights=True.

2. **ModelCheckpoint por accuracy**: Configura ModelCheckpoint para guardar el modelo solo cuando val_accuracy mejore (mode='max'). Entrénalo en un dataset de clasificación binaria de compras.

3. **ReduceLROnPlateau agresivo**: En un modelo de regresión para predecir lead time de proveedores, configura ReduceLROnPlateau con factor=0.2, patience=2, min_lr=1e-8. Observa cómo el LR cae rápidamente.

4. **LearningRateScheduler exponencial**: Implementa un scheduler que divida el LR por 2 cada 5 epochs: `lr * 0.5^(epoch/5)`. Entrena un modelo de predicción de inventario y registra el LR en cada epoch.

5. **CSVLogger con append=True**: Entrena un modelo en dos fases (20 epochs cada una) usando CSVLogger con append=True. Verifica que el CSV resultante tenga 40 filas de datos.

6. **TensorBoard con embeddings**: Crea un modelo que tenga una capa Embedding para 10 categorías de productos con dimensión 4. Configura TensorBoard con embeddings_freq=1 para visualizar los embeddings.

7. **LambdaCallback personalizado**: Crea un LambdaCallback que en on_train_begin imprima "Iniciando entrenamiento de modelo de ventas", en on_epoch_end calcule y muestre el ratio de mejora respecto al epoch anterior, y en on_train_end imprima "Entrenamiento completado".

8. **Integrador con EarlyStopping + ReduceLROnPlateau + ModelCheckpoint**: Diseña un entrenamiento completo para un modelo de ventas con 10 features que use: EarlyStopping (patience=25, min_delta=0.0005), ReduceLROnPlateau (factor=0.3, patience=5), ModelCheckpoint (save_best_only), CSVLogger y TensorBoard. Reporta la mejor epoch, mejor val_loss y test MAE.

---

## Resumen

Los callbacks de TensorFlow son herramientas esenciales para controlar y optimizar el entrenamiento de modelos de deep learning en contextos de ventas, compras e inventarios:

- **EarlyStopping**: Evita overfitting deteniendo el entrenamiento cuando la métrica de validación deja de mejorar. Clave para ahorrar tiempo en modelos que convergen rápido.
- **ModelCheckpoint**: Guarda el mejor modelo (o pesos) durante el entrenamiento. Útil para recuperar la mejor versión después de early stopping.
- **ReduceLROnPlateau**: Reduce la tasa de aprendizaje automáticamente cuando el entrenamiento se estanca. Ayuda a escapar de mínimos locales.
- **LearningRateScheduler**: Control manual del learning rate con schedules personalizados (decaimiento exponencial, escalonado, cíclico).
- **CSVLogger**: Registro persistente de métricas para auditoría y análisis post-entrenamiento.
- **TensorBoard**: Visualización en tiempo real de pérdidas, métricas, histogramas de pesos y embeddings.
- **LambdaCallback**: Flexibilidad total para ejecutar código arbitrario en cualquier punto del entrenamiento.
- **TerminateOnNaN**: Seguridad contra divergencia numérica.
- **BackupAndRestore**: Resiliencia ante interrupciones en entrenamientos largos.

Combinar múltiples callbacks (especialmente EarlyStopping + ModelCheckpoint + ReduceLROnPlateau) es una práctica recomendada para cualquier entrenamiento de producción.
