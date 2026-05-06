# A28 - Deep Learning: Autoencoders para Inventarios

## Fundamentos Teóricos

Un **Autoencoder (AE)** es una red neuronal que aprende a reconstruir su entrada. Comprime la información en un espacio latente de menor dimensión (bottleneck) y luego la reconstruye.

### Arquitectura
```
Input → Encoder → Espacio Latente (bottleneck) → Decoder → Reconstrucción
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Arquitectura.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Tipos de Autoencoders

| Tipo | Descripción | Aplicación en inventarios |
|------|-------------|--------------------------|
| **Undercomplete AE** | bottleneck < input | Reducción de dimensionalidad de productos |
| **Denoising AE** | entrada con ruido → salida limpia | Limpiar datos de ventas ruidosos |
| **Sparse AE** | regularización L1 en activaciones latentes | Selección de features relevantes |
| **Stacked AE** | múltiples capas profundas | Representaciones jerárquicas |
| **Variational AE (VAE)** | distribución latente probabilística | Generación de productos sintéticos |
| **ConvAE** | convolucional para series | Patrones temporales en ventas |
| **LSTM-AE** | recurrente para secuencias | Anomalías en series temporales |

### Componentes Clave
- **Encoder**: red que mapea entrada → espacio latente
- **Decoder**: red que mapea espacio latente → reconstrucción
- **Bottleneck**: capa más pequeña que fuerza la compresión
- **Reconstruction error**: ||x - x̂||² (MSE entre entrada y salida)

### Anomaly Detection con AE
1. Entrenar AE solo con datos "normales"
2. Calcular reconstruction error para cada muestra
3. Si error > threshold → es anomalía
4. Threshold típico: percentil 95-99 de errores normales

### Variational Autoencoder (VAE)
- Codifica la entrada como distribución (μ, σ) en lugar de punto
- Usa **reparameterization trick**: z = μ + σ * ε, ε ~ N(0,1)
- Loss = reconstruction_loss + KL_divergence
- KL divergence: fuerza la distribución latente hacia N(0,1)
- Permite **generar** nuevos datos muestreando del espacio latente

---

## Ejemplos Prácticos

### Ejemplo 1: Undercomplete AE — Comprimir Features de Productos (10 → 3)

```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

np.random.seed(42)
tf.random.set_seed(42)

# 10 features de productos: precio, costo, peso, volumen, demanda_promedio,
# lead_time, stock_seguridad, rotacion, margen, temporada
n_productos = 2000
X_prod = np.random.randn(n_productos, 10)

# Autoencoder: 10 → 5 → 3 → 5 → 10
input_dim = 10
encoding_dim = 3

entrada = Input(shape=(input_dim,))
encoded = Dense(5, activation='relu')(entrada)
encoded = Dense(encoding_dim, activation='relu')(encoded)  # bottleneck
decoded = Dense(5, activation='relu')(encoded)
decoded = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(inputs=entrada, outputs=decoded)
encoder = Model(inputs=entrada, outputs=encoded)

autoencoder.compile(optimizer=Adam(0.001), loss='mse')
autoencoder.summary()

h = autoencoder.fit(X_prod, X_prod, epochs=50, batch_size=64,
                    validation_split=0.2, verbose=1)

# Reconstrucción
X_recon = autoencoder.predict(X_prod, verbose=0)
error = np.mean((X_prod - X_recon)**2, axis=1)
print(f"Reconstruction error promedio: {error.mean():.6f}")

# Espacio latente 3D
latente = encoder.predict(X_prod, verbose=0)
print(f"Espacio latente shape: {latente.shape}")  # (2000, 3)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: Undercomplete AE — Comprimir Features de Productos (10 → 3).*

1. 10 features de productos: precio, costo, peso, volumen, demanda_promedio,
2. lead_time, stock_seguridad, rotacion, margen, temporada
3. Autoencoder: 10 → 5 → 3 → 5 → 10
4. Reconstrucción
5. Espacio latente 3D

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: Autoencoder vs PCA — Reducción de Dimensionalidad

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X_prod)

# PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)
X_pca_recon = pca.inverse_transform(X_pca)
error_pca = np.mean((X_scaled - X_pca_recon)**2, axis=1)

# AE (reutilizando el modelo anterior)
X_ae_scaled = StandardScaler().fit_transform(X_prod)
X_ae_recon = autoencoder.predict(X_ae_scaled, verbose=0)
error_ae = np.mean((X_ae_scaled - X_ae_recon)**2, axis=1)

print(f"Reconstruction Error - PCA: {error_pca.mean():.6f}")
print(f"Reconstruction Error - AE:  {error_ae.mean():.6f}")
print(f"AE es mejor si error_ae < error_pca")

# Varianza explicada por PCA
print(f"Varianza explicada PCA 3 componentes: {pca.explained_variance_ratio_.sum():.2%}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: Autoencoder vs PCA — Reducción de Dimensionalidad.*

1. PCA
2. AE (reutilizando el modelo anterior)
3. Varianza explicada por PCA

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: Denoising AE — Agregar Ruido Gaussiano a Precios

```python
# Datos de precios de productos con estructura subyacente
np.random.seed(42)
tf.random.set_seed(42)

n = 1000
X_limpio = np.random.randn(n, 8)
# Ruido gaussiano
ruido_factor = 0.3
X_ruidoso = X_limpio + ruido_factor * np.random.randn(n, 8)

# Denoising AE
input_dim = 8
entrada = Input(shape=(input_dim,))
enc = Dense(4, activation='relu')(entrada)
bottleneck = Dense(2, activation='relu')(enc)
dec = Dense(4, activation='relu')(bottleneck)
salida = Dense(input_dim, activation='linear')(dec)

denoising_ae = Model(inputs=entrada, outputs=salida)
denoising_ae.compile(optimizer=Adam(0.001), loss='mse')
denoising_ae.summary()

# Entrenar: entrada ruidosa → salida limpia
h_den = denoising_ae.fit(X_ruidoso, X_limpio, epochs=50, batch_size=32,
                         validation_split=0.2, verbose=1)

# Evaluar
X_reconstruido = denoising_ae.predict(X_ruidoso, verbose=0)
mse_ruidoso = np.mean((X_limpio - X_ruidoso)**2)
mse_reconstruido = np.mean((X_limpio - X_reconstruido)**2)
print(f"MSE entrada ruidosa vs limpia: {mse_ruidoso:.6f}")
print(f"MSE reconstruido vs limpia:    {mse_reconstruido:.6f}")
print(f"Mejora: {(1 - mse_reconstruido/mse_ruidoso)*100:.1f}%")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: Denoising AE — Agregar Ruido Gaussiano a Precios.*

1. Datos de precios de productos con estructura subyacente
2. Ruido gaussiano
3. Denoising AE
4. Entrenar: entrada ruidosa → salida limpia
5. Evaluar

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: Denoising AE — Diferentes Niveles de Ruido (0.1, 0.3, 0.5)

```python
ruidos = [0.1, 0.3, 0.5]
errores_reconstruccion = []

for ruido in ruidos:
    X_ruidoso = X_limpio + ruido * np.random.randn(n, 8)

    ae = Sequential([
        Dense(6, activation='relu', input_shape=(8,)),
        Dense(3, activation='relu'),
        Dense(6, activation='relu'),
        Dense(8, activation='linear')
    ])
    ae.compile(optimizer=Adam(0.001), loss='mse')
    ae.fit(X_ruidoso, X_limpio, epochs=30, batch_size=32, validation_split=0.2, verbose=0)

    X_recon = ae.predict(X_ruidoso, verbose=0)
    err = np.mean((X_limpio - X_recon)**2)
    errores_reconstruccion.append(err)
    print(f"Ruido={ruido:.1f} | Reconstruction MSE: {err:.6f}")

plt.bar([str(r) for r in ruidos], errores_reconstruccion)
plt.xlabel('Nivel de ruido')
plt.ylabel('Reconstruction MSE')
plt.title('Denoising AE: error por nivel de ruido')
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: Denoising AE — Diferentes Niveles de Ruido (0.1, 0.3, 0.5).*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: Sparse AE — Regularización L1 en Activaciones Latentes

```python
from tensorflow.keras.regularizers import l1
from tensorflow.keras.layers import ActivityRegularization

entrada = Input(shape=(10,))
enc = Dense(8, activation='relu')(entrada)
# Actividad regularizada con L1 para forzar sparse activations
bottleneck = Dense(5, activation='relu', activity_regularizer=l1(1e-4))(enc)
dec = Dense(8, activation='relu')(bottleneck)
salida = Dense(10, activation='linear')(dec)

sparse_ae = Model(inputs=entrada, outputs=salida)
sparse_ae.compile(optimizer=Adam(0.001), loss='mse')
sparse_ae.summary()

h_sparse = sparse_ae.fit(X_prod, X_prod, epochs=50, batch_size=32,
                         validation_split=0.2, verbose=1)

# Ver esparcidad del bottleneck
encoder_sparse = Model(inputs=entrada, outputs=bottleneck)
latente_sparse = encoder_sparse.predict(X_prod, verbose=0)
print(f"Fracción de activaciones cercanas a cero: {(np.abs(latente_sparse) < 0.1).mean():.2%}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: Sparse AE — Regularización L1 en Activaciones Latentes.*

1. Actividad regularizada con L1 para forzar sparse activations
2. Ver esparcidad del bottleneck

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: Stacked AE — 3 Capas de Encoder y 3 de Decoder

```python
stacked_ae = Sequential([
    # Encoder
    Dense(32, activation='relu', input_shape=(20,)),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),  # bottleneck
    # Decoder
    Dense(16, activation='relu'),
    Dense(32, activation='relu'),
    Dense(20, activation='linear')
])

stacked_ae.compile(optimizer=Adam(0.001), loss='mse')
stacked_ae.summary()

# Datos de 20 features
X_20 = np.random.randn(2000, 20)
h_stack = stacked_ae.fit(X_20, X_20, epochs=50, batch_size=32,
                         validation_split=0.2, verbose=1)
print(f"Stacked AE reconstruction loss: {min(h_stack.history['val_loss']):.6f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: Stacked AE — 3 Capas de Encoder y 3 de Decoder.*

1. Encoder
2. Decoder
3. Datos de 20 features

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: Visualizar Espacio Latente 2D (Scatterplot de Productos)

```python
from sklearn.datasets import make_blobs

# Simular 4 categorías de productos
X_viz, y_viz = make_blobs(n_samples=500, n_features=10, centers=4, random_state=42)

# AE con bottleneck 2D para visualización
entrada = Input(shape=(10,))
bottleneck_2d = Dense(2, activation='linear', name='bottleneck')(Dense(6, activation='relu')(entrada))
dec = Dense(6, activation='relu')(bottleneck_2d)
salida = Dense(10, activation='linear')(dec)

ae_viz = Model(inputs=entrada, outputs=salida)
encoder_viz = Model(inputs=entrada, outputs=bottleneck_2d)

ae_viz.compile(optimizer=Adam(0.001), loss='mse')
ae_viz.fit(X_viz, X_viz, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

latente_viz = encoder_viz.predict(X_viz, verbose=0)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
scatter = plt.scatter(latente_viz[:, 0], latente_viz[:, 1], c=y_viz, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Categoría')
plt.xlabel('Dimensión latente 1')
plt.ylabel('Dimensión latente 2')
plt.title('Espacio latente 2D (Autoencoder)')

plt.subplot(1, 2, 2)
pca_viz = PCA(n_components=2)
X_pca_viz = pca_viz.fit_transform(X_viz)
plt.scatter(X_pca_viz[:, 0], X_pca_viz[:, 1], c=y_viz, cmap='viridis', alpha=0.7)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Espacio 2D (PCA)')
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 7: Visualizar Espacio Latente 2D (Scatterplot de Productos).*

1. Simular 4 categorías de productos
2. AE con bottleneck 2D para visualización

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: VAE — Variational Autoencoder con KL Divergence

```python
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse

# Parámetros
original_dim = 10
intermediate_dim = 8
latent_dim = 3

# Encoder
entrada = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(entrada)
z_mean = Dense(latent_dim, name='z_mean')(h)
z_log_var = Dense(latent_dim, name='z_log_var')(h)

# Reparameterization trick
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling, name='z')([z_mean, z_log_var])

# Decoder
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_out = Dense(original_dim, activation='linear')
h_decoded = decoder_h(z)
salida = decoder_out(h_decoded)

vae = Model(inputs=entrada, outputs=salida)

# Loss: reconstrucción + KL
reconstruction_loss = mse(entrada, salida) * original_dim
kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(reconstruction_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer=Adam(0.001))
vae.summary()

X_vae = np.random.randn(2000, original_dim)
h_vae = vae.fit(X_vae, epochs=50, batch_size=64, validation_split=0.2, verbose=1)
```

**Salida:**

```
# El modelo ajusta sus parámetros durante el entrenamiento.
# Los parámetros aprendidos quedan disponibles en el objeto.
```

**Explicación línea por línea:**

*Ejemplo 8: VAE — Variational Autoencoder con KL Divergence.*

1. Parámetros
2. Encoder
3. Reparameterization trick
4. Decoder
5. Loss: reconstrucción + KL

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: VAE — Generar Nuevos Productos Sintéticos

```python
# Modelo generativo (solo decoder)
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_generated = decoder_out(_h_decoded)

generator = Model(inputs=decoder_input, outputs=_generated)
generator.compile(optimizer=Adam(0.001), loss='mse')

# Generar productos muestreando de N(0,1)
n_nuevos = 10
z_samples = np.random.randn(n_nuevos, latent_dim)
nuevos_productos = generator.predict(z_samples, verbose=0)

print("Productos sintéticos generados (10 muestras):")
for i, prod in enumerate(nuevos_productos):
    print(f"  Producto {i+1}: {prod[:5]}... (mostrando 5 de {original_dim} features)")

# Interpolación entre 2 productos en espacio latente
z1 = np.random.randn(1, latent_dim)
z2 = np.random.randn(1, latent_dim)
alphas = np.linspace(0, 1, 5)
interpolados = []
for alpha in alphas:
    z_interp = (1 - alpha) * z1 + alpha * z2
    interp = generator.predict(z_interp, verbose=0)
    interpolados.append(interp)
print(f"\nInterpolación generada: {len(interpolados)} puntos entre dos productos latentes")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: VAE — Generar Nuevos Productos Sintéticos.*

1. Modelo generativo (solo decoder)
2. Generar productos muestreando de N(0,1)
3. Interpolación entre 2 productos en espacio latente

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: ConvAE — Autoencoder Convolucional para Series de Ventas

```python
from tensorflow.keras.layers import Conv1D, MaxPooling1D, UpSampling1D, Reshape

# Series de ventas: 60 días de 3 features
n_series = 1000
X_series = np.random.randn(n_series, 60, 3)

# ConvAE
conv_ae = Sequential([
    # Encoder convolucional
    Conv1D(16, 3, activation='relu', padding='same', input_shape=(60, 3)),
    MaxPooling1D(2),  # 60 → 30
    Conv1D(8, 3, activation='relu', padding='same'),
    MaxPooling1D(2),  # 30 → 15
    # Decoder convolucional
    Conv1D(8, 3, activation='relu', padding='same'),
    UpSampling1D(2),  # 15 → 30
    Conv1D(16, 3, activation='relu', padding='same'),
    UpSampling1D(2),  # 30 → 60
    Conv1D(3, 3, activation='linear', padding='same')  # 3 features
])

conv_ae.compile(optimizer=Adam(0.001), loss='mse')
conv_ae.summary()

h_convae = conv_ae.fit(X_series, X_series, epochs=30, batch_size=32,
                       validation_split=0.2, verbose=1)
print(f"ConvAE reconstruction loss: {min(h_convae.history['val_loss']):.6f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: ConvAE — Autoencoder Convolucional para Series de Ventas.*

1. Series de ventas: 60 días de 3 features
2. ConvAE
3. Encoder convolucional
4. Decoder convolucional

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: LSTM-AE — Autoencoder Recurrente para Series Temporales

```python
from tensorflow.keras.layers import RepeatVector, TimeDistributed

# LSTM Autoencoder
timesteps, n_features = 30, 3
X_lstmae = np.random.randn(800, timesteps, n_features)

lstm_ae = Sequential([
    # Encoder recurrente
    LSTM(32, activation='tanh', input_shape=(timesteps, n_features)),
    RepeatVector(timesteps),  # repetir para decoder
    # Decoder recurrente
    LSTM(32, activation='tanh', return_sequences=True),
    TimeDistributed(Dense(n_features))
])

lstm_ae.compile(optimizer=Adam(0.001), loss='mse')
lstm_ae.summary()

h_lstmae = lstm_ae.fit(X_lstmae, X_lstmae, epochs=30, batch_size=32,
                       validation_split=0.2, verbose=1)
print(f"LSTM-AE reconstruction loss: {min(h_lstmae.history['val_loss']):.6f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: LSTM-AE — Autoencoder Recurrente para Series Temporales.*

1. LSTM Autoencoder
2. Encoder recurrente
3. Decoder recurrente

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: Detección de Anomalías — Reconstruction Error por Producto

```python
np.random.seed(42)
tf.random.set_seed(42)

# Datos normales de productos
n_norm = 1000
X_norm = np.random.randn(n_norm, 15)

# Entrenar AE solo con datos normales
ae_norm = Sequential([
    Dense(10, activation='relu', input_shape=(15,)),
    Dense(5, activation='relu'),
    Dense(10, activation='relu'),
    Dense(15, activation='linear')
])
ae_norm.compile(optimizer=Adam(0.001), loss='mse')
ae_norm.fit(X_norm, X_norm, epochs=30, batch_size=32, validation_split=0.2, verbose=0)

# Datos anómalos: productos con comportamiento extraño
n_anom = 50
X_anom = np.random.randn(n_anom, 15) + 3.0  # shift de 3 sigma

# Calcular reconstruction error
X_all = np.vstack([X_norm, X_anom])
labels = np.array([0]*n_norm + [1]*n_anom)

X_all_recon = ae_norm.predict(X_all, verbose=0)
errors = np.mean((X_all - X_all_recon)**2, axis=1)

# Estadísticas
print(f"Error normal   - media: {errors[:n_norm].mean():.4f}, std: {errors[:n_norm].std():.4f}")
print(f"Error anómalo  - media: {errors[n_norm:].mean():.4f}, std: {errors[n_norm:].std():.4f}")

plt.hist(errors[:n_norm], bins=30, alpha=0.7, label='Normal', density=True)
plt.hist(errors[n_norm:], bins=15, alpha=0.7, label='Anómalo', density=True)
plt.xlabel('Reconstruction Error')
plt.ylabel('Densidad')
plt.title('Detección de anomalías con AE')
plt.legend()
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: Detección de Anomalías — Reconstruction Error por Producto.*

1. Datos normales de productos
2. Entrenar AE solo con datos normales
3. Datos anómalos: productos con comportamiento extraño
4. Calcular reconstruction error
5. Estadísticas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: Threshold de Anomalía — Percentil 95

```python
# Determinar threshold basado en datos normales
errors_norm = ae_norm.predict(X_norm, verbose=0)
errors_norm_mse = np.mean((X_norm - errors_norm)**2, axis=1)

threshold_p95 = np.percentile(errors_norm_mse, 95)
threshold_p99 = np.percentile(errors_norm_mse, 99)

print(f"Threshold percentil 95: {threshold_p95:.4f}")
print(f"Threshold percentil 99: {threshold_p99:.4f}")

# Clasificar anomalías
predicciones_p95 = (errors > threshold_p95).astype(int)
predicciones_p99 = (errors > threshold_p99).astype(int)

from sklearn.metrics import classification_report
print("\n=== Clasificación con threshold P95 ===")
print(classification_report(labels, predicciones_p95, target_names=['Normal', 'Anómalo']))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: Threshold de Anomalía — Percentil 95.*

1. Determinar threshold basado en datos normales
2. Clasificar anomalías

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: Anomalías en Inventario — Productos con Comportamiento Extraño

```python
np.random.seed(42)

# Datos de inventario: [rotacion, stock_seguridad, lead_time, costo_mantenimiento,
#                       demanda_promedio, variabilidad, devoluciones, rupturas]
n_prod = 500
X_inv = np.random.randn(n_prod, 8)

# Algunos productos anómalos (índices 400-449)
X_inv[400:450] += np.array([2.0, -1.5, 2.5, 1.8, -2.0, 3.0, 2.2, 1.5])

ae_inv = Sequential([
    Dense(6, activation='relu', input_shape=(8,)),
    Dense(3, activation='relu'),
    Dense(6, activation='relu'),
    Dense(8, activation='linear')
])
ae_inv.compile(optimizer=Adam(0.001), loss='mse')
ae_inv.fit(X_inv[:400], X_inv[:400], epochs=40, batch_size=32, verbose=0)

X_recon = ae_inv.predict(X_inv, verbose=0)
errors_inv = np.mean((X_inv - X_recon)**2, axis=1)

threshold = np.percentile(errors_inv[:400], 95)
anomalos = np.where(errors_inv > threshold)[0]

print(f"Productos anómalos detectados: {len(anomalos)} de {n_prod}")
print(f"Threshold (P95): {threshold:.4f}")
print(f"Índices anómalos: {sorted(anomalos)}")
print(f"Esperado: productos 400-449 (anómalos inyectados)")

# Visualizar
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.scatter(range(n_prod), errors_inv, c=['red' if i in anomalos else 'blue' for i in range(n_prod)], alpha=0.6)
plt.axhline(threshold, color='k', linestyle='--', label=f'Threshold (P95)')
plt.xlabel('Producto')
plt.ylabel('Reconstruction Error')
plt.title('Detección de anomalías en inventario')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(errors_inv[:400], bins=30, alpha=0.7, label='Normal')
plt.hist(errors_inv[400:], bins=20, alpha=0.7, label='Anómalo')
plt.axvline(threshold, color='k', linestyle='--', label=f'Threshold')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frecuencia')
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

*Ejemplo 14: Anomalías en Inventario — Productos con Comportamiento Extraño.*

1. Datos de inventario: [rotacion, stock_seguridad, lead_time, costo_mantenimiento,
2. demanda_promedio, variabilidad, devoluciones, rupturas]
3. Algunos productos anómalos (índices 400-449)
4. Visualizar

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: Interpolación en Espacio Latente entre Dos Productos

```python
# Obtener 2 productos específicos
prod_a = X_prod[0:1]
prod_b = X_prod[100:101]

# Extraer sus representaciones latentes
encoder_int = encoder  # del ejemplo 1
lat_a = encoder_int.predict(prod_a, verbose=0)
lat_b = encoder_int.predict(prod_b, verbose=0)

# Interpolar en espacio latente
n_pasos = 8
alphas = np.linspace(0, 1, n_pasos)
interpolaciones = []

for alpha in alphas:
    z_interp = (1 - alpha) * lat_a + alpha * lat_b
    # Decodificar (necesitamos el decoder)
    # Extraer decoder del autoencoder
    decoder_int = Sequential([
        Dense(5, activation='relu', input_shape=(3,)),
        Dense(10, activation='linear')
    ])
    # Copiar pesos: el decoder son las últimas 2 capas del autoencoder
    decoder_int.layers[0].set_weights(autoencoder.layers[3].get_weights())
    decoder_int.layers[1].set_weights(autoencoder.layers[4].get_weights())
    prod_interp = decoder_int.predict(z_interp, verbose=0)
    interpolaciones.append(prod_interp)

print(f"Interpolación: Producto A → Producto B en {n_pasos} pasos")
for i, (alpha, prod) in enumerate(zip(alphas, interpolaciones)):
    print(f"  Paso {i+1} (α={alpha:.2f}): {prod[0, :4]}...")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: Interpolación en Espacio Latente entre Dos Productos.*

1. Obtener 2 productos específicos
2. Extraer sus representaciones latentes
3. Interpolar en espacio latente
4. Decodificar (necesitamos el decoder)
5. Extraer decoder del autoencoder
6. Copiar pesos: el decoder son las últimas 2 capas del autoencoder

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 16: Comparar AE vs PCA en Calidad de Reconstrucción

```python
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
X_comp = np.random.randn(1000, 20)

# PCA con diferentes componentes
componentes = [2, 5, 10, 15]
errores_pca = []
errores_ae = []

for c in componentes:
    # PCA
    pca = PCA(n_components=c)
    X_pca = pca.fit_transform(X_comp)
    X_pca_recon = pca.inverse_transform(X_pca)
    error_pca = mean_squared_error(X_comp, X_pca_recon)
    errores_pca.append(error_pca)

    # AE
    ae_comp = Sequential([
        Dense(16, activation='relu', input_shape=(20,)),
        Dense(c, activation='relu'),
        Dense(16, activation='relu'),
        Dense(20, activation='linear')
    ])
    ae_comp.compile(optimizer=Adam(0.01), loss='mse')
    ae_comp.fit(X_comp, X_comp, epochs=20, batch_size=32, verbose=0)
    X_ae_recon = ae_comp.predict(X_comp, verbose=0)
    error_ae = mean_squared_error(X_comp, X_ae_recon)
    errores_ae.append(error_ae)

    print(f"Componentes={c:2d} | PCA error: {error_pca:.6f} | AE error: {error_ae:.6f}")

plt.plot(componentes, errores_pca, 'o-', label='PCA')
plt.plot(componentes, errores_ae, 's-', label='AE')
plt.xlabel('Dimensiones latentes')
plt.ylabel('Reconstruction MSE')
plt.title('AE vs PCA: calidad de reconstrucción')
plt.legend()
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: Comparar AE vs PCA en Calidad de Reconstrucción.*

1. PCA con diferentes componentes
2. PCA
3. AE

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 17: Fine-tuning de AE para Transfer Learning

```python
np.random.seed(42)
tf.random.set_seed(42)

# Pre-entrenar AE en datos genéricos de productos (100 features)
X_generico = np.random.randn(3000, 100)

ae_pretrain = Sequential([
    Dense(64, activation='relu', input_shape=(100,)),
    Dense(32, activation='relu'),
    Dense(10, activation='relu'),  # bottleneck
    Dense(32, activation='relu'),
    Dense(64, activation='relu'),
    Dense(100, activation='linear')
])
ae_pretrain.compile(optimizer=Adam(0.001), loss='mse')
ae_pretrain.fit(X_generico, X_generico, epochs=50, batch_size=64,
                validation_split=0.2, verbose=0)

# Fine-tuning en datos específicos de electrónicos (más pequeños)
X_especifico = np.random.randn(200, 100)
ae_pretrain.fit(X_especifico, X_especifico, epochs=20, batch_size=16, verbose=0)
print("Fine-tuning completado: AE pre-entrenado adaptado a datos específicos")

# Extraer encoder para feature extraction
encoder_ft = Model(inputs=ae_pretrain.input,
                   outputs=ae_pretrain.layers[2].output)  # bottleneck
features = encoder_ft.predict(X_especifico, verbose=0)
print(f"Features extraídas shape: {features.shape}")  # (200, 10)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: Fine-tuning de AE para Transfer Learning.*

1. Pre-entrenar AE en datos genéricos de productos (100 features)
2. Fine-tuning en datos específicos de electrónicos (más pequeños)
3. Extraer encoder para feature extraction

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 18: Integrador — AE para Detección de Anomalías en Ventas

```python
from sklearn.metrics import precision_score, recall_score, f1_score

np.random.seed(42)
tf.random.set_seed(42)

# Dataset de ventas diarias por producto (60 días, 10 productos → 600 features)
# Simplificación: 30 features para demo
n_normal = 800
n_anomalo = 50
n_total = n_normal + n_anomalo

# Datos normales (transacciones regulares)
X_normal = np.random.randn(n_normal, 30)

# Anomalías (fraudes, devoluciones masivas, picos extraños)
X_anomalo = np.random.randn(n_anomalo, 30)
for i in range(n_anomalo):
    X_anomalo[i, np.random.choice(30, 5)] *= 4  # 5 features con valores extremos

# Entrenar AE solo con datos normales
ae_final = Sequential([
    Dense(20, activation='relu', input_shape=(30,)),
    Dense(10, activation='relu'),
    Dense(5, activation='relu'),  # bottleneck
    Dense(10, activation='relu'),
    Dense(20, activation='relu'),
    Dense(30, activation='linear')
])
ae_final.compile(optimizer=Adam(0.001), loss='mse')
ae_final.summary()

h_ae_final = ae_final.fit(X_normal, X_normal, epochs=50, batch_size=32,
                          validation_split=0.2, verbose=1)

# Evaluar en normales + anomalías
X_test = np.vstack([X_normal, X_anomalo])
y_true = np.array([0]*n_normal + [1]*n_anomalo)

X_test_recon = ae_final.predict(X_test, verbose=0)
recon_errors = np.mean((X_test - X_test_recon)**2, axis=1)

# Encontrar threshold óptimo
errors_norm_only = recon_errors[:n_normal]
threshold_opt = np.percentile(errors_norm_only, 97)
y_pred = (recon_errors > threshold_opt).astype(int)

print(f"\n=== Resultados Detección de Anomalías ===")
print(f"Threshold (P97): {threshold_opt:.4f}")
print(f"Precision: {precision_score(y_true, y_pred):.4f}")
print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
print(f"F1-Score:  {f1_score(y_true, y_pred):.4f}")

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(h_ae_final.history['loss'], label='Train', alpha=0.7)
plt.plot(h_ae_final.history['val_loss'], label='Validation', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Curva de entrenamiento AE')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(range(n_total), recon_errors, c=y_pred, cmap='coolwarm', alpha=0.6)
plt.axhline(threshold_opt, color='k', linestyle='--', label=f'Threshold (P97)')
plt.xlabel('Muestra')
plt.ylabel('Reconstruction Error')
plt.title('Anomalías detectadas (rojo=anómalo)')
plt.colorbar(label='Predicción')
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

*Ejemplo 18: Integrador — AE para Detección de Anomalías en Ventas.*

1. Dataset de ventas diarias por producto (60 días, 10 productos → 600 features)
2. Simplificación: 30 features para demo
3. Datos normales (transacciones regulares)
4. Anomalías (fraudes, devoluciones masivas, picos extraños)
5. Entrenar AE solo con datos normales
6. Evaluar en normales + anomalías
7. Encontrar threshold óptimo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejercicios

1. **Undercomplete AE para compras**: Crea un AE que comprima 12 features de compras (precio_unitario, cantidad, descuento, costo_envío, lead_time, proveedor_score, etc.) a 4 dimensiones. Entrena y evalúa reconstruction error. ¿Qué features se pierden más en la compresión?

2. **Denoising AE para limpiar datos de ventas**: Genera datos de ventas diarias con ruido gaussiano (σ=0.4) y entrena un Denoising AE para limpiarlos. Compara la serie original, ruidosa y reconstruida visualmente. ¿Qué nivel de ruido puede tolerar antes de fallar?

3. **Sparse AE para selección de features en inventario**: Entrena un Sparse AE con regularización L1 (λ=1e-3) sobre 50 features de inventario. Analiza qué activaciones latentes son cero. ¿Cuántas dimensiones latentes realmente se usan?

4. **Stacked AE profundo para clasificación de productos**: Construye un Stacked AE con 5 capas en encoder (50→40→30→20→10) y entrena en 5000 productos con 50 features. Luego usa el bottleneck (10D) como input para un clasificador de 6 categorías (Dense + softmax). Compara accuracy vs clasificar con las 50 features originales.

5. **VAE para generar productos sintéticos**: Implementa un VAE con latent_dim=4 para generar nuevos productos electrónicos con 15 features. Genera 20 productos sintéticos. ¿Son realistas? Calcula la media y varianza de las features generadas vs las originales.

6. **LSTM-AE para detección de anomalías en series de ventas**: Genera 500 series normales de 40 días con 2 features. Inyecta 20 series anómalas (con picos o caídas abruptas). Entrena un LSTM-AE solo con datos normales y detecta las anomalías. Reporta precisión y recall.

7. **ConvAE para compresión de patrones semanales**: Diseña un ConvAE que comprima 28 días de ventas (4 semanas, 1 feature) en un tensor latente de 7×8 usando Conv1D + MaxPooling. Reconstruye y visualiza 5 ejemplos originales vs reconstruidos.

8. **Integrador: AE para detección de fraude en devoluciones**: Crea un sistema completo: (a) genera datos normales de devoluciones (20 features, 2000 muestras) y 100 casos de fraude; (b) entrena un AE profundo (32→16→8→16→32) con los normales; (c) determina threshold como P96 del reconstruction error; (d) clasifica fraudes; (e) reporta matriz de confusión, precisión, recall, F1; (f) visualiza el espacio latente 2D de normales vs anómalos.

---

## Resumen

- **Autoencoder** comprime datos en un espacio latente y reconstruye. Útil para reducción de dimensionalidad, denoising, detección de anomalías y generación.
- **Undercomplete AE** fuerza compresión con bottleneck más pequeño que input.
- **Denoising AE** aprende a eliminar ruido, ideal para limpiar datos de ventas ruidosos.
- **Sparse AE** usa regularización L1 para activaciones latentes dispersas, seleccionando solo las features más relevantes.
- **Stacked AE** profundiza la representación con múltiples capas en encoder y decoder.
- **VAE** modela el espacio latente como distribución probabilística, permitiendo generar nuevos productos sintéticos mediante muestreo.
- **ConvAE** aplica convoluciones 1D para patrones temporales en series de ventas.
- **LSTM-AE** usa LSTMs para secuencias temporales, detectando anomalías en comportamiento de ventas diarias.
- **Detección de anomalías**: entrenar AE solo con datos normales; reconstruction error alto → anomalía. Threshold típico: percentil 95-99.
- **AE vs PCA**: AE captura relaciones no lineales, superando a PCA cuando los datos tienen estructura no lineal.
- **Transfer learning**: pre-entrenar AE en datos genéricos, fine-tuning en datos específicos de productos.
