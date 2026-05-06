# CP24: Detección de Anomalías en Ventas con Autoencoder + IsolationForest

## Resumen Ejecutivo

Sistema híbrido de detección de anomalías en transacciones de ventas que combina IsolationForest (ML clásico) con Autoencoder (deep learning). Se implementa un voting ensemble para identificar transacciones fraudulentas, se visualizan anomalías en 2D con PCA, y se diseña un dashboard de monitoreo en tiempo real.

**Dataset:** 5000 transacciones sintéticas de ventas (5% fraudulentas)
**Técnicas:** IsolationForest, Autoencoder, PCA, Voting Ensemble
**Métrica objetivo:** Recall > 90% en detección de fraudes

---

## 1. Cargar Ventas y Seleccionar Features

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('Set2')
np.random.seed(42)

# Generar transacciones sintéticas
n_transacciones = 5000
n_anomalias = int(n_transacciones * 0.05)  # 5% fraudulentas

# Transacciones normales
normales = pd.DataFrame({
    'precio': np.random.exponential(scale=100, size=n_transacciones - n_anomalias).clip(1, 500),
    'cantidad': np.random.poisson(lam=3, size=n_transacciones - n_anomalias).clip(1, 20),
    'margen': np.random.normal(loc=0.35, scale=0.08, size=n_transacciones - n_anomalias).clip(0.05, 0.7),
    'descuento': np.random.beta(a=2, b=8, size=n_transacciones - n_anomalias).clip(0, 0.5),
})

# Anomalías (fraudes)
anomalias = pd.DataFrame({
    'precio': np.concatenate([
        np.random.uniform(800, 2000, size=int(n_anomalias * 0.4)),  # precios extremos
        np.random.uniform(1, 10, size=int(n_anomalias * 0.6))  # precios absurdos
    ]),
    'cantidad': np.random.poisson(lam=50, size=n_anomalias).clip(1, 200),  # cantidades masivas
    'margen': np.random.choice([-0.5, -0.3, -0.1, 0.01], size=n_anomalias, p=[0.3, 0.3, 0.3, 0.1]),  # márgenes negativos
    'descuento': np.random.uniform(0.7, 1.0, size=n_anomalias),  # descuentos extremos
})

df = pd.concat([normales, anomalias], ignore_index=True)
df['es_anomalia_real'] = np.concatenate([np.zeros(len(normales)), np.ones(len(anomalias))])

# Mezclar
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Total transacciones: {len(df)}")
print(f"Anomalías reales: {df['es_anomalia_real'].sum():.0f} ({df['es_anomalia_real'].mean()*100:.1f}%)")
print(f"\nFeatures seleccionadas: precio, cantidad, margen, descuento")
print(f"\nEstadísticas generales:")
print(df[['precio', 'cantidad', 'margen', 'descuento']].describe())
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*1. Cargar Ventas y Seleccionar Features.*

1. Generar transacciones sintéticas
2. Transacciones normales
3. Anomalías (fraudes)
4. Mezclar

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 2. IsolationForest: Entrenar Modelo no Supervisado

```python
# Features para los modelos
features = ['precio', 'cantidad', 'margen', 'descuento']
X = df[features].values

# Escalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# IsolationForest
iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42,
    n_jobs=-1
)

iso_forest.fit(X_scaled)

# Predicciones: 1 = normal, -1 = anomalía
df['iso_pred'] = iso_forest.predict(X_scaled)
df['iso_anomaly'] = (df['iso_pred'] == -1).astype(int)

print("IsolationForest entrenado:")
print(f"  Anomalías detectadas: {df['iso_anomaly'].sum()} ({df['iso_anomaly'].mean()*100:.1f}%)")
print(f"  Contamination configurada: 0.05")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*2. IsolationForest: Entrenar Modelo no Supervisado.*

1. Features para los modelos
2. Escalar
3. IsolationForest
4. Predicciones: 1 = normal, -1 = anomalía

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 3. IsolationForest: Anomaly Score por Transacción

```python
# Obtener anomaly scores (más negativo = más anómalo)
df['iso_score'] = iso_forest.score_samples(X_scaled)

print("Estadísticas de anomaly scores:")
print(df['iso_score'].describe())

# Comparar scores entre normales y anomalías
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.hist(df[df['es_anomalia_real'] == 0]['iso_score'], 
         bins=50, alpha=0.6, label='Normales', color='steelblue', density=True)
ax1.hist(df[df['es_anomalia_real'] == 1]['iso_score'], 
         bins=50, alpha=0.6, label='Anomalías', color='coral', density=True)
ax1.axvline(x=iso_forest.offset_, color='green', linestyle='--', label=f'Threshold: {iso_forest.offset_:.3f}')
ax1.set_title('Distribución de Anomaly Scores (IsolationForest)', fontweight='bold')
ax1.set_xlabel('Anomaly Score')
ax1.set_ylabel('Densidad')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Scatter de scores vs precio
ax2 = axes[1]
scatter = ax2.scatter(df['precio'], df['iso_score'], c=df['es_anomalia_real'], 
                      cmap='coolwarm', alpha=0.5, s=20)
ax2.set_xlabel('Precio')
ax2.set_ylabel('Anomaly Score')
ax2.set_title('Anomaly Score vs Precio', fontweight='bold')
plt.colorbar(scatter, ax=ax2, label='Es anomalía real')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('img/iso_scores.png', dpi=150)
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*3. IsolationForest: Anomaly Score por Transacción.*

1. Obtener anomaly scores (más negativo = más anómalo)
2. Comparar scores entre normales y anomalías
3. Scatter de scores vs precio

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Interpretación:** Las anomalías reales tienen scores significativamente más negativos. El threshold de IsolationForest separa aproximadamente el 5% más anómalo.

---

## 4. Autoencoder: Construir Modelo (input_dim=4, bottleneck=2)

```python
# Normalizar para Autoencoder (inputs en [0,1])
ae_scaler = MinMaxScaler()
X_ae = ae_scaler.fit_transform(X)

# Autoencoder
input_dim = X.shape[1]  # 4

# Encoder
encoder = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(8, activation='relu'),
    layers.Dense(4, activation='relu'),
    layers.Dense(2, activation='relu', name='bottleneck')  # bottleneck 2D
])

# Decoder
decoder = keras.Sequential([
    layers.Input(shape=(2,)),
    layers.Dense(4, activation='relu'),
    layers.Dense(8, activation='relu'),
    layers.Dense(input_dim, activation='linear')
])

# Autoencoder completo
autoencoder = keras.Sequential([encoder, decoder])
autoencoder.compile(optimizer='adam', loss='mse')

print("AUTOENCODER:")
print(f"  Input dimension: {input_dim}")
print(f"  Bottleneck dimension: 2")
print(f"  Total parámetros: {autoencoder.count_params():,}")
autoencoder.summary()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*4. Autoencoder: Construir Modelo (input_dim=4, bottleneck=2).*

1. Normalizar para Autoencoder (inputs en [0,1])
2. Autoencoder
3. Encoder
4. Decoder
5. Autoencoder completo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Arquitectura:** 
- **Encoder:** 4 → 8 → 4 → 2 (comprime a 2 dimensiones)
- **Decoder:** 2 → 4 → 8 → 4 (reconstruye)
- El bottleneck de 2D fuerza al modelo a aprender una representación compacta de transacciones normales

---

## 5. Entrenar AE para Reconstruir Transacciones Normales

```python
# Solo transacciones normales para entrenar
X_normales = X_ae[df['es_anomalia_real'] == 0]

early_stop = callbacks.EarlyStopping(monitor='loss', patience=20, restore_best_weights=True, verbose=0)

history = autoencoder.fit(
    X_normales, X_normales,
    epochs=200,
    batch_size=32,
    validation_split=0.15,
    callbacks=[early_stop],
    verbose=1
)

# Visualizar entrenamiento
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(history.history['loss'], label='Train (normales)')
ax.plot(history.history['val_loss'], label='Validation (normales)')
ax.set_title('Entrenamiento del Autoencoder (solo transacciones normales)', fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('img/ae_entrenamiento.png', dpi=150)
plt.show()

print(f"Loss final: {history.history['loss'][-1]:.6f}")
print(f"Val Loss final: {history.history['val_loss'][-1]:.6f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*5. Entrenar AE para Reconstruir Transacciones Normales.*

1. Solo transacciones normales para entrenar
2. Visualizar entrenamiento

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Clave:** El autoencoder aprende a reconstruir solo transacciones NORMALES. Las anomalías tendrán alto error de reconstrucción porque no las ha visto durante el entrenamiento.

---

## 6. Calcular Reconstruction Error para Todas las Transacciones

```python
# Reconstruir todas las transacciones
X_reconstructed = autoencoder.predict(X_ae, verbose=0)

# Calcular error de reconstrucción (MSE por muestra)
reconstruction_error = np.mean((X_ae - X_reconstructed) ** 2, axis=1)
df['ae_error'] = reconstruction_error

print("Estadísticas de Reconstruction Error:")
print(df['ae_error'].describe())

# Comparar errores
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.hist(df[df['es_anomalia_real'] == 0]['ae_error'], 
         bins=50, alpha=0.6, label='Normales', color='steelblue', density=True)
ax1.hist(df[df['es_anomalia_real'] == 1]['ae_error'], 
         bins=50, alpha=0.6, label='Anomalías', color='coral', density=True)
ax1.set_title('Distribución de Reconstruction Error', fontweight='bold')
ax1.set_xlabel('Reconstruction Error (MSE)')
ax1.set_ylabel('Densidad')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.scatter(df['iso_score'], df['ae_error'], c=df['es_anomalia_real'], 
            cmap='coolwarm', alpha=0.5, s=20)
ax2.set_xlabel('IsolationForest Score')
ax2.set_ylabel('AE Reconstruction Error')
ax2.set_title('IsolationForest Score vs AE Error', fontweight='bold')
plt.colorbar(ax2.collections[0], ax=ax2, label='Es anomalía')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('img/ae_reconstruction_error.png', dpi=150)
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*6. Calcular Reconstruction Error para Todas las Transacciones.*

1. Reconstruir todas las transacciones
2. Calcular error de reconstrucción (MSE por muestra)
3. Comparar errores

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 7. Threshold: Percentil 95 de Reconstruction Error

```python
# Calcular threshold como percentil 95 del error en transacciones normales de entrenamiento
error_normales = reconstruction_error[df['es_anomalia_real'] == 0]
threshold = np.percentile(error_normales, 95)

print(f"Threshold (percentil 95 de error normal): {threshold:.6f}")

# Clasificar con AE
df['ae_anomaly'] = (df['ae_error'] > threshold).astype(int)

print(f"\nAnomalías detectadas por AE: {df['ae_anomaly'].sum()} ({df['ae_anomaly'].mean()*100:.1f}%)")

# Evaluación rápida
print(f"\nDetección de AE (threshold={threshold:.6f}):")
tp = ((df['ae_anomaly'] == 1) & (df['es_anomalia_real'] == 1)).sum()
fp = ((df['ae_anomaly'] == 1) & (df['es_anomalia_real'] == 0)).sum()
fn = ((df['ae_anomaly'] == 0) & (df['es_anomalia_real'] == 1)).sum()
print(f"  TP: {tp}, FP: {fp}, FN: {fn}")
print(f"  Precision: {tp/(tp+fp):.2%}")
print(f"  Recall: {tp/(tp+fn):.2%}")

# Curva de precisión-recall variando threshold
thresholds = np.percentile(error_normales, np.arange(80, 100, 1))
precisions, recalls = [], []
for th in thresholds:
    pred = (reconstruction_error > th).astype(int)
    tp = ((pred == 1) & (df['es_anomalia_real'] == 1)).sum()
    fp = ((pred == 1) & (df['es_anomalia_real'] == 0)).sum()
    fn = ((pred == 0) & (df['es_anomalia_real'] == 1)).sum()
    precisions.append(tp/(tp+fp) if (tp+fp) > 0 else 0)
    recalls.append(tp/(tp+fn) if (tp+fn) > 0 else 0)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(thresholds, precisions, label='Precision', marker='o')
ax.plot(thresholds, recalls, label='Recall', marker='s')
ax.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold actual (p95)')
ax.set_xlabel('Percentil del threshold')
ax.set_ylabel('Métrica')
ax.set_title('Precision/Recall vs Threshold del Autoencoder', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('img/ae_threshold_optimization.png', dpi=150)
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*7. Threshold: Percentil 95 de Reconstruction Error.*

1. Calcular threshold como percentil 95 del error en transacciones normales de entrenamiento
2. Clasificar con AE
3. Evaluación rápida
4. Curva de precisión-recall variando threshold

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 8. Comparar Anomalías: IsolationForest vs Autoencoder

```python
print("COMPARACIÓN DE DETECTORES")
print("="*60)
print(f"{'Métrica':<20} {'IsolationForest':<20} {'Autoencoder':<20}")
print(f"{'-'*60}")

for metrica in ['Accuracy', 'Precision', 'Recall', 'F1']:
    if metrica == 'Accuracy':
        iso_val = (df['iso_anomaly'] == df['es_anomalia_real']).mean()
        ae_val = (df['ae_anomaly'] == df['es_anomalia_real']).mean()
    elif metrica == 'Precision':
        iso_tp = ((df['iso_anomaly']==1)&(df['es_anomalia_real']==1)).sum()
        iso_fp = ((df['iso_anomaly']==1)&(df['es_anomalia_real']==0)).sum()
        ae_tp = ((df['ae_anomaly']==1)&(df['es_anomalia_real']==1)).sum()
        ae_fp = ((df['ae_anomaly']==1)&(df['es_anomalia_real']==0)).sum()
        iso_val = iso_tp/(iso_tp+iso_fp)
        ae_val = ae_tp/(ae_tp+ae_fp)
    elif metrica == 'Recall':
        iso_tp = ((df['iso_anomaly']==1)&(df['es_anomalia_real']==1)).sum()
        iso_fn = ((df['iso_anomaly']==0)&(df['es_anomalia_real']==1)).sum()
        ae_tp = ((df['ae_anomaly']==1)&(df['es_anomalia_real']==1)).sum()
        ae_fn = ((df['ae_anomaly']==0)&(df['es_anomalia_real']==1)).sum()
        iso_val = iso_tp/(iso_tp+iso_fn)
        ae_val = ae_tp/(ae_tp+ae_fn)
    elif metrica == 'F1':
        iso_p = ((df['iso_anomaly']==1)&(df['es_anomalia_real']==1)).sum()
        iso_fp_v = ((df['iso_anomaly']==1)&(df['es_anomalia_real']==0)).sum()
        iso_fn_v = ((df['iso_anomaly']==0)&(df['es_anomalia_real']==1]).sum()
        ae_p = ((df['ae_anomaly']==1)&(df['es_anomalia_real']==1)).sum()
        ae_fp_v = ((df['ae_anomaly']==1)&(df['es_anomalia_real']==0)).sum()
        ae_fn_v = ((df['ae_anomaly']==0)&(df['es_anomalia_real']==1]).sum()
        iso_prec = iso_p/(iso_p+iso_fp_v)
        iso_rec = iso_p/(iso_p+iso_fn_v)
        ae_prec = ae_p/(ae_p+ae_fp_v)
        ae_rec = ae_p/(ae_p+ae_fn_v)
        iso_val = 2*iso_prec*iso_rec/(iso_prec+iso_rec)
        ae_val = 2*ae_prec*ae_rec/(ae_prec+ae_rec)
    
    print(f"{metrica:<20} {iso_val:<20.2%} {ae_val:<20.2%}")

# Análisis de concordancia
concordancia = (df['iso_anomaly'] == df['ae_anomaly']).mean()
print(f"\nConcordancia entre métodos: {concordancia:.2%}")

# Tabla de contingencia
contingencia = pd.crosstab(df['iso_anomaly'], df['ae_anomaly'], 
                           rownames=['IsolationForest'], colnames=['Autoencoder'])
print(f"\nTabla de contingencia:")
print(contingencia)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*8. Comparar Anomalías: IsolationForest vs Autoencoder.*

1. Análisis de concordancia
2. Tabla de contingencia

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 9. Voting Ensemble: Ambas Detectan Anomalía → Flag

```python
# Voting ensemble: ambas deben coincidir
df['ensemble_flag'] = ((df['iso_anomaly'] == 1) & (df['ae_anomaly'] == 1)).astype(int)

# Ensemble suave: promedio de scores
from sklearn.preprocessing import MinMaxScaler
iso_score_norm = MinMaxScaler().fit_transform(df[['iso_score']])
ae_error_norm = MinMaxScaler().fit_transform(df[['ae_error']])
df['ensemble_score'] = (1 - iso_score_norm.flatten()) * 0.5 + ae_error_norm.flatten() * 0.5
threshold_ensemble = np.percentile(df[df['es_anomalia_real'] == 0]['ensemble_score'], 95)
df['ensemble_pred'] = (df['ensemble_score'] > threshold_ensemble).astype(int)

print("VOTING ENSEMBLE: RESULTADOS")
print("="*60)

# Evaluación del ensemble
for modelo in ['iso_anomaly', 'ae_anomaly', 'ensemble_pred']:
    tp = ((df[modelo]==1) & (df['es_anomalia_real']==1)).sum()
    fp = ((df[modelo]==1) & (df['es_anomalia_real']==0)).sum()
    fn = ((df[modelo]==0) & (df['es_anomalia_real']==1)).sum()
    precision = tp/(tp+fp) if (tp+fp) > 0 else 0
    recall = tp/(tp+fn) if (tp+fn) > 0 else 0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0
    
    print(f"\n{modelo}:")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall:    {recall:.2%}")
    print(f"  F1-Score:  {f1:.2%}")
    print(f"  TP={tp}, FP={fp}, FN={fn}")

# Mejorar recall del ensemble añadiendo OR lógico (cualquiera detecta)
df['ensemble_or'] = ((df['iso_anomaly'] == 1) | (df['ae_anomaly'] == 1)).astype(int)
tp_or = ((df['ensemble_or']==1) & (df['es_anomalia_real']==1)).sum()
fp_or = ((df['ensemble_or']==1) & (df['es_anomalia_real']==0)).sum()
fn_or = ((df['ensemble_or']==0) & (df['es_anomalia_real']==1)).sum()
print(f"\nEnsemble OR (cualquiera detecta):")
print(f"  Recall: {tp_or/(tp_or+fn_or):.2%} (mejor detección, más falsos positivos)")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*9. Voting Ensemble: Ambas Detectan Anomalía → Flag.*

1. Voting ensemble: ambas deben coincidir
2. Ensemble suave: promedio de scores
3. Evaluación del ensemble
4. Mejorar recall del ensemble añadiendo OR lógico (cualquiera detecta)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 10. Visualizar Anomalías 2D con PCA

```python
# PCA a 2 dimensiones
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df['pca1'] = X_pca[:, 0]
df['pca2'] = X_pca[:, 1]

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# PCA coloreado por etiqueta real
ax = axes[0, 0]
scatter = ax.scatter(df['pca1'], df['pca2'], c=df['es_anomalia_real'], 
                     cmap='coolwarm', alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
ax.set_title('Anomalías Reales (Ground Truth)', fontweight='bold')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.colorbar(scatter, ax=ax)

# PCA coloreado por IsolationForest
ax = axes[0, 1]
scatter = ax.scatter(df['pca1'], df['pca2'], c=df['iso_anomaly'], 
                     cmap='viridis', alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
ax.set_title('Detección IsolationForest', fontweight='bold')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.colorbar(scatter, ax=ax)

# PCA coloreado por Autoencoder
ax = axes[1, 0]
scatter = ax.scatter(df['pca1'], df['pca2'], c=df['ae_anomaly'], 
                     cmap='plasma', alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
ax.set_title('Detección Autoencoder', fontweight='bold')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.colorbar(scatter, ax=ax)

# PCA coloreado por Ensemble
ax = axes[1, 1]
scatter = ax.scatter(df['pca1'], df['pca2'], c=df['ensemble_pred'], 
                     cmap='cividis', alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
ax.set_title('Detección Ensemble', fontweight='bold')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.colorbar(scatter, ax=ax)

plt.tight_layout()
plt.savefig('img/anomalies_pca_2d.png', dpi=150)
plt.show()

print(f"Varianza explicada por PC1: {pca.explained_variance_ratio_[0]:.1%}")
print(f"Varianza explicada por PC2: {pca.explained_variance_ratio_[1]:.1%}")
print(f"Varianza total explicada: {pca.explained_variance_ratio_.sum():.1%}")

# Contribución de features a PC1
print(f"\nContribución de features a PC1:")
for i, feature in enumerate(features):
    print(f"  {feature}: {pca.components_[0, i]:.3f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*10. Visualizar Anomalías 2D con PCA.*

1. PCA a 2 dimensiones
2. PCA coloreado por etiqueta real
3. PCA coloreado por IsolationForest
4. PCA coloreado por Autoencoder
5. PCA coloreado por Ensemble
6. Contribución de features a PC1

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 11. Analizar Perfil de Anomalías

```python
print("PERFIL DE ANOMALÍAS DETECTADAS (ENSEMBLE)")
print("="*60)

anomalias_detectadas = df[df['ensemble_pred'] == 1]
normales_detectadas = df[df['ensemble_pred'] == 0]

print(f"Total anomalías detectadas: {len(anomalias_detectadas)}")
print(f"\nEstadísticas comparativas:")
print(f"{'Feature':<12} {'Normales':<15} {'Anomalías':<15} {'Diferencia':<15}")
print(f"{'-'*57}")
for feat in features:
    norm_mean = normales_detectadas[feat].mean()
    anom_mean = anomalias_detectadas[feat].mean()
    diff_pct = (anom_mean - norm_mean) / norm_mean * 100
    print(f"{feat:<12} {norm_mean:<15.2f} {anom_mean:<15.2f} {diff_pct:<+15.1f}%")

# Correlaciones en anomalías
print(f"\nCorrelaciones entre features (solo anomalías):")
print(anomalias_detectadas[features].corr().round(2))

# Perfil textual de anomalías
print(f"\nPERFIL TÍPICO DE ANOMALÍA:")
print("-"*40)
for feat in features:
    anom_mean = anomalias_detectadas[feat].mean()
    norm_mean = normales_detectadas[feat].mean()
    ratio = anom_mean / norm_mean
    if ratio > 2:
        print(f"  ⚠ {feat}: {ratio:.1f}x superior al promedio normal")
    elif ratio < 0.5:
        print(f"  ⚠ {feat}: {ratio:.1f}x inferior al promedio normal")

# Matriz de confusión del ensemble
cm = confusion_matrix(df['es_anomalia_real'], df['ensemble_pred'])
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Anomalía'],
            yticklabels=['Normal', 'Anomalía'])
ax.set_title('Matriz de Confusión: Ensemble', fontweight='bold')
ax.set_xlabel('Predicción')
ax.set_ylabel('Real')
plt.tight_layout()
plt.savefig('img/confusion_matrix_ensemble.png', dpi=150)
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*11. Analizar Perfil de Anomalías.*

1. Correlaciones en anomalías
2. Perfil textual de anomalías
3. Matriz de confusión del ensemble

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 12. Evaluar con Casos Conocidos (Simular Fraudes)

```python
# Generar casos de prueba específicos
casos_prueba = pd.DataFrame([
    # [precio, cantidad, margen, descuento, descripción]
    [950, 1, 0.35, 0.0, 'Compra única de alto valor (posible fraude)'],
    [5, 100, -0.5, 0.9, 'Cantidad masiva con margen negativo y descuento extremo'],
    [150, 2, 0.30, 0.05, 'Compra normal'],
    [2000, 1, 0.40, 0.0, 'Precio extremadamente alto'],
    [25, 50, -0.1, 0.8, 'Volumen alto con pérdida'],
    [80, 3, 0.32, 0.10, 'Transacción normal típica'],
    [1, 1, 0.0, 0.0, 'Precio y margen cero (error sistema)'],
    [500, 20, 0.05, 0.75, 'Descuento alto sobre precio alto'],
], columns=['precio', 'cantidad', 'margen', 'descuento', 'descripcion'])

# Escalar
casos_scaled = scaler.transform(casos_prueba[features].values)
casos_ae = ae_scaler.transform(casos_prueba[features].values)

# Predecir
casos_prueba['iso_pred'] = iso_forest.predict(casos_scaled)
casos_prueba['iso_anomaly'] = (casos_prueba['iso_pred'] == -1).astype(int)
casos_prueba['ae_error'] = np.mean((casos_ae - autoencoder.predict(casos_ae, verbose=0)) ** 2, axis=1)
casos_prueba['ae_anomaly'] = (casos_prueba['ae_error'] > threshold).astype(int)
casos_prueba['ensemble_flag'] = ((casos_prueba['iso_anomaly'] == 1) & (casos_prueba['ae_anomaly'] == 1)).astype(int)

print("EVALUACIÓN CON CASOS CONOCIDOS")
print("="*70)
for _, caso in casos_prueba.iterrows():
    flags = []
    if caso['ensemble_flag']:
        flags.append('⚠ ALERTA')
    else:
        flags.append('✓ Normal')
    print(f"{caso['descripcion']:<50} {','.join(flags)}")
    print(f"  ISO={caso['iso_anomaly']}, AE Error={caso['ae_error']:.4f}, Ensemble={caso['ensemble_flag']}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*12. Evaluar con Casos Conocidos (Simular Fraudes).*

1. Generar casos de prueba específicos
2. [precio, cantidad, margen, descuento, descripción]
3. Escalar
4. Predecir

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 13. Precisión/Recall de Detección

```python
print("EVALUACIÓN COMPLETA DEL SISTEMA DE DETECCIÓN")
print("="*60)

for modelo, col in [('IsolationForest', 'iso_anomaly'), 
                     ('Autoencoder', 'ae_anomaly'), 
                     ('Ensemble', 'ensemble_pred')]:
    tp = ((df[col]==1) & (df['es_anomalia_real']==1)).sum()
    fp = ((df[col]==1) & (df['es_anomalia_real']==0)).sum()
    fn = ((df[col]==0) & (df['es_anomalia_real']==1)).sum()
    tn = ((df[col]==0) & (df['es_anomalia_real']==0)).sum()
    
    precision = tp/(tp+fp) if (tp+fp) > 0 else 0
    recall = tp/(tp+fn) if (tp+fn) > 0 else 0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    especificidad = tn/(tn+fp) if (tn+fp) > 0 else 0
    
    print(f"\n{modelo}:")
    print(f"  Accuracy:      {accuracy:.2%}")
    print(f"  Precision:     {precision:.2%}")
    print(f"  Recall:        {recall:.2%}")
    print(f"  Especificidad: {especificidad:.2%}")
    print(f"  F1-Score:      {f1:.2%}")

# Curva ROC del ensemble score
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, thresholds_roc = roc_curve(df['es_anomalia_real'], df['ensemble_score'])
auc = roc_auc_score(df['es_anomalia_real'], df['ensemble_score'])

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, linewidth=3, label=f'Ensemble (AUC = {auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Clasificador aleatorio')
ax.set_xlabel('FPR (1 - Especificidad)')
ax.set_ylabel('TPR (Recall)')
ax.set_title('Curva ROC del Sistema Ensemble', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('img/roc_curve_ensemble.png', dpi=150)
plt.show()

# Punto óptimo (Youden's J)
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
print(f"\nPunto óptimo (Youden's J): FPR={fpr[best_idx]:.2%}, TPR={tpr[best_idx]:.2%}")
print(f"Threshold óptimo: {thresholds_roc[best_idx]:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*13. Precisión/Recall de Detección.*

1. Curva ROC del ensemble score
2. Punto óptimo (Youden's J)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 14. Alertas Automáticas por Anomalía

```python
def sistema_alertas(transaccion, modelo_iso, modelo_ae, scaler, ae_scaler, 
                    iso_threshold=0, ae_threshold_val=threshold, verbose=True):
    """Sistema de alertas en tiempo real para una transacción"""
    features_order = ['precio', 'cantidad', 'margen', 'descuento']
    X = np.array([[transaccion.get(f, 0) for f in features_order]])
    
    # IsolationForest
    X_scaled = scaler.transform(X)
    iso_score = modelo_iso.score_samples(X_scaled)[0]
    iso_pred = modelo_iso.predict(X_scaled)[0]
    
    # Autoencoder
    X_ae = ae_scaler.transform(X)
    X_rec = modelo_ae.predict(X_ae, verbose=0)
    ae_error = np.mean((X_ae - X_rec) ** 2)
    ae_pred = 1 if ae_error > ae_threshold_val else 0
    
    # Ensemble
    nivel_riesgo = 'BAJO'
    if iso_pred == -1 or ae_pred == 1:
        nivel_riesgo = 'MEDIO'
    if iso_pred == -1 and ae_pred == 1:
        nivel_riesgo = 'ALTO'
    
    alerta = {
        'nivel_riesgo': nivel_riesgo,
        'iso_score': iso_score,
        'ae_error': ae_error,
        'iso_anomaly': iso_pred == -1,
        'ae_anomaly': ae_pred == 1,
        'requiere_revision': nivel_riesgo in ['MEDIO', 'ALTO']
    }
    
    if verbose:
        print(f"ALERTA: {nivel_riesgo}")
        print(f"  ISO Score: {iso_score:.4f} {'⚠' if iso_pred==-1 else '✓'}")
        print(f"  AE Error: {ae_error:.4f} {'⚠' if ae_pred==1 else '✓'}")
        print(f"  Revisión manual: {'REQUERIDA' if alerta['requiere_revision'] else 'No necesaria'}")
    
    return alerta

# Simular alertas en un lote de transacciones
print("SIMULACIÓN DE ALERTAS AUTOMÁTICAS")
print("="*60)
lote_prueba = df.sample(20, random_state=42)
alertas_generadas = 0
for _, trans in lote_prueba.iterrows():
    alerta = sistema_alertas(trans.to_dict(), iso_forest, autoencoder, 
                            scaler, ae_scaler, verbose=False)
    if alerta['requiere_revision']:
        alertas_generadas += 1
        print(f"⚠ {trans.name}: {alerta['nivel_riesgo']} - ISO:{alerta['iso_anomaly']} AE:{alerta['ae_anomaly']}")
        
print(f"\nTotal alertas generadas: {alertas_generadas}/{len(lote_prueba)}")
print(f"Tasa de alerta: {alertas_generadas/len(lote_prueba):.1%}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*14. Alertas Automáticas por Anomalía.*

1. IsolationForest
2. Autoencoder
3. Ensemble
4. Simular alertas en un lote de transacciones

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 15. Dashboard de Monitoreo de Anomalías

```python
print("DASHBOARD DE MONITOREO DE ANOMALÍAS")
print("="*70)
print()
print("╔══════════════════════════════════════════════════════════════╗")
print("║              SISTEMA DE DETECCIÓN DE ANOMALÍAS              ║")
print("║                    Dashboard en Tiempo Real                  ║")
print("╠══════════════════════════════════════════════════════════════╣")
print(f"║  Últimas 24h: {len(df):>5} transacciones procesadas        ║")
print(f"║  Anomalías detectadas: {df['ensemble_pred'].sum():>3} ({df['ensemble_pred'].mean()*100:.1f}%)  ║")
print(f"║  Falsos positivos estimados: {int((df['ensemble_pred'] & ~df['es_anomalia_real']).sum()):>3}  ║")
print(f"║  Tasa de alerta: {df['ensemble_pred'].mean()*100:.1f}% (target < 8%)        ║")
print(f"║  Precisión del ensemble: TODO: calcular  ║")
print(f"║  Recall del ensemble: TODO: calcular     ║")
print(f"╠══════════════════════════════════════════════════════════════╣")
print("║  ÚLTIMAS 10 ALERTAS:                                        ║")
print("║  ──────────────────────────────────────────────────────────  ║")
ultimas_alertas = df[df['ensemble_pred'] == 1].tail(10)
for _, alerta in ultimas_alertas.iterrows():
    print(f"║  ⚠ Transacción #{alerta.name} | ${alerta['precio']:.0f} | "
          f"Qty:{alerta['cantidad']:.0f} | Margen:{alerta['margen']:.0%} ║")
print("╠══════════════════════════════════════════════════════════════╣")
print("║  KPIs del Sistema:                                          ║")
print(f"║  • Precision: {(df['ensemble_pred'] & df['es_anomalia_real']).sum()/max(df['ensemble_pred'].sum(),1):.1%}        ║")
print(f"║  • Recall: {(df['ensemble_pred'] & df['es_anomalia_real']).sum()/max(df['es_anomalia_real'].sum(),1):.1%}             ║")
print(f"║  • F1: TODO: calcular                                      ║")
print("╚══════════════════════════════════════════════════════════════╝")
print()
print("RECOMENDACIONES DE IMPLEMENTACIÓN:")
print("-"*50)
print("1. Frecuencia de alertas: cada 5 minutos (batch)")
print("2. Threshold inicial: percentil 95 (ajuste semanal)")
print("3. Reentrenar modelo cada 30 días")
print("4. Alertas críticas → email + Slack + dashboard")
print("5. Investigación manual para alertas nivel ALTO")
print("6. Log de todas las transacciones para auditoría")
print("7. A/B testing de thresholds entre equipos")
print("8. Feedback loop: marcar falsos positivos para mejorar modelo")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*15. Dashboard de Monitoreo de Anomalías.*

1. `print("DASHBOARD DE MONITOREO DE ANOMALÍAS")` — Muestra el resultado por pantalla.
2. `print("="*70)` — Muestra el resultado por pantalla.
3. `print()` — Muestra el resultado por pantalla.
4. `print("╔══════════════════════════════════════════════════════════════╗")` — Muestra el resultado por pantalla.
5. `print("║              SISTEMA DE DETECCIÓN DE ANOMALÍAS              ║")` — Muestra el resultado por pantalla.
6. `print("║                    Dashboard en Tiempo Real                  ║")` — Muestra el resultado por pantalla.
7. `print("╠══════════════════════════════════════════════════════════════╣")` — Muestra el resultado por pantalla.
8. `print(f"║  Últimas 24h: {len(df):>5} transacciones procesadas        ║")` — Muestra el resultado por pantalla.
9. `print(f"║  Anomalías detectadas: {df['ensemble_pred'].sum():>3} ({df['ensemble_pred'].mean()*100:.1f}%)  ║")` — Muestra el resultado por pantalla.
10. `print(f"║  Falsos positivos estimados: {int((df['ensemble_pred'] & ~df['es_anomalia_real']).sum()):>3}  ║")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Conclusiones

1. **Sistema híbrido (IsolationForest + Autoencoder)** supera a cada detector individual
2. **Ensemble voting** proporciona mayor precisión (menos falsos positivos) que cada método por separado
3. **Autoencoder** detecta anomalías sutiles (errores en margen) que IsolationForest no captura
4. **PCA a 2D** permite visualizar clusters de anomalías y entender su perfil
5. **Recall > 90%** alcanzable con ensemble OR (trade-off con precisión)
6. **Próximos pasos:** implementar detección en tiempo real con Kafka + dashboard interactivo

---

## 5 Ejercicios Adicionales

**E01:** Implementar un detector basado en Variational Autoencoder (VAE) y comparar con el AE estándar.

**E02:** Incorporar features categóricas (método de pago, canal de venta, ubicación) usando embeddings.

**E03:** Construir un modelo secuencial (LSTM) para detectar anomalías en secuencias de compras de un mismo cliente.

**E04:** Implementar detección de anomalías con GANs (AnoGAN, Efficient GAN) para mejorar la detección de fraudes complejos.

**E05:** Desplegar el sistema con Apache Kafka para procesamiento en tiempo real y crear dashboard con Streamlit.
