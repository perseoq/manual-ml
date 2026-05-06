# I22: PCA y Reducción de Dimensionalidad con scikit-learn

## Aplicación a Ventas, Compras e Inventarios

La reducción de dimensionalidad transforma datos de **muchas variables** a un espacio de **menos dimensiones** conservando la mayor cantidad de información posible. En el contexto comercial, permite **visualizar datos multivariantes, comprimir representaciones, eliminar ruido y acelerar modelos de ML**.

### Fundamentos Teóricos

**PCA (Principal Component Analysis):** encuentra direcciones (componentes principales) que maximizan la varianza de los datos proyectados.
- El primer componente captura la máxima varianza
- Cada componente siguiente captura la varianza restante, ortogonal a los anteriores
- Los datos se proyectan al subespacio de los primeros `n_components`

**Parámetros clave de PCA:**
- `n_components`: número de dimensiones a conservar (entero o float 0-1 para % varianza)
- `whiten=True`: escala los componentes a varianza unitaria
- `svd_solver`: `'auto'` (elige automáticamente), `'full'` (exacto), `'arpack'` (auto-valores), `'randomized'` (aprox. rápido)
- `random_state`: semilla para solvers aleatorios

**Atributos importantes:**
- `explained_variance_ratio_`: proporción de varianza explicada por cada componente
- `singular_values_`: valores singulares de la descomposición
- `components_`: vectores de carga (pesos de cada variable original)
- `mean_`: media de cada feature (usada para centrar)

**Otras técnicas:**
- **IncrementalPCA:** PCA por lotes para datasets que no caben en memoria
- **KernelPCA:** PCA no lineal usando el truco del kernel (rbf, poly, sigmoid)
- **SparsePCA:** PCA con penalización L1 para componentes con ceros (más interpretables)
- **TruncatedSVD:** similar a PCA pero funciona con matrices dispersas (no centra los datos)
- **FactorAnalysis:** modelo generativo con factores latentes y ruido específico
- **NMF (Non-Negative Matrix Factorization):** descomposición con restricción de no negatividad

**Aplicaciones en ventas:** reducción de features de productos, visualización de segmentos, compresión de datos transaccionales, preprocesamiento para clustering.

---

## Ejemplos Prácticos con sklearn

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import (PCA, IncrementalPCA, KernelPCA,
                                    SparsePCA, TruncatedSVD, FactorAnalysis, NMF)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# DATASET SIMULADO DE VENTAS (10 variables)
# ============================================================
np.random.seed(42)
n = 500

df = pd.DataFrame({
    'precio': np.random.uniform(10, 200, n),
    'margen': np.random.uniform(5, 60, n),
    'descuento': np.random.uniform(0, 0.5, n),
    'dias_stock': np.random.randint(1, 120, n),
    'ventas_semana': np.random.poisson(50, n),
    'devoluciones': np.random.poisson(3, n),
    'costo_logistico': np.random.uniform(2, 20, n),
    'rating': np.random.uniform(1, 5, n),
    'peso_kg': np.random.uniform(0.1, 25, n),
    'volumen_cm3': np.random.uniform(100, 5000, n),
})

# Añadir correlaciones realistas
df['margen'] = df['margen'] + 0.2 * df['precio'] + np.random.normal(0, 3, n)
df['ventas_semana'] = (df['ventas_semana'] - 0.3 * df['precio']
                        + 2 * df['descuento'] * 100 + np.random.normal(0, 10, n)).clip(0)
df['costo_logistico'] = 2 + 0.3 * df['peso_kg'] + 0.01 * df['volumen_cm3'] + np.random.normal(0, 1, n)
df['dias_stock'] = (df['dias_stock'] - 0.1 * df['ventas_semana']
                     + np.random.normal(0, 5, n)).clip(1)

# Features para PCA (todas numéricas)
features_pca = ['precio', 'margen', 'descuento', 'dias_stock', 'ventas_semana',
                'devoluciones', 'costo_logistico', 'rating', 'peso_kg', 'volumen_cm3']
X = df[features_pca].values

# Estandarizar (PCA es sensible a escalas)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print('Dataset de ventas (10 variables) creado.')
print(f'Shape: {X.shape}')
print(f'Variables: {features_pca}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplos Prácticos con sklearn.*

1. ============================================================
2. DATASET SIMULADO DE VENTAS (10 variables)
3. ============================================================
4. Añadir correlaciones realistas
5. Features para PCA (todas numéricas)
6. Estandarizar (PCA es sensible a escalas)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 1: PCA con 2 componentes — reducir 10 variables de ventas a 2D

```python
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
print(f'Shape original: {X_scaled.shape}')
print(f'Shape reducido: {X_pca.shape}')
print(f'\nVarianza explicada por componente:')
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f'  PC{i+1}: {var:.4f} ({var*100:.2f}%)')
print(f'  Total: {sum(pca.explained_variance_ratio_)*100:.2f}%')
print(f'\nValores singulares: {pca.singular_values_.round(2)}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: PCA con 2 componentes — reducir 10 variables de ventas a 2D.*

1. `print(f'Shape original: {X_scaled.shape}')` — Muestra el resultado por pantalla.
2. `print(f'Shape reducido: {X_pca.shape}')` — Muestra el resultado por pantalla.
3. `print(f'\nVarianza explicada por componente:')` — Muestra el resultado por pantalla.
4. `print(f'  PC{i+1}: {var:.4f} ({var*100:.2f}%)')` — Muestra el resultado por pantalla.
5. `print(f'  Total: {sum(pca.explained_variance_ratio_)*100:.2f}%')` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 2: PCA con 2 componentes — scatterplot de datos en espacio PCA

```python
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, edgecolors='k', c='steelblue')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% varianza)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% varianza)')
plt.title('Proyección PCA de 10 variables de ventas a 2D')
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
plt.grid(alpha=0.3)
plt.show()

print(f'Varianza total explicada en 2D: {sum(pca.explained_variance_ratio_)*100:.1f}%')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: PCA con 2 componentes — scatterplot de datos en espacio PCA.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 3: explained_variance_ratio_ — varianza explicada por componente

```python
pca_full = PCA(n_components=10, random_state=42)
pca_full.fit(X_scaled)

var_ratio = pca_full.explained_variance_ratio_
var_acum = np.cumsum(var_ratio)

df_var = pd.DataFrame({
    'Componente': range(1, 11),
    'Varianza_individual': var_ratio,
    'Varianza_acumulada': var_acum,
})
print('Varianza explicada por cada componente:')
print(df_var.to_string(index=False))
print(f'\nCon 5 componentes se explica el {var_acum[4]*100:.1f}% de la varianza')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: explained_variance_ratio_ — varianza explicada por componente.*

1. `pca_full.fit(X_scaled)` — Entrena el modelo con los datos de entrenamiento.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 4: Scree plot — componentes vs varianza explicada

```python
pca_full = PCA(n_components=10, random_state=42)
pca_full.fit(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scree plot individual
axes[0].bar(range(1, 11), pca_full.explained_variance_ratio_, color='steelblue', alpha=0.7)
axes[0].plot(range(1, 11), pca_full.explained_variance_ratio_, 'ro-', markersize=6)
axes[0].set_xlabel('Componente principal')
axes[0].set_ylabel('Varianza explicada (proporción)')
axes[0].set_title('Scree Plot: varianza por componente')
axes[0].grid(alpha=0.3)

# Scree plot acumulado
axes[1].bar(range(1, 11), np.cumsum(pca_full.explained_variance_ratio_),
            color='coral', alpha=0.7)
axes[1].plot(range(1, 11), np.cumsum(pca_full.explained_variance_ratio_),
             'bo-', markersize=6)
axes[1].axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='80%')
axes[1].axhline(y=0.9, color='g', linestyle='--', alpha=0.5, label='90%')
axes[1].axhline(y=0.95, color='orange', linestyle='--', alpha=0.5, label='95%')
axes[1].set_xlabel('Número de componentes')
axes[1].set_ylabel('Varianza acumulada')
axes[1].set_title('Varianza acumulada')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 4: Scree plot — componentes vs varianza explicada.*

1. Scree plot individual
2. Scree plot acumulado

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 5: Acumulado — número de componentes para 80%, 90%, 95%

```python
pca_full = PCA(n_components=10, random_state=42)
pca_full.fit(X_scaled)

var_acum = np.cumsum(pca_full.explained_variance_ratio_)

objetivos = [0.80, 0.90, 0.95]
print('Componentes necesarios para alcanzar cada umbral de varianza:')
for obj in objetivos:
    n = np.argmax(var_acum >= obj) + 1
    print(f'  {obj*100:.0f}% varianza → {n} componente(s) (varianza real: {var_acum[n-1]*100:.2f}%)')

# PCA con 95% de varianza
pca_95 = PCA(n_components=0.95, random_state=42)
X_pca_95 = pca_95.fit_transform(X_scaled)
print(f'\nPCA con n_components=0.95: {X_pca_95.shape[1]} componentes')
print(f'Varianza explicada: {sum(pca_95.explained_variance_ratio_)*100:.2f}%')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: Acumulado — número de componentes para 80%, 90%, 95%.*

1. PCA con 95% de varianza

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 6: Componentes — cargas (pesos de cada variable original)

```python
pca = PCA(n_components=4, random_state=42)
pca.fit(X_scaled)

cargas = pd.DataFrame(
    pca.components_,
    columns=features_pca,
    index=[f'PC{i+1}' for i in range(4)]
)

print('Cargas de los componentes (pesos de cada variable):')
print(cargas.round(3))

# Interpretar
print('\n--- Interpretación ---')
for i in range(4):
    idx = np.argsort(np.abs(pca.components_[i]))[::-1]
    top3 = [features_pca[j] for j in idx[:3]]
    print(f'PC{i+1} (var={pca.explained_variance_ratio_[i]*100:.1f}%): '
          f'dominado por {", ".join(top3)}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: Componentes — cargas (pesos de cada variable original).*

1. Interpretar

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 7: Biplot — componentes + variables originales

```python
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, c='lightgray', edgecolors='k')

# Flechas de las variables originales
escala = 3.5
for i, var in enumerate(features_pca):
    plt.arrow(0, 0, pca.components_[0, i] * escala, pca.components_[1, i] * escala,
              head_width=0.15, head_length=0.15, fc='red', ec='red', alpha=0.8)
    plt.text(pca.components_[0, i] * escala * 1.1,
             pca.components_[1, i] * escala * 1.1,
             var, color='darkred', fontsize=10, fontweight='bold')

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('Biplot: componentes principales + proyección de variables')
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
plt.grid(alpha=0.3)
plt.show()

print('Interpretación:')
print('- Variables con flechas largas están bien representadas')
print('- Ángulos pequeños entre flechas indican correlación positiva')
print('- Ángulos de 90° indican independencia')
print('- Ángulos > 90° indican correlación negativa')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: Biplot — componentes + variables originales.*

1. Flechas de las variables originales

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 8: PCA whitening — datos con covarianza identidad

```python
pca_white = PCA(n_components=5, whiten=True, random_state=42)
X_white = pca_white.fit_transform(X_scaled)

cov_white = np.cov(X_white.T)

plt.figure(figsize=(6, 5))
plt.imshow(cov_white, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label='Covarianza')
plt.title('Matriz de covarianza tras PCA whitening\n(Esperado: identidad)')
plt.xlabel('Componente')
plt.ylabel('Componente')
plt.show()

# Verificar
print('Covarianza tras whitening (diagonal = 1, fuera = 0):')
print(np.round(cov_white[:5, :5], 3))
print(f'\nMedia de covarianza fuera de diagonal: {np.mean(np.abs(cov_white - np.eye(5))):.4f}')
print('Whitening transforma los datos para que tengan covarianza identidad.')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: PCA whitening — datos con covarianza identidad.*

1. Verificar

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 9: PCA inverse_transform — reconstruir datos desde componentes

```python
pca = PCA(n_components=3, random_state=42)
X_pca = pca.fit_transform(X_scaled)
X_reconstruido = pca.inverse_transform(X_pca)

# Comparar precios originales vs reconstruidos (en escala original)
precio_original = X_scaled[:, 0]
precio_reconst = X_reconstruido[:, 0]

error_reconst = np.mean((precio_original - precio_reconst)**2)
print(f'Error de reconstrucción (MSE) para precio: {error_reconst:.6f}')
print(f'(con 3 componentes de 10)')

# Probar con más componentes
for n in [5, 7, 9]:
    pca_n = PCA(n_components=n, random_state=42)
    X_rec = pca_n.inverse_transform(pca_n.fit_transform(X_scaled))
    error = np.mean((X_scaled - X_rec)**2)
    print(f'  n_components={n}: MSE reconstrucción total = {error:.6f}')

# Visualizar reconstrucción
idx = 0
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.bar(features_pca, X_scaled[idx], color='steelblue', alpha=0.7)
plt.title('Original (escalado)')
plt.xticks(rotation=45, ha='right')

plt.subplot(1, 3, 2)
plt.bar(features_pca, X_reconstruido[idx], color='coral', alpha=0.7)
plt.title('Reconstruido (3 PCs)')
plt.xticks(rotation=45, ha='right')

plt.subplot(1, 3, 3)
plt.bar(features_pca, X_scaled[idx] - X_reconstruido[idx], color='gray', alpha=0.7)
plt.title('Error de reconstrucción')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: PCA inverse_transform — reconstruir datos desde componentes.*

1. Comparar precios originales vs reconstruidos (en escala original)
2. Probar con más componentes
3. Visualizar reconstrucción

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 10: IncrementalPCA — para datasets grandes (procesar en batches)

```python
# Simular dataset grande
n_grande = 10000
X_grande = np.random.randn(n_grande, 10)

# IncrementalPCA por lotes
batch_size = 500
ipca = IncrementalPCA(n_components=3)

for i in range(0, n_grande, batch_size):
    end = min(i + batch_size, n_grande)
    ipca.partial_fit(X_grande[i:end])

X_ipca = ipca.transform(X_grande)

# PCA normal (para comparar)
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_grande)

# Comparar
print('Comparación PCA vs IncrementalPCA:')
print(f'  PCA varianza explicada: {pca.explained_variance_ratio_.round(4)}')
print(f'  IncrementalPCA varianza: {ipca.explained_variance_ratio_.round(4)}')
print(f'  Correlación PC1: {np.corrcoef(X_pca[:, 0], X_ipca[:, 0])[0,1]:.4f}')
print(f'  Batch size usado: {batch_size}')
print('\nIncrementalPCA permite procesar datasets más grandes que la memoria RAM.')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: IncrementalPCA — para datasets grandes (procesar en batches).*

1. Simular dataset grande
2. IncrementalPCA por lotes
3. PCA normal (para comparar)
4. Comparar

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 11: KernelPCA rbf — reducción no lineal

```python
from sklearn.datasets import make_circles, make_moons

# Dataset no lineal: círculos concéntricos
X_circles, y_circles = make_circles(n_samples=300, noise=0.1, factor=0.3, random_state=42)

# PCA lineal
pca_lin = PCA(n_components=2)
X_lin = pca_lin.fit_transform(X_circles)

# PCA con kernel RBF
kpca_rbf = KernelPCA(n_components=2, kernel='rbf', gamma=15, random_state=42)
X_rbf = kpca_rbf.fit_transform(X_circles)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles, cmap='viridis', alpha=0.7)
axes[0].set_title('Datos originales (círculos)')

axes[1].scatter(X_lin[:, 0], X_lin[:, 1], c=y_circles, cmap='viridis', alpha=0.7)
axes[1].set_title('PCA lineal (no separa)')

axes[2].scatter(X_rbf[:, 0], X_rbf[:, 1], c=y_circles, cmap='viridis', alpha=0.7)
axes[2].set_title('KernelPCA RBF (separa)')

plt.tight_layout()
plt.show()

print('PCA lineal no separa círculos concéntricos.')
print('KernelPCA con RBF sí los separa gracias al truco del kernel.')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: KernelPCA rbf — reducción no lineal.*

1. Dataset no lineal: círculos concéntricos
2. PCA lineal
3. PCA con kernel RBF

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 12: KernelPCA poly — comparar con lineal

```python
# Dataset en forma de luna
X_moons, y_moons = make_moons(n_samples=300, noise=0.08, random_state=42)

kpca_linear = KernelPCA(n_components=2, kernel='linear', random_state=42)
kpca_poly = KernelPCA(n_components=2, kernel='poly', degree=3, gamma=0.5, random_state=42)
kpca_rbf = KernelPCA(n_components=2, kernel='rbf', gamma=5, random_state=42)
kpca_sig = KernelPCA(n_components=2, kernel='sigmoid', gamma=0.5, random_state=42)

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
kernels = [('linear', kpca_linear), ('poly (degree=3)', kpca_poly),
           ('rbf', kpca_rbf), ('sigmoid', kpca_sig)]
for ax, (name, kpca) in zip(axes.flat, kernels):
    X_k = kpca.fit_transform(X_moons)
    ax.scatter(X_k[:, 0], X_k[:, 1], c=y_moons, cmap='viridis', alpha=0.7)
    ax.set_title(f'KernelPCA {name}')
    ax.set_xlabel('Componente 1')
    ax.set_ylabel('Componente 2')

plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 12: KernelPCA poly — comparar con lineal.*

1. Dataset en forma de luna

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 13: SparsePCA — componentes con ceros (interpretables)

```python
pca = PCA(n_components=3, random_state=42)
pca.fit(X_scaled)

spca = SparsePCA(n_components=3, alpha=1, ridge_alpha=0.01, max_iter=200, random_state=42)
spca.fit(X_scaled)

print('--- PCA estándar (cargas densas) ---')
print(pd.DataFrame(pca.components_, columns=features_pca).round(3))

print('\n--- SparsePCA (cargas con ceros) ---')
print(pd.DataFrame(spca.components_, columns=features_pca).round(3))

# Contar ceros (valores pequeños == 0)
umbral = 0.01
ceros_pca = np.sum(np.abs(pca.components_) < umbral)
ceros_spca = np.sum(np.abs(spca.components_) < umbral)
print(f'\nElementos cercanos a cero:')
print(f'  PCA: {ceros_pca} / {pca.components_.size} ({ceros_pca/pca.components_.size*100:.0f}%)')
print(f'  SparsePCA: {ceros_spca} / {spca.components_.size} ({ceros_spca/spca.components_.size*100:.0f}%)')
print('SparsePCA produce componentes más interpretables (con ceros).')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: SparsePCA — componentes con ceros (interpretables).*

1. Contar ceros (valores pequeños == 0)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 14: TruncatedSVD — para matrices dispersas (como conteo de palabras)

```python
from sklearn.feature_extraction.text import CountVectorizer

# Descripciones de productos
descripciones = [
    'camiseta algodon manga corta verano',
    'camiseta poliester deportiva transpirable',
    'pantalon vaquero azul tela resistente',
    'pantalon chino beige oficina formal',
    'zapatillas running amortiguacion ligera',
    'zapatillas casual cuero comoda',
    'chaqueta impermeable lluvia capucha',
    'chaqueta plumas invierno caliente',
    'bufanda lana invierno caliente accesorio',
    'gorra visera sol verano accesorio',
    'vestido flores primavera mujer',
    'vestido noche elegante largo',
    'camisa formal oficina manga larga',
    'camisa casual manga corta verano',
]

vectorizer = CountVectorizer()
X_text = vectorizer.fit_transform(descripciones)

print(f'Matriz dispersa shape: {X_text.shape}')
print(f'Vocabulario: {len(vectorizer.get_feature_names_out())} términos')

# TruncatedSVD (LSA) sobre matriz de términos
svd = TruncatedSVD(n_components=3, random_state=42)
X_svd = svd.fit_transform(X_text)

print(f'\nReducción a {X_svd.shape[1]} dimensiones')
print(f'Varianza explicada: {svd.explained_variance_ratio_.round(3)}')
print(f'Total: {sum(svd.explained_variance_ratio_)*100:.1f}%')

# Tópicos latentes
topicos = pd.DataFrame(
    svd.components_,
    columns=vectorizer.get_feature_names_out(),
    index=[f'Tópico {i+1}' for i in range(3)]
)
print('\nTópicos latentes (top palabras):')
for i in range(3):
    top = topicos.iloc[i].sort_values(ascending=False).head(5)
    print(f'  Tópico {i+1}: {", ".join(top.index)}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
# Las primeras filas muestran la estructura de los datos procesados:
#    Las columnas dependen del DataFrame utilizado.
# 0    Los valores dependen de los datos de entrada.
# 1    valor4    valor5    valor6
```

**Explicación línea por línea:**

*Ejemplo 14: TruncatedSVD — para matrices dispersas (como conteo de palabras).*

1. Descripciones de productos
2. TruncatedSVD (LSA) sobre matriz de términos
3. Tópicos latentes

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 15: FactorAnalysis — modelo de factores latentes

```python
fa = FactorAnalysis(n_components=3, rotation='varimax', random_state=42)
X_fa = fa.fit_transform(X_scaled)

# Cargas factoriales
cargas_fa = pd.DataFrame(
    fa.components_,
    columns=features_pca,
    index=[f'Factor {i+1}' for i in range(3)]
)

print('Cargas factoriales (FactorAnalysis con rotación varimax):')
print(cargas_fa.round(3))

# Varianza explicada por cada factor (no es exactamente como PCA)
var_fa = np.var(X_fa, axis=0)
var_fa_ratio = var_fa / var_fa.sum()
print(f'\nProporción de varianza por factor:')
for i, v in enumerate(var_fa_ratio):
    print(f'  Factor {i+1}: {v*100:.1f}%')

# Diferencia clave con PCA
print('\nDiferencia: FactorAnalysis modela la covarianza de las variables')
print('mientras que PCA modela la varianza total.')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: FactorAnalysis — modelo de factores latentes.*

1. Cargas factoriales
2. Varianza explicada por cada factor (no es exactamente como PCA)
3. Diferencia clave con PCA

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 16: NMF — componentes no negativos (para matrices de cantidades)

```python
# Datos no negativos (cantidades de productos)
X_pos = np.abs(X_scaled)  # forzar no negatividad

nmf = NMF(n_components=3, init='nndsvd', solver='cd', random_state=42, max_iter=500)
X_nmf = nmf.fit_transform(X_pos)

# Matriz H (componentes): perfil de cada factor
componentes_nmf = pd.DataFrame(
    nmf.components_,
    columns=features_pca,
    index=[f'Factor {i+1}' for i in range(3)]
)

print('Componentes NMF (positivos, aditivos):')
print(componentes_nmf.round(3))

# Matriz W: pesos de cada producto en cada factor
print(f'\nMatriz W (pesos por producto) shape: {X_nmf.shape}')
print(f'Reconstrucción error: {nmf.reconstruction_err_:.2f}')

# Interpretación
print('\nCada factor es una combinación ADITIVA de variables (no diferencias como PCA).')
print('Ideal para datos de conteo: ventas, cantidades, frecuencias.')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: NMF — componentes no negativos (para matrices de cantidades).*

1. Datos no negativos (cantidades de productos)
2. Matriz H (componentes): perfil de cada factor
3. Matriz W: pesos de cada producto en cada factor
4. Interpretación

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 17: PCA + KMeans — clustering en espacio reducido

```python
# Pipeline PCA + KMeans
from sklearn.pipeline import Pipeline

pipeline_pca_kmeans = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=3, random_state=42)),
    ('kmeans', KMeans(n_clusters=4, random_state=42, n_init=10)),
])

pipeline_pca_kmeans.fit(X)
labels_pca_kmeans = pipeline_pca_kmeans.predict(X)

# KMeans solo (sin PCA) para comparar
kmeans_only = KMeans(n_clusters=4, random_state=42, n_init=10)
labels_only = kmeans_only.fit_predict(X_scaled)

# Comparar distribución
print('Distribución de clusters:')
print('  PCA + KMeans:')
vs = pd.Series(labels_pca_kmeans).value_counts().sort_index()
print(f'    {dict(vs)}')
print('  KMeans solo:')
vs2 = pd.Series(labels_only).value_counts().sort_index()
print(f'    {dict(vs2)}')

# Visualizar en 2D
pca_2d = PCA(n_components=2, random_state=42)
X_2d = pca_2d.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=labels_pca_kmeans, cmap='viridis', alpha=0.7)
axes[0].set_title('PCA (3D) + KMeans (4 clusters)')
axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=labels_only, cmap='viridis', alpha=0.7)
axes[1].set_title('KMeans (4 clusters) en 10D')

for ax in axes:
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

plt.tight_layout()
plt.show()

print('\nPCA reduce ruido y puede mejorar la calidad del clustering.')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: PCA + KMeans — clustering en espacio reducido.*

1. Pipeline PCA + KMeans
2. KMeans solo (sin PCA) para comparar
3. Comparar distribución
4. Visualizar en 2D

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 18: Integrador — reducir dimensionalidad de features de productos

```python
# Dataset completo de productos
productos = pd.DataFrame({
    'producto_id': range(1, 501),
    'precio': np.random.uniform(10, 500, 500),
    'costo': np.random.uniform(5, 300, 500),
    'margen_pct': np.random.uniform(10, 60, 500),
    'descuento_max': np.random.uniform(0, 0.7, 500),
    'ventas_mensuales': np.random.poisson(200, 500),
    'devoluciones_mensuales': np.random.poisson(5, 500),
    'dias_stock_promedio': np.random.randint(1, 90, 500),
    'rotacion': np.random.uniform(1, 12, 500),
    'rating_online': np.random.uniform(1, 5, 500),
    'num_reviews': np.random.poisson(50, 500),
    'peso_kg': np.random.uniform(0.1, 30, 500),
    'volumen_cm3': np.random.uniform(50, 8000, 500),
    'costo_envio': np.random.uniform(2, 50, 500),
    'competidores': np.random.randint(0, 20, 500),
})

# Añadir correlaciones
productos['costo'] = 0.4 * productos['precio'] + np.random.normal(0, 10, 500)
productos['margen_pct'] = (productos['precio'] - productos['costo']) / productos['precio'] * 100
productos['ventas_mensuales'] = (200 - 0.5 * productos['precio']
                                  + 20 * productos['descuento_max']
                                  + 3 * productos['rating_online']
                                  + np.random.normal(0, 30, 500)).clip(0)
productos['costo_envio'] = 2 + 0.5 * productos['peso_kg'] + 0.003 * productos['volumen_cm3']

# Features
feats_prod = ['precio', 'costo', 'margen_pct', 'descuento_max', 'ventas_mensuales',
              'devoluciones_mensuales', 'dias_stock_promedio', 'rotacion',
              'rating_online', 'num_reviews', 'peso_kg', 'volumen_cm3',
              'costo_envio', 'competidores']
X_prod = productos[feats_prod].values

# Pipeline de reducción + clustering
scaler = StandardScaler()
X_prod_scaled = scaler.fit_transform(X_prod)

pca_prod = PCA(n_components=0.90, random_state=42)
X_prod_red = pca_prod.fit_transform(X_prod_scaled)

print(f'Features originales: {X_prod.shape[1]}')
print(f'Features reducidas (90% varianza): {X_prod_red.shape[1]}')

# Porcentaje de compresión
compresion = (1 - X_prod_red.shape[1] / X_prod.shape[1]) * 100
print(f'Compresión: {compresion:.1f}%')

# Interpretar componentes
cargas_prod = pd.DataFrame(
    pca_prod.components_,
    columns=feats_prod,
    index=[f'PC{i+1}' for i in range(pca_prod.n_components_)]
)
print('\nCargas de los componentes principales:')
print(cargas_prod.round(3))

# Visualizar variables más importantes en PC1
pc1 = cargas_prod.iloc[0]
top_pos = pc1.nlargest(3)
top_neg = pc1.nsmallest(3)
print(f'\nPC1 explica {pca_prod.explained_variance_ratio_[0]*100:.1f}% de varianza')
print(f'  + correlacionadas: {list(top_pos.index)} ({list(top_pos.values)})')
print(f'  - correlacionadas: {list(top_neg.index)} ({list(top_neg.values)})')

# Clustering en espacio reducido
kmeans_red = KMeans(n_clusters=4, random_state=42, n_init=10)
productos['cluster_red'] = kmeans_red.fit_predict(X_prod_red)

print('\nSegmentos de productos en espacio reducido:')
perfiles = productos.groupby('cluster_red')[feats_prod].mean()
print(perfiles.round(1))

# Evaluar calidad del clustering reducido
sil_red = silhouette_score(X_prod_red, productos['cluster_red'])
print(f'\nSilhouette Score en espacio reducido: {sil_red:.3f}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador — reducir dimensionalidad de features de productos.*

1. Dataset completo de productos
2. Añadir correlaciones
3. Features
4. Pipeline de reducción + clustering
5. Porcentaje de compresión
6. Interpretar componentes
7. Visualizar variables más importantes en PC1
8. Clustering en espacio reducido

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Resumen

| Técnica | Cuándo usarla | Parámetros clave |
|---------|--------------|------------------|
| **PCA** | Reducción general, visualización, preprocesamiento | `n_components`, `whiten`, `svd_solver` |
| **IncrementalPCA** | Datos que no caben en RAM | `batch_size`, `n_components` |
| **KernelPCA** | Datos no lineales (círculos, moons) | `kernel`, `gamma`, `degree` |
| **SparsePCA** | Componentes interpretables (con ceros) | `alpha`, `ridge_alpha` |
| **TruncatedSVD** | Matrices dispersas (texto, conteos) | `n_components`, `algorithm` |
| **FactorAnalysis** | Modelo de factores latentes con ruido | `n_components`, `rotation` |
| **NMF** | Datos no negativos (cantidades, frecuencias) | `n_components`, `init`, `solver` |
| **PCA + KMeans** | Clustering en baja dimensionalidad | PCA preprocesamiento |

**Recomendación:** Siempre estandarizar antes de PCA. Usar `n_components=0.95` para conservar 95% de varianza. KernelPCA para datos no lineales. TruncatedSVD para matrices de texto. NMF para datos de conteo (ventas, cantidades).

---

## Ejercicios

1. Aplica PCA a las 10 variables numéricas del dataset de ventas. ¿Cuántos componentes necesitas para explicar el 85% de la varianza? ¿Cuáles son las 3 variables más importantes en PC1?

2. Usando PCA con 2 componentes, crea un scatterplot coloreado por categoría de producto (`df['categoria']`). ¿Se separan visualmente las categorías en el espacio PCA? ¿Qué categoría está más diferenciada?

3. Compara PCA con `svd_solver='full'` vs `'randomized'` en un dataset de 2000 filas y 20 variables. Mide el tiempo de ajuste y la diferencia en los componentes. ¿Son equivalentes?

4. Aplica `KernelPCA` con kernel rbf a los datos de ventas (gamma=0.1, 1, 10). ¿Cómo cambia la proyección? ¿Qué gamma produce la mejor separación visual?

5. Usa `TruncatedSVD` para reducir las 10 variables de ventas a 3 dimensiones. Compara los resultados con PCA estándar (mismos componentes). ¿Por qué difieren?

6. Crea un `Pipeline` con `StandardScaler` + `PCA(n_components=4)` + `KMeans(n_clusters=3)`. Aplica a todo el dataset. ¿Cómo cambian los clusters vs KMeans sin PCA? Calcula silhouette_score en ambos casos.

7. Usa `FactorAnalysis` con `n_components=3` y `rotation='varimax'`. Interpreta las cargas factoriales. ¿Qué variables definen cada factor? Compáralo con los componentes de PCA.

8. Aplica `NMF` con `n_components=4` a las variables de ventas (forzando no negatividad). Interpreta los 4 factores obtenidos. ¿Qué perfil de producto representa cada factor? ¿Cómo se diferencian de los componentes de PCA?

---

*Teoría y práctica de reducción de dimensionalidad con scikit-learn aplicada a features de productos, ventas e inventarios en el dominio comercial.*
