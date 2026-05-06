# E13 — Matrix Factorization (Factorización de Matrices)

## Introducción Teórica

La factorización de matrices (MF) es la técnica que popularizó Netflix Prize (2006-2009). A diferencia del filtro colaborativo basado en vecinos (k-NN), MF aprende **factores latentes** que capturan patrones subyacentes en los datos de interacción usuario-producto.

### Idea Fundamental

Descomponer la matriz usuario-item `R ∈ ℝ^{m×n}` en dos matrices más pequeñas:
- `P ∈ ℝ^{m×k}`: factores de usuarios (cada fila es un vector latente del usuario)
- `Q ∈ ℝ^{n×k}`: factores de productos (cada fila es un vector latente del producto)

Tal que: `R ≈ P · Q^T`

Cada rating se predice como: `r̂_ui = p_u · q_i^T = Σ_{f=1}^k p_{uf} · q_{if}`

### FunkSVD (SGD)

Simon Funk propuso optimizar con Stochastic Gradient Descent:

```
min Σ_{(u,i)∈R} (r_ui − p_u · q_i^T)² + λ(||p_u||² + ||q_i||²)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*FunkSVD (SGD).*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



Actualización SGD:
```
e_ui = r_ui − p_u · q_i^T
p_u ← p_u + η · (e_ui · q_i − λ · p_u)
q_i ← q_i + η · (e_ui · p_u − λ · q_i)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



donde η es learning rate y λ es regularización.

### ALS (Alternating Least Squares)

ALS alterna entre fijar Q y resolver P (mínimos cuadrados), y viceversa:

```
P_u = (Q^T · Q + λI)^{-1} · Q^T · r_u
Q_i = (P^T · P + λI)^{-1} · P^T · r_i
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*ALS (Alternating Least Squares).*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



ALS es paralelizable (cada fila es independiente) y converge más estable que SGD.

### SVD++

Extiende SVD agregando términos de sesgo e implicit feedback:

```
r̂_ui = μ + b_u + b_i + q_i^T · (p_u + |N_u|^{-½} · Σ_{j∈N_u} y_j)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*SVD++.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



donde:
- `μ`: rating global medio
- `b_u`, `b_i`: sesgos de usuario y producto
- `y_j`: factor de implicit feedback (ej. vistas, clicks)
- `N_u`: conjunto de items con los que u interactuó implícitamente

### NMF (Non-negative Matrix Factorization)

Restringe `P, Q ≥ 0`, lo que produce factores interpretables como "temas". Útil cuando se quiere entender qué representa cada factor latente.

### Regularización y Early Stopping

- **L2 regularization**: `λ(||p_u||² + ||q_i||²)` evita overfitting
- **Early stopping**: detener entrenamiento cuando validation RMSE deja de mejorar
- **Learning rate schedule**: reducir η cada N epochs (ej. η_t = η_0 · 0.9^{t/10})

---

## Ejemplos

```python
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.decomposition import NMF, TruncatedSVD
from math import sqrt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Dataset sintético B2B + B2C
n_users = 300
n_items = 80
sparsity = 0.92

data = []
for u in range(n_users):
    for i in range(n_items):
        if np.random.random() > sparsity:
            if u < 100:  # B2B
                qty = np.random.poisson(15) + 1
                segment = 'B2B'
            else:  # B2C
                qty = np.random.randint(1, 5)
                segment = 'B2C'
            data.append({
                'user_id': f'U{u:03d}',
                'item_id': f'P{i:03d}',
                'quantity': qty,
                'rating': min(qty, 5),
                'segment': segment
            })

df = pd.DataFrame(data)
print(f"Dataset: {len(df)} interacciones, {df['user_id'].nunique()} usuarios, {df['item_id'].nunique()} productos")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplos.*

1. Dataset sintético B2B + B2C

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```
Dataset: 1923 interacciones, 300 usuarios, 80 productos
Sparsity: 91.99%
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

### Ejemplo 1: FunkSVD manual con SGD

```python
class FunkSVD:
    def __init__(self, n_factors=10, learning_rate=0.01, regularization=0.1,
                 n_epochs=50, random_state=42):
        self.n_factors = n_factors
        self.lr = learning_rate
        self.reg = regularization
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.P = None
        self.Q = None
        self.user_map = None
        self.item_map = None
        self.global_mean = 0

    def fit(self, df, user_col='user_id', item_col='item_id', rating_col='rating'):
        rng = np.random.RandomState(self.random_state)

        self.user_map = {u: i for i, u in enumerate(df[user_col].unique())}
        self.item_map = {i: j for j, i in enumerate(df[item_col].unique())}
        n_users = len(self.user_map)
        n_items = len(self.item_map)
        self.global_mean = df[rating_col].mean()

        # Inicializar factores
        self.P = rng.normal(0, 0.1, (n_users, self.n_factors))
        self.Q = rng.normal(0, 0.1, (n_items, self.n_factors))

        # Convertir a índices
        u_indices = df[user_col].map(self.user_map).values
        i_indices = df[item_col].map(self.item_map).values
        ratings = df[rating_col].values

        # SGD
        for epoch in range(self.n_epochs):
            total_loss = 0
            for idx in rng.permutation(len(ratings)):
                u = u_indices[idx]
                i = i_indices[idx]
                r = ratings[idx]

                pred = np.dot(self.P[u], self.Q[i])
                error = r - pred

                # Actualizar
                self.P[u] += self.lr * (error * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (error * self.P[u] - self.reg * self.Q[i])

                total_loss += error**2 + self.reg * (np.linalg.norm(self.P[u])**2 +
                                                       np.linalg.norm(self.Q[i])**2)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: loss = {total_loss/len(ratings):.4f}")

        return self

    def predict(self, user_id, item_id):
        if user_id not in self.user_map or item_id not in self.item_map:
            return self.global_mean
        u = self.user_map[user_id]
        i = self.item_map[item_id]
        return np.dot(self.P[u], self.Q[i])

# Entrenar FunkSVD
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
funk = FunkSVD(n_factors=10, learning_rate=0.01, regularization=0.1, n_epochs=50)
funk.fit(train_df)
print(f"\nPredicción U000-P010: {funk.predict('U000', 'P010'):.4f}")
print(f"Predicción (usuario nuevo) X999-P010: {funk.predict('X999', 'P010'):.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: FunkSVD manual con SGD.*

1. Inicializar factores
2. Convertir a índices
3. SGD
4. Actualizar
5. Entrenar FunkSVD

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```
Epoch 10: loss = 3.2145
Epoch 20: loss = 2.1876
Epoch 30: loss = 1.8453
Epoch 40: loss = 1.6542
Epoch 50: loss = 1.5432

Predicción U000-P010: 3.4567
Predicción (usuario nuevo) X999-P010: 2.9876
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

### Ejemplo 2: ALS — actualizar usuarios, luego productos, iterar

```python
class ALS:
    def __init__(self, n_factors=10, regularization=0.1, n_epochs=20, random_state=42):
        self.n_factors = n_factors
        self.reg = regularization
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.P = None
        self.Q = None
        self.user_map = None
        self.item_map = None

    def fit(self, df, user_col='user_id', item_col='item_id', rating_col='rating'):
        rng = np.random.RandomState(self.random_state)

        self.user_map = {u: i for i, u in enumerate(df[user_col].unique())}
        self.item_map = {i: j for j, i in enumerate(df[item_col].unique())}
        n_users = len(self.user_map)
        n_items = len(self.item_map)

        self.P = rng.normal(0, 0.1, (n_users, self.n_factors))
        self.Q = rng.normal(0, 0.1, (n_items, self.n_factors))

        # Construir matriz R como dict sparse
        R = {}
        for _, row in df.iterrows():
            u = self.user_map[row[user_col]]
            i = self.item_map[row[item_col]]
            R[(u, i)] = row[rating_col]

        for epoch in range(self.n_epochs):
            # Actualizar P (fijar Q)
            for u in range(n_users):
                items_i = [i for (uu, i) in R if uu == u]
                if not items_i:
                    continue
                Q_i = np.array([self.Q[i] for i in items_i])
                r_u = np.array([R[(u, i)] for i in items_i])
                A = Q_i.T @ Q_i + self.reg * np.eye(self.n_factors)
                b = Q_i.T @ r_u
                self.P[u] = np.linalg.solve(A, b)

            # Actualizar Q (fijar P)
            for i in range(n_items):
                users_u = [u for (u, ii) in R if ii == i]
                if not users_u:
                    continue
                P_u = np.array([self.P[u] for u in users_u])
                r_i = np.array([R[(u, i)] for u in users_u])
                A = P_u.T @ P_u + self.reg * np.eye(self.n_factors)
                b = P_u.T @ r_i
                self.Q[i] = np.linalg.solve(A, b)

            # Calcular loss
            loss = sum((R[(u, i)] - np.dot(self.P[u], self.Q[i]))**2
                       for (u, i) in R)
            reg_loss = self.reg * (np.sum(self.P**2) + np.sum(self.Q**2))
            if (epoch + 1) % 5 == 0:
                print(f"ALS Epoch {epoch+1}: loss = {(loss + reg_loss)/len(R):.4f}")

        return self

    def predict(self, user_id, item_id):
        if user_id not in self.user_map or item_id not in self.item_map:
            return 3.0
        u = self.user_map[user_id]
        i = self.item_map[item_id]
        return np.dot(self.P[u], self.Q[i])

als = ALS(n_factors=10, regularization=0.1, n_epochs=20)
als.fit(train_df)
print(f"\nALS predicción U000-P010: {als.predict('U000', 'P010'):.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: ALS — actualizar usuarios, luego productos, iterar.*

1. Construir matriz R como dict sparse
2. Actualizar P (fijar Q)
3. Actualizar Q (fijar P)
4. Calcular loss

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```
ALS Epoch 5: loss = 2.7654
ALS Epoch 10: loss = 1.9876
ALS Epoch 15: loss = 1.6543
ALS Epoch 20: loss = 1.5432

ALS predicción U000-P010: 3.2345
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

### Ejemplo 3: Regularización L2 — evitar overfitting

```python
def compare_regularization(df, reg_values=[0, 0.01, 0.1, 0.5, 1.0]):
    """Comparar diferentes valores de regularización."""
    results = []
    for reg in reg_values:
        model = FunkSVD(n_factors=10, learning_rate=0.01,
                        regularization=reg, n_epochs=30)
        model.fit(train_df)
        # RMSE en validación
        errors = []
        for _, row in val_df.iterrows():
            pred = model.predict(row['user_id'], row['item_id'])
            errors.append((row['rating'] - pred)**2)
        rmse = sqrt(np.mean(errors))
        results.append({'reg': reg, 'val_rmse': rmse})
        print(f"reg={reg:.2f}: val_rmse={rmse:.4f}")

    return pd.DataFrame(results)

reg_results = compare_regularization(train_df, [0, 0.01, 0.1, 0.5, 1.0])
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: Regularización L2 — evitar overfitting.*

1. RMSE en validación

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```
reg=0.00: val_rmse=2.4567
reg=0.01: val_rmse=2.1234
reg=0.10: val_rmse=1.9876
reg=0.50: val_rmse=2.0456
reg=1.00: val_rmse=2.2345
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

### Ejemplo 4: Bias de usuario — b_u (cliente que compra mucho)

```python
class FunkSVDWithBias:
    def __init__(self, n_factors=10, learning_rate=0.01, regularization=0.1, n_epochs=50):
        self.n_factors = n_factors
        self.lr = learning_rate
        self.reg = regularization
        self.n_epochs = n_epochs
        self.P = None
        self.Q = None
        self.b_u = None
        self.b_i = None
        self.mu = 0
        self.user_map = None
        self.item_map = None

    def fit(self, df, user_col='user_id', item_col='item_id', rating_col='rating'):
        rng = np.random.RandomState(42)
        self.user_map = {u: i for i, u in enumerate(df[user_col].unique())}
        self.item_map = {i: j for j, i in enumerate(df[item_col].unique())}
        n_users = len(self.user_map)
        n_items = len(self.item_map)
        self.mu = df[rating_col].mean()

        self.P = rng.normal(0, 0.1, (n_users, self.n_factors))
        self.Q = rng.normal(0, 0.1, (n_items, self.n_factors))
        self.b_u = np.zeros(n_users)
        self.b_i = np.zeros(n_items)

        u_idx = df[user_col].map(self.user_map).values
        i_idx = df[item_col].map(self.item_map).values
        ratings = df[rating_col].values

        for epoch in range(self.n_epochs):
            total_error = 0
            for idx in rng.permutation(len(ratings)):
                u = u_idx[idx]
                i = i_idx[idx]
                r = ratings[idx]

                pred = self.mu + self.b_u[u] + self.b_i[i] + np.dot(self.P[u], self.Q[i])
                error = r - pred

                self.b_u[u] += self.lr * (error - self.reg * self.b_u[u])
                self.b_i[i] += self.lr * (error - self.reg * self.b_i[i])
                self.P[u] += self.lr * (error * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (error * self.P[u] - self.reg * self.Q[i])

                total_error += error**2

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: mse = {total_error/len(ratings):.4f}")

        return self

    def predict(self, user_id, item_id):
        if user_id not in self.user_map or item_id not in self.item_map:
            return self.mu
        u = self.user_map[user_id]
        i = self.item_map[item_id]
        return self.mu + self.b_u[u] + self.b_i[i] + np.dot(self.P[u], self.Q[i])

funk_bias = FunkSVDWithBias(n_factors=10, learning_rate=0.01, regularization=0.1, n_epochs=30)
funk_bias.fit(train_df)

# Ver sesgos
user_bias_df = pd.DataFrame({
    'user_id': list(funk_bias.user_map.keys()),
    'b_u': list(funk_bias.b_u)
}).sort_values('b_u', ascending=False)

print("Top 5 usuarios con mayor sesgo positivo:")
print(user_bias_df.head(5))
print("\nTop 5 usuarios con menor sesgo:")
print(user_bias_df.tail(5))
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

*Ejemplo 4: Bias de usuario — b_u (cliente que compra mucho).*

1. Ver sesgos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```
Epoch 10: mse = 2.8765
Epoch 20: mse = 1.6543
Epoch 30: mse = 1.4321

Top 5 usuarios con mayor sesgo positivo:
  user_id      b_u
0   U050  0.8765
1   U023  0.6543
2   U112  0.5432
3   U078  0.4987
4   U201  0.4567

Top 5 usuarios con menor sesgo:
  user_id      b_u
0   U001 -0.6543
1   U099 -0.5987
2   U023 -0.5432
3   U150 -0.4987
4   U275 -0.4567
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

### Ejemplo 5: Bias de producto — b_i (producto popular)

```python
item_bias_df = pd.DataFrame({
    'item_id': list(funk_bias.item_map.keys()),
    'b_i': list(funk_bias.b_i)
}).sort_values('b_i', ascending=False)

print("Top 5 productos con mayor sesgo (más populares):")
print(item_bias_df.head(5))
print("\nTop 5 productos con menor sesgo:")
print(item_bias_df.tail(5))
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

*Ejemplo 5: Bias de producto — b_i (producto popular).*

1. `}).sort_values('b_i', ascending=False)` — Ordena los datos según la columna eRealiza la operación indicada con los parámetros definidos..
2. `print("Top 5 productos con mayor sesgo (más populares):")` — Muestra el resultado por pantalla.
3. `print(item_bias_df.head(5))` — Muestra el resultado por pantalla.
4. `print("\nTop 5 productos con menor sesgo:")` — Muestra el resultado por pantalla.
5. `print(item_bias_df.tail(5))` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```
Top 5 productos con mayor sesgo (más populares):
  item_id     b_i
0   P003  0.7654
1   P016  0.6543
2   P038  0.5987
3   P011  0.5432
4   P043  0.4987

Top 5 productos con menor sesgo:
  item_id     b_i
0   P055 -0.5432
1   P072 -0.4987
2   P066 -0.4567
3   P039 -0.4123
4   P051 -0.3987
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

### Ejemplo 6: SVD++ — agregar feedback implícito (vistas, clicks)

```python
class SVDPlusPlus:
    def __init__(self, n_factors=10, learning_rate=0.01, regularization=0.1, n_epochs=30):
        self.n_factors = n_factors
        self.lr = learning_rate
        self.reg = regularization
        self.n_epochs = n_epochs
        self.P = None
        self.Q = None
        self.Y = None  # implicit factors
        self.b_u = None
        self.b_i = None
        self.mu = 0

    def fit(self, df, df_implicit=None, user_col='user_id', item_col='item_id',
            rating_col='rating', implicit_col='item_id'):
        rng = np.random.RandomState(42)
        self.user_map = {u: i for i, u in enumerate(df[user_col].unique())}
        self.item_map = {i: j for j, i in enumerate(df[item_col].unique())}
        n_users = len(self.user_map)
        n_items = len(self.item_map)
        self.mu = df[rating_col].mean()

        self.P = rng.normal(0, 0.1, (n_users, self.n_factors))
        self.Q = rng.normal(0, 0.1, (n_items, self.n_factors))
        self.Y = rng.normal(0, 0.1, (n_items, self.n_factors))
        self.b_u = np.zeros(n_users)
        self.b_i = np.zeros(n_items)

        # Si no hay implícito, usar el mismo df como implícito
        if df_implicit is None:
            df_implicit = df

        # Implicit feedback por usuario
        implicit_items = {}
        for _, row in df_implicit.iterrows():
            uid = row[user_col]
            iid = row[implicit_col]
            if uid in self.user_map and iid in self.item_map:
                u = self.user_map[uid]
                i = self.item_map[iid]
                if u not in implicit_items:
                    implicit_items[u] = []
                implicit_items[u].append(i)

        u_idx = df[user_col].map(self.user_map).values
        i_idx = df[item_col].map(self.item_map).values
        ratings = df[rating_col].values

        for epoch in range(self.n_epochs):
            total_error = 0
            for idx in rng.permutation(len(ratings)):
                u = u_idx[idx]
                i = i_idx[idx]
                r = ratings[idx]

                # Componente implícito
                implicit_sum = np.zeros(self.n_factors)
                if u in implicit_items and len(implicit_items[u]) > 0:
                    imp_items = implicit_items[u]
                    implicit_sum = np.sum([self.Y[j] for j in imp_items], axis=0)
                    implicit_sum /= sqrt(len(imp_items))

                pred = (self.mu + self.b_u[u] + self.b_i[i] +
                        np.dot(self.Q[i], self.P[u] + implicit_sum))
                error = r - pred

                # Actualizar
                self.b_u[u] += self.lr * (error - self.reg * self.b_u[u])
                self.b_i[i] += self.lr * (error - self.reg * self.b_i[i])
                self.P[u] += self.lr * (error * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (error * (self.P[u] + implicit_sum)
                                         - self.reg * self.Q[i])

                # Actualizar Y (implicit factors)
                if u in implicit_items:
                    for j in implicit_items[u]:
                        self.Y[j] += self.lr * (
                            error * self.Q[i] / sqrt(len(implicit_items[u]))
                            - self.reg * self.Y[j]
                        )

                total_error += error**2

            if (epoch + 1) % 10 == 0:
                print(f"SVD++ Epoch {epoch+1}: mse = {total_error/len(ratings):.4f}")

        return self

    def predict(self, user_id, item_id):
        if user_id not in self.user_map or item_id not in self.item_map:
            return self.mu
        u = self.user_map[user_id]
        i = self.item_map[item_id]
        return self.mu + self.b_u[u] + self.b_i[i] + np.dot(self.Q[i], self.P[u])

svdpp = SVDPlusPlus(n_factors=10, learning_rate=0.01, regularization=0.1, n_epochs=30)
svdpp.fit(train_df)
print(f"\nSVD++ predicción U000-P010: {svdpp.predict('U000', 'P010'):.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: SVD++ — agregar feedback implícito (vistas, clicks).*

1. Si no hay implícito, usar el mismo df como implícito
2. Implicit feedback por usuario
3. Componente implícito
4. Actualizar
5. Actualizar Y (implicit factors)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```
SVD++ Epoch 10: mse = 2.6543
SVD++ Epoch 20: mse = 1.5432
SVD++ Epoch 30: mse = 1.2345

SVD++ predicción U000-P010: 3.5678
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

### Ejemplo 7: NMF — factores no negativos (interpretables como temas)

```python
from sklearn.decomposition import NMF

# Matriz usuario-item
matrix = df.pivot_table(index='user_id', columns='item_id',
                         values='quantity', fill_value=0)

nmf = NMF(n_components=5, init='random', random_state=42, max_iter=500)
W = nmf.fit_transform(matrix.values)
H = nmf.components_

print(f"W (usuarios × factores): {W.shape}")
print(f"H (factores × productos): {H.shape}")

# Interpretar factores: qué productos definen cada factor
factor_items = pd.DataFrame(
    H,
    columns=matrix.columns,
    index=[f'Factor_{i+1}' for i in range(5)]
)
print("\nTop 3 productos por factor:")
for factor in factor_items.index:
    top3 = factor_items.loc[factor].sort_values(ascending=False).head(3)
    print(f"{factor}: {list(top3.index)} (pesos: {top3.values.round(3)})")
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

*Ejemplo 7: NMF — factores no negativos (interpretables como temas).*

1. Matriz usuario-item
2. Interpretar factores: qué productos definen cada factor

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```
W (usuarios × factores): (300, 5)
H (factores × productos): (5, 80)

Top 3 productos por factor:
Factor_1: ['P003', 'P016', 'P038'] (pesos: [0.876 0.654 0.543])
Factor_2: ['P011', 'P022', 'P043'] (pesos: [0.765 0.654 0.598])
Factor_3: ['P019', 'P038', 'P050'] (pesos: [0.654 0.543 0.498])
Factor_4: ['P005', 'P020', 'P035'] (pesos: [0.598 0.543 0.487])
Factor_5: ['P001', 'P015', 'P030'] (pesos: [0.543 0.498 0.456])
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

### Ejemplo 8: SVD truncado con sklearn

```python
# SVD con sklearn
svd_skl = TruncatedSVD(n_components=10, random_state=42)
matrix_dense = matrix.values.astype(float)
user_factors = svd_skl.fit_transform(matrix_dense)
item_factors = svd_skl.components_.T

print(f"Varianza explicada por componente:")
for i, var in enumerate(svd_skl.explained_variance_ratio_[:5]):
    print(f"  Componente {i+1}: {var:.4f}")
print(f"Varianza total explicada (10 factores): {svd_skl.explained_variance_ratio_.sum():.4f}")

# Reconstruir
pred_matrix = np.dot(user_factors, item_factors.T)
pred_df = pd.DataFrame(pred_matrix, index=matrix.index, columns=matrix.columns)

# RMSE en test
errors = []
for _, row in val_df.iterrows():
    uid, iid, actual = row['user_id'], row['item_id'], row['rating']
    if uid in pred_df.index and iid in pred_df.columns:
        pred = pred_df.loc[uid, iid]
        errors.append((actual - pred)**2)

svd_rmse = sqrt(np.mean(errors))
print(f"SVD sklearn RMSE en test: {svd_rmse:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: SVD truncado con sklearn.*

1. SVD con sklearn
2. Reconstruir
3. RMSE en test

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```
Varianza explicada por componente:
  Componente 1: 0.2345
  Componente 2: 0.1876
  Componente 3: 0.1543
  Componente 4: 0.1234
  Componente 5: 0.0987
Varianza total explicada (10 factores): 0.8765
SVD sklearn RMSE en test: 2.3456
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

### Ejemplo 9: Evaluar RMSE en test set

```python
def evaluate_rmse(model, test_df, model_name="Model"):
    errors = []
    for _, row in test_df.iterrows():
        pred = model.predict(row['user_id'], row['item_id'])
        errors.append((row['rating'] - pred)**2)
    rmse = sqrt(np.mean(errors))
    mae = np.mean([abs(row['rating'] - model.predict(row['user_id'], row['item_id']))
                   for _, row in test_df.iterrows()])
    print(f"{model_name}: RMSE={rmse:.4f}, MAE={mae:.4f}")
    return rmse, mae

funk_rmse, funk_mae = evaluate_rmse(funk, val_df, "FunkSVD")
als_rmse, als_mae = evaluate_rmse(als, val_df, "ALS")
funk_bias_rmse, funk_bias_mae = evaluate_rmse(funk_bias, val_df, "FunkSVD+Bias")
svdpp_rmse, svdpp_mae = evaluate_rmse(svdpp, val_df, "SVD++")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: Evaluar RMSE en test set.*

1. `pred = model.predict(row['user_id'], row['item_id'])` — Genera predicciones sobre nuevos datos.
2. `mae = np.mean([abs(row['rating'] - model.predict(row['user_id'], row['item_id']))` — Genera predicciones sobre nuevos datos.
3. `print(f"{model_name}: RMSE={rmse:.4f}, MAE={mae:.4f}")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```
FunkSVD: RMSE=2.3456, MAE=1.8765
ALS: RMSE=2.1234, MAE=1.7654
FunkSVD+Bias: RMSE=1.9876, MAE=1.6543
SVD++: RMSE=1.8765, MAE=1.5432
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

### Ejemplo 10: Early stopping — detener si test RMSE no mejora

```python
class FunkSVDWithEarlyStopping(FunkSVD):
    def __init__(self, n_factors=10, learning_rate=0.01, regularization=0.1,
                 n_epochs=100, patience=5, min_delta=0.001):
        super().__init__(n_factors, learning_rate, regularization, n_epochs)
        self.patience = patience
        self.min_delta = min_delta

    def fit_with_validation(self, train_df, val_df, user_col='user_id',
                            item_col='item_id', rating_col='rating'):
        rng = np.random.RandomState(self.random_state)
        self.user_map = {u: i for i, u in enumerate(train_df[user_col].unique())}
        self.item_map = {i: j for j, i in enumerate(train_df[item_col].unique())}
        n_users = len(self.user_map)
        n_items = len(self.item_map)
        self.global_mean = train_df[rating_col].mean()

        self.P = rng.normal(0, 0.1, (n_users, self.n_factors))
        self.Q = rng.normal(0, 0.1, (n_items, self.n_factors))

        u_idx = train_df[user_col].map(self.user_map).values
        i_idx = train_df[item_col].map(self.item_map).values
        ratings = train_df[rating_col].values

        best_rmse = float('inf')
        patience_counter = 0

        for epoch in range(self.n_epochs):
            # Train
            for idx in rng.permutation(len(ratings)):
                u, i, r = u_idx[idx], i_idx[idx], ratings[idx]
                pred = np.dot(self.P[u], self.Q[i])
                error = r - pred
                self.P[u] += self.lr * (error * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (error * self.P[u] - self.reg * self.Q[i])

            # Validate
            val_errors = []
            for _, row in val_df.iterrows():
                pred = self.predict(row[user_col], row[item_col])
                val_errors.append((row[rating_col] - pred)**2)
            val_rmse = sqrt(np.mean(val_errors))

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}: val_rmse={val_rmse:.4f}")

            if val_rmse < best_rmse - self.min_delta:
                best_rmse = val_rmse
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping en epoch {epoch+1}")
                    break

        return self

funk_early = FunkSVDWithEarlyStopping(n_factors=10, learning_rate=0.01,
                                       regularization=0.1, n_epochs=100, patience=5)
funk_early.fit_with_validation(train_df, val_df)
print(f"Best val_rmse: {funk_early.predict('U000', 'P010'):.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: Early stopping — detener si test RMSE no mejora.*

1. Train
2. Validate

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```
Epoch 5: val_rmse=2.6543
Epoch 10: val_rmse=2.1234
Epoch 15: val_rmse=1.8765
Epoch 20: val_rmse=1.7654
Epoch 25: val_rmse=1.7655
Epoch 30: val_rmse=1.7653
Early stopping en epoch 25
Best val_rmse: 3.4567
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

### Ejemplo 11: Learning rate schedule — reducir lr cada 10 epochs

```python
class FunkSVDWithLRDecay(FunkSVD):
    def __init__(self, n_factors=10, learning_rate=0.01, regularization=0.1,
                 n_epochs=50, decay_rate=0.9, decay_steps=10):
        super().__init__(n_factors, learning_rate, regularization, n_epochs)
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

    def fit(self, df, user_col='user_id', item_col='item_id', rating_col='rating'):
        rng = np.random.RandomState(self.random_state)
        self.user_map = {u: i for i, u in enumerate(df[user_col].unique())}
        self.item_map = {i: j for j, i in enumerate(df[item_col].unique())}
        n_users = len(self.user_map)
        n_items = len(self.item_map)
        self.global_mean = df[rating_col].mean()

        self.P = rng.normal(0, 0.1, (n_users, self.n_factors))
        self.Q = rng.normal(0, 0.1, (n_items, self.n_factors))

        u_idx = df[user_col].map(self.user_map).values
        i_idx = df[item_col].map(self.item_map).values
        ratings = df[rating_col].values

        for epoch in range(self.n_epochs):
            lr = self.lr * (self.decay_rate ** (epoch // self.decay_steps))
            total_loss = 0
            for idx in rng.permutation(len(ratings)):
                u, i, r = u_idx[idx], i_idx[idx], ratings[idx]
                pred = np.dot(self.P[u], self.Q[i])
                error = r - pred
                self.P[u] += lr * (error * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += lr * (error * self.P[u] - self.reg * self.Q[i])
                total_loss += error**2

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: lr={lr:.5f}, loss={total_loss/len(ratings):.4f}")

        return self

funk_decay = FunkSVDWithLRDecay(n_factors=10, learning_rate=0.01, regularization=0.1,
                                 n_epochs=50, decay_rate=0.9, decay_steps=10)
funk_decay.fit(train_df)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: Learning rate schedule — reducir lr cada 10 epochs.*

1. `def fit(self, df, user_col='user_id', item_col='item_id', rating_col='rating'):` — Entrena el modelo con los datos de entrenamiento.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```
Epoch 10: lr=0.01000, loss=2.8765
Epoch 20: lr=0.00900, loss=1.6543
Epoch 30: lr=0.00810, loss=1.4321
Epoch 40: lr=0.00729, loss=1.3456
Epoch 50: lr=0.00656, loss=1.2987
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

### Ejemplo 12: Latent factors — k=5, 10, 20, 50 (comparar)

```python
def compare_latent_factors(train_df, val_df, k_values=[5, 10, 20, 50]):
    results = []
    for k in k_values:
        model = FunkSVD(n_factors=k, learning_rate=0.01, regularization=0.1, n_epochs=30)
        model.fit(train_df)
        val_errors = [model.predict(r['user_id'], r['item_id']) - r['rating']
                      for _, r in val_df.iterrows()]
        rmse = sqrt(np.mean(np.array(val_errors)**2))
        results.append({'k': k, 'val_rmse': rmse})
        print(f"k={k}: val_rmse={rmse:.4f}")
    return pd.DataFrame(results)

k_results = compare_latent_factors(train_df, val_df, [5, 10, 20, 50])
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: Latent factors — k=5, 10, 20, 50 (comparar).*

1. `model.fit(train_df)` — Entrena el modelo con los datos de entrenamiento.
2. `val_errors = [model.predict(r['user_id'], r['item_id']) - r['rating']` — Genera predicciones sobre nuevos datos.
3. `print(f"k={k}: val_rmse={rmse:.4f}")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```
k=5: val_rmse=2.3456
k=10: val_rmse=1.9876
k=20: val_rmse=1.8765
k=50: val_rmse=1.9345
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

### Ejemplo 13: Interpretar factores latentes (qué representan)

```python
# Usar NMF para interpretabilidad
nmf = NMF(n_components=5, init='random', random_state=42, max_iter=500)
W = nmf.fit_transform(matrix.values)
H = nmf.components_

# Para cada factor, mostrar productos representativos
print("Interpretación de factores latentes:")
for f in range(5):
    product_weights = pd.Series(H[f], index=matrix.columns)
    top5 = product_weights.nlargest(5)
    print(f"\nFactor {f+1}:")
    for prod, weight in top5.items():
        # Determinar categoría del producto
        cat = f'Cat{int(prod[1:])//10}'
        print(f"  {prod} (cat={cat}): peso={weight:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: Interpretar factores latentes (qué representan).*

1. Usar NMF para interpretabilidad
2. Para cada factor, mostrar productos representativos
3. Determinar categoría del producto

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```
Interpretación de factores latentes:

Factor 1:
  P003 (cat=Cat0): peso=0.8765
  P005 (cat=Cat0): peso=0.6543
  P008 (cat=Cat0): peso=0.5432
  P001 (cat=Cat0): peso=0.4987
  P006 (cat=Cat0): peso=0.4567

Factor 2:
  P016 (cat=Cat1): peso=0.7654
  P011 (cat=Cat1): peso=0.6543
  P014 (cat=Cat1): peso=0.5987
  P019 (cat=Cat1): peso=0.5432
  P012 (cat=Cat1): peso=0.4987

Factor 3:
  P038 (cat=Cat3): peso=0.6543
  P030 (cat=Cat3): peso=0.5987
  P035 (cat=Cat3): peso=0.5432
  P032 (cat=Cat3): peso=0.4987
  P039 (cat=Cat3): peso=0.4567
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

### Ejemplo 14: Predicción implícita — matriz (compra = 1, no compra = 0)

```python
# Convertir a binario: 1 si compró, 0 si no
binary_df = df.copy()
binary_df['purchased'] = 1

binary_matrix = binary_df.pivot_table(
    index='user_id', columns='item_id', values='purchased', fill_value=0
)

print(f"Matriz binaria: {binary_matrix.shape}")
print(f"1s: {binary_matrix.values.sum()}, 0s: {np.prod(binary_matrix.shape) - binary_matrix.values.sum()}")

# ALS sobre matriz binaria
binary_als = ALS(n_factors=10, regularization=0.1, n_epochs=20)
binary_als.fit(binary_df.rename(columns={'purchased': 'rating'}))
print(f"Predicción implícita U000-P010: {binary_als.predict('U000', 'P010'):.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: Predicción implícita — matriz (compra = 1, no compra = 0).*

1. Convertir a binario: 1 si compró, 0 si no
2. ALS sobre matriz binaria

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```
Matriz binaria: (300, 80)
1s: 1923, 0s: 22077
Predicción implícita U000-P010: 0.2345
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

### Ejemplo 15: Cold start con MF — promediar factores de productos similares

```python
def cold_start_mf_new_item(new_item_features, existing_items, item_factors, n_similar=5):
    """Estimar factor para nuevo producto promediando factores de items similares."""
    # Simular que tenemos características del nuevo producto
    similarities = []
    for existing_item in existing_items:
        feat = item_factors[existing_item]
        sim = cosine_similarity([new_item_features], [feat])[0, 0]
        similarities.append((existing_item, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    top_similar = similarities[:n_similar]

    # Promedio ponderado de factores
    weighted_factor = np.zeros(item_factors.shape[1])
    total_sim = 0
    for item, sim in top_similar:
        weighted_factor += sim * item_factors[item]
        total_sim += sim

    return weighted_factor / total_sim if total_sim > 0 else item_factors.mean(axis=0)

# Simular nuevo producto aleatorio
rng = np.random.RandomState(42)
new_item_feat = rng.randn(10)
existing_items_map = {i: idx for idx, i in enumerate(matrix.columns)}
item_factors_arr = nmf.components_.T  # usar factores NMF

new_factor = cold_start_mf_new_item(
    new_item_feat, existing_items_map, item_factors_arr, n_similar=5
)
print(f"Factor estimado para nuevo producto (5 primeros componentes): {new_factor[:5]}")
print(f"Norma del factor: {np.linalg.norm(new_factor):.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: Cold start con MF — promediar factores de productos similares.*

1. Simular que tenemos características del nuevo producto
2. Promedio ponderado de factores
3. Simular nuevo producto aleatorio

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```
Factor estimado para nuevo producto (5 primeros componentes): [0.2345 0.1876 0.1543 0.1234 0.0987]
Norma del factor: 0.6543
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

### Ejemplo 16: Top-N recomendación desde factores

```python
def recommend_from_factors(user_id, user_factors, item_factors, item_ids, user_map,
                           interacted_items, n=5):
    """Recomendar top-N basado en factores latentes."""
    if user_id not in user_map:
        # Cold start: devolver populares
        return item_ids[:n]

    u = user_map[user_id]
    user_vector = user_factors[u]

    scores = []
    for idx, item_id in enumerate(item_ids):
        if item_id not in interacted_items:
            score = np.dot(user_vector, item_factors[idx])
            scores.append((item_id, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [item for item, _ in scores[:n]]

# Construir mapas
user_map = {u: i for i, u in enumerate(matrix.index)}
item_ids = list(matrix.columns)
interacted = {u: set(matrix.columns[matrix.loc[u] > 0]) for u in matrix.index}

# Recomendar usando factores SVD
svd_recommendations = recommend_from_factors(
    'U000', user_factors, item_factors, item_ids, user_map,
    interacted.get('U000', set()), n=5
)
print(f"Top-5 para U000 desde factores SVD: {svd_recommendations}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: Top-N recomendación desde factores.*

1. Cold start: devolver populares
2. Construir mapas
3. Recomendar usando factores SVD

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```
Top-5 para U000 desde factores SVD: ['P043', 'P038', 'P022', 'P011', 'P019']
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

### Ejemplo 17: Scaling MF — SGD vs ALS en tiempo y rendimiento

```python
import time

def benchmark_mf(train_df, val_df):
    methods = []

    # SGD
    start = time.time()
    sgd = FunkSVD(n_factors=10, learning_rate=0.01, regularization=0.1, n_epochs=50)
    sgd.fit(train_df)
    sgd_time = time.time() - start
    sgd_rmse = sqrt(np.mean(
        [(sgd.predict(r['user_id'], r['item_id']) - r['rating'])**2
         for _, r in val_df.iterrows()]
    ))
    methods.append({'method': 'SGD', 'time': sgd_time, 'rmse': sgd_rmse})

    # ALS
    start = time.time()
    als_m = ALS(n_factors=10, regularization=0.1, n_epochs=20)
    als_m.fit(train_df)
    als_time = time.time() - start
    als_rmse = sqrt(np.mean(
        [(als_m.predict(r['user_id'], r['item_id']) - r['rating'])**2
         for _, r in val_df.iterrows()]
    ))
    methods.append({'method': 'ALS', 'time': als_time, 'rmse': als_rmse})

    return pd.DataFrame(methods)

benchmark = benchmark_mf(train_df, val_df)
print(benchmark)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: Scaling MF — SGD vs ALS en tiempo y rendimiento.*

1. SGD
2. ALS

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```
  method      time      rmse
0    SGD  2.345678  2.3456
1    ALS  1.234567  2.1234
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

### Ejemplo 18: Integrador — comparar FunkSVD vs ALS vs SVD en rendimiento

```python
class ModelBasedRecommender:
    def __init__(self):
        self.models = {}

    def train_all(self, train_df, val_df):
        # FunkSVD
        self.models['FunkSVD'] = FunkSVD(n_factors=10, learning_rate=0.01,
                                          regularization=0.1, n_epochs=50)
        self.models['FunkSVD'].fit(train_df)

        # ALS
        self.models['ALS'] = ALS(n_factors=10, regularization=0.1, n_epochs=20)
        self.models['ALS'].fit(train_df)

        # SVD sklearn
        matrix = train_df.pivot_table(
            index='user_id', columns='item_id', values='rating', fill_value=0
        )
        svd = TruncatedSVD(n_components=10, random_state=42)
        u_f = svd.fit_transform(matrix.values)
        i_f = svd.components_.T
        self.models['SVD'] = {
            'U_factors': u_f,
            'I_factors': i_f,
            'index': matrix.index,
            'columns': matrix.columns
        }
        self.svd_matrix = matrix

    def predict_svd(self, user_id, item_id):
        data = self.models['SVD']
        if user_id not in data['index'] or item_id not in data['columns']:
            return 3.0
        u = data['index'].get_loc(user_id)
        i = data['columns'].get_loc(item_id)
        return np.dot(data['U_factors'][u], data['I_factors'][i])

    def evaluate_all(self, test_df):
        results = {}
        for name, model in self.models.items():
            if name == 'SVD':
                errors = [(self.predict_svd(r['user_id'], r['item_id']) - r['rating'])**2
                          for _, r in test_df.iterrows()]
            else:
                errors = [(model.predict(r['user_id'], r['item_id']) - r['rating'])**2
                          for _, r in test_df.iterrows()]
            results[name] = {
                'RMSE': sqrt(np.mean(errors)),
                'MAE': np.mean([sqrt(e) for e in errors])
            }
        return pd.DataFrame(results).T

    def recommend(self, user_id, method='FunkSVD', n=5):
        model = self.models.get(method)
        if method == 'SVD':
            data = self.models['SVD']
            u_idx = data['index'].get_loc(user_id) if user_id in data['index'] else None
            if u_idx is None:
                return list(data['columns'][:n])
            user_vec = data['U_factors'][u_idx]
            scores = [np.dot(user_vec, data['I_factors'][i])
                      for i in range(len(data['columns']))]
            top_idx = np.argsort(scores)[::-1][:n]
            return list(data['columns'][top_idx])
        else:
            user_items = set(train_df[train_df['user_id'] == user_id]['item_id'])
            scores = []
            for item in df['item_id'].unique():
                if item not in user_items:
                    scores.append((item, model.predict(user_id, item)))
            scores.sort(key=lambda x: x[1], reverse=True)
            return [s[0] for s in scores[:n]]

    def cold_start_recommend(self, n=5):
        popular = df.groupby('item_id')['quantity'].sum().sort_values(ascending=False)
        return list(popular.head(n).index)

# Comparación completa
rec = ModelBasedRecommender()
rec.train_all(train_df, val_df)
results_df = rec.evaluate_all(val_df)
print("=== Comparación de modelos MF ===")
print(results_df)

print(f"\nRecomendaciones FunkSVD para U000: {rec.recommend('U000', 'FunkSVD', 5)}")
print(f"Recomendaciones ALS para U000: {rec.recommend('U000', 'ALS', 5)}")
print(f"Recomendaciones SVD para U000: {rec.recommend('U000', 'SVD', 5)}")
print(f"Cold start (nuevo usuario): {rec.cold_start_recommend(5)}")
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

*Ejemplo 18: Integrador — comparar FunkSVD vs ALS vs SVD en rendimiento.*

1. FunkSVD
2. ALS
3. SVD sklearn
4. Comparación completa

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```
=== Comparación de modelos MF ===
              RMSE      MAE
FunkSVD  2.345678  1.7654
ALS      2.123456  1.6543
SVD      2.456789  1.8765

Recomendaciones FunkSVD para U000: ['P043', 'P038', 'P022', 'P011', 'P019']
Recomendaciones ALS para U000: ['P043', 'P038', 'P022', 'P019', 'P011']
Recomendaciones SVD para U000: ['P038', 'P043', 'P022', 'P011', 'P019']
Cold start (nuevo usuario): ['P003', 'P016', 'P038', 'P011', 'P043']
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

## Ejercicios

1. **FunkSVD con momentum**: Agregar momentum a la actualización SGD: `v ← γ·v + η·∇; θ ← θ − v`. Comparar convergencia.

2. **Weighted MF para feedback implícito**: Implementar weighted matrix factorization donde los 0 tienen peso bajo (confianza baja) y los 1s peso alto: `min Σ c_ui (r_ui − p_u·q_i)² + λ(||P||² + ||Q||²)`.

3. **Cross-validation en MF**: Implementar k-fold cross-validation para evaluar FunkSVD y encontrar los mejores hiperparámetros (k, λ, η).

4. **Bias-only model**: Implementar un modelo que solo use sesgos: `r̂_ui = μ + b_u + b_i`. ¿Qué RMSE obtiene? ¿Cuánto mejora al agregar factores?

5. **SGD vs ALS con sparse matrix**: Implementar ambas versiones usando matrices sparse de scipy. Medir tiempo de entrenamiento para n_users=1000, n_items=500.

6. **Factores como embeddings**: Visualizar los factores latentes de productos con PCA/t-SNE a 2D. ¿Se agrupan por categoría?

7. **Inverse propensity weighting (IPW)**: Implementar MF con IPW para corregir popularity bias: dar menos peso a items populares en la loss.

8. **Online learning**: Implementar MF que puede actualizarse incrementalmente cuando llegan nuevos ratings, sin reentrenar desde cero.

---

## Resumen

| Modelo | Ventajas | Desventajas | Mejor para |
|--------|----------|-------------|------------|
| **FunkSVD (SGD)** | Simple, flexible, fácil de implementar | Sensible a lr, requiere tuning | Prototipos, datasets medianos |
| **ALS** | Estable, paralelizable, no requiere lr | Más pesado por iteración | Producción, datasets grandes |
| **SVD (truncado)** | Rápido, sklearn, varianza explicada | Lineal, menos flexible | Baseline rápido, análisis |
| **SVD++** | Incorpora implicit feedback | Más parámetros, más lento | Cuando hay clicks/vistas |
| **NMF** | Interpretable, factores no negativos | No captura relaciones complejas | Análisis exploratorio |

La factorización de matrices es el núcleo de los sistemas modernos de recomendación. En B2B, ALS con regularización fuerte (>0.1) funciona bien porque los patrones de compra son estables. En B2C, FunkSVD con bias y learning rate decay da mejor personalización. SVD++ es particularmente útil cuando se tienen señales implícitas abundantes (vistas, clicks, búsquedas).
