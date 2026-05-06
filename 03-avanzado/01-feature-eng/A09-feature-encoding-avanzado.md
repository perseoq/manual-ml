# A09 — Feature Encoding Avanzado para Ventas, Compras e Inventarios

## Fundamentos Teóricos

Muchos algoritmos requieren entrada numérica. El **encoding** transforma variables categóricas a representación numérica preservando información. Para alta cardinalidad, OneHotEncoder explota la dimensionalidad. Los **encodings supervisados** usan el target para codificar.

### Familia de encoders supervisados

| Encoder | Fórmula conceptual | Cuándo usarlo |
|---------|-------------------|---------------|
| **TargetEncoder** | Media del target por categoría con smoothing | Relación lineal categoría-target |
| **LeaveOneOutEncoder** | Media del target excluyendo fila actual | Evitar leakage, datasets medianos |
| **WOEEncoder** | ln(P(Y=1)/P(Y=0)) por categoría | Modelos de riesgo/probabilidad |
| **JamesSteinEncoder** | Shrinkage hacia media global hacia prior | Categorías con pocos datos |
| **MEstimateEncoder** | Prior ponderado por m | Balance entre prior y media |
| **CatBoostEncoder** | Target encoding con orden temporal | Evitar target leakage |

### Familia de encoders de reducción dimensional

| Encoder | Descripción | Dimensiones |
|---------|-------------|-------------|
| **BinaryEncoder** | Entero → binario → columnas | log₂(k) |
| **BaseNEncoder** | Generalización de binary con base n | logₙ(k) |
| **HashingEncoder** | Hash trick con n componentes fijas | n_components |

### Familia de encoders de contraste

| Encoder | Uso típico |
|---------|-----------|
| **SumEncoder** | Comparar cada nivel vs media global |
| **HelmertEncoder** | Comparar cada nivel vs media de niveles anteriores |
| **BackwardDifferenceEncoder** | Comparar niveles adyacentes |
| **PolynomialEncoder** | Contrastes polinomiales para tendencias |

### Instalación

```bash
pip install category_encoders
```

---

## Configuración

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Instalación.*

1. Configuración

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import (
    StandardScaler, LabelEncoder, OneHotEncoder
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, mean_squared_error, classification_report, r2_score
)
from sklearn.compose import ColumnTransformer

# category_encoders
import category_encoders as ce

np.random.seed(42)
```

---

## Ejemplo 1: TargetEncoder — codificar categorías de producto por media de target

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 1: TargetEncoder — codificar categorías de producto por media de target

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
# Dataset de ventas con categorías de producto
np.random.seed(42)
n = 2000
df_ventas = pd.DataFrame({
    'categoria_producto': np.random.choice(
        ['Electrónicos', 'Ropa', 'Alimentos', 'Hogar', 'Deportes', 'Libros'],
        size=n, p=[0.15, 0.25, 0.30, 0.12, 0.10, 0.08]
    ),
    'precio': np.random.uniform(10, 500, n),
    'cantidad': np.random.randint(1, 30, n),
})

# Target binario: venta > 200 USD (1) o no (0)
df_ventas['venta_alta'] = (
    (df_ventas['precio'] * df_ventas['cantidad']) > 200
).astype(int)

# Ver medias por categoría
print("Media de target por categoría:")
print(df_ventas.groupby('categoria_producto')['venta_alta'].mean())

# TargetEncoder
te = ce.TargetEncoder(cols=['categoria_producto'])
X_te = te.fit_transform(df_ventas[['categoria_producto']],
                         df_ventas['venta_alta'])
df_ventas['categoria_encoded'] = X_te['categoria_producto']

print("\nTargetEncoder result:")
print(df_ventas[['categoria_producto', 'categoria_encoded', 'venta_alta']].head(10))
print(f"\nCorrelación encoding vs target: {df_ventas['categoria_encoded'].corr(df_ventas['venta_alta']):.4f}")
```

---

## Ejemplo 2: TargetEncoder con smoothing para evitar overfitting

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 2: TargetEncoder con smoothing para evitar overfitting

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
te_smooth = ce.TargetEncoder(
    cols=['categoria_producto'],
    smoothing=20.0,         # Mayor smoothing → más shrinkage
    min_samples_leaf=5      # Mínimo muestras por categoría
)
X_te_smooth = te_smooth.fit_transform(
    df_ventas[['categoria_producto']], df_ventas['venta_alta']
)

te_nosmooth = ce.TargetEncoder(
    cols=['categoria_producto'],
    smoothing=0.0,
    min_samples_leaf=1
)
X_te_nosmooth = te_nosmooth.fit_transform(
    df_ventas[['categoria_producto']], df_ventas['venta_alta']
)

comparacion = pd.DataFrame({
    'categoria': df_ventas['categoria_producto'],
    'target': df_ventas['venta_alta'],
    'sin_smooth': X_te_nosmooth['categoria_producto'],
    'con_smooth': X_te_smooth['categoria_producto'],
})
print("Comparación smoothing (muestra 10 filas):")
print(comparacion.sample(10))

print("\nValores únicos por método:")
print(f"  Sin smoothing: {comparacion['sin_smooth'].nunique()}")
print(f"  Con smoothing: {comparacion['con_smooth'].nunique()}")

# El smoothing encoge categorías con pocos datos hacia la media global
global_mean = df_ventas['venta_alta'].mean()
print(f"\nMedia global del target: {global_mean:.4f}")
```

---

## Ejemplo 3: LeaveOneOutEncoder — target medio excluyendo fila actual

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 3: LeaveOneOutEncoder — target medio excluyendo fila actual

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
loo = ce.LeaveOneOutEncoder(cols=['categoria_producto'])
X_loo = loo.fit_transform(df_ventas[['categoria_producto']],
                           df_ventas['venta_alta'])

# Demostrar: primera fila de Electrónicos
idx_electronica = df_ventas[df_ventas['categoria_producto'] == 'Electrónicos'].index[0]
media_excluyendo = (df_ventas[
    (df_ventas['categoria_producto'] == 'Electrónicos') &
    (df_ventas.index != idx_electronica)
]['venta_alta'].mean())

print("LeaveOneOutEncoder — verificación:")
print(f"  Categoría: Electrónicos")
print(f"  Target fila actual: {df_ventas.loc[idx_electronica, 'venta_alta']}")
print(f"  Media excluyendo esta fila: {media_excluyendo:.4f}")
print(f"  Valor LOE: {X_loo.loc[idx_electronica, 'categoria_producto']:.4f}")
```

---

## Ejemplo 4: WOEEncoder — peso de evidencia para modelos de riesgo

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 4: WOEEncoder — peso de evidencia para modelos de riesgo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
# WOE = ln(P(Y=1) / P(Y=0)) por categoría
woe = ce.WOEEncoder(cols=['categoria_producto'])
X_woe = woe.fit_transform(df_ventas[['categoria_producto']],
                           df_ventas['venta_alta'])

df_woe = df_ventas.groupby('categoria_producto')['venta_alta'].agg(
    total='count', positivos='sum'
)
df_woe['proporcion_pos'] = df_woe['positivos'] / df_woe['total']
df_woe['proporcion_neg'] = 1 - df_woe['proporcion_pos']
df_woe['woe_manual'] = np.log(
    df_woe['proporcion_pos'] / df_woe['proporcion_neg']
)
print("WOE manual vs encoder para cada categoría:")
for cat in df_woe.index:
    val_encoder = X_woe[df_ventas['categoria_producto'] == cat]['categoria_producto'].iloc[0]
    print(f"  {cat:20s} → WOE manual={df_woe.loc[cat, 'woe_manual']:.4f}, "
          f"encoder={val_encoder:.4f}")

# Interpretación: WOE > 0 significa que la categoría tiene más positivos que negativos
```

---

## Ejemplo 5: JamesSteinEncoder — shrinkage hacia media global

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 5: JamesSteinEncoder — shrinkage hacia media global

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
# Útil cuando hay categorías con muy pocas observaciones
df_js = df_ventas.copy()
# Añadir categorías raras
df_js.loc[0:50, 'categoria_producto'] = 'Instrumentos'
df_js.loc[51:80, 'categoria_producto'] = 'Jardinería'

js = ce.JamesSteinEncoder(cols=['categoria_producto'])
X_js = js.fit_transform(df_js[['categoria_producto']], df_js['venta_alta'])

js_pooled = ce.JamesSteinEncoder(
    cols=['categoria_producto'], model='pooled'
)
X_js_pooled = js_pooled.fit_transform(
    df_js[['categoria_producto']], df_js['venta_alta']
)

print("JamesSteinEncoder: shrinkage hacia media global")
print(f"Media global target: {df_js['venta_alta'].mean():.4f}\n")

for cat in ['Instrumentos', 'Jardinería', 'Electrónicos', 'Alimentos']:
    media_cat = df_js[df_js['categoria_producto'] == cat]['venta_alta'].mean()
    count = df_js[df_js['categoria_producto'] == cat].shape[0]
    val_independent = X_js[df_js['categoria_producto'] == cat]['categoria_producto'].iloc[0]
    val_pooled = X_js_pooled[df_js['categoria_producto'] == cat]['categoria_producto'].iloc[0]
    print(f"  {cat:15s} count={count:4d} media={media_cat:.3f} → "
          f"indep={val_independent:.3f} pooled={val_pooled:.3f}")

print("\n→ Categorías con pocos datos se encogen más hacia la media global")
```

---

## Ejemplo 6: MEstimateEncoder — prior para regularizar

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 6: MEstimateEncoder — prior para regularizar

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
m_encoder = ce.MEstimateEncoder(cols=['categoria_producto'], m=10.0)
X_m = m_encoder.fit_transform(df_ventas[['categoria_producto']],
                               df_ventas['venta_alta'])

# Probar diferentes valores de m
print("MEstimateEncoder con diferentes valores de m:")
for m_val in [1.0, 5.0, 10.0, 50.0, 100.0]:
    enc = ce.MEstimateEncoder(cols=['categoria_producto'], m=m_val)
    X_m_val = enc.fit_transform(df_ventas[['categoria_producto']],
                                 df_ventas['venta_alta'])
    corr = X_m_val['categoria_producto'].corr(df_ventas['venta_alta'])
    print(f"  m={m_val:5.1f} → correlación con target: {corr:.4f}")

# m pequeño → más peso a la media de la categoría
# m grande → más peso al prior (media global)
```

---

## Ejemplo 7: CatBoostEncoder — orden temporal (evita target leakage)

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 7: CatBoostEncoder — orden temporal (evita target leakage)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
# Simula datos ordenados temporalmente (ventas por día)
df_temporal = df_ventas.sort_values('precio').reset_index(drop=True)
df_temporal['fecha_orden'] = range(len(df_temporal))

catboost_enc = ce.CatBoostEncoder(cols=['categoria_producto'])
X_catboost = catboost_enc.fit_transform(
    df_temporal[['categoria_producto']], df_temporal['venta_alta']
)

print("CatBoostEncoder (orden temporal):")
print(f"  Correlación con target: {X_catboost['categoria_producto'].corr(df_temporal['venta_alta']):.4f}")

# Comparar con TargetEncoder normal
te_normal = ce.TargetEncoder(cols=['categoria_producto'])
X_te_norm = te_normal.fit_transform(
    df_temporal[['categoria_producto']], df_temporal['venta_alta']
)
print(f"  Correlación TargetEncoder normal: {X_te_norm['categoria_producto'].corr(df_temporal['venta_alta']):.4f}")

# CatBoostEncoder: cada fila solo usa información de filas anteriores
# TargetEncoder: usa toda la información (incluyendo futura → leakage)
print(f"\nDiferencia entre CatBoost y TargetEncoder:")
print((X_catboost['categoria_producto'] - X_te_norm['categoria_producto']).describe())
```

---

## Ejemplo 8: BinaryEncoder — codificar como binario (menos dimensiones)

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 8: BinaryEncoder — codificar como binario (menos dimensiones)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
# 6 categorías → OneHot = 6 columnas, Binary = 3 columnas (log2(6)=2.58→3)
categorias = df_ventas['categoria_producto'].unique().tolist()
print(f"Categorías ({len(categorias)}): {categorias}")

binary_enc = ce.BinaryEncoder(cols=['categoria_producto'])
X_binary = binary_enc.fit_transform(df_ventas[['categoria_producto']])

print("\nBinaryEncoder result (10 filas):")
print(X_binary.head(10))
print(f"\nShape: {X_binary.shape} (original: {df_ventas[['categoria_producto']].shape})")
print(f"Columnas: {list(X_binary.columns)}")
```

---

## Ejemplo 9: BaseNEncoder con base=3

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 9: BaseNEncoder con base=3

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
basen_enc = ce.BaseNEncoder(cols=['categoria_producto'], base=3)
X_basen = basen_enc.fit_transform(df_ventas[['categoria_producto']])

print("BaseNEncoder (base=3):")
print(X_basen.head(10))
print(f"Shape: {X_basen.shape}")
print(f"Columnas: {list(X_basen.columns)}")

# Comparar dimensionalidad con diferentes bases
for base in [2, 3, 4, 5, 6]:
    enc = ce.BaseNEncoder(cols=['categoria_producto'], base=base)
    X_enc = enc.fit_transform(df_ventas[['categoria_producto']])
    n_dim = X_enc.shape[1]
    print(f"  base={base} → {n_dim} dimensiones (log_{base}({len(categorias)}) ≈ {np.log(len(categorias))/np.log(base):.2f})")
```

---

## Ejemplo 10: HashingEncoder — hash trick para alta cardinalidad

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 10: HashingEncoder — hash trick para alta cardinalidad

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
# Simulamos alta cardinalidad: SKU de producto (5000 únicos)
df_ventas['sku'] = [f'SKU-{i:05d}' for i in np.random.randint(0, 5000, n)]

hashing_enc = ce.HashingEncoder(cols=['sku'], n_components=16)
X_hash = hashing_enc.fit_transform(df_ventas[['sku']])

print(f"HashingEncoder con n_components=16")
print(f"Shape input: {df_ventas[['sku']].shape}")
print(f"Shape output: {X_hash.shape}")
print(f"Columnas: {list(X_hash.columns)}")

# Probar diferentes n_components
print("\nEfecto de n_components:")
for n_comp in [4, 8, 16, 32, 64]:
    enc = ce.HashingEncoder(cols=['sku'], n_components=n_comp)
    X_h = enc.fit_transform(df_ventas[['sku']])
    print(f"  n_components={n_comp:3d} → {X_h.shape[1]} columnas")

# Verificar colisiones
print(f"\nSKUs únicos: {df_ventas['sku'].nunique()}")
print(f"n_components=16: hay colisiones (necesario comprimir >5000 valores en 16 dims)")
```

---

## Ejemplo 11: SumEncoder vs OneHot (contrastes)

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 11: SumEncoder vs OneHot (contrastes)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
# SumEncoder: k columnas, última categoría se infiere (suma = 0)
sum_enc = ce.SumEncoder(cols=['categoria_producto'])
X_sum = sum_enc.fit_transform(df_ventas[['categoria_producto']])

ohe = OneHotEncoder(sparse_output=False, drop='first')
X_ohe = ohe.fit_transform(df_ventas[['categoria_producto']])

print("SumEncoder (contrastes de suma):")
print(X_sum.head(5))
print(f"\nShape SumEncoder: {X_sum.shape}")

# Verificar que la suma por fila da 1
print(f"\nSuma por fila (primeras 5): {X_sum.select_dtypes(include=[np.number]).sum(axis=1).head().tolist()}")

# Comparativa
print(f"\nOneHot (drop='first'): {X_ohe.shape[1]} columnas")
print(f"SumEncoder: {X_sum.shape[1] - 1} columnas numéricas (1 columna de IDs)")
```

---

## Ejemplo 12: HelmertEncoder — contrastes Helmert

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 12: HelmertEncoder — contrastes Helmert

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
helmert_enc = ce.HelmertEncoder(cols=['categoria_producto'])
X_helmert = helmert_enc.fit_transform(df_ventas[['categoria_producto']])

print("HelmertEncoder:")
print(X_helmert.head(10))
print(f"Shape: {X_helmert.shape}")
print(f"Columnas: {list(X_helmert.columns)}")

# Interpretación: cada nivel se compara con la media de los niveles anteriores
# Primer nivel: no tiene comparación → intercepto
```

---

## Ejemplo 13: BackwardDifferenceEncoder

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 13: BackwardDifferenceEncoder

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
bde = ce.BackwardDifferenceEncoder(cols=['categoria_producto'])
X_bde = bde.fit_transform(df_ventas[['categoria_producto']])

print("BackwardDifferenceEncoder:")
print(X_bde.head(10))
print(f"Shape: {X_bde.shape}")

# Interpretación: cada nivel se compara con el nivel siguiente
# Útil cuando las categorías tienen un orden natural
```

---

## Ejemplo 14: Comparar OneHot vs Target vs Binary vs Hashing (dimensionalidad y métrica)

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 14: Comparar OneHot vs Target vs Binary vs Hashing (dimensionalidad y métrica)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

categoricas = ['categoria_producto']
target = 'venta_alta'
features_num = ['precio', 'cantidad']

encoders = {
    'OneHot': OneHotEncoder(sparse_output=False, drop='first'),
    'TargetEncoder': ce.TargetEncoder(cols=categoricas, smoothing=10.0),
    'BinaryEncoder': ce.BinaryEncoder(cols=categoricas),
    'HashingEncoder': ce.HashingEncoder(cols=categoricas, n_components=4),
    'WOEEncoder': ce.WOEEncoder(cols=categoricas),
    'CatBoostEncoder': ce.CatBoostEncoder(cols=categoricas),
}

resultados = []

for nombre, enc in encoders.items():
    if isinstance(enc, ce.BaseEncoder):
        X_cat = enc.fit_transform(df_ventas[categoricas], df_ventas[target])
    else:
        X_cat_vals = enc.fit_transform(df_ventas[categoricas])
        X_cat = pd.DataFrame(
            X_cat_vals,
            columns=[f'{categoricas[0]}_{i}' for i in range(X_cat_vals.shape[1])]
        )

    X_full = pd.concat([
        df_ventas[features_num].reset_index(drop=True),
        X_cat.reset_index(drop=True)
    ], axis=1)

    scores = cross_val_score(
        RandomForestClassifier(n_estimators=50, random_state=42),
        X_full, df_ventas[target], cv=3, scoring='accuracy'
    )

    resultados.append({
        'encoder': nombre,
        'dimensiones': X_full.shape[1],
        'accuracy_mean': scores.mean(),
        'accuracy_std': scores.std()
    })

df_resultados = pd.DataFrame(resultados).sort_values('accuracy_mean', ascending=False)
print("Comparación de encoders:")
print(df_resultados.to_string(index=False))

# Dimensionalidad vs precisión
plt.figure(figsize=(10, 6))
plt.scatter(df_resultados['dimensiones'], df_resultados['accuracy_mean'],
            s=100, c=range(len(df_resultados)), cmap='viridis')
for _, row in df_resultados.iterrows():
    plt.annotate(row['encoder'],
                 (row['dimensiones'], row['accuracy_mean']),
                 xytext=(5, 5), textcoords='offset points')
plt.xlabel('Dimensiones')
plt.ylabel('Accuracy (CV)')
plt.title('Dimensionalidad vs Precisión por Encoder')
plt.grid(True)
plt.show()
```

---

## Ejemplo 15: TargetEncoder en Pipeline con GridSearchCV

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 15: TargetEncoder en Pipeline con GridSearchCV

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
pipeline_enc = Pipeline([
    ('encoder', ce.TargetEncoder(cols=['categoria_producto'])),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, random_state=42)),
])

X = df_ventas[['categoria_producto', 'precio', 'cantidad']]
y = df_ventas['venta_alta']

param_grid = {
    'encoder__smoothing': [0.0, 5.0, 10.0, 20.0],
    'encoder__min_samples_leaf': [1, 5, 10, 20],
    'clf__C': [0.1, 1.0, 10.0],
}

gs_enc = GridSearchCV(
    pipeline_enc, param_grid, cv=5, scoring='accuracy', n_jobs=-1
)
gs_enc.fit(X, y)

print("GridSearchCV con TargetEncoder:")
print(f"Mejores parámetros: {gs_enc.best_params_}")
print(f"Mejor accuracy CV: {gs_enc.best_score_:.4f}")

# pred en test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
gs_enc.fit(X_train, y_train)
print(f"Accuracy en test: {gs_enc.score(X_test, y_test):.4f}")
```

---

## Ejemplo 16: Manejar valores no vistos en test (handle_unknown)

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 16: Manejar valores no vistos en test (handle_unknown)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
# Datos de entrenamiento
df_train = df_ventas.iloc[:1500].copy()
df_test = df_ventas.iloc[1500:].copy()

# Introducir categorías no vistas en test
df_test.loc[df_test.index, 'categoria_producto'] = np.random.choice(
    ['Jardinería', 'Mascotas', 'Bebés'], size=len(df_test), replace=True
)

print(f"Categorías en train: {sorted(df_train['categoria_producto'].unique())}")
print(f"Categorías en test:  {sorted(df_test['categoria_producto'].unique())}")

# TargetEncoder con handle_unknown
te_unknown = ce.TargetEncoder(
    cols=['categoria_producto'],
    handle_unknown='value'  # usa la media global para categorías no vistas
)
te_unknown.fit(df_train[['categoria_producto']], df_train['venta_alta'])
X_test_enc = te_unknown.transform(df_test[['categoria_producto']])

print("\nTargetEncoder con handle_unknown='value':")
test_with_unknown = df_test[df_test['categoria_producto'].isin(
    ['Jardinería', 'Mascotas', 'Bebés']
)]
for cat in ['Jardinería', 'Mascotas', 'Bebés']:
    enc_vals = X_test_enc.loc[
        df_test[df_test['categoria_producto'] == cat].index, 'categoria_producto'
    ]
    print(f"  {cat}: valor encoding = {enc_vals.iloc[0]:.4f} (media global = {df_train['venta_alta'].mean():.4f})")

# Otra opción: handle_unknown='error' lanza excepción
# handle_unknown='return_nan' devuelve NaN
```

---

## Ejemplo 17: Visualizar diferencias entre métodos de encoding

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 17: Visualizar diferencias entre métodos de encoding

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
# Tomar una categoría específica y ver cómo cada método la transforma
categoria_ejemplo = 'Electrónicos'
idx_cat = df_ventas[df_ventas['categoria_producto'] == categoria_ejemplo].index[:1]

metodos_vis = {
    'Target (smooth=10)': ce.TargetEncoder(cols=['categoria_producto'], smoothing=10.0),
    'Target (smooth=0)': ce.TargetEncoder(cols=['categoria_producto'], smoothing=0.0),
    'LeaveOneOut': ce.LeaveOneOutEncoder(cols=['categoria_producto']),
    'WOE': ce.WOEEncoder(cols=['categoria_producto']),
    'JamesStein': ce.JamesSteinEncoder(cols=['categoria_producto']),
    'MEstimate (m=10)': ce.MEstimateEncoder(cols=['categoria_producto'], m=10.0),
    'CatBoost': ce.CatBoostEncoder(cols=['categoria_producto']),
}

resultados_vis = []
for nombre, enc in metodos_vis.items():
    X_enc = enc.fit_transform(df_ventas[['categoria_producto']], df_ventas['venta_alta'])
    valor = X_enc.loc[idx_cat[0], 'categoria_producto']
    resultados_vis.append({'método': nombre, 'valor_encoding': valor})

df_vis = pd.DataFrame(resultados_vis)
print(f"Valores de encoding para la categoría '{categoria_ejemplo}':")
print(df_vis.to_string(index=False))

# Gráfico comparativo
plt.figure(figsize=(10, 5))
bars = plt.barh(df_vis['método'], df_vis['valor_encoding'], color='cornflowerblue')
plt.axvline(df_ventas['venta_alta'].mean(), color='red', linestyle='--',
            label=f'Media global target = {df_ventas["venta_alta"].mean():.2f}')
plt.xlabel('Valor de encoding')
plt.title(f'Comparación de encoders para categoría: {categoria_ejemplo}')
plt.legend()
plt.tight_layout()
plt.show()
```

---

## Ejemplo 18: Integrador — elegir mejor encoding para categorías de producto

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 18: Integrador — elegir mejor encoding para categorías de producto

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
# === DATOS DE COMPRAS ===
np.random.seed(42)
n = 3000
df_compras = pd.DataFrame({
    'proveedor': np.random.choice(
        ['Distribuidora A', 'Distribuidora B', 'Importadora X', 'Fabrica Y',
         'Mayorista Z', 'Proveedor Local', 'Importador Global', 'Cooperativa'],
        size=n, p=[0.20, 0.15, 0.12, 0.18, 0.10, 0.13, 0.07, 0.05]
    ),
    'metodo_pago': np.random.choice(
        ['Contado', 'Crédito 30d', 'Crédito 60d', 'Crédito 90d', 'Leasing'],
        size=n, p=[0.30, 0.35, 0.20, 0.10, 0.05]
    ),
    'categoria': np.random.choice(
        ['Materia Prima', 'Insumo', 'Empaque', 'Servicio', 'Equipo'],
        size=n, p=[0.35, 0.25, 0.20, 0.12, 0.08]
    ),
    'monto': np.random.uniform(100, 50000, n),
    'dias_entrega': np.random.exponential(15, n),
})

# Target: compra considerada "cara" (> percentil 75)
threshold = np.percentile(df_compras['monto'], 75)
df_compras['compra_cara'] = (df_compras['monto'] > threshold).astype(int)

# Evaluar múltiples encoders con CV
X_compras = df_compras[['proveedor', 'metodo_pago', 'categoria', 'monto', 'dias_entrega']]
y_compras = df_compras['compra_cara']
cols_cat = ['proveedor', 'metodo_pago', 'categoria']

encoders_a_evaluar = {
    'OneHot': OneHotEncoder(sparse_output=False, drop='first'),
    'TargetEncoder': ce.TargetEncoder(cols=cols_cat, smoothing=15.0),
    'LeaveOneOut': ce.LeaveOneOutEncoder(cols=cols_cat),
    'WOE': ce.WOEEncoder(cols=cols_cat),
    'JamesStein': ce.JamesSteinEncoder(cols=cols_cat),
    'MEstimate': ce.MEstimateEncoder(cols=cols_cat, m=5.0),
    'CatBoost': ce.CatBoostEncoder(cols=cols_cat),
    'Binary': ce.BinaryEncoder(cols=cols_cat),
    'Hashing(8)': ce.HashingEncoder(cols=cols_cat, n_components=8),
}

rf = RandomForestClassifier(n_estimators=80, random_state=42)
resultados_finales = []

for nombre, enc in encoders_a_evaluar.items():
    if isinstance(enc, ce.BaseEncoder):
        X_enc = enc.fit_transform(X_compras[cols_cat], y_compras)
    else:
        X_enc_vals = enc.fit_transform(X_compras[cols_cat])
        X_enc = pd.DataFrame(X_enc_vals,
                              columns=[f'cat_{i}' for i in range(X_enc_vals.shape[1])])

    X_full = pd.concat([
        X_compras[['monto', 'dias_entrega']].reset_index(drop=True),
        X_enc.reset_index(drop=True)
    ], axis=1)

    scores = cross_val_score(rf, X_full, y_compras, cv=5, scoring='roc_auc')

    resultados_finales.append({
        'encoder': nombre,
        'dimensiones': X_full.shape[1],
        'roc_auc_mean': scores.mean(),
        'roc_auc_std': scores.std()
    })

df_final = pd.DataFrame(resultados_finales).sort_values('roc_auc_mean', ascending=False)
print("=== Comparación Final de Encoders para COMPRAS ===")
print(df_final.to_string(index=False))

# Mejor encoder
best = df_final.iloc[0]
print(f"\n✓ Mejor encoder: {best['encoder']} (AUC={best['roc_auc_mean']:.4f})")

# Si el mejor es TargetEncoder, mostrar parámetros óptimos
if 'Target' in best['encoder']:
    print("\nRecomendación: usar TargetEncoder con smoothing vía GridSearch")

# Gráfico final: AUC vs Dimensiones
plt.figure(figsize=(12, 6))
colors = plt.cm.Set2(np.linspace(0, 1, len(df_final)))
plt.subplot(1, 2, 1)
bars = plt.bar(df_final['encoder'], df_final['roc_auc_mean'], color=colors)
plt.xticks(rotation=45, ha='right')
plt.ylabel('ROC-AUC (CV)')
plt.title('Precisión por Encoder')
plt.axhline(y=df_final['roc_auc_mean'].max(), color='r', linestyle='--',
            label=f'Máximo: {df_final["roc_auc_mean"].max():.4f}')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(df_final['dimensiones'], df_final['roc_auc_mean'],
            s=100, c=range(len(df_final)), cmap='Set2')
for _, row in df_final.iterrows():
    plt.annotate(row['encoder'],
                 (row['dimensiones'], row['roc_auc_mean']),
                 fontsize=8, xytext=(3, 3), textcoords='offset points')
plt.xlabel('Dimensiones')
plt.ylabel('ROC-AUC')
plt.title('Trade-off: Dimensiones vs Precisión')
plt.grid(True)
plt.tight_layout()
plt.show()
```

---

## Resumen

| Encoder | Tipo | Dimensiones | Target Leakage | Ideal para |
|---------|------|-------------|----------------|------------|
| **TargetEncoder** | Supervisado | = categorías | Posible (usar CV) | Relación lineal target-categoría |
| **LeaveOneOutEncoder** | Supervisado | = categorías | Menos que Target | Datasets medianos |
| **WOEEncoder** | Supervisado | = categorías | Posible | Modelos de probabilidad |
| **JamesSteinEncoder** | Supervisado | = categorías | Bajo | Categorías con pocos datos |
| **MEstimateEncoder** | Supervisado | = categorías | Bajo | Balance prior/media |
| **CatBoostEncoder** | Supervisado | = categorías | Mínimo | Datos ordenados temporalmente |
| **BinaryEncoder** | No supervisado | log₂(k) | No | Alta cardinalidad |
| **BaseNEncoder** | No supervisado | logₙ(k) | No | Flexibilidad dimensional |
| **HashingEncoder** | No supervisado | n_components | No | Cardinalidad extrema |
| **SumEncoder** | Contraste | k-1 | No | Modelos lineales |
| **HelmertEncoder** | Contraste | k-1 | No | ANOVA / contrastes |
| **BackwardDifference** | Contraste | k-1 | No | Categorías ordenadas |

**Recomendación:** Para datos de ventas/compras con categorías de cardinalidad moderada (< 50 niveles), TargetEncoder con smoothing es la opción más popular. Para cardinalidad alta (SKUs, proveedores), BinaryEncoder o HashingEncoder evitan la explosión dimensional.

---

## Ejercicios

1. Crea un dataset de ventas con 5 categorías de producto y target continuo (monto total). Aplica TargetEncoder, LeaveOneOutEncoder y CatBoostEncoder. Calcula la correlación de cada encoding con el target y explica por qué difieren.

2. Usando el dataset de compras del ejemplo 18, compara WOEEncoder vs JamesSteinEncoder para la columna 'proveedor' (8 niveles). ¿Cuál es la diferencia clave? ¿En qué casos JamesSteinEncoder es preferible?

3. Simula un dataset de inventarios con 100 SKUs diferentes (alta cardinalidad). Codifícalos con OneHotEncoder, BinaryEncoder y HashingEncoder(n_components=8). Compara el número de dimensiones generadas y el accuracy de un RandomForestClassifier con cada método.

4. Implementa un Pipeline con ColumnTransformer que aplique TargetEncoder a las columnas categóricas y StandardScaler a las numéricas. Usa GridSearchCV para optimizar smoothing y min_samples_leaf del TargetEncoder sobre un dataset de ventas. Reporta los mejores parámetros.

5. Genera datos donde intentionalmente haya categorías no vistas en test. Aplica TargetEncoder con handle_unknown='value', 'return_nan' y 'error'. Documenta el comportamiento de cada opción.

6. Para el dataset de compras con target binario, compara visualmente 5 encoders (Target, WOE, Binary, Hashing, OneHot) usando un barplot de ROC-AUC y otro de dimensiones. ¿Qué encoder recomiendas?

7. Aplica HelmertEncoder y BackwardDifferenceEncoder al dataset de ventas con 6 categorías de producto. Interpreta los coeficientes de un modelo de regresión logística entrenado con estos encodings. ¿Qué categorías se comparan?

8. **Integrador:** Diseña un experimento completo para elegir el mejor encoding en datos de inventarios: (a) Dataset con 3 columnas categóricas (tipo_producto, ubicación_almacén, proveedor) y target "rotación_alta", (b) Evalúa 6 encoders diferentes con validación cruzada estratificada, (c) Para el mejor encoder, optimiza parámetros con GridSearchCV, (d) Entrena el modelo final y evalúa en test, (e) Documenta todo el proceso y concluye.
