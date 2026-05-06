# CP28: 30 Ejercicios Integradores de Nivel Avanzado

## Resumen Ejecutivo

Colección de 30 ejercicios integradores que consolidan todo lo aprendido en el nivel avanzado de Data Science y Machine Learning aplicado a Ventas, Compras e Inventarios. Cada ejercicio incluye enunciado, pista, código solución y explicación. Se cubren pipelines, ensemble, NLP, TensorFlow, PyTorch, CNN, RNN, LSTM, autoencoders y transfer learning.

**Nivel:** Avanzado
**Requisitos:** Python, ML intermedio, deep learning básico
**Duración estimada:** 40-60 horas

---

## A01: Pipeline Completo con GridSearchCV

**Enunciado:** Construir un pipeline de sklearn que incluya StandardScaler, PCA (con 95% de varianza), y RandomForestClassifier. Optimizar hiperparámetros con GridSearchCV sobre datos de clasificación de productos por precio (alto/bajo).

```python
# Pista: usa Pipeline de sklearn, define param_grid con n_estimators y max_depth
# La columna target es 'precio_alto' (1 si precio > mediana, 0 si no)

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Datos sintéticos
np.random.seed(42)
n = 1000
X_clf = pd.DataFrame({
    'volumen': np.random.normal(100, 30, n),
    'peso': np.random.normal(5, 2, n),
    'popularidad': np.random.uniform(0, 10, n),
    'costo_produccion': np.random.normal(50, 20, n)
})
y_clf = (X_clf['costo_produccion'] + X_clf['popularidad'] * 5 > 100).astype(int)

# Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
    ('clf', RandomForestClassifier(random_state=42))
])

param_grid = {
    'clf__n_estimators': [50, 100, 200],
    'clf__max_depth': [5, 10, None],
    'clf__min_samples_split': [2, 5, 10]
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_clf, y_clf)

print(f"Mejores parámetros: {grid.best_params_}")
print(f"Mejor accuracy CV: {grid.best_score_:.4f}")
print(f"Test score: {grid.score(X_clf, y_clf):.4f}")
print(f"PCA componentes seleccionadas: {grid.best_estimator_.named_steps['pca'].n_components_}")
```

**Explicación:** Pipeline encadena transformaciones y modelo. GridSearchCV optimiza hiperparámetros sobre todas las etapas. PCA con 95% reduce dimensionalidad automáticamente.

---

## A02: Pipeline con Transformadores Personalizados

**Enunciado:** Crear un transformador personalizado que agregue features de interacción (producto de pares de features) y lo integre en un pipeline con Ridge regression.



**Explicación:** Pipeline encadena transformaciones y modelo. GridSearchCV optimiza hiperparámetros sobre todas las etapas. PCA con 95% reduce dimensionalidad automáticamente.

---

## A02: Pipeline con Transformadores Personalizados

**Enunciado:** Crear un transformador personalizado que agregue features de interacción (producto de pares de features) y lo integre en un pipeline con Ridge regression.

```python
# Pista: implementa BaseEstimator y TransformerMixin

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

class InteractionFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, degree=2):
        self.degree = degree
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = np.asarray(X)
        n_features = X.shape[1]
        interactions = []
        for i in range(n_features):
            for j in range(i+1, n_features):
                interactions.append(X[:, i] * X[:, j])
        return np.column_stack([X] + interactions)

# Datos
np.random.seed(42)
X_inter = np.random.randn(500, 4)
y_inter = X_inter[:, 0] * X_inter[:, 1] + X_inter[:, 2] + np.random.randn(500) * 0.1

pipeline_inter = Pipeline([
    ('interactions', InteractionFeatures()),
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=1.0))
])

scores = cross_val_score(pipeline_inter, X_inter, y_inter, cv=5, scoring='r2')
print(f"R2 con interacciones: {scores.mean():.4f} ± {scores.std():.4f}")

# Comparar sin interacciones
pipeline_base = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=1.0))
])
scores_base = cross_val_score(pipeline_base, X_inter, y_inter, cv=5, scoring='r2')
print(f"R2 sin interacciones: {scores_base.mean():.4f} ± {scores_base.std():.4f}")
```

**Explicación:** Los transformadores personalizados permiten ingeniería de features arbitraria dentro del pipeline. Las interacciones capturan relaciones no lineales entre pares de features.

---

## A03: ColumnTransformer con Features Mixtas

**Enunciado:** Usar ColumnTransformer para aplicar diferentes transformaciones a features numéricas y categóricas: escalado para numéricas, one-hot para categóricas.



**Explicación:** Los transformadores personalizados permiten ingeniería de features arbitraria dentro del pipeline. Las interacciones capturan relaciones no lineales entre pares de features.

---

## A03: ColumnTransformer con Features Mixtas

**Enunciado:** Usar ColumnTransformer para aplicar diferentes transformaciones a features numéricas y categóricas: escalado para numéricas, one-hot para categóricas.

```python
# Pista: usa ColumnTransformer con make_column_selector

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Datos mixtos
np.random.seed(42)
df_mix = pd.DataFrame({
    'precio': np.random.normal(100, 20, 1000),
    'cantidad': np.random.poisson(5, 1000),
    'categoria': np.random.choice(['A', 'B', 'C', 'D'], 1000),
    'metodo_pago': np.random.choice(['tarjeta', 'efectivo', 'transferencia'], 1000),
    'target': np.random.normal(50, 10, 1000)
})

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), make_column_selector(dtype_include=np.number)),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), make_column_selector(dtype_include=object))
])

pipeline_mix = Pipeline([
    ('prep', preprocessor),
    ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))
])

X = df_mix.drop('target', axis=1)
y = df_mix['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline_mix.fit(X_train, y_train)
print(f"R2 en test: {pipeline_mix.score(X_test, y_test):.4f}")

# Inspeccionar features después de transformación
X_transformed = preprocessor.fit_transform(X)
print(f"Features originales: {X.shape[1]}")
print(f"Features después de transformación: {X_transformed.shape[1]}")
```

**Explicación:** ColumnTransformer aplica transformaciones diferentes por tipo de columna. Esencial para datasets del mundo real con features mixtas.

---

## A04: Ensemble Avanzado (Stacking + Blending)

**Enunciado:** Implementar StackingClassifier con 3 modelos base (RandomForest, XGBoost, LogisticRegression) y un meta-modelo (GradientBoosting) para clasificación de ventas.



**Explicación:** ColumnTransformer aplica transformaciones diferentes por tipo de columna. Esencial para datasets del mundo real con features mixtas.

---

## A04: Ensemble Avanzado (Stacking + Blending)

**Enunciado:** Implementar StackingClassifier con 3 modelos base (RandomForest, XGBoost, LogisticRegression) y un meta-modelo (GradientBoosting) para clasificación de ventas.

```python
# Pista: usa StackingClassifier de sklearn

from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
import numpy as np

# Generar datos
X_stack, y_stack = make_classification(n_samples=2000, n_features=20, n_informative=15, 
                                         n_redundant=3, random_state=42)

# Stacking
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('xgb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('lr', LogisticRegression(max_iter=1000, random_state=42))
]

meta_model = GradientBoostingClassifier(n_estimators=50, random_state=42)

stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,
    stack_method='predict_proba'
)

# Evaluar cada modelo individualmente
for name, model in base_models:
    scores = cross_val_score(model, X_stack, y_stack, cv=5, scoring='accuracy')
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")

# Evaluar stacking
scores_stack = cross_val_score(stacking, X_stack, y_stack, cv=5, scoring='accuracy')
print(f"Stacking: {scores_stack.mean():.4f} ± {scores_stack.std():.4f}")
print(f"Mejora vs mejor base: {(scores_stack.mean() - max([cross_val_score(m, X_stack, y_stack, cv=5).mean() for _, m in base_models]))*100:.2f}%")
```

**Explicación:** Stacking combina predicciones de modelos base usando un meta-modelo. Suele superar a modelos individuales al corregir sesgos complementarios.

---

## A05: Regresión Lineal con Regularización (Lasso + Ridge + ElasticNet)

**Enunciado:** Comparar Lasso, Ridge y ElasticNet para seleccionar features relevantes en predicción de demanda. Usar validación cruzada para alpha óptimo.



**Explicación:** Stacking combina predicciones de modelos base usando un meta-modelo. Suele superar a modelos individuales al corregir sesgos complementarios.

---

## A05: Regresión Lineal con Regularización (Lasso + Ridge + ElasticNet)

**Enunciado:** Comparar Lasso, Ridge y ElasticNet para seleccionar features relevantes en predicción de demanda. Usar validación cruzada para alpha óptimo.

```python
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

np.random.seed(42)
n = 500
p = 50
X_reg = np.random.randn(n, p)
# Solo 5 features relevantes
beta = np.zeros(p)
beta[:5] = [2, -1.5, 3, 0.5, -2]
y_reg = X_reg @ beta + np.random.randn(n) * 0.5

X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

# Lasso (selecciona features)
lasso = LassoCV(cv=5, random_state=42).fit(X_tr_s, y_tr)
print(f"Lasso - alpha óptimo: {lasso.alpha_:.4f}")
print(f"Features no cero: {np.sum(lasso.coef_ != 0)} de {p}")
print(f"R2 test: {lasso.score(X_te_s, y_te):.4f}")

# Ridge (encoge pero no selecciona)
ridge = RidgeCV(cv=5).fit(X_tr_s, y_tr)
print(f"\nRidge - alpha óptimo: {ridge.alpha_:.4f}")
print(f"Features no cero: {np.sum(ridge.coef_ != 0)} de {p}")
print(f"R2 test: {ridge.score(X_te_s, y_te):.4f}")

# ElasticNet (balance entre Lasso y Ridge)
en = ElasticNetCV(cv=5, l1_ratio=[.1, .5, .7, .9, .95, .99, 1], random_state=42).fit(X_tr_s, y_tr)
print(f"\nElasticNet - alpha: {en.alpha_:.4f}, l1_ratio: {en.l1_ratio_:.2f}")
print(f"Features no cero: {np.sum(en.coef_ != 0)} de {p}")
print(f"R2 test: {en.score(X_te_s, y_te):.4f}")
```

**Explicación:** Lasso hace selección de features (coeficientes a cero). Ridge encoge todos los coeficientes. ElasticNet combina ambas.

---

## A06: Métricas de Negocio: Profit Curve y Lift Chart

**Enunciado:** Implementar profit curve y lift chart para evaluar un modelo de clasificación de clientes, considerando costo de campaña y beneficio por cliente convertido.



**Explicación:** Lasso hace selección de features (coeficientes a cero). Ridge encoge todos los coeficientes. ElasticNet combina ambas.

---

## A06: Métricas de Negocio: Profit Curve y Lift Chart

**Enunciado:** Implementar profit curve y lift chart para evaluar un modelo de clasificación de clientes, considerando costo de campaña y beneficio por cliente convertido.

```python
# Pista: ordena por probabilidad decreciente y calcula profit acumulado

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n = 5000
X_prof = np.random.randn(n, 10)
y_prof = (X_prof[:, 0] + X_prof[:, 1] * 2 + np.random.randn(n) * 0.5 > 0).astype(int)

X_tr, X_te, y_tr, y_te = train_test_split(X_prof, y_prof, test_size=0.3, random_state=42)
model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_tr, y_tr)
probs = model.predict_proba(X_te)[:, 1]

# Parámetros de negocio
costo_contacto = 5
beneficio_conversion = 50
umbral_costo = costo_contacto / beneficio_conversion

# Profit curve
idx = np.argsort(probs)[::-1]
y_sorted = y_te[idx]
probs_sorted = probs[idx]

profit_acum = np.zeros(len(y_sorted))
for i in range(len(y_sorted)):
    ingresos = y_sorted[:i+1].sum() * beneficio_conversion
    costos = (i+1) * costo_contacto
    profit_acum[i] = ingresos - costos

opt_idx = np.argmax(profit_acum)
print(f"Máximo profit: ${profit_acum[opt_idx]:.0f} contactando {opt_idx+1} clientes")
print(f"Threshold óptimo: {probs_sorted[opt_idx]:.4f} (umbral costo: {umbral_costo:.4f})")

# Lift chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(range(len(profit_acum)), profit_acum)
axes[0].axvline(opt_idx, color='red', linestyle='--', label=f'Óptimo: {opt_idx} clientes')
axes[0].set_xlabel('Clientes contactados')
axes[0].set_ylabel('Profit acumulado ($)')
axes[0].set_title('Profit Curve')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Lift chart
baseline = y_te.mean()
lift = np.cumsum(y_sorted) / (np.arange(1, len(y_sorted)+1)) / baseline
axes[1].plot(range(len(lift)), lift)
axes[1].axhline(1, color='gray', linestyle='--')
axes[1].set_xlabel('Clientes contactados')
axes[1].set_ylabel('Lift (vs aleatorio)')
axes[1].set_title('Lift Chart')
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('img/profit_lift_chart.png', dpi=150)
plt.show()
```

**Explicación:** Profit curve traduce métricas de ML a impacto financiero. Lift chart muestra cuánto mejora el modelo vs selección aleatoria.

---

## A07: Feature Engineering con Polinomiales y Splines

**Enunciado:** Generar features polinomiales (grado 3) y B-splines para capturar no linealidades en predicción de demanda vs precio.



**Explicación:** Profit curve traduce métricas de ML a impacto financiero. Lift chart muestra cuánto mejora el modelo vs selección aleatoria.

---

## A07: Feature Engineering con Polinomiales y Splines

**Enunciado:** Generar features polinomiales (grado 3) y B-splines para capturar no linealidades en predicción de demanda vs precio.

```python
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n = 300
precio = np.linspace(10, 100, n)
# Relación no lineal: demanda = 500 - 10*precio + 0.1*precio^2 + ruido
demanda = 500 - 10*precio + 0.1*precio**2 + np.random.normal(0, 20, n)

X = precio.reshape(-1, 1)

# Lineal
lr = LinearRegression()
score_lin = cross_val_score(lr, X, demanda, cv=5, scoring='r2').mean()

# Polinomial (grado 3)
poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly.fit_transform(X)
score_poly = cross_val_score(LinearRegression(), X_poly, demanda, cv=5, scoring='r2').mean()

# Splines
spline = SplineTransformer(n_knots=8, degree=3)
X_spline = spline.fit_transform(X)
score_spline = cross_val_score(LinearRegression(), X_spline, demanda, cv=5, scoring='r2').mean()

print(f"R2 lineal: {score_lin:.4f}")
print(f"R2 polinomial (grado 3): {score_poly:.4f}")
print(f"R2 splines (knots=8): {score_spline:.4f}")

# Visualizar
X_plot = np.linspace(10, 100, 300).reshape(-1, 1)
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(precio, demanda, alpha=0.3, label='Datos')
ax.plot(X_plot, LinearRegression().fit(X_poly, demanda).predict(poly.transform(X_plot)), 
        label=f'Polinomial (R2={score_poly:.3f})', linewidth=2)
ax.plot(X_plot, LinearRegression().fit(X_spline, demanda).predict(spline.transform(X_plot)), 
        label=f'Splines (R2={score_spline:.3f})', linewidth=2)
ax.set_xlabel('Precio')
ax.set_ylabel('Demanda')
ax.set_title('Captura de No Linealidad: Polinomiales vs Splines')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('img/polinomiales_splines.png', dpi=150)
plt.show()
```

**Explicación:** Features polinomiales y splines capturan relaciones no lineales sin cambiar de modelo. Splines son más estables en los extremos.

---

## A08: Target Encoding para Categorías de Alta Cardinalidad

**Enunciado:** Implementar Target Encoding para codificar 100 categorías de producto usando la media del target, con smoothing para evitar overfitting.



**Explicación:** Features polinomiales y splines capturan relaciones no lineales sin cambiar de modelo. Splines son más estables en los extremos.

---

## A08: Target Encoding para Categorías de Alta Cardinalidad

**Enunciado:** Implementar Target Encoding para codificar 100 categorías de producto usando la media del target, con smoothing para evitar overfitting.

```python
# Pista: calcula la media global y pondera con media por categoría

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

np.random.seed(42)
n = 5000
n_categorias = 100

# Datos
categorias = np.random.choice([f'CAT_{i}' for i in range(n_categorias)], n)
target = np.random.normal(0, 1, n)
# Algunas categorías tienen efecto real
efectos = {f'CAT_{i}': np.random.normal(0, 0.5) for i in range(20)}
for i, cat in enumerate(categorias):
    if cat in efectos:
        target[i] += efectos[cat]

df_te = pd.DataFrame({'categoria': categorias, 'target': target})

# Target Encoding con validación cruzada (evita leakage)
def target_encoding_cv(df, col, target_col, k=5, alpha=10):
    df = df.copy()
    global_mean = df[target_col].mean()
    df[f'{col}_encoded'] = global_mean
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(df):
        train = df.iloc[train_idx]
        val = df.iloc[val_idx]
        
        # Media por categoría en train
        cat_means = train.groupby(col)[target_col].mean()
        cat_counts = train.groupby(col)[target_col].count()
        
        # Smoothing
        smoothed = (cat_means * cat_counts + global_mean * alpha) / (cat_counts + alpha)
        
        # Asignar a validación
        df.loc[val_idx, f'{col}_encoded'] = val[col].map(smoothed).fillna(global_mean)
    
    return df

df_encoded = target_encoding_cv(df_te, 'categoria', 'target', alpha=10)
print(f"Target Encoding completado para {n_categorias} categorías")
print(f"Correlación encoding vs target: {df_encoded['categoria_encoded'].corr(df_encoded['target']):.4f}")
print(f"Ejemplos:")
print(df_encoded.head(10).to_string())
```

**Explicación:** Target Encoding reemplaza cada categoría por la media del target en esa categoría. Smoothing evita overfitting en categorías con pocas muestras.

---

## A09: Feature Selection con Boruta

**Enunciado:** Implementar el algoritmo Boruta para seleccionar features relevantes comparando importancia de features originales vs versiones shuffladas.



**Explicación:** Target Encoding reemplaza cada categoría por la media del target en esa categoría. Smoothing evita overfitting en categorías con pocas muestras.

---

## A09: Feature Selection con Boruta

**Enunciado:** Implementar el algoritmo Boruta para seleccionar features relevantes comparando importancia de features originales vs versiones shuffladas.

```python
# Pista: crea copias shuffladas de cada feature, entrena RF, 
# y selecciona features con importancia > max(shadow)

from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd

def boruta_selection(X, y, max_iter=20, threshold='mean'):
    n_features = X.shape[1]
    shadow_names = [f'shadow_{i}' for i in range(n_features)]
    
    X_boruta = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(n_features)])
    
    selected = []
    rejected = []
    
    for iteration in range(max_iter):
        # Crear shadow features (shuffled)
        X_shadow = X_boruta.apply(np.random.permutation)
        X_shadow.columns = shadow_names
        
        X_combined = pd.concat([X_boruta, X_shadow], axis=1)
        
        # Entrenar RF
        rf = RandomForestRegressor(n_estimators=100, random_state=42 + iteration)
        rf.fit(X_combined, y)
        
        # Importancias
        real_importances = rf.feature_importances_[:n_features]
        shadow_importances = rf.feature_importances_[n_features:]
        
        max_shadow = shadow_importances.max()
        
        for i, imp in enumerate(real_importances):
            feat_name = f'feat_{i}'
            if feat_name in selected or feat_name in rejected:
                continue
            if imp > max_shadow:
                selected.append(feat_name)
            elif imp < shadow_importances.mean():
                rejected.append(feat_name)
        
        if len(selected) + len(rejected) >= n_features:
            break
    
    return selected, rejected

# Datos con features relevantes e irrelevantes
np.random.seed(42)
n = 1000
p = 20
X_bor = np.random.randn(n, p)
y_bor = X_bor[:, 0] + X_bor[:, 1] * 2 - X_bor[:, 2] + np.random.randn(n) * 0.3

selected, rejected = boruta_selection(X_bor, y_bor, max_iter=10)
print(f"Features seleccionadas: {selected}")
print(f"Features rechazadas: {rejected}")
print(f"Relevantes verdaderas: 0, 1, 2")
```

**Explicación:** Boruta compara cada feature real contra versiones aleatorizadas. Features con importancia consistentemente mayor que las sombra se consideran relevantes.

---

## A10: Feature Engineering Automático con Featuretools

**Enunciado:** Usar Featuretools para generar automáticamente features temporales (media móvil, tendencia, etc.) a partir de datos transaccionales con múltiples entidades.



**Explicación:** Boruta compara cada feature real contra versiones aleatorizadas. Features con importancia consistentemente mayor que las sombra se consideran relevantes.

---

## A10: Feature Engineering Automático con Featuretools

**Enunciado:** Usar Featuretools para generar automáticamente features temporales (media móvil, tendencia, etc.) a partir de datos transaccionales con múltiples entidades.

```python
# Pista: define EntitySet con relaciones entre tablas

# Featuretools requiere instalación: pip install featuretools
# Simulación del concepto (implementación manual del DFS)

import pandas as pd
import numpy as np

# Datos relacionales simulados
np.random.seed(42)
clientes = pd.DataFrame({
    'cliente_id': range(100),
    'antiguedad': np.random.randint(1, 60, 100)
})

compras = pd.DataFrame({
    'cliente_id': np.random.choice(range(100), 2000),
    'fecha': pd.date_range('2024-01-01', periods=2000, freq='H'),
    'monto': np.random.exponential(100, 2000),
    'producto_id': np.random.choice(range(50), 2000)
})

# Feature engineering manual equivalente
compras_agg = compras.groupby('cliente_id').agg({
    'monto': ['sum', 'mean', 'std', 'count'],
    'fecha': ['max', 'min']
})
compras_agg.columns = ['monto_total', 'monto_promedio', 'monto_std', 'num_compras', 'ultima_compra', 'primera_compra']
compras_agg = compras_agg.reset_index()
compras_agg['recencia_dias'] = (pd.Timestamp.now() - compras_agg['ultima_compra']).dt.total_seconds() / (3600*24)
compras_agg['frecuencia_semanal'] = compras_agg['num_compras'] / clientes['antiguedad'].mean() * 4

features_final = clientes.merge(compras_agg, on='cliente_id', how='left').fillna(0)
print(f"Features generadas: {features_final.shape[1]-1}")
print(features_final.head())
```

**Explicación:** La ingeniería de features automatizada extrae agregaciones, tendencias y patrones de datos relacionales, reduciendo el trabajo manual.

---

## A11: Clasificación de Texto con TF-IDF + Naive Bayes

**Enunciado:** Clasificar reseñas de productos en positivas/negativas usando TF-IDF y MultinomialNB. Evaluar con accuracy y matriz de confusión.



**Explicación:** La ingeniería de features automatizada extrae agregaciones, tendencias y patrones de datos relacionales, reduciendo el trabajo manual.

---

## A11: Clasificación de Texto con TF-IDF + Naive Bayes

**Enunciado:** Clasificar reseñas de productos en positivas/negativas usando TF-IDF y MultinomialNB. Evaluar con accuracy y matriz de confusión.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
n_reviews = 2000

# Generar reseñas sintéticas
positivas = [
    "excelente producto muy recomendable",
    "me encanta la calidad y el precio",
    "perfecto superó mis expectativas",
    "muy bueno lo recomiendo ampliamente",
    "gran calidad precio inmejorable"
]
negativas = [
    "pésimo producto no funciona bien",
    "mala calidad se rompió rápido",
    "terrible experiencia no lo compren",
    "decepcionante no sirve para nada",
    "horrible pérdida de dinero"
]

reviews = []
labels = []
for _ in range(n_reviews // 2):
    review = np.random.choice(positivas) + " " + np.random.choice(positivas)
    reviews.append(review)
    labels.append(1)
for _ in range(n_reviews // 2):
    review = np.random.choice(negativas) + " " + np.random.choice(negativas)
    reviews.append(review)
    labels.append(0)

# TF-IDF
tfidf = TfidfVectorizer(max_features=200, stop_words='english')
X_tfidf = tfidf.fit_transform(reviews)
X_tr, X_te, y_tr, y_te = train_test_split(X_tfidf, labels, test_size=0.2, random_state=42)

# Naive Bayes
nb = MultinomialNB()
nb.fit(X_tr, y_tr)
y_pred = nb.predict(X_te)

print("Classification Report:")
print(classification_report(y_te, y_pred, target_names=['Negativa', 'Positiva']))

# Matriz de confusión
cm = confusion_matrix(y_te, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negativa', 'Positiva'],
            yticklabels=['Negativa', 'Positiva'])
plt.title('Matriz de Confusión - TF-IDF + Naive Bayes')
plt.tight_layout()
plt.savefig('img/tfidf_nb_confusion.png', dpi=150)
plt.show()
```

**Explicación:** TF-IDF convierte texto en vectores numéricos. MultinomialNB asume distribución multinomial y funciona bien para clasificación de texto con vocabularios grandes.

---

## A12: Word2Vec con Gensim para Embeddings de Productos

**Enunciado:** Entrenar Word2Vec sobre descripciones de productos y visualizar embeddings con PCA. Encontrar palabras similares a "laptop".



**Explicación:** TF-IDF convierte texto en vectores numéricos. MultinomialNB asume distribución multinomial y funciona bien para clasificación de texto con vocabularios grandes.

---

## A12: Word2Vec con Gensim para Embeddings de Productos

**Enunciado:** Entrenar Word2Vec sobre descripciones de productos y visualizar embeddings con PCA. Encontrar palabras similares a "laptop".

```python
# Pista: gensim.models.Word2Vec, entrena con oraciones tokenizadas

from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Descripciones simuladas
descripciones = [
    "laptop profesional con procesador potente",
    "laptop gaming con grafica dedicada",
    "tablet ligera para viajes",
    "smartphone camara alta resolucion",
    "auricular inalambrico cancelacion ruido",
    "monitor ultra wide profesional",
    "teclado mecanico retroiluminado gamer",
    "mouse ergonomico inalambrico recargable",
    "cargador rapido universal usb",
    "laptop empresarial seguridad avanzada",
    "computadora escritorio torre gaming",
    "disco duro solido externo 1tb"
]

# Tokenizar
oraciones = [desc.lower().split() for desc in descripciones]

# Entrenar Word2Vec
w2v = Word2Vec(sentences=oraciones, vector_size=50, window=3, min_count=1, epochs=100)

# Palabras similares a "laptop"
similares = w2v.wv.most_similar('laptop', topn=5)
print("Palabras similares a 'laptop':")
for word, score in similares:
    print(f"  {word}: {score:.4f}")

# Visualizar embeddings (PCA 2D)
words = list(w2v.wv.index_to_key)
vectors = w2v.wv[words]
pca = PCA(n_components=2)
vectors_pca = pca.fit_transform(vectors)

plt.figure(figsize=(10, 8))
plt.scatter(vectors_pca[:, 0], vectors_pca[:, 1], alpha=0.6)
for i, word in enumerate(words):
    plt.annotate(word, (vectors_pca[i, 0], vectors_pca[i, 1]), fontsize=9)
plt.title('Word2Vec Embeddings (PCA 2D)')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('img/word2vec_pca.png', dpi=150)
plt.show()
```

**Explicación:** Word2Vec aprende representaciones vectoriales densas donde palabras semánticamente similares tienen vectores cercanos.

---

## A13: Embeddings con GloVe Pre-entrenado

**Enunciado:** Cargar embeddings GloVe pre-entrenados y usarlos como capa de embedding congelada para clasificación de textos cortos.



**Explicación:** Word2Vec aprende representaciones vectoriales densas donde palabras semánticamente similares tienen vectores cercanos.

---

## A13: Embeddings con GloVe Pre-entrenado

**Enunciado:** Cargar embeddings GloVe pre-entrenados y usarlos como capa de embedding congelada para clasificación de textos cortos.

```python
# Pista: descarga glove.6B.50d.txt, construye embedding matrix, 
# usa en capa Embedding con trainable=False

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense

# Simular GloVe (diccionario pequeño)
glove_embeddings = {
    'laptop': np.random.randn(50),
    'bueno': np.random.randn(50),
    'malo': np.random.randn(50),
    'producto': np.random.randn(50),
    'calidad': np.random.randn(50),
    'precio': np.random.randn(50),
    'excelente': np.random.randn(50),
    'pesimo': np.random.randn(50),
}

# Textos
textos = [
    "laptop excelente calidad precio",
    "producto malo pesimo calidad",
    "bueno precio producto excelente",
    "pesimo malo no funciona"
]
labels = [1, 0, 1, 0]

# Tokenizer
tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(textos)
seqs = tokenizer.texts_to_sequences(textos)
padded = pad_sequences(seqs, maxlen=5, padding='post')

# Embedding matrix
vocab_size = min(100, len(tokenizer.word_index) + 1)
embedding_dim = 50
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    if i < vocab_size:
        embedding_vector = glove_embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# Modelo con GloVe congelado
model_glove = Sequential([
    Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], 
              input_length=5, trainable=False),
    GlobalAveragePooling1D(),
    Dense(1, activation='sigmoid')
])

model_glove.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("Modelo con GloVe embeddings pre-entrenados (congelados):")
model_glove.summary()

# Entrenar (pocos datos solo para demostración)
model_glove.fit(padded, np.array(labels), epochs=10, verbose=0)
print(f"Accuracy: {model_glove.evaluate(padded, labels, verbose=0)[1]:.2%}")
```

**Explicación:** Embeddings pre-entrenados (GloVe) capturan semántica general del lenguaje. Congelarlos evita que se desajusten con datasets pequeños.

---

## A14: Modelo Secuencial con LSTM para Clasificación de Texto

**Enunciado:** Construir un modelo LSTM para clasificar reseñas de productos en positivas/negativas, comparando con GRU.



**Explicación:** Embeddings pre-entrenados (GloVe) capturan semántica general del lenguaje. Congelarlos evita que se desajusten con datasets pequeños.

---

## A14: Modelo Secuencial con LSTM para Clasificación de Texto

**Enunciado:** Construir un modelo LSTM para clasificar reseñas de productos en positivas/negativas, comparando con GRU.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

np.random.seed(42)
n = 1000
positivas = [
    "excelente producto calidad precio me encanta",
    "muy bueno funciona perfecto recomendable",
    "gran compra supera expectativas calidad"
]
negativas = [
    "pesimo malo no funciona calidad pesima",
    "terrible producto no sirve decepcionante",
    "horrible pesimo no comprar basura"
]

textos_nlp = []
labels_nlp = []
for _ in range(n // 2):
    textos_nlp.append(np.random.choice(positivas) + " " + np.random.choice(positivas))
    labels_nlp.append(1)
for _ in range(n // 2):
    textos_nlp.append(np.random.choice(negativas) + " " + np.random.choice(negativas))
    labels_nlp.append(0)

# Tokenización
tokenizer = Tokenizer(num_words=500, oov_token='<OOV>')
tokenizer.fit_on_texts(textos_nlp)
seqs = tokenizer.texts_to_sequences(textos_nlp)
padded = pad_sequences(seqs, maxlen=20, padding='post', truncating='post')

# Dividir
split = int(len(padded) * 0.8)
X_train_nlp, X_test_nlp = padded[:split], padded[split:]
y_train_nlp, y_test_nlp = np.array(labels_nlp[:split]), np.array(labels_nlp[split:])

# Modelo LSTM
model_lstm_nlp = Sequential([
    Embedding(500, 32, input_length=20),
    LSTM(32, return_sequences=False),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model_lstm_nlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_lstm_nlp.fit(X_train_nlp, y_train_nlp, validation_split=0.2, epochs=20, verbose=0)
lstm_acc = model_lstm_nlp.evaluate(X_test_nlp, y_test_nlp, verbose=0)[1]

# Modelo GRU
model_gru = Sequential([
    Embedding(500, 32, input_length=20),
    GRU(32, return_sequences=False),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model_gru.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_gru.fit(X_train_nlp, y_train_nlp, validation_split=0.2, epochs=20, verbose=0)
gru_acc = model_gru.evaluate(X_test_nlp, y_test_nlp, verbose=0)[1]

print(f"LSTM Accuracy: {lstm_acc:.4f}")
print(f"GRU Accuracy: {gru_acc:.4f}")
print(f"Diferencia: {lstm_acc - gru_acc:+.4f}")
```

**Explicación:** LSTM y GRU capturan dependencias secuenciales en texto. GRU tiene menos parámetros (más rápido) con desempeño similar.

---

## A15-19: Ejercicios con TensorFlow

**A15 - Custom Training Loop:** Implementar un training loop personalizado con tf.GradientTape para regresión lineal.



**Explicación:** LSTM y GRU capturan dependencias secuenciales en texto. GRU tiene menos parámetros (más rápido) con desempeño similar.

---

## A15-19: Ejercicios con TensorFlow

**A15 - Custom Training Loop:** Implementar un training loop personalizado con tf.GradientTape para regresión lineal.

```python
import tensorflow as tf

# Datos
X_tf = tf.random.normal((1000, 1))
y_tf = 3 * X_tf + 2 + tf.random.normal((1000, 1)) * 0.1

w = tf.Variable(tf.random.normal((1, 1)))
b = tf.Variable(0.0)

optimizer = tf.optimizers.SGD(learning_rate=0.1)

for epoch in range(100):
    with tf.GradientTape() as tape:
        y_pred = X_tf @ w + b
        loss = tf.reduce_mean((y_pred - y_tf) ** 2)
    grads = tape.gradient(loss, [w, b])
    optimizer.apply_gradients(zip(grads, [w, b]))
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss={loss.numpy():.4f}, w={w.numpy()[0,0]:.2f}, b={b.numpy():.2f}")

print(f"Final: w={w.numpy()[0,0]:.2f} (esperado 3), b={b.numpy():.2f} (esperado 2)")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*A15-19: Ejercicios con TensorFlow.*

1. Datos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**A16 - Functional API vs Sequential:** Construir el mismo modelo con ambas APIs.

```python
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense

# Sequential API
seq_model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(32, activation='relu'),
    Dense(1)
])

# Functional API
inputs = Input(shape=(10,))
x = Dense(64, activation='relu')(inputs)
x = Dense(32, activation='relu')(x)
outputs = Dense(1)(x)
func_model = Model(inputs=inputs, outputs=outputs)

print("Sequential params:", seq_model.count_params())
print("Functional params:", func_model.count_params())
print("Functional API permite arquitecturas más complejas (multi-input, multi-output)")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Sequential API
2. Functional API

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**A17 - Custom Callback:** Crear un callback personalizado que detenga entrenamiento si la loss no mejora y guarde el mejor modelo.

```python
from tensorflow.keras.callbacks import Callback

class EarlyStoppingWithSave(Callback):
    def __init__(self, patience=10, filepath='best_model.h5'):
        super().__init__()
        self.patience = patience
        self.filepath = filepath
        self.best_weights = None
        self.best_loss = np.inf
        self.wait = 0
    
    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_weights = self.model.get_weights()
            self.wait = 0
            print(f"  ✓ Nueva mejor loss: {val_loss:.4f}")
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.set_weights(self.best_weights)
                self.model.save(self.filepath)
                print(f"\n⏹ Early stopping en epoch {epoch}")
                self.model.stop_training = True

# Uso: model.fit(X, y, callbacks=[EarlyStoppingWithSave(patience=10)])
print("Callback personalizado creado. Usar: model.fit(X, y, callbacks=[EarlyStoppingWithSave(...)])")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Uso: model.fit(X, y, callbacks=[EarlyStoppingWithSave(patience=10)])

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**A18 - Model Subclassing:** Implementar un modelo con capas personalizadas usando herencia de tf.keras.Model.

```python
class CustomMLP(tf.keras.Model):
    def __init__(self, hidden_units=[64, 32], dropout_rate=0.2):
        super().__init__()
        self.dense1 = Dense(hidden_units[0], activation='relu')
        self.dropout1 = Dropout(dropout_rate)
        self.dense2 = Dense(hidden_units[1], activation='relu')
        self.dropout2 = Dropout(dropout_rate)
        self.output_layer = Dense(1)
    
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        return self.output_layer(x)

model_sub = CustomMLP()
model_sub.compile(optimizer='adam', loss='mse')
model_sub.build((None, 10))
model_sub.summary()
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**A19 - TensorBoard + Hyperparameter Tuning:** Configurar logging con TensorBoard para monitorear entrenamiento y ajustar hiperparámetros.

```python
import datetime
from tensorflow.keras.callbacks import TensorBoard

# Configurar TensorBoard
log_dir = f"logs/fit/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True)

# Modelo simple
model_tb = Sequential([
    Dense(64, activation='relu', input_shape=(20,)),
    Dense(1)
])
model_tb.compile(optimizer='adam', loss='mse')

# Datos
X_tb = np.random.randn(500, 20)
y_tb = X_tb[:, 0] + X_tb[:, 1] * 2 + np.random.randn(500) * 0.1

# Entrenar con TensorBoard
# model_tb.fit(X_tb, y_tb, epochs=50, validation_split=0.2, callbacks=[tensorboard_cb])
print(f"TensorBoard logs en: {log_dir}")
print("Ejecutar: tensorboard --logdir logs/fit")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Configurar TensorBoard
2. Modelo simple
3. Datos
4. Entrenar con TensorBoard
5. model_tb.fit(X_tb, y_tb, epochs=50, validation_split=0.2, callbacks=[tensorboard_cb])

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## A20-24: Ejercicios con PyTorch

**A20 - Regresión Lineal con PyTorch:**

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Datos
np.random.seed(42)
X_pt = torch.tensor(np.random.randn(500, 1), dtype=torch.float32)
y_pt = torch.tensor(3 * X_pt.numpy() + 2 + np.random.randn(500, 1) * 0.1, dtype=torch.float32)

class LinearRegressionTorch(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

model_lr = LinearRegressionTorch()
criterion = nn.MSELoss()
optimizer = optim.SGD(model_lr.parameters(), lr=0.01)

for epoch in range(200):
    optimizer.zero_grad()
    outputs = model_lr(X_pt)
    loss = criterion(outputs, y_pt)
    loss.backward()
    optimizer.step()

w, b = list(model_lr.parameters())
print(f"PyTorch Regresión: w={w.item():.2f}, b={b.item():.2f} (esperado: w=3, b=2)")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*A20-24: Ejercicios con PyTorch.*

1. Datos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**A21 - Red Neuronal con PyTorch para Clasificación:**

```python
class ChurnClassifier(nn.Module):
    def __init__(self, input_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

model_churn = ChurnClassifier(10)
criterion = nn.BCELoss()
optimizer = optim.Adam(model_churn.parameters(), lr=0.001)

# Datos simulados
X_churn = torch.randn(1000, 10)
y_churn = (X_churn[:, 0] + X_churn[:, 1] > 0).float().unsqueeze(1)

for epoch in range(50):
    optimizer.zero_grad()
    outputs = model_churn(X_churn)
    loss = criterion(outputs, y_churn)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    preds = (model_churn(X_churn) > 0.5).float()
    acc = (preds == y_churn).float().mean()
print(f"PyTorch Churn Classifier - Accuracy: {acc:.2%}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Datos simulados

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**A22 - DataLoader y Dataset Personalizado:**

```python
from torch.utils.data import Dataset, DataLoader
import torch

class VentasDataset(Dataset):
    def __init__(self, n_samples=1000, n_features=10):
        self.X = torch.randn(n_samples, n_features)
        self.y = torch.randn(n_samples, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = VentasDataset(2000, 10)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

print(f"Dataset: {len(dataset)} muestras")
print(f"DataLoader: {len(dataloader)} batches de 64")
for X_batch, y_batch in dataloader:
    print(f"Batch shape: X={X_batch.shape}, y={y_batch.shape}")
    break
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. `from torch.utils.data import Dataset, DataLoader` — Importa las librerías necesarias para el análisis.
2. `import torch` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**A23 - Guardar y Cargar Modelos PyTorch:**

```python
# Guardar
model = ChurnClassifier(10)
torch.save(model.state_dict(), 'modelo_churn.pth')
print("Modelo guardado: modelo_churn.pth")

# Cargar
model_loaded = ChurnClassifier(10)
model_loaded.load_state_dict(torch.load('modelo_churn.pth'))
model_loaded.eval()
print("Modelo cargado exitosamente")

# Guardar checkpoint con optimizer
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': 50,
    'loss': 0.05
}
torch.save(checkpoint, 'checkpoint_epoch50.pth')
print("Checkpoint guardado con estado del optimizer")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Guardar
2. Cargar
3. Guardar checkpoint con optimizer

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**A24 - GPU/CPU Device Management:**

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# Mover modelo a GPU
model_gpu = ChurnClassifier(10).to(device)
X_gpu = torch.randn(100, 10).to(device)
y_gpu = torch.randn(100, 1).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model_gpu.parameters())

for epoch in range(10):
    optimizer.zero_grad()
    outputs = model_gpu(X_gpu)
    loss = criterion(outputs, y_gpu)
    loss.backward()
    optimizer.step()

print(f"Entrenamiento en {device} completado. Loss final: {loss.item():.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Mover modelo a GPU

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## A25-26: CNN, RNN, LSTM

**A25 - CNN 1D para Clasificación de Señales de Ventas:**

```python
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten

# Datos: secuencias de ventas de 60 días
np.random.seed(42)
n_cnn = 2000
X_cnn = np.random.randn(n_cnn, 60, 1)
y_cnn = (X_cnn.mean(axis=(1,2)) > 0).astype(int)

model_cnn = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(60, 1)),
    MaxPooling1D(2),
    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(2),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_cnn.fit(X_cnn, y_cnn, validation_split=0.2, epochs=10, verbose=0)
print(f"CNN 1D Accuracy: {model_cnn.evaluate(X_cnn, y_cnn, verbose=0)[1]:.4f}")
print(f"Total parámetros: {model_cnn.count_params():,}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*A25-26: CNN, RNN, LSTM.*

1. Datos: secuencias de ventas de 60 días

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**A26 - LSTM Multicapa para Predicción de Series:**

```python
# LSTM con 3 capas apiladas para predicción de demanda
model_lstm_stack = Sequential([
    LSTM(64, return_sequences=True, input_shape=(30, 1)),
    Dropout(0.2),
    LSTM(32, return_sequences=True),
    Dropout(0.2),
    LSTM(16, return_sequences=False),
    Dense(1)
])

model_lstm_stack.compile(optimizer='adam', loss='mse')
print("LSTM Stacked (3 capas):")
model_lstm_stack.summary()

# Datos de ejemplo
X_lstm_stack = np.random.randn(500, 30, 1)
y_lstm_stack = np.random.randn(500, 1)
model_lstm_stack.fit(X_lstm_stack, y_lstm_stack, epochs=5, verbose=0)
print("LSTM multicapa entrenado (demostración)")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. LSTM con 3 capas apiladas para predicción de demanda
2. Datos de ejemplo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## A27-28: Autoencoders y Regularización

**A27 - Autoencoder para Reducción de Ruido (Denoising AE):**

```python
# Denoising Autoencoder
np.random.seed(42)
n_ae = 1000
X_clean = np.random.randn(n_ae, 20)
X_noisy = X_clean + np.random.randn(n_ae, 20) * 0.5

# Autoencoder
input_ae = Input(shape=(20,))
encoded = Dense(10, activation='relu')(input_ae)
encoded = Dense(5, activation='relu')(encoded)  # bottleneck
decoded = Dense(10, activation='relu')(encoded)
decoded = Dense(20, activation='linear')(decoded)

dae = Model(input_ae, decoded)
dae.compile(optimizer='adam', loss='mse')

dae.fit(X_noisy, X_clean, epochs=30, batch_size=32, validation_split=0.2, verbose=0)

# Evaluar denoising
X_test_noisy = X_noisy[:10]
X_denoised = dae.predict(X_test_noisy, verbose=0)
mse_before = np.mean((X_test_noisy - X_clean[:10])**2)
mse_after = np.mean((X_denoised - X_clean[:10])**2)
print(f"MSE antes del denoising: {mse_before:.4f}")
print(f"MSE después del denoising: {mse_after:.4f}")
print(f"Mejora: {(1 - mse_after/mse_before)*100:.1f}%")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*A27-28: Autoencoders y Regularización.*

1. Denoising Autoencoder
2. Autoencoder
3. Evaluar denoising

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**A28 - Regularización: L1, L2, Dropout, Batch Normalization:**

```python
from tensorflow.keras.regularizers import l1, l2, l1_l2

# Comparación de regularizadores
def build_model(regularizer_type='none'):
    reg_dict = {
        'none': None,
        'l1': l1(0.01),
        'l2': l2(0.01),
        'l1_l2': l1_l2(l1=0.01, l2=0.01)
    }
    
    model = Sequential([
        Dense(128, activation='relu', kernel_regularizer=reg_dict[regularizer_type], input_shape=(50,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=reg_dict[regularizer_type]),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Comparar
X_reg = np.random.randn(1000, 50)
y_reg = np.random.randn(1000, 1)

for reg in ['none', 'l1', 'l2', 'l1_l2']:
    m = build_model(reg)
    h = m.fit(X_reg, y_reg, validation_split=0.2, epochs=20, verbose=0)
    print(f"{reg:>6} | Train loss: {h.history['loss'][-1]:.4f} | Val loss: {h.history['val_loss'][-1]:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Comparación de regularizadores
2. Comparar

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## A29-30: Transfer Learning e Integradores

**A29 - Transfer Learning con Modelo Pre-entrenado:**

```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D

# Cargar modelo pre-entrenado (sin top)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # congelar

# Añadir capas personalizadas
model_tl = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(5, activation='softmax')  # 5 categorías de productos
])

model_tl.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Transfer Learning con MobileNetV2:")
model_tl.summary()
print(f"Total parámetros: {model_tl.count_params():,}")
print(f"Entrenables (solo top): {sum(1 for l in model_tl.layers[-3:] for _ in l.trainable_variables)}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*A29-30: Transfer Learning e Integradores.*

1. Cargar modelo pre-entrenado (sin top)
2. Añadir capas personalizadas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**A30 - Pipeline Integrador Completo (End-to-End):**

```python
# Pipeline completo: datos → features → modelos → ensemble → evaluación → recomendación

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
import numpy as np

class PipelineIntegrador:
    """Pipeline completo de ML para ventas"""
    
    def __init__(self):
        self.models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'ridge': Ridge(alpha=1.0)
        }
        self.meta_model = GradientBoostingRegressor(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()
    
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        
        # OOB predictions para meta-modelo
        meta_features = np.column_stack([
            cross_val_predict(model, X_scaled, y, cv=5)
            for model in self.models.values()
        ])
        
        # Entrenar modelos base
        for model in self.models.values():
            model.fit(X_scaled, y)
        
        # Entrenar meta-modelo
        self.meta_model.fit(meta_features, y)
        return self
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        meta_features = np.column_stack([
            model.predict(X_scaled)
            for model in self.models.values()
        ])
        return self.meta_model.predict(meta_features)
    
    def get_feature_importance(self):
        return self.models['rf'].feature_importances_

# Demostración
np.random.seed(42)
X_pipe = np.random.randn(1000, 10)
y_pipe = X_pipe[:, 0] * 2 + X_pipe[:, 1] + np.random.randn(1000) * 0.2

pipeline = PipelineIntegrador()
pipeline.fit(X_pipe[:800], y_pipe[:800])
preds = pipeline.predict(X_pipe[800:])
print(f"Pipeline Integrador - Test R2: {np.corrcoef(preds, y_pipe[800:])[0,1]**2:.4f}")
importances = pipeline.get_feature_importance()
print(f"Top 3 features importantes: {np.argsort(importances)[-3:][::-1]}")
```

**Explicación:** Pipeline integrador combina múltiples modelos base con stacking, escalado automático y evaluación cross-validada. Listo para producción.

---

## 5 Ejercicios Extra

**E31:** Implementar un sistema de recomendación híbrido (content-based + collaborative filtering) usando SVD y TF-IDF, evaluado con RMSE en hold-out.

**E32:** Construir un agente de Reinforcement Learning que optimice precios en tiempo real usando Q-Learning con estado (stock, demanda, precio competidor).

**E33:** Crear un modelo de detección de anomalías usando Variational Autoencoder (VAE) con pérdida KL + reconstruction, visualizando el espacio latente.

**E34:** Implementar un sistema de forecast distribuido con PySpark para 10,000 productos, usando RandomForest y ventanas temporales con spark.ml.

**E35:** Desplegar un modelo de clasificación de churn como API REST con FastAPI, dockerizado, con endpoint `/predict` y `/retrain` para actualización en caliente.

---

## Soluciones Rápidas (Referencia)

| Ejercicio | Técnica principal | Librería | Dificultad |
|-----------|-------------------|----------|------------|
| A01 | GridSearch + Pipeline | sklearn | ⭐⭐ |
| A05 | Regularización L1/L2 | sklearn | ⭐⭐ |
| A08 | Target Encoding | manual | ⭐⭐⭐ |
| A11 | TF-IDF + Naive Bayes | sklearn | ⭐⭐ |
| A12 | Word2Vec | gensim | ⭐⭐⭐ |
| A15 | Custom training loop | TensorFlow | ⭐⭐⭐ |
| A20 | Regresión lineal | PyTorch | ⭐⭐ |
| A25 | CNN 1D | TensorFlow | ⭐⭐⭐ |
| A27 | Denoising AE | TensorFlow | ⭐⭐⭐ |
| A30 | Pipeline integrador | sklearn | ⭐⭐⭐⭐ |

---

## Conclusiones

1. Los 30 ejercicios cubren todo el espectro del nivel avanzado: pipelines, ensemble, NLP, DL, autoencoders
2. Cada ejercicio es autónomo y reproducible con datos sintéticos
3. Se enfatiza el balance entre teoría y práctica aplicada a ventas/compras/inventarios
4. Los ejercicios extra (E31-E35) representan proyectos integradores de nivel profesional
5. Las soluciones en referencia permiten al estudiante verificar su progreso


**Explicación:** Pipeline integrador combina múltiples modelos base con stacking, escalado automático y evaluación cross-validada. Listo para producción.

---

## 5 Ejercicios Extra

**E31:** Implementar un sistema de recomendación híbrido (content-based + collaborative filtering) usando SVD y TF-IDF, evaluado con RMSE en hold-out.

**E32:** Construir un agente de Reinforcement Learning que optimice precios en tiempo real usando Q-Learning con estado (stock, demanda, precio competidor).

**E33:** Crear un modelo de detección de anomalías usando Variational Autoencoder (VAE) con pérdida KL + reconstruction, visualizando el espacio latente.

**E34:** Implementar un sistema de forecast distribuido con PySpark para 10,000 productos, usando RandomForest y ventanas temporales con spark.ml.

**E35:** Desplegar un modelo de clasificación de churn como API REST con FastAPI, dockerizado, con endpoint `/predict` y `/retrain` para actualización en caliente.

---

## Soluciones Rápidas (Referencia)

| Ejercicio | Técnica principal | Librería | Dificultad |
|-----------|-------------------|----------|------------|
| A01 | GridSearch + Pipeline | sklearn | ⭐⭐ |
| A05 | Regularización L1/L2 | sklearn | ⭐⭐ |
| A08 | Target Encoding | manual | ⭐⭐⭐ |
| A11 | TF-IDF + Naive Bayes | sklearn | ⭐⭐ |
| A12 | Word2Vec | gensim | ⭐⭐⭐ |
| A15 | Custom training loop | TensorFlow | ⭐⭐⭐ |
| A20 | Regresión lineal | PyTorch | ⭐⭐ |
| A25 | CNN 1D | TensorFlow | ⭐⭐⭐ |
| A27 | Denoising AE | TensorFlow | ⭐⭐⭐ |
| A30 | Pipeline integrador | sklearn | ⭐⭐⭐⭐ |

---

## Conclusiones

1. Los 30 ejercicios cubren todo el espectro del nivel avanzado: pipelines, ensemble, NLP, DL, autoencoders
2. Cada ejercicio es autónomo y reproducible con datos sintéticos
3. Se enfatiza el balance entre teoría y práctica aplicada a ventas/compras/inventarios
4. Los ejercicios extra (E31-E35) representan proyectos integradores de nivel profesional
5. Las soluciones en referencia permiten al estudiante verificar su progreso
