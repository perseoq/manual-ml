# E23: Explicabilidad de Modelos — SHAP, LIME, PDP, InterpretML en Ventas

## Objetivo
Dominar técnicas de explicabilidad (SHAP, LIME, PDP, ICE, ALE, InterpretML) para interpretar predicciones de modelos ML en ventas, compras e inventarios, generando confianza en stakeholders.

---

## 1. Fundamentos Teóricos

### 1.1 SHAP (SHapley Additive exPlanations)
Basado en teoría de juegos cooperativos: cada feature es un "jugador" que contribuye a la predicción.

```python
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*1.1 SHAP (SHapley Additive exPlanations).*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Tipos de Explainer**:
- **TreeExplainer**: Para modelos basados en árboles (RF, XGBoost, LightGBM)
- **KernelExplainer**: Para cualquier modelo (genérico, basado en muestreo)
- **DeepExplainer**: Para redes neuronales profundas
- **GradientExplainer**: Para modelos diferenciables
- **LinearExplainer**: Para modelos lineales (asume independencia de features)
- **PartitionExplainer**: Para modelos con estructura de partición

### 1.2 Visualizaciones SHAP
```python
shap.summary_plot(shap_values, X, plot_type='dot'/'bar'/'violin', max_display=20)
shap.dependence_plot('feature', shap_values, X, interaction_index='auto')
shap.force_plot(explainer.expected_value, shap_values[0], X.iloc[0])
shap.waterfall_plot(shap.Explanation(values, base_values, data, feature_names))
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*1.2 Visualizaciones SHAP.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### 1.3 LIME (Local Interpretable Model-agnostic Explanations)
Explica predicciones individuales aproximando el modelo localmente con un modelo interpretable.

```python
explainer = lime_tabular.LimeTabularExplainer(X_train, feature_names=features)
exp = explainer.explain_instance(X_test[i], model.predict_proba, num_features=5)
exp.show_in_notebook()
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*1.3 LIME (Local Interpretable Model-agnostic Explanations).*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### 1.4 Eli5
Inspección de pesos y predicciones para modelos sklearn, XGBoost, LightGBM, CatBoost.

```python
eli5.explain_weights(model)
eli5.explain_prediction(model, X_test[i])
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*1.4 Eli5.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### 1.5 Partial Dependence Plots (PDP)
Muestra el efecto marginal de una o dos features en la predicción.

```python
from sklearn.inspection import plot_partial_dependence
plot_partial_dependence(model, X, features, kind='average'/'individual'/'both')
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*1.5 Partial Dependence Plots (PDP).*

1. `from sklearn.inspection import plot_partial_dependence` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### 1.6 ICE (Individual Conditional Expectation)
Muestra el efecto de una feature para cada instancia individual (heterogeneidad de efectos).

### 1.7 ALE (Accumulated Local Effects)
Alternativa no sesgada a PDP cuando hay correlación entre features.

### 1.8 InterpretML
- **Explainable Boosting Classifier (EBC)**: Modelo intrinsicamente interpretable (glassbox)
- **Show in dashboard**: Visualización web interactiva

---

## 2. Ejemplos Prácticos

### Ejemplo 1: TreeExplainer — SHAP para RandomForest (explicar predicción de demanda)

```python
import shap
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Datos sintéticos de ventas
np.random.seed(42)
n = 2000
X_ventas = pd.DataFrame({
    'precio': np.random.uniform(10, 200, n),
    'descuento': np.random.uniform(0, 0.5, n),
    'inventario': np.random.randint(0, 1000, n),
    'dia_semana': np.random.randint(0, 7, n),
    'rating': np.random.uniform(1, 5, n),
    'tiempo_envio': np.random.randint(1, 10, n)
})
y_demanda = (100 - X_ventas['precio'] * 0.3 + X_ventas['descuento'] * 50
             + X_ventas['inventario'] * 0.1 + X_ventas['rating'] * 10
             - X_ventas['tiempo_envio'] * 2 + np.random.randn(n) * 10)

X_train, X_test = train_test_split(X_ventas, test_size=0.2, random_state=42)
y_train, y_test = train_test_split(y_demanda, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# SHAP TreeExplainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

print(f"Expected value (base): {explainer.expected_value:.2f}")
print(f"SHAP values shape: {shap_values.shape}")
print(f"Primera predicción: {model.predict(X_test[:1])[0]:.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: TreeExplainer — SHAP para RandomForest (explicar predicción de demanda).*

1. Datos sintéticos de ventas
2. SHAP TreeExplainer

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: summary_plot — Importancia global de features

```python
import matplotlib.pyplot as plt

shap.summary_plot(shap_values, X_test, max_display=10, show=False)
plt.title('SHAP Summary Plot - Importancia de Features')
plt.tight_layout()
plt.show()
print("Interpretación:")
print("  - Rojo: valor alto de feature")
print("  - Azul: valor bajo de feature")
print("  - Eje X: impacto en predicción (positivo = aumenta demanda)")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: summary_plot — Importancia global de features.*

1. `import matplotlib.pyplot as plt` — Importa las librerías necesarias para el análisis.
2. `print("Interpretación:")` — Muestra el resultado por pantalla.
3. `print("  - Rojo: valor alto de feature")` — Muestra el resultado por pantalla.
4. `print("  - Azul: valor bajo de feature")` — Muestra el resultado por pantalla.
5. `print("  - Eje X: impacto en predicción (positivo = aumenta demanda)")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: summary_plot type='bar' — Ranking de features

```python
shap.summary_plot(shap_values, X_test, plot_type='bar', max_display=10, show=False)
plt.title('SHAP Feature Importance (Bar)')
plt.tight_layout()
plt.show()

# Obtener importancia numérica
feature_importance = np.abs(shap_values).mean(axis=0)
features = X_test.columns
ranking = pd.DataFrame({'feature': features, 'importance': feature_importance}).sort_values('importance', ascending=False)
print("Ranking de features por importancia SHAP:")
print(ranking.to_string(index=False))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: summary_plot type='bar' — Ranking de features.*

1. Obtener importancia numérica

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: Dependence plot — Efecto de precio en predicción

```python
# Dependence plot para 'precio'
shap.dependence_plot('precio', shap_values, X_test, show=False)
plt.title('SHAP Dependence Plot: Efecto del Precio en Demanda')
plt.show()

print("Interpretación:")
print("  - A mayor precio, menor contribución a demanda (SHAP negativo)")
print("  - Relación aproximadamente lineal decreciente")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: Dependence plot — Efecto de precio en predicción.*

1. Dependence plot para 'precio'

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: Dependence plot con interacción — precio × categoría

```python
# Añadir categoría para demostrar interacción
X_test_con_cat = X_test.copy()
np.random.seed(42)
X_test_con_cat['categoria'] = np.random.choice(['A', 'B', 'C'], len(X_test_con_cat))

# Reentrenar modelo con categoría
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X_test_con_cat['categoria_enc'] = le.fit_transform(X_test_con_cat['categoria'])
X_interact = X_test_con_cat[['precio', 'descuento', 'inventario', 'dia_semana',
                              'rating', 'tiempo_envio', 'categoria_enc']]

model_interact = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model_interact.fit(X_interact, y_test)
explainer_interact = shap.TreeExplainer(model_interact)
shap_values_interact = explainer_interact.shap_values(X_interact)

# Dependence plot con interacción automática
shap.dependence_plot('precio', shap_values_interact, X_interact,
                     interaction_index='auto', show=False)
plt.title('Dependence Plot: Precio con Interacción Automática')
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 5: Dependence plot con interacción — precio × categoría.*

1. Añadir categoría para demostrar interacción
2. Reentrenar modelo con categoría
3. Dependence plot con interacción automática

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: Force plot — Explicar predicción individual

```python
# Explicar primera predicción
i = 0
print(f"Predicción para instancia {i}: {model.predict(X_test.iloc[[i]])[0]:.2f}")
print(f"Valor base (demanda promedio): {explainer.expected_value:.2f}")

# Force plot
shap.initjs()
force_plot = shap.force_plot(explainer.expected_value, shap_values[i, :],
                              X_test.iloc[i], matplotlib=True, show=False)
plt.title(f'Force Plot - Instancia {i}')
plt.tight_layout()
plt.show()

# Interpretación
contributions = pd.DataFrame({
    'feature': X_test.columns,
    'value': X_test.iloc[i].values,
    'shap_value': shap_values[i, :]
}).sort_values('shap_value', key=abs, ascending=False)
print("\nContribuciones ordenadas por magnitud:")
print(contributions.head().to_string(index=False))
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

*Ejemplo 6: Force plot — Explicar predicción individual.*

1. Explicar primera predicción
2. Force plot
3. Interpretación

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: Waterfall plot — Descomponer predicción en contribuciones

```python
# Crear Explanation object para waterfall plot
shap_waterfall = shap.Explanation(
    values=shap_values[1, :],
    base_values=explainer.expected_value,
    data=X_test.iloc[1].values,
    feature_names=X_test.columns.tolist()
)

shap.waterfall_plot(shap_waterfall, max_display=10, show=False)
plt.title('Waterfall Plot - Descomposición de Predicción')
plt.show()

print(f"Predicción: {model.predict(X_test.iloc[[1]])[0]:.2f}")
print(f"Base value (promedio): {explainer.expected_value:.2f}")
print("Cada barra muestra cómo cada feature contribuye al valor final")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: Waterfall plot — Descomponer predicción en contribuciones.*

1. Crear Explanation object para waterfall plot

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: LIME — Explicar predicción individual de clasificación

```python
import lime
import lime.lime_tabular
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Datos de clasificación binaria
X_clf, y_clf = make_classification(n_samples=500, n_features=10, n_informative=5,
                                    random_state=42)
feature_names = [f'feature_{i}' for i in range(10)]

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_clf, y_clf)

# LIME Explainer
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    X_clf, feature_names=feature_names, class_names=['No Compra', 'Compra'],
    mode='classification', random_state=42
)

# Explicar instancia
i = 10
exp = explainer_lime.explain_instance(X_clf[i], clf.predict_proba, num_features=5)
print(f"Predicción: {clf.predict(X_clf[i:i+1])[0]}")
print(f"Probabilidades: {clf.predict_proba(X_clf[i:i+1])[0]}")
print("\nFeatures más importantes (LIME):")
for feature, weight in exp.as_list():
    print(f"  {feature}: {weight:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: LIME — Explicar predicción individual de clasificación.*

1. Datos de clasificación binaria
2. LIME Explainer
3. Explicar instancia

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: LIME — Features que más contribuyen a positiva/negativa

```python
# Mostrar contribuciones positivas y negativas
print("Contribuciones a clase 'Compra':")
pos = [(f, w) for f, w in exp.as_list() if w > 0]
neg = [(f, w) for f, w in exp.as_list() if w < 0]

print("  Positivas (aumentan probabilidad de compra):")
for f, w in sorted(pos, key=lambda x: x[1], reverse=True):
    print(f"    + {f}: +{w:.4f}")

print("  Negativas (disminuyen probabilidad de compra):")
for f, w in sorted(neg, key=lambda x: x[1]):
    print(f"    - {f}: {w:.4f}")

# Visualizar
exp.show_in_notebook(show_table=True)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: LIME — Features que más contribuyen a positiva/negativa.*

1. Mostrar contribuciones positivas y negativas
2. Visualizar

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: Eli5 — explain_weights de modelo

```python
import eli5

# Para modelos sklearn lineales
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)

print("Pesos del modelo (eli5):")
eli5.show_weights(lr, feature_names=list(X_train.columns))

# Para XGBoost
import xgboost as xgb
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
print("\nImportancia de features (XGBoost):")
eli5.explain_weights(xgb_model, feature_names=list(X_train.columns))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: Eli5 — explain_weights de modelo.*

1. Para modelos sklearn lineales
2. Para XGBoost

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: Eli5 — explain_prediction para una instancia

```python
i = 5
print(f"Explicación de predicción para instancia {i}:")
print(f"Valor real: {y_test.iloc[i]:.2f}")
print(f"Predicción: {model.predict(X_test.iloc[[i]])[0]:.2f}")

# Solo sklearn lineal
print("\nExplicación con modelo lineal:")
eli5.explain_prediction(lr, X_test.iloc[i], feature_names=list(X_train.columns))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: Eli5 — explain_prediction para una instancia.*

1. Solo sklearn lineal

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: Partial Dependence Plot — Efecto marginal de feature

```python
from sklearn.inspection import PartialDependenceDisplay

# PDP para 'precio'
fig, ax = plt.subplots(figsize=(8, 5))
PartialDependenceDisplay.from_estimator(model, X_train, ['precio'], kind='average', ax=ax)
plt.title('Partial Dependence Plot - Efecto del Precio en Demanda')
plt.tight_layout(); plt.show()

# PDP para múltiples features
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
features_plot = ['precio', 'descuento', 'inventario', 'dia_semana', 'rating', 'tiempo_envio']
PartialDependenceDisplay.from_estimator(model, X_train, features_plot, kind='average',
                                        ax=axes.ravel())
plt.tight_layout(); plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 12: Partial Dependence Plot — Efecto marginal de feature.*

1. PDP para 'precio'
2. PDP para múltiples features

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: ICE — Efecto individual por instancia (heterogeneidad)

```python
# ICE plot (kind='individual')
fig, ax = plt.subplots(figsize=(10, 6))
PartialDependenceDisplay.from_estimator(
    model, X_train, ['precio'], kind='individual', ax=ax, subsample=50
)
plt.title('ICE Plot - Efecto Individual del Precio en Demanda (50 instancias)')
plt.tight_layout(); plt.show()

print("Interpretación:")
print("  - Cada línea es una instancia individual")
print("  - Muestra heterogeneidad: diferentes pendientes para diferentes productos")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: ICE — Efecto individual por instancia (heterogeneidad).*

1. ICE plot (kind='individual')

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: ALE — Alternativa no sesgada a PDP

```python
# ALE requiere instalación de PyALE
try:
    from PyALE import ale
    ale_eff = ale(X=X_train, model=model, feature=['precio'], grid_size=50)
    plt.title('ALE Plot - Efecto Acumulado Local del Precio')
    plt.show()
    print("ALE generado correctamente")
except ImportError:
    print("PyALE no instalado. Alternativa conceptual:")
    print("ALE es similar a PDP pero no sesgado cuando features están correlacionadas.")
    print("  - PDP: promedia predicciones marginales (puede incluir puntos imposibles)")
    print("  - ALE: usa diferencias condicionales (solo puntos reales)")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: ALE — Alternativa no sesgada a PDP.*

1. ALE requiere instalación de PyALE

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: InterpretML — Explainable Boosting Classifier (glassbox)

```python
try:
    from interpret.glassbox import ExplainableBoostingClassifier
    from interpret import show

    ebc = ExplainableBoostingClassifier()
    ebc.fit(X_clf, y_clf)

    # Interpretar globalmente
    ebc_global = ebc.explain_global()
    show(ebc_global)

    # Interpretar localmente
    ebc_local = ebc.explain_local(X_clf[:5], y_clf[:5])
    show(ebc_local)

    print("InterpretML: EBC entrenado y explicado")
    print(f"Precisión EBC: {ebc.score(X_clf, y_clf):.3f}")
except ImportError:
    print("InterpretML no instalado. Instalar: pip install interpret")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: InterpretML — Explainable Boosting Classifier (glassbox).*

1. Interpretar globalmente
2. Interpretar localmente

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 16: Comparar SHAP vs LIME vs PDP en misma predicción

```python
i = 0
print(f"{'='*60}")
print(f"COMPARACIÓN SHAP vs LIME vs PDP - Instancia {i}")
print(f"{'='*60}")
print(f"Predicción: {model.predict(X_test.iloc[[i]])[0]:.2f} (real: {y_test.iloc[i]:.2f})")
print()

# SHAP
print("--- SHAP ---")
shap_contrib = pd.DataFrame({
    'feature': X_test.columns,
    'value': X_test.iloc[i].values,
    'shap': shap_values[i]
}).sort_values('shap', key=abs, ascending=False)
print(shap_contrib.head(5).to_string(index=False))

# LIME (para el mismo modelo, usando KernelExplainer)
print("\n--- LIME (aproximación lineal local) ---")
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values, feature_names=X_train.columns.tolist(), mode='regression',
    random_state=42
)
lime_exp = lime_explainer.explain_instance(
    X_test.values[i], model.predict, num_features=5
)
for f, w in lime_exp.as_list():
    print(f"  {f}: {w:.4f}")

# PDP (efecto global)
print("\n--- PDP (efecto marginal global) ---")
pdp_vals = PartialDependenceDisplay.from_estimator(
    model, X_train, ['precio'], kind='average'
)
print("PDP mostrado en figura (efecto marginal del precio en demanda)")
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

*Ejemplo 16: Comparar SHAP vs LIME vs PDP en misma predicción.*

1. SHAP
2. LIME (para el mismo modelo, usando KernelExplainer)
3. PDP (efecto global)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 17: SHAP interaction values — Interacciones entre features

```python
# Calcular interacciones SHAP (solo TreeExplainer)
shap_interaction = explainer.shap_interaction_values(X_test)

print(f"Shape interaction values: {shap_interaction.shape}")
print(f"shap_interaction[i][j,k]: interacción entre feature j y feature k para instancia i")

# Matriz de interacción para la primera instancia
interaction_matrix = pd.DataFrame(
    shap_interaction[0],
    index=X_test.columns,
    columns=X_test.columns
)
print("\nMatriz de interacciones SHAP (instancia 0):")
print("Diagonal: efecto principal, Off-diagonal: interacciones")
print(interaction_matrix.round(2))

# Dependence plot con interacción específica
shap.dependence_plot('precio', shap_values, X_test,
                     interaction_index='descuento', show=False)
plt.title('Dependence Plot: Precio × Descuento (Interacción)')
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: SHAP interaction values — Interacciones entre features.*

1. Calcular interacciones SHAP (solo TreeExplainer)
2. Matriz de interacción para la primera instancia
3. Dependence plot con interacción específica

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 18: Integrador — Explicar modelo de predicción de ventas completo

```python
def explicar_modelo_ventas(model, X_train, X_test, y_test, feature_names):
    print("="*60)
    print("EXPLICABILIDAD DEL MODELO DE PREDICCIÓN DE VENTAS")
    print("="*60)
    
    # 1. Importancia global (SHAP summary)
    print("\n1. IMPORTANCIA GLOBAL DE FEATURES")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    importance = np.abs(shap_values).mean(axis=0)
    ranking = pd.DataFrame({'feature': feature_names, 'importance': importance})
    ranking = ranking.sort_values('importance', ascending=False)
    print(ranking.to_string(index=False))
    
    # 2. Resumen visual
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Summary plot
    shap.summary_plot(shap_values, X_test, max_display=6, show=False, ax=axes[0, 0])
    axes[0, 0].set_title('SHAP Summary')
    
    # Bar plot
    shap.summary_plot(shap_values, X_test, plot_type='bar', max_display=6,
                      show=False, ax=axes[0, 1])
    axes[0, 1].set_title('SHAP Feature Importance')
    
    # Dependence para feature más importante
    top_feat = ranking.iloc[0]['feature']
    shap.dependence_plot(top_feat, shap_values, X_test, show=False, ax=axes[1, 0])
    axes[1, 0].set_title(f'Dependence: {top_feat}')
    
    # PDP para top feature
    PartialDependenceDisplay.from_estimator(model, X_train, [top_feat],
                                             kind='average', ax=axes[1, 1])
    axes[1, 1].set_title(f'PDP: {top_feat}')
    
    plt.tight_layout()
    plt.show()
    
    # 3. Explicación individual (mejor y peor predicción)
    print("\n3. EXPLICACIÓN INDIVIDUAL - MEJOR PREDICCIÓN")
    preds = model.predict(X_test)
    errors = np.abs(preds - y_test.values)
    
    best_idx = errors.argmin()
    worst_idx = errors.argmax()
    
    print(f"Instancia con menor error: {best_idx}")
    print(f"  Predicción: {preds[best_idx]:.2f}, Real: {y_test.iloc[best_idx]:.2f}")
    force_best = shap.force_plot(explainer.expected_value, shap_values[best_idx],
                                  X_test.iloc[best_idx], matplotlib=True, show=False)
    plt.title('SHAP Force Plot - Mejor Predicción')
    plt.tight_layout(); plt.show()
    
    # 4. Interacciones
    print("\n4. INTERACCIONES PRINCIPALES")
    shap_interaction = explainer.shap_interaction_values(X_test)
    mean_interactions = np.abs(shap_interaction).mean(axis=0)
    np.fill_diagonal(mean_interactions, 0)  # Ignorar diagonal
    max_interaction = np.unravel_index(mean_interactions.argmax(), mean_interactions.shape)
    print(f"Interacción más fuerte: {feature_names[max_interaction[0]]} × {feature_names[max_interaction[1]]}")
    
    return {
        'explainer': explainer,
        'shap_values': shap_values,
        'ranking': ranking,
        'best_prediction_idx': best_idx,
        'worst_prediction_idx': worst_idx
    }

# Ejecutar
resultados_explicacion = explicar_modelo_ventas(model, X_train, X_test, y_test, X_train.columns)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador — Explicar modelo de predicción de ventas completo.*

1. 1. Importancia global (SHAP summary)
2. 2. Resumen visual
3. Summary plot
4. Bar plot
5. Dependence para feature más importante
6. PDP para top feature
7. 3. Explicación individual (mejor y peor predicción)
8. 4. Interacciones

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 3. Ejercicios Propuestos

1. **Explicar clasificador de productos exitosos**: Entrena un RandomForestClassifier para predecir si un producto será exitoso (>100 unidades vendidas/mes). Usa SHAP TreeExplainer para identificar los 5 features más importantes. Muestra summary_plot y dependence_plot para el feature principal.

2. **Comparar explicaciones SHAP para diferentes tipos de modelos**: Entrena un RandomForest, XGBoost y GradientBoosting para el mismo problema de regresión de demanda. Compara sus SHAP summary_plots. ¿Coinciden en las features más importantes?

3. **LIME para explicar predicciones de riesgo crediticio en clientes B2B**: Simula datos de clientes con 8 features. Para una predicción de "alto riesgo", usa LIME para mostrar qué features contribuyen positiva y negativamente. Interpreta en lenguaje de negocio.

4. **Waterfall plot para gerente de ventas**: Selecciona 3 instancias representativas (baja/media/alta demanda). Genera waterfall plots para cada una. Redacta una explicación en lenguaje natural para un gerente no técnico.

5. **PDP e ICE para elasticidad precio-demanda**: Genera PDP e ICE para el efecto del precio en la demanda. Identifica si hay heterogeneidad (diferentes pendientes para diferentes productos). Calcula la elasticidad precio promedio.

6. **SHAP interaction values para cross-selling**: En un modelo de recomendación de productos (clasificación multiclase), calcula SHAP interaction values entre precio y descuento. ¿Hay sinergia? Muestra dependence_plot con interacción.

7. **Dashboard de explicabilidad con InterpretML**: Usa ExplainableBoostingClassifier de InterpretML para entrenar un modelo de clasificación de compras. Genera el dashboard de explicabilidad global y local. Interpreta las funciones de forma de cada feature.

8. **Pipeline de explicabilidad automática**: Crea una función que, dado un modelo entrenado y datos de test, genere automáticamente: (1) tabla de importancia SHAP, (2) summary_plot, (3) dependence_plot para top-3 features, (4) waterfall para 3 instancias, (5) reporte en lenguaje natural. Guarda como HTML.

---

## 4. Resumen

| Herramienta | Tipo | Alcance | Velocidad | Interpretación |
|---|---|---|---|---|
| **SHAP** | Model-agnostic | Global y local | Lento (Kernel) / Rápido (Tree) | Teoría de juegos, Shapley values |
| **LIME** | Model-agnostic | Local | Medio | Aproximación lineal local |
| **Eli5** | Model-specific | Global y local | Rápido | Pesos/importancia directa |
| **PDP** | Model-agnostic | Global | Rápido | Efecto marginal |
| **ICE** | Model-agnostic | Individual | Rápido | Heterogeneidad de efectos |
| **ALE** | Model-agnostic | Global | Medio | No sesgado con correlación |
| **InterpretML** | Glassbox | Global y local | Rápido | Funciones de forma intrínsecas |

En ventas: SHAP es la opción más completa (global y local, cualquier modelo). PDP/ICE son ideales para comunicar elasticidad precio-demanda a stakeholders de negocio. LIME es útil para explicaciones rápidas ad-hoc.
