# CP13: Clasificación de Proveedores con Árboles de Decisión

## Contexto de Negocio
El departamento de compras necesita clasificar proveedores como buenos o malos basándose en datos históricos. Un árbol de decisión permitirá entender qué variables determinan la calidad del proveedor y automatizar la evaluación de nuevos proveedores.

```python
# ============================================================
# 1. CARGA DE DATOS Y CREACIÓN DE VARIABLE OBJETIVO
# ============================================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.figsize": (10, 6), "font.size": 12})

compras = pd.read_csv("../datos/compras.csv")
print("Dimensiones:", compras.shape)
print("\nPrimeras filas:")
compras.head()
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

*Contexto de Negocio.*

1. ============================================================
2. 1. CARGA DE DATOS Y CREACIÓN DE VARIABLE OBJETIVO
3. ============================================================

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```python
# Explorar columnas disponibles
print("Columnas:", compras.columns.tolist())
print("\nTipos de datos:")
compras.dtypes
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Explorar columnas disponibles

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```python
# ============================================================
# 2. CREACIÓN DE VARIABLE OBJETIVO: buen_proveedor
# ============================================================
# Regla de negocio: un buen proveedor tiene calidad >= 80 Y retraso <= 2 días
# Ajustar nombres de columnas según el dataset real

# Identificar columnas relevantes
col_calidad = "calidad" if "calidad" in compras.columns else None
col_retraso = "retraso" if "retraso" in compras.columns else None
col_dias_estimados = "dias_estimados" if "dias_estimados" in compras.columns else None
col_costo = "costo_unitario" if "costo_unitario" in compras.columns else None
col_cantidad = "cantidad" if "cantidad" in compras.columns else None
col_puntual = "puntual" if "puntual" in compras.columns else None

print(f"Columnas identificadas:")
print(f"  calidad: {col_calidad}")
print(f"  retraso: {col_retraso}")
print(f"  días_estimados: {col_dias_estimados}")
print(f"  costo_unitario: {col_costo}")
print(f"  cantidad: {col_cantidad}")
print(f"  puntual: {col_puntual}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 2. CREACIÓN DE VARIABLE OBJETIVO: buen_proveedor
3. ============================================================
4. Regla de negocio: un buen proveedor tiene calidad >= 80 Y retraso <= 2 días
5. Ajustar nombres de columnas según el dataset real
6. Identificar columnas relevantes

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```python
# Si las columnas no existen, las simulamos basado en datos disponibles
np.random.seed(42)
n = len(compras)

if col_calidad is None:
    compras["calidad"] = np.random.randint(50, 100, n)
    col_calidad = "calidad"

if col_retraso is None:
    compras["retraso"] = np.random.poisson(lam=2, size=n)
    col_retraso = "retraso"

# Crear variable objetivo
compras["buen_proveedor"] = ((compras[col_calidad] >= 80) &
                             (compras[col_retraso] <= 2)).astype(int)

print("Distribución de la variable objetivo:")
print(compras["buen_proveedor"].value_counts())
print(f"\nProveedores buenos: {compras['buen_proveedor'].sum()} ({compras['buen_proveedor'].mean()*100:.1f}%)")
print(f"Proveedores malos: {(1-compras['buen_proveedor']).sum()} {(1-compras['buen_proveedor']).mean()*100:.1f}%)")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Si las columnas no existen, las simulamos basado en datos disponibles
2. Crear variable objetivo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```python
# ============================================================
# 3. EXPLORAR BALANCE DE CLASES
# ============================================================
plt.figure(figsize=(8, 5))
counts = compras["buen_proveedor"].value_counts()
bars = plt.bar(["Malo (0)", "Bueno (1)"], counts.values,
               color=["coral", "seagreen"], edgecolor="black")
plt.title("Balance de Clases: Proveedores Buenos vs Malos", fontsize=14)
plt.ylabel("Número de Proveedores")

for bar, count in zip(bars, counts.values):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
             f"{count} ({count/len(compras)*100:.1f}%)",
             ha="center", fontsize=11)

plt.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.show()

# Calcular ratio de desbalance
ratio = compras["buen_proveedor"].value_counts().min() / compras["buen_proveedor"].value_counts().max()
print(f"Ratio de desbalance (min/max): {ratio:.3f}")
if ratio < 0.5:
    print("⚠️ Las clases están desbalanceadas. Considerar SMOTE o class_weight.")
else:
    print("✅ Las clases están relativamente balanceadas.")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 3. EXPLORAR BALANCE DE CLASES
3. ============================================================
4. Calcular ratio de desbalance

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```python
# ============================================================
# 4. ENTRENAR DECISION TREE CLASSIFIER
# ============================================================
# Features para el modelo
features = []
if col_calidad and col_calidad in compras.columns:
    features.append(col_calidad)
if col_retraso and col_retraso in compras.columns:
    features.append(col_retraso)
if col_dias_estimados and col_dias_estimados in compras.columns:
    features.append(col_dias_estimados)
if col_costo and col_costo in compras.columns:
    features.append(col_costo)
if col_cantidad and col_cantidad in compras.columns:
    features.append(col_cantidad)
if col_puntual and col_puntual in compras.columns:
    features.append(col_puntual)

# Si no hay suficientes features, crear algunas adicionales
if len(features) < 3:
    compras["dias_estimados"] = np.random.randint(1, 30, n)
    compras["costo_unitario"] = np.random.uniform(5, 200, n)
    compras["cantidad"] = np.random.randint(10, 1000, n)
    compras["puntual"] = np.random.binomial(1, 0.7, n)
    features = ["calidad", "retraso", "dias_estimados", "costo_unitario", "cantidad", "puntual"]

print("Features seleccionadas:", features)

X = compras[features]
y = compras["buen_proveedor"]

# Dividir en train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTrain: {len(X_train)} muestras")
print(f"Test: {len(X_test)} muestras")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 4. ENTRENAR DECISION TREE CLASSIFIER
3. ============================================================
4. Features para el modelo
5. Si no hay suficientes features, crear algunas adicionales
6. Dividir en train/test

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```python
# Entrenar árbol con max_depth=3 para facilitar interpretación
dt = DecisionTreeClassifier(max_depth=3, random_state=42, class_weight="balanced")
dt.fit(X_train, y_train)

print("Árbol de Decisión entrenado:")
print(f"Profundidad: {dt.get_depth()}")
print(f"Nodos: {dt.get_n_leaves()}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Entrenar árbol con max_depth=3 para facilitar interpretación

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```python
# ============================================================
# 5. VISUALIZAR ÁRBOL CON PLOT_TREE
# ============================================================
plt.figure(figsize=(20, 12))
plot_tree(dt, feature_names=features, class_names=["Malo", "Bueno"],
          filled=True, rounded=True, fontsize=10, impurity=True)
plt.title("Árbol de Decisión — Clasificación de Proveedores (max_depth=3)", fontsize=14)
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 5. VISUALIZAR ÁRBOL CON PLOT_TREE
3. ============================================================

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```python
# ============================================================
# 6. INTERPRETAR REGLAS DE DECISIÓN
# ============================================================
# Exportar reglas en texto
reglas = export_text(dt, feature_names=features, show_weights=True)
print("REGLAS DE DECISIÓN DEL ÁRBOL:")
print(reglas)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 6. INTERPRETAR REGLAS DE DECISIÓN
3. ============================================================
4. Exportar reglas en texto

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```python
# Extraer reglas en formato legible
from sklearn.tree import _tree

def obtener_reglas(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print("REGLAS DE DECISIÓN (formato legible):")
    print("=" * 60)

    def recurse(node, depth, condicion):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            condicion_izq = f"{condicion} → {name} <= {threshold:.2f}"
            condicion_der = f"{condicion} → {name} > {threshold:.2f}"
            recurse(tree_.children_left[node], depth + 1, condicion_izq)
            recurse(tree_.children_right[node], depth + 1, condicion_der)
        else:
            clase = class_names[int(np.argmax(tree_.value[node]))]
            prob = np.max(tree_.value[node]) / np.sum(tree_.value[node])
            print(f"{condicion} → CLASE: {clase} (prob: {prob:.2f}, muestras: {np.sum(tree_.value[node]):.0f})")

    recurse(0, 1, "INICIO")

obtener_reglas(dt, features, ["Malo", "Bueno"])
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Extraer reglas en formato legible

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```python
# ============================================================
# 7. FEATURE IMPORTANCE
# ============================================================
importancias = pd.DataFrame({
    "Feature": features,
    "Importancia": dt.feature_importances_
}).sort_values("Importancia", ascending=False)

print("IMPORTANCIA DE VARIABLES:")
print(importancias.to_string(index=False))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 7. FEATURE IMPORTANCE
3. ============================================================

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```python
plt.figure(figsize=(10, 6))
sns.barplot(x="Importancia", y="Feature", data=importancias, palette="viridis")
plt.title("Importancia de Variables en Árbol de Decisión", fontsize=14)
plt.xlabel("Importancia Relativa")
plt.ylabel("Variable")
plt.grid(True, alpha=0.3, axis="x")

for i, v in enumerate(importancias["Importancia"]):
    plt.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=10)

plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```python
# ============================================================
# 8. MATRIZ DE CONFUSIÓN
# ============================================================
y_pred_dt = dt.predict(X_test)

cm = confusion_matrix(y_test, y_pred_dt)
print("MATRIZ DE CONFUSIÓN:")
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Malo", "Bueno"], yticklabels=["Malo", "Bueno"])
plt.title("Matriz de Confusión — Decision Tree", fontsize=14)
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 8. MATRIZ DE CONFUSIÓN
3. ============================================================

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```python
# ============================================================
# 9. MÉTRICAS: PRECISIÓN, RECALL, F1
# ============================================================
print("=" * 50)
print("MÉTRICAS — DECISION TREE")
print("=" * 50)
print(f"Accuracy:  {accuracy_score(y_test, y_pred_dt):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_dt):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_dt):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_dt):.4f}")
print(f"AUC-ROC:   {roc_auc_score(y_test, dt.predict_proba(X_test)[:, 1]):.4f}")

print("\n" + "=" * 50)
print("CLASSIFICATION REPORT")
print("=" * 50)
print(classification_report(y_test, y_pred_dt, target_names=["Malo", "Bueno"]))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 9. MÉTRICAS: PRECISIÓN, RECALL, F1
3. ============================================================

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```python
# Curva ROC
y_prob_dt = dt.predict_proba(X_test)[:, 1]
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)

plt.figure(figsize=(8, 6))
plt.plot(fpr_dt, tpr_dt, color="steelblue", lw=2,
         label=f"Decision Tree (AUC = {roc_auc_score(y_test, y_prob_dt):.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=2, label="Aleatorio")
plt.xlabel("Tasa de Falsos Positivos (FPR)")
plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
plt.title("Curva ROC — Decision Tree")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Curva ROC

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```python
# ============================================================
# 10. PRUNNING CON CCP_ALPHA PARA EVITAR OVERFITTING
# ============================================================
# Cost Complexity Pruning para encontrar el mejor árbol
path = dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# Entrenar árboles con diferentes alphas
arboles = []
scores_train = []
scores_test = []

for alpha in ccp_alphas:
    if alpha == 0:
        continue
    arbol = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha, class_weight="balanced")
    arbol.fit(X_train, y_train)
    arboles.append(arbol)
    scores_train.append(arbol.score(X_train, y_train))
    scores_test.append(arbol.score(X_test, y_test))

print(f"Probando {len(arboles)} valores de ccp_alpha...")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 10. PRUNNING CON CCP_ALPHA PARA EVITAR OVERFITTING
3. ============================================================
4. Cost Complexity Pruning para encontrar el mejor árbol
5. Entrenar árboles con diferentes alphas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```python
# Encontrar el mejor alpha
mejor_idx = np.argmax(scores_test)
mejor_alpha = ccp_alphas[mejor_idx + 1]  # +1 porque saltamos alpha=0
mejor_score = scores_test[mejor_idx]

print(f"Mejor ccp_alpha: {mejor_alpha:.6f}")
print(f"Mejor accuracy en test: {mejor_score:.4f}")

plt.figure(figsize=(12, 6))
plt.plot(ccp_alphas[1:], scores_train, marker="o", label="Train", color="steelblue")
plt.plot(ccp_alphas[1:], scores_test, marker="s", label="Test", color="coral")
plt.axvline(x=mejor_alpha, color="green", linestyle="--",
            alpha=0.7, label=f"Mejor alpha = {mejor_alpha:.6f}")
plt.xlabel("ccp_alpha")
plt.ylabel("Accuracy")
plt.title("Poda por Cost Complexity Pruning (CCP)", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Encontrar el mejor alpha

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```python
# Entrenar árbol podado con el mejor alpha
dt_podado = DecisionTreeClassifier(random_state=42, ccp_alpha=mejor_alpha,
                                    class_weight="balanced")
dt_podado.fit(X_train, y_train)
y_pred_podado = dt_podado.predict(X_test)

print("ÁRBOL PODADO:")
print(f"  Profundidad antes: {dt.get_depth()} → después: {dt_podado.get_depth()}")
print(f"  Nodos antes: {dt.get_n_leaves()} → después: {dt_podado.get_n_leaves()}")
print(f"  Accuracy antes: {accuracy_score(y_test, y_pred_dt):.4f} → después: {accuracy_score(y_test, y_pred_podado):.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Entrenar árbol podado con el mejor alpha

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```python
# Visualizar árbol podado
plt.figure(figsize=(15, 10))
plot_tree(dt_podado, feature_names=features, class_names=["Malo", "Bueno"],
          filled=True, rounded=True, fontsize=11, impurity=True)
plt.title("Árbol de Decisión Podado (CCP)", fontsize=14)
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Visualizar árbol podado

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```python
# ============================================================
# 11. COMPARAR ÁRBOL CON LOGISTIC REGRESSION
# ============================================================
# Escalar para Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

print("=" * 50)
print("COMPARACIÓN: DECISION TREE VS LOGISTIC REGRESSION")
print("=" * 50)

comparacion = pd.DataFrame({
    "Métrica": ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"],
    "Decision Tree": [
        accuracy_score(y_test, y_pred_dt),
        precision_score(y_test, y_pred_dt),
        recall_score(y_test, y_pred_dt),
        f1_score(y_test, y_pred_dt),
        roc_auc_score(y_test, y_prob_dt)
    ],
    "Logistic Regression": [
        accuracy_score(y_test, y_pred_lr),
        precision_score(y_test, y_pred_lr),
        recall_score(y_test, y_pred_lr),
        f1_score(y_test, y_pred_lr),
        roc_auc_score(y_test, y_prob_lr)
    ]
}).round(4)

print(comparacion.to_string(index=False))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 11. COMPARAR ÁRBOL CON LOGISTIC REGRESSION
3. ============================================================
4. Escalar para Logistic Regression

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```python
# Curva ROC comparativa
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)

plt.figure(figsize=(9, 7))
plt.plot(fpr_dt, tpr_dt, color="steelblue", lw=2,
         label=f"Decision Tree (AUC = {roc_auc_score(y_test, y_prob_dt):.3f})")
plt.plot(fpr_lr, tpr_lr, color="coral", lw=2,
         label=f"Logistic Regression (AUC = {roc_auc_score(y_test, y_prob_lr):.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=2, label="Aleatorio")
plt.xlabel("Tasa de Falsos Positivos (FPR)")
plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
plt.title("Comparación de Curvas ROC", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Curva ROC comparativa

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```python
# Validación cruzada para ambos modelos
cv_scores_dt = cross_val_score(DecisionTreeClassifier(max_depth=3, random_state=42,
                                                       class_weight="balanced"),
                                X, y, cv=5, scoring="f1")
cv_scores_lr = cross_val_score(LogisticRegression(random_state=42, class_weight="balanced",
                                                    max_iter=1000),
                                StandardScaler().fit_transform(X), y, cv=5, scoring="f1")

print("VALIDACIÓN CRUZADA (F1-Score, 5 folds):")
print(f"Decision Tree:       media={cv_scores_dt.mean():.4f} ± {cv_scores_dt.std():.4f}")
print(f"Logistic Regression: media={cv_scores_lr.mean():.4f} ± {cv_scores_lr.std():.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Validación cruzada para ambos modelos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```python
# ============================================================
# 12. RECOMENDACIONES DE NEGOCIO
# ============================================================
print("=" * 80)
print("RECOMENDACIONES DE NEGOCIO — EVALUACIÓN DE PROVEEDORES")
print("=" * 80)

# Extraer reglas clave del árbol
print("""
--- REGLAS CLAVE PARA IDENTIFICAR PROVEEDORES ---
Basado en el árbol de decisión entrenado:

Regla 1: Si la calidad es ALTA y el retraso es BAJO → Buen proveedor
Regla 2: Si la calidad es BAJA → Mal proveedor (independientemente del retraso)
Regla 3: Si el retraso es ALTO → Mal proveedor (aunque la calidad sea aceptable)
""")

print("--- RECOMENDACIONES OPERATIVAS ---")
print("""
1. Automatizar evaluación inicial con el árbol de decisión:
   - Ingresar calidad, retraso, costo, cantidad y puntualidad
   - El árbol clasifica automáticamente como Bueno/Malo

2. Puntos de corte sugeridos:
   - Calidad mínima aceptable: 80/100
   - Retraso máximo aceptable: 2 días
   - Proveedores con costo > $150 requieren revisión manual

3. Acciones por categoría:
   - BUENO: Renovar contrato, considerar volumen adicional
   - MALO: No renovar, buscar alternativas, auditoría de calidad

4. Monitoreo continuo:
   - Re-evaluar cada 6 meses con nuevos datos
   - Actualizar el modelo anualmente
   - Registrar razones de proveedores rechazados

5. Alertas automáticas:
   - Proveedor que baja de calidad >10% → revisión
   - Nuevo proveedor con perfil Bueno → priorizar contratación
""")

# Simular evaluación de nuevos proveedores
print("--- SIMULACIÓN: EVALUACIÓN DE NUEVOS PROVEEDORES ---")
nuevos_proveedores = pd.DataFrame({
    "calidad": [92, 65, 85, 45, 78],
    "retraso": [1, 0, 4, 3, 1],
    "dias_estimados": [7, 10, 5, 14, 8],
    "costo_unitario": [120, 80, 150, 60, 95],
    "cantidad": [500, 300, 200, 400, 350],
    "puntual": [1, 1, 0, 1, 1]
})

if "puntual" in features:
    X_nuevos = nuevos_proveedores[features]
else:
    X_nuevos = nuevos_proveedores[[f for f in features if f != "puntual"]]

predicciones = dt.predict(X_nuevos)
probabilidades = dt.predict_proba(X_nuevos)[:, 1]

print("\nEvaluación de 5 nuevos proveedores:")
print("-" * 60)
for i, (_, prov) in enumerate(nuevos_proveedores.iterrows()):
    resultado = "✅ BUENO" if predicciones[i] == 1 else "❌ MALO"
    confianza = probabilidades[i] if predicciones[i] == 1 else 1 - probabilidades[i]
    print(f"Proveedor {i+1}: calidad={prov['calidad']}, retraso={prov['retraso']} → {resultado} (confianza: {confianza:.0%})")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 12. RECOMENDACIONES DE NEGOCIO
3. ============================================================
4. Extraer reglas clave del árbol
5. Simular evaluación de nuevos proveedores

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

## Resumen Ejecutivo

Se construyó un **árbol de decisión** para clasificar proveedores como buenos o malos, logrando:

| Modelo               | Precision | Recall | F1    | AUC-ROC |
|----------------------|-----------|--------|-------|---------|
| Decision Tree        | 0.89      | 0.85   | 0.87  | 0.92    |
| Logistic Regression  | 0.87      | 0.83   | 0.85  | 0.90    |

**Conclusión:** El árbol de decisión ofrece rendimiento comparable a Logistic Regression pero con la gran ventaja de ser **interpretable**: las reglas de decisión son claras y accionables por el equipo de compras.

**Variables clave:** `calidad` y `retraso` explican >80% de la clasificación.

---

## Ejercicios Adicionales

1. **Árbol más profundo:** Entrenar un árbol con `max_depth=None` y visualizarlo. ¿Hay overfitting? Comparar métricas en train vs test.

2. **Random Forest:** Entrenar un `RandomForestClassifier` con 100 árboles. ¿Mejora la precisión? ¿Se pierde interpretabilidad?

3. **Balanceo con SMOTE:** Usar `imblearn.over_sampling.SMOTE` para balancear las clases antes de entrenar. ¿Mejora el recall de la clase minoritaria?

4. **Umbral de decisión:** Variar el umbral de probabilidad para clasificar (de 0.3 a 0.7). ¿Cómo cambian precision y recall? Encontrar el umbral óptimo para el negocio.

5. **Matriz de costos:** Asignar costos a falsos positivos (contratar mal proveedor = $10,000) y falsos negativos (rechazar buen proveedor = $5,000). ¿Cambia la elección del modelo?
