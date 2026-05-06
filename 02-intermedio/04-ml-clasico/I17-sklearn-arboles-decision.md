# I17: Árboles de Decisión con scikit-learn

## 1. Fundamentos Teóricos

Los **árboles de decisión** son modelos no paramétricos que particionan el espacio de features mediante preguntas binarias (umbrales), formando una estructura jerárquica.

### Criterios de división
- **Gini**: `G = 1 - Σ(pᵢ)²` — mide impureza (0 = puro)
- **Entropy**: `H = -Σ(pᵢ·log₂(pᵢ))` — mide desorden (0 = orden perfecto)

### Hiperparámetros de poda
| Parámetro | Controla |
|-----------|----------|
| max_depth | Profundidad máxima del árbol |
| min_samples_split | Mínimo de muestras para dividir un nodo |
| min_samples_leaf | Mínimo de muestras en cada hoja |
| max_features | Features consideradas en cada división |
| max_leaf_nodes | Número máximo de hojas |
| ccp_alpha | Parámetro de poda por costo-complejidad |

### Ventajas
- Interpretables (visualizables)
- No requieren escalado
- Capturan relaciones no lineales
- Manejan mixtura de tipos de datos

### Desventajas
- Alta varianza (overfitting fácil)
- Inestables (pequeños cambios en datos → árbol diferente)
- Sesgo por features con muchos valores

---

## 2. Ejemplos Prácticos con sklearn

```python
# Configuración inicial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import (DecisionTreeClassifier, DecisionTreeRegressor,
                          plot_tree, export_text)
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc, RocCurveDisplay)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================
# EJEMPLO 1: DecisionTreeClassifier — clasificar productos rentables
# ============================================================
print("="*60)
print("EJEMPLO 1: Clasificar productos rentables")
print("="*60)

n = 500
df = pd.DataFrame({
    'precio': np.random.uniform(10, 500, n),
    'costo': np.random.uniform(5, 300, n),
    'rotacion_mensual': np.random.uniform(0, 100, n),
    'tamano_inventario': np.random.uniform(10, 1000, n),
    'perecible': np.random.choice([0, 1], n, p=[0.6, 0.4])
})
df['margen_pct'] = (df['precio'] - df['costo']) / df['precio'] * 100
df['rentable'] = (
    (df['margen_pct'] > 25) & (df['rotacion_mensual'] > 20)
) | ((df['margen_pct'] > 40) & (df['rotacion_mensual'] > 5))
df['rentable'] = (df['rentable'] | (np.random.random(n) < 0.1)).astype(int)

X = df[['precio', 'costo', 'rotacion_mensual', 'tamano_inventario',
         'perecible', 'margen_pct']]
y = df['rentable']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)
print(f"Accuracy en train: {(dt.predict(X_train) == y_train).mean():.3f}")
print(f"Accuracy en test:  {(y_pred == y_test).mean():.3f}")
print(f"Overfitting detectado: train >> test")

# ============================================================
# EJEMPLO 2: Visualizar árbol con plot_tree y export_text
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 2: Visualización del árbol")
print("="*60)

dt_viz = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_viz.fit(X_train, y_train)

# Exportar como texto
text_tree = export_text(dt_viz, feature_names=list(X.columns))
print("Árbol en texto (max_depth=3):")
print(text_tree)

# Guardar gráfico
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(dt_viz, feature_names=list(X.columns), class_names=['No Rentable', 'Rentable'],
          filled=True, rounded=True, fontsize=10, ax=ax)
plt.savefig('I17_tree_visual.png', dpi=100, bbox_inches='tight')
plt.close()
print("Gráfico guardado: I17_tree_visual.png")

# ============================================================
# EJEMPLO 3: max_depth — controlar profundidad (2, 5, 10, 20)
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 3: Control de profundidad (max_depth)")
print("="*60)

for depth in [2, 5, 10, 20, None]:
    dt_d = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt_d.fit(X_train, y_train)
    train_acc = (dt_d.predict(X_train) == y_train).mean()
    test_acc = (dt_d.predict(X_test) == y_test).mean()
    n_nodos = dt_d.tree_.node_count
    n_hojas = dt_d.tree_.n_leaves
    print(f"max_depth={str(depth):5s} | train={train_acc:.3f} test={test_acc:.3f} "
          f"| nodos={n_nodos} | hojas={n_hojas}")

print("\nA mayor profundidad: más nodos, mejor train, peor generalización (overfitting)")

# ============================================================
# EJEMPLO 4: min_samples_split — mínimo muestras para dividir
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 4: min_samples_split")
print("="*60)

for mss in [2, 5, 10, 50, 100]:
    dt_mss = DecisionTreeClassifier(min_samples_split=mss, random_state=42)
    dt_mss.fit(X_train, y_train)
    train_acc = (dt_mss.predict(X_train) == y_train).mean()
    test_acc = (dt_mss.predict(X_test) == y_test).mean()
    n_hojas = dt_mss.tree_.n_leaves
    print(f"min_samples_split={mss:3d} | train={train_acc:.3f} test={test_acc:.3f} "
          f"| hojas={n_hojas}")

print("\nMayor min_samples_split → árbol más simple, menos overfitting")

# ============================================================
# EJEMPLO 5: min_samples_leaf — mínimo muestras por hoja
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 5: min_samples_leaf")
print("="*60)

for msl in [1, 5, 10, 20, 50]:
    dt_msl = DecisionTreeClassifier(min_samples_leaf=msl, random_state=42)
    dt_msl.fit(X_train, y_train)
    train_acc = (dt_msl.predict(X_train) == y_train).mean()
    test_acc = (dt_msl.predict(X_test) == y_test).mean()
    n_hojas = dt_msl.tree_.n_leaves
    print(f"min_samples_leaf={msl:2d}  | train={train_acc:.3f} test={test_acc:.3f} "
          f"| hojas={n_hojas}")

print("\nMayor min_samples_leaf → hojas con más muestras, modelo más robusto")

# ============================================================
# EJEMPLO 6: criterion='entropy' vs 'gini'
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 6: criterion='entropy' vs 'gini'")
print("="*60)

for criterion in ['gini', 'entropy', 'log_loss']:
    dt_c = DecisionTreeClassifier(criterion=criterion, max_depth=5, random_state=42)
    dt_c.fit(X_train, y_train)
    train_acc = (dt_c.predict(X_train) == y_train).mean()
    test_acc = (dt_c.predict(X_test) == y_test).mean()
    print(f"criterion={criterion:10s} | train={train_acc:.3f} test={test_acc:.3f}")

print("\nGini y entropy suelen dar resultados similares. Gini es computacionalmente más rápido.")

# ============================================================
# EJEMPLO 7: feature_importances_ — ranking de variables
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 7: Feature Importances")
print("="*60)

dt_fi = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_fi.fit(X_train, y_train)

importancias = pd.DataFrame({
    'Feature': X.columns,
    'Importancia': dt_fi.feature_importances_
}).sort_values('Importancia', ascending=False)

print(importancias.to_string(index=False))
print(f"\nSuma de importancias: {dt_fi.feature_importances_.sum():.3f} (debe ser 1)")

# Gráfico
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(importancias['Feature'], importancias['Importancia'])
ax.set_xlabel('Importancia')
ax.set_title('Feature Importances — Árbol de Decisión')
plt.tight_layout()
plt.savefig('I17_feature_importances.png', dpi=100)
plt.close()
print("Gráfico guardado: I17_feature_importances.png")

# ============================================================
# EJEMPLO 8: Pruning con ccp_alpha — poda de costo-complejidad
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 8: Poda con ccp_alpha (costo-complejidad)")
print("="*60)

# Obtener ruta de poda
dt_ccp = DecisionTreeClassifier(random_state=42)
path = dt_ccp.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

print(f"Total de alphas disponibles: {len(ccp_alphas)}")
print(f"Alphas: {ccp_alphas[:10]}...")

# Evaluar diferentes alphas
train_scores = []
test_scores = []
ccp_alphas_subset = ccp_alphas[::len(ccp_alphas)//10]  # Tomar 10 alphas

for alpha in ccp_alphas_subset:
    dt_alpha = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    dt_alpha.fit(X_train, y_train)
    train_scores.append((dt_alpha.predict(X_train) == y_train).mean())
    test_scores.append((dt_alpha.predict(X_test) == y_test).mean())
    n_nodos = dt_alpha.tree_.node_count
    print(f"alpha={alpha:.5f} | train={train_scores[-1]:.3f} test={test_scores[-1]:.3f} "
          f"| nodos={n_nodos}")

print("\nccp_alpha mayor → más poda → árbol más simple")
print("El alpha óptimo maximiza test accuracy")

# Seleccionar mejor alpha
best_idx = np.argmax(test_scores)
best_alpha = ccp_alphas_subset[best_idx]
print(f"Mejor alpha: {best_alpha:.5f} (test={test_scores[best_idx]:.3f})")

# ============================================================
# EJEMPLO 9: DecisionTreeRegressor — predecir cantidad vendida
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 9: Árbol de regresión — predecir cantidad vendida")
print("="*60)

n_reg = 400
df_reg = pd.DataFrame({
    'precio': np.random.uniform(10, 200, n_reg),
    'stock': np.random.randint(0, 300, n_reg),
    'promocion': np.random.choice([0, 1], n_reg, p=[0.7, 0.3]),
    'dia_semana': np.random.randint(1, 8, n_reg),
    'temp_media': np.random.uniform(10, 35, n_reg)
})

# Target: cantidad vendida (regresión)
df_reg['cantidad'] = (
    30 - 0.1 * df_reg['precio']
    + 0.05 * df_reg['stock']
    + 15 * df_reg['promocion']
    + 3 * np.sin(df_reg['dia_semana'])
    + 0.5 * df_reg['temp_media']
    + np.random.normal(0, 8, n_reg)
)
df_reg['cantidad'] = df_reg['cantidad'].clip(0)

Xr = df_reg.drop('cantidad', axis=1)
yr = df_reg['cantidad']

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    Xr, yr, test_size=0.3, random_state=42
)

dtr = DecisionTreeRegressor(max_depth=5, random_state=42)
dtr.fit(Xr_train, yr_train)

yr_pred = dtr.predict(Xr_test)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print(f"R² en train: {dtr.score(Xr_train, yr_train):.3f}")
print(f"R² en test:  {dtr.score(Xr_test, yr_test):.3f}")
print(f"MAE: {mean_absolute_error(yr_test, yr_pred):.2f} unidades")
print(f"RMSE: {np.sqrt(mean_squared_error(yr_test, yr_pred)):.2f} unidades")

# ============================================================
# EJEMPLO 10: criterion='absolute_error' para robustez a outliers
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 10: criterion absolute_error (robusto a outliers)")
print("="*60)

# Añadir outliers artificiales
yr_train_out = yr_train.copy()
outlier_idx = np.random.choice(len(yr_train_out), 10, replace=False)
yr_train_out.iloc[outlier_idx] *= 5  # Multiplicar por 5 algunos valores

for criterion in ['squared_error', 'absolute_error', 'friedman_mse', 'poisson']:
    try:
        dtr_c = DecisionTreeRegressor(criterion=criterion, max_depth=5, random_state=42)
        dtr_c.fit(Xr_train, yr_train_out)
        yr_pred_c = dtr_c.predict(Xr_test)
        mae = mean_absolute_error(yr_test, yr_pred_c)
        rmse = np.sqrt(mean_squared_error(yr_test, yr_pred_c))
        r2 = r2_score(yr_test, yr_pred_c)
        print(f"criterion={criterion:16s} | MAE={mae:.2f} | RMSE={rmse:.2f} | R²={r2:.3f}")
    except Exception as e:
        print(f"criterion={criterion:16s} | Error: {e}")

print("\nabsolute_error es más robusto a outliers que squared_error")

# ============================================================
# EJEMPLO 11: class_weight='balanced' para desbalanceo
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 11: class_weight balanced en árbol de decisión")
print("="*60)

# Crear versión desbalanceada del dataset de rentabilidad
df_desb = df.copy()
# Eliminar 70% de los rentables
mask_rent = df_desb['rentable'] == 1
drop_idx = mask_rent[mask_rent].sample(frac=0.7, random_state=42).index
df_desb = df_desb.drop(drop_idx)
print(f"Rentables: {(df_desb['rentable']==1).sum()}, No rentables: {(df_desb['rentable']==0).sum()}")

Xd = df_desb[X.columns]
yd = df_desb['rentable']
Xd_train, Xd_test, yd_train, yd_test = train_test_split(Xd, yd, test_size=0.3, random_state=42)

dt_nb = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_nb.fit(Xd_train, yd_train)

dt_bal = DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42)
dt_bal.fit(Xd_train, yd_train)

print(f"\nSin balanceo — accuracy: {(dt_nb.predict(Xd_test)==yd_test).mean():.3f}")
print(f"Con balanced — accuracy: {(dt_bal.predict(Xd_test)==yd_test).mean():.3f}")
print("\nReporte sin balanceo:")
print(classification_report(yd_test, dt_nb.predict(Xd_test)))
print("Reporte con balanceo:")
print(classification_report(yd_test, dt_bal.predict(Xd_test)))

# ============================================================
# EJEMPLO 12: Decision path — ruta de decisión para una predicción
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 12: Decision path — ruta de decisión")
print("="*60)

dt_path = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_path.fit(X_train, y_train)

# Tomar 3 muestras de test
sample_indices = [0, 5, 12]
for idx in sample_indices:
    sample = X_test.iloc[[idx]]
    pred = dt_path.predict(sample)[0]
    prob = dt_path.predict_proba(sample)[0]

    # Obtener decision path
    node_indicator = dt_path.decision_path(sample)

    # Obtener reglas de las hojas
    leaf_id = dt_path.apply(sample)[0]

    print(f"\nMuestra #{idx} — Predicción: {'Rentable' if pred else 'No Rentable'} "
          f"(prob={prob[1]:.3f}) — Hoja: {leaf_id}")

    # Mostrar condiciones en el camino
    feature = dt_path.tree_.feature
    threshold = dt_path.tree_.threshold
    node_count = dt_path.tree_.node_count

    path_nodes = node_indicator.indices[
        node_indicator.indptr[0]:node_indicator.indptr[1]
    ]
    print(f"  Ruta: ", end="")
    for node_id in path_nodes:
        if feature[node_id] != -2:  # No es hoja
            fname = X.columns[feature[node_id]]
            thresh = threshold[node_id]
            val = sample[fname].values[0]
            direction = "≤" if val <= thresh else ">"
            print(f"[{fname} {direction} {thresh:.2f} (val={val:.2f})]", end=" → ")
        else:
            print(f"[HOJA {node_id}]")
    print()

# ============================================================
# EJEMPLO 13: Límite de decisión 2D — visualizar frontera
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 13: Límite de decisión 2D")
print("="*60)

X_2d = df[['precio', 'margen_pct']].values
y_2d = df['rentable'].values
X_2d_train, X_2d_test, y_2d_train, y_2d_test = train_test_split(
    X_2d, y_2d, test_size=0.3, random_state=42
)

for depth, color in [(2, 'Blues'), (5, 'Greens'), (10, 'Oranges')]:
    dt_2d = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt_2d.fit(X_2d_train, y_2d_train)
    test_acc = (dt_2d.predict(X_2d_test) == y_2d_test).mean()

    # Grid
    xx, yy = np.meshgrid(
        np.linspace(X_2d[:, 0].min()-5, X_2d[:, 0].max()+5, 200),
        np.linspace(X_2d[:, 1].min()-2, X_2d[:, 1].max()+2, 200)
    )
    Z = dt_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.contourf(xx, yy, Z, alpha=0.35, cmap=color)
    ax.scatter(X_2d_test[:, 0], X_2d_test[:, 1], c=y_2d_test,
               cmap='coolwarm', edgecolors='k', linewidth=0.5)
    ax.set_xlabel('Precio')
    ax.set_ylabel('Margen %')
    ax.set_title(f'Frontera Árbol (max_depth={depth}) — test_acc={test_acc:.3f}')
    plt.tight_layout()
    plt.savefig(f'I17_decision_boundary_depth_{depth}.png', dpi=100)
    plt.close()
    print(f"Gráfico guardado: I17_decision_boundary_depth_{depth}.png (acc={test_acc:.3f})")

print("Árboles más profundos crean fronteras más complejas y fragmentadas")

# ============================================================
# EJEMPLO 14: Comparar profundidades (2 vs 5 vs 10) train/test score
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 14: Comparación de profundidades — curva train/test")
print("="*60)

depths = range(1, 21)
train_accs = []
test_accs = []

for depth in depths:
    dt_d = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt_d.fit(X_train, y_train)
    train_accs.append((dt_d.predict(X_train) == y_train).mean())
    test_accs.append((dt_d.predict(X_test) == y_test).mean())

# Encontrar profundidad óptima
best_depth = depths[np.argmax(test_accs)]
print(f"Profundidad óptima: {best_depth} (test={max(test_accs):.3f})")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(depths, train_accs, 'b-o', label='Train')
ax.plot(depths, test_accs, 'r-o', label='Test')
ax.axvline(best_depth, color='g', linestyle='--', label=f'Óptimo depth={best_depth}')
ax.set_xlabel('max_depth')
ax.set_ylabel('Accuracy')
ax.set_title('Curva de validación: profundidad vs accuracy')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig('I17_depth_comparison.png', dpi=100)
plt.close()
print("Gráfico guardado: I17_depth_comparison.png")
print(f"\nInterpretación: depth={best_depth} da mejor generalización")
print("Depths menores → underfitting, mayores → overfitting")

# ============================================================
# EJEMPLO 15: max_leaf_nodes — limitar número de hojas
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 15: max_leaf_nodes — limitar hojas")
print("="*60)

leaf_counts = [5, 10, 20, 50, 100, None]
for ml in leaf_counts:
    dt_ml = DecisionTreeClassifier(max_leaf_nodes=ml, random_state=42)
    dt_ml.fit(X_train, y_train)
    train_acc = (dt_ml.predict(X_train) == y_train).mean()
    test_acc = (dt_ml.predict(X_test) == y_test).mean()
    actual_leaves = dt_ml.tree_.n_leaves
    print(f"max_leaf_nodes={str(ml):5s} | hojas_reales={actual_leaves:3d} | "
          f"train={train_acc:.3f} test={test_acc:.3f}")

print("\nmax_leaf_nodes controla directamente el tamaño del árbol")

# ============================================================
# EJEMPLO 16: Comparar árbol con LogisticRegression
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 16: Árbol vs Regresión Logística")
print("="*60)

# Logistic Regression (requiere escalado)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_s, y_train)

dt_final = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_final.fit(X_train, y_train)

modelos = {
    'Árbol (depth=5)': (dt_final, X_test, X_train),
    'LogisticRegression': (lr, X_test_s, X_train_s)
}

for nombre, (modelo, Xte, Xtr) in modelos.items():
    yp_te = modelo.predict(Xte)
    yp_tr = modelo.predict(Xtr)
    acc_te = (yp_te == y_test).mean()
    acc_tr = (yp_tr == y_train).mean()
    print(f"{nombre:22s} | train={acc_tr:.3f} test={acc_te:.3f}")
    if hasattr(modelo, 'coef_'):
        print(f"  Coeficientes: {np.round(modelo.coef_[0], 4)}")
    if hasattr(modelo, 'feature_importances_'):
        print(f"  Importancias: {np.round(modelo.feature_importances_, 4)}")

print("\nÁrbol captura no linealidades; logística es más estable con pocos datos")

# ============================================================
# EJEMPLO 17: GridSearchCV para árbol de decisión
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 17: GridSearchCV para optimizar árbol")
print("="*60)

param_grid = {
    'max_depth': [3, 5, 7, 10, 15],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 5, 10],
    'criterion': ['gini', 'entropy']
}

gs_tree = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)
gs_tree.fit(X_train, y_train)

print(f"Mejores parámetros: {gs_tree.best_params_}")
print(f"Mejor score CV: {gs_tree.best_score_:.4f}")
print(f"Test accuracy: {(gs_tree.predict(X_test)==y_test).mean():.4f}")

# Mostrar top 5 combinaciones
results_df = pd.DataFrame(gs_tree.cv_results_)
top5 = results_df.sort_values('rank_test_score').head(5)
print("\nTop 5 combinaciones:")
for i, row in top5.iterrows():
    print(f"  params={row['params']} | mean_score={row['mean_test_score']:.4f} "
          f"| std={row['std_test_score']:.4f}")

# ============================================================
# EJEMPLO 18: Integrador — árbol para decisión de reorden de inventario
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 18: Árbol para decisión de reorden de inventario")
print("="*60)

n_reord = 600
np.random.seed(42)
df_reord = pd.DataFrame({
    'stock_actual': np.random.randint(0, 500, n_reord),
    'demanda_diaria_prom': np.random.uniform(1, 50, n_reord),
    'lead_time_dias': np.random.uniform(1, 14, n_reord),
    'costo_pedido': np.random.uniform(50, 500, n_reord),
    'costo_almacenamiento': np.random.uniform(1, 20, n_reord),
    'dias_ultimo_reorden': np.random.randint(0, 60, n_reord),
    'tendencia_temporada': np.random.choice([-1, 0, 1], n_reord, p=[0.2, 0.5, 0.3])
})

# Regla de decisión: ¿necesita reorden?
# Punto de reorden = demanda_diaria_prom * lead_time_dias * 1.5
df_reord['punto_reorden'] = (
    df_reord['demanda_diaria_prom'] * df_reord['lead_time_dias'] * 1.5
    + df_reord['costo_pedido'] * 0.1
    - df_reord['costo_almacenamiento'] * 5
)
df_reord['necesita_reorden'] = (
    df_reord['stock_actual'] < df_reord['punto_reorden']
).astype(int)
# Ruido
mask_ruido = np.random.random(n_reord) < 0.12
df_reord.loc[mask_ruido, 'necesita_reorden'] = (
    1 - df_reord.loc[mask_ruido, 'necesita_reorden']
).astype(int)

features_reord = ['stock_actual', 'demanda_diaria_prom', 'lead_time_dias',
                  'costo_pedido', 'costo_almacenamiento', 'dias_ultimo_reorden',
                  'tendencia_temporada']
Xr_df = df_reord[features_reord]
yr_df = df_reord['necesita_reorden']

Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
    Xr_df, yr_df, test_size=0.3, random_state=42
)

# Árbol optimizado por GridSearchCV
gs_reord = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    {
        'max_depth': [3, 5, 7, 10],
        'min_samples_leaf': [5, 10, 20],
        'criterion': ['gini', 'entropy']
    },
    cv=5, scoring='f1', n_jobs=-1
)
gs_reord.fit(Xr_tr, yr_tr)

print(f"Mejores parámetros: {gs_reord.best_params_}")
print(f"F1 en test: {gs_reord.score(Xr_te, yr_te):.3f}")
print(f"Accuracy en test: {(gs_reord.predict(Xr_te) == yr_te).mean():.3f}")

# Visualizar árbol óptimo (profundidad limitada para legibilidad)
best_tree = gs_reord.best_estimator_
print(f"\nÁrbol óptimo — profundidad: {best_tree.get_depth()}, "
      f"hojas: {best_tree.get_n_leaves()}")

print("\nÁrbol en texto:")
tree_text = export_text(best_tree, feature_names=features_reord, max_depth=4)
print(tree_text)

# Interpretación: ¿qué features son más importantes?
importancias_reord = pd.DataFrame({
    'Feature': features_reord,
    'Importancia': best_tree.feature_importances_
}).sort_values('Importancia', ascending=False)
print("\nFeatures más importantes para decisión de reorden:")
print(importancias_reord.to_string(index=False))

print("\n=== FIN EJEMPLOS ÁRBOLES DE DECISIÓN ===")
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

*2. Ejemplos Prácticos con sklearn.*

1. Configuración inicial
2. ============================================================
3. EJEMPLO 1: DecisionTreeClassifier — clasificar productos rentables
4. ============================================================
5. ============================================================
6. EJEMPLO 2: Visualizar árbol con plot_tree y export_text
7. ============================================================
8. Exportar como texto

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

## 3. Ejercicios Propuestos

1. **Poda óptima**: Usa `cost_complexity_pruning_path` para encontrar el `ccp_alpha` óptimo en el dataset de rentabilidad. Grafica train vs test accuracy en función de alpha.

2. **Árbol de regresión para stock**: Usa el dataset de cantidad vendida (Ejemplo 9) y compara `max_depth=3` vs `max_depth=10`. ¿Cuál es la diferencia en MAE y en la estructura del árbol?

3. **Análisis de una predicción específica**: Toma 5 productos del test set y usa `decision_path()` para explicar por qué el árbol los clasificó como rentables o no. Presenta la ruta completa como reglas if-then-else.

4. **Comparación de criterios**: En el dataset de rentabilidad, prueba los 4 criterios de regresión (`squared_error`, `absolute_error`, `friedman_mse`, `poisson`) para predecir cantidad. ¿Cuál es mejor con outliers?

5. **Estabilidad del árbol**: Entrena 10 árboles con diferentes `random_state` y compara las feature importances. ¿Qué features son consistentemente importantes? ¿Cuáles varían?

6. **Clases desbalanceadas**: En el dataset de reorden (Ejemplo 18), compara `class_weight=None` vs `class_weight='balanced'` vs `class_weight={0:1, 1:3}`. ¿Cómo cambian precisión y recall de la clase minoritaria?

7. **max_features**: Prueba `max_features` en `['sqrt', 'log2', None, 0.3, 0.8]` para el dataset de rentabilidad. ¿Cómo afecta la diversidad del árbol y el accuracy?

8. **Pipeline árbol + PCA**: Crea un pipeline con PCA (reducir a 3 componentes) + DecisionTreeClassifier. Compara accuracy vs árbol sin PCA. ¿La pérdida de interpretabilidad vale la ganancia en velocidad?

---

## 4. Resumen

- Los **árboles de decisión** son modelos intuitivos que particionan el espacio con reglas if-then-else
- **Criterios de división**: Gini (rápido) vs Entropy (más informativo) — resultados similares
- **Hiperparámetros de poda**: `max_depth`, `min_samples_split`, `min_samples_leaf` controlan la complejidad
- **feature_importances_** suma a 1 y mide reducción de impureza ponderada
- La **poda por costo-complejidad** (`ccp_alpha`) encuentra el tamaño óptimo automáticamente
- **Árboles de regresión** (`DecisionTreeRegressor`) predicen valores continuos
- **decision_path** permite explicar predicciones individuales (transparencia)
- **No requieren escalado** de features — ventaja vs regresión logística

### Cuándo usar árboles de decisión
- Necesitas un modelo **interpretable y explicable**
- Las relaciones entre features son **no lineales**
- Tienes features de **diferentes escalas** (no requieren normalización)
- Necesitas **reglas de decisión** claras para el negocio

### Limitaciones
- **Alta varianza**: pequeños cambios en datos → árbol completamente diferente
- **Overfitting** si no se poda adecuadamente
- Sesgo hacia features con **muchos niveles** o valores únicos
- Inestables para **datos ruidosos** (las fronteras ortogonales pueden ser artificiales)
