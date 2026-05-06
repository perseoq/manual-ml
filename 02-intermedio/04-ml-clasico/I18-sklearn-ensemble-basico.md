# I18: Ensembles Básicos con scikit-learn

## 1. Fundamentos Teóricos

Los **métodos ensemble** combinan múltiples modelos (weak learners) para crear un predictor más fuerte y estable que cualquiera de los modelos individuales.

### Tipos principales de ensembles

| Método | Mecanismo | Reducción |
|--------|-----------|-----------|
| **Bagging** | Entrena modelos en paralelo con bootstrap | Reduce varianza |
| **Boosting** | Entrena modelos secuencialmente corrigiendo errores | Reduce sesgo |
| **Stacking** | Meta-modelo sobre predicciones de modelos base | Ambos |
| **Voting** | Promedia/vota predicciones de modelos independientes | Varianza |

### Algoritmos clave

- **RandomForest**: Bagging sobre árboles, con aleatoriedad en features (más robusto que árbol simple)
- **GradientBoosting**: Boosting con gradiente descendente
- **AdaBoost**: Pesos adaptativos en muestras mal clasificadas
- **ExtraTrees**: Aleatoriedad extrema en umbrales de corte
- **VotingClassifier**: Combinación por voto o promedios
- **StackingClassifier**: Meta-aprendizaje sobre modelos base

### Hiperparámetros comunes
- `n_estimators`: número de modelos base
- `learning_rate`: contribución de cada modelo (boosting)
- `max_depth`: profundidad de árboles base
- `subsample`: fracción de datos para cada iteración (boosting)

---

## 2. Ejemplos Prácticos con sklearn

```python
# Configuración inicial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, VotingClassifier, BaggingClassifier,
    StackingClassifier, ExtraTreesClassifier
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================
# EJEMPLO 1: RandomForestClassifier — clasificar + feature_importance
# ============================================================
print("="*60)
print("EJEMPLO 1: RandomForestClassifier")
print("="*60)

n = 600
df = pd.DataFrame({
    'precio': np.random.uniform(10, 500, n),
    'margen': np.random.uniform(5, 45, n),
    'stock': np.random.randint(0, 300, n),
    'rotacion': np.random.uniform(0, 100, n),
    'perecible': np.random.choice([0, 1], n),
    'promocion': np.random.choice([0, 1], n, p=[0.75, 0.25])
})
df['alta_demanda'] = (
    (df['rotacion'] > 30) & (df['margen'] > 15) |
    (df['promocion'] == 1) & (df['rotacion'] > 20) |
    (df['stock'] < 50) & (df['rotacion'] > 40)
).astype(int)
# Ruido
mask_noise = np.random.random(n) < 0.1
df.loc[mask_noise, 'alta_demanda'] = 1 - df.loc[mask_noise, 'alta_demanda']

X = df.drop('alta_demanda', axis=1)
y = df['alta_demanda']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print(f"Accuracy: {(y_pred == y_test).mean():.3f}")
print(f"OOB score: {rf.oob_score_:.3f}" if hasattr(rf, 'oob_score_') else "OOB no disponible")

# Feature importances
importancias = pd.DataFrame({
    'Feature': X.columns,
    'Importancia': rf.feature_importances_
}).sort_values('Importancia', ascending=False)
print("\nFeature Importances:")
print(importancias.to_string(index=False))

# ============================================================
# EJEMPLO 2: RandomForestRegressor — predecir demanda
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 2: RandomForestRegressor — predecir demanda")
print("="*60)

n_reg = 500
df_reg = pd.DataFrame({
    'precio': np.random.uniform(10, 300, n_reg),
    'gasto_marketing': np.random.uniform(100, 5000, n_reg),
    'num_sucursales': np.random.randint(1, 20, n_reg),
    'temporada_alta': np.random.choice([0, 1], n_reg, p=[0.7, 0.3]),
    'competidores': np.random.randint(0, 10, n_reg)
})
df_reg['demanda'] = (
    50 - 0.15 * df_reg['precio']
    + 0.02 * df_reg['gasto_marketing']
    + 5 * df_reg['num_sucursales']
    + 30 * df_reg['temporada_alta']
    - 3 * df_reg['competidores']
    + np.random.normal(0, 15, n_reg)
).clip(0)

Xr = df_reg.drop('demanda', axis=1)
yr = df_reg['demanda']
Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.3, random_state=42)

rfr = RandomForestRegressor(n_estimators=100, random_state=42)
rfr.fit(Xr_train, yr_train)

yr_pred = rfr.predict(Xr_test)
print(f"R²: {r2_score(yr_test, yr_pred):.3f}")
print(f"MAE: {mean_absolute_error(yr_test, yr_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(yr_test, yr_pred)):.2f}")

# ============================================================
# EJEMPLO 3: n_estimators — comparar 10 vs 50 vs 100 vs 500
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 3: Comparación de n_estimators")
print("="*60)

for n_est in [10, 50, 100, 200, 500]:
    rf_n = RandomForestClassifier(n_estimators=n_est, random_state=42, n_jobs=-1)
    rf_n.fit(X_train, y_train)
    train_acc = (rf_n.predict(X_train) == y_train).mean()
    test_acc = (rf_n.predict(X_test) == y_test).mean()
    print(f"n_estimators={n_est:3d} | train={train_acc:.4f} test={test_acc:.4f} "
          f"| diff={train_acc - test_acc:.4f}")

print("\nMás árboles → mejor generalización (hasta un punto) y más lento")

# ============================================================
# EJEMPLO 4: oob_score — evaluación out-of-bag
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 4: Out-of-Bag Score (oob_score)")
print("="*60)

for oob in [True, False]:
    rf_oob = RandomForestClassifier(
        n_estimators=200, oob_score=oob, random_state=42, n_jobs=-1
    )
    rf_oob.fit(X_train, y_train)
    test_acc = (rf_oob.predict(X_test) == y_test).mean()
    if oob:
        print(f"OOB activado — oob_score={rf_oob.oob_score_:.4f} | test={test_acc:.4f}")
    else:
        print(f"OOB desactivado — test={test_acc:.4f}")

print("OOB_score estima el error sin necesidad de validación cruzada")

# ============================================================
# EJEMPLO 5: ExtraTrees — aún más aleatoriedad
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 5: ExtraTrees (Extremely Randomized Trees)")
print("="*60)

et = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
et.fit(X_train, y_train)

print(f"ExtraTrees - Train: {(et.predict(X_train)==y_train).mean():.4f}")
print(f"ExtraTrees - Test:  {(et.predict(X_test)==y_test).mean():.4f}")
print(f"RF        - Test:  {(rf.predict(X_test)==y_test).mean():.4f}")

print("\nExtraTrees: más aleatoriedad (umbral aleatorio vs óptimo)")
print("→ menor varianza, puede aumentar sesgo")

# ============================================================
# EJEMPLO 6: GradientBoostingClassifier — boosting secuencial
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 6: GradientBoostingClassifier")
print("="*60)

gb = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
)
gb.fit(X_train, y_train)

y_pred_gb = gb.predict(X_test)
print(f"GB Train: {(gb.predict(X_train)==y_train).mean():.4f}")
print(f"GB Test:  {(y_pred_gb==y_test).mean():.4f}")

# Shap-like feature importance
gb_imp = pd.DataFrame({
    'Feature': X.columns,
    'Importancia': gb.feature_importances_
}).sort_values('Importancia', ascending=False)
print("\nFeature Importances (GB):")
print(gb_imp.to_string(index=False))

# ============================================================
# EJEMPLO 7: learning_rate vs n_estimators — trade-off
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 7: learning_rate vs n_estimators")
print("="*60)

configs = [
    (0.01, 500), (0.05, 200), (0.1, 100), (0.2, 50), (0.5, 20)
]
for lr, nest in configs:
    gb_lr = GradientBoostingClassifier(
        n_estimators=nest, learning_rate=lr, max_depth=3, random_state=42
    )
    gb_lr.fit(X_train, y_train)
    test_acc = (gb_lr.predict(X_test) == y_test).mean()
    print(f"lr={lr:.2f}, n_est={nest:3d} | test={test_acc:.4f}")

print("\nRegla: lr pequeño → más estimadores necesarios, mejor generalización")
print("lr grande → menos estimadores, pero riesgo de overfitting")

# ============================================================
# EJEMPLO 8: subsample — stochastic gradient boosting
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 8: subsample — Stochastic Gradient Boosting")
print("="*60)

for subsample in [0.5, 0.7, 0.9, 1.0]:
    gb_ss = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1,
        max_depth=3, subsample=subsample, random_state=42
    )
    gb_ss.fit(X_train, y_train)
    train_acc = (gb_ss.predict(X_train) == y_train).mean()
    test_acc = (gb_ss.predict(X_test) == y_test).mean()
    print(f"subsample={subsample:.1f} | train={train_acc:.4f} test={test_acc:.4f}")

print("\nsubsample < 1 → estocástico, reduce overfitting, aumenta robustez")

# ============================================================
# EJEMPLO 9: GradientBoostingRegressor — predecir ventas
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 9: GradientBoostingRegressor")
print("="*60)

gbr = GradientBoostingRegressor(
    n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
)
gbr.fit(Xr_train, yr_train)

yr_pred_gbr = gbr.predict(Xr_test)
print(f"R²: {r2_score(yr_test, yr_pred_gbr):.3f}")
print(f"MAE: {mean_absolute_error(yr_test, yr_pred_gbr):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(yr_test, yr_pred_gbr)):.2f}")
print(f"Comparación RF: R²={r2_score(yr_test, yr_pred):.3f}, MAE={mean_absolute_error(yr_test, yr_pred):.2f}")

# ============================================================
# EJEMPLO 10: AdaBoost — pesos adaptativos
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 10: AdaBoostClassifier")
print("="*60)

for n_est in [10, 50, 100, 200]:
    ada = AdaBoostClassifier(
        n_estimators=n_est, learning_rate=1.0, random_state=42
    )
    ada.fit(X_train, y_train)
    test_acc = (ada.predict(X_test) == y_test).mean()
    print(f"AdaBoost n_estimators={n_est:3d} | test={test_acc:.4f}")

print("\nAdaBoost asigna más peso a muestras mal clasificadas en cada iteración")
print("Sensible a outliers y ruido")

# ============================================================
# EJEMPLO 11: VotingClassifier hard — voto mayoritario
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 11: VotingClassifier — voto hard (mayoritario)")
print("="*60)

# Escalar datos para SVC y LR
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

voting_hard = VotingClassifier(estimators=[
    ('lr', LogisticRegression(max_iter=1000, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('svc', SVC(C=1.0, kernel='rbf', random_state=42, probability=False))
], voting='hard')

voting_hard.fit(X_train_s, y_train)
y_pred_vh = voting_hard.predict(X_test_s)

# Evaluar cada modelo individual
for nombre, model in voting_hard.named_estimators_.items():
    if nombre == 'rf':
        acc = (model.predict(X_test) == y_test).mean()
    else:
        acc = (model.predict(X_test_s) == y_test).mean()
    print(f"{nombre:10s} individual: {acc:.4f}")

print(f"Voting Hard:           {(y_pred_vh == y_test):.4f}")

# ============================================================
# EJEMPLO 12: VotingClassifier soft — promedio de probabilidades
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 12: VotingClassifier — voto soft (probabilidades)")
print("="*60)

voting_soft = VotingClassifier(estimators=[
    ('lr', LogisticRegression(max_iter=1000, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('svc', SVC(C=1.0, kernel='rbf', random_state=42, probability=True))
], voting='soft', weights=[1, 2, 1])  # RF tiene más peso

voting_soft.fit(X_train_s, y_train)
y_pred_vs = voting_soft.predict(X_test_s)

# Sin pesos
voting_soft_now = VotingClassifier(estimators=[
    ('lr', LogisticRegression(max_iter=1000, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('svc', SVC(C=1.0, kernel='rbf', random_state=42, probability=True))
], voting='soft')
voting_soft_now.fit(X_train_s, y_train)

print(f"Voting Soft (sin pesos): {(voting_soft_now.predict(X_test_s)==y_test).mean():.4f}")
print(f"Voting Soft (pesos 1,2,1): {(y_pred_vs==y_test).mean():.4f}")
print(f"Voting Hard:               {(y_pred_vh==y_test).mean():.4f}")

# ============================================================
# EJEMPLO 13: BaggingClassifier con base estimator personalizado
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 13: BaggingClassifier con base estimator")
print("="*60)

base_dt = DecisionTreeClassifier(max_depth=5, random_state=42)
bagging = BaggingClassifier(
    estimator=base_dt, n_estimators=100,
    max_samples=0.7, max_features=0.7,
    bootstrap=True, bootstrap_features=False,
    random_state=42, n_jobs=-1
)
bagging.fit(X_train, y_train)

print(f"Bagging (DT depth=5, 100 estim):")
print(f"  Train: {(bagging.predict(X_train)==y_train).mean():.4f}")
print(f"  Test:  {(bagging.predict(X_test)==y_test).mean():.4f}")

dt_simple = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_simple.fit(X_train, y_train)
print(f"DT simple (depth=5):")
print(f"  Train: {(dt_simple.predict(X_train)==y_train).mean():.4f}")
print(f"  Test:  {(dt_simple.predict(X_test)==y_test).mean():.4f}")

print("\nBagging reduce varianza del árbol simple (test mejora)")

# ============================================================
# EJEMPLO 14: StackingClassifier — meta-modelo
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 14: StackingClassifier")
print("="*60)

base_learners = [
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
    ('lr', LogisticRegression(max_iter=1000, random_state=42))
]

stacking = StackingClassifier(
    estimators=base_learners,
    final_estimator=LogisticRegression(random_state=42),
    cv=5
)
stacking.fit(X_train_s, y_train)

# También entrenar los base individualmente (en datos escalados para lr)
for nombre, modelo in base_learners:
    if nombre == 'lr':
        modelo.fit(X_train_s, y_train)
        acc = (modelo.predict(X_test_s) == y_test).mean()
    else:
        modelo.fit(X_train, y_train)
        acc = (modelo.predict(X_test) == y_test).mean()
    print(f"  {nombre:10s} individual: {acc:.4f}")

y_pred_st = stacking.predict(X_test_s)
print(f"  Stacking:             {(y_pred_st==y_test).mean():.4f}")

# ============================================================
# EJEMPLO 15: Feature importance en RandomForest — visualización
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 15: Feature importance en RandomForest (detallado)")
print("="*60)

rf_fi = RandomForestClassifier(n_estimators=200, random_state=42)
rf_fi.fit(X_train, y_train)

# Importancias con std (por la variabilidad entre árboles)
importancias = []
for tree in rf_fi.estimators_:
    importancias.append(tree.feature_importances_)
importancias = np.array(importancias)

fi_df = pd.DataFrame({
    'Feature': X.columns,
    'Mean': rf_fi.feature_importances_,
    'Std': importancias.std(axis=0)
}).sort_values('Mean', ascending=False)

print(fi_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(fi_df['Feature'], fi_df['Mean'], xerr=fi_df['Std'])
ax.set_xlabel('Importancia')
ax.set_title('Feature Importance (RF) con desviación entre árboles')
plt.tight_layout()
plt.savefig('I18_rf_importance.png', dpi=100)
plt.close()
print("Gráfico guardado: I18_rf_importance.png")

# ============================================================
# EJEMPLO 16: Comparar RF vs GB vs AdaBoost en accuracy
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 16: Comparación RF vs GB vs AdaBoost vs ExtraTrees")
print("="*60)

modelos_comp = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    'ExtraTrees': ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
}

resultados_comp = []
for nombre, model in modelos_comp.items():
    model.fit(X_train, y_train)
    train_acc = (model.predict(X_train) == y_train).mean()
    test_acc = (model.predict(X_test) == y_test).mean()
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    resultados_comp.append({
        'Modelo': nombre, 'Train': train_acc, 'Test': test_acc,
        'CV_mean': cv_scores.mean(), 'CV_std': cv_scores.std()
    })

res_df = pd.DataFrame(resultados_comp).sort_values('Test', ascending=False)
print(res_df.to_string(index=False))

# Gráfico comparativo
fig, ax = plt.subplots(figsize=(10, 5))
x_pos = np.arange(len(res_df))
ax.bar(x_pos - 0.2, res_df['Train'], 0.2, label='Train')
ax.bar(x_pos, res_df['Test'], 0.2, label='Test')
ax.bar(x_pos + 0.2, res_df['CV_mean'], 0.2, yerr=res_df['CV_std'], label='CV')
ax.set_xticks(x_pos)
ax.set_xticklabels(res_df['Modelo'])
ax.set_ylabel('Accuracy')
ax.set_title('Comparación de Métodos Ensemble')
ax.legend()
plt.tight_layout()
plt.savefig('I18_ensemble_comparison.png', dpi=100)
plt.close()
print("Gráfico guardado: I18_ensemble_comparison.png")

# ============================================================
# EJEMPLO 17: GridSearchCV para RandomForest
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 17: GridSearchCV para RandomForest")
print("="*60)

# Grid reducido por tiempo
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2']
}

gs_rf = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid_rf,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)
gs_rf.fit(X_train, y_train)

print(f"Mejores parámetros: {gs_rf.best_params_}")
print(f"Mejor score CV: {gs_rf.best_score_:.4f}")
print(f"Test accuracy: {(gs_rf.predict(X_test)==y_test).mean():.4f}")

# Mejor modelo vs default
rf_default = RandomForestClassifier(random_state=42)
rf_default.fit(X_train, y_train)
print(f"RF default - Test: {(rf_default.predict(X_test)==y_test).mean():.4f}")

# ============================================================
# EJEMPLO 18: Integrador — ensemble para predicción de demanda
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 18: Integrador — Ensemble para predicción de demanda")
print("="*60)

# Dataset combinado de ventas
n_ens = 800
np.random.seed(42)
df_ens = pd.DataFrame({
    'precio': np.random.uniform(10, 500, n_ens),
    'costo': np.random.uniform(5, 300, n_ens),
    'stock': np.random.randint(0, 500, n_ens),
    'rotacion': np.random.uniform(0, 100, n_ens),
    'num_clientes': np.random.poisson(50, n_ens),
    'temp_alta': np.random.choice([0, 1], n_ens, p=[0.6, 0.4]),
    'promocion': np.random.choice([0, 1], n_ens, p=[0.8, 0.2]),
    'dias_ult_venta': np.random.randint(0, 90, n_ens)
})

# Target continuo: demanda estimada
df_ens['demanda_est'] = (
    0.3 * (100 - df_ens['precio'] * 0.1)
    + 0.2 * df_ens['rotacion']
    + 0.15 * np.log1p(df_ens['num_clientes']) * 10
    + 0.15 * df_ens['temp_alta'] * 40
    + 0.1 * df_ens['promocion'] * 30
    - 0.1 * df_ens['dias_ult_venta'] * 0.5
    + 0.05 * np.log1p(df_ens['stock']) * 5
    + np.random.normal(0, 15, n_ens)
).clip(0)

# Target binario: demanda alta (percentil 60)
umbral_demanda = df_ens['demanda_est'].quantile(0.6)
df_ens['demanda_alta'] = (df_ens['demanda_est'] > umbral_demanda).astype(int)

X_ens = df_ens.drop(['demanda_est', 'demanda_alta'], axis=1)
y_ens_clf = df_ens['demanda_alta']
y_ens_reg = df_ens['demanda_est']

Xe_train, Xe_test, ye_train_clf, ye_test_clf = train_test_split(
    X_ens, y_ens_clf, test_size=0.3, random_state=42
)
_, _, ye_train_reg, ye_test_reg = train_test_split(
    X_ens, y_ens_reg, test_size=0.3, random_state=42
)

# Ensemble de clasificación
scaler_ens = StandardScaler()
Xe_train_s = scaler_ens.fit_transform(Xe_train)
Xe_test_s = scaler_ens.transform(Xe_test)

ensemble_clf = VotingClassifier(estimators=[
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('lr', LogisticRegression(max_iter=1000, random_state=42))
], voting='soft')
ensemble_clf.fit(Xe_train_s, ye_train_clf)

print("=== Ensemble Clasificación (demanda_alta) ===")
print(f"Accuracy: {(ensemble_clf.predict(Xe_test_s)==ye_test_clf).mean():.4f}")
print(f"\nReporte de clasificación:")
print(classification_report(ye_test_clf, ensemble_clf.predict(Xe_test_s),
      target_names=['Baja', 'Alta']))

# Ensemble de regresión
ensemble_reg = VotingClassifier(estimators=[
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
    ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42))
], voting='soft')
ensemble_reg.fit(Xe_train, ye_train_reg)

print("\n=== Ensemble Regresión (demanda estimada) ===")
ye_pred_ens = ensemble_reg.predict(Xe_test)
print(f"R²: {r2_score(ye_test_reg, ye_pred_ens):.3f}")
print(f"MAE: {mean_absolute_error(ye_test_reg, ye_pred_ens):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(ye_test_reg, ye_pred_ens)):.2f}")

# Feature importances del mejor modelo dentro del ensemble
rf_best = ensemble_clf.named_estimators_['rf']
fi_ens = pd.DataFrame({
    'Feature': X_ens.columns,
    'Importancia': rf_best.feature_importances_
}).sort_values('Importancia', ascending=False)
print("\nFeature Importances (RF del ensemble):")
print(fi_ens.to_string(index=False))

print("\n=== FIN EJEMPLOS ENSEMBLE BÁSICO ===")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*2. Ejemplos Prácticos con sklearn.*

1. Configuración inicial
2. ============================================================
3. EJEMPLO 1: RandomForestClassifier — clasificar + feature_importance
4. ============================================================
5. Ruido
6. Feature importances
7. ============================================================
8. EJEMPLO 2: RandomForestRegressor — predecir demanda

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

## 3. Ejercicios Propuestos

1. **RandomForest vs GradientBoosting**: En el dataset de demanda (Ejemplo 2), compara RF y GB variando `n_estimators` de 10 a 500 con paso 20. Grafica las curvas de test error. ¿Cuál converge más rápido? ¿Cuál da mejor resultado final?

2. **Bagging con diferentes bases**: Usa BaggingClassifier con base estimators: DecisionTree (depth=3), LogisticRegression y KNeighborsClassifier. ¿Cuál se beneficia más del bagging? ¿Por qué?

3. **Optimización de Stacking**: En el Ejemplo 14, prueba diferentes meta-modelos (RF, SVC, LogisticRegression, GradientBoosting) como `final_estimator`. ¿Cuál da mejor accuracy? ¿Cuál es más rápido?

4. **Subsample y learning rate**: En GradientBoosting, haz un grid 5×5 de `subsample ∈ [0.3, 0.5, 0.7, 0.9, 1.0]` vs `learning_rate ∈ [0.01, 0.05, 0.1, 0.2, 0.5]`. ¿Qué combinación minimiza el error de test? ¿Hay interacción entre ambos parámetros?

5. **Out-of-Bag para selección de features**: Usa `oob_score=True` en RF. Entrena modelos con diferentes subconjuntos de features (top-k por importancia). ¿Cuál es el mínimo k que mantiene el OOB score > 95% del máximo?

6. **Comparación con y sin scaling**: En VotingClassifier, entrena combinaciones de modelos que requieren scaling (SVC, LR) y los que no (RF, GB). ¿Usar `StandardScaler` en todo el pipeline mejora o empeora? ¿Por qué?

7. **Tiempo de entrenamiento**: Mide el tiempo de `fit` para RF, GB, AdaBoost y ExtraTrees con `n_estimators=500`. Usa `time` de Python. ¿Cuál es más rápido? ¿La velocidad se correlaciona con el rendimiento?

8. **Ensemble para detección de anomalías**: Crea un dataset con 5% de outliers (ventas anómalas). Entrena IsolationForest y compara con un VotingClassifier que incluye RF+GB+LR. ¿Qué métrica usarías para evaluar detección de anomalías?

---

## 4. Resumen

- **RandomForest**: Bagging + árboles. Robusto, paralelizable, difícil de overfittear. Bueno por defecto.
- **GradientBoosting**: Secuencial, corrige errores previos. Mejor precisión pero más parámetros para tunear.
- **AdaBoost**: Pesos adaptativos. Sencillo pero sensible a outliers.
- **ExtraTrees**: Aleatoriedad extrema → menor varianza, más sesgo.
- **VotingClassifier**: Combina modelos independientes. Hard (voto) vs Soft (promedio de probabilidades).
- **BaggingClassifier**: Reduce varianza de cualquier modelo base.
- **StackingClassifier**: Meta-modelo aprende a combinar predictores base.
- **Out-of-Bag**: Evaluación interna en RandomForest sin validación cruzada.
- **Feature importance**: Más estable en ensembles que en árboles individuales.

### Cuándo usar ensembles
- Necesitas la **mejor precisión posible** sin importar la interpretabilidad
- Tienes **suficientes datos** para entrenar múltiples modelos
- Quieres reducir **varianza** (bagging) o **sesgo** (boosting)
- El problema es complejo con relaciones no lineales

### Limitaciones
- **Menos interpretables** que modelos simples
- Mayor **costo computacional** (especialmente con muchos estimadores)
- Más **hiperparámetros** para optimizar
- Boosting puede **overfittear** si learning_rate es muy alto o n_estimators excesivo
- Stacking puede generar **overfitting** si no se usa CV en la generación de features
