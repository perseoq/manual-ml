# I19: SVM (Support Vector Machines) con scikit-learn

## 1. Fundamentos Teóricos

Las **Support Vector Machines (SVM)** buscan el hiperplano que maximiza el margen entre clases. Los puntos más cercanos al hiperplano se llaman **vectores de soporte**.

### Kernels y transformaciones
| Kernel | Fórmula | Uso |
|--------|---------|-----|
| Linear | `K(x,y) = x·y` | Datos linealmente separables |
| RBF | `K(x,y) = exp(-γ‖x-y‖²)` | Relaciones no lineales (por defecto) |
| Poly | `K(x,y) = (γ x·y + r)^d` | Fronteras polinómicas |
| Sigmoid | `K(x,y) = tanh(γ x·y + r)` | Similar a redes neuronales |

### Parámetros clave
| Parámetro | Descripción |
|-----------|-------------|
| C | Regularización (menor C = margen más suave) |
| gamma | Influencia de cada punto (menor = más amplia) |
| degree | Grado del kernel polinómico |
| nu | Fracción de errores permitidos (NuSVC) |
| class_weight | Balance de clases |

### Variantes de SVM en sklearn
- **SVC**: Clasificación con kernel explícito
- **LinearSVC**: Versión rápida para kernel lineal (grandes datasets)
- **SVR**: Regresión con SVM
- **NuSVC**: Control de errores con parámetro nu
- **OneClassSVM**: Detección de anomalías (no supervisado)

### IMPORTANTE: Escalado
SVM es **extremadamente sensible** a la escala de los datos. **Siempre** usar StandardScaler o MinMaxScaler antes de SVM.

---

## 2. Ejemplos Prácticos con sklearn

```python
# Configuración inicial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC, SVR, NuSVC, LinearSVC, OneClassSVM
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.datasets import make_classification, make_circles, make_moons
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================
# EJEMPLO 1: SVC lineal — clasificar productos (precio, margen)
# ============================================================
print("="*60)
print("EJEMPLO 1: SVC lineal — clasificar productos")
print("="*60)

n = 400
df = pd.DataFrame({
    'precio': np.random.uniform(10, 500, n),
    'margen': np.random.uniform(5, 40, n),
    'stock': np.random.randint(0, 200, n),
    'rotacion': np.random.uniform(0, 100, n)
})
df['producto_top'] = (
    (df['margen'] > 20) & (df['rotacion'] > 30) |
    (df['precio'] > 300) & (df['margen'] > 15)
).astype(int)
mask_ruido = np.random.random(n) < 0.1
df.loc[mask_ruido, 'producto_top'] = 1 - df.loc[mask_ruido, 'producto_top']

X = df.drop('producto_top', axis=1)
y = df['producto_top']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

svc_linear = SVC(kernel='linear', C=1.0, random_state=42)
svc_linear.fit(X_train_s, y_train)

y_pred = svc_linear.predict(X_test_s)
print(f"Accuracy (linear): {(y_pred == y_test).mean():.3f}")
print(f"Vectores de soporte: {svc_linear.n_support_}")

# ============================================================
# EJEMPLO 2: SVC rbf — límite no lineal (gamma=0.1, 1, 10)
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 2: SVC con kernel RBF — efecto de gamma")
print("="*60)

for gamma in [0.1, 1.0, 10.0]:
    svc_rbf = SVC(kernel='rbf', C=1.0, gamma=gamma, random_state=42)
    svc_rbf.fit(X_train_s, y_train)
    train_acc = (svc_rbf.predict(X_train_s) == y_train).mean()
    test_acc = (svc_rbf.predict(X_test_s) == y_test).mean()
    n_sv = len(svc_rbf.support_)
    print(f"gamma={gamma:5.1f} | train={train_acc:.3f} test={test_acc:.3f} | SV={n_sv}")

print("Gamma pequeno -> limite mas suave, menos SV, posible underfitting")
print("Gamma grande -> limite mas ajustado, mas SV, posible overfitting")

# ============================================================
# EJEMPLO 3: SVC poly — limite polinomico (degree=2,3,4)
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 3: SVC con kernel polinomial")
print("="*60)

for degree in [2, 3, 4]:
    svc_poly = SVC(kernel='poly', degree=degree, C=1.0, gamma='scale', random_state=42)
    svc_poly.fit(X_train_s, y_train)
    test_acc = (svc_poly.predict(X_test_s) == y_test).mean()
    n_sv = len(svc_poly.support_)
    print(f"degree={degree} | test={test_acc:.3f} | SV={n_sv}/{len(X_train_s)}")

print("Mayor degree -> mas flexible, mayor riesgo de overfitting")

# ============================================================
# EJEMPLO 4: C — regularizacion (C=0.1, 1, 10, 100)
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 4: Efecto de C en SVC")
print("="*60)

for C in [0.1, 1.0, 10.0, 100.0]:
    svc_c = SVC(kernel='rbf', C=C, gamma='scale', random_state=42)
    svc_c.fit(X_train_s, y_train)
    train_acc = (svc_c.predict(X_train_s) == y_train).mean()
    test_acc = (svc_c.predict(X_test_s) == y_test).mean()
    n_sv = len(svc_c.support_)
    print(f"C={C:5.1f} | train={train_acc:.3f} test={test_acc:.3f} | SV={n_sv}")

print("C pequeno -> margen mas amplio, mas SV, menor riesgo overfitting")
print("C grande -> margen mas estricto, menos SV, mayor riesgo overfitting")

# ============================================================
# EJEMPLO 5: gamma — efecto en la forma del limite
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 5: Efecto completo de gamma")
print("="*60)

X_circ, y_circ = make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=42)
Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_circ, y_circ, test_size=0.3, random_state=42)
scaler_circ = StandardScaler()
Xc_train_s = scaler_circ.fit_transform(Xc_train)
Xc_test_s = scaler_circ.transform(Xc_test)

for gamma in [0.01, 0.1, 1.0, 10.0, 100.0]:
    svc_g = SVC(kernel='rbf', C=1.0, gamma=gamma, random_state=42)
    svc_g.fit(Xc_train_s, yc_train)
    test_acc = (svc_g.predict(Xc_test_s) == yc_test).mean()
    n_sv = len(svc_g.support_)
    print(f"gamma={gamma:6.2f} | test={test_acc:.3f} | SV={n_sv}/{len(Xc_train_s)} ({100*n_sv/len(Xc_train_s):.0f}%)")

print("Gamma demasiado alto -> overfitting (cada punto es una isla)")

# ============================================================
# EJEMPLO 6: SVR — regresion con SVM para demanda
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 6: SVR — regresion para predecir demanda")
print("="*60)

n_svr = 300
df_svr = pd.DataFrame({
    'precio': np.random.uniform(10, 300, n_svr),
    'gasto_mkt': np.random.uniform(100, 5000, n_svr),
    'temp_alta': np.random.choice([0, 1], n_svr, p=[0.7, 0.3]),
    'competidores': np.random.randint(0, 8, n_svr)
})
df_svr['demanda'] = (
    80 - 0.2*df_svr['precio'] + 0.015*df_svr['gasto_mkt']
    + 25*df_svr['temp_alta'] - 4*df_svr['competidores']
    + np.random.normal(0, 12, n_svr)
).clip(0)

X_svr = df_svr.drop('demanda', axis=1)
y_svr = df_svr['demanda']
X_svr_train, X_svr_test, y_svr_train, y_svr_test = train_test_split(
    X_svr, y_svr, test_size=0.3, random_state=42
)

scaler_svr = StandardScaler()
X_svr_train_s = scaler_svr.fit_transform(X_svr_train)
X_svr_test_s = scaler_svr.transform(X_svr_test)

for kernel in ['linear', 'rbf', 'poly']:
    svr = SVR(kernel=kernel, C=1.0, gamma='scale')
    svr.fit(X_svr_train_s, y_svr_train)
    yp = svr.predict(X_svr_test_s)
    r2 = r2_score(y_svr_test, yp)
    mae = mean_absolute_error(y_svr_test, yp)
    print(f"kernel={kernel:8s} | R2={r2:.3f} | MAE={mae:.2f}")

# ============================================================
# EJEMPLO 7: NuSVC — control de errores con nu
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 7: NuSVC — control de errores con nu")
print("="*60)

for nu in [0.1, 0.3, 0.5, 0.7, 0.9]:
    nusvc = NuSVC(nu=nu, kernel='rbf', gamma='scale', random_state=42)
    nusvc.fit(X_train_s, y_train)
    test_acc = (nusvc.predict(X_test_s) == y_test).mean()
    n_sv = len(nusvc.support_)
    print(f"nu={nu:.1f} | test={test_acc:.3f} | SV={n_sv}/{len(X_train_s)} ({100*n_sv/len(X_train_s):.0f}%)")

print("nu: fraccion aproximada de vectores de soporte (limite superior)")

# ============================================================
# EJEMPLO 8: LinearSVC — version rapida para grandes datasets
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 8: LinearSVC — rapido para datos grandes")
print("="*60)

from sklearn.datasets import make_classification
import time

X_large, y_large = make_classification(
    n_samples=5000, n_features=20, n_informative=10,
    n_redundant=5, random_state=42
)
Xl_train, Xl_test, yl_train, yl_test = train_test_split(
    X_large, y_large, test_size=0.3, random_state=42
)

for nombre, model in [
    ('LinearSVC', LinearSVC(max_iter=2000, random_state=42, dual='auto')),
    ('SVC(kernel=linear)', SVC(kernel='linear', random_state=42))
]:
    start = time.time()
    model.fit(Xl_train, yl_train)
    elapsed = time.time() - start
    test_acc = (model.predict(Xl_test) == yl_test).mean()
    print(f"{nombre:20s} | time={elapsed:.3f}s | test={test_acc:.4f}")

print("LinearSVC escala mejor con n_samples O(n) vs SVC O(n^2) o O(n^3)")

# ============================================================
# EJEMPLO 9: OneClassSVM — deteccion de anomalias en ventas
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 9: OneClassSVM — deteccion de anomalias")
print("="*60)

n_norm = 200
X_normal = np.column_stack([
    np.random.normal(100, 20, n_norm),
    np.random.normal(30, 8, n_norm),
    np.random.normal(0.3, 0.05, n_norm),
])

n_anom = 20
X_anom = np.column_stack([
    np.random.uniform(200, 500, n_anom),
    np.random.uniform(5, 15, n_anom),
    np.random.uniform(0.01, 0.05, n_anom),
])

X_ocsvm = np.vstack([X_normal, X_anom])
scaler_ocsvm = StandardScaler()
X_ocsvm_s = scaler_ocsvm.fit_transform(X_ocsvm)

ocsvm = OneClassSVM(nu=0.05, kernel='rbf', gamma='scale')
ocsvm.fit(X_ocsvm_s)

y_pred_ocsvm = ocsvm.predict(X_ocsvm_s)
n_anomalias = (y_pred_ocsvm == -1).sum()
print(f"Anomalias detectadas: {n_anomalias} de {len(X_ocsvm)} ({100*n_anomalias/len(X_ocsvm):.1f}%)")
print(f"Anomalias reales: {n_anom}")

scores_ocsvm = ocsvm.score_samples(X_ocsvm_s)
top_anomalias = np.argsort(scores_ocsvm)[:10]
print("Top 10 anomalias detectadas (menor score = mas anomalo):")
for i, idx in enumerate(top_anomalias):
    etiqueta = "Anomalia" if idx >= n_norm else "Normal (FP)"
    print(f"  #{i+1}: muestra {idx} - score={scores_ocsvm[idx]:.2f} - {etiqueta}")

# ============================================================
# EJEMPLO 10: probability=True + predict_proba
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 10: SVC con probabilidades")
print("="*60)

svc_prob = SVC(kernel='rbf', C=1.0, gamma='scale',
               probability=True, random_state=42)
svc_prob.fit(X_train_s, y_train)

probas = svc_prob.predict_proba(X_test_s)[:10]
print("Probabilidades (primeros 10):")
for i, (p0, p1) in enumerate(probas):
    pred = svc_prob.predict(X_test_s)[i]
    real = y_test.values[i]
    print(f"  {i}: P(y=0)={p0:.3f}, P(y=1)={p1:.3f} -> pred={pred}, real={real}")

# ============================================================
# EJEMPLO 11: Support vectors — visualizar vectores de soporte
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 11: Visualizar vectores de soporte")
print("="*60)

X_2d = df[['precio', 'margen']].values
y_2d = df['producto_top'].values
X_2d_train, X_2d_test, y_2d_train, y_2d_test = train_test_split(
    X_2d, y_2d, test_size=0.3, random_state=42
)
scaler_2d = StandardScaler()
X_2d_train_s = scaler_2d.fit_transform(X_2d_train)
X_2d_test_s = scaler_2d.transform(X_2d_test)

svc_2d = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svc_2d.fit(X_2d_train_s, y_2d_train)

print(f"Vectores de soporte: {len(svc_2d.support_)} de {len(X_2d_train_s)}")
print(f"SV por clase: {svc_2d.n_support_}")

# Grid para visualizacion
xx, yy = np.meshgrid(
    np.linspace(X_2d_train_s[:, 0].min()-0.5, X_2d_train_s[:, 0].max()+0.5, 200),
    np.linspace(X_2d_train_s[:, 1].min()-0.5, X_2d_train_s[:, 1].max()+0.5, 200)
)
Z = svc_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(8, 6))
ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
ax.scatter(X_2d_train_s[:, 0], X_2d_train_s[:, 1], c=y_2d_train,
           cmap='coolwarm', alpha=0.4, s=30, edgecolors='gray')
ax.scatter(X_2d_train_s[svc_2d.support_, 0], X_2d_train_s[svc_2d.support_, 1],
           c='none', s=150, edgecolors='yellow', linewidth=2, label='Vectores Soporte')
ax.set_xlabel('Precio (std)')
ax.set_ylabel('Margen (std)')
ax.set_title(f'SVC RBF - Vectores de Soporte ({len(svc_2d.support_)})')
ax.legend()
plt.tight_layout()
plt.savefig('I19_support_vectors.png', dpi=100)
plt.close()
print("Grafico guardado: I19_support_vectors.png")

# ============================================================
# EJEMPLO 12: decision_function — distancia al hiperplano
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 12: decision_function — distancia al hiperplano")
print("="*60)

svc_df = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svc_df.fit(X_train_s, y_train)

distancias = svc_df.decision_function(X_test_s)
print(f"decision_function (primeros 15): {np.round(distancias[:15], 3)}")
print(f"predict (primeros 15):           {svc_df.predict(X_test_s)[:15]}")
print(f"y real (primeros 15):            {y_test.values[:15]}")

confianza = np.abs(distancias)
print(f"Casos confiables (|dist|>1.5): {(confianza>1.5).sum()} de {len(confianza)}")
print(f"Casos dudosos (|dist|<0.5):  {(confianza<0.5).sum()} de {len(confianza)}")

# ============================================================
# EJEMPLO 13: Escalado con StandardScaler ANTES de SVM
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 13: Importancia del escalado en SVM")
print("="*60)

X_scale = pd.DataFrame({
    'precio': np.random.uniform(10, 500, 300),
    'unidades': np.random.randint(1, 10000, 300),
    'tasa': np.random.uniform(0.01, 0.99, 300),
    'puntaje': np.random.uniform(0, 1000, 300)
})
y_scale = (X_scale['precio']*0.01 + X_scale['unidades']*0.0001
           - X_scale['tasa']*0.5 + X_scale['puntaje']*0.001
           + np.random.normal(0, 0.3, 300) > 0.5).astype(int)

Xs_train, Xs_test, ys_train, ys_test = train_test_split(
    X_scale, y_scale, test_size=0.3, random_state=42
)

svc_no = SVC(kernel='rbf', random_state=42)
svc_no.fit(Xs_train, ys_train)
acc_no = (svc_no.predict(Xs_test) == ys_test).mean()

scaler_x = StandardScaler()
Xs_train_s = scaler_x.fit_transform(Xs_train)
Xs_test_s = scaler_x.transform(Xs_test)
svc_sc = SVC(kernel='rbf', random_state=42)
svc_sc.fit(Xs_train_s, ys_train)
acc_sc = (svc_sc.predict(Xs_test_s) == ys_test).mean()

print(f"Sin escalar:  {acc_no:.3f}")
print(f"Con escalado: {acc_sc:.3f}")
print(f"Diferencia:   {acc_sc-acc_no:+.3f}")
print("CRITICO: SVM depende de distancias. !Siempre escalar!")

# ============================================================
# EJEMPLO 14: Pipeline — StandardScaler + SVC
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 14: Pipeline StandardScaler + SVC")
print("="*60)

pipeline_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42))
])

pipeline_svm.fit(X_train, y_train)
y_pred_pipe = pipeline_svm.predict(X_test)
print(f"Pipeline accuracy: {(y_pred_pipe == y_test).mean():.3f}")
print(pipeline_svm)
print("Pipeline garantiza escalado correcto en train y test")

# ============================================================
# EJEMPLO 15: GridSearchCV — busqueda de C, gamma, kernel
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 15: GridSearchCV para SVM")
print("="*60)

param_grid_svm = {
    'svc__C': [0.1, 1, 10],
    'svc__gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'svc__kernel': ['rbf', 'linear', 'poly']
}

pipe_gs = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(random_state=42))
])

gs_svm = GridSearchCV(
    pipe_gs, param_grid_svm, cv=3, scoring='accuracy', n_jobs=-1, verbose=0
)
gs_svm.fit(X_train, y_train)

print(f"Mejores parametros: {gs_svm.best_params_}")
print(f"Mejor score CV: {gs_svm.best_score_:.4f}")
print(f"Test accuracy: {(gs_svm.predict(X_test)==y_test).mean():.4f}")

results_svm = pd.DataFrame(gs_svm.cv_results_)
top5_svm = results_svm.sort_values('rank_test_score').head(5)
print("Top 5 combinaciones:")
for i, row in top5_svm.iterrows():
    print(f"  {row['params']} | mean={row['mean_test_score']:.4f} | std={row['std_test_score']:.4f}")

# ============================================================
# EJEMPLO 16: Comparar kernel rbf vs poly vs lineal
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 16: Comparacion kernels")
print("="*60)

kernels = {
    'rbf': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
    'poly (deg=3)': SVC(kernel='poly', degree=3, C=1.0, gamma='scale', random_state=42),
    'poly (deg=2)': SVC(kernel='poly', degree=2, C=1.0, gamma='scale', random_state=42),
    'linear': SVC(kernel='linear', C=1.0, random_state=42),
    'sigmoid': SVC(kernel='sigmoid', C=1.0, gamma='scale', random_state=42),
}

for nombre, model in kernels.items():
    model.fit(X_train_s, y_train)
    train_acc = (model.predict(X_train_s) == y_train).mean()
    test_acc = (model.predict(X_test_s) == y_test).mean()
    n_sv = len(model.support_)
    print(f"{nombre:15s} | train={train_acc:.3f} test={test_acc:.3f} | SV={n_sv}")

print("RBF suele ser el mejor kernel por defecto para datos no lineales")

# ============================================================
# EJEMPLO 17: Clases desbalanceadas con class_weight
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 17: class_weight para clases desbalanceadas")
print("="*60)

df_desb = df.copy()
mask_top = df_desb['producto_top'] == 1
drop_idx = mask_top[mask_top].sample(frac=0.6, random_state=42).index
df_desb = df_desb.drop(drop_idx)
print(f"Clase 0: {(df_desb['producto_top']==0).sum()}, Clase 1: {(df_desb['producto_top']==1).sum()}")

Xd = df_desb.drop('producto_top', axis=1)
yd = df_desb['producto_top']
Xd_train, Xd_test, yd_train, yd_test = train_test_split(Xd, yd, test_size=0.3, random_state=42)
scaler_desb = StandardScaler()
Xd_train_s = scaler_desb.fit_transform(Xd_train)
Xd_test_s = scaler_desb.transform(Xd_test)

for cw in [None, 'balanced']:
    svc_cw = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight=cw, random_state=42)
    svc_cw.fit(Xd_train_s, yd_train)
    print(f"\nclass_weight={cw}:")
    print(classification_report(yd_test, svc_cw.predict(Xd_test_s)))

# ============================================================
# EJEMPLO 18: Integrador — clasificar calidad de proveedores
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 18: Integrador — clasificar calidad de proveedores")
print("="*60)

n_prov = 500
np.random.seed(42)
df_prov = pd.DataFrame({
    'precio_prom': np.random.uniform(50, 500, n_prov),
    'calidad': np.random.uniform(1, 10, n_prov),
    'lead_time': np.random.uniform(1, 30, n_prov),
    'cumplimiento': np.random.uniform(50, 100, n_prov),
    'volumen': np.random.uniform(100, 10000, n_prov),
    'reclamos': np.random.poisson(3, n_prov),
})

score_prov = (
    0.3*df_prov['calidad'] + 0.3*(df_prov['cumplimiento']/100)
    - 0.2*(df_prov['lead_time']/30) - 0.1*np.log1p(df_prov['reclamos'])/3
)
prob_prov = 1/(1+np.exp(-3*(score_prov-0.5)))
df_prov['bueno'] = (prob_prov + np.random.normal(0, 0.1, n_prov) > 0.5).astype(int)
print(f"Buenos: {df_prov['bueno'].sum()}, Malos: {(df_prov['bueno']==0).sum()}")

Xp = df_prov.drop('bueno', axis=1)
yp = df_prov['bueno']
Xp_train, Xp_test, yp_train, yp_test = train_test_split(Xp, yp, test_size=0.3, random_state=42)

# Pipeline con GridSearch optimizado
pipe_prov = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(random_state=42))
])
grid_prov = {
    'svc__C': [0.1, 1, 10],
    'svc__gamma': ['scale', 'auto', 0.1],
    'svc__kernel': ['rbf', 'linear']
}
gs_prov = GridSearchCV(pipe_prov, grid_prov, cv=5, scoring='f1', n_jobs=-1)
gs_prov.fit(Xp_train, yp_train)

print(f"Mejores params: {gs_prov.best_params_}")
print(f"F1 CV: {gs_prov.best_score_:.3f}")
print(f"Test accuracy: {(gs_prov.predict(Xp_test)==yp_test).mean():.3f}")
print("\nReporte:")
print(classification_report(yp_test, gs_prov.predict(Xp_test), target_names=['Malo', 'Bueno']))

# Interpretar vectores de soporte del mejor modelo
best_svm = gs_prov.best_estimator_.named_steps['svc']
n_sv_prov = len(best_svm.support_)
print(f"\nVectores de soporte del mejor modelo: {n_sv_prov}/{len(Xp_train)} ({100*n_sv_prov/len(Xp_train):.1f}%)")

print("\n=== FIN EJEMPLOS SVM ===")
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
3. EJEMPLO 1: SVC lineal — clasificar productos (precio, margen)
4. ============================================================
5. ============================================================
6. EJEMPLO 2: SVC rbf — límite no lineal (gamma=0.1, 1, 10)
7. ============================================================
8. ============================================================

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

## 3. Ejercicios Propuestos

1. **Kernel personalizado**: Crea un kernel personalizado usando `sklearn.metrics.pairwise.pairwise_kernels` y úsalo en SVC. Compáralo con RBF en el dataset de productos.

2. **SVR vs GradientBoostingRegressor**: En el dataset de demanda (Ejemplo 6), compara SVR (con diferentes kernels) vs GradientBoostingRegressor. ¿Cuál da mejor R²? ¿Cuál es más rápido?

3. **NuSVC vs SVC**: En el dataset original, compara NuSVC(nu=0.3) vs SVC(C=1.0). ¿Cómo se relacionan nu y C? Varía nu de 0.1 a 0.9.

4. **OneClassSVM para fraude**: Crea un dataset de transacciones con 3% de fraudes. Usa OneClassSVM para detectarlos. ¿Qué nu elegirías? Calcula recall de fraudes.

5. **Escalado con RobustScaler**: En SVM, los outliers afectan más que en otros modelos. Compara StandardScaler vs RobustScaler en un dataset con outliers.

6. **LinearSVC con L1 penalty**: LinearSVC soporta penalty='l1'. Úsalo para selección de features en el dataset de proveedores. ¿Qué features quedan con coeficiente no cero?

7. **Calibración de probabilidades SVM**: Usa `CalibratedClassifierCV` sobre SVC(probability=True). Compara las curvas de calibración antes y después.

8. **SVM con PCA**: Crea un pipeline PCA(n_components=0.95) + SVC(rbf). Compáralo con SVC sin PCA. ¿La reducción de dimensionalidad mejora o empeora?

---

## 4. Resumen

- **SVM** maximiza el margen entre clases usando **vectores de soporte**
- **Kernel RBF** es el más usado por defecto (maneja no linealidades)
- **C** controla regularización: menor C = margen más suave
- **gamma** controla la influencia de cada punto: menor gamma = límite más suave
- **Siempre escalar datos** antes de SVM (critical)
- **SVR** extiende SVM a regresión
- **OneClassSVM** detecta anomalías (no supervisado)
- **LinearSVC** es más rápido que SVC(kernel='linear') para datasets grandes
- **NuSVC** usa nu en lugar de C para controlar errores
- Los **vectores de soporte** son los puntos que definen el hiperplano

### Cuándo usar SVM
- Datos de **dimensión media** (10-1000 features)
- Problemas donde la **separación clara** entre clases es posible
- Necesitas un modelo con **buena generalización** con pocos datos
- **Datos no lineales** (con kernel RBF)

### Limitaciones
- **No escala bien** con n_samples > 10,000 (complejidad O(n²) a O(n³))
- Sensible a **escalado** (requiere preprocesamiento cuidadoso)
- No da **probabilidades** por defecto (probability=True es costoso)
- Difícil de **interpretar** (especialmente con kernels no lineales)
- Elegir **kernel y parámetros** requiere validación cuidadosa
