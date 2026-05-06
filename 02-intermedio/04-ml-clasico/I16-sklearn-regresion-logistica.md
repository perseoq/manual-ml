# I16: Regresión Logística con scikit-learn

## 1. Fundamentos Teóricos

La **regresión logística** es un modelo lineal para clasificación binaria (y multiclase) que modela la probabilidad de pertenencia a una clase mediante la función sigmoide:

```
P(y=1|x) = 1 / (1 + e^(-z))    donde z = w·x + b
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*1. Fundamentos Teóricos.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Salidas del modelo
- **predict()**: clase {0, 1}
- **predict_proba()**: probabilidad [P(y=0), P(y=1)]
- **decision_function()**: distancia al hiperplano z = w·x + b

### Regularización
- **L1 (lasso)**: penaliza suma de |coeficientes| → selección de features
- **L2 (ridge)**: penaliza suma de coeficientes² → evita sobreajuste
- **ElasticNet**: combinación de L1 y L2

### Parámetros clave
| Parámetro | Descripción |
|-----------|-------------|
| penalty | Tipo de regularización: 'l1', 'l2', 'elasticnet' |
| C | Inversa de fuerza de regularización (menor C = más regularización) |
| solver | Algoritmo de optimización: 'lbfgs', 'liblinear', 'saga' |
| multi_class | 'ovr' (uno vs resto) o 'multinomial' (softmax) |
| class_weight | Peso de clases: None, 'balanced' o dict |
| max_iter | Iteraciones máximas para convergencia |

---

## 2. Ejemplos Prácticos con sklearn

```python
# Configuración inicial común
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_curve, auc, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             RocCurveDisplay, precision_recall_curve)
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================
# EJEMPLO 1: Binaria — clasificar si producto se vende hoy
# ============================================================
print("="*60)
print("EJEMPLO 1: Clasificación binaria — ¿Se vende hoy?")
print("="*60)

# Dataset simulado de productos
np.random.seed(42)
n = 500
df = pd.DataFrame({
    'precio': np.random.uniform(10, 500, n),
    'margen': np.random.uniform(5, 40, n),
    'stock': np.random.randint(0, 200, n),
    'dias_ult_venta': np.random.randint(0, 60, n),
    'promocion': np.random.choice([0, 1], n, p=[0.7, 0.3])
})
# Target: se_vende_hoy (1=sí, 0=no) — regla subyacente simulada
prob_venta = 1 / (1 + np.exp(-(
    -2 + 0.005*df['precio'] + 0.03*df['margen']
    + 0.002*df['stock'] - 0.04*df['dias_ult_venta']
    + 0.5*df['promocion'] + np.random.normal(0, 0.5, n)
)))
df['se_vende_hoy'] = (prob_venta > 0.5).astype(int)

X = df.drop('se_vende_hoy', axis=1)
y = df['se_vende_hoy']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
accuracy = (y_pred == y_test).mean()
print(f"Accuracy: {accuracy:.3f}")
print(f"Coeficientes: {dict(zip(X.columns, model.coef_[0]))}")
print(f"Intercepto: {model.intercept_[0]:.4f}")

# ============================================================
# EJEMPLO 2: Probabilidades con predict_proba y umbral personalizado
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 2: Probabilidades y umbral personalizado")
print("="*60)

probas = model.predict_proba(X_test_scaled)
prob_positivo = probas[:, 1]

# Umbral personalizado: solo clasificamos como "se vende" si prob > 0.7
umbral = 0.7
y_pred_custom = (prob_positivo >= umbral).astype(int)

print(f"Probabilidades (primeros 10): {prob_positivo[:10].round(3)}")
print(f"Predicción con umbral 0.5:    {model.predict(X_test_scaled)[:10]}")
print(f"Predicción con umbral {umbral}: {y_pred_custom[:10]}")

# Decisión: si probabilidad entre 0.3 y 0.7, marcar como "incierto"
incierto = (prob_positivo > 0.3) & (prob_positivo < 0.7)
print(f"Casos inciertos (0.3 < prob < 0.7): {incierto.sum()} de {len(prob_positivo)}")

# ============================================================
# EJEMPLO 3: Regularización C — comparar C = 0.01, 0.1, 1, 10, 100
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 3: Comparación de C (inversa de regularización)")
print("="*60)

valores_C = [0.01, 0.1, 1, 10, 100]
resultados_C = []

for C in valores_C:
    lr = LogisticRegression(C=C, random_state=42, max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    train_acc = (lr.predict(X_train_scaled) == y_train).mean()
    test_acc = (lr.predict(X_test_scaled) == y_test).mean()
    n_ceros = (np.abs(lr.coef_[0]) < 1e-6).sum()
    resultados_C.append({
        'C': C, 'train_acc': train_acc, 'test_acc': test_acc,
        'n_ceros': n_ceros, 'coef_norm': np.linalg.norm(lr.coef_[0])
    })
    print(f"C={C:6.2f} | train={train_acc:.3f} test={test_acc:.3f} "
          f"| coef_cero={n_ceros} | ||coef||={np.linalg.norm(lr.coef_[0]):.3f}")

print("\nConclusión: C pequeño → más regularización → coeficientes más pequeños")
print("C grande → menos regularización → coeficientes más grandes (posible overfitting)")

# ============================================================
# EJEMPLO 4: Penalty L1 — selección de features (coeficientes a cero)
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 4: Penalty L1 — selección automática de features")
print("="*60)

# L1 requiere solver 'liblinear' o 'saga'
lr_l1 = LogisticRegression(penalty='l1', solver='saga', C=0.1,
                           random_state=42, max_iter=2000)
lr_l1.fit(X_train_scaled, y_train)

coefs = pd.Series(lr_l1.coef_[0], index=X.columns).sort_values(key=abs, ascending=False)
print("Coeficientes con L1 (C=0.1):")
print(coefs)
print(f"\nFeatures con coeficiente != 0: {(np.abs(lr_l1.coef_[0]) > 1e-6).sum()} de {len(X.columns)}")
print("L1 lleva coeficientes a cero → selección de features automática")

# ============================================================
# EJEMPLO 5: class_weight='balanced' — clases desbalanceadas
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 5: class_weight balanced para clases desbalanceadas")
print("="*60)

# Crear desbalanceo artificial
df_desb = df.copy()
# Reducir proporción de ventas
mask_venta = df_desb['se_vende_hoy'] == 1
drop_idx = mask_venta[mask_venta].sample(frac=0.6, random_state=42).index
df_desb = df_desb.drop(drop_idx)

print(f"Distribución original:    0={ (y==0).sum() }, 1={ (y==1).sum() } "
      f"→ ratio={(y==1).sum()/(y==0).sum():.2f}")
print(f"Distribución desbalanceada: "
      f"0={(df_desb['se_vende_hoy']==0).sum()}, "
      f"1={(df_desb['se_vende_hoy']==1).sum()} "
      f"→ ratio={(df_desb['se_vende_hoy']==1).sum()/(df_desb['se_vende_hoy']==0).sum():.2f}")

Xb = df_desb.drop('se_vende_hoy', axis=1)
yb = df_desb['se_vende_hoy']
Xb_train, Xb_test, yb_train, yb_test = train_test_split(Xb, yb, test_size=0.3, random_state=42)
Xb_train_s = scaler.fit_transform(Xb_train)
Xb_test_s = scaler.transform(Xb_test)

# Sin balanceo
lr_no_bal = LogisticRegression(random_state=42, max_iter=1000)
lr_no_bal.fit(Xb_train_s, yb_train)
y_pred_nb = lr_no_bal.predict(Xb_test_s)

# Con balanceo
lr_bal = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
lr_bal.fit(Xb_train_s, yb_train)
y_pred_bal = lr_bal.predict(Xb_test_s)

print(f"\nSin balanceo — accuracy: {(y_pred_nb == yb_test).mean():.3f}")
print(f"Con balanceo — accuracy: {(y_pred_bal == yb_test).mean():.3f}")
print("\nReporte sin balanceo:")
print(classification_report(yb_test, y_pred_nb))
print("Reporte con balanceo:")
print(classification_report(yb_test, y_pred_bal))

# ============================================================
# EJEMPLO 6: Multiclase 'ovr' — clasificar productos en categorías
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 6: Multiclase OvR (One-vs-Rest)")
print("="*60)

# Generar 3 categorías de productos
n_multi = 600
df_multi = pd.DataFrame({
    'precio': np.random.uniform(10, 500, n_multi),
    'peso_kg': np.random.uniform(0.1, 50, n_multi),
    'volumen_cm3': np.random.uniform(100, 5000, n_multi),
    'perecible': np.random.choice([0, 1], n_multi, p=[0.6, 0.4])
})

# Categorías: 0=electrónica, 1=alimentos, 2=ropa
condiciones = [
    (df_multi['precio'] > 200) & (df_multi['peso_kg'] < 5),
    (df_multi['perecible'] == 1) | (df_multi['peso_kg'] > 10),
    True
]
df_multi['categoria'] = np.select(condiciones, [0, 1, 2], default=2)
# Mezclar un poco
ruido = np.random.choice([0, 1, 2], n_multi, p=[0.05, 0.05, 0.05])
df_multi['categoria'] = np.where(np.random.random(n_multi) < 0.85,
                                  df_multi['categoria'], ruido)

Xm = df_multi.drop('categoria', axis=1)
ym = df_multi['categoria']

Xm_train, Xm_test, ym_train, ym_test = train_test_split(Xm, ym, test_size=0.3, random_state=42)
Xm_train_s = scaler.fit_transform(Xm_train)
Xm_test_s = scaler.transform(Xm_test)

lr_ovr = LogisticRegression(multi_class='ovr', max_iter=1000, random_state=42)
lr_ovr.fit(Xm_train_s, ym_train)
print(f"OvR Accuracy: {(lr_ovr.predict(Xm_test_s) == ym_test).mean():.3f}")
print(f"Forma coef_: {lr_ovr.coef_.shape}  (n_clases × n_features)")
print("Cada clase tiene su propio conjunto de coeficientes (one-vs-rest)")

# ============================================================
# EJEMPLO 7: Multiclase 'multinomial' — clasificar sucursales
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 7: Multiclase multinomial (softmax)")
print("="*60)

# 4 sucursales
n_suc = 800
df_suc = pd.DataFrame({
    'ventas_diarias': np.random.poisson(100, n_suc) + np.random.randint(0, 50, n_suc),
    'num_clientes': np.random.poisson(30, n_suc) + np.random.randint(0, 20, n_suc),
    'stock_promedio': np.random.uniform(100, 1000, n_suc),
    'empleados': np.random.randint(3, 30, n_suc),
    'dist_central_km': np.random.uniform(0, 50, n_suc)
})

# Asignar sucursal (0..3) basado en reglas
scores = np.column_stack([
    df_suc['ventas_diarias'] * 0.3 + df_suc['num_clientes'] * 0.2,
    df_suc['stock_promedio'] * 0.01 + df_suc['empleados'] * 0.5,
    df_suc['dist_central_km'] * (-0.5) + df_suc['ventas_diarias'] * 0.1,
    df_suc['empleados'] * 0.3 + df_suc['num_clientes'] * 0.4
])
df_suc['sucursal'] = scores.argmax(axis=1)
# Ruido
mask_ruido = np.random.random(n_suc) < 0.15
df_suc.loc[mask_ruido, 'sucursal'] = np.random.randint(0, 4, mask_ruido.sum())

Xs = df_suc.drop('sucursal', axis=1)
ys = df_suc['sucursal']

Xs_train, Xs_test, ys_train, ys_test = train_test_split(Xs, ys, test_size=0.3, random_state=42)
Xs_train_s = scaler.fit_transform(Xs_train)
Xs_test_s = scaler.transform(Xs_test)

lr_multi = LogisticRegression(multi_class='multinomial', solver='lbfgs',
                              max_iter=2000, random_state=42)
lr_multi.fit(Xs_train_s, ys_train)
y_pred_multi = lr_multi.predict(Xs_test_s)

print(f"Multinomial Accuracy: {(y_pred_multi == ys_test).mean():.3f}")
probas_suc = lr_multi.predict_proba(Xs_test_s)
print(f"Probabilidades (primeros 5):\n{probas_suc[:5].round(3)}")
print("Suma por fila (debe ser 1):", probas_suc[:5].sum(axis=1).round(6))

# ============================================================
# EJEMPLO 8: Curva ROC y AUC
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 8: Curva ROC y AUC")
print("="*60)

y_score = model.decision_function(X_test_scaled)
fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

print(f"AUC: {roc_auc:.3f}")
print(f"Rango de thresholds: [{thresholds[0]:.3f}, {thresholds[-1]:.3f}]")

# Encontrar threshold que maximiza Youden's J = tpr - fpr
j_scores = tpr - fpr
best_idx = j_scores.argmax()
print(f"Mejor threshold (Youden): {thresholds[best_idx]:.3f} "
      f"→ TPR={tpr[best_idx]:.3f}, FPR={fpr[best_idx]:.3f}")

# Graficar ROC
fig, ax = plt.subplots(figsize=(8, 6))
RocCurveDisplay.from_estimator(model, X_test_scaled, y_test, ax=ax)
ax.plot([0, 1], [0, 1], 'k--', label='Aleatorio (AUC=0.5)')
ax.set_title(f'Curva ROC — AUC = {roc_auc:.3f}')
ax.legend()
plt.savefig('I16_roc_curve.png', dpi=100)
plt.close()
print("Gráfico guardado: I16_roc_curve.png")

# ============================================================
# EJEMPLO 9: Matriz de confusión + classification_report
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 9: Matriz de confusión y reporte de clasificación")
print("="*60)

cm = confusion_matrix(y_test, y_pred)
print("Matriz de confusión:")
print(cm)
print("\nReporte de clasificación completo:")
print(classification_report(y_test, y_pred,
      target_names=['No Vende', 'Sí Vende']))

# Métricas derivadas
tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)
print(f"\nMétricas derivadas:")
print(f"  Precisión: {precision:.3f}  (de los que predije que venden, cuantos aciertan)")
print(f"  Recall:    {recall:.3f}  (de los que realmente venden, cuantos detecté)")
print(f"  F1-score:  {f1:.3f}  (media armónica de precisión y recall)")
print(f"  Accuracy:  {(tp+tn)/(tp+tn+fp+fn):.3f}")
print(f"  Tasa error:{(fp+fn)/(tp+tn+fp+fn):.3f}")

# ============================================================
# EJEMPLO 10: Precisión vs Recall — trade-off según umbral
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 10: Trade-off Precisión vs Recall")
print("="*60)

precision_curve, recall_curve, thresholds_pr = precision_recall_curve(
    y_test, prob_positivo
)

print("Efecto del umbral en Precisión y Recall:")
for umbral in [0.3, 0.5, 0.7, 0.9]:
    idx = np.searchsorted(thresholds_pr[::-1], umbral, side='left')
    if idx < len(precision_curve):
        p = precision_curve[-(idx+1)]
        r = recall_curve[-(idx+1)]
    else:
        p, r = precision_curve[0], recall_curve[0]
    print(f"  Umbral={umbral:.1f} → Precisión={p:.3f}, Recall={r:.3f}")

print("\nInterpretación:")
print("  Umbral alto → pocos positivos, pero los que predice son muy confiables (alta precisión, bajo recall)")
print("  Umbral bajo → muchos positivos, captura casi todos pero con muchos falsos (baja precisión, alto recall)")

# ============================================================
# EJEMPLO 11: Coeficientes — odds ratio e interpretación
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 11: Coeficientes y odds ratio")
print("="*60)

coefs_df = pd.DataFrame({
    'feature': X.columns,
    'coeficiente': model.coef_[0],
    'odds_ratio': np.exp(model.coef_[0]),
    'abs_coef': np.abs(model.coef_[0])
}).sort_values('abs_coef', ascending=False)

print(coefs_df)
print("\nInterpretación:")
print("  odds_ratio = exp(coef): factor multiplicativo de la odds por cada unidad de x")
print("  odds_ratio > 1 → aumenta probabilidad de venta")
print("  odds_ratio < 1 → disminuye probabilidad de venta")
print(f"\n  Por cada unidad de std en 'stock', la odds de venta se multiplica por "
      f"{np.exp(model.coef_[0][X.columns.get_loc('stock')]):.3f}")
print(f"  Por cada unidad de std en 'dias_ult_venta', la odds de venta se multiplica por "
      f"{np.exp(model.coef_[0][X.columns.get_loc('dias_ult_venta')]):.3f} (disminuye)")

# ============================================================
# EJEMPLO 12: LogisticRegressionCV — búsqueda automática de C
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 12: LogisticRegressionCV — búsqueda de C")
print("="*60)

lr_cv = LogisticRegressionCV(
    Cs=10,                  # 10 valores de C (logspace de -4 a 4 por defecto)
    cv=5,                   # 5-fold cross-validation
    penalty='l2',
    solver='lbfgs',
    max_iter=1000,
    scoring='accuracy',
    random_state=42
)
lr_cv.fit(X_train_scaled, y_train)

print(f"C óptimo encontrado: {lr_cv.C_[0]:.4f}")
print(f"Mejores scores por C: {lr_cv.scores_[1].mean(axis=0).round(4)}")
print(f"Accuracy en test con C óptimo: {(lr_cv.predict(X_test_scaled) == y_test).mean():.3f}")

# ============================================================
# EJEMPLO 13: Pipeline — StandardScaler + LogisticRegression
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 13: Pipeline: StandardScaler + LogisticRegression")
print("="*60)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

pipeline.fit(X_train, y_train)
y_pred_pipe = pipeline.predict(X_test)

print(f"Accuracy del pipeline: {(y_pred_pipe == y_test).mean():.3f}")
print("\nPipeline simplifica el workflow: aplica escalado y modelo en un solo paso")
print(pipeline)

# ============================================================
# EJEMPLO 14: Comparar L1 vs L2 en el mismo dataset
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 14: Comparación L1 vs L2")
print("="*60)

modelos_penalty = {
    'L1 (saga)': LogisticRegression(penalty='l1', solver='saga', C=1.0,
                                     random_state=42, max_iter=2000),
    'L2 (lbfgs)': LogisticRegression(penalty='l2', solver='lbfgs', C=1.0,
                                      random_state=42, max_iter=1000),
    'L1 (liblinear)': LogisticRegression(penalty='l1', solver='liblinear', C=1.0,
                                          random_state=42, max_iter=1000),
}

for nombre, modelo in modelos_penalty.items():
    modelo.fit(X_train_scaled, y_train)
    yp = modelo.predict(X_test_scaled)
    acc = (yp == y_test).mean()
    nz = (np.abs(modelo.coef_[0]) > 1e-6).sum()
    print(f"{nombre:20s} | acc={acc:.3f} | coefs != 0: {nz}/{len(X.columns)} | "
          f"coefs={np.round(modelo.coef_[0], 4)}")

# ============================================================
# EJEMPLO 15: Feature importance — ranking por coeficiente absoluto
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 15: Ranking de importancia por |coeficiente|")
print("="*60)

importancia = pd.DataFrame({
    'Feature': X.columns,
    'Coeficiente': model.coef_[0],
    '|Coef|': np.abs(model.coef_[0])
}).sort_values('|Coef|', ascending=False)

print(importancia.to_string(index=False))
print(f"\nFeature más importante: {importancia.iloc[0]['Feature']} "
      f"(|coef| = {importancia.iloc[0]['|Coef|']:.4f})")
print(f"Feature menos importante: {importancia.iloc[-1]['Feature']} "
      f"(|coef| = {importancia.iloc[-1]['|Coef|']:.4f})")

# Gráfico de barras
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(importancia['Feature'], importancia['|Coef|'])
ax.set_xlabel('|Coeficiente|')
ax.set_title('Importancia de Features (Regresión Logística)')
plt.tight_layout()
plt.savefig('I16_feature_importance.png', dpi=100)
plt.close()
print("Gráfico guardado: I16_feature_importance.png")

# ============================================================
# EJEMPLO 16: ElasticNet penalty con l1_ratio
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 16: ElasticNet penalty con l1_ratio")
print("="*60)

for l1_r in [0.0, 0.25, 0.5, 0.75, 1.0]:
    lr_en = LogisticRegression(
        penalty='elasticnet', solver='saga',
        C=0.5, l1_ratio=l1_r,
        random_state=42, max_iter=3000, n_jobs=-1
    )
    lr_en.fit(X_train_scaled, y_train)
    yp_en = lr_en.predict(X_test_scaled)
    acc_en = (yp_en == y_test).mean()
    nz_en = (np.abs(lr_en.coef_[0]) > 1e-6).sum()
    print(f"l1_ratio={l1_r:.2f} | acc={acc_en:.3f} | coefs_no_cero={nz_en}/{len(X.columns)}")

print("\nElasticNet: l1_ratio=0 → Ridge (L2), l1_ratio=1 → Lasso (L1)")
print("l1_ratio entre 0 y 1 → combinación de ambos")

# ============================================================
# EJEMPLO 17: Límite de decisión visual 2D
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 17: Límite de decisión visual 2D")
print("="*60)

# Usar solo 2 features para visualizar
X_2d = df[['precio', 'margen']].values
y_2d = df['se_vende_hoy'].values

X_2d_train, X_2d_test, y_2d_train, y_2d_test = train_test_split(
    X_2d, y_2d, test_size=0.3, random_state=42
)
scaler_2d = StandardScaler()
X_2d_train_s = scaler_2d.fit_transform(X_2d_train)
X_2d_test_s = scaler_2d.transform(X_2d_test)

lr_2d = LogisticRegression(random_state=42, max_iter=1000)
lr_2d.fit(X_2d_train_s, y_2d_train)

# Crear grid para visualizar frontera
xx, yy = np.meshgrid(
    np.linspace(X_2d_train_s[:, 0].min() - 0.5, X_2d_train_s[:, 0].max() + 0.5, 200),
    np.linspace(X_2d_train_s[:, 1].min() - 0.5, X_2d_train_s[:, 1].max() + 0.5, 200)
)
Z = lr_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(8, 6))
ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
scatter = ax.scatter(X_2d_test_s[:, 0], X_2d_test_s[:, 1], c=y_2d_test,
                     cmap='coolwarm', edgecolors='k', linewidth=0.5)
ax.set_xlabel('Precio (std)')
ax.set_ylabel('Margen (std)')
ax.set_title('Límite de decisión — Regresión Logística')
plt.tight_layout()
plt.savefig('I16_decision_boundary.png', dpi=100)
plt.close()
print("Gráfico guardado: I16_decision_boundary.png")
print("La línea recta muestra el límite de decisión lineal de la regresión logística")

# ============================================================
# EJEMPLO 18: Integrador — clasificar proveedores buenos/malos
# ============================================================
print("\n" + "="*60)
print("EJEMPLO 18: Integrador — clasificar proveedores")
print("="*60)

n_prov = 300
np.random.seed(42)
df_prov = pd.DataFrame({
    'precio_promedio': np.random.uniform(50, 500, n_prov),
    'calidad_promedio': np.random.uniform(1, 10, n_prov),
    'dias_entrega': np.random.uniform(1, 30, n_prov),
    'cumplimiento_pct': np.random.uniform(50, 100, n_prov),
    'volumen_mensual': np.random.uniform(100, 10000, n_prov),
    'antiguedad_meses': np.random.uniform(1, 120, n_prov),
})

# Buen proveedor: calidad alta, cumplimiento alto, entrega rápida
score_prov = (
    0.3 * df_prov['calidad_promedio']
    + 0.3 * (df_prov['cumplimiento_pct'] / 100)
    - 0.2 * (df_prov['dias_entrega'] / 30)
    + 0.1 * (df_prov['antiguedad_meses'] / 120)
    + 0.1 * np.log1p(df_prov['volumen_mensual']) / 10
)
prob_prov = 1 / (1 + np.exp(-3 * (score_prov - 0.5)))
df_prov['buen_proveedor'] = (prob_prov + np.random.normal(0, 0.1, n_prov) > 0.5).astype(int)

print(f"Proveedores buenos: {df_prov['buen_proveedor'].sum()}")
print(f"Proveedores malos:  {(df_prov['buen_proveedor'] == 0).sum()}")

Xp = df_prov.drop('buen_proveedor', axis=1)
yp = df_prov['buen_proveedor']
Xp_train, Xp_test, yp_train, yp_test = train_test_split(Xp, yp, test_size=0.3, random_state=42)
Xp_train_s = scaler.fit_transform(Xp_train)
Xp_test_s = scaler.transform(Xp_test)

# Pipeline con GridSearchCV
pipe_prov = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(random_state=42, max_iter=1000))
])
params_prov = {
    'clf__C': [0.01, 0.1, 1, 10],
    'clf__penalty': ['l2'],
    'clf__class_weight': [None, 'balanced']
}
gs_prov = GridSearchCV(pipe_prov, params_prov, cv=5, scoring='f1')
gs_prov.fit(Xp_train, yp_train)

print(f"\nMejores parámetros: {gs_prov.best_params_}")
print(f"Mejor F1 (CV): {gs_prov.best_score_:.3f}")
print(f"Accuracy en test: {(gs_prov.predict(Xp_test) == yp_test).mean():.3f}")
print(f"\nReporte de clasificación:")
print(classification_report(yp_test, gs_prov.predict(Xp_test),
      target_names=['Malo', 'Bueno']))

# Interpretación final
final_model = gs_prov.best_estimator_.named_steps['clf']
final_scaler = gs_prov.best_estimator_.named_steps['scaler']
Xp_test_s_final = final_scaler.transform(Xp_test)
probas_prov = final_model.predict_proba(Xp_test_s_final)[:, 1]

print(f"\nTop 5 proveedores con mayor probabilidad de ser buenos:")
top5_idx = np.argsort(probas_prov)[-5:][::-1]
for i in top5_idx:
    print(f"  Proveedor #{Xp_test.index[i]}: prob={probas_prov[i]:.3f} "
          f"→ {'BUENO' if gs_prov.predict(Xp_test)[i] else 'MALO'} (real: {yp_test.values[i]})")

print("\n=== FIN EJEMPLOS REGRESIÓN LOGÍSTICA ===")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*2. Ejemplos Prácticos con sklearn.*

1. Configuración inicial común
2. ============================================================
3. EJEMPLO 1: Binaria — clasificar si producto se vende hoy
4. ============================================================
5. Dataset simulado de productos
6. Target: se_vende_hoy (1=sí, 0=no) — regla subyacente simulada
7. ============================================================
8. EJEMPLO 2: Probabilidades con predict_proba y umbral personalizado

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

## 3. Ejercicios Propuestos

1. **Ajuste de umbral**: Usando el modelo del Ejemplo 1, encuentra el umbral que maximiza F1-score. ¿Cómo cambia la matriz de confusión?

2. **L1 con diferentes C**: Entrena LogisticRegression con penalty='l1' para C = [0.001, 0.01, 0.1, 1, 10]. ¿Cuántos features quedan con coeficiente no cero en cada caso?

3. **Multiclase real**: Con el dataset de sucursales (Ejemplo 7), prueba `multi_class='ovr'` vs `'multinomial'`. ¿Cuál da mejor accuracy? ¿En qué se diferencian las probabilidades?

4. **class_weight personalizado**: En el Ejemplo 5 (desbalanceado), prueba `class_weight={0:1, 1:5}` y `{0:1, 1:10}`. ¿Cómo cambia recall de la clase minoritaria?

5. **Calibración de probabilidades**: Usa `CalibratedClassifierCV` de sklearn sobre el modelo del Ejemplo 1. Compara las probabilidades calibradas vs las originales con un gráfico de confiabilidad.

6. **Regularización por feature**: Simula un dataset con 20 features donde solo 5 son relevantes. Usa L1 y L2. ¿L1 recupera correctamente las features relevantes?

7. **ROC con costo variable**: La matriz de costo es: VP=+10, FP=-5, FN=-20, VN=+1. Encuentra el threshold óptimo que maximiza la ganancia total en el Ejemplo 1.

8. **Análisis de residuos**: Para el modelo del Ejemplo 18 (proveedores), calcula los residuos (y_true - proba). ¿Hay patrones en los errores? Agrupa por quintiles de probabilidad y calcula la tasa de error en cada grupo.

---

## 4. Resumen

- La **regresión logística** es un modelo lineal interpretable para clasificación
- **predict_proba** devuelve probabilidades; **decision_function** devuelve distancia al hiperplano
- La regularización **L1** selecciona features (coeficientes a cero); **L2** los encoge
- **C** controla la fuerza de regularización (menor C = más regularización)
- **class_weight='balanced'** ayuda con clases desbalanceadas
- **OvR** entrena un modelo por clase; **multinomial** usa softmax (mejor para clases separables)
- Los coeficientes se interpretan como log-odds: `exp(coef)` es el odds ratio
- **LogisticRegressionCV** busca automáticamente el mejor C por cross-validation
- **Siempre escalar datos** antes de regresión logística para coeficientes interpretables
- La curva **ROC** y **AUC** evalúan el poder discriminativo independientemente del umbral

### Cuándo usar regresión logística
- Necesitas un modelo **interpretable** (coeficientes con significado)
- La relación entre features y target es **aproximadamente lineal**
- Tienes **pocos datos** o necesitas un baseline rápido
- La **probabilidad** de pertenencia a una clase es relevante para el negocio

### Limitaciones
- No captura relaciones **no lineales** complejas
- Sensible a **multicolinealidad** entre features
- Asume **independencia** entre observaciones
- Requiere **escalado** para coeficientes comparables
