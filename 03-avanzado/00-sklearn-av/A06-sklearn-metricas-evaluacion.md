# A06: Métricas y Evaluación de Modelos para Ventas, Compras e Inventarios

## Introducción Teórica

Las **métricas de evaluación** traducen el rendimiento del modelo a lenguaje de negocio. No todas las métricas son relevantes para todos los problemas: en ventas con clases desbalanceadas, accuracy es engañoso; en inventarios, el costo de un falso positivo (sobrestock) es distinto al de un falso negativo (rotura de stock).

### Métricas de clasificación:

1. **classification_report**: Resumen completo (precision, recall, f1, support) por clase.
2. **confusion_matrix / ConfusionMatrixDisplay**: Tabla VP, FP, FN, VN. Con normalize para ver porcentajes.
3. **RocCurveDisplay**: Trade-off entre TPR y FPR. AUC = área bajo la curva. Cuanto más cerca de 1, mejor.
4. **PrecisionRecallDisplay**: Trade-off entre precision y recall. Útil para clases desbalanceadas.
5. **DetCurveDisplay**: Detection Error Tradeoff — TPR vs FPR en escala normal.
6. **f1_score**: Media armónica de precision y recall. average='micro'/'macro'/'weighted'.
7. **log_loss**: Pérdida logarítmica (cross-entropy). Penaliza predicciones incorrectas con alta confianza.
8. **brier_score_loss**: Error cuadrático medio de probabilidades. Mide calibración.
9. **matthews_corrcoef (MCC)**: Coeficiente de correlación de Matthews (-1 a 1). Balanceado incluso con clases muy desbalanceadas.
10. **cohen_kappa_score**: Acuerdo entre predicciones y realidad, ajustado por azar.
11. **hamming_loss**: Fracción de etiquetas incorrectas (para multilabel).
12. **jaccard_score**: Similitud entre conjuntos de etiquetas.
13. **make_scorer**: Crea métricas personalizadas para usar en GridSearchCV.

### Aplicación en negocio:
- **Ventas**: F1 para clases desbalanceadas (pocas conversiones). AUC para ranking de clientes.
- **Compras**: MCC para calidad de proveedores. Brier score para calibración de probabilidades.
- **Inventarios**: Matriz de confusión para clasificación ABC con costos diferenciados.

---

## Ejemplos

### Ejemplo 1: classification_report detallado con output_dict=True

```python
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n = 2000
df = pd.DataFrame({
    'monto': np.random.exponential(500, n),
    'frecuencia': np.random.randint(1, 30, n),
    'recencia': np.random.randint(1, 365, n),
    'conversion': np.random.binomial(1, 0.25, n)  # 25% conversiones
})
X = df[['monto', 'frecuencia', 'recencia']]
y = df['conversion']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train_s, y_train)
y_pred = rf.predict(X_test_s)

report = classification_report(y_test, y_pred, output_dict=True)
print("Classification Report:")
for label, metrics in report.items():
    if isinstance(metrics, dict):
        print(f"  Clase {label}:")
        print(f"    Precision: {metrics['precision']:.3f}")
        print(f"    Recall: {metrics['recall']:.3f}")
        print(f"    F1: {metrics['f1-score']:.3f}")
        print(f"    Support: {metrics['support']}")

print(f"\nAccuracy: {report['accuracy']:.3f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: classification_report detallado con output_dict=True.*

1. `import pandas as pd` — Importa las librerías necesarias para el análisis.
2. `import numpy as np` — Importa las librerías necesarias para el análisis.
3. `from sklearn.metrics import classification_report, confusion_matrix` — Importa las librerías necesarias para el análisis.
4. `from sklearn.model_selection import train_test_split` — Importa las librerías necesarias para el análisis.
5. `from sklearn.ensemble import RandomForestClassifier` — Importa las librerías necesarias para el análisis.
6. `from sklearn.preprocessing import StandardScaler` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: confusion_matrix con normalize='true' (porcentajes por fila)

```python
cm = confusion_matrix(y_test, y_pred, normalize='true')
print("Matriz de confusión (porcentaje por fila - recall por clase):")
print("           Pred:0  Pred:1")
print(f"Real:0   {cm[0,0]:.2f}    {cm[0,1]:.2f}")
print(f"Real:1   {cm[1,0]:.2f}    {cm[1,1]:.2f}")

# Interpretación de negocio
vp = cm[1,1]  # Conversiones correctamente predichas
fp = cm[0,1]  # Ofertas enviadas a no-compradores
fn = cm[1,0]  # Conversiones perdidas
print(f"\nInterpretación:")
print(f"  VP (acertamos conversión): {vp:.1%}")
print(f"  FP (oferta perdida): {fp:.1%}")
print(f"  FN (venta perdida): {fn:.1%}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: confusion_matrix con normalize='true' (porcentajes por fila).*

1. Interpretación de negocio

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: ConfusionMatrixDisplay.from_estimator() con pipeline

```python
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe_cm = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])
pipe_cm.fit(X_train, y_train)

# Mostrar matriz como texto
cm_display = ConfusionMatrixDisplay.from_estimator(
    pipe_cm, X_test, y_test,
    display_labels=['No Compra', 'Compra'],
    cmap='Blues', normalize='true',
    include_values=True
)
print("ConfusionMatrixDisplay generado.")
print("Matriz (normalizada por fila):")
print(cm_display.confusion_matrix.round(3))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: ConfusionMatrixDisplay.from_estimator() con pipeline.*

1. Mostrar matriz como texto

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: RocCurveDisplay.from_estimator() para comparar modelos

```python
from sklearn.metrics import RocCurveDisplay
from sklearn.ensemble import GradientBoostingClassifier

models_roc = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

print("Modelo           | AUC-ROC")
for name, model in models_roc.items():
    model.fit(X_train_s, y_train)
    viz = RocCurveDisplay.from_estimator(model, X_test_s, y_test, name=name)
    print(f" {name:18s} | {viz.roc_auc:.3f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: RocCurveDisplay.from_estimator() para comparar modelos.*

1. `from sklearn.metrics import RocCurveDisplay` — Importa las librerías necesarias para el análisis.
2. `from sklearn.ensemble import GradientBoostingClassifier` — Importa las librerías necesarias para el análisis.
3. `print("Modelo           | AUC-ROC")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: RocCurveDisplay con plot_chance_level=True

```python
from sklearn.metrics import RocCurveDisplay

viz_chance = RocCurveDisplay.from_estimator(
    rf, X_test_s, y_test,
    name='RandomForest',
    plot_chance_level=True,
    chance_level_kw={'color': 'gray', 'linestyle': '--', 'label': 'Azar'}
)
print(f"AUC-ROC: {viz_chance.roc_auc:.3f}")
print(f"Línea de azar (0.5) también trazada")
print(f"El modelo está {((viz_chance.roc_auc - 0.5) / 0.5) * 100:.0f}% por encima del azar")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: RocCurveDisplay con plot_chance_level=True.*

1. `from sklearn.metrics import RocCurveDisplay` — Importa las librerías necesarias para el análisis.
2. `print(f"AUC-ROC: {viz_chance.roc_auc:.3f}")` — Muestra el resultado por pantalla.
3. `print(f"Línea de azar (0.5) también trazada")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: PrecisionRecallDisplay.from_estimator()

```python
from sklearn.metrics import PrecisionRecallDisplay

pr_viz = PrecisionRecallDisplay.from_estimator(
    rf, X_test_s, y_test, name='RandomForest',
    pos_label=1
)
print(f"Precision-Recall AUC: {pr_viz.average_precision:.3f}")
print(f"Precision base (proporción clase positiva): {y_test.mean():.3f}")

# Interpretación
ap = pr_viz.average_precision
baseline = y_test.mean()
print(f"Mejora sobre baseline: {(ap - baseline) / baseline * 100:.1f}%")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: PrecisionRecallDisplay.from_estimator().*

1. Interpretación

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: DET curve (Detection Error Tradeoff)

```python
from sklearn.metrics import DetCurveDisplay

det_viz = DetCurveDisplay.from_estimator(rf, X_test_s, y_test, name='RandomForest')
print(f"DET Curve generada (False Positive Rate vs False Negative Rate)")
print(f"DET muestra en escala normal lo que ROC muestra en log")
print(f"Más cerca del origen = mejor modelo")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: DET curve (Detection Error Tradeoff).*

1. `from sklearn.metrics import DetCurveDisplay` — Importa las librerías necesarias para el análisis.
2. `print(f"DET Curve generada (False Positive Rate vs False Negative Rate)")` — Muestra el resultado por pantalla.
3. `print(f"DET muestra en escala normal lo que ROC muestra en log")` — Muestra el resultado por pantalla.
4. `print(f"Más cerca del origen = mejor modelo")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: roc_curve manual — calcular fpr, tpr, thresholds

```python
from sklearn.metrics import roc_curve, auc

y_proba = rf.predict_proba(X_test_s)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

print(f"AUC-ROC: {roc_auc:.3f}")
print(f"\nPuntos de la curva ROC (cada 10%):")
for i in range(0, len(fpr), max(len(fpr)//10, 1)):
    print(f"  Threshold={thresholds[i]:.3f}: FPR={fpr[i]:.3f}, TPR={tpr[i]:.3f}")

# Encontrar threshold óptimo (Youden's J)
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
print(f"\nThreshold óptimo (Youden): {thresholds[best_idx]:.3f}")
print(f"  TPR={tpr[best_idx]:.3f}, FPR={fpr[best_idx]:.3f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: roc_curve manual — calcular fpr, tpr, thresholds.*

1. Encontrar threshold óptimo (Youden's J)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: precision_recall_curve manual

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, thresholds_pr = precision_recall_curve(y_test, y_proba)
ap = average_precision_score(y_test, y_proba)

print(f"Average Precision (AP): {ap:.3f}")
print(f"\nPuntos PR (cada 10%):")
for i in range(0, len(precision), max(len(precision)//10, 1)):
    if i < len(thresholds_pr):
        print(f"  Threshold={thresholds_pr[i]:.3f}: Precision={precision[i]:.3f}, Recall={recall[i]:.3f}")

# Threshold para precision > 0.7
for i, (p, r, t) in enumerate(zip(precision, recall, np.append(thresholds_pr, 1))):
    if p >= 0.7:
        print(f"\nPara Precision >= 0.7: threshold={t:.3f}, Recall={r:.3f}")
        break
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: precision_recall_curve manual.*

1. Threshold para precision > 0.7

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: f1_score con average='micro' vs 'macro' vs 'weighted'

```python
from sklearn.metrics import f1_score

# Clasificación multiclase: ABC de inventarios
y_true_multi = np.random.choice(['A', 'B', 'C'], 500, p=[0.2, 0.3, 0.5])
y_pred_multi = np.random.choice(['A', 'B', 'C'], 500, p=[0.25, 0.35, 0.4])

print("Tipo average | F1 Score")
for avg in ['micro', 'macro', 'weighted']:
    f1 = f1_score(y_true_multi, y_pred_multi, average=avg)
    print(f"  {avg:10s} | {f1:.3f}")

print("\n  micro:    cuenta VP/FP/FN global (clase dominante pesa más)")
print("  macro:    promedio simple por clase (clases minoritarias pesan igual)")
print("  weighted: promedio ponderado por soporte de cada clase")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: f1_score con average='micro' vs 'macro' vs 'weighted'.*

1. Clasificación multiclase: ABC de inventarios

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: log_loss — pérdida logarítmica (menor es mejor)

```python
from sklearn.metrics import log_loss

# Probabilidades mal calibradas vs bien calibradas
y_true_ll = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

# Buenas probabilidades
probs_good = np.array([0.1, 0.9, 0.2, 0.8, 0.15, 0.85, 0.3, 0.7, 0.25, 0.75])
# Malas probabilidades (muy confiado pero incorrecto)
probs_bad = np.array([0.01, 0.99, 0.01, 0.99, 0.01, 0.99, 0.01, 0.99, 0.01, 0.99])

ll_good = log_loss(y_true_ll, np.column_stack([1-probs_good, probs_good]))
ll_bad = log_loss(y_true_ll, np.column_stack([1-probs_bad, probs_bad]))

print(f"Log loss (buenas probas): {ll_good:.4f}")
print(f"Log loss (malas probas):  {ll_bad:.4f}")
print(f"Diferencia: {ll_bad - ll_good:.4f}")
print("Nota: log_loss penaliza fuertemente predicciones incorrectas con alta confianza")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: log_loss — pérdida logarítmica (menor es mejor).*

1. Probabilidades mal calibradas vs bien calibradas
2. Buenas probabilidades
3. Malas probabilidades (muy confiado pero incorrecto)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: brier_score_loss — calibración de probabilidades

```python
from sklearn.metrics import brier_score_loss

# Modelos calibrados vs no calibrados
from sklearn.calibration import CalibratedClassifierCV

rf_raw = RandomForestClassifier(n_estimators=100, random_state=42)
rf_raw.fit(X_train_s, y_train)
probs_raw = rf_raw.predict_proba(X_test_s)[:, 1]

rf_cal = CalibratedClassifierCV(rf_raw, cv=3, method='sigmoid')
rf_cal.fit(X_train_s, y_train)
probs_cal = rf_cal.predict_proba(X_test_s)[:, 1]

brier_raw = brier_score_loss(y_test, probs_raw)
brier_cal = brier_score_loss(y_test, probs_cal)

print(f"Brier score (sin calibrar): {brier_raw:.4f}")
print(f"Brier score (calibrado):    {brier_cal:.4f}")
print(f"Mejora: {(brier_raw - brier_cal) / brier_raw * 100:.1f}%")
print(f"Brier score ideal = 0 (perfecta calibración)")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: brier_score_loss — calibración de probabilidades.*

1. Modelos calibrados vs no calibrados

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: matthews_corrcoef — coeficiente MCC (-1 a 1)

```python
from sklearn.metrics import matthews_corrcoef

mcc = matthews_corrcoef(y_test, y_pred)
print(f"Matthews Corrcoef (MCC): {mcc:.3f}")
print(f"  MCC=1:  predicción perfecta")
print(f"  MCC=0:  aleatorio")
print(f"  MCC=-1: inverso perfecto")

# MCC es balanceado incluso con clases desbalanceadas
# Comparar con accuracy
from sklearn.metrics import accuracy_score
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Nota: MCC es más fiable que accuracy cuando hay desbalanceo")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: matthews_corrcoef — coeficiente MCC (-1 a 1).*

1. MCC es balanceado incluso con clases desbalanceadas
2. Comparar con accuracy

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: cohen_kappa_score — acuerdo entre predicciones

```python
from sklearn.metrics import cohen_kappa_score

kappa = cohen_kappa_score(y_test, y_pred)
print(f"Cohen's Kappa: {kappa:.3f}")
print(f"  Kappa=1: acuerdo perfecto")
print(f"  Kappa=0: acuerdo aleatorio")
print(f"  Kappa<0: menos acuerdo que al azar")

# Interpretación estándar
if kappa > 0.8:
    print("  Acuerdo casi perfecto")
elif kappa > 0.6:
    print("  Acuerdo sustancial")
elif kappa > 0.4:
    print("  Acuerdo moderado")
else:
    print("  Acuerdo bajo")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: cohen_kappa_score — acuerdo entre predicciones.*

1. Interpretación estándar

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: hamming_loss — fracción de etiquetas incorrectas

```python
from sklearn.metrics import hamming_loss

# Para clasificación multiclase
y_true_hl = np.array([0, 1, 2, 0, 1, 2])
y_pred_hl = np.array([0, 1, 1, 0, 2, 2])

hamming = hamming_loss(y_true_hl, y_pred_hl)
acc_hl = accuracy_score(y_true_hl, y_pred_hl)

print(f"Hamming Loss: {hamming:.3f}")
print(f"Accuracy:     {acc_hl:.3f}")
print(f"Nota: Hamming Loss = 1 - Accuracy para clasificación simple")
print(f"  Hamming Loss = fracción de predicciones incorrectas")

# Para bilabel (multilabel)
y_true_multi = np.array([[1, 0], [0, 1], [1, 1]])
y_pred_multi = np.array([[1, 0], [0, 0], [1, 1]])
hl_multi = hamming_loss(y_true_multi, y_pred_multi)
print(f"Hamming Loss (multilabel): {hl_multi:.3f} (1 error en 6 etiquetas)")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: hamming_loss — fracción de etiquetas incorrectas.*

1. Para clasificación multiclase
2. Para bilabel (multilabel)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 16: make_scorer personalizado — profit_score

```python
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

def profit_score(y_true, y_pred):
    """
    Ganancia neta de campaña de ventas:
    - VP (acertar venta): +$100
    - FP (oferta a no-comprador): -$20
    - FN (perder venta): -$50
    """
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    ganancia = tp * 100 - fp * 20 - fn * 50
    return ganancia / max(len(y_true), 1)

profit_scorer = make_scorer(profit_score, greater_is_better=True)

# Usar en validación
from sklearn.model_selection import cross_val_score
scores_profit = cross_val_score(rf, X_train_s, y_train, cv=3, scoring=profit_scorer)
print(f"Profit por cliente (CV): ${scores_profit.mean():.2f} ± ${scores_profit.std():.2f}")
print(f"Ganancia estimada para 1000 clientes: ${scores_profit.mean() * 1000:.0f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: make_scorer personalizado — profit_score.*

1. Usar en validación

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 17: make_scorer con needs_proba=True para AUC

```python
def profit_by_threshold(y_true, y_proba, threshold=0.3):
    """Ganancia con threshold ajustable basado en probabilidad."""
    y_pred = (y_proba[:, 1] >= threshold).astype(int)
    return profit_score(y_true, y_pred)

profit_proba_scorer = make_scorer(
    profit_by_threshold,
    greater_is_better=True,
    needs_proba=True,
    threshold=0.3  # umbral de decisión
)

# Evaluar diferentes umbrales
umbrales = [0.2, 0.3, 0.4, 0.5, 0.6]
print("Umbral | Profit/cliente")
for umbral in umbrales:
    scorer = make_scorer(profit_by_threshold, greater_is_better=True, needs_proba=True, threshold=umbral)
    scores = cross_val_score(rf, X_train_s, y_train, cv=3, scoring=scorer)
    print(f"  {umbral:.1f}  |    ${scores.mean():.2f}")

print("\nEl umbral óptimo maximiza la ganancia neta de negocio")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: make_scorer con needs_proba=True para AUC.*

1. Evaluar diferentes umbrales

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 18: Integrador — comparar 5 modelos con todas las métricas en una tabla

```python
"""
Comparación exhaustiva de 5 modelos con 8 métricas cada uno.
Aplicado a clasificación de conversión en ventas.
"""

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, cohen_kappa_score, brier_score_loss
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC

models_to_compare = {
    'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM(RBF)': SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42),
}

metrics_dict = {
    'Accuracy': accuracy_score,
    'Precision': lambda yt, yp: precision_score(yt, yp, zero_division=0),
    'Recall': recall_score,
    'F1': f1_score,
    'MCC': matthews_corrcoef,
    'Kappa': cohen_kappa_score,
}

print(f"{'Modelo':20s} | {'Acc':>6s} | {'Prec':>6s} | {'Recall':>6s} | {'F1':>6s} | {'AUC':>6s} | {'MCC':>6s} | {'Kappa':>6s} | {'Brier':>6s}")
print("-" * 100)

results_table = []
for name, model in models_to_compare.items():
    model.fit(X_train_s, y_train)
    y_pred_m = model.predict(X_test_s)
    y_proba_m = model.predict_proba(X_test_s)[:, 1] if hasattr(model, 'predict_proba') else None

    row = {'Modelo': name}
    for metric_name, metric_fn in metrics_dict.items():
        row[metric_name] = metric_fn(y_test, y_pred_m)

    # AUC-ROC
    if y_proba_m is not None:
        row['AUC'] = roc_auc_score(y_test, y_proba_m)
    else:
        row['AUC'] = np.nan

    # Brier score
    if y_proba_m is not None:
        row['Brier'] = brier_score_loss(y_test, y_proba_m)
    else:
        row['Brier'] = np.nan

    results_table.append(row)

    metrics_str = ' | '.join([f"{row[m]:6.3f}" for m in ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'MCC', 'Kappa', 'Brier']])
    print(f" {name:20s} | {metrics_str}")

print("\nConclusiones:")
print(f"  - Mejor F1: {max(results_table, key=lambda r: r['F1'])['Modelo']} ({max(r['F1'] for r in results_table):.3f})")
print(f"  - Mejor AUC: {max(results_table, key=lambda r: r['AUC'])['Modelo']} ({max(r['AUC'] for r in results_table):.3f})")
print(f"  - Menor Brier: {min(results_table, key=lambda r: r['Brier'])['Modelo']} ({min(r['Brier'] for r in results_table):.4f})")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador — comparar 5 modelos con todas las métricas en una tabla.*

1. AUC-ROC
2. Brier score

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejercicios

1. **Reporte de métricas para clasificación ABC**: Genera classification_report completo para un modelo de clasificación ABC de inventarios (3 clases desbalanceadas: 20% A, 30% B, 50% C). Interpreta precision, recall, f1 por clase.

2. **Matriz de confusión con costos de negocio**: Crea una matriz de confusión para un modelo de aprobación de créditos en compras. Asigna costos: FP = $500 (costo de crédito impago), FN = $2000 (pérdida de venta mayorista). Calcula el costo total.

3. **Threshold óptimo con ROC curve**: Usa roc_curve para encontrar el threshold que maximiza (TPR - FPR) (Youden index). Aplica a un modelo de detección de fraude en ventas. Reporta el threshold y las métricas asociadas.

4. **Precision-Recall para stock de seguridad**: Para un modelo que predice ruptura de stock (clase minoritaria 10%), usa PrecisionRecallDisplay. ¿Qué threshold da recall > 0.8? ¿Cuál es la precision a ese threshold?

5. **Comparación de 4 promedios de F1**: Para un modelo multiclase de 4 categorías de productos, calcula f1_score con average='micro', 'macro', 'weighted', y 'samples' (si aplica). Explica las diferencias.

6. **Análisis de calibración con Brier score**: Compara RandomForest, GradientBoosting y LogisticRegression en términos de calibración (Brier score). ¿Cuál tiene las probabilidades mejor calibradas? ¿Afecta esto a decisiones de negocio?

7. **Custom scorer de rotación de inventarios**: Crea un scorer personalizado que calcule el costo total de inventario: costo_almacenamiento * días_en_stock + costo_rotura * stock_out. Úsalo en GridSearchCV para optimizar un modelo.

8. **Tabla comparativa de 6 modelos con 10 métricas**: Extiende el ejemplo integrador para incluir XGBoost, LightGBM y CatBoost (si están instalados). Genera una tabla con accuracy, precision, recall, f1, auc, mcc, kappa, brier, log_loss, y un custom scorer de negocio. ¿Cuál modelo recomiendas y por qué?

---

## Resumen

- **classification_report**: Visión rápida de precision/recall/f1 por clase. Usar `output_dict=True` para acceso programático.
- **confusion_matrix**: VP/FP/FN/VN. `normalize='true'` muestra recall por fila; `'pred'` muestra precisión por columna; `'all'` muestra porcentaje del total.
- **ConfusionMatrixDisplay / RocCurveDisplay / PrecisionRecallDisplay / DetCurveDisplay**: Visualizaciones integradas que funcionan con estimadores o predicciones.
- **roc_curve / precision_recall_curve**: Acceso a valores crudos para cálculo de threshold óptimo.
- **f1_score**: `average='micro'` para global, `'macro'` para equitativo por clase, `'weighted'` para ponderado por soporte.
- **log_loss**: Penaliza confianza incorrecta. Útil para calibrar modelos probabilísticos.
- **brier_score_loss**: Error cuadrático de probabilidades. Mide calibración directamente.
- **matthews_corrcoef (MCC)**: Métrica balanceada incluso con clases muy desbalanceadas (-1 a 1).
- **cohen_kappa_score**: Acuerdo ajustado por azar. Útil para medir consistencia.
- **hamming_loss**: Fracción de etiquetas incorrectas. Funciona para multilabel.
- **jaccard_score**: Similitud de conjuntos de etiquetas predichas vs reales.
- **make_scorer**: Convierte cualquier función en scorer de sklearn. Opciones: `greater_is_better`, `needs_proba`, `needs_threshold`.
- En negocio de ventas/compras/inventarios, **la métrica correcta depende de la asimetría de costos**: no es lo mismo un falso positivo (sobrestock) que un falso negativo (rotura de stock).
