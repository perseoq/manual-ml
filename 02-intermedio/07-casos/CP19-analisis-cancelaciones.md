# CP19: Análisis de Cancelaciones — Modelo Predictivo

## Contexto de Negocio
El equipo de ventas quiere predecir qué transacciones tienen riesgo de cancelarse para tomar acciones preventivas. Reducir cancelaciones impacta directamente en los ingresos y en la satisfacción del cliente.

```python
# ============================================================
# 1. CARGA DE VENTAS Y CREACIÓN DE VARIABLE OBJETIVO
# ============================================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.figsize": (12, 6), "font.size": 12})

ventas = pd.read_csv("../datos/ventas.csv")
ventas["fecha"] = pd.to_datetime(ventas["fecha"])

# Simular variable objetivo: ~10% de cancelaciones
np.random.seed(42)
prob_cancelacion = np.where(ventas["ingreso"] > ventas["ingreso"].quantile(0.9), 0.25,
                            np.where(ventas["descuento"] > ventas["descuento"].quantile(0.8), 0.20,
                                     np.where(ventas["cantidad"] > 5, 0.15, 0.05)))
ventas["cancelada"] = np.random.binomial(1, prob_cancelacion)

print("Dimensiones:", ventas.shape)
print("\nColumnas:", ventas.columns.tolist())
print("\nDistribución de cancelaciones:")
print(ventas["cancelada"].value_counts())
print(f"Tasa de cancelación: {ventas['cancelada'].mean()*100:.2f}%")

print("\nPrimeras filas:")
print(ventas.head())
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
2. 1. CARGA DE VENTAS Y CREACIÓN DE VARIABLE OBJETIVO
3. ============================================================
4. Simular variable objetivo: ~10% de cancelaciones

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 2. FEATURES: MONTO, DESCUENTO, PRODUCTO, SUCURSAL, DÍA
# ============================================================
# Ingeniería de features
ventas["dia_semana"] = ventas["fecha"].dt.day_name()
ventas["mes"] = ventas["fecha"].dt.month
ventas["dia_del_mes"] = ventas["fecha"].dt.day
ventas["fin_de_semana"] = ventas["fecha"].dt.dayofweek >= 5

# Features agregadas por producto
ventas["precio_promedio"] = ventas["ingreso"] / ventas["cantidad"].replace(0, np.nan)

# Codificar variables categóricas
le_producto = LabelEncoder()
le_sucursal = LabelEncoder()
le_dia = LabelEncoder()

ventas["producto_encoded"] = le_producto.fit_transform(ventas["producto"].astype(str))
ventas["sucursal_encoded"] = le_sucursal.fit_transform(ventas["sucursal"].astype(str))
ventas["dia_encoded"] = le_dia.fit_transform(ventas["dia_semana"])

# Seleccionar features para el modelo
feature_cols = ["ingreso", "cantidad", "descuento", "precio_promedio",
                "producto_encoded", "sucursal_encoded", "dia_encoded",
                "mes", "dia_del_mes", "fin_de_semana"]

X = ventas[feature_cols].fillna(0)
y = ventas["cancelada"]

print("Features utilizadas:")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i:2d}. {col}")

print(f"\nMatriz de features: {X.shape}")
print(f"Balance de clases:\n{y.value_counts().to_string()}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 2. FEATURES: MONTO, DESCUENTO, PRODUCTO, SUCURSAL, DÍA
3. ============================================================
4. Ingeniería de features
5. Features agregadas por producto
6. Codificar variables categóricas
7. Seleccionar features para el modelo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 3. EXPLORAR BALANCE DE CLASES
# ============================================================
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
clases = ventas["cancelada"].map({0: "No cancelada", 1: "Cancelada"})
sns.countplot(data=ventas, x=clases, palette="RdBu", hue=clases, legend=False)
plt.title("Balance de Clases", fontsize=14)
plt.xlabel("")
plt.ylabel("Cantidad")
for i, p in enumerate(plt.gca().patches):
    plt.text(p.get_x() + p.get_width()/2, p.get_height() + 20,
             f"{p.get_height()}", ha="center")

plt.subplot(1, 2, 2)
sns.barplot(data=ventas, x="dia_semana", y="cancelada",
            order=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            palette="viridis", hue="dia_semana", legend=False)
plt.title("Tasa de Cancelación por Día de la Semana", fontsize=14)
plt.xlabel("Día")
plt.ylabel("Tasa de Cancelación")
plt.xticks(rotation=30)

plt.tight_layout()
plt.show()

print(f"Clase minoritaria (canceladas): {y.sum()} ({y.mean()*100:.2f}%)")
print(f"Clase mayoritaria (no canceladas): {(1-y).sum()} ({(1-y).mean()*100:.2f}%)")
print("\nSe usará class_weight='balanced' para manejar el desbalance")
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

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 4. ENTREGAR LOGISTICREGRESSION CON CLASS_WEIGHT='BALANCED'
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Escalar features para regresión logística
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(
    class_weight="balanced",
    C=1.0,
    max_iter=1000,
    random_state=42,
    solver="lbfgs"
)
lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

print("Regresión Logística (class_weight='balanced')")
print("=" * 50)
print(classification_report(y_test, y_pred_lr, target_names=["No cancelada", "Cancelada"]))

roc_lr = roc_auc_score(y_test, y_prob_lr)
print(f"ROC AUC: {roc_lr:.4f}")

# Validación cruzada
cv_scores_lr = cross_val_score(lr, X_train_scaled, y_train, cv=5, scoring="roc_auc")
print(f"CV ROC AUC (media ± std): {cv_scores_lr.mean():.4f} ± {cv_scores_lr.std():.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 4. ENTREGAR LOGISTICREGRESSION CON CLASS_WEIGHT='BALANCED'
3. ============================================================
4. Escalar features para regresión logística
5. Validación cruzada

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 5. RANDOMFORESTCLASSIFIER PARA CANCELACIÓN
# ============================================================
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

print("Random Forest Classifier")
print("=" * 50)
print(classification_report(y_test, y_pred_rf, target_names=["No cancelada", "Cancelada"]))

roc_rf = roc_auc_score(y_test, y_prob_rf)
print(f"ROC AUC: {roc_rf:.4f}")

cv_scores_rf = cross_val_score(rf, X_train, y_train, cv=5, scoring="roc_auc")
print(f"CV ROC AUC (media ± std): {cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 5. RANDOMFORESTCLASSIFIER PARA CANCELACIÓN
3. ============================================================

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 6. COMPARAR MODELOS CON ROC AUC
# ============================================================
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

plt.figure(figsize=(10, 8))
plt.plot(fpr_lr, tpr_lr, linewidth=2,
         label=f"Logistic Regression (AUC = {roc_lr:.4f})")
plt.plot(fpr_rf, tpr_rf, linewidth=2,
         label=f"Random Forest (AUC = {roc_rf:.4f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Clasificador aleatorio")
plt.xlabel("False Positive Rate (1 - Especificidad)", fontsize=12)
plt.ylabel("True Positive Rate (Sensibilidad)", fontsize=12)
plt.title("Curvas ROC — Comparación de Modelos", fontsize=14)
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

resultados_modelos = pd.DataFrame({
    "Modelo": ["Logistic Regression", "Random Forest"],
    "ROC AUC Test": [round(roc_lr, 4), round(roc_rf, 4)],
    "ROC AUC CV": [round(cv_scores_lr.mean(), 4), round(cv_scores_rf.mean(), 4)],
    "Precisión": [round((y_pred_lr == y_test).mean(), 4), round((y_pred_rf == y_test).mean(), 4)]
})

print("Comparación de modelos:")
print(resultados_modelos.to_string(index=False))

mejor_modelo = "Random Forest" if roc_rf > roc_lr else "Logistic Regression"
print(f"\n{'='*50}")
print(f"MEJOR MODELO: {mejor_modelo} (ROC AUC = {max(roc_rf, roc_lr):.4f})")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 6. COMPARAR MODELOS CON ROC AUC
3. ============================================================

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 7. FEATURE IMPORTANCE: QUÉ LLEVA A CANCELACIÓN
# ============================================================
importancias = pd.DataFrame({
    "feature": feature_cols,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=True)

plt.figure(figsize=(10, 8))
plt.barh(importancias["feature"], importancias["importance"], color="steelblue")
plt.title("Feature Importance — Random Forest", fontsize=14)
plt.xlabel("Importancia")
plt.ylabel("Feature")
plt.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.show()

print("Importancia de features (Random Forest):")
print(importancias.sort_values("importance", ascending=False).to_string(index=False))

print(f"\nTop 3 factores de cancelación:")
top_features = importancias.sort_values("importance", ascending=False).head(3)["feature"].tolist()
for i, f in enumerate(top_features, 1):
    print(f"  {i}. {f}")

# Coeficientes de regresión logística para interpretación
coef_lr = pd.DataFrame({
    "feature": feature_cols,
    "coeficiente": lr.coef_[0]
}).sort_values("coeficiente", ascending=False)

print(f"\nCoeficientes de Logistic Regression:")
print(coef_lr.to_string(index=False))
print("\n(Coeficiente positivo = aumenta probabilidad de cancelación)")
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

*Este ejemplo.*

1. ============================================================
2. 7. FEATURE IMPORTANCE: QUÉ LLEVA A CANCELACIÓN
3. ============================================================
4. Coeficientes de regresión logística para interpretación

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 8. MATRIZ DE CONFUSIÓN E INTERPRETACIÓN
# ============================================================
# Usar el mejor modelo
modelo_final = rf if roc_rf > roc_lr else lr
y_pred_final = y_pred_rf if roc_rf > roc_lr else y_pred_lr
nombre_final = "Random Forest" if roc_rf > roc_lr else "Logistic Regression"

cm = confusion_matrix(y_test, y_pred_final)
cm_df = pd.DataFrame(cm,
                     index=["Real: No Cancelada", "Real: Cancelada"],
                     columns=["Pred: No Cancelada", "Pred: Cancelada"])

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Cancelada", "Cancelada"],
            yticklabels=["No Cancelada", "Cancelada"])
plt.title(f"Matriz de Confusión — {nombre_final}", fontsize=14)
plt.ylabel("Real")
plt.xlabel("Predicción")
plt.tight_layout()
plt.show()

print(f"Matriz de Confusión ({nombre_final}):")
print(cm_df.to_string())

tn, fp, fn, tp = cm.ravel()
print(f"\nInterpretación:")
print(f"  Verdaderos Negativos (correctamente no canceladas): {tn}")
print(f"  Falsos Positivos (falsas alarmas): {fp}")
print(f"  Falsos Negativos (cancelaciones no detectadas): {fn}")
print(f"  Verdaderos Positivos (cancelaciones detectadas): {tp}")

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nMétricas Clase Cancelación:")
print(f"  Precisión: {precision:.3f} (de las alertas, cuántas son reales)")
print(f"  Recall:    {recall:.3f} (de las cancelaciones reales, cuántas detectamos)")
print(f"  F1-score:  {f1:.3f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 8. MATRIZ DE CONFUSIÓN E INTERPRETACIÓN
3. ============================================================
4. Usar el mejor modelo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 9. THRESHOLD TUNING: AJUSTAR UMBRAL DE DECISIÓN
# ============================================================
umbrales = np.arange(0.1, 0.9, 0.05)
resultados_th = []

for umbral in umbrales:
    y_pred_th = (y_prob_rf >= umbral).astype(int)
    cm_th = confusion_matrix(y_test, y_pred_th)
    tn_th, fp_th, fn_th, tp_th = cm_th.ravel()
    precision_th = tp_th / (tp_th + fp_th) if (tp_th + fp_th) > 0 else 0
    recall_th = tp_th / (tp_th + fn_th) if (tp_th + fn_th) > 0 else 0
    f1_th = 2 * precision_th * recall_th / (precision_th + recall_th) if (precision_th + recall_th) > 0 else 0
    resultados_th.append({
        "umbral": umbral,
        "precision": precision_th,
        "recall": recall_th,
        "f1": f1_th,
        "tp": tp_th,
        "fp": fp_th,
        "fn": fn_th,
        "tn": tn_th
    })

df_th = pd.DataFrame(resultados_th)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(df_th["umbral"], df_th["precision"], label="Precisión", marker="o", linewidth=2)
plt.plot(df_th["umbral"], df_th["recall"], label="Recall", marker="s", linewidth=2)
plt.plot(df_th["umbral"], df_th["f1"], label="F1-score", marker="^", linewidth=2)
plt.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
plt.title("Métricas vs Umbral de Decisión", fontsize=13)
plt.xlabel("Umbral")
plt.ylabel("Score")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(df_th["umbral"], df_th["tp"], label="TP (detectados)", marker="o", linewidth=2)
plt.plot(df_th["umbral"], df_th["fp"], label="FP (falsas alarmas)", marker="s", linewidth=2)
plt.title("TP y FP vs Umbral", fontsize=13)
plt.xlabel("Umbral")
plt.ylabel("Cantidad")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Mejor umbral por F1
mejor_f1 = df_th.loc[df_th["f1"].idxmax()]
print(f"Mejor umbral por F1: {mejor_f1['umbral']:.2f} (F1 = {mejor_f1['f1']:.3f})")
print(f"  Precisión: {mejor_f1['precision']:.3f}")
print(f"  Recall:    {mejor_f1['recall']:.3f}")
print(f"  TP: {int(mejor_f1['tp'])}, FP: {int(mejor_f1['fp'])}, FN: {int(mejor_f1['fn'])}")

# Mejor umbral balanceando precisión y recall
df_th["distancia"] = np.sqrt((1 - df_th["precision"])**2 + (1 - df_th["recall"])**2)
mejor_balance = df_th.loc[df_th["distancia"].idxmin()]
print(f"\nMejor umbral balanceado: {mejor_balance['umbral']:.2f}")
print(f"  Precisión: {mejor_balance['precision']:.3f}, Recall: {mejor_balance['recall']:.3f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 9. THRESHOLD TUNING: AJUSTAR UMBRAL DE DECISIÓN
3. ============================================================
4. Mejor umbral por F1
5. Mejor umbral balanceando precisión y recall

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 10. GANANCIA POR REDUCIR CANCELACIONES (ROI ESTIMADO)
# ============================================================
ingreso_promedio = ventas["ingreso"].mean()
tasa_cancelacion_actual = ventas["cancelada"].mean()
total_transacciones = len(ventas)
costo_oportunidad = ingreso_promedio * tasa_cancelacion_actual * total_transacciones

# Simular escenarios de reducción
escenarios = [0.1, 0.2, 0.3, 0.4, 0.5]
resultados_roi = []

for reduccion in escenarios:
    cancelaciones_evitadas = round(total_transacciones * tasa_cancelacion_actual * reduccion)
    ingreso_recuperado = cancelaciones_evitadas * ingreso_promedio
    costo_intervencion = cancelaciones_evitadas * 10  # $10 por intervención
    roi = (ingreso_recuperado - costo_intervencion) / costo_intervencion * 100 if costo_intervencion > 0 else 0
    resultados_roi.append({
        "Reducción": f"{reduccion*100:.0f}%",
        "Cancelaciones_evitadas": cancelaciones_evitadas,
        "Ingreso_recuperado": round(ingreso_recuperado),
        "Costo_intervención": round(costo_intervencion),
        "ROI_%": round(roi, 1)
    })

df_roi = pd.DataFrame(resultados_roi)
print("Proyección ROI por reducción de cancelaciones:")
print(df_roi.to_string(index=False))

print(f"\n{'='*60}")
print(f"RESUMEN ECONÓMICO")
print(f"{'='*60}")
print(f"  Ingreso promedio por transacción: ${ingreso_promedio:.2f}")
print(f"  Tasa de cancelación actual: {tasa_cancelacion_actual*100:.1f}%")
print(f"  Total transacciones: {total_transacciones}")
print(f"  Costo de oportunidad actual: ${costo_oportunidad:,.2f}")

print(f"\n  Escenario recomendado (30% reducción):")
print(f"    Cancelaciones evitadas: {resultados_roi[2]['Cancelaciones_evitadas']}")
print(f"    Ingreso recuperado: ${resultados_roi[2]['Ingreso_recuperado']:,}")
print(f"    ROI: {resultados_roi[2]['ROI_%']}%")
print(f"    (Inversión en alertas tempranas, call center, ofertas de retención)")

plt.figure(figsize=(10, 5))
plt.bar(df_roi["Reducción"], df_roi["Ingreso_recuperado"], color="green", alpha=0.7, label="Ingreso recuperado")
plt.bar(df_roi["Reducción"], df_roi["Costo_intervención"], color="red", alpha=0.5, label="Costo de intervención")
plt.title("Análisis Costo-Beneficio de Reducción de Cancelaciones", fontsize=13)
plt.xlabel("% Reducción de Cancelaciones")
plt.ylabel("Monto ($)")
plt.legend()
plt.grid(True, alpha=0.3, axis="y")
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
2. 10. GANANCIA POR REDUCIR CANCELACIONES (ROI ESTIMADO)
3. ============================================================
4. Simular escenarios de reducción

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 11. REGLAS DE NEGOCIO PREVENTIVAS
# ============================================================
print("=" * 70)
print("REGLAS DE NEGOCIO PREVENTIVAS")
print("=" * 70)

# Reglas basadas en feature importance y coeficientes
print(f"""
Basado en el análisis de importancias y coeficientes, se proponen
las siguientes reglas de negocio para prevenir cancelaciones:

REGLAS AUTOMÁTICAS (ALERTAS):
  1. Si descuento > 30% Y monto > $3000 → ALERTA ROJA
     Razón: descuentos agresivos pueden indicar cliente insatisfecho
     Acción: llamar al cliente para confirmar satisfacción

  2. Si cantidad > 5 unidades Y es primera compra → ALERTA AMARILLA
     Razón: compras grandes de clientes nuevos tienen mayor riesgo
     Acción: verificar método de pago y datos de contacto

  3. Si producto con alta tasa histórica de cancelación → MONITOREO
     Razón: ciertos productos pueden tener problemas de calidad/expectativa
     Acción: revisar reseñas recientes del producto

  4. Si day = lunes y monto > $5000 → ALERTA AMARILLA
     Razón: las cancelaciones de alto valor son más comunes en lunes
     Acción: seguimiento prioritario

REGLAS PREVENTIVAS (PROACTIVAS):
  5. Ofrecer descuento de fidelidad del 5% a clientes con >3 compras
  6. Enviar encuesta de satisfacción post-compra (detecta insatisfacción temprana)
  7. Si pasan 7 días sin envío, contactar automáticamente al cliente
  8. Para productos premium (>$2000), incluir soporte prioritario

PROCEDIMIENTO ANTE ALERTA ROJA:
  - Contactar al cliente dentro de las primeras 24 horas
  - Ofrecer solución personalizada (cambio, descuento, soporte)
  - Escalar a gerente si el monto supera $10,000
  - Registrar motivo de la posible cancelación
""")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 11. REGLAS DE NEGOCIO PREVENTIVAS
3. ============================================================
4. Reglas basadas en feature importance y coeficientes

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 12. RECOMENDACIONES: ALERTAS TEMPRANAS DE CANCELACIÓN
# ============================================================
print("=" * 70)
print("RECOMENDACIONES — SISTEMA DE ALERTAS TEMPRANAS")
print("=" * 70)

umbral_recomendado = mejor_balance["umbral"] if "mejor_balance" in dir() else 0.4
print(f"""
SISTEMA DE ALERTAS TEMPRANAS
Umbral de riesgo recomendado: {umbral_recomendado:.2f}

ARQUITECTURA PROPUESTA:
  1. Batch diario: ejecutar modelo cada mañana sobre transacciones activas
  2. Segmentación de riesgo:
     - BAJO (prob < 0.2): Monitoreo normal
     - MEDIO (0.2-0.5): Alerta amarilla, seguimiento estándar
     - ALTO (prob > 0.5): Alerta roja, intervención prioritaria

  3. Dashboard en tiempo real:
     - Tabla de transacciones en riesgo con filtros
     - Histórico de cancelaciones vs predicciones
     - ROI acumulado del sistema

  4. Automatización:
     - Email automático al equipo de ventas con top 10 riesgos
     - SMS al cliente con oferta personalizada si riesgo alto
     - Actualización semanal del modelo con nuevos datos

MÉTRICAS DE ÉXITO:
  - Reducir tasa de cancelación de {tasa_cancelacion_actual*100:.1f}% a {tasa_cancelacion_actual*0.7*100:.1f}% (meta -30%)
  - Mantener precisión de alertas >= 40%
  - Atender al menos 80% de las alertas rojas dentro de 24h
  - ROI positivo a partir del tercer mes de implementación

PRÓXIMOS PASOS:
  1. Implementar el pipeline en producción (API Flask + base de datos)
  2. Recopilar feedback de ventas sobre calidad de alertas
  3. Iterar features: agregar datos de cliente, historial, navegación
  4. A/B test: grupo con alertas vs grupo sin alertas
""")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 12. RECOMENDACIONES: ALERTAS TEMPRANAS DE CANCELACIÓN
3. ============================================================

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# RESUMEN EJECUTIVO
# ============================================================
print("=" * 70)
print("RESUMEN EJECUTIVO — ANÁLISIS DE CANCELACIONES")
print("=" * 70)
print(f"""
Dataset: {len(ventas)} transacciones, {ventas['cancelada'].sum()} cancelaciones ({ventas['cancelada'].mean()*100:.1f}%)

Modelos evaluados:
  - Logistic Regression (AUC: {roc_lr:.4f})
  - Random Forest (AUC: {roc_rf:.4f})
  - Mejor modelo: {mejor_modelo}

Features más importantes:
  - {importancias.sort_values('importance', ascending=False).iloc[0]['feature']}
  - {importancias.sort_values('importance', ascending=False).iloc[1]['feature']}
  - {importancias.sort_values('importance', ascending=False).iloc[2]['feature']}

Impacto económico:
  - Costo de oportunidad actual: ${costo_oportunidad:,.2f}
  - Potencial de recuperación (30%): ${resultados_roi[2]['Ingreso_recuperado']:,}
  - ROI estimado: {resultados_roi[2]['ROI_%']}%

Recomendación:
  - Implementar sistema de alertas tempranas con Random Forest
  - Threshold balanceado: {mejor_balance['umbral']:.2f}
  - 8 reglas de negocio preventivas
  - Dashboard de monitoreo semanal
""")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. RESUMEN EJECUTIVO
3. ============================================================

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# EJERCICIOS COMPLEMENTARIOS
# ============================================================
print("=" * 70)
print("EJERCICIOS PARA PRACTICAR")
print("=" * 70)
print("""
1. Entrena XGBoost para cancelación y compara con Random Forest.
2. Implementa SMOTE para balancear clases y compara resultados.
3. Crea un pipeline completo con GridSearchCV para optimizar hiperparámetros.
4. Analiza el tiempo entre compra y cancelación (análisis de supervivencia).
5. Desarrolla una API simple que devuelva probabilidad de cancelación.
""")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. EJERCICIOS COMPLEMENTARIOS
3. ============================================================

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---


