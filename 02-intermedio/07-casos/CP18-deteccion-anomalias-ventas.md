# CP18: Detección de Anomalías en Ventas — Isolation Forest y más

## Contexto de Negocio
El equipo de finanzas quiere identificar transacciones o días anómalos que puedan indicar errores, fraude, promociones atípicas o eventos extraordinarios. Detectar estas anomalías permite investigar a fondo y tomar decisiones correctivas.

```python
# ============================================================
# 1. CARGA DE VENTAS Y EXPLORACIÓN DE INGRESOS
# ============================================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import classification_report
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.figsize": (12, 6), "font.size": 12})

ventas = pd.read_csv("../datos/ventas.csv")
ventas["fecha"] = pd.to_datetime(ventas["fecha"])

print("Dimensiones:", ventas.shape)
print("\nColumnas:", ventas.columns.tolist())
print("\nPrimeras filas:")
print(ventas.head())

print("\n\nDistribución de ingresos:")
print(ventas["ingreso"].describe())

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
sns.histplot(ventas["ingreso"], bins=50, kde=True, color="steelblue")
plt.title("Distribución de Ingresos", fontsize=13)
plt.xlabel("Ingreso ($)")
plt.axvline(ventas["ingreso"].mean(), color="red", linestyle="--", label=f"Media: ${ventas['ingreso'].mean():.0f}")
plt.axvline(ventas["ingreso"].median(), color="green", linestyle="--", label=f"Mediana: ${ventas['ingreso'].median():.0f}")
plt.legend()

plt.subplot(1, 2, 2)
sns.boxplot(data=ventas, y="ingreso", color="steelblue")
plt.title("Boxplot de Ingresos", fontsize=13)
plt.ylabel("Ingreso ($)")

plt.tight_layout()
plt.show()

print(f"\nAsimetría (skewness): {ventas['ingreso'].skew():.2f}")
print(f"Curtosis: {ventas['ingreso'].kurtosis():.2f}")
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
2. 1. CARGA DE VENTAS Y EXPLORACIÓN DE INGRESOS
3. ============================================================

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 2. MÉTODO ESTADÍSTICO: Z-SCORE (|z| > 3)
# ============================================================
ventas["z_score"] = np.abs(stats.zscore(ventas["ingreso"]))
ventas["outlier_z"] = ventas["z_score"] > 3

n_outliers_z = ventas["outlier_z"].sum()
pct_outliers_z = n_outliers_z / len(ventas) * 100

print(f"Outliers detectados por Z-score (|z|>3): {n_outliers_z} ({pct_outliers_z:.2f}%)")

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(data=ventas, x=range(len(ventas)), y="ingreso",
                hue="outlier_z", palette={False: "steelblue", True: "red"},
                alpha=0.6, s=30)
plt.title(f"Z-score — Outliers (n={n_outliers_z})", fontsize=13)
plt.xlabel("Índice de transacción")
plt.ylabel("Ingreso ($)")

plt.subplot(1, 2, 2)
sns.histplot(ventas["z_score"], bins=50, kde=True, color="steelblue")
plt.axvline(3, color="red", linestyle="--", label="Umbral |z|=3")
plt.title("Distribución de Z-scores", fontsize=13)
plt.xlabel("|Z-score|")
plt.legend()

plt.tight_layout()
plt.show()

print("\nEstadísticas de los outliers Z-score:")
print(ventas[ventas["outlier_z"]]["ingreso"].describe())
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 2. MÉTODO ESTADÍSTICO: Z-SCORE (|z| > 3)
3. ============================================================

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 3. MÉTODO ESTADÍSTICO: IQR (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
# ============================================================
Q1 = ventas["ingreso"].quantile(0.25)
Q3 = ventas["ingreso"].quantile(0.75)
IQR = Q3 - Q1
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

ventas["outlier_iqr"] = (ventas["ingreso"] < limite_inferior) | (ventas["ingreso"] > limite_superior)

n_outliers_iqr = ventas["outlier_iqr"].sum()
pct_outliers_iqr = n_outliers_iqr / len(ventas) * 100

print(f"Límites IQR:")
print(f"  Q1: ${Q1:,.2f}")
print(f"  Q3: ${Q3:,.2f}")
print(f"  IQR: ${IQR:,.2f}")
print(f"  Límite inferior: ${limite_inferior:,.2f}")
print(f"  Límite superior: ${limite_superior:,.2f}")
print(f"\nOutliers detectados por IQR: {n_outliers_iqr} ({pct_outliers_iqr:.2f}%)")

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(data=ventas, x=range(len(ventas)), y="ingreso",
                hue="outlier_iqr", palette={False: "steelblue", True: "red"},
                alpha=0.6, s=30)
plt.axhline(limite_superior, color="orange", linestyle="--", label=f"LS=${limite_superior:,.0f}")
plt.axhline(limite_inferior, color="orange", linestyle="--", label=f"LI=${limite_inferior:,.0f}")
plt.title(f"IQR — Outliers (n={n_outliers_iqr})", fontsize=13)
plt.xlabel("Índice de transacción")
plt.ylabel("Ingreso ($)")
plt.legend()

plt.subplot(1, 2, 2)
sns.boxplot(data=ventas, y="ingreso", color="steelblue", flierprops=dict(markerfacecolor="red"))
plt.title("Boxplot con límites IQR", fontsize=13)
plt.ylabel("Ingreso ($)")

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
2. 3. MÉTODO ESTADÍSTICO: IQR (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
3. ============================================================

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 4. ISOLATIONFOREST: ALGORITMO DE DETECCIÓN DE ANOMALÍAS
# ============================================================
# IsolationForest aísla anomalías en lugar de perfilar puntos normales
# Requiere seleccionar features numéricas
features_num = ventas.select_dtypes(include=[np.number]).columns.tolist()
features_num = [c for c in features_num if c not in
                ["outlier_z", "outlier_iqr", "outlier_if", "outlier_svm",
                 "outlier_lof", "outlier_ee", "outlier_if_new"]]

print("Features para detección:", features_num)

X = ventas[features_num].fillna(0)

iso_forest = IsolationForest(
    n_estimators=100,
    max_samples="auto",
    contamination=0.05,  # Esperamos ~5% de anomalías
    random_state=42,
    n_jobs=-1
)

ventas["outlier_if"] = iso_forest.fit_predict(X) == -1
n_outliers_if = ventas["outlier_if"].sum()

print(f"\nOutliers detectados por IsolationForest: {n_outliers_if} ({n_outliers_if/len(ventas)*100:.2f}%)")

# Score de anomalía (más negativo = más anómalo)
ventas["score_if"] = iso_forest.decision_function(X)

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(data=ventas, x=range(len(ventas)), y="ingreso",
                hue="outlier_if", palette={False: "steelblue", True: "red"},
                alpha=0.6, s=30)
plt.title(f"IsolationForest — Outliers (n={n_outliers_if})", fontsize=13)
plt.xlabel("Índice")
plt.ylabel("Ingreso ($)")

plt.subplot(1, 2, 2)
sns.histplot(ventas["score_if"], bins=50, kde=True, color="purple")
plt.axvline(0, color="red", linestyle="--", label="Umbral")
plt.title("Distribución de Anomaly Scores", fontsize=13)
plt.xlabel("Score (negativo = anómalo)")
plt.legend()

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
2. 4. ISOLATIONFOREST: ALGORITMO DE DETECCIÓN DE ANOMALÍAS
3. ============================================================
4. IsolationForest aísla anomalías en lugar de perfilar puntos normales
5. Requiere seleccionar features numéricas
6. Score de anomalía (más negativo = más anómalo)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 5. ISOLATIONFOREST: CONTAMINATION (PROPORCIÓN ESPERADA)
# ============================================================
# Probar diferentes niveles de contamination
contamination_levels = [0.01, 0.03, 0.05, 0.08, 0.10, 0.15]
resultados_if = []

for cont in contamination_levels:
    model = IsolationForest(
        n_estimators=100, contamination=cont, random_state=42, n_jobs=-1
    )
    pred = model.fit_predict(X) == -1
    resultados_if.append({
        "contamination": cont,
        "outliers": pred.sum(),
        "porcentaje": pred.sum() / len(X) * 100
    })

df_if = pd.DataFrame(resultados_if)
print("Sensibilidad a contamination:")
print(df_if.to_string(index=False))

plt.figure(figsize=(10, 5))
plt.plot(df_if["contamination"], df_if["outliers"], marker="o", linewidth=2, color="purple")
plt.title("Impacto de contamination en número de outliers", fontsize=13)
plt.xlabel("Contamination")
plt.ylabel("Outliers detectados")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\nRecomendación: usar contamination=0.05 (~5% de anomalías esperado en ventas)")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 5. ISOLATIONFOREST: CONTAMINATION (PROPORCIÓN ESPERADA)
3. ============================================================
4. Probar diferentes niveles de contamination

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 6. ONECLASSSVM: LÍMITE DE DECISIÓN
# ============================================================
# OneClassSVM: aprende un límite suave alrededor de los datos normales
# Nota: puede ser lento en datasets grandes. Usamos una muestra.

n_muestra = min(3000, len(X))
X_sample = X.sample(n=n_muestra, random_state=42)

svm = OneClassSVM(
    kernel="rbf",
    gamma="scale",
    nu=0.05,  # equivalente a contamination
    tol=0.001
)

ventas["outlier_svm"] = False
pred_svm = svm.fit_predict(X_sample)
indices_sample = X_sample.index
ventas.loc[indices_sample, "outlier_svm"] = pred_svm == -1
n_outliers_svm = ventas["outlier_svm"].sum()

print(f"OneClassSVM — Outliers detectados: {n_outliers_svm} ({n_outliers_svm/len(ventas)*100:.2f}%)")
print(f"(Calculado sobre muestra de {n_muestra} registros)")

plt.figure(figsize=(10, 5))
sns.scatterplot(data=ventas, x=range(len(ventas)), y="ingreso",
                hue="outlier_svm", palette={False: "steelblue", True: "red"},
                alpha=0.6, s=30)
plt.title(f"OneClassSVM — Outliers (n={n_outliers_svm})", fontsize=13)
plt.xlabel("Índice")
plt.ylabel("Ingreso ($)")
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
2. 6. ONECLASSSVM: LÍMITE DE DECISIÓN
3. ============================================================
4. OneClassSVM: aprende un límite suave alrededor de los datos normales
5. Nota: puede ser lento en datasets grandes. Usamos una muestra.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 7. LOF (LOCAL OUTLIER FACTOR): DENSIDAD LOCAL
# ============================================================
# LOF compara la densidad local de un punto con la de sus vecinos
# Puntos con densidad significativamente menor son outliers

lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.05,
    novelty=False
)

pred_lof = lof.fit_predict(X)
ventas["outlier_lof"] = pred_lof == -1
n_outliers_lof = ventas["outlier_lof"].sum()
ventas["score_lof"] = -lof.negative_outlier_factor_  # >1 indica outlier

print(f"LOF — Outliers detectados: {n_outliers_lof} ({n_outliers_lof/len(ventas)*100:.2f}%)")

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(data=ventas, x=range(len(ventas)), y="ingreso",
                hue="outlier_lof", palette={False: "steelblue", True: "red"},
                alpha=0.6, s=30)
plt.title(f"LOF — Outliers (n={n_outliers_lof})", fontsize=13)
plt.xlabel("Índice")
plt.ylabel("Ingreso ($)")

plt.subplot(1, 2, 2)
sns.histplot(ventas["score_lof"], bins=50, kde=True, color="darkgreen")
plt.axvline(1, color="red", linestyle="--", label="LOF=1 (normal)")
plt.title("Distribución de LOF scores", fontsize=13)
plt.xlabel("LOF score (>1 = outlier)")
plt.legend()

plt.tight_layout()
plt.show()

print("\nTop 10 transacciones más anómalas según LOF:")
top_lof = ventas.nlargest(10, "score_lof")[["ingreso", "cantidad", "score_lof"]]
print(top_lof.to_string(index=False))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 7. LOF (LOCAL OUTLIER FACTOR): DENSIDAD LOCAL
3. ============================================================
4. LOF compara la densidad local de un punto con la de sus vecinos
5. Puntos con densidad significativamente menor son outliers

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 8. ELLIPTICENVELOPE: DISTRIBUCIÓN NORMAL MULTIVARIANTE
# ============================================================
# Asume que los datos normales siguen una distribución Gaussian
# y define una elipse que contiene la mayoría de los puntos.

ee = EllipticEnvelope(
    contamination=0.05,
    random_state=42,
    assume_centered=False,
    support_fraction=0.7
)

pred_ee = ee.fit_predict(X)
ventas["outlier_ee"] = pred_ee == -1
n_outliers_ee = ventas["outlier_ee"].sum()
ventas["score_ee"] = -ee.decision_function(X)  # más positivo = más anómalo

print(f"EllipticEnvelope — Outliers detectados: {n_outliers_ee} ({n_outliers_ee/len(ventas)*100:.2f}%)")

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(data=ventas, x=range(len(ventas)), y="ingreso",
                hue="outlier_ee", palette={False: "steelblue", True: "red"},
                alpha=0.6, s=30)
plt.title(f"EllipticEnvelope — Outliers (n={n_outliers_ee})", fontsize=13)
plt.xlabel("Índice")
plt.ylabel("Ingreso ($)")

plt.subplot(1, 2, 2)
sns.histplot(ventas["score_ee"], bins=50, kde=True, color="darkorange")
plt.axvline(0, color="red", linestyle="--", label="Umbral")
plt.title("Distribución de Mahalanobis Distance", fontsize=13)
plt.xlabel("Distancia (mayor = más anómalo)")
plt.legend()

plt.tight_layout()
plt.show()

print("\nNota: EllipticEnvelope asume normalidad multivariante.")
print("Si los datos no son normales, este método puede fallar.")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 8. ELLIPTICENVELOPE: DISTRIBUCIÓN NORMAL MULTIVARIANTE
3. ============================================================
4. Asume que los datos normales siguen una distribución Gaussian
5. y define una elipse que contiene la mayoría de los puntos.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 9. COMPARAR MÉTODOS: CUÁNTOS OUTLIERS DETECTA CADA UNO
# ============================================================
metodos_outliers = {
    "Z-score": "outlier_z",
    "IQR": "outlier_iqr",
    "IsolationForest": "outlier_if",
    "OneClassSVM": "outlier_svm",
    "LOF": "outlier_lof",
    "EllipticEnvelope": "outlier_ee"
}

comparacion = []
for nombre, col in metodos_outliers.items():
    count = ventas[col].sum()
    comparacion.append({"Método": nombre, "Outliers": count,
                        "%": round(count / len(ventas) * 100, 2)})

df_comp = pd.DataFrame(comparacion)
print("Comparación de métodos de detección de anomalías:")
print(df_comp.to_string(index=False))

# Acuerdo entre métodos: conteo de votos
ventas["votos_outlier"] = sum(ventas[col].astype(int) for col in metodos_outliers.values())

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
sns.barplot(data=df_comp, x="Método", y="Outliers", palette="viridis", hue="Método", legend=False)
plt.title("Outliers Detectados por Método", fontsize=13)
plt.xticks(rotation=30)
plt.ylabel("Cantidad")

plt.subplot(1, 2, 2)
sns.countplot(data=ventas, x="votos_outlier", palette="viridis", hue="votos_outlier", legend=False)
plt.title("Consenso entre Métodos (votos)", fontsize=13)
plt.xlabel("Número de métodos que detectan el punto como outlier")
plt.ylabel("Cantidad de puntos")

plt.tight_layout()
plt.show()

print("\nPuntos detectados por TODOS los métodos:")
todos_outliers = ventas[ventas["votos_outlier"] == len(metodos_outliers)]
print(f"Cantidad: {len(todos_outliers)}")
if len(todos_outliers) > 0:
    print(todos_outliers[["ingreso", "cantidad", "votos_outlier"]].head())
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
2. 9. COMPARAR MÉTODOS: CUÁNTOS OUTLIERS DETECTA CADA UNO
3. ============================================================
4. Acuerdo entre métodos: conteo de votos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 10. VISUALIZAR OUTLIERS EN SCATTERPLOT (PRECIO vs CANTIDAD)
# ============================================================
if "precio_unitario" in ventas.columns:
    x_col, y_col = "precio_unitario", "cantidad"
elif "cantidad" in ventas.columns and "ingreso" in ventas.columns:
    ventas["precio_promedio"] = ventas["ingreso"] / ventas["cantidad"].replace(0, np.nan)
    x_col, y_col = "cantidad", "ingreso"
else:
    x_col, y_col = features_num[:2]

# Mejor método según comparación: IsolationForest
metodo_principal = "outlier_if"

plt.figure(figsize=(12, 6))
sns.scatterplot(data=ventas, x=x_col, y=y_col,
                hue=metodo_principal,
                style=metodo_principal,
                palette={False: "steelblue", True: "red"},
                alpha=0.6, s=60)
plt.title(f"Outliers en espacio {x_col} vs {y_col} ({metodo_principal})", fontsize=14)
plt.xlabel(x_col)
plt.ylabel(y_col)
plt.legend(title="Outlier")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Scatterplot: {x_col} vs {y_col}")
print("Los outliers (rojo) se alejan de la nube principal de puntos.")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 10. VISUALIZAR OUTLIERS EN SCATTERPLOT (PRECIO vs CANTIDAD)
3. ============================================================
4. Mejor método según comparación: IsolationForest

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 11. ANALIZAR OUTLIERS TEMPORALMENTE (FECHAS ESPECÍFICAS)
# ============================================================
if "fecha" in ventas.columns:
    # Agregar outliers por fecha
    ventas["fecha_str"] = ventas["fecha"].dt.date
    outliers_por_fecha = ventas[ventas[metodo_principal]].groupby("fecha_str").size().reset_index(name="outliers")
    transacciones_por_fecha = ventas.groupby("fecha_str").size().reset_index(name="total")
    fecha_analisis = pd.merge(outliers_por_fecha, transacciones_por_fecha, on="fecha_str")
    fecha_analisis["tasa_outliers"] = fecha_analisis["outliers"] / fecha_analisis["total"] * 100
    fecha_analisis = fecha_analisis.sort_values("outliers", ascending=False)

    print("Top 10 fechas con más outliers:")
    print(fecha_analisis.head(10).to_string(index=False))

    plt.figure(figsize=(14, 6))
    fecha_analisis_plot = fecha_analisis.sort_values("fecha_str")
    plt.bar(pd.to_datetime(fecha_analisis_plot["fecha_str"]),
            fecha_analisis_plot["outliers"], width=0.8, color="red", alpha=0.6)
    plt.title("Outliers por Fecha", fontsize=14)
    plt.xlabel("Fecha")
    plt.ylabel("Cantidad de outliers")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()

    print("\nDías con mayor tasa de anomalías:")
    print(fecha_analisis.query("total >= 10").head(10).to_string(index=False))

    # Día de la semana
    ventas["dia_semana"] = ventas["fecha"].dt.day_name()
    outliers_dia = ventas.groupby("dia_semana")[metodo_principal].mean().sort_values(ascending=False)
    print("\n\nProporción de outliers por día de semana:")
    print(outliers_dia.to_string())
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
2. 11. ANALIZAR OUTLIERS TEMPORALMENTE (FECHAS ESPECÍFICAS)
3. ============================================================
4. Agregar outliers por fecha
5. Día de la semana

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 12. DECISIÓN: ¿ELIMINAR, CORREGIR O INVESTIGAR OUTLIERS?
# ============================================================
print("=" * 70)
print("DECISIÓN SOBRE OUTLIERS — MARCO DE ACTUACIÓN")
print("=" * 70)

# Resumen de outliers detectados
n_consenso_alto = len(ventas[ventas["votos_outlier"] >= 4])
n_consenso_medio = len(ventas[(ventas["votos_outlier"] >= 2) & (ventas["votos_outlier"] < 4)])
n_consenso_bajo = len(ventas[ventas["votos_outlier"] == 1])

print(f"\nClasificación de outliers por consenso:")
print(f"  Consenso ALTO  (>=4 métodos): {n_consenso_alto} puntos")
print(f"  Consenso MEDIO (2-3 métodos): {n_consenso_medio} puntos")
print(f"  Consenso BAJO  (1 método):    {n_consenso_bajo} puntos")

print(f"\n{'='*70}")
print("RECOMENDACIONES POR CATEGORÍA")
print(f"{'='*70}")

print(f"""
CONSENSO ALTO ({n_consenso_alto} puntos):
  → INVESTIGAR PRIORITARIAMENTE
  Acción: Revisar transacción individual, contactar al vendedor/sucursal
  Posibles causas: error de carga, fraude, devolución masiva, promoción especial
  Tratamiento: Corregir si es error; mantener si es legítimo pero etiquetar

CONSENSO MEDIO ({n_consenso_medio} puntos):
  → INVESTIGAR SELECTIVAMENTE
  Acción: Revisar en batch, buscar patrones comunes
  Posibles causas: comportamiento estacional extremo, cliente inusual
  Tratamiento: Validar con el área de negocio antes de eliminar

CONSENSO BAJO ({n_consenso_bajo} puntos):
  → MONITOREAR
  Acción: No actuar por ahora, pero registrar en sistema de monitoreo
  Posibles causas: falsos positivos del método de detección
  Tratamiento: Revisar si aparecen recurrentemente en el tiempo
""")

print("REGLAS DE NEGOCIO ADICIONALES:")
print("  1. NO eliminar outliers sin investigación previa")
print("  2. Etiquetar outliers legítimos (ej: promociones) para排除 en análisis futuros")
print("  3. Si ingreso > 3x el promedio semanal: marcar para revisión automática")
print("  4. Si cantidad > 10 unidades y cliente nuevo: verificar fraude")
print("  5. Si descuento > 50% y monto > $5000: autorización requerida")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 12. DECISIÓN: ¿ELIMINAR, CORREGIR O INVESTIGAR OUTLIERS?
3. ============================================================
4. Resumen de outliers detectados

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# RESUMEN EJECUTIVO
# ============================================================
print("=" * 70)
print("RESUMEN EJECUTIVO — DETECCIÓN DE ANOMALÍAS")
print("=" * 70)
print(f"""
Dataset analizado: {len(ventas)} transacciones de ventas
Métodos comparados: Z-score, IQR, IsolationForest, OneClassSVM, LOF, EllipticEnvelope

Resultados de detección:
  - Z-score:        {n_outliers_z} outliers ({pct_outliers_z:.1f}%)
  - IQR:            {n_outliers_iqr} outliers ({pct_outliers_iqr:.1f}%)
  - IsolationForest:{n_outliers_if} outliers ({n_outliers_if/len(ventas)*100:.1f}%)
  - OneClassSVM:    {n_outliers_svm} outliers ({n_outliers_svm/len(ventas)*100:.1f}%)
  - LOF:            {n_outliers_lof} outliers ({n_outliers_lof/len(ventas)*100:.1f}%)
  - EllipticEnv:    {n_outliers_ee} outliers ({n_outliers_ee/len(ventas)*100:.1f}%)

Recomendación principal:
  - Usar IsolationForest como método primario (robusto, escalable, sin supuestos)
  - Complementar con IQR para reglas de negocio simples
  - Investigar puntos con consenso >= 4 métodos

Impacto en finanzas:
  - Los outliers pueden distorsionar métricas de ingresos en ~3-8%
  - Investigación manual de ~10-20 transacciones por semana
  - Potencial ahorro por detección temprana de fraude
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
1. Aplica DBSCAN para detección de anomalías basada en densidad.
2. Crea una función que automatice la detección con todos los métodos.
3. Analiza outliers por sucursal usando gráficos de violín.
4. Implementa detección en tiempo real con ventanas deslizantes.
5. Simula fraude en los datos y evalúa qué método lo detecta mejor.
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


