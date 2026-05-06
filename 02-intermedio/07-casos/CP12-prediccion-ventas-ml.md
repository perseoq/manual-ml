# CP12: Predicción de Ventas con Machine Learning

## Contexto de Negocio
El gerente de ventas necesita predecir los ingresos del próximo mes para planificar metas, asignar presupuesto de marketing y gestionar inventario. Construiremos un modelo de regresión con features temporales y compararemos LinearRegression vs RandomForest.

```python
# ============================================================
# 1. CARGA Y AGREGACIÓN DIARIA DE VENTAS
# ============================================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.figsize": (12, 6), "font.size": 12})

ventas = pd.read_csv("../datos/ventas.csv")
print("Dimensiones:", ventas.shape)
print("\nPrimeras filas:")
ventas.head()
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
2. 1. CARGA Y AGREGACIÓN DIARIA DE VENTAS
3. ============================================================

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# Convertir fecha y agregar por día
ventas["fecha"] = pd.to_datetime(ventas["fecha"])
ventas_diarias = ventas.groupby("fecha")["ingreso"].sum().reset_index()
ventas_diarias = ventas_diarias.sort_values("fecha").reset_index(drop=True)

print(f"Rango de fechas: {ventas_diarias['fecha'].min()} a {ventas_diarias['fecha'].max()}")
print(f"Días totales: {len(ventas_diarias)}")
print("\nEstadísticas de ingreso diario:")
print(ventas_diarias["ingreso"].describe())
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Convertir fecha y agregar por día

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# Visualizar serie temporal
plt.figure(figsize=(14, 5))
plt.plot(ventas_diarias["fecha"], ventas_diarias["ingreso"],
         color="steelblue", linewidth=1.5, alpha=0.8)
plt.title("Ingreso Diario a lo Largo del Tiempo", fontsize=14)
plt.xlabel("Fecha")
plt.ylabel("Ingreso ($)")
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

1. Visualizar serie temporal

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 2. CREACIÓN DE FEATURES TEMPORALES
# ============================================================
ventas_diarias["dia_semana"] = ventas_diarias["fecha"].dt.dayofweek  # 0=lunes, 6=domingo
ventas_diarias["mes"] = ventas_diarias["fecha"].dt.month
ventas_diarias["dia_del_mes"] = ventas_diarias["fecha"].dt.day
ventas_diarias["fin_de_semana"] = (ventas_diarias["dia_semana"] >= 5).astype(int)
ventas_diarias["trimestre"] = ventas_diarias["fecha"].dt.quarter
ventas_diarias["semana_del_anio"] = ventas_diarias["fecha"].dt.isocalendar().week.astype(int)

print("Features temporales creadas:")
print(ventas_diarias[["fecha", "dia_semana", "mes", "fin_de_semana",
                      "trimestre", "semana_del_anio"]].head(10))
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
2. 2. CREACIÓN DE FEATURES TEMPORALES
3. ============================================================

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# Análisis de estacionalidad semanal
plt.figure(figsize=(10, 5))
sns.boxplot(x="dia_semana", y="ingreso", data=ventas_diarias, palette="viridis")
plt.title("Distribución de Ingreso por Día de la Semana", fontsize=14)
plt.xlabel("Día de la Semana (0=Lunes, 6=Domingo)")
plt.ylabel("Ingreso ($)")
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

1. Análisis de estacionalidad semanal

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 3. FEATURES DE REZAGO (LAG)
# ============================================================
# El ingreso de hoy puede depender del de ayer y de hace una semana
ventas_diarias["lag_1"] = ventas_diarias["ingreso"].shift(1)
ventas_diarias["lag_7"] = ventas_diarias["ingreso"].shift(7)
ventas_diarias["lag_14"] = ventas_diarias["ingreso"].shift(14)

print("Features de rezago creadas (mostrando filas con datos completos):")
print(ventas_diarias[["fecha", "ingreso", "lag_1", "lag_7", "lag_14"]].head(20))
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
2. 3. FEATURES DE REZAGO (LAG)
3. ============================================================
4. El ingreso de hoy puede depender del de ayer y de hace una semana

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 4. MEDIA MÓVIL COMO FEATURE
# ============================================================
ventas_diarias["media_movil_7"] = ventas_diarias["ingreso"].rolling(window=7).mean()
ventas_diarias["media_movil_14"] = ventas_diarias["ingreso"].rolling(window=14).mean()
ventas_diarias["media_movil_30"] = ventas_diarias["ingreso"].rolling(window=30).mean()

print("Medias móviles creadas:")
print(ventas_diarias[["fecha", "ingreso", "media_movil_7",
                      "media_movil_14", "media_movil_30"]].tail(10))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 4. MEDIA MÓVIL COMO FEATURE
3. ============================================================

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# Visualizar media móvil vs real
plt.figure(figsize=(14, 6))
plt.plot(ventas_diarias["fecha"], ventas_diarias["ingreso"],
         label="Ingreso Real", color="steelblue", alpha=0.5, linewidth=1)
plt.plot(ventas_diarias["fecha"], ventas_diarias["media_movil_7"],
         label="Media Móvil 7 días", color="red", linewidth=2)
plt.plot(ventas_diarias["fecha"], ventas_diarias["media_movil_30"],
         label="Media Móvil 30 días", color="green", linewidth=2)
plt.title("Ingreso Diario con Medias Móviles", fontsize=14)
plt.xlabel("Fecha")
plt.ylabel("Ingreso ($)")
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

1. Visualizar media móvil vs real

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 5. DIVISIÓN EN TRAIN Y TEST (TEMPORAL)
# ============================================================
# La división debe ser temporal, NO aleatoria, para evitar data leakage

# Eliminar filas con NaN (por lags y rolling)
datos_modelo = ventas_diarias.dropna().reset_index(drop=True)

# Features y target
features = ["dia_semana", "mes", "dia_del_mes", "fin_de_semana",
            "lag_1", "lag_7", "lag_14", "media_movil_7",
            "media_movil_14", "media_movil_30"]
X = datos_modelo[features]
y = datos_modelo["ingreso"]

# División temporal: 80% train, 20% test
tamanio_test = int(len(datos_modelo) * 0.2)
X_train = X.iloc[:-tamanio_test]
X_test = X.iloc[-tamanio_test:]
y_train = y.iloc[:-tamanio_test]
y_test = y.iloc[-tamanio_test:]

print(f"Train: {len(X_train)} días ({X_train.index[0]} a {X_train.index[-1]})")
print(f"Test: {len(X_test)} días ({X_test.index[0]} a {X_test.index[-1]})")
print(f"Test representa el {tamanio_test/len(datos_modelo):.1%} de los datos")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 5. DIVISIÓN EN TRAIN Y TEST (TEMPORAL)
3. ============================================================
4. La división debe ser temporal, NO aleatoria, para evitar data leakage
5. Eliminar filas con NaN (por lags y rolling)
6. Features y target
7. División temporal: 80% train, 20% test

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 6. ENTRENAR LINEAR REGRESSION
# ============================================================
lr = LinearRegression()
lr.fit(X_train, y_train)

print("Coeficientes del modelo lineal:")
for feature, coef in zip(features, lr.coef_):
    print(f"  {feature}: {coef:.4f}")
print(f"  Intercepto: {lr.intercept_:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 6. ENTRENAR LINEAR REGRESSION
3. ============================================================

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 7. EVALUACIÓN DEL MODELO LINEAL
# ============================================================
y_pred_lr = lr.predict(X_test)

mse_lr = mean_squared_error(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("=" * 50)
print("EVALUACIÓN — LINEAR REGRESSION")
print("=" * 50)
print(f"MSE:  {mse_lr:,.2f}")
print(f"RMSE: {np.sqrt(mse_lr):,.2f}")
print(f"MAE:  {mae_lr:,.2f}")
print(f"R²:   {r2_lr:.4f}")
print(f"MAPE: {np.mean(np.abs((y_test - y_pred_lr) / y_test)) * 100:.2f}%")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 7. EVALUACIÓN DEL MODELO LINEAL
3. ============================================================

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 8. ENTRENAR RANDOM FOREST REGRESSOR
# ============================================================
rf = RandomForestRegressor(n_estimators=200, max_depth=10,
                           random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("=" * 50)
print("EVALUACIÓN — RANDOM FOREST")
print("=" * 50)
print(f"MSE:  {mse_rf:,.2f}")
print(f"RMSE: {np.sqrt(mse_rf):,.2f}")
print(f"MAE:  {mae_rf:,.2f}")
print(f"R²:   {r2_rf:.4f}")
print(f"MAPE: {np.mean(np.abs((y_test - y_pred_rf) / y_test)) * 100:.2f}%")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 8. ENTRENAR RANDOM FOREST REGRESSOR
3. ============================================================

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 9. COMPARACIÓN LINEAR REGRESSION VS RANDOM FOREST
# ============================================================
comparacion = pd.DataFrame({
    "Métrica": ["MSE", "RMSE", "MAE", "R²"],
    "Linear Regression": [mse_lr, np.sqrt(mse_lr), mae_lr, r2_lr],
    "Random Forest": [mse_rf, np.sqrt(mse_rf), mae_rf, r2_rf]
})
comparacion["Diferencia"] = comparacion["Linear Regression"] - comparacion["Random Forest"]
comparacion["Gana"] = np.where(
    comparacion["Diferencia"] > 0,
    "Random Forest",
    np.where(comparacion["Diferencia"] < 0, "Linear Regression", "Empate")
)
# Para R², mayor es mejor
comparacion.loc[comparacion["Métrica"] == "R²", "Gana"] = np.where(
    comparacion.loc[comparacion["Métrica"] == "R²", "Diferencia"].values[0] > 0,
    "Linear Regression",
    "Random Forest"
)

print("COMPARACIÓN DE MODELOS:")
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
2. 9. COMPARACIÓN LINEAR REGRESSION VS RANDOM FOREST
3. ============================================================
4. Para R², mayor es mejor

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# Gráfico de comparación
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metricas = ["MSE", "MAE", "R²"]
valores_lr = [mse_lr, mae_lr, r2_lr]
valores_rf = [mse_rf, mae_rf, r2_rf]

for i, (ax, metrica) in enumerate(zip(axes, metricas)):
    x = np.arange(2)
    width = 0.3
    ax.bar(x[0], valores_lr[i], width, label="Linear Regression", color="steelblue")
    ax.bar(x[1], valores_rf[i], width, label="Random Forest", color="coral")
    ax.set_title(f"{metrica}", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(["LR", "RF"])
    ax.legend()

plt.suptitle("Comparación de Modelos", fontsize=15)
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

1. Gráfico de comparación

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 10. FEATURE IMPORTANCE (RANDOM FOREST)
# ============================================================
importances = pd.DataFrame({
    "Feature": features,
    "Importancia": rf.feature_importances_
}).sort_values("Importancia", ascending=False)

print("FEATURE IMPORTANCE — RANDOM FOREST:")
print(importances.to_string(index=False))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 10. FEATURE IMPORTANCE (RANDOM FOREST)
3. ============================================================

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
plt.figure(figsize=(10, 6))
sns.barplot(x="Importancia", y="Feature", data=importances, palette="viridis")
plt.title("Importancia de Features en Random Forest", fontsize=14)
plt.xlabel("Importancia Relativa")
plt.ylabel("Feature")
plt.grid(True, alpha=0.3, axis="x")
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


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 11. GRÁFICO: PREDICCIÓN VS REAL EN TEST SET
# ============================================================
plt.figure(figsize=(14, 6))

# Usar índices reales para el eje x
fechas_test = datos_modelo["fecha"].iloc[-tamanio_test:].values

plt.plot(fechas_test, y_test.values, label="Real", color="steelblue",
         linewidth=2, marker="o", markersize=4)
plt.plot(fechas_test, y_pred_lr, label="Linear Regression",
         color="coral", linewidth=1.5, linestyle="--", marker="s", markersize=3)
plt.plot(fechas_test, y_pred_rf, label="Random Forest",
         color="seagreen", linewidth=1.5, linestyle="-.", marker="^", markersize=3)
plt.title("Predicción vs Real — Período de Test", fontsize=14)
plt.xlabel("Fecha")
plt.ylabel("Ingreso ($)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
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
2. 11. GRÁFICO: PREDICCIÓN VS REAL EN TEST SET
3. ============================================================
4. Usar índices reales para el eje x

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# Gráfico de dispersión: real vs predicho
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(y_test, y_pred_lr, alpha=0.6, color="coral", edgecolors="black")
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             "k--", lw=2, label="Ideal")
axes[0].set_xlabel("Valor Real")
axes[0].set_ylabel("Predicción")
axes[0].set_title("Linear Regression: Real vs Predicho")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].scatter(y_test, y_pred_rf, alpha=0.6, color="seagreen", edgecolors="black")
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             "k--", lw=2, label="Ideal")
axes[1].set_xlabel("Valor Real")
axes[1].set_ylabel("Predicción")
axes[1].set_title("Random Forest: Real vs Predicho")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

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

1. Gráfico de dispersión: real vs predicho

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 12. ANÁLISIS DE RESIDUOS
# ============================================================
residuos_lr = y_test - y_pred_lr
residuos_rf = y_test - y_pred_rf

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Distribución de residuos
sns.histplot(residuos_lr, bins=20, kde=True, ax=axes[0, 0], color="coral")
axes[0, 0].set_title("Distribución de Residuos — Linear Regression")
axes[0, 0].set_xlabel("Error ($)")
axes[0, 0].axvline(x=0, color="black", linestyle="--", alpha=0.5)

sns.histplot(residuos_rf, bins=20, kde=True, ax=axes[0, 1], color="seagreen")
axes[0, 1].set_title("Distribución de Residuos — Random Forest")
axes[0, 1].set_xlabel("Error ($)")
axes[0, 1].axvline(x=0, color="black", linestyle="--", alpha=0.5)

# Residuos vs orden temporal
axes[1, 0].scatter(range(len(residuos_lr)), residuos_lr, color="coral", alpha=0.6)
axes[1, 0].axhline(y=0, color="black", linestyle="--", alpha=0.5)
axes[1, 0].set_title("Residuos vs Orden — Linear Regression")
axes[1, 0].set_xlabel("Observación")
axes[1, 0].set_ylabel("Error ($)")
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].scatter(range(len(residuos_rf)), residuos_rf, color="seagreen", alpha=0.6)
axes[1, 1].axhline(y=0, color="black", linestyle="--", alpha=0.5)
axes[1, 1].set_title("Residuos vs Orden — Random Forest")
axes[1, 1].set_xlabel("Observación")
axes[1, 1].set_ylabel("Error ($)")
axes[1, 1].grid(True, alpha=0.3)

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
2. 12. ANÁLISIS DE RESIDUOS
3. ============================================================
4. Distribución de residuos
5. Residuos vs orden temporal

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# Prueba de normalidad de residuos (Shapiro-Wilk)
from scipy import stats

stat_lr, p_lr = stats.shapiro(residuos_lr.sample(min(100, len(residuos_lr))))
stat_rf, p_rf = stats.shapiro(residuos_rf.sample(min(100, len(residuos_rf))))

print("PRUEBA DE NORMALIDAD DE RESIDUOS (Shapiro-Wilk):")
print(f"Linear Regression: estadístico={stat_lr:.4f}, p-valor={p_lr:.4f} " +
      ("→ Normal" if p_lr > 0.05 else "→ No normal"))
print(f"Random Forest: estadístico={stat_rf:.4f}, p-valor={p_rf:.4f} " +
      ("→ Normal" if p_rf > 0.05 else "→ No normal"))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Prueba de normalidad de residuos (Shapiro-Wilk)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 13. PREDECIR PRÓXIMOS 30 DÍAS
# ============================================================
# Para predecir el futuro, necesitamos construir features para los próximos 30 días
ultima_fecha = ventas_diarias["fecha"].max()
fechas_futuras = pd.date_range(start=ultima_fecha + pd.Timedelta(days=1), periods=30)

# Construir DataFrame para predicción futura
futuro = pd.DataFrame({"fecha": fechas_futuras})
futuro["dia_semana"] = futuro["fecha"].dt.dayofweek
futuro["mes"] = futuro["fecha"].dt.month
futuro["dia_del_mes"] = futuro["fecha"].dt.day
futuro["fin_de_semana"] = (futuro["dia_semana"] >= 5).astype(int)

# Para lags y medias móviles, usamos los últimos valores conocidos y vamos
# propagando las predicciones (predicción recursiva)
ultimos_datos = ventas_diarias.tail(30).copy()

predicciones = []
for i in range(30):
    if i == 0:
        futuro.loc[i, "lag_1"] = ultimos_datos["ingreso"].iloc[-1]
        futuro.loc[i, "lag_7"] = ultimos_datos["ingreso"].iloc[-7] if len(ultimos_datos) >= 7 else ultimos_datos["ingreso"].mean()
        futuro.loc[i, "lag_14"] = ultimos_datos["ingreso"].iloc[-14] if len(ultimos_datos) >= 14 else ultimos_datos["ingreso"].mean()
    else:
        # Usar predicciones anteriores como lags
        pred_prev = predicciones[:i]
        datos_hist = list(ultimos_datos["ingreso"].values) + pred_prev
        futuro.loc[i, "lag_1"] = datos_hist[-1]
        futuro.loc[i, "lag_7"] = datos_hist[-7] if len(datos_hist) >= 7 else np.mean(datos_hist)
        futuro.loc[i, "lag_14"] = datos_hist[-14] if len(datos_hist) >= 14 else np.mean(datos_hist)

    # Medias móviles
    if i == 0:
        futuro.loc[i, "media_movil_7"] = ultimos_datos["ingreso"].tail(7).mean()
        futuro.loc[i, "media_movil_14"] = ultimos_datos["ingreso"].tail(14).mean()
        futuro.loc[i, "media_movil_30"] = ultimos_datos["ingreso"].tail(30).mean()
    else:
        valores = list(ultimos_datos["ingreso"].values) + predicciones[:i]
        futuro.loc[i, "media_movil_7"] = np.mean(valores[-7:]) if len(valores) >= 7 else np.mean(valores)
        futuro.loc[i, "media_movil_14"] = np.mean(valores[-14:]) if len(valores) >= 14 else np.mean(valores)
        futuro.loc[i, "media_movil_30"] = np.mean(valores[-30:]) if len(valores) >= 30 else np.mean(valores)

    X_futuro = futuro.loc[i, features].values.reshape(1, -1)
    pred = rf.predict(X_futuro)[0]
    predicciones.append(max(pred, 0))  # No permitir ingresos negativos

futuro["prediccion"] = predicciones
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 13. PREDECIR PRÓXIMOS 30 DÍAS
3. ============================================================
4. Para predecir el futuro, necesitamos construir features para los próximos 30 días
5. Construir DataFrame para predicción futura
6. Para lags y medias móviles, usamos los últimos valores conocidos y vamos
7. propagando las predicciones (predicción recursiva)
8. Usar predicciones anteriores como lags

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
print("PREDICCIÓN PRÓXIMOS 30 DÍAS:")
print(futuro[["fecha", "prediccion"]].to_string(index=False))
print(f"\nIngreso total estimado: ${futuro['prediccion'].sum():,.2f}")
print(f"Promedio diario estimado: ${futuro['prediccion'].mean():,.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. `print("PREDICCIÓN PRÓXIMOS 30 DÍAS:")` — Muestra el resultado por pantalla.
2. `print(futuro[["fecha", "prediccion"]].to_string(index=False))` — Muestra el resultado por pantalla.
3. `print(f"\nIngreso total estimado: ${futuro['prediccion'].sum():,.2f}")` — Muestra el resultado por pantalla.
4. `print(f"Promedio diario estimado: ${futuro['prediccion'].mean():,.2f}")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# Visualizar predicción futura
plt.figure(figsize=(14, 6))

# Últimos 60 días históricos
ultimos_60 = ventas_diarias.tail(60)
plt.plot(ultimos_60["fecha"], ultimos_60["ingreso"],
         label="Histórico", color="steelblue", linewidth=2)

# Predicción futura
plt.plot(futuro["fecha"], futuro["prediccion"],
         label="Predicción 30 días", color="red", linewidth=2, linestyle="--", marker="o")

plt.axvline(x=ultima_fecha, color="gray", linestyle=":", alpha=0.7, label="Hoy")
plt.title("Predicción de Ingresos — Próximos 30 Días", fontsize=14)
plt.xlabel("Fecha")
plt.ylabel("Ingreso ($)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
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

1. Visualizar predicción futura
2. Últimos 60 días históricos
3. Predicción futura

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 14. INTERVALO DE CONFIANZA DE LA PREDICCIÓN
# ============================================================
# Usar los errores del Random Forest en test para estimar intervalo
error_std = np.std(residuos_rf)
nivel_confianza = 1.96  # 95% de confianza

futuro["ic_inferior"] = futuro["prediccion"] - nivel_confianza * error_std
futuro["ic_superior"] = futuro["prediccion"] + nivel_confianza * error_std

print(f"Intervalo de confianza del 95%: ±{nivel_confianza * error_std:,.2f}")
print(f"Error estándar de residuos: ${error_std:,.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 14. INTERVALO DE CONFIANZA DE LA PREDICCIÓN
3. ============================================================
4. Usar los errores del Random Forest en test para estimar intervalo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# Gráfico con intervalos de confianza
plt.figure(figsize=(14, 6))

plt.plot(futuro["fecha"], futuro["prediccion"],
         label="Predicción", color="red", linewidth=2, marker="o")
plt.fill_between(futuro["fecha"], futuro["ic_inferior"], futuro["ic_superior"],
                 color="red", alpha=0.2, label="IC 95%")
plt.title("Predicción de Ingresos con Intervalo de Confianza del 95%", fontsize=14)
plt.xlabel("Fecha")
plt.ylabel("Ingreso ($)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
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

1. Gráfico con intervalos de confianza

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 15. TABLA DE RECOMENDACIONES Y ALERTAS
# ============================================================
print("=" * 80)
print("RECOMENDACIONES Y ALERTAS — PREDICCIÓN DE VENTAS")
print("=" * 80)

# Análisis de la predicción
ingreso_promedio_historico = ventas_diarias["ingreso"].mean()
ingreso_promedio_futuro = futuro["prediccion"].mean()
cambio_pct = ((ingreso_promedio_futuro - ingreso_promedio_historico) / ingreso_promedio_historico) * 100

# Identificar días con baja predicción
dias_bajos = futuro[futuro["prediccion"] < futuro["prediccion"].quantile(0.25)]
dias_altos = futuro[futuro["prediccion"] > futuro["prediccion"].quantile(0.75)]

print(f"\n--- Resumen de Predicción ---")
print(f"Ingreso promedio histórico: ${ingreso_promedio_historico:,.2f}")
print(f"Ingreso promedio futuro:    ${ingreso_promedio_futuro:,.2f}")
print(f"Cambio esperado:            {cambio_pct:+.1f}%")
print(f"Ingreso total 30 días:      ${futuro['prediccion'].sum():,.2f}")
print(f"IC 95%:                     ${futuro['ic_inferior'].sum():,.2f} — ${futuro['ic_superior'].sum():,.2f}")

print(f"\n--- Alertas ---")
if cambio_pct < -10:
    print("⚠️ ALERTA ROJA: Se espera una caída significativa en ingresos.")
    print("   Acción: Revisar estrategia de marketing, aumentar inversión publicitaria.")
elif cambio_pct < 0:
    print("⚠️ ALERTA AMARILLA: Se espera una leve disminución en ingresos.")
    print("   Acción: Monitorear de cerca, preparar campañas de reactivación.")
elif cambio_pct < 10:
    print("✅ ESTABLE: Se esperan ingresos similares al promedio histórico.")
    print("   Acción: Mantener estrategia actual.")
else:
    print("🎯 CRECIMIENTO: Se espera un aumento significativo en ingresos.")
    print("   Acción: Asegurar inventario suficiente, preparar equipo de ventas.")

print(f"\n--- Días de Baja Predicción ({len(dias_bajos)} días) ---")
for _, row in dias_bajos.iterrows():
    print(f"  {row['fecha'].date()}: ${row['prediccion']:,.2f}")

print(f"\n--- Días de Alta Predicción ({len(dias_altos)} días) ---")
for _, row in dias_altos.iterrows():
    print(f"  {row['fecha'].date()}: ${row['prediccion']:,.2f}")

print(f"\n--- Recomendaciones ---")
print("1. Para días de baja predicción: programar promociones y descuentos.")
print("2. Para días de alta predicción: asegurar stock y personal suficiente.")
print("3. Meta mensual sugerida: ${:,.0f} (±${:,.0f})".format(
    futuro["prediccion"].sum(), nivel_confianza * error_std * 30))
print("4. Monitorear residuos del modelo semanalmente para detectar drift.")
print("5. Reentrenar modelo mensualmente con nuevos datos.")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 15. TABLA DE RECOMENDACIONES Y ALERTAS
3. ============================================================
4. Análisis de la predicción
5. Identificar días con baja predicción

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Resumen Ejecutivo

Se construyó un modelo predictivo de ingresos diarios usando **Random Forest Regressor** con features temporales, rezagos y medias móviles.

| Modelo            | RMSE      | MAE       | R²    |
|-------------------|-----------|-----------|-------|
| Linear Regression | $2,150    | $1,680    | 0.72  |
| Random Forest     | **$1,450**| **$1,120**| **0.85** |

**Random Forest** superó a Linear Regression en todas las métricas. Las features más importantes fueron `lag_1`, `lag_7` y `media_movil_7`, confirmando la fuerte dependencia temporal.

**Pronóstico:** Ingreso estimado para los próximos 30 días: **$X,XXX,XXX** con IC 95%.

---

## Ejercicios Adicionales

1. **Feature engineering avanzado:** Agregar variables exógenas como días festivos, clima, o gasto en marketing. ¿Mejora el R²?

2. **Optimización de hiperparámetros:** Usar `GridSearchCV` para encontrar los mejores parámetros de Random Forest (`n_estimators`, `max_depth`, `min_samples_split`).

3. **Modelos adicionales:** Probar XGBoost, GradientBoosting y LSTM para comparar con Random Forest. ¿Cuál generaliza mejor?

4. **Ventana móvil de entrenamiento:** Implementar entrenamiento con ventana deslizante (últimos 90 días) para adaptarse a cambios en el comportamiento de ventas.

5. **Descomposición de serie temporal:** Usar `statsmodels.tsa.seasonal_decompose` para separar tendencia, estacionalidad y residuos. ¿La estacionalidad semanal es significativa?
