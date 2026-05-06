# E09: Prophet para Pronóstico de Demanda y Ventas

## 1. Fundamentos Teóricos

### 1.1 ¿Qué es Prophet?

Prophet (Facebook/Meta) es un modelo aditivo para series temporales diseñado para pronóstico de negocio con:

- **Tendencia**: modelo de cambio de crecimiento (lineal o logístico con puntos de cambio)
- **Estacionalidad**: Fourier series para estacionalidades múltiples (semanal, anual, diaria)
- **Efectos especiales**: días festivos, regresores externos
- **Incertidumbre**: intervalos de confianza via simulación (MCMC opcional)

### 1.2 Componentes del modelo

y(t) = g(t) + s(t) + h(t) + ε_t

- g(t): tendencia (linear o logistic growth con changepoints)
- s(t): estacionalidades (Fourier series)
- h(t): efectos de días festivos
- ε_t: término de error

### 1.3 Parámetros clave

| Parámetro | Efecto |
|-----------|--------|
| `changepoint_prior_scale` | Flexibilidad de la tendencia (default=0.05) |
| `seasonality_prior_scale` | Flexibilidad de la estacionalidad (default=10) |
| `holidays_prior_scale` | Flexibilidad de días festivos (default=10) |
| `seasonality_mode` | 'additive' o 'multiplicative' |

---

## 2. Ejemplos Prácticos

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.serialize import model_to_dict, model_from_dict
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 5)

np.random.seed(42)

# Simular datos de ventas diarias con estacionalidad, tendencia y efectos de precio
n = 365 * 2  # 2 años
dates = pd.date_range('2023-01-01', periods=n, freq='D')
t = np.arange(n)

tendencia = 100 + 0.03 * t + 0.00002 * t**2
estacional_semanal = 15 * np.sin(2 * np.pi * t / 7)
estacional_anual = 25 * np.sin(2 * np.pi * t / 365.25 - np.pi/2)

# Efecto de precio (regresor externo)
precio = 50 + 10 * np.sin(2 * np.pi * t / 180) + np.random.normal(0, 2, n)
efecto_precio = -0.3 * (precio - 50)

# Outliers de días festivos (Navidad, Año Nuevo)
festivos_efecto = np.zeros(n)
fechas_navidad = pd.to_datetime(['2023-12-25', '2024-12-25'])
for f in fechas_navidad:
    idx = np.where(dates == f)[0]
    if len(idx) > 0:
        festivos_efecto[idx[0]] = 40

ruido = np.random.normal(0, 10, n)

ventas = tendencia + estacional_semanal + estacional_anual + efecto_precio + festivos_efecto + ruido
ventas = np.maximum(ventas, 10)  # mínimo 10

# DataFrame para Prophet: debe tener columnas 'ds' (date) y 'y' (target)
df_prophet = pd.DataFrame({
    'ds': dates,
    'y': ventas,
    'precio': precio
})

df_train = df_prophet.iloc[:n-60].copy()
df_test = df_prophet.iloc[n-60:].copy()

print(f"Train: {len(df_train)} días, Test: {len(df_test)} días")
print(f"Rango y: [{df_prophet['y'].min():.1f}, {df_prophet['y'].max():.1f}]")
print(f"Media y: {df_prophet['y'].mean():.1f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*2. Ejemplos Prácticos.*

1. Simular datos de ventas, compras o inventarios diarias con estacionalidad, tendencia y efectos de precio
2. Efecto de precio (regresor externo)
3. Outliers de días festivos (Navidad, Año Nuevo)
4. DataFrame para Prophet: debe tener columnas 'ds' (date) y 'y' (target)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 1: Prophet básico - preparación de datos

```python
print("=== Prophet: Preparación de Datos ===")
print(f"Columnas requeridas: 'ds' (datetime), 'y' (numeric)")
print(f"df_train shape: {df_train.shape}")
print(f"Rango fechas: {df_train['ds'].min().date()} → {df_train['ds'].max().date()}")
print(f"Mínimo y: {df_train['y'].min():.2f}")
print(f"Máximo y: {df_train['y'].max():.2f}")
print(f"NaN en y: {df_train['y'].isna().sum()}")

# Verificar que hay al menos 2 años de historia (Prophet requiere ~1 año para estacionalidad anual)
print(f"Años de historia: {(df_train['ds'].max() - df_train['ds'].min()).days / 365.25:.1f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: Prophet básico - preparación de datos.*

1. Verificar que hay al menos 2 años de historia (Prophet requiere ~1 año para estacionalidad anual)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: Entrenar Prophet con weekly_seasonality

```python
modelo = Prophet(
    growth='linear',
    seasonality_mode='additive',
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_prior_scale=10,
    changepoint_prior_scale=0.05,
    changepoint_range=0.8,
    interval_width=0.80,
    uncertainty_samples=1000
)

modelo.fit(df_train)

print("=== Prophet Entrenado ===")
print(f"Parámetros de tendencia: {len(modelo.params['delta'])} changepoints")
print(f"Changepoints:")
for cp in modelo.changepoints[:5]:
    print(f"  {cp.date()}")
if len(modelo.changepoints) > 5:
    print(f"  ... y {len(modelo.changepoints) - 5} más")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: Entrenar Prophet con weekly_seasonality.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: make_future_dataframe y predict

```python
# Crear dataframe futuro
future = modelo.make_future_dataframe(
    periods=30,
    freq='D',
    include_history=True
)

print("=== Future DataFrame ===")
print(f"Shape: {future.shape}")
print(f"Rango: {future['ds'].min().date()} → {future['ds'].max().date()}")
print(f"Incluye histórico: {future['ds'].min() == df_train['ds'].min()}")

# Predecir
forecast = modelo.predict(future)
print(f"\nForecast shape: {forecast.shape}")
print(f"Columnas: {forecast.columns.tolist()}")
print(f"\nPredicción últimos 5 días:")
for i in range(-5, 0):
    row = forecast.iloc[i]
    print(f"  {row['ds'].date()}: yhat={row['yhat']:.1f} "
          f"[{row['yhat_lower']:.1f}, {row['yhat_upper']:.1f}]")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: make_future_dataframe y predict.*

1. Crear dataframe futuro
2. Predecir

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: Plot forecast con intervalo de confianza

```python
fig = modelo.plot(forecast, figsize=(14, 6))
plt.title('Prophet Forecast con Intervalo de Confianza al 80%')
plt.xlabel('Fecha')
plt.ylabel('Ventas')
plt.savefig('e09_ej4_forecast.png', dpi=100)
plt.close()

print("=== Forecast Plots Generados ===")

# Calcular error en test
pred_test = forecast.iloc[-60:]['yhat'].values
real_test = df_test['y'].values
mape = np.mean(np.abs((real_test - pred_test) / real_test)) * 100
rmse = np.sqrt(mean_squared_error(real_test, pred_test))
print(f"MAPE en test: {mape:.2f}%")
print(f"RMSE en test: {rmse:.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: Plot forecast con intervalo de confianza.*

1. Calcular error en test

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: plot_components - tendencia + estacionalidades

```python
fig = modelo.plot_components(forecast, figsize=(14, 10))
plt.savefig('e09_ej5_components.png', dpi=100)
plt.close()

print("=== Componentes del Modelo ===")
print("1. Tendencia: muestra el crecimiento/decrecimiento de largo plazo")
print("2. Estacionalidad semanal: patrón intra-semana")
print("3. Estacionalidad anual: patrón intra-año")

# Extraer componentes
print(f"\nTendencia final: {forecast['trend'].iloc[-1]:.2f}")
print(f"Rango estacional semanal: [{forecast['weekly'].min():.2f}, {forecast['weekly'].max():.2f}]")
print(f"Rango estacional anual: [{forecast['yearly'].min():.2f}, {forecast['yearly'].max():.2f}]")

# Efecto semanal por día
weekly_effect = forecast[['ds', 'weekly']].iloc[-7:]
print("\nEfecto semanal promedio:")
for _, row in weekly_effect.iterrows():
    print(f"  {row['ds'].strftime('%A'):10s}: {row['weekly']:+.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: plot_components - tendencia + estacionalidades.*

1. Extraer componentes
2. Efecto semanal por día

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: Holidays - add_country_holidays

```python
modelo_holidays = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    holidays_prior_scale=10,
    seasonality_prior_scale=10,
    changepoint_prior_scale=0.05
)

# Agregar días festivos de México
modelo_holidays.add_country_holidays(country_name='MX')

modelo_holidays.fit(df_train)

future_h = modelo_holidays.make_future_dataframe(periods=30, freq='D')
forecast_h = modelo_holidays.predict(future_h)

print("=== Días Festivos (México) ===")
print(f"Días festivos incluidos: {len(modelo_holidays.train_holiday_names)}")
for h in sorted(modelo_holidays.train_holiday_names)[:10]:
    print(f"  - {h}")
print(f"  ... y {len(modelo_holidays.train_holiday_names) - 10} más")

# Ver efecto de día festivo en forecast
if 'holidays' in forecast_h.columns:
    print(f"\nEfecto de holidays en forecast:")
    holiday_effect = forecast_h[['ds', 'holidays']].dropna()
    holiday_effect = holiday_effect[holiday_effect['holidays'] != 0]
    for _, row in holiday_effect.iterrows():
        print(f"  {row['ds'].date()}: {row['holidays']:+.2f}")

fig_h = modelo_holidays.plot(forecast_h, figsize=(14, 6))
plt.title('Prophet con Días Festivos (México)')
plt.savefig('e09_ej6_holidays.png', dpi=100)
plt.close()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: Holidays - add_country_holidays.*

1. Agregar días festivos de México
2. Ver efecto de día festivo en forecast

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: Custom seasonality con Fourier order

```python
modelo_custom = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)

# Agregar estacionalidad mensual con Fourier order=10
modelo_custom.add_seasonality(
    name='monthly',
    period=30.5,
    fourier_order=10
)

modelo_custom.fit(df_train)
future_c = modelo_custom.make_future_dataframe(periods=30, freq='D')
forecast_c = modelo_custom.predict(future_c)

print("=== Custom Seasonality (Mensual, Fourier Order=10) ===")
print(f"Nombre: monthly")
print(f"Periodo: 30.5 días")
print(f"Fourier order: 10")

# Estadísticas del componente mensual
if 'monthly' in forecast_c.columns:
    print(f"Rango monthly: [{forecast_c['monthly'].min():.2f}, {forecast_c['monthly'].max():.2f}]")
    print(f"σ monthly: {forecast_c['monthly'].std():.2f}")
    print(f"σ yearly: {forecast_c['yearly'].std():.2f}")
    print(f"σ weekly: {forecast_c['weekly'].std():.2f}")

fig_c = modelo_custom.plot_components(forecast_c, figsize=(14, 12))
plt.savefig('e09_ej7_custom_season.png', dpi=100)
plt.close()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: Custom seasonality con Fourier order.*

1. Agregar estacionalidad mensual con Fourier order=10
2. Estadísticas del componente mensual

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: Regresor externo - precio

```python
modelo_reg = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True
)

# Agregar regresor externo: precio del producto
modelo_reg.add_regressor(
    name='precio',
    prior_scale=10,
    mode='additive'
)

# El dataframe de entrenamiento debe incluir el regresor
modelo_reg.fit(df_train[['ds', 'y', 'precio']])

# El future dataframe también debe incluir el regresor
future_r = modelo_reg.make_future_dataframe(periods=30, freq='D')

# Para test, usamos precios conocidos; para forecast necesitamos estimar precios futuros
future_r['precio'] = np.concatenate([
    df_train['precio'].values,
    df_test['precio'].values[:30] if len(df_test) >= 30 else df_test['precio'].values
])

forecast_r = modelo_reg.predict(future_r)

print("=== Prophet con Regresor Externo (Precio) ===")
print(f"Coeficiente del regresor 'precio':")
# El coeficiente se estima internamente
regressor_coef = forecast_r['extra_regressors_additive'].iloc[-1] / future_r['precio'].iloc[-1]
print(f"  Efecto estimado del precio: {modelo_reg.params['extra_regressors_additive']['precio'].mean():.4f}")

# Comparar con modelo sin precio
print(f"\nMAPE con precio: {np.mean(np.abs((df_test['y'].values[:30] - forecast_r['yhat'].iloc[-60:-30].values) / df_test['y'].values[:30])) * 100:.2f}%")

fig_r = modelo_reg.plot(forecast_r, figsize=(14, 6))
plt.title('Prophet con Regresor: Precio del Producto')
plt.savefig('e09_ej8_regressor.png', dpi=100)
plt.close()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: Regresor externo - precio.*

1. Agregar regresor externo: precio del producto
2. El dataframe de entrenamiento debe incluir el regresor
3. El future dataframe también debe incluir el regresor
4. Para test, usamos precios conocidos; para forecast necesitamos estimar precios futuros
5. El coeficiente se estima internamente
6. Comparar con modelo sin precio

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: Logistic growth con cap máximo

```python
# Simular datos con crecimiento limitado (saturación de mercado)
n_logistic = 365 * 2
t_log = np.arange(n_logistic)
cap = 10000  # capacidad máxima de mercado

# Crecimiento logístico: S-curve
ventas_logisticas = cap / (1 + np.exp(-0.005 * (t_log - 365)))
ventas_logisticas += 30 * np.sin(2 * np.pi * t_log / 7)
ventas_logisticas += 50 * np.sin(2 * np.pi * t_log / 365.25)
ventas_logisticas += np.random.normal(0, 50, n_logistic)
ventas_logisticas = np.clip(ventas_logisticas, 0, cap)

df_logistic = pd.DataFrame({
    'ds': pd.date_range('2023-01-01', periods=n_logistic, freq='D'),
    'y': ventas_logisticas,
    'cap': cap  # columna requerida para logistic growth
})

# Prophet con growth='logistic'
modelo_logistic = Prophet(
    growth='logistic',
    yearly_seasonality=True,
    weekly_seasonality=True
)
modelo_logistic.fit(df_logistic)

future_l = modelo_logistic.make_future_dataframe(periods=90, freq='D')
future_l['cap'] = cap  # futuro también necesita cap
forecast_l = modelo_logistic.predict(future_l)

print("=== Logistic Growth (Crecimiento Limitado) ===")
print(f"Capacidad máxima (cap): {cap}")
print(f"Forecast final: {forecast_l['yhat'].iloc[-1]:.0f} (cap={cap})")
print(f"Tasa de saturación: {forecast_l['yhat'].iloc[-1] / cap:.1%}")

fig_l = modelo_logistic.plot(forecast_l, figsize=(14, 6))
plt.title('Prophet Logistic Growth - Saturación de Mercado')
plt.axhline(y=cap, color='gray', linestyle='--', alpha=0.5, label=f'Cap={cap}')
plt.legend()
plt.savefig('e09_ej9_logistic.png', dpi=100)
plt.close()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: Logistic growth con cap máximo.*

1. Simular datos con crecimiento limitado (saturación de mercado)
2. Crecimiento logístico: S-curve
3. Prophet con growth='logistic'

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: changepoint_prior_scale = 0.5 (más cambios)

```python
modelo_cp = Prophet(
    changepoint_prior_scale=0.5,  # más flexible: 10x el default
    weekly_seasonality=True,
    yearly_seasonality=True
)
modelo_cp.fit(df_train)

future_cp = modelo_cp.make_future_dataframe(periods=30, freq='D')
forecast_cp = modelo_cp.predict(future_cp)

modelo_default = Prophet(weekly_seasonality=True, yearly_seasonality=True)
modelo_default.fit(df_train)
forecast_default = modelo_default.predict(future_cp)

print("=== Comparación de Changepoint Prior Scale ===")
print(f"Default (0.05): changepoints = {len(modelo_default.changepoints)}")
print(f"Alto (0.5): changepoints = {len(modelo_cp.changepoints)}")

# Calcular cuánto cambia la tendencia
delta_default = np.std(modelo_default.params['delta'][0])
delta_cp = np.std(modelo_cp.params['delta'][0])
print(f"Δ tendencia default: {delta_default:.4f}")
print(f"Δ tendencia alto: {delta_cp:.4f}")

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(forecast_default['ds'], forecast_default['trend'], label='Default (0.05)', linewidth=2)
ax.plot(forecast_cp['ds'], forecast_cp['trend'], label='Alto (0.5)', linewidth=2, alpha=0.7)
for cp in modelo_cp.changepoints:
    ax.axvline(x=cp, color='red', alpha=0.2)
ax.set_title('Efecto de changepoint_prior_scale en la Tendencia')
ax.legend()
plt.tight_layout()
plt.savefig('e09_ej10_changepoint.png', dpi=100)
plt.close()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: changepoint_prior_scale = 0.5 (más cambios).*

1. Calcular cuánto cambia la tendencia

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: seasonality_prior_scale = 10 (más estacionalidad)

```python
modelo_ss = Prophet(
    seasonality_prior_scale=10,
    weekly_seasonality=True
)
modelo_ss.fit(df_train)
forecast_ss = modelo_ss.predict(future_cp)

modelo_ss_bajo = Prophet(
    seasonality_prior_scale=0.1,  # estacionalidad muy rígida
    weekly_seasonality=True
)
modelo_ss_bajo.fit(df_train)
forecast_ss_bajo = modelo_ss_bajo.predict(future_cp)

print("=== Comparación de Seasonality Prior Scale ===")
print(f"Bajo (0.1): weekly σ = {forecast_ss_bajo['weekly'].std():.2f}")
print(f"Default (10): weekly σ = {forecast_ss['weekly'].std():.2f}")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(forecast_ss_bajo['ds'].iloc[-60:], forecast_ss_bajo['weekly'].iloc[-60:], 
        label='seasonality_prior=0.1', linewidth=2)
ax.plot(forecast_ss['ds'].iloc[-60:], forecast_ss['weekly'].iloc[-60:], 
        label='seasonality_prior=10', linewidth=2)
ax.set_title('Efecto de seasonality_prior_scale en Estacionalidad Semanal')
ax.legend()
plt.tight_layout()
plt.savefig('e09_ej11_seasonality_prior.png', dpi=100)
plt.close()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: seasonality_prior_scale = 10 (más estacionalidad).*

1. `modelo_ss.fit(df_train)` — Entrena el modelo con los datos de entrenamiento.
2. `forecast_ss = modelo_ss.predict(future_cp)` — Genera predicciones sobre nuevos datos.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: Modo multiplicativo

```python
# Datos donde la estacionalidad crece con la tendencia
ventas_mult = 100 * (1 + 0.001 * t)  # tendencia creciente
ventas_mult *= 1 + 0.2 * np.sin(2 * np.pi * t / 7)  # estacionalidad proporcional
ventas_mult += np.random.normal(0, 5, n)

df_mult = pd.DataFrame({
    'ds': dates,
    'y': ventas_mult
})

modelo_mult = Prophet(
    seasonality_mode='multiplicative',  # la estacionalidad escala con la tendencia
    yearly_seasonality=True,
    weekly_seasonality=True
)
modelo_mult.fit(df_mult)

future_mult = modelo_mult.make_future_dataframe(periods=30, freq='D')
forecast_mult = modelo_mult.predict(future_mult)

print("=== Modo Multiplicativo ===")
print(f"Estacionalidad semanal es proporcional a la tendencia")
print(f"Tendencia final: {forecast_mult['trend'].iloc[-1]:.1f}")
print(f"Estacionalidad semanal (range): [{forecast_mult['weekly'].min():.4f}, {forecast_mult['weekly'].max():.4f}]")
print(f"→ En modo multiplicativo, weekly≈1 ± efecto")

fig_mult = modelo_mult.plot_components(forecast_mult, figsize=(14, 10))
plt.savefig('e09_ej12_multiplicativo.png', dpi=100)
plt.close()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: Modo multiplicativo.*

1. Datos donde la estacionalidad crece con la tendencia

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: Cross-validation

```python
print("=== Cross-Validation con Prophet ===")

df_cv = cross_validation(
    modelo,
    horizon='30 days',
    period='15 days',
    initial='365 days',
    parallel='processes'
)

print(f"Cross-validation shape: {df_cv.shape}")
print(f"Horizontes: {df_cv['horizon'].unique()}")
print(f"\nPrimeras filas:")
print(df_cv.head())

# Métricas por horizonte
df_metrics = performance_metrics(df_cv, rolling_window=0.1)
print("\nPerformance Metrics:")
print(df_metrics[['horizon', 'mape', 'rmse', 'mae']].head(10))
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

*Ejemplo 13: Cross-validation.*

1. Métricas por horizonte

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: Performance metrics detalladas

```python
print("=== Performance Metrics Detalladas ===")

df_pm = performance_metrics(df_cv)

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# MAPE por horizonte
axes[0, 0].plot(df_pm['horizon'].dt.days, df_pm['mape'], 'b-o')
axes[0, 0].set_xlabel('Horizonte (días)')
axes[0, 0].set_ylabel('MAPE (%)')
axes[0, 0].set_title('MAPE por Horizonte')
axes[0, 0].grid(True)

# RMSE por horizonte
axes[0, 1].plot(df_pm['horizon'].dt.days, df_pm['rmse'], 'r-o')
axes[0, 1].set_xlabel('Horizonte (días)')
axes[0, 1].set_ylabel('RMSE')
axes[0, 1].set_title('RMSE por Horizonte')
axes[0, 1].grid(True)

# MAE por horizonte
axes[1, 0].plot(df_pm['horizon'].dt.days, df_pm['mae'], 'g-o')
axes[1, 0].set_xlabel('Horizonte (días)')
axes[1, 0].set_ylabel('MAE')
axes[1, 0].set_title('MAE por Horizonte')
axes[1, 0].grid(True)

# Cobertura
axes[1, 1].plot(df_pm['horizon'].dt.days, df_pm['coverage'], 'purple-o')
axes[1, 1].axhline(y=0.80, color='gray', linestyle='--', label='Target 80%')
axes[1, 1].set_xlabel('Horizonte (días)')
axes[1, 1].set_ylabel('Cobertura')
axes[1, 1].set_title('Cobertura del IC')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('e09_ej14_performance.png', dpi=100)
plt.close()

print("Métricas promedio:")
print(f"  MAPE: {df_pm['mape'].mean():.2f}%")
print(f"  RMSE: {df_pm['rmse'].mean():.2f}")
print(f"  MAE: {df_pm['mae'].mean():.2f}")
print(f"  Cobertura IC: {df_pm['coverage'].mean():.2%}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: Performance metrics detalladas.*

1. MAPE por horizonte
2. RMSE por horizonte
3. MAE por horizonte
4. Cobertura

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: Grid search de parámetros

```python
print("=== Grid Search: Optimización de Parámetros Prophet ===")

param_grid = {
    'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1, 0.5],
    'seasonality_prior_scale': [0.1, 1, 5, 10, 20],
    'holidays_prior_scale': [0.1, 1, 5, 10, 20]
}

# Generar todas las combinaciones
from itertools import product
keys = list(param_grid.keys())
combinations = list(product(*param_grid.values()))

print(f"Total de combinaciones: {len(combinations)}")
print(f"Evaluando (limitado a 10 para velocidad)...")

results_grid = []
for i, combo in enumerate(combinations[:10]):
    params = dict(zip(keys, combo))
    
    try:
        m = Prophet(
            weekly_seasonality=True,
            yearly_seasonality=True,
            **params
        )
        m.fit(df_train)
        
        df_cv_tmp = cross_validation(
            m, horizon='30 days', period='30 days', initial='365 days',
            parallel='processes'
        )
        mape = performance_metrics(df_cv_tmp)['mape'].mean()
        
        results_grid.append({**params, 'MAPE': mape})
        print(f"  [{i+1}/10] cp={combo[0]:.3f}, sp={combo[1]:.1f}, hp={combo[2]:.1f} → MAPE={mape:.2f}%")
    except Exception as e:
        print(f"  [{i+1}/10] combo {combo} → ERROR: {e}")

if results_grid:
    best = min(results_grid, key=lambda x: x['MAPE'])
    print(f"\n✓ Mejor combinación: {best}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: Grid search de parámetros.*

1. Generar todas las combinaciones

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 16: Comparar Prophet vs SARIMA

```python
print("=== Comparación Prophet vs SARIMA ===")

# Prophet
m_prophet = Prophet(weekly_seasonality=True, yearly_seasonality=True)
m_prophet.fit(df_train)
future_p = m_prophet.make_future_dataframe(periods=60, freq='D')
fc_prophet = m_prophet.predict(future_p)
pred_prophet = fc_prophet['yhat'].iloc[-60:].values

# SARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
m_sarima = SARIMAX(
    df_train['y'],
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7)
)
res_sarima = m_sarima.fit(disp=False)
pred_sarima = res_sarima.forecast(steps=60).values

# Comparar
real = df_test['y'].values
print(f"{'Modelo':<20} {'MAPE':<10} {'RMSE':<10} {'MAE':<10}")
print("-" * 50)

for name, pred in [('Prophet', pred_prophet), ('SARIMA', pred_sarima)]:
    mape = np.mean(np.abs((real - pred) / real)) * 100
    rmse = np.sqrt(np.mean((real - pred)**2))
    mae = np.mean(np.abs(real - pred))
    print(f"{name:<20} {mape:<10.2f} {rmse:<10.2f} {mae:<10.2f}")

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df_test['ds'], real, 'k-', label='Real', linewidth=2)
ax.plot(df_test['ds'], pred_prophet, 'b--', label='Prophet')
ax.plot(df_test['ds'], pred_sarima, 'r--', label='SARIMA')
ax.set_title('Comparación: Prophet vs SARIMA en Test Set')
ax.legend()
plt.tight_layout()
plt.savefig('e09_ej16_comparison.png', dpi=100)
plt.close()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: Comparar Prophet vs SARIMA.*

1. Prophet
2. SARIMA
3. Comparar

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 17: MCMC para incertidumbre bayesiana

```python
modelo_mcmc = Prophet(
    weekly_seasonality=True,
    yearly_seasonality=True,
    mcmc_samples=100,  # MCMC fully Bayesian
    interval_width=0.95
)

print("=== Prophet con MCMC (Bayesiano Completo) ===")
print("Entrenando con MCMC... (esto toma ~1-2 min)")
modelo_mcmc.fit(df_train)

future_mcmc = modelo_mcmc.make_future_dataframe(periods=30, freq='D')
forecast_mcmc = modelo_mcmc.predict(future_mcmc)

print("Forecast con MCMC (últimos 5):")
for i in range(-5, 0):
    row = forecast_mcmc.iloc[i]
    print(f"  {row['ds'].date()}: {row['yhat']:.1f} [{row['yhat_lower']:.1f}, {row['yhat_upper']:.1f}]")

# Comparar ancho del IC con y sin MCMC
print(f"\nAncho IC promedio (MCMC): {(forecast_mcmc['yhat_upper'] - forecast_mcmc['yhat_lower']).mean():.2f}")
print(f"Ancho IC promedio (no MCMC): {(forecast['yhat_upper'] - forecast['yhat_lower']).mean():.2f}")

fig_mcmc = modelo_mcmc.plot(forecast_mcmc, figsize=(14, 6))
plt.title('Prophet con MCMC - Incertidumbre Bayesiana Completa')
plt.savefig('e09_ej17_mcmc.png', dpi=100)
plt.close()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: MCMC para incertidumbre bayesiana.*

1. Comparar ancho del IC con y sin MCMC

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 18: Integrador - Prophet para forecast de demanda con holidays y precio

```python
print("=" * 70)
print("PROPHET INTEGRAL: DEMANDA CON HOLIDAYS Y PRECIO")
print("=" * 70)

print("\n[1] CONFIGURACIÓN DEL MODELO")
modelo_integral = Prophet(
    growth='linear',
    seasonality_mode='additive',
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_prior_scale=10,
    changepoint_prior_scale=0.05,
    holidays_prior_scale=10,
    interval_width=0.90,
    uncertainty_samples=1000
)

# Agregar component extras
modelo_integral.add_country_holidays(country_name='MX')
modelo_integral.add_seasonality(name='monthly', period=30.5, fourier_order=5)
modelo_integral.add_regressor(name='precio', prior_scale=10, mode='additive')

print("Componentes del modelo:")
print(f"  - Tendencia: growth='linear'")
print(f"  - Estacionalidad: yearly, weekly, monthly (Fourier order=5)")
print(f"  - Holidays: México")
print(f"  - Regresor: precio")

print("\n[2] ENTRENAMIENTO")
modelo_integral.fit(df_train[['ds', 'y', 'precio']])
print(f"  Entrenamiento completado")

print("\n[3] PREDICCIÓN")
future_int = modelo_integral.make_future_dataframe(periods=30, freq='D')
future_int['precio'] = np.concatenate([
    df_train['precio'].values,
    df_test['precio'].values[:30]
])
forecast_int = modelo_integral.predict(future_int)

pred_test = forecast_int['yhat'].iloc[-60:].values
real_test = df_test['y'].values
mape_int = np.mean(np.abs((real_test - pred_test) / real_test)) * 100
rmse_int = np.sqrt(np.mean((real_test - pred_test)**2))

print(f"\n[4] MÉTRICAS EN TEST")
print(f"  MAPE: {mape_int:.2f}%")
print(f"  RMSE: {rmse_int:.2f}")
print(f"  MAE: {np.mean(np.abs(real_test - pred_test)):.2f}")

print(f"\n[5] FORECAST 30 DÍAS")
for i in range(-30, 0):
    row = forecast_int.iloc[i]
    print(f"  {row['ds'].date()}: {row['yhat']:.1f} [{row['yhat_lower']:.1f}, {row['yhat_upper']:.1f}]")

print("\n[6] ANÁLISIS DE COMPONENTES")
print(f"  Contribución weekly: σ={forecast_int['weekly'].std():.2f}")
print(f"  Contribución yearly: σ={forecast_int['yearly'].std():.2f}")
print(f"  Contribución monthly: σ={forecast_int['monthly'].std():.2f}")
print(f"  Contribución holidays: σ={forecast_int['holidays'].std():.2f}")
print(f"  Contribución precio: σ={forecast_int['extra_regressors_additive'].std():.2f}")

# Gráfico final
fig_final = modelo_integral.plot(forecast_int, figsize=(14, 6))
plt.title('Prophet Integral: Demanda con Holidays y Precio')
plt.savefig('e09_ej18_integrador.png', dpi=100, bbox_inches='tight')
plt.close()

fig_comp = modelo_integral.plot_components(forecast_int, figsize=(14, 14))
plt.savefig('e09_ej18_components_full.png', dpi=100, bbox_inches='tight')
plt.close()
print("\n✓ Pipeline integral completado")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador - Prophet para forecast de demanda con holidays y precio.*

1. Agregar component extras
2. Gráfico final

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 3. Ejercicios

1. **Sobreacte con changepoint alto**: Configura `changepoint_prior_scale=10` y entrena. ¿Qué observas en la tendencia? ¿Cómo afecta el pronóstico?

2. **Seasonality mode equivocado**: Aplica `seasonality_mode='multiplicative'` a la serie aditiva del Ejemplo 1. ¿Qué problemas ves en el pronóstico?

3. **Holidays personalizados**: Crea un dataframe de holidays con "Black Friday" y "Cyber Monday". Compara el forecast con y sin estos holidays personalizados.

4. **Impacto del precio**: Modifica `add_regressor(prior_scale=0.01)` (muy rígido) y compara con `prior_scale=100`. ¿Cómo cambia el peso del precio?

5. **Validación expandida**: Usa cross-validation con initial='180 days', period='7 days', horizon='14 days'. Interpreta cómo cambian MAPE y RMSE con el horizonte.

6. **Flat growth**: Configura `growth='flat'` y compara tendencia vs `growth='linear'`. ¿Qué pasa si la serie tiene tendencia real?

7. **Forecast sin historia suficiente**: Usa solo 90 días de entrenamiento con Prophet. ¿Qué calidad tiene el forecast anual? ¿Qué estacionalidades puedes incluir?

8. **Pipeline de producción**: Crea una función que reciba un DataFrame con 'ds', 'y' y regresores opcionales, ejecute grid search limitado (solo changepoint_prior_scale y seasonality_prior_scale), entrene el mejor modelo, validé con cross-validation y retorne forecast + métricas.

---

## 4. Resumen

| Concepto | Aplicación práctica |
|----------|---------------------|
| **Prophet básico** | Forecast rápido con pocos parámetros |
| **Estacionalidades** | weekly, yearly automáticas; monthly custom con Fourier |
| **Holidays** | Crucial para retail y demanda estacional (Navidad, etc.) |
| **Regresores externos** | Precio, promociones, clima como variables adicionales |
| **Logistic growth** | Cuando hay saturación de mercado o capacidad máxima |
| **Changepoint prior** | Controla cuánto puede cambiar la tendencia |
| **Cross-validation** | Evaluación robusta con ventanas temporales deslizantes |
| **MCMC** | Incertidumbre bayesiana completa (más realista) |
| **Interpretabilidad** | Componentes separados: tendencia, estacionalidad, holidays |
