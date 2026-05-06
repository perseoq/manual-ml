# E08: ARIMA y SARIMA para Pronóstico de Demanda

## 1. Fundamentos Teóricos

### 1.1 Conceptos clave

**ARIMA(p,d,q)**: Modelo autorregresivo integrado de media móvil.
- **AR(p)**: Y_t = c + φ₁Y_{t-1} + ... + φ_pY_{t-p} + ε_t
- **I(d)**: Diferenciación para hacer la serie estacionaria (d veces)
- **MA(q)**: Y_t = c + ε_t + θ₁ε_{t-1} + ... + θ_qε_{t-q}

**SARIMA(p,d,q)(P,D,Q,s)**: Versión estacional.
- s = periodo estacional (7 para semanal, 12 para mensual)
- (P,D,Q) = parte estacional del modelo

### 1.2 Identificación de orden

| Herramienta | Qué muestra |
|-------------|-------------|
| **ACF** | Cortes después de lag q (MA) o decaimiento (AR) |
| **PACF** | Cortes después de lag p (AR) o decaimiento (MA) |
| **auto_arima** | Búsqueda automática del mejor (p,d,q)(P,D,Q,s) |

### 1.3 Criterios de información

- **AIC**: Akaike Information Criterion (-2logL + 2k)
- **AICc**: AIC con corrección por muestra pequeña
- **BIC**: Bayesian Information Criterion (-2logL + k·log(n))

### 1.4 Diagnóstico

- **Ljung-Box**: H₀ = residuos son ruido blanco (p > 0.05 → buen modelo)
- **Normalidad**: QQ plot + Jarque-Bera test
- **ACF residuos**: no deben mostrar correlación significativa

---

## 2. Ejemplos Prácticos

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose
import pmdarima as pm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 5)

np.random.seed(42)

# Simular datos de ventas con estacionalidad semanal y tendencia
n = 365 * 2  # 2 años
dates = pd.date_range('2023-01-01', periods=n, freq='D')
t = np.arange(n)

tendencia = 100 + 0.05 * t
estacional = 20 * np.sin(2 * np.pi * t / 7) + 15 * np.sin(2 * np.pi * t / 365.25)
ruido = np.random.normal(0, 8, n)

ventas = tendencia + estacional + ruido
ventas = ventas + 50  # offset positivo

df = pd.DataFrame({'ventas': ventas}, index=dates)
df_train = df.iloc[:n-60].copy()
df_test = df.iloc[n-60:].copy()

print(f"Train: {len(df_train)} días, Test: {len(df_test)} días")
print(f"Media train: {df_train['ventas'].mean():.1f}, media test: {df_test['ventas'].mean():.1f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*2. Ejemplos Prácticos.*

1. Simular datos de ventas, compras o inventarios con estacionalidad semanal y tendencia

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 1: Diferenciación simple para estacionariedad

```python
# Test de estacionariedad antes de diferenciar
result_adf = adfuller(df_train['ventas'])
print("=== Diferenciación Simple ===")
print(f"ADF test (original): estadístico={result_adf[0]:.4f}, p-valor={result_adf[1]:.4f}")

# Diferenciación simple (d=1)
df_train['ventas_diff1'] = df_train['ventas'].diff().dropna()

result_adf_diff = adfuller(df_train['ventas_diff1'].dropna())
print(f"ADF test (d=1): estadístico={result_adf_diff[0]:.4f}, p-valor={result_adf_diff[1]:.4f}")

# KPSS test (H0: estacionaria)
result_kpss = kpss(df_train['ventas'].dropna(), regression='ct')
print(f"KPSS test (original): estadístico={result_kpss[0]:.4f}, p-valor={result_kpss[1]:.4f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
df_train['ventas'].plot(ax=axes[0, 0], title='Serie Original')
df_train['ventas_diff1'].plot(ax=axes[0, 1], title=f'Diferenciada (d=1)')
df_train['ventas'].hist(ax=axes[1, 0], bins=40)
df_train['ventas_diff1'].hist(ax=axes[1, 1], bins=40)
plt.tight_layout()
plt.savefig('e08_ej1_diferenciacion.png', dpi=100)
plt.close()

print(f"¿Estacionaria tras d=1? {'Sí' if result_adf_diff[1] < 0.05 else 'No'}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: Diferenciación simple para estacionariedad.*

1. Test de estacionariedad antes de diferenciar
2. Diferenciación simple (d=1)
3. KPSS test (H0: estacionaria)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: Diferenciación estacional (D=1, s=7)

```python
# Diferenciación estacional: restar el valor de hace 7 días
df_train['ventas_diff_s7'] = df_train['ventas'].diff(periods=7).dropna()

# Doble diferenciación: estacional + regular
df_train['ventas_diff_s7_d1'] = df_train['ventas'].diff(periods=7).diff().dropna()

print("=== Diferenciación Estacional (s=7) ===")
print(f"Original std: {df_train['ventas'].std():.2f}")
print(f"Tras diff(7) std: {df_train['ventas_diff_s7'].std():.2f}")
print(f"Tras diff(7)+diff(1) std: {df_train['ventas_diff_s7_d1'].std():.2f}")

for label, col in [('d=1', 'ventas_diff1'), ('D=1,s=7', 'ventas_diff_s7'), 
                    ('D=1,s=7 + d=1', 'ventas_diff_s7_d1')]:
    adf = adfuller(df_train[col].dropna())
    print(f"ADF ({label}): p={adf[1]:.6f} → {'Estacionaria' if adf[1] < 0.05 else 'No estacionaria'}")

fig, axes = plt.subplots(3, 1, figsize=(14, 8))
df_train['ventas'].iloc[:100].plot(ax=axes[0], title='Original')
df_train['ventas_diff_s7'].iloc[:100].plot(ax=axes[1], title='Dif. Estacional (s=7)')
df_train['ventas_diff_s7_d1'].iloc[:100].plot(ax=axes[2], title='Dif. Estacional + Simple')
plt.tight_layout()
plt.savefig('e08_ej2_diff_estacional.png', dpi=100)
plt.close()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: Diferenciación estacional (D=1, s=7).*

1. Diferenciación estacional: restar el valor de hace 7 días
2. Doble diferenciación: estacional + regular

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: ACF para identificar orden MA

```python
# ACF de la serie diferenciada
serie_estacionaria = df_train['ventas_diff_s7_d1'].dropna()

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(serie_estacionaria, lags=40, ax=axes[0])
axes[0].set_title('ACF - Serie Diferenciada')
plot_pacf(serie_estacionaria, lags=40, ax=axes[1])
axes[1].set_title('PACF - Serie Diferenciada')
plt.tight_layout()
plt.savefig('e08_ej3_acf_pacf.png', dpi=100)
plt.close()

acf_vals = acf(serie_estacionaria, nlags=20)
pacf_vals = pacf(serie_estacionaria, nlags=20)

print("=== Identificación de Orden via ACF/PACF ===")
print("ACF (primeros 10 lags):")
for i in range(1, 11):
    sig = '*' if abs(acf_vals[i]) > 1.96/np.sqrt(len(serie_estacionaria)) else ''
    print(f"  lag {i:2d}: {acf_vals[i]:+.4f}{sig}")
print(f"\n→ Si ACF corta después de lag q → MA(q)")
print(f"→ Si PACF corta después de lag p → AR(p)")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: ACF para identificar orden MA.*

1. ACF de la serie diferenciada

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: PACF para identificar orden AR

```python
print("=== Identificación de Orden AR via PACF ===")
print("PACF (primeros 10 lags):")
for i in range(1, 11):
    sig = '*' if abs(pacf_vals[i]) > 1.96/np.sqrt(len(serie_estacionaria)) else ''
    print(f"  lag {i:2d}: {pacf_vals[i]:+.4f}{sig}")

# Sugerencia automática de orden
sugerencia_p = sum(1 for i in range(1, 11) if abs(pacf_vals[i]) > 1.96/np.sqrt(len(serie_estacionaria)))
sugerencia_q = sum(1 for i in range(1, 11) if abs(acf_vals[i]) > 1.96/np.sqrt(len(serie_estacionaria)))
print(f"\n→ Sugerencia: p={sugerencia_p}, q={sugerencia_q}")
print(f"  (basado en número de lags significativos)")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: PACF para identificar orden AR.*

1. Sugerencia automática de orden

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: ARIMA(1,1,1) manual

```python
# Asegurar que la serie no tenga NaN
y_train = df_train['ventas'].dropna()

modelo_arima111 = ARIMA(
    y_train,
    order=(1, 1, 1),  # p=1, d=1, q=1
    enforce_stationarity=True,
    enforce_invertibility=True
)
resultado_arima111 = modelo_arima111.fit()

print("=== ARIMA(1,1,1) ===")
print(resultado_arima111.summary())

# Coeficientes
print(f"\nφ₁ (AR): {resultado_arima111.params['ar.L1']:.4f}")
print(f"θ₁ (MA): {resultado_arima111.params['ma.L1']:.4f}")
print(f"σ²: {resultado_arima111.params['sigma2']:.4f}")
print(f"AIC: {resultado_arima111.aic:.2f}")
print(f"BIC: {resultado_arima111.bic:.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: ARIMA(1,1,1) manual.*

1. Asegurar que la serie no tenga NaN
2. Coeficientes

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: ARIMA(2,1,2) con más parámetros

```python
modelo_arima212 = ARIMA(
    y_train,
    order=(2, 1, 2),
    enforce_stationarity=True,
    enforce_invertibility=True
)
resultado_arima212 = modelo_arima212.fit()

print("=== ARIMA(2,1,2) ===")
print(f"AIC: {resultado_arima212.aic:.2f} vs ARIMA(1,1,1) AIC: {resultado_arima111.aic:.2f}")
print(f"BIC: {resultado_arima212.bic:.2f} vs ARIMA(1,1,1) BIC: {resultado_arima111.bic:.2f}")

# Comparación de parámetros
for i in range(1, 3):
    print(f"AR.L{i}: {resultado_arima212.params[f'ar.L{i}']:.4f}")
    print(f"MA.L{i}: {resultado_arima212.params[f'ma.L{i}']:.4f}")

# Verificar estacionariedad e invertibilidad
print(f"\nRaíces AR: {resultado_arima212.arroots}")
print(f"Raíces MA: {resultado_arima212.maroots}")
print(f"¿Estacionario? {all(abs(r) > 1 for r in resultado_arima212.arroots)}")
print(f"¿Invertible? {all(abs(r) > 1 for r in resultado_arima212.maroots)}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: ARIMA(2,1,2) con más parámetros.*

1. Comparación de parámetros
2. Verificar estacionariedad e invertibilidad

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: SARIMA(1,1,1)(1,1,1,7) con estacionalidad semanal

```python
modelo_sarima = SARIMAX(
    y_train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False
)
resultado_sarima = modelo_sarima.fit(disp=False)

print("=== SARIMA(1,1,1)(1,1,1,7) ===")
print(resultado_sarima.summary())

print(f"\nAIC: {resultado_sarima.aic:.2f}")
print(f"BIC: {resultado_sarima.bic:.2f}")

# Componentes
print("\nParámetros AR estacional:")
print(f"  AR.S.L7: {resultado_sarima.params.get('ar.S.L7', 'N/A')}")
print(f"  MA.S.L7: {resultado_sarima.params.get('ma.S.L7', 'N/A')}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: SARIMA(1,1,1)(1,1,1,7) con estacionalidad semanal.*

1. Componentes

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: auto_arima - búsqueda automática

```python
print("=== auto_arima: Búsqueda del Mejor Modelo ===")
print("Buscando... (esto puede tomar unos segundos)")

auto_model = pm.auto_arima(
    y_train,
    start_p=0, start_q=0,
    max_p=5, max_q=5,
    max_P=2, max_Q=2,
    max_d=2, max_D=1,
    seasonal=True, m=7,
    start_P=0, start_Q=0,
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True,
    information_criterion='aic',
    alpha=0.05,
    n_fits=10,
    seasonal_test=True,
    with_intercept=True,
    random_state=42,
    n_jobs=1
)

print("\n=== MEJOR MODELO ENCONTRADO ===")
print(auto_model.summary())
print(f"\nOrden: {auto_model.order}")
print(f"Orden estacional: {auto_model.seasonal_order}")
print(f"AIC: {auto_model.aic():.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: auto_arima - búsqueda automática.*

1. `print("=== auto_arima: Búsqueda del Mejor Modelo ===")` — Muestra el resultado por pantalla.
2. `print("Buscando... (esto puede tomar unos segundos)")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: auto_arima con trace=True detallado

```python
print("=== auto_arima con trace Completo ===")
auto_trace = pm.auto_arima(
    y_train,
    start_p=0, start_q=0,
    max_p=3, max_q=3,
    max_P=1, max_Q=1,
    seasonal=True, m=7,
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True,
    information_criterion='aic',
    n_fits=5,
    random_state=42
)

print(f"\nModelo final: ARIMA{auto_trace.order}x{auto_trace.seasonal_order}")
print(f"AIC: {auto_trace.aic():.2f}")
print(f"BIC: {auto_trace.bic():.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: auto_arima con trace=True detallado.*

1. `print("=== auto_arima con trace Completo ===")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: auto_arima con information_criterion='bic'

```python
print("=== auto_arima con BIC (modelos más parsimoniosos) ===")
auto_bic = pm.auto_arima(
    y_train,
    start_p=0, start_q=0,
    max_p=5, max_q=5,
    max_P=2, max_Q=2,
    seasonal=True, m=7,
    trace=False,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True,
    information_criterion='bic',  # BIC penaliza más parámetros
    random_state=42
)

auto_aic = pm.auto_arima(
    y_train,
    start_p=0, start_q=0,
    max_p=5, max_q=5,
    max_P=2, max_Q=2,
    seasonal=True, m=7,
    trace=False,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True,
    information_criterion='aic',
    random_state=42
)

print("Comparación AIC vs BIC:")
print(f"  BIC: ARIMA{auto_bic.order}x{auto_bic.seasonal_order} (AIC={auto_bic.aic():.1f}, BIC={auto_bic.bic():.1f})")
print(f"  AIC: ARIMA{auto_aic.order}x{auto_aic.seasonal_order} (AIC={auto_aic.aic():.1f}, BIC={auto_aic.bic():.1f})")
print(f"\n  → El modelo BIC tiene {auto_bic.order[0] + auto_bic.order[2] + auto_bic.seasonal_order[0] + auto_bic.seasonal_order[2]} parámetros AR/MA")
print(f"  → El modelo AIC tiene {auto_aic.order[0] + auto_aic.order[2] + auto_aic.seasonal_order[0] + auto_aic.seasonal_order[2]} parámetros AR/MA")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: auto_arima con information_criterion='bic'.*

1. `print("=== auto_arima con BIC (modelos más parsimoniosos) ===")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: Diagnóstico de residuos - Ljung-Box

```python
residuos_sarima = resultado_sarima.resid.dropna()

print("=== Diagnóstico de Residuos: Ljung-Box Test ===")
print("H₀: Los residuos son ruido blanco (independientes)")
print(f"n = {len(residuos_sarima)}")

for lag in [7, 14, 21, 30]:
    lb = acorr_ljungbox(residuos_sarima, lags=[lag], return_df=True)
    stat = lb.loc[lag, 'lb_stat']
    pval = lb.loc[lag, 'lb_pvalue']
    conclusion = "✓ Ruido blanco" if pval > 0.05 else "✗ Hay autocorrelación"
    print(f"  Lag {lag:2d}: estadístico={stat:.4f}, p-valor={pval:.4f} → {conclusion}")

# Ljung-Box para ARIMA(1,1,1)
residuos_arima111 = resultado_arima111.resid.dropna()
lb_arima111 = acorr_ljungbox(residuos_arima111, lags=[7, 14], return_df=True)
print(f"\nARIMA(1,1,1) Lag 7 p-valor: {lb_arima111.loc[7, 'lb_pvalue']:.4f}")
print(f"SARIMA    Lag 7 p-valor: {lb.loc[7, 'lb_pvalue']:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: Diagnóstico de residuos - Ljung-Box.*

1. Ljung-Box para ARIMA(1,1,1)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: QQ plot - normalidad de residuos

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Histograma
axes[0].hist(residuos_sarima, bins=40, density=True, alpha=0.7, color='steelblue')
x = np.linspace(residuos_sarima.min(), residuos_sarima.max(), 100)
axes[0].plot(x, stats.norm.pdf(x, residuos_sarima.mean(), residuos_sarima.std()), 
             'r-', linewidth=2, label='Normal')
axes[0].set_title('Histograma de Residuos')

# QQ plot
stats.probplot(residuos_sarima, dist='norm', plot=axes[1])
axes[1].set_title('Q-Q Plot')

# Boxplot
axes[2].boxplot(residuos_sarima, vert=False)
axes[2].set_title('Boxplot de Residuos')
axes[2].set_yticks([])

plt.tight_layout()
plt.savefig('e08_ej12_qqplot.png', dpi=100)
plt.close()

# Test de normalidad
stat_norm, p_norm = stats.normaltest(residuos_sarima)
print("=== Normalidad de Residuos ===")
print(f"Jarque-Bera/D'Agostino: stat={stat_norm:.4f}, p-valor={p_norm:.6f}")
print(f"Asimetría: {residuos_sarima.skew():.4f}")
print(f"Curtosis: {residuos_sarima.kurtosis():.4f}")
if p_norm > 0.05:
    print("✓ Residuos siguen distribución normal (no se rechaza H₀)")
else:
    print("✗ Residuos NO son normales")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: QQ plot - normalidad de residuos.*

1. Histograma
2. QQ plot
3. Boxplot
4. Test de normalidad

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: Forecast - predecir próximos 30 días

```python
n_forecast = 30
forecast = resultado_sarima.get_forecast(steps=n_forecast)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int(alpha=0.05)

print("=== Forecast SARIMA: Próximos 30 días ===")
fechas_future = pd.date_range(df_train.index[-1] + pd.Timedelta(days=1), periods=n_forecast, freq='D')
for i in range(n_forecast):
    print(f"  {fechas_future[i].date()}: {forecast_mean.iloc[i]:.1f} "
          f"[{forecast_ci.iloc[i, 0]:.1f}, {forecast_ci.iloc[i, 1]:.1f}]")

# Comparación con test real
df_forecast = pd.DataFrame({
    'forecast': forecast_mean.values,
    'ci_lower': forecast_ci.iloc[:, 0].values,
    'ci_upper': forecast_ci.iloc[:, 1].values,
    'real': df_test['ventas'].iloc[:n_forecast].values
}, index=fechas_future)

df_forecast['error'] = df_forecast['real'] - df_forecast['forecast']
print(f"\nError MAPE: {np.mean(np.abs(df_forecast['error'] / df_forecast['real'])):.2%}")
print(f"RMSE: {np.sqrt(np.mean(df_forecast['error']**2)):.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: Forecast - predecir próximos 30 días.*

1. Comparación con test real

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: Intervalo de confianza del forecast

```python
fig, ax = plt.subplots(figsize=(14, 6))

# Últimos 60 días de train + forecast
train_tail = df_train.iloc[-60:]
ax.plot(train_tail.index, train_tail['ventas'], 'b-', label='Train (últimos 60d)', linewidth=1.5)
ax.plot(fechas_future, df_forecast['real'], 'g-', label='Test real', linewidth=1.5)
ax.plot(fechas_future, df_forecast['forecast'], 'r--', label='Forecast', linewidth=2)
ax.fill_between(
    fechas_future,
    df_forecast['ci_lower'],
    df_forecast['ci_upper'],
    color='red', alpha=0.15, label='IC 95%'
)
ax.set_title('Forecast SARIMA con Intervalo de Confianza al 95%')
ax.set_ylabel('Ventas')
ax.legend()
plt.tight_layout()
plt.savefig('e08_ej14_forecast_ic.png', dpi=100)
plt.close()

print("=== Intervalos de Confianza del Forecast ===")
print(f"Ancho promedio del IC: {(df_forecast['ci_upper'] - df_forecast['ci_lower']).mean():.1f} unidades")
print(f"Cobertura real: {(df_forecast['ci_lower'] <= df_forecast['real']) & (df_forecast['real'] <= df_forecast['ci_upper'])}")
print(f"Proporción dentro del IC: {((df_forecast['ci_lower'] <= df_forecast['real']) & (df_forecast['real'] <= df_forecast['ci_upper'])).mean():.0%}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: Intervalo de confianza del forecast.*

1. Últimos 60 días de train + forecast

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: Walk-forward validation

```python
print("=== Walk-Forward Validation ===")

def walk_forward_validation(y, model_class, order, seasonal_order, max_steps=30, window=100):
    """Evaluar modelo con walk-forward validation."""
    errors = []
    predictions = []
    actuals = []
    
    for t in range(window, len(y) - max_steps + 1):
        train = y.iloc[:t]
        test = y.iloc[t:t+max_steps]
        
        try:
            model = model_class(
                train,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            result = model.fit(disp=False)
            pred = result.forecast(steps=len(test))
            
            mape = np.mean(np.abs((test.values - pred.values) / test.values)) * 100
            errors.append(mape)
            predictions.append(pred)
            actuals.append(test)
        except:
            continue
    
    return errors, predictions, actuals

errors_wfv, preds_wfv, actuals_wfv = walk_forward_validation(
    df['ventas'], SARIMAX, (1,1,1), (1,1,1,7), 
    max_steps=7, window=200
)

print(f"Evaluaciones completadas: {len(errors_wfv)}")
print(f"MAPE promedio por ventana de 7 días: {np.mean(errors_wfv):.2f}%")
print(f"MAPE std: {np.std(errors_wfv):.2f}%")
print(f"MAPE min: {np.min(errors_wfv):.2f}%, max: {np.max(errors_wfv):.2f}%")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: Walk-forward validation.*

1. `print("=== Walk-Forward Validation ===")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 16: Comparar ARIMA vs SARIMA en test

```python
print("=== Comparación ARIMA vs SARIMA ===")

# Modelos
modelos = {
    'ARIMA(1,1,1)': ARIMA(y_train, order=(1, 1, 1)),
    'ARIMA(2,1,2)': ARIMA(y_train, order=(2, 1, 2)),
    'SARIMA(1,1,1)(1,1,1,7)': SARIMAX(y_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)),
    'SARIMA(0,1,1)(1,1,1,7)': SARIMAX(y_train, order=(0, 1, 1), seasonal_order=(1, 1, 1, 7))
}

resultados = {}
for nombre, modelo in modelos.items():
    try:
        result = modelo.fit(disp=False)
        pred = result.forecast(steps=len(df_test))
        
        mape = np.mean(np.abs((df_test['ventas'].values - pred.values) / df_test['ventas'].values)) * 100
        rmse = np.sqrt(np.mean((df_test['ventas'].values - pred.values)**2))
        mae = np.mean(np.abs(df_test['ventas'].values - pred.values))
        
        resultados[nombre] = {
            'AIC': result.aic,
            'MAPE': mape,
            'RMSE': rmse,
            'MAE': mae
        }
        print(f"{nombre:<35} AIC={result.aic:<8.1f} MAPE={mape:<6.2f}% RMSE={rmse:<8.2f} MAE={mae:<8.2f}")
    except Exception as e:
        print(f"{nombre:<35} ERROR: {e}")

# Mejor modelo
mejor = min(resultados, key=lambda x: resultados[x]['MAPE'])
print(f"\n✓ Mejor modelo (por MAPE): {mejor}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: Comparar ARIMA vs SARIMA en test.*

1. Modelos
2. Mejor modelo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 17: Grid search manual de parámetros

```python
print("=== Grid Search Manual de Parámetros SARIMA ===")

p_range = [0, 1, 2]
d_range = [0, 1]
q_range = [0, 1, 2]
P_range = [0, 1]
D_range = [0, 1]
Q_range = [0, 1]
s = 7

grid_results = []
total_combinaciones = len(p_range) * len(d_range) * len(q_range) * len(P_range) * len(D_range) * len(Q_range)
print(f"Total de combinaciones a evaluar: {total_combinaciones}")

count = 0
for p in p_range:
    for d in d_range:
        for q in q_range:
            for P in P_range:
                for D in D_range:
                    for Q in Q_range:
                        if p == 0 and q == 0:
                            continue  # modelo vacío
                        try:
                            modelo = SARIMAX(
                                y_train,
                                order=(p, d, q),
                                seasonal_order=(P, D, Q, s),
                                enforce_stationarity=False,
                                enforce_invertibility=False
                            )
                            result = modelo.fit(disp=False, maxiter=200)
                            grid_results.append({
                                'order': (p, d, q),
                                'seasonal_order': (P, D, Q, s),
                                'AIC': result.aic,
                                'BIC': result.bic
                            })
                            count += 1
                        except:
                            continue

# Top 5 mejores
df_grid = pd.DataFrame(grid_results).sort_values('AIC')
print(f"\nModelos evaluados exitosamente: {count}")
print("\nTop 5 modelos por AIC:")
for i, row in df_grid.head(5).iterrows():
    print(f"  ARIMA{row['order']}x{row['seasonal_order']}: AIC={row['AIC']:.2f}, BIC={row['BIC']:.2f}")

print(f"\nTop 5 modelos por BIC:")
for i, row in df_grid.sort_values('BIC').head(5).iterrows():
    print(f"  ARIMA{row['order']}x{row['seasonal_order']}: AIC={row['AIC']:.2f}, BIC={row['BIC']:.2f}")
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

*Ejemplo 17: Grid search manual de parámetros.*

1. Top 5 mejores

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 18: Integrador - SARIMA completo para pronóstico de demanda semanal

```python
print("=" * 70)
print("PRONÓSTICO INTEGRAL DE DEMANDA CON SARIMA")
print("=" * 70)

# 1. Pipeline completo
print("\n[1] ANÁLISIS EXPLORATORIO")
print(f"Datos: {len(df)} días ({len(df)/7:.1f} semanas)")
print(f"Media: {df['ventas'].mean():.1f}, σ: {df['ventas'].std():.1f}")

# 2. Estacionariedad
adf_p = adfuller(df['ventas'])[1]
print(f"\n[2] ESTACIONARIEDAD: p={adf_p:.4f}")
if adf_p > 0.05:
    print("→ Serie no estacionaria. Se aplicará diferenciación.")

# 3. Descomposición para entender componentes
des = seasonal_decompose(df['ventas'], model='additive', period=7)
print(f"\n[3] DESCOMPOSICIÓN")
print(f"Tendencia: {des.trend.mean():.1f} ± {des.trend.std():.1f}")
print(f"Estacional σ: {des.seasonal.std():.1f}")
print(f"Residuo σ: {des.resid.std():.1f}")

# 4. Auto-ARIMA óptimo
print(f"\n[4] BÚSQUEDA DE MODELO ÓPTIMO")
best_model = pm.auto_arima(
    df['ventas'],
    seasonal=True, m=7,
    stepwise=True, trace=False,
    error_action='ignore', suppress_warnings=True,
    information_criterion='aic',
    max_p=3, max_q=3, max_P=1, max_Q=1,
    random_state=42
)
print(f"Mejor modelo: ARIMA{best_model.order}x{best_model.seasonal_order}")
print(f"AIC={best_model.aic():.2f}, BIC={best_model.bic():.2f}")

# 5. Entrenar modelo final
modelo_final = SARIMAX(
    df['ventas'],
    order=best_model.order,
    seasonal_order=best_model.seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)
resultado_final = modelo_final.fit(disp=False)

# 6. Diagnóstico de residuos
resid_final = resultado_final.resid.dropna()
lb_final = acorr_ljungbox(resid_final, lags=[7, 14], return_df=True)
print(f"\n[5] DIAGNÓSTICO")
print(f"Ljung-Box lag 7: p={lb_final.loc[7, 'lb_pvalue']:.4f}")
print(f"Ljung-Box lag 14: p={lb_final.loc[14, 'lb_pvalue']:.4f}")
print(f"Residuo σ: {resid_final.std():.2f}")

# 7. Forecast 28 días (4 semanas)
n_dias = 28
forecast_final = resultado_final.get_forecast(steps=n_dias)
pred_final = forecast_final.predicted_mean
ci_final = forecast_final.conf_int(alpha=0.05)

fechas_f = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=n_dias, freq='D')

print(f"\n[6] FORECAST {n_dias} DÍAS")
for i in range(n_dias):
    print(f"  {fechas_f[i].date()}: {pred_final.iloc[i]:.1f} [{ci_final.iloc[i, 0]:.1f}, {ci_final.iloc[i, 1]:.1f}]")

# 8. Métricas de calidad
print(f"\n[7] MÉTRICAS DE CALIDAD DEL MODELO")
print(f"AIC: {resultado_final.aic:.2f}")
print(f"BIC: {resultado_final.bic:.2f}")
print(f"Log-Likelihood: {resultado_final.llf:.2f}")

# 9. Visualización final
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Serie completa + forecast
axes[0].plot(df.index[-180:], df['ventas'].iloc[-180:], 'b-', label='Histórico (180d)')
axes[0].plot(fechas_f, pred_final, 'r--', linewidth=2, label='Forecast')
axes[0].fill_between(fechas_f, ci_final.iloc[:, 0], ci_final.iloc[:, 1],
                      color='red', alpha=0.15, label='IC 95%')
axes[0].set_title(f'Pronóstico de Demanda - {n_dias} días')
axes[0].set_ylabel('Ventas')
axes[0].legend()

# Residuos
axes[1].plot(resultado_final.resid.index[-180:], resultado_final.resid.iloc[-180:], 
             'purple', marker='.', markersize=2, linestyle='-', alpha=0.7)
axes[1].axhline(0, color='gray')
axes[1].axhline(2*resultado_final.resid.std(), color='red', linestyle='--', alpha=0.5)
axes[1].axhline(-2*resultado_final.resid.std(), color='red', linestyle='--', alpha=0.5)
axes[1].set_title('Residuos del Modelo (últimos 180 días)')
axes[1].set_ylabel('Residuo')

plt.suptitle('Pipeline Completo SARIMA para Pronóstico de Ventas', fontsize=15)
plt.tight_layout()
plt.savefig('e08_ej18_integrador.png', dpi=100, bbox_inches='tight')
plt.close()
print("\n✓ Pipeline completado. Gráfico guardado.")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador - SARIMA completo para pronóstico de demanda semanal.*

1. 1. Pipeline completo
2. 2. Estacionariedad
3. 3. Descomposición para entender componentes
4. 4. Auto-ARIMA óptimo
5. 5. Entrenar modelo final
6. 6. Diagnóstico de residuos
7. 7. Forecast 28 días (4 semanas)
8. 8. Métricas de calidad

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 3. Ejercicios

1. **Diagnóstico de modelo sub-ajustado**: Ajusta un ARIMA(0,0,0) a la serie y analiza los residuos. ¿Qué dice Ljung-Box? ¿Por qué es importante no usar este modelo?

2. **Sobrediferenciación**: Ajusta un ARIMA(1,2,1) a la serie (d=2). Compara AIC y pronóstico vs ARIMA(1,1,1). ¿Qué observas en los residuos?

3. **Estacionalidad con s=30**: Usa auto_arima con m=30 (estacionalidad mensual). Compara el resultado vs m=7. ¿Cuál modelo es mejor según AIC? ¿Tiene sentido para datos diarios?

4. **Intervención (outlier)**: Agrega un outlier extremo (valor=500, día 200). ¿Cómo cambia el modelo SARIMA? ¿Lo detecta en residuos?

5. **Grid search con validación**: Implementa un grid search donde cada modelo se evalúe con walk-forward validation (MAPE en test) en lugar de AIC. Compara con los resultados del Ejemplo 17.

6. **Forecast a largo plazo**: Genera forecast a 90 días con SARIMA. ¿Cómo se comporta el intervalo de confianza? ¿Por qué se ensancha? Discute la utilidad práctica.

7. **Modelo sin estacionalidad**: Ajusta ARIMA a la serie original y a la serie desestacionalizada (resta componente estacional de STL). Compara precisión en test.

8. **Pipeline automatizado**: Escribe una función que reciba una serie temporal, detecte automáticamente la estacionalidad (usando ACF o periodoograma), ejecute auto_arima, valide residuos (Ljung-Box, normalidad) y retorne el pronóstico + diagnóstico.

---

## 4. Resumen

| Concepto | Punto clave |
|----------|-------------|
| **ARIMA(p,d,q)** | Modelar autocorrelación + diferenciación + media móvil |
| **SARIMA(P,D,Q,s)** | Extensión estacional del ARIMA |
| **ACF/PACF** | Identificar orden: ACF → MA(q), PACF → AR(p) |
| **Diferenciación** | Hacer serie estacionaria; d para tendencia, D para estacionalidad |
| **auto_arima** | Búsqueda automática eficiente con stepwise |
| **Ljung-Box** | Validar que residuos son ruido blanco (p > 0.05) |
| **Forecast + IC** | SARIMA da predicción puntual + intervalos de confianza |
| **Walk-forward** | Mejor evaluación que train/test fijo para series temporales |
