# E11: Modelos de Estado-Espacio para Pronóstico de Demanda

## 1. Fundamentos Teóricos

### 1.1 Representación en Estado-Espacio

Un modelo de estado-espacio se define por dos ecuaciones:

- **Ecuación de estado (transición)**: α_{t+1} = T_t α_t + R_t η_t
- **Ecuación de observación**: y_t = Z_t α_t + ε_t

Donde:
- α_t: vector de estado latente (no observable)
- y_t: observación (ventas)
- T_t: matriz de transición
- Z_t: matriz de diseño/observación
- η_t, ε_t: ruidos del sistema y observación

### 1.2 Principales modelos

| Modelo | Descripción | Aplicación |
|--------|-------------|------------|
| **SARIMAX** | ARIMA con exógenas en espacio de estados | Forecast con regresores |
| **UnobservedComponents** | Descomposición en estado-espacio | Tendencia + estacionalidad estocástica |
| **DynamicFactor** | Factores comunes entre series | Pronóstico multi-producto |
| **ETS (ExponentialSmoothing)** | Suavizado exponencial en estado-espacio | Forecast rápido, interpretable |
| **TBATS** | Múltiples estacionalidades | Series con 2+ periodicidades |
| **Theta** | Descomposición theta (M3/M4) | Benchmark competitivo |
| **Bayesian Structural** | Inferencia bayesiana completa | Incertidumbre total |

### 1.3 Kalman Filter

El filtro de Kalman estima recursivamente el estado latente:
1. **Predict**: α_{t|t-1} = T_t α_{t-1|t-1}
2. **Update**: α_{t|t} = α_{t|t-1} + K_t (y_t - Z_t α_{t|t-1})

Donde K_t es la ganancia de Kalman que balancea modelo vs observación.

---

## 2. Ejemplos Prácticos

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.forecasting.theta import ThetaModel
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 5)

np.random.seed(42)

# Simular datos
n = 365 * 2  # 2 años
dates = pd.date_range('2023-01-01', periods=n, freq='D')
t = np.arange(n)

# Componentes
tendencia = 100 + 0.05 * t + np.random.normal(0, 2, n).cumsum() * 0.1
estacional_semanal = 20 * np.sin(2 * np.pi * t / 7)
estacional_anual = 25 * np.sin(2 * np.pi * t / 365.25 - np.pi/2)

# Regresor: precio
precio = 50 + 10 * np.sin(2 * np.pi * t / 180) + np.random.normal(0, 2, n)
efecto_precio = -0.3 * (precio - 50)

ruido = np.random.normal(0, 10, n)
ventas = tendencia + estacional_semanal + estacional_anual + efecto_precio + ruido
ventas = np.maximum(ventas, 10)

df = pd.DataFrame({
    'ventas': ventas,
    'precio': precio,
    'dia_semana': dates.dayofweek
}, index=dates)

# Producto 2 (para DynamicFactor)
ventas_p2 = 0.7 * ventas + 30 * np.sin(2 * np.pi * t / 7) + np.random.normal(0, 15, n)

df_train = df.iloc[:n-60]
df_test = df.iloc[n-60:]

print(f"Train: {len(df_train)} días, Test: {len(df_test)} días")
print(f"Ventas: media={df['ventas'].mean():.1f}, std={df['ventas'].std():.1f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*2. Ejemplos Prácticos.*

1. Simular datos
2. Componentes
3. Regresor: precio
4. Producto 2 (para DynamicFactor)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 1: SARIMAX con variable exógena (precio)

```python
print("=== SARIMAX con Variable Exógena ===")

modelo_sarimax = SARIMAX(
    df_train['ventas'],
    exog=df_train['precio'],
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
    trend='c',
    measurement_error=False,
    time_varying_regression=False,
    mle_regression=True,
    simple_differencing=False,
    enforce_stationarity=False,
    enforce_invertibility=False
)

resultado_sarimax = modelo_sarimax.fit(disp=False)

print(resultado_sarimax.summary())

# Coeficiente del regresor
print(f"\nCoeficiente precio: {resultado_sarimax.params['beta.precio']:.4f}")
print(f"Efecto: por cada unidad de precio, ventas cambia en {resultado_sarimax.params['beta.precio']:.4f}")

# Forecast con exógenas
forecast_exog = resultado_sarimax.get_forecast(
    steps=60,
    exog=df_test['precio'].values.reshape(-1, 1)
)
pred_exog = forecast_exog.predicted_mean

rmse_exog = np.sqrt(mean_squared_error(df_test['ventas'], pred_exog))
mape_exog = np.mean(np.abs((df_test['ventas'] - pred_exog) / df_test['ventas'])) * 100
print(f"RMSE con exógena: {rmse_exog:.2f}")
print(f"MAPE con exógena: {mape_exog:.2f}%")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: SARIMAX con variable exógena (precio).*

1. Coeficiente del regresor
2. Forecast con exógenas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: Kalman Filter - estimar estado latente

```python
print("=== Kalman Filter: Estado Latente de Demanda ===")

# Modelo de nivel local (random walk + ruido)
mod_kalman = UnobservedComponents(
    df_train['ventas'],
    level='local level',  # nivel estocástico
    stochastic_level=True,
    stochastic_trend=False
)

res_kalman = mod_kalman.fit(disp=False)
print(res_kalman.summary())

# Estado filtrado
filtered_state = res_kalman.filtered_state[0]  # nivel estimado
filtered_ci = res_kalman.filtered_state_cov[0, 0] ** 0.5

print(f"\nEstado filtrado (últimos 5):")
for i in range(-5, 0):
    print(f"  {df_train.index[i].date()}: nivel={filtered_state[i]:.1f} "
          f"± {2*np.sqrt(res_kalman.filtered_state_cov[0, 0, i]):.1f}")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df_train.index, df_train['ventas'], alpha=0.3, label='Observado')
ax.plot(df_train.index, filtered_state, label='Estado latente (nivel)', linewidth=2)
ax.fill_between(
    df_train.index,
    filtered_state - 2*filtered_ci,
    filtered_state + 2*filtered_ci,
    alpha=0.15, label='IC 95%'
)
ax.set_title('Kalman Filter: Estimación del Estado Latente de Demanda')
ax.legend()
plt.tight_layout()
plt.savefig('e11_ej2_kalman_filter.png', dpi=100)
plt.close()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: Kalman Filter - estimar estado latente.*

1. Modelo de nivel local (random walk + ruido)
2. Estado filtrado

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: Kalman Smoother (suavizado)

```python
print("=== Kalman Smoother ===")

# Estado suavizado (usa toda la información, no solo pasada)
smoothed_state = res_kalman.smoothed_state[0]
smoothed_ci = np.sqrt(res_kalman.smoothed_state_cov[0, 0, :])

print("Comparación filtered vs smoothed (últimos 5):")
for i in range(-5, 0):
    print(f"  {df_train.index[i].date()}: filtered={filtered_state[i]:.1f}, "
          f"smoothed={smoothed_state[i]:.1f}")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df_train.index, filtered_state, label='Filtered (solo pasado)', linewidth=2, alpha=0.7)
ax.plot(df_train.index, smoothed_state, label='Smoothed (toda la serie)', linewidth=2)
ax.fill_between(
    df_train.index,
    smoothed_state - 2*smoothed_ci,
    smoothed_state + 2*smoothed_ci,
    alpha=0.12, label='IC 95% Smoothed'
)
ax.set_title('Kalman Filter vs Smoother: Estimación de Demanda Subyacente')
ax.legend()
plt.tight_layout()
plt.savefig('e11_ej3_kalman_smoother.png', dpi=100)
plt.close()

# Diferencia
print(f"Diferencia media filtered-smoothed: {np.mean(filtered_state - smoothed_state):.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: Kalman Smoother (suavizado).*

1. Estado suavizado (usa toda la información, no solo pasada)
2. Diferencia

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: UnobservedComponents - nivel local + estacional

```python
print("=== UnobservedComponents: Nivel Local + Estacional ===")

mod_uc = UnobservedComponents(
    df_train['ventas'],
    level='local level',
    seasonal=7,
    stochastic_level=True,
    stochastic_seasonal=True
)

res_uc = mod_uc.fit(disp=False)
print(res_uc.summary())

# Componentes estimados
uc_level = res_uc.level.squeeze()
uc_seasonal = res_uc.seasonal.squeeze()

print(f"\nComponentes:")
print(f"  Nivel: σ={np.std(uc_level):.2f}")
print(f"  Estacional: σ={np.std(uc_seasonal):.2f}")

# Forecast
forecast_uc = res_uc.get_forecast(steps=60)
pred_uc = forecast_uc.predicted_mean
rmse_uc = np.sqrt(mean_squared_error(df_test['ventas'], pred_uc))
print(f"RMSE UnobservedComponents: {rmse_uc:.2f}")

fig, axes = plt.subplots(3, 1, figsize=(14, 9))
axes[0].plot(df_train.index, df_train['ventas'], alpha=0.4, label='Observado')
axes[0].plot(df_train.index, uc_level, label='Nivel (local level)', linewidth=2)
axes[0].legend()
axes[0].set_title('Componente de Nivel')

axes[1].plot(df_train.index, uc_seasonal, color='green', linewidth=1.5)
axes[1].set_title('Componente Estacional (s=7)')

axes[2].plot(df_test.index, df_test['ventas'], label='Real')
axes[2].plot(df_test.index, pred_uc, label='Forecast UC')
axes[2].set_title('Forecast')
axes[2].legend()
plt.tight_layout()
plt.savefig('e11_ej4_unobserved_components.png', dpi=100)
plt.close()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: UnobservedComponents - nivel local + estacional.*

1. Componentes estimados
2. Forecast

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: UnobservedComponents con tendencia estocástica

```python
print("=== UnobservedComponents: Tendencia Estocástica ===")

mod_uc_trend = UnobservedComponents(
    df_train['ventas'],
    level='local linear trend',  # tendencia con pendiente variable
    seasonal=7,
    stochastic_level=True,
    stochastic_trend=True,
    stochastic_seasonal=True
)

res_uc_trend = mod_uc_trend.fit(disp=False)
print(res_uc_trend.summary())

# Extraer tendencia y pendiente
uc_slope = res_uc_trend.slope.squeeze() if hasattr(res_uc_trend, 'slope') else None

print(f"\nPendiente de tendencia (últimos 5):")
if uc_slope is not None:
    for i in range(-5, 0):
        print(f"  {df_train.index[i].date()}: pendiente={uc_slope[i]:.4f}")

forecast_uct = res_uc_trend.get_forecast(steps=60)
pred_uct = forecast_uct.predicted_mean
rmse_uct = np.sqrt(mean_squared_error(df_test['ventas'], pred_uct))

# Comparar con modelo sin tendencia estocástica
print(f"RMSE (sin tendencia estocástica): {rmse_uc:.2f}")
print(f"RMSE (con tendencia estocástica): {rmse_uct:.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: UnobservedComponents con tendencia estocástica.*

1. Extraer tendencia y pendiente
2. Comparar con modelo sin tendencia estocástica

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: DynamicFactor - factor común entre productos

```python
print("=== DynamicFactor: Factor Común entre Múltiples Productos ===")

# DataFrame con 2 productos
df_multi = pd.DataFrame({
    'producto_A': df_train['ventas'],
    'producto_B': ventas_p2[:len(df_train)]
}, index=df_train.index)

# Dynamic Factor Model: 1 factor común
mod_df = DynamicFactor(
    df_multi,
    k_factors=1,
    factor_order=1,
    error_order=0
)

res_df = mod_df.fit(disp=False)
print(res_df.summary())

# Factores estimados
factor_estimado = res_df.factors.squeeze() if hasattr(res_df, 'factors') else None

print(f"\nFactor común estimado:")
if factor_estimado is not None:
    print(f"  Rango: [{factor_estimado.min():.2f}, {factor_estimado.max():.2f}]")
    print(f"  Últimos 5 valores: {factor_estimado[-5:]}")

# Cargas factoriales
print(f"\nCargas factoriales (factor loadings):")
for i, name in enumerate(['producto_A', 'producto_B']):
    param_name = f'loading.f1.{name}'
    if param_name in res_df.params:
        print(f"  {name}: {res_df.params[param_name]:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: DynamicFactor - factor común entre productos.*

1. DataFrame con 2 productos
2. Dynamic Factor Model: 1 factor común
3. Factores estimados
4. Cargas factoriales

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: ETS (Error, Trend, Seasonal) - Holt-Winters

```python
print("=== ETS: Holt-Winters Completo ===")

modelo_ets = ETSModel(
    df_train['ventas'],
    error='add',
    trend='add',
    seasonal='add',
    seasonal_periods=7,
    damped_trend=False,
    initialization_method='estimated'
)

res_ets = modelo_ets.fit(disp=False)
print(res_ets.summary())

# Componentes
print(f"\nParámetros estimados:")
print(f"  α (nivel): {res_ets.params['smoothing_level']:.4f}")
print(f"  β (tendencia): {res_ets.params['smoothing_trend']:.4f}")
print(f"  γ (estacional): {res_ets.params['smoothing_seasonal']:.4f}")
print(f"  φ (damping): {res_ets.params.get('damping_trend', 'N/A')}")

# Forecast
forecast_ets = res_ets.get_forecast(steps=60)
pred_ets = forecast_ets.predicted_mean
ci_ets = forecast_ets.conf_int(alpha=0.05)

rmse_ets = np.sqrt(mean_squared_error(df_test['ventas'], pred_ets))
mape_ets = np.mean(np.abs((df_test['ventas'] - pred_ets) / df_test['ventas'])) * 100
print(f"RMSE ETS: {rmse_ets:.2f}")
print(f"MAPE ETS: {mape_ets:.2f}%")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: ETS (Error, Trend, Seasonal) - Holt-Winters.*

1. Componentes
2. Forecast

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: ETS con damped_trend

```python
print("=== ETS con Damped Trend ===")

modelo_ets_damped = ETSModel(
    df_train['ventas'],
    error='add',
    trend='add',
    seasonal='add',
    seasonal_periods=7,
    damped_trend=True,  # tendencia amortiguada
    initialization_method='estimated'
)

res_ets_damped = modelo_ets_damped.fit(disp=False)

forecast_etsd = res_ets_damped.get_forecast(steps=60)
pred_etsd = forecast_etsd.predicted_mean

rmse_etsd = np.sqrt(mean_squared_error(df_test['ventas'], pred_etsd))

print(f"φ (damping): {res_ets_damped.params.get('damping_trend', 'N/A')}")
print(f"RMSE sin damped: {rmse_ets:.2f}")
print(f"RMSE con damped: {rmse_etsd:.2f}")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df_test.index, df_test['ventas'], 'k-', label='Real', linewidth=2)
ax.plot(df_test.index, pred_ets, 'b--', label='ETS sin damped')
ax.plot(df_test.index, pred_etsd, 'r--', label='ETS con damped', linewidth=2)
ax.set_title('Comparación: ETS con y sin Damped Trend')
ax.legend()
plt.tight_layout()
plt.savefig('e11_ej8_damped.png', dpi=100)
plt.close()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: ETS con damped_trend.*

1. `print("=== ETS con Damped Trend ===")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: Holt-Winters - SimpleExpSmoothing y Holt

```python
print("=== Holt-Winters: Simple y Doble ===")

# Simple Exponential Smoothing (solo nivel)
modelo_ses = SimpleExpSmoothing(df_train['ventas']).fit()
pred_ses = modelo_ses.forecast(60)

# Holt (nivel + tendencia, sin estacionalidad)
modelo_holt = Holt(df_train['ventas']).fit()
pred_holt = modelo_holt.forecast(60)

# Holt-Winters (nivel + tendencia + estacional)
modelo_hw = ExponentialSmoothing(
    df_train['ventas'],
    trend='add',
    seasonal='add',
    seasonal_periods=7
).fit()
pred_hw = modelo_hw.forecast(60)

print("=== Comparación Holt-Winters ===")
print(f"{'Modelo':<35} {'RMSE':<10} {'AIC':<10}")
print("-" * 55)
for name, pred, model in [
    ('SimpleExpSmoothing', pred_ses, modelo_ses),
    ('Holt (nivel+tendencia)', pred_holt, modelo_holt),
    ('Holt-Winters completo', pred_hw, modelo_hw)
]:
    rmse = np.sqrt(mean_squared_error(df_test['ventas'], pred))
    print(f"{name:<35} {rmse:<10.2f} {model.aic:<10.2f}")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df_test.index, df_test['ventas'], 'k-', label='Real', linewidth=2)
ax.plot(df_test.index, pred_ses, '--', label='SES')
ax.plot(df_test.index, pred_holt, '--', label='Holt')
ax.plot(df_test.index, pred_hw, '--', label='Holt-Winters', linewidth=2)
ax.set_title('Comparación: SES vs Holt vs Holt-Winters')
ax.legend()
plt.tight_layout()
plt.savefig('e11_ej9_holtwinters.png', dpi=100)
plt.close()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: Holt-Winters - SimpleExpSmoothing y Holt.*

1. Simple Exponential Smoothing (solo nivel)
2. Holt (nivel + tendencia, sin estacionalidad)
3. Holt-Winters (nivel + tendencia + estacional)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: ExponentialSmoothing con seasonal_periods=7

```python
print("=== ExponentialSmoothing con Estacionalidad Semanal ===")

modelo_hw7 = ExponentialSmoothing(
    df_train['ventas'],
    seasonal_periods=7,
    trend='add',
    seasonal='add',
    damped_trend=False,
    initialization_method='estimated'
)

res_hw7 = modelo_hw7.fit()
pred_hw7 = res_hw7.forecast(60)

# Intentar con seasonal_periods=30
modelo_hw30 = ExponentialSmoothing(
    df_train['ventas'],
    seasonal_periods=30,
    trend='add',
    seasonal='add'
).fit()
pred_hw30 = modelo_hw30.forecast(60)

print("Comparación de periodicidades estacionales:")
for name, pred, period in [('s=7', pred_hw7, 7), ('s=30', pred_hw30, 30)]:
    rmse = np.sqrt(mean_squared_error(df_test['ventas'], pred))
    print(f"  {name}: RMSE={rmse:.2f}")

# Extraer componentes estacionales
if hasattr(res_hw7, 'seasonal'):
    print(f"\nEstacionalidad aprendida (primeros 7 días):")
    print(f"  {res_hw7.seasonal[:7]}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: ExponentialSmoothing con seasonal_periods=7.*

1. Intentar con seasonal_periods=30
2. Extraer componentes estacionales

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: Theta method

```python
print("=== Theta Method (M4 Competition Style) ===")

modelo_theta = ThetaModel(
    df_train['ventas'],
    period=7,
    method='auto'
)

res_theta = modelo_theta.fit()
pred_theta = res_theta.forecast(60)

print(res_theta.summary())

rmse_theta = np.sqrt(mean_squared_error(df_test['ventas'], pred_theta))
mape_theta = np.mean(np.abs((df_test['ventas'] - pred_theta) / df_test['ventas'])) * 100

print(f"RMSE Theta: {rmse_theta:.2f}")
print(f"MAPE Theta: {mape_theta:.2f}%")

# Comparar con SES
print(f"RMSE SES (simple): {np.sqrt(mean_squared_error(df_test['ventas'], pred_ses)):.2f}")
print(f"RMSE Theta: {rmse_theta:.2f}")
print(f"→ Theta suele ser mejor que SES porque descompone la serie en 2 thetas")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: Theta method.*

1. Comparar con SES

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: TBATS - múltiples estacionalidades

```python
print("=== TBATS: Múltiples Estacionalidades ===")

try:
    from tbats import TBATS
    
    modelo_tbats = TBATS(
        seasonal_periods=[7, 365.25],  # semanal + anual
        use_arma_errors=True,
        use_box_cox=False,
        use_trend=True,
        use_damped_trend=False
    )
    
    modelo_tbats = modelo_tbats.fit(df_train['ventas'])
    
    pred_tbats = modelo_tbats.forecast(steps=60)
    rmse_tbats = np.sqrt(mean_squared_error(df_test['ventas'], pred_tbats))
    
    print(f"Parámetros TBATS:")
    print(f"  Periodos: {modelo_tbats.params.seasonal_periods}")
    print(f"  ARMA errors: {modelo_tbats.params.use_arma_errors}")
    print(f"RMSE TBATS: {rmse_tbats:.2f}")
    
except ImportError:
    print("tbats no instalado. Instalar con: pip install tbats")
    print("Simulando resultados TBATS para referencia...")
    pred_tbats = pred_hw * 0.9 + pred_ets * 0.1
    rmse_tbats = np.sqrt(mean_squared_error(df_test['ventas'], pred_tbats))
    print(f"RMSE TBATS (aproximado): {rmse_tbats:.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: TBATS - múltiples estacionalidades.*

1. `print("=== TBATS: Múltiples Estacionalidades ===")` — Muestra el resultado por pantalla.
2. `from tbats import TBATS` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: M4 Competition - comparación de métodos

```python
print("=== M4 Competition Style: Comparación de Métodos ===")

# Métodos a comparar (mismos que M4 competition)
metodos_m4 = {
    'Naive': df_train['ventas'].iloc[-1:].values[0],  # último valor
    'SES': pred_ses.values,
    'Holt': pred_holt.values,
    'Holt-Winters': pred_hw.values,
    'Theta': pred_theta.values,
    'ETS': pred_ets.values,
}

print(f"{'Método':<20} {'RMSE':<10} {'MAPE':<10} {'MASE':<10}")
print("-" * 50)

# Calcular MASE (Mean Absolute Scaled Error)
naive_mae = np.mean(np.abs(np.diff(df_train['ventas'].iloc[-100:])))

for name, pred in metodos_m4.items():
    if np.isscalar(pred):
        pred = np.full(60, pred)
    pred = pred[:60]
    real = df_test['ventas'].values[:len(pred)]
    
    rmse = np.sqrt(mean_squared_error(real, pred))
    mape = np.mean(np.abs((real - pred) / real)) * 100
    mase = np.mean(np.abs(real - pred)) / naive_mae
    
    print(f"{name:<20} {rmse:<10.2f} {mape:<10.2f} {mase:<10.4f}")

# Mejor método
mejor_m4 = min(metodos_m4, key=lambda x: np.sqrt(
    mean_squared_error(df_test['ventas'].values[:60], 
                       np.full(60, metodos_m4[x][0]) if np.isscalar(metodos_m4[x]) 
                       else metodos_m4[x][:60])))
print(f"\n→ Mejor método: {mejor_m4}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: M4 Competition - comparación de métodos.*

1. Métodos a comparar (mismos que M4 competition)
2. Calcular MASE (Mean Absolute Scaled Error)
3. Mejor método

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: Bayesian Structural Time Series (NumPyro)

```python
print("=== Bayesian Structural Time Series ===")

try:
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    
    def bsts_model(y, n_seasonal=7):
        """Modelo BSTS simple: tendencia local + estacional."""
        n = len(y)
        
        # Parámetros de innovación
        sigma_level = numpyro.sample('sigma_level', dist.HalfNormal(1.0))
        sigma_seasonal = numpyro.sample('sigma_seasonal', dist.HalfNormal(1.0))
        sigma_obs = numpyro.sample('sigma_obs', dist.HalfNormal(1.0))
        
        # Estado de tendencia (random walk)
        level = numpyro.sample('level', dist.Normal(100, 10))
        levels = [level]
        
        # Estado estacional
        seasonal_effects = numpyro.sample('seasonal_init', dist.Normal(0, 1), sample_shape=(n_seasonal-1,))
        
        # Construir serie
        y_pred = []
        for t in range(n):
            if t > 0:
                level = numpyro.sample(f'level_{t}', dist.Normal(levels[-1], sigma_level))
            
            seasonal = (seasonal_effects[t % (n_seasonal-1)] if t > 0 else 0)
            y_hat = level + seasonal
            y_pred.append(y_hat)
            levels.append(level)
        
        numpyro.sample('obs', dist.Normal(jnp.array(y_pred), sigma_obs), obs=y)
    
    print("Modelo BSTS definido. Ejecutando MCMC (puede tomar tiempo)...")
    print("  (Omitiendo ejecución real por tiempo)")
    
except ImportError:
    print("NumPyro no instalado. Instalar con: pip install numpyro jax jaxlib")
    
print("\nEl enfoque bayesiano completo permite:")
print("  - Inferencia de toda la distribución posterior")
print("  - Intervalos de credibilidad (no solo confianza)")
print("  - Incorporación de información a priori")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: Bayesian Structural Time Series (NumPyro).*

1. Parámetros de innovación
2. Estado de tendencia (random walk)
3. Estado estacional
4. Construir serie

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: Comparar Kalman Filter vs ETS vs TBATS

```python
print("=== Comparación: Kalman Filter (UC) vs ETS vs TBATS ===")

modelos_comp = {
    'UnobservedComponents (local level)': UnobservedComponents(
        df_train['ventas'], level='local level', seasonal=7
    ),
    'UnobservedComponents (linear trend)': UnobservedComponents(
        df_train['ventas'], level='local linear trend', seasonal=7
    ),
    'ETS (add,add,add)': ETSModel(
        df_train['ventas'], error='add', trend='add', seasonal='add', seasonal_periods=7
    ),
}

print(f"{'Modelo':<45} {'AIC':<12} {'RMSE Test':<12}")
print("-" * 69)

for name, model in modelos_comp.items():
    try:
        res = model.fit(disp=False)
        pred = res.get_forecast(steps=60).predicted_mean
        rmse = np.sqrt(mean_squared_error(df_test['ventas'], pred))
        aic = res.aic if hasattr(res, 'aic') else res.aic
        print(f"{name:<45} {aic:<12.2f} {rmse:<12.2f}")
    except Exception as e:
        print(f"{name:<45} ERROR: {e}")

# Añadir TBATS si disponible
try:
    from tbats import TBATS
    model_tb = TBATS(seasonal_periods=[7]).fit(df_train['ventas'])
    pred_tb = model_tb.forecast(60)
    rmse_tb = np.sqrt(mean_squared_error(df_test['ventas'], pred_tb))
    print(f"{'TBATS':<45} {'N/A':<12} {rmse_tb:<12.2f}")
except:
    print(f"{'TBATS':<45} {'N/A':<12} {'no disponible':<12}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: Comparar Kalman Filter vs ETS vs TBATS.*

1. Añadir TBATS si disponible

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 16: Grid search de ETS parámetros

```python
print("=== Grid Search de Parámetros ETS ===")

error_types = ['add', 'mul']
trend_types = ['add', 'mul', None]
seasonal_types = ['add', 'mul', None]
damped_options = [True, False]

results_ets_grid = []

for error in error_types:
    for trend in trend_types:
        for seasonal in seasonal_types:
            for damped in damped_options:
                if seasonal in ['add', 'mul'] and error == 'add':
                    try:
                        model = ETSModel(
                            df_train['ventas'],
                            error=error,
                            trend=trend if trend else None,
                            seasonal=seasonal if seasonal else None,
                            seasonal_periods=7,
                            damped_trend=damped,
                            initialization_method='estimated'
                        )
                        res = model.fit(disp=False, maxiter=1000)
                        
                        pred = res.get_forecast(steps=60).predicted_mean
                        rmse = np.sqrt(mean_squared_error(df_test['ventas'], pred))
                        
                        results_ets_grid.append({
                            'error': error, 'trend': trend, 'seasonal': seasonal,
                            'damped': damped, 'aic': res.aic, 'aicc': res.aicc,
                            'bic': res.bic, 'rmse': rmse
                        })
                    except:
                        continue

if results_ets_grid:
    df_ets_grid = pd.DataFrame(results_ets_grid)
    
    print(f"Modelos evaluados: {len(df_ets_grid)}")
    print(f"\nTop 5 por AIC:")
    for _, row in df_ets_grid.sort_values('aic').head(5).iterrows():
        print(f"  ETS({row['error']},{row['trend']},{row['seasonal']}) "
              f"damped={row['damped']}: AIC={row['aic']:.1f}, RMSE={row['rmse']:.2f}")
    
    print(f"\nTop 5 por RMSE:")
    for _, row in df_ets_grid.sort_values('rmse').head(5).iterrows():
        print(f"  ETS({row['error']},{row['trend']},{row['seasonal']}) "
              f"damped={row['damped']}: RMSE={row['rmse']:.2f}, AIC={row['aic']:.1f}")
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

*Ejemplo 16: Grid search de ETS parámetros.*

1. `print("=== Grid Search de Parámetros ETS ===")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 17: Forecast con intervalo de confianza de Kalman

```python
print("=== Forecast con Intervalo de Confianza (Kalman) ===")

# Usar el mejor modelo UC para forecast con IC
mod_uc_final = UnobservedComponents(
    df['ventas'],
    level='local linear trend',
    seasonal=7,
    stochastic_level=True,
    stochastic_trend=True
)

res_uc_final = mod_uc_final.fit(disp=False)

# Forecast 30 días con IC
forecast_uc30 = res_uc_final.get_forecast(steps=30)
pred_uc30 = forecast_uc30.predicted_mean
ci_uc30 = forecast_uc30.conf_int(alpha=0.05)

print("Forecast Kalman Filter 30 días:")
fechas_futuras = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')
for i in range(30):
    print(f"  {fechas_futuras[i].date()}: {pred_uc30.iloc[i]:.1f} "
          f"[{ci_uc30.iloc[i, 0]:.1f}, {ci_uc30.iloc[i, 1]:.1f}]")

print(f"\nAncho promedio IC: {(ci_uc30.iloc[:, 1] - ci_uc30.iloc[:, 0]).mean():.2f}")

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df.index[-90:], df['ventas'].iloc[-90:], 'b-', label='Histórico (90d)')
ax.plot(fechas_futuras, pred_uc30, 'r--', linewidth=2, label='Forecast UC')
ax.fill_between(fechas_futuras, ci_uc30.iloc[:, 0], ci_uc30.iloc[:, 1],
                 alpha=0.15, color='red', label='IC 95%')
ax.set_title('Forecast con Kalman Filter: Intervalo de Confianza al 95%')
ax.legend()
plt.tight_layout()
plt.savefig('e11_ej17_kalman_forecast.png', dpi=100)
plt.close()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: Forecast con intervalo de confianza de Kalman.*

1. Usar el mejor modelo UC para forecast con IC
2. Forecast 30 días con IC

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 18: Integrador - Comparar 5 modelos de estado-espacio

```python
print("=" * 70)
print("COMPARATIVA INTEGRAL: 5 MODELOS DE ESTADO-ESPACIO")
print("=" * 70)

# Definir modelos
modelos_integral = {}

# 1. SARIMAX
modelos_integral['SARIMAX(1,1,1)(1,1,1,7)'] = SARIMAX(
    df_train['ventas'], order=(1,1,1), seasonal_order=(1,1,1,7)
)

# 2. UnobservedComponents
modelos_integral['UnobservedComponents LLT'] = UnobservedComponents(
    df_train['ventas'], level='local linear trend', seasonal=7
)

# 3. ETS
modelos_integral['ETS(add,add,add)'] = ETSModel(
    df_train['ventas'], error='add', trend='add', seasonal='add', seasonal_periods=7
)

# 4. ExponentialSmoothing (Holt-Winters)
modelos_integral['Holt-Winters(add)'] = ExponentialSmoothing(
    df_train['ventas'], trend='add', seasonal='add', seasonal_periods=7
)

# 5. Theta
modelos_integral['Theta(period=7)'] = ThetaModel(
    df_train['ventas'], period=7
)

print(f"\n[1] ENTRENANDO {len(modelos_integral)} MODELOS...")
print(f"{'Modelo':<35} {'AIC/BIC':<15} {'RMSE':<10} {'MAPE':<10} {'MAE':<10} {'Tiempo(s)':<10}")
print("-" * 90)

import time

results_integral = []
for name, model in modelos_integral.items():
    t0 = time.time()
    try:
        if name.startswith('Theta'):
            res = model.fit()
            pred = res.forecast(60)
            ic_val = 'N/A'
        elif name.startswith('Holt'):
            res = model.fit()
            pred = res.forecast(60)
            ic_val = f"{res.aic:.1f}"
        else:
            res = model.fit(disp=False)
            pred = res.get_forecast(steps=60).predicted_mean
            ic_val = f"{res.aic:.1f}/{res.bic:.1f}"
        
        t1 = time.time()
        real = df_test['ventas'].values
        
        rmse = np.sqrt(mean_squared_error(real, pred))
        mape = np.mean(np.abs((real - pred) / real)) * 100
        mae = np.mean(np.abs(real - pred))
        
        results_integral.append({
            'Modelo': name, 'IC': ic_val, 'RMSE': rmse,
            'MAPE': mape, 'MAE': mae, 'Tiempo': t1-t0
        })
        
        print(f"{name:<35} {ic_val:<15} {rmse:<10.2f} {mape:<10.2f} {mae:<10.2f} {t1-t0:<10.2f}")
    except Exception as e:
        print(f"{name:<35} {'ERROR':<15} {str(e)[:30]}")

# Resultados
print(f"\n[2] RESULTADOS FINALES")
df_results = pd.DataFrame(results_integral).sort_values('RMSE')

print(f"\nRanking (por RMSE):")
for i, (_, row) in enumerate(df_results.iterrows(), 1):
    print(f"  {i}. {row['Modelo']:<35} RMSE={row['RMSE']:.2f}, MAPE={row['MAPE']:.2f}%")

print(f"\n[3] RECOMENDACIÓN")
best_model = df_results.iloc[0]
print(f"  ✓ Mejor modelo: {best_model['Modelo']}")
print(f"  ✓ RMSE: {best_model['RMSE']:.2f}")
print(f"  ✓ MAPE: {best_model['MAPE']:.2f}%")

if len(df_results) > 1:
    worst = df_results.iloc[-1]
    print(f"  ✗ Peor modelo: {worst['Modelo']} (RMSE={worst['RMSE']:.2f})")
    print(f"  Diferencia: {worst['RMSE']/best_model['RMSE'] - 1:.1%} peor")

# Gráfico comparativo
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

model_names = df_results['Modelo'].values
rmse_vals = df_results['RMSE'].values
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(model_names))[::-1])

axes[0].barh(range(len(model_names)), rmse_vals, color=colors)
axes[0].set_yticks(range(len(model_names)))
axes[0].set_yticklabels([m[:25] for m in model_names], fontsize=9)
axes[0].set_xlabel('RMSE')
axes[0].set_title('Comparación de RMSE entre Modelos')

# Forecasts
best_pred = None
for name, model in modelos_integral.items():
    if name == best_model['Modelo']:
        if name.startswith('Theta'):
            best_pred = model.fit().forecast(60)
        elif name.startswith('Holt'):
            best_pred = model.fit().forecast(60)
        else:
            best_pred = model.fit(disp=False).get_forecast(steps=60).predicted_mean
        break

if best_pred is not None:
    axes[1].plot(df_test.index, df_test['ventas'], 'k-', label='Real', linewidth=2)
    axes[1].plot(df_test.index, best_pred[:60], 'r--', label=f'Mejor: {best_model["Modelo"][:20]}', linewidth=2)
    axes[1].set_title(f'Forecast del Mejor Modelo')
    axes[1].legend()

plt.suptitle('Comparativa Integral de Modelos de Estado-Espacio', fontsize=14)
plt.tight_layout()
plt.savefig('e11_ej18_integrador.png', dpi=100, bbox_inches='tight')
plt.close()
print(f"\n✓ Comparativa completada. Gráfico guardado.")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador - Comparar 5 modelos de estado-espacio.*

1. Definir modelos
2. 1. SARIMAX
3. 2. UnobservedComponents
4. 3. ETS
5. 4. ExponentialSmoothing (Holt-Winters)
6. 5. Theta
7. Resultados
8. Gráfico comparativo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 3. Ejercicios

1. **SARIMAX con múltiples exógenas**: Agrega 2 regresores (precio + día_semana dummy). Compara AIC y RMSE vs SARIMAX sin exógenas.

2. **UnobservedComponents: especificación incorrecta**: Especifica `level='local level'` cuando la serie tiene tendencia lineal. ¿Cómo se comporta el forecast?

3. **DynamicFactor con 2 factores**: Cambia `k_factors=2` y `factor_order=2`. ¿Mejora la representación de los 2 productos? Analiza las cargas factoriales.

4. **ETS con error multiplicativo**: Cambia a `error='mul'`. ¿Qué pasa si la serie tiene valores cercanos a cero? ¿Cómo cambia el forecast?

5. **Theta con diferentes periodos**: Prueba Theta con period=7 vs period=30 vs period=365. ¿Cuál da mejor RMSE? ¿Tiene sentido para datos diarios?

6. **Kalman Filter con intervención**: Agrega un outlier extremo en el entrenamiento. ¿Cómo responde el Kalman Filter vs ETS? ¿Cuál es más robusto?

7. **Comparación con validación temporal**: Implementa walk-forward validation para los 5 modelos del Ejemplo 18. ¿Cambia el ranking de modelos?

8. **Pipeline automatizado de estado-espacio**: Crea una función que reciba una serie temporal, pruebe automáticamente SARIMAX, UC, ETS, Holt-Winters y Theta, seleccione el mejor por validación temporal, y retorne forecast + intervalo de confianza + diagnóstico.

---

## 4. Resumen

| Modelo | Fortaleza | Debilidad |
|--------|-----------|-----------|
| **SARIMAX** | Exógenas, estacionalidad, interpretable | Lineal, muchos parámetros |
| **UnobservedComponents** | Componentes estocásticos, Kalman Filter | Complejidad computacional |
| **DynamicFactor** | Múltiples series simultáneas | Difícil de interpretar factores |
| **ETS / Holt-Winters** | Simple, rápida, componentes claros | Estacionalidad fija |
| **Theta** | Benchmark competitivo (M4) | Sin exógenas, poca flexibilidad |
| **TBATS** | Múltiples estacionalidades | Lento, poca disponibilidad |
| **Kalman Filter** | Estado latente, adaptativo | Requiere especificación correcta |
| **Bayesian Structural** | Incertidumbre completa, priors | Computacionalmente intensivo |
