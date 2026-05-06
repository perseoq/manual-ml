# AP09 — Cheatsheet Series Temporales

## 1. Configuración y Librerías

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*1. Configuración y Librerías.*

1. `import numpy as np` — Importa las librerías necesarias para el análisis.
2. `import pandas as pd` — Importa las librerías necesarias para el análisis.
3. `import matplotlib.pyplot as plt` — Importa las librerías necesarias para el análisis.
4. `from statsmodels.tsa.stattools import adfuller, acf, pacf` — Importa las librerías necesarias para el análisis.
5. `from statsmodels.tsa.seasonal import seasonal_decompose` — Importa las librerías necesarias para el análisis.
6. `from statsmodels.tsa.arima.model import ARIMA` — Importa las librerías necesarias para el análisis.
7. `from statsmodels.tsa.holtwinters import ExponentialSmoothing` — Importa las librerías necesarias para el análisis.
8. `from statsmodels.graphics.tsaplots import plot_acf, plot_pacf` — Importa las librerías necesarias para el análisis.
9. `from sklearn.metrics import mean_absolute_error, mean_squared_error` — Importa las librerías necesarias para el análisis.
10. `import warnings` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 2. Creación de Series Temporales

```python
# Crear serie con pandas
fechas = pd.date_range("2024-01-01", periods=365, freq="D")
ventas = np.random.randn(365).cumsum() + 100
ts = pd.Series(ventas, index=fechas, name="ventas")

# DataFrame temporal
df = pd.DataFrame({
    "fecha": fechas,
    "ventas": ventas,
    "precio": np.random.uniform(50, 200, 365)
}).set_index("fecha")

# Frecuencias comunes
ts.asfreq("D")    # diaria
ts.asfreq("W")    # semanal
ts.asfreq("M")    # mensual
ts.asfreq("Q")    # trimestral
ts.asfreq("Y")    # anual

# Cambiar frecuencia (downsampling)
ts.resample("W").mean()   # semanal: promedio
ts.resample("M").sum()    # mensual: suma
ts.resample("Q").last()   # trimestral: último valor
ts.resample("Y").max()    # anual: máximo
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*2. Creación de Series Temporales.*

1. Crear serie con pandas
2. DataFrame temporal
3. Frecuencias comunes
4. Cambiar frecuencia (downsampling)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 3. Visualización

```python
# Plot básico
ts.plot(figsize=(12, 5), title="Ventas Diarias")
plt.ylabel("Ventas ($)")
plt.xlabel("Fecha")
plt.grid(True)
plt.show()

# ACF y PACF
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(ts.dropna(), lags=40, ax=axes[0])
plot_pacf(ts.dropna(), lags=40, ax=axes[1])
plt.show()

# Descomposición
decomp = seasonal_decompose(ts, model="additive", period=7)
decomp.plot()
plt.show()

# Componentes individuales
trend = decomp.trend
seasonal = decomp.seasonal
residual = decomp.resid
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*3. Visualización.*

1. Plot básico
2. ACF y PACF
3. Descomposición
4. Componentes individuales

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 4. Estacionariedad

```python
# Dickey-Fuller test
def test_estacionariedad(series):
    result = adfuller(series.dropna(), autolag="AIC")
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print(f"Lags: {result[2]}")
    print(f"Nobs: {result[3]}")
    print(f"Critical Values: {result[4]}")
    if result[1] <= 0.05:
        print("=> Serie ESTACIONARIA (rechazamos H0)")
    else:
        print("=> Serie NO ESTACIONARIA (no rechazamos H0)")

test_estacionariedad(ts)

# Hacer estacionaria
ts_diff = ts.diff().dropna()         # primera diferencia
ts_diff2 = ts.diff().diff().dropna()  # segunda diferencia
ts_log = np.log(ts).diff().dropna()   # log + diferencia

# Transformaciones adicionales
ts_seasonal_diff = ts.diff(7).dropna()  # diferencia estacional (semanal)
ts_log_seasonal = np.log(ts).diff(7).dropna()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*4. Estacionariedad.*

1. Dickey-Fuller test
2. Hacer estacionaria
3. Transformaciones adicionales

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 5. ARIMA

```python
from statsmodels.tsa.arima.model import ARIMA

# ARIMA(p,d,q)
# p = orden autorregresivo, d = diferencias, q = orden media móvil

# Identificar órdenes con ACF/PACF
# AR(p): PACF corta después de lag p
# MA(q): ACF corta después de lag q

# ARIMA(1,1,1)
model = ARIMA(ts, order=(1, 1, 1))
result = model.fit()
print(result.summary())

# Diagnóstico de residuos
residuals = result.resid
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(residuals, lags=30, ax=axes[0])
residuals.plot(ax=axes[1])
plt.show()

# Predicción
forecast = result.forecast(steps=30)
forecast = result.get_forecast(steps=30)
pred_df = forecast.conf_int(alpha=0.05)
pred_df["prediction"] = forecast.predicted_mean
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*5. ARIMA.*

1. ARIMA(p,d,q)
2. p = orden autorregresivo, d = diferencias, q = orden media móvil
3. Identificar órdenes con ACF/PACF
4. AR(p): PACF corta después de lag p
5. MA(q): ACF corta después de lag q
6. ARIMA(1,1,1)
7. Diagnóstico de residuos
8. Predicción

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 6. SARIMA (Estacional)

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# SARIMA(p,d,q)(P,D,Q,s)
# P,D,Q = orden estacional, s = período estacional

model = SARIMAX(
    ts,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),  # estacionalidad semanal
    enforce_stationarity=False,
    enforce_invertibility=False
)
result = model.fit(disp=False)
print(result.summary())

# Predicción con SARIMA
forecast = result.get_forecast(steps=30)
pred_mean = forecast.predicted_mean
pred_ci = forecast.conf_int()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*6. SARIMA (Estacional).*

1. SARIMA(p,d,q)(P,D,Q,s)
2. P,D,Q = orden estacional, s = período estacional
3. Predicción con SARIMA

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 7. Prophet (Facebook/Meta)

```python
from prophet import Prophet

# Preparar datos (columnas ds y y)
df_prophet = ts.reset_index()
df_prophet.columns = ["ds", "y"]

# Modelo
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode="additive",
    changepoint_prior_scale=0.05
)
model.fit(df_prophet)

# Futuro
future = model.make_future_dataframe(periods=60)
forecast = model.predict(future)

# Componentes
model.plot(forecast)
model.plot_components(forecast)
plt.show()

# Cambiar estacionalidad a multiplicativa
model = Prophet(seasonality_mode="multiplicative")

# Añadir regresores externos
df_prophet["precio"] = df["precio"].values
model.add_regressor("precio")
model.fit(df_prophet)

# Custom seasonality
model.add_seasonality(name="mensual", period=30.5, fourier_order=5)
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*7. Prophet (Facebook/Meta).*

1. Preparar datos (columnas ds y y)
2. Modelo
3. Futuro
4. Componentes
5. Cambiar estacionalidad a multiplicativa
6. Añadir regresores externos
7. Custom seasonality

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 8. LSTM para Series Temporales

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Preparar datos supervisados
def crear_secuencias(data, seq_length=30):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Escalar
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(ts.values.reshape(-1, 1))

# Crear secuencias
seq_length = 30
X, y = crear_secuencias(data_scaled, seq_length)

# Train/test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Modelo LSTM
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")
model.summary()

# Entrenar
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=0
)

# Predecir
y_pred_scaled = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred_scaled)
y_test_actual = scaler.inverse_transform(y_test)

# Multi-step forecast
def predict_future(model, last_sequence, n_steps, scaler):
    predictions = []
    current_seq = last_sequence.copy()
    for _ in range(n_steps):
        pred = model.predict(current_seq.reshape(1, -1, 1), verbose=0)
        predictions.append(pred[0, 0])
        current_seq = np.append(current_seq[1:], pred)
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

last_seq = data_scaled[-seq_length:]
future_preds = predict_future(model, last_seq, 30, scaler)
```

**Salida:**

```
# El modelo ajusta sus parámetros durante el entrenamiento.
# Los parámetros aprendidos quedan disponibles en el objeto.
```

**Explicación línea por línea:**

*8. LSTM para Series Temporales.*

1. Preparar datos supervisados
2. Escalar
3. Crear secuencias
4. Train/test split
5. Modelo LSTM
6. Entrenar
7. Predecir
8. Multi-step forecast

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 9. Métricas de Forecast

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score

def metricas_forecast(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "R²": r2
    }

print(metricas_forecast(y_test_actual, y_pred))

# sMAPE (Simetric Mean Absolute Percentage Error)
def smape(y_true, y_pred):
    return 100 * np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))
    )

# MASE (Mean Absolute Scaled Error)
def mase(y_true, y_pred, y_train, season=1):
    naive_error = np.mean(np.abs(y_train[season:] - y_train[:-season]))
    return np.mean(np.abs(y_true - y_pred)) / naive_error
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*9. Métricas de Forecast.*

1. sMAPE (Simetric Mean Absolute Percentage Error)
2. MASE (Mean Absolute Scaled Error)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 10. Walk-Forward Validation

```python
def walk_forward_validation(ts, model_fn, train_window, horizon, step=1):
    predictions = []
    actuals = []
    n = len(ts)

    for i in range(0, n - train_window - horizon + 1, step):
        train = ts.iloc[i:i+train_window]
        test = ts.iloc[i+train_window:i+train_window+horizon]

        model = model_fn(train)
        pred = model.forecast(steps=horizon)

        predictions.extend(pred)
        actuals.extend(test.values)

    return np.array(predictions), np.array(actuals)

# Ejemplo con ARIMA
def arima_model(train):
    model = ARIMA(train, order=(1, 1, 1))
    return model.fit()

preds, actuals = walk_forward_validation(ts, arima_model,
                                         train_window=200,
                                         horizon=7, step=7)
print(metricas_forecast(actuals, preds))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*10. Walk-Forward Validation.*

1. Ejemplo con ARIMA

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 11. Exponential Smoothing

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing

# Simple Exponential Smoothing (SES)
ses = SimpleExpSmoothing(ts).fit(smoothing_level=0.3, optimized=False)
ses_pred = ses.forecast(30)

# Holt's Linear Trend
holt = ExponentialSmoothing(
    ts, trend="additive", damped_trend=True
).fit()
holt_pred = holt.forecast(30)

# Holt-Winters (con estacionalidad)
hw = ExponentialSmoothing(
    ts,
    trend="add",
    seasonal="add",
    seasonal_periods=7
).fit()
hw_pred = hw.forecast(30)

# Modelo multiplicativo
hw_mul = ExponentialSmoothing(
    ts, trend="mul", seasonal="mul", seasonal_periods=7
).fit()
```

**Salida:**

```
# El modelo ajusta sus parámetros durante el entrenamiento.
# Los parámetros aprendidos quedan disponibles en el objeto.
```

**Explicación línea por línea:**

*11. Exponential Smoothing.*

1. Simple Exponential Smoothing (SES)
2. Holt's Linear Trend
3. Holt-Winters (con estacionalidad)
4. Modelo multiplicativo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 12. Detección de Anomalías

```python
from scipy import stats

# Método de desviación estándar
def detectar_anomalias_zscore(series, threshold=3):
    z_scores = np.abs(stats.zscore(series.dropna()))
    return series[z_scores > threshold]

# Método IQR
def detectar_anomalias_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return series[(series < lower) | (series > upper)]

# Residual-based (después de descomposición)
decomp = seasonal_decompose(ts.dropna(), model="additive", period=7)
residuals = decomp.resid
anomalies = residuals[np.abs(residuals) > 2 * residuals.std()]
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*12. Detección de Anomalías.*

1. Método de desviación estándar
2. Método IQR
3. Residual-based (después de descomposición)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 13. Feature Engineering Temporal

```python
def crear_features_temporales(df, fecha_col="fecha"):
    df = df.copy()
    df["año"] = df[fecha_col].dt.year
    df["mes"] = df[fecha_col].dt.month
    df["dia"] = df[fecha_col].dt.day
    df["dia_semana"] = df[fecha_col].dt.dayofweek
    df["fin_semana"] = df["dia_semana"].isin([5, 6]).astype(int)
    df["trimestre"] = df[fecha_col].dt.quarter
    df["dia_año"] = df[fecha_col].dt.dayofyear
    df["semana_año"] = df[fecha_col].dt.isocalendar().week.astype(int)
    return df

# Lags y ventanas
df["lag_1"] = df["ventas"].shift(1)
df["lag_7"] = df["ventas"].shift(7)
df["lag_30"] = df["ventas"].shift(30)
df["rolling_mean_7"] = df["ventas"].rolling(7).mean()
df["rolling_std_7"] = df["ventas"].rolling(7).std()
df["rolling_max_30"] = df["ventas"].rolling(30).max()
df["rolling_min_7"] = df["ventas"].rolling(7).min()
df["ewm_alpha_03"] = df["ventas"].ewm(alpha=0.3).mean()
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*13. Feature Engineering Temporal.*

1. Lags y ventanas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 14. Cambio de Punto de Cambio (Changepoint Detection)

```python
# Prophet ya detecta changepoints
model = Prophet(changepoint_prior_scale=0.05)
model.fit(df_prophet)
print(model.changepoints)

# Ruptura estructural con Chow Test (usando statsmodels)
from statsmodels.tsa.stattools import breakvar
breakvar_result = breakvar(ts.dropna())
print(f"p-value: {breakvar_result[1]:.4f}")

# Detección manual
def detectar_cambios_rolling(series, window=30):
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    changes = np.abs(rolling_mean.diff()) > 2 * rolling_std
    return series[changes]
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*14. Cambio de Punto de Cambio (Changepoint Detection).*

1. Prophet ya detecta changepoints
2. Ruptura estructural con Chow Test (usando statsmodels)
3. Detección manual

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 15. Modelos Avanzados

```python
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

# Feature-based forecasting (crear features primero)
def prepare_supervised(df, target_col, lags=[1, 7, 30], windows=[7, 30]):
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    for w in windows:
        df[f"rolling_mean_{w}"] = df[target_col].rolling(w).mean()
        df[f"rolling_std_{w}"] = df[target_col].rolling(w).std()
    return df.dropna()

df_feats = prepare_supervised(df, "ventas")
cols = [c for c in df_feats.columns if c != "ventas"]
X, y = df_feats[cols], df_feats["ventas"]

split = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# LightGBM
lgbm = LGBMRegressor(n_estimators=100, learning_rate=0.1)
lgbm.fit(X_train, y_train)
lgbm_pred = lgbm.predict(X_test)

# Importancia de features
importances = pd.DataFrame({
    "feature": cols,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)
```

**Salida:**

```
# El modelo ajusta sus parámetros durante el entrenamiento.
# Los parámetros aprendidos quedan disponibles en el objeto.
```

**Explicación línea por línea:**

*15. Modelos Avanzados.*

1. Feature-based forecasting (crear features primero)
2. Random Forest
3. LightGBM
4. Importancia de features

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## Referencia Rápida

| Tarea | Función/Librería | Descripción |
|-------|------------------|-------------|
| Crear | `pd.date_range()`, `Series(index=...)` | Serie temporal |
| Visualizar | `ts.plot()`, `plot_acf()`, `plot_pacf()` | Gráficos |
| Descomponer | `seasonal_decompose(ts, period=7)` | Tendencia + estacionalidad + residuo |
| Estacionariedad | `adfuller(ts)` | Dickey-Fuller test |
| ARIMA | `ARIMA(ts, order=(p,d,q))` | Modelo ARIMA |
| SARIMA | `SARIMAX(ts, order=..., seasonal_order=...)` | Con estacionalidad |
| Prophet | `Prophet().fit(df).predict(future)` | Forecast automático |
| LSTM | `keras.Sequential([LSTM(50), Dense(1)])` | Deep learning |
| Métricas | `mean_absolute_error()`, `mean_squared_error()` | MAE, MSE, RMSE, MAPE |
| Walk-forward | Loop manual con ventana móvil | Validación temporal |
| Smoothing | `ExponentialSmoothing(ts, trend="add", seasonal="add")` | Holt-Winters |
| Anomalías | `zscore()`, `IQR()` | Detección |
