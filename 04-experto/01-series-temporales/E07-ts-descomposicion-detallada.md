# E07: Descomposición de Series Temporales para Pronóstico de Ventas

## 1. Fundamentos Teóricos

### 1.1 ¿Por qué descomponer?

La descomposición de series temporales separa una serie en sus componentes fundamentales:

- **Tendencia (T_t)**: movimiento de largo plazo (crecimiento/declive en ventas)
- **Estacionalidad (S_t)**: patrones periódicos (semanal, mensual, anual)
- **Residuo (R_t)**: componente irregular/ruido (innovaciones, shocks)

Modelos:
- **Aditivo**: Y_t = T_t + S_t + R_t (amplitud estacional constante)
- **Multiplicativo**: Y_t = T_t × S_t × R_t (amplitud estacional proporcional a tendencia)

### 1.2 Métodos de descomposición

| Método | Ventajas | Desventajas |
|--------|----------|-------------|
| `seasonal_decompose` | Simple, rápida | Estacionalidad fija, sensible a outliers |
| `STL` | Robusta a outliers, flexible | Requiere ajuste de parámetros |
| `MSTL` | Múltiples estacionalidades | Computacionalmente intensiva |
| `X13-ARIMA-SEATS` | Estándar gubernamental, ajuste calendario | Propietaria/compleja |

### 1.3 Métricas de fuerza

- **Fuerza de tendencia**: F_t = max(0, 1 - Var(R_t)/Var(T_t + R_t))
- **Fuerza estacional**: F_s = max(0, 1 - Var(R_t)/Var(S_t + R_t))

---

## 2. Ejemplos Prácticos

Usaremos ventas simuladas de una cadena de retail con 3 años de datos diarios.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.stattools import acf, adfuller
from scipy import stats
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

np.random.seed(42)

# Generar datos simulados: 3 años de ventas diarias
dates = pd.date_range('2022-01-01', '2024-12-31', freq='D')
n = len(dates)

# Tendencia: crecimiento lineal + plateau
tendencia = 100 + np.linspace(0, 50, n) + 5 * np.sin(np.linspace(0, 2*np.pi*0.5, n))

# Estacionalidad semanal: picos en fin de semana
est_semanal = 20 * np.sin(2 * np.pi * np.arange(n) / 7)

# Estacionalidad anual: pico en diciembre
est_anual = 30 * np.sin(2 * np.pi * np.arange(n) / 365.25 - np.pi/2)
est_anual = np.where(est_anual < 0, est_anual * 0.3, est_anual)

# Ruido
ruido = np.random.normal(0, 10, n)

# Outliers ocasionales
outliers = np.zeros(n)
outliers[np.random.randint(0, n, 20)] = np.random.choice([-50, 50], 20)

ventas = tendencia + est_semanal + est_anual + ruido + outliers

df = pd.DataFrame({
    'fecha': dates,
    'ventas': ventas,
    'dia_semana': dates.dayofweek,
    'mes': dates.month,
    'precio': 50 + 5 * np.sin(2 * np.pi * np.arange(n) / 180) + np.random.normal(0, 2, n)
})
df.set_index('fecha', inplace=True)

print(f"Datos generados: {len(df)} días")
print(f"Rango: {df.index.min()} a {df.index.max()}")
print(f"Ventas: media={df['ventas'].mean():.1f}, std={df['ventas'].std():.1f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*2. Ejemplos Prácticos.*

1. Generar datos simulados: 3 años de ventas diarias
2. Tendencia: crecimiento lineal + plateau
3. Estacionalidad semanal: picos en fin de semana
4. Estacionalidad anual: pico en diciembre
5. Ruido
6. Outliers ocasionales

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 1: seasonal_decompose básico (periodo semanal)

```python
from statsmodels.tsa.seasonal import seasonal_decompose

descomposicion = seasonal_decompose(
    df['ventas'],
    model='additive',
    period=7,  # estacionalidad semanal
    extrapolate_trend=0,  # no extrapolar tendencia en bordes
    two_sided=True  # filtro simétrico
)

fig, axes = plt.subplots(4, 1, figsize=(14, 10))
descomposicion.observed.plot(ax=axes[0], title='Observado (Ventas Diarias)')
descomposicion.trend.plot(ax=axes[1], title='Tendencia', color='red')
descomposicion.seasonal.plot(ax=axes[2], title='Estacionalidad Semanal', color='green')
descomposicion.resid.plot(ax=axes[3], title='Residuo', color='purple')
plt.tight_layout()
plt.savefig('e07_ej1_descompose_semanal.png', dpi=100)
plt.close()

print("=== Descomposición Aditiva Semanal ===")
print(f"Tendencia: {descomposicion.trend.min():.1f} - {descomposicion.trend.max():.1f}")
print(f"Estacionalidad rango: {descomposicion.seasonal.min():.1f} a {descomposicion.seasonal.max():.1f}")
print(f"Residuo std: {descomposicion.resid.std():.1f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: seasonal_decompose básico (periodo semanal).*

1. `from statsmodels.tsa.seasonal import seasonal_decompose` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: seasonal_decompose multiplicativo (periodo anual)

```python
# Simular datos con estacionalidad multiplicativa (amplitud crece con tendencia)
ventas_mult = 100 + np.linspace(0, 100, n)  # tendencia fuerte
est_mult = 1 + 0.3 * np.sin(2 * np.pi * np.arange(n) / 365.25)  # estacionalidad relativa
ventas_mult = ventas_mult * est_mult + np.random.normal(0, 5, n)

descomposicion_mult = seasonal_decompose(
    ventas_mult,
    model='multiplicative',
    period=365,
    extrapolate_trend=12  # extrapolar 12 pasos en bordes
)

print("=== Descomposición Multiplicativa Anual ===")
print(f"Componente estacional en dic: {descomposicion_mult.seasonal.iloc[350:360].mean():.4f}")
print(f"Componente estacional en jul: {descomposicion_mult.seasonal.ilient[180:190].mean():.4f}")

fig, axes = plt.subplots(4, 1, figsize=(14, 10))
descomposicion_mult.observed.plot(ax=axes[0], title='Observado (Multiplicativo)')
descomposicion_mult.trend.plot(ax=axes[1], title='Tendencia', color='red')
descomposicion_mult.seasonal.plot(ax=axes[2], title='Estacionalidad Anual', color='green')
descomposicion_mult.resid.plot(ax=axes[3], title='Residuo', color='purple')
plt.tight_layout()
plt.savefig('e07_ej2_descompose_mult.png', dpi=100)
plt.close()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: seasonal_decompose multiplicativo (periodo anual).*

1. Simular datos con estacionalidad multiplicativa (amplitud crece con tendencia)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: STL con LOESS (periodo=7)

```python
stl = STL(
    df['ventas'],
    period=7,
    seasonal=7,  # longitud ventana estacional
    robust=False,
    seasonal_deg=1,
    trend_deg=1,
    low_pass_deg=1,
    seasonal_jump=1,
    trend_jump=1,
    low_pass_jump=1
)
resultado_stl = stl.fit()

fig, axes = plt.subplots(4, 1, figsize=(14, 10))
resultado_stl.observed.plot(ax=axes[0], title='STL - Observado')
resultado_stl.trend.plot(ax=axes[1], title='STL - Tendencia', color='red')
resultado_stl.seasonal.plot(ax=axes[2], title='STL - Estacionalidad', color='green')
resultado_stl.resid.plot(ax=axes[3], title='STL - Residuo', color='purple')
plt.tight_layout()
plt.savefig('e07_ej3_stl.png', dpi=100)
plt.close()

print("=== STL (LOESS) Semanal ===")
print(f"Peso relativo tendencia: {resultado_stl.trend.var()/df['ventas'].var():.2%}")
print(f"Peso relativo estacionalidad: {resultado_stl.seasonal.var()/df['ventas'].var():.2%}")
print(f"Peso relativo residuo: {resultado_stl.resid.var()/df['ventas'].var():.2%}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: STL con LOESS (periodo=7).*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: STL robusto para datos con outliers

```python
stl_robust = STL(
    df['ventas'],
    period=7,
    seasonal=7,
    robust=True,  # downweight outliers automáticamente
    seasonal_deg=1,
    trend_deg=1,
    low_pass_deg=1
)
resultado_robust = stl_robust.fit()

# Comparar con no robusto
stl_norobust = STL(df['ventas'], period=7, seasonal=7, robust=False).fit()

print("=== Comparación STL Robusto vs No Robusto ===")
print(f"Robusto - Residuo std: {resultado_robust.resid.std():.2f}")
print(f"No robusto - Residuo std: {stl_norobust.resid.std():.2f}")
print(f"Robusto - Tendencia suavidad: {resultado_robust.trend.diff().std():.4f}")
print(f"No robusto - Tendencia suavidad: {stl_norobust.trend.diff().std():.4f}")

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df.index, resultado_robust.trend, label='STL Robusto', linewidth=2)
ax.plot(df.index, stl_norobust.trend, label='STL No Robusto', alpha=0.7, linestyle='--')
ax.set_title('Comparación: Tendencia STL Robusto vs No Robusto')
ax.legend()
plt.tight_layout()
plt.savefig('e07_ej4_stl_robust.png', dpi=100)
plt.close()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: STL robusto para datos con outliers.*

1. Comparar con no robusto

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: MSTL - Múltiples estacionalidades

```python
from statsmodels.tsa.seasonal import MSTL

# Verificar que tenemos suficiente historia para estacionalidad anual
print(f"Datos disponibles: {len(df)} días ≈ {len(df)/365.25:.1f} años")

mstl = MSTL(
    df['ventas'],
    periods=[7, 365],  # semanal + anual
    stl_kwargs={
        'seasonal_deg': 1,
        'trend_deg': 1,
        'low_pass_deg': 1,
        'robust': True
    }
)
resultado_mstl = mstl.fit()

print("=== MSTL: Múltiples Estacionalidades ===")
print(f"Componentes: {resultado_mstl.seasonal.keys()}")
for comp, values in resultado_mstl.seasonal.items():
    print(f"  {comp}: var={values.var():.2f} ({values.var()/df['ventas'].var():.1%})")
print(f"Tendencia var: {resultado_mstl.trend.var():.2f}")
print(f"Residuo var: {resultado_mstl.resid.var():.2f}")

fig, axes = plt.subplots(5, 1, figsize=(14, 12))
resultado_mstl.observed.plot(ax=axes[0], title='Observado')
resultado_mstl.trend.plot(ax=axes[1], title='Tendencia', color='red')
for i, comp in enumerate(resultado_mstl.seasonal.keys(), start=2):
    resultado_mstl.seasonal[comp].plot(ax=axes[i], title=f'Estacionalidad {comp}', color='green')
resultado_mstl.resid.plot(ax=axes[4], title='Residuo', color='purple')
plt.tight_layout()
plt.savefig('e07_ej5_mstl.png', dpi=100)
plt.close()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: MSTL - Múltiples estacionalidades.*

1. Verificar que tenemos suficiente historia para estacionalidad anual

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: Extraer tendencia con media móvil

```python
# Media móvil simple
window = 30  # 30 días
tendencia_ma = df['ventas'].rolling(window=window, center=True).mean()

# Media móvil centrada para periodo par
tendencia_ma_centrada = (
    df['ventas'].rolling(window=window, center=True).mean()
)

# Comparar con STL
stl_trend = STL(df['ventas'], period=7, seasonal=7).fit().trend

print("=== Extracción de Tendencia con Media Móvil ===")
print(f"Ventana: {window} días")
print(f"Correlación MA vs STL: {tendencia_ma.corr(stl_trend):.4f}")
print(f"MA - NaN iniciales/finales: {tendencia_ma.isna().sum()}")

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df.index[:300], df['ventas'].iloc[:300], alpha=0.3, label='Ventas diarias')
ax.plot(df.index[:300], tendencia_ma.iloc[:300], label=f'MA({window})', linewidth=2)
ax.plot(df.index[:300], stl_trend.iloc[:300], label='STL Tendencia', linewidth=2, linestyle='--')
ax.set_title('Comparación: Media Móvil vs STL para tendencia')
ax.legend()
plt.tight_layout()
plt.savefig('e07_ej6_tendencia_ma.png', dpi=100)
plt.close()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: Extraer tendencia con media móvil.*

1. Media móvil simple
2. Media móvil centrada para periodo par
3. Comparar con STL

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: Estacionalidad con variables dummy

```python
# Crear dummies por día de semana
dummies = pd.get_dummies(df['dia_semana'], prefix='dia')

# Modelo lineal para estimar estacionalidad
import statsmodels.api as sm

X = dummies.values.astype(float)
y = df['ventas'].values - tendencia_ma.values  # ventas des-tendenciadas

# Manejar NaN
mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
y_clean = y[mask]
X_clean = X[mask]

X_clean = sm.add_constant(X_clean)
modelo_dummy = sm.OLS(y_clean, X_clean).fit()

print("=== Efectos Estacionales por Día (Dummies) ===")
for i, coef in enumerate(modelo_dummy.params[1:]):
    print(f"  Día {i}: {coef:.1f} unidades vs promedio")
print(f"R² del modelo: {modelo_dummy.rsquared:.3f}")

# Estacionalidad estimada
est_dummies = pd.Series(index=df.index, dtype=float)
est_dummies.loc[mask] = X_clean @ modelo_dummy.params

fig, ax = plt.subplots(figsize=(14, 4))
est_dummies.iloc[:70].plot(ax=ax, marker='o', label='Estacionalidad dummies')
ax.axhline(0, color='gray', linestyle='--')
ax.set_title('Estacionalidad Semanal Estimada con Dummies')
ax.legend()
plt.tight_layout()
plt.savefig('e07_ej7_dummies.png', dpi=100)
plt.close()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: Estacionalidad con variables dummy.*

1. Crear dummies por día de semana
2. Modelo lineal para estimar estacionalidad
3. Manejar NaN
4. Estacionalidad estimada

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: Fourier terms para estacionalidad

```python
def fourier_terms(t, period, order):
    """Generar términos de Fourier para estacionalidad."""
    terms = []
    for k in range(1, order + 1):
        terms.append(np.sin(2 * np.pi * k * t / period))
        terms.append(np.cos(2 * np.pi * k * t / period))
    return np.column_stack(terms)

t = np.arange(n)

for order in [3, 5, 10]:
    X_fourier = fourier_terms(t, 365.25, order)
    print(f"Fourier order={order}: {X_fourier.shape[1]} términos generados")

# Probar diferentes órdenes
resultados_fourier = {}
for order in [3, 5, 10, 15]:
    X_fourier = fourier_terms(t, 365.25, order)
    mask = ~np.isnan(tendencia_ma.values)
    y_detrend = y[mask]
    X_use = X_fourier[mask]
    X_use = sm.add_constant(X_use)
    model = sm.OLS(y_detrend, X_use).fit()
    resultados_fourier[order] = model.rsquared_adj
    print(f"Order={order}: R²_adj={model.rsquared_adj:.3f}")

fig, ax = plt.subplots(figsize=(12, 4))
orders = list(resultados_fourier.keys())
rs = list(resultados_fourier.values())
ax.bar(orders, rs)
ax.set_xlabel('Orden Fourier')
ax.set_ylabel('R² ajustado')
ax.set_title('Calidad de ajuste estacional vs Orden Fourier')

# Visualizar estacionalidad con order=10
X_f10 = fourier_terms(t, 365.25, 10)
X_f10 = sm.add_constant(X_f10)
est_f10 = pd.Series(X_f10 @ sm.OLS(y_detrend, X_use).params, index=df.index)
plt.tight_layout()
plt.savefig('e07_ej8_fourier.png', dpi=100)
plt.close()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: Fourier terms para estacionalidad.*

1. Probar diferentes órdenes
2. Visualizar estacionalidad con order=10

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: Análisis de residuos (ruido blanco)

```python
residuos = resultado_stl.resid.dropna()

# Estadísticos descriptivos
print("=== Análisis de Residuos ===")
print(f"Media: {residuos.mean():.4f} (ideal: 0)")
print(f"Std: {residuos.std():.4f}")
print(f"Asimetría: {residuos.skew():.4f}")
print(f"Curtosis: {residuos.kurtosis():.4f} (normal: 0)")

# Test de normalidad
stat, p_normal = stats.normaltest(residuos)
print(f"Test normalidad (D'Agostino): estadístico={stat:.2f}, p-valor={p_normal:.4f}")

# Autocorrelación
acf_vals = acf(residuos, nlags=30)
print(f"ACF(1)={acf_vals[1]:.4f}")
print(f"ACF(7)={acf_vals[7]:.4f} (debe ser ~0 si estacionalidad bien capturada)")

# Ljung-Box test
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_test = acorr_ljungbox(residuos, lags=[7, 14, 21], return_df=True)
print("\nLjung-Box test:")
print(lb_test)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(residuos, bins=50, density=True, alpha=0.7)
x = np.linspace(residuos.min(), residuos.max(), 100)
axes[0].plot(x, stats.norm.pdf(x, residuos.mean(), residuos.std()), 'r-', lw=2)
axes[0].set_title('Histograma vs Normal')

axes[1].plot(acf_vals[1:], marker='o')
axes[1].axhline(0, color='gray')
axes[1].axhline(1.96/np.sqrt(len(residuos)), color='red', linestyle='--')
axes[1].axhline(-1.96/np.sqrt(len(residuos)), color='red', linestyle='--')
axes[1].set_title('ACF de Residuos')

stats.probplot(residuos, dist='norm', plot=axes[2])
axes[2].set_title('Q-Q Plot')

plt.tight_layout()
plt.savefig('e07_ej9_residuos.png', dpi=100)
plt.close()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: Análisis de residuos (ruido blanco).*

1. Estadísticos descriptivos
2. Test de normalidad
3. Autocorrelación
4. Ljung-Box test

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: Detección de cambios estructurales en tendencia

```python
from ruptures import Pelt, Binseg, Window
import ruptures as rpt

# Cambios simulados: agregar un quiebre en la tendencia
np.random.seed(123)
ventas_cambio = df['ventas'].copy()
ventas_cambio.iloc[500:] += 30  # cambio de nivel en día 500
ventas_cambio.iloc[800:] -= 20  # cambio de nivel en día 800

# Detectar puntos de cambio con PELT
algo = rpt.Pelt(model="rbf").fit(ventas_cambio.values.reshape(-1, 1))
cambios = algo.predict(pen=10)

print("=== Detección de Cambios Estructurales ===")
print(f"Puntos de cambio detectados (PELT): {cambios[:-1]}")  # último es n

for cp in cambios[:-1]:
    fecha = df.index[cp]
    print(f"  Cambio en día {cp}: {fecha.date()}")

# Visualizar
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df.index, ventas_cambio, label='Ventas con cambios')
for cp in cambios[:-1]:
    ax.axvline(x=df.index[cp], color='red', linestyle='--', alpha=0.7)
ax.set_title('Detección de Cambios Estructurales en Tendencia')
ax.legend()
plt.tight_layout()
plt.savefig('e07_ej10_cambios.png', dpi=100)
plt.close()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: Detección de cambios estructurales en tendencia.*

1. Cambios simulados: agregar un quiebre en la tendencia
2. Detectar puntos de cambio con PELT
3. Visualizar

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: Comparar modelo aditivo vs multiplicativo

```python
# Aplicar ambos modelos a la misma serie
des_add = seasonal_decompose(df['ventas'], model='additive', period=7)
des_mult = seasonal_decompose(
    df['ventas'] - df['ventas'].min() + 1,  # asegurar positividad
    model='multiplicative',
    period=7
)

print("=== Comparación Aditivo vs Multiplicativo ===")
print(f"Aditivo - Residuo std: {des_add.resid.std():.4f}")
print(f"Multiplicativo - Residuo std: {des_mult.resid.std():.4f}")

# El que tenga menor varianza en residuos es mejor
if des_add.resid.std() < des_mult.resid.std():
    print("→ Modelo aditivo es más apropiado para esta serie")
else:
    print("→ Modelo multiplicativo es más apropiado para esta serie")

# Visualizar diferencia
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
des_add.trend.plot(ax=axes[0, 0], title='Tendencia - Aditivo', color='red')
des_mult.trend.plot(ax=axes[0, 1], title='Tendencia - Multiplicativo', color='red')
des_add.seasonal.iloc[:70].plot(ax=axes[1, 0], title='Estacionalidad - Aditivo', color='green')
des_mult.seasonal.iloc[:70].plot(ax=axes[1, 1], title='Estacionalidad - Multiplicativo', color='green')
plt.tight_layout()
plt.savefig('e07_ej11_add_vs_mul.png', dpi=100)
plt.close()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: Comparar modelo aditivo vs multiplicativo.*

1. Aplicar ambos modelos a la misma serie
2. El que tenga menor varianza en residuos es mejor
3. Visualizar diferencia

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: Efecto de diferentes periodos (7 vs 14 vs 30)

```python
periodos = [7, 14, 30]
resultados_periodo = {}

for p in periodos:
    des = seasonal_decompose(df['ventas'], model='additive', period=p)
    resultados_periodo[p] = {
        'resid_std': des.resid.std(),
        'trend_var': des.trend.var(),
        'seasonal_var': des.seasonal.var()
    }
    print(f"Period={p:2d}: resid_std={des.resid.std():.2f}, "
          f"trend_var={des.trend.var():.2f}, seasonal_var={des.seasonal.var():.2f}")

fig, axes = plt.subplots(3, 1, figsize=(14, 10))
for i, p in enumerate(periodos):
    des = seasonal_decompose(df['ventas'], model='additive', period=p)
    des.trend.plot(ax=axes[i], label=f'Tendencia (period={p})')
    axes[i].legend()
plt.tight_layout()
plt.savefig('e07_ej12_periodos.png', dpi=100)
plt.close()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: Efecto de diferentes periodos (7 vs 14 vs 30).*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: Visualizar componentes en subplots

```python
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

# Observado en toda la serie
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(df.index, df['ventas'], color='royalblue', linewidth=1)
ax1.set_title('Serie Original: Ventas Diarias', fontsize=14)
ax1.set_ylabel('Ventas')

# Tendencia zoom
stl_full = STL(df['ventas'], period=7, seasonal=7).fit()
ax2 = fig.add_subplot(gs[1, :])
ax2.plot(df.index, stl_full.trend, color='crimson', linewidth=2)
ax2.set_title('Tendencia (STL)', fontsize=14)
ax2.set_ylabel('Ventas')

# Estacionalidad semanal (zoom 60 días)
ax3 = fig.add_subplot(gs[2, 0])
stl_full.seasonal.iloc[:60].plot(ax=ax3, color='forestgreen')
ax3.set_title('Estacionalidad Semanal (zoom)', fontsize=12)
ax3.set_ylabel('Efecto estacional')

# Estacionalidad anual (un año)
ax4 = fig.add_subplot(gs[2, 1])
stl_full.seasonal.iloc[:365].plot(ax=ax4, color='darkorange')
ax4.set_title('Estacionalidad Anual', fontsize=12)
ax4.set_ylabel('Efecto estacional')

# Residuo
ax5 = fig.add_subplot(gs[2, 2])
stl_full.resid.plot(ax=ax5, color='purple', alpha=0.7, marker='.', markersize=1)
ax5.set_title('Residuo', fontsize=12)
ax5.set_ylabel('Residuo')

# Distribución residuo
ax6 = fig.add_subplot(gs[3, 0])
stl_full.resid.hist(ax=ax6, bins=50, color='purple', alpha=0.7)
ax6.set_title('Distribución del Residuo', fontsize=12)

# ACF residuo
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(stl_full.resid.dropna(), lags=40, ax=ax6)
# actually put in ax7
ax7 = fig.add_subplot(gs[3, 1])
plot_acf(stl_full.resid.dropna(), lags=40, ax=ax7)
ax7.set_title('ACF Residuos', fontsize=12)

# Resumen
ax8 = fig.add_subplot(gs[3, 2])
ax8.axis('off')
stats_text = (
    f"Resumen Descomposición\n"
    f"----------------------\n"
    f"Tendencia: {stl_full.trend.mean():.0f} ± {stl_full.trend.std():.0f}\n"
    f"Estacional: {stl_full.seasonal.std():.1f}\n"
    f"Residuo σ: {stl_full.resid.std():.1f}\n"
    f"Residuo/Serie: {stl_full.resid.std()/df['ventas'].std():.1%}\n"
    f"Fuerza tendencia: {max(0, 1 - stl_full.resid.var()/(stl_full.trend.var() + stl_full.resid.var())):.2%}\n"
    f"Fuerza estacional: {max(0, 1 - stl_full.resid.var()/(stl_full.seasonal.var() + stl_full.resid.var())):.2%}"
)
ax8.text(0.1, 0.5, stats_text, fontsize=12, va='center',
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Análisis Completo de Descomposición de Series Temporales', fontsize=16, y=1.02)
plt.savefig('e07_ej13_subplots.png', dpi=100, bbox_inches='tight')
plt.close()
print("Gráfico completo de componentes generado")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: Visualizar componentes en subplots.*

1. Observado en toda la serie
2. Tendencia zoom
3. Estacionalidad semanal (zoom 60 días)
4. Estacionalidad anual (un año)
5. Residuo
6. Distribución residuo
7. ACF residuo
8. actually put in ax7

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: Fuerza de tendencia y estacionalidad

```python
def fuerza_componente(resid, componente):
    """Calcular fuerza de un componente: F = max(0, 1 - Var(resid)/Var(comp+resid))."""
    return max(0, 1 - np.var(resid) / np.var(componente + resid))

des = seasonal_decompose(df['ventas'], model='additive', period=7)
F_tendencia = fuerza_componente(des.resid.dropna(), des.trend.dropna())
F_estacional = fuerza_componente(des.resid.dropna(), des.seasonal.dropna())

print("=== Fuerza de Componentes ===")
print(f"Fuerza de tendencia: {F_tendencia:.2%}")
print(f"  → {'Muy fuerte' if F_tendencia > 0.8 else 'Moderada' if F_tendencia > 0.5 else 'Débil'}")
print(f"Fuerza estacional: {F_estacional:.2%}")
print(f"  → {'Muy fuerte' if F_estacional > 0.8 else 'Moderada' if F_estacional > 0.6 else 'Débil'}")

# Para diferentes periodos
print("\nFuerza estacional para diferentes periodos:")
for p in [7, 14, 30, 365]:
    des_p = seasonal_decompose(df['ventas'], model='additive', period=p)
    Fs = fuerza_componente(des_p.resid.dropna(), des_p.seasonal.dropna())
    print(f"  Period={p:3d}: Fuerza estacional = {Fs:.2%}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: Fuerza de tendencia y estacionalidad.*

1. Para diferentes periodos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: Detectar puntos de cambio (cambio de nivel)

```python
# Usar el detector de cambios de ruptures con Window-based search
algo_window = Window(model="l2", width=40).fit(ventas_cambio.values.reshape(-1, 1))
cambios_window = algo_window.predict(pen=15)

print("=== Puntos de Cambio Detectados (Window-based) ===")
print(f"Número de cambios: {len(cambios_window) - 1}")
for cp in cambios_window[:-1]:
    fecha = df.index[cp]
    valor = ventas_cambio.iloc[cp]
    cambio_pct = (ventas_cambio.iloc[cp] - ventas_cambio.iloc[max(0, cp-7)]) / ventas_cambio.iloc[max(0, cp-7)] * 100
    print(f"  Día {cp} ({fecha.date()}): nivel = {valor:.0f}, cambio = {cambio_pct:+.1f}%")

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df.index, ventas_cambio, label='Serie con cambios', color='steelblue')
for i, cp in enumerate(cambios_window[:-1]):
    color = ['red', 'orange', 'purple'][i % 3]
    ax.axvline(x=df.index[cp], color=color, linestyle='--', linewidth=2, alpha=0.8)
    ax.axvspan(df.index[max(0, cp-30)], df.index[min(n-1, cp+30)], 
               alpha=0.1, color=color)
ax.set_title('Detección de Cambios de Nivel en Ventas')
ax.set_ylabel('Ventas')
ax.legend()
plt.tight_layout()
plt.savefig('e07_ej15_puntos_cambio.png', dpi=100)
plt.close()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: Detectar puntos de cambio (cambio de nivel).*

1. Usar el detector de cambios de ruptures con Window-based search

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 16: Seasonal Adjustment (serie desestacionalizada)

```python
# Desestacionalizar: restar componente estacional
des_adj = seasonal_decompose(df['ventas'], model='additive', period=7)
serie_ajustada = df['ventas'] - des_adj.seasonal

# Alternativa: dividir (multiplicativo)
des_adj_mult = seasonal_decompose(
    df['ventas'] - df['ventas'].min() + 1,
    model='multiplicative',
    period=7
)
serie_ajustada_mult = df['ventas'] / des_adj_mult.seasonal

print("=== Serie Desestacionalizada ===")
print(f"Original std: {df['ventas'].std():.2f}")
print(f"Ajustada (add) std: {serie_ajustada.std():.2f}")
print(f"Reducción varianza: {(1 - serie_ajustada.var()/df['ventas'].var()):.1%}")

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df.index[:200], df['ventas'].iloc[:200], alpha=0.5, label='Original')
ax.plot(df.index[:200], serie_ajustada.iloc[:200], label='Desestacionalizada', linewidth=2)
ax.plot(df.index[:200], des_adj.trend.iloc[:200], label='Tendencia', linewidth=2, linestyle='--')
ax.set_title('Efecto de la Desestacionalización')
ax.legend()
plt.tight_layout()
plt.savefig('e07_ej16_seasonal_adj.png', dpi=100)
plt.close()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: Seasonal Adjustment (serie desestacionalizada).*

1. Desestacionalizar: restar componente estacional
2. Alternativa: dividir (multiplicativo)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 17: Comparar STL vs seasonal_decompose en calidad

```python
from sklearn.metrics import mean_squared_error

# Asumimos que la "verdadera" tendencia es conocida en datos simulados
verdadera_tendencia = tendencia[:n]
verdadera_estacional = est_semanal + est_anual

metodos = {
    'seasonal_decompose(period=7)': seasonal_decompose(df['ventas'], period=7, model='additive'),
    'seasonal_decompose(period=365)': seasonal_decompose(df['ventas'], period=365, model='additive'),
    'STL(period=7, seasonal=7)': STL(df['ventas'], period=7, seasonal=7).fit(),
    'STL(period=7, robust=True)': STL(df['ventas'], period=7, seasonal=7, robust=True).fit(),
}

print("=== Comparación de Métodos de Descomposición ===")
print(f"{'Método':<38} {'RMSE Tend':<12} {'RMSE Estac':<12} {'RMSE Resid':<12}")
print("-" * 74)

for nombre, res in metodos.items():
    rmse_trend = np.sqrt(mean_squared_error(verdadera_tendencia, res.trend))
    rmse_estac = np.sqrt(mean_squared_error(verdadera_estacional, res.seasonal))
    rmse_resid = np.sqrt(mean_squared_error(np.zeros(n), res.resid.fillna(0)))
    print(f"{nombre:<38} {rmse_trend:<12.4f} {rmse_estac:<12.4f} {rmse_resid:<12.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: Comparar STL vs seasonal_decompose en calidad.*

1. Asumimos que la "verdadera" tendencia es conocida en datos simulados

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 18: Integrador - Descomposición completa + Análisis

```python
print("=" * 70)
print("ANÁLISIS INTEGRAL DE DESCOMPOSICIÓN DE SERIE TEMPORAL")
print("=" * 70)

# 1. Preparación
print("\n[1] ESTADÍSTICAS DESCRIPTIVAS")
print(f"Dimensión: {df.shape}")
print(f"Rango: {df.index.min()} → {df.index.max()}")
print(f"Media ventas: {df['ventas'].mean():.2f}")
print(f"CV: {df['ventas'].std()/df['ventas'].mean():.2%}")

# 2. Estacionariedad
result_adf = adfuller(df['ventas'])
print(f"\n[2] ESTACIONARIEDAD (ADF)")
print(f"Estadístico: {result_adf[0]:.4f}, p-valor: {result_adf[1]:.4f}")
print(f"¿Estacionaria? {'Sí' if result_adf[1] < 0.05 else 'No'}")

# 3. Descomposición STL óptima
print("\n[3] DESCOMPOSICIÓN STL")
stl_final = STL(df['ventas'], period=7, seasonal=7, robust=True).fit()

print(f"Componente tendencia: {stl_final.trend.mean():.1f} ± {stl_final.trend.std():.1f}")
print(f"Componente estacional (rango): [{stl_final.seasonal.min():.1f}, {stl_final.seasonal.max():.1f}]")
print(f"Componente residual: media={stl_final.resid.mean():.2f}, std={stl_final.resid.std():.2f}")

# 4. Fuerza de componentes
F_t = fuerza_componente(stl_final.resid.dropna(), stl_final.trend.dropna())
F_s = fuerza_componente(stl_final.resid.dropna(), stl_final.seasonal.dropna())
print(f"\n[4] FUERZA DE COMPONENTES")
print(f"Fuerza tendencia: {F_t:.2%}")
print(f"Fuerza estacional: {F_s:.2%}")

# 5. Análisis de residuos
lb = acorr_ljungbox(stl_final.resid.dropna(), lags=[7, 14], return_df=True)
print(f"\n[5] DIAGNÓSTICO DE RESIDUOS")
print(f"Ljung-Box lag 7: p={lb.loc[7, 'lb_pvalue']:.4f}")
print(f"Ljung-Box lag 14: p={lb.loc[14, 'lb_pvalue']:.4f}")
print(f"¿Ruido blanco? {'Sí' if lb.loc[7, 'lb_pvalue'] > 0.05 else 'No'}")

# 6. Puntos de cambio
algo_final = Pelt(model="rbf").fit(stl_final.trend.dropna().values.reshape(-1, 1))
puntos_cambio = algo_final.predict(pen=5)
print(f"\n[6] PUNTOS DE CAMBIO EN TENDENCIA")
print(f"Número de puntos de cambio: {len(puntos_cambio) - 1}")
if len(puntos_cambio) > 1:
    for cp in puntos_cambio[:-1]:
        print(f"  → Día {cp}: {df.index[cp].date()}")

# 7. Serie ajustada
print("\n[7] SERIE DESESTACIONALIZADA")
print(f"Serie original σ: {df['ventas'].std():.2f}")
print(f"Serie ajustada σ: {(df['ventas'] - stl_final.seasonal).std():.2f}")

# 8. Recomendación
print(f"\n[8] RECOMENDACIÓN PARA PRONÓSTICO")
if F_t > 0.7 and F_s > 0.7:
    print("→ Serie con fuerte estructura: SARIMA o Prophet recomendado")
elif F_t > 0.7:
    print("→ Serie dominada por tendencia: ARIMA o ETS con tendencia")
elif F_s > 0.7:
    print("→ Serie dominada por estacionalidad: SARIMA o STLF")
else:
    print("→ Serie con mucho ruido: considerar modelos robustos o ensemble")

# Gráfico final integrador
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

stl_final.trend.plot(ax=axes[0, 0], color='crimson', linewidth=2)
axes[0, 0].set_title(f'Tendencia (fuerza: {F_t:.1%})')

stl_final.seasonal.iloc[:180].plot(ax=axes[0, 1], color='forestgreen')
axes[0, 1].set_title(f'Estacionalidad Semanal (fuerza: {F_s:.1%})')

stl_final.resid.plot(ax=axes[1, 0], color='purple', alpha=0.6)
axes[1, 0].set_title(f'Residuo (σ={stl_final.resid.std():.2f})')

(df['ventas'] - stl_final.seasonal).plot(ax=axes[1, 1], color='darkorange')
axes[1, 1].set_title('Serie Desestacionalizada')

plt.suptitle('Análisis Integrador - Descomposición de Ventas', fontsize=14)
plt.tight_layout()
plt.savefig('e07_ej18_integrador.png', dpi=100, bbox_inches='tight')
plt.close()
print("\nGráfico integrador guardado")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador - Descomposición completa + Análisis.*

1. 1. Preparación
2. 2. Estacionariedad
3. 3. Descomposición STL óptima
4. 4. Fuerza de componentes
5. 5. Análisis de residuos
6. 6. Puntos de cambio
7. 7. Serie ajustada
8. 8. Recomendación

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 3. Ejercicios

1. **Descomposición con periodo incorrecto**: Aplica `seasonal_decompose` con period=6 a la serie semanal. ¿Cómo cambian los residuos y la estacionalidad? Explica por qué.

2. **STL param tuning**: Varía `seasonal` (ventana LOESS) entre 3, 7, 15 en STL. ¿Cómo afecta la suavidad de la estacionalidad? Mide el RMSE del residuo.

3. **MSTL con tres periodos**: Agrega un periodo de 30 días (mensual) además de 7 y 365. ¿Cuánta varianza explica el nuevo componente? Compara con MSTL de 2 periodos.

4. **Detección de outliers en residuos**: Identifica los 10 residuos más extremos del STL. ¿Corresponden a los outliers que simulamos inicialmente? Mapea fechas.

5. **Comparación Fourier vs Dummies**: Para la estacionalidad semanal, compara el R² de dummies (7 vars) vs Fourier order=3 (6 vars). ¿Cuál prefieres y por qué?

6. **Validación cruzada temporal**: Divide la serie en train (2022-2023) y test (2024). Descompón solo con train y reconstruye test. Mide MAE del residuo en test.

7. **Cambio estructural conocido**: Inserta un cambio de tendencia artificial en día 400 (suma 50). ¿Lo detecta PELT? ¿Con qué penalización (pen)?

8. **Pipeline automático**: Escribe una función que reciba una serie, determine automáticamente si usar modelo add/mult (basado en varianza estacional vs nivel), ejecute STL y devuelva la serie ajustada + diagnóstico.

---

## 4. Resumen

| Concepto | Takeaways |
|----------|-----------|
| **seasonal_decompose** | Rápido, estacionalidad fija, sensible a outliers |
| **STL** | Flexible, robusto, mejor para pronóstico profesional |
| **MSTL** | Útil cuando hay múltiples periodicidades (semanal + anual) |
| **Fourier vs Dummies** | Fourier: más compacto para largos periodos; Dummies: interpretable |
| **Fuerza de componentes** | Métrica objetiva para decidir qué modelar |
| **Análisis de residuos** | Residuos deben ser ruido blanco (ACF ~ 0, normalidad) |
| **Cambios estructurales** | Detectar con PELT/Window; crítico para validar tendencia |
| **Serie ajustada** | Desestacionalizar mejora precisión de modelos predictivos |
