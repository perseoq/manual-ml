# E22: Monitoreo de Modelos — Data Drift, Concept Drift y Alertas en Producción

## Objetivo
Implementar monitoreo continuo de modelos ML en producción para detectar data drift, concept drift y degradación, con alertas y re-entrenamiento automático en sistemas de ventas/compras/inventarios.

---

## 1. Fundamentos Teóricos

### 1.1 Data Drift
Cambio en la distribución de las features de entrada entre training y producción.
- **Causas**: Cambios estacionales, nuevos productos, cambios de precios, nuevas temporadas
- **Consecuencias**: Predicciones inexactas, degradación del modelo

### 1.2 Concept Drift
Cambio en la relación entre features (X) y target (y).
- **Causas**: Cambios en comportamiento del consumidor, nuevas tendencias de mercado
- **Tipos**: Sudden, gradual, incremental, recurring

### 1.3 Tests Estadísticos para Data Drift
| Test | Tipo de feature | Interpretación |
|---|---|---|
| **Kolmogorov-Smirnov (KS)** | Numérica | Diferencia entre distribuciones acumuladas |
| **Chi-Squared** | Categórica | Diferencia en frecuencias esperadas vs observadas |
| **Jensen-Shannon Divergence** | Ambas | Divergencia simétrica entre distribuciones |
| **Wasserstein Distance** | Numérica | Distancia entre distribuciones (Earth Mover) |
| **Population Stability Index** | Numérica | PSI = Σ(p_i - q_i) * ln(p_i/q_i) |

### 1.4 Population Stability Index (PSI)
```
PSI = Σ(p_i - q_i) * ln(p_i / q_i)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*1.4 Population Stability Index (PSI).*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---


- PSI < 0.1: Sin drift significativo
- 0.1 ≤ PSI < 0.2: Drift moderado
- PSI ≥ 0.2: Drift severo

### 1.5 Concept Drift Detection
| Algoritmo | Descripción |
|---|---|
| **DDM** | Detecta aumento significativo en tasa de error |
| **EDDM** | Versión mejorada de DDM para cambios graduales |
| **Page-Hinkley** | Detecta cambios en media de una señal |
| **ADWIN** | Ventana adaptativa que detecta cambios en distribución |

### 1.6 Evidently AI
```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=train_df, current_data=prod_df)
report.save_html('drift_report.html')
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*1.6 Evidently AI.*

1. `from evidently.report import Report` — Importa las librerías necesarias para el análisis.
2. `from evidently.metric_preset import DataDriftPreset, RegressionPreset` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### 1.7 Alertas
- **Email**: smtplib para notificaciones por correo
- **Slack**: Webhook URL para mensajes a canal
- **Webhook**: POST a URL arbitraria
- **PagerDuty/OpsGenie**: Para incidentes críticos

### 1.8 Re-entrenamiento Automático
Trigger cuando:
- drift > threshold en N features consecutivos
- accuracy degradation > threshold
- tiempo transcurrido desde último entrenamiento > T días

---

## 2. Ejemplos Prácticos

### Ejemplo 1: Data drift — Comparar distribución de precio en training vs producción

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

# Datos de entrenamiento (referencia)
np.random.seed(42)
train_precio = np.random.lognormal(mean=4.5, sigma=0.5, size=10000)

# Datos de producción (actual)
prod_precio = np.random.lognormal(mean=4.7, sigma=0.6, size=1000)

# KS Test
stat, p_value = ks_2samp(train_precio, prod_precio)
print(f"KS statistic: {stat:.4f}")
print(f"P-value: {p_value:.6f}")
print(f"Drift detectado: {'SÍ' if p_value < 0.05 else 'NO'}")

# Visualización
plt.figure(figsize=(10, 5))
plt.hist(train_precio, bins=50, alpha=0.5, label='Training', density=True)
plt.hist(prod_precio, bins=50, alpha=0.5, label='Producción', density=True)
plt.xlabel('Precio'); plt.ylabel('Densidad')
plt.title(f'Distribución de Precio (KS={stat:.3f}, p={p_value:.4f})')
plt.legend(); plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: Data drift — Comparar distribución de precio en training vs producción.*

1. Datos de entrenamiento (referencia)
2. Datos de producción (actual)
3. KS Test
4. Visualización

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: KS test — Estadístico y p-valor para cada feature

```python
def calcular_drift_features(train_df, prod_df):
    resultados = []
    for col in train_df.select_dtypes(include=[np.number]).columns:
        if col in prod_df.columns:
            stat, p_value = ks_2samp(train_df[col].dropna(), prod_df[col].dropna())
            resultados.append({
                'feature': col,
                'ks_stat': round(stat, 4),
                'p_value': round(p_value, 6),
                'drift': 'SÍ' if p_value < 0.05 else 'NO'
            })
    return pd.DataFrame(resultados).sort_values('ks_stat', ascending=False)

# Simular features de ventas
train_df = pd.DataFrame({
    'precio': np.random.lognormal(4.5, 0.5, 5000),
    'descuento': np.random.uniform(0, 0.3, 5000),
    'inventario': np.random.poisson(200, 5000),
    'rating': np.random.uniform(1, 5, 5000)
})
prod_df = train_df.copy()
prod_df['precio'] = np.random.lognormal(4.8, 0.7, 5000)  # Drift inducido

resultados = calcular_drift_features(train_df, prod_df)
print(resultados.to_string(index=False))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: KS test — Estadístico y p-valor para cada feature.*

1. Simular features de ventas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: Chi-squared test — Drift en features categóricas

```python
from scipy.stats import chi2_contingency

def chi2_drift(train_cat, prod_cat):
    # Crear tabla de contingencia con mismas categorías
    categories = sorted(set(train_cat) | set(prod_cat))
    train_counts = [sum(train_cat == c) for c in categories]
    prod_counts = [sum(prod_cat == c) for c in categories]
    contingency = pd.DataFrame([train_counts, prod_counts], columns=categories)
    
    chi2, p, dof, expected = chi2_contingency(contingency)
    return chi2, p

# Categorías de producto
train_categoria = np.random.choice(['electronica', 'ropa', 'hogar', 'alimentos'],
                                    size=5000, p=[0.3, 0.25, 0.25, 0.2])
prod_categoria = np.random.choice(['electronica', 'ropa', 'hogar', 'alimentos'],
                                   size=1000, p=[0.4, 0.2, 0.2, 0.2])  # Drift en electrónica

chi2, p = chi2_drift(train_categoria, prod_categoria)
print(f"Chi-squared: {chi2:.2f}")
print(f"P-value: {p:.4f}")
print(f"Drift categórico: {'SÍ' if p < 0.05 else 'NO'}")

# Mostrar frecuencias
print("\nFrecuencias relativas:")
for cat in sorted(set(train_categoria)):
    train_pct = (train_categoria == cat).mean()
    prod_pct = (prod_categoria == cat).mean()
    print(f"  {cat:15s} | train={train_pct:.1%} | prod={prod_pct:.1%}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: Chi-squared test — Drift en features categóricas.*

1. Crear tabla de contingencia con mismas categorías
2. Categorías de producto
3. Mostrar frecuencias

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: PSI — Population Stability Index (PSI > 0.1 = drift significativo)

```python
def calcular_psi(expected, actual, bins=10):
    breaks = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breaks[0] = -np.inf
    breaks[-1] = np.inf
    
    expected_counts = np.histogram(expected, breaks)[0]
    actual_counts = np.histogram(actual, breaks)[0]
    
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)
    
    psi = np.sum((expected_pct - actual_pct) * np.log(expected_pct / actual_pct))
    return psi

# Calcular PSI para precio
psi_precio = calcular_psi(train_precio, prod_precio)
print(f"PSI Precio: {psi_precio:.4f}")
if psi_precio < 0.1:
    print("→ Sin drift significativo")
elif psi_precio < 0.2:
    print("→ Drift moderado - requiere investigación")
else:
    print("→ Drift severo - requiere acción inmediata")

# PSI para múltiples features
features = ['precio', 'descuento', 'inventario']
for feat in features:
    psi = calcular_psi(train_df[feat], prod_df[feat])
    print(f"PSI {feat:12s}: {psi:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: PSI — Population Stability Index (PSI > 0.1 = drift significativo).*

1. Calcular PSI para precio
2. PSI para múltiples features

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: Jensen-Shannon divergence — Drift entre distribuciones

```python
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde

def js_divergence(train, prod, n_points=100):
    # Estimar densidades
    kde_train = gaussian_kde(train)
    kde_prod = gaussian_kde(prod)
    
    # Grid común
    grid = np.linspace(min(train.min(), prod.min()), max(train.max(), prod.max()), n_points)
    p = kde_train(grid) + 1e-10
    q = kde_prod(grid) + 1e-10
    p /= p.sum()
    q /= q.sum()
    
    return jensenshannon(p, q)

js_precio = js_divergence(train_precio, prod_precio)
print(f"Jensen-Shannon divergence (precio): {js_precio:.4f}")
print(f"JS = 0: distribuciones idénticas")
print(f"JS = 1: distribuciones completamente diferentes")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: Jensen-Shannon divergence — Drift entre distribuciones.*

1. Estimar densidades
2. Grid común

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: Concept drift — Monitorear accuracy en ventana deslizante

```python
def simulate_concept_drift(n_samples=5000, drift_point=3000):
    np.random.seed(42)
    X = np.random.randn(n_samples, 2)
    y = np.zeros(n_samples)
    # Antes de drift: y = x1 + x2
    y[:drift_point] = (X[:drift_point, 0] + X[:drift_point, 1] > 0).astype(int)
    # Después de drift: y = x1 - x2 (relación cambia)
    y[drift_point:] = (X[drift_point:, 0] - X[drift_point:, 1] > 0).astype(int)
    return X, y, drift_point

X, y, drift_pt = simulate_concept_drift()
print(f"Drift simulado en muestra {drift_pt}")

# Monitoreo con ventana deslizante
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

window_size = 500
accuracies = []
for i in range(0, len(X) - window_size, 100):
    X_train = X[max(0, i-1000):i]
    y_train = y[max(0, i-1000):i]
    X_test = X[i:i+window_size]
    y_test = y[i:i+window_size]
    
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    accuracies.append((i + window_size // 2, acc))

# Detectar caída
acc_series = pd.Series([a[1] for a in accuracies])
if acc_series.iloc[-5:].mean() < acc_series.iloc[:5].mean() - 0.1:
    print("⚠️ Concept drift detectado: accuracy cayó >10%")
else:
    print("✅ Accuracy estable")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: Concept drift — Monitorear accuracy en ventana deslizante.*

1. Antes de drift: y = x1 + x2
2. Después de drift: y = x1 - x2 (relación cambia)
3. Monitoreo con ventana deslizante
4. Detectar caída

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: DDM — Detectar cambio en tasa de error

```python
class DDM:
    def __init__(self, warning_level=2.0, drift_level=3.0):
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.mean = 0
        self.std = 0
        self.n = 0
        self.warning_zone = False
        self.drift_detected = False
    
    def add_element(self, error):
        self.n += 1
        self.mean = self.mean + (error - self.mean) / self.n
        self.std = np.sqrt(self.std**2 + ((error - self.mean)**2 - self.std**2) / self.n)
        
        if self.n > 30:  # Mínimo de samples
            if self.mean + self.drift_level * self.std > 0.3:
                self.drift_detected = True
            elif self.mean + self.warning_level * self.std > 0.2:
                self.warning_zone = True

# Simular errores (aumentan después de drift)
errors = np.random.binomial(1, 0.1, 2000)
errors[1000:] = np.random.binomial(1, 0.3, 1000)  # Drift en tasa de error

ddm = DDM()
for i, err in enumerate(errors):
    ddm.add_element(err)
    if ddm.drift_detected:
        print(f"⚠️ Drift detectado en iteración {i}")
        break
if not ddm.drift_detected:
    print("✅ No se detectó drift (DDM)")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: DDM — Detectar cambio en tasa de error.*

1. Simular errores (aumentan después de drift)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: Evidently AI — DataDriftPreset para reporte completo

```python
import pandas as pd
from sklearn import datasets

# Datos de ejemplo
np.random.seed(42)
ref = pd.DataFrame({
    'precio': np.random.lognormal(4.5, 0.5, 5000),
    'descuento': np.random.uniform(0, 0.3, 5000),
    'inventario': np.random.poisson(200, 5000),
    'categoria': np.random.choice(['A', 'B', 'C'], 5000)
})

prod = ref.copy()
prod['precio'] = np.random.lognormal(4.8, 0.6, 5000)
prod['inventario'] = np.random.poisson(180, 5000)

# Evidently Report
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=prod)
    report.save_html('evidently_drift_report.html')
    print("Reporte Evidently guardado: evidently_drift_report.html")
    
    # Obtener resumen
    summary = report.as_dict()
    n_drifted = summary['metrics'][0]['result']['number_of_drifted_features']
    print(f"Features con drift: {n_drifted}/{len(ref.columns)}")
except ImportError:
    print("Evidently no instalado. Instalar: pip install evidently")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: Evidently AI — DataDriftPreset para reporte completo.*

1. Datos de ejemplo
2. Evidently Report
3. Obtener resumen

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: Evidently AI — RegressionPreset para monitoreo de modelo

```python
try:
    from evidently.metric_preset import RegressionPreset
    from evidently.metric_preset import TargetDriftPreset

    # Datos con predicciones
    ref_with_preds = ref.copy()
    ref_with_preds['prediccion'] = ref_with_preds['precio'] * 0.5 + np.random.randn(5000) * 10
    ref_with_preds['demanda'] = ref_with_preds['prediccion'] + np.random.randn(5000) * 5

    prod_with_preds = prod.copy()
    prod_with_preds['prediccion'] = prod_with_preds['precio'] * 0.5 + np.random.randn(5000) * 10 + 15
    prod_with_preds['demanda'] = prod_with_preds['prediccion'] + np.random.randn(5000) * 5 + 10

    report = Report(metrics=[
        RegressionPreset(),
        TargetDriftPreset()
    ])
    report.run(reference_data=ref_with_preds, current_data=prod_with_preds)
    report.save_html('evidently_regression_report.html')
    print("Reporte de regresión guardado")
except ImportError:
    print("Evidently no instalado")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: Evidently AI — RegressionPreset para monitoreo de modelo.*

1. Datos con predicciones

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: Evidently — report.save_html('report.html')

```python
# Guardar reporte con personalización
try:
    from evidently.test_suite import TestSuite
    from evidently.test_preset import DataDriftTestPreset
    
    suite = TestSuite(tests=[DataDriftTestPreset()])
    suite.run(reference_data=ref, current_data=prod)
    suite.save_html('evidently_test_suite.html')
    print("Test suite guardado en evidently_test_suite.html")
    
    # Resultados como dict
    result_dict = suite.as_dict()
    print(f"Tests pasados: {result_dict['summary']['all_passed']}")
    print(f"Tests totales: {result_dict['summary']['total_tests']}")
except ImportError:
    print("Evidently no instalado")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: Evidently — report.save_html('report.html').*

1. Guardar reporte con personalización
2. Resultados como dict

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: Drift por feature — Ranking de features más inestables

```python
def ranking_drift(train_df, prod_df):
    scores = {}
    for col in train_df.select_dtypes(include=[np.number]).columns:
        if col in prod_df.columns:
            psi = calcular_psi(train_df[col], prod_df[col])
            ks_stat, p_val = ks_2samp(train_df[col], prod_df[col])
            scores[col] = {'PSI': psi, 'KS': ks_stat, 'p_value': p_val}
    
    ranking = pd.DataFrame(scores).T.sort_values('PSI', ascending=False)
    return ranking

ranking = ranking_drift(train_df, prod_df)
print("Ranking de features por inestabilidad (PSI):")
print(ranking.to_string())
print(f"\nFeature más inestable: {ranking.index[0]} (PSI={ranking.iloc[0]['PSI']:.4f})")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: Drift por feature — Ranking de features más inestables.*

1. `ranking = pd.DataFrame(scores).T.sort_values('PSI', ascending=False)` — Ordena los datos según la columna eRealiza la operación indicada con los parámetros definidos..

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: Alerta — Enviar email si PSI > 0.2

```python
import smtplib
from email.message import EmailMessage
import json

def check_drift_and_alert(train_df, prod_df, threshold=0.2):
    alertas = []
    for col in train_df.select_dtypes(include=[np.number]).columns:
        if col in prod_df.columns:
            psi = calcular_psi(train_df[col], prod_df[col])
            if psi > threshold:
                alertas.append({'feature': col, 'PSI': round(psi, 4)})
    
    if alertas:
        print(f"⚠️ ALERTA: {len(alertas)} features con drift severo (> {threshold})")
        for a in alertas:
            print(f"  - {a['feature']}: PSI={a['PSI']}")
        
        # Simular envío de email
        print("\nSimulando envío de email de alerta...")
        msg = EmailMessage()
        msg.set_content(f"""
        ALERTA DE DRIFT - Modelo de Predicción de Demanda
        
        Features con drift detectado:
        {json.dumps(alertas, indent=2)}
        
        Fecha: {pd.Timestamp.now()}
        """)
        msg['Subject'] = '⚠️ ALERTA: Data Drift Detectado en Modelo de Ventas'
        msg['From'] = 'monitoreo@ventas.com'
        msg['To'] = 'datascience@ventas.com'
        print("Email listo para enviar (simulado)")
        print("Para envío real: configurar SMTP y credenciales")
    else:
        print("✅ Sin drift severo detectado")

check_drift_and_alert(train_df, prod_df)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: Alerta — Enviar email si PSI > 0.2.*

1. Simular envío de email

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: Slack webhook — Notificar drift detectado

```python
import requests
import json

def slack_alert(webhook_url, message):
    """Enviar alerta a Slack mediante webhook"""
    payload = {
        'text': f'⚠️ *Data Drift Detectado*\n{message}\nTimestamp: {pd.Timestamp.now()}',
        'channel': '#ml-monitoreo',
        'username': 'DriftBot',
        'icon_emoji': ':robot_face:'
    }
    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        if response.status_code == 200:
            print("✅ Alerta enviada a Slack")
        else:
            print(f"❌ Error Slack: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error de conexión: {e}")

# Simular (no envía realmente)
print("Slack alerta simulada:")
print(json.dumps({
    'text': '⚠️ *Data Drift Detectado*\nFeature: precio | PSI: 0.35\nFeature: inventario | PSI: 0.28',
    'channel': '#ml-monitoreo'
}, indent=2))
print("\nPara usar: slack_alert('https://hooks.slack.com/services/XXX', 'mensaje')")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: Slack webhook — Notificar drift detectado.*

1. Simular (no envía realmente)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: Dashboard — Streamlit con métricas de drift

```python
import streamlit as st

def crear_dashboard_drift():
    st.set_page_config(page_title='Monitoreo ML - Ventas', layout='wide')
    st.title('📊 Dashboard de Monitoreo de Modelos')
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Drift (PSI medio)', f'{np.random.uniform(0.05, 0.15):.3f}',
                delta='+0.02' if np.random.random() > 0.5 else '-0.01')
    col2.metric('Features con drift', f'{np.random.randint(0, 3)}/10',
                delta='+1' if np.random.random() > 0.5 else '0')
    col3.metric('Precisión modelo', f'{np.random.uniform(0.82, 0.90):.1%}',
                delta='-0.5%' if np.random.random() > 0.5 else '+0.3%')
    col4.metric('Último entrenamiento', '2024-06-15', delta='5 días')
    
    st.subheader('PSI por Feature')
    features = ['precio', 'descuento', 'inventario', 'rating', 'categoria']
    psi_values = np.random.uniform(0.02, 0.25, len(features))
    
    # Visualización simple
    st.bar_chart(pd.DataFrame({'feature': features, 'PSI': psi_values}).set_index('feature'))

    st.subheader('Últimas Alertas')
    for i in range(3):
        st.info(f'Alerta {i+1}: Feature con PSI > 0.2 detectada')
    
    st.sidebar.header('Configuración')
    st.sidebar.slider('Threshold PSI', 0.05, 0.5, 0.2)
    st.sidebar.selectbox('Periodo', ['Última hora', 'Último día', 'Última semana'])

print("Dashboard listo. Ejecutar: streamlit run dashboard.py")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: Dashboard — Streamlit con métricas de drift.*

1. Visualización simple

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: Re-entrenamiento automático — Cuando drift > threshold

```python
def auto_retrain_pipeline(train_df, prod_df, model_path='models/modelo.pkl'):
    # 1. Evaluar drift
    features_drift = 0
    for col in train_df.select_dtypes(include=[np.number]).columns:
        if col in prod_df.columns:
            psi = calcular_psi(train_df[col], prod_df[col])
            if psi > 0.2:
                features_drift += 1
    
    # 2. Decidir re-entrenamiento
    if features_drift >= 2:
        print(f"⚠️ {features_drift} features con drift severo")
        print("🔄 Iniciando re-entrenamiento automático...")
        
        # Simular re-entrenamiento
        print("  → Cargando nuevos datos de ventas")
        print("  → Entrenando nuevo modelo")
        print("  → Evaluando métricas")
        print("  → Guardando modelo en {model_path}")
        
        # 3. Validación
        new_mse = np.random.uniform(0.8, 1.2)
        old_mse = 1.15
        if new_mse < old_mse:
            print(f"✅ Nuevo modelo mejor: MSE {old_mse:.3f} → {new_mse:.3f}")
            print("  → Promoviendo a producción")
        else:
            print(f"❌ Nuevo modelo no mejora: MSE {old_mse:.3f} → {new_mse:.3f}")
            print("  → Manteniendo modelo actual")
    else:
        print(f"✅ Drift controlado ({features_drift} features)")

auto_retrain_pipeline(train_df, prod_df)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: Re-entrenamiento automático — Cuando drift > threshold.*

1. 1. Evaluar drift
2. 2. Decidir re-entrenamiento
3. Simular re-entrenamiento
4. 3. Validación

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 16: Scheduled retraining con Airflow (semanal)

```python
# DAG de Airflow para re-entrenamiento semanal
airflow_dag = """
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'datascience',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'retrain_demanda_model',
    default_args=default_args,
    description='Re-entrenamiento semanal de modelo de demanda',
    schedule_interval='0 6 * * 1',  # Lunes 6 AM
    catchup=False,
)

def check_drift():
    # Evaluar drift con datos de la última semana
    print("Verificando drift...")
    return True

def retrain_model():
    # Re-entrenar con datos actualizados
    print("Re-entrenando modelo...")
    return True

def evaluate_model():
    # Evaluar contra modelo actual
    print("Evaluando nuevo modelo...")
    return True

def promote_if_better():
    # Promover a producción si mejora
    print("Promoviendo modelo...")
    return True

check = PythonOperator(task_id='check_drift', python_callable=check_drift, dag=dag)
retrain = PythonOperator(task_id='retrain', python_callable=retrain_model, dag=dag)
evaluate = PythonOperator(task_id='evaluate', python_callable=evaluate_model, dag=dag)
promote = PythonOperator(task_id='promote', python_callable=promote_if_better, dag=dag)

check >> retrain >> evaluate >> promote
"""

with open('airflow_retrain_dag.py', 'w') as f:
    f.write(airflow_dag)
print("DAG de Airflow para re-entrenamiento semanal creado")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: Scheduled retraining con Airflow (semanal).*

1. DAG de Airflow para re-entrenamiento semanal
2. Evaluar drift con datos de la última semana
3. Re-entrenar con datos actualizados
4. Evaluar contra modelo actual
5. Promover a producción si mejora

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 17: Accuracy degradation — Monitorear error en producción con ground truth retrasado

```python
def monitorear_accuracy_diario():
    # Simular datos de producción con ground truth retrasado
    dias = 90
    fechas = pd.date_range('2024-01-01', periods=dias, freq='D')
    
    np.random.seed(42)
    # Error empieza bajo y aumenta gradualmente
    error_base = 0.08
    drift_factor = np.linspace(1, 2.5, dias)
    errores_diarios = error_base * drift_factor + np.random.randn(dias) * 0.01
    
    df_monitor = pd.DataFrame({
        'fecha': fechas,
        'error_rate': errores_diarios,
        'samples': np.random.poisson(500, dias)
    })
    
    # Detectar degradación
    ventana = 7
    df_monitor['error_ma'] = df_monitor['error_rate'].rolling(ventana).mean()
    
    # Alarma si error semanal > 1.5x error inicial
    threshold = error_base * 1.5
    df_monitor['alarma'] = df_monitor['error_ma'] > threshold
    
    print(f"Días monitoreados: {dias}")
    print(f"Días con alarma: {df_monitor['alarma'].sum()}")
    
    if df_monitor['alarma'].iloc[-1]:
        print(f"⚠️ ALERTA: Error actual ({df_monitor['error_ma'].iloc[-1]:.3f}) "
              f"supera threshold ({threshold:.3f})")
    else:
        print(f"✅ Error dentro de límites ({df_monitor['error_ma'].iloc[-1]:.3f})")
    
    # Mostrar últimas semanas
    ultimos = df_monitor.tail(14)
    print(f"\nÚltimos 14 días:")
    print(ultimos[['fecha', 'error_rate', 'error_ma', 'alarma']].to_string(index=False))
    
    return df_monitor

df_acc = monitorear_accuracy_diario()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: Accuracy degradation — Monitorear error en producción con ground truth retrasado.*

1. Simular datos de producción con ground truth retrasado
2. Error empieza bajo y aumenta gradualmente
3. Detectar degradación
4. Alarma si error semanal > 1.5x error inicial
5. Mostrar últimas semanas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 18: Integrador — Pipeline de monitoreo completo

```python
class PipelineMonitoreo:
    def __init__(self, reference_data, threshold_psi=0.2):
        self.reference = reference_data
        self.threshold = threshold_psi
        self.alert_history = []
    
    def evaluar_drift(self, current_data):
        resultados = {}
        for col in self.reference.select_dtypes(include=[np.number]).columns:
            if col in current_data.columns:
                psi = calcular_psi(self.reference[col], current_data[col])
                ks_stat, p_val = ks_2samp(self.reference[col], current_data[col])
                resultados[col] = {
                    'PSI': round(psi, 4),
                    'KS': round(ks_stat, 4),
                    'p_value': round(p_val, 6),
                    'drift_severo': psi > self.threshold
                }
        return resultados
    
    def generar_alerta(self, resultados):
        severos = {k: v for k, v in resultados.items() if v['drift_severo']}
        if severos:
            alerta = {
                'timestamp': pd.Timestamp.now(),
                'features_drift': severos,
                'n_features': len(severos),
                'mensaje': f"{len(severos)} features con drift severo"
            }
            self.alert_history.append(alerta)
            return alerta
        return None
    
    def reporte_diario(self, current_data):
        resultados = self.evaluar_drift(current_data)
        alerta = self.generar_alerta(resultados)
        
        print("="*50)
        print(f"REPORTE DIARIO - {pd.Timestamp.now().date()}")
        print("="*50)
        print(f"{'Feature':15s} {'PSI':8s} {'KS':8s} {'Drift?':8s}")
        for feat, vals in sorted(resultados.items(), key=lambda x: x[1]['PSI'], reverse=True):
            drift_str = '⚠️' if vals['drift_severo'] else '✅'
            print(f"{feat:15s} {vals['PSI']:8.4f} {vals['KS']:8.4f} {drift_str:8s}")
        
        if alerta:
            print(f"\n🚨 ALERTA: {alerta['mensaje']}")
            print(f"Acción: Re-entrenamiento automático iniciado")
        else:
            print(f"\n✅ Sin drift severo. Modelo estable.")
        
        return resultados, alerta

# Ejecutar pipeline
monitor = PipelineMonitoreo(train_df)
resultados, alerta = monitor.reporte_diario(prod_df)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador — Pipeline de monitoreo completo.*

1. Ejecutar pipeline

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 3. Ejercicios Propuestos

1. **Dashboard de drift en tiempo real con Evidently + Streamlit**: Crea un dashboard que cargue datos de entrenamiento y producción, calcule PSI y KS para cada feature, y muestre alertas en tiempo real cuando PSI > 0.2. Actualiza cada 60 segundos.

2. **Sistema multi-feature de alertas**: Implementa un sistema que monitoree 5 features de ventas (precio, descuento, inventario, rating, envíos) y envíe alerta a Slack cuando al menos 2 features tengan drift severo (PSI > 0.2). Incluye el mensaje con los valores específicos.

3. **Detección de concept drift con ADWIN**: Implementa ADWIN (Adaptive Windowing) para detectar cambios en la relación precio-demanda. Simula datos donde la elasticidad precio-demanda cambia después de 2000 muestras. Muestra cuándo ADWIN detecta el cambio.

4. **Pipeline de re-entrenamiento automático**: Crea un pipeline que: (1) calcule drift semanal, (2) si PSI medio > 0.15, re-entrena modelo con últimos 30 días, (3) compara MSE nuevo vs actual, (4) si mejora > 5%, promueve a producción. Incluye logging de cada paso.

5. **Monitoreo de calidad de predicciones con ground truth retrasado**: Implementa un sistema que almacene predicciones en SQLite y, cuando llegue el ground truth (ventas reales del día), calcule el error. Si el error acumulado en 7 días supera un threshold, genera alerta.

6. **Comparar PSI vs KS vs JS divergence**: Para un dataset de ventas con drift inducido en precio (cambio de media y varianza), compara qué método detecta primero el drift. Genera un gráfico de los 3 scores a través del tiempo.

7. **Test suite automatizado con Evidently**: Usa Evidently TestSuite para crear un suite de tests que verifique: (1) drift en features numéricas (KS test), (2) drift en features categóricas (Chi-squared), (3) drift en target, (4) drift en predicciones. Genera reporte HTML.

8. **Sistema de monitoreo con alerta por email + Slack + webhook**: Implementa un sistema que, al detectar drift, envíe alertas simultáneamente por email (SMTP simulado), Slack (webhook) y webhook HTTP. Incluye reintentos con backoff exponencial.

---

## 4. Resumen

| Concepto | Método | Threshold | Acción |
|---|---|---|---|
| **Data drift** | PSI, KS, Chi2, JS | PSI > 0.2 | Investigar features, re-entrenar |
| **Concept drift** | DDM, ADWIN, ventana deslizante | Error > 2σ | Re-entrenar urgente |
| **Degradación** | Error acumulado en N días | Error > 1.5x baseline | Re-entrenar, revisar datos |
| **Evidently** | Report, TestSuite | p-value < 0.05 | Reporte HTML automático |
| **Alertas** | Email, Slack, Webhook | PSI > threshold | Notificar al equipo |

El monitoreo continuo es esencial para mantener modelos de ventas precisos. Un sistema de detección temprana de drift puede ahorrar miles de dólares en predicciones incorrectas.
