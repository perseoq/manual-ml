# E20: MLOps — Felicitaciones, Llegaste a Producción

## Objetivo
Implementar el ciclo completo de MLOps: experiment tracking, pipelines reproducibles, modelo registry, versionado de datos y estrategias de deploy para ventas/compras/inventarios.

---

## 1. Fundamentos Teóricos

### 1.1 MLflow Tracking
Registra experimentos de ML:
```python
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.create_experiment('ventas_prediccion')
with mlflow.start_run():
    mlflow.log_param('learning_rate', 0.01)
    mlflow.log_metric('accuracy', 0.95)
    mlflow.log_artifact('confusion_matrix.png')
    mlflow.log_model(model, 'model')
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*1.1 MLflow Tracking.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---


- **autolog**: Registro automático para sklearn, TensorFlow, PyTorch
- **search_runs**: Búsqueda y comparación de experimentos

### 1.2 MLflow Model Registry
```python
mlflow.register_model('runs:/RUN_ID/model', 'VentasModel')
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage('VentasModel', 1, 'Production')
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*1.2 MLflow Model Registry.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### 1.3 MLflow Project
Formato reproducible: archivo MLproject define entry points, entorno y parámetros.

### 1.4 DVC (Data Version Control)
```bash
dvc init
dvc add data/ventas.csv
git add data/ventas.csv.dvc
dvc repro       # Reproducir pipeline
dvc push        # Subir datos a remoto
dvc pull        # Bajar datos
```
Pipeline en dvc.yaml:
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*1.4 DVC (Data Version Control).*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

yaml
stages:
  train:
    cmd: python train.py
    deps: [data/ventas.csv, src/train.py]
    outs: [models/model.pkl]
    params: [train.learning_rate, train.n_estimators]
    metrics: [metrics.json]
```

### 1.5 Champion-Challenger
- **Champion**: Modelo en producción actual
- **Challenger**: Nuevo modelo candidato
- A/B test: dividir tráfico entre champion y challenger
- Evaluar métricas de negocio (ROI, conversion rate)

### 1.6 Deployment Strategies
| Estrategia | Descripción | Downtime | Riesgo |
|---|---|---|---|
| **Blue-Green** | Dos entornos idénticos, switch instantáneo | No | Bajo |
| **Canary** | 10% → 50% → 100% tráfico gradual | No | Muy bajo |
| **Rolling** | Actualizar instancias una por una | No | Bajo |
| **Shadow** | Nuevo modelo recibe copia del tráfico | No | Mínimo |

---

## 2. Ejemplos Prácticos

### Ejemplo 1: MLflow — start_run, log_param, log_metric

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 1.5 Champion-Challenger
2. 1.6 Deployment Strategies
3. 2. Ejemplos Prácticos
4. Ejemplo 1: MLflow — start_run, log_param, log_metric

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
import mlflow
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

np.random.seed(42)
X = np.random.randn(1000, 5)
y = np.random.randn(1000)

mlflow.set_experiment('ventas_prediccion')

with mlflow.start_run(run_name='rf_base'):
    params = {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5}
    mlflow.log_params(params)

    rf = RandomForestRegressor(**params, random_state=42)
    mse = -cross_val_score(rf, X, y, cv=3, scoring='neg_mean_squared_error').mean()
    r2 = cross_val_score(rf, X, y, cv=3, scoring='r2').mean()

    mlflow.log_metrics({'mse': mse, 'r2': r2})
    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print(f"MSE: {mse:.4f}, R2: {r2:.4f}")
```

### Ejemplo 2: MLflow autolog — Registrar automáticamente parámetros de sklearn

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 2: MLflow autolog — Registrar automáticamente parámetros de sklearn

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
mlflow.sklearn.autolog()

with mlflow.start_run(run_name='autolog_rf'):
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
    rf.fit(X, y)
    score = rf.score(X, y)
    print(f"Score autolog: {score:.4f}")
    # MLflow registra automáticamente: params, metrics, model
```

### Ejemplo 3: MLflow — log_artifact('confusion_matrix.png')

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 3: MLflow — log_artifact('confusion_matrix.png')

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

with mlflow.start_run(run_name='confusion_matrix'):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    y_pred = rf.predict(X)

    # Crear matriz de confusión (para clasificación)
    y_clf = np.random.randint(0, 2, 1000)
    y_pred_clf = np.random.randint(0, 2, 1000)
    cm = confusion_matrix(y_clf, y_pred_clf)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')
    print("Artefacto guardado en MLflow")
```

### Ejemplo 4: MLflow — log_model con flavor sklearn

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 4: MLflow — log_model con flavor sklearn

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
with mlflow.start_run(run_name='log_model'):
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X, y)
    mlflow.sklearn.log_model(rf, 'random_forest_model')
    print("Modelo registrado en MLflow")
```

### Ejemplo 5: MLflow — search_runs para comparar experimentos

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 5: MLflow — search_runs para comparar experimentos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
# Ejecutar varios experimentos
for n_est in [50, 100, 200]:
    with mlflow.start_run(run_name=f'rf_{n_est}'):
        rf = RandomForestRegressor(n_estimators=n_est, max_depth=10, random_state=42)
        mse = -cross_val_score(rf, X, y, cv=3, scoring='neg_mean_squared_error').mean()
        mlflow.log_params({'n_estimators': n_est, 'max_depth': 10})
        mlflow.log_metric('mse', mse)

# Buscar y comparar
runs = mlflow.search_runs(
    experiment_names=['ventas_prediccion'],
    order_by=['metrics.mse ASC']
)
print("Experimentos ordenados por MSE:")
print(runs[['run_name', 'params.n_estimators', 'metrics.mse']].to_string(index=False))
```

### Ejemplo 6: MLflow Model Registry — Registrar modelo en registro

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 6: MLflow Model Registry — Registrar modelo en registro

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
mlflow.set_tracking_uri('sqlite:///mlflow.db')
with mlflow.start_run(run_name='registry_demo'):
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X, y)
    mlflow.sklearn.log_model(rf, 'model')
    run_id = mlflow.active_run().info.run_id

# Registrar modelo
model_uri = f'runs:/{run_id}/model'
result = mlflow.register_model(model_uri, 'VentasPrediccionModel')
print(f"Modelo registrado: {result.name} v{result.version}")
```

### Ejemplo 7: MLflow — Transition model to 'Production'

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 7: MLflow — Transition model to 'Production'

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
from mlflow.tracking import MlflowClient

client = MlflowClient()
# Asumiendo que el modelo ya está registrado
try:
    client.transition_model_version_stage(
        name='VentasPrediccionModel',
        version=1,
        stage='Production'
    )
    print("Modelo promovido a Production")
    # Verificar estado
    mv = client.get_model_version('VentasPrediccionModel', 1)
    print(f"Stage actual: {mv.current_stage}")
except Exception as e:
    print(f"Error: {e}. Asegúrate de que el modelo esté registrado.")
```

### Ejemplo 8: DVC — dvc init y dvc add (versionar datos de ventas)

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 8: DVC — dvc init y dvc add (versionar datos de ventas, compras o inventarios)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
# Simulación de DVC (comandos bash)
import subprocess
import os

# Crear datos de ejemplo
os.makedirs('data', exist_ok=True)
np.savetxt('data/ventas.csv', np.random.randn(1000, 5), delimiter=',',
           header='precio,descuento,inventario,dia_semana,demanda', comments='')

# Comandos DVC (requiere DVC instalado)
try:
    subprocess.run(['dvc', 'init'], cwd=os.getcwd(), check=True, capture_output=True)
    subprocess.run(['dvc', 'add', 'data/ventas.csv'], cwd=os.getcwd(), check=True, capture_output=True)
    print("DVC init y add completados")
    print("Archivos generados:")
    for f in os.listdir('.'):
        if f.endswith('.dvc') or f == '.dvcignore':
            print(f"  - {f}")
except FileNotFoundError:
    print("DVC no instalado. Simulación conceptual:")
    print("  dvc init")
    print("  dvc add data/ventas.csv")
    print("  git add data/ventas.csv.dvc")
```

### Ejemplo 9: DVC — dvc.yaml con pipeline de entrenamiento

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 9: DVC — dvc.yaml con pipeline de entrenamiento

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
# Crear dvc.yaml programáticamente
dvc_yaml = """
stages:
  preprocess:
    cmd: python src/preprocess.py
    deps:
      - data/ventas.csv
      - src/preprocess.py
    outs:
      - data/ventas_procesadas.csv
    params:
      - preprocess.test_size
  train:
    cmd: python src/train.py
    deps:
      - data/ventas_procesadas.csv
      - src/train.py
    outs:
      - models/model.pkl
    params:
      - train.n_estimators
      - train.max_depth
    metrics:
      - metrics/train_metrics.json:
          cache: false
  evaluate:
    cmd: python src/evaluate.py
    deps:
      - models/model.pkl
      - data/ventas_procesadas.csv
    metrics:
      - metrics/eval_metrics.json:
          cache: false
"""

with open('dvc.yaml', 'w') as f:
    f.write(dvc_yaml)
print("dvc.yaml creado con 3 stages: preprocess → train → evaluate")
```

### Ejemplo 10: DVC — dvc repro para reproducir pipeline

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 10: DVC — dvc repro para reproducir pipeline

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
try:
    result = subprocess.run(['dvc', 'repro'], cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=30)
    print("Pipeline reproducido exitosamente")
    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
except subprocess.TimeoutExpired:
    print("Timeout: la reproducción tomó más de 30s")
except FileNotFoundError:
    print("DVC no instalado. Repro simulado.")
    print("dvc repro ejecutaría: preprocess → train → evaluate")
```

### Ejemplo 11: Experiment tracking — Comparar 10 experimentos en tabla

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 11: Experiment tracking — Comparar 10 experimentos en tabla

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
import pandas as pd

# Simular 10 experimentos
experimentos = []
for i in range(10):
    exp = {
        'run_id': f'run_{i:03d}',
        'modelo': np.random.choice(['rf', 'xgb', 'gbm']),
        'n_estimators': np.random.choice([50, 100, 200, 300]),
        'max_depth': np.random.randint(3, 15),
        'mse': np.random.uniform(0.5, 2.0),
        'r2': np.random.uniform(0.6, 0.95),
        'tiempo_seg': np.random.uniform(5, 60)
    }
    experimentos.append(exp)

df_exp = pd.DataFrame(experimentos)
print("Tabla de experimentos:")
print(df_exp.to_string(index=False))
print(f"\nMejor modelo: {df_exp.loc[df_exp['mse'].idxmin(), 'run_id']}")
print(f"Mejor MSE: {df_exp['mse'].min():.4f}")
```

### Ejemplo 12: Reproducibilidad — Mismo commit + misma data = mismos resultados

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 12: Reproducibilidad — Mismo commit + misma data = mismos resultados

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
import hashlib
import json

def hash_experimento(params, data_hash, code_hash):
    return hashlib.sha256(
        json.dumps({'params': params, 'data': data_hash, 'code': code_hash},
                    sort_keys=True).encode()
    ).hexdigest()[:12]

# Simular reproducibilidad
params_v1 = {'n_estimators': 100, 'max_depth': 10, 'learning_rate': 0.05}
data_hash_v1 = 'a1b2c3d4'
code_hash_v1 = 'e5f6g7h8'

h1 = hash_experimento(params_v1, data_hash_v1, code_hash_v1)
h2 = hash_experimento(params_v1, data_hash_v1, code_hash_v1)
print(f"Hash 1: {h1}")
print(f"Hash 2: {h2}")
print(f"Reproducible: {'SÍ' if h1 == h2 else 'NO'}")

# Cambiar un parámetro
params_v2 = {'n_estimators': 200, 'max_depth': 10, 'learning_rate': 0.05}
h3 = hash_experimento(params_v2, data_hash_v1, code_hash_v1)
print(f"Hash 3 (params diferentes): {h3}")
print(f"Mismo hash: {'SÍ' if h1 == h3 else 'NO — diferente, como se espera'}")
```

### Ejemplo 13: Model versioning — v1, v2, v3 con métricas

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 13: Model versioning — v1, v2, v3 con métricas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
# Simular versiones de modelo
versiones = []
for v in range(1, 6):
    versiones.append({
        'version': f'v{v}.0',
        'fecha': f'2024-0{v}-15',
        'n_estimators': 50 + v * 50,
        'max_depth': 5 + v * 2,
        'train_mse': round(1.5 / v, 3),
        'test_mse': round(1.8 / v, 3),
        'drift_score': round(np.random.uniform(0.01, 0.15), 3),
        'status': 'Archived' if v < 4 else 'Production'
    })

df_versiones = pd.DataFrame(versiones)
print("Historial de versiones:")
print(df_versiones.to_string(index=False))
```

### Ejemplo 14: Model staging — Staging para test, Production para live

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 14: Model staging — Staging para test, Production para live

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
stages = {
    'Development': {'purpose': 'Entrenamiento inicial', 'traffic': '0%'},
    'Staging': {'purpose': 'Validación y tests', 'traffic': '5% (interna)'},
    'Production': {'purpose': 'Tráfico en vivo', 'traffic': '100%'},
    'Archived': {'purpose': 'Modelos anteriores', 'traffic': '0%'}
}

print("Model Staging:")
for stage, info in stages.items():
    print(f"  {stage:15s} | {info['purpose']:25s} | Tráfico: {info['traffic']}")

# Transición simulada
print("\nPromoviendo modelo v3.0 a Production...")
print("✓ v3.0 pasa tests de validación (MSE < 1.0, drift < 0.1)")
print("✓ v3.0 promovido a Production")
print("✗ v2.0 movido a Archived")
```

### Ejemplo 15: Champion-challenger — Modelo en producción vs nuevo candidato

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 15: Champion-challenger — Modelo en producción vs nuevo candidato

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
champion_mse = 1.25
challenger_mse = 1.18

# A/B test simulado
n_users = 10000
traffic_champion = 0.7
traffic_challenger = 0.3

users_champion = int(n_users * traffic_champion)
users_challenger = n_users - users_champion

print("="*40)
print("CHAMPION vs CHALLENGER")
print("="*40)
print(f"Champion  (v2.0): {traffic_champion:.0%} tráfico ({users_champion} usuarios)")
print(f"  MSE: {champion_mse:.3f}")
print(f"Challenger (v3.0): {traffic_challenger:.0%} tráfico ({users_challenger} usuarios)")
print(f"  MSE: {challenger_mse:.3f}")
print(f"\nMejora relativa: {(1 - challenger_mse/champion_mse)*100:.1f}%")
if challenger_mse < champion_mse:
    print("✅ Challenger gana: promover a Champion")
else:
    print("❌ Champion retiene posición")
```

### Ejemplo 16: Deployment strategies — Blue-green deploy sin downtime

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 16: Deployment strategies — Blue-green deploy sin downtime

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
def blue_green_deploy(new_version):
    print(f"Iniciando Blue-Green deploy de v{new_version}")
    print(f"  Blue (actual): v{new_version-1} — 100% tráfico")
    print(f"  Green (nueva): v{new_version} — 0% tráfico")

    # Paso 1: Desplegar Green
    print("  [1/4] Desplegando Green (v{})...".format(new_version))
    print("  [2/4] Health check Green: OK")

    # Paso 2: Switch tráfico
    print("  [3/4] Switch balanceador: 100% tráfico → Green")
    print(f"  Green (v{new_version}) — 100% tráfico")
    print(f"  Blue  (v{new_version-1}) — 0% tráfico (standby)")

    # Paso 3: Verificar
    print("  [4/4] Monitoreo 15 min: sin errores")
    print(f"✅ Blue-Green deploy de v{new_version} completado sin downtime")

blue_green_deploy(3)
```

### Ejemplo 17: Canary deploy — 10% tráfico a nuevo modelo gradualmente

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 17: Canary deploy — 10% tráfico a nuevo modelo gradualmente

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
def canary_deploy(new_version, steps=[(0.1, '10 min'), (0.3, '30 min'),
                                       (0.5, '1 hora'), (1.0, '2 horas')]):
    print(f"Iniciando Canary deploy de v{new_version}")
    for traffic, duration in steps:
        pct = traffic * 100
        print(f"  → {pct:.0f}% tráfico a v{new_version} durante {duration}")
        # Simular monitoreo
        if traffic <= 0.5:
            print(f"    ✓ Sin errores, métricas estables")
        else:
            print(f"    ✓ Rollout completo, v{new_version} al 100%")
    print(f"✅ Canary deploy completado")

canary_deploy(4)
```

### Ejemplo 18: Integrador — Ciclo MLOps completo

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 18: Integrador — Ciclo MLOps completo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
def ciclo_mlops_completo():
    print("="*60)
    print("CICLO MLOPS COMPLETO")
    print("="*60)

    # Fase 1: Experimentación
    print("\n1. EXPERIMENTACIÓN")
    with mlflow.start_run(run_name='ciclo_completo') as run:
        params = {'n_estimators': 150, 'max_depth': 8}
        mlflow.log_params(params)
        rf = RandomForestRegressor(**params, random_state=42)
        rf.fit(X, y)
        mlflow.sklearn.log_model(rf, 'modelo_final')
        run_id = run.info.run_id
    print("   ✓ Experiment tracking con MLflow")

    # Fase 2: Versionado de datos
    print("\n2. VERSIONADO DE DATOS")
    print("   ✓ DVC: data/ventas.csv versionada")
    print("   ✓ Hash del dataset: a1b2c3d4e5f6")

    # Fase 3: Pipeline reproducible
    print("\n3. PIPELINE REPRODUCIBLE")
    print("   ✓ dvc.yaml: preprocess → train → evaluate")
    print("   ✓ dvc repro ejecuta pipeline completo")

    # Fase 4: Model Registry
    print("\n4. MODEL REGISTRY")
    model_uri = f'runs:/{run_id}/modelo_final'
    mlflow.register_model(model_uri, 'VentasPrediccion')
    print("   ✓ Modelo registrado en MLflow Registry")

    # Fase 5: Staging
    print("\n5. STAGING")
    print("   ✓ Modelo en Staging para validación")
    print("   ✓ Tests de calidad: accuracy > 0.80, drift < 0.1")

    # Fase 6: Producción
    print("\n6. PRODUCCIÓN")
    print("   ✓ Blue-Green deploy sin downtime")
    print("   ✓ FastAPI + Docker + Kubernetes")

    # Fase 7: Monitoreo
    print("\n7. MONITOREO")
    print("   ✓ Evidently AI: data drift semanal")
    print("   ✓ Prometheus + Grafana: latencia, error rate")
    print("   ✓ Alertas: Slack si drift > 0.2")

    # Fase 8: Re-entrenamiento
    print("\n8. RE-ENTRENAMIENTO")
    print("   ✓ Airflow DAG semanal: detect drift → retrain → evaluate → deploy")
    print("   ✓ Champion-Challenger: comparar nuevo vs actual")

    print("\n" + "="*60)
    print("✅ CICLO MLOPS COMPLETADO")
    print("="*60)

ciclo_mlops_completo()
```

---

## 3. Ejercicios Propuestos

1. **Pipeline MLflow completo**: Crea un experimento que entrene 3 modelos (RF, XGBoost, GradientBoosting) para predicción de demanda, registre todos los parámetros y métricas, y use search_runs para identificar el mejor modelo.

2. **DVC para datos de ventas reales**: Supón que tienes 3 meses de datos de ventas en CSV. Diseña un dvc.yaml con stages: (1) limpieza, (2) feature engineering, (3) entrenamiento, (4) evaluación. Incluye dependencias, salidas y métricas.

3. **Model Registry con versionado**: Registra 5 versiones de un modelo de predicción de demanda. Promueve v3 a Staging, v4 a Production. Muestra el historial de versiones con sus métricas (MSE, MAE, R2).

4. **Comparar deployment strategies**: Para un sistema de predicción de inventarios con 1000 requests/minuto, compara Blue-Green, Canary y Rolling en términos de: downtime, riesgo, tiempo de rollback y complejidad operativa.

5. **Champion-Challenger con A/B test**: Simula un A/B test donde Champion (MSE=1.35) recibe 80% tráfico y Challenger (MSE=1.22) recibe 20%. Tras 1 semana (10000 requests), decide si promover Challenger basado en significancia estadística.

6. **MLflow autolog con TensorFlow**: Usa mlflow.tensorflow.autolog() para entrenar una LSTM de forecasting de demanda. Verifica qué parámetros y métricas se registran automáticamente.

7. **Pipeline reproducible con DVC + MLflow**: Diseña un workflow donde DVC gestiona datos y pipeline, y MLflow trackea experimentos. Muestra cómo dvc repro y mlflow trabajan juntos para garantizar reproducibilidad.

8. **Dashboard de experimentos**: Crea un script que lea de MLflow todos los experimentos de ventas, los cargue en un DataFrame, y genere un dashboard HTML con: tabla de métricas, gráfico de convergencia, y ranking de modelos.

---

## 4. Resumen

| Componente | Herramienta | Propósito |
|---|---|---|
| **Experiment Tracking** | MLflow Tracking | Parámetros, métricas, artefactos, modelos |
| **Pipeline** | MLflow Projects / DVC | Reproducibilidad, dependencias |
| **Data Versioning** | DVC | Versionar datasets, features |
| **Model Registry** | MLflow Registry | Versionar, promover, staging/production |
| **Deploy** | FastAPI + Docker + K8s | API, contenedores, orquestación |
| **Monitoreo** | Evidently + Prometheus | Drift, performance, alertas |
| **Estrategias** | Blue-Green, Canary, Rolling | Zero-downtime deploy |

El ciclo MLOps no termina con el deploy: monitoreo continuo, re-entrenamiento y champion-challenger son esenciales para mantener modelos de ventas precisos en producción.
