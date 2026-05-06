# E24: MLOps en Producción Completa — CI/CD, Airflow, Kubernetes, Monitoreo End-to-End

## Objetivo
Implementar el ciclo completo de MLOps en producción: CI/CD con GitHub Actions, pipelines de datos con Airflow/Prefect, feature store con Feast, deployment con Kubernetes, inferencia con Triton/TF Serving, alerting con Prometheus/Grafana y re-entrenamiento automático.

---

## 1. Fundamentos Teóricos

### 1.1 CI/CD para ML
```yaml
# .github/workflows/ml-pipeline.yml
name: ML CI/CD
on: [push, pull_request]
jobs:
  test:
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - run: pip install -r requirements.txt
      - run: pytest tests/
      - run: python -m lint
  deploy:
    needs: test
    steps:
      - run: docker build -t api-ventas . && docker push
```

### 1.2 Testing en ML
- **Data quality tests**: schema validation, missing values, range checks, distribution checks
- **Model validation tests**: accuracy > threshold, no data leakage, bias test
- **Slice-based evaluation**: performance por segmento (categoría, región, temporada)

### 1.3 Airflow
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*1.1 CI/CD para ML.*

1. 1.2 Testing en ML
2. 1.3 Airflow

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
dag = DAG('daily_training', schedule_interval='@daily')
task1 = PythonOperator(task_id='extract', python_callable=extract)
task2 = PythonOperator(task_id='train', python_callable=train, retries=3)
task1 >> task2
```

### 1.4 Prefect
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 1.4 Prefect

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
@flow(name='ventas_pipeline')
def pipeline():
    data = extract()
    features = transform(data)
    model = train(features)
    evaluate(model)
```

### 1.5 Feast (Feature Store)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 1.5 Feast (Feature Store)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
feature_view = FeatureView(
    name='ventas_diarias',
    entities=[producto],
    features=[Feature(name='ventas_7d', dtype=Float32),
              Feature(name='precio_promedio', dtype=Float32)],
    source=batch_source
)
```

### 1.6 Kubernetes Deployment
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 1.6 Kubernetes Deployment

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

yaml
apiVersion: apps/v1
kind: Deployment
spec:
  replicas: 3
  template:
    spec:
      containers:
        - name: api-ventas
          image: api-ventas:latest
          readinessProbe:
            httpGet: {path: /health, port: 8000}
          resources:
            requests: {cpu: 500m, memory: 512Mi}
            limits: {cpu: 2, memory: 2Gi}
```

### 1.7 Inference Servers
- **Triton Inference Server**: Multi-model, multi-framework, dynamic batching
- **TorchServe**: PyTorch native serving
- **TF Serving**: TensorFlow native serving

### 1.8 Prometheus + Grafana
- **Prometheus**: Métricas (latency_histogram, error_counter, drift_gauge)
- **Grafana**: Dashboards, alertas, panel de drift

### 1.9 Retraining Pipeline
Detectar drift → Re-entrenar → Evaluar → Promover si mejor

---

## 2. Ejemplos Prácticos

### Ejemplo 1: GitHub Actions — Workflow de CI/CD para ML

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 1.7 Inference Servers
2. 1.8 Prometheus + Grafana
3. 1.9 Retraining Pipeline
4. 2. Ejemplos Prácticos
5. Ejemplo 1: GitHub Actions — Workflow de CI/CD para ML

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
github_actions_yml = """
name: ML Pipeline CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install flake8 pytest
      - name: Lint
        run: flake8 src/ tests/
      - name: Test
        run: pytest tests/ -v --cov=src
      - name: Data validation
        run: python tests/validate_data.py

  build-and-deploy:
    needs: lint-and-test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker image
        run: docker build -t api-ventas:${{ github.sha }} .
      - name: Push to registry
        run: |
          docker tag api-ventas:${{ github.sha }} registry.example.com/api-ventas:latest
          docker push registry.example.com/api-ventas:latest
      - name: Deploy to Kubernetes
        run: kubectl set image deployment/api-ventas api-ventas=registry.example.com/api-ventas:${{ github.sha }}
"""

with open('.github/workflows/ml-pipeline.yml', 'w') as f:
    f.write(github_actions_yml)
print("GitHub Actions workflow creado")
```

### Ejemplo 2: Data quality tests — Schema check de ventas.csv

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 2: Data quality tests — Schema check de ventas.csv

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
import pandera as pa
from pandera import Column, Check, DataFrameSchema

# Schema de validación
schema_ventas = DataFrameSchema({
    'fecha': Column(pa.DateTime, nullable=False),
    'producto_id': Column(pa.Int, Check.greater_than(0), nullable=False),
    'precio': Column(pa.Float, Check.greater_than(0), nullable=False),
    'cantidad': Column(pa.Int, Check.greater_than_or_equal_to(0), nullable=False),
    'descuento': Column(pa.Float, Check.in_range(0, 1), nullable=True),
    'categoria': Column(pa.String, nullable=True)
})

# Datos de prueba
ventas_test = pd.DataFrame({
    'fecha': pd.date_range('2024-01-01', periods=100, freq='D'),
    'producto_id': np.random.randint(1, 100, 100),
    'precio': np.random.uniform(5, 500, 100),
    'cantidad': np.random.poisson(50, 100),
    'descuento': np.random.uniform(0, 0.3, 100),
    'categoria': np.random.choice(['electronica', 'ropa', 'hogar'], 100)
})

try:
    schema_ventas.validate(ventas_test)
    print("✅ Schema validation: PASSED")
except pa.errors.SchemaError as e:
    print(f"❌ Schema validation: FAILED - {e}")

# Tests adicionales
print("\nTests de calidad:")
print(f"  Missing values: {ventas_test.isnull().sum().sum()}")
print(f"  Precios negativos: {(ventas_test['precio'] <= 0).sum()}")
print(f"  Cantidades negativas: {(ventas_test['cantidad'] < 0).sum()}")
assert ventas_test['precio'].min() > 0, "Precio negativo detectado"
assert ventas_test['cantidad'].min() >= 0, "Cantidad negativa detectada"
print("✅ Data quality tests: PASSED")
```

### Ejemplo 3: Model validation — Accuracy > 0.80 threshold

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 3: Model validation — Accuracy > 0.80 threshold

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Datos
X_clf, y_clf = make_classification(n_samples=1000, n_features=10, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X_clf, y_clf, test_size=0.2)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_tr, y_tr)
y_pred = model.predict(X_te)

# Thresholds
THRESHOLDS = {
    'accuracy': 0.80,
    'precision': 0.75,
    'recall': 0.75,
    'f1': 0.75
}

print("Validación de modelo:")
all_passed = True
for metric, threshold in THRESHOLDS.items():
    value = globals()[f"{metric}_score"](y_te, y_pred)
    passed = value >= threshold
    status = "✅ PASSED" if passed else "❌ FAILED"
    print(f"  {metric:10s}: {value:.4f} (threshold: {threshold}) {status}")
    if not passed:
        all_passed = False

if all_passed:
    print(f"\n✅ Model validation: ALL PASSED")
else:
    print(f"\n❌ Model validation: SOME TESTS FAILED")
```

### Ejemplo 4: Slice-based evaluation — Performance por categoría de producto

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 4: Slice-based evaluation — Performance por categoría de producto

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
# Datos con categorías
np.random.seed(42)
n = 2000
categorias = ['electronica', 'ropa', 'hogar', 'alimentos', 'deportes']
X_slice = pd.DataFrame({
    'feature_1': np.random.randn(n),
    'feature_2': np.random.randn(n),
    'categoria': np.random.choice(categorias, n)
})
y_slice = (X_slice['feature_1'] + X_slice['feature_2'] > 0).astype(int)

X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(
    X_slice.drop('categoria', axis=1), y_slice, test_size=0.2, random_state=42
)
cat_te = X_slice.loc[X_te_s.index, 'categoria']

model_slice = RandomForestClassifier(random_state=42)
model_slice.fit(X_tr_s, y_tr_s)
pred_slice = model_slice.predict(X_te_s)

print("Performance por categoría:")
results_slice = []
for cat in categorias:
    mask = cat_te == cat
    if mask.sum() > 0:
        acc = accuracy_score(y_te_s[mask], pred_slice[mask])
        results_slice.append({'categoria': cat, 'samples': mask.sum(), 'accuracy': acc})
        status = "✅" if acc > 0.75 else "⚠️" if acc > 0.6 else "❌"
        print(f"  {status} {cat:12s} | n={mask.sum():4d} | accuracy={acc:.4f}")

df_slice = pd.DataFrame(results_slice)
worst = df_slice.loc[df_slice['accuracy'].idxmin()]
print(f"\nPeor segmento: {worst['categoria']} (accuracy={worst['accuracy']:.4f})")
```

### Ejemplo 5: Airflow DAG — Daily training pipeline

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 5: Airflow DAG — Daily training pipeline

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
airflow_dag_full = """
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

default_args = {
    'owner': 'datascience',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': True,
    'email': ['alerts@ventas.com']
}

dag = DAG(
    'daily_ventas_pipeline',
    default_args=default_args,
    description='Pipeline diario de predicción de ventas',
    schedule_interval='0 6 * * *',  # 6 AM todos los días
    catchup=False,
    tags=['ventas', 'ml'],
)

def extract():
    print('Extrayendo datos de ventas del día anterior...')
    # Simular extracción
    df = pd.DataFrame(np.random.randn(1000, 5))
    df.to_csv('/tmp/ventas_raw.csv', index=False)
    print('Extracción completada')

def transform():
    print('Transformando datos...')
    df = pd.read_csv('/tmp/ventas_raw.csv')
    df.columns = ['precio', 'descuento', 'inventario', 'demanda', 'categoria']
    df.to_parquet('/tmp/ventas_procesadas.parquet')
    print('Transformación completada')

def train():
    print('Entrenando modelo...')
    df = pd.read_parquet('/tmp/ventas_procesadas.parquet')
    X = df[['precio', 'descuento', 'inventario']]
    y = df['demanda']
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    import joblib
    joblib.dump(model, '/models/modelo_demanda.pkl')
    print('Entrenamiento completado')

def evaluate():
    print('Evaluando modelo...')
    import joblib
    model = joblib.load('/models/modelo_demanda.pkl')
    df = pd.read_parquet('/tmp/ventas_procesadas.parquet')
    X = df[['precio', 'descuento', 'inventario']]
    y = df['demanda']
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y, model.predict(X))
    print(f'MSE: {mse:.4f}')
    # Guardar métricas
    with open('/metrics/metrics.json', 'w') as f:
        import json
        json.dump({'mse': mse, 'date': str(datetime.now())}, f)

def deploy():
    print('Desplegando modelo a producción...')
    # Simular deploy
    print('Modelo desplegado en FastAPI + Kubernetes')

extract_task = PythonOperator(task_id='extract', python_callable=extract, dag=dag)
transform_task = PythonOperator(task_id='transform', python_callable=transform, dag=dag)
train_task = PythonOperator(task_id='train', python_callable=train, dag=dag)
evaluate_task = PythonOperator(task_id='evaluate', python_callable=evaluate, dag=dag)
deploy_task = PythonOperator(task_id='deploy', python_callable=deploy, dag=dag)

extract_task >> transform_task >> train_task >> evaluate_task >> deploy_task
"""

with open('dags/ventas_pipeline.py', 'w') as f:
    f.write(airflow_dag_full)
print("DAG de Airflow diario creado en dags/ventas_pipeline.py")
```

### Ejemplo 6: Airflow — PythonOperator para cada paso con retry

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 6: Airflow — PythonOperator para cada paso con retry

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
# Simular operadores con retry
def extract_with_retry(**context):
    n_try = context['ti'].try_number
    print(f"Intento {n_try} de extracción...")
    if np.random.random() < 0.3:  # 30% de fallo simulado
        raise Exception("Error de conexión a base de datos")
    print("Extracción exitosa")
    return {'records': 15000, 'source': 'postgresql'}

def validate_with_retry(**context):
    ti = context['ti']
    extracted = ti.xcom_pull(task_ids='extract')
    print(f"Validando {extracted.get('records', 0)} registros...")
    return {'valid': True, 'invalid': 0}

print("Operadores con retry configurados:")
print("  extract -> retry=3, retry_delay=5min")
print("  validate -> retry=2, retry_delay=2min")
print("  train -> retry=1, retry_delay=10min")
```

### Ejemplo 7: Prefect — Flow de feature engineering + training

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 7: Prefect — Flow de feature engineering + training

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner

@task(retries=2, retry_delay_seconds=60, timeout_seconds=300)
def extract_data():
    print("Extrayendo datos de ventas...")
    return pd.DataFrame(np.random.randn(5000, 5))

@task
def create_features(df):
    print("Creando features...")
    df.columns = ['precio', 'descuento', 'inventario', 'rating', 'demanda']
    df['precio_x_descuento'] = df['precio'] * df['descuento']
    df['inventario_rating'] = df['inventario'] * df['rating']
    return df

@task(cache_result_in_memory=True)
def train_model(df):
    print("Entrenando modelo...")
    X = df.drop('demanda', axis=1)
    y = df['demanda']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

@task
def evaluate_model(model, df):
    print("Evaluando modelo...")
    X = df.drop('demanda', axis=1)
    y = df['demanda']
    mse = ((model.predict(X) - y) ** 2).mean()
    return mse

@flow(name='ventas_feature_training')
def ventas_pipeline():
    data = extract_data()
    features = create_features(data)
    model = train_model(features)
    mse = evaluate_model(model, features)
    print(f"Flow completado - MSE: {mse:.4f}")
    return model, mse

# Ejecutar
results = ventas_pipeline()
print(f"Pipeline Prefect ejecutado. Resultados: {results[1]:.4f}")
```

### Ejemplo 8: Feast — Feature view con ventas diarias como features

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 8: Feast — Feature view con ventas diarias como features

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
try:
    from feast import FeatureView, Feature, Entity, FileSource, ValueType
    from feast.infra.offline_stores.file_source import FileSource
    import datetime

    # Definir entidad
    producto = Entity(
        name='producto_id',
        value_type=ValueType.INT64,
        description='ID del producto',
    )

    # Fuente batch
    ventas_batch_source = FileSource(
        path='data/ventas_diarias.parquet',
        event_timestamp_column='fecha',
        created_timestamp_column='created_at',
    )

    # Feature View
    ventas_feature_view = FeatureView(
        name='ventas_diarias_features',
        entities=[producto],
        ttl=datetime.timedelta(days=30),
        features=[
            Feature(name='ventas_7d', dtype=ValueType.FLOAT),
            Feature(name='ventas_30d', dtype=ValueType.FLOAT),
            Feature(name='precio_promedio_7d', dtype=ValueType.FLOAT),
            Feature(name='precio_promedio_30d', dtype=ValueType.FLOAT),
            Feature(name='descuento_promedio_7d', dtype=ValueType.FLOAT),
            Feature(name='inventario_promedio_7d', dtype=ValueType.FLOAT),
            Feature(name='rating_promedio', dtype=ValueType.FLOAT),
            Feature(name='n_ventas_7d', dtype=ValueType.INT64),
        ],
        online=True,
        batch_source=ventas_batch_source,
        tags={'team': 'datascience', 'project': 'ventas_prediccion'},
    )
    print("Feature View 'ventas_diarias_features' definido con 8 features")
    print(f"Entidad: producto_id")
    print(f"TTL: 30 days")
except ImportError:
    print("Feast no instalado. Definición conceptual creada.")
```

### Ejemplo 9: Feast — Materialize features a online store para inferencia

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 9: Feast — Materialize features a online store para inferencia

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
# Comandos Feast (simulados)
print("Comandos Feast para materialización:")
print("  feast apply                    # Aplicar configuración")
print("  feast materialize 2024-01-01 2024-06-15  # Materializar a online store")
print("  feast materialize-incremental  # Materialización incremental diaria")

# Feature retrieval simulado
def get_online_features(producto_ids, features):
    # Simular recuperación de features
    results = {}
    for pid in producto_ids:
        results[pid] = {
            'ventas_7d': np.random.poisson(100),
            'ventas_30d': np.random.poisson(500),
            'precio_promedio_7d': np.random.uniform(50, 150),
            'inventario_promedio_7d': np.random.poisson(200),
        }
    return results

features_online = get_online_features([101, 102, 103], ['ventas_7d', 'precio_promedio_7d'])
print("\nFeatures online recuperadas:")
for pid, feats in features_online.items():
    print(f"  Producto {pid}: {feats}")
```

### Ejemplo 10: MLflow registry — Promote model to production

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 10: MLflow registry — Promote model to production

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
import mlflow
from mlflow.tracking import MlflowClient

# Configurar tracking URI
mlflow.set_tracking_uri('sqlite:///mlflow.db')

# Simular registro de modelo
with mlflow.start_run(run_name='production_promote') as run:
    mlflow.log_param('n_estimators', 150)
    mlflow.log_metric('mse', 1.05)
    run_id = run.info.run_id

# Registrar modelo
model_uri = f'runs:/{run_id}/model'
try:
    result = mlflow.register_model(model_uri, 'VentasPrediccion')
    print(f"Modelo registrado: {result.name} v{result.version}")
except Exception:
    print("Modelo ya registrado, creando nueva versión")
    result = mlflow.register_model(model_uri, 'VentasPrediccion')

# Promover a Production
client = MlflowClient()
try:
    client.transition_model_version_stage(
        name='VentasPrediccion',
        version=result.version,
        stage='Production'
    )
    print(f"✅ Modelo v{result.version} promovido a Production")
    
    # Verificar
    mv = client.get_model_version('VentasPrediccion', result.version)
    print(f"Stage actual: {mv.current_stage}")
except Exception as e:
    print(f"Error: {e}")
```

### Ejemplo 11: Kubernetes — deployment.yaml para API

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 11: Kubernetes — deployment.yaml para API

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
k8s_deployment = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-ventas
  namespace: ml-production
  labels:
    app: api-ventas
    version: "2.0"
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: api-ventas
  template:
    metadata:
      labels:
        app: api-ventas
    spec:
      containers:
      - name: api-ventas
        image: registry.example.com/api-ventas:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: "/models/modelo_demanda.pkl"
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: redis-credentials
              key: url
        resources:
          requests:
            cpu: 500m
            memory: 512Mi
          limits:
            cpu: 2000m
            memory: 2Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 15
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
        volumeMounts:
        - name: model-storage
          mountPath: /models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: models-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: api-ventas-service
  namespace: ml-production
spec:
  selector:
    app: api-ventas
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
"""

with open('k8s/deployment.yaml', 'w') as f:
    f.write(k8s_deployment)
print("Kubernetes Deployment + Service creados")
```

### Ejemplo 12: Kubernetes — Horizontal Pod Autoscaler (HPA)

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 12: Kubernetes — Horizontal Pod Autoscaler (HPA)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
k8s_hpa = """
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-ventas-hpa
  namespace: ml-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-ventas
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Pods
        value: 2
        periodSeconds: 30
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Pods
        value: 1
        periodSeconds: 60
"""

with open('k8s/hpa.yaml', 'w') as f:
    f.write(k8s_hpa)
print("HPA creado: 2-10 réplicas, CPU 70%, Memory 80%")
```

### Ejemplo 13: Triton — Model config para modelo de demanda

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 13: Triton — Model config para modelo de demanda

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
triton_config = """
name: "demanda_model"
platform: "onnxruntime_onnx"
max_batch_size: 1024
input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [5]
  }
]
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [1]
  }
]
dynamic_batching {
  preferred_batch_size: [1, 4, 8, 16, 32, 64, 128, 256]
  max_queue_delay_microseconds: 100
}
instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [0]
  }
]
"""

with open('triton_model_config.pbtxt', 'w') as f:
    f.write(triton_config)

print("Configuración de Triton Inference Server:")
print("  Modelo: demanda_model (ONNX)")
print("  Batch: dinámico hasta 1024")
print("  GPU: 2 instancias en GPU 0")
```

### Ejemplo 14: TF Serving — SavedModel + REST API

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 14: TF Serving — SavedModel + REST API

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
# Guardar modelo como SavedModel
model_tf = Sequential([
    Dense(64, activation='relu', input_shape=(5,)),
    Dense(1)
])
model_tf.compile(optimizer='adam', loss='mse')
model_tf.fit(np.random.randn(1000, 5), np.random.randn(1000), epochs=3, verbose=0)

model_path = 'saved_models/demanda_model/1'
tf.saved_model.save(model_tf, model_path)
print(f"SavedModel guardado en {model_path}")

# Configuración de TF Serving
tf_serving_config = """
model_config_list {
  config {
    name: 'demanda_model'
    base_path: '/models/demanda_model'
    model_platform: 'tensorflow'
    model_version_policy {
      specific {
        versions: 1
        versions: 2
      }
    }
  }
}
batching_parameters {
  max_batch_size: 256
  batch_timeout_micros: 1000
  max_enqueued_batches: 1000
}
"""

with open('tf_serving/models.config', 'w') as f:
    f.write(tf_serving_config)
print("TF Serving config creada")

# Inferencia REST
print("\nInferencia con TF Serving REST API:")
print("  curl -X POST http://localhost:8501/v1/models/demanda_model:predict \\")
print('    -d \'{"instances": [[100, 0.1, 500, 3, 7]]}\'')
```

### Ejemplo 15: Prometheus — Métricas de latency, error rate, drift

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 15: Prometheus — Métricas de latency, error rate, drift

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
# Simular endpoint de métricas Prometheus
prometheus_metrics = """
# HELP ml_latency_ms Latencia de predicciones en ms
# TYPE ml_latency_ms histogram
ml_latency_ms_bucket{le="10"} 1500
ml_latency_ms_bucket{le="50"} 4500
ml_latency_ms_bucket{le="100"} 4900
ml_latency_ms_bucket{le="500"} 5000
ml_latency_ms_bucket{le="+Inf"} 5000
ml_latency_ms_count 5000
ml_latency_ms_sum 98500

# HELP ml_errors_total Total de errores de predicción
# TYPE ml_errors_total counter
ml_errors_total{error_type="timeout"} 12
ml_errors_total{error_type="invalid_input"} 5
ml_errors_total{error_type="model_error"} 3

# HELP ml_drift_psi PSI por feature
# TYPE ml_drift_psi gauge
ml_drift_psi{feature="precio"} 0.15
ml_drift_psi{feature="descuento"} 0.08
ml_drift_psi{feature="inventario"} 0.22
ml_drift_psi{feature="rating"} 0.04

# HELP ml_predictions_total Total de predicciones
# TYPE ml_predictions_total counter
ml_predictions_total 50000

# HELP ml_model_mse MSE del modelo actual
# TYPE ml_model_mSE gauge
ml_model_mse 1.25
"""

with open('prometheus/metrics.prom', 'w') as f:
    f.write(prometheus_metrics)

print("Métricas Prometheus definidas:")
print("  - latency_ms: histograma de latencia")
print("  - errors_total: contador de errores por tipo")
print("  - drift_psi: gauge de PSI por feature")
print("  - predictions_total: contador de predicciones")
print("  - model_mse: gauge de MSE actual")
```

### Ejemplo 16: Grafana — Dashboard de performance del modelo

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 16: Grafana — Dashboard de performance del modelo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
grafana_dashboard = {
    "dashboard": {
        "title": "ML Model Performance - Ventas",
        "panels": [
            {
                "title": "Latencia de Predicciones (P50, P95, P99)",
                "type": "timeseries",
                "targets": [{"expr": "histogram_quantile(0.5, rate(ml_latency_ms_bucket[5m]))"}]
            },
            {
                "title": "Tasa de Error",
                "type": "stat",
                "targets": [{"expr": "rate(ml_errors_total[5m])"}]
            },
            {
                "title": "Data Drift (PSI por Feature)",
                "type": "bargauge",
                "targets": [{"expr": "ml_drift_psi"}]
            },
            {
                "title": "Predicciones por Minuto",
                "type": "timeseries",
                "targets": [{"expr": "rate(ml_predictions_total[1m])"}]
            },
            {
                "title": "MSE del Modelo",
                "type": "stat",
                "targets": [{"expr": "ml_model_mse"}]
            }
        ],
        "alerts": [
            {
                "alert": "High Latency",
                "condition": "ml_latency_ms_p99 > 500",
                "duration": "5m",
                "notifications": ["slack"]
            },
            {
                "alert": "Data Drift Detected",
                "condition": "ml_drift_psi{feature='inventario'} > 0.2",
                "duration": "10m",
                "notifications": ["email", "slack"]
            }
        ]
    }
}

import json
with open('grafana/dashboard.json', 'w') as f:
    json.dump(grafana_dashboard, f, indent=2)
print("Dashboard de Grafana creado con paneles de:")
print("  - Latencia (P50/P95/P99)")
print("  - Tasa de error")
print("  - Data drift por feature")
print("  - Predicciones/minuto")
print("  - MSE del modelo")
print("  - Alertas: alta latencia, drift detectado")
```

### Ejemplo 17: Retraining pipeline — Detect drift → retrain → evaluate → deploy

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 17: Retraining pipeline — Detect drift → retrain → evaluate → deploy

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
def retraining_pipeline():
    print("="*50)
    print("PIPELINE DE RE-ENTRENAMIENTO AUTOMÁTICO")
    print("="*50)
    
    # 1. Check drift
    print("\n1. VERIFICANDO DRIFT...")
    psi_scores = {
        'precio': np.random.uniform(0.05, 0.3),
        'descuento': np.random.uniform(0.02, 0.15),
        'inventario': np.random.uniform(0.08, 0.25),
        'rating': np.random.uniform(0.01, 0.1)
    }
    n_drifted = sum(1 for v in psi_scores.values() if v > 0.2)
    print(f"  Features con drift: {n_drifted}/4")
    
    if n_drifted < 2:
        print("  ✅ Sin drift significativo. No requiere re-entrenamiento.")
        return
    
    # 2. Retrain
    print("\n2. RE-ENTRENANDO MODELO...")
    print("  → Cargando últimos 30 días de datos")
    print("  → Entrenando nuevo modelo con datos actualizados")
    new_mse = np.random.uniform(0.9, 1.3)
    print(f"  → Nuevo MSE: {new_mse:.4f}")
    
    # 3. Evaluate
    print("\n3. EVALUANDO MODELO...")
    current_mse = 1.15
    improvement = (current_mse - new_mse) / current_mse * 100
    print(f"  → MSE actual: {current_mse:.4f}")
    print(f"  → MSE nuevo: {new_mse:.4f}")
    print(f"  → Mejora: {improvement:.1f}%")
    
    # 4. Deploy
    print("\n4. DESPLEGANDO...")
    if improvement > 5:
        print("  ✅ Mejora > 5%: promoviendo a producción")
        print("  → Nuevo modelo registrado en MLflow Registry")
        print("  → Blue-Green deploy sin downtime")
        print("  → Health check: OK")
    elif improvement > 0:
        print("  ⚠️ Mejora marginal: pasa a staging para evaluación")
        print("  → Modelo registrado en Staging")
    else:
        print("  ❌ Modelo no mejora: manteniendo versión actual")
    
    print(f"\n{'='*50}")
    print("PIPELINE COMPLETADO")
    print(f"{'='*50}")

retraining_pipeline()
```

### Ejemplo 18: Integrador — Pipeline end-to-end de MLOps completo

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 18: Integrador — Pipeline end-to-end de MLOps completo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
def mlops_end_to_end():
    print("="*60)
    print("MLOps COMPLETO - PIPELINE END-TO-END")
    print("="*60)
    
    # Fase 1: Data Ingestion
    print("\n📥 1. INGESTA DE DATOS")
    print("   • Airflow DAG: daily_ventas_pipeline")
    print("   • Extracción desde PostgreSQL (ventas_db)")
    print("   • 15,000 registros/día de 500 productos")
    print("   • Validación con pandera schema")
    
    # Fase 2: Feature Engineering
    print("\n🔧 2. FEATURE ENGINEERING")
    print("   • Prefect flow: create_features")
    print("   • Feast FeatureView: ventas_diarias_features")
    print("   • Features: ventas_7d, ventas_30d, precio_promedio, etc.")
    print("   • Materialización a online store (Redis)")
    
    # Fase 3: Training
    print("\n🤖 3. ENTRENAMIENTO")
    print("   • MLflow experiment: ventas_prediccion")
    print("   • Modelo: XGBoost + Optuna hyperparameter tuning")
    print("   • 100 trials con HyperbandPruner")
    print("   • Best params guardados en MLflow")
    
    # Fase 4: Validation
    print("\n✅ 4. VALIDACIÓN")
    print("   • Data quality: schema, missing values, ranges")
    print("   • Model: accuracy > 0.80, MSE < 1.5")
    print("   • Slice evaluation: performance por categoría")
    print("   • Bias test: performance por segmento de precio")
    
    # Fase 5: Registry
    print("\n📦 5. MODEL REGISTRY")
    print("   • MLflow Model Registry: VentasPrediccion v3")
    print("   • Stage: Staging (tests) → Production (live)")
    print("   • Champion-Challenger: v2 (champion) vs v3 (challenger)")
    
    # Fase 6: Deploy
    print("\n🚀 6. DESPLIEGUE")
    print("   • GitHub Actions: CI/CD pipeline")
    print("   • Docker build → push → K8s deploy")
    print("   • Blue-Green strategy: zero downtime")
    print("   • HPA: 2-10 réplicas, CPU 70% target")
    print("   • Triton Inference Server: GPU inference")
    
    # Fase 7: Serving
    print("\n⚡ 7. SERVING")
    print("   • FastAPI: /predict, /health, /predict/batch")
    print("   • REST API con autenticación API Key")
    print("   • Rate limiting: 1000 req/min por API key")
    print("   • Swagger docs en /docs")
    
    # Fase 8: Monitoring
    print("\n📊 8. MONITOREO")
    print("   • Evidently AI: data drift diario")
    print("   • Prometheus: latency_histogram, error_counter")
    print("   • Grafana: dashboard ejecutivo")
    print("   • Alertas: Slack si PSI > 0.2 o latencia > 500ms")
    
    # Fase 9: Retraining
    print("\n🔄 9. RE-ENTRENAMIENTO")
    print("   • Airflow DAG semanal: check drift → retrain → evaluate → deploy")
    print("   • Trigger: 2+ features con PSI > 0.2")
    print("   • Condición: mejora > 5% para promover")
    
    # Fase 10: Governance
    print("\n📋 10. GOVERNANZA")
    print("    • MLflow: tracking de todos los experimentos")
    print("    • DVC: versionado de datos y pipelines")
    print("    • Audit trail: quién, cuándo, qué modelo")
    print("    • Reproducibilidad: mismo commit + data = mismos resultados")
    
    print("\n" + "="*60)
    print("✅ MLOPS COMPLETO - CICLO CERRADO")
    print("="*60)
    print("\nComandos útiles:")
    print("  kubectl get pods -n ml-production")
    print("  kubectl logs deployment/api-ventas")
    print("  kubectl get hpa -n ml-production")
    print("  mlflow ui")
    print("  airflow dags list")
    print("  feast ui")

mlops_end_to_end()
```

---

## 3. Ejercicios Propuestos

1. **GitHub Actions completo para ML**: Crea un workflow con 3 jobs: (1) lint + test + data validation, (2) build docker + push a registry, (3) deploy a staging. Incluye matrix testing para Python 3.9, 3.10, 3.11.

2. **Airflow DAG complejo con branching**: Crea un DAG que: (1) extraiga datos, (2) verifique drift (BranchPythonOperator), (3) si hay drift → re-entrena, si no → salta entrenamiento, (4) despliega si mejora. Incluye SlackWebhookOperator para notificaciones.

3. **Feature store con Feast para ventas en tiempo real**: Define 2 entidades (producto_id, tienda_id), 3 feature views (ventas_7d, precio_promedio, stock_actual), y muestra cómo obtener online features para inferencia con Redis como online store.

4. **Kubernetes con ConfigMap y Secret**: Crea ConfigMap para configuración del modelo (umbrales, paths) y Secret para credenciales de base de datos. Modifica el deployment.yaml para usarlos como variables de entorno.

5. **Triton con ensemble de modelos**: Crea un model config para Triton que ensamble 3 modelos: (1) feature encoder, (2) demanda predictor, (3) post-processor. Configura dynamic batching y pipeline de ensamble.

6. **Prometheus custom metrics en FastAPI**: Implementa métricas Prometheus personalizadas en la API FastAPI: request_count, latency_histogram, prediction_value, model_version. Expón en /metrics.

7. **Grafana alerta con Slack**: Crea una alerta en Grafana que se dispare cuando: (1) P99 latency > 1s por 5 minutos, (2) error rate > 5% en 10 minutos. Configura notificación a Slack. Incluye el payload del mensaje.

8. **Pipeline de re-entrenamiento con A/B test**: Implementa un pipeline que al re-entrenar: (1) despliegue challenger en un slot separado, (2) enrute 10% de tráfico a challenger por 24h, (3) compare MSE en producción, (4) si challenger es mejor 5% → promueve a champion.

---

## 4. Resumen

| Componente | Herramienta | Propósito |
|---|---|---|
| **CI/CD** | GitHub Actions | Build, test, deploy automático |
| **Orquestación** | Airflow / Prefect | Pipelines de datos y entrenamiento |
| **Feature Store** | Feast | Features compartidas y consistentes |
| **Model Registry** | MLflow | Versionado y staging de modelos |
| **Deploy** | FastAPI + Docker + K8s | Serving escalable y resiliente |
| **Inferencia** | Triton / TF Serving | GPU serving optimizado |
| **Monitoreo** | Prometheus + Grafana | Métricas, dashboards, alertas |
| **Drift** | Evidently AI | Detección de data/concept drift |

El MLOps end-to-end cierra el ciclo: datos → features → entrenamiento → validación → deploy → monitoreo → re-entrenamiento. Cada componente es esencial para mantener modelos de ventas precisos, escalables y confiables en producción.
