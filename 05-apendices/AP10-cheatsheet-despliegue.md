# AP10 — Cheatsheet Despliegue de Modelos ML

## 1. FastAPI — API REST

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Cargar modelo
model = joblib.load("modelo_ventas.pkl")

# App
app = FastAPI(title="API de Predicción de Ventas", version="1.0.0")

# Schema de entrada
class InputData(BaseModel):
    precio: float
    descuento: float
    stock: int
    categoria: str

class PredictionOut(BaseModel):
    prediccion: float
    probabilidad: float = None

# Endpoints
@app.get("/")
def root():
    return {"message": "API de Predicción de Ventas", "status": "ok"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionOut)
def predict(data: InputData):
    try:
        features = np.array([[data.precio, data.descuento,
                              data.stock]])
        pred = model.predict(features)[0]
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0][1]
        return PredictionOut(prediccion=float(pred), probabilidad=float(proba))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Ejecutar: uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*1. FastAPI — API REST.*

1. Cargar modelo
2. App
3. Schema de entrada
4. Endpoints
5. Ejecutar: uvicorn main:app --reload --host 0.0.0.0 --port 8000

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Validación y async
```python
from fastapi import File, UploadFile
from typing import List
import pandas as pd
import io

# Batch prediction
@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    predictions = model.predict(df)
    df["prediccion"] = predictions
    return df.to_dict(orient="records")
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Validación y async.*

1. Batch prediction

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 2. Docker

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*2. Docker.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

yaml
# docker-compose.yml
version: "3.8"
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - MODEL_PATH=/app/models/modelo_ventas.pkl
```

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

bash
# Comandos útiles
docker build -t ventas-api .
docker run -p 8000:8000 ventas-api
docker-compose up -d
docker ps
docker logs ventas-api
```

## 3. MLflow — Tracking

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 3. MLflow — Tracking

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Configurar tracking
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Prediccion_Ventas")

with mlflow.start_run():
    # Parámetros
    params = {"n_estimators": 100, "max_depth": 10}
    mlflow.log_params(params)

    # Entrenar
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    # Métricas
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mlflow.log_metric("rmse", rmse)

    # Guardar modelo
    mlflow.sklearn.log_model(model, "modelo_ventas")

    # Artefactos
    mlflow.log_artifact("features.csv")
    mlflow.log_artifact("training.log")

    # Tags
    mlflow.set_tag("equipo", "data-science")
    mlflow.set_tag("tipo", "regresion")

# Cargar modelo desde MLflow
model_uri = "runs:/<RUN_ID>/modelo_ventas"
model = mlflow.sklearn.load_model(model_uri)

# Búsqueda de runs
runs = mlflow.search_runs(
    experiment_names=["Prediccion_Ventas"],
    order_by=["metrics.rmse ASC"]
)
best_run = runs.iloc[0]
```

### MLflow Model Registry
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. MLflow Model Registry

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.register_model(
    model_uri="runs:/<RUN_ID>/modelo_ventas",
    name="PrediccionVentas"
)

# Etapas: Staging, Production, Archived
client.transition_model_version_stage(
    name="PrediccionVentas",
    version=1,
    stage="Production"
)
```

## 4. ONNX — Exportación

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 4. ONNX — Exportación

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
import onnx
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Convertir sklearn a ONNX
initial_type = [("float_input", FloatTensorType([None, X.shape[1]]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

# Guardar
with open("modelo.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# Inferencia con ONNX Runtime
session = ort.InferenceSession("modelo.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
result = session.run([output_name], {input_name: X_test.astype(np.float32)})

# Convertir PyTorch a ONNX
import torch
dummy_input = torch.randn(1, X.shape[1])
torch.onnx.export(model, dummy_input, "modelo_pytorch.onnx")

# Convertir TF/Keras a ONNX
import tf2onnx
model_proto, _ = tf2onnx.convert.from_keras(model)
```

## 5. TensorRT — Optimización

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 5. TensorRT — Optimización

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
import tensorrt as trt

# TensorRT para inferencia acelerada en GPU
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network()
config = builder.create_builder_config()

# Parse ONNX
parser = trt.OnnxParser(network, TRT_LOGGER)
with open("modelo.onnx", "rb") as f:
    parser.parse(f.read())

# Optimizar
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20)
serialized_engine = builder.build_serialized_network(network, config)

# Guardar
with open("modelo.trt", "wb") as f:
    f.write(serialized_engine)

# Cargar e inferir
runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(serialized_engine)
context = engine.create_execution_context()

# FP16 / INT8 quantization
config.set_flag(trt.BuilderFlag.FP16)
# config.set_flag(trt.BuilderFlag.INT8)
```

## 6. Evidently — Monitoreo

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 6. Evidently — Monitoreo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently import ColumnMapping
import pandas as pd

# Referencia (datos de entrenamiento) y datos actuales
reference = pd.read_csv("datos_entrenamiento.csv")
current = pd.read_csv("datos_produccion.csv")

# Column mapping
column_mapping = ColumnMapping(
    target="ventas",
    prediction="prediccion",
    numerical_features=["precio", "descuento", "stock"],
    categorical_features=["categoria"]
)

# Data Drift
drift_report = Report(metrics=[DataDriftPreset()])
drift_report.run(reference_data=reference, current_data=current,
                 column_mapping=column_mapping)
drift_report.save_html("data_drift.html")

# Performance del modelo
perf_report = Report(metrics=[RegressionPreset()])
perf_report.run(reference_data=reference, current_data=current,
                column_mapping=column_mapping)
perf_report.save_html("model_performance.html")

# Drift por características
from evidently.test_suite import TestSuite
from evidently.tests import TestColumnDrift

drift_tests = TestSuite(tests=[TestColumnDrift(column_name="precio")])
drift_tests.run(reference_data=reference, current_data=current)
```

## 7. SHAP — Explicabilidad

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 7. SHAP — Explicabilidad

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
import shap

# SHAP para modelos sklearn
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Waterfall (explicación individual)
shap.plots.waterfall(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_test[0],
    feature_names=feature_names
))

# Dependence plot
shap.dependence_plot("precio", shap_values, X_test)

# SHAP para DL (DeepExplainer)
import tensorflow as tf
explainer_dl = shap.DeepExplainer(model, X_train[:100])
shap_values_dl = explainer_dl.shap_values(X_test[:10])

# SHAP en producción (servir)
# Guardar explainer
import joblib
joblib.dump(explainer, "shap_explainer.pkl")

# Endpoint FastAPI
@app.post("/explain")
def explain(data: InputData):
    features = np.array([...])
    shap_values = explainer.shap_values(features)
    return {
        "prediction": float(model.predict(features)[0]),
        "shap_values": shap_values.tolist(),
        "base_value": float(explainer.expected_value)
    }
```

## 8. Hyperopt — Optimización

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 8. Hyperopt — Optimización

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

# Espacio de búsqueda
space = {
    "n_estimators": hp.choice("n_estimators", [50, 100, 200, 300]),
    "max_depth": hp.quniform("max_depth", 3, 20, 1),
    "min_samples_leaf": hp.choice("min_samples_leaf", [1, 2, 5, 10]),
    "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.3))
}

# Función objetivo
def objective(params):
    params["max_depth"] = int(params["max_depth"])
    model = GradientBoostingRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return {"loss": rmse, "status": STATUS_OK}

# Optimizar
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest,
            max_evals=100, trials=trials, rstate=np.random.default_rng(42))

print("Mejores parámetros:", best)
```

## 9. BentoML — Serving

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 9. BentoML — Serving

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
import bentoml
from bentoml.io import JSON, NumpyNdarray

# Guardar modelo con BentoML
bentoml.sklearn.save_model("ventas_rf", model)

# Service definition (service.py)
import bentoml
from bentoml.io import JSON
import numpy as np

model_ref = bentoml.sklearn.get("ventas_rf:latest")
model = model_ref.load_model()

svc = bentoml.Service("ventas_service", runners=[])

@svc.api(input=JSON(), output=JSON())
def predict(input_data):
    features = np.array([[
        input_data["precio"],
        input_data["descuento"],
        input_data["stock"]
    ]])
    prediction = model.predict(features)[0]
    return {"prediccion": float(prediction)}

# Ejecutar: bentoml serve service.py:svc --reload
```

## 10. Kubernetes — Orquestación

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 10. Kubernetes — Orquestación

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ventas-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ventas-api
  template:
    metadata:
      labels:
        app: ventas-api
    spec:
      containers:
      - name: api
        image: ventas-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "1"
---
apiVersion: v1
kind: Service
metadata:
  name: ventas-api-service
spec:
  selector:
    app: ventas-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

bash
kubectl apply -f deployment.yaml
kubectl get pods
kubectl logs -f pod/ventas-api-xxxx
kubectl scale deployment ventas-api --replicas=5
```

## 11. CI/CD para ML

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 11. CI/CD para ML

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline
on: [push]
jobs:
  train-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: "3.10"
      - run: pip install -r requirements.txt
      - run: python train.py
      - run: python evaluate.py
      - name: Deploy to Production
        if: github.ref == 'refs/heads/main'
        run: |
          docker build -t ventas-api .
          docker push registry.example.com/ventas-api
```

## 12. Feature Stores

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 12. Feature Stores

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
# Feast (Feature Store)
from feast import FeatureStore

store = FeatureStore(repo_path="./feature_repo")

# Obtener features para entrenamiento
feature_vector = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "ventas_ds:precio",
        "ventas_ds:descuento",
        "ventas_ds:ventas_rolling_7d"
    ]
).to_df()

# Obtener features para inferencia online
features = store.get_online_features(
    features=[
        "ventas_ds:precio",
        "ventas_ds:descuento",
    ],
    entity_rows=[{"producto_id": "laptop-123"}]
).to_dict()
```

## 13. Monitoreo de Modelos

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 13. Monitoreo de Modelos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
# Prometheus métricas
from prometheus_client import Counter, Histogram, generate_latest
import time

PREDICTIONS = Counter("model_predictions_total", "Total predictions", ["model"])
PREDICTION_TIME = Histogram("prediction_seconds", "Prediction latency", buckets=[0.01, 0.05, 0.1, 0.5])

@app.post("/predict")
def predict(data: InputData):
    start = time.time()
    pred = model.predict([...])
    PREDICTION_TIME.observe(time.time() - start)
    PREDICTIONS.labels(model="ventas_rf").inc()
    return {"prediccion": float(pred)}

# /metrics endpoint for Prometheus
@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type="text/plain")
```

## 14. A/B Testing en Producción

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 14. A/B Testing en Producción

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
import random

@app.post("/predict")
def predict(data: InputData):
    # A/B test: 50% modelo A, 50% modelo B
    if random.random() < 0.5:
        pred = model_a.predict(features)[0]
        model_used = "model_a"
    else:
        pred = model_b.predict(features)[0]
        model_used = "model_b"

    # Log para análisis
    log_ab_test(data, pred, model_used)
    return {"prediccion": float(pred), "model": model_used}
```

## 15. Data Version Control (DVC)

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 15. Data Version Control (DVC)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

bash
dvc init
dvc add datos/ventas.csv
git add datos/ventas.csv.dvc .gitignore
git commit -m "Add data"

dvc remote add minio s3://ml-bucket/datavc
dvc push

dvc repro        # reproducir pipeline
dvc metrics show # ver métricas
dvc diff         # diferencias entre versiones
```

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
# dvc.yaml
stages:
  train:
    cmd: python train.py
    deps:
      - datos/ventas.csv
      - train.py
    outs:
      - modelos/modelo.pkl
    metrics:
      - metrics.json
```

## Referencia Rápida

| Herramienta | Uso | Comando/Ejemplo |
|-------------|-----|-----------------|
| FastAPI | API REST | `uvicorn main:app --reload` |
| Docker | Contenedor | `docker build -t api . && docker run api` |
| MLflow | Tracking | `mlflow ui`, `mlflow.log_metric()`, `mlflow.log_model()` |
| ONNX | Exportación | `convert_sklearn(model, initial_types=...)` |
| TensorRT | Optimización GPU | `builder.build_serialized_network()` |
| Evidently | Monitoreo | `Report(metrics=[DataDriftPreset()])` |
| SHAP | Explicabilidad | `shap.TreeExplainer(model)` |
| Hyperopt | Optimización | `fmin(objective, space, algo=tpe.suggest)` |
| BentoML | Serving | `bentoml serve service.py:svc` |
| Kubernetes | Orquestación | `kubectl apply -f deployment.yaml` |
| Prometheus | Métricas | `Counter, Histogram`, `/metrics` endpoint |
| DVC | Data versioning | `dvc add`, `dvc push`, `dvc repro` |
