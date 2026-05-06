# E21: Despliegue con FastAPI — API para Modelos de Ventas en Producción

## Objetivo
Implementar APIs REST con FastAPI para servir modelos ML de predicción de demanda, clasificación de productos y optimización de inventarios en producción.

---

## 1. Fundamentos Teóricos

### 1.1 FastAPI
FastAPI es un framework web moderno y rápido para construir APIs con Python:
- **Basado en Starlette** (asincrónico) y **Pydantic** (validación de datos)
- **Documentación automática**: Swagger UI (/docs) y ReDoc (/redoc)
- **Async/await**: Soporte nativo para operaciones asíncronas
- **Tipado**: Validación automática basada en type hints

```python
app = FastAPI(title='API Ventas', description='Predicción de demanda', version='1.0.0')
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*1.1 FastAPI.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### 1.2 Endpoints Principales
- **GET /health**: Health check del servicio y modelo
- **POST /predict**: Endpoint de predicción
- **GET /**: Información de la API
- **on_event('startup')**: Cargar modelo al iniciar
- **on_event('shutdown')**: Liberar recursos al detener

### 1.3 Pydantic Models
```python
class PredictionRequest(BaseModel):
    precio: float = Field(gt=0, le=10000, description='Precio del producto')
    descuento: float = Field(ge=0, le=1, description='Descuento aplicado')
    inventario: int = Field(ge=0, description='Unidades en inventario')

    @field_validator('precio')
    def precio_no_negativo(cls, v):
        if v <= 0:
            raise ValueError('El precio debe ser positivo')
        return v
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*1.3 Pydantic Models.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### 1.4 Middleware
- **CORSMiddleware**: Permitir requests desde frontend
- **TrustedHostMiddleware**: Restringir hosts permitidos
- **GZipMiddleware**: Compresión de respuestas

### 1.5 Rate Limiting
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
@app.post('/predict')
@limiter.limit('100/minute')
async def predict(request: PredictionRequest):
    ...
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*1.5 Rate Limiting.*

1. `from slowapi import Limiter, _rate_limit_exceeded_handler` — Importa las librerías necesarias para el análisis.
2. `async def predict(request: PredictionRequest):` — Genera predicciones sobre nuevos datos.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### 1.6 Autenticación
- **APIKey**: API key en header `X-API-Key`
- **OAuth2**: OAuth2 con JWT Bearer tokens
- **HTTPBasic**: Basic auth para endpoints internos

### 1.7 Docker y docker-compose
```dockerfile
FROM python:3.10-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 1.8 Testing con TestClient
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*1.7 Docker y docker-compose.*

1. 1.8 Testing con TestClient

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
from fastapi.testclient import TestClient
client = TestClient(app)
response = client.post('/predict', json={'precio': 100, 'descuento': 0.1, ...})
assert response.status_code == 200
```

---

## 2. Ejemplos Prácticos

### Ejemplo 1: FastAPI app básica con endpoint /health

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 2. Ejemplos Prácticos
2. Ejemplo 1: FastAPI app básica con endpoint /health

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
from fastapi import FastAPI
import pandas as pd
import numpy as np

app = FastAPI(
    title='API Predicción de Demanda',
    description='API para servir modelos de predicción de ventas',
    version='1.0.0'
)

@app.get('/health')
async def health():
    return {
        'status': 'healthy',
        'model_loaded': True,
        'last_training': '2024-06-15',
        'version': '1.0.0'
    }

@app.get('/')
async def root():
    return {
        'message': 'API de Predicción de Ventas',
        'docs': '/docs',
        'redoc': '/redoc'
    }
```

### Ejemplo 2: Pydantic PredictionRequest con campos y validación

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 2: Pydantic PredictionRequest con campos y validación

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
from pydantic import BaseModel, Field, field_validator
from typing import Optional

class PredictionRequest(BaseModel):
    precio: float = Field(gt=0, le=10000, description='Precio del producto en USD')
    descuento: float = Field(ge=0, le=1, description='Descuento (0-1)')
    inventario: int = Field(ge=0, description='Unidades disponibles en inventario')
    dia_semana: int = Field(ge=0, le=6, description='Día de la semana (0=domingo)')
    mes: int = Field(ge=1, le=12, description='Mes del año')
    categoria: Optional[str] = Field(None, description='Categoría del producto')

    @field_validator('precio')
    def precio_razonable(cls, v):
        if v > 10000:
            raise ValueError('Precio excede el máximo permitido')
        return v

    @field_validator('inventario')
    def inventario_no_negativo(cls, v):
        if v < 0:
            raise ValueError('Inventario no puede ser negativo')
        return v

# Ejemplo de uso
req = PredictionRequest(precio=150.0, descuento=0.15, inventario=500,
                        dia_semana=2, mes=6, categoria='electrónicos')
print(f"Request válido: {req.model_dump()}")
```

### Ejemplo 3: Endpoint /predict POST que recibe JSON y devuelve predicción

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 3: Endpoint /predict POST que recibe JSON y devuelve predicción

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
import pickle
from fastapi import HTTPException

# Modelo simulado (cargarías el real)
class ModeloDemanda:
    def predict(self, features):
        return np.random.uniform(50, 200)

modelo = ModeloDemanda()

class PredictionResponse(BaseModel):
    prediccion: float = Field(description='Demanda predicha en unidades')
    intervalo_inferior: float = Field(description='Límite inferior IC 95%')
    intervalo_superior: float = Field(description='Límite superior IC 95%')
    timestamp: str = Field(description='Timestamp de la predicción')

@app.post('/predict', response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        features = np.array([[
            request.precio, request.descuento, request.inventario,
            request.dia_semana, request.mes
        ]])
        pred = modelo.predict(features)
        return PredictionResponse(
            prediccion=float(pred),
            intervalo_inferior=float(pred * 0.9),
            intervalo_superior=float(pred * 1.1),
            timestamp=pd.Timestamp.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error en predicción: {str(e)}')
```

### Ejemplo 4: Cargar modelo ML al inicio (on_event startup)

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 4: Cargar modelo ML al inicio (on_event startup)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
import joblib
import os

MODEL_PATH = 'models/modelo_demanda.pkl'

# Simular modelo guardado
os.makedirs('models', exist_ok=True)
fake_model = {'coef': [0.5, -0.3, 0.2, 0.1, -0.05], 'intercept': 100}
joblib.dump(fake_model, MODEL_PATH)

class MLModel:
    def __init__(self):
        self.model = None
        self.metadata = {}

    def load(self, path):
        self.model = joblib.load(path)
        self.metadata = {
            'loaded_at': pd.Timestamp.now().isoformat(),
            'version': '1.0.0',
            'features': ['precio', 'descuento', 'inventario', 'dia_semana', 'mes']
        }
        return self

modelo_ml = MLModel()

@app.on_event('startup')
async def startup():
    if os.path.exists(MODEL_PATH):
        modelo_ml.load(MODEL_PATH)
        print(f"Modelo cargado desde {MODEL_PATH}")
    else:
        print("Advertencia: modelo no encontrado")

@app.on_event('shutdown')
async def shutdown():
    print("API shutting down, liberando recursos...")
    modelo_ml.model = None
```

### Ejemplo 5: POST request con curl/python para probar

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 5: POST request con curl/python para probar

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
import requests
import json

# Datos de ejemplo
payload = {
    'precio': 250.0,
    'descuento': 0.1,
    'inventario': 1000,
    'dia_semana': 3,
    'mes': 7,
    'categoria': 'electrónicos'
}

# Usando requests
try:
    response = requests.post(
        'http://localhost:8000/predict',
        json=payload,
        timeout=10
    )
    if response.status_code == 200:
        print(f"Predicción: {response.json()}")
    else:
        print(f"Error {response.status_code}: {response.text}")
except requests.exceptions.ConnectionError:
    print("Servidor no disponible. Comando curl equivalente:")
    print(f"curl -X POST http://localhost:8000/predict \\")
    print(f"  -H 'Content-Type: application/json' \\")
    print(f"  -d '{json.dumps(payload)}'")
```

### Ejemplo 6: Async endpoint con BackgroundTasks para logging

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 6: Async endpoint con BackgroundTasks para logging

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
from fastapi import BackgroundTasks
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_prediccion(request: PredictionRequest, prediccion: float, duracion: float):
    logger.info(
        f"Predicción | precio={request.precio} | desc={request.descuento} | "
        f"inv={request.inventario} | pred={prediccion:.2f} | dur={duracion:.3f}s"
    )

@app.post('/predict/async', response_model=PredictionResponse)
async def predict_async(request: PredictionRequest, background_tasks: BackgroundTasks):
    start = time.perf_counter()
    features = np.array([[request.precio, request.descuento, request.inventario,
                          request.dia_semana, request.mes]])
    pred = modelo.predict(features)
    duracion = time.perf_counter() - start
    background_tasks.add_task(log_prediccion, request, float(pred), duracion)
    return PredictionResponse(
        prediccion=float(pred),
        intervalo_inferior=float(pred * 0.9),
        intervalo_superior=float(pred * 1.1),
        timestamp=pd.Timestamp.now().isoformat()
    )
```

### Ejemplo 7: Logging estructurado con loguru

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 7: Logging estructurado con loguru

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
from loguru import logger
import sys

# Configurar loguru
logger.remove()
logger.add(sys.stdout, format="{time} | {level} | {message}")
logger.add("logs/api_{time:YYYY-MM-DD}.log", rotation="1 day", retention="30 days")

@app.get('/predict/{producto_id}')
async def predict_producto(producto_id: int):
    logger.info(f"Predicción solicitada para producto {producto_id}")
    try:
        pred = np.random.uniform(50, 200)
        logger.debug(f"Producto {producto_id}: predicción={pred:.2f}")
        return {'producto_id': producto_id, 'demanda_predicha': pred}
    except Exception as e:
        logger.error(f"Error en producto {producto_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

### Ejemplo 8: Manejo de errores — HTTPException con 404, 422, 500

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 8: Manejo de errores — HTTPException con 404, 422, 500

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

# Manejador global de excepciones
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=422,
        content={'error': 'Validation Error', 'detail': str(exc), 'path': str(request.url)}
    )

@app.exception_handler(Exception)
async def general_error_handler(request: Request, exc: Exception):
    logger.error(f"Error no manejado: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={'error': 'Internal Server Error', 'detail': 'Contacta al administrador'}
    )

@app.get('/productos/{producto_id}')
async def get_producto(producto_id: int):
    if producto_id <= 0:
        raise HTTPException(status_code=422, detail='ID de producto inválido')
    if producto_id > 1000:
        raise HTTPException(status_code=404, detail='Producto no encontrado')
    return {'producto_id': producto_id, 'nombre': f'Producto {producto_id}'}
```

### Ejemplo 9: CORS middleware para frontend

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 9: CORS middleware para frontend

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:3000', 'https://dashboard.ventas.com'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
    expose_headers=['X-Prediction-Time']
)

@app.get('/cors-test')
async def cors_test():
    return {'message': 'CORS configurado correctamente'}
```

### Ejemplo 10: Rate limiting — 100 requests/minuto por IP

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 10: Rate limiting — 100 requests/minuto por IP

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

@app.get('/predict/rapido')
@limiter.limit('100/minute')
async def predict_rapido(request: Request):
    return {'prediccion': np.random.uniform(50, 200)}

@app.get('/predict/premium')
@limiter.limit('1000/minute')
async def predict_premium(request: Request):
    return {'prediccion': np.random.uniform(50, 200)}
```

### Ejemplo 11: API Key authentication simple

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 11: API Key authentication simple

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
from fastapi.security import APIKeyHeader
from fastapi import Depends, Security

API_KEYS = {'sk-prod-abc123': 'produccion', 'sk-dev-xyz789': 'desarrollo'}
api_key_header = APIKeyHeader(name='X-API-Key', auto_error=True)

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key not in API_KEYS:
        raise HTTPException(status_code=403, detail='API Key inválida')
    return API_KEYS[api_key]

@app.get('/predict/secure')
async def predict_secure(env: str = Depends(verify_api_key)):
    return {
        'prediccion': np.random.uniform(50, 200),
        'environment': env,
        'message': 'Autenticado correctamente'
    }
```

### Ejemplo 12: Swagger docs en /docs (automático)

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 12: Swagger docs en /docs (automático)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
# FastAPI genera automáticamente Swagger UI en /docs
# y ReDoc en /redoc

app = FastAPI(
    title='API Predicción de Ventas',
    description='API para predicción de demanda, clasificación de productos y optimización de inventarios.',
    version='2.0.0',
    contact={
        'name': 'Equipo de Ciencia de Datos',
        'email': 'datascience@ventas.com',
    },
    license_info={
        'name': 'MIT',
        'url': 'https://opensource.org/licenses/MIT',
    },
    openapi_tags=[
        {'name': 'predicción', 'description': 'Endpoints de predicción de demanda'},
        {'name': 'health', 'description': 'Health check y monitoreo'},
        {'name': 'admin', 'description': 'Endpoints administrativos'}
    ]
)

@app.get('/health', tags=['health'])
async def health():
    return {'status': 'ok'}

@app.post('/predict', tags=['predicción'])
async def predict(req: PredictionRequest):
    return {'prediccion': np.random.uniform(50, 200)}

print("Documentación disponible en:")
print("  Swagger UI: http://localhost:8000/docs")
print("  ReDoc: http://localhost:8000/redoc")
```

### Ejemplo 13: Dockerfile multi-stage — build + runtime

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 13: Dockerfile multi-stage — build + runtime

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

dockerfile
# Dockerfile
dockerfile_content = """
# Stage 1: Build
FROM python:3.10-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-warn-script-location -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim AS runtime
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
"""
with open('Dockerfile', 'w') as f:
    f.write(dockerfile_content)
print("Dockerfile multi-stage creado")

# requirements.txt
with open('requirements.txt', 'w') as f:
    f.write("""fastapi==0.110.0
uvicorn==0.27.0
pydantic==2.6.0
numpy==1.24.0
pandas==2.0.0
joblib==1.3.0
loguru==0.7.0
slowapi==0.1.9
""")
print("requirements.txt creado")
```

### Ejemplo 14: docker-compose con FastAPI + Redis

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 14: docker-compose con FastAPI + Redis

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
docker_compose = """
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379
      - MODEL_PATH=/app/models/modelo_demanda.pkl
    depends_on:
      - redis
    volumes:
      - ./models:/app/models
    deploy:
      replicas: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/conf.d/default.conf
    depends_on:
      - api

volumes:
  redis_data:
"""

with open('docker-compose.yml', 'w') as f:
    f.write(docker_compose)
print("docker-compose.yml creado (api + redis + nginx)")
```

### Ejemplo 15: TestClient de FastAPI para tests

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 15: TestClient de FastAPI para tests

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
from fastapi.testclient import TestClient

client = TestClient(app)

def test_health():
    response = client.get('/health')
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'healthy'
    print("✓ test_health passed")

def test_predict():
    payload = {
        'precio': 150.0,
        'descuento': 0.1,
        'inventario': 500,
        'dia_semana': 3,
        'mes': 7
    }
    response = client.post('/predict', json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 'prediccion' in data
    assert data['prediccion'] > 0
    print(f"✓ test_predict passed: predicción={data['prediccion']:.2f}")

def test_predict_invalid():
    response = client.post('/predict', json={'precio': -100, 'descuento': 2.0})
    assert response.status_code == 422
    print("✓ test_predict_invalid passed")

test_health()
test_predict()
test_predict_invalid()
```

### Ejemplo 16: pytest fixture para app de test

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 16: pytest fixture para app de test

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
import pytest
from fastapi.testclient import TestClient

@pytest.fixture
def app():
    from main import app
    return app

@pytest.fixture
def client(app):
    return TestClient(app)

def test_health_endpoint(client):
    response = client.get('/health')
    assert response.status_code == 200

def test_predict_endpoint(client):
    response = client.post('/predict', json={
        'precio': 150, 'descuento': 0.1, 'inventario': 500,
        'dia_semana': 3, 'mes': 7
    })
    assert response.status_code == 200

def test_invalid_input(client):
    response = client.post('/predict', json={'precio': -1})
    assert response.status_code == 422

def test_missing_fields(client):
    response = client.post('/predict', json={})
    assert response.status_code == 422

print("Fixtures de pytest definidos para testing")
```

### Ejemplo 17: Health endpoint con métricas del modelo

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 17: Health endpoint con métricas del modelo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
import psutil
import platform

class ModelMetrics:
    def __init__(self):
        self.predicciones_totales = 0
        self.errores = 0
        self.latencia_total = 0.0

metrics = ModelMetrics()

@app.get('/health/detailed')
async def health_detailed():
    return {
        'status': 'healthy',
        'model': {
            'loaded': modelo_ml.model is not None,
            'version': modelo_ml.metadata.get('version', 'unknown'),
            'features': modelo_ml.metadata.get('features', []),
            'last_loaded': modelo_ml.metadata.get('loaded_at', 'unknown')
        },
        'metrics': {
            'total_predictions': metrics.predicciones_totales,
            'errors': metrics.errores,
            'avg_latency_ms': (metrics.latencia_total / max(metrics.predicciones_totales, 1)) * 1000
        },
        'system': {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'python_version': platform.python_version(),
            'hostname': platform.node()
        }
    }
```

### Ejemplo 18: Integrador — API completa para predicción de demanda

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplo 18: Integrador — API completa para predicción de demanda

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---

python
"""
API Completa de Predicción de Demanda
Ejecutar con: uvicorn main:app --reload
"""
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import numpy as np
import pandas as pd
import joblib
import logging

# Configuración
app = FastAPI(
    title='API Demanda - Ventas',
    description='Predicción de demanda, clasificación de productos y optimización de inventarios',
    version='2.0.0'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*']
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modelos Pydantic
class PredictionRequest(BaseModel):
    precio: float = Field(gt=0, le=10000, description='Precio del producto')
    descuento: float = Field(ge=0, le=1, description='Descuento (0-1)')
    inventario: int = Field(ge=0, description='Unidades en inventario')
    dia_semana: int = Field(ge=0, le=6, description='Día de la semana')
    mes: int = Field(ge=1, le=12, description='Mes')

    @field_validator('dia_semana')
    def dia_valido(cls, v):
        if v < 0 or v > 6:
            raise ValueError('Día de semana debe ser 0-6')
        return v

class PredictionResponse(BaseModel):
    prediccion: float
    intervalo_inferior: float
    intervalo_superior: float
    timestamp: str

class BatchPredictionRequest(BaseModel):
    productos: List[PredictionRequest]

class BatchPredictionResponse(BaseModel):
    predicciones: List[PredictionResponse]
    total_productos: int
    timestamp: str

# Modelo
class DemandForecaster:
    def __init__(self):
        self.model = None
        self.loaded = False

    def predict(self, features: np.ndarray) -> float:
        if not self.loaded:
            # Modelo dummy para demostración
            return 100 + 0.5 * features[0, 0] - 0.3 * features[0, 1] + \
                   0.2 * features[0, 2] + 0.1 * features[0, 3] - 0.05 * features[0, 4]
        return float(self.model.predict(features)[0])

forecaster = DemandForecaster()

@app.on_event('startup')
async def startup():
    logger.info("Iniciando API de predicción de demanda")
    forecaster.loaded = True

# Endpoints
@app.get('/')
async def root():
    return {
        'api': 'Demand Forecasting API',
        'version': '2.0.0',
        'docs': '/docs',
        'endpoints': ['/health', '/predict', '/predict/batch', '/health/detailed']
    }

@app.get('/health')
async def health():
    return {'status': 'healthy', 'model_loaded': forecaster.loaded}

@app.post('/predict', response_model=PredictionResponse)
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    try:
        features = np.array([[
            request.precio, request.descuento, request.inventario,
            request.dia_semana, request.mes
        ]])
        pred = forecaster.predict(features)
        background_tasks.add_task(logger.info,
            f"Predicción: {pred:.2f} para producto (precio={request.precio})")
        return PredictionResponse(
            prediccion=float(pred),
            intervalo_inferior=float(pred * 0.9),
            intervalo_superior=float(pred * 1.1),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/predict/batch', response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    try:
        resultados = []
        for prod in request.productos:
            features = np.array([[
                prod.precio, prod.descuento, prod.inventario,
                prod.dia_semana, prod.mes
            ]])
            pred = forecaster.predict(features)
            resultados.append(PredictionResponse(
                prediccion=float(pred),
                intervalo_inferior=float(pred * 0.9),
                intervalo_superior=float(pred * 1.1),
                timestamp=datetime.now().isoformat()
            ))
        return BatchPredictionResponse(
            predicciones=resultados,
            total_productos=len(resultados),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)
```

---

## 3. Ejercicios Propuestos

1. **API para clasificación de productos**: Crea una API con endpoint /clasificar que reciba características de un producto (precio, categoría, temporada, rating) y devuelva la probabilidad de que sea exitoso (clasificación binaria). Incluye validación Pydantic.

2. **Cache con Redis**: Implementa un sistema de cache con Redis para el endpoint /predict. Si se reciben los mismos features en menos de 5 minutos, devuelve el resultado cachead en lugar de ejecutar el modelo.

3. **Batch prediction endpoint**: Extiende la API para aceptar hasta 100 productos por request en /predict/batch. Implementa procesamiento paralelo con asyncio.gather y mide el speedup.

4. **Autenticación JWT**: Implementa autenticación OAuth2 con JWT. Endpoint /token que devuelve token, y /predict/secure que requiere token Bearer válido. Usa python-jose.

5. **Paginated product catalog**: Crea endpoint GET /productos con paginación (page, size), filtros (categoria, precio_min, precio_max) y ordenamiento. Devuelve metadata de paginación.

6. **Docker compose con monitoreo**: Extiende docker-compose.yml para incluir Prometheus (puerto 9090) y Grafana (puerto 3000). Configura scraping de métricas de FastAPI.

7. **Tests de integración**: Escribe tests con pytest y TestClient que cubran: (1) health check, (2) predicción válida, (3) predicción inválida (422), (4) batch prediction, (5) rate limiting.

8. **API con múltiples modelos**: Crea una API que sirva 3 modelos: demanda, clasificación de producto, y optimización de precio. Cada modelo tiene su propio endpoint y versión. Usa router de FastAPI para organizar.

---

## 4. Resumen

| Componente | Propósito |
|---|---|
| **FastAPI** | Framework web asíncrono para APIs ML |
| **Pydantic** | Validación de datos de entrada/salida |
| **Uvicorn** | Servidor ASGI de alto rendimiento |
| **Docker** | Contenedorización del servicio |
| **TestClient** | Testing de endpoints sin servidor |
| **Middleware** | CORS, rate limiting, compresión |
| **Autenticación** | API Key, OAuth2, JWT |

FastAPI es la opción líder para servir modelos ML en producción por su rendimiento (comparable a Node.js/Go), validación automática con Pydantic, documentación interactiva generada automáticamente y soporte nativo para asincronía.
