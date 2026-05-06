# A22: torch.nn — Capas, Módulos y Redes Neuronales para Ventas, Compras e Inventarios

## Introducción Teórica

**torch.nn** es el módulo de PyTorch que proporciona los bloques fundamentales para construir redes neuronales. Su clase base es `nn.Module`, que agrupa parámetros, define la arquitectura y maneja el ciclo de vida del modelo.

### Conceptos clave:

1. **nn.Module**: Clase base para todo modelo. Se hereda definiendo `__init__` (declarar capas) y `forward` (cómo fluyen los datos). Métodos clave: `parameters()`, `named_parameters()`, `children()`, `modules()`, `apply()`, `train()`, `eval()`, `to()`, `state_dict()`, `load_state_dict()`, `zero_grad()`, `requires_grad_()`, `add_module()`, `register_buffer()`, `register_parameter()`, `register_module()`.
2. **nn.Linear(in, out, bias)**: Capa fully-connected (densa). Realiza `y = x @ W^T + b`. `weight` y `bias` son parámetros entrenables.
3. **Activaciones**: `nn.ReLU` (max(0,x)), `nn.Sigmoid` (1/(1+e^-x)), `nn.Tanh` (tanh(x)), `nn.Softmax(dim)` (e^x_i / Σe^x_j), `nn.LogSoftmax(dim)` (log softmax).
4. **nn.Dropout(p, inplace)**: Desactiva aleatoriamente neuronas durante entrenamiento (regularización). En `train()` aplica dropout (escala con 1/(1-p)); en `eval()` es identidad.
5. **nn.BatchNorm1d(num_features)**: Normaliza activaciones por mini-batch: `γ * (x - μ)/σ + β`. Parámetros: `weight` (γ), `bias` (β), `running_mean`, `running_var`. Modos train (usa batch stats) y eval (usa running stats).
6. **nn.Sequential**: Contenedor secuencial de capas. Las capas se aplican en orden. No requiere definir `forward` manual.
7. **nn.ModuleList**: Lista de módulos como atributo. PyTorch registra cada submódulo para que `parameters()` los detecte. Útil para capas dinámicas.
8. **nn.ModuleDict**: Diccionario de módulos. Similar a ModuleList pero con nombres de llave.
9. **nn.Flatten**: Aplana un tensor a 2D (batch, features). `nn.Unflatten`: operación inversa.
10. **state_dict**: Diccionario con parámetros entrenables y buffers (running_mean, etc.). `load_state_dict` para restaurar.
11. **to(device)**: Mueve el modelo completo (parámetros y buffers) a CPU/GPU.

### Aplicaciones en negocio:

- **Ventas**: Red densa para predicción de demanda mensual con features de precio, promoción, estacionalidad.
- **Compras**: Clasificación de proveedores en confiables/no confiables usando métricas de desempeño.
- **Inventarios**: Regresión multiclase para categorizar productos por rotación (alta, media, baja).

---

## Ejemplos

### Ejemplo 1: nn.Module — Definir red neuronal como clase

```python
import torch
import torch.nn as nn

class PredictorVentas(nn.Module):
    def __init__(self, n_features=5, n_ocultas=64, n_salida=1):
        super().__init__()
        self.capa1 = nn.Linear(n_features, n_ocultas)
        self.relu = nn.ReLU()
        self.capa2 = nn.Linear(n_ocultas, n_salida)

    def forward(self, x):
        x = self.relu(self.capa1(x))
        return self.capa2(x)

modelo = PredictorVentas()
print(modelo)

# Forward con datos simulados
X = torch.randn(32, 5)
y_pred = modelo(X)
print(f"Predicción shape: {y_pred.shape}")  # torch.Size([32, 1])
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: nn.Module — Definir red neuronal como clase.*

1. Forward con datos simulados

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: nn.Linear — Capa fully connected

```python
import torch
import torch.nn as nn

# 5 features de producto → 64 neuronas ocultas
capa = nn.Linear(in_features=5, out_features=64, bias=True)
print(f"Pesos shape: {capa.weight.shape}")  # [64, 5]
print(f"Sesgo shape: {capa.bias.shape}")    # [64]

# Forward: 10 productos en batch
X = torch.randn(10, 5)
salida = capa(X)
print(f"Salida shape: {salida.shape}")  # [10, 64]

# Sin bias
capa_sin_bias = nn.Linear(5, 64, bias=False)
print(f"capa_sin_bias.bias: {capa_sin_bias.bias}")  # None
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: nn.Linear — Capa fully connected.*

1. 5 features de producto → 64 neuronas ocultas
2. Forward: 10 productos en batch
3. Sin bias

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: nn.ReLU — Activación en medio de la red

```python
import torch
import torch.nn as nn

class RedReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.fc1(x)
        print(f"Antes de ReLU: min={x.min().item():.2f}, max={x.max().item():.2f}")
        x = self.relu(x)
        print(f"Después de ReLU: min={x.min().item():.2f}, max={x.max().item():.2f}")
        return self.fc2(x)

modelo = RedReLU()
X = torch.randn(5, 10)
modelo(X)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: nn.ReLU — Activación en medio de la red.*

1. `import torch` — Importa las librerías necesarias para el análisis.
2. `import torch.nn as nn` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: nn.Sigmoid — Activación final para clasificación binaria

```python
import torch
import torch.nn as nn

class ClasificadorCompra(nn.Module):
    """Clasifica si un proveedor es confiable (1) o no (0)"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.sigmoid(self.fc3(x))

modelo = ClasificadorCompra()
proveedores = torch.randn(10, 4)  # [precio, lead_time, calidad, cumplimiento]
probabilidades = modelo(proveedores)
print(f"Probabilidades de confiabilidad:\n{probabilidades.squeeze()}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: nn.Sigmoid — Activación final para clasificación binaria.*

1. `import torch` — Importa las librerías necesarias para el análisis.
2. `import torch.nn as nn` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: nn.Softmax — Activación final para multiclase

```python
import torch
import torch.nn as nn

class ClasificadorRotacion(nn.Module):
    """Clasifica productos en 3 categorías de rotación: alta, media, baja"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 32)
        self.fc2 = nn.Linear(32, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)  # logits (se aplica softmax en la pérdida)

modelo = ClasificadorRotacion()
productos = torch.randn(8, 6)
logits = modelo(productos)
print(f"Logits:\n{logits}")

softmax = nn.Softmax(dim=1)
probabilidades = softmax(logits)  # [8, 3], cada fila suma 1
print(f"Probabilidades (suma=1):\n{probabilidades}")
print(f"Suma por fila: {probabilidades.sum(dim=1)}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: nn.Softmax — Activación final para multiclase.*

1. `import torch` — Importa las librerías necesarias para el análisis.
2. `import torch.nn as nn` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: nn.Dropout — Regularización durante entrenamiento

```python
import torch
import torch.nn as nn

class RedConDropout(nn.Module):
    def __init__(self, p=0.3):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.dropout = nn.Dropout(p=p)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # en train apaga el 30% de neuronas
        return self.fc2(x)

modelo = RedConDropout(p=0.3)
X = torch.randn(5, 10)

modelo.train()  # modo entrenamiento
y_train = modelo(X)
print(f"Train mode - {y_train}")

modelo.eval()   # modo evaluación (dropout desactivado)
with torch.no_grad():
    y_eval = modelo(X)
print(f"Eval mode - {y_eval}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: nn.Dropout — Regularización durante entrenamiento.*

1. `import torch` — Importa las librerías necesarias para el análisis.
2. `import torch.nn as nn` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: nn.BatchNorm1d — Normalizar activaciones por batch

```python
import torch
import torch.nn as nn

class RedBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        return self.fc2(x)

modelo = RedBatchNorm()
print(f"BatchNorm weight: {modelo.bn1.weight}")
print(f"BatchNorm bias: {modelo.bn1.bias}")
print(f"Running mean inicial: {modelo.bn1.running_mean}")
print(f"Running var inicial: {modelo.bn1.running_var}")

X = torch.randn(64, 5)
y = modelo(X)
print(f"Salida shape: {y.shape}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: nn.BatchNorm1d — Normalizar activaciones por batch.*

1. `import torch` — Importa las librerías necesarias para el análisis.
2. `import torch.nn as nn` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: nn.Sequential — Apilar capas sin definir clase

```python
import torch
import torch.nn as nn

modelo = nn.Sequential(
    nn.Linear(5, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

print(modelo)
X = torch.randn(16, 5)
y = modelo(X)
print(f"Predicción shape: {y.shape}")

# Añadir capa después
modelo.append(nn.Sigmoid())
print(f"Con Sigmoid añadida:\n{modelo}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: nn.Sequential — Apilar capas sin definir clase.*

1. Añadir capa después

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: nn.ModuleList — Lista de capas como atributo del módulo

```python
import torch
import torch.nn as nn

class RedProfundaVentas(nn.Module):
    def __init__(self, dims=[5, 64, 128, 64, 1]):
        super().__init__()
        self.capas = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.capas.append(nn.Linear(dims[i], dims[i+1]))
        self.activacion = nn.ReLU()

    def forward(self, x):
        for i, capa in enumerate(self.capas[:-1]):
            x = self.activacion(capa(x))
        return self.capas[-1](x)  # última capa sin activación

modelo = RedProfundaVentas()
print(f"Número de capas: {len(modelo.capas)}")
X = torch.randn(8, 5)
y = modelo(X)
print(f"Predicción: {y.shape}")

# Acceder a capa específica
print(f"Pesos capa 2: {modelo.capas[2].weight.shape}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: nn.ModuleList — Lista de capas como atributo del módulo.*

1. Acceder a capa específica

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: nn.ModuleDict — Diccionario de capas

```python
import torch
import torch.nn as nn

class RedModularVentas(nn.Module):
    def __init__(self):
        super().__init__()
        self.capas = nn.ModuleDict({
            'entrada': nn.Linear(5, 64),
            'oculta1': nn.Linear(64, 32),
            'salida': nn.Linear(32, 1)
        })
        self.activacion = nn.ReLU()

    def forward(self, x):
        x = self.activacion(self.capas['entrada'](x))
        x = self.activacion(self.capas['oculta1'](x))
        return self.capas['salida'](x)

modelo = RedModularVentas()
X = torch.randn(4, 5)
print(modelo(X))

# Agregar capa dinámicamente
modelo.capas['dropout'] = nn.Dropout(0.2)
print(f"Llaves del ModuleDict: {list(modelo.capas.keys())}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: nn.ModuleDict — Diccionario de capas.*

1. Agregar capa dinámicamente

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: parameters() — Iterar sobre parámetros entrenables

```python
import torch
import torch.nn as nn

modelo = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

print("Parámetros del modelo:")
for i, param in enumerate(modelo.parameters()):
    print(f"  Parámetro {i}: shape={param.shape}, size={param.numel()}, requires_grad={param.requires_grad}")

total_params = sum(p.numel() for p in modelo.parameters())
print(f"Total parámetros entrenables: {total_params}")

# Contar solo pesos, no bias
pesos = sum(p.numel() for p in modelo.parameters() if p.dim() > 1)
print(f"Solo pesos: {pesos}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: parameters() — Iterar sobre parámetros entrenables.*

1. Contar solo pesos, no bias

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: named_parameters() — Parámetros con nombre

```python
import torch
import torch.nn as nn

modelo = nn.Sequential(
    nn.Linear(5, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)
modelo = nn.Sequential(
    nn.Linear(5, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

print("Parámetros nombrados:")
for nombre, param in modelo.named_parameters():
    print(f"  {nombre}: shape={param.shape}, requires_grad={param.requires_grad}")

# Acceder a peso específico por nombre
peso_fc1 = dict(modelo.named_parameters())['0.weight']
print(f"\nPeso capa 0: {peso_fc1.shape}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: named_parameters() — Parámetros con nombre.*

1. Acceder a peso específico por nombre

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: children() vs modules() — Capas directas vs recursivas

```python
import torch
import torch.nn as nn

class RedCompleja(nn.Module):
    def __init__(self):
        super().__init__()
        self.bloque1 = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU()
        )
        self.bloque2 = nn.Sequential(
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.bloque2(self.bloque1(x))

modelo = RedCompleja()

print("children() - solo capas directas:")
for i, child in enumerate(modelo.children()):
    print(f"  {i}: {child}")

print("\nmodules() - todas las capas recursivamente:")
for i, mod in enumerate(modelo.modules()):
    print(f"  {i}: {mod}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: children() vs modules() — Capas directas vs recursivas.*

1. `import torch` — Importa las librerías necesarias para el análisis.
2. `import torch.nn as nn` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: apply(init_weights) — Inicializar pesos personalizados

```python
import torch
import torch.nn as nn

def init_weights(modulo):
    if isinstance(modulo, nn.Linear):
        nn.init.xavier_uniform_(modulo.weight)
        if modulo.bias is not None:
            nn.init.zeros_(modulo.bias)
        print(f"Inicializado: {modulo}")

modelo = nn.Sequential(
    nn.Linear(5, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

modelo.apply(init_weights)

# Verificar inicialización
print(f"\nPeso fc1 mean: {modelo[0].weight.mean().item():.6f}")
print(f"Peso fc1 std: {modelo[0].weight.std().item():.6f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: apply(init_weights) — Inicializar pesos personalizados.*

1. Verificar inicialización

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: train() vs eval() — Modo entrenamiento vs evaluación

```python
import torch
import torch.nn as nn

class RedConDropoutBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.bn1(torch.relu(self.fc1(x)))
        x = self.dropout(x)
        return self.fc2(x)

modelo = RedConDropoutBatchNorm()
X = torch.randn(100, 5)
y_train = modelo(X)

print("train mode - dropout y batchnorm activos")
modelo.train()
y_train = modelo(X)

print("eval mode - dropout desactivado, batchnorm usa running stats")
modelo.eval()
with torch.no_grad():
    y_eval = modelo(X)

print(f"Train output[:3]: {y_train[:3].squeeze()}")
print(f"Eval output[:3]: {y_eval[:3].squeeze()}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: train() vs eval() — Modo entrenamiento vs evaluación.*

1. `import torch` — Importa las librerías necesarias para el análisis.
2. `import torch.nn as nn` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 16: state_dict/load_state_dict — Guardar y cargar pesos

```python
import torch
import torch.nn as nn

modelo = nn.Sequential(
    nn.Linear(5, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

# Guardar pesos
torch.save(modelo.state_dict(), '/tmp/modelo_ventas.pth')
print("Pesos guardados en /tmp/modelo_ventas.pth")

# Cargar en modelo nuevo
modelo_nuevo = nn.Sequential(
    nn.Linear(5, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)
modelo_nuevo.load_state_dict(torch.load('/tmp/modelo_ventas.pth'))
print("Pesos cargados correctamente")

# Verificar igualdad
for p1, p2 in zip(modelo.parameters(), modelo_nuevo.parameters()):
    print(f"Pesos iguales: {torch.allclose(p1, p2)}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: state_dict/load_state_dict — Guardar y cargar pesos.*

1. Guardar pesos
2. Cargar en modelo nuevo
3. Verificar igualdad

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 17: to(device) — Mover modelo a GPU

```python
import torch
import torch.nn as nn

modelo = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

modelo = modelo.to(device)

# Verificar que parámetros están en el device correcto
for nombre, param in modelo.named_parameters():
    print(f"{nombre}: device={param.device}")
    break

# Datos también deben ir al mismo device
X = torch.randn(32, 10).to(device)
y = modelo(X)
print(f"Salida device: {y.device}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: to(device) — Mover modelo a GPU.*

1. Verificar que parámetros están en el device correcto
2. Datos también deben ir al mismo device

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 18: Integrador — Red neuronal completa para clasificación de productos

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Dataset sintético: 1000 productos con 6 features y 3 categorías
torch.manual_seed(42)
X = torch.randn(1000, 6)
y = torch.randint(0, 3, (1000,))

# Modelo
class ClasificadorProductos(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)

    def forward(self, x):
        return self.net(x)

modelo = ClasificadorProductos()
criterio = nn.CrossEntropyLoss()
optimizador = optim.Adam(modelo.parameters(), lr=0.01)

# Entrenamiento (1 epoch)
modelo.train()
for i in range(0, len(X), 64):
    batch_X = X[i:i+64]
    batch_y = y[i:i+64]

    optimizador.zero_grad()
    logits = modelo(batch_X)
    loss = criterio(logits, batch_y)
    loss.backward()
    optimizador.step()

# Evaluación
modelo.eval()
with torch.no_grad():
    logits = modelo(X)
    preds = logits.argmax(dim=1)
    accuracy = (preds == y).float().mean()

print(f"Precisión: {accuracy:.2%}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador — Red neuronal completa para clasificación de productos.*

1. Dataset sintético: 1000 productos con 6 features y 3 categorías
2. Modelo
3. Entrenamiento (1 epoch)
4. Evaluación

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejercicios

1. Define un `nn.Module` llamado `PredictorCompras` con 3 capas lineales (8→32, 32→16, 16→1) usando ReLU entre ellas. Haz un forward con un batch de 20 muestras.

2. Crea una red usando `nn.Sequential` para regresión de inventarios con 4 features de entrada y 1 salida. Incluye BatchNorm1d después de la primera capa lineal.

3. Usa `nn.ModuleList` para construir una red con número variable de capas ocultas. Parámetros: input_dim=5, hidden_dims=[128, 64, 32], output_dim=1.

4. Implementa una función de inicialización personalizada con `apply()` que inicialice los pesos de capas lineales con `nn.init.kaiming_normal_` y sesgos en 0.01.

5. Crea un modelo con Dropout(p=0.4) y BatchNorm1d. Muestra la diferencia en el output entre `model.train()` y `model.eval()` para el mismo input.

6. Usa `state_dict` y `load_state_dict` para copiar los pesos de un modelo entrenado a otro. Verifica que las predicciones coinciden.

7. Construye un modelo con `nn.ModuleDict` que tenga tres "ramas": una para features de precio, otra para features de temporada, y una fusión final. La rama de precio tiene [3→16], temporada [4→16], fusión [32→1].

8. Diseña una red completa para clasificar productos en 5 categorías de rotación de inventario. Usa 7 features de entrada, 2 capas ocultas (128 y 64), BatchNorm, Dropout(0.3), y Softmax. Entrénela por 10 epochs con optimizador Adam.

---

## Resumen

`torch.nn` proporciona todos los componentes para construir redes neuronales modulares y reutilizables:

| Componente | Propósito |
|------------|-----------|
| `nn.Module` | Clase base para todo modelo |
| `nn.Linear` | Capa fully-connected |
| `nn.ReLU` / `nn.Sigmoid` / `nn.Softmax` | Funciones de activación |
| `nn.Dropout` | Regularización |
| `nn.BatchNorm1d` | Normalización por batch |
| `nn.Sequential` | Apilar capas secuenciales |
| `nn.ModuleList/Dict` | Colecciones de capas |
| `parameters()` / `named_parameters()` | Acceder a parámetros |
| `state_dict` / `load_state_dict` | Guardar y cargar |
| `train()` / `eval()` | Alternar modos |
| `to(device)` | Mover a GPU |

**En negocio**: Con nn.Module puedes crear desde modelos lineales simples para predicción de ventas hasta arquitecturas complejas para segmentación de productos, clasificación de proveedores y optimización de inventarios.
