# A24: Training Loop — Optimizadores, Pérdidas y Entrenamiento para Ventas, Compras e Inventarios

## Introducción Teórica

El **training loop** es el corazón del aprendizaje profundo: un ciclo que repite forward pass, cálculo de pérdida, backward pass (gradientes) y actualización de pesos. PyTorch ofrece componentes modulares para cada etapa: `torch.optim` para optimización, `torch.nn.functional` para funciones de pérdida y activación, `torch.optim.lr_scheduler` para ajustar la tasa de aprendizaje, y utilidades para logging y checkpointing.

### Conceptos clave:

1. **torch.optim.Optimizer**: Clase base para optimizadores. `SGD` (descenso gradiente estocástico con momentum), `Adam` (adaptativo más popular), `AdamW` (Adam con decaimiento de pesos desacoplado), `RMSprop` (Root Mean Square propagation), `Adagrad`/`Adadelta` (tasas adaptativas por parámetro), `Adamax`/`Nadam` (variantes de Adam), `ASGD` (SGD promedio), `LBFGS` (cuasi-Newton, pocos datos). Métodos: `zero_grad()`, `step()`, `add_param_group()`, `state_dict()`, `load_state_dict()`.
2. **loss.backward()**: Calcula gradientes de la pérdida respecto a todos los parámetros con `requires_grad=True`. Los gradientes se **acumulan** en `.grad`.
3. **optimizer.step()**: Actualiza parámetros usando los gradientes acumulados: `θ = θ - lr * ∇θ`. El algoritmo varía según el optimizador (momentum, adaptive, etc.).
4. **torch.nn.functional**: Funciones sin estado para pérdidas y activaciones: `F.mse_loss` (regresión), `F.cross_entropy` (clasificación multiclase, combina log_softmax + NLLLoss), `F.binary_cross_entropy_with_logits` (clasificación binaria, más estable numéricamente que BCE + Sigmoid), `F.binary_cross_entropy` (requiere probabilidades), `F.softmax`, `F.relu`, `F.sigmoid`, `F.pad`, `F.dropout`.
5. **model.train()**: Activa dropout, batch norm usa estadísticas del batch.
6. **model.eval()**: Desactiva dropout, batch norm usa running averages.
7. **torch.no_grad()**: Deshabilita construcción del grafo en evaluación.
8. **lr_scheduler**: Ajusta lr durante entrenamiento: `StepLR` (cada N epochs multiplica por gamma), `MultiStepLR` (en épocas específicas), `ExponentialLR` (decaimiento exponencial continuo), `ReduceLROnPlateau` (reduce cuando métrica deja de mejorar), `CosineAnnealingLR` (ciclo coseno hasta eta_min), `OneCycleLR` (aumenta luego disminuye, con warmup), `LinearLR` (lineal), `CyclicLR` (cíclico entre min y max).
9. **Checkpointing**: `torch.save(model.state_dict(), path)` para pesos; `torch.save({'epoch': e, 'model_state_dict': ..., 'optimizer_state_dict': ..., 'loss': ...}, path)` para checkpoint completo.
10. **TensorBoard**: `SummaryWriter` para registrar métricas, gráficos, histogramas. `writer.add_scalar('Loss/train', loss, epoch)`, `add_graph`, `add_histogram`.

### Aplicaciones en negocio:

- **Ventas**: Training loop con MSE y Adam para predecir demanda; ReduceLROnPlateau cuando la pérdida de validación se estanca; TensorBoard para monitorear convergencia.
- **Compras**: Binary cross-entropy con SGD para clasificar urgencia de órdenes; OneCycleLR para convergencia rápida; early stopping para evitar sobreajuste en datos de proveedores.
- **Inventarios**: CrossEntropyLoss para clasificación de rotación; StepLR cada 10 epochs; checkpoint del mejor modelo según precisión de validación.

---

## Ejemplos

### Ejemplo 1: Training loop básico

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Datos sintéticos de ventas
torch.manual_seed(42)
X = torch.randn(200, 5)
y = X.sum(dim=1, keepdim=True) + torch.randn(200, 1) * 0.1

modelo = nn.Linear(5, 1)
criterio = nn.MSELoss()
optimizador = optim.SGD(modelo.parameters(), lr=0.01)

n_epochs = 50
for epoch in range(n_epochs):
    # Forward
    pred = modelo(X)
    loss = criterio(pred, y)

    # Backward
    optimizador.zero_grad()
    loss.backward()
    optimizador.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:2d}: loss = {loss.item():.6f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: Training loop básico.*

1. Datos sintéticos de ventas
2. Forward
3. Backward

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: model.train() y model.eval()

```python
import torch
import torch.nn as nn
import torch.optim as optim

class RedVentas(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

modelo = RedVentas()
X = torch.randn(10, 5)

# Modo entrenamiento
modelo.train()
y_train = modelo(X)
print(f"Train output: {y_train.squeeze().tolist()}")

# Modo evaluación
modelo.eval()
with torch.no_grad():
    y_eval = modelo(X)
print(f"Eval output:  {y_eval.squeeze().tolist()}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: model.train() y model.eval().*

1. Modo entrenamiento
2. Modo evaluación

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: optimizer.zero_grad() — Limpiar gradientes antes de cada batch

```python
import torch
import torch.nn as nn
import torch.optim as optim

modelo = nn.Linear(3, 1)
optimizador = optim.SGD(modelo.parameters(), lr=0.01)

X = torch.randn(10, 3)
y = torch.randn(10, 1)

# Sin zero_grad() — los gradientes se acumulan
pred1 = modelo(X[:5])
loss1 = ((pred1 - y[:5]) ** 2).mean()
loss1.backward()
print(f"Grad después de batch1: {modelo.weight.grad}")

pred2 = modelo(X[5:])  # sin zero_grad!
loss2 = ((pred2 - y[5:]) ** 2).mean()
loss2.backward()
print(f"Grad después de batch2 (acumulado): {modelo.weight.grad}")

# Con zero_grad() correcto
optimizador.zero_grad()
pred3 = modelo(X[:5])
loss3 = ((pred3 - y[:5]) ** 2).mean()
loss3.backward()
print(f"Grad con zero_grad(): {modelo.weight.grad}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: optimizer.zero_grad() — Limpiar gradientes antes de cada batch.*

1. Sin zero_grad() — los gradientes se acumulan
2. Con zero_grad() correcto

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: loss.backward() — Calcular gradientes

```python
import torch
import torch.nn as nn

modelo = nn.Linear(2, 1)
X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
y = torch.tensor([[5.0], [11.0]])

pred = modelo(X)
loss = ((pred - y) ** 2).mean()

print("Antes de backward:")
print(f"  weight.grad: {modelo.weight.grad}")
print(f"  bias.grad: {modelo.bias.grad}")

loss.backward()

print("Después de backward:")
print(f"  weight.grad:\n{modelo.weight.grad}")
print(f"  bias.grad: {modelo.bias.grad}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: loss.backward() — Calcular gradientes.*

1. `import torch` — Importa las librerías necesarias para el análisis.
2. `import torch.nn as nn` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: optimizer.step() — Actualizar pesos

```python
import torch
import torch.nn as nn
import torch.optim as optim

modelo = nn.Linear(2, 1)
optimizador = optim.SGD(modelo.parameters(), lr=0.1)
X = torch.tensor([[1.0, 2.0]])
y = torch.tensor([[5.0]])

pesos_antes = modelo.weight.clone()
pred = modelo(X)
loss = ((pred - y) ** 2).mean()

optimizador.zero_grad()
loss.backward()
optimizador.step()

pesos_despues = modelo.weight
print(f"Pesos antes:  {pesos_antes.squeeze().tolist()}")
print(f"Gradiente:    {modelo.weight.grad.squeeze().tolist()}")
print(f"Pesos después: {pesos_despues.squeeze().tolist()}")
print(f"Diferencia:   {(pesos_despues - pesos_antes).squeeze().tolist()}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: optimizer.step() — Actualizar pesos.*

1. `import torch` — Importa las librerías necesarias para el análisis.
2. `import torch.nn as nn` — Importa las librerías necesarias para el análisis.
3. `import torch.optim as optim` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: torch.no_grad() — Deshabilitar gradiente en evaluación

```python
import torch
import torch.nn as nn
import torch.optim as optim
import time

modelo = nn.Sequential(
    nn.Linear(100, 1000),
    nn.ReLU(),
    nn.Linear(1000, 1000),
    nn.ReLU(),
    nn.Linear(1000, 1)
)
X = torch.randn(500, 100)

# Evaluación con gradientes (lento)
start = time.time()
for _ in range(100):
    y = modelo(X)
    _ = y.sum()
print(f"Con gradientes: {time.time() - start:.3f}s")

# Evaluación sin gradientes (rápido)
modelo.eval()
start = time.time()
for _ in range(100):
    with torch.no_grad():
        y = modelo(X)
        _ = y.sum()
print(f"Sin gradientes: {time.time() - start:.3f}s")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: torch.no_grad() — Deshabilitar gradiente en evaluación.*

1. Evaluación con gradientes (lento)
2. Evaluación sin gradientes (rápido)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: F.mse_loss — Pérdida de regresión

```python
import torch
import torch.nn.functional as F
import torch.nn as nn

# Predicción de demanda de productos
y_real = torch.tensor([100.0, 200.0, 150.0, 300.0, 250.0])
y_pred = torch.tensor([95.0, 210.0, 145.0, 290.0, 260.0])

loss_mse = F.mse_loss(y_pred, y_real)
print(f"MSE Loss: {loss_mse.item():.4f}")

# MAE (Mean Absolute Error) también disponible
loss_mae = F.l1_loss(y_pred, y_real)
print(f"MAE Loss: {loss_mae.item():.4f}")

# RMSE
rmse = torch.sqrt(loss_mse)
print(f"RMSE: {rmse.item():.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: F.mse_loss — Pérdida de regresión.*

1. Predicción de demanda de productos
2. MAE (Mean Absolute Error) también disponible
3. RMSE

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: F.cross_entropy — Pérdida de clasificación

```python
import torch
import torch.nn.functional as F

# Clasificación de productos: 0=Premium, 1=Estándar, 2=Económico
logits = torch.tensor([[2.0, 0.5, 0.1],  # predicción: clase 0
                        [0.3, 2.5, 0.2],   # predicción: clase 1
                        [0.1, 0.3, 3.0]])  # predicción: clase 2
y_true = torch.tensor([0, 1, 2])

loss = F.cross_entropy(logits, y_true)
print(f"CrossEntropyLoss: {loss.item():.4f}")

# Equivalente manual: LogSoftmax + NLLLoss
log_softmax = F.log_softmax(logits, dim=1)
loss_manual = F.nll_loss(log_softmax, y_true)
print(f"Manual NLLLoss: {loss_manual.item():.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: F.cross_entropy — Pérdida de clasificación.*

1. Clasificación de productos: 0=Premium, 1=Estándar, 2=Económico
2. Equivalente manual: LogSoftmax + NLLLoss

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: F.binary_cross_entropy_with_logits — Clasificación binaria estable

```python
import torch
import torch.nn.functional as F

# Clasificación: ¿el producto se venderá bien? (0/1)
logits = torch.tensor([2.5, -1.0, 0.3, 4.0, -2.5])  # sin sigmoid
y_true = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0])

# Forma más estable: BCEWithLogits (combina sigmoid + BCE)
loss = F.binary_cross_entropy_with_logits(logits, y_true)
print(f"BCEWithLogits: {loss.item():.4f}")

# Equivalente manual (menos estable numéricamente)
probabilidades = torch.sigmoid(logits)
loss_manual = F.binary_cross_entropy(probabilidades, y_true)
print(f"BCE manual:   {loss_manual.item():.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: F.binary_cross_entropy_with_logits — Clasificación binaria estable.*

1. Clasificación: ¿el producto se venderá bien? (0/1)
2. Forma más estable: BCEWithLogits (combina sigmoid + BCE)
3. Equivalente manual (menos estable numéricamente)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: SGD vs Adam — Comparar convergencia

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(42)
X = torch.randn(500, 10)
y = X @ torch.randn(10, 1) + torch.randn(500, 1) * 0.5

def entrenar(optimizador_cls, nombre, lr=0.01, epochs=100):
    modelo = nn.Linear(10, 1)
    criterio = nn.MSELoss()
    optimizador = optimizador_cls(modelo.parameters(), lr=lr)
    losses = []

    for _ in range(epochs):
        optimizador.zero_grad()
        loss = criterio(modelo(X), y)
        loss.backward()
        optimizador.step()
        losses.append(loss.item())

    return losses

losses_sgd = entrenar(optim.SGD, "SGD", lr=0.01)
losses_adam = entrenar(optim.Adam, "Adam", lr=0.01)

print(f"Pérdida final SGD:  {losses_sgd[-1]:.4f}")
print(f"Pérdida final Adam: {losses_adam[-1]:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: SGD vs Adam — Comparar convergencia.*

1. `import torch` — Importa las librerías necesarias para el análisis.
2. `import torch.nn as nn` — Importa las librerías necesarias para el análisis.
3. `import torch.optim as optim` — Importa las librerías necesarias para el análisis.
4. `import matplotlib.pyplot as plt` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: StepLR — Reducir lr cada N epochs

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

modelo = nn.Linear(5, 1)
optimizador = optim.SGD(modelo.parameters(), lr=0.1)
scheduler = StepLR(optimizador, step_size=10, gamma=0.5)

X = torch.randn(100, 5)
y = torch.randn(100, 1)

for epoch in range(30):
    optimizador.zero_grad()
    loss = nn.MSELoss()(modelo(X), y)
    loss.backward()
    optimizador.step()
    scheduler.step()

    if (epoch + 1) % 5 == 0:
        lr_actual = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1:2d}: loss={loss.item():.4f}, lr={lr_actual:.5f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: StepLR — Reducir lr cada N epochs.*

1. `import torch` — Importa las librerías necesarias para el análisis.
2. `import torch.nn as nn` — Importa las librerías necesarias para el análisis.
3. `import torch.optim as optim` — Importa las librerías necesarias para el análisis.
4. `from torch.optim.lr_scheduler import StepLR` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: ReduceLROnPlateau — Reducir cuando métrica se estanca

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

modelo = nn.Linear(5, 1)
optimizador = optim.Adam(modelo.parameters(), lr=0.01)
scheduler = ReduceLROnPlateau(optimizador, mode='min', factor=0.5,
                               patience=5, min_lr=1e-6, verbose=True)

X = torch.randn(200, 5)
y = X.sum(dim=1, keepdim=True) + torch.randn(200, 1) * 0.1

val_X = torch.randn(50, 5)
val_y = val_X.sum(dim=1, keepdim=True) + torch.randn(50, 1) * 0.1

for epoch in range(50):
    modelo.train()
    optimizador.zero_grad()
    train_loss = nn.MSELoss()(modelo(X), y)
    train_loss.backward()
    optimizador.step()

    modelo.eval()
    with torch.no_grad():
        val_loss = nn.MSELoss()(modelo(val_X), val_y)

    scheduler.step(val_loss)  # ReduceLROnPlateau recibe la métrica

    if (epoch + 1) % 10 == 0:
        lr_actual = optimizador.param_groups[0]['lr']
        print(f"Epoch {epoch+1:2d}: train={train_loss.item():.4f}, "
              f"val={val_loss.item():.4f}, lr={lr_actual:.6f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: ReduceLROnPlateau — Reducir cuando métrica se estanca.*

1. `import torch` — Importa las librerías necesarias para el análisis.
2. `import torch.nn as nn` — Importa las librerías necesarias para el análisis.
3. `import torch.optim as optim` — Importa las librerías necesarias para el análisis.
4. `from torch.optim.lr_scheduler import ReduceLROnPlateau` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: CosineAnnealingLR — Ciclo coseno de lr

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import math

modelo = nn.Linear(5, 1)
optimizador = optim.SGD(modelo.parameters(), lr=0.1)
scheduler = CosineAnnealingLR(optimizador, T_max=20, eta_min=0.001)

lrs = []
for epoch in range(30):
    scheduler.step()
    lrs.append(scheduler.get_last_lr()[0])

print("LR por epoch (primeros 10):")
for i, lr in enumerate(lrs[:10]):
    print(f"  Epoch {i+1}: lr={lr:.5f}")
print(f"  ...")
print(f"  Epoch 20: lr={lrs[19]:.5f} (mínimo del primer ciclo)")
print(f"  Epoch 21: lr={lrs[20]:.5f} (reinicia)")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: CosineAnnealingLR — Ciclo coseno de lr.*

1. `import torch` — Importa las librerías necesarias para el análisis.
2. `import torch.nn as nn` — Importa las librerías necesarias para el análisis.
3. `import torch.optim as optim` — Importa las librerías necesarias para el análisis.
4. `from torch.optim.lr_scheduler import CosineAnnealingLR` — Importa las librerías necesarias para el análisis.
5. `import math` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: OneCycleLR — LR cíclico con warmup

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

modelo = nn.Linear(5, 1)
optimizador = optim.SGD(modelo.parameters(), lr=0.001)

# Simular: 100 batches por epoch, 5 epochs = 500 steps total
steps_per_epoch = 100
scheduler = OneCycleLR(
    optimizador,
    max_lr=0.1,
    steps_per_epoch=steps_per_epoch,
    epochs=5,
    pct_start=0.3,  # 30% warmup
    anneal_strategy='cos'
)

lrs = []
for epoch in range(5):
    for batch in range(steps_per_epoch):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()

print(f"LR inicial: {lrs[0]:.5f}")
print(f"LR máximo (epoch ~1.5): {max(lrs):.5f}")
print(f"LR final: {lrs[-1]:.5f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: OneCycleLR — LR cíclico con warmup.*

1. Simular: 100 batches por epoch, 5 epochs = 500 steps total

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: Guardar y cargar mejor modelo (checkpoint)

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
X = torch.randn(200, 5)
y = X.sum(dim=1, keepdim=True) + torch.randn(200, 1) * 0.1
val_X = torch.randn(50, 5)
val_y = val_X.sum(dim=1, keepdim=True) + torch.randn(50, 1) * 0.1

modelo = nn.Linear(5, 1)
optimizador = optim.Adam(modelo.parameters(), lr=0.01)
criterio = nn.MSELoss()

mejor_val_loss = float('inf')
checkpoint_path = '/tmp/mejor_modelo_ventas.pth'

for epoch in range(30):
    modelo.train()
    optimizador.zero_grad()
    train_loss = criterio(modelo(X), y)
    train_loss.backward()
    optimizador.step()

    modelo.eval()
    with torch.no_grad():
        val_loss = criterio(modelo(val_X), val_y)

    if val_loss < mejor_val_loss:
        mejor_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': modelo.state_dict(),
            'optimizer_state_dict': optimizador.state_dict(),
            'train_loss': train_loss.item(),
            'val_loss': val_loss.item()
        }, checkpoint_path)
        print(f"✓ Checkpoint guardado en epoch {epoch+1}: val_loss={val_loss.item():.4f}")

# Cargar mejor modelo
checkpoint = torch.load(checkpoint_path)
modelo.load_state_dict(checkpoint['model_state_dict'])
print(f"\nMejor modelo cargado (epoch {checkpoint['epoch']+1}, loss={checkpoint['val_loss']:.4f})")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: Guardar y cargar mejor modelo (checkpoint).*

1. Cargar mejor modelo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 16: TensorBoard con SummaryWriter

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# pip install tensorboard si no está instalado

modelo = nn.Sequential(
    nn.Linear(5, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)
optimizador = optim.Adam(modelo.parameters(), lr=0.01)
criterio = nn.MSELoss()

X = torch.randn(500, 5)
y = X.sum(dim=1, keepdim=True) + torch.randn(500, 1) * 0.3

log_dir = f'/tmp/tensorboard_ventas_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
writer = SummaryWriter(log_dir=log_dir)
print(f"TensorBoard logs en: {log_dir}")

for epoch in range(50):
    modelo.train()
    optimizador.zero_grad()
    loss = criterio(modelo(X[:400]), y[:400])
    loss.backward()
    optimizador.step()

    modelo.eval()
    with torch.no_grad():
        val_loss = criterio(modelo(X[400:]), y[400:])

    # Registrar en TensorBoard
    writer.add_scalar('Loss/train', loss.item(), epoch)
    writer.add_scalar('Loss/val', val_loss.item(), epoch)
    writer.add_scalar('LR', optimizador.param_groups[0]['lr'], epoch)

    # Histograma de pesos cada 10 epochs
    if epoch % 10 == 0:
        for name, param in modelo.named_parameters():
            writer.add_histogram(f'weights/{name}', param, epoch)

    print(f"Epoch {epoch+1}: train_loss={loss.item():.4f}, val_loss={val_loss.item():.4f}")

writer.close()
print("\nPara ver: tensorboard --logdir /tmp/tensorboard_ventas_*")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: TensorBoard con SummaryWriter.*

1. pip install tensorboard si no está instalado
2. Registrar en TensorBoard
3. Histograma de pesos cada 10 epochs

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 17: Early stopping manual

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
X = torch.randn(500, 5)
y = X.sum(dim=1, keepdim=True) + torch.randn(500, 1) * 0.2
val_X = torch.randn(100, 5)
val_y = val_X.sum(dim=1, keepdim=True) + torch.randn(100, 1) * 0.2

modelo = nn.Sequential(nn.Linear(5, 64), nn.ReLU(), nn.Linear(64, 1))
optimizador = optim.Adam(modelo.parameters(), lr=0.01)
criterio = nn.MSELoss()

paciencia = 10
mejor_val_loss = float('inf')
epochs_sin_mejora = 0
mejor_estado = None

for epoch in range(200):
    modelo.train()
    optimizador.zero_grad()
    train_loss = criterio(modelo(X), y)
    train_loss.backward()
    optimizador.step()

    modelo.eval()
    with torch.no_grad():
        val_loss = criterio(modelo(val_X), val_y)

    if val_loss < mejor_val_loss:
        mejor_val_loss = val_loss
        epochs_sin_mejora = 0
        mejor_estado = modelo.state_dict()
        print(f"✓ Epoch {epoch+1}: val_loss={val_loss.item():.4f} (mejor)")
    else:
        epochs_sin_mejora += 1
        if epochs_sin_mejora >= paciencia:
            print(f"\n✗ Early stopping en epoch {epoch+1}. "
                  f"No mejora en {paciencia} epochs.")
            break

    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1}: train={train_loss.item():.4f}, "
              f"val={val_loss.item():.4f}")

modelo.load_state_dict(mejor_estado)
print(f"\nMejor val_loss: {mejor_val_loss:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: Early stopping manual.*

1. `import torch` — Importa las librerías necesarias para el análisis.
2. `import torch.nn as nn` — Importa las librerías necesarias para el análisis.
3. `import torch.optim as optim` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 18: Integrador — Training loop completo con validación, checkpoint y logging

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

torch.manual_seed(42)

# 1. Datos
X = torch.randn(2000, 8)
y = (X[:, 0] * 3.0 + X[:, 1] * 1.5 - X[:, 2] * 2.0 +
     torch.randn(2000) * 0.3).unsqueeze(1)

dataset = TensorDataset(X, y)
train_ds, val_ds, test_ds = random_split(dataset, [1400, 300, 300])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

# 2. Modelo
class RedPrediccionVentas(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        return self.net(x)

modelo = RedPrediccionVentas()
criterio = nn.MSELoss()
optimizador = optim.Adam(modelo.parameters(), lr=0.01)
scheduler = ReduceLROnPlateau(optimizador, mode='min', factor=0.5,
                               patience=5, min_lr=1e-6)

writer = SummaryWriter(log_dir=f'/tmp/ventas_completo_{datetime.now().strftime("%Y%m%d_%H%M%S")}')

# 3. Training loop
n_epochs = 100
paciencia = 15
mejor_val_loss = float('inf')
epochs_sin_mejora = 0
mejor_estado = None

for epoch in range(n_epochs):
    # --- TRAIN ---
    modelo.train()
    train_loss = 0.0
    for batch_x, batch_y in train_loader:
        optimizador.zero_grad()
        pred = modelo(batch_x)
        loss = criterio(pred, batch_y)
        loss.backward()
        optimizador.step()
        train_loss += loss.item()

    # --- VALIDATION ---
    modelo.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            pred = modelo(batch_x)
            loss = criterio(pred, batch_y)
            val_loss += loss.item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)

    scheduler.step(val_loss)

    # Logging
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('LR', optimizador.param_groups[0]['lr'], epoch)

    # Checkpoint + early stopping
    if val_loss < mejor_val_loss:
        mejor_val_loss = val_loss
        epochs_sin_mejora = 0
        mejor_estado = modelo.state_dict().copy()
        torch.save({
            'epoch': epoch,
            'model_state_dict': mejor_estado,
            'optimizer_state_dict': optimizador.state_dict(),
            'val_loss': val_loss
        }, '/tmp/mejor_modelo_completo.pth')
        print(f"✓ Epoch {epoch+1:3d}: train={train_loss:.6f}, val={val_loss:.6f} *MEJOR*")
    else:
        epochs_sin_mejora += 1
        if epochs_sin_mejora >= paciencia:
            print(f"\n✗ Early stopping en epoch {epoch+1}")
            break
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}: train={train_loss:.6f}, val={val_loss:.6f}")

# 4. Evaluación final en test
modelo.load_state_dict(mejor_estado)
modelo.eval()
test_loss = 0.0
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        pred = modelo(batch_x)
        test_loss += criterio(pred, batch_y).item()
test_loss /= len(test_loader)

print(f"\n=== RESULTADOS FINALES ===")
print(f"Mejor val_loss: {mejor_val_loss:.6f}")
print(f"Test loss:      {test_loss:.6f}")
print(f"RMSE test:      {torch.sqrt(torch.tensor(test_loss)):.4f}")
print(f"Total epochs:   {epoch+1}")
writer.close()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador — Training loop completo con validación, checkpoint y logging.*

1. 1. Datos
2. 2. Modelo
3. 3. Training loop
4. --- TRAIN ---
5. --- VALIDATION ---
6. Logging
7. Checkpoint + early stopping
8. 4. Evaluación final en test

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejercicios

1. Implementa un training loop para un modelo de regresión (predicción de compras) con 7 features, usando `F.mse_loss` como pérdida. Entrena por 100 epochs con SGD (lr=0.01) e imprime la pérdida cada 10 epochs.

2. Crea un modelo de clasificación binaria (compra urgente vs no urgente) con 6 features. Usa `F.binary_cross_entropy_with_logits` como pérdida y Adam como optimizador. Evalúa con accuracy al final.

3. Compara `StepLR` (step_size=15, gamma=0.3) vs `ExponentialLR` (gamma=0.95) en un modelo lineal de 5 features. Entrena por 50 epochs y muestra cómo evoluciona el learning rate.

4. Implementa early stopping con paciencia=8 para un modelo de predicción de inventarios con 10 features. Entrena hasta 200 epochs pero detén si no mejora. Guarda el mejor modelo.

5. Usa `torch.save` para guardar un checkpoint completo (modelo, optimizador, epoch, pérdida) cada vez que la pérdida de validación mejore. Carga el checkpoint y verifica que las predicciones coinciden.

6. Configura TensorBoard con `SummaryWriter` y registra: train_loss, val_loss, learning rate, y un histograma de los pesos de la primera capa cada 5 epochs. Explica qué comando usar para ver los resultados.

7. Implementa `OneCycleLR` con max_lr=0.05 para un modelo de clasificación de 4 clases (rotación de productos). Usa 64 batches por epoch y 10 epochs. Muestra los primeros 10 valores de lr.

8. Diseña un training loop completo (integrador) para clasificación multiclase de productos en 5 categorías. Debe incluir: DataLoader con train/val/test split, modelo con 3 capas lineales + BatchNorm + Dropout, Adam, ReduceLROnPlateau, early stopping (patience=10), checkpoint del mejor modelo, y evaluación final en test con accuracy.

---

## Resumen

El training loop en PyTorch sigue un patrón consistente que integra todos los componentes del framework:

| Componente | Rol |
|------------|-----|
| `optimizer.zero_grad()` | Limpiar gradientes previos |
| `loss.backward()` | Calcular gradientes (autograd) |
| `optimizer.step()` | Actualizar pesos |
| `model.train()` / `model.eval()` | Modo correcto para Dropout/BatchNorm |
| `torch.no_grad()` | Inferencia eficiente sin grafo |
| `lr_scheduler` | Ajustar tasa de aprendizaje |
| `torch.save` | Checkpointing |
| `SummaryWriter` | Logging y visualización |

**Optimizadores clave**: Adam (estándar), SGD+momentum (generalización), AdamW (weight decay desacoplado).

**Schedulers clave**: ReduceLROnPlateau (automático), OneCycleLR (rápida convergencia), CosineAnnealingLR (escape de mínimos locales).

**En negocio**: El training loop completo con validación, checkpoint, early stopping y logging es esencial para producción. Permite entrenar modelos de predicción de ventas, clasificación de compras y análisis de inventarios de forma reproducible, monitoreable y con garantías de no sobreajuste.
