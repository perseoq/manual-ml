# AP07 — Cheatsheet PyTorch

## 1. Tensores

```python
import torch
import numpy as np

# Creación
t = torch.tensor([[1, 2], [3, 4]])
t = torch.tensor(42)
t = torch.Tensor(5, 3)             # tensor vacío 5x3
t = torch.zeros(3, 4)              # ceros
t = torch.ones(3, 4)               # unos
t = torch.eye(5)                   # identidad
t = torch.arange(10)               # [0..9]
t = torch.linspace(0, 1, 5)        # [0, 0.25, 0.5, 0.75, 1]

# Tensores aleatorios
t = torch.rand(3, 4)               # uniforme [0, 1)
t = torch.randn(3, 4)              # normal estándar
t = torch.randint(0, 10, (3, 4))   # enteros
t = torch.randperm(10)             # permutación aleatoria

# Propiedades
t.shape          # torch.Size([3, 4])
t.size()         # igual
t.dtype          # torch.float32
t.device         # cpu o cuda:0
t.requires_grad  # bool

# Conversión con numpy
arr = t.numpy()                     # tensor -> numpy (comparten memoria)
t = torch.from_numpy(arr)           # numpy -> tensor
t = torch.tensor(arr)               # copia desde numpy

# Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t = t.to(device)
t = t.cuda() if torch.cuda.is_available() else t
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*1. Tensores.*

1. Creación
2. Tensores aleatorios
3. Propiedades
4. Conversión con numpy
5. Dispositivo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 2. Operaciones con Tensores

```python
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

# Aritméticas
a + b, torch.add(a, b)             # suma
a - b                              # resta
a * b                              # multiplicación elemento a elemento
a / b                              # división
a ** 2                             # potencia

# Producto matricial
a @ b                              # @ operator
torch.mm(a, b)                     # multiplicación matricial
torch.matmul(a, b)                 # multiplicación matricial (broadcasting)

# Reducción
a.sum()                            # suma total
a.mean()                           # media
a.std()                            # desviación estándar
a.max()                            # valor máximo
a.min()                            # valor mínimo
a.argmax()                         # índice del máximo
a.argmin()                         # índice del mínimo
a.sum(dim=0)                       # suma por columna
a.mean(dim=1)                      # media por fila

# Cambio de forma
a.view(4, 1)                       # reshape (vista)
a.reshape(4, 1)                    # reshape (puede copiar)
a.transpose(0, 1)                  # transpuesta
a.T                                # transpuesta (2D)
a.unsqueeze(0)                     # añadir dim: (1, 2, 3)
a.squeeze()                        # eliminar dims de tamaño 1
a.flatten()                        # aplanar
a.permute(2, 0, 1)                 # reordenar dimensiones

# Concatenación
torch.cat([a, b], dim=0)           # concat en dim 0
torch.cat([a, b], dim=1)           # concat en dim 1
torch.stack([a, b], dim=0)         # nuevo eje

# Indexación (similar a numpy)
a[0, 1]                            # elemento
a[:, 1]                            # columna 1
a[a > 2]                           # indexación booleana
torch.where(a > 2, a, torch.zeros_like(a))

# Clamping
torch.clamp(a, min=0, max=3)       # recortar valores

# Operaciones in-place (con sufijo _)
a.add_(5)                          # a = a + 5
a.mul_(2)                          # a = a * 2
a.zero_()                          # a = 0
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*2. Operaciones con Tensores.*

1. Aritméticas
2. Producto matricial
3. Reducción
4. Cambio de forma
5. Concatenación
6. Indexación (similar a numpy)
7. Clamping
8. Operaciones in-place (con sufijo _)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 3. Autograd — Diferenciación Automática

```python
# Habilitar gradientes
x = torch.randn(3, requires_grad=True)
w = torch.randn(3, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)

# Forward
y = torch.matmul(w, x) + b

# Calcular gradientes
y.backward(torch.ones_like(y))

# Gradientes
print(x.grad)
print(w.grad)
print(b.grad)

# Deshabilitar gradientes temporalmente
with torch.no_grad():
    y = model(x)                    # inferencia sin tracking

# Detach (separar del grafo computacional)
x_detached = x.detach()

# Zero grad (limpiar gradientes)
model.zero_grad()
# o
optimizer.zero_grad()

# Grad scaler (mixed precision)
scaler = torch.cuda.amp.GradScaler()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*3. Autograd — Diferenciación Automática.*

1. Habilitar gradientes
2. Forward
3. Calcular gradientes
4. Gradientes
5. Deshabilitar gradientes temporalmente
6. Detach (separar del grafo computacional)
7. Zero grad (limpiar gradientes)
8. Grad scaler (mixed precision)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 4. nn.Module — Modelos

```python
import torch.nn as nn
import torch.nn.functional as F

# Modelo simple
class VentasMLP(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=64, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Instanciar
model = VentasMLP(input_dim=50).to(device)
print(model)

# Parámetros
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# Contar parámetros
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*4. nn.Module — Modelos.*

1. Modelo simple
2. Instanciar
3. Parámetros
4. Contar parámetros

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 5. Capas Comunes

```python
# Lineales
nn.Linear(in_features=100, out_features=50)
nn.Bilinear(in1_features=10, in2_features=20, out_features=5)

# Convolucionales
nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3)
nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2)  # deconv
nn.MaxPool2d(kernel_size=2, stride=2)
nn.AdaptiveAvgPool2d((1, 1))

# Recurrentes
nn.RNN(input_size=10, hidden_size=20, num_layers=2)
nn.LSTM(input_size=10, hidden_size=20, batch_first=True)
nn.GRU(input_size=10, hidden_size=20, bidirectional=True)

# Normalización
nn.BatchNorm1d(64)
nn.BatchNorm2d(32)
nn.LayerNorm(64)
nn.Dropout(0.5)

# Embedding
nn.Embedding(num_embeddings=10000, embedding_dim=128)

# Activaciones
nn.ReLU(), nn.Sigmoid(), nn.Tanh(), nn.LeakyReLU(0.1)
nn.GELU(), nn.Softmax(dim=1), nn.LogSoftmax(dim=1)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*5. Capas Comunes.*

1. Lineales
2. Convolucionales
3. Recurrentes
4. Normalización
5. Embedding
6. Activaciones

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 6. DataLoader

```python
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd

# TensorDataset
X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32)
dataset = TensorDataset(X_tensor, y_tensor)

# Dataset personalizado
class VentasDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.features = torch.tensor(df.drop("target", axis=1).values,
                                      dtype=torch.float32)
        self.targets = torch.tensor(df["target"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# DataLoader
dataset = VentasDataset("ventas.csv")
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# Iterar
for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)
    # forward / backward
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*6. DataLoader.*

1. TensorDataset
2. Dataset personalizado
3. DataLoader
4. Iterar
5. forward / backward

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 7. Training Loop

```python
import torch.optim as optim

# Modelo, pérdida, optimizador
model = VentasMLP(input_dim=50).to(device)
criterion = nn.BCEWithLogitsLoss()       # binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5
)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * X_batch.size(0)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            val_loss += loss.item() * X_batch.size(0)

    train_loss /= len(train_loader.dataset)
    val_loss /= len(val_loader.dataset)

    scheduler.step(val_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*7. Training Loop.*

1. Modelo, pérdida, optimizador
2. Scheduler
3. Training loop
4. Training
5. Validation

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 8. Optimizadores

```python
# Optimizadores disponibles
optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)
optim.Adagrad(model.parameters(), lr=0.01)
optim.Adadelta(model.parameters(), lr=1.0)

# Optimizador con diferentes learning rates
optimizer = optim.Adam([
    {"params": model.fc1.parameters(), "lr": 0.001},
    {"params": model.fc2.parameters(), "lr": 0.0001},
], lr=0.01)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*8. Optimizadores.*

1. Optimizadores disponibles
2. Optimizador con diferentes learning rates

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 9. Schedulers

```python
# StepLR
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# MultiStepLR
scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[30, 60, 80], gamma=0.5
)

# ExponentialLR
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# CosineAnnealingLR
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100, eta_min=1e-6
)

# ReduceLROnPlateau (basado en métrica)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5, threshold=1e-4
)

# OneCycleLR
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.01, steps_per_epoch=len(train_loader),
    epochs=100
)

# Warm-up manual
def warmup_lr(optimizer, epoch, warmup_epochs=5):
    if epoch < warmup_epochs:
        lr = 0.001 * (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*9. Schedulers.*

1. StepLR
2. MultiStepLR
3. ExponentialLR
4. CosineAnnealingLR
5. ReduceLROnPlateau (basado en métrica)
6. OneCycleLR
7. Warm-up manual

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 10. Guardar y Cargar

```python
# Guardar solo pesos (recomendado)
torch.save(model.state_dict(), "modelo_ventas.pth")

# Cargar pesos
model = VentasMLP(input_dim=50)
model.load_state_dict(torch.load("modelo_ventas.pth", map_location=device))
model.to(device)
model.eval()

# Guardar checkpoint completo
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": val_loss,
    "config": {"input_dim": 50, "hidden_dim": 64}
}
torch.save(checkpoint, "checkpoint_epoch50.pth")

# Cargar checkpoint
checkpoint = torch.load("checkpoint_epoch50.pth")
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
epoch = checkpoint["epoch"]
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*10. Guardar y Cargar.*

1. Guardar solo pesos (recomendado)
2. Cargar pesos
3. Guardar checkpoint completo
4. Cargar checkpoint

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 11. Transfer Learning

```python
from torchvision import models

# Cargar modelo pre-entrenado
resnet = models.resnet18(pretrained=True)

# Congelar capas
for param in resnet.parameters():
    param.requires_grad = False

# Reemplazar última capa
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, 10)  # 10 clases

# O descongelar para fine-tuning
for param in resnet.layer4.parameters():
    param.requires_grad = True

# Entrenar
optimizer = optim.Adam(filter(lambda p: p.requires_grad, resnet.parameters()), lr=0.001)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*11. Transfer Learning.*

1. Cargar modelo pre-entrenado
2. Congelar capas
3. Reemplazar última capa
4. O descongelar para fine-tuning
5. Entrenar

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 12. Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for X_batch, y_batch in dataloader:
    optimizer.zero_grad()

    with autocast():                        # forward en FP16
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

    scaler.scale(loss).backward()           # backward escalado
    scaler.step(optimizer)                  # optimizer step
    scaler.update()                         # actualiza escala
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*12. Mixed Precision Training.*

1. `from torch.cuda.amp import autocast, GradScaler` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 13. Distribución y DataParallel

```python
# DataParallel (múltiples GPUs)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.to(device)

# DistributedDataParallel (recomendado para multi-GPU)
# python -m torch.distributed.launch train.py
import torch.distributed as dist
dist.init_process_group("nccl", rank=rank, world_size=world_size)
model = nn.DistributedDataParallel(model, device_ids=[rank])
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*13. Distribución y DataParallel.*

1. DataParallel (múltiples GPUs)
2. DistributedDataParallel (recomendado para multi-GPU)
3. python -m torch.distributed.launch train.py

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 14. nn.Sequential y Módulos

```python
# Sequential simple
model = nn.Sequential(
    nn.Linear(50, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

# Sequential con OrderedDict
from collections import OrderedDict
model = nn.Sequential(OrderedDict([
    ("fc1", nn.Linear(50, 64)),
    ("relu1", nn.ReLU()),
    ("drop1", nn.Dropout(0.3)),
    ("fc2", nn.Linear(64, 1))
]))

# Registro de módulos hijos
class MiModelo(nn.Module):
    def __init__(self):
        super().__init__()
        self.bloque = nn.Sequential(
            nn.Linear(50, 64),
            nn.ReLU()
        )
        self.salida = nn.Linear(64, 1)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*14. nn.Sequential y Módulos.*

1. Sequential simple
2. Sequential con OrderedDict
3. Registro de módulos hijos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 15. TensorBoard en PyTorch

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/experimento1")

for epoch in range(num_epochs):
    # ... training ...
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

# Histograma de pesos
writer.add_histogram("fc1/weights", model.fc1.weight, epoch)

# Gráfo del modelo
dummy_input = torch.randn(1, 50).to(device)
writer.add_graph(model, dummy_input)

# Imágenes
writer.add_images("predictions", images, epoch)

writer.close()
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*15. TensorBoard en PyTorch.*

1. ... training ...
2. Histograma de pesos
3. Gráfo del modelo
4. Imágenes

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## Referencia Rápida

| Operación | Código |
|-----------|--------|
| Tensor | `torch.tensor([1, 2, 3])` |
| GPU | `t.to("cuda")` o `t.cuda()` |
| Autograd | `loss.backward()`, `torch.no_grad()` |
| Modelo | `class MiModelo(nn.Module): def forward(self, x):` |
| Dataset | `class MiDataset(Dataset): __len__`, `__getitem__` |
| DataLoader | `DataLoader(dataset, batch_size=32, shuffle=True)` |
| Training Loop | `optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()` |
| Guardar | `torch.save(model.state_dict(), "modelo.pth")` |
| Cargar | `model.load_state_dict(torch.load("modelo.pth"))` |
| Scheduler | `optim.lr_scheduler.StepLR(optimizer, step_size=30)` |
| Mixed Precision | `autocast()`, `GradScaler().scale(loss).backward()` |
