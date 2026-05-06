# A23: DataLoader — Pipelines de Datos para Ventas, Compras e Inventarios

## Introducción Teórica

**torch.utils.data** proporciona las herramientas para cargar, procesar y entregar datos eficientemente durante el entrenamiento. El flujo típico es: `Dataset` → `Sampler` → `DataLoader` → modelo.

### Conceptos clave:

1. **Dataset**: Clase abstracta para representar un conjunto de datos. Debe implementar `__len__` (tamaño total) y `__getitem__` (devuelve (features, target) para un índice). Se usa `Dataset` personalizado para datos que no caben en memoria o requieren procesamiento por fila.
2. **TensorDataset**: Envuelve tensores directamente. Cada muestra es el i-ésimo elemento de cada tensor. Útil cuando los datos ya están en memoria.
3. **DataLoader**: Iterador que entrega lotes (batches). Parámetros clave: `batch_size`, `shuffle`, `num_workers` (hijos para carga paralela), `collate_fn` (cómo agrupar muestras en lote), `pin_memory` (acelera transferencia a GPU), `drop_last` (descartar último lote incompleto), `prefetch_factor` (cuántos lotes precargar por worker), `persistent_workers` (mantener workers entre epochs).
4. **Subset**: Crea una vista de un Dataset usando índices. Útil para dividir train/val/test.
5. **random_split**: Divide dataset en partes con proporciones dadas. Baraja previamente.
6. **ConcatDataset**: Une múltiples datasets en uno solo. Las muestras se indexan secuencialmente.
7. **Samplers**: Definen la estrategia de muestreo: `SequentialSampler` (orden secuencial), `RandomSampler` (aleatorio, opcionalmente con reemplazo), `SubsetRandomSampler` (índices específicos aleatorizados), `WeightedRandomSampler` (muestreo ponderado para clases desbalanceadas), `BatchSampler` (genera lotes desde un sampler base).
8. **collate_fn**: Función que recibe una lista de muestras (cada una devuelta por `__getitem__`) y las agrupa en un batch. Personalizable para padding de secuencias, mezcla de tipos, etc. Por defecto usa `default_collate`.
9. **worker_init_fn**: Función que se ejecuta al iniciar cada worker. Útil para semillas aleatorias por worker.

### Aplicaciones en negocio:

- **Ventas**: Dataset con transacciones diarias → batches para predecir demanda del día siguiente.
- **Compras**: ConcatDataset de órdenes de compra de múltiples proveedores o archivos mensuales.
- **Inventarios**: WeightedRandomSampler para balancear clases de rotación (alta/media/baja) cuando hay desbalance.

---

## Ejemplos

### Ejemplo 1: Dataset personalizado para datos de ventas

```python
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class DatasetVentas(Dataset):
    """Dataset para predicción de ventas: features → unidades vendidas"""
    def __init__(self, num_muestras=1000):
        super().__init__()
        torch.manual_seed(42)
        # Simular datos de ventas: [precio, descuento, rating, gasto_publicidad, dia_semana]
        self.features = torch.randn(num_muestras, 5)
        # Target: unidades vendidas (con ruido)
        self.targets = (self.features[:, 0] * 2.0 +
                        self.features[:, 1] * 1.5 +
                        torch.randn(num_muestras) * 0.5).unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

dataset = DatasetVentas(100)
print(f"Tamaño del dataset: {len(dataset)}")
x, y = dataset[0]
print(f"Features: {x}, Target: {y}")

# DataLoader básico
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
batch_x, batch_y = next(iter(dataloader))
print(f"Batch features: {batch_x.shape}, Batch targets: {batch_y.shape}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: Dataset personalizado para datos de ventas, compras o inventarios.*

1. Simular datos de ventas, compras o inventarios: [precio, descuento, rating, gasto_publicidad, dia_semana]
2. Target: unidades vendidas (con ruido)
3. DataLoader básico

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: TensorDataset — Combinar tensores de features y targets

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

# Datos de compras: 200 órdenes con 4 features
features = torch.randn(200, 4)
targets = torch.randint(0, 2, (200, 1)).float()  # 0=no urgente, 1=urgente

dataset = TensorDataset(features, targets)
print(f"Tamaño del TensorDataset: {len(dataset)}")
x, y = dataset[5]
print(f"Muestra 5 - features: {x}, target: {y}")

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
batch_x, batch_y = next(iter(dataloader))
print(f"Batch: X={batch_x.shape}, y={batch_y.shape}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: TensorDataset — Combinar tensores de features y targets.*

1. Datos de compras: 200 órdenes con 4 features

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: DataLoader básico — batch_size=32, shuffle=True

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

features = torch.randn(500, 3)
targets = torch.randn(500, 1)
dataset = TensorDataset(features, targets)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
print(f"Número de batches: {len(dataloader)}")  # ceil(500/32) = 16

for epoch in range(2):
    print(f"\nEpoch {epoch}:")
    for i, (x_batch, y_batch) in enumerate(dataloader):
        if i < 3:  # solo primeros 3 batches
            print(f"  Batch {i}: X={x_batch.shape}, y={y_batch.shape}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: DataLoader básico — batch_size=32, shuffle=True.*

1. `import torch` — Importa las librerías necesarias para el análisis.
2. `from torch.utils.data import TensorDataset, DataLoader` — Importa las librerías necesarias para el análisis.
3. `print(f"Número de batches: {len(dataloader)}")  # ceil(500/32) = 16` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: DataLoader con num_workers=4 — Cargar datos en paralelo

```python
import torch
from torch.utils.data import Dataset, DataLoader
import time

class DatasetLento(Dataset):
    """Simula dataset con carga lenta (lectura de disco)"""
    def __init__(self, n=200):
        super().__init__()
        self.data = torch.randn(n, 10)
        self.labels = torch.randn(n, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        time.sleep(0.01)  # simula carga de archivo
        return self.data[idx], self.labels[idx]

dataset = DatasetLento(200)

# Sin workers
start = time.time()
dl_single = DataLoader(dataset, batch_size=16, num_workers=0)
for _ in dl_single:
    pass
print(f"Sin workers: {time.time() - start:.2f}s")

# Con 4 workers
start = time.time()
dl_multi = DataLoader(dataset, batch_size=16, num_workers=4)
for _ in dl_multi:
    pass
print(f"Con 4 workers: {time.time() - start:.2f}s")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: DataLoader con num_workers=4 — Cargar datos en paralelo.*

1. Sin workers
2. Con 4 workers

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: random_split — 80/20 train/test desde Dataset

```python
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader

dataset = TensorDataset(torch.randn(1000, 5), torch.randn(1000, 1))

train_len = int(0.8 * len(dataset))
test_len = len(dataset) - train_len
train_ds, test_ds = random_split(dataset, [train_len, test_len])

print(f"Train: {len(train_ds)} muestras, Test: {len(test_ds)} muestras")

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

# Verificar que no hay overlap
train_indices = set(train_ds.indices)
test_indices = set(test_ds.indices)
print(f"Overlap train-test: {len(train_indices & test_indices)}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: random_split — 80/20 train/test desde Dataset.*

1. Verificar que no hay overlap

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: Subset — Seleccionar subconjunto por índices

```python
import torch
from torch.utils.data import TensorDataset, Subset, DataLoader

dataset = TensorDataset(
    torch.arange(100).float().unsqueeze(1),
    torch.arange(100).float().unsqueeze(1)
)

# Seleccionar productos con id par para entrenamiento
indices_train = list(range(0, 100, 2))  # índices pares
indices_val = list(range(1, 100, 2))    # índices impares

train_subset = Subset(dataset, indices_train)
val_subset = Subset(dataset, indices_val)

print(f"Train subset: {len(train_subset)} muestras")
print(f"Val subset: {len(val_subset)} muestras")

# Verificar
print(f"Primeros 3 train: {[train_subset[i][0].item() for i in range(3)]}")
print(f"Primeros 3 val: {[val_subset[i][0].item() for i in range(3)]}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: Subset — Seleccionar subconjunto por índices.*

1. Seleccionar productos con id par para entrenamiento
2. Verificar

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: ConcatDataset — Unir ventas de varios archivos

```python
import torch
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader

# Simular 3 "archivos" mensuales de ventas
enero = TensorDataset(torch.randn(300, 4), torch.randn(300, 1))
febrero = TensorDataset(torch.randn(250, 4), torch.randn(250, 1))
marzo = TensorDataset(torch.randn(280, 4), torch.randn(280, 1))

dataset_completo = ConcatDataset([enero, febrero, marzo])
print(f"Total muestras: {len(dataset_completo)}")  # 830

dataloader = DataLoader(dataset_completo, batch_size=64, shuffle=True)
batch_x, batch_y = next(iter(dataloader))
print(f"Batch del dataset combinado: X={batch_x.shape}, y={batch_y.shape}")

# También se puede usar con Subset sobre ConcatDataset
train_len = int(0.8 * len(dataset_completo))
val_len = len(dataset_completo) - train_len
train_ds, val_ds = torch.utils.data.random_split(dataset_completo, [train_len, val_len])
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: ConcatDataset — Unir ventas de varios archivos.*

1. Simular 3 "archivos" mensuales de ventas
2. También se puede usar con Subset sobre ConcatDataset

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: WeightedRandomSampler — Muestreo ponderado para clases desbalanceadas

```python
import torch
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

# Clases desbalanceadas: 90% "no compra" (clase 0), 10% "compra" (clase 1)
n_muestras = 1000
features = torch.randn(n_muestras, 5)
labels = torch.cat([torch.zeros(900), torch.ones(100)]).long()

dataset = TensorDataset(features, labels)

# Calcular pesos: inverso de frecuencia
class_counts = torch.bincount(labels)
weights = 1.0 / class_counts[labels].float()
print(f"Pesos por muestra (primeros 10): {weights[:10]}")

sampler = WeightedRandomSampler(
    weights=weights,
    num_samples=len(weights),
    replacement=True
)

dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

# Verificar balance en un batch
batch_x, batch_y = next(iter(dataloader))
print(f"Batch labels: {batch_y}")
print(f"Clase 1 en batch: {batch_y.sum().item()}/{batch_y.shape[0]}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: WeightedRandomSampler — Muestreo ponderado para clases desbalanceadas.*

1. Clases desbalanceadas: 90% "no compra" (clase 0), 10% "compra" (clase 1)
2. Calcular pesos: inverso de frecuencia
3. Verificar balance en un batch

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: SubsetRandomSampler — Índices específicos

```python
import torch
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler

dataset = TensorDataset(torch.randn(100, 5), torch.randn(100, 1))

# Seleccionar 30 muestras aleatorias para validación
indices = torch.randperm(100)[:30].tolist()
print(f"Índices seleccionados (primeros 10): {indices[:10]}")

sampler = SubsetRandomSampler(indices)
dataloader = DataLoader(dataset, batch_size=8, sampler=sampler)

print(f"Batches en dataloader: {len(dataloader)}")
for i, (x, y) in enumerate(dataloader):
    print(f"  Batch {i}: {x.shape}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: SubsetRandomSampler — Índices específicos.*

1. Seleccionar 30 muestras aleatorias para validación

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: BatchSampler — Lotes desde sampler base

```python
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, BatchSampler

dataset = TensorDataset(torch.arange(20).float().unsqueeze(1),
                         torch.arange(20).float().unsqueeze(1))

sampler = SequentialSampler(dataset)  # 0, 1, 2, ..., 19
batch_sampler = BatchSampler(sampler, batch_size=5, drop_last=False)

print("Lotes generados por BatchSampler:")
for indices in batch_sampler:
    print(f"  Índices: {indices}")

dataloader = DataLoader(dataset, batch_sampler=batch_sampler)
for x, y in dataloader:
    print(f"  Batch: {x.squeeze().tolist()}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: BatchSampler — Lotes desde sampler base.*

1. `import torch` — Importa las librerías necesarias para el análisis.
2. `from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, BatchSampler` — Importa las librerías necesarias para el análisis.
3. `print("Lotes generados por BatchSampler:")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: Custom collate_fn — Padding para lotes de secuencias

```python
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

class DatasetComprasSecuencias(Dataset):
    """Cada "compra" tiene secuencia de productos de longitud variable"""
    def __init__(self, n=50):
        super().__init__()
        self.data = [torch.randn(torch.randint(3, 10, (1,)).item(), 4)
                     for _ in range(n)]
        self.targets = torch.randn(n, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def collate_fn_ventas(batch):
    """Agrupa secuencias con padding"""
    secuencias, targets = zip(*batch)
    secuencias_padded = pad_sequence(secuencias, batch_first=True, padding_value=0.0)
    targets = torch.stack(targets)
    return secuencias_padded, targets

dataset = DatasetComprasSecuencias(50)
dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn_ventas)

batch_x, batch_y = next(iter(dataloader))
print(f"Batch padded: {batch_x.shape}")  # [8, max_len, 4]
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: Custom collate_fn — Padding para lotes de secuencias.*

1. `import torch` — Importa las librerías necesarias para el análisis.
2. `from torch.nn.utils.rnn import pad_sequence` — Importa las librerías necesarias para el análisis.
3. `from torch.utils.data import Dataset, DataLoader` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: DataLoader con pin_memory=True — Transferencia más rápida a GPU

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(torch.randn(1000, 10), torch.randn(1000, 1))

dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    pin_memory=True  # page-locked memory para transferencia GPU más rápida
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for batch_x, batch_y in dataloader:
    batch_x = batch_x.to(device, non_blocking=True)
    batch_y = batch_y.to(device, non_blocking=True)
    break

print(f"Datos en {device}: X={batch_x.device}, y={batch_y.device}")
print("pin_memory=True acelera transferencia CPU→GPU")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: DataLoader con pin_memory=True — Transferencia más rápida a GPU.*

1. `import torch` — Importa las librerías necesarias para el análisis.
2. `from torch.utils.data import TensorDataset, DataLoader` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: DataLoader con drop_last=True — Descartar lote incompleto

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(torch.randn(100, 5), torch.randn(100, 1))

dl_drop = DataLoader(dataset, batch_size=32, drop_last=True)
dl_no_drop = DataLoader(dataset, batch_size=32, drop_last=False)

print(f"Sin drop_last: {len(dl_no_drop)} batches  (último batch: {100 % 32} muestras)")
print(f"Con drop_last: {len(dl_drop)} batches    (descartado lote de {100 % 32} muestras)")

# Verificar último batch con drop_last=False
for i, (x, y) in enumerate(dl_no_drop):
    if i == len(dl_no_drop) - 1:
        print(f"Último batch (sin drop): {x.shape[0]} muestras")

for i, (x, y) in enumerate(dl_drop):
    if i == len(dl_drop) - 1:
        print(f"Último batch (con drop): {x.shape[0]} muestras")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: DataLoader con drop_last=True — Descartar lote incompleto.*

1. Verificar último batch con drop_last=False

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: prefetch_factor — Cuántos lotes precargar

```python
import torch
from torch.utils.data import Dataset, DataLoader
import time

class DatasetLento2(Dataset):
    def __init__(self, n=200):
        super().__init__()
        self.data = torch.randn(n, 10)
        self.labels = torch.randn(n, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        time.sleep(0.02)
        return self.data[idx], self.labels[idx]

dataset = DatasetLento2(200)

# prefetch_factor=2 (default): precarga 2 lotes por worker
dl = DataLoader(dataset, batch_size=16, num_workers=2, prefetch_factor=2)
start = time.time()
for _ in dl:
    pass
print(f"prefetch_factor=2: {time.time() - start:.2f}s")

# prefetch_factor=4: más precarga pero más memoria
dl2 = DataLoader(dataset, batch_size=16, num_workers=2, prefetch_factor=4)
start = time.time()
for _ in dl2:
    pass
print(f"prefetch_factor=4: {time.time() - start:.2f}s")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: prefetch_factor — Cuántos lotes precargar.*

1. prefetch_factor=2 (default): precarga 2 lotes por worker
2. prefetch_factor=4: más precarga pero más memoria

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: persistent_workers=True — Mantener workers entre epochs

```python
import torch
from torch.utils.data import Dataset, DataLoader
import time

class DatasetConInit(Dataset):
    def __init__(self):
        super().__init__()
        self.data = torch.randn(100, 5)
        self.labels = torch.randn(100, 1)
        print(f"  Dataset.__init__() llamado")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataset = DatasetConInit()

# Sin persistent_workers: workers se reinician cada epoch
print("\nSin persistent_workers:")
dl = DataLoader(dataset, batch_size=16, num_workers=2, persistent_workers=False)
for epoch in range(3):
    for _ in dl:
        pass
    print(f"  Epoch {epoch} completada")

# Con persistent_workers: workers se mantienen vivos entre epochs
print("\nCon persistent_workers=True:")
dl2 = DataLoader(dataset, batch_size=16, num_workers=2, persistent_workers=True)
for epoch in range(3):
    for _ in dl2:
        pass
    print(f"  Epoch {epoch} completada")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: persistent_workers=True — Mantener workers entre epochs.*

1. Sin persistent_workers: workers se reinician cada epoch
2. Con persistent_workers: workers se mantienen vivos entre epochs

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 16: Iterar sobre DataLoader en training loop

```python
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Dataset sintético de ventas
X = torch.randn(200, 5)
y = (X.sum(dim=1, keepdim=True) + torch.randn(200, 1) * 0.1)
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

modelo = nn.Linear(5, 1)
criterio = nn.MSELoss()
optimizador = torch.optim.SGD(modelo.parameters(), lr=0.01)

n_epochs = 3
for epoch in range(n_epochs):
    epoch_loss = 0.0
    for batch_x, batch_y in dataloader:
        pred = modelo(batch_x)
        loss = criterio(pred, batch_y)

        optimizador.zero_grad()
        loss.backward()
        optimizador.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}: pérdida = {epoch_loss/len(dataloader):.6f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: Iterar sobre DataLoader en training loop.*

1. Dataset sintético de ventas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 17: DataLoader para dataset de imágenes de productos

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

# Dataset simulado de "imágenes de productos"
class DatasetImagenesProductos(Dataset):
    def __init__(self, num_imagenes=50, transform=None):
        super().__init__()
        self.num_imagenes = num_imagenes
        self.transform = transform or transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        # Simular imágenes como tensores aleatorios
        self.imagenes = torch.randn(num_imagenes, 3, 64, 64)
        self.etiquetas = torch.randint(0, 5, (num_imagenes,))

    def __len__(self):
        return self.num_imagenes

    def __getitem__(self, idx):
        img = self.imagenes[idx]
        # Aplicar transform (simulado)
        label = self.etiquetas[idx]
        return img, label

dataset = DatasetImagenesProductos(num_imagenes=100)
dataloader = DataLoader(
    dataset, batch_size=16, shuffle=True,
    num_workers=2, pin_memory=True
)

batch_x, batch_y = next(iter(dataloader))
print(f"Batch imágenes: {batch_x.shape}")  # [16, 3, 64, 64]
print(f"Batch etiquetas: {batch_y.shape}")  # [16]
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: DataLoader para dataset de imágenes de productos.*

1. Dataset simulado de "imágenes de productos"
2. Simular imágenes como tensores aleatorios
3. Aplicar transform (simulado)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 18: Integrador — DataLoader completo para entrenamiento

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler

# Dataset completo de ventas
class DatasetVentasCompleto(Dataset):
    def __init__(self, n=2000):
        super().__init__()
        torch.manual_seed(42)
        self.features = torch.randn(n, 6)
        self.targets = (self.features[:, 0] * 3.0 +
                        self.features[:, 1] * 1.5 +
                        torch.randn(n) * 0.3).unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

dataset = DatasetVentasCompleto(2000)
train_ds, val_ds = random_split(dataset, [1600, 400])

train_loader = DataLoader(
    train_ds,
    batch_size=64,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True,
    persistent_workers=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

modelo = nn.Sequential(
    nn.Linear(6, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)
criterio = nn.MSELoss()
optimizador = torch.optim.Adam(modelo.parameters(), lr=0.01)

for epoch in range(5):
    modelo.train()
    train_loss = 0.0
    for batch_x, batch_y in train_loader:
        optimizador.zero_grad()
        pred = modelo(batch_x)
        loss = criterio(pred, batch_y)
        loss.backward()
        optimizador.step()
        train_loss += loss.item()

    modelo.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            pred = modelo(batch_x)
            loss = criterio(pred, batch_y)
            val_loss += loss.item()

    print(f"Epoch {epoch+1}: train_loss={train_loss/len(train_loader):.6f}, "
          f"val_loss={val_loss/len(val_loader):.6f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador — DataLoader completo para entrenamiento.*

1. Dataset completo de ventas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejercicios

1. Crea un `Dataset` personalizado para datos de compras que tenga 5 features numéricas y devuelva un target booleano (0/1) indicando si la compra fue urgente. Usa 500 muestras simuladas.

2. Usa `TensorDataset` para envolver dos tensores: `X` de forma (300, 8) e `y` de forma (300,). Crea un DataLoader con batch_size=16 y verifica la forma de cada batch.

3. Con `random_split`, divide un dataset de 1000 muestras en 70% train, 15% val, 15% test. Calcula cuántas muestras tiene cada split.

4. Usa `SubsetRandomSampler` para crear un DataLoader que solo entregue las primeras 50 muestras de un dataset de 200, en orden aleatorio, con batch_size=10.

5. Implementa un `collate_fn` personalizado para un dataset que devuelve tuplas de (features, target) donde features es un tensor de longitud variable (simula productos diferentes en cada compra). Aplica padding con ceros.

6. Crea un `WeightedRandomSampler` para un dataset con 800 muestras de clase 0 y 200 de clase 1. Calcula los pesos para balancear las clases y verifica que un batch de 32 tiene aproximadamente 50% de cada clase.

7. Usa `ConcatDataset` para unir 3 datasets de inventarios de diferentes almacenes (200, 300, 250 muestras cada uno) y crea un DataLoader con shuffle=True que itere sobre todos.

8. Diseña un pipeline completo de datos para un problema de clasificación de productos con 5000 muestras, 10 features, 4 clases desbalanceadas. Usa random_split (80/20), WeightedRandomSampler en train, DataLoader con num_workers=4 y pin_memory=True, y verifica que funciona en un training loop de 3 epochs.

---

## Resumen

`torch.utils.data` proporciona un pipeline modular y eficiente para alimentar datos al modelo:

| Componente | Propósito |
|------------|-----------|
| `Dataset` | Representar datos (implementar `__len__`, `__getitem__`) |
| `TensorDataset` | Envolver tensores como dataset |
| `DataLoader` | Iterar por lotes con paralelismo |
| `random_split` / `Subset` | Dividir dataset |
| `ConcatDataset` | Unir datasets |
| `WeightedRandomSampler` | Balancear clases |
| `SubsetRandomSampler` | Muestreo por índices |
| `BatchSampler` | Generar lotes desde sampler |
| `collate_fn` | Agrupar muestras en batch |
| `pin_memory` | Acelerar transferencia a GPU |

**En negocio**: El pipeline de datos es crítico para escalar modelos de ventas, compras e inventarios. Un DataLoader bien configurado (paralelismo, prefetch, pin_memory) puede reducir el tiempo de entrenamiento drásticamente al mantener la GPU ocupada sin esperar por datos.
