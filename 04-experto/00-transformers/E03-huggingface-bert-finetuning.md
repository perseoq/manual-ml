# E03: Hugging Face — Fine-Tuning de BERT para Clasificación

## Objetivo
Aprender a fine-tunear BERT (y variantes) para clasificar descripciones de productos, reseñas y categorías de catálogos B2B usando Hugging Face Transformers + Trainer API.

---

## 1. Fundamentos Teóricos

### 1.1 Fine-Tuning: ¿Por qué y cómo?

**Pre-training**: BERT se entrena en corpus masivos (Wikipedia + BookCorpus) con:
- **Masked Language Model (MLM)**: Predecir tokens enmascarados.
- **Next Sentence Prediction (NSP)**: Predecir si dos oraciones son consecutivas.

**Fine-Tuning**: Tomar el modelo pre-entrenado y ajustarlo con pocos datos etiquetados para una tarea específica.

**Ventajas**:
- Requiere menos datos que entrenar desde cero.
- Converge más rápido (épocas vs cientos de épocas).
- Generaliza mejor (transfer learning).

### 1.2 Componentes Clave

#### AutoTokenizer
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*AutoTokenizer.*

1. `from transformers import AutoTokenizer` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---


- `__call__`: Tokeniza + padding + truncation + return tensors.
- `padding='max_length'`: Rellena hasta max_length.
- `truncation=True`: Trunca textos que exceden max_length.
- `return_tensors='pt'`: Devuelve torch tensors.

#### AutoModelForSequenceClassification
```python
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased",
    num_labels=5,
    id2label={...},
    label2id={...},
    ignore_mismatched_sizes=True
)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*AutoModelForSequenceClassification.*

1. `from transformers import AutoModelForSequenceClassification` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



#### Trainer + TrainingArguments
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./resultados",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    learning_rate=2e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Trainer + TrainingArguments.*

1. `from transformers import Trainer, TrainingArguments` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### 1.3 Dataset con Hugging Face

```python
from datasets import Dataset
dataset = Dataset.from_pandas(df)
dataset = dataset.map(lambda x: tokenizer(x['text'], padding='max_length', truncation=True, max_length=128))
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*1.3 Dataset con Hugging Face.*

1. `from datasets import Dataset` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### 1.4 Métricas

```python
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(-1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='weighted'),
        'precision': precision_score(labels, predictions, average='weighted'),
        'recall': recall_score(labels, predictions, average='weighted'),
    }
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*1.4 Métricas.*

1. `from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

## 2. Ejemplos Prácticos

### Ejemplo 1: Cargar AutoTokenizer desde BERT Multilingüe

```python
from transformers import AutoTokenizer

# Modelo multilingüe: soporta español, inglés, etc.
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

print(f"Vocabulario: {tokenizer.vocab_size} tokens")
print(f"Max length: {tokenizer.model_max_length}")
print(f"Special tokens: [CLS]={tokenizer.cls_token_id}, [SEP]={tokenizer.sep_token_id}, [PAD]={tokenizer.pad_token_id}")

# Probar tokenización
texto = "Laptop gaming con RTX 3060"
tokens = tokenizer(texto)
print(f"\nTexto: '{texto}'")
print(f"Input IDs: {tokens['input_ids']}")
print(f"Decodificado: {tokenizer.decode(tokens['input_ids'])}")
print(f"Tokens: {tokenizer.convert_ids_to_tokens(tokens['input_ids'])}")
# ['[CLS]', 'Laptop', 'gaming', 'con', 'RT', '##X', '3060', '[SEP]']

print("\n✅ AutoTokenizer cargado correctamente")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: Cargar AutoTokenizer desde BERT Multilingüe.*

1. Modelo multilingüe: soporta español, inglés, etc.
2. Probar tokenización
3. ['[CLS]', 'Laptop', 'gaming', 'con', 'RT', '##X', '3060', '[SEP]']

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 2: Tokenizar Descripciones con Padding y Truncation

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

descripciones = [
    "Laptop gaming con RTX 3060, 16GB RAM y SSD 512GB. Ideal para gaming y trabajo.",
    "Mouse inalámbrico ergonómico",
    "Teclado mecánico RGB retroiluminado con switches Cherry MX Brown y reposamuñecas",
]

# Tokenización con padding y truncation
encoded = tokenizer(
    descripciones,
    padding='max_length',
    truncation=True,
    max_length=16,
    return_tensors='pt'
)

print(f"Input IDs shape: {encoded['input_ids'].shape}")  # (3, 16)
print(f"Attention mask shape: {encoded['attention_mask'].shape}")
print(f"\nInput IDs:\n{encoded['input_ids']}")
print(f"\nAttention mask (1=real, 0=padding):\n{encoded['attention_mask']}")

# Decodificar cada una
for i, desc in enumerate(descripciones):
    decoded = tokenizer.decode(encoded['input_ids'][i])
    print(f"\n{i}: '{desc[:30]}...' -> '{decoded}'")

print("\n✅ Padding y truncation garantizan lotes uniformes")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: Tokenizar Descripciones con Padding y Truncation.*

1. Tokenización con padding y truncation
2. Decodificar cada una

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 3: Crear Dataset desde Pandas

```python
import pandas as pd
from datasets import Dataset

# DataFrame con catálogo de productos
df = pd.DataFrame({
    'descripcion': [
        "Laptop gaming con RTX 3060 16GB RAM",
        "Mouse inalámbrico ergonómico 6 botones",
        "Teclado mecánico RGB Cherry MX Brown",
        "Monitor 27 pulgadas 4K IPS 144Hz",
        "Silla ergonómica ajustable soporte lumbar",
        "Auriculares bluetooth cancelación ruido",
        "Webcam 1080p con micrófono integrado",
        "Hub USB C 7 puertos HDMI SD",
        "Disco SSD 1TB NVMe lectura 3500MB/s",
        "Impresora láser wifi dúplex automático",
    ],
    'categoria': [
        "laptop", "periférico", "periférico", "monitor", "mueble",
        "audio", "periférico", "accesorio", "almacenamiento", "impresora"
    ]
})

# Crear Dataset Hugging Face
dataset = Dataset.from_pandas(df)
print(f"Dataset: {dataset}")
print(f"\nPrimer ejemplo: {dataset[0]}")
print(f"\nFeatures: {dataset.features}")
print(f"Tamaño: {len(dataset)} ejemplos")

print("\n✅ Dataset.from_pandas crea un dataset compatible con Trainer")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: Crear Dataset desde Pandas.*

1. DataFrame con catálogo de productos
2. Crear Dataset Hugging Face

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 4: Mapear Tokenización con dataset.map()

```python
from transformers import AutoTokenizer
from datasets import Dataset
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

df = pd.DataFrame({
    'descripcion': [
        "Laptop gaming con RTX 3060",
        "Mouse inalámbrico ergonómico",
        "Teclado mecánico RGB",
    ],
    'label': [0, 1, 1]
})

dataset = Dataset.from_pandas(df)

def tokenize_function(examples):
    return tokenizer(
        examples['descripcion'],
        padding='max_length',
        truncation=True,
        max_length=32
    )

# Mapear: aplica tokenización a todo el dataset
dataset_tokenizado = dataset.map(tokenize_function, batched=True)

print(f"Dataset original columns: {dataset.column_names}")
print(f"Dataset tokenizado columns: {dataset_tokenizado.column_names}")
print(f"\nEjemplo tokenizado:")
print(f"  input_ids (primeros 10): {dataset_tokenizado[0]['input_ids'][:10]}")
print(f"  attention_mask (primeros 10): {dataset_tokenizado[0]['attention_mask'][:10]}")
print(f"  label: {dataset_tokenizado[0]['label']}")

# Eliminar columnas de texto después de tokenizar
dataset_tokenizado = dataset_tokenizado.remove_columns(['descripcion'])
print(f"Columnas finales: {dataset_tokenizado.column_names}")

print("\n✅ dataset.map() tokeniza todo el dataset eficientemente")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: Mapear Tokenización con dataset.map().*

1. Mapear: aplica tokenización a todo el dataset
2. Eliminar columnas de texto después de tokenizar

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 5: AutoModelForSequenceClassification con 5 Categorías

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# 5 categorías de productos
categorias = ["laptop", "periférico", "monitor", "mueble", "almacenamiento"]
id2label = {i: cat for i, cat in enumerate(categorias)}
label2id = {cat: i for i, cat in enumerate(categorias)}

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased",
    num_labels=len(categorias),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

print(f"Modelo: {model.config._name_or_path}")
print(f"Número de labels: {model.config.num_labels}")
print(f"id2label: {model.config.id2label}")
print(f"label2id: {model.config.label2id}")
print(f"Parámetros: {sum(p.numel() for p in model.parameters()):,}")

# Probar forward pass
textos = ["Laptop gaming con RTX", "Monitor 4K 27 pulgadas"]
inputs = tokenizer(textos, padding=True, truncation=True, return_tensors='pt')
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    preds = logits.argmax(-1)
    for texto, pred in zip(textos, preds):
        print(f"  '{texto}' -> {id2label[pred.item()]}")

print("\n✅ Modelo de clasificación preparado con 5 categorías")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: AutoModelForSequenceClassification con 5 Categorías.*

1. 5 categorías de productos
2. Probar forward pass

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 6: TrainingArguments con evaluation_strategy='epoch'

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./modelo_bert_productos",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    learning_rate=2e-5,
    adam_epsilon=1e-8,
    max_grad_norm=1.0,
    lr_scheduler_type="linear",
    warmup_ratio=0.1,
    save_total_limit=2,
    fp16=__import__('torch').cuda.is_available(),
    report_to="none",
)

print("TrainingArguments listos:")
print(f"  Épocas: {training_args.num_train_epochs}")
print(f"  Batch (train): {training_args.per_device_train_batch_size}")
print(f"  Batch (eval): {training_args.per_device_eval_batch_size}")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Evaluación: {training_args.evaluation_strategy}")
print(f"  Mejor métrica: {training_args.metric_for_best_model}")
print(f"  FP16: {training_args.fp16}")

print("\n✅ TrainingArguments configurado para fine-tuning óptimo")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: TrainingArguments con evaluation_strategy='epoch'.*

1. `from transformers import TrainingArguments` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 7: Trainer con compute_metrics Personalizado

```python
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Datos de ejemplo
df = pd.DataFrame({
    'descripcion': [
        "Laptop gaming con RTX 3060",
        "Mouse inalámbrico ergonómico",
        "Teclado mecánico RGB",
        "Monitor 4K 27 pulgadas",
        "Silla ergonómica ajustable",
        "Disco SSD 1TB NVMe",
    ] * 5,
    'label': [0, 1, 1, 2, 3, 4] * 5
})

dataset = Dataset.from_pandas(df).train_test_split(test_size=0.2)
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

def tokenize(batch):
    return tokenizer(batch['descripcion'], padding='max_length', truncation=True, max_length=32)

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.remove_columns(['descripcion'])
dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased",
    num_labels=5,
    ignore_mismatched_sizes=True
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(-1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1_weighted': f1_score(labels, predictions, average='weighted'),
        'precision_weighted': precision_score(labels, predictions, average='weighted'),
        'recall_weighted': recall_score(labels, predictions, average='weighted'),
    }

training_args = TrainingArguments(
    output_dir="./test_trainer",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="no",
    logging_steps=5,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Entrenar
trainer.train()

# Evaluar
metrics = trainer.evaluate()
print(f"\nMétricas de evaluación:")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")

print("\n✅ Trainer con compute_metrics evaluando accuracy, f1, precision, recall")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: Trainer con compute_metrics Personalizado.*

1. Datos de ejemplo
2. Entrenar
3. Evaluar

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 8: Entrenar BERT para Clasificar Categorías

```python
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# ===== 1. DATOS =====
np.random.seed(42)
categorias = ["electrónica", "periférico", "monitor", "mueble", "almacenamiento",
              "audio", "accesorio", "impresora", "red", "cable"]
label2id = {c: i for i, c in enumerate(categorias)}
id2label = {i: c for i, c in enumerate(categorias)}

desc_por_cat = {
    "electrónica": ["laptop {} con procesador i7", "tablet {} 10 pulgadas", "pc {} escritorio i5"],
    "periférico": ["mouse {} inalámbrico", "teclado {} mecánico RGB", "webcam {} 1080p"],
    "monitor": ["monitor {} 27 4K IPS", "pantalla {} curva 32", "monitor {} gaming 144Hz"],
    "mueble": ["silla {} ergonómica oficina", "escritorio {} ajustable", "mesa {} computadora"],
    "almacenamiento": ["disco {} SSD 1TB NVMe", "HDD {} externo 4TB", "memoria {} USB 128GB"],
    "audio": ["audífonos {} bluetooth", "bocina {} portátil", "micrófono {} condensador"],
    "accesorio": ["hub {} USB C 7 puertos", "soporte {} monitor brazo", "base {} laptop aluminio"],
    "impresora": ["impresora {} láser wifi", "multifuncional {} tinta", "impresora {} 3D"],
    "red": ["router {} wifi 6", "switch {} gigabit 8 puertos", "adaptador {} USB wifi"],
    "cable": ["cable {} HDMI 2.1 3m", "cable {} USB C carga", "adaptador {} VGA HDMI"],
}

rows = []
for cat, templates in desc_por_cat.items():
    for t in templates:
        for marca in ["Genérico", "TechPro", "HomeBrand", "ProLine", "BasicTech"]:
            rows.append({"text": t.format(marca), "label": label2id[cat]})

df = pd.DataFrame(rows)
dataset = Dataset.from_pandas(df).train_test_split(test_size=0.2, seed=42)

# ===== 2. TOKENIZAR =====
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=32)

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.remove_columns(['text'])
dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# ===== 3. MODELO =====
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased",
    num_labels=len(categorias),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

# ===== 4. TRAINING =====
args = TrainingArguments(
    output_dir="./bert_categorias",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    learning_rate=2e-5,
    logging_steps=10,
    report_to="none",
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return {'accuracy': accuracy_score(labels, logits.argmax(-1))}

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
print("\n✅ BERT fine-tuneado para clasificar productos en 10 categorías")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: Entrenar BERT para Clasificar Categorías.*

1. ===== 1. DATOS =====
2. ===== 2. TOKENIZAR =====
3. ===== 3. MODELO =====
4. ===== 4. TRAINING =====

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 9: Evaluar en Test Set

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# Cargar modelo fine-tuneado
model = AutoModelForSequenceClassification.from_pretrained("./bert_categorias")
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

# Datos de test
test_data = [
    {"text": "laptop gaming con RTX 4070", "label": 0},
    {"text": "teclado inalámbrico slim", "label": 1},
    {"text": "monitor 32 pulgadas curvo", "label": 2},
    {"text": "silla oficina mesh negra", "label": 3},
    {"text": "SSD externo 2TB portátil", "label": 4},
]
df_test = pd.DataFrame(test_data)
test_dataset = Dataset.from_pandas(df_test)

def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=32)
test_dataset = test_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.remove_columns(['text'])
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# Evaluar
args = TrainingArguments(output_dir="./test_eval", per_device_eval_batch_size=8, report_to="none", logging_steps=10)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return {'accuracy': accuracy_score(labels, logits.argmax(-1))}

trainer = Trainer(model=model, args=args, eval_dataset=test_dataset, tokenizer=tokenizer, compute_metrics=compute_metrics)
metrics = trainer.evaluate()
print(f"Accuracy en test: {metrics['eval_accuracy']:.2%}")
print("✅ Evaluación en test set completada")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: Evaluar en Test Set.*

1. Cargar modelo fine-tuneado
2. Datos de test
3. Evaluar

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 10: save_pretrained — Guardar Modelo Fine-Tuneado

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

# Simular modelo entrenado
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased", num_labels=5, ignore_mismatched_sizes=True
)
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

# Guardar
output_dir = "./modelo_guardado"
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Modelo guardado en: {output_dir}")
print(f"Archivos:")
for f in os.listdir(output_dir):
    size = os.path.getsize(os.path.join(output_dir, f))
    print(f"  {f}: {size / 1024:.1f} KB")

print("\n✅ save_pretrained guarda modelo + config + tokenizer")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: save_pretrained — Guardar Modelo Fine-Tuneado.*

1. Simular modelo entrenado
2. Guardar

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 11: Cargar Modelo Guardado y Predecir

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Cargar modelo guardado
model = AutoModelForSequenceClassification.from_pretrained("./modelo_guardado")
tokenizer = AutoTokenizer.from_pretrained("./modelo_guardado")
model.eval()

# Predecir
nuevas_descripciones = [
    "Laptop ultradelgada con i5 y 8GB RAM",
    "Monitor curvo 34 pulgadas ultrawide",
    "Ratón vertical ergonómico inalámbrico",
]

inputs = tokenizer(nuevas_descripciones, padding=True, truncation=True, return_tensors='pt')
with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(-1)

id2label = model.config.id2label
for desc, pred in zip(nuevas_descripciones, predictions):
    print(f"  '{desc}' -> Categoría {pred.item()} ({id2label[str(pred.item())]})")

print("\n✅ Modelo cargado y prediciendo nuevas descripciones")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: Cargar Modelo Guardado y Predecir.*

1. Cargar modelo guardado
2. Predecir

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 12: Classification Report Detallado

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import classification_report

model = AutoModelForSequenceClassification.from_pretrained("./bert_categorias")
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
model.eval()

# Datos de prueba
test_texts = [
    "laptop gaming", "mouse inalámbrico", "monitor 4K", "silla oficina",
    "disco SSD", "audífonos bluetooth", "hub USB", "impresora wifi",
    "router gigabit", "cable HDMI",
]
true_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

inputs = tokenizer(test_texts, padding=True, truncation=True, return_tensors='pt')
with torch.no_grad():
    logits = model(**inputs).logits
    predictions = logits.argmax(-1).numpy()

print("Classification Report:")
print(classification_report(
    true_labels, predictions,
    target_names=list(model.config.id2label.values()),
    zero_division=0
))
print("✅ Classification report detallado por categoría")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: Classification Report Detallado.*

1. Datos de prueba

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 13: Matriz de Confusión

```python
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

model = AutoModelForSequenceClassification.from_pretrained("./bert_categorias")
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
model.eval()

# Generar predicciones
test_texts = [
    "laptop gaming", "mouse", "teclado", "monitor", "silla",
    "disco duro", "audífonos", "hub", "impresora", "router",
] * 5
true_labels = [0, 1, 1, 2, 3, 4, 5, 6, 7, 8] * 5

inputs = tokenizer(test_texts, padding=True, truncation=True, return_tensors='pt')
with torch.no_grad():
    predictions = model(**inputs).logits.argmax(-1).numpy()

# Matriz de confusión
cm = confusion_matrix(true_labels, predictions)
categorias = list(model.config.id2label.values())

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categorias)
fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
plt.title("Matriz de Confusión - Clasificación de Productos")
plt.tight_layout()
plt.savefig('/tmp/matriz_confusion.png', dpi=100)
print("✅ Matriz de confusión generada en /tmp/matriz_confusion.png")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: Matriz de Confusión.*

1. Generar predicciones
2. Matriz de confusión

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 14: Probar con Nueva Descripción

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model = AutoModelForSequenceClassification.from_pretrained("./bert_categorias")
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
model.eval()

def predecir_producto(descripcion):
    inputs = tokenizer(descripcion, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        pred_id = logits.argmax(-1).item()
        probs = torch.softmax(logits, -1).squeeze()
    return pred_id, model.config.id2label[str(pred_id)], probs[pred_id].item()

nuevos = [
    "Teclado inalámbrico slim con batería recargable para oficina",
    "Monitor gaming curvo 27 pulgadas 240Hz 1ms",
    "Impresora multifuncional a color con WiFi y duplex automático",
]

for desc in nuevos:
    pred_id, categoria, confianza = predecir_producto(desc)
    print(f"'{desc[:50]}...'")
    print(f"  -> {categoria} (confianza: {confianza:.2%})\n")

print("✅ Predicción en tiempo real para nuevas descripciones")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: Probar con Nueva Descripción.*

1. `from transformers import AutoModelForSequenceClassification, AutoTokenizer` — Importa las librerías necesarias para el análisis.
2. `import torch` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 15: Comparar LR 2e-5 vs 5e-5

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# Configuración común
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
df = pd.DataFrame({
    'text': [f"producto {i} descripción" for i in range(100)],
    'label': [i % 5 for i in range(100)]
})
dataset = Dataset.from_pandas(df).train_test_split(test_size=0.2)

def tok(b):
    return tokenizer(b['text'], padding='max_length', truncation=True, max_length=16)
dataset = dataset.map(tok, batched=True).remove_columns(['text'])
dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

def compute_metrics(ep):
    return {'accuracy': accuracy_score(ep[1], ep[0].argmax(-1))}

resultados = {}
for lr in [2e-5, 5e-5]:
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-multilingual-cased", num_labels=5, ignore_mismatched_sizes=True)
    args = TrainingArguments(
        output_dir=f"./test_lr_{lr}", num_train_epochs=2,
        per_device_train_batch_size=8, evaluation_strategy="epoch",
        learning_rate=lr, save_strategy="no", logging_steps=50, report_to="none",
    )
    trainer = Trainer(model=model, args=args, tokenizer=tokenizer,
                      train_dataset=dataset['train'], eval_dataset=dataset['test'],
                      compute_metrics=compute_metrics)
    trainer.train()
    metrics = trainer.evaluate()
    resultados[lr] = metrics['eval_accuracy']

for lr, acc in resultados.items():
    print(f"  LR {lr}: accuracy={acc:.2%}")

mejor = max(resultados, key=resultados.get)
print(f"\n✅ Mejor learning rate: {mejor}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: Comparar LR 2e-5 vs 5e-5.*

1. Configuración común

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 16: Early Stopping en Trainer

```python
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    Trainer, TrainingArguments, EarlyStoppingCallback
)
from datasets import Dataset
import pandas as pd
from sklearn.metrics import accuracy_score

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
df = pd.DataFrame({
    'text': [f"producto {i}" for i in range(200)],
    'label': [i % 5 for i in range(200)]
})
dataset = Dataset.from_pandas(df).train_test_split(test_size=0.2)

def tok(b):
    return tokenizer(b['text'], padding='max_length', truncation=True, max_length=16)
dataset = dataset.map(tok, batched=True).remove_columns(['text'])
dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased", num_labels=5, ignore_mismatched_sizes=True)

args = TrainingArguments(
    output_dir="./test_early_stopping",
    num_train_epochs=20,
    per_device_train_batch_size=8,
    evaluation_strategy="steps",
    eval_steps=10,
    save_strategy="steps",
    save_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    save_total_limit=2,
    report_to="none",
)

def compute_metrics(ep):
    return {'accuracy': accuracy_score(ep[1], ep[0].argmax(-1))}

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()
print(f"\n✅ Early stopping: entrenamiento detenido cuando accuracy dejó de mejorar")
print(f"Épocas reales: {trainer.state.epoch:.1f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: Early Stopping en Trainer.*

1. `from transformers import (` — Importa las librerías necesarias para el análisis.
2. `from datasets import Dataset` — Importa las librerías necesarias para el análisis.
3. `import pandas as pd` — Importa las librerías necesarias para el análisis.
4. `from sklearn.metrics import accuracy_score` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 17: Class Weight para Clases Desbalanceadas

```python
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    Trainer, TrainingArguments
)
from datasets import Dataset
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

# Dataset desbalanceado: 80% categoría 0, 10% categoría 1, 10% categoría 2
np.random.seed(42)
texts = []
labels = []
for i in range(500):
    texts.append(f"producto {i}")
    if i < 400:
        labels.append(0)
    elif i < 450:
        labels.append(1)
    else:
        labels.append(2)

df = pd.DataFrame({'text': texts, 'label': labels})
dataset = Dataset.from_pandas(df).train_test_split(test_size=0.2)
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

def tok(b):
    return tokenizer(b['text'], padding='max_length', truncation=True, max_length=16)
dataset = dataset.map(tok, batched=True).remove_columns(['text'])
dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# Calcular class weights
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float32)
print(f"Class weights: {class_weights}")

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased", num_labels=3, ignore_mismatched_sizes=True)

# Aplicar class weight en el loss
model.config.problem_type = "single_label_classification"

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(model.device))
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

args = TrainingArguments(
    output_dir="./test_weighted",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="no",
    report_to="none",
)

def compute_metrics(ep):
    preds = ep[0].argmax(-1)
    return {'accuracy': accuracy_score(ep[1], preds), 'f1': f1_score(ep[1], preds, average='weighted')}

trainer = WeightedTrainer(
    model=model,
    args=args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    compute_metrics=compute_metrics,
)
trainer.train()
metrics = trainer.evaluate()
print(f"\nCon class weights: accuracy={metrics['eval_accuracy']:.2%}, f1={metrics['eval_f1']:.2%}")
print("✅ Class weights mejoran rendimiento en clases desbalanceadas")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: Class Weight para Clases Desbalanceadas.*

1. Dataset desbalanceado: 80% categoría 0, 10% categoría 1, 10% categoría 2
2. Calcular class weights
3. Aplicar class weight en el loss

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 18: Integrador — BERT para Clasificación en 10 Categorías

```python
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import torch

# ===== 1. GENERAR DATOS =====
np.random.seed(42)
categorias = [
    "electrónica", "periférico", "monitor", "mueble", "almacenamiento",
    "audio", "accesorio", "impresora", "red", "cable"
]
label2id = {c: i for i, c in enumerate(categorias)}
id2label = {i: c for c, i in label2id.items()}

plantillas = {
    "electrónica": ["laptop {} i7 16GB", "tablet {} 10pulg", "pc {} escritorio"],
    "periférico": ["mouse {} ergo", "teclado {} rgb", "webcam {} 1080p"],
    "monitor": ["monitor {} 27 4K", "pantalla {} curva", "monitor {} gaming"],
    "mueble": ["silla {} oficina", "escritorio {} ajustable", "mesa {} ergo"],
    "almacenamiento": ["SSD {} 1TB", "HDD {} externo", "USB {} 128GB"],
    "audio": ["audífonos {} bt", "bocina {} portátil", "micrófono {} usb"],
    "accesorio": ["hub {} USB C", "soporte {} monitor", "base {} laptop"],
    "impresora": ["impresora {} laser", "multifuncional {} tinta", "impresora {} 3d"],
    "red": ["router {} wifi6", "switch {} gigabit", "adaptador {} wifi"],
    "cable": ["cable {} HDMI 2.1", "cable {} USB C", "adaptador {} VGA"],
}

filas = []
for cat, temps in plantillas.items():
    for t in temps:
        for marca in ["TechPro", "HomeBrand", "ProLine", "BasicTech", "Genérico"]:
            filas.append({"texto": t.format(marca), "categoria": label2id[cat]})

df = pd.DataFrame(filas)
dataset = Dataset.from_pandas(df).train_test_split(test_size=0.2, seed=42)

# ===== 2. TOKENIZAR =====
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

def tokenizar(batch):
    return tokenizer(batch['texto'], padding='max_length', truncation=True, max_length=32)

dataset = dataset.map(tokenizar, batched=True)
dataset = dataset.remove_columns(['texto'])
dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'categoria'])

dataset['train'] = dataset['train'].rename_column('categoria', 'labels')
dataset['test'] = dataset['test'].rename_column('categoria', 'labels')

# ===== 3. MODELO =====
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased",
    num_labels=len(categorias),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

# ===== 4. ENTRENAR =====
args = TrainingArguments(
    output_dir="./bert_integrador",
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    learning_rate=2e-5,
    logging_steps=10,
    report_to="none",
    save_total_limit=2,
)

def compute_metrics(ep):
    preds = ep[0].argmax(-1)
    return {
        'accuracy': (ep[1] == preds).mean(),
        'f1': __import__('sklearn').metrics.f1_score(ep[1], preds, average='weighted'),
    }

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# ===== 5. EVALUAR =====
metrics = trainer.evaluate()
print(f"\n=== RESULTADOS FINALES ===")
print(f"Accuracy: {metrics['eval_accuracy']:.2%}")
print(f"F1 Weighted: {metrics['eval_f1']:.2%}")

# ===== 6. PREDECIR NUEVOS PRODUCTOS =====
model.eval()
nuevos = [
    "Laptop gaming con RTX 4070 y 32GB RAM",
    "Teclado mecánico compacto 60% con switches azules",
    "Monitor IPS 4K 32 pulgadas para diseño gráfico",
    "Impresora láser color dúplex automático wifi",
    "Cable USB C a HDMI 4K 2 metros trenzado",
]

inputs = tokenizer(nuevos, padding=True, truncation=True, return_tensors='pt')
with torch.no_grad():
    preds = model(**inputs).logits.argmax(-1)

print("\n=== PREDICCIONES ===")
for texto, pred in zip(nuevos, preds):
    print(f"  '{texto}' -> {id2label[pred.item()]}")

# ===== 7. GUARDAR =====
model.save_pretrained("./bert_integrador_final")
tokenizer.save_pretrained("./bert_integrador_final")

print("\n✅ INTEGRADOR COMPLETO: BERT fine-tuneado para clasificar productos en 10 categorías")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador — BERT para Clasificación en 10 Categorías.*

1. ===== 1. GENERAR DATOS =====
2. ===== 2. TOKENIZAR =====
3. ===== 3. MODELO =====
4. ===== 4. ENTRENAR =====
5. ===== 5. EVALUAR =====
6. ===== 6. PREDECIR NUEVOS PRODUCTOS =====
7. ===== 7. GUARDAR =====

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

## 3. Ejercicios Propuestos

1. **Fine-tuning multiclase**: Toma 50 descripciones de productos con 5 categorías y fine-tunea BERT. Reporta accuracy y F1 en test. Varía la semilla y repite 3 veces.

2. **Comparación de modelos**: Fine-tunea 3 modelos (bert-base-multilingual, distilbert-base-multilingual, roberta-base) en el mismo dataset. Compara accuracy, velocidad de entrenamiento y tamaño del modelo.

3. **Data augmentation**: Genera 100 descripciones adicionales usando sinónimos (palabras clave: "laptop", "gaming", "oficina"). Fine-tunea BERT con y sin aumento. ¿Mejora el rendimiento?

4. **Fine-tuning con validación cruzada**: Implementa validación cruzada de 5 folds sobre BERT fine-tuneado. Reporta media y desviación estándar de accuracy.

5. **Curva de aprendizaje**: Entrena BERT con 10, 50, 100, 200, 500 ejemplos. Grafica accuracy vs cantidad de datos. ¿Cuántos datos se necesitan para saturar?

6. **Clases desbalanceadas extremo**: Crea un dataset donde una categoría tenga 90% de los ejemplos y las otras 9 compartan 10%. Aplica class_weight y compara con la versión sin weight.

7. **Exportar a ONNX**: Guarda el modelo fine-tuneado en formato ONNX. Cárgalo y haz una inferencia. Mide la diferencia de velocidad vs PyTorch puro.

8. **Proyecto final**: Construye un pipeline completo que: (a) cargue un CSV de productos, (b) fine-tunee BERT, (c) evalúe con classification report + matriz de confusión, (d) guarde el modelo, (e) exponga una función `predecir(descripcion) -> categoría`.

---

## 4. Resumen

| Componente | Rol | Parámetros Clave |
|---|---|---|
| `AutoTokenizer` | Convertir texto a tokens | `padding`, `truncation`, `max_length` |
| `AutoModelForSequenceClassification` | Modelo con cabeza de clasificación | `num_labels`, `id2label`, `label2id` |
| `TrainingArguments` | Configuración de entrenamiento | `learning_rate`, `per_device_batch_size`, `evaluation_strategy` |
| `Trainer` | Bucle de entrenamiento | `model`, `args`, `train_dataset`, `compute_metrics` |
| `Dataset.from_pandas` | Convertir pandas a HF dataset | `map()`, `remove_columns()` |
| `compute_metrics` | Evaluación personalizada | `accuracy_score`, `f1_score` |
| `save_pretrained` | Persistir modelo | Directorio de salida |

**Conclusión**: Fine-tuning de BERT con Hugging Face Trainer es el método estándar para clasificación de texto en ventas. Con ~100-500 ejemplos por categoría y 3-5 épocas, se obtienen modelos con >90% de accuracy para categorización de productos, análisis de sentimiento y detección de quejas.
