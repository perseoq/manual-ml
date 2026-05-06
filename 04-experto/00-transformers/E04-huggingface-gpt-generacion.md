# E04: Hugging Face — GPT y Generación de Texto

## Objetivo
Dominar la generación de texto con modelos autoregresivos (GPT-2, GPT, Llama) aplicada a descripciones de productos, reseñas falsas para testing y contenido de catálogos B2B.

---

## 1. Fundamentos Teóricos

### 1.1 Modelos Autoregresivos (Decoder-Only)

Los modelos como GPT generan texto token a token, condicionados en los tokens anteriores:

```
P(t_n | t_1, t_2, ..., t_{n-1})
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*1.1 Modelos Autoregresivos (Decoder-Only).*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



**Arquitectura**: Decoder-only con masked self-attention (causal). Cada token solo ve tokens a su izquierda.

### 1.2 Estrategias de Generación

#### Greedy Decoding
Siempre escoge el token con mayor probabilidad. Rápido pero produce texto repetitivo.

#### Beam Search
Mantiene `num_beams` secuencias candidatas. Mejor calidad, más lento.

#### Temperature Sampling
Escala los logits antes de softmax:
- **T → 0**: Determinista (greedy).
- **T = 1.0**: Distribución original.
- **T → ∞**: Uniforme (aleatorio).

#### Top-K Sampling
Filtra los K tokens con mayor probabilidad y re-distribuye.

#### Top-P Sampling (Nucleus)
Filtra el conjunto mínimo de tokens cuya probabilidad acumulada > p.

### 1.3 Parámetros Clave en `model.generate()`

| Parámetro | Efecto | Valor típico |
|---|---|---|
| `max_new_tokens` | Tokens a generar | 50-200 |
| `do_sample` | Activar sampling | True |
| `temperature` | Creatividad | 0.1-1.5 |
| `top_k` | Filtrar top K | 50 |
| `top_p` | Nucleus sampling | 0.9-0.95 |
| `num_beams` | Beam search | 1-10 |
| `repetition_penalty` | Penalizar repeticiones | 1.0-1.2 |
| `num_return_sequences` | Variantes a generar | 1-5 |
| `no_repeat_ngram_size` | Evitar n-gramas repetidos | 3 |

### 1.4 Padding Side

Para modelos causales, el padding debe ir a la **izquierda**:
```python
tokenizer.padding_side = 'left'
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*1.4 Padding Side.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---


Esto asegura que el último token real (no padding) sea el que se use para generar.

---

## 2. Ejemplos Prácticos

### Ejemplo 1: Cargar GPT-2 y Tokenizer

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Configurar padding
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

print(f"Modelo: {model_name}")
print(f"Vocabulario: {tokenizer.vocab_size} tokens")
print(f"Max length: {tokenizer.model_max_length}")
print(f"Parámetros: {sum(p.numel() for p in model.parameters()):,}")

# Probar forward pass
prompt = "Laptop gaming con"
inputs = tokenizer(prompt, return_tensors='pt')
outputs = model(**inputs)
print(f"\nPrompt: '{prompt}'")
print(f"Logits shape: {outputs.logits.shape}")
next_token_logits = outputs.logits[0, -1, :]
next_token = next_token_logits.argmax(-1).item()
print(f"Siguiente token (greedy): '{tokenizer.decode(next_token)}'")
print("✅ GPT-2 y tokenizer cargados correctamente")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: Cargar GPT-2 y Tokenizer.*

1. Configurar padding
2. Probar forward pass

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 2: Generar Descripción de Producto

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")

prompt = "The new gaming laptop features"
inputs = tokenizer(prompt, return_tensors='pt')

outputs = model.generate(
    **inputs,
    max_new_tokens=80,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    num_return_sequences=1,
)

descripcion = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Prompt: {prompt}")
print(f"Generado: {descripcion}")
print("\n✅ GPT-2 genera descripciones coherentes de productos")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: Generar Descripción de Producto.*

1. `from transformers import GPT2LMHeadModel, GPT2Tokenizer` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 3: Temperature Sampling — 0.1 vs 1.0 vs 2.0

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")

prompt = "The best feature of this monitor is"
inputs = tokenizer(prompt, return_tensors='pt')

temperaturas = [0.1, 1.0, 2.0]
for temp in temperaturas:
    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=True,
        temperature=temp,
        top_k=0,
        top_p=1.0,
    )
    texto = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nTemperature={temp}:")
    print(f"  {texto}")

print("\n✅ T baja = determinista; T alta = más diverso/riesgoso")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: Temperature Sampling — 0.1 vs 1.0 vs 2.0.*

1. `from transformers import GPT2LMHeadModel, GPT2Tokenizer` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 4: Top-K Sampling — k=10 vs 50 vs 100

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")

prompt = "This wireless mouse is perfect for"
inputs = tokenizer(prompt, return_tensors='pt')

for k in [10, 50, 100]:
    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=True,
        temperature=1.0,
        top_k=k,
    )
    texto = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"top_k={k}: {texto}")

print("\n✅ top_k pequeño = más enfocado; grande = más diverso")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: Top-K Sampling — k=10 vs 50 vs 100.*

1. `from transformers import GPT2LMHeadModel, GPT2Tokenizer` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 5: Top-P Sampling (Nucleus) — 0.9 vs 0.95

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")

prompt = "The keyboard has RGB lighting and"
inputs = tokenizer(prompt, return_tensors='pt')

for p in [0.9, 0.95]:
    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=True,
        top_p=p,
        top_k=0,
    )
    texto = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"top_p={p}: {texto}")

print("\n✅ top_p controla el núcleo de probabilidad acumulada")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: Top-P Sampling (Nucleus) — 0.9 vs 0.95.*

1. `from transformers import GPT2LMHeadModel, GPT2Tokenizer` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 6: Beam Search — num_beams=5

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")

prompt = "The 27-inch 4K monitor delivers"
inputs = tokenizer(prompt, return_tensors='pt')

# Greedy (beam=1)
outputs_greedy = model.generate(
    **inputs, max_new_tokens=40, num_beams=1, do_sample=False
)
texto_greedy = tokenizer.decode(outputs_greedy[0], skip_special_tokens=True)
print(f"Greedy (beam=1): {texto_greedy}")

# Beam search (beam=5)
outputs_beam = model.generate(
    **inputs, max_new_tokens=40, num_beams=5, early_stopping=True
)
texto_beam = tokenizer.decode(outputs_beam[0], skip_special_tokens=True)
print(f"Beam search (5): {texto_beam}")

print("\n✅ Beam search encuentra secuencias más probables")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: Beam Search — num_beams=5.*

1. Greedy (beam=1)
2. Beam search (beam=5)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 7: Repetition Penalty — 1.0 vs 1.2

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")

prompt = "Great product. Great product. Great"
inputs = tokenizer(prompt, return_tensors='pt')

for penalty in [1.0, 1.2]:
    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=True,
        repetition_penalty=penalty,
    )
    texto = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"repetition_penalty={penalty}: {texto}")

print("\n✅ repetition_penalty > 1.0 reduce bucles de repetición")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: Repetition Penalty — 1.0 vs 1.2.*

1. `from transformers import GPT2LMHeadModel, GPT2Tokenizer` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 8: Múltiples Secuencias — num_return_sequences=3

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")

prompt = "The new smartphone has"
inputs = tokenizer(prompt, return_tensors='pt')

outputs = model.generate(
    **inputs,
    max_new_tokens=40,
    do_sample=True,
    temperature=0.9,
    top_k=50,
    num_return_sequences=3,
)

print(f"Prompt: '{prompt}'")
for i, output in enumerate(outputs):
    texto = tokenizer.decode(output, skip_special_tokens=True)
    print(f"\nVariante {i+1}: {texto}")

print("\n✅ Múltiples secuencias para explorar alternativas de descripción")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: Múltiples Secuencias — num_return_sequences=3.*

1. `from transformers import GPT2LMHeadModel, GPT2Tokenizer` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 9: Controlar Longitud — max_new_tokens=50

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")

prompt = "This ergonomic office chair features"
inputs = tokenizer(prompt, return_tensors='pt')

for max_tokens in [20, 50, 100]:
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.8,
    )
    texto = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"max_new_tokens={max_tokens}: '{texto}'\n")

print("✅ max_new_tokens controla la extensión de la generación")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: Controlar Longitud — max_new_tokens=50.*

1. `from transformers import GPT2LMHeadModel, GPT2Tokenizer` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 10: Fine-Tuning GPT-2 con Trainer

```python
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
from datasets import Dataset
import pandas as pd

# Datos de ejemplo: descripciones de productos
descripciones = [
    "Laptop gaming con procesador i7 y RTX 3060",
    "Mouse inalámbrico ergonómico con 6 botones programables",
    "Teclado mecánico RGB con switches Cherry MX Brown",
    "Monitor 4K 27 pulgadas IPS 144Hz para gaming",
    "Silla ergonómica ajustable con soporte lumbar incorporado",
    "Auriculares bluetooth con cancelación de ruido activa",
    "Webcam 1080p con micrófono estéreo integrado",
    "Hub USB C con 7 puertos incluyendo HDMI 4K",
    "Disco SSD NVMe 1TB con velocidades de lectura 3500MB/s",
    "Cable HDMI 2.1 de 3 metros con soporte 48Gbps",
] * 5  # 50 ejemplos

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    outputs = tokenizer(
        examples['text'],
        truncation=True,
        max_length=64,
        padding='max_length',
        return_tensors=None,
    )
    outputs['labels'] = outputs['input_ids'].copy()
    return outputs

dataset = Dataset.from_pandas(pd.DataFrame({'text': descripciones}))
dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

model = GPT2LMHeadModel.from_pretrained("gpt2")

training_args = TrainingArguments(
    output_dir="./gpt2_finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_strategy="epoch",
    logging_steps=10,
    report_to="none",
    learning_rate=5e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
print("\n✅ GPT-2 fine-tuneado con descripciones de productos")

# Generar después de fine-tuning
prompt = "New product:"
inputs = tokenizer(prompt, return_tensors='pt')
outputs = model.generate(**inputs, max_new_tokens=30, do_sample=True)
print(f"Generación post fine-tuning: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: Fine-Tuning GPT-2 con Trainer.*

1. Datos de ejemplo: descripciones de productos
2. Generar después de fine-tuning

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 11: Few-Shot Prompting

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Few-shot: dar ejemplos en el prompt
prompt = """Product: laptop
Description: High-performance gaming laptop with Intel i7 and RTX 3060.

Product: mouse
Description: Wireless ergonomic mouse with 6 programmable buttons.

Product: keyboard
Description:"""

inputs = tokenizer(prompt, return_tensors='pt')
outputs = model.generate(
    **inputs,
    max_new_tokens=30,
    do_sample=True,
    temperature=0.7,
    top_k=50,
)
texto = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Few-shot generation:\n{texto}")

print("\n✅ Few-shot: el modelo imita el patrón de los ejemplos")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: Few-Shot Prompting.*

1. Few-shot: dar ejemplos en el prompt

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 12: pipeline("text-generation") para Interfaz Simple

```python
from transformers import pipeline

generador = pipeline("text-generation", model="gpt2")

prompts = [
    "The best laptop for programming is",
    "This wireless mouse has a battery life of",
    "The 4K monitor features",
]

for prompt in prompts:
    resultado = generador(
        prompt,
        max_new_tokens=30,
        do_sample=True,
        temperature=0.8,
        top_k=50,
    )
    print(f"Prompt: '{prompt}'")
    print(f"Generado: {resultado[0]['generated_text']}\n")

print("✅ pipeline() simplifica la generación de texto")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: pipeline("text-generation") para Interfaz Simple.*

1. `from transformers import pipeline` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 13: Condicionar Generación con Formato Específico

```python
from transformers import pipeline

generador = pipeline("text-generation", model="gpt2")

# Formato estructurado para descripciones
prompts = [
    "Producto: Teclado. Descripción:",
    "Producto: Monitor. Descripción:",
    "Producto: Silla. Descripción:",
]

for prompt in prompts:
    resultado = generador(
        prompt,
        max_new_tokens=40,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    texto = resultado[0]['generated_text']
    print(f"\n{texto}")

print("\n✅ El prompt estructurado condiciona el formato de salida")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: Condicionar Generación con Formato Específico.*

1. Formato estructurado para descripciones

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 14: Generar Reseñas Falsas para Testing

```python
from transformers import pipeline
import random

generador = pipeline("text-generation", model="gpt2")

# Diferentes estilos de reseña
estilos = [
    "I bought this product and",
    "This is the worst purchase I've ever made because",
    "After using this for a week, I can say that",
    "I highly recommend this product. It",
    "Disappointed with this item.",
]

reseñas_generadas = []
for estilo in estilos:
    resultado = generador(
        estilo,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        num_return_sequences=2,
    )
    for r in resultado:
        reseñas_generadas.append(r['generated_text'])

print(f"{len(reseñas_generadas)} reseñas falsas generadas:")
for i, res in enumerate(reseñas_generadas[:5], 1):
    print(f"\n{i}. {res}")

print("\n✅ Reseñas falsas útiles para testing de sistemas de sentiment analysis")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: Generar Reseñas Falsas para Testing.*

1. Diferentes estilos de reseña

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 15: Comparar GPT-2 vs DistilGPT-2

```python
from transformers import pipeline
import time

modelos = {
    "GPT-2": "gpt2",
    "DistilGPT-2": "distilgpt2",
}

prompt = "The best feature of this product is"

for nombre, modelo_id in modelos.items():
    generador = pipeline("text-generation", model=modelo_id)

    start = time.time()
    resultado = generador(prompt, max_new_tokens=40, do_sample=True)
    t = time.time() - start

    params = sum(p.numel() for p in generador.model.parameters())
    print(f"\n{nombre}:")
    print(f"  Parámetros: {params/1e6:.1f}M")
    print(f"  Tiempo: {t:.3f}s")
    print(f"  Output: {resultado[0]['generated_text']}")

print("\n✅ DistilGPT-2 es más rápido y pequeño, con calidad similar")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: Comparar GPT-2 vs DistilGPT-2.*

1. `from transformers import pipeline` — Importa las librerías necesarias para el análisis.
2. `import time` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 16: Salvar y Cargar Modelo Fine-Tuneado

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

# Guardar
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

output_dir = "./modelo_gpt2_ventas"
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Modelo guardado en {output_dir}")

# Cargar
model_cargado = GPT2LMHeadModel.from_pretrained(output_dir)
tokenizer_cargado = GPT2Tokenizer.from_pretrained(output_dir)
tokenizer_cargado.pad_token = tokenizer_cargado.eos_token

# Probar
prompt = "New laptop with"
inputs = tokenizer_cargado(prompt, return_tensors='pt')
outputs = model_cargado.generate(**inputs, max_new_tokens=30, do_sample=True)
print(f"Generado desde modelo cargado: {tokenizer_cargado.decode(outputs[0], skip_special_tokens=True)}")

print("\n✅ Modelo guardado y cargado correctamente")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: Salvar y Cargar Modelo Fine-Tuneado.*

1. Guardar
2. Cargar
3. Probar

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 17: Prevenir Contenido Sensible con no_repeat_ngram_size

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")

prompt = "Buy now! This is a limited time offer. Buy now"
inputs = tokenizer(prompt, return_tensors='pt')

# Sin control de repetición
outputs_sin = model.generate(
    **inputs, max_new_tokens=30, do_sample=False,
)
texto_sin = tokenizer.decode(outputs_sin[0], skip_special_tokens=True)
print(f"Sin control: {texto_sin}")

# Con no_repeat_ngram_size=3
outputs_con = model.generate(
    **inputs, max_new_tokens=30, do_sample=True,
    no_repeat_ngram_size=3, temperature=0.9,
)
texto_con = tokenizer.decode(outputs_con[0], skip_special_tokens=True)
print(f"Con no_repeat_ngram=3: {texto_con}")

print("\n✅ no_repeat_ngram_size evita bucles y contenido repetitivo")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: Prevenir Contenido Sensible con no_repeat_ngram_size.*

1. Sin control de repetición
2. Con no_repeat_ngram_size=3

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



### Ejemplo 18: Integrador — Generador Automático de Descripciones

```python
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
import time
import random

# ===== CONFIGURACIÓN =====
generador = pipeline("text-generation", model="gpt2")

# ===== PLANTILLAS DE PRODUCTOS =====
productos = [
    "gaming laptop",
    "wireless mouse",
    "mechanical keyboard",
    "4K monitor",
    "ergonomic chair",
    "bluetooth headphones",
    "USB-C hub",
    "NVMe SSD",
    "HDMI cable",
    "webcam",
]

def generar_descripcion(producto, estilo="profesional"):
    plantillas = {
        "profesional": [
            f"The {producto} is designed for professionals who demand",
            f"Introducing the new {producto}, featuring",
            f"The premium {producto} offers",
        ],
        "marketing": [
            f"Experience the ultimate {producto} that",
            f"Transform your workflow with the {producto}",
            f"Discover the {producto} that everyone is talking about",
        ],
        "técnico": [
            f"The {producto} specifications include",
            f"Technical details of the {producto}:",
        ],
    }

    prompt = random.choice(plantillas[estilo])
    resultado = generador(
        prompt,
        max_new_tokens=60,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.1,
    )
    return resultado[0]['generated_text']

# ===== GENERAR CATÁLOGO =====
print("=== CATÁLOGO AUTOMÁTICO DE PRODUCTOS ===\n")
for producto in productos:
    desc = generar_descripcion(producto, random.choice(["profesional", "marketing", "técnico"]))
    print(f"Producto: {producto}")
    print(f"Descripción: {desc}\n")
    print("-" * 60)

# ===== BENCHMARK =====
print("\n=== BENCHMARK DE VELOCIDAD ===")
start = time.time()
for _ in range(10):
    _ = generar_descripcion("laptop", "profesional")
t_total = time.time() - start
print(f"10 descripciones generadas en {t_total:.2f}s")
print(f"Promedio: {t_total/10:.2f}s por descripción")

# ===== GUARDAR EJEMPLOS =====
with open("/tmp/descripciones_generadas.txt", "w") as f:
    for producto in productos:
        desc = generar_descripcion(producto, "profesional")
        f.write(f"Product: {producto}\nDescription: {desc}\n\n")

print("\n✅ INTEGRADOR COMPLETO: Generador automático de descripciones de productos para catálogo")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador — Generador Automático de Descripciones.*

1. ===== CONFIGURACIÓN =====
2. ===== PLANTILLAS DE PRODUCTOS =====
3. ===== GENERAR CATÁLOGO =====
4. ===== BENCHMARK =====
5. ===== GUARDAR EJEMPLOS =====

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

## 3. Ejercicios Propuestos

1. **Exploración de temperature**: Genera 10 descripciones para "wireless mouse" con temperature=0.2, 0.5, 1.0, 1.5, 2.0. Cuenta palabras únicas en cada una. ¿Cuál temperatura da mayor diversidad sin perder coherencia?

2. **Beam search vs sampling**: Para el prompt "The best laptop for gaming", compara beam search (num_beams=5) vs sampling (temperature=0.8). Genera 5 ejemplos de cada. ¿Cuál prefieres? ¿Por qué?

3. **Fine-tuning con datos reales**: Busca 20 descripciones reales de productos en Amazon (copia los textos). Fine-tunea DistilGPT-2 con esos datos. Genera 5 nuevas descripciones y evalúa si suenan auténticas.

4. **Control de formato**: Crea un prompt que genere descripciones con formato fijo: "Nombre: X | Precio: Y | Características: Z". Genera 10 ejemplos.

5. **Generación condicional por categoría**: Crea un generador que reciba la categoría ("laptop", "mouse", "monitor") y genere una descripción apropiada. Usa un solo modelo con prompts diferenciados.

6. **Detección de reseñas falsas**: Genera 20 reseñas falsas con GPT-2. Luego entrena un clasificador BERT para distinguir reales vs falsas. Reporta accuracy del detector.

7. **Prompt engineering**: Diseña 5 prompts diferentes para generar descripción de "mechanical keyboard". Evalúa cuál produce los mejores resultados (más coherentes, más descriptivos).

8. **Proyecto final**: Crea una aplicación que: (a) reciba una lista de productos, (b) genere descripciones para cada uno, (c) guarde en formato CSV/JSON, (d) permita elegir estilo (profesional/marketing/técnico), (e) muestre las mejores 3 variantes por producto.

---

## 4. Resumen

| Estrategia | Parámetro | Efecto | Uso en ventas |
|---|---|---|---|
| Greedy | `do_sample=False` | Determinista, repetitivo | Línea base |
| Beam Search | `num_beams=5` | Mayor calidad | Descripciones oficiales |
| Temperature | `temperature=0.7` | Creatividad controlada | Marketing |
| Top-K | `top_k=50` | Filtra cola larga | Balance calidad/diversidad |
| Top-P | `top_p=0.95` | Nucleus adaptativo | Generación natural |
| Repetition Penalty | `repetition_penalty=1.1` | Evita bucles | Texto fluido |
| Fine-tuning | Trainer | Adaptar a dominio | Catálogos específicos |
| Few-shot | Prompt con ejemplos | Sin entrenar | Prototipado rápido |

**Conclusión**: Los modelos autoregresivos (GPT-2, GPT, Llama) permiten generar descripciones de productos, reseñas sintéticas y contenido de catálogo con calidad comercial. El fine-tuning con datos del dominio y la selección cuidadosa de estrategias de generación (temperature, top-k, top-p) son clave para obtener resultados coherentes y diversos.
