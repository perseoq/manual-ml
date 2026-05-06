# E01: Transformers — Fundamentos

## Objetivo
Comprender los bloques fundamentales del Transformer (Vaswani et al., 2017) y aplicarlos al procesamiento de descripciones de productos, reseñas y catálogos de ventas B2B.

---

## 1. Fundamentos Teóricos

El Transformer revolucionó el NLP al eliminar las recurrencias (RNN) y usar exclusivamente mecanismos de **atención**. Procesa toda la secuencia en paralelo, lo que permite entrenar con datos masivos.

### 1.1 Scaled Dot-Product Attention

El núcleo del Transformer es la atención por producto punto escalado:

```
Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*1.1 Scaled Dot-Product Attention.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



- **Q (Query)**: Representación de la palabra actual.
- **K (Key)**: Representación de todas las palabras contra las que se compara.
- **V (Value)**: Contenido informativo de cada palabra.
- **√d_k**: Factor de escala para evitar que el producto punto crezca con la dimensionalidad.

En ventas: Q es "laptop gaming", K es cada palabra en la descripción del catálogo, V es el embedding de cada palabra. La atención pondera qué palabras del catálogo son más relevantes para la consulta.

### 1.2 Multi-Head Attention

En lugar de una sola atención, se usan **h cabezas** en paralelo:

```
MultiHead(Q,K,V) = Concat(head_1, ..., head_h) · W_o
head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*1.2 Multi-Head Attention.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



Cada cabeza aprende un aspecto diferente de la relación entre palabras. Una cabeza puede aprender relaciones sintácticas; otra, semánticas de producto-precio.

### 1.3 Positional Encoding

Como el Transformer no tiene recurrencia, necesita codificar la posición de cada token:

**Sinusoidal** (Vaswani original):
```
PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*1.3 Positional Encoding.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



**Aprendida**: Se usa un embedding entrenable para cada posición (BERT, GPT).

### 1.4 Encoder Block

```
x → Multi-Head Attention → Add & LayerNorm → FFN → Add & LayerNorm
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*1.4 Encoder Block.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



- **Multi-Head Self-Attention**: Cada palabra atiende a todas las demás.
- **Add & LayerNorm**: Residual connection + normalización.
- **FFN**: Dense(2048, ReLU) + Dense(d_model).

En catálogos B2B: el encoder procesa descripciones de productos y captura relaciones entre atributos (marca, precio, especificaciones).

### 1.5 Decoder Block

```
x → Masked Multi-Head Attention → Add & LayerNorm 
  → Cross Multi-Head Attention → Add & LayerNorm → FFN → Add & LayerNorm
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*1.5 Decoder Block.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



- **Masked Self-Attention**: Cada token solo ve tokens anteriores (autoregresivo).
- **Cross-Attention**: Q viene del decoder, K y V del encoder.

### 1.6 Máscaras

- **Padding mask**: Ignora tokens `<PAD>` en el cálculo de atención.
- **Look-ahead mask**: En el decoder, impide que un token vea tokens futuros (causal).

### 1.7 Comparativa RNN vs Transformer

| Característica | RNN (LSTM) | Transformer |
|---|---|---|
| Procesamiento | Secuencial | Paralelo |
| Contexto lejano | Limitado (vanishing gradient) | Ilimitado (atención global) |
| Entrenamiento | Lento | Rápido (paralelizable) |
| Parámetros | Menos | Más |
| Memoria | O(1) estado oculto | O(n²) matriz de atención |

---

## 2. Ejemplos Prácticos

### Ejemplo 1: Scaled Dot-Product Attention desde cero

```python
import numpy as np

# Simulamos embeddings de 3 productos en un lote
# Frase: "laptop gaming oferta"
d_k = 4  # dimensión de cada embedding

Q = np.array([[1.0, 0.2, 0.1, 0.5],   # "laptop"
              [0.3, 1.0, 0.4, 0.1],   # "gaming"
              [0.1, 0.3, 1.0, 0.8]])  # "oferta"
K = Q.copy()  # self-attention
V = Q.copy()

def scaled_dot_product_attention(Q, K, V):
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    output = np.dot(weights, V)
    return output, weights

output, weights = scaled_dot_product_attention(Q, K, V)
print("Matriz de atención (pesos):")
print(np.round(weights, 3))
print("\nOutput atencional:")
print(np.round(output, 3))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: Scaled Dot-Product Attention desde cero.*

1. Simulamos embeddings de 3 productos en un lote
2. Frase: "laptop gaming oferta"

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: Multi-Head Attention (8 cabezas)

```python
import numpy as np

d_model = 64
h = 8
d_k = d_model // h  # 8

np.random.seed(42)
Q = np.random.randn(5, d_model)  # 5 tokens
K = np.random.randn(5, d_model)
V = np.random.randn(5, d_model)

W_q = np.random.randn(d_model, d_model)
W_k = np.random.randn(d_model, d_model)
W_v = np.random.randn(d_model, d_model)
W_o = np.random.randn(d_model, d_model)

Q_proj = Q @ W_q
K_proj = K @ W_k
V_proj = V @ W_v

# Dividir en h cabezas
Q_heads = Q_proj.reshape(5, h, d_k).transpose(1, 0, 2)
K_heads = K_proj.reshape(5, h, d_k).transpose(1, 0, 2)
V_heads = V_proj.reshape(5, h, d_k).transpose(1, 0, 2)

print(f"Q_heads shape: {Q_heads.shape}")  # (h, seq_len, d_k)

# Atención por cabeza
def attention(q, k, v):
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(d_k)
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    return weights @ v

head_outputs = []
for i in range(h):
    head_out = attention(Q_heads[i], K_heads[i], V_heads[i])
    head_outputs.append(head_out)

concat = np.concatenate(head_outputs, axis=-1)
output = concat @ W_o
print(f"Output Multi-Head shape: {output.shape}")  # (5, 64)
print("✅ Multi-Head Attention implementada")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: Multi-Head Attention (8 cabezas).*

1. Dividir en h cabezas
2. Atención por cabeza

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: Positional Encoding Sinusoidal

```python
import numpy as np
import matplotlib.pyplot as plt

def positional_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            PE[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            PE[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
    return PE

# Catalogar 20 productos
seq_len = 20  # palabras en una descripción
d_model = 32
PE = positional_encoding(seq_len, d_model)

print("Positional Encoding (primeros 5 tokens, dims 0-3):")
print(np.round(PE[:5, :4], 3))
print(f"\nShape: {PE.shape}")
print("✅ PE sinusoidal generada — cada posición tiene una firma única")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: Positional Encoding Sinusoidal.*

1. Catalogar 20 productos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: Positional Encoding Aprendido

```python
import torch
import torch.nn as nn

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        positions = torch.arange(x.size(1), device=x.device)
        return x + self.pe(positions)

max_len = 128
d_model = 256
pe_layer = LearnedPositionalEncoding(max_len, d_model)
x = torch.randn(2, 10, d_model)  # batch=2, 10 tokens
out = pe_layer(x)
print(f"Input shape: {x.shape}, Output shape: {out.shape}")
print("✅ PE aprendida añadida — útil para dominios específicos de ventas")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: Positional Encoding Aprendido.*

1. x: (batch, seq_len, d_model)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: Encoder Block Completo

```python
import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention + residual + norm
        attn_out, _ = self.mha(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        # FFN + residual + norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

d_model, n_heads, d_ff = 128, 4, 512
encoder = EncoderBlock(d_model, n_heads, d_ff)
x = torch.randn(2, 10, d_model)
out = encoder(x)
print(f"Encoder output shape: {out.shape}")
print("✅ Bloque Encoder construido — MHA + FFN + Add & Norm")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: Encoder Block Completo.*

1. Self-attention + residual + norm
2. FFN + residual + norm

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: Decoder Block con Masked Self-Attention

```python
import torch
import torch.nn as nn

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.masked_mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Masked self-attention
        attn1, _ = self.masked_mha(x, x, x, attn_mask=tgt_mask, key_padding_mask=src_mask)
        x = self.norm1(x + self.dropout(attn1))
        # Cross-attention: Q del decoder, K,V del encoder
        attn2, _ = self.cross_mha(x, encoder_output, encoder_output, key_padding_mask=src_mask)
        x = self.norm2(x + self.dropout(attn2))
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x

decoder = DecoderBlock(d_model=128, n_heads=4, d_ff=512)
x = torch.randn(2, 8, 128)
enc_out = torch.randn(2, 10, 128)
# Máscara causal: triángulo superior -inf
tgt_mask = torch.triu(torch.full((8, 8), float('-inf')), diagonal=1)
out = decoder(x, enc_out, tgt_mask=tgt_mask)
print(f"Decoder output shape: {out.shape}")
print("✅ Decoder block con masked cross-attention")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: Decoder Block con Masked Self-Attention.*

1. Masked self-attention
2. Cross-attention: Q del decoder, K,V del encoder
3. FFN
4. Máscara causal: triángulo superior -inf

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: Residual Connection + LayerNorm

```python
import torch
import torch.nn as nn

class ResidualLayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, sublayer_output):
        return self.norm(x + sublayer_output)

# Simular embeddings de productos
d_model = 64
x = torch.randn(4, 12, d_model)  # batch=4, 12 tokens
sublayer = nn.Linear(d_model, d_model)
res_norm = ResidualLayerNorm(d_model)

out = res_norm(x, sublayer(x))
print(f"Residual + LayerNorm output shape: {out.shape}")
print("✅ Residual connection evita vanishing gradient; LayerNorm estabiliza")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: Residual Connection + LayerNorm.*

1. Simular embeddings de productos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: Padding Mask

```python
import torch
import torch.nn.functional as F

# Descripciones con padding
descripciones = [
    "laptop gaming 16GB RAM",
    "mouse",
    "teclado mecánico RGB retroiluminado"
]

# Tokenización simple (simulada)
vocab = {"[PAD]": 0, "laptop": 1, "gaming": 2, "16GB": 3, "RAM": 4,
         "mouse": 5, "teclado": 6, "mecánico": 7, "RGB": 8, "retroiluminado": 9}
max_len = 5
batch = []
for desc in descripciones:
    tokens = [vocab.get(w, 0) for w in desc.split()]
    tokens = tokens[:max_len] + [0] * (max_len - len(tokens))
    batch.append(tokens)
batch = torch.tensor(batch)

# Padding mask: True donde hay padding
pad_mask = (batch == 0)
print("Padding mask (True = padding):")
print(pad_mask)

# En atención, los padding tokens deben ignorarse
# MultiheadAttention tiene key_padding_mask
mha = nn.MultiheadAttention(embed_dim=16, num_heads=2, batch_first=True)
x = torch.randn(3, 5, 16)
out, _ = mha(x, x, x, key_padding_mask=pad_mask)
print(f"Output con padding mask: {out.shape}")
print("✅ Padding mask ignora tokens de relleno en la atención")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: Padding Mask.*

1. Descripciones con padding
2. Tokenización simple (simulada)
3. Padding mask: True donde hay padding
4. En atención, los padding tokens deben ignorarse
5. MultiheadAttention tiene key_padding_mask

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: Look-Ahead Mask (Causal)

```python
import torch

def create_look_ahead_mask(size):
    mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
    return mask

seq_len = 6
mask = create_look_ahead_mask(seq_len)
print("Look-ahead mask (causal):")
print(mask)

# Aplicada: cada token ve solo sí mismo y anteriores
# Token 0: solo token 0
# Token 1: tokens 0,1
# ...
# Token 5: tokens 0,1,2,3,4,5
print("\n✅ Look-ahead mask: evita que el decoder vea tokens futuros")
print("   Esencial para generación autoregresiva (GPT)")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: Look-Ahead Mask (Causal).*

1. Aplicada: cada token ve solo sí mismo y anteriores
2. Token 0: solo token 0
3. Token 1: tokens 0,1
4. ...
5. Token 5: tokens 0,1,2,3,4,5

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: Visualizar Atención con Heatmap

```python
import numpy as np
import matplotlib.pyplot as plt

# Simular pesos de atención entre palabras de una descripción
palabras = ["laptop", "gaming", "con", "RTX", "3060", "y", "16GB", "RAM"]
n = len(palabras)

# Simular matriz de atención (atención real variaría por cabeza)
np.random.seed(42)
atencion = np.random.dirichlet(np.ones(n), size=n)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(atencion, cmap='Blues')
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(palabras, rotation=45, ha='right')
ax.set_yticklabels(palabras)
ax.set_title("Pesos de atención: ¿qué palabras atienden a cuáles?")
plt.colorbar(im, label='Peso de atención')
plt.tight_layout()
plt.savefig('/tmp/atencion_heatmap.png', dpi=100)
print("✅ Heatmap de atención generado — palabras clave en ventas destacan")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: Visualizar Atención con Heatmap.*

1. Simular pesos de atención entre palabras de una descripción
2. Simular matriz de atención (atención real variaría por cabeza)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: Comparar RNN vs Transformer

```python
import torch
import torch.nn as nn
import time

# RNN: procesa secuencial
rnn = nn.RNN(input_size=64, hidden_size=128, batch_first=True)
# Transformer: procesa en paralelo
encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, batch_first=True)
transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

# Misma entrada
x = torch.randn(32, 100, 64)  # batch=32, seq=100

# Tiempo RNN
start = time.time()
out_rnn, _ = rnn(x)
t_rnn = time.time() - start

# Tiempo Transformer
start = time.time()
out_trans = transformer(x)
t_trans = time.time() - start

print(f"RNN:        {t_rnn:.4f}s — procesa token a token")
print(f"Transformer: {t_trans:.4f}s — procesa en paralelo")
print(f"Speedup: {t_rnn/t_trans:.2f}x")
print("✅ Transformer es más rápido por paralelización")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: Comparar RNN vs Transformer.*

1. RNN: procesa secuencial
2. Transformer: procesa en paralelo
3. Misma entrada
4. Tiempo RNN
5. Tiempo Transformer

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: BERT — Encoder-Only (Bidireccional)

```python
# BERT es encoder-only: cada token ve todos los demás (bidireccional)
# Ideal para clasificación, NER, QA

# Simular clasificación de reseñas con encoder
import torch
import torch.nn as nn

class BERTLikeClassifier(nn.Module):
    def __init__(self, vocab_size=1000, d_model=128, n_heads=4, num_labels=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        # Pool del token [CLS] (posición 0)
        cls_token = x[:, 0, :]
        return self.classifier(cls_token)

# Reseñas: 0=negativa, 1=neutral, 2=positiva
model = BERTLikeClassifier()
reseñas = torch.randint(0, 100, (4, 20))  # 4 reseñas de 20 tokens
logits = model(reseñas)
print(f"Logits de clasificación (3 clases): {logits.shape}")
print("✅ BERT (encoder-only) procesa bidireccionalmente — ideal para entender contexto completo")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: BERT — Encoder-Only (Bidireccional).*

1. BERT es encoder-only: cada token ve todos los demás (bidireccional)
2. Ideal para clasificación, NER, QA
3. Simular clasificación de reseñas con encoder
4. Pool del token [CLS] (posición 0)
5. Reseñas: 0=negativa, 1=neutral, 2=positiva

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: GPT — Decoder-Only (Autoregresivo, Causal)

```python
# GPT es decoder-only: genera token a token (causal)
import torch
import torch.nn as nn

class GPTLikeGenerator(nn.Module):
    def __init__(self, vocab_size=1000, d_model=128, n_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, n_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x, memory=None, mask=None):
        x = self.embedding(x)
        x = self.decoder(x, memory if memory is not None else x, tgt_mask=mask)
        return self.lm_head(x)

vocab_size = 1000
model = GPTLikeGenerator(vocab_size)
# Máscara causal
seq_len = 10
causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
x = torch.randint(0, vocab_size, (1, seq_len))
logits = model(x, mask=causal_mask)
print(f"Logits de generación (vocab_size={vocab_size}): {logits.shape}")
next_token = logits[:, -1, :].argmax(-1)
print(f"Próximo token predicho: {next_token.item()}")
print("✅ GPT (decoder-only) genera autoregresivamente — ideal para descripciones de productos")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: GPT — Decoder-Only (Autoregresivo, Causal).*

1. GPT es decoder-only: genera token a token (causal)
2. Máscara causal

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: T5 — Encoder-Decoder (Text-to-Text)

```python
# T5: encoder-decoder, todo se formula como text-to-text
# Útil para traducción, resumen, clasificación

class T5LikeModel(nn.Module):
    def __init__(self, vocab_size=1000, d_model=128, n_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        decoder_layer = nn.TransformerDecoderLayer(d_model, n_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt)
        memory = self.encoder(src_emb, src_key_padding_mask=src_mask)
        out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        return self.lm_head(out)

# T5 recibe input y genera output (text-to-text)
model = T5LikeModel()
src = torch.randint(0, 1000, (1, 15))  # "traducir al inglés: laptop gaming"
tgt = torch.randint(0, 1000, (1, 12))  # "gaming laptop"
mask = torch.triu(torch.full((12, 12), float('-inf')), diagonal=1)
logits = model(src, tgt, tgt_mask=mask)
print(f"T5 logits shape: {logits.shape}")
print("✅ T5 (encoder-decoder) unifica todas las tareas como text-to-text")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: T5 — Encoder-Decoder (Text-to-Text).*

1. T5: encoder-decoder, todo se formula como text-to-text
2. Útil para traducción, resumen, clasificación
3. T5 recibe input y genera output (text-to-text)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: Contar Parámetros de un Transformer Pequeño

```python
import torch
import torch.nn as nn

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Transformer pequeño para clasificación de productos
class SmallTransformer(nn.Module):
    def __init__(self, vocab_size=5000, d_model=128, n_heads=4, num_layers=3, num_labels=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Embedding(128, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=512, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device)
        x = self.embedding(x) + self.pos_encoding(positions)
        x = self.encoder(x)
        return self.classifier(x[:, 0, :])

model = SmallTransformer()
params = count_parameters(model)
print(f"Parámetros totales: {params:,}")
print("\nDesglose por componente:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"  {name}: {param.numel():,} parámetros")
print(f"\n✅ Un Transformer pequeño tiene ~{params//1000}M parámetros")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: Contar Parámetros de un Transformer Pequeño.*

1. Transformer pequeño para clasificación de productos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 16: Implementar Transformer desde Cero (PyTorch)

```python
import torch
import torch.nn as nn
import math

class TransformerFromScratch(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, num_encoder_layers=3,
                 num_decoder_layers=3, d_ff=512, max_len=128, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self._create_pos_encoding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        self.output_proj = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _create_pos_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        src_emb = self.dropout(self.embedding(src) * math.sqrt(self.d_model) +
                               self.pos_encoding[:, :src.size(1), :].to(src.device))
        tgt_emb = self.dropout(self.embedding(tgt) * math.sqrt(self.d_model) +
                               self.pos_encoding[:, :tgt.size(1), :].to(tgt.device))
        memory = self.encoder(src_emb, mask=src_mask)
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return self.output_proj(output)

# Crear modelo
vocab_size = 10000
model = TransformerFromScratch(vocab_size)
src = torch.randint(0, vocab_size, (2, 20))
tgt = torch.randint(0, vocab_size, (2, 15))
tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1))
out = model(src, tgt, tgt_mask=tgt_mask)
print(f"Transformer completo — output shape: {out.shape}")
print(f"Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")
print("✅ Transformer implementado desde cero con encoder y decoder")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: Implementar Transformer desde Cero (PyTorch).*

1. Crear modelo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 17: Entrenar Transformer Pequeño para Clasificación

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Datos sintéticos: 500 descripciones de productos, 5 categorías
num_samples = 500
seq_len = 20
vocab_size = 1000
num_categories = 5

X = torch.randint(1, vocab_size, (num_samples, seq_len))
y = torch.randint(0, num_categories, (num_samples,))

# Modelo (solo encoder)
class ProductClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=2, num_labels=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=256, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        return self.classifier(x[:, 0, :])

model = ProductClassifier(vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento
for epoch in range(5):
    optimizer.zero_grad()
    logits = model(X)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()
    acc = (logits.argmax(-1) == y).float().mean()
    print(f"Época {epoch+1}: loss={loss.item():.4f}, acc={acc.item():.4f}")

print("\n✅ Transformer pequeño entrenado para clasificar productos en 5 categorías")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: Entrenar Transformer Pequeño para Clasificación.*

1. Datos sintéticos: 500 descripciones de productos, 5 categorías
2. Modelo (solo encoder)
3. Entrenamiento

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 18: Integrador — Transformer para Clasificación de Descripciones

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ===== DATOS DE EJEMPLO: Catálogo de productos =====
descripciones = [
    "laptop gaming con RTX 3060 16GB RAM",
    "mouse inalámbrico ergonómico 6 botones",
    "teclado mecánico RGB switches Cherry MX",
    "monitor 27 pulgadas 4K IPS 144Hz",
    "silla ergonómica ajustable soporte lumbar",
    "auriculares bluetooth cancelación ruido activa",
    "webcam 1080p con micrófono integrado",
    "hub USB C 7 puertos HDMI SD",
    "disco SSD 1TB NVMe lectura 3500MB/s",
    "cable HDMI 2.1 3 metros 48Gbps",
    "impresora láser wifi duplex automático",
    "router wifi 6 doble banda gigabit",
]
categorias = [
    "laptop", "periférico", "periférico", "monitor",
    "mueble", "audio", "periférico", "accesorio",
    "almacenamiento", "cable", "impresora", "red"
]
label2id = {cat: i for i, cat in enumerate(sorted(set(categorias)))}
id2label = {i: cat for cat, i in label2id.items()}
y = torch.tensor([label2id[cat] for cat in categorias])

# Tokenización simple
vocab = {"[PAD]": 0, "[UNK]": 1}
for desc in descripciones:
    for w in desc.split():
        if w not in vocab:
            vocab[w] = len(vocab)

def encode(desc, max_len=10):
    tokens = [vocab.get(w, 1) for w in desc.split()]
    tokens = tokens[:max_len] + [0] * (max_len - len(tokens))
    return tokens

X = torch.tensor([encode(d) for d in descripciones])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modelo
model = nn.Sequential(
    nn.Embedding(len(vocab), 32),
    nn.TransformerEncoder(nn.TransformerEncoderLayer(32, 2, batch_first=True), num_layers=2),
    lambda x: x[:, 0, :],
    nn.Linear(32, len(label2id))
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(30):
    optimizer.zero_grad()
    logits = model(X_train)
    loss = criterion(logits, y_train)
    loss.backward()
    optimizer.step()

# Evaluación
with torch.no_grad():
    logits = model(X_test)
    preds = logits.argmax(-1)
    y_true = y_test.tolist()
    y_pred = preds.tolist()
    acc = (preds == y_test).float().mean().item()

print(f"Accuracy en test: {acc:.2%}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=list(id2label.values()),
                            zero_division=0, labels=range(len(id2label))))

# Predecir nueva descripción
nueva = "teclado inalámbrico slim para oficina"
x_new = torch.tensor([encode(nueva)])
with torch.no_grad():
    pred = model(x_new).argmax(-1).item()
print(f"\nNueva descripción: '{nueva}'")
print(f"Categoría predicha: {id2label[pred]}")
print("\n✅ INTEGRADOR: Transformer completo para clasificación de descripciones de productos")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador — Transformer para Clasificación de Descripciones.*

1. ===== DATOS DE EJEMPLO: Catálogo de productos =====
2. Tokenización simple
3. Modelo
4. Evaluación
5. Predecir nueva descripción

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 3. Ejercicios Propuestos

1. **Atención manual**: Dada la frase "laptop oferta descuento 50%", calcula manualmente la matriz de atención con d_k=2. Muestra Q, K, scores y pesos softmax.

2. **Multi-Head variante**: Modifica el ejemplo 2 para usar 4 cabezas en lugar de 8. ¿Cómo cambia la forma de la salida?

3. **PE visual**: Genera un gráfico de la matriz de positional encoding (50x32) usando matplotlib. Describe el patrón visual que observas.

4. **Encoder stack**: Crea un encoder con 6 bloques (num_layers=6) y clasifica 20 descripciones de productos sintéticas. Reporta accuracy.

5. **Decoder con padding**: Añade padding mask al decoder del ejemplo 6. Muestra cómo cambian los pesos de atención.

6. **Comparativa de máscaras**: Crea un script que compare padding mask vs look-ahead mask. Explica cuándo usar cada una.

7. **BERT vs GPT**: Toma una descripción de producto de 10 palabras y pásala por un encoder BERT-like y un decoder GPT-like. Compara las representaciones obtenidas.

8. **Proyecto final**: Implementa un Transformer completo (encoder + decoder) para traducir descripciones de productos de inglés a español. Usa datos sintéticos (10 pares). Entrena y evalúa.

---

## 4. Resumen

| Concepto | Propósito | Aplicación en Ventas |
|---|---|---|
| Scaled Dot-Product Attention | Ponderar relevancia entre tokens | Encontrar palabras clave en descripciones |
| Multi-Head Attention | Capturar múltiples relaciones | Precio, marca, especificaciones simultáneamente |
| Positional Encoding | Codificar orden de palabras | Mantener "garantía 2 años" ≠ "2 años garantía" |
| Encoder | Representación bidireccional | Clasificación de reseñas, categorización |
| Decoder | Generación autoregresiva | Descripciones automáticas, chatbots |
| Residual + LayerNorm | Estabilizar entrenamiento | Entrenar modelos más profundos |
| Máscaras (padding/look-ahead) | Controlar contexto | Batch eficiente, generación causal |

**Conclusión**: El Transformer es la base de BERT, GPT, T5 y todos los modelos modernos de NLP. Su capacidad de procesamiento paralelo y atención global lo hace ideal para analizar grandes volúmenes de descripciones de productos, reseñas y catálogos B2B.

*"La atención es todo lo que necesitas" — Vaswani et al., 2017*
