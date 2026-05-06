# CP34: 30 Ejercicios Integradores de Nivel Experto

## Contexto
Consolidar todo lo aprendido en el nivel experto con 30 ejercicios practicos cubriendo Transformers, RAG, series temporales, sistemas de recomendacion, optimizacion, MLOps y despliegue.

---

## Ejercicios 1-3: Transformers (atencion, BERT, GPT)

### Ejercicio 1: Implementar atencion multi-cabeza desde cero

**Enunciado:** Implementa atencion multi-cabeza (Multi-Head Attention) usando solo NumPy.

**Pista:** Divide d_model en num_heads cabezas de dimension d_k. Softmax(Q*K^T / sqrt(d_k)) * V.

```python
import numpy as np

def multi_head_attention(Q, K, V, num_heads=8):
    batch, seq_len, d_model = Q.shape
    d_k = d_model // num_heads
    def split_heads(x):
        return x.reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    Qs = split_heads(Q); Ks = split_heads(K); Vs = split_heads(V)
    scores = np.matmul(Qs, Ks.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attn /= attn.sum(axis=-1, keepdims=True)
    out = np.matmul(attn, Vs)
    return out.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model), attn

Q = np.random.randn(2, 10, 512)
K = np.random.randn(2, 10, 512)
V = np.random.randn(2, 10, 512)
output, weights = multi_head_attention(Q, K, V)
print(f'Output shape: {output.shape}')
print(f'Weights shape: {weights.shape}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejercicio 1: Implementar atencion multi-cabeza desde cero.*

1. `import numpy as np` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



**Explicacion:** Atencion multi-cabeza permite atender diferentes representaciones. Cada cabeza aprende relaciones distintas entre posiciones. Division por sqrt(d_k) evita gradientes extremos en softmax.

---

### Ejercicio 2: Fine-tuning de BERT para clasificacion

**Enunciado:** Fine-tuning de BERT en resenas de productos (positiva/negativa).

**Pista:** AutoTokenizer, AutoModelForSequenceClassification, Trainer.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np, pandas as pd

np.random.seed(42)
textos = ['Excelente producto', 'Muy buena calidad'] * 60 + ['Pesimo producto', 'Mala calidad'] * 60
labels = [1] * 120 + [0] * 120
df = pd.DataFrame({'text': textos[:500], 'label': labels[:500]})
dataset = Dataset.from_pandas(df).train_test_split(test_size=0.2)
tok = AutoTokenizer.from_pretrained('bert-base-uncased')
dataset = dataset.map(lambda x: tok(x['text'], padding='max_length', truncation=True, max_length=128), batched=True)
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
trainer = Trainer(model=model, args=TrainingArguments(output_dir='./bert', num_train_epochs=3,
                  per_device_train_batch_size=16, evaluation_strategy='epoch', report_to='none'),
                  train_dataset=dataset['train'], eval_dataset=dataset['test'])
trainer.train()
print(trainer.evaluate())
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejercicio 2: Fine-tuning de BERT para clasificacion.*

1. `from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments` — Importa las librerías necesarias para el análisis.
2. `from datasets import Dataset` — Importa las librerías necesarias para el análisis.
3. `import numpy as np, pandas as pd` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



**Explicacion:** Fine-tuning adapta BERT pre-entrenado a tarea especifica. Tokenizador convierte texto a tokens [CLS], [SEP]. Capa de clasificacion se entrena desde cero.

---

### Ejercicio 3: Generacion de texto con GPT-2

**Enunciado:** Genera descripciones de productos con GPT-2 controlando temperatura y sampling.

**Pista:** pipeline('text-generation'), temperature, top_k, top_p.

```python
from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2', device=-1)
set_seed(42)
for prompt in ['Laptop profesional con', 'Auriculares inalambricos']:
    out = generator(prompt, max_length=80, temperature=0.8, top_k=50, top_p=0.9, do_sample=True)
    print(f'Prompt: {prompt}')
    print(f'Generado: {out[0][\"generated_text\"]}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejercicio 3: Generacion de texto con GPT-2.*

1. `from transformers import pipeline, set_seed` — Importa las librerías necesarias para el análisis.
2. `print(f'Prompt: {prompt}')` — Muestra el resultado por pantalla.
3. `print(f'Generado: {out[0][\"generated_text\"]}')` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



**Explicacion:** GPT-2 genera texto autoregresivamente. Temperatura controla aleatoriedad, top-k limita a k tokens mas probables, top-p (nucleus) selecciona hasta acumular probabilidad p.

---

## Ejercicios 4-6: HuggingFace (pipeline, fine-tuning, semantic search)

### Ejercicio 4: Pipeline de analisis de sentimiento

**Enunciado:** Analiza sentimiento de 100 resenas. Calcula % positivas y confianza promedio.

**Pista:** pipeline('sentiment-analysis'), batch_size=16.

```python
from transformers import pipeline
import numpy as np
sentiment = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
resenas = ['The product arrived on time', 'Terrible quality', 'Good value', 'Very disappointed'] * 25
results = sentiment(resenas, truncation=True, batch_size=16)
pos = sum(1 for r in results if r['label'] == 'POSITIVE')
conf = np.mean([r['score'] for r in results])
print(f'Positivas: {pos} ({pos}%), Confianza promedio: {conf:.3f}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejercicio 4: Pipeline de analisis de sentimiento.*

1. `from transformers import pipeline` — Importa las librerías necesarias para el análisis.
2. `import numpy as np` — Importa las librerías necesarias para el análisis.
3. `print(f'Positivas: {pos} ({pos}%), Confianza promedio: {conf:.3f}')` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



**Explicacion:** Pipeline abstrae tokenizacion, inferencia y post-procesamiento. DistilBERT: 95% rendimiento con 40% menos parametros.

---

### Ejercicio 5: Semantic search en catalogo

**Enunciado:** Busqueda semantica en 100 productos con sentence-transformers + FAISS.

**Pista:** SentenceTransformer + FAISS IndexFlatIP + busqueda por similaridad coseno.

```python
from sentence_transformers import SentenceTransformer
import faiss, numpy as np
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
catalogo = [f'Producto categoria {i%5}' for i in range(100)]
emb = model.encode(catalogo, normalize_embeddings=True)
index = faiss.IndexFlatIP(emb.shape[1]); index.add(emb.astype(np.float32))
qe = model.encode(['computadora portatil'], normalize_embeddings=True)
d, i = index.search(qe.astype(np.float32), k=5)
for j, idx in enumerate(i[0]): print(f'{j+1}. {catalogo[idx]} (score: {d[0][j]:.3f})')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejercicio 5: Semantic search en catalogo.*

1. `from sentence_transformers import SentenceTransformer` — Importa las librerías necesarias para el análisis.
2. `import faiss, numpy as np` — Importa las librerías necesarias para el análisis.
3. `for j, idx in enumerate(i[0]): print(f'{j+1}. {catalogo[idx]} (score: {d[0][j]:.3f})')` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



**Explicacion:** Busqueda semantica captura significado, no solo palabras clave. FAISS IndexFlatIP = producto punto (coseno con embeddings normalizados).

---

### Ejercicio 6: Clasificacion multiclase con BERT

**Enunciado:** BERT para clasificar consultas en precio/inventario/envio/devolucion/producto.

**Pista:** num_labels=5, dataset sintetico.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
cats = {0: 'precio', 1: 'inventario', 2: 'envio'}
textos = ['cuanto cuesta', 'hay stock', 'cuanto tarda'] * 100
labels = [0, 1, 2] * 100
df = pd.DataFrame({'text': textos[:300], 'label': labels[:300]})
dataset = Dataset.from_pandas(df).train_test_split(test_size=0.2)
tok = AutoTokenizer.from_pretrained('bert-base-uncased')
dataset = dataset.map(lambda x: tok(x['text'], padding='max_length', truncation=True, max_length=64), batched=True)
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
trainer = Trainer(model=model, args=TrainingArguments(output_dir='./cls', num_train_epochs=3,
                  per_device_train_batch_size=16, report_to='none'),
                  train_dataset=dataset['train'], eval_dataset=dataset['test'])
trainer.train()
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Ejercicio 6: Clasificacion multiclase con BERT.*

1. `from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments` — Importa las librerías necesarias para el análisis.
2. `from datasets import Dataset` — Importa las librerías necesarias para el análisis.
3. `import pandas as pd` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



**Explicacion:** Clasificacion multiclase con softmax sobre num_labels. BERT ajusta pesos para distinguir categorias semanticamente similares.

---

## Ejercicios 7-9: RAG y Chatbots (LangChain, FAISS)

### Ejercicio 7: RAG basico

**Enunciado:** RAG con FAISS + LLM local para responder preguntas sobre politicas de ventas.

**Pista:** SentenceTransformer + FAISS + pipeline text2text-generation.

```python
from sentence_transformers import SentenceTransformer
import faiss, numpy as np
from transformers import pipeline

docs = ['Precios incluyen IVA', 'Descuento 5% en compras > $10,000', 'Envio gratis > $2,000',
        'Devolucion en 30 dias', 'Garantia de 1 ano', 'Pago con tarjeta']
model = SentenceTransformer('all-MiniLM-L6-v2')
emb = model.encode(docs, normalize_embeddings=True)
index = faiss.IndexFlatIP(emb.shape[1]); index.add(emb.astype(np.float32))
gen = pipeline('text2text-generation', model='google/flan-t5-small')

def rag(pregunta):
    qe = model.encode([pregunta], normalize_embeddings=True)
    d, i = index.search(qe.astype(np.float32), k=2)
    ctx = ' '.join([docs[j] for j in i[0]])
    return gen(f'Contexto: {ctx} Pregunta: {pregunta}', max_length=100)[0]['generated_text']

for p in ['Cual es el descuento?', 'Cuantos dias de devolucion?']:
    print(f'P: {p} => R: {rag(p)}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejercicio 7: RAG basico.*

1. `from sentence_transformers import SentenceTransformer` — Importa las librerías necesarias para el análisis.
2. `import faiss, numpy as np` — Importa las librerías necesarias para el análisis.
3. `from transformers import pipeline` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



**Explicacion:** RAG = Retrieval-Augmented Generation. Recupera contexto relevante con FAISS y lo inyecta en el prompt del LLM para respuestas factuales.

---

### Ejercicio 8: Chatbot con memoria conversacional

**Enunciado:** Chatbot que recuerda contexto usando ConversationBufferMemory de LangChain.

**Pista:** ConversationChain + memory + HuggingFacePipeline.

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

pipe = pipeline('text2text-generation', model='google/flan-t5-small', max_length=200)
llm = HuggingFacePipeline(pipeline=pipe)
memory = ConversationBufferMemory(return_messages=True)
conv = ConversationChain(llm=llm, memory=memory, verbose=False)

dialogo = ['Hola, busco una laptop', 'Que tenga 16GB RAM', 'Cual es el precio?']
for p in dialogo:
    r = conv.predict(input=p)
    print(f'User: {p} => Bot: {r[:100]}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejercicio 8: Chatbot con memoria conversacional.*

1. `from langchain.memory import ConversationBufferMemory` — Importa las librerías necesarias para el análisis.
2. `from langchain.chains import ConversationChain` — Importa las librerías necesarias para el análisis.
3. `from langchain.llms import HuggingFacePipeline` — Importa las librerías necesarias para el análisis.
4. `from transformers import pipeline` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



**Explicacion:** ConversationBufferMemory almacena historial completo. La cadena pasa contexto al LLM para coherencia conversacional.

---

### Ejercicio 9: RAG con filtro por metadatos

**Enunciado:** RAG que filtra chunks por categoria (precio, envio, garantia).

**Pista:** Almacena categoria con embedding, filtra post-recuperacion.

```python
from sentence_transformers import SentenceTransformer
import faiss, numpy as np

docs = [('precio', 'Precio base $25,000'), ('precio', 'Descuento 10%'),
        ('envio', 'Envio gratis > $2,000'), ('envio', 'Tiempo entrega 3-5d'),
        ('garantia', 'Garantia 1 ano'), ('garantia', 'No cubre mal uso')]
model = SentenceTransformer('all-MiniLM-L6-v2')
texts = [d[1] for d in docs]; cats = [d[0] for d in docs]
emb = model.encode(texts, normalize_embeddings=True)
index = faiss.IndexFlatIP(emb.shape[1]); index.add(emb.astype(np.float32))

def search(consulta, filtro=None, k=2):
    qe = model.encode([consulta], normalize_embeddings=True)
    d, i = index.search(qe.astype(np.float32), k=len(docs))
    res = []
    for idx, dist in zip(i[0], d[0]):
        if filtro is None or cats[idx] == filtro:
            res.append((texts[idx], cats[idx], dist))
            if len(res) == k: break
    return res

print('Sin filtro:', search('descuento'))
print('Filtro precio:', search('descuento', filtro='precio'))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejercicio 9: RAG con filtro por metadatos.*

1. `from sentence_transformers import SentenceTransformer` — Importa las librerías necesarias para el análisis.
2. `import faiss, numpy as np` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



**Explicacion:** Filtro por metadatos mejora precision al limitar busqueda a documentos relevantes por categoria.

---

## Ejercicios 10-12: Series Temporales

### Ejercicio 10: Descomposicion de series temporales

**Enunciado:** Descompone ventas diarias en tendencia + estacionalidad semanal/anual + residuo.

**Pista:** seasonal_decompose de statsmodels, modelo aditivo y multiplicativo.

```python
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
np.random.seed(42)
fechas = pd.date_range(start='2023-01-01', periods=365, freq='D')
t = np.linspace(100, 200, 365); s = 15*np.sin(2*np.pi*fechas.dayofweek/7)
a = 25*np.sin(2*np.pi*fechas.dayofyear/365); r = np.random.normal(0,8,365)
ts = pd.Series(t+s+a+r, index=fechas)
descomp = seasonal_decompose(ts, model='additive', period=7)
print(f'Fuerza tendencia: {1 - np.var(descomp.resid)/np.var(ts):.3f}')
descomp.plot(); plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejercicio 10: Descomposicion de series temporales.*

1. `import pandas as pd, numpy as np, matplotlib.pyplot as plt` — Importa las librerías necesarias para el análisis.
2. `from statsmodels.tsa.seasonal import seasonal_decompose` — Importa las librerías necesarias para el análisis.
3. `print(f'Fuerza tendencia: {1 - np.var(descomp.resid)/np.var(ts):.3f}')` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



**Explicacion:** Descomposicion aditiva (T+S+R). Fuerza = 1 - var(resid)/var(total). Cercano a 1 indica patron fuerte.

---

### Ejercicio 11: ARIMA y SARIMA

**Enunciado:** Pronostica ventas semanales con ARIMA/SARIMA. auto_arima para parametros.

**Pista:** pmdarima.auto_arima, seasonal=True, m=52.

```python
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error
ts_w = ts.resample('W').sum(); train, test = ts_w[:-12], ts_w[-12:]
arima = auto_arima(train, seasonal=False, stepwise=True, suppress_warnings=True)
sarima = auto_arima(train, seasonal=True, m=52, stepwise=True, suppress_warnings=True)
for n, m in [('ARIMA', arima), ('SARIMA', sarima)]:
    print(f'{n} orden={m.order}: MAE={mean_absolute_error(test, m.predict(12)):.0f}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejercicio 11: ARIMA y SARIMA.*

1. `from pmdarima import auto_arima` — Importa las librerías necesarias para el análisis.
2. `from sklearn.metrics import mean_absolute_error` — Importa las librerías necesarias para el análisis.
3. `print(f'{n} orden={m.order}: MAE={mean_absolute_error(test, m.predict(12)):.0f}')` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



**Explicacion:** ARIMA(p,d,q) captura autocorrelacion. SARIMA(P,D,Q,m) anyade estacionalidad. auto_arima busca parametros optimos.

---

### Ejercicio 12: Prophet con regresores externos

**Enunciado:** Prophet con regresores promocion y precio. Compara con modelo base.

**Pista:** add_regressor para cada variable externa.

```python
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import pandas as pd, numpy as np
df = pd.DataFrame({'ds': pd.date_range('2023-01-01', periods=365, freq='D'),
    'y': 100 + 10*np.sin(2*np.pi*np.arange(365)/7) + np.random.normal(0,15,365),
    'promo': np.random.choice([0,1],365,p=[0.85,0.15]),
    'precio': 500 + np.random.normal(0,30,365)})
train, test = df.iloc[:292], df.iloc[292:]
base = Prophet(); base.fit(train[['ds','y']])
pred_base = base.predict(base.make_future_dataframe(periods=73))['yhat'].iloc[-73:].values
reg = Prophet(); reg.add_regressor('promo'); reg.add_regressor('precio')
reg.fit(train[['ds','y','promo','precio']])
fut = test[['ds','promo','precio']].copy()
pred_reg = reg.predict(fut)['yhat'].values
print(f'MAE base: {mean_absolute_error(test["y"], pred_base):.2f}')
print(f'MAE reg: {mean_absolute_error(test["y"], pred_reg):.2f}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejercicio 12: Prophet con regresores externos.*

1. `from prophet import Prophet` — Importa las librerías necesarias para el análisis.
2. `from sklearn.metrics import mean_absolute_error` — Importa las librerías necesarias para el análisis.
3. `import pandas as pd, numpy as np` — Importa las librerías necesarias para el análisis.
4. `base = Prophet(); base.fit(train[['ds','y']])` — Entrena el modelo con los datos de entrenamiento.
5. `pred_base = base.predict(base.make_future_dataframe(periods=73))['yhat'].iloc[-73:].values` — Genera predicciones sobre nuevos datos.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



**Explicacion:** Regresores en Prophet modelan impacto lineal de variables exogenas como promocion y precio, mejorando precision del forecast.

---

## Ejercicios 13-15: LSTM avanzado y Kalman Filter

### Ejercicio 13: LSTM con atencion

**Enunciado:** LSTM con mecanismo de atencion para prediccion de demanda. Compara con LSTM estandar.

**Pista:** Capa Attention() entre salida LSTM y prediccion final.

```python
import numpy as np, tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Attention
from sklearn.preprocessing import MinMaxScaler

np.random.seed(42); tf.random.set_seed(42)
data = np.sin(np.arange(1000)*0.1) + np.random.normal(0,0.1,1000)
scaler = MinMaxScaler(); data_s = scaler.fit_transform(data.reshape(-1,1))
X, y = [], []
for i in range(30, len(data_s)): X.append(data_s[i-30:i,0]); y.append(data_s[i,0])
X, y = np.array(X), np.array(y); X = X.reshape(-1,30,1)
split = int(0.8*len(X))

inp = Input(shape=(30,1))
lstm = LSTM(64, return_sequences=True)(inp)
attn = Attention()([lstm, lstm])
flat = tf.keras.layers.Flatten()(attn)
out = Dense(1)(flat)
model = Model(inp, out); model.compile(optimizer='adam', loss='mse')
model.fit(X[:split], y[:split], epochs=20, validation_data=(X[split:], y[split:]), verbose=0)
print(f'MAE: {np.mean(np.abs(y[split:]-model.predict(X[split:],verbose=0).flatten())):.4f}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejercicio 13: LSTM con atencion.*

1. `import numpy as np, tensorflow as tf` — Importa las librerías necesarias para el análisis.
2. `from tensorflow.keras.models import Model` — Importa las librerías necesarias para el análisis.
3. `from tensorflow.keras.layers import LSTM, Dense, Input, Attention` — Importa las librerías necesarias para el análisis.
4. `from sklearn.preprocessing import MinMaxScaler` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



**Explicacion:** Atencion permite al LSTM pesar diferentes timesteps, mejorando prediccion al enfocarse en patrones relevantes.

---

### Ejercicio 14: Kalman Filter para suavizado

**Enunciado:** Kalman Filter 1D para suavizar ventas diarias con ruido de observacion.

**Pista:** pykalman.KalmanFilter, transition_matrices y observation_matrices.

```python
from pykalman import KalmanFilter
import numpy as np, matplotlib.pyplot as plt
np.random.seed(42)
true = np.sin(np.arange(200)*0.1)*50 + 100
obs = true + np.random.normal(0, 15, 200)
kf = KalmanFilter(initial_state_mean=100, n_dim_obs=1,
                  transition_matrices=[1], observation_matrices=[1],
                  transition_covariance=[[1]], observation_covariance=[[225]],
                  initial_state_covariance=[[100]])
smoothed = kf.smooth(obs)[0]
print(f'MSE: {np.mean((true - smoothed.flatten())**2):.2f}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejercicio 14: Kalman Filter para suavizado.*

1. `from pykalman import KalmanFilter` — Importa las librerías necesarias para el análisis.
2. `import numpy as np, matplotlib.pyplot as plt` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



**Explicacion:** Kalman Filter estima estado oculto (ventas reales) desde observaciones ruidosas. Balance entre modelo de transicion y observacion.

---

### Ejercicio 15: Comparacion LSTM vs Kalman vs Prophet

**Enunciado:** Compara LSTM, Kalman Filter y Prophet en prediccion de ventas con MAE y RMSE.

**Pista:** Misma ventana para los 3, walk-forward.

```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
np.random.seed(42); data = np.sin(np.arange(400)*0.05)*30 + 100 + np.random.normal(0,10,400)
train, test = data[:350], data[350:]
# LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
X, y = [], []
for i in range(10, 400): X.append(data[i-10:i]); y.append(data[i])
X, y = np.array(X).reshape(-1,10,1), y
model = Sequential([LSTM(50, input_shape=(10,1)), Dense(1)])
model.compile(optimizer='adam', loss='mse')
model.fit(X[:340], y[:340], epochs=10, verbose=0)
pred = model.predict(X[340:], verbose=0).flatten()
print(f'LSTM MAE: {mean_absolute_error(y[340:], pred):.2f}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejercicio 15: Comparacion LSTM vs Kalman vs Prophet.*

1. LSTM

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



**Explicacion:** LSTM captura patrones complejos no lineales. Kalman es optimo para sistemas lineales gaussianos. Prophet maneja estacionalidades explicita.

---

## Ejercicios 16-18: Sistemas de Recomendacion

### Ejercicio 16: Filtrado colaborativo basico

**Enunciado:** User-Based CF con similaridad coseno para recomendar productos.

**Pista:** cosine_similarity entre usuarios. Recomienda productos de usuarios similares no comprados.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
np.random.seed(42)
ratings = np.random.choice([0,1], (50,30), p=[0.8,0.2])
user_sim = cosine_similarity(ratings)
def recommend(user_id, k=5):
    sim_users = np.argsort(-user_sim[user_id])[1:6]
    scores = ratings[sim_users].sum(axis=0)
    scores[ratings[user_id]==1] = -1
    return np.argsort(-scores)[:k]
print(f'Recomendaciones usuario 0: {recommend(0)}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejercicio 16: Filtrado colaborativo basico.*

1. `import numpy as np` — Importa las librerías necesarias para el análisis.
2. `from sklearn.metrics.pairwise import cosine_similarity` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



**Explicacion:** User-CF: usuarios similares tienen preferencias parecidas. Recomienda productos que usuarios similares compraron.

---

### Ejercicio 17: Matrix Factorization con SVD

**Enunciado:** Descompone matriz usuario-producto con TruncatedSVD en factores latentes.

**Pista:** sklearn.decomposition.TruncatedSVD, n_components=10.

```python
from sklearn.decomposition import TruncatedSVD
import numpy as np
ratings = np.random.choice([0,1], (50,30), p=[0.8,0.2])
svd = TruncatedSVD(n_components=10, random_state=42)
pred = np.dot(svd.fit_transform(ratings), svd.components_)
def recommend_mf(user_id, k=5):
    scores = pred[user_id].copy()
    scores[ratings[user_id]==1] = -1
    return np.argsort(-scores)[:k]
print(f'MF top-5: {recommend_mf(0)}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejercicio 17: Matrix Factorization con SVD.*

1. `from sklearn.decomposition import TruncatedSVD` — Importa las librerías necesarias para el análisis.
2. `import numpy as np` — Importa las librerías necesarias para el análisis.
3. `print(f'MF top-5: {recommend_mf(0)}')` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



**Explicacion:** SVD descubre factores latentes (preferencias ocultas). Producto punto de factores usuario x item reconstruye matriz de preferencias.

---

### Ejercicio 18: Evaluacion Hit Rate@k

**Enunciado:** Evalua recomendador con Hit Rate@k, Precision@k, Recall@k.

**Pista:** Oculta 1 item comprado, verifica si aparece en top-k.

```python
import numpy as np
ratings = np.random.choice([0,1], (100,50), p=[0.85,0.15])
svd = TruncatedSVD(n_components=10); pred = np.dot(svd.fit_transform(ratings), svd.components_)
def hit_rate(pred, ratings, k=10):
    hits = 0
    for u in range(len(ratings)):
        bought = np.where(ratings[u]==1)[0]
        if len(bought)==0: continue
        test = np.random.choice(bought)
        train = ratings[u].copy(); train[test]=0
        scores = pred[u].copy(); scores[train==1] = -np.inf
        if test in np.argsort(-scores)[:k]: hits+=1
    return hits/len(ratings)
for k in [5,10,20]: print(f'Hit Rate@{k}: {hit_rate(pred,ratings,k):.4f}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejercicio 18: Evaluacion Hit Rate@k.*

1. `import numpy as np` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



**Explicacion:** Hit Rate mide si el item oculto aparece en top-k recomendaciones. Estandar para evaluacion offline de recomendadores.

---

## Ejercicios 19-20: RecSys Hibrido y DL

### Ejercicio 19: Sistema hibrido CF + CB

**Enunciado:** Combina CF (SVD) con Content-Based (TF-IDF + coseno). Pesos 70/30.

**Pista:** TF-IDF sobre descripciones + SVD sobre matriz. Ponderacion lineal.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import numpy as np

descs = ['Laptop potente para programacion', 'Mouse ergonomico', 'Monitor 4K'] * 5
tfidf = TfidfVectorizer(); cb_sim = cosine_similarity(tfidf.fit_transform(descs))

n_users, n_items = 20, 15
ratings = np.random.choice([0,1], (n_users, n_items), p=[0.7,0.3])
svd = TruncatedSVD(n_components=5); cf_pred = np.dot(svd.fit_transform(ratings), svd.components_)

def hybrid(user, item, w_cf=0.7):
    return w_cf * cf_pred[user,item] + (1-w_cf) * cb_sim[item].mean()

user = 0; scores = np.array([hybrid(user,i) for i in range(n_items)])
scores[ratings[user]==1] = -1
print(f'Hibrido top-5: {np.argsort(-scores)[:5]}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejercicio 19: Sistema hibrido CF + CB.*

1. `from sklearn.feature_extraction.text import TfidfVectorizer` — Importa las librerías necesarias para el análisis.
2. `from sklearn.metrics.pairwise import cosine_similarity` — Importa las librerías necesarias para el análisis.
3. `from sklearn.decomposition import TruncatedSVD` — Importa las librerías necesarias para el análisis.
4. `import numpy as np` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



**Explicacion:** Sistema hibrido combina CF (preferencias comunitarias) con CB (similaridad de contenido). Robusto ante cold start.

---

### Ejercicio 20: Neural Collaborative Filtering

**Enunciado:** NCF con GMF + MLP en TensorFlow para recomendacion.

**Pista:** Embeddings separados para GMF (multiplicacion) y MLP (concatenacion + Dense).

```python
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Multiply, Concatenate, Dense
import numpy as np

n_u, n_i, dim = 50, 30, 16
ratings = np.random.choice([0,1], (n_u, n_i), p=[0.8,0.2])
users, items = np.where(ratings==1)
neg_u = np.random.randint(0, n_u, len(users))
neg_i = np.random.randint(0, n_i, len(users))
mask = ratings[neg_u, neg_i]==0
X_u = np.concatenate([users, neg_u[mask][:len(users)]])
X_i = np.concatenate([items, neg_i[mask][:len(users)]])
y = np.concatenate([np.ones(len(users)), np.zeros(len(users))])

in_u = Input(shape=(1,)); in_i = Input(shape=(1,))
mf = Multiply()([Flatten()(Embedding(n_u,dim)(in_u)), Flatten()(Embedding(n_i,dim)(in_i))])
mlp_u = Flatten()(Embedding(n_u,dim)(in_u)); mlp_i = Flatten()(Embedding(n_i,dim)(in_i))
mlp = Dense(32,activation='relu')(Concatenate()([mlp_u,mlp_i]))
out = Dense(1,activation='sigmoid')(Concatenate()([mf,mlp]))
model = Model([in_u,in_i], out)
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit([X_u,X_i], y, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
print(f'NCF AUC: {max(model.history.history[\"val_accuracy\"]):.3f}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejercicio 20: Neural Collaborative Filtering.*

1. `import tensorflow as tf` — Importa las librerías necesarias para el análisis.
2. `from tensorflow.keras.models import Model` — Importa las librerías necesarias para el análisis.
3. `from tensorflow.keras.layers import Input, Embedding, Flatten, Multiply, Concatenate, Dense` — Importa las librerías necesarias para el análisis.
4. `import numpy as np` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



**Explicacion:** NCF = GMF (lineal, embeddings multiplicados) + MLP (no lineal, Dense). Fusion captura interacciones complejas usuario-item.

---

## Ejercicios 21-23: Optimizacion (Hyperopt, Optuna, cuantizacion)

### Ejercicio 21: Optuna para hiperparametros

**Enunciado:** Optimiza Random Forest con Optuna para ventas. n_estimators, max_depth, min_samples_split.

**Pista:** study.optimize con n_trials=30.

```python
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import numpy as np

X = np.random.randn(500, 10); y = X[:,0]*3 + X[:,1]*(-1.5) + np.random.normal(0,0.5,500)

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20)
    }
    scores = cross_val_score(RandomForestRegressor(**params, random_state=42, n_jobs=-1), X, y, cv=3, scoring='neg_mean_squared_error')
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)
print(f'Mejores params: {study.best_params}, MSE: {-study.best_value:.4f}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejercicio 21: Optuna para hiperparametros.*

1. `import optuna` — Importa las librerías necesarias para el análisis.
2. `from sklearn.ensemble import RandomForestRegressor` — Importa las librerías necesarias para el análisis.
3. `from sklearn.model_selection import cross_val_score` — Importa las librerías necesarias para el análisis.
4. `import numpy as np` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



**Explicacion:** Optuna usa busqueda bayesiana (TPE). Sugiere valores basados en resultados previos, convergiendo mas rapido que grid search.

---

### Ejercicio 22: Cuantizacion TFLite

**Enunciado:** Cuantiza modelo Keras LSTM de FP32 a INT8 con TFLite. Compara tamano.

**Pista:** TFLiteConverter con optimizations=[tf.lite.Optimize.DEFAULT].

```python
import tensorflow as tf, numpy as np
model = tf.keras.Sequential([tf.keras.layers.LSTM(32, input_shape=(10,5)), tf.keras.layers.Dense(1)])
model.compile(optimizer='adam', loss='mse')
model.fit(np.random.randn(50,10,5), np.random.randn(50), epochs=3, verbose=0)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite = converter.convert()
print(f'Modelo cuantizado: {len(tflite)/1024:.1f}KB')
interp = tf.lite.Interpreter(model_content=tflite); interp.allocate_tensors()
import time
start = time.perf_counter()
interp.set_tensor(interp.get_input_details()[0]['index'], np.random.randn(1,10,5).astype(np.float32))
interp.invoke(); end = time.perf_counter()
print(f'Tiempo inferencia: {(end-start)*1000:.2f}ms')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejercicio 22: Cuantizacion TFLite.*

1. `import tensorflow as tf, numpy as np` — Importa las librerías necesarias para el análisis.
2. `model.fit(np.random.randn(50,10,5), np.random.randn(50), epochs=3, verbose=0)` — Entrena el modelo con los datos de entrenamiento.
3. `print(f'Modelo cuantizado: {len(tflite)/1024:.1f}KB')` — Muestra el resultado por pantalla.
4. `import time` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



**Explicacion:** Cuantizacion reduce pesos FP32->INT8/FP16. Tamano 4x menor, inferencia mas rapida, minima perdida de precision.

---

### Ejercicio 23: Hyperopt para XGBoost

**Enunciado:** Optimiza XGBoost para clasificacion con Hyperopt. fmin, tpe.suggest.

**Pista:** space con hp.choice, hp.uniform. max_evals=30.

```python
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import numpy as np

X = np.random.randn(500, 10); y = (X[:,0]+X[:,1]+np.random.normal(0,0.5,500)>0).astype(int)
space = {'n_estimators': hp.quniform('n_estimators',50,300,10), 'max_depth': hp.choice('max_depth',[3,5,7]),
         'learning_rate': hp.uniform('learning_rate',0.01,0.3)}

def objective(params):
    params['n_estimators'] = int(params['n_estimators'])
    scores = cross_val_score(XGBClassifier(**params, random_state=42, eval_metric='logloss'), X, y, cv=3, scoring='roc_auc')
    return {'loss': -scores.mean(), 'status': STATUS_OK}

best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=20, trials=Trials())
print(f'Mejores params: {best}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejercicio 23: Hyperopt para XGBoost.*

1. `from hyperopt import fmin, tpe, hp, Trials, STATUS_OK` — Importa las librerías necesarias para el análisis.
2. `from sklearn.model_selection import cross_val_score` — Importa las librerías necesarias para el análisis.
3. `from xgboost import XGBClassifier` — Importa las librerías necesarias para el análisis.
4. `import numpy as np` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



**Explicacion:** Hyperopt con TPE busca optimos eficientemente. Mejor que grid/random search en espacios de alta dimension.

---

## Ejercicios 24-25: MLOps (MLflow, DVC)

### Ejercicio 24: MLflow experiment tracking

**Enunciado:** MLflow para registrar parametros, metricas y modelos de experimento.

**Pista:** mlflow.start_run(), log_param, log_metric, log_model.

```python
import mlflow, mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np

X = np.random.randn(500,10); y = X[:,0]*2 + X[:,1]*(-1) + np.random.normal(0,0.3,500)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2)
mlflow.set_experiment('Ventas_Prediccion')
with mlflow.start_run():
    params = {'n_estimators': 100, 'max_depth': 10}
    model = RandomForestRegressor(**params, random_state=42)
    model.fit(Xtr, ytr)
    mae = mean_absolute_error(yte, model.predict(Xte))
    mlflow.log_params(params); mlflow.log_metric('MAE', mae)
    mlflow.sklearn.log_model(model, 'model')
    print(f'MAE: {mae:.4f}, Run: {mlflow.active_run().info.run_id}')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejercicio 24: MLflow experiment tracking.*

1. `import mlflow, mlflow.sklearn` — Importa las librerías necesarias para el análisis.
2. `from sklearn.ensemble import RandomForestRegressor` — Importa las librerías necesarias para el análisis.
3. `from sklearn.model_selection import train_test_split` — Importa las librerías necesarias para el análisis.
4. `from sklearn.metrics import mean_absolute_error` — Importa las librerías necesarias para el análisis.
5. `import numpy as np` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



**Explicacion:** MLflow registra parametros, metricas y modelos. Facilita comparacion de experimentos y reproducibilidad.

---

### Ejercicio 25: DVC para datos

**Enunciado:** Versiona datasets de ventas con DVC. Inicializa, agrega, commitea.

**Pista:** dvc init, dvc add, git commit, dvc push.

```bash
# Comandos DVC
# dvc init
# dvc add ventas.csv
# git add ventas.csv.dvc .gitignore
# git commit -m "Add ventas data"
# dvc remote add -d s3store s3://bucket/dvc
# dvc push
print('DVC versiona datos como Git versiona codigo. Almacena hash en .dvc, datos en remote storage.')
```

**Explicacion:** DVC trackea datasets y modelos. Archivos grandes van a S3/GCS, DVC guarda los hash. Reproducibilidad total.

---

## Ejercicios 26-27: Despliegue (FastAPI, Docker)

### Ejercicio 26: API con FastAPI

**Enunciado:** API REST con FastAPI para servir modelo de prediccion de ventas.

**Pista:** Pydantic para validacion, uvicorn para servir.

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Ejercicio 25: DVC para datos.*

1. Ejercicios 26-27: Despliegue (FastAPI, Docker)
2. Ejercicio 26: API con FastAPI

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=50)
model.fit(np.random.randn(100,4), np.random.randn(100))
app = FastAPI(title='API Ventas')

class Input(BaseModel):
    f1: float; f2: float; f3: float; f4: float
class Output(BaseModel):
    prediccion: float; modelo: str

@app.post('/predict', response_model=Output)
def predict(data: Input):
    X = np.array([[data.f1, data.f2, data.f3, data.f4]])
    return Output(prediccion=round(float(model.predict(X)[0]), 2), modelo='RF')
```

**Explicacion:** FastAPI con Pydantic valida entrada/salida. uvicorn es servidor ASGI de alto rendimiento. Endpoint /predict recibe JSON y retorna prediccion.

---

### Ejercicio 27: Dockerizar aplicacion

**Enunciado:** Crea Dockerfile multi-stage para la API ML.

**Pista:** FROM python:3.10-slim, multi-stage build.

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejercicio 27: Dockerizar aplicacion

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

dockerfile
FROM python:3.10-slim as builder
WORKDIR /app
COPY requirements.txt .; RUN pip install --no-cache-dir -r requirements.txt
FROM python:3.10-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

bash
# docker build -t ventas-api .
# docker run -d -p 8000:8000 ventas-api
print('Docker multi-stage reduce tamano de imagen. Asegura consistencia entre entornos.')
```

**Explicacion:** Multi-stage build separa construccion (con dependencias de build) de ejecucion (solo runtime). Imagen final pequena y segura.

---

## Ejercicios 28-29: Monitoreo y Explicabilidad

### Ejercicio 28: Data drift con Evidently

**Enunciado:** Detecta data drift en features de ventas usando Evidently.

**Pista:** Evidently Report con DataDriftPreset.

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejercicios 28-29: Monitoreo y Explicabilidad
2. Ejercicio 28: Data drift con Evidently

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import pandas as pd, numpy as np
ref = pd.DataFrame({'precio': np.random.normal(500,100,1000), 'cantidad': np.random.poisson(3,1000)})
cur = pd.DataFrame({'precio': np.random.normal(550,150,200), 'cantidad': np.random.poisson(2,200)})
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref, current_data=cur)
report.save_html('drift.html')
print(f'Drift detectado: {report.as_dict()[\"metrics\"][0][\"result\"][\"dataset_drift\"]}')
```

**Explicacion:** Evidently detecta cambios en distribuciones de datos (data drift) que degradan modelos. Reportes HTML para compartir.

---

### Ejercicio 29: SHAP explicabilidad

**Enunciado:** SHAP para explicar predicciones de modelo de ventas. Importancia local y global.

**Pista:** shap.TreeExplainer, summary_plot, waterfall_plot.

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejercicio 29: SHAP explicabilidad

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
import shap, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
X = pd.DataFrame({'precio': np.random.normal(500,100,500), 'gasto_mkt': np.random.normal(1000,300,500)})
y = X['precio']*(-0.5) + X['gasto_mkt']*0.3 + np.random.normal(0,50,500)
model = RandomForestRegressor(n_estimators=100).fit(X, y)
explainer = shap.TreeExplainer(model); shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X, show=False); import matplotlib.pyplot as plt; plt.show()
print('SHAP importance:')
for name, val in zip(X.columns, np.abs(shap_values).mean(axis=0)):
    print(f'  {name}: {val:.2f}')
```

**Explicacion:** SHAP descompone prediccion en contribuciones de cada feature (Shapley values de teoria de juegos). Ayuda a entender y debuggear modelos.

---

## Ejercicio 30: Integrador final (pipeline completo produccion)

**Enunciado:** Pipeline completo ML en produccion: carga, entrenamiento, evaluacion, despliegue, monitoreo. MLflow + FastAPI + SHAP.

**Pista:** Combina todos los anteriores en un Pipeline class.

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejercicio 30: Integrador final (pipeline completo produccion)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import mlflow, shap, json
from datetime import datetime

class PipelineML:
    def __init__(self): self.model = None; self.metrics = {}
    def cargar(self):
        X = pd.DataFrame({'precio': np.random.normal(500,100,500), 'gasto': np.random.normal(1000,300,500)})
        y = X['precio']*(-0.3) + X['gasto']*0.2 + np.random.normal(0,40,500)
        return train_test_split(X, y, test_size=0.2, random_state=42)
    def entrenar(self, Xtr, ytr):
        with mlflow.start_run():
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(Xtr, ytr)
            mlflow.sklearn.log_model(self.model, 'modelo')
        return self
    def evaluar(self, Xte, yte):
        pred = self.model.predict(Xte)
        self.metrics['mae'] = mean_absolute_error(yte, pred)
        return self
    def explicar(self, X):
        explainer = shap.TreeExplainer(self.model)
        return dict(zip(X.columns, np.abs(explainer.shap_values(X)).mean(axis=0)))
    def reporte(self):
        return {'timestamp': datetime.now().isoformat(), 'metrics': self.metrics,
                'features': list(self.model.feature_names_in_)}

pipe = PipelineML(); Xtr, Xte, ytr, yte = pipe.cargar()
pipe.entrenar(Xtr, ytr).evaluar(Xte, yte)
print(json.dumps(pipe.reporte(), indent=2))
print(f'SHAP: {pipe.explicar(Xte)}')
```

**Explicacion:** Pipeline integrador cubre todo el ciclo MLOps: datos -> MLflow -> evaluacion -> SHAP -> prediccion. Arquitectura modular, produccion-ready.

---

## Resumen

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Resumen

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
print('''
30 EJERCICIOS - HABILIDADES CUBIERTAS
=====================================
1-3:  Transformers (atencion, BERT, GPT-2)
4-6:  HuggingFace (pipeline, semantic search, clasificacion)
7-9:  RAG y Chatbots (FAISS, LangChain, memoria)
10-12: Series Temporales (descomposicion, ARIMA, Prophet)
13-15: LSTM avanzado (atencion, Kalman Filter, comparacion)
16-18: Recomendacion (CF, SVD, evaluacion)
19-20: RecSys Hibrido y NCF
21-23: Optimizacion (Optuna, cuantizacion, Hyperopt)
24-25: MLOps (MLflow, DVC)
26-27: Despliegue (FastAPI, Docker)
28-29: Monitoreo (Evidently, SHAP)
30:    Pipeline completo produccion
=====================================
''')
print('CP34: 30 Ejercicios Integradores de Nivel Experto completado.')
```

