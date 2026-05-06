# AP08 — Cheatsheet NLP

## 1. NLTK — Natural Language Toolkit

```python
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, ne_chunk, FreqDist

# nltk.download("punkt")
# nltk.download("stopwords")
# nltk.download("wordnet")
# nltk.download("averaged_perceptron_tagger")

# Tokenización
texto = "Las ventas de laptops crecieron 20% en 2024. Excelente resultado."
tokens = word_tokenize(texto, language="spanish")
oraciones = sent_tokenize(texto, language="spanish")

# Stopwords
stop_words = set(stopwords.words("spanish"))
tokens_filtrados = [t for t in tokens if t.lower() not in stop_words and t.isalpha()]

# Stemming
stemmer = PorterStemmer()
stems = [stemmer.stem(t) for t in tokens_filtrados]

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmas = [lemmatizer.lemmatize(t) for t in tokens_filtrados]

# POS Tagging
tagged = pos_tag(tokens_filtrados)

# Frecuencia de palabras
freq = FreqDist(tokens_filtrados)
freq.most_common(10)

# N-gramas
from nltk import ngrams, bigrams, trigrams
list(bigrams(tokens))
list(trigrams(tokens))
list(ngrams(tokens, 4))
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*1. NLTK — Natural Language Toolkit.*

1. nltk.download("punkt")
2. nltk.download("stopwords")
3. nltk.download("wordnet")
4. nltk.download("averaged_perceptron_tagger")
5. Tokenización
6. Stopwords
7. Stemming
8. Lemmatization

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 2. spaCy

```python
import spacy

# Cargar modelo
# python -m spacy download es_core_news_sm
nlp = spacy.load("es_core_news_sm")

# Procesar texto
doc = nlp("Las ventas de laptops en Amazon crecieron un 20% durante 2024.")

# Tokenización y análisis
for token in doc:
    print(f"{token.text:12} {token.lemma_:12} {token.pos_:8} "
          f"{token.dep_:10} {token.ent_type_}")

# Entidades nombradas (NER)
for ent in doc.ents:
    print(f"{ent.text:20} {ent.label_:10}")

# Análisis de oraciones
for sent in doc.sents:
    print(sent.text)

# Dependencias
for chunk in doc.noun_chunks:
    print(f"Noun chunk: {chunk.text}")

# Similitud (con modelo con word vectors)
doc1 = nlp("alta demanda de productos")
doc2 = nlp("incremento en ventas")
doc1.similarity(doc2)

# Matcher basado en reglas
from spacy.matcher import Matcher
matcher = Matcher(nlp.vocab)
pattern = [{"LOWER": "ventas"}, {"POS": "ADP"}, {"POS": "NOUN"}]
matcher.add("VENTAS_PATTERN", [pattern])
matches = matcher(doc)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*2. spaCy.*

1. Cargar modelo
2. python -m spacy download es_core_news_sm
3. Procesar texto
4. Tokenización y análisis
5. Entidades nombradas (NER)
6. Análisis de oraciones
7. Dependencias
8. Similitud (con modelo con word vectors)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 3. TextBlob

```python
from textblob import TextBlob

# Análisis de sentimiento
blob = TextBlob("The product is amazing and works perfectly!")
print(blob.sentiment)           # Sentiment(polarity=0.8, subjectivity=0.9)
print(blob.sentiment.polarity)  # -1 (negativo) a +1 (positivo)
print(blob.sentiment.subjectivity)  # 0 (objetivo) a 1 (subjetivo)

# Tokenización y etiquetado
blob.words                       # WordList(["The", "product", ...])
blob.tags                        # POS tags
blob.noun_phrases                # frases nominales

# Traducción (requiere internet)
blob.translate(from_lang="en", to="es")

# Corrección ortográfica
blob.correct()

# Análisis de sentimiento con TextBlob en español
blob_es = TextBlob("El producto es excelente y funciona perfectamente")
print(blob_es.sentiment)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*3. TextBlob.*

1. Análisis de sentimiento
2. Tokenización y etiquetado
3. Traducción (requiere internet)
4. Corrección ortográfica
5. Análisis de sentimiento con TextBlob en español

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 4. Word Embeddings — Word2Vec

```python
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# Preparar datos
oraciones = [
    "las ventas de laptops aumentaron en enero",
    "los clientes prefieren productos con descuento",
    "el marketing digital impulsa las ventas online"
]
sentences = [simple_preprocess(s) for s in oraciones]

# Entrenar Word2Vec
model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    sg=0  # 0=CBOW, 1=Skip-gram
)

# Obtener vector
vector = model.wv["ventas"]

# Palabras similares
model.wv.most_similar("ventas", topn=5)

# Analogías: rey - hombre + mujer = reina
model.wv.most_similar(positive=["laptop", "venta"], negative=["online"])

# Guardar y cargar
model.save("word2vec_ventas.model")
model = Word2Vec.load("word2vec_ventas.model")
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*4. Word Embeddings — Word2Vec.*

1. Preparar datos
2. Entrenar Word2Vec
3. Obtener vector
4. Palabras similares
5. Analogías: rey - hombre + mujer = reina
6. Guardar y cargar

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 5. TF-IDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd

# Documentos
docs = [
    "laptop gaming con descuento especial",
    "mouse inalámbrico para oficina",
    "laptop para programación con descuento"
]

# Bag of Words
bow = CountVectorizer(max_features=1000, stop_words="spanish")
X_bow = bow.fit_transform(docs)
print(bow.get_feature_names_out())

# TF-IDF
tfidf = TfidfVectorizer(
    max_features=1000,
    stop_words="spanish",
    ngram_range=(1, 2),      # unigramas y bigramas
    min_df=2,                  # mínimo 2 documentos
    max_df=0.8                 # ignora términos en >80% docs
)
X_tfidf = tfidf.fit_transform(docs)

# Ver resultado
df_tfidf = pd.DataFrame(
    X_tfidf.toarray(),
    columns=tfidf.get_feature_names_out()
)
print(df_tfidf)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*5. TF-IDF.*

1. Documentos
2. Bag of Words
3. TF-IDF
4. Ver resultado

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 6. Transformers — HuggingFace

```python
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# Pipeline simplificado
classifier = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)
result = classifier("Excelente producto, súper recomendado")
print(result)

# Clasificación zero-shot
zero_shot = pipeline("zero-shot-classification")
result = zero_shot(
    "Necesito una laptop para gaming",
    candidate_labels=["tecnología", "ropa", "hogar"]
)
print(result)

# Cargar modelo y tokenizer
model_name = "dccuchile/bert-base-spanish-wwm-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=3
)

# Tokenización
texts = ["Producto excelente", "Mala calidad", "Regular"]
encodings = tokenizer(
    texts,
    truncation=True,
    padding=True,
    return_tensors="pt",
    max_length=128
)

# Inferencia
with torch.no_grad():
    outputs = model(**encodings)
    predictions = torch.softmax(outputs.logits, dim=-1)

# Text Generation
generator = pipeline("text-generation", model="gpt2")
generated = generator("Las ventas de este trimestre", max_length=50)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*6. Transformers — HuggingFace.*

1. Pipeline simplificado
2. Clasificación zero-shot
3. Cargar modelo y tokenizer
4. Tokenización
5. Inferencia
6. Text Generation

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 7. Sentence Transformers

```python
from sentence_transformers import SentenceTransformer, util

# Cargar modelo de embeddings
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Obtener embeddings de oraciones
oraciones = [
    "El producto llegó en excelente estado",
    "Mala experiencia de compra",
    "Cliente satisfecho con la compra"
]
embeddings = model.encode(oraciones)

# Calcular similitud coseno
similitudes = util.cos_sim(embeddings, embeddings)
print(similitudes)

# Similitud entre pares
sim = util.cos_sim(embeddings[0], embeddings[1])
print(f"Similitud: {sim.item():.3f}")

# Búsqueda semántica
consulta = "clientes felices"
consulta_emb = model.encode(consulta)
scores = util.cos_sim(consulta_emb, embeddings)[0]
best_idx = scores.argmax()
print(f"Mejor match: {oraciones[best_idx]} (score: {scores[best_idx]:.3f})")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*7. Sentence Transformers.*

1. Cargar modelo de embeddings
2. Obtener embeddings de oraciones
3. Calcular similitud coseno
4. Similitud entre pares
5. Búsqueda semántica

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 8. Preprocesamiento de Texto

```python
import re
import unicodedata

def limpiar_texto(texto):
    # Minúsculas
    texto = texto.lower()
    # Eliminar acentos
    texto = unicodedata.normalize("NFKD", texto)
    texto = texto.encode("ascii", "ignore").decode("utf-8")
    # Eliminar URLs
    texto = re.sub(r"http\S+|www\S+|https\S+", "", texto)
    # Eliminar menciones y hashtags
    texto = re.sub(r"@\w+|#\w+", "", texto)
    # Eliminar caracteres especiales
    texto = re.sub(r"[^a-záéíóúñü\s]", " ", texto)
    # Eliminar espacios múltiples
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

# Aplicar a DataFrame
df["texto_limpio"] = df["comentario"].apply(limpiar_texto)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*8. Preprocesamiento de Texto.*

1. Minúsculas
2. Eliminar acentos
3. Eliminar URLs
4. Eliminar menciones y hashtags
5. Eliminar caracteres especiales
6. Eliminar espacios múltiples
7. Aplicar a DataFrame

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 9. Clasificación de Texto

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Pipeline con TF-IDF + Logistic Regression
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, stop_words="spanish")),
    ("clf", LogisticRegression(C=1.0, class_weight="balanced"))
])

pipe.fit(X_train, y_train)
preds = pipe.predict(X_test)
probas = pipe.predict_proba(X_test)[:, 1]
```

**Salida:**

```
# El modelo ajusta sus parámetros durante el entrenamiento.
# Los parámetros aprendidos quedan disponibles en el objeto.
```

**Explicación línea por línea:**

*9. Clasificación de Texto.*

1. Pipeline con TF-IDF + Logistic Regression

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 10. Chatbots y Preguntas/Respuestas

```python
from transformers import pipeline

# QA pipeline
qa_pipeline = pipeline("question-answering")
contexto = "Las ventas de laptops aumentaron 30% en enero debido al home office."
result = qa_pipeline(
    question="¿Por qué aumentaron las ventas?",
    context=contexto
)
print(f"Respuesta: {result['answer']} (score: {result['score']:.3f})")

# Conversational
chatbot = pipeline("conversational")
from transformers import Conversation
conv = Conversation("¿Cómo van las ventas este mes?")
chatbot(conv)
print(conv.generated_responses[-1])
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*10. Chatbots y Preguntas/Respuestas.*

1. QA pipeline
2. Conversational

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 11. Tokenización Avanzada

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Hola, me gustaría comprar una laptop gaming"

# Tokenización básica
tokens = tokenizer.tokenize(text)
print(tokens)

# IDs de tokens
ids = tokenizer.encode(text)
print(ids)

# Decodificar
decoded = tokenizer.decode(ids)
print(decoded)

# Con formato para modelo
encoding = tokenizer(
    text,
    truncation=True,
    padding="max_length",
    max_length=128,
    return_tensors="pt",
    return_attention_mask=True
)
print(encoding["input_ids"].shape)
print(encoding["attention_mask"].shape)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*11. Tokenización Avanzada.*

1. Tokenización básica
2. IDs de tokens
3. Decodificar
4. Con formato para modelo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 12. Análisis de Sentimiento Personalizado

```python
from transformers import Trainer, TrainingArguments

# Entrenar clasificador de sentimiento con HuggingFace
# (código resumido)
from datasets import Dataset

dataset = Dataset.from_pandas(df[["texto", "sentimiento"]])
train_dataset = dataset.train_test_split(test_size=0.2)

training_args = TrainingArguments(
    output_dir="./sentiment_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3
)

trainer = Trainer(
    model=AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=3
    ),
    args=training_args,
    train_dataset=train_dataset["train"],
    eval_dataset=train_dataset["test"]
)

trainer.train()
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*12. Análisis de Sentimiento Personalizado.*

1. Entrenar clasificador de sentimiento con HuggingFace
2. (código resumido)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 13. Resumen de Texto

```python
from transformers import pipeline

# Summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

reporte = """
El mercado de ventas online experimentó un crecimiento significativo durante
el último trimestre, impulsado principalmente por el incremento en la demanda
de productos electrónicos y la expansión de las plataformas de comercio
electrónico en la región. Las categorías con mayor rendimiento fueron
tecnología y hogar, mientras que moda mostró una recuperación moderada.
"""

resumen = summarizer(reporte, max_length=50, min_length=20)
print(resumen[0]["summary_text"])
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*13. Resumen de Texto.*

1. Summarization

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 14. Named Entity Recognition (NER)

```python
# spaCy NER
nlp = spacy.load("es_core_news_sm")
doc = nlp("Amazon vendió 500 laptops en México durante enero de 2024")

for ent in doc.ents:
    print(f"{ent.text:20} -> {ent.label_}")

# Extraer tipos específicos
organizaciones = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
fechas = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
montos = [ent.text for ent in doc.ents if ent.label_ == "MONEY"]

# NER con HuggingFace
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER")
results = ner_pipeline("Apple is looking to buy a UK startup for $1 billion")
for r in results:
    print(f"{r['word']:15} -> {r['entity']} ({r['score']:.2f})")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*14. Named Entity Recognition (NER).*

1. spaCy NER
2. Extraer tipos específicos
3. NER con HuggingFace

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 15. Topic Modeling

```python
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

docs = [
    "laptop gaming descuento oferta",
    "mouse teclado oficina inalámbrico",
    "laptop programación descuento oferta",
    "monitor escritorio oficina",
]

vectorizer = CountVectorizer(max_features=1000, stop_words="spanish")
X = vectorizer.fit_transform(docs)

lda = LatentDirichletAllocation(n_components=2, random_state=42)
lda.fit(X)

# Mostrar tópicos
def mostrar_topicos(model, features, n_words=5):
    for topic_idx, topic in enumerate(model.components_):
        top_words = [features[i] for i in topic.argsort()[:-n_words-1:-1]]
        print(f"Tópico {topic_idx}: {', '.join(top_words)}")

mostrar_topicos(lda, vectorizer.get_feature_names_out())
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*15. Topic Modeling.*

1. Mostrar tópicos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## Referencia Rápida

| Tarea | Librería | Código |
|-------|----------|--------|
| Tokenización | NLTK | `word_tokenize(texto)` |
| Lematización | spaCy | `token.lemma_` |
| POS Tagging | spaCy | `token.pos_` |
| NER | spaCy | `doc.ents` |
| Sentimiento | TextBlob | `TextBlob(text).sentiment` |
| Embeddings | Gensim | `Word2Vec(sentences, vector_size=100)` |
| TF-IDF | sklearn | `TfidfVectorizer().fit_transform(docs)` |
| Transformer | HuggingFace | `pipeline("sentiment-analysis")` |
| Sentence Emb | sentence-transformers | `SentenceTransformer().encode(texts)` |
| QA | HuggingFace | `pipeline("question-answering")` |
| Resumen | HuggingFace | `pipeline("summarization")` |
| Topic Model | sklearn | `LatentDirichletAllocation(n_components=5)` |
