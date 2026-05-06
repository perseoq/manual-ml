# E06: Hugging Face — RAG y Chatbots para Ventas

## Objetivo
Construir chatbots de ventas B2B usando RAG (Retrieval Augmented Generation) con LangChain, Chroma/FAISS y modelos de Hugging Face. Sistema completo para responder preguntas sobre productos, catálogos y precios.

---

## 1. Fundamentos Teóricos

### 1.1 ¿Qué es RAG?

**Retrieval Augmented Generation** combina:
1. **Retrieval**: Buscar documentos relevantes en una base de conocimiento.
2. **Generation**: Usar un LLM para generar respuestas basadas en los documentos recuperados.

**Ventajas**:
- El LLM no necesita saber todo (reduce alucinaciones).
- La base de conocimiento se actualiza independientemente.
- Responde con fuentes verificables.

### 1.2 Arquitectura RAG

```
Usuario: "¿Qué laptops tienes con RTX 3060?"
                    |
            [Retriever (FAISS/Chroma)]
                    |
    Documentos relevantes: ["Laptop A: RTX 3060...", ...]
                    |
            [LLM + Prompt Template]
                    |
    Respuesta: "Tenemos la Laptop Gaming Pro con RTX 3060..."
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*1.2 Arquitectura RAG.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### 1.3 Componentes LangChain

| Componente | Función |
|---|---|
| Document Loaders | Cargar CSV, PDF, texto |
| Text Splitters | Dividir documentos en chunks |
| Vector Stores (Chroma, FAISS) | Indexar y buscar embeddings |
| RetrievalQA | QA simple sobre documentos |
| ConversationalRetrievalChain | QA con historial |
| Memory | Buffer, ventana, resumen |
| PromptTemplate | Plantillas de prompt |

### 1.4 Chain Types

- **stuff**: Todos los documentos en un solo prompt (simple, límite de contexto).
- **map_reduce**: Cada documento por separado, luego resume.
- **refine**: Refina respuesta iterativamente con cada documento.
- **map_rerank**: Evalúa y rankea respuestas parciales.

### 1.5 Memory Types

| Tipo | Descripción | Uso |
|---|---|---|
| ConversationBufferMemory | Todo el historial | Contexto completo |
| ConversationBufferWindowMemory | Últimos K intercambios | Contexto reciente |
| ConversationSummaryMemory | Resumen de la conversación | Ahorrar tokens |
| ConversationSummaryBufferMemory | Resumen + últimos tokens | Balance |

---

## 2. Ejemplos Prácticos

### Ejemplo 1: Cargar CSV de Productos con CSVLoader

```python
from langchain.document_loaders import CSVLoader
import pandas as pd
import os

# Crear CSV de ejemplo
os.makedirs('/tmp/data', exist_ok=True)
df = pd.DataFrame({
    'nombre': ['Laptop Gaming Pro', 'Mouse Ergonómico', 'Teclado RGB', 'Monitor 4K'],
    'descripcion': [
        'Laptop gaming con RTX 3060, i7, 16GB RAM, SSD 512GB',
        'Mouse inalámbrico con 6 botones, sensor 16000 DPI',
        'Teclado mecánico Cherry MX Brown, retroiluminación RGB',
        'Monitor IPS 4K 27 pulgadas, 144Hz, HDR400',
    ],
    'precio': [1200, 45, 120, 450],
    'stock': [10, 50, 30, 15],
})
df.to_csv('/tmp/data/productos.csv', index=False)

# Cargar con CSVLoader
loader = CSVLoader(
    file_path='/tmp/data/productos.csv',
    source_column='nombre',
    csv_args={'delimiter': ','}
)
docs = loader.load()

print(f"Documentos cargados: {len(docs)}")
for i, doc in enumerate(docs):
    print(f"\nDocumento {i+1}:")
    print(f"  Source: {doc.metadata['source']}")
    print(f"  Content: {doc.page_content[:100]}...")

print("\n✅ CSVLoader carga productos desde CSV en documentos LangChain")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: Cargar CSV de Productos con CSVLoader.*

1. Crear CSV de ejemplo
2. Cargar con CSVLoader

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: RecursiveCharacterTextSplitter — Dividir en Chunks

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Documento largo con descripciones de productos
texto_largo = """
Catálogo de productos 2024.

Laptop Gaming Pro: Laptop gaming de alto rendimiento con procesador Intel Core i7-12700H,
16GB de RAM DDR5, tarjeta gráfica NVIDIA RTX 3060 de 6GB, SSD NVMe de 512GB,
pantalla de 15.6 pulgadas Full HD 144Hz. Precio: $1,200. Stock: 10 unidades.

Mouse Ergonómico: Ratón inalámbrico con diseño ergonómico para uso prolongado.
6 botones programables, sensor óptico de 16000 DPI, batería recargable de 70 horas.
Incluye receptor USB y cable de carga. Precio: $45. Stock: 50 unidades.

Teclado Mecánico RGB: Teclado mecánico gaming con switches Cherry MX Brown.
Retroiluminación RGB personalizable por tecla, construcción en aluminio anodizado.
Cable USB-C desmontable trenzado. Precio: $120. Stock: 30 unidades.

Monitor 4K 27 pulgadas: Monitor IPS 4K UHD de 27 pulgadas con tasa de refresco 144Hz.
Tiempo de respuesta de 1ms, HDR400, cubrimiento 95% DCI-P3. Altavoces integrados de 5W.
Precio: $450. Stock: 15 unidades.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=40,
    separators=["\n\n", "\n", ". ", " ", ""],
)

chunks = splitter.split_text(texto_largo)
print(f"Texto dividido en {len(chunks)} chunks:")
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1} ({len(chunk)} chars):")
    print(f"  {chunk[:80]}...")

print("\n✅ RecursiveCharacterTextSplitter divide documentos en chunks manejables")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: RecursiveCharacterTextSplitter — Dividir en Chunks.*

1. Documento largo con descripciones de productos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: Chroma Vectorstore — Indexar Chunks de Productos

```python
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import os, shutil

# Cargar documentos
loader = CSVLoader('/tmp/data/productos.csv', source_column='nombre')
docs = loader.load()

# Dividir
splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
chunks = splitter.split_documents(docs)
print(f"Chunks: {len(chunks)}")

# Embeddings multilingües
embeddings = HuggingFaceEmbeddings(
    model_name='paraphrase-multilingual-MiniLM-L12-v2'
)

# Chroma vectorstore
persist_dir = '/tmp/chroma_productos'
if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_dir,
)

print(f"Vectorstore creado: {vectorstore._collection.count()} vectores")

# Búsqueda de prueba
query = "laptop para gaming"
results = vectorstore.similarity_search(query, k=2)
print(f"\nConsulta: '{query}'")
for i, doc in enumerate(results):
    print(f"  Resultado {i+1}: {doc.page_content[:80]}...")

print("\n✅ Chroma indexa los chunks y permite búsqueda semántica")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: Chroma Vectorstore — Indexar Chunks de Productos.*

1. Cargar documentos
2. Dividir
3. Embeddings multilingües
4. Chroma vectorstore
5. Búsqueda de prueba

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: RetrievalQA — Preguntar sobre Productos (stuff)

```python
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import os, shutil, warnings
warnings.filterwarnings('ignore')

# 1. Preparar vectorstore
loader = CSVLoader('/tmp/data/productos.csv', source_column='nombre')
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
chunks = splitter.split_documents(docs)
embeddings = HuggingFaceEmbeddings(model_name='paraphrase-multilingual-MiniLM-L12-v2')
persist_dir = '/tmp/chroma_qa'
if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir)
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

# 2. LLM local (modelo pequeño para demo)
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    max_new_tokens=100,
    do_sample=True,
    temperature=0.3,
)
llm = HuggingFacePipeline(pipeline=pipe)

# 3. RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False,
    verbose=False,
)

# 4. Preguntas
preguntas = [
    "¿Qué laptops tienes disponibles?",
    "¿Cuánto cuesta el teclado mecánico?",
    "¿Qué productos tienen RTX 3060?",
]

for pregunta in preguntas:
    respuesta = qa_chain.run(pregunta)
    print(f"Q: {pregunta}")
    print(f"A: {respuesta}\n")

print("✅ RetrievalQA responde preguntas sobre productos con contexto recuperado")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: RetrievalQA — Preguntar sobre Productos (stuff).*

1. 1. Preparar vectorstore
2. 2. LLM local (modelo pequeño para demo)
3. 3. RetrievalQA
4. 4. Preguntas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: RetrievalQA con return_source_documents=True

```python
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import os, shutil, warnings
warnings.filterwarnings('ignore')

# Preparar
loader = CSVLoader('/tmp/data/productos.csv', source_column='nombre')
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
chunks = splitter.split_documents(docs)
embeddings = HuggingFaceEmbeddings(model_name='paraphrase-multilingual-MiniLM-L12-v2')
persist_dir = '/tmp/chroma_sources'
if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir)
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)

pipe = pipeline("text2text-generation", model="google/flan-t5-small", max_new_tokens=80, temperature=0.3)
llm = HuggingFacePipeline(pipeline=pipe)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
)

pregunta = "¿Cuánto cuesta el monitor 4K?"
respuesta = qa_chain(pregunta)

print(f"Pregunta: {pregunta}")
print(f"Respuesta: {respuesta['result']}")
print(f"\nFuentes utilizadas ({len(respuesta['source_documents'])}):")
for i, doc in enumerate(respuesta['source_documents']):
    print(f"  {i+1}. [{doc.metadata.get('source', 'N/A')}] {doc.page_content[:80]}...")

print("\n✅ return_source_documents permite ver qué documentos se usaron")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: RetrievalQA con return_source_documents=True.*

1. Preparar

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: ConversationalRetrievalChain con BufferMemory

```python
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import os, shutil, warnings
warnings.filterwarnings('ignore')

# Preparar vectorstore
loader = CSVLoader('/tmp/data/productos.csv', source_column='nombre')
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
chunks = splitter.split_documents(docs)
embeddings = HuggingFaceEmbeddings(model_name='paraphrase-multilingual-MiniLM-L12-v2')
persist_dir = '/tmp/chroma_conv'
if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir)
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)

pipe = pipeline("text2text-generation", model="google/flan-t5-small", max_new_tokens=100, temperature=0.3)
llm = HuggingFacePipeline(pipeline=pipe)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

conversation = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
    memory=memory,
    verbose=False,
)

# Conversación
preguntas = [
    "¿Qué laptops tienes en tu catálogo?",
    "¿Cuánto cuesta la más cara?",
    "¿Tiene RTX 3060?",
]

for pregunta in preguntas:
    respuesta = conversation({"question": pregunta})
    print(f"User: {pregunta}")
    print(f"Bot: {respuesta['answer']}\n")

print("✅ ConversationalRetrievalChain mantiene el historial de la conversación")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: ConversationalRetrievalChain con BufferMemory.*

1. Preparar vectorstore
2. Conversación

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: Conversación Multi-Turno

```python
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import os, shutil, warnings
warnings.filterwarnings('ignore')

# Preparar
loader = CSVLoader('/tmp/data/productos.csv', source_column='nombre')
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
chunks = splitter.split_documents(docs)
embeddings = HuggingFaceEmbeddings(model_name='paraphrase-multilingual-MiniLM-L12-v2')
persist_dir = '/tmp/chroma_multi'
if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir)
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)

pipe = pipeline("text2text-generation", model="google/flan-t5-small", max_new_tokens=100, temperature=0.3)
llm = HuggingFacePipeline(pipeline=pipe)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
conversation = ConversationalRetrievalChain.from_llm(
    llm=llm, retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
    memory=memory,
)

# Multi-turno
conversacion = [
    "Hola, ¿qué laptops tienen?",
    "¿Cuáles son las más baratas?",
    "¿Tienen garantía?",
    "¿Puedo comprar dos?",
]

for msg in conversacion:
    resp = conversation({"question": msg})
    print(f"User: {msg}")
    print(f"Bot: {resp['answer']}\n")

print("✅ Chat multi-turno: el bot recuerda el contexto de preguntas anteriores")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: Conversación Multi-Turno.*

1. Preparar
2. Multi-turno

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: PromptTemplate Personalizado para QA de Productos

```python
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
import os, shutil, warnings
warnings.filterwarnings('ignore')

# Preparar vectorstore
loader = CSVLoader('/tmp/data/productos.csv', source_column='nombre')
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
chunks = splitter.split_documents(docs)
embeddings = HuggingFaceEmbeddings(model_name='paraphrase-multilingual-MiniLM-L12-v2')
persist_dir = '/tmp/chroma_prompt'
if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir)
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)

pipe = pipeline("text2text-generation", model="google/flan-t5-small", max_new_tokens=100, temperature=0.3)
llm = HuggingFacePipeline(pipeline=pipe)

# Prompt personalizado para ventas
template = """Eres un asistente de ventas experto en productos tecnológicos.
Usa la siguiente información para responder la pregunta del cliente.
Sé amable, profesional y específico. Si no sabes la respuesta, di que no la sabes.

Contexto:
{context}

Pregunta: {question}

Respuesta útil:"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
    chain_type_kwargs={"prompt": prompt},
)

respuesta = qa_chain.run("¿Recomiendas el teclado mecánico?")
print(f"Respuesta con prompt personalizado:\n{respuesta}")

print("\n✅ PromptTemplate personalizado mejora el tono y estilo de las respuestas")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: PromptTemplate Personalizado para QA de Productos.*

1. Preparar vectorstore
2. Prompt personalizado para ventas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: chain_type='map_reduce' para Múltiples Documentos

```python
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import os, shutil, warnings
warnings.filterwarnings('ignore')

# Catálogo más grande
import pandas as pd
df = pd.DataFrame({
    'nombre': [f'Producto {i}' for i in range(20)],
    'descripcion': [f'Descripción detallada del producto {i} con características técnicas' for i in range(20)],
    'precio': [(i+1) * 50 for i in range(20)],
})
df.to_csv('/tmp/data/catalogo_grande.csv', index=False)

loader = CSVLoader('/tmp/data/catalogo_grande.csv', source_column='nombre')
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = splitter.split_documents(docs)
embeddings = HuggingFaceEmbeddings(model_name='paraphrase-multilingual-MiniLM-L12-v2')
persist_dir = '/tmp/chroma_mapreduce'
if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir)
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)

pipe = pipeline("text2text-generation", model="google/flan-t5-small", max_new_tokens=80, temperature=0.3)
llm = HuggingFacePipeline(pipeline=pipe)

# map_reduce: procesa cada documento individualmente y luego resume
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="map_reduce",
    retriever=vectorstore.as_retriever(search_kwargs={'k': 5}),
)

respuesta = qa_chain.run("¿Qué productos tienen precio menor a $200?")
print(f"map_reduce response: {respuesta}")

print("\n✅ map_reduce procesa cada documento por separado y combina respuestas")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: chain_type='map_reduce' para Múltiples Documentos.*

1. Catálogo más grande
2. map_reduce: procesa cada documento individualmente y luego resume

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: chain_type='refine' — Refinar Respuesta Iterativamente

```python
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import os, shutil, warnings
warnings.filterwarnings('ignore')

loader = CSVLoader('/tmp/data/catalogo_grande.csv', source_column='nombre')
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = splitter.split_documents(docs)
embeddings = HuggingFaceEmbeddings(model_name='paraphrase-multilingual-MiniLM-L12-v2')
persist_dir = '/tmp/chroma_refine'
if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir)
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)

pipe = pipeline("text2text-generation", model="google/flan-t5-small", max_new_tokens=80, temperature=0.3)
llm = HuggingFacePipeline(pipeline=pipe)

# refine: refina respuesta iterativamente con cada documento
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="refine",
    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
)

respuesta = qa_chain.run("¿Cuál es el producto más caro del catálogo?")
print(f"refine response: {respuesta}")

print("\n✅ refine mejora la respuesta iterativamente con cada documento recuperado")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: chain_type='refine' — Refinar Respuesta Iterativamente.*

1. refine: refina respuesta iterativamente con cada documento

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: FAISS como Vectorstore (Alternativa a Chroma)

```python
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

loader = CSVLoader('/tmp/data/productos.csv', source_column='nombre')
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name='paraphrase-multilingual-MiniLM-L12-v2')

# FAISS en lugar de Chroma
vectorstore = FAISS.from_documents(chunks, embeddings)

pipe = pipeline("text2text-generation", model="google/flan-t5-small", max_new_tokens=80, temperature=0.3)
llm = HuggingFacePipeline(pipeline=pipe)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
)

respuesta = qa_chain.run("¿Qué monitores tienes?")
print(f"Respuesta (FAISS): {respuesta}")

# Guardar y cargar FAISS
vectorstore.save_local("/tmp/faiss_productos")
vectorstore_cargado = FAISS.load_local("/tmp/faiss_productos", embeddings)
print(f"FAISS guardado y cargado: {vectorstore_cargado.index.ntotal} vectores")

print("\n✅ FAISS es una alternativa más rápida y eficiente que Chroma")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: FAISS como Vectorstore (Alternativa a Chroma).*

1. FAISS en lugar de Chroma
2. Guardar y cargar FAISS

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: ChatHuggingFace como LLM

```python
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.llms.huggingface_hub import HuggingFaceHub
from transformers import pipeline
import os, shutil, warnings
warnings.filterwarnings('ignore')

# Pipeline con modelo de chat (instruction-tuned)
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    max_new_tokens=150,
    temperature=0.3,
    do_sample=True,
)
llm = HuggingFacePipeline(pipeline=pipe)

# RAG completo
loader = CSVLoader('/tmp/data/productos.csv', source_column='nombre')
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
chunks = splitter.split_documents(docs)
embeddings = HuggingFaceEmbeddings(model_name='paraphrase-multilingual-MiniLM-L12-v2')
persist_dir = '/tmp/chroma_chatllm'
if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir)
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
)

preguntas = [
    "¿Qué productos tienen buen precio?",
    "¿Cuál es el producto más barato?",
]
for p in preguntas:
    print(f"Q: {p}")
    print(f"A: {qa.run(p)}\n")

print("✅ HuggingFacePipeline + modelo instruction-tuned para respuestas coherentes")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: ChatHuggingFace como LLM.*

1. Pipeline con modelo de chat (instruction-tuned)
2. RAG completo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: ConversationSummaryMemory — Resumir Conversación

```python
from langchain.memory import ConversationSummaryMemory
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationChain
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

pipe = pipeline("text2text-generation", model="google/flan-t5-small", max_new_tokens=100, temperature=0.3)
llm = HuggingFacePipeline(pipeline=pipe)

# Memoria de resumen
memory = ConversationSummaryMemory(llm=llm, max_token_limit=100)

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False,
)

conversacion = [
    "Hola, estoy buscando una laptop para gaming",
    "¿Cuáles tienen RTX 3060?",
    "¿Cuál es la más barata con esa tarjeta?",
]

for msg in conversacion:
    resp = conversation.predict(input=msg)
    print(f"User: {msg}")
    print(f"Bot: {resp[:80]}...\n")

# Ver resumen
print(f"Resumen de la conversación:\n{memory.buffer}")

print("\n✅ ConversationSummaryMemory mantiene un resumen comprimido de la conversación")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: ConversationSummaryMemory — Resumir Conversación.*

1. Memoria de resumen
2. Ver resumen

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: ConversationBufferWindowMemory(k=3)

```python
from langchain.memory import ConversationBufferWindowMemory
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationChain
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

pipe = pipeline("text2text-generation", model="google/flan-t5-small", max_new_tokens=80, temperature=0.3)
llm = HuggingFacePipeline(pipeline=pipe)

# Ventana de 3 intercambios
memory = ConversationBufferWindowMemory(k=3)

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False,
)

mensajes = [
    "Hola", "¿Tienes laptops?", "¿Cuánto cuestan?",
    "¿Y monitores?", "¿Cuál es el más barato?", "Gracias",
]

for msg in mensajes:
    resp = conversation.predict(input=msg)
    print(f"User: {msg} -> Bot: {resp[:60]}...")

print(f"\nMemoria actual ({len(memory.buffer)} caracteres):")
print(memory.buffer[:200])

print("\n✅ ConversationBufferWindowMemory mantiene solo los últimos K intercambios")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: ConversationBufferWindowMemory(k=3).*

1. Ventana de 3 intercambios

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: Chatbot con Historial — Preguntas y Respuestas Previas

```python
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import os, shutil, warnings
warnings.filterwarnings('ignore')

# Preparar
loader = CSVLoader('/tmp/data/productos.csv', source_column='nombre')
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
chunks = splitter.split_documents(docs)
embeddings = HuggingFaceEmbeddings(model_name='paraphrase-multilingual-MiniLM-L12-v2')
persist_dir = '/tmp/chroma_historial'
if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir)
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)

pipe = pipeline("text2text-generation", model="google/flan-t5-small", max_new_tokens=100, temperature=0.3)
llm = HuggingFacePipeline(pipeline=pipe)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chatbot = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
    memory=memory,
)

# Simular historial
historial_inicial = [
    "Necesito una laptop para gaming",
    "¿Tienen con RTX 3060?",
]

# Construir historial manualmente
from langchain.schema import HumanMessage, AIMessage
memory.chat_memory.add_user_message("Necesito una laptop para gaming")
memory.chat_memory.add_ai_message("Sí, tenemos la Laptop Gaming Pro con RTX 3060")

# Nueva pregunta usando historial
resp = chatbot({"question": "¿Cuánto cuesta?"})
print(f"User: ¿Cuánto cuesta?")
print(f"Bot: {resp['answer']}")

resp = chatbot({"question": "¿Qué más me recomiendas?"})
print(f"\nUser: ¿Qué más me recomiendas?")
print(f"Bot: {resp['answer']}")

print("\n✅ Chatbot con historial: responde considerando contexto de preguntas previas")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: Chatbot con Historial — Preguntas y Respuestas Previas.*

1. Preparar
2. Simular historial
3. Construir historial manualmente
4. Nueva pregunta usando historial

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 16: System Prompt — "Eres un asistente de ventas..."

```python
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
import os, shutil, warnings
warnings.filterwarnings('ignore')

# System prompt para asistente de ventas
system_prompt = """Eres un asistente de ventas profesional y amable para una tienda de tecnología.
Tu objetivo es ayudar al cliente a encontrar el producto perfecto.
Siempre sé:
- Educado y profesional
- Específico con precios y características
- Proactivo: sugiere productos relacionados
- Honesto: si no tienes un producto, dilo amablemente

Contexto de productos:
{context}

Cliente: {question}

Asistente de ventas:"""

prompt = PromptTemplate(
    template=system_prompt,
    input_variables=["context", "question"],
)

loader = CSVLoader('/tmp/data/productos.csv', source_column='nombre')
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
chunks = splitter.split_documents(docs)
embeddings = HuggingFaceEmbeddings(model_name='paraphrase-multilingual-MiniLM-L12-v2')
persist_dir = '/tmp/chroma_system'
if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir)
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)

pipe = pipeline("text2text-generation", model="google/flan-t5-small", max_new_tokens=120, temperature=0.3)
llm = HuggingFacePipeline(pipeline=pipe)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
    chain_type_kwargs={"prompt": prompt},
)

preguntas = [
    "Busco un teclado para programar",
    "¿Tienen algo con RTX 3060?",
    "Necesito un monitor para edición de video",
]

for p in preguntas:
    print(f"Cliente: {p}")
    print(f"Asistente: {qa.run(p)}\n")

print("✅ System prompt convierte el RAG en un asistente de ventas profesional")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: System Prompt — "Eres un asistente de ventas...".*

1. System prompt para asistente de ventas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 17: Evaluar Respuestas — Relevancia y Precisión

```python
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from sklearn.metrics import accuracy_score
import os, shutil, warnings
warnings.filterwarnings('ignore')

# Preparar sistema
loader = CSVLoader('/tmp/data/productos.csv', source_column='nombre')
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
chunks = splitter.split_documents(docs)
embeddings = HuggingFaceEmbeddings(model_name='paraphrase-multilingual-MiniLM-L12-v2')
persist_dir = '/tmp/chroma_eval'
if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir)
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)

pipe = pipeline("text2text-generation", model="google/flan-t5-small", max_new_tokens=80, temperature=0.3)
llm = HuggingFacePipeline(pipeline=pipe)
qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
)

# Evaluación manual
pares_eval = [
    ("¿Cuánto cuesta la laptop?", "1200", ["1200", "precio", "laptop"]),
    ("¿Hay teclados mecánicos?", "sí", ["sí", "teclado", "mecánico"]),
    ("¿Qué resolución tiene el monitor?", "4K", ["4K", "3840", "2160"]),
]

print("=== EVALUACIÓN DE RESPUESTAS ===\n")
puntajes = []
for pregunta, esperado, keywords in pares_eval:
    respuesta = qa(pregunta)
    texto = respuesta['result'].lower()
    fuentes = respuesta['source_documents']

    # Relevancia: ¿contiene palabras clave?
    relevante = any(kw.lower() in texto for kw in keywords)
    puntajes.append(1.0 if relevante else 0.0)

    print(f"P: {pregunta}")
    print(f"R: {respuesta['result'][:80]}...")
    print(f"  Esperado: {esperado}")
    print(f"  Relevante: {'✅' if relevante else '❌'}")
    print(f"  Fuentes: {len(fuentes)} documentos\n")

print(f"Precisión (relevancia): {sum(puntajes)/len(puntajes):.0%}")

print("\n✅ Evaluación manual de relevancia y precisión del RAG")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: Evaluar Respuestas — Relevancia y Precisión.*

1. Preparar sistema
2. Evaluación manual
3. Relevancia: ¿contiene palabras clave?

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 18: Integrador — Chatbot RAG Completo para Ventas

```python
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
import pandas as pd
import os, shutil, time, warnings
warnings.filterwarnings('ignore')

# ===== 1. CREAR CATÁLOGO COMPLETO =====
catalogo_completo = pd.DataFrame({
    "nombre": [
        "Laptop Gaming Pro", "Laptop UltraBook", "Mouse Ergonómico",
        "Teclado Mecánico RGB", "Monitor 4K 27\"", "Monitor Curvo 32\"",
        "Silla Ergonómica", "Auriculares Bluetooth", "Hub USB-C 7en1",
        "SSD 1TB NVMe", "Impresora Láser Color", "Router WiFi 6",
        "Webcam 4K Pro", "Cable HDMI 2.1", "Micrófono USB",
        "Base laptop ajustable", "UPS 1500VA", "Disco Duro Externo 4TB",
    ],
    "categoria": [
        "laptop", "laptop", "periférico", "periférico", "monitor", "monitor",
        "mueble", "audio", "accesorio", "almacenamiento", "impresora",
        "red", "periférico", "cable", "audio", "accesorio", "accesorio", "almacenamiento",
    ],
    "precio": [1200, 899, 45, 120, 450, 550, 350, 80, 35, 150, 280, 90, 130, 25, 60, 40, 200, 110],
    "stock": [10, 15, 50, 30, 15, 8, 20, 40, 60, 25, 12, 35, 18, 100, 22, 45, 7, 20],
    "descripcion": [
        "Laptop gaming RTX 3060 i7-12700H 16GB RAM SSD 512GB 15.6\" 144Hz",
        "Laptop ultradelgada i5-1240P 8GB RAM SSD 256GB 14\" FHD",
        "Mouse inalámbrico ergonómico 6 botones 16000 DPI batería 70h",
        "Teclado mecánico Cherry MX Brown RGB aluminio USB-C",
        "Monitor IPS 4K 27\" 144Hz 1ms HDR400 altavoces integrados",
        "Monitor curvo 32\" QHD 165Hz 1ms VA ultra-wide",
        "Silla oficina ergonómica ajustable lumbar reposabrazos 3D",
        "Auriculares BT 5.0 cancelación ruido activa 30h plegables",
        "Hub USB C HDMI 4K 2xUSB3.0 SD microSD PD 100W",
        "SSD NVMe M.2 1TB lectura 3500MB/s escritura 3000MB/s",
        "Impresora láser color WiFi dúplex 25ppm bandeja 250 hojas",
        "Router WiFi 6 AX3000 doble banda 4 antenas gigabit Mesh",
        "Webcam 4K autoenfoque micrófono estéreo corrección luz",
        "Cable HDMI 2.1 3m 48Gbps 8K@60Hz eARC compatible",
        "Micrófono USB condensador cardioide grabación studio",
        "Base laptop aluminio ajustable altura ventilación",
        "UPS 1500VA 900W 8 tomas regulación voltaje",
        "Disco duro externo portátil 4TB USB 3.0 2.5 pulgadas",
    ],
})
catalogo_completo.to_csv('/tmp/data/catalogo_completo.csv', index=False)

# ===== 2. INDEXAR =====
print("Indexando catálogo...")
loader = CSVLoader('/tmp/data/catalogo_completo.csv', source_column='nombre')
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
chunks = splitter.split_documents(docs)
embeddings = HuggingFaceEmbeddings(model_name='paraphrase-multilingual-MiniLM-L12-v2')

persist_dir = '/tmp/chroma_rag_final'
if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir)
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
print(f"Indexados {len(chunks)} chunks de {len(docs)} productos")

# ===== 3. LLM =====
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    max_new_tokens=150,
    do_sample=True,
    temperature=0.3,
    top_p=0.95,
)
llm = HuggingFacePipeline(pipeline=pipe)

# ===== 4. PROMPT DE VENTAS =====
prompt_ventas = """Eres un asistente de ventas experto en tecnología.
Información de productos disponibles:
{context}

Historial de la conversación:
{chat_history}

Cliente: {question}
Asistente de ventas (sé amable, específico y sugiere productos relacionados):"""

prompt = PromptTemplate(
    template=prompt_ventas,
    input_variables=["context", "chat_history", "question"],
)

# ===== 5. MEMORIA =====
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    k=4,
    output_key="answer",
    input_key="question",
)

# ===== 6. CHAIN =====
chatbot = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={'k': 4}),
    memory=memory,
    chain_type="stuff",
    verbose=False,
    combine_docs_chain_kwargs={"prompt": prompt},
)

# ===== 7. DEMO =====
print("\n" + "="*60)
print("🤖 CHATBOT DE VENTAS - RAG COMPLETO")
print("="*60)

conversacion = [
    "Hola, estoy buscando una laptop para gaming",
    "¿Cuáles tienen RTX 3060 y cuánto cuestan?",
    "¿Tienen algún monitor que recomiendes con esa laptop?",
    "¿Hay algún descuento por comprar los dos?",
    "Gracias, ¿cómo puedo comprar?",
]

for msg in conversacion:
    print(f"\n🧑 Cliente: {msg}")
    start = time.time()
    respuesta = chatbot({"question": msg})
    t = time.time() - start
    print(f"🤖 Asistente: {respuesta['answer']}")
    print(f"   ({t*1000:.0f}ms)")

# ===== 8. EVALUACIÓN =====
print("\n" + "="*60)
print("📊 EVALUACIÓN")
print("="*60)
consultas_test = [
    "¿Qué laptops tienes?",
    "¿Cuánto cuesta el teclado?",
    "Periféricos baratos",
    "Monitores 4K",
    "Almacenamiento externo",
]
tiempos = []
for c in consultas_test:
    start = time.time()
    chatbot({"question": c})
    tiempos.append(time.time() - start)
print(f"Tiempo promedio: {sum(tiempos)/len(tiempos)*1000:.0f}ms")
print(f"Total productos en catálogo: {len(catalogo_completo)}")
print(f"Total chunks indexados: {len(chunks)}")
print(f"Modelo LLM: google/flan-t5-small")

# ===== 9. GUARDAR =====
vectorstore.persist()
print("\nVectorstore persistido en", persist_dir)
print("\n✅ INTEGRADOR COMPLETO: Chatbot RAG para ventas con historial, catálogo completo y prompt personalizado")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador — Chatbot RAG Completo para Ventas.*

1. ===== 1. CREAR CATÁLOGO COMPLETO =====
2. ===== 2. INDEXAR =====
3. ===== 3. LLM =====
4. ===== 4. PROMPT DE VENTAS =====
5. ===== 5. MEMORIA =====
6. ===== 6. CHAIN =====
7. ===== 7. DEMO =====
8. ===== 8. EVALUACIÓN =====

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 3. Ejercicios Propuestos

1. **RAG con PDF**: Usa PDFLoader para cargar un catálogo de productos en PDF. Divide en chunks y construye un RAG que responda preguntas sobre el catálogo.

2. **Comparación de chain types**: Para un dataset de 20 productos, compara stuff vs map_reduce vs refine en 5 preguntas. Mide calidad de respuesta y tiempo de ejecución.

3. **Memoria persistente**: Implementa un chatbot cuya memoria se guarde en disco (SQLite o JSON) y pueda restaurarse al reiniciar la aplicación.

4. **RAG multilingüe**: Carga un catálogo en español y haz preguntas en inglés, francés y alemán. ¿El sistema responde correctamente en todos los idiomas?

5. **Filtros en RAG**: Modifica el retriever para filtrar por categoría y rango de precio antes de pasar al LLM. Ejemplo: "laptops entre $500 y $1000".

6. **Evaluación sistemática**: Crea 20 pares (pregunta, respuesta_esperada) y evalúa el RAG con métricas: BLEU, ROUGE, precisión de entidades (precios, nombres). Reporta resultados.

7. **RAG con múltiples fuentes**: Carga 3 CSVs diferentes (productos, reseñas, preguntas frecuentes). Usa DirectoryLoader. El chatbot debe responder combinando información de todas las fuentes.

8. **Proyecto final**: Construye una aplicación completa con:
   - Carga de catálogo desde CSV/PDF
   - Indexación en Chroma o FAISS
   - Chatbot RAG con ConversationalRetrievalChain
   - Memoria con ventana de 5 intercambios
   - Prompt de asistente de ventas personalizado
   - API REST con FastAPI (endpoints: /chat, /reset, /productos)
   - Interfaz web simple con Gradio o Streamlit
   - Evaluación de calidad de respuestas

---

## 4. Resumen

| Componente | Propósito | En RAG de Ventas |
|---|---|---|
| Document Loaders | Cargar datos | CSVLoader para catálogos |
| Text Splitters | Dividir en chunks | Chunks de 200-500 caracteres |
| Embeddings | Vectorizar texto | Multilingüe para catálogos |
| Vector Stores (Chroma/FAISS) | Indexar y buscar | Recuperar productos relevantes |
| RetrievalQA | QA simple | Preguntas directas |
| ConversationalRetrievalChain | QA con historial | Chatbot multi-turno |
| Memory (Buffer/Window/Summary) | Recordar contexto | Seguimiento de conversación |
| PromptTemplate | Controlar estilo | Asistente de ventas profesional |
| chain_type (stuff/map_reduce/refine) | Estrategia de generación | Según cantidad de docs |

**Conclusión**: RAG (Retrieval Augmented Generation) es la arquitectura ideal para chatbots de ventas B2B. Combina la precisión de la búsqueda semántica (recuperando productos y precios relevantes) con la fluidez de un LLM para generar respuestas naturales y contextuales. LangChain + HuggingFace + Chroma/FAISS proporcionan el stack completo para construir asistentes de ventas inteligentes, escalables y fáciles de mantener.
