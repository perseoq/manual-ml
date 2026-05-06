# I25: NLP con spaCy — Procesamiento Avanzado

## Introducción Teórica

**spaCy** es una biblioteca moderna de NLP diseñada para producción. A diferencia de NLTK (académico), spaCy prioriza velocidad, eficiencia y pipelines integrados. Utiliza modelos estadísticos pre-entrenados que realizan múltiples tareas en un solo paso.

### Arquitectura del Pipeline

Cuando ejecutas `nlp(texto)`, spaCy ejecuta secuencialmente:

1. **Tokenizer**: divide texto en tokens (reglas específicas por idioma)
2. **Tagger**: asigna POS tags (part-of-speech)
3. **Parser**: construye árbol de dependencias (dependency parsing)
4. **NER**: reconoce entidades nombradas (named entity recognition)
5. **Lemmatizer**: reduce a forma base (lematización)
6. **Attribute Ruler**: reglas personalizadas para atributos

Cada componente puede habilitarse/deshabilitarse individualmente.

### Modelos Disponibles para Español

| Modelo | Tamaño | Incluye |
|---|---|---|
| `es_core_news_sm` | ~12 MB | Básico: tagger, parser, NER, lematizer |
| `es_core_news_md` | ~40 MB | + word vectors de 50 dimensiones |
| `es_core_news_lg` | ~500 MB | + word vectors de 300 dimensiones + NER más preciso |

### Conceptos Clave

- **Token**: unidad mínima. Atributos: `.text`, `.lemma_`, `.pos_`, `.tag_`, `.dep_`, `.shape_`, `.is_stop`, `.is_punct`, `.is_alpha`, `.like_num`
- **Doc**: contenedor de tokens. Soporta slicing, iteración y acceso por índice.
- **Span**: slice de un Doc (ej. `doc[2:5]`). Atributos: `.text`, `.label_`, `.root`, `.lefts`, `.rights`
- **Dependency Parsing**: relaciones gramaticales entre tokens. `nsubj` (sujeto), `dobj` (objeto directo), `amod` (adjetivo modificador), `nmod` (modificador nominal)
- **NER**: etiquetas como `ORG`, `MISC`, `LOC`, `PER`, `DATE`, `MONEY`, `PRODUCT`
- **Noun Chunks**: frases nominales base. Útil para extraer nombres de productos completos.
- **Matcher**: búsqueda basada en patrones de tokens (diccionarios con atributos).
- **PhraseMatcher**: búsqueda de frases exactas (más rápido que Matcher para términos fijos).
- **EntityRuler**: agrega entidades personalizadas al pipeline NER.
- **Displacy**: visualizador integrado. `displacy.render(doc, style='dep')` para dependencias, `style='ent'` para entidades.
- **Similarity**: similitud coseno entre vectores de palabras. Requiere modelo con vectores (md o lg).

### Instalación

```bash
pip install spacy
python -m spacy download es_core_news_sm
python -m spacy download es_core_news_md  # opcional, para vectores
python -m spacy download en_core_web_sm   # opcional, para ejemplos en inglés
```

---

## Ejemplos Prácticos

### 1. Cargar modelo y procesar descripción de producto

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Instalación.*

1. Ejemplos Prácticos
2. 1. Cargar modelo y procesar descripción de producto

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
import spacy

nlp = spacy.load('es_core_news_sm')

descripcion = "Auriculares Sony WH-1000XM5 con cancelación de ruido activa y 30 horas de batería"
doc = nlp(descripcion)

print(f"Texto: {descripcion}")
print(f"Tokens: {[token.text for token in doc]}")
print(f"Entidades: {[(ent.text, ent.label_) for ent in doc.ents]}")
print(f"Lemas: {[token.lemma_ for token in doc]}")
```

**Aplicación**: punto de entrada para cualquier tarea NLP. El pipeline completo se ejecuta en una sola llamada.

---

### 2. Token: text, lemma_, pos_, tag_, dep_, shape_, is_stop, is_punct

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 2. Token: text, lemma_, pos_, tag_, dep_, shape_, is_stop, is_punct

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
import spacy

nlp = spacy.load('es_core_news_sm')
texto = "Este televisor 4K tiene una pantalla OLED de 55 pulgadas con HDR10."

doc = nlp(texto)

print(f"{'Token':15s} {'Lemma':15s} {'POS':8s} {'Tag':8s} {'Dep':12s} {'Shape':15s} {'Stop':5s} {'Punct':5s}")
print("-" * 83)
for token in doc:
    print(f"{token.text:15s} {token.lemma_:15s} {token.pos_:8s} {token.tag_:8s} {token.dep_:12s} {token.shape_:15s} {str(token.is_stop):5s} {str(token.is_punct):5s}")
```

**Aplicación**: extraer todas las características lingüísticas de cada token para usarlas como features en modelos ML.

---

### 3. Lematización en español (leer, leyó → leer)

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 3. Lematización en español (leer, leyó → leer)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
import spacy

nlp = spacy.load('es_core_news_sm')

verbos = ["leyó", "leerá", "leyendo", "leído", "leen", "leamos", "compró", "comprando", "comprado", "vendió", "vender", "venderán"]

print(f"{'Palabra':15s} {'Lemma':15s}")
print("-" * 30)
for v in verbos:
    doc = nlp(v)
    print(f"{v:15s} {doc[0].lemma_:15s}")

# Aplicación en descripción
texto = "Los clientes compraron el producto, lo leyeron y lo recomendaron."
doc = nlp(texto)
lemas = [token.lemma_ for token in doc if not token.is_punct]
print(f"\nOriginal: {texto}")
print(f"Lemas: {' '.join(lemas)}")
```

**Aplicación**: normalizar reseñas y descripciones reduciendo verbos a infinitivo para matching semántico.

---

### 4. POS tagging: identificar sustantivos, verbos, adjetivos

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 4. POS tagging: identificar sustantivos, verbos, adjetivos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
import spacy

nlp = spacy.load('es_core_news_sm')

desc = "Cómoda silla ergonómica ajustable con soporte lumbar y reposabrazos acolchados"
doc = nlp(desc)

sustantivos = [token.text for token in doc if token.pos_ == 'NOUN']
verbos = [token.text for token in doc if token.pos_ == 'VERB']
adjetivos = [token.text for token in doc if token.pos_ == 'ADJ']
adverbios = [token.text for token in doc if token.pos_ == 'ADV']

print(f"Descripción: {desc}")
print(f"Sustantivos (productos/partes): {sustantivos}")
print(f"Adjetivos (atributos): {adjetivos}")
print(f"Verbos (acciones): {verbos}")
print(f"Adverbios: {adverbios}")
```

**Aplicación**: extraer automáticamente nombres de productos (sustantivos) y sus atributos (adjetivos) de descripciones.

---

### 5. Dependency parsing: relaciones gramaticales

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 5. Dependency parsing: relaciones gramaticales

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
import spacy

nlp = spacy.load('es_core_news_sm')

texto = "El cliente compró tres laptops gaming en la tienda online"

doc = nlp(texto)

print(f"{'Token':12s} {'Dep':12s} {'Head':12s} {'Children':30s}")
print("-" * 66)
for token in doc:
    children = [child.text for child in token.children]
    print(f"{token.text:12s} {token.dep_:12s} {token.head.text:12s} {str(children):30s}")

# Visualizar árbol
print("\nÁrbol de dependencias:")
for token in doc:
    if token.dep_ == 'ROOT':
        print(f"  RAÍZ: {token.text}")
        for child in token.subtree:
            indent = "    " * (len(list(child.ancestors)) - 1)
            print(f"{indent}-> {child.text} ({child.dep_})")
```

**Aplicación**: entender la estructura gramatical permite extraer relaciones como "producto + atributo" (ej. "laptop" ← "gaming").

---

### 6. Named Entity Recognition: extraer marcas, precios, lugares

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 6. Named Entity Recognition: extraer marcas, precios, lugares

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
import spacy

nlp = spacy.load('es_core_news_sm')

texto = """
Apple lanzó el iPhone 15 Pro en la Apple Store de Nueva York por $1,199 USD.
Samsung presentó el Galaxy S24 en el evento Unpacked en San José.
La Sony PlayStation 5 cuesta 499 euros en la tienda de Madrid.
"""

doc = nlp(texto)

print(f"{'Entidad':20s} {'Label':12s} {'Descripción':30s}")
print("-" * 62)
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_:12s} {spacy.explain(ent.label_):30s}")
```

**Aplicación**: extraer marcas (ORG), precios (MONEY), ubicaciones (LOC/GPE) de descripciones de productos automáticamente.

---

### 7. noun_chunks: Extraer frases nominales

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 7. noun_chunks: Extraer frases nominales

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
import spacy

nlp = spacy.load('es_core_news_sm')

descripcion = """
El nuevo monitor gaming curvo de 27 pulgadas con panel OLED y tasa de refresco de 240Hz
incluye un soporte ajustable en altura y puertos HDMI 2.1 y DisplayPort.
"""

doc = nlp(descripcion)

print("Frases nominales (noun chunks) extraídas:")
for chunk in doc.noun_chunks:
    print(f"  - {chunk.text} (raíz: {chunk.root.text}, POS raíz: {chunk.root.pos_})")
```

**Aplicación**: extraer nombres completos de productos y sus modificadores. "El nuevo monitor gaming curvo" identifica el producto completo.

---

### 8. spacy.explain: qué significa cada etiqueta POS

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 8. spacy.explain: qué significa cada etiqueta POS

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
import spacy

nlp = spacy.load('es_core_news_sm')

texto = "El nuevo producto funciona perfectamente y tiene excelente calidad."
doc = nlp(texto)

print("Explicación de etiquetas:")
etiquetas_vistas = set()
for token in doc:
    if token.pos_ not in etiquetas_vistas:
        etiquetas_vistas.add(token.pos_)
        print(f"  {token.pos_}: {spacy.explain(token.pos_)}")
    if token.tag_ not in etiquetas_vistas:
        etiquetas_vistas.add(token.tag_)
        print(f"  {token.tag_}: {spacy.explain(token.tag_)}")

print("\nTabla completa de etiquetas POS comunes:")
etiquetas_comunes = ['NOUN', 'VERB', 'ADJ', 'ADV', 'ADP', 'DET', 'PRON', 'PROPN', 'NUM', 'CCONJ', 'SCONJ', 'INTJ']
for et in etiquetas_comunes:
    print(f"  {et:8s} → {spacy.explain(et)}")
```

**Aplicación**: entender qué significa cada etiqueta es esencial para interpretar correctamente los resultados del POS tagging.

---

### 9. Similarity: similitud semántica entre dos descripciones

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 9. Similarity: similitud semántica entre dos descripciones

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
import spacy

# Necesitamos el modelo mediano para vectores
nlp = spacy.load('es_core_news_md')

desc1 = "Auriculares inalámbricos con cancelación de ruido"
desc2 = "Cascos Bluetooth con aislamiento de sonido ambiental"
desc3 = "Teclado mecánico retroiluminado para gaming"

doc1 = nlp(desc1)
doc2 = nlp(desc2)
doc3 = nlp(desc3)

print(f"Descripción 1: {desc1}")
print(f"Descripción 2: {desc2}")
print(f"Descripción 3: {desc3}")
print(f"\nSimilitud 1 vs 2: {doc1.similarity(doc2):.4f} (alta, mismo producto)")
print(f"Similitud 1 vs 3: {doc1.similarity(doc3):.4f} (baja, productos diferentes)")
print(f"Similitud 2 vs 3: {doc2.similarity(doc3):.4f}")
```

**Aplicación**: recomendar productos similares, detectar duplicados en catálogos, agrupar descripciones por similitud semántica.

---

### 10. displacy.render: Visualizar dependencias gráficamente

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 10. displacy.render: Visualizar dependencias gráficamente

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
import spacy
from spacy import displacy

nlp = spacy.load('es_core_news_sm')

texto = "El cliente devolvió el monitor defectuoso ayer"
doc = nlp(texto)

# Visualizar dependencias
displacy.render(doc, style='dep', jupyter=False, options={'compact': True})

# Visualizar entidades
texto2 = "Apple vende MacBooks en Madrid por 2000 euros"
doc2 = nlp(texto2)
displacy.render(doc2, style='ent', jupyter=False)
```

**Aplicación**: generar visualizaciones de dependencias gramaticales y entidades para reportes y análisis cualitativos.

---

### 11. Matcher: Buscar patrones "producto + precio"

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 11. Matcher: Buscar patrones "producto + precio"

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
import spacy
from spacy.matcher import Matcher

nlp = spacy.load('es_core_news_sm')
matcher = Matcher(nlp.vocab)

# Patrón: sustantivo + verbo + MONEY (ej. "cuesta 500 euros")
patron_precio = [
    {'POS': 'NOUN'},  # producto
    {'LEMMA': {'IN': ['costar', 'valer', 'precio', 'cuesta']}},
    {'ENT_TYPE': 'MONEY'}  # precio
]
matcher.add('PRODUCTO_PRECIO', [patron_precio])

# Patrón: verbo + DET? + NUM? + MONEY (ej. "compró por 100 euros")
patron_compro = [
    {'LEMMA': {'IN': ['comprar', 'adquirir', 'pagar']}},
    {'OP': '*'},  # opcionales entre verbo y precio
    {'ENT_TYPE': 'MONEY', 'OP': '+'}
]
matcher.add('COMPRA_PRECIO', [patron_compro])

texto = "El monitor cuesta 599 euros. El cliente compró la laptop por 1200 dólares."
doc = nlp(texto)

print("Patrones encontrados:")
for match_id, start, end in matcher(doc):
    span = doc[start:end]
    print(f"  {span.text} (match: {nlp.vocab[match_id].text})")
```

**Aplicación**: extraer pares (producto, precio) de textos libres para enriquecer catálogos automáticamente.

---

### 12. PhraseMatcher: Buscar frases específicas en catálogo

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 12. PhraseMatcher: Buscar frases específicas en catálogo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
import spacy
from spacy.matcher import PhraseMatcher

nlp = spacy.load('es_core_news_sm')
matcher = PhraseMatcher(nlp.vocab, attr='LOWER')

terminos_busqueda = ["cancelación de ruido", "pantalla oled", "carga rápida", "asistente de voz", "inteligencia artificial"]
patrones = [nlp.make_doc(termino) for termino in terminos_busqueda]
matcher.add("CARACTERISTICAS", patrones)

catalogo = [
    "Auriculares con cancelación de ruido activa y asistente de voz integrado",
    "Smart TV con pantalla OLED y resolución 4K HDR",
    "Smartphone con carga rápida de 65W y batería de 5000mAh",
    "Altavoz inteligente con asistente de voz e inteligencia artificial"
]

print("Búsqueda de características en catálogo:")
for producto in catalogo:
    doc = nlp(producto)
    matches = matcher(doc)
    if matches:
        caracteristicas = [doc[start:end].text for _, start, end in matches]
        print(f"  Producto: {producto}")
        print(f"  Características encontradas: {caracteristicas}\n")
```

**Aplicación**: etiquetar productos automáticamente según características mencionadas en sus descripciones.

---

### 13. EntityRuler: Agregar entidades personalizadas

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 13. EntityRuler: Agregar entidades personalizadas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
import spacy
from spacy.pipeline import EntityRuler
from spacy import displacy

nlp = spacy.load('es_core_news_sm')

# Crear ruler y agregarlo al pipeline
ruler = nlp.add_pipe('entity_ruler', before='ner')

# Patrones para marcas y modelos de productos
patrones = [
    {"label": "MARCA", "pattern": [{"LOWER": "sony"}]},
    {"label": "MARCA", "pattern": [{"LOWER": "samsung"}, {"LOWER": "electronics", "OP": "?"}]},
    {"label": "MARCA", "pattern": [{"LOWER": "lg"}]},
    {"label": "PRODUCTO", "pattern": [{"LOWER": "wh-1000xm5"}, {"LOWER": "de", "OP": "?"}]},
    {"label": "PRODUCTO", "pattern": [{"LOWER": "galaxy"}, {"LOWER": "s24"}]},
    {"label": "PRODUCTO", "pattern": [{"LOWER": "ultragear"}]},
    # Patrones basados en POS
    {"label": "PRECIO", "pattern": [{"ENT_TYPE": "MONEY"}]},
]

ruler.add_patterns(patrones)

# Procesar texto
texto = "El Sony WH-1000XM5 cuesta 349 euros. El Samsung Galaxy S24 vale 899 dólares. El LG UltraGear 27GN950 está en oferta."
doc = nlp(texto)

print("Entidades (incluyendo personalizadas):")
for ent in doc.ents:
    print(f"  {ent.text:30s} → {ent.label_}")
```

**Aplicación**: agregar entidades específicas del dominio (marcas, modelos, SKUs) que el NER pre-entrenado no reconoce.

---

### 14. sentencizer: Segmentación en oraciones

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 14. sentencizer: Segmentación en oraciones

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
import spacy

# Usar solo el sentencizer (rápido, sin parser)
nlp = spacy.load('es_core_news_sm', exclude=['parser'])
from spacy.pipeline import Sentencizer
sentencizer = nlp.add_pipe('sentencizer')

resena = """
Excelente producto. La calidad de sonido es increíble.
Los bajos son potentes y la batería dura mucho.
Sin embargo, el estuche de carga es algo grande.
Lo recomendaría totalmente. Llegó antes de lo esperado.
"""

doc = nlp(resena)

print("Oraciones detectadas:")
for i, sent in enumerate(doc.sents, 1):
    print(f"  {i}. {sent.text.strip()}")

print(f"\nTotal oraciones: {len(list(doc.sents))}")
```

**Aplicación**: segmentación rápida de reseñas en oraciones para análisis de sentimiento por frase. El sentencizer no requiere el parser.

---

### 15. spaCy + pandas: Procesar DataFrame de reseñas

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 15. spaCy + pandas: Procesar DataFrame de reseñas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
import spacy
import pandas as pd

nlp = spacy.load('es_core_news_sm')

# DataFrame de reseñas
df = pd.DataFrame({
    'reseña': [
        'Excelente laptop, muy rápida y con buena batería',
        'El monitor llegó con píxeles muertos, mala experiencia',
        'Buen producto relación calidad-precio, lo recomiendo',
        'La calidad de sonido es buena pero los bajos son débiles',
        'Producto defectuoso, no lo compren'
    ],
    'puntuacion': [5, 1, 4, 3, 1]
})

def procesar_resena(texto):
    doc = nlp(texto)
    tokens = [token.text for token in doc if not token.is_punct]
    lemas = [token.lemma_ for token in doc if not token.is_punct]
    pos = [token.pos_ for token in doc if not token.is_punct]
    entidades = [(ent.text, ent.label_) for ent in doc.ents]
    return pd.Series({
        'tokens': tokens,
        'lemas': lemas,
        'pos_tags': pos,
        'entidades': entidades,
        'n_tokens': len(tokens),
        'n_sustantivos': sum(1 for t in doc if t.pos_ == 'NOUN'),
        'n_adjetivos': sum(1 for t in doc if t.pos_ == 'ADJ'),
        'n_verbos': sum(1 for t in doc if t.pos_ == 'VERB')
    })

df_proc = df.join(df['reseña'].apply(procesar_resena))
print(df_proc[['reseña', 'n_tokens', 'n_sustantivos', 'n_adjetivos', 'n_verbos']].to_string())
```

**Aplicación**: procesar lotes de reseñas con spaCy para extraer features lingüísticas como entrada para modelos ML.

---

### 16. Extraer características lingüísticas como features

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 16. Extraer características lingüísticas como features

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
import spacy
import pandas as pd

nlp = spacy.load('es_core_news_sm')

descripciones = [
    "Auriculares inalámbricos con cancelación de ruido activa",
    "Laptop gaming ultraligera con procesador i9 de 13ra generación",
    "Monitor curvo 4K de 32 pulgadas con HDR1000",
    "Teclado mecánico retroiluminado con switches Cherry MX",
    "Mouse ergonómico inalámbrico con 16000 DPI"
]

def extraer_features(texto):
    doc = nlp(texto)
    features = {
        'texto': texto,
        'longitud': len(texto),
        'n_tokens': len(doc),
        'n_oraciones': len(list(doc.sents)),
        'n_sustantivos': sum(1 for t in doc if t.pos_ == 'NOUN'),
        'n_adjetivos': sum(1 for t in doc if t.pos_ == 'ADJ'),
        'n_verbos': sum(1 for t in doc if t.pos_ == 'VERB'),
        'n_numeros': sum(1 for t in doc if t.like_num),
        'n_entidades': len(doc.ents),
        'tiene_marca': any(ent.label_ in ['ORG', 'MISC'] for ent in doc.ents),
        'contiene_precio': any(ent.label_ == 'MONEY' for ent in doc.ents),
        'adjetivos': [t.text for t in doc if t.pos_ == 'ADJ'],
        'entidades': [(e.text, e.label_) for e in doc.ents],
    }
    return features

features_df = pd.DataFrame([extraer_features(desc) for desc in descripciones])
print(features_df[['texto', 'n_tokens', 'n_sustantivos', 'n_adjetivos', 'n_numeros']].to_string())
```

**Aplicación**: convertir texto en features numéricas y categóricas para modelos de clasificación, clustering o regresión.

---

### 17. nlp.pipeline personalizado (deshabilitar componentes)

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 17. nlp.pipeline personalizado (deshabilitar componentes)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
import spacy

nlp = spacy.load('es_core_news_sm')

print("Pipeline completo:", nlp.pipe_names)

texto = "Samsung lanzó un nuevo monitor gaming curvo en CES 2024 en Las Vegas"

# Todos los componentes
doc_completo = nlp(texto)
print(f"\nCompleto ({nlp.pipe_names}):")
print(f"  Entidades: {[(e.text, e.label_) for e in doc_completo.ents]}")

# Solo tokenizer (más rápido)
with nlp.select_pipes(enable=['tokenizer']):
    doc_tokenizer = nlp(texto)
    print(f"\nSolo tokenizer ({nlp.pipe_names}):")
    print(f"  Tokens: {[t.text for t in doc_tokenizer]}")
    # No se ejecuta NER
    print(f"  Entidades: {list(doc_tokenizer.ents)}")

# Deshabilitar NER
with nlp.select_pipes(disable=['ner']):
    doc_sin_ner = nlp(texto)
    print(f"\nSin NER:")
    print(f"  POS tags: {[(t.text, t.pos_) for t in doc_sin_ner[:8]]}")
    print(f"  Entidades: {list(doc_sin_ner.ents)}")
```

**Aplicación**: cuando solo necesitas tokenización (rápido) o POS tagging (sin NER), deshabilitar componentes acelera el procesamiento.

---

### 18. Integrador: Extraer producto, marca, precio de texto libre

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 18. Integrador: Extraer producto, marca, precio de texto libre

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
import spacy
from spacy.matcher import Matcher, PhraseMatcher
from spacy.pipeline import EntityRuler
import pandas as pd

nlp = spacy.load('es_core_news_sm')

# Agregar EntityRuler con marcas conocidas
ruler = nlp.add_pipe('entity_ruler', before='ner')
marcas = [
    {"label": "MARCA", "pattern": [{"LOWER": "sony"}]},
    {"label": "MARCA", "pattern": [{"LOWER": "samsung"}]},
    {"label": "MARCA", "pattern": [{"LOWER": "lg"}]},
    {"label": "MARCA", "pattern": [{"LOWER": "apple"}]},
    {"label": "MARCA", "pattern": [{"LOWER": "dell"}]},
    {"label": "MARCA", "pattern": [{"LOWER": "logitech"}]},
    {"label": "MARCA", "pattern": [{"LOWER": "hp"}, {"LOWER": "de", "OP": "?"}]},
]
ruler.add_patterns(marcas)

# Matcher para precios explícitos
matcher = Matcher(nlp.vocab)
patron_precio = [
    {'ENT_TYPE': 'MONEY'}
]
matcher.add('PRECIO', [patron_precio])

# Matcher para modelos (dígitos + letras)
patron_modelo = [
    {'SHAPE': {'REGEX': r'[A-Z0-9]+'}, 'LENGTH': {">=": 4}}
]
matcher.add('MODELO', [patron_modelo])


def extraer_info_producto(texto):
    doc = nlp(texto)
    info = {
        'texto_original': texto,
        'marcas': [],
        'precios': [],
        'modelos': [],
        'productos': [],
        'caracteristicas': []
    }

    # Extraer marcas de entidades personalizadas
    for ent in doc.ents:
        if ent.label_ == 'MARCA':
            info['marcas'].append(ent.text)
        elif ent.label_ == 'MONEY':
            info['precios'].append(ent.text)

    # Extraer frases nominales (posibles productos)
    for chunk in doc.noun_chunks:
        if chunk.root.pos_ == 'NOUN' and chunk.root.dep_ in ['nsubj', 'dobj', 'attr']:
            info['productos'].append(chunk.text)

    # Extraer adjetivos como características
    for token in doc:
        if token.pos_ == 'ADJ':
            info['caracteristicas'].append(token.text)

    return info


# Prueba con textos reales
textos_prueba = [
    "El Sony WH-1000XM5 cuesta 349 euros en Amazon",
    "Samsung Galaxy S24 ultra con 256GB por 899 dólares",
    "Laptop Dell XPS 15 con i9 por 2499 euros",
    "Monitor LG 27 pulgadas 4K a 599 mil pesos colombianos",
]

print("Extracción automática de información de productos:\n")
for texto in textos_prueba:
    info = extraer_info_producto(texto)
    print(f"Texto: {info['texto_original']}")
    print(f"  Marcas: {info['marcas'] or 'N/A'}")
    print(f"  Precios: {info['precios'] or 'N/A'}")
    print(f"  Productos candidatos: {info['productos'][:3]}")
    print(f"  Características: {info['caracteristicas'][:5]}")
    print()
```

**Aplicación**: pipeline completo de extracción de información estructurada desde texto libre de productos, utilizable para enriquecer catálogos automáticamente.

---

## Resumen

| Componente/Técnica | spaCy | Aplicación en Ventas |
|---|---|---|
| Pipeline NLP | `nlp(texto)` | Procesamiento integral en una llamada |
| Token | `token.text, .lemma_, .pos_, .tag_` | Extraer atributos lingüísticos |
| Lematización | `token.lemma_` | Normalizar verbos a infinitivo |
| POS Tagging | `token.pos_` | Identificar productos (NOUN), atributos (ADJ) |
| Dependency Parsing | `token.dep_, token.head` | Relaciones gramaticales entre términos |
| NER | `doc.ents` | Extraer marcas, precios, ubicaciones |
| Noun Chunks | `doc.noun_chunks` | Frases nominales completas (productos) |
| Similarity | `doc1.similarity(doc2)` | Recomendación, detección de duplicados |
| Displacy | `displacy.render` | Visualización de dependencias y entidades |
| Matcher | `Matcher, PhraseMatcher` | Patrones de búsqueda flexibles |
| EntityRuler | `EntityRuler` | Entidades personalizadas del dominio |
| Sentencizer | `Sentencizer` | Segmentación rápida en oraciones |

---

## Ejercicios

1. **Pipeline completo**: carga `es_core_news_sm` y procesa 10 descripciones de productos. Para cada una extrae: tokens, lemas, POS tags, entidades y noun chunks. Almacena en un DataFrame.

2. **Extracción de atributos**: usa POS tagging para extraer pares (sustantivo, adjetivo) de 8 descripciones de productos. Ejemplo: "monitor curvo" → (monitor, curvo), "silla ergonómica" → (silla, ergonómica).

3. **NER personalizado**: crea un EntityRuler con 10 marcas del mercado colombiano (Éxito, Alkosto, MercadoLibre, etc.) y 5 nombres de productos (televisor, lavadora, nevera, etc.). Procesa 5 textos.

4. **Matcher de patrones**: diseña 3 patrones con Matcher para extraer:
   - Producto + "de" + número + unidades ("monitor de 27 pulgadas")
   - "hasta" + número + unidades ("hasta 30 horas")
   - Número + "años de garantía"

5. **Similitud semántica**: toma 10 descripciones de productos similares (5 pares) y calcula su similitud con vectores. Establece un umbral para considerar productos como "mismos" vs "diferentes".

6. **Noun chunks para catálogo**: de 15 descripciones, extrae todos los noun_chunks. Filtra aquellos donde la raíz sea un sustantivo de producto (laptop, monitor, teclado, etc.). Construye un DataFrame con producto y sus modificadores.

7. **Displacy + EntityRuler**: procesa 5 reseñas, agrega entidades personalizadas para marcas, y genera visualizaciones con `displacy.render(style='ent')`. Guarda como HTML si es posible.

8. **Pipeline de extracción**: diseña una función que dado un texto de reseña o descripción devuelva un diccionario con: marca, modelo, precio, características (adjetivos), sentimiento (basado en adjetivos positivos/negativos). Pruébala con 10 textos variados.
