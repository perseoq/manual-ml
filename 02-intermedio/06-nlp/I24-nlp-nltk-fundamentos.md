# I24: NLP con NLTK — Fundamentos

## Introducción Teórica

**NLTK (Natural Language Toolkit)** es la biblioteca más completa de Python para NLP académico. Proporciona más de 50 corpus léxicos (WordNet, stopwords, etc.) y una amplia gama de herramientas de procesamiento: tokenización, stemming, lematización, POS tagging, NER, análisis de frecuencia, concordancia, y más.

### Conceptos Clave

- **Tokenización**: dividir texto en unidades mínimas (palabras, oraciones). `word_tokenize` y `sent_tokenize` son los más usados. El tokenizador interno (`PunktSentenceTokenizer`) está entrenado en múltiples idiomas.
- **Frecuencia**: `FreqDist` construye un diccionario {palabra: conteo} con métodos útiles: `.most_common(n)`, `.plot()`, `.tabulate()`. Útil para identificar términos dominantes en catálogos.
- **Stopwords**: palabras vacías (artículos, preposiciones, conjunciones) que no aportan significado. NLTK incluye stopwords para ~22 idiomas. `stopwords.words('spanish')` para español.
- **Stemming**: reduce palabras a su raíz eliminando sufijos. Tres algoritmos principales:
  - `PorterStemmer`: algoritmo original, moderado, inglés.
  - `LancasterStemmer`: más agresivo, produce raíces muy cortas.
  - `SnowballStemmer`: versión mejorada, soporta múltiples idiomas (spanish, english, french, german, etc.).
- **Lematización**: a diferencia del stemming, reduce a la forma canónica (lemma) usando un diccionario. `WordNetLemmatizer` requiere POS tag para precisión. "mejor" → "good", "corrió" → "correr".
- **POS Tagging**: etiqueta cada token con su parte de la oración usando el corpus `averaged_perceptron_tagger`. Etiquetas Penn Treebank: NN (sustantivo), VB (verbo), JJ (adjetivo), RB (adverbio), etc.
- **NER (Named Entity Recognition)**: `ne_chunk` identifica entidades nombradas (PERSON, ORGANIZATION, GPE, etc.) usando el chunker de NLTK.
- **N-gramas**: secuencias de n tokens consecutivos. `bigrams` (n=2), `trigrams` (n=3), `ngrams` (n general). Útiles para frases hechas y colocaciones.
- **Text Concordance**: muestra contexto de una palabra dentro de un texto. Útil para explorar usos de términos en catálogos.
- **Collocations**: combinaciones de palabras que co-ocurren más de lo esperado por azar. `Text.collocations()` usa bigramas con filtros estadísticos.
- **ConditionalFreqDist**: tabla de frecuencias condicionales {condición: {palabra: conteo}}. Ideal para comparar vocabulario entre categorías de productos.

### Instalación

```bash
pip install nltk
```

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Instalación.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
```

---

## Ejemplos Prácticos

### 1. word_tokenize: Tokenizar descripción de producto

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Ejemplos Prácticos
2. 1. word_tokenize: Tokenizar descripción de producto

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from nltk.tokenize import word_tokenize

descripcion = "Auriculares Bluetooth inalámbricos con cancelación de ruido activa y 30 horas de batería."
tokens = word_tokenize(descripcion, language='spanish')

print(f"Texto original: {descripcion}")
print(f"Tokens ({len(tokens)}): {tokens}")
# Output: ['Auriculares', 'Bluetooth', 'inalámbricos', 'con', 'cancelación', 'de', 'ruido', 'activa', 'y', '30', 'horas', 'de', 'batería', '.']
```

**Aplicación**: primer paso en cualquier pipeline NLP. Dividir descripciones en tokens permite análisis posteriores como frecuencia, POS tagging y NER.

---

### 2. sent_tokenize: Dividir reseña en oraciones

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 2. sent_tokenize: Dividir reseña en oraciones

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from nltk.tokenize import sent_tokenize

resena = """Compré este producto la semana pasada. La calidad del sonido es increíble. Sin embargo, los bajos podrían ser más potentes. Lo recomendaría a mis amigos."""

oraciones = sent_tokenize(resena, language='spanish')
print(f"Total oraciones: {len(oraciones)}")
for i, oracion in enumerate(oraciones, 1):
    print(f"Oración {i}: {oracion.strip()}")
```

**Aplicación**: dividir reseñas largas en oraciones permite analizar sentimiento por frase o extraer opiniones específicas.

---

### 3. FreqDist: Frecuencia de palabras en catálogo de productos

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 3. FreqDist: Frecuencia de palabras en catálogo de productos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords

catalogo = [
    "Laptop gaming con procesador i9 y 32GB de RAM",
    "Laptop ultraligera para oficina con 16GB de RAM",
    "Monitor 4K de 27 pulgadas para gaming",
    "Teclado mecánico retroiluminado para gaming",
    "Mouse inalámbrico ergonómico para oficina"
]

stop_words = set(stopwords.words('spanish'))
todos_tokens = []

for producto in catalogo:
    tokens = word_tokenize(producto.lower(), language='spanish')
    tokens_filtrados = [t for t in tokens if t.isalpha() and t not in stop_words]
    todos_tokens.extend(tokens_filtrados)

freq = FreqDist(todos_tokens)
print("Top 10 palabras más frecuentes en catálogo:")
for palabra, conteo in freq.most_common(10):
    print(f"  {palabra}: {conteo}")

freq.plot(10, title="Frecuencia de palabras en catálogo")
```

**Aplicación**: identificar términos dominantes en un catálogo ayuda a optimizar SEO, tags de productos y categorización.

---

### 4. Stopwords: Eliminar palabras vacías en español

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 4. Stopwords: Eliminar palabras vacías en español

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

descripcion = "El producto es muy bueno y tiene una excelente calidad, pero el precio es un poco elevado para lo que ofrece."
stop_words_es = set(stopwords.words('spanish'))

tokens = word_tokenize(descripcion.lower(), language='spanish')
tokens_limpios = [t for t in tokens if t not in stop_words_es and t.isalpha()]

print(f"Original: {descripcion}")
print(f"Stopwords: {stop_words_es}")
print(f"Sin stopwords: {' '.join(tokens_limpios)}")
```

**Aplicación**: limpiar descripciones eliminando ruido lingüístico mejora la calidad de features para modelos de clasificación y clustering.

---

### 5. PorterStemmer: Reducir palabras a raíz (inglés)

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 5. PorterStemmer: Reducir palabras a raíz (inglés)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()
palabras = ["computadora", "computación", "computarizado", "computador", "cómputo"]

descripcion = "Computing computational computers computed computerized computation"
tokens = word_tokenize(descripcion.lower())

print("Palabra -> Raíz (Porter):")
for palabra in tokens:
    print(f"  {palabra} -> {stemmer.stem(palabra)}")
```

**Nota**: PorterStemmer funciona mejor en inglés. Para español usar SnowballStemmer. "computadora" en inglés → "comput".

---

### 6. SnowballStemmer: Stemmer multilingüe (español)

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 6. SnowballStemmer: Stemmer multilingüe (español)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

stemmer_es = SnowballStemmer('spanish')
palabras = ["corriendo", "corredor", "corrió", "correrán", "corrimos", "carrera"]

print("Stemmer español (Snowball):")
for p in palabras:
    print(f"  {p} -> {stemmer_es.stem(p)}")

# Comparar idiomas
desc_es = "Los auriculares inalámbricos funcionan perfectamente"
desc_en = "The wireless headphones work perfectly"
desc_fr = "Les écouteurs sans fil fonctionnent parfaitement"

stemmer_en = SnowballStemmer('english')
stemmer_fr = SnowballStemmer('french')

for p in word_tokenize(desc_es.lower()):
    print(f"ES: {p} -> {stemmer_es.stem(p)}")
for p in word_tokenize(desc_en.lower()):
    print(f"EN: {p} -> {stemmer_en.stem(p)}")
for p in word_tokenize(desc_fr.lower()):
    print(f"FR: {p} -> {stemmer_fr.stem(p)}")
```

**Aplicación**: procesar catálogos multilingües con un stemmer por idioma, reduciendo todas las variantes morfológicas a una raíz común.

---

### 7. LancasterStemmer: Stemmer más agresivo

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 7. LancasterStemmer: Stemmer más agresivo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from nltk.stem import LancasterStemmer, PorterStemmer, SnowballStemmer

lancaster = LancasterStemmer()
porter = PorterStemmer()
snowball = SnowballStemmer('english')

palabras = ["computing", "computer", "computation", "computational", "computers"]

print(f"{'Palabra':20s} {'Porter':15s} {'Lancaster':15s} {'Snowball':15s}")
print("-" * 65)
for p in palabras:
    print(f"{p:20s} {porter.stem(p):15s} {lancaster.stem(p):15s} {snowball.stem(p):15s}")
```

**Aplicación**: Lancaster es útil cuando se necesita máxima reducción, pero puede producir raíces no palabras reales. Elegir según necesidad.

---

### 8. Comparar Porter vs Lancaster vs Snowball en español

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 8. Comparar Porter vs Lancaster vs Snowball en español

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer

porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball_es = SnowballStemmer('spanish')

palabras = ["comunicación", "comunicando", "comunicado", "comunicarse", "comunicativo"]

print(f"{'Palabra':20s} {'Porter (EN)':20s} {'Lancaster (EN)':20s} {'Snowball (ES)':20s}")
print("-" * 80)
for p in palabras:
    print(f"{p:20s} {porter.stem(p):20s} {lancaster.stem(p):20s} {snowball_es.stem(p):20s}")

# Conclusión: Snowball para español da mejores resultados
```

**Aplicación**: al procesar productos en español, SnowballStemmer('spanish') es la opción correcta. Porter y Lancaster están diseñados para inglés y darán resultados pobres.

---

### 9. WordNetLemmatizer: Lematización

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 9. WordNetLemmatizer: Lematización

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()

palabras = ["better", "best", "good", "went", "gone", "going", "ran", "running", "runs"]

print(f"{'Palabra':15s} {'Lemma (default)':20s} {'Lemma (POS)':20s}")
print("-" * 55)
for p in palabras:
    lemma_default = lemmatizer.lemmatize(p)
    lemma_pos = lemmatizer.lemmatize(p, pos='v')  # verb
    print(f"{p:15s} {lemma_default:20s} {lemma_pos:20s}")
```

**Aplicación**: la lematización es superior al stemming para búsqueda semántica y comparación de textos, porque produce palabras reales. "mejor" → "good" (WordNet).

---

### 10. pos_tag: Etiquetar partes de la oración

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 10. pos_tag: Etiquetar partes de la oración

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from nltk import pos_tag
from nltk.tokenize import word_tokenize

descripcion = "Este televisor 4K tiene una pantalla OLED de 55 pulgadas con HDR10 y Dolby Vision."
tokens = word_tokenize(descripcion, language='spanish')
etiquetado = pos_tag(tokens, lang='spa')

print(f"{'Token':20s} {'POS Tag':10s} {'Descripción':30s}")
print("-" * 60)
for token, tag in etiquetado:
    print(f"{token:20s} {tag:10s}")
```

**Aplicación**: identificar sustantivos (productos), adjetivos (atributos) y verbos (acciones) en descripciones para extracción de características.

---

### 11. ne_chunk: Reconocer entidades nombradas (NER)

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 11. ne_chunk: Reconocer entidades nombradas (NER)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize

texto = "Apple lanzó el nuevo iPhone 15 en Cupertino, California. El CEO Tim Cook presentó el producto en septiembre de 2024."
tokens = word_tokenize(texto)
etiquetado = pos_tag(tokens)
entidades = ne_chunk(etiquetado)

print("Entidades nombradas detectadas:")
print(entidades)

print("\n--- Formato árbol ---")
for subtree in entidades:
    if hasattr(subtree, 'label'):
        entidad = ' '.join(c[0] for c in subtree)
        print(f"  {subtree.label()}: {entidad}")
```

**Aplicación**: extraer marcas (Apple), lugares (Cupertino), personas (Tim Cook) y fechas de descripciones de productos automáticamente.

---

### 12. ngrams/bigrams/trigrams: Extraer frases de descripciones

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 12. ngrams/bigrams/trigrams: Extraer frases de descripciones

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from nltk import ngrams, bigrams, trigrams
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

descripcion = "Auriculares inalámbricos con cancelación de ruido activa y micrófono incorporado para llamadas manos libres"
tokens = word_tokenize(descripcion.lower(), language='spanish')

print("Bigramas (pares de palabras):")
for bg in bigrams(tokens):
    print(f"  {bg}")

print("\nTrigramas (tríos de palabras):")
for tg in trigrams(tokens):
    print(f"  {tg}")

# Frecuencia de n-gramas
todos_bigrams = list(bigrams(tokens))
freq_bg = FreqDist(todos_bigrams)
print(f"\nBigrama más frecuente: {freq_bg.most_common(3)}")
```

**Aplicación**: extraer frases compuestas como "cancelación de ruido" o "micrófono incorporado" que tienen significado como unidad.

---

### 13. Concordance: Ver contexto de palabra en texto

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 13. Concordance: Ver contexto de palabra en texto

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from nltk.text import Text
from nltk.tokenize import word_tokenize

catalogo_texto = """
Laptop gaming con procesador i9 y 32GB de RAM. Laptop ultraligera para oficina.
Monitor 4K gaming de 27 pulgadas. Teclado mecánico gaming retroiluminado.
Mouse gaming inalámbrico con 16000 DPI. Silla gaming ergonómica ajustable.
Escritorio gaming con soporte para monitores. Auriculares gaming con micrófono.
"""

tokens = word_tokenize(catalogo_texto.lower())
text_obj = Text(tokens)

print("Concordance para 'gaming':")
text_obj.concordance('gaming', width=80, lines=10)
```

**Aplicación**: explorar cómo se usa un término en catálogos para entender contexto y usos correctos.

---

### 14. Similar: Palabras que aparecen en contexto similar

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 14. Similar: Palabras que aparecen en contexto similar

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from nltk.text import Text
from nltk.tokenize import word_tokenize

texto_largo = """
El televisor OLED tiene negros profundos y colores vivos. 
El monitor OLED ofrece contraste infinito y ángulos de visión amplios.
La pantalla OLED consume menos energía que las LCD tradicionales.
El panel OLED es más delgado y flexible. 
La tecnología OLED permite diseñar televisores curvos.
OLED es la mejor tecnología para gaming por su tiempo de respuesta.
Los televisores OLED de LG son los más vendidos.
"""

tokens = word_tokenize(texto_largo.lower())
text_obj = Text(tokens)

print("Palabras similares a 'oled':")
text_obj.similar('oled', num=5)
```

**Aplicación**: encontrar términos relacionados en el mismo dominio semántico para expandir vocabulario de búsqueda.

---

### 15. dispersion_plot: Dispersión de palabras clave en texto

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 15. dispersion_plot: Dispersión de palabras clave en texto

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from nltk.text import Text
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

resena_larga = """
El producto llegó a tiempo. La calidad de construcción es excelente.
El sonido es nítido y claro. Me encanta la cancelación de ruido.
La batería dura mucho tiempo. Sin embargo, el estuche es grande.
La calidad del micrófono es buena para llamadas.
Recomiendo este producto por su calidad y duración.
El sonido supera mis expectativas. La cancelación de ruido funciona muy bien.
"""

tokens = word_tokenize(resena_larga.lower())
text_obj = Text(tokens)

palabras_clave = ['calidad', 'sonido', 'cancelación', 'batería', 'producto']
text_obj.dispersion_plot(palabras_clave)
plt.title("Dispersión de palabras clave en reseña")
plt.show()
```

**Aplicación**: visualizar dónde aparecen términos clave en reseñas para entender el flujo narrativo.

---

### 16. ConditionalFreqDist: Frecuencia condicional por categoría

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 16. ConditionalFreqDist: Frecuencia condicional por categoría

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from nltk.probability import ConditionalFreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

catalogo_categorizado = {
    'laptops': [
        "Laptop gaming con procesador i9",
        "Laptop ultraligera con SSD",
        "Laptop para programación con 32GB RAM"
    ],
    'audifonos': [
        "Auriculares inalámbricos con Bluetooth",
        "Audífonos con cancelación de ruido",
        "Cascos gaming con micrófono"
    ],
    'monitores': [
        "Monitor 4K de 27 pulgadas",
        "Monitor gaming 144Hz",
        "Monitor UltraWide para productividad"
    ]
}

stop_words = set(stopwords.words('spanish'))
cfd = ConditionalFreqDist()

for categoria, productos in catalogo_categorizado.items():
    for producto in productos:
        tokens = word_tokenize(producto.lower(), language='spanish')
        tokens_filtrados = [t for t in tokens if t.isalpha() and t not in stop_words]
        for token in tokens_filtrados:
            cfd[categoria][token] += 1

# Mostrar top 3 por categoría
for categoria in cfd.conditions():
    print(f"\nCategoría: {categoria}")
    for palabra, freq in cfd[categoria].most_common(3):
        print(f"  {palabra}: {freq}")
```

**Aplicación**: identificar qué términos diferencian categorías de productos. "gaming" aparece solo en laptops/monitores, "cancelación" en audífonos.

---

### 17. Text.collocations: Colocaciones frecuentes

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 17. Text.collocations: Colocaciones frecuentes

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from nltk.text import Text
from nltk.tokenize import word_tokenize

catalogo_texto = """
cancelación de ruido cancelación de ruido cancelación activa
calidad de sonido calidad de construcción calidad de imagen
tiempo de respuesta alta definición resolución 4k
pulgadas de pantalla pantalla táctil carga rápida
conexión inalámbrica duración de batería eficiencia energética
asistente de voz control por gestos reconocimiento facial
"""

tokens = word_tokenize(catalogo_texto.lower())
text_obj = Text(tokens)

print("Colocaciones (collocations):")
text_obj.collocations(num=10)
```

**Aplicación**: identificar frases compuestas típicas en descripciones de productos para mejorar extracción de características.

---

### 18. Integrador: Pipeline NLP completo sobre descripciones de productos

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 18. Integrador: Pipeline NLP completo sobre descripciones de productos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from nltk import (
    word_tokenize, sent_tokenize, pos_tag, ne_chunk, bigrams, trigrams
)
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.text import Text
import pandas as pd

# Dataset simulado
catalogo = pd.DataFrame({
    'producto': [
        'Auriculares Sony WH-1000XM5',
        'Laptop Dell XPS 15',
        'Monitor LG UltraGear 27GN950',
        'Teclado Logitech MX Mechanical',
    ],
    'descripcion': [
        'Auriculares inalámbricos Sony con cancelación de ruido activa y 30 horas de batería. Ideal para viajes y oficina.',
        'Laptop Dell XPS 15 con procesador Intel i9, 32GB RAM y pantalla OLED 4K. Perfecta para creadores de contenido.',
        'Monitor LG 27 pulgadas 4K UHD con tasa de refresco 144Hz y HDR10. Gaming de alta gama.',
        'Teclado mecánico inalámbrico Logitech con retroiluminación RGB y switches táctiles. Ideal para productividad.'
    ],
    'precio': [349990, 2499990, 599990, 189990]
})

# Pipeline NLP
stop_words = set(stopwords.words('spanish'))
stemmer = SnowballStemmer('spanish')

def pipeline_nlp(texto):
    # 1. Tokenización de oraciones
    oraciones = sent_tokenize(texto, language='spanish')
    # 2. Tokenización de palabras
    tokens = word_tokenize(texto.lower(), language='spanish')
    # 3. Limpieza
    tokens_limpios = [t for t in tokens if t.isalpha() and t not in stop_words]
    # 4. Stemming
    stems = [stemmer.stem(t) for t in tokens_limpios]
    # 5. POS tagging
    pos = pos_tag(tokens, lang='spa')
    # 6. NER básico
    entidades = ne_chunk(pos)
    # 7. Extraer entidades
    entidades_list = []
    for subtree in entidades:
        if hasattr(subtree, 'label'):
            entidades_list.append((subtree.label(), ' '.join(c[0] for c in subtree)))
    # 8. Bigramas
    bigramas = list(bigrams(tokens_limpios))
    # 9. Frecuencia
    freq = FreqDist(tokens_limpios)
    return {
        'oraciones': len(oraciones),
        'tokens_raw': len(tokens),
        'tokens_limpios': len(tokens_limpios),
        'stems': stems,
        'pos_tags': pos,
        'entidades': entidades_list,
        'top_palabras': freq.most_common(5),
        'bigramas': bigramas[:5]
    }

for idx, row in catalogo.iterrows():
    print(f"\n{'='*60}")
    print(f"Producto: {row['producto']}")
    print(f"{'='*60}")
    resultado = pipeline_nlp(row['descripcion'])
    print(f"Oraciones: {resultado['oraciones']}, Tokens: {resultado['tokens_raw']}, Limpios: {resultado['tokens_limpios']}")
    print(f"Top palabras: {resultado['top_palabras']}")
    print(f"Entidades: {resultado['entidades']}")
    print(f"Bigramas: {resultado['bigramas']}")
```

**Aplicación**: pipeline completo para preprocesar descripciones antes de alimentar modelos de recomendación, búsqueda o clasificación.

---

## Resumen

| Técnica | Función/Clase | Aplicación en Ventas |
|---|---|---|
| Tokenización | `word_tokenize`, `sent_tokenize` | Dividir descripciones en unidades procesables |
| Frecuencia | `FreqDist` | Identificar términos clave en catálogos |
| Stopwords | `stopwords.words('spanish')` | Filtrar ruido lingüístico |
| Stemming | `PorterStemmer`, `SnowballStemmer`, `LancasterStemmer` | Normalizar variantes morfológicas |
| Lematización | `WordNetLemmatizer` | Obtener forma canónica de palabras |
| POS Tagging | `pos_tag` | Identificar sustantivos (productos), adjetivos (atributos) |
| NER | `ne_chunk` | Extraer marcas, lugares, personas |
| N-gramas | `ngrams`, `bigrams`, `trigrams` | Extraer frases compuestas |
| Concordancia | `Text.concordance` | Explorar contexto de uso |
| Colocaciones | `Text.collocations` | Identificar frases típicas del dominio |
| Frecuencia Condicional | `ConditionalFreqDist` | Comparar vocabulario entre categorías |

---

## Ejercicios

1. **Tokenización**: toma 5 descripciones de productos electrónicos, tokenízalas con `word_tokenize` y reporta el número promedio de tokens por descripción.

2. **Análisis de frecuencia**: con un conjunto de 10 reseñas de productos, usa `FreqDist` para encontrar las 15 palabras más comunes después de eliminar stopwords. Genera un gráfico de barras.

3. **Stemming comparativo**: aplica PorterStemmer, LancasterStemmer y SnowballStemmer('spanish') a 20 palabras en español del dominio de ventas (electrodomésticos, ropa, tecnología). Crea una tabla comparativa y determina cuál funciona mejor.

4. **Lematización en contexto**: toma 5 reseñas, etiqueta POS cada token, y aplica lematización solo a verbos y sustantivos. Muestra el texto lematizado vs. original.

5. **NER en catálogo**: extrae manualmente de 10 descripciones de productos las entidades nombradas (marcas, modelos, precios). Compara con la salida de `ne_chunk`. ¿Qué diferencias encuentras?

6. **Bigramas de productos**: de un catálogo de 20 productos, extrae los bigramas más frecuentes. Filtra aquellos que contienen al menos un adjetivo (JJ) o sustantivo (NN) usando pos_tag.

7. **ConditionalFreqDist**: crea 3 categorías (tecnología, hogar, ropa) con 5 descripciones cada una. Usa ConditionalFreqDist para encontrar las 5 palabras más distintivas de cada categoría.

8. **Pipeline integrador**: diseña una función que dado un DataFrame de productos con columna 'descripcion', devuelva: tokens limpios, stems, POS tags, entidades, bigramas y top-10 palabras. Pruébala con un mínimo de 5 productos reales o simulados.
