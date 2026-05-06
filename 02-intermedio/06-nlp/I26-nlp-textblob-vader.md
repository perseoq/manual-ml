# I26: NLP con TextBlob y VADER — Análisis de Sentimiento

## Introducción Teórica

**TextBlob** y **VADER** son dos bibliotecas especializadas en análisis de sentimiento y procesamiento básico de texto. TextBlob ofrece un enfoque general (basado en lexicon + patrones), mientras que VADER está específicamente diseñado para el lenguaje de redes sociales y reseñas.

### TextBlob

Construido sobre NLTK y Pattern, TextBlob proporciona una API sencilla para tareas comunes de NLP:

- **Sentiment Analysis**: `TextBlob(text).sentiment` devuelve un `namedtuple` con `polarity` (−1 a 1) y `subjectivity` (0 a 1).
  - Polaridad: −1 (muy negativo), 0 (neutro), +1 (muy positivo)
  - Subjetividad: 0 (objetivo/hechos), 1 (subjetivo/opinión)
- **Noun Phrase Extraction**: `.noun_phrases` extrae frases nominales usando el chunker integrado.
- **Tokenización**: `.words` (palabras), `.sentences` (oraciones)
- **Corrección ortográfica**: `.correct()` usa un corrector basado en probabilidades.
- **Traducción**: `.translate(to=...)` y `.detect_language()` usan Google Translate.
- **Flexión**: `.singularize()`, `.pluralize()`
- **N-gramas**: `.ngrams(n=3)`

### VADER (Valence Aware Dictionary and sEntiment Reasoner)

Diseñado específicamente para el lenguaje de redes sociales, reseñas y opiniones. Sus ventajas:

- **Funciona sin entrenamiento**: lexicon curado con intensidades de −4 a +4 para ~7500 palabras
- **Maneja**: mayúsculas (INTENSIFICADOR), signos de exclamación (!!!), emojis (😊, 😞), modificadores ("extremadamente"), negaciones ("no es bueno")
- **Salida**: `compound` (−1 a 1, normalizado), `pos`, `neu`, `neg` (0 a 1)
- **Reglas lingüísticas**: intensificadores (very, extremely → +0.293), atenuadores (kind of, sort of → −0.5), inversores de polaridad (negación → multiplica por −0.74)

### Cuándo usar cada uno

| Criterio | TextBlob | VADER |
|---|---|---|
| Idioma | Inglés (principal), español limitado | Inglés, español limitado (léxico traducible) |
| Velocidad | Más lento (depende de NLTK) | Muy rápido |
| Emojis | No soporta | Soporta |
| Mayúsculas/!!! | No | Sí |
| Subjetividad | Sí | No |
| Corrección ortográfica | Sí | No |
| Traducción | Sí | No |
| Reseñas largas | Bueno | Excelente |
| Redes sociales | Regular | Excelente |

### Instalación

```bash
pip install textblob
python -m textblob.download_corpora
pip install vaderSentiment
```

---

## Ejemplos Prácticos

### 1. TextBlob: Sentimiento de reseña de producto (polaridad −1 a 1)

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Instalación.*

1. Ejemplos Prácticos
2. 1. TextBlob: Sentimiento de reseña de producto (polaridad −1 a 1)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from textblob import TextBlob

resenas = [
    "This product is amazing, I love it! Best purchase ever.",
    "Terrible quality, broke after one week. Do not buy.",
    "The product is okay, nothing special but works fine.",
    "Absolutely fantastic! Exceeded all my expectations.",
    "Not bad, but I expected more for the price."
]

print(f"{'Reseña':70s} {'Polaridad':10s}")
print("-" * 80)
for resena in resenas:
    blob = TextBlob(resena)
    print(f"{resena:70s} {blob.sentiment.polarity:+.3f}")
```

**Aplicación**: clasificar reseñas como positivas/negativas automáticamente según la polaridad.

---

### 2. TextBlob: Subjetividad de reseña (0=objetivo, 1=subjetivo)

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 2. TextBlob: Subjetividad de reseña (0=objetivo, 1=subjetivo)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from textblob import TextBlob

textos = [
    ("La laptop tiene 16GB de RAM y procesador i7", "Hecho objetivo"),
    ("Esta laptop es increíblemente rápida y hermosa", "Opinión subjetiva"),
    ("El producto cuesta 500 dólares y pesa 2 kilos", "Hecho objetivo"),
    ("Odio este producto, es una porquería", "Opinión negativa fuerte"),
    ("La batería dura 8 horas según el fabricante", "Hecho"),
    ("Recomiendo este producto a todos mis amigos", "Recomendación subjetiva"),
]

print(f"{'Texto':55s} {'Tipo':25s} {'Polaridad':10s} {'Subjetividad':12s}")
print("-" * 102)
for texto, tipo in textos:
    blob = TextBlob(texto)
    print(f"{texto:55s} {tipo:25s} {blob.sentiment.polarity:+.3f} {blob.sentiment.subjectivity:.3f}")
```

**Aplicación**: identificar si una reseña expresa opinión (subjetiva) o solo informa hechos (objetiva). Útil para filtrar reseñas útiles.

---

### 3. TextBlob: Extraer noun_phrases de descripción

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 3. TextBlob: Extraer noun_phrases de descripción

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from textblob import TextBlob

descripcion = "The new wireless noise-canceling headphones with Bluetooth 5.0 and a long-lasting battery"
blob = TextBlob(descripcion)

print("Noun phrases extraídas:")
for frase in blob.noun_phrases:
    print(f"  - {frase}")

# Aplicación en múltiples descripciones
descripciones = [
    "High-performance gaming laptop with RGB keyboard and 144Hz display",
    "Ergonomic office chair with lumbar support and adjustable armrests",
    "4K Ultra HD smart TV with HDR10 and built-in streaming apps",
]

print("\nProductos y sus frases nominales:")
for desc in descripciones:
    blob = TextBlob(desc)
    print(f"\n  Descripción: {desc}")
    print(f"  Frases: {blob.noun_phrases}")
```

**Aplicación**: extraer nombres de productos y características clave de descripciones.

---

### 4. TextBlob: word_counts de palabras clave

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 4. TextBlob: word_counts de palabras clave

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from textblob import TextBlob
from collections import Counter

resena_larga = """
The headphones are great. Great sound quality and great build.
The battery life is amazing. Amazing sound and amazing comfort.
Comfort is key for long sessions and these are comfortable.
Sound quality is awesome. Awesome product overall.
"""

blob = TextBlob(resena_larga.lower())

print("Conteo de palabras:")
conteos = blob.word_counts
for palabra, conteo in conteos.most_common(15):
    print(f"  {palabra}: {conteo}")

# Frecuencia de frases nominales
print("\nNoun phrases más frecuentes:")
noun_counts = Counter(blob.noun_phrases)
for frase, conteo in noun_counts.most_common(5):
    print(f"  {frase}: {conteo}")
```

**Aplicación**: identificar términos y frases recurrentes en reseñas para entender qué aspectos valora más el cliente.

---

### 5. TextBlob: ngrams para encontrar patrones

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 5. TextBlob: ngrams para encontrar patrones

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from textblob import TextBlob

resena = "The battery life is excellent but the sound quality is poor"
blob = TextBlob(resena)

print("Unigramas (1-gram):")
for ng in blob.ngrams(n=1):
    print(f"  {ng}")

print("\nBigramas (2-gram):")
for ng in blob.ngrams(n=2):
    print(f"  {ng}")

print("\nTrigramas (3-gram):")
for ng in blob.ngrams(n=3):
    print(f"  {ng}")

# Analizar sentimiento de cada bigrama
print("\nSentimiento por bigrama:")
for ng in blob.ngrams(n=2):
    bigrama_texto = ' '.join(ng)
    bigrama_blob = TextBlob(bigrama_texto)
    print(f"  '{bigrama_texto}': polaridad={bigrama_blob.sentiment.polarity:+.3f}")
```

**Aplicación**: descomponer reseñas en n-gramas para analizar sentimiento a nivel de frase corta.

---

### 6. TextBlob: singularize/pluralize palabras

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 6. TextBlob: singularize/pluralize palabras

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from textblob import TextBlob
from textblob.word import Word

palabras = ["laptop", "mouse", "headphone", "keyboard", "monitor", "cable", "charger", "battery"]

print(f"{'Singular':15s} {'Plural':15s} {'Plural->Singular':20s}")
print("-" * 50)
for p in palabras:
    w = Word(p)
    plural = w.pluralize()
    volver = Word(plural).singularize()
    print(f"{p:15s} {plural:15s} {volver:20s}")

# Aplicación a descripciones
desc = "I bought three laptops, two monitors, and five mice for the office"
blob = TextBlob(desc)
print(f"\nOriginal: {desc}")
print(f"Singularizado: {' '.join(Word(w).singularize() for w in blob.words)}")
```

**Aplicación**: normalizar plurales a singular para estandarizar términos en catálogos.

---

### 7. TextBlob: correct (corregir ortografía de reseñas)

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 7. TextBlob: correct (corregir ortografía de reseñas)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from textblob import TextBlob

resenas_con_errores = [
    "Ths product is amazng, I luv it!",
    "Terible qualtiy, brok after one wek.",
    "Excelent servise and fast shippping.",
    "The batry life is awesom, recomendd.",
    "Greate value for the priice, five starz.",
]

print("Corrección de reseñas:")
for original in resenas_con_errores:
    blob = TextBlob(original)
    corregido = blob.correct()
    print(f"  Original:  {original}")
    print(f"  Corregido: {corregido}")
    print(f"  Polaridad original: {original}/corregida: {TextBlob(corregido.string).sentiment.polarity:+.3f}\n")
```

**Aplicación**: corregir reseñas con errores ortográficos antes de analizar sentimiento. Las reseñas de usuarios reales suelen tener errores.

---

### 8. TextBlob: translate (traducir reseña a inglés)

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 8. TextBlob: translate (traducir reseña a inglés)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from textblob import TextBlob

resena_es = "Excelente producto, muy buena calidad y llegó rápido. Lo recomiendo totalmente."
blob = TextBlob(resena_es)

try:
    idioma = blob.detect_language()
    print(f"Idioma detectado: {idioma}")

    traduccion = blob.translate(to='en')
    print(f"Original: {resena_es}")
    print(f"Traducción: {traduccion}")

    # Analizar sentimiento en ambos idiomas
    sent_es = blob.sentiment
    sent_en = TextBlob(traduccion.string).sentiment
    print(f"\nSentimiento (ES): polaridad={sent_es.polarity:+.3f}, subjetividad={sent_es.subjectivity:.3f}")
    print(f"Sentimiento (EN): polaridad={sent_en.polarity:+.3f}, subjetividad={sent_en.subjectivity:.3f}")
except Exception as e:
    print(f"Error de traducción: {e}. Se requiere conexión a internet.")
```

**Aplicación**: traducir reseñas de múltiples idiomas a inglés para analizarlas con VADER o TextBlob (que tienen mejor soporte en inglés).

---

### 9. VADER: analyzer.polarity_scores()

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 9. VADER: analyzer.polarity_scores()

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

textos = [
    "This product is absolutely amazing and wonderful!",
    "Terrible quality, completely disappointed.",
    "The product is fine, average quality.",
    "I love this! Best purchase ever!!!",
    "This is the worst product I have ever bought.",
    "Not bad, could be better, but acceptable for the price.",
]

print(f"{'Texto':50s} {'Compound':10s} {'Pos':8s} {'Neu':8s} {'Neg':8s}")
print("-" * 84)
for texto in textos:
    scores = analyzer.polarity_scores(texto)
    print(f"{texto:50s} {scores['compound']:+.3f} {scores['pos']:.3f} {scores['neu']:.3f} {scores['neg']:.3f}")
```

**Aplicación**: evaluación completa de sentimiento con compound como métrica unRealiza la operación indicada con los parámetros definidos.

---

### 10. VADER: compound score como métrica unificada

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 10. VADER: compound score como métrica unificada

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

resenas = [
    "Absolutely love this product! Exceeded all expectations.",
    "Pretty good, satisfied with the purchase.",
    "It's okay, nothing special but works.",
    "Not great, has some issues.",
    "Terrible, complete waste of money.",
    "Best headphones I've ever owned! Highly recommend!",
    "Decent for the price, but build quality could be better.",
    "Horrible customer service and defective product."
]

print(f"{'Reseña':65s} {'Compound':10s} {'Clasificación':15s}")
print("-" * 90)
for resena in resenas:
    score = analyzer.polarity_scores(resena)['compound']
    if score >= 0.5:
        clasificacion = "Muy positivo"
    elif score >= 0.05:
        clasificacion = "Positivo"
    elif score <= -0.5:
        clasificacion = "Muy negativo"
    elif score <= -0.05:
        clasificacion = "Negativo"
    else:
        clasificacion = "Neutro"
    print(f"{resena:65s} {score:+.3f} {clasificacion:15s}")
```

**Aplicación**: clasificar reseñas en 5 niveles de sentimiento usando thresholds sobre compound.

---

### 11. VADER: intensificadores (mayúsculas, !!!)

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 11. VADER: intensificadores (mayúsculas, !!!)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

pares = [
    ("good product", "GOOD PRODUCT"),
    ("This is amazing", "THIS IS AMAZING!!!"),
    ("great quality", "GREAT QUALITY!!!"),
    ("terrible", "TERRIBLE!!!"),
    ("bad experience", "BAD EXPERIENCE!!!"),
    ("I love it", "I LOVE IT!!!"),
]

print(f"{'Normal':30s} {'Compound':10s} {'INTENSIFICADO':30s} {'Compound':10s}")
print("-" * 80)
for normal, intenso in pares:
    score_n = analyzer.polarity_scores(normal)['compound']
    score_i = analyzer.polarity_scores(intenso)['compound']
    print(f"{normal:30s} {score_n:+.3f} {intenso:30s} {score_i:+.3f}")
```

**Aplicación**: VADER captura correctamente que "GOOD PRODUCT!!!" es más positivo que "good product" gracias a las reglas de intensificación.

---

### 12. VADER: negaciones ("no es bueno" → negativo)

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 12. VADER: negaciones ("no es bueno" → negativo)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

pares = [
    ("The product is good", "The product is not good"),
    ("This works well", "This does not work well"),
    ("I like the quality", "I do not like the quality"),
    ("Excellent value", "Not excellent value at all"),
    ("The battery lasts long", "The battery does not last long"),
    ("It is comfortable", "It is not comfortable"),
    ("Good sound", "Not good sound"),
    ("Recommended", "Not recommended"),
]

print(f"{'Afirmación':30s} {'Comp':8s} {'Negación':35s} {'Comp':8s}")
print("-" * 81)
for afirmacion, negacion in pares:
    score_a = analyzer.polarity_scores(afirmacion)['compound']
    score_n = analyzer.polarity_scores(negacion)['compound']
    print(f"{afirmacion:30s} {score_a:+.3f} {negacion:35s} {score_n:+.3f}")
```

**Aplicación**: VADER maneja correctamente negaciones, invirtiendo la polaridad. "not good" → negativo.

---

### 13. VADER: manejo de emojis 😊😞

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 13. VADER: manejo de emojis 😊😞

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

textos_con_emojis = [
    "Great product 😊",
    "Love it! 😍👍",
    "Terrible quality 😞",
    "Disappointed 😢",
    "So happy with my purchase! 😄🎉",
    "Waste of money 😡",
    "It's okay 😐",
    "Excellent service! 😊👏✨",
    "Bad experience 😠👎",
    "Perfect! ❤️🔥",
]

print(f"{'Texto':45s} {'Compound':10s} {'Clasificación':15s}")
print("-" * 70)
for texto in textos_con_emojis:
    scores = analyzer.polarity_scores(texto)
    compound = scores['compound']
    if compound >= 0.5:
        clasif = "😊 Positivo"
    elif compound <= -0.5:
        clasif = "😞 Negativo"
    else:
        clasif = "😐 Neutro"
    print(f"{texto:45s} {compound:+.3f} {clasif:15s}")
```

**Aplicación**: VADER es la mejor opción para analizar reseñas que contienen emojis (redes sociales, reseñas desde móviles).

---

### 14. Comparar TextBlob vs VADER en mismas reseñas

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 14. Comparar TextBlob vs VADER en mismas reseñas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

analyzer = SentimentIntensityAnalyzer()

resenas = [
    "This product is absolutely amazing and wonderful! I love it so much!!!",
    "Terrible quality, completely disappointed. Would not recommend.",
    "The product is okay, nothing special but it works fine.",
    "GOOD PRODUCT!!! BEST PURCHASE EVER!!!",
    "Not bad, could be better. Average quality.",
    "I HATE THIS!!! COMPLETELY USELESS!!!",
    "Excellent value for money, exceeded expectations 😊",
    "The product arrived broken. Very disappointed 😞",
]

datos = []
for resena in resenas:
    tb = TextBlob(resena)
    vader = analyzer.polarity_scores(resena)
    datos.append({
        'Reseña': resena[:50],
        'TextBlob_polarity': tb.sentiment.polarity,
        'TextBlob_subjectivity': tb.sentiment.subjectivity,
        'VADER_compound': vader['compound'],
        'VADER_pos': vader['pos'],
        'VADER_neu': vader['neu'],
        'VADER_neg': vader['neg'],
    })

df = pd.DataFrame(datos)
print(df.to_string(index=False))
```

**Aplicación**: entender las diferencias entre ambos analizadores. VADER capta mejor intensificadores y emojis; TextBlob da más matiz en subjetividad.

---

### 15. Clasificar reseñas como positivo/negativo/neutro con umbrales

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 15. Clasificar reseñas como positivo/negativo/neutro con umbrales

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

analyzer = SentimentIntensityAnalyzer()

resenas = [
    "Amazing laptop, incredibly fast and beautiful design!",
    "The monitor has dead pixels, very disappointed.",
    "Average keyboard, does the job but nothing special.",
    "Love these headphones! Best sound quality ever!",
    "Terrible customer service, never buying again.",
    "Good value for the price, reasonable quality.",
    "The battery drains too fast, not recommended.",
    "Perfect product, exactly what I needed!",
]

def clasificar_vader(compound):
    if compound >= 0.5: return "positivo"
    elif compound <= -0.5: return "negativo"
    else: return "neutro"

def clasificar_textblob(polarity):
    if polarity >= 0.3: return "positivo"
    elif polarity <= -0.3: return "negativo"
    else: return "neutro"

print(f"{'Reseña':55s} {'VADER':12s} {'TextBlob':12s} {'Coinciden':10s}")
print("-" * 89)
for resena in resenas:
    v_compound = analyzer.polarity_scores(resena)['compound']
    tb_polarity = TextBlob(resena).sentiment.polarity
    v_cls = clasificar_vader(v_compound)
    tb_cls = clasificar_textblob(tb_polarity)
    coincide = "✓" if v_cls == tb_cls else "✗"
    print(f"{resena:55s} {v_cls:12s} {tb_cls:12s} {coincide:10s}")
```

**Aplicación**: establecer thresholds de clasificación para automatizar etiquetado de reseñas en sistemas de feedback.

---

### 16. Análisis de sentimiento por producto (agregado)

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 16. Análisis de sentimiento por producto (agregado)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from textblob import TextBlob
import pandas as pd

resenas_por_producto = {
    "Auriculares Sony": [
        "Amazing sound quality, best headphones ever!",
        "Comfortable but expensive. Sound is good.",
        "Not worth the price, average at best.",
        "Love the noise cancellation! Perfect for travel.",
    ],
    "Laptop Dell XPS": [
        "Incredible performance, beautiful screen!",
        "Overheats after 30 minutes of use.",
        "Good build quality but battery life is poor.",
        "Perfect for programming, fast and reliable.",
    ],
    "Monitor LG 4K": [
        "Crystal clear image, colors are vibrant!",
        "Dead pixel after one month, disappointing.",
        "Great for gaming, 144Hz makes a difference.",
        "Excellent value for a 4K monitor.",
    ],
}

print("Sentimiento agregado por producto:")
for producto, resenas in resenas_por_producto.items():
    polaridades = []
    subjetividades = []
    for resena in resenas:
        blob = TextBlob(resena)
        polaridades.append(blob.sentiment.polarity)
        subjetividades.append(blob.sentiment.subjectivity)

    polaridad_promedio = sum(polaridades) / len(polaridades)
    subjetividad_promedio = sum(subjetividades) / len(subjetividades)
    positivas = sum(1 for p in polaridades if p > 0)
    negativas = sum(1 for p in polaridades if p < 0)
    neutras = sum(1 for p in polaridades if p == 0)

    print(f"\n  {producto}:")
    print(f"    Polaridad promedio: {polaridad_promedio:+.3f}")
    print(f"    Subjetividad promedio: {subjetividad_promedio:.3f}")
    print(f"    Distribución: {positivas} pos / {negativas} neg / {neutras} neu")
```

**Aplicación**: agregar sentimiento por producto para identificar cuáles tienen mejor/b peor percepción del cliente.

---

### 17. Correlación sentimiento vs puntuación numérica

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 17. Correlación sentimiento vs puntuación numérica

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np

analyzer = SentimentIntensityAnalyzer()

# Dataset simulado de reseñas con puntuación numérica
data = pd.DataFrame({
    'reseña': [
        "Absolutely love it! Perfect product.",
        "Good quality, satisfied with purchase.",
        "It's okay, nothing special.",
        "Not great, has some issues.",
        "Terrible, complete waste of money.",
        "Excellent! Better than expected.",
        "Average product, does what it says.",
        "Disappointed, low quality.",
        "Amazing! Highly recommend!",
        "Poor build quality, broke quickly.",
    ],
    'puntuacion': [5, 4, 3, 2, 1, 5, 3, 2, 5, 1]
})

# Calcular polaridades
data['polarity_textblob'] = data['reseña'].apply(lambda r: TextBlob(r).sentiment.polarity)
data['compound_vader'] = data['reseña'].apply(lambda r: analyzer.polarity_scores(r)['compound'])

corr_tb = data['puntuacion'].corr(data['polarity_textblob'])
corr_vader = data['puntuacion'].corr(data['compound_vader'])

print("Correlación con puntuación numérica:")
print(f"  TextBlob polarity: R = {corr_tb:.3f}")
print(f"  VADER compound: R = {corr_vader:.3f}")

print("\nTabla comparativa:")
print(data.to_string(index=False))

# Interpretación
print(f"\n{'Interpretación:':20s}")
print(f"{'TB correlación fuerte' if abs(corr_tb) > 0.7 else 'TB correlación moderada' if abs(corr_tb) > 0.4 else 'TB correlación débil'}")
print(f"{'VADER correlación fuerte' if abs(corr_vader) > 0.7 else 'VADER correlación moderada' if abs(corr_vader) > 0.4 else 'VADER correlación débil'}")
```

**Aplicación**: validar que el análisis de sentimiento se correlaciona con las puntuaciones numéricas de los usuarios.

---

### 18. Integrador: Pipeline de sentimiento para todas las reseñas

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 18. Integrador: Pipeline de sentimiento para todas las reseñas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from datetime import datetime

analyzer = SentimentIntensityAnalyzer()

# Dataset simulado de reseñas
df = pd.DataFrame({
    'fecha': ['2025-01-15', '2025-01-16', '2025-01-16', '2025-01-17', '2025-01-17',
              '2025-01-18', '2025-01-18', '2025-01-19', '2025-01-19', '2025-01-20'],
    'producto': ['Auriculares Sony', 'Auriculares Sony', 'Laptop Dell', 'Auriculares Sony',
                  'Monitor LG', 'Laptop Dell', 'Monitor LG', 'Auriculares Sony', 'Laptop Dell', 'Monitor LG'],
    'reseña': [
        "Amazing noise cancellation! Best headphones ever! 😊",
        "Good sound but uncomfortable after 1 hour.",
        "Excellent laptop, very fast and reliable.",
        "Great product, love the battery life!",
        "Colors are vibrant, perfect for design work!",
        "Overheating issue, disappointed with build quality.",
        "Dead pixel after 2 weeks, requesting refund.",
        "Comfortable and great sound. Highly recommend!",
        "Perfect for work, light and powerful.",
        "Crystal clear 4K, excellent value! 😍"
    ],
    'puntuacion': [5, 3, 5, 4, 5, 2, 1, 4, 5, 5]
})


def pipeline_sentimiento(resena):
    blob = TextBlob(resena)
    vader_scores = analyzer.polarity_scores(resena)

    # Clasificación compuesta
    compound = vader_scores['compound']
    if compound >= 0.5:
        clasificacion = "MUY POSITIVO"
    elif compound >= 0.05:
        clasificacion = "POSITIVO"
    elif compound <= -0.5:
        clasificacion = "MUY NEGATIVO"
    elif compound <= -0.05:
        clasificacion = "NEGATIVO"
    else:
        clasificacion = "NEUTRO"

    # Score combinado (0-100)
    score_combinado = (compound + 1) * 50

    return pd.Series({
        'polarity_tb': blob.sentiment.polarity,
        'subjectivity_tb': blob.sentiment.subjectivity,
        'compound_vader': compound,
        'pos_vader': vader_scores['pos'],
        'neu_vader': vader_scores['neu'],
        'neg_vader': vader_scores['neg'],
        'clasificacion': clasificacion,
        'score_combinado': round(score_combinado, 1)
    })


# Aplicar pipeline
df_resultado = df.join(df['reseña'].apply(pipeline_sentimiento))

print("Pipeline de sentimiento completo:")
print(df_resultado[['fecha', 'producto', 'reseña', 'clasificacion', 'score_combinado', 'puntuacion']].to_string(index=False))

# Resumen por producto
print("\n--- Resumen por producto ---")
resumen = df_resultado.groupby('producto').agg({
    'score_combinado': 'mean',
    'compound_vader': 'mean',
    'polarity_tb': 'mean',
    'puntuacion': 'mean',
    'clasificacion': lambda x: x.mode()[0] if not x.mode().empty else 'N/A'
}).round(3)

print(resumen.to_string())

# Calcular correlación
correlacion = df_resultado['score_combinado'].corr(df_resultado['puntuacion'])
print(f"\nCorrelación score combinado vs puntuación real: R = {correlacion:.3f}")
```

**Aplicación**: pipeline completo que procesa, clasifica y agrega sentimiento de reseñas para dashboards de customer feedback.

---

## Resumen

| Técnica | TextBlob | VADER | Aplicación |
|---|---|---|---|
| Polaridad | `sentiment.polarity` (−1 a 1) | `compound` (−1 a 1) | Medir positividad/negatividad |
| Subjetividad | `sentiment.subjectivity` (0 a 1) | No disponible | Separar hechos de opiniones |
| Noun Phrases | `.noun_phrases` | No disponible | Extraer productos y características |
| Corrección ortográfica | `.correct()` | No disponible | Limpiar reseñas con errores |
| Traducción | `.translate()` | No disponible | Multilingüe |
| Intensificadores | No | Sí (!!!) | Capturar énfasis |
| Emojis | No | Sí 😊😞 | Redes sociales, reseñas móviles |
| Negaciones | No | Sí | "no es bueno" → negativo |
| Velocidad | Lento | Muy rápido | Grandes volúmenes |

---

## Ejercicios

1. **Comparación de analizadores**: toma 20 reseñas de productos reales (o simuladas) y calcula polaridad con TextBlob y compound con VADER. ¿En cuántas difieren? ¿Por qué? Crea una tabla de contingencia.

2. **Umbrales óptimos**: determina los thresholds óptimos para clasificar reseñas en positivo/negativo/neutro con VADER. Usa 30 reseñas con etiquetado manual. Calcula precisión, recall y F1 para diferentes thresholds.

3. **Análisis temporal**: simula 30 reseñas con fechas a lo largo de 3 meses para 3 productos. Calcula el sentimiento promedio por semana y visualiza la evolución temporal. ¿Hay tendencias estacionales?

4. **Correlación con ventas**: crea un DataFrame con 10 productos, su sentimiento promedio (calculado con VADER) y su volumen de ventas semanal. Calcula la correlación entre sentimiento y ventas. Genera un scatter plot.

5. **Corrección ortográfica + sentimiento**: toma 10 reseñas con errores ortográficos, corrígelas con TextBlob.correct() y compara el sentimiento antes y después. ¿Cambia significativamente?

6. **Noun phrases para features**: de 15 descripciones de productos, extrae noun_phrases con TextBlob y agrúpalas por producto. Identifica las 10 frases nominales más comunes en el catálogo.

7. **Emojis y sentimiento**: crea un conjunto de 10 reseñas que contengan emojis (alegres, tristes, neutros). Calcula el compound de VADER y verifica que los emojis contribuyen correctamente al score.

8. **Dashboard de sentimiento**: diseña un pipeline que procese un CSV de reseñas (con columnas: fecha, producto, reseña, puntuacion) y genere: (a) clasificación por reseña, (b) sentimiento promedio por producto, (c) evolución temporal, (d) correlación con puntuación. Prueba con mínimo 20 reseñas simuladas.
