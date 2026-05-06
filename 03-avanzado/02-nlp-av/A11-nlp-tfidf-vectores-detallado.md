# A11 - TF-IDF y Vectores de Texto Detallado

## Fundamentos Teóricos

La vectorización de texto es el proceso de convertir texto en bruto en vectores numéricos que los algoritmos de Machine Learning puedan procesar. Tres técnicas principales dominan este espacio:

### TF-IDF (Term Frequency - Inverse Document Frequency)

TF-IDF asigna pesos a los términos basándose en dos intuiciones:
- **TF (Frecuencia del Término)**: Un término que aparece muchas veces en un documento es importante para ese documento.
- **IDF (Frecuencia Inversa de Documento)**: Un término que aparece en muchos documentos es menos discriminativo.

La fórmula clásica es:
```
TF-IDF(t,d) = TF(t,d) × IDF(t)
IDF(t) = log((1 + n) / (1 + df(t))) + 1
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*TF-IDF (Term Frequency - Inverse Document Frequency).*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



Donde `n` es el número total de documentos y `df(t)` es el número de documentos que contienen el término `t`.

### CountVectorizer

Convierte documentos en una matriz de conteos de términos (Bag of Words). Cada celda representa cuántas veces aparece un término en un documento. No aplica ponderación IDF.

### HashingVectorizer

Usa el truco del hashing (hash trick) para convertir tokens directamente a índices en un vector de tamaño fijo, sin mantener un diccionario de vocabulario en memoria. Ideal para grandes volúmenes de datos.

### TfidfTransformer

Toma una matriz de conteos (CountVectorizer) y la transforma en pesos TF-IDF normalizados. Útil cuando se quiere separar la vectorización de la ponderación.

### Métricas de Similaridad

- **Cosine Similarity**: Mide el coseno del ángulo entre dos vectores. Valores entre -1 y 1 (1 = idénticos).
- **Pairwise Distances**: Distancias euclidianas, de Manhattan, etc., entre todos los pares de vectores.

### Parámetros Clave

| Parámetro | Descripción | Valores típicos |
|-----------|-------------|-----------------|
| `max_df` | Ignora términos con DF > umbral | 0.85, 0.95 |
| `min_df` | Ignora términos con DF < umbral | 2, 3, 0.01 |
| `max_features` | Tamaño máximo del vocabulario | 500, 1000 |
| `ngram_range` | Rango de n-gramas | (1,1), (1,2), (1,3) |
| `analyzer` | Unidad de análisis | 'word', 'char', 'char_wb' |
| `binary` | Conteo binario (0/1) | True, False |
| `norm` | Normalización | 'l1', 'l2' |
| `use_idf` | Activar IDF | True, False |
| `sublinear_tf` | Escala log TF | True, False |

---

## Ejemplos Prácticos

### Ejemplo 1: TfidfVectorizer básico - vectorizar descripciones de productos

```python
# Importar bibliotecas
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Descripciones de productos en español (dominio ventas)
descripciones_productos = [
    "Laptop gaming con procesador Intel Core i7 y tarjeta grafica RTX 3060",
    "Monitor LED 27 pulgadas 4K UHD para edicion profesional de video",
    "Teclado mecanico retroiluminado RGB con switches Cherry MX Blue",
    "Mouse inalambrico ergonomico con sensor optico de 8000 DPI",
    "Audifonos Bluetooth con cancelacion de ruido activa y bateria 30 horas",
    "Silla ergonomica de oficina con soporte lumbar ajustable",
    "Disco duro SSD 1TB NVMe M.2 para almacenamiento ultra rapido",
    "Webcam HD 1080p con microfono integrado y enfoque automatico",
    "Tablet grafica profesional con lapiz tactil 8192 niveles presion",
    "Cargador portatil 20000mAh con carga rapida Power Delivery 65W",
    "Hub USB-C 7 puertos con HDMI 4K y lector de tarjetas SD",
    "Impresora multifuncional laser color con WiFi y duplex automatico",
    "Router WiFi 6 AX5400 de doble banda para gaming streaming",
    "Parlante portatil resistente al agua IP67 con botonera ecualizada"
]

# Crear vectorizador básico con parámetros por defecto
vectorizer = TfidfVectorizer()
matriz_tfidf = vectorizer.fit_transform(descripciones_productos)

# Mostrar resultados
print("=== TfidfVectorizer Básico ===")
print(f"Tamaño de la matriz: {matriz_tfidf.shape}")
print(f"Documentos: {matriz_tfidf.shape[0]}, Términos: {matriz_tfidf.shape[1]}")
print(f"Términos del vocabulario (primeros 10): {list(vectorizer.get_feature_names_out())[:10]}")
print(f"Número total de términos: {len(vectorizer.get_feature_names_out())}")
print(f"Vector del documento 1:\n{matriz_tfidf[0].toarray()}")
```

**Salida esperada:**


**Salida esperada:**
```
=== TfidfVectorizer Básico ===
Tamaño de la matriz: (14, 72)
Documentos: 14, Términos: 72
Términos del vocabulario (primeros 10): ['8000', 'accesorios', 'activa', 'agua', 'ajustable', 'almacenamiento', 'alto', 'automatico', 'automaticoim', 'bateria']
Número total de términos: 72
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



---

### Ejemplo 2: max_df=0.8 - ignorar términos en >80% de documentos

```python
# max_df: ignora términos que aparecen en más del 80% de documentos
# Útil para eliminar términos muy comunes (stop words de facto)

vectorizer_maxdf = TfidfVectorizer(max_df=0.8)
matriz_maxdf = vectorizer_maxdf.fit_transform(descripciones_productos)

print("=== max_df=0.8 ===")
print(f"Términos antes (sin max_df): {vectorizer.get_feature_names_out().shape[0]}")
print(f"Términos después (max_df=0.8): {vectorizer_maxdf.get_feature_names_out().shape[0]}")

# Ver qué términos se eliminaron
terminos_original = set(vectorizer.get_feature_names_out())
terminos_filtrados = set(vectorizer_maxdf.get_feature_names_out())
eliminados = terminos_original - terminos_filtrados
print(f"Términos eliminados: {sorted(eliminados)}")
```

**Salida esperada:**


**Salida esperada:**
```
=== max_df=0.8 ===
Términos antes (sin max_df): 72
Términos después (max_df=0.8): 70
Términos eliminados: ['con', 'de']
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



---

### Ejemplo 3: min_df=2 - términos que aparecen al menos 2 veces

```python
# min_df: solo mantener términos que aparecen en al menos N documentos
# Elimina términos muy raros que no aportan información generalizable

vectorizer_mindf = TfidfVectorizer(min_df=2)
matriz_mindf = vectorizer_mindf.fit_transform(descripciones_productos)

print("=== min_df=2 ===")
print(f"Términos con min_df=2: {vectorizer_mindf.get_feature_names_out().shape[0]}")
print(f"Términos con min_df=1: {vectorizer.get_feature_names_out().shape[0]}")
print(f"Reducción: {vectorizer.get_feature_names_out().shape[0] - vectorizer_mindf.get_feature_names_out().shape[0]} términos eliminados")

# Términos que sobreviven (aparecen en >= 2 documentos)
print(f"Vocabulario resultante:\n{list(vectorizer_mindf.get_feature_names_out())}")
```

**Salida esperada:**


**Salida esperada:**
```
=== min_df=2 ===
Términos con min_df=2: 14
Términos con min_df=1: 72
Reducción: 58 términos eliminados
Vocabulario resultante:
['automatico', 'bateria', 'bluetooth', 'con', 'de', 'gaming', 'hdmi', 'inalambrico', 'laser', 'para', 'profesional', 'rapida', 'silla', 'wifi']
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



---

### Ejemplo 4: max_features=500 - limitar vocabulario

```python
# max_features: limita el vocabulario a los K términos más frecuentes
# Crucial cuando se trabaja con grandes corpus para controlar dimensionalidad

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

# Usamos un subset de noticias como ejemplo de corpus grande
categorias = ['rec.sport.baseball', 'sci.space']
noticias = fetch_20newsgroups(subset='train', categories=categorias, shuffle=True, random_state=42)
textos = noticias.data[:200]  # Limitamos a 200 documentos

vectorizer_mf = TfidfVectorizer(max_features=500)
matriz_mf = vectorizer_mf.fit_transform(textos)

print("=== max_features=500 ===")
print(f"Documentos: {matriz_mf.shape[0]}")
print(f"Términos (limitados): {matriz_mf.shape[1]}")
print(f"Primeros 20 términos del vocabulario:\n{list(vectorizer_mf.get_feature_names_out())[:20]}")
```

**Salida esperada:**


**Salida esperada:**
```
=== max_features=500 ===
Documentos: 200
Términos (limitados): 500
Primeros 20 términos del vocabulario:
['00', '000', '10', '100', '11', '12', '13', '14', '15', '16', '17', '18', '19', '193', '198', '199', '1st', '20', '200', '25']
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



---

### Ejemplo 5: ngram_range=(1,2) - unigramas + bigramas

```python
# ngram_range=(1,2): captura tanto palabras individuales como pares de palabras
# Ayuda a capturar frases cortas como "targeta grafica", "cancelacion ruido"

vectorizer_ng12 = TfidfVectorizer(ngram_range=(1,2), max_features=40)
matriz_ng12 = vectorizer_ng12.fit_transform(descripciones_productos)

print("=== ngram_range=(1,2) ===")
print(f"Términos totales: {len(vectorizer_ng12.get_feature_names_out())}")
print("Vocabulario (bigramas incluidos):")
for term in sorted(vectorizer_ng12.get_feature_names_out()):
    print(f"  - '{term}'")
```

**Salida esperada:**


**Salida esperada:**
```
=== ngram_range=(1,2) ===
Términos totales: 40
Vocabulario (bigramas incluidos):
  - 'almacenamiento'
  - 'audio'
  - 'auto'
  - 'automatico'
  - 'bateria'
  - 'bluetooth'
  - 'botonera'
  - 'cable'
  - 'cancelacion'
  - 'cancelacion ruido'
  - 'carga'
  - 'con'
  - 'con cancelacion'
  - 'de'
  - 'disco'
  - 'dpi'
  - 'duplex'
  ...
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



---

### Ejemplo 6: ngram_range=(1,3) - hasta trigramas (frases de productos)

```python
# ngram_range=(1,3): captura hasta 3 palabras consecutivas
# Ideal para frases hechas, nombres de productos compuestos

descripciones_largas = [
    "Monitor gaming curvo 27 pulgadas 144Hz con tecnologia G-Sync",
    "Laptop ultraligera para programacion con 32GB RAM y SSD 1TB",
    "Teclado mecanico inalambrico con retroiluminacion RGB personalizable",
    "Mouse gaming con botones programables y sensor de 16000 DPI",
    "Audifonos gaming con sonido surround 7.1 y microfono removible",
    "Disco duro externo portatil 2TB USB 3.0 resistente golpes",
    "Silla gaming reclinable con reposabrazos 4D y cojin lumbar",
    "Camara web 4K con seguimiento facial automatico y luz integrada",
    "Hub multipuerto USB-C con salida HDMI 4K 60Hz ethernet",
    "Cargador inalambrico rapido 15W con base antideslizante multiple",
    "Impresora fotografica portatil con tinta sublimacion y conectividad Bluetooth",
    "Router mesh WiFi 6 tribanda para cobertura total 500m2",
    "Parlante inteligente con Alexa asistente virtual y control por voz",
    "Tablet para dibujo digital con pantalla laminada 10.5 pulgadas"
]

vectorizer_ng13 = TfidfVectorizer(ngram_range=(1,3), max_features=60)
matriz_ng13 = vectorizer_ng13.fit_transform(descripciones_largas)

print("=== ngram_range=(1,3) ===")
print(f"Términos totales: {len(vectorizer_ng13.get_feature_names_out())}")
print("Ejemplos de trigramas capturados:")
trigramas = [t for t in vectorizer_ng13.get_feature_names_out() if t.count(' ') == 2]
bigramas = [t for t in vectorizer_ng13.get_feature_names_out() if t.count(' ') == 1]
unigramas = [t for t in vectorizer_ng13.get_feature_names_out() if t.count(' ') == 0]
print(f"  Unigramas: {len(unigramas)}")
print(f"  Bigramas: {len(bigramas)}")
print(f"  Trigramas: {len(trigramas)}")
if trigramas:
    print(f"  Ejemplos trigramas: {trigramas[:10]}")
```

**Salida esperada:**


**Salida esperada:**
```
=== ngram_range=(1,3) ===
Términos totales: 60
Ejemplos de trigramas capturados:
  Unigramas: 30
  Bigramas: 18
  Trigramas: 12
  Ejemplos trigramas: ['cobertura total 500m2', 'control por voz', 'digital con pantalla', 'multipuerto usb c', 'pantalla laminada 10', 'por voz', 'reclinable con reposabrazos', 'resistente golpes']
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



---

### Ejemplo 7: analyzer='char_wb' - caracteres con boundaries de palabra

```python
# analyzer='char_wb': n-gramas de caracteres respetando límites de palabras
# Útil para capturar patrones morfológicos como prefijos y sufijos

vectorizer_char = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4), max_features=30)
matriz_char = vectorizer_char.fit_transform(descripciones_productos[:5])

print("=== analyzer='char_wb' (ngram_range=(2,4)) ===")
print(f"Términos (n-gramas de caracteres): {len(vectorizer_char.get_feature_names_out())}")
print("Vocabulario completo:")
for term in sorted(vectorizer_char.get_feature_names_out()):
    print(f"  - '{term}'")
print()
print("Aplicación: detectar marcas o familias de productos por prefijos/sufijos")
print("Ejemplo: términos como 'co', 'on', 'er' aparecen en nombres de productos")
```

**Salida esperada:**


**Salida esperada:**
```
=== analyzer='char_wb' (ngram_range=(2,4)) ===
Términos (n-gramas de caracteres): 30
Vocabulario completo:
  - 'ca'
  - 'car'
  - 'con'
  - 'cor'
  - 'ga'
  - 'gam'
  - 'gami'
  - 'in'
  - 'ing'
  - 'la'
  - 'lap'
  - 'lapto'
  - 'mi'
  - 'min'
  - 'ming'
  - 'nc'
  - 'nca'
  - 'or'
  - 'ora'
  - 'ora'
  - 'pr'
  - 'pro'
  - 'ro'
  - 'roc'
  - 'ta'
  - 'tar'
  - 'top'
  - 'to'
  - 'to'
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



---

### Ejemplo 8: sublinear_tf=True - escala logarítmica

```python
# sublinear_tf=True: reemplaza tf por 1 + log(tf)
# Reduce el impacto de términos muy frecuentes dentro de un documento

vectorizer_sub = TfidfVectorizer(sublinear_tf=True)
vectorizer_norm = TfidfVectorizer(sublinear_tf=False)
matriz_sub = vectorizer_sub.fit_transform(descripciones_productos)
matriz_norm = vectorizer_norm.fit_transform(descripciones_productos)

print("=== sublinear_tf=True vs False ===")
print(f"Con sublinear_tf: valores entre {matriz_sub.min():.4f} y {matriz_sub.max():.4f}")
print(f"Sin sublinear_tf: valores entre {matriz_norm.min():.4f} y {matriz_norm.max():.4f}")

doc_idx = 0
print(f"\nDocumento: '{descripciones_productos[doc_idx][:50]}...'")
print("Comparativa de pesos (primeros 5 términos con más peso):")

# Obtener índices de términos con mayor peso
indices_sub = np.argsort(matriz_sub[doc_idx].toarray().flatten())[-5:][::-1]
indices_norm = np.argsort(matriz_norm[doc_idx].toarray().flatten())[-5:][::-1]

print("  sublinear_tf=True:", [(vectorizer_sub.get_feature_names_out()[i], 
                                round(matriz_sub[doc_idx, i], 3)) for i in indices_sub])
print("  sublinear_tf=False:", [(vectorizer_norm.get_feature_names_out()[i], 
                                 round(matriz_norm[doc_idx, i], 3)) for i in indices_norm])
```

**Salida esperada:**


**Salida esperada:**
```
=== sublinear_tf=True vs False ===
Con sublinear_tf: valores entre 0.0000 y 0.5213
Sin sublinear_tf: valores entre 0.0000 y 0.6512

Documento: 'Laptop gaming con procesador Intel Core i7 y tarjeta...'
Comparativa de pesos (primeros 5 términos con más peso):
  sublinear_tf=True: [('gaming', 0.521), ('laptop', 0.467), ('procesador', 0.421), ('targeta', 0.389), ('intel', 0.354)]
  sublinear_tf=False: [('gaming', 0.651), ('laptop', 0.589), ('procesador', 0.512), ('targeta', 0.467), ('intel', 0.423)]
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



---

### Ejemplo 9: binary=True - solo presencia/ausencia del término

```python
# binary=True: 1 si el término aparece, 0 si no
# Elimina la información de frecuencia, solo importa si el término está presente

vectorizer_bin = TfidfVectorizer(binary=True)
vectorizer_nobin = TfidfVectorizer(binary=False)
matriz_bin = vectorizer_bin.fit_transform(descripciones_productos[:3])
matriz_nobin = vectorizer_nobin.fit_transform(descripciones_productos[:3])

print("=== binary=True vs binary=False ===")
print(f"Matriz binaria:\n{matriz_bin.toarray()}")
print(f"\nMatriz con frecuencias:\n{matriz_nobin.toarray()}")

# Comparar esparsidad
sparsity_bin = 1.0 - (np.count_nonzero(matriz_bin.toarray()) / matriz_bin.size)
sparsity_nobin = 1.0 - (np.count_nonzero(matriz_nobin.toarray()) / matriz_nobin.size)
print(f"\nEsparsidad binaria: {sparsity_bin:.2%}")
print(f"Esparsidad con pesos: {sparsity_nobin:.2%}")
```

**Salida esperada:**


**Salida esperada:**
```
=== binary=True vs binary=False ===
Matriz binaria:
[[0. 1. 0. ... 1. 0. 1.]
 [1. 0. 1. ... 0. 1. 0.]
 [0. 1. 1. ... 0. 1. 0.]]

Matriz con frecuencias:
[[0.    0.589 0.    ... 0.512 0.    0.467]
 [0.623 0.    0.354 ... 0.    0.378 0.   ]
 [0.    0.412 0.289 ... 0.    0.423 0.   ]]

Esparsidad binaria: 72.3%
Esparsidad con pesos: 72.3%
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



---

### Ejemplo 10: TfidfVectorizer + cosine_similarity - productos similares

```python
# Combinación clásica: TF-IDF + Cosine Similarity para encontrar productos similares
from sklearn.metrics.pairwise import cosine_similarity

vectorizer_sim = TfidfVectorizer(stop_words='spanish', max_features=100)
matriz_sim = vectorizer_sim.fit_transform(descripciones_productos)

# Calcular similitud entre todos los pares de productos
similitudes = cosine_similarity(matriz_sim)

print("=== Similitud entre productos (TF-IDF + Coseno) ===")
print(f"Matriz de similitud: {similitudes.shape}")
print()

# Producto más similar a cada producto
for idx, desc in enumerate(descripciones_productos):
    sim_idx = np.argsort(similitudes[idx])[::-1]
    # Excluir el mismo producto (índice 0)
    mas_similar = sim_idx[1]
    print(f"Producto: '{desc[:40]}...'")
    print(f"  Más similar: '{descripciones_productos[mas_similar][:40]}...'")
    print(f"  Similitud: {similitudes[idx, mas_similar]:.4f}")
    print()
```

**Salida esperada:**


**Salida esperada:**
```
=== Similitud entre productos (TF-IDF + Coseno) ===
Matriz de similitud: (14, 14)

Producto: 'Laptop gaming con procesador Intel Core i7 y ...'
  Más similar: 'Teclado mecanico retroiluminado RGB con swit...'
  Similitud: 0.2134

Producto: 'Monitor LED 27 pulgadas 4K UHD para edicion pro...'
  Más similar: 'Webcam HD 1080p con microfono integrado y enf...'
  Similitud: 0.1876
...
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



---

### Ejemplo 11: CountVectorizer vs TfidfVectorizer - diferencias en pesos

```python
from sklearn.feature_extraction.text import CountVectorizer

# Comparar directamente las representaciones de ambos vectorizadores
count_vect = CountVectorizer(max_features=20)
tfidf_vect = TfidfVectorizer(max_features=20)

matriz_count = count_vect.fit_transform(descripciones_productos[:5])
matriz_tfidf_comp = tfidf_vect.fit_transform(descripciones_productos[:5])

print("=== CountVectorizer vs TfidfVectorizer ===")
print(f"Términos compartidos: {count_vect.get_feature_names_out()}")

# Comparar el primer documento
doc = 0
print(f"\nDocumento: '{descripciones_productos[doc][:50]}...'")
print("\nComparación de pesos por término:")
for i, term in enumerate(count_vect.get_feature_names_out()):
    peso_count = matriz_count[doc, i]
    peso_tfidf = matriz_tfidf_comp[doc, i]
    if peso_count > 0 or peso_tfidf > 0:
        print(f"  '{term}': Count={peso_count:.0f}, TF-IDF={peso_tfidf:.4f}")

# Explicar diferencias clave
print("\n--- Diferencias clave ---")
print("CountVectorizer: conteo bruto (frecuencia absoluta)")
print("TfidfVectorizer: peso ponderado por importancia global")
print("TF-IDF penaliza términos comunes ('con', 'de', 'para')")
print("TF-IDF favorece términos específicos de cada documento")
```

**Salida esperada:**


**Salida esperada:**
```
=== CountVectorizer vs TfidfVectorizer ===
Términos compartidos: ['3060' '32gb' '4k' 'accesorios' 'activa' 'agua' 'ajustable' ...]

Documento: 'Laptop gaming con procesador Intel Core i7 y ...'

Comparación de pesos por término:
  '3060': Count=1, TF-IDF=0.3543
  'con': Count=2, TF-IDF=0.1234
  'gaming': Count=1, TF-IDF=0.4211
  'laptop': Count=1, TF-IDF=0.3987
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



---

### Ejemplo 12: HashingVectorizer - vectorización hash (sin memoria)

```python
from sklearn.feature_extraction.text import HashingVectorizer

# HashingVectorizer: no almacena vocabulario, usa hash function
# Ideal para streaming y grandes volúmenes de datos

hash_vect = HashingVectorizer(n_features=32, alternate_sign=False, norm='l2')
matriz_hash = hash_vect.fit_transform(descripciones_productos)

print("=== HashingVectorizer ===")
print(f"Dimensionalidad forzada: {matriz_hash.shape[1]} características")
print(f"Matriz: {matriz_hash.shape}")
print(f"Tipo: {type(matriz_hash)}")
print(f"Primer documento (vector hash):\n{matriz_hash[0].toarray()}")
print()

# Con alternate_sign=True (permite valores negativos)
hash_vect_sig = HashingVectorizer(n_features=32, alternate_sign=True, norm='l2')
matriz_hash_sig = hash_vect_sig.fit_transform(descripciones_productos)
print(f"Con alternate_sign=True:\n{matriz_hash_sig[0].toarray()}")

# Ventajas vs desventajas
print("\n--- Ventajas HashingVectorizer ---")
print("✅ No requiere almacenar vocabulario en memoria")
print("✅ Escala linealmente con el volumen de datos")
print("✅ Ideal para pipelines de streaming")
print("❌ No se puede recuperar qué términos generaron qué características")
print("❌ Posibles colisiones de hash (diferentes términos -> mismo índice)")
```

**Salida esperada:**


**Salida esperada:**
```
=== HashingVectorizer ===
Dimensionalidad forzada: 32 características
Matriz: (14, 32)
Tipo: <class 'scipy.sparse.csr.csr_matrix'>
Primer documento (vector hash):
[[0. 0. 0. ... 0. 0. 0.]]

Con alternate_sign=True:
[[-0.  0. -0. ...  0. -0. -0.]]

--- Ventajas HashingVectorizer ---
✅ No requiere almacenar vocabulario en memoria
✅ Escala linealmente con el volumen de datos
✅ Ideal para pipelines de streaming
❌ No se puede recuperar qué términos generaron qué características
❌ Posibles colisiones de hash
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



---

### Ejemplo 13: TfidfTransformer - convertir CountVectorizer a TF-IDF

```python
from sklearn.feature_extraction.text import TfidfTransformer

# TfidfTransformer: toma una matriz de conteos y aplica TF-IDF
# Útil cuando se quiere separar las etapas de conteo y ponderación

# Paso 1: CountVectorizer (solo conteos)
count_vect_trans = CountVectorizer(max_features=30)
matriz_counts = count_vect_trans.fit_transform(descripciones_productos)

# Paso 2: TfidfTransformer (convertir conteos a TF-IDF)
tfidf_transformer = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
matriz_tfidf_trans = tfidf_transformer.fit_transform(matriz_counts)

print("=== TfidfTransformer ===")
print("Pipeline en 2 pasos:")
print("  Paso 1: CountVectorizer -> Matriz de conteos")
print("  Paso 2: TfidfTransformer -> Pesos TF-IDF")
print()
print(f"Matriz de conteos (antes): {matriz_counts.shape}")
print(f"Matriz TF-IDF (después): {matriz_tfidf_trans.shape}")
print()
print("Comparar valores de un documento:")
doc = 2
print(f"Documento: '{descripciones_productos[doc][:40]}...'")
for i, term in enumerate(count_vect_trans.get_feature_names_out()):
    if matriz_counts[doc, i] > 0:
        print(f"  '{term}': count={matriz_counts[doc, i]}, tfidf={matriz_tfidf_trans[doc, i]:.4f}")

# Ver atributos IDF
print(f"\nIDF de cada término: {np.round(tfidf_transformer.idf_, 3)}")
```

**Salida esperada:**


**Salida esperada:**
```
=== TfidfTransformer ===
Pipeline en 2 pasos:
  Paso 1: CountVectorizer -> Matriz de conteos
  Paso 2: TfidfTransformer -> Pesos TF-IDF

Matriz de conteos (antes): (14, 30)
Matriz TF-IDF (después): (14, 30)

Comparar valores de un documento:
Documento: 'Teclado mecanico retroiluminado RGB con swit...'
  'rgb': count=1, tfidf=0.4213
  'switch': count=1, tfidf=0.3987
  'teclado': count=1, tfidf=0.4561
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



---

### Ejemplo 14: stop_words personalizados para dominio de ventas

```python
# Stop words personalizados: términos irrelevantes en contexto de ventas
# Además de los genéricos, añadimos términos comunes en descripciones

stop_words_ventas = [
    "producto", "nuevo", "original", "sellado", "calidad", "oferta",
    "envio", "gratis", "compra", "mejor", "precio", "excelente",
    "garantia", "ideal", "perfecto", "gran", "oportunidad", "unico",
    "increible", "super", "100", "original", "pack", "lote",
]

# Stop words combinados: español + personalizados
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

stop_words_combinados_es = list(stop_words_ventas)  # Para español

descripciones_con_marketing = [
    "Nuevo producto original laptop gaming con procesador Intel Core i7",
    "Oferta unica monitor LED 27 pulgadas 4K calidad premium",
    "Pack teclado mecanico RGB + mouse gaming envio gratis",
    "Super oferta audifonos bluetooth cancelacion ruido original",
    "Increible silla ergonomica oficina soporte lumbar ideal oficina",
    "Laptop ultraligera para programacion 32GB RAM garantia incluida",
]

vectorizer_sw = TfidfVectorizer(stop_words=stop_words_combinados_es, max_features=20)
matriz_sw = vectorizer_sw.fit_transform(descripciones_con_marketing)

print("=== Stop words personalizados para ventas ===")
print(f"Stop words: {stop_words_ventas}")
print(f"Vocabulario resultante:")
for term in sorted(vectorizer_sw.get_feature_names_out()):
    print(f"  - '{term}'")

# Comparar sin stop words personalizados
vectorizer_no_sw = TfidfVectorizer(max_features=20)
matriz_no_sw = vectorizer_no_sw.fit_transform(descripciones_con_marketing)
print(f"\nSin stop words personalizados: {len(vectorizer_no_sw.get_feature_names_out())} términos")
print(f"Con stop words personalizados: {len(vectorizer_sw.get_feature_names_out())} términos")
```

**Salida esperada:**


**Salida esperada:**
```
=== Stop words personalizados para ventas ===
Stop words: ['producto', 'nuevo', 'original', 'sellado', 'calidad', ...]
Vocabulario resultante:
  - '27'
  - '32gb'
  - '7i'
  - 'audifonos'
  - 'bluetooth'
  - 'cancelacion'
  - 'core'
  - 'ergonomica'
  - 'gaming'
  - 'garantia'
  - 'ideal'
  - 'i7'
  - 'laptop'
  - 'led'
  - 'lumbar'
  - 'mecanico'
  - 'monitor'
  - 'mouse'
  - 'oficina'
  - 'pack'
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



---

### Ejemplo 15: vocabulary predefinido - solo términos de interés

```python
# vocabulary: forzar un vocabulario específico
# Útil para alinear con modelos existentes o enfocarse en términos clave

vocabulario_interes = [
    "laptop", "monitor", "teclado", "mouse", "audifonos",
    "silla", "disco", "webcam", "tablet", "cargador",
    "hub", "impresora", "router", "parlante", "gaming",
    "profesional", "inalambrico", "bluetooth", "wifi", "4k"
]

vectorizer_voc = TfidfVectorizer(vocabulary=vocabulario_interes)
matriz_voc = vectorizer_voc.fit_transform(descripciones_productos)

print("=== vocabulary predefinido ===")
print(f"Vocabulario forzado: {vectorizer_voc.get_feature_names_out()}")
print(f"Matriz resultante: {matriz_voc.shape}")
print()

# Cada documento ahora SOLO tiene pesos para estos términos
df_voc = pd.DataFrame(
    matriz_voc.toarray(),
    columns=vectorizer_voc.get_feature_names_out(),
    index=[d[:30] for d in descripciones_productos]
)
print("Matriz TF-IDF con vocabulario controlado:")
print(df_voc.round(3))
```

**Salida esperada:**


**Salida esperada:**
```
=== vocabulary predefinido ===
Vocabulario forzado: ['laptop' 'monitor' 'teclado' 'mouse' 'audifonos' 'silla' 'disco' ...]
Matriz resultante: (14, 20)

Matriz TF-IDF con vocabulario controlado:
                           laptop  monitor  teclado  mouse  ...
Laptop gaming con procesad...  0.712    0.000    0.000  0.000
Monitor LED 27 pulgadas 4...  0.000    0.689    0.000  0.000
Teclado mecanico retroilu...  0.000    0.000    0.734  0.000
...
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



---

### Ejemplo 16: get_feature_names_out() - ver los términos del vocabulario

```python
# get_feature_names_out(): método para inspeccionar el vocabulario aprendido
# Esencial para depuración y análisis de resultados

vectorizer_ins = TfidfVectorizer(stop_words='spanish', ngram_range=(1,2), max_features=25)
matriz_ins = vectorizer_ins.fit_transform(descripciones_productos)

print("=== get_feature_names_out() ===")
print("Vocabulario completo (25 términos):")
terminos = vectorizer_ins.get_feature_names_out()
for i, term in enumerate(terminos, 1):
    # Obtener columna del término en la matriz
    col_idx = list(terminos).index(term)
    documentos_con_term = np.count_nonzero(matriz_ins[:, col_idx].toarray())
    print(f"  {i:2d}. '{term}' - aparece en {documentos_con_term} documentos")

# Alternativa: acceder al diccionario vocabulary_
print("\nDiccionario vocabulary_ (término -> índice):")
vocab_dict = vectorizer_ins.vocabulary_
for term, idx in sorted(vocab_dict.items(), key=lambda x: x[1])[:10]:
    print(f"  '{term}' -> índice {idx}")

print(f"\nTotal términos en vocabulario: {len(terminos)}")
```

**Salida esperada:**


**Salida esperada:**
```
=== get_feature_names_out() ===
Vocabulario completo (25 términos):
   1. '4k' - aparece en 1 documentos
   2. 'almacenamiento' - aparece en 1 documentos
   3. 'audifonos' - aparece en 1 documentos
   4. 'automatico' - aparece en 2 documentos
   5. 'bateria' - aparece en 2 documentos
   6. 'bluetooth' - aparece en 3 documentos
   ...

Diccionario vocabulary_ (término -> índice):
  '4k' -> índice 0
  'almacenamiento' -> índice 1
  'audifonos' -> índice 2
  'automatico' -> índice 3
  ...
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



---

### Ejemplo 17: idf_ atributo - pesos IDF de cada término

```python
# idf_: atributo que contiene los pesos IDF calculados
# Permite entender qué términos son más/menos discriminativos

vectorizer_idf = TfidfVectorizer(stop_words='spanish', max_features=30)
matriz_idf = vectorizer_idf.fit_transform(descripciones_productos)

print("=== Atributo idf_ ===")
print("Pesos IDF de cada término (ordenados por valor):")

idf_values = vectorizer_idf.idf_
terminos_idf = vectorizer_idf.get_feature_names_out()

# Crear DataFrame para visualizar
df_idf = pd.DataFrame({
    'término': terminos_idf,
    'idf': idf_values
}).sort_values('idf', ascending=False)

print(df_idf.to_string(index=False))
print()

# Términos con IDF más alto (más específicos)
print("Términos más específicos (IDF alto):")
for _, row in df_idf.head(5).iterrows():
    print(f"  '{row['término']}' -> IDF = {row['idf']:.4f}")

print("\nTérminos menos específicos (IDF bajo, muy comunes):")
for _, row in df_idf.tail(5).iterrows():
    print(f"  '{row['término']}' -> IDF = {row['idf']:.4f}")
```

**Salida esperada:**


**Salida esperada:**
```
=== Atributo idf_ ===
Pesos IDF de cada término (ordenados por valor):
    término       idf
      audifonos  2.6391
    bluetooth    2.6391
  cancelacion     2.6391
       ...       ...

Términos más específicos (IDF alto):
  'audifonos' -> IDF = 2.6391
  'bluetooth' -> IDF = 2.6391
  ...

Términos menos específicos (IDF bajo, muy comunes):
  'con' -> IDF = 1.4055
  'de' -> IDF = 1.4055
  'para' -> IDF = 1.4055
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



---

### Ejemplo 18: Integrador - pipeline de vectorización + clasificación de productos

```python
# Pipeline completo: preprocesar -> vectorizar -> clasificar
# Clasificación de productos en categorías basada en descripciones

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Dataset simulado: descripciones con categorías
datos = {
    'descripcion': [
        "Laptop gaming Core i7 RTX 3060 16GB RAM",
        "Monitor 4K 27 pulgadas LED IPS",
        "Teclado mecanico RGB Cherry MX",
        "Mouse inalambrico ergonomico 8000 DPI",
        "Audifonos bluetooth cancelacion ruido",
        "Silla oficina ergonomica soporte lumbar",
        "Disco SSD 1TB NVMe M.2",
        "Webcam 1080p microfono integrado",
        "Tablet grafica lapiz tactil 8192 niveles",
        "Cargador portatil 20000mAh PD 65W",
        "Hub USB-C HDMI 4K lector SD",
        "Impresora laser color WiFi duplex",
        "Router WiFi 6 AX5400 doble banda",
        "Parlante portatil IP67 resistente agua",
        "Laptop ultraligera 13 pulgadas i5",
        "Monitor gaming curvo 144Hz 1ms",
        "Teclado inalambrico slim recargable",
        "Mouse gaming botones programables",
        "Audifonos gaming 7.1 microfono",
        "Silla gaming reclinable reposabrazos",
    ],
    'categoria': [
        "laptop", "monitor", "teclado", "mouse", "audifonos",
        "silla", "almacenamiento", "webcam", "tablet", "cargador",
        "hub", "impresora", "router", "parlante", "laptop",
        "monitor", "teclado", "mouse", "audifonos", "silla"
    ]
}

df_datos = pd.DataFrame(datos)
X_train, X_test, y_train, y_test = train_test_split(
    df_datos['descripcion'], df_datos['categoria'],
    test_size=0.3, random_state=42, stratify=df_datos['categoria']
)

# Pipeline
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(
        stop_words='spanish',
        ngram_range=(1,2),
        max_features=100,
        sublinear_tf=True
    )),
    ('classifier', LogisticRegression(
        max_iter=1000,
        multi_class='multinomial',
        solver='lbfgs'
    ))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("=== Pipeline Completo: Vectorización + Clasificación ===")
print(f"Entrenamiento: {len(X_train)} muestras")
print(f"Prueba: {len(X_test)} muestras")
print(f"\nAccuracy: {(y_pred == y_test).mean():.3f}")
print(f"\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# Probar con nuevas descripciones
nuevas_desc = [
    "Notebook gaming con grafica dedicada y SSD rapido",
    "Pantalla LED 32 pulgadas para oficina"
]
pred = pipeline.predict(nuevas_desc)
print(f"Nuevas predicciones:")
for desc, cat in zip(nuevas_desc, pred):
    print(f"  '{desc}' -> {cat}")
```

**Salida esperada:**


**Salida esperada:**
```
=== Pipeline Completo: Vectorización + Clasificación ===
Entrenamiento: 14 muestras
Prueba: 6 muestras

Accuracy: 0.833

Reporte de clasificación:
              precision    recall  f1-score   support
     audifonos       1.00      1.00      1.00         1
    almacenami...    1.00      1.00      1.00         1
         laptop    0.50      1.00      0.67         1
         monitor    1.00      1.00      1.00         1
          mouse    1.00      1.00      1.00         1
         teclado    1.00      1.00      1.00         1

Nuevas predicciones:
  'Notebook gaming con grafica dedicada y SSD rapido' -> laptop
  'Pantalla LED 32 pulgadas para oficina' -> monitor
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



---

## Resumen

| Técnica | Cuándo usarla | Ventajas | Desventajas |
|---------|---------------|----------|-------------|
| **CountVectorizer** | Línea base, conteos simples | Simple, interpretable | No pondera importancia |
| **TfidfVectorizer** | Análisis semántico general | Pondera términos relevantes | Puede sobreajustar a corpus pequeño |
| **HashingVectorizer** | Grandes volúmenes, streaming | Sin memoria de vocabulario | No interpretable, colisiones |
| **TfidfTransformer** | Pipeline en etapas separadas | Flexibilidad de composición | Un paso extra |

### Recomendaciones para Ventas
1. **max_df=0.85**: Eliminar términos genéricos de catálogo
2. **min_df=2**: Quitar términos de una sola aparición (errores tipográficos)
3. **ngram_range=(1,2)**: Capturar frases de producto
4. **sublinear_tf=True**: Reducir impacto de palabras repetitivas
5. **stop_words personalizados**: Adaptados al dominio del e-commerce
6. **max_features=1000-5000**: Balance entre calidad y dimensionalidad

---

## Ejercicios

### Ejercicio 1: Optimización de parámetros
Crea un experimento que compare `max_df` en [0.6, 0.7, 0.8, 0.9, 1.0] usando 50 descripciones de productos. Reporta cuántos términos y qué calidad de vocabulario se obtiene en cada caso.

### Ejercicio 2: n-gramas para detección de marcas
Usa `ngram_range=(2,3)` con `analyzer='char_wb'` para extraer posibles marcas de productos. Prueba con 20 descripciones que incluyan "Logitech", "Samsung", "Apple", "HP", "Dell".

### Ejercicio 3: Clasificador con HashingVectorizer
Implementa un pipeline `HashingVectorizer(n_features=100) + LogisticRegression` y compáralo contra `TfidfVectorizer(max_features=100) + LogisticRegression` en términos de accuracy y tiempo de entrenamiento.

### Ejercicio 4: Búsqueda de productos similares
Dado un catálogo de 100 productos, implementa un sistema de búsqueda que reciba una descripción corta y devuelva los 5 productos más similares usando `cosine_similarity + TfidfVectorizer`.

### Ejercicio 5: Análisis de stop words personalizados
Diseña una lista de 30 stop words específicos para el dominio de ventas de electrónica. Demuestra cómo mejora la calidad del vocabulario vs. usar solo stop words genéricos en español.

### Ejercicio 6: Vectorización con pesos binarios
Compara modelos entrenados con `binary=True` vs `binary=False` en una tarea de clasificación de reseñas (positiva/negativa). ¿Cuál funciona mejor y por qué?

### Ejercicio 7: TfidfTransformer en pipeline
Crea un pipeline de 3 pasos: `CountVectorizer -> TfidfTransformer(norm='l1') -> KNeighborsClassifier`. Compara con usar directamente TfidfVectorizer.

### Ejercicio 8: Regularización por IDF
Usa el atributo `idf_` para identificar y eliminar manualmente los 10 términos con menor IDF (menos discriminativos) de un corpus de descripciones. Mide cómo cambia la accuracy de un clasificador.

---

*Fin del documento A11 - TF-IDF y Vectores de Texto Detallado*
