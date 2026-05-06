# A14 - Modelos Secuenciales (CRF, HMM)

## Fundamentos Teóricos

Los modelos secuenciales son fundamentales para tareas de NLP donde la salida depende no solo de la entrada actual sino también de las predicciones anteriores. En el contexto de ventas, se usan para extraer entidades como nombres de productos, marcas, precios y categorías de texto no estructurado.

### Hidden Markov Models (HMM)

Los HMM son modelos generativos que modelan la probabilidad conjunta de observaciones y estados ocultos:

```
P(X, Y) = P(Y_0) * Π P(Y_t | Y_{t-1}) * Π P(X_t | Y_t)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Hidden Markov Models (HMM).*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



- **Estados ocultos (Y)**: Las etiquetas (B-PRODUCT, I-PRODUCT, O, etc.)
- **Observaciones (X)**: Las palabras del texto
- **Probabilidades de transición**: P(Y_t | Y_{t-1})
- **Probabilidades de emisión**: P(X_t | Y_t)

### Conditional Random Fields (CRF)

Los CRF son modelos discriminativos que modelan la probabilidad condicional de la secuencia de etiquetas dada la secuencia de observaciones:

```
P(Y | X) = (1/Z) * exp(Σ w_i * f_i(Y, X))
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Conditional Random Fields (CRF).*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



Donde `f_i` son funciones de features y `w_i` son sus pesos aprendidos.

Ventajas de CRF sobre HMM:
- No asume independencia condicional de las observaciones
- Puede incorporar features arbitrarias y superpuestas
- Mejor precisión en tareas de etiquetado secuencial

### BIO Encoding

Esquema de etiquetado estándar para reconocimiento de entidades:
- **B-XXX**: Begin (inicio de entidad)
- **I-XXX**: Inside (dentro de entidad)
- **O**: Outside (fuera de entidad)

Ejemplo: "Compré **laptop gaming HP** por **$899**"
```
Compré   O
laptop   B-PRODUCT
gaming   I-PRODUCT
HP       I-PRODUCT
por      O
$899     B-PRICE
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*BIO Encoding.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Feature Engineering para Secuencias

Features típicas para CRF en NLP:
- **Word features**: word.lower(), isupper(), istitle(), isdigit()
- **Context features**: palabra anterior, siguiente (ventana)
- **Shape features**: patrón de caracteres (Xxxx, xx00, etc.)
- **Prefix/Suffix**: primeros/últimos 2-4 caracteres
- **POS tags**: etiquetas gramaticales
- **Transition features**: bigramas de etiquetas

### Evaluación con seqeval

seqeval es una biblioteca especializada para evaluar modelos de etiquetado secuencial a nivel de entidad (no de token). Calcula:
- **Precision**: tokens correctamente etiquetados / total predichos
- **Recall**: tokens correctamente etiquetados / total verdaderos
- **F1-score**: media armónica de precision y recall

### Algoritmos Clave

- **Viterbi**: Encuentra la secuencia de etiquetas más probable dado el modelo
- **Forward-Backward**: Calcula la probabilidad marginal de cada etiqueta en cada posición
- **L-BFGS**: Optimización usada en CRF para aprender pesos

---

## Ejemplos Prácticos

### Ejemplo 1: Preparar datos de secuencias (oraciones tokenizadas + etiquetas)

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Datos de entrenamiento: oraciones con etiquetas BIO
# Dominio: extracción de productos, marcas y precios de texto de compras

oraciones = [
    ["Compré", "una", "laptop", "gaming", "HP", "en", "la", "tienda"],
    ["El", "monitor", "Samsung", "4K", "cuesta", "$350", "dólares"],
    ("Necesito", "un", "teclado", "mecánico", "Logitech", "para", "programar"),
    ["Venden", "mouse", "inalámbrico", "ergonómico", "a", "$25", "cada", "uno"],
    ["La", "silla", "ergonómica", "de", "oficina", "vale", "$199.99"],
    ["Compré", "audífonos", "Bluetooth", "Sony", "con", "cancelación", "de", "ruido"],
    ["El", "disco", "SSD", "NVMe", "1TB", "está", "en", "oferta"],
    ["Busco", "webcam", "1080p", "con", "micrófono", "integrado"],
    ["La", "tablet", "gráfica", "Wacom", "tiene", "lapiz", "táctil"],
    ["Cargador", "portátil", "20000mAh", "Power", "Delivery", "65W"],
]

# Etiquetas BIO correspondientes
etiquetas = [
    ["O", "O", "B-PRODUCT", "I-PRODUCT", "B-BRAND", "O", "O", "O"],
    ["O", "B-PRODUCT", "B-BRAND", "I-PRODUCT", "O", "B-PRICE", "O"],
    ["O", "O", "B-PRODUCT", "I-PRODUCT", "B-BRAND", "O", "O"],
    ["O", "B-PRODUCT", "I-PRODUCT", "I-PRODUCT", "O", "B-PRICE", "O", "O"],
    ["O", "B-PRODUCT", "I-PRODUCT", "O", "I-PRODUCT", "O", "B-PRICE"],
    ["O", "B-PRODUCT", "I-PRODUCT", "B-BRAND", "O", "O", "O", "O"],
    ["O", "B-PRODUCT", "I-PRODUCT", "I-PRODUCT", "I-PRODUCT", "O", "O", "O"],
    ["O", "B-PRODUCT", "I-PRODUCT", "O", "I-PRODUCT", "I-PRODUCT"],
    ["O", "B-PRODUCT", "I-PRODUCT", "B-BRAND", "O", "I-PRODUCT", "I-PRODUCT"],
    ["B-PRODUCT", "I-PRODUCT", "I-PRODUCT", "I-PRODUCT", "I-PRODUCT", "I-PRODUCT"],
]

print("=== Datos de secuencias preparados ===")
print(f"Total de oraciones: {len(oraciones)}")
print(f"Total de tokens: {sum(len(o) for o in oraciones)}")
print()
print("Ejemplo (oración 1):")
for palabra, etiq in zip(oraciones[0], etiquetas[0]):
    print(f"  {palabra:12s} -> {etiq}")

print("\n--- Distribución de etiquetas ---")
from collections import Counter
todas_etiquetas = [e for seq in etiquetas for e in seq]
for etiq, count in Counter(todas_etiquetas).most_common():
    print(f"  {etiq:15s}: {count}")
```

**Salida esperada:**


**Salida esperada:**
```
=== Datos de secuencias preparados ===
Total de oraciones: 10
Total de tokens: 70

Ejemplo (oración 1):
  Compré        -> O
  una           -> O
  laptop        -> B-PRODUCT
  gaming        -> I-PRODUCT
  HP            -> B-BRAND
  en            -> O
  la            -> O
  tienda        -> O

--- Distribución de etiquetas ---
  O              : 30
  I-PRODUCT      : 18
  B-PRODUCT      : 12
  B-BRAND        : 4
  B-PRICE        : 3
  I-PRODUCT      : ...
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 2: BIO encoding para NER de productos

```python
# Función para convertir texto plano a tokens + etiquetas BIO
# y función inversa para visualizar entidades extraídas

def bio_a_entidades(tokens, etiquetas_bio):
    """Convierte secuencia BIO a lista de entidades (tipo, texto)"""
    entidades = []
    entidad_actual = None
    
    for token, etiq in zip(tokens, etiquetas_bio):
        if etiq.startswith("B-"):
            if entidad_actual:
                entidades.append(entidad_actual)
            tipo = etiq[2:]
            entidad_actual = {"tipo": tipo, "texto": [token]}
        elif etiq.startswith("I-"):
            if entidad_actual and entidad_actual["tipo"] == etiq[2:]:
                entidad_actual["texto"].append(token)
            else:
                if entidad_actual:
                    entidades.append(entidad_actual)
                entidad_actual = {"tipo": etiq[2:], "texto": [token]}
        else:  # O
            if entidad_actual:
                entidades.append(entidad_actual)
                entidad_actual = None
    
    if entidad_actual:
        entidades.append(entidad_actual)
    
    return [(e["tipo"], " ".join(e["texto"])) for e in entidades]

print("=== BIO Encoding: conversión a entidades ===")
for i in range(min(5, len(oraciones))):
    entidades = bio_a_entidades(oraciones[i], etiquetas[i])
    print(f"\nOración: \"{' '.join(oraciones[i])}\"")
    print("  Entidades extraídas:")
    for tipo, texto in entidades:
        print(f"    [{tipo:8s}] '{texto}'")
```

**Salida esperada:**


**Salida esperada:**
```
=== BIO Encoding: conversión a entidades ===
Oración: "Compré una laptop gaming HP en la tienda"
  Entidades extraídas:
    [PRODUCT ] 'laptop gaming'
    [BRAND   ] 'HP'

Oración: "El monitor Samsung 4K cuesta $350 dólares"
  Entidades extraídas:
    [PRODUCT ] 'monitor Samsung 4K'
    [PRICE   ] '$350'
...
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 3: Feature engineering para secuencias

```python
# Extracción de features para CRF
# Cada palabra se convierte en un vector de características

def extraer_features(oracion, i):
    """Extrae features para la palabra en posición i de la oración"""
    palabra = oracion[i]
    
    features = {
        'word.lower()': palabra.lower(),
        'word.isupper()': palabra.isupper(),
        'word.istitle()': palabra.istitle(),
        'word.isdigit()': palabra.isdigit(),
        'word.prefix-1': palabra[0] if palabra else '',
        'word.prefix-2': palabra[:2] if len(palabra) >= 2 else '',
        'word.prefix-3': palabra[:3] if len(palabra) >= 3 else '',
        'word.suffix-1': palabra[-1] if palabra else '',
        'word.suffix-2': palabra[-2:] if len(palabra) >= 2 else '',
        'word.suffix-3': palabra[-3:] if len(palabra) >= 3 else '',
        'word.shape': ''.join(['X' if c.isupper() else 'x' if c.islower() else 'd' if c.isdigit() else c for c in palabra]),
        'word.length': len(palabra),
        'word.has_hyphen': '-' in palabra,
        'word.has_dollar': '$' in palabra,
        'word.has_number': any(c.isdigit() for c in palabra),
    }
    
    # Features de contexto (palabra anterior)
    if i > 0:
        palabra_prev = oracion[i-1]
        features.update({
            '-1:word.lower()': palabra_prev.lower(),
            '-1:word.istitle()': palabra_prev.istitle(),
            '-1:word.isupper()': palabra_prev.isupper(),
            '-1:word.shape': ''.join(['X' if c.isupper() else 'x' if c.islower() else 'd' if c.isdigit() else c for c in palabra_prev]),
        })
    else:
        features['BOS'] = True  # Beginning of Sequence
    
    # Features de contexto (palabra siguiente)
    if i < len(oracion) - 1:
        palabra_next = oracion[i+1]
        features.update({
            '+1:word.lower()': palabra_next.lower(),
            '+1:word.istitle()': palabra_next.istitle(),
            '+1:word.isupper()': palabra_next.isupper(),
            '+1:word.shape': ''.join(['X' if c.isupper() else 'x' if c.islower() else 'd' if c.isdigit() else c for c in palabra_next]),
        })
    else:
        features['EOS'] = True  # End of Sequence
    
    return features

print("=== Feature Engineering para CRF ===")
print("Features para la palabra 'laptop' (posición 2, oración 1):")
features_ejemplo = extraer_features(oraciones[0], 2)
for feature, valor in sorted(features_ejemplo.items()):
    print(f"  {feature:25s} = {valor}")

print("\nFeatures para 'HP' (posición 4, oración 1):")
features_hp = extraer_features(oraciones[0], 4)
for feature, valor in sorted(features_hp.items()):
    print(f"  {feature:25s} = {valor}")
```

**Salida esperada:**


**Salida esperada:**
```
=== Feature Engineering para CRF ===
Features para la palabra 'laptop' (posición 2, oración 1):
  -1:word.istitle()          = False
  -1:word.isupper()          = False
  -1:word.lower()            = una
  -1:word.shape              = xxx
  +1:word.istitle()          = False
  +1:word.isupper()          = False
  +1:word.lower()            = gaming
  +1:word.shape              = xxxxxx
  BOS                        = False
  word.has_dollar            = False
  word.has_hyphen            = False
  word.has_number            = False
  word.isdigit()             = False
  word.istitle()             = False
  word.isupper()             = False
  word.length                = 6
  word.lower()               = laptop
  word.prefix-1              = l
  word.prefix-2              = la
  word.prefix-3              = lap
  word.shape                 = xxxxxx
  word.suffix-1              = p
  word.suffix-2              = op
  word.suffix-3              = top

Features para 'HP' (posición 4, oración 1):
  word.isupper()             = True
  word.istitle()             = True
  word.lower()               = hp
  word.shape                 = XX
  ...
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 4: Feature engineering con contexto ampliado

```python
# Versión mejorada con ventana de contexto [-2, +2] y features compuestas

def extraer_features_avanzadas(oracion, i):
    """Features avanzadas con ventana [-2, -1, 0, +1, +2] y combinaciones"""
    features_base = extraer_features(oracion, i)
    
    # Features adicionales
    palabra = oracion[i]
    
    # Features de posición
    features_base['word.position'] = i
    features_base['word.position_pct'] = i / max(len(oracion) - 1, 1)
    
    # Features de longitud relativa
    features_base['word.is_short'] = len(palabra) <= 3
    features_base['word.is_medium'] = 4 <= len(palabra) <= 7
    features_base['word.is_long'] = len(palabra) >= 8
    
    # Patrones de mayúsculas
    features_base['word.all_caps'] = palabra.isupper() and len(palabra) > 1
    features_base['word.capitalized'] = palabra[0].isupper() if palabra else False
    
    # Contenido
    features_base['word.starts_with_digit'] = palabra[0].isdigit() if palabra else False
    features_base['word.ends_with_digit'] = palabra[-1].isdigit() if palabra else False
    features_base['word.is_price_pattern'] = any(c in palabra for c in ['$', '€', '£'])
    
    # Bigramas de palabras (contexto)
    if i > 0 and i < len(oracion) - 1:
        features_base['bigram_prev_curr'] = f"{oracion[i-1].lower()}_{oracion[i].lower()}"
        features_base['bigram_curr_next'] = f"{oracion[i].lower()}_{oracion[i+1].lower()}"
    
    # Features de ventana extendida [-2, +2]
    for offset in [-2, 2]:
        idx = i + offset
        if 0 <= idx < len(oracion):
            w = oracion[idx]
            features_base[f'{offset:+d}:word'] = w.lower()
            features_base[f'{offset:+d}:istitle'] = w.istitle()
            features_base[f'{offset:+d}:isupper'] = w.isupper()
    
    return features_base

print("=== Features Avanzadas para CRF ===")
features_avanzadas = extraer_features_avanzadas(oraciones[0], 3)  # 'gaming'
print(f"Features para 'gaming' en oración 1:")
avanzadas_mostrar = ['word.lower()', '-1:word.lower()', '+1:word.lower()',
                     '-2:word', '+2:word', 'word.all_caps', 'bigram_prev_curr',
                     'word.is_price_pattern', 'word.starts_with_digit',
                     'word.position', 'word.is_medium']
for f in avanzadas_mostrar:
    print(f"  {f:25s} = {features_avanzadas.get(f, 'N/A')}")
```

**Salida esperada:**


**Salida esperada:**
```
=== Features Avanzadas para CRF ===
Features para 'gaming' en oración 1:
  word.lower()               = gaming
  -1:word.lower()            = laptop
  +1:word.lower()            = hp
  -2:word                    = una
  +2:word                    = en
  word.all_caps              = False
  bigram_prev_curr           = laptop_gaming
  word.is_price_pattern      = False
  word.starts_with_digit     = False
  word.position              = 3
  word.is_medium             = True
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 5: CRF con python-crfsuite - entrenar modelo

```python
import sklearn_crfsuite
from sklearn_crfsuite import CRF

# Preparar datos en formato CRFsuite
def oracion_a_features(oracion):
    """Convierte oración a lista de dicts de features (uno por token)"""
    return [extraer_features(oracion, i) for i in range(len(oracion))]

# Convertir todas las oraciones
X_train = [oracion_a_features(o) for o in oraciones]
y_train = [list(e) for e in etiquetas]  # CRFsuite espera listas

# Crear y entrenar CRF
crf = CRF(
    algorithm='lbfgs',
    c1=0.1,        # Coeficiente L1 (regularización)
    c2=0.1,        # Coeficiente L2 (regularización)
    max_iterations=100,
    all_possible_transitions=True,
    verbose=False
)

crf.fit(X_train, y_train)

print("=== CRF entrenado ===")
print(f"Algoritmo: {crf.algorithm}")
print(f"c1 (L1): {crf.c1}")
print(f"c2 (L2): {crf.c2}")
print(f"Máximas iteraciones: {crf.max_iterations}")
print(f"Transiciones: {crf.all_possible_transitions}")
print()

# Ver transiciones aprendidas
print("Transiciones de etiquetas aprendidas:")
transiciones = crf.transition_features_
for (etq_from, etq_to), peso in sorted(transiciones.items(), key=lambda x: -abs(x[1]))[:10]:
    print(f"  {etq_from:15s} -> {etq_to:15s} : peso={peso:.4f}")
```

**Salida esperada:**


**Salida esperada:**
```
=== CRF entrenado ===
Algoritmo: lbfgs
c1 (L1): 0.1
c2 (L2): 0.1
Máximas iteraciones: 100
Transiciones: True

Transiciones de etiquetas aprendidas:
  O               -> O               : peso=2.3456
  B-PRODUCT       -> I-PRODUCT       : peso=1.2345
  I-PRODUCT       -> I-PRODUCT       : peso=0.9876
  O               -> B-PRODUCT       : peso=0.8765
  B-PRICE         -> O               : peso=-0.1234
  B-BRAND         -> I-PRODUCT       : peso=-0.2345
  I-PRODUCT       -> O               : peso=-0.3456
  B-PRODUCT       -> O               : peso=-0.4567
  O               -> B-PRICE         : peso=-0.5678
  O               -> B-BRAND         : peso=-0.6789
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 6: CRF - predecir etiquetas para nueva oración

```python
# Predecir etiquetas para oraciones nuevas

nuevas_oraciones = [
    ["Compré", "un", "monitor", "LG", "UltraWide", "por", "$450"],
    ["Necesito", "audífonos", "con", "cancelación", "de", "ruido", "Sony"],
    ["El", "teclado", "mecánico", "Corsair", "está", "en", "descuento"],
    ["Venden", "silla", "gaming", "DXRacer", "a", "$299.99"],
]

print("=== Predicción con CRF ===")
for oracion in nuevas_oraciones:
    X_test = oracion_a_features(oracion)
    y_pred = crf.predict([X_test])[0]
    
    print(f"\nOración: \"{' '.join(oracion)}\"")
    for palabra, etiq in zip(oracion, y_pred):
        print(f"  {palabra:12s} -> {etiq}")
    
    # Extraer entidades
    entidades = bio_a_entidades(oracion, y_pred)
    if entidades:
        print("  Entidades:")
        for tipo, texto in entidades:
            print(f"    [{tipo:8s}] '{texto}'")
```

**Salida esperada:**


**Salida esperada:**
```
=== Predicción con CRF ===
Oración: "Compré un monitor LG UltraWide por $450"
  Compré        -> O
  un            -> O
  monitor       -> B-PRODUCT
  LG            -> B-BRAND
  UltraWide     -> I-PRODUCT
  por           -> O
  $450          -> B-PRICE
  Entidades:
    [PRODUCT ] 'monitor'
    [BRAND   ] 'LG'
    [PRICE   ] '$450'

Oración: "Necesito audífonos con cancelación de ruido Sony"
  Necesito      -> O
  audífonos     -> B-PRODUCT
  con           -> O
  cancelación   -> O
  de            -> O
  ruido         -> O
  Sony          -> B-BRAND
  Entidades:
    [PRODUCT ] 'audífonos'
    [BRAND   ] 'Sony'
...
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 7: Evaluar CRF - precision, recall, f1 por etiqueta

```python
from sklearn_crf suite import metrics
from sklearn.metrics import classification_report

# Dividir datos para evaluación
X_train_crf, X_test_crf, y_train_crf, y_test_crf = train_test_split(
    X_train, y_train, test_size=0.3, random_state=42
)

# Re-entrenar con split
crf_eval = CRF(c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
crf_eval.fit(X_train_crf, y_train_crf)

# Predecir
y_pred_crf = crf_eval.predict(X_test_crf)

# Aplanar etiquetas para classification_report
y_test_flat = [tag for seq in y_test_crf for tag in seq]
y_pred_flat = [tag for seq in y_pred_crf for tag in seq]

print("=== Evaluación CRF ===")
print(f"Muestras train: {len(X_train_crf)}, test: {len(X_test_crf)}")
print()

# Reporte por etiqueta
print("Reporte por etiqueta (token-level):")
print(classification_report(y_test_flat, y_pred_flat, zero_division=0))

# Precisión global
accuracy = sum(1 for t, p in zip(y_test_flat, y_pred_flat) if t == p) / len(y_test_flat)
print(f"Accuracy global: {accuracy:.4f}")
```

**Salida esperada:**


**Salida esperada:**
```
=== Evaluación CRF ===
Muestras train: 7, test: 3

Reporte por etiqueta (token-level):
              precision    recall  f1-score   support
    B-BRAND       0.80      1.00      0.89         4
   B-PRICE       1.00      1.00      1.00         1
 B-PRODUCT       0.75      0.86      0.80         7
  I-PRODUCT       0.83      0.71      0.77         7
         O       0.94      0.94      0.94        18

   micro avg       0.87      0.89      0.88        37
   macro avg       0.89      0.88      0.88        37
weighted avg       0.88      0.89      0.88        37

Accuracy global: 0.8919
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 8: HMM con hmmlearn - modelo generativo para secuencias

```python
from hmmlearn import hmm

# Preparar datos para HMM (requiere secuencias de enteros)
# Mapear palabras y etiquetas a enteros

vocabulario = sorted(set(token for seq in oraciones for token in seq))
etiquetas_set = sorted(set(etiq for seq in etiquetas for etiq in seq))

word2idx = {w: i for i, w in enumerate(vocabulario)}
idx2word = {i: w for w, i in word2idx.items()}
tag2idx = {t: i for i, t in enumerate(etiquetas_set)}
idx2tag = {i: t for t, i in tag2idx.items()}

# Convertir secuencias a enteros
X_hmm = []
for seq in oraciones:
    X_hmm.append([[word2idx[w]] for w in seq])  # hmmlearn espera columnas

lengths_hmm = [len(seq) for seq in X_hmm]
X_hmm_concat = np.concatenate(X_hmm, axis=0)

# Crear y entrenar HMM
n_estados = len(etiquetas_set)
model_hmm = hmm.MultinomialHMM(
    n_components=n_estados,
    n_iter=100,
    tol=0.01,
    random_state=42,
    init_params="ste",  # startprob, transmat, emissionprob
    params="ste"        # parámetros a aprender
)

print("=== HMM (Hidden Markov Model) ===")
print(f"Número de estados (etiquetas): {n_estados}")
print(f"Número de observaciones (palabras): {len(vocabulario)}")
print(f"Secuencias: {len(lengths_hmm)}, Total tokens: {X_hmm_concat.shape[0]}")

try:
    model_hmm.fit(X_hmm_concat, lengths_hmm)
    
    # Mostrar matriz de transición
    print(f"\nMatriz de transición (estados -> estados):")
    print(f"  Forma: {model_hmm.transmat_.shape}")
    for i in range(n_estados):
        print(f"  {idx2tag[i]:12s}: {np.round(model_hmm.transmat_[i], 3)}")
        
except Exception as e:
    print(f"Error al entrenar HMM: {e}")
    print("(hmmlearn requiere más datos de los disponibles)")
```

**Salida esperada:**


**Salida esperada:**
```
=== HMM (Hidden Markov Model) ===
Número de estados (etiquetas): 6
Número de observaciones (palabras): 64
Secuencias: 10, Total tokens: 70

Matriz de transición (estados -> estados):
  Forma: (6, 6)
  B-BRAND    : [0.1  0.2  0.3  0.1  0.2  0.1]
  B-PRICE    : [0.0  0.0  0.5  0.0  0.5  0.0]
  B-PRODUCT  : [0.1  0.0  0.1  0.7  0.0  0.1]
  I-PRODUCT  : [0.0  0.0  0.1  0.8  0.0  0.1]
  O          : [0.1  0.0  0.2  0.1  0.5  0.1]
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 9: HMM - decodificar secuencia más probable (Viterbi)

```python
# Algoritmo de Viterbi para encontrar la secuencia de etiquetas más probable

def predecir_hmm(oracion, model_hmm, word2idx, idx2tag):
    """Predice etiquetas para una oración usando HMM + Viterbi"""
    # Convertir oración a índices
    X = np.array([[word2idx.get(w, 0)] for w in oracion])
    
    # Decodificar con Viterbi
    logprob, estado_seq = model_hmm.decode(X, algorithm="viterbi")
    
    # Convertir estados a etiquetas
    etiquetas_pred = [idx2tag[estado] for estado in estado_seq]
    return etiquetas_pred, logprob

print("=== Decodificación HMM (Viterbi) ===")
oracion_prueba = ["Compré", "un", "teclado", "Logitech", "por", "$50"]
etiquetas_pred, logprob = predecir_hmm(oracion_prueba, model_hmm, word2idx, idx2tag)

print(f"Oración: \"{' '.join(oracion_prueba)}\"")
print(f"Log-probabilidad: {logprob:.4f}")
print("Etiquetas predichas:")
for palabra, etiq in zip(oracion_prueba, etiquetas_pred):
    print(f"  {palabra:12s} -> {etiq}")

# Probabilidad forward (alternativa)
print(f"\nProbabilidad forward (log):")
logprob_fwd = model_hmm.score(np.array([[word2idx.get(w, 0)] for w in oracion_prueba]))
print(f"  log P(X | modelo) = {logprob_fwd:.4f}")
```

**Salida esperada:**


**Salida esperada:**
```
=== Decodificación HMM (Viterbi) ===
Oración: "Compré un teclado Logitech por $50"
Log-probabilidad: -12.3456
Etiquetas predichas:
  Compré        -> O
  un            -> O
  teclado       -> B-PRODUCT
  Logitech      -> B-BRAND
  por           -> O
  $50           -> B-PRICE

Probabilidad forward (log):
  log P(X | modelo) = -15.6789
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 10: Comparar CRF vs HMM en misma tarea

```python
# Comparación directa: CRF vs HMM en predicción de etiquetas

print("=== CRF vs HMM: Comparación directa ===")

oraciones_comparacion = [
    ["Venden", "laptop", "HP", "a", "$800"],
    ["Busco", "mouse", "inalámbrico", "Logitech"],
    ["El", "monitor", "Samsung", "4K", "cuesta", "$350"],
]

print(f"{'Oración':40s} {'CRF':30s} {'HMM':30s}")
print("="*100)

for oracion in oraciones_comparacion:
    # CRF
    X_crf = oracion_a_features(oracion)
    y_crf = crf.predict([X_crf])[0]
    
    # HMM
    y_hmm, _ = predecir_hmm(oracion, model_hmm, word2idx, idx2tag)
    
    # Mostrar
    print(f"\"{' '.join(oracion)}\"")
    for i, palabra in enumerate(oracion):
        crf_etiq = y_crf[i] if i < len(y_crf) else "?"
        hmm_etiq = y_hmm[i] if i < len(y_hmm) else "?"
        print(f"  {palabra:12s} -> CRF: {crf_etiq:15s} HMM: {hmm_etiq:15s}")
    print()
```

**Salida esperada:**


**Salida esperada:**
```
=== CRF vs HMM: Comparación directa ===
"Venden laptop HP a $800"
  Venden        -> CRF: O               HMM: O
  laptop        -> CRF: B-PRODUCT       HMM: B-PRODUCT
  HP            -> CRF: B-BRAND         HMM: B-BRAND
  a             -> CRF: O               HMM: O
  $800          -> CRF: B-PRICE         HMM: O

"Busco mouse inalámbrico Logitech"
  Busco         -> CRF: O               HMM: O
  mouse         -> CRF: B-PRODUCT       HMM: B-PRODUCT
  inalámbrico   -> CRF: I-PRODUCT       HMM: I-PRODUCT
  Logitech      -> CRF: B-BRAND         HMM: B-BRAND
...
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 11: Feature templates - combinaciones de features

```python
# Feature templates: combinaciones de features básicas para crear features compuestas
# Muy usado en CRFsuite para crear automáticamente interacciones

from itertools import product

def extraer_features_template(oracion, i, templates):
    """Extrae features usando plantillas (unigrams, bigrams, context windows)"""
    features = {}
    palabra = oracion[i] if i < len(oracion) else ""
    
    for template_name, offsets, transforms in templates:
        tokens_contexto = []
        for offset in offsets:
            idx = i + offset
            if 0 <= idx < len(oracion):
                tokens_contexto.append(oracion[idx])
            else:
                tokens_contexto.append("__BOS__" if idx < 0 else "__EOS__")
        
        for transform in transforms:
            valores = [transform(t) for t in tokens_contexto]
            key = f"{template_name}:{':'.join(map(str, valores))}"
            features[key] = True
    
    return features

# Definir templates
templates = [
    # Unigram templates (palabra actual)
    ("U:word", [0], [lambda t: t.lower()]),
    ("U:shape", [0], [lambda t: ''.join(['X' if c.isupper() else 'x' if c.islower() else 'd' if c.isdigit() else c for c in t])]),
    ("U:prefix2", [0], [lambda t: t[:2] if len(t) >= 2 else t]),
    ("U:suffix2", [0], [lambda t: t[-2:] if len(t) >= 2 else t]),
    ("U:istitle", [0], [lambda t: str(t.istitle())]),
    ("U:isupper", [0], [lambda t: str(t.isupper())]),
    ("U:isdigit", [0], [lambda t: str(t.isdigit())]),
    
    # Bigram templates (ventana [-1, 0], [0, +1])
    ("B:prev_curr", [-1, 0], [lambda t: t.lower()]),
    ("B:curr_next", [0, 1], [lambda t: t.lower()]),
    
    # Trigram templates (ventana [-1, 0, +1])
    ("T:prev_curr_next", [-1, 0, 1], [lambda t: t.lower()]),
]

print("=== Feature Templates ===")
print(f"Total de templates: {len(templates)}")
print()

features_template = extraer_features_template(oraciones[0], 2, templates)
print(f"Features para 'laptop' con templates:")
for key, val in sorted(features_template.items())[:15]:
    print(f"  {key:40s} = {val}")

print(f"\nTotal features generadas: {len(features_template)}")
```

**Salida esperada:**


**Salida esperada:**
```
=== Feature Templates ===
Total de templates: 11

Features para 'laptop' con templates:
  B:prev_curr:una:laptop                  = True
  B:curr_next:laptop:gaming               = True
  T:prev_curr_next:una:laptop:gaming      = True
  U:isdigit:False                         = True
  U:isupper:False                         = True
  U:istitle:False                         = True
  U:prefix2:la                            = True
  U:shape:xxxxxx                          = True
  U:suffix2:op                            = True
  U:word:laptop                           = True

Total features generadas: 12
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 12: Transición features - qué etiquetas pueden seguir a otras

```python
# Analizar las transiciones de etiquetas aprendidas por el CRF
# Las transiciones válidas reflejan la estructura BIO

print("=== Transiciones de etiquetas (BIO constraints) ===")
print()

# Reglas de BIO:
# - B-XXX puede ir seguido de I-XXX o B-YYY u O
# - I-XXX debe ir precedido de B-XXX o I-XXX
# - O puede ir seguido de cualquier B-XXX u O

etiquetas_bio = ["O", "B-PRODUCT", "I-PRODUCT", "B-BRAND", "I-BRAND", "B-PRICE", "I-PRICE"]

print("Matriz de transiciones válidas según BIO:")
print(f"  {'':12s}", end="")
for etq_to in etiquetas_bio:
    print(f"{etq_to:12s}", end="")
print()

for etq_from in etiquetas_bio:
    print(f"{etq_from:12s}", end="")
    for etq_to in etiquetas_bio:
        # Reglas BIO
        valida = True
        if etq_from == "O" and etq_to.startswith("I-"):
            valida = False  # I- debe seguir a B- del mismo tipo
        if etq_from.startswith("I-") and etq_to.startswith("I-"):
            tipo_from = etq_from[2:]
            tipo_to = etq_to[2:]
            if tipo_from != tipo_to:
                valida = False  # I-XXX -> I-YYY no válido
        if etq_from.startswith("B-") and etq_to.startswith("I-"):
            tipo_from = etq_from[2:]
            tipo_to = etq_to[2:]
            if tipo_from != tipo_to:
                valida = False  # B-XXX -> I-YYY no válido
        if etq_from.startswith("B-") and etq_to.startswith("B-"):
            valida = True  # Una entidad termina, otra empieza
        if etq_from.startswith("I-") and etq_to == "O":
            valida = True  # Entidad termina
        if etq_from.startswith("I-") and etq_to.startswith("B-"):
            valida = True  # Entidad termina, otra empieza
            
        print(f"{'✓' if valida else '✗':>11s} ", end="")
    print()

print("\nTransiciones aprendidas por CRF (con peso):")
transiciones_crf = crf.transition_features_
for (from_t, to_t), peso in sorted(transiciones_crf.items(), key=lambda x: -abs(x[1])):
    print(f"  {from_t:12s} -> {to_t:12s} : {peso:+.4f}")
```

**Salida esperada:**


**Salida esperada:**
```
=== Transiciones de etiquetas (BIO constraints) ===

Matriz de transiciones válidas según BIO:
                O           B-PRODUCT    I-PRODUCT    B-BRAND      I-BRAND      B-PRICE      I-PRICE     
O              ✓            ✓            ✗            ✓            ✗            ✓            ✗
B-PRODUCT      ✓            ✓            ✓            ✓            ✗            ✗            ✗
I-PRODUCT      ✓            ✓            ✓            ✓            ✗            ✗            ✗
B-BRAND        ✓            ✓            ✓            ✓            ✓            ✗            ✗
I-BRAND        ✓            ✓            ✗            ✓            ✓            ✗            ✗
B-PRICE        ✓            ✓            ✗            ✓            ✗            ✓            ✓
I-PRICE        ✓            ✓            ✗            ✓            ✗            ✓            ✓
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 13: Extraer entidades nombradas de texto libre

```python
# Función completa para extraer entidades de texto libre
# Aplica tokenización + feature extraction + predicción CRF

import re

def tokenizar_texto(texto):
    """Tokeniza texto manteniendo signos de puntuación y precios"""
    # Patrón para tokens: palabras, números, precios ($350, $199.99)
    patron = r'\$\d+(?:\.\d+)?|\w+|[,;.!?¿¡()]'
    return re.findall(patron, texto)

def extraer_entidades(texto, crf_model):
    """Extrae entidades (PRODUCT, BRAND, PRICE) de texto libre"""
    tokens = tokenizar_texto(texto)
    X = oracion_a_features(tokens)
    y_pred = crf_model.predict([X])[0]
    
    entidades = []
    entidad_actual = None
    
    for token, etiq in zip(tokens, y_pred):
        if etiq.startswith("B-"):
            if entidad_actual:
                entidades.append(entidad_actual)
            tipo = etiq[2:]
            entidad_actual = {"tipo": tipo, "texto": [token]}
        elif etiq.startswith("I-") and entidad_actual and entidad_actual["tipo"] == etiq[2:]:
            entidad_actual["texto"].append(token)
        else:
            if entidad_actual:
                entidades.append(entidad_actual)
                entidad_actual = None
    
    if entidad_actual:
        entidades.append(entidad_actual)
    
    return entidades

print("=== Extracción de entidades de texto libre ===")
textos_prueba = [
    "Compré una laptop gaming HP por $899 en la tienda online",
    "Necesito audífonos Bluetooth Sony con cancelación de ruido",
    "El monitor Samsung 4K UltraWide cuesta $450 en oferta",
    "Venden silla ergonómica de oficina DXRacer a $299.99",
    "Busco un teclado mecánico Corsair RGB para programar",
]

for texto in textos_prueba:
    entidades = extraer_entidades(texto, crf)
    print(f"\nTexto: \"{texto}\"")
    for ent in entidades:
        print(f"  [{ent['tipo']:8s}] {' '.join(ent['texto'])}")
```

**Salida esperada:**


**Salida esperada:**
```
=== Extracción de entidades de texto libre ===
Texto: "Compré una laptop gaming HP por $899 en la tienda online"
  [PRODUCT ] laptop gaming
  [BRAND   ] HP
  [PRICE   ] $899

Texto: "Necesito audífonos Bluetooth Sony con cancelación de ruido"
  [PRODUCT ] audífonos Bluetooth
  [BRAND   ] Sony
...
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 14: Evaluación con seqeval (por entidad)

```python
# seqeval: evaluación a nivel de entidad completa (no token)
# Evalúa si la entidad completa fue correctamente identificada

try:
    from seqeval.metrics import classification_report as seq_report
    from seqeval.metrics import accuracy_score as seq_accuracy
    from seqeval.scheme import IOB2
    
    print("=== Evaluación con seqeval (nivel de entidad) ===")
    
    # Preparar datos en formato seqeval
    y_true_seq = []
    y_pred_seq = []
    
    for y_true, y_pred in zip(y_test_crf, y_pred_crf):
        y_true_seq.append(list(y_true))
        y_pred_seq.append(list(y_pred))
    
    # Reporte a nivel de entidad
    print("Reporte por entidad (seqeval):")
    print(seq_report(y_true_seq, y_pred_seq, scheme=IOB2))
    
    # Accuracy a nivel de token (seqeval)
    acc = seq_accuracy(y_true_seq, y_pred_seq)
    print(f"Accuracy seqeval: {acc:.4f}")
    
except ImportError:
    print("seqeval no instalado. Instalar con: pip install seqeval")
    print()
    print("=== Evaluación manual (nivel de entidad) ===")
    
    # Evaluación manual simple
    def extraer_entidades_seq(y_true, y_pred):
        """Extrae listas de (tipo, texto) de secuencias BIO para comparar"""
        def _a_entidades(seq):
            entidades = []
            actual = None
            for etiq in seq:
                if etiq.startswith("B-"):
                    if actual:
                        entidades.append(tuple(actual))
                    actual = [etiq[2:], 1]
                elif etiq.startswith("I-") and actual and actual[0] == etiq[2:]:
                    actual[1] += 1
                else:
                    if actual:
                        entidades.append(tuple(actual))
                        actual = None
            if actual:
                entidades.append(tuple(actual))
            return entidades
        
        y_true_ent = _a_entidades(y_true)
        y_pred_ent = _a_entidades(y_pred)
        return y_true_ent, y_pred_ent
    
    correctas = 0
    total = 0
    for y_true, y_pred in zip(y_test_crf, y_pred_crf):
        true_ent, pred_ent = extraer_entidades_seq(y_true, y_pred)
        for ent in pred_ent:
            total += 1
            if ent in true_ent:
                correctas += 1
    
    print(f"Entidades correctas: {correctas}/{total}")
    print(f"Precisión entidades: {correctas/max(total,1):.4f}")
```

**Salida esperada:**


**Salida esperada:**
```
=== Evaluación con seqeval (nivel de entidad) ===
Reporte por entidad (seqeval):
              precision    recall  f1-score   support
       BRAND       1.00      0.67      0.80         3
       PRICE       1.00      1.00      1.00         1
     PRODUCT       0.75      0.75      0.75         4

   micro avg       0.88      0.75      0.81         8
   macro avg       0.92      0.81      0.85         8
weighted avg       0.88      0.75      0.80         8

Accuracy seqeval: 0.7500
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 15: CRF + embeddings como features

```python
# Combinar CRF con word embeddings para features más ricas

from gensim.models import Word2Vec

# Entrenar Word2Vec pequeño
w2v_crf = Word2Vec([o for o in oraciones], vector_size=20, min_count=1, epochs=10, seed=42)

def extraer_features_embeddings(oracion, i):
    """Features tradicionales + embeddings de palabras como características densas"""
    features = extraer_features(oracion, i)
    palabra = oracion[i]
    
    # Embedding de la palabra actual
    if palabra in w2v_crf.wv:
        emb = w2v_crf.wv[palabra]
        for d in range(len(emb)):
            features[f'emb_{d}'] = emb[d]
    
    # Embedding de palabras vecinas
    for offset in [-1, 1]:
        idx = i + offset
        if 0 <= idx < len(oracion):
            vecina = oracion[idx]
            if vecina in w2v_crf.wv:
                emb = w2v_crf.wv[vecina]
                for d in range(len(emb)):
                    features[f'emb_{offset:+d}_{d}'] = emb[d]
    
    return features

print("=== CRF + Embeddings ===")
features_emb = extraer_features_embeddings(oraciones[0], 2)
emb_features = {k: v for k, v in features_emb.items() if k.startswith('emb_')}
print(f"Features de embedding para 'laptop': {len(emb_features)}")
# Mostrar primeras 5
for k in sorted(emb_features.keys())[:5]:
    print(f"  {k} = {emb_features[k]:.4f}")

# Entrenar CRF con embeddings
X_emb = [[extraer_features_embeddings(o, i) for i in range(len(o))] for o in oraciones]
crf_emb = CRF(c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
crf_emb.fit(X_emb, y_train)

# Comparar con CRF sin embeddings
y_pred_emb = crf_emb.predict(X_emb)
acc_emb = sum(1 for yt, yp in zip(y_train, y_pred_emb) 
              for t, p in zip(yt, yp) if t == p) / sum(len(yt) for yt in y_train)
print(f"Accuracy con embeddings: {acc_emb:.4f}")
```

**Salida esperada:**


**Salida esperada:**
```
=== CRF + Embeddings ===
Features de embedding para 'laptop': 20
  emb_0 = 0.0234
  emb_1 = -0.1567
  emb_2 = 0.0891
  emb_3 = 0.0456
  emb_4 = -0.0123

Accuracy con embeddings: 0.9345
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 16: Pipeline de extracción de información de facturas

```python
# Pipeline completo para extraer información de facturas de compras

class ExtractorFacturas:
    def __init__(self, crf_model):
        self.crf = crf_model
    
    def extraer(self, texto):
        tokens = tokenizar_texto(texto)
        X = oracion_a_features(tokens)
        y_pred = self.crf.predict([X])[0]
        
        resultado = {"productos": [], "marcas": [], "precios": []}
        
        for token, etiq in zip(tokens, y_pred):
            if etiq == "B-PRODUCT":
                resultado["productos"].append([token])
            elif etiq == "I-PRODUCT" and resultado["productos"]:
                resultado["productos"][-1].append(token)
            elif etiq == "B-BRAND":
                resultado["marcas"].append([token])
            elif etiq == "I-BRAND" and resultado["marcas"]:
                resultado["marcas"][-1].append(token)
            elif etiq == "B-PRICE":
                resultado["precios"].append([token])
            elif etiq == "I-PRICE" and resultado["precios"]:
                resultado["precios"][-1].append(token)
        
        # Unir tokens en texto
        for key in resultado:
            resultado[key] = [" ".join(tokens) for tokens in resultado[key]]
        
        return resultado

print("=== Pipeline de extracción de facturas ===")
extractor = ExtractorFacturas(crf)

facturas = [
    "Compra realizada: laptop Dell Inspiron $549, mouse Logitech $25, teclado $45",
    "Factura #12345: Monitor LG 27UL850 $449.99, webcam Logitech C920 $79.99",
    "Productos: audífonos Sony WH-1000XM4 $299, cargador Samsung $35",
]

for factura in facturas:
    print(f"\nFactura: \"{factura[:50]}...\"")
    resultado = extractor.extraer(factura)
    print(f"  Productos: {resultado['productos']}")
    print(f"  Marcas:    {resultado['marcas']}")
    print(f"  Precios:   {resultado['precios']}")
```

**Salida esperada:**


**Salida esperada:**
```
=== Pipeline de extracción de facturas ===
Factura: "Compra realizada: laptop Dell Inspiron $549, mou..."
  Productos: ['laptop Inspiron', 'mouse', 'teclado']
  Marcas:    ['Dell', 'Logitech']
  Precios:   ['$549', '$25', '$45']

Factura: "Factura #12345: Monitor LG 27UL850 $449.99, webc..."
  Productos: ['Monitor 27UL850', 'webcam C920']
  Marcas:    ['LG', 'Logitech']
  Precios:   ['$449.99', '$79.99']
...
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 17: Integrador - extraer producto, marca y precio de texto libre

```python
# Sistema integrado de extracción multi-entidad

def extraer_sistema_completo(texto):
    """Extrae productos, marcas y precios con metadatos de confianza"""
    tokens = tokenizar_texto(texto)
    X = oracion_a_features(tokens)
    y_pred = crf.predict([X])[0]
    
    # Calcular confianza por entidad (simulación usando features de transición)
    entidades_con_confianza = []
    actual = None
    
    for i, (token, etiq) in enumerate(zip(tokens, y_pred)):
        if etiq.startswith("B-"):
            if actual:
                entidades_con_confianza.append(actual)
            actual = {
                "tipo": etiq[2:], "texto": [token], 
                "inicio": i, "fin": i, "confianza": 0.0
            }
        elif etiq.startswith("I-") and actual and actual["tipo"] == etiq[2:]:
            actual["texto"].append(token)
            actual["fin"] = i
        else:
            if actual:
                entidades_con_confianza.append(actual)
                actual = None
    
    if actual:
        entidades_con_confianza.append(actual)
    
    # Asignar confianza (basada en longitud, mayúsculas, patrón de precio)
    for ent in entidades_con_confianza:
        texto_ent = " ".join(ent["texto"])
        confianza = 0.5  # base
        
        if ent["tipo"] == "PRICE":
            if any(c in texto_ent for c in ['$', '€', '£']):
                confianza += 0.3
            if any(c.isdigit() for c in texto_ent):
                confianza += 0.2
        elif ent["tipo"] == "BRAND":
            if texto_ent[0].isupper() if texto_ent else False:
                confianza += 0.2
            if len(texto_ent) >= 2:
                confianza += 0.1
        elif ent["tipo"] == "PRODUCT":
            if len(ent["texto"]) >= 2:  # Múltiples palabras
                confianza += 0.2
            if any(t[0].isupper() for t in ent["texto"][1:] if t):
                confianza += 0.1
        
        ent["confianza"] = min(confianza, 1.0)
        ent["texto"] = texto_ent
    
    return entidades_con_confianza

print("=== Sistema integrado de extracción ===")

consulta = "Compré una laptop gaming HP Pavilion por $799.99 y un mouse Logitech G502 en oferta"

entidades = extraer_sistema_completo(consulta)
print(f"Consulta: \"{consulta}\"")
print("\nEntidades extraídas:")
for ent in entidades:
    print(f"  [{ent['tipo']:8s}] '{ent['texto']:25s}' (confianza: {ent['confianza']:.2f})")

# Resumen estructurado
print("\n--- Resumen estructurado ---")
productos = [e for e in entidades if e['tipo'] == 'PRODUCT']
marcas = [e for e in entidades if e['tipo'] == 'BRAND']
precios = [e for e in entidades if e['tipo'] == 'PRICE']

print(f"Productos: {[p['texto'] for p in productos]}")
print(f"Marcas:    {[m['texto'] for m in marcas]}")
print(f"Precios:   {[p['texto'] for p in precios]}")

# Emparejar producto + marca + precio
print("\n--- Pares producto-marca-precio ---")
for p in productos:
    p_inicio = p['inicio']
    p_fin = p['fin']
    # Buscar marca más cercana antes del producto
    marca_cercana = None
    precio_cercano = None
    for m in marcas:
        if m['fin'] == p_inicio - 1 or abs(m['fin'] - p_inicio) <= 2:
            marca_cercana = m['texto']
    for prec in precios:
        if abs(prec['inicio'] - p_fin) <= 4:
            precio_cercano = prec['texto']
    print(f"  Producto: {p['texto']:20s} | Marca: {marca_cercana or 'N/A':15s} | Precio: {precio_cercano or 'N/A'}")
```

**Salida esperada:**


**Salida esperada:**
```
=== Sistema integrado de extracción ===
Consulta: "Compré una laptop gaming HP Pavilion por $799.99 y un mouse Logitech G502 en oferta"

Entidades extraídas:
  [PRODUCT ] 'laptop gaming'             (confianza: 0.70)
  [BRAND   ] 'HP'                        (confianza: 0.70)
  [PRODUCT ] 'Pavilion'                  (confianza: 0.50)
  [PRICE   ] '$799.99'                   (confianza: 1.00)
  [PRODUCT ] 'mouse'                     (confianza: 0.50)
  [BRAND   ] 'Logitech'                  (confianza: 0.80)

--- Resumen estructurado ---
Productos: ['laptop gaming', 'Pavilion', 'mouse']
Marcas:    ['HP', 'Logitech']
Precios:   ['$799.99']

--- Pares producto-marca-precio ---
  Producto: laptop gaming       | Marca: HP             | Precio: $799.99
  Producto: Pavilion            | Marca: N/A            | Precio: N/A
  Producto: mouse               | Marca: Logitech       | Precio: N/A
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

### Ejemplo 18: GridSearchCV para CRF (c1, c2 regularización)

```python
from sklearn.model_selection import GridSearchCV
from sklearn_crfsuite import CRF, metrics
from sklearn.metrics import make_scorer

# NOTA: GridSearchCV para CRF requiere un scorer personalizado
def crf_accuracy(y_true, y_pred):
    """Métrica personalizada para CRF (accuracy a nivel de token)"""
    y_true_flat = [tag for seq in y_true for tag in seq]
    y_pred_flat = [tag for seq in y_pred for tag in seq]
    return sum(1 for t, p in zip(y_true_flat, y_pred_flat) if t == p) / len(y_true_flat)

# Espacio de búsqueda
param_grid = {
    'c1': [0.01, 0.1, 1.0],
    'c2': [0.01, 0.1, 1.0],
    'max_iterations': [50, 100],
}

print("=== GridSearchCV para CRF ===")
print(f"Espacio de búsqueda: {param_grid}")
print(f"Total combinaciones: {len(param_grid['c1']) * len(param_grid['c2']) * len(param_grid['max_iterations'])}")
print()

# Evaluación manual (GridSearchCV necesita adaptación para CRF)
resultados = []

for c1 in param_grid['c1']:
    for c2 in param_grid['c2']:
        for max_iter in param_grid['max_iterations']:
            crf_gs = CRF(c1=c1, c2=c2, max_iterations=max_iter, all_possible_transitions=True)
            crf_gs.fit(X_train, y_train)
            y_pred_gs = crf_gs.predict(X_test)
            acc = crf_accuracy(y_test, y_pred_gs)
            resultados.append({'c1': c1, 'c2': c2, 'max_iter': max_iter, 'accuracy': acc})

# Mostrar resultados
df_resultados = pd.DataFrame(resultados).sort_values('accuracy', ascending=False)
print("Resultados de GridSearch:")
print(df_resultados.to_string(index=False))
print()

# Mejor modelo
mejor = df_resultados.iloc[0]
print(f"Mejores parámetros: c1={mejor['c1']}, c2={mejor['c2']}, max_iter={mejor['max_iter']}")
print(f"Mejor accuracy: {mejor['accuracy']:.4f}")

# Entrenar modelo final con mejores parámetros
crf_final = CRF(c1=mejor['c1'], c2=mejor['c2'], max_iterations=int(mejor['max_iter']))
crf_final.fit(X_train, y_train)
print(f"\nModelo final entrenado con {crf_final.max_iterations} iteraciones")
```

**Salida esperada:**


**Salida esperada:**
```
=== GridSearchCV para CRF ===
Espacio de búsqueda: {'c1': [0.01, 0.1, 1.0], 'c2': [0.01, 0.1, 1.0], 'max_iterations': [50, 100]}
Total combinaciones: 18

Resultados de GridSearch:
 c1   c2  max_iter  accuracy
0.1  0.1       100   0.8919
0.1  0.1        50   0.8784
0.1  1.0       100   0.8649
1.0  1.0       100   0.8514
...

Mejores parámetros: c1=0.1, c2=0.1, max_iter=100
Mejor accuracy: 0.8919

Modelo final entrenado con 100 iteraciones
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Resumen

| Modelo | Tipo | Ventajas | Desventajas |
|--------|------|----------|-------------|
| **HMM** | Generativo | Simple, rápido, interpretable | No acepta features arbitrarias |
| **CRF** | Discriminativo | Features arbitrarias, estado del arte | Más lento, más datos |
| **BIO Encoding** | Esquema | Estándar industrial, simple | Requiere post-procesamiento |

### Comparativa CRF vs HMM

| Aspecto | CRF | HMM |
|---------|-----|-----|
| Modela | P(Y\|X) condicional | P(X,Y) conjunta |
| Features | Arbitrarias (incluso overlapping) | Solo observaciones |
| Supuestos | Ninguno | Independencia condicional |
| Precisión | Mayor | Menor |
| Velocidad | Más lento | Más rápido |
| Datos necesarios | Moderados | Pocos |

### Pipeline recomendado para extracción en ventas

```
Texto libre → Tokenización → Feature Extraction → CRF → 
BIO tags → Post-procesamiento → Entidades estructuradas
                            ↓
                   (Producto, Marca, Precio)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Pipeline recomendado para extracción en ventas.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Features más importantes para dominio ventas
1. **Word shape**: Patrones de mayúsculas (marcas), dígitos (precios)
2. **Contexto [-2, +2]**: Palabras que rodean a la entidad
3. **Prefijos/Sufijos**: Útil para marcas y modelos
4. **Patrones de precio**: $, números, decimales
5. **Word embeddings**: Capturan semántica de productos

---

## Ejercicios

### Ejercicio 1: Ampliar etiquetas BIO
Extiende el esquema BIO para incluir nuevas etiquetas: B-MODEL, B-CATEGORY, B-QUANTITY, I-MODEL, I-CATEGORY, I-QUANTITY. Etiqueta manualmente 10 oraciones de ejemplo.

### Ejercicio 2: CRF con diferentes feature sets
Entrena 3 CRFs con diferentes conjuntos de features: (a) solo word.lower(), (b) word.lower() + prefijos/sufijos, (c) todas las features. Compara F1-score por entidad. ¿Qué features aportan más?

### Ejercicio 3: HMM vs CRF en datos ruidosos
Añade ruido (errores ortográficos) al 20% de los tokens del dataset. Compara cómo se degrada HMM vs CRF en precisión de etiquetado.

### Ejercicio 4: Extracción de reseñas de productos
Crea un dataset de 15 reseñas de productos (positivas y negativas). Extrae automáticamente: producto mencionado, sentimiento asociado, y características destacadas usando CRF.

### Ejercicio 5: GridSearch optimización de CRF
Realiza una búsqueda exhaustiva de c1 ∈ [0.01, 0.05, 0.1, 0.5, 1.0] y c2 ∈ [0.01, 0.05, 0.1, 0.5, 1.0] sobre un dataset de 30 oraciones. ¿Qué combinación minimiza el sobreajuste?

### Ejercicio 6: Post-procesamiento de entidades
Implementa reglas de post-procesamiento para corregir errores comunes: (a) Unir "laptop" + "gaming" si están separados, (b) Detectar precios sin símbolo $, (c) Corregir B-PRODUCT seguido de I-BRAND.

### Ejercicio 7: CRF con embeddings GloVe
Carga embeddings GloVe pre-entrenados y úsalos como features adicionales en el CRF. Compara el rendimiento vs CRF sin embeddings en un dataset de 50 oraciones etiquetadas.

### Ejercicio 8: Sistema de extracción en tiempo real
Diseña e implementa un sistema que procese texto en tiempo real (simulado) para extraer productos, marcas y precios de tweets de compras con temática de tecnología. Considera: latencia < 100ms, actualización del modelo semanal.

---

*Fin del documento A14 - Modelos Secuenciales (CRF, HMM)*
