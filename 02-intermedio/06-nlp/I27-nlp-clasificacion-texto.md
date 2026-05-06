# I27: Clasificación de Texto — Vectorización + ML

## Introducción Teórica

La clasificación de texto asigna una categoría predefinida a un documento basándose en su contenido. El pipeline típico es: **texto → vectorización → modelo ML → predicción**.

### Vectorización

Convierte texto en números que los algoritmos ML puedan procesar:

#### CountVectorizer
Construye una matriz de conteo de palabras (documentos × vocabulario). Parámetros clave:
- `max_features`: máx. palabras en vocabulario
- `min_df` / `max_df`: frecuencia mín/máx de documento (elimina raras/muy comunes)
- `stop_words`: lista predefinida o personalizada
- `ngram_range`: (1,1) solo palabras, (1,2) unigramas+bigramas, (2,2) solo bigramas
- `analyzer`: 'word' (default), 'char' (caracteres), 'char_wb' (caracteres dentro de palabras)
- `binary`: True → presencia/ausencia (0/1) en vez de conteo
- `lowercase`: True por defecto
- `token_pattern`: regex para tokens (default: `(?u)\\b\\w\\w+\\b`)

#### TfidfVectorizer
Peso TF-IDF = Term Frequency × Inverse Document Frequency. Palabras frecuentes en un doc pero raras en el corpus reciben más peso.
- `norm`: 'l1', 'l2' (normalización), None (raw counts)
- `use_idf`: True (default), False → solo conteo normalizado
- `smooth_idf`: True (suma 1 a IDF, evita división por 0)
- `sublinear_tf`: True → 1 + log(tf) (escala logarítmica)

#### HashingVectorizer
Usa trucos de hash para convertir texto a vector sin mantener vocabulario en memoria. Ideal para grandes volúmenes (no se puede invertir).
- `n_features`: dimensión del vector hash
- `alternate_sign`: True → alterna signo (reduce sesgo)
- `norm`: normalización

### Clasificadores

#### Naive Bayes para texto
- **MultinomialNB**: asume distribución multinomial (conteos). Ideal para documentos con conteo de palabras.
- **ComplementNB**: variante de MultinomialNB para datos desbalanceados. Usa estadísticas de clases complementarias.
- **BernoulliNB**: para matrices binarias (presencia/ausencia). Bueno cuando la ausencia de palabras también importa.

#### SGDClassifier
Clasificador lineal con descenso de gradiente estocástico. Parámetro `loss`:
- `'log'`: regresión logística (probabilidades)
- `'modified_huber'`: robusto a outliers, da probabilidades suavizadas
- `'hinge'`: SVM lineal (margen máximo)
- `'perceptron'`: algoritmo perceptrón

### Pipeline

`Pipeline([('vect', TfidfVectorizer()), ('clf', MultinomialNB())])` encadena transformación + modelo. Beneficios:
- Un solo objeto para fit/predict
- GridSearchCV sobre parámetros combinados
- Prevención de data leakage

### GridSearchCV

Búsqueda exhaustiva de hiperparámetros:
- `param_grid`: diccionario {`'vect__max_features'`: [100, 500, 1000], `'clf__alpha'`: [0.1, 0.5, 1.0]}
- `cv`: folds de validación cruzada
- `scoring`: métrica ('accuracy', 'f1_macro', etc.)

### Métricas de Evaluación

- **Matriz de confusión**: verdaderos vs predichos
- **classification_report**: precisión, recall, f1-score por clase
- **Accuracy**: (TP+TN)/(total) — cuidado con desbalanceo
- **Precision**: TP/(TP+FP) — qué % de predicciones positivas son correctas
- **Recall**: TP/(TP+FN) — qué % de positivos reales capturamos
- **F1-score**: media armónica de precisión y recall

### Instalación

```bash
pip install scikit-learn pandas matplotlib seaborn
```

---

## Ejemplos Prácticos

### 1. CountVectorizer: Convertir descripciones a matriz de conteo

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Instalación.*

1. Ejemplos Prácticos
2. 1. CountVectorizer: Convertir descripciones a matriz de conteo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

descripciones = [
    "Laptop gaming con procesador i9",
    "Auriculares inalámbricos con Bluetooth",
    "Monitor 4K de 27 pulgadas",
    "Teclado mecánico retroiluminado",
    "Mouse ergonómico inalámbrico"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(descripciones)

df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
print("Matriz de conteo (documentos × palabras):")
print(df.to_string())

print(f"\nVocabulario ({len(vectorizer.get_feature_names_out())} palabras):")
print(vectorizer.get_feature_names_out())
```

**Aplicación**: convertir descripciones de productos en features numéricas para modelos ML.

---

### 2. CountVectorizer con max_features y stop_words

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 2. CountVectorizer con max_features y stop_words

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from sklearn.feature_extraction.text import CountVectorizer

descripciones = [
    "El producto es muy bueno y tiene excelente calidad",
    "La calidad del sonido es increíblemente buena",
    "Mala experiencia, el producto llegó defectuoso",
    "Excelente calidad precio, lo recomiendo mucho",
    "Buen producto pero el precio es muy elevado"
]

vectorizer = CountVectorizer(
    max_features=100,
    stop_words='spanish',
    lowercase=True
)

X = vectorizer.fit_transform(descripciones)
print(f"Dimensiones: {X.shape}")
print(f"Vocabulario ({len(vectorizer.get_feature_names_out())} palabras):")
print(vectorizer.get_feature_names_out())
```

**Aplicación**: limitar vocabulario a las 100 palabras más informativas y eliminar stopwords en español.

---

### 3. CountVectorizer con ngram_range=(1,2)

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 3. CountVectorizer con ngram_range=(1,2)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

descripciones = [
    "cancelación de ruido activa",
    "cancelación de ruido pasiva",
    "alta definición 4K",
    "alta definición completa",
    "carga rápida inalámbrica"
]

vectorizer = CountVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(descripciones)

df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
print("Bigramas + unigramas como features:")
print(df.to_string())

print(f"\nTotal features: {len(vectorizer.get_feature_names_out())}")
```

**Aplicación**: capturar frases de 2 palabras como "cancelación de ruido" o "carga rápida" que tienen significado como unidad.

---

### 4. TfidfVectorizer: Peso TF-IDF para palabras relevantes

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 4. TfidfVectorizer: Peso TF-IDF para palabras relevantes

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

descripciones = [
    "Laptop gaming con procesador i9 y 32GB RAM",
    "Laptop ultraligera para oficina con 16GB RAM",
    "Monitor 4K gaming de 27 pulgadas",
    "Teclado mecánico gaming retroiluminado",
    "Monitor 4K para diseño gráfico profesional"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(descripciones)

df = pd.DataFrame(
    X.toarray(),
    columns=vectorizer.get_feature_names_out(),
    index=[f'Doc_{i}' for i in range(len(descripciones))]
)

print("Matriz TF-IDF:")
print(df.round(3))

print("\nPalabras con mayor TF-IDF por documento:")
for i, desc in enumerate(descripciones):
    row = X[i].toarray().flatten()
    top_indices = np.argsort(row)[-5:]
    top_palabras = [(vectorizer.get_feature_names_out()[j], row[j]) for j in top_indices if row[j] > 0]
    top_palabras.sort(key=lambda x: x[1], reverse=True)
    print(f"  Doc {i}: {[f'{p}({v:.3f})' for p, v in top_palabras]}")
```

**Aplicación**: TF-IDF destaca palabras distintivas de cada documento mientras reduce peso de términos comunes.

---

### 5. TfidfVectorizer con sublinear_tf=True

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 5. TfidfVectorizer con sublinear_tf=True

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

descripciones = [
    "gaming gaming gaming gaming laptop gaming gaming gaming gaming gaming",
    "laptop laptop laptop laptop laptop ultraligera oficina",
    "monitor 4K gaming 27 pulgadas 4K",
    "teclado mecánico gaming retroiluminado",
    "monitor 4K diseño gráfico profesional 4K"
]

# Sin sublinear_tf
vect1 = TfidfVectorizer(sublinear_tf=False)
X1 = vect1.fit_transform(descripciones)

# Con sublinear_tf (escala logarítmica)
vect2 = TfidfVectorizer(sublinear_tf=True)
X2 = vect2.fit_transform(descripciones)

print(f"TF-IDF normal vs sublinear (palabra 'gaming' en Doc 0):")
print(f"  Normal: {X1[0, vect1.vocabulary_['gaming']]:.4f}")
print(f"  Sublinear: {X2[0, vect2.vocabulary_['gaming']]:.4f}")
```

**Aplicación**: sublinear_tf reduce el impacto de palabras muy repetitivas usando escala logarítmica 1+log(tf).

---

### 6. HashingVectorizer: Vectorización hash

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 6. HashingVectorizer: Vectorización hash

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
import numpy as np

descripciones = [
    "Laptop gaming con procesador i9",
    "Auriculares inalámbricos con Bluetooth",
    "Monitor 4K de 27 pulgadas",
    "Teclado mecánico retroiluminado",
    "Mouse ergonómico inalámbrico"
]
categorias = ["tecnologia", "audio", "tecnologia", "accesorios", "accesorios"]

vectorizer = HashingVectorizer(n_features=100, alternate_sign=True, norm='l2')
X = vectorizer.fit_transform(descripciones)

print(f"Matriz hash dispersa: {X.shape}")
print(f"Tamaño en memoria: {X.data.nbytes} bytes")
print(f"Densidad: {X.nnz / (X.shape[0] * X.shape[1]):.4f}")

clf = SGDClassifier(loss='log')
clf.fit(X, categorias)
pred = clf.predict(vectorizer.transform(["Teclado inalámbrico RGB"]))
print(f"Predicción: {pred[0]}")
```

**Aplicación**: vectorización sin memoria para grandes datasets. No requiere mantener vocabulario. Ideal para streaming.

---

### 7. MultinomialNB: Clasificar categoría de producto

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 7. MultinomialNB: Clasificar categoría de producto

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

descripciones = [
    "Laptop gaming con procesador i9 y 32GB RAM",
    "Laptop ultraligera para oficina con SSD",
    "Monitor 4K de 27 pulgadas para gaming",
    "Monitor ultra wide para productividad",
    "Teclado mecánico retroiluminado RGB",
    "Teclado inalámbrico slim para oficina",
    "Auriculares con cancelación de ruido",
    "Audífonos deportivos resistentes al agua",
    "Mouse gaming con 16000 DPI",
    "Mouse ergonómico vertical inalámbrico"
]
categorias = [
    "laptop", "laptop",
    "monitor", "monitor",
    "teclado", "teclado",
    "audio", "audio",
    "mouse", "mouse"
]

X_train, X_test, y_train, y_test = train_test_split(
    descripciones, categorias, test_size=0.3, random_state=42
)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

clf = MultinomialNB()
clf.fit(X_train_vec, y_train)

pred = clf.predict(X_test_vec)
acc = accuracy_score(y_test, pred)
print(f"Precisión: {acc:.2f}")

for real, predicha, desc in zip(y_test, pred, X_test):
    print(f"  Real: {real:10s} | Pred: {predicha:10s} | {desc}")
```

**Aplicación**: clasificar automáticamente nuevos productos en categorías según su descripción.

---

### 8. ComplementNB: Mejor para datos desbalanceados

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 8. ComplementNB: Mejor para datos desbalanceados

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.model_selection import cross_val_score
import numpy as np

descripciones = [
    "Laptop gaming", "Laptop oficina", "Laptop programación",
    "Auriculares Bluetooth", "Audífonos con micrófono",
    "Teclado mecánico", "Teclado inalámbrico"
] * 10  # balanceado

# Desbalancear: mucho más de una clase
desc_imbalance = [
    "Laptop gaming", "Laptop oficina", "Laptop programación",
    "Laptop ultraligera", "Laptop para estudiantes",
    "Laptop creadores contenido", "Laptop empresarial",
    "Laptop 2 en 1", "Laptop touch", "Laptop barata",
    "Auriculares Bluetooth",
    "Teclado mecánico",
]
cat_imbalance = (["laptop"] * 10) + ["audio"] + ["teclado"]

vect = TfidfVectorizer()
X = vect.fit_transform(desc_imbalance)
y = np.array(cat_imbalance)

mnb = MultinomialNB()
cnb = ComplementNB()

scores_mnb = cross_val_score(mnb, X, y, cv=3)
scores_cnb = cross_val_score(cnb, X, y, cv=3)

print(f"MultinomialNB: {scores_mnb.mean():.3f} ± {scores_mnb.std():.3f}")
print(f"ComplementNB:  {scores_cnb.mean():.3f} ± {scores_cnb.std():.3f}")
print("ComplementNB es mejor para clases minoritarias")
```

**Aplicación**: cuando una categoría domina el catálogo (ej. 80% laptops, 10% audio, 10% teclados), ComplementNB rinde mejor.

---

### 9. BernoulliNB: Para matrices binarias

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 9. BernoulliNB: Para matrices binarias

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.model_selection import cross_val_score

descripciones = [
    "Laptop gaming con i9 y RAM",
    "Auriculares con Bluetooth",
    "Monitor 4K de 27 pulgadas",
    "Teclado mecánico RGB",
    "Mouse ergonómico vertical"
] * 6
categorias = (["laptop"] * 6 + ["audio"] * 6 + ["monitor"] * 6 +
              ["teclado"] * 6 + ["mouse"] * 6)

# Vectorización binaria (presencia/ausencia)
vect_binary = CountVectorizer(binary=True)
X_binary = vect_binary.fit_transform(descripciones)

vect_counts = CountVectorizer(binary=False)
X_counts = vect_counts.fit_transform(descripciones)

bnb = BernoulliNB()
mnb = MultinomialNB()

score_bnb = cross_val_score(bnb, X_binary, categorias, cv=3).mean()
score_mnb_counts = cross_val_score(mnb, X_counts, categorias, cv=3).mean()
score_mnb_binary = cross_val_score(mnb, X_binary, categorias, cv=3).mean()

print(f"BernoulliNB (binaria): {score_bnb:.3f}")
print(f"MultinomialNB (conteos): {score_mnb_counts:.3f}")
print(f"MultinomialNB (binaria): {score_mnb_binary:.3f}")
```

**Aplicación**: BernoulliNB es ideal cuando solo importa si una palabra aparece o no, no cuántas veces.

---

### 10. SGDClassifier con loss='log': Regresión logística

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 10. SGDClassifier con loss='log': Regresión logística

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

descripciones = [
    "Laptop gaming rápida con buenos gráficos",
    "Laptop para oficina ligera y eficiente",
    "Monitor 4K con colores vibrantes",
    "Monitor gaming 144Hz para jugar",
    "Teclado mecánico ruidoso pero preciso",
    "Teclado silencioso ideal para oficina",
] * 15

np.random.seed(42)
categorias = np.array(
    ["laptop"] * 15 + ["monitor"] * 15 + ["teclado"] * 15 +
    ["laptop"] * 15 + ["monitor"] * 15 + ["teclado"] * 15
)

X_train, X_test, y_train, y_test = train_test_split(
    descripciones, categorias, test_size=0.3, random_state=42
)

vect = TfidfVectorizer()
X_train_vec = vect.fit_transform(X_train)
X_test_vec = vect.transform(X_test)

clf = SGDClassifier(loss='log', max_iter=1000, random_state=42)
clf.fit(X_train_vec, y_train)
pred = clf.predict(X_test_vec)

print(classification_report(y_test, pred))
```

**Aplicación**: regresión logística con SGD da probabilidades calibradas y escala a grandes datasets.

---

### 11. SGDClassifier con loss='modified_huber': Robusto

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 11. SGDClassifier con loss='modified_huber': Robusto

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import numpy as np

descripciones = [
    "Laptop gaming rápida", "Laptop oficina ligera",
    "Laptop programación potente", "Monitor 4K",
    "Monitor gaming 144Hz", "Monitor ultra wide",
    "Teclado mecánico RGB", "Teclado silencioso",
    "Mouse gaming 16000 DPI", "Mouse ergonómico"
] + ["PRODUCTO FALSO !!!@@@ ###"] * 5  # outliers ruidosos

categorias = (["laptop"] * 3 + ["monitor"] * 3 +
              ["teclado"] * 2 + ["mouse"] * 2 + ["basura"] * 5)

vect = TfidfVectorizer()
X = vect.fit_transform(descripciones)

clf_log = SGDClassifier(loss='log', max_iter=1000, random_state=42)
clf_huber = SGDClassifier(loss='modified_huber', max_iter=1000, random_state=42)

from sklearn.model_selection import cross_val_score
score_log = cross_val_score(clf_log, X, categorias, cv=3).mean()
score_huber = cross_val_score(clf_huber, X, categorias, cv=3).mean()

print(f"loss='log':          {score_log:.3f}")
print(f"loss='modified_huber': {score_huber:.3f}")
print("modified_huber es más robusto a outliers en texto")
```

**Aplicación**: cuando hay textos ruidosos o mal clasificados en el dataset, modified_huber es más tolerante.

---

### 12. Pipeline: TfidfVectorizer + MultinomialNB

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 12. Pipeline: TfidfVectorizer + MultinomialNB

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

descripciones = [
    "Laptop gaming con procesador i9",
    "Laptop ultraligera para oficina",
    "Monitor 4K de 27 pulgadas",
    "Monitor gaming 144Hz curvo",
    "Teclado mecánico retroiluminado",
    "Teclado inalámbrico slim",
    "Auriculares con cancelación de ruido",
    "Audífonos deportivos Bluetooth",
    "Mouse gaming con 16000 DPI",
    "Mouse ergonómico vertical"
] * 5
categorias = (["laptop"] * 10 + ["monitor"] * 10 +
              ["teclado"] * 10 + ["audio"] * 10 + ["mouse"] * 10)

X_train, X_test, y_train, y_test = train_test_split(
    descripciones, categorias, test_size=0.3, random_state=42
)

pipeline = Pipeline([
    ('vect', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)

print(f"Precisión del pipeline: {accuracy_score(y_test, pred):.3f}")

# Probar nuevas descripciones
nuevas = [
    "Computador portátil gaming",
    "Pantalla 4K para diseño",
    "Teclado con luces RGB",
    "Cascos inalámbricos"
]
pred_nuevas = pipeline.predict(nuevas)
print("\nPredicciones para nuevos productos:")
for desc, cat in zip(nuevas, pred_nuevas):
    print(f"  {desc:40s} → {cat}")
```

**Aplicación**: Pipeline encapsula todo el proceso en un solo objeto. Facilita el reuso y evita data leakage.

---

### 13. Pipeline con parámetros

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 13. Pipeline con parámetros

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

descripciones = [
    "Laptop gaming con i9 y 32GB RAM",
    "Laptop oficina ultraligera con SSD",
    "Monitor 4K de 27 pulgadas gaming",
    "Monitor ultra wide productividad",
    "Teclado mecánico retroiluminado RGB",
    "Teclado inalámbrico slim oficina",
    "Auriculares cancelación ruido activa",
    "Audífonos deportivos Bluetooth 5.0",
    "Mouse gaming 16000 DPI RGB",
    "Mouse ergonómico vertical inalámbrico"
] * 3
categorias = (["laptop"] * 3 + ["monitor"] * 3 + ["teclado"] * 3 +
              ["audio"] * 3 + ["mouse"] * 3) * 3

pipeline = Pipeline([
    ('vect', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

# Probar diferentes configuraciones
configs = [
    {'vect__ngram_range': (1,1), 'vect__max_features': 100},
    {'vect__ngram_range': (1,2), 'vect__max_features': 100},
    {'vect__ngram_range': (1,1), 'vect__max_features': 50},
    {'vect__ngram_range': (1,2), 'vect__sublinear_tf': True},
]

for config in configs:
    pipeline.set_params(**config)
    scores = cross_val_score(pipeline, descripciones, categorias, cv=3)
    print(f"Config: {config} → Score: {scores.mean():.3f} ± {scores.std():.3f}")
```

**Aplicación**: probar rápida y limpiamente diferentes configuraciones de vectorización antes de grid search exhaustivo.

---

### 14. GridSearchCV sobre pipeline

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 14. GridSearchCV sobre pipeline

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
import pandas as pd

descripciones = [
    "Laptop gaming con i9 y 32GB RAM",
    "Laptop oficina ultraligera con SSD",
    "Monitor 4K de 27 pulgadas gaming",
    "Monitor ultra wide productividad",
    "Teclado mecánico retroiluminado RGB",
    "Teclado inalámbrico slim oficina",
    "Auriculares cancelación ruido activa",
    "Audífonos deportivos Bluetooth 5.0",
    "Mouse gaming 16000 DPI RGB",
    "Mouse ergonómico vertical inalámbrico"
] * 5
categorias = (["laptop"] * 5 + ["monitor"] * 5 + ["teclado"] * 5 +
              ["audio"] * 5 + ["mouse"] * 5) * 5

pipeline = Pipeline([
    ('vect', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

param_grid = {
    'vect__ngram_range': [(1,1), (1,2)],
    'vect__max_features': [50, 100],
    'vect__sublinear_tf': [True, False],
    'clf__alpha': [0.1, 0.5, 1.0],
}

grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
grid.fit(descripciones, categorias)

print(f"\nMejores parámetros: {grid.best_params_}")
print(f"Mejor score: {grid.best_score_:.3f}")

# Resultados detallados
resultados = pd.DataFrame(grid.cv_results_)
cols = [c for c in resultados.columns if 'param_' in c or 'mean_test_score' == c]
print("\nResultados de grid search:")
print(resultados[cols].sort_values('mean_test_score', ascending=False).head().to_string())
```

**Aplicación**: búsqueda óptima de hiperparámetros para maximizar precisión de clasificación.

---

### 15. Matriz de confusión

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 15. Matriz de confusión

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

descripciones = [
    "Laptop gaming i9", "Laptop oficina SSD",
    "Monitor 4K 27 pulgadas", "Monitor gaming 144Hz",
    "Teclado mecánico RGB", "Teclado inalámbrico",
    "Auriculares Bluetooth", "Audífonos deportivos",
    "Mouse gaming DPI", "Mouse ergonómico",
] * 10
categorias = (["laptop"] * 10 + ["monitor"] * 10 + ["teclado"] * 10 +
              ["audio"] * 10 + ["mouse"] * 10) * 2

X_train, X_test, y_train, y_test = train_test_split(
    descripciones, categorias, test_size=0.3, random_state=42
)

pipeline = Pipeline([
    ('vect', TfidfVectorizer()),
    ('clf', MultinomialNB())
])
pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)

cm = confusion_matrix(y_test, pred, labels=pipeline.classes_)

print("Matriz de confusión:")
print(f"{'':10s}", end="")
for c in pipeline.classes_:
    print(f"{c:10s}", end="")
print()

for i, real in enumerate(pipeline.classes_):
    print(f"{real:10s}", end="")
    for j in range(len(pipeline.classes_)):
        print(f"{cm[i, j]:<10d}", end="")
    print()

# Heatmap visual
sns.heatmap(cm, annot=True, fmt='d', xticklabels=pipeline.classes_,
            yticklabels=pipeline.classes_)
plt.title("Matriz de confusión - Clasificación de productos")
plt.ylabel("Real")
plt.xlabel("Predicho")
plt.show()
```

**Aplicación**: visualizar errores del clasificador. Ideal para identificar confusiones entre categorías similares.

---

### 16. classification_report de categorías

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 16. classification_report de categorías

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

descripciones = [
    "Laptop gaming i9 RAM", "Laptop oficina SSD",
    "Monitor 4K pulgadas", "Monitor gaming 144Hz",
    "Teclado mecánico RGB", "Teclado slim oficina",
    "Auriculares Bluetooth", "Audífonos deportivos",
    "Mouse gaming DPI", "Mouse ergonómico",
    "Tablet con pantalla 10 pulgadas",
    "Smartwatch con monitor cardíaco",
] * 5
categorias = (["laptop"] * 5 + ["monitor"] * 5 + ["teclado"] * 5 +
              ["audio"] * 5 + ["mouse"] * 5 + ["tablet"] * 5 +
              ["smartwatch"] * 5) * 5  # desbalance intencional

X_train, X_test, y_train, y_test = train_test_split(
    descripciones, categorias, test_size=0.3, random_state=42
)

pipeline = Pipeline([
    ('vect', TfidfVectorizer(max_features=50)),
    ('clf', MultinomialNB())
])
pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, pred, zero_division=0))
```

**Aplicación**: evaluar rendimiento por categoría. Identificar qué clases tienen bajo recall o precisión.

---

### 17. Palabras más frecuentes por categoría

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 17. Palabras más frecuentes por categoría

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd

descripciones = [
    "Laptop gaming i9 RAM", "Laptop oficina SSD ultraligera",
    "Monitor 4K 27 pulgadas OLED",
    "Monitor gaming 144Hz curvo",
    "Teclado mecánico RGB retroiluminado",
    "Teclado slim inalámbrico oficina",
    "Auriculares cancelación ruido Bluetooth",
    "Audífonos deportivos resistentes agua",
    "Mouse gaming 16000 DPI RGB",
    "Mouse ergonómico vertical inalámbrico"
]
categorias = ["laptop", "laptop", "monitor", "monitor",
              "teclado", "teclado", "audio", "audio", "mouse", "mouse"]

vect = TfidfVectorizer(max_features=100)
X = vect.fit_transform(descripciones)
clf = MultinomialNB()
clf.fit(X, categorias)

feature_names = vect.get_feature_names_out()

print("Palabras más importantes por categoría:")
for i, categoria in enumerate(clf.classes_):
    # Log-probabilidad de cada palabra dado la clase
    log_prob = clf.feature_log_prob_[i]
    top_indices = np.argsort(log_prob)[-8:]
    top_palabras = [feature_names[j] for j in top_indices]
    print(f"  {categoria}: {', '.join(reversed(top_palabras))}")
```

**Aplicación**: entender qué palabras usa el modelo para distinguir categorías. Útil para validar que las features son razonables.

---

### 18. Integrador: Clasificador completo de productos

```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. 18. Integrador: Clasificador completo de productos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---

python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np

# Dataset completo
catalogo = pd.DataFrame({
    'descripcion': [
        "Laptop gaming con procesador i9 y 32GB RAM",
        "Laptop ultraligera para oficina con SSD 512GB",
        "Monitor 4K de 27 pulgadas con HDR10",
        "Monitor gaming curvo 144Hz 32 pulgadas",
        "Teclado mecánico retroiluminado con switches Cherry MX",
        "Teclado inalámbrico slim para oficina",
        "Auriculares Bluetooth con cancelación de ruido activa",
        "Audífonos deportivos resistentes al agua IPX7",
        "Mouse gaming RGB con 16000 DPI",
        "Mouse ergonómico vertical inalámbrico",
    ] * 10,
    'categoria': (["laptop"] * 10 + ["monitor"] * 10 + ["teclado"] * 10 +
                  ["audio"] * 10 + ["mouse"] * 10) * 2
})

# 1. Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    catalogo['descripcion'], catalogo['categoria'],
    test_size=0.3, random_state=42, stratify=catalogo['categoria']
)

# 2. Pipeline con GridSearch
pipeline = Pipeline([
    ('vect', TfidfVectorizer()),
    ('clf', ComplementNB())
])

param_grid = {
    'vect__ngram_range': [(1,1), (1,2)],
    'vect__max_features': [50, 100, 200],
    'vect__sublinear_tf': [True, False],
    'clf__alpha': [0.1, 0.5, 1.0],
}

grid = GridSearchCV(
    pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
)

# 3. Entrenar
grid.fit(X_train, y_train)

# 4. Evaluar
best_model = grid.best_estimator_
pred = best_model.predict(X_test)

print(f"\n{'='*60}")
print(f"Mejores parámetros: {grid.best_params_}")
print(f"Mejor CV score: {grid.best_score_:.3f}")
print(f"Test accuracy: {np.mean(pred == y_test):.3f}")
print(f"{'='*60}")

print("\n--- Classification Report ---")
print(classification_report(y_test, pred))

print("\n--- Matriz de Confusión ---")
cm = confusion_matrix(y_test, pred, labels=best_model.classes_)
print(pd.DataFrame(cm, index=best_model.classes_, columns=best_model.classes_))

# 5. Predecir nuevos productos
nuevos_productos = [
    "Portátil para gaming con pantalla 15 pulgadas",
    "Monitor 4K para diseño profesional",
    "Teclado con iluminación LED personalizable",
    "Cascos con micrófono para gaming",
    "Ratón vertical para evitar dolor de muñeca",
]

print("\n--- Predicciones para nuevos productos ---")
pred_nuevos = best_model.predict(nuevos_productos)
prob_nuevos = best_model.predict_proba(nuevos_productos)

for desc, cat, prob in zip(nuevos_productos, pred_nuevos, prob_nuevos):
    max_prob = prob.max() * 100
    print(f"  {desc:55s} → {cat:10s} ({max_prob:.1f}% confianza)")
```

**Aplicación**: pipeline completo de clasificación listo para producción. Entrena, evalúa y predice automáticamente la categoría de nuevos productos.

---

## Resumen

| Técnica | Vectorización | Clasificador | Ideal para |
|---|---|---|---|
| CountVectorizer | Conteo de palabras | Naive Bayes/Bernoulli | Categorización simple |
| Count + ngram_range | Unigramas + bigramas | Cualquiera | Capturar frases |
| TfidfVectorizer | TF-IDF | MultinomialNB, SGD | Textos con ruido |
| HashingVectorizer | Hash | SGD | Grandes volúmenes |
| Pipeline | Cualquiera | Cualquiera | Flujo integrado |
| GridSearchCV | Cualquiera | Cualquiera | Optimización automática |

| Clasificador | Ideal para | Ventaja principal |
|---|---|---|
| MultinomialNB | Conteos de palabras | Simple, rápido, buen baseline |
| ComplementNB | Datos desbalanceados | Mejor en clases minoritarias |
| BernoulliNB | Features binarias | Presencia/ausencia |
| SGDClassifier(log) | Grandes datasets | Escalable, probabilidades |
| SGDClassifier(modified_huber) | Outliers | Robusto a ruido |

---

## Ejercicios

1. **Comparación de vectorizadores**: con un dataset de 50 descripciones de productos en 5 categorías, compara CountVectorizer vs TfidfVectorizer vs HashingVectorizer usando MultinomialNB. Reporta accuracy en validación cruzada de 5 folds.

2. **GridSearch exhaustivo**: sobre el mismo dataset, realiza GridSearchCV probando: `ngram_range=(1,1),(1,2),(2,2)`, `max_features=50,100,200`, `sublinear_tf=True,False` y `alpha=0.1,0.5,1.0`. Encuentra la mejor combinación.

3. **Clasificación multiclase con SGD**: usa SGDClassifier con loss='log', 'modified_huber', 'hinge' y compara resultados sobre un dataset de categorías de productos. ¿Cuál loss da mejor F1-macro?

4. **Análisis de errores**: genera una matriz de confusión para tus predicciones. Identifica las 3 confusiones más comunes entre categorías (ej. laptop vs tablet). ¿Por qué ocurren? Propón mejoras.

5. **Features importantes**: para un modelo MultinomialNB entrenado, extrae las 10 palabras con mayor probabilidad logarítmica por categoría. ¿Tienen sentido para el dominio de productos? ¿Hay palabras que deberían ser stopwords?

6. **Pipeline + GridSearch en datos reales**: crea un dataset con 30 descripciones de productos colombianos (tecnología, hogar, ropa, deporte, libros). Entrena un pipeline optimizado con GridSearch y reporta métricas de test.

7. **Clasificación binaria**: convierte el problema en binario (ej. "electrónica" vs "no electrónica"). Compara BernoulliNB vs ComplementNB para esta tarea. ¿Cuál es mejor para cada clase?

8. **Sistema en producción**: diseña una clase `ClasificadorProductos` que: (a) se entrene con un DataFrame, (b) guarde el pipeline con `joblib.dump`, (c) tenga método `predecir(texto)` que devuelva categoría + probabilidades. Pruébala con 10 productos nuevos.
