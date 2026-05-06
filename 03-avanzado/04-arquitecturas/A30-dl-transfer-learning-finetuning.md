# A30 - Deep Learning: Transfer Learning y Fine-tuning para Ventas

## Fundamentos Teóricos

**Transfer Learning** (aprendizaje por transferencia) aprovecha conocimiento adquirido en una tarea para resolver otra tarea relacionada. En deep learning, consiste en tomar un modelo pre-entrenado en un dataset masivo (ImageNet, Wikipedia) y adaptarlo a un dominio específico (productos, reseñas).

### Conceptos Clave

| Técnica | Descripción | Cuándo usarla |
|---------|-------------|---------------|
| **Feature Extractor** | Congelar todo, añadir clasificador | Pocos datos (< 1k imágenes) |
| **Fine-tuning** | Descongelar últimas capas | Datos moderados (1k-10k) |
| **Progressive Unfreezing** | Descongelar gradualmente | Cuando fine-tuning completo da overfitting |
| **Discriminative LR** | LR diferente por capa | Fine-tuning de modelos profundos |

### Modelos Pre-entrenados (tf.keras.applications)

| Modelo | Parámetros | Año | Característica |
|--------|-----------|-----|----------------|
| VGG16 | 138M | 2014 | Simple, profundo |
| ResNet50 | 25.6M | 2015 | Residual connections |
| EfficientNetB0 | 5.3M | 2019 | Eficiente (SOTA) |
| MobileNetV2 | 3.5M | 2018 | Ligero, móvil |
| InceptionV3 | 23.9M | 2016 | Factorized convolutions |
| DenseNet121 | 8.1M | 2017 | Dense connections |

### Estrategias de Fine-tuning

1. **Feature extractor**: congelar todas las capas pre-entrenadas, añadir capas nuevas
2. **Fine-tuning parcial**: descongelar las últimas N capas
3. **Fine-tuning completo**: descongelar todo, lr muy pequeño (1e-5)
4. **Discriminative learning rates**: lr más pequeño para capas pre-entrenadas, más grande para capas nuevas
5. **Progressive unfreezing**: empezar con todo congelado, descongelar de a poco

### Transfer Learning para Texto (BERT)
- BERT (Bidirectional Encoder Representations from Transformers)
- Modelo pre-entrenado en Wikipedia + BookCorpus
- Congelar embeddings de BERT como feature extractor
- Añadir clasificador Dense para tarea específica
- Fine-tuning completo con pocas épocas (2-4)

### Data Augmentation
- Aumenta datos de entrenamiento con transformaciones
- Para imágenes: rotation, flip, zoom, shift, brightness, contrast
- Reduce overfitting, mejora generalización
- Se aplica solo al train, no al val/test

---

## Ejemplos Prácticos

### Ejemplo 1: Cargar ResNet50 Pre-entrenado (sin top)

```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

np.random.seed(42)
tf.random.set_seed(42)

# Cargar ResNet50 sin la cabeza clasificadora (include_top=False)
resnet = ResNet50(weights='imagenet', include_top=False,
                  input_shape=(224, 224, 3))
print("ResNet50 cargado:")
resnet.summary()

# Ver cuántas capas tiene
print(f"Número total de capas: {len(resnet.layers)}")

# Ver las últimas capas
print("\nÚltimas 5 capas:")
for layer in resnet.layers[-5:]:
    print(f"  {layer.name:20s} | trainable={layer.trainable}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: Cargar ResNet50 Pre-entrenado (sin top).*

1. Cargar ResNet50 sin la cabeza clasificadora (include_top=False)
2. Ver cuántas capas tiene
3. Ver las últimas capas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: Feature Extractor — Congelar Todo, Añadir Clasificador

```python
# Congelar todas las capas del modelo base
resnet.trainable = False

# Añadir nuevas capas clasificadoras
entrada = tf.keras.Input(shape=(224, 224, 3))
x = resnet(entrada)
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
salida = Dense(10, activation='softmax')(x)  # 10 categorías de productos

modelo_fe = Model(inputs=entrada, outputs=salida)
modelo_fe.compile(optimizer=Adam(0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
modelo_fe.summary()

# Verificar que el modelo base está congelado
print(f"\nCapas entrenables en ResNet50: {sum(l.trainable for l in resnet.layers)}")
print(f"Total parámetros entrenables del modelo: {sum(tf.keras.backend.count_params(w) for w in modelo_fe.trainable_weights):,}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: Feature Extractor — Congelar Todo, Añadir Clasificador.*

1. Congelar todas las capas del modelo base
2. Añadir nuevas capas clasificadoras
3. Verificar que el modelo base está congelado

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: Fine-tuning — Descongelar Últimas 10 Capas

```python
# Descongelar el modelo base
resnet.trainable = True

# Congelar todo excepto las últimas 10 capas
for layer in resnet.layers[:-10]:
    layer.trainable = False

# Verificar
print("Estado de trainable en últimas 15 capas:")
for layer in resnet.layers[-15:]:
    print(f"  {layer.name:25s} | trainable={layer.trainable}")

# Recompilar (necesario después de cambiar trainable)
modelo_fe.compile(optimizer=Adam(learning_rate=1e-5),  # lr pequeño para fine-tuning
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Ver parámetros entrenables
trainable = sum(tf.keras.backend.count_params(w) for w in modelo_fe.trainable_weights)
total = sum(tf.keras.backend.count_params(w) for w in modelo_fe.weights)
print(f"\nParámetros entrenables: {trainable:,} / {total:,} ({trainable/total:.1%})")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: Fine-tuning — Descongelar Últimas 10 Capas.*

1. Descongelar el modelo base
2. Congelar todo excepto las últimas 10 capas
3. Verificar
4. Recompilar (necesario después de cambiar trainable)
5. Ver parámetros entrenables

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: Fine-tuning con Diferentes Learning Rates

```python
# Discriminative learning rates usando diferentes optimizadores
# Capas pre-entrenadas (mayoría de ResNet)
resnet.trainable = True
for layer in resnet.layers[:-15]:
    layer.trainable = False

# Compilar con lr diferente para capas base vs nuevas
# Estrategia: usar 2 fases de entrenamiento
# Fase 1: solo capas nuevas (con lr alto)
# Fase 2: fine-tuning (lr bajo)

# Fase 1: solo clasificador
resnet.trainable = False
modelo_fe.compile(optimizer=Adam(0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
print("Fase 1: Solo clasificador (lr=0.001)")

# Dataset sintético para demostración
X_dummy = np.random.randn(100, 224, 224, 3)
y_dummy = np.random.randint(0, 10, 100)
# modelo_fe.fit(X_dummy, y_dummy, epochs=5, verbose=0)  # Fase 1

# Fase 2: fine-tuning con lr bajo
resnet.trainable = True
for layer in resnet.layers[:-15]:
    layer.trainable = False
modelo_fe.compile(optimizer=Adam(1e-5),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
print("Fase 2: Fine-tuning (lr=1e-5)")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: Fine-tuning con Diferentes Learning Rates.*

1. Discriminative learning rates usando diferentes optimizadores
2. Capas pre-entrenadas (mayoría de ResNet)
3. Compilar con lr diferente para capas base vs nuevas
4. Estrategia: usar 2 fases de entrenamiento
5. Fase 1: solo capas nuevas (con lr alto)
6. Fase 2: fine-tuning (lr bajo)
7. Fase 1: solo clasificador
8. Dataset sintético para demostración

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: Progressive Unfreezing — Descongelar 20%, 50%, 80%

```python
resnet_pu = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

def unfreeze_percentage(model, percentage):
    """Descongela un porcentaje de las capas (desde el final)"""
    total = len(model.layers)
    n_unfreeze = int(total * percentage / 100)
    for layer in model.layers[:-n_unfreeze]:
        layer.trainable = False
    for layer in model.layers[-n_unfreeze:]:
        layer.trainable = True
    return model

# Progressive unfreezing: 0% → 20% → 50% → 80%
for pct in [0, 20, 50, 80]:
    modelo = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    modelo = unfreeze_percentage(modelo, pct)
    n_trainable = sum(l.trainable for l in modelo.layers)
    print(f"Unfreeze {pct:3d}% | Capas entrenables: {n_trainable}/{len(modelo.layers)}")

# En la práctica: entrenar 5 épocas con 20%, luego 5 con 50%, etc.
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: Progressive Unfreezing — Descongelar 20%, 50%, 80%.*

1. Progressive unfreezing: 0% → 20% → 50% → 80%
2. En la práctica: entrenar 5 épocas con 20%, luego 5 con 50%, etc.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: Discriminative Learning Rates (lr diferente por capa)

```python
# Implementación manual de discriminative learning rates
# Congelar capas tempranas, lr medio para capas medias, lr alto para nuevas

resnet_dlr = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Crear 3 grupos de capas
grupo1 = resnet_dlr.layers[:50]   # tempranas → congeladas
grupo2 = resnet_dlr.layers[50:120] # medias → lr=1e-6
grupo3 = resnet_dlr.layers[120:]   # últimas → lr=1e-5
grupo4 = []  # nuevas capas → lr=1e-3

for layer in grupo1:
    layer.trainable = False
for layer in grupo2:
    layer.trainable = True
for layer in grupo3:
    layer.trainable = True

print("Grupos de discriminative learning rates:")
print(f"  Grupo 1 (capas 0-50): congeladas")
print(f"  Grupo 2 (capas 50-120): lr=1e-6")
print(f"  Grupo 3 (capas 120+): lr=1e-5")
print(f"  Grupo 4 (nuevas capas): lr=1e-3")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: Discriminative Learning Rates (lr diferente por capa).*

1. Implementación manual de discriminative learning rates
2. Congelar capas tempranas, lr medio para capas medias, lr alto para nuevas
3. Crear 3 grupos de capas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: EfficientNetB0 — Modelo Eficiente para Clasificación de Productos

```python
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator

effnet = EfficientNetB0(weights='imagenet', include_top=False,
                        input_shape=(224, 224, 3))
effnet.trainable = False

modelo_eff = Sequential([
    effnet,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(20, activation='softmax')  # 20 categorías de productos
])

modelo_eff.compile(optimizer=Adam(0.001),
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
modelo_eff.summary()

print(f"EfficientNetB0 parámetros: {effnet.count_params():,}")
print(f"Modelo completo parámetros: {modelo_eff.count_params():,}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: EfficientNetB0 — Modelo Eficiente para Clasificación de Productos.*

1. `from tensorflow.keras.applications import EfficientNetB0` — Importa las librerías necesarias para el análisis.
2. `from tensorflow.keras.preprocessing.image import ImageDataGenerator` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: MobileNetV2 — Modelo Ligero para Dispositivos Móviles

```python
from tensorflow.keras.applications import MobileNetV2

mobilenet = MobileNetV2(weights='imagenet', include_top=False,
                        input_shape=(224, 224, 3))
mobilenet.trainable = False

modelo_mobile = Sequential([
    mobilenet,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

modelo_mobile.compile(optimizer=Adam(0.001),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
modelo_mobile.summary()

params_mobile = modelo_mobile.count_params()
params_eff = modelo_eff.count_params()
print(f"MobileNetV2 params: {params_mobile:,}")
print(f"EfficientNetB0 params: {params_eff:,}")
print(f"MobileNetV2 es {params_eff/params_mobile:.1f}x más pequeño que EfficientNetB0")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: MobileNetV2 — Modelo Ligero para Dispositivos Móviles.*

1. `from tensorflow.keras.applications import MobileNetV2` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: Adaptar Última Capa a Número de Categorías de Productos

```python
def crear_modelo_transfer(modelo_base, num_clases, input_shape=(224, 224, 3)):
    """Adapta cualquier modelo pre-entrenado a num_clases categorías"""
    modelo_base.trainable = False
    entrada = tf.keras.Input(shape=input_shape)
    x = modelo_base(entrada)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    salida = Dense(num_clases, activation='softmax')(x)
    return Model(inputs=entrada, outputs=salida)

# Probar con diferentes modelos y categorías
for modelo_nombre, modelo_fn in [('ResNet50', ResNet50),
                                  ('EfficientNetB0', EfficientNetB0),
                                  ('MobileNetV2', MobileNetV2)]:
    base = modelo_fn(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = crear_modelo_transfer(base, num_clases=15)  # 15 categorías de productos
    model.compile(optimizer=Adam(0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print(f"{modelo_nombre:15s} | Params: {model.count_params():>10,}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: Adaptar Última Capa a Número de Categorías de Productos.*

1. Probar con diferentes modelos y categorías

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: Transfer Learning para Texto — Cargar BERT Embeddings Congelados

```python
import tensorflow_hub as hub
import tensorflow_text as text

# Usar BERT pre-entrenado de TensorFlow Hub para embeddings de reseñas de productos
# Nota: requiere `pip install tensorflow-hub tensorflow-text`

def crear_modelo_bert(num_clases=5):
    """Modelo con embeddings BERT congelados + clasificador"""
    # Capa de preprocesamiento y encoder BERT
    text_input = tf.keras.Input(shape=(), dtype=tf.string)

    # Cargar modelo BERT pre-entrenado
    # En un entorno real, descargar de TF Hub:
    # preprocessor = hub.KerasLayer(
    #     "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    # encoder = hub.KerasLayer(
    #     "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")
    # Para este ejemplo, usamos un placeholder

    # Placeholder para simular BERT embeddings (768-d)
    bert_embeddings = tf.keras.layers.Lambda(
        lambda x: tf.zeros((tf.shape(x)[0], 768))
    )(text_input)
    bert_embeddings.trainable = False  # congelado

    # Clasificador
    x = Dense(256, activation='relu')(bert_embeddings)
    x = Dropout(0.3)(x)
    salida = Dense(num_clases, activation='softmax')(x)

    return Model(inputs=text_input, outputs=salida)

modelo_bert = crear_modelo_bert(num_clases=5)
modelo_bert.compile(optimizer=Adam(0.001),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
modelo_bert.summary()

print("BERT embeddings: congelados (trainable=False)")
print("Clasificador Dense: entrenable")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: Transfer Learning para Texto — Cargar BERT Embeddings Congelados.*

1. Usar BERT pre-entrenado de TensorFlow Hub para embeddings de reseñas de productos
2. Nota: requiere `pip install tensorflow-hub tensorflow-text`
3. Capa de preprocesamiento y encoder BERT
4. Cargar modelo BERT pre-entrenado
5. En un entorno real, descargar de TF Hub:
6. preprocessor = hub.KerasLayer(
7. "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
8. encoder = hub.KerasLayer(

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: BERT + Clasificador Dense para Clasificación de Reseñas

```python
# Simular entrenamiento de BERT con reseñas de productos
np.random.seed(42)
tf.random.set_seed(42)

# Reseñas simuladas
textos = np.array([
    "Excelente producto, muy recomendado",
    "Mala calidad, no volvería a comprar",
    "Buen precio, calidad aceptable",
    "No funciona como esperaba",
    "Perfecto para lo que necesito",
    # ... más reseñas
] * 20)  # 100 reseñas
sentimientos = np.array([2, 0, 1, 0, 2] * 20)  # 0=malo, 1=regular, 2=bueno

# Simplificación: usar embedding aleatorio como features BERT
X_bert_sim = np.random.randn(len(textos), 768)

modelo_bert_clasif = Sequential([
    Dense(256, activation='relu', input_shape=(768,)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')  # 3 clases de sentimiento
])
modelo_bert_clasif.compile(optimizer=Adam(0.001),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

print("Entrenando clasificador sobre embeddings BERT congelados...")
# En un escenario real, los embeddings vendrían del modelo BERT real
# modelo_bert_clasif.fit(X_bert_sim, sentimientos, epochs=10, batch_size=16, validation_split=0.2)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: BERT + Clasificador Dense para Clasificación de Reseñas.*

1. Simular entrenamiento de BERT con reseñas de productos
2. Reseñas simuladas
3. ... más reseñas
4. Simplificación: usar embedding aleatorio como features BERT
5. En un escenario real, los embeddings vendrían del modelo BERT real
6. modelo_bert_clasif.fit(X_bert_sim, sentimientos, epochs=10, batch_size=16, validation_split=0.2)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: Data Augmentation para Imágenes (rotation, flip, zoom)

```python
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomBrightness, RandomContrast

# Capas de data augmentation (se aplican solo durante entrenamiento)
data_augmentation = tf.keras.Sequential([
    RandomFlip('horizontal'),
    RandomRotation(0.2),
    RandomZoom(0.2),
    RandomBrightness(0.1),
    RandomContrast(0.1),
])

# Incorporar en el modelo (se ejecuta solo en train)
entrada = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(entrada)
x = resnet(x)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
salida = Dense(10, activation='softmax')(x)

modelo_aug = Model(inputs=entrada, outputs=salida)
modelo_aug.compile(optimizer=Adam(0.001),
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
print("Data augmentation integrado en el modelo:")
print(f"  RandomFlip(horizontal)")
print(f"  RandomRotation(0.2)")
print(f"  RandomZoom(0.2)")
print(f"  RandomBrightness(0.1)")
print(f"  RandomContrast(0.1)")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: Data Augmentation para Imágenes (rotation, flip, zoom).*

1. Capas de data augmentation (se aplican solo durante entrenamiento)
2. Incorporar en el modelo (se ejecuta solo en train)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: Reentrenar Solo el Clasificador (Capa Final) con Datos Pequeños

```python
# Escenario: solo 100 imágenes por categoría para entrenar
# Estrategia: feature extractor + clasificador lineal

resnet_small = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
resnet_small.trainable = False

modelo_small = Sequential([
    resnet_small,
    GlobalAveragePooling2D(),
    Dense(10, activation='softmax')  # Solo la capa final!
])

modelo_small.compile(optimizer=Adam(0.01),
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])

print("Modelo para datos pequeños (solo clasificador lineal):")
modelo_small.summary()
print(f"Entrenable: solo la capa Dense final ({10*2048+10} parámetros)")
print(f"Total parámetros: {modelo_small.count_params():,}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: Reentrenar Solo el Clasificador (Capa Final) con Datos Pequeños.*

1. Escenario: solo 100 imágenes por categoría para entrenar
2. Estrategia: feature extractor + clasificador lineal

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: Fine-tuning con Datos Limitados (100 imágenes por categoría)

```python
# Estrategia completa para datos limitados:
# 1. Data augmentation fuerte
# 2. Feature extractor (base congelada)
# 3. Dropout alto para evitar overfitting
# 4. Early stopping agresivo

effnet_ft = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
effnet_ft.trainable = False

modelo_limitado = Sequential([
    # Data augmentation
    RandomFlip('horizontal'),
    RandomRotation(0.15),
    RandomZoom(0.15),
    RandomBrightness(0.05),

    # Backbone congelado
    effnet_ft,
    GlobalAveragePooling2D(),

    # Clasificador con regularización fuerte
    Dense(128, activation='relu'),
    Dropout(0.5),  # dropout alto
    Dense(5, activation='softmax')
])

modelo_limitado.compile(optimizer=Adam(0.001),
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])

print("Modelo para datos limitados:")
modelo_limitado.summary()

# Early stopping agresivo
early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: Fine-tuning con Datos Limitados (100 imágenes por categoría).*

1. Estrategia completa para datos limitados:
2. 1. Data augmentation fuerte
3. 2. Feature extractor (base congelada)
4. 3. Dropout alto para evitar overfitting
5. 4. Early stopping agresivo
6. Data augmentation
7. Backbone congelado
8. Clasificador con regularización fuerte

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: Comparar — Entrenar desde Cero vs Transfer Learning vs Fine-tuning

```python
# Escenario: 500 imágenes por categoría, 5 categorías

# 1. Entrenar desde cero (random weights)
modelo_scratch = Sequential([
    RandomFlip('horizontal'),
    RandomRotation(0.1),
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    GlobalAvgPooling2D(),
    Dense(128, activation='relu'),
    Dense(5, activation='softmax')
])
modelo_scratch.compile(optimizer=Adam(0.001),
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])
print(f"Entrenar desde cero: {modelo_scratch.count_params():,} parámetros")

# 2. Transfer learning (feature extractor)
resnet_tl = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
resnet_tl.trainable = False
modelo_tl = Sequential([
    resnet_tl,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(5, activation='softmax')
])
modelo_tl.compile(optimizer=Adam(0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
print(f"Transfer learning: {modelo_tl.count_params():,} parámetros")

# 3. Fine-tuning
resnet_ft = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
resnet_ft.trainable = True
for layer in resnet_ft.layers[:-20]:
    layer.trainable = False
modelo_ft = Sequential([
    resnet_ft,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(5, activation='softmax')
])
modelo_ft.compile(optimizer=Adam(1e-5),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
print(f"Fine-tuning: {modelo_ft.count_params():,} parámetros")

print("\nComparación esperada:")
print("  Transfer learning > Fine-tuning > Scratch (con pocos datos)")
print("  Fine-tuning > Transfer learning > Scratch (con suficientes datos)")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: Comparar — Entrenar desde Cero vs Transfer Learning vs Fine-tuning.*

1. Escenario: 500 imágenes por categoría, 5 categorías
2. 1. Entrenar desde cero (random weights)
3. 2. Transfer learning (feature extractor)
4. 3. Fine-tuning

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 16: Curva de Aprendizaje — Transfer Converge Más Rápido

```python
# Simular curvas de aprendizaje para los 3 enfoques
np.random.seed(42)

epochs = 50
# Transfer learning: parte de loss bajo, converge rápido
tl_loss = 1.5 * np.exp(-0.15 * np.arange(epochs)) + 0.1 + 0.02*np.random.randn(epochs)
# Fine-tuning: parte de loss medio, converge a mejor mínimo
ft_loss = 2.0 * np.exp(-0.10 * np.arange(epochs)) + 0.08 + 0.015*np.random.randn(epochs)
# Scratch: parte de loss alto, converge lento
sc_loss = 3.5 * np.exp(-0.06 * np.arange(epochs)) + 0.15 + 0.03*np.random.randn(epochs)

# Suavizar
def suavizar(y, alpha=0.3):
    s = np.zeros_like(y)
    s[0] = y[0]
    for t in range(1, len(y)):
        s[t] = alpha * y[t] + (1 - alpha) * s[t-1]
    return s

plt.figure(figsize=(10, 5))
plt.plot(suavizar(sc_loss), label='Scratch (desde cero)', linewidth=2)
plt.plot(suavizar(tl_loss), label='Transfer Learning', linewidth=2)
plt.plot(suavizar(ft_loss), label='Fine-tuning', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Comparación de convergencia: Scratch vs Transfer Learning vs Fine-tuning')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("Observaciones:")
print("  - Transfer learning converge más rápido (parte de buenos features)")
print("  - Fine-tuning suele dar mejor accuracy final")
print("  - Scratch necesita muchos datos para igualar")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: Curva de Aprendizaje — Transfer Converge Más Rápido.*

1. Simular curvas de aprendizaje para los 3 enfoques
2. Transfer learning: parte de loss bajo, converge rápido
3. Fine-tuning: parte de loss medio, converge a mejor mínimo
4. Scratch: parte de loss alto, converge lento
5. Suavizar

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 17: Save y Load Modelo Fine-tuneado

```python
# Guardar modelo fine-tuneado
modelo_ft.save('modelo_productos_finetuneado.h5')
print("Modelo guardado: modelo_productos_finetuneado.h5")

# Cargar modelo
modelo_cargado = tf.keras.models.load_model('modelo_productos_finetuneado.h5')
print("Modelo cargado exitosamente")

# Verificar arquitectura
modelo_cargado.summary()

# Guardar solo los pesos (más ligero)
modelo_ft.save_weights('pesos_finetuneado.h5')
print("Pesos guardados: pesos_finetuneado.h5")

# Cargar pesos en modelo con misma arquitectura
# modelo_nuevo.load_weights('pesos_finetuneado.h5')
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: Save y Load Modelo Fine-tuneado.*

1. Guardar modelo fine-tuneado
2. Cargar modelo
3. Verificar arquitectura
4. Guardar solo los pesos (más ligero)
5. Cargar pesos en modelo con misma arquitectura
6. modelo_nuevo.load_weights('pesos_finetuneado.h5')

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 18: Integrador — Fine-tuning de Modelo para Clasificación de Productos

```python
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom
from sklearn.metrics import classification_report

np.random.seed(42)
tf.random.set_seed(42)

# Dataset simulado de imágenes de productos (224x224, 15 categorías)
n_train = 3000
n_val = 500
X_train_img = np.random.randn(n_train, 224, 224, 3)
y_train_img = np.random.randint(0, 15, n_train)
X_val_img = np.random.randn(n_val, 224, 224, 3)
y_val_img = np.random.randint(0, 15, n_val)

# Cargar modelo base
base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base.trainable = False

# Modelo completo con data augmentation
modelo_integrador = Sequential([
    # Data augmentation (solo train)
    RandomFlip('horizontal'),
    RandomRotation(0.2),
    RandomZoom(0.15),

    # Backbone
    base,
    GlobalAveragePooling2D(),

    # Clasificador personalizado
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(15, activation='softmax')
])

modelo_integrador.compile(optimizer=Adam(0.001),
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
modelo_integrador.summary()

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7)
]

print("\n=== Fase 1: Feature Extractor (solo clasificador) ===")
h_fase1 = modelo_integrador.fit(X_train_img, y_train_img, epochs=30,
                                batch_size=32, validation_data=(X_val_img, y_val_img),
                                callbacks=callbacks, verbose=1)

print("\n=== Fase 2: Fine-tuning (descongelar parte del backbone) ===")
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

modelo_integrador.compile(optimizer=Adam(1e-5),
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

h_fase2 = modelo_integrador.fit(X_train_img, y_train_img, epochs=20,
                                batch_size=32, validation_data=(X_val_img, y_val_img),
                                callbacks=callbacks, verbose=1)

# Evaluación final
y_pred = modelo_integrador.predict(X_val_img, verbose=0)
y_pred_clases = np.argmax(y_pred, axis=1)
accuracy = (y_pred_clases == y_val_img).mean()
print(f"\n=== Resultados Finales ===")
print(f"Accuracy en validación: {accuracy:.4f}")
print(f"Mejor val_loss: {min(h_fase2.history['val_loss']):.4f}")

# Curvas de aprendizaje
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
# Combinar historia de fase 1 y 2
loss_f1 = h_fase1.history['loss']
loss_f2 = h_fase2.history['loss']
val_loss_f1 = h_fase1.history['val_loss']
val_loss_f2 = h_fase2.history['val_loss']

plt.plot(loss_f1, label='Train (Fase 1)', alpha=0.7)
plt.plot(val_loss_f1, label='Val (Fase 1)', alpha=0.7)
plt.axvline(x=len(loss_f1)-1, color='gray', linestyle='--', alpha=0.5)
plt.plot(range(len(loss_f1), len(loss_f1)+len(loss_f2)), loss_f2, label='Train (Fase 2)', alpha=0.7)
plt.plot(range(len(loss_f1), len(loss_f1)+len(loss_f2)), val_loss_f2, label='Val (Fase 2)', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Curva de aprendizaje (2 fases)')
plt.legend()

plt.subplot(1, 2, 2)
acc_f1 = h_fase1.history['accuracy']
acc_f2 = h_fase2.history['accuracy']
val_acc_f1 = h_fase1.history['val_accuracy']
val_acc_f2 = h_fase2.history['val_accuracy']
plt.plot(acc_f1, label='Train (Fase 1)', alpha=0.7)
plt.plot(val_acc_f1, label='Val (Fase 1)', alpha=0.7)
plt.axvline(x=len(acc_f1)-1, color='gray', linestyle='--', alpha=0.5)
plt.plot(range(len(acc_f1), len(acc_f1)+len(acc_f2)), acc_f2, label='Train (Fase 2)', alpha=0.7)
plt.plot(range(len(acc_f1), len(acc_f1)+len(acc_f2)), val_acc_f2, label='Val (Fase 2)', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy (2 fases)')
plt.legend()
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador — Fine-tuning de Modelo para Clasificación de Productos.*

1. Dataset simulado de imágenes de productos (224x224, 15 categorías)
2. Cargar modelo base
3. Modelo completo con data augmentation
4. Data augmentation (solo train)
5. Backbone
6. Clasificador personalizado
7. Callbacks
8. Evaluación final

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejercicios

1. **Feature extractor con ResNet50 para clasificación de productos**: Carga ResNet50 pre-entrenado, congélalo, añade GlobalAveragePooling2D + Dense(128, relu) + Dense(20, softmax). Entrena en un dataset simulado de 5000 imágenes en 20 categorías. Reporta accuracy de validación y número de parámetros entrenables.

2. **Fine-tuning de MobileNetV2 para detección de defectos**: Usa MobileNetV2 pre-entrenado para clasificar imágenes de productos como "defectuoso" o "normal" (2 clases). Descongela las últimas 15 capas y fine-tunea con lr=1e-5. Compara accuracy vs feature extractor.

3. **Progressive unfreezing con VGG16**: Implementa progressive unfreezing con VGG16 para clasificación de 10 categorías de productos. Entrena en 4 etapas: (a) solo clasificador, (b) descongelar últimas 15%, (c) últimas 30%, (d) últimas 50%. Reporta accuracy después de cada etapa.

4. **Discriminative learning rates para ResNet50**: Implementa 3 grupos de LR para ResNet50: capas iniciales (lr=1e-6), capas medias (lr=1e-5), capas finales + clasificador (lr=1e-3). Compara convergencia vs un solo lr global.

5. **Data augmentation para clasificación de inventario**: Crea un pipeline de data augmentation con: rotation(0.3), zoom(0.2), flip horizontal, brightness(0.15), contrast(0.15). Entrena un clasificador de 8 categorías con y sin augmentation usando solo 200 imágenes por categoría. ¿Cuánto mejora la augmentation?

6. **Transfer learning para texto con BERT**: Carga un modelo BERT pre-entrenado (via TF Hub o transformers) y congela los embeddings. Añade un clasificador Dense (2 capas, 256 y 128) para clasificar reseñas de productos en 5 estrellas. Entrena con 2000 reseñas sintéticas. Reporta accuracy.

7. **Comparación de modelos pre-entrenados (imagen)**: Para un dataset de 15 categorías de productos con 500 imágenes cada uno, compara: ResNet50, EfficientNetB0, MobileNetV2, DenseNet121. Para cada uno: (a) accuracy final, (b) número de parámetros, (c) tiempo de inferencia por imagen. ¿Cuál es el mejor balance accuracy/eficiencia?

8. **Integrador: pipeline completo de clasificación de imágenes de productos**: Diseña un sistema completo: (a) carga EfficientNetB4 pre-entrenado, (b) data augmentation (rotation, flip, zoom, brightness), (c) fine-tuning en 2 fases (feature extractor → discriminative LR), (d) early stopping + ReduceLROnPlateau, (e) evaluación con classification_report, (f) guardar modelo para producción. Dataset: 30 categorías, 200 imágenes por categoría. Reporta accuracy, precision, recall, F1-macro y confusión matrix.

---

## Resumen

- **Transfer Learning**: aprovecha modelos pre-entrenados (ImageNet, BERT) para tareas específicas de ventas/productos.
- **Feature extractor**: congelar backbone, solo entrenar clasificador nuevo. Ideal para pocos datos (< 1k).
- **Fine-tuning**: descongelar últimas capas del backbone con lr bajo (1e-5). Para datos moderados (1k-10k).
- **Progressive unfreezing**: descongelar gradualmente de la última a la primera capa. Previene catastrophic forgetting.
- **Discriminative learning rates**: lr más pequeño para capas pre-entrenadas, más grande para capas nuevas.
- **Data augmentation**: esencial con pocos datos. rotation, flip, zoom, brightness, contrast.
- **Modelos recomendados**: EfficientNet (mejor accuracy/params), MobileNetV2 (ligero), ResNet50 (clásico robusto).
- **BERT**: transfer learning para texto (reseñas, descripciones de productos). Congelar embeddings, entrenar clasificador.
- **Estrategia de 2 fases**: (1) feature extractor con lr alto, (2) fine-tuning con lr bajo. Es la práctica más común y efectiva.
- **Transfer learning siempre supera al entrenamiento desde cero** cuando los datos son limitados, que es el escenario típico en ventas/productos.
