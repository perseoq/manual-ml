# Manual Completo de Ciencia de Datos, NLP, Machine Learning y Deep Learning

## Aplicado al Sector Ventas, Compras e Inventarios

---

## Inicio Rápido

- [README — Presentación general](00-base/00-README.md)
- [Instalación del entorno](00-base/00-INSTALACION.md)
- [Convenciones del manual](00-base/00-CONVENCIONES.md)
- [Generador de datos sintéticos](00-base/00-GENERADOR-DATOS.md)
- [Rutas de aprendizaje](00-base/00-NAVIGATION.md)
- [Índice maestro detallado](00-base/00-INDEX.md)

---

## Tabla de Contenidos

### Nivel 00: Base

Archivos de configuración, instalación y referencia general.

| Archivo | Descripción |
|---------|-------------|
| [00-README.md](00-base/00-README.md) | Presentación general, objetivos, estructura, audiencia y datos de ejemplo |
| [00-INSTALACION.md](00-base/00-INSTALACION.md) | Instalación de Python, entorno virtual, librerías por nivel |
| [00-CONVENCIONES.md](00-base/00-CONVENCIONES.md) | Convenciones de código, estilo y estructura de archivos |
| [00-NAVIGATION.md](00-base/00-NAVIGATION.md) | Mapa de rutas de aprendizaje por perfil |
| [00-GENERADOR-DATOS.md](00-base/00-GENERADOR-DATOS.md) | Documentación del generador de datos sintéticos |
| [00-INDEX.md](00-base/00-INDEX.md) | Índice maestro detallado con todos los archivos |

---

### Nivel 01: Básico

Fundamentos: Python, NumPy, Pandas, Seaborn, SciPy, Scikit-learn y casos prácticos.

#### 00-Python

| Archivo | Tema |
|---------|------|
| [B01-python-tipos-variables.md](01-basico/00-python/B01-python-tipos-variables.md) | Tipos de datos, variables, operadores |
| [B02-python-listas-tuplas.md](01-basico/00-python/B02-python-listas-tuplas.md) | Listas, tuplas, conjuntos, diccionarios |
| [B03-python-diccionarios.md](01-basico/00-python/B03-python-diccionarios.md) | Diccionarios y estructuras clave-valor |
| [B04-python-control-flujo.md](01-basico/00-python/B04-python-control-flujo.md) | if/else, for, while, break, continue |
| [B05-python-funciones.md](01-basico/00-python/B05-python-funciones.md) | Funciones, argumentos, lambda, docstrings |
| [B06-python-archivos-json.md](01-basico/00-python/B06-python-archivos-json.md) | Manejo de archivos CSV, JSON, TXT |
| [B07-python-errores-debugging.md](01-basico/00-python/B07-python-errores-debugging.md) | Manejo de errores, excepciones, logging |
| [B08-python-fechas-datetime.md](01-basico/00-python/B08-python-fechas-datetime.md) | Fechas, timedelta, formateo |

#### 01-NumPy

| Archivo | Tema |
|---------|------|
| [B09-numpy-creacion-arrays.md](01-basico/01-numpy/B09-numpy-creacion-arrays.md) | Creación de arrays, dtype, reshape |
| [B10-numpy-indexing-slicing.md](01-basico/01-numpy/B10-numpy-indexing-slicing.md) | Indexación, slicing, boolean masking |
| [B11-numpy-operaciones.md](01-basico/01-numpy/B11-numpy-operaciones.md) | Operaciones vectorizadas, broadcasting, ufuncs |
| [B12-numpy-estadisticas.md](01-basico/01-numpy/B12-numpy-estadisticas.md) | mean, std, sum, percentiles, correlación |
| [B13-numpy-manipulacion.md](01-basico/01-numpy/B13-numpy-manipulacion.md) | reshape, concatenate, split, broadcasting |
| [B14-numpy-aleatorios.md](01-basico/01-numpy/B14-numpy-aleatorios.md) | random, seed, distribuciones |
| [B15-numpy-algebra-lineal.md](01-basico/01-numpy/B15-numpy-algebra-lineal.md) | dot, matmul, inv, det, eig, solve |

#### 02-Pandas

| Archivo | Tema |
|---------|------|
| [B16-pandas-series.md](01-basico/02-pandas/B16-pandas-series.md) | Series, índices, operaciones básicas |
| [B17-pandas-dataframe-creacion.md](01-basico/02-pandas/B17-pandas-dataframe-creacion.md) | DataFrames, atributos, info, describe |
| [B18-pandas-seleccion-filtros.md](01-basico/02-pandas/B18-pandas-seleccion-filtros.md) | loc, iloc, query, filtros booleanos |
| [B19-pandas-nuevas-columnas.md](01-basico/02-pandas/B19-pandas-nuevas-columnas.md) | Creación de columnas, operaciones, map |
| [B20-pandas-groupby.md](01-basico/02-pandas/B20-pandas-groupby.md) | groupby, aggregate, transform, apply |
| [B21-pandas-merge-join.md](01-basico/02-pandas/B21-pandas-merge-join.md) | merge, join, concat, combinación de DataFrames |
| [B22-pandas-pivot-reshape.md](01-basico/02-pandas/B22-pandas-pivot-reshape.md) | pivot, melt, stack, unstack, crosstab |
| [B23-pandas-fechas-series-temporales.md](01-basico/02-pandas/B23-pandas-fechas-series-temporales.md) | to_datetime, resample, rolling, shift |

#### 03-Seaborn

| Archivo | Tema |
|---------|------|
| [B24-seaborn-intro.md](01-basico/03-seaborn/B24-seaborn-intro.md) | Introducción, temas, paletas, figuras básicas |
| [B25-seaborn-distribuciones.md](01-basico/03-seaborn/B25-seaborn-distribuciones.md) | histplot, kdeplot, boxplot, violinplot |
| [B26-seaborn-categoricos.md](01-basico/03-seaborn/B26-seaborn-categoricos.md) | barplot, countplot, catplot, pointplot |
| [B27-seaborn-relacionales-matriciales.md](01-basico/03-seaborn/B27-seaborn-relacionales-matriciales.md) | scatterplot, lineplot, heatmap, pairplot |

#### 04-SciPy

| Archivo | Tema |
|---------|------|
| [B28-scipy-estadistica.md](01-basico/04-scipy/B28-scipy-estadistica.md) | Estadística descriptiva, distribuciones, tests |
| [B29-scipy-tests-hipotesis.md](01-basico/04-scipy/B29-scipy-tests-hipotesis.md) | Pruebas de hipótesis, t-test, chi-cuadrado |

#### 05-Scikit-learn

| Archivo | Tema |
|---------|------|
| [B30-sklearn-intro.md](01-basico/05-sklearn/B30-sklearn-intro.md) | API, train/test split, métricas básicas |
| [B31-sklearn-preprocesamiento.md](01-basico/05-sklearn/B31-sklearn-preprocesamiento.md) | Escalado, encoding, imputación, pipelines |

#### 06-Casos Prácticos

| Archivo | Tema |
|---------|------|
| [CP01-analisis-ventas-basico.md](01-basico/06-casos/CP01-analisis-ventas-basico.md) | Análisis exploratorio de ventas |
| [CP02-analisis-inventario-basico.md](01-basico/06-casos/CP02-analisis-inventario-basico.md) | Análisis de inventario y rotación |
| [CP03-analisis-compras-basico.md](01-basico/06-casos/CP03-analisis-compras-basico.md) | Análisis de compras y proveedores |
| [CP04-deteccion-outliers.md](01-basico/06-casos/CP04-deteccion-outliers.md) | Detección de valores atípicos |
| [CP05-segmentacion-precios.md](01-basico/06-casos/CP05-segmentacion-precios.md) | Segmentación de precios |
| [CP06-analisis-estacionalidad.md](01-basico/06-casos/CP06-analisis-estacionalidad.md) | Análisis de estacionalidad en ventas |
| [CP07-productos-criticos.md](01-basico/06-casos/CP07-productos-criticos.md) | Identificación de productos críticos |
| [CP08-analisis-clientes-basico.md](01-basico/06-casos/CP08-analisis-clientes-basico.md) | Análisis básico de clientes |
| [CP09-comparativa-sucursales.md](01-basico/06-casos/CP09-comparativa-sucursales.md) | Comparativa de sucursales |
| [CP10-ejercicios-basico.md](01-basico/06-casos/CP10-ejercicios-basico.md) | Ejercicios de nivel básico |

---

### Nivel 02: Intermedio

ML clásico, clustering, NLP introductorio, análisis avanzado y casos prácticos.

#### 00-NumPy Avanzado

| Archivo | Tema |
|---------|------|
| [I01-numpy-broadcasting-detallado.md](02-intermedio/00-numpy-av/I01-numpy-broadcasting-detallado.md) | Broadcasting avanzado, reglas, eficiencia |
| [I02-numpy-funciones-maestras.md](02-intermedio/00-numpy-av/I02-numpy-funciones-maestras.md) | Funciones maestras, stride tricks, memoria |

#### 01-Pandas Avanzado

| Archivo | Tema |
|---------|------|
| [I03-pandas-groupby-avanzado.md](02-intermedio/01-pandas-av/I03-pandas-groupby-avanzado.md) | Groupby avanzado, MultiIndex |
| [I04-pandas-merge-avanzado.md](02-intermedio/01-pandas-av/I04-pandas-merge-avanzado.md) | Merge y join avanzados |
| [I05-pandas-pivot-avanzado.md](02-intermedio/01-pandas-av/I05-pandas-pivot-avanzado.md) | Pivot tables, crosstab avanzado |
| [I06-pandas-ventanas-temporales.md](02-intermedio/01-pandas-av/I06-pandas-ventanas-temporales.md) | rolling, expanding, ewm, ventanas de tiempo |
| [I07-pandas-fechas-avanzado.md](02-intermedio/01-pandas-av/I07-pandas-fechas-avanzado.md) | Fechas avanzadas, periodos, zonas horarias |
| [I08-pandas-funciones-avanzadas.md](02-intermedio/01-pandas-av/I08-pandas-funciones-avanzadas.md) | eval, query, categorías, optimización |

#### 02-Seaborn Avanzado

| Archivo | Tema |
|---------|------|
| [I09-seaborn-personalizacion.md](02-intermedio/02-seaborn-av/I09-seaborn-personalizacion.md) | rcParams, paletas custom, leyendas |
| [I10-seaborn-facet-multipanel.md](02-intermedio/02-seaborn-av/I10-seaborn-facet-multipanel.md) | FacetGrid, PairGrid, subplots |
| [I11-seaborn-graficos-avanzados.md](02-intermedio/02-seaborn-av/I11-seaborn-graficos-avanzados.md) | heatmap, clustermap, jointplot |

#### 03-SciPy Avanzado

| Archivo | Tema |
|---------|------|
| [I12-scipy-optimizacion.md](02-intermedio/03-scipy-av/I12-scipy-optimizacion.md) | minimize, curve_fit, programación lineal |
| [I13-scipy-interpolacion.md](02-intermedio/03-scipy-av/I13-scipy-interpolacion.md) | interp1d, splines, relleno de valores |
| [I14-scipy-procesamiento-senales.md](02-intermedio/03-scipy-av/I14-scipy-procesamiento-senales.md) | Filtros, convolución, FFT, tendencias |

#### 04-ML Clásico

| Archivo | Tema |
|---------|------|
| [I15-sklearn-regresion-lineal.md](02-intermedio/04-ml-clasico/I15-sklearn-regresion-lineal.md) | Ridge, Lasso, ElasticNet, regularización |
| [I16-sklearn-regresion-logistica.md](02-intermedio/04-ml-clasico/I16-sklearn-regresion-logistica.md) | Regresión logística, clasificación binaria |
| [I17-sklearn-arboles-decision.md](02-intermedio/04-ml-clasico/I17-sklearn-arboles-decision.md) | Árboles de decisión, poda, feature importance |
| [I18-sklearn-ensemble-basico.md](02-intermedio/04-ml-clasico/I18-sklearn-ensemble-basico.md) | Random Forest, Bagging, Boosting básico |
| [I19-sklearn-svm.md](02-intermedio/04-ml-clasico/I19-sklearn-svm.md) | SVM, kernels, márgenes |
| [I20-sklearn-knn-vecinos.md](02-intermedio/04-ml-clasico/I20-sklearn-knn-vecinos.md) | KNN, distancia, K óptimo |

#### 05-Clustering

| Archivo | Tema |
|---------|------|
| [I21-sklearn-kmeans-clustering.md](02-intermedio/05-clustering/I21-sklearn-kmeans-clustering.md) | K-Means, codo, silhouette |
| [I22-sklearn-pca-reduccion.md](02-intermedio/05-clustering/I22-sklearn-pca-reduccion.md) | PCA, reducción de dimensionalidad |
| [I23-sklearn-clustering-avanzado.md](02-intermedio/05-clustering/I23-sklearn-clustering-avanzado.md) | DBSCAN, jerárquico, métricas |

#### 06-NLP

| Archivo | Tema |
|---------|------|
| [I24-nlp-nltk-fundamentos.md](02-intermedio/06-nlp/I24-nlp-nltk-fundamentos.md) | NLTK, tokenización, stemming, stopwords |
| [I25-nlp-spacy.md](02-intermedio/06-nlp/I25-nlp-spacy.md) | spaCy, POS tagging, NER, pipelines |
| [I26-nlp-textblob-vader.md](02-intermedio/06-nlp/I26-nlp-textblob-vader.md) | TextBlob, VADER, análisis de sentimiento |
| [I27-nlp-clasificacion-texto.md](02-intermedio/06-nlp/I27-nlp-clasificacion-texto.md) | Clasificación de texto con ML |

#### 07-Casos Prácticos

| Archivo | Tema |
|---------|------|
| [CP11-segmentacion-rfm-completo.md](02-intermedio/07-casos/CP11-segmentacion-rfm-completo.md) | Segmentación RFM de clientes |
| [CP12-prediccion-ventas-ml.md](02-intermedio/07-casos/CP12-prediccion-ventas-ml.md) | Predicción de ventas con ML |
| [CP13-clasificacion-proveedores.md](02-intermedio/07-casos/CP13-clasificacion-proveedores.md) | Clasificación de proveedores |
| [CP14-analisis-cesta-compra.md](02-intermedio/07-casos/CP14-analisis-cesta-compra.md) | Análisis de cesta de compra |
| [CP15-optimizacion-precios.md](02-intermedio/07-casos/CP15-optimizacion-precios.md) | Optimización de precios |
| [CP16-pronostico-inventario.md](02-intermedio/07-casos/CP16-pronostico-inventario.md) | Pronóstico de inventario |
| [CP17-analisis-sentimiento-resenas.md](02-intermedio/07-casos/CP17-analisis-sentimiento-resenas.md) | Análisis de sentimiento en reseñas |
| [CP18-deteccion-anomalias-ventas.md](02-intermedio/07-casos/CP18-deteccion-anomalias-ventas.md) | Detección de anomalías en ventas |
| [CP19-analisis-cancelaciones.md](02-intermedio/07-casos/CP19-analisis-cancelaciones.md) | Análisis de cancelaciones |
| [CP20-ejercicios-intermedio.md](02-intermedio/07-casos/CP20-ejercicios-intermedio.md) | Ejercicios de nivel intermedio |

---

### Nivel 03: Avanzado

ML avanzado, feature engineering, NLP avanzado, Deep Learning (TF + PyTorch) y casos complejos.

#### 00-Scikit-learn Avanzado

| Archivo | Tema |
|---------|------|
| [A01-sklearn-pipelines.md](03-avanzado/00-sklearn-av/A01-sklearn-pipelines.md) | Pipelines, ColumnTransformer, FeatureUnion |
| [A02-sklearn-grid-search-optimizacion.md](03-avanzado/00-sklearn-av/A02-sklearn-grid-search-optimizacion.md) | GridSearchCV, RandomizedSearchCV |
| [A03-sklearn-validacion-curvas.md](03-avanzado/00-sklearn-av/A03-sklearn-validacion-curvas.md) | learning_curve, validation_curve |
| [A04-sklearn-ensemble-avanzado.md](03-avanzado/00-sklearn-av/A04-sklearn-ensemble-avanzado.md) | Voting, Stacking, Bagging avanzado |
| [A05-sklearn-modelos-lineales-avanzados.md](03-avanzado/00-sklearn-av/A05-sklearn-modelos-lineales-avanzados.md) | SGD, Huber, modelos lineales robustos |
| [A06-sklearn-metricas-evaluacion.md](03-avanzado/00-sklearn-av/A06-sklearn-metricas-evaluacion.md) | Métricas avanzadas, scoring personalizado |

#### 01-Feature Engineering

| Archivo | Tema |
|---------|------|
| [A07-feature-engineering-creacion.md](03-avanzado/01-feature-eng/A07-feature-engineering-creacion.md) | Creación de features, polynomial, splines |
| [A08-feature-selection.md](03-avanzado/01-feature-eng/A08-feature-selection.md) | SelectKBest, RFE, RFECV, SelectFromModel |
| [A09-feature-encoding-avanzado.md](03-avanzado/01-feature-eng/A09-feature-encoding-avanzado.md) | TargetEncoder, FrequencyEncoder, categorías |
| [A10-feature-scaling-transformacion.md](03-avanzado/01-feature-eng/A10-feature-scaling-transformacion.md) | StandardScaler, RobustScaler, PowerTransformer |

#### 02-NLP Avanzado

| Archivo | Tema |
|---------|------|
| [A11-nlp-tfidf-vectores-detallado.md](03-avanzado/02-nlp-av/A11-nlp-tfidf-vectores-detallado.md) | TF-IDF, n-gramas, HashingVectorizer |
| [A12-nlp-word2vec-embeddings.md](03-avanzado/02-nlp-av/A12-nlp-word2vec-embeddings.md) | Word2Vec, CBOW, Skip-gram |
| [A13-nlp-glove-fasttext-embeddings.md](03-avanzado/02-nlp-av/A13-nlp-glove-fasttext-embeddings.md) | GloVe, FastText, embeddings pre-entrenados |
| [A14-nlp-modelos-secuenciales.md](03-avanzado/02-nlp-av/A14-nlp-modelos-secuenciales.md) | LSTM, GRU, atención para NLP |

#### 03-Deep Learning

| Archivo | Tema |
|---------|------|
| [A15-tensorflow-intro.md](03-avanzado/03-dl/A15-tensorflow-intro.md) | TensorFlow, tensores, Keras, Sequential API |
| [A16-tensorflow-sequential.md](03-avanzado/03-dl/A16-tensorflow-sequential.md) | Modelo secuencial, capas, compile, fit |
| [A17-tensorflow-funcional-api.md](03-avanzado/03-dl/A17-tensorflow-funcional-api.md) | API funcional, modelos multi-input/output |
| [A18-tensorflow-callbacks-detallado.md](03-avanzado/03-dl/A18-tensorflow-callbacks-detallado.md) | Callbacks, early stopping, model checkpoint |
| [A19-tensorflow-datasets-pipeline.md](03-avanzado/03-dl/A19-tensorflow-datasets-pipeline.md) | tf.data, pipeline de datos, prefetch |
| [A20-pytorch-intro.md](03-avanzado/03-dl/A20-pytorch-intro.md) | PyTorch, tensores, autograd, GPU |
| [A21-pytorch-autograd.md](03-avanzado/03-dl/A21-pytorch-autograd.md) | Autograd, gradientes, backward |
| [A22-pytorch-nn-module.md](03-avanzado/03-dl/A22-pytorch-nn-module.md) | nn.Module, capas personalizadas, secuencial |
| [A23-pytorch-dataloader.md](03-avanzado/03-dl/A23-pytorch-dataloader.md) | DataLoader, Dataset, transforms, batching |
| [A24-pytorch-training-loop.md](03-avanzado/03-dl/A24-pytorch-training-loop.md) | Training loop, optimizadores, scheduler |

#### 04-Arquitecturas

| Archivo | Tema |
|---------|------|
| [A25-dl-redes-neuronales.md](03-avanzado/04-arquitecturas/A25-dl-redes-neuronales.md) | Redes densas, activaciones, dropout, batch norm |
| [A26-dl-cnn-ventas.md](03-avanzado/04-arquitecturas/A26-dl-cnn-ventas.md) | CNNs, Conv2D, MaxPooling, arquitecturas |
| [A27-dl-rnn-lstm.md](03-avanzado/04-arquitecturas/A27-dl-rnn-lstm.md) | RNN, LSTM, GRU, Bidirectional |
| [A28-dl-autoencoders-inventario.md](03-avanzado/04-arquitecturas/A28-dl-autoencoders-inventario.md) | Autoencoders, VAE, detección de anomalías |
| [A29-dl-regularizacion-optimizacion.md](03-avanzado/04-arquitecturas/A29-dl-regularizacion-optimizacion.md) | Regularización, optimización, hiperparámetros |
| [A30-dl-transfer-learning-finetuning.md](03-avanzado/04-arquitecturas/A30-dl-transfer-learning-finetuning.md) | Transfer Learning, fine-tuning, feature extraction |

#### 05-Casos Prácticos

| Archivo | Tema |
|---------|------|
| [CP21-prediccion-demanda-tf.md](03-avanzado/05-casos/CP21-prediccion-demanda-tf.md) | Predicción de demanda con TensorFlow |
| [CP22-clasificacion-texto-productos.md](03-avanzado/05-casos/CP22-clasificacion-texto-productos.md) | Clasificación de texto de productos |
| [CP23-recomendacion-content-based.md](03-avanzado/05-casos/CP23-recomendacion-content-based.md) | Sistema de recomendación content-based |
| [CP24-deteccion-anomalias-avanzado.md](03-avanzado/05-casos/CP24-deteccion-anomalias-avanzado.md) | Detección de anomalías avanzada |
| [CP25-segmentacion-dinamica-precios.md](03-avanzado/05-casos/CP25-segmentacion-dinamica-precios.md) | Segmentación dinámica de precios |
| [CP26-prediccion-abandono-clientes.md](03-avanzado/05-casos/CP26-prediccion-abandono-clientes.md) | Predicción de abandono de clientes |
| [CP27-forecast-inventario-ml.md](03-avanzado/05-casos/CP27-forecast-inventario-ml.md) | Forecast de inventario con ML |
| [CP28-ejercicios-avanzado.md](03-avanzado/05-casos/CP28-ejercicios-avanzado.md) | Ejercicios de nivel avanzado |

---

### Nivel 04: Experto

Transformers, series temporales, sistemas de recomendación, MLOps y casos complejos.

#### 00-Transformers

| Archivo | Tema |
|---------|------|
| [E01-transformers-fundamentos.md](04-experto/00-transformers/E01-transformers-fundamentos.md) | Fundamentos de transformers, atención |
| [E02-huggingface-pipeline.md](04-experto/00-transformers/E02-huggingface-pipeline.md) | HuggingFace pipeline, AutoModel, tokenizers |
| [E03-huggingface-bert-finetuning.md](04-experto/00-transformers/E03-huggingface-bert-finetuning.md) | BERT fine-tuning para clasificación |
| [E04-huggingface-gpt-generacion.md](04-experto/00-transformers/E04-huggingface-gpt-generacion.md) | GPT, generación de texto, prompting |
| [E05-huggingface-semantic-search.md](04-experto/00-transformers/E05-huggingface-semantic-search.md) | Búsqueda semántica con transformers |
| [E06-huggingface-rag-chatbot.md](04-experto/00-transformers/E06-huggingface-rag-chatbot.md) | RAG, LangChain, chatbots con LLMs |

#### 01-Series Temporales

| Archivo | Tema |
|---------|------|
| [E07-ts-descomposicion-detallada.md](04-experto/01-series-temporales/E07-ts-descomposicion-detallada.md) | Descomposición STL, tendencias, estacionalidad |
| [E08-ts-arima-sarima-detallado.md](04-experto/01-series-temporales/E08-ts-arima-sarima-detallado.md) | ARIMA, SARIMA, ACF/PACF, auto_arima |
| [E09-ts-prophet-detallado.md](04-experto/01-series-temporales/E09-ts-prophet-detallado.md) | Prophet, changepoints, festividades |
| [E10-ts-lstm-detallado.md](04-experto/01-series-temporales/E10-ts-lstm-detallado.md) | LSTM para forecasting, seq2seq, attention |
| [E11-ts-modelos-estado-espacio.md](04-experto/01-series-temporales/E11-ts-modelos-estado-espacio.md) | Modelos de estado de espacio, Kalman filter |

#### 02-Sistemas de Recomendación

| Archivo | Tema |
|---------|------|
| [E12-recsys-filtro-colaborativo.md](04-experto/02-recsys/E12-recsys-filtro-colaborativo.md) | Filtro colaborativo user/item-based |
| [E13-recsys-matrix-factorization.md](04-experto/02-recsys/E13-recsys-matrix-factorization.md) | SVD, NMF, FunkSVD, factorización |
| [E14-recsys-content-based.md](04-experto/02-recsys/E14-recsys-content-based.md) | Filtrado basado en contenido, perfiles |
| [E15-recsys-deep-learning.md](04-experto/02-recsys/E15-recsys-deep-learning.md) | Deep Learning para recomendación |
| [E16-recsys-hibrido-evaluacion.md](04-experto/02-recsys/E16-recsys-hibrido-evaluacion.md) | Sistemas híbridos, FAISS, evaluación |

#### 03-Optimización y Producción

| Archivo | Tema |
|---------|------|
| [E17-optimizacion-hyperopt.md](04-experto/03-optim-prod/E17-optimizacion-hyperopt.md) | Hyperopt, optimización de hiperparámetros |
| [E18-optimizacion-optuna.md](04-experto/03-optim-prod/E18-optimizacion-optuna.md) | Optuna, estudios, trials, pruning |
| [E19-optimizacion-modelos.md](04-experto/03-optim-prod/E19-optimizacion-modelos.md) | Optimización de modelos, cuantización |
| [E20-mlops-felicitaciones.md](04-experto/03-optim-prod/E20-mlops-felicitaciones.md) | MLOps, CI/CD, experiment tracking |
| [E21-despliegue-fastapi-detallado.md](04-experto/03-optim-prod/E21-despliegue-fastapi-detallado.md) | FastAPI, endpoints, validación, deployment |
| [E22-monitoreo-modelos.md](04-experto/03-optim-prod/E22-monitoreo-modelos.md) | Monitoreo, data drift, alertas |
| [E23-explicabilidad-modelos.md](04-experto/03-optim-prod/E23-explicabilidad-modelos.md) | SHAP, LIME, interpretabilidad |
| [E24-mlops-produccion-completa.md](04-experto/03-optim-prod/E24-mlops-produccion-completa.md) | Pipeline MLOps completo en producción |

#### 04-Casos Prácticos

| Archivo | Tema |
|---------|------|
| [CP29-inventario-predictivo-pipeline.md](04-experto/04-casos/CP29-inventario-predictivo-pipeline.md) | Pipeline de inventario predictivo |
| [CP30-sistema-recomendacion-b2b.md](04-experto/04-casos/CP30-sistema-recomendacion-b2b.md) | Sistema de recomendación B2B |
| [CP31-chatbot-ventas-rag.md](04-experto/04-casos/CP31-chatbot-ventas-rag.md) | Chatbot de ventas con RAG |
| [CP32-deteccion-fraude-tiempo-real.md](04-experto/04-casos/CP32-deteccion-fraude-tiempo-real.md) | Detección de fraude en tiempo real |
| [CP33-optimizacion-precios-dinamica.md](04-experto/04-casos/CP33-optimizacion-precios-dinamica.md) | Optimización dinámica de precios |
| [CP34-ejercicios-experto.md](04-experto/04-casos/CP34-ejercicios-experto.md) | Ejercicios de nivel experto |

---

### Nivel 05: Apéndices

Cheatsheets, glosario y recursos de referencia rápida.

| Archivo | Descripción |
|---------|-------------|
| [AP01-cheatsheet-python.md](05-apendices/AP01-cheatsheet-python.md) | Cheatsheet de Python |
| [AP02-cheatsheet-numpy.md](05-apendices/AP02-cheatsheet-numpy.md) | Cheatsheet de NumPy |
| [AP03-cheatsheet-pandas.md](05-apendices/AP03-cheatsheet-pandas.md) | Cheatsheet de Pandas |
| [AP04-cheatsheet-seaborn.md](05-apendices/AP04-cheatsheet-seaborn.md) | Cheatsheet de Seaborn |
| [AP05-cheatsheet-scikit-learn.md](05-apendices/AP05-cheatsheet-scikit-learn.md) | Cheatsheet de Scikit-learn |
| [AP06-cheatsheet-tensorflow.md](05-apendices/AP06-cheatsheet-tensorflow.md) | Cheatsheet de TensorFlow/Keras |
| [AP07-cheatsheet-pytorch.md](05-apendices/AP07-cheatsheet-pytorch.md) | Cheatsheet de PyTorch |
| [AP08-cheatsheet-nlp.md](05-apendices/AP08-cheatsheet-nlp.md) | Cheatsheet de NLP |
| [AP09-cheatsheet-series-temporales.md](05-apendices/AP09-cheatsheet-series-temporales.md) | Cheatsheet de Series Temporales |
| [AP10-cheatsheet-despliegue.md](05-apendices/AP10-cheatsheet-despliegue.md) | Cheatsheet de Despliegue y MLOps |
| [AP11-glosario.md](05-apendices/AP11-glosario.md) | Glosario de términos técnicos |
| [AP12-recursos-recomendados.md](05-apendices/AP12-recursos-recomendados.md) | Recursos, libros, cursos y más |

---

### Datos

| Archivo | Descripción |
|---------|-------------|
| [datos_sinteticos.py](datos/datos_sinteticos.py) | Módulo generador de datos sintéticos (5 datasets) |
| [ventas.csv](datos/ventas.csv) | Transacciones de venta |
| [inventario.csv](datos/inventario.csv) | Estado del inventario |
| [compras.csv](datos/compras.csv) | Órdenes de compra |
| [clientes.csv](datos/clientes.csv) | Perfil RFM de clientes |
| [resenas.csv](datos/resenas.csv) | Reseñas de productos |

---

## Estadísticas

| Nivel | Archivos | Módulos |
|-------|----------|---------|
| 00-base | 6 | Configuración y referencia |
| 01-básico | 41 | Python, NumPy, Pandas, Seaborn, SciPy, sklearn |
| 02-intermedio | 39 | NumPy/Pandas/Seaborn/SciPy av, ML, clustering, NLP |
| 03-avanzado | 38 | sklearn av, feature eng, NLP av, DL (TF + PyTorch) |
| 04-experto | 30 | Transformers, series temp, recsys, MLOps |
| 05-apéndices | 12 | Cheatsheets, glosario, recursos |
| **Total** | **~166** | **~30 librerías** |

---

*Volver al [README principal](00-base/00-README.md) — [Índice maestro detallado](00-base/00-INDEX.md)*
