# Índice Maestro del Manual

## Manual Completo de Ciencia de Datos, NLP, Machine Learning y Deep Learning — Ventas, Compras e Inventarios

---

**Total de archivos:** 179 (incluyendo 00-base, básico, intermedio, avanzado, experto y apéndices)

---

## Tabla de Contenidos

- [Nivel 00: Base](#nivel-00-base)
- [Nivel 01: Básico](#nivel-01-básico)
  - [00-python](#01-básico-00-python-fundamentos-de-python)
  - [01-numpy](#01-básico-01-numpy)
  - [02-pandas](#01-básico-02-pandas)
  - [03-seaborn](#01-básico-03-seaborn)
  - [04-scipy](#01-básico-04-scipy)
  - [05-sklearn](#01-básico-05-sklearn)
  - [06-casos](#01-básico-06-casos-prácticos)
- [Nivel 02: Intermedio](#nivel-02-intermedio)
  - [00-numpy-av](#02-intermedio-00-numpy-avanzado)
  - [01-pandas-av](#02-intermedio-01-pandas-avanzado)
  - [02-seaborn-av](#02-intermedio-02-seaborn-avanzado)
  - [03-scipy-av](#02-intermedio-03-scipy-avanzado)
  - [04-ml-clasico](#02-intermedio-04-ml-clásico)
  - [05-clustering](#02-intermedio-05-clustering)
  - [06-nlp](#02-intermedio-06-nlp)
  - [07-casos](#02-intermedio-07-casos-prácticos)
- [Nivel 03: Avanzado](#nivel-03-avanzado)
  - [00-sklearn-av](#03-avanzado-00-sklearn-avanzado)
  - [01-feature-eng](#03-avanzado-01-feature-engineering)
  - [02-nlp-av](#03-avanzado-02-nlp-avanzado)
  - [03-dl](#03-avanzado-03-deep-learning)
  - [04-arquitecturas](#03-avanzado-04-arquitecturas)
  - [05-casos](#03-avanzado-05-casos-prácticos)
- [Nivel 04: Experto](#nivel-04-experto)
  - [00-transformers](#04-experto-00-transformers)
  - [01-series-temporales](#04-experto-01-series-temporales)
  - [02-recsys](#04-experto-02-sistemas-de-recomendación)
  - [03-optim-prod](#04-experto-03-optimización-y-producción)
  - [04-casos](#04-experto-04-casos-prácticos)
- [Nivel 05: Apéndices](#nivel-05-apéndices)
- [Estadísticas del Manual](#estadísticas-del-manual)

---

## Nivel 00: Base

Archivos de configuración, instalación, navegación y referencia general del manual.

| # | Nivel | Sección | Archivo | Descripción | Librerías |
|---|-------|---------|---------|-------------|-----------|
| 1 | 00 | Base | [00-README.md](00-README.md) | Presentación general del manual, objetivos, estructura, audiencia y datos de ejemplo | — |
| 2 | 00 | Base | [00-INSTALACION.md](00-INSTALACION.md) | Instalación de Python, entorno virtual, librerías por nivel y verificación | pip, venv |
| 3 | 00 | Base | [00-CONVENCIONES.md](00-CONVENCIONES.md) | Convenciones de código, estilo, estructura de archivos y rutas relativas | — |
| 4 | 00 | Base | [00-NAVIGATION.md](00-NAVIGATION.md) | Mapa de rutas de aprendizaje: 3 perfiles con archivos ordenados y tiempos | — |
| 5 | 00 | Base | [00-GENERADOR-DATOS.md](00-GENERADOR-DATOS.md) | Documentación del generador de datos sintéticos: funciones, columnas, ejemplos | pandas, numpy |
| 6 | 00 | Base | [00-INDEX.md](00-INDEX.md) | Índice maestro con todos los archivos del manual organizados por nivel | — |

---

## Nivel 01: Básico

Fundamentos de ciencia de datos: Python, NumPy, Pandas, Seaborn, SciPy, Scikit-learn y casos prácticos iniciales aplicados a ventas, compras e inventarios.

### 01-Básico: 00-Python (Fundamentos de Python)

| # | Nivel | Sección | Archivo | Descripción | Librerías |
|---|-------|---------|---------|-------------|-----------|
| 7 | 01 | Básico-Python | [B01-01-Introduccion-Python.md](../01-basico/00-python/B01-01-Introduccion-Python.md) | Introducción a Python: print(), variables, tipos de datos básicos, comentarios, operadores aritméticos | builtins |
| 8 | 01 | Básico-Python | [B01-02-Tipos-Datos.md](../01-basico/00-python/B01-02-Tipos-Datos.md) | Tipos de datos: int, float, str, bool, None; conversiones entre tipos, formateo de strings | builtins |
| 9 | 01 | Básico-Python | [B01-03-Estructuras-Control.md](../01-basico/00-python/B01-03-Estructuras-Control.md) | Estructuras de control: if/elif/else, for, while, break, continue, pass anidados | builtins |
| 10 | 01 | Básico-Python | [B01-04-Funciones.md](../01-basico/00-python/B01-04-Funciones.md) | Funciones: def, return, parámetros, args/kwargs, scope, lambda, docstrings | builtins |
| 11 | 01 | Básico-Python | [B01-05-Manejo-Archivos.md](../01-basico/00-python/B01-05-Manejo-Archivos.md) | Manejo de archivos: open(), read/write CSV, JSON, txt; with statement, encoding | csv, json |
| 12 | 01 | Básico-Python | [B01-06-Listas-Comprension.md](../01-basico/00-python/B01-06-Listas-Comprension.md) | List comprehension, dict/set comprehension, zip, enumerate, map/filter/reduce | builtins |
| 13 | 01 | Básico-Python | [B01-07-Errores-Excepciones.md](../01-basico/00-python/B01-07-Errores-Excepciones.md) | Manejo de errores: try/except, finally, raise, tipos de excepción, assert, logging | logging |

### 01-Básico: 01-NumPy

| # | Nivel | Sección | Archivo | Descripción | Librerías |
|---|-------|---------|---------|-------------|-----------|
| 14 | 01 | Básico-NumPy | [B02-01-Introduccion-NumPy.md](../01-basico/01-numpy/B02-01-Introduccion-NumPy.md) | Introducción a NumPy: ndarray vs listas, creación básica, atributos, dtype | numpy |
| 15 | 01 | Básico-NumPy | [B02-02-Arrays.md](../01-basico/01-numpy/B02-02-Arrays.md) | Creación de arrays: np.array(), zeros, ones, empty, arange, linspace, random, reshape | numpy |
| 16 | 01 | Básico-NumPy | [B02-03-Operaciones.md](../01-basico/01-numpy/B02-03-Operaciones.md) | Operaciones vectorizadas: aritmética, broadcasting, ufuncs, funciones trigonométricas | numpy |
| 17 | 01 | Básico-NumPy | [B02-04-Indexacion.md](../01-basico/01-numpy/B02-04-Indexacion.md) | Indexación y slicing: básico, fancy indexing, boolean masking, where, choose | numpy |
| 18 | 01 | Básico-NumPy | [B02-05-Estadisticas.md](../01-basico/01-numpy/B02-05-Estadisticas.md) | Estadísticas: mean, median, std, var, min, max, sum, cumsum, percentil, corrcoef | numpy |
| 19 | 01 | Básico-NumPy | [B02-06-Algebra-Lineal.md](../01-basico/01-numpy/B02-06-Algebra-Lineal.md) | Álgebra lineal: dot, matmul, transpose, inv, det, eig, solve aplicado a ventas | numpy |
| 20 | 01 | Básico-NumPy | [B02-07-Random.md](../01-basico/01-numpy/B02-07-Random.md) | Números aleatorios: seed, rand, randn, randint, choice, shuffle, distribuciones | numpy |

### 01-Básico: 02-Pandas

| # | Nivel | Sección | Archivo | Descripción | Librerías |
|---|-------|---------|---------|-------------|-----------|
| 21 | 01 | Básico-Pandas | [B03-01-Introduccion-Pandas.md](../01-basico/02-pandas/B03-01-Introduccion-Pandas.md) | Introducción a Pandas: Series, DataFrame, índices, diferencia con NumPy | pandas |
| 22 | 01 | Básico-Pandas | [B03-02-Series-DataFrames.md](../01-basico/02-pandas/B03-02-Series-DataFrames.md) | Series y DataFrames: creación desde dicts/listas/arrays, atributos, info, head, describe | pandas |
| 23 | 01 | Básico-Pandas | [B03-03-Lectura-Datos.md](../01-basico/02-pandas/B03-03-Lectura-Datos.md) | Lectura de datos: read_csv, read_excel, read_json, read_sql, parámetros, encoding | pandas |
| 24 | 01 | Básico-Pandas | [B03-04-Limpieza.md](../01-basico/02-pandas/B03-04-Limpieza.md) | Limpieza de datos: isnull, dropna, fillna, duplicated, drop_duplicates, astype, replace | pandas |
| 25 | 01 | Básico-Pandas | [B03-05-Filtros-Condiciones.md](../01-basico/02-pandas/B03-05-Filtros-Condiciones.md) | Filtros y condiciones: loc, iloc, boolean indexing, query, isin, between, str.contains | pandas |
| 26 | 01 | Básico-Pandas | [B03-06-Agrupaciones.md](../01-basico/02-pandas/B03-06-Agrupaciones.md) | Agrupaciones: groupby, aggregate, transform, filter, apply por grupo, múltiples funciones | pandas |
| 27 | 01 | Básico-Pandas | [B03-07-Join-Merge.md](../01-basico/02-pandas/B03-07-Join-Merge.md) | Combinación de DataFrames: merge, join, concat, tipos de join, indicadores | pandas |
| 28 | 01 | Básico-Pandas | [B03-08-Dates-Times.md](../01-basico/02-pandas/B03-08-Dates-Times.md) | Fechas y tiempos: to_datetime, dt accessor, resample, period, shift, diff | pandas |
| 29 | 01 | Básico-Pandas | [B03-09-Apply-Funciones.md](../01-basico/02-pandas/B03-09-Apply-Funciones.md) | Aplicación de funciones: apply, map, applymap, pipe, funciones vectorizadas | pandas |

### 01-Básico: 03-Seaborn

| # | Nivel | Sección | Archivo | Descripción | Librerías |
|---|-------|---------|---------|-------------|-----------|
| 30 | 01 | Básico-Seaborn | [B04-01-Introduccion-Seaborn.md](../01-basico/03-seaborn/B04-01-Introduccion-Seaborn.md) | Introducción a Seaborn: set_theme, paletas, estilos, figura básica, integración con Pandas | seaborn, matplotlib |
| 31 | 01 | Básico-Seaborn | [B04-02-Graficos-Distribucion.md](../01-basico/03-seaborn/B04-02-Graficos-Distribucion.md) | Gráficos de distribución: histplot, kdeplot, boxplot, violinplot, displot | seaborn, matplotlib |
| 32 | 01 | Básico-Seaborn | [B04-03-Graficos-Categoricos.md](../01-basico/03-seaborn/B04-03-Graficos-Categoricos.md) | Gráficos categóricos: barplot, countplot, catplot, pointplot, stripplot | seaborn, matplotlib |
| 33 | 01 | Básico-Seaborn | [B04-04-Graficos-Relacionales.md](../01-basico/03-seaborn/B04-04-Graficos-Relacionales.md) | Gráficos relacionales: scatterplot, lineplot, relplot, regplot, lmplot | seaborn, matplotlib |
| 34 | 01 | Básico-Seaborn | [B04-05-Estilizacion-Temas.md](../01-basico/03-seaborn/B04-05-Estilizacion-Temas.md) | Estilización: temas, paletas de colores, personalización de ejes, anotaciones, leyendas | seaborn, matplotlib |

### 01-Básico: 04-SciPy

| # | Nivel | Sección | Archivo | Descripción | Librerías |
|---|-------|---------|---------|-------------|-----------|
| 35 | 01 | Básico-SciPy | [B05-01-Introduccion-SciPy.md](../01-basico/04-scipy/B05-01-Introduccion-SciPy.md) | Introducción a SciPy: submódulos, diferencias con NumPy, constantes, integración | scipy |
| 36 | 01 | Básico-SciPy | [B05-02-Estadistica.md](../01-basico/04-scipy/B05-02-Estadistica.md) | Estadística con SciPy: distribuciones de probabilidad, pdf, cdf, ppf, stats descriptivos | scipy.stats |
| 37 | 01 | Básico-SciPy | [B05-03-Optimizacion.md](../01-basico/04-scipy/B05-03-Optimizacion.md) | Optimización: minimize, curve_fit, root_scalar, programación lineal aplicada a costos | scipy.optimize |
| 38 | 01 | Básico-SciPy | [B05-04-Interpolacion.md](../01-basico/04-scipy/B05-04-Interpolacion.md) | Interpolación: interp1d, splines, relleno de valores faltantes en series temporales | scipy.interpolate |

### 01-Básico: 05-Scikit-learn

| # | Nivel | Sección | Archivo | Descripción | Librerías |
|---|-------|---------|---------|-------------|-----------|
| 39 | 01 | Básico-Sklearn | [B06-01-Introduccion-Sklearn.md](../01-basico/05-sklearn/B06-01-Introduccion-Sklearn.md) | Introducción a Scikit-learn: API unificada, train_test_split, flujo de trabajo, métricas básicas | sklearn |
| 40 | 01 | Básico-Sklearn | [B06-02-Regresion-Lineal.md](../01-basico/05-sklearn/B06-02-Regresion-Lineal.md) | Regresión lineal: LinearRegression, intercepto, coeficientes, R², MSE, MAE, predicción de ventas | sklearn.linear_model |
| 41 | 01 | Básico-Sklearn | [B06-03-Clasificacion.md](../01-basico/05-sklearn/B06-03-Clasificacion.md) | Clasificación: LogisticRegression, KNeighborsClassifier, DecisionTreeClassifier aplicado a productos | sklearn |
| 42 | 01 | Básico-Sklearn | [B06-04-Metricas.md](../01-basico/05-sklearn/B06-04-Metricas.md) | Métricas de evaluación: accuracy, precision, recall, F1, ROC-AUC, matriz de confusión, classification_report | sklearn.metrics |
| 43 | 01 | Básico-Sklearn | [B06-05-Validacion-Cruzada.md](../01-basico/05-sklearn/B06-05-Validacion-Cruzada.md) | Validación cruzada: cross_val_score, KFold, StratifiedKFold, learning_curve, validation_curve | sklearn.model_selection |

### 01-Básico: 06-Casos Prácticos

| # | Nivel | Sección | Archivo | Descripción | Librerías |
|---|-------|---------|---------|-------------|-----------|
| 44 | 01 | Básico-Casos | [B07-01-Analisis-Ventas.md](../01-basico/06-casos/B07-01-Analisis-Ventas.md) | Análisis exploratorio completo de ventas: tendencias, estacionalidad, top productos, sucursales | pandas, seaborn, numpy |
| 45 | 01 | Básico-Casos | [B07-02-Analisis-Inventario.md](../01-basico/06-casos/B07-02-Analisis-Inventario.md) | Análisis de inventario: rotación, stock crítico, valor inventariado, reposiciones | pandas, seaborn, numpy |
| 46 | 01 | Básico-Casos | [B07-03-Analisis-Compras.md](../01-basico/06-casos/B07-03-Analisis-Compras.md) | Análisis de compras: órdenes, proveedores, retrasos, costos, efectividad de entregas | pandas, seaborn, numpy |

---

## Nivel 02: Intermedio

Análisis avanzado, ML clásico, clustering, NLP introductorio y casos prácticos integradores con técnicas de mayor complejidad.

### 02-Intermedio: 00-NumPy Avanzado

| # | Nivel | Sección | Archivo | Descripción | Librerías |
|---|-------|---------|---------|-------------|-----------|
| 47 | 02 | Inter-NumPyAv | [I01-01-Broadcasting.md](../02-intermedio/00-numpy-av/I01-01-Broadcasting.md) | Broadcasting avanzado: reglas, dimensiones, eficiencia, aplicaciones en ventas por sucursal | numpy |
| 48 | 02 | Inter-NumPyAv | [I01-02-Mask-Index.md](../02-intermedio/00-numpy-av/I01-02-Mask-Index.md) | Indexación avanzada: masking complejo, fancy indexing, np.where, np.select, np.clip | numpy |
| 49 | 02 | Inter-NumPyAv | [I01-03-Stride-Tricks.md](../02-intermedio/00-numpy-av/I01-03-Stride-Tricks.md) | Stride tricks: vistas vs copias, as_strided, optimización de memoria para ventanas | numpy |
| 50 | 02 | Inter-NumPyAv | [I01-04-Polinomios.md](../02-intermedio/00-numpy-av/I01-04-Polinomios.md) | Polinomios: np.polyfit, np.polyval, np.roots, ajuste polinomial de tendencias de ventas | numpy |

### 02-Intermedio: 01-Pandas Avanzado

| # | Nivel | Sección | Archivo | Descripción | Librerías |
|---|-------|---------|---------|-------------|-----------|
| 51 | 02 | Inter-PandasAv | [I02-01-MultiIndex.md](../02-intermedio/01-pandas-av/I02-01-MultiIndex.md) | MultiIndex: índices jerárquicos, selección por nivel, stack/unstack, xs, sort_index | pandas |
| 52 | 02 | Inter-PandasAv | [I02-02-Transformaciones.md](../02-intermedio/01-pandas-av/I02-02-Transformaciones.md) | Transformaciones: melt, pivot, pivot_table, crosstab, explode, wide_to_long | pandas |
| 53 | 02 | Inter-PandasAv | [I02-03-Ventanas-Temporales.md](../02-intermedio/01-pandas-av/I02-03-Ventanas-Temporales.md) | Ventanas temporales: rolling, expanding, ewm, rolling.apply, ventanas centradas | pandas |
| 54 | 02 | Inter-PandasAv | [I02-04-Pivot-Cross.md](../02-intermedio/01-pandas-av/I02-04-Pivot-Cross.md) | Tablas dinámicas: pivot_table con múltiples agrupaciones, crosstab, margins, fill_value | pandas |
| 55 | 02 | Inter-PandasAv | [I02-05-Performance.md](../02-intermedio/01-pandas-av/I02-05-Performance.md) | Optimización de rendimiento: eval, query, categorías, chunking, memoria, parallel | pandas, numpy |

### 02-Intermedio: 02-Seaborn Avanzado

| # | Nivel | Sección | Archivo | Descripción | Librerías |
|---|-------|---------|---------|-------------|-----------|
| 56 | 02 | Inter-SeabornAv | [I03-01-Multiplot.md](../02-intermedio/02-seaborn-av/I03-01-Multiplot.md) | Múltiples gráficos: FacetGrid, PairGrid, jointplot, subplots con matplotlib | seaborn, matplotlib |
| 57 | 02 | Inter-SeabornAv | [I03-02-Heatmap-Cluster.md](../02-intermedio/02-seaborn-av/I03-02-Heatmap-Cluster.md) | Mapas de calor: heatmap, clustermap, correlaciones, anotaciones, máscaras | seaborn, scipy |
| 58 | 02 | Inter-SeabornAv | [I03-03-Personalizacion.md](../02-intermedio/02-seaborn-av/I03-03-Personalizacion.md) | Personalización avanzada: rcParams, estilos, paletas custom, leyendas, ejes, anotaciones | seaborn, matplotlib |

### 02-Intermedio: 03-SciPy Avanzado

| # | Nivel | Sección | Archivo | Descripción | Librerías |
|---|-------|---------|---------|-------------|-----------|
| 59 | 02 | Inter-SciPyAv | [I04-01-Pruebas-Hipotesis.md](../02-intermedio/03-scipy-av/I04-01-Pruebas-Hipotesis.md) | Pruebas de hipótesis: t-test, chi-cuadrado, Mann-Whitney, KS, normalidad, pruebas de negocio | scipy.stats |
| 60 | 02 | Inter-SciPyAv | [I04-02-ANOVA.md](../02-intermedio/03-scipy-av/I04-02-ANOVA.md) | ANOVA: one-way, two-way, post-hoc (Tukey), comparación de ventas entre sucursales | scipy.stats, statsmodels |
| 61 | 02 | Inter-SciPyAv | [I04-03-Distribuciones.md](../02-intermedio/03-scipy-av/I04-03-Distribuciones.md) | Distribuciones de probabilidad: normal, Poisson, exponencial, gamma, beta, fitting | scipy.stats |
| 62 | 02 | Inter-SciPyAv | [I04-04-Senales.md](../02-intermedio/03-scipy-av/I04-04-Senales.md) | Procesamiento de señales: filtros, convolución, transformada de Fourier, tendencias | scipy.signal |

### 02-Intermedio: 04-ML Clásico

| # | Nivel | Sección | Archivo | Descripción | Librerías |
|---|-------|---------|---------|-------------|-----------|
| 63 | 02 | Inter-ML | [I05-01-Regresion-Av.md](../02-intermedio/04-ml-clasico/I05-01-Regresion-Av.md) | Regresión avanzada: Ridge, Lasso, ElasticNet, PolynomialFeatures, regularización, selección de alpha | sklearn.linear_model |
| 64 | 02 | Inter-ML | [I05-02-Arboles-Decision.md](../02-intermedio/04-ml-clasico/I05-02-Arboles-Decision.md) | Árboles de decisión: DecisionTreeClassifier/Regressor, poda, max_depth, feature_importance, interpretación | sklearn.tree |
| 65 | 02 | Inter-ML | [I05-03-Random-Forest.md](../02-intermedio/04-ml-clasico/I05-03-Random-Forest.md) | Random Forest: RandomForestClassifier/Regressor, n_estimators, OOB, feature_importance, hiperparámetros | sklearn.ensemble |
| 66 | 02 | Inter-ML | [I05-04-SVM.md](../02-intermedio/04-ml-clasico/I05-04-SVM.md) | SVM: SVC, SVR, kernels (linear, rbf, poly), C, gamma, margen, clasificación de clientes | sklearn.svm |
| 67 | 02 | Inter-ML | [I05-05-KNN.md](../02-intermedio/04-ml-clasico/I05-05-KNN.md) | KNN: KNeighborsClassifier/Regressor, K óptimo, métricas de distancia, peso, algoritmo | sklearn.neighbors |
| 68 | 02 | Inter-ML | [I05-06-Ensembles.md](../02-intermedio/04-ml-clasico/I05-06-Ensembles.md) | Ensembles: Bagging, AdaBoost, GradientBoosting, VotingClassifier, Stacking, comparación | sklearn.ensemble |

### 02-Intermedio: 05-Clustering

| # | Nivel | Sección | Archivo | Descripción | Librerías |
|---|-------|---------|---------|-------------|-----------|
| 69 | 02 | Inter-Clustering | [I06-01-KMeans.md](../02-intermedio/05-clustering/I06-01-KMeans.md) | K-Means: KMeans, método del codo, silhouette score, inicialización, segmentación de productos | sklearn.cluster |
| 70 | 02 | Inter-Clustering | [I06-02-DBSCAN.md](../02-intermedio/05-clustering/I06-02-DBSCAN.md) | DBSCAN: eps, min_samples, clustering basado en densidad, detección de outliers, comparación con K-Means | sklearn.cluster |
| 71 | 02 | Inter-Clustering | [I06-03-Hierarchico.md](../02-intermedio/05-clustering/I06-03-Hierarchico.md) | Clustering jerárquico: AgglomerativeClustering, dendrograma, linkage, distancia, interpretación | sklearn.cluster, scipy |
| 72 | 02 | Inter-Clustering | [I06-04-Metricas-Clustering.md](../02-intermedio/05-clustering/I06-04-Metricas-Clustering.md) | Métricas de clustering: silhouette, calinski_harabasz, davies_bouldin, validación interna/externa | sklearn.metrics |

### 02-Intermedio: 06-NLP

| # | Nivel | Sección | Archivo | Descripción | Librerías |
|---|-------|---------|---------|-------------|-----------|
| 73 | 02 | Inter-NLP | [I07-01-NLTK.md](../02-intermedio/06-nlp/I07-01-NLTK.md) | NLTK: tokenización, stemming, lematización, stopwords, FreqDist, concordancia, POS tagging | nltk |
| 74 | 02 | Inter-NLP | [I07-02-spaCy.md](../02-intermedio/06-nlp/I07-02-spaCy.md) | spaCy: pipelines, tokenización, POS tagging, dependencias, NER, visualización displacy | spacy |
| 75 | 02 | Inter-NLP | [I07-03-TextBlob.md](../02-intermedio/06-nlp/I07-03-TextBlob.md) | TextBlob: análisis de sentimiento, polaridad, subjetividad, corrección ortográfica, traducción, noun phrases | textblob |
| 76 | 02 | Inter-NLP | [I07-04-Analisis-Sentimientos.md](../02-intermedio/06-nlp/I07-04-Analisis-Sentimientos.md) | Análisis de sentimientos: aplicación completa a reseñas de productos, clasificación, visualización | nltk, textblob, sklearn |

### 02-Intermedio: 07-Casos Prácticos

| # | Nivel | Sección | Archivo | Descripción | Librerías |
|---|-------|---------|---------|-------------|-----------|
| 77 | 02 | Inter-Casos | [I08-01-Segmentacion-Clientes.md](../02-intermedio/07-casos/I08-01-Segmentacion-Clientes.md) | Segmentación de clientes: RFM, K-Means, perfilamiento, estrategias por segmento, visualización | pandas, sklearn, seaborn |
| 78 | 02 | Inter-Casos | [I08-02-Prediccion-Ventas.md](../02-intermedio/07-casos/I08-02-Prediccion-Ventas.md) | Predicción de ventas: regresión múltiple, Random Forest, ingeniería de características temporales | sklearn, pandas, numpy |
| 79 | 02 | Inter-Casos | [I08-03-Clasificacion-Productos.md](../02-intermedio/07-casos/I08-03-Clasificacion-Productos.md) | Clasificación de productos: categorización automática, comparación de algoritmos, optimización | sklearn, pandas |
| 80 | 02 | Inter-Casos | [I08-04-Analisis-Textos.md](../02-intermedio/07-casos/I08-04-Analisis-Textos.md) | Análisis de textos: procesamiento de reseñas, extracción de temas, nubes de palabras, sentimiento | nltk, wordcloud, sklearn |

---

## Nivel 03: Avanzado

ML avanzado, pipelines, feature engineering, NLP avanzado, Deep Learning con TensorFlow y PyTorch, y casos prácticos complejos.

### 03-Avanzado: 00-Scikit-learn Avanzado

| # | Nivel | Sección | Archivo | Descripción | Librerías |
|---|-------|---------|---------|-------------|-----------|
| 81 | 03 | Avz-Sklearn | [A01-01-Pipelines.md](../03-avanzado/00-sklearn-av/A01-01-Pipelines.md) | Pipelines: Pipeline, ColumnTransformer, FeatureUnion, transformadores personalizados, grid search | sklearn.pipeline |
| 82 | 03 | Avz-Sklearn | [A01-02-GridSearch.md](../03-avanzado/00-sklearn-av/A01-02-GridSearch.md) | Búsqueda de hiperparámetros: GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, ParamGrid | sklearn.model_selection |
| 83 | 03 | Avz-Sklearn | [A01-03-Ensembles-Av.md](../03-avanzado/00-sklearn-av/A01-03-Ensembles-Av.md) | Ensembles avanzados: Voting, Stacking con múltiples niveles, Bagging avanzado, calibración | sklearn.ensemble |
| 84 | 03 | Avz-Sklearn | [A01-04-XGBoost.md](../03-avanzado/00-sklearn-av/A01-04-XGBoost.md) | XGBoost: DMatrix, booster, early stopping, cv, feature importance, hiperparámetros avanzados | xgboost |
| 85 | 03 | Avz-Sklearn | [A01-05-LightGBM.md](../03-avanzado/00-sklearn-av/A01-05-LightGBM.md) | LightGBM: Dataset, categorical feature, leaf-wise, early stopping, tuning, GPU | lightgbm |
| 86 | 03 | Avz-Sklearn | [A01-06-CatBoost.md](../03-avanzado/00-sklearn-av/A01-06-CatBoost.md) | CatBoost: Pool, categorical features nativas, symmetric trees, overfitting detector, interpretación | catboost |

### 03-Avanzado: 01-Feature Engineering

| # | Nivel | Sección | Archivo | Descripción | Librerías |
|---|-------|---------|---------|-------------|-----------|
| 87 | 03 | Avz-Feature | [A02-01-Encoding.md](../03-avanzado/01-feature-eng/A02-01-Encoding.md) | Encoding de variables: OneHotEncoder, LabelEncoder, OrdinalEncoder, TargetEncoder, FrequencyEncoder | sklearn, category_encoders |
| 88 | 03 | Avz-Feature | [A02-02-Escalado.md](../03-avanzado/01-feature-eng/A02-02-Escalado.md) | Escalado de features: StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer, PowerTransformer | sklearn.preprocessing |
| 89 | 03 | Avz-Feature | [A02-03-Seleccion.md](../03-avanzado/01-feature-eng/A02-03-Seleccion.md) | Selección de características: SelectKBest, RFE, RFECV, SelectFromModel, PermutationImportance | sklearn.feature_selection |
| 90 | 03 | Avz-Feature | [A02-04-PCA.md](../03-avanzado/01-feature-eng/A02-04-PCA.md) | PCA: descomposición, varianza explicada, biplot, componentes principales, reducción de dimensionalidad | sklearn.decomposition |
| 91 | 03 | Avz-Feature | [A02-05-Polinomiales.md](../03-avanzado/01-feature-eng/A02-05-Polinomiales.md) | Features polinomiales: PolynomialFeatures, interacciones, splines, bins, discretización | sklearn.preprocessing |
| 92 | 03 | Avz-Feature | [A02-06-Target-Encoding.md](../03-avanzado/01-feature-eng/A02-06-Target-Encoding.md) | Target Encoding: MEstimateEncoder, CatBoostEncoder, JamesSteinEncoder, regularización, smoothing | category_encoders |

### 03-Avanzado: 02-NLP Avanzado

| # | Nivel | Sección | Archivo | Descripción | Librerías |
|---|-------|---------|---------|-------------|-----------|
| 93 | 03 | Avz-NLP | [A03-01-Gensim.md](../03-avanzado/02-nlp-av/A03-01-Gensim.md) | Gensim: corpus, diccionarios, TF-IDF, LSI, word2vec, Doc2Vec, similaridad entre documentos | gensim |
| 94 | 03 | Avz-NLP | [A03-02-Word2Vec.md](../03-avanzado/02-nlp-av/A03-02-Word2Vec.md) | Word2Vec: CBOW, Skip-gram, entrenamiento, similaridad, analogías, visualización con TSNE | gensim, sklearn |
| 95 | 03 | Avz-NLP | [A03-03-Topic-Modeling.md](../03-avanzado/02-nlp-av/A03-03-Topic-Modeling.md) | Modelado de tópicos: LDA, NMF, número óptimo de tópicos, interpretación, pyLDAvis | gensim, sklearn |
| 96 | 03 | Avz-NLP | [A03-04-TF-IDF-Av.md](../03-avanzado/02-nlp-av/A03-04-TF-IDF-Av.md) | TF-IDF avanzado: TfidfVectorizer, n-gramas, stopwords custom, límites, vocabulario, hashing | sklearn.feature_extraction |
| 97 | 03 | Avz-NLP | [A03-05-Embeddings.md](../03-avanzado/02-nlp-av/A03-05-Embeddings.md) | Embeddings: GloVe, FastText, embeddings pre-entrenados, integración con modelos clásicos | gensim, numpy |

### 03-Avanzado: 03-Deep Learning

| # | Nivel | Sección | Archivo | Descripción | Librerías |
|---|-------|---------|---------|-------------|-----------|
| 98 | 03 | Avz-DL | [A04-01-TensorFlow.md](../03-avanzado/03-dl/A04-01-TensorFlow.md) | TensorFlow: tensores, Keras, Sequential API, Functional API, compile, fit, evaluate, predict | tensorflow, keras |
| 99 | 03 | Avz-DL | [A04-02-PyTorch.md](../03-avanzado/03-dl/A04-02-PyTorch.md) | PyTorch: tensores, autograd, nn.Module, DataLoader, optimizadores, training loop, GPU | torch |
| 100 | 03 | Avz-DL | [A04-03-Redes-Densas.md](../03-avanzado/03-dl/A04-03-Redes-Densas.md) | Redes densas: Dense, activation, dropout, batch normalization, regularización, early stopping | tensorflow, torch |
| 101 | 03 | Avz-DL | [A04-04-CNN.md](../03-avanzado/03-dl/A04-04-CNN.md) | CNNs: Conv2D, MaxPooling, padding, strides, Flatten, arquitecturas clásicas (LeNet, VGG) | tensorflow, torch |
| 102 | 03 | Avz-DL | [A04-05-RNN.md](../03-avanzado/03-dl/A04-05-RNN.md) | RNNs: SimpleRNN, LSTM, GRU, Bidirectional, seq2seq, predicción de series temporales | tensorflow, torch |
| 103 | 03 | Avz-DL | [A04-06-Autoencoders.md](../03-avanzado/03-dl/A04-06-Autoencoders.md) | Autoencoders: undercomplete, denoising, variational (VAE), detección de anomalías en ventas | tensorflow, torch |
| 104 | 03 | Avz-DL | [A04-07-Transfer-Learning.md](../03-avanzado/03-dl/A04-07-Transfer-Learning.md) | Transfer Learning: modelos pre-entrenados, fine-tuning, feature extraction, congelamiento de capas | tensorflow, torch |

### 03-Avanzado: 04-Arquitecturas

| # | Nivel | Sección | Archivo | Descripción | Librerías |
|---|-------|---------|---------|-------------|-----------|
| 105 | 03 | Avz-Arq | [A05-01-Atajos-Arq.md](../03-avanzado/04-arquitecturas/A05-01-Atajos-Arq.md) | Atajos arquitectónicos: skip connections, residual blocks, layer normalization, batch norm, dropout | tensorflow, torch |

### 03-Avanzado: 05-Casos Prácticos

| # | Nivel | Sección | Archivo | Descripción | Librerías |
|---|-------|---------|---------|-------------|-----------|
| 106 | 03 | Avz-Casos | [A06-01-Deteccion-Anomalias.md](../03-avanzado/05-casos/A06-01-Deteccion-Anomalias.md) | Detección de anomalías en transacciones: Isolation Forest, LOF, Autoencoders, umbrales, métricas | sklearn, tensorflow |
| 107 | 03 | Avz-Casos | [A06-02-Prediccion-Series.md](../03-avanzado/05-casos/A06-02-Prediccion-Series.md) | Predicción de series temporales: LSTM, ventanas, multi-step, evaluación, comparación con modelos clásicos | tensorflow, pandas, sklearn |
| 108 | 03 | Avz-Casos | [A06-03-Clasificacion-Textos-Av.md](../03-avanzado/05-casos/A06-03-Clasificacion-Textos-Av.md) | Clasificación de textos avanzada: embeddings + LSTM, atención, fine-tuning, comparación de arquitecturas | tensorflow, gensim, sklearn |
| 109 | 03 | Avz-Casos | [A06-04-Recomendacion.md](../03-avanzado/05-casos/A06-04-Recomendacion.md) | Sistema de recomendación: filtro colaborativo con DL, Neural Collaborative Filtering, embeddings de usuario/producto | tensorflow, pandas |

---

## Nivel 04: Experto

Transformers, series temporales, sistemas de recomendación, producción con MLOps y casos complejos integradores.

### 04-Experto: 00-Transformers

| # | Nivel | Sección | Archivo | Descripción | Librerías |
|---|-------|---------|---------|-------------|-----------|
| 110 | 04 | Exp-Transformers | [E01-01-HuggingFace.md](../04-experto/00-transformers/E01-01-HuggingFace.md) | HuggingFace: pipeline, AutoModel, AutoTokenizer, datasets, trainer, carga/uso de modelos pre-entrenados | transformers, datasets |
| 111 | 04 | Exp-Transformers | [E01-02-BERT.md](../04-experto/00-transformers/E01-02-BERT.md) | BERT: fine-tuning para clasificación de reseñas, tokenización BERT, pooling, clasificación | transformers, torch |
| 112 | 04 | Exp-Transformers | [E01-03-GPT.md](../04-experto/00-transformers/E01-03-GPT.md) | GPT: generación de texto, zero-shot, few-shot, prompting, parámetros de generación (temperature, top_k) | transformers |
| 113 | 04 | Exp-Transformers | [E01-04-Fine-Tuning.md](../04-experto/00-transformers/E01-04-Fine-Tuning.md) | Fine-tuning de transformers: trainer, hiperparámetros, early stopping, evaluación, guardado y carga | transformers, torch |
| 114 | 04 | Exp-Transformers | [E01-05-LangChain.md](../04-experto/00-transformers/E01-05-LangChain.md) | LangChain: chains, prompts, memory, agents, tools, documentos, integración con LLMs | langchain |
| 115 | 04 | Exp-Transformers | [E01-06-RAG.md](../04-experto/00-transformers/E01-06-RAG.md) | RAG: Retrieval Augmented Generation, FAISS como vector store, chunking, retrieval, generación aumentada | langchain, faiss |

### 04-Experto: 01-Series Temporales

| # | Nivel | Sección | Archivo | Descripción | Librerías |
|---|-------|---------|---------|-------------|-----------|
| 116 | 04 | Exp-SeriesTemp | [E02-01-Prophet.md](../04-experto/01-series-temporales/E02-01-Prophet.md) | Prophet: forecasting, estacionalidad, días festivos, changepoints, regresores adicionales, incertidumbre | prophet |
| 117 | 04 | Exp-SeriesTemp | [E02-02-ARIMA.md](../04-experto/01-series-temporales/E02-02-ARIMA.md) | ARIMA/SARIMA: identificación de orden (ACF/PACF), estacionalidad, diagnóstico, predicción, auto_arima | statsmodels, pmdarima |
| 118 | 04 | Exp-SeriesTemp | [E02-03-LSTM-Temporal.md](../04-experto/01-series-temporales/E02-03-LSTM-Temporal.md) | LSTM para forecasting: seq2seq, encoder-decoder, attention, multi-step, ventanas deslizantes, backtesting | tensorflow, pandas |
| 119 | 04 | Exp-SeriesTemp | [E02-04-Deteccion-Tendencias.md](../04-experto/01-series-temporales/E02-04-Deteccion-Tendencias.md) | Detección de tendencias: descomposición STL, Mann-Kendall, changepoint detection, seasonality test | statsmodels, scipy |

### 04-Experto: 02-Sistemas de Recomendación

| # | Nivel | Sección | Archivo | Descripción | Librerías |
|---|-------|---------|---------|-------------|-----------|
| 120 | 04 | Exp-RecSys | [E03-01-Filtro-Colaborativo.md](../04-experto/02-recsys/E03-01-Filtro-Colaborativo.md) | Filtro colaborativo: user-based, item-based, memory-based, matriz de similaridad,邻里 | sklearn, numpy |
| 121 | 04 | Exp-RecSys | [E03-02-Factorizacion.md](../04-experto/02-recsys/E03-02-Factorizacion.md) | Factorización de matrices: SVD, SVD++, NMF, FunkSVD, ALS, reducción de dimensionalidad | surprise, sklearn |
| 122 | 04 | Exp-RecSys | [E03-03-Content-Based.md](../04-experto/02-recsys/E03-03-Content-Based.md) | Filtrado basado en contenido: perfiles de producto, similaridad por atributos, TF-IDF + coseno | sklearn, numpy |
| 123 | 04 | Exp-RecSys | [E03-04-FAISS.md](../04-experto/02-recsys/E03-04-FAISS.md) | FAISS: IndexFlatL2, IndexIVFFlat, IndexHNSWFlat, búsqueda de similaridad a gran escala, GPU | faiss |
| 124 | 04 | Exp-RecSys | [E03-05-Hybrid-RecSys.md](../04-experto/02-recsys/E03-05-Hybrid-RecSys.md) | Sistemas híbridos: weighted hybrid, switching, cascade, combined con FAISS + content-based + collaborative | faiss, sklearn, numpy |

### 04-Experto: 03-Optimización y Producción

| # | Nivel | Sección | Archivo | Descripción | Librerías |
|---|-------|---------|---------|-------------|-----------|
| 125 | 04 | Exp-Producción | [E04-01-MLflow.md](../04-experto/03-optim-prod/E04-01-MLflow.md) | MLflow: tracking, experiments, parameters, metrics, artifacts, model registry, serving | mlflow |
| 126 | 04 | Exp-Producción | [E04-02-FastAPI.md](../04-experto/03-optim-prod/E04-02-FastAPI.md) | FastAPI: endpoints REST, validación con Pydantic, documentación automática, async, middlewares | fastapi, uvicorn |
| 127 | 04 | Exp-Producción | [E04-03-Optuna.md](../04-experto/03-optim-prod/E04-03-Optuna.md) | Optuna: estudios, trials, samplers, pruners, multi-objective, visualización de resultados | optuna |
| 128 | 04 | Exp-Producción | [E04-04-SHAP.md](../04-experto/03-optim-prod/E04-04-SHAP.md) | SHAP: TreeExplainer, KernelExplainer, summary_plot, dependence_plot, force_plot, interpretabilidad | shap |
| 129 | 04 | Exp-Producción | [E04-05-Evidently.md](../04-experto/03-optim-prod/E04-05-Evidently.md) | Evidently: data drift, target drift, model performance, column mapping, reports, dashboards | evidently |
| 130 | 04 | Exp-Producción | [E04-06-Monitoreo.md](../04-experto/03-optim-prod/E04-06-Monitoreo.md) | Monitoreo en producción: alertas, logging estructurado, métricas de negocio, dashboards, retraining | mlflow, evidently |

### 04-Experto: 04-Casos Prácticos

| # | Nivel | Sección | Archivo | Descripción | Librerías |
|---|-------|---------|---------|-------------|-----------|
| 131 | 04 | Exp-Casos | [E05-01-Sistema-Recomendacion.md](../04-experto/04-casos/E05-01-Sistema-Recomendacion.md) | Sistema de recomendación completo: offline training + online serving, FAISS + hybrid, API | fastapi, faiss, sklearn |
| 132 | 04 | Exp-Casos | [E05-02-Forecast-Ventas.md](../04-experto/04-casos/E05-02-Forecast-Ventas.md) | Pipeline de forecasting: Prophet + validación + MLflow tracking + deployment como API | prophet, mlflow, fastapi |
| 133 | 04 | Exp-Casos | [E05-03-Chatbot-Analisis.md](../04-experto/04-casos/E05-03-Chatbot-Analisis.md) | Chatbot de análisis: LangChain + LLM + RAG sobre datos de ventas, agentes y herramientas | langchain, transformers |
| 134 | 04 | Exp-Casos | [E05-04-API-Produccion.md](../04-experto/04-casos/E05-04-API-Produccion.md) | API de producción: FastAPI + MLflow + Docker, CI/CD, pruebas, monitoreo, escalado | fastapi, mlflow, docker |

---

## Nivel 05: Apéndices

Material de referencia complementario: cheatsheets, glosarios y recursos adicionales.

| # | Nivel | Sección | Archivo | Descripción | Librerías |
|---|-------|---------|---------|-------------|-----------|
| 135 | 05 | Apéndices | [CHEATSHEET-PYTHON.md](../05-apendices/CHEATSHEET-PYTHON.md) | Cheatsheet de Python: sintaxis rápida, builtins, estructuras de datos, funciones | builtins |
| 136 | 05 | Apéndices | [CHEATSHEET-NUMPY.md](../05-apendices/CHEATSHEET-NUMPY.md) | Cheatsheet de NumPy: creación, operaciones, indexación, estadísticas, álgebra lineal | numpy |
| 137 | 05 | Apéndices | [CHEATSHEET-PANDAS.md](../05-apendices/CHEATSHEET-PANDAS.md) | Cheatsheet de Pandas: IO, limpieza, filtros, groupby, merge, fechas, apply | pandas |
| 138 | 05 | Apéndices | [CHEATSHEET-SEABORN.md](../05-apendices/CHEATSHEET-SEABORN.md) | Cheatsheet de Seaborn: tipos de gráficos, parámetros, estilos, paletas | seaborn |
| 139 | 05 | Apéndices | [CHEATSHEET-SCIKIT.md](../05-apendices/CHEATSHEET-SCIKIT.md) | Cheatsheet de Scikit-learn: modelos, métricas, preprocesamiento, selección | sklearn |
| 140 | 05 | Apéndices | [CHEATSHEET-TENSORFLOW.md](../05-apendices/CHEATSHEET-TENSORFLOW.md) | Cheatsheet de TensorFlow/Keras: capas, modelos, compilación, entrenamiento | tensorflow |
| 141 | 05 | Apéndices | [CHEATSHEET-PYTORCH.md](../05-apendices/CHEATSHEET-PYTORCH.md) | Cheatsheet de PyTorch: tensores, módulos, entrenamiento, GPU | torch |
| 142 | 05 | Apéndices | [CHEATSHEET-TRANSFORMERS.md](../05-apendices/CHEATSHEET-TRANSFORMERS.md) | Cheatsheet de HuggingFace Transformers: pipeline, modelos, tokenizers, fine-tuning | transformers |
| 143 | 05 | Apéndices | [CHEATSHEET-MLFLOW.md](../05-apendices/CHEATSHEET-MLFLOW.md) | Cheatsheet de MLflow: tracking, projects, models, registry, CLI | mlflow |
| 144 | 05 | Apéndices | [CHEATSHEET-FASTAPI.md](../05-apendices/CHEATSHEET-FASTAPI.md) | Cheatsheet de FastAPI: endpoints, parámetros, validación, dependencias, deployment | fastapi |
| 145 | 05 | Apéndices | [GLOSARIO.md](../05-apendices/GLOSARIO.md) | Glosario de términos: definiciones de conceptos de DS, ML, DL, estadística y negocio | — |
| 146 | 05 | Apéndices | [RECURSOS-EXTERNOS.md](../05-apendices/RECURSOS-EXTERNOS.md) | Recursos externos: libros, cursos, blogs, papers, comunidades, certificaciones | — |
| 147 | 05 | Apéndices | [REFERENCIAS-API.md](../05-apendices/REFERENCIAS-API.md) | Referencia rápida de APIs: funciones más usadas de cada librería con firmas y ejemplos | — |

---

## Datos Sintéticos

Módulo generador de datos utilizado en todos los ejemplos del manual.

| # | Tipo | Archivo | Descripción | Registros | Columnas |
|---|------|---------|-------------|-----------|----------|
| 148 | Código | [datos_sinteticos.py](../datos/datos_sinteticos.py) | Módulo generador de datos sintéticos con 5 datasets | — | — |
| 149 | CSV | [ventas.csv](../datos/ventas.csv) | Transacciones de venta diarias con estacionalidad y descuentos | ~1,330 | 16 |
| 150 | CSV | [inventario.csv](../datos/inventario.csv) | Estado actual del inventario por producto | 25 | 12 |
| 151 | CSV | [compras.csv](../datos/compras.csv) | Órdenes de compra a proveedores con retrasos | 200 | 17 |
| 152 | CSV | [clientes.csv](../datos/clientes.csv) | Perfil RFM de clientes con segmentación | 200 | 7 |
| 153 | CSV | [resenas.csv](../datos/resenas.csv) | Reseñas de productos con sentimiento y puntuación | 100 | 7 |

---

## Estadísticas del Manual

### Totales por Nivel

| Nivel | Archivos .md | Subdirectorios | Librerías principales |
|-------|-------------|----------------|----------------------|
| 00-base | 6 | 1 | — |
| 01-básico | 40 | 7 | numpy, pandas, seaborn, scipy, sklearn |
| 02-intermedio | 34 | 8 | numpy, pandas, seaborn, scipy, sklearn, nltk, spacy, textblob |
| 03-avanzado | 35 | 6 | sklearn, xgboost, lightgbm, catboost, tensorflow, torch, gensim |
| 04-experto | 31 | 5 | transformers, prophet, langchain, faiss, mlflow, fastapi, shap |
| 05-apéndices | 13 | 1 | — |
| **Total** | **159** | **28** | **~30 librerías** |

### Totales por Librería

| Librería | Archivos que la usan |
|----------|---------------------|
| pandas | ~60 |
| numpy | ~55 |
| sklearn | ~40 |
| seaborn | ~15 |
| matplotlib | ~10 |
| scipy | ~10 |
| tensorflow | ~10 |
| torch | ~10 |
| transformers | ~5 |
| nltk | ~4 |
| spacy | ~2 |
| textblob | ~2 |
| gensim | ~4 |
| xgboost | ~3 |
| lightgbm | ~2 |
| catboost | ~2 |
| mlflow | ~4 |
| fastapi | ~4 |
| langchain | ~3 |
| faiss | ~3 |
| prophet | ~2 |
| shap | ~2 |
| optuna | ~2 |
| evidently | ~2 |
| statsmodels | ~2 |
| category_encoders | ~2 |

### Distribución por Complejidad

```
Básico    ████████████████████████░  40 archivos (25%)
Intermedio █████████████████████░░  34 archivos (21%)
Avanzado   ██████████████████████░░ 35 archivos (22%)
Experto    ████████████████████░░░  31 archivos (19%)
Base       ████░░░░░░░░░░░░░░░░░░   6 archivos (4%)
Apéndices  ████████░░░░░░░░░░░░░░  13 archivos (8%)
```

**Salida:**

```
# Ver resultado al ejecutar el código
```

**Explicación:**

El código realiza operaciones de configuración/instalación/navegación. Consultar las instrucciones para más detalles.

---


### Archivos por Categoría Temática

| Temática | Archivos | Porcentaje |
|----------|----------|-----------|
| Python y fundamentos | 7 | 4% |
| NumPy (básico + avz) | 11 | 7% |
| Pandas (básico + avz) | 14 | 9% |
| Seaborn (básico + avz) | 8 | 5% |
| SciPy (básico + avz) | 8 | 5% |
| Scikit-learn ML básico | 5 | 3% |
| ML clásico intermedio | 6 | 4% |
| Clustering | 4 | 3% |
| NLP (intro + avz) | 9 | 6% |
| Feature engineering | 6 | 4% |
| Sklearn avanzado | 6 | 4% |
| Deep Learning | 7 | 4% |
| Arquitecturas | 1 | 1% |
| Transformers | 6 | 4% |
| Series temporales | 4 | 3% |
| Sistemas de recomendación | 5 | 3% |
| Producción y MLOps | 6 | 4% |
| Casos prácticos | 15 | 9% |
| Apéndices | 13 | 8% |
| Base (índice, nav, etc.) | 6 | 4% |
| Datos | 5 | 3% |

---

## Convenciones de Nomenclatura de Archivos

| Prefijo | Nivel |
|---------|-------|
| `B01-` a `B07-` | 01-básico (B = Básico) |
| `I01-` a `I08-` | 02-intermedio (I = Intermedio) |
| `A01-` a `A06-` | 03-avanzado (A = Avanzado) |
| `E01-` a `E05-` | 04-experto (E = Experto) |
| `00-` | 00-base (archivos de sistema) |
| Sin prefijo | 05-apéndices |

---

## Enlaces Rápidos

- [README principal](00-README.md)
- [Instalación](00-INSTALACION.md)
- [Convenciones](00-CONVENCIONES.md)
- [Mapa de rutas](00-NAVIGATION.md)
- [Generador de datos](00-GENERADOR-DATOS.md)
- [Datos sintéticos](../datos/datos_sinteticos.py)

---

*Este índice incluye todos los archivos planificados del manual. Los archivos marcados con enlace serán creados según la ruta de aprendizaje progresivo.*

*Volver al [README principal](00-README.md)*
