# Mapa de Rutas de Aprendizaje

## Manual Completo de Ciencia de Datos, NLP, Machine Learning y Deep Learning — Ventas, Compras e Inventarios

---

# Tabla de Contenidos

- [Introducción](#introducción)
- [Prerrequisitos Generales](#prerrequisitos-generales)
- [Dependencias entre Archivos](#dependencias-entre-archivos)
- [Ruta 1: Analista de Datos](#ruta-1-analista-de-datos)
- [Ruta 2: Científico de Datos](#ruta-2-científico-de-datos)
- [Ruta 3: Ingeniero de Machine Learning](#ruta-3-ingeniero-de-machine-learning)
- [Archivos Compartidos (Todos los Perfiles)](#archivos-compartidos-todos-los-perfiles)
- [Resumen de Tiempos Estimados](#resumen-de-tiempos-estimados)

---

## Introducción

Este mapa de rutas de aprendizaje está diseñado para guiarte a través del manual completo según tu perfil profesional. Existen tres rutas principales:

| Ruta | Perfil | Enfoque | Duración estimada |
|------|--------|---------|-------------------|
| 🟢 **Ruta 1** | Analista de Datos | Visualización, reportes, estadística descriptiva, limpieza de datos | 4-6 semanas |
| 🔵 **Ruta 2** | Científico de Datos | Modelado predictivo, clustering, NLP, Deep Learning | 12-16 semanas |
| 🟣 **Ruta 3** | Ingeniero ML | Producción, pipelines, optimización, MLOps, arquitecturas | 16-20 semanas |

Cada ruta especifica qué archivos leer y en qué orden. Los archivos están organizados en niveles:

- **00-base**: Instalación, convenciones, navegación, índice
- **01-basico**: Fundamentos (Python, NumPy, Pandas, Seaborn, SciPy, Scikit-learn)
- **02-intermedio**: ML clásico, clustering, NLP introductorio, análisis avanzado
- **03-avanzado**: Deep Learning, feature engineering, NLP avanzado, pipelines
- **04-experto**: Transformers, series temporales, sistemas de recomendación, producción
- **05-apendices**: Cheatsheets, glosarios, recursos

---

## Prerrequisitos Generales

### Prerrequisitos Técnicos

| Prerrequisito | Nivel requerido | ¿Dónde aprenderlo? |
|--------------|-----------------|-------------------|
| Conocimientos básicos de programación | Cualquier lenguaje | [B01-01](01-basico/00-python/B01-01-Introduccion-Python.md) |
| Matemáticas básicas (álgebra, estadística) | Nivel preparatoria | [B05-01](01-basico/04-scipy/B05-01-Introduccion-SciPy.md) |
| Python instalado | Funcionando | [00-INSTALACION.md](00-INSTALACION.md) |
| Entorno virtual configurado | Funcionando | [00-INSTALACION.md](00-INSTALACION.md) |

### Prerrequisitos por Nivel

| Nivel | Prerrequisitos |
|-------|---------------|
| **Básico** | Ninguno. Se empieza desde cero con Python. |
| **Intermedio** | Haber completado todo el nivel Básico. |
| **Avanzado** | Haber completado Básico e Intermedio. |
| **Experto** | Haber completado todos los niveles anteriores. |

### Prerrequisitos de Software

- Python 3.9+
- pip (gestor de paquetes)
- Git (opcional, para control de versiones)
- Jupyter Notebook o JupyterLab (recomendado)
- 8 GB RAM mínimo (16 GB recomendado para Deep Learning)

---

## Dependencias entre Archivos

El manual tiene una estructura de dependencias jerárquica. Cada archivo asume que ya conoces el contenido de los archivos anteriores dentro del mismo nivel y de los niveles previos.

### Mapa de Dependencias por Módulo

```
00-base/*
    │
    ▼
01-basico/00-python/* ──► 01-basico/01-numpy/*
    │                            │
    │                            ▼
    │                 ┌──── 01-basico/02-pandas/*
    │                 │
    │                 ▼
    │           ┌─ 01-basico/03-seaborn/*
    │           │
    │           ▼
    │     01-basico/04-scipy/*
    │           │
    │           ▼
    │     01-basico/05-sklearn/*
    │           │
    │           ▼
    │     01-basico/06-casos/*
    │           │
    ▼           ▼
02-intermedio/00-numpy-av/* ──► 02-intermedio/01-pandas-av/*
    │                                     │
    │                                     ▼
    │                          ┌─── 02-intermedio/02-seaborn-av/*
    │                          │
    │                          ▼
    │                    02-intermedio/03-scipy-av/*
    │                          │
    │                          ▼
    │              ┌─── 02-intermedio/04-ml-clasico/*
    │              │
    │              ▼
    │        02-intermedio/05-clustering/* ──► 02-intermedio/06-nlp/*
    │                                                     │
    │                                                     ▼
    │                                               02-intermedio/07-casos/*
    │                                                     │
    ▼                                                     ▼
03-avanzado/00-sklearn-av/* ──► 03-avanzado/01-feature-eng/*
    │                                     │
    │                                     ▼
    │                          ┌─── 03-avanzado/02-nlp-av/*
    │                          │
    │                          ▼
    │                    03-avanzado/03-dl/*
    │                          │
    │                          ▼
    │                    03-avanzado/04-arquitecturas/*
    │                          │
    │                          ▼
    │                    03-avanzado/05-casos/*
    │                          │
    ▼                          ▼
04-experto/00-transformers/* ──► 04-experto/01-series-temporales/*
    │                                     │
    │                                     ▼
    │                          ┌─── 04-experto/02-recsys/*
    │                          │
    │                          ▼
    │                    04-experto/03-optim-prod/*
    │                          │
    │                          ▼
    │                    04-experto/04-casos/*
    │
    ▼
05-apendices/*
```

**Salida:**

```
# Ver resultado al ejecutar el código
```

**Explicación:**

El código realiza operaciones de configuración/instalación/navegación. Consultar las instrucciones para más detalles.

---


### Dependencias Detalladas

| Archivo | Depende de |
|---------|-----------|
| 00-base/* | Ninguno (leer primero) |
| 01-basico/00-python/* | Ninguno |
| 01-basico/01-numpy/* | 01-basico/00-python/* |
| 01-basico/02-pandas/* | 01-basico/01-numpy/* |
| 01-basico/03-seaborn/* | 01-basico/02-pandas/* |
| 01-basico/04-scipy/* | 01-basico/01-numpy/* |
| 01-basico/05-sklearn/* | 01-basico/02-pandas/*, 01-basico/04-scipy/* |
| 01-basico/06-casos/* | Todos los anteriores del nivel básico |
| 02-intermedio/00-numpy-av/* | 01-basico/01-numpy/* |
| 02-intermedio/01-pandas-av/* | 01-basico/02-pandas/* |
| 02-intermedio/02-seaborn-av/* | 01-basico/03-seaborn/* |
| 02-intermedio/03-scipy-av/* | 01-basico/04-scipy/* |
| 02-intermedio/04-ml-clasico/* | 01-basico/05-sklearn/* |
| 02-intermedio/05-clustering/* | 02-intermedio/04-ml-clasico/* |
| 02-intermedio/06-nlp/* | 01-basico/02-pandas/* |
| 02-intermedio/07-casos/* | Todos los anteriores del nivel intermedio |
| 03-avanzado/00-sklearn-av/* | 02-intermedio/04-ml-clasico/* |
| 03-avanzado/01-feature-eng/* | 03-avanzado/00-sklearn-av/* |
| 03-avanzado/02-nlp-av/* | 02-intermedio/06-nlp/* |
| 03-avanzado/03-dl/* | 01-basico/01-numpy/* |
| 03-avanzado/04-arquitecturas/* | 03-avanzado/03-dl/* |
| 03-avanzado/05-casos/* | Todos los anteriores del nivel avanzado |
| 04-experto/00-transformers/* | 03-avanzado/02-nlp-av/*, 03-avanzado/03-dl/* |
| 04-experto/01-series-temporales/* | 03-avanzado/03-dl/* |
| 04-experto/02-recsys/* | 03-avanzado/05-casos/* |
| 04-experto/03-optim-prod/* | Todos los niveles anteriores |
| 04-experto/04-casos/* | Todos los anteriores del nivel experto |

---

## Ruta 1: Analista de Datos

### Perfil del Analista de Datos

Un **Analista de Datos** se enfoca en:
- Explorar y limpiar datos
- Crear visualizaciones y dashboards
- Generar reportes descriptivos
- Responder preguntas de negocio con datos históricos
- Comunicar hallazgos a stakeholders

**No requiere** modelado predictivo profundo ni despliegue en producción.

**Duración estimada:** 4-6 semanas (dedicando 10-15 horas/semana)

### Ruta de Archivos para Analista de Datos

#### Fase 0: Base (Día 1)

| Orden | Archivo | Descripción | Tiempo |
|-------|---------|-------------|--------|
| 0.1 | [00-README.md](00-README.md) | Presentación general del manual, objetivos y estructura | 15 min |
| 0.2 | [00-INSTALACION.md](00-INSTALACION.md) | Instalación de Python, entorno virtual y librerías | 1 hora |
| 0.3 | [00-CONVENCIONES.md](00-CONVENCIONES.md) | Convenciones de código y estilo del manual | 15 min |
| 0.4 | [00-NAVIGATION.md](00-NAVIGATION.md) | Este mapa de rutas | 15 min |
| 0.5 | [00-GENERADOR-DATOS.md](00-GENERADOR-DATOS.md) | Documentación del generador de datos sintéticos | 30 min |
| 0.6 | [00-INDEX.md](00-INDEX.md) | Índice maestro de todos los archivos del manual | 15 min |

#### Fase 1: Python Básico (Días 2-4)

| Orden | Archivo | Descripción | Tiempo |
|-------|---------|-------------|--------|
| 1.1 | [B01-01-Introduccion-Python.md](../01-basico/00-python/B01-01-Introduccion-Python.md) | Primeros pasos con Python: print, variables, tipos básicos | 2 horas |
| 1.2 | [B01-02-Tipos-Datos.md](../01-basico/00-python/B01-02-Tipos-Datos.md) | Números, strings, booleanos, conversiones | 2 horas |
| 1.3 | [B01-03-Estructuras-Control.md](../01-basico/00-python/B01-03-Estructuras-Control.md) | if/else, for, while, break, continue | 3 horas |
| 1.4 | [B01-04-Funciones.md](../01-basico/00-python/B01-04-Funciones.md) | Definición de funciones, parámetros, scope, lambda | 3 horas |
| 1.5 | [B01-05-Manejo-Archivos.md](../01-basico/00-python/B01-05-Manejo-Archivos.md) | Lectura/escritura de archivos CSV, JSON, TXT | 2 horas |
| 1.6 | [B01-06-Listas-Comprension.md](../01-basico/00-python/B01-06-Listas-Comprension.md) | List comprehension, dict comprehension, zip, enumerate | 2 horas |
| 1.7 | [B01-07-Errores-Excepciones.md](../01-basico/00-python/B01-07-Errores-Excepciones.md) | try/except, finally, raise, debugging básico | 1 hora |

#### Fase 2: NumPy (Días 5-6)

| Orden | Archivo | Descripción | Tiempo |
|-------|---------|-------------|--------|
| 2.1 | [B02-01-Introduccion-NumPy.md](../01-basico/01-numpy/B02-01-Introduccion-NumPy.md) | ¿Qué es NumPy?, arrays vs listas, creación básica | 2 horas |
| 2.2 | [B02-02-Arrays.md](../01-basico/01-numpy/B02-02-Arrays.md) | Creación de arrays: zeros, ones, arange, linspace, random | 2 horas |
| 2.3 | [B02-03-Operaciones.md](../01-basico/01-numpy/B02-03-Operaciones.md) | Operaciones vectorizadas, broadcasting básico, universal functions | 3 horas |
| 2.4 | [B02-04-Indexacion.md](../01-basico/01-numpy/B02-04-Indexacion.md) | Indexación básica, slicing, indexación booleana | 2 horas |
| 2.5 | [B02-05-Estadisticas.md](../01-basico/01-numpy/B02-05-Estadisticas.md) | mean, std, sum, min, max, percentiles, correlación | 2 horas |
| 2.6 | [B02-06-Algebra-Lineal.md](../01-basico/01-numpy/B02-06-Algebra-Lineal.md) | Producto punto, matrices, transpuesta, inversa (solo lo esencial) | 2 horas |

#### Fase 3: Pandas (Días 7-11)

| Orden | Archivo | Descripción | Tiempo |
|-------|---------|-------------|--------|
| 3.1 | [B03-01-Introduccion-Pandas.md](../01-basico/02-pandas/B03-01-Introduccion-Pandas.md) | Series, DataFrames, conceptos básicos | 2 horas |
| 3.2 | [B03-02-Series-DataFrames.md](../01-basico/02-pandas/B03-02-Series-DataFrames.md) | Creación, atributos, información del DataFrame | 2 horas |
| 3.3 | [B03-03-Lectura-Datos.md](../01-basico/02-pandas/B03-03-Lectura-Datos.md) | read_csv, read_excel, read_json, parámetros importantes | 2 horas |
| 3.4 | [B03-04-Limpieza.md](../01-basico/02-pandas/B03-04-Limpieza.md) | Valores nulos, duplicados, tipos de datos, outliers | 3 horas |
| 3.5 | [B03-05-Filtros-Condiciones.md](../01-basico/02-pandas/B03-05-Filtros-Condiciones.md) | Filtros, query, loc, iloc, condiciones compuestas | 3 horas |
| 3.6 | [B03-06-Agrupaciones.md](../01-basico/02-pandas/B03-06-Agrupaciones.md) | groupby, aggregate, transform, filtros por grupo | 3 horas |
| 3.7 | [B03-07-Join-Merge.md](../01-basico/02-pandas/B03-07-Join-Merge.md) | merge, join, concat, combinación de DataFrames | 3 horas |
| 3.8 | [B03-08-Dates-Times.md](../01-basico/02-pandas/B03-08-Dates-Times.md) | Fechas, resample, periodos, desplazamientos | 2 horas |
| 3.9 | [B03-09-Apply-Funciones.md](../01-basico/02-pandas/B03-09-Apply-Funciones.md) | apply, map, applymap, funciones por fila/columna | 2 horas |

#### Fase 4: Visualización con Seaborn (Días 12-14)

| Orden | Archivo | Descripción | Tiempo |
|-------|---------|-------------|--------|
| 4.1 | [B04-01-Introduccion-Seaborn.md](../01-basico/03-seaborn/B04-01-Introduccion-Seaborn.md) | Filosofía de Seaborn, temas, estilos básicos | 1 hora |
| 4.2 | [B04-02-Graficos-Distribucion.md](../01-basico/03-seaborn/B04-02-Graficos-Distribucion.md) | histplot, kdeplot, boxplot, violinplot | 2 horas |
| 4.3 | [B04-03-Graficos-Categoricos.md](../01-basico/03-seaborn/B04-03-Graficos-Categoricos.md) | barplot, countplot, catplot, pointplot | 2 horas |
| 4.4 | [B04-04-Graficos-Relacionales.md](../01-basico/03-seaborn/B04-04-Graficos-Relacionales.md) | scatterplot, lineplot, relplot, regplot | 2 horas |
| 4.5 | [B04-05-Estilizacion-Temas.md](../01-basico/03-seaborn/B04-05-Estilizacion-Temas.md) | Paletas de colores, temas, anotaciones, figsize | 1 hora |

#### Fase 5: Estadística Descriptiva con SciPy (Días 15-16)

| Orden | Archivo | Descripción | Tiempo |
|-------|---------|-------------|--------|
| 5.1 | [B05-01-Introduccion-SciPy.md](../01-basico/04-scipy/B05-01-Introduccion-SciPy.md) | ¿Qué es SciPy?, submódulos principales | 1 hora |
| 5.2 | [B05-02-Estadistica.md](../01-basico/04-scipy/B05-02-Estadistica.md) | Estadística descriptiva, distribuciones, probabilidad | 3 horas |
| 5.3 | [B05-04-Interpolacion.md](../01-basico/04-scipy/B05-04-Interpolacion.md) | Interpolación de datos, relleno de valores faltantes | 2 horas |

#### Fase 6: Casos Prácticos Básicos (Días 17-19)

| Orden | Archivo | Descripción | Tiempo |
|-------|---------|-------------|--------|
| 6.1 | [B07-01-Analisis-Ventas.md](../01-basico/06-casos/B07-01-Analisis-Ventas.md) | Análisis exploratorio completo de ventas con Pandas + Seaborn | 4 horas |
| 6.2 | [B07-02-Analisis-Inventario.md](../01-basico/06-casos/B07-02-Analisis-Inventario.md) | Análisis de inventario, rotación, productos críticos | 3 horas |
| 6.3 | [B07-03-Analisis-Compras.md](../01-basico/06-casos/B07-03-Analisis-Compras.md) | Análisis de órdenes de compra, proveedores, retrasos | 3 horas |

#### Fase 7: Intermedio — Análisis Avanzado (Días 20-23)

| Orden | Archivo | Descripción | Tiempo |
|-------|---------|-------------|--------|
| 7.1 | [I02-01-MultiIndex.md](../02-intermedio/01-pandas-av/I02-01-MultiIndex.md) | Índices jerárquicos, selección avanzada | 2 horas |
| 7.2 | [I02-02-Transformaciones.md](../02-intermedio/01-pandas-av/I02-02-Transformaciones.md) | melt, pivot, stack, unstack, cross-tab | 2 horas |
| 7.3 | [I02-03-Ventanas-Temporales.md](../02-intermedio/01-pandas-av/I02-03-Ventanas-Temporales.md) | rolling, expanding, ewm, ventanas de tiempo | 3 horas |
| 7.4 | [I02-04-Pivot-Cross.md](../02-intermedio/01-pandas-av/I02-04-Pivot-Cross.md) | Tablas dinámicas, pivot tables, crosstab para reportes | 2 horas |
| 7.5 | [I03-01-Multiplot.md](../02-intermedio/02-seaborn-av/I03-01-Multiplot.md) | FacetGrid, PairGrid, subplots múltiples | 2 horas |
| 7.6 | [I03-02-Heatmap-Cluster.md](../02-intermedio/02-seaborn-av/I03-02-Heatmap-Cluster.md) | Mapas de calor, clustermaps, correlaciones | 2 horas |
| 7.7 | [I04-02-ANOVA.md](../02-intermedio/03-scipy-av/I04-02-ANOVA.md) | ANOVA, comparación de grupos, pruebas post-hoc | 2 horas |

#### Fase 8: Casos Integradores (Días 24-26)

| Orden | Archivo | Descripción | Tiempo |
|-------|---------|-------------|--------|
| 8.1 | [I08-01-Segmentacion-Clientes.md](../02-intermedio/07-casos/I08-01-Segmentacion-Clientes.md) | Segmentación RFM, análisis de clientes | 4 horas |
| 8.2 | [I08-03-Clasificacion-Productos.md](../02-intermedio/07-casos/I08-03-Clasificacion-Productos.md) | Clasificación de productos por desempeño | 3 horas |

---

## Ruta 2: Científico de Datos

### Perfil del Científico de Datos

Un **Científico de Datos** se enfoca en:
- Modelado predictivo y prescriptivo
- Machine Learning clásico y avanzado
- Procesamiento de lenguaje natural (NLP)
- Deep Learning para tareas complejas
- Experimentación y validación de modelos
- Comunicación de resultados con respaldo estadístico

**Requiere** toda la ruta del Analista de Datos como base.

**Duración estimada:** 12-16 semanas (dedicando 15-20 horas/semana)

### Ruta de Archivos para Científico de Datos

#### Fase 0-7: Todo lo de la Ruta 1 (Analista)

Completar todas las fases de la Ruta 1 (Analista de Datos) antes de continuar.

#### Fase 8: ML Clásico Intermedio (Semanas 5-7)

| Orden | Archivo | Descripción | Tiempo |
|-------|---------|-------------|--------|
| 8.1 | [B06-01-Introduccion-Sklearn.md](../01-basico/05-sklearn/B06-01-Introduccion-Sklearn.md) | API de Scikit-learn, train/test split, primer modelo | 2 horas |
| 8.2 | [B06-02-Regresion-Lineal.md](../01-basico/05-sklearn/B06-02-Regresion-Lineal.md) | Regresión lineal simple y múltiple, métricas | 3 horas |
| 8.3 | [B06-03-Clasificacion.md](../01-basico/05-sklearn/B06-03-Clasificacion.md) | Logistic Regression, KNN, árbol de decisión | 4 horas |
| 8.4 | [B06-04-Metricas.md](../01-basico/05-sklearn/B06-04-Metricas.md) | Accuracy, precision, recall, F1, ROC-AUC, matriz de confusión | 3 horas |
| 8.5 | [B06-05-Validacion-Cruzada.md](../01-basico/05-sklearn/B06-05-Validacion-Cruzada.md) | Cross-validation, overfitting, curvas de validación | 3 horas |
| 8.6 | [I01-01-Broadcasting.md](../02-intermedio/00-numpy-av/I01-01-Broadcasting.md) | Broadcasting avanzado, eficiencia computacional | 2 horas |
| 8.7 | [I01-02-Mask-Index.md](../02-intermedio/00-numpy-av/I01-02-Mask-Index.md) | Indexación avanzada, masking, fancy indexing | 2 horas |
| 8.8 | [I05-01-Regresion-Av.md](../02-intermedio/04-ml-clasico/I05-01-Regresion-Av.md) | Regresión polinomial, Ridge, Lasso, ElasticNet | 3 horas |
| 8.9 | [I05-02-Arboles-Decision.md](../02-intermedio/04-ml-clasico/I05-02-Arboles-Decision.md) | Árboles de decisión avanzados, poda, interpretación | 3 horas |
| 8.10 | [I05-03-Random-Forest.md](../02-intermedio/04-ml-clasico/I05-03-Random-Forest.md) | Random Forest, feature importance, OOB score | 3 horas |
| 8.11 | [I05-04-SVM.md](../02-intermedio/04-ml-clasico/I05-04-SVM.md) | SVM, kernels, márgenes, clasificación no lineal | 3 horas |
| 8.12 | [I05-05-KNN.md](../02-intermedio/04-ml-clasico/I05-05-KNN.md) | KNN a profundidad, búsqueda de K óptimo, distancias | 2 horas |
| 8.13 | [I05-06-Ensembles.md](../02-intermedio/04-ml-clasico/I05-06-Ensembles.md) | Bagging, Boosting, Stacking, Voting classifiers | 4 horas |

#### Fase 9: Clustering y Segmentación (Semanas 7-8)

| Orden | Archivo | Descripción | Tiempo |
|-------|---------|-------------|--------|
| 9.1 | [I06-01-KMeans.md](../02-intermedio/05-clustering/I06-01-KMeans.md) | K-Means, método del codo, silueta, interpretación | 3 horas |
| 9.2 | [I06-02-DBSCAN.md](../02-intermedio/05-clustering/I06-02-DBSCAN.md) | DBSCAN, detección de outliers basada en densidad | 3 horas |
| 9.3 | [I06-03-Hierarchico.md](../02-intermedio/05-clustering/I06-03-Hierarchico.md) | Clustering jerárquico, dendrogramas, interpretación | 2 horas |
| 9.4 | [I06-04-Metricas-Clustering.md](../02-intermedio/05-clustering/I06-04-Metricas-Clustering.md) | Score de silueta, Davies-Bouldin, Calinski-Harabasz | 2 horas |

#### Fase 10: NLP (Semanas 8-9)

| Orden | Archivo | Descripción | Tiempo |
|-------|---------|-------------|--------|
| 10.1 | [I07-01-NLTK.md](../02-intermedio/06-nlp/I07-01-NLTK.md) | Tokenización, stemming, stopwords, frecuencia de términos | 3 horas |
| 10.2 | [I07-02-spaCy.md](../02-intermedio/06-nlp/I07-02-spaCy.md) | spaCy: POS tagging, entidades, dependencias, pipelines | 3 horas |
| 10.3 | [I07-03-TextBlob.md](../02-intermedio/06-nlp/I07-03-TextBlob.md) | TextBlob: análisis de sentimiento, corrección, traducción | 2 horas |
| 10.4 | [I07-04-Analisis-Sentimientos.md](../02-intermedio/06-nlp/I07-04-Analisis-Sentimientos.md) | Análisis de sentimientos aplicado a reseñas de productos | 3 horas |

#### Fase 11: Casos Intermedios (Semana 9)

| Orden | Archivo | Descripción | Tiempo |
|-------|---------|-------------|--------|
| 11.1 | [I08-02-Prediccion-Ventas.md](../02-intermedio/07-casos/I08-02-Prediccion-Ventas.md) | Modelo predictivo de ventas con ML clásico | 4 horas |
| 11.2 | [I08-04-Analisis-Textos.md](../02-intermedio/07-casos/I08-04-Analisis-Textos.md) | Análisis de texto aplicado a reseñas y comentarios | 3 horas |

#### Fase 12: Avanzado — Sklearn y Feature Engineering (Semanas 10-11)

| Orden | Archivo | Descripción | Tiempo |
|-------|---------|-------------|--------|
| 12.1 | [A01-01-Pipelines.md](../03-avanzado/00-sklearn-av/A01-01-Pipelines.md) | Pipelines de sklearn, transformadores personalizados | 3 horas |
| 12.2 | [A01-02-GridSearch.md](../03-avanzado/00-sklearn-av/A01-02-GridSearch.md) | Grid Search, Random Search, optimización de hiperparámetros | 3 horas |
| 12.3 | [A01-03-Ensembles-Av.md](../03-avanzado/00-sklearn-av/A01-03-Ensembles-Av.md) | Ensembles avanzados, Voting, Stacking personalizado | 3 horas |
| 12.4 | [A01-04-XGBoost.md](../03-avanzado/00-sklearn-av/A01-04-XGBoost.md) | XGBoost: gradient boosting, early stopping, interpretación | 3 horas |
| 12.5 | [A01-05-LightGBM.md](../03-avanzado/00-sklearn-av/A01-05-LightGBM.md) | LightGBM: entrenamiento eficiente, categorical features | 3 horas |
| 12.6 | [A01-06-CatBoost.md](../03-avanzado/00-sklearn-av/A01-06-CatBoost.md) | CatBoost: manejo nativo de categóricas, interpretación | 2 horas |
| 12.7 | [A02-01-Encoding.md](../03-avanzado/01-feature-eng/A02-01-Encoding.md) | One-hot, label, ordinal, target encoding, frequency encoding | 3 horas |
| 12.8 | [A02-02-Escalado.md](../03-avanzado/01-feature-eng/A02-02-Escalado.md) | StandardScaler, MinMaxScaler, RobustScaler, Normalizer | 2 horas |
| 12.9 | [A02-03-Seleccion.md](../03-avanzado/01-feature-eng/A02-03-Seleccion.md) | Selección de características: filter, wrapper, embedded | 3 horas |
| 12.10 | [A02-04-PCA.md](../03-avanzado/01-feature-eng/A02-04-PCA.md) | PCA: reducción de dimensionalidad, varianza explicada | 3 horas |

#### Fase 13: NLP Avanzado (Semanas 11-12)

| Orden | Archivo | Descripción | Tiempo |
|-------|---------|-------------|--------|
| 13.1 | [A03-01-Gensim.md](../03-avanzado/02-nlp-av/A03-01-Gensim.md) | Gensim: word embeddings, Doc2Vec, similaridad | 3 horas |
| 13.2 | [A03-02-Word2Vec.md](../03-avanzado/02-nlp-av/A03-02-Word2Vec.md) | Word2Vec: CBOW, Skip-gram, entrenamiento | 3 horas |
| 13.3 | [A03-03-Topic-Modeling.md](../03-avanzado/02-nlp-av/A03-03-Topic-Modeling.md) | LDA, NMF, modelado de tópicos en reseñas | 3 horas |
| 13.4 | [A03-04-TF-IDF-Av.md](../03-avanzado/02-nlp-av/A03-04-TF-IDF-Av.md) | TF-IDF avanzado, n-gramas, límites y alternativas | 2 horas |
| 13.5 | [A03-05-Embeddings.md](../03-avanzado/02-nlp-av/A03-05-Embeddings.md) | Embeddings pre-entrenados, GloVe, FastText | 3 horas |

#### Fase 14: Deep Learning (Semanas 12-14)

| Orden | Archivo | Descripción | Tiempo |
|-------|---------|-------------|--------|
| 14.1 | [A04-01-TensorFlow.md](../03-avanzado/03-dl/A04-01-TensorFlow.md) | TensorFlow: tensores, Keras API, modelo secuencial | 4 horas |
| 14.2 | [A04-02-PyTorch.md](../03-avanzado/03-dl/A04-02-PyTorch.md) | PyTorch: tensores, autograd, nn.Module, entrenamiento | 4 horas |
| 14.3 | [A04-03-Redes-Densas.md](../03-avanzado/03-dl/A04-03-Redes-Densas.md) | Redes densas: arquitectura, activaciones, regularización, dropout | 4 horas |
| 14.4 | [A04-05-RNN.md](../03-avanzado/03-dl/A04-05-RNN.md) | RNN, LSTM, GRU para secuencias temporales de ventas | 4 horas |
| 14.5 | [A04-06-Autoencoders.md](../03-avanzado/03-dl/A04-06-Autoencoders.md) | Autoencoders para detección de anomalías en ventas | 3 horas |

#### Fase 15: Casos Avanzados (Semana 14)

| Orden | Archivo | Descripción | Tiempo |
|-------|---------|-------------|--------|
| 15.1 | [A06-01-Deteccion-Anomalias.md](../03-avanzado/05-casos/A06-01-Deteccion-Anomalias.md) | Detección de anomalías en transacciones con DL | 4 horas |
| 15.2 | [A06-02-Prediccion-Series.md](../03-avanzado/05-casos/A06-02-Prediccion-Series.md) | Predicción de series temporales de ventas con LSTM | 4 horas |
| 15.3 | [A06-03-Clasificacion-Textos-Av.md](../03-avanzado/05-casos/A06-03-Clasificacion-Textos-Av.md) | Clasificación avanzada de textos con embeddings + DL | 3 horas |

#### Fase 16: Experto — Transformers y Modelos de Lenguaje (Semanas 15-16)

| Orden | Archivo | Descripción | Tiempo |
|-------|---------|-------------|--------|
| 16.1 | [E01-01-HuggingFace.md](../04-experto/00-transformers/E01-01-HuggingFace.md) | HuggingFace: pipeline, datasets, tokenizers | 3 horas |
| 16.2 | [E01-02-BERT.md](../04-experto/00-transformers/E01-02-BERT.md) | BERT: fine-tuning para clasificación de reseñas | 4 horas |
| 16.3 | [E01-03-GPT.md](../04-experto/00-transformers/E01-03-GPT.md) | GPT: generación de texto, zero-shot, few-shot | 3 horas |
| 16.4 | [E01-04-Fine-Tuning.md](../04-experto/00-transformers/E01-04-Fine-Tuning.md) | Fine-tuning de transformers para tareas específicas de negocio | 4 horas |

---

## Ruta 3: Ingeniero de Machine Learning

### Perfil del Ingeniero de ML

Un **Ingeniero de Machine Learning** se enfoca en:
- Diseñar arquitecturas de ML escalables
- Implementar pipelines de datos y modelos en producción
- Optimizar modelos para latencia y memoria
- Desplegar APIs y servicios de ML
- Monitorear y mantener modelos en producción
- Infraestructura y MLOps

**Requiere** las rutas 1 y 2 completas como base.

**Duración estimada:** 16-20 semanas (dedicando 20+ horas/semana)

### Ruta de Archivos para Ingeniero de ML

#### Fase 0-15: Rutas 1 y 2 completas

Completar todas las fases de las Rutas 1 (Analista) y 2 (Científico de Datos).

#### Fase 16: Arquitecturas y Optimización (Semanas 10-12)

| Orden | Archivo | Descripción | Tiempo |
|-------|---------|-------------|--------|
| 16.1 | [A04-04-CNN.md](../03-avanzado/03-dl/A04-04-CNN.md) | CNNs: convoluciones, pooling, arquitecturas para imágenes | 4 horas |
| 16.2 | [A04-07-Transfer-Learning.md](../03-avanzado/03-dl/A04-07-Transfer-Learning.md) | Transfer Learning, modelos pre-entrenados, fine-tuning | 3 horas |
| 16.3 | [A05-01-Atajos-Arq.md](../03-avanzado/04-arquitecturas/A05-01-Atajos-Arq.md) | Atajos arquitectónicos, skip connections, normalization | 3 horas |
| 16.4 | [A02-05-Polinomiales.md](../03-avanzado/01-feature-eng/A02-05-Polinomiales.md) | Features polinomiales, interacciones, splines | 2 horas |
| 16.5 | [A02-06-Target-Encoding.md](../03-avanzado/01-feature-eng/A02-06-Target-Encoding.md) | Target encoding, smoothing, regularización | 2 horas |

#### Fase 17: Transformers y LangChain (Semanas 12-13)

| Orden | Archivo | Descripción | Tiempo |
|-------|---------|-------------|--------|
| 17.1 | [E01-05-LangChain.md](../04-experto/00-transformers/E01-05-LangChain.md) | LangChain: chains, agents, memory, tools, RAG básico | 4 horas |
| 17.2 | [E01-06-RAG.md](../04-experto/00-transformers/E01-06-RAG.md) | RAG: Retrieval Augmented Generation, FAISS + LLM | 4 horas |

#### Fase 18: Series Temporales para Producción (Semanas 13-14)

| Orden | Archivo | Descripción | Tiempo |
|-------|---------|-------------|--------|
| 18.1 | [E02-01-Prophet.md](../04-experto/01-series-temporales/E02-01-Prophet.md) | Prophet: forecasting, estacionalidad, festividades, changepoints | 3 horas |
| 18.2 | [E02-02-ARIMA.md](../04-experto/01-series-temporales/E02-02-ARIMA.md) | ARIMA, SARIMA, identificación de orden, diagnóstico | 3 horas |
| 18.3 | [E02-03-LSTM-Temporal.md](../04-experto/01-series-temporales/E02-03-LSTM-Temporal.md) | LSTM para forecasting: seq2seq, attention, multi-step | 4 horas |
| 18.4 | [E02-04-Deteccion-Tendencias.md](../04-experto/01-series-temporales/E02-04-Deteccion-Tendencias.md) | Detección de tendencias, estacionalidad, residuos | 2 horas |

#### Fase 19: Sistemas de Recomendación (Semanas 14-15)

| Orden | Archivo | Descripción | Tiempo |
|-------|---------|-------------|--------|
| 19.1 | [E03-01-Filtro-Colaborativo.md](../04-experto/02-recsys/E03-01-Filtro-Colaborativo.md) | Filtro colaborativo: user-based, item-based, memory-based | 3 horas |
| 19.2 | [E03-02-Factorizacion.md](../04-experto/02-recsys/E03-02-Factorizacion.md) | Factorización de matrices: SVD, NMF, FunkSVD | 3 horas |
| 19.3 | [E03-03-Content-Based.md](../04-experto/02-recsys/E03-03-Content-Based.md) | Filtrado basado en contenido: perfiles, similaridad | 3 horas |
| 19.4 | [E03-04-FAISS.md](../04-experto/02-recsys/E03-04-FAISS.md) | FAISS: búsqueda de similaridad a gran escala, indexación | 3 horas |
| 19.5 | [E03-05-Hybrid-RecSys.md](../04-experto/02-recsys/E03-05-Hybrid-RecSys.md) | Sistemas híbridos: combinando enfoques, weighted hybrid | 3 horas |

#### Fase 20: Producción y MLOps (Semanas 15-17)

| Orden | Archivo | Descripción | Tiempo |
|-------|---------|-------------|--------|
| 20.1 | [E04-01-MLflow.md](../04-experto/03-optim-prod/E04-01-MLflow.md) | MLflow: tracking, registries, models, experiment management | 4 horas |
| 20.2 | [E04-02-FastAPI.md](../04-experto/03-optim-prod/E04-02-FastAPI.md) | FastAPI: endpoints REST, validación, documentación automática | 4 horas |
| 20.3 | [E04-03-Optuna.md](../04-experto/03-optim-prod/E04-03-Optuna.md) | Optuna: optimización de hiperparámetros, pruning, estudios | 3 horas |
| 20.4 | [E04-04-SHAP.md](../04-experto/03-optim-prod/E04-04-SHAP.md) | SHAP: interpretabilidad de modelos, feature importance, plots | 4 horas |
| 20.5 | [E04-05-Evidently.md](../04-experto/03-optim-prod/E04-05-Evidently.md) | Evidently: monitoreo de datos, drift, calidad de modelo | 3 horas |
| 20.6 | [E04-06-Monitoreo.md](../04-experto/03-optim-prod/E04-06-Monitoreo.md) | Monitoreo en producción: alertas, logging, dashboards | 3 horas |

#### Fase 21: Casos de Producción (Semanas 17-18)

| Orden | Archivo | Descripción | Tiempo |
|-------|---------|-------------|--------|
| 21.1 | [E05-01-Sistema-Recomendacion.md](../04-experto/04-casos/E05-01-Sistema-Recomendacion.md) | Sistema de recomendación completo: offline + online | 5 horas |
| 21.2 | [E05-02-Forecast-Ventas.md](../04-experto/04-casos/E05-02-Forecast-Ventas.md) | Pipeline de forecasting: Prophet + validación + deployment | 5 horas |
| 21.3 | [E05-03-Chatbot-Analisis.md](../04-experto/04-casos/E05-03-Chatbot-Analisis.md) | Chatbot de análisis de ventas con LangChain + LLM | 5 horas |
| 21.4 | [E05-04-API-Produccion.md](../04-experto/04-casos/E05-04-API-Produccion.md) | API de modelo en producción: FastAPI + MLflow + Docker | 5 horas |

#### Fase 22: Casos Avanzados Adicionales (Semana 18)

| Orden | Archivo | Descripción | Tiempo |
|-------|---------|-------------|--------|
| 22.1 | [A06-04-Recomendacion.md](../03-avanzado/05-casos/A06-04-Recomendacion.md) | Sistema de recomendación de productos con DL | 4 horas |
| 22.2 | [I08-01-Segmentacion-Clientes.md](../02-intermedio/07-casos/I08-01-Segmentacion-Clientes.md) | Segmentación RFM avanzada con clustering | 3 horas |

---

## Archivos Compartidos (Todos los Perfiles)

Independientemente de la ruta elegida, los siguientes archivos son útiles para todos los perfiles:

| Archivo | Perfil | Utilidad |
|---------|--------|----------|
| [I04-01-Pruebas-Hipotesis.md](../02-intermedio/03-scipy-av/I04-01-Pruebas-Hipotesis.md) | Todos | Pruebas estadísticas para validar hipótesis de negocio |
| [I04-03-Distribuciones.md](../02-intermedio/03-scipy-av/I04-03-Distribuciones.md) | Todos | Distribuciones de probabilidad aplicadas a ventas |
| [I03-03-Personalizacion.md](../02-intermedio/02-seaborn-av/I03-03-Personalizacion.md) | Analista, CD | Personalización avanzada de gráficos para reportes |
| [I04-04-Senales.md](../02-intermedio/03-scipy-av/I04-04-Senales.md) | CD, Ing. ML | Procesamiento de señales, filtros, tendencias |
| [I01-03-Stride-Tricks.md](../02-intermedio/00-numpy-av/I01-03-Stride-Tricks.md) | Ing. ML | Optimización de memoria con view vs copy |
| [I01-04-Polinomios.md](../02-intermedio/00-numpy-av/I01-04-Polinomios.md) | CD, Ing. ML | Ajuste polinomial, interpolación avanzada |
| [B05-03-Optimizacion.md](../01-basico/04-scipy/B05-03-Optimizacion.md) | CD, Ing. ML | Optimización de funciones, programación lineal |
| [I02-05-Performance.md](../02-intermedio/01-pandas-av/I02-05-Performance.md) | Todos | Optimización de código Pandas, memoria, velocidad |
| [I05-01-Regresion-Av.md](../02-intermedio/04-ml-clasico/I05-01-Regresion-Av.md) | CD, Ing. ML | Regresión avanzada para predicción de ventas |
| [B07-01-Analisis-Ventas.md](../01-basico/06-casos/B07-01-Analisis-Ventas.md) | Todos | Caso práctico de análisis exploratorio de ventas |
| [B07-02-Analisis-Inventario.md](../01-basico/06-casos/B07-02-Analisis-Inventario.md) | Todos | Análisis de inventario y rotación de productos |
| [B07-03-Analisis-Compras.md](../01-basico/06-casos/B07-03-Analisis-Compras.md) | Todos | Análisis de compras y desempeño de proveedores |

---

## Resumen de Tiempos Estimados

### Por Perfil

| Perfil | Archivos | Horas totales | Semanas (10h/sem) | Semanas (20h/sem) |
|--------|----------|---------------|-------------------|-------------------|
| **Analista de Datos** | ~45 | 100-120 | 10-12 | 5-6 |
| **Científico de Datos** | ~85 | 240-280 | 24-28 | 12-14 |
| **Ingeniero de ML** | ~110 | 320-380 | 32-38 | 16-19 |

### Por Nivel

| Nivel | Archivos | Horas estimadas |
|-------|----------|----------------|
| **00-base** | 6 | 2-3 |
| **01-basico** | 40 | 60-80 |
| **02-intermedio** | 34 | 60-80 |
| **03-avanzado** | 35 | 80-100 |
| **04-experto** | 31 | 80-100 |
| **05-apendices** | — | — |
| **Total** | ~146 | 280-360 |

### Consejos de Estudio

1. **No te saltes ejercicios**: Cada archivo tiene ejercicios propuestos que refuerzan el aprendizaje.
2. **Ejecuta todo el código**: No solo leas; ejecuta cada ejemplo y modifica parámetros.
3. **Toma notas**: Crea un cuaderno Jupyter personal con tus apuntes y experimentos.
4. **Repite los casos prácticos**: Los casos al final de cada nivel integran múltiples conceptos.
5. **Usa los apéndices**: Consulta las cheatsheets y glosarios como referencia rápida.
6. **Forma grupos de estudio**: Discutir conceptos con otros acelera el aprendizaje.

---

*Este mapa de rutas es una guía recomendada. Siéntete libre de saltar entre secciones según tu experiencia previa y necesidades específicas.*

*Volver al [Índice Maestro](00-INDEX.md) | [README principal](00-README.md)*
