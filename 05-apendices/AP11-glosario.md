# AP11 — Glosario de Términos Técnicos (Contexto Ventas ML)

## A

**Accuracy** — Proporción de predicciones correctas sobre el total. En ventas: ¿qué tan seguido acertamos si un cliente comprará o no? Fórmula: (TP+TN)/(TP+TN+FP+FN). No recomendada para datos desbalanceados.

**Activation Function** — Función no lineal (ReLU, sigmoid, tanh) en una neurona que decide si se activa. En ventas: capas ocultas en redes que aprenden patrones de compra.

**AdaBoost** — Ensemble que combina clasificadores débiles, ponderando errores anteriores. Útil para detectar churn en ventas.

**Agarwal's Rule** — Heurística para feature engineering: crear interacciones entre features numéricas y categóricas.

**AIC / BIC** — Criterios de información para selección de modelos. Menor valor = mejor balance ajuste-complejidad.

**API** — Interfaz de programación para comunicar servicios. Ej: API REST para predicción de ventas.

**ARIMA** — Modelo de series temporales (AutoRegressive Integrated Moving Average). Para forecast de ventas diarias/mensuales.

**AUC-ROC** — Área bajo la curva ROC. Mide capacidad de discriminación del modelo. AUC=0.5 es aleatorio, AUC=1.0 es perfecto.

**Autoencoder** — Red que comprime y reconstruye datos. Útil para detección de anomalías en transacciones.

**Autograd** — Diferenciación automática en PyTorch/TensorFlow. Permite backpropagation sin derivadas manuales.

**Average Precision** — Promedio de precisión a diferentes thresholds. Para ranking de productos recomendados.

## B

**Backpropagation** — Algoritmo que calcula gradientes del error respecto a pesos, propagando el error hacia atrás. Esencial para entrenar redes profundas.

**Bagging** — Bootstrap Aggregating: entrenar múltiples modelos en submuestras y promediar. Ej: Random Forest.

**Baseline** — Modelo simple de referencia (predictir media, último valor). Toda mejora debe superar el baseline.

**Batch** — Subconjunto de datos usado en una iteración de entrenamiento. Batch size=32 es común.

**Batch Normalization** — Normaliza activaciones entre capas. Acelera entrenamiento y reduce overfitting.

**Bayesian Optimization** — Optimización de hiperparámetros usando modelos probabilísticos (Gaussian Processes).

**Bias (sesgo)** — Error por asumir que la relación es más simple de lo que es. Alto bias = underfitting.

**Bias (sesgo en datos)** — Desbalance en datos que lleva a predicciones injustas. Ej: recomendar más productos a un grupo demográfico.

**Bias-Variance Tradeoff** — Balance entre sesgo (underfitting) y varianza (overfitting). El objetivo es minimizar ambos.

**Big Data** — Datos que exceden capacidad de procesamiento tradicional. Volumen, velocidad, variedad.

**Boosting** — Ensemble secuencial donde cada modelo corrige errores del anterior. Ej: XGBoost, LightGBM.

**Bootstrapping** — Muestreo con reemplazo. Base de Bagging y Random Forest.

**Box Plot** — Gráfico de distribución basado en cuartiles. Útil para detectar outliers en montos de venta.

**Broadcasting** — Operaciones entre arrays de diferentes formas en NumPy.

## C

**Categorical Feature** — Variable con valores discretos (ej: categoría de producto, región).

**Chain Rule** — Regla de la cadena del cálculo usada en backpropagation.

**Chi-Square Test** — Prueba estadística para independencia entre variables categóricas.

**Classification** — Predecir una etiqueta discreta. Ej: clasificar si un cliente comprará o no.

**Clustering** — Agrupar datos similares sin etiquetas. Ej: segmentar clientes por comportamiento de compra.

**Cold Start** — Problema de recomendar items nuevos o a usuarios nuevos sin historial.

**Collinearity** — Correlación alta entre features predictores. Degrada modelos lineales.

**ColumnTransformer** — Clase sklearn para aplicar transformaciones diferentes a distintas columnas.

**Confusion Matrix** — Tabla TP/FP/TN/FN. Base para calcular precisión, recall, F1.

**Content-Based Filtering** — Recomendación basada en atributos del item. Ej: recomendar productos similares a los que compró.

**Convergence** — Cuando el algoritmo de optimización alcanza un mínimo (local o global).

**Conv1D / Conv2D** — Capas convolutionales para datos 1D (series) y 2D (imágenes).

**Correlation** — Relación lineal entre variables. Pearson: -1 a 1.

**Cosine Similarity** — Similitud basada en ángulo entre vectores. Usada en recomendación y NLP.

**Cross-Validation** — Dividir datos en k folds para evaluar el modelo robustamente. k=5 es estándar.

**Cumulative Gain** — Suma de relevancia en ranking. Base para métricas de recomendación.

**Curse of Dimensionality** — Problemas que surgen con muchas dimensiones: datos se vuelven sparse, distancia pierde significado.

## D

**Data Augmentation** — Crear datos sintéticos aumentando el dataset. Útil para imágenes de productos.

**Data Drift** — Cambio en la distribución de los datos de entrada respecto a entrenamiento. Detectable con Evidently.

**Data Leakage** — Información del futuro filtrada al entrenamiento. Ej: usar ventas futuras para predecir ventas presentes.

**DataLoader** — Clase PyTorch para iterar sobre datasets en batches.

**DBSCAN** — Clustering basado en densidad. Detecta clusters de forma arbitraria y outliers.

**Decision Tree** — Árbol de decisiones con splits basados en features. Base de Random Forest.

**Deep Learning** — Redes neuronales con múltiples capas ocultas. Para patrones complejos en datos.

**Degree of Freedom** — Número de parámetros libres en un modelo. Más grados = más complejidad.

**Dendrogram** — Diagrama de árbol para clustering jerárquico.

**Dense Layer** — Capa fully-connected en redes neuronales. Cada neurona conectada a todas las de la capa anterior.

**Dependency Parsing** — Análisis de dependencias gramaticales en NLP.

**Det (Determinant)** — Valor escalar asociado a una matriz. Si det=0, la matriz es singular.

**Device** — CPU o GPU donde se ejecutan tensores en PyTorch/TensorFlow.

**DF (DataFrame)** — Estructura tabular principal de pandas. Filas y columnas con tipos heterogéneos.

**Dimensionality Reduction** — Reducir número de features. PCA, t-SNE, UMAP.

**Discriminator** — En GANs, red que distingue datos reales de falsos.

**Dropout** — Técnica de regularización que apaga aleatoriamente neuronas durante entrenamiento.

**Dtype** — Tipo de datos en numpy/pandas (float32, int64, object).

**DVC (Data Version Control)** — Herramienta para versionar datasets y pipelines ML.

## E

**Eager Execution** — Modo por defecto en TF 2.x: operaciones se ejecutan inmediatamente.

**Early Stopping** — Detener entrenamiento cuando la métrica de validación deja de mejorar.

**Eigenvalue / Eigenvector** — Autovalores/autovectores de una matriz. Base de PCA.

**ElasticNet** — Regresión lineal con regularización L1+L2. Balance entre Lasso y Ridge.

**Embedding** — Representación densa de baja dimensión de datos categóricos. Ej: word embeddings, user/item embeddings.

**Ensemble** — Combinación de múltiples modelos. Voting, Bagging, Boosting, Stacking.

**Epoch** — Una pasada completa por todo el dataset de entrenamiento.

**Error** — Diferencia entre predicción y valor real.

**Euclidean Distance** — Distancia geométrica entre puntos. sqrt(sum((x-y)²)).

**Evaluation Metric** — Métrica para medir rendimiento del modelo. Accuracy, F1, RMSE, etc.

**Exponential Smoothing** — Técnica de forecast que pondera observaciones pasadas exponencialmente.

## F

**F1-Score** — Media armónica de precision y recall. 2 * (P * R) / (P + R). Balance entre ambas.

**FacetGrid** — Sistema de subplots en seaborn por categorías.

**FastAPI** — Framework moderno para APIs en Python. Async, validación automática con Pydantic.

**Feature** — Variable predictora/columna de entrada al modelo.

**Feature Engineering** — Creación de nuevas features a partir de datos existentes. Lags, rolling, interacciones.

**Feature Importance** — Score de relevancia de cada feature en el modelo. Disponible en tree-based models.

**Feature Scaling** — Estandarizar features a escalas comparables. StandardScaler, MinMaxScaler, RobustScaler.

**Feature Selection** — Seleccionar subconjunto relevante de features. SelectKBest, RFE, SelectFromModel.

**Feedforward** — Paso hacia adelante en una red: entrada → capas → salida.

**Filtering** — En recomendación: collaborative filtering, content-based filtering.

**Fine-tuning** — Ajuste de capas pre-entrenadas con datos específicos del dominio.

**Flatten** — Aplanar tensor multidimensional a 1D. Necesario entre Conv layers y Dense layers.

**Forecast Horizon** — Número de pasos hacia adelante a predecir en series temporales.

**Fourier Transform** — Descomponer señal en componentes de frecuencia. Útil para estacionalidad.

**FPR (False Positive Rate)** — FP / (FP + TN). Tasa de falsos positivos.

## G

**GAN (Generative Adversarial Network)** — Red generativa + discriminativa. Para generar datos sintéticos.

**Gated Recurrent Unit (GRU)** — Variante de RNN más simple que LSTM.

**Gaussian Distribution** — Distribución normal. Asumida por muchos modelos estadísticos.

**Gaussian Mixture Model (GMM)** — Modelo generativo que asume datos provienen de mezcla de Gaussianas.

**GBDT / GBM** — Gradient Boosting Decision Tree. Árboles entrenados secuencialmente corrigiendo residuos.

**Generator** — En GANs, red que genera datos sintéticos.

**Gini Impurity** — Medida de impureza usada en árboles de decisión. Menor = mejor split.

**Global Average Pooling** — Pooling que promedia toda la matriz de activaciones.

**Gradient** — Derivada parcial de la función de pérdida respecto a cada peso. Dirección de mayor ascenso.

**Gradient Boosting** — Ensemble que añade árboles secuencialmente para corregir errores. XGBoost, LightGBM, CatBoost.

**Gradient Descent** — Optimización iterativa moviendo pesos en dirección opuesta al gradiente.

**Gradient Tape** — API de TF para registrar operaciones y calcular gradientes automáticamente.

**Grid Search** — Búsqueda exhaustiva sobre espacio de hiperparámetros.

**GRU (Gated Recurrent Unit)** — Variante de RNN con puertas de reset y update.

## H

**Heatmap** — Mapa de color para visualizar matrices (correlación, confusión).

**Hidden Layer** — Capa entre input y output en una red neuronal.

**Hierarchical Clustering** — Agrupación jerárquica: aglomerativa (bottom-up) o divisiva (top-down).

**Hinge Loss** — Función de pérdida usada en SVM. max(0, 1 - y * f(x)).

**Holdout** — Conjunto de datos separado para evaluación final.

**Holt-Winters** — Suavizado exponencial con tendencia y estacionalidad.

**Hyperparameter** — Parámetro del modelo no aprendido durante entrenamiento (learning rate, n_estimators, max_depth). Se optimiza con Grid/Random/Bayesian Search.

**Hyperparameter Tuning** — Proceso de encontrar mejores hiperparámetros.

**Hypothesis Test** — Prueba estadística para evaluar evidencia contra hipótesis nula.

## I

**Imbalance** — Clases con frecuencias muy diferentes. Técnicas: SMOTE, class_weight, oversampling.

**Imputation** — Rellenar valores nulos. Estrategias: mean, median, mode, KNN, forward-fill.

**Incremental Learning** — Entrenamiento continuo con nuevos datos sin reentrenar desde cero.

**Independent Variable** — Variable predictora (feature). Opuesta a dependent variable (target).

**Index** — Etiqueta de filas en pandas. Puede ser numérica, datetime, string.

**Inference** — Fase de predicción del modelo entrenado sobre datos nuevos.

**Information Gain** — Reducción de entropía al hacer split en árbol de decisión.

**Input Shape** — Forma de los datos de entrada (batch_size, features) en redes neuronales.

**Intercept** — Término independiente (β₀) en regresión lineal.

**Inverse Transform** — Revertir transformación (StandardScaler, MinMaxScaler) a escala original.

**IoU (Intersection over Union)** — Métrica para detección de objetos. Área de superposición / área de unión.

**IPS (Inverse Propensity Scoring)** — Técnica de debiasing para recomendación.

**IQR (Interquartile Range)** — Rango intercuartil: Q3 - Q1. Para detectar outliers.

**Iteration** — Una actualización de pesos del modelo (forward + backward con un batch).

## J

**Joint Plot** — Gráfico bivariado en seaborn con histogramas marginales.

**JSON** — Formato de intercambio de datos ligero. Común en APIs.

## K

**K-Means** — Algoritmo de clustering que particiona datos en k clusters basados en centroides.

**K-Nearest Neighbors (KNN)** — Algoritmo supervisado que predice basado en k vecinos más cercanos.

**Kernel** — Función de similitud en SVM (linear, rbf, poly, sigmoid). También: matriz de convolución.

**Kernel Density Estimation (KDE)** — Estimación suavizada de la función de densidad probabilística.

**K-fold Cross Validation** — Particionar datos en k folds, entrenar en k-1, validar en 1.

**KL Divergence** — Medida de diferencia entre dos distribuciones de probabilidad.

**Kurtosis** — Medida de qué tan "picuda" es una distribución.

## L

**L1 Regularization (Lasso)** — Penaliza suma de valores absolutos de pesos. Produce features sparse.

**L2 Regularization (Ridge)** — Penaliza suma de cuadrados de pesos. Reduce todos los pesos.

**Label** — Variable target/objetivo a predecir.

**Label Encoding** — Convertir categorías a enteros (manzana→0, pera→1).

**Lag** — Valor pasado de una serie temporal (lag_1 = valor de ayer).

**Lambda** — Función anónima en Python. También: parámetro de regularización.

**Layer** — Componente de red neuronal: Dense, Conv2D, LSTM, Dropout, BatchNorm.

**Learning Curve** — Gráfico de error en train/val vs cantidad de datos o épocas.

**Learning Rate** — Tamaño del paso en gradient descent. Muy alto: no converge. Muy bajo: lento.

**Lemmatization** — Reducir palabra a su lema (corriendo→correr). Más informativo que stemming.

**LightGBM** — Implementación eficiente de Gradient Boosting. Maneja grandes volúmenes.

**Linear Regression** — Modelo que asume relación lineal entre features y target. y = β₀ + β₁x₁ + ... + βₙxₙ.

**Linguistic Feature** — Features lingüísticas en NLP: POS tags, sentimiento, entidades.

**Linkage** — Criterio de unión en clustering jerárquico (ward, complete, average, single).

**Load Balancer** — Distribuye tráfico entre múltiples instancias de un servicio.

**Log Loss** — Función de pérdida para clasificación binaria. -[y log(p) + (1-y) log(1-p)].

**Logistic Regression** — Regresión para clasificación binaria. Usa sigmoid para producir probabilidades.

**Long Short-Term Memory (LSTM)** — Red recurrente que captura dependencias largas. Ideal para series temporales.

**Loss Function** — Función a minimizar durante entrenamiento. MSE para regresión, Cross-entropy para clasificación.

## M

**MAE (Mean Absolute Error)** — Promedio del error absoluto. |y - ŷ|. Robusto a outliers.

**MAPE (Mean Absolute Percentage Error)** — Error porcentual absoluto medio. |(y-ŷ)/y| * 100.

**Matplotlib** — Librería base de visualización en Python.

**Matrix Factorization** — Descomponer matriz en factores latentes. Base de SVD en recomendación.

**MaxPooling** — Reducción de dimensionalidad tomando máximo en ventana. En CNNs.

**Mean** — Promedio aritmético. Sensible a outliers.

**Median** — Valor central de datos ordenados. Robusto a outliers.

**Memory (GPU)** — Memoria de video utilizada para entrenar modelos. TF permite limitar con `memory_limit`.

**Metric** — Función para evaluar modelo distincta de loss (accuracy, F1, AUC).

**MinMaxScaler** — Escala features al rango [0, 1]. (x - min) / (max - min).

**Mira metric** — Métrica de ranking en recomendación: Mean Reciprocal Rank.

**MLflow** — Plataforma para ciclo de vida ML: tracking, models, registry.

**MLOps** — Prácticas DevOps aplicadas a ML: CI/CD, monitoreo, versionado.

**Mode** — Valor más frecuente en un conjunto de datos.

**Model** — Representación matemática aprendida de los datos. Puede ser árbol, red, regresión, etc.

**Model Registry** — Repositorio de versiones de modelos (MLflow Model Registry).

**Momentum** — Técnica que acelera SGD usando dirección previa del gradiente.

**MSE (Mean Squared Error)** — Promedio del error al cuadrado. Penaliza más errores grandes.

**Multi-label Classification** — Clasificación donde cada instancia puede tener múltiples etiquetas.

**Multicollinearity** — Correlación alta entre predictores. Degrada coeficientes de regresión.

**Mutual Information** — Medida de dependencia entre variables. No paramétrica.

## N

**Naive Bayes** — Clasificador basado en teorema de Bayes asumiendo independencia condicional.

**NaN (Not a Number)** — Valor nulo/ausente en datos.

**Named Entity Recognition (NER)** — Identificar entidades en texto (personas, empresas, fechas).

**Neural Network** — Red de neuronas artificiales con capas ocultas y activaciones no lineales.

**N-gram** — Secuencia de n tokens contiguos en texto. bigram, trigram.

**NLP (Natural Language Processing)** — Procesamiento de lenguaje natural. Análisis de texto, sentimiento, traducción.

**NMF (Non-negative Matrix Factorization)** — Factorización de matrices no negativas. Para topic modeling.

**Noise** — Variabilidad aleatoria en datos no explicada por el modelo.

**Normalization** — Escalar datos a rango [0,1] o media 0 std 1.

**Null Hypothesis** — Hipótesis de no efecto. Se rechaza si p-value < α.

**NumPy** — Librería fundamental para computación numérica en Python. Arrays, álgebra lineal, aleatorios.

## O

**Objective Function** — Función a optimizar (minimizar o maximizar). Sinónimo de loss function.

**One-Hot Encoding** — Convertir categorías a vectores binarios (rojo→[1,0,0]).

**One-vs-Rest (OvR)** — Estrategia para clasificación multiclase: un clasificador por clase.

**ONNX (Open Neural Network Exchange)** — Formato estándar para exportar modelos entre frameworks.

**Optimizer** — Algoritmo de optimización: SGD, Adam, RMSprop.

**Outlier** — Valor extremo que se desvía significativamente del resto. Puede ser error o dato real.

**Overfitting** — Modelo memoriza entrenamiento pero no generaliza. Síntoma: train loss << val loss.

**Oversampling** — Técnica para balancear clases: duplicar instancias de clase minoritaria.

## P

**P-value** — Probabilidad de observar resultado tan extremo si H₀ es cierta. p < 0.05 es significativo.

**Padding** — Añadir bordes (zeros) a entrada de convolución para mantener tamaño.

**Paired Plot** — Matriz de scatter plots para todas las combinaciones de variables.

**Pandas** — Librería para manipulación de datos tabulares. DataFrame, Series, groupby, merge.

**Parameter** — Variable interna del modelo aprendida durante entrenamiento (pesos, bias).

**PCA (Principal Component Analysis)** — Reducción de dimensionalidad encontrando direcciones de máxima varianza.

**Pearson Correlation** — Correlación lineal entre variables continuas. r ∈ [-1, 1].

**Percentile** — Valor debajo del cual cae un porcentaje de datos. Percentil 50 = mediana.

**Pipeline** — Secuencia de transformaciones + modelo final. sklearn Pipeline.

**Precision** — TP / (TP + FP). De las predicciones positivas, ¿cuántas son correctas?

**Pre-training** — Entrenamiento inicial en dataset grande (ImageNet, Wikipedia). Base de transfer learning.

**Prophet** — Librería de forecast de Facebook. Maneja estacionalidad, changepoints, festivos.

**Pruning** — Reducir tamaño del modelo eliminando conexiones/pesos pequeños.

**PyTorch** — Framework de deep learning. Dinámico, flexible, preferido en investigación.

## Q

**Q-Learning** — Algoritmo de reinforcement learning. Aprende tabla Q de acción-estado.

**Quantile** — Puntos que dividen distribución en intervalos con igual probabilidad (cuartiles, percentiles).

**Quantization** — Reducir precisión de pesos (FP32→INT8) para acelerar inferencia en dispositivos edge.

**Query** — Consulta en pandas: `df.query("precio > 100")`.

## R

**R² (Coefficient of Determination)** — Proporción de varianza explicada por el modelo. 1 - SS_res/SS_tot.

**Random Forest** — Ensemble de árboles con Bagging + random subspace. Robusto y popular.

**Random Search** — Búsqueda aleatoria sobre hiperparámetros. Más eficiente que grid search.

**Ranking** — Ordenar items por relevancia. Evaluado con NDCG, MAP, MRR.

**Recall** — TP / (TP + FN). De los positivos reales, ¿cuántos detectamos?

**Rectified Linear Unit (ReLU)** — Activación: max(0, x). No lineal, barata computacionalmente.

**Recurrent Neural Network (RNN)** — Red con conexiones temporales. Para secuencias.

**Reduction (Dimensionality)** — PCA, t-SNE, UMAP, autoencoders.

**Regression** — Predecir variable continua. Ventas en $, temperatura, precio.

**Regularization** — Técnica para reducir overfitting. L1, L2, dropout, early stopping, data augmentation.

**Reinforcement Learning** — Aprendizaje por refuerzo. Agente interactúa con entorno, maximiza recompensa.

**Resampling** — Cambiar frecuencia de serie temporal. Upsampling (interpolar), downsampling (agregar).

**Residual** — Diferencia entre valor real y predicción. y - ŷ.

**ResNet** — Red residual con skip connections. Permite entrenar redes muy profundas.

**Ridge Regression** — Regresión lineal con regularización L2.

**RMSE (Root Mean Squared Error)** — Raíz del MSE. √(Σ(y-ŷ)²/n). En misma unidad que target.

**RobustScaler** — Escala basada en mediana e IQR. Robusto a outliers.

**ROC Curve** — Gráfico de TPR vs FPR a diferentes thresholds.

## S

**SARIMA** — ARIMA con componente estacional. SARIMA(p,d,q)(P,D,Q,s).

**Scaler** — Transformación de escala. StandardScaler, MinMaxScaler, RobustScaler.

**Seasonal Decompose** — Separar serie en tendencia + estacionalidad + residuo.

**Segmentation** — División en grupos homogéneos. Clustering de clientes.

**Self-Attention** — Mecanismo que pondera importancia de cada token respecto a otros. Base de Transformers.

**Sentiment Analysis** — Clasificar texto como positivo, negativo o neutral.

**Sequential Model** — Modelo Keras con capas en secuencia lineal.

**SHAP (SHapley Additive exPlanations)** — Explicabilidad de predicciones basada en teoría de juegos.

**Sigmoid** — Activación que mapea a (0,1). σ(x) = 1/(1+e⁻ˣ). Para clasificación binaria.

**Silhouette Score** — Métrica de clustering: qué tan similar a su cluster vs clusters vecinos.

**Skewness** — Asimetría de distribución. Skew > 0: cola a la derecha.

**SMOTE** — Synthetic Minority Over-sampling Technique. Genera instancias sintéticas de clase minoritaria.

**Softmax** — Función que convierte logits en probabilidades que suman 1. Para clasificación multiclase.

**Sparse Data** — Datos con muchos valores cero/vacíos. Común en recomendación y NLP.

**Spatial Data** — Datos con componente geográfica. Coordenadas, regiones.

**StandardScaler** — Escala a media 0, desviación estándar 1. z = (x - μ) / σ.

**Stationarity** — Serie temporal con media y varianza constantes en el tiempo.

**Stemming** — Reducir palabra a raíz (corriendo→corr). Menos preciso que lemmatization.

**Stochastic Gradient Descent (SGD)** — Gradient descent con un batch aleatorio por iteración.

**Stratification** — Mantener proporción de clases en splits. `train_test_split(stratify=y)`.

**SVD (Singular Value Decomposition)** — Descomposición matricial. Para reducción de dimensionalidad y recomendación.

**SVM (Support Vector Machine)** — Clasificador que encuentra hiperplano con máximo margen entre clases.

## T

**t-SNE (t-Distributed Stochastic Neighbor Embedding)** — Reducción no lineal para visualización de alta dimensión.

**T-test** — Prueba estadística para comparar medias de dos grupos.

**Tanh** — Activación: tanh(x) ∈ (-1, 1). Centrada en cero.

**Target** — Variable a predecir. Dependiente, label, y, output.

**Tensor** — Array multidimensional. Base de TensorFlow y PyTorch. Escalar (0D), vector (1D), matriz (2D), tensor (3D+).

**TensorBoard** — Herramienta de visualización de métricas, gráficos e histogramas de TF.

**TF-IDF (Term Frequency-Inverse Document Frequency)** — Ponderación de términos en texto. Mide importancia relativa.

**Threshold** — Valor corte para convertir probabilidad a clase. Por defecto: 0.5.

**Tokenization** — Dividir texto en unidades (tokens, palabras, subpalabras).

**TPR (True Positive Rate)** — Recall. TP / (TP + FN).

**Train/Test Split** — Dividir datos en conjunto de entrenamiento y prueba. 80/20 típico.

**Transfer Learning** — Usar modelo pre-entrenado en tarea similar y fine-tuning en tarea específica.

**Transformer** — Arquitectura basada en self-attention. GPT, BERT, T5. Revolucionó NLP.

**Tree-based Model** — Modelo basado en árboles de decisión: Decision Tree, Random Forest, GBM.

**TruncatedSVD** — SVD truncado para matrices grandes. Similar a PCA pero eficiente para sparse.

## U

**UDF (User Defined Function)** — Función personalizada aplicada con `apply()` en pandas.

**UMAP (Uniform Manifold Approximation and Projection)** — Reducción de dimensionalidad no lineal. Más rápido que t-SNE.

**Underfitting** — Modelo demasiado simple, no captura patrones. Alto bias.

**Undersampling** — Remover instancias de clase mayoritaria para balancear.

**Unsupervised Learning** — Aprendizaje no supervisado. Sin etiquetas. Clustering, reducción, asociación.

**Upsampling** — Aumentar frecuencia de serie temporal (interpolar). También: aumentar resolución en CNNs.

**User Embedding** — Representación vectorial de preferencias de usuario. Aprendida por MF o DL.

## V

**Validation** — Conjunto de datos para ajustar hiperparámetros y detectar overfitting.

**Validation Curve** — Gráfico de rendimiento vs valor de hiperparámetro.

**Variance** — Medida de dispersión. También: sensibilidad del modelo a pequeñas variaciones en entrenamiento.

**Variational Autoencoder (VAE)** — Autoencoder generativo. Aprende distribución latente.

**Vector** — Array 1D. Representación matemática de puntos en espacio n-dimensional.

**Vectorization** — Convertir texto a vectores. TF-IDF, Word2Vec, BERT embeddings.

**Voting Classifier** — Ensemble que combina predicciones por votación (hard/soft).

## W

**Warm Start** — Inicializar modelo con pesos de entrenamiento previo. Acelera reentrenamiento.

**Weight** — Parámetro aprendible que conecta neuronas en una red.

**Weight Decay** — Término de regularización L2 combinado con optimizer. Sinónimo de L2 regularization.

**Weighted Average** — Promedio ponderado. Cada elemento contribuye con peso diferente.

**Word Embedding** — Representación densa de palabras. Word2Vec, GloVe, FastText.

**Word2Vec** — Modelo para aprender word embeddings. CBOW y Skip-gram.

## X

**XGBoost** — Implementación optimizada de Gradient Boosting. Muy popular en competencias Kaggle.

## Y

**YAML** — Formato de serialización legible. Usado en configuraciones (docker-compose, dvc, kubernetes).

**Y-hat (ŷ)** — Notación para predicción del modelo. Valor estimado.

## Z

**Z-score** — (x - μ) / σ. Número de desviaciones estándar de la media. Para detectar outliers (>3).

**Zero-inflated Model** — Modelo para datos con exceso de ceros. Ej: ventas de productos poco frecuentes.

**Zero-shot Learning** — Clasificar clases no vistas durante entrenamiento. Usando descripciones semánticas.

**Zip** — Función Python para iterar múltiples secuencias en paralelo.

---

## Tabla de Métricas

| Métrica | Fórmula | Interpretación | Uso en Ventas |
|---------|---------|---------------|---------------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | % aciertos total | No usar con desbalance |
| Precision | TP/(TP+FP) | % predicciones positivas correctas | Minimizar falsas ofertas |
| Recall | TP/(TP+FN) | % positivos reales detectados | No perder clientes valiosos |
| F1 | 2*(P*R)/(P+R) | Balance P-R | Métrica estándar |
| RMSE | √(Σ(y-ŷ)²/n) | Error promedio en $ | Forecast de ventas |
| MAE | Σ|y-ŷ|/n | Error absoluto promedio | Forecast robusto |
| R² | 1 - SS_res/SS_tot | Varianza explicada | Bondad de ajuste |
| MAPE | mean(|y-ŷ|/y)*100 | Error porcentual | Comparar modelos |
| AUC | Área bajo ROC | Poder discriminativo | Ranking de predicciones |
