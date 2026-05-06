# AP05 — Cheatsheet Scikit-learn

## 1. Preprocesamiento

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (StandardScaler, MinMaxScaler,
                                   RobustScaler, LabelEncoder,
                                   OneHotEncoder, OrdinalEncoder,
                                   PolynomialFeatures, KBinsDiscretizer)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.pipeline import Pipeline

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Escalado
scaler = StandardScaler()           # media 0, std 1
scaler = MinMaxScaler()             # [0, 1]
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = RobustScaler()             # mediana e IQR (robusto a outliers)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Imputación de valores nulos
imputer = SimpleImputer(strategy="mean")         # mean, median, most_frequent, constant
imputer = SimpleImputer(strategy="constant", fill_value=0)
imputer = KNNImputer(n_neighbors=5)

X_imputed = imputer.fit_transform(X)

# Codificación de categóricas
le = LabelEncoder()
y_encoded = le.fit_transform(y)

ohe = OneHotEncoder(sparse_output=False, drop="first")
X_ohe = ohe.fit_transform(X_categorical)

# Discretización
kbins = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile")
X_binned = kbins.fit_transform(X)

# Features polinomiales
poly = PolynomialFeatures(degree=2, interaction_only=True)
X_poly = poly.fit_transform(X)

# Selección de características
selector = SelectKBest(chi2, k=10)
X_selected = selector.fit_transform(X, y)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*1. Preprocesamiento.*

1. Train/Test split
2. Escalado
3. Imputación de valores nulos
4. Codificación de categóricas
5. Discretización
6. Features polinomiales
7. Selección de características

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 2. Modelos de Clasificación

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier,
                              AdaBoostClassifier,
                              ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Regresión Logística
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)
lr.predict(X_test)
lr.predict_proba(X_test)[:, 1]

# Árbol de Decisión
dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42)
dt.fit(X_train, y_train)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=10,
                            min_samples_leaf=5, random_state=42,
                            n_jobs=-1)
rf.fit(X_train, y_train)
rf.feature_importances_          # importancia de features

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                max_depth=3, random_state=42)
gb.fit(X_train, y_train)

# SVM
svm = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)
svm.fit(X_train, y_train)

# KNN
knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
knn.fit(X_train, y_train)
```

**Salida:**

```
# El modelo ajusta sus parámetros durante el entrenamiento.
# Los parámetros aprendidos quedan disponibles en el objeto.
```

**Explicación línea por línea:**

*2. Modelos de Clasificación.*

1. Regresión Logística
2. Árbol de Decisión
3. Random Forest
4. Gradient Boosting
5. SVM
6. KNN

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 3. Modelos de Regresión

```python
from sklearn.linear_model import (LinearRegression, Ridge, Lasso,
                                  ElasticNet, SGDRegressor)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor,
                              GradientBoostingRegressor,
                              AdaBoostRegressor)
from sklearn.svm import SVR

# Regresión Lineal
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
lin_reg.coef_                     # coeficientes
lin_reg.intercept_                # intercepto

# Ridge (L2)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Lasso (L1)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# ElasticNet (L1 + L2)
en = ElasticNet(alpha=0.1, l1_ratio=0.5)
en.fit(X_train, y_train)

# Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)

# Gradient Boosting Regressor
gb_reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                   max_depth=3, random_state=42)
gb_reg.fit(X_train, y_train)
```

**Salida:**

```
# El modelo ajusta sus parámetros durante el entrenamiento.
# Los parámetros aprendidos quedan disponibles en el objeto.
```

**Explicación línea por línea:**

*3. Modelos de Regresión.*

1. Regresión Lineal
2. Ridge (L2)
3. Lasso (L1)
4. ElasticNet (L1 + L2)
5. Random Forest Regressor
6. Gradient Boosting Regressor

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 4. Clustering

```python
from sklearn.cluster import (KMeans, DBSCAN, AgglomerativeClustering,
                             MeanShift, SpectralClustering)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# K-Means
kmeans = KMeans(n_clusters=5, random_state=42, n_init="auto")
kmeans.fit(X)
kmeans.labels_                    # clusters asignados
kmeans.cluster_centers_           # centroides
kmeans.inertia_                   # suma de distancias al cuadrado

# Elegir k con el método del codo
inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(X)
dbscan.labels_                    # -1 = ruido

# Clustering jerárquico
agg = AgglomerativeClustering(n_clusters=5, linkage="ward")
agg.fit(X)

# Métricas de clustering
silhouette_score(X, labels)
calinski_harabasz_score(X, labels)

# Gaussian Mixture
gmm = GaussianMixture(n_components=5, random_state=42)
gmm.fit(X)
gmm.predict_proba(X)              # probabilidades de pertenencia
```

**Salida:**

```
# El modelo ajusta sus parámetros durante el entrenamiento.
# Los parámetros aprendidos quedan disponibles en el objeto.
```

**Explicación línea por línea:**

*4. Clustering.*

1. K-Means
2. Elegir k con el método del codo
3. DBSCAN
4. Clustering jerárquico
5. Métricas de clustering
6. Gaussian Mixture

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 5. Reducción de Dimensionalidad

```python
from sklearn.decomposition import PCA, TruncatedSVD, NMF, FactorAnalysis
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
pca.explained_variance_ratio_     # varianza explicada por componente
pca.components_                    # cargas (loadings)
pca.n_components_                  # número de componentes

# Varianza acumulada
pca = PCA(n_components=0.95)      # retener 95% de varianza
X_pca = pca.fit_transform(X)

# t-SNE (visualización)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X)

# TruncatedSVD (para matrices sparse)
svd = TruncatedSVD(n_components=50)
X_svd = svd.fit_transform(X_tfidf)

# NMF (para datos no negativos)
nmf = NMF(n_components=10)
X_nmf = nmf.fit_transform(X)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*5. Reducción de Dimensionalidad.*

1. PCA
2. Varianza acumulada
3. t-SNE (visualización)
4. TruncatedSVD (para matrices sparse)
5. NMF (para datos no negativos)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 6. Métricas

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, mean_squared_error,
                             r2_score, mean_absolute_error, log_loss,
                             silhouette_score)

# Clasificación
accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred, average="weighted")
recall_score(y_test, y_pred, average="macro")
f1_score(y_test, y_pred, average="micro")
roc_auc_score(y_test, y_proba)

# Reporte completo
print(classification_report(y_test, y_pred,
      target_names=["No compra", "Compra"]))

# Matriz de confusión
confusion_matrix(y_test, y_pred)

# Regresión
mean_squared_error(y_test, y_pred)     # MSE
mean_absolute_error(y_test, y_pred)    # MAE
r2_score(y_test, y_pred)               # R²
np.sqrt(mean_squared_error(y_test, y_pred))  # RMSE

# Log Loss
log_loss(y_test, y_proba)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*6. Métricas.*

1. Clasificación
2. Reporte completo
3. Matriz de confusión
4. Regresión
5. Log Loss

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 7. Validación Cruzada

```python
from sklearn.model_selection import (cross_val_score, cross_validate,
                                     KFold, StratifiedKFold,
                                     RepeatedKFold, LeaveOneOut)

# Cross-validation simple
scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print(f"{scores.mean():.3f} ± {scores.std():.3f}")

# Con más métricas
cv_results = cross_validate(model, X, y, cv=5,
                            scoring=["accuracy", "f1", "roc_auc"],
                            return_train_score=True)

# K-Fold personalizado
kf = KFold(n_splits=5, shuffle=True, random_state=42)
strat_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in kf.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

# Repeated K-Fold
rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*7. Validación Cruzada.*

1. Cross-validation simple
2. Con más métricas
3. K-Fold personalizado
4. Repeated K-Fold

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 8. Grid Search y Optimización

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint, uniform

# Grid Search
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, None],
    "min_samples_leaf": [1, 5, 10]
}
grid = GridSearchCV(RandomForestClassifier(random_state=42),
                    param_grid, cv=5, scoring="f1",
                    n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)
print(grid.best_estimator_)

# Randomized Search
param_dist = {
    "n_estimators": randint(50, 500),
    "max_depth": randint(3, 20),
    "learning_rate": uniform(0.01, 0.3)
}
random_search = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_dist, n_iter=50, cv=5, scoring="roc_auc",
    random_state=42, n_jobs=-1
)
random_search.fit(X_train, y_train)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*8. Grid Search y Optimización.*

1. Grid Search
2. Randomized Search

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 9. Pipelines

```python
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer

# Pipeline simple
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(random_state=42))
])
pipe.fit(X_train, y_train)
pipe.predict(X_test)

# make_pipeline (nombres automáticos)
pipe = make_pipeline(StandardScaler(), LogisticRegression())

# Pipeline con preprocesamiento mixto
numeric_features = ["edad", "ingreso"]
categorical_features = ["categoria", "region"]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(random_state=42))
])

pipe.fit(X_train, y_train)

# Pipeline con GridSearch
param_grid = {
    "clf__n_estimators": [50, 100],
    "clf__max_depth": [5, 10]
}
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
```

**Salida:**

```
# El modelo ajusta sus parámetros durante el entrenamiento.
# Los parámetros aprendidos quedan disponibles en el objeto.
```

**Explicación línea por línea:**

*9. Pipelines.*

1. Pipeline simple
2. make_pipeline (nombres automáticos)
3. Pipeline con preprocesamiento mixto
4. Pipeline con GridSearch

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 10. Feature Selection

```python
from sklearn.feature_selection import (SelectKBest, SelectPercentile,
                                       RFE, RFECV, SelectFromModel,
                                       mutual_info_classif, chi2,
                                       f_classif)

# SelectKBest
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)
selector.scores_                   # scores de cada feature

# RFE (Recursive Feature Elimination)
rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=10)
rfe.fit(X, y)
rfe.support_                       # máscara de features seleccionados
rfe.ranking_                       # ranking de features

# RFE con CV
rfecv = RFECV(estimator=RandomForestClassifier(), cv=5)
rfecv.fit(X, y)
print(f"Optimal features: {rfecv.n_features_}")

# SelectFromModel (basado en importancia)
selector = SelectFromModel(RandomForestClassifier(), threshold="median")
X_selected = selector.fit_transform(X, y)

# Mutual Information
mi = mutual_info_classif(X, y, random_state=42)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*10. Feature Selection.*

1. SelectKBest
2. RFE (Recursive Feature Elimination)
3. RFE con CV
4. SelectFromModel (basado en importancia)
5. Mutual Information

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 11. Curvas de Aprendizaje y Validación

```python
from sklearn.model_selection import learning_curve, validation_curve

# Curva de aprendizaje
train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
)

# Curva de validación (efecto de un hiperparámetro)
param_range = [0.01, 0.1, 1, 10, 100]
train_scores, val_scores = validation_curve(
    model, X, y, param_name="C", param_range=param_range,
    cv=5, scoring="accuracy", n_jobs=-1
)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*11. Curvas de Aprendizaje y Validación.*

1. Curva de aprendizaje
2. Curva de validación (efecto de un hiperparámetro)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 12. Model Persistence (Pickle/Joblib)

```python
import joblib

# Guardar modelo
joblib.dump(grid.best_estimator_, "modelo_ventas.pkl")

# Cargar modelo
model = joblib.load("modelo_ventas.pkl")
predicciones = model.predict(X_new)

# Guardar pipeline completo
joblib.dump(pipe, "pipeline_ventas.pkl")
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*12. Model Persistence (Pickle/Joblib).*

1. Guardar modelo
2. Cargar modelo
3. Guardar pipeline completo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 13. Ensambles Avanzados

```python
from sklearn.ensemble import VotingClassifier, VotingRegressor, StackingClassifier
from sklearn.linear_model import LogisticRegression

# Voting (votación)
voting = VotingClassifier([
    ("lr", LogisticRegression()),
    ("rf", RandomForestClassifier(n_estimators=50)),
    ("gb", GradientBoostingClassifier(n_estimators=50))
], voting="soft")  # soft = promedios de probabilidades
voting.fit(X_train, y_train)

# Stacking
stack = StackingClassifier([
    ("rf", RandomForestClassifier(n_estimators=50)),
    ("svm", SVC(probability=True)),
    ("knn", KNeighborsClassifier())
], final_estimator=LogisticRegression(), cv=5)
stack.fit(X_train, y_train)
```

**Salida:**

```
# El modelo ajusta sus parámetros durante el entrenamiento.
# Los parámetros aprendidos quedan disponibles en el objeto.
```

**Explicación línea por línea:**

*13. Ensambles Avanzados.*

1. Voting (votación)
2. Stacking

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 14. Calibration

```python
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# Calibrar probabilidades
calibrated = CalibratedClassifierCV(model, method="sigmoid", cv=5)
calibrated.fit(X_train, y_train)
calibrated.predict_proba(X_test)

# Curva de calibración
prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
```

**Salida:**

```
# El modelo ajusta sus parámetros durante el entrenamiento.
# Los parámetros aprendidos quedan disponibles en el objeto.
```

**Explicación línea por línea:**

*14. Calibration.*

1. Calibrar probabilidades
2. Curva de calibración

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## 15. Métricas Personalizadas

```python
from sklearn.metrics import make_scorer

# Crear scorer personalizado
def revenue_score(y_true, y_pred):
    # Beneficio: ventas acertadas - costo de falsos positivos
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return (tp * 100) - (fp * 10) - (fn * 50)

revenue_scorer = make_scorer(revenue_score, greater_is_better=True)

grid = GridSearchCV(model, param_grid, scoring=revenue_scorer, cv=5)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*15. Métricas Personalizadas.*

1. Crear scorer personalizado
2. Beneficio: ventas acertadas - costo de falsos positivos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



## Referencia Rápida

| Categoría | Clase/Función | Descripción |
|-----------|---------------|-------------|
| Preprocesamiento | `StandardScaler`, `MinMaxScaler`, `OneHotEncoder` | Escalar/codificar |
| Imputación | `SimpleImputer`, `KNNImputer` | Valores nulos |
| Clasificación | `LogisticRegression`, `RandomForestClassifier`, `SVC` | Modelos supervisados |
| Regresión | `LinearRegression`, `Ridge`, `Lasso`, `RandomForestRegressor` | Regresión |
| Clustering | `KMeans`, `DBSCAN`, `AgglomerativeClustering` | No supervisado |
| Reducción | `PCA`, `TSNE`, `TruncatedSVD` | Dimensionalidad |
| Selección | `SelectKBest`, `RFE`, `SelectFromModel` | Feature selection |
| Validación | `cross_val_score`, `GridSearchCV`, `RandomizedSearchCV` | Evaluación |
| Pipeline | `Pipeline`, `ColumnTransformer`, `make_pipeline` | Workflows |
| Métricas | `accuracy_score`, `f1_score`, `roc_auc_score`, `r2_score` | Evaluación |
| Persistencia | `joblib.dump`, `joblib.load` | Guardar/cargar modelos |
