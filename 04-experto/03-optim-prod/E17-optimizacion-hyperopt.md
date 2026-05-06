# E17: Optimización con Hyperopt — Búsqueda Bayesiana de Hiperparámetros

## Objetivo
Dominar Hyperopt para optimizar hiperparámetros de modelos ML en ventas/compras/inventarios mediante búsqueda bayesiana (TPE), búsqueda aleatoria y simulated annealing.

---

## 1. Fundamentos Teóricos

### 1.1 ¿Qué es Hyperopt?
Hyperopt es una biblioteca para optimización de hiperparámetros que implementa:
- **TPE (Tree-structured Parzen Estimator)**: Modela P(score|params) y P(params) con estimadores de Parzen. Selecciona candidatos maximizando P(params|score < γ).
- **Random Search**: Muestreo uniforme del espacio de búsqueda.
- **Anneal**: Simulated annealing con temperatura decreciente.

### 1.2 TPE Algorithm
TPE construye dos densidades:
- l(x) = P(x | loss < γ*) — parámetros que dan buen resultado
- g(x) = P(x | loss ≥ γ*) — parámetros que dan mal resultado

Donde γ* es un cuantil (por defecto 15%). El siguiente candidato maximiza l(x)/g(x).

### 1.3 Espacio de Búsqueda
- **hp.choice(label, options)**: Selección categórica
- **hp.uniform(label, low, high)**: Uniforme continuo
- **hp.quniform(label, low, high, q)**: Uniforme discretizado
- **hp.loguniform(label, low, high)**: Log-uniforme (para escalas logarítmicas)
- **hp.randint(label, low, high)**: Entero uniforme
- **hp.normal(label, mu, sigma)**: Normal
- **hp.lognormal(label, mu, sigma)**: Log-normal
- **hp.pchoice(label, p_options)**: Categórica con probabilidades

### 1.4 fmin
```python
fmin(fn, space, algo=tpe.suggest, max_evals=100, trials=None,
     rstate=np.random.default_rng(), verbose=True, return_argmin=True,
     points_to_evaluate=None, directions=None)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*1.4 fmin.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---


- **fn**: función objetivo a minimizar
- **space**: espacio de búsqueda
- **algo**: algoritmo de optimización
- **max_evals**: número máximo de evaluaciones
- **trials**: objeto Trials para almacenar historial
- **points_to_evaluate**: puntos iniciales conocidos
- **directions**: multi-objetivo (None, o lista de 'minimize'/'maximize')

### 1.5 Trials
```python
trials.trials       # lista de diccionarios con resultados
trials.results      # lista de resultados {'loss': ..., 'status': ...}
trials.losses()     # lista de pérdidas
trials.statuses()   # lista de estados ('ok', 'fail', 'new')
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*1.5 Trials.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### 1.6 Early Stopping
Hiperopt no tiene early stopping nativo, pero se puede implementar:
- Verificar mejora en últimas N iteraciones
- Detener si no hay mejora absoluta/relativa
- Usar `trials.losses()` para monitorear convergencia

### 1.7 SparkTrials
Para ejecución paralela en clúster Spark:
```python
from hyperopt import SparkTrials
spark_trials = SparkTrials(parallelism=4)
fmin(fn, space, algo=tpe.suggest, max_evals=100, trials=spark_trials)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*1.7 SparkTrials.*

1. `from hyperopt import SparkTrials` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### 1.8 Parallel Evaluations con MongoDB
```python
from hyperopt import MongoTrials
trials = MongoTrials('mongo://host:port/db/jobs', exp_key='exp1')
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*1.8 Parallel Evaluations con MongoDB.*

1. `from hyperopt import MongoTrials` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 2. Ejemplos Prácticos

### Ejemplo 1: Minimizar función de pérdida de modelo (max_depth, n_estimators)

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

# Datos sintéticos de ventas
np.random.seed(42)
n = 1000
ventas = pd.DataFrame({
    'precio': np.random.uniform(10, 100, n),
    'descuento': np.random.uniform(0, 0.3, n),
    'inventario': np.random.randint(0, 500, n),
    'dia_semana': np.random.randint(0, 7, n),
    'demanda': np.random.poisson(50, n)
})
X = ventas[['precio', 'descuento', 'inventario', 'dia_semana']]
y = ventas['demanda']

def objetivo(params):
    rf = RandomForestRegressor(
        max_depth=int(params['max_depth']),
        n_estimators=int(params['n_estimators']),
        random_state=42
    )
    mse = -cross_val_score(rf, X, y, cv=3, scoring='neg_mean_squared_error').mean()
    return {'loss': mse, 'status': STATUS_OK}

space = {
    'max_depth': hp.quniform('max_depth', 2, 20, 1),
    'n_estimators': hp.quniform('n_estimators', 50, 500, 10)
}

trials = Trials()
best = fmin(fn=objetivo, space=space, algo=tpe.suggest,
            max_evals=50, trials=trials, verbose=True)
print(f"Mejores params: {best}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: Minimizar función de pérdida de modelo (max_depth, n_estimators).*

1. Datos sintéticos de ventas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: hp.choice — Categorías de optimización

```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

def objetivo_choice(params):
    modelo_cls = params['modelo']
    if modelo_cls == 'rf':
        modelo = RandomForestRegressor(
            n_estimators=int(params['n_estimators']), random_state=42)
    elif modelo_cls == 'gbm':
        modelo = GradientBoostingRegressor(
            n_estimators=int(params['n_estimators']), learning_rate=params['lr'],
            random_state=42)
    else:
        modelo = Ridge(alpha=params['alpha'])
    mse = -cross_val_score(modelo, X, y, cv=3, scoring='neg_mean_squared_error').mean()
    return {'loss': mse, 'status': STATUS_OK}

space_choice = hp.choice('modelo', [
    {'modelo': 'rf', 'n_estimators': hp.quniform('n_estimators_rf', 50, 300, 10)},
    {'modelo': 'gbm', 'n_estimators': hp.quniform('n_estimators_gbm', 50, 300, 10),
     'lr': hp.uniform('lr', 0.01, 0.3)},
    {'modelo': 'ridge', 'alpha': hp.uniform('alpha', 0.1, 10)}
])

best_choice = fmin(fn=objetivo_choice, space=space_choice, algo=tpe.suggest, max_evals=30)
print(f"Mejor modelo: {best_choice}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: hp.choice — Categorías de optimización.*

1. `from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor` — Importa las librerías necesarias para el análisis.
2. `from sklearn.linear_model import Ridge` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: hp.uniform — Parámetros continuos (learning_rate 0.001–0.1)

```python
from sklearn.ensemble import GradientBoostingRegressor

def objetivo_lr(params):
    gbm = GradientBoostingRegressor(
        learning_rate=params['lr'],
        n_estimators=200, max_depth=5, random_state=42
    )
    mse = -cross_val_score(gbm, X, y, cv=3, scoring='neg_mean_squared_error').mean()
    return {'loss': mse, 'status': STATUS_OK}

space_lr = {'lr': hp.uniform('lr', 0.001, 0.1)}
best_lr = fmin(fn=objetivo_lr, space=space_lr, algo=tpe.suggest, max_evals=20)
print(f"Mejor learning_rate: {best_lr['lr']:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: hp.uniform — Parámetros continuos (learning_rate 0.001–0.1).*

1. `from sklearn.ensemble import GradientBoostingRegressor` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: hp.loguniform — Parámetros en escala logarítmica (reg_lambda)

```python
import xgboost as xgb

def objetivo_reg(params):
    model = xgb.XGBRegressor(
        n_estimators=100,
        reg_lambda=params['lambda'],
        reg_alpha=params['alpha'],
        random_state=42
    )
    mse = -cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error').mean()
    return {'loss': mse, 'status': STATUS_OK}

space_reg = {
    'lambda': hp.loguniform('lambda', np.log(0.001), np.log(10)),
    'alpha': hp.loguniform('alpha', np.log(0.001), np.log(10))
}
best_reg = fmin(fn=objetivo_reg, space=space_reg, algo=tpe.suggest, max_evals=30)
print(f"Mejor lambda: {best_reg['lambda']:.4f}, alpha: {best_reg['alpha']:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: hp.loguniform — Parámetros en escala logarítmica (reg_lambda).*

1. `import xgboost as xgb` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: hp.quniform — Parámetros discretos (max_depth 2–20)

```python
def objetivo_depth(params):
    rf = RandomForestRegressor(
        max_depth=int(params['max_depth']),
        min_samples_split=int(params['min_samples_split']),
        random_state=42
    )
    mse = -cross_val_score(rf, X, y, cv=3, scoring='neg_mean_squared_error').mean()
    return {'loss': mse, 'status': STATUS_OK}

space_depth = {
    'max_depth': hp.quniform('max_depth', 2, 20, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 50, 1)
}
best_depth = fmin(fn=objetivo_depth, space=space_depth, algo=tpe.suggest, max_evals=30)
print(f"Mejor max_depth: {int(best_depth['max_depth'])}, min_samples_split: {int(best_depth['min_samples_split'])}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: hp.quniform — Parámetros discretos (max_depth 2–20).*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: fmin con algo=tpe.suggest (Tree of Parzen Estimators)

```python
def objetivo_dummy(params):
    x = params['x']
    y = params['y']
    return {'loss': (x - 3)**2 + (y + 2)**2 + 1, 'status': STATUS_OK}

space_dummy = {
    'x': hp.uniform('x', -10, 10),
    'y': hp.uniform('y', -10, 10)
}

trials_tpe = Trials()
best_tpe = fmin(fn=objetivo_dummy, space=space_dummy, algo=tpe.suggest,
                max_evals=50, trials=trials_tpe)
print(f"TPE: x={best_tpe['x']:.3f}, y={best_tpe['y']:.3f}")
print(f"Mejor loss: {trials_tpe.best_trial['result']['loss']:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: fmin con algo=tpe.suggest (Tree of Parzen Estimators).*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: fmin con algo=rand.suggest (búsqueda aleatoria)

```python
from hyperopt import rand

best_random = fmin(fn=objetivo_dummy, space=space_dummy, algo=rand.suggest,
                   max_evals=50)
print(f"Random: x={best_random['x']:.3f}, y={best_random['y']:.3f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: fmin con algo=rand.suggest (búsqueda aleatoria).*

1. `from hyperopt import rand` — Importa las librerías necesarias para el análisis.
2. `print(f"Random: x={best_random['x']:.3f}, y={best_random['y']:.3f}")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: Trials — Historial completo de evaluaciones

```python
trials_full = Trials()
_ = fmin(fn=objetivo_dummy, space=space_dummy, algo=tpe.suggest,
         max_evals=20, trials=trials_full)

print(f"Total trials: {len(trials_full.trials)}")
print(f"Losses: {trials_full.losses()}")
print(f"Best trial: {trials_full.best_trial}")
print("Params de cada trial:")
for i, t in enumerate(trials_full.trials):
    vals = t['misc']['vals']
    vals_clean = {k: v[0] if isinstance(v, list) and len(v)==1 else v
                  for k, v in vals.items()}
    loss = t['result']['loss']
    print(f"  Trial {i+1}: {vals_clean} -> loss={loss:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: Trials — Historial completo de evaluaciones.*

1. `print(f"Total trials: {len(trials_full.trials)}")` — Muestra el resultado por pantalla.
2. `print(f"Losses: {trials_full.losses()}")` — Muestra el resultado por pantalla.
3. `print(f"Best trial: {trials_full.best_trial}")` — Muestra el resultado por pantalla.
4. `print("Params de cada trial:")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: Visualizar trials — pérdida vs hiperparámetros

```python
import matplotlib.pyplot as plt

losses = trials_full.losses()
params_x = [t['misc']['vals']['x'][0] for t in trials_full.trials]
params_y = [t['misc']['vals']['y'][0] for t in trials_full.trials]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].scatter(range(len(losses)), losses)
axes[0].set_xlabel('Trial'); axes[0].set_ylabel('Loss'); axes[0].set_title('Loss por trial')

axes[1].scatter(params_x, losses)
axes[1].set_xlabel('x'); axes[1].set_ylabel('Loss'); axes[1].set_title('Loss vs x')

axes[2].scatter(params_y, losses)
axes[2].set_xlabel('y'); axes[2].set_ylabel('Loss'); axes[2].set_title('Loss vs y')
plt.tight_layout(); plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Ejemplo 9: Visualizar trials — pérdida vs hiperparámetros.*

1. `import matplotlib.pyplot as plt` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: Optimizar XGBoost con hyperopt

```python
def objetivo_xgb(params):
    model = xgb.XGBRegressor(
        max_depth=int(params['max_depth']),
        learning_rate=params['learning_rate'],
        n_estimators=int(params['n_estimators']),
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        reg_lambda=params['reg_lambda'],
        random_state=42
    )
    mse = -cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error').mean()
    return {'loss': mse, 'status': STATUS_OK}

space_xgb = {
    'max_depth': hp.quniform('max_depth', 3, 12, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
    'n_estimators': hp.quniform('n_estimators', 50, 300, 10),
    'subsample': hp.uniform('subsample', 0.5, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
    'reg_lambda': hp.loguniform('reg_lambda', np.log(0.1), np.log(10))
}

trials_xgb = Trials()
best_xgb = fmin(fn=objetivo_xgb, space=space_xgb, algo=tpe.suggest,
                max_evals=50, trials=trials_xgb)
print(f"Mejores params XGBoost: {best_xgb}")
print(f"Mejor loss: {trials_xgb.best_trial['result']['loss']:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: Optimizar XGBoost con hyperopt.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: Optimizar LSTM con hyperopt (units, dropout, lr)

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def objetivo_lstm(params):
    model = Sequential([
        LSTM(int(params['units']), input_shape=(10, 1), return_sequences=False),
        Dropout(params['dropout']),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['lr']),
                  loss='mse')
    X_seq = np.random.randn(200, 10, 1)
    y_seq = np.random.randn(200)
    history = model.fit(X_seq, y_seq, epochs=10, verbose=0, validation_split=0.2)
    val_loss = min(history.history['val_loss'])
    tf.keras.backend.clear_session()
    return {'loss': val_loss, 'status': STATUS_OK}

space_lstm = {
    'units': hp.quniform('units', 16, 128, 16),
    'dropout': hp.uniform('dropout', 0.1, 0.5),
    'lr': hp.loguniform('lr', np.log(0.0001), np.log(0.01))
}

trials_lstm = Trials()
best_lstm = fmin(fn=objetivo_lstm, space=space_lstm, algo=tpe.suggest,
                 max_evals=15, trials=trials_lstm)
print(f"Mejores params LSTM: {best_lstm}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: Optimizar LSTM con hyperopt (units, dropout, lr).*

1. `import tensorflow as tf` — Importa las librerías necesarias para el análisis.
2. `from tensorflow.keras.models import Sequential` — Importa las librerías necesarias para el análisis.
3. `from tensorflow.keras.layers import LSTM, Dense, Dropout` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: Early stopping en hyperopt — max_evals con mejora mínima

```python
def objetivo_early(params):
    rf = RandomForestRegressor(
        max_depth=int(params['max_depth']),
        n_estimators=int(params['n_estimators']),
        random_state=42
    )
    mse = -cross_val_score(rf, X, y, cv=2, scoring='neg_mean_squared_error').mean()
    return {'loss': mse, 'status': STATUS_OK}

trials_early = Trials()
patience = 5
best_loss_so_far = float('inf')
no_improve = 0

for i in range(100):
    best = fmin(fn=objetivo_early, space={'max_depth': hp.quniform('md', 2, 20, 1),
                                          'n_estimators': hp.quniform('ne', 50, 300, 10)},
                algo=tpe.suggest, max_evals=i+1, trials=trials_early, verbose=False)
    current_best = trials_early.best_trial['result']['loss']
    if current_best < best_loss_so_far - 0.001:
        best_loss_so_far = current_best
        no_improve = 0
    else:
        no_improve += 1
    if no_improve >= patience:
        print(f"Early stopping en trial {i+1}, best_loss={best_loss_so_far:.4f}")
        break
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: Early stopping en hyperopt — max_evals con mejora mínima.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: SparkTrials para GPU cluster

```python
# Simulación de SparkTrials (requiere Spark instalado)
try:
    from hyperopt import SparkTrials
    spark_trials = SparkTrials(parallelism=4)
    best_spark = fmin(fn=objetivo_dummy, space=space_dummy, algo=tpe.suggest,
                      max_evals=20, trials=spark_trials)
    print(f"SparkTrials best: {best_spark}")
except ImportError:
    print("SparkTrials no disponible (requiere pyspark)")
    # Alternativa: simular paralelismo con trials ordinarios
    print("Usando Trials secuencial como fallback")
    t = Trials()
    best_s = fmin(fn=objetivo_dummy, space=space_dummy, algo=tpe.suggest,
                  max_evals=20, trials=t)
    print(f"Best: {best_s}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: SparkTrials para GPU cluster.*

1. Simulación de SparkTrials (requiere Spark instalado)
2. Alternativa: simular paralelismo con trials ordinarios

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: points_to_evaluate — Puntos iniciales conocidos

```python
# Puntos conocidos de experimentos anteriores
known_points = [
    {'x': 0.0, 'y': 0.0},
    {'x': 3.0, 'y': -2.0},
    {'x': -1.0, 'y': 1.0}
]

best_with_prior = fmin(fn=objetivo_dummy, space=space_dummy, algo=tpe.suggest,
                       max_evals=20, points_to_evaluate=known_points)
print(f"Con prior knowledge: {best_with_prior}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: points_to_evaluate — Puntos iniciales conocidos.*

1. Puntos conocidos de experimentos anteriores

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: directions — Minimizar pérdida, maximizar precisión (multi-objetivo)

```python
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X_clf, y_clf = make_classification(n_samples=500, n_features=10, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X_clf, y_clf, test_size=0.2)

def objetivo_multi(params):
    rf = RandomForestRegressor(
        max_depth=int(params['max_depth']),
        n_estimators=int(params['n_estimators']),
        random_state=42
    )
    rf.fit(X_tr, y_tr)
    preds = rf.predict(X_te)
    mse = ((preds - y_te) ** 2).mean()
    # Minimizar loss = mse
    return {'loss': mse, 'status': STATUS_OK}

# hyperopt natively only minimizes; multi-objective se simula combinando métricas
def objetivo_combinado(params):
    rf = RandomForestRegressor(
        max_depth=int(params['max_depth']),
        n_estimators=int(params['n_estimators']),
        random_state=42
    )
    rf.fit(X_tr, y_tr)
    preds = rf.predict(X_te)
    mse = ((preds - y_te) ** 2).mean()
    num_params = int(params['max_depth']) * int(params['n_estimators'])
    # Minimizar MSE y cantidad de parámetros
    return {'loss': mse + 1e-6 * num_params, 'status': STATUS_OK}

space_multi = {
    'max_depth': hp.quniform('max_depth', 2, 15, 1),
    'n_estimators': hp.quniform('n_estimators', 50, 200, 10)
}
best_multi = fmin(fn=objetivo_combinado, space=space_multi, algo=tpe.suggest, max_evals=30)
print(f"Multi-objetivo: {best_multi}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: directions — Minimizar pérdida, maximizar precisión (multi-objetivo).*

1. Minimizar loss = mse
2. hyperopt natively only minimizes; multi-objective se simula combinando métricas
3. Minimizar MSE y cantidad de parámetros

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 16: Comparar TPE vs random vs anneal en convergencia

```python
from hyperopt import anneal
import time

algoritmos = {
    'TPE': tpe.suggest,
    'Random': rand.suggest,
    'Anneal': anneal.suggest
}

resultados = {}
for name, algo in algoritmos.items():
    t = Trials()
    start = time.time()
    best = fmin(fn=objetivo_dummy, space=space_dummy, algo=algo,
                max_evals=30, trials=t, verbose=False)
    elapsed = time.time() - start
    losses = t.losses()
    resultados[name] = {
        'best': best,
        'best_loss': min(losses),
        'elapsed': elapsed,
        'losses': losses
    }
    print(f"{name:8s} -> loss={min(losses):.4f}, tiempo={elapsed:.2f}s")

# Visualizar convergencia
plt.figure(figsize=(10, 5))
for name, res in resultados.items():
    plt.plot(res['losses'], label=f"{name} (best={res['best_loss']:.3f})")
plt.xlabel('Trial'); plt.ylabel('Loss'); plt.legend()
plt.title('Comparación de algoritmos de optimización')
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: Comparar TPE vs random vs anneal en convergencia.*

1. Visualizar convergencia

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 17: Best params y best loss

```python
trials_best = Trials()
best_final = fmin(fn=objetivo_dummy, space=space_dummy, algo=tpe.suggest,
                  max_evals=50, trials=trials_best)

print("="*40)
print("MEJORES PARÁMETROS ENCONTRADOS")
print("="*40)
print(f"x* = {best_final['x']:.4f}")
print(f"y* = {best_final['y']:.4f}")
print(f"Loss* = {trials_best.best_trial['result']['loss']:.4f}")
print(f"Mínimo teórico: en x=3, y=-2, loss=1.0")
print(f"Error: {abs(trials_best.best_trial['result']['loss'] - 1.0):.4f}")

# Mostrar top 5 trials
losses = trials_best.losses()
top5 = sorted(enumerate(losses), key=lambda x: x[1])[:5]
print("\nTop 5 trials:")
for idx, loss in top5:
    t = trials_best.trials[idx]
    vals = {k: v[0] for k, v in t['misc']['vals'].items()}
    print(f"  Trial {idx+1}: {vals} -> loss={loss:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: Best params y best loss.*

1. Mostrar top 5 trials

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 18: Integrador — Optimizar pipeline ML completo

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def objetivo_pipeline(params):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=int(params['pca_components']))),
        ('modelo', RandomForestRegressor(
            max_depth=int(params['max_depth']),
            n_estimators=int(params['n_estimators']),
            min_samples_split=int(params['min_samples_split']),
            random_state=42
        ))
    ])
    mse = -cross_val_score(pipeline, X, y, cv=3, scoring='neg_mean_squared_error').mean()
    return {'loss': mse, 'status': STATUS_OK}

space_pipeline = {
    'pca_components': hp.quniform('pca_components', 1, 4, 1),
    'max_depth': hp.quniform('max_depth', 2, 20, 1),
    'n_estimators': hp.quniform('n_estimators', 50, 300, 10),
    'min_samples_split': hp.quniform('min_samples_split', 2, 20, 1)
}

trials_pipeline = Trials()
best_pipe = fmin(fn=objetivo_pipeline, space=space_pipeline, algo=tpe.suggest,
                 max_evals=40, trials=trials_pipeline)
print(f"Mejor pipeline: {best_pipe}")
print(f"Mejor loss: {trials_pipeline.best_trial['result']['loss']:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador — Optimizar pipeline ML completo.*

1. `from sklearn.pipeline import Pipeline` — Importa las librerías necesarias para el análisis.
2. `from sklearn.preprocessing import StandardScaler` — Importa las librerías necesarias para el análisis.
3. `from sklearn.decomposition import PCA` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 3. Ejercicios Propuestos

1. **Optimizar GradientBoosting para demanda**: Usando hyperopt, encuentra los mejores learning_rate, n_estimators, max_depth y min_samples_leaf para un GBM que predice demanda de 50 productos. Define un espacio de búsqueda con hp.uniform, hp.quniform y hp.choice.

2. **Comparar TPE vs Random en 100 evaluaciones**: Ejecuta 100 evaluaciones con tpe.suggest y rand.suggest para el mismo problema de clasificación de productos. Genera un gráfico de convergencia superpuesto.

3. **Multi-objetivo: minimizar error y tamaño del modelo**: Usa hp.choice para seleccionar entre RandomForest, XGBoost y GradientBoosting. La función objetivo debe minimizar [MSE + 0.001 * tamaño_modelo_en_MB]. Muestra los 5 mejores resultados.

4. **Optimizar red neuronal para forecasting de inventarios**: Usando TensorFlow/Keras, optimiza units (32-256), dropout (0.1-0.5), lr (1e-4 a 1e-2) y batch_size (16-128) con hp.choice. Usa hp.loguniform para lr.

5. **Early stopping personalizado**: Implementa un bucle que ejecute fmin incrementando max_evals hasta 200, pero deteniéndose si no hay mejora ≥ 1% en las últimas 10 iteraciones. Muestra en qué trial se detuvo y la mejor pérdida.

6. **points_to_evaluate con datos de ventas**: Realiza una optimización de XGBoost comenzando desde 3 puntos conocidos (basados en búsquedas anteriores). Compara con una ejecución sin puntos iniciales en términos de convergencia.

7. **Visualizar espacio de búsqueda**: Para el problema de la función cuadrática (x-3)^2+(y+2)^2+1, genera un contour plot del espacio de pérdida superpuesto con los puntos evaluados por TPE durante 30 trials. Colorea por orden de evaluación.

8. **Optimización de pipeline completo**: Crea un pipeline con StandardScaler → SelectKBest → XGBoost y optimiza simultáneamente k (1-10), max_depth (3-10), learning_rate (0.01-0.3), subsample (0.5-1.0). Usa 60 evaluaciones y muestra la evolución de la pérdida.

---

## 4. Resumen

| Concepto | Descripción |
|---|---|
| **TPE** | Árbol de estimadores Parzen; modela P(loss|params) para guiar búsqueda |
| **Random Search** | Línea base: muestreo uniforme, útil para espacios de baja dimensión |
| **Anneal** | Simulated annealing; temperatura decreciente para exploración→explotación |
| **hp.choice/uniform/quniform/loguniform** | Espacio de búsqueda categórico, continuo, discreto, logarítmico |
| **fmin** | Función principal: fn, space, algo, max_evals, trials |
| **Trials** | Almacena historial completo de evaluaciones para análisis |
| **SparkTrials** | Paralelización en clúster Spark para acelerar búsqueda |

Hyperopt es ideal para espacios de búsqueda pequeños a medianos (< 100 hiperparámetros). TPE ofrece mejor convergencia que random search en la mayoría de problemas de ventas, especialmente cuando el costo por evaluación es alto (redes neuronales, pipelines complejos).
