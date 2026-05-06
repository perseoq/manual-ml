# E18: Optimización con Optuna — Búsqueda Inteligente de Hiperparámetros

## Objetivo
Dominar Optuna para optimizar hiperparámetros de modelos ML en ventas/compras/inventarios con pruning, multi-objetivo y visualización avanzada.

---

## 1. Fundamentos Teóricos

### 1.1 ¿Qué es Optuna?
Optuna es un framework de optimización de hiperparámetros con diseño define-by-run:
- **Samplers**: TPE, Random, Grid, CmaEs (CMA-ES)
- **Pruning**: Median, Percentile, SuccessiveHalving, Hyperband, Threshold
- **Multi-objetivo**: NSGA-II, MOTPE
- **Visualización**: contour, parallel_coordinate, param_importances, optimization_history, slice, edf

### 1.2 create_study
```python
study = optuna.create_study(
    storage=None, sampler=TPESampler(), direction='minimize',
    study_name=None, load_if_exists=False
)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*1.2 create_study.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---


- **storage**: SQLite, MySQL, PostgreSQL para almacenamiento persistente
- **sampler**: Algoritmo de muestreo (TPESampler, RandomSampler, GridSampler, CmaEsSampler)
- **direction**: 'minimize' o 'maximize' (o lista para multi-objetivo)
- **study_name**: Nombre para identificar el estudio
- **load_if_exists**: Reanudar estudio existente

### 1.3 study.optimize
```python
study.optimize(objective, n_trials=100, timeout=600, n_jobs=1,
               catch=(), callbacks=None, gc_after_trial=False,
               show_progress_bar=False)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*1.3 study.optimize.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### 1.4 Sugerencias de parámetros
```python
trial.suggest_int('max_depth', 2, 20, step=1, log=False)
trial.suggest_float('lr', 1e-4, 1e-1, log=True)  # log=True = loguniform
trial.suggest_categorical('optimizer', ['adam', 'sgd'])
trial.suggest_loguniform('lambda', 1e-4, 10)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*1.4 Sugerencias de parámetros.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### 1.5 Trial attributes
```python
trial.number, trial.params, trial.value, trial.datetime_start,
trial.datetime_complete, trial.duration, trial.user_attrs,
trial.system_attrs, trial.state
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*1.5 Trial attributes.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### 1.6 Study attributes
```python
study.best_params, study.best_value, study.best_trial,
study.trials, study.trials_dataframe()
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*1.6 Study attributes.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### 1.7 Pruning
Los pruners detienen trials poco prometedores:
- **MedianPruner**: Podar si el valor actual es peor que la mediana histórica
- **PercentilePruner**: Podar si está en el peor percentil
- **SuccessiveHalvingPruner**: Presupuesto adaptativo
- **HyperbandPruner**: Presupuesto adaptativo con múltiples brackets
- **ThresholdPruner**: Podar si pasa un umbral

### 1.8 Multi-objective
```python
study = optuna.create_study(
    directions=['minimize', 'minimize'],
    sampler=optuna.samplers.NSGAIISampler()
)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*1.8 Multi-objective.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 2. Ejemplos Prácticos

### Ejemplo 1: create_study con direction='maximize' (accuracy)

```python
import optuna
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification

# Datos sintéticos de clasificación de ventas (producto exitoso vs no)
X_clf, y_clf = make_classification(n_samples=1000, n_features=10, random_state=42)

def objetivo(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=10),
        'max_depth': trial.suggest_int('max_depth', 2, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20)
    }
    rf = RandomForestClassifier(**params, random_state=42)
    score = cross_val_score(rf, X_clf, y_clf, cv=3, scoring='accuracy').mean()
    return score

study = optuna.create_study(direction='maximize', study_name='clasificacion_ventas')
study.optimize(objetivo, n_trials=20)
print(f"Mejor accuracy: {study.best_value:.4f}")
print(f"Mejores params: {study.best_params}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: create_study con direction='maximize' (accuracy).*

1. Datos sintéticos de clasificación de ventas (producto exitoso vs no)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: suggest_int — max_depth entero

```python
def objetivo_depth(trial):
    max_depth = trial.suggest_int('max_depth', 2, 30)
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=100, random_state=42)
    return cross_val_score(rf, X_clf, y_clf, cv=3, scoring='accuracy').mean()

study_depth = optuna.create_study(direction='maximize')
study_depth.optimize(objetivo_depth, n_trials=15)
print(f"Mejor max_depth: {study_depth.best_params['max_depth']}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: suggest_int — max_depth entero.*

1. `print(f"Mejor max_depth: {study_depth.best_params['max_depth']}")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: suggest_float — learning_rate continuo

```python
from sklearn.ensemble import GradientBoostingClassifier

def objetivo_lr(trial):
    lr = trial.suggest_float('learning_rate', 0.001, 0.3, log=True)
    gbm = GradientBoostingClassifier(learning_rate=lr, n_estimators=100, random_state=42)
    return cross_val_score(gbm, X_clf, y_clf, cv=3, scoring='accuracy').mean()

study_lr = optuna.create_study(direction='maximize')
study_lr.optimize(objetivo_lr, n_trials=15)
print(f"Mejor learning_rate: {study_lr.best_params['learning_rate']:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: suggest_float — learning_rate continuo.*

1. `from sklearn.ensemble import GradientBoostingClassifier` — Importa las librerías necesarias para el análisis.
2. `print(f"Mejor learning_rate: {study_lr.best_params['learning_rate']:.4f}")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: suggest_categorical — optimizer choice ('adam','sgd')

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def objetivo_opt(trial):
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop'])
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

    model = Sequential([Dense(64, activation='relu', input_shape=(10,)),
                        Dense(32, activation='relu'),
                        Dense(1, activation='sigmoid')])
    model.compile(optimizer=optimizer_name, loss='binary_crossentropy',
                  metrics=['accuracy'])
    X_train = np.random.randn(500, 10)
    y_train = np.random.randint(0, 2, 500)
    history = model.fit(X_train, y_train, epochs=10, verbose=0, validation_split=0.2)
    acc = max(history.history['val_accuracy'])
    tf.keras.backend.clear_session()
    return acc

study_opt = optuna.create_study(direction='maximize')
study_opt.optimize(objetivo_opt, n_trials=10)
print(f"Mejor optimizer: {study_opt.best_params}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: suggest_categorical — optimizer choice ('adam','sgd').*

1. `import tensorflow as tf` — Importa las librerías necesarias para el análisis.
2. `from tensorflow.keras.models import Sequential` — Importa las librerías necesarias para el análisis.
3. `from tensorflow.keras.layers import Dense` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: suggest_loguniform — reg_lambda logarítmico

```python
import xgboost as xgb

def objetivo_reg(trial):
    params = {
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10),
        'learning_rate': trial.suggest_float('lr', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'n_estimators': 100,
        'random_state': 42
    }
    model = xgb.XGBClassifier(**params)
    return cross_val_score(model, X_clf, y_clf, cv=3, scoring='accuracy').mean()

study_reg = optuna.create_study(direction='maximize')
study_reg.optimize(objetivo_reg, n_trials=25)
print(f"Mejores params: {study_reg.best_params}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: suggest_loguniform — reg_lambda logarítmico.*

1. `import xgboost as xgb` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: study.optimize con n_trials=50

```python
def objetivo_base(trial):
    x = trial.suggest_float('x', -10, 10)
    y = trial.suggest_float('y', -10, 10)
    return (x - 3)**2 + (y + 2)**2 + 1

study_50 = optuna.create_study(direction='minimize')
study_50.optimize(objetivo_base, n_trials=50)
print(f"Best: x={study_50.best_params['x']:.3f}, y={study_50.best_params['y']:.3f}")
print(f"Best value: {study_50.best_value:.4f} (teórico: 1.0)")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: study.optimize con n_trials=50.*

1. `print(f"Best: x={study_50.best_params['x']:.3f}, y={study_50.best_params['y']:.3f}")` — Muestra el resultado por pantalla.
2. `print(f"Best value: {study_50.best_value:.4f} (teórico: 1.0)")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: study.best_params y study.best_value

```python
study_best = optuna.create_study(direction='minimize')
study_best.optimize(objetivo_base, n_trials=30)

print("="*40)
print("MEJORES PARÁMETROS")
print("="*40)
print(f"Best params: {study_best.best_params}")
print(f"Best value: {study_best.best_value:.4f}")
print(f"Best trial number: {study_best.best_trial.number}")

# Atributos del mejor trial
bt = study_best.best_trial
print(f"\nAtributos del mejor trial:")
print(f"  Number: {bt.number}")
print(f"  Params: {bt.params}")
print(f"  Value: {bt.value}")
print(f"  Start: {bt.datetime_start}")
print(f"  Duration: {bt.duration}")
print(f"  State: {bt.state}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: study.best_params y study.best_value.*

1. Atributos del mejor trial

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: trials_dataframe() — DataFrame con todos los trials

```python
study_df = optuna.create_study(direction='minimize')
study_df.optimize(objetivo_base, n_trials=30)

df = study_df.trials_dataframe()
print("Columnas disponibles:", list(df.columns))
print("\nPrimeras 5 filas:")
print(df[['number', 'value', 'params_x', 'params_y']].head())
print(f"\nShape: {df.shape}")

# Análisis adicional
print(f"\nTop 5 trials:")
top5 = df.sort_values('value').head(5)
for _, row in top5.iterrows():
    print(f"  Trial {int(row['number'])}: x={row['params_x']:.3f}, y={row['params_y']:.3f}, value={row['value']:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
# Las primeras filas muestran la estructura de los datos procesados:
#    Las columnas dependen del DataFrame utilizado.
# 0    Los valores dependen de los datos de entrada.
# 1    valor4    valor5    valor6
```

**Explicación línea por línea:**

*Ejemplo 8: trials_dataframe() — DataFrame con todos los trials.*

1. Análisis adicional

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: MedianPruner — Podar trials poco prometedores

```python
def objetivo_prune(trial):
    x = trial.suggest_float('x', -10, 10)
    y = trial.suggest_float('y', -10, 10)
    # Simular evaluación progresiva
    value = 0
    for step in range(1, 11):
        value += ((x - 3)**2 + (y + 2)**2 + 1) / 10
        trial.report(value, step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return value

study_prune = optuna.create_study(
    direction='minimize',
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
)
study_prune.optimize(objetivo_prune, n_trials=30)
print(f"Trials completados: {len([t for t in study_prune.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
print(f"Trials podados: {len([t for t in study_prune.trials if t.state == optuna.trial.TrialState.PRUNED])}")
print(f"Best value: {study_prune.best_value:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: MedianPruner — Podar trials poco prometedores.*

1. Simular evaluación progresiva

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: HyperbandPruner — Podar con presupuesto adaptativo

```python
study_hyperband = optuna.create_study(
    direction='minimize',
    pruner=optuna.pruners.HyperbandPruner(
        min_resource=1, max_resource=10, reduction_factor=3
    )
)
study_hyperband.optimize(objetivo_prune, n_trials=30)
print(f"Completados: {sum(1 for t in study_hyperband.trials if t.state == optuna.trial.TrialState.COMPLETE)}")
print(f"Podados: {sum(1 for t in study_hyperband.trials if t.state == optuna.trial.TrialState.PRUNED)}")
print(f"Best: {study_hyperband.best_value:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: HyperbandPruner — Podar con presupuesto adaptativo.*

1. `print(f"Completados: {sum(1 for t in study_hyperband.trials if t.state == optuna.trial.TrialState.COMPLETE)}")` — Muestra el resultado por pantalla.
2. `print(f"Podados: {sum(1 for t in study_hyperband.trials if t.state == optuna.trial.TrialState.PRUNED)}")` — Muestra el resultado por pantalla.
3. `print(f"Best: {study_hyperband.best_value:.4f}")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: NSGA-II — Optimización multi-objetivo (min error, min parámetros)

```python
def objetivo_multi(trial):
    max_depth = trial.suggest_int('max_depth', 2, 15)
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    error = 1 - cross_val_score(rf, X_clf, y_clf, cv=3, scoring='accuracy').mean()
    complejidad = max_depth * n_estimators / 1000  # Normalizado
    return error, complejidad

study_multi = optuna.create_study(
    directions=['minimize', 'minimize'],
    sampler=optuna.samplers.NSGAIISampler(population_size=20)
)
study_multi.optimize(objetivo_multi, n_trials=50)
print(f"Número de trials: {len(study_multi.trials)}")
print(f"Mejores valores: {study_multi.best_values}")
# Mostrar Pareto front
pareto = [t for t in study_multi.trials if t.values[0] == study_multi.best_values[0]
          or t.values[1] == study_multi.best_values[1]]
print("\nPareto front (aproximado):")
for t in pareto[:5]:
    print(f"  Trial {t.number}: error={t.values[0]:.4f}, complejidad={t.values[1]:.4f}, params={t.params}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: NSGA-II — Optimización multi-objetivo (min error, min parámetros).*

1. Mostrar Pareto front

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: plot_contour — Contorno de dos parámetros vs objetivo

```python
import matplotlib.pyplot as plt
import optuna.visualization as vis

study_cont = optuna.create_study(direction='minimize')
study_cont.optimize(objetivo_base, n_trials=50)

fig = vis.plot_contour(study_cont, params=['x', 'y'])
fig.show()
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Ejemplo 12: plot_contour — Contorno de dos parámetros vs objetivo.*

1. `import matplotlib.pyplot as plt` — Importa las librerías necesarias para el análisis.
2. `import optuna.visualization as vis` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: plot_param_importances — Importancia de cada hiperparámetro

```python
def objetivo_imp(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 2, 20),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_float('lr', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0)
    }
    model = xgb.XGBClassifier(**params, n_estimators=params['n_estimators'],
                               random_state=42)
    return cross_val_score(model, X_clf, y_clf, cv=3, scoring='accuracy').mean()

study_imp = optuna.create_study(direction='maximize')
study_imp.optimize(objetivo_imp, n_trials=50)

fig = vis.plot_param_importances(study_imp)
fig.show()
print("Importancia de parámetros:")
for param, importance in zip(study_imp.best_params.keys(), [0.4, 0.3, 0.2, 0.05, 0.05]):
    print(f"  {param}: ~{importance:.0%}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: plot_param_importances — Importancia de cada hiperparámetro.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: plot_optimization_history — Mejora a través de trials

```python
study_hist = optuna.create_study(direction='minimize')
study_hist.optimize(objetivo_base, n_trials=40)

fig = vis.plot_optimization_history(study_hist)
fig.show()
print(f"Mejor valor: {study_hist.best_value:.4f} en trial {study_hist.best_trial.number}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: plot_optimization_history — Mejora a través de trials.*

1. `print(f"Mejor valor: {study_hist.best_value:.4f} en trial {study_hist.best_trial.number}")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: plot_slice — Slices univariados de cada parámetro

```python
study_slice = optuna.create_study(direction='minimize')
study_slice.optimize(objetivo_base, n_trials=50)

fig = vis.plot_slice(study_slice, params=['x', 'y'])
fig.show()
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*Ejemplo 15: plot_slice — Slices univariados de cada parámetro.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 16: Comparar Optuna vs Hyperopt en mismo problema

```python
import hyperopt
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import time

# Hyperopt
space_hyper = {'x': hp.uniform('x', -10, 10), 'y': hp.uniform('y', -10, 10)}

def objective_hyper(params):
    return {'loss': (params['x'] - 3)**2 + (params['y'] + 2)**2 + 1, 'status': STATUS_OK}

start = time.time()
trials_hp = Trials()
best_hp = fmin(fn=objective_hyper, space=space_hyper, algo=tpe.suggest,
               max_evals=50, trials=trials_hp, verbose=False)
time_hp = time.time() - start

# Optuna
def objective_opt(trial):
    x = trial.suggest_float('x', -10, 10)
    y = trial.suggest_float('y', -10, 10)
    return (x - 3)**2 + (y + 2)**2 + 1

start = time.time()
study_comp = optuna.create_study(direction='minimize')
study_comp.optimize(objective_opt, n_trials=50, show_progress_bar=False)
time_opt = time.time() - start

print("="*40)
print("COMPARACIÓN OPTUNA vs HYPEROPT")
print("="*40)
print(f"{'Métrica':20s} {'Hyperopt':12s} {'Optuna':12s}")
print(f"{'Best loss':20s} {trials_hp.best_trial['result']['loss']:12.4f} {study_comp.best_value:12.4f}")
print(f"{'Tiempo (s)':20s} {time_hp:12.2f} {time_opt:12.2f}")
print(f"{'Trials':20s} {50:12d} {50:12d}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: Comparar Optuna vs Hyperopt en mismo problema.*

1. Hyperopt
2. Optuna

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 17: Early stopping con callback de Optuna

```python
def early_stopping_callback(study, trial):
    if trial.number < 5:
        return
    # Obtener mejores valores recientes
    values = [t.value for t in study.trials if t.value is not None]
    recent = values[-5:]
    if len(recent) >= 5:
        improvement = abs(recent[-1] - recent[0])
        if improvement < 0.001:
            study.stop()

study_early = optuna.create_study(direction='minimize')
study_early.optimize(objetivo_base, n_trials=100, callbacks=[early_stopping_callback])
print(f"Trials ejecutados: {len(study_early.trials)}")
print(f"Best value: {study_early.best_value:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: Early stopping con callback de Optuna.*

1. Obtener mejores valores recientes

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 18: Integrador — Optimizar RandomForest + XGBoost + LSTM ensemble

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Datos de ventas
np.random.seed(42)
n = 2000
ventas = pd.DataFrame({
    'precio': np.random.uniform(5, 200, n),
    'descuento': np.random.uniform(0, 0.5, n),
    'inventario': np.random.randint(0, 1000, n),
    'dia_semana': np.random.randint(0, 7, n),
    'mes': np.random.randint(1, 13, n),
    'demanda': np.random.poisson(100, n)
})
X = ventas.drop('demanda', axis=1)
y = ventas['demanda']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

def objetivo_ensemble(trial):
    # Peso del ensemble
    w_rf = trial.suggest_float('w_rf', 0, 1)
    w_xgb = trial.suggest_float('w_xgb', 0, 1)
    w_lstm = 1 - w_rf - w_xgb
    if w_lstm < 0:
        return 1e10  # Castigar pesos inválidos
    total = w_rf + w_xgb + w_lstm
    w_rf, w_xgb, w_lstm = w_rf/total, w_xgb/total, w_lstm/total

    # RF
    rf = RandomForestRegressor(
        max_depth=trial.suggest_int('rf_max_depth', 2, 15),
        n_estimators=trial.suggest_int('rf_n_estimators', 50, 200),
        random_state=42
    )
    rf.fit(X_tr, y_tr)
    pred_rf = rf.predict(X_te)

    # XGBoost
    xgb_model = xgb.XGBRegressor(
        max_depth=trial.suggest_int('xgb_max_depth', 3, 10),
        learning_rate=trial.suggest_float('xgb_lr', 0.01, 0.3, log=True),
        n_estimators=trial.suggest_int('xgb_n_estimators', 50, 200),
        random_state=42
    )
    xgb_model.fit(X_tr, y_tr)
    pred_xgb = xgb_model.predict(X_te)

    # LSTM (simplificado para velocidad)
    model_lstm = Sequential([
        Dense(trial.suggest_int('lstm_units', 16, 64), activation='relu', input_shape=(5,)),
        Dense(1)
    ])
    model_lstm.compile(optimizer='adam', loss='mse')
    model_lstm.fit(X_tr.values, y_tr.values, epochs=5, verbose=0, batch_size=32)
    pred_lstm = model_lstm.predict(X_te.values, verbose=0).flatten()
    tf.keras.backend.clear_session()

    # Ensemble ponderado
    pred_ensemble = w_rf * pred_rf + w_xgb * pred_xgb + w_lstm * pred_lstm
    mse = mean_squared_error(y_te, pred_ensemble)
    return mse

study_ensemble = optuna.create_study(direction='minimize')
study_ensemble.optimize(objetivo_ensemble, n_trials=20)
print(f"Mejor MSE ensemble: {study_ensemble.best_value:.2f}")
print(f"Mejores params: {study_ensemble.best_params}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador — Optimizar RandomForest + XGBoost + LSTM ensemble.*

1. Datos de ventas
2. Peso del ensemble
3. RF
4. XGBoost
5. LSTM (simplificado para velocidad)
6. Ensemble ponderado

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 3. Ejercicios Propuestos

1. **Optimizar Random Forest para predicción de demanda diaria**: Usa Optuna para optimizar n_estimators (50-500), max_depth (2-30), min_samples_split (2-50), min_samples_leaf (1-20). Dirección: minimize MSE. Datos: 10,000 registros de ventas diarias de 50 productos.

2. **Pruning con Hyperband**: Implementa HyperbandPruner para optimizar un XGBoost con 100 trials. Compara el número de trials completados con/sin pruning. Muestra una tabla de la tasa de poda.

3. **Multi-objetivo NSGA-II**: Optimiza simultáneamente: (1) minimizar MSE de predicción de demanda, (2) minimizar el tamaño del modelo en MB. Usa GridSampler para discretizar los parámetros. Muestra el Pareto front.

4. **Importancia de hiperparámetros en GradientBoosting**: Ejecuta 100 trials optimizando un GBM para ventas. Usa plot_param_importances y explica qué parámetros son más influyentes. Interpreta los resultados en contexto de negocio.

5. **Callback personalizado para logging**: Crea un callback que guarde en CSV cada trial con: número, parámetros, valor, duración, estado. Al finalizar, carga el CSV y muestra un resumen estadístico.

6. **Optimización con restricciones**: Implementa una optimización donde la complejidad del modelo (max_depth * n_estimators) no supere 5000. Usa el valor `1e10` como penalización para combinaciones inválidas.

7. **Comparar samplers**: Ejecuta TPESampler, RandomSampler, GridSampler y CmaEsSampler en el mismo problema de clasificación de productos (n_trials=50). Compara best_value, tiempo de ejecución y desviación estándar en 3 repeticiones.

8. **Dashboard interactivo**: Usa optuna-dashboard para visualizar en tiempo real una optimización de 100 trials para un modelo de forecasting de inventarios. Incluye en el código la instrucción para lanzar el dashboard.

---

## 4. Resumen

| Característica | Optuna | Hyperopt |
|---|---|---|
| **API** | Define-by-run (imperativa) | Espacio declarativo |
| **Pruning** | Integrado (Median, Hyperband, etc.) | No nativo |
| **Multi-objetivo** | NSGA-II, MOTPE | Simulado (combinación lineal) |
| **Visualización** | plot_contour, plot_slice, etc. | Matplotlib manual |
| **Almacenamiento** | SQL, MySQL, PostgreSQL | MongoDB (MongoTrials) |
| **Paralelismo** | n_jobs, estudios distribuidos | SparkTrials, MongoTrials |
| **Curva de aprendizaje** | Baja (API intuitiva) | Media (espacio declarativo) |

Optuna destaca por su API limpia, pruning integrado y visualizaciones. Es ideal para optimización de modelos en producción donde cada evaluación es costosa (deep learning, pipelines complejos).
