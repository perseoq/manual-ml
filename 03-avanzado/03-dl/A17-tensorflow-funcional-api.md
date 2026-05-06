# A17: TensorFlow Functional API — Modelos Complejos para Ventas, Compras e Inventarios

## Introducción Teórica

La **Functional API** de tf.keras permite construir modelos con topologías no lineales: múltiples entradas, múltiples salidas, capas compartidas, skip connections y arquitecturas ensambladas. Es más flexible que Sequential y necesaria para problemas reales de negocio donde los datos tienen estructuras heterogéneas.

### Componentes principales:

1. **tf.keras.layers.Input(shape)**: Define el tensor de entrada del modelo. Es el punto de partida de todo modelo funcional.
2. **tf.keras.Model(inputs, outputs)**: Crea el modelo especificando sus entradas y salidas. Puede tener múltiples de cada una.
3. **Capas de combinación**: `concatenate`, `add`, `multiply`, `subtract`, `average`, `maximum`, `minimum` para fusionar ramas.
4. **Skip connections**: Conexiones que saltan capas intermedias, fundamentales en redes profundas (ResNet).
5. **Capas compartidas**: Una misma capa Dense (con los mismos pesos) se reutiliza en múltiples entradas.
6. **Subclassing API**: `class MyModel(tf.keras.Model)` con método `call(self, inputs, training=False, mask=None)` para máximo control.
7. **Custom Layer**: `class MyLayer(tf.keras.layers.Layer)` con métodos `build`, `call`, `compute_output_shape`, `get_config`, `from_config`.
8. **Modelo ensamblado**: Combinar múltiples modelos independientes promediando sus salidas.

### Aplicaciones en negocio:

- **Ventas**: Multi-input (precios + texto de descripciones → predicción); multi-output (cantidad + margen simultáneamente).
- **Compras**: Capas compartidas para evaluar múltiples proveedores con la misma red; skip connections para estabilizar entrenamiento.
- **Inventarios**: Modelo ensamblado combinando redes de rotación, estacionalidad y costos.

---

## Ejemplos

### Ejemplo 1: Modelo funcional básico — Input → Dense → Output para predicción de ventas

```python
import tensorflow as tf
import numpy as np

inputs = tf.keras.layers.Input(shape=(3,), name='features_venta')
x = tf.keras.layers.Dense(32, activation='relu', name='capa_oculta')(inputs)
outputs = tf.keras.layers.Dense(1, name='salida')(x)

modelo = tf.keras.Model(inputs=inputs, outputs=outputs, name='modelo_ventas')
modelo.compile(optimizer='adam', loss='mse', metrics=['mae'])

X = np.random.rand(500, 3).astype(np.float32)
y = np.random.rand(500, 1).astype(np.float32)
modelo.fit(X, y, epochs=10, batch_size=32, verbose=0)
print(f"Modelo funcional básico: {modelo.name}")
print(f"Input: {modelo.input_shape}, Output: {modelo.output_shape}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: Modelo funcional básico — Input → Dense → Output para predicción de ventas.*

1. `import tensorflow as tf` — Importa las librerías necesarias para el análisis.
2. `import numpy as np` — Importa las librerías necesarias para el análisis.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: Multi-input — Entrada de precio (numérica) + descripción (texto) para clasificación

```python
# Rama numérica: precio, descuento, días_promoción
input_num = tf.keras.layers.Input(shape=(3,), name='input_numerico')
x_num = tf.keras.layers.Dense(16, activation='relu')(input_num)
x_num = tf.keras.layers.Dense(8, activation='relu')(x_num)

# Rama textual: descripción del producto (embeddings simulados)
input_text = tf.keras.layers.Input(shape=(50,), name='input_texto')
x_text = tf.keras.layers.Dense(32, activation='relu')(input_text)
x_text = tf.keras.layers.Dense(8, activation='relu')(x_text)

# Concatenar ambas ramas
concat = tf.keras.layers.concatenate([x_num, x_text], name='fusion')
output = tf.keras.layers.Dense(1, activation='sigmoid', name='clasificacion')(concat)

modelo_multi = tf.keras.Model(inputs=[input_num, input_text], outputs=output)
modelo_multi.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

X_num = np.random.rand(300, 3).astype(np.float32)
X_text = np.random.rand(300, 50).astype(np.float32)
y_bin = np.random.randint(0, 2, (300, 1)).astype(np.float32)
modelo_multi.fit([X_num, X_text], y_bin, epochs=10, batch_size=16, verbose=0)
print(f"Model multi-input entrenado. Accuracy: {modelo_multi.evaluate([X_num, X_text], y_bin, verbose=0)[1]:.3f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: Multi-input — Entrada de precio (numérica) + descripción (texto) para clasificación.*

1. Rama numérica: precio, descuento, días_promoción
2. Rama textual: descripción del producto (embeddings simulados)
3. Concatenar ambas ramas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: Multi-input — Concatenar features numéricas + categóricas para predicción de compras

```python
# Features numéricas: precio_unitario, cantidad, costo_envío
input_num2 = tf.keras.layers.Input(shape=(3,), name='numerico')

# Features categóricas: encoded como one-hot de 10 categorías
input_cat = tf.keras.layers.Input(shape=(10,), name='categorico')

x_num2 = tf.keras.layers.Dense(8, activation='relu')(input_num2)
x_cat = tf.keras.layers.Dense(8, activation='relu')(input_cat)

fusion = tf.keras.layers.concatenate([x_num2, x_cat])
x = tf.keras.layers.Dense(16, activation='relu')(fusion)
output_compra = tf.keras.layers.Dense(1, name='monto_total')(x)

modelo_compra = tf.keras.Model(inputs=[input_num2, input_cat], outputs=output_compra)
modelo_compra.compile(optimizer='adam', loss='mse')

X_n = np.random.rand(400, 3).astype(np.float32)
X_c = np.random.rand(400, 10).astype(np.float32)
y_c = np.random.rand(400, 1).astype(np.float32)
modelo_compra.fit([X_n, X_c], y_c, epochs=15, batch_size=32, verbose=0)
print(f"Modelo multi-input (numérico + categórico) entrenado")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: Multi-input — Concatenar features numéricas + categóricas para predicción de compras.*

1. Features numéricas: precio_unitario, cantidad, costo_envío
2. Features categóricas: encoded como one-hot de 10 categorías

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: Multi-output — Predecir cantidad y margen simultáneamente

```python
input_mo = tf.keras.layers.Input(shape=(5,), name='entrada')
x = tf.keras.layers.Dense(32, activation='relu')(input_mo)
x = tf.keras.layers.Dense(16, activation='relu')(x)

# Dos salidas: cantidad vendida (regresión) y margen (regresión)
output_cantidad = tf.keras.layers.Dense(1, name='cantidad')(x)
output_margen = tf.keras.layers.Dense(1, name='margen')(x)

modelo_mo = tf.keras.Model(inputs=input_mo, outputs=[output_cantidad, output_margen])
modelo_mo.compile(
    optimizer='adam',
    loss={'cantidad': 'mse', 'margen': 'mse'},
    loss_weights={'cantidad': 0.7, 'margen': 0.3},
    metrics={'cantidad': 'mae', 'margen': 'mae'}
)

X_mo = np.random.rand(600, 5).astype(np.float32)
y_cant = np.random.rand(600, 1).astype(np.float32)
y_marg = np.random.rand(600, 1).astype(np.float32)
hist_mo = modelo_mo.fit(X_mo, {'cantidad': y_cant, 'margen': y_marg},
                        epochs=20, batch_size=32, verbose=0)
print(f"Cantidad loss final: {hist_mo.history['cantidad_loss'][-1]:.4f}")
print(f"Margen loss final: {hist_mo.history['margen_loss'][-1]:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: Multi-output — Predecir cantidad y margen simultáneamente.*

1. Dos salidas: cantidad vendida (regresión) y margen (regresión)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: Modelo con skip connection (residual block simple) para inventarios

```python
input_sk = tf.keras.layers.Input(shape=(10,), name='features_inventario')

# Rama principal
x = tf.keras.layers.Dense(32, activation='relu', name='dense1')(input_sk)
x = tf.keras.layers.Dense(32, activation='relu', name='dense2')(x)

# Skip connection: suma la entrada transformada
skip = tf.keras.layers.Dense(32, name='skip_projection')(input_sk)  # proyectar a 32
added = tf.keras.layers.add([x, skip], name='skip_connection')
output_sk = tf.keras.layers.Dense(1, name='rotacion_predicha')(added)

modelo_skip = tf.keras.Model(inputs=input_sk, outputs=output_sk)
modelo_skip.compile(optimizer='adam', loss='mse')

X_sk = np.random.rand(400, 10).astype(np.float32)
y_sk = np.random.rand(400, 1).astype(np.float32)
modelo_skip.fit(X_sk, y_sk, epochs=15, batch_size=32, verbose=0)
print(f"Modelo con skip connection: {modelo_skip.count_params()} parámetros")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: Modelo con skip connection (residual block simple) para inventarios.*

1. Rama principal
2. Skip connection: suma la entrada transformada

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: Capa compartida — Misma Dense procesa dos entradas de proveedores

```python
# Una misma capa Dense evalúa dos proveedores por separado
capa_compartida = tf.keras.layers.Dense(16, activation='relu', name='evaluador_proveedor')
capa_salida_compartida = tf.keras.layers.Dense(1, name='score_proveedor')

# Entradas de dos proveedores
input_prov1 = tf.keras.layers.Input(shape=(4,), name='proveedor1')
input_prov2 = tf.keras.layers.Input(shape=(4,), name='proveedor2')

# Mismos pesos para ambos
score1 = capa_salida_compartida(capa_compartida(input_prov1))
score2 = capa_salida_compartida(capa_compartida(input_prov2))

# Diferencia de scores
diferencia = tf.keras.layers.subtract([score1, score2], name='diferencia')

modelo_compartido = tf.keras.Model(inputs=[input_prov1, input_prov2], outputs=diferencia)
modelo_compartido.compile(optimizer='adam', loss='mse')

X_p1 = np.random.rand(300, 4).astype(np.float32)
X_p2 = np.random.rand(300, 4).astype(np.float32)
y_diff = np.random.rand(300, 1).astype(np.float32)
modelo_compartido.fit([X_p1, X_p2], y_diff, epochs=10, batch_size=16, verbose=0)

# Verificar que comparten pesos
w1 = capa_compartida.get_weights()
print(f"Pesos de capa compartida: {w1[0].shape}, id: {id(capa_compartida.get_weights()[0])}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: Capa compartida — Misma Dense procesa dos entradas de proveedores.*

1. Una misma capa Dense evalúa dos proveedores por separado
2. Entradas de dos proveedores
3. Mismos pesos para ambos
4. Diferencia de scores
5. Verificar que comparten pesos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: Concatenate + Dense — Fusionar features de ventas y devoluciones

```python
input_ventas = tf.keras.layers.Input(shape=(5,), name='ventas')
input_dev = tf.keras.layers.Input(shape=(3,), name='devoluciones')

x_ventas = tf.keras.layers.Dense(16, activation='relu')(input_ventas)
x_dev = tf.keras.layers.Dense(8, activation='relu')(input_dev)

concat_fusion = tf.keras.layers.concatenate([x_ventas, x_dev])
x_final = tf.keras.layers.Dense(16, activation='relu')(concat_fusion)
output_fusion = tf.keras.layers.Dense(1, name='venta_neta')(x_final)

modelo_fusion = tf.keras.Model(inputs=[input_ventas, input_dev], outputs=output_fusion)
modelo_fusion.compile(optimizer='adam', loss='mse')

X_v = np.random.rand(500, 5).astype(np.float32)
X_d = np.random.rand(500, 3).astype(np.float32)
y_fn = np.random.rand(500, 1).astype(np.float32)
modelo_fusion.fit([X_v, X_d], y_fn, epochs=10, batch_size=32, verbose=0)
print(f"Fusión ventas+devoluciones entrenada")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: Concatenate + Dense — Fusionar features de ventas y devoluciones.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: Add layer — Sumar dos ramas (skip connection) para estabilidad

```python
input_add = tf.keras.layers.Input(shape=(8,), name='entrada')

rama_a = tf.keras.layers.Dense(16, activation='relu', name='rama_a')(input_add)
rama_a = tf.keras.layers.Dense(8, name='rama_a_out')(rama_a)

rama_b = tf.keras.layers.Dense(8, activation='tanh', name='rama_b')(input_add)

suma = tf.keras.layers.add([rama_a, rama_b], name='suma_ramas')
output_add = tf.keras.layers.Dense(1, name='salida')(suma)

modelo_add = tf.keras.Model(inputs=input_add, outputs=output_add)
modelo_add.compile(optimizer='adam', loss='mse')
print(f"Arquitectura con Add: {[layer.name for layer in modelo_add.layers]}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: Add layer — Sumar dos ramas (skip connection) para estabilidad.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: Subclassing API — Modelo personalizado con call() para predicción de demanda

```python
class ModeloDemanda(tf.keras.Model):
    def __init__(self, units_1=32, units_2=16, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(units_1, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units_2, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False, mask=None):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)

modelo_sub = ModeloDemanda(units_1=64, units_2=32)
modelo_sub.compile(optimizer='adam', loss='mse')
X_sub = np.random.rand(300, 5).astype(np.float32)
y_sub = np.random.rand(300, 1).astype(np.float32)
modelo_sub.fit(X_sub, y_sub, epochs=10, batch_size=32, verbose=0)
print(f"Modelo Subclassing: {modelo_sub.count_params()} parámetros")
pred_sub = modelo_sub.predict(X_sub[:2], verbose=0)
print(f"Predicción subclassing: {pred_sub.flatten()}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: Subclassing API — Modelo personalizado con call() para predicción de demanda.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: Subclassing con training=True/False para dropout y batchnorm

```python
class ModeloConDropout(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(0.3)
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        self.out = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False, mask=None):
        x = self.dense1(inputs)
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        return self.out(x)

modelo_drop = ModeloConDropout()
modelo_drop.compile(optimizer='adam', loss='mse')
X_drop = np.random.rand(500, 10).astype(np.float32)
y_drop = np.random.rand(500, 1).astype(np.float32)
modelo_drop.fit(X_drop, y_drop, epochs=20, batch_size=32, verbose=0)
# Inferencia (training=False automático)
pred_drop = modelo_drop.predict(X_drop[:3], verbose=0)
print(f"Predicciones con dropout en inferencia (desactivado): {pred_drop.flatten()}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: Subclassing con training=True/False para dropout y batchnorm.*

1. Inferencia (training=False automático)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: Custom Layer — Capa personalizada con pesos entrenables para escalado de precios

```python
class EscaladoPrecios(tf.keras.layers.Layer):
    def __init__(self, escala_inicial=1.0, **kwargs):
        super().__init__(**kwargs)
        self.escala_inicial = escala_inicial

    def build(self, input_shape):
        self.escala = self.add_weight(
            name='escala',
            shape=(1,),
            initializer=tf.keras.initializers.Constant(self.escala_inicial),
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs * self.escala

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({'escala_inicial': self.escala_inicial})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

inputs_cl = tf.keras.layers.Input(shape=(3,))
x_cl = tf.keras.layers.Dense(16, activation='relu')(inputs_cl)
x_cl = EscaladoPrecios(escala_inicial=1.5)(x_cl)
outputs_cl = tf.keras.layers.Dense(1)(x_cl)

modelo_cl = tf.keras.Model(inputs=inputs_cl, outputs=outputs_cl)
modelo_cl.compile(optimizer='adam', loss='mse')
X_cl = np.random.rand(200, 3).astype(np.float32)
y_cl = np.random.rand(200, 1).astype(np.float32)
modelo_cl.fit(X_cl, y_cl, epochs=10, batch_size=16, verbose=0)
escala_layer = modelo_cl.get_layer('escalado_precios')
print(f"Escala aprendida: {escala_layer.escala.numpy()[0]:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: Custom Layer — Capa personalizada con pesos entrenables para escalado de precios.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: Modelo ensamblado — Promediar 3 modelos independientes de ventas

```python
def crear_modelo_ventas(seed):
    tf.random.set_seed(seed)
    inputs = tf.keras.layers.Input(shape=(5,))
    x = tf.keras.layers.Dense(16, activation='relu')(inputs)
    x = tf.keras.layers.Dense(8, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

# Crear 3 modelos
m1 = crear_modelo_ventas(1)
m2 = crear_modelo_ventas(2)
m3 = crear_modelo_ventas(3)

X_ens = np.random.rand(400, 5).astype(np.float32)
y_ens = np.random.rand(400, 1).astype(np.float32)

m1.fit(X_ens, y_ens, epochs=10, batch_size=32, verbose=0)
m2.fit(X_ens, y_ens, epochs=10, batch_size=32, verbose=0)
m3.fit(X_ens, y_ens, epochs=10, batch_size=32, verbose=0)

# Ensamble por promedio
input_ens = tf.keras.layers.Input(shape=(5,))
out1 = m1(input_ens)
out2 = m2(input_ens)
out3 = m3(input_ens)
promedio = tf.keras.layers.average([out1, out2, out3])

modelo_ensemble = tf.keras.Model(inputs=input_ens, outputs=promedio)
X_test_ens = np.random.rand(100, 5).astype(np.float32)
pred_ensemble = modelo_ensemble.predict(X_test_ens, verbose=0)
print(f"Ensemble predictions shape: {pred_ensemble.shape}")
print(f"Promedio de los 3 modelos: {pred_ensemble[:3].flatten()}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: Modelo ensamblado — Promediar 3 modelos independientes de ventas.*

1. Crear 3 modelos
2. Ensamble por promedio

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: Modelo con múltiples pérdidas (loss_weights) para ventas y costos

```python
input_ml = tf.keras.layers.Input(shape=(6,), name='features')
x = tf.keras.layers.Dense(32, activation='relu')(input_ml)
x = tf.keras.layers.Dense(16, activation='relu')(x)

out_ingresos = tf.keras.layers.Dense(1, name='ingresos')(x)
out_costos = tf.keras.layers.Dense(1, name='costos')(x)
out_utilidad = tf.keras.layers.Dense(1, name='utilidad')(x)

modelo_ml = tf.keras.Model(inputs=input_ml, outputs=[out_ingresos, out_costos, out_utilidad])
modelo_ml.compile(
    optimizer='adam',
    loss={'ingresos': 'mse', 'costos': 'mse', 'utilidad': 'mse'},
    loss_weights={'ingresos': 0.3, 'costos': 0.3, 'utilidad': 0.4}
)

X_ml = np.random.rand(500, 6).astype(np.float32)
y_ing = np.random.rand(500, 1).astype(np.float32)
y_cost = np.random.rand(500, 1).astype(np.float32)
y_util = np.random.rand(500, 1).astype(np.float32)
hist_ml = modelo_ml.fit(X_ml, {'ingresos': y_ing, 'costos': y_cost, 'utilidad': y_util},
                        epochs=15, batch_size=32, verbose=0)
print(f"Pérdidas finales: ingresos={hist_ml.history['ingresos_loss'][-1]:.4f}, "
      f"costos={hist_ml.history['costos_loss'][-1]:.4f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: Modelo con múltiples pérdidas (loss_weights) para ventas y costos.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: Modelo con múltiples métricas por salida

```python
input_mm = tf.keras.layers.Input(shape=(4,))
x = tf.keras.layers.Dense(16, activation='relu')(input_mm)
out_precio = tf.keras.layers.Dense(1, name='precio')(x)
out_cantidad = tf.keras.layers.Dense(1, name='cantidad')(x)

modelo_mm = tf.keras.Model(inputs=input_mm, outputs=[out_precio, out_cantidad])
modelo_mm.compile(
    optimizer='adam',
    loss='mse',
    metrics={
        'precio': ['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')],
        'cantidad': ['mae', 'mse']
    }
)
X_mm = np.random.rand(300, 4).astype(np.float32)
y_precio = np.random.rand(300, 1).astype(np.float32)
y_cant = np.random.rand(300, 1).astype(np.float32)
hist_mm = modelo_mm.fit(X_mm, {'precio': y_precio, 'cantidad': y_cant},
                        epochs=10, batch_size=32, verbose=0)
print(f"Métricas disponibles: {list(hist_mm.history.keys())}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: Modelo con múltiples métricas por salida.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: Visualizar arquitectura con tf.keras.utils.plot_model

```python
input_v = tf.keras.layers.Input(shape=(3,), name='entrada')
x = tf.keras.layers.Dense(16, activation='relu', name='oculta1')(input_v)
x = tf.keras.layers.Dense(8, activation='relu', name='oculta2')(x)
output_v = tf.keras.layers.Dense(1, name='salida')(x)

modelo_viz = tf.keras.Model(inputs=input_v, outputs=output_v)

# Generar plot (requiere graphviz/pydot)
try:
    tf.keras.utils.plot_model(modelo_viz, show_shapes=True, show_layer_names=True)
    print("Diagrama generado exitosamente")
except Exception as e:
    print(f"No se pudo generar el diagrama (requiere graphviz): {e}")

print(f"Capas del modelo: {[layer.name for layer in modelo_viz.layers]}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: Visualizar arquitectura con tf.keras.utils.plot_model.*

1. Generar plot (requiere graphviz/pydot)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 16: model.summary en modelo funcional multi-input

```python
in_a = tf.keras.layers.Input(shape=(4,), name='input_A')
in_b = tf.keras.layers.Input(shape=(6,), name='input_B')
x_a = tf.keras.layers.Dense(8, activation='relu', name='dense_A')(in_a)
x_b = tf.keras.layers.Dense(8, activation='relu', name='dense_B')(in_b)
concat = tf.keras.layers.concatenate([x_a, x_b], name='concat')
x = tf.keras.layers.Dense(8, activation='relu', name='dense_fusion')(concat)
out = tf.keras.layers.Dense(1, name='output')(x)
modelo_sum_f = tf.keras.Model(inputs=[in_a, in_b], outputs=out)
modelo_sum_f.summary()
print(f"Total de parámetros: {modelo_sum_f.count_params()}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: model.summary en modelo funcional multi-input.*

1. `print(f"Total de parámetros: {modelo_sum_f.count_params()}")` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 17: Entrenar con diferentes learning rates por capa

```python
input_lr = tf.keras.layers.Input(shape=(5,))

# Capas con nombres para acceder a ellas
dense_a = tf.keras.layers.Dense(16, activation='relu', name='capa_a')
dense_b = tf.keras.layers.Dense(8, activation='relu', name='capa_b')
dense_out = tf.keras.layers.Dense(1, name='capa_out')

x = dense_a(input_lr)
x = dense_b(x)
output_lr = dense_out(x)

modelo_lr = tf.keras.Model(inputs=input_lr, outputs=output_lr)

# Learning rates diferenciales
optimizer_lr = tf.keras.optimizers.Adam(learning_rate=0.001)
modelo_lr.compile(optimizer=optimizer_lr, loss='mse')

# Asignar LR diferente a capas específicas
tf.keras.backend.set_value(dense_a.optimizer.lr, 0.0001)
print(f"Learning rates: capa_a={tf.keras.backend.get_value(dense_a.optimizer.lr) if hasattr(dense_a, 'optimizer') else 'compilar primero'}")

# Alternativa: usar pesos congelados
dense_a.trainable = False
print(f"Capa_a entrenable: {dense_a.trainable}")
modelo_lr.compile(optimizer='adam', loss='mse')  # re-compilar necesario

X_lr = np.random.rand(200, 5).astype(np.float32)
y_lr = np.random.rand(200, 1).astype(np.float32)
modelo_lr.fit(X_lr, y_lr, epochs=10, batch_size=32, verbose=0)
print("Modelo con capas congeladas entrenado")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: Entrenar con diferentes learning rates por capa.*

1. Capas con nombres para acceder a ellas
2. Learning rates diferenciales
3. Asignar LR diferente a capas específicas
4. Alternativa: usar pesos congelados

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 18: Integrador — Modelo multi-input multi-output para ventas completo

```python
# Entrada 1: features del producto (precio, costo, margen)
input_prod = tf.keras.layers.Input(shape=(3,), name='producto')
x_prod = tf.keras.layers.Dense(16, activation='relu')(input_prod)
x_prod = tf.keras.layers.Dropout(0.2)(x_prod)

# Entrada 2: features de temporada (mes, día_semana, es_festivo, promoción)
input_temp = tf.keras.layers.Input(shape=(4,), name='temporada')
x_temp = tf.keras.layers.Dense(8, activation='relu')(input_temp)

# Entrada 3: features del cliente (frecuencia, ticket_promedio, antigüedad)
input_cli = tf.keras.layers.Input(shape=(3,), name='cliente')
x_cli = tf.keras.layers.Dense(8, activation='relu')(input_cli)

# Fusión
fusion_mio = tf.keras.layers.concatenate([x_prod, x_temp, x_cli])
x = tf.keras.layers.Dense(32, activation='relu')(fusion_mio)
x = tf.keras.layers.Dense(16, activation='relu')(x)

# Salidas
out_venta = tf.keras.layers.Dense(1, name='prob_venta', activation='sigmoid')(x)
out_cantidad = tf.keras.layers.Dense(1, name='cantidad_estimada')(x)
out_margen = tf.keras.layers.Dense(1, name='margen_estimado')(x)

modelo_integrador = tf.keras.Model(
    inputs=[input_prod, input_temp, input_cli],
    outputs=[out_venta, out_cantidad, out_margen]
)
modelo_integrador.compile(
    optimizer='adam',
    loss={'prob_venta': 'binary_crossentropy', 'cantidad_estimada': 'mse', 'margen_estimado': 'mse'},
    loss_weights={'prob_venta': 0.5, 'cantidad_estimada': 0.25, 'margen_estimado': 0.25},
    metrics={'prob_venta': 'accuracy'}
)

# Datos sintéticos
n_mio = 2000
Xp = np.random.rand(n_mio, 3).astype(np.float32)
Xt = np.random.rand(n_mio, 4).astype(np.float32)
Xc = np.random.rand(n_mio, 3).astype(np.float32)
yv = np.random.randint(0, 2, (n_mio, 1)).astype(np.float32)
yq = np.random.rand(n_mio, 1).astype(np.float32)
ym = np.random.rand(n_mio, 1).astype(np.float32)

split = 1600
hist_int = modelo_integrador.fit(
    [Xp[:split], Xt[:split], Xc[:split]],
    {'prob_venta': yv[:split], 'cantidad_estimada': yq[:split], 'margen_estimado': ym[:split]},
    epochs=30, batch_size=64, validation_split=0.2, verbose=0
)
print(f"Train loss final: {hist_int.history['loss'][-1]:.4f}")
eval_int = modelo_integrador.evaluate(
    [Xp[split:], Xt[split:], Xc[split:]],
    {'prob_venta': yv[split:], 'cantidad_estimada': yq[split:], 'margen_estimado': ym[split:]},
    verbose=0
)
print(f"Test loss: {eval_int[0]:.4f}")
modelo_integrador.summary()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Integrador — Modelo multi-input multi-output para ventas completo.*

1. Entrada 1: features del producto (precio, costo, margen)
2. Entrada 2: features de temporada (mes, día_semana, es_festivo, promoción)
3. Entrada 3: features del cliente (frecuencia, ticket_promedio, antigüedad)
4. Fusión
5. Salidas
6. Datos sintéticos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Ejercicios

1. **Modelo funcional para clasificación de productos**: Crea un modelo funcional con Input(shape=(4,)), dos capas Dense ocultas (32, 16 con ReLU) y salida softmax de 3 clases para clasificar productos en económica/estándar/premium.

2. **Multi-input con precios históricos**: Crea un modelo con dos entradas (precios_7días de forma (7,) y features_estáticas de forma (5,)). Concaténalas tras capas densas individuales y predice el precio del día siguiente.

3. **Multi-output para proveedores**: Crea un modelo que prediga simultáneamente (a) lead_time en días y (b) calidad_esperada (0-1). Usa loss_weights para dar más importancia a la calidad.

4. **Skip connection profunda**: Implementa un modelo con 3 bloques residuales (cada bloque: Dense → ReLU → Dense → Add con la entrada del bloque). Entrénalo para predecir rotación de inventario.

5. **Capa compartida para similitud de productos**: Crea una capa Dense compartida que procese las features de dos productos. Calcula la distancia coseno entre sus representaciones (usa `tf.keras.layers.dot(axes=1, normalize=True)`).

6. **Custom Layer de umbral**: Crea una capa personalizada `UmbralVentas` que aplique un umbral entrenable: si el valor < umbral → 0, si no → valor. Úsala en un modelo de predicción de ventas.

7. **Subclassing con atención simple**: Implementa un modelo vía Subclassing que tenga una capa de atención sobre features temporales (8 pasos × 4 features). Usa `training` flag para batch normalization.

8. **Integrador multi-input multi-output**: Diseña un modelo que reciba (features producto 4, features cliente 3, features temporales 6) y prediga (probabilidad_compra binaria, monto_estimado regresión). Incluye dropout y batch normalization. Reporta métricas de test.

---

## Resumen

La Functional API de tf.keras permite modelos con topologías complejas y flexibles para problemas reales de ventas, compras e inventarios:

- **Multi-input**: Combinar datos numéricos, categóricos, textuales y temporales en un mismo modelo. Esencial cuando los datos provienen de diferentes fuentes (ERP, CRM, POS).
- **Multi-output**: Predecir múltiples objetivos simultáneamente (cantidad, margen, probabilidad). Reduce el costo computacional y aprovecha representaciones compartidas.
- **Skip connections**: Sumar (add) o concatenar salidas de capas tempranas con capas tardías. Estabiliza el entrenamiento en redes profundas.
- **Capas compartidas**: Reutilizar pesos en múltiples entradas (misma red evaluando diferentes proveedores, productos, sucursales).
- **Capas de combinación**: `concatenate`, `add`, `multiply`, `subtract`, `average`, `maximum`, `minimum` para fusionar ramas de formas diversas.
- **Subclassing API**: Control total sobre el forward pass con `call(self, inputs, training, mask)`. Útil para arquitecturas dinámicas.
- **Custom Layer**: Crear capas con pesos entrenables propios, con `build` (crear pesos), `call` (forward), `get_config`/`from_config` (serialización).
- **Modelo ensamblado**: Promediar múltiples modelos independientes para reducir varianza y mejorar generalización.
- **Pérdidas y métricas múltiples**: Asignar diferentes pesos a diferentes salidas, combinando objetivos de negocio.

La Functional API es indispensable cuando Sequential se queda corto, particularmente en escenarios de datos heterogéneos y objetivos múltiples comunes en analytics de ventas y operaciones.
