# CP15: Optimización de Precios con SciPy

## Contexto de Negocio
El equipo de pricing busca determinar el precio óptimo que maximice el beneficio total. Utilizaremos modelos de elasticidad de demanda y optimización numérica con SciPy para encontrar el punto de precio que equilibre volumen e ingreso.

```python
# ============================================================
# 1. CARGA Y ANÁLISIS DE RELACIÓN PRECIO-CANTIDAD
# ============================================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats, optimize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.figsize": (10, 6), "font.size": 12})

ventas = pd.read_csv("../datos/ventas.csv")
print("Dimensiones:", ventas.shape)
print("\nPrimeras filas:")
ventas.head()
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

*Contexto de Negocio.*

1. ============================================================
2. 1. CARGA Y ANÁLISIS DE RELACIÓN PRECIO-CANTIDAD
3. ============================================================

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# Explorar columnas relacionadas con precio y cantidad
print("Columnas disponibles:", ventas.columns.tolist())

# Identificar columnas de precio y cantidad
col_precio = None
for col in ["precio", "precio_unitario", "costo", "price", "pvp"]:
    if col in ventas.columns:
        col_precio = col
        break

col_cantidad = None
for col in ["cantidad", "qty", "units", "quantity", "num_unidades"]:
    if col in ventas.columns:
        col_cantidad = col
        break

col_producto = None
for col in ["producto", "producto_id", "item", "sku"]:
    if col in ventas.columns:
        col_producto = col
        break

print(f"Columna de precio: {col_precio}")
print(f"Columna de cantidad: {col_cantidad}")
print(f"Columna de producto: {col_producto}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Explorar columnas relacionadas con precio y cantidad
2. Identificar columnas de precio y cantidad

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# Si no existen las columnas exactas, crear datos simulados realistas
if col_precio is None or col_cantidad is None:
    print("Creando datos simulados de precio-demanda...")
    np.random.seed(42)

    # Relacion precio-demanda: a mayor precio, menor cantidad demandada
    # Usamos una funcion de demanda lineal con ruido
    precios = np.linspace(10, 100, 200)
    demanda_real = 1000 - 8 * precios
    demanda = demanda_real + np.random.normal(0, 50, size=len(precios))
    demanda = np.maximum(demanda, 0)

    ventas_demanda = pd.DataFrame({
        "precio": precios,
        "cantidad": demanda.astype(int),
        "producto": "Producto_X",
        "costo_unitario": 15 + np.random.normal(0, 2, size=len(precios))
    })
    col_precio = "precio"
    col_cantidad = "cantidad"
    col_producto = "producto"
else:
    ventas_demanda = ventas.groupby([col_producto, col_precio])[col_cantidad].sum().reset_index()
    if "costo_unitario" not in ventas_demanda.columns:
        ventas_demanda["costo_unitario"] = ventas_demanda[col_precio] * 0.4

print("Datos de demanda preparados:")
print(ventas_demanda.head())
print(f"\nRango de precios: ${ventas_demanda[col_precio].min():.2f} - ${ventas_demanda[col_precio].max():.2f}")
print(f"Rango de cantidades: {ventas_demanda[col_cantidad].min()} - {ventas_demanda[col_cantidad].max()}")
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

*Este ejemplo.*

1. Si no existen las columnas exactas, crear datos simulados realistas
2. Relacion precio-demanda: a mayor precio, menor cantidad demandada
3. Usamos una funcion de demanda lineal con ruido

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# Visualizar relacion precio-cantidad
plt.figure(figsize=(10, 6))
sns.scatterplot(x=col_precio, y=col_cantidad, data=ventas_demanda,
                alpha=0.6, color="steelblue", s=80)
plt.title("Relacion Precio vs Cantidad Demandada", fontsize=14)
plt.xlabel("Precio ($)")
plt.ylabel("Cantidad Demandada")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Visualizar relacion precio-cantidad

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 2. CALCULAR ELASTICIDAD PRECIO DE LA DEMANDA
# ============================================================
# Elasticidad = (% cambio cantidad) / (% cambio precio)
# Se estima con regresion log-log: ln(Q) = B0 + B1 ln(P)
# B1 es la elasticidad

precio = ventas_demanda[col_precio].values
cantidad = ventas_demanda[col_cantidad].values

log_precio = np.log(precio)
log_cantidad = np.log(cantidad + 1e-10)

slope, intercept, r_value, p_value, std_err = stats.linregress(log_precio, log_cantidad)
elasticidad = slope

print(f"ELASTICIDAD PRECIO DE LA DEMANDA:")
print(f"Elasticidad estimada: {elasticidad:.4f}")
print(f"R2 de la regresion log-log: {r_value**2:.4f}")
print(f"p-value: {p_value:.6f}")
print(f"Error estandar: {std_err:.4f}")

if elasticidad < -1:
    print("-> Demanda ELASTICA: los consumidores responden fuertemente a cambios de precio.")
    print("  Reducir precio puede aumentar ingreso total.")
elif elasticidad < 0:
    print("-> Demanda INELASTICA: los consumidores responden debilmente a cambios de precio.")
    print("  Aumentar precio puede aumentar ingreso total.")
else:
    print("-> Demanda con elasticidad positiva (inusual). Revisar datos.")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 2. CALCULAR ELASTICIDAD PRECIO DE LA DEMANDA
3. ============================================================
4. Elasticidad = (% cambio cantidad) / (% cambio precio)
5. Se estima con regresion log-log: ln(Q) = B0 + B1 ln(P)
6. B1 es la elasticidad

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# Visualizar regresion log-log
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Escala original
axes[0].scatter(precio, cantidad, alpha=0.6, color="steelblue", s=60)
precio_teorico = np.linspace(precio.min(), precio.max(), 100)
cantidad_teorica = np.exp(intercept + slope * np.log(precio_teorico))
axes[0].plot(precio_teorico, cantidad_teorica, color="red", lw=2,
             label=f"Q = e^({intercept:.2f}) x P^({slope:.2f})")
axes[0].set_xlabel("Precio ($)")
axes[0].set_ylabel("Cantidad Demandada")
axes[0].set_title("Curva de Demanda (escala original)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Escala log-log
axes[1].scatter(log_precio, log_cantidad, alpha=0.6, color="coral", s=60)
axes[1].plot(log_precio, intercept + slope * log_precio,
             color="red", lw=2, label=f"ln(Q) = {intercept:.2f} + {slope:.2f} ln(P)")
axes[1].set_xlabel("ln(Precio)")
axes[1].set_ylabel("ln(Cantidad)")
axes[1].set_title("Regresion Log-Log (elasticidad)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Visualizar regresion log-log
2. Escala original
3. Escala log-log

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 3. AJUSTAR CURVA DE DEMANDA
# ============================================================
# Modelo lineal: Q = a - b*P
# Modelo log-lineal: ln(Q) = a + b*ln(P) -> Q = e^a * P^b

lr_demanda = LinearRegression()
lr_demanda.fit(precio.reshape(-1, 1), cantidad)
a_lineal = lr_demanda.intercept_
b_lineal = -lr_demanda.coef_[0]

a_log = intercept
b_log = slope

print("MODELOS DE DEMANDA AJUSTADOS:")
print(f"\nLineal: Q(P) = {a_lineal:.2f} - {b_lineal:.2f} x P")
print(f"  R2: {r2_score(cantidad, lr_demanda.predict(precio.reshape(-1, 1))):.4f}")
print(f"\nLog-Lineal: Q(P) = e^({a_log:.2f}) x P^({b_log:.2f})")
print(f"  R2: {r_value**2:.4f}")

r2_lineal = r2_score(cantidad, lr_demanda.predict(precio.reshape(-1, 1)))
print(f"\nComparacion: R2 Lineal = {r2_lineal:.4f} vs R2 Log-Lineal = {r_value**2:.4f}")
mejor_modelo = "Log-Lineal" if r_value**2 > r2_lineal else "Lineal"
print(f"-> Mejor modelo: {mejor_modelo}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 3. AJUSTAR CURVA DE DEMANDA
3. ============================================================
4. Modelo lineal: Q = a - b*P
5. Modelo log-lineal: ln(Q) = a + b*ln(P) -> Q = e^a * P^b

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# Visualizar ambos ajustes
plt.figure(figsize=(10, 6))
plt.scatter(precio, cantidad, alpha=0.5, color="steelblue", s=60, label="Datos reales")

precio_lin = np.linspace(precio.min(), precio.max(), 100)
cantidad_lin = a_lineal - b_lineal * precio_lin
plt.plot(precio_lin, cantidad_lin, color="red", lw=2, label=f"Lineal (R2={r2_lineal:.3f})")

cantidad_log = np.exp(a_log + b_log * np.log(precio_lin))
plt.plot(precio_lin, cantidad_log, color="green", lw=2,
         label=f"Log-Lineal (R2={r_value**2:.3f})")

plt.xlabel("Precio ($)")
plt.ylabel("Cantidad Demandada")
plt.title("Ajuste de Curvas de Demanda", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Visualizar ambos ajustes

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 4. DEFINIR FUNCION DE INGRESO
# ============================================================
# Ingreso(P) = P x Demanda(P)
# Usamos el mejor modelo (log-lineal)

def demanda_loglineal(P):
    return np.exp(a_log + b_log * np.log(P + 1e-10))

def ingreso(P):
    return P * demanda_loglineal(P)

# Probar la funcion en un rango de precios
precios_prueba = np.linspace(5, 150, 50)
ingresos_prueba = ingreso(precios_prueba)

print("FUNCION DE INGRESO:")
print(f"  Ingreso(P) = P x exp({a_log:.4f} + {b_log:.4f} x ln(P))")
print(f"\n  Precio minimo: ${precios_prueba[0]:.2f} -> Ingreso: ${ingresos_prueba[0]:,.2f}")
print(f"  Precio maximo: ${precios_prueba[-1]:.2f} -> Ingreso: ${ingresos_prueba[-1]:,.2f}")
print(f"  Precio con maximo ingreso aproximado: ${precios_prueba[np.argmax(ingresos_prueba)]:.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 4. DEFINIR FUNCION DE INGRESO
3. ============================================================
4. Ingreso(P) = P x Demanda(P)
5. Usamos el mejor modelo (log-lineal)
6. Probar la funcion en un rango de precios

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# Visualizar funcion de ingreso
plt.figure(figsize=(10, 6))
plt.plot(precios_prueba, ingresos_prueba, color="steelblue", lw=2.5)
plt.xlabel("Precio ($)")
plt.ylabel("Ingreso Total ($)")
plt.title("Funcion de Ingreso: I(P) = P x Q(P)", fontsize=14)
plt.grid(True, alpha=0.3)
idx_max = np.argmax(ingresos_prueba)
plt.scatter(precios_prueba[idx_max], ingresos_prueba[idx_max],
            color="red", s=150, zorder=5, label=f"Max ingreso: ${ingresos_prueba[idx_max]:,.0f}")
plt.legend()
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Visualizar funcion de ingreso

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 5. DEFINIR FUNCION DE COSTO
# ============================================================
# Costo(P) = CostoUnitario x Demanda(P)

costo_unitario_promedio = ventas_demanda["costo_unitario"].mean()

def costo(P):
    return costo_unitario_promedio * demanda_loglineal(P)

print(f"Costo unitario promedio: ${costo_unitario_promedio:.2f}")
print(f"  Costo(P) = ${costo_unitario_promedio:.2f} x exp({a_log:.4f} + {b_log:.4f} x ln(P))")

costos_prueba = costo(precios_prueba)
print(f"\n  Precio minimo: ${precios_prueba[0]:.2f} -> Costo: ${costos_prueba[0]:,.2f}")
print(f"  Precio maximo: ${precios_prueba[-1]:.2f} -> Costo: ${costos_prueba[-1]:,.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 5. DEFINIR FUNCION DE COSTO
3. ============================================================
4. Costo(P) = CostoUnitario x Demanda(P)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 6. DEFINIR FUNCION DE BENEFICIO
# ============================================================
def beneficio(P):
    return ingreso(P) - costo(P)

beneficios_prueba = beneficio(precios_prueba)
idx_max_benef = np.argmax(beneficios_prueba)
precio_aprox_optimo = precios_prueba[idx_max_benef]

print("FUNCION DE BENEFICIO:")
print(f"  Beneficio(P) = P x Q(P) - {costo_unitario_promedio:.2f} x Q(P)")
print(f"  Beneficio(P) = (P - {costo_unitario_promedio:.2f}) x Q(P)")
print(f"\nRango de beneficios:")
print(f"  Precio minimo: ${precios_prueba[0]:.2f} -> Beneficio: ${beneficios_prueba[0]:,.2f}")
print(f"  Precio maximo: ${precios_prueba[-1]:.2f} -> Beneficio: ${beneficios_prueba[-1]:,.2f}")
print(f"  Precio aproximado optimo: ${precio_aprox_optimo:.2f} -> Beneficio: ${beneficios_prueba[idx_max_benef]:,.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 6. DEFINIR FUNCION DE BENEFICIO
3. ============================================================

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# Visualizar Ingreso, Costo y Beneficio
plt.figure(figsize=(12, 6))
plt.plot(precios_prueba, ingresos_prueba, label="Ingreso", color="steelblue", lw=2)
plt.plot(precios_prueba, costos_prueba, label="Costo", color="coral", lw=2)
plt.plot(precios_prueba, beneficios_prueba, label="Beneficio", color="seagreen", lw=2.5)
plt.axhline(y=0, color="black", alpha=0.3)
plt.axvline(x=precio_aprox_optimo, color="red", linestyle="--", alpha=0.7, label=f"P aproximado = ${precio_aprox_optimo:.2f}")
plt.xlabel("Precio ($)")
plt.ylabel("$")
plt.title("Ingreso, Costo y Beneficio vs Precio", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Visualizar Ingreso, Costo y Beneficio

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 7. OPTIMIZACION CON MINIMIZE_SCALAR
# ============================================================
# minimize_scalar minimiza, asi que usamos -beneficio

resultado = optimize.minimize_scalar(
    lambda P: -beneficio(P),
    bounds=(costo_unitario_promedio * 1.1, precios_prueba[-1]),
    method="bounded"
)

precio_optimo_scalar = resultado.x
beneficio_optimo_scalar = -resultado.fun

print("OPTIMIZACION CON MINIMIZE_SCALAR:")
print(f"Metodo: bounded")
print(f"Precio optimo encontrado: ${precio_optimo_scalar:.2f}")
print(f"Beneficio maximo: ${beneficio_optimo_scalar:,.2f}")
print(f"Cantidad al precio optimo: {demanda_loglineal(precio_optimo_scalar):.0f} unidades")
print(f"Ingreso al precio optimo: ${ingreso(precio_optimo_scalar):,.2f}")
print(f"Costo al precio optimo: ${costo(precio_optimo_scalar):,.2f}")
print(f"Margen: {(precio_optimo_scalar - costo_unitario_promedio) / precio_optimo_scalar * 100:.1f}%")
print(f"Estatus de optimizacion: {resultado.success}")
print(f"Iteraciones: {resultado.nfev}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 7. OPTIMIZACION CON MINIMIZE_SCALAR
3. ============================================================
4. minimize_scalar minimiza, asi que usamos -beneficio

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 8. OPTIMIZACION CON RESTRICCIONES (SLSQP)
# ============================================================
# Usar minimize con restriccion: precio > costo_unitario

def beneficio_negativo(P):
    return -beneficio(P[0])

def restriccion_precio(P):
    return P[0] - costo_unitario_promedio * 1.05  # Precio >= 105% del costo

restricciones = [{"type": "ineq", "fun": restriccion_precio}]
limites = [(costo_unitario_promedio * 1.05, precios_prueba[-1] * 2)]

resultado_constr = optimize.minimize(
    beneficio_negativo,
    x0=[precio_aprox_optimo],
    bounds=limites,
    constraints=restricciones,
    method="SLSQP"
)

precio_optimo_constr = resultado_constr.x[0]
beneficio_optimo_constr = -resultado_constr.fun

print("OPTIMIZACION CON RESTRICCIONES (SLSQP):")
print(f"Precio optimo: ${precio_optimo_constr:.2f}")
print(f"Beneficio maximo: ${beneficio_optimo_constr:,.2f}")
print(f"Restriccion: precio >= ${costo_unitario_promedio * 1.05:.2f} (105% del costo)")
print(f"Restriccion cumplida: {restriccion_precio([precio_optimo_constr]) >= 0}")
print(f"Estatus: {resultado_constr.success}")
print(f"Iteraciones: {resultado_constr.nit}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 8. OPTIMIZACION CON RESTRICCIONES (SLSQP)
3. ============================================================
4. Usar minimize con restriccion: precio > costo_unitario

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# Comparar ambos metodos de optimizacion
print("\nCOMPARACION DE METODOS:")
print("-" * 50)
print(f"{'Metodo':20s} {'Precio optimo':>15s} {'Beneficio':>15s}")
print("-" * 50)
print(f"{'minimize_scalar':20s} ${precio_optimo_scalar:>10.2f} ${beneficio_optimo_scalar:>10,.0f}")
print(f"{'minimize (SLSQP)':20s} ${precio_optimo_constr:>10.2f} ${beneficio_optimo_constr:>10,.0f}")
print("-" * 50)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Comparar ambos metodos de optimizacion

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 9. SENSIBILIDAD: PROBAR DIFERENTES ELASTICIDADES
# ============================================================
# La elasticidad tiene incertidumbre. Probamos distintos escenarios.
elasticidad_baja = b_log * 0.8
elasticidad_alta = b_log * 1.2

escenarios = {
    "Baja elasticidad (menos sensible)": elasticidad_baja,
    "Elasticidad estimada": b_log,
    "Alta elasticidad (mas sensible)": elasticidad_alta
}

resultados_sensibilidad = []
for nombre, elasticidad_esc in escenarios.items():
    def demanda_esc(P, e=elasticidad_esc):
        return np.exp(a_log + e * np.log(P + 1e-10))

    def beneficio_esc(P, e=elasticidad_esc):
        return P * demanda_esc(P, e) - costo_unitario_promedio * demanda_esc(P, e)

    res = optimize.minimize_scalar(
        lambda P, e=elasticidad_esc: -beneficio_esc(P, e),
        bounds=(costo_unitario_promedio * 1.1, precios_prueba[-1]),
        method="bounded"
    )

    resultados_sensibilidad.append({
        "Escenario": nombre,
        "Elasticidad": elasticidad_esc,
        "Precio Optimo": res.x,
        "Beneficio": -res.fun,
        "Cantidad": demanda_esc(res.x)
    })

df_sensibilidad = pd.DataFrame(resultados_sensibilidad)
print("ANALISIS DE SENSIBILIDAD - ELASTICIDAD:")
print(df_sensibilidad.round(2).to_string(index=False))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 9. SENSIBILIDAD: PROBAR DIFERENTES ELASTICIDADES
3. ============================================================
4. La elasticidad tiene incertidumbre. Probamos distintos escenarios.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# Visualizar sensibilidad
plt.figure(figsize=(12, 6))
colores = {"Baja elasticidad (menos sensible)": "green",
           "Elasticidad estimada": "steelblue",
           "Alta elasticidad (mas sensible)": "red"}

for nombre, elasticidad_esc in escenarios.items():
    def demanda_esc(P, e=elasticidad_esc):
        return np.exp(a_log + e * np.log(P + 1e-10))

    def beneficio_esc(P, e=elasticidad_esc):
        return P * demanda_esc(P, e) - costo_unitario_promedio * demanda_esc(P, e)

    beneficios_sens = [beneficio_esc(p, elasticidad_esc) for p in precios_prueba]
    plt.plot(precios_prueba, beneficios_sens, label=nombre,
             color=colores[nombre], lw=2)
    idx_opt = np.argmax(beneficios_sens)
    plt.scatter(precios_prueba[idx_opt], beneficios_sens[idx_opt],
                color=colores[nombre], s=100, zorder=5,
                edgecolors="black", linewidth=1.5)

plt.xlabel("Precio ($)")
plt.ylabel("Beneficio Total ($)")
plt.title("Analisis de Sensibilidad: Beneficio vs Precio", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Visualizar sensibilidad

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 10. VISUALIZAR PRECIO VS BENEFICIO CON PUNTO OPTIMO
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Panel 1: Ingreso, Costo y Beneficio
precios_viz = np.linspace(costo_unitario_promedio * 1.1, precios_prueba[-1], 200)
ingresos_viz = ingreso(precios_viz)
costos_viz = costo(precios_viz)
beneficios_viz = beneficio(precios_viz)

axes[0].plot(precios_viz, ingresos_viz, label="Ingreso", color="steelblue", lw=2)
axes[0].plot(precios_viz, costos_viz, label="Costo", color="coral", lw=2)
axes[0].plot(precios_viz, beneficios_viz, label="Beneficio", color="seagreen", lw=2.5)
axes[0].axvline(x=precio_optimo_scalar, color="red", linestyle="--",
                alpha=0.7, label=f"P* = ${precio_optimo_scalar:.2f}")
axes[0].axhline(y=0, color="black", alpha=0.3)
axes[0].set_xlabel("Precio ($)")
axes[0].set_ylabel("$")
axes[0].set_title("Ingreso, Costo y Beneficio vs Precio", fontsize=13)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Panel 2: Beneficio con punto optimo destacado
axes[1].plot(precios_viz, beneficios_viz, color="seagreen", lw=2.5)
axes[1].scatter([precio_optimo_scalar], [beneficio_optimo_scalar],
                color="red", s=200, zorder=5, edgecolors="black",
                linewidth=2, label=f"Optimo: ${precio_optimo_scalar:.2f}")
axes[1].annotate(f"P* = ${precio_optimo_scalar:.2f}\nBen = ${beneficio_optimo_scalar:,.0f}",
                 xy=(precio_optimo_scalar, beneficio_optimo_scalar),
                 xytext=(precio_optimo_scalar * 1.2, beneficio_optimo_scalar * 0.9),
                 arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
                 fontsize=11, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
axes[1].set_xlabel("Precio ($)")
axes[1].set_ylabel("Beneficio ($)")
axes[1].set_title("Funcion de Beneficio con Punto Optimo", fontsize=13)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 10. VISUALIZAR PRECIO VS BENEFICIO CON PUNTO OPTIMO
3. ============================================================
4. Panel 1: Ingreso, Costo y Beneficio
5. Panel 2: Beneficio con punto optimo destacado

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 11. COMPARAR PRECIO ACTUAL VS PRECIO OPTIMO
# ============================================================
precio_actual = ventas_demanda[col_precio].mean()
cantidad_actual = ventas_demanda[col_cantidad].sum()
ingreso_actual = (ventas_demanda[col_precio] * ventas_demanda[col_cantidad]).sum()
costo_actual = costo_unitario_promedio * cantidad_actual
beneficio_actual = ingreso_actual - costo_actual

cantidad_optima = demanda_loglineal(precio_optimo_scalar)
ingreso_optimo = precio_optimo_scalar * cantidad_optima
costo_optimo = costo_unitario_promedio * cantidad_optima

print("=" * 60)
print("COMPARACION: PRECIO ACTUAL VS PRECIO OPTIMO")
print("=" * 60)

comparacion_precios = pd.DataFrame({
    "Metrica": ["Precio Promedio", "Cantidad Total", "Ingreso Total",
                "Costo Total", "Beneficio Total", "Margen %"],
    "Actual": [f"${precio_actual:.2f}", f"{cantidad_actual:,.0f}",
               f"${ingreso_actual:,.0f}", f"${costo_actual:,.0f}",
               f"${beneficio_actual:,.0f}",
               f"{(precio_actual - costo_unitario_promedio) / precio_actual * 100:.1f}%"],
    "Optimo": [f"${precio_optimo_scalar:.2f}", f"{cantidad_optima:,.0f}",
               f"${ingreso_optimo:,.0f}", f"${costo_optimo:,.0f}",
               f"${beneficio_optimo_scalar:,.0f}",
               f"{(precio_optimo_scalar - costo_unitario_promedio) / precio_optimo_scalar * 100:.1f}%"],
    "Cambio": [f"{((precio_optimo_scalar - precio_actual) / precio_actual) * 100:+.1f}%",
               f"{((cantidad_optima - cantidad_actual) / cantidad_actual) * 100:+.1f}%",
               f"{((ingreso_optimo - ingreso_actual) / ingreso_actual) * 100:+.1f}%",
               f"{((costo_optimo - costo_actual) / costo_actual) * 100:+.1f}%",
               f"{((beneficio_optimo_scalar - beneficio_actual) / beneficio_actual) * 100:+.1f}%",
               f"{((precio_optimo_scalar - costo_unitario_promedio) / precio_optimo_scalar - (precio_actual - costo_unitario_promedio) / precio_actual) * 100:+.1f}pp"]
})

print(comparacion_precios.to_string(index=False))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 11. COMPARAR PRECIO ACTUAL VS PRECIO OPTIMO
3. ============================================================

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# Grafico de comparacion
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metricas = ["Precio", "Cantidad", "Beneficio"]
valores_actual = [precio_actual, cantidad_actual, beneficio_actual]
valores_optimo = [precio_optimo_scalar, cantidad_optima, beneficio_optimo_scalar]
colores_comp = [["steelblue", "coral"], ["seagreen", "gold"], ["purple", "orange"]]

for i, (ax, metrica, v_act, v_opt, col) in enumerate(zip(
    axes, metricas, valores_actual, valores_optimo, colores_comp)):

    x = np.arange(2)
    bars = ax.bar(x, [v_act, v_opt], width=0.4, color=col, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(["Actual", "Optimo"])
    ax.set_title(f"{metrica}", fontsize=13)

    for bar, val in zip(bars, [v_act, v_opt]):
        if metrica == "Precio":
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(valores_actual)*0.02,
                    f"${val:.2f}", ha="center", fontsize=10)
        elif metrica == "Cantidad":
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(valores_actual)*0.02,
                    f"{val:,.0f}", ha="center", fontsize=10)
        else:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(abs(v_act), abs(v_opt))*0.02,
                    f"${val:,.0f}", ha="center", fontsize=10)

    ax.grid(True, alpha=0.3, axis="y")

plt.suptitle("Comparacion: Precio Actual vs Precio Optimo", fontsize=14)
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. Grafico de comparacion

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



```python
# ============================================================
# 12. RECOMENDACIONES DE ESTRATEGIA DE PRECIOS
# ============================================================
print("=" * 80)
print("RECOMENDACIONES DE ESTRATEGIA DE PRECIOS")
print("=" * 80)

cambio_precio_pct = (precio_optimo_scalar - precio_actual) / precio_actual * 100
cambio_beneficio_pct = (beneficio_optimo_scalar - beneficio_actual) / beneficio_actual * 100

direccion = "AUMENTAR" if cambio_precio_pct > 0 else "REDUCIR" if cambio_precio_pct < 0 else "MANTENER"

print(f"""
--- RESUMEN DE HALLAZGOS ---
* Elasticidad precio de la demanda: {elasticidad:.4f}
* Precio actual promedio: ${precio_actual:.2f}
* Precio optimo recomendado: ${precio_optimo_scalar:.2f}
* Cambio de precio sugerido: {cambio_precio_pct:+.1f}%
* Beneficio actual: ${beneficio_actual:,.0f}
* Beneficio optimo estimado: ${beneficio_optimo_scalar:,.0f}
* Mejora en beneficio: {cambio_beneficio_pct:+.1f}%

--- RECOMENDACIONES ---

1. AJUSTE DE PRECIO - {direccion}
   Se recomienda {'aumentar' if cambio_precio_pct > 0 else 'reducir'} el precio en un {abs(cambio_precio_pct):.1f}%
   para maximizar el beneficio.

2. IMPLEMENTACION GRADUAL
   - Fase 1 (semanas 1-2): Ajustar precio en {abs(cambio_precio_pct)/3:.1f}% y monitorear demanda
   - Fase 2 (semanas 3-4): Ajustar otro {abs(cambio_precio_pct)/3:.1f}% si la respuesta es favorable
   - Fase 3 (semanas 5-6): Completar ajuste al precio optimo

3. SEGMENTACION DE PRECIOS
   - Clientes VIP: precio optimo (menos sensibles al precio)
   - Clientes nuevos: precio de penetracion (descuento 10%)
   - Ventas por volumen: descuento por cantidad

4. MONITOREO CONTINUO
   - Recalcular elasticidad cada mes
   - Actualizar curva de demanda con nuevos datos
   - Ajustar precio si cambian costos o competencia

5. RIESGOS Y MITIGACIONES
   - Riesgo: Reaccion negativa de clientes -> Comunicar valor agregado
   - Riesgo: Competencia responde -> Monitorear precios semanalmente
   - Riesgo: Elasticidad cambia -> Tener precio piso y techo definidos

--- PROXIMOS PASOS ---
1. Validar el modelo con datos de una prueba A/B
2. Implementar el cambio en un grupo piloto (10% de clientes)
3. Medir impacto durante 2 semanas
4. Escalar si los resultados son positivos
5. Documentar y repetir para otros productos
""")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 12. RECOMENDACIONES DE ESTRATEGIA DE PRECIOS
3. ============================================================

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## Resumen Ejecutivo

Se determinó el **precio óptimo** que maximiza el beneficio utilizando optimización numérica con SciPy:

| Variable            | Valor Actual     | Valor Optimo    | Cambio    |
|--------------------|------------------|-----------------|-----------|
| Precio Promedio    | $XX.XX           | $XX.XX          | +/-X.X%   |
| Cantidad           | X,XXX            | X,XXX           | +/-X.X%   |
| Ingreso Total      | $XX,XXX          | $XX,XXX         | +/-X.X%   |
| Costo Total        | $XX,XXX          | $XX,XXX         | +/-X.X%   |
| Beneficio Total    | $XX,XXX          | $XX,XXX         | +/-X.X%   |
| Margen             | X.X%             | X.X%            | +/-X.Xpp  |

**Conclusión:** El precio óptimo de **$XX.XX** maximiza el beneficio en **$XX,XXX**, representando una mejora del **X.X%** vs el precio actual. Se recomienda implementar el cambio de forma gradual con monitoreo continuo.

**Impacto esperado:** Incremento del XX% en beneficio neto mediante optimización de precios basada en elasticidad de demanda.

---

## Ejercicios Adicionales

1. **Modelo de demanda alternativo:** Probar una función de demanda exponencial Q(P) = a * exp(-b*P) y comparar con el modelo log-lineal. ¿Cuál se ajusta mejor a los datos?

2. **Costo variable:** En lugar de usar costo unitario constante, modelar costo como función de la cantidad (economías de escala): Costo(Q) = CF + CV * Q^0.9. ¿Cambia el precio óptimo?

3. **Optimización con múltiples productos:** Si hay dos productos sustitutos, la demanda de uno depende del precio del otro. Modelar demanda cruzada y optimizar ambos precios simultáneamente.

4. **Simulación de Monte Carlo:** Incorporar incertidumbre en la elasticidad (distribución normal con media y std_error) y simular 1000 escenarios. ¿Cuál es la distribución del precio óptimo?

5. **Prueba A/B:** Diseñar un experimento con 3 precios (actual, óptimo, intermedio) asignados aleatoriamente a clientes durante 2 semanas. ¿Cómo validarías estadísticamente que el precio óptimo es superior?
