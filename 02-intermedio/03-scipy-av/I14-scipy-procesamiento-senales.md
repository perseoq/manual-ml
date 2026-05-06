# I14 — Procesamiento de Señales con SciPy aplicado a Ventas/Compras/Inventarios

## 1. Introducción Teórica

El procesamiento de señales analiza series temporales como si fueran señales continuas. Aplicado a ventas, compras e inventarios:

- **Suavizado**: eliminar ruido diario para ver tendencia subyacente.
- **Filtrado**: separar componentes de alta frecuencia (ruido) de baja frecuencia (tendencia).
- **Detección de picos**: identificar promociones exitosas, días pico.
- **Correlación**: medir similitud entre demanda de productos relacionados.
- **Remuestreo**: cambiar frecuencia (diaria → semanal) para análisis macro.

### Funciones principales de `scipy.signal`

| Función | Descripción | Aplicación |
|---|---|---|
| `convolve` | Convolución con kernel | Suavizado con media móvil ponderada |
| `correlate` | Correlación cruzada / autocorrelación | Periodicidad en demanda |
| `medfilt` | Filtro de mediana | Eliminar spikes (outliers) |
| `savgol_filter` | Savitzky-Golay (suavizado + derivada) | Tendencia preservando picos |
| `detrend` | Eliminar tendencia lineal | Estacionalidad pura |
| `resample` | Remuestreo con FFT | Cambiar frecuencia |
| `decimate` | Diezmar (reducir frecuencia) | Señales de alta frecuencia |
| `spectrogram` | Espectrograma tiempo-frecuencia | Patrones estacionales múltiples |
| `find_peaks` | Detectar picos en señal | Picos de demanda |
| `argrelextrema` | Extremos locales | Máximos/mínimos en tendencia |
| `argrelmax` / `argrelmin` | Solo máximos / solo mínimos | Identificar ciclos |
| `hilbert` | Transformada de Hilbert | Fase instantánea de demanda |
| `butter` + `filtfilt` | Filtro digital Butterworth | Filtro pasa-bajos sin desfase |

---

## 2. Ejemplos Prácticos

### Ejemplo 1: convolve — Suavizar serie de ventas con kernel [1,2,1]

```python
import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

ventas = pd.read_csv("../../datos/ventas.csv")
inventario = pd.read_csv("../../datos/inventario.csv")

# Crear serie diaria de ventas totales
ventas["fecha"] = pd.to_datetime(ventas["fecha"])
serie_diaria = ventas.groupby("fecha")["cantidad"].sum().reset_index()
serie_diaria = serie_diaria.sort_values("fecha")

dias = np.arange(len(serie_diaria))
cantidades = serie_diaria["cantidad"].values

kernel = np.array([1, 2, 1]) / 4
suavizado = signal.convolve(cantidades, kernel, mode="same")

print("Primeros 10 días - Original vs Suavizado:")
for i in range(10):
    print(f"  Día {i+1}: {cantidades[i]:5.0f} → {suavizado[i]:5.1f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: convolve — Suavizar serie de ventas con kernel [1,2,1].*

1. Crear serie diaria de ventas totales

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: correlate — Autocorrelación de demanda diaria (buscar periodicidad)

```python
# Normalizar la señal
cant_norm = (cantidades - cantidades.mean()) / cantidades.std()
autocorr = signal.correlate(cant_norm, cant_norm, mode="full")
autocorr = autocorr[len(autocorr) // 2:]  # Solo mitad positiva
autocorr /= autocorr[0]  # Normalizar

# Encontrar picos de autocorrelación (periodicidad)
picos_autocorr, _ = signal.find_peaks(autocorr, height=0.3, distance=3)
print(f"Picos de autocorrelación en lags: {picos_autocorr[:5]}")
print(f"Periodicidades sugeridas (días): {picos_autocorr[:5]}")
print(f"Autocorrelación en lag 1: {autocorr[1]:.3f}")
print(f"Autocorrelación en lag 7: {autocorr[7]:.3f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: correlate — Autocorrelación de demanda diaria (buscar periodicidad).*

1. Normalizar la señal
2. Encontrar picos de autocorrelación (periodicidad)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: medfilt — Filtro mediana para eliminar spikes de ventas

```python
# Introducir spikes artificiales
np.random.seed(42)
spikes = cantidades.copy()
dias_spike = np.random.choice(len(spikes), 5, replace=False)
spikes[dias_spike] *= 3  # Triplicar ventas en días aleatorios

mediana_3 = signal.medfilt(spikes, kernel_size=3)
mediana_7 = signal.medfilt(spikes, kernel_size=7)

print(f"Spike original día {dias_spike[0]}: {spikes[dias_spike[0]]:.0f}")
print(f"  medfilt k=3: {mediana_3[dias_spike[0]]:.0f}")
print(f"  medfilt k=7: {mediana_7[dias_spike[0]]:.0f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: medfilt — Filtro mediana para eliminar spikes de ventas.*

1. Introducir spikes artificiales

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: savgol_filter — Suavizado que preserva tendencia

```python
from scipy.signal import savgol_filter

# Savitzky-Golay con ventana 7 días, polinomio orden 2
savgol_7_2 = savgol_filter(cantidades, window_length=7, polyorder=2)
savgol_14_3 = savgol_filter(cantidades, window_length=14, polyorder=3)

print("Comparación Savgol en punto medio:")
idx = len(cantidades) // 2
print(f"  Original: {cantidades[idx]:.0f}")
print(f"  Savgol(7,2): {savgol_7_2[idx]:.1f}")
print(f"  Savgol(14,3): {savgol_14_3[idx]:.1f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 4: savgol_filter — Suavizado que preserva tendencia.*

1. Savitzky-Golay con ventana 7 días, polinomio orden 2

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: savgol_filter — Derivada del suavizado (tasa de cambio)

```python
from scipy.signal import savgol_filter

# Derivada con Savgol (polyorder=3, deriv=1)
derivada = savgol_filter(cantidades, window_length=7, polyorder=3, deriv=1)
derivada2 = savgol_filter(cantidades, window_length=7, polyorder=3, deriv=2)

print("Tasa de cambio (derivada) días seleccionados:")
dias_sel = [5, 10, 15, 20]
for d in dias_sel:
    if d < len(derivada):
        print(f"  Día {d}: tasa={derivada[d]:.2f}, aceleración={derivada2[d]:.2f}")

# Máxima tasa de crecimiento
idx_max = np.argmax(derivada)
print(f"Máxima tasa de crecimiento: día {idx_max}, {derivada[idx_max]:.2f} unds/día")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: savgol_filter — Derivada del suavizado (tasa de cambio).*

1. Derivada con Savgol (polyorder=3, deriv=1)
2. Máxima tasa de crecimiento

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: detrend — Eliminar tendencia lineal de ventas para analizar estacionalidad

```python
from scipy.signal import detrend

# Eliminar tendencia lineal
sin_tendencia = detrend(cantidades)
media_original = cantidades.mean()
residuo = sin_tendencia + media_original  # Re-centrar

print(f"Media original: {media_original:.1f}")
print(f"Media sin tendencia: {sin_tendencia.mean():.1f}")
print(f"Desvío estándar del residuo: {sin_tendencia.std():.1f}")
print(f"¿Tendencia eliminada? Pendiente antes/después:")

# Verificar pendiente antes y después
t = np.arange(len(cantidades))
pend_original = np.polyfit(t, cantidades, 1)[0]
pend_detrend = np.polyfit(t, sin_tendencia, 1)[0]
print(f"  Pendiente original: {pend_original:.4f}")
print(f"  Pendiente detrend: {pend_detrend:.6f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: detrend — Eliminar tendencia lineal de ventas para analizar estacionalidad.*

1. Eliminar tendencia lineal
2. Verificar pendiente antes y después

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: resample — Cambiar frecuencia de ventas (diaria a semanal)

```python
from scipy.signal import resample

# Ventas diarias a semanales (7 días)
ventas_semanales = resample(cantidades, len(cantidades) // 7)
semanas = np.arange(len(ventas_semanales))

print("Ventas semanales (remuestreadas):")
for s, v in zip(semanas[:8], ventas_semanales[:8]):
    print(f"  Semana {s+1}: {v:.0f} unidades")

print(f"Total original (diario): {cantidades.sum():.0f}")
print(f"Total remuestreado (semanal): {ventas_semanales.sum():.0f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: resample — Cambiar frecuencia de ventas (diaria a semanal).*

1. Ventas diarias a semanales (7 días)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: decimate — Diezmar señal de alta frecuencia

```python
from scipy.signal import decimate

# Simular señal de alta frecuencia (cada hora, 24h * 30 días)
np.random.seed(42)
horas = np.arange(720)
ventas_hora = 10 + 5 * np.sin(2 * np.pi * horas / 24) + np.random.normal(0, 3, 720)

# Diezmar: reducir de horaria a diaria (factor 24)
ventas_diarias = decimate(ventas_hora, q=24, ftype="iir")

print(f"Original: {len(ventas_hora)} puntos (horarios)")
print(f"Diezmado: {len(ventas_diarias)} puntos (diarios)")
print(f"Primeros 5 días diezmados: {ventas_diarias[:5].round(1)}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: decimate — Diezmar señal de alta frecuencia.*

1. Simular señal de alta frecuencia (cada hora, 24h * 30 días)
2. Diezmar: reducir de horaria a diaria (factor 24)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: find_peaks — Detectar picos de demanda (promociones, eventos)

```python
from scipy.signal import find_peaks

# Encontrar picos en la serie diaria
picos, propiedades = signal.find_peaks(cantidades, height=cantidades.mean() + cantidades.std())
print(f"Número de picos detectados: {len(picos)}")
print(f"Alturas de los primeros 5 picos:")
for i, p in enumerate(picos[:5]):
    print(f"  Pico {i+1}: día {p}, cantidad={cantidades[p]:.0f}")
print(f"Altura promedio de picos: {propiedades['peak_heights'].mean():.1f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 9: find_peaks — Detectar picos de demanda (promociones, eventos).*

1. Encontrar picos en la serie diaria

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: find_peaks con distance y prominence — Filtrar picos pequeños

```python
# Picos con distancia mínima 7 días y prominencia > 20
picos_filt, props = signal.find_peaks(cantidades,
                                       distance=7,
                                       prominence=20,
                                       height=cantidades.mean())

print(f"Picos relevantes detectados: {len(picos_filt)}")
print(f"{'Día':>4} {'Cantidad':>10} {'Prominencia':>12} {'Ancho':>6}")
for p, prom, w in zip(picos_filt, props["prominences"], props["widths"]):
    print(f"{p:4d} {cantidades[p]:10.0f} {prom:12.1f} {w:6.1f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: find_peaks con distance y prominence — Filtrar picos pequeños.*

1. Picos con distancia mínima 7 días y prominencia > 20

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: argrelextrema — Máximos y mínimos locales en tendencia

```python
from scipy.signal import argrelextrema

# Suavizar primero para evitar falsos extremos
savgol = savgol_filter(cantidades, window_length=7, polyorder=2)

maximos = argrelextrema(savgol, np.greater)[0]
minimos = argrelextrema(savgol, np.less)[0]

print(f"Máximos locales: {len(maximos)} encontrados")
print(f"Mínimos locales: {len(minimos)} encontrados")
print(f"Primeros máximos en días: {maximos[:5]}")
print(f"Primeros mínimos en días: {minimos[:5]}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: argrelextrema — Máximos y mínimos locales en tendencia.*

1. Suavizar primero para evitar falsos extremos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: argrelmax — Solo máximos de ventas

```python
from scipy.signal import argrelmax

# Máximos directos sobre datos suavizados
max_ventas = argrelmax(savgol, order=3)[0]

print(f"Máximos de ventas (order=3): {len(max_ventas)}")
for m in max_ventas[:8]:
    print(f"  Día {m}: {cantidades[m]:.0f} (suavizado: {savgol[m]:.1f})")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: argrelmax — Solo máximos de ventas.*

1. Máximos directos sobre datos suavizados

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: argrelmin — Solo mínimos de ventas

```python
from scipy.signal import argrelmin

# Mínimos directos sobre datos suavizados
min_ventas = argrelmin(savgol, order=3)[0]

print(f"Mínimos de ventas (order=3): {len(min_ventas)}")
for m in min_ventas[:8]:
    print(f"  Día {m}: {cantidades[m]:.0f} (suavizado: {savgol[m]:.1f})")

# Ciclo completo: días entre mínimo consecutivos
if len(min_ventas) >= 2:
    ciclos = np.diff(min_ventas)
    print(f"Duración promedio de ciclos: {ciclos.mean():.1f} días")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: argrelmin — Solo mínimos de ventas.*

1. Mínimos directos sobre datos suavizados
2. Ciclo completo: días entre mínimo consecutivos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: butter + filtfilt — Filtro pasabajos para eliminar ruido diario

```python
from scipy.signal import butter, filtfilt

# Diseñar filtro Butterworth pasabajos
# Frecuencia de corte: 0.1 (periodo 10 días, elimina oscilaciones < 10 días)
b, a = signal.butter(4, 0.1, btype="low")
senal_filtrada = signal.filtfilt(b, a, cantidades)

# Comparar con original
print("Comparación filtro pasabajos (Butterworth 4to orden):")
idx_test = [0, 5, 10, 20, 50, 100]
for i in idx_test:
    if i < len(cantidades):
        print(f"  Día {i}: original={cantidades[i]:.0f}, filtrada={senal_filtrada[i]:.1f}")

# Ruido eliminado (residuo)
ruido = cantidades - senal_filtrada
print(f"Ruido RMS: {np.sqrt(np.mean(ruido**2)):.2f}")
print(f"Relación señal/ruido: {20 * np.log10(cantidades.std() / ruido.std()):.1f} dB")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: butter + filtfilt — Filtro pasabajos para eliminar ruido diario.*

1. Diseñar filtro Butterworth pasabajos
2. Frecuencia de corte: 0.1 (periodo 10 días, elimina oscilaciones < 10 días)
3. Comparar con original
4. Ruido eliminado (residuo)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: Integrador — Pipeline de procesamiento

```python
from scipy.signal import butter, filtfilt, savgol_filter, find_peaks

# Pipeline: detrend → suavizado → find_peaks → análisis
# 1. Eliminar tendencia
sin_tend = detrend(cantidades) + cantidades.mean()

# 2. Filtro pasabajos
b, a = butter(4, 0.15, btype="low")
filtrada = filtfilt(b, a, sin_tend)

# 3. Suavizado Savgol
suave = savgol_filter(filtrada, window_length=7, polyorder=2)

# 4. Detectar picos
picos_final, props_final = find_peaks(suave, distance=10, prominence=15)

print("=== PIPELINE COMPLETO ===")
print(f"Señal original: {len(cantidades)} puntos")
print(f"Tendencia eliminada: Sí")
print(f"Filtro pasabajos: Butterworth orden 4, fc=0.15")
print(f"Suavizado: Savgol(7,2)")
print(f"")

# Análisis de picos
print("=== ANÁLISIS DE PICOS ===")
print(f"Picos detectados: {len(picos_final)}")
print(f"{'Pico #':>6} {'Día':>5} {'Ventas':>8} {'Prom.':>8} {'Ancho':>7}")
for i, (p, prom, w) in enumerate(zip(picos_final[:8],
                                       props_final["prominences"][:8],
                                       props_final["widths"][:8])):
    print(f"{i+1:6d} {p:5d} {cantidades[p]:8.0f} {prom:8.1f} {w:7.1f}")

# Distribución de picos por día de semana
dias_semana = pd.to_datetime(serie_diaria["fecha"].iloc[picos_final]).dayofweek
print(f"\nPicos por día de semana:")
for ds in range(7):
    print(f"  Día {ds}: {(dias_semana == ds).sum()} picos")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: Integrador — Pipeline de procesamiento.*

1. Pipeline: detrend → suavizado → find_peaks → análisis
2. 1. Eliminar tendencia
3. 2. Filtro pasabajos
4. 3. Suavizado Savgol
5. 4. Detectar picos
6. Análisis de picos
7. Distribución de picos por día de semana

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 3. Resumen

| Operación | Función | Aplicación en Ventas |
|---|---|---|
| Suavizado | `convolve`, `medfilt`, `savgol_filter` | Eliminar ruido diario |
| Derivada | `savgol_filter(deriv=1)`, `savgol_filter(deriv=2)` | Tasa de cambio, aceleración |
| Tendencia | `detrend` | Separar tendencia de estacionalidad |
| Frecuencia | `resample`, `decimate` | Cambiar granularidad temporal |
| Picos | `find_peaks`, `argrelextrema`, `argrelmax`, `argrelmin` | Detectar eventos, ciclos |
| Filtrado | `butter` + `filtfilt` | Eliminar rangos de frecuencia |
| Correlación | `correlate` | Periodicidad, similitud entre productos |
| Fase | `hilbert` | Fase instantánea de ciclo de demanda |

**Pipeline recomendado**:
```
Datos brutos → detrend → filtro pasabajos → savgol_filter → find_peaks → análisis
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*3. Resumen.*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 4. Ejercicios Propuestos

1. Usa `convolve` con kernels de diferentes tamaños ([1,1,1]/3, [1,2,3,2,1]/9, [1,1,1,1,1]/5) para suavizar las ventas diarias de la sucursal "Querétaro". ¿Cuál kernel produce el suavizado más agresivo?

2. Aplica `correlate` para medir la correlación cruzada entre las ventas de "Laptop Pro 15" y "Laptop Air 13". ¿Hay un desfase temporal entre ambas? ¿Una lidera a la otra?

3. Usa `find_peaks` para detectar los días de mayor venta en cada mes. Calcula qué porcentaje de las ventas mensuales ocurren en días pico. Usa `prominence` para filtrar solo los picos relevantes (>1.5 desviaciones estándar sobre la media).

4. Implementa un filtro `butter` pasabajos de orden 6 con frecuencia de corte 0.08 para filtrar las ventas diarias de la categoría "Periféricos". Compara visualmente el resultado con `savgol_filter(window_length=5, polyorder=2)`.

5. Calcula la autocorrelación de la serie de demanda diaria de "Electrónica" y encuentra los 3 lags más significativos. ¿Hay evidencia de estacionalidad semanal (lag 7)?

6. Aplica `detrend` a la serie de ventas acumuladas por semana y luego busca periodicidad con `correlate`. ¿Qué patrón estacional emerge al eliminar la tendencia de crecimiento?

7. Usa `argrelmax` y `argrelmin` (con order=5) sobre la serie suavizada con `savgol_filter` para identificar los ciclos de inventario: momentos de máximo y mínimo stock para un SKU específico en `inventario.csv`.

8. **Integrador**: Construye un pipeline completo para una sucursal específica: (a) detrend, (b) filtro Butterworth pasabajos, (c) suavizado Savgol, (d) detección de picos con `find_peaks`, (e) agrupación de picos por día de semana. Reporta: número de picos, día de semana con más picos, altura promedio de picos, y distancia media entre picos.
