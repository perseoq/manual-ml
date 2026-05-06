# I07 — Fechas y Tiempos Avanzados en Pandas

## 1. Introducción Teórica

El manejo correcto de fechas es crítico en análisis de ventas: tendencias semanales, comparaciones año contra año, estacionalidad, y ventanas temporales.

### Componentes clave

**pd.to_datetime():**
- Parsea strings a datetime con formatos personalizados
- Maneja errores con `errors='coerce'`

**dt accesor:**
- Propiedades: year, month, day, dayofweek, day_name(), quarter, weekofyear
- Flags: is_month_start, is_month_end, is_quarter_end, is_year_start
- Información: days_in_month

**date_range():**
- Frecuencias: D (diario), B (business days), W (semanal), M (fin de mes), MS (inicio de mes), Q (trimestre), Y (año)
- Modificadores: W-MON (semanal lunes), BMS (business month start)
- Personalizadas: SM (semi-mes), BH (business hour), CBH (custom business hour)

**bdate_range() y CustomBusinessDay:**
- Días hábiles con holidays personalizados
- weekmask para semanas no estándar (ej. domingo-jueves)

**Resample:**
- label y closed controlan la etiqueta y los límites del intervalo
- Combinable con agg para múltiples métricas

**Time zones:**
- tz_localize: asigna zona horaria a datos naive
- tz_convert: convierte entre zonas horarias

**Timedelta / Timestamp:**
- Timedelta: diferencias de tiempo (días, horas)
- Timestamp: puntos en el tiempo con operaciones aritméticas

---

## 2. Ejemplos Prácticos

### Ejemplo 1: pd.to_datetime con format para CSV

```python
import pandas as pd
import numpy as np

fechas_str = ['2024-15-01', '2024-16-01', '2024-17-01']  # día-mes
try:
    fechas = pd.to_datetime(fechas_str, format='%Y-%d-%m')
    print("Parseado correctamente:", fechas)
except Exception as e:
    print(f"Error con formato estándar, usando format personalizado: {e}")
    fechas = pd.to_datetime(fechas_str, format='%Y-%d-%m')
    print("Fechas:", fechas)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 1: pd.to_datetime con format para CSV.*

1. `import pandas as pd` — Importa las librerías necesarias para el análisis.
2. `import numpy as np` — Importa las librerías necesarias para el análisis.
3. `fechas = pd.to_datetime(fechas_str, format='%Y-%d-%m')` — Convierte la columna a formato datetime.
4. `print("Parseado correctamente:", fechas)` — Muestra el resultado por pantalla.
5. `print(f"Error con formato estándar, usando format personalizado: {e}")` — Muestra el resultado por pantalla.
6. `fechas = pd.to_datetime(fechas_str, format='%Y-%d-%m')` — Convierte la columna a formato datetime.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 2: dt.year, month, day, dayofweek, day_name()

```python
fechas = pd.date_range('2024-01-01', periods=10, freq='D')
df = pd.DataFrame({'fecha': fechas})

df['year'] = df['fecha'].dt.year
df['month'] = df['fecha'].dt.month
df['day'] = df['fecha'].dt.day
df['dayofweek'] = df['fecha'].dt.dayofweek  # 0=lunes
df['day_name'] = df['fecha'].dt.day_name()
df['is_weekend'] = df['fecha'].dt.dayofweek >= 5

print(df)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 2: dt.year, month, day, dayofweek, day_name().*


*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 3: dt.is_month_start, is_month_end, is_quarter_end

```python
fechas = pd.date_range('2024-12-28', periods=10, freq='D')
df = pd.DataFrame({'fecha': fechas})

df['month_start'] = df['fecha'].dt.is_month_start
df['month_end'] = df['fecha'].dt.is_month_end
df['quarter_end'] = df['fecha'].dt.is_quarter_end
df['year_end'] = df['fecha'].dt.is_year_end

print(df)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 3: dt.is_month_start, is_month_end, is_quarter_end.*

1. `print(df)` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 4: dt.days_in_month, dt.weekofyear, dt.quarter

```python
fechas = pd.date_range('2024-01-01', periods=365, freq='D')
df = pd.DataFrame({'fecha': fechas})
df_mes = df[df['fecha'].dt.day == 15]  # día 15 de cada mes

df_mes['days_in_month'] = df_mes['fecha'].dt.days_in_month
df_mes['weekofyear'] = df_mes['fecha'].dt.isocalendar().week.astype(int)
df_mes['quarter'] = df_mes['fecha'].dt.quarter

print(df_mes[['fecha', 'days_in_month', 'weekofyear', 'quarter']].head(12))
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

*Ejemplo 4: dt.days_in_month, dt.weekofyear, dt.quarter.*

1. `print(df_mes[['fecha', 'days_in_month', 'weekofyear', 'quarter']].head(12))` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 5: date_range con freq="B" (business days)

```python
biz_days = pd.date_range('2024-01-01', periods=10, freq='B')
print("Días hábiles:", biz_days.tolist())
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 5: date_range con freq="B" (business days).*

1. `print("Días hábiles:", biz_days.tolist())` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 6: date_range con freq="W-MON" (semanales lunes)

```python
lunes = pd.date_range('2024-01-01', periods=5, freq='W-MON')
print("Lunes:", lunes.tolist())

findes = pd.date_range('2024-01-01', periods=5, freq='W-SAT')
print("Sábados:", findes.tolist())
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 6: date_range con freq="W-MON" (semanales lunes).*

1. `print("Lunes:", lunes.tolist())` — Muestra el resultado por pantalla.
2. `print("Sábados:", findes.tolist())` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 7: bdate_range con holidays personalizados

```python
from pandas.tseries.holiday import USFederalHolidayCalendar

cal = USFederalHolidayCalendar()
holidays = cal.holidays('2024-01-01', '2024-12-31')

biz_ene = pd.bdate_range('2024-01-01', '2024-01-15', holidays=holidays)
print("Días hábiles enero (sin feriados):", biz_ene.tolist())
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 7: bdate_range con holidays personalizados.*

1. `from pandas.tseries.holiday import USFederalHolidayCalendar` — Importa las librerías necesarias para el análisis.
2. `print("Días hábiles enero (sin feriados):", biz_ene.tolist())` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 8: CustomBusinessDay con weekmask

```python
from pandas.tseries.offsets import CustomBusinessDay

# Semana domingo-jueves (típico en Medio Oriente)
bday_custom = CustomBusinessDay(weekmask='Sun Mon Tue Wed Thu')

fechas_custom = pd.date_range('2024-01-01', periods=10, freq=bday_custom)
print("Días hábiles (Dom-Jue):", fechas_custom.tolist())
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 8: CustomBusinessDay con weekmask.*

1. Semana domingo-jueves (típico en Medio Oriente)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 9: resample("W") con label y closed

```python
np.random.seed(42)
idx = pd.date_range('2024-01-01', '2024-03-31', freq='D')
ventas = pd.Series(np.random.randint(100, 500, len(idx)), index=idx, name='ventas')

# Resample semanal: lunes como etiqueta, domingo como cierre
semanal = ventas.resample('W', label='left', closed='left').sum()
print("Semanal (label=left, closed=left):\n", semanal.head())
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

*Ejemplo 9: resample("W") con label y closed.*

1. Resample semanal: lunes como etiqueta, domingo como cierre

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 10: resample("M") con agg múltiple

```python
mensual = ventas.resample('M').agg(['sum', 'mean', 'std', 'min', 'max'])
print("Resumen mensual:\n", mensual.round(2))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 10: resample("M") con agg múltiple.*

1. `mensual = ventas.resample('M').agg(['sum', 'mean', 'std', 'min', 'max'])` — Aplica funciones de agregación a los grupos.
2. `print("Resumen mensual:\n", mensual.round(2))` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 11: resample("Q") con agg=[sum, mean]

```python
trimestral = ventas.resample('Q').agg([sum, 'mean'])
print("Trimestral:\n", trimestral.round(2))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 11: resample("Q") con agg=[sum, mean].*

1. `trimestral = ventas.resample('Q').agg([sum, 'mean'])` — Aplica funciones de agregación a los grupos.
2. `print("Trimestral:\n", trimestral.round(2))` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 12: resample("Y") para anual

```python
anual = ventas.resample('Y').agg(['sum', 'mean', 'count'])
print("Anual:\n", anual.round(2))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 12: resample("Y") para anual.*

1. `anual = ventas.resample('Y').agg(['sum', 'mean', 'count'])` — Aplica funciones de agregación a los grupos.
2. `print("Anual:\n", anual.round(2))` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 13: period_range para rangos de período

```python
periodos = pd.period_range('2024-01', periods=12, freq='M')
print("Períodos mensuales:", periodos.tolist())

# Trimestres
trimestres = pd.period_range('2024-Q1', periods=4, freq='Q')
print("Trimestres:", trimestres.tolist())
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 13: period_range para rangos de período.*

1. Trimestres

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 14: to_period("M") — convertir fechas a períodos

```python
fechas = pd.date_range('2024-01-15', periods=6, freq='D')
df = pd.DataFrame({'fecha': fechas, 'ventas': [100, 200, 150, 300, 250, 180]})

df['periodo'] = df['fecha'].dt.to_period('M')
print(df)

# Agrupar por período
print("\nVentas por mes:")
print(df.groupby('periodo')['ventas'].sum())
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 14: to_period("M") — convertir fechas a períodos.*

1. Agrupar por período

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 15: tz_localize — localizar a zona horaria

```python
# Fechas naive (sin zona horaria)
fechas_naive = pd.date_range('2024-01-01', periods=3, freq='h')
print("Naive:", fechas_naive)

# Localizar a UTC
fechas_utc = fechas_naive.tz_localize('UTC')
print("UTC:", fechas_utc)

# Localizar a México
fechas_mx = fechas_naive.tz_localize('America/Mexico_City')
print("México:", fechas_mx)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 15: tz_localize — localizar a zona horaria.*

1. Fechas naive (sin zona horaria)
2. Localizar a UTC
3. Localizar a México

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 16: tz_convert — convertir a otra zona

```python
# Convertir UTC a otras zonas
fechas_utc = pd.date_range('2024-01-01 08:00:00', periods=3, freq='h',
                            tz='UTC')

ny = fechas_utc.tz_convert('America/New_York')
tokyo = fechas_utc.tz_convert('Asia/Tokyo')
madrid = fechas_utc.tz_convert('Europe/Madrid')

print("UTC:", fechas_utc)
print("NY:", ny)
print("Tokyo:", tokyo)
print("Madrid:", madrid)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 16: tz_convert — convertir a otra zona.*

1. Convertir UTC a otras zonas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 17: Timedelta — sumar/restar días

```python
fecha_base = pd.Timestamp('2024-01-15')

# Sumar días
fechas = {
    'entrega_3d': fecha_base + pd.Timedelta(days=3),
    'entrega_1sem': fecha_base + pd.Timedelta(weeks=1),
    'entrega_30d': fecha_base + pd.Timedelta(days=30),
    'mes_anterior': fecha_base - pd.Timedelta(days=30),
    'siguiente_hora': fecha_base + pd.Timedelta(hours=48)
}

for nombre, fecha in fechas.items():
    print(f"{nombre}: {fecha}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 17: Timedelta — sumar/restar días.*

1. Sumar días

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 18: Timestamp — propiedades y operaciones

```python
ts = pd.Timestamp('2024-06-15 14:30:00')

print(f"Timestamp: {ts}")
print(f"Año: {ts.year}, Mes: {ts.month}, Día: {ts.day}")
print(f"Hora: {ts.hour}, Minuto: {ts.minute}")
print(f"Día de semana: {ts.day_name()}")
print(f"Semana del año: {ts.isocalendar()[1]}")
print(f"Trimestre: {ts.quarter}")

# Operaciones
ts2 = ts + pd.Timedelta(days=45)
print(f"45 días después: {ts2}")
print(f"Diferencia: {ts2 - ts}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 18: Timestamp — propiedades y operaciones.*

1. Operaciones

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 19: TimedeltaIndex — índice temporal calculado

```python
# Días hasta próximo pedido
pedidos = pd.DataFrame({
    'producto': ['A', 'B', 'C'],
    'fecha_pedido': pd.to_datetime(['2024-01-01', '2024-01-05', '2024-01-10']),
    'fecha_entrega': pd.to_datetime(['2024-01-04', '2024-01-08', '2024-01-12'])
})

pedidos['dias_entrega'] = pedidos['fecha_entrega'] - pedidos['fecha_pedido']
pedidos['dias_int'] = pedidos['dias_entrega'].dt.days

print(pedidos)

# TimedeltaIndex como índice
idx_timedelta = pd.TimedeltaIndex(['1 days', '2 days', '3 days', '4 days', '5 days'])
serie = pd.Series([100, 200, 150, 300, 250], index=idx_timedelta)
print("\nSerie con TimedeltaIndex:\n", serie)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Ejemplo 19: TimedeltaIndex — índice temporal calculado.*

1. Días hasta próximo pedido
2. TimedeltaIndex como índice

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



### Ejemplo 20: Integrador — desde CSV hasta análisis temporal completo

```python
# Simular carga de CSV
np.random.seed(42)
n = 500
data = {
    'fecha': pd.date_range('2024-01-01', periods=n, freq='h'),
    'sucursal': np.random.choice(['Norte', 'Sur', 'Este'], n),
    'producto': np.random.choice(['A', 'B', 'C', 'D'], n),
    'cantidad': np.random.randint(1, 10, n),
    'precio': np.random.uniform(10, 100, n).round(2)
}
df = pd.DataFrame(data)
df['total'] = df['cantidad'] * df['precio']

# 1. Extraer componentes de fecha
df['hora'] = df['fecha'].dt.hour
df['dia_semana'] = df['fecha'].dt.day_name()
df['mes'] = df['fecha'].dt.month
df['semana'] = df['fecha'].dt.isocalendar().week.astype(int)

# 2. Resample diario
df = df.set_index('fecha')
diario = df.resample('D').agg({'total': 'sum', 'cantidad': 'sum'})
diario['transacciones'] = df.resample('D').size()

# 3. Media móvil 7 días
diario['media_movil_7'] = diario['total'].rolling(7).mean()

# 4. Tendencia semanal
semanal = diario.resample('W-MON').agg({'total': 'sum'})
semanal['pct_cambio'] = semanal['total'].pct_change() * 100

# 5. Horas pico
horas_pico = df.groupby('hora').agg({'total': 'sum', 'cantidad': 'sum'})

print("=== Resumen diario ===")
print(diario.head(10))
print("\n=== Tendencia semanal ===")
print(semanal.head())
print("\n=== Horas pico ===")
print(horas_pico.sort_values('total', ascending=False).head())
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

*Ejemplo 20: Integrador — desde CSV hasta análisis temporal completo.*

1. Simular carga de CSV
2. 1. Extraer componentes de fecha
3. 2. Resample diario
4. 3. Media móvil 7 días
5. 4. Tendencia semanal
6. 5. Horas pico

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



---

## 3. Resumen Teórico

| Operación | Aplicación en Ventas/Compras/Inventarios |
|-----------|----------------------------------------|
| to_datetime(format) | Parsear fechas de sistemas legacy o CSVs |
| dt accesor | Extraer día de semana, mes, trimestre para segmentación |
| is_month_start/end | Cortes contables, cierres mensuales |
| date_range(freq) | Generar calendarios de días hábiles, festivos |
| bdate_range(holidays) | Calendario realista con días feriados |
| CustomBusinessDay | Semanas laborales no estándar (dom-jue) |
| resample(label, closed) | Agregación temporal con control de intervalos |
| to_period | Agrupar por mes/trimestre para reportes |
| tz_localize/convert | Coordinar ventas en múltiples zonas horarias |
| Timedelta | Tiempos de entrega, lead times, vencimientos |
| Timestamp | Operaciones precisas con fechas y horas |

---

## 4. Ejercicios Propuestos

**Ejercicio 1:** Parsea un CSV de ventas con fechas en formato `%d/%m/%Y %H:%M` y extrae día de semana y hora.

**Ejercicio 2:** Genera un calendario de días hábiles para 2024 en México (incluye festivos: 1 ene, 5 feb, 21 mar, 1 may, 16 sep, 20 nov, 25 dic).

**Ejercicio 3:** Usa resample("W") con label y closed para calcular ventas semanales de lunes a domingo, etiquetando con el lunes.

**Ejercicio 4:** Convierte una serie de ventas diarias a períodos mensuales con to_period("M") y agrupa.

**Ejercicio 5:** Simula una transacción a las 14:30 UTC y muéstrala en horario de CDMX, NY y Tokyo.

**Ejercicio 6:** Calcula los días de entrega de 10 pedidos usando Timedelta entre fecha_pedido y fecha_entrega.

**Ejercicio 7:** Encuentra el día con más ventas de cada mes usando resample("M") y idxmax.

**Ejercicio 8:** Crea un análisis completo: carga datos horarios → resample diario → extrae componentes → media móvil → reporte semanal con cambio porcentual.

---

*Fin del documento I07 — Fechas y Tiempos Avanzados en Pandas*
