# Módulo B23 — Pandas: Fechas y Series Temporales

## Teoría

Pandas tiene soporte nativo para trabajar con datos temporales:

- `pd.to_datetime`: convertir strings a datetime
- `dt` accessor: extraer año, mes, día, día de semana, etc.
- `date_range` / `bdate_range`: generar rangos de fechas
- `DateOffset`: desplazamientos temporales
- `resample`: cambiar frecuencia temporal con agregación
- `asfreq`: cambiar frecuencia sin agregación
- `shift`, `diff`, `pct_change`: operaciones con rezago
- `rolling`, `expanding`, `ewm`: ventanas móviles
- `to_period`: convertir a período
- `tz_localize` / `tz_convert`: manejo de zonas horarias
- `Timedelta` / `Timestamp`: tipos temporales

Aplicado a: tendencia de ventas, media móvil, comparación mensual.

## Setup

import pandas as pd
import numpy as np

ventas = pd.read_csv("../datos/ventas.csv")
ventas["fecha"] = pd.to_datetime(ventas["fecha"])
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La operación transforma o filtra los datos según la lógica implementada.


---

## Ejemplos

### 1. pd.to_datetime — convertir a datetime

ventas["fecha"] = pd.to_datetime(ventas["fecha"])
print(ventas["fecha"].dtype)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 1. pd.to_datetime — convertir a datetime*


### 2. dt.year — extraer año

print(ventas["fecha"].dt.year.value_counts())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 2. dt.year — extraer año*


### 3. dt.month — extraer mes

print(ventas["fecha"].dt.month.value_counts().sort_index())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 3. dt.month — extraer mes*


### 4. dt.day — extraer día

print(ventas["fecha"].dt.day.head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 4. dt.day — extraer día*


### 5. dt.dayofweek — día de la semana (0=lunes)

ventas["dia_semana_calc"] = ventas["fecha"].dt.dayofweek
print(ventas[["fecha", "dia_semana_calc"]].head(10))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 5. dt.dayofweek — día de la semana (0=lunes)*


### 6. dt.quarter — trimestre

print(ventas["fecha"].dt.quarter.value_counts().sort_index())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 6. dt.quarter — trimestre*


### 7. DateOffset — desplazamiento

from pandas.tseries.offsets import DateOffset
dia_actual = pd.Timestamp("2024-01-15")
print("Día:", dia_actual)
print("+1 semana:", dia_actual + DateOffset(weeks=1))
print("+1 mes:", dia_actual + DateOffset(months=1))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 7. DateOffset — desplazamiento*


### 8. date_range — generar rango de fechas

rango = pd.date_range(start="2024-01-01", end="2024-01-10", freq="D")
print(rango)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 8. date_range — generar rango de fechas*


### 9. bdate_range — días hábiles

rango = pd.bdate_range(start="2024-01-01", end="2024-01-15")
print(rango)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 9. bdate_range — días hábiles*


### 10. resample D — agregación diaria

ventas_diarias = ventas.set_index("fecha").resample("D")["ingreso"].sum()
print(ventas_diarias.head(10))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 10. resample D — agregación diaria*


### 11. resample W — semanal

ventas_semanales = ventas.set_index("fecha").resample("W")["ingreso"].sum()
print(ventas_semanales.head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 11. resample W — semanal*


### 12. resample ME — mensual

ventas_mensuales = ventas.set_index("fecha").resample("ME")["ingreso"].sum()
print(ventas_mensuales)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 12. resample ME — mensual*


### 13. resample Q — trimestral

ventas_trimestrales = ventas.set_index("fecha").resample("QE")["ingreso"].sum()
print(ventas_trimestrales)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 13. resample Q — trimestral*


### 14. resample YE — anual

ventas_anuales = ventas.set_index("fecha").resample("YE")["ingreso"].sum()
print(ventas_anuales)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 14. resample YE — anual*


### 15. asfreq — cambiar frecuencia sin agregación

diario = ventas.set_index("fecha")["ingreso"].head(10)
print(diario.asfreq("h", method="ffill").head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 15. asfreq — cambiar frecuencia sin agregación*


### 16. shift — desplazar en el tiempo

ingresos_diarios = ventas.set_index("fecha").resample("D")["ingreso"].sum()
print(ingresos_diarios.shift(1).head(10))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 16. shift — desplazar en el tiempo*


### 17. diff — diferencias temporales

ingresos_diarios = ventas.set_index("fecha").resample("D")["ingreso"].sum()
print(ingresos_diarios.diff().head(10))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 17. diff — diferencias temporales*


### 18. pct_change — cambio porcentual

ingresos_diarios = ventas.set_index("fecha").resample("D")["ingreso"].sum()
print(ingresos_diarios.pct_change().head(10))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 18. pct_change — cambio porcentual*


### 19. rolling — media móvil

ingresos_diarios = ventas.set_index("fecha").resample("D")["ingreso"].sum()
media_7d = ingresos_diarios.rolling(window=7).mean()
print(pd.concat([ingresos_diarios, media_7d], axis=1).head(15))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 19. rolling — media móvil*


### 20. expanding — ventana expansiva

ingresos_diarios = ventas.set_index("fecha").resample("D")["ingreso"].sum()
exp_mean = ingresos_diarios.expanding().mean()
print(pd.concat([ingresos_diarios, exp_mean], axis=1).head(10))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 20. expanding — ventana expansiva*


### 21. ewm — media ponderada exponencial

ingresos_diarios = ventas.set_index("fecha").resample("D")["ingreso"].sum()
ewm_mean = ingresos_diarios.ewm(span=7).mean()
print(pd.concat([ingresos_diarios, ewm_mean], axis=1).head(10))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 21. ewm — media ponderada exponencial*


### 22. to_period — convertir a período

ventas["periodo_mensual"] = ventas["fecha"].dt.to_period("M")
print(ventas[["fecha", "periodo_mensual"]].head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 22. to_period — convertir a período*


### 23. tz_localize y tz_convert

ts = pd.Timestamp("2024-01-15 12:00:00")
ts_local = ts.tz_localize("America/Mexico_City")
print("Localizado:", ts_local)
ts_convert = ts_local.tz_convert("Europe/Madrid")
print("Convertido:", ts_convert)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 23. tz_localize y tz_convert*


### 24. Timedelta — diferencias de tiempo

ventas["fecha"] = pd.to_datetime(ventas["fecha"])
hoy = pd.Timestamp("2024-06-01")
ventas["dias_desde_inicio"] = (hoy - ventas["fecha"]).dt.days
print(ventas[["fecha", "dias_desde_inicio"]].head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 24. Timedelta — diferencias de tiempo*


### 25. Timestamp — crear marcas temporales

ts = pd.Timestamp("2024-01-15 14:30:00")
print("Año:", ts.year)
print("Mes:", ts.month)
print("Hora:", ts.hour)
print("Timestamp:", ts.timestamp())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 25. Timestamp — crear marcas temporales*


### 26. Resample con múltiples agregaciones

resumen_mensual = ventas.set_index("fecha").resample("ME").agg({
    "ingreso": "sum",
    "cantidad": "sum",
    "margen": "mean"
})
print(resumen_mensual)
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 26. Resample con múltiples agregaciones*


### 27. Resample con apply

resumen = ventas.set_index("fecha").resample("W").apply(
    lambda x: x["ingreso"].sum() / x["cantidad"].sum()
)
print(resumen.head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 27. Resample con apply*


### 28. dt.strftime — formatear fechas

ventas["fecha_str"] = ventas["fecha"].dt.strftime("%Y-%m-%d")
print(ventas[["fecha", "fecha_str"]].head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 28. dt.strftime — formatear fechas*


### 29. dt.is_month_end — fin de mes

print(ventas["fecha"].dt.is_month_end.head())
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 29. dt.is_month_end — fin de mes*


### 30. rolling con apply personalizado

ingresos_diarios = ventas.set_index("fecha").resample("D")["ingreso"].sum()
def rango_movil(x):
    return x.max() - x.min()
print(ingresos_diarios.rolling(7).apply(rango_movil).head(15))
```

**Salida:**
   (los resultados se muestran al ejecutar el código)

**Explicación:**
   La función print() muestra los resultados de las operaciones realizadas.

*Contexto: 30. rolling con apply personalizado*


---

## Ejercicios

1. Convierte la columna "fecha" a datetime si no lo está.
2. Calcula el ingreso total por mes usando resample "ME".
3. Calcula la media móvil de 7 días del ingreso diario.
4. Usa diff para calcular el cambio de ingreso día a día.
5. Con pct_change obtén el crecimiento porcentual diario.
6. Genera un rango de fechas con date_range del primer al último día del dataset.
7. Extrae el día de la semana de cada fecha con dt.dayofweek.
8. Usa expanding para calcular el promedio acumulado de ingresos diarios.

---

## Resumen

- `pd.to_datetime`: conversión a datetime
- `dt` accessor: `.year`, `.month`, `.day`, `.dayofweek`, `.quarter`, `.strftime`
- `date_range` / `bdate_range`: generación de rangos
- `DateOffset`: desplazamiento temporal
- `resample`: cambio de frecuencia con agregación (D, W, ME, Q, YE)
- `asfreq`: cambio de frecuencia sin agregación
- `shift`, `diff`, `pct_change`: operaciones con rezago
- `rolling`, `expanding`, `ewm`: ventanas móviles
- `to_period`: conversión a períodos
- `tz_localize` / `tz_convert`: zonas horarias
- `Timedelta` / `Timestamp`: tipos temporales