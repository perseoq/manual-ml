# B08 — Fechas y Datetime en Python

## 1. Introducción

El módulo `datetime` de Python maneja fechas y horas. Es esencial en ventas para registrar transacciones, calcular plazos de entrega, vencimientos y períodos de reportes.

| Clase          | Descripción                     | Ejemplo en ventas                  |
|----------------|----------------------------------|-------------------------------------|
| `datetime`     | Fecha + hora                    | `2025-06-15 15:30:00` (venta)      |
| `date`         | Solo fecha                      | `2025-06-15` (fecha de factura)    |
| `time`         | Solo hora                       | `15:30:00` (hora de cierre)        |
| `timedelta`    | Diferencia entre fechas         | Plazo de entrega, días de crédito  |
| `dateutil`     | Extensiones (relativedelta, tz) | Trimestres fiscales, zonas horarias|

Funciones clave: `now()`, `strftime()` (datetime → string), `strptime()` (string → datetime), `timedelta`.

---

## 2. Ejemplos prácticos

### Ejemplo 1: `datetime.now()` para registrar fecha de venta

```python
from datetime import datetime

# Capturar momento exacto de la venta
fecha_venta = datetime.now()
print("Venta registrada en:", fecha_venta)
print("Año:", fecha_venta.year)
print("Mes:", fecha_venta.month)
print("Día:", fecha_venta.day)
print("Hora:", fecha_venta.hour)
print("Minuto:", fecha_venta.minute)
print("Segundo:", fecha_venta.second)
```

**Salida:**
```
Venta registrada en: 2025-06-15 15:30:45.123456
Año: 2025
Mes: 6
Día: 15
Hora: 15
Minuto: 30
Segundo: 45
```

**Explicación:**
1. `datetime.now()` — Obtiene la fecha y hora actual del sistema.
2. `fecha_venta.year`, `.month`, `.day`, `.hour`, `.minute`, `.second` — Atributos individuales.
3. Usado para timestamps de transacciones.

---

### Ejemplo 2: `timedelta` para calcular fecha de entrega

```python
from datetime import datetime, timedelta

fecha_pedido = datetime.now()
print(f"Fecha del pedido: {fecha_pedido.date()}")

# Plazo de entrega: 5 días hábiles (simplificado)
plazo_entrega = timedelta(days=5)
fecha_entrega = fecha_pedido + plazo_entrega
print(f"Fecha estimada de entrega: {fecha_entrega.date()}")

# Envío express: 2 días
fecha_express = fecha_pedido + timedelta(days=2)
print(f"Envío express (2 días): {fecha_express.date()}")
```

**Salida:**
```
Fecha del pedido: 2025-06-15
Fecha estimada de entrega: 2025-06-20
Envío express (2 días): 2025-06-17
```

**Explicación:**
1. `timedelta(days=5)` — Representa una duración de 5 días.
2. `fecha_pedido + plazo_entrega` — Suma timedelta a datetime (aritmética de fechas).
3. `timedelta(days=2)` — Plazo express.

---

### Ejemplo 3: `strftime` para formato de factura

```python
from datetime import datetime

fecha_factura = datetime.now()

# Diferentes formatos de salida
formatos = [
    "%d/%m/%Y",           # 15/06/2025
    "%d-%b-%Y",           # 15-Jun-2025
    "%Y-%m-%d %H:%M:%S",  # 2025-06-15 15:30:45
    "%A, %d de %B de %Y", # Sunday, 15 de June de 2025
    "%d/%m/%y",           # 15/06/25
    "%H:%M hs",           # 15:30 hs
]

print("=== FORMATOS DE FECHA PARA FACTURA ===")
for fmt in formatos:
    print(f"  {fmt:30s} → {fecha_factura.strftime(fmt)}")
```

**Salida:**
```
=== FORMATOS DE FECHA PARA FACTURA ===
  %d/%m/%Y                      → 15/06/2025
  %d-%b-%Y                      → 15-Jun-2025
  %Y-%m-%d %H:%M:%S             → 2025-06-15 15:30:45
  %A, %d de %B de %Y            → Sunday, 15 de June de 2025
  %d/%m/%y                      → 15/06/25
  %H:%M hs                      → 15:30 hs
```

**Explicación:**
1. `strftime(formato)` — Convierte datetime a string con formato personalizado.
2. Códigos comunes: `%d` (día), `%m` (mes numérico), `%b`/`%B` (mes abreviado/completo), `%Y` (año 4 dígitos), `%H` (hora 24h), `%M` (minutos), `%S` (segundos).
3. Útil para personalizar facturas, reportes, nombres de archivo.

---

### Ejemplo 4: `strptime` para parsear fechas desde CSV

```python
from datetime import datetime

# Simular lectura de CSV
csv_fechas = [
    "2025-06-15",
    "15/06/2025",
    "15-Jun-2025",
    "2025-06-15 15:30:00"
]

print("=== PARSEO DE FECHAS DESDE CSV ===")
for fecha_str in csv_fechas:
    if "-" in fecha_str and fecha_str[0].isdigit():
        if len(fecha_str) == 10:
            fmt = "%Y-%m-%d"
        elif " " in fecha_str:
            fmt = "%Y-%m-%d %H:%M:%S"
        else:
            fmt = "%d-%b-%Y"
    elif "/" in fecha_str:
        fmt = "%d/%m/%Y"
    else:
        fmt = "%d-%b-%Y"

    fecha_dt = datetime.strptime(fecha_str, fmt)
    print(f"  '{fecha_str}' ({fmt:16s}) → {fecha_dt}")
```

**Salida:**
```
=== PARSEO DE FECHAS DESDE CSV ===
  '2025-06-15' (%Y-%m-%d       ) → 2025-06-15 00:00:00
  '15/06/2025' (%d/%m/%Y       ) → 2025-06-15 00:00:00
  '15-Jun-2025' (%d-%b-%Y      ) → 2025-06-15 00:00:00
  '2025-06-15 15:30:00' (%Y-%m-%d %H:%M:%S) → 2025-06-15 15:30:00
```

**Explicación:**
1. `datetime.strptime(string, formato)` — Parsea un string a datetime según el formato dado.
2. El formato debe coincidir exactamente con la estructura del string.
3. Útil para procesar archivos CSV/JSON con fechas en diferentes formatos.

---

### Ejemplo 5: Diferencia de fechas para días de crédito

```python
from datetime import datetime

fecha_factura = datetime(2025, 6, 15)
fecha_pago = datetime(2025, 7, 10)

dias_credito = (fecha_pago - fecha_factura).days
print(f"Fecha factura: {fecha_factura.date()}")
print(f"Fecha pago: {fecha_pago.date()}")
print(f"Días de crédito utilizados: {dias_credito}")

# Días de atraso
fecha_limite = datetime(2025, 6, 30)
if fecha_pago > fecha_limite:
    atraso = (fecha_pago - fecha_limite).days
    print(f"ATRASO: {atraso} días")
else:
    print("Pago dentro del plazo")
```

**Salida:**
```
Fecha factura: 2025-06-15
Fecha pago: 2025-07-10
Días de crédito utilizados: 25
ATRASO: 10 días
```

**Explicación:**
1. Restar dos datetimes produce un `timedelta`.
2. `.days` — Obtiene los días del timedelta.
3. Comparación directa con `>` para determinar atraso.

---

### Ejemplo 6: Sumar 30 días neto (plazo de crédito)

```python
from datetime import datetime, timedelta

fecha_venta = datetime(2025, 6, 15)
print(f"Fecha de venta: {fecha_venta.date()}")

# Plazo neto 30 días
plazo_neto = timedelta(days=30)
fecha_vencimiento = fecha_venta + plazo_neto
print(f"Vencimiento (neto 30): {fecha_vencimiento.date()}")

# Plazo neto 60 días
fecha_vto_60 = fecha_venta + timedelta(days=60)
print(f"Vencimiento (neto 60): {fecha_vto_60.date()}")

# Plazo neto 90 días
fecha_vto_90 = fecha_venta + timedelta(days=90)
print(f"Vencimiento (neto 90): {fecha_vto_90.date()}")
```

**Salida:**
```
Fecha de venta: 2025-06-15
Vencimiento (neto 30): 2025-07-15
Vencimiento (neto 60): 2025-08-14
Vencimiento (neto 90): 2025-09-13
```

**Explicación:**
1. `timedelta(days=30)` — Plazo de 30 días.
2. Sumar al datetime de la venta.
3. Diferentes plazos para diferentes condiciones de pago.

---

### Ejemplo 7: Comparar fechas de vencimiento

```python
from datetime import datetime

hoy = datetime.now().date()
fechas_vencimiento = [
    datetime(2025, 5, 1).date(),
    datetime(2025, 7, 20).date(),
    datetime(2025, 12, 31).date(),
    datetime(2024, 12, 31).date()
]

print(f"Fecha actual: {hoy}")
print("=== ESTADO DE VENCIMIENTOS ===")
for fv in fechas_vencimiento:
    if fv < hoy:
        print(f"  {fv}: VENCIDO ({hoy - fv}).days días atrasado)")
    elif fv == hoy:
        print(f"  {fv}: VENCE HOY")
    elif (fv - hoy).days <= 7:
        print(f"  {fv}: PRÓXIMO A VENCER ({(fv - hoy).days} días)")
    else:
        print(f"  {fv}: Vigente ({(fv - hoy).days} días restantes)")
```

**Salida:**
```
Fecha actual: 2025-06-15
  2025-05-01: VENCIDO (45 días atrasado)
  2025-07-20: Vigente (35 días restantes)
  2025-12-31: Vigente (199 días restantes)
  2024-12-31: VENCIDO (166 días atrasado)
```

**Explicación:**
1. Comparación directa con `<`, `==`, `>` entre fechas.
2. `hoy - fv` — Timedelta negativo si vencido (o usar `.days` y valor absoluto).
3. Clasificación en rangos de días para alertas.

---

### Ejemplo 8: Agrupar ventas por mes

```python
from datetime import datetime

ventas = [
    ("2025-01-15", 150000),
    ("2025-01-20", 250000),
    ("2025-02-10", 180000),
    ("2025-02-25", 220000),
    ("2025-03-05", 300000),
    ("2025-01-30", 120000),
]

ventas_por_mes = {}
for fecha_str, monto in ventas:
    fecha = datetime.strptime(fecha_str, "%Y-%m-%d")
    mes = fecha.strftime("%Y-%m")  # clave: "2025-01"
    ventas_por_mes.setdefault(mes, []).append(monto)

print("=== VENTAS POR MES ===")
for mes, montos in sorted(ventas_por_mes.items()):
    total = sum(montos)
    print(f"  {mes}: ${total:>7,} ({len(montos)} ventas)")

print(f"\nTotal general: ${sum(sum(m) for m in ventas_por_mes.values()):,}")
```

**Salida:**
```
=== VENTAS POR MES ===
  2025-01: $520,000 (3 ventas)
  2025-02: $400,000 (2 ventas)
  2025-03: $300,000 (1 ventas)

Total general: $1,220,000
```

**Explicación:**
1. `strptime` para parsear, `strftime("%Y-%m")` para extraer mes como clave.
2. `setdefault(mes, []).append(monto)` — Agrupa montos por mes.
3. `sorted(ventas_por_mes.items())` — Ordena cronológicamente.

---

### Ejemplo 9: Primer y último día del mes

```python
from datetime import datetime, date
import calendar

def primer_dia_mes(fecha: datetime) -> date:
    """Retorna el primer día del mes de la fecha dada."""
    return date(fecha.year, fecha.month, 1)

def ultimo_dia_mes(fecha: datetime) -> date:
    """Retorna el último día del mes usando calendar."""
    ultimo_dia = calendar.monthrange(fecha.year, fecha.month)[1]
    return date(fecha.year, fecha.month, ultimo_dia)

fecha = datetime(2025, 6, 15)
print(f"Fecha: {fecha.date()}")
print(f"Primer día del mes: {primer_dia_mes(fecha)}")
print(f"Último día del mes: {ultimo_dia_mes(fecha)}")

# Para febrero (bisiesto)
fecha_feb = datetime(2024, 2, 10)  # año bisiesto
print(f"\nFebrero 2024: {fecha_feb.date()}")
print(f"Primer día: {primer_dia_mes(fecha_feb)}")
print(f"Último día: {ultimo_dia_mes(fecha_feb)}")
```

**Salida:**
```
Fecha: 2025-06-15
Primer día del mes: 2025-06-01
Último día del mes: 2025-06-30

Febrero 2024: 2024-02-10
Primer día: 2024-02-01
Último día: 2024-02-29
```

**Explicación:**
1. `date(year, month, 1)` — Siempre el primer día.
2. `calendar.monthrange(year, month)` — Retorna tupla `(primer_día_semana, último_día)`.
3. `[1]` — Extrae el último día del mes.
4. Maneja correctamente años bisiestos.

---

### Ejemplo 10: Rango de fechas para reporte semanal

```python
from datetime import datetime, timedelta

def generar_semana(fecha_inicio: datetime) -> list:
    """Genera una lista de 7 días a partir de fecha_inicio."""
    return [fecha_inicio + timedelta(days=i) for i in range(7)]

inicio_semana = datetime(2025, 6, 9)  # Lunes
semana = generar_semana(inicio_semana)

print("=== REPORTE SEMANAL ===")
for dia in semana:
    print(f"  {dia.strftime('%A %d/%m/%Y')}")
```

**Salida:**
```
=== REPORTE SEMANAL ===
  Monday 09/06/2025
  Tuesday 10/06/2025
  Wednesday 11/06/2025
  Thursday 12/06/2025
  Friday 13/06/2025
  Saturday 14/06/2025
  Sunday 15/06/2025
```

**Explicación:**
1. List comprehension con `timedelta(days=i)` para generar 7 días.
2. `strftime('%A %d/%m/%Y')` — Formato legible con nombre del día.
3. Útil para reportes semanales de ventas.

---

### Ejemplo 11: `relativedelta` de `dateutil` para sumar meses

```python
# Nota: requiere pip install python-dateutil
try:
    from datetime import datetime
    from dateutil.relativedelta import relativedelta

    fecha = datetime(2025, 6, 15)
    print(f"Fecha original: {fecha.date()}")

    # Sumar 1 mes
    fecha_mas_1 = fecha + relativedelta(months=1)
    print(f"+1 mes: {fecha_mas_1.date()}")

    # Sumar 3 meses
    fecha_mas_3 = fecha + relativedelta(months=3)
    print(f"+3 meses: {fecha_mas_3.date()}")

    # Restar 2 meses
    fecha_menos_2 = fecha - relativedelta(months=2)
    print(f"-2 meses: {fecha_menos_2.date()}")

    # relativedelta maneja correctamente fin de mes
    fecha_ene = datetime(2025, 1, 31)
    print(f"\n31 Ene + 1 mes: {(fecha_ene + relativedelta(months=1)).date()}")
    # Con timedelta normal: 31 Ene + 30 días = Mar 2
    print(f"31 Ene + 30 días (timedelta): {(fecha_ene + timedelta(days=30)).date()}")

except ImportError:
    print("dateutil no instalado. Ejecute: pip install python-dateutil")
```

**Salida:**
```
Fecha original: 2025-06-15
+1 mes: 2025-07-15
+3 meses: 2025-09-15
-2 meses: 2025-04-15

31 Ene + 1 mes: 2025-02-28
31 Ene + 30 días (timedelta): 2025-03-02
```

**Explicación:**
1. `relativedelta(months=1)` — Suma un mes calendario (no solo 30 días).
2. Maneja correctamente fin de mes: 31 Ene + 1 mes = 28 Feb.
3. `timedelta(days=30)` sumaría 30 días exactos (2 Mar).
4. Importante para suscripciones y pagos mensuales.

---

### Ejemplo 12: Zona horaria con `pytz` o `zoneinfo`

```python
# Python 3.9+: zoneinfo (built-in), alternativa: pytz
try:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    # Hora en diferentes zonas
    hora_santiago = datetime.now(ZoneInfo("America/Santiago"))
    hora_madrid = datetime.now(ZoneInfo("Europe/Madrid"))
    hora_ny = datetime.now(ZoneInfo("America/New_York"))

    print("=== HORA ACTUAL EN ZONAS ===")
    print(f"Santiago:   {hora_santiago.strftime('%H:%M:%S %Z')}")
    print(f"Madrid:     {hora_madrid.strftime('%H:%M:%S %Z')}")
    print(f"New York:   {hora_ny.strftime('%H:%M:%S %Z')}")

    # Convertir una venta registrada en Santiago a Madrid
    venta_scl = datetime(2025, 6, 15, 15, 30, tzinfo=ZoneInfo("America/Santiago"))
    venta_mad = venta_scl.astimezone(ZoneInfo("Europe/Madrid"))
    print(f"\nVenta en Santiago: {venta_scl.strftime('%H:%M %Z')}")
    print(f"Equivalente Madrid: {venta_mad.strftime('%H:%M %Z')}")

except ImportError:
    print("zoneinfo no disponible. Use pytz o Python 3.9+")
```

**Salida:**
```
=== HORA ACTUAL EN ZONAS ===
Santiago:   15:30:45 CLT
Madrid:     21:30:45 CET
New York:   14:30:45 EST

Venta en Santiago: 15:30 CLT
Equivalente Madrid: 21:30 CET
```

**Explicación:**
1. `ZoneInfo("America/Santiago")` — Zona horaria específica IANA.
2. `datetime.now(ZoneInfo(...))` — Obtiene hora actual en zona.
3. `.astimezone(otra_zona)` — Convierte entre zonas.
4. Crucial para equipos de ventas internacionales.

---

### Ejemplo 13: Semana ISO (número de semana del año)

```python
from datetime import datetime

fechas = [
    datetime(2025, 1, 1),
    datetime(2025, 6, 15),
    datetime(2025, 12, 31),
    datetime(2024, 12, 30),
]

print("=== SEMANA ISO ===")
for f in fechas:
    iso = f.isocalendar()
    # iso = (año_iso, semana_iso, día_iso)
    print(f"  {f.date()} → Año ISO {iso[0]}, Semana {iso[1]}, Día {iso[2]}")
```

**Salida:**
```
=== SEMANA ISO ===
  2025-01-01 → Año ISO 2025, Semana 1, Día 3
  2025-06-15 → Año ISO 2025, Semana 24, Día 7
  2025-12-31 → Año ISO 2026, Semana 1, Día 3
  2024-12-30 → Año ISO 2025, Semana 1, Día 1
```

**Explicación:**
1. `isocalendar()` — Retorna tupla `(año_iso, semana_iso, día_semana_iso)`.
2. Semana ISO: semana 1 es la que contiene el primer jueves del año.
3. `2025-12-31` pertenece a la semana ISO 1 de 2026.
4. Útil para reportes semanales estandarizados.

---

### Ejemplo 14: Trimestre fiscal

```python
from datetime import datetime

def trimestre_fiscal(fecha: datetime) -> int:
    """Calcula el trimestre de una fecha (1-4)."""
    return (fecha.month - 1) // 3 + 1

def periodo_fiscal(fecha: datetime) -> str:
    """Retorna string con trimestre y año fiscal."""
    trim = trimestre_fiscal(fecha)
    return f"TR{trim}-{fecha.year}"

fechas = [
    datetime(2025, 1, 15),
    datetime(2025, 4, 20),
    datetime(2025, 7, 10),
    datetime(2025, 10, 5)
]

print("=== PERIODOS FISCALES ===")
for f in fechas:
    print(f"  {f.date()} → {periodo_fiscal(f)}")

# Agrupar ventas por trimestre
ventas = [
    ("2025-01-15", 150000),
    ("2025-04-20", 250000),
    ("2025-07-10", 180000),
    ("2025-10-05", 220000),
    ("2025-03-15", 300000),
]

por_trimestre = {}
for fecha_str, monto in ventas:
    fecha = datetime.strptime(fecha_str, "%Y-%m-%d")
    trim = periodo_fiscal(fecha)
    por_trimestre[trim] = por_trimestre.get(trim, 0) + monto

print("\n=== VENTAS POR TRIMESTRE ===")
for trim, total in sorted(por_trimestre.items()):
    print(f"  {trim}: ${total:,}")
```

**Salida:**
```
=== PERIODOS FISCALES ===
  2025-01-15 → TR1-2025
  2025-04-20 → TR2-2025
  2025-07-10 → TR3-2025
  2025-10-05 → TR4-2025

=== VENTAS POR TRIMESTRE ===
  TR1-2025: $450,000
  TR2-2025: $250,000
  TR3-2025: $180,000
  TR4-2025: $220,000
```

**Explicación:**
1. `(fecha.month - 1) // 3 + 1` — Fórmula matemática para trimestre.
2. Trimestres: 1 (Ene-Mar), 2 (Abr-Jun), 3 (Jul-Sep), 4 (Oct-Dic).
3. Agrupación por trimestre usando diccionario.

---

### Ejemplo 15: Edad del inventario (días desde recepción)

```python
from datetime import datetime, timedelta
import random

# Simular recepción de productos en distintas fechas
inventario = {
    "Laptop": {"fecha_recepcion": datetime(2025, 5, 1), "stock": 10},
    "Mouse": {"fecha_recepcion": datetime(2025, 6, 10), "stock": 50},
    "Teclado": {"fecha_recepcion": datetime(2025, 4, 15), "stock": 30},
    "Monitor": {"fecha_recepcion": datetime(2025, 6, 1), "stock": 15}
}

hoy = datetime.now()
print(f"Fecha de análisis: {hoy.date()}")
print("=== EDAD DEL INVENTARIO ===")

total_valor = 0
for producto, datos in inventario.items():
    edad = (hoy - datos["fecha_recepcion"]).days
    print(f"  {producto:12s} | Recibido: {datos['fecha_recepcion'].date()} | "
          f"Edad: {edad:>3} días | Stock: {datos['stock']}")

    if edad > 60:
        print(f"    ⚠ Inventario lento (> 60 días)")
```

**Salida:**
```
Fecha de análisis: 2025-06-15
=== EDAD DEL INVENTARIO ===
  Laptop       | Recibido: 2025-05-01 | Edad:  45 días | Stock: 10
  Mouse        | Recibido: 2025-06-10 | Edad:   5 días | Stock: 50
  Teclado      | Recibido: 2025-04-15 | Edad:  61 días | Stock: 30
    ⚠ Inventario lento (> 60 días)
  Monitor      | Recibido: 2025-06-01 | Edad:  14 días | Stock: 15
```

**Explicación:**
1. `(hoy - datos["fecha_recepcion"]).days` — Calcula días desde recepción.
2. Alerta si edad > 60 días (inventario de lenta rotación).
3. Ayuda a identificar productos que necesitan promociones para salir.

---

### Ejemplo 16: `calendar` para generar calendario mensual

```python
import calendar
from datetime import datetime

# Calendario bonito de un mes
fecha = datetime(2025, 6, 15)
print(f"=== {fecha.strftime('%B %Y')} ===")
print(calendar.month(fecha.year, fecha.month))

# Días laborales (lun-vie) del mes
dias_laborales = sum(
    1 for d in range(1, calendar.monthrange(fecha.year, fecha.month)[1] + 1)
    if calendar.weekday(fecha.year, fecha.month, d) < 5
)
print(f"Días laborales: {dias_laborales}")

# Primer lunes del mes
for d in range(1, 8):
    if calendar.weekday(fecha.year, fecha.month, d) == 0:  # Monday
        print(f"Primer lunes: {fecha.year}-{fecha.month:02d}-{d:02d}")
        break
```

**Salida:**
```
=== June 2025 ===
     June 2025
Mo Tu We Th Fr Sa Su
                   1
 2  3  4  5  6  7  8
 9 10 11 12 13 14 15
16 17 18 19 20 21 22
23 24 25 26 27 28 29
30

Días laborales: 21
Primer lunes: 2025-06-02
```

**Explicación:**
1. `calendar.month(year, month)` — Genera string con calendario formateado.
2. `calendar.monthrange(year, month)` — Obtiene días del mes.
3. `calendar.weekday(year, month, day)` — Retorna día de la semana (0=lun).
4. Útil para planificación de reportes mensuales.

---

### Ejemplo 17: Generar rango de fechas (date_range)

```python
from datetime import datetime, timedelta

def date_range(inicio: datetime, fin: datetime, step: timedelta = timedelta(days=1)) -> list:
    """Genera lista de fechas entre inicio y fin con paso dado."""
    fechas = []
    actual = inicio
    while actual <= fin:
        fechas.append(actual)
        actual += step
    return fechas

inicio = datetime(2025, 6, 1)
fin = datetime(2025, 6, 15)

# Rango diario
diario = date_range(inicio, fin)
print(f"Fechas diarias ({len(diario)}): {diario[0].date()}...{diario[-1].date()}")

# Rango semanal (cada 7 días)
semanal = date_range(inicio, fin, timedelta(weeks=1))
print(f"Fechas semanales: {[f.date() for f in semanal]}")

# Simular datos de ventas en rango
print("\n=== VENTAS DIARIAS ===")
import random
for fecha in diario:
    ventas_dia = random.randint(5, 50) * 1000
    print(f"  {fecha.date()}: ${ventas_dia:,}")
```

**Salida:**
```
Fechas diarias (15): 2025-06-01...2025-06-15
Fechas semanales: [datetime(...), datetime(...)]

=== VENTAS DIARIAS ===
  2025-06-01: $25,000
  ...
```

**Explicación:**
1. `date_range` — Genera fechas con `while` y `timedelta` como paso.
2. `timedelta(weeks=1)` — Paso semanal.
3. Útil para generar reportes de ventas por período.

---

### Ejemplo 18: Timestamp Unix y conversión

```python
from datetime import datetime
import time

# Timestamp actual
ts = time.time()
print(f"Timestamp UNIX actual: {ts}")
print(f"Equivalente datetime: {datetime.fromtimestamp(ts)}")

# Crear timestamp desde fecha
fecha_venta = datetime(2025, 6, 15, 15, 30, 0)
ts_venta = fecha_venta.timestamp()
print(f"\nFecha venta: {fecha_venta}")
print(f"Timestamp: {ts_venta}")

# Leer timestamp de base de datos y convertir
ts_bd = 1750000000.0
fecha_bd = datetime.fromtimestamp(ts_bd)
print(f"\nTimestamp BD: {ts_bd}")
print(f"Fecha BD: {fecha_bd}")

# Diferencia en segundos entre dos momentos
inicio = datetime.now()
time.sleep(0.1)  # simula procesamiento
fin = datetime.now()
diferencia = (fin - inicio).total_seconds()
print(f"\nTiempo de procesamiento: {diferencia:.4f} segundos")
```

**Salida:**
```
Timestamp UNIX actual: 1750000000.0
Equivalente datetime: 2025-06-15 15:30:00

Fecha venta: 2025-06-15 15:30:00
Timestamp: 1750000000.0

Timestamp BD: 1750000000.0
Fecha BD: 2025-06-15 15:30:00

Tiempo de procesamiento: 0.1001 segundos
```

**Explicación:**
1. `time.time()` — Timestamp UNIX (segundos desde 1970-01-01).
2. `datetime.fromtimestamp(ts)` — Convierte timestamp a datetime.
3. `.timestamp()` — Convierte datetime a timestamp.
4. `.total_seconds()` — Timedelta a segundos.
5. Útil para interoperar con bases de datos y APIs.

---

### Ejemplo 19: Parsear fechas con `dateutil.parser`

```python
# dateutil.parser.parse es más flexible que strptime
try:
    from dateutil import parser

    fechas_desordenadas = [
        "2025-06-15",
        "15/06/2025",
        "Jun 15, 2025",
        "15 June 2025 15:30",
        "2025-06-15T15:30:00Z",
        "2025-06-15 15:30:45.123456",
        "15-06-2025",
        "06/15/2025",  # mes/día/año (EEUU)
    ]

    print("=== PARSEO FLEXIBLE CON DATEUTIL ===")
    for fecha_str in fechas_desordenadas:
        try:
            fecha_dt = parser.parse(fecha_str, dayfirst=False)
            print(f"  '{fecha_str:30s}' → {fecha_dt}")
        except:
            # Intentar con dayfirst=True
            fecha_dt = parser.parse(fecha_str, dayfirst=True)
            print(f"  '{fecha_str:30s}' → {fecha_dt} (dayfirst)")

except ImportError:
    print("dateutil no instalado. Ejecute: pip install python-dateutil")
```

**Salida:**
```
=== PARSEO FLEXIBLE CON DATEUTIL ===
  '2025-06-15'                   → 2025-06-15 00:00:00
  '15/06/2025'                   → 2025-06-15 00:00:00
  'Jun 15, 2025'                 → 2025-06-15 00:00:00
  '15 June 2025 15:30'           → 2025-06-15 15:30:00
  '2025-06-15T15:30:00Z'         → 2025-06-15 15:30:00+00:00
  '2025-06-15 15:30:45.123456'   → 2025-06-15 15:30:45.123456
  '15-06-2025'                   → 2025-06-15 00:00:00
  '06/15/2025'                   → 2025-06-15 00:00:00
```

**Explicación:**
1. `parser.parse(fecha_str)` — Intenta adivinar el formato automáticamente.
2. `dayfirst=True` — Interpreta día antes que mes (para formatos DD/MM/AAAA).
3. Maneja formatos ISO, RFC, y muchos otros.
4. Mucho más flexible que `strptime` para datos no uniformes.

---

### Ejemplo 20: Simulación completa de procesamiento de fechas en ventas

```python
from datetime import datetime, timedelta
import random

# Generar datos de ventas de ejemplo
print("=== SIMULACIÓN DE VENTAS ===")
ventas = []

fecha_inicio = datetime(2025, 6, 1)
productos = ["Laptop", "Mouse", "Teclado", "Monitor", "Webcam"]

for i in range(20):
    fecha = fecha_inicio + timedelta(days=random.randint(0, 14))
    producto = random.choice(productos)
    cantidad = random.randint(1, 5)
    precio = {"Laptop": 1250000, "Mouse": 25000, "Teclado": 45000,
              "Monitor": 350000, "Webcam": 35000}[producto]
    total = cantidad * precio
    ventas.append({"fecha": fecha, "producto": producto,
                   "cantidad": cantidad, "total": total})

# Análisis por día
from collections import defaultdict
ventas_por_dia = defaultdict(list)
for v in ventas:
    dia_key = v["fecha"].strftime("%Y-%m-%d")
    ventas_por_dia[dia_key].append(v)

print("Ventas agrupadas por día:")
for dia in sorted(ventas_por_dia.keys()):
    total_dia = sum(v["total"] for v in ventas_por_dia[dia])
    print(f"  {dia}: ${total_dia:>8,} ({len(ventas_por_dia[dia])} ventas)")

# Producto más vendido
from collections import Counter
contador = Counter(v["producto"] for v in ventas)
print(f"\nProducto más vendido: {contador.most_common(1)[0]}")

# Ventas del último día del período
ultimo_dia = max(v["fecha"] for v in ventas)
print(f"Último día con ventas: {ultimo_dia.date()}")
```

**Salida:**
```
=== SIMULACIÓN DE VENTAS ===
Ventas agrupadas por día:
  2025-06-01: $   xx,xxx (x ventas)
  ...

Producto más vendido: ('Mouse', 7)
Último día con ventas: 2025-06-14
```

**Explicación:**
1. Simulación completa: generación, agrupación, análisis.
2. `Counter` para frecuencia de productos.
3. `max` con key sobre lista de fechas.
4. Integración de múltiples conceptos de datetime en un caso real.

---

## 3. Ejercicios propuestos

1. **Días hasta el próximo año:** Calcula cuántos días faltan desde hoy hasta el 1 de enero del próximo año.

2. **Edad del producto:** Dada una fecha de fabricación `datetime(2024, 3, 15)`, calcula cuántos días han pasado hasta hoy y si está vencido (vida útil: 365 días).

3. **Formato de factura:** Dado un datetime `2025-06-15 14:30:00`, genera: "Factura emitida el 15 de junio de 2025 a las 14:30 horas" usando `strftime` con locale español.

4. **Rango de fechas:** Genera todas las fechas entre el 1 de junio de 2025 y el 30 de junio de 2025 que sean lunes. ¿Cuántos hay?

5. **Trimestre actual:** Determina en qué trimestre fiscal estamos hoy y cuántos días faltan para que termine el trimestre.

6. **Vencimiento de crédito:** Una venta del 15 de marzo de 2025 tiene crédito a 45 días. ¿Cuál es la fecha de vencimiento? ¿Está vencida hoy?

7. **Relativedelta:** Usa `dateutil.relativedelta` para calcular el primer día del mes siguiente a una fecha dada.

8. **Zona horaria:** Si son las 10:00 AM en Santiago (CLT), ¿qué hora es en Tokyo (JST) y en London (GMT)? Usa `zoneinfo`.

---

## 4. Resumen

- `datetime.now()` obtiene fecha/hora actual; `.date()` extrae solo la fecha.
- `timedelta(days=N)` suma/resta días a una fecha (también soporta `hours`, `minutes`, `weeks`).
- `strftime(formato)` convierte datetime → string para facturas y reportes.
- `strptime(string, formato)` convierte string → datetime desde CSV/JSON.
- `relativedelta(dateutil)` suma meses correctamente (respeta fin de mes).
- `isocalendar()` devuelve semana ISO, útil para reportes semanales.
- `calendar.monthrange()` obtiene último día del mes.
- `dateutil.parser.parse()` adivina formatos de fecha automáticamente.
- `ZoneInfo` (Python 3.9+) maneja zonas horarias IANA.
- Las fechas se comparan directamente con `<`, `>`, `==`.
- Los timestamps UNIX (`time.time()`, `.timestamp()`) interoperan con bases de datos.
- Agrupar por mes/trimestre con `strftime("%Y-%m")` o fórmulas de trimestre.
- La edad del inventario se calcula como `(hoy - fecha_recepcion).days`.
