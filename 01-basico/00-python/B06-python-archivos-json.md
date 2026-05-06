# B06 — Archivos y JSON en Python

## 1. Introducción

Python lee y escribe archivos de texto, CSV y JSON. Estos formatos son esenciales para persistir datos de ventas, catálogos e inventarios.

| Formato | Módulo          | Uso típico en ventas           |
|---------|-----------------|--------------------------------|
| TXT     | `open()`        | Logs, reportes simples         |
| CSV     | `csv`           | Exportar ventas, inventarios   |
| JSON    | `json`          | Catálogos, APIs, backup        |
| Excel   | `openpyxl`      | Reportes financieros           |

Modos de apertura: `"r"` (lectura), `"w"` (escritura), `"a"` (append), `"r+"` (lectura/escritura). Usar `with` garantiza cierre automático.

---

## 2. Ejemplos prácticos

### Ejemplo 1: Leer archivo CSV de ventas

```python
# Simulamos un archivo CSV
import csv, io

csv_ventas = """producto,cantidad,precio,total
Laptop,2,1250000,2500000
Mouse,5,25000,125000
Teclado,3,45000,135000
Monitor,1,350000,350000
"""

# Leer desde un string como si fuera archivo
archivo = io.StringIO(csv_ventas)
lector = csv.reader(archivo)
cabeceras = next(lector)
print("Cabeceras:", cabeceras)
for fila in lector:
    print(f"  {fila[0]:12s} | {fila[1]:>2} und. | ${int(fila[3]):>7,}")
```

**Salida:**
```
Cabeceras: ['producto', 'cantidad', 'precio', 'total']
  Laptop       |  2 und. | $2,500,000
  Mouse        |  5 und. | $  125,000
  Teclado      |  3 und. | $  135,000
  Monitor      |  1 und. | $  350,000
```

**Explicación:**
1. `csv.reader(archivo)` — Crea un lector CSV que itera sobre filas.
2. `next(lector)` — Obtiene la primera fila (cabeceras).
3. Cada fila es una lista de strings. Se accede por índice.
4. `io.StringIO` — Simula un archivo en memoria.

---

### Ejemplo 2: Escribir archivo CSV de reporte

```python
import csv

reporte = [
    ["Producto", "Cantidad", "Precio Unit.", "Total"],
    ["Laptop", 2, 1250000, 2500000],
    ["Mouse", 5, 25000, 125000],
    ["Teclado", 3, 45000, 135000]
]

with open("reporte_ventas.csv", "w", newline="") as f:
    escritor = csv.writer(f)
    escritor.writerows(reporte)

# Verificar contenido
with open("reporte_ventas.csv", "r") as f:
    print(f.read())
```

**Salida:**
```
Producto,Cantidad,Precio Unit.,Total
Laptop,2,1250000,2500000
Mouse,5,25000,125000
Teclado,3,45000,135000
```

**Explicación:**
1. `with open("reporte_ventas.csv", "w", newline="") as f:` — Abre en modo escritura. `newline=""` evita saltos de línea extra en Windows.
2. `csv.writer(f)` — Crea escritor CSV vinculado al archivo.
3. `writerows(reporte)` — Escribe todas las filas de una vez.
4. `"w"` — Sobrescribe el archivo si existe.

---

### Ejemplo 3: Leer archivo JSON de catálogo

```python
import json, io

json_catalogo = """
{
    "LAP-001": {"nombre": "Laptop Gamer", "precio": 1250000, "stock": 10},
    "MOU-002": {"nombre": "Mouse RGB", "precio": 25000, "stock": 50},
    "TCL-003": {"nombre": "Teclado Mecánico", "precio": 45000, "stock": 30}
}
"""

catalogo = json.loads(json_catalogo)
print("=== CATÁLOGO DESDE JSON ===")
for sku, datos in catalogo.items():
    print(f"  {sku}: {datos['nombre']:15s} | ${datos['precio']:>7,} | Stock: {datos['stock']}")
```

**Salida:**
```
=== CATÁLOGO DESDE JSON ===
  LAP-001: Laptop Gamer    | $1,250,000 | Stock: 10
  MOU-002: Mouse RGB       | $   25,000 | Stock: 50
  TCL-003: Teclado Mecánico | $   45,000 | Stock: 30
```

**Explicación:**
1. `json.loads(json_catalogo)` — Parsea string JSON a diccionario Python (`loads` = load string).
2. El resultado es un dict anidado navegable con `items()`.
3. `json.loads` convierte automáticamente: `{}` → dict, `[]` → list, `true` → `True`, `null` → `None`.

---

### Ejemplo 4: Escribir archivo JSON de inventario

```python
import json

inventario = {
    "Laptop": {"precio": 1250000, "stock": 10, "ubicacion": "A1"},
    "Mouse": {"precio": 25000, "stock": 50, "ubicacion": "B2"},
    "Teclado": {"precio": 45000, "stock": 30, "ubicacion": "B3"}
}

with open("inventario.json", "w", encoding="utf-8") as f:
    json.dump(inventario, f, indent=2, ensure_ascii=False)

# Verificar
with open("inventario.json", "r", encoding="utf-8") as f:
    print(f.read())
```

**Salida:**
```
{
  "Laptop": {
    "precio": 1250000,
    "stock": 10,
    "ubicacion": "A1"
  },
  "Mouse": {
    "precio": 25000,
    "stock": 50,
    "ubicacion": "B2"
  },
  "Teclado": {
    "precio": 45000,
    "stock": 30,
    "ubicacion": "B3"
  }
}
```

**Explicación:**
1. `json.dump(inventario, f, indent=2, ensure_ascii=False)` — Escribe el dict como JSON formateado.
2. `indent=2` — Sangría de 2 espacios para legibilidad.
3. `ensure_ascii=False` — Permite caracteres UTF-8 (tildes, ñ).
4. `"w"` — Sobrescribe.

---

### Ejemplo 5: `with open` seguro (cierre automático)

```python
# MAL: olvidar f.close() puede corromper datos
f = open("reporte.txt", "w")
f.write("Venta realizada")
# f.close()  # Fácil de olvidar

# BIEN: with garantiza cierre automático
with open("reporte.txt", "w", encoding="utf-8") as f:
    f.write("Venta realizada\n")
    f.write("Total: $1,250,000\n")
# Al salir del bloque, Python cierra el archivo automáticamente

with open("reporte.txt", "r") as f:
    print(f.read())
```

**Salida:**
```
Venta realizada
Total: $1,250,000
```

**Explicación:**
1. `with open(...) as f:` — Context manager: al salir del bloque (incluso por error), cierra el archivo.
2. Se evitan fugas de recursos y datos corruptos.
3. Siempre usar `with` para archivos.

---

### Ejemplo 6: `csv.DictReader` para leer CSV con cabeceras

```python
import csv, io

csv_ventas = """producto,cantidad,precio,total
Laptop,2,1250000,2500000
Mouse,5,25000,125000
Teclado,3,45000,135000
"""

archivo = io.StringIO(csv_ventas)
lector = csv.DictReader(archivo)

print("=== VENTAS (DictReader) ===")
for fila in lector:
    total = int(fila["cantidad"]) * int(fila["precio"])
    print(f"  {fila['producto']:12s} | {fila['cantidad']} und. | ${total:>7,}")
```

**Salida:**
```
=== VENTAS (DictReader) ===
  Laptop       | 2 und. | $2,500,000
  Mouse        | 5 und. | $  125,000
  Teclado      | 3 und. | $  135,000
```

**Explicación:**
1. `csv.DictReader(archivo)` — Lee CSV usando la primera fila como nombres de clave.
2. Cada fila es un `OrderedDict` accesible por nombre de columna: `fila["producto"]`.
3. Más legible que indices numéricos.

---

### Ejemplo 7: Escribir CSV con cabeceras usando DictWriter

```python
import csv

ventas = [
    {"producto": "Laptop", "cantidad": 2, "precio": 1250000, "total": 2500000},
    {"producto": "Mouse", "cantidad": 5, "precio": 25000, "total": 125000},
]

with open("ventas_dict.csv", "w", newline="", encoding="utf-8") as f:
    campos = ["producto", "cantidad", "precio", "total"]
    escritor = csv.DictWriter(f, fieldnames=campos)
    escritor.writeheader()
    escritor.writerows(ventas)

with open("ventas_dict.csv", "r") as f:
    print(f.read())
```

**Salida:**
```
producto,cantidad,precio,total
Laptop,2,1250000,2500000
Mouse,5,25000,125000
```

**Explicación:**
1. `csv.DictWriter(f, fieldnames=campos)` — Escritor que acepta diccionarios.
2. `writeheader()` — Escribe la fila de cabeceras.
3. `writerows(ventas)` — Escribe cada dict como fila, mapeando claves a columnas.
4. El orden de las columnas sigue `fieldnames`.

---

### Ejemplo 8: Append a CSV existente

```python
import csv

# Crear archivo inicial
with open("ventas_diarias.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["fecha", "producto", "monto"])

# Append: agregar ventas del día
with open("ventas_diarias.csv", "a", newline="") as f:
    w = csv.writer(f)
    w.writerow(["2025-06-15", "Laptop", 1250000])
    w.writerow(["2025-06-15", "Mouse", 25000])

with open("ventas_diarias.csv", "r") as f:
    print(f.read())
```

**Salida:**
```
fecha,producto,monto
2025-06-15,Laptop,1250000
2025-06-15,Mouse,25000
```

**Explicación:**
1. `"a"` — Modo append: agrega al final sin sobrescribir.
2. La cabecera se escribe solo una vez.
3. Cada `writerow()` agrega una línea al final del archivo.

---

### Ejemplo 9: `csv.writer` con dialect personalizado (punto y coma)

```python
import csv

datos = [
    ["Producto", "Precio", "Stock"],
    ["Laptop", "1250000", "10"],
    ["Mouse", "25000", "50"]
]

with open("inventario_ps.csv", "w", newline="", encoding="utf-8") as f:
    escritor = csv.writer(f, delimiter=";", quotechar='"')
    escritor.writerows(datos)

with open("inventario_ps.csv", "r") as f:
    print(f.read())
```

**Salida:**
```
Producto;Precio;Stock
Laptop;1250000;10
Mouse;25000;50
```

**Explicación:**
1. `delimiter=";"` — Usa punto y coma como separador (útil cuando los datos contienen comas).
2. `quotechar='"'` — Carácter para encerrar campos con separador interno.
3. Útil para configuraciones regionales donde la coma es separador decimal.

---

### Ejemplo 10: `json.loads` desde API simulada

```python
import json

# Simular respuesta de API de proveedor
respuesta_api = '{"status": "ok", "data": [{"sku": "LAP-001", "nombre": "Laptop", "precio": 1250000}]}'

datos_api = json.loads(respuesta_api)
print("Status:", datos_api["status"])
print("Productos:")
for item in datos_api["data"]:
    print(f"  {item['sku']}: {item['nombre']} - ${item['precio']:,}")
```

**Salida:**
```
Status: ok
Productos:
  LAP-001: Laptop - $1,250,000
```

**Explicación:**
1. `json.loads(respuesta_api)` — Convierte JSON string a dict Python.
2. Navegación normal con claves: `datos_api["status"]`, `datos_api["data"]`.
3. Simula recibir datos desde una API REST de proveedor.

---

### Ejemplo 11: `json.dumps` con formato para visualización

```python
import json

ventas_resumen = {
    "fecha": "2025-06-15",
    "total_ventas": 4,
    "ingresos": 3110000,
    "productos": ["Laptop", "Mouse", "Teclado", "Monitor"]
}

# Sin indent (una línea)
compacto = json.dumps(ventas_resumen)
print("Compacto:", compacto)

# Con indent para legibilidad
formateado = json.dumps(ventas_resumen, indent=2, ensure_ascii=False)
print("\nFormateado:")
print(formateado)

# Ordenado por clave
ordenado = json.dumps(ventas_resumen, indent=2, sort_keys=True)
print("\nOrdenado:")
print(ordenado)
```

**Salida:**
```
Compacto: {"fecha": "2025-06-15", "total_ventas": 4, "ingresos": 3110000, "productos": ["Laptop", "Mouse", "Teclado", "Monitor"]}

Formateado:
{
  "fecha": "2025-06-15",
  "total_ventas": 4,
  "ingresos": 3110000,
  "productos": [
    "Laptop",
    "Mouse",
    "Teclado",
    "Monitor"
  ]
}

Ordenado:
{
  "fecha": "2025-06-15",
  "ingresos": 3110000,
  "productos": [
    "Laptop",
    "Mouse",
    "Teclado",
    "Monitor"
  ],
  "total_ventas": 4
}
```

**Explicación:**
1. `json.dumps(dict)` — Convierte dict Python a string JSON (`dumps` = dump string).
2. `indent=2` — Formato legible con sangría.
3. `sort_keys=True` — Ordena alfabéticamente las claves.

---

### Ejemplo 12: Leer Excel con `openpyxl` (si está instalado)

```python
# Nota: requiere pip install openpyxl
try:
    import openpyxl
    # Crear libro de ejemplo
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Ventas"
    ws.append(["Producto", "Cantidad", "Precio", "Total"])
    ws.append(["Laptop", 2, 1250000, 2500000])
    ws.append(["Mouse", 5, 25000, 125000])
    wb.save("ventas.xlsx")

    # Leer el archivo
    wb2 = openpyxl.load_workbook("ventas.xlsx")
    ws2 = wb2.active
    print("=== VENTAS DESDE EXCEL ===")
    for fila in ws2.iter_rows(min_row=1, values_only=True):
        print(f"  {fila}")
except ImportError:
    print("openpyxl no instalado. Ejecute: pip install openpyxl")
```

**Salida:**
```
=== VENTAS DESDE EXCEL ===
  ('Producto', 'Cantidad', 'Precio', 'Total')
  ('Laptop', 2, 1250000, 2500000)
  ('Mouse', 5, 25000, 125000)
```

**Explicación:**
1. `openpyxl.Workbook()` — Crea un libro Excel nuevo.
2. `ws.append([...])` — Agrega filas.
3. `wb.save("ventas.xlsx")` — Guarda el archivo.
4. `ws2.iter_rows(values_only=True)` — Itera filas obteniendo solo valores (no objetos de celda).

---

### Ejemplo 13: Manejar `FileNotFoundError` al leer

```python
try:
    with open("archivo_inexistente.csv", "r") as f:
        contenido = f.read()
except FileNotFoundError:
    print("Error: El archivo no existe. Verifique la ruta.")
except PermissionError:
    print("Error: No tiene permisos para leer este archivo.")
else:
    print("Archivo leído exitosamente.")
```

**Salida:**
```
Error: El archivo no existe. Verifique la ruta.
```

**Explicación:**
1. `try/except FileNotFoundError` — Captura el error de archivo no encontrado.
2. `else:` — Se ejecuta solo si no hubo excepción.
3. Siempre usar manejo de errores al leer archivos del usuario.

---

### Ejemplo 14: `encoding="utf-8"` para caracteres especiales

```python
with open("productos_especiales.txt", "w", encoding="utf-8") as f:
    f.write("Productos con tildes: café, estación, máquina\n")
    f.write("Precios: $12.500, €450, £350\n")

with open("productos_especiales.txt", "r", encoding="utf-8") as f:
    print(f.read())

# Sin UTF-8, esto podría fallar
with open("productos_especiales.txt", "r", encoding="latin-1") as f:
    print("Con latin-1:", f.read())
```

**Salida:**
```
Productos con tildes: café, estación, máquina
Precios: $12.500, €450, £350

Con latin-1: Productos con tildes: cafÃ©, estaciÃ³n, mÃ¡quina
Precios: $12.500, â‚¬450, Â£350
```

**Explicación:**
1. `encoding="utf-8"` — Codificación universal con soporte para tildes, ñ, símbolos.
2. Usar UTF-8 consistentemente al leer y escribir evita caracteres corruptos.
3. `latin-1` (ISO-8859-1) no soporta todos los caracteres UTF-8.

---

### Ejemplo 15: Leer múltiples archivos con `glob`

```python
import glob, csv, os

# Crear archivos de ejemplo
for mes in ["enero", "febrero", "marzo"]:
    with open(f"ventas_{mes}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["producto", "total"])
        w.writerow(["Laptop", 5000000])
        w.writerow(["Mouse", 250000])

# Leer todos
print("=== CONSOLIDADO MENSUAL ===")
total_general = 0
for archivo in sorted(glob.glob("ventas_*.csv")):
    with open(archivo, "r") as f:
        lector = csv.DictReader(f)
        total_mes = sum(int(fila["total"]) for fila in lector)
        print(f"  {archivo}: ${total_mes:,}")
        total_general += total_mes

print(f"\nTotal general: ${total_general:,}")

# Limpiar
for mes in ["enero", "febrero", "marzo"]:
    os.remove(f"ventas_{mes}.csv")
```

**Salida:**
```
=== CONSOLIDADO MENSUAL ===
  ventas_enero.csv: $5,250,000
  ventas_febrero.csv: $5,250,000
  ventas_marzo.csv: $5,250,000

Total general: $15,750,000
```

**Explicación:**
1. `glob.glob("ventas_*.csv")` — Busca todos los archivos que coinciden con el patrón.
2. Itera sobre cada archivo, lo abre y lo procesa.
3. Acumula totales de cada archivo en `total_general`.
4. `os.remove()` — Elimina los archivos temporales.

---

### Ejemplo 16: `pathlib` para manejo moderno de rutas

```python
from pathlib import Path
import json

# Crear directorio si no existe
datos_dir = Path("datos_ventas")
datos_dir.mkdir(exist_ok=True)

# Guardar archivo
archivo_json = datos_dir / "ventas_diarias.json"
ventas = {"fecha": "2025-06-15", "total": 1250000}

with open(archivo_json, "w") as f:
    json.dump(ventas, f)

# Verificar
print(f"Archivo: {archivo_json}")
print(f"Existe: {archivo_json.exists()}")
print(f"Tamaño: {archivo_json.stat().st_size} bytes")

# Leer
with open(archivo_json, "r") as f:
    print(f"Contenido: {json.load(f)}")

# Limpiar
import shutil
shutil.rmtree(datos_dir)
```

**Salida:**
```
Archivo: datos_ventas/ventas_diarias.json
Existe: True
Tamaño: 37 bytes
Contenido: {'fecha': '2025-06-15', 'total': 1250000}
```

**Explicación:**
1. `Path("datos_ventas")` — Objeto Path con métodos útiles.
2. `datos_dir.mkdir(exist_ok=True)` — Crea directorio sin error si ya existe.
3. `datos_dir / "ventas_diarias.json"` — Operador `/` para construir rutas (cross-platform).
4. `archivo_json.exists()`, `.stat().st_size` — Métodos de Path.

---

### Ejemplo 17: `tempfile` para archivos temporales

```python
import tempfile, csv, json

# Crear archivo temporal
with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
    ruta_temp = f.name
    w = csv.writer(f)
    w.writerow(["producto", "precio"])
    w.writerow(["Laptop", 1250000])

print(f"Archivo temporal creado en: {ruta_temp}")

# Leer y procesar
with open(ruta_temp, "r") as f:
    lector = csv.DictReader(f)
    for fila in lector:
        print(f"  {fila['producto']}: ${int(fila['precio']):,}")

# Limpiar
import os
os.unlink(ruta_temp)
print(f"Archivo temporal eliminado")
```

**Salida:**
```
Archivo temporal creado en: /tmp/tmpXXXXXX.csv
  Laptop: $1,250,000
Archivo temporal eliminado
```

**Explicación:**
1. `tempfile.NamedTemporaryFile(...)` — Crea archivo temporal con nombre único.
2. `delete=False` — No elimina al cerrar (para poder leerlo después).
3. `f.name` — Ruta completa del archivo temporal.
4. `os.unlink()` — Elimina manualmente.

---

### Ejemplo 18: `shutil` para copiar y mover archivos de respaldo

```python
import shutil, json, os

# Crear archivo original
with open("inventario_actual.json", "w") as f:
    json.dump({"Laptop": {"stock": 10}}, f)

# Crear respaldo automático
shutil.copy2("inventario_actual.json", "backup_inventario.json")
print("Respaldo creado:", os.path.exists("backup_inventario.json"))

# Mover (renombrar) archivo de respaldo
os.rename("backup_inventario.json", "backup_2025_06_15.json")
print("Archivos en directorio:", [f for f in os.listdir(".") if "backup" in f or "inventario" in f])

# Limpiar
os.remove("inventario_actual.json")
os.remove("backup_2025_06_15.json")
```

**Salida:**
```
Respaldo creado: True
Archivos en directorio: ['backup_2025_06_15.json', 'inventario_actual.json']
```

**Explicación:**
1. `shutil.copy2(origen, destino)` — Copia archivo preservando metadatos.
2. `os.rename()` — Renombra o mueve un archivo.
3. Útil para crear backups automáticos antes de modificar inventarios.

---

### Ejemplo 19: Actualizar un JSON existente (leer-modificar-escribir)

```python
import json

# Crear archivo inicial
with open("stock.json", "w") as f:
    json.dump({"Laptop": 10, "Mouse": 50}, f)

# Leer, modificar, escribir
with open("stock.json", "r+") as f:
    inventario = json.load(f)
    # Vender 2 laptops
    inventario["Laptop"] -= 2
    # Agregar nuevo producto
    inventario["Teclado"] = 30
    # Volver al inicio y sobrescribir
    f.seek(0)
    json.dump(inventario, f, indent=2)
    f.truncate()

with open("stock.json", "r") as f:
    print(f.read())
```

**Salida:**
```
{
  "Laptop": 8,
  "Mouse": 50,
  "Teclado": 30
}
```

**Explicación:**
1. `"r+"` — Modo lectura+escritura (no sobrescribe al abrir).
2. `json.load(f)` — Lee el contenido actual.
3. `f.seek(0)` — Vuelve al inicio del archivo para sobrescribir.
4. `f.truncate()` — Elimina contenido sobrante si el nuevo JSON es más corto.

---

### Ejemplo 20: Procesar CSV grande línea por línea (eficiente)

```python
import csv, io

# Simular archivo grande
csv_grande = "\n".join([f"producto_{i},{i*1000},{i*5}" for i in range(1, 1001)])

archivo = io.StringIO(csv_grande)
lector = csv.reader(archivo)

total_ingresos = 0
lineas_procesadas = 0
for fila in lector:
    if lineas_procesadas == 0:
        print("Procesando archivo de 1000 registros...")
    cantidad = int(fila[2])
    precio = int(fila[1])
    total_ingresos += cantidad * precio
    lineas_procesadas += 1

print(f"Registros procesados: {lineas_procesadas}")
print(f"Ingreso total estimado: ${total_ingresos:,}")
```

**Salida:**
```
Procesando archivo de 1000 registros...
Registros procesados: 1000
Ingreso total estimado: $16,671,675,000
```

**Explicación:**
1. Itera línea por línea con `for fila in lector` — no carga todo en memoria.
2. `csv.reader` es un iterador eficiente para archivos grandes.
3. Procesa 1000 registros en milisegundos.
4. Estrategia adecuada para archivos de millones de registros.

---

## 3. Ejercicios propuestos

1. **CSV a JSON:** Lee un archivo CSV con columnas `producto, precio, stock` y conviértelo a un archivo JSON con `json.dump`.

2. **Append de ventas:** Crea un script que agregue una nueva venta a un archivo CSV existente. Cada venta tiene: fecha, producto, cantidad, precio total.

3. **Backup automático:** Antes de modificar un archivo JSON de inventario, crea una copia de respaldo usando `shutil.copy2`.

4. **Consolidar múltiples CSVs:** Dados 3 archivos CSV con ventas diarias, escribe un script que los lea y genere un archivo resumen JSON con el total y promedio.

5. **DictWriter con filtro:** Lee un CSV de inventario y escribe un nuevo CSV solo con los productos cuyo stock > 0.

6. **Pathlib explorer:** Usa `pathlib.Path` para listar todos los archivos `.csv` y `.json` en un directorio, mostrando su tamaño y fecha de modificación.

7. **Validación JSON:** Lee un archivo JSON, valida que tenga las claves esperadas ("producto", "precio", "stock") usando `try/except KeyError`.

8. **Procesar con encoding:** Lee un archivo CSV con encoding "latin-1" y escríbelo como UTF-8, manejando caracteres especiales.

---

## 4. Resumen

- `open()` con modos `"r"`, `"w"`, `"a"`, `"r+"` para lectura/escritura/append.
- `with open(...) as f:` garantiza cierre automático (context manager).
- `csv.reader` y `csv.DictReader` leen CSV; `csv.writer` y `csv.DictWriter` escriben.
- `json.loads()` y `json.dumps()` trabajan con strings; `json.load()` y `json.dump()` con archivos.
- `openpyxl` permite leer/escribir Excel.
- `pathlib.Path` ofrece manejo moderno de rutas (operador `/`).
- `glob.glob(patrón)` busca archivos por patrón.
- `tempfile` crea archivos temporales seguros.
- `shutil.copy2` copia archivos; `os.rename` renombra.
- Siempre especificar `encoding="utf-8"` para compatibilidad.
- Manejar `FileNotFoundError`, `PermissionError` con `try/except`.
- Procesar archivos grandes línea por línea (no cargar todo en memoria).
