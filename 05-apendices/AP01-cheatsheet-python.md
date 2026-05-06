# AP01 — Cheatsheet Python

## 1. Tipos de Datos Básicos

```python
# Enteros, flotantes, booleanos, cadenas
x = 42              # int
y = 3.1416          # float
z = True            # bool
s = "ventas 2024"   # str

# Conversión entre tipos
int("42")           # 42
float("3.14")       # 3.14
str(100)            # "100"
bool(1)             # True
bool(0)             # False
bool("")            # False
bool("texto")       # True

# Verificar tipo
type(x)             # <class 'int'>
isinstance(x, int)  # True
isinstance(s, (str, int))  # True
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*1. Tipos de Datos Básicos.*

1. Enteros, flotantes, booleanos, cadenas
2. Conversión entre tipos
3. Verificar tipo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 2. Listas

```python
# Creación
lista = [1, 2, 3, 4, 5]
mixta = [1, "texto", 3.14, True]
vacia = []
anidada = [[1, 2], [3, 4]]

# Indexación y slicing
lista[0]        # 1
lista[-1]       # 5
lista[1:3]      # [2, 3]
lista[::2]      # [1, 3, 5]
lista[::-1]     # [5, 4, 3, 2, 1]

# Métodos principales
lista.append(6)         # [1,2,3,4,5,6]
lista.extend([7, 8])    # [1,2,3,4,5,6,7,8]
lista.insert(0, 0)      # [0,1,2,3,4,5,6,7,8]
lista.remove(3)         # elimina el 3
lista.pop()             # elimina y retorna el último
lista.pop(0)            # elimina y retorna índice 0
lista.index(4)          # índice del valor 4
lista.count(2)          # cuántas veces aparece 2
lista.sort()            # ordena in-place
lista.reverse()         # invierte in-place
sorted(lista)           # retorna copia ordenada
len(lista)              # longitud
sum(lista)              # suma
max(lista)              # máximo
min(lista)              # mínimo
any([False, True])      # True
all([True, False])      # False
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*2. Listas.*

1. Creación
2. Indexación y slicing
3. Métodos principales

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 3. Diccionarios

```python
# Creación
dic = {"nombre": "Laptop", "precio": 899.99, "stock": 15}
vacio = {}
dict(zip(["a", "b"], [1, 2]))  # {'a': 1, 'b': 2}

# Acceso
dic["nombre"]              # 'Laptop'
dic.get("precio")          # 899.99
dic.get("descuento", 0)    # 0 (default si no existe)
dic.setdefault("descuento", 10)  # setea si no existe

# Modificación
dic["precio"] = 799.99     # actualiza
dic["categoria"] = "tech"  # agrega
dic.update({"stock": 20, "marca": "Dell"})  # merge

# Eliminación
del dic["marca"]
dic.pop("descuento")
dic.pop("inexistente", None)  # sin error
dic.clear()                  # vacía el diccionario

# Iteración
for k, v in dic.items():   # clave y valor
for k in dic.keys():       # solo claves
for v in dic.values():     # solo valores

# Comprensión de diccionarios
{x: x**2 for x in range(5)}      # {0:0, 1:1, 2:4, 3:9, 4:16}
{k: v for k, v in dic.items() if v > 100}
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*3. Diccionarios.*

1. Creación
2. Acceso
3. Modificación
4. Eliminación
5. Iteración
6. Comprensión de diccionarios

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 4. Sets

```python
# Creación
s = {1, 2, 3, 4, 5}
t = {4, 5, 6, 7, 8}
vacio = set()

# Operaciones
s.add(6)            # agrega elemento
s.remove(6)         # error si no existe
s.discard(10)       # sin error si no existe
s.pop()             # elimina elemento arbitrario
s.clear()           # vacía

# Operaciones de conjunto
s | t               # unión: {1,2,3,4,5,6,7,8}
s & t               # intersección: {4,5}
s - t               # diferencia: {1,2,3}
s ^ t               # diferencia simétrica: {1,2,3,6,7,8}
s.issubset(t)       # False
s.issuperset(t)     # False
s.isdisjoint(t)     # False
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*4. Sets.*

1. Creación
2. Operaciones
3. Operaciones de conjunto

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 5. Control de Flujo

```python
# if-elif-else
edad = 18
if edad < 13:
    categoria = "niño"
elif edad < 18:
    categoria = "adolescente"
else:
    categoria = "adulto"

# Operador ternario
categoria = "adulto" if edad >= 18 else "menor"

# for loop
for i in range(5):           # 0,1,2,3,4
for i in range(2, 10, 2):    # 2,4,6,8
for i, val in enumerate(lista):
for k, v in dic.items():
for a, b in zip(lista1, lista2):

# while loop
contador = 0
while contador < 5:
    contador += 1

# break / continue / else
for x in range(10):
    if x == 3: continue       # salta iteración
    if x == 7: break          # sale del loop
else:
    print("Loop completado sin break")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*5. Control de Flujo.*

1. if-elif-else
2. Operador ternario
3. for loop
4. while loop
5. break / continue / else

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 6. Funciones

```python
# Definición básica
def sumar(a, b):
    return a + b

# Valores por defecto
def saludar(nombre, saludo="Hola"):
    return f"{saludo}, {nombre}"

# *args (tupla variable) y **kwargs (dict variable)
def log(*args, **kwargs):
    print("args:", args)
    print("kwargs:", kwargs)

log(1, 2, 3, nivel="INFO", modulo="ventas")

# Type hints
def procesar_venta(monto: float, impuesto: float = 0.16) -> float:
    return monto * (1 + impuesto)

# Documentación (docstring)
def calcular_iva(monto: float) -> float:
    """Calcula el IVA (16%) de un monto de venta."""
    return monto * 0.16

# Funciones anidadas
def exterior(x):
    def interior(y):
        return x + y
    return interior

suma_5 = exterior(5)
print(suma_5(3))  # 8
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*6. Funciones.*

1. Definición básica
2. Valores por defecto
3. *args (tupla variable) y **kwargs (dict variable)
4. Type hints
5. Documentación (docstring)
6. Funciones anidadas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 7. Lambdas

```python
# Sintaxis: lambda args: expresión
cuadrado = lambda x: x**2
print(cuadrado(4))  # 16

# Uso común con sorted, map, filter, reduce
ventas = [("enero", 100), ("febrero", 300), ("marzo", 200)]
sorted(ventas, key=lambda x: x[1])           # ordenar por monto
sorted(ventas, key=lambda x: x[0])           # ordenar por mes
sorted(ventas, key=lambda x: -x[1])          # descendente

list(map(lambda x: x * 1.16, [100, 200]))    # [116.0, 232.0]
list(filter(lambda x: x > 150, [100, 200]))  # [200]

from functools import reduce
reduce(lambda a, b: a + b, [1, 2, 3, 4])    # 10
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*7. Lambdas.*

1. Sintaxis: lambda args: expresión
2. Uso común con sorted, map, filter, reduce

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 8. Comprehensions

```python
# List comprehension
cuadrados = [x**2 for x in range(10)]           # [0,1,4,9,...,81]
pares = [x for x in range(20) if x % 2 == 0]
productos = [(a, b) for a in [1,2] for b in [3,4]]
ventas_mayores = [v for v in ventas if v[1] > 200]

# Dict comprehension
cuadrados_dict = {x: x**2 for x in range(5)}
ventas_dict = {v[0]: v[1] * 1.16 for v in ventas}

# Set comprehension
pares_set = {x for x in range(10) if x % 2 == 0}

# Generator expression (no almacena en memoria)
suma_cuadrados = sum(x**2 for x in range(1000))
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*8. Comprehensions.*

1. List comprehension
2. Dict comprehension
3. Set comprehension
4. Generator expression (no almacena en memoria)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 9. Manejo de Archivos

```python
# Leer archivo
with open("ventas.csv", "r") as f:
    contenido = f.read()           # todo el archivo
    lineas = f.readlines()         # lista de líneas
    for linea in f:                # iterar línea por línea
        print(linea.strip())

# Escribir archivo
with open("reporte.txt", "w") as f:
    f.write("Reporte de ventas\n")
    f.writelines(["linea1\n", "linea2\n"])

# Append
with open("log.txt", "a") as f:
    f.write("nueva entrada\n")

# Context manager para csv
import csv
with open("ventas.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row["producto"], row["monto"])

with open("salida.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["producto", "monto"])
    writer.writerows([("laptop", 899), ("mouse", 25)])

# JSON
import json
with open("datos.json", "r") as f:
    data = json.load(f)           # dict/list desde JSON
with open("salida.json", "w") as f:
    json.dump(data, f, indent=2)
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*9. Manejo de Archivos.*

1. Leer archivo
2. Escribir archivo
3. Append
4. Context manager para csv
5. JSON

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 10. datetime

```python
from datetime import datetime, date, time, timedelta

# Obtener fecha/hora actual
ahora = datetime.now()
hoy = date.today()

# Crear fechas manualmente
fecha = datetime(2024, 12, 25)
fecha = datetime.strptime("2024-12-25", "%Y-%m-%d")

# Formatear
fecha_str = ahora.strftime("%Y-%m-%d %H:%M:%S")
fecha_str = ahora.strftime("%d/%m/%Y")
fecha_str = ahora.strftime("%A, %d de %B de %Y")

# Componentes
ahora.year, ahora.month, ahora.day
ahora.hour, ahora.minute, ahora.second

# Operaciones con timedelta
ayer = ahora - timedelta(days=1)
manana = ahora + timedelta(days=1)
dentro_de_3h = ahora + timedelta(hours=3)

# Diferencia entre fechas
diff = datetime(2025, 1, 1) - datetime(2024, 1, 1)
diff.days        # 366
diff.total_seconds()  # en segundos

# Parsear fechas comunes
from dateutil.parser import parse
fecha = parse("2024-12-25")
fecha = parse("Dec 25, 2024")
fecha = parse("25/12/2024", dayfirst=True)
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*10. datetime.*

1. Obtener fecha/hora actual
2. Crear fechas manualmente
3. Formatear
4. Componentes
5. Operaciones con timedelta
6. Diferencia entre fechas
7. Parsear fechas comunes

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 11. Strings

```python
s = "  analisis de ventas  "

# Manipulación
s.upper()           # "  ANALISIS DE VENTAS  "
s.lower()           # "  analisis de ventas  "
s.strip()           # "analisis de ventas"
s.replace("ventas", "datos")
s.split()           # ["analisis", "de", "ventas"]
s.split(" ")        # ["", "", "analisis", "de", "ventas", "", ""]
" ".join(["a", "b", "c"])  # "a b c"
s.startswith("ana")
s.endswith("s")
s.find("ventas")    # índice de inicio
s.count("a")

# Formateo
nombre = "Laptop"
precio = 899.99
f"{nombre}: ${precio:.2f}"       # f-string
"{:>10}".format("hola")          # alinear derecha
"{:<10}".format("hola")          # alinear izquierda
"{:^10}".format("hola")          # centrar
"{:.2%}".format(0.1666)          # "16.67%"
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*11. Strings.*

1. Manipulación
2. Formateo

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 12. Errores y Excepciones

```python
try:
    resultado = 10 / 0
except ZeroDivisionError:
    print("No se puede dividir entre cero")
except (TypeError, ValueError) as e:
    print(f"Error de tipo: {e}")
except Exception as e:
    print(f"Error inesperado: {e}")
else:
    print("Sin errores")
finally:
    print("Siempre se ejecuta")

# Lanzar excepciones
if monto < 0:
    raise ValueError("El monto no puede ser negativo")

# Assertions
assert len(ventas) > 0, "Lista de ventas vacía"
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*12. Errores y Excepciones.*

1. Lanzar excepciones
2. Assertions

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 13. Módulos y Paquetes

```python
# Importar módulos
import math
from math import sqrt, pi
from datetime import datetime as dt

# Módulo propio (ventas.py)
# from ventas import calcular_total

# Módulos útiles
import os               # sistema operativo
import sys              # argumentos de línea
import re               # expresiones regulares
import json             # JSON
import csv              # CSV
import random           # números aleatorios
import itertools        # iteradores avanzados
import collections      # defaultdict, Counter
import statistics       # mean, median, stdev
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*13. Módulos y Paquetes.*

1. Importar módulos
2. Módulo propio (ventas.py)
3. from ventas import calcular_total
4. Módulos útiles

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 14. itertools y collections

```python
from itertools import chain, cycle, product, combinations
from collections import Counter, defaultdict, OrderedDict

# Counter
ventas = ["ene", "feb", "ene", "mar", "ene", "feb"]
Counter(ventas)                  # Counter({'ene': 3, 'feb': 2, 'mar': 1})
Counter(ventas).most_common(2)   # [('ene', 3), ('feb', 2)]

# defaultdict
por_mes = defaultdict(list)
por_mes["enero"].append(100)
por_mes["enero"].append(200)

# Itertools
list(chain([1,2], [3,4]))        # [1,2,3,4]
list(product([1,2], [3,4]))      # [(1,3),(1,4),(2,3),(2,4)]
list(combinations([1,2,3], 2))   # [(1,2),(1,3),(2,3)]
```

**Salida:**

```
# La salida varía según los datos específicos cargados.
# Los resultados concretos dependen de los datos de entrada.
```

**Explicación línea por línea:**

*14. itertools y collections.*

1. Counter
2. defaultdict
3. Itertools

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## 15. Funciones Útiles de Built-in

```python
# enumerate, zip, reversed, sorted, any, all, map, filter
nombres = ["Ana", "Luis", "Pablo"]
edades = [28, 35, 22]

for i, nombre in enumerate(nombres, start=1):
    print(i, nombre)

for nombre, edad in zip(nombres, edades):
    print(f"{nombre}: {edad} años")

list(zip(nombres, edades))                          # lista de tuplas
dict(zip(nombres, edades))                          # {'Ana': 28, ...}

reversed([1, 2, 3])                                 # 3, 2, 1
sorted([3, 1, 2])                                   # [1, 2, 3]
sorted([3, 1, 2], reverse=True)                     # [3, 2, 1]
sorted(["manzana", "kiwi", "pera"], key=len)        # por longitud

any([False, True, False])                           # True
all([True, True, False])                            # False
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*15. Funciones Útiles de Built-in.*

1. enumerate, zip, reversed, sorted, any, all, map, filter

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios, compras o inventarios.*

---



## Referencia Rápida — Tabla de Operadores

| Operador | Uso | Descripción |
|----------|-----|-------------|
| `+` | a + b | Suma / concatenación |
| `-` | a - b | Resta |
| `*` | a * b | Multiplicación / repetición |
| `/` | a / b | División flotante |
| `//` | a // b | División entera |
| `%` | a % b | Módulo (residuo) |
| `**` | a ** b | Potencia |
| `==` | a == b | Igualdad |
| `!=` | a != b | Diferencia |
| `<`, `>`, `<=`, `>=` | a < b | Comparación |
| `is` | a is None | Identidad |
| `in` | x in lista | Pertenencia |
| `not` | not a | Negación |
| `and` / `or` | a and b | Lógicos |
| `:=` | (n := len(x)) | Operador walrus |
