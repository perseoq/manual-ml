# B02 — Listas y Tuplas en Python

## 1. Introducción

Las **listas** (`list`) y **tuplas** (`tuple`) son secuencias ordenadas de elementos. La diferencia clave: las listas son **mutables** (se pueden modificar) y las tuplas son **inmutables** (no se pueden modificar después de creadas).

| Característica      | Lista                | Tupla                 |
|---------------------|----------------------|-----------------------|
| Sintaxis            | `[1, 2, 3]`          | `(1, 2, 3)`           |
| Mutable             | Sí                   | No                    |
| Métodos útiles      | `append`, `sort`     | `count`, `index`      |
| Uso típico          | Catálogos dinámicos  | SKU fijos, constantes |

**Indexación:** empieza en 0. `lista[0]` es el primer elemento.  
**Slicing:** `lista[inicio:fin:paso]` extrae subsecuencias.

---

## 2. Ejemplos prácticos

### Ejemplo 1: Catálogo de productos como lista de strings

```python
catalogo = ["Laptop", "Mouse", "Teclado", "Monitor", "Webcam"]
print("Catálogo completo:", catalogo)
print("Primer producto:", catalogo[0])
print("Último producto:", catalogo[-1])
print("Cantidad de productos:", len(catalogo))
```

**Salida:**
```
Catálogo completo: ['Laptop', 'Mouse', 'Teclado', 'Monitor', 'Webcam']
Primer producto: Laptop
Último producto: Webcam
Cantidad de productos: 5
```

**Explicación:**
1. `catalogo = [...]` — Crea una lista con 5 strings (nombres de productos).
2. `catalogo[0]` — Indexación: primer elemento (índice 0).
3. `catalogo[-1]` — Indexación negativa: último elemento.
4. `len(catalogo)` — Devuelve la longitud de la lista (5).

---

### Ejemplo 2: Lista de precios y operaciones básicas

```python
precios = [12500, 34000, 8900, 42000, 15500]
print("Precios:", precios)
print("Precio máximo:", max(precios))
print("Precio mínimo:", min(precios))
print("Suma total del catálogo:", sum(precios))
print("Precio promedio:", round(sum(precios) / len(precios), 0))
```

**Salida:**
```
Precios: [12500, 34000, 8900, 42000, 15500]
Precio máximo: 42000
Precio mínimo: 8900
Suma total del catálogo: 112900
Precio promedio: 22580.0
```

**Explicación:**
1. `precios = [12500, ...]` — Lista de enteros.
2. `max(precios)` — Función incorporada que devuelve el valor máximo de la lista.
3. `min(precios)` — Devuelve el valor mínimo.
4. `sum(precios)` — Suma todos los elementos.
5. `sum(precios) / len(precios)` — Promedio: suma dividida por cantidad.

---

### Ejemplo 3: Slicing del top 5 más vendidos

```python
top_ventas = ["Laptop Gamer", "Mouse RGB", "Teclado Mecánico", "Monitor 27\"", "Webcam HD",
              "Audífonos BT", "Hub USB", "SSD 1TB", "RAM 16GB", "Mousepad"]
print("Top 10:", top_ventas)
print("Top 3:", top_ventas[:3])
print("Posiciones 4 a 6:", top_ventas[3:6])
print("Cada 2 productos (saltos):", top_ventas[::2])
print("Top 10 invertido:", top_ventas[::-1])
```

**Salida:**
```
Top 10: ['Laptop Gamer', 'Mouse RGB', 'Teclado Mecánico', 'Monitor 27"', 'Webcam HD', 'Audífonos BT', 'Hub USB', 'SSD 1TB', 'RAM 16GB', 'Mousepad']
Top 3: ['Laptop Gamer', 'Mouse RGB', 'Teclado Mecánico']
Posiciones 4 a 6: ['Monitor 27"', 'Webcam HD', 'Audífonos BT']
Cada 2 productos (saltos): ['Laptop Gamer', 'Teclado Mecánico', 'Webcam HD', 'Hub USB', 'RAM 16GB']
Top 10 invertido: ['Mousepad', 'RAM 16GB', 'SSD 1TB', 'Hub USB', 'Audífonos BT', 'Webcam HD', 'Monitor 27"', 'Teclado Mecánico', 'Mouse RGB', 'Laptop Gamer']
```

**Explicación:**
1. `top_ventas[:3]` — Slicing desde inicio hasta índice 3 (exclusivo): primeros 3 elementos.
2. `top_ventas[3:6]` — Desde índice 3 hasta 6 (exclusivo): elementos en posiciones 4, 5, 6.
3. `top_ventas[::2]` — Paso 2: toma cada 2 elementos.
4. `top_ventas[::-1]` — Paso negativo: invierte toda la lista.

---

### Ejemplo 4: Agregar producto al catálogo con `append`

```python
catalogo = ["Laptop", "Mouse"]
print("Antes:", catalogo)

catalogo.append("Teclado")
print("Después de append:", catalogo)

nuevos = ["Monitor", "Webcam"]
catalogo.extend(nuevos)
print("Después de extend:", catalogo)

catalogo.insert(1, "Hub USB")
print("Después de insert en 1:", catalogo)
```

**Salida:**
```
Antes: ['Laptop', 'Mouse']
Después de append: ['Laptop', 'Mouse', 'Teclado']
Después de extend: ['Laptop', 'Mouse', 'Teclado', 'Monitor', 'Webcam']
Después de insert en 1: ['Laptop', 'Hub USB', 'Mouse', 'Teclado', 'Monitor', 'Webcam']
```

**Explicación:**
1. `catalogo.append("Teclado")` — `append` agrega un elemento al final de la lista (mutable).
2. `catalogo.extend(nuevos)` — `extend` agrega todos los elementos de otra lista al final.
3. `catalogo.insert(1, "Hub USB")` — Inserta en la posición 1, desplazando el resto a la derecha.

---

### Ejemplo 5: Buscar producto por índice

```python
catalogo = ["Laptop", "Mouse", "Teclado", "Monitor", "Webcam"]
producto_buscar = "Monitor"

if producto_buscar in catalogo:
    idx = catalogo.index(producto_buscar)
    print(f"{producto_buscar} encontrado en índice {idx}")
else:
    print(f"{producto_buscar} NO está en el catálogo")

# También podemos buscar con find simulado
pos = catalogo.index("Teclado") if "Teclado" in catalogo else -1
print("Índice de Teclado:", pos)
```

**Salida:**
```
Monitor encontrado en índice 3
Índice de Teclado: 2
```

**Explicación:**
1. `producto_buscar in catalogo` — Operador `in`: verifica si el elemento existe en la lista, devuelve `True`/`False`.
2. `catalogo.index(producto_buscar)` — Devuelve el índice de la primera ocurrencia del elemento.
3. Expresión ternaria: devuelve el índice o `-1` si no existe.

---

### Ejemplo 6: Más vendidos con `sort` y `reverse`

```python
ventas_diarias = [12, 45, 8, 33, 27, 19, 41]
print("Ventas diarias (original):", ventas_diarias)

ventas_copia = ventas_diarias.copy()
ventas_copia.sort()
print("Ventas ordenadas ascendente:", ventas_copia)

ventas_copia.sort(reverse=True)
print("Ventas ordenadas descendente:", ventas_copia)

ventas_diarias.reverse()
print("Lista reversada (no ordenada):", ventas_diarias)
```

**Salida:**
```
Ventas diarias (original): [12, 45, 8, 33, 27, 19, 41]
Ventas ordenadas ascendente: [8, 12, 19, 27, 33, 41, 45]
Ventas ordenadas descendente: [45, 41, 33, 27, 19, 12, 8]
Lista reversada (no ordenada): [41, 19, 27, 33, 8, 45, 12]
```

**Explicación:**
1. `ventas_diarias.copy()` — Crea una copia superficial para no modificar la original.
2. `sort()` — Ordena la lista **in place** (modifica la lista original) de forma ascendente.
3. `sort(reverse=True)` — Ordena descendente.
4. `reverse()` — Invierte el orden actual sin ordenar.

---

### Ejemplo 7: Tupla SKU-nombre-precio (inmutable)

```python
producto = ("LAP-001", "Laptop Gamer", 1250000)
print("SKU:", producto[0])
print("Nombre:", producto[1])
print("Precio: $", producto[2])
print("Tupla completa:", producto)

# Intentar modificar la tupla causaría TypeError
# producto[0] = "NUEVO-SKU"  # ¡Error!
```

**Salida:**
```
SKU: LAP-001
Nombre: Laptop Gamer
Precio: $ 1250000
Tupla completa: ('LAP-001', 'Laptop Gamer', 1250000)
```

**Explicación:**
1. `producto = ("LAP-001", "Laptop Gamer", 1250000)` — Tupla de 3 elementos: str, str, int.
2. `producto[0]`, `producto[1]`, `producto[2]` — Acceso por índice como en listas.
3. Las tuplas son inmutables: no se puede reasignar `producto[0]` después de creadas.
4. Útiles para registros que no deben cambiar (ej. SKU maestro).

---

### Ejemplo 8: Desempaquetar tupla en variables

```python
producto = ("TCL-042", "Teclado Mecánico RGB", 45000)
sku, nombre, precio = producto

print("SKU:", sku)
print("Nombre:", nombre)
print("Precio: $", precio)

# Desempaquetar con * para el resto
venta = ("VENTA-001", "Carlos", "Laptop", 2, 1250000)
id_venta, cliente, *resto = venta
print("\nID:", id_venta, "| Cliente:", cliente, "| Resto:", resto)
```

**Salida:**
```
SKU: TCL-042
Nombre: Teclado Mecánico RGB
Precio: $ 45000

ID: VENTA-001 | Cliente: Carlos | Resto: ['Laptop', 2, 1250000]
```

**Explicación:**
1. `sku, nombre, precio = producto` — Desempaqueta cada elemento de la tupla en variables independientes.
2. `*resto` — Sintaxis de desempaquetado extendido: captura 0 o más elementos restantes en una lista.
3. Muy útil al iterar sobre listas de tuplas.

---

### Ejemplo 9: Unir catálogos (concatenar listas)

```python
computacion = ["Laptop", "Monitor", "Teclado"]
perifericos = ["Mouse", "Webcam", "Audífonos"]

catalogo_completo = computacion + perifericos
print("Concatenado con +:", catalogo_completo)

computacion.extend(perifericos)
print("Después de extend:", computacion)
```

**Salida:**
```
Concatenado con +: ['Laptop', 'Monitor', 'Teclado', 'Mouse', 'Webcam', 'Audífonos']
Después de extend: ['Laptop', 'Monitor', 'Teclado', 'Mouse', 'Webcam', 'Audífonos']
```

**Explicación:**
1. `computacion + perifericos` — Crea una **nueva** lista con los elementos de ambas.
2. `computacion.extend(perifericos)` — Modifica `computacion` **in place** agregando los elementos de `perifericos`.
3. En `+` se puede concatenar cualquier número de listas.

---

### Ejemplo 10: Ordenar lista de tuplas por precio con `sort` y `key`

```python
productos = [
    ("Laptop", 1250000, 10),
    ("Mouse", 25000, 50),
    ("Monitor", 350000, 15),
    ("Teclado", 45000, 30)
]

# Ordenar por precio (índice 1)
productos.sort(key=lambda prod: prod[1])
print("Ordenado por precio (menor a mayor):")
for p in productos:
    print(f"  {p[0]:12s} -> ${p[1]:>7,} | Stock: {p[2]}")

# Ordenar por stock descendente
productos.sort(key=lambda p: p[2], reverse=True)
print("\nOrdenado por stock (mayor a menor):")
for p in productos:
    print(f"  {p[0]:12s} -> Stock: {p[2]}")
```

**Salida:**
```
Ordenado por precio (menor a mayor):
  Mouse        -> $  25,000 | Stock: 50
  Teclado      -> $  45,000 | Stock: 30
  Monitor      -> $ 350,000 | Stock: 15
  Laptop       -> $1,250,000 | Stock: 10

Ordenado por stock (mayor a menor):
  Mouse        -> Stock: 50
  Teclado      -> Stock: 30
  Monitor      -> Stock: 15
  Laptop       -> Stock: 10
```

**Explicación:**
1. `productos` — Lista de tuplas (nombre, precio, stock).
2. `key=lambda prod: prod[1]` — Función lambda que extrae el precio (índice 1) como clave de ordenación.
3. `sort(reverse=True)` — Orden descendente.
4. La función `key` se aplica a cada elemento antes de comparar.

---

### Ejemplo 11: Invertir ranking de ventas

```python
ranking = ["Laptop", "Mouse", "Teclado", "Monitor", "Webcam"]
print("Ranking original:", ranking)

ranking.reverse()
print("Ranking invertido:", ranking)

# También se puede usar slicing para no modificar original
ranking_original = ["Laptop", "Mouse", "Teclado", "Monitor", "Webcam"]
invertido = ranking_original[::-1]
print("Invertido sin mutar:", invertido)
```

**Salida:**
```
Ranking original: ['Laptop', 'Mouse', 'Teclado', 'Monitor', 'Webcam']
Ranking invertido: ['Webcam', 'Monitor', 'Teclado', 'Mouse', 'Laptop']
Invertido sin mutar: ['Webcam', 'Monitor', 'Teclado', 'Mouse', 'Laptop']
```

**Explicación:**
1. `ranking.reverse()` — Invierte la lista in place.
2. `ranking_original[::-1]` — Crea una **nueva** lista invertida sin modificar la original.

---

### Ejemplo 12: Eliminar duplicados con `set` y volver a lista

```python
ventas_diarias = ["Mouse", "Teclado", "Mouse", "Monitor", "Teclado", "Mouse", "Webcam"]
print("Ventas (con duplicados):", ventas_diarias)

unicos = list(set(ventas_diarias))
print("Productos únicos:", unicos)

# Contar frecuencias con count
for producto in unicos:
    freq = ventas_diarias.count(producto)
    print(f"  {producto}: {freq} ventas")
```

**Salida:**
```
Ventas (con duplicados): ['Mouse', 'Teclado', 'Mouse', 'Monitor', 'Teclado', 'Mouse', 'Webcam']
Productos únicos: ['Webcam', 'Monitor', 'Mouse', 'Teclado']
  Webcam: 1 ventas
  Monitor: 1 ventas
  Mouse: 3 ventas
  Teclado: 2 ventas
```

**Explicación:**
1. `set(ventas_diarias)` — Convierte la lista a conjunto, eliminando duplicados (pierde orden).
2. `list(...)` — Vuelve a convertir a lista.
3. `ventas_diarias.count(producto)` — Cuenta cuántas veces aparece cada producto en la lista original.

---

### Ejemplo 13: Productos en oferta como tupla de tuplas

```python
ofertas = (
    ("Laptop", 1250000, 999000),
    ("Monitor", 350000, 299000),
    ("Teclado", 45000, 35000)
)

print("=== OFERTAS SEMANALES ===")
for producto, precio_normal, precio_oferta in ofertas:
    ahorro = precio_normal - precio_oferta
    desc_pct = (ahorro / precio_normal) * 100
    print(f"{producto:12s}: ${precio_normal:>7,} → ${precio_oferta:>7,} "
          f"(ahorra ${ahorro:>5,} | {desc_pct:.0f}% OFF)")
```

**Salida:**
```
=== OFERTAS SEMANALES ===
Laptop       : $1,250,000 → $  999,000 (ahorra $251,000 | 20% OFF)
Monitor      : $  350,000 → $  299,000 (ahorra $ 51,000 | 15% OFF)
Teclado      : $   45,000 → $   35,000 (ahorra $ 10,000 | 22% OFF)
```

**Explicación:**
1. `ofertas = (...)` — Tupla que contiene 3 tuplas anidadas (inmutable).
2. `for producto, precio_normal, precio_oferta in ofertas` — Itera desempaquetando cada tupla.
3. Calcula ahorro y porcentaje de descuento.
4. `:>7,` — Formato: alinear a derecha (7 espacios), con separador de miles.

---

### Ejemplo 14: Lista de listas (sucursales y sus ventas)

```python
ventas_sucursales = [
    ["Santiago", 120, 145, 98, 150],
    ["Valparaíso", 85, 90, 76, 110],
    ["Concepción", 60, 55, 70, 80]
]

print("Ventas por sucursal:")
for sucursal in ventas_sucursales:
    nombre = sucursal[0]
    total = sum(sucursal[1:])
    promedio = total / len(sucursal[1:])
    print(f"{nombre:12s}: Total={total:>3} | Promedio={promedio:.0f}")
```

**Salida:**
```
Ventas por sucursal:
Santiago    : Total=513 | Promedio=128
Valparaíso  : Total=361 | Promedio=90
Concepción  : Total=265 | Promedio=66
```

**Explicación:**
1. `ventas_sucursales` — Lista de listas: cada sublista tiene [sucursal, venta1, venta2, ...].
2. `sucursal[0]` — Primer elemento: nombre.
3. `sucursal[1:]` — Slicing: desde índice 1 hasta el final (solo números de ventas).
4. `sum(sucursal[1:])` — Suma todas las ventas de la sucursal.

---

### Ejemplo 15: Comprensión de listas con IVA

```python
precios_sin_iva = [25000, 45000, 120000, 34000, 89000]

# Aplicar 19% de IVA a cada precio
precios_con_iva = [round(p * 1.19, 0) for p in precios_sin_iva]

print("Precios sin IVA:", precios_sin_iva)
print("Precios con IVA:", [int(p) for p in precios_con_iva])

# Filtrar productos caros (>= $50,000 con IVA)
caros = [p for p in precios_con_iva if p >= 50000]
print("Productos >= $50,000:", [int(p) for p in caros])
```

**Salida:**
```
Precios sin IVA: [25000, 45000, 120000, 34000, 89000]
Precios con IVA: [29750, 53550, 142800, 40460, 105910]
Productos >= $50,000: [53550, 142800, 105910]
```

**Explicación:**
1. `[round(p * 1.19, 0) for p in precios_sin_iva]` — List comprehension: para cada `p` en `precios_sin_iva`, calcula `p * 1.19` y redondea.
2. `[p for p in precios_con_iva if p >= 50000]` — Filtra: solo incluye precios >= 50,000.
3. Las list comprehensions son más rápidas y legibles que los bucles manuales.

---

### Ejemplo 16: `enumerate` para numerar productos

```python
catalogo = ["Laptop", "Mouse", "Teclado", "Monitor"]

print("=== Catálogo numerado ===")
for idx, producto in enumerate(catalogo, start=1):
    print(f"{idx}. {producto}")

# También se puede crear lista de tuplas (índice, producto)
numerados = list(enumerate(catalogo, start=1))
print("\nEnumerate como lista:", numerados)
```

**Salida:**
```
=== Catálogo numerado ===
1. Laptop
2. Mouse
3. Teclado
4. Monitor

Enumerate como lista: [(1, 'Laptop'), (2, 'Mouse'), (3, 'Teclado'), (4, 'Monitor')]
```

**Explicación:**
1. `enumerate(catalogo, start=1)` — Genera tuplas `(índice, elemento)` empezando en 1.
2. Desempaquetado en el for: `idx` recibe el índice, `producto` recibe el elemento.
3. `list(enumerate(...))` — Convierte el iterador a lista de tuplas.

---

### Ejemplo 17: `zip` para combinar listas paralelas

```python
productos = ["Laptop", "Mouse", "Teclado"]
precios = [1250000, 25000, 45000]
stocks = [10, 50, 30]

print("=== Inventario combinado ===")
for prod, precio, stock in zip(productos, precios, stocks):
    valor = precio * stock
    print(f"{prod:12s} | ${precio:>7,} | Stock: {stock:>2} | Valor: ${valor:>8,}")

# zip devuelve tuplas
print("\nZip como lista:", list(zip(productos, precios, stocks)))
```

**Salida:**
```
=== Inventario combinado ===
Laptop       | $1,250,000 | Stock: 10 | Valor: $12,500,000
Mouse        | $   25,000 | Stock: 50 | Valor: $ 1,250,000
Teclado      | $   45,000 | Stock: 30 | Valor: $ 1,350,000

Zip como lista: [('Laptop', 1250000, 10), ('Mouse', 25000, 50), ('Teclado', 45000, 30)]
```

**Explicación:**
1. `zip(productos, precios, stocks)` — Combina las listas en paralelo, elemento a elemento.
2. En el for, cada iteración desempaqueta una tupla de 3 elementos.
3. `zip` se detiene cuando la lista más corta se agota.
4. Útil para procesar datos relacionados desde fuentes separadas.

---

### Ejemplo 18: `sorted` para orden sin mutar

```python
precios = [45000, 12000, 89000, 25000, 34000]
print("Original:", precios)

# sorted() devuelve NUEVA lista ordenada (no modifica la original)
ascendente = sorted(precios)
descendente = sorted(precios, reverse=True)

print("Ascendente:", ascendente)
print("Descendente:", descendente)
print("Original intacta:", precios)

# sorted con key para strings con longitud
productos = ["Laptop Gamer", "Mouse", "Teclado Mecánico", "Monitor"]
por_longitud = sorted(productos, key=len)
print("\nProductos ordenados por longitud:", por_longitud)
```

**Salida:**
```
Original: [45000, 12000, 89000, 25000, 34000]
Ascendente: [12000, 25000, 34000, 45000, 89000]
Descendente: [89000, 45000, 34000, 25000, 12000]
Original intacta: [45000, 12000, 89000, 25000, 34000]

Productos ordenados por longitud: ['Mouse', 'Monitor', 'Laptop Gamer', 'Teclado Mecánico']
```

**Explicación:**
1. `sorted(precios)` — Devuelve nueva lista ordenada. `sort()` es in-place, `sorted()` crea una copia.
2. `sorted(precios, reverse=True)` — Orden descendente.
3. `sorted(productos, key=len)` — Ordena usando la longitud del string como clave.
4. La lista original nunca se modifica.

---

### Ejemplo 19: Eliminar elementos con `pop` y `remove`

```python
inventario = ["Laptop", "Mouse", "Teclado", "Monitor", "Webcam"]
print("Inventario inicial:", inventario)

# Eliminar por índice
eliminado = inventario.pop(2)
print(f"pop(2) eliminó: {eliminado}")
print("Inventario después de pop:", inventario)

# Eliminar por valor
inventario.remove("Mouse")
print("remove('Mouse'):", inventario)

# pop() sin índice elimina el último
ultimo = inventario.pop()
print(f"pop() eliminó: {ultimo}")
print("Inventario final:", inventario)
```

**Salida:**
```
Inventario inicial: ['Laptop', 'Mouse', 'Teclado', 'Monitor', 'Webcam']
pop(2) eliminó: Teclado
Inventario después de pop: ['Laptop', 'Mouse', 'Monitor', 'Webcam']
remove('Mouse'): ['Laptop', 'Monitor', 'Webcam']
pop() eliminó: Webcam
Inventario final: ['Laptop', 'Monitor']
```

**Explicación:**
1. `pop(2)` — Elimina el elemento en índice 2 y lo **retorna**.
2. `remove("Mouse")` — Elimina la **primera ocurrencia** del valor "Mouse". No retorna el elemento.
3. `pop()` — Sin argumentos, elimina y retorna el **último** elemento.
4. `pop` también acepta índices negativos: `pop(-1)` elimina el último.

---

### Ejemplo 20: Matriz de inventario con listas anidadas

```python
inventario = [
    # [producto, precio, stock, categoria]
    ["Laptop", 1250000, 10, "Computación"],
    ["Mouse", 25000, 50, "Periféricos"],
    ["Teclado", 45000, 30, "Periféricos"],
    ["Monitor", 350000, 15, "Computación"]
]

print("=== INVENTARIO COMPLETO ===")
for item in inventario:
    prod, precio, stock, cat = item
    valor = precio * stock
    print(f"{prod:12s} | Cat: {cat:12s} | ${precio:>7,} | Stock: {stock:>2} | Valor: ${valor:>9,}")

# Calcular valor total del inventario
valor_total = sum(item[1] * item[2] for item in inventario)
print(f"\nValor total del inventario: ${valor_total:>10,}")

# Producto con mayor valor en stock
mas_valioso = max(inventario, key=lambda x: x[1] * x[2])
print(f"Producto más valioso: {mas_valioso[0]} (${mas_valioso[1] * mas_valioso[2]:,})")
```

**Salida:**
```
=== INVENTARIO COMPLETO ===
Laptop       | Cat: Computación  | $1,250,000 | Stock: 10 | Valor: $12,500,000
Mouse        | Cat: Periféricos  | $   25,000 | Stock: 50 | Valor: $ 1,250,000
Teclado      | Cat: Periféricos  | $   45,000 | Stock: 30 | Valor: $ 1,350,000
Monitor      | Cat: Computación  | $  350,000 | Stock: 15 | Valor: $ 5,250,000

Valor total del inventario: $ 20,350,000
Producto más valioso: Laptop ($12,500,000)
```

**Explicación:**
1. Matriz: lista de listas, cada sublista tiene 4 columnas.
2. `item[1] * item[2]` — Precio × Stock para valor individual.
3. `sum(item[1] * item[2] for item in inventario)` — Generator expression para sumar todos los valores.
4. `max(..., key=lambda x: x[1] * x[2])` — Encuentra el producto con mayor valor en inventario.

---

## 3. Ejercicios propuestos

1. **Top 5 ventas:** Dada una lista `ventas = [34, 56, 12, 89, 45, 67, 23, 91, 78, 44]`, obtén los 3 valores más altos usando `sorted` con `reverse=True` y slicing.

2. **Unir y ordenar:** Dos sucursales tienen listas de precios: `suc1 = [4500, 12000, 8000]` y `suc2 = [9500, 3000, 15000]`. Une ambas listas y ordénalas de menor a mayor.

3. **Comprensión con filtro:** Dada `precios = [5000, 15000, 25000, 35000, 45000]`, crea una nueva lista con los precios que tengan IVA incluido (× 1.19) solo si el resultado es menor a $40,000.

4. **Inventario como tuplas:** Crea una tupla de tuplas con 4 productos (SKU, nombre, precio, stock). Luego itera e imprime solo los productos con stock < 20.

5. **Zip para factura:** Dadas `productos = ["Pan", "Leche", "Huevos"]`, `cantidades = [2, 3, 1]`, `precios = [1500, 2200, 350]`, usa `zip` para calcular el subtotal de cada producto y el total.

6. **Matriz de sucursales:** Crea una lista de listas con 3 sucursales, cada una con 4 montos de ventas diarias. Calcula el total por sucursal usando una list comprehension.

7. **Enumerate con descuento:** Dada `compras = [15000, 25000, 8000, 42000]`, usa `enumerate` para aplicar un 10% de descuento a las compras en posiciones pares (índice 0, 2).

8. **Eliminar duplicados sin perder orden:** Dada `ventas = ["A", "B", "A", "C", "B", "D"]`, elimina duplicados preservando el orden de primera aparición (investiga cómo hacerlo con un bucle y `if`).

---

## 4. Resumen

- Las **listas** `[ ]` son mutables: `append`, `extend`, `insert`, `remove`, `pop`, `sort`, `reverse`.
- Las **tuplas** `( )` son inmutables: ideales para datos fijos como SKU.
- **Indexación:** `lista[0]` (primero), `lista[-1]` (último).
- **Slicing:** `lista[inicio:fin:paso]` para sublistas.
- `enumerate()` numera elementos al iterar.
- `zip()` combina múltiples listas en paralelo.
- List comprehensions: `[expr for x in lista if cond]`.
- `sorted()` devuelve nueva lista ordenada; `.sort()` ordena in-place.
- `max()`, `min()`, `sum()`, `len()`, `count()` son funciones útiles sobre listas.
- Las matrices (listas de listas) permiten representar tablas de datos.
