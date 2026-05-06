# B03 — Diccionarios en Python

## 1. Introducción

Los **diccionarios** (`dict`) almacenan pares **clave → valor**. Son la estructura más importante para datos estructurados. En ventas, cada producto es un diccionario con claves como `"nombre"`, `"precio"`, `"stock"`.

| Característica      | Descripción                            |
|---------------------|----------------------------------------|
| Sintaxis            | `{"clave": "valor"}`                  |
| Claves              | Inmutables: str, int, tuple            |
| Valores             | Cualquier tipo (listas, dicts, etc.)   |
| Mutables            | Sí                                     |
| Orden               | Preservado desde Python 3.7+           |

Métodos principales: `get()`, `keys()`, `values()`, `items()`, `update()`, `pop()`.

---

## 2. Ejemplos prácticos

### Ejemplo 1: Producto como diccionario

```python
producto = {
    "sku": "LAP-001",
    "nombre": "Laptop Gamer",
    "precio": 1250000,
    "stock": 10,
    "categoria": "Computación"
}

print("Producto completo:", producto)
print("Nombre:", producto["nombre"])
print("Precio: $", producto["precio"])
print("Claves:", list(producto.keys()))
print("Valores:", list(producto.values()))
```

**Salida:**
```
Producto completo: {'sku': 'LAP-001', 'nombre': 'Laptop Gamer', 'precio': 1250000, 'stock': 10, 'categoria': 'Computación'}
Nombre: Laptop Gamer
Precio: $ 1250000
Claves: ['sku', 'nombre', 'precio', 'stock', 'categoria']
Valores: ['LAP-001', 'Laptop Gamer', 1250000, 10, 'Computación']
```

**Explicación:**
1. `producto = {...}` — Diccionario con 5 pares clave:valor.
2. `producto["nombre"]` — Accede al valor usando la clave entre corchetes.
3. `producto.keys()` — Devuelve vista de todas las claves.
4. `producto.values()` — Devuelve vista de todos los valores.

---

### Ejemplo 2: Catálogo como diccionario de diccionarios

```python
catalogo = {
    "LAP-001": {"nombre": "Laptop Gamer", "precio": 1250000, "stock": 10},
    "MOU-002": {"nombre": "Mouse RGB", "precio": 25000, "stock": 50},
    "TCL-003": {"nombre": "Teclado Mecánico", "precio": 45000, "stock": 30}
}

print("Catálogo completo:")
for sku, datos in catalogo.items():
    print(f"  {sku}: {datos['nombre']} - ${datos['precio']:,} (stock: {datos['stock']})")

print("\nPrecio del SKU LAP-001: $", catalogo["LAP-001"]["precio"])
```

**Salida:**
```
Catálogo completo:
  LAP-001: Laptop Gamer - $1,250,000 (stock: 10)
  MOU-002: Mouse RGB - $25,000 (stock: 50)
  TCL-003: Teclado Mecánico - $45,000 (stock: 30)

Precio del SKU LAP-001: $ 1250000
```

**Explicación:**
1. `catalogo` — Dict donde cada clave es un SKU y cada valor es otro dict con los detalles.
2. `catalogo.items()` — Itera sobre pares (clave, valor).
3. `catalogo["LAP-001"]["precio"]` — Acceso anidado: primero obtiene el dict del SKU, luego el precio.

---

### Ejemplo 3: Actualizar stock de un producto

```python
inventario = {"Laptop": 10, "Mouse": 50, "Teclado": 30}
print("Inventario inicial:", inventario)

# Actualizar stock existente
inventario["Mouse"] = 45
print("Después de vender 5 Mouse:", inventario)

# Agregar nuevo producto
inventario["Monitor"] = 15
print("Después de agregar Monitor:", inventario)

# Usar update para múltiples cambios
inventario.update({"Laptop": 8, "Teclado": 25})
print("Después de update:", inventario)
```

**Salida:**
```
Inventario inicial: {'Laptop': 10, 'Mouse': 50, 'Teclado': 30}
Después de vender 5 Mouse: {'Laptop': 10, 'Mouse': 45, 'Teclado': 30}
Después de agregar Monitor: {'Laptop': 10, 'Mouse': 45, 'Teclado': 30, 'Monitor': 15}
Después de update: {'Laptop': 8, 'Mouse': 45, 'Teclado': 25, 'Monitor': 15}
```

**Explicación:**
1. `inventario["Mouse"] = 45` — Asigna nuevo valor a clave existente.
2. `inventario["Monitor"] = 15` — Si la clave no existe, **la crea**.
3. `inventario.update({...})` — Actualiza múltiples claves a la vez.

---

### Ejemplo 4: Buscar producto por SKU con `get` (seguro)

```python
catalogo = {
    "LAP-001": "Laptop Gamer",
    "MOU-002": "Mouse RGB",
    "TCL-003": "Teclado Mecánico"
}

sku_buscar = "MOU-002"
producto = catalogo.get(sku_buscar)
print(f"SKU {sku_buscar}: {producto}")

# get con valor por defecto
sku_inexistente = "XXX-999"
resultado = catalogo.get(sku_inexistente, "Producto no encontrado")
print(f"SKU {sku_inexistente}: {resultado}")

# Sin get, esto lanza KeyError
# catalogo["XXX-999"]  # ERROR
```

**Salida:**
```
SKU MOU-002: Mouse RGB
SKU XXX-999: Producto no encontrado
```

**Explicación:**
1. `catalogo.get(sku_buscar)` — Retorna el valor si la clave existe, `None` si no.
2. `catalogo.get(sku_inexistente, "Producto no encontrado")` — Retorna el valor por defecto si la clave no existe.
3. `catalogo["XXX-999"]` — Lanzaría `KeyError` (comentado para no detener el programa).

---

### Ejemplo 5: Inventario completo con verificación de existencia

```python
inventario = {
    "Laptop": {"precio": 1250000, "stock": 10, "ubicacion": "A1"},
    "Mouse": {"precio": 25000, "stock": 50, "ubicacion": "B2"},
    "Teclado": {"precio": 45000, "stock": 0, "ubicacion": "B3"}
}

print("=== INVENTARIO COMPLETO ===")
for producto, info in inventario.items():
    estado = "CON STOCK" if info["stock"] > 0 else "AGOTADO"
    valor = info["precio"] * info["stock"]
    print(f"{producto:12s} | ${info['precio']:>7,} | Stock: {info['stock']:>2} | "
          f"Ubic: {info['ubicacion']} | {estado:9s} | Valor: ${valor:>9,}")
```

**Salida:**
```
=== INVENTARIO COMPLETO ===
Laptop       | $1,250,000 | Stock: 10 | Ubic: A1 | CON STOCK  | Valor: $12,500,000
Mouse        | $   25,000 | Stock: 50 | Ubic: B2 | CON STOCK  | Valor: $ 1,250,000
Teclado      | $   45,000 | Stock:  0 | Ubic: B3 | AGOTADO    | Valor: $         0
```

**Explicación:**
1. Valores anidados: `info["precio"]`, `info["stock"]`, `info["ubicacion"]`.
2. Operador ternario `if info["stock"] > 0 else "AGOTADO"`.
3. `info["precio"] * info["stock"]` — Valor total en inventario de ese producto.
4. Formato `:>9,` — Alineado a derecha (9 espacios), con separador de miles.

---

### Ejemplo 6: Iterar ventas del día con `items`

```python
ventas_dia = {
    "Laptop": 3,
    "Mouse": 15,
    "Teclado": 8,
    "Monitor": 2,
    "Webcam": 6
}

print("=== VENTAS DEL DÍA ===")
total_ingresos = 0
for producto, cantidad in ventas_dia.items():
    print(f"  {producto:12s}: {cantidad:>2} unidades")
    total_ingresos += cantidad

print(f"\nTotal unidades vendidas: {total_ingresos}")
```

**Salida:**
```
=== VENTAS DEL DÍA ===
  Laptop       :  3 unidades
  Mouse        : 15 unidades
  Teclado      :  8 unidades
  Monitor      :  2 unidades
  Webcam       :  6 unidades

Total unidades vendidas: 34
```

**Explicación:**
1. `ventas_dia.items()` — Retorna pares (producto, cantidad) para iterar.
2. Se desempaqueta directamente en el for: `producto, cantidad`.
3. Se acumula en `total_ingresos` sumando solo las cantidades.

---

### Ejemplo 7: Agrupar productos por categoría

```python
catalogo = [
    {"nombre": "Laptop", "categoria": "Computación", "precio": 1250000},
    {"nombre": "Mouse", "categoria": "Periféricos", "precio": 25000},
    {"nombre": "Monitor", "categoria": "Computación", "precio": 350000},
    {"nombre": "Teclado", "categoria": "Periféricos", "precio": 45000},
    {"nombre": "Webcam", "categoria": "Periféricos", "precio": 35000}
]

agrupado = {}
for item in catalogo:
    cat = item["categoria"]
    if cat not in agrupado:
        agrupado[cat] = []
    agrupado[cat].append(item["nombre"])

print("=== PRODUCTOS POR CATEGORÍA ===")
for categoria, productos in agrupado.items():
    print(f"{categoria}: {', '.join(productos)}")
```

**Salida:**
```
=== PRODUCTOS POR CATEGORÍA ===
Computación: Laptop, Monitor
Periféricos: Mouse, Teclado, Webcam
```

**Explicación:**
1. Se itera sobre la lista de productos.
2. `if cat not in agrupado:` — Si la categoría no existe como clave, se crea con lista vacía.
3. `agrupado[cat].append(...)` — Agrega el nombre del producto a la lista de su categoría.
4. `', '.join(productos)` — Une los nombres con coma.

---

### Ejemplo 8: Contar frecuencias de productos vendidos

```python
ventas = ["Mouse", "Teclado", "Mouse", "Monitor", "Mouse", "Teclado", "Webcam"]

frecuencias = {}
for producto in ventas:
    frecuencias[producto] = frecuencias.get(producto, 0) + 1

print("=== FRECUENCIA DE VENTAS ===")
for producto, count in sorted(frecuencias.items(), key=lambda x: x[1], reverse=True):
    print(f"  {producto:12s}: {count} {'⭐' * min(count, 5)}")
```

**Salida:**
```
=== FRECUENCIA DE VENTAS ===
  Mouse       : 3 ⭐⭐⭐
  Teclado     : 2 ⭐⭐
  Monitor     : 1 ⭐
  Webcam      : 1 ⭐
```

**Explicación:**
1. `frecuencias.get(producto, 0)` — Obtiene el valor actual o 0 si no existe.
2. `+ 1` — Incrementa el contador.
3. `sorted(frecuencias.items(), key=lambda x: x[1], reverse=True)` — Ordena por frecuencia descendente.
4. `min(count, 5)` — Limita a 5 estrellas.

---

### Ejemplo 9: Merge de dos catálogos (combinar diccionarios)

```python
catalogo_actual = {
    "LAP-001": {"nombre": "Laptop", "precio": 1250000, "stock": 10},
    "MOU-002": {"nombre": "Mouse", "precio": 25000, "stock": 50}
}

catalogo_nuevo = {
    "MOU-002": {"nombre": "Mouse RGB", "precio": 28000, "stock": 40},
    "TCL-003": {"nombre": "Teclado", "precio": 45000, "stock": 30}
}

# Merge: los valores nuevos sobrescriben a los existentes
catalogo_final = {**catalogo_actual, **catalogo_nuevo}
# Alternativa con update:
# catalogo_actual.update(catalogo_nuevo)

print("=== CATÁLOGO COMBINADO ===")
for sku, datos in catalogo_final.items():
    print(f"  {sku}: {datos['nombre']:12s} | ${datos['precio']:>7,} | Stock: {datos['stock']}")
```

**Salida:**
```
=== CATÁLOGO COMBINADO ===
  LAP-001: Laptop       | $1,250,000 | Stock: 10
  MOU-002: Mouse RGB    | $   28,000 | Stock: 40
  TCL-003: Teclado      | $   45,000 | Stock: 30
```

**Explicación:**
1. `{**catalogo_actual, **catalogo_nuevo}` — Unpacking de diccionarios (Python 3.5+). Las claves duplicadas toman el valor del último dict.
2. Alternativamente `catalogo_actual.update(catalogo_nuevo)` modifica in-place.
3. El SKU `MOU-002` fue sobrescrito con los datos nuevos.

---

### Ejemplo 10: Productos agotados (stock = 0)

```python
inventario = {
    "Laptop": {"stock": 10, "precio": 1250000},
    "Mouse": {"stock": 0, "precio": 25000},
    "Teclado": {"stock": 30, "precio": 45000},
    "Monitor": {"stock": 0, "precio": 350000},
    "Webcam": {"stock": 5, "precio": 35000}
}

agotados = []
for producto, info in inventario.items():
    if info["stock"] == 0:
        agotados.append(producto)

print("Productos agotados:", agotados)

# List comprehension más elegante
agotados_v2 = [p for p, i in inventario.items() if i["stock"] == 0]
print("(con list comp):", agotados_v2)
```

**Salida:**
```
Productos agotados: ['Mouse', 'Monitor']
(con list comp): ['Mouse', 'Monitor']
```

**Explicación:**
1. Bucle tradicional: itera con `.items()`, verifica `info["stock"] == 0`, agrega a lista.
2. List comprehension: más compacta, genera la lista directamente con `[p for p, i in ... if ...]`.

---

### Ejemplo 11: Valor total del inventario

```python
inventario = {
    "Laptop": {"precio": 1250000, "stock": 10},
    "Mouse": {"precio": 25000, "stock": 50},
    "Teclado": {"precio": 45000, "stock": 30},
    "Monitor": {"precio": 350000, "stock": 15}
}

valor_total = sum(info["precio"] * info["stock"] for info in inventario.values())
print("Valor total del inventario: ${:,}".format(valor_total))

# Desglose por producto
print("\n=== DESGLOSE ===")
for prod, info in inventario.items():
    valor = info["precio"] * info["stock"]
    print(f"  {prod:12s}: ${info['precio']:>7,} × {info['stock']:>2} = ${valor:>10,}")
```

**Salida:**
```
Valor total del inventario: $20,350,000

=== DESGLOSE ===
  Laptop       : $1,250,000 × 10 = $ 12,500,000
  Mouse        : $   25,000 × 50 = $  1,250,000
  Teclado      : $   45,000 × 30 = $  1,350,000
  Monitor      : $  350,000 × 15 = $  5,250,000
```

**Explicación:**
1. `sum(info["precio"] * info["stock"] for info in inventario.values())` — Generator expression dentro de `sum()` para calcular total.
2. `"${:,}".format(valor_total)` — Formato con separador de miles.
3. Desglose: itera cada producto y muestra precio × stock = valor.

---

### Ejemplo 12: Ventas por mes (diccionario de listas)

```python
ventas_por_mes = {
    "Enero": [120, 145, 98, 150],
    "Febrero": [85, 90, 76, 110],
    "Marzo": [60, 55, 70, 80]
}

print("=== VENTAS POR MES ===")
for mes, ventas in ventas_por_mes.items():
    total = sum(ventas)
    promedio = total / len(ventas)
    print(f"{mes:10s}: Total={total:>3}, Promedio={promedio:.0f}, "
          f"Mín={min(ventas)}, Máx={max(ventas)}")

# Agregar ventas de Abril
ventas_por_mes["Abril"] = [95, 110, 88, 102]
print(f"\nDespués de agregar Abril:")
print(f"  Meses registrados: {list(ventas_por_mes.keys())}")
```

**Salida:**
```
=== VENTAS POR MES ===
Enero     : Total=513, Promedio=128, Mín=98, Máx=150
Febrero   : Total=361, Promedio=90, Mín=76, Máx=110
Marzo     : Total=265, Promedio=66, Mín=55, Máx=80

Después de agregar Abril:
  Meses registrados: ['Enero', 'Febrero', 'Marzo', 'Abril']
```

**Explicación:**
1. `ventas_por_mes` — Dict donde cada clave es un string (mes) y cada valor es una lista de enteros.
2. `sum(ventas)`, `len(ventas)`, `min(ventas)`, `max(ventas)` — Funciones sobre la lista.
3. `ventas_por_mes["Abril"] = [...]` — Agrega nuevo mes asignando una lista.

---

### Ejemplo 13: Diccionario anidado en 3 niveles (sucursal → mes → producto)

```python
ventas = {
    "Santiago": {
        "Enero": {"Laptop": 5, "Mouse": 20, "Teclado": 12},
        "Febrero": {"Laptop": 7, "Mouse": 18, "Teclado": 15}
    },
    "Valparaíso": {
        "Enero": {"Laptop": 2, "Mouse": 10, "Teclado": 8}
    }
}

print("=== VENTAS 3 NIVELES ===")
for sucursal, meses in ventas.items():
    print(f"\n{sucursal}:")
    for mes, productos in meses.items():
        print(f"  {mes}:")
        for prod, cant in productos.items():
            print(f"    {prod:12s}: {cant}")
```

**Salida:**
```
=== VENTAS 3 NIVELES ===

Santiago:
  Enero:
    Laptop       : 5
    Mouse        : 20
    Teclado      : 12
  Febrero:
    Laptop       : 7
    Mouse        : 18
    Teclado      : 15

Valparaíso:
  Enero:
    Laptop       : 2
    Mouse        : 10
    Teclado      : 8
```

**Explicación:**
1. 3 niveles: sucursal → mes → producto → cantidad.
2. Iteración triple: `for sucursal, meses in ventas.items()`, luego `for mes, productos in meses.items()`, luego `for prod, cant in productos.items()`.
3. Útil para reportes jerárquicos.

---

### Ejemplo 14: `defaultdict` para inventario inicial

```python
from collections import defaultdict

# default dict: si la clave no existe, la crea con valor por defecto (int → 0)
inventario = defaultdict(int)

# Simular entrada de mercadería
entradas = ["Laptop", "Mouse", "Laptop", "Teclado", "Mouse", "Laptop"]
for producto in entradas:
    inventario[producto] += 1

print("=== INVENTARIO CON DEFAULTDICT ===")
for producto, cantidad in sorted(inventario.items()):
    print(f"  {producto}: {cantidad}")

# Acceder a clave inexistente devuelve 0, no KeyError
print(f"\nMonitor: {inventario['Monitor']}")
```

**Salida:**
```
=== INVENTARIO CON DEFAULTDICT ===
  Laptop: 3
  Mouse: 2
  Teclado: 1

Monitor: 0
```

**Explicación:**
1. `defaultdict(int)` — Crea un dict donde el valor por defecto para claves nuevas es `int()` = 0.
2. `inventario[producto] += 1` — Funciona aunque el producto no exista aún (se crea con 0 y se incrementa).
3. `inventario['Monitor']` — Devuelve 0 porque no se agregó, en lugar de lanzar `KeyError`.

---

### Ejemplo 15: `OrderedDict` para preservar orden de inserción (legado)

```python
from collections import OrderedDict

# En Python 3.7+ los dicts normales ya preservan orden.
# OrderedDict es útil para compatibilidad y métodos extra.

pedido = OrderedDict()
pedido["Laptop"] = 2
pedido["Mouse"] = 5
pedido["Teclado"] = 3
pedido["Monitor"] = 1

print("=== PEDIDO (orden preservado) ===")
for producto, cantidad in pedido.items():
    print(f"  {producto:12s}: {cantidad}")

# Mover al final
pedido.move_to_end("Mouse")
print(f"\nDespués de move_to_end('Mouse'): {list(pedido.keys())}")

# Mover al inicio (last=False)
pedido.move_to_end("Teclado", last=False)
print(f"Después de move_to_end('Teclado', last=False): {list(pedido.keys())}")
```

**Salida:**
```
=== PEDIDO (orden preservado) ===
  Laptop       : 2
  Mouse        : 5
  Teclado      : 3
  Monitor      : 1

Después de move_to_end('Mouse'): ['Laptop', 'Teclado', 'Monitor', 'Mouse']
Después de move_to_end('Teclado', last=False): ['Teclado', 'Laptop', 'Monitor', 'Mouse']
```

**Explicación:**
1. `OrderedDict` — Preserva orden de inserción (dict normal también desde Python 3.7).
2. `move_to_end(clave)` — Mueve la clave al final.
3. `move_to_end(clave, last=False)` — Mueve la clave al inicio.
4. Útil para implementar LRU cache o reordenar items.

---

### Ejemplo 16: Ordenar diccionario por valor (precios)

```python
productos = {
    "Laptop": 1250000,
    "Mouse": 25000,
    "Monitor": 350000,
    "Teclado": 45000,
    "Webcam": 35000
}

# Ordenar por precio ascendente
ordenado_asc = dict(sorted(productos.items(), key=lambda item: item[1]))
print("Ordenado por precio (asc):")
for prod, precio in ordenado_asc.items():
    print(f"  {prod:12s}: ${precio:>7,}")

# Ordenar por precio descendente
ordenado_desc = dict(sorted(productos.items(), key=lambda item: item[1], reverse=True))
print("\nOrdenado por precio (desc):")
for prod, precio in ordenado_desc.items():
    print(f"  {prod:12s}: ${precio:>7,}")
```

**Salida:**
```
Ordenado por precio (asc):
  Mouse        : $  25,000
  Webcam       : $  35,000
  Teclado      : $  45,000
  Monitor      : $ 350,000
  Laptop       : $1,250,000

Ordenado por precio (desc):
  Laptop       : $1,250,000
  Monitor      : $ 350,000
  Teclado      : $  45,000
  Webcam       : $  35,000
  Mouse        : $  25,000
```

**Explicación:**
1. `productos.items()` — Tuplas `(clave, valor)`.
2. `sorted(..., key=lambda item: item[1])` — Ordena por el valor (índice 1 de cada tupla).
3. `dict(...)` — Convierte la lista ordenada de tuplas de vuelta a diccionario.
4. `reverse=True` — Orden descendente.

---

### Ejemplo 17: `dict` comprehension para aplicar descuento

```python
precios_originales = {
    "Laptop": 1250000,
    "Mouse": 25000,
    "Teclado": 45000,
    "Monitor": 350000
}

# Aplicar 10% de descuento a todos
precios_oferta = {prod: round(precio * 0.9, 0) for prod, precio in precios_originales.items()}

print("=== PRECIOS ORIGINALES vs OFERTA ===")
for prod in precios_originales:
    print(f"  {prod:12s}: ${precios_originales[prod]:>7,} → ${int(precios_oferta[prod]):>7,}")

# Filtrar productos cuyo precio con descuento sea < $100,000
baratos = {prod: precio for prod, precio in precios_oferta.items() if precio < 100000}
print(f"\nProductos en oferta < $100,000: {list(baratos.keys())}")
```

**Salida:**
```
=== PRECIOS ORIGINALES vs OFERTA ===
  Laptop       : $1,250,000 → $ 1,125,000
  Mouse        : $   25,000 → $    22,500
  Teclado      : $   45,000 → $    40,500
  Monitor      : $  350,000 → $   315,000

Productos en oferta < $100,000: ['Mouse', 'Teclado']
```

**Explicación:**
1. Dict comprehension: `{prod: round(precio * 0.9, 0) for prod, precio in ...}`.
2. Filtro adicional: `if precio < 100000` solo incluye ciertos productos.
3. `int(precio)` — Casting para presentación sin decimales.

---

### Ejemplo 18: `pop` para eliminar y obtener valores

```python
inventario = {
    "Laptop": {"stock": 10, "precio": 1250000},
    "Mouse": {"stock": 50, "precio": 25000},
    "Teclado": {"stock": 30, "precio": 45000},
    "Agotado": {"stock": 0, "precio": 10000}
}

print("Inventario inicial:", len(inventario), "productos")

# pop: elimina clave y devuelve su valor
producto_eliminado = inventario.pop("Agotado")
print(f"Se eliminó 'Agotado': {producto_eliminado}")

# pop con valor por defecto (no lanza KeyError)
inexistente = inventario.pop("NoExiste", None)
print(f"pop de 'NoExiste': {inexistente}")

print(f"Inventario después de pop: {len(inventario)} productos")
print(f"Claves restantes: {list(inventario.keys())}")
```

**Salida:**
```
Inventario inicial: 4 productos
Se eliminó 'Agotado': {'stock': 0, 'precio': 10000}
pop de 'NoExiste': None
Inventario después de pop: 3 productos
Claves restantes: ['Laptop', 'Mouse', 'Teclado']
```

**Explicación:**
1. `inventario.pop("Agotado")` — Elimina la clave y retorna su valor (el dict anidado).
2. `inventario.pop("NoExiste", None)` — Si la clave no existe, retorna `None` en vez de lanzar `KeyError`.
3. Útil para mover o archivar registros.

---

### Ejemplo 19: `setdefault` para valores iniciales

```python
inventario = {}
productos = ["Laptop", "Mouse", "Teclado", "Mouse", "Laptop", "Laptop"]

for prod in productos:
    # Si la clave no existe, la crea con 0
    inventario.setdefault(prod, 0)
    inventario[prod] += 1

print("=== INVENTARIO con setdefault ===")
for prod, cant in sorted(inventario.items()):
    print(f"  {prod}: {cant}")

# Alternativa más corta con get (vista antes)
# inventario[prod] = inventario.get(prod, 0) + 1
```

**Salida:**
```
=== INVENTARIO con setdefault ===
  Laptop: 3
  Mouse: 2
  Teclado: 1
```

**Explicación:**
1. `inventario.setdefault(prod, 0)` — Si `prod` no existe como clave, la crea con valor 0. Si existe, no hace nada.
2. Luego `inventario[prod] += 1` incrementa.
3. Es equivalente a `inventario[prod] = inventario.get(prod, 0) + 1`.

---

### Ejemplo 20: Diccionario con valores por defecto anidados (`defaultdict` de `defaultdict`)

```python
from collections import defaultdict

# defaultdict anidado para inventario por sucursal y producto
inventario = defaultdict(lambda: defaultdict(int))

# Agregar entradas
inventario["Santiago"]["Laptop"] += 10
inventario["Santiago"]["Mouse"] += 50
inventario["Valparaíso"]["Laptop"] += 5
inventario["Valparaíso"]["Teclado"] += 30
inventario["Santiago"]["Laptop"] += 3  # Incrementa

print("=== INVENTARIO POR SUCURSAL ===")
for sucursal, productos in sorted(inventario.items()):
    print(f"\n{sucursal}:")
    for prod, cant in sorted(productos.items()):
        print(f"  {prod}: {cant}")
```

**Salida:**
```
=== INVENTARIO POR SUCURSAL ===

Santiago:
  Laptop: 13
  Mouse: 50

Valparaíso:
  Laptop: 5
  Teclado: 30
```

**Explicación:**
1. `defaultdict(lambda: defaultdict(int))` — Crea un dict de 2 niveles. El nivel exterior mapea sucursal → dict interior. El interior mapea producto → entero.
2. `inventario["Santiago"]["Laptop"] += 10` — Si "Santiago" no existe, se crea con un `defaultdict(int)`. Luego si "Laptop" no existe, se crea con 0 y se incrementa.
3. Evita chequeos manuales de existencia.

---

## 3. Ejercicios propuestos

1. **Catálogo desde listas:** Dadas `skus = ["A01", "B02", "C03"]` y `nombres = ["Laptop", "Mouse", "Teclado"]`, crea un diccionario `{sku: nombre}` usando `zip` y dict comprehension.

2. **Actualizar stock:** Dado `inventario = {"Laptop": 10, "Mouse": 50}` y una venta `{"Laptop": 2, "Mouse": 5}`, actualiza el inventario restando las cantidades vendidas. Asegúrate de que no queden negativos.

3. **Productos bajo stock mínimo:** Dado `inventario = {"Laptop": 3, "Mouse": 8, "Teclado": 1}` y `stock_minimo = 5`, crea una lista de productos que necesitan reorden (stock < mínimo).

4. **Frecuencia de palabras:** Dada una lista `categorias = ["Electro", "Ropa", "Electro", "Libro", "Ropa", "Electro"]`, usa un diccionario para contar cuántas veces aparece cada categoría.

5. **Merge con descuento:** Combina dos diccionarios de precios `precios_a = {"Laptop": 1250000, "Mouse": 25000}` y `precios_b = {"Mouse": 22000, "Teclado": 45000}`. El merge debe aplicar el precio más bajo cuando hay duplicados.

6. **Valor total por categoría:** Dado un catálogo anidado (categoría → productos → precio), calcula el valor total de cada categoría.

7. **Orden inverso:** Dado `ventas = {"Enero": 150, "Febrero": 200, "Marzo": 120}`, ordena el diccionario por valor de mayor a menor.

8. **Anidación de 3 niveles:** Crea un diccionario que represente: año → mes → día → lista de ventas. Agrega datos para al menos 2 días.

---

## 4. Resumen

- Los diccionarios almacenan pares `clave → valor` y son la estructura principal para datos de ventas.
- `get(clave, default)` evita `KeyError`.
- `keys()`, `values()`, `items()` iteran sobre componentes del dict.
- `update()` fusiona diccionarios; `pop()` elimina claves.
- Dict comprehensions: `{k: v for k, v in dict.items() if cond}`.
- `defaultdict` de `collections` evita inicializaciones manuales.
- `OrderedDict` preserva orden (aunque los dicts normales también desde Python 3.7).
- Los diccionarios pueden anidarse para representar jerarquías (sucursal → mes → producto).
- `sorted(dict.items(), key=lambda x: x[1])` ordena por valor.
- Los diccionarios son la base para el procesamiento de datos JSON en APIs.
