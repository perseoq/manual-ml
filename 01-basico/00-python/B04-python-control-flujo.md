# B04 — Control de Flujo en Python

## 1. Introducción

El control de flujo determina **qué** código se ejecuta y **cuántas veces**. Python ofrece:

| Estructura       | Uso principal                      |
|------------------|------------------------------------|
| `if`/`elif`/`else`| Decisiones condicionales          |
| `for`            | Iterar sobre secuencias            |
| `while`          | Repetir mientras condición sea True|
| `break`          | Salir del bucle inmediatamente     |
| `continue`       | Saltar a la siguiente iteración    |
| `match`/`case`   | Pattern matching (Python 3.10+)    |

En ventas: clasificar precios, aplicar descuentos, recorrer catálogos, menús interactivos.

---

## 2. Ejemplos prácticos

### Ejemplo 1: Clasificar producto como caro o barato

```python
precio = 150000
if precio < 50000:
    print("Producto económico")
elif precio < 200000:
    print("Producto de gama media")
else:
    print("Producto premium o caro")
```

**Salida:**
```
Producto de gama media
```

**Explicación:**
1. `if precio < 50000:` — Primera condición: si es menor a $50,000.
2. `elif precio < 200000:` — Segunda condición: si es menor a $200,000 (pero ≥ $50,000).
3. `else:` — Caso por defecto: $200,000 o más.
4. Solo un bloque se ejecuta (el primero que sea `True`).

---

### Ejemplo 2: Determinar rango de precios con múltiples elif

```python
precio = 7800

if precio <= 5000:
    rango = "Muy barato"
elif precio <= 15000:
    rango = "Barato"
elif precio <= 50000:
    rango = "Moderado"
elif precio <= 200000:
    rango = "Caro"
else:
    rango = "Muy caro"

print(f"Precio: ${precio:,} → {rango}")
```

**Salida:**
```
Precio: $7,800 → Barato
```

**Explicación:**
1. Las condiciones se evalúan en orden. `precio = 7800` es ≤ 15000 pero no ≤ 5000.
2. Entra en el segundo `elif` porque `7800 <= 15000` es `True`.
3. Se asigna `"Barato"` a `rango`.
4. Las condiciones siguientes ya no se evalúan.

---

### Ejemplo 3: Aplicar descuento a todos los productos con `for`

```python
precios = [25000, 45000, 120000, 34000, 89000]
print("Precios originales:", precios)

for i in range(len(precios)):
    precios[i] = int(precios[i] * 0.9)  # 10% descuento

print("Precios con 10% desc.:", precios)
```

**Salida:**
```
Precios originales: [25000, 45000, 120000, 34000, 89000]
Precios con 10% desc.: [22500, 40500, 108000, 30600, 80100]
```

**Explicación:**
1. `for i in range(len(precios)):` — Itera sobre los índices 0, 1, 2, 3, 4.
2. `precios[i] = int(precios[i] * 0.9)` — Modifica cada elemento in place.
3. `range(5)` genera `range(0, 5)` que produce 0, 1, 2, 3, 4.

---

### Ejemplo 4: Reducir stock hasta 0 con `while`

```python
producto = "Mouse"
stock = 5
print(f"Stock inicial de {producto}: {stock}")

while stock > 0:
    stock -= 1
    print(f"Vendido. Stock restante: {stock}")

print(f"¡{producto} AGOTADO!")
```

**Salida:**
```
Stock inicial de Mouse: 5
Vendido. Stock restante: 4
Vendido. Stock restante: 3
Vendido. Stock restante: 2
Vendido. Stock restante: 1
Vendido. Stock restante: 0
¡Mouse AGOTADO!
```

**Explicación:**
1. `while stock > 0:` — El bloque se ejecuta mientras la condición sea `True`.
2. `stock -= 1` — Decrementa el stock en 1 cada iteración.
3. Cuando `stock` llega a 0, la condición `stock > 0` es `False` y el bucle termina.
4. Útil para simulaciones de ventas hasta agotar existencias.

---

### Ejemplo 5: `break` para encontrar el primer producto agotado

```python
inventario = [
    {"nombre": "Laptop", "stock": 10},
    {"nombre": "Mouse", "stock": 0},
    {"nombre": "Teclado", "stock": 30},
    {"nombre": "Monitor", "stock": 0}
]

print("Buscando primer producto agotado...")
for producto in inventario:
    if producto["stock"] == 0:
        print(f"  ¡{producto['nombre']} está AGOTADO! (deteniendo búsqueda)")
        break
    print(f"  {producto['nombre']}: OK (stock={producto['stock']})")
else:
    print("  Todos los productos tienen stock")
```

**Salida:**
```
Buscando primer producto agotado...
  Laptop: OK (stock=10)
  ¡Mouse está AGOTADO! (deteniendo búsqueda)
```

**Explicación:**
1. `for producto in inventario:` — Itera sobre cada dict en la lista.
2. `if producto["stock"] == 0:` — Detecta stock cero.
3. `break` — Sale del bucle inmediatamente, sin procesar el resto.
4. El bloque `else:` del `for` solo se ejecuta si el bucle **no** fue interrumpido por `break`.

---

### Ejemplo 6: `continue` para saltar productos sin stock

```python
inventario = [
    {"nombre": "Laptop", "stock": 10, "precio": 1250000},
    {"nombre": "Mouse", "stock": 0, "precio": 25000},
    {"nombre": "Teclado", "stock": 30, "precio": 45000},
    {"nombre": "Monitor", "stock": 0, "precio": 350000}
]

print("=== PRODUCTOS DISPONIBLES ===")
for producto in inventario:
    if producto["stock"] == 0:
        continue  # salta al siguiente sin imprimir
    print(f"{producto['nombre']:12s} | ${producto['precio']:>7,} | Stock: {producto['stock']}")
```

**Salida:**
```
=== PRODUCTOS DISPONIBLES ===
Laptop       | $1,250,000 | Stock: 10
Teclado      | $   45,000 | Stock: 30
```

**Explicación:**
1. `if producto["stock"] == 0: continue` — Si el stock es 0, `continue` salta al siguiente elemento del bucle.
2. Los productos con stock 0 son omitidos.
3. A diferencia de `break`, **no** termina el bucle, solo salta la iteración actual.

---

### Ejemplo 7: `enumerate` para numerar productos en catálogo

```python
catalogo = ["Laptop", "Mouse", "Teclado", "Monitor", "Webcam"]
print("=== CATÁLOGO ===")
for i, producto in enumerate(catalogo, start=1):
    print(f"{i}. {producto}")
```

**Salida:**
```
=== CATÁLOGO ===
1. Laptop
2. Mouse
3. Teclado
4. Monitor
5. Webcam
```

**Explicación:**
1. `enumerate(catalogo, start=1)` — Genera pares `(índice, elemento)` empezando en 1.
2. Desempaquetado: `i` recibe el índice, `producto` recibe el nombre.
3. Más legible que usar `range(len(lista))`.

---

### Ejemplo 8: `zip` para iterar productos y precios simultáneamente

```python
productos = ["Laptop", "Mouse", "Teclado"]
precios = [1250000, 25000, 45000]

print("=== PRODUCTOS Y PRECIOS ===")
for prod, precio in zip(productos, precios):
    print(f"  {prod:12s}: ${precio:>7,}")
```

**Salida:**
```
=== PRODUCTOS Y PRECIOS ===
  Laptop       : $1,250,000
  Mouse        : $   25,000
  Teclado      : $   45,000
```

**Explicación:**
1. `zip(productos, precios)` — Combina ambas listas en pares.
2. `for prod, precio in zip(...)` — Desempaqueta cada par.
3. Si las listas tienen diferente longitud, `zip` se detiene en la más corta.

---

### Ejemplo 9: `if` anidado para categoría + precio

```python
producto = "Laptop"
categoria = "Computación"
precio = 1250000

if categoria == "Computación":
    if precio > 1000000:
        print(f"{producto}: Computación PREMIUM ($ {precio:,})")
    else:
        print(f"{producto}: Computación estándar")
elif categoria == "Periféricos":
    if precio > 50000:
        print(f"{producto}: Periférico PREMIUM")
    else:
        print(f"{producto}: Periférico estándar")
else:
    print(f"{producto}: Otra categoría")
```

**Salida:**
```
Laptop: Computación PREMIUM ($ 1,250,000)
```

**Explicación:**
1. Primer nivel: `if categoria == "Computación"`, después `elif categoria == "Periféricos"`, después `else`.
2. Segundo nivel (anidado): dentro de cada categoría, otro `if/else` según precio.
3. Python usa indentación (4 espacios) para definir bloques anidados.

---

### Ejemplo 10: `match`/`case` para tipo de pago (Python 3.10+)

```python
tipo_pago = "credito"

match tipo_pago:
    case "efectivo":
        descuento = 5
        print(f"Pago en efectivo: {descuento}% de descuento")
    case "debito":
        descuento = 3
        print(f"Pago con débito: {descuento}% de descuento")
    case "credito":
        recargo = 2
        print(f"Pago con crédito: {recargo}% de recargo")
    case _:
        print("Tipo de pago no reconocido")
```

**Salida:**
```
Pago con crédito: 2% de recargo
```

**Explicación:**
1. `match tipo_pago:` — Evalúa el valor de `tipo_pago`.
2. `case "efectivo":` — Si el valor es exactamente "efectivo", ejecuta este bloque.
3. `case _:` — Patrón comodín (wildcard): captura cualquier otro valor.
4. `match`/`case` es más legible que múltiples `elif` para comparaciones exactas.

---

### Ejemplo 11: `match` con patrones compuestos (tuplas)

```python
venta = ("Laptop", 2, 1250000)  # (producto, cantidad, precio_unitario)

match venta:
    case (producto, 1, precio):
        print(f"Venta unitaria de {producto} a ${precio:,}")
    case (producto, cantidad, precio) if cantidad <= 5:
        print(f"Venta pequeña: {cantidad} × {producto} = ${cantidad * precio:,}")
    case (producto, cantidad, precio) if cantidad > 5:
        print(f"Venta MAYORISTA: {cantidad} × {producto} = ${cantidad * precio:,}")
    case _:
        print("Formato de venta no reconocido")
```

**Salida:**
```
Venta pequeña: 2 × Laptop = $2,500,000
```

**Explicación:**
1. `match venta:` — La variable `venta` es una tupla de 3 elementos.
2. `case (producto, 1, precio):` — Coincide si el segundo elemento (cantidad) es exactamente 1.
3. `case (producto, cantidad, precio) if cantidad <= 5:` — Guard condicional: además del patrón, se evalúa `if cantidad <= 5`.
4. El orden de los `case` importa: el primero que coincide se ejecuta.

---

### Ejemplo 12: `any` y `all` para verificar condiciones en listas

```python
inventario = [
    {"nombre": "Laptop", "stock": 10},
    {"nombre": "Mouse", "stock": 0},
    {"nombre": "Teclado", "stock": 30}
]

# all(): ¿todos los productos tienen stock?
todo_con_stock = all(p["stock"] > 0 for p in inventario)
print("¿Todos los productos tienen stock?", todo_con_stock)

# any(): ¿hay al menos un producto con stock?
algun_con_stock = any(p["stock"] > 0 for p in inventario)
print("¿Hay al menos uno con stock?", algun_con_stock)

# any(): ¿hay algún producto agotado?
algun_agotado = any(p["stock"] == 0 for p in inventario)
print("¿Hay algún agotado?", algun_agotado)
```

**Salida:**
```
¿Todos los productos tienen stock? False
¿Hay al menos uno con stock? True
¿Hay algún agotado? True
```

**Explicación:**
1. `all(p["stock"] > 0 for p in inventario)` — Retorna `True` solo si **todos** los stock son > 0.
2. `any(...)` — Retorna `True` si **al menos uno** cumple la condición.
3. Usan generator expressions para evaluar perezosamente.

---

### Ejemplo 13: `for`/`else` para buscar producto

```python
catalogo = ["Laptop", "Mouse", "Teclado", "Monitor"]
buscar = "Webcam"

for producto in catalogo:
    if producto == buscar:
        print(f"¡{buscar} encontrado en el catálogo!")
        break
else:
    print(f"{buscar} NO está en el catálogo")

print("Búsqueda finalizada")
```

**Salida:**
```
Webcam NO está en el catálogo
Búsqueda finalizada
```

**Explicación:**
1. El bucle `for` itera sobre todos los elementos.
2. `if producto == buscar: break` — Si encuentra el producto, imprime y sale del bucle.
3. El bloque `else:` del `for` se ejecuta solo si nunca se ejecutó `break`.
4. Como "Webcam" no está en la lista, se ejecuta el `else`.

---

### Ejemplo 14: `while` con `input` para menú interactivo

```python
print("=== MENÚ DE VENTAS ===")
opcion = ""
while opcion != "4":
    print("\n1. Ver productos")
    print("2. Agregar producto")
    print("3. Buscar producto")
    print("4. Salir")
    opcion = input("Seleccione opción: ")

    if opcion == "1":
        print("  → Mostrando productos...")
    elif opcion == "2":
        print("  → Agregar nuevo producto...")
    elif opcion == "3":
        print("  → Buscar producto...")
    elif opcion == "4":
        print("  → Saliendo del sistema...")
    else:
        print("  → Opción inválida, intente de nuevo")

print("¡Gracias por usar el sistema!")
```

**Salida:**
```
=== MENÚ DE VENTAS ===

1. Ver productos
2. Agregar producto
3. Buscar producto
4. Salir
Seleccione opción: 1
  → Mostrando productos...

1. Ver productos
...
```

**Explicación:**
1. `opcion = ""` — Inicializa la variable para que la condición `while opcion != "4"` sea `True`.
2. `while opcion != "4":` — El bucle se repite hasta que el usuario ingrese "4".
3. `input("Seleccione opción: ")` — Lee entrada del usuario (siempre string).
4. `if/elif/else` procesa la opción.

---

### Ejemplo 15: Comprensión condicional para filtrar y transformar

```python
precios = [25000, 45000, 120000, 34000, 89000, 5000]

# Filtrar precios >= $30,000 y aplicar 19% IVA
precios_filtrados = [int(p * 1.19) for p in precios if p >= 30000]

print("Precios originales:", precios)
print("Precios >= $30,000 con IVA:", precios_filtrados)

# Comprensión con if-else (ternario)
etiquetas = ["CARO" if p > 50000 else "BARATO" for p in precios]
print("Etiquetas:", etiquetas)
```

**Salida:**
```
Precios originales: [25000, 45000, 120000, 34000, 89000, 5000]
Precios >= $30,000 con IVA: [53550, 142800, 40460, 105910]
Etiquetas: ['BARATO', 'BARATO', 'CARO', 'BARATO', 'CARO', 'BARATO']
```

**Explicación:**
1. `[int(p * 1.19) for p in precios if p >= 30000]` — Filtro (`if`) antes de transformar.
2. `["CARO" if p > 50000 else "BARATO" for p in precios]` — Ternario dentro de la comprensión (sin filtro).
3. La primera forma filtra elementos; la segunda transforma cada elemento.

---

### Ejemplo 16: Filtro múltiple con condiciones compuestas

```python
productos = [
    {"nombre": "Laptop", "precio": 1250000, "stock": 10, "categoria": "Computación"},
    {"nombre": "Mouse", "precio": 25000, "stock": 0, "categoria": "Periféricos"},
    {"nombre": "Teclado", "precio": 45000, "stock": 30, "categoria": "Periféricos"},
    {"nombre": "Monitor", "precio": 350000, "stock": 15, "categoria": "Computación"},
    {"nombre": "Webcam", "precio": 35000, "stock": 0, "categoria": "Periféricos"}
]

print("=== PRODUCTOS FILTRADOS ===")
print("(Precio < $100,000 AND stock > 0) OR (Categoría = Computación)")

for p in productos:
    cond1 = p["precio"] < 100000 and p["stock"] > 0
    cond2 = p["categoria"] == "Computación"
    if cond1 or cond2:
        print(f"  {p['nombre']:12s} | ${p['precio']:>7,} | Stock: {p['stock']} | {p['categoria']}")
```

**Salida:**
```
=== PRODUCTOS FILTRADOS ===
(Precio < $100,000 AND stock > 0) OR (Categoría = Computación)
  Laptop       | $1,250,000 | Stock: 10 | Computación
  Teclado      | $   45,000 | Stock: 30 | Periféricos
  Monitor      | $  350,000 | Stock: 15 | Computación
```

**Explicación:**
1. `cond1 = p["precio"] < 100000 and p["stock"] > 0` — Productos económicos con stock.
2. `cond2 = p["categoria"] == "Computación"` — Todos los de computación.
3. `if cond1 or cond2` — Operador `or`: basta que una condición sea `True`.
4. El `and` tiene mayor precedencia que `or`, pero los paréntesis aclaran la intención.

---

### Ejemplo 17: `range` con `step` para inventario alternado

```python
productos = ["Laptop", "Mouse", "Teclado", "Monitor", "Webcam", "Hub USB"]
print("Productos en índices pares:", productos[::2])

# Usando range con step
print("\nIterando cada 2 productos:")
for i in range(0, len(productos), 2):
    print(f"  Índice {i}: {productos[i]}")

# range descendente
print("\nIterando en orden inverso:")
for i in range(len(productos) - 1, -1, -1):
    print(f"  Índice {i}: {productos[i]}")
```

**Salida:**
```
Productos en índices pares: ['Laptop', 'Teclado', 'Webcam']

Iterando cada 2 productos:
  Índice 0: Laptop
  Índice 2: Teclado
  Índice 4: Webcam

Iterando en orden inverso:
  Índice 5: Hub USB
  Índice 4: Webcam
  Índice 3: Monitor
  Índice 2: Teclado
  Índice 1: Mouse
  Índice 0: Laptop
```

**Explicación:**
1. `range(start, stop, step)` — `range(0, 6, 2)` produce 0, 2, 4.
2. `range(len(productos) - 1, -1, -1)` — 5, 4, 3, 2, 1, 0 (descendente).
3. Slicing `productos[::2]` logra lo mismo de forma más concisa.

---

### Ejemplo 18: Simulación de carrito con `for` anidado

```python
carrito = [
    {"producto": "Laptop", "cantidad": 2, "precio": 1250000},
    {"producto": "Mouse", "cantidad": 5, "precio": 25000},
    {"producto": "Teclado", "cantidad": 1, "precio": 45000}
]

print("=== DETALLE DEL CARRITO ===")
total_general = 0
for item in carrito:
    subtotal = item["cantidad"] * item["precio"]
    total_general += subtotal
    print(f"  {item['producto']:12s} × {item['cantidad']} = ${subtotal:>10,}")

print(f"\n  TOTAL: ${total_general:>10,}")

# Aplicar descuento por volumen
if total_general > 1000000:
    descuento = int(total_general * 0.05)
    total_final = total_general - descuento
    print(f"  Descuento 5%: -${descuento:>8,}")
    print(f"  TOTAL FINAL: ${total_final:>10,}")
```

**Salida:**
```
=== DETALLE DEL CARRITO ===
  Laptop       × 2 = $ 2,500,000
  Mouse        × 5 = $   125,000
  Teclado      × 1 = $    45,000

  TOTAL: $ 2,670,000
  Descuento 5%: -$ 133,500
  TOTAL FINAL: $ 2,536,500
```

**Explicación:**
1. `for item in carrito:` — Itera sobre cada producto en el carrito.
2. `subtotal = item["cantidad"] * item["precio"]` — Calcula subtotal por línea.
3. `total_general += subtotal` — Acumula el total.
4. `if total_general > 1000000:` — Aplica descuento condicional por volumen.

---

### Ejemplo 19: `while` con centinela para procesar ventas

```python
ventas_pendientes = [12000, 34000, 0, 25000, 0, 45000]
print("Procesando ventas...")

while ventas_pendientes:
    venta = ventas_pendientes.pop(0)
    if venta == 0:
        print("  Venta cancelada (monto = 0)")
        continue
    print(f"  Venta procesada: ${venta:,}")

print("Todas las ventas procesadas")
```

**Salida:**
```
Procesando ventas...
  Venta procesada: $12,000
  Venta procesada: $34,000
  Venta cancelada (monto = 0)
  Venta procesada: $25,000
  Venta cancelada (monto = 0)
  Venta procesada: $45,000
Todas las ventas procesadas
```

**Explicación:**
1. `while ventas_pendientes:` — Una lista vacía es `False`, con elementos es `True`.
2. `ventas_pendientes.pop(0)` — Extrae y elimina el primer elemento.
3. `if venta == 0: continue` — Si la venta es 0, la salta sin procesar.
4. El bucle termina cuando la lista está vacía.

---

### Ejemplo 20: `match`/`case` con pattern matching de diccionarios

```python
def procesar_venta(venta: dict):
    match venta:
        case {"tipo": "contado", "total": total}:
            print(f"Venta al contado: ${total:,}")
        case {"tipo": "credito", "total": total, "cuotas": cuotas}:
            print(f"Venta a crédito: ${total:,} en {cuotas} cuotas")
        case {"tipo": "empresa", "total": total, "rut": rut}:
            print(f"Venta empresarial (RUT {rut}): ${total:,}")
        case _:
            print("Tipo de venta no soportado")

procesar_venta({"tipo": "credito", "total": 250000, "cuotas": 6})
procesar_venta({"tipo": "contado", "total": 120000})
procesar_venta({"tipo": "desconocido"})
```

**Salida:**
```
Venta a crédito: $250,000 en 6 cuotas
Venta al contado: $120,000
Tipo de venta no soportado
```

**Explicación:**
1. `match venta:` sobre un diccionario.
2. `case {"tipo": "credito", "total": total, "cuotas": cuotas}:` — Coincide si el dict tiene las claves "tipo" (="credito"), "total" y "cuotas", y extrae los valores a variables.
3. `case _:` — Atrapa cualquier otro dict que no coincida con los patrones anteriores.
4. Pattern matching de dicts es muy potente para procesar datos de APIs.

---

## 3. Ejercicios propuestos

1. **Clasificación múltiple:** Dada una lista de precios `[12000, 45000, 7800, 95000, 250000]`, clasifica cada uno como "Económico" (< $20k), "Medio" ($20k-$80k) o "Premium" (> $80k). Usa `for` e `if/elif/else`.

2. **Búsqueda con break:** Dado `inventario = [{"prod": "A", "stock": 5}, {"prod": "B", "stock": 0}, {"prod": "C", "stock": 3}]`, encuentra el primer producto con stock 0 e imprime su nombre. Si todos tienen stock, imprime "Todo en stock".

3. **Filtro con continue:** De la lista `precios = [0, 15000, -500, 25000, 0, 45000]`, suma solo los precios positivos (mayores a 0). Usa `continue` para saltar los inválidos.

4. **Menú while:** Crea un menú con opciones 1: Ver stock, 2: Vender, 3: Salir. En la opción 2, pide el nombre del producto y reduce su stock en 1. Usa un diccionario `stock = {"Laptop": 5, "Mouse": 10}`.

5. **Comprensión con doble filtro:** Dada `productos = [{"nombre": "A", "precio": 5000, "stock": 10}, ...]`, crea una lista de nombres de productos cuyo precio sea < $50,000 y stock > 0.

6. **For anidado para matriz:** Dada `ventas_sucursales = [[100, 200], [150, 250], [80, 120]]` (3 sucursales, 2 días), calcula el total por sucursal y el total general.

7. **Match para descuento:** Usa `match`/`case` para aplicar: si `cliente = "normal"` → 0% descuento, `"premium"` → 10%, `"vip"` → 20%, `_` → "Cliente no válido".

8. **Simulación de caja:** Usa un `while True` con `break` para simular una caja registradora que acumula precios hasta que el usuario ingresa 0 (terminar) o negativo (ignorar con `continue`). Muestra el total al final.

---

## 4. Resumen

- `if`/`elif`/`else` evalúa condiciones en orden y ejecuta el primer bloque verdadero.
- `for` itera sobre secuencias (listas, dicts, strings, rangos).
- `while` repite mientras la condición sea `True`.
- `break` sale del bucle inmediatamente.
- `continue` salta a la siguiente iteración.
- `match`/`case` (Python 3.10+) ofrece pattern matching para comparaciones estructuradas.
- `any()` y `all()` evalúan condiciones sobre colecciones.
- `for`/`else` ejecuta el `else` solo si no hubo `break`.
- Las list comprehensions con `if` filtran y transforman datos en una línea.
- `range(start, stop, step)` controla iteraciones numéricas.
- Los condicionales anidados y los operadores lógicos (`and`, `or`, `not`) construyen lógica compleja.
