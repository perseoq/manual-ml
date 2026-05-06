# B05 — Funciones en Python

## 1. Introducción

Las funciones encapsulan lógica reutilizable. Se definen con `def` y pueden recibir argumentos, devolver valores con `return`, y aceptar número variable de parámetros con `*args` y `**kwargs`.

| Concepto       | Sintaxis                         | Uso en ventas                      |
|----------------|----------------------------------|-------------------------------------|
| `def`          | `def calc_total(p, c):`          | Calcular total de compra            |
| `return`       | `return p * c`                   | Devolver resultado del cálculo      |
| `*args`        | `def sumar(*precios):`           | Sumar lista variable de precios     |
| `**kwargs`     | `def filtrar(**conds):`          | Filtros dinámicos por categoría     |
| `lambda`       | `lambda x: x * 1.19`             | Funciones rápidas en `map`/`filter` |
| `map`          | `map(lambda p: p*1.19, precios)` | Aplicar IVA a toda una lista        |
| `filter`       | `filter(lambda p: p>0, stock)`   | Filtrar productos con stock         |
| `reduce`       | `reduce(mul, cantidades)`        | Producto de cantidades              |

---

## 2. Ejemplos prácticos

### Ejemplo 1: Función para calcular total (precio × cantidad)

```python
def calcular_total(precio: float, cantidad: int) -> float:
    """Calcula el total de una venta: precio * cantidad."""
    return precio * cantidad

# Uso
total = calcular_total(12500, 3)
print(f"Total: ${total:,}")

total_2 = calcular_total(45000, 5)
print(f"Total 2: ${total_2:,}")
```

**Salida:**
```
Total: $37,500
Total 2: $225,000
```

**Explicación:**
1. `def calcular_total(precio: float, cantidad: int) -> float:` — Define la función con anotaciones de tipo (documentativas, no restrictivas).
2. `"""..."""` — Docstring: describe qué hace la función.
3. `return precio * cantidad` — Devuelve el resultado de la multiplicación.
4. Se llama dos veces con diferentes argumentos (reutilización).

---

### Ejemplo 2: Función para aplicar impuesto (IVA)

```python
def aplicar_iva(precio: float, tasa: float = 0.19) -> float:
    """Aplica un porcentaje de IVA al precio. Tasa por defecto: 19%."""
    return round(precio * (1 + tasa), 2)

# Usos
print(f"Precio $10,000 + IVA 19%: ${aplicar_iva(10000):,}")
print(f"Precio $50,000 + IVA 10%: ${aplicar_iva(50000, 0.10):,}")
print(f"Precio $25,000 + IVA 5%:  ${aplicar_iva(25000, 0.05):,}")
```

**Salida:**
```
Precio $10,000 + IVA 19%: $11,900.0
Precio $50,000 + IVA 10%: $55,000.0
Precio $25,000 + IVA 5%:  $26,250.0
```

**Explicación:**
1. `tasa: float = 0.19` — Parámetro con valor por defecto (opcional al llamar).
2. `round(precio * (1 + tasa), 2)` — Calcula precio con IVA y redondea a 2 decimales.
3. Se puede pasar tasa diferente para IVA reducido o países con otras tasas.

---

### Ejemplo 3: Descuento por volumen

```python
def descuento_volumen(cantidad: int, precio_unitario: float) -> float:
    """Aplica descuento escalonado según cantidad."""
    subtotal = cantidad * precio_unitario
    if cantidad >= 50:
        desc = 0.20
    elif cantidad >= 20:
        desc = 0.15
    elif cantidad >= 10:
        desc = 0.10
    else:
        desc = 0.0
    return round(subtotal * (1 - desc), 2)

print(f"5 unidades a $12,000: ${descuento_volumen(5, 12000):,}")
print(f"15 unidades a $12,000: ${descuento_volumen(15, 12000):,}")
print(f"25 unidades a $12,000: ${descuento_volumen(25, 12000):,}")
print(f"50 unidades a $12,000: ${descuento_volumen(50, 12000):,}")
```

**Salida:**
```
5 unidades a $12,000: $60,000.0
15 unidades a $12,000: $162,000.0
25 unidades a $12,000: $255,000.0
50 unidades a $12,000: $480,000.0
```

**Explicación:**
1. Escalas de descuento: 0% (< 10), 10% (10-19), 15% (20-49), 20% (50+).
2. `subtotal * (1 - desc)` — Aplica descuento como multiplicador.
3. `round(subtotal * (1 - desc), 2)` — Redondeo para precisión financiera.

---

### Ejemplo 4: Margen de ganancia

```python
def margen_ganancia(costo: float, venta: float) -> dict:
    """Calcula margen bruto y porcentaje."""
    ganancia = venta - costo
    margen_pct = (ganancia / venta) * 100
    return {
        "costo": costo,
        "venta": venta,
        "ganancia": round(ganancia, 2),
        "margen_pct": round(margen_pct, 1)
    }

resultado = margen_ganancia(8000, 15000)
print(f"Costo: ${resultado['costo']:,}")
print(f"Venta: ${resultado['venta']:,}")
print(f"Ganancia: ${resultado['ganancia']:,}")
print(f"Margen: {resultado['margen_pct']}%")
```

**Salida:**
```
Costo: $8,000.0
Venta: $15,000.0
Ganancia: $7,000.0
Margen: 46.7%
```

**Explicación:**
1. La función retorna un diccionario con múltiples valores calculados.
2. `(venta - costo) / venta * 100` — Fórmula de margen sobre precio de venta.
3. Retornar dict permite acceder por nombre a cada resultado.

---

### Ejemplo 5: Rotación de stock

```python
def rotacion_stock(ventas_periodo: int, stock_promedio: float) -> float:
    """Calcula la rotación de inventario (índice)."""
    if stock_promedio == 0:
        return 0.0
    return round(ventas_periodo / stock_promedio, 2)

print(f"Rotación (100 ventas, 50 stock): {rotacion_stock(100, 50)}")
print(f"Rotación (30 ventas, 45 stock): {rotacion_stock(30, 45)}")
print(f"Rotación (0 stock promedio): {rotacion_stock(20, 0)}")
```

**Salida:**
```
Rotación (100 ventas, 50 stock): 2.0
Rotación (30 ventas, 45 stock): 0.67
Rotación (0 stock promedio): 0.0
```

**Explicación:**
1. `ventas_periodo / stock_promedio` — Mide cuántas veces se renueva el stock.
2. `if stock_promedio == 0: return 0.0` — Guard clause para evitar división por cero.
3. `round(..., 2)` — Redondeo a 2 decimales.

---

### Ejemplo 6: Valor total del inventario

```python
def valor_inventario(inventario: dict) -> dict:
    """Calcula valor total y por producto de un inventario."""
    total = 0
    desglose = {}
    for prod, datos in inventario.items():
        valor = datos["precio"] * datos["stock"]
        desglose[prod] = valor
        total += valor
    return {"total": total, "desglose": desglose}

inventario = {
    "Laptop": {"precio": 1250000, "stock": 10},
    "Mouse": {"precio": 25000, "stock": 50},
    "Teclado": {"precio": 45000, "stock": 30}
}

resultado = valor_inventario(inventario)
print(f"Valor total inventario: ${resultado['total']:,}")
print("Desglose:")
for prod, val in resultado["desglose"].items():
    print(f"  {prod}: ${val:,}")
```

**Salida:**
```
Valor total inventario: $14,000,000
Desglose:
  Laptop: $12,500,000
  Mouse: $1,250,000
  Teclado: $1,350,000
```

**Explicación:**
1. Función que recibe un diccionario estructurado y retorna un resumen.
2. `inventario.items()` — Itera sobre pares (producto, datos).
3. Retorna diccionario con `total` y `desglose` para acceso flexible.

---

### Ejemplo 7: Clasificar producto según precio

```python
def clasificar_producto(precio: float) -> str:
    """Clasifica un producto según su precio."""
    if precio < 10000:
        return "Económico"
    elif precio < 50000:
        return "Estándar"
    elif precio < 200000:
        return "Gama Media"
    elif precio < 1000000:
        return "Premium"
    else:
        return "Ultra Premium"

for precio in [5000, 25000, 75000, 350000, 2500000]:
    print(f"${precio:>7,} → {clasificar_producto(precio)}")
```

**Salida:**
```
$  5,000 → Económico
$ 25,000 → Estándar
$ 75,000 → Gama Media
$350,000 → Premium
$2,500,000 → Ultra Premium
```

**Explicación:**
1. Función pura: misma entrada → misma salida, sin efectos secundarios.
2. Múltiples `elif` para rangos de precio.
3. `return` termina la función inmediatamente (no necesita `else` final).

---

### Ejemplo 8: Recomendar reorden de stock

```python
def recomendar_reorden(stock: int, stock_minimo: int, punto_pedido: int = 10) -> str:
    """Recomienda si reordenar o no según stock actual."""
    if stock <= 0:
        return "URGENTE: Producto agotado. Reordenar inmediatamente."
    elif stock <= stock_minimo:
        return f"ALERTA: Stock bajo ({stock}). Pedir {punto_pedido} unidades."
    elif stock <= stock_minimo + punto_pedido:
        return f"PRECAUCIÓN: Stock {stock}. Considere reordenar pronto."
    else:
        return f"OK: Stock suficiente ({stock} unidades)."

print(recomendar_reorden(0, 5))
print(recomendar_reorden(3, 5))
print(recomendar_reorden(12, 5))
print(recomendar_reorden(30, 5))
```

**Salida:**
```
URGENTE: Producto agotado. Reordenar inmediatamente.
ALERTA: Stock bajo (3). Pedir 10 unidades.
PRECAUCIÓN: Stock 12. Considere reordenar pronto.
OK: Stock suficiente (30 unidades).
```

**Explicación:**
1. `stock_minimo` y `punto_pedido` (por defecto 10) parametrizan la lógica.
2. Múltiples niveles de alerta con condicionales.
3. Útil para sistemas de gestión de inventario automatizados.

---

### Ejemplo 9: `*args` para sumar precios variables

```python
def suma_precios(*args: float) -> float:
    """Suma una cantidad variable de precios."""
    print(f"Recibidos {len(args)} precios: {args}")
    return sum(args)

total = suma_precios(12000, 34000, 8500, 22000, 15000)
print(f"Total: ${total:,}")

total_2 = suma_precios(50000, 25000)
print(f"Total 2: ${total_2:,}")
```

**Salida:**
```
Recibidos 5 precios: (12000, 34000, 8500, 22000, 15000)
Total: $91,500
Recibidos 2 precios: (50000, 25000)
Total 2: $75,000
```

**Explicación:**
1. `*args` — Empaqueta argumentos posicionales en una tupla.
2. `sum(args)` — Suma todos los elementos de la tupla.
3. Permite llamar con cualquier cantidad de argumentos.

---

### Ejemplo 10: `**kwargs` para filtros dinámicos

```python
def filtrar_productos(**kwargs) -> None:
    """Muestra productos que coinciden con los filtros dados."""
    catalogo = [
        {"nombre": "Laptop", "precio": 1250000, "stock": 10, "categoria": "Computación"},
        {"nombre": "Mouse", "precio": 25000, "stock": 0, "categoria": "Periféricos"},
        {"nombre": "Teclado", "precio": 45000, "stock": 30, "categoria": "Periféricos"},
        {"nombre": "Monitor", "precio": 350000, "stock": 15, "categoria": "Computación"}
    ]

    print(f"Filtros aplicados: {kwargs}")
    for prod in catalogo:
        coincide = all(prod.get(k) == v for k, v in kwargs.items())
        if coincide:
            print(f"  → {prod['nombre']:12s} | ${prod['precio']:>7,} | Stock: {prod['stock']}")

filtrar_productos(categoria="Periféricos")
print()
filtrar_productos(categoria="Computación", stock=10)
```

**Salida:**
```
Filtros aplicados: {'categoria': 'Periféricos'}
  → Mouse        | $  25,000 | Stock: 0
  → Teclado      | $  45,000 | Stock: 30

Filtros aplicados: {'categoria': 'Computación', 'stock': 10}
  → Laptop       | $1,250,000 | Stock: 10
```

**Explicación:**
1. `**kwargs` — Diccionario de argumentos con nombre: `clave=valor`.
2. `prod.get(k) == v` — Compara cada filtro con el valor del producto.
3. `all(...)` — Todos los filtros deben coincidir.

---

### Ejemplo 11: `lambda` para ordenar productos por precio

```python
productos = [
    ("Laptop", 1250000, 10),
    ("Mouse", 25000, 50),
    ("Monitor", 350000, 15),
    ("Teclado", 45000, 30)
]

# Ordenar por precio usando lambda
productos.sort(key=lambda prod: prod[1])
print("Productos ordenados por precio:")
for p in productos:
    print(f"  {p[0]:12s}: ${p[1]:>7,} | Stock: {p[2]}")

# Ordenar por stock descendente
print("\nProductos por stock (desc):")
for p in sorted(productos, key=lambda x: x[2], reverse=True):
    print(f"  {p[0]:12s}: Stock: {p[2]}")
```

**Salida:**
```
Productos ordenados por precio:
  Mouse        : $  25,000 | Stock: 50
  Teclado      : $  45,000 | Stock: 30
  Monitor      : $ 350,000 | Stock: 15
  Laptop       : $1,250,000 | Stock: 10

Productos por stock (desc):
  Mouse        : Stock: 50
  Teclado      : Stock: 30
  Monitor      : Stock: 15
  Laptop       : Stock: 10
```

**Explicación:**
1. `lambda prod: prod[1]` — Función anónima que retorna el segundo elemento.
2. Usada como `key` en `sort()` y `sorted()`.
3. `lambda x: x[2], reverse=True` — Orden descendente por stock.

---

### Ejemplo 12: `map` para aplicar IVA a una lista de precios

```python
precios = [25000, 45000, 120000, 34000, 89000]

# map aplica una función a cada elemento
precios_con_iva = list(map(lambda p: round(p * 1.19, 0), precios))

print("Precios originales:", precios)
print("Precios con IVA:   ", [int(p) for p in precios_con_iva])

# También con función definida
def con_iva(p):
    return round(p * 1.19, 0)

precios_iva_v2 = list(map(con_iva, precios))
print("(con función def):", [int(p) for p in precios_iva_v2])
```

**Salida:**
```
Precios originales: [25000, 45000, 120000, 34000, 89000]
Precios con IVA:    [29750, 53550, 142800, 40460, 105910]
(con función def):  [29750, 53550, 142800, 40460, 105910]
```

**Explicación:**
1. `map(lambda p: round(p * 1.19, 0), precios)` — Aplica la lambda a cada elemento.
2. `list(...)` — Convierte el iterador de `map` a lista.
3. `map` también acepta funciones definidas con `def`.

---

### Ejemplo 13: `filter` para productos en stock

```python
productos = [
    {"nombre": "Laptop", "stock": 10},
    {"nombre": "Mouse", "stock": 0},
    {"nombre": "Teclado", "stock": 30},
    {"nombre": "Monitor", "stock": 0}
]

en_stock = list(filter(lambda p: p["stock"] > 0, productos))
print("Productos con stock:")
for p in en_stock:
    print(f"  {p['nombre']}: {p['stock']} unidades")

# Equivalente con list comprehension
en_stock_v2 = [p for p in productos if p["stock"] > 0]
print(f"\nTotal con stock: {len(en_stock_v2)}")
```

**Salida:**
```
Productos con stock:
  Laptop: 10 unidades
  Teclado: 30 unidades

Total con stock: 2
```

**Explicación:**
1. `filter(lambda p: p["stock"] > 0, productos)` — Filtra solo los elementos que cumplen la condición.
2. La lambda debe retornar `True`/`False`.
3. `list(...)` — Materializa el iterador filtrado.

---

### Ejemplo 14: `reduce` para calcular total del carrito

```python
from functools import reduce

precios_carrito = [1250000, 25000, 45000, 35000]

# reduce acumula: (((a + b) + c) + d)
total = reduce(lambda acc, p: acc + p, precios_carrito)
print(f"Total del carrito: ${total:,}")

# Con valor inicial
total_con_envio = reduce(lambda acc, p: acc + p, precios_carrito, 0) + 5000
print(f"Total con envío: ${total_con_envio:,}")
```

**Salida:**
```
Total del carrito: $1,355,000
Total con envío: $1,360,000
```

**Explicación:**
1. `reduce(función, iterable)` — Aplica la función acumulativamente.
2. `lambda acc, p: acc + p` — `acc` es el acumulador, `p` es cada precio.
3. El tercer argumento (0) es el valor inicial del acumulador.

---

### Ejemplo 15: Función generadora de SKU

```python
def generar_sku(categoria: str, numero: int, anio: int = 2025) -> str:
    """Genera un SKU con formato: CAT-NNNN-YYYY."""
    cat_part = categoria[:3].upper()
    num_part = f"{numero:04d}"
    return f"{cat_part}-{num_part}-{anio}"

print(generar_sku("Laptop", 1))
print(generar_sku("Mouse", 15))
print(generar_sku("Teclado", 123, 2024))
print(generar_sku("Monitor", 9999))
```

**Salida:**
```
LAP-0001-2025
MOU-0015-2025
TEC-0123-2024
MON-9999-2025
```

**Explicación:**
1. `categoria[:3].upper()` — Toma las primeras 3 letras en mayúsculas.
2. `f"{numero:04d}"` — Formatea el número con 4 dígitos y ceros a la izquierda.
3. `anio: int = 2025` — Valor por defecto para el año.

---

### Ejemplo 16: Decorador para logging de ventas

```python
def log_ventas(func):
    """Decorador que registea cada llamada a una función de venta."""
    def wrapper(*args, **kwargs):
        print(f"[LOG] Llamando a {func.__name__} con args={args}, kwargs={kwargs}")
        resultado = func(*args, **kwargs)
        print(f"[LOG] Resultado: {resultado}")
        return resultado
    return wrapper

@log_ventas
def vender(producto: str, cantidad: int, precio: float) -> float:
    """Procesa una venta y retorna el total."""
    total = cantidad * precio
    print(f"  Ventas: {cantidad} × {producto} = ${total:,}")
    return total

# Uso
vender("Laptop", 2, 1250000)
print()
vender("Mouse", 5, 25000)
```

**Salida:**
```
[LOG] Llamando a vender con args=('Laptop', 2, 1250000), kwargs={}
  Ventas: 2 × Laptop = $2,500,000
[LOG] Resultado: 2500000

[LOG] Llamando a vender con args=('Mouse', 5, 25000), kwargs={}
  Ventas: 5 × Mouse = $125,000
[LOG] Resultado: 125000
```

**Explicación:**
1. Decorador: función que envuelve a otra para extender su comportamiento.
2. `def wrapper(*args, **kwargs):` — Acepta cualquier combinación de argumentos.
3. `@log_ventas` — Syntactic sugar que equivale a `vender = log_ventas(vender)`.
4. Útil para logging, medición de tiempo, control de acceso.

---

### Ejemplo 17: Funciones con `docstring` y anotaciones de tipo

```python
def calcular_descuento(
    precio_original: float,
    porcentaje: float,
    redondear: bool = True
) -> float:
    """
    Calcula el precio final después de aplicar un descuento.

    Args:
        precio_original: Precio antes del descuento.
        porcentaje: Porcentaje a descontar (ej. 15 para 15%).
        redondear: Si True, redondea a entero.

    Returns:
        Precio final con descuento aplicado.
    """
    precio_final = precio_original * (1 - porcentaje / 100)
    if redondear:
        precio_final = round(precio_final, 0)
    return precio_final

# Ayuda desde código
print(calcular_descuento.__doc__)
print(f"\nPrecio $50,000 - 20%: ${calcular_descuento(50000, 20):.0f}")
print(f"Precio $50,000 - 20% (sin redondear): ${calcular_descuento(50000, 20, False):.2f}")
```

**Salida:**
```
Calcula el precio final después de aplicar un descuento.

    Args:
        precio_original: Precio antes del descuento.
        porcentaje: Porcentaje a descontar (ej. 15 para 15%).
        redondear: Si True, redondea a entero.

    Returns:
        Precio final con descuento aplicado.
    

Precio $50,000 - 20%: $40000
Precio $50,000 - 20% (sin redondear): $40000.00
```

**Explicación:**
1. Anotaciones de tipo (`: float`, `-> float`) documentan los tipos esperados.
2. Docstring con formato Google/NumPy describe parámetros y retorno.
3. `func.__doc__` accede al docstring en tiempo de ejecución.
4. Mejora la legibilidad y permite generación automática de documentación.

---

### Ejemplo 18: Función recursiva para calcular descuento en cadena

```python
def descuento_escalonado(precio: float, nivel: int = 0) -> float:
    """Aplica descuentos escalonados recursivamente hasta 3 niveles."""
    if nivel >= 3 or precio <= 10000:
        return precio
    descuento = 0.10 if nivel < 2 else 0.05
    nuevo_precio = precio * (1 - descuento)
    print(f"  Nivel {nivel + 1}: 10% → ${nuevo_precio:,.0f}")
    return descuento_escalonado(nuevo_precio, nivel + 1)

print("Aplicando descuentos escalonados:")
resultado = descuento_escalonado(200000)
print(f"Precio final: ${resultado:,.0f}")
```

**Salida:**
```
Aplicando descuentos escalonados:
  Nivel 1: 10% → $180,000
  Nivel 2: 10% → $162,000
  Nivel 3: 5% → $153,900
Precio final: $153,900
```

**Explicación:**
1. Función recursiva: se llama a sí misma con argumentos modificados.
2. `if nivel >= 3 or precio <= 10000:` — Caso base: detiene la recursión.
3. Cada nivel aplica un descuento diferente.
4. La recursión permite lógica de descuentos en cadena de forma elegante.

---

### Ejemplo 19: `partial` para fijar parámetros

```python
from functools import partial

def calcular_precio(base: float, impuesto: float, descuento: float) -> float:
    """Calcula precio final: base + impuesto - descuento."""
    return base + (base * impuesto) - descuento

# Crear funciones especializadas con partial
precio_iva_19 = partial(calcular_precio, impuesto=0.19, descuento=0)
precio_iva_19_desc_10 = partial(calcular_precio, impuesto=0.19, descuento=10000)

print(f"Precio $50,000 + IVA 19%: ${precio_iva_19(50000):,.0f}")
print(f"Precio $80,000 + IVA 19%: ${precio_iva_19(80000):,.0f}")
print(f"Precio $80,000 + IVA 19% - $10k: ${precio_iva_19_desc_10(80000):,.0f}")
```

**Salida:**
```
Precio $50,000 + IVA 19%: $59,500
Precio $80,000 + IVA 19%: $95,200
Precio $80,000 + IVA 19% - $10k: $85,200
```

**Explicación:**
1. `partial(función, parámetro=valor)` — Crea una nueva función con algunos argumentos prefijados.
2. `precio_iva_19 = partial(calcular_precio, impuesto=0.19, descuento=0)` — Solo requiere `base`.
3. Útil para crear variantes de una función sin repetir código.

---

### Ejemplo 20: Función como ciudadano de primera clase

```python
def aplicar_operacion(precios: list, operacion) -> list:
    """Aplica una función operación a cada precio."""
    return [operacion(p) for p in precios]

def duplicar(p):
    return p * 2

def mitad(p):
    return p // 2

precios = [10000, 20000, 30000]

print(f"Original: {precios}")
print(f"Duplicar: {aplicar_operacion(precios, duplicar)}")
print(f"Mitad:   {aplicar_operacion(precios, mitad)}")
print(f"IVA 19%: {aplicar_operacion(precios, lambda p: int(p * 1.19))}")
```

**Salida:**
```
Original: [10000, 20000, 30000]
Duplicar: [20000, 40000, 60000]
Mitad:   [5000, 10000, 15000]
IVA 19%: [11900, 23800, 35700]
```

**Explicación:**
1. `operacion` es un parámetro que recibe una función.
2. `aplicar_operacion` llama a `operacion(p)` para cada elemento.
3. Se pasa `duplicar`, `mitad` (funciones con nombre) y `lambda p: int(p * 1.19)` (anónima).
4. Las funciones en Python son objetos de primera clase: se pueden pasar como argumentos.

---

## 3. Ejercicios propuestos

1. **Función de descuento:** Crea `aplicar_descuento(precio, porcentaje)` que retorne el precio con descuento. Si el porcentaje es > 50, lanza un error con `raise ValueError`.

2. **Función con *args:** Crea `total_compra(*precios, iva=0.19)` que sume todos los precios y aplique IVA al total.

3. **Map para margen:** Usa `map` con una lambda para calcular el margen de ganancia de `costos = [5000, 12000, 8000]` con precios de venta `ventas = [7500, 19500, 12500]`.

4. **Filter para stock crítico:** Dada una lista de diccionarios con stock, usa `filter` para obtener solo los productos con stock < 5.

5. **Reduce para precio promedio:** Usa `reduce` para calcular el precio promedio de `precios = [12000, 45000, 7800, 25000]`.

6. **Decorador de tiempo:** Crea un decorador `medir_tiempo` que imprima cuánto tardó una función en ejecutarse.

7. **Función generadora de IDs:** Crea `generar_id_venta(sucursal, numero)` que genere IDs como `SCL-00042`.

8. **Lambda con sort:** Ordena una lista de productos `[(nombre, precio, stock)]` primero por precio ascendente, y si hay empate, por stock descendente. Usa una lambda con tupla como key.

---

## 4. Resumen

- `def nombre(parametros):` define funciones reutilizables.
- `return` devuelve valores; sin `return`, la función retorna `None`.
- `*args` captura argumentos posicionales variables (tupla).
- `**kwargs` captura argumentos nominales variables (dict).
- `lambda args: expresión` crea funciones anónimas de una línea.
- `map(función, iterable)` aplica función a cada elemento.
- `filter(función, iterable)` filtra elementos que cumplen condición.
- `reduce(función, iterable)` acumula resultados secuencialmente.
- Los decoradores (`@decorador`) envuelven funciones para extenderlas.
- Las anotaciones de tipo y docstrings mejoran la documentación.
- `partial` fija parámetros, creando versiones especializadas.
- Las funciones son objetos de primera clase: se pasan como argumentos.
