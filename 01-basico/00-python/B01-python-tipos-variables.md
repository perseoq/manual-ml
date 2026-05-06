# B01 — Tipos de Datos y Variables en Python

## 1. Introducción

Las variables en Python almacenan datos en memoria. Cada valor tiene un tipo asociado: `int` (enteros), `float` (decimales), `str` (cadenas) y `bool` (booleanos). Python es **dinámicamente tipado**: no necesitas declarar el tipo, se infiere automáticamente.

| Tipo   | Ejemplo           | Uso típico en ventas           |
|--------|-------------------|--------------------------------|
| `int`  | `cantidad = 5`    | Cantidad de productos          |
| `float`| `precio = 12000.0`| Precios, IVA, descuentos       |
| `str`  | `prod = "Laptop"` | Nombres, SKU, categorías       |
| `bool` | `en_stock = True` | Stock disponible, promociones   |

La función `type()` revela el tipo. El *casting* convierte entre tipos con `int()`, `float()`, `str()`, `bool()`.

---

## 2. Ejemplos prácticos

### Ejemplo 1: Calcular IVA (19%) de un producto

```python
# Variables de tipo float para precio e IVA
precio_sin_iva = 15000.50
iva = precio_sin_iva * 0.19
precio_con_iva = precio_sin_iva + iva

print("Precio sin IVA:", precio_sin_iva)
print("IVA (19%):", round(iva, 2))
print("Precio con IVA:", round(precio_con_iva, 2))
```

**Salida:**
```
Precio sin IVA: 15000.5
IVA (19%): 2850.09
Precio con IVA: 17850.59
```

**Explicación línea por línea:**
1. `precio_sin_iva = 15000.50` — Asigna un valor `float` con el precio base del producto.
2. `iva = precio_sin_iva * 0.19` — Calcula el 19% de IVA multiplicando el precio base por 0.19.
3. `precio_con_iva = precio_sin_iva + iva` — Suma el precio base más el IVA.
4. `print("Precio sin IVA:", precio_sin_iva)` — Muestra el precio original; `print` acepta múltiples argumentos separados por coma, los convierte a string automáticamente.
5. `print("IVA (19%):", round(iva, 2))` — `round(iva, 2)` redondea el IVA a 2 decimales para presentación.
6. `print("Precio con IVA:", round(precio_con_iva, 2))` — Ídem para el total con IVA.

---

### Ejemplo 2: Precio con descuento

```python
precio_original = 25000.00
descuento_pct = 15
precio_final = precio_original * (1 - descuento_pct / 100)

print("Precio original: $", precio_original)
print("Descuento:", descuento_pct, "%")
print("Precio final: $", round(precio_final, 2))
```

**Salida:**
```
Precio original: $ 25000.0
Descuento: 15 %
Precio final: $ 21250.0
```

**Explicación:**
1. `precio_original = 25000.00` — Precio de lista del producto como `float`.
2. `descuento_pct = 15` — Porcentaje de descuento como `int`.
3. `precio_final = precio_original * (1 - descuento_pct / 100)` — Primero divide 15/100 = 0.15, luego resta 1 - 0.15 = 0.85, luego multiplica 25000 * 0.85.
4. `print(...)` — Muestra resultados; `round(precio_final, 2)` asegura 2 decimales.

---

### Ejemplo 3: Redondeo del total de una compra

```python
subtotal = 47328.678
total_redondeado = round(subtotal, 0)
print("Subtotal exacto:", subtotal)
print("Total redondeado (entero):", int(total_redondeado))
```

**Salida:**
```
Subtotal exacto: 47328.678
Total redondeado (entero): 47329
```

**Explicación:**
1. `subtotal = 47328.678` — Float con 3 decimales simulando un subtotal.
2. `total_redondeado = round(subtotal, 0)` — `round(num, 0)` redondea al entero más cercano (0 decimales), devuelve float.
3. `int(total_redondeado)` — Casting a `int` para eliminar el `.0` decimal.

---

### Ejemplo 4: Conversión de moneda (USD a CLP)

```python
precio_usd = 399.99
tipo_cambio = 950.50  # 1 USD = 950.50 CLP
precio_clp = precio_usd * tipo_cambio

print("Precio en USD:", precio_usd)
print("Tipo de cambio:", tipo_cambio)
print("Precio en CLP: $", int(precio_clp))
```

**Salida:**
```
Precio en USD: 399.99
Tipo de cambio: 950.5
Precio en CLP: $ 380180
```

**Explicación:**
1. `precio_usd = 399.99` — Precio del producto en dólares.
2. `tipo_cambio = 950.50` — Tipo de cambio del día.
3. `precio_clp = precio_usd * tipo_cambio` — Multiplica ambos floats.
4. `int(precio_clp)` — Casting a entero porque los precios en CLP no usan decimales.

---

### Ejemplo 5: Impuesto por tramo (progresivo)

```python
total_venta = 850000
impuesto = 0

if total_venta <= 500000:
    impuesto = total_venta * 0.05
elif total_venta <= 1000000:
    # Primeros 500k al 5%, el resto al 10%
    impuesto = 500000 * 0.05 + (total_venta - 500000) * 0.10
else:
    impuesto = total_venta * 0.15

print("Total venta: $", total_venta)
print("Impuesto: $", int(impuesto))
```

**Salida:**
```
Total venta: $ 850000
Impuesto: $ 60000
```

**Explicación:**
1. `total_venta = 850000` — Entero que representa el monto total de la venta.
2. `impuesto = 0` — Inicializa variable donde se almacenará el resultado.
3. `if total_venta <= 500000` — Primer tramo: hasta $500,000 paga 5%.
4. `elif ...` — Segundo tramo: entre $500,001 y $1,000,000. $500,000 * 5% + excedente * 10%.
5. `else:` — Más de $1,000,000 paga 15% sobre todo.
6. `int(impuesto)` — Convierte el float a entero para mostrar sin decimales.

---

### Ejemplo 6: Verificar tipo de producto con `type()`

```python
producto = "Teclado Mecánico"
precio = 45000
stock = 30
disponible = stock > 0

print("producto:", producto, "->", type(producto))
print("precio:", precio, "->", type(precio))
print("stock:", stock, "->", type(stock))
print("disponible:", disponible, "->", type(disponible))
```

**Salida:**
```
producto: Teclado Mecánico -> <class 'str'>
precio: 45000 -> <class 'int'>
stock: 30 -> <class 'int'>
disponible: True -> <class 'bool'>
```

**Explicación:**
1. `producto = "Teclado Mecánico"` — String con el nombre del producto.
2. `precio = 45000` — Entero con el precio.
3. `stock = 30` — Entero con la cantidad en inventario.
4. `disponible = stock > 0` — Expresión booleana: `True` si stock > 0.
5. `type(producto)` — Devuelve `<class 'str'>`, indicando que es una cadena.
6. `type(disponible)` — Devuelve `<class 'bool'>` porque `disponible` es un booleano.

---

### Ejemplo 7: Validar formato de SKU (código alfanumérico)

```python
sku = "LAP-2024-001"
es_valido = len(sku) == 12 and sku[:3].isalpha() and sku[3] == '-'

print("SKU:", sku)
print("¿Tiene 12 caracteres?", len(sku) == 12)
print("¿Primeros 3 son letras?", sku[:3].isalpha())
print("¿Posición 3 es guión?", sku[3] == '-')
print("SKU válido:", es_valido)
```

**Salida:**
```
SKU: LAP-2024-001
¿Tiene 12 caracteres? True
¿Primeros 3 son letras? True
¿Posición 3 es guión? True
SKU válido: True
```

**Explicación:**
1. `sku = "LAP-2024-001"` — String con código de producto.
2. `len(sku)` — Devuelve 12 (cantidad de caracteres).
3. `sku[:3].isalpha()` — Toma los primeros 3 caracteres "LAP" y verifica que sean letras.
4. `sku[3] == '-'` — Verifica que el 4º carácter (índice 3) sea un guión.
5. `len(sku) == 12 and ...` — Operador `and`: todas las condiciones deben ser `True`.
6. El resultado booleano se asigna a `es_valido`.

---

### Ejemplo 8: Comparar precios de dos productos

```python
precio_a = 45000
precio_b = 52000

print("Precio A:", precio_a)
print("Precio B:", precio_b)
print("¿A es más caro que B?", precio_a > precio_b)
print("¿A es igual a B?", precio_a == precio_b)
print("¿A es más barato o igual?", precio_a <= precio_b)
```

**Salida:**
```
Precio A: 45000
Precio B: 52000
¿A es más caro que B? False
¿A es igual a B? False
¿A es más barato o igual? True
```

**Explicación:**
1. `precio_a = 45000` y `precio_b = 52000` — Enteros.
2. `precio_a > precio_b` — Compara si A es mayor que B (45,000 > 52,000 → `False`).
3. `precio_a == precio_b` — Compara igualdad (45,000 == 52,000 → `False`).
4. `precio_a <= precio_b` — Compara menor o igual (45,000 <= 52,000 → `True`).

---

### Ejemplo 9: Calcular cambio en efectivo

```python
total_compra = 18750
pago_efectivo = 20000
cambio = pago_efectivo - total_compra

print("Total compra: $", total_compra)
print("Paga con: $", pago_efectivo)
print("Cambio: $", cambio)
print("¿Cambio exacto?", cambio == 1250)
```

**Salida:**
```
Total compra: $ 18750
Paga con: $ 20000
Cambio: $ 1250
¿Cambio exacto? True
```

**Explicación:**
1. `total_compra = 18750` — Entero con el monto a pagar.
2. `pago_efectivo = 20000` — Entero con el efectivo entregado.
3. `cambio = pago_efectivo - total_compra` — Resta enteros, el cambio es $1,250.
4. `cambio == 1250` — Expresión booleana que evalúa si el cambio es el esperado.

---

### Ejemplo 10: Margen porcentual de ganancia

```python
costo = 8000.00
venta = 12500.00
margen = ((venta - costo) / venta) * 100

print("Costo: $", costo)
print("Venta: $", venta)
print("Margen:", round(margen, 1), "%")
```

**Salida:**
```
Costo: $ 8000.0
Venta: $ 12500.0
Margen: 36.0 %
```

**Explicación:**
1. `costo = 8000.00` — Float, costo de adquisición del producto.
2. `venta = 12500.00` — Float, precio de venta al público.
3. `((venta - costo) / venta) * 100` — Calcula margen como (ganancia / precio venta) × 100.
4. `round(margen, 1)` — Redondea a 1 decimal para claridad.

---

### Ejemplo 11: IVA desglosado (19%) a partir del total con IVA

```python
total_con_iva = 59500.00
iva_incluido = total_con_iva * 0.19 / 1.19
sin_iva = total_con_iva - iva_incluido

print("Total con IVA: $", total_con_iva)
print("IVA incluido: $", round(iva_incluido, 2))
print("Neto (sin IVA): $", round(sin_iva, 2))
```

**Salida:**
```
Total con IVA: $ 59500.0
IVA incluido: $ 9500.0
Neto (sin IVA): $ 50000.0
```

**Explicación:**
1. `total_con_iva = 59500.00` — Precio final del producto con IVA incluido.
2. `total_con_iva * 0.19 / 1.19` — Fórmula para extraer el IVA del total (factor = 0.19/1.19).
3. `sin_iva = total_con_iva - iva_incluido` — Resta para obtener el valor neto.
4. Resultado: $9,500 de IVA y $50,000 de base imponible.

---

### Ejemplo 12: Promedio de precios de varias unidades

```python
p1 = 12000
p2 = 34000
p3 = 8500
p4 = 22000
promedio = (p1 + p2 + p3 + p4) / 4

print("Precios:", p1, p2, p3, p4)
print("Promedio: $", round(promedio, 0))
```

**Salida:**
```
Precios: 12000 34000 8500 22000
Promedio: $ 19125.0
```

**Explicación:**
1. `p1` a `p4` — Cuatro enteros con precios de distintos productos.
2. `(p1 + p2 + p3 + p4) / 4` — Suma todos los precios y divide entre 4.
3. `round(promedio, 0)` — Redondea el float a entero.

---

### Ejemplo 13: Descuento por volumen (más de 10 unidades)

```python
cantidad = 15
precio_unitario = 3500
precio_sin_desc = cantidad * precio_unitario
descuento_volumen = 5000 if cantidad > 10 else 0
total_final = precio_sin_desc - descuento_volumen

print("Cantidad:", cantidad)
print("Precio unitario: $", precio_unitario)
print("Subtotal: $", precio_sin_desc)
print("Descuento volumen: $", descuento_volumen)
print("Total final: $", total_final)
```

**Salida:**
```
Cantidad: 15
Precio unitario: $ 3500
Subtotal: $ 52500
Descuento volumen: $ 5000
Total final: $ 47500
```

**Explicación:**
1. `cantidad = 15` — Entero, unidades compradas.
2. `precio_unitario = 3500` — Entero, precio por unidad.
3. `precio_sin_desc = cantidad * precio_unitario` — Subtotal antes de descuento.
4. `5000 if cantidad > 10 else 0` — Expresión ternaria: si cantidad > 10, descuento de $5,000; si no, $0.
5. `total_final = precio_sin_desc - descuento_volumen` — Aplica el descuento.

---

### Ejemplo 14: Conversión de kilogramos a gramos (inventario)

```python
peso_kg = 2.5
peso_g = peso_kg * 1000
peso_g_entero = int(peso_g)

print("Peso en kg:", peso_kg)
print("Peso en gramos:", peso_g)
print("Peso en gramos (entero):", peso_g_entero)
```

**Salida:**
```
Peso en kg: 2.5
Peso en gramos: 2500.0
Peso en gramos (entero): 2500
```

**Explicación:**
1. `peso_kg = 2.5` — Float representando 2.5 kg de un producto.
2. `peso_kg * 1000` — Multiplica para convertir a gramos (2.5 × 1000 = 2500.0).
3. `int(peso_g)` — Casting a entero: elimina el `.0` decimal.
4. Este patrón se usa al registrar entradas/salidas de inventario en gramos.

---

### Ejemplo 15: Verificar fecha de vencimiento con cadenas

```python
fecha_venc = "2025-12-31"
hoy = "2025-06-15"
vencido = fecha_venc < hoy

print("Fecha vencimiento:", fecha_venc)
print("Fecha hoy:", hoy)
print("¿Está vencido?", vencido)
print("Tipo de vencido:", type(vencido))
```

**Salida:**
```
Fecha vencimiento: 2025-12-31
Fecha hoy: 2025-06-15
¿Está vencido? False
Tipo de vencido: <class 'bool'>
```

**Explicación:**
1. `fecha_venc = "2025-12-31"` — String con formato ISO de fecha de vencimiento.
2. `hoy = "2025-06-15"` — String con la fecha actual.
3. `fecha_venc < hoy` — Comparación lexicográfica de strings en formato ISO. "2025-12-31" > "2025-06-15" porque "1" > "0" en el 5º carácter. El resultado es `False`.
4. `type(vencido)` — Muestra que el resultado es un booleano.

---

### Ejemplo 16: Flag de disponibilidad con casting a bool

```python
stock = 0
disponible = bool(stock)
mensaje = ""

print("Stock:", stock)
print("disponible = bool(stock) ->", disponible)

# Casting de bool a str para mensaje
if disponible:
    mensaje = "Producto disponible"
else:
    mensaje = "Producto AGOTADO"

print("Mensaje:", mensaje)
```

**Salida:**
```
Stock: 0
disponible = bool(stock) -> False
Mensaje: Producto AGOTADO
```

**Explicación:**
1. `stock = 0` — Entero con valor cero (sin stock).
2. `bool(stock)` — Casting a booleano: `0`, `None`, `""` evalúan a `False`; todo lo demás a `True`.
3. `if disponible:` — Condicional que usa el booleano.
4. Se asigna el mensaje correspondiente según disponibilidad.

---

### Ejemplo 17: Casting de string a int para operaciones

```python
entrada_usuario = "5"  # Simula input()
cantidad = int(entrada_usuario)
precio_unit = 12000
total = cantidad * precio_unit

print("Entrada cruda:", repr(entrada_usuario), type(entrada_usuario))
print("Cantidad convertida:", cantidad, type(cantidad))
print("Total: $", total)
```

**Salida:**
```
Entrada cruda: '5' <class 'str'>
Cantidad convertida: 5 <class 'int'>
Total: $ 60000
```

**Explicación:**
1. `entrada_usuario = "5"` — Simula lo que devuelve `input()`: siempre es string.
2. `int(entrada_usuario)` — Casting de string "5" a entero 5 para poder multiplicar.
3. Sin el casting, `"5" * 12000` sería `"1200012000..."` (repetición de string), no una multiplicación aritmética.
4. `repr(entrada_usuario)` — Muestra la representación con comillas, útil para debug.

---

### Ejemplo 18: Validar tipo con `isinstance()`

```python
dato = 45000
print("dato =", dato)
print("isinstance(dato, int):", isinstance(dato, int))
print("isinstance(dato, float):", isinstance(dato, float))
print("isinstance(dato, (int, float)):", isinstance(dato, (int, float)))

precio_str = "45000"
print("\nprecio_str =", repr(precio_str))
print("isinstance(precio_str, str):", isinstance(precio_str, str))
```

**Salida:**
```
dato = 45000
isinstance(dato, int): True
isinstance(dato, float): False
isinstance(dato, (int, float)): True

precio_str = '45000'
isinstance(precio_str, str): True
```

**Explicación:**
1. `isinstance(dato, int)` — Verifica si `dato` es de tipo `int`.
2. `isinstance(dato, (int, float))` — Verifica si `dato` es alguno de los tipos en la tupla.
3. Útil en funciones que aceptan múltiples tipos o en validaciones de datos de entrada.

---

## 3. Ejercicios propuestos

1. **Cálculo de IVA con descuento:** Un producto cuesta $45,000. Aplica 10% descuento y luego 19% IVA. ¿Cuánto paga el cliente?

2. **Rango de precios:** Dado `precios = [1200, 4500, 7800, 25000]`, determina cuántos productos están entre $2,000 y $10,000 usando comparaciones booleanas.

3. **Conversión de unidades:** Convierte 7500 gramos a kilogramos usando casting y operadores. Muestra el resultado como `int` si no tiene decimales.

4. **SKU válido:** Crea una variable `sku = "MON-2024-05A"` y valida que tenga 12 caracteres, comience con 3 letras mayúsculas y tenga guiones en posiciones 3 y 8.

5. **Margen total:** Calcula el margen de ganancia total para 3 productos cuyos costos son [5000, 12000, 8000] y precios de venta [7500, 19500, 12500]. Usa variables individuales.

6. **Flag de reorden:** Un producto tiene `stock = 3` y `stock_minimo = 10`. Si `stock < stock_minimo`, debe reordenarse. Asigna el resultado a una variable booleana `reordenar`.

7. **Promedio ponderado:** Un cliente compra 2 unidades a $15,000 y 3 unidades a $22,000. Calcula el precio promedio ponderado usando floats.

---

## 4. Resumen

- Python tiene 4 tipos básicos: `int`, `float`, `str`, `bool`.
- `type()` y `isinstance()` revelan/verifican el tipo de una variable.
- El casting (`int()`, `float()`, `str()`, `bool()`) convierte entre tipos.
- Las variables en ventas almacenan precios (`float`/`int`), cantidades (`int`), nombres (`str`) y disponibilidad (`bool`).
- Las comparaciones (`>`, `<`, `==`) producen valores booleanos.
- `round()` es esencial para manejar decimales en montos financieros.
- Los operadores aritméticos (`+`, `-`, `*`, `/`) funcionan entre tipos compatibles.
- Strings en formato ISO permiten comparaciones básicas de fechas.
- `repr()` es útil para debuggear strings mostrando las comillas.
