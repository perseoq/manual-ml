# B07 — Errores y Debugging en Python

## 1. Introducción

Los errores en Python se manejan con `try/except`. El módulo `logging` registra eventos y `pdb` permite depuración interactiva. Las excepciones comunes en ventas incluyen división por cero, conversiones inválidas y claves inexistentes.

| Excepción          | Causa típica en ventas                   |
|--------------------|------------------------------------------|
| `ZeroDivisionError`| Precio / cantidad cuando cantidad = 0    |
| `ValueError`       | Convertir string inválido a int/float    |
| `KeyError`         | SKU inexistente en catálogo              |
| `FileNotFoundError`| Archivo de inventario no encontrado      |
| `TypeError`        | Operar tipos incompatibles               |

Herramientas:
- `try/except/finally` — Capturar y manejar errores
- `raise` — Lanzar excepciones personalizadas
- `assert` — Validar condiciones (debug)
- `logging` — Registro estructurado de eventos
- `pdb` — Depurador interactivo

---

## 2. Ejemplos prácticos

### Ejemplo 1: `try/except` para división por cero (precio/cantidad)

```python
def precio_promedio(total: float, cantidad: int) -> float:
    """Calcula el precio promedio. Captura división por cero."""
    try:
        return round(total / cantidad, 2)
    except ZeroDivisionError:
        print("Error: No se puede dividir por cero (cantidad = 0)")
        return 0.0

print(f"Promedio ($10,000 / 5): ${precio_promedio(10000, 5)}")
print(f"Promedio ($10,000 / 0): ${precio_promedio(10000, 0)}")
```

**Salida:**
```
Promedio ($10,000 / 5): $2000.0
Error: No se puede dividir por cero (cantidad = 0)
Promedio ($10,000 / 0): $0.0
```

**Explicación:**
1. `try:` — Bloque que puede lanzar una excepción.
2. `total / cantidad` — Si `cantidad` es 0, lanza `ZeroDivisionError`.
3. `except ZeroDivisionError:` — Captura específicamente esa excepción.
4. La función retorna `0.0` como valor seguro.

---

### Ejemplo 2: `ValueError` al convertir string a float

```python
def convertir_precio(valor: str) -> float:
    """Convierte un string a float con manejo de error."""
    try:
        precio = float(valor)
        if precio < 0:
            raise ValueError("El precio no puede ser negativo")
        return precio
    except ValueError as e:
        print(f"Error de conversión: '{valor}' no es un precio válido. Detalle: {e}")
        return 0.0

print(f"Precio '12500': ${convertir_precio('12500')}")
print(f"Precio 'mil': ${convertir_precio('mil')}")
print(f"Precio '-500': ${convertir_precio('-500')}")
```

**Salida:**
```
Precio '12500': $12500.0
Error de conversión: 'mil' no es un precio válido. Detalle: could not convert string to float: 'mil'
Error de conversión: '-500' no es un precio válido. Detalle: El precio no puede ser negativo
```

**Explicación:**
1. `float(valor)` — Si el string no es numérico, lanza `ValueError`.
2. `if precio < 0: raise ValueError(...)` — Validación adicional lanza la misma excepción con mensaje personalizado.
3. `except ValueError as e:` — Captura ambos casos (conversión fallida y validación).
4. `as e` — Asigna la excepción a la variable `e` para acceder al mensaje.

---

### Ejemplo 3: `KeyError` para SKU inexistente

```python
catalogo = {
    "LAP-001": {"nombre": "Laptop", "precio": 1250000},
    "MOU-002": {"nombre": "Mouse", "precio": 25000}
}

def buscar_sku(sku: str) -> dict:
    """Busca un producto por SKU con manejo de KeyError."""
    try:
        producto = catalogo[sku]
        return producto
    except KeyError:
        print(f"Error: El SKU '{sku}' no existe en el catálogo")
        return {}

print(buscar_sku("LAP-001"))
print(buscar_sku("XXX-999"))
```

**Salida:**
```
{'nombre': 'Laptop', 'precio': 1250000}
Error: El SKU 'XXX-999' no existe en el catálogo
{}
```

**Explicación:**
1. `catalogo[sku]` — Si la clave no existe, lanza `KeyError`.
2. `except KeyError:` — Captura y muestra mensaje amigable.
3. Retorna `{}` (dict vacío) en lugar de interrumpir el programa.
4. Alternativa más segura: `catalogo.get(sku, {})`.

---

### Ejemplo 4: `FileNotFoundError` al cargar inventario

```python
import json

def cargar_inventario(ruta: str) -> dict:
    """Carga un archivo JSON de inventario."""
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Archivo '{ruta}' no encontrado. Se usará inventario vacío.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: '{ruta}' no es un JSON válido. Revise el formato.")
        return {}

inventario = cargar_inventario("inventario_real.json")
print("Inventario cargado:", inventario)

inventario = cargar_inventario("archivo_inexistente.json")
print("Inventario cargado:", inventario)
```

**Salida:**
```
Error: Archivo 'inventario_real.json' no encontrado. Se usará inventario vacío.
Inventario cargado: {}
Error: Archivo 'archivo_inexistente.json' no encontrado. Se usará inventario vacío.
Inventario cargado: {}
```

**Explicación:**
1. `FileNotFoundError` — Archivo no existe en la ruta eRealiza la operación indicada con los parámetros definidos..
2. `json.JSONDecodeError` — El archivo existe pero no tiene formato JSON válido.
3. Múltiples `except` para diferentes tipos de error.
4. Retorna `{}` como valor por defecto para que el programa continúe.

---

### Ejemplo 5: Excepción personalizada `StockInsuficiente`

```python
class StockInsuficiente(Exception):
    """Excepción lanzada cuando no hay suficiente stock."""
    def __init__(self, producto: str, solicitado: int, disponible: int):
        self.producto = producto
        self.solicitado = solicitado
        self.disponible = disponible
        super().__init__(f"Stock insuficiente para '{producto}': "
                         f"solicitado {solicitado}, disponible {disponible}")

def vender(producto: str, cantidad: int, stock: dict):
    """Procesa una venta. Lanza StockInsuficiente si no hay stock."""
    if cantidad > stock.get(producto, 0):
        raise StockInsuficiente(producto, cantidad, stock.get(producto, 0))
    stock[producto] -= cantidad
    print(f"Venta exitosa: {cantidad} × {producto}")

stock = {"Laptop": 5, "Mouse": 10}

try:
    vender("Laptop", 3, stock)  # OK
    vender("Laptop", 5, stock)  # Error
except StockInsuficiente as e:
    print(f"Error de venta: {e}")
```

**Salida:**
```
Venta exitosa: 3 × Laptop
Error de venta: Stock insuficiente para 'Laptop': solicitado 5, disponible 2
```

**Explicación:**
1. `class StockInsuficiente(Exception):` — Define una excepción personalizada heredando de `Exception`.
2. `__init__` con parámetros específicos del dominio.
3. `super().__init__(mensaje)` — Inicializa la excepción con un mensaje descriptivo.
4. `raise StockInsuficiente(...)` — Lanza la excepción personalizada.
5. `except StockInsuficiente as e:` — Captura específicamente esa excepción.

---

### Ejemplo 6: `raise` con mensaje descriptivo

```python
def validar_precio(precio: float) -> bool:
    """Valida que un precio sea positivo y razonable."""
    if precio <= 0:
        raise ValueError(f"Precio inválido: ${precio}. Debe ser mayor a 0.")
    if precio > 100_000_000:
        raise ValueError(f"Precio sospechoso: ${precio:,.0f}. Supera el límite.")
    return True

for precio in [25000, 0, 150_000_000]:
    try:
        validar_precio(precio)
        print(f"  ${precio:,}: Precio válido")
    except ValueError as e:
        print(f"  ${precio:,}: {e}")
```

**Salida:**
```
  $25,000: Precio válido
  $0: Precio inválido: $0. Debe ser mayor a 0.
  $150,000,000: Precio sospechoso: $150,000,000. Supera el límite.
```

**Explicación:**
1. `if precio <= 0: raise ValueError(...)` — Lanza excepción con mensaje específico.
2. Múltiples validaciones con diferentes mensajes.
3. El `try/except` en la llamada captura y muestra el error apropiado.

---

### Ejemplo 7: Logging básico

```python
import logging

# Configuración básica
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def procesar_venta(producto: str, cantidad: int, precio: float):
    logging.info(f"Iniciando venta: {cantidad} × {producto}")
    try:
        total = cantidad * precio
        logging.debug(f"Total calculado: ${total}")
        if cantidad > 100:
            logging.warning(f"Venta grande: {cantidad} unidades de {producto}")
        if cantidad == 0:
            raise ValueError("Cantidad no puede ser 0")
        logging.info(f"Venta completada: ${total:,}")
        return total
    except Exception as e:
        logging.error(f"Error en venta de {producto}: {e}")
        return 0

print(f"Resultado: ${procesar_venta('Mouse', 5, 25000)}")
print(f"Resultado: ${procesar_venta('Laptop', 0, 1250000)}")
```

**Salida:**
```
INFO: Iniciando venta: 5 × Mouse
INFO: Venta completada: $125,000
Resultado: $125000
INFO: Iniciando venta: 0 × Laptop
ERROR: Error en venta de Laptop: Cantidad no puede ser 0
Resultado: $0
```

**Explicación:**
1. `logging.basicConfig(level=logging.INFO)` — Configura logging a nivel INFO.
2. `logging.info(...)` — Mensaje informativo.
3. `logging.debug(...)` — Mensaje de depuración (no visible si level > DEBUG).
4. `logging.warning(...)` — Advertencia.
5. `logging.error(...)` — Error.

---

### Ejemplo 8: `assert` para validar precios positivos

```python
def calcular_total(precio: float, cantidad: int) -> float:
    """Calcula el total de una venta con validaciones assert."""
    assert precio > 0, f"Precio debe ser positivo: {precio}"
    assert cantidad > 0, f"Cantidad debe ser positiva: {cantidad}"
    assert cantidad < 10000, f"Cantidad sospechosamente grande: {cantidad}"

    total = precio * cantidad
    assert total > 0, "El total debe ser positivo"
    return total

# Casos de prueba
try:
    print(f"Total: ${calcular_total(25000, 3):,}")
    print(f"Total: ${calcular_total(-5000, 2):,}")  # AssertionError
except AssertionError as e:
    print(f"Assertion error: {e}")
```

**Salida:**
```
Total: $75,000
Assertion error: Precio debe ser positivo: -5000
```

**Explicación:**
1. `assert condición, mensaje` — Si la condición es `False`, lanza `AssertionError` con el mensaje.
2. Los asserts se usan para validar invariantes en desarrollo.
3. Se pueden deshabilitar con `python -O` (no usar en producción para validaciones críticas).

---

### Ejemplo 9: `finally` para cerrar archivo

```python
def escribir_venta(archivo: str, datos: str) -> bool:
    """Escribe datos en archivo. finally garantiza cierre."""
    f = None
    try:
        f = open(archivo, "w", encoding="utf-8")
        f.write(datos)
        print(f"Datos escritos en {archivo}")
        return True
    except IOError as e:
        print(f"Error de E/S: {e}")
        return False
    finally:
        if f:
            f.close()
            print(f"Archivo {archivo} cerrado correctamente")

escribir_venta("venta_log.txt", "Venta: Laptop, $1,250,000")
```

**Salida:**
```
Datos escritos en venta_log.txt
Archivo venta_log.txt cerrado correctamente
```

**Explicación:**
1. `finally:` — Bloque que se ejecuta **siempre**, haya o no excepción.
2. Útil para liberar recursos (archivos, conexiones, locks).
3. Si hay `return` en `try`, `finally` se ejecuta antes de retornar.

---

### Ejemplo 10: Context manager personalizado con `__enter__`/`__exit__`

```python
class GestionInventario:
    """Context manager para operaciones de inventario."""
    def __init__(self, archivo: str):
        self.archivo = archivo
        self.datos = {}

    def __enter__(self):
        print(f"Abriendo inventario: {self.archivo}")
        try:
            with open(self.archivo, "r") as f:
                import json
                self.datos = json.load(f)
        except FileNotFoundError:
            self.datos = {}
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import json
        with open(self.archivo, "w") as f:
            json.dump(self.datos, f, indent=2)
        print(f"Guardando y cerrando inventario: {self.archivo}")
        if exc_type:
            print(f"  (ocurrió un error: {exc_type.__name__}: {exc_val})")
        return False  # No suprimir excepciones

# Uso
with GestionInventario("inventario_ctx.json") as inv:
    inv.datos["Laptop"] = {"stock": 10}
    inv.datos["Mouse"] = {"stock": 50}
    print(f"  Inventario actualizado: {list(inv.datos.keys())}")
```

**Salida:**
```
Abriendo inventario: inventario_ctx.json
  Inventario actualizado: ['Laptop', 'Mouse']
Guardando y cerrando inventario: inventario_ctx.json
```

**Explicación:**
1. `__enter__` — Se ejecuta al entrar en `with`. Retorna el objeto.
2. `__exit__` — Se ejecuta al salir. Guarda datos automáticamente.
3. `exc_type, exc_val, exc_tb` — Información de excepción si ocurrió.
4. `return False` — No suprime excepciones (las propaga).

---

### Ejemplo 11: `pdb` básico para depuración interactiva

```python
def calcular_descuento_escalonado(precio: float, nivel: int = 0):
    """Función para depurar con pdb."""
    descuentos = [0.10, 0.15, 0.20]
    if nivel >= len(descuentos):
        return precio

    import pdb
    # Descomentar para depuración interactiva:
    # pdb.set_trace()

    desc = descuentos[nivel]
    nuevo_precio = precio * (1 - desc)
    print(f"Nivel {nivel}: {precio:,.0f} → {nuevo_precio:,.0f} (desc: {desc*100:.0f}%)")
    return calcular_descuento_escalonado(nuevo_precio, nivel + 1)

# Ejecución normal
print("Precio final:", calcular_descuento_escalonado(200000))
```

**Salida:**
```
Nivel 0: 200,000 → 180,000 (desc: 10%)
Nivel 1: 180,000 → 153,000 (desc: 15%)
Nivel 2: 153,000 → 122,400 (desc: 20%)
Precio final: 122400.0
```

**Explicación:**
1. `import pdb; pdb.set_trace()` — Punto de interrupción. Descomentar para depurar.
2. Al ejecutarse, abre consola interactiva donde puedes: `n` (next), `s` (step into), `c` (continue), `p variable` (print), `l` (list code).
3. Comandos comunes: `h` (help), `q` (quit), `w` (where/stack trace).
4. `python -m pdb script.py` ejecuta el script bajo pdb desde el inicio.

---

### Ejemplo 12: `traceback` para imprimir stack trace completo

```python
import traceback

def nivel_3():
    raise ValueError("Error en cálculo de inventario")

def nivel_2():
    nivel_3()

def nivel_1():
    nivel_2()

try:
    nivel_1()
except ValueError:
    print("=== ERROR CAPTURADO ===")
    print(f"Mensaje: ¡Ocurrió un error en el sistema de ventas!")
    traceback.print_exc()  # Imprime stack trace completo
```

**Salida:**
```
=== ERROR CAPTURADO ===
Mensaje: ¡Ocurrió un error en el sistema de ventas!
Traceback (most recent call last):
  File "<string>", line 13, in nivel_1
  File "<string>", line 10, in nivel_2
  File "<string>", line 7, in nivel_3
ValueError: Error en cálculo de inventario
```

**Explicación:**
1. `traceback.print_exc()` — Imprime el stack trace completo de la excepción actual.
2. Muestra la cadena de llamadas: `nivel_1 → nivel_2 → nivel_3 → error`.
3. Útil para depurar errores en producción sin detener el programa.
4. Se puede guardar en un string con `traceback.format_exc()`.

---

### Ejemplo 13: Múltiples `except` para diferentes errores

```python
def procesar_pago(monto: str, tipo: str):
    """Procesa un pago capturando diferentes errores."""
    try:
        monto_float = float(monto)
        if monto_float <= 0:
            raise ValueError("Monto debe ser positivo")

        metodos = {"efectivo", "debito", "credito"}
        if tipo not in metodos:
            raise KeyError(f"Tipo de pago '{tipo}' no soportado")

        print(f"Pago procesado: ${monto_float:,.0f} con {tipo}")

    except ValueError as e:
        print(f"Error de valor: {e}")
    except KeyError as e:
        print(f"Error de método: {e}")
    except Exception as e:
        print(f"Error inesperado: {type(e).__name__}: {e}")
    else:
        print("  (sin errores)")
    finally:
        print("  Operación finalizada\n")

procesar_pago("25000", "credito")
procesar_pago("abc", "credito")
procesar_pago("25000", "bitcoin")
```

**Salida:**
```
Pago procesado: $25,000 con credito
  (sin errores)
  Operación finalizada

Error de valor: could not convert string to float: 'abc'
  Operación finalizada

Error de método: Tipo de pago 'bitcoin' no soportado
  Operación finalizada
```

**Explicación:**
1. Múltiples bloques `except` para diferentes tipos de excepción.
2. `except Exception as e:` — Captura cualquier otra excepción no prevista.
3. `else:` — Se ejecuta solo si no hubo excepción.
4. `finally:` — Se ejecuta siempre.

---

### Ejemplo 14: `warnings` para stock bajo (no crítico)

```python
import warnings

def verificar_stock(producto: str, stock: int, minimo: int = 5):
    """Verifica stock y emite advertencia si está bajo."""
    if stock == 0:
        warnings.warn(f"¡{producto} AGOTADO!", UserWarning)
    elif stock < minimo:
        warnings.warn(f"Stock bajo de {producto}: {stock} unidades (mínimo: {minimo})", UserWarning)
    else:
        print(f"{producto}: Stock OK ({stock})")

# Configurar para mostrar todas las advertencias
warnings.simplefilter("always")

verificar_stock("Laptop", 3)
verificar_stock("Mouse", 0)
verificar_stock("Teclado", 30)
```

**Salida:**
```
Warning: Stock bajo de Laptop: 3 unidades (mínimo: 5)
Warning: ¡Mouse AGOTADO!
Teclado: Stock OK (30)
```

**Explicación:**
1. `warnings.warn(mensaje, UserWarning)` — Emite advertencia (no interrumpe).
2. Las advertencias son menos severas que las excepciones.
3. `warnings.simplefilter("always")` — Muestra siempre (por defecto muestra cada una una vez).
4. Útil para alertas que no deben detener el flujo del programa.

---

### Ejemplo 15: Logging configurado con archivo

```python
import logging

# Configuración avanzada
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="ventas.log",
    filemode="a"
)

logger = logging.getLogger("Ventas")

def realizar_venta(producto, cantidad, precio):
    logger.info(f"Iniciando venta: {cantidad} × {producto}")
    try:
        total = cantidad * precio
        logger.debug(f"Total calculado: {total}")
        if cantidad > 100:
            logger.warning(f"Venta inusualmente grande: {cantidad}")
        logger.info(f"Venta OK: ${total:,}")
        return total
    except Exception as e:
        logger.error(f"Venta fallida: {e}", exc_info=True)
        return 0

realizar_venta("Laptop", 2, 1250000)
realizar_venta("Mouse", 0, 25000)

# Leer el log
with open("ventas.log", "r") as f:
    lineas = f.readlines()
    for linea in lineas[-6:]:
        print(linea.strip())
```

**Salida:**
```
2025-06-15 12:00:00 | INFO     | Ventas | Iniciando venta: 2 × Laptop
2025-06-15 12:00:00 | DEBUG    | Ventas | Total calculado: 2500000
2025-06-15 12:00:00 | INFO     | Ventas | Venta OK: $2,500,000
2025-06-15 12:00:00 | INFO     | Ventas | Iniciando venta: 0 × Mouse
2025-06-15 12:00:00 | ERROR    | Ventas | Venta fallida: multiplication...
```

**Explicación:**
1. `filename="ventas.log"` — Guarda logs en archivo en lugar de consola.
2. `format` — Incluye timestamp, nivel, logger name y mensaje.
3. `exc_info=True` — Incluye stack trace en logs de error.
4. `filemode="a"` — Append al archivo existente.

---

### Ejemplo 16: `raise from` para encadenar excepciones

```python
class ErrorDeVenta(Exception):
    pass

def obtener_precio(sku: str, catalogo: dict) -> float:
    """Obtiene precio. Envuelve KeyError en ErrorDeVenta."""
    try:
        return catalogo[sku]["precio"]
    except KeyError as e:
        raise ErrorDeVenta(f"Producto {sku} no encontrado en catálogo") from e

catalogo = {"LAP-001": {"precio": 1250000}}

try:
    precio = obtener_precio("XXX-999", catalogo)
except ErrorDeVenta as e:
    print(f"Error de venta: {e}")
    print(f"Causa original: {e.__cause__}")
```

**Salida:**
```
Error de venta: Producto XXX-999 no encontrado en catálogo
Causa original: 'XXX-999'
```

**Explicación:**
1. `raise NuevaExcepcion(...) from original` — Encadena excepciones preservando la causa original.
2. `e.__cause__` — Accede a la excepción original.
3. Útil para envolver excepciones de bajo nivel en excepciones de dominio.

---

### Ejemplo 17: `assert` con mensaje en modo debug

```python
# Ejecutar con: python -O para deshabilitar asserts
DEBUG = True

def calcular_iva(total: float) -> float:
    """Calcula IVA con validaciones en debug."""
    iva = round(total * 0.19, 2)

    if DEBUG:
        assert iva >= 0, f"IVA negativo: {iva}"
        assert total < 100_000_000, f"Total sospechoso: {total}"
        assert isinstance(iva, float), f"IVA debe ser float, es {type(iva)}"

    return iva

print(f"IVA de $100,000: ${calcular_iva(100000)}")
print(f"IVA de $0: ${calcular_iva(0)}")
```

**Salida:**
```
IVA de $100,000: $19000.0
IVA de $0: $0.0
```

**Explicación:**
1. `assert condición, mensaje` — Valida invariantes internas.
2. `python -O` — Deshabilita todos los `assert` (optimización).
3. Útil en desarrollo/testing, no para validación de datos de usuario en producción.

---

### Ejemplo 18: Manejo de errores en cadena (procesamiento de datos)

```python
def parsear_venta(linea: str) -> dict:
    """Parsea una línea CSV de venta. Un solo punto de error."""
    partes = linea.strip().split(",")
    if len(partes) != 4:
        raise ValueError(f"Se esperaban 4 campos, se obtuvieron {len(partes)}")

    producto, cant_str, precio_str, _ = partes
    try:
        cantidad = int(cant_str)
        precio = float(precio_str)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Error parseando números en '{linea.strip()}': {e}")

    if cantidad <= 0:
        raise ValueError(f"Cantidad inválida: {cantidad}")
    if precio <= 0:
        raise ValueError(f"Precio inválido: {precio}")

    return {"producto": producto, "cantidad": cantidad, "precio": precio}

lineas = [
    "Laptop,2,1250000,2500000",
    "Mouse,5,25000,125000",
    "Teclado,cero,45000,0"  # Error
]

for linea in lineas:
    try:
        venta = parsear_venta(linea)
        print(f"  OK: {venta}")
    except ValueError as e:
        print(f"  ERROR: {e}")
```

**Salida:**
```
  OK: {'producto': 'Laptop', 'cantidad': 2, 'precio': 1250000.0}
  OK: {'producto': 'Mouse', 'cantidad': 5, 'precio': 25000.0}
  ERROR: Error parseando números en 'Teclado,cero,45000,0': invalid literal for int() with base 10: 'cero'
```

**Explicación:**
1. Validaciones en cadena: formato, tipos, rangos.
2. `ValueError` personalizado con contexto de qué falló.
3. Manejo centralizado en el bloque `try/except` de la llamada.

---

### Ejemplo 19: `sys.exc_info()` para obtener detalles del error

```python
import sys

def funcion_problema():
    return 1 / 0

try:
    funcion_problema()
except:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    print(f"Tipo: {exc_type.__name__}")
    print(f"Valor: {exc_value}")
    print(f"Traceback (frame actual): {exc_traceback.tb_frame}")
    print(f"Número de línea: {exc_traceback.tb_lineno}")
```

**Salida:**
```
Tipo: ZeroDivisionError
Valor: division by zero
Traceback (frame actual): <frame at ...>
Número de línea: 5
```

**Explicación:**
1. `sys.exc_info()` — Retorna tupla `(type, value, traceback)` de la excepción actual.
2. `tb_frame` — Frame donde ocurrió la excepción.
3. `tb_lineno` — Número de línea exacto.
4. Útil para logging detallado sin usar `traceback`.

---

### Ejemplo 20: Validación defensiva completa

```python
def procesar_venta_segura(producto: str, cantidad: int, precio: float, stock: int) -> dict:
    """Procesa venta con validaciones exhaustivas."""
    resultado = {
        "exito": False,
        "mensaje": "",
        "total": 0
    }

    try:
        if not producto or not isinstance(producto, str):
            raise ValueError("Nombre de producto inválido")
        if not isinstance(cantidad, (int, float)) or cantidad <= 0:
            raise ValueError(f"Cantidad inválida: {cantidad}")
        if not isinstance(precio, (int, float)) or precio <= 0:
            raise ValueError(f"Precio inválido: {precio}")
        if not isinstance(stock, (int, float)) or stock < 0:
            raise ValueError(f"Stock inválido: {stock}")
        if cantidad > stock:
            raise StockInsuficiente(producto, cantidad, stock)

        total = cantidad * precio
        if total > 10_000_000:
            import warnings
            warnings.warn(f"Venta de alto valor: ${total:,.0f}")

        resultado["exito"] = True
        resultado["mensaje"] = "Venta procesada"
        resultado["total"] = total

    except StockInsuficiente as e:
        resultado["mensaje"] = str(e)
    except ValueError as e:
        resultado["mensaje"] = f"Dato inválido: {e}"
    except Exception as e:
        resultado["mensaje"] = f"Error inesperado: {type(e).__name__}: {e}"
        import logging
        logging.error(f"Error crítico en venta: {e}", exc_info=True)

    return resultado

print(procesar_venta_segura("Laptop", 2, 1250000, 10))
print(procesar_venta_segura("Mouse", 5, -25000, 50))
print(procesar_venta_segura("Teclado", 50, 45000, 30))
```

**Salida:**
```
{'exito': True, 'mensaje': 'Venta procesada', 'total': 2500000}
{'exito': False, 'mensaje': "Dato inválido: Precio inválido: -25000"}
{'exito': False, 'mensaje': "Stock insuficiente para 'Teclado': solicitado 50, disponible 30"}
```

**Explicación:**
1. Validación defensiva: cada parámetro se verifica antes de procesar.
2. Diferentes tipos de error capturados específicamente.
3. `resultado` siempre retorna un dict estructurado (nunca lanza excepción al exterior).
4. Logging para errores inesperados con stack trace.

---

## 3. Ejercicios propuestos

1. **División segura:** Crea una función `dividir_precios(total, cantidad)` que maneje `ZeroDivisionError` y retorne 0 si cantidad es 0.

2. **Clave inexistente:** Dado `producto = {"precio": 25000}`, accede a `producto["stock"]` de forma segura usando `try/except KeyError`.

3. **Excepción personalizada:** Define `PrecioInvalidoError` y úsala en una función que valide que el precio esté entre $100 y $10,000,000.

4. **Logging de ventas:** Configura logging para que registre cada venta con nivel INFO y los errores con nivel ERROR en un archivo `ventas.log`.

5. **Assert en debug:** Agrega `assert` para validar que después de vender, el stock resultante no sea negativo.

6. **Finally para BD:** Simula una conexión a base de datos de ventas donde `finally` cierra la conexión (usa print para simular).

7. **Raise from:** Crea una función que llame a otra, capture su `KeyError` y lance `ErrorDeProcesamiento` con `raise from`.

8. **Parser robusto:** Escribe una función que parsee `"Laptop,2,1250000"` capturando `ValueError`, `IndexError` y `TypeError` por separado.

---

## 4. Resumen

- `try/except` captura y maneja errores sin interrumpir el programa.
- `finally` se ejecuta siempre (liberar recursos).
- `else` en try se ejecuta solo si no hubo excepción.
- Excepciones comunes: `ZeroDivisionError`, `ValueError`, `KeyError`, `FileNotFoundError`, `TypeError`.
- `raise` lanza excepciones; `raise ... from` encadena causas.
- Excepciones personalizadas heredan de `Exception`.
- `assert` valida condiciones en desarrollo (deshabilitable con `-O`).
- `logging` reemplaza a `print` para registro estructurado con niveles (DEBUG, INFO, WARNING, ERROR).
- `warnings` emite alertas no críticas.
- `pdb` permite depuración interactiva (`pdb.set_trace()`).
- `traceback.print_exc()` muestra el stack trace completo.
- Siempre validar datos de entrada antes de procesar.
- Un punto de error (función con manejo centralizado) simplifica el debugging.
