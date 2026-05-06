# Documentación del Generador de Datos Sintéticos

## Módulo: `datos/datos_sinteticos.py`

---

# Tabla de Contenidos

- [Introducción](#introducción)
- [Estructura del Módulo](#estructura-del-módulo)
- [Constantes Globales](#constantes-globales)
- [Función: `generar_ventas()`](#función-generar_ventas)
- [Función: `generar_inventario()`](#función-generar_inventario)
- [Función: `generar_compras()`](#función-generar_compras)
- [Función: `generar_clientes_con_compras()`](#función-generar_clientes_con_compras)
- [Función: `generar_resenas()`](#función-generar_resenas)
- [Ejecución Directa](#ejecución-directa)
- [Guía de Uso](#guía-de-uso)
- [Ejemplos de Carga](#ejemplos-de-carga)
- [Consideraciones sobre los Datos](#consideraciones-sobre-los-datos)

---

## Introducción

El módulo `datos/datos_sinteticos.py` es el generador de datos sintéticos que alimenta todos los ejemplos y ejercicios del manual. Genera **5 datasets** interrelacionados que simulan las operaciones de una empresa comercializadora de productos tecnológicos:

| Dataset | Descripción | Registros aprox. |
|---------|-------------|-----------------|
| `ventas` | Transacciones de venta diarias | ~1,330 |
| `inventario` | Estado actual del inventario de productos | 25 |
| `compras` | Órdenes de compra a proveedores | 200 |
| `clientes` | Perfil RFM de clientes | 200 |
| `reseñas` | Reseñas y opiniones de productos | 100 |

### Propósito

- Proveer datos realistas con **sesgo de negocio** (estacionalidad, descuentos, retrasos)
- Asegurar **reproducibilidad** mediante `np.random.seed(42)`
- Cubrir distintos **tipos de datos**: numéricos, categóricos, texto, fechas
- Incluir **valores nulos y atípicos** para practicar limpieza de datos
- Permitir **análisis relacional** entre tablas (JOINs, merges)

### Dependencias

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
```

**Salida:**

```
# Ver resultado al ejecutar el código
```

**Explicación:**

El código realiza operaciones de configuración/instalación/navegación. Consultar las instrucciones para más detalles.

---


---

## Estructura del Módulo

```
datos_sinteticos.py
├── CONSTANTES GLOBALES
│   ├── PRODUCTOS (25 productos en 8 categorías)
│   ├── PROVEEDORES (7 proveedores)
│   ├── SUCURSALES (9 sucursales)
│   └── CLIENTES (12 nombres de cliente)
│
├── FUNCIONES GENERADORAS
│   ├── generar_ventas(n_dias, productos, sucursales, clientes)
│   ├── generar_inventario(productos)
│   ├── generar_compras(n_ordenes, productos, proveedores)
│   ├── generar_clientes_con_co

**Salida:**

```
# Ver resultado al ejecutar el código
```

**Explicación:**

El código realiza operaciones de configuración/instalación/navegación. Consultar las instrucciones para más detalles.

---
mpras(n_clientes)
│   └── generar_resenas(n_resenas)
│
└── EJECUCIÓN DIRECTA (if __name__ == "__main__")
    ├── Genera todos los datasets
    ├── Exporta a CSV en datos/
    └── Imprime estadísticas
```

---

## Constantes Globales

### `PRODUCTOS`

Lista de 25 tuplas con la estructura `(sku, nombre, categoria, costo, precio)`.

| SKU | Nombre | Categoría | Costo | Precio |
|-----|--------|-----------|-------|--------|
| LAP001 | Laptop Pro 15 | Electrónica | $12,000 | $15,000 |
| LAP002 | Laptop Air 13 | Electrónica | $9,000 | $11,500 |
| MON001 | Monitor 27 4K | Electrónica | $5,500 | $7,200 |
| MON002 | Monitor 24 HD | Electrónica | $2,500 | $3,400 |
| TEC001 | Teclado Mecánico | Periféricos | $800 | $1,400 |
| TEC002 | Teclado Inalámbrico | Periféricos | $450 | $750 |
| MOU001 | Mouse Ergonómico | Periféricos | $350 | $650 |
| MOU002 | Mouse Gamer | Periféricos | $600 | $1,100 |
| AUD001 | Audífonos Bluetooth | Audio | $1,200 | $2,200 |
| AUD002 | Parlante Portátil | Audio | $800 | $1,500 |
| AUD003 | Micrófono USB | Audio | $1,500 | $2,800 |
| WEB001 | Webcam HD | Cámaras | $900 | $1,700 |
| WEB002 | Cámara Seguridad | Cámaras | $1,800 | $3,200 |
| DIS001 | SSD 1TB | Almacenamiento | $1,500 | $2,500 |
| DIS002 | HDD 4TB | Almacenamiento | $1,200 | $1,900 |
| DIS003 | USB 64GB | Almacenamiento | $150 | $350 |
| RED001 | Router WiFi 6 | Redes | $1,200 | $2,100 |
| RED002 | Switch 8 puertos | Redes | $600 | $1,100 |
| SOF001 | Office 365 1 año | Software | $900 | $1,800 |
| SOF002 | Antivirus 3 equipos | Software | $400 | $800 |
| PAP001 | Papel Bond 5000 hojas | Papelería | $200 | $400 |
| PAP002 | Tinta Impresora | Papelería | $250 | $500 |
| MUE001 | Silla Ergonómica | Muebles | $3,500 | $6,500 |
| MUE002 | Escritorio Eléctrico | Muebles | $4,500 | $8,500 |
| MUE003 | Lámpara LED | Muebles | $300 | $600 |

**Categorías disponibles:** Electrónica, Periféricos, Audio, Cámaras, Almacenamiento, Redes, Software, Papelería, Muebles.

### `PROVEEDORES`

Lista de 7 tuplas con la estructura `(id, nombre, calidad_pct, plazo_dias)`.

| ID | Nombre | Calidad (%) | Plazo (días) |
|----|--------|-------------|--------------|
| PROV001 | Distribuidora Tecnológica S.A. | 85 | 3 |
| PROV002 | Importaciones Globales Ltda. | 70 | 7 |
| PROV003 | Suministros Empresariales C.A. | 92 | 2 |
| PROV004 | TecnoPartes del Sur | 60 | 10 |
| PROV005 | Logística Integral de Cómputo | 88 | 4 |
| PROV006 | Comercializadora Digital Express | 75 | 5 |
| PROV007 | Mayorista de Tecnología | 95 | 1 |

**Nota:** La calidad es un porcentaje (0-100) basado en cumplimiento. El plazo son días estimados de entrega.

### `SUCURSALES`

Lista de 9 nombres de sucursales:

| # | Sucursal |
|---|----------|
| 1 | Matriz CDMX |
| 2 | Sucursal Monterrey |
| 3 | Sucursal Guadalajara |
| 4 | Sucursal Puebla |
| 5 | Sucursal Querétaro |
| 6 | Sucursal Cancún |
| 7 | Sucursal Toluca |
| 8 | Sucursal Mérida |
| 9 | Sucursal Tijuana |

### `CLIENTES`

Lista de 12 nombres de cliente usados en ventas y reseñas:

| # | Cliente |
|---|---------|
| 1 | Cliente Corp

**Salida:**

```
# Ver resultado al ejecutar el código
```

**Explicación:**

El código realiza operaciones de configuración/instalación/navegación. Consultar las instrucciones para más detalles.

---
 |
| 2 | Empresa XYZ |
| 3 | Comercial MX |
| 4 | Distribuidora ABC |
| 5 | Tienda 123 |
| 6 | Mayorista JJ |
| 7 | Comprador Final |
| 8 | Empresa Beta |
| 9 | Soluciones Inc |
| 10 | Grupo Gamma |
| 11 | Venta Mostrador |
| 12 | Cliente Premium |

---

## Función: `generar_ventas()`

### Firma

```python
def generar_ventas(n_dias=180, productos=None, sucursales=None, clientes=None) -> pd.DataFrame
```

### Parámetros

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `n_dias` | `int` | 180 | Número de días a generar (desde 2024-01-01) |
| `productos` | `list[tuple]` | `PRODUCTOS` | Lista de productos a usar |
| `sucursales` | `list[str]` | `SUCURSALES` | Lista de sucursales |
| `clientes` | `list[str]` | `CLIENTES` | Lista de clientes |

### Lógica de Generación

1. Itera sobre cada día desde `2024-01-01` por `n_dias` días
2. Para cada día, calcula factores de estacionalidad:
   - **Factor fin de mes** (1.3×): últimos 3 días del mes
   - **Factor semana** (0.7×): fines de semana (sábado/domingo)
   - **Factor mes** (1.5×): noviembre/diciembre; (1.2×): junio/julio
3. Para cada producto, decide si se vende usando `np.random.beta(2, 5)`
4. Calcula cantidad con `np.random.poisson` modulada por factores
5. Aplica descuento aleatorio con probabilidades sesgadas (50% sin descuento)

### Columnas del DataFrame Resultante

| Columna | Tipo | Rango/Valores | Descripción |
|---------|------|---------------|-------------|
| `fecha` | `datetime64` | 2024-01-01 a 2024-06-28 | Fecha de la venta |
| `sku` | `object` | LAP001, MON001, etc. | Código único del producto |
| `producto` | `object` | "Laptop Pro 15", etc. | Nombre del producto |
| `categoria` | `object` | 9 categorías | Categoría del producto |
| `sucursal` | `object` | 9 sucursales | Sucursal donde se vendió |
| `cliente` | `object` | 12 clientes | Cliente que compró |
| `cantidad` | `int64` | 1-50 | Unidades vendidas |
| `precio_unitario` | `float64` | 120-14,250 | Precio final con descuento |
| `costo_unitario` | `int64` | 150-12,000 | Costo por unidad (sin descuento) |
| `ingreso` | `float64` | 120-712,500 | Ingreso total (precio × cantidad) |
| `costo_total` | `float64` | 150-600,000 | Costo total (costo × cantidad) |
| `margen` | `float64` | -30,000 a 112,500 | Ingreso - Costo total |
| `margen_pct` | `float64` | -50% a 300% | Margen como porcentaje del costo |
| `descuento` | `float64` | 0, 0.05, 0.10, 0.15, 0.20 | Porcentaje de descuento aplicado |
| `dia_semana` | `int64` | 0-6 | Día de la semana (0=lunes) |
| `mes` | `int64` | 1-12 | Mes de la venta |

### Distribución de Descuentos

| Descuento |

**Salida:**

```
# Ver resultado al ejecutar el código
```

**Explicación:**

El código realiza operaciones de configura

**Salida:**

```
# Ver resultado al ejecutar el código
```

**Explicación:**

El código realiza operaciones de configuración/instalación/navegación. Consultar las instrucciones para más detalles.

---
ción/instalación/navegación. Consultar las instrucciones para más detalles.

---
 Probabilidad |
|-----------|-------------|
| 0% (sin descuento) | 50% |
| 5% | 25% |
| 10% | 15% |
| 15% | 7% |
| 20% | 3% |

### Ejemplo de Salida

```python
>>> df = generar_ventas(5)
>>> df[["fecha", "producto", "cantidad", "ingreso", "margen"]].head()

       fecha        producto  cantidad   ingreso   margen
0 2024-01-01  Monitor 27 4K         3  20520.00   4050.00
1 2024-01-01  Teclado Mecánico      1   1330.00    530.00
2 2024-01-01  Mouse Ergonómico      2   1235.00    535.00
3 2024-01-01  SSD 1TB               1   2375.00    875.00
4 2024-01-01  Silla Ergonómica      1   6175.00   2675.00
```

---

## Función: `generar_inventario()`

### Firma

```python
def generar_inventario(productos=None) -> pd.DataFrame
```

### Parámetros

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `productos` | `list[tuple]` | `PRODUCTOS` | Lista de productos |

### Lógica de Generación

1. Para cada producto, genera valores aleatorios realistas:
   - `stock_actual`: entero uniforme 0-200
   - `stock_minimo`: entero uniforme 5-30
   - `stock_maximo`: entero uniforme 50-300
   - `demanda_diaria_prom`: Poisson basada en distribución exponencial

### Columnas del DataFrame Resultante

| Columna | Tipo | Rango/Valores | Descripción |
|---------|------|---------------|-------------|
| `sku` | `object` | LAP001, MUE003, etc. | Código único del producto |
| `producto` | `object` | "Laptop Pro 15", etc. | Nombre del producto |
| `categoria` | `object` | 9 categorías | Categoría del producto |
| `costo` | `int64` | 150-12,000 | Costo unitario |
| `precio` | `int64` | 350-15,000 | Precio de venta |
| `stock_actual` | `int64` | 0-200 | Unidades actuales en inventar

**Salida:**

```
# Ver resultado al ejecutar el código
```

**Explicación:**

El código realiza operaciones de configuración/instalación/navegación

**Salida:**

```
# Ver resultado al ejecutar el código
```

**Explicación:**

El código realiza operaciones de configuración/instalación/navegación. Consultar las instrucciones para más detalles.

---
. Consultar las instrucciones para más detalles.

---
io |
| `stock_minimo` | `int64` | 5-30 | Stock mínimo antes de reordenar |
| `stock_maximo` | `int64` | 50-300 | Capacidad máxima de almacenamiento |
| `demanda_diaria_prom` | `int64` | 0-15 | Demanda diaria promedio estimada |
| `dias_para_agotar` | `float64` | 0.0 - ∞ | stock_actual / demanda_diaria_prom |
| `valor_inventario` | `float64` | 0 - 1,200,000 | stock_actual × costo |
| `necesita_reposicion` | `bool` | True/False | stock_actual < stock_minimo |

### Ejemplo de Salida

```python
>>> df = generar_inventario()
>>> df[["producto", "stock_actual", "stock_minimo", "necesita_reposicion"]].head()

         producto  stock_actual  stock_minimo  necesita_reposicion
0   Laptop Pro 15            67            21                False
1   Laptop Air 13            52            19                False
2   Monitor 27 4K            25            14                False
3   Monitor 24 HD           101            22                False
4  Teclado Mecánico          29            13                False
```

---

## Función: `generar_compras()`

### Firma

```python
def generar_compras(n_ordenes=200, productos=None, proveedores=None) -> pd.DataFrame
```

### Parámetros

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `n_ordenes` | `int` | 200 | Número de órdenes de compra |
| `productos` | `list[tuple]` | `PRODUCTOS` | Catálogo de productos |
| `proveedores` | `list[tuple]` | `PROVEEDORES` | Catálogo de proveedores |

### Lógica de Generación

1. Genera `n_ordenes` órdenes de compra con fechas aleatorias en 2024
2. Para cada orden, selecciona un producto y proveedor aleatorios
3. Calcula cantidad con `np.random.gamma(3, 10)` + 1 (distribución con cola larga)
4. Asigna costo con variación ±15% del costo base
5. 85% de las órdenes se marcan como entregadas; 15% como pendientes
6. Calcula retraso: normal(0, 3) para órdenes entregadas

### Columnas del DataFrame Resultante

| Columna | Tipo | Rango/Valores | Descripción |
|---------|------|---------------|-------------|
| `orden_id` | `object` | OC-00001 a OC-00200 | Identificador único de orden |
| `fecha_orden` | `datetime64` | 2024-01-01 a 2024-12-31 | Fecha en que se realizó la orden |
| `fecha_entrega` | `datetime64` o NaT | 2024-01-02 a 2025-01-15 | Fecha de entrega real (o nula) |
| `proveedor_id` | `object` | PROV001 a PROV007 | ID del proveedor |
| `proveedor` | `object` | "Distribuidora Tecnológica S.A." | Nombre del proveedor |
| `calidad_proveedor` | `int64` | 60-95 | Calificación del proveedor |
| `sku` | `object` | LAP001, AUD0

**Salida:**

```
# Ver resultado al ejecutar el código
```

**Explicación:**

El código realiza operaciones de configuración/instalación/nav

**Salida:**

```
# Ver resultado al ejecutar el código
```

**Explicación:**

El código realiza operaciones de configuración/instalación/navegación. Consultar las instrucciones para más detalles.

---
egación. Consultar las instrucciones para más detalles.

---
03, etc. | SKU del producto ordenado |
| `producto` | `object` | "Laptop Pro 15", etc. | Nombre del producto |
| `categoria` | `object` | 9 categorías | Categoría del producto |
| `cantidad` | `int64` | 1-60+ | Unidades ordenadas |
| `costo_unitario` | `float64` | 128-13,800 | Costo por unidad en la orden |
| `costo_total` | `float64` | 128-828,000 | Costo total de la orden |
| `dias_estimados` | `int64` | 1-10 | Plazo estimado del proveedor |
| `dias_reales` | `int64` o NaN | 1-20+ | Días reales hasta entrega |
| `retraso` | `int64` o NaN | -5 a 15 | Días de retraso (negativo = anticipado) |
| `entregado` | `bool` | True/False | Si la orden fue entregada |
| `puntual` | `bool` | True/False/NaN | Si llegó a tiempo o antes |

### Ejemplo de Salida

```python
>>> df = generar_compras(5)
>>> df[["orden_id", "proveedor", "producto", "cantidad", "entregado", "puntual"]]

   orden_id                          proveedor        producto  cantidad  entregado  puntual
0  OC-00001     Suministros Empresariales C.A.  Webcam HD           7       True     True
1  OC-00002         Mayorista de Tecnología     SSD 1TB           22       True     True
2  OC-00003         TecnoPartes del Sur         Laptop Pro 15      2      False      NaN
3  OC-00004         Distribuidora Tecnológica   Teclado Mecánico   9       True     True
4  OC-00005         Distribuidora Tecnológica   Laptop Air 13     18       True    False
```

---

## Función: `generar_clientes_con_compras()`

### Firma

```python
def generar_clientes_con_compras(n_clientes=200) -> pd.DataFrame
```

### Pa

**Salida:**

```
# Ver resultado al ejecutar el código
```

**Explicación:**

El código realiza operaciones de configuración/instalación/navegación. Consultar las instrucciones para más detalles.

---
rámetros

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `n_clientes` | `int` | 200 | Número de clientes a generar |

### Lógica de Generación

1. Genera `n_clientes` con métricas RFM (Recencia, Frecuencia, Monto)
2. **Recencia**: uniforme 1-365 días 

**Salida:**

```
# Ver resultado al ejecutar el código
```

**Explicación:**

El código realiza operaciones de configuración/instalación/navegación. Consultar las instrucciones para más detalles.

---
desde la última compra
3. **Frecuencia**: exponencial(5) + 1 (# total de compras históricas)
4. **Monto total**: gamma(5, 200) + 100 (gasto total histórico)
5. **Antigüedad**: uniforme 30-2000 días como cliente
6. Segmentación RFM usando cuartiles de recencia

### Columnas del DataFrame Resultante

| Columna | Tipo | Rango/Valores | Descripción |
|---------|------|---------------|-------------|
| `cliente_id` | `object` | C00001 a C00200 | ID único del cliente |
| `recencia_dias` | `int64` | 1-365 | Días desde la última compra |
| `frecuencia_compras` | `int64` | 1-30+ | Número total de compras |
| `monto_total` | `float64` | 120-4,000+ | Gasto total acumulado |
| `ticket_promedio` | `float64` | 20-2,000+ | Monto total / frecuencia |
| `antiguedad_dias` | `int64` | 30-2,000 | Días desde el registro |
| `segmento_rfm` | `category` | 4 niveles | Segmento según recencia (Alto, Medio-Alto, Medio-Bajo, Bajo) |

### Ejemplo de Salida

```python
>>> df = generar_clientes_con_compras(5)
>>> df[["cliente_id", "recencia_dias", "frecuencia_compras", "monto_total", "segmento_rfm"]]

  cliente_id  recencia_dias  frecuencia_compras  monto_total segmento_rfm
0    C00001            257                  10       972.54    Medio-Bajo
1    C00002             12                   5       772.05         Alto
2    C00003            331                   2       433.92         Bajo
3    C00004            127                   7       819.83   Medio-Alto
4    C00005             10                   9      1426.40         Alto
```

### Interpretación del Segmento RFM

| Segmento | Recencia | Significado |
|----------|----------|-------------|
| **Alto** | Menor recencia (compró recientemente) | Cliente activo, leal |
| **Medio-Alto** | Recencia moderada | Cliente regular |
| **Medio-Bajo** | Recencia alta | Cliente en riesgo |
| **Bajo** | Mayor recencia (no compra hace tiempo) | Cliente perdido/ inactivo |

---

## Función: `generar_resenas()`

### Firma

```python
def generar_resenas(n_resenas=100) -> pd.DataFrame
```

### Parámetros

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `n_resenas` | `int` | 100 | Número de reseñas a generar |

### Lógica de Generación

1. Usa una lista de 8 tipos de producto genéricos (no los SKU específicos)
2. Distribución de sentimiento: 60% positivo, 15% negativo, 25% neutro
3. Cada sentimiento tiene textos predefinidos realistas
4. Puntuación: 4-5 para positivo, 1-2 para negativo, 3 

**Salida:**

```
# Ver resultado al ejecutar el código
```

**Explicación:**

El código realiza operaciones de configuración/instalación/navegación. Consultar las instrucciones 

**Salida:**

```
# Ver resultado al ejecutar el código
```

**Explicación:**

El código realiza operaciones de configuración/instalación/navegación. Consultar las instrucciones para más detalles.

---
para más detalles.

---
para neutro
5. Fecha aleatoria entre enero y diciembre 2024

### Textos Predefinidos

| Sentimiento | Textos |
|-------------|--------|
| **Positivo** | "Excelente producto, muy recomendable", "Buena calidad y rápido envío", "Cumple con lo prometido, estoy satisfecho", "Mejor de lo que esperaba, volvería a comprar", "Muy buen rendimiento a este precio" |
| **Negativo** | "No funciona como esperaba, defectuoso", "Mala calidad, se rompió en una semana", "No cumple con las especificaciones", "Pésimo servicio, llegó dañado", "Sobrevalorado, no vale lo que cuesta" |
| **Neutro** | "Cumple su función básica", "Está bien para el 

**Salida:**

```
# Ver resultado al ejecutar el código
```

**Explicación:**

El código realiza operaciones de configuración/instalación/navegación. Consultar las instrucciones para más detalles.

---
p

**Salida:**

```
# Ver resultado al ejecutar el código
```

**Explicación:**

El código realiza operaciones de configuración/instalación/navegación. Consultar las instrucciones para más detalles.

---
recio que tiene", "Podría ser mejor pero es aceptable", "Entrega a tiempo, el producto es normal", "Es lo que esperaba, ni más ni menos" |

### Columnas del DataFrame Resultante

| Columna | Tipo | Rango/Valores | Descripción |
|---------|------|---------------|-------------|
| `reseña_id` | `object` | R00001 a R00100 | ID único de la reseña |
| `producto` | `object` | 8 tipos | Producto reseñado |
| `cliente` | `object` | 12 clientes | Cliente que emitió la reseña |
| `texto` | `object` | Textos predefinidos | Contenido de la reseña |
| `puntuacion` | `int64` | 1-5 | Calificación numérica |

**Salida:**

```
# Ver resultado al ejecutar el código
```

**Explicación:**

El código realiza operaciones de configuración/instalación/navegación. Consultar las instrucciones para más detalles.

---

| `sentimiento` | `object` | positivo, negativo, neutro | Clasificación de sentimiento |
| `fecha` |

**Salida:**

```
# Ver resultado al ejecutar el código
```

**Explicación:**

El código realiza operaciones de configuración/instalación/navegación. Consultar las instrucciones para más detalles.

---
 `datetime64` | 2024-01-01 a 2024-12-31 | Fecha de la reseña |

### Distribución de Sentimiento y Puntuación

| Sentimiento | Probabilidad | Puntuación |
|-------------|-------------|------------|
| Pos

**Salida:**

```
# Ver resultado al ejecutar el código
```

**Explicación:**

El código realiza operaciones de configuración/instalación/navegación. Consultar las instrucciones para más detalles.

---
itivo | 60% | 4-5 |
| Negativo | 15% | 1-2 |
| Neutro | 25% | 3 |

### Ejemplo de Salida

```python
>>> df = generar_resenas(5)
>>> df[["reseña_id", "producto", "puntuacion", "sentimiento", "texto"]]

  reseña_id  producto  puntuacion sentimiento                              

**Salida:**

```
# Ver resultado al ejecutar el código
```

**Explicación:**

El código realiza operaciones de configuración/instalación/navegación. Consultar las instrucciones para más detalles.

---
  texto
0   R00001     SSD            5    positivo     Excelente producto, muy recomendable
1   R00002  Teclado           3      neutro     Cumple su funció

**Salida:**

```
# Ver resultado al ejecutar el código
```

**Explicación:**

El código realiza operaciones de configuración/instalación/navegación. Consultar las instrucciones para más detalles.

---
n básica
2   R00003    Mouse           5    positivo     Buena calidad y rápido envío
3   R00004    Router          2    negativo     No funciona como esperaba, defectuoso
4

**Salida:**

```
# Ver resultado al ejecutar el código
```

**Explicación:**

El código realiza operaciones de configuración/instalación/navegación. Consultar las instrucciones para más detalles.

---
   R00005  Monitor           5    positivo     Mejor de lo que esperaba, volvería a comprar


**Salida:**

```
# Ver resultado al ejecutar el código
```

**Explicación:**

El código realiza operaciones de configuración/instalación/navegación. Consultar las instrucciones para más detalles.

---
```

---

## Ejecución Directa

El módulo puede ejecutarse directamente como script para generar todos los datasets y exportarlos a CSV:

```bash
python datos/datos_sinteticos.py
```

### Flujo de Ejecución

```python
if __name__ == "__main__":
    print("Generando datos sintéticos...")
    ventas = generar

**Salida:**

```
# Ver resultado al ejecutar el código
```

**Explicación:**

El código realiza operaciones de configuración/instalación/navegación. Consultar las instrucciones para más detalles.

---
_ventas(180)
    inventario = generar_inventario()
    compras = generar_compras(200)
    clientes 

**Salida:**

```
# Ver resultado al ejecutar el código
```

**Explicación:**

El código realiza operaciones de configuración/instalación/navegación. Consultar las instrucciones para más detalles.

---
= generar_clientes_con_compras(200)
    resenas = generar_resenas(100)

    ventas.to_csv("datos/ventas.csv", index=False)
    inventario.to_csv("datos/inventario.csv", index=False)
    compras.to_csv("datos/compras.csv", index=False)
    clientes.to_csv("datos/clientes.csv", index=False)
    resenas.to_csv("datos/resenas.csv", index=False)

    print(f"Ventas: {len(ventas)} registros")
    print(f"Inventario: {len(inventario)} productos")
    print(f"Compras: {len(compras)} órdenes")
    print(f"Clientes: {len(clientes)} clientes")
    print(f"Reseñas: {len(resenas)} reseñas")
    print("¡Datos generados exitosamente!")
```

### Salida Esperada

```
Generando datos sintéticos...
Ventas: 1330 registros
Inventario: 25 productos
Compras: 200 órdenes
Clientes: 200 clientes
Reseñas: 100 reseñas
¡Datos generados exitosamente!
```

### Archivos Generados

| Archivo | Ruta | Tamaño aprox. |
|---------|------|---------------|
| `ventas.csv` | `datos/ventas.csv` | ~120 KB |
| `inventario.csv` | `datos/inventario.csv` | ~2 KB |
| `compras.csv` | `datos/compras.csv` | ~25 KB |
| `clientes.csv` | `datos/clientes.csv` | ~15 KB |
| `resenas.csv` | `datos/resenas.csv` | ~8 KB |

---

## Guía de Uso

### Importación del Módulo

```python
# Opción 1: Importar funciones individuales
import sys
sys.path.append("..")  # si ejecutas desde 00-base/
fr

**Salida:**

```
# Ver resultado al ejecutar el código
```

**Explicación:**

El código realiza operaciones de configuración/instalación/navegación. Consultar las instrucciones para más detalles.

---
om datos.datos_sinteticos import (
    generar_ventas,
    generar_inventario,
    generar_compras,
    generar_clientes_con_compras,
    generar_resenas,
    PRODUCTOS,
    PROVEEDORES,
    SUCURSALES,
    CLIENTES,
)

# Opción 2: Importar todo
from datos.datos_sinteticos import *
```

### Carga de CSVs (alternativa)

```python
import pandas as pd
ventas = pd.read_csv("../datos/ventas.csv")
inventario = pd.read_csv("../datos/inventario.csv")
compras = pd.read_csv("../datos/compr

**Salida:**

```
# Ver resultado al ejecutar el código
```

**Explicación:**

El código realiza operaciones de configuración/instalación/navegación. Consultar las instrucciones para más detalles.

---
as.csv")
clientes = pd.read_csv("../datos/clientes.csv")
resenas = pd.read_csv("../datos/resenas.csv")
```

### Generación Personalizada

```python
# Generar solo 30 días de ventas
ventas_rapidas = generar_ventas(30)

# Generar ventas solo para ciertos productos
mis_productos = PRODUCTOS[:5]  # Solo laptops y monitores
ventas_filtradas = generar_ventas(90, productos=mis_productos)

# Generar compras con más órdenes
comps = generar_compras(500)

# Generar más reseñas para NLP
res = generar_resenas(1000)
```

### Parámetros Avanzados

```python
# Personalizar sucursales
sucs = ["Matriz CDMX", "Sucursal Monterrey"]
ventas_regionales = generar_ventas(180, sucursales=sucs)

# Personalizar clientes
cls = ["Cliente Premium", "Empresa XYZ", "Mayorista JJ"]
ventas_clientes = generar_ventas(180, clientes=cls)

# Combinar personalizaciones
ventas_custom = generar_ventas(
    n_dias=365,
    productos=PRODUCTOS[:10],
    sucursales=["Matriz CDMX"],
    clientes=["Cliente Premium"],
)
```

---

## Ejemplos de Carga

### Ejemplo 1: Carga Básica y Primer Vistazo

```python
import pandas as pd
import sys
sys.path.append("..")
from datos.datos_sinteticos import generar_ventas, generar_inventario

ventas = generar_ventas()
inventario = generar_inventario()

print(ventas.shape)
print(ventas.dtypes)
print(ventas.describe())
print(ventas.head())
```

### Ejemplo 2: Combinar Datasets

```python
# Cargar todos los datasets
ventas = generar_ventas()
inventario = generar_inventario()
compras = generar_compras()

# Merge de ventas con costos de inventario
ventas_con_costos = ventas.merge(
    inventario[["sku", "costo", "precio"]],
    on="sku",
    how="left"
)

# Verificar consistencia
print(ventas_con_costos.head())
```

### Ejemplo 3: Análisis Rápido de Ventas por Categoría

```python
ventas = generar_ventas()
ventas_por_categoria = (
    ventas.groupby("categoria")
    .agg({"ingreso": "sum", "cantidad": "sum", "margen": "mean"})
    .sort_values("ingreso", ascending=False)
)
print(ventas_por_categoria)
```

### Ejemplo 4: Usar Datos para ML

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

clientes = generar_clientes_con_compras(200)

X = clientes[["recencia_dias", "frecuencia_compras", "antiguedad_dias"]]
y = clientes["monto_total"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor()
model.fit(X_train, y_train)
print(f"Score: {model.score(X_test, y_test):.3f}")
```

### Ejemplo 5: Análisis de Sentimiento

```python
from textblob import TextBlob

resenas = generar_resenas(100)
for _, row in resenas.head().iterrows():
    blob = TextBlob(row["texto"])
    print(f"{row['sentimiento']:10s} | {row['puntuacion']} | {blob.sentiment.polarity:.2f} | {row['texto']}")
```

---

## Consideraciones sobre los Datos

### Sesgos Incorporados

Los datos incluyen sesgos realistas para practicar detección y corrección:

| Sesgo | Dataset | Descripción |
|-------|---------|-------------|
| Estacionalidad | Ventas | Más ventas en fines de mes y temporada navideña |
| Fines de semana bajos | Ventas | 30% menos ventas en sábado/domingo |
| Descuentos | Ventas | 50% de transacciones tienen algún descuento |
| Órdenes pendientes | Compras | 15% de órdenes no han sido entregadas |
| Retrasos | Compras | Algunos proveedores son menos puntuales |
| Calidad de proveedor | Compras | Varía de 60% a 95% |
| Segmentación RFM | Clientes | Clientes con diferentes comportamientos de compra |
| Balance de sentimiento | Reseñas | 60% positivas, 15% negativas, 25% neutras |

### Valores Atípicos y Nulos

| Condición | Dónde aparece |
|-----------|--------------|
| `fecha_entrega` nula | Compras no entregadas (15%) |
| `dias_reales` nulo | Compras no entregadas |
| `retraso` nulo | Compras no entregadas |
| `puntual` NaN | Compras no entregadas |
| Ventas con margen negativo | Cuando el descuento es alto y el costo supera al ingreso |
| Stock en cero | Algunos productos pueden tener inventario agotado |

### Relaciones entre Datasets

```
PRODUCTOS ───────┬──────► ventas.sku
                 ├──────► inventario.sku
                 └──────► compras.sku

PROVEEDORES ───────────► compras.proveedor_id

SUCURSALES ────────────► ventas.sucursal

CLIENTES ──────────────► ventas.cliente
                   ────► reseñas.cliente
```

### Limitaciones

1. Los datos son completamente sintéticos: no representan patrones reales de consumo
2. No hay relaciones de llave foránea explícitas entre todos los datasets
3. Las reseñas usan textos fijos, no generación de lenguaje natural
4. Los clientes de `generar_clientes_con_compras()` son independientes de los clientes en `CLIENTES`
5. El inventario no se actualiza dinámicamente con las ventas (es una foto fija)

### Reproducibilidad

La semilla `np.random.seed(42)` al inicio del módulo asegura que los datos sean reproducibles en cada ejecución. Para obtener datos diferentes:

```python
import numpy as np
np.random.seed(123)  # Cambia la semilla antes de generar
```

---

*Volver al [Índice Maestro](00-INDEX.md) | [README principal](00-README.md)*
