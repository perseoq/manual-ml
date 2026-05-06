# Convenciones del Manual

## Estructura de Cada Archivo

Cada archivo `.md` del manual sigue esta estructura:

```
# Título del Tema

## 🎯 Objetivos de Aprendizaje
- Objetivo 1
- Objetivo 2

## 📖 Explicación Teórica
Breve introducción conceptual con lenguaje claro y metáforas del negocio.

## 💻 Ejemplo 1: [Nombre del ejemplo]
Código ejecutable:
```python
# Código con comentarios
```

**Salida:**
```
Resultado esperado
```

**Explicación línea por línea:**
- Línea 1: qué hace
- Línea 2: qué hace
...

## 💻 Ejemplo 2: [Nombre del ejemplo]
... (15-20 ejemplos por archivo)

## 📝 Ejercicios Propuestos
1. Ejercicio 1 (con pista)
2. Ejercicio 2 (con pista)
...

## 📌 Resumen
- Punto clave 1
- Punto clave 2

## 🔗 Enlaces Relacionados
- [Archivo anterior](ruta)
- [Archivo siguiente](ruta)
```

## Estilo de Código

- `import numpy as np`, `import pandas as pd`, etc. (convenciones estándar)
- Nombres de variables en español (relacionados a ventas/compras/inventarios)
- Comentarios EXPLICATIVOS en cada ejemplo
- `print()` para mostrar resultados
- Compatible con Python 3.9+

## Datos

- Todos los ejemplos usan datos del módulo `datos/datos_sinteticos.py`
- Importar con: `import sys; sys.path.append(".."); from datos.datos_sinteticos import *`
- O cargar CSVs: `pd.read_csv("../datos/ventas.csv")`

## Convenciones de Rutas

- Las rutas relativas asumen ejecución desde la carpeta del nivel correspondiente
- Ejemplo: si estás en `01-basico/`, los datos están en `../datos/ventas.csv`
