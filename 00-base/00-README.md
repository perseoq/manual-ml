# Manual Completo de Ciencia de Datos, NLP, Machine Learning y Deep Learning con Python

## Aplicado al Sector Ventas, Compras e Inventarios

---

### 📘 Presentación

Este manual es un recurso integral diseñado para llevar a cualquier persona — desde un **novato absoluto** hasta un **científico de datos experto** — a dominar las herramientas fundamentales y avanzadas de Ciencia de Datos, NLP, Machine Learning y Deep Learning con Python.

Cada concepto se explica con un enfoque **100% práctico y aplicado**, utilizando datos sintéticos realistas del sector de **Ventas, Compras e Inventarios**. Todos los ejemplos son ejecutables y están diseñados para resolverse en un cuaderno Jupyter o script de Python.

---

### 🎯 Objetivos

- Proveer una **ruta de aprendizaje progresiva** desde cero hasta nivel experto
- Cubrir **todas las librerías esenciales**: NumPy, Pandas, Seaborn, SciPy, Scikit-learn, NLTK, spaCy, TextBlob, Gensim, TensorFlow, PyTorch, HuggingFace Transformers, Prophet, SHAP, LangChain, FAISS, MLflow, FastAPI
- Ofrecer **ejemplos concretos y ejecutables** (15-20 por archivo) aplicados al negocio
- Incluir **casos prácticos reales** de ventas, compras e inventarios
- Proporcionar **ejercicios progresivos** con soluciones para autoevaluación

---

### 👥 Audiencia

| Perfil | Ruta recomendada |
|---|---|
| **Novato** (sin experiencia en programación) | Comenzar en Básico, archivos B01-B08 |
| **Analista de datos** (conoce Excel, quiere Python) | Básico completo → Intermedio |
| **Ingeniero de ML** (quiere profundizar) | Intermedio → Avanzado |
| **Científico de datos** (busca producción) | Avanzado → Experto |

---

### 📚 Estructura del Manual

```
manual-ds-ml-ventas/
├── 00-base/           # README, Índice, Navegación, Convenciones
├── 01-basico/         # Fundamentos: Python DS, NumPy, Pandas, Seaborn, SciPy, Sklearn
├── 02-intermedio/     # Análisis avanzado, ML clásico, Clustering, NLP intro
├── 03-avanzado/       # ML avanzado, Feature Eng, NLP, Deep Learning (TF + PyTorch)
├── 04-experto/        # Transformers, Series Temporales, RecSys, Producción
├── 05-apendices/      # Cheatsheets, Glosario, Recursos
└── datos/             # Datos sintéticos generados
```

**Salida:**

```
# Ver resultado al ejecutar el código
```

**Explicación:**

El código realiza operaciones de configuración/instalación/navegación. Consultar las instrucciones para más detalles.

---


---

### 🧰 Librerías Cubiertas

| Nivel | Librerías |
|---|---|
| **Básico** | Python stdlib, NumPy, Pandas, Seaborn, SciPy, Scikit-learn |
| **Intermedio** | Avanzado NumPy/Pandas/Seaborn/SciPy, Scikit-learn ML, NLTK, spaCy, TextBlob |
| **Avanzado** | Scikit-learn pipelines/ensemble, Feature engineering, Gensim, TensorFlow, PyTorch |
| **Experto** | HuggingFace, Prophet, SHAP, LangChain, FAISS, MLflow, FastAPI, Optuna |

---

### 📊 Datos de Ejemplo

Todos los ejemplos usan datos generados por `datos/datos_sinteticos.py`:

| Dataset | Registros | Columnas |
|---|---|---|
| `ventas.csv` | ~1,330 | 15 (fecha, producto, sucursal, cantidad, ingreso, margen, etc.) |
| `inventario.csv` | 25 | 11 (stock, costos, reposición, etc.) |
| `compras.csv` | 200 | 16 (órdenes, proveedores, costos, retrasos) |
| `clientes.csv` | 200 | 7 (recencia, frecuencia, monto, segmento) |
| `resenas.csv` | 100 | 7 (texto, puntuación, sentimiento) |

---

### 🚀 Cómo Usar Este Manual

1. **Lee en orden secuencial** si eres principiante (Básico → Intermedio → Avanzado → Experto)
2. **Salta directamente** a secciones específicas si ya tienes experiencia
3. **Ejecuta cada ejemplo** en tu entorno local; modifica parámetros para experimentar
4. **Completa los ejercicios** al final de cada archivo
5. **Usa las cheatsheets** del apéndice como referencia rápida

---

### 🛠️ Configuración del Entorno

Ver `00-INSTALACION.md` para instrucciones detalladas de instalación.

```bash
python -m venv venv
source venv/bin/activate  # 

**Salida:**

```
# Ver resultado al ejecutar el código
```

**Explicación:**

El código realiza operaciones de configuración/instalación/navegación. Consultar las instrucciones para más detalles.

---
o venv\Scripts\activate en Windows
pip install numpy pandas seaborn scipy scikit-learn nltk spacy textblob gensim tensorflow torch transformers prophet shap langchain faiss-cpu mlflow fastapi uvicorn
```

---

### 📖 Convenciones de Este Manual

Ver `00-CONVENCIONES.md` para el estilo de código, formato de ejemplos y estructura de cada archivo.

---

**¡Comienza tu viaje en Ciencia de Datos aplicada al negocio!**
