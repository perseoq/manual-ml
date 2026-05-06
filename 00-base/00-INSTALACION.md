# Instalación y Configuración del Entorno

## Requisitos Mínimos

- **Python:** 3.9 o superior
- **RAM:** 8 GB mínimo (16 GB recomendado para Deep Learning)
- **Disco:** 5 GB libres
- **GPU:** Opcional (CUDA para TensorFlow/PyTorch)

## Instalación Paso a Paso

### 1. Verificar Python

```bash
python --version
# Python 3.9.0 o superior
```

**Salida:**

```
# Ver resultado al ejecutar el código
```

**Explicación:**

El código realiza operaciones de configuración/instalación/navegación. Co

**Salida:**

```
# Ver resultado al ejecutar el código
```

**Explicación:**

El código realiza operaciones de configu

**Salida:**

```
# Ver resultado al ejecutar el código
```

**Explicación:**

El código realiza operaciones de configuración/instalación/navegación. Consultar las instrucciones para m

**Salida:**

```
# Ver resultado al ejecutar el código
```

**Explicación:**

El có

**Salida:**

```
# Ver resultado al ejecutar el código
```

**Explicación:**

El código realiza operaciones de configuración/instalación/navegación. Consultar las instrucciones para más detalles.

---
digo realiza operaciones de configuración/instalación/navegación

**Salida:**

```
# Ver resultado al ejecutar el código
```

**Explicación:**

El código realiza operaciones de configuración/instalación/navegación. Consultar las instrucciones para más detalles.

---
. Consultar las instrucciones para más detalles.

---
ás detalles.

---
ración/instalación/navegación. Consultar las instrucciones para más detalles.

---
nsultar las instrucciones para más detalles.

---


### 2. Crear 

**Salida:**

```
# Ver resultado al ejecutar el código
```

**Explicación:**

El código realiza operaciones de configuración/instalación/navegación. Consultar las instrucciones para más detalles.

---
Entorno Virtual

```bash
# Linux/Mac
python -m venv venv
source venv/bin/activate



**Salida:**

```
# Ver resultado al ejecutar el código
```

**Explicación:**

El código realiza operaciones de configuración/instalación/navegación. Consultar las instrucciones para más detalles.

---
# Windows


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
python -m venv venv
venv\Scripts\activate
```

### 3. Instalar Librerías por Nivel

#### Nivel Básico
```bash
pip install numpy pandas seaborn scipy scikit-learn
```

#### Nivel Intermedio
```bash
pip install nltk spacy textblob gensim
python -m spacy download es_core_news_sm
python -m spacy download en_core_web_sm
python -m nltk.downloader all
```

#### Nivel Avanzado
```bash
pip install tensorflow torch torchvision torchaudio
```

#### Nivel Experto
```bash
pip install transformers datasets sentence-transformers
pip install prophet pmdarima shap lime
pip install langchain faiss-cpu chromadb
pip install mlflow fastapi uvicorn optuna hyperopt
pip install evidently xgboost lightgbm catboost
```

### 4. Instalación Todo-en-Uno

```bash
pip install numpy pandas seaborn scipy scikit-learn nltk spacy textblob gensim \
            tensorflow torch torchvision torchaudio \
            transformers datasets sentence-transformers \
            prophet pmdarima shap lime \
            langchain faiss-cpu chromadb \
            mlflow fastapi uvicorn optuna hyperopt \
            evidently xgboost lightgbm catboost
```

### 5. Verificar Instalación

```python
import numpy as np
import pandas as pd
import seaborn as sns
import scipy
import sklearn
import nltk
import spacy
import gensim
import tensorflow as tf
import torch
import transformers
print("¡Todas las librerías instaladas correctamente!")
```

## CUDA (Opcional, para GPU)

```bash
# Verificar CUDA disponible
python -c "import torch; print(torch.cuda.is_available())"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Jupyter Notebook (Recomendado)

```bash
pip install jupyter ipykernel
python -m ipykernel install --user --name=venv
jupyter notebook
```
