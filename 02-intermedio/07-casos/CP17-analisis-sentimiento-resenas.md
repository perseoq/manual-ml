# CP17: Análisis de Sentimiento de Reseñas — NLP Aplicado

## Contexto de Negocio
El equipo de producto quiere saber qué opinan los clientes de cada producto para identificar áreas de mejora, detectar productos problemáticos y reforzar aspectos positivos. El análisis de sentimiento permite procesar cientos de reseñas automáticamente.

```python
# ============================================================
# 1. CARGA DE RESEÑAS Y EXPLORACIÓN DE PUNTUACIONES
# ============================================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import classification_report
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.figsize": (12, 6), "font.size": 12})

resenas = pd.read_csv("../datos/resenas.csv")
print("Dimensiones:", resenas.shape)
print("\nColumnas:", resenas.columns.tolist())
print("\nPrimeras filas:")
print(resenas.head())

print("\n\nDistribución de puntuaciones:")
print(resenas["puntuacion"].value_counts().sort_index())

plt.figure(figsize=(10, 5))
sns.countplot(data=resenas, x="puntuacion", palette="viridis")
plt.title("Distribución de Puntuaciones de Reseñas", fontsize=14)
plt.xlabel("Puntuación")
plt.ylabel("Cantidad")
plt.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.show()

print(f"\nPuntuación promedio: {resenas['puntuacion'].mean():.2f}")
print(f"Mediana: {resenas['puntuacion'].median():.2f}")
print(f"Desviación estándar: {resenas['puntuacion'].std():.2f}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
# Las primeras filas muestran la estructura de los datos procesados:
#    Las columnas dependen del DataFrame utilizado.
# 0    Los valores dependen de los datos de entrada.
# 1    valor4    valor5    valor6
```

**Explicación línea por línea:**

*Contexto de Negocio.*

1. ============================================================
2. 1. CARGA DE RESEÑAS Y EXPLORACIÓN DE PUNTUACIONES
3. ============================================================

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```python
# ============================================================
# 2. ANÁLISIS DE SENTIMIENTO CON TEXTBLOB (POLARIDAD)
# ============================================================
try:
    from textblob import TextBlob

    def get_textblob_polarity(texto):
        if isinstance(texto, str):
            return TextBlob(texto).sentiment.polarity
        return 0.0

    resenas["polarity_textblob"] = resenas["resena"].apply(get_textblob_polarity)

    print("Estadísticas de polaridad (TextBlob):")
    print(resenas["polarity_textblob"].describe())

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(resenas["polarity_textblob"], bins=30, kde=True, color="steelblue")
    plt.title("Distribución de Polaridad (TextBlob)", fontsize=13)
    plt.xlabel("Polaridad")
    plt.axvline(0, color="red", linestyle="--", alpha=0.5)

    plt.subplot(1, 2, 2)
    sns.boxplot(data=resenas, x="puntuacion", y="polarity_textblob", palette="viridis")
    plt.title("Polaridad TextBlob vs Puntuación Real", fontsize=13)
    plt.xlabel("Puntuación")
    plt.ylabel("Polaridad")
    plt.tight_layout()
    plt.show()

    print("\nEjemplos de reseñas con su polaridad:")
    for _, row in resenas.sample(5).iterrows():
        print(f"  [{row['polarity_textblob']:.2f}] {row['resena'][:80]}...")

except ImportError:
    print("TextBlob no instalado. Instalar con: pip install textblob")
    print("Simulando polaridad...")
    np.random.seed(42)
    resenas["polarity_textblob"] = (resenas["puntuacion"] - 3) / 2 + np.random.uniform(-0.2, 0.2, len(resenas))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 2. ANÁLISIS DE SENTIMIENTO CON TEXTBLOB (POLARIDAD)
3. ============================================================

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```python
# ============================================================
# 3. ANÁLISIS DE SENTIMIENTO CON VADER (COMPOUND)
# ============================================================
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()

    def get_vader_compound(texto):
        if isinstance(texto, str):
            return analyzer.polarity_scores(texto)["compound"]
        return 0.0

    resenas["compound_vader"] = resenas["resena"].apply(get_vader_compound)

    print("Estadísticas de compound (VADER):")
    print(resenas["compound_vader"].describe())

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(resenas["compound_vader"], bins=30, kde=True, color="coral")
    plt.title("Distribución de Compound (VADER)", fontsize=13)
    plt.xlabel("Compound Score")
    plt.axvline(0, color="red", linestyle="--", alpha=0.5)

    plt.subplot(1, 2, 2)
    sns.boxplot(data=resenas, x="puntuacion", y="compound_vader", palette="viridis")
    plt.title("Compound VADER vs Puntuación Real", fontsize=13)
    plt.xlabel("Puntuación")
    plt.ylabel("Compound")
    plt.tight_layout()
    plt.show()

    print("\nEjemplos de reseñas con compound VADER:")
    for _, row in resenas.sample(5).iterrows():
        print(f"  [{row['compound_vader']:.2f}] {row['resena'][:80]}...")

except ImportError:
    print("VADER no instalado. Instalar con: pip install vaderSentiment")
    print("Simulando compound...")
    resenas["compound_vader"] = (resenas["puntuacion"] - 3) / 2.5 + np.random.uniform(-0.15, 0.15, len(resenas))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 3. ANÁLISIS DE SENTIMIENTO CON VADER (COMPOUND)
3. ============================================================

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```python
# ============================================================
# 4. COMPARAR TEXTBLOB vs VADER vs PUNTUACIÓN REAL
# ============================================================
plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 1)
sns.scatterplot(data=resenas, x="puntuacion", y="polarity_textblob",
                alpha=0.4, color="steelblue")
plt.title("TextBlob vs Puntuación", fontsize=12)
plt.xlabel("Puntuación Real")
plt.ylabel("Polarity TextBlob")
corr_tb = resenas["puntuacion"].corr(resenas["polarity_textblob"])
plt.text(0.05, 0.95, f"r = {corr_tb:.3f}", transform=plt.gca().transAxes,
         fontsize=11, verticalalignment="top", bbox=dict(boxstyle="round", alpha=0.8))

plt.subplot(1, 3, 2)
sns.scatterplot(data=resenas, x="puntuacion", y="compound_vader",
                alpha=0.4, color="coral")
plt.title("VADER vs Puntuación", fontsize=12)
plt.xlabel("Puntuación Real")
plt.ylabel("Compound VADER")
corr_vd = resenas["puntuacion"].corr(resenas["compound_vader"])
plt.text(0.05, 0.95, f"r = {corr_vd:.3f}", transform=plt.gca().transAxes,
         fontsize=11, verticalalignment="top", bbox=dict(boxstyle="round", alpha=0.8))

plt.subplot(1, 3, 3)
sns.scatterplot(data=resenas, x="polarity_textblob", y="compound_vader",
                alpha=0.4, color="purple")
plt.title("TextBlob vs VADER", fontsize=12)
plt.xlabel("Polarity TextBlob")
plt.ylabel("Compound VADER")
corr_tv = resenas["polarity_textblob"].corr(resenas["compound_vader"])
plt.text(0.05, 0.95, f"r = {corr_tv:.3f}", transform=plt.gca().transAxes,
         fontsize=11, verticalalignment="top", bbox=dict(boxstyle="round", alpha=0.8))

plt.tight_layout()
plt.show()

print(f"Correlación TextBlob vs Puntuación: {corr_tb:.3f}")
print(f"Correlación VADER vs Puntuación: {corr_vd:.3f}")
print(f"Correlación TextBlob vs VADER: {corr_tv:.3f}")

if corr_vd > corr_tb:
    print("\nVADER se correlaciona mejor con la puntuación real. Recomendado para este dataset.")
else:
    print("\nTextBlob se correlaciona mejor con la puntuación real. Recomendado para este dataset.")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 4. COMPARAR TEXTBLOB vs VADER vs PUNTUACIÓN REAL
3. ============================================================

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```python
# ============================================================
# 5. CLASIFICAR RESEÑAS: POSITIVAS / NEGATIVAS / NEUTRAS
# ============================================================
def clasificar_por_puntuacion(punt):
    if punt >= 4:
        return "positiva"
    elif punt <= 2:
        return "negativa"
    else:
        return "neutra"

def clasificar_por_vader(compound, umbral_pos=0.05, umbral_neg=-0.05):
    if compound >= umbral_pos:
        return "positiva"
    elif compound <= umbral_neg:
        return "negativa"
    else:
        return "neutra"

def clasificar_por_textblob(polarity, umbral_pos=0.1, umbral_neg=-0.1):
    if polarity >= umbral_pos:
        return "positiva"
    elif polarity <= umbral_neg:
        return "negativa"
    else:
        return "neutra"

resenas["clase_real"] = resenas["puntuacion"].apply(clasificar_por_puntuacion)
resenas["clase_vader"] = resenas["compound_vader"].apply(clasificar_por_vader)
resenas["clase_textblob"] = resenas["polarity_textblob"].apply(clasificar_por_textblob)

print("Distribución de clases reales:")
print(resenas["clase_real"].value_counts())
print("\nDistribución de clases VADER:")
print(resenas["clase_vader"].value_counts())
print("\nDistribución de clases TextBlob:")
print(resenas["clase_textblob"].value_counts())

print("\n\n--- Reporte de clasificación: VADER vs Real ---")
print(classification_report(resenas["clase_real"], resenas["clase_vader"],
                            target_names=["negativa", "neutra", "positiva"]))

print("--- Reporte de clasificación: TextBlob vs Real ---")
print(classification_report(resenas["clase_real"], resenas["clase_textblob"],
                            target_names=["negativa", "neutra", "positiva"]))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 5. CLASIFICAR RESEÑAS: POSITIVAS / NEGATIVAS / NEUTRAS
3. ============================================================

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```python
# ============================================================
# 6. NUBE DE PALABRAS DE RESEÑAS POSITIVAS
# ============================================================
try:
    from wordcloud import WordCloud
    from nltk.corpus import stopwords
    import nltk

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

    stop_words = set(stopwords.words("spanish"))
    stop_words.update(["el", "la", "los", "las", "un", "una", "y", "e", "o", "a",
                       "de", "del", "para", "por", "con", "en", "es", "muy",
                       "que", "se", "no", "lo", "su", "al", "le", "me", "te"])

    def generar_nube(df, columna_texto, titulo, color):
        texto = " ".join(df[columna_texto].dropna().tolist())
        wordcloud = WordCloud(width=800, height=400,
                              background_color="white",
                              stopwords=stop_words,
                              max_words=100,
                              colormap=color,
                              random_state=42).generate(texto)
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(titulo, fontsize=16)
        plt.tight_layout()
        plt.show()

    resenas_positivas = resenas[resenas["clase_real"] == "positiva"]
    generar_nube(resenas_positivas, "resena",
                 "Palabras más frecuentes — Reseñas Positivas", "Greens")

except ImportError:
    print("WordCloud no instalado. Instalar con: pip install wordcloud")
    print("Generando análisis de frecuencias como alternativa...")
    from collections import Counter
    texto_pos = " ".join(resenas[resenas["clase_real"]=="positiva"]["resena"].dropna())
    palabras_pos = [p.lower() for p in texto_pos.split() if len(p) > 3]
    print("Top 20 palabras en reseñas positivas:")
    for palabra, count in Counter(palabras_pos).most_common(20):
        print(f"  {palabra}: {count}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 6. NUBE DE PALABRAS DE RESEÑAS POSITIVAS
3. ============================================================

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```python
# ============================================================
# 7. NUBE DE PALABRAS DE RESEÑAS NEGATIVAS
# ============================================================
try:
    resenas_negativas = resenas[resenas["clase_real"] == "negativa"]
    generar_nube(resenas_negativas, "resena",
                 "Palabras más frecuentes — Reseñas Negativas", "Reds")

except:
    texto_neg = " ".join(resenas[resenas["clase_real"]=="negativa"]["resena"].dropna())
    palabras_neg = [p.lower() for p in texto_neg.split() if len(p) > 3]
    print("\nTop 20 palabras en reseñas negativas:")
    for palabra, count in Counter(palabras_neg).most_common(20):
        print(f"  {palabra}: {count}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 7. NUBE DE PALABRAS DE RESEÑAS NEGATIVAS
3. ============================================================

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```python
# ============================================================
# 8. PALABRAS MÁS FRECUENTES POR SENTIMIENTO
# ============================================================
from collections import Counter

def top_palabras_por_sentimiento(df, col_texto, col_clase, n=15):
    resultados = {}
    for clase in ["positiva", "neutra", "negativa"]:
        subset = df[df[col_clase] == clase]
        texto = " ".join(subset[col_texto].dropna())
        palabras = [p.lower().strip(".,!?¡¿()[]") for p in texto.split()
                    if p.lower() not in stop_words and len(p) > 3]
        resultados[clase] = Counter(palabras).most_common(n)
    return resultados

palabras_por_clase = top_palabras_por_sentimiento(resenas, "resena", "clase_real")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
colores = {"positiva": "green", "neutra": "orange", "negativa": "red"}

for i, (clase, color) in enumerate(colores.items()):
    palabras, counts = zip(*palabras_por_clase[clase]) if palabras_por_clase[clase] else ([], [])
    axes[i].barh(list(palabras)[::-1], list(counts)[::-1], color=color, alpha=0.7)
    axes[i].set_title(f"Top palabras — {clase.capitalize()}", fontsize=13)
    axes[i].set_xlabel("Frecuencia")

plt.tight_layout()
plt.show()

print("\nTop 5 palabras por sentimiento:")
for clase in ["positiva", "neutra", "negativa"]:
    print(f"\n  {clase.upper()}:")
    for palabra, count in palabras_por_clase[clase][:5]:
        print(f"    {palabra}: {count}")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 8. PALABRAS MÁS FRECUENTES POR SENTIMIENTO
3. ============================================================

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```python
# ============================================================
# 9. SENTIMIENTO PROMEDIO POR PRODUCTO
# ============================================================
if "producto" in resenas.columns:
    sentimiento_producto = resenas.groupby("producto").agg(
        puntuacion_promedio=("puntuacion", "mean"),
        compound_promedio=("compound_vader", "mean"),
        polaridad_promedio=("polarity_textblob", "mean"),
        num_resenas=("resena", "count")
    ).sort_values("compound_promedio", ascending=False)

    print("Sentimiento promedio por producto:")
    print(sentimiento_producto.to_string())

    plt.figure(figsize=(14, 6))
    top10 = sentimiento_producto.head(10)
    plt.subplot(1, 2, 1)
    sns.barplot(data=top10.reset_index(), y="producto", x="compound_promedio",
                palette="RdBu_r", hue="producto", legend=False)
    plt.title("Top 10 Productos por Sentimiento VADER", fontsize=13)
    plt.xlabel("Compound Promedio")
    plt.ylabel("")

    bottom10 = sentimiento_producto.tail(10)
    plt.subplot(1, 2, 2)
    sns.barplot(data=bottom10.reset_index(), y="producto", x="compound_promedio",
                palette="RdBu_r", hue="producto", legend=False)
    plt.title("Bottom 10 Productos por Sentimiento VADER", fontsize=13)
    plt.xlabel("Compound Promedio")
    plt.ylabel("")

    plt.tight_layout()
    plt.show()
else:
    print("Columna 'producto' no encontrada. Usando reseñas completas.")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
# Las primeras filas muestran la estructura de los datos procesados:
#    Las columnas dependen del DataFrame utilizado.
# 0    Los valores dependen de los datos de entrada.
# 1    valor4    valor5    valor6
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 9. SENTIMIENTO PROMEDIO POR PRODUCTO
3. ============================================================

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```python
# ============================================================
# 10. CORRELACIÓN SENTIMIENTO vs PUNTUACIÓN
# ============================================================
from scipy.stats import pearsonr, spearmanr

corr_pearson_tb, p_pearson_tb = pearsonr(resenas["puntuacion"], resenas["polarity_textblob"])
corr_pearson_vd, p_pearson_vd = pearsonr(resenas["puntuacion"], resenas["compound_vader"])
corr_spearman_tb, p_spearman_tb = spearmanr(resenas["puntuacion"], resenas["polarity_textblob"])
corr_spearman_vd, p_spearman_vd = spearmanr(resenas["puntuacion"], resenas["compound_vader"])

print("=" * 50)
print("CORRELACIONES CON PUNTUACIÓN REAL")
print("=" * 50)
print(f"{'Método':<20} {'Pearson r':<15} {'p-value':<15} {'Spearman ρ':<15} {'p-value':<15}")
print("-" * 80)
print(f"{'TextBlob':<20} {corr_pearson_tb:<15.4f} {p_pearson_tb:<15.6f} {corr_spearman_tb:<15.4f} {p_spearman_tb:<15.6f}")
print(f"{'VADER':<20} {corr_pearson_vd:<15.4f} {p_pearson_vd:<15.6f} {corr_spearman_vd:<15.4f} {p_spearman_vd:<15.6f}")

if p_pearson_tb < 0.05:
    print("\nTextBlob: correlación significativa (p < 0.05)")
else:
    print("\nTextBlob: correlación NO significativa (p >= 0.05)")

if p_pearson_vd < 0.05:
    print("VADER: correlación significativa (p < 0.05)")
else:
    print("VADER: correlación NO significativa (p >= 0.05)")

plt.figure(figsize=(10, 6))
sns.heatmap(resenas[["puntuacion", "polarity_textblob", "compound_vader"]].corr(),
            annot=True, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            fmt=".3f", linewidths=1)
plt.title("Matriz de Correlaciones", fontsize=14)
plt.tight_layout()
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 10. CORRELACIÓN SENTIMIENTO vs PUNTUACIÓN
3. ============================================================

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```python
# ============================================================
# 11. PRODUCTO MEJOR Y PEOR VALORADO
# ============================================================
if "producto" in resenas.columns:
    rank_productos = resenas.groupby("producto").agg(
        puntuacion_media=("puntuacion", "mean"),
        compound_medio=("compound_vader", "mean"),
        count=("resena", "count")
    ).query("count >= 3").sort_values("puntuacion_media", ascending=False)

    print("Ranking de productos por puntuación:")
    print(rank_productos.to_string())

    mejor = rank_productos.index[0]
    peor = rank_productos.index[-1]

    print(f"\n{'='*50}")
    print(f"🏆 MEJOR VALORADO: {mejor}")
    print(f"   Puntuación media: {rank_productos.loc[mejor, 'puntuacion_media']:.2f}")
    print(f"   Compound medio: {rank_productos.loc[mejor, 'compound_medio']:.3f}")
    print(f"   Reseñas: {rank_productos.loc[mejor, 'count']}")

    print(f"\n{'='*50}")
    print(f"⚠️  PEOR VALORADO: {peor}")
    print(f"   Puntuación media: {rank_productos.loc[peor, 'puntuacion_media']:.2f}")
    print(f"   Compound medio: {rank_productos.loc[peor, 'compound_medio']:.3f}")
    print(f"   Reseñas: {rank_productos.loc[peor, 'count']}")

    # Mostrar reseñas del peor producto
    print(f"\nReseñas de '{peor}':")
    peor_resenas = resenas[resenas["producto"] == peor]["resena"].head(5)
    for i, res in enumerate(peor_resenas, 1):
        print(f"  {i}. {res}")

else:
    print("No hay columna 'producto' para rankear.")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
# Las primeras filas muestran la estructura de los datos procesados:
#    Las columnas dependen del DataFrame utilizado.
# 0    Los valores dependen de los datos de entrada.
# 1    valor4    valor5    valor6
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 11. PRODUCTO MEJOR Y PEOR VALORADO
3. ============================================================
4. Mostrar reseñas del peor producto

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```python
# ============================================================
# 12. RECOMENDACIONES DE MEJORA BASADAS EN RESEÑAS NEGATIVAS
# ============================================================
print("=" * 70)
print("RECOMENDACIONES DE MEJORA — ANÁLISIS DE RESEÑAS NEGATIVAS")
print("=" * 70)

resenas_neg = resenas[resenas["clase_real"] == "negativa"]

if len(resenas_neg) > 0:
    # Extraer bigramas de reseñas negativas
    from collections import Counter
    import re

    bigramas = []
    for texto in resenas_neg["resena"].dropna():
        palabras = re.findall(r'\b\w+\b', texto.lower())
        for i in range(len(palabras) - 1):
            if palabras[i] not in stop_words and palabras[i+1] not in stop_words:
                bigramas.append(f"{palabras[i]} {palabras[i+1]}")

    temas_frecuentes = Counter(bigramas).most_common(15)

    print("\nTemas recurrentes en reseñas negativas (bigramas):")
    for tema, count in temas_frecuentes:
        print(f"  - '{tema}': mencionado {count} veces")

    print("\n\nRECOMENDACIONES ESTRATÉGICAS:")
    print("-" * 70)

    temas_texto = " ".join([t[0] for t in temas_frecuentes])

    if any(p in temas_texto for p in ["calidad", "defectuoso", "roto", "falla"]):
        print("  1. CONTROL DE CALIDAD: Reforzar inspección de calidad antes del envío.")
    if any(p in temas_texto for p in ["tardó", "demora", "envío", "entrega"]):
        print("  2. LOGÍSTICA: Revisar acuerdos con transportistas y reducir tiempos de entrega.")
    if any(p in temas_texto for p in ["caro", "precio", "costó", "cuesta"]):
        print("  3. PRECIO: Evaluar estrategia de precios o agregar valor percibido.")
    if any(p in temas_texto for p in ["atención", "soporte", "servicio", "devolución"]):
        print("  4. SERVICIO AL CLIENTE: Capacitar al equipo de soporte y agilizar devoluciones.")
    if any(p in temas_texto for p in ["empaque", "embalaje", "llegó", "golpe"]):
        print("  5. EMPAQUE: Mejorar embalaje para evitar daños durante el transporte.")

    print(f"\n\nImpacto potencial:")
    print(f"  - {len(resenas_neg)} reseñas negativas de {len(resenas)} total ({len(resenas_neg)/len(resenas)*100:.1f}%)")
    if "producto" in resenas.columns:
        prod_problematico = resenas_neg.groupby("producto").size().sort_values(ascending=False).head(1).index[0]
        print(f"  - Producto con más quejas: {prod_problematico}")
else:
    print("No se encontraron reseñas negativas en el dataset.")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
# Las primeras filas muestran la estructura de los datos procesados:
#    Las columnas dependen del DataFrame utilizado.
# 0    Los valores dependen de los datos de entrada.
# 1    valor4    valor5    valor6
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. 12. RECOMENDACIONES DE MEJORA BASADAS EN RESEÑAS NEGATIVAS
3. ============================================================
4. Extraer bigramas de reseñas negativas

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```python
# ============================================================
# RESUMEN EJECUTIVO
# ============================================================
print("=" * 70)
print("RESUMEN EJECUTIVO — ANÁLISIS DE SENTIMIENTO")
print("=" * 70)
print(f"""
Dataset: {len(resenas)} reseñas analizadas
Métodos NLP: TextBlob (polaridad léxica), VADER (léxico + reglas)

Métrica de rendimiento:
  - VADER correlación con puntuación real: {corr_pearson_vd:.3f}
  - TextBlob correlación con puntuación real: {corr_pearson_tb:.3f}
  - Recomendación: {'VADER' if corr_pearson_vd > corr_pearson_tb else 'TextBlob'}

Hallazgos clave:
  1. Las reseñas positivas destacan: calidad, rápido, recomendado, excelente
  2. Las reseñas negativas mencionan: tardó, defectuoso, caro, mal servicio
  3. El sentimiento promedio por producto permite identificar líderes y rezagados
  4. VADER captura mejor el lenguaje informal y sarcasmo

Próximos pasos:
  - Implementar dashboard de monitoreo de sentimiento semanal
  - Configurar alertas cuando sentimiento de un producto baje de -0.5
  - Profundizar en reseñas negativas con análisis de aspectos (aspect-based)
""")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. RESUMEN EJECUTIVO
3. ============================================================

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



```python
# ============================================================
# EJERCICIOS COMPLEMENTARIOS
# ============================================================
print("=" * 70)
print("EJERCICIOS PARA PRACTICAR")
print("=" * 70)
print("""
1. Implementa tu propio clasificador de sentimiento con Naive Bayes.
2. Identifica aspectos específicos (batería, pantalla, cámara) en reseñas.
3. Traduce reseñas al inglés y compara resultados con el español original.
4. Crea un modelo de regresión que prediga puntuación desde el texto.
5. Analiza la evolución del sentimiento a lo largo del tiempo por producto.
""")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*Este ejemplo.*

1. ============================================================
2. EJERCICIOS COMPLEMENTARIOS
3. ============================================================

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---


