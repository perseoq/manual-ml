# CP25: Segmentación Dinámica de Precios con K-Means + PCA

## Resumen Ejecutivo

Estrategia de precios basada en segmentación de clientes usando K-Means sobre features de RFM (Recencia, Frecuencia, Monto) más elasticidad de precio. Se definen 4 segmentos con perfiles de sensibilidad al precio, se calcula el precio óptimo por segmento y se simula el impacto en ingresos.

**Dataset:** 2000 clientes con historial de compras
**Técnicas:** PCA, K-Means, Elasticidad de Precio, Simulación Monte Carlo
**Objetivo:** Incrementar ingresos 8-12% con precios dinámicos

---

## 1. Cargar Datos de Clientes con RFM

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('Set2')
np.random.seed(42)

# Generar datos de clientes sintéticos
n_clientes = 2000

# Semillas para cada segmento (perfiles RFM)
segmento_seeds = {
    'leales_alto_valor': {'recencia': (5, 15), 'frecuencia': (15, 30), 'monto': (300, 800)},
    'leales_bajo_valor': {'recencia': (10, 30), 'frecuencia': (8, 20), 'monto': (50, 200)},
    'ocasionales':       {'recencia': (30, 90), 'frecuencia': (2, 8), 'monto': (100, 400)},
    'perdidos':          {'recencia': (90, 365), 'frecuencia': (1, 4), 'monto': (30, 150)}
}

clientes = []
for i in range(n_clientes):
    tipo = np.random.choice(list(segmento_seeds.keys()), p=[0.3, 0.3, 0.25, 0.15])
    seed = segmento_seeds[tipo]
    recencia = max(1, int(np.random.normal(
        np.mean(seed['recencia']), 
        np.std(seed['recencia']) / 2
    )))
    frecuencia = max(1, int(np.random.normal(
        np.mean(seed['frecuencia']),
        np.std(seed['frecuencia']) / 3
    )))
    monto = max(10, round(np.random.normal(
        np.mean(seed['monto']),
        np.std(seed['monto']) / 3
    ), 2))
    
    clientes.append({
        'cliente_id': f'C-{i+1:04d}',
        'recencia': recencia,
        'frecuencia': frecuencia,
        'monto': monto,
        'ticket_promedio': round(monto / max(frecuencia, 1), 2),
        'antiguedad_meses': np.random.randint(1, 60)
    })

df = pd.DataFrame(clientes)

print(f"Clientes totales: {len(df)}")
print(f"\nEstadísticas RFM:")
print(df[['recencia', 'frecuencia', 'monto', 'ticket_promedio', 'antiguedad_meses']].describe())
print(f"\nCorrelaciones RFM:")
print(df[['recencia', 'frecuencia', 'monto']].corr().round(3))
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*1. Cargar Datos de Clientes con RFM.*

1. Generar datos de clientes sintéticos
2. Semillas para cada segmento (perfiles RFM)

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

## 2. Agregar Features de Sensibilidad al Precio (Elasticidad)

```python
# Simular elasticidad de precio por cliente
# Elasticidad = (ΔQ / Q) / (ΔP / P)
# Clientes sensibles: elasticidad < -1 (elásticos)
# Clientes no sensibles: elasticidad > -1 (inelásticos)

np.random.seed(42)
df['elasticidad'] = np.where(
    df['recencia'] < 30,  # clientes recientes (activos)
    np.random.normal(-1.5, 0.5, len(df)),  # más sensibles
    np.random.normal(-0.8, 0.4, len(df))   # menos sensibles
)
df['elasticidad'] = df['elasticidad'].clip(-3.0, -0.1)

# Feature compuesta: sensibilidad al precio
df['sensibilidad_precio'] = np.abs(df['elasticidad']) * (1 / df['monto'].clip(lower=1))

print("ESTADÍSTICAS DE ELASTICIDAD:")
print(df['elasticidad'].describe())
print(f"\nClientes elásticos (sensibles): {(df['elasticidad'] < -1).sum()} ({(df['elasticidad'] < -1).mean()*100:.1f}%)")
print(f"Clientes inelásticos (no sensibles): {(df['elasticidad'] >= -1).sum()} ({(df['elasticidad'] >= -1).mean()*100:.1f}%)")

# Visualizar distribución
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df['elasticidad'], bins=40, edgecolor='white', color='steelblue')
axes[0].axvline(x=-1, color='red', linestyle='--', label='Umbral elástico/inelástico')
axes[0].set_title('Distribución de Elasticidad de Precio', fontweight='bold')
axes[0].set_xlabel('Elasticidad')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].scatter(df['monto'], df['elasticidad'], alpha=0.3, s=15, c='coral')
axes[1].set_xlabel('Monto Total ($)')
axes[1].set_ylabel('Elasticidad')
axes[1].set_title('Elasticidad vs Monto', fontweight='bold')
axes[1].axhline(y=-1, color='gray', linestyle='--', alpha=0.5)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('img/elasticidad_precio.png', dpi=150)
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*2. Agregar Features de Sensibilidad al Precio (Elasticidad).*

1. Simular elasticidad de precio por cliente
2. Elasticidad = (ΔQ / Q) / (ΔP / P)
3. Clientes sensibles: elasticidad < -1 (elásticos)
4. Clientes no sensibles: elasticidad > -1 (inelásticos)
5. Feature compuesta: sensibilidad al precio
6. Visualizar distribución

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

## 3. Escalar Features y Aplicar PCA a 2D

```python
# Features para segmentación
features_seg = ['recencia', 'frecuencia', 'monto', 'ticket_promedio', 
                'antiguedad_meses', 'elasticidad', 'sensibilidad_precio']
X = df[features_seg].values

# Escalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df['pca1'] = X_pca[:, 0]
df['pca2'] = X_pca[:, 1]

print("PCA - Varianza explicada:")
for i, ratio in enumerate(pca.explained_variance_ratio_, 1):
    print(f"  PC{i}: {ratio:.2%}")
print(f"  Total: {pca.explained_variance_ratio_.sum():.2%}")

# Contribución de features a cada PC
print(f"\nContribución de features a PC1:")
for i, feat in enumerate(features_seg):
    print(f"  {feat}: {pca.components_[0, i]:.3f}")
print(f"\nContribución de features a PC2:")
for i, feat in enumerate(features_seg):
    print(f"  {feat}: {pca.components_[1, i]:.3f}")

# Visualizar loadings
fig, ax = plt.subplots(figsize=(10, 6))
for i, feat in enumerate(features_seg):
    ax.arrow(0, 0, pca.components_[0, i], pca.components_[1, i], 
             head_width=0.05, head_length=0.05, fc='steelblue', ec='steelblue', alpha=0.7)
    ax.text(pca.components_[0, i]*1.15, pca.components_[1, i]*1.15, feat, fontsize=10)
circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', alpha=0.5)
ax.add_patch(circle)
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax.set_title('Círculo de Correlaciones PCA', fontweight='bold')
ax.axhline(0, color='gray', alpha=0.3)
ax.axvline(0, color='gray', alpha=0.3)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('img/pca_circulo_correlaciones.png', dpi=150)
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*3. Escalar Features y Aplicar PCA a 2D.*

1. Features para segmentación
2. Escalar
3. PCA
4. Contribución de features a cada PC
5. Visualizar loadings

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



**Interpretación:** PC1 se correlaciona positivamente con frecuencia/monto y negativamente con recencia. PC2 se correlaciona con elasticidad y sensibilidad al precio.

---

## 4. K-Means con k=4 para Segmentar Clientes

```python
# Determinar k óptimo
k_range = range(2, 10)
inertias = []
silhouettes = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(k_range, inertias, marker='o', linewidth=2)
axes[0].axvline(x=4, color='red', linestyle='--', alpha=0.7, label='k=4 elegido')
axes[0].set_title('Método del Codo (Inertia)', fontweight='bold')
axes[0].set_xlabel('Número de clusters (k)')
axes[0].set_ylabel('Inertia')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(k_range, silhouettes, marker='s', linewidth=2, color='green')
axes[1].axvline(x=4, color='red', linestyle='--', alpha=0.7, label='k=4 elegido')
axes[1].set_title('Coeficiente de Silhouette', fontweight='bold')
axes[1].set_xlabel('Número de clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('img/k_selection.png', dpi=150)
plt.show()

# K-Means con k=4
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['segmento'] = kmeans.fit_predict(X_scaled)

print(f"Silhouette Score (k=4): {silhouette_score(X_scaled, df['segmento']):.4f}")
print(f"\nDistribución de segmentos:")
print(df['segmento'].value_counts().sort_index().to_string())
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*4. K-Means con k=4 para Segmentar Clientes.*

1. Determinar k óptimo
2. K-Means con k=4

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

## 5. Interpretar Perfiles de Segmentos

```python
# Perfil de cada segmento
perfil = df.groupby('segmento')[features_seg].mean()
perfil['count'] = df.groupby('segmento').size()

# Añadir columna de recencia inversa para interpretación
perfil['lealtad_score'] = 1 / perfil['recencia'].clip(lower=1) * perfil['frecuencia']

print("PERFIL DE SEGMENTOS")
print("="*80)
print(perfil.round(2).to_string())

# Renombrar segmentos basado en perfil
segmento_nombres = {}
for seg in sorted(df['segmento'].unique()):
    mask = df['segmento'] == seg
    r_mean = df.loc[mask, 'recencia'].mean()
    f_mean = df.loc[mask, 'frecuencia'].mean()
    m_mean = df.loc[mask, 'monto'].mean()
    e_mean = df.loc[mask, 'elasticidad'].mean()
    
    if r_mean < 20 and f_mean > 12 and m_mean > 300:
        nombre = 'Premium Leales'
    elif r_mean < 40 and f_mean > 6:
        nombre = 'Regulares'
    elif e_mean < -1.2:
        nombre = 'Cazadores de Ofertas'
    else:
        nombre = 'Ocasionales / Perdidos'
    
    segmento_nombres[seg] = nombre

df['segmento_nombre'] = df['segmento'].map(segmento_nombres)
print(f"\nSegmentos identificados:")
for seg, nombre in sorted(segmento_nombres.items()):
    print(f"  {seg}: {nombre} ({len(df[df['segmento']==seg])} clientes)")

# Visualizar perfiles
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
features_plot = ['recencia', 'frecuencia', 'monto', 'ticket_promedio', 'elasticidad', 'antiguedad_meses']
for ax, feat in zip(axes.flatten(), features_plot):
    for seg in sorted(df['segmento'].unique()):
        data = df[df['segmento'] == seg][feat]
        ax.hist(data, bins=30, alpha=0.5, label=segmento_nombres[seg])
    ax.set_title(f'{feat} por Segmento')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('img/perfiles_segmentos.png', dpi=150)
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*5. Interpretar Perfiles de Segmentos.*

1. Perfil de cada segmento
2. Añadir columna de recencia inversa para interpretación
3. Renombrar segmentos basado en perfil
4. Visualizar perfiles

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



**Perfiles típicos:**
| Segmento | Recencia | Frecuencia | Monto | Elasticidad |
|----------|----------|------------|-------|-------------|
| Premium Leales | 5-15 días | 15-30 | $300-800 | -0.8 (inelástico) |
| Regulares | 10-30 días | 8-20 | $50-200 | -1.0 (neutro) |
| Cazadores de Ofertas | 30-90 días | 2-8 | $100-400 | -2.0 (elástico) |
| Ocasionales | 90-365 días | 1-4 | $30-150 | -0.6 (poco sensible) |

---

## 6. Calcular Precio Óptimo por Segmento (Elasticidad)

```python
def precio_optimo(precio_base, elasticidad, costo, margen_objetivo=0.3):
    """
    Calcula precio óptimo basado en elasticidad y margen.
    Maximiza: beneficio = (P - C) * Q(P)
    donde Q(P) = Q0 * (P/P0)^elasticidad
    """
    # Precio que maximiza beneficio para demanda con elasticidad constante
    if elasticidad < -1:
        # Demanda elástica: precio óptimo = C * e / (1 + e)
        p_opt = costo * elasticidad / (1 + elasticidad)
    else:
        # Demanda inelástica: podemos subir precio
        p_opt = precio_base * 1.2
    
    # Asegurar margen mínimo
    margen = (p_opt - costo) / p_opt
    if margen < margen_objetivo:
        p_opt = costo / (1 - margen_objetivo)
    
    return max(p_opt, costo * 1.1)  # mínimo 10% de margen

# Simular costos y precios base
costo_promedio = 50  # costo promedio por compra
precio_base = 100  # precio de referencia

df['precio_optimo'] = df.apply(
    lambda row: precio_optimo(precio_base, row['elasticidad'], costo_promedio), 
    axis=1
)

df['precio_actual'] = precio_base

print("PRECIOS ÓPTIMOS POR SEGMENTO")
print("="*60)
for seg in sorted(df['segmento'].unique()):
    mask = df['segmento'] == seg
    nombre = segmento_nombres[seg]
    precio_prom = df.loc[mask, 'precio_optimo'].mean()
    precio_actual = df.loc[mask, 'precio_actual'].mean()
    cambio = (precio_prom - precio_actual) / precio_actual * 100
    print(f"{nombre}:")
    print(f"  Precio actual: ${precio_actual:.2f}")
    print(f"  Precio óptimo: ${precio_prom:.2f} ({cambio:+.1f}%)")
    print(f"  Elasticidad media: {df.loc[mask, 'elasticidad'].mean():.2f}")
    print()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*6. Calcular Precio Óptimo por Segmento (Elasticidad).*

1. Precio que maximiza beneficio para demanda con elasticidad constante
2. Demanda elástica: precio óptimo = C * e / (1 + e)
3. Demanda inelástica: podemos subir precio
4. Asegurar margen mínimo
5. Simular costos y precios base

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

## 7. Estrategia de Precios por Segmento

```python
print("ESTRATEGIA DE PRECIOS POR SEGMENTO")
print("="*70)
print()

estrategias = []
for seg in sorted(df['segmento'].unique()):
    mask = df['segmento'] == seg
    nombre = segmento_nombres[seg]
    e_mean = df.loc[mask, 'elasticidad'].mean()
    p_opt = df.loc[mask, 'precio_optimo'].mean()
    p_actual = df.loc[mask, 'precio_actual'].mean()
    
    if e_mean < -1.5:
        estrategia = 'OFERTA: Reducir precio 15-20% para atraer volumen'
        cambio_pct = -0.15
    elif e_mean < -1.0:
        estrategia = 'DESCUENTO: Reducir precio 5-10% (promociones selectivas)'
        cambio_pct = -0.08
    elif e_mean < -0.5:
        estrategia = 'ESTÁNDAR: Mantener precio (competitivo)'
        cambio_pct = 0.0
    else:
        estrategia = 'PREMIUM: Aumentar precio 5-15% (clientes cautivos)'
        cambio_pct = 0.10
    
    precio_nuevo = p_actual * (1 + cambio_pct)
    estrategias.append({
        'segmento': seg,
        'nombre': nombre,
        'elasticidad': round(e_mean, 2),
        'precio_actual': p_actual,
        'precio_nuevo_sugerido': round(precio_nuevo, 2),
        'cambio_pct': cambio_pct * 100,
        'estrategia': estrategia,
        'clientes': len(df[mask])
    })

df_estrategias = pd.DataFrame(estrategias)
print(df_estrategias[['nombre', 'elasticidad', 'precio_actual', 
                       'precio_nuevo_sugerido', 'cambio_pct', 'estrategia']].to_string(index=False))
print()

# Visualizar estrategia
fig, ax = plt.subplots(figsize=(10, 6))
for _, row in df_estrategias.iterrows():
    color = 'green' if row['cambio_pct'] > 0 else 'red' if row['cambio_pct'] < 0 else 'gray'
    ax.barh(row['nombre'], row['cambio_pct'], color=color, alpha=0.7, 
            height=0.4, label=f"{row['clientes']} clientes")
    ax.text(row['cambio_pct'], row['nombre'], f" {row['precio_nuevo_sugerido']:.0f}$ ({row['cambio_pct']:+.0f}%)", 
            va='center', fontweight='bold')
ax.axvline(0, color='black', linewidth=1)
ax.set_xlabel('Cambio de Precio (%)')
ax.set_title('Estrategia de Precios por Segmento', fontweight='bold')
ax.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('img/estrategia_precios.png', dpi=150)
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*7. Estrategia de Precios por Segmento.*

1. Visualizar estrategia

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

## 8. Simular Impacto de Precios Dinámicos en Ingresos

```python
def simular_impacto(df, estrategias, n_simulaciones=1000):
    """
    Simula el impacto de precios dinámicos en ingresos totales.
    """
    ingresos_actuales = []
    ingresos_nuevos = []
    
    for _ in range(n_simulaciones):
        # Ingreso actual
        ingreso_act = (df['frecuencia'] * df['precio_actual']).sum()
        ingresos_actuales.append(ingreso_act)
        
        # Ingreso con precios dinámicos
        ingreso_nuevo = 0
        for _, row in df.iterrows():
            seg = row['segmento']
            e = row['elasticidad']
            cambio = estrategias.loc[estrategias['segmento'] == seg, 'cambio_pct'].values[0] / 100
            
            # Nueva cantidad basada en elasticidad
            cambio_q = (1 + cambio) ** e - 1
            nueva_cantidad = row['frecuencia'] * (1 + cambio_q)
            nuevo_precio = row['precio_actual'] * (1 + cambio)
            
            ingreso_nuevo += max(nueva_cantidad, 0) * max(nuevo_precio, 0)
        
        ingresos_nuevos.append(ingreso_nuevo)
    
    return np.array(ingresos_actuales), np.array(ingresos_nuevos)

ingresos_act, ingresos_nue = simular_impacto(df, df_estrategias, n_simulaciones=1000)

print("SIMULACIÓN DE IMPACTO EN INGRESOS")
print("="*60)
print(f"Ingreso actual promedio: ${ingresos_act.mean():,.2f}")
print(f"Ingreso con precios dinámicos: ${ingresos_nue.mean():,.2f}")
print(f"Incremento: ${(ingresos_nue - ingresos_act).mean():,.2f}")
print(f"Incremento porcentual: {(ingresos_nue / ingresos_act - 1).mean()*100:.2f}%")
print(f"Intervalo de confianza 95% del incremento: "
      f"[{(ingresos_nue - ingresos_act).mean() - 1.96*(ingresos_nue - ingresos_act).std():,.0f}, "
      f"{(ingresos_nue - ingresos_act).mean() + 1.96*(ingresos_nue - ingresos_act).std():,.0f}]")

# Histograma de ingresos
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(ingresos_act / 1000, bins=40, alpha=0.6, label='Precio fijo', color='steelblue')
ax.hist(ingresos_nue / 1000, bins=40, alpha=0.6, label='Precios dinámicos', color='coral')
ax.axvline(ingresos_act.mean()/1000, color='steelblue', linestyle='--', linewidth=2)
ax.axvline(ingresos_nue.mean()/1000, color='coral', linestyle='--', linewidth=2)
ax.set_xlabel('Ingresos totales ($K)')
ax.set_ylabel('Frecuencia')
ax.set_title('Simulación de Ingresos: Precio Fijo vs Dinámicos', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('img/simulacion_ingresos.png', dpi=150)
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*8. Simular Impacto de Precios Dinámicos en Ingresos.*

1. Ingreso actual
2. Ingreso con precios dinámicos
3. Nueva cantidad basada en elasticidad
4. Histograma de ingresos

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

## 9. Visualizar Segmentos 2D con Scatterplot

```python
# PCA con anotaciones de segmentos
fig, ax = plt.subplots(figsize=(12, 8))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for seg in sorted(df['segmento'].unique()):
    mask = df['segmento'] == seg
    ax.scatter(df.loc[mask, 'pca1'], df.loc[mask, 'pca2'], 
               label=segmento_nombres[seg], color=colors[seg], alpha=0.5, s=30)

# Centroides
centroids_pca = pca.transform(kmeans.cluster_centers_)
ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
           marker='X', s=300, color='black', edgecolors='white', linewidth=2, zorder=5)
for seg, (cx, cy) in enumerate(centroids_pca):
    ax.annotate(segmento_nombres[seg], (cx, cy), 
                textcoords="offset points", xytext=(0, 15), ha='center', fontweight='bold')

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax.set_title('Segmentación de Clientes (PCA + K-Means)', fontweight='bold', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('img/segmentacion_2d.png', dpi=150)
plt.show()

# 3D de recencia, frecuencia y monto
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
for seg in sorted(df['segmento'].unique()):
    mask = df['segmento'] == seg
    ax.scatter(df.loc[mask, 'recencia'], df.loc[mask, 'frecuencia'], df.loc[mask, 'monto'],
               label=segmento_nombres[seg], color=colors[seg], alpha=0.4, s=15)
ax.set_xlabel('Recencia (días)')
ax.set_ylabel('Frecuencia (compras)')
ax.set_zlabel('Monto ($)')
ax.set_title('Segmentos en Espacio RFM 3D', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('img/segmentacion_3d.png', dpi=150)
plt.show()
```

**Salida:**

```
# Se genera un gráfico estadístico con los datos seleccionados.
# El gráfico se muestra en pantalla; usar plt.savefig() para guardarlo.
```

**Explicación línea por línea:**

*9. Visualizar Segmentos 2D con Scatterplot.*

1. PCA con anotaciones de segmentos
2. Centroides
3. 3D de recencia, frecuencia y monto

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

## 10. Tabla de Recomendaciones por Segmento

```python
print("TABLA DE RECOMENDACIONES POR SEGMENTO")
print("="="*80)

for _, row in df_estrategias.iterrows():
    print(f"\n📌 SEGMENTO: {row['nombre']} ({row['clientes']} clientes)")
    print(f"{'─'*60}")
    print(f"  Perfil: Elasticidad {row['elasticidad']}, Precio actual ${row['precio_actual']:.0f}")
    print(f"  Estrategia: {row['estrategia']}")
    print(f"  Precio sugerido: ${row['precio_nuevo_sugerido']:.0f}")
    print()
    
    if 'Premium' in row['nombre']:
        print("  Acciones:")
        print("  ├ Programa de lealtad VIP (puntos dobles)")
        print("  ├ Acceso anticipado a nuevos productos")
        print("  ├ Precios premium (5-15% sobre precio base)")
        print("  └ Atención personalizada con ASM dedicado")
    elif 'Regulares' in row['nombre']:
        print("  Acciones:")
        print("  ├ Mantener precio competitivo")
        print("  ├ Promociones cruzadas (cross-sell)")
        print("  ├ Programa de referidos")
        print("  └ Encuestas de satisfacción periódicas")
    elif 'Ofertas' in row['nombre']:
        print("  Acciones:")
        print("  ├ Ofertas semanales personalizadas")
        print("  ├ Cupones de descuento por tiempo limitado")
        print("  ├ Notificaciones push de ofertas")
        print("  └ Bundle discounts (2x1, 3x2)")
    else:
        print("  Acciones:")
        print("  ├ Campaña de re-engagement (email)")
        print("  ├ Oferta de bienvenida con 20% desc.")
        print("  ├ Encuesta de abandono")
        print("  └ Recomendación de productos populares")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*10. Tabla de Recomendaciones por Segmento.*

1. `print("TABLA DE RECOMENDACIONES POR SEGMENTO")` — Muestra el resultado por pantalla.
2. `print("="="*80)` — Muestra el resultado por pantalla.
3. `print(f"\n📌 SEGMENTO: {row['nombre']} ({row['clientes']} clientes)")` — Muestra el resultado por pantalla.
4. `print(f"{'─'*60}")` — Muestra el resultado por pantalla.
5. `print(f"  Perfil: Elasticidad {row['elasticidad']}, Precio actual ${row['precio_actual']:.0f}")` — Muestra el resultado por pantalla.
6. `print(f"  Estrategia: {row['estrategia']}")` — Muestra el resultado por pantalla.
7. `print(f"  Precio sugerido: ${row['precio_nuevo_sugerido']:.0f}")` — Muestra el resultado por pantalla.
8. `print()` — Muestra el resultado por pantalla.

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

## 11. ROI Estimado de Precios Dinámicos vs Fijos

```python
# Calcular ROI
costo_implementacion = 50000  # $50K para implementar sistema
ingreso_actual_anual = df['monto'].sum() * 12 / df['antiguedad_meses'].mean() * 12

# Estimación conservadora: 8% de incremento
incremento_estimado = 0.08
ingreso_adicional = ingreso_actual_anual * incremento_estimado

roi_anual = (ingreso_adicional - costo_implementacion) / costo_implementacion * 100
payback_meses = costo_implementacion / (ingreso_adicional / 12)

print("ANÁLISIS DE ROI")
print("="*60)
print(f"Costo de implementación: ${costo_implementacion:,.0f}")
print(f"Ingreso actual anual estimado: ${ingreso_actual_anual:,.0f}")
print(f"Incremento estimado (8%): ${ingreso_adicional:,.0f}")
print()
print(f"ROI Anual: {roi_anual:.0f}%")
print(f"Payback: {payback_meses:.1f} meses")
print()

# Simulación de diferentes escenarios
escenarios = [0.05, 0.08, 0.10, 0.12, 0.15]
print(f"Análisis de sensibilidad:")
print(f"{'Incremento':<15} {'Ingreso Adic.':<20} {'ROI':<15} {'Payback (meses)':<15}")
print(f"{'-'*65}")
for esc in escenarios:
    ing_adic = ingreso_actual_anual * esc
    roi_esc = (ing_adic - costo_implementacion) / costo_implementacion * 100
    payback_esc = costo_implementacion / (ing_adic / 12)
    print(f"{esc:<15.0%} ${ing_adic:<17,.0f} {roi_esc:<15.0f}% {payback_esc:<15.1f}")

# Visualizar
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.bar([f'{e:.0%}' for e in escenarios], 
        [ingreso_actual_anual * e / 1000 for e in escenarios], 
        color=['#ff6b6b' if e*ingreso_actual_anual < costo_implementacion else '#51cf66' for e in escenarios])
ax1.axhline(y=costo_implementacion/1000, color='red', linestyle='--', label=f'Costo implementación ${costo_implementacion/1000:.0f}K')
ax1.set_xlabel('Incremento de ingresos')
ax1.set_ylabel('Ingreso adicional ($K)')
ax1.set_title('ROI de Precios Dinámicos por Escenario', fontweight='bold')
ax1.legend()
ax1.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('img/roi_escenarios.png', dpi=150)
plt.show()
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*11. ROI Estimado de Precios Dinámicos vs Fijos.*

1. Calcular ROI
2. Estimación conservadora: 8% de incremento
3. Simulación de diferentes escenarios
4. Visualizar

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

## 12. Implementación: Reglas de Pricing Automáticas

```python
print("SISTEMA DE PRICING AUTOMÁTICO")
print("="*70)
print()
print("REGLAS DE NEGOCIO IMPLEMENTADAS:")
print("-"*50)
print()
print("1. IDENTIFICACIÓN DEL CLIENTE")
print("   Al llegar un cliente (login/cookie):")
print("   ├ Buscar en BD de clientes")
print("   ├ Calcular RFM en tiempo real")
print("   ├ Asignar segmento con modelo K-Means")
print("   └ Aplicar regla de precio correspondiente")
print()
print("2. REGLAS DE PRECIO POR SEGMENTO")
print()
print("   SEGMENTO PREMIUM LEALES:")
print(f"   ├ Precio: +10% sobre precio base")
print("   ├ Condición: frecuencia > 12 compras/año")
print("   ├ Excepción: si ticket < $100, mantener precio base")
print("   └ Caducidad: 30 días sin actualizar")
print()
print("   SEGMENTO REGULARES:")
print("   ├ Precio: precio base")
print("   ├ Condición: recencia < 60 días")
print("   ├ Promoción: 5% descuento si compra > $200")
print("   └ Caducidad: 15 días")
print()
print("   SEGMENTO CAZADORES DE OFERTAS:")
print(f"   ├ Precio: -15% sobre precio base")
print("   ├ Condición: alerta de oferta activa")
print("   ├ Límite: máximo 3 ofertas/mes")
print("   └ Caducidad: 7 días")
print()
print("   SEGMENTO OCASIONALES:")
print("   ├ Precio: -20% primera compra")
print("   ├ Condición: antigüedad > 6 meses sin compra")
print("   ├ Estrategia: recuperación (win-back)")
print("   └ Caducidad: 14 días")
print()
print("3. MONITOREO Y AJUSTE")
print("   ├ Recalcular segmentos semanalmente")
print("   ├ Alertas si elasticidad cambia >0.3")
print("   ├ Reentrenar K-Means mensualmente")
print("   └ Dashboard de KPIs: ingresos, margen, conversión")
print()
print("4. A/B TESTING CONTINUO")
print("   ├ 10% tráfico: precio fijo (control)")
print("   ├ 90% tráfico: precios dinámicos")
print("   ├ Duración: 2 semanas")
print("   └ Métrica: revenue per visitor")
print()
print("5. CÓDIGO DE IMPLEMENTACIÓN (PSEUDO)")
print("-"*50)
print("""
def get_price(client_id, product_base_price):
    # Obtener RFM del cliente
    rfm = get_rfm(client_id)
    
    # Escalar y predecir segmento
    X = scaler.transform([rfm])
    segment = kmeans.predict(X)[0]
    
    # Aplicar regla de precio
    price_rules = {
        0: {'mult': 1.10, 'label': 'Premium'},     # Premium Leales
        1: {'mult': 1.00, 'label': 'Regular'},      # Regulares
        2: {'mult': 0.85, 'label': 'Oferta'},       # Cazadores de Ofertas
        3: {'mult': 0.80, 'label': 'Winback'}       # Ocasionales
    }
    
    rule = price_rules[segment]
    final_price = product_base_price * rule['mult']
    
    return round(final_price, 2), rule['label']
""")
```

**Salida:**

```
# Al ejecutar el código se imprimen los resultados en consola.
# Se imprimen los resultados obtenidos tras ejecutar las operaciones.
```

**Explicación línea por línea:**

*12. Implementación: Reglas de Pricing Automáticas.*

1. Obtener RFM del cliente
2. Escalar y predecir segmento
3. Aplicar regla de precio

*Este análisis permite comprender cada operación aplicada a los datos de ventas, compras o inventarios.*

---



---

## Conclusiones

1. **K-Means + PCA** identifica 4 segmentos de clientes con perfiles RFM y de elasticidad diferenciados
2. **Precios dinámicos** basados en elasticidad incrementan ingresos estimados 8-12%
3. **Cazadores de Ofertas** requieren descuentos para activar compras; **Premium Leales** toleran incrementos de precio
4. **ROI positivo** desde el primer año (payback < 6 meses con incremento > 8%)
5. **Implementación automática** viable con reglas de negocio claras por segmento
6. **Próximos pasos:** A/B testing en producción para validar simulación

---

## 5 Ejercicios Adicionales

**E01:** Implementar segmentación jerárquica (AgglomerativeClustering) y comparar con K-Means.

**E02:** Añadir features de estacionalidad (gasto por mes, comportamiento en Black Friday/Cyber Monday).

**E03:** Construir un modelo de elasticidad individual por cliente usando regresión log-log en lugar de elasticidad promedio.

**E04:** Optimizar el número de segmentos con Gap Statistic o Calinski-Harabasz Index.

**E05:** Implementar pricing en tiempo real con Reinforcement Learning (Q-Learning o Bandits) para ajustar precios dinámicamente.
