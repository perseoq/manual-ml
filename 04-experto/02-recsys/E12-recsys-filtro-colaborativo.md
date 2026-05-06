# E12 — Filtro Colaborativo (Collaborative Filtering)

## Introducción Teórica

El filtro colaborativo (CF) es la técnica más clásica y extendida en sistemas de recomendación. Se basa en la premisa de que **usuarios similares tienen gustos similares** (user-based CF) o que **productos similares son comprados por los mismos usuarios** (item-based CF). En el contexto de ventas B2B y B2C, esto se traduce en recomendar productos que otros clientes similares han comprado.

### Matriz Usuario-Producto (User-Item Matrix)

La base de todo CF es la matriz de interacciones:

- **Filas**: usuarios/clientes
- **Columnas**: productos/items
- **Valores**: rating implícito (cantidad comprada, 0/1) o explícito (estrellas, puntuación)

En B2B, la matriz suele ser muy dispersa (>99% sparsity) porque un cliente empresarial compra pocos productos de un catálogo enorme. En B2C, la dispersión también es alta pero con más interacciones por usuario.

### User-Based Collaborative Filtering

1. **Calcular similitud** entre el usuario activo `u` y todos los demás usuarios `v`
2. **Seleccionar K vecinos** más similares que hayan comprado/interactuado con el producto `i`
3. **Predecir rating** como promedio ponderado de los ratings de los vecinos:

   ```
   ```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación línea por línea:**


---
 — Realiza la operación indicada con los parámetros definidos..

---

---


   donde `r̄_u` es el rating medio del usuario `u` (centrado para eliminar sesgos).

### Medidas de Similitud

- **Coseno**: `sim(u,v) = (r_u · r_v) / (||r_u|| · ||r_v||)`
- **Pearson**: coseno sobre ratings centrados (restar media de cada usuario)
- **Ajustada (Adjusted Cosine)**: centrar medias de cada usuario antes del coseno, eliminando sesgo de escala

### Item-Based Collaborative Filtering

Similar pero entre items: "clientes que compraron este producto también compraron...". La similitud entre items se calcula observando qué usuarios los compraron juntos.

### Evaluación

- **RMSE**: raíz del error cuadrático medio entre predicción y real
- **MAE**: error absoluto medio
- **precision@k**: fracción de relevantes en top-k recomendados
- **recall@k**: fracción de relevantes recuperados entre todos los relevantes
- **NDCG@k**: ranking con descuento acumulado normalizado
- **Hit Rate@k**: si al menos un relevante está en top-k

### Cold Start

- **Nuevo usuario**: no hay historial → no se puede calcular similitud → fallback a populares
- **Nuevo producto**: nadie lo ha comprado → no tiene vecinos → fallback a metadata

### Model-Based CF: SVD

SVD trunca la matriz usuario-item en factores latentes: `R ≈ U · Σ · V^T`. Cada usuario y producto se representa como un vector de k factores. La predicción es `r̂_ui = u_u · v_i^T`. Esto escala mejor que vecinos (k-NN) y generaliza mejor.

---

## Ejemplos

```python
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las opera**Explicación línea por línea:**

*Ejemplos*


---
:**

1. ````` — Realiza la operación indicada con los parámetros definidos..

---

---


```
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver **Explicación línea por línea:**


---
iones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


---

### Ejemplo 1: Crear matriz usuario-producto (pivot_table)

```python
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variab**Explicación línea por línea:**

*Ejemplo 1: Crear matriz usuario-producto (pivot_table)*


---
de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


```
```

**Salida:**

```
# Los result**Explicación línea por línea:**


---
ariables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


---

### Ejemplo 2: Calcular similitud coseno entre usuarios

```python
```

**Salida:**

```**Explicación línea por línea:**

*Ejemplo 2: Calcular similitud coseno entre usuarios*


---
enan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---

ada.

---

---


```
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---

plo 3: Calcular similitud Pearson entre usuarios

Pearson centra los datos restando la media de cada usuario antes del coseno.

```python
```

**Salida:**

```
# Los resultados se almacenan en las **Explicación línea por línea:**


---
peccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


```
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


---

### Ejemplo 4: Similitud ajustada (centrar medias de cada usuario)

La similitud **Explicación línea por línea:**

*Ejemplo 4: Similitud ajustada (centrar medias de cada usuario)*


---
 es exactamente el coseno sobre datos centrados por usuario. Es particularmente útil en item-based CF porque elimina el sesgo de es**Explicación línea por línea:**


---
python
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


```
```

**Salida:**

```
# Los resultados se almac**Explicación línea por línea:**


---
nidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


---

### Ejemplo 5: User-based — prede**Explicación línea por línea:**

*Ejemplo 5: User-based — prede*

1. `---` — Realiza la operación indicada con los parámetros definidos..
2. `nidas.` — Realiza la operación indicada con los parámetros definidos..

---
 producto no comprado

```python
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para **Explicación línea por línea:**


---
eraciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


```
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las va**Explicación línea por línea:**


---
ado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


---

### Ejemplo 6: User-based — top-N recomendaciones para un cliente

```python
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Ins**Explicación línea por línea:**

*Ejemplo 6: User-based — top-N recomendaciones para un cliente*


---
ra ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


```
```

****Explicación línea por línea:**


---
tados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


---

### Ejemplo 7: Item-based — calcular similitud entre productos

```python
```

***Explicación línea por línea:**

*Ejemplo 7: Item-based — calcular similitud entre productos*


---
ltados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicaci**Explicación línea por línea:**


---
lRealiza la operación indicada con los parámetros definidos.

---

---


```
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


---

#**Explicación línea por línea:**

1. `**Explicación:**` — Realiza la operación indicada con los parámetros definidos..
2. `1.` — Realiza la operación indicada con los parámetros definidos..

---
 recomendar productos similares a los comprados

```python
```

**Salida:**

```
# Los result**Explicación línea por línea:**


---
ariables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


```
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas**Explicación línea por línea:**


---
bles para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


---

### Ejemplo 9: Comparar user-based vs item-based (precisión, cob**Explicación línea por línea:**

*Ejemplo 9: Comparar user-based vs item-based (precisión, cob*

1. `---` — Realiza la operación indicada con los parámetros definidos..
2. `bles para ver el resultado de las operaciones.` — Realiza la operación indicada con los parámetros definidos..

---
Salida:**

```
# Los resultados se almacenan en las variables definidas.
# I**Explicación línea por línea:**

1. `---` — Realiza la operación indicada con los parámetros definidos..
2. `bles para ver el resultado de las operaciones.` — Realiza la operación indicada con los parámetros definidos..

---
para ver el resultado de las operaciones.
```

**Explicación:***Explicación línea por línea:**


1. El código prepara y procesa los datos según las operaciones definidas.

---

— Realiza la operación indicada con los parámetros definidos..
4. `para ver el resultado de las operaciones.` — Realiza la operación indicada con los parámetros definidos..

---
iones.` — Realiza la operación indicada con los parámetros definidos..

---
— Realiza la operación indicada con los parámetros definidos..
4. `para ver el resultado de las operaciones.` — Realiza la operación indicada con los parámetros definidos..

---
Realiza la operación indicada con los parámetros definidos.

---
— Realiza la operación indicada con los parámetros definidos..
4. `para ver el resultado de las operaciones.` — Realiza la operación indicada con los parámetros definidos..

---
ones.` — Realiza la operación indicada con los parámetros definidos..

---
— Realiza la operación indicada con los parámetros definidos..
4. `para ver el resultado de las operaciones.` — Realiza la operación indicada con los parámetros definidos..

---
nes.` — Realiza la operación indicada con los parámetros definidos..

---
— Realiza la operación indicada con los parámetros definidos..
4. `para ver el resultado de las operaciones.` — Realiza la operación indicada con los parámetros definidos..

---
s.` — Realiza la operación indicada con los parámetros definidos..

---
— Realiza la operación indicada con los parámetros definidos..
4. `para ver el resultado de las operaciones.` — Realiza la operación indicada con los parámetros definidos..

---
s.` — Realiza la operación indicada con los parámetros definidos..

---
— Realiza la operación indicada con los parámetros definidos..
4. `para ver el resultado de las operaciones.` — Realiza la operación indicada con los parámetros definidos..

---
.` — Realiza la operación indicada con los parámetros definidos..

---
peración eRealiza la operación indicada con los parámetros definidos..

---

---


```
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


---

### Ejemplo 10: Evaluar RMSE de predicciones

```python
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


```
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


---

### Ejemplo 11: precision@k y recall@k para top-5 recomendaciones

```python
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


```
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


---

### Ejemplo 12: NDCG@k — ranking con descuento acumulado normalizado

```python
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


```
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


---

### Ejemplo 13: Train/test split temporal (última compra como test)

```python
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


```
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


---

### Ejemplo 14: Cold start — recomendar a nuevo usuario (populares)

```python
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


```
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


---

### Ejemplo 15: Cold start — nuevo producto (basado en categoría)

```python
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


```
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


---

### Ejemplo 16: Matriz usuario-item dispersa (sparsity analysis)

```python
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


```
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


---

### Ejemplo 17: SVD truncado como modelo de factorización (model-based CF)

```python
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


```
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


---

### Ejemplo 18: Integrador — sistema CF completo con evaluación

```python
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


```
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


---

## Ejercicios

1. **Implementar similitud de Jaccard entre usuarios**:
   `sim(u,v) = |I_u ∩ I_v| / |I_u ∪ I_v|` donde `I_u` son los items que compró el usuario `u`. Comparar con coseno en términos de cobertura.

2. **Top-N con filtro de categoría**: Modificar `recommend_top_n_user_based` para que solo recomiende productos de una categoría específica (ej. solo Cat1).

3. **Item-based con adjusted cosine**: Implementar item-based CF usando adjusted cosine (centrar filas de usuario antes de coseno entre items). ¿Mejora RMSE?

4. **k-NN óptimo**: Probar k = [5, 10, 20, 50, 100] vecinos en user-based. Reportar RMSE para cada k. ¿Cuál es el mejor?

5. **Hit Rate@k**: Implementar hit_rate@k que mide si al menos un item relevante está en top-k. Calcular para k = [1, 3, 5, 10].

6. **Descomposición SVD con diferentes factores**: Probar k = [2, 5, 10, 20, 50] en SVD. ¿Cómo cambia RMSE? ¿Hay overfitting con k grande?

7. **Cold start híbrido**: Combinar popularidad + categoría para nuevo producto: `score = α * popularidad + (1-α) * similitud_categoría`. Probar α = [0.2, 0.5, 0.8].

8. **Sistema híbrido user+item**: Implementar un predictor que combine user-based e item-based: `pred = β * pred_user + (1-β) * pred_item`. Encontrar β óptimo en validación.

---

## Resumen

| Técnica | Ventajas | Desventajas | Cuándo usarla |
|---------|----------|-------------|---------------|
| User-based CF | Simple, intuitiva | No escala bien (>10k usuarios), cold start usuario | Catálogo pequeño, comunidad similar |
| Item-based CF | Más estable que user-based | Cold start producto, necesita matriz de items | Tienda con muchos usuarios, pocos items |
| SVD (model-based) | Escala bien, generaliza | Menos interpretable, requiere ajuste | Grandes volúmenes, producción |
| Populares (cold start) | Siempre funciona | Baja personalización | Nuevos usuarios, primera interacción |

En ventas B2B, el item-based CF suele funcionar mejor porque los patrones de compra empresarial son más consistentes por industria/categoría. En B2C, user-based y SVD dan mejor personalización. La combinación híbrida de técnicas cubre la mayoría de casos de uso reales.
