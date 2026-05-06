# CP30: Sistema de Recomendación Híbrido B2B con NCF + Content-Based

## Contexto
Recomendar productos a compradores mayoristas usando deep learning. Sistema híbrido que combina Neural Collaborative Filtering (NCF) con contenido basado en descripciones para maximizar relevancia en un entorno B2B.

---

## 1. Cargar transacciones B2B (cliente → productos)

```python
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación línea por línea:**

*1. Cargar transacciones B2B (cliente → productos)*


---
 — Realiza la operación indicada con los parámetros definidos..

---

---


---

## 2. Crear matriz usuario-producto implícita (compras = 1)

```python
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspecci**Explicación línea por línea:**

*2. Crear matriz usuario-producto implícita (compras = 1)*


---
r el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


---

## 3. Construir modelo NCF (GMF + MLP) con TensorFlow

```python
```

**Salida:**
**Explicación línea por línea:**

*3. Construir modelo NCF (GMF + MLP) con TensorFlow*


---
lmacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


---

## 4. Entrenar con nega**Explicación línea por línea:**

*4. Entrenar con nega*

1. `**Explicación:**` — Realiza la operación indicada con los parámetros definidos..
2. `1.` — Realiza la operación indicada con los parámetros definidos..

---
o comprados)

```python
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccion**Explicación línea por línea:**


---
el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


---

## 5. Evaluar: Hit Rate@k y NDCG@k

```python
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las var**Explicación línea por línea:**

*5. Evaluar: Hit Rate@k y NDCG@k*


---
do de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


---

## 6. Content-based: TF-IDF de descripciones de productos

```python
```

**Salida:**

```
# Los result**Explicación línea por línea:**

*6. Content-based: TF-IDF de descripciones de productos*


---
ariables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


---

## 7. Content-based: **Explicación línea por línea:**

*7. Content-based:*

1. `*6. Content-based: TF-IDF de descripciones de productos*` — Realiza la operación indicada con los parámetros definidos..
2. `---` — Realiza la operación indicada con los parámetros definidos..
3. `ariables definidas.` — Realiza la operación indicada con los parámetros definidos..

---
n
```

**Salida:**

```
# Los resultados se alma**Explicación línea por línea:**

1. `**Salida:**` — Realiza la operación indicada con los parámetros definidos..

---
inidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


---

## 8. Híbrido: NCF score + CB score ponderad**Explicación línea por línea:**

*8. Híbrido: NCF score + CB score ponderad*

1. `**Explicación:**` — Realiza la operación indicada con los parámetros definidos..
2. `1.` — Realiza la operación indicada con los parámetros definidos..

---
a:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ve**Explicación línea por línea:**

1. `**Explicación:**` — Realiza la operación indicada con los parámetros definidos..
2. `1.` — Realiza la operación indicada con los parámetros definidos..

---
aciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


---

## 9. Cold start: producto nuevo usa CB, cliente nuevo usa**Explicación línea por línea:**

*9. Cold start: producto nuevo usa CB, cliente nuevo usa*

1. `1. `**Explicación:**` — Realiza la operación indicada con los parámetros definidos..` — Realiza la operación indicada con los parámetros definidos..
2. `2. `1.` — Realiza la operación indicada con los parámetros definidos..` — Realiza la operación indicada con los parámetros definidos..
3. `---` — Ej**Explicación línea por línea:**

1. `1. `**Explicación:**` — Realiza la operación indicada con los parámetros definidos..` — Realiza la operación indicada con los parámetros definidos..
2. `2. `1.` — Realiza la operación indicada con los parámetros definidos..` — Realiza la operación indicada con los parámetros definidos..
3. `---` — Realiza la operación indicada con los parámetros definidos..
4. `aciones.` — Ejecuta la operació**Explicación línea por línea:**

1. `1. `**Explicación:**` — Realiza la operación indicada con los parámetros definidos..` — Realiza la operación indicada con los parámetros definidos..
2. `2. `1.` — Realiza la operación indicada con los parámetros definidos..` — Realiza la operación indicada con los parámetros definidos..
3. `---` — Realiza la operación indicada con los parámetros definidos..
4. `aciones.` — Ej**Explicación línea por línea:**

1. `1. `**Explicación:**` — Realiza la operación indicada con los parámetros definidos..` — Realiza la operación indicada con los parámetros definidos..
2. `2. `1.` — Realiza la operación indicada con los parámetros definidos..` — Realiza la operación indicada con los parámetros definidos..
3. `---` — Realiza la operación indicada con los parámetros definidos..
4. `aciones.` — Ejecuta**Explicación línea por línea:**

1. `1. `**Explicación:**` — Realiza la operación indicada con los parámetros definidos..` — Realiza la operación indicada con los parámetros definidos..
2. `2. `1.` — Realiza la operación indicada con los parámetros definidos..` — Realiza la operación indicada con los parámetros definidos..
3. `---` — Ejecuta l**Explicación línea por línea:**

1. `1. `**Explicación:**` — Realiza la operación indicada con los parámetros definidos..` — Realiza la operación indicada con los parámetros definidos..
2. `2. `1.` — Realiza la operación indicada con los parámetros definidos..` — Realiza la operación indicada con los parámetros definidos..
3. `---` — Realiza la operación indicada con los parámetros definidos..
4. `aciones.` — Realiza la operación indicada con los parámetros definidos..

---
4. `aciones.` — Realiza la operación indicada con los parámetros definidos..

---
.

---
icada.

---
.
4. `aciones.` — Realiza la operación indicada con los parámetros definidos..

---
*Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


---

## 10. Generar top-10 recomendaciones para un cliente

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


---

## 11. Evaluar hit rate del sistema híbrido

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


---

## 12. A/B test: comparar NCF vs híbrido en CTR

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


---

## 13. Dockerizar el sistema de recomendación

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


---

## 14. API para obtener recomendaciones en tiempo real

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


---

## 15. Recomendaciones de negocio B2B

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


## Código completo del sistema de recomendación

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

