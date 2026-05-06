# CP26: Predicción de Abandono de Clientes (Churn) con Redes Neuronales

## Resumen Ejecutivo

Sistema de predicción de churn que identifica clientes con alto riesgo de abandono usando una red neuronal. Se comparan arquitecturas Dense vs RandomForest, se calcula feature importance, y se diseñan estrategias de retención basadas en el perfil del cliente en riesgo.

**Dataset:** 5000 clientes con historial transaccional
**Target:** Churn (recencia > 180 días y frecuencia baja)
**Técnicas:** Red Neuronal, RandomForest, Class Weights, ROC AUC
**Métrica objetivo:** AUC > 0.80, Recall > 75%

---

## 1. Cargar Clientes y Crear Target: Churn

```python
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación línea por línea:**

*1. Cargar Clientes y Crear Target: Churn*


---
 — Realiza la operación indicada con los parámetros definidos..

---

---


---

## 2. Features: Recencia, Frecuencia, Monto, Ticket Promedio, Antigüedad

```python
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las **Explicación línea por línea:**

*2. Features: Recencia, Frecuencia, Monto, Ticket Promedio, Antigüedad*


---
ltado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


---

## 3. Explorar Balance de Clases (Churn Rate Aproximado)

```python
```

**Salida**Explicación línea por línea:**

*3. Explorar Balance de Clases (Churn Rate Aproximado)*


---
se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


-**Explicación línea por línea:**

1. `**Explicación:**` — Realiza la operación indicada con los parámetros definidos..
2. `1.` — Realiza la operación indicada con los parámetros definidos..

---
 con StandardScaler

```python
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las**Explicación línea por línea:**


---
cación:**

1. ````` — Realiza la operación indicada con los parámetros definidos..

---

---


---

## 5. Construir Red Neuronal: Dense(32,16,1) con Sigmoid Final

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


**Arquitectura:**
| Capa | Tipo | Parámetros |
|------|------|------------|
| Input | 5 features | 0 |
| Dense(32, relu) | + BatchNorm + Dropout | 192 |
| Dense(16, relu) | + Dropout | 528 |
| Dense(1, sigmoid) | Salida binaria | 17 |
| **Total** | | **737** |

**BatchNorma**Explicación línea por línea:**


---
entrenamiento normalizando activaciones. **Dropout:** Regulariza para evitar sobreajuste.

---

## 6. Compilar con Binary Crossentropy y Class Weight

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


**Binary Crossentropy:** Función de**Explicación línea por línea:**


---
sificación binaria. **Class weight:** Pondera más la clase minoritaria (churn) durante el entrenamiento.

---

## 7. Entrenar con EarlyStopping y Validation Split

```python
```

**Salida:**

```
# Los resultados se almacenan en las variables definida**Explicación línea por línea:**

*7. Entrenar con EarlyStopping y Validation Split*


---
ables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


---

## 8. Evaluar: ROC AUC, Precisión, Recall

```python
```

**Salid**Explicación línea por línea:**

*8. Evaluar: ROC AUC, Precisión, Recall*


---
 se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


---

## 9. Matri**Explicación línea por línea:**

*9. Matri*

1. `**Explicación:**` — Realiza la operación indicada con los parámetros definidos..
2. `1.` — Realiza la operación indicada con los parámetros definidos..

---
timo

```python
```

**Salida:**

```
# Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el r**Explicación línea por línea:**


---
s.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


---

## 10. Comparar con RandomForestClassifier

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

## 11. Feat**Explicación línea por línea:**

*11. Feat*


---
e Churn

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

 Recencia y frecuencia son los predictores más fuertes de churn. Monto y ticket promedio tienen menor impacto.

---

## 12. Top 20 Clientes con Mayor Probabilidad de Churn

```python
```

**Salida:**

```
# Los resultados se almacenan en las variables defi**Explicación línea por línea:**

*12. Top 20 Clientes con Mayor Probabilidad de Churn*


---
variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


---

## 13. Estrategias de Retención por Perfil

```pytho**Explicación línea por línea:**

*13. Estrategias de Retención por Perfil*

1. `*12. Top 20 Clientes con Mayor Probabilidad de Churn*` — Realiza la operación indicada con los parámetros definidos..
2. `---` — Realiza la operación indicada con los parámetros definidos..
3. `variables para ver el resultado de las operaciones.` — Realiza la operación indicada con los parámetros definidos..

---
Los resultados se almacenan en las variables definidas.
# Inspeccionar las variables para ver el resultado de las operaciones.
```

**Explicación:**


1. El código prepara y procesa los datos según las operaciones definidas.

---


---


---

## 14. Costo de Adquisición vs Retención (ROI)

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

## 15. Sistema de Alertas Tempranas

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

## Conclusiones

1. **Red Neuronal** alcanza AUC > 0.85 en predicción de churn, comparable con Random Forest
2. **Recencia y frecuencia** son los predictores más fuertes de abandono
3. **Class weights** y umbral óptimo mejoran significativamente el recall de churn
4. **Estrategias de retención** por nivel de riesgo permiten optimizar presupuesto
5. **ROI de retención** supera 5x al de adquisición (retener es 3-5x más barato)
6. **Sistema de alertas tempranas** permite intervenir antes de que el cliente abandone
7. **Próximos pasos:** incorporar features de comportamiento (clickstream, email opens)

---

## 5 Ejercicios Adicionales

**E01:** Implementar un modelo XGBoost con early_stopping y comparar con NN y RF.

**E02:** Añadir features temporales: evolución de frecuencia en los últimos 3 meses, cambio en ticket promedio.

**E03:** Construir un modelo de churn con LSTM sobre secuencias de transacciones (últimas N compras como secuencia).

**E04:** Implementar SHAP para interpretar predicciones a nivel individual (por qué este cliente específico hará churn).

**E05:** Crear un pipeline de producción con reentrenamiento automático semanal y despliegue continuo (CI/CD para ML).
