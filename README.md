# Práctica 4 — IA: Clasificación Académica

Implementación **desde cero** de clasificadores de Machine Learning y métodos de validación, sin usar scikit-learn ni bibliotecas similares para el núcleo algorítmico.

![CI](https://github.com/AuroraCarbajal03/Practica4_IA/actions/workflows/python-app.yml/badge.svg)

---

## Dataset

**Wine** — UCI Machine Learning Repository  
178 muestras · 13 atributos químicos · 3 clases (vinos italianos)

| Clase | Muestras | % |
|-------|----------|---|
| Clase 1 | 59 | 33.1 % |
| Clase 2 | 71 | 39.9 % |
| Clase 3 | 48 | 27.0 % |

---

## Estructura del proyecto

```
Practica4_IA/
├── eda.py               # Análisis Exploratorio de Datos
├── classifiers.py       # Distancias y clasificadores desde cero
├── main.py              # Validación cruzada y tabla comparativa
├── test_classifiers.py  # Suite de 71 pruebas unitarias (pytest)
├── requirements.txt     # Dependencias del proyecto
└── .github/
    └── workflows/
        └── python-app.yml  # Pipeline CI/CD (GitHub Actions)
```

---

## Implementaciones

### Métricas de distancia (`classifiers.py`)

| Métrica | Fórmula | Familia |
|---------|---------|---------|
| Euclidiana | $\sqrt{\sum(a_i - b_i)^2}$ | Minkowski (p=2) |
| Manhattan | $\sum\|a_i - b_i\|$ | Minkowski (p=1) |
| Chebyshev | $\max\|a_i - b_i\|$ | Minkowski (p→∞) |
| Coseno | $1 - \frac{a \cdot b}{\|a\|\|b\|}$ | — (angular) |

### Clasificadores

- **NearestCentroidClassifier** — asigna cada muestra a la clase cuyo centroide sea más cercano
- **KNNClassifier** — k vecinos más cercanos con k ∈ {1, 3, 5, 7, 9, 11}; desempate por distancia acumulada

Ambos aceptan cualquier función de distancia como parámetro.

### Métodos de validación (`main.py`)

| Método | Descripción |
|--------|-------------|
| Hold-Out (70/30) | Partición estratificada por clase |
| 10-Fold CV | Validación cruzada con 10 pliegues |
| LOO | Leave-One-Out — n iteraciones, 1 muestra de prueba por iteración |

La normalización Z-score se recalcula en cada fold usando **solo los datos de entrenamiento** (sin data leakage).

---

## Resultados

84 combinaciones evaluadas: 7 clasificadores × 4 distancias × 3 métodos de validación.

| Método | Mejor configuración | Exactitud |
|--------|-------------------|-----------|
| Hold-Out (70/30) | KNN k=3, Euclidiana | 100.00 % |
| LOO | KNN k=11, Manhattan | 98.88 % |
| 10-Fold CV | KNN k=11, Manhattan | 98.33 % |

**Exactitud media por métrica de distancia:**

| Distancia | Media |
|-----------|-------|
| Manhattan | 97.88 % |
| Euclidiana | 97.43 % |
| Coseno | 97.02 % |
| Chebyshev | 92.78 % |

---

## Instalación y ejecución

```bash
# 1. Clonar el repositorio
git clone https://github.com/AuroraCarbajal03/Practica4_IA.git
cd Practica4_IA

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Análisis exploratorio (genera scatter_plots.png y boxplots.png)
python eda.py

# 4. Pipeline principal — tabla comparativa completa
python main.py

# 5. Pruebas unitarias
pytest test_classifiers.py -v
```

---

## Pruebas unitarias

71 tests organizados en 7 clases:

- `TestDistances` — valores conocidos + propiedades matemáticas (no negatividad, simetría)
- `TestNearestCentroidClassifier` — fit/predict, centroides, manejo de errores
- `TestKNNClassifier` — todos los k válidos, 1NN perfecto en train, `predict_proba`
- `TestHoldoutSplit` — tamaños, sin solapamiento, reproducibilidad
- `TestKFoldIndices` — cobertura completa, sin repetición entre folds
- `TestLOOIndices` — n iteraciones, test = 1 muestra exactamente
- `TestAccuracy` / `TestStandardize` — casos extremos y ausencia de data leakage

---

## CI/CD — GitHub Actions

El pipeline se ejecuta automáticamente en cada `push` o `pull request`:

```
Checkout → Python 3.11 → pip install → pytest → python main.py
```

---

## Tecnologías

- **Python 3.11+**
- **NumPy** — operaciones matriciales y cálculo de distancias
- **Pandas** — carga de datos y tabla de resultados
- **Matplotlib / Seaborn** — visualizaciones EDA únicamente
- **pytest** — suite de pruebas unitarias
- **GitHub Actions** — integración continua

> **Restricción del proyecto:** scikit-learn está **prohibido** para cualquier componente del núcleo algorítmico (clasificadores, distancias, validación cruzada).

---

## Autora

**Itzel Aurora Carbajal Martínez**
