"""
Práctica 4 - IA: Clasificación Académica
Paso 3: Métodos de Validación y Tabla Comparativa de Resultados.

Restricción estricta: NO se utiliza scikit-learn ni bibliotecas similares.
Núcleo algorítmico: Python estándar + NumPy + Pandas.
"""

from __future__ import annotations

import time
from typing import Callable, Iterator

import numpy as np
import pandas as pd

from classifiers import (
    BaseClassifier,
    Distances,
    DistanceFunction,
    KNNClassifier,
    NearestCentroidClassifier,
)
from eda import COLUMN_NAMES, DATASET_URL, TARGET_COL, load_dataset

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

#: Función de fábrica que devuelve una nueva instancia del clasificador.
ClassifierFactory = Callable[[], BaseClassifier]

#: Par de arrays de índices (entrenamiento, prueba).
SplitPair = tuple[np.ndarray, np.ndarray]


# ---------------------------------------------------------------------------
# Normalización Z-score (desde cero, sin scikit-learn)
# ---------------------------------------------------------------------------

def standardize(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estandarización Z-score ajustada SOLO sobre el conjunto de entrenamiento.

    Aplica la misma transformación al conjunto de prueba para evitar
    data leakage: la media y desviación estándar provienen exclusivamente
    del split de entrenamiento de cada fold/partición.

    Fórmula: z = (x − μ_train) / σ_train   [ddof=1, muestral]

    Los atributos con σ=0 (varianza nula) se dejan sin transformar (σ←1)
    para evitar divisiones por cero.
    """
    mean: np.ndarray = np.mean(X_train, axis=0)
    std: np.ndarray  = np.std(X_train, axis=0, ddof=1)
    std[std == 0.0]  = 1.0                          # columnas constantes

    X_train_z: np.ndarray = (X_train - mean) / std
    X_test_z:  np.ndarray = (X_test  - mean) / std
    return X_train_z, X_test_z


# ---------------------------------------------------------------------------
# Métricas de evaluación
# ---------------------------------------------------------------------------

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Exactitud (Accuracy): fracción de predicciones correctas.

    acc = #{y_true_i == y_pred_i} / n
    """
    return float(np.sum(y_true == y_pred) / len(y_true))


# ---------------------------------------------------------------------------
# Métodos de validación implementados desde cero
# ---------------------------------------------------------------------------

def holdout_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.70,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Partición Hold-Out aleatoria ESTRATIFICADA (70 % train / 30 % test).

    La estratificación garantiza que la distribución de clases en ambas
    particiones refleje la del conjunto original, lo cual es especialmente
    relevante cuando las clases no están perfectamente balanceadas
    (Wine: 59 / 71 / 48 muestras por clase).

    Algoritmo:
        Para cada clase c:
            1. Obtener índices de c y mezclarlos aleatoriamente.
            2. Tomar los primeros ⌊n_c × train_ratio⌋ para train.
            3. El resto va a test.

    Args:
        X          : Matriz de características (n, d).
        y          : Vector de etiquetas (n,).
        train_ratio: Proporción de muestras para entrenamiento. Default 0.70.
        seed       : Semilla para reproducibilidad.

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    test_idx:  list[int] = []

    for cls in np.unique(y):
        cls_idx: np.ndarray = rng.permutation(np.where(y == cls)[0])
        cut: int = max(1, int(len(cls_idx) * train_ratio))
        train_idx.extend(cls_idx[:cut].tolist())
        test_idx.extend(cls_idx[cut:].tolist())

    tr = np.array(train_idx)
    te = np.array(test_idx)
    return X[tr], X[te], y[tr], y[te]


def kfold_indices(
    n: int,
    k: int = 10,
    seed: int = 42,
) -> list[SplitPair]:
    """
    Genera los índices de partición para K-Fold Cross-Validation.

    Algoritmo:
        1. Mezclar los n índices aleatoriamente.
        2. Dividir en k partes con np.array_split (las últimas partes pueden
           tener un elemento más si n no es divisible entre k).
        3. Para el fold i: test = parte i,  train = concatenación del resto.

    Args:
        n   : Número total de muestras.
        k   : Número de folds. Default 10.
        seed: Semilla aleatoria para reproducibilidad.

    Returns:
        Lista de k tuplas (train_indices, test_indices).
    """
    rng = np.random.default_rng(seed)
    indices: np.ndarray  = rng.permutation(n)
    folds: list[np.ndarray] = np.array_split(indices, k)

    splits: list[SplitPair] = []
    for i in range(k):
        test_idx  = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        splits.append((train_idx, test_idx))
    return splits


def loo_indices(n: int) -> Iterator[SplitPair]:
    """
    Generador de índices para Leave-One-Out Cross-Validation (LOO).

    En cada una de las n iteraciones, exactamente UNA muestra es el
    conjunto de prueba y las n-1 restantes forman el conjunto de
    entrenamiento.

    Complejidad temporal: O(n²) evaluaciones en total.
    Complejidad espacial: O(n) por iteración (no materializa todo a la vez).

    Args:
        n: Número total de muestras.

    Yields:
        (train_indices, test_indices) con len(test_indices) == 1.
    """
    all_idx: np.ndarray = np.arange(n)
    for i in range(n):
        test_idx  = all_idx[i : i + 1]                           # shape (1,)
        train_idx = np.concatenate([all_idx[:i], all_idx[i + 1:]])
        yield train_idx, test_idx


# ---------------------------------------------------------------------------
# Funciones de evaluación (una por método de validación)
# ---------------------------------------------------------------------------

def evaluate_holdout(
    factory: ClassifierFactory,
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.70,
    seed: int = 42,
) -> float:
    """Evalúa un clasificador con Hold-Out estratificado. Retorna accuracy."""
    X_tr, X_te, y_tr, y_te = holdout_split(X, y, train_ratio, seed)
    X_tr_z, X_te_z = standardize(X_tr, X_te)
    clf = factory()
    clf.fit(X_tr_z, y_tr)
    return accuracy(y_te, clf.predict(X_te_z))


def evaluate_kfold(
    factory: ClassifierFactory,
    X: np.ndarray,
    y: np.ndarray,
    k: int = 10,
    seed: int = 42,
) -> float:
    """
    Evalúa un clasificador con K-Fold CV.

    Retorna la accuracy media sobre los k folds.
    La normalización se reajusta en cada fold para evitar data leakage.
    """
    fold_accs: list[float] = []
    for train_idx, test_idx in kfold_indices(len(y), k, seed):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        X_tr_z, X_te_z = standardize(X_tr, X_te)
        clf = factory()
        clf.fit(X_tr_z, y_tr)
        fold_accs.append(accuracy(y_te, clf.predict(X_te_z)))
    return float(np.mean(fold_accs))


def evaluate_loo(
    factory: ClassifierFactory,
    X: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    Evalúa un clasificador con Leave-One-Out CV.

    Retorna accuracy global: (# predicciones correctas) / n.

    La normalización se recalcula en cada una de las n iteraciones
    usando solo los n-1 puntos de entrenamiento (sin data leakage).
    """
    n: int   = len(y)
    hits: int = 0
    for train_idx, test_idx in loo_indices(n):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        X_tr_z, X_te_z = standardize(X_tr, X_te)
        clf = factory()
        clf.fit(X_tr_z, y_tr)
        if clf.predict(X_te_z)[0] == y_te[0]:
            hits += 1
    return hits / n


# ---------------------------------------------------------------------------
# Cuadrícula de experimentos
# ---------------------------------------------------------------------------

def build_experiment_grid() -> list[dict]:
    """
    Genera la lista de todos los experimentos a ejecutar.

    Combinaciones:
        Clasificadores  : Centroide + KNN (k ∈ {1,3,5,7,9,11})  → 7 clasificadores
        Distancias      : Euclidiana, Manhattan, Chebyshev, Coseno → 4 métricas
        Validaciones    : Hold-Out, 10-Fold CV, LOO                → 3 métodos
        ─────────────────────────────────────────────────────────
        Total de filas  : 7 × 4 × 3 = 84 experimentos

    Returns:
        Lista de dicts con claves: 'algo_name', 'dist_name', 'factory'.
    """
    distance_map: dict[str, DistanceFunction] = {
        "Euclidiana": Distances.euclidean,
        "Manhattan":  Distances.manhattan,
        "Chebyshev":  Distances.chebyshev,
        "Coseno":     Distances.cosine,
    }

    experiments: list[dict] = []

    for dist_name, dist_fn in distance_map.items():
        # Clasificador de Centroide
        experiments.append({
            "algo_name": "Centroide",
            "dist_name": dist_name,
            # Captura explícita de dist_fn para evitar closure tardía
            "factory": (lambda d=dist_fn: NearestCentroidClassifier(distance_fn=d)),
        })
        # KNN para cada k soportado
        for k in sorted(KNNClassifier.SUPPORTED_K):
            experiments.append({
                "algo_name": f"KNN (k={k:2d})",
                "dist_name": dist_name,
                "factory": (lambda d=dist_fn, kk=k: KNNClassifier(k=kk, distance_fn=d)),
            })

    return experiments


# ---------------------------------------------------------------------------
# Formateador de tabla
# ---------------------------------------------------------------------------

def print_table(df: pd.DataFrame, title: str) -> None:
    """Imprime un DataFrame con bordes simples y título centrado."""
    sep = "─" * 72
    print(f"\n{'─' * 72}")
    print(f"  {title}")
    print(sep)
    print(df.to_string(index=False))
    print(sep)


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def main() -> None:

    # ── 1. Cargar dataset ────────────────────────────────────────────────
    print("\nCargando dataset Wine desde UCI...")
    df_raw: pd.DataFrame = load_dataset(DATASET_URL, COLUMN_NAMES)
    X: np.ndarray = df_raw.drop(columns=[TARGET_COL]).to_numpy(dtype=float)
    y: np.ndarray = df_raw[TARGET_COL].to_numpy()
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    print(
        f"  {n_samples} muestras  ×  {n_features} atributos  |  "
        f"{n_classes} clases: {np.unique(y).tolist()}"
    )

    # ── 2. Construir cuadrícula de experimentos ──────────────────────────
    experiments = build_experiment_grid()
    n_experiments = len(experiments)

    # Evaluadores: nombre → función parcial
    evaluators: dict[str, Callable] = {
        "Hold-Out (70/30)": evaluate_holdout,
        "10-Fold CV":       evaluate_kfold,
        "LOO":              evaluate_loo,
    }

    # ── 3. Ejecutar todos los experimentos ───────────────────────────────
    results: list[dict] = []
    t_global = time.perf_counter()

    for val_name, eval_fn in evaluators.items():
        t_val = time.perf_counter()
        print(f"\n[{val_name}]  ejecutando {n_experiments} combinaciones ...")

        for i, exp in enumerate(experiments, start=1):
            acc = eval_fn(exp["factory"], X, y)
            results.append({
                "Algoritmo":  exp["algo_name"],
                "Distancia":  exp["dist_name"],
                "Validación": val_name,
                "Exactitud":  acc,
            })
            # Progreso en línea cada 7 experimentos (un "bloque" de distancia)
            if i % 7 == 0 or i == n_experiments:
                elapsed = time.perf_counter() - t_val
                print(
                    f"  {i:2d}/{n_experiments} completados  "
                    f"({elapsed:.1f}s transcurridos)",
                    end="\r",
                    flush=True,
                )

        elapsed_val = time.perf_counter() - t_val
        print(f"  {n_experiments}/{n_experiments} completados  "
              f"({elapsed_val:.1f}s)          ")

    total_time = time.perf_counter() - t_global
    print(f"\nTiempo total de evaluación: {total_time:.1f}s")

    # ── 4. Construir DataFrame de resultados ─────────────────────────────
    df_res = pd.DataFrame(results)
    df_res["Exactitud (%)"] = (df_res["Exactitud"] * 100).round(2)

    # ── 5. Tabla completa ────────────────────────────────────────────────
    print_table(
        df_res[["Algoritmo", "Distancia", "Validación", "Exactitud (%)"]],
        title="TABLA COMPARATIVA DE RESULTADOS — Dataset Wine (UCI)",
    )

    # ── 6. Top 10 configuraciones ────────────────────────────────────────
    top10 = (
        df_res
        .sort_values("Exactitud", ascending=False)
        .head(10)[["Algoritmo", "Distancia", "Validación", "Exactitud (%)"]]
        .reset_index(drop=True)
    )
    top10.index += 1
    print_table(top10, title="TOP 10 CONFIGURACIONES")

    # ── 7. Mejor configuración por método de validación ──────────────────
    best_per_val = (
        df_res
        .loc[df_res.groupby("Validación")["Exactitud"].idxmax()]
        .sort_values("Exactitud", ascending=False)
        [["Validación", "Algoritmo", "Distancia", "Exactitud (%)"]]
        .reset_index(drop=True)
    )
    print_table(best_per_val, title="MEJOR CONFIGURACIÓN POR MÉTODO DE VALIDACIÓN")

    # ── 8. Accuracy media por algoritmo (promediando distancias y métodos) ──
    mean_by_algo = (
        df_res
        .groupby("Algoritmo")["Exactitud (%)"]
        .mean()
        .round(2)
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"Exactitud (%)": "Exactitud media (%)"})
    )
    print_table(mean_by_algo, title="EXACTITUD MEDIA POR ALGORITMO (promedio sobre distancias y métodos)")

    # ── 9. Accuracy media por distancia ──────────────────────────────────
    mean_by_dist = (
        df_res
        .groupby("Distancia")["Exactitud (%)"]
        .mean()
        .round(2)
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"Exactitud (%)": "Exactitud media (%)"})
    )
    print_table(mean_by_dist, title="EXACTITUD MEDIA POR MÉTRICA DE DISTANCIA")


if __name__ == "__main__":
    main()
