"""
Práctica 4 - IA: Clasificación Académica
Paso 2: Funciones de Distancia y Clasificadores implementados desde cero.

Restricción estricta: NO se utiliza scikit-learn ni bibliotecas similares
para ningún componente del núcleo algorítmico.
Dependencias permitidas: Python estándar + NumPy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

#: Firma de cualquier función de distancia entre dos vectores 1-D.
DistanceFunction = Callable[[np.ndarray, np.ndarray], float]


# ---------------------------------------------------------------------------
# Módulo de Distancias
# ---------------------------------------------------------------------------

class Distances:
    """
    Colección estática de métricas de distancia entre vectores 1-D.

    Todas las funciones asumen vectores de igual longitud y devuelven
    un escalar float no negativo.

    Distancias de la familia Minkowski (parametrizadas por p):
        - Euclidiana  → p = 2
        - Manhattan   → p = 1
        - Chebyshev   → p → ∞

    Distancia fuera de la familia Minkowski:
        - Coseno (basada en ángulo, no en norma)
    """

    # ------------------------------------------------------------------
    @staticmethod
    def euclidean(a: np.ndarray, b: np.ndarray) -> float:
        """
        Distancia Euclidiana — norma L2.

        d(a, b) = sqrt( Σ (aᵢ − bᵢ)² )

        Propiedades: simétrica, cumple desigualdad triangular.
        Sensible a diferencias de escala entre atributos → requiere
        normalización previa cuando los rangos difieren mucho.
        """
        diff: np.ndarray = a - b
        # np.dot es más rápido que np.sum(diff**2) para vectores 1-D
        return float(np.sqrt(np.dot(diff, diff)))

    # ------------------------------------------------------------------
    @staticmethod
    def manhattan(a: np.ndarray, b: np.ndarray) -> float:
        """
        Distancia Manhattan — norma L1 (distancia "taxicab").

        d(a, b) = Σ |aᵢ − bᵢ|

        Más robusta que Euclidiana ante valores atípicos (no eleva
        al cuadrado las diferencias).
        """
        return float(np.sum(np.abs(a - b)))

    # ------------------------------------------------------------------
    @staticmethod
    def chebyshev(a: np.ndarray, b: np.ndarray) -> float:
        """
        Distancia Chebyshev — norma L∞.

        d(a, b) = max |aᵢ − bᵢ|

        Equivalente al límite de la norma de Minkowski cuando p → ∞.
        Captura únicamente la dimensión con mayor diferencia absoluta.
        """
        return float(np.max(np.abs(a - b)))

    # ------------------------------------------------------------------
    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        """
        Distancia Coseno — NO pertenece a la familia Minkowski.

        d(a, b) = 1 − ( a·b ) / ( ‖a‖₂ · ‖b‖₂ )

        Rango resultante: [0, 2].
        Mide divergencia angular entre vectores; es invariante a escala
        (dos vectores paralelos tienen d = 0 independientemente de su
        magnitud), lo que la hace adecuada para datos de frecuencia o
        de texto.

        Caso degenerado: vector cero → se retorna 1.0 (incertidumbre
        máxima; el ángulo no está definido).
        """
        norm_a: float = float(np.linalg.norm(a))
        norm_b: float = float(np.linalg.norm(b))

        if norm_a == 0.0 or norm_b == 0.0:
            return 1.0

        cosine_sim: float = float(np.dot(a, b)) / (norm_a * norm_b)
        # Clamp numérico estricto para evitar dominio inválido de arccos
        cosine_sim = max(-1.0, min(1.0, cosine_sim))
        return 1.0 - cosine_sim


# ---------------------------------------------------------------------------
# Clase base abstracta — interfaz común de los clasificadores
# ---------------------------------------------------------------------------

class BaseClassifier(ABC):
    """
    Contrato fit / predict compartido por todos los clasificadores.

    No puede instanciarse directamente. Los subtipos deben implementar
    fit() y predict().
    """

    _fitted: bool = False  # bandera de estado

    # ------------------------------------------------------------------
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseClassifier:
        """
        Entrena (o memoriza) el modelo con los datos proporcionados.

        Args:
            X: Matriz de características — forma (n_muestras, n_atributos).
            y: Vector de etiquetas      — forma (n_muestras,).

        Returns:
            self, para permitir encadenamiento: clf.fit(X, y).predict(X_test)
        """

    # ------------------------------------------------------------------
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice la clase de cada muestra en X.

        Args:
            X: Matriz de muestras — forma (n_muestras, n_atributos).

        Returns:
            Array de etiquetas predichas — forma (n_muestras,).
        """

    # ------------------------------------------------------------------
    def _check_is_fitted(self) -> None:
        """Lanza RuntimeError si se intenta predecir antes de entrenar."""
        if not self._fitted:
            raise RuntimeError(
                f"[{self.__class__.__name__}] El modelo no ha sido entrenado. "
                "Llama a .fit(X, y) antes de llamar a .predict(X)."
            )

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        dist_name: str = getattr(self.distance_fn, "__name__", str(self.distance_fn))  # type: ignore[attr-defined]
        return (
            f"{self.__class__.__name__}("
            f"distance_fn={dist_name}, "
            f"fitted={self._fitted})"
        )


# ---------------------------------------------------------------------------
# Clasificador 1 — Distancia Mínima al Centroide (Nearest Centroid)
# ---------------------------------------------------------------------------

class NearestCentroidClassifier(BaseClassifier):
    """
    Clasificador de Distancia Mínima al Centroide.

    Entrenamiento: calcula el centroide (media aritmética) de cada clase.
    Predicción   : asigna la muestra a la clase cuyo centroide sea más
                   cercano según la métrica configurada.

    Complejidades:
        fit    → O(n · d)          n = muestras, d = dimensiones
        predict → O(m · C · d)    m = muestras test, C = clases

    Args:
        distance_fn: Métrica de distancia. Por defecto Euclidiana.
                     Puede ser cualquier callable de Distances o una
                     función externa con firma (np.ndarray, np.ndarray) → float.
    """

    def __init__(
        self,
        distance_fn: DistanceFunction = Distances.euclidean,
    ) -> None:
        self.distance_fn: DistanceFunction = distance_fn
        self._centroids: dict[object, np.ndarray] = {}
        self._classes: np.ndarray = np.array([])
        self._fitted: bool = False

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> NearestCentroidClassifier:
        """
        Calcula un centroide por clase como la media aritmética de sus muestras.

        Args:
            X: Matriz de características (n_muestras, n_atributos).
            y: Etiquetas de clase        (n_muestras,).

        Returns:
            self
        """
        self._classes = np.unique(y)
        self._centroids = {}
        for cls in self._classes:
            mask: np.ndarray = y == cls
            # Media vectorial: shape (n_atributos,)
            self._centroids[cls] = np.mean(X[mask], axis=0)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Asigna a cada muestra la clase del centroide más cercano.

        Args:
            X: Muestras a clasificar (n_muestras, n_atributos).

        Returns:
            Etiquetas predichas (n_muestras,).
        """
        self._check_is_fitted()
        predictions: list[object] = []

        for sample in X:
            best_cls: object = None
            best_dist: float = float("inf")
            for cls, centroid in self._centroids.items():
                dist = self.distance_fn(sample, centroid)
                if dist < best_dist:
                    best_dist = dist
                    best_cls = cls
            predictions.append(best_cls)

        return np.array(predictions)

    # ------------------------------------------------------------------
    def get_centroids(self) -> dict[object, np.ndarray]:
        """
        Devuelve una copia del diccionario {clase → centroide}.

        Útil para inspección y visualización post-entrenamiento.
        """
        self._check_is_fitted()
        return {cls: centroid.copy() for cls, centroid in self._centroids.items()}


# ---------------------------------------------------------------------------
# Clasificador 2 — K-Nearest Neighbors (KNN)
# ---------------------------------------------------------------------------

class KNNClassifier(BaseClassifier):
    """
    Clasificador K-Nearest Neighbors implementado desde cero.

    Es un clasificador perezoso (lazy learning): no construye un modelo
    explícito durante el entrenamiento; simplemente memoriza los datos y
    calcula distancias en tiempo de inferencia.

    Valores de k soportados: 1, 3, 5, 7, 9, 11
        k=1  → 1NN: clasifica según el vecino más cercano.
        k impar → evita empates en problemas binarios.

    Desempate de votos (tie-breaking):
        Cuando dos o más clases obtienen el mismo número de votos entre
        los k vecinos, se elige la clase cuya suma de distancias a sus
        vecinos representativos sea menor (vecindad más compacta). Esto
        favorece clases cuyos ejemplos cercanos están más "apretados"
        alrededor de la muestra de consulta.

    Complejidades:
        fit    → O(n · d)    (copia de datos)
        predict → O(m · n · d + n log n)  por muestra

    Args:
        k           : Número de vecinos. Debe pertenecer a SUPPORTED_K.
        distance_fn : Métrica de distancia. Por defecto Euclidiana.
    """

    SUPPORTED_K: frozenset[int] = frozenset({1, 3, 5, 7, 9, 11})

    def __init__(
        self,
        k: int = 5,
        distance_fn: DistanceFunction = Distances.euclidean,
    ) -> None:
        if k not in self.SUPPORTED_K:
            raise ValueError(
                f"k={k} no está soportado. "
                f"Valores válidos: {sorted(self.SUPPORTED_K)}"
            )
        self.k: int = k
        self.distance_fn: DistanceFunction = distance_fn
        self._X_train: np.ndarray = np.empty((0,))
        self._y_train: np.ndarray = np.empty((0,))
        self._fitted: bool = False

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> KNNClassifier:
        """
        Memoriza el conjunto de entrenamiento.

        KNN no construye ningún modelo: el "entrenamiento" es sólo guardar
        los datos para usarlos durante la predicción.

        Args:
            X: Características de entrenamiento (n_muestras, n_atributos).
            y: Etiquetas de entrenamiento       (n_muestras,).

        Returns:
            self
        """
        self._X_train = X.copy()
        self._y_train = y.copy()
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    def _distances_to_all(self, sample: np.ndarray) -> np.ndarray:
        """
        Calcula la distancia de una muestra a todos los puntos de entrenamiento.

        Returns:
            Array de distancias de longitud n_muestras_entrenamiento.
        """
        return np.array(
            [self.distance_fn(sample, x_tr) for x_tr in self._X_train]
        )

    # ------------------------------------------------------------------
    def _majority_vote(
        self,
        k_labels: np.ndarray,
        k_distances: np.ndarray,
    ) -> object:
        """
        Determina la clase ganadora por voto de mayoría simple.

        En caso de empate se resuelve por compacidad: gana la clase
        cuya suma de distancias a sus vecinos representativos es menor.

        Args:
            k_labels   : Etiquetas de los k vecinos más cercanos.
            k_distances: Distancias correspondientes a esos k vecinos.

        Returns:
            Etiqueta de clase ganadora.
        """
        counts: Counter = Counter(k_labels.tolist())
        max_votes: int = max(counts.values())
        tied: list[object] = [
            cls for cls, cnt in counts.items() if cnt == max_votes
        ]

        if len(tied) == 1:
            return tied[0]

        # Desempate: menor distancia acumulada entre sus vecinos
        cum_dist: dict[object, float] = {
            cls: float(np.sum(k_distances[k_labels == cls]))
            for cls in tied
        }
        return min(cum_dist, key=lambda c: cum_dist[c])

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice la clase de cada muestra por votación de los k vecinos más cercanos.

        Args:
            X: Muestras de prueba (n_muestras, n_atributos).

        Returns:
            Etiquetas predichas (n_muestras,).
        """
        self._check_is_fitted()
        predictions: list[object] = []

        for sample in X:
            all_dists: np.ndarray = self._distances_to_all(sample)
            # Índices de los k vecinos más cercanos (argsort es O(n log n))
            k_idx: np.ndarray = np.argsort(all_dists)[: self.k]
            k_labels: np.ndarray = self._y_train[k_idx]
            k_dists: np.ndarray = all_dists[k_idx]

            winner = self._majority_vote(k_labels, k_dists)
            predictions.append(winner)

        return np.array(predictions)

    # ------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> dict[object, np.ndarray]:
        """
        Devuelve la proporción de votos por clase para cada muestra.

        Útil para análisis de confianza y curvas ROC.

        Returns:
            Dict {clase: array(n_muestras,)} con proporciones en [0, 1].
        """
        self._check_is_fitted()
        classes: np.ndarray = np.unique(self._y_train)
        proba: dict[object, list[float]] = {cls: [] for cls in classes}

        for sample in X:
            all_dists = self._distances_to_all(sample)
            k_idx = np.argsort(all_dists)[: self.k]
            k_labels = self._y_train[k_idx]
            vote_counts: Counter = Counter(k_labels.tolist())
            for cls in classes:
                proba[cls].append(vote_counts.get(cls, 0) / self.k)

        return {cls: np.array(votes) for cls, votes in proba.items()}

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        dist_name = getattr(self.distance_fn, "__name__", str(self.distance_fn))
        return (
            f"KNNClassifier("
            f"k={self.k}, "
            f"distance_fn={dist_name}, "
            f"fitted={self._fitted})"
        )


# ---------------------------------------------------------------------------
# Smoke-test rápido (ejecutar directamente: python classifiers.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Smoke test: Distances ===")
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 6.0, 3.0])

    print(f"  Euclidiana : {Distances.euclidean(a, b):.4f}")   # esperado ≈ 5.0
    print(f"  Manhattan  : {Distances.manhattan(a, b):.4f}")   # esperado = 7.0
    print(f"  Chebyshev  : {Distances.chebyshev(a, b):.4f}")   # esperado = 4.0
    print(f"  Coseno     : {Distances.cosine(a, b):.4f}")      # esperado ≈ 0.0977

    # Dataset sintético de 3 clases bien separadas
    rng = np.random.default_rng(42)
    X_train = np.vstack([
        rng.normal(loc=[0, 0], scale=0.5, size=(20, 2)),   # clase 0
        rng.normal(loc=[5, 5], scale=0.5, size=(20, 2)),   # clase 1
        rng.normal(loc=[0, 5], scale=0.5, size=(20, 2)),   # clase 2
    ])
    y_train = np.array([0] * 20 + [1] * 20 + [2] * 20)

    X_test = np.array([[0.1, 0.1], [4.9, 5.0], [0.0, 4.9]])
    y_expected = np.array([0, 1, 2])

    print("\n=== Smoke test: NearestCentroidClassifier ===")
    for dist_name, dist_fn in [
        ("euclidean", Distances.euclidean),
        ("manhattan", Distances.manhattan),
        ("chebyshev", Distances.chebyshev),
    ]:
        nc = NearestCentroidClassifier(distance_fn=dist_fn)
        preds = nc.fit(X_train, y_train).predict(X_test)
        ok = "OK" if np.array_equal(preds, y_expected) else "FAIL"
        print(f"  [{ok}] distance={dist_name:12s} → pred={preds}  expected={y_expected}")

    # Coseno se valida con datos cuya separabilidad es ANGULAR (no posicional).
    # Un dataset con centroide cercano al origen produce vector ~cero → caso degenerado.
    X_cos = np.vstack([
        rng.normal(loc=[1, 0],  scale=0.1, size=(20, 2)),   # clase A  ángulo ~0°
        rng.normal(loc=[0, 1],  scale=0.1, size=(20, 2)),   # clase B  ángulo ~90°
        rng.normal(loc=[-1, 0], scale=0.1, size=(20, 2)),   # clase C  ángulo ~180°
    ])
    y_cos = np.array(["A"] * 20 + ["B"] * 20 + ["C"] * 20)
    X_cos_test = np.array([[2.0, 0.1], [0.1, 3.0], [-2.0, 0.1]])
    y_cos_expected = np.array(["A", "B", "C"])
    nc_cos = NearestCentroidClassifier(distance_fn=Distances.cosine)
    preds_cos = nc_cos.fit(X_cos, y_cos).predict(X_cos_test)
    ok = "OK" if np.array_equal(preds_cos, y_cos_expected) else "FAIL"
    print(f"  [{ok}] distance={'cosine':12s} → pred={preds_cos}  expected={y_cos_expected}")

    print("\n=== Smoke test: KNNClassifier ===")
    for k in sorted(KNNClassifier.SUPPORTED_K):
        knn = KNNClassifier(k=k, distance_fn=Distances.euclidean)
        preds = knn.fit(X_train, y_train).predict(X_test)
        ok = "OK" if np.array_equal(preds, y_expected) else "FAIL"
        print(f"  [{ok}] k={k:2d} → pred={preds}  expected={y_expected}")

    print("\nTodos los tests completados.")
