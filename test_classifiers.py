"""
Práctica 4 - IA: Clasificación Académica
Suite de pruebas unitarias para classifiers.py y main.py.

Cubre:
    - Corrección matemática de las cuatro métricas de distancia.
    - Propiedades universales de toda métrica (no negatividad, simetría).
    - Instanciación, flujo fit/predict y manejo de errores de los clasificadores.
    - Corrección estructural de los tres métodos de validación.
    - Cálculo de accuracy.
    - Normalización Z-score sin data leakage.

No se descarga ningún dataset externo: todos los tests usan datos sintéticos.
"""

import numpy as np
import pytest

from classifiers import (
    Distances,
    KNNClassifier,
    NearestCentroidClassifier,
)
from main import (
    accuracy,
    holdout_split,
    kfold_indices,
    loo_indices,
    standardize,
)


# ---------------------------------------------------------------------------
# Fixtures compartidos
# ---------------------------------------------------------------------------

@pytest.fixture
def two_class_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Dataset sintético 2D con dos clases bien separadas (30 muestras)."""
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.normal(loc=[0.0, 0.0], scale=0.3, size=(15, 2)),
        rng.normal(loc=[5.0, 5.0], scale=0.3, size=(15, 2)),
    ])
    y = np.array(["A"] * 15 + ["B"] * 15)
    return X, y


@pytest.fixture
def three_class_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Dataset sintético 2D con tres clases bien separadas (60 muestras)."""
    rng = np.random.default_rng(7)
    X = np.vstack([
        rng.normal(loc=[0.0, 0.0], scale=0.4, size=(20, 2)),
        rng.normal(loc=[6.0, 0.0], scale=0.4, size=(20, 2)),
        rng.normal(loc=[3.0, 5.0], scale=0.4, size=(20, 2)),
    ])
    y = np.array([0] * 20 + [1] * 20 + [2] * 20)
    return X, y


# ===========================================================================
# Tests de métricas de distancia
# ===========================================================================

class TestDistances:
    """Verifica corrección matemática y propiedades de las cuatro métricas."""

    # ---- Valores conocidos ------------------------------------------------

    def test_euclidean_3_4_5(self) -> None:
        """Triángulo rectángulo 3-4-5: distancia debe ser exactamente 5."""
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        assert Distances.euclidean(a, b) == pytest.approx(5.0)

    def test_manhattan_known(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 6.0, 3.0])
        # |3| + |4| + |0| = 7
        assert Distances.manhattan(a, b) == pytest.approx(7.0)

    def test_chebyshev_known(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 6.0, 3.0])
        # max(3, 4, 0) = 4
        assert Distances.chebyshev(a, b) == pytest.approx(4.0)

    def test_cosine_identical_vectors_is_zero(self) -> None:
        """Vectores idénticos tienen ángulo 0 → distancia coseno = 0."""
        a = np.array([1.0, 2.0, 3.0])
        assert Distances.cosine(a, a) == pytest.approx(0.0, abs=1e-9)

    def test_cosine_orthogonal_vectors_is_one(self) -> None:
        """Vectores ortogonales tienen ángulo 90° → distancia coseno = 1."""
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert Distances.cosine(a, b) == pytest.approx(1.0)

    def test_cosine_opposite_vectors_is_two(self) -> None:
        """Vectores opuestos tienen ángulo 180° → distancia coseno = 2."""
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert Distances.cosine(a, b) == pytest.approx(2.0, abs=1e-9)

    def test_cosine_zero_vector_returns_one(self) -> None:
        """Vector cero es caso degenerado → debe retornar 1.0 (sin ángulo definido)."""
        a = np.zeros(5)
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert Distances.cosine(a, b) == 1.0

    # ---- Propiedades universales ------------------------------------------

    @pytest.mark.parametrize("dist_fn", [
        Distances.euclidean,
        Distances.manhattan,
        Distances.chebyshev,
        Distances.cosine,
    ])
    def test_non_negativity(self, dist_fn) -> None:
        rng = np.random.default_rng(1)
        a, b = rng.random(10), rng.random(10)
        assert dist_fn(a, b) >= 0.0

    @pytest.mark.parametrize("dist_fn", [
        Distances.euclidean,
        Distances.manhattan,
        Distances.chebyshev,
        Distances.cosine,
    ])
    def test_symmetry(self, dist_fn) -> None:
        rng = np.random.default_rng(2)
        a, b = rng.random(10), rng.random(10)
        assert dist_fn(a, b) == pytest.approx(dist_fn(b, a), rel=1e-9)

    @pytest.mark.parametrize("dist_fn", [
        Distances.euclidean,
        Distances.manhattan,
        Distances.chebyshev,
    ])
    def test_self_distance_is_zero(self, dist_fn) -> None:
        a = np.array([1.0, 2.0, 3.0, 4.0])
        assert dist_fn(a, a) == pytest.approx(0.0, abs=1e-10)


# ===========================================================================
# Tests de NearestCentroidClassifier
# ===========================================================================

class TestNearestCentroidClassifier:

    def test_predict_before_fit_raises_runtime_error(self) -> None:
        clf = NearestCentroidClassifier()
        with pytest.raises(RuntimeError, match="no ha sido entrenado"):
            clf.predict(np.array([[1.0, 2.0]]))

    def test_fit_returns_self(self, two_class_dataset) -> None:
        X, y = two_class_dataset
        clf = NearestCentroidClassifier()
        assert clf.fit(X, y) is clf

    def test_fit_chaining(self, two_class_dataset) -> None:
        X, y = two_class_dataset
        preds = NearestCentroidClassifier().fit(X, y).predict(X)
        assert len(preds) == len(y)

    def test_predict_output_shape(self, two_class_dataset) -> None:
        X, y = two_class_dataset
        clf = NearestCentroidClassifier().fit(X, y)
        assert clf.predict(X).shape == (len(y),)

    def test_high_accuracy_on_separated_data(self, two_class_dataset) -> None:
        X, y = two_class_dataset
        clf = NearestCentroidClassifier().fit(X, y)
        assert accuracy(y, clf.predict(X)) >= 0.90

    def test_get_centroids_keys_match_classes(self, two_class_dataset) -> None:
        X, y = two_class_dataset
        clf = NearestCentroidClassifier().fit(X, y)
        centroids = clf.get_centroids()
        assert set(centroids.keys()) == set(np.unique(y))

    def test_get_centroids_shape(self, two_class_dataset) -> None:
        X, y = two_class_dataset
        clf = NearestCentroidClassifier().fit(X, y)
        for centroid in clf.get_centroids().values():
            assert centroid.shape == (X.shape[1],)

    def test_get_centroids_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            NearestCentroidClassifier().get_centroids()

    @pytest.mark.parametrize("dist_fn", [
        Distances.euclidean,
        Distances.manhattan,
        Distances.chebyshev,
        Distances.cosine,
    ])
    def test_all_distance_functions_run_without_error(
        self, dist_fn, two_class_dataset
    ) -> None:
        X, y = two_class_dataset
        preds = NearestCentroidClassifier(distance_fn=dist_fn).fit(X, y).predict(X)
        assert len(preds) == len(y)

    def test_three_classes(self, three_class_dataset) -> None:
        X, y = three_class_dataset
        clf = NearestCentroidClassifier().fit(X, y)
        assert len(np.unique(clf.predict(X))) <= 3


# ===========================================================================
# Tests de KNNClassifier
# ===========================================================================

class TestKNNClassifier:

    def test_invalid_k_raises_value_error(self) -> None:
        for bad_k in [0, 2, 4, 6, 8, 10, 12, 100]:
            with pytest.raises(ValueError, match="no está soportado"):
                KNNClassifier(k=bad_k)

    def test_predict_before_fit_raises_runtime_error(self) -> None:
        clf = KNNClassifier(k=3)
        with pytest.raises(RuntimeError, match="no ha sido entrenado"):
            clf.predict(np.array([[1.0, 2.0]]))

    def test_fit_returns_self(self, two_class_dataset) -> None:
        X, y = two_class_dataset
        clf = KNNClassifier(k=3)
        assert clf.fit(X, y) is clf

    @pytest.mark.parametrize("k", [1, 3, 5, 7, 9, 11])
    def test_all_supported_k_values(self, k, two_class_dataset) -> None:
        X, y = two_class_dataset
        clf = KNNClassifier(k=k).fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (len(y),)

    def test_1nn_perfect_on_training_data(self, two_class_dataset) -> None:
        """1NN siempre acierta sobre sus propios datos de entrenamiento."""
        X, y = two_class_dataset
        clf = KNNClassifier(k=1).fit(X, y)
        assert accuracy(y, clf.predict(X)) == pytest.approx(1.0)

    def test_high_accuracy_on_separated_data(self, two_class_dataset) -> None:
        X, y = two_class_dataset
        clf = KNNClassifier(k=5).fit(X, y)
        assert accuracy(y, clf.predict(X)) >= 0.90

    def test_predict_proba_sums_to_one(self, two_class_dataset) -> None:
        X, y = two_class_dataset
        clf = KNNClassifier(k=5).fit(X, y)
        proba = clf.predict_proba(X[:8])
        classes = list(proba.keys())
        for i in range(8):
            total = sum(proba[cls][i] for cls in classes)
            assert total == pytest.approx(1.0, abs=1e-9)

    def test_predict_proba_output_length(self, two_class_dataset) -> None:
        X, y = two_class_dataset
        clf = KNNClassifier(k=3).fit(X, y)
        proba = clf.predict_proba(X[:5])
        for cls_probs in proba.values():
            assert len(cls_probs) == 5

    @pytest.mark.parametrize("dist_fn", [
        Distances.euclidean,
        Distances.manhattan,
        Distances.chebyshev,
        Distances.cosine,
    ])
    def test_all_distance_functions_run_without_error(
        self, dist_fn, two_class_dataset
    ) -> None:
        X, y = two_class_dataset
        preds = KNNClassifier(k=3, distance_fn=dist_fn).fit(X, y).predict(X)
        assert len(preds) == len(y)


# ===========================================================================
# Tests de métodos de validación
# ===========================================================================

class TestHoldoutSplit:

    def test_sizes_sum_to_n(self) -> None:
        X = np.zeros((50, 4))
        y = np.array(["A"] * 25 + ["B"] * 25)
        X_tr, X_te, y_tr, y_te = holdout_split(X, y, train_ratio=0.70)
        assert len(y_tr) + len(y_te) == 50

    def test_train_larger_than_test(self) -> None:
        X = np.zeros((50, 4))
        y = np.array(["A"] * 25 + ["B"] * 25)
        _, _, y_tr, y_te = holdout_split(X, y, train_ratio=0.70)
        assert len(y_tr) > len(y_te)

    def test_no_overlap_between_splits(self) -> None:
        X = np.arange(100).reshape(50, 2).astype(float)
        y = np.array(["A"] * 25 + ["B"] * 25)
        X_tr, X_te, _, _ = holdout_split(X, y)
        train_rows = {tuple(r) for r in X_tr}
        test_rows  = {tuple(r) for r in X_te}
        assert train_rows.isdisjoint(test_rows)

    def test_reproducible_with_same_seed(self) -> None:
        X = np.eye(40)
        y = np.array(["A"] * 20 + ["B"] * 20)
        split1 = holdout_split(X, y, seed=99)
        split2 = holdout_split(X, y, seed=99)
        assert np.array_equal(split1[2], split2[2])   # y_train identical


class TestKFoldIndices:

    def test_returns_k_splits(self) -> None:
        assert len(kfold_indices(100, k=10)) == 10

    def test_all_indices_covered_exactly_once(self) -> None:
        n, k = 50, 5
        splits = kfold_indices(n, k=k)
        all_test = np.concatenate([te for _, te in splits])
        assert set(all_test.tolist()) == set(range(n))
        assert len(all_test) == n  # sin repetición

    def test_no_overlap_within_fold(self) -> None:
        for train_idx, test_idx in kfold_indices(60, k=6):
            assert len(set(train_idx.tolist()) & set(test_idx.tolist())) == 0

    def test_train_larger_than_test_per_fold(self) -> None:
        for train_idx, test_idx in kfold_indices(100, k=10):
            assert len(train_idx) > len(test_idx)

    def test_reproducible_with_same_seed(self) -> None:
        s1 = kfold_indices(50, k=5, seed=0)
        s2 = kfold_indices(50, k=5, seed=0)
        for (tr1, te1), (tr2, te2) in zip(s1, s2):
            assert np.array_equal(tr1, tr2)
            assert np.array_equal(te1, te2)


class TestLOOIndices:

    def test_number_of_iterations_equals_n(self) -> None:
        n = 12
        assert sum(1 for _ in loo_indices(n)) == n

    def test_test_set_has_exactly_one_sample(self) -> None:
        for _, test_idx in loo_indices(8):
            assert len(test_idx) == 1

    def test_train_size_is_n_minus_one(self) -> None:
        n = 10
        for train_idx, _ in loo_indices(n):
            assert len(train_idx) == n - 1

    def test_no_overlap_per_iteration(self) -> None:
        for train_idx, test_idx in loo_indices(10):
            assert test_idx[0] not in train_idx

    def test_every_index_appears_as_test_exactly_once(self) -> None:
        n = 15
        test_indices = [test_idx[0] for _, test_idx in loo_indices(n)]
        assert sorted(test_indices) == list(range(n))


# ===========================================================================
# Tests de accuracy y standardize
# ===========================================================================

class TestAccuracy:

    def test_perfect_prediction(self) -> None:
        y = np.array([0, 1, 2, 0, 1, 2])
        assert accuracy(y, y) == pytest.approx(1.0)

    def test_zero_accuracy(self) -> None:
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        assert accuracy(y_true, y_pred) == pytest.approx(0.0)

    def test_partial_accuracy(self) -> None:
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        # 4 aciertos de 8
        assert accuracy(y_true, y_pred) == pytest.approx(0.5)

    def test_single_sample(self) -> None:
        assert accuracy(np.array([5]), np.array([5])) == pytest.approx(1.0)
        assert accuracy(np.array([5]), np.array([9])) == pytest.approx(0.0)


class TestStandardize:

    def test_train_mean_is_zero_after_scaling(self) -> None:
        X_tr = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
        X_te = np.array([[2.0, 15.0]])
        X_tr_z, _ = standardize(X_tr, X_te)
        assert np.allclose(np.mean(X_tr_z, axis=0), 0.0, atol=1e-10)

    def test_train_std_is_one_after_scaling(self) -> None:
        X_tr = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
        X_te = np.array([[2.0, 15.0]])
        X_tr_z, _ = standardize(X_tr, X_te)
        assert np.allclose(np.std(X_tr_z, axis=0, ddof=1), 1.0, atol=1e-9)

    def test_test_uses_train_statistics(self) -> None:
        """El test set NO debe tener media cero si sus valores difieren del train."""
        X_tr = np.array([[1.0], [2.0], [3.0]])
        X_te = np.array([[100.0]])     # muy alejado del rango de train
        _, X_te_z = standardize(X_tr, X_te)
        assert abs(X_te_z[0, 0]) > 1.0

    def test_zero_variance_column_does_not_raise(self) -> None:
        """Columna constante (σ=0) debe sobrevivir sin división por cero."""
        X_tr = np.array([[1.0, 5.0], [1.0, 6.0], [1.0, 7.0]])
        X_te = np.array([[1.0, 8.0]])
        X_tr_z, X_te_z = standardize(X_tr, X_te)
        assert not np.any(np.isnan(X_tr_z))
        assert not np.any(np.isnan(X_te_z))

    def test_output_shapes_preserved(self) -> None:
        X_tr = np.random.default_rng(0).random((20, 5))
        X_te = np.random.default_rng(1).random((8, 5))
        X_tr_z, X_te_z = standardize(X_tr, X_te)
        assert X_tr_z.shape == X_tr.shape
        assert X_te_z.shape == X_te.shape
