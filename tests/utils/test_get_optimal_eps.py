"""Testes de elbow_method.get_optimal_eps.

Estima o raio de vizinhança a partir do joelho da curva de distâncias ao
k-ésimo vizinho, seguindo a heurística de Sander et al.: minPts = 2·d, com a
curva construída sobre a distância ao (2d−1)-ésimo vizinho.

Os testes verificam a heurística — a dependência da dimensionalidade, a
sensibilidade à escala e o determinismo — e não um valor numérico específico,
que depende da geometria de cada conjunto.
"""

import numpy as np
import pandas as pd
import pytest

from elbow_method import get_optimal_eps


def make_clustered_data(n_dense=400, n_sparse=40, n_features=4, spread=40.0, seed=42):
    """Núcleo denso cercado de pontos esparsos, com joelho bem definido."""
    rng = np.random.default_rng(seed)
    dense = rng.normal(0.0, 1.0, size=(n_dense, n_features))
    sparse = rng.uniform(-spread, spread, size=(n_sparse, n_features))

    columns = [f"V{i}" for i in range(1, n_features + 1)]
    return pd.DataFrame(np.vstack([dense, sparse]), columns=columns)


@pytest.fixture
def clustered_data():
    return make_clustered_data()


def test_returns_a_positive_distance(clustered_data):
    eps = get_optimal_eps(clustered_data)

    assert eps is not None
    assert eps > 0


def test_returns_a_real_number(clustered_data):
    eps = get_optimal_eps(clustered_data)
    assert isinstance(eps, (float, np.floating))


def test_is_deterministic(clustered_data):
    """A heurística não tem componente aleatório: repetições devem coincidir."""
    assert get_optimal_eps(clustered_data) == get_optimal_eps(clustered_data)


def test_scales_with_the_dispersion_of_the_data(clustered_data):
    """Multiplicar as coordenadas por um fator escala eps pelo mesmo fator.

    A distância euclidiana é homogênea de grau um, então o joelho da curva se
    desloca proporcionalmente. É a razão pela qual o escalonamento precede a
    estimativa no pipeline.
    """
    eps = get_optimal_eps(clustered_data)
    eps_scaled = get_optimal_eps(clustered_data * 3.0)

    assert eps_scaled == pytest.approx(3.0 * eps, rel=1e-6)


def test_denser_data_yields_a_smaller_radius():
    dense = make_clustered_data(n_dense=800, spread=10.0)
    sparse = make_clustered_data(n_dense=800, spread=60.0)

    assert get_optimal_eps(dense) < get_optimal_eps(sparse)


def test_is_invariant_to_the_order_of_the_rows(clustered_data):
    shuffled = clustered_data.sample(frac=1.0, random_state=7)

    assert get_optimal_eps(shuffled) == pytest.approx(get_optimal_eps(clustered_data))


@pytest.mark.parametrize("n_features", [2, 4, 8])
def test_accepts_different_dimensionalities(n_features):
    data = make_clustered_data(n_features=n_features)
    eps = get_optimal_eps(data)

    assert eps is None or eps > 0


def test_requires_more_samples_than_the_neighbourhood_size():
    """Com minPts = 2·d, um conjunto menor que isso não admite a estimativa."""
    data = make_clustered_data(n_dense=5, n_sparse=1, n_features=8)

    with pytest.raises(ValueError):
        get_optimal_eps(data)


def test_accepts_a_numpy_array(clustered_data):
    from_frame = get_optimal_eps(clustered_data)
    from_array = get_optimal_eps(clustered_data.to_numpy())

    assert from_array == pytest.approx(from_frame)


def test_the_estimated_radius_stays_within_the_observed_distances(clustered_data):
    """eps precisa ser plausível: nem menor que a menor distância entre
    vizinhos, nem maior que o diâmetro do conjunto."""
    from sklearn.neighbors import NearestNeighbors

    dimension = clustered_data.shape[1]
    neighbours = NearestNeighbors(n_neighbors=2 * dimension).fit(clustered_data)
    distances, _ = neighbours.kneighbors(clustered_data)

    eps = get_optimal_eps(clustered_data)

    assert eps > distances[:, 1].min()
    assert eps < distances.max()
