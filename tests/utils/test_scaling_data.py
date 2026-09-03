"""Testes de elbow_method.scaling_data.

A função padroniza treino e teste por Z-score. A propriedade que importa não é
a fórmula, e sim quem a parametriza: o escalonador é ajustado apenas no treino
e depois aplicado ao teste. Ajustá-lo sobre a base inteira faria a média e o
desvio do conjunto de teste influenciarem a transformação — uma forma sutil de
vazamento que não aparece em nenhuma métrica.
"""

import numpy as np
import pandas as pd
import pytest

from elbow_method import scaling_data


@pytest.fixture
def partitions(dataset):
    features = dataset.drop(columns=["Class"])
    return features.iloc[:800], features.iloc[800:]


def test_returns_two_dataframes(partitions):
    train, test = partitions
    scaled_train, scaled_test = scaling_data(train, test)

    assert isinstance(scaled_train, pd.DataFrame)
    assert isinstance(scaled_test, pd.DataFrame)


def test_preserves_the_shape_of_both_partitions(partitions):
    train, test = partitions
    scaled_train, scaled_test = scaling_data(train, test)

    assert scaled_train.shape == train.shape
    assert scaled_test.shape == test.shape


def test_preserves_the_column_names_and_their_order(partitions):
    train, test = partitions
    scaled_train, scaled_test = scaling_data(train, test)

    assert list(scaled_train.columns) == list(train.columns)
    assert list(scaled_test.columns) == list(test.columns)


def test_centres_the_training_partition_at_zero(partitions):
    train, test = partitions
    scaled_train, _ = scaling_data(train, test)

    np.testing.assert_allclose(scaled_train.mean().to_numpy(), 0.0, atol=1e-10)


def test_scales_the_training_partition_to_unit_variance(partitions):
    train, test = partitions
    scaled_train, _ = scaling_data(train, test)

    np.testing.assert_allclose(scaled_train.std(ddof=0).to_numpy(), 1.0, atol=1e-10)


def test_the_test_partition_is_transformed_by_the_training_scaler(partitions):
    """O teste não é recentrado no próprio conjunto.

    Se o escalonador fosse reajustado no teste, sua média seria exatamente
    zero. Com partições de distribuições ligeiramente distintas, a média do
    teste transformado precisa diferir de zero.
    """
    train, test = partitions
    _, scaled_test = scaling_data(train, test)

    assert np.abs(scaled_test.mean().to_numpy()).max() > 1e-8


def test_the_test_transformation_matches_the_training_statistics(partitions):
    train, test = partitions
    _, scaled_test = scaling_data(train, test)

    expected = (test - train.mean()) / train.std(ddof=0)
    np.testing.assert_allclose(scaled_test.to_numpy(), expected.to_numpy(), rtol=1e-9)


def test_preserves_the_index_of_the_test_partition(partitions):
    train, test = partitions
    _, scaled_test = scaling_data(train, test)

    pd.testing.assert_index_equal(scaled_test.index, test.index)


def test_does_not_modify_the_inputs(partitions):
    train, test = partitions
    train_before, test_before = train.copy(), test.copy()

    scaling_data(train, test)

    pd.testing.assert_frame_equal(train, train_before)
    pd.testing.assert_frame_equal(test, test_before)


def test_brings_heterogeneous_scales_to_a_common_range(partitions):
    """Amount e Time diferem em ordens de grandeza das componentes de PCA.

    Sem padronização, a distância euclidiana usada pelos métodos baseados em
    densidade seria dominada por essas duas colunas.
    """
    train, test = partitions
    scaled_train, _ = scaling_data(train, test)

    spreads = scaled_train.std(ddof=0)
    assert spreads.max() / spreads.min() == pytest.approx(1.0, abs=1e-9)


def test_is_deterministic(partitions):
    train, test = partitions

    first_train, first_test = scaling_data(train, test)
    second_train, second_test = scaling_data(train, test)

    pd.testing.assert_frame_equal(first_train, second_train)
    pd.testing.assert_frame_equal(first_test, second_test)
