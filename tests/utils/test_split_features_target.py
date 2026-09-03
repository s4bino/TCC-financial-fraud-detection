"""Testes de elbow_method.split_features_target.

Separa preditoras e alvo em treino e teste. Um erro aqui — manter Class entre
as preditoras — vazaria o rótulo diretamente para o modelo, produzindo
desempenho perfeito e sem sentido.
"""

import pandas as pd
import pytest

from elbow_method import split_features_target


@pytest.fixture
def partitions(dataset):
    return dataset.iloc[:800], dataset.iloc[800:]


def test_returns_four_objects(partitions):
    train, test = partitions
    result = split_features_target(train, test, "Class")

    assert len(result) == 4


def test_removes_the_target_from_the_predictors(partitions):
    train, test = partitions
    X_train, _, X_test, _ = split_features_target(train, test, "Class")

    assert "Class" not in X_train.columns
    assert "Class" not in X_test.columns


def test_keeps_every_other_column_as_a_predictor(partitions, dataset):
    train, test = partitions
    X_train, _, _, _ = split_features_target(train, test, "Class")

    expected = [column for column in dataset.columns if column != "Class"]
    assert list(X_train.columns) == expected


def test_returns_the_target_column_as_the_label(partitions):
    train, test = partitions
    _, y_train, _, y_test = split_features_target(train, test, "Class")

    pd.testing.assert_series_equal(y_train, train["Class"])
    pd.testing.assert_series_equal(y_test, test["Class"])


def test_preserves_the_number_of_rows(partitions):
    train, test = partitions
    X_train, y_train, X_test, y_test = split_features_target(train, test, "Class")

    assert len(X_train) == len(y_train) == len(train)
    assert len(X_test) == len(y_test) == len(test)


def test_predictors_and_label_stay_aligned_by_index(partitions):
    train, test = partitions
    X_train, y_train, X_test, y_test = split_features_target(train, test, "Class")

    pd.testing.assert_index_equal(X_train.index, y_train.index)
    pd.testing.assert_index_equal(X_test.index, y_test.index)


def test_accepts_any_column_as_the_target(partitions):
    train, test = partitions
    X_train, y_train, _, _ = split_features_target(train, test, "Amount")

    assert "Amount" not in X_train.columns
    assert "Class" in X_train.columns
    assert y_train.name == "Amount"


def test_rejects_a_column_that_does_not_exist(partitions):
    train, test = partitions

    with pytest.raises(KeyError):
        split_features_target(train, test, "Fraude")


def test_does_not_modify_the_inputs(partitions):
    train, test = partitions
    train_before, test_before = train.copy(), test.copy()

    split_features_target(train, test, "Class")

    pd.testing.assert_frame_equal(train, train_before)
    pd.testing.assert_frame_equal(test, test_before)
