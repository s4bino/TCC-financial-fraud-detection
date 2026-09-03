"""Testes de stratified_sampling.amostragem.

Produz recortes reduzidos da base para experimentos que não comportam as
284.807 transações — caso dos métodos baseados em densidade, cujo custo cresce
de forma superlinear. O recorte só é útil se preservar a proporção de fraudes:
uma amostra com desbalanceamento diferente responde a uma pergunta diferente.
"""

import pandas as pd
import pytest

from stratified_sampling import amostragem


@pytest.mark.parametrize("n_rows", [50, 100, 500])
def test_returns_exactly_the_requested_number_of_rows(dataset, n_rows):
    sample = amostragem(dataset, coluna_alvo="Class", n_linhas=n_rows)
    assert len(sample) == n_rows


def test_preserves_the_fraud_ratio(dataset):
    population_ratio = dataset["Class"].mean()
    sample = amostragem(dataset, "Class", n_linhas=500)

    assert sample["Class"].mean() == pytest.approx(population_ratio, abs=0.01)


def test_keeps_both_classes_present(dataset):
    sample = amostragem(dataset, "Class", n_linhas=200)
    assert set(sample["Class"].unique()) == {0, 1}


def test_preserves_the_columns_of_the_base(dataset):
    sample = amostragem(dataset, "Class", n_linhas=100)
    assert list(sample.columns) == list(dataset.columns)


def test_every_sampled_row_comes_from_the_base(dataset):
    sample = amostragem(dataset, "Class", n_linhas=100)
    assert sample.index.isin(dataset.index).all()


def test_does_not_repeat_rows(dataset):
    sample = amostragem(dataset, "Class", n_linhas=300)
    assert sample.index.is_unique


def test_sampled_rows_are_identical_to_the_originals(dataset):
    sample = amostragem(dataset, "Class", n_linhas=100)
    pd.testing.assert_frame_equal(sample, dataset.loc[sample.index])


def test_is_reproducible(dataset):
    """A semente é fixa em 42 dentro da função: dois recortes coincidem."""
    first = amostragem(dataset, "Class", n_linhas=200)
    second = amostragem(dataset, "Class", n_linhas=200)

    pd.testing.assert_frame_equal(first, second)


def test_does_not_modify_the_base(dataset):
    before = dataset.copy()
    amostragem(dataset, "Class", n_linhas=200)
    pd.testing.assert_frame_equal(dataset, before)


def test_supports_stratifying_by_another_binary_column(dataset):
    base = dataset.copy()
    base["Alto_valor"] = (base["Amount"] > base["Amount"].median()).astype(int)

    sample = amostragem(base, coluna_alvo="Alto_valor", n_linhas=200)

    assert sample["Alto_valor"].mean() == pytest.approx(0.5, abs=0.02)


def test_rejects_a_size_larger_than_the_base(dataset):
    with pytest.raises(ValueError):
        amostragem(dataset, "Class", n_linhas=len(dataset) + 1)


def test_rejects_a_size_smaller_than_the_number_of_classes(dataset):
    with pytest.raises(ValueError):
        amostragem(dataset, "Class", n_linhas=1)


def test_fails_when_a_class_has_a_single_representative(dataset_factory):
    """A estratificação exige ao menos duas ocorrências de cada classe."""
    base = dataset_factory(n_samples=100, n_frauds=1)

    with pytest.raises(ValueError):
        amostragem(base, "Class", n_linhas=50)
