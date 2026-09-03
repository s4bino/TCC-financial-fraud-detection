"""Testes de cross_validation.resolve_target_column.

A função aceita a coluna alvo por nome ou por índice posicional. É o ponto de
entrada de toda a geração de folds: um erro silencioso aqui faria o pipeline
estratificar pela variável errada, invalidando todos os resultados.
"""

import pandas as pd
import pytest

from cross_validation import resolve_target_column


def test_returns_the_name_when_given_a_valid_name(dataset):
    assert resolve_target_column(dataset, "Class") == "Class"


def test_accepts_a_non_target_column_by_name(dataset):
    assert resolve_target_column(dataset, "Amount") == "Amount"


def test_resolves_the_last_index_to_the_target_column(dataset):
    # Na base do trabalho, Class é a 31ª e última coluna.
    assert resolve_target_column(dataset, len(dataset.columns) - 1) == "Class"


def test_resolves_the_first_index(dataset):
    assert resolve_target_column(dataset, 0) == "Time"


@pytest.mark.parametrize("index, expected", [(0, "Time"), (1, "V1"), (28, "V28"), (29, "Amount")])
def test_resolves_intermediate_indices(dataset, index, expected):
    assert resolve_target_column(dataset, index) == expected


def test_rejects_an_index_beyond_the_last_column(dataset):
    with pytest.raises(IndexError, match="Índice de coluna inválido"):
        resolve_target_column(dataset, len(dataset.columns))


def test_rejects_a_negative_index(dataset):
    # Índices negativos são válidos em Python, mas ambíguos como especificação
    # de coluna alvo; a função os rejeita explicitamente.
    with pytest.raises(IndexError):
        resolve_target_column(dataset, -1)


def test_error_message_for_an_absent_name_lists_the_available_columns(dataset):
    with pytest.raises(ValueError) as error:
        resolve_target_column(dataset, "Fraude")

    message = str(error.value)
    assert "Fraude" in message
    assert "Class" in message


@pytest.mark.parametrize("target", [1.0, None, ["Class"], {"Class"}, b"Class"])
def test_rejects_types_that_are_neither_name_nor_index(dataset, target):
    with pytest.raises(TypeError):
        resolve_target_column(dataset, target)


def test_is_case_sensitive(dataset):
    with pytest.raises(ValueError):
        resolve_target_column(dataset, "class")


def test_works_on_a_single_column_frame():
    frame = pd.DataFrame({"Class": [0, 1, 0]})
    assert resolve_target_column(frame, 0) == "Class"
    assert resolve_target_column(frame, "Class") == "Class"


def test_does_not_modify_the_frame(dataset):
    before = dataset.copy()
    resolve_target_column(dataset, "Class")
    pd.testing.assert_frame_equal(dataset, before)
