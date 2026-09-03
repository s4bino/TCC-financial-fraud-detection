"""Testes de stratified_sampling.verifica_diretorio.

Garante a existência do diretório de saída antes da gravação. Precisa ser
idempotente: é chamada em toda execução, inclusive quando o diretório já
existe e contém recortes anteriores que não podem ser perdidos.
"""

import pytest

from stratified_sampling import verifica_diretorio


def test_creates_a_directory_that_does_not_exist(tmp_path):
    target = tmp_path / "amostras"
    assert not target.exists()

    verifica_diretorio(str(target))

    assert target.is_dir()


def test_does_nothing_when_the_directory_already_exists(tmp_path):
    target = tmp_path / "amostras"
    target.mkdir()

    verifica_diretorio(str(target))

    assert target.is_dir()


def test_is_idempotent_across_repeated_calls(tmp_path):
    target = tmp_path / "amostras"

    for _ in range(3):
        verifica_diretorio(str(target))

    assert target.is_dir()


def test_preserves_existing_content(tmp_path):
    """Uma chamada não pode descartar recortes já gravados."""
    target = tmp_path / "amostras"
    target.mkdir()
    existing = target / "amostra_16000_train_fold1.csv"
    existing.write_text("Time,Amount,Class\n0,1.0,0\n", encoding="utf-8")

    verifica_diretorio(str(target))

    assert existing.exists()
    assert existing.read_text(encoding="utf-8").startswith("Time,Amount,Class")


def test_creates_intermediate_levels_of_a_nested_path(tmp_path):
    target = tmp_path / "data" / "processed" / "samples"

    verifica_diretorio(str(target))

    assert target.is_dir()


@pytest.mark.parametrize("name", ["saida", "com espaco", "com-hifen", "com_underscore"])
def test_accepts_usual_directory_names(tmp_path, name):
    target = tmp_path / name

    verifica_diretorio(str(target))

    assert target.is_dir()
