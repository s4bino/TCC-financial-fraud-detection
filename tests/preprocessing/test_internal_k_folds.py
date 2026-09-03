"""Testes de cross_validation.internal_k_folds.

A função deriva, de cada treino externo, o par treino/validação usado na busca
em grade. A propriedade decisiva é o isolamento: nada do teste externo pode
alcançar a partição de calibração, sob pena de o desempenho reportado deixar de
ser uma estimativa de generalização.
"""

import json

import pandas as pd
import pytest

from cross_validation import external_k_folds, internal_k_folds

N_SPLITS = 3
VAL_SIZE = 0.2


@pytest.fixture
def outer_dir(tmp_path, dataset_csv):
    directory = tmp_path / "outer_folds"
    external_k_folds(dataset_csv, str(directory), "Class", n_splits=N_SPLITS, random_state=42)
    return directory


@pytest.fixture
def inner_dir(tmp_path, outer_dir):
    directory = tmp_path / "inner_folds"
    internal_k_folds(
        folds_dir=str(outer_dir),
        internal_output_dir=str(directory),
        target="Class",
        val_size=VAL_SIZE,
        random_state=42,
    )
    return directory


def read_csv(directory, name):
    return pd.read_csv(directory / name)


def test_creates_a_train_and_a_validation_file_per_fold(inner_dir):
    for fold in range(1, N_SPLITS + 1):
        assert (inner_dir / f"fold_{fold}_train.csv").exists()
        assert (inner_dir / f"fold_{fold}_val.csv").exists()


def test_creates_the_internal_metadata_per_fold(inner_dir):
    for fold in range(1, N_SPLITS + 1):
        assert (inner_dir / f"fold_{fold}_internal_info.json").exists()


def test_creates_the_output_directory_when_absent(tmp_path, outer_dir):
    directory = tmp_path / "ainda" / "nao" / "criado"
    internal_k_folds(str(outer_dir), str(directory), "Class")
    assert directory.is_dir()


def test_writes_to_a_separate_directory_from_the_outer_folds(inner_dir, outer_dir):
    assert inner_dir != outer_dir
    assert not (inner_dir / "fold_1_test.csv").exists()


def test_does_not_modify_the_outer_fold_files(tmp_path, dataset_csv):
    outer = tmp_path / "outer"
    external_k_folds(dataset_csv, str(outer), "Class", n_splits=2, random_state=42)

    before = {path.name: path.read_bytes() for path in outer.iterdir()}
    internal_k_folds(str(outer), str(tmp_path / "inner"), "Class")
    after = {path.name: path.read_bytes() for path in outer.iterdir()}

    assert before == after


def test_train_and_validation_reconstruct_the_outer_train(inner_dir, outer_dir):
    for fold in range(1, N_SPLITS + 1):
        outer_train = read_csv(outer_dir, f"fold_{fold}_train.csv")
        inner_train = read_csv(inner_dir, f"fold_{fold}_train.csv")
        validation = read_csv(inner_dir, f"fold_{fold}_val.csv")

        assert len(inner_train) + len(validation) == len(outer_train)


def test_respects_the_requested_validation_size(inner_dir, outer_dir):
    for fold in range(1, N_SPLITS + 1):
        outer_train = read_csv(outer_dir, f"fold_{fold}_train.csv")
        validation = read_csv(inner_dir, f"fold_{fold}_val.csv")

        assert len(validation) == pytest.approx(VAL_SIZE * len(outer_train), rel=0.02)


def test_train_and_validation_are_disjoint(inner_dir):
    for fold in range(1, N_SPLITS + 1):
        train = read_csv(inner_dir, f"fold_{fold}_train.csv")
        validation = read_csv(inner_dir, f"fold_{fold}_val.csv")

        assert pd.merge(train, validation, how="inner").empty


def test_the_outer_test_partition_never_reaches_the_inner_folds(inner_dir, outer_dir):
    """Isolamento entre calibração e estimativa de desempenho.

    Se uma transação do teste externo aparecesse no treino ou na validação
    interna, o hiperparâmetro escolhido teria sido calibrado sobre os mesmos
    dados usados para medir a generalização.
    """
    for fold in range(1, N_SPLITS + 1):
        outer_test = read_csv(outer_dir, f"fold_{fold}_test.csv")

        for partition in ("train", "val"):
            inner = read_csv(inner_dir, f"fold_{fold}_{partition}.csv")
            leaked = pd.merge(outer_test, inner, how="inner")
            assert leaked.empty, f"fold {fold}: teste externo vazou para {partition}"


def test_preserves_the_fraud_ratio_in_the_validation_partition(inner_dir, outer_dir):
    for fold in range(1, N_SPLITS + 1):
        outer_ratio = read_csv(outer_dir, f"fold_{fold}_train.csv")["Class"].mean()
        validation = read_csv(inner_dir, f"fold_{fold}_val.csv")

        assert validation["Class"].mean() == pytest.approx(outer_ratio, abs=0.01)


def test_both_classes_survive_in_every_inner_partition(inner_dir):
    for fold in range(1, N_SPLITS + 1):
        for partition in ("train", "val"):
            frame = read_csv(inner_dir, f"fold_{fold}_{partition}.csv")
            assert set(frame["Class"].unique()) == {0, 1}


def test_preserves_the_columns_of_the_base(inner_dir, dataset):
    for fold in range(1, N_SPLITS + 1):
        for partition in ("train", "val"):
            frame = read_csv(inner_dir, f"fold_{fold}_{partition}.csv")
            assert list(frame.columns) == list(dataset.columns)


def test_internal_metadata_embeds_the_external_metadata(inner_dir):
    with open(inner_dir / "fold_1_internal_info.json", encoding="utf-8") as file:
        metadata = json.load(file)

    assert metadata["fold"] == 1
    assert metadata["external_metadata"]["fold"] == 1
    assert metadata["external_metadata"]["target_column"] == "Class"


def test_internal_metadata_declares_the_split_parameters(inner_dir):
    with open(inner_dir / "fold_1_internal_info.json", encoding="utf-8") as file:
        internal = json.load(file)["internal_validation"]

    assert internal["method"] == "train_test_split"
    assert internal["val_size"] == VAL_SIZE
    assert internal["random_state"] == 42


def test_internal_metadata_sizes_match_the_written_files(inner_dir):
    for fold in range(1, N_SPLITS + 1):
        with open(inner_dir / f"fold_{fold}_internal_info.json", encoding="utf-8") as file:
            internal = json.load(file)["internal_validation"]

        train = read_csv(inner_dir, f"fold_{fold}_train.csv")
        validation = read_csv(inner_dir, f"fold_{fold}_val.csv")

        assert internal["num_train_final_samples"] == len(train)
        assert internal["num_val_samples"] == len(validation)


@pytest.mark.parametrize("val_size", [0.1, 0.25, 0.5])
def test_honours_different_validation_sizes(tmp_path, outer_dir, val_size):
    directory = tmp_path / f"inner_{val_size}"
    internal_k_folds(str(outer_dir), str(directory), "Class", val_size=val_size)

    outer_train = read_csv(outer_dir, "fold_1_train.csv")
    validation = read_csv(directory, "fold_1_val.csv")

    assert len(validation) == pytest.approx(val_size * len(outer_train), rel=0.05)


def test_is_reproducible_for_the_same_seed(tmp_path, outer_dir):
    first = tmp_path / "primeira"
    second = tmp_path / "segunda"

    for directory in (first, second):
        internal_k_folds(str(outer_dir), str(directory), "Class", random_state=42)

    pd.testing.assert_frame_equal(
        read_csv(first, "fold_1_val.csv"), read_csv(second, "fold_1_val.csv")
    )
