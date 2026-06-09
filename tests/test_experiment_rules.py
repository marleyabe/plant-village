from __future__ import annotations

import json

import pytest

from plant_village.config import ExperimentConfig, make_experiment_config
from plant_village.data import configure_tensorflow_resources, split_counts
from plant_village.evaluate import load_training_metadata
from plant_village.predict import load_class_names


def test_default_experiment_config_matches_project_methodology() -> None:
    config = make_experiment_config()

    assert config.image_size == (224, 224)
    assert config.batch_size == 32
    assert config.seed == 42
    assert config.train_ratio == 0.70
    assert config.val_ratio == 0.15
    assert config.test_ratio == 0.15


@pytest.mark.parametrize(
    "kwargs",
    [
        {"image_size": (0, 224)},
        {"batch_size": 0},
        {"epochs": 0},
        {"learning_rate": 0},
        {"train_ratio": 0.80, "val_ratio": 0.15, "test_ratio": 0.15},
        {"train_ratio": 0.0, "val_ratio": 0.50, "test_ratio": 0.50},
    ],
)
def test_experiment_config_rejects_invalid_business_rules(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(**kwargs)


def test_split_counts_preserves_every_example() -> None:
    config = ExperimentConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    split = split_counts(54303, config)

    assert split.train == 38012
    assert split.val == 8145
    assert split.test == 8146
    assert split.total == 54303


def test_split_counts_requires_non_empty_train_validation_and_test_sets() -> None:
    config = ExperimentConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    with pytest.raises(ValueError):
        split_counts(2, config)


def test_configure_tensorflow_resources_enables_gpu_memory_growth(monkeypatch) -> None:
    calls: dict[str, int] = {}
    gpu = object()
    memory_growth_calls = []

    monkeypatch.setattr(
        "plant_village.data.tf.config.threading.set_intra_op_parallelism_threads",
        lambda value: calls.update({"intra_op_threads": value}),
    )
    monkeypatch.setattr(
        "plant_village.data.tf.config.threading.set_inter_op_parallelism_threads",
        lambda value: calls.update({"inter_op_threads": value}),
    )
    monkeypatch.setattr(
        "plant_village.data.tf.config.list_physical_devices",
        lambda device_type: [gpu] if device_type == "GPU" else [],
    )
    monkeypatch.setattr(
        "plant_village.data.tf.config.experimental.set_memory_growth",
        lambda device, enabled: memory_growth_calls.append((device, enabled)),
    )

    configure_tensorflow_resources(intra_op_threads=3, inter_op_threads=2)

    assert calls == {"intra_op_threads": 3, "inter_op_threads": 2}
    assert memory_growth_calls == [(gpu, True)]


def test_load_training_metadata_uses_explicit_metadata_file(tmp_path) -> None:
    metadata_path = tmp_path / "run_metadata.json"
    metadata = {
        "seed": 123,
        "image_size": [224, 224],
        "batch_size": 16,
        "max_examples": 1000,
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    assert load_training_metadata("ignored.keras", str(metadata_path)) == metadata


def test_load_training_metadata_rejects_missing_explicit_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_training_metadata("ignored.keras", str(tmp_path / "missing.json"))


def test_load_class_names_preserves_training_order(tmp_path) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps({"class_names": ["Tomato_healthy", "Tomato_Late_blight"]}),
        encoding="utf-8",
    )

    assert load_class_names(str(metadata_path)) == ["Tomato_healthy", "Tomato_Late_blight"]
