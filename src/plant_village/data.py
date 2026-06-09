from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf
import tensorflow_datasets as tfds

from plant_village.config import ExperimentConfig, TFDS_DIR


@dataclass(frozen=True)
class DatasetBundle:
    train: tf.data.Dataset
    val: tf.data.Dataset
    test: tf.data.Dataset
    class_names: list[str]
    num_classes: int
    total_examples: int
    train_examples: int
    val_examples: int
    test_examples: int


@dataclass(frozen=True)
class SplitCounts:
    train: int
    val: int
    test: int

    @property
    def total(self) -> int:
        return self.train + self.val + self.test


def set_global_seed(seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def configure_tensorflow_resources(intra_op_threads: int = 2, inter_op_threads: int = 1) -> None:
    tf.config.threading.set_intra_op_parallelism_threads(intra_op_threads)
    tf.config.threading.set_inter_op_parallelism_threads(inter_op_threads)
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)


def load_datasets(config: ExperimentConfig, max_examples: int | None = None) -> DatasetBundle:
    """Load PlantVillage and create deterministic train/validation/test splits."""
    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples deve ser positivo quando informado.")

    dataset, info = tfds.load(
        "plant_village",
        split="train",
        as_supervised=True,
        with_info=True,
        data_dir=str(TFDS_DIR),
        shuffle_files=False,
    )

    total_examples = int(info.splits["train"].num_examples)
    if max_examples is not None:
        total_examples = min(total_examples, max_examples)
        dataset = dataset.take(total_examples)

    dataset = dataset.shuffle(
        min(total_examples, config.split_shuffle_buffer),
        seed=config.seed,
        reshuffle_each_iteration=False,
    )

    split = split_counts(total_examples, config)

    train = dataset.take(split.train)
    val = dataset.skip(split.train).take(split.val)
    test = dataset.skip(split.train + split.val).take(split.test)

    train = _prepare(train, config, training=True)
    val = _prepare(val, config, training=False)
    test = _prepare(test, config, training=False)

    class_names = list(info.features["label"].names)
    return DatasetBundle(
        train=train,
        val=val,
        test=test,
        class_names=class_names,
        num_classes=len(class_names),
        total_examples=total_examples,
        train_examples=split.train,
        val_examples=split.val,
        test_examples=split.test,
    )


def split_counts(total_examples: int, config: ExperimentConfig) -> SplitCounts:
    if total_examples <= 0:
        raise ValueError("total_examples deve ser positivo.")

    train_count = int(total_examples * config.train_ratio)
    val_count = int(total_examples * config.val_ratio)
    test_count = total_examples - train_count - val_count

    if min(train_count, val_count, test_count) <= 0:
        raise ValueError(
            "A quantidade de exemplos deve permitir treino, validacao e teste nao vazios."
        )

    return SplitCounts(train=train_count, val=val_count, test=test_count)


def _prepare(
    dataset: tf.data.Dataset,
    config: ExperimentConfig,
    *,
    training: bool,
) -> tf.data.Dataset:
    dataset = dataset.map(
        lambda image, label: _resize(image, label, config),
        num_parallel_calls=config.num_parallel_calls,
    )
    if training:
        dataset = dataset.shuffle(config.train_shuffle_buffer, seed=config.seed)
    return dataset.batch(config.batch_size).prefetch(config.prefetch_buffer)


def _resize(
    image: tf.Tensor,
    label: tf.Tensor,
    config: ExperimentConfig,
) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.image.resize(image, config.image_size)
    image = tf.cast(image, tf.float32)
    return image, label
