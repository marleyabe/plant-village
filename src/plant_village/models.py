from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from plant_village.config import ExperimentConfig


def load_trained_model(model_path: str) -> keras.Model:
    return tf.keras.models.load_model(
        model_path,
        custom_objects={
            "preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input,
        },
    )


def build_model(
    model_name: str,
    num_classes: int,
    config: ExperimentConfig,
    *,
    fine_tune: bool = False,
) -> keras.Model:
    if model_name == "baseline":
        return build_baseline_cnn(num_classes, config)
    if model_name == "mobilenet":
        return build_mobilenet(num_classes, config, fine_tune=fine_tune)
    raise ValueError(f"Modelo desconhecido: {model_name}")


def build_baseline_cnn(num_classes: int, config: ExperimentConfig) -> keras.Model:
    inputs = keras.Input(shape=(*config.image_size, 3))
    x = _augmentation()(inputs)
    x = layers.Rescaling(1.0 / 255.0)(x)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="baseline_cnn")
    return _compile(model, config)


def build_mobilenet(
    num_classes: int,
    config: ExperimentConfig,
    *,
    fine_tune: bool,
) -> keras.Model:
    inputs = keras.Input(shape=(*config.image_size, 3))
    x = _augmentation()(inputs)
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenet_preprocess")(x)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*config.image_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = fine_tune

    if fine_tune:
        for layer in base_model.layers[:-30]:
            layer.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="mobilenet_v2")
    return _compile(model, config)


def _augmentation() -> keras.Sequential:
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.10),
        ],
        name="augmentation",
    )


def _compile(model: keras.Model, config: ExperimentConfig) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
