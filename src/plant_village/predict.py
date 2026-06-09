from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from plant_village.config import ExperimentConfig, make_experiment_config
from plant_village.models import load_trained_model


def main() -> None:
    args = parse_args()
    config = make_experiment_config(image_size=args.image_size)

    model = load_trained_model(args.model_path)
    class_names = load_class_names(args.class_names)
    image = load_image(args.image_path, config)

    probabilities = model.predict(image, verbose=0)[0]
    best_index = int(np.argmax(probabilities))
    confidence = float(probabilities[best_index])

    print(f"Classe prevista: {class_names[best_index]}")
    print(f"Confianca: {confidence:.4f}")


def load_image(path: str, config: ExperimentConfig) -> tf.Tensor:
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, config.image_size)
    image = tf.cast(image, tf.float32)
    return tf.expand_dims(image, axis=0)


def load_class_names(metadata_path: str) -> list[str]:
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    return list(metadata["class_names"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classifica uma imagem de folha.")
    parser.add_argument("model_path")
    parser.add_argument("image_path")
    parser.add_argument(
        "--class-names",
        required=True,
        help="Arquivo *_metadata.json gerado no treinamento.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


if __name__ == "__main__":
    main()
