from __future__ import annotations

import argparse
import json

import pandas as pd
import tensorflow as tf
from tensorflow import keras

from plant_village.config import (
    MODELS_DIR,
    PLOTS_DIR,
    REPORTS_DIR,
    ensure_artifact_dirs,
    make_experiment_config,
)
from plant_village.data import configure_tensorflow_resources, load_datasets, set_global_seed
from plant_village.models import build_model
from plant_village.plots import plot_history


def main() -> None:
    args = parse_args()
    ensure_artifact_dirs()

    config = make_experiment_config(
        image_size=args.image_size,
        batch_size=args.batch_size,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    configure_tensorflow_resources()
    set_global_seed(config.seed)

    bundle = load_datasets(config, max_examples=args.max_examples)
    model = build_model(args.model, bundle.num_classes, config, fine_tune=args.fine_tune)

    run_name = args.run_name or args.model
    model_path = MODELS_DIR / f"{run_name}.keras"
    history_path = REPORTS_DIR / f"{run_name}_history.csv"
    metadata_path = REPORTS_DIR / f"{run_name}_metadata.json"
    history_plot_path = PLOTS_DIR / f"{run_name}_history.png"

    callbacks = [
        keras.callbacks.ModelCheckpoint(model_path, monitor="val_accuracy", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.2),
    ]

    history = model.fit(
        bundle.train,
        validation_data=bundle.val,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    model.save(model_path)
    pd.DataFrame(history.history).to_csv(history_path, index=False)
    plot_history(history.history, history_plot_path)

    metadata = {
        "model": args.model,
        "fine_tune": args.fine_tune,
        "num_classes": bundle.num_classes,
        "class_names": bundle.class_names,
        "total_examples": bundle.total_examples,
        "train_examples": bundle.train_examples,
        "val_examples": bundle.val_examples,
        "test_examples": bundle.test_examples,
        "image_size": config.image_size,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "seed": config.seed,
        "train_ratio": config.train_ratio,
        "val_ratio": config.val_ratio,
        "test_ratio": config.test_ratio,
        "max_examples": args.max_examples,
        "tensorflow_version": tf.__version__,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Modelo salvo em: {model_path}")
    print(f"Historico salvo em: {history_path}")
    print(f"Grafico salvo em: {history_plot_path}")
    print(f"Metadados salvos em: {metadata_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina modelos no PlantVillage.")
    parser.add_argument("--model", choices=["baseline", "mobilenet"], default="mobilenet")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--fine-tune", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
