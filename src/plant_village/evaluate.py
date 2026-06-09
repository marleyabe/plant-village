from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from plant_village.config import PLOTS_DIR, REPORTS_DIR, ensure_artifact_dirs, make_experiment_config
from plant_village.data import configure_tensorflow_resources, load_datasets, set_global_seed
from plant_village.models import load_trained_model
from plant_village.plots import plot_confusion_matrix, plot_prediction_grid


def main() -> None:
    args = parse_args()
    ensure_artifact_dirs()

    metadata = load_training_metadata(args.model_path, args.metadata_path)
    image_size = args.image_size if args.image_size is not None else int(metadata.get("image_size", [224])[0])
    batch_size = args.batch_size if args.batch_size is not None else int(metadata.get("batch_size", 32))
    seed = args.seed if args.seed is not None else int(metadata.get("seed", 42))
    max_examples = args.max_examples
    if max_examples is None:
        max_examples = metadata.get("max_examples")

    config = make_experiment_config(
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
    )
    configure_tensorflow_resources()
    set_global_seed(config.seed)

    bundle = load_datasets(config, max_examples=max_examples)
    model = load_trained_model(args.model_path)

    y_true, y_pred, confidences, correct_examples, error_examples = collect_predictions(
        model,
        bundle.test,
        bundle.class_names,
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=bundle.class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred)

    run_name = Path(args.model_path).stem
    report_path = REPORTS_DIR / f"{run_name}_classification_report.csv"
    summary_path = REPORTS_DIR / f"{run_name}_metrics.json"
    examples_path = REPORTS_DIR / f"{run_name}_prediction_examples.csv"
    matrix_path = PLOTS_DIR / f"{run_name}_confusion_matrix.png"
    correct_grid_path = PLOTS_DIR / f"{run_name}_correct_examples.png"
    error_grid_path = PLOTS_DIR / f"{run_name}_error_examples.png"

    pd.DataFrame(report).transpose().to_csv(report_path)
    plot_confusion_matrix(matrix, bundle.class_names, matrix_path)
    plot_prediction_grid(correct_examples, correct_grid_path, "Acertos no conjunto de teste")
    plot_prediction_grid(error_examples, error_grid_path, "Erros no conjunto de teste")
    save_prediction_examples(y_true, y_pred, confidences, bundle.class_names, examples_path)

    metrics = {
        "accuracy": report["accuracy"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "total_examples": bundle.total_examples,
        "test_examples": bundle.test_examples,
    }
    summary_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"Relatorio por classe: {report_path}")
    print(f"Matriz de confusao: {matrix_path}")
    print(f"Exemplos de acertos: {correct_grid_path}")
    print(f"Exemplos de erros: {error_grid_path}")
    print(f"Exemplos de predicao: {examples_path}")


def load_training_metadata(model_path: str, metadata_path: str | None) -> dict:
    path = Path(metadata_path) if metadata_path else REPORTS_DIR / f"{Path(model_path).stem}_metadata.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if metadata_path:
        raise FileNotFoundError(f"Metadata nao encontrado: {path}")
    return {}


def collect_predictions(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    class_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], list[dict]]:
    y_true_batches = []
    y_pred_batches = []
    confidence_batches = []
    correct_examples = []
    error_examples = []

    for images, labels in dataset:
        probabilities = model.predict(images, verbose=0)
        predictions = np.argmax(probabilities, axis=1)
        confidences = np.max(probabilities, axis=1)
        y_true_batches.append(labels.numpy())
        y_pred_batches.append(predictions)
        confidence_batches.append(confidences)
        _collect_visual_examples(
            images.numpy(),
            labels.numpy(),
            predictions,
            confidences,
            class_names,
            correct_examples,
            error_examples,
        )

    return (
        np.concatenate(y_true_batches),
        np.concatenate(y_pred_batches),
        np.concatenate(confidence_batches),
        correct_examples,
        error_examples,
    )


def _collect_visual_examples(
    images: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    confidences: np.ndarray,
    class_names: list[str],
    correct_examples: list[dict],
    error_examples: list[dict],
    limit: int = 12,
) -> None:
    for image, actual, predicted, confidence in zip(
        images,
        labels,
        predictions,
        confidences,
        strict=True,
    ):
        target = correct_examples if actual == predicted else error_examples
        if len(target) >= limit:
            continue
        target.append(
            {
                "image": image,
                "actual": class_names[int(actual)],
                "predicted": class_names[int(predicted)],
                "confidence": float(confidence),
            }
        )


def save_prediction_examples(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidences: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    rows = []
    for actual, predicted, confidence in zip(y_true, y_pred, confidences, strict=True):
        rows.append(
            {
                "actual": class_names[int(actual)],
                "predicted": class_names[int(predicted)],
                "confidence": float(confidence),
                "correct": bool(actual == predicted),
            }
        )

    frame = pd.DataFrame(rows)
    examples = pd.concat(
        [
            frame[frame["correct"]].sort_values("confidence", ascending=False).head(20),
            frame[~frame["correct"]].sort_values("confidence", ascending=False).head(20),
        ]
    )
    examples.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avalia um modelo treinado no PlantVillage.")
    parser.add_argument("model_path")
    parser.add_argument("--metadata-path", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
