from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
TFDS_DIR = ROOT_DIR / "data" / "tfds"


@dataclass(frozen=True)
class ExperimentConfig:
    image_size: tuple[int, int] = (224, 224)
    batch_size: int = 32
    seed: int = 42
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    epochs: int = 20
    learning_rate: float = 1e-3
    split_shuffle_buffer: int = 2048
    train_shuffle_buffer: int = 512
    num_parallel_calls: int = 1
    prefetch_buffer: int = 1

    def __post_init__(self) -> None:
        width, height = self.image_size
        if width <= 0 or height <= 0:
            raise ValueError("image_size deve ter valores positivos.")
        if self.batch_size <= 0:
            raise ValueError("batch_size deve ser positivo.")
        if self.epochs <= 0:
            raise ValueError("epochs deve ser positivo.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate deve ser positivo.")
        if self.split_shuffle_buffer <= 0:
            raise ValueError("split_shuffle_buffer deve ser positivo.")
        if self.train_shuffle_buffer <= 0:
            raise ValueError("train_shuffle_buffer deve ser positivo.")
        if self.num_parallel_calls <= 0:
            raise ValueError("num_parallel_calls deve ser positivo.")
        if self.prefetch_buffer <= 0:
            raise ValueError("prefetch_buffer deve ser positivo.")

        ratios = (self.train_ratio, self.val_ratio, self.test_ratio)
        if any(ratio <= 0 for ratio in ratios):
            raise ValueError("train_ratio, val_ratio e test_ratio devem ser positivos.")
        if not abs(sum(ratios) - 1.0) < 1e-9:
            raise ValueError("train_ratio, val_ratio e test_ratio devem somar 1.0.")


def make_experiment_config(
    *,
    image_size: int = 224,
    batch_size: int = 32,
    seed: int = 42,
    epochs: int = 20,
    learning_rate: float = 1e-3,
) -> ExperimentConfig:
    return ExperimentConfig(
        image_size=(image_size, image_size),
        batch_size=batch_size,
        seed=seed,
        epochs=epochs,
        learning_rate=learning_rate,
    )


def ensure_artifact_dirs() -> None:
    for path in (MODELS_DIR, PLOTS_DIR, REPORTS_DIR, TFDS_DIR):
        path.mkdir(parents=True, exist_ok=True)
