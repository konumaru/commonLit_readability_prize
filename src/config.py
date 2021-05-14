from dataclasses import dataclass
from typing import Tuple

from omegaconf import OmegaConf


@dataclass
class PreprocessConfig:
    target_size: Tuple[int, int] = (64, 64)
    normalize: bool = True


@dataclass
class ModelConfig:
    image_size: Tuple[int, int] = (64, 64)
    drop_rate: float = 0.5


@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.001


@dataclass
class ExperimentConfig:
    preprocess: PreprocessConfig = PreprocessConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    logdir: str = "outputs"


if __name__ == "__main__":
    # Ref: https://zenn.dev/dhirooka/articles/f2c12ceae3a0a5
    base_config = OmegaConf.structured(ExperimentConfig)
    cli_config = OmegaConf.from_cli()
    config = OmegaConf.merge(base_config, cli_config)
    print(config)

    OmegaConf.save(config, "../data/versions/config.yaml")
