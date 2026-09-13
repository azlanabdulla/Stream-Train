from dataclasses import dataclass
from typing import Optional, Union

import yaml

from .exceptions import ConfigurationError


@dataclass
class ModelConfig:
    name: str

@dataclass
class TrainingConfig:
    epochs: int
    batch_size: Union[str, int]
    gradient_accumulation: Union[str, int]
    precision: str

@dataclass
class ResourcesConfig:
    mode: str
    max_ram: Optional[str]
    max_vram: Optional[str]
    max_cpu_percent: Optional[int]

@dataclass
class CheckpointConfig:
    directory: str
    interval_steps: int
    keep_last: int

@dataclass
class LoggingConfig:
    directory: str
    interval_steps: int

@dataclass
class StreamTrainConfig:
    model: ModelConfig
    training: TrainingConfig
    resources: ResourcesConfig
    checkpoint: CheckpointConfig
    logging: LoggingConfig

    @classmethod
    def from_yaml(cls, path: str) -> "StreamTrainConfig":
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            raise ConfigurationError(f"Failed to read config file {path}: {e}")
        
        if not data:
            raise ConfigurationError("Config file is empty.")

        try:
            return cls(
                model=ModelConfig(**data.get("model", {})),
                training=TrainingConfig(**data.get("training", {})),
                resources=ResourcesConfig(**data.get("resources", {})),
                checkpoint=CheckpointConfig(**data.get("checkpoint", {})),
                logging=LoggingConfig(**data.get("logging", {}))
            )
        except TypeError as e:
            raise ConfigurationError(f"Missing or invalid configuration keys: {e}")

