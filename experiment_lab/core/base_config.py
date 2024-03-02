"""The file to hold the base configuration object."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict
from hydra.core.config_store import ConfigStore


class NRunMethodEnum(Enum):
    """Method to run multi runs (series or parallel)
    """
    series = 0
    parallel = 1


@dataclass
class BaseConfig:
    """Basic configuration for general experiment runs."""

    experiment_name: str | None = None

    seed: int | None = None

    n_runs: int = 1
    n_run_method: NRunMethodEnum = NRunMethodEnum.series

    wandb: Dict[str, Any] | None = None
    ignore_wandb: bool = False


def register_configs():
    """Registers the configs created."""
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=BaseConfig)
