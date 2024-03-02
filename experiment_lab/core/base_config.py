"""The file to hold the base configuration object."""

from dataclasses import dataclass
from enum import Enum
from hydra.core.config_store import ConfigStore


class MultiRunMethodEnum(int, Enum):
    series = 0
    parallel = 1


@dataclass
class BaseConfig:
    experiment_name: str = "run"

    seed: int | None = None

    n_runs: int = 1
    multi_run_method: MultiRunMethodEnum = MultiRunMethodEnum.series


cs = ConfigStore.instance()
cs.store(name="base_config", node=BaseConfig)
