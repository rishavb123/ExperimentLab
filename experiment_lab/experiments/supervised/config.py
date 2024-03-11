"""Python file containing the configs for the supervised learning experiments."""

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore

from experiment_lab.core.base_config import BaseConfig


@dataclass
class SupervisedConfig(BaseConfig):
    """The Supervised dataclass."""

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    batch_size: int | None = 64

    def __post_init__(self) -> None:
        """Does additional checks on the loaded config."""
        super().__post_init__()
        assert (
            self.train_ratio + self.val_ratio + self.test_ratio == 1.0
        ), "All of the data must be split up into train, val, and test such that train_ratio + val_ratio + test_ratio == 1.0"


def register_configs():
    """Registers the supervised config."""
    cs = ConfigStore.instance()
    cs.store(name="supervised_config", node=SupervisedConfig)
