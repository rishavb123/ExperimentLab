from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore

from experiment_lab.core.base_config import BaseConfig


class RLConfig(BaseConfig):
    pass


def register_configs():
    cs = ConfigStore.instance()
    cs.store(name="rl_config", node=RLConfig)
