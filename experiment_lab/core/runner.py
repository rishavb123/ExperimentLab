import os

from typing import Any, Callable, Type, List
import hydra
from omegaconf import DictConfig, OmegaConf

from experiment_lab.common.resolvers import register_resolvers
from experiment_lab.core.base_config import BaseConfig, register_configs
from experiment_lab.core.base_experiment import BaseExperiment


def run_experiment(
    experiment_cls: Type[BaseExperiment],
    config_cls: Type[BaseConfig] = BaseConfig,
    register_configs: Callable[[], None] = register_configs,
    register_resolvers: Callable[[], None] = register_resolvers,
    config_path: str = "./configs",
    config_name: str = "config",
) -> List[Any]:
    register_resolvers()
    register_configs()

    config_path = os.path.join(os.getcwd(), config_path)

    @hydra.main(config_path=config_path, config_name=config_name, version_base=None)
    def main(dict_cfg: DictConfig) -> List[Any]:
        OmegaConf.resolve(dict_cfg)
        cfg: config_cls = OmegaConf.to_object(dict_cfg)  # type: ignore
        e = experiment_cls(cfg)
        return e.run()

    return main()
