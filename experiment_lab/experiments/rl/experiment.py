from typing import Any
from experiment_lab.core.base_config import BaseConfig
from experiment_lab.core.base_experiment import BaseExperiment


class RLExperiment(BaseExperiment):

    def __init__(self, cfg: BaseConfig) -> None:
        super().__init__(cfg)

    def single_run(self, seed: int | None = None) -> Any:
        print("This is an RL experiment!")
        return 0
