import time
import numpy as np
import wandb
import logging

from experiment_lab.core.base_config import BaseConfig
from experiment_lab.core.base_experiment import BaseExperiment
from experiment_lab.core.runner import run_experiment

logger = logging.getLogger(__name__)

class RandomWaits(BaseExperiment):

    def __init__(self, cfg: BaseConfig) -> None:
        assert type(cfg) == BaseConfig
        super().__init__(cfg)

    def single_run(self, run_id: str = "", seed: int | None = None) -> None:
        logger.info("Starting")
        rng = np.random.default_rng(seed=seed)
        num = rng.integers(10)
        if self.use_wandb:
            wandb.log({"num": num})
        time.sleep(rng.integers(num))
        logger.info("Done!")


if __name__ == "__main__":
    run_experiment(experiment_cls=RandomWaits)
