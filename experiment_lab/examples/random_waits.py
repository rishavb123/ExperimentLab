import time
import numpy as np
import hydra
import wandb

from experiment_lab.core.base_config import BaseConfig, register_configs
from experiment_lab.core.base_experiment import BaseExperiment


class RandomWaits(BaseExperiment):

    def __init__(self, cfg: BaseConfig) -> None:
        super().__init__(cfg)

    def single_run(self, seed: int | None = None) -> None:
        print("Starting")
        rng = np.random.default_rng(seed=seed)
        num = rng.integers(10)
        if self.use_wandb:
            wandb.log({"num": num})
        time.sleep(rng.integers(num))
        print("Done!")

register_configs()

@hydra.main(
    config_path="../../configs", 
    config_name="base_config", 
    version_base=None
)
def main(cfg: BaseConfig):
    e = RandomWaits(cfg=cfg)
    e.run()


if __name__ == "__main__":
    main()
