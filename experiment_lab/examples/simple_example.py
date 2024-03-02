import time
import numpy as np
import hydra

from experiment_lab.core.base_config import BaseConfig
from experiment_lab.core.base_experiment import BaseExperiment


class SimpleExperiment(BaseExperiment):

    def __init__(self, cfg: BaseConfig) -> None:
        super().__init__(cfg)

    def single_run(self, seed: int | None = None) -> None:
        print("Starting")
        rng = np.random.default_rng(seed=seed)
        time.sleep(rng.integers(10))
        print("Done!")


@hydra.main(config_path="../../configs", config_name="base_config", version_base=None)
def main(cfg: BaseConfig):
    e = SimpleExperiment(cfg=cfg)
    e.run()


if __name__ == "__main__":
    main()
