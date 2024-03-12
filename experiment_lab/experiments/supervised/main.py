"""Main entrypoint to run supervised learning experiments."""

from experiment_lab.core.runner import run_experiment
from experiment_lab.experiments.supervised.experiment import SupervisedExperiment
from experiment_lab.experiments.supervised.config import (
    SupervisedConfig,
    register_configs,
)

from functools import partial

run_supervised_experiment = partial(
    run_experiment,
    experiment_cls=SupervisedExperiment,
    config_cls=SupervisedConfig,
    register_configs=register_configs,
    config_path="./configs/supervised",
)


if __name__ == "__main__":
    run_supervised_experiment()
