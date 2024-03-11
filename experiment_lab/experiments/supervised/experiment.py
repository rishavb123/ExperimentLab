"""The supervised learning experiment file."""

from typing import Any
from experiment_lab.core.base_experiment import BaseExperiment
from experiment_lab.experiments.supervised.config import SupervisedConfig


class SupervisedExperiment(BaseExperiment):
    """The supervised learning experiment class."""

    def __init__(self, cfg: SupervisedConfig) -> None:
        """The constructor for the supervised learning experiment."""
        self.cfg = cfg
        self.initialize_experiment()

    def initialize_experiment(self) -> None:
        """Initializes the supervised experiment config."""
        super().initialize_experiment()

    def single_run(
        self, run_id: str, run_output_path: str, seed: int | None = None
    ) -> Any:
        """A single run of the supervised learning experiment.

        Args:
            run_id (str): The run id.
            run_output_path (str): The run output path.
            seed (int | None, optional): The initial random seed. Defaults to None.

        Returns:
            Any: The post processed results.
        """
        pass
