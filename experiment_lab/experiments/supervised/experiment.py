"""The supervised learning experiment file."""

from typing import Any, Dict
import os
import hydra
import torch
import torch.utils.data

from experiment_lab.common.utils import default
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

        self.dataset: torch.utils.data.Dataset = hydra.utils.instantiate(
            self.cfg.dataset
        )
        self.device: torch.device = torch.device(self.cfg.device)
        self.data_loader_kwargs: Dict[str, Any] = default(
            self.cfg.data_loader_kwargs, {}
        )

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
        # Setup the device to train on
        if self.cfg.gpu_idx is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.cfg.gpu_idx)

        if seed is not None:
            torch.manual_seed(seed=seed)

        train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(
            dataset=self.dataset,
            lengths=[self.cfg.train_ratio, self.cfg.val_ratio, self.cfg.test_ratio],
        )

        train_dataloader, validation_dataloader, test_dataloader = (
            torch.utils.data.DataLoader(
                dataset=train_dataset, **self.data_loader_kwargs
            ),
            torch.utils.data.DataLoader(
                dataset=validation_dataset, **self.data_loader_kwargs
            ),
            torch.utils.data.DataLoader(
                dataset=test_dataset, **self.data_loader_kwargs
            ),
        )

        model = hydra.utils.instantiate(self.cfg.model)

        import pdb

        pdb.set_trace()
