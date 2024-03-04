"""The file to hold the base experiment code."""

from typing import Any, List, Tuple

import abc
import logging
import multiprocessing as mp
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import wandb
import time

from experiment_lab.common.utils import time_f, camel_to_snake_case
from experiment_lab.core.base_config import BaseConfig, NRunMethodEnum

logger = logging.getLogger(__name__)


class BaseExperiment(abc.ABC):
    """A generic base experiment class for all experiments to inherit from."""

    def __init__(self, cfg: BaseConfig) -> None:
        """The constructor for the base experiment class."""
        self.cfg = cfg
        self.initialize_experiment()

    def initialize_experiment(self) -> None:
        """Initializes the experiment that is about to get run."""
        if self.cfg.experiment_name:
            self.experiment_name = self.cfg.experiment_name
        else:
            self.experiment_name = (
                f"{camel_to_snake_case(self.__class__.__name__.lower())}"
            )
        self.timestamp = int(time.time())
        self.experiment_id = f"{self.experiment_name}_{self.timestamp}"
        self.output_directory = HydraConfig.get().runtime.output_dir

    @abc.abstractmethod
    def single_run(self, run_id: str = "", seed: int | None = None) -> Any:
        """The entrypoint to the experiment.

        Args:
            run_id (str): The a unique string id for the run.
            seed (int): The random seed to use for the experiment run.

        Returns:
            Any: The results from this experiment run.
        """
        pass

    def _single_run_wrapper(
        self, intial_seed_and_run_num: Tuple[int | None, int] = (None, 0)
    ) -> Any:
        """A wrapper for the single run call to do run specific setup and post processing.

        Args:
            intial_seed_and_run_num (Tuple[int  |  None, int], optional): A tuple with both the seed and the run number. Defaults to (None, 0).

        Returns:
            Any: The results from the experiment run.
        """
        seed, run_num = intial_seed_and_run_num
        seed = None if seed is None else seed + run_num
        logger.info(f"Starting individual run with seed {seed}")
        start_ns = time.time_ns()
        wandb_run = None
        run_id = f"{self.experiment_id}_{run_num}_{self.cfg.n_runs}"
        if not self.cfg.ignore_wandb and self.cfg.wandb:
            wandb_run = wandb.init(
                id=run_id,
                config={
                    **self.cfg.__dict__,
                    "experiment_name": self.experiment_name,
                    "timestamp": self.timestamp,
                    "experiment_id": self.experiment_id,
                },
                reinit=True,
                **self.cfg.wandb,
            )
        result = self.single_run(seed=seed)
        end_ns = time.time_ns()
        logger.info(
            f"Finished run with seed {seed}. Time elapsed: {(end_ns - start_ns) / 1e9}s"
        )
        if wandb_run is not None:
            wandb_run.save(self.output_directory)
            wandb_run.finish()
        return result

    @time_f
    def run(self) -> List[Any]:
        """Runs the experiment multiple times in series and aggregates the results.

        Args:
            n_runs (int, optional): The number of times to run the experiment. Defaults to 1.
            seed (int | None, optional): The initial seed of the first experiment run. The seeds of further experiment runs increment by 1 each run. Defaults to None.

        Returns:
            List[Any]: The list of results from the runs.
        """
        results = None
        if self.cfg.n_runs <= 1 or self.cfg.n_run_method == NRunMethodEnum.series:
            results = [
                self._single_run_wrapper((self.cfg.seed, run_num))
                for run_num in range(self.cfg.n_runs)
            ]
        elif self.cfg.n_run_method == NRunMethodEnum.parallel:
            with mp.Pool() as pool:
                results = pool.map(
                    func=self._single_run_wrapper,
                    iterable=[
                        (self.cfg.seed, run_num) for run_num in range(self.cfg.n_runs)
                    ],
                )
        assert results is not None, "Unknown n_run_method!"
        return results

    @property
    def use_wandb(self) -> bool:
        """Whether or not to use wandb in this experiment

        Returns:
            bool: To use wandb.
        """
        return not self.cfg.ignore_wandb and bool(self.cfg.wandb)
