"""The file to hold the base experiment code."""

from typing import Any, List

import abc
import logging
import multiprocessing as mp

from experiment_lab.common.utils import time_f
from experiment_lab.core.base_config import BaseConfig, MultiRunMethodEnum

logger = logging.getLogger(__name__)


class BaseExperiment(abc.ABC):
    """A generic base experiment class for all experiments to inherit from."""

    def __init__(self, cfg: BaseConfig) -> None:
        """The constructor for the base experiment class."""
        self.cfg = cfg
        self.full_name = f"{self.__class__.__name__.lower()}_{self.cfg.experiment_name}"

        assert self.cfg.n_runs >= 0

    @time_f
    @abc.abstractmethod
    def single_run(self, seed: int | None = None) -> Any:
        """The entrypoint to the experiment.

        Args:
            run_num (int): The run number/index of the single run.
            seed (int): The random seed to use for the experiment run.

        Returns:
            Any: The results from this experiment run.
        """
        pass

    @time_f
    def run(self) -> List[Any]:
        """Runs the experiment multiple times in series and aggregates the results.

        Args:
            n_runs (int, optional): The number of times to run the experiment. Defaults to 1.
            seed (int | None, optional): The initial seed of the first experiment run. The seeds of further experiment runs increment by 1 each run. Defaults to None.

        Returns:
            List[Any]: The list of results from the runs.
        """
        if self.cfg.n_runs == 1:
            results = self.single_run(seed=self.cfg.seed)
        elif self.cfg.multi_run_method == MultiRunMethodEnum.series:
            results = [
                self.single_run(
                    seed=None if self.cfg.seed is None else self.cfg.seed + run_num
                )
                for run_num in range(self.cfg.n_runs)
            ]
        else:
            with mp.Pool() as pool:
                results = pool.map(
                    func=self.single_run,
                    iterable=[
                        (None if self.cfg.seed is None else self.cfg.seed + run_num)
                        for run_num in range(self.cfg.n_runs)
                    ],
                )
        return results
