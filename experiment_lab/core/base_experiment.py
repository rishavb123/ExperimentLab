"""The file to hold the base experiment code."""

import abc
from typing import Any, List, Optional


class BaseExperiment(abc.ABC):
    """A generic base experiment class for all experiments to inherit from."""

    def __init__(self, cfg: Any) -> None:
        """The constructor for the base experiment class.

        Args:
            cfg (Any): The config for the experiment run.
        """
        self.cfg = cfg

    @abc.abstractmethod
    def run(self, seed: Optional[int] = None) -> Any:
        """The entrypoint to the experiment.

        Args:
            seed (int): The random seed to use for the experiment run.
        """
        pass

    def nruns(self, n_runs: int = 1, initial_seed: Optional[int] = None) -> List[Any]:
        """Runs the experiment multiple times in series and aggregates the results.

        Args:
            n_runs (int, optional): The number of times to run the experiment. Defaults to 1.
            initial_seed (Optional[int], optional): The initial seed of the first experiment run. The seeds of further experiment runs increment by 1 each run. Defaults to None.
        """
        return [
            self.run(seed=None if initial_seed is None else initial_seed + run_num)
            for run_num in range(n_runs)
        ]
