"""Python file for the aggregators."""

import abc
from typing import Any, Callable
import numpy as np


class BaseAggregator(abc.ABC):
    """The base aggregator class."""

    def __init__(self) -> None:
        """The constructor for a generic aggregator."""
        pass

    @abc.abstractmethod
    def aggregate(self, samples: np.ndarray) -> Any:
        """The aggregate function.

        Args:
            samples (np.ndarray): The samples to aggregate.

        Returns:
            Any: The aggregated result.
        """
        pass


class NpAggregator(BaseAggregator):
    """Aggregator to apply a numpy function to the axis=0 of the samples."""

    def __init__(self, np_func: Callable[..., np.ndarray] = np.mean) -> None:
        """The constructor of the NpAggregator that sets the np function to use.

        Args:
            np_func (Callable, optional): The np function to use. Defaults to np.mean.
        """
        super().__init__()
        self.np_func = np_func

    def aggregate(self, samples: np.ndarray) -> Any:
        """The aggregate function that applies the np function to the samples.

        Args:
            samples (np.ndarray): The samples.

        Returns:
            Any: The aggregated result.
        """
        return self.np_func(samples, axis=0)
