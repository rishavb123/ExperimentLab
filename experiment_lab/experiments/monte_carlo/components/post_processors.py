"""Python file for the post proessors"""

from typing import Any
import abc


class BasePostProcessor(abc.ABC):
    """The base post processor class."""

    def __init__(self) -> None:
        """The constructor for a generic post processor."""
        super().__init__()

    @abc.abstractmethod
    def process(self, result: Any) -> Any:
        """Applies post processing on the aggregated result.

        Args:
            result (Any): The aggregated result.

        Returns:
            Any: The post processed result
        """
        pass


class PassThroughPostProcessor(BasePostProcessor):
    """The pass through post processor."""

    def __init__(self) -> None:
        """The constructor for the pass through post processor."""
        super().__init__()

    def process(self, result: Any) -> Any:
        """Returns the passed in result.

        Args:
            result (Any): The aggregated result.

        Returns:
            Any: The same result.
        """
        return result
