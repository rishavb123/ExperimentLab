import logging
from typing import Dict
import numpy as np

from experiment_lab.experiments.monte_carlo import UniformSampler, BasePostProcessor

logger = logging.getLogger(__name__)


class RandomPointInCircle(UniformSampler):

    def __init__(self) -> None:
        super().__init__(low=0, high=1, k=2)

    def sample(self, num_samples: int = 1) -> np.ndarray:
        return (np.linalg.norm(super().sample(num_samples), axis=1) < 1).astype(int)


class MultiplyByFour(BasePostProcessor):

    def __init__(self) -> None:
        super().__init__()

    def process(self, result: Dict[str, float]) -> Dict[str, float]:
        return {"pi_estimate": result["mean"] * 4, **result}
