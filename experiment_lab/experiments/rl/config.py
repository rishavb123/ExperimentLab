from dataclasses import dataclass
from typing import Any, Dict, List, Type
from hydra.core.config_store import ConfigStore
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy, ActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback

from experiment_lab.core.base_config import BaseConfig


@dataclass
class WandbCallbackConfig:
    verbose: int = 2
    model_save_path: str = "./models"
    model_save_freq: int = 1_000
    gradient_save_freq: int = 0


class RLConfig(BaseConfig):
    env_config: Dict[str, Any]

    total_time_steps: int = 10_000
    n_envs: int = 1

    wrappers: List[gym.Wrapper] | None = None
    wrapper_kwargs_lst: List[Dict[str, Any]] | None = None

    model_cls: Type[BaseAlgorithm] = PPO
    model_kwargs: Dict[str, Any] | None = None

    policy_cls: Type[BasePolicy] = ActorCriticPolicy
    policy_kwargs: Dict[str, Any] | None = None

    callbacks: List[BaseCallback] | None = None
    callback_kwargs_lst: List[Dict[str, Any]] | None = None
    wandb_callback_kwargs: WandbCallbackConfig | None = None

    device: str = "mps"
    gpu_idx: int | None = None

    def __post_init__(self) -> None:
        """Validation checks for rl config and kwargs None replace."""
        if self.wrappers is None:
            self.wrappers = []
        if self.wrapper_kwargs_lst is None:
            self.wrapper_kwargs_lst = []
        for _ in range(len(self.wrapper_kwargs_lst), len(self.wrappers)):
            self.wrapper_kwargs_lst.append({})

        if self.callbacks is None:
            self.callbacks = []
        if self.callback_kwargs_lst is None:
            self.callback_kwargs_lst = []
        for _ in range(len(self.callback_kwargs_lst), len(self.callbacks)):
            self.callback_kwargs_lst.append({})

        if self.model_kwargs is None:
            self.model_kwargs = {}
        if self.policy_kwargs is None:
            self.policy_kwargs = {}


def register_configs():
    cs = ConfigStore.instance()
    cs.store(name="rl_config", node=RLConfig)
