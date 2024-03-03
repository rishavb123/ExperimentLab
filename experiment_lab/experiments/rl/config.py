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
    """The wandb callback config dataclass."""

    verbose: int = 2
    model_save_freq: int = 1_000
    gradient_save_freq: int = 0


class RLConfig(BaseConfig):
    """The RL Config dataclass."""

    env_config: Dict[str, Any] = {"env_id": "CartPole-v1"}
    transfer_steps: List[int] | None = None

    total_time_steps: int = 10_000
    n_envs: int = 1

    wrappers: List[Type[gym.Wrapper]] | None = None
    wrapper_kwargs_lst: List[Dict[str, Any]] | None = None

    model_cls: Type[BaseAlgorithm] = PPO
    model_kwargs: Dict[str, Any] | None = None

    policy_cls: Type[BasePolicy] = ActorCriticPolicy
    policy_kwargs: Dict[str, Any] | None = None

    callback_cls_lst: List[Type[BaseCallback]] | None = None
    callback_kwargs_lst: List[Dict[str, Any]] | None = None
    wandb_callback_kwargs: WandbCallbackConfig | None = None

    monitor_dir: str | None = None
    monitor_kwargs: Dict[str, Any] | None = None

    start_method: str | None = None
    render_mode: str | None = None

    log: bool = True
    log_interval: int = 1
    save_model: bool = False

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

        if self.callback_cls_lst is None:
            self.callback_cls_lst = []
        if self.callback_kwargs_lst is None:
            self.callback_kwargs_lst = []
        for _ in range(len(self.callback_kwargs_lst), len(self.callback_cls_lst)):
            self.callback_kwargs_lst.append({})

        if self.model_kwargs is None:
            self.model_kwargs = {}
        if self.policy_kwargs is None:
            self.policy_kwargs = {}

        if self.monitor_kwargs is None:
            self.monitor_kwargs = {}

        if self.transfer_steps is None:
            self.transfer_steps = []


def register_configs():
    """Registers the rl config."""
    cs = ConfigStore.instance()
    cs.store(name="rl_config", node=RLConfig)
