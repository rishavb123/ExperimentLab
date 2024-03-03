"""The rl experiment python file."""

import os
from typing import Any
import torch
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import CallbackList

from experiment_lab.core.base_experiment import BaseExperiment
from experiment_lab.experiments.rl.config import RLConfig
from experiment_lab.experiments.rl.environment import GeneralVecEnv


class RLExperiment(BaseExperiment):
    """The rl experiment class."""

    def __init__(self, cfg: RLConfig) -> None:
        """Constructor for rl experiment.

        Args:
            cfg (RLConfig): The config to run the experiment with.
        """
        self.cfg = cfg
        self.initialize_experiment()

    def single_run(self, run_id: str = "", seed: int | None = None) -> Any:
        # Setup the device to train on
        if self.cfg.gpu_idx is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.cfg.gpu_idx)
        device = torch.device(self.cfg.device)

        # Create the environment
        env = GeneralVecEnv(
            env_config=self.cfg.env_config,
            transfer_steps=self.cfg.transfer_steps,
            wrappers=self.cfg.wrappers,
            wrapper_kwargs_lst=self.cfg.wrapper_kwargs_lst,
            n_envs=self.cfg.n_envs,
            seed=self.cfg.seed,
            monitor_dir=self.cfg.monitor_dir,
            monitor_kwargs=self.cfg.monitor_kwargs,
            start_method=self.cfg.start_method,
            render_mode=self.cfg.render_mode,
        )
        env.seed(seed)
        env.reset()

        # Setup the model
        model_kwargs = {} if self.cfg.model_kwargs is None else self.cfg.model_kwargs
        model = self.cfg.model_cls(
            env=env,
            policy=self.cfg.policy_cls,
            policy_kwargs=self.cfg.policy_kwargs,
            device=device,
            tensorboard_log=f"{self.output_directory}/logs/" if self.cfg.log else None,
            **model_kwargs,
        )
        model.set_random_seed(self.cfg.seed)
        model_save_path = f"{self.output_directory}/models/{run_id}/"

        # Instantiate the callbacks
        callback_instances = []
        if (
            self.cfg.log
            and self.use_wandb
            and self.cfg.wandb_callback_kwargs is not None
        ):
            callback_instances.append(
                WandbCallback(
                    gradient_save_freq=self.cfg.wandb_callback_kwargs.gradient_save_freq,
                    model_save_freq=self.cfg.wandb_callback_kwargs.model_save_freq,
                    model_save_path=model_save_path,
                    verbose=self.cfg.wandb_callback_kwargs.verbose,
                )
            )
        if (
            self.cfg.callback_cls_lst is not None
            and self.cfg.callback_kwargs_lst is not None
        ):
            for callback_cls, callback_kwargs in zip(
                self.cfg.callback_cls_lst, self.cfg.callback_kwargs_lst
            ):
                callback_instances.append(callback_cls(**callback_kwargs))

        # Run the algorithm
        model.learn(
            total_timesteps=self.cfg.total_time_steps,
            log_interval=self.cfg.log_interval,
            tb_log_name=run_id,
            callback=(
                CallbackList(callback_instances)
                if len(callback_instances) > 0
                else None
            ),
        )

        if self.cfg.save_model and not self.use_wandb:
            os.makedirs(model_save_path, exist_ok=True)
            model.save(f"{model_save_path}/final.zip")
