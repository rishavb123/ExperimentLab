"""Tests the environment construction and step function with the env wrapper for the rl experiments."""

from typing import Any, SupportsFloat, Tuple, Dict
import gymnasium as gym
import numpy as np

from experiment_lab.experiments.rl.environment import GeneralVecEnv


class CounterEnv(gym.Env):

    def __init__(
        self, maxiter: int = 20, render_mode: str | None = None, randomness: int = 2
    ) -> None:
        super().__init__()
        self.rng = np.random.default_rng()
        self.observation_space = gym.spaces.Discrete(10)
        self.action_space = gym.spaces.Discrete(1)
        self.maxiter = maxiter
        self.randomness = randomness

    def reset(
        self, *, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[Any, Dict[str, Any]]:
        self.rng = np.random.default_rng(seed=seed)
        self.counter = self.rng.integers(3, 7)

        self.iter = 0
        return self.counter, {}

    def step(
        self, action: Any
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        self.counter += int(
            action * 4
            - 2
            + self.rng.integers(self.randomness + 1)
            - self.randomness / 2
        )
        self.iter += 1

        rew = 0.0
        terminated = False
        truncated = False
        if self.counter == 9:
            rew = 1.0
            terminated = True
        elif self.counter < 0 or self.counter > 9:
            rew = -1.0
            terminated = True
            self.counter = max(min(9, self.counter), 0)

        if self.iter >= self.maxiter:
            truncated = True

        return self.counter, rew, terminated, truncated, {}


if "CountereEnv" not in gym.registry.keys():
    gym.register(
        id="CounterEnv",
        entry_point="experiment_lab.tests.experiments.rl.test_environment:CounterEnv",
    )


def constant_action_in_counter_env(seed: int = 0, action: int = 1) -> float:
    """Applying a constant action in the counter env.

    Args:
        seed (int, optional): The seed to reset with. Defaults to 0.
        action (int, optional): The constant action to use. Defaults to 1.

    Returns:
        float: The total reward.
    """
    env = gym.make("CounterEnv")
    env.reset(seed=seed)

    total_reward = 0

    done = False
    while not done:
        _obs, rew, terminated, truncated, _info = env.step(action)
        total_reward += float(rew)
        done = terminated or truncated

    return total_reward


def test_constant_action_in_counter() -> None:
    """Tests different actions and seeds with constant_action_in_counter_env"""
    failing_seed = 33
    success_seed = 1

    wrong_action = 0
    correct_action = 1

    # Failing seed for constant 1 action
    assert (
        constant_action_in_counter_env(seed=failing_seed, action=correct_action) == -1
    )
    assert (
        constant_action_in_counter_env(seed=failing_seed, action=correct_action) == -1
    )

    # Failing seed with bad action
    assert constant_action_in_counter_env(seed=failing_seed, action=wrong_action) == -1
    assert constant_action_in_counter_env(seed=failing_seed, action=wrong_action) == -1

    # Success seed for constant 1 action
    assert constant_action_in_counter_env(seed=success_seed, action=correct_action) == 1
    assert constant_action_in_counter_env(seed=success_seed, action=correct_action) == 1

    # Success seed with bad action
    assert constant_action_in_counter_env(seed=success_seed, action=wrong_action) == -1
    assert constant_action_in_counter_env(seed=success_seed, action=wrong_action) == -1


def test_general_vec_wrapper() -> None:
    """Ensures to errors while running the GeneralVecEnv."""
    n_envs = 2
    env = GeneralVecEnv(
        {"env_id": "CounterEnv", "randomness": [0, 2]},
        transfer_steps=[10],
        n_envs=n_envs,
    )
    env.reset()
    for _ in range(50):
         env.step(np.ones(n_envs))
