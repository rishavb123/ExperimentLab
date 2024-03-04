"""Main entrypoint to run generic rl experiments."""

from experiment_lab.core.runner import run_experiment
from experiment_lab.experiments.rl import RLExperiment, RLConfig, register_configs


def run_rl_experiment():
    """Main function to run an rl experiment"""
    run_experiment(
        experiment_cls=RLExperiment,
        config_cls=RLConfig,
        register_configs=register_configs,
        config_path="./configs/rl",
    )


if __name__ == "__main__":
    run_rl_experiment()
