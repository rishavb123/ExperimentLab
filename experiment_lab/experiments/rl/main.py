from experiment_lab.core.runner import run_experiment
from experiment_lab.experiments.rl import RLExperiment, RLConfig, register_configs


def run_rl_experiment():
    run_experiment(
        experiment_cls=RLExperiment,
        config_cls=RLConfig,
        register_configs=register_configs,
        config_path="./configs/rl",
    )
