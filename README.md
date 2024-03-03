# Experiment Lab

A suite of flexible experiments of a variety of different tasks.

## Installation

To create a conda environment for this package use the following commands to do so

```bash
conda create -n experiment_lab python=3.11 -y
conda activate experiment_lab
```

Install the version of pytorch that is compatible with your hardware using the pip or conda command from their [website](https://pytorch.org/get-started/locally/).

Then to install this package and its dependencies use the following commands:

```bash
git clone https://github.com/rishavb123/ExperimentLab.git
cd ExperimentLab
pip install -e .
```

As a test, try running the following script:

```bash
python experiment_lab/experiments/examples/random_waits.py n_runs=20 n_run_method=parallel seed=0
```

For specific types of experiments or dev optional dependencies use:

```bash
pip install -e .[{name}]
```

where `{name}` can be `rl` or `dev`.
