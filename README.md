# ExperimentLab

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

For specific optional dependencies use:

```bash
pip install -e ".[{name}]"
```

where `{name}` can be `all`, `rl`, `rl-vid`, or `dev`.

To just install the experiment runner framework in one command:

```bash
pip install git+https://github.com/rishavb123/ExperimentLab.git
```

### Debugging Tips

#### Recording Videos on RL experiments

The video recording wrapper from the stable baselines library requires a few dependencies that must be installed before it can be used.
1. Install moviepy to your python environment. This can be done using through this package via:
```bash
pip install "experiment_lab[rl-vid]"
```
or directly using,
```bash
pip install moviepy
```
2. Install ffmpeg on your system using the following command (depending on your OS):
- Ubuntu: `sudo apt install ffmpeg` 
- Mac: `brew instlal ffmpeg`
- Windows: Follow the instructions [here](https://phoenixnap.com/kb/ffmpeg-windows). Note that I did not write or review this instructions.
3. Make sure to set the location of the ffmpeg binary to the `$IMAGEIO_F

Also, it should be noted that there are some issues with using the multiprocessing vector environment with the video recorder. Ensure that the start_method set in the RL config is "spawn" (which is the default if it is left as None) for best results. On MacOS, see [this](https://stackoverflow.com/questions/50168647/multiprocessing-causes-python-to-crash-and-gives-an-error-may-have-been-in-progr) page for potential problems with using "fork".
