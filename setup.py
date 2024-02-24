import pathlib

from setuptools import setup


CWD = pathlib.Path(__file__).absolute().parent


def get_version():
    """Gets the gymnasium version."""
    path = CWD / "gymnasium" / "__init__.py"
    content = path.read_text()

    for line in content.splitlines():
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")


def get_description():
    """Gets the description from the readme."""
    with open("README.md") as fh:
        long_description = "".join(fh.readlines())
    return long_description

setup(
    name="experiment_lab",
    url="https://github.com/rishavb123/ExperimentLab",
)