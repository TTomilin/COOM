from setuptools import find_packages, setup

# Required dependencies
required = [
    "tensorflow",
    "tensorflow-probability",
    "matplotlib",
    "seaborn",
    "wandb",
    "vizdoom",
    "gym",
    "numpy",
    "moviepy",
    "opencv-python",
    "imageio-ffmpeg",
    "promise",
    "numba",
    "scipy",
]

setup(
    name="COOM",
    description="COOM: Benchmarking Continual Reinforcement Learning on Doom",
    packages=find_packages(),
    include_package_data=True,
    install_requires=required,
)
