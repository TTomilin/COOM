from setuptools import find_packages, setup

# Required dependencies
required = [
    "tensorflow>=2.0",
    "tensorflow-probability",
    "matplotlib",
    "seaborn",
    "wandb",
    "vizdoom",
    "gym",
    "numpy",
    "moviepy",
    "opencv-python",
    "imageio",
    "imageio-ffmpeg",
    "promise",
]

setup(
    name="COOM",
    description="COOM: Benchmarking Continual Reinforcement Learning on Doom",
    packages=find_packages(),
    include_package_data=True,
    install_requires=required,
)
