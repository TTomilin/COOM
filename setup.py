from setuptools import find_packages, setup

# Required dependencies
required = [
    "tensorflow==2.5.0",
    "tensorflow-probability==0.11.0",
    "matplotlib",
    "seaborn",
    "wandb",
    "vizdoom",
    "gym",
    "gymnasium==0.28.0",
    "numpy==1.19.5",
    "moviepy",
    "opencv-python",
    "imageio-ffmpeg",
    "promise",
    "scipy==1.8.0",
]

setup(
    name="COOM",
    description="COOM: Benchmarking Continual Reinforcement Learning on Doom",
    packages=find_packages(),
    include_package_data=True,
    install_requires=required,
)
