from setuptools import find_packages, setup

# Required dependencies
required = [
    "tensorflow>=2.0",
    "pandas",
    "matplotlib",
    "seaborn",
    "wandb",
    "vizdoom"
]

setup(
    name="COOM",
    description="COOM: Benchmarking Continual Reinforcement Learning on Doom",
    packages=find_packages(),
    include_package_data=True,
    install_requires=required,
)
