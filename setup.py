from setuptools import find_packages, setup

# Required dependencies
required = [
    "tensorflow>=2.0",
    "mujoco-py<2.1,>=2.0",
    "metaworld @ git+https://github.com/rlworkgroup/metaworld.git@0875192baaa91c43523708f55866d98eaf3facaf#egg=metaworld",
    "pandas",
    "matplotlib",
    "seaborn",
]

setup(
    name="COOM",
    description="COOM: Benchmarking Continual Reinforcement Learning on Doom",
    packages=find_packages(),
    include_package_data=True,
    install_requires=required,
)
