from setuptools import find_packages, setup

# Required dependencies
coom_requirements = [
    "vizdoom",
    "opencv-python",
    "scipy==1.11.4",
    "gymnasium==0.28.1"
]

cl_requirements = [
    "tensorflow==2.11",
    "tensorflow-probability==0.19",
    "wandb",
]

results_processing_requirements = [
    "wandb",
    "matplotlib",
    "seaborn",
    'pandas',
]

setup(
    name="COOM",
    description="COOM: Benchmarking Continual Reinforcement Learning on Doom",
    version='1.0.0',
    url='https://github.com/TTomilin/COOM',
    author='Tristan Tomilin',
    author_email='tristan.tomilin@hotmail.com',
    license='MIT',
    keywords=["continual learning", "vizdoom", "reinforcement learning", "benchmarking"],
    packages=find_packages(),
    include_package_data=True,
    install_requires=coom_requirements,
    extras_require={
        'cl': cl_requirements,
        'results': results_processing_requirements,
    },
)
