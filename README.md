# COOM

COOM is a benchmark for continual reinforcement learning. It is based on tasks created in the [ViZDoom](https://github.com/mwydmuch/ViZDoom) platform.

[//]: # (The core of our benchmark is CW20 sequence, in which 20 tasks are run, each with budget of 1M steps.)

[//]: # (We provide the complete source code for the benchmark together with the tested algorithms implementations and code for producing result tables and plots.)

## Installation
1. Install the dependencies for ViZDoom: [Linux](https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md#-linux), [MacOS](https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md#-linux) or [Windows](https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md#-windows).
2. Clone the repository
```bash
$ git clone https://github.com/TTomilin/COOM
```
3. Navigate into the repository
```bash
$ cd COOM
```
4. Install the dependencies 
```bash 
$ python install setup.py
```

# Running

You can run single task, continual learning or multi-task learning experiments with `run_single.py`, `run_cl.py`
, `run_mt.py` scripts, respectively.

To see available script arguments, run with `--help` option, e.g.

`python3 run_single.py --help`

## Examples

Below are given example commands that will run experiments with a very limited scale.

### Single task

`python3 run_single.py --scenario health_gathering --test_envs lava slime`

### Continual learning

#### Task sequence CE4 with PackNet
`python3 run_cl.py --scenarios seek_and_slay --envs default red blue obstacles --cl_method packnet --packnet_retrain_steps 10000 --clipnorm 2e-05`

#### Task sequence CE8 with PackNet
`python3 run_cl.py --scenarios seek_and_slay --envs default red blue obstacles mixed_enemies shadows invulnerable shadows_obstacles --cl_method packnet --packnet_retrain_steps 10000 --clipnorm 2e-05`

#### Task sequence CO4 with PackNet
`python3 run_cl.py --scenarios chainsaw raise_the_roof seek_and_slay health_gathering --cl_method packnet --packnet_retrain_steps 10000 --clipnorm 2e-05`

#### Task sequence CO8 with PackNet
`python3 run_cl.py --scenarios chainsaw raise_the_roof seek_and_slay health_gathering arms_dealer floor_is_lava hide_and_seek parkour --cl_method packnet --packnet_retrain_steps 10000 --clipnorm 2e-05`

# Acknowledgements

COOM heavily relies on [ViZDoom](https://github.com/mwydmuch/ViZDoom).

The implementation of Discrete SAC used in our code comes from [Tianshou](https://github.com/thu-ml/tianshou).

Our experiments were managed using [WandB](https://wandb.ai).
