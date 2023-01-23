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

### Single task

#### Quick start  
`python3 run_single.py --scenario pitfall`

#### Training examples  
`python3 run_single.py --scenario health_gathering --envs obstacles --test_envs lava slime --seed 0 --steps 2e5 --log_every 250`

### Continual learning

#### Quick start  
`python run_cl.py --sequence CO4 --cl_method packnet`

#### Training examples
```
python run_cl.py --sequence CD4 --cl_method packnet --packnet_retrain_steps 10000 --clipnorm 2e-05
python run_cl.py --sequence CO8 --cl_method agem --regularize_critic True --episodic_mem_per_task 10000 --episodic_batch_size 128
python run_cl.py --sequence COC --batch_size 512 --buffer_type reservoir --reset_buffer_on_task_change False --replay_size 2e5
```

# Reproducing results

### Running experiments
The scripts for running all our experiments in the paper can be found [here](https://github.com/TTomilin/COOM/tree/main/experiments/scripts).

### Downloading results
We recommend using [Weights & Biases](https://wandb.ai/) to log your experiments. 
Having done so, [download,py](https://github.com/TTomilin/COOM/tree/main/experiments/results/download.py) can be used to download results:  
`python download.py --project <PROJECT> --sequence <SEQUENCE> --metric <METRIC>`  
The relevant metrics used in the paper are: `success` and `kills`.

### Plotting results

Figures from the paper can be drawn using the [plotting scripts](https://github.com/TTomilin/COOM/tree/main/experiments/results).  
`python plot_results_envs.py --sequence <SEQUENCE> --metric <METRIC>`  
`python plot_results_methods.py --sequence <SEQUENCE> --metric <METRIC>`

### Calculating metrics
All the numeric results displayed in the paper can be calculated using [cl_metrics.py](https://github.com/TTomilin/COOM/tree/main/experiments/results/cl_metrics.py).  
`python cl_metrics.py --sequence <SEQUENCE> --metric <METRIC>`


# Acknowledgements

COOM heavily relies on [ViZDoom](https://github.com/mwydmuch/ViZDoom).  
The `run_and_gun` scenario and its environment modification were inspired by [LevDoom](https://github.com/TTomilin/LevDoom).  
The implementation of Discrete SAC used in our code comes from [Tianshou](https://github.com/thu-ml/tianshou).  
Our experiments were managed using [WandB](https://wandb.ai).  
