# COOM

COOM is a Continual Learning benchmark for embodied pixel-based RL, consisting of task sequences in visually 
distinct 3D environments with diverse objectives and egocentric perception. COOM is designed for task-incremental learning,
in which task boundaries are clearly defined. A short presentation of COOM can be found on 
[NeurIPS](https://neurips.cc/virtual/2023/poster/73450)
and a demo is available on [Youtube](https://www.youtube.com/watch?v=FUm2B8MZ6d0&list=PL6nJZHA3y2fxQK73jmuI5teM3n6Mydcf7).

<p align="center">
  <img src="assets/gifs/demo1.gif" alt="Demo1" style="vertical-align: top;"/>
  <img src="assets/gifs/demo2.gif" alt="Demo2" style="vertical-align: top;"/>
</p>

## Installation
To install COOM from PyPi, just run:
```bash
$ pip install COOM
```

Alternatively, to install COOM from source:
1. Clone the repository
```bash
$ git clone https://github.com/hyintell/COOM
```
2. Navigate into the repository
```bash
$ cd COOM
```
3. Install COOM from source with pip
```bash 
$ pip install .
```

## Environments
COOM contains 8 scenarios:

| Scenario         | Success Metric    | Enemies | Weapon  | Items   | Max Steps | Execute Action | Stochasticity                              | Image                                                  | 
|------------------|-------------------|---------|---------|---------|-----------|----------------|--------------------------------------------|--------------------------------------------------------|  
| Pitfall          | Distance Covered  | &cross; | &cross; | &cross; | 1000      | JUMP           | Pitfall tile locations                     | ![Default](assets/images/CO_envs/pitfall.png)          |
| Arms Dealer      | Weapons Delivered | &cross; | &check; | &check; | 1000      | SPEED          | Weapon spawn locations, delivery locations | ![Default](assets/images/CO_envs/arms_dealer.png)      |
| Hide and Seek    | Frames Alive      | &check; | &cross; | &check; | 2500      | SPEED          | Enemy behaviour, item spawn locations      | ![Default](assets/images/CO_envs/hide_and_seek.png)    |
| Floor is Lava    | Frames Alive      | &cross; | &cross; | &cross; | 2500      | SPEED          | Platform locations                         | ![Default](assets/images/CO_envs/floor_is_lava.png)    |
| Chainsaw         | Kill Count        | &check; | &check; | &cross; | 2500      | ATTACK         | Enemy and agent spawn locations            | ![Default](assets/images/CO_envs/chainsaw.png)         |
| Raise the Roof   | Frames Alive      | &cross; | &cross; | &cross; | 2500      | USE            | Agent spawn location                       | ![Default](assets/images/CO_envs/raise_the_roof.png)   |
| Run and Gun      | Kill Count        | &check; | &check; | &cross; | 2500      | ATTACK         | Enemy and agent spawn locations            | ![Default](assets/images/CO_envs/run_and_gun.png)      |
| Health Gathering | Frames Alive      | &cross; | &cross; | &check; | 2500      | SPEED          | Health kit spawn locations                 | ![Default](assets/images/CO_envs/health_gathering.png) |

Every scenario except `Run and Gun` has 2 environments: `default` and `hard`. The full list of environment is the following:
- `pitfall-default-v0` - traverse a tunnel as far as possible without falling into a pit
- `pitfall-hard-v0` - the agent has reduced movement speed and there are more pits in the surface
- `arms_dealer-default-v0` - collect and deliver weapons to the marked locations as fast as possible
- `arms_dealer-hard-v0` - the map is larger and there are fewer weapons to collect
- `hide_and_seek-default-v0` - escape and hide from enemies as long as possible
- `hide_and_seek-hard-v0` - there are more enemies and they are faster
- `floor_is_lava-default-v0` - keep off the laval by navigating to new platforms as they reappear at new locations
- `floor_is_lava-hard-v0` - there are fewer platforms to stand on and their locations are in more rapid change
- `chainsaw-default-v0` - seek out and melee as many enemies as possible
- `chainsaw-hard-v0` - there are fewer enemies to find, and they are tougher to eliminate
- `raise_the_roof-default-v0` - locate and press switches on the walls to raise the ceiling and avoid getting crushed
- `raise_the_roof-hard-v0` - the ceiling is lowered faster and the switches are harder to spot
- `run_and_gun-default-v0` - use a pistol to eliminate as many enemies as possible
- `run_and_gun-hard-v0` - there are fewer enemies to find, and they are tougher to eliminate
- `run_and_gun-obstacles-v0` - there are obstacles in the environment which block the agent's movement
- `run_and_gun-green-v0` - the wall, ceiling and floor textures are green
- `run_and_gun-resized-v0` - the agent's view height is randomized and the targets are randomly resized
- `run_and_gun-monsters-v0` - the targets are replaced with monsters that move around and fight back
- `run_and_gun-red-v0` - the wall, ceiling and floor textures are red
- `run_and_gun-blue-v0` - the wall, ceiling and floor textures are blue
- `run_and_gun-shadows-v0` - the targets are less visible due to lowered opacity
- `health_gathering-default-v0` - collect health kits as fast as possible to stay alive
- `health_gathering-hard-v0` - there are fewer health kits to find and the environment layout is more complex

## Task Sequences for Continual Learning
To formulate a continual learning problem, we compose sequences of tasks, where each task is an environment of a 
scenario. The agent is trained on each task sequentially, without access to the previous tasks. The agent is continually
evaluated on all tasks throughout training. The task sequence is considered solved if the agent achieves maximum success 
on all tasks.
There are three lengths of Continual Learning task sequences in our benchmark: 
1) 8-task sequences serve as the core of the benchmark
2) 4-task sequences are comprised of the 2<sup>nd</sup> half of an 8-task sequence
3) 16-task sequences combine tasks of two 8-task sequences

We further distinguish between the `Cross-Domain` and `Cross-Objective` sequences. 

### Cross-Domain
In the cross-domain setting, the agent is sequentially trained on modified versions of the same scenario. 
`Run and Gun` is selected as basis for this CL sequence, since out of the 8 scenarios in the benchmark, it best resembles 
the actual Doom game, requiring the agent to navigate the map and eliminate enemies by firing a weapon. The objective and
the layout of the map remain the same across tasks, whereas we modify the environment in the following ways: 
1) Changing the textures of the surrounding walls, ceiling and floor 
2) Varying the size, shape and type of enemies 
3) Randomizing the view height of the agent, and 
4) Adding objects to the environment which act as obstacles, blocking the agent’s movement.

#### Tasks in the Cross-Domain 8 (CD8) sequence
![Default](assets/images/sequences/CD8_sequence.png)
### Cross-Objective
Cross-objective task sequences employ a different scenario with a novel objective for each consecutive task, apart from 
only changing the visuals and dynamics of a single scenario. This presents a diverse challenge, as the goal might 
drastically change from locating and eliminating enemies (`Run and Gun` and `Chainsaw`) to running away and hiding 
from them (`Hide and Seek`). In a similar fashion, the scenario `Floor is Lava` often requires the agent to remain at a 
bounded location for optimal performance, whereas scenarios `Pitfall`, `Arms Dealer`, `Raise the Roof`, and `Health 
Gathering` endorse constant movement.
#### Tasks in the Cross-Objective 8 (CO8) sequence
![Default](assets/images/sequences/CO8_sequence.png)

# Getting Started
Below we provide a short code snippet to run a sequence with the COOM benchmark.

## Basic Usage
Find examples of using COOM environments in the 
[run_single](examples/run_single.py) and [run_sequence](examples/run_sequence.py) scripts.

### Single Environment
```
from COOM.env.builder import make_env
from COOM.utils.config import Scenario

env = make_env(Scenario.RAISE_THE_ROOF)
env.reset()
for steps in range(1000):
    action = env.action_space.sample()
    state, reward, done, truncated, info = env.step(action)
    env.render()
    if done:
        break
env.close()
```

### Task Sequence
```
from COOM.env.continual import ContinualLearningEnv
from COOM.utils.config import Sequence

cl_env = ContinualLearningEnv(Sequence.CO8)
for env in cl_env.tasks:
    env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        state, reward, done, truncated, info = env.step(action)
        env.render()
        if done:
            break
    env.close()
```

# Baseline Results
We have employed various popular continual learning algorithms to evaluate their performance on the COOM benchmark.
The algorithms are implemented on top of the Soft-Actor-Critic (SAC) reinforcement learning algorithm.
Please follow the instructions in the [Continual Learning module](CL/README.md) to use the algorithms.
The following table ranks the baselines from best to worst performing

| Method                                                                                                               | Type           | Score |                                                                                                        
|----------------------------------------------------------------------------------------------------------------------|----------------|-------|
| [PackNet](https://openaccess.thecvf.com/content_cvpr_2018/papers/Mallya_PackNet_Adding_Multiple_CVPR_2018_paper.pdf) | Structure      | 0.74  |
| [ClonEx-SAC](https://arxiv.org/pdf/2209.13900.pdf)                                                                   | Memory         | 0.73  |
| L2                                                                                                                   | Regularization | 0.64  |
| [MAS](https://arxiv.org/pdf/1711.09601.pdf)                                                                          | Regularization | 0.56  |
| [EWC](https://www.pnas.org/doi/epdf/10.1073/pnas.1611835114)                                                         | Regularization | 0.54  |
| Fine-Tuning                                                                                                          | Naïve          | 0.40  |
| [VCL](https://arxiv.org/pdf/1710.10628.pdf)                                                                          | Regularization | 0.33  |                                                          
| [AGEM](https://arxiv.org/pdf/1812.00420.pdf)                                                                         | Memory         | 0.28  |
| [Perfect Memory*](https://arxiv.org/abs/2105.10919)                                                                  | Memory         | 0.89* |

_*The memory consumption of the method is too high to feasible run it on the longer sequences of the benchmark, so it does not follow the ranking in the table._ 

## Evaluation Metrics
We evaluate the continual learning methods on the COOM benchmark based on Average Performance, Forgetting, and Forward Transfer.
### Average Performance
The performance (success rate) averaged over tasks is a typical metric for the continual learning setting. The agent is continually
evaluated on all tasks in the sequence even before encountering it. By the end of the sequence, the agent should have mastered all tasks.

![Default](assets/images/plots/performance.png)

### Forgetting
Forgetting occurs when the agent's performance on a task decreases after training on a subsequent task.
This is a common problem in continual learning, as the agent has to learn new tasks while retaining the knowledge of the previous ones.
We measure forgetting by comparing the performance of the agent on a task after training and at the end of the entire sequence.
The image below depicts heavy forgetting in the example of AGEM. 
![Default](assets/images/plots/forgetting_AGEM.png)

Contrary to AGEM, ClonEx-SAC is able to retain the knowledge of the previous tasks.
![Default](assets/images/plots/forgetting_clonex.png)

### Forward Transfer
Transferring learned knowledge from one task to another is a key aspect of continual learning. We measure the forward 
transfer of the continual learning methods by how efficiently they train on each given task compared to the 
Soft Actor-Critic (SAC) baseline, which is trained directly on the same from scratch. The red areas between the curves 
represent negative forward transfer and other colors represent positive forward transfer as depicted on the image below.

![Default](assets/images/plots/transfer_clonex.png)

## Reproducing results
For reproducing the results in our paper please follow the instructions in the [results module](results/README.md).

# Acknowledgements

COOM is based on the [ViZDoom](https://github.com/mwydmuch/ViZDoom) platform.  
The `Cross-Domain` task sequences and the `run_and_gun` scenario environment modification were inspired by the [LevDoom](https://github.com/TTomilin/LevDoom) generalization benchmark.  
The base implementations of SAC and continual learning methods originate from [Continual World](https://github.com/awarelab/continual_world).  
Our experiments were managed using [WandB](https://wandb.ai).

# Citation
If you use our work in your research, please cite it as follows:
```
@inproceedings{tomilin2023coom,
    title={COOM: A Game Benchmark for Continual Reinforcement Learning},
    author={Tomilin, Tristan and Fang, Meng and Zhang, Yudi and Pechenizkiy, Mykola},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
    year={2023}
}
```