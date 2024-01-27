# Continual Learning Module

This module provides popular continual learning baseline implementations on top of the Soft-Actor-Critic (SAC) algorithm.
The implementation is based on Tensorflow.

## Installation
To install the continual learning module, run the following command:
```bash 
$ pip install COOM[cl]
```

## Running Experiments
You can run single task or continual learning experiments with `run_single.py` and `run_cl.py` scripts, respectively.
To see available script arguments, run with `--help` option, e.g. `python run_single.py --help`

### Single task
```
python run_single.py --scenario pitfall
```

### Continual learning
```
python run_cl.py --sequence CO4 --cl_method packnet
```

## Reproducing Experimental Results
We have also listed all the commands for running the experiments in our paper in 
[cl.sh](scripts/cl.sh) and [single.sh](scripts/single.sh).
We used seeds [0, 1, ..., 9] for all experiments in the paper.

### Average Performance, Forgetting, Forward Transfer and Action Distributions
We evaluate the continual learning methods on the COOM benchmark based on Average Performance, Forgetting, and Forward Transfer.
We use the following CL methods:
```
python CL/run_cl.py --sequence [SEQUENCE] --seed [SEED] --cl_method packnet --packnet_retrain_steps 10000 --clipnorm 2e-05
python CL/run_cl.py --sequence [SEQUENCE] --seed [SEED] --cl_method mas --cl_reg_coef=10000
python CL/run_cl.py --sequence [SEQUENCE] --seed [SEED] --cl_method agem --regularize_critic --episodic_mem_per_task 10000 --episodic_batch_size 128
python CL/run_cl.py --sequence [SEQUENCE] --seed [SEED] --cl_method l2 --cl_reg_coef=100000
python CL/run_cl.py --sequence [SEQUENCE] --seed [SEED] --cl_method ewc --cl_reg_coef=250
python CL/run_cl.py --sequence [SEQUENCE] --seed [SEED] --cl_method vcl --cl_reg_coef=1 --vcl_first_task_kl False
python CL/run_cl.py --sequence [SEQUENCE] --seed [SEED] --cl_method clonex --exploration_kind 'best_return' --cl_reg_coef=100 --episodic_mem_per_task 10000 --episodic_batch_size 128
python CL/run_cl.py --sequence [SEQUENCE] --seed [SEED] --batch_size 512 --buffer_type reservoir --reset_buffer_on_task_change False --replay_size 8e5  # Perfect Memory
python CL/run_cl.py --sequence [SEQUENCE] --seed [SEED]  # Fine-tuning
```

We ran the COC sequence with sparse reward and only with PackNet:
```
python CL/run_cl.py --sequence COC --seed [SEED] --sparse_rewards --cl_method packnet --packnet_retrain_steps 10000 --clipnorm 2e-05
```

Measuring Forward Transfer also requires running SAC on each task in isolation:
```
python CL/run_single.py --scenario [SCENARIO] --envs [ENVS] --seed [SEED] --no_test
```

### Network plasticity
To reproduce our network plasticity experiments from the paper, run the following command:
```
python CL/run_continual.py --sequence CO8 --seed [SEED] --repeat_sequence 10 --no_test --steps_per_env 100000
```

### Method Variations
To reproduce our method variations experiments from the paper, run the following command:
#### Image Augmentations
1. Random Convolution
2. Random Shift
3. Random Noise
```
python CL/run_continual.py --sequence CO8 --cl_method [METHOD] --seed [SEED] --augment --augmentation conv
python CL/run_continual.py --sequence CO8 --cl_method [METHOD] --seed [SEED] --augment --augmentation shift
python CL/run_continual.py --sequence CO8 --cl_method [METHOD] --seed [SEED] --augment --augmentation noise
```
#### Prioritized Experience Replay (PER)
```
python CL/run_continual.py --sequence CO8 --cl_method [METHOD] --seed [SEED] --buffer_type prioritized
```
#### LSTM
```
python CL/run_continual.py --sequence CO8 --cl_method [METHOD] --seed [SEED] --use_lstm
```
#### Critic Regularization
```
python CL/run_continual.py --sequence CO8 --cl_method [METHOD] --seed [SEED] --regularize_critic
```

## Command Line Arguments

Below is a table of the available command line arguments for the script:

| Category               | Argument                           | Default                | Description                                                                                                                                                                 |
|------------------------|------------------------------------|------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Core**               | `--scenarios`                      | None                   | Scenarios to run. Choices: `health_gathering`, `run_and_gun`, `dodge_projectiles`, `chainsaw`, `raise_the_roof`, `floor_is_lava`, `hide_and_seek`, `arms_dealer`, `pitfall` |
|                        | `--envs`                           | `['default']`          | Name of the environments in the scenario(s) to run                                                                                                                          |
|                        | `--test_envs`                      | []                     | Name of the environments to periodically evaluate the agent on                                                                                                              |
|                        | `--no_test`                        | False                  | If True, no test environments will be used                                                                                                                                  |
|                        | `--seed`                           | 0                      | Seed for randomness                                                                                                                                                         |
|                        | `--gpu`                            | None                   | Which GPU to use                                                                                                                                                            |
|                        | `--sparse_rewards`                 | False                  | Whether to use the sparse reward setting                                                                                                                                    |
| **Continual Learning** | `--sequence`                       | None                   | Name of the continual learning sequence. Choices: `CD4`, `CD8`, `CD16`, `CO4`, `CO8`, `CO16`, `COC`, `MIXED`                                                                |
|                        | `--cl_method`                      | None                   | Continual learning method. Choices: `clonex`, `owl`, `l2`, `ewc`, `mas`, `vcl`, `packnet`, `agem`                                                                           |
|                        | `--start_from`                     | 0                      | Which task to start/continue the training from                                                                                                                              |
|                        | `--num_repeats`                    | 1                      | How many times to repeat the sequence                                                                                                                                       |
|                        | `--random_order`                   | False                  | Whether to randomize the order of the tasks                                                                                                                                 |
| **DOOM**               | `--render`                         | False                  | Render the environment                                                                                                                                                      |
|                        | `--render_sleep`                   | 0.0                    | Sleep time between frames when rendering                                                                                                                                    |
|                        | `--variable_queue_length`          | 5                      | Number of game variables to remember                                                                                                                                        |
|                        | `--frame_skip`                     | 4                      | Number of frames to skip                                                                                                                                                    |
|                        | `--resolution`                     | None                   | Screen resolution of the game. Choices: `800X600`, `640X480`, `320X240`, `160X120`                                                                                          |
| **Save/Load**          | `--save_freq_epochs`               | 25                     | Save the model parameters after n epochs                                                                                                                                    |
|                        | `--model_path`                     | None                   | Path to load the model from                                                                                                                                                 |
| **Recording**          | `--record`                         | False                  | Whether to record gameplay videos                                                                                                                                           |
|                        | `--record_every`                   | 100                    | Record gameplay video every n episodes                                                                                                                                      |
|                        | `--video_folder`                   | 'videos'               | Path to save the gameplay videos                                                                                                                                            |
| **Logging**            | `--with_wandb`                     | False                  | Enables Weights and Biases                                                                                                                                                  |
|                        | `--logger_output`                  | ["tsv", "tensorboard"] | Types of logger used. Choices: `neptune`, `tensorboard`, `tsv`                                                                                                              |
|                        | `--group_id`                       | "default_group"        | Group ID, for grouping logs from different experiments into common directory                                                                                                |
|                        | `--log_every`                      | 1000                   | Number of steps between subsequent evaluations and logging                                                                                                                  |
| **Model**              | `--use_lstm`                       | False                  | Whether to use an LSTM after the CNN encoder head                                                                                                                           |
|                        | `--hidden_sizes`                   | [256, 256]             | Hidden sizes list for the MLP models                                                                                                                                        |
|                        | `--activation`                     | "lrelu"                | Activation kind for the models                                                                                                                                              |
|                        | `--use_layer_norm`                 | True                   | Whether to use layer normalization                                                                                                                                          |
|                        | `--multihead_archs`                | True                   | Whether to use multi-head architecture                                                                                                                                      |
|                        | `--hide_task_id`                   | False                  | Whether the model knows the task during test time                                                                                                                           |
| **Learning Rate**      | `--lr`                             | 1e-3                   | Learning rate for the optimizer                                                                                                                                             |
|                        | `--lr_decay`                       | 'linear'               | Method to decay the learning rate over time. Choices: None, 'linear', 'exponential'                                                                                         |
|                        | `--lr_decay_rate`                  | 0.1                    | Rate to decay the learning                                                                                                                                                  |
|                        | `--lr_decay_steps`                 | 1e5                    | Number of steps to decay the learning rate                                                                                                                                  |
| **Replay Buffer**      | `--replay_size`                    | 5e4                    | Size of the replay buffer                                                                                                                                                   |
|                        | `--buffer_type`                    | "fifo"                 | Strategy of inserting examples into the buffer. Choices: fifo, other values as per BufferType enum                                                                          |
|                        | `--episodic_memory_from_buffer`    | True                   | [Description]                                                                                                                                                               |
| **Training**           | `--steps_per_env`                  | 2e5                    | Number of steps the algorithm will run per environment                                                                                                                      |
|                        | `--update_after`                   | 5000                   | Number of env interactions to collect before starting to do update the gradient                                                                                             |
|                        | `--update_every`                   | 500                    | Number of env interactions to do between every update                                                                                                                       |
|                        | `--n_updates`                      | 50                     | Number of consecutive policy gradient descent updates to perform                                                                                                            |
|                        | `--batch_size`                     | 128                    | Minibatch size for the optimization                                                                                                                                         |
|                        | `--gamma`                          | 0.99                   | Discount factor                                                                                                                                                             |
|                        | `--alpha`                          | "auto"                 | Entropy regularization coefficient                                                                                                                                          |
|                        | `--target_output_std`              | 0.089                  | Target standard deviation of the action distribution for dynamic alpha tuning                                                                                               |
|                        | `--regularize_critic`              | False                  | Whether to regularize both actor and critic, or only actor                                                                                                                  |
|                        | `--clipnorm`                       | None                   | Value for gradient clipping                                                                                                                                                 |
| **Testing**            | `--test`                           | True                   | Whether to test the model                                                                                                                                                   |
|                        | `--test_only`                      | False                  | Whether to only test the model                                                                                                                                              |
|                        | `--test_episodes`                  | 3                      | Number of episodes to test the model                                                                                                                                        |
| **Exploration**        | `--start_steps`                    | 10000                  | Number of steps for uniform-random action selection                                                                                                                         |
|                        | `--agent_policy_exploration`       | False                  | Whether to use uniform exploration only in the first task                                                                                                                   |
|                        | `--exploration_kind`               | None                   | Kind of exploration to use at the beginning of a new task                                                                                                                   |
| **Task Change**        | `--reset_buffer_on_task_change`    | True                   | Whether to reset the replay buffer on task change                                                                                                                           |
|                        | `--reset_optimizer_on_task_change` | True                   | Whether to reset the optimizer on task change                                                                                                                               |
|                        | `--reset_critic_on_task_change`    | False                  | Whether to reset the critic on task change                                                                                                                                  |
| **CL Method Specific** | `--packnet_retrain_steps`          | 0                      | Number of retrain steps after network pruning per task                                                                                                                      |
|                        | `--cl_reg_coef`                    | 0.0                    | Regularization strength for certain CL methods                                                                                                                              |
|                        | `--vcl_first_task_kl`              | False                  | Use KL regularization for the first task in VCL                                                                                                                             |
|                        | `--episodic_mem_per_task`          | 0                      | Number of examples to keep in memory per task for AGEM                                                                                                                      |
|                        | `--episodic_batch_size`            | 0                      | Minibatch size for additional loss computation in AGEM                                                                                                                      |
| **Observation**        | `--frame_stack`                    | 4                      | Number of frames to stack                                                                                                                                                   |
|                        | `--frame_height`                   | 84                     | Height of the frame                                                                                                                                                         |
|                        | `--frame_width`                    | 84                     | Width of the frame                                                                                                                                                          |
|                        | `--augment`                        | False                  | Whether to use image augmentation                                                                                                                                           |
|                        | `--augmentation`                   | None                   | Type of image augmentation. Choices: 'conv', 'shift', 'noise'                                                                                                               |
| **Reward**             | `--reward_frame_survived`          | 0.01                   | Reward for surviving a frame                                                                                                                                                |
|                        | `--reward_switch_pressed`          | 15.0                   | Reward for pressing a switch                                                                                                                                                |
|                        | `--reward_kill_dtc`                | 1.0                    | Reward for eliminating an enemy                                                                                                                                             |
|                        | `--reward_kill_rag`                | 5.0                    | Reward for eliminating an enemy                                                                                                                                             |
|                        | `--reward_kill_chain`              | 5.0                    | Reward for eliminating an enemy                                                                                                                                             |
|                        | `--reward_health_hg`               | 15.0                   | Reward for picking up a health kit                                                                                                                                          |
|                        | `--reward_health_has`              | 5.0                    | Reward for picking a health kit                                                                                                                                             |
|                        | `--reward_weapon_ad`               | 15.0                   | Reward for picking a weapon                                                                                                                                                 |
|                        | `--reward_delivery`                | 30.0                   | Reward for delivering an item                                                                                                                                               |
|                        | `--reward_platform_reached`        | 1.0                    | Reward for reaching a platform                                                                                                                                              |
|                        | `--reward_on_platform`             | 0.1                    | Reward for staying on a platform                                                                                                                                            |
|                        | `--reward_scaler_pitfall`          | 0.1                    | Reward scaler for traversal                                                                                                                                                 |
|                        | `--reward_scaler_traversal`        | 1e-3                   | Reward scaler for traversal                                                                                                                                                 |
| **Penalty**            | `--penalty_passivity`              | -0.1                   | Penalty for not moving                                                                                                                                                      |
|                        | `--penalty_death`                  | -1.0                   | Negative reward for dying                                                                                                                                                   |
|                        | `--penalty_projectile`             | -0.01                  | Negative reward for projectile hit                                                                                                                                          |
|                        | `--penalty_health_hg`              | -0.01                  | Negative reward for losing health                                                                                                                                           |
|                        | `--penalty_health_dtc`             | -1.0                   | Negative reward for losing health                                                                                                                                           |
|                        | `--penalty_health_has`             | -5.0                   | Negative reward for losing health                                                                                                                                           |
|                        | `--penalty_lava`                   | -0.1                   | Penalty for stepping on lava                                                                                                                                                |
|                        | `--penalty_ammo_used`              | -0.1                   | Negative reward for using ammo                                                                                                                                              |

