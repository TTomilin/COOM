import argparse

from cl.sac.replay_buffers import BufferType
from cl.utils.run_utils import str2bool, sci2int, float_or_str


def parse_args():
    def arg(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    parser = argparse.ArgumentParser(description="Continual Doom")

    # Core
    arg('--scenarios', type=str, nargs="+", default=None,
        choices=['defend_the_center', 'health_gathering', 'run_and_gun', 'dodge_projectiles', 'chainsaw',
                 'raise_the_roof', 'floor_is_lava', 'hide_and_seek', 'arms_dealer', 'parkour', 'pitfall'])
    arg("--cl_method", type=str, choices=[None, "clonex", "owl", "l2", "ewc", "mas", "vcl", "packnet", "agem"],
        default=None, help="If None, the fine-tuning method will be used")
    arg("--envs", type=str, nargs="+", default=['default'], help="Name of the environments in the scenario(s) to run")
    arg("--test_envs", type=str, nargs="+", default=[],
        help="Name of the environments to periodically evaluate the agent on")
    arg("--no_test", default=False, action='store_true', help="If True, no test environments will be used")
    arg("--sequence", type=str, default=None, choices=['CD4', 'CD8', 'CD16', 'CO4', 'CO8', 'CO16', 'COC'],
        help="Name of the continual learning sequence")
    arg('--seed', type=int, default=0, help='Seed for randomness')
    arg('--gpu', '-g', default=None, type=int, help='Which GPU to use')
    arg("--sparse_rewards", default=False, action='store_true', help="Whether to use the sparse reward setting")
    arg('--start_from', type=int, default=0, help='Which task to start/continue the training from')
    arg('--repeat_sequence', type=int, default=1, help='How many times to repeat the sequence')

    # Save/Load
    arg("--save_freq_epochs", type=int, default=25, help="Save the model parameters after n epochs")
    arg("--model_path", type=str, default=None, help="Path to load the model from")

    # Recording
    arg("--record", type=str2bool, default=False, help="Whether to record gameplay videos")
    arg("--record_every", type=int, default=100, help="Record gameplay video every n episodes")
    arg("--video_folder", type=str, default='videos', help="Path to save the gameplay videos")

    # Logging
    arg('--with_wandb', default=False, action='store_true', help='Enables Weights and Biases')
    arg("--logger_output", type=str, nargs="+", choices=["neptune", "tensorboard", "tsv"],
        default=["tsv", "tensorboard"], help="Types of logger used.")
    arg("--group_id", type=str, default="default_group",
        help="Group ID, for grouping logs from different experiments into common directory")
    arg("--log_every", type=sci2int, default=int(1000),
        help="Number of steps between subsequent evaluations and logging")

    # Model
    arg("--use_lstm", default=False, action='store_true', help="Whether to use an LSTM after the CNN encoder head")
    arg("--hidden_sizes", type=int, nargs="+", default=[256, 256], help="Hidden sizes list for the MLP models")
    arg("--activation", type=str, default="lrelu", help="Activation kind for the models")
    arg("--use_layer_norm", type=str2bool, default=True, help="Whether to use layer normalization")
    arg("--multihead_archs", type=str2bool, default=True, help="Whether to use multi-head architecture")
    arg("--hide_task_id", action='store_true', default=False, help="Whether the model knows the task during test time")

    # Learning rate
    arg("--lr", type=float, default=1e-3, help="Learning rate for the optimizer")
    arg('--lr_decay', type=str, default='linear', choices=[None, 'linear', 'exponential'],
        help='Method to decay the learning rate over time')
    arg('--lr_decay_rate', type=float, default=0.1, help='Rate to decay the learning')
    arg('--lr_decay_steps', type=sci2int, default=int(1e5), help='Number of steps to decay the learning rate')

    # Replay buffer
    arg("--replay_size", type=sci2int, default=int(5e4), help="Size of the replay buffer")
    arg("--buffer_type", type=str, default="fifo", choices=[b.value for b in BufferType],
        help="Strategy of inserting examples into the buffer")
    arg("--episodic_memory_from_buffer", type=str2bool, default=True)

    # DOOM
    arg('--render', default=False, action='store_true', help='Render the environment')
    arg('--render_mode', type=str, default='rgb_array', help='Mode of rendering')
    arg('--render_sleep', type=float, default=0.0, help='Sleep time between frames when rendering')
    arg('--variable_queue_length', type=int, default=5, help='Number of game variables to remember')
    arg('--frame_skip', type=int, default=4, help='Number of frames to skip')
    arg('--resolution', type=str, default='160X120', choices=['800X600', '640X480', '320X240', '160X120'],
        help='Screen resolution of the game')

    # Training
    arg("--steps_per_env", type=sci2int, default=int(2e5),
        help="Number of steps the algorithm will run per environment")
    arg("--update_after", type=sci2int, default=int(5000),
        help="Number of env interactions to collect before starting to do update the gradient")
    arg("--update_every", type=sci2int, default=int(500), help="Number of env interactions to do between every update")
    arg("--n_updates", type=sci2int, default=int(50),
        help="Number of consecutive policy gradient descent updates to perform")
    arg("--batch_size", type=int, default=128, help="Minibatch size for the optimization")
    arg("--gamma", type=float, default=0.99, help="Discount factor")
    arg("--alpha", type=float_or_str, default="auto",
        help="Entropy regularization coefficient. "
             "Can be either float value, or 'auto', in which case it is dynamically tuned.")
    arg("--target_output_std", type=float, default=0.089,
        help="If alpha is 'auto', alpha is dynamically tuned so that standard deviation "
             "of the action distribution on every dimension matches target_output_std.")
    arg("--regularize_critic", default=False, action='store_true',
        help="If True, both actor and critic are regularized; if False, only actor is")
    arg("--clipnorm", type=float, default=None, help="Value for gradient clipping")

    # Testing
    arg("--test", type=str2bool, default=True, help="Whether to test the model")
    arg("--test_only", default=False, action='store_true', help="Whether to only test the model")
    arg("--test_episodes", default=3, type=int, help="Number of episodes to test the model")

    # Exploration
    arg("--start_steps", type=sci2int, default=int(10000),
        help="Number of steps for uniform-random action selection, before running real policy. Helps exploration.")
    arg("--agent_policy_exploration", default=False, action='store_true',
        help="If True, uniform exploration for start_steps steps is used only in the "
             "first task (in continual learning). Otherwise, it is used in every task")
    arg("--exploration_kind", type=str, default=None,
        choices=[None, "previous", "uniform_previous", "uniform_previous_or_current", "best_return"],
        help="Kind of exploration to use at the beginning of a new task.", )

    # Task change
    arg("--reset_buffer_on_task_change", type=str2bool, default=True,
        help="Whether to reset the replay buffer on task change")
    arg("--reset_optimizer_on_task_change", type=str2bool, default=True,
        help="Whether to reset the optimizer on task change")
    arg("--reset_critic_on_task_change", type=str2bool, default=False,
        help="Whether to reset the critic on task change")

    # CL method specific
    arg("--packnet_retrain_steps", type=int, default=0,
        help="Number of retrain steps after network pruning, which occurs after each task")
    arg("--cl_reg_coef", type=float, default=0.0,
        help="Regularization strength for continual learning methods. Valid for 'l2', 'ewc', 'mas' continual learning methods.")
    arg("--vcl_first_task_kl", type=str2bool, default=False,
        help="If True, use KL regularization also for the first task in 'vcl' continual learning method.")
    arg("--episodic_mem_per_task", type=int, default=0,
        help="Number of examples to keep in additional memory per task. Valid for 'agem' continual learning method.")
    arg("--episodic_batch_size", type=int, default=0,
        help="Minibatch size to compute additional loss in 'agem' continual learning method.")

    # Observation
    arg('--frame_stack', type=int, default=4, help='Number of frames to stack')
    arg('--frame_height', type=int, default=84, help='Height of the frame')
    arg('--frame_width', type=int, default=84, help='Width of the frame')
    arg("--augment", default=False, action='store_true', help="Whether to use image augmentation")
    arg("--augmentation", type=str, default=None, choices=['conv', 'shift', 'noise'], help="Type of image augmentation")

    # Reward
    arg('--reward_frame_survived', default=0.01, type=float, help='For surviving a frame')
    arg('--reward_switch_pressed', default=15.0, type=float, help='For pressing a switch')
    arg('--reward_kill_dtc', default=1.0, type=float, help='For eliminating an enemy')
    arg('--reward_kill_rag', default=5.0, type=float, help='For eliminating an enemy')
    arg('--reward_kill_chain', default=5.0, type=float, help='For eliminating an enemy')
    arg('--reward_health_hg', default=15.0, type=float, help='For picking up a health kit')
    arg('--reward_health_has', default=5.0, type=float, help='For picking a health kit')
    arg('--reward_weapon_ad', default=15.0, type=float, help='For picking a weapon')
    arg('--reward_delivery', default=30.0, type=float, help='For delivering an item')
    arg('--reward_platform_reached', default=1.0, type=float, help='For reaching a platform')
    arg('--reward_on_platform', default=0.1, type=float, help='For staying on a platform')
    arg('--reward_scaler_pitfall', default=0.1, type=float, help='Reward scaler for traversal')
    arg('--reward_scaler_traversal', default=1e-3, type=float, help='Reward scaler for traversal')

    # Penalty
    arg('--penalty_passivity', default=-0.1, type=float, help='Penalty for not moving')
    arg('--penalty_death', default=-1.0, type=float, help='Negative reward for dying')
    arg('--penalty_projectile', default=-0.01, type=float, help='Negative reward for projectile hit')
    arg('--penalty_health_hg', default=-0.01, type=float, help='Negative reward for losing health')
    arg('--penalty_health_dtc', default=-1.0, type=float, help='Negative reward for losing health')
    arg('--penalty_health_has', default=-5.0, type=float, help='Negative reward for losing health')
    arg('--penalty_lava', default=-0.1, type=float, help='Penalty for stepping on lava')
    arg('--penalty_ammo_used', default=-0.1, type=float, help='Negative reward for using ammo')

    return parser
