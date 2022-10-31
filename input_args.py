import argparse

from coom.tasks import TASK_SEQS
from coom.utils.enums import BufferType
from coom.utils.utils import float_or_str, sci2int, str2bool


def cl_parse_args(args=None):
    parser = argparse.ArgumentParser(description="Continual World")

    parser.add_argument('--scenarios', type=str, nargs="+", default=None,
                        choices=['defend_the_center', 'health_gathering', 'seek_and_slay', 'dodge_projectiles',
                                 'chainsaw', 'raise_the_roof', 'floor_is_lava', 'hide_and_seek', 'arms_dealer',
                                 'parkour'])
    parser.add_argument("--cl_method", type=str, choices=[None, "l2", "ewc", "mas", "vcl", "packnet", "agem"],
                        default=None, help="If None, the fine-tuning method will be used")
    parser.add_argument("--envs", type=str, nargs="+", default=['default'],
                        help="Name of the environments in the scenario(s) to run")
    parser.add_argument("--sequence", type=str, default=None, choices=['cross-environment', 'cross-scenario'],
                        help="Type of the continual learning sequence")
    parser.add_argument("--seed", type=int, default=0, help="Seed for randomness")

    # Save/Load
    parser.add_argument("--save_freq_epochs", type=int, default=25, help="Save the model parameters after n epochs")
    parser.add_argument("--model_path", type=str, default=None, help="Path to load the model from")

    # Recording
    parser.add_argument("--record", type=str2bool, default=True, help="Whether to record gameplay videos")
    parser.add_argument("--record_every", type=int, default=100, help="Record gameplay video every n episodes")
    parser.add_argument("--video_folder", type=str, default='videos', help="Path to save the gameplay videos")

    # Logging
    parser.add_argument("--logger_output", type=str, nargs="+", choices=["neptune", "tensorboard", "tsv"],
                        default=["tsv", "tensorboard"], help="Types of logger used.")
    parser.add_argument("--group_id", type=str, default="default_group",
                        help="Group ID, for grouping logs from different experiments into common directory")
    parser.add_argument("--log_every", type=sci2int, default=int(1000),
                        help="Number of steps between subsequent evaluations and logging")

    # Model
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[256, 256],
                        help="Hidden sizes list for the MLP models")
    parser.add_argument("--activation", type=str, default="lrelu", help="Activation kind for the models")
    parser.add_argument("--use_layer_norm", type=str2bool, default=True, help="Whether or not use layer normalization")
    parser.add_argument("--multihead_archs", type=str2bool, default=True, help="Whether use multi-head architecture")
    parser.add_argument("--hide_task_id", type=str2bool, default=True,
                        help="if True, one-hot encoding of the task will not be appended to observation")

    # Learning rate
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument('--lr_decay', type=str, default='linear', choices=[None, 'linear', 'exponential'],
                        help='Method to decay the learning rate over time')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='Rate to decay the learning')
    parser.add_argument('--lr_decay_steps', type=sci2int, default=int(1e5),
                        help='Number of steps to decay the learning rate')

    # Replay buffer
    parser.add_argument("--replay_size", type=sci2int, default=int(3e5), help="Size of the replay buffer")
    parser.add_argument("--buffer_type", type=str, default="fifo", choices=[b.value for b in BufferType],
                        help="Strategy of inserting examples into the buffer")

    # Training
    parser.add_argument("--steps_per_env", type=sci2int, default=int(2e5),
                        help="Number of steps the algorithm will run per environment")
    parser.add_argument("--start_steps", type=sci2int, default=int(10000),
                        help="Number of steps for uniform-random action selection, before running real policy. Helps exploration.")
    parser.add_argument("--update_after", type=sci2int, default=int(5000),
                        help="Number of env interactions to collect before starting to do update the gradient")
    parser.add_argument("--update_every", type=sci2int, default=int(500),
                        help="Number of env interactions to do between every update")
    parser.add_argument("--n_updates", type=sci2int, default=int(50),
                        help="Number of consecutive policy gradient descent updates to perform")
    parser.add_argument("--batch_size", type=int, default=128, help="Minibatch size for the optimization")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--alpha", type=float_or_str, default="auto",
                        help="Entropy regularization coefficient. Can be either float value, or 'auto', in which case it is dynamically tuned.")
    parser.add_argument("--target_output_std", type=float, default=0.089,
                        help="If alpha is 'auto', alpha is dynamically tuned so that standard deviation of the action distribution on every dimension matches target_output_std.")
    parser.add_argument("--regularize_critic", type=str2bool, default=False,
                        help="If True, both actor and critic are regularized; if False, only actor is")
    parser.add_argument("--clipnorm", type=float, default=None, help="Value for gradient clipping")
    parser.add_argument("--agent_policy_exploration", type=str2bool, default=False,
                        help="If True, uniform exploration for start_steps steps is used only in the first task (in continual learning). Otherwise, it is used in every task")

    # Testing
    parser.add_argument("--test", type=str2bool, default=True, help="Whether to test the model")
    parser.add_argument("--test_deterministic", default=False, action='store_true',
                        help="Whether to also evaluate deterministic actions at test time")

    # Task change
    parser.add_argument("--reset_buffer_on_task_change", type=str2bool, default=True,
                        help="If true, replay buffer is reset on each task change")
    parser.add_argument("--reset_optimizer_on_task_change", type=str2bool, default=True,
                        help="If true, optimizer is reset on each task change")
    parser.add_argument("--reset_critic_on_task_change", type=str2bool, default=False,
                        help="If true, critic model is reset on each task change")

    # CL method specific
    parser.add_argument("--packnet_retrain_steps", type=int, default=0,
                        help="Number of retrain steps after network pruning, which occurs after each task")
    parser.add_argument("--cl_reg_coef", type=float, default=0.0,
                        help="Regularization strength for continual learning methods. Valid for 'l2', 'ewc', 'mas' continual learning methods.")
    parser.add_argument("--vcl_first_task_kl", type=str2bool, default=False,
                        help="If True, use KL regularization also for the first task in 'vcl' continual learning method.")
    parser.add_argument("--episodic_mem_per_task", type=int, default=0,
                        help="Number of examples to keep in additional memory per task. Valid for 'agem' continual learning method.")
    parser.add_argument("--episodic_batch_size", type=int, default=0,
                        help="Minibatch size to compute additional loss in 'agem' continual learning method.")

    # DOOM
    parser.add_argument('--render_sleep', type=float, default=0.03, help='Sleep time between frames when rendering')
    parser.add_argument('--render', default=False, action='store_true', help='Render the environment')
    parser.add_argument('--variable_queue_len', type=int, default=5, help='Number of game variables to remember')
    parser.add_argument('--normalize', type=str2bool, default=True, help='Normalize the game state')
    parser.add_argument('--frame_height', type=int, default=84, help='Height of the frame')
    parser.add_argument('--frame_width', type=int, default=84, help='Width of the frame')
    parser.add_argument('--frame_stack', type=int, default=4, help='Number of frames to stack')
    parser.add_argument('--frame_skip', type=int, default=4, help='Number of frames to skip')
    parser.add_argument('--acceleration', default=False, action='store_true', help='Grant the acceleration action')

    # WandB
    parser.add_argument('--with_wandb', default=False, action='store_true', help='Enables Weights and Biases')
    parser.add_argument('--wandb_entity', default=None, type=str, help='WandB username (entity).')
    parser.add_argument('--wandb_project', default='COOM', type=str, help='WandB "Project"')
    parser.add_argument('--wandb_group', default=None, type=str, help='WandB "Group". Name of the env by default.')
    parser.add_argument('--wandb_job_type', default='train', type=str, help='WandB job type')
    parser.add_argument('--wandb_tags', default=[], type=str, nargs='*', help='Tags can help finding experiments')
    parser.add_argument('--wandb_key', default=None, type=str, help='API key for authorizing WandB')
    parser.add_argument('--wandb_dir', default=None, type=str, help='the place to save WandB files')
    parser.add_argument('--wandb_experiment', default='', type=str, help='Identifier to specify the experiment')

    # Reward
    parser.add_argument('--reward_switch_pressed', default=15.0, type=float, help='For pressing a switch')
    parser.add_argument('--reward_frame_survived', default=0.01, type=float, help='For surviving a frame')
    parser.add_argument('--reward_kill', default=5.0, type=float, help='For eliminating an enemy')
    parser.add_argument('--reward_item_acquired', default=15.0, type=float, help='For picking up weapons/health kits')
    parser.add_argument('--reward_delivery', default=30.0, type=float, help='For delivering an item')
    parser.add_argument('--reward_scaler_height', default=1.0, type=float, help='Reward scaler for height')
    parser.add_argument('--reward_scaler_traversal', default=1e-3, type=float, help='Reward scaler for traversal')

    # Penalty
    parser.add_argument('--penalty_health_loss', default=0.01, type=float, help='Negative reward for losing health')
    parser.add_argument('--penalty_ammo_used', default=0.1, type=float, help='Negative reward for using ammo')
    parser.add_argument('--penalty_frame_passed', default=0.01, type=float, help='Negative reward for wasting time')

    return parser.parse_args(args=args)


def mt_parse_args(args=None):
    parser = argparse.ArgumentParser(description="Continual World")
    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument(
        "--tasks",
        type=str,
        choices=TASK_SEQS.keys(),
        default=None,
        help="Name of the sequence you want to run",
    )
    task_group.add_argument(
        "--task_list",
        nargs="+",
        default=None,
        help="List of tasks you want to run, by name or by the MetaWorld index",
    )
    parser.add_argument(
        "--logger_output",
        type=str,
        nargs="+",
        choices=["neptune", "tensorboard", "tsv"],
        default=["tsv"],
        help="Types of logger used.",
    )
    parser.add_argument(
        "--group_id",
        type=str,
        default="default_group",
        help="Group ID, for grouping logs from different experiments into common directory",
    )
    parser.add_argument("--seed", type=int, help="Seed for randomness")
    parser.add_argument(
        "--log_every",
        type=sci2int,
        default=int(2e4),
        help="Number of steps between subsequent evaluations and logging",
    )
    parser.add_argument(
        "--replay_size", type=sci2int, default=int(1e4), help="Size of the replay buffer"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Minibatch size for the optimization"
    )
    parser.add_argument(
        "--hidden_sizes",
        type=int,
        nargs="+",
        default=[256, 256],
        help="Hidden sizes list for the MLP models",
    )
    parser.add_argument(
        "--activation", type=str, default="lrelu", help="Activation kind for the models"
    )
    parser.add_argument(
        "--use_layer_norm",
        type=str2bool,
        default=True,
        help="Whether or not use layer normalization",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--alpha",
        default="auto",
        help="Entropy regularization coefficient. "
             "Can be either float value, or 'auto', in which case it is dynamically tuned.",
    )
    parser.add_argument(
        "--target_output_std",
        type=float,
        default=0.089,
        help="If alpha is 'auto', alpha is dynamically tuned so that standard deviation "
             "of the action distribution on every dimension matches target_output_std.",
    )
    parser.add_argument(
        "--use_popart", type=str2bool, default=True, help="Whether use PopArt normalization"
    )
    parser.add_argument(
        "--popart_beta",
        type=float,
        default=3e-4,
        help="Beta parameter for updating statistics in PopArt",
    )
    parser.add_argument(
        "--multihead_archs", type=str2bool, default=True, help="Whether use multi-head architecture"
    )
    parser.add_argument(
        "--hide_task_id",
        type=str2bool,
        default=True,
        help="if True, one-hot encoding of the task will not be appended to observation",
    )
    return parser.parse_args(args=args)


def single_parse_args(args=None):
    parser = argparse.ArgumentParser(description="Run single task")
    parser.add_argument('--scenario', type=str, default=None,
                        choices=['defend_the_center', 'health_gathering', 'seek_and_slay', 'dodge_projectiles',
                                 'chainsaw', 'raise_the_roof', 'floor_is_lava', 'hide_and_seek', 'arms_dealer',
                                 'parkour'])
    parser.add_argument("--task", type=str, help="Name of the task")
    parser.add_argument('--tasks', type=str, nargs='*', default=['default'])
    parser.add_argument('--test_tasks', type=str, nargs='*', default=[])
    parser.add_argument("--seed", type=int, default=0, help="Seed for randomness")

    # Save/Load
    parser.add_argument("--save_freq_epochs", type=int, default=25, help="Save the model parameters after n epochs")
    parser.add_argument("--model_path", type=str, default=None, help="Path to load the model from")

    # Recording
    parser.add_argument("--record", type=str2bool, default=True, help="Whether to record gameplay videos")
    parser.add_argument("--record_every", type=int, default=100, help="Record gameplay video every n steps")
    parser.add_argument("--video_folder", type=str, default='videos', help="Path to save the gameplay videos")

    # Logging
    parser.add_argument("--logger_output", type=str, nargs="+", choices=["neptune", "tensorboard", "tsv"],
                        default=["tsv", "tensorboard"], help="Types of logger used.")
    parser.add_argument("--group_id", type=str, default="default_group",
                        help="Group ID, for grouping logs from different experiments into common directory")
    parser.add_argument("--log_every", type=sci2int, default=int(1000),
                        help="Number of steps between subsequent evaluations and logging")

    # Model
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[256, 256],
                        help="Hidden sizes list for the MLP models")
    parser.add_argument("--activation", type=str, default="lrelu", help="Activation kind for the models")
    parser.add_argument("--use_layer_norm", type=str2bool, default=True, help="Whether or not use layer normalization")

    # Learning rate
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument('--lr_decay', type=str, default='linear', choices=[None, 'linear', 'exponential'],
                        help='Method to decay the learning rate over time')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='Rate to decay the learning')
    parser.add_argument('--lr_decay_steps', type=sci2int, default=int(1e5),
                        help='Number of steps to decay the learning rate')

    # Training
    parser.add_argument("--steps_per_env", type=sci2int, default=int(2e5),
                        help="Number of steps the algorithm will run per environment")
    parser.add_argument("--start_steps", type=sci2int, default=int(10000),
                        help="Number of steps for uniform-random action selection, before running real policy. Helps exploration.")
    parser.add_argument("--update_after", type=sci2int, default=int(5000),
                        help="Number of env interactions to collect before starting to do update the gradient")
    parser.add_argument("--update_every", type=sci2int, default=int(500),
                        help="Number of env interactions to do between every update")
    parser.add_argument("--n_updates", type=sci2int, default=int(50),
                        help="Number of consecutive policy gradient descent updates to perform")
    parser.add_argument("--replay_size", type=sci2int, default=int(2e5), help="Size of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=128, help="Minibatch size for the optimization")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--alpha", type=float_or_str, default="auto",
                        help="Entropy regularization coefficient. Can be either float value, or 'auto', in which case it is dynamically tuned.")
    parser.add_argument("--target_output_std", type=float, default=0.089,
                        help="If alpha is 'auto', alpha is dynamically tuned so that standard deviation of the action distribution on every dimension matches target_output_std.")

    # DOOM
    parser.add_argument('--render_sleep', type=float, default=0.03, help='Sleep time between frames when rendering')
    parser.add_argument('--render', default=False, action='store_true', help='Render the environment')
    parser.add_argument('--variable_queue_len', type=int, default=5, help='Number of game variables to remember')
    parser.add_argument('--normalize', type=str2bool, default=True, help='Normalize the game state')
    parser.add_argument('--frame_height', type=int, default=84, help='Height of the frame')
    parser.add_argument('--frame_width', type=int, default=84, help='Width of the frame')
    parser.add_argument('--frame_stack', type=int, default=4, help='Number of frames to stack')
    parser.add_argument('--frame_skip', type=int, default=4, help='Number of frames to skip')
    parser.add_argument('--acceleration', default=False, action='store_true', help='Grant the acceleration action')

    # WandB
    parser.add_argument('--with_wandb', default=False, action='store_true', help='Enables Weights and Biases')
    parser.add_argument('--wandb_entity', default=None, type=str, help='WandB username (entity).')
    parser.add_argument('--wandb_project', default='COOM', type=str, help='WandB "Project"')
    parser.add_argument('--wandb_group', default=None, type=str, help='WandB "Group". Name of the env by default.')
    parser.add_argument('--wandb_job_type', default='train', type=str, help='WandB job type')
    parser.add_argument('--wandb_tags', default=[], type=str, nargs='*', help='Tags can help finding experiments')
    parser.add_argument('--wandb_key', default=None, type=str, help='API key for authorizing WandB')
    parser.add_argument('--wandb_dir', default=None, type=str, help='the place to save WandB files')
    parser.add_argument('--wandb_experiment', default='', type=str, help='Identifier to specify the experiment')

    # Reward
    parser.add_argument('--reward_switch_pressed', default=15.0, type=float, help='For pressing a switch')
    parser.add_argument('--reward_frame_survived', default=0.01, type=float, help='For surviving a frame')
    parser.add_argument('--reward_kill', default=5.0, type=float, help='For eliminating an enemy')
    parser.add_argument('--reward_item_acquired', default=15.0, type=float, help='For picking up weapons/health kits')
    parser.add_argument('--reward_delivery', default=30.0, type=float, help='For delivering an item')
    parser.add_argument('--reward_scaler_height', default=1.0, type=float, help='Reward scaler for height')
    parser.add_argument('--reward_scaler_traversal', default=1e-3, type=float,
                        help='Reward scaler for traversing the map')

    # Penalties
    parser.add_argument('--penalty_health_loss', default=0.1, type=float, help='Negative reward for losing health')
    parser.add_argument('--penalty_ammo_used', default=0.1, type=float, help='Negative reward for using ammo')
    parser.add_argument('--penalty_frame_passed', default=0.01, type=float, help='Negative reward for wasting time')

    return parser.parse_args(args=args)
