import argparse

from coom.tasks import TASK_SEQS
from coom.utils.enums import BufferType
from coom.utils.utils import float_or_str, sci2int, str2bool


def cl_parse_args(args=None):
    parser = argparse.ArgumentParser(description="Continual World")

    parser.add_argument('--scenario', type=str, default=None,
                        choices=['defend_the_center', 'health_gathering', 'seek_and_slay', 'dodge_projectiles'])
    parser.add_argument('--algorithm', type=str, default='sac', choices=['sac'])
    parser.add_argument("--cl_method", type=str, choices=[None, "l2", "ewc", "mas", "vcl", "packnet", "agem"],
                        default=None,
                        help="If None, finetuning method will be used. If one of 'l2', 'ewc', 'mas', 'vcl', 'packnet', 'agem', respective method will be used.")
    parser.add_argument("--tasks", type=str, nargs="+", default=None, help="Name of the tasks you want to run")
    parser.add_argument("--seed", type=int, default=0, help="Seed for randomness")
    parser.add_argument('--watch', default=False, action='store_true', help='watch the play of pre-trained policy only')

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

    # Learning
    parser.add_argument("--steps", type=sci2int, default=int(1e7), help="Number of steps the algorithm will run for")
    parser.add_argument("--replay_size", type=sci2int, default=int(1e5), help="Size of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=128, help="Minibatch size for the optimization")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument('--lr_decay', type=str, default=None, choices=['linear', 'exponential'], help='Decay the learning rate over time')
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--alpha", type=float_or_str, default="auto",
                        help="Entropy regularization coefficient. Can be either float value, or 'auto', in which case it is dynamically tuned.")
    parser.add_argument("--target_output_std", type=float, default=0.089,
                        help="If alpha is 'auto', alpha is dynamically tuned so that standard deviation of the action distribution on every dimension matches target_output_std.")
    parser.add_argument("--steps_per_task", type=sci2int, default=int(1e6), help="Numer of steps per task")
    parser.add_argument("--buffer_type", type=str, default="fifo", choices=[b.value for b in BufferType],
                        help="Strategy of inserting examples into the buffer")
    parser.add_argument("--regularize_critic", type=str2bool, default=False,
                        help="If True, both actor and critic are regularized; if False, only actor is")
    parser.add_argument("--multihead_archs", type=str2bool, default=True, help="Whether use multi-head architecture")
    parser.add_argument("--hide_task_id", type=str2bool, default=True,
                        help="if True, one-hot encoding of the task will not be appended to observation")
    parser.add_argument("--clipnorm", type=float, default=None, help="Value for gradient clipping")
    parser.add_argument("--agent_policy_exploration", type=str2bool, default=False,
                        help="If True, uniform exploration for start_steps steps is used only in the first task (in continual learning). Otherwise, it is used in every task")

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
    parser.add_argument('--render_sleep', type=float, default=0.)
    parser.add_argument('--render', default=False, action='store_true')
    parser.add_argument('--variable_queue_len', type=int, default=5)
    parser.add_argument('--normalize', type=bool, default=True)
    parser.add_argument('--frame-height', type=int, default=84)
    parser.add_argument('--frame-width', type=int, default=84)
    parser.add_argument('--frame-stack', type=int, default=4)
    parser.add_argument('--frame-skip', type=int, default=4)

    # WandB
    parser.add_argument('--with_wandb', default=True, type=bool, help='Enables Weights and Biases')
    parser.add_argument('--wandb_entity', default=None, type=str, help='WandB username (entity).')
    parser.add_argument('--wandb_project', default='COOM', type=str, help='WandB "Project"')
    parser.add_argument('--wandb_group', default=None, type=str, help='WandB "Group". Name of the env by default.')
    parser.add_argument('--wandb_job_type', default='train', type=str, help='WandB job type')
    parser.add_argument('--wandb_tags', default=[], type=str, nargs='*', help='Tags can help finding experiments')
    parser.add_argument('--wandb_key', default=None, type=str, help='API key for authorizing WandB')
    parser.add_argument('--wandb_dir', default=None, type=str, help='the place to save WandB files')

    # Scenario specific
    parser.add_argument('--kill_reward', default=1.0, type=float, help='For eliminating an enemy')
    parser.add_argument('--health_acquired_reward', default=1.0, type=float, help='For picking up health kits')
    parser.add_argument('--health_loss_penalty', default=0.1, type=float, help='Negative reward for losing health')
    parser.add_argument('--ammo_used_penalty', default=0.1, type=float, help='Negative reward for using ammo')
    parser.add_argument('--traversal_reward_scaler', default=1e-3, type=float,
                        help='Reward scaler for traversing the map')
    parser.add_argument('--add_speed', default=False, action='store_true')

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
        "--steps_per_task", type=sci2int, default=int(1e6), help="Numer of steps per task"
    )
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
        # default=[256, 256, 256, 256],
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
                        choices=['defend_the_center', 'health_gathering', 'seek_and_slay', 'dodge_projectiles'])
    parser.add_argument('--algorithm', type=str, default='sac', choices=['sac'])
    parser.add_argument("--task", type=str, help="Name of the task")
    parser.add_argument('--tasks', type=str, nargs='*', default=['default'])
    parser.add_argument('--test_tasks', type=str, nargs='*', default=['default'])
    parser.add_argument("--seed", type=int, default=0, help="Seed for randomness")
    parser.add_argument('--watch', default=False, action='store_true', help='watch the play of pre-trained policy only')
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
    # Learning
    parser.add_argument("--steps", type=sci2int, default=int(1e6), help="Number of steps the algorithm will run for")
    parser.add_argument("--replay_size", type=sci2int, default=int(1e5), help="Size of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=128, help="Minibatch size for the optimization")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument('--lr_decay', type=str, default=None, choices=['linear', 'exponential'], help='Decay the learning rate over time')
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--alpha", type=float_or_str, default="auto",
                        help="Entropy regularization coefficient. Can be either float value, or 'auto', in which case it is dynamically tuned.")
    parser.add_argument("--target_output_std", type=float, default=0.089,
                        help="If alpha is 'auto', alpha is dynamically tuned so that standard deviation of the action distribution on every dimension matches target_output_std.")
    # DOOM
    parser.add_argument('--render_sleep', type=float, default=0.)
    parser.add_argument('--render', default=False, action='store_true')
    parser.add_argument('--variable_queue_len', type=int, default=5)
    parser.add_argument('--normalize', type=bool, default=True)
    parser.add_argument('--frame-height', type=int, default=84)
    parser.add_argument('--frame-width', type=int, default=84)
    parser.add_argument('--frame-stack', type=int, default=4)
    parser.add_argument('--frame-skip', type=int, default=4)
    # WandB
    parser.add_argument('--with_wandb', default=True, type=bool, help='Enables Weights and Biases')
    parser.add_argument('--wandb_entity', default=None, type=str, help='WandB username (entity).')
    parser.add_argument('--wandb_project', default='COOM', type=str, help='WandB "Project"')
    parser.add_argument('--wandb_group', default=None, type=str, help='WandB "Group". Name of the env by default.')
    parser.add_argument('--wandb_job_type', default='train', type=str, help='WandB job type')
    parser.add_argument('--wandb_tags', default=[], type=str, nargs='*', help='Tags can help finding experiments')
    parser.add_argument('--wandb_key', default=None, type=str, help='API key for authorizing WandB')
    parser.add_argument('--wandb_dir', default=None, type=str, help='the place to save WandB files')
    # Scenario specific
    parser.add_argument('--kill_reward', default=1.0, type=float, help='For eliminating an enemy')
    parser.add_argument('--health_acquired_reward', default=1.0, type=float, help='For picking up health kits')
    parser.add_argument('--health_loss_penalty', default=0.1, type=float, help='Negative reward for losing health')
    parser.add_argument('--ammo_used_penalty', default=0.1, type=float, help='Negative reward for using ammo')
    parser.add_argument('--traversal_reward_scaler', default=1e-3, type=float,
                        help='Reward scaler for traversing the map')
    parser.add_argument('--add_speed', default=False, action='store_true')

    return parser.parse_args(args=args)
