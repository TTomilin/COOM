from argparse import Namespace
from pathlib import Path

from coom.envs import get_single_env, get_cl_env_doom, get_single_env_doom
from coom.methods.vcl import VclMlpActor
from coom.sac.models import MlpActor
from coom.sac.utils.logx import EpochLogger
from coom.utils.enums import BufferType
from coom.utils.run_utils import get_sac_class
from coom.utils.utils import get_activation_from_str
from input_args import cl_parse_args


def main(
    logger: EpochLogger,
    args: Namespace,
    # tasks: str,
    # task_list: List[str],
    # seed: int,
    # steps_per_task: int,
    # log_every: int,
    # replay_size: int,
    # batch_size: int,
    # hidden_sizes: Iterable[int],
    # buffer_type: str,
    # reset_buffer_on_task_change: bool,
    # reset_optimizer_on_task_change: bool,
    # activation: Callable,
    # use_layer_norm: bool,
    # lr: float,
    # gamma: float,
    # alpha: str,
    # target_output_std: float,
    # cl_method: str,
    # packnet_retrain_steps: int,
    # regularize_critic: bool,
    # cl_reg_coef: float,
    # vcl_first_task_kl: bool,
    # episodic_mem_per_task: int,
    # episodic_batch_size: int,
    # reset_critic_on_task_change: bool,
    # multihead_archs: bool,
    # hide_task_id: bool,
    # clipnorm: float,
    # agent_policy_exploration: bool,
):
    tasks = args.tasks
    args.experiment_dir = Path(__file__).parent.resolve()
    train_env = get_cl_env_doom(args)
    # Consider normalizing test envs in the future.
    num_tasks = len(tasks)
    test_envs = [
        get_single_env_doom(args, task, one_hot_idx=i, one_hot_len=num_tasks) for i, task in enumerate(tasks)
    ]
    steps = args.steps_per_task * num_tasks

    num_heads = num_tasks if args.multihead_archs else 1
    actor_kwargs = dict(
        hidden_sizes=args.hidden_sizes,
        activation=get_activation_from_str(args.activation),
        use_layer_norm=args.use_layer_norm,
        num_heads=num_heads,
        hide_task_id=args.hide_task_id,
    )
    critic_kwargs = dict(
        hidden_sizes=args.hidden_sizes,
        activation=get_activation_from_str(args.activation),
        use_layer_norm=args.use_layer_norm,
        num_heads=num_heads,
        hide_task_id=args.hide_task_id,
    )

    if args.cl_method == "vcl":
        actor_cl = VclMlpActor
    else:
        actor_cl = MlpActor

    vanilla_sac_kwargs = {
        "env": train_env,
        "test_envs": test_envs,
        "logger": logger,
        "seed": args.seed,
        "steps": steps,
        "log_every": args.log_every,
        "replay_size": args.replay_size,
        "batch_size": args.batch_size,
        "actor_cl": actor_cl,
        "actor_kwargs": actor_kwargs,
        "critic_kwargs": critic_kwargs,
        "buffer_type": BufferType(args.buffer_type),
        "reset_buffer_on_task_change": args.reset_buffer_on_task_change,
        "reset_optimizer_on_task_change": args.reset_optimizer_on_task_change,
        "lr": args.lr,
        "alpha": args.alpha,
        "reset_critic_on_task_change": args.reset_critic_on_task_change,
        "clipnorm": args.clipnorm,
        "gamma": args.gamma,
        "target_output_std": args.target_output_std,
        "agent_policy_exploration": args.agent_policy_exploration,
    }

    sac_class = get_sac_class(args.cl_method)

    if args.cl_method is None:
        sac = sac_class(**vanilla_sac_kwargs)
    elif args.cl_method in ["l2", "ewc", "mas"]:
        sac = sac_class(
            **vanilla_sac_kwargs, cl_reg_coef=args.cl_reg_coef, regularize_critic=args.regularize_critic
        )
    elif args.cl_method == "vcl":
        sac = sac_class(
            **vanilla_sac_kwargs,
            cl_reg_coef=args.cl_reg_coef,
            regularize_critic=args.regularize_critic,
            first_task_kl=args.vcl_first_task_kl
        )
    elif args.cl_method == "packnet":
        sac = sac_class(
            **vanilla_sac_kwargs,
            regularize_critic=args.regularize_critic,
            retrain_steps=args.packnet_retrain_steps
        )
    elif args.cl_method == "agem":
        sac = sac_class(
            **vanilla_sac_kwargs,
            episodic_mem_per_task=args.episodic_mem_per_task,
            episodic_batch_size=args.episodic_batch_size
        )
    else:
        raise NotImplementedError("This method is not implemented")
    sac.run()


if __name__ == "__main__":
    args = cl_parse_args()
    logger = EpochLogger(args.logger_output, config=vars(args), group_id=args.group_id)
    main(logger, args)
