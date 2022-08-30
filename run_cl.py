from argparse import Namespace
from pathlib import Path

from coom.envs import get_cl_env
from coom.methods.vcl import VclMlpActor
from coom.sac.models import MlpActor
from coom.sac.utils.logx import EpochLogger
from coom.utils.enums import BufferType
from coom.utils.run_utils import get_sac_class
from coom.utils.utils import get_activation_from_str
from input_args import cl_parse_args


def main(logger: EpochLogger, args: Namespace):
    args.experiment_dir = Path(__file__).parent.resolve()
    args.cfg_path = f"{args.experiment_dir}/coom/doom/maps/{args.scenario}/{args.scenario}.cfg"

    train_env = get_cl_env(args)
    num_tasks = len(args.tasks)
    test_envs = []
    steps_per_env = args.steps_per_env

    num_heads = num_tasks if args.multihead_archs else 1
    policy_kwargs = dict(
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
        "steps_per_env": steps_per_env,
        "log_every": args.log_every,
        "replay_size": args.replay_size,
        "batch_size": args.batch_size,
        "actor_cl": actor_cl,
        "policy_kwargs": policy_kwargs,
        "buffer_type": BufferType(args.buffer_type),
        "reset_buffer_on_task_change": args.reset_buffer_on_task_change,
        "reset_optimizer_on_task_change": args.reset_optimizer_on_task_change,
        "lr": args.lr,
        "lr_decay": args.lr_decay,
        "lr_decay_rate": args.lr_decay_rate,
        "lr_decay_steps": args.lr_decay_steps,
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
