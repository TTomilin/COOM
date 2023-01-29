import numpy as np
import tensorflow as tf
from argparse import Namespace
from datetime import datetime
from pathlib import Path

from coom.envs import get_cl_env, get_single_envs, ContinualLearningEnv
from coom.methods.vcl import VclMlpActor
from coom.sac.models import MlpActor
from coom.sac.utils.logx import EpochLogger
from coom.utils.enums import BufferType, Sequence, DoomScenario
from coom.utils.run_utils import get_sac_class
from coom.utils.utils import get_activation_from_str
from coom.utils.wandb_utils import init_wandb
from input_args import parse_args


def main(args: Namespace):
    sequence = Sequence[args.sequence.upper()].value
    scenarios = sequence['scenarios']
    envs = sequence['envs']
    test_scenarios = [DoomScenario[scenario.upper()] for scenario in args.scenarios] if args.test_only else scenarios
    test_envs = args.envs if args.test_only else envs
    args.experiment_dir = Path(__file__).parent.resolve()
    args.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.num_tasks = len(scenarios) * len(envs)

    if args.gpu:
        # Restrict TensorFlow to only use the specified GPU
        tf.config.experimental.set_visible_devices(args.gpu, 'GPU')
        print("Using GPU: ", args.gpu)

    # Logging
    init_wandb(args, [scenario.name.lower() for scenario in scenarios])
    logger = EpochLogger(args.logger_output, config=vars(args), group_id=args.group_id)
    logger.log(f'Task sequence: {args.sequence}')
    logger.log(f'Scenarios: {[s.name for s in scenarios]}')
    logger.log(f'Environments: {envs}')

    one_hot_id = scenarios.index(test_scenarios[0]) if args.test_only else None
    test_envs = get_single_envs(args, test_scenarios, test_envs, one_hot_id)
    if args.test_only:
        args.render = False
    train_env = get_cl_env(args, scenarios, envs)

    num_heads = args.num_tasks if args.multihead_archs else 1
    policy_kwargs = dict(
        hidden_sizes=args.hidden_sizes,
        activation=get_activation_from_str(args.activation),
        use_layer_norm=args.use_layer_norm,
        num_heads=num_heads,
        hide_task_id=args.hide_task_id,
    )

    actor_cl = VclMlpActor if args.cl_method == "vcl" else MlpActor

    vanilla_sac_kwargs = {
        "env": train_env,
        "test_envs": test_envs,
        "test": args.test,
        "test_only": args.test_only,
        "num_test_eps_stochastic": args.test_episodes,
        "logger": logger,
        "scenarios": scenarios,
        "cl_method": args.cl_method,
        "seed": args.seed,
        "steps_per_env": args.steps_per_env,
        "start_steps": args.start_steps,
        "log_every": args.log_every,
        "update_after": args.update_after,
        "update_every": args.update_every,
        "n_updates": args.n_updates,
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
        "save_freq_epochs": args.save_freq_epochs,
        "experiment_dir": args.experiment_dir,
        "model_path": args.model_path,
        "timestamp": args.timestamp,
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
    main(parse_args())
