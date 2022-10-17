from argparse import Namespace
from datetime import datetime
from pathlib import Path

from coom.envs import get_single_env
from coom.sac.sac import SAC
from coom.sac.utils.logx import EpochLogger
from coom.utils.enums import DoomScenario
from coom.utils.utils import get_activation_from_str
from coom.utils.wandb import init_wandb
from input_args import single_parse_args


def main(args: Namespace):
    policy_kwargs = dict(
        hidden_sizes=args.hidden_sizes,
        activation=get_activation_from_str(args.activation),
        use_layer_norm=args.use_layer_norm,
    )

    args.experiment_dir = Path(__file__).parent.resolve()
    args.cfg_path = f"{args.experiment_dir}/coom/doom/maps/{args.scenario}/{args.scenario}.cfg"
    args.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Logging
    init_wandb(args)
    logger = EpochLogger(args.logger_output, config=vars(args), group_id=args.group_id)

    # Task
    task = 'default'
    one_hot_idx = 0  # one-hot identifier (indicates order among different tasks that we consider)
    one_hot_len = 1  # number of tasks, i.e., length of the one-hot encoding, number of tasks that we consider

    # Environment
    scenario_class = DoomScenario[args.scenario.upper()].value
    env = get_single_env(args, scenario_class, task, one_hot_idx, one_hot_len)
    test_envs = [get_single_env(args, scenario_class, task, one_hot_idx, one_hot_len) for task in args.test_tasks]

    sac = SAC(
        env,
        test_envs,
        logger,
        scenarios=[args.scenario],
        seed=args.seed,
        steps_per_env=args.steps_per_env,
        start_steps=args.start_steps,
        log_every=args.log_every,
        update_after=args.update_after,
        update_every=args.update_every,
        n_updates=args.n_updates,
        replay_size=args.replay_size,
        batch_size=args.batch_size,
        policy_kwargs=policy_kwargs,
        lr=args.lr,
        lr_decay=args.lr_decay,
        lr_decay_rate=args.lr_decay_rate,
        lr_decay_steps=args.lr_decay_steps,
        alpha=args.alpha,
        gamma=args.gamma,
        target_output_std=args.target_output_std,
        render=args.render,
        render_sleep=args.render_sleep,
        save_freq_epochs=args.save_freq_epochs,
        experiment_dir=args.experiment_dir,
        model_path=args.model_path,
        timestamp=args.timestamp,
    )
    sac.run()


if __name__ == "__main__":
    main(single_parse_args())
