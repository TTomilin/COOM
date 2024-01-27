import argparse
from datetime import datetime
from pathlib import Path

import tensorflow as tf

from CL.replay.buffers import BufferType
from CL.rl.sac import SAC
from CL.utils.logging import EpochLogger, WandBLogger
from CL.utils.running import get_activation_from_str
from COOM.env.builder import make_env, build_multi_discrete_actions
from COOM.utils.config import Scenario, scenario_config, default_wrapper_config
from config import parse_args, update_wrapper_config


def main(parser: argparse.ArgumentParser):
    args, _ = parser.parse_known_args()

    experiment_dir = Path(__file__).parent.parent.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    record_dir = f"{experiment_dir}/{args.video_folder}/sac/{timestamp}"

    # Task
    task_idx = 0  # one-hot identifier (indicates order among different tasks that we consider)
    num_tasks = 1  # number of tasks, i.e., length of the one-hot encoding, number of tasks that we consider

    # Scenario
    scenario_name = args.scenarios[0]
    scenario_enum = Scenario[scenario_name.upper()]
    scenario_conf = scenario_config[scenario_enum]
    scenario_kwargs = {key: vars(args)[key] for key in scenario_conf['args']}

    # Logging
    if args.with_wandb:
        WandBLogger.add_cli_args(parser)
        WandBLogger(parser, [scenario_name], timestamp)
    logger = EpochLogger(args.logger_output, config=vars(args), group_id=args.group_id)

    # Assign a specified GPU
    if args.gpu:
        # Restrict TensorFlow to only use the specified GPU
        tf.config.experimental.set_visible_devices(args.gpu, 'GPU')
        logger.log(f"Using GPU: {args.gpu}", color='magenta')

    # Configure the arguments
    args = parser.parse_args()
    doom_kwargs = dict(
        num_tasks=num_tasks,
        frame_skip=args.frame_skip,
        record_every=args.record_every,
        seed=args.seed,
        render=args.render,
        render_sleep=args.render_sleep,
        resolution=args.resolution,
        variable_queue_length=args.variable_queue_length,
        action_space_fn=build_multi_discrete_actions,
    )
    wrapper_conf = update_wrapper_config(default_wrapper_config, args)
    wrapper_conf['record_dir'] = record_dir

    # Create the environment
    env = make_env(scenario_enum, args.envs[0], task_idx, scenario_kwargs, doom_kwargs, wrapper_conf)
    test_envs = [make_env(scenario_enum, task, task_idx, scenario_kwargs, doom_kwargs, wrapper_conf)
                 for task in args.test_envs]
    if not test_envs and args.test_only:
        test_envs = [env]

    policy_kwargs = dict(
        hidden_sizes=args.hidden_sizes,
        activation=get_activation_from_str(args.activation),
        use_layer_norm=args.use_layer_norm,
    )

    sac = SAC(
        env,
        test_envs,
        logger,
        scenarios=args.scenarios,
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
        save_freq_epochs=args.save_freq_epochs,
        experiment_dir=experiment_dir,
        model_path=args.model_path,
        timestamp=timestamp,
        test_only=args.test_only,
        num_test_eps=args.test_episodes,
        buffer_type=BufferType(args.buffer_type),
    )
    sac.run()


if __name__ == "__main__":
    main(parse_args())
