import argparse
import tensorflow as tf
from datetime import datetime
from pathlib import Path

from cl.sac.replay_buffers import BufferType
from cl.sac.sac import SAC
from cl.utils.logx import EpochLogger
from cl.utils.run_utils import get_activation_from_str
from cl.utils.wandb_utils import WandBLogger
from coom.envs import get_single_env, wrap_env
from coom.utils.enums import DoomScenario
from input_args import parse_args


def main(parser: argparse.ArgumentParser):
    args, _ = parser.parse_known_args()

    experiment_dir = Path(__file__).parent.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Task
    task_idx = 0  # one-hot identifier (indicates order among different tasks that we consider)
    num_tasks = 1  # number of tasks, i.e., length of the one-hot encoding, number of tasks that we consider

    # Environment
    scenario = args.scenarios[0]
    scenario_enum = DoomScenario[scenario.upper()].value
    scenario_class = scenario_enum['class']
    scenario_kwargs = {key: vars(args)[key] for key in scenario_enum['kwargs']}
    scenario_class.add_cli_args(parser)

    # Logging
    if args.with_wandb:
        WandBLogger.add_cli_args(parser)
        WandBLogger(parser, [scenario], timestamp)
    logger = EpochLogger(args.logger_output, config=vars(args), group_id=args.group_id)

    if args.gpu:
        # Restrict TensorFlow to only use the specified GPU
        tf.config.experimental.set_visible_devices(args.gpu, 'GPU')
        logger.log(f"Using GPU: {args.gpu}", color='magenta')

    args = parser.parse_args()

    doom_kwargs = dict(
        num_tasks=num_tasks,
        frame_skip=args.frame_skip,
        record_every=args.record_every,
        seed=args.seed,
        render=args.render,
        render_mode=args.render_mode,
        render_sleep=args.render_sleep,
        resolution=args.resolution,
        variable_queue_length=args.variable_queue_length,
    )

    # Create the environment
    record_dir = f"{experiment_dir}/{args.video_folder}/sac/{timestamp}"
    env = get_single_env(logger, scenario_class, args.envs[0], task_idx, scenario_kwargs, doom_kwargs)
    env = wrap_env(env, args.sparse_rewards, args.frame_height, args.frame_width, args.frame_stack, args.use_lstm,
                   args.record, record_dir)
    test_envs = [
        wrap_env(get_single_env(logger, scenario_class, task, task_idx, scenario_kwargs, doom_kwargs),
                 args.sparse_rewards, args.frame_height, args.frame_width, args.frame_stack, args.use_lstm, args.record,
                 record_dir) for task in args.test_envs]
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
