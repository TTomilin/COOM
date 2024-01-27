import argparse
from datetime import datetime
from enum import Enum
from pathlib import Path

import tensorflow as tf

from CL.methods.agem import AGEM_SAC
from CL.methods.clonex import ClonExSAC
from CL.methods.ewc import EWC_SAC
from CL.methods.l2 import L2_SAC
from CL.methods.mas import MAS_SAC
from CL.methods.owl import OWL_SAC
from CL.methods.packnet import PackNet_SAC
from CL.methods.vcl import VCL_SAC, VclMlpActor
from CL.replay.buffers import BufferType
from CL.rl.models import MlpActor
from CL.rl.sac import SAC
from CL.utils.logging import EpochLogger, WandBLogger
from CL.utils.running import get_activation_from_str
from COOM.env.builder import make_envs, build_multi_discrete_actions
from COOM.env.continual import ContinualLearningEnv
from COOM.utils.config import Sequence, Scenario, sequence_scenarios, sequence_tasks, default_wrapper_config, \
    scenario_config
from config import parse_args, update_wrapper_config


class CLMethod(Enum):
    SAC = (SAC, [])
    L2 = (L2_SAC, ['cl_reg_coef', 'regularize_critic'])
    EWC = (EWC_SAC, ['cl_reg_coef', 'regularize_critic'])
    MAS = (MAS_SAC, ['cl_reg_coef', 'regularize_critic'])
    VCL = (VCL_SAC, ['cl_reg_coef', 'regularize_critic', 'vcl_first_task_kl'])
    PACKNET = (PackNet_SAC, ['regularize_critic', 'packnet_retrain_steps'])
    AGEM = (AGEM_SAC, ['episodic_mem_per_task', 'episodic_batch_size'])
    OWL = (OWL_SAC, ['cl_reg_coef', 'regularize_critic'])
    CLONEX = (ClonExSAC, ['episodic_mem_per_task', 'episodic_batch_size', 'regularize_critic', 'cl_reg_coef',
                          'episodic_memory_from_buffer'])


def main(parser: argparse.ArgumentParser):
    args, _ = parser.parse_known_args()
    sequence = Sequence[args.sequence.upper()]
    scenarios = sequence_scenarios[sequence] * args.num_repeats
    tasks = sequence_tasks[sequence]
    test_scenarios = [Scenario[scenario.upper()] for scenario in args.scenarios] if args.test_only else scenarios
    test_tasks = args.envs if args.test_only else [] if args.no_test else tasks
    experiment_dir = Path(__file__).parent.parent.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    num_tasks = len(scenarios) * len(tasks)

    # Logging
    if args.with_wandb:
        WandBLogger.add_cli_args(parser)
        WandBLogger(parser, [scenario.name.lower() for scenario in scenarios], timestamp, sequence.name)
    logger = EpochLogger(args.logger_output, config=vars(args), group_id=args.group_id)
    logger.log(f'Task sequence: {args.sequence}', color='magenta')
    logger.log(f'Scenarios: {[s.name for s in scenarios]}', color='magenta')
    logger.log(f'Environments: {tasks}', color='magenta')

    if args.gpu is not None:
        # Restrict TensorFlow to only use the specified GPU
        physical_devices = tf.config.list_physical_devices('GPU')
        logger.log(f"list of physical devices GPU: {physical_devices}", color='magenta')
        try:
            gpu = physical_devices[args.gpu]
        except Exception:
            raise ValueError(f"GPU {args.gpu} not found")
        tf.config.experimental.set_visible_devices(gpu, 'GPU')
        logger.log(f"Using GPU: {gpu}", color='magenta')

    task_idx = scenarios.index(test_scenarios[0]) if args.test_only else None
    record_dir = f"{experiment_dir}/{args.video_folder}/{args.cl_method}/{timestamp}"

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
    if args.test_only:
        args.render = False
    scenario_kwargs = [{key: vars(args)[key] for key in scenario_config[scenario]['args']} for scenario in scenarios]
    wrapper_config = update_wrapper_config(default_wrapper_config, args)
    wrapper_config['record_dir'] = record_dir

    # Create the test tasks
    test_tasks = make_envs(test_scenarios, test_tasks, args.random_order, task_idx,
                           scenario_kwargs, doom_kwargs, wrapper_config)

    # Create the continual learning environment
    cl_env = ContinualLearningEnv(sequence, args.steps_per_env, args.start_from, args.random_order,
                                  scenario_kwargs, doom_kwargs, wrapper_config)

    num_heads = num_tasks if args.multihead_archs else 1
    policy_kwargs = dict(
        hidden_sizes=args.hidden_sizes,
        activation=get_activation_from_str(args.activation),
        use_layer_norm=args.use_layer_norm,
        use_lstm=args.use_lstm,
        num_heads=num_heads,
        hide_task_id=args.hide_task_id,
    )

    cl_method = args.cl_method if args.cl_method is not None else 'sac'
    actor_cl = VclMlpActor if cl_method == "vcl" else MlpActor

    sac_kwargs = dict(
        env=cl_env,
        test_envs=test_tasks,
        test=args.test,
        test_only=args.test_only,
        num_test_eps=args.test_episodes,
        logger=logger,
        scenarios=scenarios,
        cl_method=cl_method,
        seed=args.seed,
        steps_per_env=args.steps_per_env,
        start_steps=args.start_steps,
        start_from_task=args.start_from,
        log_every=args.log_every,
        update_after=args.update_after,
        update_every=args.update_every,
        n_updates=args.n_updates,
        replay_size=args.replay_size,
        batch_size=args.batch_size,
        actor_cl=actor_cl,
        policy_kwargs=policy_kwargs,
        buffer_type=BufferType(args.buffer_type),
        reset_buffer_on_task_change=args.reset_buffer_on_task_change,
        reset_optimizer_on_task_change=args.reset_optimizer_on_task_change,
        lr=args.lr,
        lr_decay=args.lr_decay,
        lr_decay_rate=args.lr_decay_rate,
        lr_decay_steps=args.lr_decay_steps,
        alpha=args.alpha,
        reset_critic_on_task_change=args.reset_critic_on_task_change,
        clipnorm=args.clipnorm,
        gamma=args.gamma,
        target_output_std=args.target_output_std,
        agent_policy_exploration=args.agent_policy_exploration,
        save_freq_epochs=args.save_freq_epochs,
        experiment_dir=experiment_dir,
        model_path=args.model_path,
        timestamp=timestamp,
        exploration_kind=args.exploration_kind,
    )

    sac_class, sac_arg_names = CLMethod[cl_method.upper()].value
    cl_args = [vars(args)[arg] for arg in sac_arg_names]
    sac = sac_class(*cl_args, **sac_kwargs)
    sac.run()


if __name__ == "__main__":
    main(parse_args())
