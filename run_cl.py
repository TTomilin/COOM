import argparse
import tensorflow as tf
from datetime import datetime
from enum import Enum
from pathlib import Path

from cl.methods.agem import AGEM_SAC
from cl.methods.clonex import ClonExSAC
from cl.methods.ewc import EWC_SAC
from cl.methods.l2 import L2_SAC
from cl.methods.mas import MAS_SAC
from cl.methods.owl import OWL_SAC
from cl.methods.packnet import PackNet_SAC
from cl.methods.vcl import VCL_SAC, VclMlpActor
from cl.sac.models import MlpActor
from cl.sac.replay_buffers import BufferType
from cl.sac.sac import SAC
from cl.utils.logx import EpochLogger
from cl.utils.run_utils import get_activation_from_str
from cl.utils.wandb_utils import WandBLogger
from coom.envs import get_doom_envs, ContinualLearningEnv, wrap_env
from coom.utils.enums import Sequence, DoomScenario
from input_args import parse_args


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
    scenarios = sequence.value['scenarios'] * args.repeat_sequence
    envs = sequence.value['envs']
    test_scenarios = [DoomScenario[scenario.upper()] for scenario in args.scenarios] if args.test_only else scenarios
    test_envs = args.envs if args.test_only else [] if args.no_test else envs
    experiment_dir = Path(__file__).parent.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    num_tasks = len(scenarios) * len(envs)

    # Logging
    if args.with_wandb:
        WandBLogger.add_cli_args(parser)
        WandBLogger(parser, [scenario.name.lower() for scenario in scenarios], timestamp, sequence.name)
    logger = EpochLogger(args.logger_output, config=vars(args), group_id=args.group_id)
    logger.log(f'Task sequence: {args.sequence}', color='magenta')
    logger.log(f'Scenarios: {[s.name for s in scenarios]}', color='magenta')
    logger.log(f'Environments: {envs}', color='magenta')

    if args.gpu is not None:
        # Restrict TensorFlow to only use the specified GPU
        physical_devices = tf.config.list_physical_devices('GPU')
        logger.log(f"list of physical devices GPU: {physical_devices}", color='magenta')
        gpu = physical_devices[args.gpu]
        tf.config.experimental.set_visible_devices(gpu, 'GPU')
        logger.log(f"Using GPU: {gpu}", color='magenta')

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

    record_dir = f"{experiment_dir}/{args.video_folder}/sac/{timestamp}"
    task_idx = scenarios.index(test_scenarios[0]) if args.test_only else None
    test_envs = get_doom_envs(logger, test_scenarios, test_envs, task_idx=task_idx, doom_kwargs=doom_kwargs)
    test_envs = [wrap_env(env, args.sparse_rewards, args.frame_height, args.frame_width, args.frame_stack,
                          args.use_lstm, args.record, record_dir)
                 for env in test_envs]
    if args.test_only:
        args.render = False

    scenario_kwargs = [{key: vars(args)[key] for key in scenario_enum.value['kwargs']} for scenario_enum in scenarios]
    cl_env = ContinualLearningEnv(logger, scenarios, envs, args.steps_per_env, args.start_from, scenario_kwargs,
                                  doom_kwargs)
    cl_env.tasks = [
        wrap_env(env, args.sparse_rewards, args.frame_height, args.frame_width, args.frame_stack, args.use_lstm,
                 args.record, record_dir, args.augment, args.augmentation)
        for env in cl_env.tasks]

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
        test_envs=test_envs,
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
