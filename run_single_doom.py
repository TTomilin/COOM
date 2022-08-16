from argparse import Namespace
from enum import Enum
from pathlib import Path

from gym.wrappers import FrameStack, NormalizeObservation

from coom.doom.env.extended.defend_the_center_impl import DefendTheCenterImpl
from coom.doom.env.extended.dodge_projectiles_impl import DodgeProjectilesImpl
from coom.doom.env.extended.health_gathering_impl import HealthGatheringImpl
from coom.doom.env.extended.seek_and_slay_impl import SeekAndSlayImpl
from coom.doom.env.utils.wrappers import RescaleWrapper, ResizeWrapper
from coom.sac.sac import SAC
from coom.sac.utils.logx import EpochLogger
from coom.utils.utils import get_activation_from_str
from input_args import single_parse_args


class DoomScenario(Enum):
    DEFEND_THE_CENTER = DefendTheCenterImpl
    HEALTH_GATHERING = HealthGatheringImpl
    SEEK_AND_SLAY = SeekAndSlayImpl
    DODGE_PROJECTILES = DodgeProjectilesImpl


def main(logger: EpochLogger, args: Namespace):
    actor_kwargs = dict(
        hidden_sizes=args.hidden_sizes,
        activation=get_activation_from_str(args.activation),
        use_layer_norm=args.use_layer_norm,
    )
    critic_kwargs = dict(
        hidden_sizes=args.hidden_sizes,
        activation=get_activation_from_str(args.activation),
        use_layer_norm=args.use_layer_norm,
    )

    args.experiment_dir = Path(__file__).parent.resolve()

    # Determine scenario and algorithm classes
    scenario_class = DoomScenario[args.scenario.upper()].value

    args.cfg_path = f"{args.experiment_dir}/coom/doom/maps/{args.scenario}/{args.scenario}.cfg"

    task = 'default'
    one_hot_idx = 0  # one-hot identifier (indicates order among different tasks that we consider)
    one_hot_len = 1  # number of tasks, i.e., length of the one-hot encoding, number of tasks that we consider

    env = scenario_class(args, task, one_hot_idx, one_hot_len)
    env = ResizeWrapper(env, args.frame_height, args.frame_width)
    env = RescaleWrapper(env)
    env = NormalizeObservation(env)
    env = FrameStack(env, args.frame_stack)

    sac = SAC(
        env,
        [env],
        logger,
        seed=args.seed,
        steps=args.steps,
        log_every=args.log_every,
        replay_size=args.replay_size,
        batch_size=args.batch_size,
        actor_kwargs=actor_kwargs,
        critic_kwargs=critic_kwargs,
        lr=args.lr,
        alpha=args.alpha,
        gamma=args.gamma,
        target_output_std=args.target_output_std,
    )
    sac.run()


if __name__ == "__main__":
    args = single_parse_args()
    logger = EpochLogger(args.logger_output, config=vars(args), group_id=args.group_id)
    main(logger, args)
