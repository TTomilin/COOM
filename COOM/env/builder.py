import itertools
from typing import Dict, List

from gymnasium.wrappers import NormalizeObservation, FrameStack, RecordVideo

from COOM.env.scenario import DoomEnv
from COOM.utils.config import Sequence, sequence_scenarios, sequence_tasks, scenario_config, Scenario, \
    default_wrapper_config
from COOM.wrappers.observation import Augment, Resize, Rescale, RGBStack


def make_sequence(sequence: Sequence,
                  random_order: bool = False,
                  scenarios_kwargs: List[Dict[str, any]] = None,
                  doom_kwargs: Dict[str, any] = None,
                  wrapper_config: Dict[str, any] = None,
                  task_idx: int = None) -> List[DoomEnv]:
    """
    Creates a list of Doom environments based on the given sequence configuration.

    Args:
        sequence (Sequence): The sequence enumeration to determine which scenarios to include.
        random_order (bool): Whether to randomize the order of the scenarios.
        task_idx (int): Optional task index to be used for all environments.
        scenarios_kwargs (List[Dict[str, any]]): List of dictionaries with specific kwargs for each scenario.
        doom_kwargs (Dict[str, any]): Common kwargs applicable to all Doom environments.
        wrapper_config (Dict[str, any]): Configuration for environment wrappers.

    Returns:
        List[DoomEnv]: A list of Doom environment instances.
    """

    # Retrieve scenarios and tasks based on the sequence
    scenarios = sequence_scenarios[sequence]
    tasks = sequence_tasks[sequence]
    return make_envs(scenarios, tasks, random_order, task_idx, scenarios_kwargs, doom_kwargs, wrapper_config)


def make_envs(scenarios: List[Scenario],
              tasks: List[str],
              random_order: bool = False,
              task_idx: int = None,
              scenarios_kwargs: List[Dict[str, any]] = None,
              doom_kwargs: Dict[str, any] = None,
              wrapper_config: Dict[str, any] = None) -> List[DoomEnv]:

    # Optionally shuffle scenarios and tasks for randomization
    if random_order:
        import random
        random.shuffle(scenarios)
        random.shuffle(tasks)

    # Default kwargs for scenarios and Doom environments
    scenarios_kwargs = scenarios_kwargs or [{} for _ in range(len(scenarios))]
    doom_kwargs = doom_kwargs or {}

    # Create and wrap environments
    envs = []
    for i, pair in enumerate(itertools.product(zip(scenarios, scenarios_kwargs), tasks)):
        # If task_idx is specified, use that otherwise use the current index.
        task_id = task_idx if task_idx is not None else i
        scenario_and_kwargs = pair[0]
        task = pair[1]
        scenario = scenario_and_kwargs[0]
        scenario_kwargs = scenario_and_kwargs[1]
        env = make_env(scenario, task, task_id, scenario_kwargs, doom_kwargs, wrapper_config)
        envs.append(env)
    return envs


def make_env(scenario: Scenario,
             task: str = 'default',
             task_idx: int = 0,
             scenario_kwargs: Dict[str, any] = None,
             doom_kwargs: Dict[str, any] = None,
             wrapper_config: Dict[str, any] = None) -> DoomEnv:
    """
    Creates a single Doom environment instance with specified configurations.

    Args:
        scenario (Scenario): The specific Doom scenario to create.
        task (str): The task name within the scenario.
        task_idx (int): The index of the task within the scenario.
        scenario_kwargs (Dict[str, any]): Additional kwargs for the scenario.
        doom_kwargs (Dict[str, any]): Common kwargs for Doom environments.
        wrapper_config (Dict[str, any]): Configuration for environment wrappers.

    Returns:
        DoomEnv: An instance of the Doom environment.
    """

    # Retrieve the scenario class and create an instance
    scenario_class = scenario_config[scenario]['class']
    scenario_kwargs = scenario_kwargs or {}
    doom_kwargs = doom_kwargs or {'env': task, 'task_idx': task_idx, 'action_space_fn': build_multi_discrete_actions}
    env = scenario_class(doom_kwargs, **scenario_kwargs)

    # Apply wrappers to the environment
    env = wrap_env(env, wrapper_config or default_wrapper_config)
    return env


def wrap_env(env: DoomEnv, wrap_conf: Dict[str, any]):
    """
    Applies a series of wrappers to the Doom environment based on the provided configuration.

    Args:
        env (DoomEnv): The Doom environment to be wrapped.
        wrap_conf (Dict[str, any]): Configuration dict specifying which wrappers to apply.

    Returns:
        gymnasium.Env: The wrapped environment.
    """

    # Apply reward wrappers based on the sparse_rewards configuration
    sparse_rewards = wrap_conf.get('sparse_rewards', False)
    reward_wrappers = env.reward_wrappers_sparse() if sparse_rewards else env.reward_wrappers_dense()
    for wrapper in reward_wrappers:
        env = wrapper.wrapper_class(env, **wrapper.kwargs)

    # Apply various observation and utility wrappers
    if wrap_conf.get('augment', False):
        env = Augment(env, wrap_conf['augmentation'])
    if wrap_conf.get('resize', False):
        assert wrap_conf.get('frame_height', None) is not None and wrap_conf.get('frame_width', None) is not None
        env = Resize(env, wrap_conf['frame_height'], wrap_conf['frame_width'])
    if wrap_conf.get('rescale', False):
        env = Rescale(env)
    if wrap_conf.get('normalize_observation', False):
        env = NormalizeObservation(env)
    if wrap_conf.get('frame_stack', False):
        env = FrameStack(env, wrap_conf['frame_stack'])
    if wrap_conf.get('lstm', False):
        env = RGBStack(env)
    if wrap_conf.get('record', False):
        env = RecordVideo(env, wrap_conf['record_dir'], episode_trigger=env.video_schedule, name_prefix=f'{env.name}')
    return env


def build_discrete_actions():
    """
    Builds a list of discrete actions where each action is represented by a list of four booleans.
    The actions are:
        - TURN_LEFT: [True, False, False, False]
        - TURN_RIGHT: [False, True, False, False]
        - MOVE_FORWARD: [False, False, True, False]
        - EXECUTE: [False, False, False, True]
    :return: List of actions
    """
    turn_left = [True, False, False, False]
    turn_right = [False, True, False, False]
    move_forward = [False, False, True, False]
    execute = [False, False, False, True]

    return [turn_left, turn_right, move_forward, execute]


def build_multi_discrete_actions():
    """
    Builds a unified multi-discrete action space for all COOM environments.
    A unified action space simplifies training a single agent on multiple environments.
    There are 12 possible actions:
        - Turn left or right
        - Move forward or not
        - Execute action or not
    The execute action depends on the scenario. It is used to press buttons, shoot, jump or accelerate.
    :return: MultiDiscrete action space
    """
    actions = []
    t_left_right = [[False, False], [False, True], [True, False]]
    m_forward = [[False], [True]]
    execute = [[False], [True]]
    for turn in t_left_right:
        for move in m_forward:
            for action in execute:
                actions.append(turn + move + action)
    return actions
