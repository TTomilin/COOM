import gym
import itertools
import numpy as np
from gym.wrappers import NormalizeObservation, FrameStack, RecordVideo
from typing import Any, Dict, List, Tuple, Type

from cl.utils.logx import Logger
from coom.env.scenario.common import CommonEnv
from coom.env.scenario.scenario import DoomEnv
from coom.env.wrappers.observation import RescaleWrapper, ResizeWrapper, RGBStack
from coom.utils.enums import Sequence


class ContinualLearningEnv(CommonEnv):

    def __init__(self, logger: Logger, sequence: Sequence, steps_per_env: int = 2e5, scenario_kwargs: List[Dict[str, any]] = None,
                 doom_kwargs: Dict[str, any] = None):
        self.steps_per_env = steps_per_env
        self._envs = get_doom_envs(logger, sequence.value['scenarios'], sequence.value['envs'], scenario_kwargs, doom_kwargs)
        self._num_tasks = len(self._envs)
        self.steps = steps_per_env * self.num_tasks
        self.cur_step = 0
        self.cur_seq_idx = 0

    def _check_steps_bound(self) -> None:
        if self.cur_step >= self.steps:
            raise RuntimeError("Steps limit exceeded for ContinualLearningEnv!")

    def _get_active_env(self) -> DoomEnv:
        return self.tasks[self.cur_seq_idx]

    @property
    def name(self) -> str:
        return "ContinualLearningEnv"

    @property
    def task(self) -> str:
        return self._get_active_env().name

    @property
    def task_id(self) -> int:
        return self.cur_seq_idx

    @property
    def num_tasks(self) -> int:
        return self._num_tasks

    @property
    def action_space(self) -> gym.spaces.Discrete:
        return self.tasks[0].action_space

    @property
    def observation_space(self) -> gym.Space:
        return self.tasks[0].observation_space

    @property
    def tasks(self):
        return self._envs

    @tasks.setter
    def tasks(self, envs):
        self._envs = envs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self._check_steps_bound()
        obs, reward, done, truncated, info = self._get_active_env().step(action)
        info["seq_idx"] = self.cur_seq_idx

        self.cur_step += 1
        if self.cur_step % self.steps_per_env == 0:
            # If we hit limit for current env, end the episode.
            # This may cause border episodes to end before self-terminating.
            done = True
            info["TimeLimit.truncated"] = True

            if self.cur_seq_idx < self.num_tasks - 1:
                self.cur_seq_idx += 1

        return obs, reward, done, truncated, info

    def render(self, mode="rgb_array"):
        self._get_active_env().render(mode)

    def reset(self) -> np.ndarray:
        self._check_steps_bound()
        return self._get_active_env().reset()

    def get_statistics(self, mode: str = '') -> Dict[str, float]:
        return self._get_active_env().get_statistics(mode)

    def clear_episode_statistics(self) -> None:
        return self._get_active_env().clear_episode_statistics()


def get_doom_envs(logger: Logger, scenarios: List[Type[DoomEnv]], env_names: List[str], scenario_kwargs: List[Dict[str, any]] = None,
                  doom_kwargs: Dict[str, any] = None, task_idx: int = None) -> List[DoomEnv]:
    """
    Returns a list of doom environments.
    :param logger: Logger object
    :param scenarios: list of doom scenarios
    :param env_names: list of doom environment names
    :param scenario_kwargs: scenario specific kwargs
    :param doom_kwargs: doom game specific kwargs
    :param task_idx: task index
    :return: list of doom environments
    """
    scenario_kwargs = scenario_kwargs or [{} for _ in range(len(scenarios))]
    doom_kwargs = doom_kwargs or {}
    envs = []
    for i, pair in enumerate(itertools.product(zip(scenarios, scenario_kwargs), env_names)):
        # If task_idx is specified, use that otherwise use the current index.
        task_id = task_idx if task_idx is not None else i
        doom_scenario = pair[0]
        task = pair[1]
        scenario_class = doom_scenario[0].value['class']
        scenario_kwargs = doom_scenario[1]
        env = get_single_env(logger, scenario_class, task, task_id, scenario_kwargs, doom_kwargs)
        envs.append(env)
    return envs


def get_single_env(logger: Logger, scenario: Type[DoomEnv], task: str = 'default', task_idx: int = 0,
                   scenario_kwargs: Dict[str, any] = None, doom_kwargs: Dict[str, any] = None) -> DoomEnv:
    scenario_kwargs = scenario_kwargs or {}
    doom_kwargs = doom_kwargs or {}
    doom_kwargs['logger'] = logger
    doom_kwargs['env'] = task
    doom_kwargs['task_idx'] = task_idx
    return scenario(doom_kwargs, **scenario_kwargs)


def wrap_env(env: DoomEnv, sparse_rewards: bool = False, frame_height: int = 84, frame_width: int = 84,
             frame_stack: int = 4, record: bool = False, record_dir: str = 'videos') -> gym.Env:
    reward_wrappers = env.reward_wrappers_sparse() if sparse_rewards else env.reward_wrappers_dense()
    for wrapper in reward_wrappers:
        env = wrapper.wrapper_class(env, **wrapper.kwargs)  # Apply the scenario specific reward wrappers
    env = ResizeWrapper(env, frame_height, frame_width)
    env = RescaleWrapper(env)
    env = NormalizeObservation(env)
    env = FrameStack(env, frame_stack)
    env = RGBStack(env)
    if record:
        env = RecordVideo(env, record_dir, episode_trigger=env.video_schedule, name_prefix=f'{env.name}')
    return env
