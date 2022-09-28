import itertools
from argparse import Namespace
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union, Type

import gym
import numpy as np
from gym.wrappers import NormalizeObservation, FrameStack, RecordVideo

from coom.env.base.common import CommonEnv
from coom.env.base.scenario import DoomEnv
from coom.env.utils.wrappers import ResizeWrapper, RescaleWrapper, RGBStack
from coom.utils.enums import DoomScenario


class ContinualLearningEnv(CommonEnv):

    def __init__(self, envs: List[DoomEnv], steps_per_env: int) -> None:
        for i in range(len(envs)):
            assert envs[0].action_space == envs[i].action_space
        self.action_space = envs[0].action_space
        self.observation_space = deepcopy(envs[0].observation_space)
        self.envs = envs
        self.n_tasks = len(envs)
        self.steps_per_env = steps_per_env
        self.steps = steps_per_env * self.num_tasks
        self.cur_step = 0
        self.cur_seq_idx = 0

    def _check_steps_bound(self) -> None:
        if self.cur_step >= self.steps:
            raise RuntimeError("Steps limit exceeded for ContinualLearningEnv!")

    def _get_active_env(self) -> DoomEnv:
        return self.envs[self.cur_seq_idx]

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
        return self.n_tasks

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict]:
        self._check_steps_bound()
        obs, reward, done, info = self._get_active_env().step(action)
        info["seq_idx"] = self.cur_seq_idx

        self.cur_step += 1
        if self.cur_step % self.steps_per_env == 0:
            # If we hit limit for current env, end the episode.
            # This may cause border episodes to end before self-terminating.
            done = True
            info["TimeLimit.truncated"] = True

            if self.cur_seq_idx < self.num_tasks - 1:
                self.cur_seq_idx += 1

        return obs, reward, done, info

    def render(self, mode="rgb_array"):
        self._get_active_env().render(mode)

    def reset(self) -> np.ndarray:
        self._check_steps_bound()
        return self._get_active_env().reset()

    def get_statistics(self, mode: str = '') -> Dict[str, float]:
        return self._get_active_env().get_statistics(mode)

    def clear_episode_statistics(self) -> None:
        return self._get_active_env().clear_episode_statistics()


def get_cl_env(args: Namespace) -> ContinualLearningEnv:
    """Returns a continual learning environment.
    Args:
      args: list of the input arguments
    Returns:
      gym.Env: continual learning environment
    """
    envs = get_single_envs(args)
    cl_env = ContinualLearningEnv(envs, args.steps_per_env)
    return cl_env


class MultiTaskEnv(gym.Env):

    def __init__(
            self, envs: List[gym.Env], steps_per_env: int, cycle_mode: str = "episode"
    ) -> None:
        assert cycle_mode == "episode"
        for i in range(len(envs)):
            assert envs[0].action_space == envs[i].action_space
        self.action_space = envs[0].action_space
        self.observation_space = deepcopy(envs[0].observation_space)
        self.envs = envs
        self.num_envs = len(envs)
        self.steps_per_env = steps_per_env
        self.cycle_mode = cycle_mode
        self.steps = self.num_envs * self.steps_per_env
        self.cur_step = 0
        self._cur_seq_idx = 0

    def _check_steps_bound(self) -> None:
        if self.cur_step >= self.steps:
            raise RuntimeError("Steps limit exceeded for MultiTaskEnv!")

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict]:
        self._check_steps_bound()
        obs, reward, done, info = self.envs[self._cur_seq_idx].step(action)
        info["mt_seq_idx"] = self._cur_seq_idx
        if self.cycle_mode == "step":
            self._cur_seq_idx = (self._cur_seq_idx + 1) % self.num_envs
        self.cur_step += 1

        return obs, reward, done, info

    def render(self, mode="human"):
        self.envs[self._cur_seq_idx].render(mode)

    def reset(self) -> np.ndarray:
        self._check_steps_bound()
        if self.cycle_mode == "episode":
            self._cur_seq_idx = (self._cur_seq_idx + 1) % self.num_envs
        obs = self.envs[self._cur_seq_idx].reset()
        return obs


def get_mt_env(
        tasks: List[Union[int, str]], steps_per_task: int, randomization: str = "random_init_all"
):
    """Returns multi-task learning environment.

    Args:
      tasks: list of task names or MT50 numbers
      steps_per_task: agent will be limited to steps_per_task * len(tasks) steps
      randomization: randomization kind, one of 'deterministic', 'random_init_all',
                     'random_init_fixed20', 'random_init_small_box'.

    Returns:
      gym.Env: continual learning environment
    """
    # task_names = [get_task_name(task) for task in tasks]
    # num_tasks = len(task_names)
    envs = []
    # for i, task_name in enumerate(task_names):
    # env = MT50.train_classes[task_name]()
    # env = RandomizationWrapper(env, get_subtasks(task_name), randomization)
    # env = OneHotAdder(env, one_hot_idx=i, one_hot_len=num_tasks)
    # env.name = task_name
    # env = TimeLimit(env, META_WORLD_TIME_HORIZON)
    # env = SuccessCounter(env)
    # envs.append(env)
    mt_env = MultiTaskEnv(envs, steps_per_task)
    mt_env.name = "MultiTaskEnv"
    return mt_env


def get_single_envs(args):
    envs = [get_single_env(args, DoomScenario[scenario_task[0].upper()].value, scenario_task[1], one_hot_idx=i,
                           one_hot_len=args.num_tasks) for i, scenario_task in
            enumerate(itertools.product(args.scenarios, args.envs))]
    return envs


def get_single_env(args: Namespace, scenario_class: Type[DoomEnv], task: str, one_hot_idx: int,
                   one_hot_len: int) -> DoomEnv:
    """Returns a single task environment.

    Args:
      :param args: Dictionary of input arguments
      :param scenario_class: Class of the Doom scenario
      :param task: task name
      :param one_hot_idx: one-hot identifier (indicates order among different tasks that we consider)
      :param one_hot_len: length of the one-hot encoding, number of tasks that we consider

    Returns:
      :return DoomEnv: single-task Doom environment
    """
    env = scenario_class(args, task, one_hot_idx, one_hot_len)
    for wrapper in env.reward_wrappers():
        env = wrapper.wrapper_class(env, *wrapper.args)  # Apply the scenario specific reward wrappers
    env = ResizeWrapper(env, args.frame_height, args.frame_width)
    env = RescaleWrapper(env)
    if args.normalize:
        env = NormalizeObservation(env)
    env = FrameStack(env, args.frame_stack)
    env = RGBStack(env)
    if args.record:
        method = args.cl_method if 'cl_method' in args else 'sac'
        env = RecordVideo(env, f"{args.experiment_dir}/{args.video_folder}/{method}/{args.timestamp}",
                          step_trigger=env.video_schedule, name_prefix=f'{env.name}')
    return env
