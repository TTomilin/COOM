from typing import Dict, Any, Tuple

import gym
import numpy as np


class CommonEnv(gym.Env):

    def step(self, action):
        raise NotImplementedError

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        raise NotImplementedError

    def render(self, mode="human"):
        raise NotImplementedError

    @property
    def task(self) -> str:
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def task_id(self) -> int:
        raise NotImplementedError

    @property
    def num_tasks(self) -> int:
        raise NotImplementedError

    def get_statistics(self, mode: str = '') -> Dict[str, float]:
        raise NotImplementedError

    def clear_episode_statistics(self) -> None:
        raise NotImplementedError
