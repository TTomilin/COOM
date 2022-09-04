from abc import ABC
from typing import Dict

import gym


class CommonEnv(gym.Env, ABC):

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
