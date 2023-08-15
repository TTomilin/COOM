import gym
import numpy as np
from typing import Dict, Any, Tuple, Optional


class CommonEnv(gym.Env):

    def step(self, action):
        raise NotImplementedError

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None, ) -> Tuple[
        np.ndarray, Dict[str, Any]]:
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

    @property
    def action_space(self) -> gym.spaces.Discrete:
        raise NotImplementedError

    @property
    def observation_space(self) -> gym.Space:
        raise NotImplementedError

    def get_statistics(self, mode: str = '') -> Dict[str, float]:
        raise NotImplementedError

    def clear_episode_statistics(self) -> None:
        raise NotImplementedError

    def get_active_env(self):
        raise NotImplementedError
