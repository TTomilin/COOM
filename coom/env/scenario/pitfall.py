from collections import deque

from argparse import Namespace
from typing import Dict, List
from vizdoom import DEAD

from coom.env.scenario.scenario import DoomEnv
from coom.env.utils.wrappers import WrapperHolder, ProportionalVariableRewardWrapper, BooleanVariableRewardWrapper


class Pitfall(DoomEnv):

    def __init__(self, args: Namespace, env: str, task_id: int, num_tasks: int = 1):
        super().__init__(args, env, task_id, num_tasks)
        self.reward_scaler = args.reward_scaler_pitfall
        self.penalty_death = args.penalty_death
        self.frames = 0
        self.total_dist = 0
        self.current_dist = 0
        self.distance_buffer = []

    def store_statistics(self, game_var_buf: deque) -> None:
        self.frames += 1
        self.current_dist = game_var_buf[-1][0]
        self.total_dist += self.current_dist
        self.distance_buffer.append(self.current_dist)

    def get_success(self) -> float:
        return self.total_dist

    def reward_wrappers(self) -> List[WrapperHolder]:
        return [WrapperHolder(ProportionalVariableRewardWrapper, self.reward_scaler, 0, True),
                WrapperHolder(BooleanVariableRewardWrapper, self.penalty_death, DEAD)]

    @property
    def performance_upper_bound(self) -> float:
        return 100000

    @property
    def performance_lower_bound(self) -> float:
        return 20000

    def extra_statistics(self, mode: str = '') -> Dict[str, float]:
        return {f'{mode}/distance': self.get_success(), f'{mode}/movement': self.get_success() / self.frames}

    def clear_episode_statistics(self) -> None:
        super().clear_episode_statistics()
        self.distance_buffer.clear()
        self.total_dist = 0
