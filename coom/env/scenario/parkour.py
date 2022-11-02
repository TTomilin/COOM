from argparse import Namespace
from collections import deque
from typing import Dict, List

import numpy as np
from vizdoom import GameVariable

from coom.env.scenario.scenario import DoomEnv
from coom.env.utils.utils import distance_traversed
from coom.env.utils.wrappers import WrapperHolder, ConstantRewardWrapper, MovementRewardWrapper, \
    UserVariableRewardWrapper, GameVariableRewardWrapper


class Parkour(DoomEnv):

    def __init__(self, args: Namespace, env: str, task_id: int, num_tasks: int = 1):
        super().__init__(args, env, task_id, num_tasks)
        self.reward_scaler_height = args.reward_scaler_height
        self.frames = 0
        self.total_height = 0
        self.current_height = 0
        self.distance_buffer = []

    def store_statistics(self, game_var_buf: deque) -> None:
        self.frames += 1
        self.current_height = game_var_buf[-1][2]
        self.total_height += self.current_height
        if len(game_var_buf) < 2:
            return
        distance = distance_traversed(game_var_buf, 0, 1)
        self.distance_buffer.append(distance)

    def get_success(self) -> float:
        return self.total_height / self.frames

    def reward_wrappers(self) -> List[WrapperHolder]:
        return [WrapperHolder(GameVariableRewardWrapper, self.reward_scaler_height, 2)]

    @property
    def performance_upper_bound(self) -> float:
        return 200.0  # TODO Figure this out

    @property
    def performance_lower_bound(self) -> float:
        return 0.0  # No height gained

    def extra_statistics(self, mode: str = '') -> Dict[str, float]:
        return {f'{mode}/height': self.get_success(),
                f'{mode}/movement': np.mean(self.distance_buffer).round(3)}

    def clear_episode_statistics(self) -> None:
        super().clear_episode_statistics()
        self.distance_buffer.clear()
