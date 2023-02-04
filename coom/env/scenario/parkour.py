from collections import deque

import numpy as np
from argparse import Namespace
from typing import Dict, List

from coom.env.scenario.scenario import DoomEnv
from coom.env.utils.utils import distance_traversed
from coom.env.wrappers.reward import WrapperHolder, LocationVariableRewardWrapper


class Parkour(DoomEnv):

    def __init__(self, args: Namespace, env: str, task_id: int, num_tasks: int = 1):
        super().__init__(args, env, task_id, num_tasks)
        self.reward_scaler_traversal = args.reward_scaler_traversal
        self.frames = 0
        self.current_height = 0
        self.distance_buffer = []

    def store_statistics(self, game_var_buf: deque) -> None:
        self.frames += 1
        self.current_height = game_var_buf[-1][2]
        if len(game_var_buf) > 1:
            distance = distance_traversed(game_var_buf, 0, 1)
            self.distance_buffer.append(distance)

    def get_success(self) -> float:
        return self.current_height

    def reward_wrappers_dense(self) -> List[WrapperHolder]:
        return [WrapperHolder(LocationVariableRewardWrapper,  x_index=0, y_index=1, x_start=608, y_start=608)]

    @property
    def performance_upper_bound(self) -> float:
        return 500

    @property
    def performance_lower_bound(self) -> float:
        return 0  # No height gained

    def extra_statistics(self, mode: str = '') -> Dict[str, float]:
        return {f'{mode}/height': self.get_success(),
                f'{mode}/movement': np.mean(self.distance_buffer).round(3)}

    def clear_episode_statistics(self) -> None:
        super().clear_episode_statistics()
        self.distance_buffer.clear()
        self.current_height = 0
