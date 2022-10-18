from argparse import Namespace
from collections import deque
from typing import Dict, List

import numpy as np
from scipy import spatial

from coom.env.scenario.scenario import DoomEnv
from coom.env.utils.utils import distance_traversed
from coom.env.utils.wrappers import WrapperHolder, GameVariableRewardWrapper, MovementRewardWrapper


class Chainsaw(DoomEnv):

    def __init__(self, args: Namespace, env: str, task_id: int, num_tasks=1):
        super().__init__(args, env, task_id, num_tasks)
        self.reward_scaler_traversal = args.reward_scaler_traversal
        self.reward_kill = args.reward_kill
        self.distance_buffer = []
        self.hits_taken = 0

    def store_statistics(self, game_var_buf: deque) -> None:
        distance = distance_traversed(game_var_buf, 2, 3)
        self.distance_buffer.append(distance)

        current_vars = game_var_buf[-1]
        previous_vars = game_var_buf[-2]
        if current_vars[0] < previous_vars[0]:
            self.hits_taken += 1

    def get_success(self) -> float:
        return self.game_variable_buffer[-1][1]  # Kills

    def reward_wrappers(self) -> List[WrapperHolder]:
        return [
            WrapperHolder(GameVariableRewardWrapper, self.reward_kill, 1),
            WrapperHolder(MovementRewardWrapper),
        ]

    @property
    def performance_upper_bound(self) -> float:
        return 40.0

    @property
    def performance_lower_bound(self) -> float:
        return 0.0  # No kills

    def extra_statistics(self, mode: str = '') -> Dict[str, float]:
        variables = self.game_variable_buffer[-1]
        return {f'{mode}/health': variables[0],
                f'{mode}/kills': variables[1],
                f'{mode}/movement': np.mean(self.distance_buffer).round(3),
                f'{mode}/hits_taken': self.hits_taken}

    def clear_episode_statistics(self) -> None:
        super().clear_episode_statistics()
        self.distance_buffer.clear()
        self.hits_taken = 0
