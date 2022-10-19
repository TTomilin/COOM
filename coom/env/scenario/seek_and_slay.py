from argparse import Namespace
from collections import deque
from typing import List, Dict

import numpy as np

from coom.env.scenario.scenario import DoomEnv
from coom.env.utils.utils import distance_traversed
from coom.env.utils.wrappers import GameVariableRewardWrapper, MovementRewardWrapper, WrapperHolder


class SeekAndSlay(DoomEnv):
    """
    In this scenario, the agent is randomly spawned in one of 20 possible locations within a maze-like environment, and
    equipped with a weapon and unlimited ammunition. A fixed number of enemies are spawned at random locations at the
    beginning of an episode. Additional enemies will continually be added at random unoccupied locations after a time
    interval. The enemies are rendered immobile, forcing them to remain at their fixed locations. The goal of the agent
    is to locate and shoot the enemies. The agent can move forward, turn left and right, and shoot. The agent is granted
    a reward for each enemy killed.
    """

    def __init__(self, args: Namespace, env: str, task_id: int, num_tasks=1, reward_kill=1.0):
        super().__init__(args, env, task_id, num_tasks)
        self.reward_scaler_traversal = args.reward_scaler_traversal
        self.reward_kill = reward_kill
        self.distance_buffer = []
        self.hits_taken = 0
        self.ammo_used = 0

    def store_statistics(self, game_var_buf: deque) -> None:
        if len(game_var_buf) < 2:
            return

        distance = distance_traversed(game_var_buf, 3, 4)
        self.distance_buffer.append(distance)

        current_vars = game_var_buf[-1]
        previous_vars = game_var_buf[-2]
        if current_vars[0] < previous_vars[0]:
            self.hits_taken += 1
        if current_vars[2] < previous_vars[2]:
            self.ammo_used += 1

    def get_success(self) -> float:
        return self.game_variable_buffer[-1][1] if self.game_variable_buffer else 0.0  # Kills

    def reward_wrappers(self) -> List[WrapperHolder]:
        return [
            WrapperHolder(GameVariableRewardWrapper, self.reward_kill, 1),
            WrapperHolder(MovementRewardWrapper),
        ]

    @property
    def performance_upper_bound(self) -> float:
        return 50.0

    @property
    def performance_lower_bound(self) -> float:
        return 5.0  # Mean kills achievable by random actions

    def extra_statistics(self, mode: str = '') -> Dict[str, float]:
        if not self.game_variable_buffer:
            return {}
        variables = self.game_variable_buffer[-1]
        return {f'{mode}/health': variables[0],
                f'{mode}/kills': variables[1],
                f'{mode}/ammo': self.ammo_used,
                f'{mode}/movement': np.mean(self.distance_buffer).round(3),
                f'{mode}/hits_taken': self.hits_taken}

    def clear_episode_statistics(self) -> None:
        super().clear_episode_statistics()
        self.distance_buffer.clear()
        self.hits_taken = 0
        self.ammo_used = 0
