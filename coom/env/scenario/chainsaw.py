from collections import deque

import numpy as np
from argparse import Namespace
from typing import Dict, List

from coom.env.scenario.scenario import DoomEnv
from coom.env.utils.utils import distance_traversed
from coom.env.wrappers.reward import WrapperHolder, GameVariableRewardWrapper, MovementRewardWrapper


class Chainsaw(DoomEnv):
    """
    In this scenario, the agent is randomly spawned in one of 20 possible locations within a maze-like environment, and
    equipped with a chainsaw. A fixed number of enemies are spawned at random locations at the beginning of an episode.
    Additional enemies will continually be added at random unoccupied locations after each time interval. The enemies
    are rendered immobile, forcing them to remain at their fixed locations. The goal of the agent is to find the
    enemies, walk up to melee distance from them and saw them in half. The agent can move forward, turn left and right,
    and use the chainsaw. The agent is granted a reward for moving in the environment and each enemy killed.
    """

    def __init__(self, args: Namespace, env: str, task_id: int, num_tasks=1):
        super().__init__(args, env, task_id, num_tasks)
        self.reward_scaler_traversal = args.reward_scaler_traversal
        self.reward_kill = args.reward_kill_chain
        self.distance_buffer = []
        self.hits_taken = 0

    def store_statistics(self, game_var_buf: deque) -> None:
        if len(game_var_buf) < 2:
            return

        distance = distance_traversed(game_var_buf, 2, 3)
        self.distance_buffer.append(distance)

        current_vars = game_var_buf[-1]
        previous_vars = game_var_buf[-2]
        if current_vars[0] < previous_vars[0]:
            self.hits_taken += 1

    def get_success(self) -> float:
        return self.game_variable_buffer[-1][1] if self.game_variable_buffer else 0.0  # Kills

    def reward_wrappers_dense(self) -> List[WrapperHolder]:
        return [
            WrapperHolder(GameVariableRewardWrapper, reward=self.reward_kill, var_index=1),
            WrapperHolder(MovementRewardWrapper, scaler=self.reward_scaler_traversal),
        ]

    def reward_wrappers_sparse(self) -> List[WrapperHolder]:
        return [WrapperHolder(GameVariableRewardWrapper, reward=self.reward_kill, var_index=1)]

    @property
    def performance_upper_bound(self) -> float:
        return 40.0

    @property
    def performance_lower_bound(self) -> float:
        return 10.0  # Mean kills achievable by random actions

    def extra_statistics(self, mode: str = '') -> Dict[str, float]:
        if not self.game_variable_buffer:
            return {}
        variables = self.game_variable_buffer[-1]
        return {f'{mode}/health': variables[0],
                f'{mode}/kills': variables[1],
                f'{mode}/movement': np.mean(self.distance_buffer).round(3),
                f'{mode}/hits_taken': self.hits_taken}

    def clear_episode_statistics(self) -> None:
        super().clear_episode_statistics()
        self.distance_buffer.clear()
        self.hits_taken = 0
