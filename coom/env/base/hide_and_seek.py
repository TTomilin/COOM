from argparse import Namespace
from collections import deque
from typing import List, Dict

import numpy as np

from coom.env.base.scenario import DoomEnv
from coom.env.utils.utils import distance_traversed
from coom.env.utils.wrappers import MovementRewardWrapper, WrapperHolder, GameVariableRewardWrapper, \
    ConstantRewardWrapper


class HideAndSeek(DoomEnv):

    def __init__(self, args: Namespace, env: str, task_id: int, num_tasks=1):
        self.distance_buffer = []
        self.kits_obtained = 0
        self.hits_taken = 0
        self.reward_frame_survived = args.reward_frame_survived
        self.reward_scaler_traversal = args.reward_scaler_traversal
        self.reward_item_acquired = args.reward_item_acquired
        self.penalty_health_loss = args.penalty_health_loss
        super().__init__(args, env, task_id, num_tasks)

    def store_statistics(self, game_var_buf: deque) -> None:
        distance = distance_traversed(game_var_buf, 1, 2)
        self.distance_buffer.append(distance)

        current_vars = game_var_buf[-1]
        previous_vars = game_var_buf[-2]
        if current_vars[0] < previous_vars[0]:
            self.hits_taken += 1
        elif current_vars[0] > previous_vars[0]:
            self.kits_obtained += 1

    def reward_wrappers(self) -> List[WrapperHolder]:
        return [
            WrapperHolder(ConstantRewardWrapper, self.reward_frame_survived),
            WrapperHolder(GameVariableRewardWrapper, self.reward_item_acquired, 0),
            WrapperHolder(GameVariableRewardWrapper, self.penalty_health_loss, 0, True, True),
            WrapperHolder(MovementRewardWrapper),
        ]

    def get_statistics(self, mode: str = '') -> Dict[str, float]:
        variables = self.game_variable_buffer[-1]
        return {f'{mode}/health': variables[0],
                f'{mode}/movement': np.mean(self.distance_buffer).round(3),
                f'{mode}/hits_taken': self.hits_taken,
                f'{mode}/kits_obtained': self.kits_obtained}

    def clear_episode_statistics(self) -> None:
        self.hits_taken = 0
        self.kits_obtained = 0
        self.distance_buffer.clear()
