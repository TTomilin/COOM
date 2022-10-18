from argparse import Namespace
from collections import deque
from typing import List, Dict

import numpy as np

from coom.env.scenario.scenario import DoomEnv
from coom.env.utils.utils import distance_traversed
from coom.env.utils.wrappers import MovementRewardWrapper, WrapperHolder, GameVariableRewardWrapper, \
    ConstantRewardWrapper


class HideAndSeek(DoomEnv):

    def __init__(self, args: Namespace, env: str, task_id: int, num_tasks=1):
        super().__init__(args, env, task_id, num_tasks)
        self.reward_scaler_traversal = args.reward_scaler_traversal
        self.reward_frame_survived = args.reward_frame_survived
        self.reward_item_acquired = args.reward_item_acquired
        self.penalty_health_loss = args.penalty_health_loss
        self.distance_buffer = []
        self.frames_survived = 0
        self.kits_obtained = 0
        self.hits_taken = 0

    def store_statistics(self, game_var_buf: deque) -> None:
        self.frames_survived += 1

        distance = distance_traversed(game_var_buf, 1, 2)
        self.distance_buffer.append(distance)

        current_vars = game_var_buf[-1]
        previous_vars = game_var_buf[-2]
        if current_vars[0] < previous_vars[0]:
            self.hits_taken += 1
        elif current_vars[0] > previous_vars[0]:
            self.kits_obtained += 1

    def get_success(self) -> float:
        return self.frames_survived * self.frame_skip

    def reward_wrappers(self) -> List[WrapperHolder]:
        return [
            WrapperHolder(ConstantRewardWrapper, self.reward_frame_survived),
            WrapperHolder(GameVariableRewardWrapper, self.reward_item_acquired, 0),
            WrapperHolder(GameVariableRewardWrapper, self.penalty_health_loss, 0, True, True),
            WrapperHolder(MovementRewardWrapper),
        ]

    @property
    def performance_upper_bound(self) -> float:
        return 2500.0  # Scenario length

    @property
    def performance_lower_bound(self) -> float:
        return 500.0  # Frames until getting slaughtered by the monsters  # TODO Verify this value

    def extra_statistics(self, mode: str = '') -> Dict[str, float]:
        variables = self.game_variable_buffer[-1]
        return {f'{mode}/health': variables[0],
                f'{mode}/movement': np.mean(self.distance_buffer).round(3),
                f'{mode}/hits_taken': self.hits_taken,
                f'{mode}/kits_obtained': self.kits_obtained}

    def clear_episode_statistics(self) -> None:
        super().clear_episode_statistics()
        self.distance_buffer.clear()
        self.frames_survived = 0
        self.kits_obtained = 0
        self.hits_taken = 0
