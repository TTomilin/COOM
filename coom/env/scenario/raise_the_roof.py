from argparse import Namespace
from collections import deque
from typing import List, Dict

import numpy as np
from vizdoom import GameVariable

from coom.env.scenario.scenario import DoomEnv
from coom.env.utils.utils import distance_traversed
from coom.env.utils.wrappers import MovementRewardWrapper, WrapperHolder, ConstantRewardWrapper, \
    UserVariableRewardWrapper


class RaiseTheRoof(DoomEnv):

    def __init__(self, args: Namespace, env: str, task_id: int, num_tasks=1):
        super().__init__(args, env, task_id, num_tasks)
        self.reward_scaler_traversal = args.reward_scaler_traversal
        self.reward_frame_survived = args.reward_frame_survived
        self.reward_switch_pressed = args.reward_switch_pressed
        self.distance_buffer = []
        self.frames_survived = 0

    @property
    def user_vars(self) -> List[GameVariable]:
        return [GameVariable.USER2]

    def store_statistics(self, game_var_buf: deque) -> None:
        self.frames_survived += 1

        distance = distance_traversed(game_var_buf, 0, 1)
        self.distance_buffer.append(distance)

    def get_success(self) -> float:
        return self.frames_survived * self.frame_skip

    def reward_wrappers(self) -> List[WrapperHolder]:
        return [
            WrapperHolder(ConstantRewardWrapper, self.reward_frame_survived),
            WrapperHolder(UserVariableRewardWrapper, self.reward_switch_pressed, GameVariable.USER2),
            WrapperHolder(MovementRewardWrapper),
        ]

    @property
    def performance_upper_bound(self) -> float:
        return 5000.0  # Scenario length

    @property
    def performance_lower_bound(self) -> float:
        return 650.0  # Frames until getting crushed

    def extra_statistics(self, mode: str = '') -> Dict[str, float]:
        return {f'{mode}/movement': np.mean(self.distance_buffer).round(3),
                f'{mode}/switches_pressed': self.user_variables[GameVariable.USER2]}

    def clear_episode_statistics(self) -> None:
        super().clear_episode_statistics()
        self.distance_buffer.clear()
        self.frames_survived = 0
