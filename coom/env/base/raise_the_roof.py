from argparse import Namespace
from collections import deque
from typing import List, Dict

import numpy as np
from vizdoom import GameVariable

from coom.env.base.scenario import DoomEnv
from coom.env.utils.utils import distance_traversed
from coom.env.utils.wrappers import MovementRewardWrapper, WrapperHolder, ConstantRewardWrapper, \
    UserVariableRewardWrapper


class RaiseTheRoof(DoomEnv):

    def __init__(self, args: Namespace, env: str, task_id: int, num_tasks=1):
        self.distance_buffer = []
        self.reward_frame_survived = args.reward_frame_survived
        self.reward_switch_pressed = args.reward_switch_pressed
        self.reward_scaler_traversal = args.reward_scaler_traversal
        super().__init__(args, env, task_id, num_tasks)

    @property
    def user_vars(self) -> List[GameVariable]:
        return [GameVariable.USER2]

    def store_statistics(self, game_var_buf: deque) -> None:
        distance = distance_traversed(game_var_buf, 0, 1)
        self.distance_buffer.append(distance)

    def reward_wrappers(self) -> List[WrapperHolder]:
        return [
            WrapperHolder(ConstantRewardWrapper, self.reward_frame_survived),
            WrapperHolder(UserVariableRewardWrapper, self.reward_switch_pressed, GameVariable.USER2),
            WrapperHolder(MovementRewardWrapper),
        ]

    def get_statistics(self, mode: str = '') -> Dict[str, float]:
        return {f'{mode}/movement': np.mean(self.distance_buffer).round(3),
                f'{mode}/switches_pressed': self.user_variables[GameVariable.USER2]}

    def clear_episode_statistics(self) -> None:
        super().clear_episode_statistics()
        self.distance_buffer.clear()
