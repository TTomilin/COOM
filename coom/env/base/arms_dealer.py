from argparse import Namespace
from collections import deque
from typing import Dict, List

import numpy as np
from vizdoom import GameVariable

from coom.env.base.scenario import DoomEnv
from coom.env.utils.utils import distance_traversed
from coom.env.utils.wrappers import WrapperHolder, ConstantRewardWrapper, MovementRewardWrapper, \
    UserVariableRewardWrapper


class ArmsDealer(DoomEnv):

    def __init__(self, args: Namespace, env: str, task_id: int, num_tasks: int = 1):
        self.penalty_frame_passed = args.penalty_frame_passed
        self.reward_delivery = args.reward_delivery
        self.distance_buffer = []
        self.reward_scaler_traversal = args.reward_scaler_traversal
        self.reward_item_acquired = args.reward_item_acquired
        super().__init__(args, env, task_id, num_tasks)

    @property
    def user_vars(self) -> List[GameVariable]:
        return [GameVariable.USER1, GameVariable.USER2]

    def store_statistics(self, game_var_buf: deque) -> None:
        distance = distance_traversed(game_var_buf, 0, 1)
        self.distance_buffer.append(distance)

    def reward_wrappers(self) -> List[WrapperHolder]:
        return [
            WrapperHolder(ConstantRewardWrapper, self.penalty_frame_passed, True),
            WrapperHolder(UserVariableRewardWrapper, self.reward_item_acquired, GameVariable.USER1),
            WrapperHolder(UserVariableRewardWrapper, self.reward_delivery, GameVariable.USER2),
            WrapperHolder(MovementRewardWrapper),
        ]

    def get_statistics(self, mode: str = '') -> Dict[str, float]:
        return {f'{mode}/weapons_acquired': self.user_variables[GameVariable.USER1],
                f'{mode}/arms_dealt': self.user_variables[GameVariable.USER2],
                f'{mode}/movement': np.mean(self.distance_buffer).round(3)}

    def clear_episode_statistics(self) -> None:
        super().clear_episode_statistics()
        self.distance_buffer.clear()
