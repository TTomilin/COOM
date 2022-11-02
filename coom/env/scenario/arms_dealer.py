from argparse import Namespace
from collections import deque
from typing import Dict, List

import numpy as np
from vizdoom import GameVariable

from coom.env.scenario.scenario import DoomEnv
from coom.env.utils.utils import distance_traversed
from coom.env.utils.wrappers import WrapperHolder, ConstantRewardWrapper, MovementRewardWrapper, \
    UserVariableRewardWrapper


class ArmsDealer(DoomEnv):
    """
    In this scenario, the agent has to pick up weapons from the ground in a square shaped room, and deliver them to a
    platform which appears at a random location after a weapon is obtained. The agent can carry multiple (the exact
    amount is dependent on the weapon) weapons before making the delivery. A new weapon is spawned at a random location
    whenever one is picked up, i.e., there is always a constant number of weapons on the ground. A new platform is
    created for every weapon picked up. After a successful delivery, all the platforms disappear and the agent has to
    start over by collecting weapons. The agent can turn left and right, move forward, and accelerate. The agent is
    rewarded for both obtaining a weapon and delivering it. Additionally, a small reward is granted for how much the
    agent moves in the environment. Finally, the agent is penalized for being idle.
    """

    def __init__(self, args: Namespace, env: str, task_id: int, num_tasks: int = 1):
        super().__init__(args, env, task_id, num_tasks)
        self.reward_scaler_traversal = args.reward_scaler_traversal
        self.penalty_passivity = args.penalty_passivity
        self.reward_item_acquired = args.reward_item_acquired
        self.reward_delivery = args.reward_delivery
        self.distance_buffer = []

    @property
    def user_vars(self) -> List[GameVariable]:
        return [GameVariable.USER1, GameVariable.USER2]

    def store_statistics(self, game_var_buf: deque) -> None:
        if len(game_var_buf) < 2:
            return
        distance = distance_traversed(game_var_buf, 0, 1)
        self.distance_buffer.append(distance)

    def get_success(self) -> float:
        return self.user_variables[GameVariable.USER2]  # Arms dealt

    def reward_wrappers(self) -> List[WrapperHolder]:
        return [
            WrapperHolder(ConstantRewardWrapper, self.penalty_passivity),
            WrapperHolder(UserVariableRewardWrapper, self.reward_item_acquired, GameVariable.USER1),
            WrapperHolder(UserVariableRewardWrapper, self.reward_delivery, GameVariable.USER2),
            WrapperHolder(MovementRewardWrapper),
        ]

    @property
    def performance_upper_bound(self) -> float:
        return 10.0  # Sufficient arms dealt

    @property
    def performance_lower_bound(self) -> float:
        return 0.0  # No arms dealt

    def extra_statistics(self, mode: str = '') -> Dict[str, float]:
        return {f'{mode}/weapons_acquired': self.user_variables[GameVariable.USER1],
                f'{mode}/arms_dealt': self.user_variables[GameVariable.USER2],
                f'{mode}/movement': np.mean(self.distance_buffer).round(3)}

    def clear_episode_statistics(self) -> None:
        super().clear_episode_statistics()
        self.distance_buffer.clear()
