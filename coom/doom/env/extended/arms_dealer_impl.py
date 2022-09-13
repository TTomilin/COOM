from argparse import Namespace
from typing import Dict

import numpy as np
from scipy import spatial
from vizdoom import GameVariable

from coom.doom.env.base.arms_dealer import ArmsDealer


class ArmsDealerImpl(ArmsDealer):

    def __init__(self, args: Namespace, task: str, task_id: int, num_tasks=1):
        self.distance_buffer = []
        self.reward_scaler_traversal = args.reward_scaler_traversal
        self.reward_item_acquired = args.reward_item_acquired
        self.penalty_frame_passed = args.penalty_frame_passed
        self.add_speed = args.add_speed
        self.weapons_acquired = 0
        super().__init__(args, task, task_id, num_tasks, args.reward_delivery)

    def calc_reward(self) -> float:
        reward = super().calc_reward()
        # Utilize a dense reward system by encouraging movement over previous iterations
        distance = self.distance_traversed()
        self.distance_buffer.append(distance)
        reward += distance * self.reward_scaler_traversal  # Increase reward linearly for movement

        weapons_acquired = self.game.get_game_variable(GameVariable.USER2)
        if weapons_acquired > self.weapons_acquired:
            reward = self.reward_item_acquired
            self.weapons_acquired = weapons_acquired
        return reward

    def distance_traversed(self) -> float:
        current_coords = [self.game_variable_buffer[-1][1],
                          self.game_variable_buffer[-1][2]]
        past_coords = [self.game_variable_buffer[0][1],
                       self.game_variable_buffer[0][2]]
        return spatial.distance.euclidean(current_coords, past_coords)

    def get_statistics(self, mode: str = '') -> Dict[str, float]:
        return {
            f'{mode}/arms_dealt': self.arms_dealt,
            f'{mode}/weapons_acquired': self.weapons_acquired,
            f'{mode}/movement': np.mean(self.distance_buffer).round(3)
        }

    def clear_episode_statistics(self) -> None:
        self.weapons_acquired = 0
        self.arms_dealt = 0
        self.distance_buffer.clear()

    def get_available_actions(self):
        actions = []
        speed = [[0.0], [1.0]]
        m_forward = [[0.0], [1.0]]
        t_left_right = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        for m in m_forward:
            for t in t_left_right:
                if self.add_speed and m == [1.0]:
                    for s in speed:
                        actions.append(t + m + s)
                else:
                    actions.append(t + m)
        return actions
