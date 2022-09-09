from argparse import Namespace
from typing import Dict

import numpy as np
from scipy import spatial

from coom.doom.env.base.hide_and_seek import HideAndSeek


class HideAndSeekImpl(HideAndSeek):

    def __init__(self, args: Namespace, task: str, task_id: int, num_tasks=1):
        self.distance_buffer = []
        self.hits_taken = 0
        self.reward_scaler_traversal = args.reward_scaler_traversal
        self.penalty_health_loss = args.penalty_health_loss
        self.add_speed = args.add_speed
        super().__init__(args, task, task_id, num_tasks)

    def calc_reward(self) -> float:
        reward = super().calc_reward()
        # Utilize a dense reward system by encouraging movement over previous iterations
        distance = self.distance_traversed()
        self.distance_buffer.append(distance)
        reward += distance * self.reward_scaler_traversal  # Increase reward linearly

        current_vars = self.game_variable_buffer[-1]
        previous_vars = self.game_variable_buffer[-2]

        if current_vars[0] < previous_vars[0]:
            self.hits_taken += 1
            reward -= self.penalty_health_loss  # Loss of health

        return reward

    def distance_traversed(self) -> float:
        current_coords = [self.game_variable_buffer[-1][3],
                          self.game_variable_buffer[-1][4]]
        past_coords = [self.game_variable_buffer[0][3],
                       self.game_variable_buffer[0][4]]
        return spatial.distance.euclidean(current_coords, past_coords)

    def get_statistics(self, mode: str = '') -> Dict[str, float]:
        variables = self.game_variable_buffer[-1]
        statistics = {f'{mode}/health': variables[0],
                      f'{mode}/movement': np.mean(self.distance_buffer).round(3),
                      f'{mode}/hits_taken': self.hits_taken}
        return statistics

    def clear_episode_statistics(self) -> None:
        self.hits_taken = 0
        self.distance_buffer.clear()

    def get_available_actions(self):
        actions = []
        m_forward_backward = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        t_left_right = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        speed = [[0.0], [1.0]]

        for t in t_left_right:
            for m in m_forward_backward:
                if self.add_speed and m == [1.0]:
                    for s in speed:
                        actions.append(t + m + s)
                else:
                    actions.append(t + m)
        return actions
