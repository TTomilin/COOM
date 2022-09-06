from argparse import Namespace
from typing import Dict, List

import numpy as np
from scipy import spatial

from coom.doom.env.base.raise_the_roof import RaiseTheRoof


class RaiseTheRoofImpl(RaiseTheRoof):

    def __init__(self, args: Namespace, task: str, task_id: int, num_tasks=1):
        super().__init__(args, task, task_id, num_tasks, args.reward_frame_survived)
        self.distance_buffer = []
        self.switch_reward = args.switch_reward
        self.traversal_reward_scaler = args.traversal_reward_scaler
        self.add_speed = args.add_speed

    def calc_reward(self) -> float:
        reward = super().calc_reward()
        # Utilize a dense reward system by encouraging movement over previous iterations
        distance = self.distance_traversed()
        self.distance_buffer.append(distance)
        reward += distance * self.traversal_reward_scaler  # Increase reward linearly

        current_vars = self.game_variable_buffer[-1]
        previous_vars = self.game_variable_buffer[-2]

        if current_vars[0] > previous_vars[0]:  # Obtain the switch press count
            reward += self.switch_reward
            self.switches_pressed += 1

        return reward

    def distance_traversed(self) -> float:
        current_coords = [self.game_variable_buffer[-1][3],
                          self.game_variable_buffer[-1][4]]
        past_coords = [self.game_variable_buffer[0][3],
                       self.game_variable_buffer[0][4]]
        return spatial.distance.euclidean(current_coords, past_coords)

    def get_statistics(self, mode: str = '') -> Dict[str, float]:
        statistics = {f'{mode}/movement': np.mean(self.distance_buffer).round(3),
                      f'{mode}/switches_pressed': self.switches_pressed}
        return statistics

    def clear_episode_statistics(self) -> None:
        self.switches_pressed = 0
        self.distance_buffer.clear()

    def get_available_actions(self) -> List[List[float]]:
        actions = []
        m_forward = [[0.0], [1.0]]
        t_left_right = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        use = [[0.0], [1.0]]
        speed = [[0.0], [1.0]]

        for t in t_left_right:
            for m in m_forward:
                for u in use:
                    if self.add_speed:
                        for s in speed:
                            actions.append(t + m + u + s)
                    else:
                        actions.append(t + m + u)
        return actions
