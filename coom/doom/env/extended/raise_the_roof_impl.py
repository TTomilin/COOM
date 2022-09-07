from argparse import Namespace
from typing import Dict, List

import numpy as np
from scipy import spatial
from vizdoom import GameVariable

from coom.doom.env.base.raise_the_roof import RaiseTheRoof


class RaiseTheRoofImpl(RaiseTheRoof):

    def __init__(self, args: Namespace, task: str, task_id: int, num_tasks=1):
        self.distance_buffer = []
        self.switches_pressed = 0
        self.reward_switch_pressed = args.reward_switch_pressed
        self.reward_scaler_traversal = args.reward_scaler_traversal
        self.add_speed = args.add_speed
        super().__init__(args, task, task_id, num_tasks, args.reward_frame_survived)

    def calc_reward(self) -> float:
        reward = super().calc_reward()
        # Utilize a dense reward system by encouraging movement over previous iterations
        distance = self.distance_traversed()
        self.distance_buffer.append(distance)
        reward += distance * self.reward_scaler_traversal  # Increase reward linearly

        switches_pressed = self.game.get_game_variable(GameVariable.USER2)
        if switches_pressed > self.switches_pressed:
            reward += self.reward_switch_pressed
            self.switches_pressed += 1

        return reward

    def distance_traversed(self) -> float:
        current_coords = [self.game_variable_buffer[-1][0],
                          self.game_variable_buffer[-1][1]]
        past_coords = [self.game_variable_buffer[0][0],
                       self.game_variable_buffer[0][1]]
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
