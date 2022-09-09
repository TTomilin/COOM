from argparse import Namespace
from typing import Dict

import numpy as np
from scipy import spatial

from coom.doom.env.extended.seek_and_slay_impl import SeekAndSlayImpl


class ChainsawImpl(SeekAndSlayImpl):

    def __init__(self, args: Namespace, task: str, task_id: int, num_tasks=1):
        super().__init__(args, task, task_id, num_tasks)

    def distance_traversed(self) -> float:
        current_coords = [self.game_variable_buffer[-1][2],
                          self.game_variable_buffer[-1][3]]
        past_coords = [self.game_variable_buffer[0][2],
                       self.game_variable_buffer[0][3]]
        return spatial.distance.euclidean(current_coords, past_coords)

    def get_statistics(self, mode: str = '') -> Dict[str, float]:
        variables = self.game_variable_buffer[-1]
        statistics = {f'{mode}/health': variables[0],
                      f'{mode}/kills': variables[1],
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
        attack = [[0.0], [1.0]]
        use = [[0.0], [1.0]]
        speed = [[0.0], [1.0]]

        for t in t_left_right:
            for m in m_forward_backward:
                for a in attack:
                    for u in use:
                        if self.add_speed and m == [1.0]:
                            for s in speed:
                                actions.append(t + m + a + u + s)
                        else:
                            actions.append(t + m + a + u)
        return actions
