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
