from argparse import Namespace
from typing import Dict

import numpy as np
from scipy import spatial

from coom.doom.env.base.seek_and_slay import SeekAndSlay


class SeekAndSlayImpl(SeekAndSlay):

    def __init__(self, args: Namespace, task: str, task_id: int, num_tasks=1):
        super().__init__(args, task, task_id, num_tasks, args.kill_reward)
        self.distance_buffer = []
        self.ammo_used = 0
        self.hits_taken = 0
        self.traversal_reward_scaler = args.traversal_reward_scaler
        self.health_loss_penalty = args.health_loss_penalty
        self.ammo_used_penalty = args.ammo_used_penalty

    @property
    def n_spawn_points(self) -> int:
        return 11

    def calc_reward(self) -> float:
        reward = super().calc_reward()
        # Utilize a dense reward system by encouraging movement over previous iterations
        distance = self.distance_traversed()
        self.distance_buffer.append(distance)
        reward += distance * self.traversal_reward_scaler  # Increase reward linearly

        current_vars = self.game_variable_buffer[-1]
        previous_vars = self.game_variable_buffer[-2]

        if current_vars[0] < previous_vars[0]:
            self.hits_taken += 1
        if current_vars[2] < previous_vars[2]:
            self.ammo_used += 1

        return reward

    def distance_traversed(self) -> float:
        current_coords = [self.game_variable_buffer[-1][3],
                          self.game_variable_buffer[-1][4]]
        past_coords = [self.game_variable_buffer[0][3],
                       self.game_variable_buffer[0][4]]
        return spatial.distance.euclidean(current_coords, past_coords)

    def get_statistics(self) -> Dict[str, float]:
        variables = self.game_variable_buffer[-1]
        statistics = {'health': variables[0],
                      'kills': variables[1],
                      'ammo': self.ammo_used,
                      'movement': np.mean(self.distance_buffer),
                      'hits_taken': self.hits_taken}
        return statistics

    def clear_episode_statistics(self) -> None:
        self.ammo_used = 0
        self.hits_taken = 0
        self.distance_buffer.clear()
