from argparse import Namespace
from typing import Dict

from coom.doom.env.base.defend_the_center import DefendTheCenter


class DefendTheCenterImpl(DefendTheCenter):

    def __init__(self, args: Namespace, task: str, task_id: int, num_tasks=1):
        self.penalty_health_loss = args.penalty_health_loss
        self.penalty_ammo_used = args.penalty_ammo_used
        super().__init__(args, task, task_id, num_tasks, args.reward_kill)

    def calc_reward(self) -> float:
        current_vars = self.game_variable_buffer[-1]
        previous_vars = self.game_variable_buffer[-2]

        reward = super().calc_reward()

        if current_vars[1] < previous_vars[1]:
            reward -= self.penalty_health_loss  # Loss of health
        if current_vars[2] < previous_vars[2]:
            reward -= self.penalty_ammo_used  # Use of ammunition

        return reward

    def get_statistics(self, mode: str = '') -> Dict[str, float]:
        variables = self.game_variable_buffer[-1]
        return {f'{mode}/kills': variables[0], f'{mode}/ammo': variables[2]}
