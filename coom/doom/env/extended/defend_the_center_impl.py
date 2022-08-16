from argparse import Namespace
from typing import Dict

from coom.doom.env.base.defend_the_center import DefendTheCenter


class DefendTheCenterImpl(DefendTheCenter):

    def __init__(self, args: Namespace, task: str, task_id: int, num_tasks=1):
        super().__init__(args, task, task_id, num_tasks, args.kill_reward)
        self.health_loss_penalty = args.health_loss_penalty
        self.ammo_used_penalty = args.ammo_used_penalty

    def calc_reward(self) -> float:
        current_vars = self.game_variable_buffer[-1]
        previous_vars = self.game_variable_buffer[-2]

        reward = super().calc_reward()

        if current_vars[1] < previous_vars[1]:
            reward -= self.health_loss_penalty  # Loss of health
        if current_vars[2] < previous_vars[2]:
            reward -= self.ammo_used_penalty  # Use of ammunition

        return reward

    def get_statistics(self) -> Dict[str, float]:
        variables = self.game_variable_buffer[-1]
        return {'kills': variables[0], 'ammo': variables[2]}
