from argparse import Namespace
from typing import Dict

from coom.doom.env.base.dodge_projectiles import DodgeProjectiles


class DodgeProjectilesImpl(DodgeProjectiles):

    def __init__(self, args: Namespace, task: str, task_id: int, num_tasks=1):
        self.penalty_health_loss = args.penalty_health_loss
        self.hits_taken = 0
        super().__init__(args, task, task_id, num_tasks, args.reward_frame_survived)

    def calc_reward(self) -> float:
        reward = super().calc_reward()
        current_vars = self.game_variable_buffer[-1]
        previous_vars = self.game_variable_buffer[-2]
        if current_vars[0] < previous_vars[0]:
            reward -= self.penalty_health_loss  # Loss of health
            self.hits_taken += 1
        return reward

    def get_statistics(self, mode: str = '') -> Dict[str, float]:
        return {f'{mode}/hits_taken': self.hits_taken}

    def clear_episode_statistics(self) -> None:
        self.hits_taken = 0
