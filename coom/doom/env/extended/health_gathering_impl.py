from argparse import Namespace
from typing import Dict

from coom.doom.env.base.health_gathering import HealthGathering


class HealthGatheringImpl(HealthGathering):

    def __init__(self, args: Namespace, task: str, task_id: int, num_tasks=1):
        super().__init__(args, task, task_id, num_tasks)
        self.health_acquired_reward = args.health_acquired_reward
        self.kits_obtained = 0

    def calc_reward(self) -> float:
        reward = super().calc_reward()
        current_vars = self.game_variable_buffer[-1]
        previous_vars = self.game_variable_buffer[-2]
        if current_vars[0] > previous_vars[0]:
            reward += self.health_acquired_reward  # Picked up health kit
            self.kits_obtained += 1
        return reward

    def get_statistics(self) -> Dict[str, float]:
        return {'kits_obtained': self.kits_obtained}

    def clear_episode_statistics(self) -> None:
        self.kits_obtained = 0
