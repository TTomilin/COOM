from argparse import Namespace
from typing import Dict

from coom.doom.env.base.floor_is_lava import FloorIsLava
from coom.doom.env.base.health_gathering import HealthGathering


class FloorIsLavaImpl(FloorIsLava):

    def __init__(self, args: Namespace, task: str, task_id: int, num_tasks=1):
        self.penalty_health_loss = args.penalty_health_loss
        self.add_speed = args.add_speed
        super().__init__(args, task, task_id, num_tasks, args.reward_frame_survived)

    def calc_reward(self) -> float:
        reward = super().calc_reward()
        current_vars = self.game_variable_buffer[-1]
        previous_vars = self.game_variable_buffer[-2]
        if current_vars[0] < previous_vars[0]:
            reward -= self.penalty_health_loss  # Stood on lava
        return reward

    def get_available_actions(self):
        actions = []
        speed = [[0.0], [1.0]]
        m_forward = [[0.0], [1.0]]
        t_left_right = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        for m in m_forward:
            for t in t_left_right:
                if self.add_speed and m == [1.0]:
                    for s in speed:
                        actions.append(t + m + s)
                else:
                    actions.append(t + m)
        return actions