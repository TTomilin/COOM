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
        m_forward_backward = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        t_left_right = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        execute = [[0.0], [1.0]]
        speed = [[0.0], [1.0]]

        for t in t_left_right:
            for m in m_forward_backward:
                for e in execute:
                    if self.add_speed and m == [1.0]:
                        for s in speed:
                            actions.append(t + m + e + s)
                    else:
                        actions.append(t + m + e)
        return actions
