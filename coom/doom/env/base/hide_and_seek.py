from argparse import Namespace
from typing import List

from coom.doom.env.base.scenario import DoomEnv


class HideAndSeek(DoomEnv):

    def __init__(self, args: Namespace, task: str, task_id: int, num_tasks=1, reward_frame_survived=0.01):
        self.reward_frame_survived = reward_frame_survived
        super().__init__(args, task, task_id, num_tasks)

    def get_available_actions(self) -> List[List[float]]:
        actions = []
        m_forward_backward = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        t_left_right = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        for t in t_left_right:
            for m in m_forward_backward:
                actions.append(t + m)
        return actions

    def calc_reward(self) -> float:
        return self.reward_frame_survived
