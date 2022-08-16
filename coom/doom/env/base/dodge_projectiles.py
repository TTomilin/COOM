from argparse import Namespace
from typing import List

from coom.doom.env.base.scenario import DoomEnv


class DodgeProjectiles(DoomEnv):

    def __init__(self, args: Namespace, task: str, task_id: int, num_tasks=1, frame_survived_reward=0.01):
        super().__init__(args, task, task_id, num_tasks)
        self.reward_frame_survived = frame_survived_reward

    def get_available_actions(self) -> List[List[float]]:
        actions = []
        speed = [[0.0], [1.0]]
        m_left_right = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        for m in m_left_right:
            for s in speed:
                actions.append(m + s)
        return actions

    def calc_reward(self) -> float:
        return self.reward_frame_survived

