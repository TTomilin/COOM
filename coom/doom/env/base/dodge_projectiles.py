from argparse import Namespace
from typing import List

from coom.doom.env.base.scenario import DoomEnv


class DodgeProjectiles(DoomEnv):
    """
    In this scenario, the agent is positioned in one end of a rectangular room, facing the opposite wall. Immobile
    enemies, equipped with projectile attacks, are lined up in front of the opposing wall, equal distance from one
    another. The objective is to survive as long as possible, ultimately until the termination of the episode. The agent
    is given no weapon nor ammunition and can only move laterally to dodge enemy projectiles. The agent is rewarded for
    each frame that it survives.
    """

    def __init__(self, args: Namespace, task: str, task_id: int, num_tasks=1, reward_frame_survived=0.01):
        super().__init__(args, task, task_id, num_tasks)
        self.reward_frame_survived = reward_frame_survived

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

