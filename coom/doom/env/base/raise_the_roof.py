from argparse import Namespace
from typing import List

from coom.doom.env.base.scenario import DoomEnv


class RaiseTheRoof(DoomEnv):
    """
    In this scenario, the agent is randomly spawned in one of 20 possible locations within a maze-like environment, and
    equipped with a weapon and unlimited ammunition. A fixed number of enemies are spawned at random locations at the
    beginning of an episode. Additional enemies will continually be added at random unoccupied locations after a time
    interval. The enemies are rendered immobile, forcing them to remain at their fixed locations. The goal of the agent
    is to locate and shoot the enemies. The agent can move forward, turn left and right, and shoot. The agent is granted
    a reward for each enemy killed.
    """

    def __init__(self, args: Namespace, task: str, task_id: int, num_tasks=1, reward_frame_survived=0.01):
        self.add_speed = args.add_speed
        self.reward_frame_survived = reward_frame_survived
        super().__init__(args, task, task_id, num_tasks)

    def get_available_actions(self) -> List[List[float]]:
        actions = []
        m_forward = [[0.0], [1.0]]
        t_left_right = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        use = [[0.0], [1.0]]

        for t in t_left_right:
            for m in m_forward:
                for u in use:
                    actions.append(t + m + u)
        return actions

    def calc_reward(self) -> float:
        return self.reward_frame_survived
