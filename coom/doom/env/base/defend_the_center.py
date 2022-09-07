from argparse import Namespace
from typing import List

from coom.doom.env.base.scenario import DoomEnv


class DefendTheCenter(DoomEnv):
    """
    In this scenario, the agent is spawned in the center of a circular room. Enemies are spawned at fixed positions
    alongside the wall of the area. The enemies do not possess a projectile attack and therefore have to make their way
    within melee range to inflict damage. The agent is rendered immobile, but equipped with a weapon and limited
    ammunition to fend off the encroaching enemies. Once the enemies are killed, they respawn at their original location
    after a fixed time delay. The objective of the agent is to survive as long as possible. The agent is rewarded for
    each enemy killed.
    """

    def __init__(self, args: Namespace, task: str, task_id: int, num_tasks=1, reward_kill=1.0):
        super().__init__(args, task, task_id, num_tasks)
        self.reward_kill = reward_kill

    def get_available_actions(self) -> List[List[float]]:
        actions = []
        attack = [[0.0], [1.0]]
        t_left_right = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        for t in t_left_right:
            for a in attack:
                actions.append(t + a)
        return actions

    def calc_reward(self) -> float:
        reward = 0.0
        current_vars = self.game_variable_buffer[-1]
        previous_vars = self.game_variable_buffer[-2]

        if current_vars[0] > previous_vars[0]:
            reward += self.reward_kill  # Elimination of enemy

        return reward
