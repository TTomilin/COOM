from argparse import Namespace
from typing import List

from coom.doom.env.base.scenario import DoomEnv


class DefendTheCenter(DoomEnv):
    """
    Scenario in which the agent stands in the center of a circular room
    with the goal of maximizing its time of survival. The agent is given
    a pistol with a limited amount of ammo to shoot approaching enemies
    who will invoke damage to the agent as they reach melee distance.
    """

    def __init__(self, args: Namespace, task: str, task_id: int, num_tasks=1, kill_reward=1.0):
        super().__init__(args, task, task_id, num_tasks)
        self.kill_reward = kill_reward

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
            reward += self.kill_reward  # Elimination of enemy

        return reward
