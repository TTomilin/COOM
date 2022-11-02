from argparse import Namespace
from typing import List, Dict

from coom.env.scenario.scenario import DoomEnv
from coom.env.utils.wrappers import GameVariableRewardWrapper, WrapperHolder


class DefendTheCenter(DoomEnv):
    """
    In this scenario, the agent is spawned in the center of a circular room. Enemies are spawned at fixed positions
    alongside the wall of the area. The enemies do not possess a projectile attack and therefore have to make their way
    within melee range to inflict damage. The agent is rendered immobile, but equipped with a weapon and limited
    ammunition to fend off the encroaching enemies. Once the enemies are killed, they respawn at their original location
    after a fixed time delay. The objective of the agent is to survive as long as possible. The agent is rewarded for
    each enemy killed.
    """

    def __init__(self, args: Namespace, env: str, task_id: int, num_tasks=1, reward_kill=1.0):
        super().__init__(args, env, task_id, num_tasks)
        self.penalty_health_loss = args.penalty_health_loss
        self.penalty_ammo_used = args.penalty_ammo_used
        self.reward_kill = reward_kill

    def get_available_actions(self) -> List[List[float]]:
        actions = []
        attack = [[0.0], [1.0]]
        t_left_right = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        for t in t_left_right:
            for a in attack:
                actions.append(t + a)
        return actions

    def get_success(self) -> float:
        return self.game_variable_buffer[-1][0] if self.game_variable_buffer else 0.0  # Kills

    def reward_wrappers(self) -> List[WrapperHolder]:
        return [
            WrapperHolder(GameVariableRewardWrapper, self.reward_kill, 0),
            WrapperHolder(GameVariableRewardWrapper, self.penalty_health_loss, 1, True),
            WrapperHolder(GameVariableRewardWrapper, self.penalty_ammo_used, 2, True),
        ]

    @property
    def performance_upper_bound(self) -> float:
        return 26.0  # Number of bullets in the clip, 1 kill per bullet

    @property
    def performance_lower_bound(self) -> float:
        return 0.0  # No kills

    def extra_statistics(self, mode: str = '') -> Dict[str, float]:
        if not self.game_variable_buffer:
            return {}
        variables = self.game_variable_buffer[-1]
        return {f'{mode}/kills': variables[0], f'{mode}/ammo': variables[2]}
