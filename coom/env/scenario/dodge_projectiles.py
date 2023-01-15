from collections import deque

from argparse import Namespace
from typing import List, Dict

from coom.env.scenario.scenario import DoomEnv
from coom.env.utils.wrappers import WrapperHolder, ConstantRewardWrapper, GameVariableRewardWrapper


class DodgeProjectiles(DoomEnv):
    """
    In this scenario, the agent is positioned in one end of a rectangular room, facing the opposite wall. Immobile
    enemies, equipped with projectile attacks, are lined up in front of the opposing wall, equal distance from one
    another. The objective is to survive as long as possible, ultimately until the termination of the episode. The agent
    is given no weapon nor ammunition and can only move laterally to dodge enemy projectiles. The agent is rewarded for
    each frame that it survives.
    """

    def __init__(self, args: Namespace, env: str, task_id: int, num_tasks=1, reward_frame_survived=0.01):
        super().__init__(args, env, task_id, num_tasks)
        self.penalty_projectile = args.penalty_projectile
        self.reward_frame_survived = reward_frame_survived
        self.frames_survived = 0
        self.hits_taken = 0

    def get_available_actions(self) -> List[List[float]]:
        actions = []
        speed = [[0.0], [1.0]]
        m_left_right = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        for m in m_left_right:
            for s in speed:
                actions.append(m + s)
        return actions

    def store_statistics(self, game_var_buf: deque) -> None:
        self.frames_survived += 1
        if len(game_var_buf) < 2:
            return
        current_vars = game_var_buf[-1]
        previous_vars = game_var_buf[-2]
        if current_vars[0] < previous_vars[0]:
            self.hits_taken += 1

    def get_success(self) -> float:
        return self.frames_survived * self.frame_skip

    def reward_wrappers(self) -> List[WrapperHolder]:
        return [
            WrapperHolder(ConstantRewardWrapper, self.reward_frame_survived),
            WrapperHolder(GameVariableRewardWrapper, self.penalty_projectile, 0, True),
        ]

    @property
    def performance_upper_bound(self) -> float:
        return 2500.0  # Scenario length

    @property
    def performance_lower_bound(self) -> float:
        return 300.0  # Frames until the projectiles slaughter the player  # TODO Verify this value

    def extra_statistics(self, mode: str = '') -> Dict[str, float]:
        return {f'{mode}/hits_taken': self.hits_taken}

    def clear_episode_statistics(self) -> None:
        super().clear_episode_statistics()
        self.frames_survived = 0
        self.hits_taken = 0
